import Foundation

/// Local HTTP server serving the enhanced MPEG-TS stream, implemented with
/// blocking POSIX sockets.
///
///   GET /stream.ts?session=N   -> live TS bytes (single consumer per session)
///   GET /status.json           -> StatusSnapshot JSON
///   GET /                      -> tiny info page
///
/// Why raw sockets instead of Network.framework: NWConnection copies every
/// chunk into its own internal queue, which showed up as slow steady-state RSS
/// growth (~0.6 MB/s). A blocking `send(2)` applies TCP backpressure directly
/// and keeps process memory flat — the bounded ring is then the only place
/// stream bytes live.
final class StreamServer {
    private let ring: RingBuffer
    private let status: StatusStore
    private(set) var port: UInt16

    private var listenFD: Int32 = -1
    private var acceptThread: Thread?
    private let clientLock = NSLock()
    private var clientFD: Int32 = -1

    init(ring: RingBuffer, status: StatusStore, port: UInt16 = 8090) {
        self.ring = ring
        self.status = status
        self.port = port
    }

    // MARK: - Lifecycle

    func start() throws {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw EngineError.serverFailure("socket() failed") }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                bind(fd, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, listen(fd, 8) == 0 else {
            let err = String(cString: strerror(errno))
            close(fd)
            throw EngineError.serverFailure("bind/listen on 127.0.0.1:\(port) failed: \(err)")
        }
        listenFD = fd

        let t = Thread { [weak self] in self?.acceptLoop() }
        t.name = "stream-accept"
        t.qualityOfService = .userInitiated
        acceptThread = t
        t.start()
    }

    func stop() {
        let fd = listenFD
        listenFD = -1
        if fd >= 0 { close(fd) }
        dropClients()
        acceptThread = nil
    }

    /// Drops the current consumer. Only shuts the socket down; the serving
    /// thread owns the descriptor and closes it exactly once. Closing here as
    /// well risked a double close, and once the fd number is reused that would
    /// close an unrelated socket.
    func dropClients() {
        clientLock.lock()
        let fd = clientFD
        clientFD = -1
        clientLock.unlock()
        if fd >= 0 { shutdown(fd, SHUT_RDWR) }
        status.update { $0.clients = 0 }
    }

    // MARK: - Accept loop

    private func acceptLoop() {
        while listenFD >= 0 {
            var addr = sockaddr_in()
            var len = socklen_t(MemoryLayout<sockaddr_in>.size)
            let cfd = withUnsafeMutablePointer(to: &addr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    accept(listenFD, sa, &len)
                }
            }
            if cfd < 0 {
                if errno == EBADF || listenFD < 0 { break }
                continue
            }
            var yes: Int32 = 1
            setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
            let t = Thread { [weak self] in self?.handleClient(cfd) }
            t.name = "stream-client"
            t.start()
        }
    }

    // MARK: - Request handling

    private func handleClient(_ fd: Int32) {
        var request = Data()
        var buf = [UInt8](repeating: 0, count: 4096)
        while request.count <= 32 * 1024 {
            let n = recv(fd, &buf, buf.count, 0)
            if n <= 0 { close(fd); return }
            request.append(contentsOf: buf[0..<n])
            if request.range(of: Data("\r\n\r\n".utf8)) != nil { break }
        }
        let head = String(data: request, encoding: .utf8) ?? ""
        let firstLine = head.split(separator: "\r\n").first.map(String.init) ?? ""
        let parts = firstLine.split(separator: " ")
        let path = parts.count > 1 ? String(parts[1]) : "/"

        if path.hasPrefix("/status.json") {
            let body = status.json
            sendAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n")
            sendAll(fd, body)
            close(fd)
            return
        }
        if path.hasPrefix("/stream.ts") {
            serveStream(fd, path: path)
            return
        }
        let body = Data("MacVideoUpscaler: open /stream.ts?session=<n> in VLC, or /status.json\n".utf8)
        sendAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n")
        sendAll(fd, body)
        close(fd)
    }

    private func serveStream(_ fd: Int32, path: String) {
        var session = -1
        if let q = path.firstIndex(of: "?") {
            for kv in path[path.index(after: q)...].split(separator: "&") {
                let p = kv.split(separator: "=")
                if p.count == 2, p[0] == "session", let v = Int(p[1]) { session = v }
            }
        }
        let current = status.current.sessionId
        guard session == current else {
            let body = Data("stale session (current: \(current)) — reopen the stream from the app\n".utf8)
            sendAll(fd, "HTTP/1.1 410 Gone\r\nContent-Type: text/plain\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n")
            sendAll(fd, body)
            close(fd)
            return
        }

        clientLock.lock()
        if clientFD >= 0 {
            clientLock.unlock()
            let body = Data("a stream client is already connected to this session\n".utf8)
            sendAll(fd, "HTTP/1.1 409 Conflict\r\nContent-Type: text/plain\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n")
            sendAll(fd, body)
            close(fd)
            return
        }
        clientFD = fd
        clientLock.unlock()
        status.update { $0.clients = 1 }

        sendAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: video/mp2t\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n")

        let chunk = 256 * 1024
        let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: chunk)
        defer { buf.deallocate() }

        // Prebuffer gate: hold the first TS byte until a short buffer is ready.
        // Also release as soon as the producer is finished: for a stream shorter
        // than the threshold there is nothing left to wait for, and waiting for
        // `isEOS` would never fire because unread bytes keep the ring non-empty
        // (that is how short clips served headers and zero video bytes).
        while ring.availableNow < ring.prebufferBytes && !ring.producerDone {
            if !isClientCurrent(fd) { break }
            usleep(20_000)
        }

        var alive = true
        while alive {
            // A seek/session change resets the ring (unblocking the read below),
            // but also stop explicitly if we were superseded mid-stream.
            if status.current.sessionId != session { break }
            let n = ring.read(into: buf, maxLength: chunk)
            if n == 0 { break }  // EOS or session reset
            alive = sendAll(fd, buf, n)
        }

        clientLock.lock()
        let stillMine = clientFD == fd
        if stillMine { clientFD = -1 }
        clientLock.unlock()
        if stillMine { status.update { $0.clients = 0 } }
        close(fd)   // sole owner: see dropClients()
    }

    private func isClientCurrent(_ fd: Int32) -> Bool {
        clientLock.lock(); defer { clientLock.unlock() }
        return clientFD == fd
    }

    @discardableResult
    private func sendAll(_ fd: Int32, _ text: String) -> Bool {
        sendAll(fd, Data(text.utf8))
    }

    @discardableResult
    private func sendAll(_ fd: Int32, _ data: Data) -> Bool {
        data.withUnsafeBytes { raw -> Bool in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return true }
            return sendAll(fd, base, raw.count)
        }
    }

    /// Blocking send of the whole buffer. Returns false on error/disconnect.
    @discardableResult
    private func sendAll(_ fd: Int32, _ ptr: UnsafePointer<UInt8>, _ length: Int) -> Bool {
        var off = 0
        while off < length {
            let n = send(fd, ptr + off, length - off, 0)  // blocks: TCP backpressure
            if n > 0 {
                off += n
            } else if n < 0 && errno == EINTR {
                continue
            } else {
                return false
            }
        }
        return true
    }
}
