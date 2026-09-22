import Foundation

/// Owns a VLC instance and reads its playback clock.
///
/// Why this exists: the engine can enhance far ahead of real time, so its own
/// position is the *production frontier*, not what the viewer is watching. Any
/// control expressed against that position (relative seek, audio/scale/subtitle
/// restart, pause) would jump far beyond the current picture. VLC exposes its
/// real position through its HTTP interface, so the engine drives an instance it
/// launched itself and uses that clock as the source of truth.
///
/// The trade-off: this is an instance we own, launched directly from the VLC
/// binary, so it is separate from a VLC the user may already have open (we cannot
/// reach the HTTP interface of an instance we did not launch).
final class VLCController {
    struct Clock {
        var seconds: Double = 0        // raw VLC `time` (carries an unknown base)
        var elapsed: Double = 0        // seconds played since this session opened
        var length: Double = 0
        var state: String = "stopped"   // playing | paused | stopped
        var valid = false
    }

    static let defaultHTTPPort: UInt16 = 8099

    let httpPort: UInt16
    private let password: String
    private let binary: String
    private let lock = NSLock()
    private var process: Process?
    private var clockValue = Clock()
    /// First valid `time` seen for this generation. VLC's absolute time carries a
    /// base we do not control — on a session starting at 600 s it reported 638
    /// before even a second had played — but the *elapsed* time advances at real
    /// time, so playback position is derived from the delta.
    private var baseTime: Double?
    private var pollThread: Thread?
    private var stopFlag = false
    /// Guards the boundary between sessions: a stale poll must not import the
    /// clock of the previous stream.
    private var generation = 0

    /// Latest known clock (thread-safe snapshot).
    var clock: Clock {
        lock.lock(); defer { lock.unlock() }
        return clockValue
    }

    init?(binary: String = "/Applications/VLC.app/Contents/MacOS/VLC",
          httpPort: UInt16 = VLCController.defaultHTTPPort) {
        guard FileManager.default.isExecutableFile(atPath: binary) else { return nil }
        self.binary = binary
        // Pick a free port. If an earlier instance is still alive (e.g. orphaned
        // when an engine was killed), a second VLC would fail to bind its HTTP
        // interface and every clock read would then be answered by the stale
        // instance — silently reporting another stream's state.
        self.httpPort = VLCController.freePort(startingAt: httpPort)
        self.password = UUID().uuidString
    }

    /// First free TCP port at or after `start` on loopback.
    static func freePort(startingAt start: UInt16) -> UInt16 {
        for offset in 0..<20 {
            let port = start &+ UInt16(offset)
            let fd = socket(AF_INET, SOCK_STREAM, 0)
            if fd < 0 { return start }
            var addr = sockaddr_in()
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = port.bigEndian
            addr.sin_addr.s_addr = inet_addr("127.0.0.1")
            let bound = withUnsafePointer(to: &addr) { p in
                p.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            close(fd)
            if bound == 0 { return port }
        }
        return start
    }

    // MARK: - Lifecycle

    /// Starts an instance playing `url`, replacing any previous one.
    func play(_ url: URL) {
        if let p = process, p.isRunning {
            // Reuse the running instance: swap the input instead of spawning a
            // second window.
            send("in_play", extra: ["input": url.absoluteString])
            bumpGeneration()
            return
        }
        stopFlag = false
        let p = Process()
        p.executableURL = URL(fileURLWithPath: binary)
        // ZU_VLC_HEADLESS=1 runs VLC with dummy outputs so tests exercise the
        // real clock and command path without opening a window.
        let headless = ProcessInfo.processInfo.environment["ZU_VLC_HEADLESS"] == "1"
        var args = headless
            ? ["-I", "dummy", "--vout=dummy", "--aout=dummy"]
            : ["-I", "macosx", "--no-video-title-show"]
        args += [
            "--extraintf=http",
            "--http-password=\(password)",
            "--http-port=\(httpPort)",
            "--network-caching=1500",
            url.absoluteString,
        ]
        p.arguments = args
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        p.standardInput = FileHandle.nullDevice   // don't let VLC inherit our terminal
        do {
            try p.run()
        } catch {
            return
        }
        process = p
        bumpGeneration()
        startPolling()
    }

    func pause() { send("pl_forcepause") }
    func resume() { send("pl_forceresume") }

    /// Absolute seek on VLC's own timeline (seconds).
    func seek(to seconds: Double) {
        let v = Int(max(0, seconds))
        send("seek", extra: ["val": String(v)])
    }

    var isRunning: Bool {
        guard let p = process else { return false }
        return p.isRunning
    }

    func stop() {
        stopFlag = true
        pollThread = nil
        guard let p = process else { return }
        process = nil
        if p.isRunning { p.terminate() }
        let deadline = Date().addingTimeInterval(1.0)
        while p.isRunning && Date() < deadline { usleep(20_000) }
        if p.isRunning { kill(p.processIdentifier, SIGKILL) }
    }

    deinit { stop() }

    private func bumpGeneration() {
        lock.lock()
        generation += 1
        clockValue = Clock()
        baseTime = nil
        lock.unlock()
    }

    // MARK: - Clock polling

    private func startPolling() {
        guard pollThread == nil else { return }
        let t = Thread { [weak self] in
            while let self, !self.stopFlag {
                // Bare thread, no run loop: drain autoreleased objects each pass.
                autoreleasepool {
                    let gen = self.currentGeneration
                    if let c = self.fetchClock(), gen == self.currentGeneration {
                        self.lock.lock()
                        if self.baseTime == nil, c.valid { self.baseTime = c.seconds }
                        var stored = c
                        stored.elapsed = max(0, c.seconds - (self.baseTime ?? c.seconds))
                        self.clockValue = stored
                        self.lock.unlock()
                    }
                }
                usleep(500_000)
            }
        }
        t.name = "vlc-clock"
        t.qualityOfService = .utility
        pollThread = t
        t.start()
    }

    private var currentGeneration: Int {
        lock.lock(); defer { lock.unlock() }
        return generation
    }

    private func fetchClock() -> Clock? {
        guard let data = httpGet("/requests/status.json") else { return nil }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        var c = Clock()
        c.state = (obj["state"] as? String) ?? "unknown"
        if let t = obj["time"] as? Double { c.seconds = t } else if let t = obj["time"] as? Int { c.seconds = Double(t) }
        if let l = obj["length"] as? Double { c.length = l } else if let l = obj["length"] as? Int { c.length = Double(l) }
        c.valid = (c.state == "playing" || c.state == "paused")
        return c
    }

    // MARK: - HTTP

    private func send(_ command: String, extra: [String: String] = [:]) {
        var path = "/requests/status.json?command=\(command)"
        for (k, v) in extra {
            let esc = v.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? v
            path += "&\(k)=\(esc)"
        }
        _ = httpGet(path)
    }

    private func httpGet(_ path: String) -> Data? {
        guard let url = URL(string: "http://127.0.0.1:\(httpPort)\(path)") else { return nil }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        let cred = Data(":\(password)".utf8).base64EncodedString()
        req.setValue("Basic \(cred)", forHTTPHeaderField: "Authorization")
        let sem = DispatchSemaphore(value: 0)
        var result: Data?
        let task = URLSession.shared.dataTask(with: req) { data, _, _ in
            result = data
            sem.signal()
        }
        task.resume()
        if sem.wait(timeout: .now() + 2.5) == .timedOut {
            task.cancel()
            return nil
        }
        return result
    }
}
