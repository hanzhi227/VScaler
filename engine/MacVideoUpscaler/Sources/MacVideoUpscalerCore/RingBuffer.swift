import Foundation

/// Blocking bounded byte FIFO shared between the processing pipeline (producer)
/// and the HTTP stream server (single consumer). When full, `write` blocks —
/// this is the backpressure that keeps memory bounded. When empty and not EOS,
/// `read` blocks — this is what makes VLC wait on the live frontier.
final class RingBuffer {
    private var buf: UnsafeMutablePointer<UInt8>
    private let capacity: Int
    private var start = 0      // read offset
    private var count = 0      // buffered bytes
    private var eosFlag = false
    private var closed = false
    /// Optional soft limit in bytes (below `capacity`), used to keep the
    /// read-ahead bounded in *time* rather than only in bytes.
    private var softCap: Int?
    private let cond = NSCondition()

    let prebufferBytes: Int

    init(capacity: Int, prebufferBytes: Int) {
        self.capacity = max(capacity, 1 << 20)
        // Clamp to capacity: a prebuffer larger than the ring could never be
        // satisfied and would deadlock the stream gate.
        self.prebufferBytes = min(max(prebufferBytes, 64 << 10), self.capacity)
        self.buf = UnsafeMutablePointer<UInt8>.allocate(capacity: self.capacity)
    }

    deinit { buf.deallocate() }

    var bufferedBytes: Int {
        cond.lock(); defer { cond.unlock() }
        return count
    }

    /// True once the producer has finished, regardless of unread bytes.
    ///
    /// Distinct from `isEOS` (producer finished AND ring drained) because a
    /// stream shorter than the prebuffer threshold can leave bytes unread: a
    /// consumer waiting for `isEOS` would then wait forever, which is how short
    /// clips produced HTTP headers and zero video bytes.
    var producerDone: Bool {
        cond.lock(); defer { cond.unlock() }
        return eosFlag
    }

    var isEOS: Bool {
        cond.lock(); defer { cond.unlock() }
        return eosFlag && count == 0
    }

    /// Blocks while full. Returns false if the buffer was closed/reset while waiting.
    @discardableResult
    func write(_ bytes: UnsafePointer<UInt8>, length: Int) -> Bool {
        cond.lock()
        defer { cond.unlock() }
        var off = 0
        while off < length {
            let limit = min(capacity, softCap ?? capacity)
            while count >= limit && !closed {
                cond.wait()
            }
            if closed { return false }
            let head = (start + count) % capacity
            let n = min(length - off, limit - count, capacity - head)
            memcpy(buf + head, bytes + off, n)
            count += n
            off += n
            cond.broadcast()
        }
        return true
    }

    /// Caps the buffered read-ahead. Used to hold the producer's lead over the
    /// player to a bounded number of seconds so controls act near what is being
    /// watched, instead of minutes ahead. nil removes the cap.
    func setSoftCap(bytes: Int?) {
        cond.lock()
        softCap = bytes.map { max($0, 256 << 10) }
        cond.broadcast()
        cond.unlock()
    }

    /// Blocks while empty. Returns bytes read; 0 means EOS (and drained).
    func read(into dst: UnsafeMutablePointer<UInt8>, maxLength: Int) -> Int {
        cond.lock()
        defer { cond.unlock() }
        while count == 0 && !eosFlag && !closed {
            cond.wait()
        }
        if count == 0 { return 0 }  // eos or closed
        let head = start
        let n = min(maxLength, count, capacity - head)
        memcpy(dst, buf + head, n)
        start = (head + n) % capacity
        count -= n
        cond.broadcast()
        return n
    }

    /// Bytes currently available without blocking (for prebuffer gating).
    var availableNow: Int {
        cond.lock(); defer { cond.unlock() }
        return count
    }

    func markEOS() {
        cond.lock()
        eosFlag = true
        cond.broadcast()
        cond.unlock()
    }

    /// Reset for a new session (seek). Readers in flight get 0 and must reconnect.
    func reset() {
        cond.lock()
        closed = true
        start = 0
        count = 0
        eosFlag = false
        cond.broadcast()
        cond.unlock()
    }

    /// Reopen after a reset for a fresh session.
    func reopen() {
        cond.lock()
        closed = false
        start = 0
        count = 0
        eosFlag = false
        cond.broadcast()
        cond.unlock()
    }
}
