import Foundation
import CoreVideo
import Metal

/// Output frames that have been rendered but not yet shown.
///
/// This exists because `AVSampleBufferDisplayLayer` **cannot** be the buffer. It
/// is a display pipeline with a two-frame deep queue: measured with a standalone
/// probe (`scripts/probe/layer_queue.swift`), it accepted 2 frames and then
/// refused, at 30 fps and at any presentation timestamp — frames stamped a minute
/// into the future were refused exactly as readily as frames already due. Feeding
/// it faster than real time is therefore impossible, and the read-ahead the player
/// reports has always been its own 3-frame `CVPixelBufferPool` threshold rather
/// than anything the layer held.
///
/// So the read-ahead lives here: the producer renders into this store as fast as
/// the model allows, and the feeder hands frames to the layer just in time. That
/// is what lets a mode slower than real time play smoothly after a wait instead of
/// stuttering — the deficit is paid once, up front, and then in bursts.
///
/// Ring-addressed by frame index, like `SourceRing`: frame `i` always occupies
/// slot `i % capacity`, and `oldest...newest` is the window that is actually held.
/// Indices are global and monotonic within a session, so a seek rebuilds the store
/// rather than rewinding it.
///
/// Bounded in bytes rather than frames, because the same capacity is ~8 MB a frame
/// at 1080p and ~26 MB at 4K. `capacity` is whichever of the configured seconds
/// and the configured byte budget is smaller.
final class FrameStore {

    /// One rendered frame on its way to the display layer.
    ///
    /// The pixel buffer and its Metal texture are held together because the
    /// texture is a view onto that buffer's IOSurface; the frame is only usable
    /// while both are alive, and the layer takes over the buffer on enqueue.
    struct Frame {
        let index: Int
        let pb: CVPixelBuffer
        let cvTexture: CVMetalTexture
        let texture: MTLTexture
    }

    private let cond = NSCondition()
    private var slots: [Frame?]
    private var oldest = 0
    private var newest = -1
    private var closed = false
    /// Set once the producer has run off the end of the source: no frame after
    /// `newest` will ever arrive. Unlike `closed`, the held frames stay readable,
    /// so a consumer waiting for a fill level this store can no longer reach is
    /// released to drain what is here instead of waiting forever.
    private var producerFinished = false

    /// Frames this store can hold, i.e. the maximum read-ahead in frames.
    let capacity: Int
    /// Bytes one frame occupies, for the status and for the byte budget.
    let frameBytes: Int

    init(capacity: Int, frameBytes: Int) {
        self.capacity = max(2, capacity)
        self.frameBytes = frameBytes
        self.slots = Array(repeating: nil, count: max(2, capacity))
    }

    // MARK: - State

    var count: Int {
        cond.lock(); defer { cond.unlock() }
        return heldLocked
    }

    var bytes: Int { count * frameBytes }

    /// Whether the store has been shut down (its session ended).
    var isClosed: Bool {
        cond.lock(); defer { cond.unlock() }
        return closed
    }

    /// The newest index pushed, or -1 before anything is.
    var newestIndex: Int {
        cond.lock(); defer { cond.unlock() }
        return newest
    }

    /// Indices currently held, for diagnostics.
    var heldRange: ClosedRange<Int>? {
        cond.lock(); defer { cond.unlock() }
        return heldLocked > 0 ? oldest...newest : nil
    }

    private var heldLocked: Int {
        guard newest >= oldest else { return 0 }
        return min(newest - oldest + 1, capacity)
    }

    // MARK: - Producer side

    /// Appends the next frame in sequence. Fails if the store is full — the
    /// caller should wait on `waitForRoom` instead of dropping the frame, since a
    /// dropped frame is a hole the feeder would stall on forever.
    @discardableResult
    func push(_ frame: Frame) -> Bool {
        cond.lock(); defer { cond.broadcast(); cond.unlock() }
        guard !closed, frame.index == newest + 1, heldLocked < capacity else { return false }
        slots[frame.index % capacity] = frame
        newest = frame.index
        if newest - oldest + 1 > capacity { oldest = newest - capacity + 1 }
        return true
    }

    /// Blocks until there is room for another frame, or the store closes.
    /// Returns false when the caller should give up (closed).
    func waitForRoom() -> Bool {
        cond.lock(); defer { cond.unlock() }
        while !closed && heldLocked >= capacity { cond.wait() }
        return !closed
    }

    // MARK: - Consumer side

    /// The frame at `index`, if it is still held. Indices below `oldest` are gone
    /// — they were handed on already — and are refused rather than confused with
    /// a newer frame that has wrapped into the same slot.
    func frame(at index: Int) -> Frame? {
        cond.lock(); defer { cond.unlock() }
        guard index >= oldest, index <= newest, heldLocked > 0 else { return nil }
        guard let f = slots[index % capacity], f.index == index else { return nil }
        return f
    }

    /// Blocks until `index` is held or the store closes. Nil when closed without
    /// the frame arriving.
    func waitForFrame(at index: Int) -> Frame? {
        cond.lock(); defer { cond.unlock() }
        while !closed {
            if index >= oldest, index <= newest, heldLocked > 0,
               let f = slots[index % capacity], f.index == index {
                return f
            }
            cond.wait()
        }
        return nil
    }

    /// Records that the producer has reached the end of the source and wakes
    /// anyone waiting for a count that can now never be reached. Unlike `close`
    /// the held frames remain readable, so the consumer can play them out.
    func finishProducing() {
        cond.lock(); defer { cond.broadcast(); cond.unlock() }
        producerFinished = true
    }

    /// Blocks until at least `target` frames are held, the store closes, or the
    /// producer finishes short of the target. Returns whether the target was
    /// met: a false with the store still open means the producer has run off the
    /// end of the source, so the caller should drain what is held rather than
    /// wait for a refill that will never come.
    func waitForCount(_ target: Int) -> Bool {
        cond.lock(); defer { cond.unlock() }
        while !closed && !producerFinished && heldLocked < target { cond.wait() }
        return heldLocked >= target
    }

    /// Marks `index` as handed to the layer. Everything up to it is dropped: the
    /// layer owns those buffers now, and holding a reference here would keep the
    /// pool from reusing them.
    func release(through index: Int) {
        cond.lock(); defer { cond.broadcast(); cond.unlock() }
        guard index >= oldest else { return }
        for i in oldest...min(index, newest) where i >= oldest { slots[i % capacity] = nil }
        oldest = min(index + 1, newest + 1)
    }

    /// Drops everything held without changing the sequence: the next push still
    /// has to be `newest + 1`. Used when the paused redraw has already re-rendered
    /// and re-queued the frames that were in here, so the copies in the store are
    /// stale (they were rendered with the settings that were just changed).
    func discardAll() {
        cond.lock(); defer { cond.broadcast(); cond.unlock() }
        for i in oldest...max(oldest, newest) where i <= newest { slots[i % capacity] = nil }
        oldest = newest + 1
    }

    /// Shuts the store down and wakes everyone waiting on it.
    func close() {
        cond.lock(); defer { cond.broadcast(); cond.unlock() }
        closed = true
        for i in 0..<slots.count { slots[i] = nil }
        oldest = newest + 1
    }
}
