import Foundation
import AVFoundation
import AppKit
import CoreVideo
import CoreMedia
import Metal
import os

/// The last few source frames of a session, newest last.
///
/// Once a frame has been handed to the renderer the player no longer reads it,
/// so re-rendering one — which is what lets the comparison respond while paused
/// — needs its *source* kept. The display layer holds up to `bufferFrames`
/// frames at once, so the frame on screen is at most that many behind the newest
/// one produced; the ring is one larger, and it is only built once the
/// comparison has been used, so ordinary playback pays nothing for it.
///
/// Slots are addressed by frame index, so the ring is a sliding window rather
/// than a queue: index `i` always lives in slot `i % capacity`, and the window
/// `oldest...newest` is what may be read back. Owned by the video thread.
final class SourceRing {
    private let device: MTLDevice
    private let width: Int
    private let height: Int
    let capacity: Int
    private var slots: [MTLTexture?]
    /// The frame indices currently held.
    private(set) var oldest = 0
    private(set) var newest = -1

    init(device: MTLDevice, width: Int, height: Int, capacity: Int) {
        self.device = device
        self.width = width
        self.height = height
        self.capacity = max(2, capacity)
        self.slots = Array(repeating: nil, count: max(2, capacity))
    }

    var isEmpty: Bool { newest < oldest }

    /// The texture frame `index` is uploaded into, which is also that frame's
    /// source for the scalers. Frames arrive in increasing index order.
    func slot(for index: Int) -> MTLTexture? {
        let i = index % capacity
        if slots[i] == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            d.usage = [.shaderRead]
            d.storageMode = .shared        // written from the decoder's staging buffer
            slots[i] = device.makeTexture(descriptor: d)
        }
        if index > newest { newest = index }
        oldest = max(oldest, newest - capacity + 1)
        return slots[i]
    }

    /// The source of frame `index`, or nil when it is no longer (or not yet)
    /// held. Slots below `oldest` are stale — they hold a frame one lap older —
    /// so they are refused rather than returned.
    func texture(at index: Int) -> MTLTexture? {
        guard index >= oldest, index <= newest else { return nil }
        return slots[index % capacity]
    }
}

/// In-app playback controller: decode (ffmpeg) -> Metal upscale -> display layer,
/// with audio on a shared render clock.
///
/// This is the "own player" counterpart to `PipelineController`. The difference
/// that matters to the viewer:
///   * frames go straight to `AVSampleBufferDisplayLayer` — no H.264 encode, no
///     MPEG-TS, no HTTP server, no external VLC to launch or own;
///   * the read-ahead is a `FrameStore` of rendered frames, sized in seconds and
///     capped in bytes, which the display layer is fed from just in time. The
///     layer holds only 2 frames (measured, `FrameStore`), so the store is what
///     makes a mode slower than real time watchable: it runs ahead, and playback
///     waits once and then plays, rather than stuttering every frame;
///   * the timeline is the render synchronizer's clock, so the reported position
///     *is* the playback position rather than an estimate minus a lead.
///
/// `PipelineController` (and the CLI / validation scripts built on it) is left
/// untouched: it is still the way to serve a stream to an external player.
public final class PlayerController {
    public struct Config {
        public var audioEnabled = true
        public var audioStreamIndex = 0      // 0-based index into the file's audio streams
        /// Output volume, 0...1. Zero still decodes and feeds audio, so it is a
        /// mute for listening rather than a way to skip the audio path.
        public var volume: Float = 1
        public var scaleFactor: UpscaleFactor = .x2
        public var burnSubtitles = false
        public var subtitleStreamIndex = 0
        /// How much *rendered* video to keep ahead of the clock, in seconds.
        ///
        /// This is the read-ahead the player can actually use. The producer renders
        /// into a `FrameStore` as fast as the model allows and the display layer is
        /// fed from it just in time, so a mode slower than real time plays smoothly
        /// and pays for the shortfall in one wait rather than in per-frame stutter —
        /// and when the store runs dry the clock is held and a spinner shown until
        /// it refills. `FrameStore` explains why the layer itself cannot be the
        /// buffer.
        public var bufferSeconds: Double = 5
        /// Ceiling on that buffer in bytes. The same 5 s is ~1.2 GB at 1080p and
        /// ~3.9 GB at 3416x1920, so whichever limit is smaller decides the
        /// capacity. `status.bufferFrames` reports what was actually taken.
        public var bufferBudgetBytes: Int = 1_500_000_000
        /// Floor on the capacity, so an enormous frame size cannot shrink the
        /// buffer below what the display layer's own queue needs.
        public var minBufferFrames = 4
        public init() {}
    }

    /// Live ceiling for 4x. The old limit was the H.264 encoder's session size;
    /// this player has no encoder, so the limit is now throughput and memory.
    /// 4x is two chained 2x passes, so the second pass runs at 4x the output
    /// pixels: measured, 640x480 -> 2560x1920 sustains real time at ~210 MB, but
    /// 1080p -> 7680x4320 runs at **0.61x** (constant stutter) with an ~880 MB
    /// in-flight buffer. So 4x stays where it can actually keep up.
    static func maxDisplayFactor(width: Int, height: Int) -> UpscaleFactor {
        (width * 4 <= 4096 && height * 4 <= 2304) ? .x4 : .x2
    }

    public var config: Config
    public let status = StatusStore()

    /// Called with each upscaled frame (the output pixel buffer and its session
    /// frame index), synchronously on the video thread, just before it is handed
    /// to the renderer. Diagnostic hook for frame dumps; leave nil in normal use.
    public var onOutputFrame: ((CVPixelBuffer, Int) -> Void)?

    private let ctx: MetalContext
    private let player: EmbeddedPlayer
    /// The comparison compositor, built once because the kernel depends only on
    /// the device, not on the frame geometry.
    private let comparison: SplitCompositor?
    /// Whether the two halves are being shown. It lives on the controller rather
    /// than in a `Session`, so it survives seeks and rebuilds.
    private let compareFlag = OSAllocatedUnfairLock(initialState: false)
    /// Which model fills the left of the divider. `.bypass` — the plain bilinear
    /// upscale — is what a normal player would show, i.e. the original video.
    private let compareLeft = OSAllocatedUnfairLock<ScalerKind>(initialState: .bypass)
    /// Where the divider sits, 0 (right model fills the picture) to 1 (left model
    /// does). Maintained by the control thread; the video thread reads it per frame.
    private let compareSplit = OSAllocatedUnfairLock<Double>(initialState: 0.5)

    private let lock = NSCondition()
    private var stopFlag = false
    private var restartFlag = false
    /// Whether the producers should be held back (the video loop parks on it).
    /// A rebuild clears it so the new session can prime and present its first
    /// frame, which is why it is not the same thing as the viewer's intent.
    private var pausedFlag = false
    /// The viewer's intent, which survives a rebuild. Without this, pausing while
    /// a seek is still rebuilding was silently undone when that rebuild started
    /// the clock — the click simply did nothing.
    private var pauseRequested = false
    private var sessionId = 0
    private var startAt: Double = 0
    private var session: Session?
    /// How long a burst of skips waits for more presses before rebuilding.
    static let seekDebounce: TimeInterval = 0.5
    /// Running total of the skips waiting on the debounce, for the on-screen
    /// "+30 s". Guarded by `lock`.
    private var pendingSkipDelta: Double?
    /// Bumped by every seek/restart request: a debounced skip only runs if it is
    /// still the newest when its delay expires.
    private var seekGeneration = 0
    /// The position a queued restart will land on, if a request is still in flight.
    /// A rebuild takes most of a second, and it rewinds the render clock on the way,
    /// so a second seek has to accumulate against what was *asked for* rather than
    /// the clock still on screen — otherwise pressing "forward 10" twice moves 10 s,
    /// or even moves backwards. Guarded by `lock`.
    private var pendingSeek: Double?

    /// Serializes the session-lifecycle operations (restart/stop). They release
    /// `lock` while tearing the old session down, so without this a second
    /// open/seek arriving during that window could build a session the first one
    /// then tears down, orphaning its ffmpeg children.
    private let controlMutex = NSLock()

    /// Where a session is torn down and rebuilt. A restart reaps two ffmpeg
    /// children, respawns them and runs ffprobe — about a second of work — so it
    /// must not run on the main thread (which stalled the UI on every seek, and
    /// let `Process.waitUntilExit()`'s nested run loop deliver the *next* seek
    /// into the middle of this one). One serial queue both keeps the work off the
    /// main thread and orders overlapping requests.
    private let sessionQueue = DispatchQueue(label: "macvideoupscaler.session", qos: .userInitiated)
    private static let sessionQueueKey = DispatchSpecificKey<Void>()

    public private(set) var currentFile: URL?
    public private(set) var kind: ScalerKind = ScalerKind.defaultKind
    public private(set) var effectiveFactor: UpscaleFactor = .x2
    public private(set) var durationSeconds: Double = 0

    private var probedFile: URL?
    private var audioTracks: [AudioTrack] = []
    private var subtitleTracks: [SubtitleTrack] = []

    public init(config: Config = Config()) throws {
        signal(SIGPIPE, SIG_IGN)
        self.config = config
        let context = try MetalContext()
        self.ctx = context
        // A missing kernel is not fatal: the rest of the player works, and the
        // comparison simply reports itself as unavailable.
        self.comparison = try? SplitCompositor(ctx: context)
        self.player = EmbeddedPlayer(audioEnabled: config.audioEnabled, volume: config.volume)
        sessionQueue.setSpecific(key: Self.sessionQueueKey, value: ())
        status.update { $0.state = .idle }
        // Dragging the divider on the picture changes the same split the slider
        // does; both end up in `setCompareSplit`.
        player.splitFractionProvider = { [weak self] in
            guard let self, self.compareFlag.value, self.comparison != nil else { return nil }
            return self.compareSplit.value
        }
        player.onSplitDrag = { [weak self] fraction in
            self?.setCompareSplit(fraction)
        }
    }

    /// The position of the comparison divider in the surface view's coordinates,
    /// or nil when there is no comparison. For harnesses that need to aim a
    /// synthetic drag at the line.
    public var dividerXInView: CGFloat? { player.dividerXInView }

    public func makeView() -> NSView { player.makeView() }

    /// The frame the renderer is showing, as a copy, or nil when nothing has been
    /// presented yet. Diagnostic: the paused redraw's real effect is a change to
    /// what is on screen, which is not the same as what was rendered last, so a
    /// harness needs to read the presented image to check it.
    public var displayedFrame: CVPixelBuffer? { player.displayedFrame }

    /// Invoked when the viewer clicks the picture, on the main thread. The app
    /// wires this to play/pause; the engine does not decide what a click means.
    /// Set this before the view goes on screen.
    public var onSurfaceClick: (() -> Void)? {
        get { player.onSurfaceClick }
        set { player.onSurfaceClick = newValue }
    }

    /// Invoked when the viewer double clicks the picture, on the main thread.
    /// The app wires this to full screen.
    public var onSurfaceDoubleClick: (() -> Void)? {
        get { player.onSurfaceDoubleClick }
        set { player.onSurfaceDoubleClick = newValue }
    }

    // MARK: - Session

    /// Everything a playback position owns. Rebuilt from scratch on open/seek, so
    /// a seek never has to reason about in-flight state from the old position.
    private final class Session {
        let id: Int
        let startAt: Double
        let fps: Double
        let outputWidth: Int
        let outputHeight: Int
        let decoder: FFmpegDecoder
        let audio: FFmpegAudioDecoder?
        let textRenderer: TextSubtitleRenderer?
        let srcTexture: MTLTexture
        let staging: MTLBuffer
        let pool: CVPixelBufferPool
        let aux: CFDictionary
        let textureCache: CVMetalTextureCache
        var scaler: FrameScaler
        /// The scaler actually in use. Only the video thread writes it, after a
        /// successful rebuild.
        let kind: OSAllocatedUnfairLock<ScalerKind>
        /// A mode change requested by the control thread. It stays set until the
        /// video thread has *successfully* built that scaler, so a failed build
        /// cannot silently leave `kind` claiming a mode the picture is not using.
        let requestedKind = OSAllocatedUnfairLock<ScalerKind?>(initialState: nil)
        let factor: UpscaleFactor
        /// Session note (scale clamp / anamorphic) to restore after a transient
        /// message such as a failed mode switch.
        let baseMessage: String
        /// How many recent source frames the paused redraw may keep (video
        /// thread only; see `SourceRing`).
        let recentFrames: Int
        /// Rendered frames waiting to be shown: the producer fills it, the feeder
        /// drains it into the display layer. See `FrameStore`.
        let store: FrameStore
        /// Fill level the feeder (re)starts playback at. Reached either after the
        /// initial wait or after a rebuffer, and deliberately the same number both
        /// times so the first wait and every later one feel alike.
        let resumeFrames: Int
        /// The next frame index the feeder owes the layer. Shared because the
        /// paused redraw re-queues frames itself and has to move this past them.
        let nextToFeed = OSAllocatedUnfairLock(initialState: 0)
        /// Whether playback is currently held because the store ran dry. Read by
        /// the status and the UI; written only by the feeder.
        let bufferingNow = OSAllocatedUnfairLock(initialState: false)
        /// Set by the producer when it reaches the end of the source. The feeder
        /// needs it to tell "the buffer is empty because we are behind" from "the
        /// buffer is empty because there is no more video", which are a spinner and
        /// the end of the film respectively.
        let producerDone = OSAllocatedUnfairLock(initialState: false)

        // MARK: comparison view (video thread only, except via its locks)
        /// The left model's scaler, built when the comparison first needs it and
        /// rebuilt when the chosen model changes.
        var leftScaler: FrameScaler?
        /// The model `leftScaler` was built for, so a request is only built once.
        var leftKind: ScalerKind?
        /// The left model's render target, at the output geometry.
        var leftScratch: MTLTexture?
        /// Recent source frames, so the frame on screen can be re-rendered after
        /// a setting changes while paused. Nil until the comparison is used.
        var sources: SourceRing?
        /// Bumped by the control thread when a setting that only changes how the
        /// picture looks is altered. The video thread renders the frozen frame
        /// again whenever it sees a generation it has not rendered yet.
        let redrawWanted = OSAllocatedUnfairLock(initialState: 0)
        /// The generation the picture reflects. While playing, every produced
        /// frame acknowledges the generation it was rendered under.
        let redrawDone = OSAllocatedUnfairLock(initialState: 0)

        let frameIndex = OSAllocatedUnfairLock(initialState: 0)
        let audioFrames = OSAllocatedUnfairLock<Int64>(initialState: 0)
        let videoPrimed = OSAllocatedUnfairLock(initialState: false)
        let audioPrimed = OSAllocatedUnfairLock(initialState: false)
        /// Cross-thread: set by the video thread on failure, read by the control
        /// thread while priming.
        let failed = OSAllocatedUnfairLock(initialState: false)
        /// True once the render clock has been started for this session, so
        /// status does not report "paused" while the first frames are queuing.
        let started = OSAllocatedUnfairLock(initialState: false)
        let rate = OSAllocatedUnfairLock(initialState: RateWindow())

        var videoThread: Thread?
        var audioThread: Thread?
        var feedThread: Thread?

        init(id: Int, startAt: Double, fps: Double, outputWidth: Int, outputHeight: Int,
             decoder: FFmpegDecoder, audio: FFmpegAudioDecoder?, textRenderer: TextSubtitleRenderer?,
             srcTexture: MTLTexture, staging: MTLBuffer, pool: CVPixelBufferPool,
             aux: CFDictionary, textureCache: CVMetalTextureCache,
             scaler: FrameScaler, kind: ScalerKind, factor: UpscaleFactor,
             baseMessage: String, recentFrames: Int,
             store: FrameStore, resumeFrames: Int) {
            self.id = id
            self.startAt = startAt
            self.fps = fps
            self.outputWidth = outputWidth
            self.outputHeight = outputHeight
            self.decoder = decoder
            self.audio = audio
            self.textRenderer = textRenderer
            self.srcTexture = srcTexture
            self.staging = staging
            self.pool = pool
            self.aux = aux
            self.textureCache = textureCache
            self.scaler = scaler
            self.kind = OSAllocatedUnfairLock(initialState: kind)
            self.factor = factor
            self.baseMessage = baseMessage
            self.recentFrames = recentFrames
            self.store = store
            self.resumeFrames = resumeFrames
        }
    }

    public func open(file: URL, kind: ScalerKind, startAt: Double = 0) throws {
        lock.lock()
        currentFile = file
        self.kind = kind
        // An open is also how a stopped controller starts again; without this a
        // restart queued behind a `stop()` would be discarded by its own guard.
        stopFlag = false
        pendingSeek = startAt
        lock.unlock()
        restart(at: startAt)
    }

    /// Restart playback at an absolute content time. This is the seek path and
    /// also the implementation of the track/scale/subtitle changes, all of which
    /// alter what the decoders are asked for.
    ///
    /// The work happens on `sessionQueue`: callers get control back immediately (a
    /// seek used to freeze the UI for most of a second), and overlapping requests
    /// queue up in order instead of re-entering each other.
    private func restart(at target: Double) {
        lock.lock()
        seekGeneration += 1        // supersedes a skip that is still debouncing
        let hadSkip = pendingSkipDelta != nil
        pendingSkipDelta = nil
        lock.unlock()
        if hadSkip { status.update { $0.pendingSkipSeconds = 0 } }
        sessionQueue.async { [weak self] in self?.performRestart(at: target) }
    }

    private func performRestart(at target: Double) {
        controlMutex.lock()
        defer { controlMutex.unlock() }
        // Tear the previous session down first: mark it ended, unblock its
        // decoders, and join its producers BEFORE touching the renderers, so no
        // producer can enqueue a frame from the old position after the flush.
        lock.lock()
        sessionId += 1
        restartFlag = true
        // The producers are released so the new session can prime, but the
        // viewer's pause intent is kept separately and wins at the end.
        pausedFlag = false
        lock.broadcast()
        let old = session
        session = nil
        let file = currentFile
        lock.unlock()

        old?.decoder.stop()
        old?.audio?.stop()
        // Closing the store releases whatever it still holds and wakes the
        // feeders, so neither thread can outlive the session holding buffers.
        old?.store.close()
        old?.videoThread?.join(timeout: 2)
        old?.feedThread?.join(timeout: 2)
        old?.audioThread?.join(timeout: 2)
        player.flush()
        player.rewindToStart()

        guard let file else { return }

        lock.lock()
        guard !stopFlag else { lock.unlock(); return }
        // Clamp only once the duration is known: on the first open it is still 0,
        // and clamping against it would silently force every start offset to 0.
        let upper = durationSeconds > 0 ? max(0, durationSeconds - 0.5) : .greatestFiniteMagnitude
        let clamped = min(max(0, target), upper)
        beginSessionLocked(file: file, startAt: clamped)
        let fresh = session
        let consumedSkip = pendingSeek == target
        if consumedSkip {
            pendingSeek = nil
            pendingSkipDelta = nil
        }
        lock.unlock()
        if consumedSkip { status.update { $0.pendingSkipSeconds = 0 } }
        if let fresh { primeAndStart(fresh) }
    }

    /// Builds a session and starts its producers. Call with `lock` held.
    private func beginSessionLocked(file: URL, startAt newStart: Double) {
        restartFlag = false
        pausedFlag = false
        stopFlag = false
        startAt = max(0, newStart)

        refreshTrackListsLocked()

        // Subtitle burn is applied at decode time, so resolve it before starting.
        var subtitleLabel = "off"
        var burn: SubtitleBurn?
        var textRenderer: TextSubtitleRenderer?
        if config.burnSubtitles {
            let picked = subtitleTracks.first { $0.ordinal == config.subtitleStreamIndex } ?? subtitleTracks.first
            if let track = picked {
                let probe = try? FFmpegDecoder(url: file)
                if track.bitmap {
                    burn = SubtitleBurn(track: track, sourceWidth: probe?.meta.width ?? 0,
                                        sourceHeight: probe?.meta.height ?? 0)
                    subtitleLabel = "\(track.codec) #\(track.index) (bitmap)"
                } else {
                    textRenderer = TextSubtitleRenderer.load(url: file, ordinal: track.ordinal)
                    subtitleLabel = textRenderer == nil
                        ? "\(track.codec) #\(track.index): no cues found"
                        : "\(track.codec) #\(track.index) (native text render)"
                }
                config.subtitleStreamIndex = track.ordinal
            } else {
                subtitleLabel = "none found in file"
            }
        }

        if !audioTracks.contains(where: { $0.ordinal == config.audioStreamIndex }) {
            config.audioStreamIndex = 0
        }
        let hasAudio = config.audioEnabled && !audioTracks.isEmpty

        let dec: FFmpegDecoder
        do {
            dec = try FFmpegDecoder(url: file)
            try dec.start(at: startAt, subtitle: burn)
        } catch {
            status.update { $0.state = .failed; $0.message = "\(error)" }
            return
        }
        durationSeconds = dec.meta.durationSeconds

        let sw = dec.meta.width, sh = dec.meta.height
        let req = config.scaleFactor
        guard let eff = ScalerFactory.resolveFactor(kind: kind, requested: req,
                                                    width: sw, height: sh) else {
            status.update {
                $0.state = .failed
                $0.message = ScalerFactory.pinnedFactorRefusal(kind: kind, width: sw, height: sh)
            }
            return
        }
        effectiveFactor = eff
        // `off` is the one mode that does not magnify, so there is no factor to
        // apply and nothing to clamp: the output is the source's own geometry and
        // the scaler is a 1:1 copy.
        let ow = kind.magnifies ? sw * eff.rawValue : sw
        let oh = kind.magnifies ? sh * eff.rawValue : sh

        var note = ""
        if !kind.magnifies {
            note = "Off — playing at the source's own \(sw)x\(sh); the magnification "
                + "setting does not apply."
        } else if let pinned = kind.pinnedFactor, pinned != req {
            // Not a clamp: this mode has no other scale to fall back to, so the
            // factor control is inert here rather than the request being reduced.
            note = "\(kind.displayName) is \(pinned.displayName) only — the network has "
                + "no 2× checkpoint. Running at \(pinned.displayName)."
        } else if eff != req {
            note = "\(req.displayName) would be \(sw * req.rawValue)x\(sh * req.rawValue); "
                + "clamped to \(eff.displayName) to keep the uncompressed frame buffer bounded."
        }

        let scaler: FrameScaler
        let pool: CVPixelBufferPool
        let cache: CVMetalTextureCache
        let staging: MTLBuffer
        let srcTex: MTLTexture
        let srcDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: sw, height: sh, mipmapped: false)
        srcDesc.usage = [.shaderRead]
        srcDesc.storageMode = .shared
        do {
            scaler = try ScalerFactory.make(kind: kind, factor: eff, ctx: ctx)
            pool = try Self.makePool(width: ow, height: oh)
            cache = try Self.makeTextureCache(device: ctx.device)
            guard let s = ctx.device.makeBuffer(length: sw * sh * 4, options: .storageModeShared),
                  let t = ctx.device.makeTexture(descriptor: srcDesc) else {
                throw EngineError.metalUnavailable
            }
            staging = s
            srcTex = t
        } catch {
            status.update { $0.state = .failed; $0.message = "scaler/pool: \(error)" }
            return
        }

        let audio: FFmpegAudioDecoder?
        if hasAudio {
            let a = FFmpegAudioDecoder()
            do {
                try a.start(url: file, at: startAt, ordinal: config.audioStreamIndex)
                audio = a
            } catch {
                audio = nil
            }
        } else {
            audio = nil
        }

        // The read-ahead: `bufferSeconds` of rendered video, capped by
        // `bufferBudgetBytes` and floored at `minBufferFrames`. Whichever limit
        // binds first wins, so a 4K output gets a shorter buffer than 1080p out of
        // the same budget rather than the same number of frames at 3x the memory.
        let frameBytes = ow * oh * 4
        let bySeconds = Int((config.bufferSeconds * max(dec.meta.fps, 1)).rounded(.up))
        let byBudget = config.bufferBudgetBytes / max(1, frameBytes)
        let capacity = max(config.minBufferFrames, min(bySeconds, byBudget))
        // Fill level playback (re)starts at: half the buffer, so the first wait is
        // not the whole buffer but a stall never starts again with a nearly empty
        // one — that is what turns a rebuffer into a stutter loop.
        let resumeFrames = max(2, min(capacity, capacity / 2))
        let store = FrameStore(capacity: capacity, frameBytes: frameBytes)
        // The pool has to be able to hand out every frame the store holds, plus
        // the two the display layer keeps in its own queue, or the producer would
        // block on allocation instead of on the store.
        let depth = capacity + 4
        let aux: [String: Any] = [kCVPixelBufferPoolAllocationThresholdKey as String: depth]

        // How much source history the paused redraw may keep (see `SourceRing`):
        // one more than the frames the renderer can be holding, so the frame on
        // screen is always still available, capped by bytes — a source big enough
        // to make the ring cost more than the frame buffers it exists to serve
        // gets fewer entries instead of a bigger budget.
        let recentFrames = max(2, min(depth + 1, (64 << 20) / max(1, sw * sh * 4)))

        // Display shape: an anamorphic source keeps its DAR even though the raster
        // is uniformly scaled, so the UI lays the surface out at that ratio.
        let displayAspect: Double
        if let darString = dec.meta.displayAspect, let r = FFmpegDecoder.parseRatio(darString) {
            displayAspect = Double(r.0) / Double(r.1)
        } else {
            displayAspect = Double(ow) / Double(oh)
        }
        player.setDisplayAspect(displayAspect)

        // A/V: geometry for the status panel.
        let codecNote: String
        if let dar = dec.meta.displayAspect {
            codecNote = "Anamorphic source (display \(dar)) — stretched to its display ratio by the player."
        } else {
            codecNote = ""
        }
        let baseMessage = [note, codecNote].filter { !$0.isEmpty }.joined(separator: " ")

        let s = Session(id: sessionId, startAt: startAt, fps: dec.meta.fps,
                        outputWidth: ow, outputHeight: oh, decoder: dec, audio: audio,
                        textRenderer: textRenderer, srcTexture: srcTex, staging: staging,
                        pool: pool, aux: aux as CFDictionary, textureCache: cache,
                        scaler: scaler, kind: kind, factor: eff, baseMessage: baseMessage,
                        recentFrames: recentFrames, store: store, resumeFrames: resumeFrames)
        session = s

        status.update {
            $0.state = .preparing
            $0.file = file.lastPathComponent
            $0.mode = kind.displayName
            $0.sessionId = sessionId
            $0.sourceWidth = sw; $0.sourceHeight = sh
            $0.outputWidth = ow; $0.outputHeight = oh
            $0.displayAspectRatio = displayAspect
            $0.durationSeconds = dec.meta.durationSeconds
            $0.presentationEndSeconds = 0
            $0.framesProcessed = 0
            $0.processFps = 0
            $0.realtimeRatio = 0
            $0.positionSeconds = startAt
            $0.playbackSeconds = startAt
            $0.playbackActive = false
            $0.bufferFrames = capacity
            $0.bufferMilliseconds = Double(capacity) / max(dec.meta.fps, 1) * 1000
            $0.bufferBytes = capacity * frameBytes
            $0.bufferAheadFrames = 0
            $0.bufferAheadSeconds = 0
            $0.bufferAheadBytes = 0
            $0.bufferTargetSeconds = config.bufferSeconds
            $0.buffering = false
            $0.audioFramesEnqueued = 0
            $0.clients = 0
            $0.subtitles = subtitleLabel
            $0.subtitlesEnabled = config.burnSubtitles
            $0.audioTracks = audioTrackInfos()
            $0.subtitleTracks = subtitleTrackInfos()
            $0.audioTrackOrdinal = config.audioStreamIndex
            $0.subtitleTrackOrdinal = config.subtitleStreamIndex
            $0.scaleFactor = eff.rawValue
            $0.requestedScaleFactor = req.rawValue
            $0.videoCodec = "bgra (no encode)"
            $0.encodeBitrateBps = 0
            $0.playbackEngine = "embedded"
            $0.message = baseMessage
        }

        let vt = Thread { [weak self] in self?.videoLoop(s) }
        vt.name = "player-video"
        vt.qualityOfService = .userInitiated
        s.videoThread = vt
        vt.start()

        // The feeder runs the render clock, not the producer: it hands frames to
        // the layer at the rate the layer will take them, so the producer's being
        // slower than real time costs the buffer rather than the picture.
        let ft = Thread { [weak self] in self?.feedLoop(s) }
        ft.name = "player-feed"
        ft.qualityOfService = .userInitiated
        s.feedThread = ft
        ft.start()

        if let audio {
            let at = Thread { [weak self] in self?.audioLoop(s, audio: audio) }
            at.name = "player-audio"
            at.qualityOfService = .userInitiated
            s.audioThread = at
            at.start()
        }

        startStallWatchdog(s)
    }

    /// Fails the session visibly if the decoders yield nothing.
    ///
    /// The producers block in `read(2)` on the ffmpeg pipes, so a decoder that
    /// never produces output — stopped by a signal, blocked on a permission
    /// prompt, or sitting on a volume that went away — would otherwise leave the
    /// UI on "Preparing…" indefinitely with no explanation. This turns that silent
    /// hang into a reported failure.
    private func startStallWatchdog(_ s: Session) {
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 12) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let current = self.sessionId == s.id
            let paused = self.pausedFlag
            self.lock.unlock()
            guard current, !paused, s.frameIndex.value == 0, !s.failed.value else { return }
            s.failed.value = true
            self.status.update {
                $0.state = .failed
                $0.message = "No video frames after 12 s — the decoder produced nothing. The file "
                    + "may be unreadable, or the app may not be allowed to open it (System "
                    + "Settings ▸ Privacy & Security ▸ Files and Folders)."
            }
            // Unblock the readers so the stalled session can be torn down.
            s.decoder.stop()
            s.audio?.stop()
        }
    }

    /// Starts the render clock once the buffer holds enough to play from, so the
    /// first frame is not already late by the time it is enqueued.
    ///
    /// With a buffer this is a real wait, and deliberately so: for a mode slower
    /// than real time, the alternative to waiting here is stuttering immediately.
    /// The deadline is generous because that wait is the feature — a slow mode is
    /// supposed to make the viewer wait rather than tear — and the stall watchdog
    /// covers the case where nothing is being produced at all. A restart that
    /// replaced a paused session presents the frame at the new position and then
    /// holds it, so scrubbing while paused does not start playback.
    private func primeAndStart(_ s: Session) {
        let deadline = Date().addingTimeInterval(10)
        while Date() < deadline {
            lock.lock()
            let stale = stopFlag || restartFlag || sessionId != s.id
            let paused = pausedFlag
            lock.unlock()
            if stale || s.failed.value { return }
            if paused { return }
            // A short clip may end before filling the normal prebuffer. Its
            // available frames are enough to start once production is done.
            let videoOK = s.videoPrimed.value || (s.producerDone.value && s.store.count > 0)
                || s.store.isClosed
            let audioOK = s.audio == nil || s.audioPrimed.value
            if videoOK && audioOK { break }
            usleep(5_000)
        }
        lock.lock()
        let stale = stopFlag || restartFlag || sessionId != s.id || pausedFlag
        lock.unlock()
        guard !stale else { return }
        s.started.value = true
        player.play(fromSeconds: 0)
        // Let the first sample present, so a seek made while paused still shows the
        // new frame, then freeze on it if that is still what the viewer wants.
        // Read here rather than captured earlier: they may have paused or resumed
        // while this rebuild was running.
        usleep(40_000)
        lock.lock()
        let hold = pauseRequested
        if hold { pausedFlag = true }
        lock.unlock()
        if hold {
            player.pause()
            status.update { $0.state = .paused; $0.playbackActive = false }
        } else {
            status.update { $0.playbackActive = true; $0.state = .processing }
        }
    }

    // MARK: - Controls

    public func pause() {
        lock.lock()
        pausedFlag = true
        pauseRequested = true
        lock.unlock()
        player.pause()
        status.update { $0.state = .paused; $0.playbackActive = false }
    }

    public func resume() {
        lock.lock()
        let s = session
        pausedFlag = false
        pauseRequested = false
        lock.broadcast()
        lock.unlock()
        guard let s else { return }
        // Resuming a session that was paused before its first frames were queued
        // (still on "Preparing…") takes over what `primeAndStart` would have
        // done, so status does not stay on "Preparing…" while the picture plays.
        s.started.value = true
        player.play(fromSeconds: player.currentSeconds)
        status.update { $0.state = .processing; $0.playbackActive = true }
    }

    /// The viewer-facing state: what they last asked for, not whether the
    /// producers happen to be parked right now (a rebuild releases them).
    public var isPaused: Bool {
        lock.lock(); defer { lock.unlock() }
        return pauseRequested
    }

    /// The position the viewer is actually at: the session's content offset plus
    /// the render clock's elapsed time.
    public var playbackSeconds: Double {
        lock.lock()
        let origin = startAt
        let live = session != nil && !stopFlag
        lock.unlock()
        guard live else { return origin }
        return origin + max(0, player.currentSeconds)
    }

    public func seek(to seconds: Double) {
        let target = max(0, seconds)
        lock.lock()
        pendingSeek = target
        lock.unlock()
        restart(at: target)
    }

    public func seek(by delta: Double) {
        lock.lock()
        let target = requestedPositionLocked() + delta
        pendingSeek = target
        // Every press counts: the running total is published so the UI can show
        // "+10 s", "+20 s", "+30 s" as the viewer presses, which is what makes the
        // wait for the debounce feel like a response rather than a delay.
        let total = (pendingSkipDelta ?? 0) + delta
        pendingSkipDelta = total
        seekGeneration += 1
        let generation = seekGeneration
        lock.unlock()
        status.update { $0.pendingSkipSeconds = total }
        // One rebuild for the whole burst: each one costs most of a second, so
        // acting per press queued a string of them and made fast skipping crawl.
        sessionQueue.asyncAfter(deadline: .now() + Self.seekDebounce) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            // A later press, or any other restart request, supersedes this one.
            let superseded = self.seekGeneration != generation
            let target = self.pendingSeek
            self.lock.unlock()
            guard !superseded, let target else { return }
            self.performRestart(at: target)
        }
    }

    /// The position the next restart should land on: a seek that is still queued if
    /// there is one, otherwise the live clock. Call with `lock` held.
    private func requestedPositionLocked() -> Double {
        pendingSeek ?? playbackSecondsFromLocked()
    }

    /// Switch the enhancement mode mid-playback.
    ///
    /// Between two *magnifying* modes the output dimensions are unchanged, so this
    /// is a hot swap: the video thread builds the new scaler before the next frame
    /// and no session restart (and no rebuffer) happens.
    ///
    /// Crossing to or from `off` is the exception, because off is 1:1: the output
    /// geometry itself changes, so a hot swap would leave the scaler writing into a
    /// buffer the session did not allocate. That case restarts at the current
    /// position, exactly as a scale change does.
    ///
    /// A mode *pinned* to one factor is the second exception, and it is easy to
    /// miss: switching from the CNN at 2x to Real-ESRGAN changes the output
    /// geometry from 2x to 4x even though both modes magnify, so the old
    /// `magnifies` test said "hot-swap" and the build then failed at 2x and left
    /// the picture on the old mode. The test is on the resolved factor instead.
    /// A kind that cannot run on this source at all is refused here rather than
    /// restarting into a failure, so the playing picture is kept.
    public func setKind(_ newKind: ScalerKind) {
        lock.lock()
        let s = session
        var geometryChanges = false
        var refusal: String?
        if let s {
            let w = s.srcTexture.width, h = s.srcTexture.height
            if let newFactor = ScalerFactory.resolveFactor(kind: newKind,
                                                           requested: config.scaleFactor,
                                                           width: w, height: h) {
                geometryChanges = s.kind.value.magnifies != newKind.magnifies
                    || newFactor != s.factor
            } else {
                refusal = ScalerFactory.pinnedFactorRefusal(kind: newKind, width: w, height: h)
            }
        }
        if refusal == nil { kind = newKind }
        let pos = requestedPositionLocked()
        if geometryChanges { pendingSeek = pos + 0.1 }
        lock.unlock()

        if let refusal {
            // Same shape as a failed build: say why, keep playing what is playing.
            status.update { $0.message = refusal }
            viewSettingChanged()
            return
        }
        if geometryChanges {
            restart(at: pos + 0.1)
        } else {
            // Recorded as a *request*; `kind` is only updated once the new scaler
            // has actually been built, so a failed build is reported instead of
            // leaving the mode claiming something the picture is not using.
            s?.requestedKind.value = newKind
            status.update { $0.mode = newKind.displayName }
        }
        // The enhancement is the right-hand side of the comparison, so changing it
        // has to redraw the frame on screen too.
        viewSettingChanged()
    }

    /// Show two models side by side across a movable divider (left half = the
    /// model chosen by `setCompareLeft`, right half = the enhancement).
    ///
    /// This is a view-time overlay, not a pipeline change: the right half is the
    /// frame the session would have produced anyway, and the left half is a
    /// second scaler's output composited over it. Switching it on mid-playback
    /// needs no rebuild, and the split can be moved to either edge — at 0 the
    /// enhancement fills the picture, at 1 the left model does.
    public var isComparing: Bool { compareFlag.value }
    public var compareLeftKind: ScalerKind { compareLeft.value }
    public var compareSplitFraction: Double { compareSplit.value }

    public func setCompare(_ on: Bool) {
        guard comparison != nil else {
            status.update {
                $0.comparing = false
                $0.message = "Comparison view unavailable: the split kernel did not build."
            }
            return
        }
        compareFlag.value = on
        status.update {
            $0.comparing = on
            $0.compareLeftMode = compareLeft.value.displayName
            $0.compareSplit = compareSplit.value
        }
        viewSettingChanged()
    }

    /// Choose the model shown on the left of the divider. `.bypass` is the plain
    /// bilinear upscale — the original video as any other player would show it.
    public func setCompareLeft(_ kind: ScalerKind) {
        compareLeft.value = kind
        status.update { $0.compareLeftMode = kind.displayName }
        viewSettingChanged()
    }

    /// Move the divider: 0 shows only the enhancement, 1 only the left model.
    public func setCompareSplit(_ fraction: Double) {
        let clamped = min(max(fraction, 0), 1)
        compareSplit.value = clamped
        status.update { $0.compareSplit = clamped }
        viewSettingChanged()
    }

    /// A setting changed that only affects how the current frame looks: the
    /// comparison, its divider, or the enhancement model.
    ///
    /// While playing, the next frame the producer renders uses the new value and
    /// nothing has to happen. While paused the producer is parked, so it is woken
    /// to render the frame on screen again — that is what makes the comparison
    /// usable on a stopped frame, which is the only way to compare two models on
    /// one instant properly. Requests made while the producer is busy with a
    /// rebuild are dropped: that session renders with the new value anyway.
    private func viewSettingChanged() {
        lock.lock()
        let s = session
        let parked = pausedFlag && pauseRequested
        lock.unlock()
        guard let s, parked else { return }
        s.redrawWanted.value &+= 1
        lock.lock()
        lock.broadcast()          // the producer is waiting on this condition
        lock.unlock()
    }

    public func setScaleFactor(_ factor: UpscaleFactor) {
        lock.lock()
        let changed = config.scaleFactor != factor
        config.scaleFactor = factor
        let pos = requestedPositionLocked()
        if changed { pendingSeek = pos + 0.1 }
        lock.unlock()
        guard changed else { return }
        restart(at: pos + 0.1)
    }

    public func setAudioTrack(_ ordinal: Int) {
        lock.lock()
        let changed = config.audioStreamIndex != max(0, ordinal)
        config.audioStreamIndex = max(0, ordinal)
        let pos = requestedPositionLocked()
        if changed { pendingSeek = pos + 0.1 }
        lock.unlock()
        guard changed else { return }
        restart(at: pos + 0.1)
    }

    public func setSubtitles(_ enabled: Bool, streamIndex: Int? = nil) {
        lock.lock()
        let changed = config.burnSubtitles != enabled
            || (streamIndex != nil && config.subtitleStreamIndex != streamIndex!)
        config.burnSubtitles = enabled
        if let streamIndex { config.subtitleStreamIndex = streamIndex }
        let pos = requestedPositionLocked()
        if changed { pendingSeek = pos + 0.1 }
        lock.unlock()
        guard changed else { return }
        restart(at: pos + 0.1)
    }

    public func setSubtitleTrack(_ ordinal: Int) { setSubtitles(true, streamIndex: max(0, ordinal)) }

    public var availableAudioTracks: [TrackInfo] {
        lock.lock(); defer { lock.unlock() }
        return audioTrackInfos()
    }
    public var availableSubtitleTracks: [TrackInfo] {
        lock.lock(); defer { lock.unlock() }
        return subtitleTrackInfos()
    }

    public func stop() {
        // Runs on the session queue so it cannot interleave with a rebuild that is
        // already in flight, but blocks the caller: the app stops the engine on
        // quit, and returning before the ffmpeg children are dead would leave them
        // behind. Called from the queue itself (a deinit during a rebuild) it just
        // runs in place — `sync` there would deadlock.
        if DispatchQueue.getSpecific(key: Self.sessionQueueKey) != nil {
            performStop()
        } else {
            sessionQueue.sync { performStop() }
        }
    }

    private func performStop() {
        controlMutex.lock()
        defer { controlMutex.unlock() }
        lock.lock()
        stopFlag = true
        restartFlag = true
        pausedFlag = false
        pauseRequested = false
        pendingSeek = nil
        pendingSkipDelta = nil
        sessionId += 1
        lock.broadcast()
        let old = session
        session = nil
        lock.unlock()

        old?.decoder.stop()
        old?.audio?.stop()
        old?.store.close()
        old?.videoThread?.join(timeout: 2)
        old?.feedThread?.join(timeout: 2)
        old?.audioThread?.join(timeout: 2)
        player.stop()
        status.update { $0.state = .idle; $0.playbackActive = false }
    }

    deinit { stop() }

    // MARK: - Producers

    private struct Output {
        let pb: CVPixelBuffer
        let cvTexture: CVMetalTexture
        let texture: MTLTexture
    }

    private func videoLoop(_ s: Session) {
        while true {
            let wake = waitForWork(s)
            if wake == .ended { break }
            if wake == .pausedRedraw { redrawPausedFrame(s); continue }
            // Read before rendering: a setting changed after this point is not in
            // the frame about to be queued, and must be picked up by a redraw.
            let generation = s.redrawWanted.value

            // The store is the backpressure now, not the display layer: the
            // producer runs ahead until the buffer is full, and that lead is the
            // whole point of `bufferSeconds`.
            if !s.store.waitForRoom() { break }

            guard let out = acquireOutput(s) else {
                // nil = session ended, or paused with the pool full; re-loop so
                // the pause check parks us instead of spinning.
                continue
            }

            guard s.decoder.readFrame(into: s.staging.contents()) else {
                reportEmptyDecode(s)
                break
            }

            let meta = s.decoder.meta
            if let renderer = s.textRenderer {
                let t = s.startAt + Double(s.frameIndex.value) / max(s.fps, 1)
                renderer.blend(into: s.staging.contents(), bytesPerRow: meta.width * 4,
                               width: meta.width, height: meta.height, time: t)
            }

            let region = MTLRegionMake2D(0, 0, s.srcTexture.width, s.srcTexture.height)
            let source = sourceTexture(for: s, index: s.frameIndex.value)
            source.replace(region: region, mipmapLevel: 0,
                           withBytes: s.staging.contents(), bytesPerRow: meta.width * 4)

            // Hot scaler swap (mode change) without restarting the session. The
            // request is only cleared once the build succeeds; on failure the old
            // scaler keeps running and the UI is told, rather than silently
            // ignoring the switch.
            applyRequestedKind(s)

            do {
                try s.scaler.scale(src: source, dst: out.texture)
            } catch {
                status.update { $0.state = .failed; $0.message = "scale: \(error)" }
                s.failed.value = true
                break
            }

            // Comparison overlay: the right half of the frame is the enhancement
            // that was just rendered; the left half is a second model composited
            // over it, so the two can be seen against each other. Everything is
            // ordered after the scale above, and the output geometry is unchanged.
            if compareFlag.value, let cmp = comparison {
                do {
                    try applyComparison(cmp, to: s, source: source, out: out)
                } catch {
                    // A Metal failure here says nothing about the session, so the
                    // picture goes back to normal instead of the run being killed.
                    compareFlag.value = false
                    status.update {
                        $0.comparing = false
                        $0.message = "Comparison view stopped: \(error)"
                    }
                }
            }

            let idx = s.frameIndex.value
            onOutputFrame?(out.pb, idx)
            // The scale above was synchronous, so the GPU is done with the
            // texture. The buffer goes into the store rather than straight to the
            // layer: the feeder owns the handoff, and this frame may sit here for
            // as long as the buffer is deep.
            if !s.store.push(FrameStore.Frame(index: idx, pb: out.pb,
                                              cvTexture: out.cvTexture, texture: out.texture)) {
                // Impossible unless the store closed or the wait above was raced
                // by a close; dropping the frame would leave a hole the feeder
                // stalls on for the rest of the session.
                break
            }
            s.frameIndex.value = idx + 1
            // "Primed" now means the buffer holds enough to start, not that one
            // frame was queued — `primeAndStart` waits on it.
            if s.store.count >= s.resumeFrames { s.videoPrimed.value = true }
            s.redrawDone.value = generation
            postStatus(s)
        }
        // The end of the source, or a failure. Completion is reported by the
        // *feeder*, once it has drained what is still buffered — with a buffer of
        // five seconds, saying "finished" the moment the decoder runs out would
        // declare the film over while seconds of it are still queued to play.
        s.producerDone.value = true
        if s.failed.value { finishSession(s) }
    }

    /// Hands rendered frames to the display layer, in order, as fast as the layer
    /// will take them — and holds the clock when there is nothing to hand over.
    ///
    /// This is the half of the old single producer loop that has to run at the
    /// *presentation* rate rather than the render rate. Keeping them together is
    /// what made a mode slower than real time stutter per frame: the loop could
    /// only enqueue as fast as it could render, and the layer queued a further two
    /// frames and no more (`FrameStore`), so there was nowhere for a lead to live.
    private func feedLoop(_ s: Session) {
        while true {
            // Re-read the index each time round: the paused redraw moves it past
            // the frames it re-queued itself.
            let next = s.nextToFeed.value

            if parkFeederIfNeededOrEnded(s) { break }
            if s.store.isClosed { break }

            // Only feed a clock that is running. The layer's queue is two frames,
            // so handing it anything while the clock is stopped (before playback,
            // or held for a refill) fills that queue and blocks this thread — and
            // it would drain the store into the layer, which is the opposite of
            // what holding the clock is for.
            if !s.started.value || !player.isPlaying {
                if s.bufferingNow.value {
                    // Held for a refill: wait for the watermark without touching
                    // the store, then start the clock again.
                    if !s.store.waitForCount(s.resumeFrames) { break }
                    endBuffering(s)
                    continue
                }
                usleep(2_000)
                continue
            }

            guard let frame = s.store.frame(at: next) else {
                // Out of video. If the producer has finished there is none coming:
                // play out what is left and end the session, rather than holding
                // the clock on a spinner that will never fill.
                if s.producerDone.value { break }
                // Otherwise an underrun. The whole point of the buffer is to make
                // this rare, and to make it a pause rather than a torn frame.
                beginBuffering(s)
                continue
            }
            if !waitForLayerReady(s) { break }

            guard let sample = player.makeVideoSample(pixelBuffer: frame.pb,
                                                      pts: Double(next) / max(s.fps, 1),
                                                      fps: s.fps) else {
                s.store.release(through: next)
                s.nextToFeed.value = next + 1
                continue
            }
            if player.enqueueVideo(sample) {
                withExtendedLifetime(frame.cvTexture) {}
                s.store.release(through: next)
                s.nextToFeed.value = next + 1
            } else {
                usleep(1_000)
            }
        }
        // Either the source ended and the buffer is drained, or something failed.
        // Both are reported here rather than by the producer because the frames it
        // already rendered are still being played out at this point.
        finishSession(s)
    }

    /// Holds the clock and tells the viewer why, once. Called on underrun.
    private func beginBuffering(_ s: Session) {
        if s.bufferingNow.value { return }
        s.bufferingNow.value = true
        // Hold where the clock is, so the frame on screen stays and the timeline
        // does not advance past picture we do not have yet.
        player.holdClock(atSeconds: player.currentSeconds)
        status.update {
            $0.buffering = true
            $0.bufferAheadFrames = s.store.count
            $0.bufferAheadSeconds = Double(s.store.count) / max(s.fps, 1)
            $0.bufferAheadBytes = s.store.bytes
            $0.playbackActive = false
        }
    }

    /// Restarts the clock once the store has refilled to the watermark.
    private func endBuffering(_ s: Session) {
        s.bufferingNow.value = false
        player.play(fromSeconds: player.currentSeconds)
        status.update {
            $0.buffering = false
            $0.playbackActive = true
            $0.state = .processing
        }
    }

    /// Waits until the display layer will accept a frame. Returns false when the
    /// session ended or its renderer failed and will never accept one again.
    ///
    /// A paused layer keeps a full queue for the whole pause and reports
    /// not-ready the entire time. The feeder spends most of its life in this
    /// wait, so those seconds must not count: giving up used to end the session
    /// with "Playback stopped before the final frame" and leave playback,
    /// seeking, and the menus disabled.
    private func waitForLayerReady(_ s: Session) -> Bool {
        var stalled = 0
        while true {
            if player.videoReady { return true }
            lock.lock()
            let paused = pausedFlag
            let ended = isEndedLocked(s.id)
            lock.unlock()
            if ended || s.failed.value { return false }
            if paused {
                if parkFeederIfNeededOrEnded(s) { return false }
                stalled = 0
                continue
            }
            if player.videoFailed { return false }
            usleep(2_000)
            stalled += 1
            // ~6 s with no accept while the clock is running: wedged, not busy.
            if stalled > 3000 { return false }
        }
    }

    /// The feeder parks while paused so a pause does not drain the buffer into the
    /// layer — the frames were expensive and resuming should be instant.
    private func parkFeederIfNeededOrEnded(_ s: Session) -> Bool {
        lock.lock()
        while pausedFlag && !isEndedLocked(s.id) { lock.wait() }
        let ended = isEndedLocked(s.id)
        lock.unlock()
        return ended
    }

    /// The source texture for frame `index`: a slot in the comparison's ring of
    /// recent frames once that exists, otherwise the one reusable source texture.
    ///
    /// The ring is created the first time the comparison is used and kept for the
    /// rest of the session, so the frames behind the one on screen stay available
    /// for the paused redraw — including when the comparison is switched off
    /// again, which is what makes switching it back on while paused work.
    private func sourceTexture(for s: Session, index: Int) -> MTLTexture {
        guard compareFlag.value || s.sources != nil else { return s.srcTexture }
        if s.sources == nil {
            s.sources = SourceRing(device: ctx.device, width: s.srcTexture.width,
                                   height: s.srcTexture.height, capacity: s.recentFrames)
        }
        return s.sources?.slot(for: index) ?? s.srcTexture
    }

    /// The source of frame `index`, if it is still held: the ring keeps the
    /// recent ones, and the newest frame is always the reusable source texture.
    private func retainedSource(for s: Session, index: Int) -> MTLTexture? {
        if let held = s.sources?.texture(at: index) { return held }
        return index == s.frameIndex.value - 1 ? s.srcTexture : nil
    }

    /// Builds the scaler for a mode the viewer picked and swaps it in, or reports
    /// why it could not. Shared by the play loop and the paused redraw, so the
    /// enhancement can be changed while paused rather than only on resume.
    private func applyRequestedKind(_ s: Session) {
        guard let wanted = s.requestedKind.value else { return }
        if let fresh = try? ScalerFactory.make(kind: wanted, factor: s.factor, ctx: ctx) {
            s.scaler = fresh
            s.kind.value = wanted
            s.requestedKind.value = nil
            status.update { $0.mode = wanted.displayName; $0.message = s.baseMessage }
        } else {
            s.requestedKind.value = nil
            let kept = s.kind.value
            status.update {
                $0.mode = kept.displayName
                $0.message = "Could not switch to \(wanted.displayName); still using \(kept.displayName)."
            }
        }
    }

    /// Re-renders the frame the renderer is showing with the settings as they are
    /// now, and re-queues it — so moving the divider, changing the model on either
    /// side, or turning the comparison on or off takes effect while paused, on the
    /// frame the viewer stopped on, instead of waiting for playback to resume.
    ///
    /// Runs on the video thread while the session is parked. The layer's queue is
    /// flushed first, because a frame can only replace the image on screen if its
    /// timestamp does not follow what is already queued — and the frames that were
    /// queued behind the shown one are re-rendered with it, in order, so resuming
    /// still plays them rather than skipping the flush.
    private func redrawPausedFrame(_ s: Session) {
        let generation = s.redrawWanted.value
        defer { s.redrawDone.value = generation }

        // A model picked while paused is built here, exactly as the play path
        // builds it, so choosing one in the comparison bar is not queued up behind
        // the resume. A failure is reported by `applyRequestedKind` and the old
        // model is redrawn, which is what the viewer was looking at anyway.
        applyRequestedKind(s)

        let fps = max(s.fps, 1)
        let last = max(0, s.frameIndex.value - 1)
        // The frame on screen is the newest one whose timestamp the render clock
        // has reached — which is what the layer shows, and with the producer
        // behind the clock it is simply the newest frame produced.
        let shown = Int((max(0, player.currentSeconds) * fps).rounded(.down))
        // The oldest frame whose source is still held. Without one — the
        // comparison was switched on while paused, so nothing was being retained —
        // the newest frame stands in for the one on screen.
        let oldestHeld = s.sources.flatMap { $0.isEmpty ? nil : $0.oldest } ?? last
        let from = min(last, max(shown, oldestHeld))
        // A stand-in frame is stamped *after* the paused clock, so it would only
        // queue up rather than replace the picture; the clock moves onto it
        // instead, which costs a few frames of content and is the one case where
        // the still is not the frame the viewer stopped on.
        let standIn = from > shown

        // Keep the image: the frame on screen stays there until the redraw
        // replaces it a few milliseconds later, instead of flashing black.
        player.flush(removeImage: false)
        if standIn { player.holdClock(atSeconds: (Double(from) + 0.5) / fps) }

        for index in from...last {
            guard let source = retainedSource(for: s, index: index),
                  let out = acquireOutput(s, redrawing: true) else { break }
            do {
                try s.scaler.scale(src: source, dst: out.texture)
            } catch {
                status.update { $0.state = .failed; $0.message = "scale: \(error)" }
                s.failed.value = true
                return
            }
            if compareFlag.value, let cmp = comparison {
                do {
                    try applyComparison(cmp, to: s, source: source, out: out)
                } catch {
                    compareFlag.value = false
                    status.update {
                        $0.comparing = false
                        $0.message = "Comparison view stopped: \(error)"
                    }
                }
            }
            guard let sample = player.makeVideoSample(pixelBuffer: out.pb,
                                                      pts: Double(index) / fps, fps: s.fps) else { break }
            withExtendedLifetime(out.cvTexture) {}
            guard enqueueWhenReady(sample, s) else { break }
            onOutputFrame?(out.pb, index)
        }
        // These frames were just re-rendered with the new settings and handed to
        // the layer, so the copies the store is holding are stale — they were
        // rendered with the settings the viewer just changed. Drop them and move
        // the feeder past them; it continues from whatever the producer renders
        // next, with the new settings.
        s.store.discardAll()
        s.nextToFeed.value = last + 1
        s.bufferingNow.value = false
        postStatus(s)
    }

    /// Hands a re-rendered frame to the renderer, waiting briefly if the layer is
    /// not ready for it yet (it can report that for a moment after a flush).
    /// Dropping one here would leave the picture showing a frame from before the
    /// change, so this waits rather than falling through.
    private func enqueueWhenReady(_ sample: CMSampleBuffer, _ s: Session) -> Bool {
        let deadline = Date().addingTimeInterval(1)
        while true {
            if player.enqueueVideo(sample) { return true }
            lock.lock()
            let ended = isEndedLocked(s.id)
            lock.unlock()
            if ended || Date() >= deadline { return false }
            usleep(2_000)
        }
    }
    /// Puts the left model's pixels (and the divider) into a frame that already
    /// holds the enhancement. Runs on the video thread, once per frame.
    ///
    /// A split at 0 needs no work at all: the enhancement already fills the
    /// picture, so "show only the enhanced side" costs exactly what normal
    /// playback costs. A split at 1 copies the whole frame across.
    private func applyComparison(_ cmp: SplitCompositor, to s: Session,
                                 source: MTLTexture, out: Output) throws {
        let width = s.outputWidth
        let splitX = min(width, max(0, Int((Double(width) * compareSplit.value).rounded())))
        guard splitX > 0 else { return }

        // Off presents the frame at the source's own geometry while a model's
        // output is 2x/4x, so the two have no shared pixel grid to be composited
        // on. Refusing is the honest outcome — the alternative, scaling one side,
        // would make the comparison a picture of the scaler rather than of the
        // models.
        guard s.kind.value.magnifies else {
            compareFlag.value = false
            status.update {
                $0.comparing = false
                $0.message = "The comparison needs a mode that scales; Off plays at "
                    + "the source's own size."
            }
            return
        }

        let divider = SplitCompositor.dividerWidth(forWidth: width)
        let wanted = compareLeft.value

        // Same reasoning for a left half set to a non-magnifying kind. Every route
        // in — the menu, `_COMPARE_LEFT`, the `compareleft:` script verb — arrives
        // here, so the refusal lives here rather than in each of them.
        guard wanted.magnifies else {
            compareFlag.value = false
            status.update {
                $0.comparing = false
                $0.message = "\(wanted.displayName) cannot be a comparison side: a half "
                    + "is composited at the output geometry, and it does not scale."
            }
            return
        }

        // Two sides showing the same model need no second pass and no scratch
        // buffer: the pixels under the left half are already that model's output,
        // so only the divider is drawn. That keeps "compare CNN with CNN" from
        // paying twice for nothing.
        if wanted == s.kind.value {
            try cmp.drawDivider(into: out.texture, at: splitX, divider: divider)
            return
        }

        // Rebuilding the left scaler is a per-frame comparison against a stored
        // value: it only changes when the viewer picks another model.
        if s.leftScaler == nil || s.leftKind != wanted {
            guard let fresh = try? ScalerFactory.make(kind: wanted, factor: s.factor, ctx: ctx) else {
                compareFlag.value = false
                // A half is built at the *session's* factor, so a mode pinned to
                // another one cannot be a side of this comparison. That is worth
                // saying precisely, because the fix is a different magnification
                // rather than a different model.
                let why: String
                if let pinned = wanted.pinnedFactor, pinned != s.factor {
                    why = "\(wanted.displayName) is \(pinned.displayName) only, so it cannot be "
                        + "a half of a \(s.factor.displayName) comparison — switch to "
                        + "\(pinned.displayName) to compare it."
                } else {
                    why = "Could not build \(wanted.displayName) for the comparison."
                }
                status.update {
                    $0.comparing = false
                    $0.message = why
                }
                return
            }
            s.leftScaler = fresh
            s.leftKind = wanted
            status.update { $0.compareLeftMode = wanted.displayName }
        }
        if s.leftScratch == nil || s.leftScratch?.width != width
            || s.leftScratch?.height != s.outputHeight {
            s.leftScratch = try cmp.makeScratch(width: width, height: s.outputHeight)
        }
        guard let scaler = s.leftScaler, let scratch = s.leftScratch else { return }
        try scaler.scale(src: source, dst: scratch)
        try cmp.composite(left: scratch, into: out.texture, split: splitX, divider: divider)
    }

    /// End of stream: the last frames are already queued on the renderer, so the
    /// clock is left running to play them out; only the status changes.
    private func finishSession(_ s: Session) {
        lock.lock()
        let ended = isEndedLocked(s.id)
        lock.unlock()
        guard !ended, !s.failed.value else { return }
        let allFramesFed = s.nextToFeed.value >= s.frameIndex.value
        status.update {
            if $0.state != .failed {
                $0.state = allFramesFed ? .finished : .failed
                if !allFramesFed { $0.message = "Playback stopped before the final frame." }
            }
            $0.framesProcessed = s.frameIndex.value
            $0.presentationEndSeconds = allFramesFed
                ? s.startAt + Double(s.nextToFeed.value) / max(s.fps, 1) : 0
            $0.playbackSeconds = min($0.durationSeconds, $0.playbackSeconds)
        }
    }

    /// Waits while paused; returns true when the session has ended.
    private func parkIfNeededOrEnded(_ s: Session) -> Bool {
        lock.lock()
        while pausedFlag && !isEndedLocked(s.id) { lock.wait() }
        let ended = isEndedLocked(s.id)
        lock.unlock()
        return ended
    }

    /// Why the producer stopped waiting.
    private enum ProducerWake { case ended, run, pausedRedraw }

    /// The video thread's park: waits until the session ends, playback resumes, or
    /// a setting changed that the frame on screen has to be re-rendered for.
    /// Only the video thread redraws, so unlike `parkIfNeededOrEnded` this does not
    /// absorb a redraw request — the audio thread must leave it for the video one.
    private func waitForWork(_ s: Session) -> ProducerWake {
        lock.lock()
        while true {
            if isEndedLocked(s.id) { lock.unlock(); return .ended }
            if !pausedFlag { lock.unlock(); return .run }
            if s.redrawWanted.value != s.redrawDone.value { lock.unlock(); return .pausedRedraw }
            lock.wait()
        }
    }

    private func isEndedLocked(_ id: Int) -> Bool {
        stopFlag || restartFlag || sessionId != id
    }

    /// A pool buffer for one output frame, or nil when the session ended / is
    /// paused. The pool's allocation threshold is the backstop: once the store's
    /// capacity worth of frames are held the call returns
    /// kCVReturnWouldExceedAllocationThreshold until the feeder releases one.
    ///
    /// The producer does *not* wait for the display layer here any more — that is
    /// the feeder's job. Waiting on it was what capped the read-ahead at the
    /// layer's own 2-frame queue.
    ///
    /// `redrawing` is the one caller that runs *while* parked, so it must not stop
    /// for the pause — and it allocates outside the threshold, because it is
    /// re-queueing frames that were counted before the flush and the pool would
    /// otherwise refuse the last of them.
    private func acquireOutput(_ s: Session, redrawing: Bool = false) -> Output? {
        var stalled = 0
        while true {
            lock.lock()
            let ended = isEndedLocked(s.id)
            let paused = pausedFlag
            lock.unlock()
            if ended || paused { return nil }
            if redrawing || !s.store.isClosed {
                var pb: CVPixelBuffer?
                let st = redrawing
                    ? CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, s.pool, &pb)
                    : CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
                        kCFAllocatorDefault, s.pool, s.aux, &pb)
                if st == kCVReturnWouldExceedAllocationThreshold {
                    usleep(3_000)
                    stalled += 1
                    if stalled > 2000 { return nil }   // ~6 s; give up on a wedged store
                    continue
                }
                guard st == kCVReturnSuccess, let pb else { return nil }
                var cv: CVMetalTexture?
                guard CVMetalTextureCacheCreateTextureFromImage(
                    kCFAllocatorDefault, s.textureCache, pb, nil, .bgra8Unorm,
                    s.outputWidth, s.outputHeight, 0, &cv) == kCVReturnSuccess,
                    let cv, let tex = CVMetalTextureGetTexture(cv) else { return nil }
                return Output(pb: pb, cvTexture: cv, texture: tex)
            }
            usleep(3_000)
        }
    }

    private func audioLoop(_ s: Session, audio: FFmpegAudioDecoder) {
        let framesPerChunk = 1024
        let byteCount = framesPerChunk * FFmpegAudioDecoder.bytesPerFrame
        let buf = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
        defer { buf.deallocate() }

        while !parkIfNeededOrEnded(s) {
            let n = audio.readFrames(into: buf, frames: framesPerChunk)
            if n <= 0 { break }   // EOF or decoder stopped

            // The renderer only accepts data when it is ready, and a *paused* one
            // legitimately stays full for the whole pause — so this waits for the
            // session to end or the renderer to fail, parking while paused. It
            // must never give up on a timer: the audio path has no restart, so
            // abandoning the loop silences the rest of the movie (which is what
            // a paused interval used to do after ~4.5 s).
            guard waitForAudioReady(s) else { return }

            let pts = Double(s.audioFrames.value) / Double(FFmpegAudioDecoder.sampleRate)
            let data = Data(bytes: buf, count: n * FFmpegAudioDecoder.bytesPerFrame)
            if let sample = player.makeAudioSample(pcm: data, frames: n, pts: pts) {
                player.enqueueAudio(sample)
                s.audioFrames.value += Int64(n)
                s.audioPrimed.value = true
            }
        }
    }

    /// Waits until the audio renderer will accept a chunk. Parks while the session
    /// is paused; returns false only when there is nothing left to feed — the
    /// session ended, or the renderer failed and will never be ready again.
    private func waitForAudioReady(_ s: Session) -> Bool {
        while true {
            if player.audioReady { return true }
            // Blocks while paused and returns true once the session has ended.
            if parkIfNeededOrEnded(s) { return false }
            // A failed renderer never becomes ready again, so say so rather than
            // spinning: audio stops for this session while the picture continues.
            if player.audioFailed {
                status.update {
                    $0.message = "Audio stopped: the output device rejected the audio renderer."
                }
                return false
            }
            usleep(3_000)
        }
    }

    // MARK: - Status

    /// Throughput over a sliding window, in content seconds produced per second of
    /// render clock. Keyed to the *clock* rather than wall time on purpose: while
    /// paused neither the frame counter nor the clock advances, so a pause adds
    /// nothing to either delta instead of dragging the average down for the rest
    /// of the session, and the burst of frames that fills the queue before the
    /// clock starts is excluded rather than averaged in.
    private struct RateWindow {
        var at = Date()
        var frames = 0
        var clock: Double = 0
        var primed = false
        var fps: Double = 0
    }

    private func postStatus(_ s: Session) {
        let idx = s.frameIndex.value
        let now = Date()
        let clock = max(0, player.currentSeconds)
        let fps = max(s.fps, 1)

        let measured = s.rate.withLock { w -> Double in
            guard w.primed else {
                w.primed = true; w.at = now; w.frames = idx; w.clock = clock
                return 0
            }
            let dClock = clock - w.clock
            // A full second of playing time: shorter windows read high, because a
            // refilled queue produces a few frames in an instant. A window whose
            // clock never advanced is a pause, and keeps the last measurement.
            guard now.timeIntervalSince(w.at) >= 1.0, dClock >= 1.0 else { return w.fps }
            let value = Double(idx - w.frames) / dClock
            w.at = now; w.frames = idx; w.clock = clock; w.fps = value
            return value
        }

        let contentPos = s.startAt + Double(idx) / fps
        let playPos = s.startAt + clock
        let playing = player.isPlaying
        let started = s.started.value
        // Held-for-refill is not a pause: the clock is stopped, so `playing` is
        // false, but the viewer's intent has not changed and the UI should show a
        // spinner rather than a Play button.
        let buffering = s.bufferingNow.value
        let ahead = s.store.count
        let thermKey: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermKey = "nominal"
        case .fair: thermKey = "fair"
        case .serious: thermKey = "serious"
        case .critical: thermKey = "critical"
        @unknown default: thermKey = "nominal"
        }
        status.update {
            // Before the clock starts the first frames are still being queued, so
            // report "preparing" rather than the misleading "paused".
            $0.state = started ? ((playing || buffering) ? .processing : .paused) : .preparing
            $0.framesProcessed = idx
            $0.processFps = measured
            $0.realtimeRatio = measured / fps
            $0.positionSeconds = contentPos
            $0.playbackSeconds = playPos
            $0.playbackActive = playing
            $0.buffering = buffering
            $0.bufferAheadFrames = ahead
            $0.bufferAheadSeconds = Double(ahead) / fps
            $0.bufferAheadBytes = s.store.bytes
            $0.audioFramesEnqueued = s.audioFrames.value
            $0.vlcReportedSeconds = 0
            $0.thermalState = thermKey
            $0.memoryFootprintMB = PipelineController.memoryFootprintMB()
            $0.renderStatus = player.renderStatus
        }
    }

    // MARK: - Tracks & probing

    private func playbackSecondsFromLocked() -> Double {
        let origin = startAt
        guard session != nil else { return origin }
        return origin + max(0, player.currentSeconds)
    }

    private func refreshTrackListsLocked() {
        guard probedFile != currentFile else { return }
        probedFile = currentFile
        audioTracks = []
        subtitleTracks = []
        guard let file = currentFile else { return }
        audioTracks = FFmpegDecoder.probeAudio(url: file)
        subtitleTracks = FFmpegDecoder.probeSubtitles(url: file)
    }

    private func audioTrackInfos() -> [TrackInfo] {
        audioTracks.map { TrackInfo(index: $0.index, ordinal: $0.ordinal, codec: $0.codec,
                                    language: $0.language, title: $0.title, channels: $0.channels) }
    }

    private func subtitleTrackInfos() -> [TrackInfo] {
        subtitleTracks.map { TrackInfo(index: $0.index, ordinal: $0.ordinal, codec: $0.codec,
                                       language: $0.language, title: $0.title, channels: nil) }
    }

    /// A session that dies before its first frame is a decode failure, not EOF.
    private func reportEmptyDecode(_ s: Session) {
        guard s.decoder.framesRead == 0, !s.failed.value else { return }
        lock.lock()
        let tearingDown = stopFlag || restartFlag
        lock.unlock()
        guard !tearingDown else { return }
        s.decoder.waitForExit(timeout: 0.5)
        let why = s.decoder.exitCode.map { "ffmpeg exit \($0)" } ?? "decoder stopped early"
        s.failed.value = true
        status.update {
            $0.state = .failed
            $0.message = "This file produced no video frames (\(why)). Check that it is readable "
                + "and its codec is supported — files under ~/Movies or ~/Documents may need a "
                + "one-time folder-permission prompt."
        }
    }

    // MARK: - Pool / cache helpers

    private static func makePool(width: Int, height: Int) throws -> CVPixelBufferPool {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess,
              let pool else { throw EngineError.pixelBufferFailure }
        return pool
    }

    private static func makeTextureCache(device: MTLDevice) throws -> CVMetalTextureCache {
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else { throw EngineError.pixelBufferFailure }
        return cache
    }
}

private extension OSAllocatedUnfairLock {
    /// Ergonomic accessor for the session's small counters/flags. Each access
    /// takes the lock; use `withLock` directly for read-modify-write.
    var value: State {
        get { withLock { $0 } }
        nonmutating set { withLock { $0 = newValue } }
    }
}
