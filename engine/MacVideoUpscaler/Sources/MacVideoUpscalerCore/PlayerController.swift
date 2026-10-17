import Foundation
import AVFoundation
import AppKit
import CoreVideo
import CoreMedia
import Metal
import os

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
        /// Compress movie dynamics and lift quiet passages during embedded playback.
        public var volumeLeveling = false
        public var scaleFactor: UpscaleFactor = .x2
        public var burnSubtitles = false
        public var subtitleStreamIndex = 0
        public var subtitleStyle = SubtitleStyle()
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
    ///
    /// Reads and writes take `lock` for the access alone; the hook itself is
    /// invoked outside it, so it can be assigned from any thread and take its
    /// own time and locks without holding up the threads that share the
    /// condition.
    public var onOutputFrame: ((CVPixelBuffer, Int) -> Void)? {
        get { lock.withLock { onOutputFrameHook } }
        set { lock.withLock { onOutputFrameHook = newValue } }
    }
    /// Backing store for `onOutputFrame`. Guarded by `lock`.
    private var onOutputFrameHook: ((CVPixelBuffer, Int) -> Void)?

    private let ctx: MetalContext
    private let player: EmbeddedPlayer
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
    private struct ExternalSubtitle {
        let url: URL
        let cues: [TextSubtitleRenderer.Cue]?
        let track: SubtitleTrack?
    }
    private var subtitleFiles: [ExternalSubtitle] = []
    /// True when the running decoder is burning a bitmap subtitle the overlay
    /// cannot draw (PGS, DVB, XSUB). DVD and text cues never take this path.
    private var decoderBurnsBitmap = false
    private struct PictureGeometry {
        var outputWidth = 0
        var outputHeight = 0
        var unrotatedWidth = 0
        var unrotatedHeight = 0
        var rotation = 0
    }
    private var geometry = PictureGeometry()
    private let overlay = OverlayModel()
    private var bitmapCache: [String: BitmapSubtitleSequence] = [:]
    private var overlayTimer: DispatchSourceTimer?

    /// Cues currently drawn above the picture. The video thread never reads this.
    private final class OverlayModel {
        private let lock = NSLock()
        private var text: TextSubtitleRenderer?
        private var bitmap: BitmapSubtitleSequence?
        private var style = SubtitleStyle()
        private var outputWidth = 0
        private var outputHeight = 0
        private var enabled = false

        func replace(cues: [TextSubtitleRenderer.Cue]?, bitmap: BitmapSubtitleSequence?,
                     style: SubtitleStyle, outputWidth: Int, outputHeight: Int, enabled: Bool) {
            lock.lock(); defer { lock.unlock() }
            self.style = style.normalized
            self.outputWidth = outputWidth
            self.outputHeight = outputHeight
            self.enabled = enabled
            self.bitmap = bitmap
            if let cues, outputWidth > 0, outputHeight > 0 {
                text = TextSubtitleRenderer(cues: cues, sourceWidth: outputWidth,
                                            sourceHeight: outputHeight, style: style.normalized)
            } else {
                text = nil
            }
        }

        func setStyle(_ style: SubtitleStyle) {
            lock.lock(); defer { lock.unlock() }
            let normalized = style.normalized
            guard self.style != normalized else { return }
            self.style = normalized
            if let cues = text?.cues, outputWidth > 0 {
                text = TextSubtitleRenderer(cues: cues, sourceWidth: outputWidth,
                                            sourceHeight: outputHeight, style: normalized)
            }
        }

        func clear() {
            lock.lock(); defer { lock.unlock() }
            text = nil
            bitmap = nil
            enabled = false
        }

        func frame(at time: Double) -> SubtitleOverlayFrame? {
            lock.lock(); defer { lock.unlock() }
            guard enabled, outputWidth > 0, outputHeight > 0 else { return nil }
            if let text, let placed = text.placement(at: time) {
                let index = text.activeCueIndex(at: time) ?? -1
                return SubtitleOverlayFrame(image: placed.image, rect: placed.rect,
                                            videoWidth: outputWidth, videoHeight: outputHeight,
                                            key: "t\(index) \(outputWidth)x\(outputHeight) \(Self.key(style))")
            }
            if let bitmap, let placed = bitmap.placement(at: time, style: style,
                                                         outputWidth: outputWidth, outputHeight: outputHeight) {
                let index = bitmap.eventIndex(at: time) ?? -1
                return SubtitleOverlayFrame(image: placed.image, rect: placed.rect,
                                            videoWidth: outputWidth, videoHeight: outputHeight,
                                            key: "b\(index) \(outputWidth)x\(outputHeight) \(Self.key(style))")
            }
            return nil
        }

        private static func key(_ style: SubtitleStyle) -> String {
            let style = style.normalized
            return "\(style.typeface.rawValue) \(style.size) \(style.tint.rawValue) \(style.outline.rawValue) \(style.horizontalOffset) \(style.verticalOffset)"
        }
    }

    public init(config: Config = Config()) throws {
        signal(SIGPIPE, SIG_IGN)
        self.config = config
        let context = try MetalContext()
        self.ctx = context
        self.player = EmbeddedPlayer(audioEnabled: config.audioEnabled, volume: config.volume)
        sessionQueue.setSpecific(key: Self.sessionQueueKey, value: ())
        status.update { $0.state = .idle }
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + 0.1, repeating: 0.08)
        timer.setEventHandler { [weak self] in self?.refreshOverlay() }
        timer.resume()
        overlayTimer = timer
        // Dragging the divider on the picture changes the same split the slider
        // does; both end up in `setCompareSplit`.
        player.splitFractionProvider = { [weak self] in
            guard let self, self.compareFlag.value else { return nil }
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
    /// presented yet. Reports the full enhancement image, before the view's wipe mask.
    public var displayedFrame: CVPixelBuffer? { player.displayedFrame }

    /// The picture the viewer sees, including the subtitle layer. Playback does
    /// not bake the cue into the frame; this draws it on a copy for snapshots.
    public func compositedPicture(base: CVPixelBuffer? = nil) -> CGImage? {
        let clock: (Double, Int, Int) = {
            lock.lock(); defer { lock.unlock() }
            return (playbackSecondsFromLocked(), geometry.outputWidth, geometry.outputHeight)
        }()
        let frame = overlay.frame(at: clock.0)
        let buffer = base ?? displayedFrame
        let width = buffer.map { CVPixelBufferGetWidth($0) } ?? clock.1
        let height = buffer.map { CVPixelBufferGetHeight($0) } ?? clock.2
        return FrameCapture.composite(base: buffer, overlay: frame, width: width, height: height)
    }

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
        /// Rendered frames waiting to be shown: the producer fills it, the feeder
        /// drains it into the display layer. See `FrameStore`.
        let store: FrameStore
        /// Fill level the feeder (re)starts playback at. Reached either after the
        /// initial wait or after a rebuffer, and deliberately the same number both
        /// times so the first wait and every later one feel alike.
        let resumeFrames: Int
        /// The next paired frame the feeder owes the two display layers.
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
        let comparisonFailure = OSAllocatedUnfairLock<String?>(initialState: nil)
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
             decoder: FFmpegDecoder, audio: FFmpegAudioDecoder?,
             srcTexture: MTLTexture, staging: MTLBuffer, pool: CVPixelBufferPool,
             aux: CFDictionary, textureCache: CVMetalTextureCache,
             scaler: FrameScaler, kind: ScalerKind, factor: UpscaleFactor,
             baseMessage: String,
             store: FrameStore, resumeFrames: Int) {
            self.id = id
            self.startAt = startAt
            self.fps = fps
            self.outputWidth = outputWidth
            self.outputHeight = outputHeight
            self.decoder = decoder
            self.audio = audio
            self.srcTexture = srcTexture
            self.staging = staging
            self.pool = pool
            self.aux = aux
            self.textureCache = textureCache
            self.scaler = scaler
            self.kind = OSAllocatedUnfairLock(initialState: kind)
            self.factor = factor
            self.baseMessage = baseMessage
            self.store = store
            self.resumeFrames = resumeFrames
        }
    }

    public func open(file: URL, kind: ScalerKind, startAt: Double = 0) throws {
        lock.lock()
        if currentFile != file {
            bitmapCache.removeAll()
            if !subtitleFiles.isEmpty {
                if config.subtitleStreamIndex >= subtitleTracks.count {
                    config.burnSubtitles = false
                    config.subtitleStreamIndex = 0
                }
                subtitleFiles = []
            }
        }
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
    /// `snapLongGaps` is for a scrub to an arbitrary time. A relative skip or a
    /// restart that must resume exactly where playback was leaves it false, so
    /// those still land on the requested frame.
    private func restart(at target: Double, snapLongGaps: Bool = false) {
        lock.lock()
        seekGeneration += 1        // supersedes a skip that is still debouncing
        let hadSkip = pendingSkipDelta != nil
        pendingSkipDelta = nil
        lock.unlock()
        if hadSkip { status.update { $0.pendingSkipSeconds = 0 } }
        sessionQueue.async { [weak self] in
            self?.performRestart(at: target, snapLongGaps: snapLongGaps)
        }
    }

    private func performRestart(at target: Double, snapLongGaps: Bool = false) {
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
        overlay.clear()
        player.setSubtitleOverlay(nil)

        guard let file else { return }

        lock.lock()
        guard !stopFlag else { lock.unlock(); return }
        // Clamp only once the duration is known: on the first open it is still 0,
        // and clamping against it would silently force every start offset to 0.
        let upper = durationSeconds > 0 ? max(0, durationSeconds - 0.5) : .greatestFiniteMagnitude
        let clamped = min(max(0, target), upper)
        beginSessionLocked(file: file, startAt: clamped, snapLongGaps: snapLongGaps)
        let fresh = session
        let consumedSkip = pendingSeek == target
        if consumedSkip {
            pendingSeek = nil
            pendingSkipDelta = nil
        }
        lock.unlock()
        if consumedSkip { status.update { $0.pendingSkipSeconds = 0 } }
        refreshOverlay()
        if let fresh { primeAndStart(fresh) }
    }

    /// Builds a session and starts its producers. Call with `lock` held.
    private func beginSessionLocked(file: URL, startAt newStart: Double, snapLongGaps: Bool = false) {
        restartFlag = false
        pausedFlag = false
        stopFlag = false
        startAt = max(0, newStart)

        refreshTrackListsLocked()

        if !audioTracks.contains(where: { $0.ordinal == config.audioStreamIndex }) {
            config.audioStreamIndex = 0
        }
        let hasAudio = config.audioEnabled && !audioTracks.isEmpty

        let dec: FFmpegDecoder
        var landingNotice: String?
        var subtitleLabel = "off"
        var subtitleCues: [TextSubtitleRenderer.Cue]?
        var subtitleBitmap: BitmapSubtitleSequence?
        do {
            dec = try FFmpegDecoder(url: file)
            let quarter = dec.meta.rotation == 90 || dec.meta.rotation == 270
            geometry.unrotatedWidth = quarter ? dec.meta.height : dec.meta.width
            geometry.unrotatedHeight = quarter ? dec.meta.width : dec.meta.height
            geometry.rotation = dec.meta.rotation
            // Text and DVD cues are drawn in a layer. Only bitmap codecs this
            // player cannot decode still go through the ffmpeg overlay, and
            // that is the only reason the decoder is told about subtitles.
            var built = buildOverlayLocked(file: file)
            subtitleLabel = built.label
            subtitleCues = built.cues
            subtitleBitmap = built.bitmap
            decoderBurnsBitmap = built.burn != nil
            let position = config.subtitleStyle.normalized
            if var burn = built.burn {
                burn.horizontalOffset = position.horizontalOffset
                burn.verticalOffset = position.verticalOffset
                built.burn = burn
            }
            // A scrub across a long GOP must not accurate-decode every frame back
            // to the previous keyframe. The clock then starts where the picture
            // does, so audio and the timeline follow the frame that is shown.
            if snapLongGaps {
                let landing = dec.landing(at: startAt)
                startAt = landing.origin
                landingNotice = landing.notice
                try dec.start(at: startAt, accurate: landing.accurate, subtitle: built.burn)
            } else {
                try dec.start(at: startAt, subtitle: built.burn)
            }
        } catch {
            status.update { $0.state = .failed; $0.message = "\(error)"; $0.seekNotice = "" }
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
                $0.seekNotice = ""
            }
            return
        }
        effectiveFactor = eff
        // `off` is the one mode that does not magnify, so there is no factor to
        // apply and nothing to clamp: the output is the source's own geometry and
        // the scaler is a 1:1 copy.
        let ow = kind.magnifies ? sw * eff.rawValue : sw
        let oh = kind.magnifies ? sh * eff.rawValue : sh
        geometry.outputWidth = ow
        geometry.outputHeight = oh
        overlay.replace(cues: subtitleCues, bitmap: subtitleBitmap,
                        style: config.subtitleStyle, outputWidth: ow, outputHeight: oh,
                        enabled: config.burnSubtitles && (subtitleBitmap != nil || subtitleCues != nil))

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
                try a.start(url: file, at: startAt, ordinal: config.audioStreamIndex,
                            volumeLeveling: config.volumeLeveling)
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
        let frameBytes = ow * oh * 4 * 2 // two full frames for the presentation-time wipe
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
        let depth = (capacity + 4) * 2
        let aux: [String: Any] = [kCVPixelBufferPoolAllocationThresholdKey as String: depth]

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
                        srcTexture: srcTex, staging: staging,
                        pool: pool, aux: aux as CFDictionary, textureCache: cache,
                        scaler: scaler, kind: kind, factor: eff, baseMessage: baseMessage,
                        store: store, resumeFrames: resumeFrames)
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
            $0.volumeLeveling = config.volumeLeveling
            $0.subtitleTrackOrdinal = config.subtitleStreamIndex
            $0.scaleFactor = eff.rawValue
            $0.requestedScaleFactor = req.rawValue
            $0.videoCodec = "bgra (no encode)"
            $0.encodeBitrateBps = 0
            $0.playbackEngine = "embedded"
            $0.message = baseMessage
            $0.seekNotice = landingNotice ?? ""
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
        lock.lock(); defer { lock.unlock() }
        return requestedPositionLocked()
    }

    public func seek(to seconds: Double) {
        let target = max(0, seconds)
        lock.lock()
        pendingSeek = target
        lock.unlock()
        restart(at: target, snapLongGaps: true)
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
            return
        }
        if geometryChanges {
            restart(at: pos + 0.1)
            return
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
        lock.lock()
        let s = session
        let currentKind = s?.kind.value ?? kind
        lock.unlock()
        let refusal = s?.comparisonFailure.value
            ?? (!currentKind.magnifies ? "Comparison needs an enhancement mode that scales." : nil)
            ?? (!compareLeft.value.magnifies ? "The left comparison model must scale the video." : nil)
        if on, let refusal {
            status.update { $0.message = refusal }
            return
        }
        compareFlag.value = on
        status.update {
            $0.comparing = on
            $0.compareLeftMode = compareLeft.value.displayName
            $0.compareSplit = compareSplit.value
        }
        updateComparisonPresentation()
    }

    public func setCompareLeft(_ kind: ScalerKind) {
        guard compareLeft.value != kind else { return }
        compareLeft.value = kind
        status.update { $0.compareLeftMode = kind.displayName }
        // A different model needs new pixels; moving the wipe never does.
        lock.lock()
        let position = comparisonPositionLocked()
        let hasSession = session != nil
        lock.unlock()
        if hasSession { restart(at: position) }
    }

    public func setCompareSplit(_ fraction: Double) {
        guard fraction.isFinite else { return }
        let clamped = min(max(fraction, 0), 1)
        compareSplit.value = clamped
        status.update { $0.compareSplit = clamped }
        updateComparisonPresentation()
    }

    private func updateComparisonPresentation() {
        player.setComparison(enabled: compareFlag.value, fraction: compareSplit.value)
    }

    private func viewSettingChanged() {
        lock.lock()
        let parked = pausedFlag && pauseRequested
        let position = comparisonPositionLocked()
        lock.unlock()
        if parked { restart(at: position) }
    }

    private func comparisonPositionLocked() -> Double {
        let position = requestedPositionLocked()
        guard pauseRequested, let s = session else { return position }
        let frame = floor(max(0, position - s.startAt) * max(s.fps, 1))
        return s.startAt + frame / max(s.fps, 1)
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

    public func setVolumeLeveling(_ enabled: Bool) {
        lock.lock()
        let changed = config.volumeLeveling != enabled
        config.volumeLeveling = enabled
        let pos = requestedPositionLocked()
        let needsRestart = changed && session != nil
        if needsRestart { pendingSeek = pos }
        lock.unlock()
        if needsRestart { restart(at: pos) }
    }

    public func setSubtitles(_ enabled: Bool, streamIndex: Int? = nil) {
        lock.lock()
        let changed = config.burnSubtitles != enabled
            || (streamIndex != nil && config.subtitleStreamIndex != streamIndex!)
        config.burnSubtitles = enabled
        if let streamIndex { config.subtitleStreamIndex = streamIndex }
        let burns = changed && (selectionBurnsBitmapLocked() || decoderBurnsBitmap)
        let pos = requestedPositionLocked()
        if burns { pendingSeek = pos + 0.1 }
        lock.unlock()
        guard changed else { return }
        if burns {
            restart(at: pos + 0.1)
        } else {
            sessionQueue.async { [weak self] in self?.reloadOverlay() }
        }
    }

    public func setSubtitleStyle(_ style: SubtitleStyle) {
        lock.lock()
        let normalized = style.normalized
        let changed = config.subtitleStyle != normalized
        let moved = config.subtitleStyle.horizontalOffset != normalized.horizontalOffset
            || config.subtitleStyle.verticalOffset != normalized.verticalOffset
        config.subtitleStyle = normalized
        let burns = changed && moved && decoderBurnsBitmap && config.burnSubtitles
        let pos = comparisonPositionLocked()
        if burns { pendingSeek = pos }
        lock.unlock()
        guard changed else { return }
        if burns {
            restart(at: pos)
            return
        }
        overlay.setStyle(normalized)
        refreshOverlay()
    }

    public func setSubtitleTrack(_ ordinal: Int) { setSubtitles(true, streamIndex: max(0, ordinal)) }

    /// Text and DVD cues swap in place. A codec that still has to be burned in
    /// restarts, because that changes the decoder's filter graph.
    private func reloadOverlay() {
        lock.lock()
        guard let file = currentFile else {
            overlay.clear()
            decoderBurnsBitmap = false
            lock.unlock()
            status.update { $0.subtitles = "off"; $0.subtitlesEnabled = false }
            refreshOverlay()
            return
        }
        let built = buildOverlayLocked(file: file)
        if built.burn != nil {
            lock.unlock()
            return
        }
        decoderBurnsBitmap = false
        let style = config.subtitleStyle
        let enabled = config.burnSubtitles
        let ordinal = config.subtitleStreamIndex
        let tracks = subtitleTrackInfos()
        let output = (geometry.outputWidth, geometry.outputHeight)
        lock.unlock()
        overlay.replace(cues: built.cues, bitmap: built.bitmap, style: style,
                        outputWidth: output.0, outputHeight: output.1,
                        enabled: enabled && (built.cues != nil || built.bitmap != nil))
        status.update {
            $0.subtitles = built.label
            $0.subtitlesEnabled = enabled
            $0.subtitleTracks = tracks
            $0.subtitleTrackOrdinal = ordinal
        }
        refreshOverlay()
    }

    private func refreshOverlay() {
        let time: Double = {
            lock.lock(); defer { lock.unlock() }
            return playbackSecondsFromLocked()
        }()
        player.setSubtitleOverlay(overlay.frame(at: time))
    }

    private struct BuiltOverlay {
        var label = "off"
        var cues: [TextSubtitleRenderer.Cue]?
        var bitmap: BitmapSubtitleSequence?
        var burn: SubtitleBurn?
    }

    /// Call with `lock` held.
    private func selectionBurnsBitmapLocked() -> Bool {
        guard config.burnSubtitles else { return false }
        let externalIndex = config.subtitleStreamIndex - subtitleTracks.count
        if subtitleFiles.indices.contains(externalIndex) {
            guard let track = subtitleFiles[externalIndex].track else { return false }
            return track.bitmap && track.codec != "dvd_subtitle"
        }
        guard let track = subtitleTracks.first(where: { $0.ordinal == config.subtitleStreamIndex })
                ?? subtitleTracks.first else { return false }
        return track.bitmap && track.codec != "dvd_subtitle"
    }

    /// Call with `lock` held. DVD bitmaps are decoded once and cached; text cues
    /// are parsed here. Other bitmap codecs are left for the decoder to burn in.
    private func buildOverlayLocked(file: URL) -> BuiltOverlay {
        var built = BuiltOverlay()
        guard config.burnSubtitles else { return built }
        let externalIndex = config.subtitleStreamIndex - subtitleTracks.count
        if subtitleFiles.indices.contains(externalIndex) {
            let subtitleFile = subtitleFiles[externalIndex]
            built.label = subtitleFile.url.lastPathComponent
            if let cues = subtitleFile.cues {
                built.cues = cues
            } else if let track = subtitleFile.track {
                if track.codec == "dvd_subtitle" {
                    built.bitmap = cachedBitmapLocked(url: subtitleFile.url, ordinal: track.ordinal)
                    if built.bitmap == nil { built.label += " (no cues found)" }
                } else if track.bitmap {
                    built.burn = SubtitleBurn(track: track, sourceWidth: geometry.unrotatedWidth,
                                              sourceHeight: geometry.unrotatedHeight,
                                              externalURL: subtitleFile.url)
                }
            }
            return built
        }
        guard let track = subtitleTracks.first(where: { $0.ordinal == config.subtitleStreamIndex })
                ?? subtitleTracks.first else {
            built.label = "none found in file"
            return built
        }
        config.subtitleStreamIndex = track.ordinal
        if track.bitmap && track.codec != "dvd_subtitle" {
            built.burn = SubtitleBurn(track: track, sourceWidth: geometry.unrotatedWidth,
                                      sourceHeight: geometry.unrotatedHeight)
            built.label = "\(track.codec) #\(track.index) (bitmap)"
        } else if track.bitmap {
            built.bitmap = cachedBitmapLocked(url: file, ordinal: track.ordinal)
            built.label = built.bitmap == nil
                ? "\(track.codec) #\(track.index): no cues found"
                : "\(track.codec) #\(track.index) (bitmap)"
        } else {
            built.cues = TextSubtitleRenderer.loadCues(url: file, ordinal: track.ordinal)
            built.label = built.cues == nil
                ? "\(track.codec) #\(track.index): no cues found"
                : "\(track.codec) #\(track.index) (overlay)"
        }
        return built
    }

    private func cachedBitmapLocked(url: URL, ordinal: Int) -> BitmapSubtitleSequence? {
        let key = "\(url.path)#\(ordinal)#\(geometry.unrotatedWidth)x\(geometry.unrotatedHeight)#\(geometry.rotation)"
        if let cached = bitmapCache[key] { return cached }
        guard let sequence = BitmapSubtitleSequence.load(
            url: url, ordinal: ordinal, unrotatedWidth: geometry.unrotatedWidth,
            unrotatedHeight: geometry.unrotatedHeight, rotation: geometry.rotation) else { return nil }
        bitmapCache[key] = sequence
        return sequence
    }

    /// A failed selection leaves the current subtitles intact, including multi-file loads.
    public func loadSubtitleFile(_ url: URL, for video: URL) throws {
        try loadSubtitleFiles([url], for: video)
    }

    public func loadSubtitleFiles(_ urls: [URL], for video: URL) throws {
        guard !urls.isEmpty else { return }
        var loaded: [ExternalSubtitle] = []
        var seen = Set<URL>()
        for url in urls {
            let files = try Self.readSubtitleFile(url)
            guard let source = files.first?.url, seen.insert(source).inserted else { continue }
            loaded.append(contentsOf: files)
        }
        lock.lock()
        guard currentFile == video else {
            lock.unlock()
            throw CocoaError(.fileReadUnknown, userInfo: [NSLocalizedDescriptionKey:
                "The video changed while loading subtitles. Select the subtitle files again."])
        }
        refreshTrackListsLocked()
        for source in seen { subtitleFiles.removeAll { $0.url == source } }
        let firstIndex = subtitleFiles.count
        subtitleFiles.append(contentsOf: loaded)
        config.burnSubtitles = true
        config.subtitleStreamIndex = subtitleTracks.count + firstIndex
        let burns = selectionBurnsBitmapLocked() || decoderBurnsBitmap
        let pos = requestedPositionLocked()
        if burns { pendingSeek = pos }
        lock.unlock()
        if burns {
            restart(at: pos)
        } else {
            sessionQueue.async { [weak self] in self?.reloadOverlay() }
        }
    }

    private static func readSubtitleFile(_ url: URL) throws -> [ExternalSubtitle] {
        let ext = url.pathExtension.lowercased()
        guard url.isFileURL, ["srt", "vtt", "ass", "ssa", "idx", "sub"].contains(ext) else {
            throw CocoaError(.fileReadUnsupportedScheme)
        }
        if ["idx", "sub"].contains(ext) {
            let folder = url.deletingLastPathComponent()
            let stem = url.deletingPathExtension().lastPathComponent
            let siblings = (try? FileManager.default.contentsOfDirectory(at: folder,
                includingPropertiesForKeys: nil)) ?? []
            func companion(_ ext: String) -> URL? {
                let matches = siblings.filter {
                    $0.pathExtension.lowercased() == ext
                        && $0.deletingPathExtension().lastPathComponent.caseInsensitiveCompare(stem) == .orderedSame
                        && FileManager.default.isReadableFile(atPath: $0.path)
                }
                return matches.first { $0.deletingPathExtension().lastPathComponent == stem } ?? matches.first
            }
            guard let source = companion("idx"), companion("sub") != nil else {
                throw CocoaError(.fileReadNoSuchFile, userInfo: [NSLocalizedDescriptionKey:
                    "VobSub needs both \(stem).idx and \(stem).sub in the same folder. Select either file or both once they are present."])
            }
            let tracks = FFmpegDecoder.probeSubtitles(url: source).filter(\.bitmap)
            if !tracks.isEmpty {
                return tracks.map { ExternalSubtitle(url: source.standardizedFileURL, cues: nil, track: $0) }
            }
        } else if let cues = TextSubtitleRenderer.loadCues(url: url) {
            return [ExternalSubtitle(url: url.standardizedFileURL, cues: cues, track: nil)]
        }
        throw CocoaError(.fileReadCorruptFile, userInfo: [NSLocalizedDescriptionKey:
            "Could not read subtitles from \(url.lastPathComponent). Choose a text subtitle file or a matching VobSub .idx/.sub pair."])
    }

    private static func discoverSubtitleFiles(for video: URL) -> [ExternalSubtitle] {
        let folder = video.deletingLastPathComponent()
        guard let enumerator = FileManager.default.enumerator(at: folder,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return [] }
        var candidates: [URL] = []
        for case let url as URL in enumerator {
            guard ["srt", "vtt", "ass", "ssa", "idx"].contains(url.pathExtension.lowercased()),
                  let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]),
                  values.isRegularFile == true, values.isSymbolicLink != true else { continue }
            candidates.append(url)
        }
        let stem = video.deletingPathExtension().lastPathComponent.lowercased()
        candidates.sort {
            let leftMatch = $0.deletingPathExtension().lastPathComponent.lowercased().hasPrefix(stem)
            let rightMatch = $1.deletingPathExtension().lastPathComponent.lowercased().hasPrefix(stem)
            if leftMatch != rightMatch { return leftMatch }
            return $0.path.localizedStandardCompare($1.path) == .orderedAscending
        }
        return candidates.flatMap { (try? readSubtitleFile($0)) ?? [] }
    }

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
            // Raise the flag before queueing: a restart in flight runs its
            // up-to-10-second prebuffer wait on the session queue while holding
            // `controlMutex`, and this call's `sync` cannot start until the
            // restart leaves the queue either way. The wait polls `stopFlag`
            // every few milliseconds, so setting it here ends the wait at once
            // instead of stalling quit for the full ten seconds. `performStop`
            // sets the flag again once it actually runs.
            lock.lock()
            stopFlag = true
            lock.broadcast()
            lock.unlock()
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
        overlay.clear()
        player.setSubtitleOverlay(nil)
        status.update { $0.state = .idle; $0.playbackActive = false }
    }

    deinit {
        overlayTimer?.cancel()
        stop()
    }

    // MARK: - Producers

    private struct Output {
        let pb: CVPixelBuffer
        let cvTexture: CVMetalTexture
        let texture: MTLTexture
    }

    private func videoLoop(_ s: Session) {
        while true {
            if parkIfNeededOrEnded(s) { break }

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
            let region = MTLRegionMake2D(0, 0, s.srcTexture.width, s.srcTexture.height)
            let source = s.srcTexture
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

            // Keep the reference as a full frame, even while the wipe is hidden,
            // so enabling comparison on a paused frame needs no decoder work.
            var left: Output?
            do {
                left = try renderComparison(to: s, source: source)
            } catch {
                disableComparison(s, message: "Comparison view stopped: \(error)")
            }

            let idx = s.frameIndex.value
            // The getter only snapshots the hook under the lock; the call runs
            // outside it.
            if let hook = onOutputFrame { hook(out.pb, idx) }
            // The scale above was synchronous, so the GPU is done with the
            // texture. The buffer goes into the store rather than straight to the
            // layer: the feeder owns the handoff, and this frame may sit here for
            // as long as the buffer is deep.
            if !s.store.push(FrameStore.Frame(index: idx, pb: out.pb,
                                              cvTexture: out.cvTexture, texture: out.texture,
                                              leftPB: left?.pb, leftCVTexture: left?.cvTexture)) {
                // Impossible unless the store closed or the wait above was raced
                // by a close; dropping the frame would leave a hole the feeder
                // stalls on for the rest of the session.
                break
            }
            s.frameIndex.value = idx + 1
            // "Primed" now means the buffer holds enough to start, not that one
            // frame was queued — `primeAndStart` waits on it.
            if s.store.count >= s.resumeFrames { s.videoPrimed.value = true }
            postStatus(s)
        }
        // The end of the source, or a failure. Completion is reported by the
        // *feeder*, once it has drained what is still buffered — with a buffer of
        // five seconds, saying "finished" the moment the decoder runs out would
        // declare the film over while seconds of it are still queued to play.
        s.producerDone.value = true
        // Near the end of a file the buffer can hold fewer frames than the
        // feeder's refill watermark, which can now never fill: wake it so it
        // plays out what is buffered and ends the session rather than waiting
        // on the watermark forever.
        s.store.finishProducing()
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
            if parkFeederIfNeededOrEnded(s) { break }
            if s.store.isClosed { break }
            let next = s.nextToFeed.value

            // Only feed a clock that is running. The layer's queue is two frames,
            // so handing it anything while the clock is stopped (before playback,
            // or held for a refill) fills that queue and blocks this thread — and
            // it would drain the store into the layer, which is the opposite of
            // what holding the clock is for.
            if !s.started.value || !player.isPlaying {
                if s.bufferingNow.value {
                    // Held for a refill: wait for the watermark without touching
                    // the store, then start the clock again. False with the store
                    // still open means the producer reached the end of the source
                    // short of the watermark, so the wait can never succeed: play
                    // out what is buffered, and the end-of-stream check below
                    // ends the session once it is drained.
                    if !s.store.waitForCount(s.resumeFrames) && s.store.isClosed { break }
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
            guard let leftSample = player.makeVideoSample(pixelBuffer: frame.leftPB ?? frame.pb,
                                                          pts: Double(next) / max(s.fps, 1),
                                                          fps: s.fps) else { break }
            if player.enqueueVideo(sample, comparison: leftSample) {
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

    /// Builds the scaler for a mode the viewer picked and swaps it in, or reports
    /// why it could not, keeping the current model on a failed hot swap.
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

    /// Render a second full image on the same grid, never a baked-in split.
    private func renderComparison(to s: Session, source: MTLTexture) throws -> Output? {
        guard s.comparisonFailure.value == nil else { return nil }
        guard s.kind.value.magnifies, compareLeft.value.magnifies else {
            if compareFlag.value {
                disableComparison(s, message: "Comparison needs two modes that scale the video.")
            }
            return nil
        }
        let wanted = compareLeft.value
        if wanted == s.kind.value { return nil }

        // Rebuilding the left scaler is a per-frame comparison against a stored
        // value: it only changes when the viewer picks another model.
        if s.leftScaler == nil || s.leftKind != wanted {
            guard let fresh = try? ScalerFactory.make(kind: wanted, factor: s.factor, ctx: ctx) else {
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
                disableComparison(s, message: why)
                return nil
            }
            s.leftScaler = fresh
            s.leftKind = wanted
            status.update { $0.compareLeftMode = wanted.displayName }
        }
        guard let scaler = s.leftScaler, let left = acquireOutput(s, completingPair: true) else { return nil }
        try scaler.scale(src: source, dst: left.texture)
        return left
    }

    private func disableComparison(_ s: Session, message: String) {
        s.comparisonFailure.value = message
        compareFlag.value = false
        updateComparisonPresentation()
        status.update { $0.comparing = false; $0.message = message }
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
    private func acquireOutput(_ s: Session, completingPair: Bool = false) -> Output? {
        var stalled = 0
        while true {
            lock.lock()
            let ended = isEndedLocked(s.id)
            let paused = pausedFlag
            lock.unlock()
            if ended || (paused && !completingPair) { return nil }
            if !s.store.isClosed {
                var pb: CVPixelBuffer?
                let st = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
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
        subtitleFiles = Self.discoverSubtitleFiles(for: file)
    }

    private func audioTrackInfos() -> [TrackInfo] {
        audioTracks.map { TrackInfo(index: $0.index, ordinal: $0.ordinal, codec: $0.codec,
                                    language: $0.language, title: $0.title, channels: $0.channels) }
    }

    private func subtitleTrackInfos() -> [TrackInfo] {
        var tracks = subtitleTracks.map { TrackInfo(index: $0.index, ordinal: $0.ordinal, codec: $0.codec,
                                       language: $0.language, title: $0.title, channels: nil) }
        let root = currentFile?.deletingLastPathComponent().standardizedFileURL.path.appending("/")
        for (index, file) in subtitleFiles.enumerated() {
            let path = file.url.standardizedFileURL.path
            var title = root.flatMap { path.hasPrefix($0) ? String(path.dropFirst($0.count)) : nil }
                ?? file.url.lastPathComponent
            if let track = file.track {
                title += " · \(track.language ?? "Track") #\(track.ordinal + 1)"
            }
            tracks.append(TrackInfo(index: -(index + 1), ordinal: tracks.count,
                                    codec: file.track?.codec ?? "external",
                                    language: file.track?.language, title: title))
        }
        return tracks
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
