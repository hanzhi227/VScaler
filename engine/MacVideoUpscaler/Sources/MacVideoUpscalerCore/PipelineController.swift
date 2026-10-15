import Foundation
import AVFoundation
import CoreVideo
import Metal
import os

/// Orchestrates: decode (AVAssetReader) -> 2x Metal upscale -> ffmpeg
/// (VideoToolbox H.264 + source audio -> MPEG-TS on stdout) -> bounded byte
/// ring -> local HTTP server for VLC.
///
/// The engine processes FASTER than realtime whenever it can; the bounded ring
/// caps memory and gives VLC a long buffered lead. Pause/seek are coordinated
/// here: seek ends the current session (stream clients are dropped) and starts
/// a new session at the target time — the UI should reopen VLC with the new
/// streamURL (listen to onSessionRestart).
public final class PipelineController {
    public struct Config {
        public var port: UInt16 = 8090
        public var ringCapacityBytes = 128 << 20   // bounded lead: ~45 s at 4K30, ~2.5 min at 960p30
        public var prebufferBytes = 4 << 20        // ~1.5 s at 22 Mb/s before serving
        public var audioEnabled = true
        public var audioStreamIndex = 0      // 0-based index into the file's audio streams
        public var scaleFactor: UpscaleFactor = .x2
        /// How far the producer may run ahead of the player, in seconds of
        /// stream. This is the read-ahead; it bounds memory AND keeps the
        /// production frontier close to what the viewer is watching, so
        /// relative seeks and track/preset restarts act near the current
        /// picture. A large lead used to mean controls could jump minutes ahead.
        public var leadSeconds: Double = 30
        public var burnSubtitles = false     // burn the source's subtitle track into the video
        public var subtitleStreamIndex = 0   // 0-based index into the file's subtitle streams
        public init() {}
    }

    public var config: Config
    public let status = StatusStore()
    /// Called after a seek/session change: UI should reopen the stream URL.
    public var onSessionRestart: ((URL) -> Void)?
    private let ctx: MetalContext
    private let ring: RingBuffer
    private lazy var server = StreamServer(ring: ring, status: status, port: config.port)

    private var worker: Thread?
    /// Owned VLC session; provides the real playback clock (see VLCController).
    private var vlc: VLCController?
    private var pendingSubtitleBurn: SubtitleBurn?
    private var textRenderer: TextSubtitleRenderer?
    private var textRendererFailed = false
    /// Track lists for the current file, probed once (seeks reuse them).
    private var probedFile: URL?
    private var audioTracks: [AudioTrack] = []
    private var subtitleTracks: [SubtitleTrack] = []
    private var stdoutPump: Thread?
    private var decoder: FFmpegDecoder?
    private var muxer: FFmpegMuxer?
    private var scaler: FrameScaler?
    private var outBuffers: OutputBufferRing?
    private var srcTexture: MTLTexture?
    private var stagingBuffer: MTLBuffer?

    private let ctrl = NSCondition()
    private var pausedFlag = false
    private var stopFlag = false
    private var restartFlag = false
    /// Mirror of `restartFlag` readable WITHOUT holding `ctrl`.
    ///
    /// beginSessionLocked() holds `ctrl` while it joins the worker, and the
    /// worker's finishSession() needs to know whether the session ended by
    /// restart. Reading `restartFlag` there would deadlock on `ctrl`, so the
    /// join would always burn its full timeout and every seek would stall for
    /// seconds. This atomic lets finishSession decide without the lock.
    private let restartMirror = OSAllocatedUnfairLock(initialState: false)
    /// Mirror of `stopFlag` readable WITHOUT holding `ctrl`, for the same
    /// reason as `restartMirror`: finishSession runs while beginSessionLocked
    /// holds `ctrl` joining this very worker, so the flag itself is unreadable
    /// there.
    private let stopMirror = OSAllocatedUnfairLock(initialState: false)

    public private(set) var currentFile: URL?
    public private(set) var kind: ScalerKind = ScalerKind.defaultKind
    /// The factor actually in use for the current session; a requested 4x is
    /// clamped to 2x when the output would exceed the H.264 encoder's limit.
    public private(set) var effectiveFactor: UpscaleFactor = .x2
    public private(set) var startAt: Double = 0
    public private(set) var sessionId = 0
    private var frameIndex = 0
    private var lastStatusPost = Date()
    private var lastPostFrames = 0
    private var lastPostTime = Date()
    private var sessionStartTime = Date()

    public init(config: Config = Config()) throws {
        // The muxer's stdin is a pipe, and session teardown closes it from
        // another thread while the worker may be blocked writing to it. That
        // racing write raises SIGPIPE, whose default action would kill the app
        // mid-seek, so ignore it process-wide (writes then fail with EPIPE).
        signal(SIGPIPE, SIG_IGN)
        self.config = config
        self.ctx = try MetalContext()
        self.ring = RingBuffer(capacity: config.ringCapacityBytes, prebufferBytes: config.prebufferBytes)
        try server.start()
        status.update { $0.state = .idle }
    }

    public var streamURL: URL {
        URL(string: "http://127.0.0.1:\(config.port)/stream.ts?session=\(sessionId)")!
    }

    /// Real playback position in *absolute content time*, from the owned VLC's
    /// clock. nil when nothing is playing through us.
    ///
    /// Two corrections are folded in: VLC reports time relative to the stream it
    /// opened (it restarts at 0 for each session), so `startAt` must be added;
    /// and VLC's absolute time carries an unknown base (observed +38 s on a
    /// session starting at 600 s), so we use the elapsed-since-open delta.
    public var playbackSeconds: Double? {
        // `vlc` is read under `ctrl` like every other access to it; the Clock
        // snapshot itself is guarded by VLCController's own lock.
        ctrl.lock()
        let c = vlc?.clock
        let origin = startAt
        ctrl.unlock()
        guard let c, c.valid else { return nil }
        return origin + c.elapsed
    }

    public var isPlayingThroughVLC: Bool { playbackSeconds != nil }

    /// Where a restart should resume from.
    ///
    /// The production frontier leads the viewer by the buffered read-ahead, so
    /// subtracting that lead estimates what the player has actually consumed.
    /// With the lead bounded (config.leadSeconds) this is within seconds of the
    /// picture; it used to be minutes because the ring held minutes of content.
    ///
    /// The VLC clock is not used here: its absolute value carries an unknown base
    /// and, with dummy outputs, can run at input rate rather than real time. Its
    /// elapsed value is still published for diagnostics.
    ///
    /// Reads `frameIndex`/`startAt`/`muxer`/`decoder`, so call with `ctrl` held
    /// (every caller does; `seek(by:)` takes it around the call).
    private func resumePosition() -> Double {
        let lead = Double(ring.bufferedBytes) * 8.0 / Double(max(muxer?.videoBitrateBps ?? 1, 1))
        let estimate = startAt + Double(frameIndex) / max(decoder?.meta.fps ?? 30, 1) - lead
        return max(startAt, estimate)
    }

    // MARK: - Controls

    public func open(file: URL, kind: ScalerKind, startAt: Double = 0) throws {
        ctrl.lock()
        defer { ctrl.unlock() }
        self.currentFile = file
        self.kind = kind
        beginSessionLocked(startAt: startAt)
    }

    /// Enable/disable subtitle burn-in. The muxer's filter graph changes, so the
    /// session is restarted at the current position (client reconnects to the new URL).
    public func setSubtitles(_ enabled: Bool, streamIndex: Int? = nil) {
        ctrl.lock()
        let changed = config.burnSubtitles != enabled ||
                      (streamIndex != nil && config.subtitleStreamIndex != streamIndex!)
        // Persist the new state before restarting: without this assignment the
        // restart re-read the OLD value, so toggling did nothing (enabling left
        // subtitles off, disabling left them on).
        config.burnSubtitles = enabled
        if let streamIndex { config.subtitleStreamIndex = streamIndex }
        // Resume where the viewer is, not where production has got to.
        let pos = resumePosition()
        ctrl.unlock()
        guard changed, currentFile != nil else { return }
        seek(to: pos + 0.5)
    }

    /// Select which of the file's audio tracks is muxed (0 = first).
    /// The muxer's input mapping changes, so the session restarts at the current
    /// position and the client reconnects to the new stream URL.
    public func setAudioTrack(_ ordinal: Int) {
        ctrl.lock()
        let changed = config.audioStreamIndex != max(0, ordinal)
        config.audioStreamIndex = max(0, ordinal)
        // Resume where the viewer is, not where production has got to.
        let pos = resumePosition()
        ctrl.unlock()
        guard changed, currentFile != nil else { return }
        seek(to: pos + 0.5)
    }

    /// Select which of the file's subtitle tracks is burned in (0 = first),
    /// enabling burn-in if it was off.
    public func setSubtitleTrack(_ ordinal: Int) {
        setSubtitles(true, streamIndex: max(0, ordinal))
    }

    /// Current track lists (also mirrored into the status snapshot for the UI).
    public var availableAudioTracks: [TrackInfo] { ctrl.lock(); defer { ctrl.unlock() }; return audioTrackInfos() }
    public var availableSubtitleTracks: [TrackInfo] { ctrl.lock(); defer { ctrl.unlock() }; return subtitleTrackInfos() }

    private func audioTrackInfos() -> [TrackInfo] {
        audioTracks.map { TrackInfo(index: $0.index, ordinal: $0.ordinal, codec: $0.codec,
                                    language: $0.language, title: $0.title, channels: $0.channels) }
    }
    private func subtitleTrackInfos() -> [TrackInfo] {
        subtitleTracks.map { TrackInfo(index: $0.index, ordinal: $0.ordinal, codec: $0.codec,
                                       language: $0.language, title: $0.title, channels: nil) }
    }

    /// Probes audio/subtitle streams once per opened file. Call with ctrl locked.
    private func refreshTrackListsLocked() {
        guard probedFile != currentFile else { return }
        probedFile = currentFile
        audioTracks = []
        subtitleTracks = []
        guard let file = currentFile else { return }
        audioTracks = FFmpegDecoder.probeAudio(url: file)
        subtitleTracks = FFmpegDecoder.probeSubtitles(url: file)
    }

    /// Hot-swaps the scaler when the output geometry is unchanged.
    ///
    /// Two cases are not a hot swap, and both come from modes that do not magnify
    /// by the session's factor. A mode pinned to another factor (Real-ESRGAN is x4
    /// only) changes the output dimensions even though both kinds magnify, so it
    /// restarts like a scale change — the old `magnifies` test called that a hot
    /// swap and the rebuild then failed once per frame with "scaler unavailable".
    /// A mode that cannot run on this source at all is refused outright, so the
    /// playing stream is left alone and the reason is reported.
    public func setKind(_ kind: ScalerKind) {
        ctrl.lock()
        let w = decoder?.meta.width ?? 0, h = decoder?.meta.height ?? 0
        let resolved = w > 0
            ? ScalerFactory.resolveFactor(kind: kind, requested: config.scaleFactor,
                                          width: w, height: h)
            : config.scaleFactor
        if w > 0, resolved == nil {
            status.update {
                $0.message = ScalerFactory.pinnedFactorRefusal(kind: kind, width: w, height: h)
            }
            ctrl.unlock()
            return
        }
        // Crossing to or from `off` changes the output geometry (1:1 against the
        // factor) even when the resolved factor is unchanged, so it is a restart
        // like a scale change; the player's setKind applies the same rule.
        let factorChanged = resolved != effectiveFactor || self.kind.magnifies != kind.magnifies
        self.kind = kind
        self.scaler = nil
        status.update { $0.mode = kind.displayName }
        // Resume where the viewer is, not where production has got to.
        let pos = factorChanged ? resumePosition() : 0
        ctrl.unlock()
        if factorChanged { seek(to: pos + 0.5) }
    }

    /// Change the magnification. Output dimensions change, so the session
    /// restarts at the current position and the client reconnects (the GUI
    /// reopens VLC, as with a seek).
    public func setScaleFactor(_ factor: UpscaleFactor) {
        ctrl.lock()
        let changed = config.scaleFactor != factor
        config.scaleFactor = factor
        // Resume where the viewer is, not where production has got to.
        let pos = resumePosition()
        ctrl.unlock()
        guard changed, currentFile != nil else { return }
        seek(to: pos + 0.5)
    }

    /// Starts the owned VLC session on the current stream (the "Open in VLC" action).
    @discardableResult
    public func startPlayback() -> Bool {
        guard currentFile != nil else { return false }
        // The reference is created/reused under `ctrl`, but play() runs outside
        // it: VLCController calls issue HTTP commands and can block.
        ctrl.lock()
        let session = vlc ?? VLCController()
        vlc = session
        ctrl.unlock()
        guard let session else { return false }
        session.play(streamURL)
        status.update { $0.vlcHttpPort = Int(session.httpPort) }
        return true
    }

    public func pause() {
        // Coordinate with the player: pausing only the frame worker left VLC
        // playing out the queued ring, which is not what "pause" means to a viewer.
        // The reference is read under `ctrl`; pause() itself runs outside it.
        ctrl.lock()
        let session = vlc
        ctrl.unlock()
        session?.pause()
        ctrl.lock()
        pausedFlag = true
        ctrl.unlock()
        status.update { $0.state = .paused }
    }

    public func resume() {
        ctrl.lock()
        pausedFlag = false
        ctrl.signal()
        let session = vlc
        ctrl.unlock()
        session?.resume()
        status.update { $0.state = .processing }
    }

    public var isPaused: Bool {
        ctrl.lock(); defer { ctrl.unlock() }
        return pausedFlag
    }

    /// Restart the pipeline at an absolute time. Drops stream clients; emits
    /// onSessionRestart with the fresh URL.
    public func seek(to seconds: Double) {
        ctrl.lock()
        let dur = decoder?.meta.durationSeconds ?? 0
        let target = min(max(0, seconds), max(0, dur - 1))
        startAt = target
        restartFlag = true
        restartMirror.withLock { $0 = true }
        pausedFlag = false
        ctrl.broadcast()
        ctrl.unlock()

        // beginSessionLocked performs the teardown (ring reset + muxer kill),
        // which is what unblocks a worker parked on full-ring backpressure; do
        // not join the worker here or that wait just times out first.
        ctrl.lock()
        beginSessionLocked(startAt: target)
        let session = vlc
        ctrl.unlock()
        // Reopen the player on the new session instead of asking the user to do
        // it by hand; the old URL is gone (410). play() runs outside `ctrl`.
        session?.play(streamURL)
        onSessionRestart?(streamURL)
    }

    public func seek(by delta: Double) {
        // Relative to what the viewer is watching, not to the production
        // frontier (which can be minutes ahead). resumePosition() reads
        // frameIndex/muxer/decoder, so it runs under `ctrl` like its other
        // callers; seek(to:) takes the lock itself.
        ctrl.lock()
        let pos = resumePosition()
        ctrl.unlock()
        seek(to: pos + delta)
    }

    public func stop() {
        // Nil the reference under `ctrl` so no other thread picks it up, then
        // stop the controller outside the lock (it waits on the VLC process).
        ctrl.lock()
        let session = vlc
        vlc = nil
        stopFlag = true
        stopMirror.withLock { $0 = true }
        restartFlag = false
        restartMirror.withLock { $0 = false }
        pausedFlag = false
        ctrl.broadcast()
        ctrl.unlock()
        session?.stop()
        // Same backpressure problem as a seek: unblock the parked worker/pump
        // before joining, or stop() blocks for the join timeout. The decoder's
        // read(2) is one of those parking spots, so it is stopped before the
        // join as well.
        ring.reset()
        muxer?.terminate()
        muxer = nil
        decoder?.stop()
        decoder = nil
        worker?.join(timeout: 3)
        worker = nil
        stdoutPump?.join(timeout: 2)
        stdoutPump = nil
        status.update { $0.state = .idle }
    }

    // MARK: - Session lifecycle (call with ctrl locked)

    private func beginSessionLocked(startAt newStart: Double) {
        // Teardown order matters when the previous session is under backpressure
        // — the normal playback state, where the ring is full, so the stdout
        // pump is parked in ring.write() and the worker is parked writing to the
        // muxer's stdin. Both only wake if the ring is reset AND the muxer is
        // killed, so do that BEFORE joining: otherwise each join below waits out
        // its full timeout and a single seek takes tens of seconds.
        //
        // The worker's third parking spot is the decoder's blocking read(2),
        // released only by decoder.stop(), so the decoder is stopped before the
        // join too (the same order as the player's session teardown). Joining
        // first let a worker that outlasted the join timeout wake up after the
        // flag resets below and run finishSession() against the NEW session,
        // tearing the new muxer/decoder down.
        //
        // reset() also drops any leftover TS bytes from the old session, which
        // would otherwise be flushed into the new one (stale/garbled start).
        ring.reset()          // unblocks a pump parked in ring.write()
        let timing = ProcessInfo.processInfo.environment["ZU_DEBUG_TIMING"] == "1"
        let tTeardown = Date()
        func mark(_ what: String) {
            guard timing else { return }
            let dt = Date().timeIntervalSince(tTeardown)
            FileHandle.standardError.write(
                Data(String(format: "[timing] %@ %.3fs\n", what, dt).utf8))
        }
        muxer?.terminate()    // closes stdin + kills ffmpeg -> unblocks the worker
        muxer = nil
        mark("muxer.terminate")
        decoder?.stop()       // releases a worker parked in the decoder's read(2)
        decoder = nil
        mark("decoder.stop")

        ctrl.broadcast()      // wake a paused worker so it sees restart/stop
        worker?.join(timeout: 3)
        worker = nil
        mark("worker.join")
        stdoutPump?.join(timeout: 2)
        stdoutPump = nil
        mark("pump.join")
        outBuffers = nil
        srcTexture = nil
        stagingBuffer = nil

        restartFlag = false
        restartMirror.withLock { $0 = false }
        stopFlag = false
        stopMirror.withLock { $0 = false }
        pausedFlag = false
        frameIndex = 0
        sessionStartTime = Date()
        sessionId += 1
        startAt = max(0, newStart)

        server.dropClients()
        ring.reopen()   // fresh, empty ring for the new session

        guard let file = currentFile else { return }

        refreshTrackListsLocked()

        // Resolve the subtitle burn before starting: it is applied at decode time.
        var subtitleLabel = "off"
        pendingSubtitleBurn = nil
        textRenderer = nil
        textRendererFailed = false
        if config.burnSubtitles {
            let picked: SubtitleTrack? = subtitleTracks.first { $0.ordinal == config.subtitleStreamIndex }
                ?? subtitleTracks.first
            if let track = picked {
                let probe = try? FFmpegDecoder(url: file)
                let sw0 = probe?.meta.width ?? 0, sh0 = probe?.meta.height ?? 0
                if track.bitmap {
                    pendingSubtitleBurn = SubtitleBurn(track: track, sourceWidth: sw0, sourceHeight: sh0)
                    subtitleLabel = "\(track.codec) #\(track.index) (bitmap)"
                } else {
                    // native CoreText rendering (no libass in this ffmpeg build)
                    textRenderer = TextSubtitleRenderer.load(url: file, ordinal: track.ordinal)
                    subtitleLabel = textRenderer == nil
                        ? "\(track.codec) #\(track.index): no cues found"
                        : "\(track.codec) #\(track.index) (native text render)"
                    if textRenderer == nil { textRendererFailed = true }
                }
                config.subtitleStreamIndex = track.ordinal
            } else {
                subtitleLabel = "none found in file"
            }
        }

        // Fall back to the first audio stream if the requested one is absent.
        if !audioTracks.contains(where: { $0.ordinal == config.audioStreamIndex }) {
            config.audioStreamIndex = 0
        }

        let dec: FFmpegDecoder
        do {
            dec = try FFmpegDecoder(url: file)
            try dec.start(at: startAt, subtitle: self.pendingSubtitleBurn)
        } catch {
            status.update { $0.state = .failed; $0.message = "\(error)" }
            return
        }
        decoder = dec

        let sw = dec.meta.width, sh = dec.meta.height

        // Clamp the requested factor to what the hardware encoder can take
        // (a 4x pass on HD would exceed 4096x2304 and the encoder would fail).
        // A mode pinned to one factor (Real-ESRGAN is x4 only) either gets that
        // factor or does not run — clamping it would silently change the
        // magnification rather than the quality.
        let reqFactor = config.scaleFactor
        guard let eff = ScalerFactory.resolveFactor(kind: kind, requested: reqFactor,
                                                    width: sw, height: sh) else {
            status.update {
                $0.state = .failed
                $0.message = ScalerFactory.pinnedFactorRefusal(kind: kind, width: sw, height: sh)
            }
            return
        }
        effectiveFactor = eff
        var scaleNote = ""
        if !kind.magnifies {
            scaleNote = "Off — playing at the source's own \(sw)x\(sh); the "
                + "magnification setting does not apply."
        } else if let pinned = kind.pinnedFactor, pinned != reqFactor {
            scaleNote = "\(kind.displayName) is \(pinned.displayName) only; using \(pinned.displayName)."
        } else if eff != reqFactor {
            scaleNote = "\(reqFactor.displayName) needs \(sw * reqFactor.rawValue)×\(sh * reqFactor.rawValue), "
                + "beyond the H.264 encoder's 4096×2304 limit — using \(eff.displayName)."
        }

        do {
            scaler = try ScalerFactory.make(kind: kind, factor: eff, ctx: ctx)
        } catch {
            status.update { $0.state = .failed; $0.message = "scaler: \(error)" }
            return
        }

        stagingBuffer = ctx.device.makeBuffer(length: sw * sh * 4, options: .storageModeShared)
        let srcDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: sw, height: sh, mipmapped: false)
        srcDesc.usage = [.shaderRead]
        // Shared storage: replaceRegion() is then a direct memcpy. A managed
        // texture allocates a CPU->GPU staging buffer on every update, which
        // showed up as steady-state RSS growth during long playback.
        srcDesc.storageMode = .shared
        srcTexture = ctx.device.makeTexture(descriptor: srcDesc)

        // `off` is the one kind that does not magnify: its output is the source's
        // own geometry (CopyScaler is 1:1 and rejects anything else), so the
        // resolved factor is not applied. Same rule as the player's session build.
        let ow = kind.magnifies ? sw * eff.rawValue : sw
        let oh = kind.magnifies ? sh * eff.rawValue : sh
        do {
            outBuffers = try OutputBufferRing(device: ctx.device, width: ow, height: oh, depth: 3)
        } catch {
            status.update { $0.state = .failed; $0.message = "\(error)" }
            return
        }

        do {
            muxer = try FFmpegMuxer(
                sourceURL: file, audioStartSeconds: startAt,
                width: ow, height: oh, fps: dec.meta.fps,   // rawvideo = 2x output size
                audioEnabled: config.audioEnabled,
                audioOrdinal: config.audioStreamIndex,
                displayAspect: dec.meta.displayAspect)
        } catch {
            status.update { $0.state = .failed; $0.message = "\(error)" }
            return
        }

        let bitrate = muxer?.videoBitrateBps ?? 0
        // Bound the read-ahead in seconds (as well as by the ring's byte capacity).
        ring.setSoftCap(bytes: Int(config.leadSeconds * Double(max(bitrate, 1)) / 8.0))
        let codec = muxer?.videoCodec ?? ""
        // An anamorphic source forces the encoder switch, which is worth saying
        // out loud: it is the only reason geometry is preserved.
        if let dar = dec.meta.displayAspect {
            let note = "Anamorphic source (display \(dar)): encoding with HEVC, because "
                + "hardware H.264 in MPEG-TS cannot carry a display aspect ratio."
            scaleNote = scaleNote.isEmpty ? note : scaleNote + " " + note
        }
        let audioInfos = audioTrackInfos()
        let subInfos = subtitleTrackInfos()
        let audioOrd = config.audioStreamIndex
        let subOrd = config.subtitleStreamIndex
        let subsOn = config.burnSubtitles
        status.update {
            $0.state = .preparing
            $0.file = file.lastPathComponent
            $0.mode = kind.displayName
            $0.sessionId = sessionId
            $0.sourceWidth = sw; $0.sourceHeight = sh
            $0.outputWidth = ow; $0.outputHeight = oh
            $0.durationSeconds = dec.meta.durationSeconds
            $0.encodeBitrateBps = bitrate
            $0.videoCodec = codec
            $0.framesProcessed = 0
            $0.processFps = 0
            $0.realtimeRatio = 0
            $0.positionSeconds = startAt   // so the timeline snaps to the seek target
            $0.clients = 0
            $0.subtitles = subtitleLabel
            $0.subtitlesEnabled = subsOn
            $0.audioTracks = audioInfos
            $0.subtitleTracks = subInfos
            $0.audioTrackOrdinal = audioOrd
            $0.subtitleTrackOrdinal = subOrd
            $0.scaleFactor = eff.rawValue
            $0.requestedScaleFactor = reqFactor.rawValue
            $0.message = scaleNote
        }

        startStdoutPump()
        let t = Thread { [weak self] in self?.workerLoop() }
        t.name = "engine-worker"
        t.qualityOfService = .userInitiated
        worker = t
        t.start()
    }

    /// Env-gated stderr diagnostics for the stream pump/worker (ZU_DEBUG_PUMP=1).
    /// Used to trace cases where the session ends without bytes reaching the ring.
    fileprivate static let pumpDebug =
        ProcessInfo.processInfo.environment["ZU_DEBUG_PUMP"] == "1"

    private func dlog(_ s: String) {
        guard Self.pumpDebug else { return }
        FileHandle.standardError.write(Data("[pump] \(s)\n".utf8))
    }

    /// ffmpeg stdout (MPEG-TS) -> bounded ring. Marks EOS only if it belongs
    /// to the current session (a killed muxer's EOF must not end a new one).
    private func startStdoutPump() {
        guard let muxer else { return }
        let sid = sessionId
        let t = Thread { [weak self] in
            guard let self, let fh = muxer.stdoutHandle else { return }
            let chunk = UnsafeMutablePointer<UInt8>.allocate(capacity: 256 << 10)
            defer { chunk.deallocate() }
            var total = 0
            self.dlog("pump start session=\(sid)")
            while true {
                // Each iteration allocates an autoreleased Data from
                // availableData(); this bare thread has no run loop to drain
                // them, so without a per-iteration pool they accumulate for the
                // life of the session (steady RSS growth during long playback).
                let keepGoing = autoreleasepool { () -> Bool in
                    if self.sessionId != sid { self.dlog("pump superseded"); return false }
                    let data = fh.availableData  // blocks until data or EOF
                    if data.isEmpty { self.dlog("pump EOF after \(total) bytes"); return false }
                    let n = data.count
                    total += n
                    let ok = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Bool in
                        guard let base = raw.baseAddress else { return false }
                        return self.ring.write(base.assumingMemoryBound(to: UInt8.self), length: n)
                    }
                    if !ok { self.dlog("pump ring write refused"); return false }
                    return true
                }
                if !keepGoing { break }
            }
            if self.sessionId == sid {
                self.dlog("pump markEOS total=\(total)")
                self.ring.markEOS()
            }
        }
        t.name = "stdout-pump"
        stdoutPump = t
        t.start()
    }

    // MARK: - Worker loop

    private func workerLoop() {
        // Each frame allocates Metal/CoreVideo temporaries; on a bare Thread
        // there is no run loop to drain autoreleased objects, so without a
        // per-frame pool RSS creeps upward indefinitely.
        while autoreleasepool(invoking: { processOneFrame() }) {
            // keep going until processOneFrame returns false
        }
        finishSession()
    }

    /// Processes exactly one frame. Returns false when the session ends.
    private func processOneFrame() -> Bool {
        do {
            ctrl.lock()
            while pausedFlag && !stopFlag && !restartFlag { ctrl.wait() }
            if stopFlag || restartFlag {
                ctrl.unlock()
                return false
            }
            let dec = decoder
            let mx = muxer
            // Resolve the scaler under the same lock that setKind()/beginSession
            // use, and keep a strong reference for the whole frame. Previously
            // the worker read `self.scaler` unlocked, so a mode switch clearing it
            // mid-frame made `scaler?.scale` a no-op — silently skipping a frame.
            var sc = scaler
            if sc == nil {
                sc = try? ScalerFactory.make(kind: kind, factor: effectiveFactor, ctx: ctx)
                scaler = sc
            }
            let modeLabel = kind.displayName
            ctrl.unlock()

            guard let dec, let mx else { return false }
            guard let scaler = sc else {
                status.update { $0.state = .failed; $0.message = "scaler unavailable" }
                return false
            }
            if mx.hasExited {
                dlog("worker: muxer exited early")
                status.update { $0.state = .failed; $0.message = "ffmpeg exited early" }
                return false
            }
            if status.current.mode != modeLabel {
                status.update { $0.mode = modeLabel }
            }

            // 1) Read one decoded BGRA frame straight into the staging buffer
            //    and upload it to the GPU texture (blocks at the live frontier).
            guard let staging = stagingBuffer, let srcTex = srcTexture else { return false }
            guard dec.readFrame(into: staging.contents()) else {  // EOF, or the decoder failed
                dlog("worker: readFrame false after \(dec.framesRead) frames "
                     + "(decoder exit=\(String(describing: dec.exitCode)))")
                reportEmptyDecode(dec)
                return false
            }

            // Burn text subtitles into the source-resolution frame (before upscale).
            if let renderer = textRenderer {
                let t = startAt + Double(frameIndex) / max(dec.meta.fps, 1)
                renderer.blend(into: staging.contents(), bytesPerRow: dec.meta.width * 4,
                               width: dec.meta.width, height: dec.meta.height, time: t)
            }
            let w = dec.meta.width * 4
            let region = MTLRegionMake2D(0, 0, srcTex.width, srcTex.height)
            srcTex.replace(region: region, mipmapLevel: 0, withBytes: staging.contents(), bytesPerRow: w)

            // 2) 2x output buffer from the rotation (already a writable texture).
            guard let outBuffers else { return false }
            let (outPB, outTex) = outBuffers.next()

            // 3) Scale (synchronous GPU work).
            do {
                try scaler.scale(src: srcTex, dst: outTex)
            } catch {
                status.update { $0.state = .failed; $0.message = "scale: \(error)" }
                return false
            }

            // 4) GPU result -> ffmpeg stdin (blocks while encoder is behind).
            var wrote = false
            CVPixelBufferLockBaseAddress(outPB, .readOnly)
            if let base = CVPixelBufferGetBaseAddress(outPB) {
                let bpr = CVPixelBufferGetBytesPerRow(outPB)
                let w = CVPixelBufferGetWidth(outPB) * 4
                let h = CVPixelBufferGetHeight(outPB)
                do {
                    if bpr == w {
                        try mx.writeFrame(base.assumingMemoryBound(to: UInt8.self), length: w * h)
                    } else {
                        for row in 0..<h {
                            try mx.writeFrame((base + row * bpr).assumingMemoryBound(to: UInt8.self), length: w)
                        }
                    }
                    wrote = true
                } catch {
                    status.update { $0.message = "muxer write failed" }
                }
            }
            CVPixelBufferUnlockBaseAddress(outPB, .readOnly)
            if !wrote { return false }

            // 5) Stats.
            // frameIndex is read under `ctrl` (resumePosition), so the
            // increment takes it too; released before postStatusIfNeeded().
            ctrl.lock()
            frameIndex += 1
            ctrl.unlock()
            postStatusIfNeeded()
            return true
        }
    }

    /// A session that ends without a single frame is a decode failure, not a
    /// clean EOF — an unreadable or truncated file, or macOS withholding folder
    /// access (files under ~/Movies and ~/Documents need one-time consent).
    /// Without this the UI would just show "finished" with 0 frames and no
    /// reason. ffmpeg exits 0 on some truncated inputs, so this does not rely
    /// on the exit code; it is only reported when available.
    private func reportEmptyDecode(_ dec: FFmpegDecoder) {
        guard dec.framesRead == 0 else { return }
        ctrl.lock()
        let tearingDown = stopFlag || restartFlag
        ctrl.unlock()
        guard !tearingDown else { return }
        dec.waitForExit(timeout: 0.5)
        let why = dec.exitCode.map { "ffmpeg exit \($0)" } ?? "decoder stopped early"
        status.update {
            $0.state = .failed
            $0.message = "This file produced no video frames (\(why)). Check that it is "
                + "readable and its codec is supported — files under ~/Movies or "
                + "~/Documents may need a one-time folder-permission prompt."
        }
    }

    /// Session teardown after processOneFrame returns false.
    private func finishSession() {
        // Both read through their atomics, NOT `ctrl`: beginSessionLocked holds
        // `ctrl` while joining this worker, so a locked read here would deadlock
        // and make every seek wait out the join timeout. See `restartMirror`.
        let endedByRestart = restartMirror.withLock { $0 }
        if endedByRestart || stopMirror.withLock({ $0 }) {
            // Seek or stop path: whatever ended the session tears everything down
            // and posts the state itself. Returning keeps a worker that outlived
            // stop()'s join from flipping the just-posted .idle back to .finished.
            return
        }
        // EOF or stop: flush the encoder so the TS tail lands in the ring.
        muxer?.finish()
        // Let the pump drain that tail into the ring BEFORE reporting, so the
        // final counters are authoritative. Progress is otherwise sampled every
        // 0.4 s, which is why a finished short clip used to report frames=3 of
        // 24 and buffer=0 while the ring actually held the whole stream.
        stdoutPump?.join(timeout: 3)
        stdoutPump = nil
        decoder?.stop()
        decoder = nil
        let totalFrames = frameIndex
        let buffered = ring.bufferedBytes
        let footprint = Self.memoryFootprintMB()
        status.update {
            if $0.state != .failed { $0.state = .finished }
            $0.framesProcessed = totalFrames
            $0.bufferBytes = buffered
            $0.memoryFootprintMB = footprint
        }
    }

    /// Resident footprint of this process (phys_footprint), in MB. This is the
    /// metric that matters for the memory question; `leaks` and RSS alone do not
    /// cover Metal/driver allocations.
    static func memoryFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { ip in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), ip, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / (1024 * 1024)
    }

    private func postStatusIfNeeded() {
        let now = Date()
        guard now.timeIntervalSince(lastStatusPost) > 0.4 else { return }
        lastStatusPost = now
        // Sustained throughput since the session began, in frames/second of
        // wall time. An interval or instantaneous rate is misleading here: with
        // the read-ahead bounded the engine works in bursts (fill the buffer,
        // then block on backpressure), so short-window rates read 8-17x while
        // the actual sustained rate is ~1x. This is the number that answers
        // "can it keep up?", and the one the UI warning uses.
        let avgFps = Double(frameIndex) / max(now.timeIntervalSince(sessionStartTime), 0.5)
        lastPostFrames = frameIndex
        lastPostTime = now
        let fpsSrc = max(decoder?.meta.fps ?? 30, 1)
        let idx = frameIndex
        let pos = startAt + Double(idx) / fpsSrc
        let thermKey: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermKey = "nominal"
        case .fair: thermKey = "fair"
        case .serious: thermKey = "serious"
        case .critical: thermKey = "critical"
        @unknown default: thermKey = "nominal"
        }
        // The player's position estimated from the bounded read-ahead: content
        // sent to the player is the frontier minus whatever is still buffered.
        // Accurate to seconds now that the lead is bounded; the VLC elapsed value
        // is carried alongside for diagnostics only.
        let buffered = ring.bufferedBytes
        let leadSeconds = Double(buffered) * 8.0 / Double(max(muxer?.videoBitrateBps ?? 1, 1))
        // `vlc` is read under `ctrl` like every other access to it; the clock
        // and isRunning reads themselves are the controller's own locked state.
        ctrl.lock()
        let session = vlc
        ctrl.unlock()
        let vlcClock = session?.clock
        let hasClient = status.current.clients > 0 || (session?.isRunning ?? false)
        let playPos = max(startAt, pos - leadSeconds)
        status.update {
            $0.state = .processing
            $0.framesProcessed = idx
            $0.processFps = avgFps
            $0.realtimeRatio = avgFps / fpsSrc
            $0.positionSeconds = pos
            $0.playbackSeconds = playPos
            $0.playbackActive = hasClient
            $0.vlcReportedSeconds = vlcClock?.elapsed ?? 0
            $0.bufferBytes = buffered
            $0.ringCapacityBytes = config.ringCapacityBytes
            $0.prebufferSeconds = Double(buffered) * 8.0 / Double(max(muxer?.videoBitrateBps ?? 1, 1))
            $0.thermalState = thermKey
            $0.memoryFootprintMB = Self.memoryFootprintMB()
        }
    }

}

extension Thread {
    func join(timeout: TimeInterval) {
        let deadline = Date().addingTimeInterval(timeout)
        while isFinished == false && isExecuting && Date() < deadline {
            usleep(20_000)
        }
    }
}
