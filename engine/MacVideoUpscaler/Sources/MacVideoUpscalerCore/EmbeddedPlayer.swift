import Foundation
import AVFoundation
import AppKit
import CoreMedia

/// In-process playback surface: a video layer and an audio renderer driven by one
/// `AVSampleBufferRenderSynchronizer` clock.
///
/// This replaces the encode → MPEG-TS → HTTP → VLC path. Because the player is
/// inside the app it consumes the upscaled frames directly (no H.264 encode, no
/// stream server, no external instance to launch), and because video and audio
/// are attached to a *shared* render clock, A/V sync is the synchronizer's job
/// rather than an estimate derived from a read-ahead.
public final class EmbeddedPlayer {
    public let synchronizer = AVSampleBufferRenderSynchronizer()
    public let displayLayer = AVSampleBufferDisplayLayer()
    public let audioRenderer: AVSampleBufferAudioRenderer?

    private let view: PlayerView
    private var audioFormat: CMAudioFormatDescription?
    /// Guard for the cached video format below: `makeVideoSample` runs on both
    /// the feed thread and the video thread (paused redraw).
    private let formatLock = NSLock()
    /// Cached video format description (one per output geometry).
    private var videoFormat: CMVideoFormatDescription?
    private var videoFormatWidth = 0
    private var videoFormatHeight = 0

    public init(audioEnabled: Bool, volume: Float = 1.0) {
        let renderer = audioEnabled ? AVSampleBufferAudioRenderer() : nil
        // A muted renderer still consumes samples, so a diagnostic run can check
        // the whole audio path — decoder, timestamps, renderer accepting data —
        // without playing the film out loud.
        renderer?.volume = max(0, min(1, volume))
        self.audioRenderer = renderer
        // The layer stretches the raster into whatever rect the host view gives
        // it, and the UI sizes that rect to the *display* aspect ratio. Together
        // those preserve anamorphic geometry (3:2 raster shown as 16:9) without
        // re-encoding it as HEVC.
        displayLayer.videoGravity = .resize
        displayLayer.backgroundColor = NSColor.black.cgColor
        synchronizer.addRenderer(displayLayer)
        if let renderer { synchronizer.addRenderer(renderer) }
        view = PlayerView(layer: displayLayer)
        audioFormat = Self.makeAudioFormat()
    }

    /// Shape of the video (display aspect ratio, e.g. 16/9). The view letterboxes
    /// the picture into its bounds, so this is what keeps anamorphic sources from
    /// being shown at the wrong proportions.
    ///
    /// Sessions are rebuilt off the main thread and each one sets this, so the
    /// view mutation is hopped back to the main thread.
    public func setDisplayAspect(_ ratio: Double) {
        guard ratio > 0 else { return }
        DispatchQueue.main.async { [weak self] in
            self?.view.displayAspect = CGFloat(ratio)
        }
    }

    /// The layer to show in the UI.
    public func makeView() -> NSView { view }

    /// A copy of the frame the layer is showing, or nil when nothing has been
    /// presented yet (or the OS is too old to report it). This is the *presented*
    /// image, which is what a viewer sees — the frames handed to `enqueueVideo`
    /// are not the same thing, since a queued one may never be reached.
    var displayedFrame: CVPixelBuffer? {
        if #available(macOS 14.4, *) {
            return displayLayer.sampleBufferRenderer.displayedPixelBuffer()
        }
        return nil
    }

    // MARK: - Video

    public var videoReady: Bool { displayLayer.isReadyForMoreMediaData }

    /// True once the display layer has failed. It does not recover on its own, so
    /// a producer waiting for it to become ready must stop waiting rather than
    /// spin forever.
    public var videoFailed: Bool { displayLayer.status == .failed }

    @discardableResult
    public func enqueueVideo(_ sample: CMSampleBuffer) -> Bool {
        guard displayLayer.isReadyForMoreMediaData else { return false }
        displayLayer.enqueue(sample)
        return true
    }

    /// Builds a display-ready sample buffer around an already-rendered pixel
    /// buffer. `pts` is seconds from the start of the current session.
    ///
    /// The format description only depends on the output dimensions, so it is
    /// built once per geometry rather than once per frame. The cache is shared
    /// by the feed thread and the video thread's paused redraw, so it is read
    /// and rebuilt under `formatLock`; the sample itself is built outside it.
    public func makeVideoSample(pixelBuffer: CVPixelBuffer,
                                pts: Double, fps: Double) -> CMSampleBuffer? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let format: CMVideoFormatDescription? = formatLock.withLock {
            if let cached = videoFormat, videoFormatWidth == width, videoFormatHeight == height {
                return cached
            }
            var built: CMVideoFormatDescription?
            guard CMVideoFormatDescriptionCreateForImageBuffer(
                allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer,
                formatDescriptionOut: &built) == noErr, let built else { return nil }
            videoFormat = built
            videoFormatWidth = width
            videoFormatHeight = height
            return built
        }
        guard let format else { return nil }
        let timescale: CMTimeScale = 90000
        var timing = CMSampleTimingInfo(
            duration: CMTime(seconds: 1.0 / max(fps, 1), preferredTimescale: timescale),
            presentationTimeStamp: CMTime(seconds: pts, preferredTimescale: timescale),
            decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer,
            formatDescription: format, sampleTiming: &timing,
            sampleBufferOut: &sample) == noErr else { return nil }
        return sample
    }

    // MARK: - Audio

    public var audioReady: Bool { audioRenderer?.isReadyForMoreMediaData ?? false }

    /// True once the audio renderer has failed (typically the output device went
    /// away). It does not recover on its own, so a producer waiting for it to
    /// become ready must stop waiting rather than spin forever.
    public var audioFailed: Bool { audioRenderer?.status == .failed }

    /// Called when the viewer clicks the picture itself, on the main thread. The
    /// app wires this to play/pause; the engine does not decide what a click means.
    public var onSurfaceClick: (() -> Void)? {
        get { view.onClick }
        set { view.onClick = newValue }
    }

    /// Called when the viewer double clicks the picture, on the main thread.
    public var onSurfaceDoubleClick: (() -> Void)? {
        get { view.onDoubleClick }
        set { view.onDoubleClick = newValue }
    }

    /// Where the comparison divider sits, 0...1, or nil when the comparison is
    /// off. Read on mouse-down to decide whether a drag is moving the divider or
    /// the window.
    public var splitFractionProvider: (() -> Double?)? {
        get { view.splitFractionProvider }
        set { view.splitFractionProvider = newValue }
    }

    /// Called on the main thread as the divider is dragged, with the new fraction.
    public var onSplitDrag: ((Double) -> Void)? {
        get { view.onSplitDrag }
        set { view.onSplitDrag = newValue }
    }

    /// Where the divider is drawn in the surface view's coordinates, or nil when
    /// there is no comparison. Exists so a harness can aim a synthetic drag at the
    /// line, which is otherwise only reachable with a real mouse.
    public var dividerXInView: CGFloat? { view.dividerX }

    @discardableResult
    public func enqueueAudio(_ sample: CMSampleBuffer) -> Bool {
        guard let audioRenderer, audioRenderer.isReadyForMoreMediaData else { return false }
        audioRenderer.enqueue(sample)
        return true
    }

    /// Wraps interleaved float32 stereo PCM in a sample buffer stamped at `pts`
    /// seconds from the start of the current session.
    public func makeAudioSample(pcm: Data, frames: Int, pts: Double) -> CMSampleBuffer? {
        guard let audioFormat, frames > 0 else { return nil }
        let byteCount = frames * FFmpegAudioDecoder.bytesPerFrame
        guard pcm.count >= byteCount else { return nil }
        var block: CMBlockBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: byteCount,
            blockAllocator: kCFAllocatorDefault, customBlockSource: nil,
            offsetToData: 0, dataLength: byteCount, flags: 0,
            blockBufferOut: &block) == noErr, let block else { return nil }
        CMBlockBufferAssureBlockMemory(block)
        let copied = pcm.withUnsafeBytes { raw -> OSStatus in
            guard let base = raw.baseAddress else { return -1 }
            return CMBlockBufferReplaceDataBytes(
                with: base, blockBuffer: block, offsetIntoDestination: 0, dataLength: byteCount)
        }
        guard copied == noErr else { return nil }
        var sampleSize = FFmpegAudioDecoder.bytesPerFrame
        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: FFmpegAudioDecoder.sampleRate),
            presentationTimeStamp: CMTime(seconds: pts, preferredTimescale: FFmpegAudioDecoder.sampleRate),
            decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        guard CMSampleBufferCreateReady(
            allocator: kCFAllocatorDefault, dataBuffer: block, formatDescription: audioFormat,
            sampleCount: frames, sampleTimingEntryCount: 1, sampleTimingArray: &timing,
            sampleSizeEntryCount: 1, sampleSizeArray: &sampleSize,
            sampleBufferOut: &sample) == noErr else { return nil }
        return sample
    }

    private static func makeAudioFormat() -> CMAudioFormatDescription? {
        var asbd = AudioStreamBasicDescription(
            mSampleRate: Double(FFmpegAudioDecoder.sampleRate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(FFmpegAudioDecoder.bytesPerFrame),
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(FFmpegAudioDecoder.bytesPerFrame),
            mChannelsPerFrame: UInt32(FFmpegAudioDecoder.channelCount),
            mBitsPerChannel: 32,
            mReserved: 0)
        var format: CMAudioFormatDescription?
        guard CMAudioFormatDescriptionCreate(
            allocator: kCFAllocatorDefault, asbd: &asbd, layoutSize: 0, layout: nil,
            magicCookieSize: 0, magicCookie: nil, extensions: nil,
            formatDescriptionOut: &format) == noErr else { return nil }
        return format
    }

    // MARK: - Transport

    public var rate: Float { synchronizer.rate }
    public var currentSeconds: Double { synchronizer.currentTime().seconds }
    public var isPlaying: Bool { synchronizer.rate > 0 }

    /// Health of the two renderers. A layer that failed to display (or an audio
    /// renderer the output device rejected) reports it here instead of silently
    /// consuming samples, so "frames advanced" can be told apart from "frames
    /// were shown".
    public var renderStatus: String {
        func name(_ s: AVQueuedSampleBufferRenderingStatus) -> String {
            switch s {
            case .unknown: return "unknown"
            case .rendering: return "rendering"
            case .failed: return "failed"
            @unknown default: return "?"
            }
        }
        var parts = ["video=\(name(displayLayer.status))"]
        parts.append("ready=\(displayLayer.isReadyForMoreMediaData)")
        parts.append("window=\(view.window != nil)")
        if let e = displayLayer.error { parts.append("videoError=\(e.localizedDescription)") }
        if let r = audioRenderer {
            parts.append("audio=\(name(r.status))")
            if let e = r.error { parts.append("audioError=\(e.localizedDescription)") }
        }
        return parts.joined(separator: " ")
    }

    /// Starts (or resumes) playback at `seconds` on the session timeline.
    public func play(fromSeconds seconds: Double) {
        synchronizer.setRate(1.0, time: CMTime(seconds: max(0, seconds), preferredTimescale: 90000))
    }

    public func pause() {
        synchronizer.setRate(0, time: synchronizer.currentTime())
    }

    /// Moves the clock without starting playback: the layer shows the newest
    /// frame whose timestamp has been reached, so a frame stamped *ahead* of the
    /// paused position only appears once the clock is moved onto it.
    public func holdClock(atSeconds seconds: Double) {
        synchronizer.setRate(0, time: CMTime(seconds: max(0, seconds), preferredTimescale: 90000))
    }

    /// Pauses and returns the clock to the start of a fresh session, so the first
    /// enqueued samples (PTS 0, 1/fps, …) are in the future when playback starts.
    public func rewindToStart() {
        synchronizer.setRate(0, time: .zero)
    }

    /// Drops everything queued (used on seek/session restart). Anything already
    /// enqueued belongs to the old position and must not be shown.
    public func flush(removeImage: Bool = true) {
        if removeImage { displayLayer.flushAndRemoveImage() } else { displayLayer.flush() }
        audioRenderer?.flush()
    }

    public func stop() {
        synchronizer.setRate(0, time: .zero)
        flush()
    }
}

/// Hosts the display layer as a sublayer and keeps it fitted to the display
/// aspect ratio inside the view's bounds. Hosting it as a sublayer (rather than
/// as the view's backing layer) is what lets the picture be letterboxed inside a
/// full-window/full-screen surface, and it survives SwiftUI re-parenting.
final class PlayerView: NSView {
    private let displayLayer: AVSampleBufferDisplayLayer

    /// Display aspect ratio (width / height) of the current video.
    var displayAspect: CGFloat = 16.0 / 9.0 {
        didSet {
            guard displayAspect != oldValue else { return }
            needsLayout = true
        }
    }

    /// Invoked on a click in the picture, from the main thread.
    var onClick: (() -> Void)?
    /// Invoked on a double click in the picture, from the main thread.
    var onDoubleClick: (() -> Void)?
    /// Current comparison split (0...1) or nil when not comparing.
    var splitFractionProvider: (() -> Double?)?
    /// Invoked with the new fraction while the divider is dragged.
    var onSplitDrag: ((Double) -> Void)?
    private var dragged = false
    private var draggingDivider = false
    private var pendingClick: DispatchWorkItem?
    /// How close a press has to be to the divider to grab it instead of the
    /// window, in points. Tight enough that it does not steal ordinary
    /// press-anywhere-to-move-the-window, wide enough to hit without aiming.
    static let dividerGrabRadius: CGFloat = 18

    init(layer: AVSampleBufferDisplayLayer) {
        self.displayLayer = layer
        super.init(frame: .zero)
        wantsLayer = true
        self.layer?.backgroundColor = NSColor.black.cgColor
        self.layer?.masksToBounds = true
        // .resize stretches the raster into the (aspect-fitted) frame; combined
        // with the DAR-shaped frame below that is correct for both square-pixel
        // and anamorphic sources.
        layer.videoGravity = .resize
        self.layer?.addSublayer(layer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("not supported") }

    // The picture is both the play/pause control and the window's drag handle —
    // the window has a hidden title bar, so there is no visible bar to drag by.
    // AppKit's background drag would swallow the click, so the view takes the
    // mouse itself: a press that moves drags the window, a press that does not is
    // a click. `mouseDownCanMoveWindow` must be false for the press to arrive
    // here at all, and `acceptsFirstMouse` lets the click that activates an
    // inactive window count too.
    override var mouseDownCanMoveWindow: Bool { false }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func mouseDown(with event: NSEvent) {
        dragged = false
        // Grabbing the divider only makes sense while comparing, and only for a
        // press that lands on it: everywhere else still drags the window.
        draggingDivider = false
        if splitFractionProvider?() != nil,
           let divider = dividerX,
           abs(convert(event.locationInWindow, from: nil).x - divider) <= Self.dividerGrabRadius {
            draggingDivider = true
        }
    }

    override func mouseDragged(with event: NSEvent) {
        if draggingDivider {
            let x = convert(event.locationInWindow, from: nil).x
            let picture = Self.aspectFitRect(aspect: displayAspect, in: bounds)
            guard picture.width > 0 else { return }
            let fraction = Double((x - picture.minX) / picture.width)
            onSplitDrag?(min(max(fraction, 0), 1))
            return
        }
        dragged = true
        window?.performDrag(with: event)
    }

    override func mouseUp(with event: NSEvent) {
        // A divider drag is a drag, not a click: it must not also pause.
        if draggingDivider {
            draggingDivider = false
            dragged = true
        }
        guard !dragged else { return }
        // A single click is held back for the double-click interval, because the
        // first click of a double click arrives as an ordinary click: acting on
        // it immediately would pause the film every time someone went full
        // screen. Every player makes this trade; the window is capped at 300 ms
        // (the usual figure, and the system's own interval is 500 ms) so pausing
        // stays responsive, at the cost of a deliberately slow double click also
        // toggling playback.
        pendingClick?.cancel()
        pendingClick = nil
        if event.clickCount >= 2 {
            onDoubleClick?()
            return
        }
        let item = DispatchWorkItem { [weak self] in self?.onClick?() }
        pendingClick = item
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.singleClickDelay, execute: item)
    }

    /// How long a single click waits to find out whether it was half of a double
    /// click. Never longer than the system's double-click speed, so a faster
    /// setting is honoured.
    static var singleClickDelay: TimeInterval { min(NSEvent.doubleClickInterval, 0.3) }

    /// Where the comparison divider is drawn, in view coordinates, or nil when
    /// there is no comparison. Derived from the same aspect-fit rect the display
    /// layer uses, so the grab zone is where the divider actually appears.
    var dividerX: CGFloat? {
        guard let fraction = splitFractionProvider?() else { return nil }
        let picture = Self.aspectFitRect(aspect: displayAspect, in: bounds)
        return picture.minX + CGFloat(min(max(fraction, 0), 1)) * picture.width
    }

    override func layout() {
        super.layout()
        CATransaction.begin()
        CATransaction.setDisableActions(true)   // no implicit frame animation
        displayLayer.frame = Self.aspectFitRect(aspect: displayAspect, in: bounds)
        CATransaction.commit()
    }

    /// Largest rect with `aspect` that fits in `bounds`, centred.
    static func aspectFitRect(aspect: CGFloat, in bounds: CGRect) -> CGRect {
        guard aspect > 0, bounds.width > 0, bounds.height > 0 else { return bounds }
        var size = bounds.size
        if aspect > bounds.width / bounds.height {
            size.height = bounds.width / aspect
        } else {
            size.width = bounds.height * aspect
        }
        return CGRect(x: bounds.midX - size.width / 2,
                      y: bounds.midY - size.height / 2,
                      width: size.width, height: size.height)
    }
}
