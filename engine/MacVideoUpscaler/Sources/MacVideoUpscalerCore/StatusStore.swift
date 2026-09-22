import Foundation

public enum PipelineState: String, Codable {
    case idle, preparing, processing, paused, finished, failed
}

/// One selectable stream in the source file, published so the UI can list it.
public struct TrackInfo: Codable, Hashable, Identifiable {
    public var index: Int        // absolute stream index in the file (shown to the user)
    public var ordinal: Int      // 0-based position among streams of this type (used to select)
    public var codec: String
    public var language: String?
    public var title: String?
    public var channels: Int?

    public var id: Int { index }

    public init(index: Int, ordinal: Int, codec: String,
                language: String? = nil, title: String? = nil, channels: Int? = nil) {
        self.index = index
        self.ordinal = ordinal
        self.codec = codec
        self.language = language
        self.title = title
        self.channels = channels
    }

    /// e.g. "#2 · jpn · aac stereo · Stereo"
    public var label: String {
        var parts = ["#\(index)"]
        if let language, !language.isEmpty, language != "und" { parts.append(language) }
        var c = codec
        switch channels {
        case .some(1): c += " mono"
        case .some(2): c += " stereo"
        case .some(let n): c += " \(n)ch"
        case .none: break
        }
        parts.append(c)
        if let title, !title.isEmpty { parts.append(title) }
        return parts.joined(separator: " · ")
    }
}

public struct StatusSnapshot: Codable {
    public var state: PipelineState = .idle
    public var file: String = ""
    public var mode: String = ""
    public var sessionId: Int = 0
    public var framesProcessed = 0
    public var processFps: Double = 0          // processing throughput (fps, EMA)
    public var realtimeRatio: Double = 0       // processFps / source fps
    public var positionSeconds: Double = 0
    /// Real playback position, read from the owned VLC instance's clock.
    /// Meaningful only when `playbackActive` is true; this is the position the
    /// viewer is at, as opposed to `positionSeconds` (the production frontier).
    public var playbackSeconds: Double = 0
    public var playbackActive = false
    /// VLC's own elapsed playback time, for diagnostics. Not used for control:
    /// its absolute value carries an unknown base and headless VLC can run it at
    /// input rate rather than real time.
    public var vlcReportedSeconds: Double = 0
    /// Port the owned VLC's HTTP interface actually bound (0 = none). Tools read
    /// this instead of assuming the default, which another instance may hold.
    public var vlcHttpPort = 0
    public var durationSeconds: Double = 0
    /// End of the last video frame handed to the renderer. The feeder can finish
    /// before that frame is displayed, so a playlist must wait for this clock time.
    public var presentationEndSeconds: Double = 0
    public var bufferBytes = 0
    public var ringCapacityBytes = 0
    public var prebufferSeconds: Double = 0    // ring occupancy in seconds at the stream bitrate
    /// Embedded-player read-ahead, in frames. This is the `FrameStore`'s capacity,
    /// i.e. the whole buffer: the producer renders into it and the display layer is
    /// fed from it just in time, so `bufferMilliseconds` is the longest a mode can
    /// run behind real time and still play smoothly.
    public var bufferFrames = 0
    public var bufferMilliseconds: Double = 0
    /// How much of that buffer is filled right now, and how much video it holds.
    /// `bufferAheadSeconds` is the runway: at a steady state the buffer drains at
    /// (1 - realtimeRatio) per second, so this is how long playback lasts before
    /// the next wait.
    public var bufferAheadFrames = 0
    public var bufferAheadSeconds: Double = 0
    public var bufferAheadBytes = 0
    /// The configured target for `bufferMilliseconds`, so the UI can say how full
    /// the buffer is against what it is trying to be.
    public var bufferTargetSeconds: Double = 0
    /// True while playback is held because the buffer ran dry and is refilling —
    /// the equivalent of a streaming player's spinner. `state` stays `.processing`
    /// throughout: the viewer still wants to play, this is a transient stall.
    public var buffering = false
    /// Skips the viewer has pressed that are still waiting on the debounce, as a
    /// signed total (+30 = three "forward 10" presses). Zero when nothing is
    /// pending; shown on screen so a burst feels acknowledged while it waits.
    public var pendingSkipSeconds: Double = 0
    /// Cumulative audio sample frames handed to the renderer. Freezing while
    /// `playbackSeconds` advances is the signature of a starved audio path, which
    /// is otherwise invisible (silence looks the same as a working output).
    public var audioFramesEnqueued: Int64 = 0
    /// Side-by-side view: the left half of the picture is the model named by
    /// `compareLeftMode` (`.bypass` = the original video, plainly upscaled), the
    /// right half is the enhancement. 0 = enhancement only, 1 = left model only.
    public var comparing = false
    public var compareLeftMode: String = ""
    public var compareSplit: Double = 0.5
    /// "embedded" (in-app render clock) or "stream" (MPEG-TS to an external player).
    public var playbackEngine: String = ""
    /// Renderer health, e.g. "video=rendering audio=rendering". A failed display
    /// or audio renderer surfaces here rather than silently dropping samples.
    public var renderStatus: String = ""
    public var sourceWidth = 0, sourceHeight = 0
    public var outputWidth = 0, outputHeight = 0
    /// Shape the video surface should be laid out at (display aspect ratio, not
    /// the raster's) so anamorphic sources are shown stretched, not pillarboxed.
    public var displayAspectRatio: Double = 16.0 / 9.0
    public var encodeBitrateBps = 0
    public var videoCodec: String = ""        // e.g. h264_videotoolbox / hevc_videotoolbox
    public var clients = 0
    public var thermalState: String = "nominal"
    public var subtitles: String = "off"
    public var subtitlesEnabled = false
    /// Streams available in the current file (empty until a file is opened).
    public var audioTracks: [TrackInfo] = []
    public var subtitleTracks: [TrackInfo] = []
    public var audioTrackOrdinal = 0
    public var subtitleTrackOrdinal = 0
    /// Effective magnification (2 or 4) and what was requested before clamping.
    public var scaleFactor = 2
    public var requestedScaleFactor = 2
    public var memoryFootprintMB: Double = 0
    public var message: String = ""

    public init() {}
}

/// Thread-safe status holder with change callbacks (UI + /status.json).
public final class StatusStore {
    private var snap = StatusSnapshot()
    private let lock = NSLock()
    public var on_change: ((StatusSnapshot) -> Void)?

    public init() {}

    func update(_ mutate: (inout StatusSnapshot) -> Void) {
        lock.lock()
        mutate(&snap)
        let copy = snap
        lock.unlock()
        on_change?(copy)
    }

    public var current: StatusSnapshot {
        lock.lock(); defer { lock.unlock() }
        return snap
    }

    public var json: Data {
        lock.lock(); defer { lock.unlock() }
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        return (try? enc.encode(snap)) ?? Data("{}".utf8)
    }
}
