import Foundation
import os

/// Decodes video via an ffmpeg subprocess (handles MKV/MP4/anything, uses
/// VideoToolbox hardware decode where available) and delivers raw BGRA frames
/// on stdout. Replaces AVFoundation, which cannot read Matroska at all.
///
/// Frames are read with blocking `read(2)` on the pipe: the decode process only
/// runs ahead as far as the pipeline consumes, so memory stays flat.
final class FFmpegDecoder {
    struct Meta {
        var width: Int
        var height: Int
        var fps: Double
        var durationSeconds: Double
        /// Display aspect ratio to re-apply on output, e.g. "16:9". Raw BGRA
        /// carries no pixel-geometry metadata, so anamorphic sources (DVD-era
        /// 720x480 SAR 32:27) would otherwise be delivered as square-pixel
        /// 1440x960 — a 3:2 raster instead of 16:9. nil when the source is
        /// square-pixel.
        var displayAspect: String?
        /// Absolute index of the video stream to decode — not simply 0. A poster
        /// frame is a video stream too, so a file whose cover art comes first
        /// would otherwise decode one still image instead of the film.
        var videoStreamIndex: Int
        /// Display-matrix rotation, normalised to 0/90/180/270. `width`/`height`
        /// are the *rotated* geometry and the decoder transposes to match, so a
        /// phone-shot portrait clip is not shown on its side.
        var rotation: Int
    }

    let url: URL
    let meta: Meta
    var frameSize: Int { meta.width * meta.height * 4 }

    private var process: Process?
    private var outHandle: FileHandle?
    private var errHandle: FileHandle?
    private(set) var framesRead = 0

    /// Exit code of the decoder, recorded by the termination handler. It must
    /// NOT be read from `Process.terminationStatus` directly: Foundation raises
    /// NSInvalidArgumentException ("task still running") if the child has not
    /// been reaped yet, which is exactly the state right after a read hits EOF.
    private let exitBox = OSAllocatedUnfairLock<Int32?>(initialState: nil)
    var exitCode: Int32? { exitBox.withLock { $0 } }

    /// Bounded wait for the decoder to exit so `exitCode` becomes available.
    func waitForExit(timeout: TimeInterval) {
        guard let p = process, p.isRunning else { return }
        let deadline = Date().addingTimeInterval(timeout)
        while p.isRunning && Date() < deadline { usleep(20_000) }
        Subprocess.wait(for: p, timeout: 0)  // non-pumping reap
    }

    static let ffmpegPath = "/opt/homebrew/bin/ffmpeg"
    static let ffprobePath = "/opt/homebrew/bin/ffprobe"

    init(url: URL) throws {
        self.url = url
        self.meta = try Self.probe(url: url)
    }

    static func probe(url: URL) throws -> Meta {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: ffprobePath)
        // Every video stream, not just the first: a poster frame is a video
        // stream, and picking it means showing a still instead of the film.
        p.arguments = [
            "-v", "error", "-select_streams", "v",
            "-show_entries", "stream=index,width,height,avg_frame_rate,r_frame_rate,"
                + "sample_aspect_ratio,display_aspect_ratio,duration",
            "-show_entries", "stream_disposition=attached_pic",
            "-show_entries", "stream_side_data=rotation",
            "-show_entries", "stream_tags=rotate",
            "-show_entries", "format=duration",
            "-of", "json", url.path,
        ]
        let out = Pipe()
        p.standardOutput = out
        p.standardError = Pipe()
        p.standardInput = FileHandle.nullDevice   // never touch the controlling terminal
        try p.run()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        Subprocess.wait(for: p, timeout: 5)
        guard p.terminationStatus == 0,
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let streams = obj["streams"] as? [[String: Any]], !streams.isEmpty else {
            throw EngineError.decoderFailure("ffprobe found no video stream in \(url.lastPathComponent)")
        }

        // Attached pictures are skipped outright; among what is left the longest
        // stream wins, which also covers a still that was not flagged as a cover
        // (some muxers do not set the disposition) sitting in front of the film.
        func isCover(_ s: [String: Any]) -> Bool {
            ((s["disposition"] as? [String: Any])?["attached_pic"] as? Int ?? 0) == 1
        }
        func seconds(_ s: [String: Any]) -> Double { Double(s["duration"] as? String ?? "") ?? 0 }
        let candidates = streams.filter { !isCover($0) }
        guard !candidates.isEmpty else {
            // Every video stream is a poster frame: this is an audio file with
            // cover art (a track dragged onto the player, say). Saying so beats
            // "playing" the artwork as if it were a film.
            throw EngineError.decoderFailure(
                "\(url.lastPathComponent) has no video — only cover art.")
        }
        var v = candidates[0]
        for c in candidates.dropFirst() where seconds(c) > seconds(v) { v = c }

        guard let w = v["width"] as? Int, let h = v["height"] as? Int, w > 0, h > 0 else {
            throw EngineError.decoderFailure("ffprobe could not read video dimensions of \(url.lastPathComponent)")
        }
        func parseFPS(_ s: String?) -> Double? {
            guard let s, s != "0/0" else { return nil }
            let parts = s.split(separator: "/")
            if parts.count == 2, let n = Double(parts[0]), let d = Double(parts[1]), d != 0 {
                return n / d
            }
            return Double(s)
        }
        let fps = parseFPS(v["avg_frame_rate"] as? String) ?? parseFPS(v["r_frame_rate"] as? String) ?? 30.0
        var duration = 0.0
        if let fmt = obj["format"] as? [String: Any], let ds = fmt["duration"] as? String {
            duration = Double(ds) ?? 0
        }

        // Rotation: the display matrix is what phones and cameras write, the
        // legacy `rotate` tag is what older muxers wrote.
        let sideRotation = (v["side_data_list"] as? [[String: Any]])?
            .compactMap { $0["rotation"] as? Int }.first
        let tagRotation = Int((v["tags"] as? [String: Any])?["rotate"] as? String ?? "")
        let raw = sideRotation ?? tagRotation ?? 0
        let rotation = ((raw % 360) + 360) % 360

        // A quarter turn swaps the raster: the decoder transposes, so everything
        // downstream (texture, output size, letterboxing) must use the swapped
        // geometry.
        let quarterTurn = rotation == 90 || rotation == 270
        let (outW, outH) = quarterTurn ? (h, w) : (w, h)

        var aspect = displayAspect(from: v, width: w, height: h)
        if quarterTurn, let a = aspect, let r = parseRatio(a) {
            aspect = "\(r.1):\(r.0)"      // a quarter turn inverts the display ratio
        }

        return Meta(width: outW, height: outH, fps: (fps > 1 && fps < 241) ? fps : 30.0,
                    durationSeconds: duration, displayAspect: aspect,
                    videoStreamIndex: v["index"] as? Int ?? 0, rotation: rotation)
    }

    /// The filter that applies `meta.rotation`, or nil when there is nothing to
    /// do. Chosen to match ffmpeg's own autorotation exactly (verified
    /// bit-identical on a `-display_rotation` sample at 90, 180 and 270), with
    /// `-noautorotate` passed so the two can never be applied twice.
    var rotateFilter: String? {
        switch meta.rotation {
        case 90: return "transpose=2"          // 90° counter-clockwise
        case 180: return "vflip,hflip"
        case 270: return "transpose=1"         // 90° clockwise
        default: return nil
        }
    }

    /// "num:den" ratio parser ("32:27"). Rejects "N/A" and "0:1".
    static func parseRatio(_ s: String?) -> (Int, Int)? {
        guard let s, s != "N/A" else { return nil }
        let parts = s.split(separator: ":")
        guard parts.count == 2, let n = Int(parts[0]), let d = Int(parts[1]), n > 0, d > 0 else {
            return nil
        }
        return (n, d)
    }

    /// The display aspect ratio to stamp on the output. Derived from the sample
    /// aspect ratio (most reliable), falling back to the probed display ratio;
    /// nil for square pixels so nothing is written unnecessarily.
    private static func displayAspect(from stream: [String: Any], width: Int, height: Int) -> String? {
        if let sar = parseRatio(stream["sample_aspect_ratio"] as? String) {
            if sar == (1, 1) { return nil }
            let num = sar.0 * width, den = sar.1 * height
            let g = gcd(num, den)
            return "\(num / g):\(den / g)"
        }
        if let dar = parseRatio(stream["display_aspect_ratio"] as? String) {
            let source = (width, height)
            let g = gcd(dar.0, dar.1)
            let reduced = (dar.0 / g, dar.1 / g)
            let sg = gcd(source.0, source.1)
            if reduced != (source.0 / sg, source.1 / sg) { return "\(reduced.0):\(reduced.1)" }
        }
        return nil
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var x = abs(a), y = abs(b)
        while y != 0 { (x, y) = (y, x % y) }
        return max(x, 1)
    }

    /// Runs ffprobe for one stream type and returns the raw stream dictionaries.
    private static func probeStreams(url: URL, select: String) -> [[String: Any]] {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: ffprobePath)
        p.arguments = [
            "-v", "error", "-select_streams", select,
            "-show_entries", "stream=index,codec_name,channels:stream_tags=language,title",
            "-of", "json", url.path,
        ]
        let out = Pipe()
        p.standardOutput = out
        p.standardError = Pipe()
        p.standardInput = FileHandle.nullDevice   // never touch the controlling terminal
        guard (try? p.run()) != nil else { return [] }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        Subprocess.wait(for: p, timeout: 5)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let streams = obj["streams"] as? [[String: Any]] else { return [] }
        return streams
    }

    /// Audio tracks present in the source (for track selection).
    static func probeAudio(url: URL) -> [AudioTrack] {
        probeStreams(url: url, select: "a").enumerated().compactMap { ordinal, st in
            guard let idx = st["index"] as? Int,
                  let codec = st["codec_name"] as? String else { return nil }
            let tags = st["tags"] as? [String: Any]
            return AudioTrack(index: idx, ordinal: ordinal, codec: codec,
                              channels: st["channels"] as? Int,
                              language: tags?["language"] as? String,
                              title: tags?["title"] as? String)
        }
    }

    /// Subtitle tracks present in the source (for burn-in).
    static func probeSubtitles(url: URL) -> [SubtitleTrack] {
        let bitmapCodecs: Set<String> = ["dvd_subtitle", "dvb_subtitle",
                                         "hdmv_pgs_subtitle", "xsub"]
        return probeStreams(url: url, select: "s").enumerated().compactMap { ordinal, st in
            guard let idx = st["index"] as? Int,
                  let codec = st["codec_name"] as? String else { return nil }
            let tags = st["tags"] as? [String: Any]
            return SubtitleTrack(index: idx, ordinal: ordinal, codec: codec,
                                 bitmap: bitmapCodecs.contains(codec),
                                 language: tags?["language"] as? String,
                                 title: tags?["title"] as? String)
        }
    }

    /// Spawns the decoder starting at `seconds` (input seeking).
    ///
    /// Subtitles are burned in HERE, at source resolution, so the 2x upscaler
    /// sees them as picture content: glyph edges get the same line
    /// reconstruction as the rest of the frame, and text is scaled once
    /// (instead of being stretched as a bitmap over the upscaled video).
    func start(at seconds: Double, subtitle: SubtitleBurn? = nil) throws {
        stop()
        exitBox.withLock { $0 = nil }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: Self.ffmpegPath)
        var args = [
            "-v", "error",
            "-hwaccel", "videotoolbox",
            // The rotation is applied by an explicit filter below, so ffmpeg must
            // not also apply it — that would double it.
            "-noautorotate",
            "-ss", String(format: "%.3f", max(0, seconds)),
        ]
        args += [
            "-i", url.path,
        ]
        // Map by absolute stream index: the first video stream may be cover art.
        let video = "0:\(meta.videoStreamIndex)"
        if let sub = subtitle {
            if sub.track.bitmap {
                if let externalURL = sub.externalURL {
                    // Keep cues that began before a seek but are still visible.
                    args += ["-itsoffset", String(format: "%.3f", -max(0, seconds)), "-i", externalURL.path]
                }
                // A bitmap subpicture is a picture layer: its position comes from
                // a window defined against the *source* frame, so it is composited
                // before the rotation and the finished frame is turned as a whole.
                // Overlaying after the turn instead reinterprets that window
                // against the rotated geometry, which moves the cue (measured on a
                // 640x480 disc stream tagged 90°: the same cue lands on the right
                // edge rather than bottom-centre where the disc put it). It also
                // distorts the subpicture, since the window is not rotated with
                // it. The unrotated path is byte-identical either way. Text
                // subtitles are the opposite case: they are drawn upright on the
                // rotated frame, because text is not part of the picture.
                let quarterTurn = meta.rotation == 90 || meta.rotation == 270
                let srcW = quarterTurn ? meta.height : meta.width
                let srcH = quarterTurn ? meta.width : meta.height
                let input = sub.externalURL == nil ? 0 : 1
                var graph = "[\(input):\(sub.track.index)]scale=\(srcW):\(srcH):flags=bicubic[sub];"
                    + "[\(video)][sub]overlay=format=auto:eof_action=repeat[v0]"
                if let rotate = rotateFilter { graph += ";[v0]\(rotate)[v]" }
                args += [
                    "-filter_complex",
                    graph,
                    "-map", rotateFilter == nil ? "[v0]" : "[v]",
                ]
            } else {
                // Text subtitles are rendered natively (see TextSubtitleRenderer)
                // because this ffmpeg build has no libass.
                args += ["-map", video]
            }
        } else {
            args += ["-map", video]
        }
        if let rotate = rotateFilter, subtitle?.track.bitmap != true {
            args += ["-vf", rotate]
        }
        args += [
            "-fps_mode", "cfr",
            "-r", String(format: "%.5f", meta.fps),
            "-f", "rawvideo",
            "-pix_fmt", "bgra",
            "-an", "-sn",
            "pipe:1",
        ]
        p.arguments = args
        // Record the exit code here, where Foundation permits reading it.
        p.terminationHandler = { [weak self] proc in
            self?.exitBox.withLock { $0 = proc.terminationStatus }
        }
        if ProcessInfo.processInfo.environment["ZU_DEBUG_ARGS"] == "1" {
            FileHandle.standardError.write(Data("DECODER ARGS: \(args.joined(separator: " "))\n".utf8))
        }
        let out = Pipe()
        let err = Pipe()
        p.standardOutput = out
        p.standardError = err
        // ffmpeg reads stdin for interactive keyboard commands when it is a
        // terminal. Inheriting our terminal makes it fail badly when our process
        // group is not the terminal's foreground group: the read raises SIGTTIN
        // and the kernel STOPS the decoder, which then never produces a frame and
        // hangs playback with no error. /dev/null is not a tty, so interaction is
        // disabled and the terminal is left alone.
        p.standardInput = FileHandle.nullDevice
        err.fileHandleForReading.readabilityHandler = { fh in
            let d = fh.availableData
            if !d.isEmpty, let s = String(data: d, encoding: .utf8) {
                FileHandle.standardError.write("ffmpeg-decode: \(s)".data(using: .utf8)!)
            }
        }
        try p.run()
        process = p
        outHandle = out.fileHandleForReading
        errHandle = err.fileHandleForReading
        framesRead = 0
    }

    /// Blocking read of exactly one frame into `dst` (frameSize bytes).
    /// Returns false at end of stream.
    func readFrame(into dst: UnsafeMutableRawPointer) -> Bool {
        guard let fd = outHandle?.fileDescriptor else { return false }
        var got = 0
        let need = frameSize
        while got < need {
            let n = read(fd, dst + got, need - got)
            if n > 0 {
                got += n
            } else if n == 0 {
                return false  // EOF
            } else if errno == EINTR {
                continue
            } else {
                return false
            }
        }
        framesRead += 1
        return true
    }

    func stop() {
        guard let p = process else { return }
        if p.isRunning { p.terminate() }
        // Short grace, then SIGKILL: during an abort the decoder is typically
        // blocked writing rawvideo to a full, unread stdout pipe, where a
        // graceful SIGTERM exit can hang (this is on the seek critical path).
        let deadline = Date().addingTimeInterval(0.15)
        while p.isRunning && Date() < deadline { usleep(20_000) }
        if p.isRunning { kill(p.processIdentifier, SIGKILL) }
        Subprocess.wait(for: p, timeout: 2)
        outHandle = nil
        errHandle = nil
        process = nil
    }

    deinit { stop() }   // never leave an ffmpeg child behind a discarded session
}
