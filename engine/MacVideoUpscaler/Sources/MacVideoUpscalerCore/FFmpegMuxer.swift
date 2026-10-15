import Foundation
import os

/// A subtitle track found in the source, and how it must be drawn.
struct SubtitleTrack {
    let index: Int          // absolute stream index in the source file (for display)
    let ordinal: Int        // 0-based position among the file's subtitle streams (for filters)
    let codec: String       // e.g. dvd_subtitle, subrip, ass
    let bitmap: Bool        // true = image-based (needs overlay), false = text (libass)
    let language: String?
    let title: String?
}

/// An audio track found in the source (for selection).
struct AudioTrack {
    let index: Int          // absolute stream index in the source file (for display)
    let ordinal: Int        // 0-based position among the file's audio streams (for -map 1:a:ORD)
    let codec: String
    let channels: Int?
    let language: String?
    let title: String?
}

/// Subtitle burn-in configuration handed to the muxer.
struct SubtitleBurn {
    let track: SubtitleTrack
    let sourceWidth: Int
    let sourceHeight: Int
    var externalURL: URL? = nil
}

/// Wraps an ffmpeg subprocess that:
///   * reads raw BGRA video frames on stdin (already upscaled),
///   * optionally burns in a subtitle track from the source file,
///   * maps the source file's audio track (aligned to the pipeline start offset),
///   * hardware-encodes video via VideoToolbox (h264_videotoolbox),
///   * writes an MPEG-TS stream to stdout (consumed by StreamServer's ring).
///
/// Backpressure: writing a frame to stdin blocks while ffmpeg is busy; the TS
/// bytes land in the caller's bounded ring so total memory stays bounded.
final class FFmpegMuxer {
    private(set) var process: Process!
    private var stdinHandle: FileHandle?
    private var stdinFD: Int32?
    private(set) var stdoutHandle: FileHandle!
    private(set) var stderrHandle: FileHandle!
    /// Set while terminating, so the expected short-frame complaint from a
    /// mid-frame stdin close is not forwarded as if it were a real error.
    private let terminating = OSAllocatedUnfairLock(initialState: false)

    let videoBitrateBps: Int
    let maxrateBps: Int
    let bufsizeBits: Int
    /// Encoder actually used. HEVC for anamorphic sources: hardware H.264 does
    /// not write the aspect ratio into the bitstream, and MPEG-TS has nowhere
    /// else to put it, so a 720x480 SAR 32:27 source would arrive as square-pixel
    /// 3:2. h264_videotoolbox cannot carry it (verified); hevc_videotoolbox can.
    let videoCodec: String
    /// Full ffmpeg argument list (kept for diagnostics/reporting).
    let arguments: [String]

    private(set) var framesWritten = 0

    /// `w`/`h` are the rawvideo frame dimensions being fed (i.e. the 2x output).
    static func bitrate(forWidth w: Int, height h: Int, fps: Double) -> (bps: Int, max: Int, bufsize: Int) {
        // ~0.09 bits per output pixel per frame: 4K30 -> ~22 Mb/s, 960p30 -> ~7.6 Mb/s.
        let bps = Int(Double(w * h) * max(fps, 20.0) * 0.09)
        let capped = min(max(bps, 2_000_000), 32_000_000)
        return (capped, capped * 3 / 2, capped * 2)
    }

    /// Escapes a path for use inside an ffmpeg filter argument.
    static func escapeFilterPath(_ path: String) -> String {
        var s = path
        s = s.replacingOccurrences(of: "\\", with: "\\\\")
        s = s.replacingOccurrences(of: ":", with: "\\:")
        s = s.replacingOccurrences(of: "'", with: "\\'")
        s = s.replacingOccurrences(of: ",", with: "\\,")
        s = s.replacingOccurrences(of: "[", with: "\\[")
        s = s.replacingOccurrences(of: "]", with: "\\]")
        return s
    }

    /// - Parameters:
    ///   - sourceURL: file whose audio track is muxed in.
    ///   - audioStartSeconds: seek offset matching the video pipeline start.
    ///   - width/height: rawvideo frame size being fed (= 2x output size).
    ///   - fps: nominal source frame rate (rawvideo stdin has no timestamps).
    ///   - audioOrdinal: which of the file's audio streams to mux (0 = first).
    ///   - displayAspect: DAR to stamp on the output ("16:9") so anamorphic
    ///     sources keep their display geometry; nil for square pixels.
    init(sourceURL: URL, audioStartSeconds: Double,
         width: Int, height: Int, fps: Double,
         audioEnabled: Bool = true, audioOrdinal: Int = 0,
         displayAspect: String? = nil) throws {
        let br = Self.bitrate(forWidth: width, height: height, fps: fps)
        self.videoBitrateBps = br.bps
        self.maxrateBps = br.max
        self.bufsizeBits = br.bufsize
        let encoder = displayAspect == nil ? "h264_videotoolbox" : "hevc_videotoolbox"
        self.videoCodec = encoder

        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg")
        var args = [
            "-hide_banner", "-v", "error",
            "-f", "rawvideo",
            "-pixel_format", "bgra",
            "-video_size", "\(width)x\(height)",
            "-framerate", String(format: "%.5f", fps),
            "-i", "pipe:0",
        ]
        if audioEnabled {
            args += [
                "-ss", String(format: "%.3f", max(0, audioStartSeconds)),
                "-i", sourceURL.path,
                "-map", "0:v:0", "-map", "1:a:\(max(0, audioOrdinal))?",
                "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            ]
        } else {
            args += ["-map", "0:v:0", "-an"]
        }
        args += [
            "-c:v", encoder,
            "-b:v", "\(videoBitrateBps)",
            "-maxrate", "\(maxrateBps)",
            "-bufsize", "\(bufsizeBits)",
        ]
        if encoder == "h264_videotoolbox" {
            args += ["-profile:v", "high"]
        }
        args += [
            "-pix_fmt", "yuv420p",
            "-g", "60",
            "-r", String(format: "%.5f", fps),
        ]
        if let displayAspect {
            // Uniform 2x/4x scaling leaves the DAR unchanged, so restating the
            // source DAR reproduces the original pixel geometry.
            args += ["-aspect", displayAspect]
        }
        args += [
            "-f", "mpegts",
            "pipe:1",
        ]
        p.arguments = args
        self.arguments = args
        if ProcessInfo.processInfo.environment["ZU_DEBUG_ARGS"] == "1" {
            FileHandle.standardError.write(Data("MUXER ARGS: \(args.joined(separator: " "))\n".utf8))
        }

        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = errPipe

        // Surface muxer errors to the log rather than a blocked pipe. Teardown
        // noise is filtered: closing stdin mid-frame makes ffmpeg report a short
        // rawvideo packet ("Invalid buffer size ... < expected frame_size"),
        // which is expected on every seek and would otherwise look alarming.
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] fh in
            let data = fh.availableData
            guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
            if self?.terminating.withLock({ $0 }) == true { return }
            FileHandle.standardError.write("ffmpeg: \(s)".data(using: .utf8)!)
        }

        try p.run()
        self.process = p
        self.stdinHandle = inPipe.fileHandleForWriting
        self.stdinFD = inPipe.fileHandleForWriting.fileDescriptor
        self.stdoutHandle = outPipe.fileHandleForReading
        self.stderrHandle = errPipe.fileHandleForReading
    }

    /// Blocking write of one BGRA frame (rawvideo has no framing).
    ///
    /// Uses raw write(2) on the pipe directly: no per-chunk Data allocation, so
    /// the steady state performs zero heap work per frame (a 4K frame would
    /// otherwise allocate 33 x 1 MB temporary buffers).
    func writeFrame(_ bytes: UnsafePointer<UInt8>, length: Int) throws {
        guard let fd = stdinFD else { throw EngineError.muxerFailure("stdin closed") }
        var off = 0
        while off < length {
            let n = write(fd, bytes + off, length - off)  // blocks when pipe full
            if n > 0 {
                off += n
            } else if n < 0 && errno == EINTR {
                continue
            } else {
                throw EngineError.muxerFailure("write to ffmpeg stdin failed (errno \(errno))")
            }
        }
        framesWritten += 1
    }

    /// True if ffmpeg exited (e.g. encoder error).
    var hasExited: Bool { process != nil && process.isRunning == false }

    /// Close stdin and wait for the tail of the TS stream.
    func finish() {
        guard let p = process, p.isRunning else { return }
        try? stdinHandle?.close()
        stdinHandle = nil
        stdinFD = nil
        Subprocess.wait(for: p, timeout: 30)
    }

    func terminate() {
        guard let p = process, p.isRunning else { return }
        terminating.withLock { $0 = true }
        try? stdinHandle?.close()
        stdinHandle = nil
        stdinFD = nil
        p.terminate()
        // Short grace: on a seek the encoder is usually blocked writing to a
        // full, unread stdout pipe, where SIGTERM's graceful exit can hang —
        // waiting seconds here is what made seeking feel broken.
        let deadline = Date().addingTimeInterval(0.15)
        while p.isRunning && Date() < deadline { usleep(20_000) }
        if p.isRunning { kill(p.processIdentifier, SIGKILL) }
        Subprocess.wait(for: p, timeout: 2)
    }
}
