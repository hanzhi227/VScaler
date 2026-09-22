import Foundation
import os

/// Decodes the source file's selected audio track to interleaved float32 PCM.
///
/// The embedded player cannot reuse the muxer's audio path: it needs *raw PCM*
/// so each chunk can be stamped with a presentation time and scheduled on the
/// same render clock as the upscaled video. ffmpeg does the container/codec work
/// (it reads Matroska, which AVFoundation cannot) and the resample to the fixed
/// 48 kHz stereo layout happens in the same process.
final class FFmpegAudioDecoder {
    static let sampleRate: Int32 = 48000
    static let channelCount = 2
    /// f32 interleaved stereo.
    static let bytesPerFrame = 4 * channelCount
    private static let ffmpegPath = FFmpegDecoder.ffmpegPath

    private var process: Process?
    private var outHandle: FileHandle?
    private let exitBox = OSAllocatedUnfairLock<Int32?>(initialState: nil)
    var exitCode: Int32? { exitBox.withLock { $0 } }

    /// Total PCM frames delivered so far (also the timeline position).
    private(set) var framesRead: Int64 = 0

    /// Spawns the audio decoder starting at `seconds`, using the source's
    /// `ordinal`-th audio stream (0-based).
    func start(url: URL, at seconds: Double, ordinal: Int) throws {
        stop()
        exitBox.withLock { $0 = nil }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: Self.ffmpegPath)
        p.arguments = [
            "-v", "error",
            "-ss", String(format: "%.3f", max(0, seconds)),
            "-i", url.path,
            "-map", "0:a:\(max(0, ordinal))",
            "-vn", "-sn",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", "\(Self.sampleRate)",
            "-ac", "\(Self.channelCount)",
            "pipe:1",
        ]
        p.terminationHandler = { [weak self] proc in
            self?.exitBox.withLock { $0 = proc.terminationStatus }
        }
        let out = Pipe()
        p.standardOutput = out
        // Audio decode errors are reported through the exit code; the decoder is
        // optional, so stderr noise is not forwarded.
        p.standardError = FileHandle.nullDevice
        // Same reason as the video decoder: an inherited terminal makes ffmpeg
        // read stdin for keyboard commands, which raises SIGTTIN and STOPS the
        // child when we are not the terminal's foreground process group.
        p.standardInput = FileHandle.nullDevice
        try p.run()
        process = p
        outHandle = out.fileHandleForReading
        framesRead = 0
    }

    /// Blocking read of exactly `frames` PCM frames. Returns the number actually
    /// read; a short count means end of stream (or the decoder was stopped).
    func readFrames(into dst: UnsafeMutableRawPointer, frames: Int) -> Int {
        guard let fd = outHandle?.fileDescriptor else { return 0 }
        let need = frames * Self.bytesPerFrame
        var got = 0
        while got < need {
            let n = read(fd, dst + got, need - got)
            if n > 0 {
                got += n
            } else if n == 0 {
                break  // EOF
            } else if errno == EINTR {
                continue
            } else {
                break
            }
        }
        let whole = got / Self.bytesPerFrame
        framesRead += Int64(whole)
        return whole
    }

    func stop() {
        guard let p = process else { return }
        if p.isRunning { p.terminate() }
        // Same reasoning as the video decoder: a decoder blocked writing PCM into
        // a full, unread pipe can ignore SIGTERM for the whole grace period, and
        // this is on the seek critical path.
        let deadline = Date().addingTimeInterval(0.15)
        while p.isRunning && Date() < deadline { usleep(20_000) }
        if p.isRunning { kill(p.processIdentifier, SIGKILL) }
        Subprocess.wait(for: p, timeout: 2)
        outHandle = nil
        process = nil
    }

    deinit { stop() }   // never leave an ffmpeg child behind a discarded session
}
