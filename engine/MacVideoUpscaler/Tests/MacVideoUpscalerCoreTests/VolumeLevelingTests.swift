import XCTest
@testable import MacVideoUpscalerCore

final class VolumeLevelingTests: XCTestCase {
    func testLevelingReducesDynamicsWithoutClippingOrChangingTiming() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("dynamics.wav")
        let ffmpeg = Process()
        ffmpeg.executableURL = URL(fileURLWithPath: FFmpegDecoder.ffmpegPath)
        // Silence, quiet dialogue level, then a sudden loud effect; the right
        // channel is 6 dB quieter so we can also check stereo gain linking.
        let signal = "if(lt(t,1),0,if(lt(t,3),0.02,1.0))*sin(2*PI*1000*t)"
        ffmpeg.arguments = ["-v", "error", "-f", "lavfi", "-i",
                            "aevalsrc='\(signal)|0.5*(\(signal))':s=48000:d=6",
                            "-c:a", "pcm_f32le", file.path]
        ffmpeg.standardInput = FileHandle.nullDevice
        try ffmpeg.run()
        ffmpeg.waitUntilExit()
        XCTAssertEqual(ffmpeg.terminationStatus, 0)

        func decode(leveling: Bool, at seconds: Double = 0) throws -> [Float] {
            let decoder = FFmpegAudioDecoder()
            try decoder.start(url: file, at: seconds, ordinal: 0, volumeLeveling: leveling)
            defer { decoder.stop() }
            var samples: [Float] = []
            var chunk = [Float](repeating: 0, count: 2048 * 2)
            while true {
                let frames = chunk.withUnsafeMutableBytes { decoder.readFrames(into: $0.baseAddress!, frames: 2048) }
                guard frames > 0 else { break }
                samples.append(contentsOf: chunk.prefix(frames * 2))
            }
            return samples
        }
        let original = try decode(leveling: false)
        let leveled = try decode(leveling: true)
        XCTAssertEqual(original.count, 6 * 48000 * 2)
        XCTAssertEqual(leveled.count, original.count, "Lookahead must not add delay or lose the tail")
        guard leveled.count == original.count, !leveled.isEmpty else { return }
        XCTAssertTrue(leveled.allSatisfy { $0.isFinite && abs($0) <= 0.95001 })
        XCTAssertTrue(leveled.prefix(48000 * 2).allSatisfy { abs($0) < 0.000001 })

        func rms(_ samples: [Float], from start: Int, to end: Int) -> Double {
            let values = stride(from: start * 48000 * 2, to: end * 48000 * 2, by: 2)
            return sqrt(values.reduce(0.0) { $0 + pow(Double(samples[$1]), 2) }
                        / Double((end - start) * 48000))
        }
        let quietGain = rms(leveled, from: 2, to: 3) / rms(original, from: 2, to: 3)
        let loudGain = rms(leveled, from: 4, to: 5) / rms(original, from: 4, to: 5)
        XCTAssertEqual(quietGain, 2.5, accuracy: 0.05)
        XCTAssertLessThan(loudGain, 0.6)
        for i in stride(from: 0, to: leveled.count, by: 2) {
            XCTAssertEqual(leveled[i + 1], leveled[i] * 0.5, accuracy: 0.00001)
        }
        let sought = try decode(leveling: true, at: 2)
        XCTAssertEqual(sought.count, 4 * 48000 * 2, "Seeking must preserve PCM duration")
        print(String(format: "Volume leveling: quiet gain %.2fx, loud gain %.2fx, range reduced %.1f dB, peak %.3f",
                     quietGain, loudGain, 20 * log10(quietGain / loudGain),
                     leveled.map { abs($0) }.max() ?? 0))
    }
}
