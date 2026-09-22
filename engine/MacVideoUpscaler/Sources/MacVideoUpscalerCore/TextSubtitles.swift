import Foundation
import CoreGraphics
import CoreText

/// Minimal text-subtitle renderer used because the installed ffmpeg has no
/// libass (this Homebrew build is compiled without it, so `-vf subtitles=`
/// does not exist on this machine).
///
/// Cues are extracted with ffmpeg as SRT, rendered once per cue with CoreText
/// (cached), then blended into the source-resolution frame **before** the 2x
/// upscale — so subtitle glyphs get the same line reconstruction as the rest
/// of the picture instead of being stretched as a bitmap over the upscaled
/// video. ASS/SSA styling is not supported (no libass); plain cues are.
final class TextSubtitleRenderer {
    struct Cue {
        let start: Double
        let end: Double
        let lines: [String]
    }

    let cues: [Cue]
    private let sourceWidth: Int
    private let sourceHeight: Int
    private let fontSize: CGFloat
    private var cache: [Int: CGImage] = [:]

    init(cues: [Cue], sourceWidth: Int, sourceHeight: Int) {
        self.cues = cues
        self.sourceWidth = max(1, sourceWidth)
        self.sourceHeight = max(1, sourceHeight)
        self.fontSize = max(10, min(CGFloat(self.sourceHeight) * 0.055, 64))
    }

    // MARK: - Loading

    /// Extracts the subtitle stream as SRT via ffmpeg and parses it.
    static func load(url: URL, ordinal: Int = 0) -> TextSubtitleRenderer? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: FFmpegDecoder.ffmpegPath)
        p.arguments = ["-v", "error", "-i", url.path,
                       "-map", "0:s:\(ordinal)", "-f", "srt", "pipe:1"]
        let out = Pipe()
        p.standardOutput = out
        p.standardError = Pipe()
        p.standardInput = FileHandle.nullDevice   // never touch the controlling terminal
        guard (try? p.run()) != nil else { return nil }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        Subprocess.wait(for: p, timeout: 10)
        guard let text = String(data: data, encoding: .utf8), !text.isEmpty else { return nil }
        let cues = parseSRT(text)
        guard !cues.isEmpty else { return nil }
        let meta = (try? FFmpegDecoder(url: url))?.meta
        return TextSubtitleRenderer(cues: cues,
                                    sourceWidth: meta?.width ?? 1920,
                                    sourceHeight: meta?.height ?? 1080)
    }

    static func parseSRT(_ text: String) -> [Cue] {
        var cues: [Cue] = []
        let blocks = text.replacingOccurrences(of: "\r\n", with: "\n")
            .components(separatedBy: "\n\n")
        let timeRe = try! NSRegularExpression(
            pattern: #"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"#)
        for block in blocks {
            let lines = block.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            guard let timeLine = lines.first(where: { $0.contains("-->") }) else { continue }
            let ns = timeLine as NSString
            guard let m = timeRe.firstMatch(in: timeLine,
                                            range: NSRange(location: 0, length: ns.length)) else { continue }
            func g(_ i: Int) -> Double {
                guard let r = Range(m.range(at: i), in: timeLine) else { return 0 }
                return Double(timeLine[r]) ?? 0
            }
            let start = g(1) * 3600 + g(2) * 60 + g(3) + g(4) / 1000
            let end = g(5) * 3600 + g(6) * 60 + g(7) + g(8) / 1000
            let body = lines
                .filter { !$0.contains("-->") }
                .map { stripTags($0).trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && Int($0) == nil }
            if !body.isEmpty {
                cues.append(Cue(start: start, end: end, lines: body))
            }
        }
        return cues
    }

    private static func stripTags(_ s: String) -> String {
        s.replacingOccurrences(of: #"<[^>]+>"#, with: "", options: .regularExpression)
         .replacingOccurrences(of: #"\{\\[^}]*\}"#, with: "", options: .regularExpression)
    }

    // MARK: - Rendering

    private func wrappedLines(_ cue: Cue, font: CTFont) -> [String] {
        let maxWidth = CGFloat(sourceWidth) * 0.80
        var out: [String] = []
        for raw in cue.lines {
            let useSpaces = raw.contains(" ")
            let units = useSpaces ? raw.split(separator: " ").map(String.init) : raw.map(String.init)
            var current = ""
            for u in units {
                let candidate = current.isEmpty ? u : (useSpaces ? current + " " + u : current + u)
                if measure(candidate, font: font) > maxWidth && !current.isEmpty {
                    out.append(current)
                    current = u
                } else {
                    current = candidate
                }
            }
            if !current.isEmpty { out.append(current) }
        }
        return out
    }

    private func measure(_ s: String, font: CTFont) -> CGFloat {
        let attr = NSAttributedString(string: s, attributes: [kCTFontAttributeName as NSAttributedString.Key: font])
        let line = CTLineCreateWithAttributedString(attr)
        return CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
    }

    private func image(for index: Int) -> CGImage? {
        if let c = cache[index] { return c }
        let cue = cues[index]
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, fontSize, nil)
        let lines = wrappedLines(cue, font: font)
        guard !lines.isEmpty else { return nil }

        let lineHeight = fontSize * 1.22
        let canvasH = Int(lineHeight * CGFloat(lines.count) + fontSize * 0.6)
        let canvasW = sourceWidth
        guard let ctx = CGContext(data: nil, width: canvasW, height: canvasH,
                                  bitsPerComponent: 8, bytesPerRow: canvasW * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue |
                                              CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.clear(CGRect(x: 0, y: 0, width: canvasW, height: canvasH))
        ctx.textMatrix = .identity

        let white = CGColor(gray: 1.0, alpha: 1.0)
        let black = CGColor(gray: 0.0, alpha: 1.0)
        // Memory row 0 is the top row; CoreText draws with y up, so line 0 sits
        // highest by placing it at the largest y.
        for (i, text) in lines.enumerated() {
            let attrs: [NSAttributedString.Key: Any] = [
                kCTFontAttributeName as NSAttributedString.Key: font,
                kCTForegroundColorAttributeName as NSAttributedString.Key: white,
                kCTStrokeColorAttributeName as NSAttributedString.Key: black,
                kCTStrokeWidthAttributeName as NSAttributedString.Key: -7.0,
            ]
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs))
            let width = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
            let x = (CGFloat(canvasW) - width) / 2
            let y = CGFloat(canvasH) - lineHeight * CGFloat(i + 1)
            ctx.textPosition = CGPoint(x: x, y: y)
            CTLineDraw(line, ctx)
        }
        let img = ctx.makeImage()
        if let img { cache[index] = img }
        return img
    }

    // MARK: - Compositing

    func activeCueIndex(at time: Double) -> Int? {
        cues.firstIndex { time >= $0.start && time < $0.end }
    }

    func isActive(at time: Double) -> Bool { activeCueIndex(at: time) != nil }

    /// Blends the active cue into a BGRA frame buffer (source resolution).
    func blend(into buffer: UnsafeMutableRawPointer, bytesPerRow: Int,
               width: Int, height: Int, time: Double) {
        guard let idx = activeCueIndex(at: time), let cg = image(for: idx) else { return }
        guard let data = cg.dataProvider?.data, let src = CFDataGetBytePtr(data) else { return }
        let cgW = cg.width, cgH = cg.height
        let bottomMargin = Int(Double(height) * 0.05)
        let dstY = height - bottomMargin - cgH
        guard dstY >= 0, cgW <= width else { return }
        let x0 = (width - cgW) / 2
        for row in 0..<cgH {
            let sRow = UnsafeRawPointer(src + row * cgW * 4).assumingMemoryBound(to: UInt8.self)
            let dRow = (buffer + (dstY + row) * bytesPerRow + x0 * 4).assumingMemoryBound(to: UInt8.self)
            for col in 0..<cgW {
                // premultiplied BGRA: [0]=A? no — byteOrder32Little + premultipliedFirst
                // stores BGRA in memory, i.e. [0]=B,[1]=G,[2]=R,[3]=A.
                let b = sRow[col * 4 + 0], g = sRow[col * 4 + 1]
                let r = sRow[col * 4 + 2], a = sRow[col * 4 + 3]
                if a == 0 { continue }
                if a == 255 {
                    dRow[col * 4 + 0] = b
                    dRow[col * 4 + 1] = g
                    dRow[col * 4 + 2] = r
                } else {
                    let inv = 255 - Int(a)
                    dRow[col * 4 + 0] = UInt8(min(255, Int(b) + Int(dRow[col * 4 + 0]) * inv / 255))
                    dRow[col * 4 + 1] = UInt8(min(255, Int(g) + Int(dRow[col * 4 + 1]) * inv / 255))
                    dRow[col * 4 + 2] = UInt8(min(255, Int(r) + Int(dRow[col * 4 + 2]) * inv / 255))
                }
            }
        }
    }
}
