import Foundation
import CoreGraphics
import CoreText
import CoreVideo

/// Minimal text-subtitle renderer used because the installed ffmpeg has no
/// libass (this Homebrew build is compiled without it, so `-vf subtitles=`
/// does not exist on this machine).
///
/// The in-app player draws the cached image in a layer above the picture, so
/// the upscaler never sees the lettering and a cue is rasterised once. The
/// stream encoder still blends into the frame, because an MPEG-TS player has
/// no separate subtitle layer. ASS/SSA files use their plain text cues;
/// embedded styling is not interpreted.
final class TextSubtitleRenderer {
    struct Cue {
        let start: Double
        let end: Double
        let lines: [String]
    }

    let cues: [Cue]
    private let sourceWidth: Int
    private let sourceHeight: Int
    private let style: SubtitleStyle
    private let fontSize: CGFloat
    private var cache: [Int: CGImage] = [:]

    init(cues: [Cue], sourceWidth: Int, sourceHeight: Int, style: SubtitleStyle = SubtitleStyle()) {
        self.cues = cues
        self.sourceWidth = max(1, sourceWidth)
        self.sourceHeight = max(1, sourceHeight)
        self.style = style.normalized
        self.fontSize = max(10, CGFloat(self.sourceHeight) * 0.045 * CGFloat(style.normalized.size))
    }

    // MARK: - Loading

    /// Extracts the subtitle stream as SRT via ffmpeg and parses it.
    static func load(url: URL, ordinal: Int = 0) -> TextSubtitleRenderer? {
        guard let cues = loadCues(url: url, ordinal: ordinal) else { return nil }
        let meta = (try? FFmpegDecoder(url: url))?.meta
        return TextSubtitleRenderer(cues: cues,
                                    sourceWidth: meta?.width ?? 1920,
                                    sourceHeight: meta?.height ?? 1080)
    }

    static func loadCues(url: URL, ordinal: Int = 0) -> [Cue]? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: FFmpegDecoder.ffmpegPath)
        p.arguments = ["-v", "error", "-i", url.path,
                       "-map", "0:s:\(ordinal)", "-f", "srt", "pipe:1"]
        let out = Pipe()
        let err = Pipe()
        p.standardOutput = out
        p.standardError = err
        p.standardInput = FileHandle.nullDevice   // never touch the controlling terminal
        guard (try? p.run()) != nil else { return nil }
        // Drain stderr concurrently (see FFmpegDecoder.probe): a child blocked
        // on a full stderr pipe never closes stdout, hanging the read below.
        DispatchQueue.global(qos: .utility).async {
            _ = err.fileHandleForReading.readDataToEndOfFile()
        }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        Subprocess.wait(for: p, timeout: 10)
        guard !p.isRunning, p.terminationStatus == 0,
              let text = String(data: data, encoding: .utf8), !text.isEmpty else { return nil }
        let cues = parseSRT(text)
        guard !cues.isEmpty else { return nil }
        return cues
    }

    static func parseSRT(_ text: String) -> [Cue] {
        var cues: [Cue] = []
        let blocks = text.replacingOccurrences(of: "\r\n", with: "\n")
            .components(separatedBy: "\n\n")
        let timeRe = try! NSRegularExpression(
            pattern: #"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"#)
        for block in blocks {
            let lines = block.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            guard let timeIdx = lines.firstIndex(where: { $0.contains("-->") }) else { continue }
            let timeLine = lines[timeIdx]
            let ns = timeLine as NSString
            guard let m = timeRe.firstMatch(in: timeLine,
                                            range: NSRange(location: 0, length: ns.length)) else { continue }
            func g(_ i: Int) -> Double {
                guard let r = Range(m.range(at: i), in: timeLine) else { return 0 }
                return Double(timeLine[r]) ?? 0
            }
            let start = g(1) * 3600 + g(2) * 60 + g(3) + g(4) / 1000
            let end = g(5) * 3600 + g(6) * 60 + g(7) + g(8) / 1000
            // The body is everything AFTER the timing line; the SRT cue index is
            // whatever sits BEFORE it. Filtering numeric lines anywhere in the
            // block would also swallow a cue whose whole text is a number
            // ("1", "007"), so the index is dropped by position instead.
            let body = lines.dropFirst(timeIdx + 1)
                .map { stripTags($0).trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
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
        let font = style.typeface.font(size: fontSize) as CTFont
        let lines = wrappedLines(cue, font: font)
        guard !lines.isEmpty else { return nil }

        let lineHeight = fontSize * 1.22
        let canvasH = Int(lineHeight * CGFloat(lines.count) + fontSize * 0.6)
        let canvasW = min(sourceWidth, Int(ceil((lines.map { measure($0, font: font) }.max() ?? 0) + fontSize)))
        guard let ctx = CGContext(data: nil, width: canvasW, height: canvasH,
                                  bitsPerComponent: 8, bytesPerRow: canvasW * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue |
                                              CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.clear(CGRect(x: 0, y: 0, width: canvasW, height: canvasH))
        ctx.textMatrix = .identity

        let foreground = style.tint.cgColor
        let black = CGColor(gray: 0.0, alpha: 1.0)
        // Memory row 0 is the top row; CoreText draws with y up, so line 0 sits
        // highest by placing it at the largest y.
        for (i, text) in lines.enumerated() {
            let attrs: [NSAttributedString.Key: Any] = [
                kCTFontAttributeName as NSAttributedString.Key: font,
                kCTForegroundColorAttributeName as NSAttributedString.Key: foreground,
            ]
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs))
            let width = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
            let x = (CGFloat(canvasW) - width) / 2
            let y = CGFloat(canvasH) - lineHeight * CGFloat(i + 1)
            // Stroke first, then fill: the outline must not eat into the glyphs.
            var stroke = attrs
            stroke[kCTStrokeColorAttributeName as NSAttributedString.Key] = black
            stroke[kCTStrokeWidthAttributeName as NSAttributedString.Key] = style.outline.width
            let outline = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: stroke))
            ctx.saveGState()
            ctx.setShadow(offset: CGSize(width: 0, height: -fontSize * 0.06),
                          blur: fontSize * 0.10, color: CGColor(gray: 0, alpha: 0.7))
            ctx.textPosition = CGPoint(x: x, y: y)
            CTLineDraw(outline, ctx)
            ctx.restoreGState()
            ctx.textPosition = CGPoint(x: x, y: y)
            CTLineDraw(line, ctx)
        }
        let img = ctx.makeImage()
        if let img {
            // Keep recent cues, rather than a whole film's output-resolution glyphs.
            if cache.count >= 8 { cache.removeAll(keepingCapacity: true) }
            cache[index] = img
        }
        return img
    }

    func previewImage() -> CGImage? {
        guard let context = CGContext(data: nil, width: sourceWidth, height: sourceHeight,
                bitsPerComponent: 8, bytesPerRow: sourceWidth * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
              let buffer = context.data else { return nil }
        context.setFillColor(CGColor(gray: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight))
        blend(into: buffer, bytesPerRow: context.bytesPerRow,
              width: sourceWidth, height: sourceHeight, time: 0)
        return context.makeImage()
    }

    // MARK: - Compositing

    func activeCueIndex(at time: Double) -> Int? {
        cues.firstIndex { time >= $0.start && time < $0.end }
    }

    func isActive(at time: Double) -> Bool { activeCueIndex(at: time) != nil }

    /// Where the active cue sits in this renderer's pixel space. Origin is the
    /// top-left of the frame and y grows downward, matching the picture buffer.
    func placement(at time: Double) -> (image: CGImage, rect: CGRect)? {
        guard let idx = activeCueIndex(at: time), let cg = image(for: idx) else { return nil }
        let cgW = cg.width, cgH = cg.height
        let bottomMargin = Int(Double(sourceHeight) * 0.05)
        guard cgH <= sourceHeight, cgW <= sourceWidth else { return nil }
        let desiredY = sourceHeight - bottomMargin - cgH - Int(Double(sourceHeight) * style.verticalOffset)
        let dstY = min(sourceHeight - cgH, max(0, desiredY))
        let desiredX = (sourceWidth - cgW) / 2 + Int(Double(sourceWidth) * style.horizontalOffset)
        let x0 = min(sourceWidth - cgW, max(0, desiredX))
        return (cg, CGRect(x: x0, y: dstY, width: cgW, height: cgH))
    }

    /// The Metal passes have completed before the CPU writes this output buffer.
    func blend(into pixelBuffer: CVPixelBuffer, time: Double) {
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        blend(into: base, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
              width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer), time: time)
    }

    /// Blends the active cue into a BGRA output frame. The stream encoder uses
    /// this; playback draws `placement(at:)` in a layer instead.
    func blend(into buffer: UnsafeMutableRawPointer, bytesPerRow: Int,
               width: Int, height: Int, time: Double) {
        guard let placed = placement(at: time) else { return }
        let cg = placed.image
        guard let data = cg.dataProvider?.data, let src = CFDataGetBytePtr(data) else { return }
        let cgW = cg.width, cgH = cg.height
        let x0 = Int(placed.rect.minX), dstY = Int(placed.rect.minY)
        guard dstY >= 0, x0 >= 0, dstY + cgH <= height, x0 + cgW <= width else { return }
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
