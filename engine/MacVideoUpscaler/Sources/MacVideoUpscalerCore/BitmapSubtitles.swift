import CoreGraphics
import Foundation

/// Timed DVD/VobSub images for the playback overlay.
///
/// Bitmap cues used to be composited by an ffmpeg `overlay` filter on every
/// decoded frame, which pulled the picture through a software filter graph
/// before the upscaler ever saw it. The packets are decoded once here. Each
/// cue is a small image placed over the picture, and it is shown again only
/// when the clock enters that cue.
final class BitmapSubtitleSequence {
    struct Event {
        let start: Double
        let end: Double
        let image: CGImage
        /// Tight bounds on the subtitle canvas. Origin is top-left, y down.
        let rect: CGRect
    }

    let canvasWidth: Int
    let canvasHeight: Int
    /// Video raster before the display rotation, which is the plane ffmpeg used
    /// to scale the subtitle canvas onto.
    let unrotatedWidth: Int
    let unrotatedHeight: Int
    /// Same convention as `FFmpegDecoder.rotateFilter`: 90 is counter-clockwise.
    let rotation: Int
    let events: [Event]

    init(canvasWidth: Int, canvasHeight: Int, unrotatedWidth: Int, unrotatedHeight: Int,
         rotation: Int, events: [Event]) {
        self.canvasWidth = max(1, canvasWidth)
        self.canvasHeight = max(1, canvasHeight)
        self.unrotatedWidth = max(1, unrotatedWidth)
        self.unrotatedHeight = max(1, unrotatedHeight)
        self.rotation = rotation
        self.events = events
    }

    func eventIndex(at time: Double) -> Int? {
        events.firstIndex { time >= $0.start && time < $0.end }
    }

    /// Position in output pixels, top-left origin, y down.
    func placement(at time: Double, style: SubtitleStyle, outputWidth: Int, outputHeight: Int)
        -> (image: CGImage, rect: CGRect)? {
        guard outputWidth > 0, outputHeight > 0, let index = eventIndex(at: time) else { return nil }
        let event = events[index]
        let quarter = rotation == 90 || rotation == 270
        let displayW = quarter ? unrotatedHeight : unrotatedWidth
        let displayH = quarter ? unrotatedWidth : unrotatedHeight
        let sx = CGFloat(unrotatedWidth) / CGFloat(canvasWidth)
        let sy = CGFloat(unrotatedHeight) / CGFloat(canvasHeight)
        var rect = CGRect(x: event.rect.minX * sx, y: event.rect.minY * sy,
                          width: event.rect.width * sx, height: event.rect.height * sy)
        let image = Self.rotate(event.image, rotation: rotation)
        switch rotation {
        case 90:
            let x = rect.minY
            let y = CGFloat(unrotatedWidth) - rect.maxX
            rect = CGRect(x: x, y: y, width: rect.height, height: rect.width)
        case 180:
            rect = CGRect(x: CGFloat(unrotatedWidth) - rect.maxX,
                          y: CGFloat(unrotatedHeight) - rect.maxY,
                          width: rect.width, height: rect.height)
        case 270:
            let x = CGFloat(unrotatedHeight) - rect.maxY
            let y = rect.minX
            rect = CGRect(x: x, y: y, width: rect.height, height: rect.width)
        default:
            break
        }
        let ox = CGFloat(outputWidth) / CGFloat(max(1, displayW))
        let oy = CGFloat(outputHeight) / CGFloat(max(1, displayH))
        rect = CGRect(x: (rect.minX * ox).rounded(), y: (rect.minY * oy).rounded(),
                      width: (rect.width * ox).rounded(), height: (rect.height * oy).rounded())
        let dx = (Double(outputWidth) * style.normalized.horizontalOffset).rounded()
        let dy = (Double(outputHeight) * style.normalized.verticalOffset).rounded()
        // Shift in display space and let drawing clip. Pulling the cue back
        // inside the frame would shorten the move whenever it meets an edge.
        rect.origin.x += dx
        rect.origin.y -= dy
        guard rect.width >= 1, rect.height >= 1 else { return nil }
        return (image, rect)
    }

    /// Embedded `dvd_subtitle` packets, or a VobSub `.idx` (ffmpeg reads the
    /// matching `.sub`). `ordinal` is the subtitle stream's position in the file.
    static func load(url: URL, ordinal: Int, unrotatedWidth: Int, unrotatedHeight: Int,
                     rotation: Int) -> BitmapSubtitleSequence? {
        guard let header = headerText(url: url, ordinal: ordinal),
              let canvas = parseSize(header) else { return nil }
        let palette = parsePalette(header) ?? Array(repeating: (255, 255, 255), count: 16)
        let origin = parseOrigin(header)
        guard let stamps = packetTimes(url: url, ordinal: ordinal), !stamps.isEmpty,
              let blob = packetBytes(url: url, ordinal: ordinal) else { return nil }
        var packets: [Data] = []
        var cursor = 0
        while cursor + 2 <= blob.count {
            let size = Int(blob[cursor]) << 8 | Int(blob[cursor + 1])
            guard size >= 4, cursor + size <= blob.count else { break }
            packets.append(blob.subdata(in: cursor..<(cursor + size)))
            cursor += size
        }
        guard packets.count == stamps.count else { return nil }
        var events: [Event] = []
        for (packet, stamp) in zip(packets, stamps) {
            guard let decoded = decodeSPU(packet, palette: palette, origin: origin) else { continue }
            let start = stamp.pts + decoded.startOffset
            let end = stamp.pts + (decoded.endOffset ?? stamp.duration ?? decoded.startOffset + 5)
            guard end > start else { continue }
            events.append(Event(start: start, end: end, image: decoded.image, rect: decoded.rect))
        }
        guard !events.isEmpty else { return nil }
        return BitmapSubtitleSequence(canvasWidth: canvas.0, canvasHeight: canvas.1,
                                      unrotatedWidth: unrotatedWidth, unrotatedHeight: unrotatedHeight,
                                      rotation: rotation, events: events)
    }

    // MARK: - SPU

    private struct Decoded {
        let image: CGImage
        let rect: CGRect
        let startOffset: Double
        let endOffset: Double?
    }

    /// DVD subtitle time units: the control date is in 1024/90 kHz steps.
    private static let timeUnit = 1024.0 / 90000.0

    private static func decodeSPU(_ data: Data, palette: [(Int, Int, Int)], origin: (Int, Int)) -> Decoded? {
        guard data.count >= 4 else { return nil }
        let ctrl = Int(data[2]) << 8 | Int(data[3])
        guard ctrl >= 4, ctrl + 4 <= data.count else { return nil }
        var pos = ctrl
        var seen = Set<Int>()
        var colors = [0, 0, 0, 0]
        var alphas = [0, 0, 0, 0]
        var sx = 0, ex = 0, sy = 0, ey = 0
        var top = 0, bottom = 0
        var startDate = 0
        var stopDate: Int?
        var haveRect = false
        while pos + 4 <= data.count, seen.insert(pos).inserted {
            let date = Int(data[pos]) << 8 | Int(data[pos + 1])
            let next = Int(data[pos + 2]) << 8 | Int(data[pos + 3])
            var p = pos + 4
            while p < data.count {
                let cmd = data[p]
                p += 1
                if cmd == 0xFF { break }
                switch cmd {
                case 0x00:
                    break
                case 0x01:
                    startDate = date
                case 0x02:
                    stopDate = date
                case 0x03, 0x04:
                    guard p + 2 <= data.count else { return nil }
                    let b0 = Int(data[p]), b1 = Int(data[p + 1])
                    p += 2
                    let nibbles = [b1 & 0xF, (b1 >> 4) & 0xF, b0 & 0xF, (b0 >> 4) & 0xF]
                    if cmd == 0x03 { colors = nibbles } else { alphas = nibbles }
                case 0x05:
                    guard p + 6 <= data.count else { return nil }
                    let b = (0..<6).map { Int(data[p + $0]) }
                    p += 6
                    sx = (b[0] << 4) | (b[1] >> 4)
                    ex = ((b[1] & 0xF) << 8) | b[2]
                    sy = (b[3] << 4) | (b[4] >> 4)
                    ey = ((b[4] & 0xF) << 8) | b[5]
                    haveRect = true
                case 0x06:
                    guard p + 4 <= data.count else { return nil }
                    top = Int(data[p]) << 8 | Int(data[p + 1])
                    bottom = Int(data[p + 2]) << 8 | Int(data[p + 3])
                    p += 4
                default:
                    return nil
                }
            }
            if next <= pos { break }
            pos = next
        }
        guard haveRect, ex >= sx, ey >= sy else { return nil }
        let width = ex - sx + 1
        let height = ey - sy + 1
        guard width > 0, height > 0, width < 8000, height < 8000 else { return nil }
        var index = [UInt8](repeating: 0, count: width * height)
        decodeRLE(data, offset: top, width: width, height: height, y0: 0, into: &index)
        decodeRLE(data, offset: bottom, width: width, height: height, y0: 1, into: &index)
        guard let made = image(index: index, width: width, height: height, colors: colors,
                               alphas: alphas, palette: palette) else { return nil }
        let rect = CGRect(x: CGFloat(sx + origin.0) + made.crop.minX,
                          y: CGFloat(sy + origin.1) + made.crop.minY,
                          width: made.crop.width, height: made.crop.height)
        return Decoded(image: made.image, rect: rect,
                       startOffset: Double(startDate) * timeUnit,
                       endOffset: stopDate.map { Double($0) * timeUnit })
    }

    private static func decodeRLE(_ data: Data, offset: Int, width: Int, height: Int, y0: Int,
                                  into index: inout [UInt8]) {
        guard offset >= 0, offset < data.count else { return }
        var npos = offset * 2
        let limit = data.count * 2
        var y = y0
        var x = 0
        func nibble() -> Int {
            guard npos < limit else { return 0 }
            let byte = Int(data[npos >> 1])
            let value = (npos & 1) == 0 ? (byte >> 4) : (byte & 0xF)
            npos += 1
            return value
        }
        while y < height && npos < limit {
            var v = nibble()
            if v < 0x4 {
                v = (v << 4) | nibble()
                if v < 0x10 {
                    v = (v << 4) | nibble()
                    if v < 0x40 {
                        v = (v << 4) | nibble()
                        if v < 0x4 { v |= (width - x) << 2 }
                    }
                }
            }
            let color = v & 3
            var run = v >> 2
            if run <= 0 { run = width - x }
            let count = min(run, width - x)
            if count > 0, y < height {
                let row = y * width + x
                for i in 0..<count { index[row + i] = UInt8(color) }
                x += count
            }
            if x >= width {
                if npos & 1 == 1 { npos += 1 }
                x = 0
                y += 2
            }
        }
    }

    private static func image(index: [UInt8], width: Int, height: Int, colors: [Int], alphas: [Int],
                              palette: [(Int, Int, Int)])
        -> (image: CGImage, crop: CGRect)? {
        var minX = width, minY = height, maxX = -1, maxY = -1
        for y in 0..<height {
            for x in 0..<width where alphas[Int(index[y * width + x])] > 0 {
                minX = min(minX, x); minY = min(minY, y)
                maxX = max(maxX, x); maxY = max(maxY, y)
            }
        }
        guard maxX >= minX, maxY >= minY else { return nil }
        let cropW = maxX - minX + 1
        let cropH = maxY - minY + 1
        var bytes = [UInt8](repeating: 0, count: cropW * cropH * 4)
        for y in 0..<cropH {
            for x in 0..<cropW {
                let sample = Int(index[(y + minY) * width + (x + minX)])
                let alpha = alphas[sample] * 17
                guard alpha > 0 else { continue }
                let entry = palette[colors[sample] & 15]
                let i = (y * cropW + x) * 4
                bytes[i] = UInt8(entry.2 * alpha / 255)
                bytes[i + 1] = UInt8(entry.1 * alpha / 255)
                bytes[i + 2] = UInt8(entry.0 * alpha / 255)
                bytes[i + 3] = UInt8(alpha)
            }
        }
        let crop = CGRect(x: minX, y: minY, width: cropW, height: cropH)
        guard let image = bgraImage(bytes, width: cropW, height: cropH) else { return nil }
        return (image, crop)
    }

    private static func rotate(_ image: CGImage, rotation: Int) -> CGImage {
        guard rotation == 90 || rotation == 180 || rotation == 270 else { return image }
        let w = image.width, h = image.height
        let swap = rotation == 90 || rotation == 270
        let nw = swap ? h : w
        let nh = swap ? w : h
        guard let data = image.dataProvider?.data, let src = CFDataGetBytePtr(data) else { return image }
        var bytes = [UInt8](repeating: 0, count: nw * nh * 4)
        for y in 0..<h {
            for x in 0..<w {
                let nx: Int
                let ny: Int
                switch rotation {
                case 90: nx = y; ny = w - 1 - x
                case 180: nx = w - 1 - x; ny = h - 1 - y
                default: nx = h - 1 - y; ny = x
                }
                let s = (y * w + x) * 4
                let d = (ny * nw + nx) * 4
                bytes[d] = src[s]
                bytes[d + 1] = src[s + 1]
                bytes[d + 2] = src[s + 2]
                bytes[d + 3] = src[s + 3]
            }
        }
        return bgraImage(bytes, width: nw, height: nh) ?? image
    }

    private static func bgraImage(_ bytes: [UInt8], width: Int, height: Int) -> CGImage? {
        let row = width * 4
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: row, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue),
              let base = ctx.data else { return nil }
        bytes.withUnsafeBytes { raw in
            guard let address = raw.baseAddress else { return }
            base.copyMemory(from: address, byteCount: min(bytes.count, row * height))
        }
        return ctx.makeImage()
    }

    // MARK: - Container

    private static func headerText(url: URL, ordinal: Int) -> String? {
        if url.pathExtension.lowercased() == "idx",
           let text = try? String(contentsOf: url, encoding: .utf8), text.contains("size:") {
            return text
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: FFmpegDecoder.ffprobePath)
        p.arguments = ["-v", "error", "-select_streams", "s:\(ordinal)",
                       "-show_entries", "stream=extradata", "-show_data", "-of", "json", url.path]
        return toolText(p).flatMap(decodeExtradata)
    }

    /// ffprobe `-show_data` prints `offset: hhhh hhhh ...  ascii`. The codec
    /// private data of a DVD subtitle is the VobSub header, as text.
    private static func decodeExtradata(_ json: String) -> String? {
        var bytes = [UInt8]()
        for line in json.split(separator: "\n") {
            guard let range = line.range(of: ": ") else { continue }
            let hex = line[range.upperBound...].split(separator: "  ", maxSplits: 1).first ?? Substring()
            for token in hex.split(separator: " ") where token.count == 4 && token.allSatisfy(\.isHexDigit) {
                bytes.append(UInt8(token.prefix(2), radix: 16) ?? 0)
                bytes.append(UInt8(token.suffix(2), radix: 16) ?? 0)
            }
        }
        guard !bytes.isEmpty else { return nil }
        return String(bytes: bytes, encoding: .utf8)
    }

    private static func packetTimes(url: URL, ordinal: Int) -> [(pts: Double, duration: Double?)]? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: FFmpegDecoder.ffprobePath)
        p.arguments = ["-v", "error", "-select_streams", "s:\(ordinal)",
                       "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0", url.path]
        guard let text = toolText(p) else { return nil }
        var stamps: [(Double, Double?)] = []
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: ",", omittingEmptySubsequences: false)
            guard let pts = Double(parts.first ?? "") else { continue }
            let duration = parts.count > 1 ? Double(parts[1]) : nil
            stamps.append((pts, duration))
        }
        return stamps
    }

    private static func packetBytes(url: URL, ordinal: Int) -> Data? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: FFmpegDecoder.ffmpegPath)
        p.arguments = ["-v", "error", "-i", url.path, "-map", "0:s:\(ordinal)",
                       "-c", "copy", "-f", "data", "pipe:1"]
        return toolData(p)
    }

    private static func parseSize(_ text: String) -> (Int, Int)? {
        guard let line = text.split(separator: "\n").first(where: { $0.lowercased().hasPrefix("size:") }) else { return nil }
        let parts = line.dropFirst(5).split(separator: "x")
        guard parts.count == 2, let w = Int(parts[0].trimmingCharacters(in: .whitespaces)),
              let h = Int(parts[1].trimmingCharacters(in: .whitespaces)), w > 0, h > 0 else { return nil }
        return (w, h)
    }

    private static func parseOrigin(_ text: String) -> (Int, Int) {
        guard let line = text.split(separator: "\n").first(where: { $0.lowercased().hasPrefix("org:") }) else { return (0, 0) }
        let parts = line.dropFirst(4).split(separator: ",")
        guard parts.count == 2, let x = Int(parts[0].trimmingCharacters(in: .whitespaces)),
              let y = Int(parts[1].trimmingCharacters(in: .whitespaces)) else { return (0, 0) }
        return (x, y)
    }

    private static func parsePalette(_ text: String) -> [(Int, Int, Int)]? {
        guard let line = text.split(separator: "\n").first(where: { $0.lowercased().hasPrefix("palette:") }) else { return nil }
        let words = line.dropFirst(8).split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        guard words.count >= 16 else { return nil }
        return words.prefix(16).map { word in
            let hex = String(word)
            guard hex.count == 6, let value = Int(hex, radix: 16) else { return (255, 255, 255) }
            return ((value >> 16) & 255, (value >> 8) & 255, value & 255)
        }
    }

    private static func toolText(_ p: Process) -> String? {
        guard let data = toolData(p), let text = String(data: data, encoding: .utf8), !text.isEmpty else { return nil }
        return text
    }

    private static func toolData(_ p: Process) -> Data? {
        let out = Pipe()
        let err = Pipe()
        p.standardOutput = out
        p.standardError = err
        p.standardInput = FileHandle.nullDevice
        guard (try? p.run()) != nil else { return nil }
        DispatchQueue.global(qos: .utility).async { _ = err.fileHandleForReading.readDataToEndOfFile() }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        Subprocess.wait(for: p, timeout: 20)
        guard !p.isRunning, p.terminationStatus == 0, !data.isEmpty else { return nil }
        return data
    }
}
