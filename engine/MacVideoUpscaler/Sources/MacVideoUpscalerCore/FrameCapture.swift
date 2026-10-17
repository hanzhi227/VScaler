import Foundation
import CoreVideo
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// One subtitle image placed over the picture. Coordinates are output pixels
/// with a top-left origin and y growing downward.
struct SubtitleOverlayFrame {
    let image: CGImage
    let rect: CGRect
    let videoWidth: Int
    let videoHeight: Int
    let key: String
}

/// Writes a BGRA pixel buffer to a PNG file.
///
/// Diagnostics only: used by the frame-dump test hook to prove what the player
/// actually rendered (the environment running these tests has no Screen
/// Recording permission, so a window screenshot is not available).
public enum FrameCapture {
    @discardableResult
    public static func writePNG(_ pixelBuffer: CVPixelBuffer, to url: URL) -> Bool {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return false }
        // 32BGRA little-endian: skip the (unused) alpha byte, keep channel order.
        let bitmapInfo = CGBitmapInfo(rawValue:
            CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let ctx = CGContext(data: base, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: bitmapInfo.rawValue),
              let image = ctx.makeImage() else { return false }
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return false }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest)
    }

    @discardableResult
    public static func writePNG(_ image: CGImage, to url: URL) -> Bool {
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return false }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest)
    }

    static func image(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bitmapInfo = CGBitmapInfo(rawValue:
            CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let ctx = CGContext(data: base, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: bitmapInfo.rawValue) else { return nil }
        return ctx.makeImage()
    }

    /// Picture plus the overlay. Used when a snapshot has to show what the
    /// viewer sees; playback itself never writes the cue into the frame.
    static func composite(base: CVPixelBuffer?, overlay: SubtitleOverlayFrame?,
                          width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0 else { return nil }
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        if let base, let image = image(from: base) {
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        if let overlay, overlay.videoWidth > 0, overlay.videoHeight > 0 {
            let sx = CGFloat(width) / CGFloat(overlay.videoWidth)
            let sy = CGFloat(height) / CGFloat(overlay.videoHeight)
            let rect = overlay.rect
            let placed = CGRect(x: rect.minX * sx,
                                y: CGFloat(height) - rect.maxY * sy,
                                width: rect.width * sx, height: rect.height * sy)
            ctx.draw(overlay.image, in: placed)
        }
        return ctx.makeImage()
    }
}
