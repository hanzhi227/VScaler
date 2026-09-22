import Foundation
import CoreVideo
import Metal

/// Small rotation of IOSurface-backed BGRA output buffers, each pre-wrapped as
/// a shader-writable Metal texture.
///
/// Replaces CVPixelBufferPool, which returns kCVReturnInvalidPixelBufferAttributes
/// for every attribute set on this macOS build. Direct CVPixelBufferCreate +
/// CVMetalTextureCache wrapping is cheap (~0.07 ms for 4K here), so a fixed
/// rotation gives the same steady-state behaviour with pooled reuse.
final class OutputBufferRing {
    private struct Slot {
        let pb: CVPixelBuffer
        let cvTexture: CVMetalTexture
        let texture: MTLTexture
    }

    private var slots: [Slot] = []
    private var index = 0

    init(device: MTLDevice, width: Int, height: Int, depth: Int = 3) throws {
        var cacheRef: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cacheRef) == kCVReturnSuccess,
              let cache = cacheRef else {
            throw EngineError.pixelBufferFailure
        }
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        for _ in 0..<max(2, depth) {
            var pbRef: CVPixelBuffer?
            guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                      kCVPixelFormatType_32BGRA,
                                      attrs as CFDictionary, &pbRef) == kCVReturnSuccess,
                  let pb = pbRef else {
                throw EngineError.pixelBufferFailure
            }
            var cvRef: CVMetalTexture?
            guard CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, cache, pb, nil, .bgra8Unorm,
                width, height, 0, &cvRef) == kCVReturnSuccess,
                let cv = cvRef, let tex = CVMetalTextureGetTexture(cv) else {
                throw EngineError.pixelBufferFailure
            }
            slots.append(Slot(pb: pb, cvTexture: cv, texture: tex))
        }
    }

    /// Next buffer in rotation (caller must finish with it before depth calls).
    func next() -> (pb: CVPixelBuffer, texture: MTLTexture) {
        let s = slots[index]
        index = (index + 1) % slots.count
        return (s.pb, s.texture)
    }
}
