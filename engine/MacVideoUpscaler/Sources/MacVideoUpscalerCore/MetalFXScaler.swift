import Foundation
import Metal
import MetalFX

/// MetalFX Spatial (`MTLFXSpatialScaler`) — Apple's real-time spatial upscaler,
/// wired in as the general / live-action video option.
///
/// It is the only one of the shortlisted general-video upscalers that can run in
/// a live player: measured 0.33 ms for 640x480 -> 1280x960 and 0.99 ms for
/// 1920x1080 -> 3840x2160 on an M5 Pro, against 3.81 ms for the Anime4K CNN at
/// the latter size. It also takes the input and output geometry directly, so one
/// encode does the whole magnification — the 4x here is a single pass rather than
/// two chained 2x passes, which measured better on both axes (acutance 1.94x vs
/// 2.18x of the truth, PSNR 29.76 vs 28.80 dB), because the sharpening is
/// applied once instead of twice.
///
/// What it is *not* is the faithful option, which is what the evaluation set out
/// expecting. It carries a fixed sharpening stage with no knob to dial it back —
/// `MTLFXSpatialScaler` exposes geometry, formats and `colorProcessingMode`, and
/// nothing else — and measured against high-resolution references it overshoots:
/// acutance 1.54-2.18x the ground truth over four clips, the lowest PSNR/SSIM of
/// the four modes in every live-action run, and on animation *below plain
/// bilinear*. It reads best on material that is already sharp, and it is the mode
/// to choose when GPU cost or perceived crispness matters more than fidelity.
/// Method, numbers and crops: docs/GENERAL-VIDEO-UPSCALING.md.
///
/// Geometry is not known when the scaler is built (the factory is handed a kind
/// and a factor), so the underlying scaler is created on first use and rebuilt
/// if the frame geometry ever changes.
final class MetalFXScaler: FrameScaler {
    let label: String

    private let ctx: MetalContext
    private var scaler: MTLFXSpatialScaler?
    private var builtFor: Geometry?

    private struct Geometry: Equatable {
        let srcWidth: Int, srcHeight: Int
        let dstWidth: Int, dstHeight: Int
    }

    init(factor: UpscaleFactor, ctx: MetalContext) throws {
        self.ctx = ctx
        self.label = "metalfx-spatial-\(factor.rawValue)x"
    }

    func scale(src: MTLTexture, dst: MTLTexture) throws {
        let scaler = try scalerFor(src: src, dst: dst)
        scaler.colorTexture = src
        scaler.outputTexture = dst

        let desc = MTLCommandBufferDescriptor()
        desc.retainedReferences = false
        guard let cb = ctx.queue.makeCommandBuffer(descriptor: desc) else {
            throw EngineError.metalUnavailable
        }
        scaler.encode(commandBuffer: cb)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw err
        }
    }

    /// Creates the scaler for this geometry, reusing it while the geometry holds.
    /// At the 4x the app offers this is a *single* pass — MetalFX accepts the
    /// full magnification, so 4x is not two chained 2x passes the way the Anime4K
    /// path has to be. That also means the sharpening is applied once rather than
    /// twice, which matters on textured material.
    private func scalerFor(src: MTLTexture, dst: MTLTexture) throws -> MTLFXSpatialScaler {
        let g = Geometry(srcWidth: src.width, srcHeight: src.height,
                         dstWidth: dst.width, dstHeight: dst.height)
        if let scaler, builtFor == g { return scaler }

        let d = MTLFXSpatialScalerDescriptor()
        d.inputWidth = g.srcWidth
        d.inputHeight = g.srcHeight
        d.outputWidth = g.dstWidth
        d.outputHeight = g.dstHeight
        d.colorTextureFormat = src.pixelFormat
        d.outputTextureFormat = dst.pixelFormat
        // The pipeline carries display-ready, gamma-encoded BGRA, which is what
        // MetalFX calls perceptual — declaring it linear would have the scaler
        // filter in the wrong space. Measured on a real decoded frame, perceptual
        // is also the better of the two: acutance 1.69x vs 1.86x of the truth and
        // PSNR 32.21 vs 32.07 dB.
        d.colorProcessingMode = .perceptual

        guard let made = d.makeSpatialScaler(device: ctx.device) else {
            throw EngineError.metalFXUnavailable(
                "MetalFX Spatial cannot scale \(g.srcWidth)x\(g.srcHeight) to "
                    + "\(g.dstWidth)x\(g.dstHeight) on this GPU")
        }
        scaler = made
        builtFor = g
        return made
    }
}
