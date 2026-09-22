import Foundation
import Metal
import CoreVideo

/// A 2x-only frame scaler. Both back ends are fixed at 2x: the Anime4K CNN
/// models output exactly 2x, and the bilinear kernel hard-codes a 0.5 step.
protocol TwoXScaler: AnyObject {
    var label: String { get }
    func scale(src: MTLTexture, dst: MTLTexture) throws
}

/// Selectable output magnification.
///
/// There is no native 3x or 4x kernel: Anime4K ships the CNN at 2x and its own
/// documented 4x setup is two chained passes, so 4x here is the 2x scaler
/// applied twice (through a reused intermediate). A 3x mode would need a
/// different model, so it is deliberately not offered.
public enum UpscaleFactor: Int, CaseIterable, Codable, Sendable, Comparable {
    case x2 = 2
    case x4 = 4

    public var displayName: String { "\(rawValue)×" }
    public static func < (a: UpscaleFactor, b: UpscaleFactor) -> Bool { a.rawValue < b.rawValue }
}

/// Scales a frame to `factor` times the source size.
protocol FrameScaler: AnyObject {
    var label: String { get }
    func scale(src: MTLTexture, dst: MTLTexture) throws
}

/// Applies a 2x scaler once (2x) or twice (4x). The 4x intermediate is
/// allocated once and reused, so the steady state allocates nothing per frame.
final class ChainedScaler: FrameScaler {
    let label: String
    private let base: TwoXScaler
    private let factor: UpscaleFactor
    private let device: MTLDevice
    private var mid: MTLTexture?

    init(base: TwoXScaler, factor: UpscaleFactor, device: MTLDevice) {
        self.base = base
        self.factor = factor
        self.device = device
        self.label = factor == .x2 ? base.label : "\(base.label) x2 (chained)"
    }

    func scale(src: MTLTexture, dst: MTLTexture) throws {
        guard factor == .x4 else {
            try base.scale(src: src, dst: dst)
            return
        }
        let mw = src.width * 2, mh = src.height * 2
        if mid == nil || mid?.width != mw || mid?.height != mh {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: mw, height: mh, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            mid = device.makeTexture(descriptor: d)
        }
        guard let m = mid else { throw EngineError.metalUnavailable }
        try base.scale(src: src, dst: m)
        try base.scale(src: m, dst: dst)
    }
}

public enum ScalerKind: String, CaseIterable, Codable {
    case off         // no scaling and no model: the source, pixel for pixel
    case bypass      // plain bilinear ("enhancement off", but still scaled)
    case cnn         // Anime4K CNN (Metal port)
    case cnnDenoise  // Anime4K CNN + denoise
    case metalFX     // MetalFX Spatial — general (live-action) video
    case realESRGAN  // Real-ESRGAN animevideov3 (Metal/MPSGraph port), 4x only
    case realESRGANGeneral  // Real-ESRGAN general-x4v3, same port, 4x only

    public var displayName: String {
        switch self {
        case .off: return "Off (no scaling)"
        case .bypass: return "Bypass (bilinear)"
        case .cnn: return "Anime4K CNN"
        case .cnnDenoise: return "Anime4K CNN + denoise"
        case .metalFX: return "MetalFX Spatial"
        case .realESRGAN: return "Real-ESRGAN AnimeVideo-v3 (4×)"
        case .realESRGANGeneral: return "Real-ESRGAN General-x4 (4×)"
        }
    }

    /// One line of guidance for the Enhancement menu, so what a model is *for* is
    /// visible where it is chosen rather than only in the docs.
    public var hint: String {
        switch self {
        case .off:
            return "Off — no scaling and no processing: the decoded frame is "
                + "presented as it is, at its own resolution"
        case .bypass:
            return "No enhancement, but still scaled — plain bilinear, what another "
                + "player would show"
        case .cnn:
            return "Anime4K CNN — best for animation"
        case .cnnDenoise:
            return "Anime4K CNN + denoise — for noisy or heavily compressed animation"
        case .metalFX:
            return "MetalFX Spatial — for live action and clean HD: crisp and "
                + "much cheaper than the CNN, but it sharpens past the source"
        case .realESRGAN:
            return "Real-ESRGAN AnimeVideo-v3 — a generative model trained on anime "
                + "video, 4× and SD only. Sharp on clean animation, but it rebuilds "
                + "fine texture as smooth blobs and expands bright highlights, which "
                + "shows on real (non-anime) video; the Anime4K CNN and MetalFX keep "
                + "that detail better"
        case .realESRGANGeneral:
            return "Real-ESRGAN General-x4 — the same network, twice as deep, "
                + "trained on general photographs rather than anime. Kept as an "
                + "option to judge against AnimeVideo-v3 on your own footage: in "
                + "this project's scores it loses to that one on every clip and "
                + "column tested, for 1.8× the time"
        }
    }

    /// The kinds whose network exists at a single scale.
    ///
    /// `SRVGGNetCompact` is released for video as x4 only — there is no x2
    /// checkpoint for either variant — so unlike the 2x CNN kernels these modes
    /// cannot be chained to reach 2x, and asking for 2x must not silently produce
    /// a picture that is half the magnification the other modes would give. A
    /// session either runs it at 4x or refuses.
    public var pinnedFactor: UpscaleFactor? {
        switch self {
        case .realESRGAN, .realESRGANGeneral: return .x4
        default: return nil
        }
    }

    /// `off` is the one kind that does not magnify, so the session sizes its
    /// output to the source instead of to a factor.
    public var magnifies: Bool { self != .off }

    /// Parses the CLI / environment spelling of a mode.
    ///
    /// Shared by the app, the stream CLI and the test harness so the accepted
    /// names cannot drift apart between them; an unrecognised value keeps the
    /// default rather than failing the run. `bilinear`/`original` mean the
    /// scaled-but-unenhanced bypass; `off`/`none` mean no scaling at all.
    public static func parse(_ name: String?) -> ScalerKind {
        switch name ?? "" {
        case "off", "none", "native", "unscaled": return .off
        case "bypass", "bilinear", "original": return .bypass
        case "cnnDenoise", "denoise": return .cnnDenoise
        case "metalfx", "metal", "spatial": return .metalFX
        case "realesrgan", "real-esrgan", "esrgan", "realesr": return .realESRGAN
        case "realesrgan-general", "esrgan-general", "general-x4v3", "general":
            return .realESRGANGeneral
        case "cnn", "anime4k": return .cnn
        default: return defaultKind
        }
    }

    /// What a run uses when nothing asks for a specific mode. MetalFX: it is the
    /// cheapest of the models by ~4x and the general-video option, so it is the
    /// safest thing to do to a file the viewer has not looked at yet.
    public static let defaultKind: ScalerKind = .metalFX
}

enum ScalerFactory {
    static func make(kind: ScalerKind, factor: UpscaleFactor, ctx: MetalContext) throws -> FrameScaler {
        // The Anime4K/bilinear back ends are 2x kernels, so 4x chains two of them.
        // MetalFX takes the output geometry directly, so it needs no chaining: one
        // encode does the whole magnification, which also means its sharpening is
        // applied once rather than twice. `off` does not magnify at all, and
        // Real-ESRGAN is a 4x network in its own right — chaining it would run the
        // whole model twice.
        switch kind {
        case .off:
            return try CopyScaler(ctx: ctx)
        case .bypass:
            return ChainedScaler(base: BilinearScaler(ctx: ctx), factor: factor, device: ctx.device)
        case .cnn:
            return ChainedScaler(base: try Anime4KScaler(mode: .cnnX2M, ctx: ctx),
                                 factor: factor, device: ctx.device)
        case .cnnDenoise:
            return ChainedScaler(base: try Anime4KScaler(mode: .cnnDenoiseX2M, ctx: ctx),
                                 factor: factor, device: ctx.device)
        case .metalFX:
            return try MetalFXScaler(factor: factor, ctx: ctx)
        case .realESRGAN:
            return try RealESRGANScaler(model: RealESRGANAnimeV3Weights.model,
                                        factor: factor, ctx: ctx)
        case .realESRGANGeneral:
            return try RealESRGANScaler(model: RealESRGANGeneralWeights.model,
                                        factor: factor, ctx: ctx)
        }
    }

    /// Largest factor the hardware H.264 encoder can accept.
    ///
    /// h264_videotoolbox fails to create a compression session above
    /// 4096x2304 (verified: 4096x2304 OK, 5120x2880 fails with -12903), so a
    /// requested 4x is clamped to 2x for sources that would exceed it — i.e.
    /// anything wider than 1024 or taller than 576.
    static func maxFactor(forWidth w: Int, height h: Int) -> UpscaleFactor {
        (w * 4 <= 4096 && h * 4 <= 2304) ? .x4 : .x2
    }

    /// The factor a session will really use, shared by both pipelines so the
    /// live player and the stream server cannot disagree about it.
    ///
    /// Returns nil when the mode cannot run on this source at all. That case is
    /// new with Real-ESRGAN: every previous model was either a 2x kernel that
    /// could be chained to 4x or a scaler that took the output geometry
    /// directly, so a request could always be met by clamping. A model pinned to
    /// one factor can only be met at that factor, and clamping it would quietly
    /// show a different magnification than the one asked for.
    static func resolveFactor(kind: ScalerKind, requested: UpscaleFactor,
                              width: Int, height: Int) -> UpscaleFactor? {
        let ceiling = maxFactor(forWidth: width, height: height)
        if let pinned = kind.pinnedFactor {
            return pinned <= ceiling ? pinned : nil
        }
        return min(requested, ceiling)
    }

    /// Why `resolveFactor` returned nil, in terms a viewer can act on.
    static func pinnedFactorRefusal(kind: ScalerKind, width: Int, height: Int) -> String {
        let f = kind.pinnedFactor ?? .x4
        return "\(kind.displayName) is \(f.displayName) only, and \(width * f.rawValue)×"
            + "\(height * f.rawValue) is past the H.264 encoder's 4096×2304 limit. "
            + "It runs on sources up to 1024×576 — choose another mode for this one."
    }

    static var cnnAvailable: Bool {
        #if HAVE_ANIME4K_METAL
        return true
        #else
        return false
        #endif
    }
}

/// Engine's own bilinear scaler (used for the bypass path; no external deps).
final class BilinearScaler: TwoXScaler {
    let label = "bilinear"
    private let ctx: MetalContext
    private let pipeline: MTLComputePipelineState

    init(ctx: MetalContext) {
        self.ctx = ctx
        let fn = try! ctx.function("bilinear2x")
        self.pipeline = try! ctx.device.makeComputePipelineState(function: fn)
    }

    func scale(src: MTLTexture, dst: MTLTexture) throws {
        let desc = MTLCommandBufferDescriptor()
        desc.retainedReferences = false
        guard let cb = ctx.queue.makeCommandBuffer(descriptor: desc),
              let enc = cb.makeComputeCommandEncoder() else {
            throw EngineError.metalUnavailable
        }
        enc.setComputePipelineState(pipeline)
        enc.setTexture(src, index: 0)
        enc.setTexture(dst, index: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(
            width: (dst.width + tg.width - 1) / tg.width,
            height: (dst.height + tg.height - 1) / tg.height,
            depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw err
        }
    }
}

/// The "off" position: no scaling and no processing. The decoded frame is copied
/// pixel for pixel at the source's own geometry, so whatever magnifying happens
/// to make it fit the window is the display layer's, not this app's.
///
/// This is deliberately not the same thing as `bypass`. Bypass still scales —
/// bilinearly, to the requested 2x/4x — so it does real GPU work and is what
/// invents the interpolated pixels. Off does neither, and it is the only mode
/// whose output is exactly the decoded source.
final class CopyScaler: FrameScaler {
    let label = "off"
    private let ctx: MetalContext
    private let pipeline: MTLComputePipelineState

    init(ctx: MetalContext) throws {
        self.ctx = ctx
        self.pipeline = try ctx.device.makeComputePipelineState(function: ctx.function("copyTexel"))
    }

    func scale(src: MTLTexture, dst: MTLTexture) throws {
        // A 1:1 copy is only defined at matching sizes. The session sizes the
        // output from the source when this scaler is chosen, so a mismatch means
        // the geometry was computed for a magnifying mode and this would read
        // past the source.
        guard src.width == dst.width, src.height == dst.height else {
            throw EngineError.scalerGeometryMismatch(
                "off is 1:1 but got \(src.width)x\(src.height) -> \(dst.width)x\(dst.height)")
        }
        let desc = MTLCommandBufferDescriptor()
        desc.retainedReferences = false
        guard let cb = ctx.queue.makeCommandBuffer(descriptor: desc),
              let enc = cb.makeComputeCommandEncoder() else {
            throw EngineError.metalUnavailable
        }
        enc.setComputePipelineState(pipeline)
        enc.setTexture(src, index: 0)
        enc.setTexture(dst, index: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(
            width: (dst.width + tg.width - 1) / tg.width,
            height: (dst.height + tg.height - 1) / tg.height,
            depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw err
        }
    }
}

#if HAVE_ANIME4K_METAL
import Anime4KMetal

/// Anime4K v3.2 CNN x2, Metal compute port (see engine/Anime4KMetal).
final class Anime4KScaler: TwoXScaler {
    let label: String
    private let pipeline: Anime4KPipeline

    init(mode: UpscaleMode, ctx: MetalContext) throws {
        self.pipeline = try Anime4KPipeline(device: ctx.device, mode: mode)
        self.label = "anime4k-\(mode.rawValue)"
    }

    func scale(src: MTLTexture, dst: MTLTexture) throws {
        try pipeline.upscale(src, into: dst)
    }
}
#endif

import RealESRGANMetal

/// Real-ESRGAN as a live mode, Metal port (see engine/RealESRGANMetal).
///
/// A *generative* model: `SRVGGNetCompact`, 3x3 convolutions with PReLU. Both
/// released video checkpoints run through this one class — `animevideov3`, 18
/// convolutions and 621,424 parameters, trained adversarially on anime video, and
/// `general-x4v3`, 34 and 1,213,296, trained on general photographs. Which one
/// runs is the `RealESRGANModel` handed to the pipeline.
///
/// The convolutions run through MPSGraph and two small Metal kernels handle the
/// frame conversion and the pixel shuffle, which is 3x faster here than the same
/// model run eagerly through PyTorch/MPS because the graph fuses conv+bias+PReLU
/// where eager dispatch cannot — measured 7.33 ms at 480x270 -> 1920x1080 against
/// 22.59 ms, both on this machine.
///
/// They take the output geometry directly rather than being chained from 2x
/// kernels, because x4 is the only scale these networks have.
final class RealESRGANScaler: FrameScaler {
    let label: String
    private let pipeline: RealESRGANPipeline

    init(model: RealESRGANModel, factor: UpscaleFactor, ctx: MetalContext) throws {
        // `resolveFactor` pins these modes to 4x before the factory is reached, so
        // this is a guard against a future caller rather than a live path.
        guard factor == .x4 else {
            throw EngineError.scalerGeometryMismatch(
                "real-esrgan is 4x only (there is no x2 checkpoint), got \(factor.displayName)")
        }
        // `general-x4v3` is a mode so it can be judged on a real clip, not only on
        // the scores; it is the weaker of the two on every column measured here at
        // 1.84x the time. See `scripts/realesrgan/README.md`.
        self.pipeline = try RealESRGANPipeline(model: model, device: ctx.device)
        self.label = "real-esrgan-\(model.name)-4x"
    }

    func scale(src: MTLTexture, dst: MTLTexture) throws {
        try pipeline.upscale(src, into: dst)
    }
}

/// Composes the side-by-side comparison: the left of `split` comes from a second
/// scaler's output, the right keeps what the main pipeline produced, and a thin
/// bright divider marks the boundary.
///
/// Each side is produced by a real `FrameScaler`, so a half is *exactly* what
/// that model outputs (not an approximation of it), and any pair of models can be
/// compared — including one model against itself, or the plain bilinear upscale
/// that stands in for the original video.
///
/// The extra cost is the second model's pass, and it is smaller than it looks:
/// the frame loop is paced by the renderer, not by the GPU. Measured with two
/// *different* CNNs, 640x480 -> 1280x960 held 0.995x realtime against 0.997x for
/// one model, and the 4x worst case (640x480 -> 2560x1920) held 1.000x against
/// 0.999x — so no frame rate is lost on this machine. It is real GPU work, on a
/// GPU with headroom to spare: a slower GPU or a larger output could show it.
/// The memory is less forgiving, since a second CNN allocates its own weights and
/// intermediates (4x: ~305 MB in flight against ~205 MB for one model).
final class SplitCompositor {
    private let ctx: MetalContext
    private let pipeline: MTLComputePipelineState
    private let dividerPipeline: MTLComputePipelineState

    init(ctx: MetalContext) throws {
        self.ctx = ctx
        self.pipeline = try ctx.device.makeComputePipelineState(function: ctx.function("compositeSplit"))
        self.dividerPipeline = try ctx.device.makeComputePipelineState(function: ctx.function("dividerSplit"))
    }

    /// Thickness of the divider in output pixels, proportional so it reads the
    /// same at 2x 1080p and 4x 4K.
    static func dividerWidth(forWidth width: Int) -> Int { max(2, width / 640) }

    /// Draws the divider alone, for when the two sides are the same model and the
    /// picture underneath is already correct.
    func drawDivider(into dst: MTLTexture, at split: Int, divider: Int) throws {
        try dispatch(pipeline: dividerPipeline, read: nil, write: dst,
                     to: max(0, split) + divider, split: split, divider: divider)
    }

    /// Copies `left`'s pixels in [0, split) into `dst`, then paints the divider
    /// columns at [split, split + divider).
    func composite(left: MTLTexture, into dst: MTLTexture,
                   split: Int, divider: Int) throws {
        try dispatch(pipeline: pipeline, read: left, write: dst,
                     to: max(0, split) + divider, split: split, divider: divider)
    }

    /// `to` is the first output column after the divider; the grid always starts
    /// at column 0.
    ///
    /// The kernels address columns absolutely (`gid.x`), and a threadgroup grid is
    /// laid out from the origin, so a grid spanning only the columns the pass will
    /// actually change runs its threads *before* the split and paints nothing —
    /// which is what left "compare the same model with itself" with no divider at
    /// all. Both kernels ignore the columns they are not responsible for, so the
    /// extra threads are a wider dispatch of a pass that already covers the frame.
    private func dispatch(pipeline: MTLComputePipelineState, read: MTLTexture?,
                          write dst: MTLTexture, to: Int,
                          split: Int, divider: Int) throws {
        let end = min(dst.width, max(0, to))
        guard end > 0 else { return }
        var splitValue = Int32(max(0, min(split, dst.width)))
        var dividerValue = Int32(max(0, divider))

        let desc = MTLCommandBufferDescriptor()
        desc.retainedReferences = false
        guard let cb = ctx.queue.makeCommandBuffer(descriptor: desc),
              let enc = cb.makeComputeCommandEncoder() else {
            throw EngineError.metalUnavailable
        }
        enc.setComputePipelineState(pipeline)
        if let read { enc.setTexture(read, index: 0) }
        enc.setTexture(dst, index: 1)
        enc.setBytes(&splitValue, length: MemoryLayout<Int32>.size, index: 0)
        enc.setBytes(&dividerValue, length: MemoryLayout<Int32>.size, index: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: (end + tg.width - 1) / tg.width,
                             height: (dst.height + tg.height - 1) / tg.height,
                             depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
    }

    /// A scratch target for the left model, at the same geometry as the output.
    /// Private: it is only ever read by the composite, never presented.
    func makeScratch(width: Int, height: Int) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        guard let t = ctx.device.makeTexture(descriptor: d) else {
            throw EngineError.metalUnavailable
        }
        return t
    }
}
