import Foundation
import Metal

/// Shared Metal context + the engine's own kernels (bypass scaler).
final class MetalContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let source: String
    private var library: MTLLibrary?

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw EngineError.metalUnavailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw EngineError.metalUnavailable
        }
        self.device = device
        self.queue = queue
        self.source = Self.kernelSource
        self.library = try device.makeLibrary(source: source, options: nil)
    }

    func function(_ name: String) throws -> MTLFunction {
        guard let lib = library, let fn = lib.makeFunction(name: name) else {
            throw EngineError.metalFunctionMissing(name)
        }
        return fn
    }

    static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    // 2x bilinear upsample: dst is (2W, 2H), src is (W, H), both unorm RGBA/BGRA.
    kernel void bilinear2x(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) { return; }
        float2 uv = (float2(gid) + 0.5f) * 0.5f - 0.5f;
        int2 p00 = int2(floor(uv));
        int sw = src.get_width(), sh = src.get_height();
        float2 f = fract(uv);
        int2 p10 = int2(min(p00.x + 1, sw - 1), p00.y);
        int2 p01 = int2(p00.x, min(p00.y + 1, sh - 1));
        int2 p11 = int2(min(p00.x + 1, sw - 1), min(p00.y + 1, sh - 1));
        p00 = int2(max(p00.x, 0), max(p00.y, 0));
        float4 c = (1.0f - f.x) * (1.0f - f.y) * src.read(uint2(p00))
                 + f.x * (1.0f - f.y) * src.read(uint2(p10))
                 + (1.0f - f.x) * f.y * src.read(uint2(p01))
                 + f.x * f.y * src.read(uint2(p11));
        dst.write(c, gid);
    }

    // Comparison overlay: copies x < `split` from `left` into dst and paints
    // `divider` bright columns at the boundary. dst already holds the right-hand
    // model, so this only moves the left model's pixels into place — the two
    // sides are each produced by a real scaler, which is what makes the halves
    // exactly what those models output rather than an approximation of it.
    kernel void compositeSplit(texture2d<float, access::read> left [[texture(0)]],
                               texture2d<float, access::write> dst [[texture(1)]],
                               constant int &split [[buffer(0)]],
                               constant int &divider [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
        int dw = dst.get_width(), dh = dst.get_height();
        if (gid.x >= dw || gid.y >= dh) { return; }
        int sx = clamp(split, 0, dw);
        if (gid.x < sx) { dst.write(left.read(gid), gid); return; }
        if (gid.x < min(sx + divider, dw)) { dst.write(float4(1.0f), gid); }
    }

    // The "off" position: a straight 1:1 copy, no interpolation and no model.
    // dst is the source's own geometry, so this moves the decoded frame through
    // unchanged and the display layer does whatever magnifying is needed to fit
    // the window. `bilinear2x` cannot stand in for it: that kernel is written for
    // dst = 2 x src and samples at (o + 0.5) / 2 - 0.5, which at 1:1 would read
    // the wrong texels entirely.
    kernel void copyTexel(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) { return; }
        dst.write(src.read(gid), gid);
    }

    // Just the divider, for when both sides are the same model: the pixels are
    // already right everywhere, so nothing has to be read — which is also why
    // this is a separate kernel rather than one reading and writing the same
    // texture, which Metal leaves undefined. The output is at index 1 (the
    // dispatcher binds a read texture at 0 first), so a divider-only run writes
    // the frame it was handed rather than into whatever index 0 holds.
    kernel void dividerSplit(texture2d<float, access::write> dst [[texture(1)]],
                             constant int &split [[buffer(0)]],
                             constant int &divider [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
        int dw = dst.get_width(), dh = dst.get_height();
        if (gid.x >= dw || gid.y >= dh) { return; }
        if (gid.x >= split && gid.x < min(split + divider, dw)) {
            dst.write(float4(1.0f), gid);
        }
    }
    """
}

public enum EngineError: Error, CustomStringConvertible {
    case metalUnavailable
    case metalFunctionMissing(String)
    case metalFXUnavailable(String)
    case scalerGeometryMismatch(String)
    case decoderFailure(String)
    case muxerFailure(String)
    case serverFailure(String)
    case pixelBufferFailure

    public var description: String {
        switch self {
        case .metalUnavailable: return "Metal device unavailable"
        case .metalFunctionMissing(let f): return "Metal function missing: \(f)"
        case .metalFXUnavailable(let m): return "MetalFX: \(m)"
        case .scalerGeometryMismatch(let m): return "Scaler geometry: \(m)"
        case .decoderFailure(let m): return "Decoder: \(m)"
        case .muxerFailure(let m): return "Muxer: \(m)"
        case .serverFailure(let m): return "Server: \(m)"
        case .pixelBufferFailure: return "CVPixelBuffer allocation/wrap failed"
        }
    }
}
