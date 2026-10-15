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
        int sw = src.get_width(), sh = src.get_height();
        // Clamp the base texel into range BEFORE deriving its neighbours: at
        // the left/top edge uv goes negative, and an unclamped floor of -1 was
        // inherited by p01/p10/p11, reading outside the texture (Metal returns
        // 0 there) and darkening the whole first column and row. Interior
        // texels are nowhere near the clamps and are unchanged.
        int2 p00 = clamp(int2(floor(uv)), int2(0), int2(sw - 1, sh - 1));
        float2 f = fract(uv);
        int2 p10 = int2(min(p00.x + 1, sw - 1), p00.y);
        int2 p01 = int2(p00.x, min(p00.y + 1, sh - 1));
        int2 p11 = int2(min(p00.x + 1, sw - 1), min(p00.y + 1, sh - 1));
        float4 c = (1.0f - f.x) * (1.0f - f.y) * src.read(uint2(p00))
                 + f.x * (1.0f - f.y) * src.read(uint2(p10))
                 + (1.0f - f.x) * f.y * src.read(uint2(p01))
                 + f.x * f.y * src.read(uint2(p11));
        dst.write(c, gid);
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
