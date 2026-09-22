import Foundation

/// Metal source for the two things MPSGraph cannot do here: get a decoded video
/// frame into the tensor layout the graph wants, and turn the graph's 48-channel
/// output into a 4x picture.
///
/// The 18 convolutions are *not* here. They run through MPSGraph, because this
/// GPU's honest ceiling for hand-written fp16 arithmetic is 8.4 TFLOP/s and the
/// network needs 160.7 GFLOP at 480x270 — so even a perfect SIMT kernel only
/// reaches ~19 ms, and `simdgroup_matrix` measured slower still (4.3 TFLOP/s).
/// MPS is already at ~7.1 TFLOP/s effective on this model, i.e. ~85% of that
/// ceiling, which leaves a hand-written kernel nothing to win and a lot to lose.
/// Both numbers are in `scripts/bench_alu_throughput.swift`.
enum RealESRGANKernels {

    static let source: String = #"""
    #include <metal_stdlib>
    using namespace metal;

    // ---------------------------------------------------------------------
    // Decoded frame -> NHWC fp16 tensor.
    //
    // The graph's input is [1, H, W, 4] rather than 3 channels: the 4th is
    // always zero and its weight column is zero too, so it costs one wasted
    // multiply in the first convolution and buys 8-byte aligned, fully
    // coalesced writes here. Reading a bgra8Unorm texture as float4 already
    // swizzles to RGBA, so .rgb is the model's input order.
    // ---------------------------------------------------------------------
    kernel void frameToNHWC4(texture2d<float, access::read>  src [[texture(0)]],
                             device half4                 *dst [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= src.get_width() || gid.y >= src.get_height()) return;
        float4 c = src.read(gid);
        dst[gid.y * src.get_width() + gid.x] = half4(half(c.r), half(c.g), half(c.b), 0.0h);
    }

    // ---------------------------------------------------------------------
    // 48-channel tensor -> 4x picture, with the network's residual.
    //
    // `SRVGGNetCompact` ends in a PixelShuffle(4) and then adds a
    // nearest-neighbour magnification of the *input* — the network only
    // produces a correction, not the picture. Both happen here, in one pass,
    // so the 4x image is written once and never round-trips through memory.
    //
    // Channel c of the shuffled output comes from tensor channel
    // c*16 + dy*4 + dx, where (dx, dy) is the sub-pixel position. The five
    // halves this thread needs are adjacent in the tensor, so a warp's reads
    // are close together even though each is 2 bytes.
    //
    // The sum is in fp32: fp16 has ~11 mantissa bits, and a correction added to
    // an 8-bit sample in fp16 loses about 6% of one output level to rounding
    // before the write. The output texture is 8-bit anyway, so this is the last
    // place precision is free.
    // ---------------------------------------------------------------------
    kernel void shuffleResidual(device const half             *feat [[buffer(0)]],
                                texture2d<float, access::read>  src [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
        uint ow = dst.get_width(), oh = dst.get_height();
        if (gid.x >= ow || gid.y >= oh) return;
        uint w = src.get_width();
        uint x = gid.x >> 2, y = gid.y >> 2;
        uint dx = gid.x & 3u, dy = gid.y & 3u;
        uint sub = dy * 4u + dx;

        device const half *p = feat + (y * w + x) * 48u;
        float3 corr = float3(float(p[0u  + sub]), float(p[16u + sub]), float(p[32u + sub]));
        float3 base = src.read(uint2(x, y)).rgb;
        float3 v = clamp(base + corr, 0.0f, 1.0f);
        dst.write(float4(v, 1.0f), gid);
    }
    """#
}
