import Foundation

/// Metal Shading Language source for the CNN 2× network (weights from Anime4K
/// v3.2 `Upscale_CNN_x2_M` / `Upscale_Denoise_CNN_x2_M`), plus `bilinear2x`.
///
/// Coordinate convention: `gid == (0,0)` is the top-left pixel and +y goes
/// DOWN, matching the texel-offset semantics of the upstream mpv GLSL
/// (`//!HOOK` / `texOff`). The Depth-to-Space pass maps channel `dy*2+dx` to
/// output pixel `(2x+dx, 2y+dy)`, so no vertical flip is introduced.
///
/// Arithmetic is fp32 throughout, matching GLSL `highp`. Neighbour fetches use
/// explicit integer clamped addressing (clamp-to-edge), so no samplers and no
/// texture caches are involved.
enum MetalKernels {

    static let source: String = #"""
    #include <metal_stdlib>
    using namespace metal;

    // The 9 texel offsets of the 3x3 convolutions, in exactly the order the
    // upstream GLSL writes them: (-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),
    // (1,-1),(1,0),(1,1).
    constant int2 kOff3x3[9] = {
        int2(-1, -1), int2(-1, 0), int2(-1, 1),
        int2( 0, -1), int2( 0, 0), int2( 0, 1),
        int2( 1, -1), int2( 1, 0), int2( 1, 1)
    };

    // Clamp-to-edge integer addressing.
    static inline int2 clamp2(int2 c, uint2 d) {
        return int2(clamp(c.x, 0, int(d.x) - 1), clamp(c.y, 0, int(d.y) - 1));
    }

    // GLSL `mat4 * vec4`. GLSL mat4 literals are COLUMN-major, i.e. column j is
    // m[4j..4j+3], and (M*v)[c] = sum_j m[4*j + c] * v[j].
    static inline float4 mul_mat4(constant float *m, float4 v) {
        return float4(
            m[0] * v.x + m[4]  * v.y + m[8]  * v.z + m[12] * v.w,
            m[1] * v.x + m[5]  * v.y + m[9]  * v.z + m[13] * v.w,
            m[2] * v.x + m[6]  * v.y + m[10] * v.z + m[14] * v.w,
            m[3] * v.x + m[7]  * v.y + m[11] * v.z + m[15] * v.w);
    }

    // 2x bilinear tap of `src` evaluated at output pixel coordinate `o`.
    // Reproduces mpv/GLSL bilinear sampling with clamp-to-edge: the sample
    // position in source pixels is (o + 0.5) / 2 - 0.5.
    static inline float4 bilinear2x_sample(texture2d<float, access::read> src, uint2 o) {
        float W = float(src.get_width());
        float H = float(src.get_height());
        float cx = (float(o.x) + 0.5f) * 0.5f - 0.5f;
        float cy = (float(o.y) + 0.5f) * 0.5f - 0.5f;
        float fx = floor(cx);
        float fy = floor(cy);
        float tx = cx - fx;
        float ty = cy - fy;
        int x0 = clamp(int(fx),     0, int(W) - 1);
        int x1 = clamp(int(fx) + 1, 0, int(W) - 1);
        int y0 = clamp(int(fy),     0, int(H) - 1);
        int y1 = clamp(int(fy) + 1, 0, int(H) - 1);
        float4 r0 = mix(src.read(uint2(x0, y0)), src.read(uint2(x1, y0)), tx);
        float4 r1 = mix(src.read(uint2(x0, y1)), src.read(uint2(x1, y1)), tx);
        return mix(r0, r1, ty);
    }

    // ---------------------------------------------------------------------
    // Fallback: plain 2x bilinear (mode == .bilinear)
    // ---------------------------------------------------------------------
    kernel void bilinear2x(texture2d<float, access::read>  src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
        dst.write(bilinear2x_sample(src, gid), gid);
    }

    // ---------------------------------------------------------------------
    // Pass 1/9: Conv-4x3x3x3  (input RGB -> 4 feature channels)
    // weights: 9 * mat4 (16 floats each) + 4 bias = 148 floats
    // ---------------------------------------------------------------------
    kernel void conv3x3_1to4(texture2d<float, access::read>  src [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             constant float *w [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
        uint2 dims = uint2(src.get_width(), src.get_height());
        if (gid.x >= dims.x || gid.y >= dims.y) return;
        int2 p = int2(gid);
        float4 acc = float4(w[144], w[145], w[146], w[147]);
        for (int k = 0; k < 9; ++k) {
            float4 v = src.read(uint2(clamp2(p + kOff3x3[k], dims)));
            acc += mul_mat4(w + k * 16, v);
        }
        dst.write(acc, gid);
    }

    // ---------------------------------------------------------------------
    // Passes 2..7/9: Conv-4x3x3x8 (ReLU split: pos/neg halves)
    // weights: 9 * mat4 (positive) then 9 * mat4 (negative) + 4 bias = 292
    // ---------------------------------------------------------------------
    kernel void conv3x3_8to4(texture2d<float, access::read>  src [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             constant float *w [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
        uint2 dims = uint2(src.get_width(), src.get_height());
        if (gid.x >= dims.x || gid.y >= dims.y) return;
        int2 p = int2(gid);
        float4 acc = float4(w[288], w[289], w[290], w[291]);
        for (int k = 0; k < 9; ++k) {
            float4 v = src.read(uint2(clamp2(p + kOff3x3[k], dims)));
            acc += mul_mat4(w + k * 16, max(v, 0.0f));
            acc += mul_mat4(w + (9 + k) * 16, max(-v, 0.0f));
        }
        dst.write(acc, gid);
    }

    // ---------------------------------------------------------------------
    // Pass 8/9: Conv-4x1x1x56 (all 7 feature maps, pos/neg split)
    // weights: 14 * mat4 in order pos(f0),neg(f0),pos(f1),neg(f1),... + bias4
    // ---------------------------------------------------------------------
    #define AN4K_APPLY(tex, idx)                                            \
        do {                                                                \
            float4 v = tex.read(gid);                                       \
            acc += mul_mat4(w + (2 * (idx)) * 16,     max(v,  0.0f));       \
            acc += mul_mat4(w + (2 * (idx) + 1) * 16, max(-v, 0.0f));       \
        } while (0)

    kernel void conv1x1_56to4(texture2d<float, access::read> f0 [[texture(0)]],
                              texture2d<float, access::read> f1 [[texture(1)]],
                              texture2d<float, access::read> f2 [[texture(2)]],
                              texture2d<float, access::read> f3 [[texture(3)]],
                              texture2d<float, access::read> f4 [[texture(4)]],
                              texture2d<float, access::read> f5 [[texture(5)]],
                              texture2d<float, access::read> f6 [[texture(6)]],
                              texture2d<float, access::write> dst [[texture(7)]],
                              constant float *w [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
        uint2 dims = uint2(f0.get_width(), f0.get_height());
        if (gid.x >= dims.x || gid.y >= dims.y) return;
        float4 acc = float4(w[224], w[225], w[226], w[227]);
        AN4K_APPLY(f0, 0);
        AN4K_APPLY(f1, 1);
        AN4K_APPLY(f2, 2);
        AN4K_APPLY(f3, 3);
        AN4K_APPLY(f4, 4);
        AN4K_APPLY(f5, 5);
        AN4K_APPLY(f6, 6);
        dst.write(acc, gid);
    }
    #undef AN4K_APPLY

    // ---------------------------------------------------------------------
    // Pass 9/9: Depth-to-Space (pixel shuffle) + bilinear residual of the
    // original input. Produces the 2x output.
    // ---------------------------------------------------------------------
    kernel void depth_to_space_add(texture2d<float, access::read>  feature [[texture(0)]],
                                   texture2d<float, access::read>  src      [[texture(1)]],
                                   texture2d<float, access::write> dst      [[texture(2)]],
                                   uint2 gid [[thread_position_in_grid]]) {
        uint ow = dst.get_width();
        uint oh = dst.get_height();
        if (gid.x >= ow || gid.y >= oh) return;
        uint x = gid.x >> 1;
        uint y = gid.y >> 1;
        uint dx = gid.x & 1u;
        uint dy = gid.y & 1u;
        float4 f = feature.read(uint2(x, y));
        float c = f[dy * 2u + dx];
        float3 base = bilinear2x_sample(src, gid).rgb;
        dst.write(float4(c + base, 1.0f), gid);
    }
    """#
}
