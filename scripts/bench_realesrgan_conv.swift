import Foundation
import Metal

// How fast can this GPU run the kind of convolution a Real-ESRGAN body is made
// of? Measures one 3x3, 64->64 channel layer with PReLU — 32 of those are the
// dominant cost of SRVGGNetCompact, which is what `realesr-general-x4v3` is —
// then projects the whole network.
//
// Deliberately a *favourable* estimate:
//   * feature maps live in textures, so reads hit the texture cache;
//   * each thread computes a block of 8 output channels for one pixel and reuses
//     every input fetch across them, which is how this kernel is made
//     compute-bound rather than read-bound.
// So this is close to the best case for a hand-written Metal conv, not a
// pessimistic one. If even this is nowhere near real time, tuning cannot make
// the model a live playback mode.
//
// Run:
//   xcrun -sdk macosx swiftc -O scripts/bench_realesrgan_conv.swift \
//       -o /tmp/convbench -framework Metal && /tmp/convbench
// Recorded result: results/general/realesrgan_feasibility.txt.

setvbuf(stdout, nil, _IONBF, 0)
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { exit(1) }
print("device: \(device.name)")

let CIN = 64, COUT = 64, PLANES = CIN / 4

let shader = """
#include <metal_stdlib>
using namespace metal;

kernel void conv3x3_64to64(texture2d<float, access::read>  src  [[texture(0)]],
                           texture2d<float, access::write> dst  [[texture(1)]],
                           device float *w    [[buffer(0)]],
                           device float *bias [[buffer(1)]],
                           device float *prelu[[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint W = dst.get_width(), H = dst.get_height();
    // gid.x packs (pixel x, output-channel block): 8 threads per pixel column.
    uint px = gid.x / 8u;
    int  cg = int(gid.x % 8u);          // which block of 8 output channels
    if (px >= W || gid.y >= H) return;
    int2 p = int2(int(px), int(gid.y));

    float acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = bias[cg * 8 + i];

    for (int g = 0; g < 16; ++g) {
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int2 q = int2(clamp(p.x + kx, 0, int(W) - 1), clamp(p.y + ky, 0, int(H) - 1));
                float4 v = src.read(uint2(uint(q.x), uint(q.y) + uint(g) * H));
                // One fetch, reused across the 8 output channels this thread owns.
                for (int i = 0; i < 8; ++i) {
                    int co = cg * 8 + i;
                    int base = ((co * 3 + (ky + 1)) * 3 + (kx + 1)) * 64 + g * 4;
                    acc[i] += dot(*(const device float4 *)(w + base), v);
                }
            }
        }
    }
    float total = 0.0f;
    for (int i = 0; i < 8; ++i) {
        float s = prelu[cg * 8 + i];
        total += (acc[i] > 0.0f) ? acc[i] : acc[i] * s;
    }
    dst.write(float4(total * 1e-6f, 0.0f, 0.0f, 1.0f), uint2(px, gid.y));
}
"""

let lib: MTLLibrary
do { lib = try device.makeLibrary(source: shader, options: nil) }
catch { print("shader compile failed: \(error)"); exit(1) }
guard let fn = lib.makeFunction(name: "conv3x3_64to64") else { print("no fn"); exit(1) }
let pso: MTLComputePipelineState
do { pso = try device.makeComputePipelineState(function: fn) }
catch { print("pso failed: \(error)"); exit(1) }

let W = 640, H = 480
let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                 width: W, height: H * PLANES, mipmapped: false)
sd.usage = [.shaderRead]; sd.storageMode = .private
let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                 width: W, height: H, mipmapped: false)
dd.usage = [.shaderWrite]; dd.storageMode = .private
guard let src = device.makeTexture(descriptor: sd), let dst = device.makeTexture(descriptor: dd) else {
    print("texture alloc failed"); exit(1)
}
let wCount = COUT * 9 * CIN
guard let wBuf = device.makeBuffer(length: wCount * 4, options: .storageModePrivate),
      let biasBuf = device.makeBuffer(bytes: [Float](repeating: 0, count: COUT),
                                      length: COUT * 4, options: .storageModePrivate),
      let preluBuf = device.makeBuffer(bytes: [Float](repeating: 0.25, count: COUT),
                                       length: COUT * 4, options: .storageModePrivate) else {
    print("buffer alloc failed"); exit(1)
}

func encode(_ cb: MTLCommandBuffer) {
    guard let enc = cb.makeComputeCommandEncoder() else { return }
    enc.setComputePipelineState(pso)
    enc.setTexture(src, index: 0)
    enc.setTexture(dst, index: 1)
    enc.setBuffer(wBuf, offset: 0, index: 0)
    enc.setBuffer(biasBuf, offset: 0, index: 1)
    enc.setBuffer(preluBuf, offset: 0, index: 2)
    let tg = MTLSize(width: 16, height: 16, depth: 1)
    enc.dispatchThreadgroups(MTLSize(width: (W * 8 + 15) / 16, height: (H + 15) / 16, depth: 1),
                             threadsPerThreadgroup: tg)
    enc.endEncoding()
}

for _ in 0..<3 { let cb = queue.makeCommandBuffer()!; encode(cb); cb.commit(); cb.waitUntilCompleted() }
let iters = 30
let t0 = CFAbsoluteTimeGetCurrent()
for _ in 0..<iters { let cb = queue.makeCommandBuffer()!; encode(cb); cb.commit(); cb.waitUntilCompleted() }
let ms = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000

let macsPerPixelPerLayer = Double(9 * CIN * COUT)
let tflops = macsPerPixelPerLayer * Double(W * H) * 2.0 / (ms / 1000.0) / 1e12
print(String(format: "one 3x3 %d->%d fp32 PReLU layer at %dx%d: %.3f ms  (%.2f TFLOP/s achieved)",
             CIN, COUT, W, H, ms, tflops))

let totalMacsPerPixel = 9.0 * 3 * 64 + 32.0 * macsPerPixelPerLayer + 9.0 * 64 * 48
print(String(format: "general-x4v3: %.2f M MACs per input pixel (32 body layers dominate)",
             totalMacsPerPixel / 1e6))
let scale = totalMacsPerPixel / macsPerPixelPerLayer
print("projected whole network, same kernel, no allowance for pixel-shuffle or I/O:")
for (w, h, name) in [(640, 480, "480p"), (854, 480, "SD-wide"), (960, 540, "540p"),
                     (1280, 720, "720p"), (1920, 1080, "1080p")] {
    let frameMs = ms * scale * (Double(w * h) / Double(W * H))
    let fps = 1000.0 / frameMs
    print(String(format: "  %-8@ %4dx%-4d %9.1f ms/frame  %6.2f fps  %5.2fx realtime @30fps",
                 name as NSString, w, h, frameMs, fps, fps / 30.0))
}
