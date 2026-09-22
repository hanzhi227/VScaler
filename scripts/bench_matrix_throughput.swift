import Foundation
import Metal

// Is there a faster path than fp16 SIMT FMA (measured 8.4 TFLOP/s in
// `bench_alu_throughput.swift`) for the 160.7 GFLOP that animevideov3 needs at
// 480x270? Two candidates: `simdgroup_matrix` (Apple's per-SIMD-group matrix
// multiply) and, on Metal 4, the MPP tensor ops that drive the neural
// accelerators.
//
// This measures simdgroup_matrix at the shapes an implicit-GEMM conv would use.
//
//   xcrun -sdk macosx swiftc -O scripts/bench_matrix_throughput.swift \
//       -o /tmp/matbench -framework Metal && /tmp/matbench

setvbuf(stdout, nil, _IONBF, 0)
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    exit(1)
}
print("device: \(device.name)")

let shader = """
#include <metal_stdlib>
using namespace metal;

// simdgroup_matrix elements are distributed across the SIMD group's lanes, so a
// matrix is seeded through `thread_elements()` rather than a broadcast cast.
static inline simdgroup_matrix<half, 8, 8> seed_half(half s) {
    simdgroup_matrix<half, 8, 8> m;
    auto e = m.thread_elements();
    e[0] = s; e[1] = s * 0.5h + 0.25h;
    return m;
}

// 8x8x8 half matrix multiply-accumulate, chained end to end: a dependency chain
// of MMAs, the same shape an implicit-GEMM inner loop has. ITER * 512 MACs per
// SIMD group.
kernel void mma_half_8x8x8(device float *out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half seed[256];
    seed[tid] = half(tid & 255u) * 1e-3h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half s = seed[tid & 7u];
    simdgroup_matrix<half, 8, 8> a = seed_half(s + 0.5h);
    simdgroup_matrix<half, 8, 8> b = seed_half(s * 0.5h + 0.25h);
    simdgroup_matrix<float, 8, 8> c;
    auto ce = c.thread_elements();
    ce[0] = 0.0f; ce[1] = 0.0f; ce[2] = 0.0f; ce[3] = 0.0f;
    for (int k = 0; k < 512; ++k) {
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    out[gid] = c.thread_elements()[0] + float(s);
}

static inline simdgroup_matrix<float, 8, 8> seed_float(float s) {
    simdgroup_matrix<float, 8, 8> m;
    auto e = m.thread_elements();
    e[0] = s; e[1] = s * 0.5f;
    return m;
}

kernel void mma_float_8x8x8(device float *out [[buffer(0)]],
                            uint gid [[thread_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float seed[256];
    seed[tid] = float(tid & 255u) * 1e-3f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = seed[tid & 7u];
    simdgroup_matrix<float, 8, 8> a = seed_float(0.5f + s);
    simdgroup_matrix<float, 8, 8> b = seed_float(0.25f + s);
    simdgroup_matrix<float, 8, 8> c = seed_float(0.0f);
    for (int k = 0; k < 512; ++k) {
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    out[gid] = c.thread_elements()[0] + s;
}

// The shape a real GEMM uses: 4 output tiles sharing loads, so the ratio of
// MACs to register traffic is 2x better than the bare chain above.
kernel void mma_half_tiles(device float *out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half seed[256];
    seed[tid] = half(tid & 255u) * 1e-3h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half s = seed[tid & 7u];
    simdgroup_matrix<half, 8, 8> a0 = seed_half(s + 0.5h);
    simdgroup_matrix<half, 8, 8> a1 = seed_half(s + 0.25h);
    simdgroup_matrix<half, 8, 8> b0 = seed_half(s * 0.5h);
    simdgroup_matrix<half, 8, 8> b1 = seed_half(s * 0.5h + 0.125h);
    simdgroup_matrix<float, 8, 8> c00 = seed_float(0.0f);
    simdgroup_matrix<float, 8, 8> c01 = seed_float(0.0f);
    simdgroup_matrix<float, 8, 8> c10 = seed_float(0.0f);
    simdgroup_matrix<float, 8, 8> c11 = seed_float(0.0f);
    for (int k = 0; k < 512; ++k) {
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    out[gid] = c00.thread_elements()[0] + c01.thread_elements()[0]
             + c10.thread_elements()[0] + c11.thread_elements()[0] + float(s);
}
"""

let lib: MTLLibrary
do { lib = try device.makeLibrary(source: shader, options: nil) } catch {
    print("shader compile failed: \(error)"); exit(1)
}

func makePSO(_ name: String) -> MTLComputePipelineState? {
    guard let fn = lib.makeFunction(name: name) else { print("no fn \(name)"); return nil }
    do { return try device.makeComputePipelineState(function: fn) }
    catch { print("pso \(name) failed: \(error)"); return nil }
}

let threadgroups = 1 << 12
let outBuf = device.makeBuffer(length: threadgroups * 256 * 4, options: .storageModePrivate)!

func timeKernel(_ name: String, _ pso: MTLComputePipelineState, macsPerThread: Double) {
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    func encode(_ cb: MTLCommandBuffer) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
    for _ in 0..<2 { let cb = queue.makeCommandBuffer()!; encode(cb); cb.commit(); cb.waitUntilCompleted() }
    var best = Double.greatestFiniteMagnitude
    for _ in 0..<10 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let cb = queue.makeCommandBuffer()!; encode(cb); cb.commit(); cb.waitUntilCompleted()
        best = min(best, CFAbsoluteTimeGetCurrent() - t0)
    }
    // 8 SIMD groups per 256-thread threadgroup.
    let totalMacs = macsPerThread * 8 * Double(threadgroups)
    print(String(format: "%-18@ %7.3f ms   %7.2f TFLOP/s",
                 name as NSString, best * 1000, 2 * totalMacs / best / 1e12))
}

print("\n-- simdgroup_matrix (512 chained MMAs, 8 SIMD groups/threadgroup) --")
if let p = makePSO("mma_half_8x8x8") { timeKernel("mma_half_8x8x8", p, macsPerThread: 512 * 512) }
if let p = makePSO("mma_float_8x8x8") { timeKernel("mma_float_8x8x8", p, macsPerThread: 512 * 512) }
if let p = makePSO("mma_half_tiles") { timeKernel("mma_half_tiles", p, macsPerThread: 512 * 512 * 4) }

let macs = 0.62e6 * 480.0 * 270.0
print(String(format: "\nanimevideov3 at 480x270 = %.1f GFLOP; at the rates above:",
             2 * macs / 1e9))
for r in [8.4, 16.0, 30.0, 60.0] {
    print(String(format: "  %5.1f TFLOP/s -> %6.1f ms/frame  %5.2fx realtime @30fps",
                 r, 2 * macs / (r * 1e12) * 1000, (1.0 / (2 * macs / (r * 1e12))) / 30))
}
