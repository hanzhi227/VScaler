import Foundation
import Metal

// What this GPU can actually do, measured, before designing a convolution
// kernel around guesses: fp32 SIMT FMA rate, fp16 SIMT FMA rate, and
// threadgroup-memory read bandwidth.
//
// The reason this file exists: `scripts/bench_realesrgan_conv.swift` projected a
// whole Real-ESRGAN network from one layer and was wrong by ~2.6x. Rates taken
// from vendor specs are the same kind of mistake. So: measure the primitive
// rates, then measure the real kernel, then believe only the latter.
//
//   xcrun -sdk macosx swiftc -O scripts/bench_alu_throughput.swift \
//       -o /tmp/alubench -framework Metal && /tmp/alubench

setvbuf(stdout, nil, _IONBF, 0)
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    exit(1)
}
print("device: \(device.name)")
print("max threads/threadgroup: \(device.maxThreadsPerThreadgroup)")
print("max threadgroup memory:   \(device.maxThreadgroupMemoryLength) bytes")

let shader = """
#include <metal_stdlib>
using namespace metal;

// Each thread does ITER dependent fp32 FMAs on 8 independent accumulators, then
// writes one value so nothing is optimised away.
kernel void fma_fp32(device float *out [[buffer(0)]],
                     uint gid [[thread_position_in_grid]]) {
    float a[8];
    float x = float(gid & 1023u) * 1e-6f + 0.5f;
    for (int i = 0; i < 8; ++i) a[i] = x + float(i);
    float c = 1.000001f;
    for (int k = 0; k < 512; ++k) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = fma(a[i], c, x);
    }
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) s += a[i];
    out[gid] = s;
}

// Same shape, half4 vectors: 4 FMAs per instruction.
kernel void fma_fp16(device half *out [[buffer(0)]],
                     uint gid [[thread_position_in_grid]]) {
    half4 a[4];
    half x = half(gid & 1023u) * 1e-3h + 0.5h;
    for (int i = 0; i < 4; ++i) a[i] = half4(x + half(i));
    half4 c = half4(1.0005h);
    for (int k = 0; k < 512; ++k) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = fma(a[i], c, half4(x));
    }
    half s = 0.0h;
    for (int i = 0; i < 4; ++i) s += a[i].x + a[i].y + a[i].z + a[i].w;
    out[gid] = s;
}

// half4 fma on 8 accumulators (what a wider kernel would look like).
kernel void fma_fp16_wide(device half *out [[buffer(0)]],
                          uint gid [[thread_position_in_grid]]) {
    half4 a[8];
    half x = half(gid & 1023u) * 1e-3h + 0.5h;
    for (int i = 0; i < 8; ++i) a[i] = half4(x + half(i));
    half4 c = half4(1.0005h);
    for (int k = 0; k < 512; ++k) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = fma(a[i], c, half4(x));
    }
    half s = 0.0h;
    for (int i = 0; i < 8; ++i) s += a[i].x + a[i].y + a[i].z + a[i].w;
    out[gid] = s;
}

// float4 fma on 8 accumulators: does fp32 vectorise to the same rate as scalar?
kernel void fma_fp32_vec(device float *out [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
    float4 a[8];
    float x = float(gid & 1023u) * 1e-6f + 0.5f;
    for (int i = 0; i < 8; ++i) a[i] = float4(x + float(i));
    float4 c = float4(1.000001f);
    for (int k = 0; k < 512; ++k) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = fma(a[i], c, float4(x));
    }
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) s += a[i].x + a[i].y + a[i].z + a[i].w;
    out[gid] = s;
}

constant uint TG = 256;

// Threadgroup-memory read bandwidth: 256 threads each read 8 half4 per step out
// of a 16 KB tile. Layout is the padded one a real conv kernel would use, so a
// bank-conflict problem would show up here as a low number.
kernel void smem_read(device float *out [[buffer(0)]],
                      uint gid [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[256 * 36];          // 256 pixels x 36 half (32 + 4 pad)
    for (uint i = tid; i < 256 * 36; i += TG) tile[i] = half(i & 63u) * 0.01h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half4 acc = half4(0.0h);
    uint base = (tid & 255u) * 36u;
    for (int k = 0; k < 256; ++k) {
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            uint idx = base + uint((k + g) & 31);
            acc += *(threadgroup half4 *)&tile[idx & ~3u];
        }
    }
    out[gid] = float(acc.x + acc.y + acc.z + acc.w);
}
"""

let lib: MTLLibrary
do { lib = try device.makeLibrary(source: shader, options: nil) } catch {
    print("shader compile failed: \(error)"); exit(1)
}

func makePSO(_ name: String) -> MTLComputePipelineState {
    guard let fn = lib.makeFunction(name: name) else { print("no fn \(name)"); exit(1) }
    do { return try device.makeComputePipelineState(function: fn) }
    catch { print("pso \(name) failed: \(error)"); exit(1) }
}

let threads = 1 << 20
let outBuf = device.makeBuffer(length: threads * 4, options: .storageModePrivate)!

func timeKernel(_ name: String, _ pso: MTLComputePipelineState, flopsPerThread: Double,
                tg: MTLSize) -> Double {
    func encode(_ cb: MTLCommandBuffer) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        let groups = MTLSize(width: threads / tg.width, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
    for _ in 0..<2 { let cb = queue.makeCommandBuffer()!; encode(cb); cb.commit(); cb.waitUntilCompleted() }
    let iters = 10
    let t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters { let cb = queue.makeCommandBuffer()!; encode(cb); cb.commit(); cb.waitUntilCompleted() }
    let s = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters)
    let tflops = flopsPerThread * Double(threads) / s / 1e12
    print(String(format: "%-14@ %7.3f ms   %7.2f TFLOP/s", name as NSString, s * 1000, tflops))
    return tflops
}

let tg256 = MTLSize(width: 256, height: 1, depth: 1)
print("\n-- FMA throughput (512 iterations x 8 accumulators) --")
_ = timeKernel("fma_fp32", try makePSO("fma_fp32"), flopsPerThread: 512 * 8 * 2, tg: tg256)
_ = timeKernel("fma_fp32_vec", try makePSO("fma_fp32_vec"), flopsPerThread: 512 * 8 * 4 * 2, tg: tg256)
_ = timeKernel("fma_fp16", try makePSO("fma_fp16"), flopsPerThread: 512 * 4 * 4 * 2, tg: tg256)
_ = timeKernel("fma_fp16_wide", try makePSO("fma_fp16_wide"), flopsPerThread: 512 * 8 * 4 * 2, tg: tg256)

// smem bandwidth: 256 iterations x 8 half4 reads = 256*8*8 bytes per thread
let smemBytesPerThread = Double(256 * 8 * 8)
let p = try makePSO("smem_read")
func encodeSmem(_ cb: MTLCommandBuffer) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(outBuf, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: threads / 256, height: 1, depth: 1),
                             threadsPerThreadgroup: tg256)
    enc.endEncoding()
}
for _ in 0..<2 { let cb = queue.makeCommandBuffer()!; encodeSmem(cb); cb.commit(); cb.waitUntilCompleted() }
var best = Double.greatestFiniteMagnitude
for _ in 0..<10 {
    let t0 = CFAbsoluteTimeGetCurrent()
    let cb = queue.makeCommandBuffer()!; encodeSmem(cb); cb.commit(); cb.waitUntilCompleted()
    best = min(best, CFAbsoluteTimeGetCurrent() - t0)
}
print(String(format: "\n-- threadgroup memory --\n%-14@ %7.3f ms   %7.1f GB/s",
             "smem_read" as NSString, best * 1000,
             smemBytesPerThread * Double(threads) / best / 1e9))

// For reference: what a 480x270 animevideov3 frame needs.
let macs = 0.62e6 * 480.0 * 270.0        // MACs per input pixel x pixels
print(String(format: "\nanimevideov3 at 480x270: %.1f GMAC = %.1f GFLOP",
             macs / 1e9, 2 * macs / 1e9))
