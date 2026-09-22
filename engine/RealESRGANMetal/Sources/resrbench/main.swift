import Foundation
import Metal
import RealESRGANMetal

// Real-ESRGAN as a live scaler: how fast, and is it right?
//
//   resrbench --bench [--variant animevideov3|general-x4v3] [--sizes 480x270]
//   resrbench --dump IN.raw OUT.raw W H [--variant ...]   one frame, raw BGRA8
//                                                   in, raw BGRA8 4x out, for
//                                                   comparison against PyTorch
//
// `--dump` exists so correctness is checked against the same code path that
// plays live. The Python side (`scripts/realesrgan/verify_metal.py`) writes the
// input, runs this, and compares the output with the fp32 model.

setvbuf(stdout, nil, _IONBF, 0)

func fail(_ s: String) -> Never {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
    exit(1)
}

var args = Array(CommandLine.arguments.dropFirst())
func flag(_ name: String) -> Bool {
    if let i = args.firstIndex(of: name) { args.remove(at: i); return true }
    return false
}
func value(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let v = args[i + 1]
    args.removeSubrange(i...(i + 1))
    return v
}

let bench = flag("--bench")
let variant = value("--variant") ?? "animevideov3"
let model: RealESRGANModel
switch variant {
case "animevideov3", "anime": model = RealESRGANAnimeV3Weights.model
case "general-x4v3", "general": model = RealESRGANGeneralWeights.model
default: fail("unknown --variant \(variant); try animevideov3 or general-x4v3")
}
let dump = flag("--dump")
let sizesArg = value("--sizes")
let iters = Int(value("--iters") ?? "10") ?? 10

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
print("device: \(device.name)")

let pipeline: RealESRGANPipeline
do { pipeline = try RealESRGANPipeline(model: model, device: device) }
catch { fail("pipeline init failed: \(error)") }

func makeTexture(width: Int, height: Int, write: Bool) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
    d.usage = write ? [.shaderWrite, .shaderRead] : [.shaderRead]
    d.storageMode = .shared
    guard let t = device.makeTexture(descriptor: d) else { fail("texture alloc failed") }
    return t
}

if dump {
    guard args.count >= 4, let w = Int(args[2]), let h = Int(args[3]) else {
        fail("usage: resrbench --dump IN.raw OUT.raw W H")
    }
    let inPath = args[0], outPath = args[1]
    guard let raw = FileManager.default.contents(atPath: inPath) else { fail("cannot read \(inPath)") }
    guard raw.count == w * h * 4 else {
        fail("\(inPath) is \(raw.count) bytes, expected \(w * h * 4) for \(w)x\(h) BGRA8")
    }
    let src = makeTexture(width: w, height: h, write: false)
    raw.withUnsafeBytes { buf in
        src.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                    withBytes: buf.baseAddress!, bytesPerRow: w * 4)
    }
    let dst = makeTexture(width: w * 4, height: h * 4, write: true)
    do { try pipeline.upscale(src, into: dst) } catch { fail("upscale failed: \(error)") }

    let ow = w * 4, oh = h * 4
    var out = [UInt8](repeating: 0, count: ow * oh * 4)
    out.withUnsafeMutableBytes { buf in
        dst.getBytes(buf.baseAddress!, bytesPerRow: ow * 4,
                     from: MTLRegionMake2D(0, 0, ow, oh), mipmapLevel: 0)
    }
    try! Data(out).write(to: URL(fileURLWithPath: outPath))
    print("wrote \(outPath) (\(ow)x\(oh))")
    exit(0)
}

if bench {
    let sizes: [(Int, Int)] = (sizesArg ?? "480x270,640x360,854x480,960x540,1280x720,1920x1080")
        .split(separator: ",").compactMap {
            let p = $0.split(separator: "x")
            guard p.count == 2, let w = Int(p[0]), let h = Int(p[1]) else { return nil }
            return (w, h)
        }
    print("\(model.name): \(model.bodyConvs) body convs, \(model.parameterCount) params, "
          + "fixed 4x, MPSGraph + 2 Metal kernels\n")
    print("   input          output        ms/frame     fps   x30fps  activation MB")
    for (w, h) in sizes {
        let src = makeTexture(width: w, height: h, write: false)
        let dst = makeTexture(width: w * 4, height: h * 4, write: true)
        do {
            for _ in 0..<3 { try pipeline.upscale(src, into: dst) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters { try pipeline.upscale(src, into: dst) }
            let ms = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            let mb = Double(model.activationBytes(width: w, height: h)) / 1e6
            print(String(format: "  %4dx%-4d -> %4dx%-4d  %9.2f  %7.2f  %5.2fx  %8.1f",
                         w, h, w * 4, h * 4, ms, 1000 / ms, 30 / (1000 / ms), mb))
        } catch { fail("\(w)x\(h) failed: \(error)") }
    }
    exit(0)
}

fail("""
usage:
  resrbench --bench [--sizes WxH,...] [--iters N]
  resrbench --dump IN.raw OUT.raw W H
""")
