import VideoUpscaleMetal
import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers

// vupbench --input <png> --output <png> --mode bilinear|cnnX2M [--iters N]
//           [--repeat-to-1080] [--out-format bgra8|rgba16f]

struct Options {
    var input: String = ""
    var output: String = ""
    var mode: UpscaleMode = .cnnX2M
    var iters: Int = 20
    var repeatTo1080 = false
    var outFormat: MTLPixelFormat = .bgra8Unorm
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("error: " + message + "\n").data(using: .utf8)!)
    exit(2)
}

func parseOptions() -> Options {
    var o = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    func next(_ flag: String) -> String {
        guard !args.isEmpty else { fail("missing value for \(flag)") }
        return args.removeFirst()
    }
    while !args.isEmpty {
        let a = args.removeFirst()
        switch a {
        case "--input":  o.input = next(a)
        case "--output": o.output = next(a)
        case "--mode":
            let m = next(a)
            switch m {
            case "bilinear":      o.mode = .bilinear
            case "cnnX2M":        o.mode = .cnnX2M
            case "cnnDenoiseX2M": o.mode = .cnnDenoiseX2M
            default: fail("unknown --mode \(m) (expected bilinear|cnnX2M|cnnDenoiseX2M)")
            }
        case "--iters": o.iters = Int(next(a)) ?? 20
        case "--repeat-to-1080": o.repeatTo1080 = true
        case "--out-format":
            switch next(a) {
            case "bgra8", "bgra8Unorm": o.outFormat = .bgra8Unorm
            case "rgba16f", "rgba16Float": o.outFormat = .rgba16Float
            default: fail("unknown --out-format (expected bgra8|rgba16f)")
            }
        case "-h", "--help":
            print("usage: vupbench --input <png> --output <png> --mode bilinear|cnnX2M|cnnDenoiseX2M [--iters N] [--repeat-to-1080] [--out-format bgra8|rgba16f]")
            exit(0)
        default:
            fail("unknown argument \(a)")
        }
    }
    if o.input.isEmpty { fail("--input is required") }
    if o.output.isEmpty { fail("--output is required") }
    if o.iters < 1 { o.iters = 1 }
    return o
}

// MARK: - PNG I/O (BGRA8, byte order 32Little + premultipliedFirst)

let bgraBitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
    | CGBitmapInfo.byteOrder32Little.rawValue

func loadCGImage(_ path: String) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        fail("could not decode image at \(path)")
    }
    return img
}

/// Rasterises `img` into a tightly packed BGRA8 buffer of the given size
/// (a no-op resize when width/height equal the image's own size).
func rasterizeBGRA(_ img: CGImage, width: Int, height: Int, highQuality: Bool) -> [UInt8] {
    var buf = [UInt8](repeating: 0, count: width * height * 4)
    guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { fail("no sRGB colorspace") }
    buf.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress,
                                  width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: cs, bitmapInfo: bgraBitmapInfo) else {
            fail("could not create bitmap context")
        }
        ctx.interpolationQuality = highQuality ? .high : .none
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
    }
    return buf
}

func makeTexture(device: MTLDevice, bytes: [UInt8], width: Int, height: Int,
                 format: MTLPixelFormat = .bgra8Unorm,
                 bytesPerRow: Int? = nil) -> MTLTexture {
    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: format, width: width, height: height, mipmapped: false)
    desc.usage = [.shaderRead, .shaderWrite]
    desc.storageMode = .shared
    guard let tex = device.makeTexture(descriptor: desc) else { fail("makeTexture failed") }
    let bpr = bytesPerRow ?? width * 4
    bytes.withUnsafeBytes { raw in
        tex.replace(region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: bpr)
    }
    return tex
}

func writePNG(texture: MTLTexture, path: String) {
    let w = texture.width, h = texture.height
    var buf = [UInt8](repeating: 0, count: w * h * 4)
    buf.withUnsafeMutableBytes { raw in
        texture.getBytes(raw.baseAddress!, bytesPerRow: w * 4,
                         from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    }
    guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
          let provider = CGDataProvider(data: Data(buf) as CFData),
          let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: w * 4, space: cs,
                           bitmapInfo: CGBitmapInfo(rawValue: bgraBitmapInfo),
                           provider: provider, decode: nil, shouldInterpolate: false,
                           intent: .defaultIntent) else {
        fail("could not build CGImage for output")
    }
    let url = URL(fileURLWithPath: path) as CFURL
    guard let dest = CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil) else {
        fail("could not create PNG destination at \(path)")
    }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { fail("could not write PNG at \(path)") }
}

/// Reads back a shared rgba16Float texture and reports fp32 health, plus
/// converts it to BGRA8 bytes (clamped) for PNG output.
func floatTextureStatsAndBytes(_ tex: MTLTexture) -> (minV: Float, maxV: Float, nan: Int, inf: Int, bytes: [UInt8]) {
    let w = tex.width, h = tex.height
    // rgba16Float is 2 bytes per channel => 8 bytes per pixel. Read as halves.
    var halves = [Float16](repeating: 0, count: w * h * 4)
    halves.withUnsafeMutableBytes { raw in
        tex.getBytes(raw.baseAddress!, bytesPerRow: w * 8,
                     from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    }
    var minV = Float.greatestFiniteMagnitude
    var maxV = -Float.greatestFiniteMagnitude
    var nan = 0, inf = 0
    var out = [UInt8](repeating: 0, count: w * h * 4)
    for i in stride(from: 0, to: halves.count, by: 4) {
        let r = Float(halves[i]), g = Float(halves[i + 1]), b = Float(halves[i + 2])
        for v in [r, g, b] {
            if v.isNaN { nan += 1 }
            else if v.isInfinite { inf += 1 }
            else { minV = min(minV, v); maxV = max(maxV, v) }
        }
        // bytes are BGRA little-endian
        out[i + 0] = UInt8(max(0, min(1, b)) * 255 + 0.5)
        out[i + 1] = UInt8(max(0, min(1, g)) * 255 + 0.5)
        out[i + 2] = UInt8(max(0, min(1, r)) * 255 + 0.5)
        out[i + 3] = 255
    }
    return (minV, maxV, nan, inf, out)
}

// MARK: - main

let opts = parseOptions()

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
FileHandle.standardError.write("device: \(device.name)\n".data(using: .utf8)!)

let img = loadCGImage(opts.input)
let srcW = img.width, srcH = img.height

// Effective pipeline input resolution.
var inW = srcW
var inH = srcH
if opts.repeatTo1080 {
    // "1080p-class height": scale so height == 1080 (preserving aspect).
    inH = 1080
    inW = max(1, Int((Double(srcW) * 1080.0 / Double(srcH)).rounded()))
}
let bgra = rasterizeBGRA(img, width: inW, height: inH, highQuality: opts.repeatTo1080)

let inTex = makeTexture(device: device, bytes: bgra, width: inW, height: inH)

let outW = inW * 2
let outH = inH * 2
let outTex: MTLTexture
let outFormat = opts.outFormat
if outFormat == .rgba16Float {
    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba16Float, width: outW, height: outH, mipmapped: false)
    desc.usage = [.shaderRead, .shaderWrite]
    desc.storageMode = .shared
    guard let t = device.makeTexture(descriptor: desc) else { fail("makeTexture failed") }
    outTex = t
} else {
    outTex = makeTexture(device: device, bytes: [UInt8](repeating: 0, count: outW * outH * 4),
                         width: outW, height: outH)
}

let pipeline: VideoUpscalingPipeline
do {
    pipeline = try VideoUpscalingPipeline(device: device, mode: opts.mode)
} catch {
    fail("pipeline init failed: \(error)")
}

// iters runs total; the first is a warmup and excluded from the average.
var samples: [Double] = []
samples.reserveCapacity(max(1, opts.iters - 1))
var lastError: Error?
for i in 0..<opts.iters {
    let t0 = DispatchTime.now().uptimeNanoseconds
    do {
        try pipeline.upscale(inTex, into: outTex)
    } catch {
        lastError = error
        break
    }
    let t1 = DispatchTime.now().uptimeNanoseconds
    if i > 0 { samples.append(Double(t1 - t0) / 1_000_000.0) }
}
if let e = lastError { fail("upscale failed: \(e)") }
if samples.isEmpty {
    // iters == 1: measure a single run so the JSON still carries a number.
    let t0 = DispatchTime.now().uptimeNanoseconds
    try! pipeline.upscale(inTex, into: outTex)
    let t1 = DispatchTime.now().uptimeNanoseconds
    samples.append(Double(t1 - t0) / 1_000_000.0)
}
let mean = samples.reduce(0, +) / Double(samples.count)
let sorted = samples.sorted()
let median = sorted[sorted.count / 2]
let minMs = sorted.first!
let maxMs = sorted.last!

var statsLine = ""
if outFormat == .rgba16Float {
    let s = floatTextureStatsAndBytes(outTex)
    let bgraBytes = s.bytes
    let pngTex = makeTexture(device: device, bytes: bgraBytes, width: outW, height: outH)
    writePNG(texture: pngTex, path: opts.output)
    statsLine = ", \"float_min\": \(s.minV), \"float_max\": \(s.maxV), \"nan_count\": \(s.nan), \"inf_count\": \(s.inf)"
} else {
    writePNG(texture: outTex, path: opts.output)
}

let modeStr = opts.mode.rawValue
let fmtStr = (outFormat == .rgba16Float) ? "rgba16f" : "bgra8"
let msStr = String(format: "%.4f", mean)
let minStr = String(format: "%.4f", minMs)
let medStr = String(format: "%.4f", median)
let maxStr = String(format: "%.4f", maxMs)
print("{\"input_w\": \(inW), \"input_h\": \(inH), \"ms_per_frame\": \(msStr), \"mode\": \"\(modeStr)\", \"output_w\": \(outW), \"output_h\": \(outH), \"iters\": \(opts.iters), \"min_ms\": \(minStr), \"median_ms\": \(medStr), \"max_ms\": \(maxStr), \"out_format\": \"\(fmtStr)\"\(statsLine)}")
