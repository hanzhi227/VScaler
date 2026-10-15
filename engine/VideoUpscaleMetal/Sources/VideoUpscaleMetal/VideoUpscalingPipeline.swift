import Foundation
import Metal

#if canImport(CoreVideo)
import CoreVideo
#endif

/// Which 2x upscaler to run.
public enum UpscaleMode: String, Sendable, CaseIterable {
    /// Plain bilinear 2x (the "enhancement off" / fallback path).
    case bilinear
    /// Anime4K v3.2 `Upscale_CNN_x2_M` (medium CNN, 9 passes).
    case cnnX2M
    /// Anime4K v3.2 `Upscale_Denoise_CNN_x2_M` (same topology, denoise weights).
    case cnnDenoiseX2M

    public var isCNN: Bool { self != .bilinear }
}

public enum VideoUpscaleError: Error, CustomStringConvertible {
    case commandQueueUnavailable
    case libraryCompilationFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case textureAllocationFailed
    case bufferAllocationFailed(String)
    case invalidInput(String)
    case pixelBufferFailed(String)

    public var description: String {
        switch self {
        case .commandQueueUnavailable:      return "MTLDevice.makeCommandQueue() returned nil"
        case .libraryCompilationFailed(let s): return "Metal library compilation failed: \(s)"
        case .functionNotFound(let s):      return "Metal function not found: \(s)"
        case .pipelineCreationFailed(let s): return "Pipeline state creation failed: \(s)"
        case .textureAllocationFailed:      return "MTLDevice.makeTexture() returned nil"
        case .bufferAllocationFailed(let s): return "MTLDevice.makeBuffer() failed: \(s)"
        case .invalidInput(let s):          return "Invalid input: \(s)"
        case .pixelBufferFailed(let s):     return "CVPixelBuffer path failed: \(s)"
        }
    }
}

/// A synchronous, allocation-light 2x video upscaling pipeline.
///
/// CNN modes use weights from Anime4K v3.2 (`Upscale_CNN_x2_M` /
/// `Upscale_Denoise_CNN_x2_M`). One instance owns its compute pipelines and a
/// cache of intermediate `rgba16Float` feature textures sized to the current
/// input. Feeding a stream at a constant resolution therefore performs no
/// per-frame texture allocation on the intermediate path.
///
/// Thread safety: not thread-safe. Create one instance per stream / per thread.
public final class VideoUpscalingPipeline {

    public let device: MTLDevice
    public let mode: UpscaleMode

    /// 16x16 is a good default on Apple GPUs for these pointwise/stencil kernels.
    public var threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

    private let commandQueue: MTLCommandQueue
    private let pool: TexturePool

    private var bilinearPSO: MTLComputePipelineState?
    private var conv1PSO: MTLComputePipelineState?
    private var convMidPSO: MTLComputePipelineState?
    private var convLastPSO: MTLComputePipelineState?
    private var depthToSpacePSO: MTLComputePipelineState?

    private struct WeightSet {
        let conv1: [Float]
        let mid: [[Float]]
        let last: [Float]
    }
    private var weights: WeightSet?

    private var weightBuffers: [MTLBuffer] = []

    #if canImport(CoreVideo)
    private var textureCache: CVMetalTextureCache?
    #endif

    public init(device: MTLDevice, mode: UpscaleMode) throws {
        self.device = device
        self.mode = mode
        guard let queue = device.makeCommandQueue() else {
            throw VideoUpscaleError.commandQueueUnavailable
        }
        self.commandQueue = queue
        self.pool = TexturePool(device: device)

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: MetalKernels.source, options: nil)
        } catch {
            throw VideoUpscaleError.libraryCompilationFailed(String(describing: error))
        }

        func makePSO(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw VideoUpscaleError.functionNotFound(name)
            }
            do {
                return try device.makeComputePipelineState(function: fn)
            } catch {
                throw VideoUpscaleError.pipelineCreationFailed("\(name): \(error)")
            }
        }

        bilinearPSO = try makePSO("bilinear2x")

        switch mode {
        case .bilinear:
            break
        case .cnnX2M:
            weights = WeightSet(conv1: CNNWeights.conv1,
                                mid: CNNWeights.convMid,
                                last: CNNWeights.convLast)
            try buildCNNPSOs(makePSO)
        case .cnnDenoiseX2M:
            weights = WeightSet(conv1: CNNDenoiseWeights.conv1,
                                mid: CNNDenoiseWeights.convMid,
                                last: CNNDenoiseWeights.convLast)
            try buildCNNPSOs(makePSO)
        }

        if weights != nil {
            try buildWeightBuffers()
        }
    }

    private func buildCNNPSOs(_ makePSO: (String) throws -> MTLComputePipelineState) throws {
        conv1PSO = try makePSO("conv3x3_1to4")
        convMidPSO = try makePSO("conv3x3_8to4")
        convLastPSO = try makePSO("conv1x1_56to4")
        depthToSpacePSO = try makePSO("depth_to_space_add")
    }

    private func buildWeightBuffers() throws {
        guard let w = weights else { return }
        var arrays: [[Float]] = [w.conv1]
        arrays.append(contentsOf: w.mid)
        arrays.append(w.last)
        // A nil buffer used to be dropped by compactMap, so init could succeed
        // with fewer than 8 buffers and only fail later in encodeCNN with a
        // misleading "not initialised". Report the allocation failure here,
        // where it actually happened.
        weightBuffers = try arrays.map { arr -> MTLBuffer in
            let buf = arr.withUnsafeBytes { raw in
                raw.baseAddress.flatMap {
                    device.makeBuffer(bytes: $0, length: raw.count, options: .storageModeShared)
                }
            }
            guard let buf else {
                throw VideoUpscaleError.bufferAllocationFailed("CNN weight buffer, \(arr.count) floats")
            }
            return buf
        }
    }

    // MARK: - Public API

    /// Upscales `input` by exactly 2x into a freshly created texture.
    ///
    /// - Parameters:
    ///   - input: an `MTLTexture` with `usage` including `.shaderRead`. Common
    ///     formats (`rgba8Unorm`, `bgra8Unorm`, `rgba16Float`) all work: Metal
    ///     normalises BGRA reads to RGBA in shaders.
    ///   - outputFormat: defaults to `.bgra8Unorm` for video interop.
    /// - Returns: a new texture of size `2*input.width x 2*input.height`.
    public func upscale(_ input: MTLTexture,
                        outputFormat: MTLPixelFormat = .bgra8Unorm) throws -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: outputFormat,
            width: input.width * 2,
            height: input.height * 2,
            mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        guard let out = device.makeTexture(descriptor: desc) else {
            throw VideoUpscaleError.textureAllocationFailed
        }
        try upscale(input, into: out)
        return out
    }

    /// Allocation-light variant: renders into a caller-provided texture.
    ///
    /// `output` must be exactly `2*input` in both dimensions, have `usage`
    /// including `.shaderWrite`, and a format compatible with float writes
    /// (`bgra8Unorm`, `rgba8Unorm`, `rgba16Float`, ...). Reuse the same output
    /// texture every frame to avoid per-frame allocations entirely.
    public func upscale(_ input: MTLTexture, into output: MTLTexture) throws {
        guard input.width > 0, input.height > 0 else {
            throw VideoUpscaleError.invalidInput("zero-sized input")
        }
        guard output.width == input.width * 2, output.height == input.height * 2 else {
            throw VideoUpscaleError.invalidInput(
                "output must be 2x input (got \(output.width)x\(output.height) for \(input.width)x\(input.height))")
        }
        guard output.usage.contains(.shaderWrite) else {
            throw VideoUpscaleError.invalidInput("output texture is missing .shaderWrite usage")
        }

        // retainedReferences = false: the caller guarantees referenced textures
        // outlive the frame (we commit+wait synchronously), and dropping the
        // default reference retention removes a per-frame bookkeeping cost that
        // showed up as slow RSS growth during long playback.
        let cbDesc = MTLCommandBufferDescriptor()
        cbDesc.retainedReferences = false
        guard let cb = commandQueue.makeCommandBuffer(descriptor: cbDesc) else {
            throw VideoUpscaleError.commandQueueUnavailable
        }
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw VideoUpscaleError.commandQueueUnavailable
        }

        switch mode {
        case .bilinear:
            try encodeBilinear(enc, input: input, output: output)
        case .cnnX2M, .cnnDenoiseX2M:
            try encodeCNN(enc, input: input, output: output)
        }

        enc.endEncoding()
        cb.commit()
        // v1 is intentionally synchronous. For an async/live pipeline, keep the
        // command buffer and drive presentation from a completion handler
        // (or use a shared event) instead of blocking here.
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw VideoUpscaleError.invalidInput("command buffer error: \(err)")
        }
    }

    // MARK: - Encoding

    private func dispatch(_ enc: MTLComputeCommandEncoder, width: Int, height: Int) {
        let tg = threadgroupSize
        let grid = MTLSize(width: (width + tg.width - 1) / tg.width,
                           height: (height + tg.height - 1) / tg.height,
                           depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    }

    private func encodeBilinear(_ enc: MTLComputeCommandEncoder,
                                input: MTLTexture, output: MTLTexture) throws {
        guard let pso = bilinearPSO else { throw VideoUpscaleError.functionNotFound("bilinear2x") }
        enc.setComputePipelineState(pso)
        enc.setTexture(input, index: 0)
        enc.setTexture(output, index: 1)
        dispatch(enc, width: output.width, height: output.height)
    }

    private func encodeCNN(_ enc: MTLComputeCommandEncoder,
                           input: MTLTexture, output: MTLTexture) throws {
        guard weights != nil,
              let c1 = conv1PSO, let cm = convMidPSO,
              let cl = convLastPSO, let d2s = depthToSpacePSO,
              weightBuffers.count == 8 else {
            throw VideoUpscaleError.pipelineCreationFailed("CNN resources not initialised")
        }

        let W = input.width
        let H = input.height
        let feats = pool.featureTextures(width: W, height: H, count: 8)
        guard feats.count == 8 else { throw VideoUpscaleError.textureAllocationFailed }

        // Pass 1: user input -> feat[0]
        enc.setComputePipelineState(c1)
        enc.setTexture(input, index: 0)
        enc.setTexture(feats[0], index: 1)
        enc.setBuffer(weightBuffers[0], offset: 0, index: 0)
        dispatch(enc, width: W, height: H)

        // Passes 2..7: feat[i] -> feat[i+1]
        enc.setComputePipelineState(cm)
        for i in 0..<6 {
            enc.setTexture(feats[i], index: 0)
            enc.setTexture(feats[i + 1], index: 1)
            enc.setBuffer(weightBuffers[1 + i], offset: 0, index: 0)
            dispatch(enc, width: W, height: H)
        }

        // Pass 8: 56-channel 1x1 -> feat[7]
        enc.setComputePipelineState(cl)
        for k in 0..<7 {
            enc.setTexture(feats[k], index: k)
        }
        enc.setTexture(feats[7], index: 7)
        enc.setBuffer(weightBuffers[7], offset: 0, index: 0)
        dispatch(enc, width: W, height: H)

        // Pass 9: depth-to-space + bilinear residual -> output
        enc.setComputePipelineState(d2s)
        enc.setTexture(feats[7], index: 0)
        enc.setTexture(input, index: 1)
        enc.setTexture(output, index: 2)
        dispatch(enc, width: output.width, height: output.height)
    }

    // MARK: - CVPixelBuffer convenience (BGRA in / BGRA out)

    #if canImport(CoreVideo)
    /// Zero-copy-ish path: wraps the input `CVPixelBuffer` (BGRA) and the
    /// pooled output `CVPixelBuffer` as Metal textures via
    /// `CVMetalTextureCache`, runs the pipeline, and returns the output buffer.
    ///
    /// - Parameters:
    ///   - input: a BGRA `CVPixelBuffer` with `kCVPixelBufferMetalCompatibilityKey`.
    ///   - outputPool: pool to allocate the 2x output from. If `nil`, a buffer
    ///     is created directly (slower; prefer a pool in a live pipeline).
    ///   - outputFormat: only `kCVPixelFormatType_32BGRA` is supported.
    @discardableResult
    public func upscale(_ input: CVPixelBuffer,
                        outputPool: CVPixelBufferPool?,
                        outputFormat: OSType = kCVPixelFormatType_32BGRA) throws -> CVPixelBuffer {
        guard outputFormat == kCVPixelFormatType_32BGRA else {
            throw VideoUpscaleError.pixelBufferFailed("only 32BGRA output is supported")
        }
        guard CVPixelBufferGetPixelFormatType(input) == kCVPixelFormatType_32BGRA else {
            throw VideoUpscaleError.pixelBufferFailed("only 32BGRA input is supported")
        }
        guard CVPixelBufferIsPlanar(input) == false else {
            throw VideoUpscaleError.pixelBufferFailed("planar input is not supported")
        }
        let cache = try metalTextureCache()

        let w = CVPixelBufferGetWidth(input)
        let h = CVPixelBufferGetHeight(input)

        var cvIn: CVMetalTexture?
        let s0 = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, input, nil, .bgra8Unorm, w, h, 0, &cvIn)
        guard s0 == kCVReturnSuccess, let cvInRef = cvIn,
              let inTex = CVMetalTextureGetTexture(cvInRef) else {
            throw VideoUpscaleError.pixelBufferFailed("failed to wrap input (status \(s0))")
        }

        var outPB: CVPixelBuffer?
        if let pool = outputPool {
            let s = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &outPB)
            guard s == kCVReturnSuccess else {
                throw VideoUpscaleError.pixelBufferFailed("pool allocation failed (status \(s))")
            }
        } else {
            let attrs: [String: Any] = [
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary,
                kCVPixelBufferPixelFormatTypeKey as String: outputFormat,
            ]
            let s = CVPixelBufferCreate(kCFAllocatorDefault, w * 2, h * 2, outputFormat,
                                        attrs as CFDictionary, &outPB)
            guard s == kCVReturnSuccess else {
                throw VideoUpscaleError.pixelBufferFailed("CVPixelBufferCreate failed (status \(s))")
            }
        }
        guard let outBuffer = outPB else {
            throw VideoUpscaleError.pixelBufferFailed("nil output pixel buffer")
        }

        var cvOut: CVMetalTexture?
        let s1 = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, outBuffer, nil, .bgra8Unorm, w * 2, h * 2, 0, &cvOut)
        guard s1 == kCVReturnSuccess, let cvOutRef = cvOut,
              let outTex = CVMetalTextureGetTexture(cvOutRef) else {
            throw VideoUpscaleError.pixelBufferFailed("failed to wrap output (status \(s1))")
        }

        try upscale(inTex, into: outTex)   // synchronous; cvIn/cvOut stay alive here
        CVMetalTextureCacheFlush(cache, 0)
        return outBuffer
    }

    private func metalTextureCache() throws -> CVMetalTextureCache {
        if let c = textureCache { return c }
        var created: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &created)
        guard status == kCVReturnSuccess, let c = created else {
            throw VideoUpscaleError.pixelBufferFailed("CVMetalTextureCacheCreate failed (status \(status))")
        }
        textureCache = c
        return c
    }
    #endif
}

/// Caches the eight `rgba16Float` feature textures for the current input size.
/// Rebuilt only when the resolution changes, so a steady stream allocates
/// nothing per frame.
final class TexturePool {
    private let device: MTLDevice
    private var cachedWidth = 0
    private var cachedHeight = 0
    private var cachedCount = 0
    private var textures: [MTLTexture] = []

    init(device: MTLDevice) {
        self.device = device
    }

    func featureTextures(width: Int, height: Int, count: Int) -> [MTLTexture] {
        if width == cachedWidth, height == cachedHeight, count == cachedCount {
            return textures
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width,
            height: height,
            mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        textures = (0..<count).compactMap { _ in device.makeTexture(descriptor: desc) }
        cachedWidth = width
        cachedHeight = height
        cachedCount = count
        return textures
    }
}
