import Foundation
import Metal
import MetalPerformanceShadersGraph

public enum RealESRGANError: Error, CustomStringConvertible {
    case metalUnavailable
    case weightsUndecodable
    case graphBuildFailed(String)
    case geometryMismatch(String)
    case bufferAllocationFailed(String)

    public var description: String {
        switch self {
        case .metalUnavailable:            return "no Metal device or command queue"
        case .weightsUndecodable:          return "the embedded weight blob did not decode"
        case .graphBuildFailed(let s):     return "MPSGraph could not build the network: \(s)"
        case .geometryMismatch(let s):     return "invalid geometry: \(s)"
        case .bufferAllocationFailed(let s): return "buffer allocation failed: \(s)"
        }
    }
}

/// A Real-ESRGAN `SRVGGNetCompact` checkpoint — a stack of 3x3 convolutions with
/// PReLU and a fixed 4x output — run as a live scaler. Both released video
/// checkpoints use this class; which one is decided by the `RealESRGANModel` it is
/// given (`animevideov3`, 16 body convolutions and 621,424 parameters, or
/// `general-x4v3`, 32 and 1,213,296).
///
/// **The convolutions run through MPSGraph, not a hand-written kernel.** That is
/// a measurement, not a preference. This GPU's fp16 SIMT ceiling is 8.4 TFLOP/s
/// (`scripts/bench_alu_throughput.swift`), `simdgroup_matrix` measured *slower*
/// at 4.3, and the network needs 160.7 GFLOP at 480x270 — so a perfect
/// hand-written kernel would reach ~19 ms and a realistic one 30 ms or worse,
/// while MPS is already at ~7.1 TFLOP/s effective on this exact model. There is
/// no headroom for a port to win, so the model is expressed as a graph and the
/// two things MPSGraph cannot do are done in Metal.
///
/// Two Metal kernels bracket the graph: one turns a decoded BGRA frame into the
/// NHWC fp16 tensor the graph wants, and one turns the graph's 48-channel output
/// into the 4x picture with the network's nearest-neighbour residual added.
///
/// Fixed at 4x — there is no x2 checkpoint for this model, so the app offers it
/// only where 4x output fits the encoder's limits (see `ScalerKind.realESRGAN`).
public final class RealESRGANPipeline {

    public static let scale = 4

    /// The graph is shape-specialised, so it is built and compiled for the first
    /// geometry it sees and rebuilt if the frame size ever changes — the same
    /// pattern `MetalFXScaler` uses, and for the same reason: the factory is
    /// handed a kind and a factor, not a frame size.
    private struct Geometry: Equatable { let width: Int, height: Int }

    public let device: MTLDevice
    public let model: RealESRGANModel
    private let queue: MTLCommandQueue
    private let graphDevice: MPSGraphDevice
    private let framePSO: MTLComputePipelineState
    private let shufflePSO: MTLComputePipelineState
    private let weightBlob: Data

    private var graph: MPSGraph?
    private var executable: MPSGraphExecutable?
    private var builtFor: Geometry?
    private var inputBuffer: MTLBuffer?
    private var featureBuffer: MTLBuffer?

    public init(model: RealESRGANModel, device: MTLDevice) throws {
        self.device = device
        self.model = model
        guard let q = device.makeCommandQueue() else { throw RealESRGANError.metalUnavailable }
        self.queue = q
        self.graphDevice = MPSGraphDevice(mtlDevice: device)

        // The blob is one wrapped multi-line literal, so the newlines have to be
        // ignored rather than treated as bad input.
        guard let blob = Data(base64Encoded: model.blobBase64,
                              options: .ignoreUnknownCharacters) else {
            throw RealESRGANError.weightsUndecodable
        }
        guard blob.count == model.blobByteCount else {
            throw RealESRGANError.weightsUndecodable
        }
        self.weightBlob = blob

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: RealESRGANKernels.source, options: nil)
        } catch {
            throw RealESRGANError.graphBuildFailed("Metal library: \(error)")
        }
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw RealESRGANError.graphBuildFailed("missing function \(name)")
            }
            do { return try device.makeComputePipelineState(function: fn) }
            catch { throw RealESRGANError.graphBuildFailed("\(name): \(error)") }
        }
        self.framePSO = try pso("frameToNHWC4")
        self.shufflePSO = try pso("shuffleResidual")
    }

    // MARK: - Graph construction

    private func build(for g: Geometry) throws {
        let w = g.width, h = g.height
        let graph = MPSGraph()

        let src = graph.placeholder(shape: [1, NSNumber(value: h), NSNumber(value: w), 4],
                                    dataType: .float16, name: "src")
        var x = src

        for conv in model.convs {
            let wCount = conv.outCh * conv.inCh * 9
            let wData = weightBlob.subdata(in: conv.weightOffset ..< conv.weightOffset + wCount * 2)
            let bData = weightBlob.subdata(in: conv.biasOffset ..< conv.biasOffset + conv.outCh * 2)

            let wTensor = graph.constant(wData,
                                         shape: [NSNumber(value: conv.outCh),
                                                 NSNumber(value: conv.inCh), 3, 3],
                                         dataType: .float16)
            let bTensor = graph.constant(bData, shape: [NSNumber(value: conv.outCh)],
                                         dataType: .float16)

            guard let desc = MPSGraphConvolution2DOpDescriptor(
                strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1,
                paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1,
                paddingStyle: .explicit, dataLayout: .NHWC, weightsLayout: .OIHW) else {
                throw RealESRGANError.graphBuildFailed("convolution descriptor")
            }

            x = graph.convolution2D(x, weights: wTensor, descriptor: desc, name: nil)
            x = graph.addition(x, bTensor, name: nil)

            if conv.hasActivation {
                // PReLU(x) = max(x,0) + slope * min(x,0), and
                // min(x,0) = -max(-x,0), so this is two ReLUs and a subtraction
                // rather than a scalar constant MPSGraph would have to broadcast.
                let sData = weightBlob.subdata(in: conv.actOffset ..< conv.actOffset + conv.outCh * 2)
                let sTensor = graph.constant(sData, shape: [NSNumber(value: conv.outCh)],
                                             dataType: .float16)
                let pos = graph.reLU(with: x, name: nil)
                let neg = graph.reLU(with: graph.negative(with: x, name: nil), name: nil)
                x = graph.subtraction(pos, graph.multiplication(neg, sTensor, name: nil), name: nil)
            }
        }

        let inBytes = w * h * 4 * 2
        let featBytes = w * h * model.outputChannels * 2
        guard let inBuf = device.makeBuffer(length: inBytes, options: .storageModeShared) else {
            throw RealESRGANError.bufferAllocationFailed("input tensor \(inBytes) bytes")
        }
        guard let featBuf = device.makeBuffer(length: featBytes, options: .storageModeShared) else {
            throw RealESRGANError.bufferAllocationFailed("feature tensor \(featBytes) bytes")
        }

        // `compile` wants shapes, not a concrete tensor data: it is what fixes
        // the geometry the executable is specialised for.
        let shaped = MPSGraphShapedType(
            shape: [1, NSNumber(value: h), NSNumber(value: w), 4], dataType: .float16)

        self.graph = graph
        self.inputBuffer = inBuf
        self.featureBuffer = featBuf
        self.executable = graph.compile(with: graphDevice, feeds: [src: shaped],
                                        targetTensors: [x], targetOperations: nil,
                                        compilationDescriptor: nil)
        self.builtFor = g
    }

    // MARK: - Run

    /// Up to scale x4. `dst` must be exactly 4x `src` and have `.shaderWrite`.
    public func upscale(_ src: MTLTexture, into dst: MTLTexture) throws {
        let geometry = Geometry(width: src.width, height: src.height)
        if builtFor != geometry { try build(for: geometry) }

        guard let executable, let inputBuffer, let featureBuffer else {
            throw RealESRGANError.graphBuildFailed("not built")
        }
        guard dst.width == src.width * Self.scale, dst.height == src.height * Self.scale else {
            throw RealESRGANError.geometryMismatch(
                "4x is fixed: got \(src.width)x\(src.height) -> \(dst.width)x\(dst.height)")
        }

        let cb = queue.makeCommandBuffer()
        cb?.label = "real-esrgan frame->tensor"
        encode(framePSO, into: cb, src: src, dst: nil, buffer: inputBuffer,
               width: src.width, height: src.height)
        cb?.commit()

        let inTD = MPSGraphTensorData(inputBuffer,
                                      shape: [1, NSNumber(value: src.height),
                                              NSNumber(value: src.width), 4],
                                      dataType: .float16)
        let outTD = MPSGraphTensorData(
            featureBuffer,
            shape: [1, NSNumber(value: src.height), NSNumber(value: src.width),
                    NSNumber(value: model.outputChannels)],
            dataType: .float16)
        // inputs/results are positional: one feed, one target, so there is
        // nothing to disambiguate. The descriptor makes the otherwise
        // asynchronous call block, which is what a `FrameScaler` has to do —
        // the player hands us a frame and expects the output to be there.
        let execDesc = MPSGraphExecutableExecutionDescriptor()
        execDesc.waitUntilCompleted = true
        _ = executable.run(with: queue, inputs: [inTD], results: [outTD],
                           executionDescriptor: execDesc)

        let cb2 = queue.makeCommandBuffer()
        cb2?.label = "real-esrgan tensor->4x"
        encode(shufflePSO, into: cb2, src: src, dst: dst, buffer: featureBuffer,
               width: dst.width, height: dst.height)
        // The queue is FIFO, so completing this buffer means the graph's work is
        // done too — one synchronisation point per frame rather than three.
        cb2?.commit()
        cb2?.waitUntilCompleted()
        if let err = cb2?.error { throw err }
    }

    private func encode(_ pso: MTLComputePipelineState, into cb: MTLCommandBuffer?,
                        src: MTLTexture, dst: MTLTexture?, buffer: MTLBuffer,
                        width: Int, height: Int) {
        guard let cb, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setTexture(src, index: 0)
        if let dst { enc.setTexture(dst, index: 1) }
        enc.setBuffer(buffer, offset: 0, index: 0)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        enc.dispatchThreadgroups(
            MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1),
            threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}
