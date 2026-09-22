import Foundation

/// One Real-ESRGAN checkpoint, as the pipeline needs it.
///
/// Both released `SRVGGNetCompact` video checkpoints are the same network at
/// different depths — `animevideov3` has 16 body convolutions and `general-x4v3`
/// has 32 — so the pipeline is written against this descriptor rather than
/// against one checkpoint's constants. Each generated weights file provides one
/// instance; `scripts/realesrgan/export_weights.py` writes them.
public struct RealESRGANModel {

    public struct Conv {
        public let outCh: Int, inCh: Int, hasActivation: Bool
        public let weightOffset: Int, biasOffset: Int, actOffset: Int

        // Explicit, because the generated weights live in a different module for
        // the benchmark tool and the memberwise initialiser is internal.
        public init(outCh: Int, inCh: Int, hasActivation: Bool,
                    weightOffset: Int, biasOffset: Int, actOffset: Int) {
            self.outCh = outCh; self.inCh = inCh; self.hasActivation = hasActivation
            self.weightOffset = weightOffset; self.biasOffset = biasOffset
            self.actOffset = actOffset
        }
    }

    /// Checkpoint name, for labels and diagnostics.
    public let name: String
    /// Body convolutions, excluding the input and output ones. 16 or 32.
    public let bodyConvs: Int
    /// Channels the last convolution produces (3 x scale x scale).
    public let outputChannels: Int
    /// Total bytes of the decoded blob; checked against the decode.
    public let blobByteCount: Int
    /// The fp16 blob, one wrapped base64 literal.
    public let blobBase64: String
    /// The convolutions in execution order, with their PReLU folded in.
    public let convs: [Conv]

    public init(name: String, bodyConvs: Int, outputChannels: Int,
                blobByteCount: Int, blobBase64: String, convs: [Conv]) {
        self.name = name; self.bodyConvs = bodyConvs
        self.outputChannels = outputChannels; self.blobByteCount = blobByteCount
        self.blobBase64 = blobBase64; self.convs = convs
    }

    public var parameterCount: Int {
        convs.reduce(0) { $0 + $1.outCh * $1.inCh * 9 + $1.outCh
            + ($1.hasActivation ? $1.outCh : 0) }
    }

    /// Tensors the pipeline holds for a given source size: the NHWC input and the
    /// `outputChannels`-deep feature tensor the graph writes, both fp16. Called
    /// out because it is the whole of this model's memory cost — the weights are
    /// 2.5/4.9 MB and never the deciding term.
    public func activationBytes(width: Int, height: Int) -> Int {
        width * height * (4 + outputChannels) * 2
    }
}
