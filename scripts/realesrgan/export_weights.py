#!/usr/bin/env python3
"""Export an `SRVGGNetCompact` checkpoint to a flat fp16 blob the Metal app can load.

Both released video checkpoints go through this — `realesr-animevideov3` (16 body
convolutions) and `realesr-general-x4v3` (32); `--all` writes both, and each
becomes a live mode in the app.

The app has no model file and no runtime dependency: its Metal kernels are
compiled from source strings inside the binary and its weights are baked into
the source too (that is why `scripts/make_app.sh` can say the bundle is
"the executable and Info.plist"). A megabyte of weights would be an unpleasant
Swift literal, so the generated file carries one base64 string and a table of
byte offsets into it; the Swift side decodes it once into a single MTLBuffer.

Layout, per convolution, in order:
    weights  O x I x 3 x 3   fp16   (PyTorch/ONNX OIHW, which is what MPSGraph
                                     takes for convolution2D)
    bias     O               fp16
    slope    O               fp16   (PReLU, on every conv but the last)
Every section starts on a 256-byte boundary, because Metal wants buffer
offsets aligned and this way each conv can be bound with `setBuffer(offset:)`
from one shared MTLBuffer.

Weight values are unmodified from the checkpoint; the *layout* is what changes.
Precision is fp16 for weights and activations alike, which is the only way this
is real time on this GPU (measured: fp16 8.4 TFLOP/s vs fp32 4.8 in
`scripts/bench_alu_throughput.swift`). `verify.py` measures what that costs in
dB against the fp32 reference rather than assuming it is free.

  python3 scripts/realesrgan/export_weights.py [--variant NAME | --all] [--check]
"""
import argparse
import base64
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np
import torch

import srvgg

ALIGN = 256
LIB_DIR = os.path.join(ROOT, "engine", "RealESRGANMetal", "Sources", "RealESRGANMetal")

# Both released video checkpoints are the same class at different depths, so the
# only thing that varies here is num_conv, the symbol and the file name. Both now
# land in the `RealESRGANMetal` library, because both are selectable live modes —
# `general-x4v3` is offered so it can be judged on a real clip rather than on the
# scores alone.
#
# That is a deliberate cost, not tidiness. Swift does not dead-strip a large
# literal from a public enum, so carrying the second blob adds ~3.3 MB to the app
# binary (3,160,784 -> 6,464,752 bytes). It previously lived in the `resrbench`
# executable, out of the app's reach, because it was bench-only.
#
# One caution from that episode: measure binary size against a build that
# succeeded. The first check ran against the binary left over from a *failed*
# build and reported the previous build's size and a clean absence.
VARIANTS = {
    "animevideov3": dict(num_conv=16, file="realesr-animevideov3.pth",
                         symbol="RealESRGANAnimeV3Weights",
                         out="RealESRGANWeights.swift", dir=LIB_DIR,
                         name="animevideov3"),
    "general-x4v3": dict(num_conv=32, file="realesr-general-x4v3.pth",
                         symbol="RealESRGANGeneralWeights",
                         out="RealESRGANGeneralWeights.swift", dir=LIB_DIR,
                         name="general-x4v3"),
}


def layers_from(model):
    """The 18 convolutions in execution order, with the activation that follows
    each one folded in as its PReLU slope.

    `nn.Sequential` is [conv, prelu] x 17 then a final conv with no activation,
    so the slope that follows conv i belongs to conv i and not to conv i+1 —
    getting that off by one is invisible in the shapes and changes the picture.
    """
    seq = list(model.body)
    out = []
    for i, m in enumerate(seq):
        if isinstance(m, torch.nn.Conv2d):
            act = None
            if i + 1 < len(seq) and isinstance(seq[i + 1], torch.nn.PReLU):
                act = seq[i + 1].weight.detach()
            out.append((m, act))
    return out


def build_blob(model):
    """-> (bytes, list of dicts). Returns the blob and its section table."""
    blob = bytearray()
    table = []

    def put(arr):
        """Appends fp16 data at a 256-byte boundary; returns the offset."""
        nonlocal blob
        while len(blob) % ALIGN:
            blob.append(0)
        off = len(blob)
        blob += np.ascontiguousarray(arr, dtype=np.float16).tobytes()
        return off

    for conv, act in layers_from(model):
        o, i, kh, kw = conv.weight.shape
        w = conv.weight.detach().numpy()
        if not table:
            # The first conv takes the frame; pad its 3 input channels to 4 so
            # the graph's input tensor and the kernel that fills it are one
            # aligned half4 per pixel. The extra column is zero, so this changes
            # the arithmetic only by an add of zero.
            padded = np.zeros((o, 4, kh, kw), dtype=w.dtype)
            padded[:, :i] = w
            w, i = padded, 4
        entry = {
            "outCh": o, "inCh": i,
            "hasActivation": act is not None,
            "weightOffset": put(w),
            "biasOffset": put(conv.bias.detach().numpy()),
            "actOffset": 0,
        }
        if act is not None:
            entry["actOffset"] = put(act.numpy())
        table.append(entry)

    return bytes(blob), table


def swift_source(blob, table, v):
    """Emit the generated Swift file.

    The multiline-literal delimiters are built from a constant rather than
    written inline: three adjacent quote characters inside a Python string are
    exactly the thing that is easy to get wrong, and getting it wrong produces a
    file that looks fine until Swift reads it.
    """
    Q3 = chr(34) * 3
    lines = []
    lines.append("// Generated by scripts/realesrgan/export_weights.py -- DO NOT EDIT BY HAND.")
    lines.append("// Source: %s (Real-ESRGAN v0.2.5.0, BSD-3-Clause)." % v["file"])
    lines.append("//")
    lines.append("// The blob is fp16 little-endian; see the exporter for the section")
    lines.append("// layout. Offsets are byte offsets into the decoded blob, each rounded up")
    lines.append("// to 256 bytes so a conv's weights, bias and PReLU slope can be bound")
    lines.append("// straight out of one MTLBuffer.")
    # public so `resrbench` can reach it: that is a separate module, and an
    # internal enum is invisible across the boundary.
    lines.append("public enum %s {" % v["symbol"])
    lines.append("    public static let model = RealESRGANModel(")
    lines.append("        name: " + chr(34) + v["name"] + chr(34) + ",")
    lines.append("        bodyConvs: %d," % v["num_conv"])
    lines.append("        outputChannels: %d," % (3 * 4 * 4))
    lines.append("        blobByteCount: %d," % len(blob))
    lines.append("        blobBase64: " + Q3)
    b64 = base64.b64encode(blob).decode("ascii")
    # One multiline literal, wrapped for readability. It has to be a *single*
    # literal: the first version emitted `"..." + "..."` per line and
    # swift-frontend sat at 100% CPU for eleven minutes on the resulting
    # 14,000-term expression before it was killed. The Swift side decodes with
    # .ignoreUnknownCharacters, so the newlines cost nothing.
    for i in range(0, len(b64), 120):
        lines.append("        " + b64[i:i + 120])
    lines.append("        " + Q3 + ",")
    lines.append("        convs: [")
    for e in table:
        lines.append("            .init(outCh: %d, inCh: %d, hasActivation: %s, "
                     "weightOffset: %d, biasOffset: %d, actOffset: %d),"
                     % (e["outCh"], e["inCh"], "true" if e["hasActivation"] else "false",
                        e["weightOffset"], e["biasOffset"], e["actOffset"]))
    lines.append("        ])")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="animevideov3", choices=sorted(VARIANTS))
    ap.add_argument("--all", action="store_true", help="export every variant")
    ap.add_argument("--check", action="store_true",
                    help="only verify the round trip against PyTorch, write nothing")
    a = ap.parse_args()

    rc = 0
    for name in (sorted(VARIANTS) if a.all else [a.variant]):
        rc |= export(name, a.check)
    return rc


def export(name, check_only):
    v = VARIANTS[name]
    path = srvgg.variant_path(name)
    try:
        model = srvgg.load(path, num_conv=v["num_conv"])
    except FileNotFoundError as e:
        raise SystemExit("weights not found (%s)\n"
                         "run scripts/realesrgan/fetch_weights.py first" % e)

    blob, table = build_blob(model)
    nparam = sum(t["outCh"] * t["inCh"] * 9 + t["outCh"]
                 + (t["outCh"] if t["hasActivation"] else 0) for t in table)
    print("== %s" % name)
    print("   convolutions: %d  (%d activated)  body: %d"
          % (len(table), sum(t["hasActivation"] for t in table), v["num_conv"]))
    print("   scalars: %d   blob: %d bytes (%.2f MB)  base64: %.2f MB"
          % (nparam, len(blob), len(blob) / 1e6, len(blob) * 4 / 3 / 1e6))

    # Round trip: rebuild the network from the exported fp16 blob and check the
    # output against the original fp32 model. Doing this here means the Swift
    # side only has to match a reference already known to be right, so a
    # mismatch there is a kernel bug rather than a layout bug.
    flat = np.frombuffer(blob, dtype=np.float16)

    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 48)
    with torch.no_grad():
        ref = model(x).clamp(0, 1)

    # Recompute from the exported fp16 tensors, same conv order, fp32 maths.
    # The first conv's weights carry a 4th zero input channel, so the input is
    # padded to match — the same thing the Metal kernel's zero 4th component does.
    y = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 1))
    for e in table:
        w = torch.from_numpy(flat[e["weightOffset"] // 2:
                                   e["weightOffset"] // 2 + e["outCh"] * e["inCh"] * 9]
                             .reshape(e["outCh"], e["inCh"], 3, 3).astype(np.float32))
        b = torch.from_numpy(flat[e["biasOffset"] // 2:
                                  e["biasOffset"] // 2 + e["outCh"]].astype(np.float32))
        y = torch.nn.functional.conv2d(y, w, b, padding=1)
        if e["hasActivation"]:
            s_ = torch.from_numpy(flat[e["actOffset"] // 2:
                                       e["actOffset"] // 2 + e["outCh"]].astype(np.float32))
            y = torch.maximum(y, torch.zeros_like(y)) + s_.view(1, -1, 1, 1) * torch.minimum(y, torch.zeros_like(y))
    ps = torch.nn.PixelShuffle(4)
    got = (ps(y) + torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")).clamp(0, 1)
    err = (got - ref).abs()
    psnr = 10 * np.log10(1.0 / max(float(err.pow(2).mean()), 1e-20))
    print("   export round trip vs fp32 model: max |err| %.5f  PSNR %.1f dB"
          % (float(err.max()), psnr))

    if check_only:
        return 0

    out_path = os.path.join(v["dir"], v["out"])
    os.makedirs(v["dir"], exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(swift_source(blob, table, v))
    print("   wrote %s (%.2f MB)" % (os.path.relpath(out_path, ROOT),
                                     os.path.getsize(out_path) / 1e6))

    manifest = os.path.join(HERE, "weights_manifest_%s.json" % v["name"])
    with open(manifest, "w") as fh:
        json.dump({"name": v["name"], "numConv": v["num_conv"], "scale": 4,
                   "blobByteCount": len(blob), "parameters": nparam,
                   "convs": table}, fh, indent=1)
    print("   wrote %s" % os.path.relpath(manifest, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
