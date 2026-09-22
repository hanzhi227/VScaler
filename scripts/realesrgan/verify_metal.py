#!/usr/bin/env python3
"""Does the Metal Real-ESRGAN agree with the PyTorch one?

Two references, because one number cannot separate two different things:

  * fp32 CPU        what the model ideally produces. The gap to this is the
                    honest cost of running in fp16, and it is not a bug.
  * fp16 MPS        the same arithmetic in a different runtime. The gap to
                    *this* is implementation disagreement — a layout mistake, a
                    wrong PReLU index, a broken pixel shuffle — and it should be
                    near the noise floor. If it is not, the port is wrong.

Runs the shipping code path: the frame goes into `resrbench --dump`, which is the
same `RealESRGANPipeline` the player uses.

  python3 scripts/realesrgan/verify_metal.py [--size 480x270] [--source PATH]
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np
import torch

import srvgg

BIN = os.path.join(ROOT, "engine", "RealESRGANMetal", ".build", "release", "resrbench")
DEFAULT_SOURCE = os.path.join(ROOT, "test-media", "general", "naruto_270p_c28.mp4")


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-20))


def load_frame(path, w, h):
    """One frame as BGRA8 bytes at exactly w x h, via ffmpeg (which is how the
    app gets its frames too, so the input is not a special case)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=%d:%d" % (w, h),
           "-frames:v", "1", "-pix_fmt", "bgra", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    if len(out) != w * h * 4:
        raise SystemExit("ffmpeg returned %d bytes, expected %d" % (len(out), w * h * 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="480x270")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--keep", action="store_true", help="keep the raw files")
    a = ap.parse_args()

    w, h = (int(v) for v in a.size.split("x"))
    if not os.path.exists(BIN):
        raise SystemExit("build it first:\n"
                         "  swift build -c release --package-path engine/RealESRGANMetal")
    if not os.path.exists(a.source):
        raise SystemExit("no source at %s (--source PATH)" % a.source)

    raw_in = load_frame(a.source, w, h)
    tmp = tempfile.mkdtemp(prefix="resrverify")
    in_path = os.path.join(tmp, "in.raw")
    out_path = os.path.join(tmp, "out.raw")
    with open(in_path, "wb") as fh:
        fh.write(raw_in)

    r = subprocess.run([BIN, "--dump", in_path, out_path, str(w), str(h)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("resrbench failed:\n%s\n%s" % (r.stdout, r.stderr))
    out_raw = open(out_path, "rb").read()
    ow, oh = w * 4, h * 4
    if len(out_raw) != ow * oh * 4:
        raise SystemExit("resrbench wrote %d bytes, expected %d" % (len(out_raw), ow * oh * 4))

    # BGRA -> RGB float [0,1], HWC.
    got = np.frombuffer(out_raw, dtype=np.uint8).reshape(oh, ow, 4)[:, :, [2, 1, 0]].astype(np.float32) / 255.0

    # The same frame the Metal side saw, as the model's input tensor.
    src = np.frombuffer(raw_in, dtype=np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    x = torch.from_numpy(src).permute(2, 0, 1).unsqueeze(0)

    model32 = srvgg.load(device="cpu")
    with torch.no_grad():
        ref32 = model32(x).clamp(0, 1)[0].permute(1, 2, 0).numpy()

    have_mps = torch.backends.mps.is_available()
    ref16 = None
    if have_mps:
        model16 = srvgg.load(device="mps").half()
        with torch.no_grad():
            ref16 = model16(x.half().to("mps")).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    print("source %s" % a.source)
    print("size   %dx%d -> %dx%d   device %s" % (w, h, ow, oh, torch.backends.mps.is_available() and "mps" or "cpu"))
    print()
    print("Metal vs fp32 CPU   PSNR %6.2f dB   max |err| %.4f" % (psnr(got, ref32), np.abs(got - ref32).max()))
    if ref16 is not None:
        print("Metal vs fp16 MPS   PSNR %6.2f dB   max |err| %.4f" % (psnr(got, ref16), np.abs(got - ref16).max()))
        print("fp16 MPS vs fp32    PSNR %6.2f dB   (the floor fp16 alone imposes)"
              % psnr(ref16, ref32))

    if a.keep:
        print("\nraw files in %s" % tmp)
    else:
        for p in (in_path, out_path):
            os.unlink(p)
        os.rmdir(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
