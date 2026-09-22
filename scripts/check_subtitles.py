#!/usr/bin/env python3
"""Subtitle burn-in verification for the MacVideoUpscaler engine.

Checks, for a stream captured with subtitle burn-in enabled:

  1. presence/timing — at a cue time the subtitle region differs from the same
     frame captured without subtitles; between cues it does not.
  2. fidelity — the burned frame is compared against a reference built the same
     way in plain ffmpeg (subtitle burned at source resolution, then 2x bicubic):
     subtitle-region MAE (faithfulness) and gradient energy ratio (sharpness).

Usage:
  check_subtitles.py --subs ON.ts --plain OFF.ts --times 3,12 --blank 8
                     --source file.mkv --ref-filter "<ffmpeg -vf/-filter_complex spec>"
                     --src-size 640x480 --bottom 0.28
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def frame(ts, t):
    d = tempfile.mkdtemp()
    out = Path(d) / "f.png"
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(ts),
                    "-frames:v", "1", "-y", str(out)], check=True)
    return np.asarray(Image.open(out).convert("L"), dtype=np.float32)


def grad_energy(img):
    gy, gx = np.gradient(img)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", required=True, help="TS captured WITH subtitle burn-in")
    ap.add_argument("--plain", required=True, help="TS captured WITHOUT subtitles")
    ap.add_argument("--times", required=True, help="cue times, comma separated")
    ap.add_argument("--blank", required=True, type=float, help="a time between cues")
    ap.add_argument("--source", help="source file, to build the plain-ffmpeg reference")
    ap.add_argument("--ref-args", help="extra ffmpeg args that burn subs at source res")
    ap.add_argument("--src-size", help="WxH of the source video (for the 2x reference)")
    ap.add_argument("--bottom", type=float, default=0.28,
                    help="fraction of frame height to treat as the subtitle region")
    args = ap.parse_args()

    cue_times = [float(x) for x in args.times.split(",")]
    ok = True

    def region(img):
        h = img.shape[0]
        return img[int(h * (1 - args.bottom)):, :]

    print(f"{'time':>6} {'bottom MAE':>11} {'top MAE':>9}  {'verdict':<22}")
    for t in cue_times + [args.blank]:
        a = frame(args.subs, t)
        b = frame(args.plain, t)
        h = a.shape[0]
        bot = float(np.abs(region(a) - region(b)).mean())
        top = float(np.abs(a[:int(h * (1 - args.bottom))] - b[:int(h * (1 - args.bottom))]).mean())
        # Presence is judged by the bottom region standing out from the rest of
        # the frame (absolute delta varies with cue length/contrast).
        present = bot > 1.5 and bot > 4 * max(top, 1e-6)
        is_blank = abs(t - args.blank) < 1e-9
        if is_blank:
            verdict = "absent (as expected)" if not present else "PRESENT but should be absent"
            if present:
                ok = False
        else:
            verdict = "present" if present else "MISSING"
            if not present:
                ok = False
        print(f"{t:>6.1f} {bot:>11.2f} {top:>9.2f}  {verdict:<22}")

    if args.source and args.ref_args and args.src_size:
        w, h = (int(v) for v in args.src_size.lower().split("x"))
        print("\nfidelity vs plain-ffmpeg reference (burned at source res, then 2x bicubic)")
        print(f"{'time':>6} {'sub-region MAE':>15} {'sharpness ours':>15} {'ref':>8} {'ratio':>7}")
        for t in cue_times:
            d = tempfile.mkdtemp()
            ref = Path(d) / "ref.png"
            # --ref-args is a complete filter_complex that burns the subtitles at
            # source resolution and ends with the 2x bicubic upscale into [v].
            cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", args.source,
                   "-filter_complex", args.ref_args, "-map", "[v]",
                   "-frames:v", "1", "-y", str(ref)]
            subprocess.run(cmd, check=True)
            r = np.asarray(Image.open(ref).convert("L"), dtype=np.float32)
            o = frame(args.subs, t)
            if r.shape != o.shape:
                r = np.asarray(Image.fromarray(r.astype(np.uint8)).resize(
                    (o.shape[1], o.shape[0]), Image.BICUBIC), dtype=np.float32)
            mae = float(np.abs(region(r) - region(o)).mean())
            so, sr = grad_energy(region(o)), grad_energy(region(r))
            ratio = so / max(sr, 1e-6)
            flag = "" if ratio >= 0.95 else "  <-- softer than reference"
            print(f"{t:>6.1f} {mae:>15.2f} {so:>15.2f} {sr:>8.2f} {ratio:>7.2f}{flag}")
            if ratio < 0.9:
                ok = False

    print("\nverdict:", "OK" if ok else "CHECK")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
