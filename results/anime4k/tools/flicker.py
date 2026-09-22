#!/usr/bin/env python3
"""Temporal amplification ratio for the seq_action 12-frame sequence.

For each preset: render f01..f12 through the preset to 2x PNGs, then compute
  M(out) = mean over consecutive pairs of mean|out[i+1]-out[i]|   (0-255, RGB mean)
  M(in)  = same on the 480p source frames
  temporal amplification ratio = M(out) / M(in)
~1.0 ideal; > 1.05 flagged as flicker risk.

Rendering is done by the caller (render_seqs.sh); this script only consumes a
directory of rendered frames named <base>_<preset>.png (or per-preset subdirs).
"""
import sys, glob, os
import numpy as np
from PIL import Image


def pair_metric(paths):
    prev = None
    vals = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
        if prev is not None:
            vals.append(np.abs(a - prev).mean())
        prev = a
    return float(np.mean(vals)) if vals else float("nan")


def main():
    src_dir = sys.argv[1]          # dir with f01..f12 source PNGs
    rendered_root = sys.argv[2]    # results/anime4k
    presets = sys.argv[3].split(",")
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 12

    src = sorted(glob.glob(os.path.join(src_dir, "f*.png")))[:n]
    if len(src) < 2:
        print("no source frames found", file=sys.stderr)
        sys.exit(1)
    m_in = pair_metric(src)
    print(f"input_pair_metric={m_in:.6f}  ({len(src)} frames)")
    rows = []
    for preset in presets:
        # rendered files: <root>/<preset>/seq_action_fNN.png
        d = os.path.join(rendered_root, preset)
        rp = sorted(glob.glob(os.path.join(d, "seq_f*.png")))
        if len(rp) < 2:
            # fall back: any seq_* files
            rp = sorted(glob.glob(os.path.join(d, "seq_*.png")))
        m_out = pair_metric(rp) if len(rp) >= 2 else float("nan")
        ratio = m_out / m_in if m_in else float("nan")
        rows.append((preset, len(rp), m_out, ratio))
        print(f"{preset:12s} frames={len(rp):2d} out_metric={m_out:.6f} ratio={ratio:.4f}")
    # machine-readable
    import json
    print("JSON:" + json.dumps({
        "input_pair_metric": m_in,
        "results": [{"preset": r[0], "frames": r[1], "out_pair_metric": r[2], "ratio": r[3]} for r in rows],
    }))


if __name__ == "__main__":
    main()
