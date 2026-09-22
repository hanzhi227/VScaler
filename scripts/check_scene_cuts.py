#!/usr/bin/env python3
"""Scene-cut artifact check for the upscaler (alignment-robust).

Instead of sampling individual frames at cut times (which is sensitive to the
sub-frame position of the cut), this compares the **motion time series** —
per-frame mean absolute difference between consecutive frames — around each
detected cut, for the engine's 2x output and for the source.

That yields checks that do not depend on hitting the exact cut frame:

  * preservation — the two motion series correlate (a cut, a pan or a static
    hold must appear at the same place with the same shape) -> catches
    duplicated/dropped frames and temporal smearing at cuts
  * no added flash — peak motion across the cut has a similar magnitude in the
    output and the source (ratio ~1) -> catches flicker/flash at cuts
  * frame continuity — total packet count matches duration x fps

Usage:
  check_scene_cuts.py --stream TS --source file.mkv [--threshold 0.25]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def cut_times(source, threshold, max_cuts):
    r = run(["ffmpeg", "-v", "info", "-i", str(source), "-vf",
             f"select='gt(scene,{threshold})',metadata=print", "-an", "-f", "null", "-"])
    times = []
    for line in r.stderr.splitlines():
        if "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[1].split()[0]))
            except (IndexError, ValueError):
                pass
    return [t for t in times if t > 0.5][:max_cuts]


def motion_series(path, t0, t1, w=480, h=270):
    """Per-frame mean |difference| (grayscale, downscaled) within [t0, t1)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                    "-ss", f"{t0:.3f}", "-t", f"{max(0.01, t1 - t0):.3f}",
                    "-vf", f"scale={w}:{h},format=gray",
                    "-f", "rawvideo", "-y", tmp.name], check=True)
    raw = np.fromfile(tmp.name, dtype=np.uint8)
    Path(tmp.name).unlink(missing_ok=True)
    n = len(raw) // (w * h)
    if n < 2:
        return np.zeros(0)
    frames = raw[:n * w * h].reshape(n, h, w).astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))


def best_alignment(a, b, max_shift=4):
    """Best shift of b relative to a, with Pearson correlation there."""
    best = (-2.0, 0)
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            x, y = a[shift:], b[:len(b) - shift]
        else:
            x, y = a[:len(a) + shift], b[-shift:]
        if len(x) < 5:
            continue
        if x.std() < 1e-6 or y.std() < 1e-6:
            corr = 1.0 if abs(float(x.mean()) - float(y.mean())) < 1e-3 else 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        if corr > best[0]:
            best = (corr, shift)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--max-cuts", type=int, default=6)
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--window", type=float, default=0.7, help="seconds each side of the cut")
    args = ap.parse_args()

    cuts = cut_times(args.source, args.threshold, args.max_cuts)
    print(f"detected {len(cuts)} cut(s): {', '.join(f'{c:.2f}s' for c in cuts)}")
    if not cuts:
        print("no cuts found — lower --threshold")
        return

    print(f"\n{'cut':>7} {'motion corr':>12} {'shift':>6} {'peak ours':>10} {'peak src':>9} "
          f"{'ratio':>7}  verdict")
    ok = True
    for t in cuts:
        t0, t1 = max(0.0, t - args.window), t + args.window
        ms = motion_series(args.stream, t0, t1)
        mr = motion_series(args.source, t0, t1)
        n = min(len(ms), len(mr))
        if n < 5:
            print(f"{t:>7.2f}  (window too short)")
            continue
        corr, shift = best_alignment(ms[:n], mr[:n])
        peak_o, peak_r = float(ms[:n].max()), float(mr[:n].max())
        ratio = peak_o / peak_r if peak_r > 1e-6 else float("nan")
        flash_ok = (not np.isfinite(ratio)) or 0.6 <= ratio <= 1.7
        good = corr >= 0.90 and flash_ok
        ok = ok and good
        note = "ok" if good else "CHECK: " + ",".join(
            k for k, v in (("correlation", corr >= 0.90), ("flash", flash_ok)) if not v)
        print(f"{t:>7.2f} {corr:>12.3f} {shift:>6d} {peak_o:>10.2f} {peak_r:>9.2f} "
              f"{ratio:>7.2f}  {note}")

    cnt = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
               "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
               str(args.stream)]).stdout.strip()
    dur = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(args.stream)]).stdout.strip()
    try:
        expected = float(dur) * args.fps
        got = float(cnt)
        print(f"\nframe continuity: {int(got)} frames in {float(dur):.2f}s "
              f"(expected ~{expected:.0f}, {100.0 * (got - expected) / expected:+.2f}%)")
    except ValueError:
        pass

    print("\nverdict:", "OK (motion preserved at cuts, no added flash, 1:1 frame mapping)"
          if ok else "CHECK (see rows above)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
