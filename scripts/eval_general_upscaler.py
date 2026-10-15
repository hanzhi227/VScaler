#!/usr/bin/env python3
"""Measure the app's upscalers against a known-good high-resolution reference.

What is measured is what the *player* actually rendered: the harness drives
`playercheck`, which writes the exact pixel buffers handed to the display layer,
so no model is re-implemented here and nothing is inferred from source code.

The method is the standard super-resolution proxy. A high-resolution reference
is downscaled and re-compressed to make the source; each method upscales that
source back to the reference's geometry; the result is compared with the
reference. That measures how much of the *real* detail a method recovers, and
how much of what it draws is not in the source.

Reported per mode:

  psnr / ssim    fidelity to the reference. Higher = more faithful. A method
                 that invents plausible detail is penalised here by construction.
  acutance       mean |Laplacian| of luma: how much edge energy the output has,
                 as a fraction of the reference's. 1.00 = as sharp as the truth,
                 <1 = softer, >1 = sharper than the real thing (which is where
                 ringing and invented edges show up).
  smooth_rmse    RMSE restricted to the reference's flat regions (the smoothest
                 40% of pixels). Blocking and banding from the source's
                 compression land here; lower means better cleanup.
  flicker        mean absolute frame-to-frame difference over a consecutive run
                 of frames, as a ratio to the bypass (bilinear) run, which is
                 temporally neutral by construction. 1.00 = the model adds no
                 temporal instability; >1 means it does. This is the metric that
                 still-image comparisons cannot show.

Usage:
  scripts/eval_general_upscaler.py \
      --source test-media/liveaction/tt_540p_c30.mp4 \
      --reference test-media/liveaction/tt_1080p_ref.mp4 \
      --scale 2 --modes bypass,cnn,metalfx --out results/general/liveaction_2x
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

STILL_FRAMES = [45, 105, 165, 225]
SEQ_START = 100
SEQ_COUNT = 12


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def dump_player_frames(playercheck, source, mode, scale, frames, out_dir):
    """Runs the player once per mode and keeps the frames it renders."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # A dump of frame index N only happens if the run reaches frame N, so the run
    # length is derived from the highest index asked for rather than fixed.
    seconds = int(max(frames) / 30.0) + 6
    cmd = [playercheck, "--file", source, "--mode", mode, "--scale", str(scale),
           "--mute", "--quiet", "--seconds", str(seconds),
           "--dump-at", ",".join(str(f) for f in sorted(frames)),
           "--dump-dir", out_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    got = {}
    for name in sorted(os.listdir(out_dir)):
        m = re.match(r"frame_(\d+)\.png$", name)
        if m:
            got[int(m.group(1))] = os.path.join(out_dir, name)
    if not got:
        sys.stderr.write("no frames dumped for mode %s\n%s\n%s\n"
                         % (mode, r.stdout[-2000:], r.stderr[-2000:]))
        raise SystemExit(1)
    return got


def frames_from_video(video, frames, out_dir):
    """Extracts given frame indices from an already-rendered video.

    Lets a model that cannot run in the player (an offline render produced by
    another tool) be scored by exactly the same code and against the same
    reference as the live modes.
    """
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sel = "+".join("eq(n\\,%d)" % n for n in sorted(frames))
    pattern = os.path.join(out_dir, "frame_%04d.png")
    run(["ffmpeg", "-v", "error", "-y", "-i", video,
         "-vf", "select='%s'" % sel, "-fps_mode", "passthrough", pattern])
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    wanted = sorted(set(frames))
    if len(files) != len(wanted):
        # A shorter-than-asked render would otherwise be scored on fewer frames
        # with no sign anything went wrong (the zip below just truncates).
        raise SystemExit("%s yielded %d of the %d requested frame(s); refusing to score "
                         "an offline row on fewer frames than asked for"
                         % (video, len(files), len(wanted)))
    # Rename to the true frame index: one ffmpeg pass writes them in sequence, but
    # everything downstream (the flicker run, the grids) addresses them by index.
    mapping = {}
    for idx, name in zip(wanted, files):
        target = "frame_%04d.png" % idx
        os.rename(os.path.join(out_dir, name), os.path.join(out_dir, target))
        mapping[idx] = os.path.join(out_dir, target)
    return mapping


def extract_reference_frames(reference, frames, out_dir):
    """Pulls the same frame indices from the high-resolution reference."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sel = "+".join("eq(n\\,%d)" % n for n in sorted(frames))
    pattern = os.path.join(out_dir, "ref_%04d.png")
    run(["ffmpeg", "-v", "error", "-y", "-i", reference,
         "-vf", "select='%s'" % sel, "-fps_mode", "passthrough", pattern])
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    # select + -fps_mode passthrough emits the kept frames in input order, so the
    # k-th output is the k-th requested index.
    return {idx: os.path.join(out_dir, f) for idx, f in zip(sorted(frames), files)}


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def luma(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255.0))


def acutance(img):
    """Mean absolute Laplacian of luma — edge energy, comparable across images."""
    return float(np.mean(np.abs(ndimage.laplace(luma(img)))))


def flat_mask(reference, fraction=0.40):
    """The reference's flattest pixels: where blocking and banding are visible."""
    g = np.hypot(*np.gradient(luma(reference)))
    return g <= np.quantile(g, fraction)


def smooth_rmse(out, ref, mask):
    d = out[mask] - ref[mask]
    return float(np.sqrt(np.mean(d ** 2)))


def flicker_ratio(seq_dir, frames):
    """Mean |consecutive-frame difference|, in the metric the project already
    uses for temporal stability (see results/anime4k/flicker.txt).

    This is an ABSOLUTE number on a 0..255 scale, and it is the honest one. Any
    ratio built from it is only meaningful when the denominator is large; see
    `FLICKER_FLOOR` in main().
    """
    prev, diffs = None, []
    for idx in sorted(frames):
        img = load(os.path.join(seq_dir, "frame_%04d.png" % idx))
        if prev is not None:
            diffs.append(float(np.mean(np.abs(img - prev))))
        prev = img
    return float(np.mean(diffs)) if diffs else float("nan")


# Below this absolute flicker (mean |frame-to-frame difference| in 0..255), a
# ratio against the bypass is arithmetic noise rather than a measurement: the
# bypass is the denominator, and dividing by a number that is essentially zero
# turns a sub-quantisation wobble into a headline. Measured source material
# spans 0.5 (a near-frozen metro shot) to 13.0 (handheld table tennis), so the
# floor is set at one grey level.
FLICKER_FLOOR = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--playercheck", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--modes", default="bypass,cnn,metalfx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--stills", default=",".join(str(f) for f in STILL_FRAMES),
                    help="frame indices for the fidelity metrics")
    ap.add_argument("--seq", default="%d:%d" % (SEQ_START, SEQ_COUNT),
                    help="start:count of consecutive frames for the flicker metric")
    ap.add_argument("--extra-video", action="append", default=[],
                    metavar="NAME=PATH",
                    help="also score an already-rendered video under this name — "
                         "for offline models that cannot run in the player")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    stills = [int(v) for v in args.stills.split(",") if v.strip()]
    seq_start, seq_count = (int(v) for v in args.seq.split(":"))
    seq = list(range(seq_start, seq_start + seq_count))
    wanted = sorted(set(stills) | set(seq))

    os.makedirs(args.out, exist_ok=True)
    refs = extract_reference_frames(args.reference, wanted,
                                    os.path.join(args.out, "reference"))
    # ffmpeg names the kept frames in sequence, so record the index -> file map
    # rather than leaving a reader to reconstruct it from the sorted index list.
    with open(os.path.join(args.out, "reference", "index.json"), "w") as fh:
        json.dump({str(k): os.path.basename(v) for k, v in sorted(refs.items())}, fh, indent=2)
    missing = [i for i in wanted if i not in refs]
    if missing:
        raise SystemExit("reference is missing frame(s) %s" % missing)

    ref_imgs = {i: load(refs[i]) for i in stills}
    # The reference's own sharpness is the yardstick for "as sharp as the truth".
    ref_acut = float(np.mean([acutance(ref_imgs[i]) for i in stills]))
    # One flat mask per still, from the reference, so every mode is scored on the
    # same pixels.
    masks = {i: flat_mask(ref_imgs[i]) for i in stills}
    # The source's own frame-to-frame difference is the floor a model is measured
    # against: matching it adds no temporal instability, and going below it means
    # the model is smoothing the source's motion rather than leaving it alone.
    src_seq_dir = os.path.join(args.out, "source_seq")
    frames_from_video(args.source, seq, src_seq_dir)
    src_flicker = flicker_ratio(src_seq_dir, seq)

    report = {"label": args.label, "scale": args.scale, "source": args.source,
              "reference": args.reference, "reference_acutance": ref_acut,
              "source_flicker": src_flicker,
              "stills": stills, "seq": seq, "modes": {}}

    def score(name, dumps, out_dir):
        """Scores one model's rendered frames. Same code for a live mode and for
        an offline export, so their numbers are comparable."""
        got = [i for i in stills if i in dumps]
        if not got:
            raise SystemExit("model %s produced none of the still frames" % name)
        ps, ss, ac, sr = [], [], [], []
        for i in got:
            out = load(dumps[i])
            ref = ref_imgs[i]
            if out.shape != ref.shape:
                raise SystemExit("model %s frame %d is %s, reference is %s"
                                 % (name, i, out.shape, ref.shape))
            ps.append(psnr(out, ref))
            ss.append(ssim(out, ref))
            ac.append(acutance(out))
            sr.append(smooth_rmse(out, ref, masks[i]))
        entry = {
            "frames": len(got),
            "psnr": float(np.mean(ps)),
            "ssim": float(np.mean(ss)),
            "acutance": float(np.mean(ac)),
            "acutance_ratio": float(np.mean(ac) / ref_acut),
            "smooth_rmse": float(np.mean(sr)),
            "flicker": flicker_ratio(out_dir, [i for i in seq if i in dumps]),
            "source": "player" if out_dir.startswith(args.out) and name in modes
                      else "rendered file",
        }
        report["modes"][name] = entry
        return entry

    for mode in modes:
        score(mode, dump_player_frames(args.playercheck, args.source, mode,
                                       args.scale, wanted,
                                       os.path.join(args.out, mode)),
              os.path.join(args.out, mode))

    for spec in args.extra_video:
        if "=" not in spec:
            raise SystemExit("--extra-video wants NAME=PATH, got %r" % spec)
        name, path = spec.split("=", 1)
        out_dir = os.path.join(args.out, name)
        score(name, frames_from_video(path, wanted, out_dir), out_dir)

    # The bypass (bilinear) run is the temporally neutral yardstick, and it has to
    # be measured before anything can be expressed against it.
    base = report["modes"].get("bypass", {}).get("flicker")
    reliable = base is not None and base >= FLICKER_FLOOR
    print("source's own frame-to-frame difference: %.4f" % src_flicker)
    for name, entry in sorted(report["modes"].items()):
        entry["flicker_ratio"] = (entry["flicker"] / base) if base else None
        entry["flicker_vs_source"] = (entry["flicker"] / src_flicker) if src_flicker else None
        entry["flicker_ratio_reliable"] = bool(reliable)
        print("%-10s psnr=%6.2f dB  ssim=%.4f  acutance=%.2fx  smooth_rmse=%5.2f  "
              "flicker=%7.4f (%.3fx bypass, %.3fx source)"
              % (name, entry["psnr"], entry["ssim"], entry["acutance_ratio"],
                 entry["smooth_rmse"], entry["flicker"],
                 entry["flicker_ratio"] or float("nan"),
                 entry["flicker_vs_source"] or float("nan")))
    if not reliable:
        print("NOTE  bypass flicker is %.4f, under the %.1f floor, so this clip is "
              "too static for the flicker RATIOS to mean anything -- a near-zero "
              "denominator turns sub-quantisation wobble into a headline number. "
              "Read the absolute flicker instead." % (base or float("nan"), FLICKER_FLOOR))

    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print("wrote %s" % os.path.join(args.out, "metrics.json"))


if __name__ == "__main__":
    main()
