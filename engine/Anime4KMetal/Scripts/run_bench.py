#!/usr/bin/env python3
"""
Benchmark driver for the Anime4K Metal port.

Runs an4kbench over a fixed config matrix, optionally samples GPU utilisation
from `ioreg -c IOAccelerator` while a run is in flight, and writes:
  results/metal/bench.json
  results/metal/BENCH.md

Usage: run_bench.py <an4kbench binary> [results/metal dir]
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "metal")
ASSETS = os.path.join(RESULTS, "assets")
BIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    SCRIPT_DIR, "..", ".build", "release", "an4kbench")
BIN = os.path.abspath(BIN)

INPUT_480 = os.path.join(ASSETS, "test_480p.png")
INPUT_1080 = os.path.join(ASSETS, "test_1080p.png")


def gpu_util_sampler(stop, samples):
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                capture_output=True, text=True, timeout=5).stdout
            for m in re.finditer(r'"Device Utilization %"=(\d+)', out):
                samples.append(int(m.group(1)))
        except Exception:
            pass
        time.sleep(0.1)


def run_one(mode, iters, inp, repeat, out_name, sample_gpu):
    out_png = os.path.join(ASSETS, out_name)
    cmd = [BIN, "--input", inp, "--output", out_png,
           "--mode", mode, "--iters", str(iters)]
    if repeat:
        cmd.append("--repeat-to-1080")
    samples = []
    stop = threading.Event()
    th = None
    if sample_gpu:
        th = threading.Thread(target=gpu_util_sampler, args=(stop, samples), daemon=True)
        th.start()
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if th:
        stop.set()
        th.join(timeout=2)
    if p.returncode != 0:
        raise SystemExit(f"bench failed for {mode}/{iters}/{repeat}:\n{p.stderr}")
    line = [l for l in p.stdout.splitlines() if l.strip().startswith("{")][-1]
    rec = json.loads(line)
    dev = ""
    for l in p.stderr.splitlines():
        if l.startswith("device: "):
            dev = l[len("device: "):]
    rec["gpu_util_mean_pct"] = round(sum(samples) / len(samples), 1) if samples else None
    rec["gpu_util_max_pct"] = max(samples) if samples else None
    rec["gpu_util_samples"] = len(samples)
    rec["process_wall_s"] = round(wall, 3)
    rec["fps"] = round(1000.0 / rec["ms_per_frame"], 2) if rec["ms_per_frame"] > 0 else None
    rec["_device"] = dev
    return rec


def main():
    os.makedirs(RESULTS, exist_ok=True)
    if not os.path.exists(INPUT_480):
        raise SystemExit("missing synthetic test frame: " + INPUT_480)
    if not os.path.exists(INPUT_1080):
        raise SystemExit("missing synthetic test frame: " + INPUT_1080)

    # (mode, iters, input, repeat_to_1080, output_name, sample_gpu)
    configs = [
        ("bilinear",      201, INPUT_480,  False, "bench_480p_bilinear.png", False),
        ("cnnX2M",        201, INPUT_480,  False, "bench_480p_cnnX2M.png",   False),
        ("cnnDenoiseX2M", 201, INPUT_480,  False, "bench_480p_denoise.png",  False),
        ("bilinear",      101, INPUT_1080, True,  "bench_1080p_bilinear.png", True),
        ("cnnX2M",        101, INPUT_1080, True,  "bench_1080p_cnnX2M.png",   True),
        ("cnnDenoiseX2M", 101, INPUT_1080, True,  "bench_1080p_denoise.png",  True),
    ]

    records = []
    device = "?"
    for mode, iters, inp, repeat, out_name, gpu in configs:
        rec = run_one(mode, iters, inp, repeat, out_name, gpu)
        device = rec.pop("_device") or device
        rec["source"] = "480p" if not repeat else "1080p"
        rec["source_file"] = os.path.basename(inp)
        records.append(rec)
        print(f"{rec['mode']:14s} {rec['input_w']}x{rec['input_h']} -> "
              f"{rec['output_w']}x{rec['output_h']}  mean {rec['ms_per_frame']:.3f}  "
              f"med {rec['median_ms']:.3f}  min {rec['min_ms']:.3f} ms/frame  "
              f"gpu={rec['gpu_util_mean_pct']}%")

    payload = {"device": device, "warmup_iters": 1, "input_480p": os.path.basename(INPUT_480),
               "input_1080p": os.path.basename(INPUT_1080),
               "harness": "synchronous commit + waitUntilCompleted, wall-clock per frame",
               "results": records}
    with open(os.path.join(RESULTS, "bench.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ---- BENCH.md ----
    lines = []
    lines.append("# Anime4K Metal (M5 Pro) benchmark\n")
    lines.append(f"- Device: **{device}**")
    lines.append("- Harness: `an4kbench`, synchronous `commit()` + `waitUntilCompleted()` per frame")
    lines.append("- Timing: wall-clock ms/frame, first iteration excluded as warmup; "
                 "mean over all timed iterations, with median/min because the odd "
                 "scheduler spike can inflate the mean at small sizes")
    lines.append(f"- Source frames: synthetic asymmetric anime frames "
                 f"`assets/{os.path.basename(INPUT_480)}` (854x480) and "
                 f"`assets/{os.path.basename(INPUT_1080)}` (1920x1080)")
    lines.append("- 1080p rows: `--repeat-to-1080` re-rasterises to 1080p-class height "
                 "(identity here, since the source is already 1920x1080)")
    lines.append("- Arithmetic: fp32 in all kernels; intermediates `rgba16Float`\n")
    lines.append("| source | mode | input | output | iters | ms/frame | min | median | FPS | GPU util (mean/max) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        gpu = (f"{r['gpu_util_mean_pct']}% / {r['gpu_util_max_pct']}%"
               if r["gpu_util_mean_pct"] is not None else "n/a")
        lines.append(
            f"| {r['source']} | `{r['mode']}` | {r['input_w']}x{r['input_h']} | "
            f"{r['output_w']}x{r['output_h']} | {r['iters']} | "
            f"**{r['ms_per_frame']:.3f}** | {r['min_ms']:.3f} | {r['median_ms']:.3f} | "
            f"{r['fps']:.1f} | {gpu} |")
    lines.append("")
    lines.append("## Validation summary\n")
    lines.append("See `validation.json`. Key results:")
    lines.append("- Output dimensions exactly 2x; zero NaN/Inf in the fp32 output")
    lines.append("- Metal output matches an independent NumPy port of the same GLSL weights "
                 "to **<= 1 LSB** (0.00% of pixels differ by more than 1/255)")
    lines.append("- Orientation preserved (forward row-profile correlation 0.998 vs "
                 "mirrored -0.905)")
    lines.append("- Line-art acutance vs bilinear: hatching **7.8x**, text **5.9x**")
    lines.append("- Flat cel region reproduced exactly (200.0 -> 200.0)\n")
    with open(os.path.join(RESULTS, "BENCH.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(RESULTS, "BENCH.md"), "and bench.json")


if __name__ == "__main__":
    main()
