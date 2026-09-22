#!/bin/bash
# Benchmark playback throughput of one segment through one shader preset.
#
# NOTE on methodology: the task suggested `--vo=image`, but vo=image silently ignores
# --glsl-shaders (verified byte-identical output), so shader cost would be zero. We use
# the real GPU renderer (vo=gpu-next + Vulkan) with --untimed (decode/render as fast as
# possible) and --frames=<N>; fps = N / walltime. GPU utilisation is sampled at ~2 Hz
# from ioreg (no sudo needed).
#
# Usage: speed_run.sh <preset> <segment-file> [max-frames]
# Emits one CSV line: preset,segment,frames,wall_s,fps,gpu_avg,gpu_max
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/presets.sh"

PRESET="$1"; SEG="$2"; CAP="${3:-0}"
BASE="$(basename "$SEG")"

TOTAL=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$SEG")
N="$TOTAL"
if [ "$CAP" -gt 0 ] && [ "$TOTAL" -gt "$CAP" ]; then N="$CAP"; fi

SHADERS="$(preset_shaders "$PRESET")"
# --window-scale=2 makes the render target 2x the source, which is required for
# Anime4K shaders to activate at all (final pass: //!WHEN OUTPUT.w MAIN.w / 1.200 >).
args=(mpv --no-config --no-audio --untimed --frames="$N" --vo=gpu-next --gpu-api=vulkan
      --window-scale=2 --osd-level=0 --deband=no)
[ -n "$SHADERS" ] && args+=(--glsl-shaders="$SHADERS")

SAMPLES="/tmp/an4kbench/gpu_${PRESET}_${BASE}.log"
: > "$SAMPLES"
( while :; do
    v=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | grep -o '[0-9]*$' | sort -n | tail -1)
    [ -n "$v" ] && echo "$v" >> "$SAMPLES"
    sleep 0.5
  done ) >/dev/null 2>&1 &
SPID=$!

t0=$(python3 -c 'import time;print(time.time())')
"${args[@]}" "$SEG" >/dev/null 2>&1
rc=$?
t1=$(python3 -c 'import time;print(time.time())')
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null

python3 - "$PRESET" "$BASE" "$N" "$t0" "$t1" "$SAMPLES" "$rc" <<'PY'
import sys
preset, base, n, t0, t1, sf, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], int(sys.argv[7])
wall = t1 - t0
vals = []
try:
    with open(sf) as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                vals.append(int(line))
except FileNotFoundError:
    pass
vals_s = sorted(vals)
avg = round(sum(vals)/len(vals), 1) if vals else -1
p50 = vals_s[len(vals_s)//2] if vals else -1
mx = max(vals) if vals else -1
fps = round(n/wall, 2) if wall > 0 else -1
print(f"{preset},{base},{n},{wall:.3f},{fps},{avg},{p50},{mx},{len(vals)},{rc}")
PY
