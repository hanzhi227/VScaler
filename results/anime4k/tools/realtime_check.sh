#!/bin/bash
# Real-time playback check: play a segment windowed for N seconds with real timing
# (no --untimed), report frame drops and average GPU utilisation.
# Usage: realtime_check.sh <preset> <segment> <seconds>
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/presets.sh"

PRESET="$1"; SEG="$2"; SECS="${3:-15}"
SHADERS="$(preset_shaders "$PRESET")"
SAMPLES="/tmp/an4kbench/gpu_realtime.log"
: > "$SAMPLES"

( while :; do
    v=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | grep -o '[0-9]*$' | sort -n | tail -1)
    [ -n "$v" ] && echo "$v" >> "$SAMPLES"
    sleep 0.5
  done ) >/dev/null 2>&1 &
SPID=$!

args=(mpv --no-config --no-audio --length="$SECS" --vo=gpu-next --gpu-api=vulkan
      --osd-level=0 --deband=no --script="$DIR/stats.lua")
[ -n "$SHADERS" ] && args+=(--glsl-shaders="$SHADERS")

"${args[@]}" "$SEG" 2>&1 | tee /tmp/an4kbench/realtime_out.txt | grep -a "^STATS" | tail -1
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null

python3 - "$PRESET" "$SAMPLES" <<'PY'
import sys
vals=[int(x) for x in open(sys.argv[2]) if x.strip().isdigit()]
if vals:
    print(f"PRESET={sys.argv[1]} gpu_avg={sum(vals)/len(vals):.1f} gpu_p50={sorted(vals)[len(vals)//2]} gpu_max={max(vals)} n={len(vals)}")
else:
    print(f"PRESET={sys.argv[1]} gpu=NA")
PY
