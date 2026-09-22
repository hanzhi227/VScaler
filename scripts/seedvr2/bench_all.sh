#!/usr/bin/env bash
# Every SeedVR2 figure quoted in this directory, regenerated in one go.
#
#   scripts/seedvr2/bench_all.sh [OUTFILE]
#
# Defaults to results/seedvr2/bench.txt. Each run builds the transformer at the
# shipped dimensions (not a projection: see bench.py) and times real forward
# passes on the MPS GPU.
#
# Two things are being measured, and the second is not an afterthought:
#
#   throughput   ms/frame for a clip that fits, across output resolutions
#   memory wall  which configurations this machine cannot run at all. SeedVR2 is
#                a datacenter model -- upstream's README quotes "1 H100-80G can
#                handle videos with 100x720x1280" -- and this machine has 26 GB
#                of unified memory shared with the OS, so the honest sweep has to
#                include the cases that do not fit. bench.py caps MPS allocation
#                (--mem-fraction) so an oversized case fails cleanly instead of
#                swapping the whole system.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-$ROOT/results/seedvr2/bench.txt}"
mkdir -p "$(dirname "$OUT")"
BENCH="python3 scripts/seedvr2/bench.py"

# Each run allocates several GB of GPU memory, and a `bench.py` whose shell is
# killed carries on holding it: killing the wrapper does not kill the python
# child, which then competes with the next run for the GPU. Two of these
# overlapping was measured at ~20 GB of unified memory on a 26 GB machine. So
# the children are tracked and killed explicitly, on the way out of any exit
# path including a signal.
CHILDREN=""
cleanup() {
  for pid in $CHILDREN; do
    kill -TERM "$pid" 2>/dev/null
  done
  sleep 1
  for pid in $CHILDREN; do
    kill -KILL "$pid" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

run() {  # run <variant> <res> <frames> <iters> [extra...]
  local variant="$1" res="$2" frames="$3" iters="$4"; shift 4
  local log
  log="$(mktemp)"
  echo "=== $variant  output $res  $frames frames per pass ==="
  # Not `$BENCH ... | grep`: in a pipeline `$!` is the *last* element's pid, so
  # the tracked pid would be grep's and the python child would survive the kill.
  $BENCH --variant "$variant" --res "$res" --frames "$frames" --iters "$iters" "$@" \
      >"$log" 2>&1 &
  local pid=$!
  CHILDREN="$CHILDREN $pid"
  wait "$pid"
  CHILDREN=""
  grep -vE "UserWarning|return torch\._C" "$log"
  rm -f "$log"
  echo
}

{
  echo "SeedVR2 throughput on $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
  echo "macOS $(sw_vers -productVersion), $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1/1e9}') unified memory, $(sysctl -n hw.ncpu) CPUs"
  echo "measured $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "Method: scripts/seedvr2/bench.py builds the NaDiT block stack at the"
  echo "dimensions in the shipped Comfy-Org safetensors headers and times real"
  echo "forward passes through it with random weights (a forward pass does not"
  echo "depend on weight values). This is a FLOOR on the real thing: the 3D causal"
  echo "VAE's encode and decode are not included."
  echo
  echo "MPS allocation is capped at 85% of the GPU working set (macOS reports"
  echo "~17.7 GB of it on this 26 GB machine) so an oversized case fails in seconds"
  echo "instead of swapping the whole system."
  echo

  echo "=== reference: what this GPU does on plain dense matmul ==="
  $BENCH --res 640x480 --frames 4 --iters 1 --matmul 2>/dev/null | grep "dense matmul"
  echo

  echo "############ 3B: scaling with clip length, at a fixed 640x480 output ############"
  echo
  for frames in 1 4 9 25 49; do run 3b 640x480 "$frames" 2; done

  echo "############ 3B: scaling with output resolution, 9 frames ############"
  echo
  for res in 640x480 1280x960 1920x1080 3840x2160; do run 3b "$res" 9 2; done

  echo "############ 3B: upstream's own example shape, 100x720x1280 ############"
  echo
  run 3b 1280x720 100 1

  echo "############ 7B ############"
  echo
  for spec in "640x480:4" "640x480:9" "1280x960:4"; do
    res="${spec%%:*}"; frames="${spec##*:}"
    run 7b "$res" "$frames" 1
  done

  echo "############ the other bracket: GLOBAL attention instead of the (4,3,3) window ############"
  echo "Global attention is quadratic in the token count, which is why upstream"
  echo "windows every layer. At small token counts it is actually cheaper here --"
  echo "the window fold has real padding and reshape overhead -- and at the sizes"
  echo "that matter it does not fit at all. Both are shown."
  echo
  run 3b 640x480  9 1 --attention full
  run 3b 1280x960 9 1 --attention full
} 2>&1 | tee "$OUT"

echo
echo "wrote $OUT"
