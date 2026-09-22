#!/usr/bin/env bash
# What SeedVR2 actually costs on this machine through the one runtime that runs
# it here: MLX, via mlx-gen's 4-bit build.
#
#   MLX_PY=/path/to/venv/bin scripts/seedvr2/mlx_probe.sh [OUTFILE]
#
# Defaults to results/seedvr2/mlx.txt. `bench.py` measures the DiT's shape on
# the GPU with PyTorch/MPS; this measures the real end-to-end pipeline with real
# weights in 4-bit. They answer different questions and both are kept:
#
#   bench.py    is the DiT alone, random weights, batched across frames, no VAE.
#               It is a FLOOR -- and the numbers here show how optimistic: the
#               4-bit MLX pipeline costs ~9x more per frame than that floor,
#               because of dequantisation, the VAE (250.7 M params at pixel
#               resolution), tone correction and I/O.
#   this script is what a user would actually experience.
#
# mlx-gen is NOT a project dependency and is not imported by anything in
# `engine/`. It is installed in a throwaway venv (see README.md); this script
# only shells out to its CLI. It downloads ~2.6 GB of weights on first run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-$ROOT/results/seedvr2/mlx.txt}"
mkdir -p "$(dirname "$OUT")"
MODEL="${SEEDVR2_MLX_MODEL:-AbstractFramework/seedvr2-3b-4bit}"
CLIP="$ROOT/test-media/general/naruto_270p_c28.mp4"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ -z "${MLX_PY:-}" ]; then
  echo "set MLX_PY to a venv with mlx-gen installed, e.g." >&2
  echo "  python3 -m venv /tmp/mlxenv && /tmp/mlxenv/bin/pip install -U mlx-gen" >&2
  echo "  MLX_PY=/tmp/mlxenv/bin $0" >&2
  exit 2
fi
UP="$MLX_PY/mflux-upscale-seedvr2"
[ -x "$UP" ] || { echo "no mflux-upscale-seedvr2 in $MLX_PY" >&2; exit 2; }

FFMPEG=/opt/homebrew/bin/ffmpeg
"$FFMPEG" -v error -y -i "$CLIP" -vf "select=eq(n\,105)" -frames:v 1 "$TMP/frame.png"
"$FFMPEG" -v error -y -i "$TMP/frame.png" -vf "crop=128:128:0:0" "$TMP/tiny.png"

# wall time and peak RSS for one invocation; prints "wall peakRSS_GB"
run() {
  local label="$1"; shift
  /usr/bin/time -l "$@" >"$TMP/run.log" 2>&1
  local rc=$?
  local wall peak
  wall=$(grep -E "^ +[0-9.]+ real" "$TMP/run.log" | head -1 | awk '{print $1}')
  peak=$(grep "maximum resident set size" "$TMP/run.log" | awk '{print $1}')
  printf '%-42s exit=%-3s wall=%7ss  peakRSS=%5.2f GB\n' "$label" "$rc" "${wall:-?}" \
    "$(echo "${peak:-0}" | awk '{print $1/1e9}')"
  grep -E "Peak MLX|Physical footprint|budget|safe video mode|streaming plan" "$TMP/run.log" \
    | sed 's/^/      /' | head -4
}

{
  echo "SeedVR2 through mlx-gen (MLX, 4-bit) on $(sysctl -n machdep.cpu.brand_string)"
  echo "macOS $(sw_vers -productVersion), $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1/1e9}') unified memory"
  echo "measured $(date '+%Y-%m-%d %H:%M')"
  echo "model: $MODEL"
  "$MLX_PY/python" -c "import mlx.core as mx; print('mlx', mx.__version__)" 2>/dev/null
  "$MLX_PY/pip" show mlx-gen 2>/dev/null | grep -E "^(Name|Version)" | tr '\n' ' '; echo
  echo
  echo "Peak RSS is the process's resident set (the number to hold against the"
  echo "project's 2 GB model budget). 'Peak MLX' is mlx-gen's own accounting of"
  echo "MLX-managed memory, which is larger and which it also budget-checks."
  echo

  echo "=== image path: 128x128 crop, 2x (fixed load overhead) ==="
  run "128x128 -> 256x256, 2x" "$UP" --model "$MODEL" \
      --image-path "$TMP/tiny.png" --resolution 2x --no-progress --output "$TMP/o1.png"
  echo
  echo "=== image path: one real SD frame, 2x ==="
  run "480x270 -> 960x540, 2x" "$UP" --model "$MODEL" \
      --image-path "$TMP/frame.png" --resolution 2x --no-progress --output "$TMP/o2.png"
  echo "  (the difference between those two rows is the marginal cost of a real"
  echo "   frame: the 128x128 image mostly measures model load.)"
  echo

  echo "=== video path, 3 frames of the same SD clip ==="
  run "video 1x (restoration, source size)" "$UP" --model "$MODEL" \
      --video-path "$CLIP" --max-frames 3 --resolution 1x --drop-audio --no-progress \
      --no-validate-health --output "$TMP/v1.mp4"
  run "video 1x with a 1 GB MLX cache" "$UP" --model "$MODEL" \
      --video-path "$CLIP" --max-frames 3 --resolution 1x --drop-audio --no-progress \
      --no-validate-health --low-ram --mlx-cache-limit-gb 1 --output "$TMP/v2.mp4"
  run "video 2x (upscaling)" "$UP" --model "$MODEL" \
      --video-path "$CLIP" --max-frames 3 --resolution 2x --drop-audio --no-progress \
      --output "$TMP/v3.mp4"
} 2>&1 | tee "$OUT"

echo
echo "wrote $OUT"
