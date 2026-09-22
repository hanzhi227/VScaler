#!/usr/bin/env bash
# Regenerate results/general/* : score every upscaler against the same
# high-resolution references.
#
# The live modes (bypass, cnn, cnnDenoise, metalfx, realesrgan,
# realesrgan-general) are measured by driving `playercheck`, so what is scored is
# what the app actually renders. The offline models are rendered to a lossless
# file first and then scored by exactly the same code and against the same
# reference frames, which is what makes their numbers comparable to the live ones.
#
# The two Real-ESRGAN modes appear in the 4x runs only: they are 4x networks with
# no 2x checkpoint, so at 2x the player refuses them, and forcing one would
# produce 2160p against a 1080p reference. Scoring it at 2x would be scoring a
# geometry mismatch rather than a model.
#
#   scripts/eval_all_general.sh [WORKDIR]
#
# Renders land in WORKDIR (default /tmp/general-eval) and are large: the
# lossless 4K files are hundreds of MB each.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORK="/tmp/general-eval"
if [ $# -gt 0 ]; then WORK="$1"; fi
mkdir -p "$WORK"

PC="$ROOT/engine/MacVideoUpscaler/.build/release/playercheck"
[ -x "$PC" ] || { echo "build playercheck first: swift build -c release --product playercheck" >&2; exit 1; }

EVAL="$ROOT/scripts/eval_general_upscaler.py"
RENDER="$ROOT/scripts/rt4ksr/render.py"
LA="$ROOT/test-media/liveaction"
AN="$ROOT/test-media/general"
# The 1080p -> 4K material is a generated intermediate, not a checked-in fixture.
METRO_SRC="${METRO_SRC:-/tmp/metro_1080p_c30.mp4}"
METRO_REF="${METRO_REF:-/tmp/metro_2160p_ref.mp4}"

score() {  # score <outdir> <label> <source> <reference> <scale> <modes> [extra...]
  local out="$1" label="$2" src="$3" ref="$4" scale="$5" modes="$6"; shift 6
  echo "=== $label"
  python3 "$EVAL" --playercheck "$PC" --source "$src" --reference "$ref" \
    --scale "$scale" --modes "$modes" --out "$out" --label "$label" "$@"
  echo
}

# general-x4v3 is rendered offline too, even though it is now a live mode: the
# offline row is the cross-check that the live mode reproduces the checkpoint, and
# it is the only way to score this checkpoint's frame at full precision rather
# than through the player. See scripts/realesrgan/README.md for the verdict.
ESR_DIR="$ROOT/scripts/realesrgan"
if [ -f "$ESR_DIR/render.py" ]; then
  echo "--- rendering Real-ESRGAN general-x4v3 offline (cross-check for the live mode)"
  python3 "$ESR_DIR/render.py" --input "$AN/naruto_270p_c28.mp4" \
      --output "$WORK/general_anime_270p.mkv" --variant general-x4v3 --quiet
  python3 "$ESR_DIR/render.py" --input "$LA/tt_270p_c28.mp4" \
      --output "$WORK/general_liveaction_270p.mkv" --variant general-x4v3 --quiet
fi

echo "--- rendering offline models into $WORK"
python3 "$RENDER" --input "$LA/tt_540p_c30.mp4"      --output "$WORK/rt4ksr_liveaction_540p.mkv" --scale 2 --quiet
python3 "$RENDER" --input "$LA/tt_270p_c28.mp4"      --output "$WORK/rt4ksr_liveaction_270p.mkv" --scale 4 --quiet
python3 "$RENDER" --input "$AN/naruto_540p_c30.mp4"  --output "$WORK/rt4ksr_anime_540p.mkv"      --scale 2 --quiet
python3 "$RENDER" --input "$AN/naruto_270p_c28.mp4"  --output "$WORK/rt4ksr_anime_270p.mkv"      --scale 4 --quiet

EXTRA_270=()
EXTRA_AN270=()

score results/general/liveaction_540p_to_1080p "live action 960x540 -> 1920x1080" \
  "$LA/tt_540p_c30.mp4" "$LA/tt_1080p_ref.mp4" 2 bypass,cnn,cnnDenoise,metalfx \
  --extra-video "rt4ksr=$WORK/rt4ksr_liveaction_540p.mkv"

score results/general/liveaction_270p_to_1080p_4x "live action 480x270 -> 1920x1080 (4x)" \
  "$LA/tt_270p_c28.mp4" "$LA/tt_1080p_ref.mp4" 4 bypass,cnn,metalfx,realesrgan,realesrgan-general \
  --extra-video "rt4ksr=$WORK/rt4ksr_liveaction_270p.mkv" \
  --extra-video "general-x4v3=$WORK/general_liveaction_270p.mkv" "${EXTRA_270[@]}"

score results/general/anime_540p_to_1080p "anime 960x540 -> 1920x1080" \
  "$AN/naruto_540p_c30.mp4" "$AN/naruto_1080p_ref.mp4" 2 bypass,cnn,cnnDenoise,metalfx \
  --extra-video "rt4ksr=$WORK/rt4ksr_anime_540p.mkv"

score results/general/anime_270p_to_1080p_4x "anime 480x270 -> 1920x1080 (4x)" \
  "$AN/naruto_270p_c28.mp4" "$AN/naruto_1080p_ref.mp4" 4 bypass,cnn,metalfx,realesrgan,realesrgan-general \
  --extra-video "rt4ksr=$WORK/rt4ksr_anime_270p.mkv" \
  --extra-video "general-x4v3=$WORK/general_anime_270p.mkv" "${EXTRA_AN270[@]}"

if [ -f "$METRO_SRC" ] && [ -f "$METRO_REF" ]; then
  # Only the first 170 frames are needed: stills 45/105/165 and seq 60..67.
  python3 "$RENDER" --input "$METRO_SRC" --output "$WORK/rt4ksr_metro_1080p.mkv" \
    --scale 2 --frames 0:170 --quiet
  score results/general/liveaction_sharp_1080p_to_2160p "live action SHARP 1920x1080 -> 3840x2160" \
    "$METRO_SRC" "$METRO_REF" 2 bypass,cnn,cnnDenoise,metalfx \
    --stills 45,105,165 --seq 60:8 \
    --extra-video "rt4ksr=$WORK/rt4ksr_metro_1080p.mkv"
else
  echo "skipping the 1080p -> 4K setup: $METRO_SRC not present"
fi

echo "done; see results/general/*/metrics.json"
