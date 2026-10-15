#!/usr/bin/env bash
# Put the two Real-ESRGAN checkpoints next to each other on real footage.
#
# The published scores say `general-x4v3` loses to `animevideov3` on every column
# measured here, but every scored clip except one is a *synthetic* lanczos
# downscale of a known reference, and on the one genuinely degraded clip the
# ranking of these models is not what the numbers suggest. So the honest way to
# answer "which of these two do I want" is to look, on your own material.
#
# This dumps one frame through both live modes (plus bilinear, plus the source)
# and composes 1:1 crops. It changes nothing in the repo except the grids it
# writes, and it never touches the app.
#
#   scripts/realesrgan/pair_grid.sh [--file CLIP] [--frame N] [--out DIR]
#
# Defaults: test-media/yt/mhw_ascendance_480p.mp4, frame 45, into that clip's
# results directory. The clip is third-party promotional material kept for local
# testing only (`test-media/yt/README.md`), so treat the grids it produces the
# same way and do not redistribute them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FILE="test-media/yt/mhw_ascendance_480p.mp4"
FRAME=45
OUT="results/general/yt_mhw_480p"
while [ $# -gt 0 ]; do
  case "$1" in
    --file)  FILE="$2"; shift 2 ;;
    --frame) FRAME="$2"; shift 2 ;;
    --out)   OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -f "$FILE" ] || { echo "no such clip: $FILE" >&2; exit 1; }

PC="$ROOT/engine/MacVideoUpscaler/.build/release/playercheck"
if [ ! -x "$PC" ]; then
  echo "building playercheck..."
  swift build -c release --package-path "$ROOT/engine/MacVideoUpscaler" --product playercheck >/dev/null
fi

FFMPEG=/opt/homebrew/bin/ffmpeg
FFPROBE=/opt/homebrew/bin/ffprobe
WORK="${TMPDIR:-/tmp}/esrgan-pair-grid"
rm -rf "$WORK"; mkdir -p "$WORK"/{src,bilinear,animevideov3,general-x4v3}

# Two lines of plain numbers rather than CSV: this ffmpeg build rejects the
# `-of csv=p=0:s=' '` option string. A while-read loop instead of readarray,
# which does not exist in the stock macOS bash 3.2.
W=""; H=""
while IFS= read -r dim; do
  if [ -z "$W" ]; then W="$dim"; else H="$dim"; fi
done < <("$FFPROBE" -v error -select_streams v:0 \
  -show_entries stream=width,height -of default=nw=1:nk=1 "$FILE")
[ -n "$W" ] && [ -n "$H" ] || { echo "could not read the size of $FILE" >&2; exit 1; }
if [ $((W * 4)) -gt 4096 ] || [ $((H * 4)) -gt 2304 ]; then
  echo "$FILE is ${W}x${H}: 4x is $((W * 4))x$((H * 4)), past the encoder's 4096x2304" >&2
  echo "limit, so both models refuse it. Use a clip up to 1024x576." >&2
  exit 1
fi

# Everything below is indexed from source frame 0, so the dump and the extracted
# frame line up. That is checked rather than assumed: bilinear is a real 4x
# upscale of this frame, so it has to beat its neighbours by a wide margin in dB.
echo "extracting source frames 0..$FRAME from $(basename "$FILE") (${W}x${H})"
$FFMPEG -v error -y -i "$FILE" -frames:v $((FRAME + 1)) "$WORK/src/f_%04d.png"

for pair in "bilinear:bypass" "animevideov3:realesrgan" "general-x4v3:realesrgan-general"; do
  dir="${pair%%:*}"; mode="${pair##*:}"
  echo "dumping frame $FRAME through --mode $mode"
  "$PC" --file "$FILE" --mode "$mode" --scale 4 --seconds 3 --no-audio \
      --dump-at "$FRAME" --dump-dir "$WORK/$dir" --quiet >/dev/null 2>&1 || true
  [ -f "$WORK/$dir/frame_$(printf %04d "$FRAME").png" ] \
    || { echo "no frame dumped for $mode" >&2; exit 1; }
done

python3 - "$WORK" "$FRAME" "$W" "$H" <<'PY'
import sys
from PIL import Image
work, frame, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
src = Image.open("%s/src/f_%04d.png" % (work, frame + 1)).convert("RGB")
assert src.size == (w, h), "source frame is %s, expected %s" % (src.size, (w, h))
src.resize((w * 4, h * 4), Image.NEAREST).save("%s/src_nearest4x.png" % work)

# Same alignment check the harness uses elsewhere: bilinear is a real 4x upscale
# of frame N, so it must resemble frame N far more than the frame before it.
# ffmpeg numbered the extracted frames from 1, so source frame N is f_%04d(N+1).
import numpy as np
byp = np.asarray(Image.open("%s/bilinear/frame_%04d.png" % (work, frame))
                 .convert("RGB"), dtype=np.float32)
def db(n):
    other = Image.open("%s/src/f_%04d.png" % (work, n)).convert("RGB") \
        .resize((byp.shape[1], byp.shape[0]), Image.NEAREST)
    mse = ((np.asarray(other, dtype=np.float32) - byp) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
here, prev = db(frame + 1), db(frame)
print("alignment: source frame %d matches the dump at %.2f dB, frame %d at %.2f dB %s"
      % (frame, here, frame - 1, prev, "OK" if here > prev + 6 else "SUSPECT"))
sys.exit(0 if here > prev + 6 else 1)
PY

# Three crops: the blown specular highlight and blue effect, foliage texture, and
# flat text on a card — the three things the two checkpoints disagree about most.
LABELS="source (nearest 4×),bilinear,AnimeVideo-v3,General-x4v3"
declare -a CROPS=(
  "highlight:$((W * 4 * 47 / 100)),$((H * 4 * 61 / 100)),$((W * 4 * 11 / 100)),$((H * 4 * 15 / 100))"
  "foliage:$((W * 4 * 12 / 100)),$((H * 4 * 31 / 100)),$((W * 4 * 11 / 100)),$((H * 4 * 15 / 100))"
  "rating_card:$((W * 4 * 88 / 100)),$((H * 4 * 84 / 100)),$((W * 4 * 11 / 100)),$((H * 4 * 15 / 100))"
)
mkdir -p "$OUT"
for spec in "${CROPS[@]}"; do
  name="${spec%%:*}"; crop="${spec##*:}"
  python3 scripts/make_eval_grid.py --dir "$WORK" --ref "$WORK/src_nearest4x.png" \
    --modes bilinear,animevideov3,general-x4v3 --labels "$LABELS" \
    --frame "$FRAME" --crop "$crop" --zoom 3 \
    --out "$OUT/grid_real_${name}_esrgan_pair.png"
done

echo
echo "wrote 3 grids to $OUT/grid_real_*_esrgan_pair.png"
echo "panels: source / bilinear / AnimeVideo-v3 / General-x4v3, 3x, frame $FRAME"
echo "(the source clip is third-party promotional material kept for local testing"
echo " only -- see test-media/yt/README.md; do not redistribute these grids)"
