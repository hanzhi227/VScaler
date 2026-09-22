#!/bin/bash
# Render one image through mpv + Anime4K shaders and capture the exact 2x result.
#
# Why this is not `--vo=image`: vo=image silently ignores --glsl-shaders (verified:
# output was byte-identical to the source). We therefore use vo=gpu-next (libplacebo)
# and save the rendered window via a Lua screenshot script. --window-scale=2 makes the
# render target exactly 2x the source, which (a) satisfies Anime4K's
# `//!WHEN OUTPUT.w MAIN.w / 1.200 >` activation condition and (b) yields an exact 2x PNG.
#
# Usage: render_png.sh <preset> <input-image> <output.png>
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/presets.sh"

PRESET="$1"; IN="$2"; OUT="$3"
SHADERS="$(preset_shaders "$PRESET")"
mkdir -p "$(dirname "$OUT")"

# Some inputs (naruto *_src1080.png) are anamorphic (SAR 3:4). --window-scale=2 would
# then size the window from the *display* size (4:3) and the capture would not be 2x the
# storage resolution. Force square pixels so the output is exactly 2x the input file dims.
DIMS=$(python3 -c "from PIL import Image;im=Image.open('$IN');print(f'{im.size[0]}:{im.size[1]}')")

args=(mpv --no-config --no-audio --untimed --frames=1 --pause=yes
      --vo=gpu-next --gpu-api=vulkan --window-scale=2
      --video-aspect-override="$DIMS"
      --osd-level=0 --dither=no --deband=no
      --screenshot-format=png --screenshot-png-compression=1
      --script="$DIR/shot.lua" --script-opts="shot-file=$OUT,shot-mode=window,shot-delay=0.35")

if [ -n "$SHADERS" ]; then
    args+=(--glsl-shaders="$SHADERS")
fi

timeout 180 "${args[@]}" "$IN" >/dev/null 2>&1
rc=$?
if [ ! -f "$OUT" ]; then
    echo "FAILED preset=$PRESET in=$IN rc=$rc" >&2
    exit 1
fi
exit 0
