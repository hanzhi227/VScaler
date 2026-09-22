#!/bin/bash
# Render the 12 consecutive seq_action frames through off/cnnM/modeA for the flicker
# metric. Output: results/anime4k/<preset>/seq_fNN.png (2x).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DIR="$ROOT/results/anime4k/tools"
SRC="$ROOT/test-media/frames/seq_action"
OUTROOT="$ROOT/results/anime4k"
PRESETS="${1:-off cnnM modeA}"

for p in $PRESETS; do
    mkdir -p "$OUTROOT/$p"
    for i in $(seq -w 1 12); do
        bash "$DIR/render_png.sh" "$p" "$SRC/f$i.png" "$OUTROOT/$p/seq_f$i.png" || echo "FAIL $p f$i"
    done
done
echo "SEQ_DONE"
