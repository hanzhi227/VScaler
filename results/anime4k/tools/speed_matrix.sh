#!/bin/bash
# Full speed matrix: segments x presets, with GPU utilisation sampling.
# bbb/synth are whole files (10 s); naruto segments are ~28 s (all frames).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DIR="$ROOT/results/anime4k/tools"
SEG="$ROOT/test-media/segments"
OUT="$ROOT/results/anime4k/speed.csv"

SEGMENTS="naruto_t290.mkv naruto_t580.mkv naruto_t880.mkv bbb_1080p.mp4 synth_clean_1080p.mp4"
PRESETS="off cnnM cnnDenoiseM modeA modeB modeC"

echo "preset,segment,frames,wall_s,fps,gpu_avg,gpu_p50,gpu_max,gpu_n,rc" > "$OUT"
for s in $SEGMENTS; do
    for p in $PRESETS; do
        bash "$DIR/speed_run.sh" "$p" "$SEG/$s" >> "$OUT"
    done
done
echo "SPEED_DONE"
