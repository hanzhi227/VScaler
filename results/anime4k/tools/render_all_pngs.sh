#!/bin/bash
# Render every test frame through every preset, verifying exact 2x output dims.
# Usage: render_all_pngs.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DIR="$ROOT/results/anime4k/tools"
FRAMES="$ROOT/test-media/frames"
OUTROOT="$ROOT/results/anime4k"

FRAME_LIST="naruto_t300_src480 naruto_t300_src1080 naruto_t450_src480 naruto_t450_src1080 naruto_t600_src480 naruto_t600_src1080 naruto_t900_src480 naruto_t900_src1080 naruto_t590_compressed bbb_t5_src1080 synth_t5_src1080"
PRESETS="off cnnM cnnDenoiseM modeA modeB modeC"

echo "preset,frame,in_w,in_h,out_w,out_h,ok,render_s" > "$OUTROOT/render_dims.csv"
for p in $PRESETS; do
    mkdir -p "$OUTROOT/$p"
    for f in $FRAME_LIST; do
        in="$FRAMES/$f.png"
        out="$OUTROOT/$p/$f.png"
        t0=$(python3 -c 'import time;print(time.time())')
        bash "$DIR/render_png.sh" "$p" "$in" "$out"
        t1=$(python3 -c 'import time;print(time.time())')
        python3 - "$p" "$f" "$in" "$out" "$t0" "$t1" >> "$OUTROOT/render_dims.csv" <<'PY'
import sys
from PIL import Image
p,f,fin,fout,t0,t1=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],float(sys.argv[5]),float(sys.argv[6])
i=Image.open(fin); o=Image.open(fout)
ok = (o.size[0]==2*i.size[0] and o.size[1]==2*i.size[1])
print(f"{p},{f},{i.size[0]},{i.size[1]},{o.size[0]},{o.size[1]},{int(ok)},{t1-t0:.2f}")
PY
    done
done
echo "DONE"
