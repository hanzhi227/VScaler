#!/usr/bin/env python3
"""Side-by-side 1:1 crops of one frame from the eval dumps, to look at rather
than to score. Usage:
  scripts/make_eval_grid.py --dir results/general/liveaction_540p_to_1080p \
      --frame 105 --crop 700,380,420,300 --out /tmp/grid.png

`--ref` names the first panel directly, for comparing dumps that came from
`playercheck --dump-at` rather than from an eval run: those have no
`reference/index.json`, and the crop is then in *output* pixels.
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--frame", type=int, default=105)
ap.add_argument("--crop", required=True, help="x,y,w,h in the reference's pixels")
ap.add_argument("--zoom", type=int, default=2)
ap.add_argument("--modes", default="bypass,cnn,metalfx")
ap.add_argument("--cols", type=int, default=0, help="wrap into this many columns")
ap.add_argument("--ref", default=None,
                help="first panel's image, instead of <dir>/reference/index.json")
ap.add_argument("--labels", default=None,
                help="comma-separated panel labels, reference first; defaults to the names")
ap.add_argument("--out", required=True)
a = ap.parse_args()

x, y, w, h = (int(v) for v in a.crop.split(","))
modes = [m for m in a.modes.split(",") if m]

if a.ref:
    ref_path = a.ref
else:
    # The reference frames are written in sequence, so the index -> file map comes
    # from what the eval run recorded.
    with open(os.path.join(a.dir, "reference", "index.json")) as fh:
        index = json.load(fh)
    ref_path = os.path.join(a.dir, "reference", index[str(a.frame)])

panels = [("reference", ref_path)]
for m in modes:
    panels.append((m, os.path.join(a.dir, m, "frame_%04d.png" % a.frame)))

if a.labels:
    names = [s for s in a.labels.split(",")]
    if len(names) != len(panels):
        ap.error("--labels has %d entries for %d panels" % (len(names), len(panels)))
    panels = [(names[i], p[1]) for i, p in enumerate(panels)]

tiles = []
for name, path in panels:
    img = Image.open(path).convert("RGB").crop((x, y, x + w, y + h))
    img = img.resize((w * a.zoom, h * a.zoom), Image.NEAREST)
    tiles.append((name, img))

label_h = 22
cols = a.cols if a.cols > 0 else len(tiles)
rows = (len(tiles) + cols - 1) // cols
tw, th = tiles[0][1].width, tiles[0][1].height
W = cols * tw + 8 * (cols - 1)
H = rows * (th + label_h) + 8 * (rows - 1)
grid = Image.new("RGB", (W, H), (24, 24, 24))
d = ImageDraw.Draw(grid)
for i, (name, img) in enumerate(tiles):
    r, c = divmod(i, cols)
    xo = c * (tw + 8)
    yo = r * (th + label_h + 8)
    grid.paste(img, (xo, yo + label_h))
    d.text((xo + 6, yo + 6), name, fill=(255, 255, 255))
grid.save(a.out)
print(a.out, grid.size)
