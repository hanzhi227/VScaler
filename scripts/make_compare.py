#!/usr/bin/env python3
"""Build side-by-side 2x upscale comparison grids for the report.

Rows:    source (bicubic 2x) | Anime4K CNN x2 M | Anime4K Mode A | AnimeJaNai SUC
Columns: full frame (downscaled) | 1:1 detail crop (shown at 2x pixel scale)

Writes results/compare/<frame>_grid.png
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "test-media/frames"
A4K = ROOT / "results/anime4k"
AJN = ROOT / "results/animejanai/HD_V3_SuperUltraCompact"
OUT = ROOT / "results/compare"

# frame -> (crop centre as fraction of source, crop size in source pixels)
CASES = {
    "naruto_t300_src480":  ((0.52, 0.35), (200, 112)),   # clean line art / faces
    "naruto_t450_src480":  ((0.45, 0.45), (200, 112)),   # dark scene
    "naruto_t600_src480":  ((0.45, 0.40), (200, 112)),   # motion
    "naruto_t900_src480":  ((0.40, 0.55), (200, 112)),   # fine detail / architecture
    "naruto_t590_compressed": ((0.50, 0.40), (200, 112)),  # heavy compression
    "bbb_t5_src1080":      ((0.35, 0.45), (360, 200)),   # clean gradients/fur
    "synth_t5_src1080":    ((0.45, 0.45), (360, 200)),   # synthetic line art
}

FULL_W = 520          # width of the full-frame preview column
CROP_LABEL_H = 22
METHODS = ["Source (bicubic 2x)", "Anime4K CNN x2 M", "Anime4K Mode A", "AnimeJaNai SUC"]


def load_2x(path, src_size):
    """Load a 2x output, resizing to exactly 2x source if the method differs."""
    target = (src_size[0] * 2, src_size[1] * 2)
    im = Image.open(path).convert("RGB")
    if im.size != target:
        im = im.resize(target, Image.LANCZOS)
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(CASES.keys())
    for name in names:
        if name not in CASES:
            print(f"skip unknown frame {name}")
            continue
        src_path = FRAMES / f"{name}.png"
        if not src_path.exists():
            print(f"missing source {src_path}")
            continue
        src = Image.open(src_path).convert("RGB")
        w, h = src.size
        (cfx, cfy), (cw, ch) = CASES[name]
        cw, ch = min(cw, w), min(ch, h)
        # auto-select the highest-detail window so the crop always shows lines/texture
        g = np.asarray(src.convert("L"), dtype=np.float32)
        gy, gx = np.gradient(g)
        energy = np.abs(gx) + np.abs(gy)
        ii = np.cumsum(np.cumsum(energy, 0), 1)
        ii = np.pad(ii, ((1, 0), (1, 0)))
        best, bx, by = -1.0, 0, 0
        for yy in range(0, h - ch + 1, 8):
            for xx in range(0, w - cw + 1, 8):
                ssum = ii[yy + ch, xx + cw] - ii[yy, xx + cw] - ii[yy + ch, xx] + ii[yy, xx]
                if ssum > best:
                    best, bx, by = ssum, xx, yy
        cfx, cfy = (bx + cw / 2) / w, (by + ch / 2) / h

        variants = [("Source (bicubic 2x)", src.resize((w * 2, h * 2), Image.BICUBIC))]
        for label, p in (("Anime4K CNN x2 M", A4K / "cnnM" / f"{name}.png"),
                         ("Anime4K Mode A", A4K / "modeA" / f"{name}.png"),
                         ("AnimeJaNai SUC", AJN / f"{name}.png")):
            if p.exists():
                variants.append((label, load_2x(p, (w, h))))
            else:
                variants.append((label, None))

        # crop rect in source coords -> 2x
        cx = int(cfx * w - cw / 2)
        cy = int(cfy * h - ch / 2)
        cx = max(0, min(cx, w - cw))
        cy = max(0, min(cy, h - ch))
        crop2x = (cx * 2, cy * 2, (cx + cw) * 2, (cy + ch) * 2)
        crop_w, crop_h = crop2x[2] - crop2x[0], crop2x[3] - crop2x[1]
        # display crops at 1:1 of the 2x output (true pixel detail), capped
        disp_w = min(crop_w, 640)
        disp_h = int(crop_h * disp_w / crop_w)

        row_h = max(int(h * 2 * FULL_W / (w * 2)), disp_h)
        canvas = Image.new("RGB", (FULL_W + disp_w + 24, row_h * len(variants) + CROP_LABEL_H),
                           (24, 24, 28))
        d = ImageDraw.Draw(canvas)
        d.text((8, 4), f"{name}  ({w}x{h} → 2x; right column = 1:1 detail "
                       f"crop at {cx},{cy} {cw}x{ch} px)", fill=(240, 240, 240))

        for i, (label, im) in enumerate(variants):
            y = CROP_LABEL_H + i * row_h
            d.rectangle([0, y, canvas.width, y + row_h - 2], fill=(24, 24, 28))
            d.text((8, y + 4), f"{i+1}. {label}", fill=(255, 220, 120))
            if im is None:
                d.text((8, y + 24), "  (render missing)", fill=(255, 120, 120))
                continue
            full = im.resize((FULL_W, int(im.height * FULL_W / im.width)), Image.LANCZOS)
            canvas.paste(full, (8, y + CROP_LABEL_H))
            crop = im.crop(crop2x).resize((disp_w, disp_h), Image.NEAREST)
            canvas.paste(crop, (FULL_W + 16, y + CROP_LABEL_H))
            d.rectangle([FULL_W + 15, y + CROP_LABEL_H - 1, FULL_W + 16 + disp_w, y + CROP_LABEL_H + disp_h],
                        outline=(90, 90, 100))
        out = OUT / f"{name}_grid.png"
        canvas.save(out)
        print(f"wrote {out}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
