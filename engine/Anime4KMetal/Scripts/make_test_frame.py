#!/usr/bin/env python3
"""Generate asymmetric synthetic anime-style test frames.

Designed so that porting bugs are visible:
  * vertical luminance gradient (top bright -> bottom dark) => detects v-flips
  * an up-pointing triangle at the top and a filled square at the bottom =>
    unambiguous orientation markers
  * flat cel fill inside black outlines => the CNN must reproduce flat colour
    exactly (weight/bias/order correctness)
  * 1px diagonal hatching and text => sharpness comparison vs bilinear

Outputs (results/metal/assets/):
  test_480p.png  (854x480)
  test_1080p.png (1920x1080)  -- native, same scene scaled 2.25x
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ASSETS = os.path.join(ROOT, "results", "metal", "assets")


def render(W, H):
    s = H / 480.0

    def S(v):
        return int(round(v * s))

    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        t = y / (H - 1)
        base = (int(235 - 175 * t), int(235 - 175 * t), int(245 - 180 * t))
        for x in range(W):
            px[x, y] = (min(255, base[0] + (x * 10) // W),
                        base[1],
                        min(255, base[2] + (x * 6) // W))

    d = ImageDraw.Draw(img)
    lw = max(1, S(3))

    # flat cels with black outlines
    d.ellipse([S(60), S(90), S(300), S(330)], fill=(238, 196, 120), outline=(0, 0, 0), width=lw)
    d.rectangle([S(330), S(100), S(520), S(260)], fill=(120, 190, 235), outline=(0, 0, 0), width=lw)
    d.polygon([(S(560), S(90)), (S(660), S(260)), (S(460), S(260))],
              fill=(215, 120, 140), outline=(0, 0, 0))
    d.rectangle([S(80), S(360), S(300), S(440)], fill=(200, 200, 200),
                outline=(0, 0, 0), width=max(1, S(2)))  # flat calibration patch

    # orientation markers
    d.polygon([(W // 2 - S(40), S(60)), (W // 2 + S(40), S(60)), (W // 2, S(20))],
              fill=(0, 0, 0))
    d.rectangle([W // 2 - S(30), H - S(45), W // 2 + S(30), H - S(15)],
                fill=(255, 255, 255), outline=(0, 0, 0), width=max(1, S(2)))

    # fine 1px hatching (sharpness target)
    for i in range(0, S(220), max(2, S(6))):
        d.line([(S(600) + i, S(300)), (S(600) + i + S(60), S(360))], fill=(0, 0, 0), width=1)
    for i in range(0, S(200), max(2, S(6))):
        d.line([(S(330) + i, S(280)), (S(330) + i + S(60), S(340))], fill=(255, 255, 255), width=1)

    # text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", S(46))
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", S(22))
    except Exception:
        font = ImageFont.load_default()
        small = font
    d.text((S(72), S(20)), "ANIME4K", fill=(0, 0, 0), font=font)
    d.text((S(340), S(380)), "Metal x2 CNN", fill=(10, 10, 10), font=small)
    d.text((S(700), S(20)), "M", fill=(0, 0, 0), font=font)
    return img


def main():
    os.makedirs(ASSETS, exist_ok=True)
    for name, w, h in (("test_480p.png", 854, 480), ("test_1080p.png", 1920, 1080)):
        p = os.path.join(ASSETS, name)
        render(w, h).save(p)
        print(p, (w, h))


if __name__ == "__main__":
    main()
