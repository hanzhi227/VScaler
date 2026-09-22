#!/usr/bin/env python3
"""Generate a synthetic cel-shaded anime-style clip: 1920x1080 @ 24fps, ~10s.

Scene: sunset gradient sky, flat-color hills with darker cel shading, a hoodie
character silhouette, fine outlines, grass tufts, telephone pole + wires.
Camera pans right; character bobs. Designed to stress line integrity,
flat-region cleanliness, and gradients for upscaler comparison.
"""
import math
import subprocess
from PIL import Image, ImageDraw

W, H, SW = 1920, 1080, 2600  # output size, scene width (pan range = SW - W)
FPS, SECS = 24, 10
FRAME_DIR = "/tmp/synth"

# flat palette (cel style)
SKY_TOP, SKY_BOT = (250, 190, 120), (255, 230, 170)
HILL_BACK, HILL_BACK_SH = (96, 156, 108), (74, 130, 88)
HILL_FRONT, HILL_FRONT_SH = (58, 128, 78), (44, 102, 62)
LINE = (24, 20, 26)
CHAR_BODY, CHAR_HOOD = (236, 120, 70), (250, 150, 90)
CHAR_SH = (200, 88, 52)
POLE = (60, 46, 40)

def hill_y(x, base, amp, phase):
    return base - int(amp * math.sin(x * 0.004 + phase) + amp * 0.4 * math.sin(x * 0.011 + phase * 2))

def draw_frame(t):
    off = int((SW - W) * t / (FPS * SECS))  # pan offset
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    # sky gradient
    for y in range(H):
        f = y / H
        c = tuple(int(SKY_TOP[i] + (SKY_BOT[i] - SKY_TOP[i]) * f) for i in range(3))
        d.line([(0, y), (W, y)], fill=c)
    # sun + halo
    d.ellipse([W - 480, 110, W - 300, 290], fill=(255, 244, 214), outline=(255, 200, 140), width=4)
    # speed-line clouds (flat white, outlined)
    for cx, cy, cw in [(180, 140, 260), (700, 90, 330), (1250, 200, 220), (2100, 120, 300)]:
        x = cx - off
        if -cw < x < W + 50:
            d.ellipse([x, cy, x + cw, cy + 60], fill=(255, 252, 244), outline=LINE, width=3)
    # back hills
    pts = [(x - off, hill_y(x, 560, 90, 1.2)) for x in range(off - 20, off + W + 260, 8)]
    d.polygon(pts + [(W, H), (0, H)], fill=HILL_BACK)
    # front hills
    pts = [(x - off, hill_y(x, 760, 120, 4.0)) for x in range(off - 20, off + W + 260, 8)]
    d.polygon(pts + [(W, H), (0, H)], fill=HILL_FRONT)
    d.line(pts, fill=LINE, width=4)
    # grass tufts along front crest (fine lines)
    for x in range(off + 30, off + W + 200, 46):
        hx, hy = x - off, hill_y(x, 760, 120, 4.0)
        for k in (-1, 0, 1):
            d.line([(hx + k * 5, hy + 3), (hx + k * 11, hy - 16 - 4 * abs(k))], fill=LINE, width=2)
    # telephone pole + wires
    px = 1500 - off
    if -80 < px < W + 80:
        py = hill_y(1500, 760, 120, 4.0)
        d.rectangle([px - 7, py - 560, px + 7, py], fill=POLE, outline=LINE, width=3)
        d.rectangle([px - 70, py - 540, px + 70, py - 524], fill=POLE, outline=LINE, width=3)
        for wy in (py - 530, py - 520):
            d.arc([px - 400, wy - 14, px + 400, wy + 40], 180, 360, fill=LINE, width=2)
    # character (bobbing)
    bx = 760 - off
    bob = int(5 * math.sin(t * 2 * math.pi / FPS))
    if -200 < bx < W + 200:
        base_y = hill_y(760, 760, 120, 4.0) + bob
        d.polygon([(bx - 95, base_y), (bx + 95, base_y), (bx + 75, base_y - 300), (bx - 75, base_y - 300)],
                  fill=CHAR_BODY, outline=LINE, width=5)                       # body
        d.polygon([(bx - 75, base_y - 300), (bx + 75, base_y - 300), (bx, base_y - 400)],
                  fill=CHAR_BODY, outline=LINE, width=5)                       # hood
        d.ellipse([bx - 52, base_y - 385, bx + 52, base_y - 295], fill=CHAR_HOOD, outline=LINE, width=5)
        d.ellipse([bx - 30, base_y - 360, bx - 6, base_y - 330], fill=LINE)    # eyes
        d.ellipse([bx + 6, base_y - 360, bx + 30, base_y - 330], fill=LINE)
        d.arc([bx - 22, base_y - 335, bx + 22, base_y - 305], 20, 160, fill=LINE, width=3)  # mouth
        d.rectangle([bx - 140, base_y - 200, bx - 95, base_y - 40], fill=CHAR_SH, outline=LINE, width=4)  # arms
        d.rectangle([bx + 95, base_y - 200, bx + 140, base_y - 40], fill=CHAR_SH, outline=LINE, width=4)
    return im

import os
os.makedirs(FRAME_DIR, exist_ok=True)
for t in range(FPS * SECS):
    draw_frame(t).save(f"{FRAME_DIR}/f{t:04d}.png")
print("frames done")
subprocess.run(["ffmpeg", "-v", "error", "-framerate", str(FPS), "-i", f"{FRAME_DIR}/f%04d.png",
                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                "-y", "test-media/segments/synth_clean_1080p.mp4"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-ss", "5", "-i", "test-media/segments/synth_clean_1080p.mp4",
                "-frames:v", "1", "-y", "test-media/frames/synth_t5_src1080.png"], check=True)
print("clip done")
