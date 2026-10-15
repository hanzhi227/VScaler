#!/usr/bin/env python3
"""Verify output-resolution subtitle fonts, size/color/outline and paused updates."""
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PLAYER = ROOT / "engine/MacVideoUpscaler/.build/release/playercheck"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


with tempfile.TemporaryDirectory(prefix="vscaler-appearance-") as directory:
    folder = Path(directory)
    video = folder / "clip.mp4"
    sub = folder / "clip.en.srt"
    sub.write_text("1\n00:00:00,000 --> 00:00:19,000\nA clearer view of every word.\n")
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=black:size=640x360:rate=24", "-t", "20", "-c:v", "libx264",
                    "-preset", "ultrafast", str(video)], check=True)
    run = subprocess.run([str(PLAYER), "--file", str(video), "--mode", "cnn",
                          "--no-audio", "--swiftui", "--window", "--sub-index", "0",
                          "--compare", "--dump-wipe", "--seconds", "22", "--dump-at", "0",
                          "--dump-sessions", "--dump-dir", str(folder), "--script",
                          "3:pause,4:readback,4.1:wipecheck,"
                          "5:substyle:avenir/1.5/yellow/strong,7:readback,7.1:wipecheck,"
                          "8:substyle:helvetica/0.75/warm/subtle,10:readback,"
                          "11:substyle:georgia/1/white/standard,13:readback,"
                          "14:substyle:system/1/white/standard,16:readback,"
                          "17:sub:off,19:readback,20:resume"],
                         capture_output=True, text=True, timeout=40)
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    check(run.returncode == 0, f"Playback failed: {rows[-2:]}\n{run.stderr}")
    events = [row for row in rows if "event" in row]
    check(not any(row.get("error") for row in events), "Appearance control failed")
    changes = [row for row in events if row["event"] == "substyle"]
    check(len(changes) == 4 and all(row["state_before"] == "paused" for row in changes),
          "Appearance changes lost pause")
    check(all(abs(row["playback_before"] - row["playback_after"]) < 0.15 for row in changes),
          "Appearance changes moved the requested position")
    captures = [row for row in events if row["event"] == "readback"]
    images = [np.asarray(Image.open(folder / f"session_{row['session_before']:04d}_frame_0000.png").convert("RGB"))
              for row in captures]
    check(len(images) == 6, "Missing style checkpoints")
    default, large, small, serif, reset, off = images
    check(default.shape == (720, 1280, 3), "Subtitles were not composed at output resolution")
    for image in images[:-1]:
        check((image > 100).sum() > 500 and image[:-200].max() < 10,
              "Subtitle missing or drawn outside its bottom margin")
    check(np.array_equal(default, reset), "Reset style did not reproduce the default")
    check(not np.array_equal(default, serif), "Font change did not change glyphs")
    check(off.max() < 10, "Turning subtitles Off left text behind")
    yellow = (large[:, :, 0] > 180) & (large[:, :, 1] > 140) & (large[:, :, 2] < 100)
    check(yellow.sum() > 500, "Yellow setting was not rendered")
    def glyph_height(image):
        ys = np.where(image.max(axis=2) > 100)[0]
        return int(ys.max() - ys.min() + 1)
    check(glyph_height(large) > glyph_height(small) * 1.5, "Subtitle size did not scale")
    # Compare visible RGB glyphs, excluding pixel-buffer alpha and row padding.
    for index in range(2):
        right = np.asarray(Image.open(folder / f"comparison_{index:03d}_0.png").convert("RGB"))
        left = np.asarray(Image.open(folder / f"comparison_{index:03d}_1.png").convert("RGB"))
        right_mask, left_mask = right[:, :, 0] > 128, left[:, :, 0] > 128
        overlap = np.count_nonzero(right_mask & left_mask)
        union = np.count_nonzero(right_mask | left_mask)
        check(union > 500 and overlap / union > 0.995,
              f"Comparison altered subtitle glyphs: overlap={overlap}/{union}")
    print("PASS: four fonts, size/color/outline, output-resolution text, identical comparison lettering, paused application, reset and Off")
