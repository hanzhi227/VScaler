#!/usr/bin/env python3
"""Check text/VobSub positions, rotated video, paused updates and saved-style migration."""
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
PLAYER = ROOT / "engine/MacVideoUpscaler/.build/release/playercheck"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def probe(*args):
    return json.loads(subprocess.check_output([FFPROBE, "-v", "error", *map(str, args), "-of", "json"]))


with tempfile.TemporaryDirectory(prefix="vscaler-position-") as directory:
    folder = Path(directory)
    subs = folder / "Subs"
    subs.mkdir()
    video = folder / "movie.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                    "color=black:size=640x480:rate=24", "-t", "12", "-c:v", "libx264",
                    "-preset", "ultrafast", str(video)], check=True)
    (folder / "movie.en.srt").write_text("1\n00:00:00,000 --> 00:00:11,500\nSUBTITLE POSITION\n")
    index, data = subs / "bitmap.idx", subs / "bitmap.sub"
    source = ROOT / "test-media/segments/subs_bitmap.mkv"
    subprocess.run([FFMPEG, "-v", "fatal", "-i", str(source), "-map", "0:s:0",
                    "-c", "copy", "-f", "mpeg", str(data)], check=True)
    extra = probe("-select_streams", "s", "-show_entries", "stream=extradata", "-show_data", source)
    header = b"".join(bytes.fromhex(line.split(": ", 1)[1].split("  ", 1)[0])
                      for line in extra["streams"][0]["extradata"].splitlines() if ": " in line)
    packets = probe("-show_packets", "-show_entries", "packet=pts_time,pos", data)["packets"]
    lines = ["# VobSub index file, v7 (do not modify this line!)",
             header.decode().rstrip("\0"), "id: en, index: 0"]
    for packet in packets:
        ms = round(float(packet["pts_time"]) * 1000)
        lines.append(f"timestamp: {ms // 3600000:02}:{ms // 60000 % 60:02}:"
                     f"{ms // 1000 % 60:02}:{ms % 1000:03}, "
                     f"filepos: {int(packet['pos']) // 2048 * 2048:09x}")
    index.write_text("\n".join(lines) + "\n")
    cases = [("text", video, 0), ("external bitmap", video, 1)]
    for rotation in (90, 180, 270):
        rotated = folder / f"rotation{rotation}.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-display_rotation", str(rotation),
                        "-i", str(video), "-map", "0:v", "-c", "copy", str(rotated)], check=True)
        cases.append((f"bitmap rotated {rotation}", rotated, 1))
    embedded = folder / "embedded.mkv"
    subprocess.run([FFMPEG, "-v", "error", "-i", str(video), "-i", str(index),
                    "-map", "0:v", "-map", "1:s", "-c", "copy", str(embedded)], check=True)
    cases.append(("embedded bitmap", embedded, 0))

    for case_index, (label, clip, ordinal) in enumerate(cases):
        out = folder / f"case{case_index}"
        out.mkdir()
        run = subprocess.run([str(PLAYER), "--file", str(clip), "--mode", "off",
                              "--no-audio", "--swiftui", "--buffer-seconds", "0.3",
                              "--buffer-mb", "64", "--seconds", "8", "--sub-index", str(ordinal),
                              "--check-style-compatibility", "--dump-at", "0", "--dump-sessions",
                              "--dump-dir", str(out), "--script",
                              "1:pause,1.2:seek:1.7,2.5:readback,"
                              "3:subpos:0.1/0.2,4.5:readback,5:subpos:0/0,6.5:readback,7:resume"],
                             capture_output=True, text=True, timeout=25)
        rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
        check(run.returncode == 0, f"{label} failed: {rows[-2:]}\n{run.stderr}")
        captures = [row for row in rows if row.get("event") == "readback"]
        changes = [row for row in rows if row.get("event") == "subpos"]
        check(len(captures) == 3 and len(changes) == 2, f"{label}: missing controls")
        check(all(row["state_before"] == "paused" for row in captures + changes),
              f"{label}: changing position resumed playback")
        check(all(abs(row["playback_before"] - row["playback_after"]) < 0.15 for row in changes),
              f"{label}: position change sought to a different time")
        if "bitmap" in label:
            check(any(row.get("subtitles", "").endswith(".idx") or "bitmap)" in row.get("subtitles", "")
                      for row in rows), f"{label}: bitmap track was not selected")
        images = [np.asarray(Image.open(out / f"displayed_{i:03d}.png").convert("RGB"))
                  for i in range(len(captures))]
        original, moved, reset = images
        check((original > 100).sum() > 500, f"{label}: no subtitle pixels")
        height, width, _ = original.shape
        dx, dy = round(width * 0.1), -round(height * 0.2)
        expected = np.zeros_like(original)
        expected[:height + dy, dx:] = original[-dy:, :width - dx]
        difference = np.abs(expected.astype(int) - moved.astype(int))
        check(difference[2:-2, 2:-2].mean() < 0.1,
              f"{label}: wrong movement direction or distance: mean error {difference.mean()}")
        check(np.array_equal(original, reset), f"{label}: reset changed the original position")
        print(f"PASS: {label}, 10% right / 20% up, reset and paused time")
    print("PASS: backward-compatible preferences and offset validation")
