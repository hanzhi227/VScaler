#!/usr/bin/env python3
"""Check external .idx/.sub selection and seek timing with a local bitmap fixture.

Usage: check_vobsub.py [path/to/subs_bitmap.mkv]
The optional fixture must be the same 640x480 bitmap test clip used by the repo.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test-media/segments/subs_bitmap.mkv"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def probe(*args):
    return json.loads(subprocess.check_output([FFPROBE, "-v", "error", *map(str, args), "-of", "json"]))


with tempfile.TemporaryDirectory(prefix="vscaler-vobsub-") as directory:
    folder = Path(directory)
    subs = folder / "subs"
    subs.mkdir()
    index, data = subs / "test.idx", subs / "test.sub"
    subprocess.run([FFMPEG, "-v", "fatal", "-i", str(source), "-map", "0:s:0",
                    "-c", "copy", "-f", "mpeg", str(data)], check=True)
    extra = probe("-select_streams", "s", "-show_entries", "stream=extradata", "-show_data", source)
    # Matroska stores the VobSub canvas/palette as its subtitle codec private data.
    header = b"".join(bytes.fromhex(line.split(": ", 1)[1].split("  ", 1)[0])
                      for line in extra["streams"][0]["extradata"].splitlines() if ": " in line)
    packets = probe("-show_packets", "-show_entries", "packet=pts_time,pos", data)["packets"]
    lines = ["# VobSub index file, v7 (do not modify this line!)",
             header.decode().rstrip("\0"), "id: en, index: 0"]
    for packet in packets:
        ms = round(float(packet["pts_time"]) * 1000)
        # MPEG program stream packs are 2048 bytes; index the containing pack.
        lines.append(f"timestamp: {ms // 3600000:02}:{ms // 60000 % 60:02}:"
                     f"{ms // 1000 % 60:02}:{ms % 1000:03}, "
                     f"filepos: {int(packet['pos']) // 2048 * 2048:09x}")
    index.write_text("\n".join(lines) + "\n")
    missing = subs / "missing.idx"
    missing.write_text(index.read_text())
    video, next_video = folder / "black.mp4", folder / "next.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                    "color=black:size=640x480:rate=24", "-t", "24", "-c:v", "libx264",
                    "-preset", "ultrafast", str(video)], check=True)
    next_video.write_bytes(video.read_bytes())
    script = (f"3:pause,4:seek:1.7,5:readback,6:subfile:{index},8:readback,9:sub:off,11:readback,"
              f"12:subfile:{data},14:readback,15:seek:8,17:readback,18:seek:10,20:readback,"
              f"21:subfile:{missing},23:readback,24:open:{next_video},26:readback")
    run = subprocess.run([
        str(ROOT / "engine/MacVideoUpscaler/.build/release/playercheck"),
        "--file", str(video), "--mode", "bypass", "--no-audio", "--swiftui",
        "--dump-at", "0", "--dump-sessions",
        "--seek", "1", "--seconds", "28", "--dump-dir", str(folder), "--script", script,
    ], capture_output=True, text=True, timeout=50)
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    check(run.returncode == 0, f"Playback failed: {rows[-2:]}\n{run.stderr}")
    errors = [row for row in rows if row.get("error")]
    check(len(errors) == 1 and ".idx" in errors[0]["error"] and ".sub" in errors[0]["error"],
          f"Missing-pair validation failed: {errors}")
    check(any(row.get("file") == next_video.name and not row.get("subtitlesEnabled") for row in rows), "VobSub leaked to the next video")
    check(any(any(track["codec"] == "dvd_subtitle" and track["index"] < 0
                  for track in row.get("subtitleTracks", [])) for row in rows), "VobSub track missing")
    captures = [row for row in rows if row.get("event") == "readback"]
    check(len(captures) == 8, "Missing frame checkpoints")
    images = [np.asarray(Image.open(folder / f"displayed_{i:03d}.png").convert("RGB"))
              for i in range(len(captures))]
    for i in (0, 2, 4, 7):
        check(images[i].max() < 10, f"Unexpected subtitle pixels in blank frame {i}")
    for i in (1, 3, 5, 6):
        check((images[i] > 100).sum() > 500, f"VobSub cue missing in frame {i}")
    check(np.array_equal(images[1], images[3]), "Selecting .sub differs from selecting .idx")
    check(np.array_equal(images[5], images[6]), "Missing companion replaced working subtitles")
    print("PASS: .idx/.sub pixels, Off, seek into cue and gap, missing companion, new video")
