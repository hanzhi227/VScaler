#!/usr/bin/env python3
"""Check subtitle file loading, rendered pixels, switching, errors, and video scope."""
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
FFMPEG = "/opt/homebrew/bin/ffmpeg"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def ffmpeg(*args):
    subprocess.run([FFMPEG, "-v", "error", *map(str, args)], check=True)


with tempfile.TemporaryDirectory(prefix="vscaler-subs-") as directory:
    folder = Path(directory)
    plain = folder / "plain.mp4"
    embedded = folder / "embedded.srt"
    embedded.write_text("1\n00:00:00,000 --> 00:00:20,000\nEMBEDDED\n")
    ffmpeg("-f", "lavfi", "-i", "color=black:size=640x360:rate=24", "-t", "20",
           "-c:v", "libx264", "-preset", "ultrafast", plain)
    video = folder / "video.mkv"
    ffmpeg("-i", plain, "-i", embedded, "-map", "0:v", "-map", "1:s", "-c", "copy", video)
    external = folder / "external: viewer's subtitles.srt"
    external.write_text("1\n00:00:00,000 --> 00:00:04,000\nEXTERNAL SUBTITLES\n")
    vtt = folder / "external.vtt"
    ffmpeg("-i", external, vtt)
    ass = folder / "external.ass"
    later = folder / "later.srt"
    later.write_text("1\n00:00:06,000 --> 00:00:10,000\nASS SUBTITLES\n")
    ffmpeg("-i", later, ass)
    ssa = folder / "external.ssa"
    ssa.write_text("[Script Info]\nScriptType: v4.00\n\n[Events]\n"
                   "Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                   "Dialogue: Marked=0,0:00:00.00,0:00:10.00,Default,,0,0,0,,SSA SUBTITLES\n")
    bad = folder / "invalid.srt"
    bad.write_text("This is not a subtitle file.")
    script = (f"3:pause,4:readback,5:subfile:{external},7:readback,8:sub:off,10:readback,"
              f"11:subname:{external.name},13:readback,14:subfile:{bad},15:readback,16:sub:0,18:readback,"
              f"19:subfile:{vtt},21:readback,22:seek:6,24:readback,25:subfile:{ass},"
              f"27:readback,28:open:{plain},30:pause,31:readback,32:subfile:{ssa},34:readback")
    run = subprocess.run([
        str(ROOT / "engine/MacVideoUpscaler/.build/release/playercheck"),
        "--file", str(video), "--mode", "bypass", "--no-audio", "--swiftui",
        "--dump-at", "0", "--dump-sessions",
        "--seconds", "36", "--dump-dir", str(folder), "--script", script,
    ], capture_output=True, text=True, timeout=60)
    print(run.stderr, end="")
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    check(run.returncode == 0, f"Playback failed: {rows[-2:]}")
    events = [row for row in rows if "event" in row]
    errors = [row for row in events if row.get("error")]
    check(len(errors) == 1 and errors[0]["value"] == str(bad), f"Unexpected errors: {errors}")
    loads = [row for row in events if row["event"] == "subfile" and not row["error"]]
    check(len(loads) == 4, "Not all subtitle formats loaded")
    check(all(row["state_before"] == "paused" for row in loads), "Loading did not preserve pause")
    check(all(abs(row["playback_after"] - row["playback_before"]) < 0.2 for row in loads),
          f"Loading changed playback position: {loads}")
    check(any(row.get("subtitles") == external.name and any(track.get("title") == external.name for track in row.get("subtitleTracks", []))
              for row in rows), "Embedded and external tracks were not both listed")
    check(any(row.get("file") == plain.name and not row.get("subtitlesEnabled") for row in rows), "Subtitles leaked to the next video")
    captures = [row for row in events if row["event"] == "readback"]
    check(len(captures) == 11, "Missing frame checkpoints")
    images = [np.asarray(Image.open(folder / f"displayed_{i:03d}.png").convert("RGB"))
              for i in range(len(captures))]
    for i in (0, 2, 7, 9):
        check(images[i].max() < 10, f"Unexpected subtitles in blank frame {i}")
    for i in (1, 3, 4, 5, 6, 8, 10):
        check((images[i][-200:] > 100).sum() > 500, f"Missing rendered subtitles in frame {i}")
        check(images[i][:-200].max() < 10, f"Subtitle geometry overflow in frame {i}")
    check(np.array_equal(images[1], images[3]), "Reselection changed external rendering")
    check(np.array_equal(images[3], images[4]), "Invalid file replaced working subtitles")
    check(np.array_equal(images[1], images[6]), "VTT rendering differs from SRT")
    print("PASS: SRT/VTT/ASS/SSA pixels, embedded switching, Off/reselect, paused seek, invalid file, new video")
