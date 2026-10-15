#!/usr/bin/env python3
"""Discover nested subtitles and load multiple files without duplicating VobSub pairs."""
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


with tempfile.TemporaryDirectory(prefix="vscaler-discovery-") as directory:
    root = Path(directory)
    folder = root / "movie"
    nested = folder / "Subtitles" / "English" / "Nested"
    nested.mkdir(parents=True)
    manual = root / "manual"
    manual.mkdir()
    next_folder = root / "next"
    next_folder.mkdir()
    video = folder / "film.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                    "color=black:size=640x480:rate=24", "-t", "20", "-c:v", "libx264",
                    "-preset", "ultrafast", str(video)], check=True)
    next_video = next_folder / "next.mp4"
    next_video.write_bytes(video.read_bytes())
    for path, text in [(folder / "film.en.srt", "NEARBY"),
                       (nested / "deep.srt", "NESTED SUBTITLE"),
                       (manual / "outside.srt", "MANUALLY SELECTED")]:
        path.write_text(f"1\n00:00:00,000 --> 00:00:19,000\n{text}\n")
    (folder / "linked-folder").symlink_to(manual, target_is_directory=True)
    (folder / "linked.srt").symlink_to(manual / "outside.srt")
    (folder / "invalid.srt").write_text("not a subtitle")
    source = ROOT / "test-media/segments/subs_bitmap.mkv"
    index, data = nested / "bitmap.IDX", nested / "bitmap.SUB"
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
    missing = nested / "missing.idx"
    missing.write_text(index.read_text())
    outside = manual / "outside.srt"
    script = (f"3:pause,4:subname:deep.srt,6:readback,"
              f"7:subfiles:{index}|{data}|{outside},9:readback,"
              f"10:subname:outside.srt,12:readback,"
              f"13:subfiles:{outside}|{missing},15:readback,"
              f"16:open:{next_video},18:readback")
    run = subprocess.run([str(PLAYER), "--file", str(video), "--mode", "bypass",
                          "--no-audio", "--swiftui", "--seconds", "20",
                          "--dump-at", "0", "--dump-sessions", "--dump-dir", str(root),
                          "--script", script], capture_output=True, text=True, timeout=40)
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    check(run.returncode == 0, f"Playback failed: {rows[-2:]}\n{run.stderr}")
    initial = next(row for row in rows if row.get("file") == video.name and "subtitleTracks" in row)
    tracks = initial["subtitleTracks"]
    check(len(tracks) == 3, f"Unexpected discovered tracks: {tracks}")
    check(any(track.get("title") == "film.en.srt" for track in tracks), "Nearby text subtitle missing")
    check(any("Subtitles/English/Nested/deep.srt" == track.get("title") for track in tracks),
          "Nested subtitle missing")
    check(sum(track["codec"] == "dvd_subtitle" for track in tracks) == 1,
          "VobSub pair not discovered exactly once")
    check(not initial["subtitlesEnabled"], "Discovery unexpectedly overrode Off")
    events = [row for row in rows if "event" in row]
    errors = [row for row in events if row.get("error")]
    check(len(errors) == 1 and errors[0]["at"] == 13, f"Multi-file load validation failed: {errors}")
    merged = [row for row in rows if row.get("subtitles") == index.name and "subtitleTracks" in row]
    check(merged and all(len(row["subtitleTracks"]) == 4 for row in merged),
          f"Multi-file load replaced discovered files or duplicated the pair: {merged}")
    check(all(sum(track["codec"] == "dvd_subtitle" for track in row["subtitleTracks"]) == 1
              for row in merged), "Selecting .idx and .sub duplicated streams")
    captures = [row for row in events if row["event"] == "readback"]
    images = [np.asarray(Image.open(root / f"session_{row['session_before']:04d}_frame_0000.png").convert("RGB"))
              for row in captures]
    for index in (0, 2, 3):
        check((images[index] > 100).sum() > 500, f"Text subtitle not rendered at checkpoint {index}")
    check(np.array_equal(images[2], images[3]), "Invalid selection replaced working subtitles")
    check(images[4].max() < 10, "External subtitles leaked to the next video")
    check(any(row.get("file") == next_video.name and row.get("subtitleTracks") == [] and
              not row.get("subtitlesEnabled") for row in rows), "Video change retained old discoveries")
    print("PASS: recursive discovery, relative menu labels, uppercase VobSub pair, multi-selection, deduplication, rendered text, atomic errors, video scope and symlink boundaries")
