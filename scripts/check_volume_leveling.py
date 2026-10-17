#!/usr/bin/env python3
"""Check volume leveling across live toggles, pause, track changes, and seeks.

Run after swift test --package-path engine/MacVideoUpscaler -c release.
The Swift audio test separately measures dynamics, clipping, and PCM duration.
"""
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
PLAYER = ROOT / "engine/MacVideoUpscaler/.build/release/playercheck"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


with tempfile.TemporaryDirectory(prefix="vscaler-leveling-") as directory:
    clip = Path(directory) / "clip.mkv"
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=320x180:rate=24", "-f", "lavfi", "-i",
        "sine=frequency=1000:sample_rate=48000", "-map", "0:v", "-map", "1:a",
        "-map", "1:a", "-t", "16", "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "pcm_s16le", str(clip),
    ], check=True)
    run = subprocess.run([
        str(PLAYER), "--file", str(clip), "--mode", "bypass", "--mute",
        "--volume-leveling", "--seconds", "20", "--script",
        "2:leveling:off,4:pause,5:leveling:on,7:resume,9:audio:1,11:seek:4",
    ], capture_output=True, text=True, timeout=40)
    print(run.stderr, end="")
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    check(run.returncode == 0, f"playercheck failed: {rows[-1:]}")
    check(not any(row.get("state") == "failed" for row in rows), "Playback failed")
    snapshots = [row for row in rows if "volumeLeveling" in row]
    events = [row for row in rows if "event" in row]
    check([row["event"] for row in events] ==
          ["leveling", "pause", "leveling", "resume", "audio", "seek"], "Missing control events")
    off, pause, on, resume, track, seek = events
    check(snapshots[0]["volumeLeveling"], "Launch flag did not enable leveling")
    check(any(not row["volumeLeveling"] for row in snapshots), "Toggle off was ignored")
    check(pause["session_before"] > off["session_before"], "Live toggle did not restart")
    check(resume["session_before"] > on["session_before"], "Paused toggle did not restart")
    check(resume["state_before"] == "paused", "Toggle lost the paused state")
    check(abs(resume["playback_before"] - on["playback_before"]) < 0.1,
          "Paused toggle moved the playback position")
    check(track["audio_frames_before"] > resume["audio_frames_before"], "Audio did not resume")
    check(seek["session_before"] > track["session_before"], "Track change did not restart")
    last = snapshots[-1]
    check(last["volumeLeveling"] and last["audioTrackOrdinal"] == 1,
          "Track change or seek lost audio settings")
    check(last["audioFramesEnqueued"] > 48000, "Audio stopped after seeking")
    check("audio=rendering" in last["renderStatus"], "Audio renderer did not recover")
    summary = next(row for row in rows if "final_state" in row)
    check(summary["playback"] > 10, "Playback did not advance after seeking")
    print("PASS: launch setting, live toggles, paused position, resume, track switch, and seek")
