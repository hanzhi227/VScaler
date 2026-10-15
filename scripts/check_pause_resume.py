#!/usr/bin/env python3
"""Regression: long pauses must preserve video/audio, seeking, and end of stream.

Run after building the release playercheck. Generates its own silent test clip.
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


with tempfile.TemporaryDirectory(prefix="vscaler-pause-") as directory:
    clip = Path(directory) / "clip.mp4"
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=320x180:rate=24", "-f", "lavfi", "-i",
        "anullsrc=r=48000:cl=stereo", "-t", "12", "-c:v", "libx264",
        "-preset", "ultrafast", "-c:a", "aac", str(clip),
    ], check=True)
    run = subprocess.run([
        str(PLAYER), "--file", str(clip), "--mode", "bypass", "--mute",
        "--swiftui", "--seconds", "43", "--script",
        "3:pause,15:resume,18:pause,20:seek:4,32:resume",
    ], capture_output=True, text=True, timeout=65)
    print(run.stderr, end="")
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    check(run.returncode == 0, f"playercheck failed: {rows[-1:]}")
    check(not any(row.get("state") == "failed" for row in rows), "Playback failed")
    events = [row for row in rows if "event" in row]
    check([row["event"] for row in events] ==
          ["pause", "resume", "pause", "seek", "resume"], "Missing control events")

    first_pause, first_resume, second_pause, seek, second_resume = events
    check(first_resume["wall"] - first_pause["wall"] >= 11, "Long pause was not observed")
    check(first_resume["state_before"] == "paused", "Pause did not hold")
    check(abs(first_resume["playback_before"] - first_pause["playback_after"]) < 0.1,
          "Playback clock advanced while paused")
    check(abs(first_resume["audio_frames_before"] - first_pause["audio_frames_before"]) <= 1024,
          "Audio continued feeding while paused")
    check(second_pause["playback_before"] > first_resume["playback_after"] + 1,
          "Video did not advance after resume")
    check(second_pause["audio_frames_before"] > first_resume["audio_frames_before"],
          "Audio feeder stopped during pause")
    check(second_resume["session_before"] > seek["session_before"],
          "Seek did not start a new session")
    check(second_resume["state_before"] == "paused" and
          abs(second_resume["playback_before"] - 4) < 0.2,
          "Seek did not preserve pause at the target")
    summary = next(row for row in rows if "final_state" in row)
    check(summary["final_state"] == "finished", f"Did not reach EOF: {summary}")
    check(summary["audio_frames"] > second_resume["audio_frames_before"],
          "Audio did not resume after seeking")
    for event in events:
        print(f'{event["event"]}: wall={event["wall"]:.1f}s '
              f'position={event["playback_before"]:.2f}s '
              f'state_before={event["state_before"]}')
    print("PASS: 12-second pauses, audio/video resume, paused seek, and natural EOF")
