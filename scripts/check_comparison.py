#!/usr/bin/env python3
"""Native wipe regression: paired presentation, paused dragging, audio and seeks."""
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
PLAYER = ROOT / "engine/MacVideoUpscaler/.build/release/playercheck"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


with tempfile.TemporaryDirectory(prefix="vscaler-comparison-") as directory:
    clip = Path(directory) / "clip.mp4"
    # 4:3 in a 16:9 window also exercises letterboxing and divider hit testing.
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=320x240:rate=24", "-f", "lavfi", "-i",
        "anullsrc=r=48000:cl=stereo", "-t", "12", "-c:v", "libx264",
        "-preset", "ultrafast", "-c:a", "aac", str(clip),
    ], check=True)
    run = subprocess.run([
        str(PLAYER), "--file", str(clip), "--mode", "cnn", "--mute",
        "--swiftui", "--window", "--seconds", "37", "--script",
        "3:pause,3.3:wipecheck,4:compare:on,4.1:wipecheck,"
        "5:split:0,5.1:wipecheck,6:split:1,6.1:wipecheck,"
        "7:dragdivider:0.25,7.1:wipecheck,8:dragdivider:0.75,8.1:wipecheck,"
        "9:compare:off,9.1:wipecheck,10:compare:on,10.1:wipecheck,"
        "11:resume,12:wipecheck,13:split:0.4,13.1:wipecheck,"
        "14:pause,14.3:wipecheck,15:compareleft:cnn,17:wipecheck,"
        "18:compareleft:bypass,20:wipecheck,21:seek:4,23:wipecheck,24:resume",
    ], capture_output=True, text=True, timeout=55)
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    check(run.returncode == 0, f"playercheck failed: {run.stderr}\n{rows[-1:]}")
    check(not any(row.get("state") == "failed" for row in rows), "Playback failed")
    events = [row for row in rows if "event" in row]
    checks = [row for row in events if row["event"] == "wipecheck"]
    check(len(checks) == 14, f"Missing presentation checks: {len(checks)}")
    for row in events:
        wipe = row["wipe"]
        check(wipe.get("layer_count") == 2 and wipe.get("aligned"),
              f"Full-frame layers lost alignment: {row}")
        check(not wipe["animated"], f"Mask has delayed animation: {row}")
        check(abs(wipe["mask_fraction"] - row["compare_split"]) < 1e-6,
              f"Mask did not move in the same control event: {row}")
        if row["event"] in ("split", "dragdivider", "compare"):
            check(row["action_ms"] < 50, f"Wipe update blocked: {row}")
    paused = [row for row in checks if row["at"] < 11]
    baseline = paused[0]
    for row in paused:
        check(row["state_before"] == "paused", "Wipe resumed playback")
        for key in ("playback_before", "audio_frames_before", "frames", "session_before"):
            check(row[key] == baseline[key], f"Wipe changed {key} while paused")
        for index in (0, 1):
            key = f"frame_{index}_sha256"
            check(key in row["wipe"], f"No presented image for layer {index}: {row}")
            check(row["wipe"][key] == baseline["wipe"][key],
                  "Dragging or toggling replaced the frozen image")
    check(baseline["wipe"]["frame_0_sha256"] != baseline["wipe"]["frame_1_sha256"],
          "Reference layer duplicated the enhancement instead of showing bilinear")
    for row in paused:
        check(row["wipe"]["hidden"] == (row["at"] in (3.3, 9.1)), "Comparison visibility is wrong")
    playing = next(row for row in checks if row["at"] == 12)
    check(playing["playback_before"] > baseline["playback_before"] + 0.5,
          "Video did not resume")
    check(playing["audio_frames_before"] > baseline["audio_frames_before"],
          "Wipe stopped the audio feeder")
    same_model = next(row for row in checks if row["at"] == 17)
    check(same_model["state_before"] == "paused", "Changing the reference resumed playback")
    check(same_model["wipe"].get("frame_0_sha256") == same_model["wipe"].get("frame_1_sha256"),
          "Same-model comparison did not show identical frames")
    different_model = next(row for row in checks if row["at"] == 20)
    check(different_model["state_before"] == "paused", "Reference change lost pause")
    check(different_model["wipe"].get("frame_0_sha256") != different_model["wipe"].get("frame_1_sha256"),
          "Reference change did not present a different model")
    sought = next(row for row in checks if row["at"] == 23)
    check(sought["state_before"] == "paused" and abs(sought["playback_before"] - 4) < 0.2,
          "Paused seek did not preserve comparison and position")
    summary = next(row for row in rows if "final_state" in row)
    check(summary["final_state"] == "finished", f"Did not reach EOF: {summary}")
    latency = max(row["action_ms"] for row in events if row["event"] in ("split", "dragdivider", "compare"))
    print(f"PASS: full-frame overlay, letterboxed drags, frozen images, toggles, model changes, paused seek, audio/resume and EOF; worst wipe update {latency:.2f} ms")
