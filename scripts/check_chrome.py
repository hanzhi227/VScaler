#!/usr/bin/env python3
"""Check idle controls through the GUI's opt-in input diagnostics.

Run after a release build. Opens a muted synthetic clip for 35 seconds.
"""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def check(condition, message):
    if not condition:
        raise AssertionError(message)


with tempfile.TemporaryDirectory(prefix="vscaler-chrome-") as directory:
    clip = Path(directory) / "clip.mp4"
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=320x180:rate=24", "-t", "45", "-c:v", "libx264",
        "-preset", "ultrafast", str(clip),
    ], check=True)
    env = dict(os.environ, MACVIDEOUPSCALER_OPEN=str(clip),
               MACVIDEOUPSCALER_MODE="bypass", MACVIDEOUPSCALER_MUTE="1",
               MACVIDEOUPSCALER_DIAG="1", MACVIDEOUPSCALER_MOVE_AT="5,17,23",
               MACVIDEOUPSCALER_CLICK_AT="7,12", MACVIDEOUPSCALER_DBLCLICK_AT="18,28")
    started = datetime.now()
    process = subprocess.Popen([
        str(ROOT / "engine/MacVideoUpscaler/.build/release/MacVideoUpscalerApp"),
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = process.communicate(timeout=35)
    except subprocess.TimeoutExpired:
        process.terminate()
        output, _ = process.communicate(timeout=10)
    rows = []
    for line in output.splitlines():
        if not any(tag in line for tag in ("[diag]", "[chrome]", "[move]", "[click]", "[fullscreen]")):
            continue
        at = (datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f") - started).total_seconds()
        rows.append((at, line))
        if "[diag]" not in line:
            print(f"{at:.2f}s {line}")

    def observed(start, end, *parts):
        return any(start <= at <= end and all(part in line for part in parts)
                   for at, line in rows)

    check(not observed(0, 35, "state=failed"), "Playback failed")
    check(observed(8, 12, "state=paused"), "Pause did not occur")
    check(not observed(8, 12, "[chrome] visible=0"), "Controls hid while paused")
    check(observed(14, 18, "[chrome] visible=0"), "Controls did not hide after resume")
    check(observed(17, 18.5, "[chrome] visible=1", "full=0"), "Windowed mouse reveal failed")
    check(observed(18, 23, "[fullscreen] entered=1 menuAutoHide=1"), "Menu bar auto-hide was not enabled")
    check(observed(20, 23, "[chrome] visible=0", "full=1"), "Fullscreen controls did not hide")
    check(observed(23, 25, "[chrome] visible=1", "full=1"), "Fullscreen mouse reveal failed")
    check(observed(25, 28, "[chrome] visible=0", "full=1"), "Fullscreen controls did not hide again")
    check(observed(31, 35, "[chrome] visible=0", "full=0"), "Controls did not hide after fullscreen exit")
    check(observed(28, 33, "[fullscreen] entered=0 menuAutoHide=0"), "Menu bar settings were not restored")
    print("PASS: idle hide, mouse reveal, pause/resume, fullscreen entry/exit, menu bar restoration")
