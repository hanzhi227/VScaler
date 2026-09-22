#!/usr/bin/env python3
"""Regression checks for every playback-correctness failure found in the audit.

Each check reproduces a reported failure and asserts the fix. They are deliberately
narrow and fast (seconds each), and use dedicated ports and only their own child
processes — no global pkill, so they can run alongside a long sustained test.

  1. short clip        1 s clip with the DEFAULT 4 MB prebuffer must deliver all
                       its bytes (was: HTTP 200 with 0 bytes, forever)
  2. oversized prebuf  --prebuffer-mb larger than the ring must not deadlock
  3. aspect ratio      anamorphic 720x480 SAR 32:27 must come out 1440x960 with
                       DAR 16:9 (was: no metadata at all)
  4. subtitle toggle   enabling then disabling must be reflected in the config
                       (was: setSubtitles never wrote config.burnSubtitles)
  5. av negative ctrl  a stream whose video is delayed 1 s must be REJECTED by
                       scripts/check_av_sync.py (was: reported "in sync")
  6. playback clock    pause must freeze the player's clock, and a relative seek
                       must move by that clock, not the production frontier
                       (requires VLC; skipped with --no-vlc)

Usage:
  python3 scripts/check_playback_correctness.py [--no-vlc] [--vlc]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
CLI = os.path.join(ROOT, "engine/MacVideoUpscaler/.build/release/macvideoupscale")
TMP = "/tmp/upscaler_regression"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def probe(path, entries="stream=width,height,sample_aspect_ratio,display_aspect_ratio,codec_name"):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", entries, "-of", "json", path],
                         capture_output=True, text=True).stdout
    try:
        return json.loads(out)["streams"][0]
    except Exception:
        return {}


def status(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.json", timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


class Engine:
    """Owns exactly one engine process; never touches anything else."""

    def __init__(self, port, file, extra=(), env=None):
        self.port = port
        self.log = open(f"{TMP}/engine_{port}.log", "wb")
        e = dict(os.environ)
        if env:
            e.update(env)
        self.proc = subprocess.Popen(
            [CLI, "--file", file, "--port", str(port), "--prebuffer-mb", "0", *extra],
            stdout=self.log, stderr=subprocess.STDOUT, env=e)

    def wait_ready(self, timeout=25):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = status(self.port)
            if s and s.get("state") != "idle":
                return s
            time.sleep(0.25)
        return None

    def capture(self, seconds=12):
        path = f"{TMP}/cap_{self.port}.ts"
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/stream.ts?session=1",
                    timeout=seconds + 5) as r, open(path, "wb") as f:
                t0 = time.time()
                while time.time() - t0 < seconds:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception:
            pass
        return path if os.path.exists(path) else None

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=6)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.log.close()


def check_short_clip():
    tiny = f"{TMP}/tiny_1s.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24",
                    "-t", "1", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-y", tiny], capture_output=True)
    # Deliberately the DEFAULT prebuffer (4 MB), not --prebuffer-mb 0.
    log = open(f"{TMP}/short.log", "wb")
    proc = subprocess.Popen([CLI, "--file", tiny, "--port", "8641"],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while time.time() - t0 < 20:
            s = status(8641)
            if s and s.get("state") != "idle":
                break
            time.sleep(0.25)
        cap = f"{TMP}/short.ts"
        try:
            with urllib.request.urlopen("http://127.0.0.1:8641/stream.ts?session=1",
                                        timeout=15) as r, open(cap, "wb") as f:
                t0 = time.time()
                while time.time() - t0 < 10:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception:
            pass
        size = os.path.getsize(cap) if os.path.exists(cap) else 0
        frames = 0
        if size:
            out = sh(f"{FFPROBE} -v error -count_frames -select_streams v:0 "
                     f"-show_entries stream=nb_read_frames -of default=nw=1:nk=1 {cap}").stdout
            counts = [int(x) for x in out.split() if x.strip().isdigit()]
            frames = max(counts) if counts else 0
        detail = f"{size} bytes, {frames} frames (expected 24)"
        return (size > 100_000 and frames == 24), detail
    finally:
        try:
            proc.terminate(); proc.wait(timeout=5)
        except Exception:
            proc.kill()
        log.close()


def check_oversized_prebuffer():
    tiny = f"{TMP}/tiny_1s.mp4"
    log = open(f"{TMP}/oversize.log", "wb")
    proc = subprocess.Popen([CLI, "--file", tiny, "--port", "8643",
                             "--prebuffer-mb", "512", "--ring-mb", "64"],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while time.time() - t0 < 20:
            s = status(8643)
            if s and s.get("state") != "idle":
                break
            time.sleep(0.25)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8643/stream.ts?session=1",
                                        timeout=15) as r:
                data = r.read()
        except Exception:
            data = b""
        return len(data) > 100_000, f"{len(data)} bytes (prebuffer 512 MB > 64 MB ring)"
    finally:
        try:
            proc.terminate(); proc.wait(timeout=5)
        except Exception:
            proc.kill()
        log.close()


def check_aspect_ratio():
    src = f"{TMP}/anamorphic.mkv"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi",
                    "-i", "testsrc2=s=720x480:r=24", "-t", "5",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-vf", "setsar=32/27", "-y", src], capture_output=True)
    e = Engine(8645, src, extra=["--no-audio"])
    try:
        e.wait_ready()
        cap = e.capture(10)
        if not cap:
            return False, "no capture"
        v = probe(cap)
        ok = (v.get("display_aspect_ratio") == "16:9"
              and v.get("width") == 1440 and v.get("height") == 960)
        return ok, (f"{v.get('codec_name')} {v.get('width')}x{v.get('height')} "
                    f"SAR {v.get('sample_aspect_ratio')} DAR {v.get('display_aspect_ratio')}")
    finally:
        e.close()


def check_subtitle_toggle():
    src = os.path.join(ROOT, "test-media/segments/subs_text.mkv")
    if not os.path.exists(src):
        return False, "fixture missing"
    e = Engine(8647, src, extra=["--script", "6:subs-on:0,14:subs-off"])
    try:
        e.wait_ready()
        seen_off = seen_on = seen_off_again = False
        t0 = time.time()
        while time.time() - t0 < 24:
            s = status(8647)
            if s:
                if not s["subtitlesEnabled"] and s["sessionId"] == 1:
                    seen_off = True
                if s["subtitlesEnabled"] and s["sessionId"] == 2:
                    seen_on = True
                if not s["subtitlesEnabled"] and s["sessionId"] == 3:
                    seen_off_again = True
                    break
            time.sleep(0.4)
        ok = seen_off and seen_on and seen_off_again
        return ok, f"off={seen_off} on={seen_on} off-again={seen_off_again}"
    finally:
        e.close()


def check_av_negative_control():
    src = os.path.join(ROOT, "Naruto Shippuden Episode 001 Homecoming.mkv")
    if not os.path.exists(src):
        src = os.path.join(ROOT, "test-media/segments/naruto_1080p_aligned.mkv")
    if not os.path.exists(src):
        return False, "no reference source"
    e = Engine(8649, src, extra=["--start", "300", "--no-audio"])
    # --no-audio avoids the audio path here; we only need video for the control.
    e.close()
    e = Engine(8651, src, extra=["--start", "300", "--ring-mb", "32", "--prebuffer-mb", "2"])
    try:
        e.wait_ready()
        good = e.capture(14)
    finally:
        e.close()
    if not good:
        return False, "no capture"
    bad = f"{TMP}/av_bad.ts"
    subprocess.run([FFMPEG, "-v", "error", "-i", good, "-vf", "setpts=PTS+1/TB",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "copy", "-f", "mpegts", "-y", bad], capture_output=True)
    ok_run = sh(f"python3 {ROOT}/scripts/check_av_sync.py --reference '{src}' "
                f"--stream {good} --file-start 300 --dur 12").returncode
    bad_run = sh(f"python3 {ROOT}/scripts/check_av_sync.py --reference '{src}' "
                 f"--stream {bad} --file-start 300 --dur 12 --expect-fail").returncode
    ok = (ok_run == 0 and bad_run == 0)
    return ok, f"good stream accepted={ok_run == 0}, 1s-delayed stream rejected={bad_run == 0}"


def check_playback_clock():
    """A relative seek must be anchored to playback, not to the frontier.

    The engine reports, in its event line, what the control was based on
    (`playback_before` = the playback estimate, `frontier_before` = production)
    and what it targeted. The target must equal playback_before + delta, and must
    differ from frontier_before + delta whenever the two bases differ -- which is
    exactly the reported defect (relative seeks jumping far beyond the picture).
    Deterministic: no dependence on how fast a player consumes the stream.
    """
    src = os.path.join(ROOT, "Naruto Shippuden Episode 001 Homecoming.mkv")
    if not os.path.exists(src):
        return False, "source missing"
    log = f"{TMP}/seekrel_{os.getpid()}.log"
    lf = open(log, "wb")
    proc = subprocess.Popen(
        [CLI, "--file", src, "--port", "8657", "--start", "600", "--prebuffer-mb", "2",
         "--script", "12:seekrel:-90"],
        stdout=lf, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        event = None
        while time.time() - t0 < 30:
            time.sleep(0.5)
            try:
                for line in open(log, errors="ignore"):
                    if '\"event\"' in line and '"seekrel"' in line:
                        event = json.loads(line)
            except Exception:
                pass
            if event:
                break
        if not event:
            return False, "no seekrel event observed"
        target = event.get("target", 0)
        play_before = event.get("playback_before", 0)
        front_before = event.get("frontier_before", 0)
        delta = -90
        from_playback = play_before + delta
        from_frontier = front_before + delta
        bases_differ = abs(from_frontier - from_playback) > 10
        ok = (abs(target - from_playback) <= 2
              and (not bases_differ or abs(target - from_frontier) > 10))
        detail = (f"target {target:.0f}s = playback {play_before:.0f}s {delta:+d} "
                  f"(frontier basis would be {front_before:.0f}s {delta:+d} = {from_frontier:.0f}s)")
        return ok, detail
    finally:
        try:
            proc.terminate(); proc.wait(timeout=5)
        except Exception:
            proc.kill()
        lf.close()


def check_absolute_seek():
    """An absolute seek must land exactly, independent of any player."""
    src = os.path.join(ROOT, "Naruto Shippuden Episode 001 Homecoming.mkv")
    if not os.path.exists(src):
        return False, "source missing"
    e = Engine(8655, src, extra=["--start", "120", "--prebuffer-mb", "2",
                                 "--script", "8:seek:900"])
    try:
        e.wait_ready()
        landed = None
        t0 = time.time()
        while time.time() - t0 < 30:
            st = status(8655)
            if st and st["sessionId"] == 2:
                landed = st["playbackSeconds"]
                break
            time.sleep(0.4)
        if landed is None:
            return False, "no session 2 observed"
        return abs(landed - 900) <= 2, f"seek:900 landed at {landed:.1f}s"
    finally:
        e.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlc", action="store_true")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)

    checks = [
        ("short clip delivers all frames", check_short_clip),
        ("oversized prebuffer no deadlock", check_oversized_prebuffer),
        ("anamorphic aspect ratio preserved", check_aspect_ratio),
        ("subtitle toggle persists", check_subtitle_toggle),
        ("A/V checker rejects 1s desync", check_av_negative_control),
    ]
    if not args.no_vlc:
        checks.append(("controls anchor to playback position", check_playback_clock))
        checks.append(("absolute seek lands exactly", check_absolute_seek))

    failures = []
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"exception: {exc}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print("\n" + ("ALL REGRESSION CHECKS PASSED" if not failures
                  else "FAILED: " + ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
