#!/usr/bin/env python3
"""Measure whether the stream plays at real time, and why.

A player clocks a stream by its timestamps (PTS/PCR), so the playback rate must
be 1.0x regardless of how much faster than realtime the engine produces frames.
This check measures both sides of that:

  1. timestamps - the TS packet PTS spacing of the captured stream (ground truth:
     video PTS must advance by exactly 1/fps; the capture's duration must equal
     frames/fps)
  2. player     - VLC's reported playback position vs wall clock. VLC runs with
     dummy video/audio outputs so no window appears; it is driven through its
     HTTP interface.

The correctness of the VLC measurement itself is validated first on a local file
(--local), which must measure 1.0x; otherwise the method (not the stream) is wrong.

Usage:
  python3 scripts/check_playback_rate.py --file <video> [--scale 4] [--port 8430]
  python3 scripts/check_playback_rate.py --local /tmp/clip.mkv   # validate method
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
CLI = os.path.join(ROOT, "engine/MacVideoUpscaler/.build/release/macvideoupscale")
VLC_BIN = "/Applications/VLC.app/Contents/MacOS/VLC"
HTTP_PORT = 8085
HTTP_PASS = "animelab"


def vlc_status():
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{HTTP_PORT}/requests/status.json")
        req.add_header("Authorization", "Basic " + _b64(f":{HTTP_PASS}"))
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _b64(s):
    import base64
    return base64.b64encode(s.encode()).decode()


def video_pts_stats(path):
    """Median video packet PTS delta and audio PTS span from a TS capture."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout
    pts = []
    for line in out.splitlines():
        field = line.split(",")[0].strip()
        if not field or field == "N/A":
            continue
        try:
            pts.append(float(field))
        except ValueError:
            pass
    pts.sort()
    d = [round(b - a, 6) for a, b in zip(pts, pts[1:])]
    med = statistics.median(d) if d else 0.0
    return len(pts), med, (d.count(med) / len(d) if d else 0.0)


def _slope(points):
    """Least-squares slope of y vs x, or None if too few points."""
    if len(points) < 4:
        return None
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in points)
    den = sum((p[0] - mx) ** 2 for p in points)
    return (num / den) if den else None


def measure_vlc(url, seconds, status_url=None):
    """Sample VLC playback position and the engine frontier -> rates.

    Returns (vlc_rate, engine_rate, est_play_rate, samples).
    """
    proc = subprocess.Popen(
        [VLC_BIN, "-I", "dummy", "--vout=dummy", "--aout=dummy",
         "--extraintf=http", f"--http-password={HTTP_PASS}",
         f"--http-port={HTTP_PORT}", "--network-caching=1500", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    samples, eng = [], []
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            st = vlc_status()
            if st and st.get("state") == "playing":
                t = st.get("time")
                if isinstance(t, (int, float)):
                    samples.append((time.time() - t0, float(t)))
            if status_url:
                try:
                    with urllib.request.urlopen(status_url, timeout=2) as r:
                        d = json.loads(r.read())
                    w = time.time() - t0
                    eng.append((w, float(d.get("positionSeconds", 0)),
                                float(d.get("playbackSeconds", 0))))
                except Exception:
                    pass
            time.sleep(0.5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            proc.kill()
    # Keep only monotonic samples (ignore startup jumps/buffering).
    good = [s for i, s in enumerate(samples) if i == 0 or s[1] >= samples[i - 1][1]]
    egood = [e for i, e in enumerate(eng) if i == 0 or e[1] >= eng[i - 1][1]]
    return (_slope(good),
            _slope([(e[0], e[1]) for e in egood]),
            _slope([(e[0], e[2]) for e in egood if e[2] > 0]),
            samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--local", help="validate the VLC method on a local file")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--port", type=int, default=8430)
    ap.add_argument("--capture", type=float, default=15.0)
    ap.add_argument("--vlc-seconds", type=float, default=25.0)
    args = ap.parse_args()

    # --- validate the measurement method itself -------------------------------
    if args.local:
        print(f"method check on local file: {args.local}")
        rate, _, _, samples = measure_vlc(args.local, args.vlc_seconds)
        print(f"  samples: {len(samples)}   measured rate: "
              f"{rate:.3f}x" if rate else "  no samples")
        print("  -> method " + ("OK (local file measures ~1.0x)"
              if rate and abs(rate - 1.0) < 0.15 else "SUSPECT"))
        return 0 if rate and abs(rate - 1.0) < 0.15 else 1

    if not args.file:
        ap.error("--file or --local required")

    # --- 1. timestamp ground truth -------------------------------------------
    log = open("/tmp/rate_engine.log", "wb")
    engine = subprocess.Popen(
        [CLI, "--file", args.file, "--port", str(args.port), "--scale", str(args.scale),
         "--ring-mb", "64", "--prebuffer-mb", "2"],
        stdout=log, stderr=subprocess.STDOUT)
    time.sleep(6)
    cap = "/tmp/rate_capture.ts"
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/stream.ts?session=1", timeout=10) as r, \
                open(cap, "wb") as f:
            t0 = time.time()
            while time.time() - t0 < args.capture:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        pass

    dur = float(subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", cap],
        capture_output=True, text=True).stdout.strip() or 0)
    n, med, frac = video_pts_stats(cap)
    print(f"captured {os.path.getsize(cap)/1e6:.1f} MB in {args.capture:.0f}s of wall time")
    print(f"  video packets: {n}, median PTS spacing {med*1000:.3f} ms, "
          f"{frac*100:.0f}% identical")
    if med > 0:
        print(f"  -> implied frame rate {1/med:.3f} fps")
    print(f"  container duration: {dur:.1f} s "
          f"(content produced in {args.capture:.0f} s of wall time)")

    # --- 2. what VLC actually does -------------------------------------------
    print(f"sampling VLC playback position for {args.vlc_seconds:.0f}s "
          f"(dummy outputs, no window) …")
    status_url = f"http://127.0.0.1:{args.port}/status.json"
    rate, eng_rate, _, samples = measure_vlc(
        f"http://127.0.0.1:{args.port}/stream.ts?session=1",
        args.vlc_seconds, status_url)
    engine.terminate()
    try:
        engine.wait(timeout=8)
    except subprocess.TimeoutExpired:
        engine.kill()
    log.close()

    if samples:
        first, last = samples[0], samples[-1]
        print(f"  VLC position {first[1]:.1f}s -> {last[1]:.1f}s over "
              f"{last[0]-first[0]:.1f}s of wall time")
    if eng_rate is not None:
        print(f"  engine enhancement frontier advanced {eng_rate:.2f}x realtime")
    if rate:
        print(f"  effective playback rate: {rate:.3f}x")
        verdict = "OK (real time)" if abs(rate - 1.0) <= 0.1 else \
                  ("PLAYING FAST" if rate > 1.1 else "PLAYING SLOW")
        print(f"  -> {verdict}")
        print("  (the engine leading the player by a large margin is expected and is not "
              "skipped content: the player buffers it)")
        return 0 if abs(rate - 1.0) <= 0.1 else 1
    print("  -> no VLC samples")
    return 1


if __name__ == "__main__":
    sys.exit(main())
