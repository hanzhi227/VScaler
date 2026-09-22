#!/usr/bin/env python3
"""Verify source-stream handling: listing, audio-track selection, time seek.

All headless (no VLC, no windowed player). Three checks:

  1. listing  - status.json reports the file's audio and subtitle streams
  2. audio    - --audio-index N really muxes stream N. Both language tracks are
                compared against the captured stream's audio envelope, on the
                segment where the two source tracks differ most (so the test is
                discriminative even if the opening theme is shared).
  3. seek     - an arbitrary seek target restarts the session AT that time

Usage:
  python3 scripts/check_tracks.py --file "episode.mkv" [--port 8400] [--seek 600]
Exit code 0 = all checks passed.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

# Keep BLAS/FFT backends single-threaded: this is a light analysis job and must
# never saturate the machine.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
CLI = os.path.join(ROOT, "engine/MacVideoUpscaler/.build/release/macvideoupscale")
SR = 8000
WIN_MS = 20
WIN = SR * WIN_MS // 1000


# ---------------------------------------------------------------- audio utils

def envelope(path, stream="0:a:0", start=None, dur=None):
    """20 ms RMS envelope, mono 8 kHz."""
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-map", stream, "-vn", "-sn", "-ac", "1", "-ar", str(SR),
            "-f", "s16le", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    x = np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0
    n = len(x) // WIN
    if n == 0:
        return np.zeros(0, np.float32)
    return np.sqrt((x[:n * WIN].reshape(n, WIN) ** 2).mean(axis=1))


def best_corr(a, b, max_lag_windows=15):
    """Peak Pearson correlation over +-max_lag_windows of alignment."""
    best = (-2.0, 0)
    for lag in range(-max_lag_windows, max_lag_windows + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(a) - lag]
        else:
            x, y = a[:len(a) + lag], b[-lag:]
        n = min(len(x), len(y))
        if n < 25:
            continue
        x, y = x[:n], y[:n]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best[0]:
            best = (r, lag * WIN_MS)
    return best


def probe_tracks(path):
    p = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name,channels:stream_tags=language,title",
         "-of", "json", path], capture_output=True)
    streams = json.loads(p.stdout)["streams"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    subs = [s for s in streams if s["codec_type"] == "subtitle"]
    return audio, subs


# ---------------------------------------------------------------- engine utils

def status(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.json", timeout=5) as r:
        return json.loads(r.read())


def start_engine(port, path, extra, log_path):
    log = open(log_path, "wb")
    p = subprocess.Popen([CLI, "--file", path, "--port", str(port),
                          "--ring-mb", "16", "--prebuffer-mb", "1"] + extra,
                         stdout=log, stderr=subprocess.STDOUT)
    return p, log


def wait_ready(port, timeout=25):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = status(port)
            if s.get("state") in ("processing", "paused", "finished", "failed"):
                return s
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError("engine did not become ready")


def stop_engine(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    log.close()


def capture(port, path, seconds):
    """Pull the live TS for `seconds` of wall time."""
    total = 0
    url = f"http://127.0.0.1:{port}/stream.ts?session=1"
    with urllib.request.urlopen(url, timeout=10) as r, open(path, "wb") as f:
        t0 = time.time()
        while time.time() - t0 < seconds:
            chunk = r.read(256 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    return total


# ---------------------------------------------------------------------- checks

def check_listing(path, port, tmp):
    """status.json must list the source's audio/subtitle streams."""
    extra = ["--subtitles", "--sub-index", "0"]
    proc, log = start_engine(port, path, extra, os.path.join(tmp, "listing.log"))
    try:
        s = wait_ready(port)
        at, st = s.get("audioTracks", []), s.get("subtitleTracks", [])
        print(f"  audio tracks   : {[(t['index'], t['codec'], t.get('language')) for t in at]}")
        print(f"  subtitle tracks: {[(t['index'], t['codec'], t.get('language')) for t in st]}")
        print(f"  subtitles field: {s.get('subtitles')} (enabled={s.get('subtitlesEnabled')})")

        exp_a, exp_s = probe_tracks(path)
        ok = len(at) == len(exp_a) and len(st) == len(exp_s)
        ok = ok and [t["ordinal"] for t in at] == list(range(len(exp_a)))
        ok = ok and [t["codec"] for t in st] == [t["codec_name"] for t in exp_s]
        ok = ok and [t["codec"] for t in at] == [t["codec_name"] for t in exp_a]
        ok = ok and s.get("subtitlesEnabled") is True
        if exp_s:
            ok = ok and str(s.get("subtitles", "")).startswith(exp_s[0]["codec_name"])
        return ok, {"audio": at, "subs": st}
    finally:
        stop_engine(proc, log)


def find_discriminative_segment(path, duration, candidates):
    """Pick the offset whose two audio tracks differ most (music-only segments
    can make the language tracks nearly identical and the test inconclusive)."""
    best = None
    for c in candidates:
        if c + 25 > duration:
            continue
        e0 = envelope(path, "0:a:0", start=c, dur=25)
        e1 = envelope(path, "0:a:1", start=c, dur=25)
        r, _ = best_corr(e0, e1)
        print(f"  offset {c:5.0f}s: eng-vs-jpn envelope correlation {r:+.3f}")
        if best is None or r < best[0]:
            best = (r, c)
    return best  # (correlation, offset)


def check_audio_track(path, port, tmp, offset, want_ordinal):
    extra = ["--audio-index", str(want_ordinal), "--start", f"{offset:.0f}"]
    tag = f"a{want_ordinal}"
    proc, log = start_engine(port, path, extra, os.path.join(tmp, f"cap_{tag}.log"))
    cap = os.path.join(tmp, f"cap_{tag}.ts")
    try:
        s = wait_ready(port)
        if s.get("audioTrackOrdinal") != want_ordinal:
            print(f"  [{tag}] status audioTrackOrdinal={s.get('audioTrackOrdinal')} (wanted {want_ordinal})")
            return None
        n = capture(port, cap, 12)
        print(f"  [{tag}] captured {n/1e6:.1f} MB of TS")
    finally:
        stop_engine(proc, log)

    cap_env = envelope(cap, "0:a:0")
    res = {}
    for i in (0, 1):
        ref = envelope(path, f"0:a:{i}", start=offset, dur=25)
        r, lag = best_corr(cap_env, ref)
        res[i] = (r, lag)
        print(f"  [{tag}] vs source a:{i}: corr {r:+.3f} (lag {lag:+d} ms)")
    return res


def check_seek(path, port, tmp, target):
    extra = ["--script", f"6:seek:{target:.0f}"]
    proc, log = start_engine(port, path, extra, os.path.join(tmp, "seek.log"))
    first = {}
    try:
        wait_ready(port)
        t0 = time.time()
        while time.time() - t0 < 22:
            try:
                s = status(port)
            except Exception:
                time.sleep(0.2)
                continue
            sid = s.get("sessionId")
            if sid not in first and s.get("state") in ("processing", "preparing"):
                first[sid] = s.get("positionSeconds")
            time.sleep(0.2)
    finally:
        stop_engine(proc, log)
    print(f"  first observed position per session: "
          f"{ {k: round(v, 1) for k, v in sorted(first.items())} }")
    if len(first) < 2:
        return False
    newest = max(first)
    pos = first[newest]
    return abs(pos - target) <= 8.0


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--seek", type=float, default=600.0)
    ap.add_argument("--tmp", default="/tmp/upscaler_tracks")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)

    dur = float(subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", args.file],
        capture_output=True, text=True).stdout.strip() or 0)

    print(f"file: {os.path.basename(args.file)}  ({dur:.0f}s)")
    failures = []

    print("\n[1/3] track listing in status.json")
    ok, _ = check_listing(args.file, args.port, args.tmp)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("listing")

    print("\n[2/3] audio-track selection reaches the muxed stream")
    cands = [c for c in (120, 300, 480, 700, 900, 1100) if c + 25 < dur]
    r, offset = find_discriminative_segment(args.file, dur, cands)
    print(f"  using offset {offset:.0f}s (tracks differ most here: r={r:+.3f})")
    res0 = check_audio_track(args.file, args.port, args.tmp, offset, 0)
    res1 = check_audio_track(args.file, args.port, args.tmp, offset, 1)
    ok2 = False
    if res0 and res1:
        ok2 = (res0[0][0] > res0[1][0] + 0.08 and res1[1][0] > res1[0][0] + 0.08)
        print(f"  -> index 0 matches source a:0 by {res0[0][0]-res0[1][0]:+.3f}; "
              f"index 1 matches source a:1 by {res1[1][0]-res1[0][0]:+.3f}")
    print(f"  -> {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        failures.append("audio-selection")

    print(f"\n[3/3] arbitrary seek to {args.seek:.0f}s")
    ok3 = check_seek(args.file, args.port, args.tmp, args.seek)
    print(f"  -> {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        failures.append("seek")

    print("\n" + ("ALL CHECKS PASSED" if not failures
                  else "FAILED: " + ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
