#!/usr/bin/env python3
"""Event-based A/V synchronization check for an MacVideoUpscaler stream.

Why this is not "compare the audio to the audio": that version passed a stream
whose video had been delayed by a full second. Two mistakes caused it — it never
evaluated the video at all, and both a content-order comparison *and* a +-300 ms
lag search are blind to a one-second timestamp shift.

This version places each stream's events on a common content timeline using the
stream's own presentation timestamps, then compares where the video landed with
where the audio landed:

  1. video: per-frame motion signal (mean |diff| of downscaled gray frames),
     each sample stamped with its frame PTS
  2. audio: 20 ms RMS envelope, stamped from the audio stream's start time
  3. each is aligned independently against the reference segment starting at
     --file-start, then  A/V error = video_lag - audio_lag

A delay applied to only one stream therefore shows up directly, and the reported
resolution is the PTS precision (milliseconds), not the 20 ms envelope step.

Usage:
  check_av_sync.py --reference FILE --stream TS --file-start S [--dur 20]
  check_av_sync.py ... --expect-fail     # asserts a measured desync is rejected
Exit code 0 = in sync; with --expect-fail, 0 only when the check measured a real
desync and rejected the stream. A stream too weak to judge fails either way,
because then the desync was never actually detected.
"""
import argparse
import json
import subprocess
import sys

import numpy as np

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
SR = 8000
WIN = SR * 20 // 1000       # 20 ms audio windows
GW, GH = 160, 90            # motion-signal frame size
SEARCH_S = 0.30             # refine within +-0.3 s of the PTS-implied alignment
TOLERANCE_S = 0.100         # one AAC encoder-delay frame is ~21 ms
MIN_CORR = 0.70


def run(cmd):
    return subprocess.run(cmd, capture_output=True)


def ffprobe_json(path, entries, stream="v:0"):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", stream,
         "-show_entries", entries, "-of", "json", path],
        capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return {}


def video_fps(path):
    d = ffprobe_json(path, "stream=avg_frame_rate,r_frame_rate")
    try:
        s = d["streams"][0]
        for key in ("avg_frame_rate", "r_frame_rate"):
            raw = s.get(key, "")
            if "/" in raw:
                n, den = raw.split("/")
                if float(den) > 0:
                    return float(n) / float(den)
            elif raw:
                return float(raw)
    except Exception:
        pass
    return 30.0


def first_pts(path, stream):
    """First packet PTS of `stream` (e.g. 'v:0'/'a:0'), or None."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", stream,
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        v = line.split(",")[0].strip()
        if v and v != "N/A":
            try:
                return float(v)
            except ValueError:
                continue
    return None


def video_pts(path, stream="v:0"):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", stream,
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout
    pts = []
    for line in out.splitlines():
        v = line.split(",")[0].strip()
        if v and v != "N/A":
            try:
                pts.append(float(v))
            except ValueError:
                pass
    return np.array(pts)


def audio_envelope(path, start=None, dur=None, stream="0:a:0"):
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-map", stream, "-vn", "-sn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    x = np.frombuffer(run(cmd).stdout, dtype="<i2").astype(np.float32) / 32768.0
    n = len(x) // WIN
    if n == 0:
        return np.zeros(0, np.float32)
    return np.sqrt((x[:n * WIN].reshape(n, WIN) ** 2).mean(axis=1))


def motion_series(path, start=None, dur=None):
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-map", "0:v:0", "-vf", f"scale={GW}:{GH}", "-pix_fmt", "gray",
            "-f", "rawvideo", "-"]
    raw = run(cmd).stdout
    fsz = GW * GH
    n = len(raw) // fsz
    if n < 3:
        return np.zeros(0, np.float32)
    fr = np.frombuffer(raw[:n * fsz], dtype=np.uint8).reshape(n, fsz).astype(np.int16)
    return np.abs(np.diff(fr, axis=0)).mean(axis=1)


def refine_lag(out_t, out_v, ref_t, ref_v, expected, step):
    """Best lag L (content time = stream time - L) near `expected`."""
    best = (-2.0, expected)
    n_steps = max(int(SEARCH_S / step), 1)
    for k in range(-n_steps, n_steps + 1):
        L = expected + k * step
        ref_sampled = np.interp(out_t - L, ref_t, ref_v, left=np.nan, right=np.nan)
        m = ~np.isnan(ref_sampled)
        if m.sum() < 20:
            continue
        a, b = out_v[m], ref_sampled[m]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best[0]:
            best = (r, L)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--file-start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--expect-fail", action="store_true",
                    help="return 0 only if a real desync is measured and rejected")
    args = ap.parse_args()

    fps = video_fps(args.reference)
    fps_out = video_fps(args.stream)
    v_step = 1.0 / max(fps_out, 1.0)

    v_start = first_pts(args.stream, "v:0")
    a_start = first_pts(args.stream, "a:0")
    if v_start is None or a_start is None:
        print("FAIL: stream is missing video or audio packets (too weak to judge)")
        return 1

    ref_v = motion_series(args.reference, start=args.file_start, dur=args.dur)
    out_v = motion_series(args.stream)
    ref_a = audio_envelope(args.reference, start=args.file_start, dur=args.dur)
    out_a = audio_envelope(args.stream)
    if len(ref_v) < 20 or len(out_v) < 20 or len(ref_a) < 20 or len(out_a) < 20:
        print("FAIL: not enough signal to analyse (too weak to judge)")
        return 1

    # Content timeline: the reference signals are indexed from --file-start.
    # motion[i] describes the transition into frame i+1, so stamp it with that
    # frame's presentation time.
    out_v_pts = video_pts(args.stream)
    if len(out_v_pts) >= len(out_v) + 1:
        out_v_t = out_v_pts[1:len(out_v) + 1]
    else:
        out_v_t = v_start + (np.arange(len(out_v)) + 1) * v_step
    ref_v_t = args.file_start + (np.arange(len(ref_v)) + 1) / max(fps, 1.0)
    out_a_t = a_start + np.arange(len(out_a)) * 0.02
    ref_a_t = args.file_start + np.arange(len(ref_a)) * 0.02

    # expected lag: content = stream_time - L, and stream_time a_start is content
    v_expected = v_start - args.file_start
    a_expected = a_start - args.file_start

    v_corr, v_lag = refine_lag(out_v_t, out_v, ref_v_t, ref_v, v_expected, v_step)
    a_corr, a_lag = refine_lag(out_a_t, out_a, ref_a_t, ref_a, a_expected, 0.02)
    error = v_lag - a_lag

    print(f"reference : {args.reference}")
    print(f"stream    : {args.stream}")
    print(f"  video {fps:.3f} fps ref / {fps_out:.3f} fps stream (PTS step {v_step*1000:.1f} ms)")
    print(f"  first PTS: video {v_start:.3f}s, audio {a_start:.3f}s "
          f"(difference {(v_start-a_start)*1000:+.0f} ms)")
    print("\n-- independent alignment on the content timeline --")
    print(f"  video lag : {v_lag:+.3f}s from stream origin  (corr {v_corr:.3f})")
    print(f"  audio lag : {a_lag:+.3f}s from stream origin  (corr {a_corr:.3f})")
    print(f"  A/V error : {error*1000:+8.1f} ms   (video lag - audio lag)")
    print(f"  resolution: {v_step*1000:.0f} ms (one frame at the stream's rate)")

    strong = v_corr > MIN_CORR and a_corr > MIN_CORR
    ok = strong and abs(error) <= TOLERANCE_S
    if not strong:
        print(f"\nFAIL: alignment too weak to judge "
              f"(video corr {v_corr:.2f}, audio corr {a_corr:.2f})")
    else:
        print(f"\n{'OK: in sync' if ok else 'FAIL: out of sync'} "
              f"(|{error*1000:.0f} ms| vs {TOLERANCE_S*1000:.0f} ms tolerance)")
    if args.expect_fail:
        # Only a rejection for a measured desync satisfies the negative control;
        # "could not judge" would mean the desync was never actually detected.
        return 0 if (strong and not ok) else 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
