#!/usr/bin/env python3
"""Offline render with RT4KSR, for scoring and for comparison grids.

RT4KSR is frame-independent — no recurrence, no optical flow — so this is a
straight decode -> per-frame model -> encode loop with no chunking and no state
to carry. It is also fast enough to run live (3.3-6.0 ms/frame at these sizes,
see bench.py), so this script exists to *score* the model against a reference,
not because the model needs an offline path.

The default encoder is lossless (`libx264rgb -qp 0`), so what the harness scores
is the model's actual output rather than the model plus an encode. Use --crf for
a small file when the output is only going to be looked at.

Usage:
  render.py --input in.mp4 --output out.mkv --scale 2 [options]

Options:
  --scale {2,3,4}   4 is two chained x2 passes (there is no x4 model)
  --device DEV      mps | cpu | auto   (default auto)
  --weights DIR     directory holding RT4KSR_x2.pth / RT4KSR_x3.pth
  --frames A:B      half-open frame range; default is the whole clip
  --crf N           x264 CRF (default: lossless)
  --preset P        x264 preset (default medium)
  --quiet           suppress progress lines

Progress is written to stdout as one JSON object per line.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

DEFAULT_WEIGHTS_DIR = os.path.expanduser('~/.cache/macvideoupscaler')
FFMPEG = '/opt/homebrew/bin/ffmpeg'
FFPROBE = '/opt/homebrew/bin/ffprobe'


def emit(**kw):
    sys.stdout.write(json.dumps(kw, sort_keys=True) + '\n')
    sys.stdout.flush()


def die(msg, code=1):
    emit(event='error', message=msg)
    sys.stderr.write(msg + '\n')
    sys.exit(code)


def ffprobe(path):
    out = subprocess.run(
        [FFPROBE, '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
         '-show_entries', 'format=duration',
         '-of', 'json', path],
        capture_output=True, text=True)
    if out.returncode != 0:
        die('ffprobe failed on %s: %s' % (path, out.stderr.strip()))
    d = json.loads(out.stdout)
    st = d['streams'][0]
    num, _, den = st['r_frame_rate'].partition('/')
    fps = float(num) / float(den or 1)
    return int(st['width']), int(st['height']), fps, float(d['format']['duration'])


def pick_device(requested):
    import torch
    if requested != 'auto':
        return requested
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--scale', type=int, default=2, choices=[2, 3, 4])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--frames', default=None, help='A:B, half-open')
    ap.add_argument('--crf', type=int, default=None, help='default: lossless')
    ap.add_argument('--preset', default='medium')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    if not os.path.exists(a.input):
        die('no such input: %s' % a.input)
    for tool in (FFMPEG, FFPROBE):
        if not os.path.exists(tool):
            die('missing %s' % tool)

    import rt4ksr
    import torch

    device = pick_device(a.device)
    # Which checkpoints this scale needs: x2 for a 2x or a chained 4x, and the
    # single x3 model for 3x. An explicitly named --weights directory has to
    # actually contain them, rather than silently falling back to the cache.
    needed = [3] if a.scale == 3 else [2]
    models = {}
    for s in needed:
        path = None
        if a.weights:
            path = os.path.join(a.weights, 'RT4KSR_x%d.pth' % s)
            if not os.path.exists(path):
                die('no weights at %s' % path)
        models[s] = path
    try:
        up = rt4ksr.Upscaler(scale=a.scale, device=device, models=models)
    except FileNotFoundError as e:
        die('weights not found (%s)\nrun scripts/rt4ksr/fetch_weights.py first' % e)

    W, H, fps, dur = ffprobe(a.input)
    if W % 2 or H % 2:
        die('RT4KSR needs even dimensions (pixel unshuffle by 2); got %dx%d' % (W, H))

    n_total = int(round(dur * fps))
    start, end = 0, n_total
    if a.frames:
        lo, _, hi = a.frames.partition(':')
        if lo:
            start = int(lo)
        if hi:
            end = int(hi)
    end = max(start, min(end, n_total))
    count = end - start

    if not a.quiet:
        emit(event='start', input=a.input, output=a.output, scale=a.scale,
             device=device, width=W, height=H, fps=fps,
             frames_total=count, start=start,
             out_width=W * a.scale, out_height=H * a.scale)

    # Decode: seek to the start frame, then take exactly `count` frames.
    dec_cmd = [FFMPEG, '-v', 'error']
    if start:
        dec_cmd += ['-ss', '%.6f' % (start / fps)]
    dec_cmd += ['-i', a.input, '-frames:v', str(count),
                '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    enc_cmd = [FFMPEG, '-v', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '%dx%d' % (W * a.scale, H * a.scale), '-r', '%f' % fps, '-i', '-',
               '-an']
    if a.crf is None:
        enc_cmd += ['-c:v', 'libx264rgb', '-qp', '0']      # lossless RGB
    else:
        enc_cmd += ['-c:v', 'libx264', '-crf', str(a.crf), '-pix_fmt', 'yuv420p']
    enc_cmd += ['-preset', a.preset, a.output]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # A cancelled or failed run must not leave two ffmpeg processes behind, and
    # must not leave a half-written output that looks like a finished one.
    def shutdown(*_):
        for p in (dec, enc):
            try:
                p.kill()
            except Exception:
                pass
        try:
            os.unlink(a.output)
        except OSError:
            pass
        sys.exit(130)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    in_bytes = W * H * 3
    done = 0
    try:
        while done < count:
            raw = dec.stdout.read(in_bytes)
            if not raw or len(raw) < in_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
            out = up.upscale_np(frame)
            enc.stdin.write(np.ascontiguousarray(out).tobytes())
            done += 1
            if not a.quiet and (done % 10 == 0 or done == count):
                emit(event='progress', frames_done=done, frames_total=count,
                     fraction=done / count if count else 1.0)
    except BrokenPipeError:
        pass
    finally:
        try:
            enc.stdin.close()
        except Exception:
            pass
        dec.stdout.close()
        enc.wait()
        dec.wait()

    if done < count:
        try:
            os.unlink(a.output)
        except OSError:
            pass
        die('decoder produced %d of %d frames' % (done, count))

    if enc.returncode != 0:
        err = enc.stderr.read().decode('utf-8', 'replace')[-800:]
        try:
            os.unlink(a.output)
        except OSError:
            pass
        die('encoder failed: %s' % err)

    size = os.path.getsize(a.output)
    if not a.quiet:
        emit(event='done', output=a.output, frames_done=done, bytes=size,
             lossless=a.crf is None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
