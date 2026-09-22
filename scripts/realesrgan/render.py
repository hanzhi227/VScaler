#!/usr/bin/env python3
"""Offline render with a Real-ESRGAN x4 checkpoint, for scoring against a reference.

Both video checkpoints are x4 only, so this fits a 480x270 -> 1920x1080 source
exactly. There is no x2 checkpoint, which is why the 540p setups in this project
cannot use them.

Lossless by default, so what the harness scores is the model's output rather than
the model plus an encode.

Usage:
  render.py --input in.mp4 --output out.mkv [--variant animevideov3|general-x4v3]
            [--device mps] [--frames A:B] [--crf N]
"""
import argparse
import json
import os
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

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
         '-show_entries', 'stream=width,height,r_frame_rate',
         '-show_entries', 'format=duration', '-of', 'json', path],
        capture_output=True, text=True)
    if out.returncode != 0:
        die('ffprobe failed on %s: %s' % (path, out.stderr.strip()))
    d = json.loads(out.stdout)
    st = d['streams'][0]
    num, _, den = st['r_frame_rate'].partition('/')
    return int(st['width']), int(st['height']), float(num) / float(den or 1), \
        float(d['format']['duration'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--variant', default='animevideov3',
                    choices=['animevideov3', 'general-x4v3'])
    ap.add_argument('--frames', default=None, help='A:B, half-open')
    ap.add_argument('--crf', type=int, default=None, help='default: lossless')
    ap.add_argument('--preset', default='medium')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    if not os.path.exists(a.input):
        die('no such input: %s' % a.input)

    import srvgg
    # general-x4v3 is the same class with 32 body convolutions instead of 16, so
    # the variant is a depth *and* a file — `Upscaler` defaults to the anime
    # checkpoint, so passing only num_conv would load the wrong one.
    num_conv, _ = srvgg.VARIANTS[a.variant]
    try:
        up = srvgg.Upscaler(device=a.device,
                            weights=a.weights or srvgg.variant_path(a.variant),
                            num_conv=num_conv)
    except FileNotFoundError as e:
        die('weights not found (%s)\nrun scripts/realesrgan/fetch_weights.py first' % e)

    W, H, fps, dur = ffprobe(a.input)
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
    s = up.scale

    if not a.quiet:
        emit(event='start', input=a.input, output=a.output, scale=s, device=a.device,
             variant=a.variant,
             width=W, height=H, fps=fps, frames_total=count, start=start,
             out_width=W * s, out_height=H * s)

    dec_cmd = [FFMPEG, '-v', 'error']
    if start:
        dec_cmd += ['-ss', '%.6f' % (start / fps)]
    dec_cmd += ['-i', a.input, '-frames:v', str(count),
                '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    enc_cmd = [FFMPEG, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '%dx%d' % (W * s, H * s), '-r', '%f' % fps, '-i', '-', '-an']
    if a.crf is None:
        enc_cmd += ['-c:v', 'libx264rgb', '-qp', '0']
    else:
        enc_cmd += ['-c:v', 'libx264', '-crf', str(a.crf), '-pix_fmt', 'yuv420p']
    enc_cmd += ['-preset', a.preset, a.output]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

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

    if not a.quiet:
        emit(event='done', output=a.output, frames_done=done,
             bytes=os.path.getsize(a.output), lossless=a.crf is None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
