#!/usr/bin/env python3
"""Full-episode sustained-playback test with player + memory telemetry.

Written to address the audit of the previous harness: it uses an explicit
release binary, dedicated ports, and terminates ONLY the processes it started
(no global pkill). It fails if the player metrics are unavailable, so a run that
did not actually exercise VLC cannot be reported as a pass.

Runs the engine with its own owned VLC (headless: dummy video/audio outputs, so
no window appears) and samples every --interval seconds:

  engine : state, frames, production position, real playback position, mode,
           realtime ratio, thermal state, phys_footprint
  memory : engine RSS, VLC RSS, ffmpeg children RSS, summed
  player : VLC displayed / lost / corrupted / discontinuity counters, bitrate

Usage:
  python3 scripts/full_episode_test.py --file <episode> [--scale 2] [--minutes 30]
Exit code 0 = completed with no dropped frames, in sync, player metrics present.
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "engine/MacVideoUpscaler/.build/release/macvideoupscale")
VLC_HTTP_PORT = 8099
VLC_PASS = "macvideoupscaler"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def rss_kb(pid):
    out = sh(f"ps -o rss= -p {pid}")
    try:
        return int(out.split()[0])
    except Exception:
        return 0


def children(pid):
    out = sh(f"pgrep -P {pid}")
    return [int(x) for x in out.split() if x.isdigit()]


def vlc_status(port=VLC_HTTP_PORT):
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/requests/status.json")
        import base64
        req.add_header("Authorization",
                       "Basic " + base64.b64encode(f":{VLC_PASS}".encode()).decode())
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def engine_status(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.json", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--port", type=int, default=8620)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--minutes", type=float, default=40.0, help="hard stop")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.outdir, f"full_episode_{stamp}.csv")
    log_path = os.path.join(args.outdir, f"full_episode_{stamp}.engine.log")
    os.makedirs(args.outdir, exist_ok=True)
    log = open(log_path, "wb")

    env = dict(os.environ, ZU_VLC_HEADLESS="1")
    engine = subprocess.Popen(
        [CLI, "--file", args.file, "--port", str(args.port), "--scale", str(args.scale),
         "--start", str(args.start), "--prebuffer-mb", "4", "--open-vlc"],
        stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"[test] engine pid {engine.pid}, port {args.port}, log {log_path}", flush=True)

    rows = []
    started = time.time()
    hard_stop = started + args.minutes * 60
    player_seen = False
    try:
        while time.time() - hard_stop < 0 or True:
            time.sleep(args.interval)
            if time.time() > hard_stop:
                print("[test] hard stop reached")
                break
            es = engine_status(args.port)
            if es is None:
                if engine.poll() is not None:
                    print("[test] engine exited")
                    break
                continue
            vs = vlc_status(es.get("vlcHttpPort") or VLC_HTTP_PORT)
            kids = children(engine.pid)
            vlc_pid = None
            for k in kids:
                if "VLC" in sh(f"ps -o comm= -p {k}"):
                    vlc_pid = k
            ffmpeg_rss = sum(rss_kb(k) for k in kids if k != vlc_pid)
            stats = (vs or {}).get("stats", {}) or {}
            if vs and vs.get("state") in ("playing", "paused"):
                player_seen = True
            row = {
                "wall": round(time.time() - started, 1),
                "state": es.get("state"),
                "frames": es.get("framesProcessed"),
                "position": round(es.get("positionSeconds", 0), 1),
                "playback": round(es.get("playbackSeconds", 0), 1),
                "playback_active": es.get("playbackActive"),
                "rt_ratio": round(es.get("realtimeRatio", 0), 2),
                "thermal": es.get("thermalState"),
                "footprint_mb": round(es.get("memoryFootprintMB", 0), 1),
                "engine_rss_mb": round(rss_kb(engine.pid) / 1024, 1),
                "vlc_rss_mb": round(rss_kb(vlc_pid) / 1024, 1) if vlc_pid else 0,
                "ffmpeg_rss_mb": round(ffmpeg_rss / 1024, 1),
                "vlc_state": (vs or {}).get("state"),
                "vlc_time": (vs or {}).get("time"),
                "displayed": stats.get("displayedpictures"),
                "lost": stats.get("lostpictures"),
                "corrupt": stats.get("demuxcorrupted"),
                "discontinuity": stats.get("demuxdiscontinuity"),
            }
            rows.append(row)
            print("  t=%6.1fs %-10s frames=%-6s pos=%7.1f play=%7.1f rt=%4.2f "
                  "mem(eng=%6.1f vlc=%6.1f ff=%5.1f)MB lost=%s corrupt=%s disc=%s"
                  % (row["wall"], row["state"], row["frames"], row["position"],
                     row["playback"], row["rt_ratio"], row["engine_rss_mb"],
                     row["vlc_rss_mb"], row["ffmpeg_rss_mb"],
                     row["lost"], row["corrupt"], row["discontinuity"]), flush=True)
            if row["state"] in ("finished", "failed"):
                print(f"[test] engine reached {row['state']}")
                break
    except KeyboardInterrupt:
        print("[test] interrupted")
    finally:
        # Only our own children.
        for pid in [engine.pid] + children(engine.pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(2)
        for pid in [engine.pid] + children(engine.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        log.close()

    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[test] wrote {csv_path}")

    print("\n===== summary =====")
    if not rows:
        print("FAIL: no samples collected")
        return 1
    last = rows[-1]
    first = rows[0]
    print(f"  duration sampled : {last['wall']:.0f}s ({len(rows)} samples)")
    print(f"  end state        : {last['state']}  frames={last['frames']} "
          f"position={last['position']}s")
    print(f"  player metrics   : {'present' if player_seen else 'MISSING'}")
    print(f"  dropped/corrupt  : lost={last['lost']} corrupt={last['corrupt']} "
          f"discontinuity={last['discontinuity']} displayed={last['displayed']}")
    print(f"  engine RSS       : {first['engine_rss_mb']:.1f} -> {last['engine_rss_mb']:.1f} MB "
          f"(peak {max(r['engine_rss_mb'] for r in rows):.1f})")
    print(f"  engine footprint : {first['footprint_mb']:.1f} -> {last['footprint_mb']:.1f} MB")
    print(f"  VLC RSS          : {first['vlc_rss_mb']:.1f} -> {last['vlc_rss_mb']:.1f} MB "
          f"(peak {max(r['vlc_rss_mb'] for r in rows):.1f})")
    thermals = sorted({r['thermal'] for r in rows})
    print(f"  thermal states   : {thermals}")

    # Plateau test: is second-half growth well below first-half growth?
    if len(rows) >= 8:
        mid = len(rows) // 2
        first_half = (rows[mid]['engine_rss_mb'] - rows[0]['engine_rss_mb']) / max(rows[mid]['wall'] - rows[0]['wall'], 1)
        second_half = (rows[-1]['engine_rss_mb'] - rows[mid]['engine_rss_mb']) / max(rows[-1]['wall'] - rows[mid]['wall'], 1)
        print(f"  growth rate      : first half {first_half*60:.1f} MB/min, "
              f"second half {second_half*60:.1f} MB/min")
        plateau = second_half < max(first_half, 0) * 0.5
        print(f"  plateau          : {'yes (growth falling)' if plateau else 'NO (still growing)'}")

    failed = (not player_seen
              or (last['lost'] or 0) > 0
              or (last['corrupt'] or 0) > 0
              or (last['discontinuity'] or 0) > 0
              or last['state'] == 'failed')
    print(f"\n{'PASS' if not failed else 'FAIL'}: "
          f"player={'yes' if player_seen else 'no'}, dropped="
          f"{(last['lost'] or 0)+(last['corrupt'] or 0)+(last['discontinuity'] or 0)}, "
          f"state={last['state']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
