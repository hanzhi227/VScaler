#!/usr/bin/env python3
"""Sustained-playback validation for the MacVideoUpscaler engine + VLC.

Starts the engine, launches VLC against the stream with its HTTP interface
enabled, and samples engine + VLC telemetry into a CSV:

  * engine: state, fps, frames, ring bytes, thermal state, RSS, CPU%
  * GPU utilization (ioreg IOAccelerator, no sudo needed)
  * VLC: state, playback time, *lost_pictures* (real dropped frames),
    displayed_pictures, input bitrate, and the decoded video resolution

At the end it prints a summary (fps range, RSS growth, thermal states,
dropped frames, whether VLC reported the expected output resolution).

Usage: sustained_test.py --file F --mode cnn --port 8098 --seconds 180
"""
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine/MacVideoUpscaler/.build/debug/macvideoupscale"
VLC_BIN = "/Applications/VLC.app/Contents/MacOS/VLC"
HTTP_PORT = 8080
HTTP_PASS = "anime"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def gpu_util():
    out = sh("ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '\"Device Utilization %\"=[0-9]*' | head -1")
    m = re.search(r"=(\d+)", out)
    return int(m.group(1)) if m else None


def vlc_status():
    out = sh(f"curl -s --max-time 2 -u :{HTTP_PASS} http://127.0.0.1:{HTTP_PORT}/requests/status.json")
    if not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def vlc_resolution_from_log(log_path):
    """VLC 3 does not expose resolution in status.json; the -vv log does."""
    try:
        txt = Path(log_path).read_text(errors="ignore")
    except OSError:
        return None
    m = re.findall(r"(\d{3,4}x\d{3,4})", txt)
    if not m:
        return None
    # largest reported resolution is the decoded video size
    def px(s):
        w, h = s.split("x")
        return int(w) * int(h)
    return max(set(m), key=px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--mode", default="cnn")
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--interval", type=float, default=5)
    ap.add_argument("--prebuffer-mb", type=int, default=4)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--no-vlc", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else Path(f"/tmp/sustained_{args.mode}.csv")
    eng_log = Path(f"/tmp/sustained_engine_{args.mode}.log")
    eng_err = Path(f"/tmp/sustained_engine_{args.mode}.err")
    vlc_log = Path(f"/tmp/vlc_sustained_{args.mode}.log")

    # No global cleanup here: pkill would take down every engine on the machine
    # (including a concurrently running test) and the AppleScript quit would close
    # the user's own VLC. Only the processes started below are stopped, in the
    # finally block.
    print(f"[test] engine: {args.file} mode={args.mode} port={args.port}")
    eo, ee = open(eng_log, "w"), open(eng_err, "w")
    eng = subprocess.Popen(
        [str(ENGINE), "--file", args.file, "--mode", args.mode,
         "--port", str(args.port), "--prebuffer-mb", str(args.prebuffer_mb)],
        stdout=eo, stderr=ee)
    time.sleep(6)

    vlc = None
    if not args.no_vlc:
        url = f"http://127.0.0.1:{args.port}/stream.ts?session=1"
        vlc = subprocess.Popen(
            [VLC_BIN, "-I", "macosx", "--extraintf=http",
             f"--http-password={HTTP_PASS}", f"--http-port={HTTP_PORT}",
             "--file-logging", f"--logfile={vlc_log}", "-vv",
             "--network-caching=2000", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[test] vlc launched on {url}")

    rows = []
    t0 = time.time()
    print(f"[test] sampling for {args.seconds:.0f}s …")
    try:
        while time.time() - t0 < args.seconds:
            t = round(time.time() - t0, 1)
            eng_json = {}
            try:
                eng_json = json.loads(eng_log.read_text().strip().splitlines()[-1])
            except Exception:
                pass
            rss = cpu = None
            if eng.poll() is None:
                out = sh(f"ps -o rss=,%cpu= -p {eng.pid}").split()
                if len(out) >= 2:
                    rss, cpu = int(out[0]) / 1024.0, float(out[1])
            st = vlc_status() if vlc else None
            stats = (st or {}).get("stats", {})
            row = {
                "t": t,
                "eng_state": eng_json.get("state"),
                "fps": eng_json.get("processFps"),
                "rt": eng_json.get("realtimeRatio"),
                "frames": eng_json.get("framesProcessed"),
                "buf_mb": round(eng_json.get("bufferBytes", 0) / 1e6, 1),
                "eng_thermal": eng_json.get("thermalState"),
                "rss_mb": round(rss, 1) if rss else None,
                "cpu_pct": cpu,
                "gpu_pct": gpu_util(),
                "vlc_state": (st or {}).get("state"),
                "vlc_time": (st or {}).get("time"),
                "vlc_lost": stats.get("lostpictures"),
                "vlc_disp": stats.get("displayedpictures"),
                "vlc_decoded": stats.get("decodedvideo"),
                "vlc_corrupt": stats.get("demuxcorrupted"),
                "vlc_disc": stats.get("demuxdiscontinuity"),
                "vlc_kbps": round((stats.get("inputbitrate") or 0) * 1000, 0),
                "vlc_res": vlc_resolution_from_log(vlc_log),
            }
            rows.append(row)
            time.sleep(args.interval)
    finally:
        print("\n[test] stopping …")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)
        if vlc and vlc.poll() is None:
            vlc.terminate()
            try:
                vlc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                vlc.kill()
                vlc.wait(timeout=5)
        if eng.poll() is None:
            eng.send_signal(signal.SIGINT)
            try:
                eng.wait(timeout=8)
            except subprocess.TimeoutExpired:
                eng.kill()
        eo.close()
        ee.close()

    # Summary
    def nums(key):
        return [r[key] for r in rows if isinstance(r.get(key), (int, float))]

    print(f"\n===== sustained summary ({args.mode}, {len(rows)} samples, csv: {csv_path}) =====")
    fps = nums("fps")
    if fps:
        print(f"engine fps      : min {min(fps):.1f} / mean {sum(fps)/len(fps):.1f} / max {max(fps):.1f}")
    rt = nums("rt")
    if rt:
        print(f"realtime ratio  : min {min(rt):.2f}× / mean {sum(rt)/len(rt):.2f}×")
    rss = nums("rss_mb")
    if rss:
        print(f"engine RSS      : first {rss[0]:.0f} MB → last {rss[-1]:.0f} MB (peak {max(rss):.0f} MB)")
    cpu = nums("cpu_pct")
    if cpu:
        print(f"engine CPU      : mean {sum(cpu)/len(cpu):.0f}% / peak {max(cpu):.0f}%")
    g = nums("gpu_pct")
    if g:
        print(f"GPU util        : mean {sum(g)/len(g):.0f}% / peak {max(g):.0f}%")
    thermals = sorted({r["eng_thermal"] for r in rows if r.get("eng_thermal")})
    print(f"thermal states  : {', '.join(thermals) if thermals else 'n/a'}")
    lost = nums("vlc_lost")
    if lost:
        print(f"VLC lost frames : {max(lost)}   displayed: {max(nums('vlc_disp') or [0])}   "
              f"corrupt: {max(nums('vlc_corrupt') or [0])}   discontinuities: {max(nums('vlc_disc') or [0])}")
    states = sorted({r["vlc_state"] for r in rows if r.get("vlc_state")})
    print(f"VLC states      : {', '.join(states) if states else 'n/a'}")
    res = [r["vlc_res"] for r in rows if r.get("vlc_res")]
    print(f"VLC video info  : {res[-1] if res else 'n/a'}")


if __name__ == "__main__":
    main()
