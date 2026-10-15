#!/usr/bin/env python3
"""
Benchmark AnimeJaNai 2x ONNX models on Apple Silicon (ONNX Runtime).

Stages (run one or more, default = all):
    speed    ms/frame at 480p->960p and 1080p->2160p, CoreML EP (MLComputeUnits=ALL) and CPU EP
    render   2x PNG renders for every quality-test frame
    flicker  temporal amplification ratio on the 12-frame seq_action clip
    video    end-to-end decode+upscale of a clip (no encode), with CPU% / GPU util sampling
    report   write results/animejanai/bench.json

Input contract (verified from model io metadata):
    input  : NCHW float32 (or float16 for some variants), RGB, values in [0,1]
    output : NCHW float, exactly 2x spatial size
    V3 SuperUltraCompact is dynamic in H/W and accepts arbitrary (incl. odd) sizes;
    no padding is required. A pad-to-multiple-of-32 path is implemented anyway and
    is exercised for documentation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AJ_DIR = os.path.join(ROOT, "third_party", "AnimeJaNai")
FRAMES_DIR = os.path.join(ROOT, "test-media", "frames")
SEQ_DIR = os.path.join(FRAMES_DIR, "seq_action")
SEG_DIR = os.path.join(ROOT, "test-media", "segments")
OUT_DIR = os.path.join(ROOT, "results", "animejanai")

ORT_AVAILABLE = ort.get_available_providers()
TENSOR_FLOAT = 1
TENSOR_FLOAT16 = 10


# --------------------------------------------------------------------------- models
def model_tag(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    tag = stem
    if tag.startswith("2x_AnimeJaNai_"):
        tag = tag[len("2x_AnimeJaNai_"):]
    # keep it filesystem friendly
    return tag.replace(" ", "_")


def chosen_model() -> str:
    """General/default 2x SuperUltraCompact (SD/HD V3 model family, fp32)."""
    return os.path.join(
        AJ_DIR, "2x_AnimeJaNai_HD_V3_ModelsOnly",
        "2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx",
    )


def alternative_models() -> list[str]:
    """Other SUC variants / tier-equivalents shipped in the same or newer releases.

    - V3Sharp1 SUC : sharper sibling of the chosen model
    - V2 SUC       : previous-generation SuperUltraCompact (fp16)
    - V3.1 Performance (3.6.0): the tier that replaced 'SuperUltraCompact' naming
    - SD V1beta34 Compact (480p-tuned): per-resolution tuned compact model
    """
    return [
        os.path.join(AJ_DIR, "2x_AnimeJaNai_HD_V3_ModelsOnly",
                      "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact.onnx"),
        os.path.join(AJ_DIR, "2x_AnimeJaNai_V2_SuperUltraCompact_100k.onnx"),
        os.path.join(AJ_DIR, "3.6.0-onnx", "mpv-upscale-2x_animejanai-v3.6.0",
                      "animejanai", "onnx",
                      "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"),
        os.path.join(AJ_DIR, "3.6.0-onnx", "mpv-upscale-2x_animejanai-v3.6.0",
                      "animejanai", "onnx",
                      "2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo.onnx"),
    ]


def model_io(path: str) -> dict:
    """Return dict with input/output names and dtypes (from a zero-IO session)."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    o = s.get_outputs()[0]
    return {
        "path": path,
        "input_name": i.name,
        "input_dtype": i.type,
        "input_shape": list(i.shape),
        "output_name": o.name,
        "output_dtype": o.type,
        "output_shape": list(o.shape),
        "file_bytes": os.path.getsize(path),
    }


# --------------------------------------------------------------------------- sessions
def make_session(path: str, provider: str, compute_units: str = "ALL"):
    """provider: 'cpu' or 'coreml'. Returns (session, binding_info)."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    binding = {"requested_provider": provider}
    if provider == "cpu":
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    elif provider == "coreml":
        opts = {}
        if compute_units:
            opts["MLComputeUnits"] = compute_units
        s = ort.InferenceSession(
            path, so, providers=[("CoreMLExecutionProvider", opts), "CPUExecutionProvider"]
        )
    else:
        raise ValueError(provider)
    binding["session_providers"] = s.get_providers()
    return s, binding


def coreml_partition_info(path: str, compute_units: str = "ALL", size=(480, 640)) -> dict:
    """Capture ORT's C++ stderr while creating/using a CoreML session and parse the
    GetCapability line to prove the model actually binds to CoreML."""
    h, w = size
    fd, tmp = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    old = os.dup(2)
    out = {}
    try:
        with open(tmp, "w") as fh:
            os.dup2(fh.fileno(), 2)
            try:
                so = ort.SessionOptions()
                so.log_severity_level = 1
                s = ort.InferenceSession(
                    path, so,
                    providers=[("CoreMLExecutionProvider", {"MLComputeUnits": compute_units}),
                               "CPUExecutionProvider"],
                )
                x = np.random.rand(1, 3, h, w).astype(np.float32)
                if "float16" in s.get_inputs()[0].type:
                    x = x.astype(np.float16)
                s.run(None, {s.get_inputs()[0].name: x})
            finally:
                os.dup2(old, 2)
        text = open(tmp).read()
    finally:
        try:
            os.close(old)
        except OSError:
            pass
        if os.path.exists(tmp):
            os.unlink(tmp)
    import re
    m = re.search(
        r"number of partitions supported by CoreML: (\d+).*?number of nodes in the graph: (\d+).*?number of nodes supported by CoreML: (\d+)",
        text,
    )
    if m:
        out = {
            "coreml_partitions": int(m.group(1)),
            "graph_nodes": int(m.group(2)),
            "nodes_on_coreml": int(m.group(3)),
        }
    out["bound_coreml"] = bool(out.get("nodes_on_coreml", 0) > 0)
    return out


# --------------------------------------------------------------------------- inference
def frame_to_tensor(rgb: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """HWC uint8/float -> NCHW float in [0,1]."""
    if rgb.dtype == np.uint8:
        x = rgb.astype(np.float32) / 255.0
    else:
        x = rgb.astype(np.float32)
        if x.max() > 1.5:
            x /= 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    return x.astype(dtype)


def tensor_to_frame(y: np.ndarray) -> np.ndarray:
    """NCHW float -> HWC uint8 in [0,255]."""
    y = np.asarray(y)
    if y.ndim == 4:
        y = y[0]
    y = np.transpose(y, (1, 2, 0))
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


_PAD_HISTORY: dict = {}


class Upscaler:
    def __init__(self, path: str, provider: str, compute_units: str = "ALL",
                 pad_multiple: int = 1):
        self.path = path
        self.tag = model_tag(path)
        self.io = model_io(path)
        self.input_dtype = np.float16 if self.io["input_dtype"] == "tensor(float16)" else np.float32
        self.output_dtype = np.float16 if self.io["output_dtype"] == "tensor(float16)" else np.float32
        self.pad_multiple = max(1, int(pad_multiple))
        self.sess, self.binding = make_session(path, provider, compute_units)
        self.provider = provider

    def _pad(self, x: np.ndarray):
        _, _, h, w = x.shape
        m = self.pad_multiple
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        if ph == 0 and pw == 0:
            return x, (h, w)
        xp = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="edge")
        return xp, (h, w)

    def run_tensor(self, x: np.ndarray) -> np.ndarray:
        x, (h, w) = self._pad(x)
        y = self.sess.run(None, {self.io["input_name"]: x})[0]
        y = np.asarray(y)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        if y.ndim == 4:
            y = y[0]
        # crop padded margins (output is 2x input)
        y = y[:, : 2 * h, : 2 * w]
        return y

    def upscale(self, rgb: np.ndarray) -> np.ndarray:
        """rgb HWC uint8 -> 2x HWC uint8."""
        x = frame_to_tensor(rgb, self.input_dtype)
        y = self.run_tensor(x)
        return tensor_to_frame(y)


def time_runs(fn, warmup: int = 3, runs: int = 12):
    for _ in range(warmup):
        fn()
    ts = []
    t_all = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    wall = (time.perf_counter() - t_all) * 1000.0
    return {
        "warmup": warmup,
        "runs": runs,
        "median_ms": statistics.median(ts),
        "mean_ms": float(np.mean(ts)),
        "min_ms": min(ts),
        "max_ms": max(ts),
        "stdev_ms": statistics.pstdev(ts),
        "overall_ms": wall / runs,
    }


# --------------------------------------------------------------------------- inputs
def load_png(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def quality_frames() -> list[tuple[str, str]]:
    """(out_basename, src_path) for every quality frame PNG."""
    out = []
    for p in sorted(glob.glob(os.path.join(FRAMES_DIR, "*.png"))):
        out.append((os.path.basename(p), p))
    for p in sorted(glob.glob(os.path.join(SEQ_DIR, "*.png"))):
        out.append((os.path.basename(p), p))
    return out


def speed_cases() -> dict:
    return {
        "480p->960p": [
            os.path.join(FRAMES_DIR, f)
            for f in ["naruto_t300_src480.png", "naruto_t450_src480.png",
                      "naruto_t600_src480.png", "naruto_t900_src480.png",
                      "naruto_t590_compressed.png"]
        ],
        "1080p->2160p": [
            os.path.join(FRAMES_DIR, f)
            for f in ["naruto_t300_src1080.png", "naruto_t450_src1080.png",
                      "naruto_t600_src1080.png", "naruto_t900_src1080.png",
                      "bbb_t5_src1080.png", "synth_t5_src1080.png"]
        ],
    }


# --------------------------------------------------------------------------- sampling
def gpu_util_percent():
    try:
        out = subprocess.run(
            "ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '\"Device Utilization %\"=[0-9]*'",
            shell=True, capture_output=True, text=True, timeout=5,
        ).stdout
        vals = [int(t.split("=")[1]) for t in out.split() if "=" in t]
        return max(vals) if vals else None
    except Exception:
        return None


def ane_power_state():
    """0 = ANE powered down, 1 = powered up. Used as evidence that CoreML engages the ANE.
    (There is no unprivileged ANE utilization counter; powermetrics needs sudo.)
    GPU util from ioreg is NOT usable on this shared machine: ambient/other-agent load
    keeps Device Utilization pinned near ~60% even when our process is idle."""
    try:
        out = subprocess.run(
            "ioreg -r -c H11ANEIn -d 1 -w 0 | grep -o '\"CurrentPowerState\"=[0-9]*'",
            shell=True, capture_output=True, text=True, timeout=5,
        ).stdout
        vals = [int(t.split("=")[1]) for t in out.split() if "=" in t]
        return max(vals) if vals else None
    except Exception:
        return None


def cpu_percent(pid: int):
    try:
        out = subprocess.run(["ps", "-o", "%cpu=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


class Sampler(threading.Thread):
    def __init__(self, pid: int, interval: float = 0.5):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.cpu, self.gpu, self.ane = [], [], []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            c = cpu_percent(self.pid)
            g = gpu_util_percent()
            a = ane_power_state()
            if c is not None:
                self.cpu.append(c)
            if g is not None:
                self.gpu.append(g)
            if a is not None:
                self.ane.append(a)
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=3)

    def summary(self):
        def agg(v, key):
            return ({"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)}
                    if v else {key: None})
        return {
            "cpu_percent": agg(self.cpu, "cpu"),
            "gpu_util_percent": agg(self.gpu, "gpu"),
            "ane_power_state": agg(self.ane, "ane"),
            "gpu_note": "ioreg GPU util is contaminated by concurrent activity on this shared box",
            "samples": len(self.cpu),
        }


# --------------------------------------------------------------------------- stages
def stage_speed(chosen: str, alts: list[str], out: dict):
    frames = speed_cases()
    res = {}
    models = [(chosen, "chosen")] + [(a, "alternative") for a in alts]
    for mpath, role in models:
        if not os.path.exists(mpath):
            continue
        mt = model_tag(mpath)
        res[mt] = {"role": role, "path": os.path.relpath(mpath, ROOT), "cases": {}}
        try:
            io = model_io(mpath)
            res[mt]["dtype"] = io["input_dtype"]
        except Exception as e:
            res[mt]["error"] = str(e)
            continue
        for prov in ["coreml", "cpu"]:
            for case, paths in frames.items():
                rgb = load_png(paths[0])
                h, w = rgb.shape[:2]
                try:
                    up = Upscaler(mpath, prov, compute_units="ALL")
                    t = time_runs(lambda: up.upscale(rgb), warmup=3, runs=12)
                    out_shape = [int(v) for v in up.run_tensor(frame_to_tensor(rgb, up.input_dtype)).shape]
                    res[mt]["cases"].setdefault(case, {})[prov] = {
                        "input_hw": [h, w], "output_chw": out_shape,
                        "session_providers": up.binding["session_providers"],
                        **t,
                        "ms_per_mp_out": t["median_ms"] / (out_shape[1] * out_shape[2] / 1e6),
                    }
                    print(f"  [speed] {mt:48s} {prov:6s} {case:12s} "
                          f"{t['median_ms']:8.2f} ms  ({out_shape[1]}x{out_shape[2]})")
                except Exception as e:
                    res[mt]["cases"].setdefault(case, {})[prov] = {"error": f"{type(e).__name__}: {e}"}
                    print(f"  [speed] {mt:48s} {prov:6s} {case:12s} ERROR {e}")
    # padding requirement probe on chosen model
    probe = {}
    rgb = load_png(os.path.join(FRAMES_DIR, "naruto_t300_src480.png"))
    up0 = Upscaler(chosen, "cpu", pad_multiple=1)
    up32 = Upscaler(chosen, "cpu", pad_multiple=32)
    a = up0.upscale(rgb).astype(np.int16)
    b = up32.upscale(rgb).astype(np.int16)
    diff = int(np.abs(a - b).max())
    probe["arbitrary_size_output"] = list(up0.run_tensor(frame_to_tensor(rgb, up0.input_dtype)).shape)
    probe["pad_multiple_32_max_abs_diff"] = diff
    # Padding is required when 32-multiple padding changes the output beyond 8-bit
    # rounding noise, so arbitrary-size input is not safe for this model.
    probe["padding_required"] = bool(diff > 1)
    res["_padding_probe"] = probe
    # CoreML compute-unit sweep + ANE power evidence (chosen model only)
    cu_res = {"ane_idle_power_state": ane_power_state(), "configs": {}}
    for case, paths in frames.items():
        rgb = load_png(paths[0])
        for cu in ["ALL", "CPUAndNeuralEngine", "CPUAndGPU", "CPUOnly"]:
            try:
                up = Upscaler(chosen, "coreml", compute_units=cu)
                t = time_runs(lambda: up.upscale(rgb), warmup=3, runs=15)
                cu_res["configs"].setdefault(cu, {})[case] = {
                    "median_ms": t["median_ms"], "min_ms": t["min_ms"], "stdev_ms": t["stdev_ms"]}
                print(f"  [cu] {cu:20s} {case:12s} {t['median_ms']:8.2f} ms")
            except Exception as e:
                cu_res["configs"].setdefault(cu, {})[case] = {"error": str(e)}
        # ANE power state observed while this config runs (worker subprocess)
        cu_res.setdefault("ane_power_state_during", {})
        for cu in ["ALL", "CPUAndNeuralEngine", "CPUOnly"]:
            cu_res["ane_power_state_during"][cu] = ane_under_load(chosen, cu)
        up = Upscaler(chosen, "cpu")
        t = time_runs(lambda: up.upscale(rgb), warmup=3, runs=15)
        cu_res["configs"].setdefault("ORT_CPUExecutionProvider", {})[case] = {
            "median_ms": t["median_ms"], "min_ms": t["min_ms"], "stdev_ms": t["stdev_ms"]}
        print(f"  [cu] {'ORT CPUExecutionProvider':20s} {case:12s} {t['median_ms']:8.2f} ms")
        break  # one representative frame is enough for the mode sweep
    res["_coreml_compute_units"] = cu_res
    out["speed"] = res


def ane_under_load(model: str, compute_units: str, seconds: float = 4.0):
    """Run a CoreML worker subprocess and poll ANE CurrentPowerState. Returns the set of
    observed states while the worker is active."""
    import textwrap
    code = textwrap.dedent(f"""
        import sys, time, numpy as np
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        import bench_animejanai as B
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        up = B.Upscaler({model!r}, "coreml", compute_units={compute_units!r})
        t0 = time.time()
        while time.time() - t0 < {seconds + 1}:
            up.upscale(rgb)
    """)
    w = subprocess.Popen([sys.executable, "-c", code],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        states = set()
        t0 = time.time()
        while time.time() - t0 < seconds:
            s = ane_power_state()
            if s is not None:
                states.add(s)
            time.sleep(0.2)
    finally:
        w.wait(timeout=30)
    return sorted(states)


def stage_render(chosen: str, out: dict):
    mt = model_tag(chosen)
    dest = os.path.join(OUT_DIR, mt)
    os.makedirs(dest, exist_ok=True)
    up = Upscaler(chosen, "coreml", compute_units="ALL")
    meta = []
    for name, src in quality_frames():
        rgb = load_png(src)
        y = up.upscale(rgb)
        Image.fromarray(y).save(os.path.join(dest, name))
        meta.append({"frame": name, "src": os.path.relpath(src, ROOT),
                     "src_hw": list(rgb.shape[:2]), "out_hw": list(y.shape[:2])})
        print(f"  [render] {name:28s} {rgb.shape[1]}x{rgb.shape[0]} -> {y.shape[1]}x{y.shape[0]}  -> {dest}")
    out["render"] = {"model_tag": mt, "dir": os.path.relpath(dest, ROOT),
                     "provider_used": up.binding["session_providers"], "frames": meta}
    # also render the 480p-tuned alternative on the 480p frames for comparison
    sd = os.path.join(AJ_DIR, "3.6.0-onnx", "mpv-upscale-2x_animejanai-v3.6.0",
                      "animejanai", "onnx",
                      "2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo.onnx")
    if os.path.exists(sd):
        sd_tag = "SD_V1beta34_Compact_480p-tuned"
        sd_dest = os.path.join(OUT_DIR, sd_tag)
        os.makedirs(sd_dest, exist_ok=True)
        try:
            uso = Upscaler(sd, "coreml", compute_units="ALL")
            for name, src in quality_frames():
                if "480" not in name and "compressed" not in name:
                    continue
                rgb = load_png(src)
                y = uso.upscale(rgb)
                Image.fromarray(y).save(os.path.join(sd_dest, name))
                print(f"  [render-alt-SD] {name} -> {sd_dest}")
            out["render_alternative_480p_tuned"] = {
                "model_tag": sd_tag, "dir": os.path.relpath(sd_dest, ROOT),
                "provider_used": uso.binding["session_providers"],
            }
        except Exception as e:
            out["render_alternative_480p_tuned"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [render-alt-SD] ERROR {e}")


def _temporal_metric(batch: list[np.ndarray]) -> float:
    """mean over consecutive pairs of mean(|A-B|) on the 0-255 RGB arrays."""
    vals = []
    for a, b in zip(batch[:-1], batch[1:]):
        vals.append(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def stage_flicker(chosen: str, out: dict):
    seq = sorted(glob.glob(os.path.join(SEQ_DIR, "*.png")))
    ins = [load_png(p) for p in seq]
    res = {"frames": [os.path.basename(p) for p in seq], "n_frames": len(ins)}
    for prov in ["coreml", "cpu"]:
        try:
            up = Upscaler(chosen, prov, compute_units="ALL")
            outs = [up.upscale(f) for f in ins]
            mi = _temporal_metric(ins)
            mo = _temporal_metric(outs)
            res[prov] = {
                "input_metric": mi, "output_metric": mo,
                "amplification_ratio": mo / mi,
                "flicker_risk": bool(mo / mi > 1.05),
            }
            print(f"  [flicker] {prov:6s} in={mi:.4f} out={mo:.4f} ratio={mo/mi:.4f}")
        except Exception as e:
            res[prov] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [flicker] {prov} ERROR {e}")
    out["flicker"] = res


def stage_video(chosen: str, seg: str, out: dict, max_frames: int | None = None):
    path = os.path.join(SEG_DIR, seg)
    cap = cv2.VideoCapture(path)
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # decode-only baseline
    t0 = time.perf_counter()
    ndec = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        ndec += 1
    decode_s = time.perf_counter() - t0
    cap.release()

    res = {"segment": seg, "fps_src": fps_src, "frames_src": n_src, "size_hw": [h, w],
           "decode_only_fps": ndec / decode_s, "decode_only_s": decode_s}
    for prov in ["coreml", "cpu"]:
        up = Upscaler(chosen, prov, compute_units="ALL")
        cap = cv2.VideoCapture(path)
        sampler = Sampler(os.getpid(), interval=0.5)
        sampler.start()
        t0 = time.perf_counter()
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            up.upscale(rgb)
            n += 1
            if max_frames and n >= max_frames:
                break
        elapsed = time.perf_counter() - t0
        sampler.stop()
        cap.release()
        res[prov] = {
            "frames": n, "elapsed_s": elapsed, "achieved_fps": n / elapsed,
            "realtime_ratio_vs_29.97": (n / elapsed) / 29.97,
            "session_providers": up.binding["session_providers"],
            **sampler.summary(),
        }
        print(f"  [video] {prov:6s} {n} frames in {elapsed:.2f}s -> {n/elapsed:.2f} fps "
              f"(cpu% {res[prov]['cpu_percent']['mean'] if res[prov]['cpu_percent'].get('mean') else '?'} "
              f"gpu% {res[prov]['gpu_util_percent'].get('mean')})")
    out["video"] = res


def stage_report(chosen: str, out: dict, args):
    os.makedirs(OUT_DIR, exist_ok=True)
    io = model_io(chosen)
    bench = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": {"platform": sys.platform, "cpu": "Apple M5 Pro"},
        "onnxruntime": ort.__version__,
        "available_providers": ORT_AVAILABLE,
        "chosen_model": {"path": os.path.relpath(chosen, ROOT), **io,
                         "tag": model_tag(chosen)},
        "coreml_capability_chosen": {},
        **out,
    }
    try:
        bench["coreml_capability_chosen"] = coreml_partition_info(chosen, "ALL")
    except Exception as e:
        bench["coreml_capability_chosen"] = {"error": str(e)}
    with open(os.path.join(OUT_DIR, "bench.json"), "w") as f:
        json.dump(bench, f, indent=2)
    print(f"  [report] wrote {os.path.join(OUT_DIR, 'bench.json')}")
    return bench


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=["all"],
                    choices=["all", "speed", "render", "flicker", "video", "report"])
    ap.add_argument("--model", default=chosen_model())
    ap.add_argument("--no-alts", action="store_true", help="skip alternative models")
    ap.add_argument("--segment", default="naruto_t580.mkv")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "bench.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stages = args.stages
    if "all" in stages:
        stages = ["speed", "render", "flicker", "video", "report"]
    chosen = args.model
    alts = [] if args.no_alts else [m for m in alternative_models() if os.path.exists(m)]
    print(f"chosen: {chosen}\nalternatives: {len(alts)}")
    accumulator = {}
    # reuse prior bench.json so stages can run independently
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            for k in ("speed", "render", "flicker", "video", "render_alternative_480p_tuned"):
                if k in prev:
                    accumulator[k] = prev[k]
        except Exception:
            pass
    accumulator["_cli_model"] = chosen
    if "speed" in stages:
        stage_speed(chosen, alts, accumulator)
    if "render" in stages:
        stage_render(chosen, accumulator)
    if "flicker" in stages:
        stage_flicker(chosen, accumulator)
    if "video" in stages:
        stage_video(chosen, args.segment, accumulator, args.max_frames)
    if "report" in stages or any(s in stages for s in ("speed", "render", "flicker", "video")):
        stage_report(chosen, accumulator, args)
    print("done")


if __name__ == "__main__":
    main()
