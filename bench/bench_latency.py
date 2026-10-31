#!/usr/bin/env python3
"""
bench_latency.py - batch-1 latency benchmark with warmup + percentiles + power.

Usage
  python bench_latency.py --backend ort --model best.onnx --imgsz 640 --label "laptop/ORT-INT8"
  python bench_latency.py --backend openvino --model ov/best.xml --device CPU --label "155H/OV-INT8"
  python bench_latency.py --backend torch --model runs/seg/train/weights/best.pt --label "4060/FP16"
  python bench_latency.py --backend trt --model best.engine --label "Orin/TRT-FP16"
  # add --power rapl|tegrastats|vcgencmd|nvidia to record watts

Writes results/<slug>.json  (consumed by plot_pareto.py)

Notes
  - batch is always 1: this measures the number that matters for a reject decision.
  - random input is fine for LATENCY; run a separate accuracy pass for quality.
  - first N iterations are warmup (TRT tactic selection, ORT graph opt, cuDNN autotune).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- power sampling
class PowerSampler:
    """Background sampler. Returns mean watts over the measured window."""

    def __init__(self, mode: str, interval: float = 0.2):
        self.mode = mode
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> float | None:
        try:
            if self.mode == "rapl":
                # Intel package-0 energy counter, microjoules (cumulative)
                p = "/sys/class/powercap/intel-rapl:0/energy_uj"
                e0 = int(Path(p).read_text())
                time.sleep(self.interval)
                e1 = int(Path(p).read_text())
                return (e1 - e0) / 1e6 / self.interval
            if self.mode == "tegrastats":
                out = subprocess.run(["tegrastats", "--interval", "200", "--stop"],
                                     capture_output=True, text=True, timeout=3).stdout
                m = re.search(r"VDD_IN (\d+)mW", out)
                return int(m.group(1)) / 1000 if m else None
            if self.mode == "vcgencmd":
                out = subprocess.run(["vcgencmd", "pmic_read_adc"],
                                     capture_output=True, text=True, timeout=3).stdout
                # sum EXT5V rail: "..._V volt(5.0) ..._A current(0.42)"
                total = 0.0
                for line in out.splitlines():
                    if "EXT5V" in line and "current" in line and "_A" in line:
                        total += float(re.search(r"\(([\d.]+)\)", line).group(1)) * 5.0
                return total or None
            if self.mode == "nvidia":
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3).stdout
                return float(out.strip().splitlines()[0])
        except Exception:
            return None
        return None

    def _run(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self.samples.append(v)
            if self.mode != "rapl":
                self._stop.wait(self.interval)

    def start(self):
        self._t.start()
        return self

    def stop(self) -> float | None:
        self._stop.set()
        self._t.join(timeout=2)
        return statistics.fmean(self.samples) if self.samples else None


# ---------------------------------------------------------------- backends
def build_backend(args):
    """Returns (infer_fn, x, sync_fn)."""
    shape = (1, 3, args.imgsz, args.imgsz)

    if args.backend == "torch":
        import torch
        from ultralytics import YOLO

        dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        m = YOLO(args.model)
        net = m.model.to(dev).eval()
        if args.fp16 and dev == "cuda":
            net = net.half()
        x = torch.randn(shape, device=dev)
        if args.fp16 and dev == "cuda":
            x = x.half()

        def infer():
            with torch.inference_mode():
                net(x)

        return infer, x, (torch.cuda.synchronize if dev == "cuda" else (lambda: None))

    if args.backend == "ort":
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if args.cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(args.model, so, providers=providers)
        name = sess.get_inputs()[0].name
        x = np.random.rand(*shape).astype(np.float32)

        def infer():
            sess.run(None, {name: x})

        return infer, x, (lambda: None)

    if args.backend == "openvino":
        import openvino as ov

        core = ov.Core()
        if os.path.isdir(args.cache):
            core.set_property({"CACHE_DIR": args.cache})
        model = core.compile_model(args.model, args.device.upper())
        x = np.random.rand(*shape).astype(np.float32)

        def infer():
            model(x)

        return infer, x, (lambda: None)

    if args.backend == "trt":
        import tensorrt as trt  # noqa: F401
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(args.model, "rb") as f, trt.Runtime(logger) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        ctx = engine.create_execution_context()
        # TensorRT 10 API: tensors by name
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        inp = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        ctx.set_input_shape(inp, shape)
        x = np.random.rand(*shape).astype(np.float32)
        d_in = cuda.mem_alloc(x.nbytes)
        outs = []
        for n in names:
            if n == inp:
                continue
            oshape = tuple(ctx.get_tensor_shape(n))
            buf = np.empty(oshape, dtype=np.float32)
            outs.append((n, buf, cuda.mem_alloc(buf.nbytes)))
        stream = cuda.Stream()

        def infer():
            cuda.memcpy_htod_async(d_in, x, stream)
            ctx.set_tensor_address(inp, int(d_in))
            for n, buf, d in outs:
                ctx.set_tensor_address(n, int(d))
            ctx.execute_async_v3(stream.handle)
            stream.synchronize()

        return infer, x, (lambda: None)

    raise SystemExit(f"unknown backend {args.backend}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["torch", "ort", "openvino", "trt"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="openvino: CPU | GPU.0 | NPU")
    ap.add_argument("--cache", default="ov_cache")
    ap.add_argument("--cuda", action="store_true", help="allow CUDA EP for ort")
    ap.add_argument("--cpu", action="store_true", help="force CPU for torch")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--acc", type=float, default=None, help="accuracy on your val set (F1/AUROC)")
    ap.add_argument("--label", default="")
    ap.add_argument("--power", choices=["rapl", "tegrastats", "vcgencmd", "nvidia"], default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    infer, _, sync = build_backend(args)

    for _ in range(args.warmup):
        infer()
    sync()

    lat: list[float] = []
    sampler = PowerSampler(args.power).start() if args.power else None
    t_all = time.perf_counter()
    for _ in range(args.iters):
        t0 = time.perf_counter()
        infer()
        sync()
        lat.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t_all
    watts = sampler.stop() if sampler else None

    lat.sort()

    def pct(p):
        return lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))]

    result = {
        "label": args.label or f"{args.backend}-{args.imgsz}",
        "device": platform.node(),
        "platform": platform.platform(),
        "backend": args.backend,
        "model": args.model,
        "imgsz": args.imgsz,
        "threads": args.threads,
        "iters": args.iters,
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
        "p99_ms": round(pct(99), 3),
        "min_ms": round(lat[0], 3),
        "max_ms": round(lat[-1], 3),
        "mean_ms": round(statistics.fmean(lat), 3),
        "fps_p50": round(1000 / pct(50), 1),
        "fps_wall": round(args.iters / wall, 1),
        "power_w": round(watts, 2) if watts else None,
        "fps_per_w": round((1000 / pct(50)) / watts, 2) if watts else None,
        "acc": args.acc,
        "jitter_p99_p50": round(pct(99) / pct(50), 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    os.makedirs(args.out, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", result["label"])
    path = Path(args.out) / f"{slug}.json"
    path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\n-> {path}")
    if result["jitter_p99_p50"] > 2.0:
        print("!! p99 is >2x p50 - suspect throttling, page faults, or a noisy neighbour")
    if watts and result["fps_per_w"]:
        print(f"   efficiency: {result['fps_per_w']} FPS/W at {watts:.1f} W")


if __name__ == "__main__":
    main()
