#!/usr/bin/env python3
"""Measure single-thread CPU inference for representative validation images."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/runs/unified_yolo11s_640/weights/best.pt"
VAL = ROOT / "artifacts/datasets/unified_defects/images/val"
OUT = ROOT / "artifacts/eval/unified_finetune_20261101/cpu_latency.json"


def main() -> None:
    torch.set_num_threads(1)
    model = YOLO(str(MODEL))
    paths = []
    for profile in ("steel", "metal_nut", "screw", "transistor", "bottle"):
        paths.extend(sorted(VAL.glob(f"{profile}__*.jpg"))[:2])
        paths.extend(sorted(VAL.glob(f"{profile}__*.png"))[:2])
    paths = paths[:20]
    frames = [cv2.imread(str(path)) for path in paths]
    if not frames or any(frame is None for frame in frames):
        raise RuntimeError("Could not read representative validation images")
    report = {"device": "cpu", "torch_threads": 1, "image_count": len(frames), "sizes": {}}
    for size in (640, 960):
        model.predict(source=frames[0], imgsz=size, device="cpu", conf=0.25, verbose=False)
        wall_ms, infer_ms = [], []
        for frame in frames:
            start = time.perf_counter()
            result = model.predict(source=frame, imgsz=size, device="cpu", conf=0.25, verbose=False)[0]
            wall_ms.append((time.perf_counter() - start) * 1000)
            infer_ms.append(float(result.speed["inference"]))
        report["sizes"][str(size)] = {
            "wall_median_ms": statistics.median(wall_ms),
            "wall_p95_ms": sorted(wall_ms)[int(0.95 * (len(wall_ms) - 1))],
            "inference_median_ms": statistics.median(infer_ms),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
