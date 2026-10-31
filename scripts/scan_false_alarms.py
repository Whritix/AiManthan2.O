#!/usr/bin/env python3
"""Measure any-detection rates on held-out novel and good images."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {
    "heldout": ROOT / "data" / "neu-det" / "heldout" / "images",
    "good": ROOT / "data" / "neu-good" / "frames" / "test",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    weights = weights.resolve()
    model = YOLO(str(weights))
    results = {}
    for split, folder in FOLDERS.items():
        files = sorted(folder.glob("*.jpg"))
        fired, detection_count = 0, 0
        for index, path in enumerate(files, start=1):
            image = cv2.imread(str(path))
            if image is None:
                continue
            pred = model.predict(source=image, imgsz=args.imgsz, conf=0.001, iou=args.iou,
                                 device=args.device, verbose=False)[0]
            scores = pred.boxes.conf.cpu().tolist() if pred.boxes is not None else []
            kept = sum(score >= args.conf for score in scores)
            fired += int(kept > 0)
            detection_count += kept
            if index % 100 == 0:
                print(f"{split}: {index}/{len(files)}", flush=True)
        results[split] = {
            "images": len(files), "images_with_detection": fired,
            "detection_rate": fired / len(files) if files else None,
            "detections_at_threshold": detection_count,
        }
    payload = {
        "created_at_ms": int(time.time() * 1000), "weights": str(weights),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "imgsz": args.imgsz, "confidence_threshold": args.conf, "nms_iou": args.iou,
        "results": results,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
