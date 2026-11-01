#!/usr/bin/env python3
"""Save comparable per-class YOLO metrics for a candidate checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    weights = (ROOT / args.weights).resolve()
    data = (ROOT / args.data).resolve()
    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    result = model.val(data=str(data), split=args.split, imgsz=args.imgsz, device=args.device,
                       conf=0.001, iou=0.6, rect=True, plots=False, verbose=False,
                       project=str(out.parent), name=out.stem, exist_ok=True)
    box = result.box
    per_class = {}
    for i, name in names.items():
        i = int(i)
        per_class[str(name)] = {
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "map50": float(box.ap50[i]), "map50_95": float(box.ap[i].mean()),
        }
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "dataset_yaml": str(data), "split": args.split, "imgsz": args.imgsz,
        "confidence_for_map": 0.001, "nms_iou": 0.6, "rect": True,
        "overall": {k: float(v) for k, v in result.results_dict.items()},
        "per_class": per_class,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "overall": report["overall"],
                      "weak_classes": {k: v for k, v in per_class.items() if k in {
                          "steel_crazing", "screw_manipulated_front", "screw_thread_side",
                          "screw_thread_top", "transistor_bent_lead", "metal_nut_scratch"}}}, indent=2))


if __name__ == "__main__":
    main()
