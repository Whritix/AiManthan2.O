#!/usr/bin/env python3
"""Compare IoU and center matching on frozen unified validation labels."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import compute_ap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/datasets/unified_defects"
WEIGHTS = ROOT / "artifacts/runs/unified_yolo11s_640/weights/best.pt"
OUT = ROOT / "artifacts/eval/unified_finetune_20261101/matching_diagnostic.json"
PREDICTIONS_OUT = ROOT / "artifacts/eval/unified_finetune_20261101/matching_predictions.json"
IDS = [0, 1, 2, 3, 4, 8, 9, 12, 13, 14, 16]


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return overlap / max(area_a + area_b - overlap, 1e-12)


def score(gts: dict, preds: dict, ids: list[str], cid: int, criterion: str) -> float | None:
    targets = {j: [box for c, box in gts[image] if c == cid] for j, image in enumerate(ids)}
    total = sum(map(len, targets.values()))
    if not total:
        return None
    ranked = sorted(((conf, j, box) for j, image in enumerate(ids) for c, conf, box in preds[image] if c == cid),
                    reverse=True)
    claimed = defaultdict(set)
    tp = []
    for _, j, predicted in ranked:
        eligible = []
        for k, truth in enumerate(targets[j]):
            if k in claimed[j]:
                continue
            overlap = iou(predicted, truth)
            if criterion == "center":
                cx = (predicted[0] + predicted[2]) / 2
                cy = (predicted[1] + predicted[3]) / 2
                valid = truth[0] <= cx <= truth[2] and truth[1] <= cy <= truth[3]
            else:
                valid = overlap >= float(criterion)
            if valid:
                eligible.append((overlap, k))
        if eligible:
            _, k = max(eligible)
            claimed[j].add(k)
            tp.append(1)
        else:
            tp.append(0)
    if not tp:
        return 0.0
    tp = np.cumsum(tp)
    fp = np.arange(1, len(tp) + 1) - tp
    ap, _, _ = compute_ap(tp / total, tp / (tp + fp))
    return float(ap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()
    weights = (ROOT / args.weights).resolve()
    report_out = OUT.with_name(f"{args.tag}_matching_diagnostic.json")
    predictions_out = PREDICTIONS_OUT.with_name(f"{args.tag}_matching_predictions.json")
    paths = sorted(p for p in (DATA / "images/val").glob("*") if p.suffix.lower() in {".jpg", ".png"})
    gts = {}
    for path in paths:
        label = DATA / "labels/val" / f"{path.stem}.txt"
        rows = []
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = map(float, line.split())
            rows.append((int(c), [(cx-w/2), (cy-h/2), (cx+w/2), (cy+h/2)]))
        gts[path.name] = rows
    model = YOLO(str(weights))
    preds = {}
    for begin in range(0, len(paths), 32):
        for result in model.predict([str(p) for p in paths[begin:begin+32]], imgsz=640,
                                    conf=0.001, iou=0.6, device="0", batch=16, verbose=False):
            h, w = result.orig_shape
            rows = []
            for box in result.boxes:
                c = int(box.cls.item())
                if c not in IDS:
                    continue
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                rows.append((c, float(box.conf.item()), [x0/w, y0/h, x1/w, y1/h]))
            preds[Path(result.path).name] = rows
        print(f"predicted {min(begin+32, len(paths))}/{len(paths)}", flush=True)
    images = [p.name for p in paths]
    names = model.names
    per_class = {}
    for cid in IDS:
        positives = [image for image in images if any(c == cid for c, _ in gts[image])]
        per_class[str(names[cid])] = {
            "class_id": cid,
            "n_images": len(positives),
            "n_boxes": sum(c == cid for image in images for c, _ in gts[image]),
            "ap_iou_0_50": score(gts, preds, images, cid, "0.5"),
            "ap_iou_0_30": score(gts, preds, images, cid, "0.3"),
            "ap_center_in_gt": score(gts, preds, images, cid, "center"),
        }
    report = {"weights": str(weights), "split": "val", "image_count": len(images),
              "prediction_confidence": 0.001, "nms_iou": 0.6,
              "center_definition": "Predicted box center inside an unmatched GT box of the same class",
              "note": "Diagnostic AP implementation; compare IoU levels within this report, not its absolute numbers to Ultralytics mAP.",
              "per_class": per_class}
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions_out.write_text(json.dumps({"images": images, "gts": gts, "preds": preds}), encoding="utf-8")
    print(json.dumps(per_class, indent=2))


if __name__ == "__main__":
    main()
