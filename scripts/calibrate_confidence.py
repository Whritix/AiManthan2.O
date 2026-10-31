#!/usr/bin/env python3
"""Choose a confidence threshold on val defects and good calibration frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "scratches"]


def read_gt(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        cls, xc, yc, w, h = int(float(values[0])), *map(float, values[1:])
        rows.append((cls, np.array([(xc-w/2)*width, (yc-h/2)*height,
                                    (xc+w/2)*width, (yc+h/2)*height])))
    return rows


def collect(model: YOLO, paths: list[Path], imgsz: int, iou: float, device: str,
            labels_dir: Path | None = None) -> list[dict]:
    predictions = model.predict(source=[str(p) for p in paths], imgsz=imgsz, conf=0.001,
                                iou=iou, rect=False, device=device, verbose=False)
    records = []
    for path, result in zip(paths, predictions):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        gt = read_gt(labels_dir / f"{path.stem}.txt", width, height) if labels_dir else []
        pred = []
        if result.boxes is not None:
            pred = [{"class_id": int(cls), "confidence": float(conf), "box": box.astype(float)}
                    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                              result.boxes.cls.cpu().numpy(),
                                              result.boxes.conf.cpu().numpy())]
        records.append({"image": path.name, "gt": gt, "pred": pred})
    return records


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0)
    lt, rb = np.maximum(box[:2], boxes[:, :2]), np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(0, rb-lt)
    inter = wh[:, 0] * wh[:, 1]
    area_a = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
    area_b = np.maximum(0, boxes[:, 2]-boxes[:, 0]) * np.maximum(0, boxes[:, 3]-boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def evaluate(records: list[dict], threshold: float, n_classes: int, match_iou: float) -> dict:
    gt_total = Counter()
    tp = Counter()
    fp = Counter()
    for record in records:
        gt = record["gt"]
        pred = [p for p in record["pred"] if p["confidence"] >= threshold]
        for cls, _ in gt:
            gt_total[cls] += 1
        used_gt: set[int] = set()
        for p in sorted(pred, key=lambda x: x["confidence"], reverse=True):
            choices = [(float(iou_one(p["box"], np.array([box]))[0]), gi)
                       for gi, (cls, box) in enumerate(gt)
                       if cls == p["class_id"] and gi not in used_gt]
            choices = [pair for pair in choices if pair[0] >= match_iou]
            if choices:
                _, gi = max(choices)
                used_gt.add(gi)
                tp[p["class_id"]] += 1
            else:
                fp[p["class_id"]] += 1
    recalls, precisions, per_class = [], [], {}
    for cls in range(n_classes):
        recall = tp[cls] / gt_total[cls] if gt_total[cls] else 0.0
        precision = tp[cls] / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] else 0.0
        if gt_total[cls]:
            recalls.append(recall)
            precisions.append(precision)
        per_class[CLASS_NAMES[cls]] = {"tp": tp[cls], "fp": fp[cls], "gt": gt_total[cls],
                                       "precision": precision, "recall": recall}
    return {"macro_recall": float(np.mean(recalls)) if recalls else 0.0,
            "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
            "micro_recall": sum(tp.values()) / max(1, sum(gt_total.values())),
            "micro_precision": sum(tp.values()) / max(1, sum(tp.values()) + sum(fp.values())),
            "per_class": per_class, "true_positives": sum(tp.values()),
            "false_positives": sum(fp.values()), "ground_truth_instances": sum(gt_total.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--good-fpr-max", type=float, default=0.10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    weights = weights.resolve()
    val_images = sorted((ROOT / "data/neu-det/images/val").glob("*.jpg"))
    val_labels = ROOT / "data/neu-det/labels/val"
    good_images = sorted((ROOT / "data/neu-good/frames/calib").glob("*.jpg"))
    model = YOLO(str(weights))
    names = model.names
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if names != CLASS_NAMES:
        raise SystemExit(f"Unexpected model class order: {names}")
    print(f"Collecting val predictions: {len(val_images)} images", flush=True)
    val_records = collect(model, val_images, args.imgsz, args.iou, args.device, val_labels)
    print(f"Collecting good calibration predictions: {len(good_images)} images", flush=True)
    good_records = collect(model, good_images, args.imgsz, args.iou, args.device)

    sweep = []
    candidates = np.round(np.arange(0.01, 1.0, 0.005), 3)
    for threshold in candidates:
        defect = evaluate(val_records, float(threshold), len(CLASS_NAMES), args.match_iou)
        good_fires = sum(any(p["confidence"] >= threshold for p in rec["pred"]) for rec in good_records)
        good_rate = good_fires / max(1, len(good_records))
        sweep.append({"threshold": float(threshold), "good_false_alarm_rate": good_rate,
                      "good_fires": good_fires, "val": defect})
    feasible = [row for row in sweep if row["good_false_alarm_rate"] <= args.good_fpr_max]
    if not feasible:
        raise SystemExit("No confidence threshold meets the good-calibration false-alarm cap")
    # Recall is the primary objective; precision and then the higher threshold break ties.
    chosen = max(feasible, key=lambda row: (row["val"]["macro_recall"],
                                            row["val"]["macro_precision"], row["threshold"]))
    payload = {
        "created_at_ms": int(time.time() * 1000), "weights": str(weights),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "imgsz": args.imgsz, "nms_iou": args.iou, "match_iou": args.match_iou,
        "validation_images": len(val_images), "good_calibration_images": len(good_images),
        "good_false_alarm_cap": args.good_fpr_max,
        "selection_rule": "Among thresholds with good-calibration any-detection rate <= cap, maximize macro object recall on validation at same-class IoU >= match_iou; break ties by macro precision then higher threshold.",
        "selected": chosen, "sweep": sweep,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "sweep"}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
