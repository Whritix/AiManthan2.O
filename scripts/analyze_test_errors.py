#!/usr/bin/env python3
"""Export image-level detector errors on a labeled YOLO test split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "scratches"]


def xywhn_to_xyxy(values: list[float], width: int, height: int) -> np.ndarray:
    x, y, w, h = values
    return np.array([(x - w / 2) * width, (y - h / 2) * height,
                     (x + w / 2) * width, (y + h / 2) * height], dtype=float)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=float)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(0.0, right_bottom - left_top)
    intersection = wh[:, 0] * wh[:, 1]
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def load_labels(path: Path, width: int, height: int) -> list[dict]:
    objects = []
    if not path.exists():
        return objects
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        objects.append({"class_id": cls, "box": xywhn_to_xyxy(list(map(float, parts[1:])), width, height)})
    return objects


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = np.round(box).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/models/detector_best.pt")
    parser.add_argument("--images", default="data/neu-det/images/test")
    parser.add_argument("--labels", default="data/neu-det/labels/test")
    parser.add_argument("--out", default="artifacts/eval/test_error_analysis")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    parser.add_argument("--match-iou", type=float, default=0.5, help="GT/prediction match IoU")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    resolve = lambda value: (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value)
    weights, image_dir, label_dir, out_dir = map(resolve, (args.weights, args.images, args.labels, args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "error_images"
    vis_dir.mkdir(exist_ok=True)

    model = YOLO(str(weights))
    weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    names_map = model.names
    names = [names_map[i] for i in range(len(names_map))] if isinstance(names_map, dict) else list(names_map)
    if names != DEFAULT_NAMES:
        raise SystemExit(f"Unexpected model class order: {names}")

    paths = sorted(image_dir.glob("*.jpg"))
    predictions = model.predict(source=[str(p) for p in paths], imgsz=args.imgsz,
                                conf=0.001, iou=args.iou, rect=False,
                                device=args.device, verbose=False, stream=False)

    rows = []
    image_summaries = []
    gt_outcomes = Counter()
    prediction_outcomes = Counter()
    matched_by_class = Counter()
    for path, result in zip(paths, predictions):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        gt = load_labels(label_dir / f"{path.stem}.txt", width, height)
        pred = []
        if result.boxes is not None:
            for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                      result.boxes.cls.cpu().numpy(),
                                      result.boxes.conf.cpu().numpy()):
                pred.append({"box": box.astype(float), "class_id": int(cls), "confidence": float(conf)})
        active = [p for p in pred if p["confidence"] >= args.conf]
        used_pred: set[int] = set()
        image_errors = []

        # Greedy score-ordered one-to-one matching among same-class boxes.
        for gidx, target in enumerate(gt):
            candidates = [(iou_one_to_many(target["box"], np.array([p["box"]]))[0], idx)
                          for idx, p in enumerate(active)
                          if p["class_id"] == target["class_id"] and idx not in used_pred]
            candidates = [item for item in candidates if item[0] >= args.match_iou]
            if candidates:
                overlap, pidx = max(candidates)
                used_pred.add(pidx)
                matched_by_class[names[target["class_id"]]] += 1
                rows.append({"image": path.name, "kind": "matched", "ground_truth_class": names[target["class_id"]],
                             "predicted_class": names[active[pidx]["class_id"]], "confidence": round(active[pidx]["confidence"], 6),
                             "iou": round(float(overlap), 6), "ground_truth_xyxy": json.dumps(np.round(target["box"], 2).tolist()),
                             "prediction_xyxy": json.dumps(np.round(active[pidx]["box"], 2).tolist())})
                continue

            overlaps = [(iou_one_to_many(target["box"], np.array([p["box"]]))[0], idx)
                        for idx, p in enumerate(active) if idx not in used_pred]
            same = [(overlap, idx) for overlap, idx in overlaps if active[idx]["class_id"] == target["class_id"]]
            other = [(overlap, idx) for overlap, idx in overlaps if active[idx]["class_id"] != target["class_id"]]
            low_conf = [(iou_one_to_many(target["box"], np.array([p["box"]]))[0], p)
                        for p in pred if p["confidence"] < args.conf and p["class_id"] == target["class_id"]]
            if low_conf and max(x[0] for x in low_conf) >= args.match_iou:
                kind = "below_threshold"
            elif same and max(x[0] for x in same) >= 0.1:
                kind = "poor_localization"
            elif other and max(x[0] for x in other) >= 0.1:
                kind = "class_confusion"
            else:
                kind = "missed"
            gt_outcomes[f"{names[target['class_id']]}:{kind}"] += 1
            error = {"kind": kind, "class": names[target["class_id"]], "box": target["box"]}
            image_errors.append(error)
            rows.append({"image": path.name, "kind": f"ground_truth_{kind}", "ground_truth_class": names[target["class_id"]],
                         "predicted_class": "", "confidence": "", "iou": "",
                         "ground_truth_xyxy": json.dumps(np.round(target["box"], 2).tolist()), "prediction_xyxy": ""})

        for pidx, prediction in enumerate(active):
            if pidx in used_pred:
                continue
            overlaps = [(float(iou_one_to_many(prediction["box"], np.array([target["box"]]))[0]), target)
                        for target in gt]
            best, nearest = max(overlaps, key=lambda item: item[0]) if overlaps else (0.0, None)
            if best >= 0.1 and nearest["class_id"] != prediction["class_id"]:
                kind = "wrong_class_extra"
            elif best >= 0.1:
                kind = "duplicate_same_class"
            else:
                kind = "false_positive"
            prediction_outcomes[f"{names[prediction['class_id']]}:{kind}"] += 1
            image_errors.append({"kind": kind, "class": names[prediction["class_id"]], "box": prediction["box"],
                                 "confidence": prediction["confidence"]})
            rows.append({"image": path.name, "kind": f"prediction_{kind}", "ground_truth_class": "",
                         "predicted_class": names[prediction["class_id"]], "confidence": round(prediction["confidence"], 6),
                         "iou": round(best, 6), "ground_truth_xyxy": "",
                         "prediction_xyxy": json.dumps(np.round(prediction["box"], 2).tolist())})

        if image_errors:
            for target in gt:
                draw_box(image, target["box"], (0, 200, 0), f"GT {names[target['class_id']]}")
            for p in active:
                draw_box(image, p["box"], (0, 0, 255), f"P {names[p['class_id']]} {p['confidence']:.2f}")
            cv2.imwrite(str(vis_dir / path.name), image)
            image_summaries.append({"image": path.name, "errors": [e["kind"] + ":" + e["class"] for e in image_errors]})

    with (out_dir / "errors.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "kind", "ground_truth_class", "predicted_class",
                                                    "confidence", "iou", "ground_truth_xyxy", "prediction_xyxy"])
        writer.writeheader()
        writer.writerows(rows)

    gt_class_counts = Counter()
    for path in paths:
        image = cv2.imread(str(path))
        gt_class_counts.update(names[obj["class_id"]] for obj in load_labels(label_dir / f"{path.stem}.txt", image.shape[1], image.shape[0]))
    summary = {
        "created_at_ms": int(time.time() * 1000), "weights": str(weights), "weights_sha256": weights_sha256,
        "images": str(image_dir),
        "labels": str(label_dir), "split": "test", "n_images": len(paths), "imgsz": args.imgsz,
        "prediction_confidence_floor": 0.001, "operating_confidence": args.conf,
        "nms_iou": args.iou, "match_iou": args.match_iou, "rect": False,
        "ground_truth_instances_per_class": dict(gt_class_counts),
        "ground_truth_outcomes": dict(gt_outcomes), "matched_ground_truth_per_class": dict(matched_by_class),
        "unmatched_prediction_outcomes": dict(prediction_outcomes),
        "images_with_errors": len(image_summaries), "error_image_list": image_summaries,
        "method_note": "Greedy one-to-one same-class match at IoU >= match_iou after filtering predictions at operating_confidence. Ground-truth outcomes and unmatched prediction outcomes are counted separately. These instance-level diagnostics are not the evaluator's PR-curve metrics.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'errors.csv'} and annotated errors to {vis_dir}")


if __name__ == "__main__":
    main()
