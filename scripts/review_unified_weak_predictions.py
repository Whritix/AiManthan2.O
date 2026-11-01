#!/usr/bin/env python3
"""Draw validation ground truth and baseline predictions for weak classes."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/datasets/unified_defects"
OUT = ROOT / "artifacts/eval/unified_weak_audit"
WEIGHTS = ROOT / "artifacts/runs/unified_yolo11s_640/weights/best.pt"
IDS = {0, 8, 9, 12, 13, 14, 16}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    examples: dict[int, list[Path]] = defaultdict(list)
    for label in sorted((DATA / "labels/val").glob("*.txt")):
        ids = {int(row.split()[0]) for row in label.read_text(encoding="utf-8").splitlines() if row.strip()}
        for cid in IDS & ids:
            if len(examples[cid]) < 4:
                image = DATA / "images/val" / f"{label.stem}.jpg"
                if image.exists():
                    examples[cid].append(image)
    model = YOLO(str(WEIGHTS))
    report = {}
    for cid, paths in sorted(examples.items()):
        sheet = np.full((len(paths) * 300, 600, 3), 248, dtype=np.uint8)
        rows = []
        for index, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            gt = frame.copy()
            pred = frame.copy()
            h, w = frame.shape[:2]
            labels = DATA / "labels/val" / f"{path.stem}.txt"
            expected = [list(map(float, line.split())) for line in labels.read_text(encoding="utf-8").splitlines()
                        if line.strip() and int(line.split()[0]) == cid]
            for _, cx, cy, bw, bh in expected:
                cv2.rectangle(gt, (round((cx-bw/2)*w), round((cy-bh/2)*h)),
                              (round((cx+bw/2)*w), round((cy+bh/2)*h)), (0, 170, 0), 3)
            result = model.predict(frame, imgsz=640, conf=0.25, iou=0.6, device="cpu", verbose=False)[0]
            predicted = []
            for box in result.boxes:
                if int(box.cls.item()) != cid:
                    continue
                x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                predicted.append(round(conf, 3))
                cv2.rectangle(pred, (x0, y0), (x1, y1), (0, 0, 220), 3)
                cv2.putText(pred, f"{conf:.2f}", (x0, max(15, y0-3)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 220), 2)
            for col, image in enumerate((gt, pred)):
                tile = cv2.resize(image, (300, 270), interpolation=cv2.INTER_AREA)
                x, y = col*300, index*300
                sheet[y:y+30, x:x+300] = 255
                cv2.putText(sheet, f"{path.stem[-24:]} {'GT' if col==0 else 'PRED'}", (x+3,y+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,0,0), 1, cv2.LINE_AA)
                sheet[y+30:y+300, x:x+300] = tile
            rows.append({"image": path.name, "ground_truth_boxes": len(expected), "predicted_boxes_at_025": len(predicted),
                         "prediction_confidences": predicted})
        cv2.imwrite(str(OUT / f"class_{cid}_val_gt_vs_pred.jpg"), sheet)
        report[str(cid)] = rows
    (OUT / "baseline_val_prediction_review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({cid: {"images": len(rows), "images_with_prediction": sum(row["predicted_boxes_at_025"]>0 for row in rows)}
                      for cid, rows in report.items()}, indent=2))


if __name__ == "__main__":
    main()
