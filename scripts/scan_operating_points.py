"""Regenerate the UI operating-point scan for the promoted weights.

Run this after training and canonical evaluation, when the GPU is free:

    python scripts/scan_operating_points.py --weights artifacts/models/detector_best.pt --imgsz 640

It writes artifacts/eval/conf_scan.json and marks test_metrics.json as
operating_scan_status=current, which re-enables the /api/eval/summary scan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "artifacts" / "eval" / "conf_scan.json"
SUMMARY = ROOT / "artifacts" / "eval" / "test_metrics.json"
FOLDERS = {
    "test": ROOT / "data" / "neu-det" / "images" / "test",
    "heldout": ROOT / "data" / "neu-det" / "heldout" / "images",
    "good": ROOT / "data" / "neu-good" / "frames" / "test",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_folder(model: YOLO, folder: Path, imgsz: int, device: str, iou: float) -> list[list[list]]:
    files = sorted(folder.glob("*.jpg"))
    rows: list[list[list]] = []
    for index, path in enumerate(files, start=1):
        image = cv2.imread(str(path))
        if image is None:
            rows.append([])
            continue
        result = model.predict(
            source=image,
            imgsz=imgsz,
            conf=0.001,
            iou=iou,
            device=device,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is not None:
            names = result.names or {}
            for box in result.boxes:
                class_id = int(box.cls.item())
                name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                detections.append([name, float(box.conf.item())])
        rows.append(detections)
        if index % 50 == 0:
            print(f"{folder.name}: {index}/{len(files)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/models/detector_best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    weights = weights.resolve()
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    model = YOLO(str(weights))
    scan = {name: scan_folder(model, folder, args.imgsz, args.device, args.iou) for name, folder in FOLDERS.items()}
    scan["_meta"] = {
        "weights": str(weights.relative_to(ROOT)),
        "weights_sha256": sha256(weights),
        "imgsz": args.imgsz,
        "nms_iou": args.iou,
        "conf": 0.001,
        "created_at_ms": int(time.time() * 1000),
        "counts": {name: len(rows) for name, rows in scan.items() if name != "_meta"},
    }
    SCAN.write_text(json.dumps(scan, indent=2), encoding="utf-8")
    print(f"wrote {SCAN}")

    if SUMMARY.exists():
        metrics = json.loads(SUMMARY.read_text(encoding="utf-8"))
        metrics["operating_scan_status"] = "current"
        metrics["operating_scan_weights_sha256"] = scan["_meta"]["weights_sha256"]
        metrics["operating_scan_imgsz"] = args.imgsz
        metrics["operating_scan_created_at_ms"] = scan["_meta"]["created_at_ms"]
        SUMMARY.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"updated {SUMMARY}: operating_scan_status=current")


if __name__ == "__main__":
    main()
