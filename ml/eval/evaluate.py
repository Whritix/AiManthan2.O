"""Evaluate the trained detector on the locked NEU-DET test split and heldout class."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="artifacts/models/detector_best.pt")
    ap.add_argument("--data", default="data/neu-det/neu-det.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--heldout", default="data/neu-det/heldout/images")
    ap.add_argument("--good", default="data/neu-good/frames/test")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    names = model.names

    print(f"validating {args.weights} on split={args.split}")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=0.001,
        iou=0.6,
        plots=False,
        verbose=True,
    )
    per_class = {}
    for i, name in enumerate(names.values() if isinstance(names, dict) else names):
        try:
            per_class[str(name)] = {
                "precision": float(metrics.box.p[i]),
                "recall": float(metrics.box.r[i]),
                "map50": float(metrics.box.ap50[i]),
                "map50_95": float(metrics.box.ap[i]),
                "n_images": int(metrics.box.np[i]) if hasattr(metrics.box, "np") else None,
            }
        except Exception:
            pass
    payload = {
        "weights": str(args.weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "n_images": int(getattr(metrics, "nt", 0) or 0),
        },
        "per_class": per_class,
        "speed_ms": dict(metrics.speed),
        "created_at_ms": int(time.time() * 1000),
    }

    # Held-out class: no vocabulary, so any confident box is a false positive
    # that should be caught by the novelty gate (not by the detector).
    def scan(folder: str, limit: int | None = None) -> dict:
        files = sorted(Path(folder).glob("*.jpg"))
        if limit:
            files = files[:limit]
        detected = 0
        boxes = 0
        by_class: dict[str, int] = {}
        confs: list[float] = []
        for path in files:
            img = cv2.imread(str(path))
            if img is None:
                continue
            result = model.predict(
                source=img,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            n = 0 if result.boxes is None else len(result.boxes)
            if n:
                detected += 1
                boxes += n
                for row in result.boxes:
                    cid = int(row.cls.item())
                    cls = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
                    by_class[cls] = by_class.get(cls, 0) + 1
                    confs.append(float(row.conf.item()))
        return {
            "folder": folder,
            "n_images": len(files),
            "images_with_detection": detected,
            "detection_rate": detected / max(1, len(files)),
            "boxes": boxes,
            "by_class": by_class,
            "conf_max": max(confs) if confs else None,
            "conf_median": sorted(confs)[len(confs) // 2] if confs else None,
        }

    if Path(args.heldout).exists():
        payload["heldout_scan"] = scan(args.heldout)
        print("heldout:", json.dumps(payload["heldout_scan"], indent=2))
    if Path(args.good).exists():
        payload["good_scan"] = scan(args.good)
        print("good:", json.dumps(payload["good_scan"], indent=2))

    path = out / f"{args.split}_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
