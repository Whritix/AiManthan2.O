"""Canonical detector evaluation: one weights file, one split, one protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_names(model: YOLO) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/models/detector_best.pt")
    parser.add_argument("--data", default="data/neu-det/neu-det.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35, help="Operating confidence for P/R")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS IoU; mAP matching always uses 0.50:0.95",
    )
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--rect",
        action="store_true",
        help="Use Ultralytics rectangular val batching. Off by default because it inflates mAP on this dataset; deployment uses rect=False.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="artifacts/eval/canonical.json")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    weights = weights.resolve()

    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    data = data.resolve()

    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    model = YOLO(str(weights))
    names = class_names(model)

    common = dict(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        iou=args.iou,
        augment=args.tta,
        rect=args.rect,
        device=args.device,
        workers=0,
        plots=False,
        verbose=False,
        project=str(ROOT / "artifacts/eval/runs"),
        exist_ok=True,
    )

    print("Running mAP protocol at conf=0.001 ...")
    metrics = model.val(
        conf=0.001,
        name=f"canonical_map_{args.split}_{args.imgsz}",
        **common,
    )

    # Ultralytics reports P/R at the max-F1 point, not at a fixed confidence.
    # Derive the requested operating point from the full precision/recall curves.
    px = metrics.box.px
    curve_index = int(round(args.conf * (len(px) - 1)))
    curve_index = max(0, min(curve_index, len(px) - 1))

    per_class_map = {}
    per_class_operating = {}
    for row, class_id in enumerate(metrics.box.ap_class_index):
        class_id = int(class_id)
        per_class_map[names[class_id]] = {
            "map50": float(metrics.box.ap50[row]),
            "map50_95": float(metrics.box.ap[row]),
        }
        per_class_operating[names[class_id]] = {
            "precision": float(metrics.box.p_curve[row, curve_index]),
            "recall": float(metrics.box.r_curve[row, curve_index]),
        }

    macro_precision = sum(v["precision"] for v in per_class_operating.values()) / max(
        1, len(per_class_operating)
    )
    macro_recall = sum(v["recall"] for v in per_class_operating.values()) / max(
        1, len(per_class_operating)
    )

    payload = {
        "created": int(time.time() * 1000),
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "data": str(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "nms_iou": args.iou,
        "tta": bool(args.tta),
        "rect": bool(args.rect),
        "map_protocol": {
            "conf": 0.001,
            "iou_range": "0.50:0.95",
            "note": "mAP is computed from the full precision-recall curve; NMS IoU is separate",
        },
        "metrics": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "per_class": per_class_map,
            "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        },
        "operating_point": {
            "conf": args.conf,
            "precision": macro_precision,
            "recall": macro_recall,
            "average": "macro over classes with ground-truth instances",
            "per_class": per_class_operating,
        },
        "max_f1_point": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "note": "Ultralytics default P/R point; used by the legacy audit table",
        },
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
