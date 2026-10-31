"""Evaluate the saved unified YOLO detector once on the held-out test split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def count_test_labels(dataset_root: Path, class_count: int) -> tuple[int, list[int], list[int]]:
    label_dir = dataset_root / "labels" / "test"
    files = sorted(label_dir.glob("*.txt"))
    instances = [0] * class_count
    image_counts = [0] * class_count
    for file in files:
        seen: set[int] = set()
        for line in file.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            class_id = int(float(fields[0]))
            if 0 <= class_id < class_count:
                instances[class_id] += 1
                seen.add(class_id)
        for class_id in seen:
            image_counts[class_id] += 1
    return len(files), image_counts, instances


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_epoch_from_csv(path: Path) -> tuple[int | None, dict[str, float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        return None, {}
    def fitness(row: dict[str, str]) -> float:
        return 0.1 * float(row["metrics/mAP50(B)"]) + 0.9 * float(row["metrics/mAP50-95(B)"])
    row = max(rows, key=fitness)
    return int(row["epoch"]) + 1, {
        "precision": float(row["metrics/precision(B)"]),
        "recall": float(row["metrics/recall(B)"]),
        "map50": float(row["metrics/mAP50(B)"]),
        "map50_95": float(row["metrics/mAP50-95(B)"]),
        "fitness": fitness(row),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/runs/unified_yolo11s_640/weights/best.pt")
    parser.add_argument("--data", default="artifacts/datasets/unified_defects/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="artifacts/eval/unified_yolo11s_640_test.json")
    args = parser.parse_args()

    weights = (ROOT / args.weights).resolve()
    data = (ROOT / args.data).resolve()
    output = (ROOT / args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = data.parent

    model = YOLO(str(weights))
    names = model.names
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    names = {int(i): str(name) for i, name in names.items()}

    metrics = model.val(
        data=str(data), split="test", imgsz=args.imgsz, device=args.device,
        conf=0.001, iou=0.6, plots=True, project=str(output.parent),
        name=output.stem.replace("_test", ""), exist_ok=True,
    )
    image_count, images_by_class, instances_by_class = count_test_labels(dataset_root, len(names))
    box = metrics.box
    per_class = {}
    for class_id, name in names.items():
        per_class[name] = {
            "test_images_with_class": images_by_class[class_id],
            "ground_truth_instances": instances_by_class[class_id],
            "precision": float(box.p[class_id]) if class_id < len(box.p) else None,
            "recall": float(box.r[class_id]) if class_id < len(box.r) else None,
            "map50": float(box.ap50[class_id]) if class_id < len(box.ap50) else None,
            "map50_95": float(box.ap[class_id].mean()) if class_id < len(box.ap) else None,
        }

    best_epoch, best_validation = best_epoch_from_csv(weights.parents[1] / "results.csv")
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "dataset_yaml": str(data),
        "split": "test",
        "imgsz": args.imgsz,
        "confidence_for_map": 0.001,
        "nms_iou": 0.6,
        "test_images": image_count,
        "class_count": len(names),
        "best_epoch_by_training_fitness": best_epoch,
        "best_validation": best_validation,
        "test_overall": {key: float(value) for key, value in metrics.results_dict.items()},
        "per_class": per_class,
        "limitations": [
            "This is an internal prepared test split, not fresh production-camera data.",
            "Several non-steel classes have very few held-out images; their class metrics are highly uncertain.",
            "Some source defect examples were reassigned from original anomaly test folders into this prepared split; these results are not canonical source-dataset test scores.",
            "mAP uses confidence 0.001 for ranking and is not the live UI display confidence threshold.",
        ],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(output),
        "weights_sha256": report["weights_sha256"],
        "best_epoch": best_epoch,
        "test_images": image_count,
        "test_overall": report["test_overall"],
        "per_class": per_class,
    }, indent=2))


if __name__ == "__main__":
    main()
