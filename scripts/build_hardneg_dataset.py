#!/usr/bin/env python3
"""Build a hard-negative training list for the 640 real-only detector recipe.

The real-only model has the best crazing ranking but fires on too many clean
steel crops. This script mines the clean crops it fires on, adds them to the
real-only train list as empty-label (background) images, and writes a new
dataset YAML. Validation and test splits are left untouched.

Usage:
    python scripts/build_hardneg_dataset.py \
        --weights artifacts/runs/neu_yolo11s_640_realonly/weights/best.pt \
        --out-dir artifacts/eval/crazing_hardneg
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "data" / "neu-good" / "images" / "bank"
DEFAULT_REAL_TRAIN = ROOT / "artifacts" / "eval" / "crazing_realonly" / "train_originals.txt"
DEFAULT_DATA_ROOT = ROOT / "data" / "neu-det"
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "scratches"]


def scan_bank(model: YOLO, bank_dir: Path, imgsz: int, iou: float, device: str) -> list[dict]:
    rows: list[dict] = []
    paths = sorted(bank_dir.glob("*.jpg"))
    for index, path in enumerate(paths, start=1):
        result = model.predict(
            source=str(path),
            imgsz=imgsz,
            conf=0.001,
            iou=iou,
            device=device,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is not None:
            for cls, conf in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                detections.append((CLASS_NAMES[int(cls)], float(conf)))
        max_conf = max((c for _, c in detections), default=0.0)
        max_class = max(detections, key=lambda item: item[1])[0] if detections else ""
        crazing_max = max((c for name, c in detections if name == "crazing"), default=0.0)
        rows.append(
            {
                "path": str(path),
                "max_conf": max_conf,
                "max_class": max_class,
                "crazing_max": crazing_max,
                "detections_at_025": sum(1 for _, c in detections if c >= 0.25),
                "detections_at_035": sum(1 for _, c in detections if c >= 0.35),
            }
        )
        if index % 200 == 0:
            print(f"scanned {index}/{len(paths)}", flush=True)
    return rows


def select_negatives(rows: list[dict], max_negatives: int, seed: int) -> list[dict]:
    """Prioritise the 0.25-0.35 borderline fires, then high-confidence fires."""
    rng = random.Random(seed)
    borderline = [r for r in rows if 0.25 <= r["max_conf"] < 0.35]
    high = [r for r in rows if r["max_conf"] >= 0.35]
    low = [r for r in rows if r["max_conf"] < 0.25]
    borderline.sort(key=lambda r: (r["crazing_max"], r["max_conf"]), reverse=True)
    high.sort(key=lambda r: (r["crazing_max"], r["max_conf"]), reverse=True)
    rng.shuffle(low)

    # Keep a bias toward crazing fires, but include any-class fires so the
    # detector also learns to suppress non-crazing background texture.
    selected: list[dict] = []
    for group, cap in ((borderline, int(max_negatives * 0.6)), (high, max_negatives)):
        crazing = [r for r in group if r["crazing_max"] >= 0.25]
        other = [r for r in group if r["crazing_max"] < 0.25]
        take_crazing = min(len(crazing), int(cap * 0.6))
        selected.extend(crazing[:take_crazing])
        remaining = cap - take_crazing
        selected.extend(other[:remaining])
        if len(selected) >= max_negatives:
            break

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for row in selected:
        if row["path"] not in seen:
            seen.add(row["path"])
            unique.append(row)
    selected = unique[:max_negatives]

    # Add a small number of easy negatives so the model does not learn that
    # every clean crop is a hard case.
    easy_target = min(50, max(0, max_negatives - len(selected)))
    for row in low:
        if len(selected) >= max_negatives:
            break
        if easy_target <= 0:
            break
        selected.append(row)
        easy_target -= 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/runs/neu_yolo11s_640_realonly/weights/best.pt")
    parser.add_argument("--bank-dir", default=str(DEFAULT_BANK))
    parser.add_argument("--real-train-list", default=str(DEFAULT_REAL_TRAIN))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default="artifacts/eval/crazing_hardneg")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-negatives", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    bank_dir = Path(args.bank_dir)
    if not bank_dir.is_absolute():
        bank_dir = ROOT / bank_dir
    real_train_list = Path(args.real_train_list)
    if not real_train_list.is_absolute():
        real_train_list = ROOT / real_train_list
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    rows = scan_bank(model, bank_dir, args.imgsz, args.iou, args.device)
    selected = select_negatives(rows, args.max_negatives, args.seed)

    real_images = [line.strip() for line in real_train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not real_images:
        raise SystemExit(f"empty real train list: {real_train_list}")

    # Create empty labels next to the bank images. Ultralytics infers the
    # label path by replacing /images/ with /labels/ and changing the suffix.
    for row in selected:
        image_path = Path(row["path"])
        label_path = Path(str(image_path).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("", encoding="utf-8")

    train_list = out_dir / "train_hardneg.txt"
    train_list.write_text("\n".join(real_images + [row["path"] for row in selected]) + "\n", encoding="utf-8")

    yaml_path = out_dir / "neu-det-hardneg.yaml"
    yaml_path.write_text(
        f"path: {data_root.as_posix()}\n"
        f"train: {train_list.as_posix()}\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )

    summary = {
        "weights": str(weights),
        "bank_dir": str(bank_dir),
        "real_train_images": len(real_images),
        "bank_images_scanned": len(rows),
        "selected_negatives": len(selected),
        "selected_with_crazing_fire_025": sum(1 for r in selected if r["crazing_max"] >= 0.25),
        "selected_with_any_fire_025": sum(1 for r in selected if r["max_conf"] >= 0.25),
        "selected_with_any_fire_035": sum(1 for r in selected if r["max_conf"] >= 0.35),
        "selected_borderline_025_035": sum(1 for r in selected if 0.25 <= r["max_conf"] < 0.35),
        "selected_easy_below_025": sum(1 for r in selected if r["max_conf"] < 0.25),
        "train_list": str(train_list),
        "yaml": str(yaml_path),
        "selection": selected,
    }
    (out_dir / "hardneg_selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "selection"}, indent=2))


if __name__ == "__main__":
    main()
