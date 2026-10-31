#!/usr/bin/env python3
"""Mine pitted-surface false positives from clean deployment-size bank crops."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "scratches"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/runs/neu_yolo11s_640_realonly/weights/best.pt")
    parser.add_argument("--bank", default="data/neu-good/frames/bank")
    parser.add_argument("--train-list", default="artifacts/eval/crazing_realonly/train_originals.txt")
    parser.add_argument("--out", default="artifacts/eval/crazing_pitted_hardneg")
    parser.add_argument("--count", type=int, default=350)
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--max-conf", type=float, default=0.60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    weights = (ROOT / args.weights).resolve()
    bank = (ROOT / args.bank).resolve()
    train_list_path = (ROOT / args.train_list).resolve()
    out = (ROOT / args.out).resolve()
    images_out, labels_out = out / "images" / "train", out / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    train_originals = [line.strip() for line in train_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_test = {p.resolve() for split in ("val", "test") for p in (ROOT / "data/neu-det/images" / split).glob("*.jpg")}
    if any(Path(p).resolve() in val_test for p in train_originals):
        raise SystemExit("train list overlaps validation/test; refusing to build dataset")

    model = YOLO(str(weights))
    pitted_id = next((i for i, name in enumerate(CLASSES) if name == "pitted_surface"), None)
    rows = []
    paths = sorted(bank.glob("*.jpg"))
    for n, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        result = model.predict(source=frame, imgsz=args.imgsz, conf=0.001, iou=args.iou,
                               device=args.device, verbose=False)[0]
        scores = [float(c) for c, s in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())
                  if int(c) == pitted_id] if result.boxes is not None else []
        peak = max(scores, default=0.0)
        if args.min_conf <= peak <= args.max_conf:
            rows.append({"path": str(path), "pitted_peak_conf": peak})
        if n % 200 == 0:
            print(f"scanned {n}/{len(paths)}; selected candidates={len(rows)}", flush=True)

    rows.sort(key=lambda row: (abs(row["pitted_peak_conf"] - 0.35), -row["pitted_peak_conf"]))
    selected = rows[:args.count]
    if len(selected) < args.count:
        raise SystemExit(f"only {len(selected)} pitted fires in requested confidence band; need {args.count}")

    negative_paths = []
    for index, row in enumerate(selected):
        src = Path(row["path"])
        name = f"pitted_bank_{index:04d}{src.suffix.lower()}"
        dst = images_out / name
        shutil.copy2(src, dst)
        (labels_out / f"{dst.stem}.txt").write_text("", encoding="utf-8")
        row["copy"] = str(dst)
        negative_paths.append(str(dst))

    combined = train_originals + negative_paths
    train_path = out / "train.txt"
    train_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
    yaml_path = out / "neu-det-pitted-hardneg.yaml"
    yaml_path.write_text(
        f"path: {(ROOT / 'data/neu-det').as_posix()}\ntrain: {train_path.as_posix()}\n"
        "val: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)), encoding="utf-8")
    manifest = {
        "base_weights": str(weights), "bank": str(bank), "bank_images_scanned": len(paths),
        "selection_class": "pitted_surface", "selection_confidence_band": [args.min_conf, args.max_conf],
        "selected_negatives": len(selected), "original_train_images": len(train_originals),
        "combined_train_images": len(combined), "validation_test_unchanged": True,
        "train_list": str(train_path), "yaml": str(yaml_path), "selected": selected,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected"}, indent=2))


if __name__ == "__main__":
    main()
