#!/usr/bin/env python3
"""Mine pitted-surface false positives from clean deployment-size bank crops.

The real-only detector's good-image false alarms are dominated by
``pitted_surface`` on 256x256 deployment frames. This script scans the
individual clean crops in ``data/neu-good/frames/bank`` (not the 200x200
collage files in ``data/neu-good/images/bank``), selects crops whose peak
pitted-surface confidence falls in the borderline band, and writes them as
empty-label training images.

Usage:
    python scripts/mine_pitted_hardneg.py --count 350
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "scratches"]


def source_stem(path: Path) -> str:
    """Map ``crazing_104_good0.jpg`` back to its source image stem."""
    return re.sub(r"_good\d+$", "", path.stem)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/runs/neu_yolo11s_640_realonly/weights/best.pt")
    parser.add_argument("--bank", default="data/neu-good/frames/bank")
    parser.add_argument("--train-list", default="artifacts/eval/crazing_realonly/train_originals.txt")
    parser.add_argument("--out", default="artifacts/eval/crazing_pitted_hardneg")
    parser.add_argument("--count", type=int, default=350)
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--max-conf", type=float, default=0.60)
    parser.add_argument("--high-max-conf", type=float, default=0.85)
    parser.add_argument(
        "--borderline-fraction",
        type=float,
        default=0.60,
        help="Share of the selection taken from the borderline band; the rest come from high-confidence fires.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    weights = (ROOT / args.weights).resolve()
    bank = (ROOT / args.bank).resolve()
    train_list_path = (ROOT / args.train_list).resolve()
    out = (ROOT / args.out).resolve()
    images_out, labels_out = out / "images" / "train", out / "labels" / "train"
    reset_dir(images_out)
    reset_dir(labels_out)

    train_originals = [line.strip() for line in train_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_test = {p.resolve() for split in ("val", "test") for p in (ROOT / "data/neu-det/images" / split).glob("*.jpg")}
    if any(Path(p).resolve() in val_test for p in train_originals):
        raise SystemExit("train list overlaps validation/test; refusing to build dataset")

    # Bank crops are cut from training images, but verify it explicitly. A bank
    # crop whose source image lives in val/test would leak validation pixels.
    val_test_stems = {p.stem for p in val_test}
    bank_paths = sorted(bank.glob("*.jpg"))
    leaked = [p.name for p in bank_paths if source_stem(p) in val_test_stems]
    if leaked:
        raise SystemExit(
            f"{len(leaked)} bank crops come from validation/test sources, e.g. {leaked[:5]}; refusing to build dataset"
        )

    model = YOLO(str(weights))
    pitted_id = next((int(i) for i, name in model.names.items() if name == "pitted_surface"), None)
    if pitted_id is None:
        raise SystemExit(f"model has no pitted_surface class: {model.names}")

    rows = []
    for n, path in enumerate(bank_paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        result = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=0.001,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        scores = (
            [
                float(conf)
                for cls, conf in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())
                if int(cls) == pitted_id
            ]
            if result.boxes is not None
            else []
        )
        peak = max(scores, default=0.0)
        if args.min_conf <= peak <= args.high_max_conf:
            rows.append({"path": str(path), "pitted_peak_conf": peak})
        if n % 200 == 0:
            print(f"scanned {n}/{len(bank_paths)}; candidates={len(rows)}", flush=True)

    (out / "candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    borderline = [r for r in rows if args.min_conf <= r["pitted_peak_conf"] <= args.max_conf]
    high = [r for r in rows if args.max_conf < r["pitted_peak_conf"] <= args.high_max_conf]
    borderline.sort(key=lambda r: (abs(r["pitted_peak_conf"] - 0.35), -r["pitted_peak_conf"]))
    high.sort(key=lambda r: r["pitted_peak_conf"], reverse=True)

    borderline_quota = min(len(borderline), int(round(args.count * args.borderline_fraction)))
    high_quota = min(len(high), args.count - borderline_quota)
    selected = borderline[:borderline_quota] + high[:high_quota]
    if len(selected) < args.count:  # backfill if one tier is short
        used = {r["path"] for r in selected}
        for row in high + borderline:
            if len(selected) >= args.count:
                break
            if row["path"] not in used:
                selected.append(row)
                used.add(row["path"])
    if len(selected) < args.count:
        raise SystemExit(
            f"only {len(selected)} pitted fires in {args.min_conf}-{args.high_max_conf}; need {args.count}"
        )

    negative_paths = []
    for index, row in enumerate(selected):
        src = Path(row["path"])
        name = f"pitted_bank_{index:04d}{src.suffix.lower()}"
        dst = images_out / name
        shutil.copy2(src, dst)
        (labels_out / f"{dst.stem}.txt").write_text("", encoding="utf-8")
        row["copy"] = str(dst)
        negative_paths.append(dst.as_posix())

    # Put negatives first so Ultralytics writes its label cache next to the
    # experiment (labels/train.cache) instead of into data/neu-det/labels/.
    combined = negative_paths + train_originals
    train_path = out / "train.txt"
    train_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
    yaml_path = out / "neu-det-pitted-hardneg.yaml"
    yaml_path.write_text(
        f"path: {(ROOT / 'data/neu-det').as_posix()}\ntrain: {train_path.as_posix()}\n"
        "val: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)),
        encoding="utf-8",
    )

    confs = [row["pitted_peak_conf"] for row in selected]
    sub_bands = {
        "0.25-0.30": sum(1 for c in confs if 0.25 <= c < 0.30),
        "0.30-0.35": sum(1 for c in confs if 0.30 <= c < 0.35),
        "0.35-0.40": sum(1 for c in confs if 0.35 <= c < 0.40),
        "0.40-0.45": sum(1 for c in confs if 0.40 <= c < 0.45),
        "0.45-0.60": sum(1 for c in confs if 0.45 <= c <= 0.60),
        "0.60-0.85": sum(1 for c in confs if 0.60 < c <= 0.85),
    }
    manifest = {
        "base_weights": str(weights),
        "bank": str(bank),
        "bank_images_scanned": len(bank_paths),
        "selection_class": "pitted_surface",
        "selection_confidence_band": [args.min_conf, args.high_max_conf],
        "borderline_band": [args.min_conf, args.max_conf],
        "borderline_fraction": args.borderline_fraction,
        "borderline_candidates": len(borderline),
        "high_conf_candidates": len(high),
        "selected_negatives": len(selected),
        "selected_borderline": sum(1 for c in confs if args.min_conf <= c <= args.max_conf),
        "selected_high_conf": sum(1 for c in confs if c > args.max_conf),
        "selected_conf_min": min(confs),
        "selected_conf_median": sorted(confs)[len(confs) // 2],
        "selected_conf_max": max(confs),
        "selected_by_sub_band": sub_bands,
        "original_train_images": len(train_originals),
        "combined_train_images": len(combined),
        "validation_test_unchanged": True,
        "train_list": str(train_path),
        "yaml": str(yaml_path),
        "selected": selected,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected"}, indent=2))


if __name__ == "__main__":
    main()
