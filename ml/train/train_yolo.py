"""Train the NEU-DET detector with the frozen Indore recipe.

Usage:
    python ml/train/train_yolo.py --model yolo11s.pt --epochs 120
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--data", default="data/neu-det/neu-det.yaml")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="artifacts/runs")
    ap.add_argument("--name", default="neu_yolo11s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate; lower it for fine-tuning.")
    ap.add_argument("--close-mosaic", type=int, default=15, help="Disable mosaic for the last N epochs.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cache", default="disk", help="Dataset caching strategy: 'disk', 'ram', or False")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    if not data.exists():
        raise SystemExit(f"dataset yaml not found: {data}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="SGD",
        lr0=args.lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        patience=35,
        close_mosaic=args.close_mosaic,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.5,
        degrees=5,
        translate=0.10,
        scale=0.30,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        auto_augment=None,
        amp=True,
        cache=False if str(args.cache).lower() in ["false", "none", "0"] else args.cache,
        seed=args.seed,
        deterministic=True,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
