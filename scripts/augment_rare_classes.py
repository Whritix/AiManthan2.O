"""Targeted Rare-Class Augmentation Script for YOLO datasets.

Balances under-represented classes by applying label-safe transformations:
- Horizontal Flip (inverts box x_center)
- Vertical Flip (inverts box y_center)
- Photometric / CLAHE contrast enhancement
- Multi-transform combinations

Works with standard YOLO format: <class_id> <x_center> <y_center> <width> <height>
Requires only OpenCV (cv2) and NumPy.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append((cid, xc, yc, w, h))
    return boxes


def write_yolo_label(label_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for cid, xc, yc, w, h in boxes:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    elif img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img


def apply_gamma(img: np.ndarray, gamma: float = 1.3) -> np.ndarray:
    """Adjust brightness/gamma curve."""
    inv_gamma = 1.0 / max(1e-5, gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def augment_rare_classes(
    data_yaml_or_dir: str,
    target_classes: list[int] | None = None,
    multiplier: int = 2,
    img_dir_rel: str = "images/train",
    lbl_dir_rel: str = "labels/train",
) -> None:
    base_path = Path(data_yaml_or_dir)
    if base_path.is_file() and base_path.suffix in [".yaml", ".yml"]:
        # Parse data.yaml path
        import yaml
        with open(base_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        root = Path(meta.get("path", base_path.parent))
        train_img_dir = root / meta.get("train", img_dir_rel)
    else:
        root = base_path
        train_img_dir = root / img_dir_rel

    train_lbl_dir = root / lbl_dir_rel
    if not train_lbl_dir.exists():
        # Fallback: check sibling directory
        train_lbl_dir = train_img_dir.parent.parent / "labels" / train_img_dir.name
        if not train_lbl_dir.exists():
            train_lbl_dir = train_img_dir.parent / "labels"

    print(f"Dataset root: {root}")
    print(f"Images dir:   {train_img_dir}")
    print(f"Labels dir:   {train_lbl_dir}")

    # Count initial distributions
    label_files = sorted(list(train_lbl_dir.glob("*.txt")))
    initial_counts: Counter[int] = Counter()
    image_has_class: dict[int, list[Path]] = {}

    for lbl_path in label_files:
        boxes = parse_yolo_label(lbl_path)
        classes_in_file = set()
        for cid, _, _, _, _ in boxes:
            initial_counts[cid] += 1
            classes_in_file.add(cid)
        for cid in classes_in_file:
            image_has_class.setdefault(cid, []).append(lbl_path)

    print("\n[Initial Class Distribution in Train Split]")
    for cid in sorted(initial_counts.keys()):
        print(f"  Class {cid}: {initial_counts[cid]} instances across {len(image_has_class.get(cid, []))} images")

    if not initial_counts:
        print("No labels found. Check paths.")
        return

    # Automatically identify rare classes if none provided
    if target_classes is None or len(target_classes) == 0:
        median_count = np.median(list(initial_counts.values()))
        target_classes = [cid for cid, count in initial_counts.items() if count <= median_count]
        print(f"\nAuto-selected rare classes (below/at median {median_count:.1f}): {target_classes}")
    else:
        print(f"\nTarget rare classes specified: {target_classes}")

    # Process augmentation for rare classes
    augmented_count = 0
    generated_instances: Counter[int] = Counter()

    for target_cid in target_classes:
        candidate_lbls = image_has_class.get(target_cid, [])
        print(f"\nAugmenting Class {target_cid} ({len(candidate_lbls)} source images, multiplier={multiplier})...")

        for lbl_path in candidate_lbls:
            # Find matching image
            stem = lbl_path.stem
            # Ignore already augmented files to prevent compounding loops
            if "_aug_" in stem:
                continue

            img_candidates = list(train_img_dir.glob(f"{stem}.*"))
            valid_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
            img_path = next((p for p in img_candidates if p.suffix.lower() in valid_exts), None)
            if not img_path:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            boxes = parse_yolo_label(lbl_path)

            # Strategy 1: Horizontal Flip
            h_boxes = [(cid, 1.0 - xc, yc, w, h) for cid, xc, yc, w, h in boxes]
            h_img = cv2.flip(img, 1)
            h_img_path = train_img_dir / f"{stem}_aug_hflip{img_path.suffix}"
            h_lbl_path = train_lbl_dir / f"{stem}_aug_hflip.txt"
            cv2.imwrite(str(h_img_path), h_img)
            write_yolo_label(h_lbl_path, h_boxes)
            augmented_count += 1
            for cid, _, _, _, _ in h_boxes:
                generated_instances[cid] += 1

            # Strategy 2: CLAHE + Gamma Contrast Boost (boxes unchanged)
            if multiplier >= 2:
                clahe_img = apply_clahe(img)
                c_img_path = train_img_dir / f"{stem}_aug_clahe{img_path.suffix}"
                c_lbl_path = train_lbl_dir / f"{stem}_aug_clahe.txt"
                cv2.imwrite(str(c_img_path), clahe_img)
                write_yolo_label(c_lbl_path, boxes)
                augmented_count += 1
                for cid, _, _, _, _ in boxes:
                    generated_instances[cid] += 1

            # Strategy 3: Vertical Flip + Gamma (for 3x+ multiplier)
            if multiplier >= 3:
                v_boxes = [(cid, xc, 1.0 - yc, w, h) for cid, xc, yc, w, h in boxes]
                v_img = apply_gamma(cv2.flip(img, 0), gamma=1.25)
                v_img_path = train_img_dir / f"{stem}_aug_vflip{img_path.suffix}"
                v_lbl_path = train_lbl_dir / f"{stem}_aug_vflip.txt"
                cv2.imwrite(str(v_img_path), v_img)
                write_yolo_label(v_lbl_path, v_boxes)
                augmented_count += 1
                for cid, _, _, _, _ in v_boxes:
                    generated_instances[cid] += 1

    print("\n" + "=" * 50)
    print(f"SUCCESS: Generated {augmented_count} new synthetic training samples.")
    print("[Post-Augmentation Added Instances]")
    for cid in sorted(generated_instances.keys()):
        print(f"  Class {cid}: +{generated_instances[cid]} instances (New total: {initial_counts[cid] + generated_instances[cid]})")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted Rare-Class Augmentation for YOLO")
    parser.add_argument("--data", default="data/neu-det/neu-det.yaml", help="Path to data.yaml or dataset root")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Specific class IDs to augment (e.g. 0 1)")
    parser.add_argument("--multiplier", type=int, default=2, help="Augmentation intensity (1 to 3 copies per image)")
    parser.add_argument("--img-dir", default="images/train", help="Relative path to images directory")
    parser.add_argument("--lbl-dir", default="labels/train", help="Relative path to labels directory")
    args = parser.parse_args()

    augment_rare_classes(
        data_yaml_or_dir=args.data,
        target_classes=args.classes,
        multiplier=args.multiplier,
        img_dir_rel=args.img_dir,
        lbl_dir_rel=args.lbl_dir,
    )


if __name__ == "__main__":
    main()
