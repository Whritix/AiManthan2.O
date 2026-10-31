"""Prepare a unified YOLO detection dataset for the five current profiles.

Existing steel boxes are remapped to the canonical taxonomy. MVTec defect
masks are resized to the packaged profile crops and converted to component
bounding boxes. MVTec defect examples are newly split by class; this is a
project-specific split, not the canonical MVTec test protocol.

Run from the project root: python scripts/prepare_unified_defect_dataset.py
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/datasets/unified_defects"
MVTEC = ROOT.parent / "mvtec_anomaly_detection"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
STEEL_SPLITS = {"train": "train", "val": "val", "test": "test"}
PROFILE_GOOD_DIRS = {
    "steel": {"train": "bank", "val": "calib", "test": "test"},
    "metal_nut": {"train": "bank", "val": "calib", "test": "test_good"},
    "screw": {"train": "bank", "val": "calib", "test": "test_good"},
    "transistor": {"train": "bank", "val": "calib", "test": "test_good"},
    "bottle": {"train": "bank", "val": "calib", "test": "test_good"},
}


def all_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def write_pair(image: Path, profile: str, split: str, label_lines: list[str],
               summary: dict, *, source_key: str) -> None:
    image_out = OUT / "images" / split / f"{source_key}{image.suffix.lower()}"
    label_out = OUT / "labels" / split / f"{source_key}.txt"
    if image_out.exists() or label_out.exists():
        raise FileExistsError(f"output collision while preparing {source_key}")
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_out)
    label_out.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    summary["images"][split] += 1
    summary["boxes"][split] += len(label_lines)
    summary["by_profile"][profile][split] += 1
    summary["source_files"].append({"split": split, "source": str(image), "output": str(image_out),
                                   "label_count": len(label_lines)})


def mask_to_yolo_boxes(mask: np.ndarray, image_width: int, image_height: int,
                       class_id: int) -> list[str]:
    if mask.shape[:2] != (image_height, image_width):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[str] = []
    for component in range(1, component_count):
        x, y, width, height, area = (int(v) for v in stats[component])
        if area < 4 or width < 1 or height < 1:
            continue
        xc = (x + width / 2) / image_width
        yc = (y + height / 2) / image_height
        boxes.append(f"{class_id} {xc:.6f} {yc:.6f} {width / image_width:.6f} {height / image_height:.6f}")
    if not boxes and binary.any():
        ys, xs = np.where(binary > 0)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        width, height = x1 - x0, y1 - y0
        boxes.append(f"{class_id} {(x0 + width / 2) / image_width:.6f} {(y0 + height / 2) / image_height:.6f} {width / image_width:.6f} {height / image_height:.6f}")
    return boxes


def split_defect_images(images: list[Path], profile: str, defect_class: str) -> dict[str, list[Path]]:
    ordered = list(images)
    random.Random(f"{SEED}:{profile}:{defect_class}").shuffle(ordered)
    count = len(ordered)
    if count < 3:
        raise ValueError(f"Need at least 3 defect examples for {profile}/{defect_class}; found {count}")
    train_count = max(1, round(count * 0.70))
    val_count = max(1, round(count * 0.15))
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1
    return {
        "train": ordered[:train_count],
        "val": ordered[train_count:train_count + val_count],
        "test": ordered[train_count + val_count:],
    }


def preflight(taxonomy: dict, splits: dict, class_id: dict[tuple[str, str], int]) -> None:
    steel_names = list(splits["trained_classes"])
    for split in STEEL_SPLITS:
        image_root = ROOT / "data/neu-det/images" / split
        label_root = ROOT / "data/neu-det/labels" / split
        paths = all_images(image_root)
        if not paths:
            raise FileNotFoundError(f"steel image split is empty: {image_root}")
        for image in paths:
            label = label_root / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"steel label missing: {label}")
            for row in label.read_text(encoding="utf-8").splitlines():
                parts = row.split()
                if len(parts) != 5 or not parts[0].isdigit():
                    raise ValueError(f"malformed steel label row in {label}: {row}")
                if int(parts[0]) >= len(steel_names) or ("steel", steel_names[int(parts[0])]) not in class_id:
                    raise ValueError(f"unmapped steel label ID in {label}: {row}")

    for profile in taxonomy["profiles"]:
        profile_root = ROOT / "data/profiles" / profile
        for folder in PROFILE_GOOD_DIRS[profile].values():
            good_dir = ROOT / "data/neu-good/frames" / folder if profile == "steel" else profile_root / folder
            if not all_images(good_dir):
                raise FileNotFoundError(f"good-image source is empty: {good_dir}")
        if profile == "steel":
            continue
        defect_dir = profile_root / "test_defect"
        paths_by_class: dict[str, list[Path]] = defaultdict(list)
        for image in all_images(defect_dir):
            defect_class, _, index_text = image.stem.rpartition("__")
            if not defect_class or not index_text.isdigit() or (profile, defect_class) not in class_id:
                raise ValueError(f"unrecognized defect source/class: {image}")
            raw_image = MVTEC / profile / "test" / defect_class / f"{index_text}.png"
            mask_path = MVTEC / profile / "ground_truth" / defect_class / f"{index_text}_mask.png"
            if not raw_image.exists() or not mask_path.exists():
                raise FileNotFoundError(f"missing MVTec image/mask for {profile}/{defect_class}/{index_text}")
            frame = cv2.imread(str(image))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if frame is None or mask is None or not np.any(mask):
                raise ValueError(f"empty or unreadable profile image/mask: {image} / {mask_path}")
            paths_by_class[defect_class].append(image)
        for defect_class, images in paths_by_class.items():
            if len(images) < 3:
                raise ValueError(f"not enough images to split {profile}/{defect_class}: {len(images)}")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"Output already exists and is non-empty: {OUT}; move it aside before regenerating.")
    taxonomy = json.loads((ROOT / "data/defect_taxonomy.json").read_text(encoding="utf-8"))
    profiles = json.loads((ROOT / "data/profiles/manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "data/neu-det/splits.json").read_text(encoding="utf-8"))
    class_id = {(entry["profile"], entry["source_class"]): int(entry["id"])
                for entry in taxonomy["classes"]}
    class_names = {int(entry["id"]): entry["name"] for entry in taxonomy["classes"]}
    preflight(taxonomy, splits, class_id)
    summary = {
        "taxonomy": str(ROOT / "data/defect_taxonomy.json"), "seed": SEED,
        "split_policy": "Steel uses its frozen NEU-DET train/val/test split. Other profiles use bank/calib/test_good for good images; defect examples are stratified per profile/class at approximately 70/15/15.",
        "mvtec_defect_split_note": "MVTec defect samples originate from its anomaly benchmark test folders and are re-partitioned for this unified prototype. Results on this dataset must not be reported as canonical MVTec test results.",
        "images": Counter(), "boxes": Counter(), "by_profile": defaultdict(Counter),
        "by_class": defaultdict(Counter), "source_files": [],
    }

    # Preserve the existing steel detector's frozen split and box annotations.
    steel_names = list(splits["trained_classes"])
    for split in STEEL_SPLITS:
        image_root = ROOT / "data/neu-det/images" / split
        label_root = ROOT / "data/neu-det/labels" / split
        image_paths = all_images(image_root)
        for image in image_paths:
            old_label = label_root / f"{image.stem}.txt"
            if not old_label.exists():
                raise FileNotFoundError(f"steel label missing: {old_label}")
            lines: list[str] = []
            for row in old_label.read_text(encoding="utf-8").splitlines():
                parts = row.split()
                if len(parts) != 5:
                    raise ValueError(f"unexpected steel YOLO row in {old_label}: {row}")
                old_id = int(parts[0])
                source_class = steel_names[old_id]
                new_id = class_id[("steel", source_class)]
                lines.append(" ".join([str(new_id), *parts[1:]]))
                summary["by_class"][class_names[new_id]][split] += 1
            write_pair(image, "steel", split, lines, summary, source_key=f"steel__{image.stem}")

    # Add clean images as negatives and convert MVTec pixel masks into boxes.
    for profile in taxonomy["profiles"]:
        profile_root = ROOT / "data/profiles" / profile
        for split, folder_name in PROFILE_GOOD_DIRS[profile].items():
            good_dir = (ROOT / "data/neu-good/frames" / folder_name) if profile == "steel" else (profile_root / folder_name)
            for image in all_images(good_dir):
                write_pair(image, profile, split, [], summary,
                           source_key=f"{profile}__good__{folder_name}__{image.stem}")

        if profile == "steel":
            continue
        if not MVTEC.exists():
            raise FileNotFoundError(f"Raw MVTec dataset not found: {MVTEC}")
        defect_dir = profile_root / "test_defect"
        if not defect_dir.exists():
            raise FileNotFoundError(f"Profile defect images missing: {defect_dir}")
        paths_by_class: dict[str, list[Path]] = defaultdict(list)
        for image in all_images(defect_dir):
            class_name, _, index_text = image.stem.rpartition("__")
            if not class_name or not index_text.isdigit():
                raise ValueError(f"unrecognized MVTec profile filename: {image.name}")
            paths_by_class[class_name].append(image)

        for defect_class, images in sorted(paths_by_class.items()):
            new_class_id = class_id[(profile, defect_class)]
            class_splits = split_defect_images(images, profile, defect_class)
            for split, class_images in class_splits.items():
                for image in class_images:
                    raw_index = image.stem.rpartition("__")[2]
                    raw_image = MVTEC / profile / "test" / defect_class / f"{raw_index}.png"
                    mask_path = MVTEC / profile / "ground_truth" / defect_class / f"{raw_index}_mask.png"
                    if not raw_image.exists() or not mask_path.exists():
                        raise FileNotFoundError(f"missing MVTec source or mask for {profile}/{defect_class}/{raw_index}")
                    frame = cv2.imread(str(image))
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if frame is None or mask is None:
                        raise ValueError(f"could not decode image/mask for {image}")
                    height, width = frame.shape[:2]
                    box_lines = mask_to_yolo_boxes(mask, width, height, new_class_id)
                    if not box_lines:
                        raise ValueError(f"mask produced no boxes: {mask_path}")
                    source_key = f"{profile}__{defect_class}__{raw_index}"
                    write_pair(image, profile, split, box_lines, summary, source_key=source_key)
                    summary["by_class"][class_names[new_class_id]][split] += len(box_lines)

    # Emit YOLO YAML, class-wise counts, and an auditable source list.
    OUT.mkdir(parents=True, exist_ok=True)
    yaml_text = [
        "path: artifacts/datasets/unified_defects",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {idx}: {class_names[idx]}" for idx in range(len(class_names))]
    (OUT / "data.yaml").write_text("\n".join(yaml_text) + "\n", encoding="utf-8")
    report = {
        "taxonomy": summary["taxonomy"], "seed": summary["seed"],
        "split_policy": summary["split_policy"],
        "mvtec_defect_split_note": summary["mvtec_defect_split_note"],
        "class_count": len(class_names), "classes": class_names,
        "images": dict(summary["images"]), "boxes": dict(summary["boxes"]),
        "by_profile": {name: dict(counts) for name, counts in summary["by_profile"].items()},
        "by_class_boxes": {name: dict(counts) for name, counts in summary["by_class"].items()},
        "source_files": summary["source_files"],
    }
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("class_count", "images", "boxes", "by_profile", "by_class_boxes")}, indent=2))


if __name__ == "__main__":
    main()
