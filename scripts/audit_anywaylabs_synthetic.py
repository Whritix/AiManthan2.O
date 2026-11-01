#!/usr/bin/env python3
"""Audit the isolated AnywayLabs screw/transistor YOLO download."""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "datasets/anywaylabs_synthetic_mvtec_554be045"
UNIFIED = ROOT / "artifacts/datasets/unified_defects"
PROFILES = ("screw", "transistor")


def summarize_areas(areas: list[float]) -> dict:
    return {"count": len(areas), "median": statistics.median(areas) if areas else None}


def main() -> None:
    report = {"source": str(SOURCE), "profiles": {}, "notes": []}
    for profile in PROFILES:
        base = SOURCE / profile
        config = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8"))
        names = {int(k): v for k, v in config["names"].items()}
        profile_report = {"classes": names, "splits": {}}
        for split in ("train", "val"):
            images = sorted(p for p in (base / "images" / split).glob("*") if p.is_file())
            labels = sorted((base / "labels" / split).glob("*.txt"))
            image_stems = {p.stem for p in images}
            label_stems = {p.stem for p in labels}
            count_by_class = Counter()
            areas_by_class = defaultdict(list)
            invalid = []
            for label in labels:
                for line in label.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                    except (ValueError, IndexError):
                        invalid.append(f"{label.name}: {line}")
                        continue
                    # Published labels use three decimals, so a rounded edge may be 0.0005 outside [0, 1].
                    eps = 0.001
                    if len(parts) != 5 or cid not in names or w <= 0 or h <= 0 or not (-eps <= cx-w/2 <= cx+w/2 <= 1+eps) or not (-eps <= cy-h/2 <= cy+h/2 <= 1+eps):
                        invalid.append(f"{label.name}: {line}")
                        continue
                    count_by_class[names[cid]] += 1
                    areas_by_class[names[cid]].append(w*h)
            source_overlap = []
            if split == "val":
                for image in images:
                    defect, _, index = image.stem.rpartition("_")
                    if index.isdigit() and (UNIFIED / "images/val" / f"{profile}__{defect}__{index}.jpg").exists():
                        source_overlap.append(image.name)
                    elif index.isdigit() and (UNIFIED / "images/test" / f"{profile}__{defect}__{index}.jpg").exists():
                        source_overlap.append(image.name)
                    elif index.isdigit() and (UNIFIED / "images/train" / f"{profile}__{defect}__{index}.jpg").exists():
                        source_overlap.append(image.name)
            profile_report["splits"][split] = {
                "images": len(images), "labels": len(labels),
                "unpaired_images": sorted(image_stems - label_stems)[:20],
                "unpaired_labels": sorted(label_stems - image_stems)[:20],
                "invalid_labels": invalid[:20], "invalid_label_count": len(invalid),
                "boxes_by_class": dict(count_by_class),
                "box_area_fraction_by_class": {name: summarize_areas(areas) for name, areas in areas_by_class.items()},
                "source_id_overlap_with_existing_unified": len(source_overlap),
                "source_id_overlap_examples": source_overlap[:10],
            }
        # Compare the box conventions to current unified training labels.
        unified_areas = defaultdict(list)
        for label in (UNIFIED / "labels/train").glob(f"{profile}__*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                name = label.stem.split("__")[1] if "__" in label.stem else ""
                if name in names.values():
                    unified_areas[name].append(float(parts[3])*float(parts[4]))
        profile_report["existing_unified_train_box_area_fraction"] = {
            name: summarize_areas(areas) for name, areas in unified_areas.items()}
        report["profiles"][profile] = profile_report
    report["notes"] = [
        "Only the new synthetic train folders are candidates for training; source validation overlaps existing MVTec-derived splits.",
        "Different median box sizes may indicate incompatible annotation conventions; inspect images before any merge.",
    ]
    out = SOURCE / "intake_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
