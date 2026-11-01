#!/usr/bin/env python3
"""Compare MVTec mask components to the unified YOLO boxes, without edits."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/datasets/unified_defects"
MVTEC = ROOT.parent / "mvtec_anomaly_detection"
OUT = ROOT / "artifacts/eval/unified_finetune_20261101/mask_box_audit.json"
TARGETS = {
    "metal_nut_scratch": 8,
    "screw_manipulated_front": 9,
    "screw_thread_side": 12,
    "screw_thread_top": 13,
    "transistor_bent_lead": 14,
    "transistor_damaged_case": 16,
}


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "p90": sorted(values)[int(0.9 * (len(values) - 1))], "max": max(values)}


def main() -> None:
    rows = []
    for split in ("train", "val", "test"):
        for image in sorted((DATA / "images" / split).glob("*")):
            parts = image.stem.split("__")
            if len(parts) != 3 or f"{parts[0]}_{parts[1]}" not in TARGETS:
                continue
            profile, defect, index = parts
            frame = cv2.imread(str(image))
            mask_path = MVTEC / profile / "ground_truth" / defect / f"{index}_mask.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if frame is None or mask is None:
                raise FileNotFoundError(f"Could not read image or mask: {image}, {mask_path}")
            h, w = frame.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
            components = [(int(x), int(y), int(bw), int(bh), int(area))
                          for x, y, bw, bh, area in stats[1:n] if area >= 4 and bw >= 1 and bh >= 1]
            label = DATA / "labels" / split / f"{image.stem}.txt"
            boxes = [list(map(float, line.split())) for line in label.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
            if len(boxes) != len(components):
                raise ValueError(f"Component/label count mismatch: {image}: {len(components)} vs {len(boxes)}")
            component_ratios = []
            for (x, y, bw, bh, area), (cid, cx, cy, lw, lh) in zip(components, boxes):
                if int(cid) != TARGETS[f"{profile}_{defect}"]:
                    raise ValueError(f"Wrong class ID: {label}")
                if abs(lw * w - bw) > 0.01 or abs(lh * h - bh) > 0.01:
                    raise ValueError(f"Box does not match component: {label}")
                component_ratios.append(bw * bh / area)
            if components:
                x0 = min(c[0] for c in components)
                y0 = min(c[1] for c in components)
                x1 = max(c[0] + c[2] for c in components)
                y1 = max(c[1] + c[3] for c in components)
                union_ratio = (x1 - x0) * (y1 - y0) / sum(c[4] for c in components)
            else:
                union_ratio = None
            rows.append({"split": split, "class": f"{profile}_{defect}", "image": image.name,
                         "components": len(components), "component_box_area_over_mask_area": component_ratios,
                         "union_box_area_over_mask_area": union_ratio})
    summary = {}
    for class_name in TARGETS:
        selected = [r for r in rows if r["class"] == class_name]
        summary[class_name] = {
            "images_by_split": {s: sum(r["split"] == s for r in selected) for s in ("train", "val", "test")},
            "multi_component_images": sum(r["components"] > 1 for r in selected),
            "component_count": describe([r["components"] for r in selected]),
            "component_box_to_mask_area": describe([v for r in selected for v in r["component_box_area_over_mask_area"]]),
            "union_box_to_mask_area": describe([r["union_box_area_over_mask_area"] for r in selected
                                                if r["union_box_area_over_mask_area"] is not None]),
        }
    report = {"definition": "Box area / binary mask area after nearest-neighbor resize; lower is tighter.",
              "summary": summary, "images": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
