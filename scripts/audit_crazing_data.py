#!/usr/bin/env python3
"""Check source-label fidelity and offline crazing augmentation labels."""
from __future__ import annotations

import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "NEUDATA" / "NEU-DET"
CLASS_TO_ID = {name: i for i, name in enumerate(
    ["crazing", "inclusion", "patches", "pitted_surface", "scratches"]
)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_labels(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_TO_ID:
            continue
        box = obj.find("bndbox")
        x0, y0, x1, y1 = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        xc, yc = ((x0 + x1) / 2 / 200, (y0 + y1) / 2 / 200)
        w, h = ((x1 - x0) / 200, (y1 - y0) / 200)
        out.append(f"{CLASS_TO_ID[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return out


def source_annotation(image_path: Path) -> Path | None:
    for split in ("train", "validation"):
        source_image = RAW / split / "images" / "crazing" / image_path.name
        if source_image.exists() and sha256(source_image) == sha256(image_path):
            for annotation_split in (split, "train" if split == "validation" else "validation"):
                annotation = RAW / annotation_split / "annotations" / f"{image_path.stem}.xml"
                if annotation.exists():
                    return annotation
    return None


def read_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def main() -> None:
    data = ROOT / "data" / "neu-det"
    manifest = json.loads((data / "splits.json").read_text(encoding="utf-8"))
    report_dir = ROOT / "artifacts" / "eval" / "crazing_data_audit_20261029"
    report_dir.mkdir(parents=True, exist_ok=True)

    source_checks = []
    for split in ("train", "val", "test"):
        for image in sorted((data / "images" / split).glob("crazing_*.jpg")):
            if "_aug_" in image.stem:
                continue
            annotation = source_annotation(image)
            label = data / "labels" / split / f"{image.stem}.txt"
            expected = expected_labels(annotation) if annotation else None
            actual = label.read_text(encoding="utf-8").splitlines() if label.exists() else None
            source_checks.append({
                "split": split, "image": image.name,
                "source_xml": str(annotation) if annotation else None,
                "status": "match" if expected is not None and actual == expected else "mismatch",
                "expected_boxes": len(expected) if expected is not None else None,
                "actual_boxes": len(actual) if actual is not None else None,
            })

    aug_checks = []
    image_diffs = {"hflip": [], "clahe": []}
    sample_names = []
    aug_files = manifest["augmentations"]["files"]
    for row in aug_files:
        stem = Path(row["file"]).stem
        transform = row["transformation"]
        parent = data / "images" / "train" / row["parent_image"]
        image = data / "images" / "train" / row["file"]
        parent_label = data / "labels" / "train" / f"{parent.stem}.txt"
        label = data / "labels" / "train" / row["label_file"]
        src_rows = read_yolo(parent_label)
        got_rows = read_yolo(label)
        if transform == "clahe":
            expected_rows = src_rows
        elif transform == "hflip":
            expected_rows = [(cid, 1.0 - xc, yc, w, h) for cid, xc, yc, w, h in src_rows]
        else:
            expected_rows = []
        labels_match = len(got_rows) == len(expected_rows) and all(
            g[0] == e[0] and all(abs(g[i] - e[i]) <= 1.1e-6 for i in range(1, 5))
            for g, e in zip(got_rows, expected_rows)
        )
        parent_img = cv2.imread(str(parent))
        aug_img = cv2.imread(str(image))
        if parent_img is None or aug_img is None or parent_img.shape != aug_img.shape:
            pixel_mae = None
        else:
            if transform == "hflip":
                expected_img = cv2.flip(parent_img, 1)
            else:
                lab = cv2.cvtColor(parent_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
                expected_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            pixel_mae = float(np.mean(np.abs(aug_img.astype(np.int16) - expected_img.astype(np.int16))))
            image_diffs[transform].append(pixel_mae)
        aug_checks.append({"file": row["file"], "transform": transform,
                           "label_match": labels_match, "pixel_mae_vs_regenerated": pixel_mae})
        if row["parent_image"] not in sample_names:
            sample_names.append(row["parent_image"])

    # Contact sheet of eight source/CLAHE/flip triplets with boxes drawn.
    chosen = sorted(sample_names)[:8]
    tile_w, tile_h, title_h = 200, 200, 24
    canvas = np.full((len(chosen) * (tile_h + title_h), 3 * tile_w, 3), 245, dtype=np.uint8)
    for r, parent_name in enumerate(chosen):
        base = Path(parent_name).stem
        for c, variant in enumerate(("", "_aug_clahe", "_aug_hflip")):
            img_path = data / "images" / "train" / f"{base}{variant}.jpg"
            lbl_path = data / "labels" / "train" / f"{base}{variant}.txt"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            for cid, xc, yc, w, h in read_yolo(lbl_path):
                x1, y1 = int((xc - w / 2) * tile_w), int((yc - h / 2) * tile_h)
                x2, y2 = int((xc + w / 2) * tile_w), int((yc + h / 2) * tile_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 1)
            x, y = c * tile_w, r * (tile_h + title_h)
            canvas[y:y + title_h, x:x + tile_w] = 255
            cv2.putText(canvas, f"{base} {('original','CLAHE','hflip')[c]}", (x + 3, y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
            canvas[y + title_h:y + title_h + tile_h, x:x + tile_w] = img
    sheet_path = report_dir / "augmentation_review.jpg"
    cv2.imwrite(str(sheet_path), canvas)

    # All 45 crazing test frames, split into readable contact sheets.
    error_images_dir = ROOT / "artifacts" / "eval" / "test_error_analysis_20261029" / "error_images"
    test_crazing = sorted(error_images_dir.glob("crazing_*.jpg"),
                          key=lambda p: int(p.stem.split("_")[-1]))
    error_sheets = []
    for sheet_idx in range((len(test_crazing) + 11) // 12):
        subset = test_crazing[sheet_idx * 12:(sheet_idx + 1) * 12]
        grid = np.full((4 * (tile_h + title_h), 3 * tile_w, 3), 245, dtype=np.uint8)
        for idx, image_path in enumerate(subset):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            row, col = divmod(idx, 3)
            x, y = col * tile_w, row * (tile_h + title_h)
            grid[y:y + title_h, x:x + tile_w] = 255
            cv2.putText(grid, image_path.name, (x + 3, y + 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (20, 20, 20), 1, cv2.LINE_AA)
            grid[y + title_h:y + title_h + tile_h, x:x + tile_w] = img
        path = report_dir / f"test_crazing_review_{sheet_idx + 1:02d}.jpg"
        cv2.imwrite(str(path), grid)
        error_sheets.append(str(path))

    # Summarize the test-set crazing failures for quick visual review.
    error_csv = ROOT / "artifacts" / "eval" / "test_error_analysis_20261029" / "errors.csv"
    gt_failure_counts = Counter()
    if error_csv.exists():
        import csv
        with error_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["ground_truth_class"] == "crazing" and row["kind"].startswith("ground_truth_"):
                    gt_failure_counts[row["kind"].removeprefix("ground_truth_")] += 1

    report = {
        "raw_dataset_root": str(RAW),
        "source_label_checks": {"checked": len(source_checks),
                                "matches": sum(x["status"] == "match" for x in source_checks),
                                "mismatches": [x for x in source_checks if x["status"] != "match"]},
        "augmentation_checks": {"checked": len(aug_checks),
                                "label_matches": sum(x["label_match"] for x in aug_checks),
                                "label_mismatches": [x for x in aug_checks if not x["label_match"]],
                                "image_dimensions_ok": all(x["pixel_mae_vs_regenerated"] is not None for x in aug_checks),
                                "regenerated_pixel_mae_mean": {k: float(np.mean(v)) if v else None for k, v in image_diffs.items()},
                                "regenerated_pixel_mae_max": {k: float(np.max(v)) if v else None for k, v in image_diffs.items()}},
        "crazing_test_ground_truth_failures": dict(gt_failure_counts),
        "visual_review_sheet": str(sheet_path),
        "test_crazing_error_sheets": error_sheets,
        "notes": ["Image-label correspondence is checked against source XML by exact source image hash.",
                  "All 420 augmentation labels are checked against their parent labels and stated transformation.",
                  "Pixel regeneration error includes JPEG encode/decode differences; the label check is the geometry check."],
        "source_details": source_checks,
        "augmentation_details": aug_checks,
    }
    report_path = report_dir / "audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_details", "augmentation_details")}, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
