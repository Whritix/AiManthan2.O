#!/usr/bin/env python3
"""Check weak-class boxes and create a validation-only label review sheet."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/datasets/unified_defects"
OUT = ROOT / "artifacts/eval/unified_weak_audit"
IDS = {0, 8, 9, 12, 13, 14, 16}


def rows(path: Path) -> list[tuple[int, float, float, float, float]]:
    return [(int(x[0]), *map(float, x[1:])) for line in path.read_text(encoding="utf-8").splitlines()
            if len(x := line.split()) == 5]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples: dict[int, list[tuple[Path, list[tuple]]]] = defaultdict(list)
    summary = {split: {str(cid): {"images": 0, "boxes": 0, "small_boxes": 0, "min_area_fraction": None}
                       for cid in sorted(IDS)} for split in ("train", "val", "test")}
    bad_boxes = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for split in ("train", "val", "test"):
        for image in sorted((DATA / "images" / split).glob("*")):
            label = DATA / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                bad_boxes.append(f"missing label: {image}")
                continue
            lines = rows(label)
            seen = set()
            for cid, cx, cy, w, h in lines:
                eps = 0.00001  # six-decimal YOLO coordinates can round an edge past zero
                if not 0 <= cid < 21 or not 0 < w <= 1 or not 0 < h <= 1 or not -eps <= cx-w/2 < cx+w/2 <= 1+eps or not -eps <= cy-h/2 < cy+h/2 <= 1+eps:
                    bad_boxes.append(f"invalid box: {label} {cid} {cx} {cy} {w} {h}")
                if cid in IDS:
                    entry = summary[split][str(cid)]
                    entry["boxes"] += 1
                    entry["small_boxes"] += w*h < 0.0001
                    entry["min_area_fraction"] = min(entry["min_area_fraction"] or 1, w*h)
                    seen.add(cid)
            for cid in seen:
                summary[split][str(cid)]["images"] += 1
                if split == "val" and len(samples[cid]) < 6:
                    samples[cid].append((image, lines))
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            hashes[digest].append(f"{split}/{image.name}")
    leaks = [paths for paths in hashes.values() if len({p.split('/')[0] for p in paths}) > 1]

    tile_w, tile_h, title_h = 256, 256, 28
    for cid, selected in samples.items():
        sheet = np.full((2*(tile_h+title_h), 3*tile_w, 3), 245, dtype=np.uint8)
        for i, (image, labels) in enumerate(selected):
            frame = cv2.imread(str(image))
            if frame is None:
                bad_boxes.append(f"unreadable: {image}")
                continue
            h, w = frame.shape[:2]
            for label_cid, cx, cy, bw, bh in labels:
                if label_cid != cid:
                    continue
                cv2.rectangle(frame, (round((cx-bw/2)*w), round((cy-bh/2)*h)),
                              (round((cx+bw/2)*w), round((cy+bh/2)*h)), (0, 255, 255), 2)
            frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            x, y = (i % 3)*tile_w, (i // 3)*(tile_h+title_h)
            sheet[y:y+title_h, x:x+tile_w] = 255
            cv2.putText(sheet, image.name[-35:], (x+4, y+18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (0, 0, 0), 1, cv2.LINE_AA)
            sheet[y+title_h:y+title_h+tile_h, x:x+tile_w] = frame
        cv2.imwrite(str(OUT / f"class_{cid}_val_labels.jpg"), sheet)
    report = {"summary": summary, "invalid_or_missing_labels": bad_boxes[:100],
              "invalid_or_missing_count": len(bad_boxes), "cross_split_exact_image_leaks": leaks[:30],
              "cross_split_exact_image_leak_count": len(leaks)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(OUT / 'report.json'), "bad_count": len(bad_boxes),
                      "leak_count": len(leaks), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
