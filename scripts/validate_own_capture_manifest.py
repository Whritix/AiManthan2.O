#!/usr/bin/env python3
"""Validate the real-capture intake manifest and group-safe splits."""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "own"
IMAGE_FIELDS = {
    "image_id", "relative_path", "part_id", "session_id", "view_id",
    "target_label", "hard_negative_type", "split", "review_status",
}
SESSION_FIELDS = {"session_id", "capture_date", "camera_model", "lighting_setup", "resolution"}
VALID_SPLITS = {"bank", "calib", "test"}
VALID_LABELS = {"good"}


def read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], [f"Missing file: {path.relative_to(ROOT)}"]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        rows = list(reader)
    errors = [f"{path.name} missing columns: {', '.join(sorted(missing))}"] if missing else []
    return rows, errors


def main() -> int:
    sessions, errors = read_csv(DATA / "sessions.csv", SESSION_FIELDS)
    images, image_errors = read_csv(DATA / "manifest.csv", IMAGE_FIELDS)
    errors.extend(image_errors)
    session_ids = [row.get("session_id", "").strip() for row in sessions]
    if len(session_ids) != len(set(session_ids)):
        errors.append("sessions.csv contains duplicate session_id values")
    known_sessions = set(session_ids)
    image_ids: set[str] = set()
    image_paths: set[str] = set()
    part_splits: dict[str, set[str]] = defaultdict(set)
    session_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()

    for line, row in enumerate(images, start=2):
        get = lambda key: row.get(key, "").strip()
        image_id, relative_path = get("image_id"), get("relative_path")
        part_id, session_id, split = get("part_id"), get("session_id"), get("split")
        if not all((image_id, relative_path, part_id, session_id, get("view_id"), get("target_label"), split, get("review_status"))):
            errors.append(f"manifest.csv row {line}: one or more required values are blank")
            continue
        if image_id in image_ids:
            errors.append(f"manifest.csv row {line}: duplicate image_id {image_id}")
        image_ids.add(image_id)
        if relative_path in image_paths:
            errors.append(f"manifest.csv row {line}: duplicate relative_path {relative_path}")
        image_paths.add(relative_path)
        if split not in VALID_SPLITS:
            errors.append(f"manifest.csv row {line}: invalid split {split!r}")
        if get("target_label") not in VALID_LABELS:
            errors.append(f"manifest.csv row {line}: this intake only accepts target_label=good")
        negative_type = get("hard_negative_type")
        if negative_type:
            negative_counts[negative_type] += 1
        if session_id not in known_sessions:
            errors.append(f"manifest.csv row {line}: unknown session_id {session_id}")
        part_splits[part_id].add(split)
        session_splits[session_id].add(split)
        split_counts[split] += 1
        path = (DATA / relative_path).resolve()
        try:
            path.relative_to(DATA.resolve())
        except ValueError:
            errors.append(f"manifest.csv row {line}: path escapes data/own: {relative_path}")
            continue
        if not path.is_file():
            errors.append(f"manifest.csv row {line}: image file not found: {relative_path}")

    for part_id, splits in part_splits.items():
        if len(splits) > 1:
            errors.append(f"physical part {part_id} leaks across splits: {sorted(splits)}")
    for session_id, splits in session_splits.items():
        if len(splits) > 1:
            errors.append(f"capture session {session_id} leaks across splits: {sorted(splits)}")

    if errors:
        print("Real-capture manifest validation: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    if not images:
        print("Real-capture manifest templates are valid; no images have been added yet.")
        return 0
    print(f"Real-capture manifest validation: PASS ({len(images)} images, {len(known_sessions)} sessions)")
    print(f"Images by split: {dict(split_counts)}")
    print(f"Hard negatives by type: {dict(negative_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
