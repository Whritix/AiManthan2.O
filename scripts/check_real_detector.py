"""Preflight the currently configured profile and ensure the trained YOLO loads."""
from __future__ import annotations

import sys
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))

from server.app.config import Settings  # noqa: E402
from server.app.detector import build_detector  # noqa: E402
from server.app.profiles import ProfileRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--video", dest="input_path", required=True, type=Path)
    args = parser.parse_args()
    if not args.input_path.is_file():
        print(f"ERROR: input file does not exist: {args.input_path}", file=sys.stderr)
        return 4
    import cv2
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if args.input_path.suffix.lower() in image_extensions:
        has_frame = cv2.imread(str(args.input_path)) is not None
    else:
        capture = cv2.VideoCapture(str(args.input_path))
        opened = capture.isOpened()
        has_frame, _ = capture.read() if opened else (False, None)
        capture.release()
    if not has_frame:
        print("ERROR: OpenCV could not decode the selected input", file=sys.stderr)
        return 5

    settings = Settings()
    settings.detector_mode = "real"
    registry = ProfileRegistry(settings)
    profile = registry.get(settings.profile)
    weights = settings.promoted_weights
    if weights is None:
        print("ERROR: no promoted detector weights found in artifacts/models/", file=sys.stderr)
        return 2
    detector = build_detector(settings, profile)
    if not detector.engine.startswith("yolo") or not detector.ready:
        print(f"ERROR: expected a loaded YOLO detector; got {detector.engine}", file=sys.stderr)
        return 3
    print(f"YOLO ready: {detector.engine} on {detector.device}")
    print(f"Weights: {weights}")
    print(f"Input readable: {args.input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
