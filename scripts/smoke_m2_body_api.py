"""Offline M2 Body model smoke; live HTTP endpoints need a network-enabled run."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.app.m2_body import M2BodyClassifier  # noqa: E402

SOURCE = ROOT.parent / "datasets/m2_screw_v2/extracted/BODY/Train/NG/BARI-1.BMP"
WRONG_SIZE = ROOT / "artifacts/demo_images/recommended/steel_patches.jpg"
WEIGHTS = ROOT / "artifacts/runs/m2_body_resnet18_v1/best.pt"


def main() -> None:
    classifier = M2BodyClassifier(WEIGHTS)
    report = json.loads(WEIGHTS.with_name("report.json").read_text(encoding="utf-8"))
    assert classifier.available and report["test_images"] == 1000
    image = cv2.imread(str(SOURCE))
    assert classifier.classify(image)["result"] == "NG"
    try:
        classifier.classify(cv2.imread(str(WRONG_SIZE)))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-size image was accepted")
    print("M2 model: checkpoint available; NG classified; wrong-size image rejected")


if __name__ == "__main__":
    main()
