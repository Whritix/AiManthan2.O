"""Train and report the upload-time part-family classifier.

Usage: python scripts/train_auto_profile.py
Training uses NEU-DET train for steel and MVTec train/good for the four
additional supported families. Evaluation is kept on their test directories.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.app.auto_profile import AutoProfileClassifier

MVTec = ROOT.parent / "mvtec_anomaly_detection"
PROFILES = ("steel", "metal_nut", "screw", "transistor", "bottle")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def images(paths: list[Path], label: str, features: list[np.ndarray], labels: list[str]) -> int:
    added = 0
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        features.append(AutoProfileClassifier.features(frame))
        labels.append(label)
        added += 1
    return added


def all_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def main() -> None:
    train_x: list[np.ndarray] = []
    train_y: list[str] = []
    eval_x: list[np.ndarray] = []
    eval_y: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for profile in PROFILES:
        if profile == "steel":
            train_paths = all_images(ROOT / "data/neu-det/images/train")
            eval_paths = all_images(ROOT / "data/neu-det/images/test")
        else:
            category = MVTec / profile
            packaged = ROOT / "data/profiles" / profile
            train_paths = all_images(category / "train/good")
            eval_paths = all_images(category / "test")
            if not train_paths:
                train_paths = all_images(packaged / "bank") + all_images(packaged / "calib")
            if not eval_paths:
                eval_paths = all_images(packaged / "test_good") + all_images(packaged / "test_defect")
        n_train = images(train_paths, profile, train_x, train_y)
        n_eval = images(eval_paths, profile, eval_x, eval_y)
        counts[profile] = {"train": n_train, "evaluation": n_eval}
    if any(value["train"] == 0 or value["evaluation"] == 0 for value in counts.values()):
        raise SystemExit(f"Missing train/evaluation images: {counts}")

    model = make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3,
                      class_weight="balanced", random_state=42),
    )
    model.fit(np.stack(train_x), np.asarray(train_y))
    prediction = model.predict(np.stack(eval_x))
    report = classification_report(eval_y, prediction, labels=PROFILES, output_dict=True, zero_division=0)
    summary = {
        "families": list(PROFILES), "counts": counts,
        "accuracy": float(accuracy_score(eval_y, prediction)),
        "per_class_recall": {name: float(report[name]["recall"]) for name in PROFILES},
        "confusion_matrix": confusion_matrix(eval_y, prediction, labels=PROFILES).tolist(),
        "classification_report": report,
        "method": "HOG shape descriptor + HSV histogram + balanced logistic SGD classifier",
        "warning": "Profile identification is an image-level routing classifier, not a defect detector.",
    }
    model_path = ROOT / "artifacts/models/auto_profile_classifier.joblib"
    report_path = ROOT / "artifacts/eval/auto_profile_report.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(model_path), "report": str(report_path), "accuracy": summary["accuracy"],
                      "counts": counts, "per_class_recall": summary["per_class_recall"],
                      "confusion_matrix": summary["confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    main()
