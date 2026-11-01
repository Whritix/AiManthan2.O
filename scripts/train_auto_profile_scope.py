"""Train a candidate five-family classifier with an internal out-of-scope reject class."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.app.auto_profile import AutoProfileClassifier, SUPPORTED_FAMILIES
from scripts.train_auto_profile import MVTec, PROFILES, all_images, images

OTHER_CATEGORIES = ("cable", "capsule", "carpet", "grid", "hazelnut", "leather", "pill", "tile", "toothbrush", "wood", "zipper")


def main() -> None:
    train_x: list[np.ndarray] = []
    train_y: list[str] = []
    eval_x: list[np.ndarray] = []
    eval_y: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for family in PROFILES:
        if family == "steel":
            train_paths = all_images(ROOT / "data/neu-det/images/train")
            eval_paths = all_images(ROOT / "data/neu-det/images/test")
        else:
            category = MVTec / family
            packaged = ROOT / "data/profiles" / family
            train_paths = all_images(category / "train/good") or all_images(packaged / "bank") + all_images(packaged / "calib")
            eval_paths = all_images(category / "test") or all_images(packaged / "test_good") + all_images(packaged / "test_defect")
        counts[family] = {
            "train": images(train_paths, family, train_x, train_y),
            "evaluation": images(eval_paths, family, eval_x, eval_y),
        }
    other_train = 0
    other_eval = 0
    for category in OTHER_CATEGORIES:
        folder = MVTec / category
        other_train += images(all_images(folder / "train/good")[:100], "other", train_x, train_y)
        other_eval += images(all_images(folder / "test")[:40], "other", eval_x, eval_y)
    counts["other"] = {"train": other_train, "evaluation": other_eval}
    if any(not item["train"] or not item["evaluation"] for item in counts.values()):
        raise SystemExit(f"Missing scope training or evaluation images: {counts}")

    model = make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3,
                      class_weight="balanced", random_state=42),
    )
    model.fit(np.stack(train_x), np.asarray(train_y))
    features = np.stack(eval_x)
    probabilities = np.nan_to_num(model.predict_proba(features), nan=0.0)
    best = np.argmax(probabilities, axis=1)
    predicted = np.asarray(model.classes_)[best]
    confidence = probabilities[np.arange(len(best)), best]
    accepted = np.isin(predicted, list(SUPPORTED_FAMILIES)) & (confidence >= 0.55)
    truth = np.asarray(eval_y)
    known = truth != "other"
    other = ~known
    candidate_path = ROOT / "artifacts/models/auto_profile_scoped_candidate.joblib"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, candidate_path)
    classifier = AutoProfileClassifier(candidate_path)
    demo_results = []
    for family in PROFILES:
        for kind in ("defect", "clean"):
            folder = ROOT / "artifacts/demo_images" / ("recommended" if kind == "defect" else family)
            pattern = f"{family}_*.jpg" if kind == "defect" else "clean--*.jpg"
            path = next(iter(sorted(folder.glob(pattern))), None)
            result = classifier.identify(cv2.imread(str(path))) if path else {"status": "missing", "profile": None}
            demo_results.append({"family": family, "kind": kind, "prediction": result.get("profile"), "status": result["status"]})
    report = {
        "families": list(PROFILES), "internal_reject_label": "other", "other_categories": list(OTHER_CATEGORIES),
        "counts": counts, "acceptance_threshold": 0.55,
        "known_correct_acceptance": float(np.mean(accepted[known] & (predicted[known] == truth[known]))),
        "other_rejection": float(np.mean(~accepted[other])),
        "known_rejected": float(np.mean(~accepted[known])),
        "other_false_acceptance": float(np.mean(accepted[other])),
        "classification_report": classification_report(truth, predicted, labels=list(PROFILES) + ["other"], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(PROFILES) + ["other"]).tolist(),
        "demo_results": demo_results,
        "limitation": "Other-category evaluation uses MVTec classes seen during other-class training. Arbitrary unknown objects may still be misclassified.",
    }
    report_path = ROOT / "artifacts/eval/auto_profile_scope_candidate.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("counts", "known_correct_acceptance", "other_rejection", "known_rejected", "other_false_acceptance", "demo_results")}, indent=2))


if __name__ == "__main__":
    main()
