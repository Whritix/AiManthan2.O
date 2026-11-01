#!/usr/bin/env python3
"""Build a small, separate Render demo bundle; never alter the local datasets."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT.parent / f"render_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FAMILIES = ("steel", "metal_nut", "screw", "transistor", "bottle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, relative: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = TARGET / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def sample_classes(items: list[dict], count: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[item.get("class", "unknown")].append(item)
    chosen: list[dict] = []
    while len(chosen) < count and any(buckets.values()):
        for name in sorted(buckets):
            if buckets[name] and len(chosen) < count:
                chosen.append(buckets[name].pop(0))
    return chosen


def package_item(item: dict, family: str, kind: str, index: int) -> dict:
    source = Path(item["path"])
    relative = Path("data") / "demo_replay" / family / kind / f"{index:02d}_{source.name}"
    copy_file(source, relative)
    result = copy.deepcopy(item)
    result["path"] = relative.as_posix()
    return result


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    TARGET.mkdir(parents=True)
    for folder in ("server", "ml", "web/dist"):
        shutil.copytree(ROOT / folder, TARGET / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log"))

    original = json.loads((ROOT / "data/profiles/manifest.json").read_text(encoding="utf-8"))
    packaged = {}
    for family in FAMILIES:
        entry = copy.deepcopy(original[family])
        good = sample_classes(entry.get("good", []), 5)
        defect = sample_classes(entry.get("defect", []), 5)
        unknown = sample_classes(entry.get("unknown", []), 2) if family == "steel" else []
        entry["good"] = [package_item(item, family, "good", i) for i, item in enumerate(good)]
        entry["defect"] = [package_item(item, family, "defect", i) for i, item in enumerate(defect)]
        entry["unknown"] = [package_item(item, family, "unknown", i) for i, item in enumerate(unknown)] if family == "steel" else entry["defect"]
        entry["bank_crops"] = []
        entry["bank_images"] = []
        entry["calib_frames"] = []
        entry["counts"] = {"good": len(entry["good"]), "known_defect": len(entry["defect"]) if family == "steel" else 0,
                           "unknown_defect": len(entry["unknown"])}
        packaged[family] = entry
    manifest = TARGET / "data/profiles/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(packaged, indent=2), encoding="utf-8")

    model_files = [
        "models/auto_profile_classifier.joblib",
        "models/gate_summary.json",
        *[f"models/gate_{family}.npz" for family in FAMILIES],
        "runs/unified_yolo11s_640/weights/best.pt",
        "datasets/unified_defects/manifest.json",
        "eval/unified_yolo11s_640_test.json",
        "eval/test_metrics.json",
    ]
    for name in model_files:
        source = ROOT / "artifacts" / name
        if source.exists():
            copy_file(source, Path("artifacts") / name)
    shutil.copytree(ROOT / "artifacts/demo_images", TARGET / "artifacts/demo_images")

    (TARGET / "requirements.txt").write_text(
        "fastapi>=0.115,<1\nuvicorn[standard]>=0.30,<1\npydantic>=2.8,<3\n"
        "numpy>=1.26,<3\nopencv-python>=4.10,<5\n"
        "ultralytics>=8.3,<9\nscikit-learn>=1.4,<2\njoblib>=1.3,<2\n"
        "requests>=2.32,<3\n", encoding="utf-8")
    (TARGET / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg "
        "YOLO_CONFIG_DIR=/tmp/ultralytics INDORE_RENDER=1 INDORE_DETECTOR=unified "
        "INDORE_DEVICE=cpu INDORE_SOURCE=dataset INDORE_SEQUENCE=mixed "
        "INDORE_TARGET_FPS=2 INDORE_MAX_EVIDENCE=100\n"
        "WORKDIR /app\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
        "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY . .\n"
        "CMD [\"sh\", \"-c\", \"exec uvicorn server.app.main:app --host 0.0.0.0 --port ${PORT:-10000}\"]\n",
        encoding="utf-8")
    (TARGET / ".dockerignore").write_text(".git\n__pycache__\n*.pyc\n*.db*\n*.log\nartifacts/evidence/\nartifacts/demo_inputs/\nultralytics-settings/\n", encoding="utf-8")
    (TARGET / ".gitignore").write_text("__pycache__/\n*.pyc\nartifacts/evidence/\nartifacts/demo_inputs/\nartifacts/current_report.json\nartifacts/*.db*\nultralytics-settings/\n", encoding="utf-8")
    weights = TARGET / "artifacts/runs/unified_yolo11s_640/weights/best.pt"
    (TARGET / "BUNDLE_INFO.json").write_text(json.dumps({
        "purpose": "Temporary private Render hackathon demo",
        "families": list(FAMILIES),
        "replay_images": {family: len(packaged[family]["good"]) + len(packaged[family]["defect"]) for family in FAMILIES},
        "weights_sha256": sha256(weights),
        "note": "Includes limited MVTec AD and NEU-DET sample images. Keep the repository private and observe dataset licenses."
    }, indent=2), encoding="utf-8")
    (TARGET / "render.yaml").write_text(
        "services:\n  - type: web\n    name: inspection-ai-demo\n    runtime: docker\n"
        "    plan: 2c-8g\n    region: singapore\n    branch: main\n"
        "    autoDeployTrigger: off\n    healthCheckPath: /healthz\n"
        "    envVars:\n"
        "      - key: INDORE_ADMIN_USERNAME\n        sync: false\n"
        "      - key: INDORE_ADMIN_PASSWORD\n        sync: false\n"
        "      - key: INDORE_EMPLOYEE_USERNAME\n        sync: false\n"
        "      - key: INDORE_EMPLOYEE_PASSWORD\n        sync: false\n"
        "      - key: INDORE_AUTH_SECRET\n        generateValue: true\n",
        encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
