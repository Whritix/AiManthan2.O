"""Detector wrapper: real YOLO11s, or an explicitly labelled ground-truth fixture.

The fixture exists so the application is demoable while weights are training or
when CUDA/torch is unavailable. It never pretends to be a model: the engine
string, API field and burned-in watermark all say FIXTURE.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from server.app.config import Settings


@dataclass
class Box:
    class_name: str
    conf: float
    bbox: tuple[int, int, int, int]
    source: str = "detector"


def _sha256(path: Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(limit)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class FixtureDetector:
    """Reads ground-truth boxes from the replay item. Clearly a fixture."""

    engine = "fixture-groundtruth"
    device = "cpu"
    ready = True
    warning = "fixture_ground_truth"

    def __init__(self, classes: list[str]) -> None:
        self.classes = set(classes)
        self.model_version = "fixture@neu-det-labels"

    def predict(self, frame: np.ndarray, item: Any | None = None) -> list[Box]:
        if item is None or item.kind != "known_defect":
            return []
        h, w = frame.shape[:2]
        out: list[Box] = []
        for idx, box in enumerate(item.boxes):
            cls = box.get("class")
            if not cls:
                cid = int(box.get("class_id", 0))
                cls = item.class_name if cid >= 5 else self._class_from_id(cid, item)
            if cls not in self.classes:
                continue
            cx, cy = float(box["cx"]) * w, float(box["cy"]) * h
            bw, bh = float(box["w"]) * w, float(box["h"]) * h
            x = int(max(0, round(cx - bw / 2)))
            y = int(max(0, round(cy - bh / 2)))
            bw_i = int(max(1, min(w - x, round(bw))))
            bh_i = int(max(1, min(h - y, round(bh))))
            out.append(
                Box(
                    class_name=cls,
                    conf=0.94,
                    bbox=(x, y, bw_i, bh_i),
                    source="fixture",
                )
            )
        return out

    @staticmethod
    def _class_from_id(cid: int, item: Any) -> str:
        # The manifest's steel boxes carry NEU-DET class ids 0..4 in class order.
        order = ["crazing", "inclusion", "patches", "pitted_surface", "scratches"]
        if 0 <= cid < len(order):
            return order[cid]
        return item.class_name

    def info(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "device": self.device,
            "model_version": self.model_version,
            "ready": True,
            "warning": self.warning,
        }


class YoloDetector:
    def __init__(
        self,
        weights: Path,
        settings: Settings,
        classes: list[str],
        class_prefix: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights = weights
        self.settings = settings
        self.allowed = set(classes)
        self.class_prefix = class_prefix
        self.model = YOLO(str(weights), task="detect")
        if settings.device == "auto":
            self.device = "0" if _cuda_available() else "cpu"
        else:
            self.device = settings.device
        suffix = weights.suffix.lower()
        engine_suffix = "onnx" if suffix == ".onnx" else "pt"
        self.engine = f"yolo11s-unified-{engine_suffix}" if class_prefix else f"yolo11s-{engine_suffix}"
        digest = _sha256(weights)
        self.model_version = f"{weights.stem}@{digest[:10]}"
        self.ready = True
        self.warning = "" if self.device != "cpu" else "cpu_fallback"

    def predict(self, frame: np.ndarray, item: Any | None = None) -> list[Box]:
        results = self.model.predict(
            source=frame,
            imgsz=self.settings.imgsz,
            conf=self.settings.conf,
            iou=self.settings.iou,
            device=self.device,
            verbose=False,
        )
        boxes: list[Box] = []
        if not results:
            return boxes
        names = results[0].names or {}
        for row in results[0].boxes:
            cid = int(row.cls.item())
            cls = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
            if self.class_prefix:
                prefix = f"{self.class_prefix}_"
                if not cls.startswith(prefix):
                    continue
                cls = cls[len(prefix):]
            if cls not in self.allowed:
                continue
            xyxy = row.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy
            boxes.append(
                Box(
                    class_name=cls,
                    conf=float(row.conf.item()),
                    bbox=(int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))),
                    source="detector",
                )
            )
        return boxes

    def info(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "device": self.device,
            "model_version": self.model_version,
            "ready": True,
            "warning": self.warning,
            "weights": str(self.weights),
        }


class NullDetector:
    engine = "detector-off"
    device = "cpu"
    ready = False
    warning = "detector_disabled"
    model_version = "none"

    def predict(self, frame: np.ndarray, item: Any | None = None) -> list[Box]:
        return []

    def info(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "device": self.device,
            "model_version": self.model_version,
            "ready": False,
            "warning": self.warning,
        }


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_detector(settings: Settings, profile: Any) -> Any:
    """Resolve the configured detector; strict real mode fails instead of masking errors."""
    unified = settings.detector_mode == "unified"
    if not profile.detector and not unified:
        return NullDetector()

    mode = settings.detector_mode
    if mode == "fixture":
        return FixtureDetector(profile.classes)
    if mode == "off":
        return NullDetector()

    weights = settings.detector_weights
    if weights is None:
        if mode in {"real", "unified"}:
            raise RuntimeError("real_detector_requested_but_no_promoted_weights_found")
        det = FixtureDetector(profile.classes)
        det.warning = "fixture_ground_truth_no_promoted_weights"
        return det
    try:
        prefix = {"steel": "steel"}.get(profile.id, profile.id) if unified else None
        detector = YoloDetector(weights, settings, profile.classes, class_prefix=prefix)
        # Warm up once so the first live frame does not pay CUDA kernel setup.
        t0 = time.perf_counter()
        detector.predict(np.zeros((640, 640, 3), dtype=np.uint8))
        detector.warmup_ms = (time.perf_counter() - t0) * 1000.0
        return detector
    except Exception as exc:  # pragma: no cover - environment dependent
        if mode in {"real", "unified"}:
            raise RuntimeError(f"real_detector_failed:{type(exc).__name__}:{exc}") from exc
        det = FixtureDetector(profile.classes)
        det.warning = f"real_detector_failed:{type(exc).__name__}"
        return det


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if requested.startswith("cuda") and not _cuda_available():
        return "cpu"
    return requested
