"""Small visual product classifier used to choose an inspection profile.

This is intentionally separate from the defect detector: it identifies the
part family, then the selected profile's detector/gate handles inspection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np


class AutoProfileClassifier:
    def __init__(self, model_path: Path, confidence_floor: float = 0.55) -> None:
        self.model_path = model_path
        self.confidence_floor = confidence_floor
        self._model: Any | None = None
        if model_path.exists():
            self._model = joblib.load(model_path)

    @property
    def ready(self) -> bool:
        return self._model is not None

    @staticmethod
    def features(frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("empty image")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        # HOG descriptor (8x8 cells, 2x2-cell blocks), implemented here
        # because some Windows OpenCV wheels omit cv2.HOGDescriptor.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        angle %= 180.0
        bins = np.floor(angle / 20.0).astype(np.int32).clip(0, 8)
        weighted_bins = np.eye(9, dtype=np.float32)[bins] * magnitude[:, :, None]
        cells = weighted_bins.reshape(16, 8, 16, 8, 9).sum(axis=(1, 3))
        blocks = np.stack((cells[:-1, :-1], cells[1:, :-1], cells[:-1, 1:], cells[1:, 1:]), axis=2)
        norms = np.sqrt(np.sum(blocks * blocks, axis=(2, 3), keepdims=True) + 1e-6)
        shape = (blocks / norms).reshape(-1)
        if frame.ndim == 2:
            color = np.zeros(8 * 8 * 4, dtype=np.float32)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
            color = cv2.normalize(color, None).reshape(-1)
        return np.concatenate((shape.astype(np.float32), color.astype(np.float32)))

    def identify(self, frame: np.ndarray) -> dict[str, Any]:
        if not self.ready:
            return {"status": "unavailable", "profile": None, "confidence": 0.0,
                    "message": "Automatic product identification model is not installed."}
        vector = self.features(frame).reshape(1, -1)
        probabilities = self._model.predict_proba(vector)[0]
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        profile = str(self._model.classes_[best])
        status = "identified" if confidence >= self.confidence_floor else "uncertain"
        return {"status": status, "profile": profile if status == "identified" else None,
                "candidate": profile, "confidence": confidence,
                "message": "Product family identified." if status == "identified"
                else "Product type is uncertain; use a clearer, centered image."}
