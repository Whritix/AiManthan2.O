"""Optional, isolated classifier for the Mendeley M2 screw Body camera domain."""
from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class M2BodyClassifier:
    def __init__(self, weights: Path) -> None:
        self.weights = weights
        self._model: nn.Module | None = None
        self._threshold: float | None = None
        self._sha256: str | None = None
        self._lock = Lock()
        self._preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @property
    def available(self) -> bool:
        return self.weights.is_file()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError(f"M2 Body checkpoint missing: {self.weights}")
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=False)
        if checkpoint.get("classes") != ["OK", "NG"] or checkpoint.get("crop_source_xyxy") != (350, 450, 670, 870):
            raise ValueError("M2 Body checkpoint has unexpected classes or crop")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._threshold = float(checkpoint["operating_threshold"])
        self._sha256 = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self._model = model

    def classify(self, bgr: cv2.typing.MatLike) -> dict:
        # This fixed crop is validated only for the source M2 Body frame size.
        if bgr is None or bgr.shape[:2] != (1280, 1024):
            raise ValueError("M2 Body check requires a 1024×1280 image from the M2 Body camera setup")
        with self._lock:
            self._load()
            assert self._model is not None and self._threshold is not None
            crop = bgr[450:870, 350:670]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self._preprocess(Image.fromarray(rgb)).unsqueeze(0)
            with torch.inference_mode():
                score = float(torch.softmax(self._model(tensor), dim=1)[0, 1])
        return {
            "task": "M2 screw Body OK/NG",
            "result": "NG" if score >= self._threshold else "OK",
            "ng_probability": round(score, 6),
            "operating_threshold": round(self._threshold, 6),
            "model_sha256": self._sha256,
            "scope": "M2 Body camera setup only; image-level classification, no defect box or subtype",
        }
