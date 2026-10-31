"""Lazy PatchCore-lite gate loader with explicit CPU fallback."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ml.anomaly.gate import AnomalyResult, PatchCoreGate
from server.app.config import Settings
from server.app.detector import resolve_device
from server.app.profiles import Profile


class GateService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.profile_id: str | None = None
        self.gate: PatchCoreGate | None = None
        self.device = resolve_device(settings.device)
        self.error: str | None = None
        self.load_ms = 0.0

    @property
    def ready(self) -> bool:
        return self.gate is not None

    def ensure(self, profile: Profile) -> PatchCoreGate | None:
        if profile.id == self.profile_id and self.gate is not None:
            return self.gate
        self.unload()
        if not profile.gate_path or not profile.gate_path.exists():
            self.error = "gate_bank_missing"
            self.profile_id = profile.id
            return None
        t0 = time.perf_counter()
        try:
            self.gate = PatchCoreGate(profile.gate_path, device=self.device)
            self.error = None
        except Exception as exc:  # pragma: no cover - environment dependent
            if self.device != "cpu":
                try:
                    self.device = "cpu"
                    self.gate = PatchCoreGate(profile.gate_path, device="cpu")
                    self.error = "gate_cpu_fallback"
                except Exception as exc2:
                    self.gate = None
                    self.error = f"gate_failed:{type(exc2).__name__}"
            else:
                self.gate = None
                self.error = f"gate_failed:{type(exc).__name__}"
        self.profile_id = profile.id
        self.load_ms = (time.perf_counter() - t0) * 1000.0
        return self.gate

    def unload(self) -> None:
        self.gate = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score(self, frame: np.ndarray) -> AnomalyResult | None:
        if self.gate is None:
            return None
        return self.gate.score_frame(frame)

    def info(self, profile: Profile) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "device": self.device,
            "error": self.error,
            "load_ms": round(self.load_ms, 1),
            "profile": self.profile_id,
            "gate_trusted": profile.gate_trusted,
            "gate_quality": profile.gate_quality,
            "separation": profile.gate_separation,
        }
