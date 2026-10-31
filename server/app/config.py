"""Runtime configuration for the Indore QA cell."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    profile: str = field(default_factory=lambda: _env("INDORE_PROFILE", "steel"))
    source: str = field(default_factory=lambda: _env("INDORE_SOURCE", "dataset"))
    detector_mode: str = field(default_factory=lambda: _env("INDORE_DETECTOR", "auto"))
    device: str = field(default_factory=lambda: _env("INDORE_DEVICE", "auto"))
    imgsz: int = field(default_factory=lambda: _env_int("INDORE_IMGSZ", 640))
    conf: float = field(default_factory=lambda: _env_float("INDORE_CONF", 0.25))
    iou: float = field(default_factory=lambda: _env_float("INDORE_IOU", 0.60))
    part_frames: int = field(default_factory=lambda: _env_int("INDORE_PART_FRAMES", 12))
    target_fps: float = field(default_factory=lambda: _env_float("INDORE_TARGET_FPS", 12.0))
    gate_every: int = field(default_factory=lambda: _env_int("INDORE_GATE_EVERY", 0))
    mjpeg_quality: int = field(default_factory=lambda: _env_int("INDORE_JPEG_QUALITY", 82))
    stream_size: int = field(default_factory=lambda: _env_int("INDORE_STREAM_SIZE", 800))
    sequence: str = field(default_factory=lambda: _env("INDORE_SEQUENCE", "mixed"))
    mes_url: str = field(default_factory=lambda: _env("INDORE_MES_URL", ""))
    webhook_secret: str = field(default_factory=lambda: _env("INDORE_WEBHOOK_SECRET", "dev-secret"))
    actor: str = field(default_factory=lambda: _env("INDORE_ACTOR", "operator.demo"))
    camera_index: int = field(default_factory=lambda: _env_int("INDORE_CAMERA_INDEX", 0))
    video_path: str = field(default_factory=lambda: _env("INDORE_VIDEO_PATH", ""))
    image_path: str = field(default_factory=lambda: _env("INDORE_IMAGE_PATH", ""))
    max_evidence: int = field(default_factory=lambda: _env_int("INDORE_MAX_EVIDENCE", 0))
    version: str = "1.0.0"

    @property
    def data_dir(self) -> Path:
        return Path(_env("INDORE_DATA_DIR", str(self.project_root / "data")))

    @property
    def artifacts_dir(self) -> Path:
        return Path(_env("INDORE_ARTIFACTS_DIR", str(self.project_root / "artifacts")))

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def evidence_dir(self) -> Path:
        return self.artifacts_dir / "evidence"

    @property
    def db_path(self) -> Path:
        return Path(_env("INDORE_DB", str(self.artifacts_dir / "indore.db")))

    @property
    def profiles_manifest(self) -> Path:
        return self.data_dir / "profiles" / "manifest.json"

    @property
    def splits_path(self) -> Path:
        return self.data_dir / "neu-det" / "splits.json"

    @property
    def gate_summary_path(self) -> Path:
        return self.models_dir / "gate_summary.json"

    @property
    def auto_profile_model(self) -> Path:
        return self.models_dir / "auto_profile_classifier.joblib"

    @property
    def web_dist(self) -> Path:
        return self.project_root / "web" / "dist"

    @property
    def promoted_weights(self) -> Path | None:
        for name in ("detector_best.pt", "detector_best.onnx"):
            candidate = self.models_dir / name
            if candidate.exists():
                return candidate
        return None

    @property
    def training_weights(self) -> Path:
        return self.artifacts_dir / "runs" / "neu_yolo11s" / "weights" / "best.pt"

    @property
    def unified_weights(self) -> Path:
        """Candidate 21-class checkpoint trained from the combined dataset."""
        return self.artifacts_dir / "runs" / "unified_yolo11s_640" / "weights" / "best.pt"

    @property
    def detector_weights(self) -> Path | None:
        """Promoted artifact first. A live training checkpoint is opt-in only."""
        if self.detector_mode == "unified" and self.unified_weights.exists():
            return self.unified_weights
        promoted = self.promoted_weights
        if promoted is not None:
            return promoted
        if self.detector_mode == "real-run" and self.training_weights.exists():
            return self.training_weights
        return None

    def ensure_dirs(self) -> None:
        for p in (self.artifacts_dir, self.models_dir, self.evidence_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
