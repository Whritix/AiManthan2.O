"""Pydantic contracts shared by the server and the web client."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["pass", "review", "fail"]
Disposition = Literal["true_defect", "false_alarm", "reclassify"]
SourceKind = Literal["dataset", "camera", "video", "image"]


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    class_name: str = Field(alias="class")
    conf: float
    bbox: list[int] = Field(description="[x, y, w, h] in source-frame pixels")
    severity: int = 0
    source: str = "detector"
    votes: int = 0
    area_frac: float = 0.0


class AnomalyRegionOut(BaseModel):
    bbox: list[int]
    score: float
    area: int


class AnomalyInfo(BaseModel):
    raw_score: float
    z: float
    threshold: float
    is_anomaly: bool
    model: str
    gate_trusted: bool = True
    gate_quality: str = "unknown"
    separation: float | None = None
    regions: list[AnomalyRegionOut] = Field(default_factory=list)
    heatmap_url: str | None = None
    cached: bool = False


class LatencyMs(BaseModel):
    capture: float = 0.0
    preprocess: float = 0.0
    infer_detector: float = 0.0
    infer_gate: float = 0.0
    post: float = 0.0
    encode: float = 0.0
    total: float = 0.0


class Routing(BaseModel):
    destination: str
    reason: str


class PartState(BaseModel):
    index: int
    total: int
    item_id: str
    kind: str
    ground_truth_class: str | None = None
    ground_truth_kind: str | None = None
    frame_in_part: int = 1
    frames_in_part: int = 1


class FrameResult(BaseModel):
    schema_version: str = "1.0"
    frame_id: str
    inspection_id: str
    ts_ms: int
    stream_id: str = "cam-1"
    profile: str
    source: str
    device: str
    engine: str
    model_version: str
    threshold_profile: str
    latency_ms: LatencyMs
    detections: list[Detection] = Field(default_factory=list)
    anomaly: AnomalyInfo
    verdict: Verdict
    routing: Routing | None = None
    severity: int = 1
    part_state: PartState
    image_url: str | None = None
    overlay_url: str | None = None
    crop_url: str | None = None
    fps: float = 0.0
    frame_index: int = 0
    warnings: list[str] = Field(default_factory=list)


class ReviewAction(BaseModel):
    disposition: Disposition
    label: str | None = None
    note: str | None = None
    actor: str | None = None


class ThresholdUpdate(BaseModel):
    pass_max: float | None = None
    review_max: float | None = None
    unknown_z: float | None = None
    conf_min: float | None = None
    name: str | None = None


class ThresholdProfile(BaseModel):
    id: str
    name: str
    pass_max: float
    review_max: float
    unknown_z: float
    conf_min: float
    active: bool
    updated_at_ms: int


class InspectionSummary(BaseModel):
    id: str
    ts_ms: int
    profile: str
    source: str
    status: str
    verdict: Verdict | str
    final_verdict: Verdict | str | None = None
    severity: int = 1
    route: str | None = None
    reason: str | None = None
    anomaly_z: float = 0.0
    anomaly_raw: float = 0.0
    latency_total_ms: float = 0.0
    detections: list[Detection] = Field(default_factory=list)
    ground_truth_kind: str | None = None
    ground_truth_class: str | None = None
    operator_disposition: str | None = None
    operator_label: str | None = None
    operator_note: str | None = None
    reviewed_at_ms: int | None = None
    image_url: str | None = None
    heatmap_url: str | None = None
    overlay_url: str | None = None
    crop_url: str | None = None
    model_version: str | None = None
    threshold_profile: str | None = None
    frame_count: int = 0
    gate_trusted: bool = True
    warnings: list[str] = Field(default_factory=list)


class ProfileInfo(BaseModel):
    id: str
    display_name: str
    subtitle: str
    detector: bool
    gate_ready: bool
    gate_model: str | None = None
    gate_separation: float | None = None
    gate_quality: str = "unknown"
    license: str
    classes: list[str]
    counts: dict[str, Any]
    heldout_class: str | None = None


class HealthInfo(BaseModel):
    status: str
    version: str
    uptime_s: float
    profile: str
    source: str
    detector_mode: str
    detector_engine: str
    gate_ready: bool
    gate_trusted: bool
    device: str
    db: dict[str, Any]
    fps: float
    ws_clients: int
    warnings: list[str] = Field(default_factory=list)


class DatasetReport(BaseModel):
    neu: dict[str, Any]
    mvtec: dict[str, Any]
    profiles: list[dict[str, Any]]
    warnings: list[str]
    verified_at: str


class ControlRequest(BaseModel):
    action: Literal["pause", "resume", "skip", "profile", "sequence", "detector", "source"]
    value: str | None = None


class MesEventIn(BaseModel):
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
