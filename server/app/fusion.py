"""Part-level fusion: detector evidence + novelty score -> PASS/REVIEW/FAIL.

Rules are intentionally explainable and ordered. The gate never *classifies*;
it can only escalate to a human. A known class with support fails. Anything
ambiguous or novel goes to REVIEW. PASS is the only silent path.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from server.app.detector import Box
from server.app.schemas import Detection, Routing

# Domain judgement, documented in docs/FUSION_RULES.md.
CLASS_BASE_SEVERITY = {
    "crazing": 2,
    "inclusion": 3,
    "patches": 2,
    "pitted_surface": 3,
    "scratches": 1,
    "rolled-in_scale": 3,
}

CLASS_COLORS = {
    "crazing": "#7dd3fc",
    "inclusion": "#f472b6",
    "patches": "#fbbf24",
    "pitted_surface": "#a78bfa",
    "scratches": "#fb7185",
    "unknown": "#f59e0b",
}


def area_band(area_frac: float) -> int:
    if area_frac < 0.02:
        return 1
    if area_frac < 0.10:
        return 2
    if area_frac < 0.30:
        return 3
    return 4


def detection_severity(class_name: str, area_frac: float, conf: float, source: str = "detector") -> int:
    base = CLASS_BASE_SEVERITY.get(class_name, 2)
    sev = max(base, area_band(area_frac))
    if source != "fixture" and conf >= 0.90:
        sev += 1
    return int(max(1, min(4, sev)))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    class_name: str
    bbox: tuple[int, int, int, int]
    boxes: list[Box] = field(default_factory=list)
    frame_ids: set[int] = field(default_factory=set)

    @property
    def votes(self) -> int:
        return len(self.frame_ids)

    @property
    def conf(self) -> float:
        return statistics.median(b.conf for b in self.boxes)

    @property
    def area_frac(self) -> float:
        return statistics.median(
            (b.bbox[2] * b.bbox[3]) / max(1, self._frame_area) for b in self.boxes
        ) if getattr(self, "_frame_area", 0) else 0.0


@dataclass
class FusionDecision:
    detections: list[Detection]
    verdict: str
    severity: int
    route: str
    reason: str
    anomaly_z: float
    anomaly_raw: float
    gate_trusted: bool
    frame_count: int
    warnings: list[str] = field(default_factory=list)
    uncertain: list[dict[str, Any]] = field(default_factory=list)

    def routing(self) -> Routing | None:
        if self.verdict == "pass" and self.route == "release":
            return None
        return Routing(destination=self.route, reason=self.reason)


class PartAccumulator:
    def __init__(self, frame_area: int) -> None:
        self.frame_area = max(1, frame_area)
        self.frames: list[list[Box]] = []
        self.gate_z: list[float] = []
        self.gate_raw: list[float] = []
        self.gate_trusted = True
        self.warnings: list[str] = []

    def add(
        self,
        boxes: list[Box],
        gate_z: float | None = None,
        gate_raw: float | None = None,
        gate_trusted: bool = True,
        warnings: list[str] | None = None,
    ) -> None:
        self.frames.append(boxes)
        self.gate_trusted = self.gate_trusted and gate_trusted
        if gate_z is not None:
            self.gate_z.append(float(gate_z))
            self.gate_raw.append(float(gate_raw or 0.0))
        if warnings:
            for w in warnings:
                if w not in self.warnings:
                    self.warnings.append(w)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def _tracks(self) -> list[Track]:
        tracks: list[Track] = []
        for fidx, boxes in enumerate(self.frames):
            for box in boxes:
                match = None
                for track in tracks:
                    if track.class_name != box.class_name:
                        continue
                    if _iou(track.bbox, box.bbox) >= 0.45:
                        match = track
                        break
                if match is None:
                    match = Track(class_name=box.class_name, bbox=box.bbox)
                    tracks.append(match)
                match.boxes.append(box)
                match.frame_ids.add(fidx)
        for track in tracks:
            track._frame_area = self.frame_area  # type: ignore[attr-defined]
        return tracks

    def finalize(self, profile: Any, thresholds: dict[str, Any]) -> FusionDecision:
        tracks = self._tracks()
        n = max(1, self.frame_count)
        support_needed = 1 if n <= 2 else max(2, math.ceil(n * 0.25))
        conf_min = float(thresholds.get("conf_min", 0.35))
        pass_max = float(thresholds.get("pass_max", 0.55))
        unknown_z = float(thresholds.get("unknown_z", 0.75))

        detections: list[Detection] = []
        uncertain: list[dict[str, Any]] = []
        warnings = list(self.warnings)
        if self.gate_z:
            gate_z = float(statistics.median(self.gate_z))
            gate_raw = float(statistics.median(self.gate_raw))
        else:
            gate_z, gate_raw = 0.0, 0.0
            warnings.append("gate_skipped")

        for track in sorted(tracks, key=lambda t: (-t.votes, -t.conf)):
            area_frac = (track.bbox[2] * track.bbox[3]) / self.frame_area
            conf = track.conf
            if track.votes < support_needed or conf < conf_min:
                uncertain.append(
                    {
                        "class": track.class_name,
                        "conf": conf,
                        "votes": track.votes,
                        "support_needed": support_needed,
                    }
                )
                if track.votes < support_needed:
                    warnings.append("transient_detection")
                if conf < conf_min:
                    warnings.append("low_confidence_detection")
                continue
            detections.append(
                Detection(
                    id=f"{track.class_name}-{len(detections) + 1}",
                    class_name=track.class_name,
                    conf=round(conf, 4),
                    bbox=list(track.bbox),
                    severity=detection_severity(track.class_name, area_frac, conf, track.boxes[-1].source),
                    source=track.boxes[-1].source,
                    votes=track.votes,
                    area_frac=round(area_frac, 4),
                )
            )

        gate_trusted = self.gate_trusted and profile.gate_trusted
        if not profile.gate_trusted:
            warnings.append("gate_not_validated")

        severity = 1
        route = "release"
        if detections:
            verdict = "fail"
            severity = max(d.severity for d in detections)
            route = "quarantine" if severity >= 4 else ("rework" if severity >= 3 else "release_with_note")
            reason = "known_defect"
        elif uncertain:
            verdict = "review"
            severity = 2
            route = "human_review"
            reason = "transient_detection" if any(
                u["votes"] < u["support_needed"] for u in uncertain
            ) else "low_confidence_detection"
        elif gate_trusted and gate_z >= unknown_z:
            verdict = "review"
            severity = 2
            route = "human_review"
            reason = "unknown_anomaly"
        elif gate_trusted and profile.detector and gate_z >= pass_max:
            verdict = "review"
            severity = 2
            route = "human_review"
            reason = "elevated_novelty"
        else:
            verdict = "pass"
            route = "release"
            reason = "no_known_defect"

        if not gate_trusted and not detections:
            warnings.append("gate_does_not_gate_this_profile")

        return FusionDecision(
            detections=detections,
            verdict=verdict,
            severity=severity,
            route=route,
            reason=reason,
            anomaly_z=round(gate_z, 4),
            anomaly_raw=round(gate_raw, 4),
            gate_trusted=gate_trusted,
            frame_count=n,
            warnings=sorted(set(warnings)),
            uncertain=uncertain[:6],
        )
