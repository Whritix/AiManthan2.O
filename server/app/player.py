"""Dataset replay / camera capture loop and the live inspection pipeline."""
from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from server.app.config import Settings
from server.app.db import Database
from server.app.detector import Box, NullDetector, YoloDetector, build_detector
from server.app.evidence import EvidenceStore
from server.app.fusion import PartAccumulator, detection_severity
from server.app.demo_routing import simulated_plc_action
from server.app.gate_service import GateService
from server.app.hub import Hub
from server.app.overlay import encode_jpeg, render
from server.app.profiles import Item, Profile, ProfileRegistry
from server.app.risk_explanation import explain_flagged_risk
from server.app.schemas import (
    AnomalyInfo,
    AnomalyRegionOut,
    Detection,
    FrameResult,
    LatencyMs,
    PartState,
)

STREAM_IDS = {"dataset": "replay-1", "camera": "cam-1", "video": "video-1", "image": "image-1"}


class DatasetSequence:
    def __init__(self, profile: Profile, mode: str = "mixed", seed: int = 42) -> None:
        self.profile = profile
        self.mode = mode
        self.items = self._build(profile, mode, seed)
        self.index = 0
        self.cycle = 0
        if not self.items:
            raise RuntimeError(f"profile {profile.id} has no replay items")

    @staticmethod
    def _shuffle(items: list[Item], profile: Profile, kind: str, seed: int) -> list[Item]:
        out = list(items)
        random.Random(f"{seed}:{profile.id}:{kind}").shuffle(out)
        return out

    @classmethod
    def _build(cls, profile: Profile, mode: str, seed: int) -> list[Item]:
        good = cls._shuffle(profile.good, profile, "good", seed)
        defect = cls._shuffle(profile.defect, profile, "defect", seed)
        unknown = cls._shuffle(profile.unknown, profile, "unknown", seed)
        if mode == "good":
            return good or defect or unknown
        if mode == "defect":
            return defect or unknown or good
        if mode == "unknown":
            return unknown or defect or good

        # mixed: merge category streams by normalised progress so every class
        # appears throughout the shift instead of one long block per class.
        sources: list[tuple[list[Item], str]] = [(good, "good"), (defect, "defect"), (unknown, "unknown")]
        if not profile.detector and {i.path for i in defect} == {i.path for i in unknown}:
            sources = [(good, "good"), (unknown, "unknown")]
        merged: list[tuple[float, int, Item]] = []
        order = {"good": 0, "defect": 1, "unknown": 2}
        for items, kind in sources:
            n = len(items)
            for idx, item in enumerate(items):
                merged.append(((idx + 0.5) / n, order[kind], item))
        merged.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in merged]

    def current(self) -> Item:
        return self.items[self.index]

    def current_profile(self) -> Profile:
        return self.profile

    def advance(self) -> None:
        self.index += 1
        if self.index >= len(self.items):
            self.index = 0
            self.cycle += 1

    @property
    def total(self) -> int:
        return len(self.items)


class ProductBlockSequence:
    """Replay ten dataset images per product before moving to the next family."""

    def __init__(self, profiles: list[Profile], mode: str = "mixed", seed: int = 42) -> None:
        streams: list[tuple[Profile, list[Item]]] = []
        for profile in profiles:
            if mode == "mixed":
                defects = DatasetSequence._shuffle(profile.defect or profile.unknown, profile, "defect", seed)
                good = DatasetSequence._shuffle(profile.good, profile, "good", seed)
                if defects and good:
                    items = []
                    for i in range(10):
                        bank = defects if i % 2 == 0 else good
                        items.append(bank[(i // 2) % len(bank)])
                else:
                    items = defects or good
            else:
                items = DatasetSequence(profile, mode, seed).items
            if items:
                streams.append((profile, items))
        if not streams:
            raise RuntimeError("no product datasets available for replay")
        self.entries = [
            (profile, items[i % len(items)])
            for profile, items in streams
            for i in range(10)
        ]
        self.index = 0
        self.cycle = 0

    def current(self) -> Item:
        return self.entries[self.index][1]

    def current_profile(self) -> Profile:
        return self.entries[self.index][0]

    def advance(self) -> None:
        self.index += 1
        if self.index >= len(self.entries):
            self.index = 0
            self.cycle += 1

    @property
    def total(self) -> int:
        return len(self.entries)


@dataclass
class PartContext:
    inspection_id: str
    index: int
    item: Item | None
    kind: str
    gt_class: str | None
    gt_kind: str | None
    frames_total: int
    frame_in_part: int
    accumulator: PartAccumulator
    started_ms: int
    latencies: list[float]
    persist: bool = True
    last_anomaly: Any = None


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        registry: ProfileRegistry,
        db: Database,
        hub: Hub,
        evidence: EvidenceStore,
        gate: GateService,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.db = db
        self.hub = hub
        self.evidence = evidence
        self.gate = gate

        self.profile: Profile = registry.get(settings.profile)
        self.sequence_mode = settings.sequence
        self.sequence = self._new_dataset_sequence()
        self.detector = build_detector(settings, self.profile)

        self._stop = threading.Event()
        self._paused = threading.Event()
        self._skip = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._pending_profile: str | None = None
        self._pending_sequence: str | None = None
        self._pending_source: str | None = None
        self._pending_detector: str | None = None

        self.source = settings.source
        self.source_warning = ""
        self.input_warning = ""
        self.capture: cv2.VideoCapture | None = None
        self.capture_fallbacks = 0
        self.warnings: list[str] = []

        self.part: PartContext | None = None
        self.part_counter = 0
        self.frame_index = 0
        self.frame_in_part = 0
        self._current_image: np.ndarray | None = None
        self._image_path: str | None = None
        self._recorded_image_path: str | None = None
        self._video_completed_once = False
        self._last_gate = None
        self._last_gate_ms = 0.0
        self._last_gate_warning = ""
        self.started_at = time.time()
        self._sleep_until = 0.0

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.db.abandon_open("server_restart")
        self._thread = threading.Thread(target=self._run, name="indore-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self.part and self.part.persist:
            self.db.abandon_inspection(self.part.inspection_id, "server_stop")
        if self.capture is not None:
            self.capture.release()
        self.gate.unload()

    def command(self, action: str, value: str | None = None) -> dict[str, Any]:
        if action == "pause":
            self._paused.set()
            self.db.audit_event(self.settings.actor, "pipeline.pause", "pipeline", None, {})
        elif action == "resume":
            self._paused.clear()
            self.db.audit_event(self.settings.actor, "pipeline.resume", "pipeline", None, {})
        elif action == "skip":
            self._skip.set()
        elif action == "profile" and value:
            self.registry.get(value)
            self._pending_profile = value
        elif action == "sequence" and value:
            self._pending_sequence = value
        elif action == "source" and value:
            self._pending_source = value
        elif action == "detector" and value:
            self._pending_detector = value
        else:
            raise ValueError(f"unknown control action: {action}")
        return self.state()

    def state(self) -> dict[str, Any]:
        det = self.detector.info()
        gate = self.gate.info(self.profile)
        return {
            "paused": self._paused.is_set(),
            "profile": self.profile.id,
            "profile_name": self.profile.display_name,
            "source": self.source,
            "sequence": self.sequence_mode,
            "detector": det,
            "gate": gate,
            "device": self._device_label(),
            "engine": self._engine_label(),
            "fps": round(self.hub.fps, 2),
            "frame_index": self.frame_index,
            "part_index": self.part_counter,
            "part": (
                {
                    "inspection_id": self.part.inspection_id,
                    "kind": self.part.kind,
                    "gt_class": self.part.gt_class,
                    "frame": self.part.frame_in_part,
                    "frames": self.part.frames_total,
                }
                if self.part
                else None
            ),
            "sequence_total": self.sequence.total if self.source == "dataset" else 0,
            "uptime_s": round(time.time() - self.started_at, 1),
            "warnings": sorted(set(self.warnings + [warning for warning in (
                self.source_warning,
                self.input_warning if self.source in {"image", "video"} else "",
            ) if warning])),
        }

    def _pending(self) -> bool:
        return any(
            (
                self._pending_profile,
                self._pending_sequence,
                self._pending_source,
                self._pending_detector,
            )
        )

    def _apply_pending(self) -> None:
        with self._lock:
            pid = self._pending_profile
            seq = self._pending_sequence
            src = self._pending_source
            det = self._pending_detector
            self._pending_profile = self._pending_sequence = self._pending_source = self._pending_detector = None
        if self.part and self.part.persist:
            self.db.abandon_inspection(self.part.inspection_id, "operator_reconfigure")
        if self.part:
            self.part = None
            self.frame_in_part = 0
        if pid and pid != self.profile.id:
            self.profile = self.registry.get(pid)
            self.settings.profile = pid
            self.detector = build_detector(self.settings, self.profile)
            self.gate.ensure(self.profile)
            self._last_gate = None
            self.sequence = self._new_dataset_sequence()
            self._reset_capture()
            self.db.audit_event(self.settings.actor, "profile.change", "profile", pid, {})
        if seq and seq in {"mixed", "good", "defect", "unknown"}:
            self.sequence_mode = seq
            self.settings.sequence = seq
            self.sequence = self._new_dataset_sequence()
            self.db.audit_event(self.settings.actor, "sequence.change", "pipeline", None, {"sequence": seq})
        if src and src in {"dataset", "camera", "video", "image"}:
            self.source = src
            self.settings.source = src
            if src == "video":
                self._video_completed_once = False
            self._reset_capture()
            self.db.audit_event(self.settings.actor, "source.change", "pipeline", None, {"source": src})
        if det and det in {"auto", "real", "real-run", "unified", "fixture", "off"}:
            self.settings.detector_mode = det
            self.detector = build_detector(self.settings, self.profile)
            self.sequence = self._new_dataset_sequence()
            self.db.audit_event(self.settings.actor, "detector.change", "pipeline", None, {"mode": det})

    def _reset_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.source_warning = ""

    def _new_dataset_sequence(self) -> DatasetSequence | ProductBlockSequence:
        if self.settings.detector_mode == "unified":
            families = ("steel", "metal_nut", "screw", "transistor", "bottle")
            profiles = [self.registry.get(family) for family in families]
            return ProductBlockSequence(profiles, self.sequence_mode)
        return DatasetSequence(self.profile, self.sequence_mode)

    def _activate_replay_profile(self, profile: Profile) -> None:
        if profile.id == self.profile.id:
            return
        self.profile = profile
        self.settings.profile = profile.id
        if isinstance(self.detector, YoloDetector) and self.settings.detector_mode == "unified":
            self.detector.set_profile(profile)
        else:
            self.detector = build_detector(self.settings, profile)
        self.gate.ensure(profile)
        self._last_gate = None

    # -------------------------------------------------------------- loop helpers
    def _run(self) -> None:
        self.gate.ensure(self.profile)
        while not self._stop.is_set():
            try:
                if self._paused.is_set():
                    time.sleep(0.08)
                    continue
                if self._pending():
                    self._apply_pending()
                if self._skip.is_set():
                    self._skip.clear()
                    if self.part and self.part.persist:
                        self.db.abandon_inspection(self.part.inspection_id, "operator_skip", self.settings.actor)
                    self.part = None
                    self.frame_in_part = 0
                    if self.source == "dataset":
                        self.sequence.advance()
                    continue
                if self.source == "dataset":
                    self._run_dataset_frame()
                elif self.source == "image":
                    self._run_image_frame()
                else:
                    self._run_capture_frame()
                self._pace()
            except Exception as exc:  # keep the cell alive and visible
                msg = f"pipeline_error:{type(exc).__name__}"
                if msg not in self.warnings:
                    self.warnings.append(msg)
                    self.db.audit_event("system", "pipeline.error", "pipeline", None, {"error": str(exc)})
                time.sleep(0.5)

    def _run_dataset_frame(self) -> None:
        item = self.sequence.current()
        if self.part is None:
            self._activate_replay_profile(self.sequence.current_profile())
            frame = self._load_image(item.path)
            if frame is None:
                self.sequence.advance()
                return
            self.part_counter += 1
            self._start_part(item, frame, self.settings.part_frames)
        elif self.part.item is not None and self.part.item.path != item.path:
            # Defensive: item changed without a completed part.
            if self.part.persist:
                self.db.abandon_inspection(self.part.inspection_id, "sequence_desync")
            self.part = None
            return

        frame = self._load_image(item.path)
        if frame is None:
            if self.part.persist:
                self.db.abandon_inspection(self.part.inspection_id, "missing_frame")
            self.part = None
            self.frame_in_part = 0
            self.sequence.advance()
            return
        self.frame_in_part += 1
        final = self.frame_in_part >= self.part.frames_total
        self._process(frame, item, self.frame_in_part, self.part.frames_total, final)
        if final:
            self.sequence.advance()
            self.part = None
            self.frame_in_part = 0

    def _run_image_frame(self) -> None:
        frame = self._load_image(self.settings.image_path) if self.settings.image_path else None
        if frame is None:
            self.source_warning = "image_unavailable"
            self._fallback_to_dataset()
            return
        if self.part is None:
            self.part_counter += 1
            self._start_part(None, frame, self.settings.part_frames)
        self.frame_in_part += 1
        final = self.frame_in_part >= self.part.frames_total
        self._process(frame, None, self.frame_in_part, self.part.frames_total, final)
        if final:
            if self.part.persist:
                self._recorded_image_path = self.settings.image_path
            self.part = None
            self.frame_in_part = 0

    def _run_capture_frame(self) -> None:
        if self.capture is None and not self._open_capture():
            self._fallback_to_dataset()
            return
        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            if self.source == "video" and self.capture is not None:
                # Loop the uploaded clip so a short demo video remains live.
                if self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    self.capture_fallbacks = 0
                    ok, frame = self.capture.read()
                    if ok and frame is not None:
                        self._video_completed_once = True
                        if self.part and self.part.persist:
                            self.db.abandon_inspection(self.part.inspection_id, "video_loop")
                        self.part = None
                        self.frame_in_part = 0
                    else:
                        self._fallback_to_dataset()
                        return
                else:
                    self._fallback_to_dataset()
                    return
            else:
                self.capture_fallbacks += 1
                if self.capture_fallbacks > 3:
                    self._fallback_to_dataset()
                else:
                    time.sleep(0.1)
                return
        if self.part is None:
            self.part_counter += 1
            self._start_part(None, frame, self.settings.part_frames)
        self.frame_in_part += 1
        final = self.frame_in_part >= self.part.frames_total
        self._process(frame, None, self.frame_in_part, self.part.frames_total, final)
        if final:
            self.part = None
            self.frame_in_part = 0

    def _open_capture(self) -> bool:
        target: int | str = self.settings.camera_index if self.source == "camera" else self.settings.video_path
        if self.source == "video" and not target:
            self.source_warning = "video_path_missing"
            return False
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            cap.release()
            self.source_warning = f"{self.source}_unavailable"
            return False
        self.capture = cap
        self.source_warning = ""
        self.capture_fallbacks = 0
        return True

    def _fallback_to_dataset(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.source = "dataset"
        self.settings.source = "dataset"
        self.source_warning = "source_fallback_dataset_replay"
        self.warnings.append(self.source_warning)
        self.db.audit_event("system", "source.fallback", "pipeline", None, {"reason": self.source_warning})
        if self.part and self.part.persist:
            self.db.abandon_inspection(self.part.inspection_id, "source_fallback")
        self.part = None
        self.frame_in_part = 0

    # --------------------------------------------------------------- inspection
    def _start_part(self, item: Item | None, frame: np.ndarray, frames_total: int) -> None:
        inspection_id = "insp_" + uuid.uuid4().hex[:20]
        h, w = frame.shape[:2]
        thresholds = self.db.active_threshold()
        kind = item.kind if item else "live"
        gt_class = item.class_name if item else None
        gt_kind = item.kind if item else None
        persist = (
            self.source == "camera"
            or (self.source == "image" and self.settings.image_path != self._recorded_image_path)
            or (self.source == "video" and not self._video_completed_once)
        )
        if persist:
            self.db.create_inspection(
                {
                    "id": inspection_id,
                    "ts_ms": int(time.time() * 1000),
                    "profile": self.profile.id,
                    "source": self.source,
                    "status": "open",
                    "verdict": "pass",
                    "final_verdict": None,
                    "severity": 1,
                    "route": None,
                    "reason": None,
                    "anomaly_z": 0.0,
                    "anomaly_raw": 0.0,
                    "latency_total_ms": 0.0,
                    "frame_count": 0,
                    "detections_json": [],
                    "ground_truth_kind": gt_kind,
                    "ground_truth_class": gt_class,
                    "model_version": self._model_version(),
                    "threshold_profile": thresholds["id"],
                }
            )
        self.part = PartContext(
            inspection_id=inspection_id,
            index=self.part_counter,
            item=item,
            kind=kind,
            gt_class=gt_class,
            gt_kind=gt_kind,
            frames_total=max(1, frames_total),
            frame_in_part=0,
            accumulator=PartAccumulator(h * w),
            started_ms=int(time.time() * 1000),
            latencies=[],
            persist=persist,
            last_anomaly=None,
        )

    def _process(
        self,
        frame: np.ndarray,
        item: Item | None,
        frame_in_part: int,
        frames_total: int,
        final: bool,
    ) -> None:
        assert self.part is not None
        t0 = time.perf_counter()
        frame = frame.copy()
        self.frame_index += 1
        self.part.frame_in_part = frame_in_part
        thresholds = self.db.active_threshold()
        warnings = list(self.warnings)
        if self.source in {"image", "video"} and self.input_warning:
            warnings.append(self.input_warning)

        # Detector
        t_det = time.perf_counter()
        boxes: list[Box] = self.detector.predict(frame, item)
        det_ms = (time.perf_counter() - t_det) * 1000.0
        det_info = self.detector.info()
        if self.profile.detector and det_info.get("warning") and det_info["warning"] not in warnings:
            warnings.append(det_info["warning"])
        if det_info.get("warning") == "cpu_fallback":
            warnings.append("cpu_fallback")

        # Gate (possibly cached on CPU to keep the cell responsive)
        gate_every = self._gate_every()
        should_run_gate = self.gate.ready and (self.part.last_anomaly is None or frame_in_part % gate_every == 0)
        gate_ms = 0.0
        anomaly = self.part.last_anomaly
        gate_cached = False
        if should_run_gate:
            t_gate = time.perf_counter()
            try:
                anomaly = self.gate.score(frame)
                self.part.last_anomaly = anomaly
                self._last_gate_warning = ""
            except Exception as exc:  # pragma: no cover - environment dependent
                self._last_gate_warning = f"gate_failed:{type(exc).__name__}"
                anomaly = None
            gate_ms = (time.perf_counter() - t_gate) * 1000.0
        elif anomaly is not None:
            gate_cached = True
        if self.gate.error and self.gate.error not in warnings:
            warnings.append(self.gate.error)
        if self._last_gate_warning:
            warnings.append(self._last_gate_warning)

        gate_z = anomaly.z if anomaly is not None else 0.0
        gate_raw = anomaly.raw_score if anomaly is not None else 0.0
        gate_trusted = bool(self.profile.gate_trusted and anomaly is not None)
        if anomaly is None:
            warnings.append("gate_unavailable")

        # Per-frame boxes for the accumulator (with local severity)
        per_frame = []
        for box in boxes:
            area_frac = (box.bbox[2] * box.bbox[3]) / max(1, frame.shape[0] * frame.shape[1])
            per_frame.append(
                Detection(
                    id=box.class_name,
                    class_name=box.class_name,
                    conf=round(box.conf, 4),
                    bbox=list(box.bbox),
                    severity=detection_severity(box.class_name, area_frac, box.conf, box.source),
                    source=box.source,
                    votes=1,
                    area_frac=round(area_frac, 4),
                )
            )
        self.part.accumulator.add(
            boxes,
            gate_z=gate_z if anomaly is not None else None,
            gate_raw=gate_raw,
            gate_trusted=gate_trusted,
            warnings=warnings,
        )
        decision = self.part.accumulator.finalize(self.profile, thresholds)
        if self.source in {"image", "video"} and self.input_warning and decision.verdict == "pass":
            decision.verdict = "review"
            decision.route = "human_review"
            decision.reason = "product_appearance_outside_training_domain"
            decision.severity = max(decision.severity, 2)
        warnings = decision.warnings

        # Overlay
        t_post = time.perf_counter()
        heatmap = anomaly.heatmap if anomaly is not None else None
        heatmap_gray = anomaly.heatmap_gray if anomaly is not None else None
        unknown_threshold = float(thresholds["unknown_z"])
        show_unknown_regions = (
            decision.gate_trusted
            and decision.anomaly_z >= unknown_threshold
            and not decision.detections
        )
        regions = anomaly.regions if anomaly is not None and show_unknown_regions else []
        part_label = f"PART {self.part.index:04d}  {self.part.kind.replace('_', ' ').upper()}"
        warning_text = ""
        if self.source_warning:
            warning_text = "DATASET FALLBACK"
        elif det_info.get("warning") == "cpu_fallback":
            warning_text = "CPU FALLBACK"
        overlay_frame = render(
            frame,
            decision.detections,
            decision.verdict,  # type: ignore[arg-type]
            profile_id=self.profile.id,
            engine=det_info["engine"],
            source=self.source,
            gate_z=decision.anomaly_z,
            gate_trusted=decision.gate_trusted,
            gate_quality=self.profile.gate_quality,
            heatmap_bgr=heatmap,
            heatmap_gray=heatmap_gray,
            regions=regions,
            size=self.settings.stream_size,
            part_label=part_label,
            warning=warning_text,
            hud=False,
        )
        jpeg = encode_jpeg(overlay_frame, self.settings.mjpeg_quality)
        post_ms = (time.perf_counter() - t_post) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        self.part.latencies.append(total_ms)

        anomaly_info = AnomalyInfo(
            raw_score=round(decision.anomaly_raw, 5),
            z=round(decision.anomaly_z, 4),
            threshold=float(thresholds["unknown_z"]),
            is_anomaly=bool(decision.anomaly_z >= float(thresholds["unknown_z"])),
            model=f"patchcore-lite/{self.profile.gate_model or 'none'}",
            gate_trusted=decision.gate_trusted,
            gate_quality=self.profile.gate_quality,
            separation=self.profile.gate_separation,
            regions=[
                AnomalyRegionOut(bbox=list(r.bbox), score=round(float(r.score), 3), area=int(r.area))
                for r in list(regions)[:8]
            ],
            heatmap_url=None,
            cached=gate_cached,
        )
        result = FrameResult(
            frame_id="frm_" + uuid.uuid4().hex[:16],
            inspection_id=self.part.inspection_id,
            ts_ms=int(time.time() * 1000),
            stream_id=STREAM_IDS.get(self.source, "cam-1"),
            profile=self.profile.id,
            source=self.source,
            device=self._device_label(),
            engine=self._engine_label(),
            model_version=self._model_version(),
            threshold_profile=thresholds["id"],
            latency_ms=LatencyMs(
                capture=0.0,
                preprocess=0.0,
                infer_detector=round(det_ms, 2),
                infer_gate=round(gate_ms, 2),
                post=round(post_ms, 2),
                encode=0.0,
                total=round(total_ms, 2),
            ),
            detections=decision.detections,
            anomaly=anomaly_info,
            verdict=decision.verdict,  # type: ignore[arg-type]
            routing=decision.routing(),
            severity=decision.severity,
            simulated_plc=simulated_plc_action(decision.detections, decision.verdict, decision.severity),
            risk=explain_flagged_risk(self.profile.id, [d.model_dump(by_alias=True) for d in decision.detections], decision.verdict),
            part_state=PartState(
                index=(
                    ((self.part.index - 1) % self.sequence.total) + 1
                    if self.source == "dataset" and self.sequence.total
                    else self.part.index
                ),
                total=self.sequence.total if self.source == "dataset" else 0,
                item_id=self.part.inspection_id,
                kind=self.part.kind,
                ground_truth_class=self.part.gt_class,
                ground_truth_kind=self.part.gt_kind,
                frame_in_part=frame_in_part,
                frames_in_part=frames_total,
            ),
            fps=round(self.hub.fps, 2),
            frame_index=self.frame_index,
            warnings=warnings,
        )

        # The final frame is committed before it is published. PASS evidence is
        # still stored; REVIEW/FAIL get the same stills plus heatmap/crop.
        # Evidence stills keep the full burned-in provenance HUD so a still opened
        # outside the app can never be mistaken for a live camera capture.
        if final and self.part.persist:
            evidence_frame = render(
                frame,
                decision.detections,
                decision.verdict,  # type: ignore[arg-type]
                profile_id=self.profile.id,
                engine=det_info["engine"],
                source=self.source,
                gate_z=decision.anomaly_z,
                gate_trusted=decision.gate_trusted,
                gate_quality=self.profile.gate_quality,
                heatmap_bgr=heatmap,
                heatmap_gray=heatmap_gray,
                regions=regions,
                size=self.settings.stream_size,
                part_label=part_label,
                warning=warning_text,
                hud=True,
            )
            self._commit(result, frame, evidence_frame, decision, anomaly)

        self.hub.publish(result.model_dump(by_alias=True, mode="json"), jpeg)

    def _commit(self, result: FrameResult, source_frame: np.ndarray, overlay_frame: np.ndarray, decision: Any, anomaly: Any) -> None:
        assert self.part is not None
        evidence: dict[str, Any] = {}
        try:
            evidence = self.evidence.save(
                self.part.inspection_id,
                source_frame,
                overlay_frame,
                heatmap=anomaly.heatmap if anomaly is not None else None,
                detections=decision.detections,
                regions=anomaly.regions if anomaly is not None else None,
                meta={
                    "inspection_id": self.part.inspection_id,
                    "profile": self.profile.id,
                    "source": self.source,
                    "kind": self.part.kind,
                    "ground_truth_class": self.part.gt_class,
                    "ground_truth_kind": self.part.gt_kind,
                    "verdict": decision.verdict,
                    "reason": decision.reason,
                    "severity": decision.severity,
                    "route": decision.route,
                    "anomaly_z": decision.anomaly_z,
                    "detector": self.detector.info(),
                    "gate": self.gate.info(self.profile),
                    "warnings": decision.warnings,
                },
            )
            result.image_url = evidence.get("image_url")
            result.overlay_url = evidence.get("overlay_url")
            result.crop_url = evidence.get("crop_url")
            result.anomaly.heatmap_url = evidence.get("heatmap_url")
        except Exception as exc:
            decision.warnings.append(f"evidence_failed:{type(exc).__name__}")

        latency = float(np.mean(self.part.latencies)) if self.part.latencies else result.latency_ms.total
        payload = {
            "id": self.part.inspection_id,
            "final_verdict": decision.verdict,
            "severity": decision.severity,
            "route": decision.route,
            "reason": decision.reason,
            "anomaly_z": decision.anomaly_z,
            "anomaly_raw": decision.anomaly_raw,
            "latency_total_ms": round(latency, 2),
            "frame_count": self.part.accumulator.frame_count,
            "detections": [d.model_dump(by_alias=True) for d in decision.detections],
            "warnings": decision.warnings,
        }
        outbox = [
            {
                "event_type": "inspection.closed",
                "payload": {
                    "inspection_id": self.part.inspection_id,
                    "profile": self.profile.id,
                    "verdict": decision.verdict,
                    "severity": decision.severity,
                    "route": decision.route,
                    "reason": decision.reason,
                    "anomaly_z": decision.anomaly_z,
                    "ground_truth_kind": self.part.gt_kind,
                    "image_url": result.image_url,
                    "ts_ms": result.ts_ms,
                },
            }
        ]
        if decision.route in {"rework", "quarantine"}:
            outbox.append(
                {
                    "event_type": "mes.route",
                    "payload": {
                        "inspection_id": self.part.inspection_id,
                        "route": decision.route,
                        "severity": decision.severity,
                        "reason": decision.reason,
                    },
                }
            )
        try:
            self.db.finalize_inspection(payload, [d.model_dump(by_alias=True) for d in decision.detections], evidence, outbox)
        except Exception as exc:
            decision.warnings.append(f"db_finalize_failed:{type(exc).__name__}:{exc}")
            self.warnings.append(f"db_finalize_failed:{type(exc).__name__}:{exc}")

    # ------------------------------------------------------------------ helpers
    def _load_image(self, path: str) -> np.ndarray | None:
        if self._image_path == path and self._current_image is not None:
            return self._current_image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        self._image_path = path
        self._current_image = img
        return img

    def _gate_every(self) -> int:
        if self.settings.gate_every > 0:
            return self.settings.gate_every
        if self.gate.device != "cpu":
            return 1
        return 2 if self.profile.detector else 1

    def _device_label(self) -> str:
        devices = {self.detector.device, self.gate.device}
        return "cuda" if "cuda" in devices else "cpu"

    def _engine_label(self) -> str:
        gate = f"patchcore-lite/{self.profile.gate_model}" if self.gate.ready else "gate-off"
        return f"{self.detector.engine} + {gate}"

    def _model_version(self) -> str:
        gate = self.profile.gate_model or "no-gate"
        return f"{self.detector.model_version}+{gate}"

    def _pace(self) -> None:
        interval = 1.0 / max(0.1, self.settings.target_fps)
        now = time.perf_counter()
        if self._sleep_until <= 0 or now - self._sleep_until > interval * 3:
            self._sleep_until = now
        self._sleep_until += interval
        delay = self._sleep_until - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
