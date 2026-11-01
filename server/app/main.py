"""FastAPI application: live stream, review workflow, metrics, audit, MES outbox."""
from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac
import io
import json
import time
import uuid
import zipfile
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from server.app.config import Settings, settings
from server.app.auto_profile import AutoProfileClassifier, SUPPORTED_FAMILIES, confirm_near_miss
from server.app.db import Database
from server.app.demo_auth import DemoAuthMiddleware
from server.app.detector import _sha256
from server.app.evidence import EvidenceStore
from server.app.gate_service import GateService
from server.app.hub import Hub
from server.app.mes import OutboxWorker
from server.app.m2_body import M2BodyClassifier
from server.app.player import Pipeline
from server.app.profiles import ProfileRegistry
from server.app.risk_explanation import explain_flagged_risk
from server.app.vlm_advisor import LocalVLM
from server.app.schemas import (
    ControlRequest,
    DatasetReport,
    HealthInfo,
    InspectionSummary,
    ReviewAction,
    ThresholdProfile,
    ThresholdUpdate,
)

START_TIME = time.time()
runtime: dict[str, Any] = {}


def _public_inspection(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    data["detections"] = data.get("detections", data.pop("detections_json", [])) or []
    data["warnings"] = data.get("warnings_json", []) or []
    data.pop("warnings_json", None)
    return data


def _validation_checks(settings: Settings, registry: ProfileRegistry) -> list[str]:
    warnings: list[str] = []
    profile = registry.get(settings.profile)
    if profile.gate_quality == "not_validated":
        warnings.append("gate_not_validated")
    if settings.detector_mode == "fixture" or (
        settings.detector_mode == "auto" and settings.promoted_weights is None
    ):
        warnings.append("fixture_detector")
    return warnings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    registry = ProfileRegistry(settings)
    hub = Hub()
    hub.bind_loop(asyncio.get_running_loop())
    gate = GateService(settings)
    evidence = EvidenceStore(settings)
    pipeline = Pipeline(settings, registry, db, hub, evidence, gate)
    report_marker = settings.artifacts_dir / "current_report.json"
    try:
        current_report = json.loads(report_marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current_report = None
    auto_profile = AutoProfileClassifier(settings.auto_profile_model)
    m2_body = M2BodyClassifier(settings.m2_body_weights)
    vlm = LocalVLM(settings)
    worker = OutboxWorker(settings, db)
    runtime.update(
        {
            "db": db,
            "registry": registry,
            "hub": hub,
            "gate": gate,
            "evidence": evidence,
            "pipeline": pipeline,
            "worker": worker,
            "started_at": time.time(),
        }
    )
    app.state.db = db
    app.state.registry = registry
    app.state.hub = hub
    app.state.gate = gate
    app.state.evidence = evidence
    app.state.pipeline = pipeline
    app.state.current_report = current_report
    app.state.auto_profile_classifier = auto_profile
    app.state.m2_body_classifier = m2_body
    app.state.vlm = vlm
    app.state.vlm_cache = {}
    app.state.worker = worker
    _seed_models(settings, db, registry)
    pipeline.start()
    worker.start()
    try:
        yield
    finally:
        pipeline.stop()
        worker.stop()
        db.conn.close()


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_registry(request: Request) -> ProfileRegistry:
    return request.app.state.registry


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def get_hub(request: Request) -> Hub:
    return request.app.state.hub


def _seed_models(settings: Settings, db: Database, registry: ProfileRegistry) -> None:
    now = int(time.time() * 1000)
    weights = settings.promoted_weights
    if weights is not None:
        try:
            db.upsert_model(
                {
                    "id": "detector_yolo11s",
                    "kind": "detector",
                    "name": weights.name,
                    "path": str(weights),
                    "sha256": _sha256(weights),
                    "metrics": {"classes": 5, "dataset": "NEU-DET", "status": "promoted"},
                    "created_at_ms": now,
                }
            )
        except Exception:
            pass
    else:
        db.upsert_model(
            {
                "id": "detector_fixture",
                "kind": "detector",
                "name": "Fixture ground-truth labels",
                "path": "",
                "sha256": "",
                "metrics": {"dataset": "NEU-DET", "status": "demo_fixture", "note": "No promoted weights"},
                "created_at_ms": now,
            }
        )
    for profile in registry.all():
        if not profile.gate_path or not profile.gate_path.exists():
            continue
        gate = registry.gate_entry(profile.id)
        db.upsert_model(
            {
                "id": f"gate_{profile.id}",
                "kind": "gate",
                "name": profile.gate_model or profile.id,
                "path": str(profile.gate_path),
                "sha256": _sha256(profile.gate_path),
                "metrics": {
                    "quality": profile.gate_quality,
                    "trusted": profile.gate_trusted,
                    "separation": profile.gate_separation,
                    "check": gate.get("check", {}),
                    "license": profile.license,
                },
                "created_at_ms": now,
            }
        )


app = FastAPI(
    title="Inspectra",
    version=settings.version,
    description="Real-time surface inspection with YOLO11s + PatchCore-lite novelty fusion.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(DemoAuthMiddleware)


@app.get("/api/health", response_model=HealthInfo)
async def health(request: Request) -> HealthInfo:
    db = get_db(request)
    pipeline = get_pipeline(request)
    hub = get_hub(request)
    state = pipeline.state()
    warnings = list(state.get("warnings", []))
    warnings += _validation_checks(settings, get_registry(request))
    db_counts = db.counts()
    db_counts["current_open_reviews"] = len(_current_review_rows(request))
    return HealthInfo(
        status="ok" if pipeline._thread and pipeline._thread.is_alive() else "degraded",
        version=settings.version,
        uptime_s=round(time.time() - runtime.get("started_at", START_TIME), 1),
        profile=pipeline.profile.id,
        source=pipeline.source,
        detector_mode=settings.detector_mode,
        detector_engine=pipeline.detector.engine,
        gate_ready=pipeline.gate.ready,
        gate_trusted=pipeline.profile.gate_trusted,
        device=state.get("device", "cpu"),
        db=db_counts,
        fps=round(hub.fps, 2),
        ws_clients=hub.ws_clients,
        warnings=sorted(set(warnings)),
    )


@app.get("/api/m2-body/status")
async def m2_body_status(request: Request) -> dict[str, Any]:
    classifier: M2BodyClassifier = request.app.state.m2_body_classifier
    report_path = classifier.weights.with_name("report.json")
    metrics: dict[str, Any] | None = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            metrics = {
                "test_images": report["test_images"],
                "test_auc": report["test_auc"],
                "test_at_operating_threshold": report["test_at_operating_threshold"],
            }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return {
        "available": classifier.available,
        "task": "M2 screw Body OK/NG classification",
        "scope": "Only 1024×1280 images from the M2 Body camera setup; no defect subtype or box",
        "metrics": metrics,
    }


@app.put("/api/m2-body/classify")
async def classify_m2_body(request: Request) -> dict[str, Any]:
    """Classify a source-format M2 Body image without changing live inspection state."""
    import cv2
    import numpy as np

    classifier: M2BodyClassifier = request.app.state.m2_body_classifier
    if not classifier.available:
        raise HTTPException(503, "M2 Body classifier checkpoint is unavailable on this server")
    max_bytes = 20 * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(413, "image exceeds the 20 MB limit")
        except ValueError as exc:
            raise HTTPException(400, "invalid Content-Length header") from exc
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(413, "image exceeds the 20 MB limit")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        raise HTTPException(400, "image could not be decoded")
    try:
        return await run_in_threadpool(classifier.classify, image)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/profiles")
async def list_profiles(request: Request) -> list[dict[str, Any]]:
    registry = get_registry(request)
    active = get_pipeline(request).profile.id
    out = []
    for profile in registry.all():
        data = profile.info().model_dump()
        data["active"] = profile.id == active
        if get_pipeline(request).settings.detector_mode == "unified":
            data["detector"] = True
        out.append(data)
    return out


@app.get("/api/profiles/{profile_id}")
async def profile_detail(profile_id: str, request: Request) -> dict[str, Any]:
    registry = get_registry(request)
    try:
        profile = registry.get(profile_id)
    except KeyError:
        raise HTTPException(404, "profile not found")
    samples: dict[str, list[dict[str, Any]]] = {}
    for kind in ("good", "defect", "unknown"):
        items = profile.items(kind)
        samples[kind] = [
            {
                **item.public(),
                "index": idx,
                "image_url": f"/api/datasets/image?profile={profile.id}&kind={kind}&index={idx}",
            }
            for idx, item in enumerate(items[:6])
        ]
    gate = registry.gate_entry(profile.id)
    info = profile.info().model_dump()
    if get_pipeline(request).settings.detector_mode == "unified":
        info["detector"] = True
    return {
        "info": info,
        "samples": samples,
        "gate": {
            "quality": profile.gate_quality,
            "trusted": profile.gate_trusted,
            "separation": profile.gate_separation,
            "check": gate.get("check", {}),
            "model": profile.gate_model,
        },
        "license": profile.license,
        "heldout_class": profile.heldout_class,
    }


@app.get("/api/gate/summary")
async def gate_summary(request: Request) -> dict[str, Any]:
    registry = get_registry(request)
    pipeline = get_pipeline(request)
    return {
        "summary": registry.gate_summary(),
        "active": {
            "profile": pipeline.profile.id,
            "quality": pipeline.profile.gate_quality,
            "trusted": pipeline.profile.gate_trusted,
            "separation": pipeline.profile.gate_separation,
            "ready": pipeline.gate.ready,
            "device": pipeline.gate.device,
            "error": pipeline.gate.error,
        },
        "profiles": [
            {
                "id": p.id,
                "quality": p.gate_quality,
                "trusted": p.gate_trusted,
                "separation": p.gate_separation,
                "model": p.gate_model,
            }
            for p in registry.all()
        ],
    }


@app.get("/api/datasets/verify", response_model=DatasetReport)
async def verify_datasets(request: Request, force: bool = False) -> dict[str, Any]:
    return get_registry(request).verify(force=force)


@app.get("/api/datasets/unified")
async def unified_dataset_summary(request: Request) -> dict[str, Any]:
    """Present the prepared YOLO dataset separately from legacy source checks."""
    manifest_path = settings.artifacts_dir / "datasets" / "unified_defects" / "manifest.json"
    eval_path = settings.artifacts_dir / "eval" / "unified_yolo11s_640_test.json"
    if not manifest_path.is_file():
        return {"available": False, "active": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else {}
    pipeline = get_pipeline(request)
    return {
        "available": True,
        "active": settings.detector_mode == "unified" and pipeline.detector.info().get("ready", False),
        "engine": pipeline.detector.engine,
        "class_count": manifest.get("class_count", 0),
        "classes": list(manifest.get("classes", {}).values()),
        "images": manifest.get("images", {}),
        "by_profile": manifest.get("by_profile", {}),
        "test_overall": evaluation.get("test_overall", {}),
        "test_images": evaluation.get("test_images"),
        "split_note": manifest.get("mvtec_defect_split_note", ""),
    }


@app.get("/api/eval/summary")
async def eval_summary() -> dict[str, Any]:
    """Serve locked offline measurements; never mixes them with replay metrics."""
    eval_dir = settings.artifacts_dir / "eval"
    metrics_path = eval_dir / "test_metrics.json"
    if not metrics_path.exists():
        return {"available": False, "montage_url": None, "operating_points": []}
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "operating_points": []}

    points: list[dict[str, Any]] = []
    scan_status = metrics.get("operating_scan_status", "legacy")
    scan_path = eval_dir / "conf_scan.json"
    if scan_status == "current" and scan_path.exists():
        try:
            scan = json.loads(scan_path.read_text(encoding="utf-8"))

            def hit_rate(rows: list[Any], floor: float) -> float:
                if not rows:
                    return 0.0
                hits = sum(1 for boxes in rows if any(float(conf) >= floor for _, conf in boxes))
                return hits / len(rows)

            for floor in (0.25, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65):
                points.append(
                    {
                        "conf": floor,
                        "test_hit": hit_rate(scan.get("test", []), floor),
                        "heldout_hit": hit_rate(scan.get("heldout", []), floor),
                        "good_fp": hit_rate(scan.get("good", []), floor),
                    }
                )
        except Exception:
            points = []

    montage = eval_dir / "good_fp_montage.jpg"
    return {
        "available": True,
        "metrics": metrics,
        "operating_points": points,
        "operating_scan_status": scan_status,
        "operating_scan_note": (
            "Operating-point scan pending a re-run on the promoted weights."
            if scan_status != "current"
            else ""
        ),
        "montage_url": "/eval/good_fp_montage.jpg" if montage.exists() else None,
    }


@app.get("/api/datasets/image")
async def dataset_image(
    request: Request,
    profile: str,
    kind: str,
    index: int = Query(0, ge=0),
) -> FileResponse:
    registry = get_registry(request)
    try:
        p = registry.get(profile)
        items = p.items(kind)
    except (KeyError, ValueError):
        raise HTTPException(404, "profile or kind not found")
    if index >= len(items):
        raise HTTPException(404, "sample index out of range")
    path = Path(items[index].path).resolve()
    try:
        path.relative_to(settings.data_dir.resolve())
    except ValueError:
        raise HTTPException(403, "sample is outside the generated data directory")
    if not path.exists():
        raise HTTPException(404, "sample image missing")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/frames/latest")
async def latest_frame(request: Request) -> dict[str, Any]:
    hub = get_hub(request)
    return {"seq": hub.seq, "frame": hub.latest, "fps": round(hub.fps, 2)}


@app.put("/api/video")
@app.put("/api/input")
async def upload_demo_input(request: Request, filename: str = Query(..., min_length=1, max_length=255)) -> dict[str, Any]:
    """Store a selected image or video and run it through the live pipeline."""
    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    safe_name = Path(filename.replace("\\", "/")).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in video_suffixes | image_suffixes:
        raise HTTPException(415, "unsupported file type; use JPG, PNG, BMP, WEBP, MP4, AVI, MOV, MKV, or M4V")
    source = "image" if suffix in image_suffixes else "video"
    video_frame_count = 0
    if request.url.path == "/api/video" and source != "video":
        raise HTTPException(415, "the /api/video endpoint accepts video files only")

    max_bytes = (20 if source == "image" else 500) * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(413, "file exceeds the upload limit")
        except ValueError as exc:
            raise HTTPException(400, "invalid Content-Length header") from exc

    pipeline = get_pipeline(request)
    classifier: AutoProfileClassifier = request.app.state.auto_profile_classifier
    if not classifier.ready:
        raise HTTPException(503, "automatic product identification model is missing; train it with scripts/train_auto_profile.py")

    input_dir = settings.artifacts_dir / "demo_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    destination = input_dir / f"demo_{uuid.uuid4().hex}{suffix}"
    written = 0
    try:
        with destination.open("wb") as output:
            async for chunk in request.stream():
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, "file exceeds the upload limit")
                output.write(chunk)
        if written == 0:
            raise HTTPException(400, "uploaded file is empty")
        import cv2
        if source == "image":
            first_frame = cv2.imread(str(destination))
            if first_frame is None:
                raise HTTPException(400, "image could not be decoded")
        else:
            capture = cv2.VideoCapture(str(destination))
            readable = capture.isOpened()
            video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
            has_frame, first_frame = capture.read() if readable else (False, None)
            capture.release()
            if not readable or not has_frame:
                raise HTTPException(400, "video has no decodable frames; try MP4 or AVI")
        identification = classifier.identify(first_frame)
        if (
            identification["status"] == "uncertain"
            and identification.get("candidate") in SUPPORTED_FAMILIES
            and 0.45 <= float(identification.get("confidence") or 0) < classifier.confidence_floor
        ):
            vlm: LocalVLM = request.app.state.vlm
            if (await run_in_threadpool(vlm.status))["ready"]:
                try:
                    observation = await run_in_threadpool(vlm.identify_product, first_frame)
                    identification = confirm_near_miss(identification, observation)
                except (RuntimeError, ValueError):
                    pass  # Preserve the uncertain result when the local VLM fails.
        if identification["status"] != "identified":
            # Product identity is a prerequisite for choosing a defect profile.
            # Preserve the input and ask a person rather than guessing a family.
            now_ms = int(time.time() * 1000)
            inspection_id = "insp_" + uuid.uuid4().hex[:20]
            db = get_db(request)
            evidence = request.app.state.evidence.save(
                inspection_id, first_frame, first_frame,
                meta={
                    "reason": "product_not_recognized",
                    "identification": identification,
                    "filename": safe_name,
                    "source": source,
                    "stored_upload": destination.name,
                    "note": "First video frame only" if source == "video" else "Uploaded image",
                },
            )
            db.create_inspection({
                "id": inspection_id, "ts_ms": now_ms, "profile": "unrecognized",
                "source": source, "status": "open", "verdict": "review",
                "final_verdict": None, "severity": 2, "route": "human_review",
                "reason": "product_not_recognized", "anomaly_z": 0.0,
                "anomaly_raw": 0.0, "latency_total_ms": 0.0,
                "frame_count": 1, "detections_json": [],
                "model_version": "auto_profile_classifier",
                "threshold_profile": db.active_threshold()["id"],
            })
            db.finalize_inspection(
                {
                    "id": inspection_id, "profile": "unrecognized",
                    "final_verdict": "review", "severity": 2,
                    "route": "human_review", "reason": "product_not_recognized",
                    "frame_count": 1, "detections": [],
                    "warnings": ["product_not_recognized", "defect_detection_not_run"],
                },
                [], evidence, [],
            )
            marker = {
                "filename": safe_name, "source": source,
                "profile": "unrecognized", "started_at_ms": now_ms,
                "max_inspections": 1,
                "inspection_id": inspection_id,
            }
            (settings.artifacts_dir / "current_report.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            request.app.state.current_report = marker
            request.app.state.vlm_cache = {}
            return {
                "ok": True, "review_required": True,
                "inspection_id": inspection_id, "filename": safe_name,
                "bytes": written, "source": source,
                "profile": "unrecognized", "reason": "product_not_recognized",
                "auto_identification": identification,
                "evidence_url": evidence["image_url"],
                "message": "Product not recognized. Upload sent to manual review; defect detection was not run.",
            }
        selected_profile = str(identification["profile"])
        if selected_profile not in SUPPORTED_FAMILIES:
            raise HTTPException(422, "Only steel, screw, metal nut, transistor, and bottle are supported")
        profile = get_registry(request).get(selected_profile)
        if (profile.detector or pipeline.settings.detector_mode == "unified") and (
            pipeline.settings.detector_mode not in {"real", "real-run", "unified"}
            or pipeline.settings.detector_weights is None
        ):
            raise HTTPException(409, "YOLO weights are unavailable. Start the server with INDORE_DETECTOR=real for the steel-only model or INDORE_DETECTOR=unified for the 21-class candidate.")
        if source == "image":
            pipeline.settings.image_path = str(destination)
        else:
            pipeline.settings.video_path = str(destination)
        pipeline.input_warning = str(identification.get("domain_warning") or "")
        pipeline.command("pause")
        pipeline.command("profile", selected_profile)
        pipeline.command("source", source)
        max_inspections = 1 if source == "image" else (video_frame_count // max(1, settings.part_frames) if video_frame_count else None)
        marker = {"filename": safe_name, "source": source, "profile": selected_profile, "started_at_ms": int(time.time() * 1000), "max_inspections": max_inspections}
        if identification.get("domain_warning"):
            marker["domain_warning"] = identification["domain_warning"]
        (settings.artifacts_dir / "current_report.json").write_text(json.dumps(marker), encoding="utf-8")
        request.app.state.current_report = marker
        request.app.state.vlm_cache = {}
        state = pipeline.command("resume")
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, f"could not select input: {type(exc).__name__}") from exc
    return {
        "ok": True, "filename": safe_name, "bytes": written, "source": source,
        "auto_identification": identification, "profile": selected_profile, "state": state,
    }


@app.get("/api/frames/recent")
async def recent_frames(request: Request) -> dict[str, Any]:
    hub = get_hub(request)
    return {"frames": hub.recent[-60:], "fps": round(hub.fps, 2)}


def _report_rows(request: Request) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    marker = request.app.state.current_report
    if not marker:
        raise HTTPException(404, "Upload an image or video to create an inspection report")
    first_pass_count = marker.get("max_inspections")
    if first_pass_count is None and marker["source"] == "image":
        first_pass_count = 1
    limit = min(2000, first_pass_count) if first_pass_count is not None else 2000
    rows = get_db(request).list_closed_since(marker["started_at_ms"], marker["profile"], marker["source"], limit=limit + 1)
    return marker, rows[:limit], len(rows) > 2000


def _current_review_rows(request: Request) -> list[dict[str, Any]]:
    if not request.app.state.current_report:
        return []
    _, rows, _ = _report_rows(request)
    return [row for row in rows if (row.get("final_verdict") or row.get("verdict")) in {"review", "fail"}
            and row.get("operator_disposition") is None]


def _report_summary(marker: dict[str, Any], rows: list[dict[str, Any]], truncated: bool) -> dict[str, Any]:
    classes: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    for row in rows:
        decisions[row.get("final_verdict") or row.get("verdict") or "unknown"] += 1
        for detection in row.get("detections") or []:
            classes[detection.get("class") or detection.get("class_name") or "unknown"] += 1
    flagged = [row for row in rows if (row.get("final_verdict") or row.get("verdict")) in {"review", "fail"} and row.get("detections")]
    top_row = max(flagged, key=lambda row: (int(row.get("severity") or 1), len(row.get("detections") or []))) if flagged else None
    return {
        "filename": marker["filename"], "profile": marker["profile"], "source": marker["source"],
        "domain_warning": marker.get("domain_warning"),
        "risk": explain_flagged_risk(marker["profile"], top_row["detections"], top_row.get("final_verdict") or top_row.get("verdict")) if top_row else None,
        "started_at_ms": marker["started_at_ms"], "inspections": len(rows),
        "first_pass_only": marker.get("max_inspections") is not None,
        "detections": sum(classes.values()), "counts_by_class": dict(classes),
        "decisions": dict(decisions), "truncated": truncated,
        "avg_processing_ms": round(sum(float(row.get("latency_total_ms") or 0) for row in rows) / len(rows), 1) if rows else 0,
    }


def _evidence_path(url: str | None) -> Path | None:
    if not url or not url.startswith("/media/"):
        return None
    relative = url.removeprefix("/media/").removeprefix("evidence/")
    root = settings.evidence_dir.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path if path.is_file() else None


def _report_csv(summary: dict[str, Any], rows: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(["filename", "profile", "source", "inspection_id", "timestamp_ms", "decision", "severity", "route", "class", "confidence", "class_total", "inspections_total", "detections_total", "pass_total", "review_total", "fail_total", "processing_ms", "source_image", "annotated_image", "crop_image", "potential_concern", "possible_later_failure", "time_to_failure", "needed_for_life_estimate"])
    for row in rows or [{}]:
        detections = row.get("detections") or [{}]
        for detection in detections:
            name = detection.get("class") or detection.get("class_name") or ""
            risk = explain_flagged_risk(summary["profile"], [detection] if name else [], row.get("final_verdict") or row.get("verdict") or "") or {}
            writer.writerow([
                summary["filename"], summary["profile"], summary["source"], row.get("id", ""), row.get("ts_ms", ""),
                row.get("final_verdict") or row.get("verdict") or "", row.get("severity", ""), row.get("route", ""),
                name, detection.get("conf", ""), summary["counts_by_class"].get(name, 0),
                summary["inspections"], summary["detections"], summary["decisions"].get("pass", 0),
                summary["decisions"].get("review", 0), summary["decisions"].get("fail", 0),
                row.get("latency_total_ms", ""),
                ("/media/" + row["image_url"].removeprefix("/media/").removeprefix("evidence/")) if row.get("image_url") else "",
                ("/media/" + row["overlay_url"].removeprefix("/media/").removeprefix("evidence/")) if row.get("overlay_url") else "",
                ("/media/" + row["crop_url"].removeprefix("/media/").removeprefix("evidence/")) if row.get("crop_url") else "",
                risk.get("immediate_concern", ""), risk.get("if_used", ""), risk.get("time_to_failure", ""), risk.get("needed_for_estimate", ""),
            ])
    return buf.getvalue().encode("utf-8-sig")


@app.get("/api/report")
async def inspection_report(request: Request) -> dict[str, Any]:
    marker, rows, truncated = _report_rows(request)
    return _report_summary(marker, rows, truncated)


@app.get("/api/vlm/status")
async def vlm_status(request: Request) -> dict[str, Any]:
    return await run_in_threadpool(request.app.state.vlm.status)


@app.post("/api/vlm/advice")
async def vlm_advice(request: Request) -> dict[str, Any]:
    marker, rows, _ = _report_rows(request)
    if not rows:
        raise HTTPException(409, "Wait for the first completed inspection before requesting VLM advice")
    chosen = max(rows, key=lambda row: (row.get("severity") or 0, len(row.get("detections") or [])))
    inspection_id = chosen["id"]
    cached = request.app.state.vlm_cache.get(inspection_id)
    if cached:
        return cached
    image = _evidence_path(chosen.get("image_url"))
    if image is None:
        raise HTTPException(404, "Inspection image evidence is unavailable")
    vlm: LocalVLM = request.app.state.vlm
    status = await run_in_threadpool(vlm.status)
    if not status["ready"]:
        raise HTTPException(503, f"{status['message']}: {status['model']}")
    try:
        result = await run_in_threadpool(vlm.advise, image, marker["profile"], chosen.get("detections") or [], inspection_id, bool(marker.get("domain_warning")))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc
    request.app.state.vlm_cache[inspection_id] = result
    return result


@app.get("/api/report.csv")
async def inspection_report_csv(request: Request) -> Response:
    marker, rows, truncated = _report_rows(request)
    content = _report_csv(_report_summary(marker, rows, truncated), rows)
    return Response(content=content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=indore-inspection-report.csv"})


@app.get("/api/report.zip")
async def inspection_report_zip(request: Request) -> Response:
    marker, rows, truncated = _report_rows(request)
    summary = _report_summary(marker, rows, truncated)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("inspections.csv", _report_csv(summary, rows))
        archive.writestr("summary.json", json.dumps(summary, indent=2))
        for row in rows:
            for key in ("image_url", "overlay_url", "crop_url", "heatmap_url"):
                path = _evidence_path(row.get(key))
                if path:
                    archive.write(path, f"evidence/{path.name}")
    return Response(content=output.getvalue(), media_type="application/zip", headers={"Content-Disposition": "attachment; filename=indore-inspection-report.zip"})


DEMO_FAMILIES = ("steel", "metal_nut", "screw", "transistor", "bottle")


def _demo_image_path(family: str, kind: str) -> Path:
    if family not in DEMO_FAMILIES or kind not in {"defect", "clean"}:
        raise HTTPException(404, "demo image not found")
    root = settings.artifacts_dir / "demo_images"
    candidates = sorted((root / "recommended").glob(f"{family}_*.jpg")) if kind == "defect" else sorted((root / family).glob("clean--*.jpg"))
    if not candidates:
        raise HTTPException(404, "demo image not found")
    return candidates[0]


@app.get("/api/demo/catalog")
async def demo_catalog() -> dict[str, Any]:
    items = []
    for family in DEMO_FAMILIES:
        try:
            defect = _demo_image_path(family, "defect")
            clean = _demo_image_path(family, "clean")
        except HTTPException:
            continue
        items.append({
            "family": family, "defect_label": defect.stem.removeprefix(family + "_").replace("_", " "),
            "defect_url": f"/api/demo/image/{family}/defect",
            "clean_url": f"/api/demo/image/{family}/clean",
            "defect_filename": defect.name, "clean_filename": clean.name,
        })
    return {"items": items, "note": "Curated held-out examples for demonstration, not an unbiased accuracy estimate."}


@app.get("/api/demo/image/{family}/{kind}")
async def demo_image(family: str, kind: str) -> FileResponse:
    return FileResponse(_demo_image_path(family, kind), media_type="image/jpeg")


@app.get("/live.mjpg")
async def live_mjpeg(request: Request) -> StreamingResponse:
    hub = get_hub(request)
    boundary = "indoreframe"

    async def generator() -> AsyncIterator[bytes]:
        last_seq = -1
        while True:
            if await request.is_disconnected():
                break
            seq = hub.seq
            jpeg = hub.latest_jpeg
            if jpeg and seq != last_seq:
                last_seq = seq
                yield (
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode(
                        "ascii"
                    )
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(
        generator(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    await websocket.accept()
    hub: Hub = websocket.app.state.hub
    queue = await hub.subscribe()
    try:
        if hub.latest is not None:
            await websocket.send_json({"type": "frame", "frame": hub.latest})
        while True:
            frame = await queue.get()
            await websocket.send_json({"type": "frame", "frame": frame})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        hub.unsubscribe(queue)


@app.get("/api/control/state")
async def control_state(request: Request) -> dict[str, Any]:
    registry = get_registry(request)
    pipeline = get_pipeline(request)
    state = pipeline.state()
    state["profiles"] = [
        {
            "id": p.id,
            "name": p.display_name,
            "detector": p.detector or pipeline.settings.detector_mode == "unified",
            "gate_quality": p.gate_quality,
            "gate_trusted": p.gate_trusted,
        }
        for p in registry.all()
    ]
    state["sequence_modes"] = ["mixed", "good", "defect", "unknown"]
    state["detector_modes"] = ["auto", "real", "real-run", "unified", "fixture", "off"]
    return state


@app.post("/api/control")
async def control(body: ControlRequest, request: Request) -> dict[str, Any]:
    pipeline = get_pipeline(request)
    try:
        state = pipeline.command(body.action, body.value)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "state": state}


@app.get("/api/inspections")
async def inspections(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    verdict: str | None = None,
    profile: str | None = None,
    status: str = "closed",
) -> dict[str, Any]:
    db = get_db(request)
    status_filter = None if status == "all" else status
    rows = db.list_inspections(limit=limit, offset=offset, verdict=verdict, profile=profile, status=status_filter)
    total = db.inspection_count(verdict=verdict, profile=profile, status=status_filter)
    return {
        "items": [_public_inspection(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/inspections/{inspection_id}")
async def inspection_detail(inspection_id: str, request: Request) -> dict[str, Any]:
    row = get_db(request).get_inspection(inspection_id)
    if row is None:
        raise HTTPException(404, "inspection not found")
    return _public_inspection(row) or {}


@app.get("/api/review/queue")
async def review_queue(request: Request, limit: int = Query(100, ge=1, le=500), scope: Literal["current", "all"] = "current") -> dict[str, Any]:
    db = get_db(request)
    rows = _current_review_rows(request) if scope == "current" else db.review_queue(limit)
    counts = db.counts()
    return {
        "items": [_public_inspection(r) for r in rows[:limit]],
        "count": min(len(rows), limit),
        "open_reviews": len(rows) if scope == "current" else counts["open_reviews"],
        "historical_open_reviews": counts["open_reviews"],
        "scope": scope,
    }


@app.post("/api/inspections/{inspection_id}/review")
async def review_inspection(inspection_id: str, body: ReviewAction, request: Request) -> dict[str, Any]:
    db = get_db(request)
    session_user = getattr(request.state, "auth_user", None)
    actor = session_user["username"] if session_user else (body.actor or settings.actor)
    row = db.review(inspection_id, body.model_dump(exclude_none=True), actor)
    if row is None:
        raise HTTPException(404, "inspection not found")
    return _public_inspection(row) or {}


@app.get("/api/metrics")
async def metrics(request: Request, profile: str | None = None) -> dict[str, Any]:
    db = get_db(request)
    registry = get_registry(request)
    payload = db.metrics(profile=profile)
    payload["gate"] = {
        p.id: {
            "quality": p.gate_quality,
            "trusted": p.gate_trusted,
            "separation": p.gate_separation,
            "model": p.gate_model,
        }
        for p in registry.all()
    }
    payload["warnings"] = _validation_checks(settings, registry)
    payload["counts"] = db.counts()
    return payload


@app.get("/api/audit")
async def audit(request: Request, limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    return {"items": get_db(request).audit(limit)}


@app.get("/api/outbox")
async def outbox(request: Request, limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    db = get_db(request)
    worker = request.app.state.worker
    return {"items": db.outbox(limit), "status": worker.status(), "counts": db.counts()}


@app.get("/api/outbox/status")
async def outbox_status(request: Request) -> dict[str, Any]:
    db = get_db(request)
    return {"status": request.app.state.worker.status(), "counts": db.counts()}


@app.post("/api/outbox/flush")
async def outbox_flush(request: Request) -> dict[str, Any]:
    worker = request.app.state.worker
    count = worker.flush_simulated(settings.actor)
    return {"ok": True, "flushed": count, "status": worker.status()}


@app.post("/api/outbox/{outbox_id}/retry")
async def outbox_retry(outbox_id: int, request: Request) -> dict[str, Any]:
    get_db(request).retry_outbox(outbox_id)
    get_db(request).audit_event(settings.actor, "outbox.retry", "outbox", str(outbox_id), {})
    return {"ok": True}


@app.get("/api/mes/events")
async def mes_events(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    return {"items": get_db(request).mes_events(limit)}


@app.post("/api/mes/events")
async def mes_event(request: Request) -> dict[str, Any]:
    body = await request.body()
    signature = request.headers.get("x-indore-signature", "")
    expected = "sha256=" + hmac.new(
        settings.webhook_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(401, "invalid HMAC signature")
    try:
        data = json.loads(body or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "invalid JSON")
    event_type = str(data.get("event_type", "unknown"))
    payload = data.get("payload", data)
    get_db(request).add_mes_event(event_type, payload if isinstance(payload, dict) else {"value": payload}, signature)
    get_db(request).audit_event("mes", "mes.event.received", "mes_event", event_type, {"payload": payload})
    return {"ok": True, "event_type": event_type, "received_at_ms": int(time.time() * 1000)}


@app.get("/api/thresholds", response_model=ThresholdProfile)
async def thresholds(request: Request) -> dict[str, Any]:
    return get_db(request).active_threshold()


@app.patch("/api/thresholds", response_model=ThresholdProfile)
async def update_thresholds(body: ThresholdUpdate, request: Request) -> dict[str, Any]:
    db = get_db(request)
    patch = body.model_dump(exclude_none=True)
    current = db.active_threshold()
    for key in ("pass_max", "review_max", "unknown_z", "conf_min"):
        if key in patch:
            value = float(patch[key])
            if not 0.0 <= value <= 3.0:
                raise HTTPException(400, f"{key} must be between 0 and 3")
            current[key] = value
    if not 0.0 < current["pass_max"] < current["review_max"]:
        raise HTTPException(400, "require 0 < pass_max < review_max")
    if current["review_max"] > current["unknown_z"]:
        raise HTTPException(400, "review_max must be <= unknown_z")
    if patch.get("name"):
        current["name"] = str(patch["name"])[:80]
    updated = db.update_threshold(current, settings.actor)
    return updated


@app.get("/api/models")
async def models(request: Request) -> dict[str, Any]:
    db = get_db(request)
    registry = get_registry(request)
    pipeline = get_pipeline(request)
    artifacts = []
    for path in sorted(settings.models_dir.glob("*")):
        if path.suffix.lower() not in {".pt", ".onnx", ".npz", ".json"}:
            continue
        artifacts.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "modified_ms": int(path.stat().st_mtime * 1000),
                "kind": "gate" if path.suffix == ".npz" else "detector" if path.suffix in {".pt", ".onnx"} else "metadata",
            }
        )
    return {
        "versions": db.model_versions(),
        "artifacts": artifacts,
        "runtime": {
            "detector": pipeline.detector.info(),
            "gate": pipeline.gate.info(pipeline.profile),
            "device": pipeline.state().get("device"),
            "engine": pipeline.state().get("engine"),
            "promoted_weights": str(settings.promoted_weights) if settings.promoted_weights else None,
            "training_weights": str(settings.training_weights) if settings.training_weights.exists() else None,
        },
        "profiles": [
            {
                "id": p.id,
                "gate_model": p.gate_model,
                "gate_quality": p.gate_quality,
                "gate_trusted": p.gate_trusted,
                "separation": p.gate_separation,
            }
            for p in registry.all()
        ],
    }


@app.get("/api/config")
async def config(request: Request) -> dict[str, Any]:
    pipeline = get_pipeline(request)
    return {
        "version": settings.version,
        "profile": pipeline.profile.id,
        "source": pipeline.source,
        "detector_mode": settings.detector_mode,
        "device": pipeline.state().get("device"),
        "imgsz": settings.imgsz,
        "conf": settings.conf,
        "iou": settings.iou,
        "part_frames": settings.part_frames,
        "target_fps": settings.target_fps,
        "stream_size": settings.stream_size,
        "jpeg_quality": settings.mjpeg_quality,
        "gate_every": settings.gate_every,
        "mes_configured": bool(settings.mes_url),
        "actor": settings.actor,
        "project_root": str(settings.project_root),
        "db_path": str(settings.db_path),
    }


@app.get("/api/export/inspections.csv")
async def export_csv(request: Request, profile: str | None = None) -> Response:
    db = get_db(request)
    rows = db.list_inspections(limit=100000, profile=profile, status="closed")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "id",
            "ts_ms",
            "profile",
            "source",
            "verdict",
            "severity",
            "route",
            "reason",
            "anomaly_z",
            "latency_ms",
            "ground_truth_kind",
            "ground_truth_class",
            "operator_disposition",
            "operator_label",
            "detections",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row["id"],
                row["ts_ms"],
                row["profile"],
                row["source"],
                row.get("final_verdict") or row.get("verdict"),
                row.get("severity"),
                row.get("route"),
                row.get("reason"),
                row.get("anomaly_z"),
                row.get("latency_total_ms"),
                row.get("ground_truth_kind"),
                row.get("ground_truth_class"),
                row.get("operator_disposition"),
                row.get("operator_label"),
                ";".join(f"{d.get('class')}:{d.get('conf')}" for d in (row.get("detections") or [])),
            ]
        )
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=indore-inspections.csv"},
    )


# --------------------------------------------------------------------- static web
app.mount("/media", StaticFiles(directory=str(settings.evidence_dir)), name="media")
_eval_dir = settings.artifacts_dir / "eval"
if _eval_dir.exists():
    app.mount("/eval", StaticFiles(directory=str(_eval_dir)), name="eval")

_dist = settings.web_dist
if _dist.exists() and (_dist / "index.html").exists():
    assets = _dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_dist / "index.html")

    @app.get("/{full_path:path}")
    async def spa(full_path: str) -> FileResponse:
        if full_path.startswith(("api/", "ws/", "media/")):
            raise HTTPException(404, "not found")
        candidate = (_dist / full_path).resolve()
        try:
            candidate.relative_to(_dist.resolve())
        except ValueError:
            raise HTTPException(404, "not found")
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
else:

    @app.get("/")
    async def index_missing() -> HTMLResponse:
        return HTMLResponse(
            """
            <html><body style="background:#0b0f14;color:#d6e2ea;font-family:system-ui;padding:48px">
            <h1>INSPECTRA</h1>
            <p>The API is running. The web client is not built yet.</p>
            <pre style="background:#131b24;padding:16px">cd web
npm install
npm run build</pre>
            <p>Or run the dev server with <code>npm run dev</code> and open http://localhost:5173</p>
            <p><a style="color:#2dd4bf" href="/docs">API docs</a></p>
            </body></html>
            """
        )


def run() -> None:
    import uvicorn

    uvicorn.run("server.app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
