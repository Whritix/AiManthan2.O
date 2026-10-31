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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server.app.config import Settings, settings
from server.app.auto_profile import AutoProfileClassifier
from server.app.db import Database
from server.app.detector import _sha256
from server.app.evidence import EvidenceStore
from server.app.gate_service import GateService
from server.app.hub import Hub
from server.app.mes import OutboxWorker
from server.app.player import Pipeline
from server.app.profiles import ProfileRegistry
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
    auto_profile = AutoProfileClassifier(settings.auto_profile_model)
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
    app.state.auto_profile_classifier = auto_profile
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
    title="Indore QA Cell",
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


@app.get("/api/health", response_model=HealthInfo)
async def health(request: Request) -> HealthInfo:
    db = get_db(request)
    pipeline = get_pipeline(request)
    hub = get_hub(request)
    state = pipeline.state()
    warnings = list(state.get("warnings", []))
    warnings += _validation_checks(settings, get_registry(request))
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
        db=db.counts(),
        fps=round(hub.fps, 2),
        ws_clients=hub.ws_clients,
        warnings=sorted(set(warnings)),
    )


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
            has_frame, first_frame = capture.read() if readable else (False, None)
            capture.release()
            if not readable or not has_frame:
                raise HTTPException(400, "video has no decodable frames; try MP4 or AVI")
        identification = classifier.identify(first_frame)
        if identification["status"] != "identified":
            candidate = identification.get("candidate") or "none"
            confidence = float(identification.get("confidence", 0.0))
            raise HTTPException(422, detail=f"{identification['message']} Top candidate: {candidate} ({confidence:.0%}).")
        selected_profile = str(identification["profile"])
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
        pipeline.command("pause")
        pipeline.command("profile", selected_profile)
        pipeline.command("source", source)
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
async def review_queue(request: Request, limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    db = get_db(request)
    rows = db.review_queue(limit)
    counts = db.counts()
    return {
        "items": [_public_inspection(r) for r in rows],
        "count": len(rows),
        "open_reviews": counts["open_reviews"],
    }


@app.post("/api/inspections/{inspection_id}/review")
async def review_inspection(inspection_id: str, body: ReviewAction, request: Request) -> dict[str, Any]:
    db = get_db(request)
    actor = body.actor or settings.actor
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
            <h1>INDORE QA CELL</h1>
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
