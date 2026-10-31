#!/usr/bin/env python3
"""In-memory end-to-end verification of FastAPI route handlers and pipeline logic."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.app.config import settings
from server.app.db import Database
from server.app.evidence import EvidenceStore
from server.app.gate_service import GateService
from server.app.hub import Hub
from server.app.main import (
    _seed_models,
    app,
    audit,
    control,
    export_csv,
    gate_summary,
    health,
    inspections,
    list_profiles,
    models,
    outbox,
    outbox_flush,
    profile_detail,
    review_inspection,
    review_queue,
    thresholds,
    update_thresholds,
)
from server.app.mes import OutboxWorker
from server.app.player import Pipeline
from server.app.profiles import ProfileRegistry
from server.app.schemas import ControlRequest, ReviewAction, ThresholdUpdate


class MockRequest:
    def __init__(self, app_instance):
        self.app = app_instance
        self.headers = {}
        self.state = app_instance.state


async def main():
    print("=== Indore QA Cell In-Memory E2E Test ===", flush=True)

    # 1. State setup
    settings.ensure_dirs()
    settings.part_frames = 2  # speed up part completion for test
    db = Database(settings.db_path)
    registry = ProfileRegistry(settings)
    hub = Hub()
    gate = GateService(settings)
    evidence = EvidenceStore(settings)
    pipeline = Pipeline(settings, registry, db, hub, evidence, gate)
    worker = OutboxWorker(settings, db)

    app.state.db = db
    app.state.registry = registry
    app.state.hub = hub
    app.state.gate = gate
    app.state.evidence = evidence
    app.state.pipeline = pipeline
    app.state.worker = worker

    _seed_models(settings, db, registry)
    gate.ensure(pipeline.profile)
    req = MockRequest(app)

    # 2. Pipeline processing (2 frames = 1 full part commit)
    print("\n[Step 1] Running pipeline frames to commit inspection...", flush=True)
    pipeline._run_dataset_frame()
    pipeline._run_dataset_frame()
    print("  -> Part processed and committed to SQLite.", flush=True)

    # 3. Health check
    print("\n[Step 2] Testing /api/health...", flush=True)
    h = await health(req)
    print(f"  -> Health: status={h.status}, profile={h.profile}, fps={h.fps}", flush=True)
    assert h.status in ("ok", "degraded")

    # 4. Profiles
    print("\n[Step 3] Testing /api/profiles...", flush=True)
    profs = await list_profiles(req)
    print(f"  -> Found {len(profs)} profiles: {[p['id'] for p in profs]}", flush=True)
    assert len(profs) >= 5

    p_det = await profile_detail("steel", req)
    print(f"  -> Profile 'steel' gate quality: {p_det['gate']['quality']}", flush=True)

    # 5. Thresholds
    print("\n[Step 4] Testing /api/thresholds GET & PATCH...", flush=True)
    th = await thresholds(req)
    orig_pass = th["pass_max"]
    print(f"  -> Initial thresholds: pass_max={th['pass_max']}, review_max={th['review_max']}", flush=True)

    up = await update_thresholds(ThresholdUpdate(pass_max=0.50, review_max=0.70), req)
    print(f"  -> Patched thresholds: pass_max={up['pass_max']}, review_max={up['review_max']}", flush=True)
    assert up["pass_max"] == 0.50

    # Restore
    await update_thresholds(ThresholdUpdate(pass_max=orig_pass, review_max=0.75), req)

    # 6. Control commands
    print("\n[Step 5] Testing /api/control (pause & resume)...", flush=True)
    c1 = await control(ControlRequest(action="pause"), req)
    assert c1["state"]["paused"] is True
    print("  -> Paused: OK", flush=True)

    c2 = await control(ControlRequest(action="resume"), req)
    assert c2["state"]["paused"] is False
    print("  -> Resumed: OK", flush=True)

    # 7. Inspections list
    print("\n[Step 6] Testing /api/inspections...", flush=True)
    insp_list = await inspections(req, limit=10, offset=0, status="all")
    print(f"  -> Total inspections: {insp_list['total']}, returned items: {len(insp_list['items'])}", flush=True)
    assert len(insp_list["items"]) > 0
    sample_id = insp_list["items"][0]["id"]
    print(f"  -> Sample inspection ID: {sample_id}, verdict: {insp_list['items'][0].get('verdict')}", flush=True)

    # 8. Review queue & action
    print("\n[Step 7] Testing /api/review/queue and review action...", flush=True)
    rq = await review_queue(req, limit=20)
    print(f"  -> Review queue open items: {rq['count']}", flush=True)

    # Operator review action
    rev = await review_inspection(
        sample_id,
        ReviewAction(disposition="true_defect", label="scratches", note="E2E test disposition", actor="lead_inspector"),
        req,
    )
    print(f"  -> Review action recorded: disposition={rev.get('operator_disposition')}, label={rev.get('operator_label')}", flush=True)
    assert rev.get("operator_disposition") == "true_defect"

    # 9. Audit trail
    print("\n[Step 8] Testing /api/audit...", flush=True)
    aud = await audit(req, limit=10)
    print(f"  -> Audit log events recorded: {len(aud['items'])}", flush=True)
    assert len(aud["items"]) > 0

    # 10. Outbox
    print("\n[Step 9] Testing /api/outbox & flush...", flush=True)
    ob = await outbox(req, limit=10)
    print(f"  -> Pending outbox count: {len(ob['items'])}", flush=True)
    flushed = await outbox_flush(req)
    print(f"  -> Flushed simulated outbox: {flushed}", flush=True)

    # 11. Models and Gate Summary
    print("\n[Step 10] Testing /api/models & /api/gate/summary...", flush=True)
    m = await models(req)
    print(f"  -> Models engine: {m['runtime']['engine']}, device: {m['runtime']['device']}", flush=True)

    gs = await gate_summary(req)
    print(f"  -> Gate summary active profile: {gs['active']['profile']}, ready: {gs['active']['ready']}", flush=True)

    # 12. CSV Export
    print("\n[Step 11] Testing /api/export/inspections.csv...", flush=True)
    csv_resp = await export_csv(req)
    lines = csv_resp.body.decode("utf-8").strip().splitlines()
    print(f"  -> CSV exported: {len(lines)} lines", flush=True)
    print(f"  -> Header: {lines[0]}", flush=True)
    assert len(lines) >= 2

    db.conn.close()
    print("\n=======================================================", flush=True)
    print(" [PASS] ALL 11 IN-MEMORY E2E TESTS PASSED CLEANLY!     ", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
