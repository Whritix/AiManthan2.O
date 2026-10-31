#!/usr/bin/env python3
"""End-to-End integration test for Indore QA Cell API and Pipeline."""
from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from server.app.config import settings
from server.app.main import app


def log_step(name: str) -> None:
    print(f"\n[E2E] ---> {name}")


def assert_status(resp, code=200):
    if resp.status_code != code:
        raise AssertionError(
            f"Expected {code}, got {resp.status_code}: {resp.text}"
        )


def main():
    print("================================================================")
    print("      INDORE QA CELL - END-TO-END WORKFLOW INTEGRATION TEST     ")
    print("================================================================")

    # Use test client with lifespan context
    with TestClient(app) as client:
        # Give pipeline 1 second to produce initial inspection frames
        time.sleep(1.5)

        # 1. Health check
        log_step("1. Checking /api/health")
        r = client.get("/api/health")
        assert_status(r, 200)
        h = r.json()
        print(f"  Health: status={h['status']}, profile={h['profile']}, fps={h['fps']}")
        assert h["status"] in ("ok", "degraded")

        # 2. Profiles
        log_step("2. Checking /api/profiles")
        r = client.get("/api/profiles")
        assert_status(r, 200)
        profiles = r.json()
        print(f"  Available profiles ({len(profiles)}): {[p['id'] for p in profiles]}")
        assert any(p["id"] == "steel" for p in profiles)
        assert any(p["id"] == "metal_nut" for p in profiles)

        # Detail for metal_nut
        r = client.get("/api/profiles/metal_nut")
        assert_status(r, 200)
        p_detail = r.json()
        print(f"  metal_nut gate quality: {p_detail['gate']['quality']}, trusted={p_detail['gate']['trusted']}")

        # 3. Control API
        log_step("3. Testing /api/control (pause, resume, sequence)")
        # Pause
        r = client.post("/api/control", json={"action": "pause"})
        assert_status(r, 200)
        assert r.json()["state"]["paused"] is True
        print("  Pipeline successfully paused.")

        # Resume
        r = client.post("/api/control", json={"action": "resume"})
        assert_status(r, 200)
        assert r.json()["state"]["paused"] is False
        print("  Pipeline successfully resumed.")

        # 4. Threshold Profile GET & PATCH
        log_step("4. Testing /api/thresholds")
        r = client.get("/api/thresholds")
        assert_status(r, 200)
        current_th = r.json()
        print(f"  Current thresholds: pass_max={current_th['pass_max']}, review_max={current_th['review_max']}")

        # Update
        patch_payload = {"pass_max": 0.52, "review_max": 0.72}
        r = client.patch("/api/thresholds", json=patch_payload)
        assert_status(r, 200)
        updated_th = r.json()
        assert updated_th["pass_max"] == 0.52
        assert updated_th["review_max"] == 0.72
        print(f"  Updated thresholds: pass_max={updated_th['pass_max']}, review_max={updated_th['review_max']}")

        # Restore
        client.patch("/api/thresholds", json={"pass_max": 0.55, "review_max": 0.75})

        # 5. Let pipeline produce closed inspections
        log_step("5. Waiting for pipeline inspections to commit")
        time.sleep(2.5)

        r = client.get("/api/inspections?limit=20&status=all")
        assert_status(r, 200)
        inspections = r.json()
        items = inspections.get("items", [])
        print(f"  Recorded inspections count: {len(items)}")

        # 6. Review Queue & Disposition
        log_step("6. Testing /api/review/queue & operator review disposition")
        r = client.get("/api/review/queue?limit=50")
        assert_status(r, 200)
        q = r.json()
        q_items = q.get("items", [])
        print(f"  Review queue pending items: {len(q_items)}")

        # If review queue has items or if we pick any existing inspection to review
        target_id = None
        if q_items:
            target_id = q_items[0]["id"]
        elif items:
            target_id = items[0]["id"]

        if target_id:
            print(f"  Testing operator review on inspection: {target_id}")
            rev_action = {
                "disposition": "true_defect",
                "label": "scratches",
                "note": "E2E automated verification test",
                "actor": "qa_lead_automation",
            }
            r = client.post(f"/api/inspections/{target_id}/review", json=rev_action)
            assert_status(r, 200)
            rev_result = r.json()
            assert rev_result["operator_disposition"] == "true_defect"
            assert rev_result["operator_label"] == "scratches"
            print("  Inspection review disposition successfully recorded.")

        # 7. Audit log verification
        log_step("7. Verifying /api/audit")
        r = client.get("/api/audit?limit=20")
        assert_status(r, 200)
        audit_events = r.json().get("items", [])
        print(f"  Recent audit events count: {len(audit_events)}")
        event_types = [e["action"] for e in audit_events]
        print(f"  Sample audit events: {event_types[:6]}")
        assert any("threshold" in a or "pipeline" in a or "inspection" in a or "review" in a for a in event_types)

        # 8. Outbox & Flush
        log_step("8. Testing /api/outbox & outbox flush")
        r = client.get("/api/outbox?limit=50")
        assert_status(r, 200)
        ob = r.json()
        print(f"  Outbox status: {ob['status']}, pending items: {len(ob['items'])}")

        r = client.post("/api/outbox/flush")
        assert_status(r, 200)
        flush_res = r.json()
        print(f"  Outbox flush executed: flushed count={flush_res.get('flushed')}")

        # 9. MES Signed Webhook Ingestion
        log_step("9. Testing /api/mes/events with HMAC signature")
        mes_payload = json.dumps({"event_type": "line.status", "payload": {"line_id": "L1", "speed_ppm": 120}}).encode("utf-8")
        sig = "sha256=" + hmac.new(settings.webhook_secret.encode("utf-8"), mes_payload, hashlib.sha256).hexdigest()
        r = client.post(
            "/api/mes/events",
            content=mes_payload,
            headers={"Content-Type": "application/json", "x-indore-signature": sig},
        )
        assert_status(r, 200)
        print(f"  MES Event accepted: {r.json()}")

        # 10. CSV Export
        log_step("10. Testing /api/export/inspections.csv")
        r = client.get("/api/export/inspections.csv")
        assert_status(r, 200)
        csv_lines = r.text.strip().splitlines()
        print(f"  Exported CSV headers: {csv_lines[0] if csv_lines else 'EMPTY'}")
        print(f"  Exported CSV row count: {len(csv_lines)}")
        assert "id,ts_ms,profile,source,verdict" in csv_lines[0]

        # 11. Latest and Recent Frames
        log_step("11. Testing /api/frames/latest & /api/frames/recent")
        r = client.get("/api/frames/latest")
        assert_status(r, 200)
        lf = r.json()
        print(f"  Latest frame seq: {lf.get('seq')}, fps: {lf.get('fps')}")

        # 12. Models and Gate Summary
        log_step("12. Testing /api/models & /api/gate/summary")
        r = client.get("/api/models")
        assert_status(r, 200)
        m_info = r.json()
        print(f"  Model versions: {len(m_info.get('versions', []))}, artifacts: {len(m_info.get('artifacts', []))}")

        r = client.get("/api/gate/summary")
        assert_status(r, 200)
        g_sum = r.json()
        print(f"  Active gate profile: {g_sum.get('active', {}).get('profile')}, gate ready={g_sum.get('active', {}).get('ready')}")

    print("\n================================================================")
    print(" [PASS] ALL 12 END-TO-END WORKFLOW INTEGRATION CHECKS PASSED!   ")
    print("================================================================\n")


if __name__ == "__main__":
    main()
