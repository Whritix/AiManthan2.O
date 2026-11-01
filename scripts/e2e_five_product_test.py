#!/usr/bin/env python3
"""Integration check for five product uploads and a short video, with real YOLO.

Set INDORE_ARTIFACTS_DIR to an isolated test directory containing models and
INDORE_DETECTOR=unified before running this script.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.app.config import settings  # noqa: E402
from server.app.main import app  # noqa: E402

SAMPLES = {
    "steel": "steel_patches.jpg",
    "metal_nut": "metal_nut_flip.jpg",
    "screw": "screw_thread_top.jpg",
    "transistor": "transistor_cut_lead.jpg",
    "bottle": "bottle_contamination.jpg",
}


def wait_for_closed(client: TestClient, source: str, started_ms: int) -> dict:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        response = client.get("/api/inspections?limit=20&status=all")
        assert response.status_code == 200, response.text
        rows = [row for row in response.json()["items"]
                if row["source"] == source and row["status"] == "closed"
                and int(row["ts_ms"]) >= started_ms]
        if rows:
            return rows[0]
        time.sleep(0.4)
    raise AssertionError(f"no closed {source} inspection after upload")


def main() -> None:
    if settings.artifacts_dir.resolve() == (ROOT / "artifacts").resolve():
        raise SystemExit("Use an isolated INDORE_ARTIFACTS_DIR")
    if settings.detector_mode != "unified":
        raise SystemExit("Use INDORE_DETECTOR=unified")
    samples_dir = ROOT / "artifacts/demo_images/recommended"
    with TestClient(app) as client:
        for family, filename in SAMPLES.items():
            image = samples_dir / filename
            started_ms = int(time.time() * 1000)
            response = client.put(f"/api/input?filename={filename}", content=image.read_bytes(),
                                  headers={"Content-Type": "image/jpeg"})
            assert response.status_code == 200, f"{family}: {response.status_code} {response.text}"
            assert response.json()["profile"] == family, response.json()
            row = wait_for_closed(client, "image", started_ms)
            assert row["profile"] == family, row
            report = client.get("/api/report")
            assert report.status_code == 200, report.text
            assert report.json()["profile"] == family, report.json()
            print(f"{family}: uploaded, identified, inspected ({row['verdict']}), reported", flush=True)

        frame = cv2.imread(str(samples_dir / SAMPLES["steel"]))
        video_path = settings.artifacts_dir / "test_steel.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 4,
                                 (frame.shape[1], frame.shape[0]))
        assert writer.isOpened(), "could not create AVI fixture"
        for _ in range(4):
            writer.write(frame)
        writer.release()
        started_ms = int(time.time() * 1000)
        response = client.put("/api/input?filename=test_steel.avi", content=video_path.read_bytes(),
                              headers={"Content-Type": "video/x-msvideo"})
        assert response.status_code == 200, f"video: {response.status_code} {response.text}"
        assert response.json()["profile"] == "steel", response.json()
        row = wait_for_closed(client, "video", started_ms)
        assert row["profile"] == "steel", row
        report = client.get("/api/report")
        assert report.status_code == 200 and report.json()["source"] == "video", report.text
        print(f"video: uploaded, identified, inspected ({row['verdict']}), reported", flush=True)

        # A product outside the confident identification range must reach a
        # human queue without borrowing the previous YOLO frame or verdict.
        classifier = app.state.auto_profile_classifier
        original_identify = classifier.identify
        classifier.identify = lambda _frame: {
            "status": "uncertain", "profile": None, "candidate": "transistor",
            "confidence": 0.2, "message": "Product type is uncertain.",
        }
        try:
            image = samples_dir / SAMPLES["transistor"]
            response = client.put(f"/api/input?filename={image.name}", content=image.read_bytes(),
                                  headers={"Content-Type": "image/jpeg"})
        finally:
            classifier.identify = original_identify
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["review_required"] and payload["profile"] == "unrecognized", payload
        inspection = client.get(f"/api/inspections/{payload['inspection_id']}")
        assert inspection.status_code == 200, inspection.text
        record = inspection.json()
        assert record["verdict"] == "review" and record["reason"] == "product_not_recognized", record
        assert not record["detections"] and record["image_url"], record
        assert client.get(record["image_url"]).status_code == 200
        queue = client.get("/api/review/queue?scope=current").json()
        assert any(item["id"] == record["id"] for item in queue["items"]), queue
        assert client.get("/api/report").json()["profile"] == "unrecognized"
        disposition = client.post(f"/api/inspections/{record['id']}/review", json={
            "disposition": "reclassify", "label": "unsupported", "note": "Manual product review",
        })
        assert disposition.status_code == 200, disposition.text
        assert disposition.json()["operator_label"] == "unsupported"
        print("unrecognized product: saved with evidence, queued, manually reviewed", flush=True)

        classifier.identify = lambda _frame: {
            "status": "uncertain", "profile": None, "candidate": "steel",
            "confidence": 0.2, "message": "Product type is uncertain.",
        }
        try:
            response = client.put("/api/input?filename=unrecognized.avi",
                                  content=video_path.read_bytes(),
                                  headers={"Content-Type": "video/x-msvideo"})
        finally:
            classifier.identify = original_identify
        assert response.status_code == 200 and response.json()["review_required"], response.text
        assert response.json()["source"] == "video"
        video_review = client.get(f"/api/inspections/{response.json()['inspection_id']}").json()
        assert video_review["verdict"] == "review" and video_review["image_url"]
        assert client.get("/api/report").json()["profile"] == "unrecognized"
        print("unrecognized video: first frame saved and queued for manual review", flush=True)


if __name__ == "__main__":
    main()
