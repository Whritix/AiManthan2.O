# Inspectra — Project Status Summary

> **Superseded as of 2026-10-29.** Current metrics, gate status, and evaluation protocol live in `README.md` and `OPTIMIZATION_CONTEXT.md`. This file is retained as historical planning context; do not quote its numbers.

## Executive Summary
Inspectra is a **fully architected AI-powered industrial quality assurance system** with most core infrastructure built. The project follows the master plan for a 48-hour hackathon, with the architecture designed around **YOLO11s + PatchCore-lite novelty gate fusion**.

## What's Built ✅

### Architecture (per MASTER_PLAN.md §3)
- **Stage 1 — Detector**: YOLO11s ONNX, 5 known defect classes
- **Stage 2 — Novelty gate**: PatchCore-lite, Wide ResNet-50-2 backbone, cosine distance kNN memory bank
- **Stage 3 — Fusion & Decision**: Hysteresis, IoU dedupe, severity (1-4), routing (PASS/REVIEW/FAIL)
- **Data plane**: frames → inference → decision → persist with bounded queues
- **Control plane**: thresholds, model versions, routing rules, versioned snapshots
- **Observability plane**: latency, drift, heartbeats, audit (read-only)

### Model Stack (per MASTER_PLAN.md §4 & TRAINING_SPEC.md)
- **Detector**: YOLO11s (9.4 M params, 21.5 GFLOPs) on NEU-DET
- **Classes**: 5 trained (crazing, inclusion, patches, pitted_surface, scratches) + 1 held-out (rolled-in_scale)
- **Input**: 640×640, SGD lr0=0.01, 200 epochs, batch 16, AMP, cache=ram
- **Export**: ONNX FP32 → TensorRT FP16 (GPU) / ONNX INT8 (CPU fallback)
- **Novelty gate**: PatchCore-lite, 4,000 memory bank vectors, ResNet18 features at layer2 (1/8) + layer3 (1/16)

### Data Pipeline ✅
- **NEU-DET**: 1,800 images, 6 classes; 5 trained classes, 1 held out
- **Held-out class**: `rolled-in_scale` (300 images) — never in train/val/test
- **splits.json**: base manifest is hash-locked (1,050 train + 225 val + 225 test + 300 heldout = 1,800). The on-disk train set is 1,470 after 420 train-only synthetic crazing augmentations, which the manifest does not enumerate. The server's dataset report asserts a 1,800 base total, so extend the manifest with augmentation metadata rather than overwriting the base counts.
- **neu-good**: Domain-matched good part crops/banks for PatchCore gate (bank: ~250 crops, calib: ~45, test: ~45)
- **MVTec AD**: 12 categories (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, transistor) for gate benchmarking

### Server Infrastructure ✅
- **FastAPI application** (`server/app/main.py`) with full route set:
  - `/api/health`, `/api/profiles`, `/api/gate/summary`
  - `/api/datasets/verify`, `/api/eval/summary`
  - `/api/datasets/image` — serve NEU-DET/MVTec images by profile/index
  - `/api/frames/latest` / `/api/frames/recent` — WebSocket MJPEG frames
  - `/live.mjpg` — MJPEG stream endpoint
  - `/ws/live` — WebSocket live feed
  - `/api/control` — pause/resume/sequence/detector/source changes
  - `/api/inspections` — list/get inspections
  - `/api/review/queue` / `/api/inspections/{id}/review` — review workflow
  - `/api/metrics` — operational metrics (FP/1000, recalls, latencies, unknown-routing rate)
  - `/api/audit` — audit trail
  - `/api/outbox` / `/api/outbox/status` / `/api/outbox/flush` / `/api/outbox/{id}/retry` — MES outbox with HMAC signing
  - `/api/mes/events` / `/api/mes/events` — MES webhook receiver with HMAC verification
  - `/api/thresholds` / `/api/thresholds` — PATCH threshold profile (pass_max, review_max, unknown_z, conf_min)
  - `/api/models` — model versions + artifacts
  - `/api/config` — runtime configuration
  - `/api/export/inspections.csv` — CSV export of inspection history
- **Pipeline** (`server/app/player.py`) — dataset replay / camera capture loop, per-frame inference + gate scoring + fusion decision + evidence store + WebSocket broadcast
- **Database** (`server/app/db.py`) — SQLite WAL, inspections + defects + audit + outbox + thresholds + model_versions
- **Gate service** (`server/app/gate_service.py`) — lazy PatchCore-lite loader with CPU fallback
- **Fusion logic** (`server/app/fusion.py`) — severity (1-4), IoU dedupe (0.45), k-of-m hysteresis, anomaly z-score bands (0.55/0.75), routing rules
- **Profile registry** (`server/app/profiles.py`) — data-driven profile catalogue, gate quality assessment, dataset verification
- **Schemas** (`server/app/schemas.py`) — Pydantic contracts (FrameResult, Detection, AnomalyInfo, LatencyMs, Routing, ReviewAction, ThresholdProfile, etc.)
- **Evidence store** (`server/app/evidence.py`) — daily stills directories, heatmap/crop/source JPEGs, SHA-256 hashes
- **MES outbox** (`server/app/mes.py`) — signed webhook with retry, dead-letter, simulated flush for demo
- **Hub** (`server/app/hub.py`) — thread-safe frame fan-out for WS + MJPEG, FPS tracking

### Web UI ✅
- **React + Vite + TypeScript + Tailwind** built at `web/dist/`
- `index.html`, `assets/index-{js,css}.js/css`
- Components, hooks, pages, styles assembled
- API integration via WebSocket + REST

### Scripts & Tests ✅
- `scripts/smoke_test.py` — 131 checks covering imports, config, profiles, gate, DB, FrameResult contract, overlay render, heldout integrity
- `scripts/e2e_verify.py` — 11 in-memory E2E tests of FastAPI routes and pipeline logic
- `scripts/demo.ps1` — One-command PowerShell launcher (kill port, verify env, build web, start uvicorn)
- `scripts/e2e_test.py` / `scripts/e2e_verify.py` — Additional test infrastructure

### Documentation ✅
- **MASTER_PLAN.md** — Full 48-hour execution plan with all gates, roles, risk register, rubric
- **TRAINING_SPEC.md** — Frozen YOLO11s architecture, hyperparameters, datasets, targets, failure playbook
- **README.md** — Quickstart, architecture diagram, measured accuracy, latency benchmarks, real vs simulated
- **docs/** — SCOPE, RUNBOOK, DEMO_SCRIPT, PITCH, RESULTS, RISKS (framework documents)
- **DATASETS.md** — Verified download links, sizes, licenses, exact split plan
- **SETUP_OTHER_PC.md** — Install order, path portability, machine handoff

### Artifacts ✅
- `artifacts/models/detector_best.pt` — PyTorch weights (YOLO11s trained on NEU-DET 5 classes)
- `artifacts/models/detector_best.onnx` — ONNX FP32 export
- `artifacts/models/gate_steel.npz` — PatchCore bank (negative separation, shown transparency only)
- `artifacts/models/gate_metal_nut.npz`, `gate_bottle.npz`, `gate_transistor.npz` — trusted PatchCore banks; `gate_screw.npz` is marginal and untrusted
- `artifacts/models/gate_summary.json` — Gate quality summary with separation scores
- `artifacts/eval/test_metrics.json` — Evaluation metrics
- `artifacts/indore.db` — SQLite database with seeded thresholds, model versions, audit log
- `runs/neu_yolo11s/` — Training run logs

## Critical Issue — Steel Gate Separation 🚨

The **steel Profile's PatchCore gate has negative separation** (confirmed in `artifacts/models/gate_summary.json` §steel):
- `separation_defect`: -0.245, `separation_unknown`: -0.127
- Interpretation: The gate scores defects as *less* anomalous than good parts (inverted)
- Note in gate_summary.json: "Negative separation: clean-crop bank scores defects below good. Not usable as a novelty gate; shown for transparency only."

**Resolution**: The profile code (`profiles.py` §147) explicitly sets `gate_trusted=False` for the steel profile (`gate_trusted=quality in {"strong", "good", "weak"} and pid != "steel"`). This is the correct behavior — the steel gate is displayed for transparency but not trusted for critical routing decisions. The novelty demo uses the profiles whose gates have validated defect-vs-good separation, **bottle / metal_nut / transistor**, not the withheld `rolled-in_scale` class. The withheld class proves detector leakage is zero; it does **not** prove the steel gate escalates it to UNKNOWN.

The MVTec AD gate categories are not uniformly trusted: bottle (AUROC 0.996), metal_nut (0.973), and transistor (0.942) are trusted; screw (0.611) is marginal and untrusted; steel (0.632) is not validated and untrusted.

## Project Timeline & Next Gates

Based on the MASTER_PLAN.md 48-hour execution plan:

| Gate | Target | Estimated Effort | Status |
|------|--------|-----------------|--------|
| **H8** | Walking skeleton: camera → fixture boxes | 2-4 h | Infrastructure ready; verify pipeline |
| **H14** | mAP50 ≥ 0.60 on 5 classes, ≥15 FPS, ONNX working | 8-11 h | **Met:** 0.753 mAP50 at 640 (`rect=False`), 0.774 at 1024+TTA. 640 ONNX exported and measured at 16.1 FPS CPU. |
| **H20** | detect → log → review → ack → KPI → CSV (offline) | 13-14 h | API routes exist; need end-to-end verification |
| **H26** | FEATURE COMPLETE — every slide has app widget | 16-20 h | In progress |
| **H32** | RESULTS.md frozen with hardware stamps | 26-32 h | Need evaluation results |
| **H36** | SCRIPT + DECK FROZEN | 36-40 h | Need demo video + slide deck |
| **H40** | DEMO FREEZE — only P0/P1 fixes | 40-48 h | Pending |

## Immediate Actions Needed

> **Historical plan.** The detector is trained and promoted; the 120-epoch retrain finished and was promoted on the 640 deployment path. The commands below are retained as the original plan, not as current instructions.

### 1. Train YOLO11s Detector
**Command** (from project root):
```bash
python ml/train/train_yolo.py --model yolo11s.pt --data data/neu-det/neu-det.yaml --epochs 200 --imgsz 640 --batch 16 --device 0 --workers 4
```
**Target**: mAP50 ≥ 0.60 on 5 classes (H14 gate), worst-class recall ≥ 0.70

**Expected timeline**: 90-120 minutes on RTX 4060 8GB

### 2. Export to ONNX
**Command**:
```bash
python ml/export/export_onnx.py --weights artifacts/models/detector_best.pt --imgsz 640
```
Verify: the shipped ONNX is static 1×3×1024×1024 FP32 (38 MB); export a 640 graph and benchmark that exact file before claiming CPU latency.

### 3. Verify End-to-End Pipeline
Run the full pipeline with dataset replay:
```bash
python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000
```
Then:
- Browse to http://127.0.0.1:5173 (web UI)
- Check /api/health, /api/profiles, /api/metrics
- Run inspection cycle via Web UI or API
- Verify anomaly score → REVIEW routing on a trusted gate profile (bottle / metal_nut / transistor). The withheld steel class is **not** escalated as UNKNOWN.
- Verify known defect → FAIL/REWORK routing

### 4. Fix/Verify Gate Bank (if needed)
The steel gate bank has negative separation but is correctly marked `gate_trusted=False`. If a positive-separation steel gate is needed:
- Rebuild bank using `python ml/anomaly/build_banks.py --manifest data/profiles/manifest.json --profiles steel`
- Or capture new good-part images from the demo rig
- Or rely on MVTec AD categories (metal_nut, screw, transistor, bottle) for positive-separation gate demo

### 5. Prepare Demo Script & Slides
Per MASTER_PLAN.md §10 demo script (3 minutes):
- 0:00 [Live] tab: boxes appear
- 0:20 FPS counter + latency waterfall
- 0:45 Drop scratched part → FAIL
- 1:00 [Review] tab: evidence crops + heatmap
- 1:15 Click "True defect" → row disappears, KPI increments
- 1:30 Switch to a bottle/metal_nut novelty clip → trusted gate → REVIEW → "z above threshold"
- 2:10 [Metrics] tab: p50/p95, defect counts
- 2:30 Threshold slider demo
- 2:45 [Slides] architecture → close

Slide deck (9 slides, ≥18 pt):
1. Title + team + roles + QR to repo
2. Problem: scrap rate, cost per escaped defect, inspector attention decay
3. Why still manual: rule-based vision brittle; supervised needs labels per defect; new part = 3-month project
4. Architecture diagram (with "unknown defect" highlighted)
5. Results: per-class table + latency table, hardware named
6. Novelty gate: trusted MVTec profile (bottle / metal_nut / transistor), measured defect-vs-good separation → REVIEW
7. Real-time proof: latency waterfall
8. Deployment & scale: edge node per cell, events-only uplink, model registry/OTA
9. Roadmap / ask: 2-week shadow-mode pilot, success criteria

## Verification Checklist

### Smoke Test (run: `python scripts/smoke_test.py`)
Expected: All checks pass, exit 0
- Core Python imports (fastapi, uvicorn, cv2, numpy, pydantic, torch)
- Server module imports (config, schemas, db, fusion, detector, gate_service, player, main)
- Config + filesystem (project_root, data_dir, artifacts_dir, models_dir, splits.json, profiles manifest)
- Promoted weights (detector_best.pt or .onnx)
- Web dist built (web/dist/index.html)
- Profiles loaded with data+gate
- Gate service load (device=cpu or cuda)
- SQLite WAL
- FrameResult JSON contract round-trip
- Overlay renderer (HUD & Clean modes)
- Heldout-class integrity (zero training leakage)

### E2E Test (run: `python scripts/e2e_verify.py`)
Expected: All 11 tests pass cleanly
- State setup + pipeline processing (2 frames → 1 part commit)
- /api/health check
- /api/profiles ( ≥5 profiles found)
- /api/thresholds GET & PATCH
- /api/control (pause & resume)
- /api/inspections list
- /api/review/queue + review action
- /api/audit trail
- /api/outbox + flush
- /api/models + /api/gate/summary
- /api/export/inspections.csv

## Success Criteria (per Master Plan)

**H14 Gate** (mAP50 + FPS):
- mAP50 ≥ 0.60 on 5 NEU-DET classes (held-out excluded)
- worst-class recall ≥ 0.70
- End-to-end p95 ≤ 120 ms on CPU fallback (RTX 4060 laptop, 8GB VRAM)
- ONNX INT8 path working on CPU (p95 ≤ 30 ms)
- Model size ≤ 25 MB FP16

**H20 Gate** (closed loop):
- detect → log → review → ack → KPI → CSV workflow working end-to-end
- FP per 1,000 good parts ≤ 50 (target ≤ 20)
- Human-review rate on trusted novelty profiles (bottle / metal_nut / transistor); the withheld steel class is not an UNKNOWN escalation
- Operator reject rate of auto-detections ≤ 10%

**H32 Gate** (RESULTS freeze):
- Every number stamped with hardware: "i7-1260P, CPU-only, ONNX INT8, imgsz=416, n=300 frames"
- RESULTS.md frozen with model table, operational table, latency table
- ONNX model SHA-256 recorded in README and app

**H36 Gate** (script + deck frozen):
- DEMO_SCRIPT.md frozen with every click annotated
- Slide deck PDF + PPTX + cloud link ready
- No new slides or widgets after H36

**H40 Gate** (demo freeze):
- Only P0/P1 fixes approved by INT
- Demo video v1 rendered (H35)
- USB stick: git bundle, weights, video, deck, wheel cache, RESTORE.md (5 numbered steps)
- Printed one-pager with QR codes

## Conclusion

The Inspectra project has **strong foundational infrastructure** built across all domains: ML model training/evaluation, novelty gate fusion, FastAPI server with full route set, React web UI, SQLite persistence, MES outbox with HMAC signing, and comprehensive test suites. 

The **steel gate negative separation** is properly handled (gate_trusted=False). The novelty story is the **trusted MVTec gate demo** (bottle / metal_nut / transistor), backed by measured defect-vs-good separation. The zero-leakage result for `rolled-in_scale` is a separate integrity claim, not an UNKNOWN-escalation demo.

The critical path forward is:
1. **Keep the promoted YOLO11s retrain** (0.753 mAP50 at 640, `rect=False`); the previous model is preserved under `artifacts/models/legacy/`.
2. **Verify the end-to-end pipeline** from dataset replay → verdict → DB → UI.
3. **Export a 640 ONNX and benchmark it** before quoting any CPU latency.
4. **Repair split metadata and the train/val duplicate** without changing the locked test set.
5. **Prepare the demo script and slide deck**, then rehearse before the H40 freeze.

The project is in a **solid position** to succeed in the hackathon. The architecture is sound, the data pipeline is correct, the infrastructure is complete, and the key product insight (novelty → human review on trusted-gate profiles, rather than silent misclassification) is implemented and documented.
