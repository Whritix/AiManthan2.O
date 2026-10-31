# Indore QA Cell — Complete Project Transfer Guide

**Purpose:** Give the next laptop's developer enough context to install, run, demo, evaluate, and continue this project without relying on the original computer. This document is paired with the source tree; it is not a replacement for model weights or datasets.

**Project:** Real-Time AI Quality Assurance and Defect Detection  
**Domain:** AI for Mechanical and Industrial Engineering  
**Primary task:** Detect and classify known steel surface defects in images/video, show boxes and confidence, decide PASS / REVIEW / FAIL, retain evidence, and route work for human review.  
**Current stack:** Python, FastAPI, OpenCV, Ultralytics YOLO11s, PatchCore-lite, SQLite, HTML/CSS/JavaScript operator console, Vite.  
**Original measured hardware:** Windows 11, Intel i7-14700HX, NVIDIA RTX 4060 Laptop GPU (8 GB), 32 GB RAM.  
**Last project state documented:** October 31, 2026.

---

## 1. Quick start on the new laptop

Use PowerShell from the extracted project root. Do not reuse `.venv` or `node_modules` from another machine.

### Requirements

- Windows 10/11 is the primary supported demo environment.
- Python 3.11 or 3.12 is recommended. The original environment used Python 3.13, but recreate the environment if package/CUDA compatibility differs.
- Node.js and npm for frontend builds.
- For GPU acceleration, an NVIDIA GPU and compatible driver/CUDA-enabled PyTorch wheel. CPU inference may work, but will be slower.
- Free disk for the code, datasets, model artifacts, and environment. Dataset folders and model files can be large.

### Install Python and frontend dependencies

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd web
npm install
npm run build
cd ..
```

If Python 3.12 is unavailable, use the installed supported interpreter to create `.venv`. If CUDA is needed, check `torch.cuda.is_available()` after install; install the PyTorch wheel matching the new laptop's driver and desired CUDA runtime if it reports false. The project will fall back to CPU when configured with `INDORE_DEVICE=auto` and CUDA is unavailable.

### Start with an image or video using real YOLO

Promoted model weights must exist at `artifacts/models/detector_best.pt` or `artifacts/models/detector_best.onnx`.

```powershell
.\scripts\demo_input.ps1 -InputPath "C:\path\to\defect.jpg"
.\scripts\demo_input.ps1 -InputPath "C:\path\to\inspection.mp4"
```

Supported image types: JPG/JPEG, PNG, BMP, WEBP (up to 20 MB). Supported video types: MP4, AVI, MOV, MKV, WEBM, M4V (up to 500 MB). The launcher checks file decoding and loads YOLO in strict mode; it stops with an error if the real detector cannot load. It starts the dashboard at `http://127.0.0.1:8000` and streams the selected input. Images remain active for repeated inspection; videos loop at end-of-file.

The older `scripts/demo_video.ps1 -VideoPath <path>` launcher remains supported for videos. The ordinary `scripts/demo.ps1` and `DEMO.bat` launch dataset replay by default and may use a fixture depending on configuration; inspect `/api/health` before describing a demo as real inference.

### Automatic product identification on upload

Image and video uploads are classified before inspection. The five supported profile families are steel, metal nut, screw, transistor, and bottle. Videos use their first decodable frame to select one profile; the pipeline then uses that profile's configured detector or anomaly gate. The UI displays the selected profile with the result. Low-confidence predictions are rejected with an uncertainty message instead of silently keeping the previous profile.

The profile router is a separate HOG + HSV feature classifier, not an extra YOLO defect class. Its trained artifact is `artifacts/models/auto_profile_classifier.joblib`; the held-out evaluation summary is `artifacts/eval/auto_profile_report.json`. The current report shows 98.2% accuracy on the project's held-out profile datasets. This measures product-family routing on these datasets and does not guarantee accuracy on a different camera, background, scale, or unseen product. To recreate the artifact, run `python scripts/train_auto_profile.py`; it uses the sibling MVTec dataset when present, otherwise it falls back to the packaged profile bank/calibration and held-out images. The model artifact must be copied with the project, and `scikit-learn` is included in `requirements.txt`.

### Start manually

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location).Path 'Ultralytics'
$env:INDORE_PROFILE = 'steel'
$env:INDORE_SOURCE = 'dataset'
$env:INDORE_DETECTOR = 'real'
python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, run the frontend without building:

```powershell
cd web
npm run dev
```

Vite runs on `http://127.0.0.1:5173` and proxies API, evidence, MJPEG, and WebSocket traffic to port 8000. A production build is served from the backend at port 8000.

### Browser pages

- `/` — Live inspection, file input picker, runtime status, controls, detections, overlays, latency.
- `/review` — Inspection evidence and human disposition.
- `/metrics` — Counts, verdicts, latency, and quality metrics.
- `/datasets` — Dataset availability and verification.
- `/audit` — Audit history.
- `/settings` — Thresholds and configuration details.
- `/docs` — FastAPI Swagger docs.
- `/openapi.json` — Generated API schema.

---

## 2. What to copy to the other laptop

Copy the complete project directory, retaining this layout. Use an external drive, a private repository, or a trusted file transfer. The workspace may contain several gigabytes of images and models.

### Required for the full trained detector demo

- Source code and docs: `server/`, `web/src/`, `scripts/`, `ml/`, `docs/`, root `.md`, `.py`, `.bat`, `.ps1`, `.json`, and YAML files.
- Promoted model: `artifacts/models/detector_best.pt` (recommended for the PyTorch path). The optional ONNX counterpart is `artifacts/models/detector_best.onnx`.
- Profile definitions and anomaly banks: `data/profiles/`, plus the required MVTec/NEU data folders if using the gate and dataset pages.
- Dataset used for detector training/evaluation: `data/neu-det/` including images, labels, YAML, and `splits.json`.
- Clean steel/gate data if the novelty gate or dataset verification is part of the demo: `data/neu-good/`.
- MVTec profile files if showing `bottle`, `metal_nut`, `screw`, or `transistor` profiles.
- `web/package.json` and `web/package-lock.json`; run `npm install` on the new machine.
- `requirements.txt`; create a fresh `.venv` and install on the new machine.

### Optional or regenerable

- `artifacts/runs/` — training logs, curves, checkpoints, and historical experiments; copy if documenting/reviewing training. Not needed for inference when promoted weights are present.
- `artifacts/eval/` — evaluation reports and images used for slides/reproducibility.
- `artifacts/demo_inputs/` and `artifacts/videos/` — previously uploaded demo media; copy only if useful. New media can be selected in the dashboard.
- `artifacts/evidence/` and `artifacts/indore.db*` — generated inspections and local history. Copy only if the new machine should retain that history; otherwise the app recreates its database.
- `web/dist/` — generated frontend output. Rebuild with `npm run build` if absent or stale.

### Do not copy as an active environment

- `.venv/`, Python `__pycache__/`, and `web/node_modules/` are machine-specific/generated; recreate them.
- Do not delete or regenerate datasets/splits if reproducing the recorded evaluation. That changes the benchmark.
- Do not share `.env`, webhook secrets, API tokens, private URLs, or database contents publicly. No secrets are needed for a local demo.

Suggested zip from the project parent (review exclusions before use):

```powershell
Compress-Archive -Path .\Indore\* -DestinationPath .\Indore-project.zip
```

This includes generated outputs too. For a smaller package, exclude `.venv`, `web/node_modules`, cache folders, and optionally training runs/evidence/SQLite while retaining `artifacts/models` and required data.

---

## 3. System architecture and data flow

```text
Image / video / camera / NEU dataset replay
       ↓
OpenCV capture and image preprocessing
       ↓
YOLO11s: known defect classes and bounding boxes
       + PatchCore-lite: anomaly score and heatmap (profile dependent)
       ↓
Fusion: temporal vote, deduplication, severity, thresholds
       ↓
PASS / REVIEW / FAIL and route reason
       ↓
SQLite inspection + evidence images + audit + MES outbox
       ↓
FastAPI JSON/WebSocket/MJPEG → HTML/CSS/JavaScript operator dashboard
```

The server owns the input and inference. The browser receives a rendered MJPEG stream plus structured JSON over WebSocket and HTTP. The current image/video upload API writes files under `artifacts/demo_inputs/`, checks that the active detector is YOLO, validates decoding, changes input source, and resumes the pipeline. Images are inspected repeatedly. Video clips loop. The camera source uses OpenCV camera index configured by `INDORE_CAMERA_INDEX`.

### Main directories

- `server/app/` — FastAPI routes, runtime config, detector, pipeline, fusion, evidence, database, gate, MES.
- `web/index.html` — current operator console document.
- `web/public/app.js` and `web/public/app-integration.js` — console behavior and API integration.
- `web/public/styles.css` and `web/public/integration.css` — supplied visual design and integration styling.
- `web/src/` and `web/original-react-entry/` — previous React frontend, preserved as a fallback/reference.
- `data/neu-det/` — detector dataset and split metadata.
- `data/neu-good/` — clean steel images used in gate/calibration workflows.
- `data/profiles/` — profile manifest and associated profile assets.
- `data/own/` — captured real-world image intake templates and validation script.
- `ml/` — detector/anomaly model tooling.
- `scripts/` — training, evaluation, audits, validation, launchers.
- `artifacts/models/` — promoted deployable weights.
- `artifacts/runs/` — training outputs.
- `artifacts/eval/` — evaluation reports and diagnostic assets.
- `artifacts/evidence/` — runtime inspection evidence.
- `artifacts/indore.db` — runtime SQLite database (path can be overridden).
- `Ultralytics/` — writable Ultralytics configuration used by launchers to avoid inaccessible user-profile settings.

---

## 4. Detector, classes, and data

### Primary dataset: NEU-DET

The detector is trained on five classes:

1. `crazing`
2. `inclusion`
3. `patches`
4. `pitted_surface`
5. `scratches`

`rolled-in_scale` is held out as an unknown class and excluded from detector train/validation/test. The original NEU-DET images are 200×200 pixels; resizing to 640/1024 does not add image detail.

Project split counts: train 1,470 (1,050 original + 420 synthetic), validation 225, test 225, held-out `rolled-in_scale` 300. Synthetic augmentations are train-only: 210 CLAHE and 210 horizontal flips. `data/neu-det/splits.json` records original split metadata and augmentation provenance.

Known data caveats: an exact duplicate is documented across train and validation (`patches_101` / `patches_105`). The published test split does not include that duplicate. Do not remake the split when comparing to the existing results. See `docs/DATASETS.md`, `data_synth_playbook.md`, and `data/neu-det/splits.json` for label, source, and augmentation details.

### Promoted detector

- Architecture: YOLO11s.
- Promoted artifact: `artifacts/models/detector_best.pt` and optional ONNX artifact.
- Training: 120 epochs, 1,470-image augmented train set. The promoted run used 1024px training; deployed runtime default is 640px.
- Recorded SHA-256 for promoted PT weights: `617494900d7082ee309e93f3fac8dc97a7a8b9a8b1bf8545e5d7153ba2f3e209`.
- Detector confidence default: 0.35; NMS IoU in runtime config is 0.70 unless changed. The canonical model comparison uses NMS IoU 0.6; do not mix runtime settings with benchmark protocol.
- Other experiments and real-only retraining are in `artifacts/runs/` and evaluation records. The real-only checkpoint was not promoted.

### Reported canonical metrics

On the 225-image NEU-DET test split, 640×640, deployment-style `rect=False`, NMS IoU 0.6:

| Class | Precision @0.35 | Recall @0.35 | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| patches | 85.3% | 80.8% | 0.879 | 0.511 |
| scratches | 84.8% | 88.7% | 0.907 | 0.384 |
| inclusion | 78.0% | 68.5% | 0.792 | 0.362 |
| pitted_surface | 90.1% | 71.1% | 0.798 | 0.470 |
| crazing | 51.9% | 34.3% | 0.390 | 0.138 |
| Overall | 78.0% | 68.7% | 0.753 | 0.373 |

Crazing is the main weakness. Be honest about this in a presentation. A reported 1024 + TTA benchmark is 0.774 mAP50 / 0.397 mAP50–95, but takes materially longer and is not the default deployment path. Metrics and protocol history are in `README.md`, `OPTIMIZATION_CONTEXT.md`, `artifacts/eval/`, and `TRAINING_SPEC.md`.

### Anomaly gate status

PatchCore-lite compares features with good-part memory banks. It is trusted for selected profiles such as `bottle`, `metal_nut`, and `transistor`. The steel gate is marked untrusted/not validated (reported AUROC 0.632) and does not drive steel routing. Gate scores may be shown for transparency, but the UI must not describe the steel gate as a validated unknown-defect detector. Read `/api/health`, `/api/gate/summary`, and profile state dynamically.

---

## 5. Real image and video intake

### Dashboard path

1. Start with a real YOLO model (`scripts/demo_input.ps1` or `INDORE_DETECTOR=real`).
2. Open the Live page at `http://127.0.0.1:8000`.
3. Choose **Choose image or video** and select a supported file.
4. The server validates the file, stores it under `artifacts/demo_inputs/`, changes source to `image` or `video`, and resumes the pipeline.
5. The live MJPEG picture contains the YOLO overlay. Detection list, class, confidence, severity, verdict, and latency are driven by frame JSON.

`PUT /api/input?filename=<name>` takes raw file bytes. `PUT /api/video` remains as a video-only compatibility route. The server rejects input when the active detector is not YOLO. Uploaded external images/videos do not have labels, so they must not be counted as labeled evaluation examples.

### Important for a fair demo

- Use a steel surface image or video similar to NEU-DET for the five trained classes.
- Position and lighting affect confidence. NEU-DET's small grayscale crops differ from uncontrolled phone/camera footage.
- The model cannot label defect types outside its five trained classes; a steel novelty gate is not validated.
- Dataset replay uses known sample context; use `source=image` or `source=video` and `engine=yolo11s-*` to prove model inference on external media.
- Camera is optional; the user's video/image is the most reliable live demo input.

---

## 6. API and frontend reference

API base is same origin. Swagger at `/docs` and `/openapi.json` is authoritative if code changes. JSON APIs generally return errors as `{ "detail": "..." }`.

### Core endpoints

| Endpoint | Function |
|---|---|
| `GET /api/health` | Runtime status, detector/gate/source, device, FPS, DB counters, warnings. |
| `GET /api/profiles`, `GET /api/profiles/{id}` | Profile catalogue/details. |
| `GET /api/gate/summary` | Gate readiness, quality, trust and separation. |
| `GET /api/datasets/verify?force=` | Dataset integrity and availability report. |
| `GET /api/eval/summary` | Evaluation summary/operating point artifacts. |
| `GET /api/frames/latest`, `GET /api/frames/recent` | Latest/recent structured frame data. |
| `GET /live.mjpg` | Rendered live stream with detector overlay. |
| `WS /ws/live` | JSON frame messages `{type:"frame", frame: FrameResult}`. |
| `GET /api/control/state` | Source, profile, pause, detector/gate state and accepted mode lists. |
| `POST /api/control` | Body `{action,value}`; actions include pause/resume/skip/profile/sequence/detector/source. |
| `PUT /api/input?filename=` | Raw image/video bytes; validates and runs as source image/video. |
| `PUT /api/video?filename=` | Video-only compatibility upload. |
| `GET /api/inspections?limit=&offset=&verdict=&profile=&status=` | Inspection list. |
| `GET /api/inspections/{id}` | Inspection details and evidence. |
| `GET /api/review/queue`, `POST /api/inspections/{id}/review` | Human review queue and disposition. |
| `GET /api/metrics?profile=` | Inspection/verdict/latency/review metrics. |
| `GET /api/audit?limit=` | Audit events. |
| `GET /api/outbox`, `GET /api/outbox/status`, `POST /api/outbox/flush`, `POST /api/outbox/{id}/retry` | MES outbox status and retry actions. |
| `GET/POST /api/mes/events` | MES event history and ingestion. |
| `GET/PATCH /api/thresholds` | Active decision thresholds. |
| `GET /api/models`, `GET /api/config` | Runtime models and effective public configuration. |
| `GET /api/export/inspections.csv?profile=` | CSV download. |
| `GET /api/datasets/image?profile=&kind=&index=` | Dataset sample media. |

Detection bbox is `[x,y,width,height]` in source pixels. Verdicts are lowercase `pass`, `review`, `fail`. Review dispositions are `true_defect`, `false_alarm`, `reclassify`. Timestamps ending `_ms` are Unix milliseconds. Media URLs are same-origin paths.

### Frontend source map

- `web/index.html` — page layout and initial content.
- `web/public/app.js` — navigation and file selection/upload flow.
- `web/public/app-integration.js` — API-backed health, WebSocket, review, metrics, datasets, audit, and settings.
- `web/public/styles.css` and `web/public/integration.css` — user-supplied styling and integration fixes.
- `web/src/` — preserved prior React implementation; it is not the active Vite entry.
- `web/vite.config.ts` — development proxies.

---

## 7. Configuration reference

Environment variables are read when the server process starts. Default values are in `server/app/config.py`.

| Variable | Purpose | Default / notes |
|---|---|---|
| `INDORE_PROFILE` | Active inspection profile | `steel` |
| `INDORE_SOURCE` | Initial source | `dataset`; supports dataset/camera/video/image |
| `INDORE_DETECTOR` | Detector mode | `auto`; use `real` for strict YOLO startup |
| `INDORE_DEVICE` | Device selection | `auto`; `cuda`, `cpu`, or auto |
| `INDORE_IMGSZ` | Inference image size | 640 |
| `INDORE_CONF` | Detector confidence floor | 0.35 |
| `INDORE_IOU` | NMS IoU | 0.70 runtime default |
| `INDORE_PART_FRAMES` | Frames accumulated per part | 12 |
| `INDORE_TARGET_FPS` | Pipeline pacing target | 12 |
| `INDORE_GATE_EVERY` | Gate inference cadence | 0 means auto |
| `INDORE_SEQUENCE` | Dataset replay ordering | `mixed` |
| `INDORE_CAMERA_INDEX` | OpenCV camera index | 0 |
| `INDORE_VIDEO_PATH` | Initial local video path | empty |
| `INDORE_IMAGE_PATH` | Initial local image path | empty |
| `INDORE_DATA_DIR` | Dataset root override | `<project>/data` |
| `INDORE_ARTIFACTS_DIR` | Artifacts root override | `<project>/artifacts` |
| `INDORE_DB` | SQLite DB path override | `<artifacts>/indore.db` |
| `INDORE_MES_URL` | Optional MES webhook URL | empty |
| `INDORE_WEBHOOK_SECRET` | HMAC secret for webhook | development default; replace for real integrations |
| `INDORE_ACTOR` | Audit actor | `operator.demo` |
| `YOLO_CONFIG_DIR` | Ultralytics writable config | Launcher sets `<project>/Ultralytics` |

Detector confidence and fusion threshold profile `conf_min` are separate floors. For a controlled operating point, keep them aligned. Changing environment settings requires restarting the server.

---

## 8. Training and evaluation workflows

Do not retrain just to run the demo. The promoted weights are already included. If training is needed, use project scripts and preserve the test split.

- Training notes and acceptance criteria: `TRAINING_SPEC.md`.
- Overall technical context and historical measurements: `OPTIMIZATION_CONTEXT.md`.
- Plans/status: `MASTER_PLAN.md`, `PROJECT_STATUS.md`, `ACTION_PLAN.md`.
- Canonical evaluator: `scripts/eval_canonical.py`.
- Dataset audit: `scripts/audit_crazing_data.py`.
- Test failure analysis: `scripts/analyze_test_errors.py`.
- Threshold calibration: `scripts/calibrate_confidence.py`.
- False alarm scan: `scripts/scan_false_alarms.py`.
- Augmentation script: `scripts/augment_rare_classes.py` (review its arguments and outputs before running).

Canonical 640 evaluation example:

```powershell
python scripts/eval_canonical.py `
  --weights artifacts/models/detector_best.pt `
  --split test --imgsz 640 --conf 0.35 --iou 0.6 `
  --out artifacts/eval/canonical_640.json
```

The canonical protocol uses 640, `rect=False`, NMS IoU 0.6, test split, and standard AP matching. Do not publish legacy `rect=True` numbers as deployment results. Any training/evaluation changes should record weight SHA, split, parameters, environment, and artifact output.

---

## 9. Dataset governance and real capture intake

- Read `docs/DATASETS.md` and `data_synth_playbook.md` before changing data.
- Keep test data out of training, augmentation, and threshold fitting.
- Group capture sessions and parts into only one split to avoid leakage.
- Original synthetic augmented image files remain training-only and should be traceable to parents.
- Real capture intake instructions are in `data/own/README.md`.
- Per-session metadata template: `data/own/sessions.csv`.
- Per-image metadata template: `data/own/manifest.csv`.
- Put original captured files under `data/own/raw/`; do not overwrite originals during preprocessing.
- Validate new capture metadata with `python scripts/validate_own_capture_manifest.py`.

The project does not include an industrial camera calibration or production conveyor controller. Camera is an optional OpenCV input; video/image intake is the easiest repeatable presentation route.

---

## 10. Known limitations and honest presentation claims

1. Crazing detection is weak; reported test mAP50 is 0.390 and recall at confidence 0.35 is 34.3%.
2. The steel novelty gate is untrusted/not validated and must not be presented as proven unknown-defect escalation for steel.
3. The production conveyor is simulated through replay; no physical conveyor actuation or safety-rated PLC is integrated.
4. Performance numbers belong to the named RTX 4060 test machine and stated protocol. New hardware needs fresh measurements.
5. NEU-DET images are small and controlled; real lighting, camera focus, working distance, surface finish, motion blur, and domain shift can reduce accuracy.
6. Dataset image/video intake executes real YOLO if health shows `detector_engine` starting with `yolo`; fixture mode is explicitly marked and is not model inference.
7. Current local API has no authentication. Bind to `127.0.0.1` for local use; LAN access requires network/firewall consideration and appropriate security controls.
8. Manual thresholds and measured benchmark confidence may differ; state the actual operating settings used in a demo.

Suggested presentation: show a selected external image/video, confirm `/api/health` says `detector_mode=real` and `detector_engine=yolo11s-*`, show detection overlays, then demonstrate review routing and metrics. Mention the weak class and gate limitations when discussing readiness.

---

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Launcher says promoted weights missing | Copy `artifacts/models/detector_best.pt` or `.onnx` into the expected folder. |
| Launcher says YOLO failed to load | Reinstall compatible Python dependencies; check GPU/CUDA; inspect error. Keep `YOLO_CONFIG_DIR` pointed inside the project. |
| Dashboard does not show latest UI | Run `cd web; npm run build`, then refresh browser. |
| Port 8000 occupied | Stop the previous Uvicorn process or choose another `-Port` with the launcher. |
| Video rejected | Re-encode to MP4/H.264 or AVI; verify file decodes with OpenCV and is under 500 MB. |
| Image rejected | Use JPG/PNG/BMP/WEBP, under 20 MB, and verify it opens normally. |
| Upload gets 409 | Switch detector to `real` or restart with `demo_input.ps1`; verify `/api/health`. |
| Input falls back to dataset | Check `/api/control/state` warnings, path access, and that OpenCV can decode the input. |
| GPU unavailable | Check NVIDIA driver, CUDA-compatible torch, then use `INDORE_DEVICE=auto` or `cpu`. |
| Steel gate says untrusted | Expected current status; show detector detections, do not claim validated steel novelty routing. |
| Browser cannot connect in Vite mode | Start backend on 127.0.0.1:8000 and Vite from `web/`; proxy settings are in `web/vite.config.ts`. |

---

## 12. Reference files

- `README.md` — project quickstart and benchmark summary.
- `FRONTEND_HANDOFF.md` — frontend/API-focused handoff.
- `TRAINING_SPEC.md` — training parameters and model decision rules.
- `OPTIMIZATION_CONTEXT.md` — detailed historical experiments and calibration.
- `PROJECT_STATUS.md` and `ACTION_PLAN.md` — progress and next work.
- `MASTER_PLAN.md` — original project strategy and demo flow.
- `docs/SETUP_OTHER_PC.md` — machine transfer notes; some older environment/version details may be stale, so this guide and current `requirements.txt` take precedence.
- `docs/DATASETS.md`, `data_synth_playbook.md` — datasets, splits, and augmentation.
- `server/app/main.py`, `schemas.py`, `config.py` — backend API/schema/config implementation.
- `server/app/player.py`, `detector.py`, `fusion.py`, `gate_service.py` — runtime inference and decisions.
- `web/src/lib/api.ts`, `types.ts` — current frontend API contract.

When documentation and running code disagree, inspect current source and `/openapi.json`, then update this guide with any change that affects the next laptop.
