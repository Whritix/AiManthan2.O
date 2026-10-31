# Frontend Handoff — Indore QA Cell

This is the handoff for building or continuing the frontend on another computer. It describes the current server API, selected frontend files, how to run them together, and current backend caveats. The live OpenAPI document at `/openapi.json` and Swagger UI at `/docs` are the authoritative API contracts if code changes later.

**Current frontend:** `web/index.html`, `web/public/app.js`, `web/public/app-integration.js`, `web/public/styles.css`, and `web/public/integration.css`. The earlier React implementation remains in `web/src/` and its original entry is backed up under `web/original-react-entry/`.

## 1. What this app does

The app is an industrial surface inspection dashboard. A single camera, video, or replay source feeds a detector and (where validated) an anomaly gate. The server returns frame detections and PASS / REVIEW / FAIL verdicts, stores inspections in SQLite, broadcasts live frame updates, and provides review, metrics, dataset, audit, settings, and model views.

The API does not require authentication in the current local setup. Do not expose it to an untrusted network without adding suitable access controls.

## 2. Current project layout

```text
server/app/       FastAPI backend and Pydantic response schemas
web/              React + TypeScript + Vite frontend
web/src/pages/    Live, Review, Metrics, Datasets, Audit, Settings screens
web/src/lib/api.ts  Existing frontend API wrapper
web/src/lib/types.ts  Existing TypeScript API types
```

The supplied operator console is a vanilla HTML/CSS/JavaScript UI served by Vite/FastAPI. Its client connects to the live API for image/video upload, frame WebSocket updates, metrics, review, audit, datasets, and settings.

## 3. Run on the other computer

Copy the whole project folder. Keep `data/`, `weights/`, and `artifacts/models/` if the app should use the same datasets and detector artifacts. Python packages and JavaScript dependencies need to be installed on the new computer.

### Backend

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation, run `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and use that interpreter for Uvicorn. To allow access from another device on the same LAN, use `--host 0.0.0.0`; then configure Windows Firewall and use the host laptop's LAN IP as the API host. Only do this on a trusted network.

### Frontend development server

Open another terminal:

```powershell
cd web
npm install
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. Vite proxies `/api`, `/media`, `/live.mjpg`, and `/ws` to `http://127.0.0.1:8000` (WebSocket proxy for `/ws`). The packaged frontend is served by FastAPI at `http://127.0.0.1:8000` after `npm run build`.

### Real YOLO image or video demo

From the project root, run:

```powershell
.\scripts\demo_input.ps1 -InputPath "C:\path\to\inspection-demo.mp4"
# Or: .\scripts\demo_input.ps1 -InputPath "C:\path\to\defect.jpg"
```

This launcher requires promoted weights, preflights the YOLO model, sets strict real-detector mode, selects the image or video source, and starts the dashboard. The earlier `demo_video.ps1 -VideoPath` command still works. The Live page accepts images (JPG, PNG, BMP, WEBP; 20 MB maximum) and videos (MP4, AVI, MOV, MKV, WEBM, M4V; 500 MB maximum). It uploads the selected file to the backend, verifies it can be decoded, and switches the live feed to it. Images remain on screen for repeated inspection; videos loop at end-of-file. Uploads require an active YOLO detector.

### Handy local links

- Dashboard: `http://127.0.0.1:8000/` (built UI) or `http://127.0.0.1:5173/` (Vite dev UI)
- API docs: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Health: `http://127.0.0.1:8000/api/health`
- MJPEG stream: `http://127.0.0.1:8000/live.mjpg`
- Live WebSocket: `ws://127.0.0.1:8000/ws/live`
- Upload/select image or video: `PUT /api/input?filename=<file-name>` with raw file bytes in the request body. The older `PUT /api/video` route remains available for video files.

## 4. API conventions

- JSON request and response bodies use `Content-Type: application/json`.
- Timestamps named `ts_ms` or `*_at_ms` are Unix epoch milliseconds.
- Detection bounding boxes are `[x, y, width, height]` in source-frame pixels.
- Verdict values are `pass`, `review`, or `fail` (lowercase).
- Review dispositions are `true_defect`, `false_alarm`, or `reclassify`.
- Most failures return a JSON body with a `detail` field and an HTTP error status.
- Image and overlay URLs returned by the API are paths; resolve them against the same API origin (or use Vite's proxy in development).

## 5. Endpoint catalog

### Status, setup, and data context

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Runtime status, active profile/source, detector/gate readiness, device/FPS, database counts, WebSocket clients, warnings. |
| `GET /api/profiles` | Available inspection profiles and their classes, counts, and gate status. |
| `GET /api/profiles/{profile_id}` | Detail for one profile. |
| `GET /api/gate/summary` | Gate quality/trust summary across profiles and active profile. |
| `GET /api/datasets/verify?force=false` | Dataset availability/integrity report; `force=true` asks the server to re-run verification. |
| `GET /api/eval/summary` | Saved evaluation summary, operating points, and any report image URL. |
| `GET /api/datasets/image?...` | Dataset image/media retrieval. Check `/docs` for its query parameter schema. |
| `GET /api/models` | Model versions, artifacts, active runtime model, and profile gate state. |
| `GET /api/config` | Effective server configuration safe for display. |
| `GET /api/thresholds` | Active threshold profile. |
| `PATCH /api/thresholds` | Update threshold fields; body may include `pass_max`, `review_max`, `unknown_z`, `conf_min`, and/or `name`. |

### Live inspection

| Method and path | Purpose |
|---|---|
| `GET /api/frames/latest` | Latest frame envelope: `{seq, frame, fps}`; `frame` can be null before first frame. |
| `GET /api/frames/recent` | Recent frame results and FPS. |
| `GET /live.mjpg` | Multipart MJPEG live image stream, suitable for an `<img>` element. |
| `WS /ws/live` | Live JSON updates. Server sends `{ "type": "frame", "frame": <FrameResult> }`; latest frame may be sent immediately on connect. |
| `GET /api/control/state` | Current pause/profile/source/sequence/detector/gate state and warnings. |
| `POST /api/control` | Change live controls. Body: `{ "action": "pause" | "resume" | "skip" | "profile" | "sequence" | "detector" | "source", "value": string | null }`. Check `/docs` and the current UI for accepted `value` strings. |

### Inspections, operator review, and metrics

| Method and path | Purpose |
|---|---|
| `GET /api/inspections?limit=&offset=&verdict=&profile=&status=` | Paginated inspection list. Response includes `{items,total,limit,offset}`. |
| `GET /api/inspections/{inspection_id}` | Full inspection record and evidence. |
| `GET /api/review/queue?limit=` | Open review queue; response includes `{items,count,open_reviews}`. |
| `POST /api/inspections/{inspection_id}/review` | Save human decision. Body: `{ "disposition": "true_defect" | "false_alarm" | "reclassify", "label": string | null, "note": string | null, "actor": string | null }`. The UI wrapper normally omits `actor`. |
| `GET /api/metrics?profile=` | Inspection, verdict, defect, latency, anomaly, review, and gate metrics; optional profile filter. |
| `GET /api/audit?limit=` | Audit log; response `{items}`. |
| `GET /api/export/inspections.csv?profile=` | Download inspection CSV, optionally filtered by profile. |

### MES outbox/events

| Method and path | Purpose |
|---|---|
| `GET /api/outbox?limit=` | Pending/delivered outbound events plus delivery status and counts. |
| `GET /api/outbox/status` | Delivery status summary. |
| `POST /api/outbox/flush` | Ask the backend to deliver queued events; returns `{ok,flushed}`. |
| `POST /api/outbox/{outbox_id}/retry` | Retry one queued event; returns `{ok}` or an error. |
| `GET /api/mes/events?limit=` | Recent MES events. |
| `POST /api/mes/events` | Submit a MES event; body includes `event_type` and optional `payload` object. |

## 6. Frame response shape

The central live object is `FrameResult` (also used as an inspection evidence item):

```json
{
  "schema_version": "1.0",
  "frame_id": "...",
  "inspection_id": "...",
  "ts_ms": 0,
  "stream_id": "cam-1",
  "profile": "steel",
  "source": "dataset",
  "device": "cuda",
  "engine": "...",
  "model_version": "...",
  "threshold_profile": "...",
  "latency_ms": {
    "capture": 0, "preprocess": 0, "infer_detector": 0,
    "infer_gate": 0, "post": 0, "encode": 0, "total": 0
  },
  "detections": [
    {"id":"...", "class":"crazing", "conf":0.8, "bbox":[0,0,10,10], "severity":1, "source":"detector", "votes":1, "area_frac":0.01}
  ],
  "anomaly": {
    "raw_score": 0, "z": 0, "threshold": 0, "is_anomaly": false,
    "model": "...", "gate_trusted": false, "gate_quality": "not_validated",
    "separation": null, "regions": [], "heatmap_url": null, "cached": false
  },
  "verdict": "review",
  "routing": {"destination":"...", "reason":"..."},
  "severity": 1,
  "part_state": {
    "index": 1, "total": 1, "item_id":"...", "kind":"good",
    "ground_truth_class":null, "ground_truth_kind":null,
    "frame_in_part":1, "frames_in_part":1
  },
  "image_url": "/media/...",
  "overlay_url": "/media/...",
  "crop_url": null,
  "fps": 0,
  "frame_index": 0,
  "warnings": []
}
```

Treat nullable fields (`routing`, media URLs, ground truth, separation) as optional in rendering. `anomaly.gate_trusted` controls whether a gate result is trusted for policy; display the trust/quality state instead of implying all profiles use the gate equally.

## 7. Existing frontend behavior and integration files

- `web/index.html` is the current operator console document.
- `web/public/app.js` contains console interactions and upload workflow.
- `web/public/app-integration.js` binds the console to live API endpoints and WebSocket results.
- `web/public/styles.css` and `web/public/integration.css` provide the design.
- `web/src/` contains the previous React implementation; `web/original-react-entry/` preserves its entry and source backup.
- `web/vite.config.ts` configures local API/media/MJPEG/WebSocket proxies.

Prefer extending these types and API methods when the new frontend shares this repository. If building separately, copy the relevant contracts into a typed API client and make the API origin configurable (for example `VITE_API_BASE_URL`).

## 8. Current backend caveats to show clearly in the UI

The health response observed on this machine reported `status: ok`, profile `steel`, source `dataset`, detector engine `fixture-groundtruth`, `gate_ready: true`, `gate_trusted: false`, and warning `gate_not_validated`. This means the HTTP service is running, but this active detector mode is a fixture/ground-truth mode and the steel novelty gate is not trusted. The frontend must show these runtime values from `/api/health` and `/api/control/state`; do not label the current run as a validated production camera inference.

Other backend warnings and profile trust values can change with configuration and loaded assets. Always render the API's warnings and gate trust state dynamically.

## 9. Suggested frontend implementation order

1. Build the app shell and navigation for Live, Review, Metrics, Datasets, Audit, and Settings.
2. Add a typed API client with configurable API origin and a shared error/loading/empty state.
3. Build a runtime status strip from `/api/health` and `/api/control/state`.
4. Build Live view with `/ws/live`, `/live.mjpg`, latest-frame fallback, detections, verdict, warnings, and controls.
5. Build Review queue and inspection detail with images/overlays and the review POST action.
6. Add Metrics, Dataset verification, Audit, Settings/thresholds, Models, and CSV export.
7. Keep a visible distinction between detector output and gate state/trust. Avoid turning an untrusted gate result into a confident unknown-defect decision in the UI.

## 10. Source of truth

For exact query bounds, response fields, and validation errors, inspect:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`
- `server/app/main.py`
- `server/app/schemas.py`
- `web/src/lib/api.ts` and `web/src/lib/types.ts`

