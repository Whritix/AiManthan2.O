# Indore QA Cell — Complete System Context & Technical Master Brief
**Date:** October 29, 2026  
**System Name:** Indore Industrial AI Quality Assurance Cell  
**Hardware Profile:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM), 32 GB System RAM, Windows 11  
**Core Technologies:** PyTorch 2.11.0+cu128, Ultralytics 8.4.190, OpenCV 5.0.0, ONNX Runtime 1.31.1, FastAPI, React/TypeScript, SQLite WAL

---

## 1. Project Overview & System Purpose

The **Indore QA Cell** is an edge-deployed surface inspection system for cold-rolled steel strip. It processes live camera frames or dataset replay, detects known surface defects, scores novelty against a good-part memory bank, and routes each part to **PASS / REVIEW / FAIL**.

The 640 deployment path runs in about **12.6 ms p50** on the RTX 4060. The 1024 + TTA benchmark path runs in about **73 ms**. "Sub-50 ms" therefore describes the 640 deployment path, not the 1024 TTA path.

### The 3-Stage Cascaded Architecture

```
Live Camera / Replay Frame (200x200 native, resized to 640 or 1024)
       |
       v
+-------------------------------------------------------------+
| STAGE 1: Supervised Defect Detector (YOLO11s)               |
| - 5 known classes: crazing, inclusion, patches,             |
|   pitted_surface, scratches                                  |
| - Outputs boxes, class IDs, confidence scores                |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| STAGE 2: Novelty Gate (PatchCore-lite)                      |
| - Wide ResNet-50-2 features, 4,000-vector good-part bank    |
| - Patch-level kNN cosine anomaly map, image z-score         |
| - Trusted only on profiles with validated separation        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| STAGE 3: Fusion & Operational Decision Engine               |
| - IoU dedupe, severity 1-4, k-of-m temporal hysteresis      |
| - Detections -> FAIL (release_with_note / rework /          |
|   quarantine by severity)                                    |
| - Gate, only when trusted:                                   |
|     0.55 <= z < 0.75 -> REVIEW (elevated_novelty)           |
|     z >= 0.75        -> REVIEW (unknown_anomaly)            |
| - No detections + untrusted gate -> PASS (release)          |
+------------------------------+------------------------------+
```

Important: the novelty gate never sends a part directly to FAIL or QUARANTINE. It can only escalate to **REVIEW**. FAIL and QUARANTINE come from supervised detector severity.

---

## 2. Dataset Pipeline & Integrity Rules

### Primary dataset: NEU-DET

- **Native image size:** 200x200. Training and inference upscale it; upscaling adds interpolation, not real detail.
- **Classes:** `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
- **Held-out class:** `rolled-in_scale` (300 images) is excluded from detector train/val/test. Detector leakage is zero.

### Current split counts (actual files on disk)

| Split | Images | Notes |
|---|---:|---|
| Train | **1,470** | 1,050 originals + 420 synthetic; all five trained classes |
| Validation | **225** | 45 per class |
| Test | **225** | 45 per class |
| Held-out `rolled-in_scale` | **300** | Never trained on |

The 420 synthetic images are train-only augmentations (`_aug_clahe`, `_aug_hflip`). The previous promoted model never saw them. The completed 120-epoch retrain is the only run trained on the 1,470-image set; it is now promoted (see §3).

### Bookkeeping debt (training finished; safe to repair)

- `data/neu-det/splits.json` is the **base 1,800-image manifest** (1,050 train). The 420 train-only synthetic augmentations are not enumerated. The server's dataset report asserts `total == 1800`, so add an `augmentations` block or an explicit augmented-train count; do not simply overwrite the base counts.
- One exact duplicate exists across train and validation: `train/patches_101.jpg` / `val/patches_105.jpg`. Test data is clean, so the published test metrics are not contaminated. Removing either copy requires updating the manifest file list and hash and relaxing the 1,800 total check; until then it is a documented validation-cleanliness caveat.
- Do not regenerate the split from scratch after publishing metrics: that would change the test set and make every prior number incomparable.

### Secondary datasets

- **`neu-good`:** defect-free steel frames used for gate calibration.
- **MVTec AD:** `bottle`, `metal_nut`, `screw`, `transistor` profiles used for gate quality transparency and the novelty demo.

---

## 3. Promoted Detector & Measured Results

**Promoted weights:** `artifacts/models/detector_best.pt`  
**SHA-256:** `617494900d7082ee309e93f3fac8dc97a7a8b9a8b1bf8545e5d7153ba2f3e209`  
**Training:** YOLO11s, 120 epochs at 1024 px on the 1,470-image augmented train set. Best validation epoch 110 (mAP50 0.7468 / mAP50-95 0.3919). Promoted 2026-10-29.

### Canonical test protocol

- Split: `test`, 225 images, 530 instances.
- NMS IoU: **0.6**; mAP@50-95 uses the standard 0.50:0.95 matching.
- mAP: full PR curve at `conf=0.001`.
- Precision/Recall in the table: at the fixed operating confidence **0.25**.
- **`rect=False`**, matching the server's `predict` path. Ultralytics rectangular validation (`rect=True`) inflates mAP on this dataset and is retained only behind the script's legacy `--rect` flag.
- No test-time augmentation for the deployment row.

### Deployment path: 640x640, `rect=False`

| Class | Images | Precision @0.35 | Recall @0.35 | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| patches | 45 | 85.3% | 80.8% | 0.879 | 0.511 |
| scratches | 45 | 84.8% | 88.7% | 0.907 | 0.384 |
| inclusion | 45 | 78.0% | 68.5% | 0.792 | 0.362 |
| pitted_surface | 45 | 90.1% | 71.1% | 0.798 | 0.470 |
| crazing | 45 | 51.9% | 34.3% | 0.390 | 0.138 |
| **All classes** | **225** | **78.0%** | **68.7%** | **0.753** | **0.373** |

Crazing is the bottleneck. It is diffuse and low-contrast, not rare by instance count. The retrain improved four of five classes at 640 but reduced crazing mAP50 from 0.432 to 0.390.

### Resolution and TTA comparison (promoted weights, test split, NMS IoU 0.6, `rect=False`)

| Configuration | mAP@50 | mAP@50-95 | GPU inference |
|---|---:|---:|---:|
| **640 plain (deployment)** | **0.753** | 0.373 | 5.9 ms |
| **1024 + TTA (benchmark)** | **0.774** | **0.397** | 34.5 ms |

- 1024 + TTA gains **+2.1 mAP50** over the 640 deployment path at roughly 6× the inference cost. It is an offline benchmark, not the deployed path.
- TTA is only valid at 1024, the training resolution. At 640 it reduces accuracy.
- 640 + TTA and 1024 plain were not re-run under `rect=False`; their old `rect=True` values are withdrawn and must not be quoted.
- 1280 is not planned: the source is 200x200, so it would add cost without detail.

### Completed retrain: promoted (2026-10-29)

The 120-epoch `neu_yolo11s_1024_aug` run finished at 13:50. Best validation epoch was 110 (mAP50 0.7468 / mAP50-95 0.3919). Both models were re-evaluated with the same script, split, NMS IoU, confidence floor, and `rect=False` protocol:

| Weights | 640 mAP50 | 640 mAP50-95 | 1024+TTA mAP50 | 1024+TTA mAP50-95 |
|---|---:|---:|---:|---:|
| Previous promoted (now legacy) | 0.740 | 0.352 | **0.783** | **0.400** |
| **Retrain (1,470 img, promoted)** | **0.753** | **0.373** | 0.774 | 0.397 |

- The retrain gains **+1.3 mAP50 and +2.1 mAP50-95 on the 640 deployment path**.
- It loses 0.9 mAP50 and 0.3 mAP50-95 on the offline 1024 + TTA benchmark.
- **Decision: promote the retrain**, because 640 is what the server actually runs. The previous weights are preserved at `artifacts/models/legacy/detector_best_promoted_20261028.pt`.
- Full record: `artifacts/eval/retrain_decision.json`; per-run outputs are the `canonical_*_deploy.json` files in `artifacts/eval/`.
- YOLO11m has not been started; it needs separate approval.

---

## 4. Optimization Pillars: Actual Status

| Pillar | Status | What is actually true |
|---|---|---|
| 1. 1024x1024 training | **Done; used as the TTA base** | 60 epochs, 1024 px, 1,050 images. Not re-run under `rect=False`; its old 0.791/0.797 comparison is withdrawn. The promoted retrain was trained at 1024. |
| 2. Rare-class augmentation | **Trained and promoted** | +420 synthetic images, 1,470 total. Retrain wins the 640 deployment path: 0.753/0.373 vs 0.740/0.352. It trails on the offline 1024+TTA benchmark: 0.774/0.397 vs 0.783/0.400. |
| 3. YOLO11m scaling | **Not started; needs approval** | The retrain is evaluated, so it is no longer blocked. Realistic gain is +1-3 mAP50, not +10. |
| 4. TTA and SAHI | **TTA partly valid; SAHI retracted** | TTA at 1024: +2.1 test mAP50 at ~6x inference cost. SAHI is inapplicable to 200x200 images. |

### Pillar 1 detail

- Run: `artifacts/runs/neu_yolo11s_1024`.
- 60 epochs, batch 8, 1024 px, 38.8 minutes on the pre-augmentation 1,050-image train set.
- Validation mAP50 at epoch 59: 0.7388 and still rising.
- Its old `rect=True` test numbers (0.791 / 0.797) are withdrawn. It was not re-run with `rect=False`; do not quote it.
- The trained-at-1024 configuration matters because TTA is only valid at the training resolution.

### Pillar 2 detail

- Script: `scripts/augment_rare_classes.py`.
- Transformations present in the train set: horizontal flip and CLAHE (`_aug_hflip`, `_aug_clahe`).
- 420 synthetic images added to train; labels transformed with the images.
- Augmentation is offline copies, not on-the-fly. Long term, prefer on-the-fly augmentation to avoid 420 near-duplicates.
- Crazing is 630 of 1,470 train images (43%). The retrain did **not** help the worst class: crazing mAP50 at 640 fell from 0.432 to 0.390.
- The retrain still won overall on the deployment path because the other four classes improved. It is promoted; the old `0.797`/`0.804` retrain comparison used `rect=True` and is withdrawn.

### Pillar 3 detail

- YOLO11m: 20.1M params, 68.1 GFLOPs vs YOLO11s 9.4M / 21.4 GFLOPs.
- Expected NEU-DET gain: +1-3 mAP50, roughly 2x training and inference cost.
- The 1,470-image retrain has been evaluated and promoted, so the confound is gone. Starting YOLO11m now requires explicit approval.
- YOLO11x is not appropriate for this laptop or the CPU deployment path.

### Pillar 4 detail (corrected)

**TTA** is real but modest, measured with `rect=False`:

- 640 deployment: 0.753 / 0.373.
- 1024 + TTA: 0.774 / 0.397, i.e. **+2.1 mAP50 / +2.4 mAP50-95**.
- Latency: 5.9 ms -> 34.5 ms model inference on GPU: roughly 6x.
- 640 TTA is worse. TTA belongs to the 1024 benchmark only.

**SAHI is not valid for this dataset.** The images are 200x200; a 640x640 slice covers the whole image, so slicing degenerates into a full-image pass plus a 224-pixel pass. The old report also misread its own output:

- 45 images, 37 total crazing boxes.
- That is **0.82 boxes per image**, not an 82% detection rate.
- Only **27 of 45 images** had any crazing detection (60%).
- The script never compares boxes to ground truth, so it computes no mAP and no recall.

SAHI has been removed from the claims and should not be used for NEU-DET.

---

## 5. Novelty Gate Status & Routing

`artifacts/models/gate_summary.json` now contains all five profiles after rehydration from the existing banks and audit JSON. The banks were not rebuilt or re-scored.

| Profile | Gate AUROC | Separation | Quality | Trusted | Gate can escalate? |
|---|---:|---:|---|---|---|
| bottle | 0.996 | 2.787 | strong | Yes | Yes |
| metal_nut | 0.973 | 1.723 | strong | Yes | Yes |
| transistor | 0.942 | 0.939 | good | Yes | Yes |
| screw | 0.611 | 0.279 | marginal | No | No |
| steel | 0.632 | -0.127 | not_validated | No | No |

Trust is derived from defect-vs-good separation because `separation_unknown` is null for the MVTec entries; the rehydration did not re-score the manifest `unknown` splits. The routing check below verifies the policy, not a per-profile unknown-set AUROC.

The steel gate is genuinely weak on held-out data:

- `rolled-in_scale` AUROC vs good: **0.632**.
- Per-class AUROC: crazing 0.559, pitted_surface 0.551, inclusion 0.683, patches 0.790, scratches 0.833.
- The built-in separation check is negative, so `profiles.py` hard-codes steel as `not_validated` regardless of the summary value.

### Exact routing logic in `server/app/fusion.py`

1. Any supported detection -> **FAIL**. Severity 1-2 -> `release_with_note`, 3 -> `rework`, 4 -> `quarantine`.
2. Low-support or low-confidence detections -> **REVIEW** (`transient_detection` or `low_confidence_detection`).
3. Trusted gate and `gate_z >= unknown_z` (0.75) -> **REVIEW** (`unknown_anomaly`).
4. Trusted gate and `gate_z >= pass_max` (0.55) -> **REVIEW** (`elevated_novelty`).
5. Otherwise -> **PASS** (`no_known_defect`).

If the gate is untrusted and there are no detections, the part is released and the warnings include `gate_not_validated` and `gate_does_not_gate_this_profile`.

Verified behavior with `gate_z=3.0` and no detections:

- bottle, metal_nut, transistor -> `review / human_review / unknown_anomaly`.
- screw, steel -> `pass / release / no_known_defect`.

This is the honest state of the system: the routing policy escalates an unknown anomaly on MVTec bottle/metal_nut/transistor, and the steel gate does not gate. Per-profile unknown-set AUROC remains pending a separate scoring run; the evidence above is defect-vs-good separation plus the policy check.

---

## 6. Completed Retrain (Promoted)

**Run:** `artifacts/runs/neu_yolo11s_1024_aug`  
**Started:** October 29, 2026, 12:10 local time  
**Finished:** October 29, 2026, 13:50 local time  
**Result:** 120/120 epochs; best validation epoch 110 (mAP50 0.7468 / mAP50-95 0.3919)  
**Decision:** promoted 2026-10-29; it wins the 640 deployment path

| Setting | Value |
|---|---|
| Model | `yolo11s.pt` |
| Data | `data/neu-det/neu-det.yaml` |
| Epochs | 120 |
| Image size | 1024 |
| Batch | 8 |
| Workers | 2 |
| Cache | disk |
| Device | 0 |
| Patience | 35 |
| Close mosaic | 15 |
| Seed | 0 |
| Optimizer | SGD |

Canonical test results and the promotion decision are in §3 and `artifacts/eval/retrain_decision.json`. The run is no longer active, so dataset repairs and GPU evaluation are unblocked.

```powershell
# reproduce the two evaluation paths with the deployment protocol (rect=False is the default)
python scripts/eval_canonical.py `
  --weights artifacts/models/detector_best.pt `
  --split test --imgsz 1024 --conf 0.35 --iou 0.6 --tta `
  --out artifacts/eval/canonical_yolo11s_1024aug_1024_tta_deploy.json

python scripts/eval_canonical.py `
  --weights artifacts/models/detector_best.pt `
  --split test --imgsz 640 --conf 0.35 --iou 0.6 `
  --out artifacts/eval/canonical_yolo11s_1024aug_640_deploy.json
```

---

## 7. Evaluation & Operating Threshold

### One script, one protocol

`scripts/eval_canonical.py` records:

- weights path and SHA-256;
- dataset and split;
- image size, NMS IoU, TTA flag, and `rect` mode (default `false`, matching deployment);
- mAP protocol (`conf=0.001`, IoU matching 0.50:0.95);
- per-class mAP50 and mAP50-95;
- precision and recall at the chosen operating confidence.

`--iou` is the NMS IoU, not the AP matching threshold. `--conf` is the fixed operating confidence used to read P/R from the full curves, so the reported operating point is not the max-F1 point. `--rect` is legacy only: Ultralytics rectangular validation inflates mAP on this dataset.

Do not compare numbers produced by different splits, NMS IoUs, confidence floors, or `rect` modes. The old invalid artifact is preserved as `artifacts/eval/test_metrics_invalid_legacy.json` (mAP50 0.505, `n_images: 0`). The active `artifacts/eval/test_metrics.json` is built by `scripts/export_eval_summary.py` from `canonical_yolo11s_1024aug_640_deploy.json` plus `conf_scan.json`, and reports mAP50 **0.7855 / mAP50-95 0.3961 / P@0.25 0.719 / R@0.25 0.747**.

The UI operating-point scan is current for the promoted 640 weights. `/api/eval/summary` serves the table only when `operating_scan_status=current`; rerun `scripts/scan_operating_points.py` after any weight or resolution change, otherwise it returns `operating_points: []` with a pending note.

### The server has two confidence floors

- `INDORE_CONF` (default **0.25**) filters detections inside the detector.
- The active SQLite threshold profile field `conf_min` (seeded **0.25**) filters tracks during fusion.

The selected operating point is **0.35**, set in code, in `.env` when overridden, and in the active SQLite profile:

```powershell
$env:INDORE_CONF = "0.25"

# while the server is running:
Invoke-RestMethod -Method Patch http://127.0.0.1:8000/api/thresholds `
  -ContentType "application/json" `
  -Body '{"conf_min": 0.25}'
```

Changing only `INDORE_CONF` leaves fusion dropping tracks below 0.35. The active profile also carries `pass_max=0.55`, `review_max=0.75`, and `unknown_z=0.75`; the API requires `0 < pass_max < review_max <= unknown_z`.

Pick the operating point from measured precision/recall, not from the mAP table. At 0.35, 95.1% of test images get a detection, 24.3% of held-out images fire, and 18.8% of synthetic good crops false-fire. At 0.55 those become 88.4%, 10.3%, and 7.8%. The 0.35 choice is recall-first: missed defects are more expensive than review load. Use 0.45 or 0.55 if false alarms dominate.

---

## 8. Retracted Claims

These appeared in earlier versions of this brief or the README and are not supported:

| Old claim | Correct statement |
|---|---|
| Test mAP50 0.802 at 640 | Inflated `rect=True` result. Deployment (`rect=False`) is **0.753 / 0.373** for the promoted retrain. |
| Test mAP50 0.797 at 640 / 0.804 at 1024+TTA | Both were `rect=True`. Corrected: **0.753 / 0.373** at 640 and **0.774 / 0.397** at 1024+TTA. |
| The 1,470-image retrain was rejected | That decision used `rect=True` and is reversed. Under the deployment protocol the retrain wins 640 (**0.753 vs 0.740**) and is now promoted. |
| Zero-shot 1024 test mAP50 0.505 | Invalid: `n_images: 0`. Discard. |
| 1024 training raised mAP50 | Not re-run under `rect=False`; the old 0.791/0.797 comparison is withdrawn. |
| Augmentation improved crazing | It did not: crazing mAP50 at 640 fell from 0.432 to 0.390. The retrain was promoted on the other four classes. |
| SAHI achieved ~82% crazing detection | 37 boxes / 45 images = 0.82 boxes per image; 27/45 images had any detection; no mAP was computed. |
| TTA +1.94% | +1.9 on validation (NMS IoU 0.7) under the old protocol; corrected test gain at 1024 is **+2.1 mAP50** with `rect=False`. |
| ONNX CPU 640x640 67.2 ms | The old 1024 ONNX could not support this. A 640 ONNX has now been exported and measured: **62.2 ms p50 / 69.8 ms p95**, 4 threads, model inference only. |
| YOLO11m + 1,470 aug -> 86-89% mAP50 | Hypothesis only. The YOLO11s retrain did not improve crazing; do not forecast a gain from the same data. |
| Held-out `rolled-in_scale` is escalated as UNKNOWN | Leakage is zero, but the steel gate is not validated (AUROC 0.632) and does not gate. Use bottle/metal_nut/transistor for the novelty demo. |
| `test_metrics.json` is always invalid | The old file was invalid. The current file is rebuilt from canonical JSON + scan and is valid for the promoted weights; canonical JSON remains the primary record. |

---

## 9. Repository Map & Artifacts

```
C:\Games\Indore\Indore\
├── artifacts/
│   ├── eval/
│   │   ├── legacy/                       # rect=True / old-model records; do not quote
│   │   ├── gate_audit_steel.json        # steel gate AUROC and per-class separation
│   │   ├── gate_audit_mvtec.json        # MVTec gate AUROC and separation
│   │   ├── canonical_*_deploy.json       # current rect=False evaluations (promoted + previous)
│   │   ├── canonical_yolo11s_1024aug_onnx_640.json  # promoted ONNX validation
│   │   ├── retrain_decision.json        # retrain comparison and promotion record
│   │   ├── test_metrics.json            # active 640 UI summary; operating scan current
│   │   ├── test_metrics_invalid_legacy.json  # old invalid file (n_images: 0); do not use
│   │   ├── conf_scan.json               # current 640 operating scan for promoted weights
│   │   └── sahi_tta_results/            # legacy SAHI output; claims retracted
│   ├── models/
│   │   ├── detector_best.pt             # promoted 1024-trained PyTorch weights (19.3 MB)
│   │   ├── detector_best.onnx           # static 1x3x640x640 FP32 ONNX (37.9 MB), deployment graph
│   │   ├── gate_steel.npz
│   │   ├── gate_bottle.npz
│   │   ├── gate_metal_nut.npz
│   │   ├── gate_screw.npz
│   │   ├── gate_transistor.npz
│   │   └── gate_summary.json            # all five profiles after rehydration
│   ├── runs/
│   │   ├── neu_yolo11s_1024/            # previous 60-epoch run; weights now legacy
│   │   └── neu_yolo11s_1024_aug/        # completed 120-epoch retrain; promoted
│   └── indore.db                        # SQLite WAL database
├── data/
│   ├── neu-det/
│   │   ├── images/ (train: 1,470, val: 225, test: 225)
│   │   ├── labels/ (train: 1,470, val: 225, test: 225)
│   │   ├── heldout/ (300 rolled-in_scale images)
│   │   ├── neu-det.yaml
│   │   └── splits.json                  # base 1,800 manifest; 420 train-only augs not listed
│   └── profiles/
│       └── manifest.json                # 5 profile definitions
├── ml/
│   ├── anomaly/
│   │   ├── build_banks.py               # now merges into gate_summary.json
│   │   └── gate.py                      # PatchCore-lite implementation
│   ├── data/prepare_neu.py
│   ├── eval/evaluate.py
│   └── train/train_yolo.py
├── scripts/
│   ├── augment_rare_classes.py          # offline CLAHE/flip augmentation
│   ├── eval_canonical.py                # canonical evaluation; rect=False by default
│   ├── export_eval_summary.py           # rebuild test_metrics.json from canonical JSON + scan
│   ├── rebuild_gate_summary.py          # rehydrate summary without re-scoring
│   ├── scan_operating_points.py         # regenerate conf_scan for the UI
│   ├── eval_sahi_tta.py                 # legacy TTA/SAHI suite; SAHI claims retracted
│   ├── smoke_test.py
│   └── e2e_verify.py
├── server/app/
│   ├── main.py
│   ├── player.py
│   ├── fusion.py
│   ├── gate_service.py
│   ├── detector.py
│   ├── db.py
│   └── profiles.py
├── web/dist/
├── DEMO.bat
├── yolo11s.pt
├── yolo11m.pt
└── OPTIMIZATION_CONTEXT.md
```

---

## 10. Operating Commands

### Launch the application

```powershell
.\DEMO.bat
```

Server: `http://127.0.0.1:8000`  
OpenAPI: `http://127.0.0.1:8000/docs`  
Live stream: `http://127.0.0.1:8000/live.mjpg`

### Verify the system

```powershell
python scripts/smoke_test.py
python scripts/e2e_verify.py
```

### Check gate trust after changing banks

```powershell
@'
from server.app.config import Settings
from server.app.profiles import ProfileRegistry

r = ProfileRegistry(Settings())
for p in r.all():
    print(f"{p.id:12s} quality={p.gate_quality:14s} trusted={p.gate_trusted}")
'@ | python -
```

Expected: bottle / metal_nut / transistor trusted; screw / steel untrusted.

### Rehydrate the gate summary without re-scoring

```powershell
python scripts/rebuild_gate_summary.py
```

Safe to run any time: it only reads existing `.npz` banks and the MVTec audit JSON.

### Post-retrain status

Done:

1. Both canonical evaluations (1024 + TTA and 640 plain) completed under `rect=False`; the retrain was promoted on the 640 deployment path.
2. UI operating-point scan regenerated for the new promoted 640 weights.
3. Operating point fixed at 0.35 in code and in the active SQLite profile.
4. README and this brief updated from the canonical JSON.
5. 640 ONNX exported from the promoted weights and benchmarked: **62.2 ms p50 / 69.8 ms p95**, 16.1 FPS on CPU with 4 threads.
6. UI rebuilt from the corrected sources.

Remaining:

1. Extend `splits.json` with augmentation metadata (420 train-only files) without changing the 1,800 base total; document or repair the `train/patches_101` / `val/patches_105` duplicate with a manifest update.
2. Decide separately whether YOLO11m is worth starting; the promoted retrain does not by itself justify it, and it requires explicit approval.
3. Label or move the old `rect=True` canonical JSON files so they are not mistaken for current results.
