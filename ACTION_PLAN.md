# Indore QA Cell — Action Plan

> **Superseded as of 2026-10-29.** The detector is trained and promoted; the 1,470-image retrain was evaluated and promoted on the 640 deployment path. Current commands and metrics live in `README.md` and `OPTIMIZATION_CONTEXT.md`. This file is historical.

## Trained Detector (H14 Gate — ~2 hours)

**Command:**
```bash
cd C:\Games\Indore\Indore
python ml\train\train_yolo.py --model yolo11s.pt --data data\neu-det\neu-det.yaml --epochs 200 --imgsz 640 --batch 16 --device 0 --workers 4
```

**Expected output phases:**
- Epochs 1-20: Loss decreasing, mAP climbing slowly (domain shift from COCO→steel)
- Epochs 20-100: mAP50 rising, per-class improvement (scratches & patches typically first)
- Epochs 100-150: Plateau or slight overfitting (monitor val mAP)
- Epochs 150-200: Final fine-tune, patience may trigger early stop

**Success criteria (H14 gate):**
- mAP50 ≥ 0.60 on 5 trained classes (on test split, NOT held-out)
- worst-class recall ≥ 0.70 (confusion matrix)
- If mAP50 < 0.45 at H11: check for pipeline bug (BGR↔RGB, label paths, normalization) — this causes 80% of failures

**Troubleshooting regime A (underfitting / pipeline bug):**
1. Visualize 20 predictions vs ground truth — is box on defect or background?
2. Check BGR vs RGB mismatch at inference vs training
3. Letterbox / normalization mismatch — use ultralytics predictor, don't hand-roll
4. Frozen backbone — verify requires_grad printout (default None = full fine-tune)
5. LR too low — loss decreases <20% over 20 epochs → lr0 0.01 → 0.02, or switch to AdamW 0.001
6. Too few epochs — val mAP still rising at end → 200 → 300, patience 70

**Expected gain from fixing real pipeline bug: +10 to +40 mAP.** Check before anything else.

**Troubleshooting regime B (overfitting):**
1. Freeze backbone early: `freeze=10` for first 20 epochs, then unfreeze → +1–3 val mAP
2. Stronger augmentation: scale 0.5, translate 0.15, degrees 10, enable copy_paste 0.1 → +1–3
3. More data: GC10-DET or MVTec textures as extra negatives → +2–5
4. Lower LR / longer warmup: lr0=0.005, warmup=5 → +0.5–2
5. Early stop tighter: patience=25 → protects result, no accuracy gain

**Troubleshooting regime C (instability):**
1. Loss spikes every few epochs: lr0 0.01 → 0.005, warmup_epochs 5
2. NaN loss: AMP overflow → amp=False for 5 epochs to confirm, then reduce lr
3. Val mAP oscillates ±0.05: batch too small → nbs=32 accumulation
4. Diverges at epoch 0: bad labels (box outside image, w/h=0) — run §4 data QA again

**Accuracy ladder — ranked by gain per hour (apply in order, stop at target):**
1. Fix a data/label/eval bug → 30 min → +10–40 mAP (None)
2. imsz 640 → 800 → 1.5 h train → +1–3 mAP (VRAM: batch 8)
3. Train 3 seeds, ensemble at inference → 3 h → +1–2 mAP (Latency 3×)
4. 200 → 300 epochs, patience 70 → +1 h → +1–2 mAP (Overfit risk)
5. Copy-paste rare-class defects → 1 h → +1–3 on weak class (Realism)
6. Hard-negative mining (10% background images) → 1 h → −20–40% FP (Slight recall dip)
7. TTA at inference (augment=True) → 5 min → +1–2 mAP (Demo only, 2–3× slower)
8. YOLO11s → YOLO11m → 3 h train → +2–3 mAP (8 GB VRAM marginal, slower)
9. Distill YOLO11m → YOLO11s → 4 h → +1–2 mAP (Real risk of not finishing)
10. SAHI tiling at inference → 2 h → +2–5 on high-res (4–9× compute)

**Stop rule**: If after items 1–3 you are still >5 points below target (mAP50 < 0.55), renegotiate the claim on the slide (report per-class recall at a fixed FPR instead of a single mAP) and spend the time on the demo. Judges reward honesty about limits far more than an extra 2 points.

---

## ONNX Export & CPU Inference (~30 min)

**Command:**
```bash
cd C:\Games\Indore\Indore
python ml\export\export_onnx.py --weights artifacts\models\detector_best.pt --imgsz 640
```

**Verify:**
- `artifacts\models\detector_best.onnx` exists
- Size ≤ 25 MB: `dir artifacts\models\detector_best.onnx`
- ONNX Runtime CPU inference test:
  ```bash
  python -c "
  import onnxruntime as ort
  sess = ort.InferenceSession('artifacts/models/detector_best.onnx', providers=['CPUExecutionProvider'])
  import numpy as np
  import cv2
  img = cv2.imread('data/neu-det/images/test/crazing_1.jpg')
  img = cv2.resize(img, (640, 640))
  x = img.transpose(2, 0, 1).astype(np.float32) / 255.0
  x = np.expand_dims(x, 0)
  result = sess.run(None, {'images': x})
  print('ONNX inference output shape:', [o.shape for o in result])
  print('ONNX SUCCESS')
  "
  ```

**INT8 export (fallback path):**
```bash
python ml\export\export_onnx.py --weights artifacts\models\detector_best.pt --imgsz 640 --int8
```
**Known trap**: INT8 scores collapse to 0 (all confidences ≈ 0). This is because the output tensor mixes box coords (px) and class scores (0–1) under one scale. Requires excluding non-Conv/Gemm nodes from quantization, and recalibrating conf and anomaly thresholds on quantized output.

**Expected CPU latencies (from TRAINING_SPEC.md §4.3):**
- GPU latency (TRT FP16): p95 ≤ 8 ms, batch 1, 640
- CPU latency (ONNX INT8): p95 ≤ 30 ms, batch 1, 640
- If FPS < 12: imsz 416→320, run detector every 2nd frame, keep gate every frame — slight box jitter

---

## End-to-End Pipeline Verification (~1 hour)

**Start the server:**
```bash
cd C:\Games\Indore\Indore
python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000
```

**Web UI:** open http://127.0.0.1:5173

**API verification points:**

1. **Health**: GET http://127.0.0.1:8000/api/health
   - Expected: status=ok, profile=steel, fps≥15, detector_mode=auto, gate_ready=True/False (steel = False)

2. **Profiles**: GET http://127.0.0.1:8000/api/profiles
   - Expected: ≥5 profiles, steel has gate_quality=not_validated, gate_trusted=False

3. **Thresholds**: GET http://127.0.0.1:8000/api/thresholds
   - Expected: pass_max=0.55, review_max=0.75, unknown_z=0.75, conf_min=0.55

4. **Metrics**: GET http://127.0.0.1:8000/api/metrics
   - Expected: total inspections, verdicts (pass/review/fail), fp_per_1000, unknown_routing_rate, latencies p50/p95/p99

5. **Inspections**: GET http://127.0.0.1:8000/api/inspections?status=all
   - Expected: list of inspections with verdicts, detections, anomaly_z

6. **Review queue**: GET http://127.0.0.1:8000/api/review/queue
   - Expected: open reviews (operator_disposition IS NULL), count

7. **Control**: POST http://127.0.0.1:8000/api/control
   - Body: {"action": "pause"} → pauses pipeline
   - Body: {"action": "resume"} → resumes pipeline

8. **Thresholds PATCH**: POST http://127.0.0.1:8000/api/thresholds
   - Body: {"pass_max": 0.50, "review_max": 0.70} → updates thresholds

9. **Models**: GET http://127.0.0.1:8000/api/models
   - Expected: runtime.detector.engine (yolo11s-onnx or yolo11s-pt), runtime.device (cuda or cpu)
   - runtime.gate.ready, runtime.gate.device, runtime.promoted_weights

10. **FrameResult contract**: Verify sample serializes/round-trips through FrameResult model (see smoke_test.py §7)

**End-to-end workflow test:**
1. Send a known-defect frame via the pipeline (or use dataset replay)
2. Verify /api/inspections shows new inspection with FAIL verdict
3. Check /api/review/queue shows the inspection in open reviews
4. POST /api/inspections/{id}/review with {"disposition": "true_defect", "label": "scratches"}
5. Verify inspection final_verdict=fail, operator_disposition=true_defect
6. Check /api/audit shows review.true_defect event
7. Check /api/outbox shows pending outbox event (inspection.closed, possibly mes.route)

**Withheld-class test (the money demo):**
1. Use the [Review] tab in the web UI to switch source to the withheld-class clip
2. Or send test anomaly via control action
3. Verify verdict=REVIEW, anomaly_z ≥ unknown_z (0.75 by default)
4. Verify routing = {"destination": "human_review", "reason": "unknown_anomaly"}
5. Verify anomaly score is 3.4× threshold (demonstrated on held-out frames)

---

## Gate Bank Notes (~15 min)

**Current status (from gate_summary.json):**
- metal_nut: separation_defect=1.72, separation_unknown=1.72, gate_trusted=True ✅
- screw: separation_defect=0.10, separation_unknown=0.10, gate_trusted=False (weak) — barely trusted ⚠️
- transistor: separation_defect=0.94, separation_unknown=0.94, gate_trusted=True ✅
- bottle: separation_defect=2.85, separation_unknown=2.85, gate_trusted=True ✅
- steel: separation_defect=-0.27, separation_unknown=-0.14, gate_trusted=False ✅ (negative separation — shown transparency only)

**If positive-separation steel gate is needed:**
```bash
python ml\anomaly\build_banks.py --manifest data\profiles\manifest.json --profiles steel
```
This will rebuild the gate bank using the configured bank_crops from neu-good. Note: the negative separation is due to the clean-crop bank scoring defects below good in ResNet18 feature space for steel surface texture. The domain gap between NEU-DET steel and MVTec AD steel is the likely cause.

**Recommended approach**: Rely on MVTec AD categories (metal_nut, screw, transistor, bottle) for positive-separation gate demo, and use the withheld `rolled_in_scale` class from NEU-DET for the novelty demonstration (this is the "money moment" regardless).

---

## Demo Script & Slide Deck Preparation (~6 hours)

**Demo script (3 min, tape to laptop):**
Follow the script in MASTER_PLAN.md §10, pages 492-518. Key beats:
- 0:00 [Live] — hook, boxes appear
- 0:20 FPS counter + latency waterfall (measured live, not estimated)
- 0:45 Drop scratched part → FAIL
- 1:00 [Review] — evidence crops + heatmap; "True defect" → KPI increments
- 1:30 Withheld-class clip → UNKNOWN → REVIEW → "fails safe, not silently"
- 2:10 [Metrics] — p50/p95, defect counts, FP/1000, cost saved
- 2:30 Threshold slider — "recall vs inspector load is a business decision"
- 2:45 [Slides] → architecture → close

**2-second spacebar alternative** for test anomaly injection (in case trackpad/projector misbehaves).

**Slide deck (9 slides, ≥18 pt):**
Follow MASTER_PLAN.md §11, pages 530-569. Key slides:
1. Title + team + roles + QR to repo
2. Problem: scrap rate, cost per escaped defect, inspector attention decay + one real scratch photo
3. Why still manual: rule-based vision brittle; supervised needs labels per defect; new part = 3-month project
4. Architecture diagram (large, emphasize "unknown defect" layer — this is the innovation)
5. Results: per-class table + latency table, hardware named (i7-1260P, CPU-only, ONNX INT8, imgsz=416, n=300 frames)
6. Novelty gate: withheld-class experiment (detector sees nothing → gate scores 3.4× → REVIEW → "fails safe, not silently")
7. Real-time proof: latency waterfall (capture 3ms + preprocess 4ms + inference 18ms + post 2ms + encode 6ms = 33ms p95, on laptop CPU only)
8. Deployment & scale: edge node per cell, events-only uplink (MQTT/OPC-UA), stateless streams, model registry/OTA, PLC reject signal, ROI math with stated assumptions
9. Roadmap / ask: 2-week shadow-mode pilot, success criteria, what's needed

**Delivery rules (from MASTER_PLAN.md §11, pages 542-568):**
- 20 seconds max per answer
- Lead with the number or the mechanism, never "great question"
- If you didn't measure it, say "we didn't measure that — here's how we would"
- Q1: "How is this different from a normal CNN classifier?" → "Three layers. A detector gives boxes for known classes. An embedding-distance gate trained on good parts only scores novelty — different model, different failure mode. A cost-weighted fusion policy turns both into pass/review/fail, and the review action is a labeled data point. A plain classifier can only say 'not one of my six classes,' which in a factory is indistinguishable from 'good.' We make that distinction explicit — and I can show you the frame where it matters."
- Q2: "What happens with a new defect type?" (the money question) → "Watch. This class was deliberately removed from training. The detector produces nothing — no vocabulary for it. The gate scores it 3.4× over threshold, so the verdict is REVIEW, not PASS, and it's routed with a heatmap. In production that confirmation becomes a labeled sample; after N samples we promote it to a supervised class. It fails safe and bootstraps its own training set."
- Q3: "How long to onboard a new part?" → "Two modes. Novelty gate: 20–40 good images, about five minutes on CPU — works the day the part arrives. Full supervised classification: 50–100 labeled images per class, ~40 minutes of fine-tuning. Compare that with the 6–12 weeks a traditional vision project takes."
- Q4: "False positive rate — and what does it cost?" → "2.1% of good parts flagged at our demo threshold — measured, not estimated. At 1,800 parts/hour that's ~38 extra reviews/hour, ~12 seconds each, ~8 minutes of inspector time per hour. One escaped defect costs 100× a rework. So the threshold is a *business* parameter, and we expose it as a slider so you can watch recall and queue load move together."
- Q5: "Is this actually real-time?" → "33 ms per frame end-to-end on this laptop, CPU only, ONNX INT8 at 416 — p95 is 48. That's the waterfall on screen: capture 3, preprocess 4, inference 18, post 2, encode 6. That's 300 frames of live measurement, not an estimate. On Jetson-class hardware the inference portion drops to single digits, which clears a 30 ppm line comfortably."
- Q6: "Accuracy on real production data?" → "I won't claim a number I haven't measured. 0.7 mAP50 on the public NEU-DET benchmark, and zero detections but full novelty detection on the deliberately held-out class — which is the honest and useful failure mode. Our own line is a simulated feed, clearly labeled. A 2-week shadow-mode pilot is how you'd get a real FP rate before trusting a single reject signal."
- Q7: "How does this scale to 100 cameras?" → "It's stateless per stream. Each cell gets an edge node; frames never leave it — that's both a latency and an IP argument. Only events go uplink over MQTT/OPC-UA: a few hundred bytes per second instead of a video wall. The center holds a model registry and an event store: OTA updates, per-cell A/B, drift monitoring. Adding a camera is a config line."
- Q8: "Why won't this fail like the last vision-QA startup?" → "Their failure mode is known: label everything, hit a new defect type, start over — or rack up false positives until the line ignores the system. We attacked both. Novelty detection makes the system useful *before* you've labeled anything and honest when it doesn't know. The cost-weighted threshold makes the false-positive budget a line-manager decision with a visible number."
- **Delivery rules**: 20 s max per answer. Lead with the number or the mechanism, never "great question." If you didn't measure it, say "we didn't measure that — here's how we would." If you did measure it, say the number and the machine it was measured on.

**Compressed 24-hour variant (from MASTER_PLAN.md §9.1):**
- 0–1: Scope + contract + scaffold — schema frozen
- 1–3: Walking skeleton: camera → fixture boxes
- 3–7: Train detector (40 epochs @512), swap stub → real — mAP ≥ 0.55
- 7–10: ONNX + INT8 + benchmark; overlay polish; verdict badge — ≥20 FPS, latency HUD
- 10–13: PatchCore gate + fusion + review action (in-memory, no SQLite) — UNKNOWN → REVIEW works
- 13–15: KPI tiles, README, results table — Feature complete
- 15–17: Rehearsal #1, fix P0, record video — Script frozen
- 17–21: Sleep 4 h, staggered pairs
- 21–22: Rehearsal #2 in the real room; fallback video; **Submit**
- 22–24: Deck polish, Q&A drill, pitch — Demo

**24 h keeps**: overlay, latency HUD, FPS, gate + UNKNOWN routing, review action, withheld class, fallback video, one-command start.

---

## H40 Demo Freeze Checklist (from MASTER_PLAN.md §14)

**Repo:**
- [ ] `README.md` ≤200 lines: what it is, one architecture diagram, results table, latency table, 3-command quickstart, **"what's real vs simulated"** section
- [ ] `demo.ps1` starts everything from a clean clone in <5 min (verified on a machine that has never seen the repo, at H34)
- [ ] `ml/weights/*.onnx` committed (<50 MB, no LFS)
- [ ] `docs/`: SCOPE, RUNBOOK, DEMO_SCRIPT, PITCH, RESULTS, RISKS
- [ ] `smoke_test.py` passes on a clean checkout
- [ ] License file (MIT/Apache-2.0); note that Ultralytics is **AGPL** — fine for a hackathon, a problem for a closed product. Saying this out loud earns credibility

**Video (2:45–3:00, recorded H35, captions burned in):**
- 0:00 problem with a number → 0:15 live line with boxes → 0:40 a FAIL and the review action → 1:10 withheld class → UNKNOWN → routed → 1:40 latency HUD close-up → 2:10 architecture + scale → 2:40 impact + close

**Stage kit:**
- [ ] Fallback video (plays without the app)
- [ ] Slides as PDF **and** PPTX **and** a cloud link
- [ ] USB: git bundle, weights, video, deck, wheel cache, `RESTORE.md` (5 numbered steps)
- [ ] Printed one-pager with QR codes
- [ ] Printed demo script taped to the laptop
- [ ] Spare USB camera, spare cable, charger, hotspot

**Printed one-pager with QR codes** — summary of: project purpose + key number (mAP50, FPS, FP/1000) + unknown routing rate + threshold slider demo

---

## Quick Reference: One-Command Start

```bash
cd C:\Games\Indore\Indore
.\scripts\demo.ps1
```

**What this does (per demo.ps1):**
1. Kills stale process on port 8000
2. Verifies Python environment (fastapi, uvicorn, cv2, numpy, pydantic, torch)
3. Verifies web dist built (web/dist/index.html)
4. Configures environment: INDORE_PROFILE=steel, INDORE_SOURCE=dataset, INDORE_DETECTOR=auto
5. Launches uvicorn server at 127.0.0.1:8000
6. Prints cell active status + API docs URL

**After launch:**
- Browser F11 fullscreen at 127.0.0.1:5173
- Tabs in order: [Live] [Review] [Metrics] [Slides]
- Verify: green WS dot · FPS ≥ 15 · camera LED on · battery > 70% + plugged in
- Fallback video open in second window, paused at 0:00, one click away
- **Do Not Disturb ON.** Slack/email closed. Phone silent.

---

## Risks & Mitigation (from MASTER_PLAN.md §13)

| Risk | Trigger | Mitigation | Owner |
|------|---------|------------|-------|
| Dataset won't download | >20 min blocked | Mirrors pre-found + USB copy + ultimate fallback: 60 self-taken photos (40 min) | ML-A |
| No GPU / CUDA missing | `torch.cuda.is_available()` False | **CPU-first by design**; train on Colab/Kaggle if needed, ship `.onnx` | ML-A |
| Too slow for "real-time" | p95 > 80 ms or FPS < 12 | 416→320, INT8, frame-skip 2, 2 classes. Report inference latency and end-to-end FPS separately | ML-B/BE |
| Bad labels | mAP50 < 0.45 at H11 | NEU-DET ships labeled. Debug order: label paths → class mapping → BGR/RGB + normalization mismatch → leakage → hyperparameters. First three cause 80% of failures | ML-A |
| Venue network dies | Any fetch at demo time | Weights in repo, node_modules installed, fonts vendored, no CDN. Offline test at H18 and H44 | INT |
| Bad labels | mAP50 < 0.45 at H11 | NEU-DET ships labeled. Debug order: label paths → class mapping → BGR/RGB + normalization mismatch → leakage → hyperparameters. The first three cause 80% of failures | ML-A |
| **Holdout class leaks into training** | Any run includes 6 classes | `heldout.txt` + `assert` in converter; INT greps before H26 | ML-A/INT |
| Venue network dies | Any fetch at demo time | Weights in repo, node_modules installed, fonts vendored, no CDN. Offline test at H18 and H44 | INT |
| Camera/projector failure on stage | 30 s with no frames | `SOURCE=video` one-flag fallback, spare USB camera, hotspot, fallback video rendered by H38. Rehearse the switch — 5 seconds | INT |
| Merge hell / `main` red at H39 | `main` broken >20 min | Ownership table, integration windows at H8/14/20/26/32/36/40, INT-only merges, revert-on-red | INT |
| Team collapse | A role misses 2 gates | Cross-train at H16; INT can drive the demo alone by H32 | INT |
| Laptop dies | — | Repo on GitHub + 2 local clones + USB; one designated demo machine from H32 | INT |
| Domain gap: training images ≠ demo camera | Model fails on the live part | Capture 40 normals **from the demo rig** and retrain the gate (5 min); ROI crop; lock exposure/WB | ML-B |
| Slides eat 6 hours on the last night | Deck not started by H36 | Deck generated from README metrics; max 3 h of design, ever | INT |

---

## Success Metrics (from MASTER_PLAN.md §12)

**Model table (12.1):**
| Class | Images | mAP50 | Precision | Recall |
| crazing | 45 | (target) | (target) | (target) |
| inclusion | 45 | (target) | (target) | (target) |
| patches | 45 | (target) | (target) | (target) |
| pitted_surface | 45 | (target) | (target) | (target) |
| scratches | 45 | (target) | (target) | (target) |
| **held-out: rolled-in_scale** | **n/a — never trained** | — | — | **anomaly recall @ threshold** |

**Operational table (12.2):**
| Metric | Target | Why it matters |
| FP per 1,000 good parts | < 20 excellent, < 50 credible | "How often do you stop my line for nothing?" |
| Recall per defect class @ that FPR | > 0.85 on the two main classes | "Which defects do you miss?" |
| Escapes per 1,000 bad parts | < 30 | "How many bad parts ship?" |
| Latency p50 / p95 / p99 | p95 < takt time | "Does it fit my cycle time?" |
| Throughput (FPS) | ≥ 15 sustained | "Is it real-time?" |
| Image AUROC / AUPRO | 0.95+ / 0.85+ | Comparability with papers |
| Unknown-routing rate | 2–8% of defects | Is the gate too loose or too tight? |
| Operator reject rate of auto-detections | < 10% | Is the system trusted? |

**Benchmark**: 50 warmup iterations, then 300 measured, batch 1, on the **demo machine**, at **demo resolution**, reported as p50/p95/p99. Never quote mean.

---

## Meta-Hack (from MASTER_PLAN.md §16, page 657)

> all five criteria move with one thing — *a demo that runs perfectly while you talk about a specific measured number.*

This is the key: the presentation score is mostly variance reduction. A demo that runs flawlessly while you consistently mention measured numbers (mAP50 on i7-1260P, ONNX INT8 p95 latency, FP/1000 on 300 frames) will score significantly higher than a more accurate demo with inconsistent measurements.

**The winning formula**: Demo runs perfectly → every number on screen is stamped with hardware → presenter talks to those specific numbers → judges remember the numbers, not the edge cases.

---

## End

Follow this action plan in order:
1. **Train YOLO11s** → achieve mAP50 ≥ 0.60 (H14 gate, ~2 h)
2. **Export ONNX** → verify CPU inference (30 min)
3. **Run E2E pipeline** → verify all API endpoints + workflow (~1 h)
4. **Prepare demo script + slides** per master plan (~6 h)
5. **Rehearse** and submit before H40 freeze

The project is 80% built infrastructure-wise. The remaining 20% is focused work: model training, pipeline verification, and demo preparation. All are achievable within the remaining time.