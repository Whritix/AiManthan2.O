# RESEARCH DOSSIER
## Real-Time AI Quality Assurance & Defect Detection — evidence base for the build

> **Purpose:** everything we learned, with sources, so decisions are traceable and defensible in front of judges.
> **Compiled:** 2026-10-27. **Method:** five parallel research passes (model landscape, data strategy, system architecture, edge deployment, hackathon strategy) + live link verification.
> **Read this with:** `MASTER_PLAN.md` (what we build), `TRAINING_SPEC.md` (how we train), `DATASETS.md` (what we download).

---

## 0. Confidence legend

Every claim below carries one of these. Be equally honest on stage.

| Tag | Meaning |
|---|---|
| **[H]** High | Peer-reviewed, widely reproduced, or directly measured by us |
| **[M]** Medium | Published in a recent paper/benchmark; protocol may differ from ours |
| **[V]** Verify | Collected by an automated research pass; **spot-check before quoting in the deck** |
| **[O]** Observed | We verified this ourselves on 2026-10-27 (HTTP status, file counts, repo state) |

**Rule:** never say a **[V]** number out loud without checking the source first. A manufacturing judge will know the real figure.

---

## 1. Executive findings

1. **MVTec AD is saturated and no longer discriminates.** Every competitive method is ≥99% image AUROC. The field moved to **MVTec AD 2**, where the best 2026 result is ~57 SegF1 — and the same models that score 99 on MVTec AD score **53–59 AU-PRO** on AD2. **[H]**
2. **The domain gap is the real problem, not the architecture.** Going from catalog-style MVTec AD to AD2's lighting-shifted test set costs **35–43 AU-PRO points** across every method. Whatever you train on must resemble what your camera sees. **[H]**
3. **Architecture innovation has lost to backbone quality.** Recent challenge analysis concludes the shift is "from designing increasingly complex anomaly-detection architectures toward exploiting very strong general-purpose visual representations" (DINOv3 + memory bank + post-processing). **[M]**
4. **Efficiency is now a first-class metric.** The 2026 challenge introduced a geometric-mean efficiency score combining quality × throughput × memory × power. Judges increasingly ask "FPS at what resolution?" **[M]**
5. **The 2026 winners are unusable for a live demo.** Best AD2 entry runs at **~0.12 fps** on the challenge edge hardware; runner-up ~0.25 fps. That is the opposite of "real-time." **[V]**
6. **Thresholding, not AUROC, is what breaks in production.** Challenge scoring switched to SegF1 precisely because binarization is unsolved. A 99-AUROC model with a bad threshold ships 5% false rejects. **[H]**
7. **Supervised detection wins on *assembly* defects**, where pixel-level anomaly detection is structurally blind ("the screw is present but wrong" is locally normal everywhere). A 6-class YOLO hits 90–96 mAP50 on PCB-style assembly defects. **[M]**
8. **Synthetic-only training loses; synthetic-as-filler wins.** Every 2025–2026 study found the same: synthetic-only AP ~0.39–0.45 vs real-only ~0.61–0.65. Best regime is fine-tune on synthetic → train on 50–100% real. **[M]**

**Net effect on our design:** a two-model cascade (supervised detector + unsupervised novelty gate) with a calibrated three-tier verdict, trained on domain-matched captures, evaluated at an operating threshold. Not the SOTA model. The *deployable* model.

---

## 2. Benchmark landscape

### 2.1 Dataset difficulty (why your 99% will not survive contact with a real camera)

| Method | MVTec AD AU-PRO | VisA | **MVTec AD 2 AU-PRO** | Drop |
|---|---|---|---|---|
| PatchCore | 92.7 | 79.7 | **53.8** | −39 |
| RD (reverse distillation) | 93.9 | 89.0 | 53.0 | −41 |
| RD++ | 95.0 | 89.3 | 54.3 | −41 |
| **EfficientAD** | 93.5 | 94.0 | **58.7** | −35 |
| MSFlow | 91.2 | 88.4 | 52.7 | −39 |
| SimpleNet | 89.6 | 68.9 | 46.4 | −43 |
| DSR | 90.8 | 68.1 | 49.0 | −42 |

Source: MVTec AD 2 paper (IJCV 2026). **[H]**

**Use this table on a slide.** It is the single most persuasive argument that "99% on MVTec" is meaningless and that domain-matched training is the real engineering problem.

### 2.2 Detection vs segmentation vs efficiency on AD2

| Entry | SegF1 (AD2 avg) | Throughput on edge HW |
|---|---|---|
| SuperADD (2026 winner) | 57.4 | **0.12 fps** |
| ISVL++ (runner-up) | 58.3 | **0.25 fps** |
| Forgis baseline | 47.4 | 4.0 fps |
| EfficientAD @256 (A6000 reference) | 58.7 AU-PRO | **614 fps** |

Source: 2026 VAND challenge report. **[V]** — the throughput figures are the important part and are consistent with the model designs.

**The gap between row 1 and row 4 is our entire pitch:** be the team that reports honest milliseconds at a stated resolution, not the team that quotes a benchmark number it cannot run.

---

## 3. Model zoo

### 3.1 Unsupervised / few-shot anomaly detection (normal images only)

| Method | Venue | MVTec AD I-AUROC | VisA | AD2 | Train cost | Latency | Edge | Verdict for us |
|---|---|---|---|---|---|---|---|---|
| **PatchCore** | CVPR'22 | 99.1 | 95.1 | 53.8 | **0** (feature bank) | 32 ms / 76 fps | ★★★★ | **✅ Chosen** — zero training, free heatmap |
| **EfficientAD** | WACV'24 | 99.1 | 98.1 | **58.7** | 0.3–1 GPU-h/cat | **2.2 ms / 614 fps** | ★★★★★ | Best speed/accuracy; per-category training cost |
| SimpleNet | CVPR'23 | 99.6 | 96.8 | 46.4 | ~1 GPU-h | 12 ms | ★★★★ | Weakest on AD2 |
| GLASS | ECCV'24 | 99.9 | SOTA | — | 2–3 GPU-h | ~20 ms | ★★★ | Accuracy flex only |
| Dinomaly | CVPR'25 | 99.6 multi-class | 98.9 | — | 10–20 GPU-h | 15–30 ms | ★★ | Great "one model, 15 classes" story |
| INP-Former++ | CVPR'25 / 2026 | 99.8 | 99.0 | 21.8 SegF1 | 5–10 GPU-h | 20–30 ms | ★★ | SOTA on old benchmarks, poor on AD2 |
| RD++ | CVPR'23 | 99.4 | ~96 | 54.3 | ~2 GPU-h | ~20 ms | ★★★ | Solid, no reason over EfficientAD |
| DRAEM | ICCV'21 | 98.0 | ~87 | — | 4–8 GPU-h | 10–20 ms | ★★★ | Legacy; needs synthetic defects |
| **AnomalyDINO** | WACV'25 | **96.6 one-shot** | ~87 | — | **0** | 50–100 ms | ★★★ | "New part from 1 image" demo moment |
| SubspaceAD | CVPR'26 | ~99 class | — | 46.9 SegF1 | 0 | 1–4 fps | ★★ | Steal the PCA-on-DINOv3 idea, not the runtime |

**[H] for pre-2025 venues; [M] for 2025–2026 entries; [V] for the exact AD2 numbers.**

### 3.2 Supervised defect detection

| Method | Defect benchmark | Speed | Verdict |
|---|---|---|---|
| **YOLO11n / s** | NEU-DET **80.1 mAP50** (v11n) | 2–12 ms | **✅ Chosen** — best accuracy per training hour |
| YOLOv8n | NEU-DET 77.1 mAP50 | 2–17 ms | Stable fallback |
| YOLO26n | COCO 40.9 mAP; NMS-free head | 1.7 ms | Newer, but export tooling less proven |
| RT-DETR | NEU-DET 74.7 mAP50 | 5–10 ms | No advantage here |
| Faster R-CNN / HD-YOLO | NEU-DET up to 86–87 mAP50 | 30–80 ms | Accuracy flex, latency loss |

Source: HD-YOLO comparison table (arXiv 2412.03969). **[M]**

### 3.3 Zero-shot / VLM / foundation segmentation

| Method | MVTec AD (zero-shot) | Verdict |
|---|---|---|
| WinCLIP (CVPR'23) | 91.8 I-AUROC | Baseline only |
| AnomalyCLIP (ICLR'24) | 91.5 / 96.2 AP | Best cheap zero-shot baseline |
| AnomalyGPT (AAAI'24) | 94.1 I-AUC, but **seconds/image** | Demo-only, not real-time |
| SAM 2 / SAM 3 | promptable | **Mask refinement on crops only**, never per-frame |
| YOLO-World / YOLOE | open-vocab ~35 AP | Text prompts like "scratch" **do not work**; use to find parts, not defects |
| Grounded-SAM-2 | — | 1–2 h install; not worth it in 48 h |

**[H]**

### 3.4 The 2026 SOTA, and why we are not building it

The winner (SuperADD) is DINOv3 + tiling + SAM refinement, 57.4 SegF1 at 0.12 fps. Reproducing it is 2–3 days of infrastructure for a result you cannot demo live.

**Steal the ideas, not the runtime:**
- **Tiling** on high-resolution frames (the single biggest AD2 lever — "resolution matters")
- **Morphological post-processing + connected components** before the decision (the winners all do this)
- **SAM refinement** on the defect crop only, at a few Hz (a stretch goal, not a dependency)

---

## 4. The four insights that shaped the design

### 4.1 Benchmark saturation is real; your demo will not be on MVTec
Covered in §2.1. Practical consequence: **capture 40–60 normal frames from the actual demo rig and rebuild the memory bank.** PatchCore takes 2 minutes; this single step typically recovers most of the domain gap and is the highest-ROI action in the entire project.

### 4.2 Thresholding is the deployment problem
AUROC is threshold-free; production is not. The correct procedure:
- Split normals into bank / calibration / test.
- Set `tau = quantile(score_calibration, 0.99)`.
- Report **precision, recall, F1, and FP per 1,000 parts at that tau**.
- Add a grey zone (p95–p99.5) → `REVIEW` instead of forcing a binary call.

This is also why our score is **normalized**: `z = (raw − p50_good) / (p99_good − p50_good)`, so `z = 1.0` means "worse than 99% of good parts" — a sentence a plant manager understands instantly.

### 4.3 Accuracy and throughput are on opposite ends
The AD2 leaderboard makes this explicit. Any honest system reports **p50/p95 latency at a stated resolution** and **sustained FPS**, not a mean. We report p95 because one 120 ms frame per hundred is what causes a missed reject.

### 4.4 Unsupervised AD is structurally blind to assembly defects
"Missing screw," "wrong connector," "gasket misaligned" are *logical* anomalies — every local patch looks normal. This is the MVTec LOCO problem class. Fixes:
- A supervised detector for the known assembly classes (cheap, near-perfect)
- Explicit **rule-based checks** (count parts, verify presence in an ROI) feeding the same decision engine
- Say this out loud to judges — it is the kind of limitation awareness that earns credibility

---

## 5. Synthetic defect research

### 5.1 The three rules

1. **Synthetic-only loses.** ScratchSim (2026): synthetic-only mAP50 0.451 vs real-only 0.610. SynSur (2026): AP 0.393 vs 0.652. **[V]**
2. **Realism matters less for *detection* than for *classification*.** DRAEM's own paper states its simulated anomalies only need to be *just-out-of-distribution*, not realistic. If the head is "find the defect," ugly blobs work. If the head is "which of 6 classes," morphology must match or you teach the wrong prior. **[H]**
3. **The test set comes from a phone, so the training set must contain phone physics** — white-balance drift, handheld blur, JPEG, specular highlights, a background that is not the dataset's backdrop.

### 5.2 Recipe → defect class map

| Recipe | Generates | Key signal it must reproduce |
|---|---|---|
| Random-walk crack | hairline crack, crazing | 1–3 px, branching, low contrast, dark core + faint bright edge |
| Bézier scratch | scratch, abrasion | straight-ish, tapered ends, **directional specular**, parallel micro-grooves |
| fBm/Perlin blob + height map | dent, bump, orange peel | low-frequency, soft edges, shading only, no albedo change |
| Poisson-disc pit cluster | blowhole, pitting, inclusion | small dark blobs, bright rim |
| CutPaste / CutPaste-Scar | foreign material, tears | boundary discontinuity, texture mismatch |
| DRAEM blend (texture × Perlin mask) | any OOD anomaly | "just-out-of-distribution" appearance |
| Poisson / MIXED_CLONE | stains, discoloration, spatter | gradient continuity, no seam |
| **Geometric edit (copy-move a component)** | **assembly defects** | structure, not texture — the only cheap way |
| Contour perturbation | burr, edge chip, flash | jagged silhouette |

**The forgotten one is assembly.** Pixel-noise synthesis cannot produce "missing screw." Cut a component, translate 2–20 px / rotate ±18°, paste back with a contact shadow, and also produce the "missing" variant.

### 5.3 The technique that makes procedural defects look real

**Relief shading.** Render the defect as a *height map*, compute normals via Sobel, apply Lambert + optional specular, then composite only inside the defect alpha. A scratch is visible mainly because it reflects light at a different angle than the surrounding surface — a flat grey line teaches "grey line" and fails under your test lighting.

Parameter ranges that matter: width **1–5 px** at native resolution, contrast **5–40 grey levels**, branch probability **1–5%**, elastic warp **1.5–6 px**.

### 5.4 Quality gates before you trust synthetic data

| Gate | Test | Pass |
|---|---|---|
| 1. Detectable | PatchCore trained on good images; `mean(score[mask]) / mean(score[~mask]) > 3` | ≥80% of samples |
| 2. Not cartoonish | ResNet-18 classifier real-vs-synthetic defect patches | AUC **0.65–0.85** (>0.9 = fake, <0.55 = not a defect) |
| 3. Human | 12 samples/class contact sheet | "Would I write this on an inspection sheet?" |

### 5.5 Diffusion options (GPU only, and only after Tier 0 is frozen)

| Method | Feasibility in ≤4 h | Note |
|---|---|---|
| **TF-IDG** (ICCV'25) | High — training-free, one-shot | ~22 s/image; best GPU pick |
| **SDXL inpainting + LoRA** | Medium-high — 500–1500 steps on 100 patches | Best control; you own the mask |
| AnomalyDiffusion | Medium — **do not retrain** (~390 GPU-h) | Old CUDA/LDM dependency hell |
| DFMGAN | Low — ~621 GPU-h | Cite, don't run |
| CycleGAN sim→real | Medium | Hallucinates; routinely deletes thin structures |

**The #1 diffusion failure:** a 2000×1500 image with a 40×20 px defect resized to 1024² becomes VAE mush. Crop with 2× context, generate, composite back at `strength=0.85`.

---

## 6. Edge & deployment research

### 6.1 The optimization ladder (measured, cited)

| Config | Model | Latency | Speedup |
|---|---|---|---|
| PyTorch FP32, Core Ultra 7 155H | yolo26n @640 | 33.19 ms | 1× |
| **OpenVINO INT8, same CPU** | yolo26n @640 | **5.79 ms** | **5.7×** |
| TensorRT FP16, RTX 3080 | yolo26n @640 | 0.62 ms | 53× |
| TensorRT INT8, RTX 3080 | yolo26n @640 | 0.52 ms | 64× (mAP 0.37→0.33) |
| TensorRT INT8, Jetson Orin NX 16 GB | yolo26n @640 | 2.30 ms | — |
| PyTorch, Pi 5 + AI HAT+ (Hailo-8, 26 TOPS) | yolo26n @640 | 21.7 ms | 3× vs Pi CPU |

Source: Ultralytics benchmark tables. **[M]**

**Rule:** CPU = ~5× for one afternoon (stop at OpenVINO/ORT INT8). GPU = ~20× in 45 min (TRT FP16); INT8 adds only 2× more for 3 hours of work. Everything past that is diminishing returns for a 48-hour build.

### 6.2 Latency budget — do this arithmetic before optimizing

At **300 ppm** you have **200 ms/part**. A rejector 0.5 m downstream at 1 m/s gives **360 ms of usable budget** after actuation and margin. A 25 ms inference consumes **7%** of it.

The binding constraints are therefore **not** the model:
1. **Sensor → PLC path + jitter** (30–90 ms)
2. **Motion blur** — at 1 m/s and 0.2 mm/px you need **≤200 µs exposure** → a strobe, or no model saves you
3. **p99, not mean** — one 120 ms frame per hundred causes the missed reject

That reframing ("we optimized where it changes the reject decision") is the pitch.

### 6.3 Resolution floor

```
required_px = 3 × FOV_mm / defect_mm
```

A 0.5 mm defect in a 100 mm field of view needs **600 px**. "Just downscale to 224" destroys industrial accuracy. This is the arithmetic behind our 640 input and behind tiling for high-res frames.

### 6.4 PatchCore memory reality

Full bank (100 images, 784 patches, 1024-d) = **321 MB**; 1% coreset = **3.2 MB**. But RAM is not the cost — **kNN FLOPs are**: 126 GFLOP full vs 1.26 GFLOP coreset. **Latency scales with coreset size.** Cache the bank to `.npz`; never rebuild it during a demo.

### 6.5 Hardware comparison (2025–2026)

| Device | Price (approx) | TOPS | Realistic YOLO-seg FPS | Setup hours | Verdict |
|---|---|---|---|---|---|
| Raspberry Pi 5 + Hailo-8L (13 TOPS) | ~₹13.5k | 13 | 15–30 (INT8 224) | 3–4 | Best cheap edge demo |
| Jetson Orin Nano Super | ~$249 | 67 | 60–100 | 1–2 | Best overall edge |
| Jetson Orin NX 16 GB | ~$500+ | 100 | 80–120 | 1–2 | Overkill for a demo |
| Intel N100/N305 + OpenVINO | ~$200 | — | 20–40 | 1 | Easiest x86 path |
| RTX 4060 Laptop | have it | ~120 (FP16) | 200+ | 0 | **Our demo machine** |
| Google Coral | ~$70 | 4 | <10 | 2 | EOL-ish, avoid |
| Qualcomm RB5 / QCS6490 | ~$300 | 15 | 20–40 | 4+ | Toolchain pain |

**[M]** — device estimates marked "derived" are scaled from cited measurements; re-measure before putting them on a slide.

### 6.6 Traps that cost hours

Thermal throttling under sustained load · USB camera frame drops · Python GIL in the capture path · JPEG decode cost (often > inference for small models) · model load time (cold start) · Docker-on-ARM · CUDA/cuDNN version mismatch · **INT8 score collapse when box coords and class scores share an output tensor**.

---

## 7. System architecture research

| Decision | Evidence | Rejected alternative |
|---|---|---|
| **WebSocket binary JPEG** for video (not WebRTC) | WebRTC needs STUN/TURN/certs/SDP — the #1 hackathon stage-killer. HLS adds 3–6 s latency, unusable behind a conveyor | WebRTC (100 ms gain, invisible in a demo), HLS |
| **MJPEG fallback** | Chrome caps 6 concurrent HTTP/1.1 connections per origin — a 4-camera grid silently stalls | Multiplexed WS handles many streams |
| **SQLite WAL + transactional outbox** | 5–20k writes/s on a laptop; outbox gives at-least-once delivery with retries — demonstrable by killing the MES endpoint and watching the queue drain | Redis Streams (correct at scale, extra failure mode at 9:58 on demo day) |
| **Content-addressed evidence files** | Dedupe + immutable evidence + a SHA-256 you can show a judge | Plain filename storage |
| **FastAPI single process** | No tensor IPC, one deploy, free OpenAPI docs | Node + Python sidecar |
| **React + Vite** | Real component composition; HMR for a UI sprint | Streamlit — full rerun per interaction, iframe video hacks, looks like a prototype |
| **Version every decision** | Store `model_version_id` + `threshold_profile_id` on each inspection → replayable audit | Mutable global config |

**The invariant worth defending:** the data plane never reads mutable config at decision time; it resolves a frozen snapshot at inspection start. That is what makes the audit trail real.

---

## 8. Evaluation research — the metrics that survive a plant manager

| # | Metric | Answers | Target |
|---|---|---|---|
| 1 | **FP per 1,000 good parts** | "How often do you stop my line for nothing?" | <20 excellent, <50 credible |
| 2 | Recall per class @ that FPR | "Which defects do you miss?" | >0.85 on the two main classes |
| 3 | Escapes per 1,000 bad parts | "How many bad parts ship?" | <30 |
| 4 | Latency p50/p95/p99 | "Does it fit my takt time?" | p95 < takt |
| 5 | Image AUROC / AUPRO | Comparability with papers | 0.95+ / 0.85+ |

**Context for #1:** AOI machines routinely alarm on 5–70% of *passing* boards, and up to 95% of those alarms are false calls (Siemens data), while high-end 3D inspection targets 20–50 ppm. So "FPR 2%" is normal for AOI and terrible for a pitch deck. **Speak in per-1,000.** **[V]**

### Why the three AUROCs disagree
- **Image AUROC** — easy to inflate; one bright pixel can carry it. With 30 defective test images the 95% CI is ±8–12 points, so 0.92 vs 0.95 is noise. Bootstrap it.
- **Pixel AUROC** — dominated by large defects; at 1:10⁴ imbalance a few thousand easy pixels hide total failure on cracks.
- **PRO / AUPRO** — averages TPR **per connected region**, so a 20 px crack counts as much as a 200k px stain. Right metric for "did you find *the* defect." State the FPR cutoff (0.3 or 0.1).
- **AUPIMO** (2024) is stricter and better correlated with deployment — mention it.

**Part-level metrics** (steal from Real-IAD): a part is defective if **any** view is; part score = max over views. Plants count parts, not images. Bootstrap **parts**, never images — multi-view images of one part are not independent.

---

## 9. Dataset landscape

Full details, links, sizes and licenses in **`DATASETS.md`**. Summary:

| Dataset | Role | Status |
|---|---|---|
| **NEU-DET** (KeenForgeAI corrected mirror) | Primary detector training | ✅ verified |
| **MVTec AD** | Novelty gate benchmark | ✅ verified |
| **Own capture** (phone/USB) | Live demo + gate | ✅ the highest-value data you will collect |
| GC10-DET | Cross-dataset robustness slide | ✅ verified link |
| DeepPCB | PCB pivot if needed | ✅ verified link |
| VisA | Second benchmark | ✅ verified direct tar (1.93 GB) |
| MVTec AD 2 | Credibility slide only | ✅ verified (form-gated) |
| Real-IAD | Scale story | ✅ verified (53 GB — do not download) |
| Kaggle NEU-DET mirror | ❌ **404 — dead, do not chase** | ☠ |
| Kaggle casting-product mirror | ❌ **404 — dead** | ☠ |

---

## 10. Sources

### Core papers [H]
- **MVTec AD** — Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019 / IJCV 2021. `mvtec.com/company/research/datasets/mvtec-ad`
- **PatchCore** — Roth et al., CVPR 2022. arXiv:2106.08265. `github.com/amazon-science/patchcore-inspection`
- **EfficientAD** — Batzner et al., WACV 2024 (oral). arXiv:2303.14535
- **SimpleNet** — Liu et al., CVPR 2023. arXiv:2303.15140
- **DRAEM** — Zavrtanik et al., ICCV 2021. arXiv:2108.00461
- **RD++** — Tien et al., CVPR 2023
- **GLASS** — Chen et al., ECCV 2024. arXiv:2407.09359
- **Dinomaly** — Guo et al., CVPR 2025. arXiv:2405.14325
- **INP-Former** — arXiv:2506.03660 (INP-Former++)
- **AnomalyDINO** — WACV 2025 (oral). arXiv:2405.14529
- **WinCLIP** — Jeong et al., CVPR 2023. **AnomalyCLIP** — Zhou et al., ICLR 2024. arXiv:2310.18961
- **AnomalyGPT** — Gu et al., AAAI 2024. arXiv:2308.15366
- **VisA** — Zou et al., CVPR 2022. **Real-IAD** — Wang et al., CVPR 2024. arXiv:2403.12580
- **MVTec LOCO AD** — Bergmann et al., CVPR 2022 (logical/assembly anomalies)

### 2025–2026 [M]/[V] — verify before quoting
- **MVTec AD 2** — Heckler-Kram et al., IJCV 2026. arXiv:2503.21622. `mvtec.com/company/research/datasets/mvtec-ad-2`
- **VAND 4.0 challenge report** (AD2 + Kaputt 2 leaderboards, efficiency metric) — arXiv:2610.04392
- **SuperADD** (2026 winner) — arXiv:2605.14808
- **RoBiS** (2023 winner recipe: cropping + augmentation + better binarization) — arXiv:2505.21152
- **HD-YOLO** (NEU-DET / HRIPCB comparison table) — arXiv:2412.03969
- **Real-IAD Variety** — arXiv:2511.00540

### Engineering references [M]
- Ultralytics docs — YOLO11/YOLO26 specs, export formats, benchmark tables. `docs.ultralytics.com`
- Anomalib v2.7 — 34 model implementations, OpenVINO export. `github.com/open-edge-platform/anomalib`
- TensorRT / OpenVINO / ONNX Runtime quantization guides
- Ultralytics Hailo (Pi 5 AI HAT+) and Jetson benchmark pages

### Verified infrastructure [O]
- `huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected` — 1,797 images, 1,797 XML + 1,797 YOLO txt, ImageSets, data.yaml, **HTTP 200**, class order confirmed
- `huggingface.co/datasets/foersben/mvtec-ad` — 14,603 files, **HTTP 200**
- `amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar` — 1,929,840,640 bytes, **HTTP 200**
- Kaggle `kaustubhdikshit/neu-surface-defect-database` — **HTTP 404, dead**
- Kaggle `ravirajsinh45/real-life-industrial-dataset-of-casting-product` — **HTTP 404, dead**

---

## 11. Decision log

| Finding | Decision it produced | Where it lives |
|---|---|---|
| AD2 reveals a 35–43 point domain gap | Capture our own normals from the demo rig | `MASTER_PLAN` §7 |
| Thresholding breaks in production | Normalized `z` score + p99 calibration + 3-tier verdict | `MASTER_PLAN` §5 |
| 2026 winners run at 0.12 fps | Do not build SOTA; build the cascade | `TRAINING_SPEC` §0 |
| PatchCore = 0 training + free heatmap | Gate model choice | `TRAINING_SPEC` §2.3 |
| Assembly defects are invisible to pixel AD | Supervised detector + rule layer | §4.4 above |
| Synthetic-only loses | ≥50% real; synthetic as filler only | `TRAINING_SPEC` §3 |
| WebRTC kills demos | MJPEG + WebSocket | `MASTER_PLAN` §3.2 |
| kNN latency ∝ coreset size | 4,000 coreset, cached to `.npz` | `TRAINING_SPEC` §2.3 |
| FP/1,000 is the metric plants buy | Report operational table, not just mAP | `MASTER_PLAN` §12 |
| Unsupervised AD misses assembly logic | Honest limitation + rule checks | §4.4 above |
