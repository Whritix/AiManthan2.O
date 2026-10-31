# TRAINING SPECIFICATION — YOLO11 Defect Detector
## Frozen architecture · RTX 4060 8 GB · 32 GB RAM · 48-hour budget

> **Superseded as of 2026-10-29.** Training is complete. The promoted YOLO11s retrain scored 0.753 mAP50 at 640 with `rect=False`; the previous model is preserved under `artifacts/models/legacy/`. Current metrics and protocol live in `README.md` and `OPTIMIZATION_CONTEXT.md`. This specification is retained as historical design context.

> **Status:** FROZEN. This document supersedes the CPU-first assumptions in `MASTER_PLAN.md` §4.
> The GPU changes three things: model size (YOLO11n → **YOLO11s**), training budget (40 min → **3 h of real training**), and deployment (ONNX CPU → **TensorRT FP16 + ONNX INT8 fallback**).

---

# 0. Model Freeze Declaration

**The following is locked. No substitutions, no "what about YOLO26 / RT-DETR / segmentation".**

| Frozen decision | Value |
|---|---|
| Architecture family | **YOLO11** (Ultralytics) |
| Variant | **YOLO11s** final; YOLO11n only for the H8–H14 smoke test |
| Task | **detect** (bounding boxes) — *not* segment |
| Input resolution | **640 × 640** |
| Classes | **5 trained + 1 held out** |
| Framework | PyTorch 2.x + Ultralytics 8.3.x |
| Export | ONNX opset 17 → TensorRT FP16 (GPU) / ONNX INT8 (CPU) |
| Dataset | **NEU-DET** (primary), MVTec AD (novelty gate) |

**Why YOLO11s and not n/m:**
- **n** (2.6 M params, 6.5 GFLOPs) leaves accuracy on the table and has no headroom for 5 similar-looking surface textures.
- **s** (9.4 M, 21.5 GFLOPs) is the sweet spot: ~2.5× the capacity of n, still trains in ~90 min on a 4060, still exports to <25 MB FP16, still runs at 30+ FPS on CPU with INT8.
- **m** (20.1 M, 68 GFLOPs) is ~3× slower to train and infer for maybe +2 mAP. Not worth the 48-hour risk. It is your **upgrade lever only if targets are missed and time remains**.

**No segmentation.** YOLO11s-seg costs +40% training time, needs polygon labels you do not have, and the problem statement accepts detection. Boxes + PatchCore heatmap already give the "where."

---

# 1. Hardware & Environment

| Item | Value | Implication |
|---|---|---|
| GPU | RTX 4060 Laptop, **8 GB VRAM** | YOLO11s @640 batch 16 fits. Batch 32 does not. |
| RAM | 32 GB | `cache='ram'` is viable — 1,800 × 200×200×3 ≈ 216 MB |
| OS | Windows (win32) | `workers=4` (not 8) to avoid the Windows dataloader deadlock |
| Python | 3.11 or 3.12 | 3.13 has wheel gaps for torch/tensorrt |
| CUDA | 12.4 build of torch | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |

**Install (verify in this order — 15 min):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.3.0 onnx onnxruntime onnxruntime-gpu opencv-python
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.x.x+cu124  True  NVIDIA GeForce RTX 4060 Laptop GPU
```

**TensorRT is optional and last** (hour 26+). It needs matching CUDA/cuDNN and is a known 2-hour time sink on Windows. Do not touch it before the ONNX path works.

**VRAM guardrails:**
- OOM at batch 16 → drop to 8 and set `nbs=32` (gradient accumulation in Ultralytics via nominal batch).
- OOM persists → `imgsz=512`. Accuracy cost ≈ 1–2 mAP; latency win ≈ 35%.
- Never run Chrome + training + the demo app at once on 8 GB.

---

# 2. Model Architecture

## 2.1 YOLO11s topology

```
Input 640×640×3
│
├─ Backbone (CSPDarknet-style, stride 2 each stage)
│   P1  320×320× 32   Conv k3s2
│   P2  160×160× 64   Conv k3s2 + C3k2 ×2
│   P3   80×80 ×128   Conv k3s2 + C3k2 ×2      ◄── detection scale 1 (small defects)
│   P4   40×40 ×256   Conv k3s2 + C3k2 ×2      ◄── detection scale 2
│   P5   20×20 ×512   Conv k3s2 + C3k2 ×2 + SPPF + C2PSA
│
├─ Neck: PAN-FPN (top-down FPN + bottom-up PAN)
│   P5 → upsample → concat(P4) → C3k2 → P4'
│   P4' → upsample → concat(P3) → C3k2 → P3'
│   P3' → downsample → concat(P4') → C3k2
│   P4'' → downsample → concat(P5)  → C3k2
│
└─ Head: decoupled, anchor-free, DFL regression
    Detect(P3', P4'', P5)  →  (cls 5, box 4×16 dfl bins) per anchor
    Loss = 7.5·CIoU + 0.5·BCE(cls) + 1.5·DFL
    NMS at inference (IoU 0.7 / conf 0.25)
```

**Block inventory:** `Conv` = Conv2d + BatchNorm + SiLU. `C3k2` = CSP block with two bottleneck stacks (the YOLO11 replacement for C2f). `SPPF` = spatial pyramid pooling (k=5). `C2PSA` = cross-stage partial **spatial attention** at the end of the backbone — this is YOLO11's main architectural addition over YOLOv8 and it measurably helps on low-contrast texture.

## 2.2 Size comparison (COCO, Ultralytics reference)

| Variant | Layers | Params | GFLOPs @640 | COCO mAP50-95 | Verdict |
|---|---|---|---|---|---|
| YOLO11n | 181 | 2.6 M | 6.5 | 39.5 | Smoke test only |
| **YOLO11s** | **181** | **9.4 M** | **21.5** | **47.0** | **✅ Frozen choice** |
| YOLO11m | 207 | 20.1 M | 68.0 | 51.5 | Emergency upgrade only |
| YOLO11l | 238 | 25.3 M | 86.9 | 53.4 | No — 8 GB VRAM marginal |

**Effective receptive field vs defect size — do the arithmetic before training:**
At 640×640, the P3 stride-8 head sees 80×80 cells → each cell = 8 px of input. NEU-DET defects are 200×200 crops upscaled to 640, so a 10-px defect in the original becomes 32 px → **4 cells wide**, comfortably detectable. If you had trained at 224, the same defect would be 11 px → 1.4 cells → the model would learn to miss it. *This is the single strongest argument for 640 and the answer to "why not just downscale?"*

## 2.3 The second model: PatchCore-lite (novelty gate)

Not a neural network in the training sense — no gradient steps.

| Component | Spec |
|---|---|
| Backbone | ResNet-18 (ImageNet), features at `layer2` (1/8) + `layer3` (1/16) |
| Patch grid | 40×40 + 20×20 = 2,000 patches per 320×320 tile |
| Projection | 1×1 conv → 256-d, L2-normalized |
| Memory bank | Greedy coreset, **4,000 vectors** (≈4 MB float32) from 40–80 good images |
| Score | `d_i = 1 − max_j cos(e_i, m_j)` via one GEMM |
| Image score | mean of top 0.1% patch distances |
| Heatmap | `d_i` reshaped → bilinear upsample (free) |
| "Training" | 20–60 s on GPU for 200 images (forward passes only) |
| Retrain on stage | <60 s including feature extraction |

**Why not EfficientAD here:** it is faster (2 ms) and slightly better on MVTec AD 2, but it needs per-category training and its student-teacher distillation is 15–30 min per part on CPU. PatchCore is **zero-training**, which is what enables the "new part in 5 minutes" and "retrain on stage" narratives. Use EfficientAD only if you have spare time at H30 and want a speed slide.

---

# 3. Datasets

## 3.1 Primary: NEU-DET (NEU Surface Defect Database)

| Property | Value |
|---|---|
| Content | Hot-rolled steel strip surface defects |
| Images | **1,800** grayscale, **200×200 px** |
| Classes | **6 × 300 images**: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches` |
| Labels | Pascal VOC XML bounding boxes |
| Size on disk | ~1 GB |
| License | Research use; cite Song & Yan (2013) |
| Mirror A | `kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database` |
| Mirror B | `github.com/Charmve/Surface-Defect-Detection` (index of mirrors) |
| Why this one | 6 interpretable defect types that map 1:1 to the problem statement (crack, scratch, surface irregularity, inclusion); small enough to train 20 times in a weekend; box labels already exist |

**Hold out `rolled-in_scale` entirely** (300 images). It never enters training, validation, or the detector's test set. It becomes the "unseen defect" demo set.

### Split definition (write this to `data/neu-det/splits.json`, commit it, never regenerate)

| Split | Classes | Images | Purpose |
|---|---|---|---|
| **Train** | 5 (no rolled-in_scale) | **1,050** | 70% of the 1,500 remaining, stratified by class → 210/class |
| **Val** | 5 | **225** | 15% → 45/class. Threshold calibration + early stopping |
| **Test** | 5 | **225** | 15% → 45/class. **Touched once, at H32. Never during development** |
| **Held-out** | `rolled-in_scale` | **300** | The novelty demo. Detector must produce ≈0 detections here |

Stratify **by class**, and check for near-duplicate images across splits (the same coil photographed twice is leakage — compute perceptual hash and drop cross-split duplicates).

**Optional second run for the README (20 min):** full 6-class model, 6-class split 70/15/15 → gives you a defensible 6-class mAP for the benchmark table. Do this only if the 5-class model is already at target.

## 3.2 Secondary (credibility, optional): GC10-DET

| Property | Value |
|---|---|
| Content | 10 metallic surface defect classes (welding, punching, crescent gap, water spot, oil spot, silk spot, inclusion, rolled pit, crease, waist folding) |
| Images | ~3,570 with box labels |
| Use | Cross-dataset generalization slide: "trained on NEU, evaluated on GC10 → mAP drops X, proving domain shift is the real problem, not our model" |
| When | Only at H26+ if targets are met. It is a slide, not a deliverable |

## 3.3 Novelty gate: MVTec AD + your own parts

| Source | Categories | Use |
|---|---|---|
| **Your own rig** (primary) | 1 physical part, 40–60 good frames + 12–16 per defect | The live demo. **Domain match is the whole point.** |
| **MVTec AD** (backup / benchmark) | `metal_nut`, `screw`, `transistor`, `bottle` | Credible AUROC number + fallback if your capture is bad |
| Download | `huggingface.co/datasets/foersben/mvtec-ad` (4.9 GB, fast) or mvtec.com form | CC BY-NC-SA 4.0 — research only, say so |
| Layout | `datasets/MVTecAD/<cat>/train/good/*.png`, `test/<defect>/*.png`, `ground_truth/<defect>/*_mask.png` | The #1 setup gotcha |

**Own-capture split:** 70/15/15 **by session and by physical part**, not by image. 40 good frames → 28 bank / 6 calibration / 6 test.

## 3.4 Directory layout

```
data/
├── neu-det/
│   ├── images/{train,val,test}/*.jpg
│   ├── labels/{train,val,test}/*.txt        # YOLO format: cls cx cy w h (normalized)
│   ├── heldout/                              # 300 rolled-in_scale, detector must miss
│   ├── splits.json                           # THE source of truth, committed
│   └── neu-det.yaml                          # Ultralytics dataset config
├── own/
│   ├── good/{bank,calib,test}/*.jpg
│   └── defect/{crack,scratch,dent,missing_part}/*.jpg
└── mvtec/                                    # optional
```

```yaml
# data/neu-det/neu-det.yaml
path: E:/Indore/data/neu-det
train: images/train
val: images/val
test: images/test
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: scratches
# rolled-in_scale is INTENTIONALLY ABSENT — held out for novelty demo
```

---

# 4. Label & Data QA (30 minutes, saves 6 hours)

Run this **before** the first real training run. 80% of "the model is bad" is one of these five bugs.

| # | Check | Pass criterion | If it fails |
|---|---|---|---|
| 1 | Every image has a label file; every label file has an image | 100% match | Delete orphans |
| 2 | Class IDs in labels ⊂ {0..4} | Zero occurrences of 5 | Rewrite mapping |
| 3 | Box coordinates in [0,1] and `w,h > 0.005` | 100% | Clamp / drop degenerate boxes |
| 4 | Visualize 20 random labels with boxes drawn | Boxes actually on the defect | Fix the VOC→YOLO converter (x/y or w/h swap is the classic bug) |
| 5 | No duplicate images across train/val/test | Perceptual-hash Jaccard < 0.95 | Re-split |

Also record the **box-area distribution**. NEU-DET boxes are large (defect fills much of the 200×200 crop), so a model that only detects large regions will look great on NEU and fail on your 1 mm scratch. That gap is exactly why the PatchCore gate exists.

---

# 5. Training Pipeline — 4 Stages

## Stage 0 — Smoke test (10 min, YOLO11n)

Prove the pipeline before spending GPU hours.

```bash
yolo detect train model=yolo11n.pt data=data/neu-det/neu-det.yaml \
  epochs=10 imgsz=640 batch=32 device=0 workers=4 cache=ram \
  project=runs/smoke name=n_smoke
```

**Gate:** mAP50 > 0.15 after 10 epochs and both losses trending down. If mAP is 0.00, **stop** — it is a data bug, not a model bug. Go back to §4.

## Stage 1 — Baseline (35 min, YOLO11n, full schedule)

```bash
yolo detect train model=yolo11n.pt data=data/neu-det/neu-det.yaml \
  epochs=150 imgsz=640 batch=32 device=0 workers=4 cache=ram \
  optimizer=SGD lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3 cos_lr=True patience=40 close_mosaic=15 \
  project=runs/baseline name=n_150 seed=0
```

**Gate H14:** mAP50 ≥ 0.60, ≥15 FPS end-to-end. This model is your safety net — keep the weights. If Stage 2 fails, you still demo.

## Stage 2 — Final (90–120 min, YOLO11s, full schedule)

```bash
yolo detect train model=yolo11s.pt data=data/neu-det/neu-det.yaml \
  epochs=200 imgsz=640 batch=16 device=0 workers=4 cache=ram \
  optimizer=SGD lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3 cos_lr=True patience=50 close_mosaic=20 \
  hsv_h=0.015 hsv_s=0.4 hsv_v=0.5 \
  degrees=5 translate=0.10 scale=0.30 shear=2.0 perspective=0.0005 \
  fliplr=0.5 flipud=0.0 mixup=0.0 copy_paste=0.0 erasing=0.0 \
  auto_augment=None \
  project=runs/final name=s_200 seed=0
```

**Gate H20:** mAP50 ≥ 0.75, per-class recall ≥ 0.70, ONNX export working.

## Stage 3 — Hard-negative + rare-class pass (30 min, optional)

Only if FP rate is too high or one class lags.

1. Mine false positives from the **val** set and from the good-part captures.
2. Add them as background images (empty label files) — cap at **10% of the training set** (≈105 images).
3. Oversample the weakest class by 2× (duplicate with different augmentation seeds).
4. Retrain from the Stage-2 weights with `epochs=40 lr0=0.003` (fine-tune, not from scratch).

---

# 6. Hyperparameters — Full Reference

## 6.1 Core

| Parameter | Value | Rationale |
|---|---|---|
| `epochs` | 200 (Stage 2) | NEU-DET is small; 100 underfits slightly, 300+ overfits |
| `imgsz` | 640 | See §2.2 — the small-defect argument |
| `batch` | 16 (s) / 32 (n) | 8 GB VRAM ceiling |
| `nbs` (nominal batch) | 32 | Accumulate if batch is reduced |
| `optimizer` | SGD | Battle-tested default; converges reliably on small datasets |
| `lr0` | 0.01 | With batch 16; scale to 0.005 if loss spikes |
| `lrf` | 0.01 | Final LR = lr0 × lrf (cosine decay to 1e-4) |
| `momentum` | 0.937 | Ultralytics default |
| `weight_decay` | 0.0005 | Default |
| `warmup_epochs` | 3 | Prevents early divergence on tiny datasets |
| `cos_lr` | True | Smoother final epochs than linear |
| `patience` | 50 | Early stop — saves an hour if it plateaus |
| `close_mosaic` | 20 | **Critical.** Mosaic off for the last 20 epochs so the model sees real compositions |
| `amp` | True | ~1.8× faster, ~40% less VRAM |
| `cache` | `ram` | Dataset is 216 MB; eliminates disk I/O entirely |
| `workers` | 4 | Windows; 8 causes stalls |
| `seed` | 0, then 1, 2 | Report mean ± std if time allows |
| `deterministic` | True | Reproducible numbers for the README |

## 6.2 Augmentation

| Parameter | Value | Why |
|---|---|---|
| `hsv_h` | 0.015 | Tiny; hue shifts turn blue oxidation into a red stain |
| `hsv_s` | 0.4 | Lighting variations on the line |
| `hsv_v` | 0.5 | Exposure bracketing — the dominant real-world failure mode |
| `degrees` | 5 | NEU has no strict orientation. Do not use 90/180 |
| `translate` | 0.10 | Handheld/part-placement jitter |
| `scale` | 0.30 | Aggressive scale hurts tiny defects (can shrink one below a stride cell) |
| `shear` | 2.0 | Mild perspective realism |
| `perspective` | 0.0005 | Camera angle |
| `fliplr` | 0.5 | Safe for steel texture |
| `flipud` | 0.0 | Rolling direction is a real cue; don't destroy it |
| `mosaic` | 1.0 default, off at epoch 180 | Composite context, but it downscales each tile 2× — hence close_mosaic |
| `mixup` | 0.0 | Blends defect texture into mush; meaningless labels |
| `copy_paste` | 0.0 | Enable only as a rare-class fix (Stage 3) |
| `erasing` | 0.0 | **Erases the defect** and teaches "hole = normal" |
| `auto_augment` | `None` | RandAugment applies color ops that destroy inspection cues |

## 6.3 Loss weights (defaults — change only with evidence)

| Term | Weight | Meaning |
|---|---|---|
| `box` | 7.5 | CIoU localization |
| `cls` | 0.5 | BCE classification |
| `dfl` | 1.5 | Distribution Focal Loss for box edges |

Do **not** tune these in the first 6 hours. If one class is weak, fix the data, not the loss.

## 6.4 Inference thresholds

| Parameter | Value | Note |
|---|---|---|
| `conf` | 0.25 | Detection confidence |
| `iou` (NMS) | 0.7 | High — defects can be adjacent |
| `max_det` | 20 | A part rarely has more than a few defects |
| `agnostic_nms` | False | Keep class separation |

---

# 7. Freeze Schedule (transfer learning control)

Two separate meanings — both are specified, so there is no ambiguity later.

## 7.1 "The model is frozen" (decision freeze)

YOLO11s, detect, 640, 5 classes. Changing any of these requires deleting a Tier-0 feature from `SCOPE.md`. No YOLO26, no RT-DETR, no segmentation, no SAM, no custom backbone.

## 7.2 "Freeze layers" (training technique)

Ultralytics `freeze=N` freezes the **first N layers** (0-indexed from the input). For YOLO11s the backbone occupies roughly layers 0–10.

| Scenario | `freeze` | Rationale | Expected effect |
|---|---|---|---|
| **Default (recommended)** | `None` | COCO→steel is a large domain shift; the backbone must adapt | Best accuracy |
| Overfitting (train mAP 0.95, val 0.70) | `10` for first 20 epochs, then unfreeze | Protects pretrained texture filters while the head warms up | +1–3 val mAP, less variance |
| Tiny dataset (<500 images) | `10` | Regularization | Slower convergence, higher floor |
| Stage-3 fine-tune | `0` (all trainable), `lr0=0.003` | Small LR already regularizes | Avoids catastrophic forgetting |

**Do not freeze the backbone for the whole run.** On NEU-DET that caps you around 0.65 mAP — you would be leaving 10+ points on the table because ImageNet features do not know what a rolled-in scale looks like.

**Verify what you froze** (2 minutes, saves a confusing afternoon):

```python
from ultralytics import YOLO
m = YOLO("yolo11s.pt")
for i, (name, p) in enumerate(m.model.named_parameters()):
    if i < 30: print(i, name, p.requires_grad, p.shape)
```

---

# 8. Export & Quantization

```bash
# 1. ONNX FP32 — the portable artifact. Do this first, always.
yolo export model=runs/final/s_200/weights/best.pt format=onnx opset=17 simplify=True imgsz=640

# 2. TensorRT FP16 — the GPU demo path (Jetson-style deployment story)
yolo export model=runs/final/s_200/weights/best.pt format=engine half=True imgsz=640 device=0

# 3. ONNX INT8 — the CPU fallback (what runs if the GPU is busy/absent)
yolo export model=runs/final/s_200/weights/best.pt format=onnx int8=True \
  data=data/neu-det/neu-det.yaml imgsz=640
```

**Known traps:**

| Trap | Symptom | Fix |
|---|---|---|
| TRT build hangs on Windows | 20+ min, no output | Skip TRT; ship ONNX. It is a slide, not a dependency |
| INT8 scores collapse to 0 | All confidences ≈ 0 | The output tensor mixes box coords (px) and class scores (0–1) under one scale. Exclude non-Conv/Gemm nodes from quantization |
| INT8 accuracy drops 5+ mAP | — | Calibrate on 200 **val** images, not 20; re-fit thresholds after quantization |
| ONNX dynamic axes | Slow / shape errors | Export static shape `1×3×640×640` for the demo |
| Reusing FP32 confidence thresholds after INT8 | Every part fails | **Recalibrate conf and the anomaly threshold on quantized output** |

---

# 9. Evaluation Protocol

## 9.1 Rules

1. **Test set is touched exactly twice:** once at H32 for the frozen results table, once before the pitch as a sanity check.
2. Threshold calibration happens on **val only**, never test.
3. Report **mean ± std over 3 seeds** if you trained 3; otherwise state "single seed" honestly.
4. Report a **bootstrap 95% CI** on mAP (1,000 resamples of images). With 225 test images the CI is roughly ±3–4 points — say so.
5. Pixel/box metrics need **±2 px tolerance**; boundary ambiguity on a scratch is real.

## 9.2 Metrics and targets

### Detector — hard targets

| Metric | Minimum viable | **Target** | Stretch | How measured |
|---|---|---|---|---|
| **mAP50** (5 classes) | 0.65 | **0.80** | 0.85 | `val` on test split |
| mAP50-95 | 0.38 | **0.48** | 0.55 | same |
| Precision @ conf 0.25 | 0.70 | **0.82** | 0.88 | same |
| Recall @ conf 0.25 | 0.68 | **0.80** | 0.86 | same |
| **Per-class recall (worst class)** | 0.55 | **0.70** | 0.80 | confusion matrix |
| Held-out class detections | — | **≤ 5% of images** | ≤ 2% | run detector on 300 held-out |
| Model size | ≤ 40 MB | **≤ 25 MB FP16** | ≤ 12 MB INT8 | file size |
| GPU latency (TRT FP16) | ≤ 15 ms | **≤ 8 ms** | ≤ 5 ms | p95, batch 1, 640 |
| CPU latency (ONNX INT8) | ≤ 50 ms | **≤ 30 ms** | ≤ 20 ms | p95, batch 1, 640 |

**Reference point:** published YOLOv11n on NEU-DET reports **80.1 mAP50** (YOLOv8n: 77.1). YOLO11s should land 0.80–0.85. If you measure 0.95+, suspect leakage or a broken split — that number is not credible to a judge who knows the dataset.

### Novelty gate — hard targets

| Metric | Minimum | **Target** | Note |
|---|---|---|---|
| Image AUROC (MVTec category) | 0.90 | **0.96** | PatchCore baseline is 0.99 on MVTec AD |
| Image AUROC (your own parts) | 0.88 | **0.95** | This is the one that matters |
| Pixel AUROC | 0.85 | **0.92** | Heatmap quality |
| Recall @ 2% FPR (held-out class) | 0.70 | **0.85** | The demo number |
| Gate latency (CPU, 320²) | ≤ 40 ms | **≤ 25 ms** | Includes coreset kNN |

### System — hard targets

| Metric | Minimum | **Target** |
|---|---|---|
| End-to-end p95 latency (GPU) | ≤ 60 ms | **≤ 25 ms** |
| End-to-end p95 latency (CPU fallback) | ≤ 120 ms | **≤ 80 ms** |
| Sustained FPS | 15 | **25+ GPU / 15+ CPU** |
| FP per 1,000 good parts | ≤ 50 | **≤ 20** |
| Unknown-routing rate | ≤ 15% | **2–8%** |
| Operator reject rate of auto-detections | ≤ 20% | **≤ 10%** |

---

# 10. If the Model Misses Target — Diagnosis Tree

**Do not change hyperparameters until you know which of these three regimes you are in.** Pull the `results.csv` from the run and look at the curves.

```
                    ┌─ train mAP low AND val mAP low, both flat
                    │     → UNDERFITTING / PIPELINE BUG
                    │
results.csv  ───────┼─ train mAP high (0.90+) AND val mAP low (<0.70)
                    │     → OVERFITTING
                    │
                    └─ both oscillating, loss spikes
                          → OPTIMIZATION INSTABILITY
```

## 10.1 Regime A — Underfitting / pipeline bug

| Check first (in this order) | How | If wrong |
|---|---|---|
| Label paths & class mapping | Visualize 20 predictions vs ground truth | Rewrite converter |
| BGR vs RGB mismatch at inference | Compare `val` mAP vs your own inference mAP | Fix channel order |
| Letterbox / normalization mismatch | Same comparison | Use `ultralytics` predictor, don't hand-roll |
| Frozen backbone | `requires_grad` printout | Unfreeze |
| LR too low | Loss decreases <20% over 20 epochs | lr0 0.01 → 0.02, or switch to AdamW 0.001 |
| Too few epochs | Val mAP still rising at the end | 200 → 300, patience 70 |

**Expected gain from fixing a real pipeline bug: +10 to +40 mAP.** This is where 80% of "my model is broken" lives. Check it before anything else.

## 10.2 Regime B — Overfitting

| Lever | Setting | Expected gain |
|---|---|---|
| Freeze backbone early | `freeze=10` for 20 epochs, then unfreeze | +1–3 |
| Stronger augmentation | `scale=0.5`, `translate=0.15`, `degrees=10`, enable `copy_paste=0.1` | +1–3 |
| More data | Add GC10-DET or MVTec textures as extra negatives | +2–5 |
| Lower LR / longer warmup | `lr0=0.005`, `warmup=5` | +0.5–2 |
| Early stop tighter | `patience=25` | Protects the result, no accuracy gain |
| Weight decay up | `0.0005 → 0.001` | +0.5–1 |

## 10.3 Regime C — Instability

| Symptom | Fix |
|---|---|
| Loss spikes every few epochs | `lr0 0.01 → 0.005`, `warmup_epochs 5` |
| NaN loss | AMP overflow → `amp=False` for 5 epochs to confirm, then reduce lr |
| Val mAP oscillates ±0.05 | Batch too small → `nbs=32` accumulation |
| Diverges at epoch 0 | Bad labels (box outside image, w/h=0). Run §4 again |

## 10.4 Accuracy ladder — ranked by gain per hour

Apply **in this order**. Stop when you hit target or time runs out.

| # | Action | Time | Expected mAP50 gain | Risk |
|---|---|---|---|---|
| 1 | Fix a data/label/eval bug | 30 min | **+10–40** | None |
| 2 | `imgsz 640 → 800` (defects get more cells) | 1.5 h train | +1–3 | VRAM (batch 8) |
| 3 | Train 3 seeds, ensemble at inference | 3 h | +1–2 | Latency 3× |
| 4 | 200 → 300 epochs, patience 70 | +1 h | +1–2 | Overfit |
| 5 | Copy-paste rare-class defects | 1 h | +1–3 on weak class | Realism |
| 6 | Hard-negative mining (10% background images) | 1 h | **−20–40% FP** (not mAP) | Slight recall dip |
| 7 | TTA at inference (`augment=True`) | 5 min | +1–2 | 2–3× slower — demo only |
| 8 | YOLO11s → YOLO11m | 3 h train | +2–3 | 8 GB VRAM marginal, slower |
| 9 | Distill YOLO11m → YOLO11s | 4 h | +1–2 | Real risk of not finishing |
| 10 | SAHI tiling at inference | 2 h | +2–5 on high-res | 4–9× compute |

**Stop rule:** if after items 1–3 you are still >5 points below target, **do not chase it**. Renegotiate the claim on the slide (report per-class recall at a fixed FPR instead of a single mAP) and spend the time on the demo. Judges reward honesty about limits far more than an extra 2 points.

## 10.5 If the *gate* misses target

| Symptom | Cause | Fix |
|---|---|---|
| AUROC < 0.85 | Domain gap between bank images and test | Re-capture bank **on the demo rig**, same lighting, same distance |
| AUROC high but demo fails | Threshold wrong | Recalibrate at the 99th percentile of **good val** scores, not a guessed 0.5 |
| Too many false alarms | Coreset too small / too generic | Coreset 4,000 → 8,000; add hard negatives (dust, reflections) to the bank as "good" |
| Misses small defects | Resolution | Tile at 320 instead of resizing the full frame; score = max over tiles |
| Heatmap is a blob | Single-scale features | Add `layer2` (1/8) features; apply 3×3 morphological open before the mask |
| Score drifts during demo | Auto-exposure | Lock camera exposure and white balance; add per-frame intensity normalization |

---

# 11. Time Budget on the 4060

| Task | Wall clock | Notes |
|---|---|---|
| Env + GPU verify | 15 min | Do not skip the `torch.cuda.is_available()` check |
| Dataset download + convert + QA | 45 min | The converter is where hours die |
| Stage 0 smoke (10 epochs, n) | 5 min | Gate: mAP > 0.15 |
| Stage 1 baseline (150 epochs, n) | 35 min | Your safety net |
| Stage 2 final (200 epochs, s) | 90–120 min | The deliverable |
| Stage 3 fine-tune (optional) | 30 min | Only if needed |
| ONNX + INT8 export | 20 min | |
| TensorRT build (optional) | 20–120 min | **High variance on Windows** |
| Gate: PatchCore bank build (200 imgs) | 2 min | Forward passes only |
| Gate: threshold calibration | 10 min | |
| Evaluation + tables + charts | 40 min | Generate from a script, never hand-type |
| **Total critical path** | **~4.5 h** | Leaves ~43 h for the product |

**That is the headline: the ML is 4.5 hours of a 48-hour event.** The remaining time belongs to the demo, the routing, the UI, the rehearsal, and sleep.

---

# 12. Reproducibility Checklist

- [ ] `splits.json` committed; test set hash-locked
- [ ] Held-out class asserted absent from every training run
- [ ] `seed` fixed; `deterministic=True`
- [ ] `runs/**/args.yaml` committed for the final run
- [ ] `results.csv` and `results.png` committed
- [ ] Confusion matrix + PR curve exported as PNGs for the deck
- [ ] ONNX model SHA-256 recorded; same hash in the app and the README
- [ ] Every number on the results slide stamped with hardware, engine, resolution, and n

---

# 13. One-Screen Summary

```
MODEL      YOLO11s · detect · 640×640 · 9.4 M params · 21.5 GFLOPs
CLASSES    5 trained (crazing, inclusion, patches, pitted_surface, scratches)
           + 1 held out (rolled_in_scale) for the unseen-defect demo
DATA       1,050 train / 225 val / 225 test / 300 held-out
TRAIN      SGD lr0=0.01 · cosine · 200 epochs · batch 16 · AMP · cache=ram
           mosaic off for final 20 epochs · auto_augment=None · erasing=0
FREEZE     full fine-tune (freeze=None); freeze=10 only if overfitting
EXPORT     ONNX FP32 → TensorRT FP16 (GPU) → ONNX INT8 (CPU)
TARGETS    mAP50 ≥ 0.80 · worst-class recall ≥ 0.70 · held-out detection ≤ 5%
           TRT p95 ≤ 8 ms · ONNX INT8 p95 ≤ 30 ms · model ≤ 25 MB
FALLBACK   YOLO11n baseline (keep the weights) · ONNX-only if TRT fights
IF SHORT   data bug → imgsz 800 → 3-seed ensemble → 300 epochs → m
STOP RULE  >5 points short after the first 3 levers → renegotiate the claim,
           spend the time on the demo
```
