# Data Pipeline, Synthetic Defects & Robustness Playbook
### 36–48h industrial defect-detection hackathon · laptop + phone camera (+ maybe 1 GPU)

---

## 0. TL;DR — the decision tree

| Situation | What you do | Time |
|---|---|---|
| **No GPU** (or <8 GB VRAM) | OpenCV/PIL procedural defect generator (Section 2) + DRAEM-style training + PatchCore baseline. **Do not touch diffusion.** | 3 h generator, 1 h integration |
| **1 GPU ≥12 GB** | Same as above **plus one** of: (a) TF-IDG one-shot (no training, ~22 s/img), (b) SDXL/Flux inpainting + LoRA on 100 defect patches, (c) AnomalyDiffusion **released checkpoints/data** (never retrain). | +1–3 h |
| **Only ~20 good parts, no defects at all** | PatchCore / EfficientAD unsupervised on the good parts. This is your *credible baseline* (99.1% img-AUROC on MVTec). Synthetic defects are then used only for **localization** and **defect-type classification**, not detection. | 30 min |
| **Have 50–150 real defects** | 75/25 or 100/100 real:synthetic supervised segmentation. This is the sweet spot; evidence below. | 4–6 h |

**Three rules that decide whether you win or lose:**

1. **Synthetic-only training loses.** Every 2024–2026 paper that measured it found the same thing: ScratchSim (2026) synthetic-only mAP50 0.451 vs 0.610 real-only; SynSur (2026) synthetic-only AP 0.393 vs 0.652 real-only. Synthetic data is a *regularizer and a rare-class filler*, never a replacement. Best measured regime: **fine-tune from synthetic weights → real data**, then **mix 50–100% real + 50–100% synthetic**.
2. **The realism of the defect matters far less than you think for anomaly detection, and far more than you think for classification.** DRAEM explicitly says its simulated anomalies need only be *just-out-of-distribution*, not realistic. If your head is "find the defect", ugly synthetic blobs work. If your head is "which of 6 defect classes", the morphology must match or you teach the wrong prior.
3. **Your test set will come from a phone, so your train set must already contain phone-camera physics**: white-balance drift, rolling-shutter-free but handheld blur, JPEG, specular highlights, and a background that is not the dataset's background. Domain randomization (Section 4) buys more AUROC than a better generator.

---

## 1. Framing: which head are you training?

Decide in the first 20 minutes; it changes the entire data plan.

**Head A — Unsupervised anomaly detection/localization (no defect labels needed).**
PatchCore (memory bank of nominal patches), EfficientAD (student–teacher, **2 ms/img, 600 img/s** on an A6000), PaDiM. Trained on *good parts only*. Synthetic defects here serve as **validation probes**: if your generator's defect does not raise the anomaly score, either the defect is invisible or the model is blind. This is the cheapest credible demo and the one that survives a bad dataset.

**Head B — Supervised defect segmentation/classification.**
U-Net / SegFormer-b0 / YOLO-seg on (image, mask, class). This is what the problem statement ("detecting/classifying cracks, scratches, surface irregularities, assembly defects") actually asks for. It needs *class-consistent* synthetic morphology.

**Recommended hackathon architecture: A + B, sharing one preprocessing path.**

```
image → letterbox/tile → [PatchCore or EfficientAD] → anomaly map + image score
                      → [SegFormer-b0 / U-Net]      → defect mask + class logits
                      → fusion: class = argmax(class_logits | anomaly_score > τ)
```
Report both. Head A gives you "we detect unseen defect types". Head B gives you "we classify the 4 named classes". Judges love that sentence, and it is honest.

**Dataset picks (fast, small, defect-relevant):**

| Dataset | Why | Size | Note |
|---|---|---|---|
| MVTec AD | the benchmark; 15 categories incl. carpet/leather/tile/wood/transistor/screw | 5,354 imgs | CC BY-NC-SA, research only |
| **MVTec AD 2** (2025) | 8 hard scenarios, **test lighting deliberately different from train**; SOTA <60% AU-PRO | 8,004 imgs | perfect for your "phone camera" robustness story |
| **Magnetic Tile Defect** | cracks + blowholes + fray, *multiple illumination conditions per tile* | 1,344 imgs, pixel masks | closest to "crack on a metal/ceramic part" |
| NEU-DET | 6 steel surface classes (crazing, inclusion, patches, pitted, rolled-in, scratches) | 1,800 @200² | boxes only, tiny images |
| KolektorSDD2 | real production-line surface cracks | 3,335 | binary segmentation, very imbalanced |
| VisA / Real-IAD | multi-object, multi-view, harder | 10k / 150k | Real-IAD has sample-level (multi-view) metric — steal that idea |
| DAGM / BTAD | classic weakly-supervised | small | good for ablations |

Take **one** for the demo and **one** for the robustness slide. Downloading two is 20 minutes; downloading five is your whole evening.

---

## 2. Synthetic defects without a diffusion model (< 3 hours)

### 2.0 The generator's contract

Every synthetic sample must be a triple `(image, mask, class)` where the mask is *exactly* what changed, plus a `params.json` row recording the seed and all sampled parameters. Non-negotiables:

- **Seed everything** (`np.random.default_rng(seed)`, one seed per sample, stored).
- **Place defects on the part, not the background.** Compute a part mask once per background image (Otsu on saturation + largest connected component, or SAM if available) and reject placements whose centroid falls outside it, or whose bbox touches the border (truncated defects teach truncation-invariance you don't want).
- **Match the real defect-size distribution.** Fit a log-normal to your labelled defect widths/lengths (`scipy.stats.lognorm.fit`) and sample from it. If you sample uniformly, the model learns *your* size prior and misses the 1–3 px cracks that matter.
- **Max 1–3 defects per image.** More turns into "texture wash" and destroys the class signal.
- **Emit 500–2,000 images.** SynSur kept 420 synthetic patches (~2× the real defect count) and that was enough; ScratchSim needed 10k because it was training a detector from scratch on rendered scenes. For fine-tuning a pretrained segmentation model, 1–2k is plenty. Beyond that you are just burning hours.

**Directory layout**

```
data/
  real/{train,val,test}/{images,masks}      # masks 0/255, class in filename
  synth/{train,val}/{images,masks}
  manifest.csv    # split,source,class,part_id,session,seed,area_px,severity
```
Split **by part_id and by capture session**, never by image. Same part shot twice = leakage; that single mistake makes hackathon AUROCs meaningless and judges (who are often practitioners) will ask.

---

### 2.1 Recipe → defect class map

| Recipe | Class it should generate | Key signal it must reproduce |
|---|---|---|
| R1 random-walk crack | hairline crack, crazing | thin (1–3 px), branching, low contrast, dark core + faint bright edge |
| R2 Bézier scratch | scratch, abrasion, polish mark | straight-ish, tapered ends, **directional specular**, parallel micro-grooves |
| R3 fBm/Perlin blob + relief | dent, bump, orange-peel, unevenness | low-frequency, soft edges, shading only, **no albedo change** |
| R4 pit/porosity cluster | blowhole, pitting, inclusion | small dark blobs, Poisson-disc spacing, bright rim |
| R5 Cut-Paste / Cut-Paste-Scar | generic "foreign material", dents, tears | boundary discontinuity + texture mismatch |
| R6 DRAEM blend (texture × Perlin mask) | any OOD anomaly for AD training | "just-out-of-distribution" appearance |
| R7 Poisson / MIXED_CLONE composite | stains, discoloration, weld spatter | gradient continuity, no visible seam |
| R8 geometric edit (move/remove/rotate a component + contact shadow) | **assembly defects** (missing screw, misalignment, wrong part, gap) | structure, not texture — copy-move is the only realistic cheap way |
| R9 contour perturbation (1D noise on part edge, dilate) | burr, edge chip, flash | jagged silhouette |

R8 is the one teams forget; "assembly defect" is a *logical/structural* anomaly (the MVTec LOCO class of problem) and pixel-noise synthesis cannot produce it. Cut a component, translate/rotate it by 2–20 px/±15°, paste back, add a soft contact shadow (`cv2.GaussianBlur` of the shifted alpha, multiply 0.55), and **also** produce the "missing component" variant by inpainting the hole with surrounding texture.

---

### 2.2 Core code (drop-in, numpy + OpenCV only)

```python
# synth/core.py
import numpy as np, cv2, json, math, random
from pathlib import Path

# ---------- noise ----------
def _value_noise(h, w, cell, rng):
    gh, gw = max(2, h // cell), max(2, w // cell)
    g = rng.random((gh, gw), dtype=np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)

def fbm(h, w, octaves=4, base_cell=64, persistence=0.55, rng=None):
    """Fractal value noise in [0,1]. base_cell=32..96 for dents, 8..24 for micro-texture."""
    rng = rng or np.random.default_rng()
    out = np.zeros((h, w), np.float32); amp, cell, tot = 1.0, base_cell, 0.0
    for _ in range(octaves):
        out += amp * _value_noise(h, w, cell, rng)
        tot += amp; amp *= persistence; cell = max(2, cell // 2)
    out /= tot
    return (out - out.min()) / (out.ptp() + 1e-6)

def perlin_mask(h, w, rng, thresh=(0.45, 0.75), octaves=4, cell=(24, 96)):
    n = fbm(h, w, octaves=octaves, base_cell=rng.integers(*cell), rng=rng)
    t = rng.uniform(*thresh)                      # DRAEM: threshold sampled uniformly
    m = (n > t).astype(np.uint8)
    if rng.random() < 0.5:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m

# ---------- curves ----------
def bezier(p0, p1, p2, p3, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

def scratch_polyline(rng, h, w, length, straightness=0.85):
    """R2: tapered scratch. straightness→1 = ruler-straight, →0.6 = hand-drag."""
    ang = rng.uniform(0, np.pi)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    L = length
    p0 = np.array([cx - np.cos(ang) * L / 2, cy - np.sin(ang) * L / 2])
    p3 = np.array([cx + np.cos(ang) * L / 2, cy + np.sin(ang) * L / 2])
    perp = np.array([-np.sin(ang), np.cos(ang)])
    bow = (1 - straightness) * L * 0.35
    p1 = p0 + (p3 - p0) * 0.33 + perp * rng.normal(0, bow)
    p2 = p0 + (p3 - p0) * 0.66 + perp * rng.normal(0, bow)
    return bezier(p0, p1, p2, p3, n=max(40, int(L)))

def crack_paths(rng, h, w, n_steps=80, step=(1.5, 4.0), curl=0.20, branch_p=0.03):
    """R1: random walk with momentum + branches. Returns list of (N,2) float arrays."""
    x, y = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    th = rng.uniform(0, 2 * np.pi)
    pts = [(x, y)]
    for _ in range(n_steps):
        th += rng.normal(0, curl)
        s = rng.uniform(*step)
        x, y = x + s * np.cos(th), y + s * np.sin(th)
        if not (1 < x < w - 1 and 1 < y < h - 1):
            break
        pts.append((x, y))
    paths = [np.array(pts, np.float32)]
    for i in range(2, len(pts) - 3):
        if rng.random() < branch_p:
            bx, by = pts[i]
            bth = th + rng.choice([-1, 1]) * rng.uniform(0.4, 1.1)
            bp = [(bx, by)]
            for _ in range(rng.integers(8, 25)):
                bth += rng.normal(0, 0.25)
                bx, by = bx + 2.0 * np.cos(bth), by + 2.0 * np.sin(bth)
                if not (0 < bx < w and 0 < by < h):
                    break
                bp.append((bx, by))
            if len(bp) > 3:
                paths.append(np.array(bp, np.float32))
    return paths

# ---------- rasterisation ----------
def raster_paths(paths, h, w, width=(1, 3), ss=4, rng=None):
    """Supersampled anti-aliased stroke density map in [0,1]. ss=4 kills aliasing."""
    rng = rng or np.random.default_rng()
    big = np.zeros((h * ss, w * ss), np.float32)
    for p in paths:
        p = p * ss
        for a, b in zip(p[:-1], p[1:]):
            wpx = rng.uniform(*width) * ss
            cv2.line(big, tuple(np.int32(a)), tuple(np.int32(b)),
                     float(rng.uniform(0.55, 1.0)), max(1, int(round(wpx))), cv2.LINE_AA)
    small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(small, 0, 1)

# ---------- photometric integration ----------
def relief_shade(img, height, alpha, light_deg=35.0, relief=0.9,
                 spec=0.0, shininess=24, tint=(1.0, 1.0, 1.0)):
    """
    Heightmap -> normals -> Lambert + optional specular, applied only inside `alpha`.
    height: float32 HxW (0 = flat). relief 0.3 (subtle) .. 2.0 (deep gouge).
    This is what makes a scratch look like a scratch: bright on the lit flank, dark on the other.
    """
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx, ny, nz = -gx * relief, -gy * relief, np.ones_like(gx)
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / nrm, ny / nrm, nz / nrm
    a = np.deg2rad(light_deg)
    L = np.array([np.cos(a), -np.sin(a), 0.65], np.float32); L /= np.linalg.norm(L)
    lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    flat = L[2]                                   # lam where the surface is flat
    shade = (0.55 + 0.45 * (lam / flat - 1.0) * 2.0)   # ≈1.0 on flat, ± around it
    if spec > 0:
        H = L + np.array([0, 0, 1.0]); H /= np.linalg.norm(H)
        shade = shade + spec * np.clip(nx * H[0] + ny * H[1] + nz * H[2], 0, 1) ** shininess
    shade = 1.0 + (shade - 1.0) * alpha[..., None]
    out = img.astype(np.float32) * shade
    out = out * (1 + alpha[..., None] * (np.array(tint, np.float32) - 1.0))
    return np.clip(out, 0, 255).astype(np.uint8)

def elastic(img, mask, rng, amp=(1.5, 6.0), sigma=(6, 14)):
    """Local warp of image+mask together. Makes procedural lines stop looking vector-drawn."""
    h, w = mask.shape
    dx = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), rng.uniform(*sigma))
    dy = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), rng.uniform(*sigma))
    a = rng.uniform(*amp)
    dx = dx / (np.abs(dx).max() + 1e-6) * a
    dy = dy / (np.abs(dy).max() + 1e-6) * a
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    M = np.stack([gx + dx, gy + dy], -1)
    return (cv2.remap(img, M, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT),
            cv2.remap(mask, M, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT))

# ---------- compositing ----------
def poisson_paste(dst, patch, mask01, center, mode=cv2.MIXED_CLONE):
    """Seamless insert. src/dst/mask must be the same size; centre must be inside the mask."""
    m8 = (mask01 > 0.5).astype(np.uint8) * 255
    ys, xs = np.where(m8 > 0)
    if len(xs) == 0:
        return dst
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx = int(np.clip(center[0], x0, x1)); cy = int(np.clip(center[1], y0, y1))
    try:
        blended = cv2.seamlessClone(patch, dst, m8, (cx, cy), mode)
    except cv2.error:
        blended = dst
    out = dst.copy()
    soft = cv2.GaussianBlur(mask01.astype(np.float32), (0, 0), 1.2)[..., None]
    out = (blended.astype(np.float32) * soft + dst.astype(np.float32) * (1 - soft))
    return np.clip(out, 0, 255).astype(np.uint8)

def draem_blend(img, tex, mask01, beta):
    """I_a = (1-M)I + M[(1-beta)I + beta*A];  beta ~ U(0.1, 1.0)  [DRAEM eq.4]"""
    m = mask01[..., None].astype(np.float32)
    return (img * (1 - m) + ((1 - beta) * img + beta * tex) * m).astype(np.uint8)
```

### 2.3 Per-class recipes with parameter ranges

**R1 — hairline crack (the hardest, most valuable)**

```python
def gen_crack(img, rng, part_mask=None):
    h, w = img.shape[:2]
    paths = crack_paths(rng, h, w,
                        n_steps=int(rng.integers(40, 140)),
                        step=(1.2, 3.5), curl=rng.uniform(0.12, 0.28),
                        branch_p=rng.uniform(0.01, 0.05))
    a = raster_paths(paths, h, w, width=(1.0, 2.4), rng=rng)   # 1–2.4 px → 2–5 px after 2x upscale
    a = np.clip(a * rng.uniform(0.6, 1.0), 0, 1)
    if rng.random() < 0.5:                                    # cracks are not perfectly sharp
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.4, 1.0))
    a = (a > rng.uniform(0.08, 0.2)).astype(np.float32) * a
    height = -a * rng.uniform(2.0, 8.0)                       # negative = groove
    out = relief_shade(img, height, a, light_deg=rng.uniform(-80, 80),
                       relief=rng.uniform(0.5, 1.6), spec=rng.uniform(0.0, 0.25),
                       tint=(rng.uniform(0.55, 0.85),) * 3)    # dark core
    mask = (a > 0.25).astype(np.uint8)
    return out, mask
```

Ranges that matter: **width 1–5 px at 1× resolution**, **contrast 5–40 grey levels**, **branch probability 1–5%**, **elastic amp 1.5–6 px**. Real cracks on a 2000 px-wide phone photo are 2–6 px; if your synthetic cracks are 15 px wide the model will only ever fire on thick scratches.
Mapping: crack / crazing / edge chip.

**R2 — scratch / abrasion (multi-groove, specular)**

```python
def gen_scratch(img, rng, n_grooves=None):
    h, w = img.shape[:2]
    L = rng.uniform(0.15, 0.55) * max(h, w)
    base = scratch_polyline(rng, h, w, L, straightness=rng.uniform(0.62, 0.97))
    paths = [base]
    for _ in range((n_grooves or rng.integers(1, 4)) - 1):      # parallel micro-grooves
        off = rng.normal(0, rng.uniform(2, 6), 2)
        paths.append(base + off)
    a = raster_paths(paths, h, w, width=(1.0, 3.0), rng=rng)
    taper = np.linspace(0, 1, a.shape[1])[None, :]             # fade ends (R2 signature)
    a = a * (0.35 + 0.65 * np.clip(np.minimum(taper, 1 - taper) * 3, 0, 1))
    height = -a * rng.uniform(3.0, 10.0)
    return relief_shade(img, height, a, light_deg=rng.uniform(-90, 90),
                        relief=rng.uniform(0.8, 2.0),
                        spec=rng.uniform(0.15, 0.55), shininess=rng.uniform(12, 40),
                        tint=(rng.uniform(1.05, 1.3),) * 3), (a > 0.3).astype(np.uint8)
```

Specular is the whole trick: a scratch is visible mainly because it reflects the light source at a different angle than the surrounding surface. If you paste a flat grey line, the model learns "grey line" and fails under your test lighting.

**R3 — surface irregularity (dent / bump / orange peel)**

```python
def gen_dent(img, rng):
    h, w = img.shape[:2]
    n = fbm(h, w, octaves=rng.integers(3, 6), base_cell=rng.integers(40, 160), rng=rng)
    cy, cx = rng.uniform(.2, .8) * h, rng.uniform(.2, .8) * w
    r = rng.uniform(0.04, 0.20) * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))
    z = (n - 0.5) * 2 * blob * rng.uniform(0.3, 1.0) * (1 if rng.random() < .5 else -1)
    z = cv2.GaussianBlur(z.astype(np.float32), (0, 0), rng.uniform(8, 25))
    alpha = np.clip(np.abs(z) / (np.abs(z).max() + 1e-6) * 1.6, 0, 1)
    out = relief_shade(img, z * 12, alpha, light_deg=rng.uniform(-80, 80),
                       relief=rng.uniform(0.4, 1.2), spec=0.0)
    return out, (alpha > 0.35).astype(np.uint8)
```

**R4 — pitting / porosity**: sample points with a Poisson-disc (`cv2.goodFeaturesToTrack` over random noise, or rejection sampling with min distance 8–25 px); radius 1–5 px; per-pit `tint` 0.4–0.7; rim highlight = dilate(mask) − mask shaded with `relief`. Cluster 5–40 pits.

**R5 — CutPaste / CutPaste-Scar** (respect the paper's numbers):
- patch area ratio **U(0.02, 0.15)** of the image, aspect ratio **U(0.3,1) ∪ U(1,3.3)**;
- CutPaste-Scar: width **2–16 px**, length **10–25 px**, rotation **±45°**;
- color-jitter the patch (brightness/contrast/saturation 0.1 each, hue 0.1) *before* pasting, otherwise the discontinuity is too obvious;
- label = pasted region (erode by 1 px so the boundary is not part of the label).
Optionally cheap-blend with `poisson_paste(...)` — CutPaste's naive paste is *intentionally* easy to detect; Poisson is better for supervised training.

**R6 — DRAEM blend**: `mask = perlin_mask()`, `tex = random crop of DTD/ImageNet/dataset-of-scrap-textures` (rotation ±90°, random contrast), `beta ~ U(0.1, 1.0)`, `img = draem_blend(...)`. This is the correct generator for **Head A training** (discriminative reconstruction), not for class-conditional segmentation.

**R7 — Poisson stain/discoloration**: build a smooth alpha (fbm thresholded at 0.5, blurred σ 6–20), tint in LAB (`L += -25..+10`, `a/b ±8`), `poisson_paste`. Map to: oil stain, water spot, discoloration, burn mark.

**R8 — assembly defect (copy-move)**:

```python
def gen_assembly(img, rng, region=None):
    """region = bbox of a component (from SAM/segmentation or a hand-drawn box)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region or (int(.2*w), int(.2*h), int(.45*w), int(.45*h))
    comp = img[y0:y1, x0:x1].copy()
    mode = rng.choice(["shift", "rotate", "remove", "duplicate"])
    if mode == "shift":
        dx, dy = rng.integers(3, 22, 2) * rng.choice([-1, 1], 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(comp, M, (comp.shape[1], comp.shape[0]), borderMode=cv2.BORDER_REFLECT)
    elif mode == "rotate":
        M = cv2.getRotationMatrix2D((comp.shape[1]/2, comp.shape[0]/2), rng.uniform(-18, 18), 1.0)
        warped = cv2.warpAffine(comp, M, (comp.shape[1], comp.shape[0]), borderMode=cv2.BORDER_REFLECT)
    elif mode == "remove":
        fill = cv2.inpaint(img, np.ones((h, w), np.uint8) * 0, 3)   # placeholder: use surrounding patch
        warped = cv2.resize(img[max(0,y0-30):y0, x0:x1], (comp.shape[1], comp.shape[0]))
    else:
        warped = comp.copy()
    out = img.copy(); out[y0:y1, x0:x1] = warped
    mask = np.zeros((h, w), np.uint8); mask[y0:y1, x0:x1] = 255
    # contact shadow makes it read as 3-D
    sh = cv2.GaussianBlur(mask, (0, 0), 6).astype(np.float32) / 255.
    out = np.clip(out.astype(np.float32) * (1 - 0.35 * np.roll(sh, 6, axis=0))[..., None], 0, 255).astype(np.uint8)
    return out, mask
```
Honest caveat: `remove` needs a real inpainting step (`cv2.inpaint` with a proper mask, or LaMa if available). For a hackathon, `shift`/`rotate`/`duplicate` already produce convincing "misaligned / double-fed / wrong orientation" defects, and "missing" can be simulated by pasting a *background* patch from the same part.

**R9 — burr/edge chip**: extract the part contour (`cv2.findContours`), take 1D arc-length noise `n(s)`, offset the contour outward by `n(s)` with amplitude 1–8 px in a random 10–25% arc, fill the difference; shade with relief. Map to: burr, flash, edge chip, delamination.

---

### 2.4 The generator driver + the 3 quality gates

```python
# synth/make_dataset.py
import cv2, numpy as np, pandas as pd
from pathlib import Path
from core import *   # the functions above

RECIPES = {"crack": gen_crack, "scratch": gen_scratch, "dent": gen_dent,
           "pit": gen_pit, "assembly": gen_assembly}

def part_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > max(25, np.percentile(hsv[..., 1], 60))).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    k = max(1, int(0.02 * min(img.shape[:2])))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m[:] = 0
    if cnts:
        cv2.drawContours(m, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        m = cv2.erode(m, np.ones((15, 15), np.uint8))   # keep defects away from the silhouette
    return m

def run(normals, out_dir, per_image=12, size=768, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in normals:
        img0 = cv2.imread(str(p))
        h, w = img0.shape[:2]
        s = size / max(h, w)
        img0 = cv2.resize(img0, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        pm = part_mask(img0)
        for k in range(per_image):
            cls = rng.choice(list(RECIPES))
            img = img0.copy(); total = np.zeros(img.shape[:2], np.uint8)
            for _ in range(rng.integers(1, 4)):
                for _try in range(6):
                    cand, m = RECIPES[cls](img, rng)
                    if pm.any() and (cand.astype(np.int16) - img.astype(np.int16)).__abs__().sum() == 0:
                        continue
                    ys, xs = np.where(m > 0)
                    if len(xs) == 0:
                        continue
                    if not pm.any() or pm[ys, xs].mean() > 0.8:   # centroid-ish test
                        img, total = cand, np.maximum(total, m)
                        break
            if total.sum() == 0:
                continue
            if rng.random() < 0.35:
                img, total = elastic(img, total, rng)
            sid = f"{p.stem}_{k:03d}"
            cv2.imwrite(f"{out_dir}/images/{sid}.png", img)
            cv2.imwrite(f"{out_dir}/masks/{sid}.png", total * 255)
            rows.append(dict(id=sid, src=p.name, cls=cls, area_px=int(total.sum()),
                             area_frac=float(total.mean()), seed=int(seed)))
    pd.DataFrame(rows).to_csv(f"{out_dir}/manifest.csv", index=False)
```

**Gate 1 — "does the anomaly model see it?"** Train PatchCore/EfficientAD on *good* images only, then require `mean(score[mask]) / mean(score[~mask]) > 3` and `max(score) > p99.9(normal scores)` for ≥80% of synthetic samples. Samples that fail are invisible → delete them. This single filter removes most bad procedural defects and takes 10 minutes.

**Gate 2 — "can a classifier tell synthetic from real?"** Train a tiny CNN (ResNet-18, 5 epochs) on real-defect patches vs synthetic-defect patches. Target AUC **0.65–0.85**.
- AUC > 0.9 → synthetic is cartoonishly fake (too sharp, too grey, too high contrast). Fix: more elastic, more blur, lower contrast, add JPEG, match the real size distribution.
- AUC < 0.55 → you are not generating a defect at all, or your real set is itself mostly ambiguous.
Caveat: with 40 real patches this classifier is itself noisy; use it as a *relative* signal across generator configs, not an absolute truth.

**Gate 3 — human 4×3 grid review.** 12 samples per class, contact sheet, 2 minutes. Ask: "would I write this down as a defect on an inspection sheet?" Kill any recipe that fails twice.

**Time budget (single laptop, no GPU):**
| Phase | Min |
|---|---|
| repo skeleton, IO, manifest, part mask | 20 |
| R1 crack + R2 scratch (80% of the value) | 40 |
| R3 dent + R4 pit + relief shading tuning | 30 |
| R8 assembly copy-move | 20 |
| Poisson/elastic/JPEG finishers | 15 |
| Gate 1 + Gate 3, delete bad samples | 15 |
| wire into dataloader, first training run | 30 |

---

## 3. If there IS a GPU: generative options with honest costs

| Method | Repo | Feasibility in ≤4 h on 1 GPU | Real cost | Verdict |
|---|---|---|---|---|
| **TF-IDG** (ICCV 2025) | `github.com/rubymiaomiao/TF-IDG` | **High** — *training-free*, one-shot, AnyDoor+DINOv2 checkpoints | ~22 s/image inference; 15.4 GB model | **Best GPU pick.** One real defect example → many aligned variants. No training, no LoRA, no epochs. |
| **AnomalyDiffusion** (AAAI 2024) | `github.com/sjtuplayer/anomalydiffusion` (MIT) | **Medium** — but **do not retrain** (≈390 GPU-hours per the ICCV'25 comparison table). Use the **released generated data + checkpoints** (Google Drive links in README) for MVTec categories. | env is Python 3.8 / CUDA 11.8 / old LDM — dependency hell | Good if your category is in MVTec and you just need data, fast. |
| **DFMGAN** (AAAI 2023) | `github.com/Ldhlwh/DFMGAN` | **Low** — StyleGAN2 two-stage, ≈**621 GPU-hours** total on MVTec; needs 2 GPUs in the paper | days | Do not attempt. Cite it, don't run it. |
| **SDXL inpainting + LoRA** | `diffusers` `AutoPipelineForInpainting` + `train_dreambooth_lora_sdxl.py` | **Medium-high** — LoRA 500–1500 steps on 100 patches, 30–90 min on 24 GB | 4–8 s/img at 1024², 25–30 steps, fp16 | Best *control*: you own the mask shape, you get exact labels. |
| **Flux.1-dev + LoRA (SynSur recipe)** | SynSur (2026) | **Medium** — r=16, **2,000 steps, 100 patches**, 30 sampling steps; 1000 candidates → keep 420 by DreamSim+CLIPScore | 24 GB comfortable; ~1–2 h train + 1 h gen | The most current industrial recipe; mimic its hyperparameters. |
| **ControlNet (crack, SD1.5)** | `lllyasviel/ControlNet` + `control_v11p_sd15_lineart` | **Medium** — needs paired (sketch, crack) data; a small crack set can fine-tune in ~1 h at 512² | 512² is the catch (thin cracks get lost) | Only worth it if your defect *is* cracks and you already have 200+ labelled ones. |
| **CycleGAN / CUT domain transfer** (sim→real or dataset→phone) | `junyanz/CycleGAN`, `taesungp/contrastive-unpaired-translation` | **Medium** — unpaired, 200–600 imgs/side, few hours | unstable, hallucinates defects, no mask guarantee | Use only for **style transfer of defect-free backgrounds**, then re-composite defects. Never trust its masks. |

### 3.1 SDXL inpainting that actually works on 2–6 px defects

The #1 failure mode: you pass a 2000×1500 image with a 40×20 px defect, the pipeline resizes to 1024², and the defect becomes 20×10 px of VAE mush. Fix — **crop with context, generate, composite back**:

```python
import torch, cv2, numpy as np
from diffusers import AutoPipelineForInpainting

pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.enable_vae_slicing(); pipe.enable_model_cpu_offload()   # 12 GB cards

def inpaint_defect(img, mask, prompt, seed=0, ctx=2.2, strength=0.85):
    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad_x = int((x1 - x0) * ctx) + 24; pad_y = int((y1 - y0) * ctx) + 24
    X0, Y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    X1, Y1 = min(img.shape[1], x1 + pad_x), min(img.shape[0], y1 + pad_y)
    crop = img[Y0:Y1, X0:X1]; mcrop = mask[Y0:Y1, X0:X1]
    # keep aspect, target 1024 on the long side; padding_mask_crop does the same job inside diffusers
    s = 1024 / max(crop.shape[:2])
    crop_r = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    m_r = cv2.resize(mcrop, (crop_r.shape[1], crop_r.shape[0]), interpolation=cv2.INTER_NEAREST)
    m_r = cv2.dilate(m_r, np.ones((7, 7), np.uint8))          # let the model redraw the boundary
    out = pipe(prompt=prompt, negative_prompt="blurry, blurry defect, text, watermark",
               image=cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB),
               mask_image=cv2.cvtColor(m_r, cv2.COLOR_GRAY2RGB),
               strength=strength, guidance_scale=6.0,
               num_inference_steps=30,
               generator=torch.Generator("cuda").manual_seed(seed),
               padding_mask_crop=32).images[0]
    out = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    out = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
    soft = cv2.GaussianBlur(mcrop.astype(np.float32), (0, 0), 1.5)[..., None] / 255.
    res = img.copy()
    roi = (out.astype(np.float32) * soft + crop.astype(np.float32) * (1 - soft))
    res[Y0:Y1, X0:X1] = np.clip(roi, 0, 255).astype(np.uint8)
    return res, mcrop
```

Prompt template that works (borrowed from SynSur's tag structure — scene, defect, texture, capture):
> `pitting defect on galvanized steel, irregular pitted surface, rough grainy texture, dark pits, subtle metallic sheen, close-up industrial inspection photo, shallow depth of field, low contrast, diffuse lighting`

Note `strength=0.85`, not 1.0: at 1.0 the model ignores the clean background and you lose surface continuity; at 0.6 you get a faint smudge instead of a defect. Sweep `{0.7, 0.8, 0.85, 0.9}` on 5 samples, pick by eye, move on.

**Pitfalls, in order of how much time they waste:**

1. **Do not train from scratch.** LoRA only, r=16, 100–300 patches, 1,000–2,000 steps. Full fine-tune = you won't finish.
2. **Label drift.** The generated defect rarely fills the mask exactly. Either (a) derive the label from the *mask you passed* and accept ±2 px boundary noise (fine if you evaluate with 2 px tolerance), or (b) run SAM2 on the generated crop with the mask bbox as prompt (better, +10 min). Do not trust an inpainted region as a pixel-perfect mask.
3. **Augment after generation, not before.** Diffusion output has no sensor noise, no JPEG, no motion blur. If you skip Section 5's photometric pipeline, you have created a trivially separable domain and your "improvement" won't survive the phone-camera test set.
4. **Synthetic-only = collapse.** SynSur: AP 0.393 synthetic-only vs 0.652 real-only; MSD: 0.500 vs 0.953. Mix 75/25 or union.
5. **Filter the output.** Keep ~40% of candidates. A cheap substitute for DreamSim/CLIPScore: embed all synthetic + real defect crops with DINOv2 (`torch.hub`), drop the synthetic samples whose **min** cosine distance to a real defect is in the top decile, and drop those whose **mean k=3** distance is in the bottom decile (too close = copies). 20 lines, runs in seconds.
6. **VRAM math:** SDXL 1024² fp16 ≈ 10–12 GB; attention slicing + cpu offload gets you under 8 GB at ~2× latency. Flux.1-dev is 12B — quantize (fp8/NF4) or don't play.
7. **CycleGAN hallucination:** if you use it for sim→real, verify with Gate 1 that defects survived. It routinely deletes thin structures because they look like noise to the discriminator.

---

## 4. Domain randomization and sim-to-real (why it matters *because* it's a phone)

Your train set is a dataset shot with a machine-vision camera under controlled light; your test set is a phone held by a human. The gap is not "synthetic vs real", it's **illumination geometry, white balance, exposure, focus, and background**. MVTec AD 2 was built precisely around this: test images are deliberately brighter/darker than train, and SOTA methods fall **below 60% AU-PRO**. That number is your slide.

**What to randomize, in priority order** (each line is a knob, not a vague suggestion):

| Factor | Range | Why |
|---|---|---|
| Light **direction** (azimuth/elevation) | full azimuth, elevation 10–80° | scratches/cracks are only visible at grazing angles; this is THE factor |
| Light **temperature** | 3500–9000 K (Planckian) | phone AWB drifts; dataset is 6500 K |
| **Exposure** | ×0.5–×2.0, with highlight clipping allowed | MVTec AD 2's exact stress test |
| **Specular highlight** position/size | 1–3 blobs, radius 3–15% of image | glossy metal/plastic is everywhere in assembly parts |
| **Background** | ≥20 different materials/textures, incl. cluttered bench | the model must not learn "grey backdrop = ok" |
| **Camera distance / FOV** | scale 0.8–1.25, perspective 0.02–0.08 | phone is closer and wider than a fixed inspection cam |
| **Focus** | defocus radius 0–3 px, plus 1 sharp variant | handheld + macro = missed focus |
| **Sensor** | shot noise + PRNU-ish fixed pattern, ISO 100–3200 | phone denoise smears thin cracks |
| **Compression** | JPEG q 55–92, one re-encode | phone pipeline |
| **Motion blur** | kernel 3–13 px, random angle, p=0.2 | handheld |

**Cheap 2.5D compositor (30 min, no Blender).** Render your part as a cutout on a random background, then apply a *procedural light field* as a multiplicative shading map:

```python
def light_field(h, w, rng):
    """Sum of 1–3 von Mises-ish lobes = plausible studio/bench lighting."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx, sy = rng.uniform(0.25, 0.9) * w, rng.uniform(0.25, 0.9) * h
        amp = rng.uniform(0.25, 0.8)
        f += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(20, 60))
    f = (f - f.min()) / (f.ptp() + 1e-6)
    return 0.65 + 0.55 * f          # 0.65..1.2 multiplier
```
Then `img * light_field` **before** drawing defects, and use a `light_deg` derived from the same lobe direction in `relief_shade` so shading and relief agree. This single consistency rule is what makes procedural defects stop looking pasted-on.

**BlenderProc**, if (and only if) you have a CAD/mesh of the part:
```python
import bpy, bproc, random
bproc.init()
obj = bproc.loader.load_obj("part.obj")[0]
obj.set_material(bproc.material.create_material_from_texture("albedo.png", "rough.png"))
bproc.camera.set_resolution(1024, 1024)
for i in range(300):
    # domain randomisation = the whole point
    bproc.lighting.add_light(... )        # or randomize existing lights' energy/color
    cam_pose = bproc.math.build_transformation_mat(location, rotation)
    bproc.camera.add_camera_pose(cam_pose)
    data = bproc.renderer.render()        # Cycles, 64-128 spp
    bproc.writer.write_coco_annotations(out_dir, ..., data)
```
Reality check: 300 frames at 1024², 64 spp, Cycles on one consumer GPU ≈ 1.5–4 h, **plus** you still have to author scratches as UV maps/normal maps (that's the real work — see ScratchSim's 8192² canvas with 17–25 Bézier scratches, Gaussian blur, then material-level roughness/normal perturbation). For a 36–48 h hackathon with no CAD, **the 2.5D compositor wins**. BlenderProc is the right answer only when the judge asks "could this scale to a real line?" — then you say "yes, same parameter set drives a BlenderProc pipeline" and show the table above.

**If a CAD or a physical sample exists**, the one genuinely high-value sim trick is **photometric stereo**: shoot the good part 3–5 times with a phone flashlight at different angles, compute a normal map (`cv2.Sobel` on the log-shaded images, or `photometric_stereo` in OpenCV contrib), then render synthetic defects as perturbations of *that measured* normal map. Ten minutes of shooting, and your defects inherit the real surface's micro-texture and specularity.

---

## 5. Augmentation policy for inspection (and what to avoid)

Two different policies — do not share one config between heads:

### 5.1 Train-time (Head B, supervised segmentation)

```python
import albumentations as A, cv2

train_tf = A.Compose([
    # geometry — small, because defect SIZE is diagnostic
    A.Affine(scale=(0.9, 1.12), rotate=(-6, 6), translate_percent=(-0.04, 0.04),
             shear=(-3, 3), border_mode=cv2.BORDER_REFLECT_101, p=0.6),
    A.Perspective(scale=(0.01, 0.06), p=0.25),
    # photometry — the domain gap you actually have
    A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.35), contrast_limit=(-0.30, 0.30), p=0.8),
    A.PlanckianJitter(mode="blackbody", temperature_range=(3500, 9000), p=0.4),
    A.HueSaturationValue(hue_shift_limit=(-4, 4), sat_shift_limit=(-12, 12),
                         val_shift_limit=(-10, 10), p=0.35),
    A.RandomGamma(gamma_limit=(80, 125), p=0.2),
    A.CLAHE(clip_limit=(1.0, 3.0), p=0.2),          # cheap phone "HDR" look
    A.RandomShadow(p=0.2),
    # sensor / pipeline
    A.MotionBlur(blur_limit=(3, 11), allow_shifted=True, p=0.20),
    A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.4), p=0.20),
    A.GaussNoise(p=0.35),                            # tune std per albumentations version
    A.ImageCompression(quality_range=(55, 92), p=0.30),
    A.Downscale(scale_range=(0.6, 0.95), p=0.15),
    # cut AFTER photometry so masks stay exact
    A.RandomCrop(height=512, width=512, pad_if_needed=True, p=1.0),
    A.HorizontalFlip(p=0.5),                         # see caveats below
], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], min_visibility=0.3),
   mask_interpolation=cv2.INTER_NEAREST, seed=42)
```

Order matters: **geometry → photometry → sensor → crop**. Cropping before photometry means your lighting gradient varies per crop instead of per image, which is unrealistic and teaches the model to ignore illumination.

### 5.2 What actually helps vs. what hurts

| Helps | Why |
|---|---|
| illumination **direction** + exposure bracketing | the dominant failure mode (MVTec AD 2) |
| JPEG + downscale + re-upscale | phone pipeline; also stops the model keying on high-frequency cleanliness |
| defocus / motion blur at low p | thin cracks are destroyed by phone denoise; model must not require sharpness |
| sensor noise | prevents "smooth = normal" shortcut |
| **hard negatives**: dust, fingerprints, smudges, labels, reflections, weld seams, scratches on the *fixture* | these are 80% of your false positives |
| small affine + perspective | phone is handheld |
| copy-paste of *real* defects onto *real* good parts (with Poisson) | free, perfectly labelled, in-distribution — usually beats synthetic |

| Hurts (do not do) | Why |
|---|---|
| **RandomResizedCrop / aggressive zoom-out** | changes px/mm → destroys size prior; can cut a defect and leave a mask fragment |
| **90/180° rotation, vertical flip** on oriented parts | a wood-grain/brushed-metal texture has a direction; assembly parts have gravity/orientation priors. Horizontal flip only if the part is symmetric (test it) |
| **MixUp / CutMix / Mosaic** | labels become meaningless for dense prediction; Mosaic destroys part context |
| **Large brightness swings to black/white** | clips the defect away — you train on empty labels |
| **Grayscale / channel shuffle** | colour is a real defect cue (discoloration, burn, oxidation) |
| **Strong hue jitter (±30°+)** | turns "blue oxidation" into "red stain" class confusion |
| **Heavy JPEG (q<40)** at train but clean test | makes the model blind to fine cracks; keep the *test* set realistic too |
| **Random erasing / cutout** | erases the defect itself; teaches "hole = normal" |
| **GAN/diffusion aug on top of already-augmented pixels** | compounding artifacts; generate from clean images only |

### 5.3 Test-time

**None.** No TTA except (optionally) 4× flip-average for the image-level score, and only if you measured a gain. Report the number with and without TTA; judges notice when TTA is used to hide a weak model.

**One more policy:** for Head A (unsupervised), augmentation must be applied to the *good* training images only, and it must **not** create anomalies (no cutout, no heavy blur on a single image, no extreme contrast that clips structures). DRAEM/CutPaste rely on the nominal distribution being genuinely nominal.

---

## 6. Class imbalance, tiny defects, and thresholding

### 6.1 Losses, with numbers

Pixel imbalance in inspection is typically **1:10³–1:10⁵**. Plain BCE will converge to "all normal" and look fine on accuracy.

**Dice** (region overlap, scale-free):
```
Dice = 1 - (2·Σ p·g + ε) / (Σ p + Σ g + ε),  ε = 1.0 (soft), p = sigmoid(logit)
```
**Tversky** (asymmetric, penalise misses):
```
TI = TP / (TP + α·FN + β·FP),   α = 0.6, β = 0.4  → recall-leaning
FocalTversky = (1 - TI)^γ,      γ = 4/3
```
**Focal** (hard-pixel mining):
```
FL = -α_t (1 - p_t)^γ log(p_t),  α_t = 0.75 (positive class), γ = 2.0
```
**Recommended composite for cracks/scratches (thin, rare):**
```python
bce  = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(8.0))
dice = soft_dice_loss(logits, target)          # per-image, then mean
tl   = tversky_loss(logits, target, alpha=0.6, beta=0.4)
loss = 0.4*bce + 0.4*dice + 0.2*(1 - tl)
```
Why all three: BCE gives stable gradients, Dice handles the global imbalance, Tversky buys recall. `pos_weight=8–20` is the range that works at 1:10⁴; go higher only after you check that the model isn't firing everywhere.

For **classification with 6 classes and 3 rare ones**, use class-balanced CE (`weight = (1/n_c)^0.5` normalised) + label smoothing 0.05. Do **not** use plain inverse-frequency weights when a class has 5 samples — you will overfit those 5.

**Thin structures:** add **clDice** (centerline Dice). IoU/Dice are dominated by area, so a 2 px crack that is 80% correct scores ~0.1. clDice compares skeletons:
```
Tprec = |S_P ∩ V_L| / |S_P|,  Trec = |S_L ∩ V_P| / |S_L|,  clDice = 2·Tprec·Trec/(Tprec+Trec)
```
Report Dice **and** clDice for cracks; a model can be at Dice 0.35 / clDice 0.85 which is perfectly usable, and Dice alone will make you think you failed.

### 6.2 Patch-based training (the single biggest win for small defects)

A 2 px crack in a 2000×1500 image is 0.00007% of pixels. Two options:

1. **Train on 512² crops sampled with defect-biased probability**: `p(crop contains ≥1 defect px) = 0.7`, else uniform. Keep a "context" model by including 20% crops at 2× downscale.
2. **Train on tiles at native resolution** (see 6.4) and treat the problem as dense prediction on tiles.

Rule of thumb: **the defect must be ≥ 8 px in the network's input**, otherwise no architecture will localise it. Compute your px/mm from an ArUco/ruler shot and choose tile size accordingly.

### 6.3 Hard-negative mining (30-minute loop, high ROI)

```python
# after epoch 1
model.eval()
hard = []
with torch.no_grad():
    for img, path in good_val_loader:            # GOOD parts only
        s = model(img)["anomaly_map"]            # HxW
        s = F.avg_pool2d(s[None, None], 32)[0, 0] # 32x32 grid
        for (y, x) in (s > s.quantile(0.995)).nonzero()[:3]:
            hard.append((path, y*32, x*32))
# add as (image, all-zero mask, class="normal") samples with weight 1.0
```
Two refinements that matter:
- Mine from **good parts under the same lighting as your test set** (that's where FPs come from).
- Cap hard negatives at ~15% of the batch; otherwise the model learns "everything is normal" again.

### 6.4 Tile inference for high-res phone images

Do not resize a 4000×3000 phone photo to 512². Either:
- **SAHI-style tiling**: 640² tiles, 20% overlap, batch all tiles, stitch with max/hann weighting; or
- **two-stage**: detect the part (or use a fixed ROI jig), crop, then tile the ROI.

```python
def tiled_predict(model, img, tile=640, overlap=0.2, batch=8):
    H, W = img.shape[:2]; step = int(tile * (1 - overlap))
    ys = list(range(0, max(1, H - tile + 1), step)) + [max(0, H - tile)]
    xs = list(range(0, max(1, W - tile + 1), step)) + [max(0, W - tile)]
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    for y in ys:
        for x in xs:
            t = img[y:y+tile, x:x+tile]
            th, tw = t.shape[:2]
            # pad to tile
            t = cv2.copyMakeBorder(t, 0, tile-th, 0, tw and tile-tw, cv2.BORDER_REFLECT)
            s = model(t)[0, 0].cpu().numpy()[:th, :tw]
            w = np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32)
            acc[y:y+th, x:x+tw] += s * w; wsum[y:y+th, x:x+tw] += w
    return acc / np.maximum(wsum, 1e-6), {"tiles": len(ys)*len(xs), "tile": tile}
```
Cost: 4000×3000 with 640²/20% → ~48 tiles. On a 3060 that's ~0.4–0.8 s with a small model. Report **p95 latency for the tiled path**, not the 512² path — the plant cares about the former.

### 6.5 Anomaly-map → decision (this is where hackathon demos die)

```python
def decide(amap, cfg):
    a = cv2.GaussianBlur(amap, (0, 0), cfg.sigma)          # 1.0–2.0
    # 1) normalise by a *running* normal reference, not per-image max
    a = (a - cfg.mu) / (cfg.sigma_ref + 1e-6)              # mu/sigma from good training images
    # 2) top-k image score (more robust than max, which is a single-pixel lottery)
    k = max(1, int(0.001 * a.size))
    img_score = np.partition(a.ravel(), -k)[-k:].mean()
    # 3) region filtering: a defect is a region, not a pixel
    m = (a > cfg.tau).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = [i for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= cfg.min_area        # e.g. 12 px
            and stats[i, cv2.CC_STAT_AREA] <= cfg.max_area]       # e.g. 5% of part
    return img_score, m, keep
```
`min_area` and `tau` are **calibrated on good-only data**, not on the test set: pick `tau` as the (1 − target_FPR) quantile of image scores over good validation parts, then verify on the labelled test set. Per-image max normalisation is a classic mistake: it makes every image "have a defect somewhere".

---

## 7. Evaluation that survives a plant manager's questions

### 7.1 Metric hierarchy — report all five, in this order

| # | Metric | What it answers | Target for a demo |
|---|---|---|---|
| 1 | **FP per 1,000 good parts** | "how often do you stop my line for nothing?" | < 20 (excellent), < 50 (credible) |
| 2 | **Recall per defect class @ that FPR** | "which defects do you miss?" | > 0.85 on the 2 main classes |
| 3 | **Escape rate per 1,000 defective parts** | "how many bad parts ship?" | < 30 |
| 4 | **Latency p50/p95/p99** | "does it fit my takt time?" | p95 < takt (e.g. 1 s) |
| 5 | Image AUROC / pixel AUROC / AUPRO | comparability with papers | MVTec-class: img 0.95+, AUPRO 0.85+ |

Context for #1: AOI machines in electronics routinely alarm on 5–70% of *passing* boards and up to 95% of those alarms are false calls (Siemens white paper), while high-end 3D inspection targets **20–50 ppm** false calls. So "FPR 2%" is *normal for AOI* and *terrible for a pitch deck*. Say the number in per-1000, and say what it costs: at 1,000 parts/shift, 20 FP/1000 = 20 manual reviews/shift.

### 7.2 Why the three AUROCs disagree (say this out loud in the demo)

- **Image AUROC** — easy to inflate; one bright pixel can carry the score. With 30 defective test images the 95% CI is roughly ±8–12 points, so a 0.92 vs 0.95 difference is noise. Bootstrap it.
- **Pixel AUROC** — dominated by *large* defects; with 1:10⁴ imbalance, a few thousand easy pixels can hide total failure on cracks. Inflated by construction.
- **PRO / AUPRO** — averages TPR **per connected region**, so a 20 px crack counts as much as a 200,000 px stain. This is the right localisation metric for "did you find *the* defect". Standard cutoff is FPR ≤ 0.3 (some papers 0.1) — report which.
- **AUPIMO** (2024) is stricter and better correlated with deployment; mention it, implement it only if you have time.
- **sPRO** exists for MVTec LOCO-style logical anomalies (saturating at a threshold) — relevant to your "assembly defect" class.

Formula (PRO), for the scoring script:
```
PRO(t) = (1/N) Σ_i (1/K_i) Σ_k |P_i(t) ∩ C_{i,k}| / |C_{i,k}|
```
where `C_{i,k}` is the k-th connected component of the ground truth in image i, and `P_i(t)` the thresholded prediction. AUPRO = area under PRO vs FPR, normalised, FPR ∈ [0, 0.3].

**Part-level (multi-view) metric** — steal from Real-IAD: if a part is photographed 3 times, the part is defective if **any** view is defective. Compute the part score as `max` over views, and report both image-level and sample-level. Plant metrics are per part, not per image.

### 7.3 Scoring script outline (copy-paste-able)

```python
# eval/score.py
import numpy as np, pandas as pd, cv2, time
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def pro_score(gt_masks, pred_maps, fpr_limit=0.3, n_thr=100):
    """Per-region overlap, averaged over all GT regions, integrated up to fpr_limit."""
    # 1) collect all normal-pixel scores to build the FPR axis (pooled, as in the paper)
    # 2) for each threshold: region TPR per connected component, and pixel FPR
    # 3) trapezoid over FPR in [0, fpr_limit]
    neg, pos_regions = [], []
    for g, p in zip(gt_masks, pred_maps):
        neg.append(p[g == 0].ravel())
        n, lab = cv2.connectedComponents((g > 0).astype(np.uint8), 8)
        for k in range(1, n):
            pos_regions.append((p[lab == k], (lab == k)))
    neg = np.concatenate(neg)
    lo, hi = np.quantile(np.concatenate([r[0] for r in pos_regions]), [0.0, 1.0])
    thrs = np.linspace(lo, hi, n_thr)
    fprs, pros = [], []
    for t in thrs:
        fprs.append((neg >= t).mean())
        pros.append(np.mean([ (s >= t).sum() / m.sum() for s, m in pos_regions ]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    order = np.argsort(fprs[keep])
    return float(np.trapz(pros[keep][order], fprs[keep][order]) / fpr_limit)

def score_run(df, target_fpr=0.02):
    """df: one row per image — part_id, session, y_true(0/1), y_score, pred_mask_path, gt_mask_path, cls"""
    out = {}
    out["img_auroc"] = roc_auc_score(df.y_true, df.y_score)
    # threshold chosen on GOOD validation parts only
    good = df[df.y_true == 0].y_score
    tau = np.quantile(good, 1 - target_fpr)
    df["y_pred"] = (df.y_score >= tau).astype(int)
    fp = ((df.y_pred == 1) & (df.y_true == 0)).sum(); n_good = (df.y_true == 0).sum()
    fn = ((df.y_pred == 0) & (df.y_true == 1)).sum(); n_bad = (df.y_true == 1).sum()
    out["fp_per_1000_good"] = 1000 * fp / max(n_good, 1)
    out["escapes_per_1000_bad"] = 1000 * fn / max(n_bad, 1)
    out["review_per_1000"] = 1000 * df.y_pred.sum() / len(df)
    out["recall_by_class"] = df[df.y_true == 1].groupby("cls").y_pred.mean().to_dict()
    # per-part aggregation (multi-view)
    part = df.groupby("part_id").agg(y_true=("y_true", "max"), y_score=("y_score", "max"))
    out["part_auroc"] = roc_auc_score(part.y_true, part.y_score)
    # bootstrap CI on the metric you actually report (resample PARTS, not images)
    y, sc = part.y_true.values, part.y_score.values
    boots = []
    for s in range(500):
        idx = resample(np.arange(len(y)), random_state=s)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], sc[idx]))
    out["part_auroc_ci95"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out
```
Notes for the real implementation: (a) resample **parts**, never images — multi-view images of one part are not independent; (b) use `np.trapezoid` on NumPy ≥2.0 (`np.trapz` is deprecated); (c) for pixel metrics, ground-truth boundary ambiguity is real — evaluate with a **±2 px tolerance** (dilate GT for recall, erode for precision) and say so.

**Latency harness:**
```python
def bench(fn, n=300, warm=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {f"p{q}": float(np.percentile(ts, q)) for q in (50, 95, 99)} | {"mean": float(ts.mean())}
```
Report **end-to-end** (decode + preprocess + tiled inference + postprocess), batch=1, on the machine you will demo on, with the demo's image resolution.

**The scorecard slide (5 lines):**
```
Detected 41/44 defects (93%).  Per class: crack 12/13, scratch 9/10, dent 15/16, assembly 5/5.
False alarms: 14 per 1,000 good parts (95% CI 8–24) at the shipped threshold.
Misses: 1 hairline crack < 2 px, 2 low-contrast scratches under side-light.
p95 latency 380 ms/part (tiled 4 MP) — takt budget 1,000 ms.
Ablation: +synthetic defects = +11 pts recall on crack, +3 FP/1000 (net +).
```

---

## 8. 45-minute phone capture protocol (one page)

**Crew:** 2 people — one shoots, one logs and moves parts. One phone, one A3 grey card, one sheet of white paper (diffuser), one ruler/ArUco print, tape.

**Phone settings (set once, then never touch):** Pro/Manual mode · ISO 100–400 · shutter 1/125 s · **WB fixed 5000 K** · **AF/AE locked** by long-press · HDR off · Night mode off · "scene optimizer"/AI off · 4:3 at full resolution · grid on · timer 2 s (or volume-button shutter) · flash **off** (use the paper diffuser + a second phone torch).

**Set (10 min).**
1. Table by a window, **but do not mix window light and room LED** — pick one. Mixed colour temperature is the #1 cause of "the model worked yesterday".
2. Tape the grey card as the backdrop (or dark cloth for shiny parts). **No white paper background** (blows highlights, kills specular cues) and **no glossy table** (mirror reflections read as defects).
3. One lamp + A4 white paper as a diffuser = soft dome for matte parts. For shiny/metal, remove the diffuser and use **grazing light at 10–20°** — this is what makes cracks visible at all.
4. Put the ruler/ArUco in the corner of every 10th shot for px/mm calibration.
5. Shoot the log sheet first (date, part type, defect classes, lighting setup) so the metadata exists even if you forget later.

**Shot list — per defect class, 12 frames (35 min for 4 classes + good parts):**

| Axis | Values | Count |
|---|---|---|
| Angle around part | 0°, 45°, 90° | 3 |
| Elevation | top-down, 30° oblique | 2 (×3 = 6) |
| Distance | full part, macro (fill frame with defect) | 2 |
| Light direction | raking left, raking right | ×2 on the best angle → 4 extra |
| **Total** | | **~12–16** |

**Also shoot, and label them correctly:**
- **40 good parts**, same lighting/angles. This is your normal distribution and your FPR calibration set. Without it you cannot compute FP/1000.
- **20 hard negatives**: dust, fingerprints, smudges, tape residue, labels, weld seams, machining marks, reflections, fixture scratches, shadows. Label as **good**. These will cut your false-alarm rate by more than any model change.
- **5 "same part, 3 shots"** sequences to sanity-check multi-view aggregation.
- **2–3 shots at the wrong exposure** (one blown, one dark) to prove robustness (or to discover you have none).

**Naming (do it in the field, not later):** `P{part:03d}_{cls}_{sev}_{angle}_{light}_{nn}.jpg`, e.g. `P014_crack_med_045_rakeL_02.jpg`. Log severity 1–3 by eye: 1 = visible only under raking light, 2 = visible at arm's length, 3 = obvious. Severity is what lets you report "we catch all sev≥2 and 60% of sev1" — a far more impressive sentence than an AUROC.

**Labelling (10 min for ~120 images, 2 people):**
- **CVAT** (self-hosted docker, or app.cvat.ai): create one task per class, use **SAM 2 / "AI tools"** with a click per defect, then brush-fix; export **COCO 1.0** (segmentation) + a classification CSV. Keyboard: `N` next, `F` next frame, `Ctrl+S` save.
- **Label Studio**: `label-studio` + brush/pen mask + an **ML backend with SAM 2** (`label-studio-ml-backend` → `segment_anything_model`) so a click proposes the mask. Export JSON-MIN or COCO.
- **Rules:** one polygon per defect instance; **do not** include the shadow; include the full visible extent; for cracks thinner than 2 px, draw a 2 px-wide polyline (consistency > precision). If a defect touches the border, still label it (and keep it out of training if truncated >30%).
- **2-person agreement check:** both label 5 images, compute IoU. If < 0.5, your label definition is ambiguous — write it down before labelling 100 more.

**QC before you leave the room (last 5 min):** scroll the gallery — is exposure consistent? any motion blur? is the defect actually in focus? did you get all 4 classes? **Photograph the paper log.** Copy to the laptop and `sha256sum` it now; phones get locked, cables fail, and "the photos are on Satish's phone" is how hackathons are lost.

---

## 9. 36–48h execution plan with gates

| Hour | Do | Gate to pass |
|---|---|---|
| 0–2 | pick 1 dataset, download, build manifest + part/session splits, PatchCore/EfficientAD baseline on good-only | baseline img-AUROC on val; if <0.9 your splits leak or preprocessing is wrong |
| 2–4 | 45-min phone shoot, ingest, label 30 images first (pilot) | label definition agreed (IoU > 0.5 between annotators) |
| 4–7 | procedural generator (Section 2), 3 classes first: crack, scratch, dent | Gate 1 (anomaly model sees them) on ≥80% |
| 7–9 | wire synthetic into training, run **3 arms**: real-only / 75-25 / 100-100 | synthetic arm must beat real-only on *crack* recall, not just AUROC |
| 9–12 | augmentation policy (Section 5), hard-negative mining round 1 | FP/1000 drops ≥30% on the good-val set |
| 12–14 | **sleep** (rotate; do not skip) | — |
| 14–18 | tiled inference + threshold calibration + scoring script | p95 latency measured on demo machine |
| 18–22 | GPU branch if available: TF-IDG one-shot, or SDXL inpaint + LoRA (Section 3) | generated samples pass Gate 1 & Gate 2 |
| 22–26 | fusion head (anomaly score + class logits), per-class thresholds | per-class recall table looks sane |
| 26–32 | demo app: upload → heatmap → class → verdict + "flag for review"; show FP/1000 and latency live | a stranger can use it in 30 s |
| 32–38 | ablations + scorecard slide + failure gallery (show 3 misses, explain why) | every number has a CI or a caveat |
| 38–44 | buffer, video recording, README with repro commands | `python -m eval.score` runs from scratch |

**Cut list if behind:** drop diffusion entirely (Section 2 + 5 gets you 80% of the value), drop 2 of 4 defect classes, drop tiling (use a fixed ROI jig), drop multi-view. **Never cut:** the good-part set, the FP/1000 metric, or the latency measurement.

---

## 10. Failure modes checklist

- [ ] Synthetic defects drawn on the *background*, not the part → model learns background texture.
- [ ] Synthetic cracks 10–20 px wide → model blind to real 2 px cracks.
- [ ] Synthetic:real > 1:1 with realistic-looking-but-wrong morphology → class confusion.
- [ ] Same part in train and test → AUROC 0.99 in the notebook, 0.6 at the demo.
- [ ] Threshold tuned on the test set → the number is fiction; tune on good-val only.
- [ ] Per-image max normalisation of the anomaly map → every image looks defective.
- [ ] Global resize 4000×3000 → 512² → cracks vanish; use tiles or ROI crop.
- [ ] FP/1000 not reported → judges assume you don't know what a false alarm costs.
- [ ] Latency measured at batch 32 on a GPU you won't demo on.
- [ ] Only image AUROC reported → hides that you miss every hairline crack (Dice/AUPRO would show it).
- [ ] No CI on a 30-image test set → "0.94 vs 0.91" is noise; say so before a judge says it.
- [ ] Augmentations applied after the mask was cropped, or with `INTER_LINEAR` masks → shifted labels.
- [ ] Diffusion output used as ground truth without checking the mask actually covers the generated defect.
- [ ] Demo depends on network access (model hub, license server) — pre-download weights.

---

## 11. References (2023–2026 unless classic)

- **DRAEM** — Zavrtanik et al., ICCV 2021 · `github.com/VitjanZ/DRAEM` · Perlin mask + DTD texture, `β∈[0.1,1.0]`, eq. 4 blend; "simulations need only be just-OOD".
- **CutPaste** — Li et al., CVPR 2021 · patch area **0.02–0.15**, aspect **(0.3,1)∪(1,3.3)**, scar **w2–16 / l10–25 px**, rot ±45°.
- **PatchCore** — Roth et al., CVPR 2022 · 99.1 img-AUROC MVTec; anomalib default coreset ratio 0.1.
- **EfficientAD** — Batzner et al., WACV 2024 · 99.1 MVTec, **2 ms / 600 img/s**, ~20 min training; `github.com/nelson1425/EfficientAD`, in anomalib.
- **MVTec AD 2** — arXiv 2503.21622 / IJCV 2026 · 8 scenarios, 8,004 imgs, deliberate **unseen lighting**, SOTA **<60% AU-PRO**.
- **Real-IAD** — CVPR 2024 · 150k imgs, 30 objects, **sample-level (multi-view)** metric; Real-IAD Variety (2026) for 27 defect types.
- **AnomalyDiffusion** — AAAI 2024 · `github.com/sjtuplayer/anomalydiffusion` (MIT) · released generated data + checkpoints; retraining ≈390 GPU-h.
- **DFMGAN** — AAAI 2023 · `github.com/Ldhlwh/DFMGAN` · StyleGAN2 two-stage; **≈621 GPU-h** (per ICCV'25 comparison) — cite, don't run.
- **TF-IDG** — ICCV 2025 · `github.com/rubymiaomiao/TF-IDG` · training-free, one-shot, AnyDoor+DINOv2, ~22 s/img.
- **SynSur** — arXiv 2604.26633 (2026) · Flux.1-dev LoRA **r=16, 2000 steps, 100 patches**, 30 sampling steps, 1000→420 filtered; mixed 75/25 best; synthetic-only AP 0.393 vs 0.652.
- **ScratchSim** — arXiv 2607.27065 (2026) · BlenderProc scratch pipeline; 17–25 Bézier scratches on 8192² canvas; synthetic-only 0.451 vs real 0.610 mAP50; **finetune synth→real best (0.723)**; background matching matters.
- **BlenderProc** — `github.com/DLR-RM/BlenderProc` · DR over materials/lights/camera; procedural defect rendering for surface inspection (Procedia CIRP 2022).
- **AUPIMO** — arXiv 2401.01984 (2024) · stricter, deployment-correlated localisation metric.
- **clDice** — Shit et al., CVPR 2021 · centerline Dice for thin structures.
- **AOI economics** — Siemens "Smarter false call reduction" (2023): 5–70% alarm rate on passing throughput, up to 95% false calls; high-end 3D inspection targets 20–50 ppm.
