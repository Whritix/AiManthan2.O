# DATASET CATALOG
## Verified sources, direct links, sizes, licenses, and the exact split plan

> **Verified:** 2026-10-27 against live endpoints from the build machine. Each entry states its HTTP status.
> **Company:** `TRAINING_SPEC.md` (how the data is used), `RESEARCH.md` §9 (why these datasets).
> **Rule:** download once, verify once, commit the manifest. Never re-download at the venue.

---

## 0. TL;DR — what you actually need

| Priority | Dataset | Size | Purpose | Verified |
|---|---|---|---|---|
| **P0** | **NEU-DET** (KeenForgeAI mirror) | ~120 MB | Train the YOLO11s detector | ✅ 200 |
| **P0** | **Your own captures** | ~200 MB | The live demo + the novelty gate | you shoot it |
| **P1** | **MVTec AD** | 4.9 GB | Gate benchmark + fallback | ✅ 200 |
| **P2** | GC10-DET | ~2 GB | Cross-dataset robustness slide | ✅ 200 |
| **P2** | VisA | 1.93 GB | Second benchmark, PCB subsets | ✅ 200 |
| **P3** | MVTec AD 2 / LOCO | GB-scale | Credibility slide only | ✅ 200 (form) |
| **☠** | Kaggle NEU-DET mirror | — | **404 — dead** | ❌ 404 |
| **☠** | Kaggle casting-product mirror | — | **404 — dead** | ❌ 404 |

**Minimum viable download: 120 MB.** That is the entire critical path. Everything else is slide material.

---

## 1. NEU-DET — primary detector dataset

### 1.1 What it is

| Property | Value |
|---|---|
| Full name | NEU Surface Defect Database (hot-rolled steel strip) |
| Authors | Kechen Song & Yunhui Yan, Northeastern University, China |
| Images | **1,797** grayscale, **200 × 200 px** |
| Classes | 6 — `crazing` (300), `inclusion` (300), `patches` (299), `pitted_surface` (299), `rolled-in_scale` (300), `scratches` (299) |
| Labels | **Dual format: Pascal VOC XML + YOLO `.txt`** |
| Splits | Pre-packaged 80/10/10 (1,437 / 180 / 180) — **you will override this** |
| Size | ~120 MB |
| License | **None stated upstream.** Research use. Cite Song & Yan (2013). Do not claim commercial rights |

### 1.2 Direct links (verified)

**⭐ Primary — HuggingFace, cleaned + dual-format (use this)**
```
https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected
```
Status **200**. 15,155 downloads. Last updated 2026-09-21. DOI: `10.57967/hf/10536`.
Contains: `images/{train,val,test}/*.jpg`, `labels/{train,val,test}/*.txt`, `annotations/*.xml`, `ImageSets/*.txt`, `data.yaml`, `classes.txt`, `LICENSE`.

Why this one: it fixes 1 exact duplicate, 2 near-duplicates and 3 duplicate boxes from the official release, and ships both label formats. Duplicates across splits are a silent accuracy leak — this mirror removes them for you.

**Official upstream (form / Chinese page)**
```
https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/     (200)
https://ieee-dataport.org/documents/neu-det                        (200)
```

**GitHub mirrors (if HF is blocked)**
```
https://github.com/abin24/Surface-Inspection-defect-detection-dataset   (200)
https://github.com/Charmve/Surface-Defect-Detection                     (200)  ← index of datasets/papers
```

**☠ Dead — do not chase**
```
https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database   → 404
```
This URL circulates widely in tutorials. It is gone. Several blog posts still link it; that is 30–60 minutes of a hackathon wasted.

### 1.3 Download

```bash
pip install -U huggingface_hub

# Whole dataset (~120 MB)
hf download KeenForgeAI/NEU-DET-corrected --repo-type dataset \
  --local-dir data/neu-det-raw

# Or, from Python
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('KeenForgeAI/NEU-DET-corrected', repo_type='dataset',
                      local_dir='data/neu-det-raw', max_workers=8)
print(p)
"
```

```powershell
# Windows, no Python env yet
python -m pip install -U huggingface_hub
python -m huggingface_hub.commands.huggingface_cli download KeenForgeAI/NEU-DET-corrected --repo-type dataset --local-dir data/neu-det-raw
```

**Verify after download (must print 1797, 1797, 1797):**

```powershell
(Get-ChildItem data/neu-det-raw/images -Recurse -Filter *.jpg).Count
(Get-ChildItem data/neu-det-raw/labels -Recurse -Filter *.txt).Count
(Get-ChildItem data/neu-det-raw/annotations -Recurse -Filter *.xml).Count
```

### 1.4 Class order (verified from the shipped `data.yaml`)

```
0: crazing
1: inclusion
2: patches
3: pitted_surface
4: rolled-in_scale
5: scratches
```

**Do not reorder.** If your converter renumbers, you will silently swap `crazing` and `scratches` and lose a day.

### 1.5 The split you must build yourself

The shipped 80/10/10 split includes all 6 classes — it cannot be used for our demo, because **`rolled-in_scale` must be absent from training entirely**.

**Your split (write to `data/neu-det/splits.json`, commit it, never regenerate):**

| Split | Classes | Count | Derivation |
|---|---|---|---|
| Train | 5 (no class 4) | **1,048** | 70% of 1,497, stratified → ~209/class |
| Val | 5 | **225** | 15% → 45/class |
| Test | 5 | **224** | 15% → 45/class |
| **Held-out** | `rolled-in_scale` only | **300** | Never enters any training list |

1,497 = 1,797 − 300. Stratify per class; seed the split; hash the file lists.

**Hard guard — put this in the converter:**

```python
HELDOUT = {"rolled-in_scale"}
for split in ("train", "val", "test"):
    classes = {c for f in split_files[split] for c in read_yolo_classes(f)}
    assert not (classes & HELDOUT), f"LEAK: {classes & HELDOUT} in {split}"
```

Then `grep -r "rolled-in_scale" data/neu-det/labels/train | wc -l` → must be **0**.

### 1.6 What NEU-DET does *not* give you

- **No "good" class.** Every image contains a defect. You cannot train the novelty gate on NEU-DET.
- **No assembly defects.** It is surface texture only.
- **Large boxes.** Defects fill much of the 200×200 crop — so a NEU-DET model that scores 0.85 will still miss a 1 mm scratch on a 4000 px phone photo. This is exactly why the gate exists and why you must also train on your own captures.

---

## 2. Your own captures — the highest-value data in the project

Not downloadable. 45 minutes of work. Full protocol in `MASTER_PLAN.md` §7.2.

| What | Count | Use |
|---|---|---|
| Good parts, same lighting/angles | **40–60** | Memory bank (28), calibration (6), test (6) |
| Each defect class | **12–16** | Classifier probe + demo |
| Hard negatives (dust, smudges, labels, reflections) | **20** | Labeled **good** — cuts false alarms more than any model change |
| Severity-1 samples | 3–5 | Lets you say "we catch all sev ≥ 2, 60% of sev 1" |

**Physical defects you can make in 10 minutes:** scratch (sandpaper/blade), hairline crack (score + flex), dent (rounded punch), missing screw (bracket), burr (file).

**Naming:** `P014_crack_med_045_rakeL_02.jpg` — part, class, severity, angle, light, frame.

**Split by session and by physical part, never by image.**

---

## 3. MVTec AD — novelty gate benchmark

| Property | Value |
|---|---|
| Content | 15 categories, 5,354 images, pixel-accurate masks |
| Use here | Benchmark AUROC for the gate + fallback if your own capture is poor |
| Categories worth taking | `metal_nut`, `screw`, `transistor`, `bottle`, `carpet` |
| Size | **4.9 GB** full |
| License | **CC BY-NC-SA 4.0 — non-commercial, research only.** Say this out loud if asked |

**Direct links (verified 200):**
```
⭐ Mirror:  https://huggingface.co/datasets/foersben/mvtec-ad     (14,603 files confirmed)
Official:   https://www.mvtec.com/company/research/datasets/mvtec-ad
```

```bash
# Full 4.9 GB
hf download foersben/mvtec-ad --repo-type dataset --local-dir data/mvtec

# Only the four categories you need (much faster, ~1 GB)
hf download foersben/mvtec-ad --repo-type dataset --local-dir data/mvtec \
  --include "metal_nut/*" "screw/*" "transistor/*" "bottle/*"
```

**Required layout (the #1 setup gotcha):**

```
data/mvtec/metal_nut/
├── train/good/*.png
├── test/{bent,color,flip,scratch}/*.png
└── ground_truth/{bent,color,flip,scratch}/*_mask.png
```

---

## 4. Optional datasets (P2/P3 — slides only)

### 4.1 GC10-DET — metallic surface defects
```
https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets   (200)
```
10 classes, ~3,570 images, box labels. Use for the **cross-dataset generalization slide**: train on NEU, test on GC10, show the mAP drop, conclude "domain shift is the real problem." Downloads may route through Baidu; mirror to your own drive before the event.

### 4.2 VisA — second anomaly benchmark
```
Direct tar (1.93 GB, verified 200):
https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar

Index / code:
https://github.com/amazon-science/spot-diff
```
12 subsets, 10,821 images. Contains **4 PCB subsets** — useful if you pivot to electronics. License: CC BY 4.0 (dataset), Apache-2.0 (code).

### 4.3 DeepPCB — if you pivot to electronics
```
https://github.com/tangsanli5201/DeepPCB    (200)
```
1,500 image pairs, 640×640, 6 PCB defect types (open, short, mousebite, spur, copper, pin-hole).

### 4.4 MVTec AD 2 — credibility slide only
```
https://www.mvtec.com/company/research/datasets/mvtec-ad-2   (200, form-gated)
```
8 scenarios, 8,000+ images at 2.3–5 MP, lighting deliberately shifted between train and test. Do **not** make this a 48-hour target — the benchmark server allows 3 submissions per 7 days. Cite its 35–43 point domain-gap table instead.

### 4.5 MVTec LOCO AD — assembly/logical anomalies
```
https://www.mvtec.com/company/research/datasets/mvtec-loco-ad   (200)
```
5 categories, 3,644 images. The only public set that actually tests "missing part / wrong arrangement." Use to justify the rule-based assembly layer.

### 4.6 Real-IAD — scale story, do not download
```
https://realiad4ad.github.io/Real-IAD/                (200)
https://huggingface.co/datasets/Real-IAD/Real-IAD     (200)
```
30 objects, 150K images, **53 GB @1024² (507 GB raw)**. Cite the multi-view "sample-level" metric idea; do not spend bandwidth.

### 4.7 Others worth knowing
| Dataset | Link | Note |
|---|---|---|
| Magnetic Tile Defect | `github.com/abin24/Magnetic-tile-defect-datasets.` | Cracks + blowholes, multiple illumination conditions — closest to "crack on metal" |
| BTAD | `github.com/dataset-ninja/btad` | 3 products, 2,830 imgs; original site dead |
| DAGM 2007 | `kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection` | 10 texture classes, weak labels |
| KSDD2 | via `github.com/yyang13/DivCo` | Real production-line cracks, 3,334 imgs |
| Kaputt | `huggingface.co/datasets/amazon/kaputt` (200) | Retail-logistics defects; **CC BY-NC-ND** — most restrictive |

---

## 5. License matrix — read before you say "commercial"

| Dataset | License | Can you ship a product? | What to say on stage |
|---|---|---|---|
| NEU-DET | **None stated** | ❓ Unclear — treat as research | "Public research dataset; production use needs a licensing review" |
| MVTec AD / AD2 / LOCO | **CC BY-NC-SA 4.0** | ❌ No | "Non-commercial research license — we'd license or self-capture for production" |
| VisA | CC BY 4.0 | ✅ With attribution | "CC BY 4.0" |
| GC10-DET | Research | ❓ | "Research dataset" |
| Real-IAD | Research / non-commercial | ❌ | — |
| Kaputt | CC BY-NC-ND 4.0 | ❌ | — |
| **Your own captures** | **Yours** | ✅ | "Our data, our IP" |
| **Ultralytics YOLO11** | **AGPL-3.0** | ⚠️ Viral for closed source | "AGPL — fine for a pilot, we'd license or swap for production" |

**Saying the AGPL line unprompted is a credibility multiplier.** Most teams either don't know or hope nobody asks.

---

## 6. Disk & transfer budget

| Item | Size | Needed on the training PC? | Needed at the venue? |
|---|---|---|---|
| NEU-DET-corrected | ~120 MB | ✅ | ✅ |
| Own captures | ~200 MB | ✅ | ✅ |
| MVTec AD (4 categories only) | ~1 GB | ✅ | ❌ |
| MVTec AD (full) | 4.9 GB | optional | ❌ |
| GC10-DET | ~2 GB | optional | ❌ |
| VisA | 1.93 GB | optional | ❌ |
| Model weights (`.pt` + `.onnx` + `.engine`) | ~100 MB | ✅ | ✅ |
| **Total critical** | **~1.5 GB** | | **~350 MB** |

Copy the critical set to a USB stick. Do not rely on venue Wi-Fi for 120 MB at 9:55 AM.

---

## 7. One-shot setup script

Save as `scripts/get_data.ps1` on the new PC.

```powershell
$ErrorActionPreference = "Stop"
$root = "data"
New-Item -ItemType Directory -Force -Path $root | Out-Null

Write-Host "[1/4] NEU-DET (primary) ..."
python -m pip install -q -U huggingface_hub
python -m huggingface_hub.commands.huggingface_cli download `
  KeenForgeAI/NEU-DET-corrected --repo-type dataset --local-dir "$root/neu-det-raw"

Write-Host "[2/4] Verifying NEU-DET ..."
$img = (Get-ChildItem "$root/neu-det-raw/images" -Recurse -Filter *.jpg).Count
$lbl = (Get-ChildItem "$root/neu-det-raw/labels" -Recurse -Filter *.txt).Count
if ($img -ne 1797 -or $lbl -ne 1797) { throw "NEU-DET incomplete: $img images / $lbl labels" }
Write-Host "    OK: $img images, $lbl labels"

Write-Host "[3/4] MVTec AD (4 categories) ..."
python -m huggingface_hub.commands.huggingface_cli download `
  foersben/mvtec-ad --repo-type dataset --local-dir "$root/mvtec" `
  --include "metal_nut/*" "screw/*" "transistor/*" "bottle/*"

Write-Host "[4/4] Done. Next: python scripts/make_splits.py --holdout rolled-in_scale"
```

**Then, and only then**, build the split:

```bash
python scripts/make_splits.py --src data/neu-det-raw --dst data/neu-det \
  --holdout rolled-in_scale --ratios 0.70 0.15 0.15 --seed 42
# expect: train 1048 | val 225 | test 224 | heldout 300
```

---

## 8. Verification checklist

- [ ] `images` = 1,797 · `labels` = 1,797 · `annotations` = 1,797
- [ ] Class order matches the shipped `data.yaml` exactly (do not renumber)
- [ ] `splits.json` committed; counts 1,048 / 225 / 224 / 300
- [ ] `grep -r "rolled-in_scale" data/neu-det/labels/{train,val,test}` → **0 hits**
- [ ] 20 random labels rendered over their images — boxes actually on the defects
- [ ] No near-duplicate images across splits (perceptual hash)
- [ ] MVTec layout is `train/good`, `test/<defect>`, `ground_truth/<defect>/*_mask.png`
- [ ] Dataset path in YAML is **relative**, not `E:\...` (see `SETUP_OTHER_PC.md` §3)
- [ ] Everything under `data/` is in `.gitignore`; `splits.json` and the manifests **are** committed

---

## 9. If a link dies at the event

| Dataset | Plan B |
|---|---|
| NEU-DET | GitHub `abin24` mirror → IEEE Dataport → **shoot 60 photos of printed defect images** (40 min, always works) |
| MVTec AD | Official form → any `mvtec` mirror on HF → skip the benchmark and report own-capture AUROC only |
| Own captures | They are on your phone and your USB stick. This is why they are P0 |

**The universal fallback:** print 20 defect images on matte paper, photograph them with the demo camera, label 5 boxes by hand. That is a complete, honest, demoable dataset in under an hour — and it is domain-matched, which the public sets are not.
