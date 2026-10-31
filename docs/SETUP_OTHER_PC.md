# SETUP & HANDOFF — building on a different PC

> **Why this exists:** all planning was done on one machine (`E:\Indore`, CPU-only torch 2.6.0). The build happens on a **different machine with an RTX 4060 8 GB**. Drive letters, usernames, CUDA builds and paths will differ. This document makes the project portable and the handoff boring.
> **Rule:** every command in every doc assumes **you are at the repo root**. No absolute paths anywhere in code or config.

---

## 0. The five-minute sanity check

Run this before anything else on the new PC. All five must pass or the next three hours are wasted.

```powershell
python --version                     # 3.11 or 3.12  (NOT 3.13)
nvidia-smi                           # shows RTX 4060 Laptop, driver + CUDA version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory/1e9, 'GB VRAM')"
python -c "import cv2, onnxruntime; print(cv2.__version__, onnxruntime.__version__)"
```

Expected: `True`, `NVIDIA GeForce RTX 4060 Laptop GPU`, `~8.0 GB VRAM`.

**If `torch.cuda.is_available()` is False, stop.** Do not "try training anyway" — you will silently train on CPU at 1/40th the speed and blame the model.

---

## 1. Target machine profile

| Item | Expected | If different |
|---|---|---|
| GPU | RTX 4060 Laptop, 8 GB | 6 GB → drop to `batch=8`, `imgsz=512` |
| RAM | 32 GB | 16 GB → `cache=False`, `workers=2` |
| OS | Windows 11 | WSL2 is fine for training; keep the demo native |
| Disk free | ≥ 25 GB | 10 GB → skip full MVTec, take 4 categories only |
| Python | 3.11 / 3.12 | 3.13 → install 3.12 side-by-side |

---

## 2. Install — exact order, pinned

Order matters. `torch` **before** `ultralytics`, or pip resolves a CPU wheel and you lose the GPU silently.

```powershell
# 1. Virtual environment (never train in the system Python)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools

# 2. Torch with CUDA 12.4 — the single most important line in this document
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Verify BEFORE installing anything else
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# must print: 2.6.0+cu124 True

# 4. Everything else
pip install ultralytics==8.3.0
pip install opencv-python numpy pandas scikit-learn matplotlib seaborn pyyaml tqdm
pip install onnx onnxruntime onnxruntime-gpu
pip install huggingface_hub

# 5. Optional, LAST, and only after the demo works
# pip install tensorrt            # 20–120 min on Windows, high variance
# pip install openvino            # only if you demo on Intel CPU
```

**Version pin rationale**

| Package | Pin | Why |
|---|---|---|
| torch | 2.6.0+cu124 | Matches the cu124 wheels; Ultralytics 8.3 is tested against it |
| ultralytics | 8.3.0 | Pinned — the API shifts between minors and tutorials break |
| onnxruntime-gpu | any recent | CUDA EP for the GPU path; plain `onnxruntime` is the CPU fallback |
| huggingface_hub | ≥0.27 | `hf download` CLI |

**Do not `pip install ultralytics` unpinned on the demo machine at 10 PM.** It auto-upgrades torch and can break the CUDA build.

---

## 3. Path strategy — the thing that breaks on a new PC

The plan documents were written on `E:\Indore`. **The new PC will not have an `E:\Indore`.** If any config contains that path, training fails with a confusing "dataset not found."

### Rules

1. **YAML uses relative paths only.**

```yaml
# ✅ portable
path: ../data/neu-det
train: images/train
val: images/val
test: images/test

# ❌ breaks on the other machine
path: E:/Indore/data/neu-det
```

2. **Python resolves the root once, in one place.**

```python
# src/paths.py  — the ONLY place a root is computed
from pathlib import Path
import os

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "weights"
EVIDENCE = ROOT / "data" / "evidence"

for p in (DATA, RUNS, WEIGHTS, EVIDENCE):
    p.mkdir(parents=True, exist_ok=True)
```

3. **Launch scripts derive the root from their own location.**

```powershell
# scripts/demo.ps1
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
```

4. **Never use `cd` in a notebook or a README command.** Use `-m` modules from the root.

5. **Environment file, not hardcoded strings.**

```powershell
# .env  (gitignored)
PROJECT_ROOT=D:\work\qa-cell
SOURCE=camera
ENGINE=onnx
IMGSZ=640
```

### Migration checklist for existing files

| File | What to change |
|---|---|
| `data/neu-det/neu-det.yaml` | `path:` → relative (`../neu-det`) |
| `scripts/*.ps1` | derive `$ROOT` from `$PSScriptRoot` |
| Any notebook | replace absolute strings with `PROJECT_ROOT` |
| `README.md` quickstart | `git clone` → `cd` → `demo.ps1`, no drive letters |

---

## 4. Moving the project to the new PC

### 4.1 What to move (and what not to)

| Move | How | Why |
|---|---|---|
| Code, docs, configs, `splits.json` | **git** (or `git bundle`) | Small, versioned |
| Trained weights (`.pt`, `.onnx`) | git-lfs **or** USB/cloud | <100 MB; too big for plain git history |
| Datasets | re-download on the new PC, **or** USB | 1.5 GB critical set |
| `runs/` (logs, curves) | zip and keep | Needed for the results slide |
| `.venv/`, `node_modules/`, `__pycache__` | **never** | Platform-specific, huge |
| `data/evidence/` | **never** | Regenerable runtime output |

### 4.2 Method A — git (preferred)

```powershell
# old machine
git add -A
git commit -m "plan + docs + training spec"
git remote add origin <your-repo-url>
git push -u origin main

# new machine
git clone <your-repo-url> qa-cell
cd qa-cell
```

### 4.3 Method B — git bundle (no network at the venue)

```powershell
# old machine
git bundle create qa-cell.bundle --all

# new machine (bundle is a single file on a USB stick)
git clone qa-cell.bundle qa-cell
cd qa-cell
git remote set-url origin <your-repo-url>   # optional, when the network returns
```

### 4.4 Method C — zip (fastest, no history)

```powershell
# old machine
Compress-Archive -Path E:\Indore\* -DestinationPath qa-cell.zip `
  -Exclude '*.venv*','*node_modules*','*__pycache__*','*runs*','*data/evidence*'
```

Then unzip anywhere. Verify with §6.

---

## 5. Getting the data on the new PC

**Option 1 — re-download (recommended, 5 min on decent Wi-Fi).** Run `scripts/get_data.ps1` from `DATASETS.md` §7.

**Option 2 — copy from USB.** Keep this exact layout so no config changes are needed:

```
qa-cell/
└── data/
    ├── neu-det-raw/        # 1,797 img + 1,797 lbl + 1,797 xml
    ├── neu-det/            # built splits + neu-det.yaml
    ├── mvtec/              # only if you need the benchmark
    └── own/                # captures
```

**Option 3 — copy only the built splits** (if you already ran `make_splits.py`). Then `data/neu-det/` alone is enough for training; skip `neu-det-raw`.

After copying, **rebuild the split** if the seed or ratio changed:

```bash
python scripts/make_splits.py --src data/neu-det-raw --dst data/neu-det \
  --holdout rolled-in_scale --ratios 0.70 0.15 0.15 --seed 42
```

---

## 6. Verification checklist — run all 12 before training

| # | Command | Expected |
|---|---|---|
| 1 | `python -c "import torch;print(torch.cuda.is_available())"` | `True` |
| 2 | `python -c "import torch;print(torch.cuda.get_device_name(0))"` | `RTX 4060 Laptop GPU` |
| 3 | `nvidia-smi` | < 1 GB used before training |
| 4 | `(Get-ChildItem data/neu-det-raw/images -Recurse -Filter *.jpg).Count` | `1797` |
| 5 | `(Get-ChildItem data/neu-det/labels/train -Filter *.txt).Count` | `1048` |
| 6 | `Select-String -Path data/neu-det/labels/train/*.txt -Pattern '^4 '` | **no output** (class 4 = held out) |
| 7 | `yolo checks` | CUDA + ultralytics versions listed |
| 8 | 10-epoch smoke run | mAP50 > 0.15, GPU util > 70% |
| 9 | `python -c "import cv2;print(cv2.VideoCapture(0).isOpened())"` | `True` (camera) |
| 10 | `python -m pytest -q` or `scripts/smoke_test.py` | passes |
| 11 | `nvidia-smi` **after** a run | no OOM, < 7.5 GB peak |
| 12 | `git status` | clean, no `data/`, no weights > 50 MB |

**Check 6 is the one that kills the demo silently.** If it prints anything, `rolled-in_scale` leaked into training — stop and rebuild the split.

---

## 7. Windows-specific traps

| Trap | Symptom | Fix |
|---|---|---|
| **Path length > 260 chars** | `FileNotFoundError` on a path that exists | Enable long paths: `git config --system core.longpaths true` + registry `LongPathsEnabled=1` |
| **OneDrive syncs the repo** | Training writes to `runs/` → sync storm → disk thrash | Keep the repo **outside** OneDrive/Documents |
| **Defender scans every epoch write** | Training 2–3× slower, fans at max | Exclude the repo folder from real-time scanning for the event |
| **`workers=8` deadlock** | Training hangs at epoch 0 with no error | `workers=4`, `persistent_workers=True` |
| **PowerShell execution policy** | `demo.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| **Camera locked by another app** | `VideoCapture(0)` returns False | Close Zoom/Teams/Camera app; or use the phone-as-IP-camera path |
| **Docker Desktop + WSL2 + GPU** | nvidia-smi works on host, not in container | `--gpus all` + WSL2 backend + recent driver; do not debug this before H30 |
| **`E:\` does not exist** | dataset path errors | See §3 — relative paths only |
| **Non-ASCII username** | some C++ toolchains fail on paths with spaces/unicode | Put the repo at `D:\work\qa-cell` |

---

## 8. CUDA / torch / driver mismatch matrix

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` False, `nvidia-smi` works | CPU wheel installed | Reinstall from the `cu124` index URL |
| `CUDA error: no kernel image is available` | torch built for a different arch | Use a cu12x wheel matching the driver |
| `CUDA driver version is insufficient` | Driver too old | Update the NVIDIA driver (a driver update, not a CUDA toolkit install) |
| `OOM` at batch 16 | 8 GB ceiling | batch 8 + `nbs=32`; or `imgsz=512` |
| Works in Python, fails in notebook | Kernel bound to a different venv | Select the `.venv` kernel explicitly |
| `cudnn` DLL load failure | Multiple CUDA installs on `PATH` | One CUDA version; let torch ship its own DLLs |

**You do not need to install the CUDA Toolkit.** The pip wheel ships the runtime. Installing the toolkit separately is how people break a working environment.

---

## 9. Two-machine workflow

| Artifact | Commit? | Sync direction |
|---|---|---|
| `docs/`, `src/`, `scripts/` | ✅ | both ways, via git |
| `data/neu-det/splits.json` | ✅ | **one-way: planning PC → training PC** |
| `data/**` images | ❌ (gitignore) | USB / re-download |
| `weights/*.onnx` | ✅ (<50 MB) or LFS | training PC → demo PC |
| `runs/**/results.csv`, `results.png` | ✅ (small) | training PC → demo PC |
| `runs/**/weights/*.pt` | ⚠️ LFS or USB | training PC → demo PC |
| `.env` | ❌ | per machine |

**Golden rule:** the **demo PC never trains**. It receives `best.onnx` + `splits.json` + `results.csv`. Training and demoing on the same machine at the same time is how you get a 4 fps demo.

**If you have exactly one machine:** train at hours 8–20, export, then **close every training process** before the demo. Verify `nvidia-smi` shows < 1 GB used.

---

## 10. Offline / venue bundle

Build this at **H38**, not at the venue.

```
usb/
├── qa-cell.bundle              # git bundle --all
├── weights/
│   ├── best.pt
│   ├── best.onnx
│   └── best.engine             # optional, if TRT worked
├── data/
│   ├── neu-det/                # built splits only (~120 MB)
│   └── own/                    # captures
├── wheels/                     # pip download -r requirements.txt -d wheels
├── demo_fallback.mp4
├── slides.pdf
├── slides.pptx
└── RESTORE.md                  # 5 numbered steps, written for a stranger
```

```powershell
# Pre-download every wheel so a dead venue network cannot stop you
pip download -r requirements.txt -d wheels --platform win_amd64 `
  --python-version 3.12 --only-binary=:all:

# Install offline later
pip install --no-index --find-links wheels -r requirements.txt
```

**Also vendor the frontend:** commit `web/dist/` and `package-lock.json`. If `npm install` fails at the venue, serve the pre-built `dist/` and never touch Node again.

---

## 11. Reproducibility manifest

Write this on the training PC and commit it as `docs/ENV.md`.

```yaml
machine:        RTX 4060 Laptop 8GB / 32GB RAM / Windows 11
python:         3.12.x
torch:          2.6.0+cu124
ultralytics:    8.3.0
cuda_driver:    <from nvidia-smi>
onnxruntime:    1.20.x
opencv:         4.11.0
dataset:        KeenForgeAI/NEU-DET-corrected @ <commit sha>
split_seed:     42
model:          yolo11s.pt (COCO pretrained)
train_seed:     0
command:        <the exact yolo CLI line>
weights_sha256: <sha256 of best.onnx>
```

Regenerate with:

```bash
python -c "import torch,ultralytics,cv2,onnxruntime as o; print(torch.__version__, ultralytics.__version__, cv2.__version__, o.__version__)"
python -c "import hashlib;print(hashlib.sha256(open('weights/best.onnx','rb').read()).hexdigest())"
```

---

## 12. Day-1 checklist on the new PC

```
[ ] Unzip/clone to a short path (D:\work\qa-cell), NOT under OneDrive
[ ] py -3.12 -m venv .venv && activate
[ ] Install torch cu124 FIRST, verify True
[ ] Install ultralytics 8.3.0 + the rest
[ ] Defender exclusion for the repo folder
[ ] git config --system core.longpaths true
[ ] Run scripts/get_data.ps1 (or copy data from USB)
[ ] Run scripts/make_splits.py --holdout rolled-in_scale --seed 42
[ ] Verify 1,797 / 1,048 / 225 / 224 / 300
[ ] Grep for class 4 in train labels → must be empty
[ ] 10-epoch YOLO11n smoke run → mAP > 0.15
[ ] nvidia-smi peak < 7.5 GB
[ ] Camera opens
[ ] Commit the ENV.md manifest
```

When all 14 are green, start `TRAINING_SPEC.md` Stage 1. Not before.
