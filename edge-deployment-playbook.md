# Edge Deployment & Inference Optimization Playbook
### 36–48 h industrial defect-detection hackathon · numbers first, demos second

**Scope:** real-time CV defect detection (classification / detection / anomaly) on a manufacturing line, with an optional edge-deployment differentiator.
**Assumption:** 1–4 people, a laptop (maybe CUDA), possibly one SBC/NPU. Everything below is chosen so that the *worst case* still demos.

---

## 0. TL;DR — what to actually do

| Decision | Recommendation | Why |
|---|---|---|
| Primary model | **EfficientAD-S @ 256×256** for "only good parts available"; **YOLO-seg-n @ 640** if you have defect bounding boxes/masks | EfficientAD = 2 ms, 600 img/s on a modern GPU (paper), no memory bank, no threshold-free kNN |
| Training budget | ≤ 6 h. Anomaly models train in 5–30 min/class on one GPU | Leaves 30+ h for the demo |
| Default runtime | **ONNX Runtime INT8** on the laptop CPU **or** **TensorRT FP16** on the laptop GPU | 3–6× and 20–60× over PyTorch eager respectively, ~1 h of work |
| Edge differentiator (safe) | **Pi 5 + Camera Module 3 running an INT8 ONNX classifier at 224 px, MJPEG to laptop dashboard** | 4 h, ~₹12k, no NPU toolchain, runs at 60+ FPS |
| Edge differentiator (wow) | **Pi 5 + AI HAT+ (Hailo-8/8L)** — only if you can compile the HEF on day 1 | Measured: YOLO26n 21.7 ms on Pi 5 + Hailo-8 (Ultralytics) |
| Avoid in 48 h | Coral USB, Qualcomm RB5/QCS6490, Docker on ARM, TRT INT8 calibration on Jetson, DFC on Windows | Toolchain tax > demo value |
| The number to put on the slide | **p95 end-to-end latency**, not mean FPS | Judges remember "38 ms p95, 26 FPS sustained for 10 min", not "120 FPS burst" |

**One-line thesis for judges:** *"Inspection latency is not the bottleneck — sensor→PLC latency and jitter are. We spent our optimization budget where it changes the reject decision."*

---

## 1. Latency budget math

### 1.1 From line speed to milliseconds per part

```
T_cycle [ms] = 60 000 / ppm
ppm          = (v / pitch) × 60        v = belt speed [m/s], pitch = part spacing [m]
t_dwell [ms] = (L_fov / v) × 1000      L_fov = field of view length along the belt [m]
N_frames     = t_dwell × camera_fps / 1000
```

| Line | v | pitch | ppm | T_cycle | Frames @30 fps |
|---|---|---|---|---|---|
| Slow assembly | 0.2 m/s | 0.40 m | 30 | **2000 ms** | 60 |
| Typical | 1.0 m/s | 0.30 m | 200 | **300 ms** | 9 |
| Fast | 1.0 m/s | 0.20 m | 300 | **200 ms** | 6 |
| Very fast | 2.0 m/s | 0.15 m | 800 | **75 ms** | 2.2 |
| Blister/foil | 3.0 m/s | 0.05 m | 3600 | **16.7 ms** | 0.5 ✗ |

Worked example — **300 ppm**:
`60 000 / 300 = 200 ms` per part. With a 0.40 m FOV at 1 m/s, `t_dwell = 400 ms` → you get ~12 frames at 30 fps, so you can afford a 3-frame vote (median/OR) and still have 300 ms of slack.

Worked example — **800 ppm**: `60 000/800 = 75 ms`; at 2 m/s a 0.30 m FOV gives `150 ms` dwell. A 30 fps camera sees 4.5 frames. Anything slower than ~30 ms inference forces *one frame per part* + a trigger.

### 1.2 The end-to-end chain (this is the slide)

```
trigger ─► exposure ─► readout ─► transfer ─► decode ─► preprocess ─► infer ─► NMS/decode ─► PLC out ─► valve ─► air/mech
```

| Stage | Laptop (good pipeline) | Pi 5 (typical) | Notes |
|---|---|---|---|
| Sensor → camera trigger | 1–5 ms | 1–5 ms | hardware trigger, not software polling |
| Exposure | 0.05–2 ms | same | **must be ≤ 0.2 ms at 1 m/s to avoid blur** |
| Readout + transfer | 3–10 ms | 8–20 ms | USB3/GigE vs USB2; MJPEG vs raw |
| Decode | 2–8 ms (1080p MJPEG) | 15–30 ms | libjpeg-turbo; hardware decoder if H.264 |
| Preprocess (resize+norm) | 1–3 ms | 5–12 ms | cv2 resize releases the GIL |
| **Inference** | 0.5–8 ms (GPU) / 6–30 ms (CPU) | 8–45 ms | the part everyone optimizes |
| Post-process | 0.5–4 ms | 2–8 ms | NMS / seg argmax / anomaly threshold |
| Decision + digital out | 0.5–2 ms | 1–8 ms | GPIO / Modbus TCP / EtherNet/IP |
| **Subtotal electronics** | **10–30 ms** | **40–90 ms** | |
| Valve + mechanics | 10–40 ms | same | pneumatic solenoid 5–15 ms + cylinder |
| **Total** | **25–70 ms** | **55–130 ms** | |

### 1.3 How much latency is *actually* required

The binding constraint is usually **the reject actuator**, not the network:

```
t_available = d_rejector / v − t_actuation − margin
```

- Rejector 0.5 m downstream, 1 m/s → 500 ms available; actuation 40 ms; 20 % margin → **360 ms usable**. A 25 ms inference uses 7 % of the budget.
- Diverter *at* the inspection station, 150 ms dwell → `150 − 40 = 110 ms`, and you must fire mid-dwell → **call it 80 ms**.
- Closed-loop servo correction (robot pick) → **< 30 ms** or you need to predict forward by one cycle.

| Use case | Required e2e | Reality check |
|---|---|---|
| Human-in-the-loop review station, 60 ppm | < 500 ms | trivial |
| Sorting gate, 300 ppm | < 200 ms | GPU or INT8 CPU both fine |
| In-station diverter, 800 ppm | < 50 ms | needs trigger + 1 frame + GPU |
| "Feels live" to a judge | < 100 ms | 30 fps display = 33 ms/frame anyway |

**Motion-blur arithmetic (the trap nobody budgets for):**
`blur_px = v [mm/s] × t_exp [s] / mm_per_px`
At 1 m/s and 0.2 mm/px: `t_exp ≤ (1 px × 0.2 mm) / 1000 mm/s = 200 µs`. At 200 µs you need ~50–100 klux on target → a strobe or a very bright bar light. **A 30 fps webcam with auto-exposure at 1/30 s will produce 5–10 px of blur and no model will save you.** Lock exposure (`CAP_PROP_AUTO_EXPOSURE=0.25` / V4L2 `exposure_time_absolute`) and light the part properly — this is worth more accuracy than any quantization decision.

---

## 2. The optimization ladder

### 2.1 What each rung actually buys (yolo26n, 640×640, batch 1)

| Rung | Where | Latency | Speedup | Effort | Risk |
|---|---|---|---|---|---|
| PyTorch eager FP32 | Core Ultra 7 155H CPU | 33.2 ms | 1.0× | 0 | none |
| `torch.compile` | same | ~24–28 ms | 1.2–1.4× | 20 min | graph breaks, first-call compile 30–120 s |
| ONNX + ORT FP32 | same | ~18–22 ms | 1.5–1.8× | 30 min | export op gaps |
| ONNX + ORT **INT8 QDQ** | same | ~8–11 ms | 3–4× | 1–2 h | calibration + accuracy |
| OpenVINO FP16 | same | **9.07 ms** | 3.7× | 30 min | Intel only |
| OpenVINO **INT8** | same | **5.79 ms** | **5.7×** | 1 h | calibration |
| TensorRT FP32 | RTX 3080 | 1.06 ms | — | 45 min | engine is device-locked |
| TensorRT FP16 | RTX 3080 | **0.62 ms** | — | 45 min | none |
| TensorRT INT8 | RTX 3080 | **0.52 ms** | — | 2–3 h | calibration, mAP −0.04 |
| TensorRT FP16/INT8 | Jetson Orin NX 16 GB | 3.18 / **2.30 ms** | — | 1 h (on device) | build on device, 2–15 min |

*Sources: Ultralytics OpenVINO + TensorRT benchmark pages (openvino 2026.2.1; TRT 10.0.1 on JetPack 6.0), measured with yolo26n/yolov8n, COCO128.*

**Read the table this way:** on CPU you get ~5× for one afternoon; on GPU you get ~20× for 45 minutes (FP16) and ~2× more than that for another 2 hours (INT8). **Stop at FP16 on GPU and INT8 on CPU.** That is the 80/20.

### 2.2 Rung 1 — ONNX export

```python
# ultralytics >= 8.3 (2026 API: quantize= replaces half=/int8=)
from ultralytics import YOLO
m = YOLO("runs/segment/train/weights/best.pt")
m.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, batch=1)
# -> best.onnx
```

Raw torch (for custom anomaly models):

```python
import torch
model.eval().cuda()
dummy = torch.randn(1, 3, 256, 256, device="cuda")
torch.onnx.export(
    model, dummy, "efficientad.onnx",
    opset_version=17, do_constant_folding=True,
    input_names=["images"], output_names=["anom_map"],
    dynamic_axes=None,          # static shapes = fastest, and TRT/OpenVINO prefer it
)
```

Verify before you trust it:

```bash
python -m onnxruntime.tools.check_onnx_model efficientad.onnx   # or onnx.checker.check_model
python -m onnxsim best.onnx best.sim.onnx                      # if simplify failed
```

**Ops that break export (seen in the wild):**

| Op / pattern | Symptom | Fix |
|---|---|---|
| `torch.nonzero`, `torch.unique`, boolean masking | `UnsupportedOperatorError` / tracer warning | move to post-processing (host) |
| `if x.sum() > 0:` data-dependent control flow | graph silently truncated | remove from `forward` |
| `F.interpolate(size=other.shape[2:])` + `dynamic=True` | shape inference error | hard-code size, or `scale_factor=2` |
| `topk` with tensor `k` | dynamic `k` unsupported | constant `k` |
| Custom CUDA op / DeformConv / `grid_sample` 5-D | no ONNX op | TRT plugin or drop the layer |
| `MultiheadAttention` fast path | trace fails on opset < 14 | opset ≥ 17, or `torch.nn.attention.sdpa_kernel` |
| In-place `index_put_`, `.item()` in forward | wrong values after export | clone / return tensors |
| `argmax` / NMS inside the graph | INT8 crushes it; TRT needs plugins | export raw logits, post-process in Python/C++ |
| **PatchCore kNN / coreset** | not exportable at all | export **only** the feature extractor |

### 2.3 Rung 2 — ONNX Runtime INT8 (static, QDQ)

```python
import onnx, numpy as np, cv2
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat, QuantType)

class Reader(CalibrationDataReader):
    def __init__(self, paths, size=640):
        self.it = iter(paths); self.size = size
    def get_next(self):
        p = next(self.it, None)
        if p is None: return None
        img = cv2.resize(cv2.imread(p), (self.size, self.size))[:, :, ::-1]
        x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return {"images": np.ascontiguousarray(x)}

model = "best.onnx"
# Keep the head in FP32: box coords (pixels) and class scores (0..1) share output0,
# a single INT8 scale rounds every score to zero.
exclude = [n.name for n in onnx.load(model).graph.node
           if n.op_type not in {"Conv", "Gemm", "MatMul"}]

quantize_static(
    model, "best.int8.onnx", Reader(calib_paths, 640),
    quant_format=QuantFormat.QDQ,      # QDQ = portable (OpenVINO/TRT can ingest it)
    per_channel=True, reduce_range=False,
    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
    nodes_to_exclude=exclude,
)
```

- **Calibration set:** 100–300 images *from your camera, your lighting*. In-domain > big out-of-domain (Hailo measured: 1,238 out-of-domain images gave the same accuracy as no fine-tuning at all).
- **Quick win first:** `quantize_dynamic()` (weights only, no calibration) gives ~1.7–2.2× and ~0 accuracy loss. If you're short on time, stop there.
- **Never reuse the float threshold** for a quantized anomaly model — quantization noise inflates the anomaly score baseline. Recompute the threshold on 50 good parts (p99 of scores).

Runtime knobs that matter more than people think:

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.intra_op_num_threads = 4        # sweep 1/2/4/8 — on hybrid CPUs 4 usually wins
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.enable_mem_pattern = True
sess = ort.InferenceSession("best.int8.onnx", so, providers=["CPUExecutionProvider"])
```

### 2.4 Rung 3 — OpenVINO (the best CPU deal on Intel)

```bash
pip install "openvino==2026.2.1" nncf
# convert (FP16 by default)
ovc best.onnx --input "[1,3,640,640]" -o ov_model/ --compress_to_fp16
# or straight from Ultralytics:
# model.export(format="openvino", imgsz=640, quantize=8, data="dataset.yaml")
```

INT8 with NNCF PTQ (better than raw ORT on Intel):

```python
import openvino as ov, nncf, numpy as np
core = ov.Core()
model = core.read_model("ov_model/best.xml")

def transform(x):                       # x: uint8 HWC BGR
    x = np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))[None]
    return np.float32(x) / 255.0

calib = nncf.Dataset(list_of_uint8_images, transform)
q = nncf.quantize(model, calib, subset_size=300,
                  preset=nncf.QuantizationPreset.MIXED)   # MIXED = symmetric weights, asymmetric acts
ov.save_model(q, "ov_model/best_int8.xml")
```

Measure, don't guess:

```bash
benchmark_app -m ov_model/best.xml -d CPU -api async -niter 1000 -hint latency
benchmark_app -m ov_model/best.xml -d GPU.0 -api async -niter 1000
benchmark_app -m ov_model/best_int8.xml -d NPU -api async -niter 1000   # Core Ultra only
```

`-hint latency` pins 1 stream (best p95); `-hint throughput` uses all cores (best FPS, worse p99). **For a reject decision you want latency.**

### 2.5 Rung 4 — TensorRT (GPU, 45 min for FP16)

```bash
# The engine is built FOR the exact GPU + TRT version. Build where you deploy.
trtexec --onnx=best.onnx --saveEngine=best.fp16.engine --fp16 \
        --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 \
        --maxShapes=images:1x3x640x640 --workspace=4096 --useCudaGraph \
        --warmUp=500 --duration=10 --avgRuns=100
```

INT8 (usually not worth it in 48 h — gains 0.62 → 0.52 ms on a 3080):

```bash
# polygraphy builds the calibration cache in one command
polygraphy convert best.onnx --int8 \
  --calibration-data calib/ --calibration-cache best.int8.cache \
  -o best.int8.engine
# or let Ultralytics do it:
# m.export(format="engine", quantize=8, data="dataset.yaml", workspace=4)
```

Jetson specifics:

```bash
sudo nvpmodel -m 0          # MAXN
sudo jetson_clocks
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --fp16 --saveEngine=best.engine
tegrastats --interval 500   # RAM, GPU util, temp, power
```

> **Version trap:** as of JetPack 6.x, Jetson ships **TensorRT 10.x** (DLA supported through 10.7). TensorRT 11 **does not support JetPack** and removed implicit INT8 quantization (it uses ModelOpt explicit Q/DQ). Pin `tensorrt==10.x` on Jetson; don't let pip upgrade it.

### 2.6 Rung 5 — the NPU / mobile endpoints

**Hailo (Pi 5 AI HAT+)** — compile on x86 Linux only (WSL2 works), deploy the `.hef`:

```bash
yolo export model=best.pt format=hailo name=hailo8l imgsz=640 data=dataset.yaml
hailortcli fw-control identify
hailortcli parse-hef best.hef
hailortcli run best.hef --frames 2000     # device-only throughput, no host code
```

**NCNN (Pi 5, CPU-only, best bang for zero hardware):**

```bash
yolo export model=best.pt format=ncnn imgsz=320 half=True
# C++ side: ncnn::Net, opt.num_threads = 4, opt.use_fp16_packed = true
```

**TFLite INT8 (Coral, phone):** full-integer quantization + representative dataset; EdgeTPU additionally requires ops to be mapped, no `argmax`/`NMS` in-graph, and **no segmentation head** in practice.

**Roofline sanity check before you buy hardware:**
`t_min ≈ max(GFLOPs / peak_throughput, bytes_moved / bandwidth)`
yolo26n ≈ 6.4 GFLOPs @640. On a Pi 5 CPU (~50 GFLOPS fp32 effective) → ≥ 128 ms theoretical, which is why the measured value is 220–280 ms. On Hailo-8L (13 TOPS INT8, ~40 % utilization) → ~1.2 ms compute; measured 35–45 ms is dominated by host pre/post + memory. **If your measured number is 10× the roofline, the model isn't the problem — the pipeline is.**

---

## 3. Hardware comparison (late-2025 / 2026 street prices)

Reference workloads, all batch 1, static shapes:

- **W1** = YOLO-seg-n @ 640 (detection numbers shown; seg ≈ 1.8–2.2× the cost)
- **W2** = EfficientAD-S @ 256 (teacher+student+AE)
- **W3** = PatchCore (WRN50 @ 224 feature extractor + kNN over 1 % coreset)

| Device | Price (≈) | Peak INT8 | W1 det @640 (model-only) | W1 + pipeline | W2 @256 | W3 | Power | Setup in 48 h | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **RTX 4060 laptop** (8 GB, 115 W) | bundled (~$1000) | ~240 TOPS sparse | 1.2–1.6 ms FP16 | 250–450 FPS | 1.5–2.5 ms | 4 ms + kNN 2 ms | 60–120 W | **1 h** | **Best default.** If you own it, use it. TRT FP16 + done. |
| **RTX 4090 laptop** | bundled | ~680 TOPS sparse | 0.7–0.9 ms | 400+ FPS | ~1 ms | 2 ms | 120–175 W | 1 h | Overkill for 1 line, great for 4-camera demo |
| **Apple M3/M4** | bundled | 38 TOPS ANE | 4–8 ms CoreML FP16 | 90–150 FPS | 3–4 ms | 8 ms + kNN 4 ms | 20–35 W | 2 h | Excellent perf/W; ANE silently falls back to GPU — check with `coremltools` compute-unit report |
| **Jetson Orin Nano Super 8 GB** | **$249** (₹22–26k) | **67 TOPS** sparse | ~3.4 ms INT8 (derived: NX 2.30 ms × 100/67) | 110–150 FPS | ~4 ms | ~10 ms | **7–25 W** | 4–6 h (flash, build engine, pin cuDNN) | **Best "real edge" story.** 8.7 FPS/W at 15 W |
| **Jetson Orin NX 16 GB** | $599 module + carrier | 100 TOPS | **2.30 ms INT8 / 3.18 FP16** (measured) | 150–200 FPS | ~3 ms | ~8 ms | 10–25 W | 5 h | Same toolchain as Nano, more headroom for multi-cam |
| **Pi 5 8 GB + AI HAT+ (Hailo-8L, 13 TOPS)** | $80 + **$70** (₹13–15k) | 13 TOPS | 35–45 ms (Hailo-8 26 TOPS measured **21.7 ms**) | 20–27 FPS | not practical (DFC + EfficientNet teacher) | ✗ | 8–12 W | 5–8 h (x86 Linux DFC, HEF compile) | Great demo, real risk. **Only with Hailo-8 (26 TOPS, +$40) is it worth it** |
| **Pi 5 8 GB + AI HAT+ 2 (Hailo-10H)** | $80 + ~$110 | 40 TOPS | 12–20 ms (DFC 5.x) | 40–60 FPS | ✗ | ✗ | 10–14 W | 8 h | Newer, less documented — avoid for a hackathon |
| **Pi 5 8 GB, CPU only** | $80 (₹8–9k) | ~0.05 TOPS | 220–280 ms (NCNN @640) · 60–90 ms @320 | 4–12 FPS | 90–140 ms | 300 ms+ | 6–9 W | **1 h** | Fine for **classification @224** (5–8 ms), not for 640 detection |
| **Intel N100 mini-PC** | $150–200 (₹14–18k) | ~1 TOPS (iGPU) | 25–35 ms OpenVINO INT8 @640 · 8–12 ms @320 | 25–35 FPS @320 | 12–18 ms | 60–80 ms | 6–15 W | **1.5 h** | Cheapest "x86 + OpenVINO" demo; fanless, silent, boring in a good way |
| **Intel N305** | $250–300 | ~2 TOPS | 12–18 ms INT8 @640 | 50–60 FPS | 8–12 ms | 35–50 ms | 15–25 W | 1.5 h | Best $/FPS on x86 |
| **Core Ultra 7 155H laptop** | bundled | NPU 11 TOPS | **5.79 ms INT8 (measured)** | 120–160 FPS | ~8 ms | ~25 ms | 28–45 W | 1 h | If this is your laptop, you're already done |
| **Google Coral USB** | $70 (₹8k) | 4 TOPS | 12–20 ms @320, **head must be replaced** | 30–45 FPS | ✗ | ✗ | 2 W | 6 h+ (ops hell) | **Avoid.** Stale toolchain, no seg, INT8-only ops |
| **Qualcomm RB3 Gen 2 / QCS6490** | $250–400 | 12.5 TOPS | 15–25 ms via QNN/AI Hub | 30–40 FPS | ✗ | ✗ | 5–15 W | 12 h+ | **Avoid for 48 h.** AI Hub + QNN SDK version maze |

**Efficiency math for the slide:**

| Setup | FPS (pipeline) | Watts | **FPS/W** | $ / FPS |
|---|---|---|---|---|
| Orin Nano Super @15 W | 130 | 15 | **8.7** | 1.9 |
| RTX 4060 laptop | 350 | 90 | 3.9 | 2.9 |
| Pi5 + Hailo-8L | 25 | 11 | 2.3 | 6.0 |
| N100 @320 | 30 | 12 | 2.5 | 6.0 |
| Pi5 CPU @320 | 10 | 8 | 1.3 | 8.0 |

> ⚠️ **Never quote vendor TOPS as an application number.** Hailo-8L "13 TOPS" and Orin Nano "67 TOPS" are INT8-sparse peaks with different utilization on real graphs. The two measured rows above (21.7 ms Hailo-8 vs 2.30 ms Orin NX for a similar nano model) show a **9×** gap that no TOPS ratio predicts.

---

## 4. Which anomaly-detection families are edge-friendly

### 4.1 The three you should consider

| Model | Params / state | Latency (GPU) | Latency (CPU, 1 thread) | Strength | Edge verdict |
|---|---|---|---|---|---|
| **EfficientAD-S** | teacher EfficientNet-B (frozen, ~4 M) + 4-conv student + small AE; **no memory bank** | **2 ms** (paper: 600 img/s) | 12–20 ms @256 | 99.1 % MVTec AUROC, logical anomalies via AE | ⭐ **Best.** Constant RAM, no per-class state, exports cleanly (conv-only) |
| **PatchCore** | WRN50 truncated @ layer2+3, 1024-d, **memory bank** | 4–6 ms + kNN | 40–80 ms | 99.6 % AUROC (paper), best localization | ⭐⭐ accuracy king, but bank + kNN dominate latency |
| **PaDiM** | WRN50/R18, per-patch Gaussian (diag cov) | 3–5 ms + 0.5 ms | 30–50 ms | 95.8 % AUROC, no kNN at inference | Good if you need cheap inference and can afford −3 pts |
| FastFlow / STFPM | normalizing flow / teacher-student | 5–10 ms | 60–120 ms | 99.4 / 95.0 % | flow ops export poorly — skip |

### 4.2 PatchCore memory bank, quantified (do this on the slide)

At 224×224 with WRN50, layer2 (stride 8) = 28×28 = **784 patches**, 1024-d after 3×3 neighborhood pooling:

```
bytes per vector            = 1024 × 4 B = 4 KB
full bank, 100 normal imgs  = 78 400 × 4 KB = 321 MB      ← RAM-hungry, as advertised
10 % greedy coreset         = 7 840  × 4 KB = 32 MB
1 %  greedy coreset         = 784    × 4 KB = 3.2 MB      ← paper default, fine on a Pi
```

But RAM isn't the real cost — **kNN compute is**:

```
FLOPs per image ≈ 2 × n_query × n_bank × dim
full bank  : 2 × 784 × 78 400 × 1024 = 126 GFLOP  → seconds on a Pi
1 % coreset: 2 × 784 ×    784 × 1024 = 1.26 GFLOP → 3–8 ms on a laptop CPU (BLAS), 20–40 ms numpy-loop
```

**Rule:** `latency ∝ coreset size`. Fix the coreset at 1 % and never let it grow per class; memory-bank rebuild time is 0.2–2 s — cache it to `.npz` and load at startup, never recompute during the demo.

### 4.3 Feature extractor choice

| Backbone | GFLOPs @224 | MVTec image AUROC (approx) | Edge fit |
|---|---|---|---|
| MobileNetV3-L | 0.22 | 92–95 % (noticeably worse localization) | Pi CPU OK, but you lose the fine defects you care about |
| ResNet18 | 1.8 | 97–98 % | good compromise, 3× faster than WRN50 |
| **WideResNet50 (layer2+3)** | ~5.5 (truncated) | **99.1–99.6 %** | Orin/laptop yes, Pi CPU no |
| EfficientNet-B (EfficientAD teacher) | ~0.4 | 99.1 % (with student) | designed for this, **< 1 ms on GPU** |
| DINOv2 ViT-S/14 | ~22 (full) | 99.3 % zero-shot | Orin NX yes, Pi no; ViT quantization is fragile |

### 4.4 Input-resolution trade-off (defect size is the constraint)

A defect of physical size `s` mm must cover ≥ 3×3 px in the network input for a CNN to localize it reliably:

```
required_input_px = 3 × (FOV_mm / s_mm)
```
FOV 100 mm, 0.5 mm scratch → 3 × 200 = **600 px**. At 256 px you cannot see it. This is why "just downscale to 224" kills industrial accuracy.

| Input | Relative cost | Use when |
|---|---|---|
| 224–256 | 1.0× | classification-style defects, EfficientAD native size |
| 320 | 1.9× | best latency/accuracy knee for YOLO on Pi/N100 |
| 640 | 6.4× | small defects, multi-object, seg masks |
| 1024+ | 16× | only with line-scan / tiling, and a GPU |

**Tiling beats upscaling:** 4 tiles of 320 on a 640 image costs the same as one 640 pass but preserves resolution. On a Jetson you can run 4 tiles in a batch; on a Pi you can't — which is the real reason high-res inspection is a GPU job.

---

## 5. The 4-hour edge demo (and its fallback)

### 5.1 Demo architecture

```
Pi 5 ── Camera Module 3 (or USB webcam)
  │  picamera2 / V4L2, 1280×720 @ 30 fps, MJPEG
  ├─► edge path  : resize 224 → MobileNetV3/EfficientNet INT8 ONNX  (5–8 ms)
  │                → class + score, overlaid with cv2
  │                → MJPEG multipart stream over Ethernet
  └─► laptop     : FastAPI dashboard, subscribes to the stream,
                   runs the *heavy* model (EfficientAD / YOLO-seg) on the
                   same frames, shows side-by-side latency + verdict
```

**Why this wins:** the judge sees a live defect being caught on a $80 board, *and* a latency comparison table proving the edge path is good enough. You are not betting the demo on one inference stack.

**The killer frame:** put a deliberately scratched part in front of the camera; the Pi overlay flips to `DEFECT 0.97` in < 40 ms; the dashboard prints `edge p95 = 31 ms · laptop p95 = 12 ms · both agree`.

### 5.2 Timeline (T+0 = demo task start)

| Time | Task | Go / No-Go gate |
|---|---|---|
| 0:00–0:20 | Flash Raspberry Pi OS Bookworm 64-bit, `ssh`, `apt install -y python3-opencv python3-picamera2`, `pip install onnxruntime numpy fastapi uvicorn` (venv) | `python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"` |
| 0:20–0:50 | Export MobileNetV3/EfficientNet-B0 classifier to ONNX (224, batch 1), INT8 via `quantize_dynamic` | ≤ 5 MB `.onnx` |
| 0:50–1:30 | `edge_node.py`: capture → resize → infer → annotate → MJPEG | `curl http://pi:8000/stream` shows video |
| 1:30–2:10 | FastAPI dashboard on laptop: `<img src=stream>`, rolling p50/p95, verdict log | latency numbers update live |
| 2:10–2:50 | Mount on a cardboard "conveyor", add a lamp, lock exposure/WB, print a calibration card | blur-free frames at 1/500 s |
| 2:50–3:30 | **Sustained test: 15 min continuous**, record temp + FPS drift | FPS drop < 10 % |
| 3:30–4:00 | Fallback drill: kill the Pi, run `--source 0` on the laptop, same dashboard | recovers in < 60 s |

**Hardware shopping list (India prices, Oct 2026, approximate):**

| Item | Price (₹) | Price ($) | Note |
|---|---|---|---|
| Raspberry Pi 5 8 GB | 7,500–9,000 | 80 | 8 GB is worth it; 4 GB also fine |
| Camera Module 3 (or any UVC webcam) | 2,800–3,500 | 25–35 | webcam = zero driver risk |
| Active cooler + 27 W USB-C PD PSU | 1,200 | 15 | **non-negotiable** — passive Pi 5 throttles to 1.5 GHz |
| 64 GB A2 microSD (or NVMe HAT) | 900 | 10 | A2 matters for model load time |
| Case, jumpers, LED, pushbutton | 800 | 10 | button = "manual trigger", looks great |
| **Total** | **≈ ₹13,500** | **≈ $160** | |
| *Optional:* AI HAT+ 13 TOPS (Hailo-8L) | 7,000 | 70 | only if you pre-compiled the HEF |
| *Optional:* AI HAT+ 26 TOPS (Hailo-8) | 11,000 | 110 | 21.7 ms measured for YOLO26n |

### 5.3 Software stack (pin these)

```
Raspberry Pi OS Bookworm 64-bit (or Trixie if using AI HAT+ 2)
Python 3.11 · onnxruntime 1.20+ (CPU EP) · numpy 2.x · opencv-python-headless 4.10
picamera2 0.3.x · fastapi 0.115 · uvicorn[standard] · pillow (avoid in hot path)
# optional: hailo-all (HailoRT 4.23) · ncnn (pip install ncnn) · openvino 2026.2.1 (x86 only)
```

Minimal edge node (the part that must not break):

```python
# edge_node.py — capture thread keeps ONLY the latest frame (never queue)
import threading, time, cv2, numpy as np, onnxruntime as ort
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

sess = ort.InferenceSession("cls.int8.onnx", providers=["CPUExecutionProvider"])
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual exposure
cap.set(cv2.CAP_PROP_EXPOSURE, 120)         # ~1/500 s
latest, lock, stats = None, threading.Lock(), []

def reader():
    global latest
    while True:
        ok, f = cap.read()
        if ok:
            with lock: latest = f

threading.Thread(target=reader, daemon=True).start()

def infer(frame):
    x = cv2.resize(frame, (224, 224))[:, :, ::-1].transpose(2, 0, 1)[None]
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    t0 = time.perf_counter()
    out = sess.run(None, {"input": x})[0][0]
    return out, (time.perf_counter() - t0) * 1000

app = FastAPI()

def gen():
    while True:
        with lock:
            f = None if latest is None else latest.copy()
        if f is None: time.sleep(0.005); continue
        out, ms = infer(f)
        cls, score = int(out.argmax()), float(out.max())
        stats.append(ms); stats[:] = stats[-500:]
        color = (0, 0, 255) if cls else (0, 200, 0)
        cv2.putText(f, f"{'DEFECT' if cls else 'OK'} {score:.2f} {ms:.1f}ms",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

@app.get("/stream")
def stream(): return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
```

```bash
uvicorn edge_node:app --host 0.0.0.0 --port 8000
# laptop: http://<pi-ip>:8000/stream
```

### 5.4 Fallback that always works

`python demo.py --source 0` on the laptop: same FastAPI app, same model, same dashboard, plus `--source video.mp4 --loop` for when the venue lighting is hostile or a cable dies. **Rehearse the fallback once. A demo that recovers in 30 s scores better than one that never fails but has no plan.**

---

## 6. Benchmarking methodology (this is 20 % of the score)

### 6.1 Rules

1. **Warm up 30–50 iterations.** First call includes cuDNN autotune / TRT tactic selection / ORT graph optimization (0.3–2 s). Reporting it is amateur.
2. **Batch = 1.** Real-time inference is batch 1; batching is a throughput trick that adds latency.
3. **Report p50 / p95 / p99**, never the mean. One 120 ms outlier per 100 frames is what causes a missed reject.
4. **Separate model-only from end-to-end.** Both are legitimate; conflating them is not.
5. **Pin threads and power mode:** `nvpmodel -m 0`, `jetson_clocks`, ORT `intra_op_num_threads` sweep, laptop on AC + "best performance".
6. **Measure 10 minutes sustained**, not 10 seconds. Thermal throttling shows up at minute 3–5.
7. **Report the machine.** `lscpu | head -20`, `nvidia-smi`, `tegrastats`, OS, driver, runtime versions.
8. **Accuracy on the same set** for every precision: FP32 / FP16 / INT8 and 320 / 416 / 640.

### 6.2 Script (see `bench/bench_latency.py` in this repo)

Outline:

```
for backend in [torch, ort_fp32, ort_int8, openvino, trt]:
    warmup(N=50)
    for i in range(N=1000):
        t0 = perf_counter(); backend.infer(x); sync_if_cuda(); lat.append(t0-t1)
    report(p50, p95, p99, fps, power_w, fps_per_w)
    dump json
```

Power sampling:
- Jetson: `tegrastats --interval 200` → parse `VDD_IN` mW
- Pi 5: `vcgencmd pmic_read_adc` (sum `EXT5V_V × *_A`)
- Intel: `/sys/class/powercap/intel-rapl:0/energy_uj`
- Laptop GPU: `nvidia-smi --query-gpu=power.draw --format=csv`

### 6.3 The chart that wins the room

**Pareto scatter:** x = p95 latency (log), y = F1 or AUROC, one point per (precision × resolution), annotated with device. Draw the iso-line `latency = 1/FPS` and shade the region that satisfies the line speed. Then a second small bar chart: **FPS/W** per device.

```python
# bench/plot_pareto.py
import json, glob, matplotlib.pyplot as plt
pts = [json.load(open(f)) for f in glob.glob("results/*.json")]
fig, ax = plt.subplots(figsize=(8, 5))
for p in pts:
    ax.scatter(p["p95_ms"], p["acc"], s=80)
    ax.annotate(f'{p["device"]}/{p["precision"]}@{p["imgsz"]}',
                (p["p95_ms"], p["acc"]), fontsize=8,
                xytext=(5, 5), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("p95 latency (ms, batch 1)")
ax.set_ylabel("F1 / AUROC"); ax.grid(alpha=.3)
ax.axvline(33, ls="--", c="r", label="30 fps budget")
ax.legend(); fig.tight_layout(); fig.savefig("results/pareto.png", dpi=200)
```

### 6.4 Slide skeleton

| Claim | Evidence |
|---|---|
| "Accurate" | AUROC/F1 on 200 held-out defects + 200 good parts, per-class confusion matrix |
| "Fast" | p50/p95/p99 table, 3 devices × 2 precisions |
| "Efficient" | FPS/W bar chart |
| "Deployable" | pi p95 = 31 ms, 26 FPS sustained 15 min, 71 °C, no throttle |
| "Robust" | accuracy at 320 vs 640; INT8 delta table |

---

## 7. Traps (symptom → cause → fix)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | FPS drops 30–40 % after 5 min | **Thermal throttling** — Pi 5 throttles at 80–85 °C, Orin at ~90 °C | Active cooler + fan curve; `vcgencmd measure_temp`, `tegrastats`; test 15 min before demo; `nvpmodel -m 0 && jetson_clocks` |
| 2 | Latency grows unboundedly over minutes | **USB camera buffer backlog** — you process frame #1 while camera is on #300 | Reader thread with `CAP_PROP_BUFFERSIZE=1`, keep only latest; never `cap.read()` in the inference loop |
| 3 | 15 fps with a 30 fps camera on USB2 | YUYV 1080p30 = 1 Gbit/s > USB2 480 Mbit/s | Force **MJPEG** (`FOURCC MJPG`), or 720p, or USB3 port |
| 4 | Camera "freezes" for 200 ms | Auto-exposure / auto-WB hunting | Lock exposure + WB; fixed lighting; `CAP_PROP_AUTO_WB=0` |
| 5 | **GIL**: adding a second stage makes it slower | Python threads don't parallelize Python bytecode | cv2/numpy release the GIL — keep loops out of Python; or `multiprocessing` capture/infer; or GStreamer pipeline |
| 6 | 30 % of a core burnt before inference | **JPEG decode**: 1080p MJPEG ≈ 5–8 ms laptop, **15–30 ms Pi 5** | Ask the camera for 640×480 directly (ISP downscale is free); or H.264 + `v4l2h264dec`; or `cv2.IMWRITE_JPEG_QUALITY` only at the UI |
| 7 | First request takes 3 s, judges notice | **Model load / TRT engine build / OV compile** | Pre-build engines; OpenVINO `CACHE_DIR`; warm up at startup before opening the port |
| 8 | `libcudnn.so.9 not found` | PyTorch wheel ≠ JetPack cuDNN | On Jetson use NVIDIA's wheels (jetson-ai-lab / JetPack index), never plain `pip install torch` |
| 9 | Container builds for 40 min then fails | **Docker on ARM**: x86-only wheels (tensorrt, openvino), no nvidia runtime by default | Skip Docker. venv on host. If you must: `nvcr.io/nvidia/l4t-pytorch` matching JetPack |
| 10 | INT8 mAP collapses | Head/score tensor quantized | Exclude non-Conv/Gemm/MatMul nodes; QDQ format; per-channel weights |
| 11 | INT8 anomaly model flags everything | Anomaly score baseline shifted by quantization noise | Re-fit threshold on 50 good parts (p99); never reuse the float threshold |
| 12 | HEF won't load on the Pi | DFC generation ≠ Hailo generation (3.x for Hailo-8/8L, 5.x for 10H/15) | `hailortcli fw-control identify`; compile with matching DFC and `name=` |
| 13 | TRT engine fails on the second machine | Engines are **device + TRT-version locked** | Build on the deployment device; ship the ONNX as fallback |
| 14 | Works on desk, dies on stage | Venue WiFi / DHCP / different subnet | Direct Ethernet or Pi as AP; `--source 0` fallback; pre-download every model |
| 15 | Numbers don't reproduce | Different thread count / power mode / thermal state | Record `lscpu`, `nvidia-smi`, thread count, ambient temp in the results JSON |

---

## 8. 48-hour plan (edge track)

| Window | Work | Deliverable |
|---|---|---|
| 0–4 h | Data audit, train baseline (EfficientAD or YOLO-n), laptop-only demo working end-to-end | `python demo.py --source 0` runs |
| 4–8 h | Export ONNX + INT8, verify accuracy delta, benchmark script | `results/*.json`, first Pareto plot |
| 8–12 h | **Freeze the model.** Anything after this is optimization, not modeling | tagged checkpoint |
| 12–16 h | TRT FP16 (if GPU) / OpenVINO INT8 (if Intel), re-benchmark | 2–5× on the slide |
| 16–22 h | Edge node (Pi) — only now, with the fallback already working | live MJPEG from Pi |
| 22–30 h | Dashboard polish, latency overlay, defect log, CSV export | judges can see the numbers |
| 30–40 h | Sustained + adversarial tests (lighting change, blur, occlusion), threshold tuning | robustness table |
| 40–46 h | Slides, video capture, rehearsal ×2 including failure drill | 3-min pitch |
| 46–48 h | Buffer. Do not touch the model. | |

**Risk register:** model doesn't converge by hour 8 → switch to PatchCore (no training, 10 min fit). Pi won't boot / no camera → laptop webcam. NPU toolchain fails → ONNX INT8 on laptop CPU (still 5×). Venue network → local AP. Judge asks "is this real-time?" → show p95 table, not FPS.

---

## 9. Sources

- Ultralytics, **Hailo export + benchmarks** (YOLO26n PyTorch 64.6 ms vs Hailo-8 21.7 ms on Pi 5 + AI HAT+; INT8 retention YOLOv8n ~100 %, YOLO11n ~96 %, YOLO26n ~93 % mAP50; DFC 3.33 / HailoRT 4.23) — docs.ultralytics.com/integrations/hailo
- Ultralytics, **OpenVINO benchmarks** (openvino 2026.2.1; Core Ultra 7 155H: PyTorch 33.19 ms → OV FP32 9.13 → FP16 9.07 → INT8 5.79 ms @640) — docs.ultralytics.com/integrations/openvino
- Ultralytics, **TensorRT benchmarks** (RTX 3080: 1.06 / 0.62 / 0.52 ms FP32/FP16/INT8; Jetson Orin NX 16 GB: 6.11 / 3.18 / 2.30 ms; A100 0.52 / 0.34 / 0.28 ms; mAP 0.37 → 0.33 INT8) — docs.ultralytics.com/integrations/tensorrt
- Ultralytics, **ONNX export** (head-exclusion rule for INT8; YOLO26n ~43 % faster than YOLO11n on Xeon CPU ONNX) — docs.ultralytics.com/integrations/onnx
- Batzner et al., **EfficientAD**, WACV 2024, arXiv:2303.14535 — "feature extractor < 1 ms on a modern GPU", "2 ms latency, 600 images/s", 99.1 % MVTec AUROC
- Roth et al., **PatchCore**, CVPR 2022, arXiv:2106.08265 — 99.6 % image AUROC, 1 % coreset memory bank
- NVIDIA, **Jetson Orin Nano Super Developer Kit** — 67 TOPS (INT8 sparse), 8 GB, 7–25 W, $249
- TensorRT 11 vs JetPack note (implicit quantization removed; ModelOpt explicit Q/DQ; DLA last supported in 10.7) — Ultralytics TensorRT page
- Raspberry Pi / Hailo, **AI HAT+**: 13 TOPS (Hailo-8L, $70) and 26 TOPS (Hailo-8, $110)

*Numbers marked "measured" are from the cited benchmark pages on the stated hardware and runtime; ranges marked "estimate/derived" are scaled from those measurements using the TOPS/clocks ratio and should be re-measured on your exact device before you put them on a slide.*

---

### Appendix A — one-command export matrix

```bash
# everything, one script, ~20 min on a laptop with a GPU
python - <<'PY'
from ultralytics import YOLO
m = YOLO("best.pt")
for fmt, kw in [
    ("onnx",     dict(imgsz=640, opset=17, simplify=True)),
    ("onnx",     dict(imgsz=640, quantize=8, data="dataset.yaml")),
    ("openvino", dict(imgsz=640, quantize=8, data="dataset.yaml")),
    ("engine",   dict(imgsz=640, quantize=16, workspace=4)),
    ("ncnn",     dict(imgsz=320, half=True)),
]:
    try:
        print(fmt, m.export(format=fmt, **kw))
    except Exception as e:
        print("FAIL", fmt, e)
PY
```

### Appendix B — acceptance thresholds before you present

| Metric | Threshold | If missed |
|---|---|---|
| p95 e2e latency | < 100 ms (laptop), < 150 ms (Pi) | reduce imgsz, switch to INT8 |
| Sustained FPS (10 min) | > 15 | add cooling, lower resolution |
| Accuracy drop INT8 vs FP32 | < 2 pts F1 | keep FP16, or re-calibrate in-domain |
| Cold start | < 5 s | cache compiled model, warm at boot |
| Recovery from camera unplug | < 10 s | retry loop in reader thread |
