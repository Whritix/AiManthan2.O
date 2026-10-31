#!/usr/bin/env python3
"""Smoke test — verifies the server can import cleanly and core sub-systems are sane.

Run from the project root:
    python scripts/smoke_test.py

Exit 0 = all checks passed.
Exit 1 = at least one check failed.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures: list[str] = []
warnings: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = PASS if ok else FAIL
    line = f"  {mark}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not ok:
        failures.append(label)


def warn(label: str, detail: str = "") -> None:
    line = f"  {WARN}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    warnings.append(label)


print("\n--- Indore QA Cell -- Smoke Test ---\n")

# ── 1. Core imports ───────────────────────────────────────────────────────
print("[1] Python imports")
for pkg in ("fastapi", "uvicorn", "cv2", "numpy", "pydantic", "torch"):
    try:
        importlib.import_module(pkg)
        check(pkg, True)
    except ImportError as exc:
        check(pkg, False, str(exc))

# ── 2. Server modules ─────────────────────────────────────────────────────
print("\n[2] Server module imports")
for mod in (
    "server.app.config",
    "server.app.schemas",
    "server.app.db",
    "server.app.fusion",
    "server.app.overlay",
    "server.app.detector",
    "server.app.gate_service",
    "server.app.player",
    "server.app.main",
):
    try:
        importlib.import_module(mod)
        check(mod.split(".")[-1], True)
    except Exception as exc:
        check(mod.split(".")[-1], False, str(exc)[:80])

# ── 3. Config + filesystem ────────────────────────────────────────────────
print("\n[3] Configuration and filesystem")
from server.app.config import settings  # noqa: E402

check("project_root exists", ROOT.exists(), str(ROOT))
check("data_dir exists", settings.data_dir.exists(), str(settings.data_dir))
check("artifacts_dir exists", settings.artifacts_dir.exists(), str(settings.artifacts_dir))
check("models_dir exists", settings.models_dir.exists(), str(settings.models_dir))
check("splits.json exists", settings.splits_path.exists(), str(settings.splits_path))
check("profiles manifest exists", settings.profiles_manifest.exists(), str(settings.profiles_manifest))

# yolo11s.pt or promoted weights
promoted = settings.promoted_weights
if promoted:
    check("promoted weights", True, promoted.name)
else:
    base_yolo = ROOT / "yolo11s.pt"
    if base_yolo.exists():
        warn("promoted weights", f"none — base {base_yolo.name} present but not promoted; detector will run in fixture mode")
    else:
        warn("promoted weights", "none — detector will run in fixture mode")

# web dist
dist = settings.web_dist
check("web dist built", (dist / "index.html").exists(), str(dist))

# ── 4. Profiles ───────────────────────────────────────────────────────────
print("\n[4] Profiles and datasets")
from server.app.profiles import ProfileRegistry  # noqa: E402

reg = ProfileRegistry(settings)
profiles = reg.all()
check("profiles loaded", len(profiles) > 0, f"{len(profiles)} profiles")
for p in profiles:
    good = len(p.good)
    defect = len(p.defect)
    unk = len(p.unknown)
    total = good + defect + unk
    trusted = p.gate_trusted
    if total == 0:
        check(f"  {p.id} data", False, "no images")
    elif not trusted:
        warn(f"  {p.id} gate", f"NOT TRUSTED ({p.gate_quality}) — gate shown for transparency only")
        check(f"  {p.id} data", True, f"good={good} defect={defect} unknown={unk}")
    else:
        check(f"  {p.id} data+gate", True, f"good={good} defect={defect} unknown={unk} quality={p.gate_quality}")

# ── 5. Gate service load ──────────────────────────────────────────────────
print("\n[5] Gate service (quick load, no inference)")
from server.app.gate_service import GateService  # noqa: E402

gate = GateService(settings)
default_profile = reg.get(settings.profile)
gate.ensure(default_profile)
check("gate ready", gate.ready, f"device={gate.device}")
if gate.error:
    check("gate no error", False, gate.error)

# ── 6. Database ───────────────────────────────────────────────────────────
print("\n[6] Database")
from server.app.db import Database  # noqa: E402

try:
    db = Database(":memory:")
    counts = db.counts()
    check("SQLite WAL", True, str(counts))
    db.conn.close()
except Exception as exc:
    check("SQLite WAL", False, str(exc))

# ── 7. FrameResult contract ───────────────────────────────────────────────
print("\n[7] FrameResult JSON contract")
from server.app.schemas import FrameResult, AnomalyInfo, LatencyMs, PartState  # noqa: E402

sample = FrameResult(
    frame_id="frm_test",
    inspection_id="insp_test",
    ts_ms=int(time.time() * 1000),
    profile="steel",
    source="dataset",
    device="cpu",
    engine="onnx-int8 + patchcore-lite/gate_steel.npz",
    model_version="detector_fixture+gate_steel.npz",
    threshold_profile="threshold_v1",
    latency_ms=LatencyMs(capture=3, preprocess=4, infer_detector=18, infer_gate=12, post=2, encode=6, total=45),
    anomaly=AnomalyInfo(
        raw_score=0.12, z=0.34, threshold=0.75, is_anomaly=False,
        model="patchcore-lite/gate_steel.npz", gate_trusted=False, gate_quality="not_validated",
    ),
    verdict="pass",
    part_state=PartState(index=1, total=100, item_id="insp_test", kind="good"),
)
try:
    dumped = sample.model_dump(by_alias=True, mode="json")
    check("FrameResult serialise", "schema_version" in dumped and "verdict" in dumped)
    _ = FrameResult.model_validate(dumped)
    check("FrameResult round-trip", True)
except Exception as exc:
    check("FrameResult round-trip", False, str(exc))

# ── 8. Overlay renderer ───────────────────────────────────────────────────
print("\n[8] Overlay renderer")
import cv2, numpy as np  # noqa: E402
from server.app.overlay import render  # noqa: E402

try:
    frame = np.zeros((200, 200, 3), np.uint8)
    out_hud = render(frame, [], "pass", profile_id="steel", engine="onnx", source="dataset",
                     gate_z=0.1, gate_trusted=False, gate_quality="not_validated", hud=True)
    out_clean = render(frame, [], "fail", profile_id="steel", engine="onnx", source="dataset",
                       gate_z=1.5, gate_trusted=True, gate_quality="good", hud=False)
    check("render hud=True", out_hud.shape == (800, 800, 3))
    check("render hud=False", out_clean.shape == (800, 800, 3))
except Exception as exc:
    check("overlay render", False, str(exc))

# ── 9. Heldout assertion ──────────────────────────────────────────────────
print("\n[9] Heldout-class integrity")
splits = json.loads(settings.splits_path.read_text(encoding="utf-8"))
heldout_class = splits.get("heldout_class")
trained = splits.get("trained_classes", [])
check("heldout class defined", bool(heldout_class), heldout_class or "MISSING")
if heldout_class:
    check("heldout NOT in trained", heldout_class not in trained,
          f"{'LEAKED' if heldout_class in trained else 'safe'}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n--- Summary ---")
if failures:
    print(f"\n  {FAIL}  {len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"        • {f}")
if warnings:
    print(f"\n  {WARN}  {len(warnings)} warning(s) (non-fatal):")
    for w in warnings:
        print(f"        • {w}")
if not failures:
    print(f"\n  {PASS}  All checks passed.  Server is ready to start.\n")
    print("  Run:  DEMO.bat")
    print("  Or:   python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000\n")
else:
    print(f"\n  Fix the failed checks above before running the demo.\n")

sys.exit(0 if not failures else 1)
