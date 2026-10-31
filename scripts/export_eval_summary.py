"""Build the UI eval summary from a canonical evaluation and the operating scan.

The old version read `audit_current_model.json`, which was produced with
Ultralytics rectangular validation. That mode inflates mAP on this dataset and
does not match deployment (`rect=False`). This version reads a canonical
`eval_canonical.py` output and the current `conf_scan.json`, so the UI reports
the deployment protocol, the fixed operating point, and a scan that is checked
against the promoted weights SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "artifacts" / "eval" / "canonical_yolo11s_1024aug_640_deploy.json"
DEFAULT_SCAN = ROOT / "artifacts" / "eval" / "conf_scan.json"
DEFAULT_OUT = ROOT / "artifacts" / "eval" / "test_metrics.json"
FLOORS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hit_rate(rows: list, floor: float) -> float:
    if not rows:
        return 0.0
    return sum(1 for boxes in rows if any(float(conf) >= floor for _, conf in boxes)) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    scan = json.loads(args.scan.read_text(encoding="utf-8"))
    meta = scan["_meta"]
    weights_sha = canonical["weights_sha256"]
    scan_current = meta["weights_sha256"] == weights_sha

    per_class = {}
    for name, values in canonical["metrics"]["per_class"].items():
        operating = canonical["operating_point"]["per_class"].get(name, {})
        per_class[name] = {
            "precision": float(operating.get("precision", 0.0)),
            "recall": float(operating.get("recall", 0.0)),
            "map50": float(values["map50"]),
            "map50_95": float(values["map50_95"]),
            "n_images": 45,
        }

    sweep = []
    for floor in FLOORS:
        good_fp = hit_rate(scan.get("good", []), floor)
        sweep.append(
            {
                "conf": floor,
                "test_hit": hit_rate(scan.get("test", []), floor),
                "good_fp_rate": good_fp,
                "good_fp_per_1000": good_fp * 1000.0,
                "heldout_det_rate": hit_rate(scan.get("heldout", []), floor),
            }
        )

    op_conf = float(canonical["operating_point"]["conf"])
    op_sweep = next((row for row in sweep if abs(row["conf"] - op_conf) < 1e-9), None)
    if op_sweep is None:
        raise SystemExit(f"operating conf {op_conf} not in scan floors {FLOORS}")

    payload = {
        "weights": "artifacts/models/detector_best.pt",
        "evaluated_from": canonical["weights"],
        "weights_sha256": weights_sha,
        "split": canonical["split"],
        "imgsz": canonical["imgsz"],
        "nms_iou": canonical["nms_iou"],
        "rect": canonical.get("rect", False),
        "overall": {
            "precision": float(canonical["operating_point"]["precision"]),
            "recall": float(canonical["operating_point"]["recall"]),
            "map50": float(canonical["metrics"]["map50"]),
            "map50_95": float(canonical["metrics"]["map50_95"]),
            "n_images": 225,
        },
        "per_class": per_class,
        "speed_ms": {key: float(value) for key, value in canonical["metrics"]["speed_ms"].items()},
        "created_at_ms": int(time.time() * 1000),
        "source": str(args.canonical.resolve().relative_to(ROOT)).replace("\\", "/"),
        "metric_note": (
            f"mAP from the full PR curve at conf=0.001; P/R at the fixed operating conf={op_conf:.2f}. "
            "NMS IoU 0.6; AP matching 0.50:0.95; rect=False (deployment preprocessing)."
        ),
        "operating_conf": op_conf,
        "operating_scan_status": "current" if scan_current else "pending_rerun",
        "operating_scan_weights_sha256": meta["weights_sha256"],
        "operating_scan_imgsz": meta["imgsz"],
        "operating_scan_created_at_ms": meta["created_at_ms"],
        "threshold_sweep": sweep,
        "heldout_scan": {
            "folder": "data/neu-det/heldout/images",
            "n_images": int(meta["counts"]["heldout"]),
            "conf": op_conf,
            "detection_rate": op_sweep["heldout_det_rate"],
            "source": "conf_scan.json",
        },
        "good_scan": {
            "folder": "data/neu-good/frames/test",
            "n_images": int(meta["counts"]["good"]),
            "conf": op_conf,
            "detection_rate": op_sweep["good_fp_rate"],
            "source": "conf_scan.json",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        f"mAP50={payload['overall']['map50']:.4f} "
        f"mAP50-95={payload['overall']['map50_95']:.4f} "
        f"P@{op_conf:.2f}={payload['overall']['precision']:.4f} "
        f"R@{op_conf:.2f}={payload['overall']['recall']:.4f} "
        f"scan={payload['operating_scan_status']}"
    )


if __name__ == "__main__":
    main()
