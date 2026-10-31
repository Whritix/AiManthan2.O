"""Build PatchCore-lite memory banks for every demo profile."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.anomaly.gate import PatchCoreGate


def quick_check(gate: PatchCoreGate, profile: dict, n: int = 40) -> dict:
    def scores(paths: list[str]) -> list[float]:
        out = []
        for p in paths[:n]:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            out.append(gate.score_frame(img).z)
        return out

    good = scores([g["path"] for g in profile.get("good", [])])
    known = scores([d["path"] for d in profile.get("defect", [])])
    unknown = scores([d["path"] for d in profile.get("unknown", [])])
    return {
        "good": {
            "n": len(good),
            "z_p50": float(np.percentile(good, 50)) if good else None,
            "z_p95": float(np.percentile(good, 95)) if good else None,
        },
        "defect": {
            "n": len(known),
            "z_p50": float(np.percentile(known, 50)) if known else None,
            "z_p95": float(np.percentile(known, 95)) if known else None,
        },
        "unknown": {
            "n": len(unknown),
            "z_p50": float(np.percentile(unknown, 50)) if unknown else None,
            "z_p95": float(np.percentile(unknown, 95)) if unknown else None,
        },
        "separation_defect": float(np.median(known) - np.median(good)) if good and known else None,
        "separation_unknown": float(np.median(unknown) - np.median(good)) if good and unknown else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/profiles/manifest.json")
    ap.add_argument("--out", default="artifacts/models")
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--bank-vectors", type=int, default=4000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = args.profiles or list(manifest)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    summary_path = out / "gate_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    for pid in wanted:
        profile = manifest[pid]
        print(f"\n=== {pid} ===")
        t0 = time.perf_counter()
        gate = PatchCoreGate(
            device=device,
            input_size=256,
            tile_size=96,
            grid=4,
            top_k=2,
            bank_vectors=args.bank_vectors,
            backbone_name=args.backbone,
        )
        if profile.get("bank_crops"):
            gate.build_bank(
                crop_paths=profile["bank_crops"],
                calib_frame_paths=profile["calib_frames"],
                bank_vectors=args.bank_vectors,
                seed=args.seed,
            )
        else:
            gate.build_bank(
                frame_paths=profile["bank_images"],
                calib_frame_paths=profile["calib_frames"],
                bank_vectors=args.bank_vectors,
                seed=args.seed,
            )
        gate.metadata.update(
            {
                "profile": pid,
                "display_name": profile["display_name"],
                "license": profile["license"],
                "bank_source": "clean_crops" if profile.get("bank_crops") else "good_frames",
                "built_at": int(time.time() * 1000),
            }
        )
        path = out / profile["gate_model"]
        gate.save(path)
        check = quick_check(gate, profile)
        summary[pid] = {
            "path": str(path),
            "backbone": args.backbone,
            "bank_vectors": gate.metadata["bank_vectors"],
            "frame_p50": gate.frame_p50,
            "frame_p99": gate.frame_p99,
            "patch_p50": gate.patch_p50,
            "patch_p99": gate.patch_p99,
            "check": check,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        print(json.dumps(summary[pid], indent=2))

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote", out / "gate_summary.json")


if __name__ == "__main__":
    main()
