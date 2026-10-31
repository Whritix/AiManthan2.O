"""Rehydrate gate_summary.json from existing memory banks and MVTec audit data.

This does not rebuild or re-score the banks, so it is safe to run while training.
The steel entry is preserved as-is; MVTec entries are added from the audit file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
SUMMARY = MODELS / "gate_summary.json"
AUDIT = ROOT / "artifacts" / "eval" / "gate_audit_mvtec.json"


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    for profile_id, result in audit.items():
        bank_path = MODELS / f"gate_{profile_id}.npz"
        if not bank_path.exists():
            print(f"skip {profile_id}: missing {bank_path}")
            continue

        bank = np.load(bank_path, allow_pickle=True)
        raw_meta = bank["metadata"][0]
        if isinstance(raw_meta, bytes):
            raw_meta = raw_meta.decode("utf-8")
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)

        summary[profile_id] = {
            "path": str(bank_path.relative_to(ROOT)),
            "backbone": "wide_resnet50_2",
            "bank_vectors": int(meta.get("bank_vectors", len(bank["bank"]))),
            "frame_p50": float(bank["frame_p50"].item()),
            "frame_p99": float(bank["frame_p99"].item()),
            "patch_p50": float(bank["patch_p50"].item()),
            "patch_p99": float(bank["patch_p99"].item()),
            "check": {
                "good": {
                    "n": int(result["good_n"]),
                    "z_p50": float(result["good_z_p50"]),
                    "z_p95": float(result["good_z_p95"]),
                },
                "defect": {
                    "n": int(result["defect_n"]),
                    "z_p50": float(result["defect_z_p50"]),
                    "z_p95": float(result["defect_z_p95"]),
                },
                # Unknown-set scores were not part of the MVTec audit; the server
                # falls back to separation_defect when separation_unknown is null.
                "unknown": {"n": 0, "z_p50": None, "z_p95": None},
                "separation_defect": float(result["sep_median"]),
                "separation_unknown": None,
            },
            "seconds": None,
            "note": "rehydrated from existing bank; unknown set not re-scored",
        }

    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {SUMMARY}")
    for profile_id, entry in summary.items():
        check = entry.get("check", {})
        print(
            f"{profile_id:12s} "
            f"defect_sep={check.get('separation_defect')} "
            f"unknown_sep={check.get('separation_unknown')}"
        )


if __name__ == "__main__":
    main()
