#!/usr/bin/env python3
"""Image-paired bootstrap of frozen validation predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_unified_val_matching import score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["steel_crazing", "steel_inclusion", "steel_patches", "steel_pitted_surface",
          "steel_scratches", "metal_nut_scratch", "screw_manipulated_front",
          "screw_thread_side", "screw_thread_top", "transistor_bent_lead",
          "transistor_damaged_case"]
IDS = [0, 1, 2, 3, 4, 8, 9, 12, 13, 14, 16]


def interval(values: list[float], repetitions: int) -> dict:
    if not values:
        return {"valid_replicates": 0, "valid_fraction": 0.0, "ci95": None}
    return {"valid_replicates": len(values), "valid_fraction": len(values) / repetitions,
            "ci95": list(map(float, np.quantile(values, [0.025, 0.975])))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    baseline = json.loads((ROOT / args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads((ROOT / args.candidate).read_text(encoding="utf-8")) if args.candidate else None
    images = baseline["images"]
    if candidate and (candidate["images"] != images or candidate["gts"] != baseline["gts"]):
        raise ValueError("The predictions do not share the same images and ground truth")
    rng = np.random.default_rng(42)
    base_values = {name: [] for name in LABELS}
    candidate_values = {name: [] for name in LABELS}
    differences = {name: [] for name in LABELS}
    steel_base, steel_candidate, steel_difference = [], [], []
    for iteration in range(args.repetitions):
        sampled = [images[i] for i in rng.integers(0, len(images), len(images))]
        per_base, per_candidate = {}, {}
        for name, cid in zip(LABELS, IDS):
            b = score(baseline["gts"], baseline["preds"], sampled, cid, "0.5")
            c = score(candidate["gts"], candidate["preds"], sampled, cid, "0.5") if candidate else None
            if b is not None:
                base_values[name].append(b)
                per_base[name] = b
            if c is not None:
                candidate_values[name].append(c)
                per_candidate[name] = c
            if b is not None and c is not None:
                differences[name].append(c - b)
        base_steel = float(np.mean([per_base[name] for name in LABELS[:5]])) if all(
            name in per_base for name in LABELS[:5]) else None
        candidate_steel = float(np.mean([per_candidate[name] for name in LABELS[:5]])) if candidate and all(
            name in per_candidate for name in LABELS[:5]) else None
        if base_steel is not None:
            steel_base.append(base_steel)
        if candidate_steel is not None:
            steel_candidate.append(candidate_steel)
        if base_steel is not None and candidate_steel is not None:
            steel_difference.append(candidate_steel - base_steel)
        if (iteration + 1) % 50 == 0:
            print(f"bootstrap {iteration+1}/{args.repetitions}", flush=True)
    report = {"sampling_unit": "validation image", "paired": bool(candidate), "seed": 42,
              "repetitions": args.repetitions,
              "steel_macro": {"baseline": interval(steel_base, args.repetitions),
                              "candidate": interval(steel_candidate, args.repetitions) if candidate else None,
                              "candidate_minus_baseline": interval(steel_difference, args.repetitions) if candidate else None},
              "per_class": {name: {"baseline": interval(base_values[name], args.repetitions),
                                   "candidate": interval(candidate_values[name], args.repetitions) if candidate else None,
                                   "candidate_minus_baseline": interval(differences[name], args.repetitions) if candidate else None}
                            for name in LABELS}}
    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"steel_macro": report["steel_macro"],
                      "rare": {n: report["per_class"][n] for n in ("screw_thread_side", "transistor_bent_lead")}}, indent=2))


if __name__ == "__main__":
    main()
