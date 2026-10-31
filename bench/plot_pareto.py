#!/usr/bin/env python3
"""
plot_pareto.py - turn bench_latency.py results into the two charts that matter.

  python plot_pareto.py --results results --out results/pareto.png

Chart 1: accuracy vs p95 latency (log x) with the line-speed budget shaded.
Chart 2: FPS/W per configuration (efficiency story).

Only points that have "acc" set are drawn on chart 1 - pass --acc when benchmarking.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(results_dir: str) -> list[dict]:
    rows = []
    for f in glob.glob(str(Path(results_dir) / "*.json")):
        try:
            rows.append(json.load(open(f)))
        except Exception as e:  # noqa: BLE001
            print(f"skip {f}: {e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/pareto.png")
    ap.add_argument("--budget-ms", type=float, default=33.0,
                    help="line-speed budget; 33 ms = 30 fps, 200 ms = 300 ppm")
    ap.add_argument("--fps-slide", default=None, help="optional: separate FPS/W chart path")
    args = ap.parse_args()

    rows = load(args.results)
    if not rows:
        raise SystemExit("no results/*.json found - run bench_latency.py first")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(args.budget_ms, max(r["p95_ms"] for r in rows) * 2,
               color="red", alpha=0.06)
    ax.axvline(args.budget_ms, ls="--", c="crimson", lw=1.2,
               label=f"budget {args.budget_ms:.0f} ms/part")

    for r in rows:
        if r.get("acc") is None:
            continue
        ax.scatter(r["p95_ms"], r["acc"], s=110, zorder=3)
        ax.annotate(f'{r["label"]}\n{r["p95_ms"]:.1f} ms',
                    (r["p95_ms"], r["acc"]), fontsize=7.5,
                    xytext=(6, 6), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("p95 latency (ms, batch 1)")
    ax.set_ylabel("accuracy (F1 / AUROC)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    ax.set_title("Accuracy vs latency - points to the lower-left win")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")

    eff = [r for r in rows if r.get("fps_per_w")]
    if eff and args.fps_slide:
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        labels = [r["label"] for r in eff]
        vals = [r["fps_per_w"] for r in eff]
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        ax2.barh([labels[i] for i in order], [vals[i] for i in order], color="#3b7dd8")
        ax2.set_xlabel("FPS per Watt (p50 FPS / mean package power)")
        ax2.set_title("Efficiency per device")
        ax2.grid(axis="x", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(args.fps_slide, dpi=200)
        print(f"wrote {args.fps_slide}")

    print(f"\n{'label':38s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'FPS':>7s} {'W':>6s} {'FPS/W':>7s}")
    for r in sorted(rows, key=lambda x: x["p95_ms"]):
        print(f'{r["label"][:38]:38s} {r["p50_ms"]:8.2f} {r["p95_ms"]:8.2f} '
              f'{r["p99_ms"]:8.2f} {r["fps_p50"]:7.1f} '
              f'{(r["power_w"] or 0):6.1f} {(r["fps_per_w"] or 0):7.2f}')


if __name__ == "__main__":
    main()
