#!/usr/bin/env python3
"""Export trained YOLO11s PyTorch checkpoint to ONNX runtime format."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def export(
    weights: Path,
    out_dir: Path | None = None,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
) -> Path:
    from ultralytics import YOLO

    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    print(f"Loading weights from {weights}...")
    model = YOLO(str(weights), task="detect")

    print(f"Exporting to ONNX (imgsz={imgsz}, opset={opset}, simplify={simplify}, dynamic={dynamic})...")
    t0 = time.perf_counter()
    exported_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
    )
    dt = time.perf_counter() - t0
    print(f"Export completed in {dt:.2f}s -> {exported_path_str}")

    exported_path = Path(exported_path_str)
    if out_dir is not None:
        target = out_dir / exported_path.name
        if target.resolve() != exported_path.resolve():
            import shutil
            shutil.copy2(exported_path, target)
            print(f"Copied to target: {target}")
            exported_path = target

    return exported_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO11s weights to ONNX")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "artifacts" / "models" / "detector_best.pt",
        help="Path to PyTorch .pt weights",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "models",
        help="Directory to save exported ONNX model",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Disable onnxsim")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = export(
        weights=args.weights,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
    )
    print(f"\nSuccessfully generated: {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
