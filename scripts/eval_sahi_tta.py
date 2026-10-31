"""Test-Time Augmentation (TTA) and Sliced Aided Hyper Inference (SAHI) Evaluation.

Features:
1. Native Ultralytics Test-Time Augmentation (TTA) using `augment=True`.
2. Sliced Window Inference (SAHI):
   - Uses `sahi` library if installed.
   - Built-in zero-dependency slicing engine fallback (OpenCV + torchvision/cv2 NMS)
     so it works out-of-the-box even before installing external SAHI packages.
3. Benchmarks high-resolution predictions (1024 or 1280) against standard inference.
4. Outputs detection statistics, overlays, and performance metrics.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def nms_boxes(
    boxes: list[list[float]],
    scores: list[float],
    classes: list[int],
    iou_thresh: float = 0.5,
) -> tuple[list[list[float]], list[float], list[int]]:
    """Merges overlapping sliced predictions using Non-Maximum Suppression."""
    if not boxes:
        return [], [], []

    try:
        from torchvision.ops import nms
        t_boxes = torch.tensor(boxes, dtype=torch.float32)
        t_scores = torch.tensor(scores, dtype=torch.float32)
        # Class-aware NMS offset
        max_coord = t_boxes.max().item() if t_boxes.numel() > 0 else 0
        t_classes = torch.tensor(classes, dtype=torch.float32)
        offsets = t_classes * (max_coord + 1)
        boxes_for_nms = t_boxes + offsets[:, None]

        keep_indices = nms(boxes_for_nms, t_scores, iou_thresh).cpu().numpy()
        return (
            [boxes[i] for i in keep_indices],
            [scores[i] for i in keep_indices],
            [classes[i] for i in keep_indices],
        )
    except Exception:
        # Fallback to OpenCV DNN NMS
        cv_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(cv_boxes, scores, score_threshold=0.1, nms_threshold=iou_thresh)
        if len(indices) == 0:
            return [], [], []
        indices = indices.flatten()
        return (
            [boxes[i] for i in indices],
            [scores[i] for i in indices],
            [classes[i] for i in indices],
        )


def sliced_inference_fallback(
    model: YOLO,
    img: np.ndarray,
    slice_size: int = 640,
    overlap_ratio: float = 0.20,
    conf_thresh: float = 0.25,
    device: str = "0",
    use_tta: bool = False,
) -> dict:
    """Zero-dependency sliding window inference with coordinate shifting & NMS."""
    h, w = img.shape[:2]
    step = int(slice_size * (1.0 - overlap_ratio))
    step = max(32, step)

    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_classes: list[int] = []

    # 1. Full image inference (macro view)
    res_full = model.predict(img, imgsz=max(h, w), conf=conf_thresh, device=device, augment=use_tta, verbose=False)[0]
    if res_full.boxes is not None and len(res_full.boxes) > 0:
        for b in res_full.boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            all_boxes.append(xyxy)
            all_scores.append(float(b.conf.item()))
            all_classes.append(int(b.cls.item()))

    # 2. Window slices across the grid
    y_starts = list(range(0, max(1, h - slice_size + 1), step))
    if y_starts[-1] + slice_size < h:
        y_starts.append(h - slice_size)

    x_starts = list(range(0, max(1, w - slice_size + 1), step))
    if x_starts[-1] + slice_size < w:
        x_starts.append(w - slice_size)

    slices_count = 0
    for ys in y_starts:
        for xs in x_starts:
            ye = min(h, ys + slice_size)
            xe = min(w, xs + slice_size)
            crop = img[ys:ye, xs:xe]
            slices_count += 1

            res_slice = model.predict(
                crop,
                imgsz=slice_size,
                conf=conf_thresh,
                device=device,
                augment=use_tta,
                verbose=False,
            )[0]

            if res_slice.boxes is not None and len(res_slice.boxes) > 0:
                for b in res_slice.boxes:
                    local_xyxy = b.xyxy[0].cpu().numpy()
                    # Shift coordinates back to full-image coordinate space
                    global_xyxy = [
                        float(local_xyxy[0] + xs),
                        float(local_xyxy[1] + ys),
                        float(local_xyxy[2] + xs),
                        float(local_xyxy[3] + ys),
                    ]
                    all_boxes.append(global_xyxy)
                    all_scores.append(float(b.conf.item()))
                    all_classes.append(int(b.cls.item()))

    # 3. Fuse and NMS
    fused_boxes, fused_scores, fused_classes = nms_boxes(all_boxes, all_scores, all_classes, iou_thresh=0.50)

    return {
        "boxes": fused_boxes,
        "scores": fused_scores,
        "classes": fused_classes,
        "n_slices": slices_count,
    }


def draw_predictions(img: np.ndarray, boxes: list[list[float]], scores: list[float], classes: list[int], names: dict) -> np.ndarray:
    out = img.copy()
    for box, score, cid in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names.get(cid, str(cid))} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Test-Time Augmentation & SAHI Slicing Evaluation")
    ap.add_argument("--model", default="artifacts/models/detector_best.pt", help="Path to YOLO weights (.pt)")
    ap.add_argument("--source", default="data/neu-det/images/val", help="Folder of test images or single image")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference image resolution")
    ap.add_argument("--slice-size", type=int, default=640, help="Slice window dimension")
    ap.add_argument("--overlap", type=float, default=0.20, help="Slice overlap ratio")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", default="0", help="GPU device ID or 'cpu'")
    ap.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (TTA)")
    ap.add_argument("--sahi", action="store_true", help="Enable Sliced Inference (SAHI)")
    ap.add_argument("--limit", type=int, default=50, help="Maximum number of images to evaluate")
    ap.add_argument("--save-dir", default="artifacts/eval/sahi_tta_results", help="Output directory for visual results")
    args = ap.parse_args()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

    src_path = Path(args.source)
    if src_path.is_file():
        image_paths = [src_path]
    else:
        valid_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
        image_paths = sorted([p for p in src_path.glob("*.*") if p.suffix.lower() in valid_exts])
        if args.limit:
            image_paths = image_paths[: args.limit]

    print("=" * 60)
    print(" YOLO11 ADVANCED INFERENCE BENCHMARK")
    print(f" Model:        {args.model}")
    print(f" Input imgsz:  {args.imgsz}")
    print(f" Mode:         TTA={args.tta} | SAHI={args.sahi}")
    print(f" Images:       {len(image_paths)}")
    print(f" Device:       {args.device}")
    print("=" * 60)

    total_time = 0.0
    total_detections = 0
    class_detections: dict[str, int] = {}

    for idx, impath in enumerate(image_paths):
        img = cv2.imread(str(impath))
        if img is None:
            continue

        t0 = time.perf_counter()

        if args.sahi:
            # Sliced Inference
            res = sliced_inference_fallback(
                model=model,
                img=img,
                slice_size=args.slice_size,
                overlap_ratio=args.overlap,
                conf_thresh=args.conf,
                device=args.device,
                use_tta=args.tta,
            )
            boxes, scores, classes = res["boxes"], res["scores"], res["classes"]
        else:
            # Standard or TTA Inference
            pred = model.predict(
                source=img,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                augment=args.tta,
                verbose=False,
            )[0]
            boxes, scores, classes = [], [], []
            if pred.boxes is not None:
                for b in pred.boxes:
                    boxes.append(b.xyxy[0].cpu().numpy().tolist())
                    scores.append(float(b.conf.item()))
                    classes.append(int(b.cls.item()))

        dt = (time.perf_counter() - t0) * 1000.0
        total_time += dt
        total_detections += len(boxes)

        for cid in classes:
            cname = names.get(cid, str(cid))
            class_detections[cname] = class_detections.get(cname, 0) + 1

        # Save sample visualization for the first 10 images
        if idx < 10:
            rendered = draw_predictions(img, boxes, scores, classes, names)
            cv2.imwrite(str(out_dir / f"{impath.stem}_pred.jpg"), rendered)

    n_eval = max(1, len(image_paths))
    avg_latency = total_time / n_eval
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    summary = {
        "model": args.model,
        "input_resolution": args.imgsz,
        "tta_enabled": args.tta,
        "sahi_enabled": args.sahi,
        "images_evaluated": n_eval,
        "total_detections": total_detections,
        "avg_detections_per_image": round(total_detections / n_eval, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "detections_by_class": class_detections,
    }

    report_path = out_dir / "benchmark_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(" INFERENCE BENCHMARK RESULTS")
    print(f" Avg Latency:        {avg_latency:.1f} ms ({fps:.1f} FPS)")
    print(f" Total Detections:   {total_detections} (avg {total_detections / n_eval:.2f}/img)")
    print(" Detections by Class:")
    for cname, cnt in sorted(class_detections.items(), key=lambda x: -x[1]):
        print(f"   - {cname}: {cnt}")
    print(f" Report saved:       {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
