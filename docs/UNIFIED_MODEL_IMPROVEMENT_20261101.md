# Unified detector improvement check — 1 November 2026

## Model and evaluation

- Saved checkpoint: `artifacts/runs/unified_yolo11s_640/weights/best.pt` (SHA-256 `3e500b2d9b915f0d6c61433784376b9addfd7ae0abb4950f59623683e914388b`). It was not replaced.
- Previously recorded test score at 640: mAP50 **0.7012**, mAP50-95 **0.4914** on 789 images. This test score was already known before the experiments below; it is not a fresh blind test.
- Controlled validation below used the same 785 images, confidence 0.001, NMS IoU 0.6, and rectangular batching. Reports: `artifacts/eval/unified_finetune_20261101/`.

| Checkpoint and input size | Validation mAP50 | Validation mAP50-95 |
| --- | ---: | ---: |
| Saved baseline, 640 | **0.7604** | **0.5155** |
| Saved baseline, 960 | 0.5779 | 0.2674 |
| Gentle fine-tune best checkpoint, 640 | 0.5410 | 0.3333 |

| Weak class | Baseline 640 AP50 | Baseline 960 AP50 | Fine-tune 640 AP50 |
| --- | ---: | ---: | ---: |
| Steel crazing | 0.393 | 0.311 | 0.330 |
| Screw manipulated front | 0.373 | 0.116 | 0.034 |
| Screw thread side | 0.085 | 0.042 | 0.197 |
| Screw thread top | 0.995 | 0.747 | 0.995 |
| Transistor bent lead | 0.755 | 0.421 | 0.063 |

The fine-tune was stopped after four complete epochs because its best training-validation mAP50 was only 0.581, far below the saved model. The very first attempt was stopped after two epochs because the trainer's default bias warmup learning rate was about 0.067 despite `lr0=0.001`. The second attempt capped warmup bias at 0.0005. Neither checkpoint was promoted, deployed, or run on the test set.

## Label and data audit

- `artifacts/eval/unified_weak_audit/report.json` found **zero malformed or missing labels** under a tolerance for six-decimal YOLO coordinate rounding.
- It found **one exact train/validation duplicate**: `train/steel__patches_101.jpg` and `val/steel__patches_105.jpg`. The isolated fine-tune dataset excludes the training copy (4036 rather than 4037 training images); validation and test are unchanged. The original saved model was trained with this duplicate, so its validation result retains that historical leakage.
- Contact sheets in `artifacts/eval/unified_weak_audit/` show that the screw thread and bent-lead labels are small and often cover separate mask components. They are generated mechanically from MVTec masks. An annotator should check whether the small boxes cover the full meaningful defect before changing labels. Crazing boxes are comparatively broad and diffuse.
- At a live display threshold of 0.25, a sampled baseline prediction review found screw thread side on only one of three validation examples and bent lead on neither of two. See `baseline_val_prediction_review.json` and the `*_gt_vs_pred.jpg` sheets. This sampling is diagnostic, not an AP estimate.
- The available local datasets do not add new, independently labeled examples of the same 21 product-specific classes. No copied images were counted as new distinct data. MVTec anomaly test examples were already repartitioned into this project's train/val/test, so this evaluation is not the canonical MVTec benchmark.

## Speed check

`artifacts/eval/unified_finetune_20261101/cpu_latency.json` measures ten representative validation images on this laptop with one CPU thread, including preprocessing and postprocessing: median **115 ms/image at 640** and **261 ms/image at 960**. Render hardware may differ. The 960-pixel candidate is slower and less accurate on validation.

## Decision and next data work

Keep the existing 640-pixel checkpoint and inference size. Do not use either fine-tune checkpoint or switch to 960. Before another training run, manually review and correct the small thread/lead labels, then collect genuinely new examples of screw defects and transistor bent leads with separate holdout images. Re-audit all splits for exact and near duplicates. Once labels and distinct examples are ready, run one small controlled experiment and select it on validation before a final test. A higher UI confidence threshold will not improve mAP.
