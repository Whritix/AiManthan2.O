# Unified 21-class detector

## What was trained

This is a separate YOLO11s detection checkpoint trained from pretrained `yolo11s.pt` on the prepared combined dataset. It covers the 21 named defect classes in `artifacts/datasets/unified_defects/data.yaml`:

- Steel: crazing, inclusion, patches, pitted surface, scratches
- Metal nut: bent, color, flip, scratch
- Screw: manipulated front, scratch head, scratch neck, thread side, thread top
- Transistor: bent lead, cut lead, damaged case, misplaced
- Bottle: broken large, broken small, contamination

Training used 4,037 images; validation used 785; the internal test split used 789. The training set includes 420 synthetic crazing augmentations, which are variations of existing steel images rather than 420 new independent examples. Training ran for 100 epochs at 640 px on the RTX 4060. The best checkpoint was selected by validation fitness at epoch 82.

## Evaluation

| Metric | Internal test result |
|---|---:|
| Precision | 81.1% |
| Recall | 62.0% |
| mAP50 | 70.1% |
| mAP50–95 | 49.1% |

Per-class precision, recall, mAP50, mAP50–95, and test sample counts are in [unified_yolo11s_640_test.json](../artifacts/eval/unified_yolo11s_640_test.json). At mAP50, the five steel classes scored: crazing 43.6%, inclusion 82.9%, patches 91.8%, pitted surface 82.7%, and scratches 96.0%.

The metal-nut and screw classes have only 3–4 test images each; transistor classes have one test image each; bottle classes have 3–4 test images each. Their metrics can move sharply when a single detection changes. The split is internal and is not new production-camera footage. Several defect examples were reassigned from source anomaly-test folders into the prepared split, so these are not canonical source-dataset benchmark scores. Treat this as a hackathon prototype result, not a production accuracy guarantee.

The metrics report uses low confidence for mAP ranking. The listed per-class precision and recall are the best-F1 point on each test precision-recall curve; they are not the app's fixed live confidence setting. Do not tune thresholds on this test split.

## Use the model in the app

The candidate checkpoint is kept separate from `artifacts/models/detector_best.pt`, so the existing steel-only model remains preserved. The server now supports:

```powershell
$env:INDORE_DETECTOR = 'unified'
$env:YOLO_CONFIG_DIR = (Join-Path (Get-Location).Path 'Ultralytics')
python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000
```

For uploaded files, the existing product-family classifier selects steel, metal nut, screw, transistor, or bottle automatically. The unified detector then filters its 21 outputs for that profile and reports the familiar defect name; each box displays its model confidence. For example, a metal-nut profile displays `bent` and the percentage confidence. That percentage is confidence that the box/class prediction is correct, not the proportion of the object's surface that is defective.

To return to the current steel-only promoted detector, run the server with `INDORE_DETECTOR=real`. The combined candidate has not been promoted over that checkpoint.

## Files

- Dataset: `artifacts/datasets/unified_defects/`
- Dataset preparation notes: `docs/UNIFIED_DATASET_PREPARATION.md`
- Training run and best/last checkpoints: `artifacts/runs/unified_yolo11s_640/`
- Test evaluator: `scripts/evaluate_unified_detector.py`
- Full evaluation report: `artifacts/eval/unified_yolo11s_640_test.json`
- App integration: `server/app/config.py`, `server/app/detector.py`, `server/app/main.py`
