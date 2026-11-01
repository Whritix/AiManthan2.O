# M2 screw Body OK/NG classifier — 1 November 2026

## Purpose and scope

An isolated image-level binary classifier was trained on [M2 screw defect detection dataset v2](https://data.mendeley.com/datasets/2y6mcsyrpy/2) (CC BY 4.0). It predicts **OK** or **NG** for the dataset's 1024 × 1280 Body camera image. It does not produce a defect box or one of the project's 21 defect classes. The existing unified YOLO checkpoint and its mAP remain unchanged.

Checkpoint: `artifacts/runs/m2_body_resnet18_v1/best.pt` (44,782,731 bytes; SHA-256 `ed434cdd92c28fac772f56b6cabaa0e190a0dc12551738cf3faaf9b5d18caefd`). Full training script: `scripts/train_m2_body_classifier.py`. The source `BODY.rar` hash and intake audit are in `M2_SCREW_BODY_DATASET_INTAKE_20261101.md`.

## Training and split

- Fixed central crop: source pixels `(350, 450, 670, 870)`, resized to 224 × 224.
- ImageNet-pretrained ResNet18, binary head, AdamW learning rate `1e-4`, moderate affine and color augmentation, seed 42.
- Source Body/Train: 1,600 training images (800 OK, 800 NG) and 400 validation images (200 each). Validation selected by 20-image contiguous ID blocks to reduce adjacent-frame overlap.
- Source Body/Test: 1,000 images (500 each). It was not used for training, checkpoint selection, or threshold calibration.
- Best checkpoint: epoch 6 selected by validation ROC AUC. Training stopped at epoch 10. Validation selected operating threshold: **0.781096** for at most 5% validation false alarms on OK images.
- Split list, history, weights hash, and exact metrics are in `artifacts/runs/m2_body_resnet18_v1/`.

## Frozen test result

| Metric on 1,000 M2 Body Test images | Result |
| --- | ---: |
| ROC AUC | 0.9846 |
| Accuracy at validation-selected threshold | 94.6% |
| NG recall | 96.4% (482/500) |
| NG precision | 93.1% (482/518) |
| OK false alarms | 7.2% (36/500) |
| Missed NG | 18/500 |

At the generic 0.5 threshold, false alarms were 33.2%, so the validation-selected threshold matters. The 7.2% test false-alarm rate is higher than the 5% validation target and must be reported as measured. Test GPU inference took 14.1 ms per image including data loading/batching on this laptop. A separate 25-image one-thread CPU check measured about 63 ms per image after decoding; it is **not** a Render benchmark.

## Domain check

A validation-only masking diagnostic gave AUC **0.9945** when retaining the screw region and replacing the outer background, but AUC **0.8275** when masking the screw and retaining the background. Thus the classifier uses useful screw cues **and** some background or acquisition cues. It must not be advertised as camera-independent or automatically applied to MVTec screws, other screw sizes, or arbitrary uploads. The fixed image-size guard alone does not prove the camera domain matches.

## Local app integration

- Open `/m2-body.html` on the local Inspectra server for the dedicated upload page.
- `GET /api/m2-body/status` reports availability and the frozen test metrics.
- `PUT /api/m2-body/classify` accepts raw BMP/JPEG/PNG/WEBP image bytes, rejects undecodable images and sizes other than 1024 × 1280, and returns OK/NG plus NG model probability and the fixed threshold.
- The model is optional and loaded lazily. `INDORE_M2_BODY_WEIGHTS` can point to the checkpoint. If absent on another machine or Render, the page shows unavailable. The route does not change the live YOLO inspection profile or database.

To reproduce training from the preserved source folder, use `python scripts/train_m2_body_classifier.py` only in a fresh output directory; the script refuses to overwrite the existing run. The endpoint smoke check is `python -m scripts.smoke_m2_body_api`.

This is a dataset-specific classifier prototype. Its 94.6% binary accuracy must not be described as 94.6% defect detection mAP or a measured improvement to the 21-class detector.
