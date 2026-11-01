# Unified detector: weak-class audit and training gate

Date: 1 November 2026. The protocol in `artifacts/eval/unified_finetune_20261101/PREREGISTERED_COMPARISON.md` was saved before the additional GPU run. Baseline checkpoint SHA-256: `3e500b2d9b915f0d6c61433784376b9addfd7ae0abb4950f59623683e914388b`.

## Labels and masks

The existing unified labels were **not edited**. The prior format audit found no missing or malformed labels and one exact steel-patches train/validation duplicate. The isolated training list omits the training copy, but the saved baseline model was previously trained with it. Thus its validation score still has that historical leakage.

`artifacts/eval/unified_finetune_20261101/mask_box_audit.json` compares each generated MVTec box to the resized source mask. All checked labels match their connected components. Screw thread side has multiple components in 7 of 23 images. Its median component-box area is 2.17 times its component mask area, while the median whole-image union box would be 3.01 times total mask area (maximum 24.99). A blanket union-box relabel would include substantial clean area in some images. Bent lead has multiple components in only 1 of 10 images. These measurements do not prove that the labeling convention is ideal, but they do not justify automatic expansion either. Any future convention change must be applied consistently to train, validation, and test, followed by a new baseline evaluation on the changed validation labels.

## Baseline localization diagnostic

The diagnostic uses all 785 validation images with prediction confidence 0.001 and NMS IoU 0.6. It computes separate AP values at matching IoU 0.50, matching IoU 0.30, and the condition that the predicted center lies inside an unmatched ground-truth box of the same class. Its own AP implementation differs from the canonical Ultralytics evaluator, so **compare columns within this table, not their absolute values to the official mAP**. Report: `artifacts/eval/unified_finetune_20261101/baseline_matching_diagnostic.json`.

| Class | Validation images | Boxes | Diagnostic AP50 | AP30 | Center-hit AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Steel crazing | 45 | 107 | 0.312 | 0.634 | 0.704 |
| Screw manipulated front | 4 | 4 | 0.341 | 0.497 | 0.745 |
| Screw thread side | 3 | 5 | 0.106 | 0.223 | 0.247 |
| Screw thread top | 3 | 3 | 0.995 | 0.995 | 0.995 |
| Transistor bent lead | 2 | 3 | 0.545 | 0.545 | 0.545 |

Crazing and manipulated front show substantial localization sensitivity. Thread side remains weak at lower IoU and at center hit, so box extent alone is insufficient to explain it. Bent lead's diagnostic score does not change with matching criterion; its very small sample size is the main limitation. These are model/annotation diagnostics, not evidence that the ground-truth labels are wrong.

## Pre-registered smoke gate

One unchanged-label epoch initialized from the saved model. Isolated training set: 4036 images after excluding the exact duplicate; validation and test unchanged. Recipe: 640 pixels, batch 8, SGD `lr0=0.0001`, `lrf=1`, mosaic 1.0, scale 0.5, erasing 0, no rotation, capped warmup bias learning rate 0.0001, seed 42. Full args and checkpoint are under `artifacts/runs/unified_21class_smoke_20261101/`. The model used the same 21 classes. No sampling or label treatment was mixed into this run.

Official Ultralytics validation at confidence 0.001 and NMS IoU 0.6:

| Checkpoint | All-class mAP50 | All-class mAP50-95 | Five-steel-class macro AP50 |
| --- | ---: | ---: | ---: |
| Saved baseline | **0.7604** | **0.5155** | **0.7596** |
| One-epoch smoke | 0.7457 | 0.4961 | 0.7194 |

The pre-registered gate required **both** all-class and steel macro AP50 within 0.02 of baseline. All-class fell 0.0147, but steel macro fell **0.0402**. The gate failed; no treatment run followed. The smoke checkpoint was not promoted or tested.

## Sample size and paired uncertainty

The table uses official Ultralytics per-class AP50 point estimates. The final column is a 95% paired image-bootstrap interval for **smoke minus baseline** from the same diagnostic AP implementation as above, 200 resamples of validation images with seed 42. With only 2–4 positive images in a class, some resamples contain none, and the interval cannot establish a reliable gain. Full report: `artifacts/eval/unified_finetune_20261101/paired_bootstrap_smoke.json`.

| Class | Val images | Val boxes | Official AP50 baseline → smoke | Paired delta 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Steel crazing | 45 | 107 | 0.393 → 0.378 | −0.060 to +0.042 |
| Steel inclusion | 57 | 170 | 0.774 → 0.637 | −0.211 to −0.072 |
| Steel patches | 52 | 133 | 0.923 → 0.916 | −0.025 to +0.010 |
| Steel pitted surface | 45 | 68 | 0.764 → 0.746 | −0.104 to −0.017 |
| Steel scratches | 45 | 87 | 0.943 → 0.919 | −0.059 to −0.003 |
| Metal nut scratch | 3 | 6 | 0.483 → 0.455 | −0.497 to 0.000 |
| Screw manipulated front | 4 | 4 | 0.372 → 0.495 | 0.000 to +0.497 |
| Screw thread side | 3 | 5 | 0.085 → 0.032 | −0.271 to 0.000 |
| Screw thread top | 3 | 3 | 0.995 → 0.995 | 0.000 to 0.000 |
| Transistor bent lead | 2 | 3 | 0.755 → 0.679 | −0.028 to 0.000 |
| Transistor damaged case | 2 | 2 | 0.695 → 0.745 | 0.000 to +0.638 |

The paired diagnostic steel macro delta interval was **−0.067 to −0.028**. Its absolute AP values are not interchangeable with official Ultralytics mAP, but the paired direction supports the gate decision.

## Decision

Keep the existing 640-pixel baseline checkpoint. Do not change labels merely to raise AP, do not promote the smoke checkpoint, and do not open test for model selection. The next training recipe must first preserve the steel baseline on unchanged data. Without new independent screw/transistor examples, rare-class AP remains highly uncertain; oversampling or copies cannot increase the number of distinct appearances.
