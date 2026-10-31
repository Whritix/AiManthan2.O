# Unified defect dataset preparation

## Output

Prepared dataset: [`artifacts/datasets/unified_defects`](../artifacts/datasets/unified_defects)

- `data.yaml` — YOLO dataset configuration with the 21 classes from [`data/defect_taxonomy.json`](../data/defect_taxonomy.json). Its root path is project-relative.
- `images/{train,val,test}` and `labels/{train,val,test}` — paired images and YOLO-format labels. Clean images have empty label files.
- `manifest.json` — source file, split, and per-class box provenance.

Counts:

| Split | Images | Boxes |
|---|---:|---:|
| Train | 4,037 | 3,701 |
| Validation | 785 | 621 |
| Test | 789 | 587 |
| **Total** | **5,611** | **4,909** |

## Label sources and conversion

- Steel boxes are copied from the existing NEU-DET labels and remapped from the existing five-class IDs into the unified class IDs. The frozen steel train/validation/test split is preserved.
- Clean steel and MVTec profile images are included with empty labels as negative examples. Bank, calibration, and held-out clean folders go to train, validation, and test, respectively.
- For metal nuts, screws, transistors, and bottles, each packaged defect crop is mapped by its defect name and sample index to its MVTec source image and pixel mask. Connected mask regions are converted to tight YOLO boxes. Defect images are deterministically re-split within each defect class at approximately 70/15/15 with seed 42.

## Evaluation caveat

The MVTec defect samples originally come from its anomaly benchmark test folders. They have been re-partitioned here to provide training examples for a new unified prototype, so this dataset's test scores must not be presented as results on the standard MVTec test split. Several classes remain extremely small, especially transistor defects. This dataset is ready for a first training experiment, but it is not enough to claim reliable production performance. Collect more real examples and reserve a fresh, independent test set before making that claim.

## Rebuild

From the project root, with the raw `mvtec_anomaly_detection` folder beside the project:

```powershell
python scripts/prepare_unified_defect_dataset.py
```

The script refuses to overwrite an existing non-empty output directory. It records exact source paths in the manifest for auditability.
