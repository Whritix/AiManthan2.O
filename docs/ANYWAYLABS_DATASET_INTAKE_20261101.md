# AnywayLabs synthetic dataset intake — 1 November 2026

Source: `anywaylabs/synthetic-mvtec-ad-defect-detection` on Hugging Face, pinned to revision `554be045cd0ee0efe847bacbd61cf2e4c31992a5`. Downloaded only `screw/`, `transistor/`, and `README.md` to `C:\Games\Indore\datasets\anywaylabs_synthetic_mvtec_554be045`. The source dataset card lists CC BY 4.0. Its validation examples derive from MVTec AD, whose use terms also matter. No source files were uploaded or added to the app's training data.

Hugging Face's generic `load_dataset()` example accesses the repository's dataset metadata. For the actual YOLO `images/`, `labels/`, and `data.yaml` folders, `huggingface_hub.snapshot_download()` with `allow_patterns` was used. This public download did not require login. Downloaded selection: 1,221 files, about 411 MB.

## Intake counts

| Product | Synthetic train images / labels | Source val images / labels | Existing MVTec source IDs among source val |
| --- | ---: | ---: | ---: |
| Screw | 250 / 250 | 119 / 119 | 119 |
| Transistor | 200 / 200 | 40 / 40 | 40 |

Screw train has 50 boxes per class for manipulated front, scratch head, scratch neck, thread side, and thread top. Transistor train has 85 bent-lead boxes, 86 cut-lead boxes, 50 damaged-case boxes, and 50 misplaced boxes across 200 images. Every downloaded image has a corresponding label and the labels pass numeric/class checks with a 0.001 tolerance for three-decimal coordinate rounding. The full audit is `C:\Games\Indore\datasets\anywaylabs_synthetic_mvtec_554be045\intake_audit.json`.

## Compatibility finding

The synthetic labels use broader boxes for several screw classes. Median normalized box area for synthetic screw `thread_side` is **0.0987** of the image versus **0.00556** for current unified training labels, about **18 times larger**. Synthetic `scratch_head` and `scratch_neck` medians are about 0.052–0.054 versus about 0.0038–0.0039 in the current dataset. This is a label-convention mismatch, not proof either source is wrong. Direct merging would train on broader boxes while scoring against smaller component boxes.

The source validation files must not be added to training or treated as independent validation: their source IDs overlap the MVTec defect images already repartitioned into the project's train/val/test. The 450 synthetic training images are a candidate for a later controlled experiment only after box conventions are reconciled and source/near-duplicate checks are complete. The deployed checkpoint, current train/val/test, and known mAP values are unchanged.
