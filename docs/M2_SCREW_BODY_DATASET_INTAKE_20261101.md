# M2 screw Body dataset intake — 1 November 2026

Source: [M2 screw defect detection dataset, version 2](https://data.mendeley.com/datasets/2y6mcsyrpy/2), DOI `10.17632/2y6mcsyrpy.2`, CC BY 4.0. The public `BODY.rar` file was downloaded to `C:\Games\Indore\datasets\m2_screw_v2\BODY.rar` without adding it to the repository or the Render image. Its size is 2,919,730,041 bytes and SHA-256 is `e660e1865310715451687db9657af26fd00da1f2fa69f91b10f5de80e17c4022`. Extraction with UnRAR completed successfully in the same isolated directory.

## Verified contents

| Folder | Images | Format and size | Label available |
| --- | ---: | --- | --- |
| Body/Train/OK | 1,000 | BMP, 1024 × 1280 | Image-level OK |
| Body/Train/NG | 1,000 | BMP, 1024 × 1280 | Image-level NG; `BARI` filename prefix |
| Body/Test/OK | 500 | BMP, 1024 × 1280 | Image-level OK |
| Body/Test/NG | 500 | BMP, 1024 × 1280 | Image-level NG; `BARI` filename prefix |

There are no annotation files, bounding boxes, masks, or defined mappings to the project's five screw defect classes. File SHA-256 checks found zero exact duplicates, including zero exact Train/Test duplicates. This does not prove that near-duplicate captures of the same screw or scene are absent. A train-only sample contact sheet is `C:\Games\Indore\datasets\m2_screw_v2\body_train_contact_sheet.jpg`; the machine-readable inventory is `body_audit.json` beside it.

The images show small screws in a fixed side-view camera setup. The `BARI` NG group has visibly altered thread shapes in some sampled images, but the exact defect region and whether it matches `screw_thread_side`, `screw_thread_top`, or another existing class cannot be established from the image-level label alone. The Head archive was not downloaded; the supplied dataset description indicates it also has binary OK/NG labels.

## Decision for the 21-class detector

Do **not** merge the NG images into YOLO training or assign `thread_side` from the folder name. YOLO detection training requires a named class and a box for each defect; using the whole screw as a defect box would conflict with the existing small mask-derived screw boxes. The M2 camera setup also differs from MVTec, so a direct merge could lower current validation AP. The existing checkpoint, 21-class dataset, validation/test splits, and reported metrics were not changed. This intake produced **no measured mAP improvement**.

To use this dataset for detector improvement, manually annotate a varied subset of **Train/NG** with precise defect boxes and one of the existing classes only where the class definition truly matches. Keep **Test** sealed; use Train/OK as empty-label negatives only after checking the camera/domain balance. Re-run the baseline on the same unchanged validation split, then compare one controlled fine-tune before promoting a checkpoint. Alternatively, this dataset supports a separate Body OK/NG classifier, whose accuracy would be a different metric from 21-class detection mAP.

An annotation queue is prepared at `C:\Games\Indore\datasets\m2_screw_v2\annotation_queue`: 60 NG and 20 OK crops, all from Body/Train. The reproducible selection script is `scripts/prepare_m2_screw_annotation_queue.py`. It uses edge-feature clustering to choose visually varied representatives, not known defect subtypes. The queue has a manifest, an annotation worksheet, and review instructions, but deliberately has no YOLO labels. No training run was started.

A local visual labeling tool is `scripts/annotate_m2_screw.py`. It displays the 80 crops, supports multiple tight class-labeled boxes per NG image, and permits explicit `unmapped` or `verified_clean` dispositions. It saves isolated JSON review files and validates box coordinates and class names with `--check`. The first check reported 80 unreviewed images and zero errors. Human review is still needed because the source has no defect localization or subtype labels. The tool does not export to the unified YOLO dataset or start training.

## Correction pass on the first reviewed batch

After 37 of 80 queue images were reviewed, `scripts/adjudicate_m2_screw_reviews.py` examined every saved box and preserved the original files. The report is `C:\Games\Indore\datasets\m2_screw_v2\annotation_queue\adjudication\report.json`. Three clearly deformed screw fronts received provisional `screw_manipulated_front` boxes with tighter extents; the visual proof sheet is `provisional_tip_boxes.jpg` in the same directory. The other 32 labeled images were excluded from 21-class training because their broad boxes or defect morphology do not reliably match a current MVTec class/box convention. One original review was already unmapped, one was marked clean, and 43 images were unreviewed. This is an assessment of annotation compatibility, not validated ground truth. Three provisional boxes are insufficient to justify a fine-tune, so the checkpoint and metrics remain unchanged.
