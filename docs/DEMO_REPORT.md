# Inspection report and guided demo

The Input dashboard has recommended defect and clean examples for steel, metal nut, screw, transistor, and bottle. Selecting a sample uploads the actual image through the normal product identification and detector pipeline. The model status and measured processing speed are displayed with the demo cards. These curated examples are intended for presentation, not for estimating unbiased accuracy.

After an image or video upload, Live inspection shows an **Inspection report** with counts by class, PASS/REVIEW/FAIL decisions, and average inspection processing time. **Export CSV** contains one row per stored inspection detection, with inspection decision and evidence image URLs. **Report + images ZIP** contains the CSV, `summary.json`, and stored source, annotated, crop, and heatmap images where available. The newest upload is tracked across server restarts.

Reports use completed inspection records from the first image or video pass, up to 2,000 per upload. Counts are detection records rather than unique physical defects. Very short videos that never complete an inspection part may have no stored result. The report does not claim distance along a coil or frame-by-frame event tracking.
