# Sequential dataset replay before upload

Start the server with `INDORE_DETECTOR=unified`, `INDORE_SOURCE=dataset`, and `INDORE_SEQUENCE=mixed`. The Live inspection feed plays **10 steel samples, 10 metal nuts, 10 screws, 10 transistors, then 10 bottles**, and repeats. Within each ten-sample block it alternates defect and clean examples. The same 21-class YOLO checkpoint runs on every frame; the product label and source are shown in the feed. These are dataset stills replayed as a stream, not footage from a camera or a single recorded video.

An uploaded image or video immediately takes over the feed after automatic product identification. **REPLAY** returns to the sequential dataset stream. Replay detections and decisions are demonstration records; they should not be treated as independent evaluation metrics.

With other detector modes, the existing single-profile dataset replay remains in place because those modes do not support all five product families.
