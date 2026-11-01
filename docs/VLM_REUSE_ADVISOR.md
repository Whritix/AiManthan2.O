# Local VLM reuse or recycle advisor

The Live inspection page has an **Analyze completed inspection** action. It sends one stored inspection source image, the auto-identified product, and confirmed YOLO detections to a local Ollama vision model (`qwen2.5vl:3b` by default). The VLM returns a structured description of visible condition, image evidence, uncertainty, and its own advisory route. The displayed recommendation is computed separately by deterministic product rules. A disagreement is marked for human review.

The three labels are **Reuse candidate**, **Inspect or repair**, and **Recycle candidate**. They are not certifications. The current demo policy marks clean images as reuse candidates, named noncritical defects for inspection, broken bottles for recycling, and severity-4 defects for recycling. Detector confidence is never interpreted as damage percentage. Image inspection cannot establish electrical function, hygiene, mechanical strength, or material composition.

## Local setup

1. Install [Ollama for Windows](https://ollama.com/download/windows).
2. Run `ollama pull qwen2.5vl:3b` and make sure the local service is running on `http://127.0.0.1:11434`.
3. Start the Inspectra API with the unified detector and upload one of the five supported products.
4. After the first completed inspection, use **Analyze completed inspection** on Live inspection.

The backend uses Ollama's image and JSON-schema chat API. `INDORE_VLM_URL` and `INDORE_VLM_MODEL` can override the local service URL and model name. The VLM runs only when requested on a completed evidence image; the live YOLO stream does not wait for it. The API caches the response for that inspection until a new upload.

The VLM may misdescribe an image, so its output is shown alongside the detector result and rule reason. This integration does not change the trained YOLO model or its mAP. To claim reuse/recycle accuracy, obtain human-labeled disposition data for each product and measure routing errors, especially unsafe reuse recommendations.
