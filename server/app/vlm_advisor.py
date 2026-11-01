"""Local VLM second opinion for visible condition and reuse/recycle advice."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Literal

import cv2
import requests
from pydantic import BaseModel, Field, ValidationError

from server.app.config import Settings

Route = Literal["reuse_candidate", "inspect_or_repair", "recycle_candidate"]
Product = Literal["steel", "metal_nut", "screw", "transistor", "bottle", "other"]


class ProductObservation(BaseModel):
    product: Product
    visual_evidence: list[str] = Field(min_length=1, max_length=3)
    uncertainty: str = Field(max_length=250)


class VisualObservation(BaseModel):
    visible_condition: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(max_length=4)
    suggested_route: Route
    uncertainty: str = Field(max_length=350)


def rule_route(profile: str, detections: list[dict], *, uncertain_identity: bool = False) -> tuple[Route, str]:
    """Conservative demo policy; detector confidence is not a damage score."""
    if uncertain_identity:
        return "inspect_or_repair", "This product appearance differs from the defect model's training images; manual inspection is required."
    if not detections:
        return "reuse_candidate", "No named defect was found; functional and hygiene checks are still required."
    classes = {str(d.get("class") or d.get("class_name") or "").lower() for d in detections}
    if profile == "bottle" and classes & {"broken_large", "broken_small"}:
        return "recycle_candidate", "A broken bottle is not a reuse candidate."
    if max(int(d.get("severity") or 1) for d in detections) >= 4:
        return "recycle_candidate", "The detector marked a critical visual defect."
    return "inspect_or_repair", "A named defect needs a human check before reuse or recycling."


class LocalVLM:
    def __init__(self, settings: Settings) -> None:
        self.url = settings.vlm_url.rstrip("/")
        self.model = settings.vlm_model

    def status(self) -> dict:
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=2)
            response.raise_for_status()
            names = [model.get("name", "") for model in response.json().get("models", [])]
            ready = self.model in names or self.model.split(":")[0] in names
            return {"ready": ready, "model": self.model, "backend": "ollama-local", "message": "Local vision model ready" if ready else "Download the configured vision model"}
        except requests.RequestException:
            return {"ready": False, "model": self.model, "backend": "ollama-local", "message": "Local Ollama service is unavailable"}

    def identify_product(self, frame) -> dict:
        """Resolve only a narrow classifier near miss; never infer defect quality here."""
        if frame is None or frame.size == 0:
            raise ValueError("Product image is empty")
        height, width = frame.shape[:2]
        scale = min(1.0, 768 / max(height, width))
        if scale < 1:
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("Product image could not be encoded")
        prompt = (
            "Identify the main physical object in this image. Product must be exactly one of: "
            "steel, metal_nut, screw, transistor, bottle, other. A power transistor is a transistor. "
            "Use visible shape, not watermark or text. Choose other if the object is not clearly one of these families. "
            "For visual_evidence, name actual visible features such as body shape, leads, or threads; never copy the options list. "
            "Use an empty uncertainty string when the identity is clear. Do not assess defects or usability. "
            "Return concise JSON matching this schema: "
            + json.dumps(ProductObservation.model_json_schema())
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(encoded.tobytes()).decode("ascii")]}],
            "format": ProductObservation.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": 1536, "num_predict": 120},
            "keep_alive": "2m",
            "stream": False,
        }
        try:
            response = requests.post(f"{self.url}/api/chat", json=payload, timeout=120)
            response.raise_for_status()
            observation = ProductObservation.model_validate_json(response.json()["message"]["content"])
        except (requests.RequestException, KeyError, ValueError, ValidationError) as exc:
            raise RuntimeError(f"Local VLM product identification failed: {type(exc).__name__}") from exc
        return observation.model_dump()

    def advise(self, image_path: Path, profile: str, detections: list[dict], inspection_id: str, uncertain_identity: bool = False) -> dict:
        if not image_path.is_file():
            raise ValueError("Inspection evidence image is unavailable")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError("Inspection evidence image could not be decoded")
        height, width = frame.shape[:2]
        scale = min(1.0, 1024 / max(height, width))
        if scale < 1:
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise ValueError("Inspection evidence image could not be encoded")
        route, rule_reason = rule_route(profile, detections, uncertain_identity=uncertain_identity)
        context = [{"defect": d.get("class") or d.get("class_name"), "confidence": round(float(d.get("conf") or 0), 3), "severity": d.get("severity")} for d in detections[:12]]
        prompt = (
            "You are reviewing one industrial inspection photo. The supported product is " + profile.replace("_", " ") + ". "
            "The detector returned these named boxes: " + json.dumps(context) + ". "
            "Describe only visible condition. Do not invent defects, material composition, electrical function, hygiene, structural strength, or damaged-area percentages. "
            "If a feature cannot be verified from this image, state that uncertainty. "
            "Suggest one of reuse_candidate, inspect_or_repair, recycle_candidate as an advisory opinion. "
            "The final route is decided separately by deterministic inspection rules. Return concise JSON matching this schema: "
            + json.dumps(VisualObservation.model_json_schema())
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(encoded.tobytes()).decode("ascii")]}],
            "format": VisualObservation.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 220},
            "keep_alive": "2m",
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = requests.post(f"{self.url}/api/chat", json=payload, timeout=120)
            response.raise_for_status()
            observation = VisualObservation.model_validate_json(response.json()["message"]["content"])
        except (requests.RequestException, KeyError, ValueError, ValidationError) as exc:
            raise RuntimeError(f"Local VLM analysis failed: {type(exc).__name__}") from exc
        return {
            "inspection_id": inspection_id, "profile": profile, "model": self.model,
            "visible_condition": observation.visible_condition,
            "evidence": observation.evidence,
            "uncertainty": observation.uncertainty,
            "vlm_suggestion": observation.suggested_route,
            "recommendation": route,
            "rule_reason": rule_reason,
            "disagreement": observation.suggested_route != route,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "advisory_only": True,
        }
