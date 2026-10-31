"""Evidence stills and hashes. One directory per UTC day."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from server.app.config import Settings
from server.app.schemas import Detection


def _encode(img: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return b""
    return buf.tobytes()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest() if data else ""


def _square_crop(frame: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    h, w = frame.shape[:2]
    if bbox is None:
        side = min(h, w)
        x = (w - side) // 2
        y = (h - side) // 2
    else:
        x, y, bw, bh = bbox
        cx, cy = x + bw / 2, y + bh / 2
        side = int(max(bw, bh) * 1.8)
        side = max(96, min(side, max(h, w)))
        x = int(cx - side / 2)
        y = int(cy - side / 2)
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + side > w:
            x = max(0, w - side)
        if y + side > h:
            y = max(0, h - side)
    crop = frame[y : y + min(side, h), x : x + min(side, w)]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)


class EvidenceStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.evidence_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        inspection_id: str,
        source_frame: np.ndarray,
        overlay_frame: np.ndarray,
        *,
        heatmap: np.ndarray | None = None,
        detections: Sequence[Detection] | None = None,
        regions: Sequence[Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        folder = self.root / day
        folder.mkdir(parents=True, exist_ok=True)
        base = folder / inspection_id

        source_bytes = _encode(source_frame, 90)
        overlay_bytes = _encode(overlay_frame, 86)
        (base.with_name(base.name + "_source.jpg")).write_bytes(source_bytes)
        (base.with_name(base.name + "_overlay.jpg")).write_bytes(overlay_bytes)

        heatmap_bytes = b""
        if heatmap is not None:
            heatmap_bytes = _encode(heatmap, 84)
            (base.with_name(base.name + "_heatmap.jpg")).write_bytes(heatmap_bytes)

        best_bbox = None
        if detections:
            best = max(detections, key=lambda d: (d.severity, d.conf))
            best_bbox = tuple(best.bbox)  # type: ignore[assignment]
        elif regions:
            best_bbox = tuple(regions[0].bbox)  # type: ignore[assignment]
        crop = _square_crop(source_frame, best_bbox)
        crop_bytes = _encode(crop, 90)
        (base.with_name(base.name + "_crop.jpg")).write_bytes(crop_bytes)

        meta_path = base.with_name(base.name + "_meta.json")
        meta_path.write_text(
            json.dumps(meta or {}, indent=2, default=str),
            encoding="utf-8",
        )

        rel = f"{day}/{inspection_id}"
        url_root = f"/media/evidence/{rel}"
        return {
            "image_url": f"{url_root}_source.jpg",
            "overlay_url": f"{url_root}_overlay.jpg",
            "heatmap_url": f"{url_root}_heatmap.jpg" if heatmap_bytes else None,
            "crop_url": f"{url_root}_crop.jpg",
            "image_sha": _sha(source_bytes),
            "overlay_sha": _sha(overlay_bytes),
            "heatmap_sha": _sha(heatmap_bytes),
            "crop_sha": _sha(crop_bytes),
            "meta_path": str(meta_path),
            "dir": str(folder),
        }
