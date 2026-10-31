"""Industrial overlay renderer for the MJPEG canvas and evidence stills."""
from __future__ import annotations

import time
from typing import Any, Sequence

import cv2
import numpy as np

from server.app.fusion import CLASS_COLORS
from server.app.schemas import Detection, Verdict

FONT = cv2.FONT_HERSHEY_DUPLEX
MONO = cv2.FONT_HERSHEY_PLAIN


def _hex(name: str) -> tuple[int, int, int]:
    value = CLASS_COLORS.get(name, "#f59e0b").lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def _text_box(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    *,
    bg: tuple[int, int, int] = (18, 22, 28),
    alpha: float = 0.82,
    pad: int = 6,
    font: int = FONT,
) -> None:
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    x, y = org
    x = max(pad, min(x, img.shape[1] - tw - pad))
    y = max(th + pad, min(y, img.shape[0] - base - pad))
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def _fit_square(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == w:
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), (12, 16, 20), dtype=np.uint8)
    canvas[(size - nh) // 2 : (size - nh) // 2 + nh, (size - nw) // 2 : (size - nw) // 2 + nw] = resized
    return canvas


def render(
    frame: np.ndarray,
    detections: Sequence[Detection],
    verdict: Verdict,
    *,
    profile_id: str,
    engine: str,
    source: str,
    gate_z: float,
    gate_trusted: bool,
    gate_quality: str,
    heatmap_bgr: np.ndarray | None = None,
    heatmap_gray: np.ndarray | None = None,
    regions: Sequence[Any] | None = None,
    size: int = 800,
    part_label: str = "",
    warning: str = "",
    hud: bool = True,
) -> np.ndarray:
    canvas = _fit_square(frame, size)
    sh, sw = frame.shape[:2]
    fit_scale = size / max(sw, sh)
    fitted_w, fitted_h = max(1, round(sw * fit_scale)), max(1, round(sh * fit_scale))
    scale_x, scale_y = fitted_w / sw, fitted_h / sh
    offset_x, offset_y = (size - fitted_w) // 2, (size - fitted_h) // 2
    s = size / 800.0

    if heatmap_bgr is not None and gate_trusted:
        hm = cv2.resize(heatmap_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((size, size), np.uint8)
        if heatmap_gray is not None:
            gray = cv2.resize(heatmap_gray, (size, size), interpolation=cv2.INTER_LINEAR)
            mask = np.clip((gray - 0.55) / 1.4 * 255.0, 0, 255).astype(np.uint8)
        else:
            mask[:] = 90
        mask = cv2.GaussianBlur(mask, (0, 0), 2.5)
        alpha = (mask.astype(np.float32) / 255.0 * 0.42)[..., None]
        canvas = (canvas.astype(np.float32) * (1 - alpha) + hm.astype(np.float32) * alpha).astype(np.uint8)

    # Boxes, labels, severity chips.
    for det in detections:
        x, y, w, h = det.bbox
        x1 = int(np.clip(offset_x + x * scale_x, 0, size - 1))
        y1 = int(np.clip(offset_y + y * scale_y, 0, size - 1))
        x2 = int(np.clip(offset_x + (x + w) * scale_x, 1, size - 1))
        y2 = int(np.clip(offset_y + (y + h) * scale_y, 1, size - 1))
        color = _hex(det.class_name)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, max(2, round(2 * s)))
        confidence_pct = det.conf * 100.0 if det.conf <= 1.0 else det.conf
        product = profile_id.replace("_", " ").upper()
        defect = det.class_name.replace("_", " ").upper()
        label = f"{product} / {defect} | {confidence_pct:.1f}%"
        _text_box(canvas, label, (x1, max(y1 - 6, 18)), max(0.45, 0.52 * s), (255, 255, 255), pad=max(4, round(5 * s)))

    # Gate regions have no detector class or confidence, so label them as unknown.
    if gate_trusted and not detections and regions:
        for region in list(regions)[:3]:
            x, y, w, h = region.bbox
            x1 = int(np.clip(offset_x + x * scale_x, 0, size - 1))
            y1 = int(np.clip(offset_y + y * scale_y, 0, size - 1))
            x2 = int(np.clip(offset_x + (x + w) * scale_x, 1, size - 1))
            y2 = int(np.clip(offset_y + (y + h) * scale_y, 1, size - 1))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (11, 158, 245), max(2, round(3 * s)))
            score = float(getattr(region, "score", 0.0))
            _text_box(
                canvas,
                f"UNKNOWN ANOMALY | SCORE {score:.2f}",
                (x1, max(y1 - 6, 18)),
                max(0.42, 0.48 * s),
                (255, 255, 255),
                bg=(45, 30, 10),
                pad=max(4, round(5 * s)),
            )
        if regions:
            _text_box(
                canvas,
                f"NOVELTY  z={gate_z:.2f}",
                (int(18 * s), int(96 * s)),
                max(0.45, 0.54 * s),
                (11, 158, 245),
                pad=max(5, round(7 * s)),
            )

    # Burned-in truth goes on evidence stills only. The live stream keeps a
    # clean canvas because the HTML stage already labels source, fixture mode,
    # gate trust, part state, and verdict. Duplicating those on the pixels made
    # the stage noisy and hard to read.
    if hud:
        replay = source == "dataset"
        top_text = "DATASET REPLAY" if replay else ("CAMERA LIVE" if source == "camera" else "IMAGE INSPECTION" if source == "image" else "VIDEO REPLAY")
        _text_box(
            canvas,
            top_text,
            (int(18 * s), int(34 * s)),
            max(0.42, 0.46 * s),
            (255, 255, 255),
            bg=(30, 41, 59) if replay else (20, 83, 45),
            pad=max(4, round(6 * s)),
        )
        fixture = engine.startswith("fixture")
        if fixture:
            _text_box(
                canvas,
                "FIXTURE DETECTOR / GROUND-TRUTH BOXES",
                (int(18 * s), int(68 * s)),
                max(0.36, 0.40 * s),
                (80, 220, 255),
                bg=(45, 30, 10),
                pad=max(4, round(6 * s)),
            )
        elif not gate_trusted:
            _text_box(
                canvas,
                "NOVELTY GATE NOT VALIDATED",
                (int(18 * s), int(68 * s)),
                max(0.36, 0.40 * s),
                (120, 200, 255),
                bg=(45, 30, 10),
                pad=max(4, round(6 * s)),
            )

        # Bottom HUD: profile / part / wall clock.
        stamp = time.strftime("%H:%M:%S")
        hud_text = f"{profile_id.upper()}  {part_label}  {stamp}".strip()
        (tw, th), base = cv2.getTextSize(hud_text, MONO, 1.1 * s, 1)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, size - th - 22), (tw + 26, size), (8, 11, 15), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.putText(canvas, hud_text, (12, size - 12), MONO, 1.1 * s, (210, 220, 230), 1, cv2.LINE_AA)

    # Verdict stripe, kept small; the big badge lives in the UI.
    vcolors = {"pass": (74, 222, 128), "review": (11, 158, 245), "fail": (68, 68, 239)}
    vc = vcolors.get(verdict, (200, 200, 200))
    cv2.rectangle(canvas, (0, 0), (6, size), vc, -1)
    if hud:
        _text_box(
            canvas,
            verdict.upper(),
            (size - int(104 * s), int(34 * s)),
            max(0.48, 0.58 * s),
            vc,
            bg=(10, 14, 18),
            pad=max(5, round(7 * s)),
        )
    if warning and hud:
        _text_box(
            canvas,
            warning.upper()[:64],
            (int(18 * s), size - int(38 * s)),
            max(0.34, 0.38 * s),
            (120, 200, 255),
            bg=(40, 28, 12),
            pad=max(4, round(5 * s)),
        )
    return canvas


def encode_jpeg(img: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return b""
    return buf.tobytes()
