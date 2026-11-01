#!/usr/bin/env python3
"""Render a 'Technical Approach' slide for the Indore QA Cell project.

Style mirrors a hackathon technical-approach poster: title bar, technology
grid, dual-layer architecture, process flow, and working-visual mockups.
Everything is drawn programmatically with Pillow (no external assets).
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "docs" / "technical_approach.png"
W, H = 1920, 1080

GREEN = (15, 122, 61)
DARK = (28, 32, 36)
GRAY = (110, 118, 126)
LIGHT = (246, 248, 250)
BLUE_BG = (219, 234, 254)
BLUE_BR = (59, 130, 246)
PURPLE_BG = (237, 233, 254)
PURPLE_BR = (139, 92, 246)
AMBER = (245, 158, 11)
RED = (220, 38, 38)
TEAL = (13, 148, 136)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["seguisb.ttf", "segoeuib.ttf"] if bold else ["segoeui.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)


F_TITLE = font(56, True)
F_H2 = font(34, True)
F_H3 = font(28, True)
F_BODY = font(21)
F_SMALL = font(17)
F_TINY = font(14)
F_CHIP = font(20, True)


def text(draw: ImageDraw.ImageDraw, xy, s, f, fill=DARK, anchor="la"):
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def center(draw, cx, y, s, f, fill=DARK):
    draw.text((cx, y), s, font=f, fill=fill, anchor="ma")


def rounded(draw, box, r, fill, outline=None, width=2):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def arrow(draw, x1, y1, x2, y2, color=DARK, width=4, head=14):
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    pts = [
        (x2, y2),
        (x2 - head * math.cos(ang - 0.45), y2 - head * math.sin(ang - 0.45)),
        (x2 - head * math.cos(ang + 0.45), y2 - head * math.sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=color)


def chip(draw, cx, cy, label, color, glyph, glyph_color=(255, 255, 255)):
    size = 74
    x0, y0 = cx - size // 2, cy - size // 2
    rounded(draw, (x0, y0, x0 + size, y0 + size), 14, color)
    text(draw, (cx, y0 + size // 2), glyph, font(24, True), glyph_color, "mm")
    text(draw, (cx, y0 + size + 16), label, F_CHIP, DARK, "ma")


def camera_icon(draw, cx, cy, color=DARK):
    draw.rounded_rectangle((cx - 34, cy - 20, cx + 34, cy + 26), 8, (255, 255, 255), color, 3)
    draw.polygon([(cx + 34, cy - 4), (cx + 52, cy - 14), (cx + 52, cy + 20), (cx + 34, cy + 10)], fill=color)
    draw.ellipse((cx - 16, cy - 12, cx + 16, cy + 18), outline=color, width=3)
    draw.ellipse((cx - 7, cy - 3, cx + 7, cy + 9), fill=color)


def chip_icon(draw, cx, cy, color=DARK):
    draw.rounded_rectangle((cx - 30, cy - 30, cx + 30, cy + 30), 6, (255, 255, 255), color, 3)
    draw.rounded_rectangle((cx - 14, cy - 14, cx + 14, cy + 14), 4, color)
    for i in range(-1, 2):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x0 = cx + i * 12
            draw.line((x0, cy - 30, x0, cy - 40), fill=color, width=3)
            draw.line((x0, cy + 30, x0, cy + 40), fill=color, width=3)
            y0 = cy + i * 12
            draw.line((cx - 30, y0, cx - 40, y0), fill=color, width=3)
            draw.line((cx + 30, y0, cx + 40, y0), fill=color, width=3)


def db_icon(draw, cx, cy, color=DARK):
    draw.ellipse((cx - 30, cy - 34, cx + 30, cy - 14), fill=color)
    draw.rectangle((cx - 30, cy - 24, cx + 30, cy + 24), fill=color)
    draw.ellipse((cx - 30, cy + 14, cx + 30, cy + 34), fill=color)
    for dy in (-12, 0, 12):
        draw.arc((cx - 30, cy + dy - 10, cx + 30, cy + dy + 10), 0, 180, fill=(255, 255, 255), width=3)


def monitor_icon(draw, cx, cy, color=DARK):
    draw.rounded_rectangle((cx - 44, cy - 34, cx + 44, cy + 18), 6, (255, 255, 255), color, 3)
    draw.line((cx - 12, cy + 18, cx - 16, cy + 34), fill=color, width=4)
    draw.line((cx + 12, cy + 18, cx + 16, cy + 34), fill=color, width=4)
    draw.line((cx - 24, cy + 34, cx + 24, cy + 34), fill=color, width=4)
    for i, w in enumerate((44, 30, 36)):
        draw.line((cx - 30, cy - 18 + i * 12, cx - 30 + w, cy - 18 + i * 12), fill=color, width=3)


def main() -> None:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # Header
    d.rectangle((0, 0, W, 82), fill=GREEN)
    center(d, W // 2, 10, "TECHNICAL APPROACH", F_TITLE, "white")
    d.rectangle((0, 82, W, 88), fill=(9, 92, 46))

    # ---------------- Left: Technologies Involved ----------------
    d.ellipse((300, 104, 780, 176), outline=DARK, width=3, fill="white")
    center(d, 540, 122, "Technologies Involved", F_H2)

    d.rectangle((56, 198, 940, 626), outline=DARK, width=4)
    techs = [
        ("Python", (48, 105, 160), "Py"),
        ("PyTorch", (238, 76, 44), "PT"),
        ("YOLO11s", (23, 30, 38), "YOLO"),
        ("OpenCV", (32, 42, 68), "CV"),
        ("FastAPI", (5, 150, 105), "FA"),
        ("React.js", (32, 120, 160), "Re"),
        ("SQLite", (0, 102, 153), "SQL"),
        ("ONNX Runtime", (109, 76, 190), "ONNX"),
        ("GitHub", (24, 26, 28), "GH"),
        ("Docker", (13, 118, 189), "DK"),
        ("NumPy", (60, 90, 160), "NP"),
        ("Uvicorn", (70, 60, 120), "UV"),
    ]
    for i, (label, color, glyph) in enumerate(techs):
        row, col = divmod(i, 4)
        cx = 160 + col * 210
        cy = 268 + row * 128
        chip(d, cx, cy, label, color, glyph)

    # ---------------- Left: Process Flow ----------------
    d.rectangle((56, 652, 940, 1060), outline=DARK, width=4)
    text(d, (76, 666), "Process Flow / Architecture", F_H3)

    steps = [
        ("Steel frame\ncaptured", camera_icon),
        ("YOLO11s + gate\ndefect detection", chip_icon),
        ("Fusion decision\n+ SQLite WAL", db_icon),
        ("Dashboard, audit\n& MES webhook", monitor_icon),
    ]
    step_w, gap, y0, y1 = 196, 16, 724, 1010
    for i, (label, icon) in enumerate(steps):
        x0 = 76 + i * (step_w + gap)
        rounded(d, (x0, y0, x0 + step_w, y1), 12, LIGHT, GRAY, 2)
        icon(d, x0 + step_w // 2, y0 + 74)
        for j, line in enumerate(label.split("\n")):
            center(d, x0 + step_w // 2, y0 + 150 + j * 28, line, F_BODY)
        if i < 3:
            arrow(d, x0 + step_w + 2, (y0 + y1) // 2, x0 + step_w + gap - 2, (y0 + y1) // 2)

    # ---------------- Right: Implementation Methodology ----------------
    d.ellipse((1120, 104, 1740, 176), outline=DARK, width=3, fill="white")
    center(d, 1430, 122, "Implementation Methodology", F_H2)

    d.rectangle((980, 198, 1864, 626), outline=DARK, width=4)
    center(d, 1422, 212, "DEVELOPMENT WORKFLOW & ARCHITECTURE", F_H3)
    center(d, 1422, 252, "Dual-layer system: Edge AI + Server Analytics", F_BODY, GRAY)

    rounded(d, (1006, 292, 1408, 596), 14, BLUE_BG, BLUE_BR, 3)
    center(d, 1207, 306, "EDGE AI LAYER", font(22, True), (30, 64, 175))
    center(d, 1207, 338, "(RTX 4060 - On-device inference)", F_SMALL, GRAY)
    edge_items = [
        "YOLO11s @ 640, FP16",
        "5 defect classes",
        "PatchCore novelty gate",
        "12.6 ms p50 inference",
        "Immediate PASS / REVIEW / FAIL",
    ]
    for i, item in enumerate(edge_items):
        text(d, (1032, 380 + i * 38), "\u2022 " + item, F_BODY)

    rounded(d, (1456, 292, 1858, 596), 14, PURPLE_BG, PURPLE_BR, 3)
    center(d, 1657, 306, "SERVER ANALYTICS LAYER", font(22, True), (91, 33, 182))
    center(d, 1657, 338, "(FastAPI - SQLite - Dashboard)", F_SMALL, GRAY)
    srv_items = [
        "SQLite WAL audit trail",
        "Operator review UI",
        "Signed MES webhook (HMAC)",
        "Model version + SHA-256",
        "Operating-point scan",
    ]
    for i, item in enumerate(srv_items):
        text(d, (1482, 380 + i * 38), "\u2022 " + item, F_BODY)

    arrow(d, 1414, 444, 1450, 444, GRAY, 3, 12)
    center(d, 1432, 452, "sync", font(12), GRAY)

    # ---------------- Right: Working Visuals ----------------
    d.rectangle((980, 652, 1864, 1060), outline=DARK, width=4)
    text(d, (1000, 666), "Indore QA Cell: Working Visuals", F_H3)

    panels = [(1006, 724, 1272, 1032), (1294, 724, 1560, 1032), (1582, 724, 1848, 1032)]
    titles = ["Live Inspection", "Review Queue", "Model & Metrics"]
    for (x0, y0, x1, y1), title in zip(panels, titles):
        rounded(d, (x0, y0, x1, y1), 16, (18, 22, 28))
        rounded(d, (x0 + 8, y0 + 8, x1 - 8, y1 - 8), 12, (248, 250, 252))
        d.rectangle((x0 + 8, y0 + 8, x1 - 8, y0 + 52), fill=(15, 122, 61))
        text(d, (x0 + 22, y0 + 20), title, font(18, True), "white")
        if title == "Live Inspection":
            d.ellipse((x0 + 22, y0 + 62, x0 + 38, y0 + 78), fill=RED)
            text(d, (x0 + 44, y0 + 60), "LIVE  |  detector_best", F_SMALL, DARK)
            text(d, (x0 + 22, y0 + 86), "crazing 0.41", F_TINY, AMBER)
            rounded(d, (x0 + 22, y0 + 104, x1 - 22, y0 + 196), 8, (226, 232, 240))
            d.rectangle((x0 + 34, y0 + 118, x0 + 170, y0 + 182), outline=AMBER, width=3)
            rounded(d, (x0 + 22, y0 + 206, x1 - 22, y0 + 242), 8, (220, 252, 231))
            text(d, (x0 + 36, y0 + 214), "VERDICT: REVIEW", font(17, True), (22, 101, 52))
            text(d, (x0 + 22, y0 + 250), "latency 12.6 ms  |  79 FPS", F_SMALL, GRAY)
            text(d, (x0 + 22, y0 + 274), "conf 0.25  |  NMS 0.6", F_SMALL, GRAY)
        elif title == "Review Queue":
            rows = [("crazing", "0.41", AMBER), ("inclusion", "0.62", TEAL), ("scratches", "0.55", BLUE_BR)]
            for i, (name, conf, color) in enumerate(rows):
                yy = y0 + 64 + i * 54
                rounded(d, (x0 + 18, yy, x1 - 18, yy + 42), 8, (241, 245, 249))
                d.ellipse((x0 + 30, yy + 12, x0 + 48, yy + 30), fill=color)
                text(d, (x0 + 58, yy + 10), name, F_SMALL, DARK)
                text(d, (x1 - 26, yy + 10), conf, font(17, True), color, "ra")
            rounded(d, (x0 + 18, y0 + 226, x1 - 18, y0 + 260), 8, (254, 243, 199))
            text(d, (x0 + 32, y0 + 233), "3 items awaiting disposition", F_SMALL, (120, 80, 10))
            text(d, (x0 + 18, y0 + 270), "pass_max 0.55  |  review_max 0.75", F_TINY, GRAY)
        else:
            rows = [
                ("overall mAP50", "0.7855"),
                ("crazing mAP50", "0.4490"),
                ("pitted mAP50", "0.8738"),
                ("good FPR", "11.0%"),
            ]
            for i, (k, v) in enumerate(rows):
                yy = y0 + 64 + i * 48
                text(d, (x0 + 20, yy), k, F_SMALL, GRAY)
                text(d, (x1 - 24, yy), v, font(17, True), DARK, "ra")
                d.line((x0 + 20, yy + 26, x1 - 24, yy + 26), fill=(226, 232, 240), width=1)
            text(d, (x0 + 20, y0 + 246), "SHA 7b694d7e...", F_TINY, GRAY)
            text(d, (x0 + 20, y0 + 268), "conf 0.25  |  640  |  rect=False", F_TINY, GRAY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=95)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
