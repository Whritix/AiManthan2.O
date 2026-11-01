"""Summarize saved M2 screw reviews and make an annotated QC sheet."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageOps

from annotate_m2_screw import ROOT, load_rows, review_path, validate_review


def main() -> None:
    rows = load_rows()
    annotated = []
    states = Counter()
    classes = Counter()
    area_fractions = []
    for row in rows:
        path = review_path(row)
        if not path.exists():
            states["unreviewed"] += 1
            continue
        review = json.loads(path.read_text(encoding="utf-8"))
        validate_review(row, review)
        states[review["state"]] += 1
        if review["boxes"]:
            annotated.append((row, review))
        for box in review["boxes"]:
            classes[box["class_name"]] += 1
            x0, y0, x1, y1 = box["xyxy"]
            w, h = row["crop_size"]
            area_fractions.append((x1 - x0) * (y1 - y0) / (w * h))

    outputs = []
    for page in range((len(annotated) + 11) // 12):
        examples = annotated[page * 12:(page + 1) * 12]
        sheet = Image.new("RGB", (4 * 300, 3 * 410), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (row, review) in enumerate(examples):
            with Image.open(ROOT / row["image"]) as source:
                image = source.convert("RGB")
            overlay = ImageDraw.Draw(image)
            for box in review["boxes"]:
                overlay.rectangle(box["xyxy"], outline="#e11d48", width=3)
            tile = ImageOps.fit(image, (280, 365))
            x, y = (index % 4) * 300, (index // 4) * 410
            sheet.paste(tile, (x + 10, y + 32))
            label = review["boxes"][0]["class_name"].removeprefix("screw_")
            draw.text((x + 10, y + 6), f"{Path(row['image']).stem[-8:]} {label}"[:39], fill="black")
        output = ROOT / f"review_qc_page_{page + 1}.jpg"
        sheet.save(output, quality=92)
        outputs.append(str(output))
    report = {
        "states": dict(states),
        "classes": dict(classes),
        "box_count": len(area_fractions),
        "median_box_area_fraction_of_crop": median(area_fractions) if area_fractions else None,
        "boxes_over_2_percent_of_crop": sum(x > 0.02 for x in area_fractions),
        "qc_sheets": outputs,
    }
    (ROOT / "review_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
