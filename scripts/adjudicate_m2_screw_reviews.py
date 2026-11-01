"""Conservative visual correction of the first M2 Body annotation batch.

Preserves every operator review. The few class/box matches below are
provisional manual candidates, never automatic training labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from annotate_m2_screw import ROOT, load_rows, review_path, validate_review


# Pixel boxes on the 320x420 crops, inspected visually against MVTec
# screw_manipulated_front examples. All remaining operator boxes are excluded.
TIP_CANDIDATES = {
    "m2_body_ng_bari-40.jpg": [106, 93, 204, 154],
    "m2_body_ng_bari-129.jpg": [132, 118, 202, 166],
    "m2_body_ng_bari-334.jpg": [128, 91, 201, 155],
}
OUT = ROOT / "adjudication"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    decisions = []
    rows_by_image = {row["image"]: row for row in load_rows()}
    named_candidates = set()
    for image_name, row in rows_by_image.items():
        path = review_path(row)
        if not path.exists():
            continue
        source_bytes = path.read_bytes()
        review = json.loads(source_bytes)
        validate_review(row, review)
        file_name = Path(image_name).name
        entry = {
            "image": image_name,
            "source_review": str(path),
            "source_review_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "original_state": review["state"],
            "original_boxes": review["boxes"],
        }
        if file_name in TIP_CANDIDATES:
            named_candidates.add(file_name)
            entry.update({
                "decision": "provisional_class_and_box_correction",
                "proposed_boxes": [{"class_name": "screw_manipulated_front",
                                    "xyxy": TIP_CANDIDATES[file_name]}],
                "reason": "Visible deformation localized at screw front/tip; analogous to existing manipulated_front examples. Human validation still required before training.",
            })
        elif review["state"] == "labeled":
            entry.update({
                "decision": "exclude_from_21_class_training",
                "proposed_boxes": [],
                "reason": "The original class/box cannot be verified against an existing MVTec screw subtype and component-level box convention from this binary NG source.",
            })
        elif review["state"] == "unmapped":
            entry.update({"decision": "retain_unmapped", "proposed_boxes": [],
                          "reason": "No supported defect class selected."})
        else:
            entry.update({"decision": "clean_candidate_only", "proposed_boxes": [],
                          "reason": "Operator marked clean; separate domain review needed before adding a negative training image."})
        decisions.append(entry)
    if named_candidates != set(TIP_CANDIDATES):
        raise RuntimeError(f"Expected reviewed tip candidates missing: {set(TIP_CANDIDATES) - named_candidates}")

    report = {
        "policy": "Do not overwrite operator reviews or merge these provisional corrections into YOLO training.",
        "source_label_limit": "M2 Body supplies OK/NG image labels only, without defect subtype or pixel localization.",
        "reviewed_images": len(decisions),
        "unreviewed_images": len(rows_by_image) - len(decisions),
        "provisional_corrected_boxes": sum(len(item["proposed_boxes"]) for item in decisions),
        "excluded_original_labeled_images": sum(item["decision"] == "exclude_from_21_class_training" for item in decisions),
        "decisions": decisions,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    sheet = Image.new("RGB", (3 * 320, 500), "white")
    draw = ImageDraw.Draw(sheet)
    for i, entry in enumerate(item for item in decisions if item["proposed_boxes"]):
        with Image.open(ROOT / entry["image"]) as source:
            image = source.convert("RGB")
        overlay = ImageDraw.Draw(image)
        for box in entry["proposed_boxes"]:
            overlay.rectangle(box["xyxy"], outline="#16a34a", width=3)
        sheet.paste(ImageOps.fit(image, (300, 394)), (i * 320 + 10, 45))
        draw.text((i * 320 + 10, 10), Path(entry["image"]).stem[-9:], fill="black")
        draw.text((i * 320 + 10, 450), "provisional manipulated_front", fill="black")
    sheet.save(OUT / "provisional_tip_boxes.jpg", quality=92)
    print(json.dumps({key: value for key, value in report.items() if key != "decisions"}, indent=2))


if __name__ == "__main__":
    main()
