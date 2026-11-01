"""Local visual annotator for the isolated M2 screw Body training queue.

Run: python scripts/annotate_m2_screw.py
Check progress: python scripts/annotate_m2_screw.py --check
No labels are exported to the unified YOLO dataset by this tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[2] / "datasets/m2_screw_v2/annotation_queue"
REVIEWS = ROOT / "reviews"
CLASSES = (
    "screw_manipulated_front",
    "screw_scratch_head",
    "screw_scratch_neck",
    "screw_thread_side",
    "screw_thread_top",
)
SCALE = 1.4


def load_rows() -> list[dict]:
    data = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    if len(rows) != 80 or any(row["source_split"] != "Train" for row in rows):
        raise ValueError("Expected the frozen 80-image Train-only annotation queue")
    return rows


def review_path(row: dict) -> Path:
    return REVIEWS / (Path(row["image"]).stem + ".json")


def validate_review(row: dict, review: dict) -> None:
    if review.get("image") != row["image"] or review.get("source_sha256") != row["image_sha256"]:
        raise ValueError("Review image identity does not match the queue manifest")
    state = review.get("state")
    boxes = review.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    if row["status"] == "NG" and state not in {"labeled", "unmapped"}:
        raise ValueError("NG image must be labeled or marked unmapped")
    if row["status"] == "OK" and state != "verified_clean":
        raise ValueError("OK image must be explicitly verified clean")
    if state == "labeled" and not boxes:
        raise ValueError("Labeled image needs at least one defect box")
    if state != "labeled" and boxes:
        raise ValueError("Only labeled images may contain boxes")
    width, height = row["crop_size"]
    for box in boxes:
        if box.get("class_name") not in CLASSES:
            raise ValueError(f"Unsupported class: {box.get('class_name')}")
        x0, y0, x1, y1 = box["xyxy"]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Box outside crop: {box['xyxy']}")
        if (x1 - x0) * (y1 - y0) < 4:
            raise ValueError("Box is too small")


def check_progress() -> None:
    rows = load_rows()
    counts = Counter()
    errors = []
    for row in rows:
        path = review_path(row)
        if not path.exists():
            counts["unreviewed"] += 1
            continue
        try:
            review = json.loads(path.read_text(encoding="utf-8"))
            validate_review(row, review)
            counts[review["state"]] += 1
            counts["boxes"] += len(review["boxes"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path.name}: {exc}")
    print(json.dumps({"total": len(rows), "counts": dict(counts), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


class Annotator:
    def __init__(self) -> None:
        self.rows = load_rows()
        self.index = 0
        self.boxes: list[dict] = []
        self.state = "needs_review"
        self.dirty = False
        self.drag_start: tuple[int, int] | None = None
        self.preview: int | None = None
        self.window = tk.Tk()
        self.window.title("M2 screw defect annotation — Body/Train only")
        self.window.geometry("850x760")
        self.class_name = tk.StringVar(value="Choose a defect class")
        self.info = tk.StringVar()
        self.state_text = tk.StringVar()

        left = ttk.Frame(self.window, padding=12)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="M2 Body annotation", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(left, textvariable=self.info, wraplength=280).pack(anchor="w", pady=(12, 8))
        ttk.Label(left, textvariable=self.state_text, wraplength=280).pack(anchor="w", pady=(0, 12))
        ttk.Label(left, text="Class for the next box:").pack(anchor="w")
        chooser = ttk.Combobox(left, textvariable=self.class_name,
                               values=("Choose a defect class", *CLASSES), state="readonly", width=27)
        chooser.pack(anchor="w", pady=(4, 12))
        ttk.Label(left, text="NG means defective, but does not name the defect.\n"
                  "Choose a class only when it clearly matches.\n"
                  "Drag a tight box around the altered material, not the screw.\n"
                  "Otherwise mark unmapped.", wraplength=280).pack(anchor="w")
        ttk.Button(left, text="Undo last box", command=self.undo).pack(fill=tk.X, pady=(16, 4))
        ttk.Button(left, text="Mark NG as unmapped", command=self.mark_unmapped).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Verify OK as clean", command=self.mark_clean).pack(fill=tk.X, pady=4)
        ttk.Label(left, text="Reviewer note:").pack(anchor="w", pady=(14, 3))
        self.note = tk.Text(left, width=32, height=4)
        self.note.pack()
        self.note.bind("<KeyRelease>", self.note_changed)
        ttk.Button(left, text="Save review", command=self.save).pack(fill=tk.X, pady=(14, 4))
        ttk.Button(left, text="Previous", command=lambda: self.navigate(-1)).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Next", command=lambda: self.navigate(1)).pack(fill=tk.X, pady=4)
        self.canvas = tk.Canvas(self.window, width=448, height=588, bg="#eeeeee", highlightthickness=1)
        self.canvas.pack(side=tk.RIGHT, padx=14, pady=14)
        self.canvas.bind("<ButtonPress-1>", self.start_box)
        self.canvas.bind("<B1-Motion>", self.move_box)
        self.canvas.bind("<ButtonRelease-1>", self.finish_box)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.load_current()

    def current(self) -> dict:
        return self.rows[self.index]

    def load_current(self) -> None:
        row = self.current()
        self.boxes = []
        self.state = "needs_review"
        self.note.delete("1.0", tk.END)
        path = review_path(row)
        if path.exists():
            review = json.loads(path.read_text(encoding="utf-8"))
            validate_review(row, review)
            self.boxes = review["boxes"]
            self.state = review["state"]
            self.note.insert("1.0", review.get("note", ""))
        with Image.open(ROOT / row["image"]) as original:
            display = original.convert("RGB").resize((448, 588))
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.draw_boxes()
        self.info.set(f"{self.index + 1} / {len(self.rows)}  •  {row['status']}\n{row['image']}")
        self.dirty = False
        self.update_state()

    def update_state(self) -> None:
        self.state_text.set(f"Review: {self.state}  •  Boxes: {len(self.boxes)}"
                            + ("  •  Unsaved" if self.dirty else ""))

    def note_changed(self, _event: tk.Event) -> None:
        self.dirty = True
        self.update_state()

    def draw_boxes(self) -> None:
        self.canvas.delete("box")
        for box in self.boxes:
            x0, y0, x1, y1 = (v * SCALE for v in box["xyxy"])
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#e11d48", width=2, tags="box")
            self.canvas.create_text(x0, max(8, y0 - 8), text=box["class_name"],
                                    anchor="w", fill="#b91c1c", tags="box")

    def point(self, event: tk.Event) -> tuple[int, int]:
        width, height = self.current()["crop_size"]
        return max(0, min(width, round(event.x / SCALE))), max(0, min(height, round(event.y / SCALE)))

    def start_box(self, event: tk.Event) -> None:
        if self.current()["status"] != "NG":
            return
        if self.class_name.get() not in CLASSES:
            messagebox.showinfo("Choose a class", "Select a matching defect class before drawing a box, or mark this NG image unmapped.")
            return
        self.drag_start = self.point(event)

    def move_box(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        if self.preview is not None:
            self.canvas.delete(self.preview)
        x0, y0 = self.drag_start
        x1, y1 = self.point(event)
        self.preview = self.canvas.create_rectangle(x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE,
                                                    outline="#22c55e", width=2)

    def finish_box(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self.point(event)
        self.drag_start = None
        if self.preview is not None:
            self.canvas.delete(self.preview)
            self.preview = None
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if (x1 - x0) * (y1 - y0) < 4:
            return
        self.boxes.append({"class_name": self.class_name.get(), "xyxy": [x0, y0, x1, y1]})
        self.state = "labeled"
        self.dirty = True
        self.draw_boxes()
        self.update_state()

    def undo(self) -> None:
        if self.boxes:
            self.boxes.pop()
            self.state = "labeled" if self.boxes else "needs_review"
            self.dirty = True
            self.draw_boxes()
            self.update_state()

    def mark_unmapped(self) -> None:
        if self.current()["status"] == "NG":
            self.boxes = []
            self.state = "unmapped"
            self.dirty = True
            self.draw_boxes()
            self.update_state()

    def mark_clean(self) -> None:
        if self.current()["status"] == "OK":
            self.boxes = []
            self.state = "verified_clean"
            self.dirty = True
            self.update_state()

    def save(self) -> bool:
        row = self.current()
        review = {"image": row["image"], "source_sha256": row["image_sha256"],
                  "state": self.state, "boxes": self.boxes,
                  "note": self.note.get("1.0", tk.END).strip()}
        try:
            validate_review(row, review)
        except (ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Review incomplete", str(exc))
            return False
        REVIEWS.mkdir(exist_ok=True)
        path = review_path(row)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(review, indent=2), encoding="utf-8")
        temporary.replace(path)
        self.dirty = False
        self.update_state()
        return True

    def navigate(self, step: int) -> None:
        if self.dirty and not messagebox.askyesno("Unsaved review", "Discard unsaved changes?"):
            return
        self.index = (self.index + step) % len(self.rows)
        self.load_current()

    def close(self) -> None:
        if self.dirty and not messagebox.askyesno("Unsaved review", "Close without saving changes?"):
            return
        self.window.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate reviews and print progress")
    args = parser.parse_args()
    if args.check:
        check_progress()
    else:
        Annotator().window.mainloop()


if __name__ == "__main__":
    main()
