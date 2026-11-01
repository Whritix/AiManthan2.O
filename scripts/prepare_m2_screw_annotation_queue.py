"""Select varied M2 Body train images for manual defect-box annotation.

Creates a queue only. It deliberately writes no YOLO labels and never reads Test.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/m2_screw_v2/extracted/BODY/Train"
OUT = ROOT / "datasets/m2_screw_v2/annotation_queue"
CROP = (350, 450, 670, 870)  # x0, y0, x1, y1; 1024 x 1280 source frames
SEED = 42
QUOTAS = {"NG": 60, "OK": 20}


def select_group(status: str, count: int) -> list[tuple[Path, np.ndarray]]:
    paths = sorted((SOURCE / status).glob("*.BMP"))
    if len(paths) < count:
        raise RuntimeError(f"Insufficient {status} training images: {len(paths)}")
    vectors = []
    crops = []
    x0, y0, x1, y1 = CROP
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (1280, 1024):
            raise RuntimeError(f"Unreadable or unexpected image: {path}")
        crop = image[y0:y1, x0:x1]
        crops.append(crop)
        gray = cv2.cvtColor(cv2.resize(crop, (48, 64)), cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        vectors.append(cv2.resize(magnitude, (24, 32)).ravel())
    features = np.asarray(vectors, dtype=np.float32)
    reduced = PCA(n_components=32, random_state=SEED).fit_transform(features)
    cluster = MiniBatchKMeans(n_clusters=count, random_state=SEED, n_init=3,
                              batch_size=256).fit(reduced)
    chosen: set[int] = set()
    for center in cluster.cluster_centers_:
        distances = np.sum((reduced - center) ** 2, axis=1)
        for index in np.argsort(distances):
            if int(index) not in chosen:
                chosen.add(int(index))
                break
    if len(chosen) != count:
        raise RuntimeError(f"Expected {count} selections, got {len(chosen)}")
    return [(paths[index], crops[index]) for index in sorted(chosen)]


def main() -> None:
    if OUT.exists() and any(path.is_file() for path in OUT.rglob("*")):
        raise SystemExit(f"Queue exists; refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for status, count in QUOTAS.items():
        target = OUT / "images" / status
        target.mkdir(parents=True, exist_ok=True)
        for source, crop in select_group(status, count):
            dest = target / f"m2_body_{status.lower()}_{source.stem.lower()}.jpg"
            if not cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {dest}")
            rows.append({
                "image": str(dest.relative_to(OUT)).replace("\\", "/"),
                "source": str(source),
                "source_split": "Train",
                "status": status,
                "crop_xyxy_source_pixels": list(CROP),
                "crop_size": [CROP[2] - CROP[0], CROP[3] - CROP[1]],
                "image_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "annotation_state": "needs_review" if status == "NG" else "clean_candidate",
            })
    (OUT / "manifest.json").write_text(json.dumps({
        "source": "Mendeley M2 Screw Body v2",
        "seed": SEED,
        "selection": "Sobel edge features of fixed source crop; PCA 32; MiniBatchKMeans representatives",
        "source_policy": "Only Body/Train was read. Body/Test remains sealed.",
        "labels_created": 0,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    with (OUT / "annotation_template.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "status", "class_name", "x_min", "y_min", "x_max", "y_max", "reviewer_notes"])
        for row in rows:
            writer.writerow([row["image"], row["status"], "", "", "", "", "", ""])
    print(f"Prepared {len(rows)} cropped images at {OUT} (NG={QUOTAS['NG']}, OK={QUOTAS['OK']}).")


if __name__ == "__main__":
    main()
