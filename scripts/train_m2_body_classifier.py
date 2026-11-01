"""Train an isolated M2 screw Body OK/NG classifier and evaluate once on Test.

The 21-class detector is not read or changed. Source: Mendeley 2y6mcsyrpy v2.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/m2_screw_v2/extracted/BODY"
CROPS = ROOT / "datasets/m2_screw_v2/classification_crops"
OUT = Path(__file__).resolve().parents[1] / "artifacts/runs/m2_body_resnet18_v1"
CROP = (350, 450, 670, 870)  # x0, y0, x1, y1 on 1024x1280 BMP
LABELS = {"OK": 0, "NG": 1}
SEED = 42
BLOCK = 20
EPOCHS = 12
PATIENCE = 4
BATCH = 32


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def make_crops() -> None:
    for split, expected in (("Train", 1000), ("Test", 500)):
        for status in LABELS:
            sources = sorted((SOURCE / split / status).glob("*.BMP"))
            if len(sources) != expected:
                raise RuntimeError(f"Unexpected source count: {split}/{status}: {len(sources)}")
            output = CROPS / split / status
            output.mkdir(parents=True, exist_ok=True)
            for source in sources:
                target = output / f"{source.stem}.jpg"
                if target.exists():
                    continue
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (1280, 1024):
                    raise RuntimeError(f"Unreadable or wrong image size: {source}")
                x0, y0, x1, y1 = CROP
                if not cv2.imwrite(str(target), image[y0:y1, x0:x1],
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not save: {target}")
            if len(list(output.glob("*.jpg"))) != expected:
                raise RuntimeError(f"Crop count mismatch: {output}")
            print(f"Prepared {split}/{status}: {expected}", flush=True)


def split_rows() -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    train, val, test = [], [], []
    # IDs are grouped in contiguous blocks rather than randomly splitting adjacent images.
    for status, label in LABELS.items():
        paths = sorted((CROPS / "Train" / status).glob("*.jpg"))
        blocks: dict[int, list[Path]] = {}
        for path in paths:
            number = int(path.stem.split("-")[-1])
            blocks.setdefault((number - 1) // BLOCK, []).append(path)
        keys = sorted(blocks)
        if len(keys) != 50:
            raise RuntimeError(f"Expected 50 blocks for {status}, got {len(keys)}")
        val_blocks = set(random.Random(f"{SEED}:{status}").sample(keys, 10))
        for key, members in blocks.items():
            target = val if key in val_blocks else train
            target.extend((str(path), label) for path in members)
        test.extend((str(path), label) for path in sorted((CROPS / "Test" / status).glob("*.jpg")))
    if Counter(y for _, y in train) != {0: 800, 1: 800} or Counter(y for _, y in val) != {0: 200, 1: 200}:
        raise RuntimeError("Unexpected Train/val class balance")
    if Counter(y for _, y in test) != {0: 500, 1: 500}:
        raise RuntimeError("Unexpected Test balance")
    return train, val, test


class ScrewDataset(Dataset):
    def __init__(self, rows: list[tuple[str, int]], training: bool = False) -> None:
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            *( [transforms.RandomAffine(degrees=5, translate=(0.02, 0.02),
                                       scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.12, contrast=0.12,
                                       saturation=0.08)] if training else [] ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.rows[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), label


def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for images, target in loader:
            logits = model(images.to(device, non_blocking=True))
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels.extend(target.tolist())
    return np.asarray(scores), np.asarray(labels)


def at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float((tp / (tp + fn) + tn / (tn + fp)) / 2),
        "precision_ng": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall_ng": float(tp / (tp + fn)),
        "false_alarm_rate_ok": float(fp / (tn + fp)),
    }


def threshold_for_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.05) -> float:
    # Highest recall among thresholds satisfying the empirical validation FPR.
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    feasible = [float(t) for t in candidates if at_threshold(scores, labels, float(t))["false_alarm_rate_ok"] <= max_fpr]
    if not feasible:
        raise RuntimeError("No threshold satisfies requested validation FPR")
    return min(feasible)


def main() -> None:
    seed_everything()
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"Run directory exists; refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True)
    make_crops()
    train_rows, val_rows, test_rows = split_rows()
    manifest = {
        "source": "https://data.mendeley.com/datasets/2y6mcsyrpy/2",
        "source_archive_sha256": "e660e1865310715451687db9657af26fd00da1f2fa69f91b10f5de80e17c4022",
        "crop_source_xyxy": CROP,
        "seed": SEED, "block_size": BLOCK,
        "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "train": train_rows, "val": val_rows,
        "test_policy": "Source Body/Test was untouched by training, validation, checkpoint selection, and threshold calibration.",
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = 0 if __import__("os").name == "nt" else 4
    loader_args = {"batch_size": BATCH, "num_workers": workers,
                   "pin_memory": device == "cuda"}
    train_loader = DataLoader(ScrewDataset(train_rows, training=True), shuffle=True, **loader_args)
    val_loader = DataLoader(ScrewDataset(val_rows), shuffle=False, **loader_args)
    test_loader = DataLoader(ScrewDataset(test_rows), shuffle=False, **loader_args)

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_auc = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        model.train()
        losses = []
        for images, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device, non_blocking=True))
            loss = criterion(logits, target.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scores, labels = predict(model, val_loader, device)
        auc = float(roc_auc_score(labels, scores))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_auc": auc, "val_at_0_5": at_threshold(scores, labels, 0.5),
               "seconds": round(time.perf_counter() - start, 1)}
        history.append(row)
        print(f"epoch {epoch:02d}: loss={row['train_loss']:.4f} val_AUC={auc:.4f} "
              f"val_recall={row['val_at_0_5']['recall_ng']:.3f} "
              f"val_FPR={row['val_at_0_5']['false_alarm_rate_ok']:.3f} "
              f"time={row['seconds']}s", flush=True)
        if auc > best_auc + 0.0001:
            best_auc, best_epoch = auc, epoch
            torch.save({"state_dict": model.state_dict(), "classes": ["OK", "NG"],
                        "crop_source_xyxy": CROP, "image_size": [224, 224],
                        "best_epoch": epoch, "val_auc": auc}, OUT / "best.pt")
        elif epoch - best_epoch >= PATIENCE:
            print(f"Early stop after {epoch}; best epoch {best_epoch}", flush=True)
            break

    checkpoint = torch.load(OUT / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    val_scores, val_labels = predict(model, val_loader, device)
    operating_threshold = threshold_for_fpr(val_scores, val_labels)
    checkpoint["operating_threshold"] = operating_threshold
    torch.save(checkpoint, OUT / "best.pt")
    print(f"Selected validation threshold {operating_threshold:.6f} at <=5% FPR", flush=True)

    # Original Body/Test is evaluated only after checkpoint and threshold are frozen.
    test_start = time.perf_counter()
    test_scores, test_labels = predict(model, test_loader, device)
    test_seconds = time.perf_counter() - test_start
    report = {
        "task": "M2 screw Body image-level binary OK/NG classification",
        "model": "ImageNet-pretrained ResNet18",
        "device": device,
        "best_epoch": best_epoch,
        "validation_auc": best_auc,
        "validation_threshold_for_at_most_5_percent_fpr": operating_threshold,
        "validation_at_operating_threshold": at_threshold(val_scores, val_labels, operating_threshold),
        "test_auc": float(roc_auc_score(test_labels, test_scores)),
        "test_at_0_5": at_threshold(test_scores, test_labels, 0.5),
        "test_at_operating_threshold": at_threshold(test_scores, test_labels, operating_threshold),
        "test_images": len(test_rows),
        "test_inference_seconds_total": round(test_seconds, 3),
        "test_inference_ms_per_image": round(test_seconds * 1000 / len(test_rows), 3),
        "caveat": "Binary M2 Body accuracy is not 21-class detection mAP or evidence of transfer to other screw cameras.",
    }
    (OUT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "best.sha256").write_text(hashlib.sha256((OUT / "best.pt").read_bytes()).hexdigest() + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
