"""Check whether the M2 Body classifier uses foreground or background cues.

This diagnostic uses only the frozen validation split. It does not tune the
checkpoint or threshold and does not read the M2 Test split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from train_m2_body_classifier import OUT, at_threshold, predict


class MaskedValidation(Dataset):
    def __init__(self, rows: list[tuple[str, int]], mode: str) -> None:
        self.rows = rows
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.rows[index]
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB")).copy()
        # The screw and head lie inside this fixed, deliberately generous ROI.
        x0, y0, x1, y1 = 80, 40, 250, 375
        if self.mode == "background_only":
            image[y0:y1, x0:x1] = 245
        elif self.mode == "foreground_only":
            foreground = image[y0:y1, x0:x1].copy()
            image[:] = 245
            image[y0:y1, x0:x1] = foreground
        return self.transform(Image.fromarray(image)), label


def main() -> None:
    manifest = json.loads((OUT / "split_manifest.json").read_text(encoding="utf-8"))
    rows = [(str(path), int(label)) for path, label in manifest["val"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(OUT / "best.pt", map_location=device, weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    threshold = checkpoint["operating_threshold"]
    result = {}
    for mode in ("original", "background_only", "foreground_only"):
        loader = DataLoader(MaskedValidation(rows, mode), batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=device == "cuda")
        scores, labels = predict(model, loader, device)
        result[mode] = {
            "auc": float(roc_auc_score(labels, scores)),
            "at_fixed_threshold": at_threshold(scores, labels, threshold),
        }
    (OUT / "shortcut_check_val.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
