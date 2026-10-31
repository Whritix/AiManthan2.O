"""PatchCore-lite novelty gate for the Indore inspection cell.

Design (deliberately small and auditable):
  * Frozen ImageNet ResNet-18 backbone, features from layer2 (1/8) and
    layer3 (1/16), concatenated at 1/8 resolution.
  * A memory bank of normal patch embeddings (default 4,000 vectors).
  * Score = 1 - max cosine similarity to the bank.
  * Frame score = max over 96x96 tiles of the tile's top-k patch mean.
  * Calibration on good-only frames turns raw distance into a z-score where
    z = 1.0 means "worse than 99% of known-good parts".

No gradient training. The bank is the model.
"""
from __future__ import annotations

import base64
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class AnomalyRegion:
    bbox: tuple[int, int, int, int]
    score: float
    area: int


@dataclass
class AnomalyResult:
    raw_score: float
    z: float
    is_anomaly: bool
    heatmap: np.ndarray  # uint8 BGR, same size as input frame
    heatmap_gray: np.ndarray  # float32, same size as input frame
    regions: list[AnomalyRegion] = field(default_factory=list)
    latency_ms: float = 0.0
    tile_scores: list[float] = field(default_factory=list)


class PatchCoreGate:
    def __init__(
        self,
        bank_path: str | Path | None = None,
        device: str | None = None,
        input_size: int = 256,
        tile_size: int = 96,
        grid: int = 4,
        top_k: int = 2,
        bank_vectors: int = 4000,
        backbone_name: str = "wide_resnet50_2",
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_size = input_size
        self.tile_size = tile_size
        self.grid = grid
        self.top_k = top_k
        self.bank_vectors = bank_vectors
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_name == "resnet50":
            from torchvision.models import ResNet50_Weights

            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif backbone_name in ("wide_resnet50_2", "wide_resnet50"):
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"unknown backbone: {backbone_name}")
        self.features = create_feature_extractor(
            backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
        ).to(self.device).eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.bank: torch.Tensor | None = None
        self.patch_p50 = 0.0
        self.patch_p99 = 1.0
        self.frame_p50 = 0.0
        self.frame_p99 = 1.0
        self.metadata: dict = {}
        if bank_path is not None:
            self.load(bank_path)

    # ------------------------------------------------------------------ bank
    @torch.no_grad()
    def extract_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """tiles: [N,3,H,W] normalized -> [N, P, 384] L2-normalized."""
        out = self.features(tiles.to(self.device))
        l2 = out["layer2"]
        l3 = F.interpolate(out["layer3"], size=l2.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([l2, l3], dim=1)  # [N, 384, h, w]
        feat = feat.flatten(2).transpose(1, 2)  # [N, P, 384]
        return F.normalize(feat, dim=-1)

    @staticmethod
    def _normalize_tile(bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(rgb.transpose(2, 0, 1))

    def _tile_positions(self) -> list[tuple[int, int]]:
        max_pos = self.input_size - self.tile_size
        if self.grid <= 1:
            return [(max_pos // 2, max_pos // 2)]
        return [
            (int(round(x)), int(round(y)))
            for y in np.linspace(0, max_pos, self.grid)
            for x in np.linspace(0, max_pos, self.grid)
        ]

    @torch.no_grad()
    def features_from_crops(self, paths: Sequence[str | Path], batch_size: int = 32) -> torch.Tensor:
        """Extract one-tile features from 96x96 crops (bank building)."""
        chunks: list[torch.Tensor] = []
        batch: list[torch.Tensor] = []
        for path in paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != self.tile_size or img.shape[1] != self.tile_size:
                img = cv2.resize(img, (self.tile_size, self.tile_size), interpolation=cv2.INTER_AREA)
            batch.append(self._normalize_tile(img))
            if len(batch) >= batch_size:
                chunks.append(self.extract_tiles(torch.stack(batch)).cpu())
                batch = []
        if batch:
            chunks.append(self.extract_tiles(torch.stack(batch)).cpu())
        if not chunks:
            raise RuntimeError("no readable crops for bank")
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def features_from_frames(self, paths: Sequence[str | Path], batch_tiles: int = 32) -> torch.Tensor:
        """Extract all tile features from full frames. Returns [N_frames, N_tiles, P, D]."""
        out: list[torch.Tensor] = []
        for path in paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            frame = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            tiles = [
                self._normalize_tile(frame[y : y + self.tile_size, x : x + self.tile_size])
                for x, y in self._tile_positions()
            ]
            feats = []
            for i in range(0, len(tiles), batch_tiles):
                feats.append(self.extract_tiles(torch.stack(tiles[i : i + batch_tiles])).cpu())
            out.append(torch.cat(feats, dim=0))
        if not out:
            raise RuntimeError("no readable frames")
        return torch.stack(out, dim=0)

    def build_bank(
        self,
        crop_paths: Sequence[str | Path] | None = None,
        frame_paths: Sequence[str | Path] | None = None,
        calib_frame_paths: Sequence[str | Path] | None = None,
        seed: int = 0,
        bank_vectors: int | None = None,
    ) -> None:
        """Build a memory bank and calibrate on held-out good frames.

        Use either `crop_paths` (one 96x96 normal crop per sample, steel) or
        `frame_paths` (full normal frames, tiled, MVTec profiles).
        """
        n = bank_vectors or self.bank_vectors
        if crop_paths:
            feats = self.features_from_crops(crop_paths)  # [N, P, D]
            pool = feats.reshape(-1, feats.shape[-1])
        elif frame_paths:
            feats = self.features_from_frames(frame_paths)  # [F, T, P, D]
            pool = feats.reshape(-1, feats.shape[-1])
        else:
            raise ValueError("supply crop_paths or frame_paths")

        g = torch.Generator().manual_seed(seed)
        if pool.shape[0] > n:
            idx = torch.randperm(pool.shape[0], generator=g)[:n]
            bank = pool[idx]
        else:
            bank = pool
        self.bank = F.normalize(bank.float(), dim=-1).to(self.device)
        self.metadata["bank_vectors"] = int(self.bank.shape[0])
        self.metadata["feature_dim"] = int(self.bank.shape[1])
        self.metadata["tile_size"] = self.tile_size
        self.metadata["grid"] = self.grid
        self.metadata["input_size"] = self.input_size

        if calib_frame_paths:
            self.calibrate(calib_frame_paths)
        else:
            # Fallback: calibrate on a slice of the bank itself.
            self.frame_p50 = 0.0
            self.frame_p99 = 1.0
            self.patch_p50 = 0.0
            self.patch_p99 = 1.0

    @torch.no_grad()
    def calibrate(self, frame_paths: Sequence[str | Path]) -> None:
        frame_scores: list[float] = []
        patch_scores: list[np.ndarray] = []
        for path in frame_paths:
            result = self.score_frame(cv2.imread(str(path), cv2.IMREAD_COLOR), calibrated=False)
            frame_scores.append(result.raw_score)
            patch_scores.append(result.heatmap_gray.ravel())
        if not frame_scores:
            raise RuntimeError("no calibration frames")
        fs = np.asarray(frame_scores, dtype=np.float32)
        ps = np.concatenate(patch_scores)
        self.frame_p50 = float(np.percentile(fs, 50))
        self.frame_p99 = float(np.percentile(fs, 99))
        self.patch_p50 = float(np.percentile(ps, 50))
        self.patch_p99 = float(np.percentile(ps, 99))
        if self.frame_p99 <= self.frame_p50 + 1e-6:
            self.frame_p99 = self.frame_p50 + 1e-6
        if self.patch_p99 <= self.patch_p50 + 1e-6:
            self.patch_p99 = self.patch_p50 + 1e-6

    # ------------------------------------------------------------------ infer
    @torch.no_grad()
    def _score_tiles(self, frame: np.ndarray) -> tuple[torch.Tensor, list[tuple[int, int]], np.ndarray]:
        frame_small = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        positions = self._tile_positions()
        tiles = [
            self._normalize_tile(frame_small[y : y + self.tile_size, x : x + self.tile_size])
            for x, y in positions
        ]
        feats = self.extract_tiles(torch.stack(tiles))  # [T, P, D]
        flat = feats.reshape(-1, feats.shape[-1])  # [T*P, D]
        sims = flat @ self.bank.T  # [T*P, B]
        dists = (1.0 - sims.max(dim=1).values).reshape(len(tiles), -1)
        return dists, positions, frame_small

    @torch.no_grad()
    def score_frame(self, frame: np.ndarray, calibrated: bool = True) -> AnomalyResult:
        t0 = time.perf_counter()
        if self.bank is None:
            raise RuntimeError("gate bank not loaded")
        if frame is None:
            raise ValueError("empty frame")
        dists, positions, frame_small = self._score_tiles(frame)  # [T,P]
        p = dists.shape[1]
        k = max(1, min(self.top_k, p))
        tile_scores = torch.topk(dists, k, dim=1).values.mean(dim=1)
        raw = float(tile_scores.max().item())

        # Stitch a high-resolution heatmap.
        side = int(round(math.sqrt(p)))
        patch_maps = dists.reshape(-1, side, side).cpu().numpy()
        step = (self.input_size - self.tile_size) / max(1, self.grid - 1) if self.grid > 1 else 0
        step = max(1.0, step / max(1, side - 1)) if side > 1 else self.tile_size
        canvas = np.zeros((self.input_size, self.input_size), np.float32)
        weight = np.zeros((self.input_size, self.input_size), np.float32)
        for (x, y), pm in zip(positions, patch_maps):
            h, w = pm.shape
            ph = self.tile_size / h
            pw = self.tile_size / w
            for yy in range(h):
                for xx in range(w):
                    y0, y1 = int(y + yy * ph), int(y + (yy + 1) * ph)
                    x0, x1 = int(x + xx * pw), int(x + (xx + 1) * pw)
                    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], pm[yy, xx])
                    weight[y0:y1, x0:x1] = 1.0
        canvas = cv2.GaussianBlur(canvas, (0, 0), 3.0)
        canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        if calibrated:
            z = float(np.clip((raw - self.frame_p50) / (self.frame_p99 - self.frame_p50), 0.0, 10.0))
            zmap = np.clip((canvas - self.patch_p50) / (self.patch_p99 - self.patch_p50), 0.0, 10.0)
        else:
            z = raw
            zmap = canvas
        heat_u8 = cv2.applyColorMap(
            np.clip(zmap / 3.0 * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )

        # Connected regions at patch z >= 1.0 (worse than 99% of good patches).
        mask = (zmap >= 1.0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        regions: list[AnomalyRegion] = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < max(24, int(0.0005 * mask.size)):
                continue
            x, y, w, h = (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            )
            score = float(zmap[y : y + h, x : x + w].max())
            regions.append(AnomalyRegion(bbox=(x, y, w, h), score=score, area=area))
        regions.sort(key=lambda r: r.score, reverse=True)

        return AnomalyResult(
            raw_score=raw,
            z=z,
            is_anomaly=bool(z >= 1.0),
            heatmap=heat_u8,
            heatmap_gray=zmap,
            regions=regions[:12],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tile_scores=[float(x) for x in tile_scores.cpu().tolist()],
        )

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        if self.bank is None:
            raise RuntimeError("nothing to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            bank=self.bank.detach().cpu().numpy().astype(np.float16),
            frame_p50=self.frame_p50,
            frame_p99=self.frame_p99,
            patch_p50=self.patch_p50,
            patch_p99=self.patch_p99,
            metadata=np.array([json.dumps(self.metadata)], dtype=object),
        )

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path), allow_pickle=True)
        bank = torch.from_numpy(data["bank"].astype(np.float32))
        self.bank = F.normalize(bank, dim=-1).to(self.device)
        self.frame_p50 = float(data["frame_p50"])
        self.frame_p99 = float(data["frame_p99"])
        self.patch_p50 = float(data["patch_p50"])
        self.patch_p99 = float(data["patch_p99"])
        try:
            self.metadata = json.loads(str(data["metadata"][0]))
        except Exception:
            self.metadata = {}

    def heatmap_data_url(self, result: AnomalyResult, quality: int = 80) -> str:
        ok, buf = cv2.imencode(".jpg", result.heatmap, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
