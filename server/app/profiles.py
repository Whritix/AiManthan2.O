"""Profile catalogue, gate quality, and dataset verification.

The manifest is the single source of truth for what the demo can replay.
Nothing here trusts a directory name: every item carries its own label.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from server.app.config import Settings
from server.app.schemas import ProfileInfo


def _quality(separation: float | None, *, profile_id: str) -> str:
    if separation is None:
        return "unknown"
    if profile_id == "steel":
        # Verified on held-out good/defect/unknown frames: negative separation.
        return "not_validated"
    if separation >= 1.5:
        return "strong"
    if separation >= 0.75:
        return "good"
    if separation >= 0.30:
        return "weak"
    if separation > 0.0:
        return "marginal"
    return "not_validated"


@dataclass
class Item:
    path: str
    class_name: str
    kind: str
    boxes: list[dict[str, Any]] = field(default_factory=list)
    image_size: list[int] = field(default_factory=lambda: [0, 0])

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()

    def public(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "class": self.class_name,
            "kind": self.kind,
            "boxes": self.boxes,
            "image_size": self.image_size,
        }


@dataclass
class Profile:
    id: str
    display_name: str
    subtitle: str
    detector: bool
    gate_model: str | None
    license: str
    classes: list[str]
    heldout_class: str | None
    counts: dict[str, Any]
    good: list[Item]
    defect: list[Item]
    unknown: list[Item]
    gate_separation: float | None
    gate_quality: str
    gate_trusted: bool
    gate_path: Path | None
    raw: dict[str, Any]

    def info(self) -> ProfileInfo:
        return ProfileInfo(
            id=self.id,
            display_name=self.display_name,
            subtitle=self.subtitle,
            detector=self.detector,
            gate_ready=self.gate_path is not None and self.gate_path.exists(),
            gate_model=self.gate_model,
            gate_separation=self.gate_separation,
            gate_quality=self.gate_quality,
            license=self.license,
            classes=self.classes,
            counts=self.counts,
            heldout_class=self.heldout_class,
        )

    def items(self, kind: str) -> list[Item]:
        if kind == "good":
            return self.good
        if kind == "defect":
            return self.defect
        if kind == "unknown":
            return self.unknown
        raise ValueError(f"unknown item kind: {kind}")


class ProfileRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._profiles: dict[str, Profile] = {}
        self._gate_summary: dict[str, Any] = {}
        self._report: DatasetReportCache | None = None
        self.reload()

    def reload(self) -> None:
        manifest_path = self.settings.profiles_manifest
        if not manifest_path.exists():
            raise FileNotFoundError(f"profile manifest missing: {manifest_path}")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary: dict[str, Any] = {}
        if self.settings.gate_summary_path.exists():
            summary = json.loads(self.settings.gate_summary_path.read_text(encoding="utf-8"))
        self._gate_summary = summary

        profiles: dict[str, Profile] = {}
        for pid, data in raw.items():
            entry = summary.get(pid, {})
            check = entry.get("check", {})
            separation = check.get("separation_unknown")
            if separation is None:
                separation = check.get("separation_defect")
            quality = _quality(separation, profile_id=pid)
            gate_model = data.get("gate_model")
            gate_path = self.settings.models_dir / gate_model if gate_model else None
            profiles[pid] = Profile(
                id=pid,
                display_name=data.get("display_name", pid),
                subtitle=data.get("subtitle", ""),
                detector=bool(data.get("detector")),
                gate_model=gate_model,
                license=data.get("license", "unknown"),
                classes=list(data.get("classes", [])),
                heldout_class=data.get("heldout_class"),
                counts=dict(data.get("counts", {})),
                good=[self._item(x, "good") for x in data.get("good", [])],
                defect=[self._item(x, "defect") for x in data.get("defect", [])],
                unknown=[self._item(x, "unknown") for x in data.get("unknown", [])],
                gate_separation=float(separation) if separation is not None else None,
                gate_quality=quality,
                gate_trusted=quality in {"strong", "good", "weak"} and pid != "steel",
                gate_path=gate_path,
                raw=data,
            )
        self._profiles = profiles
        self._report = None

    @staticmethod
    def _item(data: dict[str, Any], kind: str) -> Item:
        return Item(
            path=data["path"],
            class_name=data.get("class", kind),
            kind=data.get("kind", kind),
            boxes=list(data.get("boxes", [])),
            image_size=list(data.get("image_size", [0, 0])),
        )

    def get(self, profile_id: str | None = None) -> Profile:
        pid = profile_id or self.settings.profile
        if pid not in self._profiles:
            raise KeyError(f"unknown profile: {pid}")
        return self._profiles[pid]

    def all(self) -> list[Profile]:
        return list(self._profiles.values())

    def gate_summary(self) -> dict[str, Any]:
        return self._gate_summary

    def gate_entry(self, profile_id: str) -> dict[str, Any]:
        return self._gate_summary.get(profile_id, {})

    # ------------------------------------------------------------- verification
    def verify(self, force: bool = False) -> dict[str, Any]:
        if self._report is not None and not force and time.time() - self._report.at < 120:
            return self._report.payload
        payload = self._build_report()
        self._report = DatasetReportCache(payload, time.time())
        return payload

    def _build_report(self) -> dict[str, Any]:
        warnings: list[str] = []
        neu: dict[str, Any] = {}
        splits_path = self.settings.splits_path
        if splits_path.exists():
            splits = json.loads(splits_path.read_text(encoding="utf-8"))
            counts = splits.get("counts", {})
            split_detail: dict[str, Any] = {}
            all_names: set[str] = set()
            for split in ("train", "val", "test", "heldout"):
                meta = counts.get(split, {})
                files = list(meta.get("files", []))
                split_detail[split] = {
                    "count": meta.get("count", len(files)),
                    "per_class": meta.get("per_class", {}),
                    "missing": sum(1 for f in files if not self._neu_file(split, f).exists()),
                    "sha256": meta.get("files_sha256"),
                }
                if split != "heldout":
                    all_names.update(files)
            heldout = set(counts.get("heldout", {}).get("files", []))
            leaks = sorted(heldout & all_names)
            if leaks:
                warnings.append(f"HELDOUT LEAK: {len(leaks)} held-out files appear in train/val/test")
            heldout_class = splits.get("heldout_class")
            trained = set(splits.get("trained_classes", []))
            if heldout_class in trained:
                warnings.append(f"HELDOUT CLASS LEAK: {heldout_class} is listed as trained")
            neu = {
                "dataset": splits.get("dataset", "NEU-DET"),
                "source": splits.get("source"),
                "seed": splits.get("seed"),
                "classes": splits.get("class_order", []),
                "trained_classes": splits.get("trained_classes", []),
                "heldout_class": heldout_class,
                "splits": split_detail,
                "heldout_leak_free": not leaks,
                "heldout_leaks": leaks[:20],
                "total": sum(v["count"] for v in split_detail.values()),
            }
            if neu["total"] != 1800:
                warnings.append(f"NEU-DET total is {neu['total']}, expected 1,800")
        else:
            warnings.append(f"missing split manifest: {splits_path}")

        mvtec = self._mvtec_report()
        if mvtec.get("total_train_good") and mvtec["total_train_good"] < 3000:
            warnings.append("MVTec AD raw source looks incomplete")

        profiles = []
        for profile in self.all():
            gate = self.gate_entry(profile.id)
            check = gate.get("check", {})
            missing = sum(1 for item in profile.good + profile.defect + profile.unknown if not item.exists)
            if missing:
                warnings.append(f"{profile.id}: {missing} profile frames missing on disk")
            profiles.append(
                {
                    "id": profile.id,
                    "display_name": profile.display_name,
                    "detector": profile.detector,
                    "gate_quality": profile.gate_quality,
                    "gate_trusted": profile.gate_trusted,
                    "gate_separation": profile.gate_separation,
                    "gate_check": check,
                    "counts": profile.counts,
                    "missing": missing,
                    "license": profile.license,
                    "heldout_class": profile.heldout_class,
                }
            )

        if any(p["gate_quality"] == "not_validated" for p in profiles):
            warnings.append(
                "Novelty gate is not validated for steel: it scores defects below good. "
                "Unknown-defect routing is demonstrated on metal_nut / transistor / bottle."
            )

        return {
            "neu": neu,
            "mvtec": mvtec,
            "profiles": profiles,
            "warnings": warnings,
            "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _neu_file(self, split: str, name: str) -> Path:
        base = self.settings.data_dir / "neu-det"
        if split == "heldout":
            return base / "heldout" / "images" / name
        return base / "images" / split / name

    def _mvtec_report(self) -> dict[str, Any]:
        root = Path(r"C:\Games\Indore\mvtec_anomaly_detection")
        if not root.exists():
            return {"available": False, "root": str(root)}
        categories = sorted(p.name for p in root.iterdir() if p.is_dir())
        train_good = 0
        test_total = 0
        masks = 0
        for cat in categories:
            train_good += len(list((root / cat / "train" / "good").glob("*.png")))
            if (root / cat / "test").exists():
                test_total += sum(1 for p in (root / cat / "test").rglob("*.png"))
            if (root / cat / "ground_truth").exists():
                masks += sum(1 for p in (root / cat / "ground_truth").rglob("*.png"))
        return {
            "available": True,
            "root": str(root),
            "categories": categories,
            "category_count": len(categories),
            "total_train_good": train_good,
            "total_test": test_total,
            "total_masks": masks,
            "license": "CC BY-NC-SA 4.0",
        }


@dataclass
class DatasetReportCache:
    payload: dict[str, Any]
    at: float


def sampled_items(profile: Profile, kind: str, limit: int = 8) -> list[Item]:
    items: Iterable[Item] = profile.items(kind)
    return list(items)[:limit]
