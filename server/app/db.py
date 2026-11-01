"""SQLite persistence: inspections, defects, audit, outbox, labels, thresholds."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id TEXT PRIMARY KEY,
    ts_ms INTEGER NOT NULL,
    profile TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'closed',
    verdict TEXT NOT NULL,
    final_verdict TEXT,
    severity INTEGER DEFAULT 1,
    route TEXT,
    reason TEXT,
    anomaly_z REAL DEFAULT 0,
    anomaly_raw REAL DEFAULT 0,
    latency_total_ms REAL DEFAULT 0,
    frame_count INTEGER DEFAULT 0,
    detections_json TEXT DEFAULT '[]',
    ground_truth_kind TEXT,
    ground_truth_class TEXT,
    operator_disposition TEXT,
    operator_label TEXT,
    operator_note TEXT,
    reviewed_at_ms INTEGER,
    model_version TEXT,
    threshold_profile TEXT,
    image_sha TEXT,
    heatmap_sha TEXT,
    overlay_sha TEXT,
    crop_sha TEXT,
    image_url TEXT,
    heatmap_url TEXT,
    overlay_url TEXT,
    crop_url TEXT,
    warnings_json TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_inspections_ts ON inspections(ts_ms DESC);
CREATE INDEX IF NOT EXISTS idx_inspections_verdict ON inspections(final_verdict);
CREATE INDEX IF NOT EXISTS idx_inspections_profile ON inspections(profile);

CREATE TABLE IF NOT EXISTS defects (
    id TEXT PRIMARY KEY,
    inspection_id TEXT NOT NULL,
    class_name TEXT NOT NULL,
    conf REAL DEFAULT 0,
    bbox_json TEXT DEFAULT '[]',
    severity INTEGER DEFAULT 1,
    area_frac REAL DEFAULT 0,
    source TEXT DEFAULT 'detector',
    votes INTEGER DEFAULT 0,
    FOREIGN KEY(inspection_id) REFERENCES inspections(id)
);
CREATE INDEX IF NOT EXISTS idx_defects_inspection ON defects(inspection_id);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_id TEXT,
    detail_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts_ms DESC);

CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    next_attempt_ms INTEGER DEFAULT 0,
    last_error TEXT,
    delivered_at_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status, next_attempt_ms);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id TEXT NOT NULL,
    label TEXT NOT NULL,
    source TEXT NOT NULL,
    ts_ms INTEGER NOT NULL,
    embedding_json TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT,
    sha256 TEXT,
    metrics_json TEXT DEFAULT '{}',
    created_at_ms INTEGER NOT NULL,
    active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS threshold_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    pass_max REAL NOT NULL,
    review_max REAL NOT NULL,
    unknown_z REAL NOT NULL,
    conf_min REAL NOT NULL,
    active INTEGER DEFAULT 1,
    updated_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS mes_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    event_type TEXT,
    payload_json TEXT,
    signature TEXT,
    received_at_ms INTEGER NOT NULL
);
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            self._migrate()
        self.seed()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(inspections)").fetchall()}
        additions = {
            "image_url": "TEXT",
            "heatmap_url": "TEXT",
            "overlay_url": "TEXT",
            "crop_url": "TEXT",
            "warnings_json": "TEXT DEFAULT '[]'",
        }
        for name, ddl in additions.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE inspections ADD COLUMN {name} {ddl}")

    # ------------------------------------------------------------------ helpers
    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for key in ("detections_json", "detail_json", "payload_json", "metrics_json", "warnings_json", "bbox_json"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        return d

    def _rows(self, rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
        return [self._row(r) for r in rows if r is not None]

    def seed(self) -> None:
        now = _now_ms()
        with self.lock, self.conn:
            self.conn.execute(
                """INSERT OR IGNORE INTO threshold_profiles
                   (id, name, pass_max, review_max, unknown_z, conf_min, active, updated_at_ms)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
                ("threshold_v1", "Production v1", 0.55, 0.75, 0.75, 0.35, now),
            )
            self.conn.execute(
                """INSERT OR IGNORE INTO model_versions
                   (id, kind, name, path, sha256, metrics_json, created_at_ms, active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    "detector_auto",
                    "detector",
                    "YOLO11s NEU-DET",
                    "",
                    "",
                    json.dumps({"note": "auto-detected from artifacts"}),
                    now,
                ),
            )
            # Seed the audit log once so the UI is never empty on first run.
            cur = self.conn.execute("SELECT COUNT(*) AS n FROM audit")
            if cur.fetchone()["n"] == 0:
                self.conn.execute(
                    "INSERT INTO audit (ts_ms, actor, action, entity, entity_id, detail_json) VALUES (?,?,?,?,?,?)",
                    (now, "system", "system.start", "app", "indore", json.dumps({"version": "0.1.0"})),
                )

    # ------------------------------------------------------------------ inspections
    def create_inspection(self, data: dict[str, Any]) -> None:
        cols = [
            "id",
            "ts_ms",
            "profile",
            "source",
            "status",
            "verdict",
            "final_verdict",
            "severity",
            "route",
            "reason",
            "anomaly_z",
            "anomaly_raw",
            "latency_total_ms",
            "frame_count",
            "detections_json",
            "ground_truth_kind",
            "ground_truth_class",
            "model_version",
            "threshold_profile",
            "image_sha",
            "heatmap_sha",
            "overlay_sha",
            "crop_sha",
        ]
        values = []
        for c in cols:
            v = data.get(c)
            if c in ("detections_json",) and not isinstance(v, str):
                v = json.dumps(v or [])
            values.append(v)
        with self.lock, self.conn:
            self.conn.execute(
                f"INSERT OR REPLACE INTO inspections ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
                values,
            )

    def finalize_inspection(
        self,
        inspection: dict[str, Any],
        defects: list[dict[str, Any]],
        evidence: dict[str, str],
        outbox_events: list[dict[str, Any]],
    ) -> None:
        now = _now_ms()
        with self.lock, self.conn:
            self.conn.execute(
                """UPDATE inspections SET
                   status='closed', verdict=?, final_verdict=?, severity=?, route=?, reason=?,
                   anomaly_z=?, anomaly_raw=?, latency_total_ms=?, frame_count=?,
                   detections_json=?, image_sha=?, heatmap_sha=?, overlay_sha=?, crop_sha=?,
                   image_url=?, heatmap_url=?, overlay_url=?, crop_url=?, warnings_json=?
                   WHERE id=?""",
                (
                    inspection["final_verdict"],
                    inspection["final_verdict"],
                    inspection.get("severity", 1),
                    inspection.get("route"),
                    inspection.get("reason"),
                    inspection.get("anomaly_z", 0.0),
                    inspection.get("anomaly_raw", 0.0),
                    inspection.get("latency_total_ms", 0.0),
                    inspection.get("frame_count", 0),
                    json.dumps(inspection.get("detections", [])),
                    evidence.get("image_sha"),
                    evidence.get("heatmap_sha"),
                    evidence.get("overlay_sha"),
                    evidence.get("crop_sha"),
                    evidence.get("image_url"),
                    evidence.get("heatmap_url"),
                    evidence.get("overlay_url"),
                    evidence.get("crop_url"),
                    json.dumps(inspection.get("warnings", [])),
                    inspection["id"],
                ),
            )
            self.conn.execute("DELETE FROM defects WHERE inspection_id=?", (inspection["id"],))
            for d in defects:
                self.conn.execute(
                    """INSERT OR REPLACE INTO defects
                       (id, inspection_id, class_name, conf, bbox_json, severity, area_frac, source, votes)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        d["id"],
                        inspection["id"],
                        d.get("class_name") or d.get("class") or "unknown",
                        d.get("conf", 0.0),
                        json.dumps(d.get("bbox", [])),
                        d.get("severity", 1),
                        d.get("area_frac", 0.0),
                        d.get("source", "detector"),
                        d.get("votes", 0),
                    ),
                )
            self.conn.execute(
                "INSERT INTO audit (ts_ms, actor, action, entity, entity_id, detail_json) VALUES (?,?,?,?,?,?)",
                (
                    now,
                    "system",
                    "inspection.closed",
                    "inspection",
                    inspection["id"],
                    json.dumps(
                        {
                            "verdict": inspection["final_verdict"],
                            "reason": inspection.get("reason"),
                            "profile": inspection.get("profile"),
                            "z": inspection.get("anomaly_z"),
                        }
                    ),
                ),
            )
            for ev in outbox_events:
                self.conn.execute(
                    """INSERT INTO outbox (ts_ms, event_type, payload_json, status, attempts, next_attempt_ms)
                       VALUES (?,?,?, 'pending', 0, ?)""",
                    (now, ev["event_type"], json.dumps(ev["payload"]), now),
                )

    def list_inspections(
        self,
        limit: int = 50,
        offset: int = 0,
        verdict: str | None = None,
        profile: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
        if verdict:
            where.append("COALESCE(final_verdict, verdict) = ?")
            params.append(verdict)
        if profile:
            where.append("profile = ?")
            params.append(profile)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM inspections"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_ms DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self.lock:
            rows = self._rows(self.conn.execute(sql, params).fetchall())
        for r in rows:
            r["detections"] = r.pop("detections_json", []) or []
        return rows

    def list_closed_since(self, ts_ms: int, profile: str, source: str, limit: int = 2001) -> list[dict[str, Any]]:
        with self.lock:
            rows = self._rows(self.conn.execute(
                "SELECT * FROM inspections WHERE ts_ms >= ? AND profile = ? AND source = ? AND status = 'closed' ORDER BY ts_ms ASC LIMIT ?",
                (ts_ms, profile, source, limit),
            ).fetchall())
        for row in rows:
            row["detections"] = row.pop("detections_json", []) or []
        return rows

    def inspection_count(
        self,
        verdict: str | None = None,
        profile: str | None = None,
        status: str | None = None,
    ) -> int:
        where = []
        params: list[Any] = []
        if verdict:
            where.append("COALESCE(final_verdict, verdict) = ?")
            params.append(verdict)
        if profile:
            where.append("profile = ?")
            params.append(profile)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT COUNT(*) AS n FROM inspections"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self.lock:
            return int(self.conn.execute(sql, params).fetchone()["n"])

    def review_queue(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            rows = self._rows(
                self.conn.execute(
                    """SELECT * FROM inspections
                       WHERE status='closed' AND COALESCE(final_verdict, verdict) IN ('review','fail')
                       AND operator_disposition IS NULL
                       ORDER BY ts_ms DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            )
        for r in rows:
            r["detections"] = r.pop("detections_json", []) or []
        return rows

    def get_inspection(self, inspection_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self._row(
                self.conn.execute("SELECT * FROM inspections WHERE id=?", (inspection_id,)).fetchone()
            )
            if row is None:
                return None
            defects = self._rows(
                self.conn.execute(
                    "SELECT * FROM defects WHERE inspection_id=? ORDER BY severity DESC, conf DESC",
                    (inspection_id,),
                ).fetchall()
            )
        if defects:
            for d in defects:
                d["bbox"] = d.pop("bbox_json", []) or []
        row["detections"] = row.pop("detections_json", []) or []
        row["defects"] = defects
        return row

    def review(self, inspection_id: str, action: dict[str, Any], actor: str) -> dict[str, Any] | None:
        now = _now_ms()
        with self.lock, self.conn:
            row = self.conn.execute("SELECT * FROM inspections WHERE id=?", (inspection_id,)).fetchone()
            if row is None:
                return None
            self.conn.execute(
                """UPDATE inspections SET operator_disposition=?, operator_label=?, operator_note=?,
                   reviewed_at_ms=? WHERE id=?""",
                (
                    action["disposition"],
                    action.get("label"),
                    action.get("note"),
                    now,
                    inspection_id,
                ),
            )
            self.conn.execute(
                "INSERT INTO labels (inspection_id, label, source, ts_ms) VALUES (?,?,?,?)",
                (
                    inspection_id,
                    action.get("label") or action["disposition"],
                    "operator",
                    now,
                ),
            )
            self.conn.execute(
                "INSERT INTO audit (ts_ms, actor, action, entity, entity_id, detail_json) VALUES (?,?,?,?,?,?)",
                (
                    now,
                    actor,
                    "review." + action["disposition"],
                    "inspection",
                    inspection_id,
                    json.dumps({"label": action.get("label"), "note": action.get("note")}),
                ),
            )
            self.conn.execute(
                """INSERT INTO outbox (ts_ms, event_type, payload_json, status, attempts, next_attempt_ms)
                   VALUES (?,?,?, 'pending', 0, ?)""",
                (
                    now,
                    "inspection.disposition",
                    json.dumps(
                        {
                            "inspection_id": inspection_id,
                            "disposition": action["disposition"],
                            "label": action.get("label"),
                            "actor": actor,
                        }
                    ),
                    now,
                ),
            )
        return self.get_inspection(inspection_id)

    def abandon_open(self, reason: str = "server_restart") -> int:
        now = _now_ms()
        with self.lock, self.conn:
            cur = self.conn.execute(
                "UPDATE inspections SET status='abandoned', reason=? WHERE status='open'",
                (reason,),
            )
            if cur.rowcount:
                self.conn.execute(
                    "INSERT INTO audit (ts_ms, actor, action, entity, entity_id, detail_json) VALUES (?,?,?,?,?,?)",
                    (now, "system", "inspection.abandoned", "inspection", None, json.dumps({"reason": reason, "count": cur.rowcount})),
                )
            return int(cur.rowcount)

    def abandon_inspection(self, inspection_id: str, reason: str, actor: str = "system") -> None:
        now = _now_ms()
        with self.lock, self.conn:
            cur = self.conn.execute(
                "UPDATE inspections SET status='abandoned', reason=? WHERE id=? AND status='open'",
                (reason, inspection_id),
            )
            if cur.rowcount:
                self.conn.execute(
                    "INSERT INTO audit (ts_ms, actor, action, entity, entity_id, detail_json) VALUES (?,?,?,?,?,?)",
                    (now, actor, "inspection.abandoned", "inspection", inspection_id, json.dumps({"reason": reason})),
                )

    def audit_event(self, actor: str, action: str, entity: str, entity_id: str | None, detail: dict[str, Any] | None = None) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO audit (ts_ms, actor, action, entity, entity_id, detail_json) VALUES (?,?,?,?,?,?)",
                (_now_ms(), actor, action, entity, entity_id, json.dumps(detail or {})),
            )

    # ------------------------------------------------------------------ thresholds
    def active_threshold(self) -> dict[str, Any]:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM threshold_profiles WHERE active=1 ORDER BY updated_at_ms DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else {
            "id": "threshold_v1",
            "name": "Production v1",
            "pass_max": 0.55,
            "review_max": 0.75,
            "unknown_z": 0.75,
            "conf_min": 0.35,
            "active": 1,
            "updated_at_ms": _now_ms(),
        }

    def update_threshold(self, patch: dict[str, Any], actor: str) -> dict[str, Any]:
        current = self.active_threshold()
        for key in ("pass_max", "review_max", "unknown_z", "conf_min"):
            if patch.get(key) is not None:
                current[key] = float(patch[key])
        current["updated_at_ms"] = _now_ms()
        with self.lock, self.conn:
            self.conn.execute(
                """UPDATE threshold_profiles SET pass_max=?, review_max=?, unknown_z=?, conf_min=?,
                   updated_at_ms=? WHERE id=?""",
                (
                    current["pass_max"],
                    current["review_max"],
                    current["unknown_z"],
                    current["conf_min"],
                    current["updated_at_ms"],
                    current["id"],
                ),
            )
            self.conn.execute(
                "INSERT INTO audit (ts_ms, actor, action, entity, entity_id, detail_json) VALUES (?,?,?,?,?,?)",
                (
                    current["updated_at_ms"],
                    actor,
                    "thresholds.update",
                    "threshold_profile",
                    current["id"],
                    json.dumps({k: current[k] for k in ("pass_max", "review_max", "unknown_z", "conf_min")}),
                ),
            )
        return current

    # ------------------------------------------------------------------ metrics
    def metrics(self, profile: str | None = None) -> dict[str, Any]:
        where = "WHERE status='closed'"
        params: list[Any] = []
        if profile:
            where += " AND profile=?"
            params.append(profile)
        with self.lock:
            rows = self._rows(self.conn.execute(f"SELECT * FROM inspections {where}", params).fetchall())
            defects = self._rows(
                self.conn.execute(
                    """SELECT d.class_name, COUNT(*) AS n FROM defects d
                       JOIN inspections i ON i.id=d.inspection_id
                       """ + ("WHERE i.profile=? AND i.status='closed'" if profile else "WHERE i.status='closed'")
                    + " GROUP BY d.class_name ORDER BY n DESC",
                    ([profile] if profile else []),
                ).fetchall()
            )
        total = len(rows)
        verdicts = {"pass": 0, "review": 0, "fail": 0}
        profiles: dict[str, int] = {}
        severities = {"1": 0, "2": 0, "3": 0, "4": 0}
        latencies: list[float] = []
        zvals: list[float] = []
        good_total = 0
        fp_events = 0
        escapes = 0
        bad_total = 0
        unknown_events = 0
        non_pass = 0
        dispositions = {"true_defect": 0, "false_alarm": 0, "reclassify": 0}
        for r in rows:
            v = r.get("final_verdict") or r.get("verdict") or "pass"
            verdicts[v] = verdicts.get(v, 0) + 1
            profiles[r["profile"]] = profiles.get(r["profile"], 0) + 1
            severities[str(r.get("severity") or 1)] = severities.get(str(r.get("severity") or 1), 0) + 1
            latencies.append(float(r.get("latency_total_ms") or 0))
            zvals.append(float(r.get("anomaly_z") or 0))
            if r.get("reason") == "unknown_anomaly":
                unknown_events += 1
            if v != "pass":
                non_pass += 1
            gt_kind = r.get("ground_truth_kind")
            if gt_kind == "good":
                good_total += 1
                if v != "pass":
                    fp_events += 1
            elif gt_kind in ("known_defect", "unknown_defect"):
                bad_total += 1
                if v == "pass":
                    escapes += 1
            disp = r.get("operator_disposition")
            if disp in dispositions:
                dispositions[disp] += 1
        lat = np.asarray(latencies, dtype=np.float32) if latencies else np.zeros(1, dtype=np.float32)
        z = np.asarray(zvals, dtype=np.float32) if zvals else np.zeros(1, dtype=np.float32)
        recent_cut = _now_ms() - 5 * 60 * 1000
        recent = sum(1 for r in rows if int(r.get("ts_ms") or 0) >= recent_cut)
        return {
            "total": total,
            "verdicts": verdicts,
            "pass_rate": (verdicts["pass"] / total) if total else 0.0,
            "defect_rate": (non_pass / total) if total else 0.0,
            "fp_per_1000": (fp_events / good_total * 1000.0) if good_total else None,
            "good_total": good_total,
            "fp_events": fp_events,
            "escapes": escapes,
            "escapes_per_1000": (escapes / bad_total * 1000.0) if bad_total else None,
            "bad_total": bad_total,
            "unknown_routing_rate": (unknown_events / non_pass) if non_pass else 0.0,
            "unknown_events": unknown_events,
            "operator_reject_rate": (
                dispositions["false_alarm"] / (dispositions["true_defect"] + dispositions["false_alarm"])
                if (dispositions["true_defect"] + dispositions["false_alarm"])
                else None
            ),
            "dispositions": dispositions,
            "latency": {
                "p50": float(np.percentile(lat, 50)),
                "p95": float(np.percentile(lat, 95)),
                "p99": float(np.percentile(lat, 99)),
                "mean": float(lat.mean()),
                "n": int(lat.size),
            },
            "anomaly_z": {
                "p50": float(np.percentile(z, 50)),
                "p95": float(np.percentile(z, 95)),
                "max": float(z.max()),
            },
            "throughput_per_min": recent / 5.0,
            "profiles": profiles,
            "severities": severities,
            "classes": {d["class_name"]: d["n"] for d in defects},
        }

    # ------------------------------------------------------------------ audit / outbox
    def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            return self._rows(
                self.conn.execute("SELECT * FROM audit ORDER BY ts_ms DESC LIMIT ?", (limit,)).fetchall()
            )

    def outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            return self._rows(
                self.conn.execute("SELECT * FROM outbox ORDER BY ts_ms DESC LIMIT ?", (limit,)).fetchall()
            )

    def add_outbox(self, event_type: str, payload: dict[str, Any]) -> None:
        now = _now_ms()
        with self.lock, self.conn:
            self.conn.execute(
                """INSERT INTO outbox (ts_ms, event_type, payload_json, status, attempts, next_attempt_ms)
                   VALUES (?,?,?, 'pending', 0, ?)""",
                (now, event_type, json.dumps(payload), now),
            )

    def pending_outbox(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.lock:
            return self._rows(
                self.conn.execute(
                    "SELECT * FROM outbox WHERE status='pending' AND next_attempt_ms<=? ORDER BY id LIMIT ?",
                    (_now_ms(), limit),
                ).fetchall()
            )

    def mark_outbox(self, outbox_id: int, status: str, error: str | None = None, retry_ms: int = 0) -> None:
        now = _now_ms()
        with self.lock, self.conn:
            if status == "delivered":
                self.conn.execute(
                    "UPDATE outbox SET status=?, attempts=attempts+1, last_error=?, delivered_at_ms=? WHERE id=?",
                    (status, error, now, outbox_id),
                )
            else:
                self.conn.execute(
                    "UPDATE outbox SET status=?, attempts=attempts+1, last_error=?, next_attempt_ms=? WHERE id=?",
                    (status, error, now + retry_ms, outbox_id),
                )

    def retry_outbox(self, outbox_id: int) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE outbox SET status='pending', next_attempt_ms=0, last_error=NULL WHERE id=?",
                (outbox_id,),
            )

    # ------------------------------------------------------------------ models / mes
    def model_versions(self) -> list[dict[str, Any]]:
        with self.lock:
            return self._rows(
                self.conn.execute("SELECT * FROM model_versions ORDER BY created_at_ms DESC").fetchall()
            )

    def upsert_model(self, data: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO model_versions
                   (id, kind, name, path, sha256, metrics_json, created_at_ms, active)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    data["id"],
                    data["kind"],
                    data["name"],
                    data.get("path", ""),
                    data.get("sha256", ""),
                    json.dumps(data.get("metrics", {})),
                    data.get("created_at_ms", _now_ms()),
                    data.get("active", 1),
                ),
            )

    def mes_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.lock:
            return self._rows(
                self.conn.execute("SELECT * FROM mes_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            )

    def add_mes_event(self, event_type: str | None, payload: dict[str, Any], signature: str | None) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO mes_events (ts_ms, event_type, payload_json, signature, received_at_ms) VALUES (?,?,?,?,?)",
                (_now_ms(), event_type, json.dumps(payload), signature, _now_ms()),
            )

    def counts(self) -> dict[str, int]:
        with self.lock:
            insp = self.conn.execute("SELECT COUNT(*) AS n FROM inspections").fetchone()["n"]
            open_reviews = self.conn.execute(
                "SELECT COUNT(*) AS n FROM inspections WHERE COALESCE(final_verdict, verdict) IN ('review','fail') AND operator_disposition IS NULL"
            ).fetchone()["n"]
            pending = self.conn.execute("SELECT COUNT(*) AS n FROM outbox WHERE status='pending'").fetchone()["n"]
        return {"inspections": insp, "open_reviews": open_reviews, "outbox_pending": pending}
