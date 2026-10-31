"""Signed MES webhook outbox worker with retry and dead-lettering."""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from server.app.config import Settings
from server.app.db import Database

MAX_ATTEMPTS = 5


class OutboxWorker:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error = ""
        self.last_run_ms = 0
        self.delivered = 0
        self.failed = 0

    @property
    def configured(self) -> bool:
        return bool(self.settings.mes_url)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="indore-outbox", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.process_once()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(2.0)

    def process_once(self) -> int:
        self.last_run_ms = int(time.time() * 1000)
        if not self.configured:
            return 0
        rows = self.db.pending_outbox(20)
        handled = 0
        for row in rows:
            attempts = int(row.get("attempts") or 0)
            if attempts >= MAX_ATTEMPTS:
                self.db.mark_outbox(int(row["id"]), "dead_letter", "max_attempts", 0)
                continue
            ok, error, dead = self._post(row)
            if ok:
                self.db.mark_outbox(int(row["id"]), "delivered", None)
                self.delivered += 1
            else:
                attempts += 1
                retry_ms = min(60_000, 2_000 * (2 ** attempts))
                status = "dead_letter" if dead or attempts >= MAX_ATTEMPTS else "pending"
                self.db.mark_outbox(int(row["id"]), status, error, retry_ms)
                self.failed += 1
                self.last_error = error
            handled += 1
        return handled

    def _post(self, row: dict[str, Any]) -> tuple[bool, str, bool]:
        body = json.dumps(
            {
                "id": row["id"],
                "event_type": row["event_type"],
                "payload": row.get("payload_json", {}),
                "ts_ms": row["ts_ms"],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        signature = hmac.new(
            self.settings.webhook_secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        req = urllib.request.Request(
            self.settings.mes_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Indore-Signature": f"sha256={signature}",
                "X-Indore-Event": str(row["event_type"]),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if 200 <= resp.status < 300:
                    return True, "", False
                return False, f"http_{resp.status}", False
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404, 422):
                return False, f"http_{exc.code}", True
            return False, f"http_{exc.code}", False
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}", False

    def flush_simulated(self, actor: str) -> int:
        """Deliver pending rows locally so the demo can show a drained queue."""
        rows = self.db.pending_outbox(200)
        now = int(time.time() * 1000)
        for row in rows:
            self.db.mark_outbox(int(row["id"]), "delivered", "simulated_local_sink", 0)
        if rows:
            self.db.audit_event(
                actor,
                "outbox.simulated_flush",
                "outbox",
                None,
                {"count": len(rows), "at_ms": now, "sink": self.settings.mes_url or "local-demo-sink"},
            )
        return len(rows)

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "url": self.settings.mes_url or None,
            "last_error": self.last_error,
            "last_run_ms": self.last_run_ms,
            "delivered": self.delivered,
            "failed": self.failed,
            "mode": "signed_webhook" if self.configured else "outbox_only",
        }
