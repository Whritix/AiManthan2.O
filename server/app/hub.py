"""Thread-safe live frame fan-out for WebSocket and MJPEG consumers."""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any


class Hub:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latest_frame: dict[str, Any] | None = None
        self._latest_jpeg: bytes = b""
        self._seq = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._times: deque[float] = deque(maxlen=240)
        self._recent: deque[dict[str, Any]] = deque(maxlen=120)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest_frame

    @property
    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    @property
    def recent(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._recent)

    @property
    def ws_clients(self) -> int:
        with self._lock:
            return len(self._queues)

    @property
    def fps(self) -> float:
        now = time.time()
        with self._lock:
            values = [t for t in self._times if now - t <= 5.0]
        if len(values) < 2:
            return 0.0
        span = max(1e-6, values[-1] - values[0])
        return (len(values) - 1) / span

    def publish(self, frame: dict[str, Any], jpeg: bytes) -> None:
        now = time.time()
        with self._lock:
            self._latest_frame = frame
            self._latest_jpeg = jpeg
            self._seq += 1
            self._times.append(now)
            self._recent.append(frame)
            loop = self._loop
            queues = list(self._queues)
        if loop is None or not loop.is_running():
            return
        loop.call_soon_threadsafe(self._fanout, frame, queues)

    def _fanout(self, frame: dict[str, Any], queues: list[asyncio.Queue[dict[str, Any]]]) -> None:
        for q in queues:
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(frame)
                except asyncio.QueueFull:
                    pass

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=8)
        with self._lock:
            self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._queues.discard(q)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "seq": self._seq,
                "frame": self._latest_frame,
                "fps": round(self.fps, 2),
                "ws_clients": len(self._queues),
            }
