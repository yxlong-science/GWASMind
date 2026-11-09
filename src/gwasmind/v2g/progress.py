"""Measured, bounded UI progress for the gene-centric retrieval path."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from time import monotonic

from ..progress import emit_progress


class RetrievalProgress:
    def __init__(self, sink=None, *, clock=monotonic, interval=2.0):
        self.sink, self.clock, self.interval = sink, clock, interval
        self.current = None

    def update(self, **counts):
        if self.current is not None:
            self.current["counts"].update(counts)
            self._emit("running")

    def request_finished(self, *, success):
        if self.current is not None:
            key = "successful_requests" if success else "failed_request_pages"
            self.update(**{key: self.current["counts"].get(key, 0) + 1})

    def _emit(self, status):
        current = self.current
        if current is not None:
            emit_progress(self.sink, "v2g_retrieval", name=current["name"], status=status,
                          elapsed_seconds=max(0, int(self.clock() - current["start"])),
                          timing_basis="backend_monotonic", **current["counts"])

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(self.interval)
            self._emit("waiting")

    @asynccontextmanager
    async def operation(self, name, **counts):
        if self.current is not None:
            raise RuntimeError("Retrieval progress operations must not overlap")
        self.current = {"name": name, "start": self.clock(), "counts": dict(counts)}
        self._emit("started")
        heartbeat = asyncio.create_task(self._heartbeat()) if self.sink is not None else None
        try:
            yield self
        except BaseException:
            self._emit("failed")
            raise
        else:
            counts = self.current["counts"]
            self._emit("incomplete" if counts.get("failed_items", 0) or counts.get("partial_items", 0) else "completed")
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
            self.current = None
