from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, TypeAlias


ProgressSink: TypeAlias = Callable[[str, Mapping[str, Any]], None]


def emit_progress(
    sink: ProgressSink | None,
    phase: str,
    **details: Any,
) -> None:
    """Emit an observable workflow event without affecting scientific execution."""

    if sink is None:
        return
    try:
        sink(phase, details)
    except Exception:
        # UI/logging failures must not alter a scientific decision.
        return


class JsonlProgressSink:
    """Append small, structured progress events for a supervising UI process."""

    def __init__(self, path: str | Path, *, run_id: str):
        self.path = Path(path).resolve()
        self.run_id = run_id
        self._lock = Lock()
        self._last_content = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, phase: str, details: Mapping[str, Any]) -> None:
        event = {
            "run_id": self.run_id,
            "phase": phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": dict(details),
        }
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        content = json.dumps({"phase": phase, "details": dict(details)}, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if content == self._last_content:
                return
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._last_content = content
