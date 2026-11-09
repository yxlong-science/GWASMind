from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (set, frozenset, tuple)):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    created_at: str
    mode: str
    data_release: str
    snapshot_hash: str
    model_provider: str
    model_id: str
    model_parameters: dict[str, Any]
    prompt_hash: str
    policy_hash: str
    code_version: str
    python_version: str
    platform: str
    input_hash: str = "unknown"
    tool_contract_hash: str = "unknown"
    requested_model_id: str = "unknown"
    observed_model_id: str | None = None
    prompt_version: str = "unknown"
    seed: int | None = None
    benchmark: bool = False
    offline: bool = False
    configuration_provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        data_release: str,
        snapshot_hash: str,
        model_provider: str,
        model_id: str,
        model_parameters: dict[str, Any],
        prompt: str,
        policy: Any,
        code_version: str,
        run_id: str | None = None,
        input_value: Any = None,
        tool_contract: Any = None,
        requested_model_id: str | None = None,
        observed_model_id: str | None = None,
        prompt_version: str = "unknown",
        seed: int | None = None,
        benchmark: bool = False,
        offline: bool = False,
    ) -> "RunManifest":
        from .configuration import configuration_provenance

        policy_value = asdict(policy) if hasattr(policy, "__dataclass_fields__") else policy
        return cls(
            run_id=run_id or str(uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            mode=mode,
            data_release=data_release,
            snapshot_hash=snapshot_hash,
            model_provider=model_provider,
            model_id=model_id,
            model_parameters=dict(model_parameters),
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            policy_hash=sha256_json(policy_value),
            code_version=code_version,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            input_hash=sha256_json(input_value) if input_value is not None else "unknown",
            tool_contract_hash=(
                sha256_json(tool_contract) if tool_contract is not None else "unknown"
            ),
            requested_model_id=requested_model_id or model_id,
            observed_model_id=observed_model_id,
            prompt_version=prompt_version,
            seed=seed,
            benchmark=benchmark,
            offline=offline,
            configuration_provenance=configuration_provenance(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Append-only JSONL audit logger with no secret-bearing request headers."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []

    def log(self, event: str, **payload: Any) -> None:
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(item) + "\n")
        self.events.append(item)

    def write_manifest(self, manifest: RunManifest, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
