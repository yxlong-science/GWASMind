from __future__ import annotations

import ipaddress
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

from .configuration import parameter
from .model_options import (
    DEFAULT_REASONING_EFFORT,
    normalize_reasoning_effort,
    normalize_thinking_mode,
)
from .models import DataMode
from .provider_catalog import is_safe_provider_base_url
from .provider_reasoning import reasoning_request_body


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    data_mode: DataMode
    model_provider: str
    model_id: str
    benchmark: bool = False
    offline: bool = False
    model_base_url: str | None = None
    thinking_mode: str = parameter("model.ranking_thinking_mode")
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    reasoning_profile: dict[str, Any] | None = None

    def validate(self) -> None:
        normalize_thinking_mode(self.thinking_mode)
        normalize_reasoning_effort(self.reasoning_effort)
        reasoning_request_body(base_url=self.model_base_url or "", model_id=self.model_id,
            thinking_mode=self.thinking_mode, reasoning_effort=self.reasoning_effort,
            profile=self.reasoning_profile)
        if self.benchmark and self.data_mode is DataMode.LIVE:
            raise ValueError("benchmark mode cannot use live Open Targets")
        if self.offline and self.data_mode is DataMode.LIVE:
            raise ValueError("fully offline mode cannot use live Open Targets")
        if self.offline and self.model_provider != "deterministic":
            if not self.model_base_url or not _is_loopback_url(self.model_base_url):
                raise ValueError(
                    "offline model endpoints must use localhost/loopback or deterministic mode"
                )

    def public_dict(self) -> dict:
        value = asdict(self)
        if self.reasoning_profile is None:
            value.pop("reasoning_profile")
        return value


def _is_loopback_url(value: str) -> bool:
    if not is_safe_provider_base_url(value):
        return False
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
