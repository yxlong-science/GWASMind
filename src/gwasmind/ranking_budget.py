"""Versioned resource allowance for one gene-ranking decision and its repairs.

This allowance never changes scientific input or the user's reasoning controls.
Unknown usage is charged at the requested ceiling, so a disconnected call cannot
silently regain an unbounded retry allowance. Rejected requests consume no output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

from .configuration import parameter


@dataclass(frozen=True, slots=True)
class RankingBudgetPolicy:
    version: str
    request_output_tokens: int
    total_output_tokens: int
    deadline_seconds: float

    def __post_init__(self):
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("ranking budget version is required")
        for value in (self.request_output_tokens, self.total_output_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("ranking output budgets must be positive integers")
        if (isinstance(self.deadline_seconds, bool) or not isinstance(self.deadline_seconds, (int, float))
                or not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0):
            raise ValueError("ranking deadline must be finite and positive")

    @classmethod
    def configured(cls):
        return cls(parameter("model.ranking_budget_version"),
                   parameter("model.ranking_request_output_tokens"),
                   parameter("model.ranking_total_output_tokens"),
                   parameter("model.ranking_deadline_seconds"))

    @classmethod
    def for_model(cls, base_url: str, model_id: str):
        """Use a known provider ceiling for each completed JSON generation.

        The wall-clock deadline and bounded transport attempt count still apply.
        An uncatalogued provider retains the configured fallback rather than an
        invented claim about its maximum supported output.
        """
        from .model_generation import MAX_GENERATION_ATTEMPTS
        from .provider_catalog import model_for_endpoint
        policy = cls.configured()
        model = model_for_endpoint(base_url, model_id)
        maximum = getattr(model, "max_output_tokens", None)
        if type(maximum) is int and maximum > 0:
            return replace(policy, version=policy.version + ".provider-max", request_output_tokens=maximum,
                           total_output_tokens=maximum * MAX_GENERATION_ATTEMPTS)
        return policy

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        return {**value, "policy_sha256": hashlib.sha256(json.dumps(value, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
            "scope": "gene_ranking_initial_and_repairs", "includes_reasoning_tokens": True}


class RankingBudget:
    def __init__(self, policy: RankingBudgetPolicy, *, clock: Callable[[], float] | None = None):
        self.policy = policy
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.charges: dict[int, int] = {}

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.policy.total_output_tokens - sum(self.charges.values()))

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.policy.deadline_seconds - (self.clock() - self.started))

    def require_available(self) -> None:
        from .model_generation import GenerationFailure
        if self.remaining_seconds <= 0:
            raise GenerationFailure("The gene-ranking task reached its total wall-clock deadline; scientific input is retained.",
                category="ranking_deadline_exhausted", retryable=False)
        if self.remaining_tokens <= 0:
            raise GenerationFailure("The gene-ranking task exhausted its cumulative output allowance; scientific input is retained.",
                category="ranking_total_output_budget_exhausted", retryable=False)

    def request_limit(self) -> int:
        self.require_available()
        return min(self.policy.request_output_tokens, self.remaining_tokens)

    def snapshot(self) -> dict[str, Any]:
        return {**self.policy.public(), "charged_output_tokens": sum(self.charges.values()),
                "remaining_output_tokens": self.remaining_tokens,
                "elapsed_seconds": round(max(0.0, self.clock() - self.started), 6),
                "remaining_seconds": round(self.remaining_seconds, 6)}

    def reserve(self, attempt: int, requested: int) -> None:
        if type(requested) is not int or not 0 < requested <= self.request_limit():
            raise ValueError("ranking request exceeds the remaining task allowance")
        self.charges[attempt] = requested

    def settle(self, attempt: int, requested: int, *, usage: Any = None,
               failure: Mapping[str, Any] | None = None) -> dict[str, Any]:
        usage = usage if isinstance(usage, Mapping) else {}
        observed = usage.get("completion_tokens")
        actual = observed if type(observed) is int and observed >= 0 else None
        status = (failure or {}).get("http_status")
        category = (failure or {}).get("category")
        rejected = (category in {"context_capacity_rejected", "input_exceeds_model_context", "provider_output_limit", "authentication"}
                    or type(status) is int and status in {400, 401, 403, 404, 405, 413, 422, 429})
        if actual is not None:
            charged, method = actual, "provider_reported_completion_tokens"
        elif rejected:
            charged, method = 0, "request_rejected_before_generation"
        else:
            charged, method = requested, "requested_ceiling_reserved_usage_unavailable"
        self.charges[attempt] = charged
        return {**self.snapshot(), "attempt_charged_output_tokens": charged,
                "accounting_method": method, "provider_reported_output_tokens": actual,
                "provider_exceeded_requested_output": actual is not None and actual > requested}
