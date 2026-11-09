"""Safe failure diagnostics and one shared attempt allowance per scientific decision."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

MAX_GENERATION_ATTEMPTS = 3
MAX_TRANSPORT_RETRIES = 5
_FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_calls", "function_call"})
_CONTEXT_CODES = frozenset({"context_length_exceeded", "context_window_exceeded", "input_token_limit_exceeded", "prompt_too_long"})
_ERROR_CODES = _CONTEXT_CODES | frozenset({"invalid_request_error", "invalid_parameter", "invalid_value",
    "unsupported_parameter", "model_not_found", "insufficient_quota", "quota_limit_reached", "invalid_api_key"})
_ERROR_PARAMS = frozenset({"messages", "prompt", "input", "max_tokens", "max_completion_tokens", "reasoning_effort",
    "thinking", "model", "response_format", "temperature", "top_p", "stream"})
_ERROR_TYPES = frozenset({"invalid_request_error", "api_error", "authentication_error", "permission_error",
    "rate_limit_error", "server_error", "invalid_parameter_error"})
_TOKEN_NUMBER = r"([0-9][0-9,]{0,10})"


class GenerationFailure(RuntimeError):
    """Provider diagnostics without response text, credentials or hidden reasoning."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "provider_response",
        finish_reason: str | None = None,
        retryable: bool = False,
        http_status: int | None = None,
        provider_error: Any = None,
    ):
        super().__init__(message)
        self.category = category
        self.finish_reason = (
            finish_reason
            if isinstance(finish_reason, str) and finish_reason in _FINISH_REASONS
            else None
        )
        self.retryable = retryable
        self.http_status = http_status
        self.provider_error = provider_error_diagnostics(provider_error) if provider_error is not None else None

    def safe_details(self) -> dict[str, Any]:
        result = {
            "category": self.category,
            "finish_reason": self.finish_reason,
            "retryable": self.retryable,
            "http_status": self.http_status,
        }
        if self.provider_error is not None:
            result["provider_error"] = deepcopy(self.provider_error)
        return result


def _provider_error_object(body: Any) -> dict[str, Any]:
    """Accept the SDK's flat object, wrapped error, or bounded JSON response."""
    if isinstance(body, (bytes, str)):
        if len(body) > 32768:
            return {}
        try:
            body = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return {}
    if not isinstance(body, dict):
        return {}
    error = body.get("error", body)
    return error if isinstance(error, dict) else {}


def _token_count(value: Any) -> int | None:
    if isinstance(value, str) and re.fullmatch(_TOKEN_NUMBER, value):
        value = int(value.replace(",", ""))
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 10_000_000 else None


def provider_error_diagnostics(body: Any) -> dict[str, Any]:
    """Retain finite error enums and explicit capacity numbers, never raw prose.

    Recognizing a capacity rejection does not itself authorize a retry: callers
    must have actual prompt/capacity numbers and an evidence-preserving repacker.
    """
    error = _provider_error_object(body)
    result: dict[str, Any] = {}
    for key, allowed in (("code", _ERROR_CODES), ("param", _ERROR_PARAMS), ("type", _ERROR_TYPES)):
        value = error.get(key)
        if value is not None:
            result[key] = value if isinstance(value, str) and value in allowed else "unrecognized"
    message = error.get("message", "")
    message = message[:8192].casefold() if isinstance(message, str) else ""
    context_rejection = result.get("code") in _CONTEXT_CODES or bool(re.search(
        r"(?:maximum|max)\s+context\s+(?:length|window)|context[_ ](?:length|window).{0,60}(?:exceed|limit)|"
        r"(?:input|prompt)\s+tokens?.{0,40}exceed.{0,20}limit", message))
    if not context_rejection:
        return result
    result["capacity_rejection"] = True
    aliases = {
        "context_window": ("context_window", "context_length", "max_context_tokens", "max_context_length"),
        "prompt_tokens": ("prompt_tokens", "input_tokens", "message_tokens"),
        "requested_output_tokens": ("max_tokens", "max_completion_tokens", "completion_tokens", "output_tokens"),
        "requested_total_tokens": ("requested_tokens", "total_tokens"),
    }
    patterns = {
        "context_window": (r"(?:maximum|max)\s+context\s+(?:length|window)(?:\s+(?:is|of))?\s*[:=]?\s*" + _TOKEN_NUMBER,
            r"context\s+(?:length|window)\s+(?:limit|of)(?:\s+is)?\s*[:=]?\s*" + _TOKEN_NUMBER),
        "prompt_tokens": (_TOKEN_NUMBER + r"\s+(?:tokens?\s+)?in\s+(?:the\s+)?(?:messages|prompt|input)",
            r"(?:prompt|input|message)\s+(?:token\s+count|tokens|length)(?:\s+is)?\s*[:=]?\s*" + _TOKEN_NUMBER),
        "requested_output_tokens": (_TOKEN_NUMBER + r"\s+(?:tokens?\s+)?(?:in|for)\s+(?:the\s+)?(?:completion|output)",),
        "requested_total_tokens": (r"(?:you\s+)?requested\s+" + _TOKEN_NUMBER + r"\s+tokens",),
    }
    capacity: dict[str, int] = {}
    ambiguous = False
    for key, names in aliases.items():
        values = {_token_count(error.get(name)) for name in names}
        values.update(_token_count(match) for pattern in patterns[key] for match in re.findall(pattern, message))
        values.discard(None)
        if len(values) == 1:
            capacity[key] = values.pop()
        elif len(values) > 1:
            ambiguous = True
    if ambiguous:
        result["capacity_numbers_ambiguous"] = True
    if capacity:
        result["capacity"] = capacity
    return result


def provider_context_recovery(body: Any, *, context_window: int | None,
                              requested_output_tokens: int | None) -> dict[str, int] | None:
    """A joint-capacity rejection can recalibrate input only from explicit counts."""
    diagnostic = provider_error_diagnostics(body)
    if not diagnostic.get("capacity_rejection") or diagnostic.get("capacity_numbers_ambiguous"):
        return None
    capacity = diagnostic.get("capacity", {})
    actual_context = capacity.get("context_window", context_window)
    actual_prompt = capacity.get("prompt_tokens")
    reported_output = capacity.get("requested_output_tokens")
    reported_total = capacity.get("requested_total_tokens")
    if not context_window or not requested_output_tokens or not actual_context:
        return None  # Unknown models retain provider defaults; no invented budget.
    if reported_output is not None and reported_output != requested_output_tokens:
        return None
    if actual_prompt is None and reported_total is not None and reported_output is not None:
        actual_prompt = reported_total - reported_output
    if (reported_total is not None and actual_prompt is not None and reported_output is not None
            and reported_total != actual_prompt + reported_output):
        return None
    available = actual_context - requested_output_tokens
    if not actual_prompt or available <= 0 or actual_prompt <= available:
        return None
    return {"prompt_tokens": actual_prompt, "context_window": actual_context,
            "input_token_budget": available}


def provider_output_limit(body: Any) -> int | None:
    """Read only an explicit output-parameter ceiling from a bounded provider error.

    The text is never returned or logged. Context-window errors and arbitrary
    numeric prose do not authorize reducing scientific input or reasoning depth.
    """
    error = _provider_error_object(body)
    if not isinstance(error, dict) or not isinstance(error.get("message"), str):
        return None
    message = error["message"][:8192].casefold()
    if error.get("param") not in {"max_tokens", "max_completion_tokens"} and not re.search(
        r"\bmax_(?:completion_)?tokens\b", message
    ):
        return None
    patterns = (
        r"(?:at most|maximum of|maximum|no more than)\s+([0-9][0-9,]*)\s+(?:output |completion )tokens",
        r"max_(?:completion_)?tokens\b.{0,100}?(?:<=|cannot exceed|must not exceed|less than or equal to|at most)\s*([0-9][0-9,]*)",
        r"max_(?:completion_)?tokens\b.{0,40}?between\s+1\s+and\s+([0-9][0-9,]*)",
    )
    try:
        values = {
            int(match.replace(",", ""))
            for pattern in patterns
            for match in re.findall(pattern, message)
        }
    except ValueError:
        return None
    return (
        values.pop()
        if len(values) == 1 and all(0 < value <= 10_000_000 for value in values)
        else None
    )


def classify_generation_failure(error: BaseException) -> dict[str, Any]:
    """Classify known exception types; never copy provider-controlled exception text."""
    if isinstance(error, GenerationFailure):
        return error.safe_details()
    name = type(error).__name__
    status = getattr(error, "status_code", getattr(error, "code", None))
    status = (
        status
        if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599
        else None
    )
    if status in {401, 403} or name in {"AuthenticationError", "PermissionDeniedError"}:
        category, retryable = "authentication", False
    elif status == 429:
        category, retryable = "rate_limit", True
    elif status in {408, 500, 502, 503, 504}:
        category, retryable = "provider_unavailable", True
    elif status is not None and 400 <= status < 500:
        context_error = status in {400, 422} and provider_error_diagnostics(getattr(error, "body", None)).get("capacity_rejection")
        category, retryable = "context_capacity_rejected" if context_error else "request_rejected", False
    elif isinstance(error, TimeoutError) or name in {
        "APITimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
    }:
        category, retryable = "timeout", True
    elif isinstance(error, OSError) or name in {
        "APIConnectionError",
        "RemoteProtocolError",
        "ReadError",
        "ConnectError",
        "URLError",
    }:
        category, retryable = "connection", True
    elif name == "ModelIdentityError":
        category, retryable = "model_identity", False
    elif name == "SynthesisOutputError":
        category, retryable = "json_structure", True
    else:
        category, retryable = "unexpected_failure", False
    result = {
        "category": category,
        "retryable": retryable,
        "finish_reason": None,
        "http_status": status,
    }
    if status is not None and getattr(error, "body", None) is not None:
        result["provider_error"] = provider_error_diagnostics(error.body)
    return result


def _is_transport_retry(purpose: Any) -> bool:
    return purpose in {
        "timeout_retry", "connection_retry", "stream_interrupted_retry", "rate_limit_retry",
        "provider_unavailable_retry", "provider_output_limit_retry",
    } if isinstance(purpose, str) else False


class GenerationAttempts:
    """Bounded generation allowance for one scientific decision.

    By default the initial generation, transport retries and output repairs
    share ``maximum``. With ``transport_retries`` set, ``maximum`` bounds only
    generations (initial plus repairs) and each generation gets its own
    transport-retry allowance, so network failures cannot consume a repair.
    """

    def __init__(
        self,
        *,
        maximum: int = MAX_GENERATION_ATTEMPTS,
        sink: Callable[[dict[str, Any]], None] | None = None,
        resource_budget: Any = None,
        transport_retries: int | None = None,
    ):
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= MAX_GENERATION_ATTEMPTS:
            raise ValueError("Generation attempts must be between 1 and 3.")
        if transport_retries is not None and (isinstance(transport_retries, bool) or not isinstance(transport_retries, int)
                                              or not 0 <= transport_retries <= MAX_TRANSPORT_RETRIES):
            raise ValueError(f"Transport retries must be between 0 and {MAX_TRANSPORT_RETRIES}.")
        self.maximum = maximum
        self.transport_retries = transport_retries
        self.sink = sink
        self.rows: list[dict[str, Any]] = []
        self.resource_budget = resource_budget

    @property
    def remaining(self) -> int:
        if self.transport_retries is None:
            return max(0, self.maximum - len(self.rows))
        return max(0, self.maximum - sum(not _is_transport_retry(row["purpose"]) for row in self.rows))

    @property
    def retry_remaining(self) -> int:
        """Transport retries still available for the current generation."""
        if self.transport_retries is None:
            return self.remaining
        used = 0
        for row in reversed(self.rows):
            if not _is_transport_retry(row["purpose"]):
                break
            used += 1
        return max(0, self.transport_retries - used)

    @property
    def current_retry_index(self) -> int:
        """Zero-based index of the next transport retry of the current generation."""
        used = 0
        for row in reversed(self.rows):
            if not _is_transport_retry(row["purpose"]):
                break
            used += 1
        return used

    def remaining_for(self, purpose: str) -> int:
        """Apply the same request classification at retry selection and admission."""
        return self.retry_remaining if _is_transport_retry(purpose) and self.rows else self.remaining

    def begin(
        self,
        *,
        model_id: str,
        provider: str,
        messages: Any,
        options: dict[str, Any],
        purpose: str = "initial",
        token_budget: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.remaining_for(purpose):
            raise GenerationFailure(
                "The bounded generation recovery allowance was exhausted.",
                category="attempt_budget",
            )
        encoded = json.dumps(
            {"model": model_id, "messages": messages, "options": options},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        row = {
            "attempt": len(self.rows) + 1,
            "maximum_attempts": self.maximum,
            **({"transport_retry_limit": self.transport_retries} if self.transport_retries is not None else {}),
            "purpose": purpose,
            "status": "started",
            "model_id": model_id,
            "provider": provider,
            "model_input_sha256": hashlib.sha256(encoded).hexdigest(),
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "requested_max_output_tokens": options.get(
                "max_completion_tokens", options.get("max_tokens")
            ),
            "started_monotonic": time.monotonic(),
        }
        if self.resource_budget is not None:
            row["task_resource_before"] = self.resource_budget.snapshot()
            self.resource_budget.reserve(row["attempt"], row["requested_max_output_tokens"])
        self.rows.append(row)
        if token_budget is not None:
            row["token_budget"] = deepcopy(token_budget)
        if projection is not None:
            row["projection"] = deepcopy(projection)
        self._emit(row)
        return row

    def finish(
        self,
        row: dict[str, Any],
        *,
        failure: BaseException | None = None,
        finish_reason: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        row["elapsed_seconds"] = round(max(0, time.monotonic() - row["started_monotonic"]), 3)
        row["status"] = "failed" if failure else "generated"
        row["finish_reason"] = (
            finish_reason
            if isinstance(finish_reason, str) and finish_reason in _FINISH_REASONS
            else None
        )
        if failure:
            row.update(classify_generation_failure(failure))
            if usage is None:
                usage = getattr(failure, "usage", None)
        if usage:
            from .model_stream import bounded_token_usage

            row["usage"] = bounded_token_usage(usage)
            actual = row["usage"].get("prompt_tokens")
            estimate = (row.get("token_budget") or {}).get("prompt_tokens_estimate")
            if actual and estimate is not None:
                row["input_token_calibration"] = {"estimated": estimate, "actual": actual,
                    "relative_error": (estimate - actual) / actual,
                    "overestimate": estimate >= actual}
        if self.resource_budget is not None:
            row["task_resource_usage"] = self.resource_budget.settle(row["attempt"], row["requested_max_output_tokens"],
                usage=usage if usage is not None else row.get("usage"),
                failure=classify_generation_failure(failure) if failure is not None else None)
        self._emit(row)

    def validation(self, *, passed: bool, category: str = "grounding") -> None:
        if not self.rows:
            return
        row = self.rows[-1]
        row["validation"] = "passed" if passed else "failed"
        if not passed:
            row["category"] = category
        self._emit(row)

    def public(self) -> list[dict[str, Any]]:
        return [
            deepcopy({key: value for key, value in row.items() if key != "started_monotonic"})
            for row in self.rows
        ]

    def _emit(self, row: dict[str, Any]) -> None:
        if self.sink:
            self.sink(
                deepcopy({key: value for key, value in row.items() if key != "started_monotonic"})
            )
