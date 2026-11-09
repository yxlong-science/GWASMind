from __future__ import annotations

import json
import math
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..configuration import parameter
from ..llm import visible_completion_text
from ..model_generation import (
    GenerationAttempts,
    GenerationFailure,
    classify_generation_failure,
    provider_output_limit,
)
from ..model_options import (
    DEFAULT_REASONING_EFFORT,
    THINKING_DISABLED,
    THINKING_ENABLED,
    is_deepseek_model_endpoint,
    model_token_budget,
    normalize_reasoning_effort,
    normalize_thinking_mode,
    provider_completion_budget,
)
from ..model_stream import CompletionStream, CompletionStreamError
from ..prompting import load_prompt
from ..provider_catalog import infer_provider_id, is_safe_provider_base_url, model_for_endpoint
from ..provider_reasoning import (
    parse_openrouter_reasoning,
    reasoning_for_endpoint,
    reasoning_request_body,
)

MAX_PROBE_RESPONSE_BYTES = 256 * 1024
MODEL_PROBE_TIMEOUT_SECONDS = parameter("model.probe_timeout_seconds")
MODEL_PROBE_MAX_TOKENS = parameter("model.probe_max_tokens")
MAX_MODEL_LIST_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_MODEL_LIST_MODELS = 500
MAX_MODEL_LIST_ID_LENGTH = 120
MODEL_LIST_TIMEOUT_SECONDS = parameter("model.list_timeout_seconds")
MAX_DEEPSEEK_CHAT_RESPONSE_BYTES = 16 * 1024 * 1024
MODEL_CHAT_TIMEOUT_SECONDS = parameter("model.compatible_timeout_seconds")
DEEPSEEK_CHAT_TIMEOUT_SECONDS = parameter("model.deepseek_timeout_seconds")
MODEL_CHAT_MAX_RETRIES = parameter("model.sdk_max_retries")
_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class ModelProbeError(GenerationFailure):
    """A sanitized model-provider failure that is safe to show in the local UI."""


@dataclass(frozen=True, slots=True)
class ModelProbeResult:
    requested_model: str
    observed_model: str
    latency_ms: int
    thinking_mode: str = THINKING_DISABLED
    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class ModelListResult:
    models: tuple[str, ...]
    latency_ms: int
    truncated: bool = False
    reasoning_capabilities: Mapping[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelChatResult:
    requested_model: str
    observed_model: str
    content: str
    latency_ms: int
    thinking_mode: str = THINKING_DISABLED
    reasoning_effort: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    generation_attempts: tuple[dict[str, Any], ...] = ()
    token_budget: Mapping[str, Any] = field(default_factory=dict)


class _RejectRedirects(HTTPRedirectHandler):
    """Do not forward an API credential to a redirected host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def list_openai_compatible_models(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float = MODEL_LIST_TIMEOUT_SECONDS,
) -> ModelListResult:
    """Return one bounded, authenticated snapshot of OpenAI-compatible model IDs.

    Exactly one non-redirecting ``GET /models`` request is attempted. The
    credential and raw upstream body are never included in the result or in a
    raised :class:`ModelProbeError`.
    """

    if (
        not isinstance(api_key, str)
        or api_key != api_key.strip()
        or not 8 <= len(api_key) <= 512
        or any(ord(character) < 32 or ord(character) == 127 for character in api_key)
    ):
        raise ModelProbeError("The API key is invalid.")
    if not is_safe_provider_base_url(base_url):
        raise ModelProbeError("The API Base URL must be HTTPS or a loopback HTTP URL.")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= MODEL_LIST_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"timeout_seconds must be between 0 and {MODEL_LIST_TIMEOUT_SECONDS}"
        )

    value, latency_ms = _request_models_json(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=float(timeout_seconds),
        max_response_bytes=MAX_MODEL_LIST_RESPONSE_BYTES,
        user_agent="GWASMind/1.0 model-list",
    )
    models, truncated = _model_identifiers_from_response(value)
    if any(api_key in model_id for model_id in models):
        raise ModelProbeError("The provider returned an invalid model list.")
    return ModelListResult(
        models=models,
        latency_ms=latency_ms,
        truncated=truncated,
        reasoning_capabilities=_model_reasoning_capabilities(value, models, base_url),
    )


def probe_openai_compatible(
    *,
    api_key: str,
    base_url: str,
    model_id: str,
    thinking_mode: str = THINKING_DISABLED,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    reasoning_profile: Mapping[str, Any] | None = None,
    timeout_seconds: float = MODEL_PROBE_TIMEOUT_SECONDS,
) -> ModelProbeResult:
    """Make one bounded request to verify URL, credential, and exact model access.

    This connection test uses the authenticated model-list API for official
    DeepSeek/OpenAI endpoints, avoiding inference latency and token use. Other
    OpenAI-compatible endpoints use one 16-token chat request because model-list
    support is not universal. The separate Settings refresh action always makes
    one bounded model-list request. Response bodies are validated but never
    returned, and provider error bodies are only coarsely classified.
    """

    normalize_thinking_mode(thinking_mode)
    normalize_reasoning_effort(reasoning_effort)
    if _prefer_model_list_probe(base_url):
        value, latency_ms = _request_models_json(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        observed = _validate_model_list_response(value, model_id)
        return ModelProbeResult(
            requested_model=model_id,
            observed_model=observed,
            latency_ms=latency_ms,
            thinking_mode=THINKING_DISABLED,
        )

    value, latency_ms = _request_chat_json(
        api_key=api_key,
        base_url=base_url,
        model_id=model_id,
        messages=[
            {
                "role": "user",
                "content": load_prompt("connection_probe_user"),
            }
        ],
        max_tokens=MODEL_PROBE_MAX_TOKENS,
        timeout_seconds=timeout_seconds,
        max_response_bytes=MAX_PROBE_RESPONSE_BYTES,
        user_agent="GWASMind/1.0 connection-test",
        thinking_mode="auto",
        reasoning_effort="default",
        reasoning_profile=reasoning_profile,
        max_retries=0,
    )
    choices, observed = _validate_chat_response(
        value,
        model_id,
        accepted_finish_reasons=frozenset({None, "stop", "length"}),
    )
    if not isinstance(choices[0], dict):  # Defensive after validation.
        raise ModelProbeError("The provider response did not contain a chat-completion result.")
    return ModelProbeResult(
        requested_model=model_id,
        observed_model=observed,
        latency_ms=latency_ms,
        thinking_mode="auto",
        reasoning_effort=None,
    )


def _prefer_model_list_probe(base_url: str) -> bool:
    """Use the fast authenticated model-list contract for official providers."""

    hostname = (urlsplit(base_url).hostname or "").casefold().rstrip(".")
    return hostname.endswith(
        ("api.deepseek.com", ".deepseek.com", "api.openai.com", ".openai.com")
    )


def _request_models_json(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    max_response_bytes: int = MAX_PROBE_RESPONSE_BYTES,
    user_agent: str = "GWASMind/1.0 connection-test",
) -> tuple[dict[str, Any], int]:
    request = Request(
        _models_url(base_url),
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
    )
    return _execute_json_request(
        request,
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
        not_found_message=(
            "The provider's models endpoint was not found. Check the exact API Base URL."
        ),
        allow_model_array=True,
    )


def _model_identifiers_from_response(
    value: Mapping[str, Any],
) -> tuple[tuple[str, ...], bool]:
    rows = value.get("data")
    if not isinstance(rows, list):
        raise ModelProbeError("The provider returned an invalid OpenAI-compatible model list.")

    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, (Mapping, str)):
            raise ModelProbeError(
                "The provider returned an invalid OpenAI-compatible model list."
            )
        identifier = row if isinstance(row, str) else row.get("id")
        if not isinstance(identifier, str):
            raise ModelProbeError(
                "The provider returned an invalid OpenAI-compatible model list."
            )
        normalized = identifier.strip()
        if (
            not 1 <= len(normalized) <= MAX_MODEL_LIST_ID_LENGTH
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in normalized
            )
        ):
            raise ModelProbeError(
                "The provider returned an invalid OpenAI-compatible model identifier."
            )
        identifiers.add(normalized)

    ordered = sorted(identifiers)
    truncated = len(ordered) > MAX_MODEL_LIST_MODELS or value.get("has_more") is True
    return tuple(ordered[:MAX_MODEL_LIST_MODELS]), truncated


def _model_reasoning_capabilities(
    value: Mapping[str, Any], models: tuple[str, ...], base_url: str,
) -> dict[str, dict[str, Any]]:
    """Keep capability discovery bound to the same model-list snapshot."""

    capabilities = {identifier: reasoning_for_endpoint(base_url, identifier) for identifier in models}
    if infer_provider_id(base_url) != "openrouter":
        return capabilities
    seen: set[str] = set()
    conflicts: set[str] = set()
    for row in value.get("data", []):
        identifier = row if isinstance(row, str) else row.get("id") if isinstance(row, Mapping) else None
        if not isinstance(identifier, str) or identifier.strip() not in capabilities:
            continue
        identifier = identifier.strip()
        metadata = row.get("reasoning") if isinstance(row, Mapping) else None
        capability = parse_openrouter_reasoning(metadata)
        if identifier in seen and capabilities[identifier] != capability:
            conflicts.add(identifier)
        capabilities[identifier] = capability
        seen.add(identifier)
    for identifier in conflicts:
        capabilities[identifier] = parse_openrouter_reasoning(None)
    return capabilities


def _validate_model_list_response(value: Mapping[str, Any], model_id: str) -> str:
    rows = value.get("data")
    if not isinstance(rows, list):
        raise ModelProbeError("The provider returned an invalid OpenAI-compatible model list.")
    identifiers = {
        (row if isinstance(row, str) else str(row.get("id"))).strip()
        for row in rows
        if isinstance(row, str) or isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    if model_id not in identifiers:
        raise ModelProbeError(
            "The provider authenticated successfully but did not list the exact Model Name."
        )
    return model_id


def complete_openai_compatible(
    *,
    api_key: str,
    base_url: str,
    model_id: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int | None = None,
    thinking_mode: str = THINKING_DISABLED,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    reasoning_profile: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    json_output: bool = False,
    generation_attempts: GenerationAttempts | None = None,
    cancel_event: Any = None,
) -> ModelChatResult:
    """Return one validated completion, with optional numeric DeepSeek telemetry."""

    normalized_thinking_mode = normalize_thinking_mode(thinking_mode)
    if not isinstance(json_output, bool):
        raise ValueError("json_output must be a boolean")  # noqa: TRY004 - public configuration contract
    normalized_reasoning_effort = normalize_reasoning_effort(reasoning_effort)
    official_deepseek = is_deepseek_model_endpoint(
        model_id=model_id,
        base_url=base_url,
    )
    if reasoning_profile is not None and not isinstance(reasoning_profile, Mapping):
        raise ValueError("reasoning_profile must be a mapping")
    profile = deepcopy(dict(reasoning_profile)) if reasoning_profile is not None else None
    reasoning_request_body(
        base_url=base_url, model_id=model_id, thinking_mode=normalized_thinking_mode,
        reasoning_effort=normalized_reasoning_effort, profile=profile,
    )
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
        raise ValueError("max_tokens must be positive or omitted")
    effective_timeout = (
        DEEPSEEK_CHAT_TIMEOUT_SECONDS
        if timeout_seconds is None and official_deepseek
        else MODEL_CHAT_TIMEOUT_SECONDS
        if timeout_seconds is None
        else timeout_seconds
    )
    if (
        isinstance(effective_timeout, bool) or not isinstance(effective_timeout, (int, float))
        or not math.isfinite(effective_timeout) or effective_timeout <= 0
    ):
        raise ValueError("timeout_seconds must be positive")
    # Writers may add a system prompt, the current question and repair turns.
    # Keep the envelope consistent with both configured history entry points.
    max_messages = max(parameter("web.chat_history_limit"), parameter("web.general_history_turns")) + 4
    if not messages or len(messages) > max_messages:
        raise ValueError(f"messages must contain between 1 and {max_messages} items")
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("messages contain an invalid role or content")  # noqa: TRY004 - public validation contract
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("messages contain an invalid role or content")
        normalized_messages.append({"role": role, "content": content})
    # A complete JSON response gets the selected model's supported maximum.
    # max_tokens remains a legacy caller hint, not a smaller stage-level cap.
    budget = model_token_budget(base_url=base_url, model_id=model_id, messages=normalized_messages)
    effective_max_tokens = budget["effective_max_output_tokens"]
    stream_options = (
        {"progress_callback": progress_callback}
        if progress_callback is not None and official_deepseek else {}
    )
    if json_output and official_deepseek:
        stream_options["json_output"] = True
    allowance = generation_attempts if generation_attempts is not None else GenerationAttempts()
    first_attempt = len(allowance.rows)
    purpose = "initial" if not allowance.rows else "report_repair"
    adjusted_provider_limit = False
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise _call_failure(ModelProbeError("Generation cancelled.", category="cancelled"), allowance, first_attempt)
        row = allowance.begin(model_id=model_id, provider=infer_provider_id(base_url), messages=normalized_messages,
            options={"max_tokens": effective_max_tokens, "thinking_mode": normalized_thinking_mode,
                     "reasoning_effort": normalized_reasoning_effort, "json_output": json_output}, purpose=purpose,
            token_budget={**model_token_budget(base_url=base_url, model_id=model_id,
                messages=normalized_messages),
                "effective_max_output_tokens": effective_max_tokens})
        value = None
        try:
            value, latency_ms = _request_chat_json(
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
                messages=normalized_messages,
                max_tokens=effective_max_tokens,
                timeout_seconds=effective_timeout,
                max_response_bytes=max(MAX_DEEPSEEK_CHAT_RESPONSE_BYTES, (budget["provider_max_output_tokens"] or 0) * 32),
                user_agent="GWASMind/1.0 chat",
                thinking_mode=normalized_thinking_mode,
                reasoning_effort=normalized_reasoning_effort,
                reasoning_profile=profile,
                max_retries=0,
                **stream_options,
            )
            choices, observed = _validate_chat_response(value, model_id)
            message = choices[0].get("message")
            raw_content = message.get("content") if isinstance(message, dict) else None
            try:
                content = visible_completion_text(raw_content, allow_blocks=infer_provider_id(base_url) == "mistral")
            except ValueError as exc:
                raise ModelProbeError("The provider returned an unsupported chat content format.") from exc
            if not content.strip():
                raise ModelProbeError("The provider returned an empty chat response.", category="empty_response", retryable=True)
            allowance.finish(row, finish_reason=choices[0].get("finish_reason"), usage=_chat_usage(value))
            break
        except Exception as exc:
            suggested = getattr(exc, "suggested_output_limit", None)
            if (isinstance(suggested, int) and not isinstance(suggested, bool) and not adjusted_provider_limit
                    and (effective_max_tokens is None or suggested < effective_max_tokens)):
                effective_max_tokens = suggested
                budget["effective_max_output_tokens"] = suggested
                budget["budget_source"] = "provider_error_limit"
                adjusted_provider_limit = True
                ceiling_error = ModelProbeError("The provider specified its output-token ceiling.",
                    category="provider_output_limit", http_status=getattr(exc, "http_status", None), retryable=True)
                allowance.finish(row, failure=ceiling_error)
                if allowance.remaining_for("provider_output_limit_retry"):
                    purpose = "provider_output_limit_retry"
                    continue
                raise _call_failure(ceiling_error, allowance, first_attempt) from exc
            failure_usage = _chat_usage(value) if isinstance(value, Mapping) else None
            allowance.finish(row, failure=exc, usage=failure_usage)
            failure = classify_generation_failure(exc)
            next_purpose = failure["category"] + "_retry"
            available = allowance.remaining_for(next_purpose)
            if not failure["retryable"] or not available:
                _call_failure(exc, allowance, first_attempt)
                raise
            purpose = next_purpose
            if failure["category"] == "output_length":
                normalized_messages = [*normalized_messages, {"role": "user", "content":
                    "Regenerate the complete requested response concisely within the provider output limit. Preserve all scientific findings, evidence references and uncertainties; omit repeated prose."}]
            if _generation_backoff(allowance.current_retry_index, cancel_event):
                raise _call_failure(ModelProbeError("Generation cancelled.", category="cancelled"), allowance, first_attempt)
    return ModelChatResult(
        requested_model=model_id,
        observed_model=observed,
        content=content.strip(),
        latency_ms=latency_ms,
        thinking_mode=normalized_thinking_mode,
        reasoning_effort=(
            normalized_reasoning_effort
            if normalized_thinking_mode == THINKING_ENABLED
            else None
        ),
        usage=_attempt_usage(allowance.rows[first_attempt:]),
        generation_attempts=tuple(allowance.public()),
        token_budget=budget,
    )


def _call_failure(error: Exception, allowance: GenerationAttempts, first_attempt: int) -> Exception:
    """Keep terminal failures as auditable as successful calls, without changing their type."""
    error.generation_attempts = allowance.public()
    error.usage = _attempt_usage(allowance.rows[first_attempt:])
    return error


def _attempt_usage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Account for every provider response in this call, including truncated generations."""
    result: dict[str, int] = {}
    for row in rows:
        for key, count in (row.get("usage") or {}).items():
            if isinstance(count, int) and not isinstance(count, bool):
                result[key] = result.get(key, 0) + count
    return result


def _chat_usage(value: Mapping[str, Any]) -> dict[str, int]:
    """Return only bounded provider token counters safe for public progress."""

    raw = value.get("usage")
    if not isinstance(raw, Mapping):
        return {}
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = raw.get(key)
        if isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= 10_000_000:
            usage[key] = count
    details = raw.get("completion_tokens_details")
    if isinstance(details, Mapping):
        reasoning = details.get("reasoning_tokens")
        if (
            isinstance(reasoning, int)
            and not isinstance(reasoning, bool)
            and 0 <= reasoning <= 10_000_000
        ):
            usage["reasoning_tokens"] = reasoning
    return usage


def _provider_completion_budget(
    *,
    model_id: str,
    base_url: str,
    thinking_mode: str,
    reasoning_effort: str,
    requested_tokens: int | None,
) -> int | None:
    provider_budget = provider_completion_budget(
        model_id=model_id,
        base_url=base_url,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        requested_tokens=requested_tokens,
    )
    catalog_model = model_for_endpoint(base_url, model_id)
    if provider_budget is None or catalog_model is None or catalog_model.max_output_tokens is None:
        return provider_budget
    return min(provider_budget, catalog_model.max_output_tokens)


def _request_chat_json(
    *,
    api_key: str,
    base_url: str,
    model_id: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int | None,
    timeout_seconds: float,
    max_response_bytes: int,
    user_agent: str,
    thinking_mode: str,
    reasoning_effort: str,
    max_retries: int,
    reasoning_profile: Mapping[str, Any] | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    json_output: bool = False,
) -> tuple[dict[str, Any], int]:
    endpoint = _chat_completions_url(base_url)
    body: dict[str, Any] = {
        "model": model_id,
        "messages": [dict(message) for message in messages],
        "stream": False,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    hostname = (urlsplit(base_url).hostname or "").casefold().rstrip(".")
    if "max_tokens" in body and (hostname == "api.openai.com" or hostname.endswith(".openai.com")):
        body["max_completion_tokens"] = body.pop("max_tokens")
    body.update(
        reasoning_request_body(
            model_id=model_id,
            base_url=base_url,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            profile=reasoning_profile,
        )
    )
    if progress_callback is not None:
        body.update(stream=True, stream_options={"include_usage": True})
    if json_output:
        body["response_format"] = {"type": "json_object"}
    payload = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if progress_callback else "application/json",
            "User-Agent": user_agent,
        },
    )
    if progress_callback is not None:
        return _execute_stream_request(
            request, timeout_seconds=timeout_seconds, max_response_bytes=max_response_bytes,
            max_retries=max_retries, progress_callback=progress_callback,
        )
    return _execute_json_request(
        request,
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
        max_retries=max_retries,
    )


def _execute_stream_request(
    request: Request, *, timeout_seconds: float, max_response_bytes: int,
    max_retries: int, progress_callback: Callable[[dict[str, int]], None],
) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    attempt = 0
    while True:
        received = 0
        accumulator = CompletionStream(max_chars=max_response_bytes, on_progress=progress_callback)
        try:
            opener = build_opener(_RejectRedirects())
            with opener.open(request, timeout=timeout_seconds) as response:
                content_type = str(response.headers.get("Content-Type", ""))
                if "application/json" in content_type:
                    # A compatible gateway can ignore stream=True. Validate the
                    # normal envelope; absence of live counts is not a failure.
                    raw = response.read(max_response_bytes + 1)
                    if len(raw) > max_response_bytes:
                        raise ModelProbeError(
                            "The provider returned an unexpectedly large response."
                        )
                    value = json.loads(raw.decode("utf-8"))
                    if not isinstance(value, dict):
                        raise ModelProbeError("The provider returned an invalid response.")
                    return value, max(0, round((time.monotonic() - started) * 1000))
                done = False
                while True:
                    line = response.readline(max_response_bytes + 1)
                    if not line:
                        break
                    received += len(line)
                    if len(line) > max_response_bytes or received > min(
                        256 * 1024 * 1024, max_response_bytes * 16
                    ):
                        raise ModelProbeError(
                            "The provider returned an unexpectedly large response."
                        )
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    if not data:
                        continue
                    value = json.loads(data.decode("utf-8"))
                    if not isinstance(value, Mapping):
                        raise ModelProbeError("The provider returned an invalid response stream.")
                    accumulator.add(value)
                if not done:
                    raise ModelProbeError("The provider response stream ended before completion.", category="stream_interrupted", retryable=True)
            value = accumulator.completion()
            return value, max(0, round((time.monotonic() - started) * 1000))
        except HTTPError as exc:
            if not received and exc.code in _TRANSIENT_HTTP_STATUSES and attempt < max_retries:
                _retry_backoff(attempt)
                attempt += 1
                continue
            body = _bounded_error_body(exc)
            message = _http_failure_message(exc.code, body)
            failure = ModelProbeError(message, category="authentication" if exc.code in {401,403} else "rate_limit" if exc.code == 429 else "provider_unavailable" if exc.code in _TRANSIENT_HTTP_STATUSES else "request_rejected",
                                  http_status=exc.code, retryable=exc.code in _TRANSIENT_HTTP_STATUSES, provider_error=body)
            failure.suggested_output_limit = provider_output_limit(body) if exc.code in {400,422} else None
            raise failure from exc
        except CompletionStreamError as exc:
            raise ModelProbeError(str(exc), **exc.safe_details()) from exc
        except (UnicodeDecodeError, ValueError) as exc:
            raise ModelProbeError(
                "The provider returned an invalid or incomplete response stream.", category="stream_interrupted", retryable=True
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise ModelProbeError(
                "The provider response stream was interrupted. Please retry.", category="stream_interrupted", retryable=True
            ) from exc


def _execute_json_request(
    request: Request,
    *,
    timeout_seconds: float,
    max_response_bytes: int,
    not_found_message: str | None = None,
    max_retries: int = 0,
    allow_model_array: bool = False,
) -> tuple[dict[str, Any], int]:
    if not 0 <= max_retries <= MODEL_CHAT_MAX_RETRIES:
        raise ValueError(
            f"max_retries must be between 0 and {MODEL_CHAT_MAX_RETRIES}"
        )
    started = time.monotonic()
    attempt = 0
    while True:
        opener = build_opener(_RejectRedirects())
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                raw = response.read(max_response_bytes + 1)
            break
        except HTTPError as exc:
            body = _bounded_error_body(exc)
            if exc.code in _TRANSIENT_HTTP_STATUSES and attempt < max_retries:
                _retry_backoff(attempt)
                attempt += 1
                continue
            if exc.code == 404 and not_found_message:
                message = not_found_message
            else:
                message = _http_failure_message(exc.code, body)
            failure = ModelProbeError(message, category="authentication" if exc.code in {401,403} else "rate_limit" if exc.code == 429 else "provider_unavailable" if exc.code in _TRANSIENT_HTTP_STATUSES else "request_rejected",
                                  http_status=exc.code, retryable=exc.code in _TRANSIENT_HTTP_STATUSES, provider_error=body)
            failure.suggested_output_limit = provider_output_limit(body) if exc.code in {400,422} else None
            raise failure from exc
        except TimeoutError as exc:
            if attempt < max_retries:
                _retry_backoff(attempt)
                attempt += 1
                continue
            raise ModelProbeError(
                "The provider timed out. Check the API Base URL and network, then try again.", category="timeout", retryable=True
            ) from exc
        except URLError as exc:
            if attempt < max_retries:
                _retry_backoff(attempt)
                attempt += 1
                continue
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                message = (
                    "The provider timed out. Check the API Base URL and network, "
                    "then try again."
                )
            else:
                message = (
                    "Could not reach the provider. Check the API Base URL, internet access, "
                    "proxy, and firewall."
                )
            raise ModelProbeError(message, category="connection", retryable=True) from exc
        except OSError as exc:
            raise ModelProbeError(
                "Could not reach the provider. Check the API Base URL and network.", category="connection", retryable=True
            ) from exc

    if len(raw) > max_response_bytes:
        raise ModelProbeError("The provider returned an unexpectedly large response.")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ModelProbeError(
            "The provider returned an invalid response instead of OpenAI-compatible JSON.", category="json_structure", retryable=True
        ) from exc
    if allow_model_array and isinstance(value, list):
        value = {"data": value}
    if not isinstance(value, dict):
        raise ModelProbeError("The provider returned an invalid OpenAI-compatible response.")
    return value, max(0, round((time.monotonic() - started) * 1000))


def _retry_backoff(attempt: int) -> None:
    """Apply a short deterministic backoff before a transient retry."""

    time.sleep(min(10.0, 0.25 * (2**attempt)))


def _generation_backoff(retry_index: int, cancel_event: Any = None) -> bool:
    """Exponential wait before a generation transport retry; True when cancelled."""
    delay = min(30.0, 2.0 * (2**retry_index))
    if cancel_event is not None:
        return bool(cancel_event.wait(delay))
    time.sleep(delay)
    return False


def _validate_chat_response(
    value: Mapping[str, Any],
    model_id: str,
    *,
    accepted_finish_reasons: frozenset[str | None] = frozenset({None, "stop"}),
) -> tuple[list[Any], str]:
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ModelProbeError("The provider response did not contain a chat-completion result.")
    finish_reason = choices[0].get("finish_reason")
    if not (finish_reason is None or isinstance(finish_reason, str)) or finish_reason not in accepted_finish_reasons:
        safe_reason = finish_reason if finish_reason in ("length", "content_filter", "tool_calls", "function_call") else "invalid"
        raise ModelProbeError(
            "The provider did not complete the response safely "
            f"(finish_reason={safe_reason}).",
            category="output_length" if safe_reason == "length" else "provider_finish",
            finish_reason=safe_reason, retryable=safe_reason == "length",
        )
    observed = value.get("model")
    if not isinstance(observed, str) or not observed.strip():
        raise ModelProbeError("The provider response did not identify the model used.")
    observed = observed.strip()
    if observed != model_id:
        raise ModelProbeError(
            "The provider served a different model identifier than requested. "
            "Enter the exact provider model name."
        )
    return choices, observed


def _chat_completions_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if not path.casefold().endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _models_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    lowered = path.casefold()
    if lowered.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    if not path.casefold().endswith("/models"):
        path += "/models"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _bounded_error_body(error: HTTPError) -> bytes:
    try:
        return error.read(MAX_PROBE_RESPONSE_BYTES)
    except OSError:
        return b""


def _http_failure_message(status: int, body: bytes) -> str:
    hint = _provider_error_hint(body)
    if status in {301, 302, 303, 307, 308}:
        return (
            "The API endpoint redirected the authenticated request. Enter the provider's "
            "exact API Base URL."
        )
    if status in {401, 403}:
        return "Authentication failed. Check the API key and account access."
    if status == 402 or hint == "quota":
        return "The provider reports insufficient balance or quota for this API key."
    if status == 404:
        return (
            "The chat-completions endpoint or model was not found. Check the API Base URL "
            "and Model Name."
        )
    if status in {408, 504}:
        return "The provider timed out. Check the network and try again."
    if status == 429:
        return "The provider rate limit was reached. Wait briefly or check the account quota."
    if hint == "model":
        return "The provider does not recognize or permit this Model Name."
    if status in {400, 409, 415, 422}:
        return "The provider rejected the test request. Check the Model Name and API Base URL."
    if 500 <= status <= 599:
        return f"The provider service is unavailable (HTTP {status}). Try again later."
    return f"The provider rejected the connection test (HTTP {status})."


def _provider_error_hint(body: bytes) -> str | None:
    """Classify a bounded provider error without exposing its raw text."""

    try:
        value: Any = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        text = body.decode("utf-8", errors="ignore")
    else:
        error = value.get("error") if isinstance(value, dict) else None
        if isinstance(error, dict):
            text = " ".join(str(error.get(key, "")) for key in ("code", "type", "message"))
        else:
            text = str(error or "")
    normalized = " ".join(text.casefold().split())[:2_000]
    if any(
        marker in normalized
        for marker in (
            "model_not_found",
            "model not found",
            "model does not exist",
            "unknown model",
            "invalid model",
            "model not exist",
        )
    ):
        return "model"
    if any(
        marker in normalized
        for marker in (
            "insufficient balance",
            "insufficient quota",
            "billing",
            "credit balance",
        )
    ):
        return "quota"
    return None
