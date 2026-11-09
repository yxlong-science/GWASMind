"""Bounded completion assembly with numeric-only public streaming telemetry."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from .configuration import parameter
from .model_generation import GenerationFailure


class CompletionStreamError(GenerationFailure, ValueError):
    """A provider stream could not form one complete text response."""


def bounded_token_usage(raw: Any) -> dict[str, Any]:
    """Retain provider token counters, never arbitrary metadata or model prose."""

    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
        "cached_tokens", "audio_tokens", "accepted_prediction_tokens",
        "rejected_prediction_tokens",
    ):
        count = raw.get(key)
        if isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= 10_000_000:
            result[key] = count
    for key in ("prompt_tokens_details", "completion_tokens_details"):
        details = raw.get(key)
        if isinstance(details, Mapping):
            counters = {
                name: count for name in (
                    "cached_tokens", "audio_tokens", "reasoning_tokens",
                    "accepted_prediction_tokens", "rejected_prediction_tokens",
                )
                if isinstance(count := details.get(name), int)
                and not isinstance(count, bool) and 0 <= count <= 10_000_000
            }
            if counters:
                result[key] = counters
    return result


class CompletionStream:
    """Keep answer text private until validation; never retain reasoning text.

    Providers generally report token usage only in the final chunk. Before then,
    the optional token estimate is explicitly based on received characters / 4;
    it is never substituted for provider usage in the resulting completion.
    """

    def __init__(
        self,
        *,
        max_chars: int = parameter("transport.ranking_stream_max_chars"),
        on_progress: Callable[[dict[str, int]], None] | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.on_progress = on_progress
        self.parts: list[str] = []
        self.received_chars = 0
        self.model: str | None = None
        self.request_id: str | None = None
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] = {}
        self.last_progress = float("-inf")

    def add(self, chunk: Mapping[str, Any]) -> None:
        if "error" in chunk:
            raise CompletionStreamError("The provider interrupted the response stream.", category="stream_interrupted", retryable=True)
        model = chunk.get("model")
        if isinstance(model, str) and model:
            if self.model is not None and self.model != model:
                raise CompletionStreamError(
                    "The provider changed model during the response stream."
                )
            self.model = model
        identifier = chunk.get("id")
        if isinstance(identifier, str) and identifier:
            if self.request_id is not None and self.request_id != identifier:
                raise CompletionStreamError("The provider changed response during the stream.")
            self.request_id = identifier
        usage = chunk.get("usage")
        if isinstance(usage, Mapping):
            self.usage.update(bounded_token_usage(usage))
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            raise CompletionStreamError("The provider returned an invalid response stream.")
        for choice in choices:
            if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
                continue
            if self.finish_reason is not None:
                raise CompletionStreamError("The provider continued an already finished response.")
            reason = choice.get("finish_reason")
            if reason is not None:
                self.finish_reason = str(reason)
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            if delta.get("tool_calls") or delta.get("function_call"):
                raise CompletionStreamError("The provider returned an unexpected tool response.")
            content = delta.get("content")
            if isinstance(content, str):
                self.parts.append(content)
                self.received_chars += len(content)
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str):
                # Count only. Reasoning text never enters a callback or artifact.
                self.received_chars += len(reasoning)
            if self.received_chars > self.max_chars:
                raise CompletionStreamError("The provider returned an unexpectedly large response.")
        now = time.monotonic()
        if now - self.last_progress >= parameter("execution.stream_progress_interval_seconds") or isinstance(usage, Mapping):
            self.publish()
            self.last_progress = now

    def publish(self) -> None:
        if self.on_progress is None:
            return
        metrics = {"received_chars": self.received_chars}
        count = self.usage.get("completion_tokens")
        if isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= 10_000_000:
            metrics["output_tokens"] = count
        elif self.received_chars:
            metrics["estimated_output_tokens"] = (self.received_chars + 3) // 4
        try:
            self.on_progress(metrics)
        except Exception:  # noqa: BLE001, S110 - reporting cannot alter inference
            # Telemetry must not alter inference or weaken output validation.
            pass

    def completion(self) -> dict[str, Any]:
        model_missing = not isinstance(self.model, str) or not self.model.strip()
        if self.finish_reason != "stop" or model_missing:
            # Never put provider-controlled text or unfinished content in diagnostics.
            safe_reason = (
                "missing" if self.finish_reason is None
                else self.finish_reason if self.finish_reason in (
                    "stop", "length", "content_filter", "tool_calls", "function_call"
                ) else "invalid"
            )
            raise CompletionStreamError(
                "The provider did not finish the response stream safely "
                f"(finish_reason={safe_reason}, model_missing={str(model_missing).lower()}).",
                category="model_identity" if model_missing else "output_length" if safe_reason == "length" else "stream_interrupted" if safe_reason == "missing" else "provider_finish",
                finish_reason=self.finish_reason,
                retryable=not model_missing and safe_reason in {"length", "missing"},
            )
        self.publish()
        return {
            "id": self.request_id,
            "model": self.model,
            "choices": [{"finish_reason": self.finish_reason,
                         "message": {"role": "assistant", "content": "".join(self.parts)}}],
            "usage": self.usage,
        }
