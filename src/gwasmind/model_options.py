from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from .configuration import parameter

DEFAULT_MODEL_BASE_URL = parameter("model.default_base_url")
DEFAULT_MODEL_ID = parameter("model.default_id")
THINKING_ENABLED = "enabled"
THINKING_DISABLED = "disabled"
THINKING_AUTO = "auto"
THINKING_MODES = frozenset({THINKING_ENABLED, THINKING_DISABLED, THINKING_AUTO})
DEFAULT_REASONING_EFFORT = parameter("model.default_reasoning_effort")
REASONING_EFFORTS = frozenset({"default", "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})
# Official Chat Completions API states 384K = 393216; the official compatible
# client catalog states 1M context = 1048576 (verified 2026-09-13).
DEEPSEEK_MAX_OUTPUT_TOKENS = 393_216
DEEPSEEK_CONTEXT_TOKENS = 1_048_576


def normalize_thinking_mode(value: Any) -> str:
    """Validate the explicit provider thinking-mode setting."""

    if not isinstance(value, str) or value not in THINKING_MODES:
        raise ValueError("thinking_mode must be 'auto', 'enabled', or 'disabled'")
    return value


def normalize_reasoning_effort(value: Any) -> str:
    """Validate the transport vocabulary; each model further restricts its values."""

    if not isinstance(value, str) or value not in REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be a recognized reasoning level or 'default'")
    return value


def default_thinking_mode(*, model_id: str, base_url: str | None) -> str:
    """Return the safe first-run default without changing later user choices."""

    if model_id.strip().casefold() == DEFAULT_MODEL_ID and is_deepseek_model_endpoint(
        model_id=model_id,
        base_url=base_url,
    ):
        return parameter("model.first_run_thinking_mode")
    from .provider_reasoning import reasoning_for_endpoint
    return reasoning_for_endpoint(base_url or "", model_id)["default_mode"]


def is_deepseek_model_endpoint(*, model_id: str, base_url: str | None) -> bool:
    """Recognize the official DeepSeek endpoint.

    A custom OpenAI-compatible proxy may or may not forward DeepSeek's
    provider-specific ``thinking`` field, so GWASMind does not guess.
    """

    if not isinstance(model_id, str) or not model_id.strip():
        return False
    hostname = (urlsplit(base_url or "").hostname or "").casefold().rstrip(".")
    return hostname == "api.deepseek.com" or hostname.endswith(".deepseek.com")


def deepseek_thinking_body(
    *,
    model_id: str,
    base_url: str | None,
    thinking_mode: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Build the DeepSeek-compatible request extension only when applicable."""

    normalized = normalize_thinking_mode(thinking_mode)
    effort = normalize_reasoning_effort(reasoning_effort)
    if not is_deepseek_model_endpoint(model_id=model_id, base_url=base_url):
        return {}
    if normalized == THINKING_AUTO:
        return {}
    body: dict[str, Any] = {"thinking": {"type": normalized}}
    if normalized == THINKING_ENABLED and effort != "default":
        body["reasoning_effort"] = effort
    return body


def provider_completion_budget(
    *,
    model_id: str,
    base_url: str | None,
    thinking_mode: str,
    reasoning_effort: str,
    requested_tokens: int | None = None,
) -> int | None:
    """Return an exact endpoint/model ceiling; unknown models use provider defaults.

    The legacy requested_tokens argument is validated for compatibility but no
    longer imposes a separate application ceiling on scientific generation.
    Reasoning settings and deterministic scientific validation remain unchanged.
    """

    normalize_thinking_mode(thinking_mode)
    normalize_reasoning_effort(reasoning_effort)
    if requested_tokens is not None and (isinstance(requested_tokens, bool) or not isinstance(requested_tokens, int) or requested_tokens < 1):
        raise ValueError("requested_tokens must be positive")
    # Output allowance is a model capability, independent of reasoning settings.
    # Unknown IDs use the provider's default instead of an invented application cap.
    from .provider_catalog import model_for_endpoint
    model = model_for_endpoint(base_url or "", model_id)
    return model.max_output_tokens if model is not None else None


def model_token_budget(*, base_url: str | None, model_id: str, messages: Any = None,
                       task_max_output_tokens: int | None = None,
                       confirmed_context_remaining_tokens: int | None = None) -> dict[str, Any]:
    """Separate model capability, optional task allowance and confirmed capacity.

    A provider-specific tokenizer is not available for every compatible endpoint.
    The estimate counts model-visible content, including the complete embedded
    evidence JSON and system prompt, not HTTP JSON escape characters. An estimate
    cannot silently lower the chosen output allowance or switch to provider defaults.
    The provider's tokenizer enforces actual joint input/output capacity.
    """
    from .provider_catalog import model_for_endpoint
    model = model_for_endpoint(base_url or "", model_id)
    maximum = model.max_output_tokens if model else None
    context = model.context_window if model else None
    if task_max_output_tokens is not None and (type(task_max_output_tokens) is not int or task_max_output_tokens <= 0):
        raise ValueError("task_max_output_tokens must be positive or omitted")
    if confirmed_context_remaining_tokens is not None and (type(confirmed_context_remaining_tokens) is not int or confirmed_context_remaining_tokens < 0):
        raise ValueError("confirmed_context_remaining_tokens must be nonnegative or omitted")
    limits = [value for value in (maximum, task_max_output_tokens, confirmed_context_remaining_tokens) if value is not None]
    effective = min(limits) if limits else None
    estimate = None
    prompt_bytes = None
    request_bytes = None
    if messages is not None:
        import json
        encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request_bytes = len(encoded)
        prompt_bytes = sum(len(str(message.get("content", "")).encode("utf-8"))
                           for message in messages)
        import math
        from urllib.parse import urlparse
        from .configuration import parameter
        provider_key = "deepseek" if urlparse(base_url or "").hostname == "api.deepseek.com" else "default"
        import re
        table_cards = provider_key == "deepseek" and any(
            re.search(r'"encoding"\s*:\s*"gwasmind_v2g_scientific_core_v1"', str(message.get("content", "")))
            and re.search(r'"shared_definitions"\s*:\s*\{\s*\}', str(message.get("content", "")))
            for message in messages)
        ratio_key = "deepseek_native_table" if table_cards else provider_key
        ratio = parameter(f"model.token_estimate.{ratio_key}_bytes_per_token")
        estimate = math.ceil(prompt_bytes / ratio) + 4 * len(messages) + 32
    exceeds = bool(context and estimate is not None and estimate >= context)
    # A model whose advertised output ceiling equals its entire context cannot
    # reserve that whole ceiling alongside nonempty input. Do not invent a fixed
    # reserve for it; the provider must supply the effective joint limit.
    input_limit = context - effective if context and effective and context > effective else None
    return {"context_window": context, "provider_max_output_tokens": maximum,
            "effective_max_output_tokens": effective, "prompt_tokens_estimate": estimate,
            **({"task_max_output_tokens": task_max_output_tokens,
                "confirmed_context_remaining_tokens": confirmed_context_remaining_tokens}
               if task_max_output_tokens is not None or confirmed_context_remaining_tokens is not None else {}),
            "prompt_utf8_bytes": prompt_bytes, "serialized_messages_bytes": request_bytes,
            "token_count_method": "provider_calibrated_utf8_plus_framing_estimate" if estimate is not None else None,
            "bytes_per_token": ratio if estimate is not None else None,
            "token_estimate_profile": ratio_key if estimate is not None else None,
            "estimated_context_remaining_tokens": max(0, context - estimate) if context and estimate is not None else None,
            "input_estimate_exceeds_context": exceeds,
            "input_token_budget": input_limit,
            "input_budget_reason": "context_minus_task_output_allowance" if input_limit is not None and task_max_output_tokens is not None else
                "context_minus_provider_maximum_output" if input_limit is not None else "provider_joint_limit_required",
            "budget_source": "versioned_task_policy" if task_max_output_tokens is not None else "provider_catalog" if model else "provider_default",
            "budget_reason": "minimum_of_task_provider_and_confirmed_context_limits" if task_max_output_tokens is not None else
                "known_model_maximum_provider_enforces_joint_context" if model else "unknown_model_limit",
            "context_authority": "provider_tokenizer"}
