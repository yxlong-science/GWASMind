"""Explicit model reasoning controls for OpenAI-compatible request bodies.

Capabilities describe the selected provider's contract, not the underlying
weights in isolation. Unknown models deliberately use the provider default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


KINDS = frozenset({"toggle_effort", "effort", "toggle", "fixed", "none", "unknown"})
MODES = frozenset({"auto", "enabled", "disabled"})
EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class ProviderReasoningError(ValueError):
    """A safe configuration error, without provider credentials or payloads."""


@dataclass(frozen=True)
class _Contract:
    kind: str
    efforts: tuple[str, ...] = ()
    can_disable: bool = False
    default_mode: str = "auto"
    default_effort: str = "default"
    note: str = ""
    source_url: str = ""
    wire_style: str = ""
    hidden: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "efforts": list(self.efforts),
            "can_disable": self.can_disable,
            "default_mode": self.default_mode,
            "default_effort": self.default_effort,
            "note": self.note,
            "source_url": self.source_url,
        }


_UNKNOWN = _Contract(
    "unknown",
    note="Reasoning controls are not verified for this model and endpoint. Provider defaults will be used.",
)
_CONTRACTS: dict[tuple[str, str], _Contract] = {}


def _register(providers: tuple[str, ...], models: tuple[str, ...], contract: _Contract) -> None:
    for provider in providers:
        for model in models:
            _CONTRACTS[(provider, model)] = contract


_register(
    ("deepseek",),
    ("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"),
    _Contract("toggle_effort", ("low", "high", "max"), True, "enabled", "high",
              "Thinking can be disabled; enabled thinking has three depth levels.",
              "https://api-docs.deepseek.com/guides/thinking_mode/", "thinking_effort"),
)
_register(
    ("openai",), ("gpt-5.4",),
    _Contract("effort", ("none", "low", "medium", "high", "xhigh"), True, "disabled", "none",
              "Use None for no reasoning, or select a reasoning depth.",
              "https://developers.openai.com/api/docs/models/gpt-5.4", "effort"),
)
_register(
    ("openai",), ("gpt-4.1", "gpt-4.1-mini"),
    _Contract("none", default_mode="disabled", note="This model has no configurable reasoning mode.",
              source_url="https://developers.openai.com/api/docs/models/gpt-4.1"),
)
_register(
    ("xai",), ("grok-4.3",),
    _Contract("effort", ("none", "low", "medium", "high", "xhigh"), True, "enabled", "low",
              "Reasoning is configured through depth; None disables it.",
              "https://docs.x.ai/developers/models/grok-4.3", "effort"),
)
_register(
    ("moonshotai", "moonshotai-cn"), ("kimi-k2.5", "kimi-k2.6"),
    _Contract("toggle", (), True, "enabled", "default",
              "Thinking can be enabled or disabled. This model does not support adjustable reasoning depth.",
              "https://platform.kimi.ai/docs/api/models-overview", "thinking"),
)
_register(
    ("moonshotai", "moonshotai-cn"), ("kimi-k3",),
    _Contract("effort", ("low", "high", "max"), False, "enabled", "max",
              "This model always reasons; choose its depth.",
              "https://platform.kimi.ai/docs/guide/use-reasoning-effort", "effort"),
)
_register(
    ("moonshotai", "moonshotai-cn"), ("kimi-k2.7-code", "kimi-k2.7-code-highspeed"),
    _Contract("fixed", default_mode="enabled",
              note="Reasoning is always enabled; this model does not expose depth controls.",
              source_url="https://platform.kimi.ai/docs/api/models-overview"),
)
_register(
    ("zai", "zai-coding-cn"), ("glm-4.7", "glm-5.1", "glm-5-turbo"),
    _Contract("toggle", (), True, "enabled", "default",
              "Thinking can be enabled or disabled; adjustable depth is not supported for this model.",
              "https://docs.z.ai/guides/capabilities/thinking", "thinking"),
)
_register(
    ("zai", "zai-coding-cn"), ("glm-5.2",),
    _Contract("toggle_effort", ("high", "max"), True, "enabled", "max",
              "Thinking can be disabled; High and Maximum are the two distinct enabled depth levels.",
              "https://docs.z.ai/guides/capabilities/thinking", "thinking_effort"),
)
_register(
    ("zai", "zai-coding-cn"), ("glm-5.3", "glm-5.3-flash"),
    _Contract("effort", ("low", "high", "max"), False, "enabled", "max",
              "This model always reasons; lower depth does not disable thinking.",
              "https://docs.z.ai/guides/capabilities/thinking", "effort"),
)
_register(
    ("xiaomi", "xiaomi-token-plan-ams", "xiaomi-token-plan-cn", "xiaomi-token-plan-sgp"),
    ("mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro"),
    _Contract("toggle", (), True, "enabled", "default",
              "Thinking can be enabled or disabled. Fine-grained reasoning depth is not supported.",
              "https://mimo.mi.com/docs/en-US/api/chat/openai-api", "thinking"),
)
_register(
    ("ant-ling",), ("Ling-2.6-1T", "Ling-2.6-flash"),
    _Contract("none", default_mode="disabled", note="This model has no configurable reasoning mode.",
              source_url="https://developer.ant-ling.com/zh-CN/docs/api-reference/openai/"),
)
_register(
    ("ant-ling",), ("Ring-2.6-1T",),
    _Contract("effort", ("high", "xhigh"), False, "enabled", "high",
              "This reasoning model supports High and Extra high; no off mode is documented.",
              "https://developer.ant-ling.com/zh-CN/docs/tutorials/effort/", "nested_effort"),
)
_register(
    ("ant-ling",), ("Ling-3.0-flash",),
    _Contract("toggle", (), True, "enabled", "default",
              "Thinking can be enabled or disabled; no depth control is documented.",
              "https://developer.ant-ling.com/zh-CN/docs/api-reference/openai/", "thinking"),
)
_register(
    ("groq",), ("openai/gpt-oss-120b",),
    _Contract("effort", ("low", "medium", "high"), False, "enabled", "medium",
              "This model always reasons; choose its depth.",
              "https://console.groq.com/docs/reasoning", "effort"),
)
_register(
    ("groq",), ("qwen/qwen3-32b",),
    _Contract("toggle", (), True, "enabled", "default",
              "Thinking has an on/off switch only. This model was retired; refresh the model list.",
              "https://console.groq.com/docs/reasoning", "groq_toggle", True),
)
_register(
    ("groq",), ("llama-3.3-70b-versatile",),
    _Contract("none", default_mode="disabled", note="This model has no configurable reasoning mode.",
              source_url="https://console.groq.com/docs/models"),
)
_register(
    ("cerebras",), ("gpt-oss-120b",),
    _Contract("effort", ("low", "medium", "high"), False, "enabled", "medium",
              "This model always reasons; choose its depth.",
              "https://inference-docs.cerebras.ai/capabilities/reasoning", "effort", True),
)
_register(
    ("cerebras",), ("zai-glm-4.7",),
    _Contract("toggle", (), True, "enabled", "default",
              "Thinking has an on/off switch; this endpoint does not expose depth levels for this model.",
              "https://inference-docs.cerebras.ai/capabilities/reasoning", "cerebras_toggle", True),
)
_register(
    ("google",), ("gemini-3.1-pro-preview",),
    _Contract("effort", ("low", "medium", "high"), False, "enabled", "high",
              "Thinking cannot be disabled; choose the reasoning depth.",
              "https://ai.google.dev/gemini-api/docs/openai", "effort"),
)
_register(
    ("google",), ("gemini-3.5-flash",),
    _Contract("effort", ("minimal", "low", "medium", "high"), False, "enabled", "high",
              "Thinking cannot be disabled. Minimal is the lowest reasoning depth.",
              "https://ai.google.dev/gemini-api/docs/openai", "effort"),
)
_register(
    ("mistral",), ("mistral-small-2603",),
    _Contract("effort", ("none", "high"), False, "enabled", "none",
              "None means minimal reasoning for this model; it does not completely disable reasoning.",
              "https://docs.mistral.ai/capabilities/reasoning", "effort"),
)
_register(
    ("together",), ("deepseek-ai/DeepSeek-V4-Pro",),
    _Contract("toggle_effort", ("high", "max"), True, "enabled", "high",
              "This endpoint supports a thinking switch and two distinct depth levels.",
              "https://docs.together.ai/docs/deepseek-v4-quickstart", "together"),
)
_register(
    ("together",), ("moonshotai/Kimi-K2.6",),
    _Contract("toggle", (), True, "enabled", "default",
              "Thinking can be enabled or disabled; no adjustable depth is documented.",
              "https://docs.together.ai/docs/inference/chat/reasoning", "together"),
)
_register(
    ("fireworks",), ("accounts/fireworks/models/glm-5p2", "accounts/fireworks/routers/glm-5p2-fast"),
    _Contract("effort", ("none", "high", "max"), True, "enabled", "max",
              "Use None to disable reasoning, or select High or Maximum.",
              "https://docs.fireworks.ai/guides/reasoning", "effort"),
)
_register(
    ("nvidia",), ("nvidia/nemotron-3-ultra-550b-a55b",),
    _Contract("effort", ("none", "medium", "high"), True, "enabled", "high",
              "Use None to disable reasoning, or select Medium or High.",
              "https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b-infer", "effort"),
)
_register(
    ("nvidia",), ("moonshotai/kimi-k2.6",),
    _Contract("toggle", (), True, "enabled", "default",
              "This hosted endpoint provides a thinking switch, without adjustable depth.",
              "https://docs.api.nvidia.com/nim/reference/moonshotai-kimi-k2-6-infer", "nvidia_thinking"),
)
_register(
    ("nvidia",), ("openai/gpt-oss-120b",),
    _Contract("effort", ("low", "medium", "high"), False, "enabled", "medium",
              "This model always reasons; choose its depth.",
              "https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-120b-infer", "effort"),
)
_register(
    ("opencode", "opencode-go"), ("deepseek-v4-pro", "glm-5.2"),
    _Contract("effort", ("high", "max"), False, "auto", "default",
              "High and Maximum are verified for this endpoint. An explicit off mode is not verified.",
              "https://opencode.ai/docs/zen/", "effort"),
)
_register(
    ("vercel-ai-gateway",), ("zai/glm-5.2",),
    _Contract("toggle_effort", ("high", "xhigh"), True, "auto", "default",
              "This gateway exposes a thinking switch and High or Extra high depth. Provider defaults are preserved.",
              "https://vercel.com/docs/ai-gateway/models-and-providers/reasoning", "vercel"),
)
_register(
    ("vercel-ai-gateway",), ("moonshotai/kimi-k2.6",),
    _Contract("toggle", (), True, "auto", "default",
              "This gateway exposes a thinking switch without adjustable depth. Provider defaults are preserved.",
              "https://vercel.com/docs/ai-gateway/models-and-providers/reasoning", "vercel"),
)
_register(
    ("vercel-ai-gateway",), ("openai/gpt-5.6-sol",),
    _Contract("effort", ("none", "low", "medium", "high", "xhigh", "max"), True, "auto", "default",
              "Choose a supported reasoning depth or None. Provider defaults are preserved when no depth is selected.",
              "https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions/reasoning", "vercel"),
)

_OPENROUTER_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_OPENROUTER_UNKNOWN = _Contract(
    "unknown", note="Refresh the model list to verify this model's reasoning controls for OpenRouter.",
    source_url="https://openrouter.ai/docs/guides/best-practices/reasoning-tokens",
)


def parse_openrouter_reasoning(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one model's advertised reasoning object without guessing.

    Missing/invalid mandatory metadata yields unknown for this row only. A null
    effort list has the documented gateway vocabulary; an omitted list supplies
    no evidence for adjustable depth. Unrelated budget metadata is not exposed.
    """
    unknown = _OPENROUTER_UNKNOWN.public()
    if not isinstance(metadata, Mapping) or type(metadata.get("mandatory")) is not bool:
        return unknown
    mandatory = metadata["mandatory"]
    default_enabled = metadata.get("default_enabled")
    if "default_enabled" in metadata and type(default_enabled) is not bool:
        return unknown
    if mandatory and default_enabled is False:
        return unknown
    if "supported_efforts" not in metadata:
        efforts: list[str] = []
    elif metadata["supported_efforts"] is None:
        efforts = list(_OPENROUTER_EFFORTS)
    else:
        raw = metadata["supported_efforts"]
        if (not isinstance(raw, list) or len(raw) > 32
                or any(not isinstance(v, str) or v not in _OPENROUTER_EFFORTS for v in raw)):
            return unknown
        efforts = [v for v in _OPENROUTER_EFFORTS if v in raw]
    # The separate switch implements None; it is not an enabled depth. A
    # mandatory model cannot receive the gateway's non-reasoning effort.
    efforts = [v for v in efforts if v != "none"]
    default_effort = metadata.get("default_effort", "default")
    if default_effort == "none" and not mandatory and default_enabled is not True:
        default_effort = "default"
    if (not isinstance(default_effort, str)
            or default_effort not in ("default", *efforts)):
        return unknown
    default_mode = "auto" if default_enabled is None else ("enabled" if default_enabled else "disabled")
    kind = ("effort" if mandatory else "toggle_effort") if efforts else ("fixed" if mandatory else "toggle")
    # Fixed describes mandatory computation, even when the optional default
    # field was omitted. Keep default_mode auto in the public metadata to avoid
    # inventing that field, while kind/can_disable retain the mandatory fact.
    return _Contract(
        kind, tuple(efforts), not mandatory, default_mode, default_effort,
        "Controls reflect the selected model's latest OpenRouter metadata.",
        _OPENROUTER_UNKNOWN.source_url, "openrouter",
    ).public()


def reasoning_for_model(provider_id: str, model_id: str) -> dict[str, Any]:
    """Return a fresh public capability for an exact provider/model pair."""
    if not isinstance(provider_id, str) or not isinstance(model_id, str):
        return _UNKNOWN.public()
    if provider_id == "openrouter":
        return _OPENROUTER_UNKNOWN.public()
    return _CONTRACTS.get((provider_id, model_id), _UNKNOWN).public()


def _normalized_endpoint(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 2048 or value != value.strip():
        return ""
    if any(ord(c) < 32 or c.isspace() for c in value):
        return ""
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            return ""
        port = parsed.port
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        if port is not None and (parsed.scheme, port) not in {("https", 443), ("http", 80)}:
            host += f":{port}"
        return f"{parsed.scheme}://{host}{parsed.path.rstrip('/')}"
    except (ValueError, AttributeError):
        return ""


def _profile_contract(profile: Any, *, base_url: str, model_id: str) -> _Contract:
    error = "The model reasoning profile does not match this connection or is invalid. Refresh the model list."
    if not isinstance(profile, Mapping) or set(profile) != {"provider_id", "base_url", "model_id", "reasoning"}:
        raise ProviderReasoningError(error)
    if (profile["provider_id"] != "openrouter" or _openrouter_endpoint_root(base_url) != _OPENROUTER_BASE
            or _openrouter_endpoint_root(profile["base_url"]) != _OPENROUTER_BASE
            or profile["model_id"] != model_id or not isinstance(model_id, str)
            or not 1 <= len(model_id) <= 120 or any(ord(c) < 32 for c in model_id)):
        raise ProviderReasoningError(error)
    value = profile["reasoning"]
    keys = {"kind", "efforts", "can_disable", "default_mode", "default_effort", "note", "source_url"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ProviderReasoningError(error)
    kind, efforts = value["kind"], value["efforts"]
    if (not isinstance(kind, str) or kind not in KINDS or not isinstance(efforts, list)
            or len(efforts) > len(_OPENROUTER_EFFORTS) - 1
            or any(not isinstance(v, str) or v not in _OPENROUTER_EFFORTS[1:] for v in efforts)
            or len(set(efforts)) != len(efforts) or type(value["can_disable"]) is not bool
            or not isinstance(value["default_mode"], str) or value["default_mode"] not in MODES
            or not isinstance(value["default_effort"], str)
            or value["default_effort"] not in ("default", *efforts)
            or not isinstance(value["note"], str) or len(value["note"]) > 512
            or not isinstance(value["source_url"], str) or len(value["source_url"]) > 2048):
        raise ProviderReasoningError(error)
    if (any(ord(c) < 32 for c in value["note"])
            or (value["source_url"] and not _normalized_endpoint(value["source_url"]).startswith("https://"))):
        raise ProviderReasoningError(error)
    if (kind in {"toggle_effort", "effort"}) != bool(efforts):
        raise ProviderReasoningError(error)
    if kind in {"toggle_effort", "toggle"} and not value["can_disable"]:
        raise ProviderReasoningError(error)
    if kind in {"effort", "fixed", "none", "unknown"} and value["can_disable"]:
        raise ProviderReasoningError(error)
    if kind == "fixed" and value["default_mode"] == "disabled":
        raise ProviderReasoningError(error)
    if kind == "unknown" and value["default_mode"] != "auto":
        raise ProviderReasoningError(error)
    if kind == "none" and value["default_mode"] == "enabled":
        raise ProviderReasoningError(error)
    return _Contract(**{**value, "efforts": tuple(efforts)}, wire_style="openrouter")


def _openrouter_endpoint_root(value: Any) -> str:
    """Bind profiles across only the catalog's equivalent official URLs."""
    if not _normalized_endpoint(value):
        return ""
    from .provider_catalog import ProviderCatalogError, validate_provider_endpoint

    try:
        validated = validate_provider_endpoint(provider_id="openrouter", base_url=value)
    except ProviderCatalogError:
        return ""
    canonical = validated.base_url
    if canonical in {_OPENROUTER_BASE, f"{_OPENROUTER_BASE}/chat/completions"}:
        return _OPENROUTER_BASE
    return ""


def _contract_for_endpoint(base_url: str, model_id: str, profile: Any = None) -> _Contract:
    if profile is not None:
        return _profile_contract(profile, base_url=base_url, model_id=model_id)
    # Lazy import avoids the catalog -> public capability -> catalog cycle.
    from .provider_catalog import infer_provider_id

    if not isinstance(base_url, str) or not isinstance(model_id, str):
        return _UNKNOWN
    provider_id = infer_provider_id(base_url)
    if provider_id == "openrouter":
        return _OPENROUTER_UNKNOWN
    return _CONTRACTS.get((provider_id, model_id), _UNKNOWN)


def reasoning_for_endpoint(base_url: str, model_id: str, profile: Any = None) -> dict[str, Any]:
    """Resolve controls without letting a proxy inherit a first-party contract."""
    return _contract_for_endpoint(base_url, model_id, profile).public()


def reasoning_request_body(
    *, base_url: str, model_id: str, thinking_mode: str,
    reasoning_effort: str, profile: Any = None,
) -> dict[str, Any]:
    """Return only documented reasoning fields, suitable for a JSON body.

    Provider defaults omit all switching/depth fields. An unsupported explicit
    request is rejected instead of silently switching to a different behavior.
    Stale depth values on switch-only models are discarded, never forwarded.
    """
    if not isinstance(thinking_mode, str) or thinking_mode not in MODES:
        raise ProviderReasoningError("Thinking mode must be auto, enabled, or disabled.")
    if not isinstance(reasoning_effort, str) or reasoning_effort not in EFFORTS | {"default"}:
        raise ProviderReasoningError("The reasoning depth value is not recognized.")
    contract = _contract_for_endpoint(base_url, model_id, profile)
    body: dict[str, Any] = {"reasoning_format": "hidden"} if contract.hidden else {}
    if contract.kind in {"none", "unknown"}:
        if thinking_mode == "enabled":
            raise ProviderReasoningError("Reasoning controls are not available for this model and endpoint.")
        return body
    if thinking_mode == "disabled" and not contract.can_disable:
        raise ProviderReasoningError("This model cannot disable reasoning. Choose a supported depth or provider default.")
    if contract.wire_style in {"openrouter", "vercel"}:
        body["reasoning"] = {"exclude": True}
    if thinking_mode == "auto" or contract.kind == "fixed":
        return body
    if contract.kind in {"effort", "toggle_effort"} and thinking_mode == "enabled":
        if reasoning_effort != "default" and reasoning_effort not in contract.efforts:
            raise ProviderReasoningError("This reasoning depth is not supported by the selected model.")
    style = contract.wire_style
    if style == "openrouter":
        body["reasoning"]["enabled"] = thinking_mode == "enabled"
        if thinking_mode == "enabled" and contract.efforts and reasoning_effort != "default":
            body["reasoning"]["effort"] = reasoning_effort
    elif style == "vercel":
        if contract.kind == "effort":
            effort = "none" if thinking_mode == "disabled" else reasoning_effort
            if effort != "default":
                body["reasoning"]["effort"] = effort
        else:
            body["reasoning"]["enabled"] = thinking_mode == "enabled"
            if contract.efforts and thinking_mode == "enabled" and reasoning_effort != "default":
                body["reasoning"]["effort"] = reasoning_effort
    elif style in {"thinking", "thinking_effort"}:
        body["thinking"] = {"type": thinking_mode}
        if style == "thinking_effort" and thinking_mode == "enabled" and reasoning_effort != "default":
            body["reasoning_effort"] = reasoning_effort
    elif style == "groq_toggle":
        body["reasoning_effort"] = "default" if thinking_mode == "enabled" else "none"
    elif style == "cerebras_toggle":
        if thinking_mode == "disabled":
            body["reasoning_effort"] = "none"
    elif style == "together":
        body["reasoning"] = {"enabled": thinking_mode == "enabled"}
        if contract.efforts and thinking_mode == "enabled" and reasoning_effort != "default":
            body["reasoning_effort"] = reasoning_effort
    elif style == "nvidia_thinking":
        body["chat_template_kwargs"] = {"thinking": thinking_mode == "enabled"}
    elif style in {"effort", "nested_effort"}:
        effort = "none" if thinking_mode == "disabled" else reasoning_effort
        if effort != "default":
            if style == "nested_effort":
                body["reasoning"] = {"effort": effort}
            else:
                body["reasoning_effort"] = effort
    return body
