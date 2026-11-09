from __future__ import annotations

import ipaddress
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from .model_options import (
    DEEPSEEK_CONTEXT_TOKENS,
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_BASE_URL,
    DEFAULT_MODEL_ID,
    DEFAULT_REASONING_EFFORT,
    THINKING_DISABLED,
    THINKING_ENABLED,
    default_thinking_mode,
)
from .provider_reasoning import reasoning_for_model

ProviderCategory = Literal["official", "aggregator", "cloud", "custom"]
ProviderProtocol = Literal[
    "openai-chat-completions",
    "openai-responses",
    "anthropic-messages",
    "google-generative-ai",
    "google-vertex",
    "mistral-conversations",
    "bedrock-converse-stream",
    "azure-openai-responses",
]
AuthMode = Literal["api-key", "ambient", "oauth"]
ModelVerification = Literal["model-list", "chat-response", "none"]

CATALOG_VERSION = 1
DEFAULT_PROVIDER_ID = "deepseek"
CUSTOM_PROVIDER_ID = "custom"
# Curated capability identities are facts about this frozen catalog. Runtime
# model defaults may select any compatible model without rewriting these facts.
_CURATED_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_CURATED_DEEPSEEK_MODEL_ID = "deepseek-v4-pro"


class ProviderCatalogError(ValueError):
    """A fail-closed provider-selection error safe to surface in Settings."""


@dataclass(frozen=True, slots=True)
class ThinkingCapability:
    supported: bool = False
    default_mode: str = THINKING_DISABLED
    efforts: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "default_mode": self.default_mode,
            "efforts": list(self.efforts),
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    name: str
    recommended: bool = False
    native_reasoning: bool = False
    thinking: ThinkingCapability = ThinkingCapability()
    context_window: int | None = None
    max_output_tokens: int | None = None

    def to_public_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "recommended": self.recommended,
            "native_reasoning": self.native_reasoning,
            "thinking": self.thinking.to_public_dict(),
        }
        if self.context_window is not None:
            value["context_window"] = self.context_window
        if self.max_output_tokens is not None:
            value["max_output_tokens"] = self.max_output_tokens
        return value


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    id: str
    name: str
    category: ProviderCategory
    protocol: ProviderProtocol
    base_url: str | None
    auth_label: str
    auth_mode: AuthMode = "api-key"
    configurable: bool = False
    custom_base_url: bool = False
    model_verification: ModelVerification = "none"
    models: tuple[ModelSpec, ...] = ()
    disabled_reason: str | None = None
    base_url_aliases: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        thinking_supported = any(model.thinking.supported for model in self.models)
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "auth_label": self.auth_label,
            "auth_mode": self.auth_mode,
            "configurable": self.configurable,
            "custom_base_url": self.custom_base_url,
            "allow_custom_model_id": self.configurable,
            "model_verification": self.model_verification,
            "capabilities": {
                "chat_completions": (
                    self.configurable and self.protocol == "openai-chat-completions"
                ),
                "thinking": thinking_supported,
                "model_listing": self.model_verification == "model-list",
                "exact_model_verification": self.model_verification != "none",
                "runtime_memory_credentials": (
                    self.configurable and self.auth_mode == "api-key"
                ),
            },
            "models": [{**model.to_public_dict(), "reasoning": reasoning_for_model(self.id, model.id)}
                       for model in self.models],
            "disabled_reason": self.disabled_reason,
        }


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    id: ProviderProtocol
    name: str
    supported: bool

    def to_public_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "supported": self.supported}


@dataclass(frozen=True, slots=True)
class ValidatedProviderConfiguration:
    provider_id: str
    base_url: str
    model_id: str
    model_verification: ModelVerification
    thinking_supported: bool


@dataclass(frozen=True, slots=True)
class ValidatedProviderEndpoint:
    provider_id: str
    base_url: str


_NO_NATIVE_PROTOCOL = (
    "GWASMind currently executes OpenAI Chat Completions only; this provider requires "
    "a native protocol adapter."
)
_NO_CLOUD_CONFIGURATION = (
    "This provider requires cloud account, deployment, region, or ambient credential "
    "configuration that GWASMind does not yet collect."
)

_DEEPSEEK_THINKING = ThinkingCapability(
    supported=True,
    default_mode=THINKING_ENABLED,
    efforts=("low", "high", "max"),
)
_NO_THINKING = ThinkingCapability()


# Frozen from the provider-specific model catalogs bundled with
# @earendil-works/pi-ai 0.80.6 in CAUSE_Stable. Limits are keyed by both
# provider and model because aggregators may expose a model with limits that
# differ from the first-party endpoint. Keeping this table complete makes an
# omitted or stale curated entry fail closed during catalog construction.
_KIMI_LIMITS = {
    "kimi-k2.6": (262_144, 262_144),
    "kimi-k2.5": (262_144, 262_144),
}
_XIAOMI_LIMITS = {
    "mimo-v2.5-pro": (1_048_576, 131_072),
    "mimo-v2.5": (1_048_576, 131_072),
    "mimo-v2-pro": (1_048_576, 131_072),
}
_PI_MODEL_LIMITS: dict[str, dict[str, tuple[int, int]]] = {
    "deepseek": {
        "deepseek-v4-pro": (DEEPSEEK_CONTEXT_TOKENS, DEEPSEEK_MAX_OUTPUT_TOKENS),
        "deepseek-v4-flash": (DEEPSEEK_CONTEXT_TOKENS, DEEPSEEK_MAX_OUTPUT_TOKENS),
    },
    "openai": {
        # https://developers.openai.com/api/docs/models/gpt-5.4 (2026-09-13)
        "gpt-5.4": (1_050_000, 128_000),
        "gpt-5.3-chat-latest": (128_000, 16_384),
        "gpt-4.1": (1_047_576, 32_768),
        "gpt-4.1-mini": (1_047_576, 32_768),
    },
    "xai": {
        "grok-4.3": (1_000_000, 30_000),
        "grok-3": (131_072, 8_192),
    },
    "moonshotai": _KIMI_LIMITS,
    "moonshotai-cn": _KIMI_LIMITS,
    "zai": {
        "glm-5.2": (1_000_000, 131_072),
        "glm-5.1": (200_000, 131_072),
        "glm-5-turbo": (200_000, 131_072),
    },
    "zai-coding-cn": {
        "glm-5.2": (1_000_000, 131_072),
        "glm-5.1": (200_000, 131_072),
        "glm-4.7": (204_800, 131_072),
    },
    "xiaomi": _XIAOMI_LIMITS,
    "xiaomi-token-plan-ams": _XIAOMI_LIMITS,
    "xiaomi-token-plan-cn": _XIAOMI_LIMITS,
    "xiaomi-token-plan-sgp": _XIAOMI_LIMITS,
    "groq": {
        "openai/gpt-oss-120b": (131_072, 65_536),
        "qwen/qwen3-32b": (131_072, 40_960),
        "llama-3.3-70b-versatile": (131_072, 32_768),
    },
    "cerebras": {
        "gpt-oss-120b": (131_072, 40_960),
        "zai-glm-4.7": (131_072, 40_960),
        "gemma-4-31b": (131_072, 40_960),
    },
    "ant-ling": {
        "Ling-2.6-1T": (262_144, 65_536),
        "Ling-2.6-flash": (262_144, 65_536),
        "Ring-2.6-1T": (262_144, 65_536),
    },
    "openrouter": {
        "deepseek/deepseek-v4-pro": (1_048_576, 384_000),
        "anthropic/claude-opus-4.8": (1_000_000, 128_000),
        "openai/gpt-5.6-sol": (1_050_000, 128_000),
        "moonshotai/kimi-k2.6": (262_144, 262_144),
    },
    "together": {
        "deepseek-ai/DeepSeek-V4-Pro": (512_000, 384_000),
        "moonshotai/Kimi-K2.6": (262_144, 131_000),
        "Qwen/Qwen3.7-Max": (1_000_000, 500_000),
    },
    "fireworks": {
        "accounts/fireworks/models/glm-5p2": (1_048_575, 131_072),
        "accounts/fireworks/routers/glm-5p2-fast": (1_048_575, 131_072),
    },
    "huggingface": {
        "moonshotai/Kimi-K2.6": (262_144, 262_144),
        "zai-org/GLM-5.2": (262_144, 131_072),
        "openai/gpt-oss-120b": (131_072, 32_768),
    },
    "nvidia": {
        "nvidia/nemotron-3-ultra-550b-a55b": (1_000_000, 65_536),
        "moonshotai/kimi-k2.6": (262_144, 262_144),
        "openai/gpt-oss-120b": (128_000, 8_192),
    },
    "opencode": {
        "deepseek-v4-pro": (1_000_000, 384_000),
        "glm-5.2": (1_000_000, 131_072),
    },
    "opencode-go": {
        "deepseek-v4-pro": (1_000_000, 384_000),
        "glm-5.2": (1_000_000, 131_072),
        "kimi-k2.7-code": (262_144, 262_144),
    },
    "anthropic": {
        "claude-opus-4-8": (1_000_000, 128_000),
        "claude-sonnet-5": (1_000_000, 128_000),
    },
    "google": {
        "gemini-3.5-flash": (1_048_576, 65_536),
        "gemini-3.1-pro-preview": (1_048_576, 65_536),
    },
    "mistral": {
        "mistral-medium-2604": (262_144, 262_144),
        "mistral-small-2603": (256_000, 256_000),
    },
    "minimax": {
        "MiniMax-M3": (1_000_000, 128_000),
        "MiniMax-M2.7": (204_800, 131_072),
    },
    "minimax-cn": {"MiniMax-M3": (1_000_000, 128_000)},
    "kimi-coding": {"k2p7": (262_144, 32_768)},
    "openai-codex": {"gpt-5.6-sol": (372_000, 128_000)},
    "github-copilot": {"gpt-5.5": (1_000_000, 128_000)},
    "vercel-ai-gateway": {
        "zai/glm-5.2": (1_040_000, 128_000),
        "moonshotai/kimi-k2.6": (262_000, 262_000),
        "openai/gpt-5.6-sol": (1_050_000, 128_000),
    },
    "amazon-bedrock": {"zai.glm-5": (202_752, 101_376)},
    "azure-openai-responses": {"gpt-4.1": (1_047_576, 32_768)},
    "google-vertex": {"gemini-3.5-flash": (1_048_576, 65_536)},
    "cloudflare-workers-ai": {"@cf/zai-org/glm-5.2": (262_144, 262_144)},
}


def _models_with_pi_limits(
    provider_id: str,
    models: tuple[ModelSpec, ...],
) -> tuple[ModelSpec, ...]:
    provider_limits = _PI_MODEL_LIMITS.get(provider_id, {})
    enriched: list[ModelSpec] = []
    for model in models:
        limits = provider_limits.get(model.id)
        if limits is None:
            raise RuntimeError(
                f"Missing Pi 0.80.6 model limits for {provider_id}:{model.id}"
            )
        enriched.append(
            replace(
                model,
                context_window=limits[0],
                max_output_tokens=limits[1],
            )
        )
    return tuple(enriched)


def _model(
    model_id: str,
    name: str,
    *,
    recommended: bool = False,
    native_reasoning: bool = False,
    thinking: ThinkingCapability = _NO_THINKING,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        name=name,
        recommended=recommended,
        native_reasoning=native_reasoning,
        thinking=thinking,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )


def _chat_provider(
    provider_id: str,
    name: str,
    category: ProviderCategory,
    base_url: str,
    auth_label: str,
    models: tuple[ModelSpec, ...],
    *,
    verification: ModelVerification = "chat-response",
    aliases: tuple[str, ...] = (),
) -> ProviderSpec:
    return ProviderSpec(
        id=provider_id,
        name=name,
        category=category,
        protocol="openai-chat-completions",
        base_url=base_url,
        auth_label=auth_label,
        configurable=True,
        model_verification=verification,
        models=_models_with_pi_limits(provider_id, models),
        base_url_aliases=aliases,
    )


def _native_provider(
    provider_id: str,
    name: str,
    category: ProviderCategory,
    protocol: ProviderProtocol,
    base_url: str | None,
    auth_label: str,
    models: tuple[ModelSpec, ...],
    *,
    auth_mode: AuthMode = "api-key",
    reason: str = _NO_NATIVE_PROTOCOL,
) -> ProviderSpec:
    return ProviderSpec(
        id=provider_id,
        name=name,
        category=category,
        protocol=protocol,
        base_url=base_url,
        auth_label=auth_label,
        auth_mode=auth_mode,
        models=_models_with_pi_limits(provider_id, models),
        disabled_reason=reason,
    )


# This is a deliberately frozen UI catalog, not a claim that these are every
# model offered by each provider. Recommended IDs are based on the mature CAUSE
# catalog bundled beside GWASMind. Settings can fetch a current bounded /models
# snapshot, while the free-form model field and exact connection test remain as
# fallbacks for providers whose listing endpoint is incomplete or unavailable.
PROVIDERS: tuple[ProviderSpec, ...] = (
    _chat_provider(
        "deepseek",
        "DeepSeek",
        "official",
        _CURATED_DEEPSEEK_BASE_URL,
        "DeepSeek API Key",
        (
            _model(
                _CURATED_DEEPSEEK_MODEL_ID,
                "DeepSeek V4 Pro",
                recommended=True,
                native_reasoning=True,
                thinking=_DEEPSEEK_THINKING,
                context_window=1_000_000,
                max_output_tokens=DEEPSEEK_MAX_OUTPUT_TOKENS,
            ),
            _model(
                "deepseek-v4-flash",
                "DeepSeek V4 Flash",
                native_reasoning=True,
                thinking=_DEEPSEEK_THINKING,
                context_window=1_000_000,
                max_output_tokens=DEEPSEEK_MAX_OUTPUT_TOKENS,
            ),
        ),
        verification="model-list",
        aliases=("https://api.deepseek.com/v1",),
    ),
    _chat_provider(
        "openai",
        "OpenAI",
        "official",
        "https://api.openai.com/v1",
        "OpenAI API Key",
        (
            _model("gpt-5.4", "GPT-5.4", recommended=True, native_reasoning=True),
            _model(
                "gpt-5.3-chat-latest",
                "GPT-5.3 Chat",
                native_reasoning=True,
            ),
            _model("gpt-4.1", "GPT-4.1"),
            _model("gpt-4.1-mini", "GPT-4.1 mini"),
        ),
        verification="model-list",
    ),
    _chat_provider(
        "xai",
        "xAI",
        "official",
        "https://api.x.ai/v1",
        "xAI API Key",
        (
            _model("grok-4.3", "Grok 4.3", recommended=True),
            _model("grok-3", "Grok 3"),
        ),
    ),
    _chat_provider(
        "moonshotai",
        "Moonshot AI",
        "official",
        "https://api.moonshot.ai/v1",
        "Moonshot API Key",
        (
            _model("kimi-k2.6", "Kimi K2.6", recommended=True, native_reasoning=True),
            _model("kimi-k2.5", "Kimi K2.5", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "moonshotai-cn",
        "Moonshot AI (China)",
        "official",
        "https://api.moonshot.cn/v1",
        "Moonshot API Key",
        (
            _model("kimi-k2.6", "Kimi K2.6", recommended=True, native_reasoning=True),
            _model("kimi-k2.5", "Kimi K2.5", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "zai",
        "Z.ai",
        "official",
        "https://api.z.ai/api/coding/paas/v4",
        "Z.ai API Key",
        (
            _model("glm-5.2", "GLM-5.2", recommended=True, native_reasoning=True),
            _model("glm-5.1", "GLM-5.1", native_reasoning=True),
            _model("glm-5-turbo", "GLM-5 Turbo", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "zai-coding-cn",
        "Z.ai Coding (China)",
        "official",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        "Z.ai Coding API Key",
        (
            _model("glm-5.2", "GLM-5.2", recommended=True, native_reasoning=True),
            _model("glm-5.1", "GLM-5.1", native_reasoning=True),
            _model("glm-4.7", "GLM-4.7", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "xiaomi",
        "Xiaomi MiMo",
        "official",
        "https://api.xiaomimimo.com/v1",
        "Xiaomi MiMo API Key",
        (
            _model("mimo-v2.5-pro", "MiMo V2.5 Pro", recommended=True, native_reasoning=True),
            _model("mimo-v2.5", "MiMo V2.5", native_reasoning=True),
            _model("mimo-v2-pro", "MiMo V2 Pro", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "xiaomi-token-plan-ams",
        "Xiaomi Token Plan (Europe)",
        "official",
        "https://token-plan-ams.xiaomimimo.com/v1",
        "Xiaomi Token Plan API Key",
        (
            _model("mimo-v2.5-pro", "MiMo V2.5 Pro", recommended=True, native_reasoning=True),
            _model("mimo-v2.5", "MiMo V2.5", native_reasoning=True),
            _model("mimo-v2-pro", "MiMo V2 Pro", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "xiaomi-token-plan-cn",
        "Xiaomi Token Plan (China)",
        "official",
        "https://token-plan-cn.xiaomimimo.com/v1",
        "Xiaomi Token Plan API Key",
        (
            _model("mimo-v2.5-pro", "MiMo V2.5 Pro", recommended=True, native_reasoning=True),
            _model("mimo-v2.5", "MiMo V2.5", native_reasoning=True),
            _model("mimo-v2-pro", "MiMo V2 Pro", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "xiaomi-token-plan-sgp",
        "Xiaomi Token Plan (Singapore)",
        "official",
        "https://token-plan-sgp.xiaomimimo.com/v1",
        "Xiaomi Token Plan API Key",
        (
            _model("mimo-v2.5-pro", "MiMo V2.5 Pro", recommended=True, native_reasoning=True),
            _model("mimo-v2.5", "MiMo V2.5", native_reasoning=True),
            _model("mimo-v2-pro", "MiMo V2 Pro", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "groq",
        "Groq",
        "official",
        "https://api.groq.com/openai/v1",
        "Groq API Key",
        (
            _model("openai/gpt-oss-120b", "GPT OSS 120B", recommended=True),
            _model("qwen/qwen3-32b", "Qwen3 32B", native_reasoning=True),
            _model("llama-3.3-70b-versatile", "Llama 3.3 70B Versatile"),
        ),
    ),
    _chat_provider(
        "cerebras",
        "Cerebras",
        "official",
        "https://api.cerebras.ai/v1",
        "Cerebras API Key",
        (
            _model("gpt-oss-120b", "GPT OSS 120B", recommended=True, native_reasoning=True),
            _model("zai-glm-4.7", "GLM-4.7", native_reasoning=True),
            _model("gemma-4-31b", "Gemma 4 31B IT", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "ant-ling",
        "Ant Ling",
        "official",
        "https://api.ant-ling.com/v1",
        "Ant Ling API Key",
        (
            _model("Ling-2.6-1T", "Ling 2.6 1T", recommended=True),
            _model("Ling-2.6-flash", "Ling 2.6 Flash"),
            _model("Ring-2.6-1T", "Ring 2.6 1T", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "openrouter",
        "OpenRouter",
        "aggregator",
        "https://openrouter.ai/api/v1",
        "OpenRouter API Key",
        (
            _model(
                "deepseek/deepseek-v4-pro",
                "DeepSeek V4 Pro",
                recommended=True,
                native_reasoning=True,
            ),
            _model("anthropic/claude-opus-4.8", "Claude Opus 4.8", native_reasoning=True),
            _model("openai/gpt-5.6-sol", "GPT-5.6 Sol", native_reasoning=True),
            _model("moonshotai/kimi-k2.6", "Kimi K2.6", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "together",
        "Together AI",
        "aggregator",
        "https://api.together.ai/v1",
        "Together API Key",
        (
            _model(
                "deepseek-ai/DeepSeek-V4-Pro",
                "DeepSeek V4 Pro",
                recommended=True,
                native_reasoning=True,
            ),
            _model("moonshotai/Kimi-K2.6", "Kimi K2.6", native_reasoning=True),
            _model("Qwen/Qwen3.7-Max", "Qwen3.7 Max", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "fireworks",
        "Fireworks AI",
        "aggregator",
        "https://api.fireworks.ai/inference/v1",
        "Fireworks API Key",
        (
            _model(
                "accounts/fireworks/models/glm-5p2",
                "GLM 5.2",
                recommended=True,
                native_reasoning=True,
            ),
            _model(
                "accounts/fireworks/routers/glm-5p2-fast",
                "GLM 5.2 Fast",
                native_reasoning=True,
            ),
        ),
    ),
    _chat_provider(
        "huggingface",
        "Hugging Face Inference",
        "aggregator",
        "https://router.huggingface.co/v1",
        "Hugging Face Token",
        (
            _model("moonshotai/Kimi-K2.6", "Kimi K2.6", recommended=True),
            _model("zai-org/GLM-5.2", "GLM-5.2", native_reasoning=True),
            _model("openai/gpt-oss-120b", "GPT OSS 120B", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "nvidia",
        "NVIDIA NIM",
        "aggregator",
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA API Key",
        (
            _model(
                "nvidia/nemotron-3-ultra-550b-a55b",
                "Nemotron 3 Ultra",
                recommended=True,
                native_reasoning=True,
            ),
            _model("moonshotai/kimi-k2.6", "Kimi K2.6", native_reasoning=True),
            _model("openai/gpt-oss-120b", "GPT OSS 120B", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "opencode",
        "OpenCode Zen",
        "aggregator",
        "https://opencode.ai/zen/v1",
        "OpenCode API Key",
        (
            _model("deepseek-v4-pro", "DeepSeek V4 Pro", recommended=True),
            _model("glm-5.2", "GLM-5.2", native_reasoning=True),
        ),
        # OpenCode's newer inference address (as listed by models.dev, checked
        # 2026-10-23) serves chat but no /models list, so /zen/v1 stays primary.
        aliases=("https://opencode.ai/inference/openai/v1",),
    ),
    _chat_provider(
        "opencode-go",
        "OpenCode Zen Go",
        "aggregator",
        "https://opencode.ai/zen/go/v1",
        "OpenCode API Key",
        (
            _model("deepseek-v4-pro", "DeepSeek V4 Pro", recommended=True),
            _model("glm-5.2", "GLM-5.2", native_reasoning=True),
            _model("kimi-k2.7-code", "Kimi K2.7 Code", native_reasoning=True),
        ),
        aliases=("https://opencode.ai/inference/go/openai/v1",),
    ),
    ProviderSpec(
        id=CUSTOM_PROVIDER_ID,
        name="Custom OpenAI-compatible",
        category="custom",
        protocol="openai-chat-completions",
        base_url=None,
        auth_label="API Key",
        configurable=True,
        custom_base_url=True,
        model_verification="chat-response",
    ),
    _native_provider(
        "anthropic",
        "Anthropic",
        "official",
        "anthropic-messages",
        "https://api.anthropic.com",
        "Anthropic API Key",
        (
            _model("claude-opus-4-8", "Claude Opus 4.8", recommended=True, native_reasoning=True),
            _model("claude-sonnet-5", "Claude Sonnet 5", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "google",
        "Google Gemini",
        "official",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "Gemini API Key",
        (
            _model("gemini-3.5-flash", "Gemini 3.5 Flash", recommended=True, native_reasoning=True),
            _model("gemini-3.1-pro-preview", "Gemini 3.1 Pro", native_reasoning=True),
        ),
    ),
    _chat_provider(
        "mistral",
        "Mistral AI",
        "official",
        "https://api.mistral.ai/v1",
        "Mistral API Key",
        (
            _model("mistral-medium-2604", "Mistral Medium 2604", recommended=True),
            _model("mistral-small-2603", "Mistral Small 2603"),
        ),
    ),
    _native_provider(
        "minimax",
        "MiniMax",
        "official",
        "anthropic-messages",
        "https://api.minimax.io/anthropic",
        "MiniMax API Key",
        (
            _model("MiniMax-M3", "MiniMax M3", recommended=True, native_reasoning=True),
            _model("MiniMax-M2.7", "MiniMax M2.7", native_reasoning=True),
        ),
    ),
    _native_provider(
        "minimax-cn",
        "MiniMax (China)",
        "official",
        "anthropic-messages",
        "https://api.minimaxi.com/anthropic",
        "MiniMax API Key",
        (_model("MiniMax-M3", "MiniMax M3", recommended=True, native_reasoning=True),),
    ),
    _native_provider(
        "kimi-coding",
        "Kimi Coding",
        "official",
        "anthropic-messages",
        "https://api.kimi.com/coding",
        "Kimi API Key",
        (_model("k2p7", "Kimi K2.7 Code", recommended=True, native_reasoning=True),),
    ),
    _native_provider(
        "openai-codex",
        "OpenAI Codex",
        "official",
        "openai-responses",
        "https://chatgpt.com/backend-api",
        "ChatGPT OAuth",
        (_model("gpt-5.6-sol", "GPT-5.6 Sol", recommended=True, native_reasoning=True),),
        auth_mode="oauth",
        reason="Requires ChatGPT OAuth and the Codex Responses protocol.",
    ),
    _native_provider(
        "github-copilot",
        "GitHub Copilot",
        "official",
        "openai-chat-completions",
        "https://api.individual.githubcopilot.com",
        "GitHub Copilot OAuth",
        (_model("gpt-5.5", "GPT-5.5", recommended=True, native_reasoning=True),),
        auth_mode="oauth",
        reason="Requires GitHub OAuth and Copilot-specific request headers.",
    ),
    _chat_provider(
        "vercel-ai-gateway",
        "Vercel AI Gateway",
        "aggregator",
        "https://ai-gateway.vercel.sh/v1",
        "Vercel AI Gateway Key",
        (
            _model("zai/glm-5.2", "GLM-5.2", recommended=True, native_reasoning=True),
            _model("moonshotai/kimi-k2.6", "Kimi K2.6", native_reasoning=True),
            _model("openai/gpt-5.6-sol", "GPT-5.6 Sol", native_reasoning=True),
        ),
    ),
    _native_provider(
        "amazon-bedrock",
        "Amazon Bedrock",
        "cloud",
        "bedrock-converse-stream",
        "https://bedrock-runtime.us-east-1.amazonaws.com",
        "AWS credentials",
        (_model("zai.glm-5", "GLM-5", recommended=True, native_reasoning=True),),
        auth_mode="ambient",
        reason=_NO_CLOUD_CONFIGURATION,
    ),
    _native_provider(
        "azure-openai-responses",
        "Azure OpenAI",
        "cloud",
        "azure-openai-responses",
        None,
        "Azure OpenAI API Key",
        (_model("gpt-4.1", "GPT-4.1", recommended=True),),
        reason=_NO_CLOUD_CONFIGURATION,
    ),
    _native_provider(
        "google-vertex",
        "Google Vertex AI",
        "cloud",
        "google-vertex",
        None,
        "Google Cloud credentials",
        (_model("gemini-3.5-flash", "Gemini 3.5 Flash", recommended=True, native_reasoning=True),),
        auth_mode="ambient",
        reason=_NO_CLOUD_CONFIGURATION,
    ),
    _native_provider(
        "cloudflare-ai-gateway",
        "Cloudflare AI Gateway",
        "cloud",
        "anthropic-messages",
        None,
        "Cloudflare API Token",
        (),
        reason=_NO_CLOUD_CONFIGURATION,
    ),
    _native_provider(
        "cloudflare-workers-ai",
        "Cloudflare Workers AI",
        "cloud",
        "openai-chat-completions",
        None,
        "Cloudflare API Token",
        (_model("@cf/zai-org/glm-5.2", "GLM-5.2", recommended=True, native_reasoning=True),),
        reason=_NO_CLOUD_CONFIGURATION,
    ),
)


PROTOCOLS: tuple[ProtocolSpec, ...] = (
    ProtocolSpec("openai-chat-completions", "OpenAI Chat Completions", True),
    ProtocolSpec("openai-responses", "OpenAI Responses", False),
    ProtocolSpec("anthropic-messages", "Anthropic Messages", False),
    ProtocolSpec("google-generative-ai", "Google Generative AI", False),
    ProtocolSpec("google-vertex", "Google Vertex AI", False),
    ProtocolSpec("mistral-conversations", "Mistral Conversations", False),
    ProtocolSpec("bedrock-converse-stream", "Amazon Bedrock Converse", False),
    ProtocolSpec("azure-openai-responses", "Azure OpenAI Responses", False),
)

_PROVIDERS_BY_ID = {provider.id: provider for provider in PROVIDERS}


def get_provider(provider_id: str) -> ProviderSpec | None:
    """Return one immutable catalog provider, or ``None`` for an unknown ID."""

    if not isinstance(provider_id, str):
        return None
    return _PROVIDERS_BY_ID.get(provider_id.strip().casefold())


def get_model(provider_id: str, model_id: str) -> ModelSpec | None:
    """Return one exact curated model without guessing aliases."""

    provider = get_provider(provider_id)
    if provider is None or not isinstance(model_id, str):
        return None
    requested = model_id.strip()
    return next((model for model in provider.models if model.id == requested), None)


def model_for_endpoint(base_url: str, model_id: str) -> ModelSpec | None:
    """Resolve curated limits only for a canonical provider endpoint."""

    provider_id = infer_provider_id(base_url, model_id)
    model = get_model(provider_id, model_id)
    if model is None and provider_id == "deepseek" and model_id in {
        "deepseek-flash", "deepseek-v4-flash-vision-exp",
    }:
        # Explicit official Flash aliases; never infer from a model-name substring
        # or transfer a first-party limit onto an aggregator/custom endpoint.
        reference = get_model("deepseek", "deepseek-v4-flash")
        return replace(reference, id=model_id, name=model_id) if reference is not None else None
    return model


def require_provider(provider_id: str) -> ProviderSpec:
    provider = get_provider(provider_id)
    if provider is None:
        raise ProviderCatalogError("Unknown model provider.")
    return provider


def model_catalog_payload() -> dict[str, Any]:
    """Build a fresh, deterministic, credential-free Settings payload."""

    providers = [provider.to_public_dict() for provider in PROVIDERS]
    builtin_providers = [
        provider for provider in PROVIDERS if provider.id != CUSTOM_PROVIDER_ID
    ]
    return {
        "version": CATALOG_VERSION,
        "default": {
            "provider_id": infer_provider_id(DEFAULT_MODEL_BASE_URL),
            "model_id": DEFAULT_MODEL_ID,
            "base_url": DEFAULT_MODEL_BASE_URL,
            "thinking_mode": default_thinking_mode(
                model_id=DEFAULT_MODEL_ID, base_url=DEFAULT_MODEL_BASE_URL
            ),
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
        },
        "protocols": [protocol.to_public_dict() for protocol in PROTOCOLS],
        "privacy": {
            "credential_storage": "runtime-memory-only",
            "credentials_persisted": False,
            "credentials_returned": False,
        },
        "providers": providers,
        "total_providers": len(builtin_providers),
        "total_entries": len(providers),
        "configurable_providers": sum(
            provider.configurable for provider in builtin_providers
        ),
        "custom_available": CUSTOM_PROVIDER_ID in _PROVIDERS_BY_ID,
        "total_models": sum(len(provider.models) for provider in PROVIDERS),
    }


def infer_provider_id(base_url: str, model_id: str | None = None) -> str:
    """Infer a provider only from a canonical endpoint; otherwise return custom.

    ``model_id`` is accepted for caller ergonomics but deliberately cannot turn
    an unknown endpoint into an official provider. This prevents a proxy named
    like an official model from inheriting provider-specific request fields.
    """

    del model_id
    if not is_safe_provider_base_url(base_url):
        return CUSTOM_PROVIDER_ID
    for provider in PROVIDERS:
        if not provider.configurable or provider.id == CUSTOM_PROVIDER_ID:
            continue
        if _base_url_matches_provider(base_url, provider):
            return provider.id
    return CUSTOM_PROVIDER_ID


def validate_provider_configuration(
    *,
    provider_id: str,
    base_url: str,
    model_id: str,
) -> ValidatedProviderConfiguration:
    """Validate the trust boundary between a Settings choice and execution."""

    provider = require_provider(provider_id)
    if not provider.configurable:
        raise ProviderCatalogError(
            provider.disabled_reason or "The selected provider is not configurable."
        )
    if not is_safe_provider_base_url(base_url):
        raise ProviderCatalogError("API Base URL must be HTTPS or a loopback HTTP URL.")
    if not _safe_model_id(model_id):
        raise ProviderCatalogError("Model Name must contain between 1 and 120 valid characters.")
    if provider.id != CUSTOM_PROVIDER_ID and not _base_url_matches_provider(base_url, provider):
        raise ProviderCatalogError(
            "The API Base URL does not match the selected provider. "
            "Choose Custom OpenAI-compatible for a proxy or alternate endpoint."
        )
    thinking_supported = reasoning_for_model(provider.id, model_id.strip())["kind"] in {
        "toggle_effort", "effort", "toggle", "fixed",
    }
    return ValidatedProviderConfiguration(
        provider_id=provider.id,
        base_url=_normalized_base_url(base_url),
        model_id=model_id.strip(),
        model_verification=provider.model_verification,
        thinking_supported=thinking_supported,
    )


def validate_provider_endpoint(
    *,
    provider_id: str,
    base_url: str,
) -> ValidatedProviderEndpoint:
    """Validate an OpenAI-compatible provider endpoint without selecting a model."""

    provider = require_provider(provider_id)
    if not provider.configurable:
        raise ProviderCatalogError(
            provider.disabled_reason or "The selected provider is not configurable."
        )
    if provider.protocol != "openai-chat-completions" or provider.auth_mode != "api-key":
        raise ProviderCatalogError(
            "The selected provider does not support API-key model discovery."
        )
    if not is_safe_provider_base_url(base_url):
        raise ProviderCatalogError("API Base URL must be HTTPS or a loopback HTTP URL.")
    if provider.id != CUSTOM_PROVIDER_ID and not _base_url_matches_provider(base_url, provider):
        raise ProviderCatalogError(
            "The API Base URL does not match the selected provider. "
            "Choose Custom OpenAI-compatible for a proxy or alternate endpoint."
        )
    return ValidatedProviderEndpoint(
        provider_id=provider.id,
        base_url=_normalized_base_url(base_url),
    )


def is_safe_provider_base_url(value: Any) -> bool:
    """Accept public HTTPS endpoints and local HTTP development endpoints."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or (port is not None and port <= 0)
    ):
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    hostname = parsed.hostname.casefold().rstrip(".")
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _safe_model_id(value: Any) -> bool:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 120:
        return False
    return not any(
        ord(character) < 32 or ord(character) == 127 for character in value.strip()
    )


def _normalized_base_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    port = parsed.port
    default_port = (parsed.scheme == "https" and port == 443) or (
        parsed.scheme == "http" and port == 80
    )
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.casefold(), netloc, path, "", ""))


def _base_url_matches_provider(value: str, provider: ProviderSpec) -> bool:
    normalized = _normalized_base_url(value)
    accepted = (provider.base_url, *provider.base_url_aliases)
    for candidate in accepted:
        if candidate is None:
            continue
        expected = _normalized_base_url(candidate)
        if normalized == expected or normalized == f"{expected}/chat/completions":
            return True
    return False


def _validate_catalog_definition() -> None:
    if len(_PROVIDERS_BY_ID) != len(PROVIDERS):
        raise RuntimeError("Provider catalog IDs must be unique")
    if DEFAULT_PROVIDER_ID not in _PROVIDERS_BY_ID:
        raise RuntimeError("Default provider is missing from the provider catalog")
    default_provider = _PROVIDERS_BY_ID[DEFAULT_PROVIDER_ID]
    if default_provider.base_url != _CURATED_DEEPSEEK_BASE_URL:
        raise RuntimeError("Curated DeepSeek Base URL drifted from its verified endpoint")
    default_models = [
        model for model in default_provider.models if model.id == _CURATED_DEEPSEEK_MODEL_ID
    ]
    if len(default_models) != 1 or not default_models[0].recommended:
        raise RuntimeError("Default model must be a recommended model of the default provider")
    for provider in PROVIDERS:
        if provider.configurable and provider.protocol != "openai-chat-completions":
            raise RuntimeError("Only OpenAI Chat Completions providers may be configurable")
        if not provider.configurable and not provider.disabled_reason:
            raise RuntimeError("Disabled providers must explain why they are unavailable")
        model_ids = [model.id for model in provider.models]
        if len(set(model_ids)) != len(model_ids):
            raise RuntimeError(f"Duplicate model ID in provider catalog: {provider.id}")
        for model in provider.models:
            if model.context_window is None or model.context_window <= 0:
                raise RuntimeError(
                    f"Invalid context window in provider catalog: {provider.id}:{model.id}"
                )
            if model.max_output_tokens is None or model.max_output_tokens <= 0:
                raise RuntimeError(
                    f"Invalid output limit in provider catalog: {provider.id}:{model.id}"
                )


_validate_catalog_definition()


__all__ = [
    "CATALOG_VERSION",
    "CUSTOM_PROVIDER_ID",
    "DEFAULT_PROVIDER_ID",
    "PROTOCOLS",
    "PROVIDERS",
    "ModelSpec",
    "ProviderCatalogError",
    "ProviderSpec",
    "ThinkingCapability",
    "ValidatedProviderConfiguration",
    "ValidatedProviderEndpoint",
    "get_model",
    "get_provider",
    "infer_provider_id",
    "is_safe_provider_base_url",
    "model_catalog_payload",
    "model_for_endpoint",
    "require_provider",
    "validate_provider_configuration",
    "validate_provider_endpoint",
]
