"""Credential-free Settings model discovery, separate from inference contracts.

Only reviewed public metadata endpoints are contacted. Remote catalogs cannot
change provider URLs, request protocols, credentials, or curated runtime limits.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from .provider_catalog import get_model, get_provider, model_catalog_payload
from .provider_reasoning import reasoning_for_model

_SCHEMA_VERSION = 1
_MAX_BYTES = 12 * 1024 * 1024
_MAX_MODELS = 5000
_TIMEOUT_SECONDS = 8.0
_SOURCE_DEADLINE_SECONDS = 20.0
_RETRY_SECONDS = 0.25
_MODELS_DEV_URL = "https://models.dev/api.json"
_OFFICIAL_URLS = {
    "openrouter": "https://openrouter.ai/api/v1/models",
    "huggingface": "https://router.huggingface.co/v1/models",
    "nvidia": "https://integrate.api.nvidia.com/v1/models",
    "vercel-ai-gateway": "https://ai-gateway.vercel.sh/v1/models",
    "opencode": "https://opencode.ai/zen/v1/models",
    "opencode-go": "https://opencode.ai/zen/go/v1/models",
}
_ALLOWED_URLS = frozenset((_MODELS_DEV_URL, *_OFFICIAL_URLS.values()))
# This is an endpoint-specific reviewed map, not fuzzy provider-name matching.
# It is also the only source allowed when models.dev omits its API base URL.
_MODELS_DEV_IDS = {
    "deepseek": "deepseek",
    "openai": "openai",
    "xai": "xai",
    "moonshotai": "moonshotai",
    "moonshotai-cn": "moonshotai-cn",
    "zai": "zai-coding-plan",
    "zai-coding-cn": "zhipuai-coding-plan",
    "xiaomi": "xiaomi",
    "xiaomi-token-plan-ams": "xiaomi-token-plan-ams",
    "xiaomi-token-plan-cn": "xiaomi-token-plan-cn",
    "xiaomi-token-plan-sgp": "xiaomi-token-plan-sgp",
    "groq": "groq",
    "cerebras": "cerebras",
    "ant-ling": "ant-ling",
    "openrouter": "openrouter",
    "together": "togetherai",
    "fireworks": "fireworks-ai",
    "huggingface": "huggingface",
    "nvidia": "nvidia",
    "opencode": "opencode",
    "opencode-go": "opencode-go",
    "google": "google",
    "mistral": "mistral",
    "vercel-ai-gateway": "vercel",
}
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,249}\Z")
_NON_CHAT_ID = re.compile(
    r"(?:^|[/_.:-])(?:embedding(?:s|gemma)?|embed(?:ding|qa)?|rerank(?:er|qa)?|moderation|"
    r"bge|e5|gte|voyage|sentence-transformers?|colbert|"
    r"whisper|tts|transcri(?:be|ption)|realtime|audio|lyria|dall-e|imagegen|"
    r"image|video|stable-diffusion)(?:$|[/_.:-])",
    re.IGNORECASE,
)
_BUNDLE_PATH = Path(__file__).with_name("data") / "public_model_catalog.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return (
        parsed.astimezone(UTC).isoformat(timespec="seconds") if parsed.tzinfo is not None else None
    )


def _model_id(value: Any) -> str | None:
    if not isinstance(value, str) or not _MODEL_ID.fullmatch(value):
        return None
    return None if "://" in value else value


def _positive_limit(value: Any) -> int | None:
    return value if type(value) is int and 0 < value <= 100_000_000 else None


def _public_model(provider_id: str, raw: Mapping[str, Any]) -> dict[str, Any] | None:
    model_id = _model_id(raw.get("id"))
    if model_id is None:
        return None
    curated = get_model(provider_id, model_id)
    if curated is not None:
        return {
            **curated.to_public_dict(),
            "reasoning": reasoning_for_model(provider_id, model_id),
        }
    name = raw.get("name")
    if (
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 180
        or any(ord(char) < 32 for char in name)
    ):
        name = model_id
    # Public metadata can describe a display limit; it cannot create a wire
    # contract for thinking toggles or reasoning-effort request parameters.
    reasoning = reasoning_for_model(provider_id, model_id)
    result: dict[str, Any] = {
        "id": model_id,
        "name": name,
        "recommended": False,
        "native_reasoning": reasoning["kind"] in {"fixed", "toggle", "toggle_effort", "effort"},
        "thinking": {
            "supported": reasoning["kind"] in {"toggle", "toggle_effort", "effort"},
            "default_mode": reasoning["default_mode"],
            "efforts": reasoning["efforts"],
        },
        "reasoning": reasoning,
    }
    for key in ("context_window", "max_output_tokens"):
        value = _positive_limit(raw.get(key))
        if value is not None:
            result[key] = value
    return result


def _text_model(provider_id: str, row: Mapping[str, Any]) -> bool:
    model_id = _model_id(row.get("id"))
    if model_id is None or _NON_CHAT_ID.search(model_id):
        return False
    if row.get("deprecated") is True or row.get("status") in ("deprecated", "retired", "disabled"):
        return False
    for field in ("type", "task", "pipeline_tag"):
        value = row.get(field)
        if isinstance(value, str) and any(
            term in value.casefold()
            for term in ("embedding", "rerank", "image", "video", "audio", "speech", "moderation")
        ):
            return False
    for field in ("modalities", "architecture"):
        metadata = row.get(field)
        if isinstance(metadata, Mapping):
            outputs = metadata.get("output", metadata.get("output_modalities"))
            inputs = metadata.get("input", metadata.get("input_modalities"))
            if outputs is not None and outputs != ["text"]:
                return False
            if inputs is not None and (not isinstance(inputs, list) or "text" not in inputs):
                return False
            modality = metadata.get("modality")
            if (
                isinstance(modality, str)
                and "->" in modality
                and (
                    modality.rsplit("->", 1)[-1] != "text"
                    or "text" not in modality.split("->", 1)[0].split("+")
                )
            ):
                return False
    if provider_id == "google" and re.search(
        r"(?:^|[-_])(?:deep-research|computer-use|live)(?:$|[-_])", model_id, re.IGNORECASE
    ):
        return False
    # These OpenAI families require Responses, which this application does not
    # execute. Aggregators may expose their own compatible adapter separately.
    return not (
        provider_id == "openai"
        and re.search(
            r"(?:^|[-_])(?:codex|deep-research|computer-use|pro)(?:$|[-_])", model_id, re.IGNORECASE
        )
    )


def _declares_text(row: Mapping[str, Any]) -> bool:
    for field in ("modalities", "architecture"):
        metadata = row.get(field)
        if isinstance(metadata, Mapping):
            outputs = metadata.get("output", metadata.get("output_modalities"))
            if outputs == ["text"]:
                return True
            modality = metadata.get("modality")
            if (
                isinstance(modality, str)
                and "->" in modality
                and modality.rsplit("->", 1)[-1] == "text"
                and "text" in modality.split("->", 1)[0].split("+")
            ):
                return True
    return row.get("tool_call") is True or any(
        row.get(field)
        in ("language", "language-model", "text-generation", "conversational", "chat", "text")
        for field in ("type", "task", "pipeline_tag")
    )


def _normalise_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return value.rstrip("/").casefold()


def _endpoint_matches(provider_id: str, raw: Mapping[str, Any]) -> bool:
    provider = get_provider(provider_id)
    if provider is None or not provider.configurable or not provider.base_url:
        return False
    api = raw.get("api")
    if api is None or api == "":
        return provider_id in _MODELS_DEV_IDS
    if not isinstance(api, str):
        return False
    accepted = {provider.base_url, *provider.base_url_aliases}
    # Google's official OpenAI-compatible endpoint shares its model IDs with
    # the native public catalog; the local provider still fixes the protocol.
    if provider_id == "google":
        accepted.add("https://generativelanguage.googleapis.com/v1beta")
    return _normalise_url(api) in {_normalise_url(url) for url in accepted}


def _models_dev_rows(payload: Any) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise TypeError("Invalid public catalog.")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for provider_id, source_id in _MODELS_DEV_IDS.items():
        raw = payload.get(source_id)
        if not isinstance(raw, Mapping) or not _endpoint_matches(provider_id, raw):
            continue
        rows = raw.get("models")
        if not isinstance(rows, Mapping) or len(rows) > _MAX_MODELS:
            continue
        accepted: dict[str, dict[str, Any]] = {}
        for key, value in rows.items():
            if not isinstance(value, Mapping):
                continue
            row = {**value, "id": value.get("id", key)}
            override = row.get("provider")
            if (
                isinstance(override, Mapping)
                and override.get("api")
                and not _endpoint_matches(provider_id, override)
            ):
                continue
            if row.get("api") and not _endpoint_matches(provider_id, row):
                continue
            if not _text_model(provider_id, row) or not (
                _declares_text(row) or get_model(provider_id, row["id"]) is not None
            ):
                continue
            if provider_id in {"opencode", "opencode-go"}:
                npm = (
                    override.get("npm", raw.get("npm"))
                    if isinstance(override, Mapping)
                    else raw.get("npm")
                )
                if npm != "@ai-sdk/openai-compatible":
                    continue
            accepted[row["id"]] = dict(row)
        if accepted:
            result[provider_id] = accepted
    if not result:
        raise ValueError("No usable public provider metadata.")
    return result


def _source(provider_id: str | None, updated_at: str) -> dict[str, str]:
    provider = get_provider(provider_id) if provider_id else None
    return {
        "name": f"{provider.name} public model list" if provider else "models.dev",
        "url": _OFFICIAL_URLS[provider_id] if provider_id else _MODELS_DEV_URL,
        "updated_at": updated_at,
        "kind": "official-public" if provider_id else "public-metadata",
    }


def _metadata_model(provider_id: str, row: Mapping[str, Any]) -> dict[str, Any] | None:
    limits = row.get("limit")
    top_provider = row.get("top_provider")
    limits = limits if isinstance(limits, Mapping) else {}
    top_provider = top_provider if isinstance(top_provider, Mapping) else {}

    def first_limit(*values: Any) -> int | None:
        return next((value for value in values if _positive_limit(value) is not None), None)

    return _public_model(
        provider_id,
        {
            "id": row.get("id"),
            "name": row.get("name"),
            "context_window": first_limit(
                row.get("context_length"), row.get("context_window"), limits.get("context")
            ),
            "max_output_tokens": first_limit(
                row.get("max_output_tokens"),
                row.get("max_tokens"),
                top_provider.get("max_completion_tokens"),
                limits.get("output"),
            ),
        },
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        raise HTTPError(req.full_url, code, "Redirect not allowed.", headers, fp)


def _fetch_public_json(url: str) -> Any:
    if url not in _ALLOWED_URLS:
        raise ValueError("Unknown public model source.")
    # No environment proxy, authentication handler, cookie jar, API key or
    # model configuration participates in these anonymous metadata requests.
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    request = Request(
        url, headers={"Accept": "application/json", "User-Agent": "GWASMind-Public-Catalog/1"}
    )
    with opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > _MAX_BYTES:
            raise ValueError("Public catalog exceeds its size limit.")
        raw = response.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError("Public catalog exceeds its size limit.")
    return json.loads(raw)


class PublicModelCatalog:
    """An asynchronously refreshed, per-provider public metadata snapshot."""

    def __init__(
        self, cache_path: Path | None = None, *, fetch_json: Callable[[str], Any] | None = None
    ):
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._fetch_json = fetch_json or _fetch_public_json
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._providers: dict[str, dict[str, Any]] = {}
        self._status: dict[str, Any] = {"state": "idle", "updated_at": None}
        self._load(_BUNDLE_PATH)
        if self._cache_path is not None:
            self._load(self._cache_path)

    def _load(self, path: Path) -> None:
        try:
            with path.open("rb") as handle:
                content = handle.read(_MAX_BYTES + 1)
            if len(content) > _MAX_BYTES:
                return
            data = json.loads(content)
            if not isinstance(data, Mapping) or data.get("version") != _SCHEMA_VERSION:
                return
            providers = data.get("providers")
            if not isinstance(providers, Mapping):
                return
            for provider_id, raw in providers.items():
                if provider_id not in _MODELS_DEV_IDS or not isinstance(raw, Mapping):
                    continue
                source = raw.get("catalog_source")
                rows = raw.get("models")
                if (
                    not isinstance(source, Mapping)
                    or not isinstance(rows, list)
                    or len(rows) > _MAX_MODELS
                ):
                    continue
                when = _timestamp(source.get("updated_at"))
                expected = _source(
                    provider_id
                    if source.get("kind") == "official-public" and provider_id in _OFFICIAL_URLS
                    else None,
                    when or "",
                )
                if (
                    when is None
                    or source.get("url") != expected["url"]
                    or source.get("kind") != expected["kind"]
                ):
                    continue
                models = {
                    model["id"]: model
                    for row in rows
                    if isinstance(row, Mapping)
                    and _text_model(provider_id, row)
                    and (model := _public_model(provider_id, row)) is not None
                }
                previous = self._providers.get(provider_id)
                previous_when = previous["catalog_source"]["updated_at"] if previous else ""
                if models and when >= previous_when:
                    self._providers[provider_id] = {
                        "models": self._sort(models.values()),
                        "catalog_source": expected,
                    }
            if self._providers:
                self._status["updated_at"] = max(
                    value["catalog_source"]["updated_at"] for value in self._providers.values()
                )
        except (OSError, ValueError, TypeError, RecursionError):
            return

    @staticmethod
    def _sort(models: Any) -> list[dict[str, Any]]:
        return sorted(
            models, key=lambda row: (not row["recommended"], row["name"].casefold(), row["id"])
        )

    def snapshot(self) -> dict[str, Any]:
        payload = model_catalog_payload()
        with self._lock:
            providers = copy.deepcopy(self._providers)
            payload["public_refresh"] = copy.deepcopy(self._status)
        for provider in payload["providers"]:
            public = providers.get(provider["id"])
            if public is not None:
                provider.update(public)
                provider["capabilities"]["thinking"] = any(
                    row["thinking"]["supported"] for row in public["models"]
                )
            else:
                provider["catalog_source"] = {
                    "name": "Bundled fallback",
                    "url": None,
                    "updated_at": None,
                    "kind": "static",
                }
        payload["total_models"] = sum(len(row["models"]) for row in payload["providers"])
        return payload

    def export_cache(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": _SCHEMA_VERSION,
                "updated_at": self._status.get("updated_at"),
                "providers": copy.deepcopy(self._providers),
            }

    def start(self) -> None:
        with self._lock:
            if self._closed.is_set() or self._thread is not None:
                return
            self._status["state"] = "refreshing"
            self._thread = threading.Thread(
                target=self.refresh, name="public-model-catalog", daemon=True
            )
            self._thread.start()

    def close(self) -> None:
        self._closed.set()

    def _fetch(self, url: str) -> Any:
        for attempt in range(2):
            if self._closed.is_set():
                return None
            try:
                return self._fetch_json(url)
            except Exception:  # noqa: BLE001 - Keep startup usable; never expose provider responses.
                if attempt == 0:
                    self._closed.wait(_RETRY_SECONDS)
        return None

    def _publish(
        self, provider_id: str, rows: list[dict[str, Any]], source: dict[str, str]
    ) -> None:
        with self._lock:
            if not self._closed.is_set():
                self._providers[provider_id] = {
                    "models": self._sort(rows),
                    "catalog_source": source,
                }

    def refresh(self) -> None:
        if self._closed.is_set() or not self._refresh_lock.acquire(blocking=False):
            return
        try:
            self._refresh()
        except Exception:  # noqa: BLE001 - This optional background task must fail closed.
            # A malformed public source never changes inference availability,
            # exposes an exception body, or leaves an endless loading state.
            with self._lock:
                self._status.update({"state": "offline", "unavailable_sources": ["public catalog"]})
        finally:
            self._refresh_lock.release()

    def _source_results(self, sources: Mapping[str, str]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        completed = threading.Condition()
        deadline = time.monotonic() + _SOURCE_DEADLINE_SECONDS

        def fetch(provider_id: str, url: str) -> None:
            value = self._fetch(url)
            with completed:
                results[provider_id] = value
                completed.notify_all()

        for provider_id, url in sources.items():
            threading.Thread(
                target=fetch,
                args=(provider_id, url),
                daemon=True,
                name=f"public-model-{provider_id}",
            ).start()
        with completed:
            while len(results) < len(sources) and not self._closed.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                completed.wait(min(0.1, remaining))
            return dict(results)

    def _official_results(self) -> dict[str, Any]:
        return self._source_results(_OFFICIAL_URLS)

    def _refresh(self) -> None:
        with self._lock:
            self._status = {**self._status, "state": "refreshing", "attempted_at": _now()}
        updated_at = _now()
        failures: list[str] = []
        updated: set[str] = set()
        metadata: dict[str, dict[str, dict[str, Any]]] = {}
        try:
            public_sources = self._source_results({"models.dev": _MODELS_DEV_URL})
            metadata = _models_dev_rows(public_sources.get("models.dev"))
        except (ValueError, TypeError, RecursionError):
            failures.append("models.dev")
        for provider_id, rows in metadata.items():
            if provider_id in _OFFICIAL_URLS:
                continue
            models = [
                model
                for row in rows.values()
                if (model := _metadata_model(provider_id, row)) is not None
            ]
            if models:
                self._publish(provider_id, models, _source(None, updated_at))
                updated.add(provider_id)
        official = self._official_results() if not self._closed.is_set() else {}
        for provider_id in _OFFICIAL_URLS:
            if self._closed.is_set():
                break
            payload = official.get(provider_id)
            rows = payload.get("data") if isinstance(payload, Mapping) else None
            if not isinstance(rows, list) or not rows or len(rows) > _MAX_MODELS:
                failures.append(provider_id)
                self._metadata_fallback(provider_id, metadata, updated_at, updated)
                continue
            models: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, Mapping) or not _text_model(provider_id, row):
                    continue
                model_id = row["id"]
                known = metadata.get(provider_id, {}).get(model_id)
                if provider_id in {"opencode", "opencode-go"} and known is None:
                    # Their model list includes native Anthropic/Responses
                    # endpoints; availability alone is not protocol support.
                    continue
                if (
                    known is None
                    and get_model(provider_id, model_id) is None
                    and not _declares_text(row)
                ):
                    continue
                merged = {**(known or {}), **row}
                model = _metadata_model(provider_id, merged)
                if model is not None:
                    models[model_id] = model
            if models:
                self._publish(provider_id, list(models.values()), _source(provider_id, updated_at))
                updated.add(provider_id)
            else:
                failures.append(provider_id)
                self._metadata_fallback(provider_id, metadata, updated_at, updated)
        with self._lock:
            if self._closed.is_set():
                self._status["state"] = "idle"
                return
            self._status.update(
                {
                    "state": "partial"
                    if updated and failures
                    else "ready"
                    if updated
                    else "offline",
                    "updated_at": updated_at if updated else self._status.get("updated_at"),
                    "updated_providers": len(updated),
                    "unavailable_sources": failures,
                }
            )
        if updated and self._cache_path is not None:
            self._save()

    def _metadata_fallback(
        self,
        provider_id: str,
        metadata: dict[str, dict[str, dict[str, Any]]],
        updated_at: str,
        updated: set[str],
    ) -> None:
        with self._lock:
            if provider_id in self._providers:
                return
        models = [
            model
            for row in metadata.get(provider_id, {}).values()
            if (model := _metadata_model(provider_id, row)) is not None
        ]
        if models:
            self._publish(provider_id, models, _source(None, updated_at))
            updated.add(provider_id)

    def _save(self) -> None:
        temporary: Path | None = None
        try:
            assert self._cache_path is not None
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                self.export_cache(), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            if len(payload) > _MAX_BYTES:
                raise ValueError("Public cache exceeds its size limit.")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".model-catalog-",
                suffix=".tmp",
                dir=self._cache_path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if self._closed.is_set():
                return
            os.replace(temporary, self._cache_path)
            with self._lock:
                self._status.pop("cache_status", None)
        except (OSError, ValueError):
            with self._lock:
                self._status["cache_status"] = "unavailable"
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
