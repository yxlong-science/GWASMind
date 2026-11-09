"""The LLM prompt catalog; this module is the only reader of llm_prompts.toml."""
from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from string import Template
from types import MappingProxyType
from typing import Any

from .config_files import PROMPTS
from .config_files import freeze_value as _freeze
from .config_files import plain_value as _plain

PROMPT_CATALOG_RESOURCE = PROMPTS.resource
_CATALOG_SCHEMA_VERSION = 1
_PROMPT_METADATA_STRINGS = (
    "version", "role", "purpose", "stage", "lifecycle", "input_context", "output_contract",
)
_PROMPT_ROLES = frozenset({"system", "user", "system_fragment", "user_fragment", "payload_fragment", "user_seed"})


@dataclass(frozen=True, slots=True)
class _PromptCatalogSnapshot:
    version: str
    file_sha256: str
    effective_content_sha256: str
    prompts: Mapping[str, Mapping[str, Any]]
    template_hashes: Mapping[str, str]
    resolved_parameter_bindings: Mapping[str, Mapping[str, Any]]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _render_parameter_binding(binding: Mapping[str, Any]) -> tuple[Any, str]:
    """Resolve only a declared configuration key, with a bounded formatter."""

    if set(binding) != {"key", "format", "description"} or any(
        not isinstance(binding[field], str) or not binding[field].strip()
        for field in ("key", "format", "description")
    ):
        raise RuntimeError("LLM prompt parameter binding needs key, format and description")
    if binding["format"] != "english_integer_1_to_20":
        raise RuntimeError("LLM prompt parameter binding has an unsupported formatter")
    # Configuration provenance imports prompting locally. Keep the reverse
    # dependency local too, after the parameter module has defined its API.
    from .configuration import parameter

    value = parameter(binding["key"])
    if type(value) is not int or not 1 <= value <= 20:
        raise RuntimeError(f"LLM prompt parameter {binding['key']!r} must be an integer from 1 to 20")
    words = (
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen", "twenty",
    )
    return value, words[value - 1]


def _parse_prompt_catalog(raw: bytes) -> _PromptCatalogSnapshot:
    """Parse and fingerprint the same bytes, without a second filesystem read."""

    value = tomllib.loads(raw.decode("utf-8"))
    catalog = value.get("catalog")
    if not isinstance(catalog, dict) or not isinstance(catalog.get("version"), str) or not catalog["version"].strip():
        raise RuntimeError("LLM prompt catalog has no catalog version")
    if type(catalog.get("schema_version")) is not int or catalog["schema_version"] != _CATALOG_SCHEMA_VERSION:
        raise RuntimeError("LLM prompt catalog has an unsupported metadata schema version")
    prompts = value.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise RuntimeError("LLM prompt catalog does not contain a prompts table")
    normalized: dict[str, Mapping[str, Any]] = {}
    template_hashes: dict[str, str] = {}
    resolved_bindings: dict[str, Mapping[str, Any]] = {}
    for name, entry in prompts.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"LLM prompt catalog entry {name!r} must be a table")
        if not isinstance(entry.get("content"), str):
            raise RuntimeError(f"LLM prompt catalog entry {name!r} has no string content")
        for field in _PROMPT_METADATA_STRINGS:
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise RuntimeError(f"LLM prompt catalog entry {name!r} has no {field}")
        if entry["role"] not in _PROMPT_ROLES:
            raise RuntimeError(f"LLM prompt catalog entry {name!r} has an unsupported role")
        if entry["lifecycle"] not in {"runtime", "ui_seed", "legacy_only"}:
            raise RuntimeError(f"LLM prompt catalog entry {name!r} has an unsupported lifecycle")
        for field in ("constraints", "call_sites"):
            if not isinstance(entry.get(field), list) or not entry[field] or any(
                not isinstance(item, str) or not item.strip() for item in entry[field]
            ):
                raise RuntimeError(f"LLM prompt catalog entry {name!r} has no documented {field}")
        content = entry["content"].strip()
        template = Template(content)
        if not template.is_valid():
            raise RuntimeError(f"LLM prompt catalog entry {name!r} has an invalid template")
        placeholders = entry.get("placeholders")
        bindings = entry.get("parameter_bindings", {})
        if not isinstance(bindings, dict) or any(not isinstance(item, dict) for item in bindings.values()):
            raise RuntimeError(f"LLM prompt catalog entry {name!r} has invalid parameter bindings")
        if not isinstance(placeholders, dict) or set(placeholders).intersection(bindings) or set(placeholders) | set(bindings) != set(template.get_identifiers()) or any(
            not isinstance(item, str) or not item.strip() for item in placeholders.values()
        ):
            raise RuntimeError(f"LLM prompt catalog entry {name!r} has inconsistent placeholder documentation")
        template_hashes[str(name)] = _content_hash(content)
        if bindings:
            resolved = {}
            for slot, binding in bindings.items():
                parameter_value, rendered = _render_parameter_binding(binding)
                resolved[slot] = {"key": binding["key"], "value": parameter_value, "rendered": rendered}
            # Replace bound slots only. Preserve user-payload slots and $$
            # escapes verbatim for the later, single format_prompt pass.
            content = Template.pattern.sub(
                lambda match: resolved[slot]["rendered"]
                if (slot := match.group("named") or match.group("braced")) in resolved
                else match.group(0), content,
            )
            resolved_bindings[str(name)] = _freeze(resolved)
        normalized[str(name)] = _freeze({**entry, "content": content})
    # The effective hash covers the exact stripped UTF-8 templates returned by
    # load_prompt. Metadata, versions and file whitespace affect the file hash,
    # but are deliberately excluded from the effective-content fingerprint.
    contents = {name: entry["content"] for name, entry in normalized.items()}
    effective = json.dumps(contents, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _PromptCatalogSnapshot(
        version=catalog["version"], file_sha256=hashlib.sha256(raw).hexdigest(),
        effective_content_sha256=_content_hash(effective), prompts=MappingProxyType(normalized),
        template_hashes=MappingProxyType(template_hashes),
        resolved_parameter_bindings=MappingProxyType(resolved_bindings),
    )


@lru_cache(maxsize=1)
def _catalog_snapshot() -> _PromptCatalogSnapshot:
    raw = files(PROMPTS.package).joinpath(PROMPT_CATALOG_RESOURCE).read_bytes()
    return _parse_prompt_catalog(raw)


def _prompt_catalog() -> Mapping[str, Mapping[str, Any]]:
    return _catalog_snapshot().prompts


PROMPT_VERSIONS = MappingProxyType(
    {name: str(entry["version"]) for name, entry in _prompt_catalog().items()}
)


def prompt_names() -> tuple[str, ...]:
    """Return every prompt ID in the single runtime catalog."""

    return tuple(sorted(_prompt_catalog()))


def load_prompt(name: str) -> str:
    """Return only effective model text; metadata and comments never enter it."""

    entry = _prompt_catalog().get(name)
    if entry is None:
        raise ValueError(f"unknown prompt: {name}")
    return str(entry["content"])


def prompt_metadata(name: str) -> dict[str, Any]:
    """Return detached, JSON-safe documentation without the model content."""

    entry = _prompt_catalog().get(name)
    if entry is None:
        raise ValueError(f"unknown prompt: {name}")
    return {key: _plain(item) for key, item in entry.items() if key != "content"}


def prompt_catalog_provenance() -> dict[str, Any]:
    """Identify the already loaded catalog, never newer contents on disk.

    The file fingerprint and effective templates originate from one cached byte
    snapshot. There is no runtime hot reload. ``effective_content_sha256`` hashes
    sorted compact UTF-8 JSON mapping prompt IDs to the exact ``load_prompt``
    strings after declared configuration bindings; each ``content_sha256`` hashes
    that one string directly. ``template_sha256`` hashes the unresolved stripped
    catalog template. Resolved parameter values are captured once in this same
    snapshot and never recomputed against later configuration state. Per-prompt
    semantic versions are preserved when only documentation changes. Caller
    substitutions are run data, so this catalog identity does not replace the
    audit hash of a fully formatted message containing actual inputs.
    """

    snapshot = _catalog_snapshot()
    return {
        "catalog_version": snapshot.version,
        "catalog_schema_version": _CATALOG_SCHEMA_VERSION,
        "catalog_resource": f"{PROMPTS.package}/{PROMPT_CATALOG_RESOURCE}",
        "catalog_file_sha256": snapshot.file_sha256,
        "effective_content_sha256": snapshot.effective_content_sha256,
        "resolved_parameter_bindings": _plain(snapshot.resolved_parameter_bindings),
        "prompts": {
            name: {"version": entry["version"], "content_sha256": _content_hash(entry["content"]),
                   "template_sha256": snapshot.template_hashes[name]}
            for name, entry in sorted(snapshot.prompts.items())
        },
    }


def format_prompt(name: str, /, **values: Any) -> str:
    """Load and substitute one catalog template using ``$name`` placeholders."""

    try:
        return Template(load_prompt(name)).substitute(
            {key: str(value) for key, value in values.items()}
        )
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"prompt {name!r} requires template value {missing!r}") from exc
