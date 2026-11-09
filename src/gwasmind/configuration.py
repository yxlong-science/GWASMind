"""Validated process-lifetime configuration with reproducible, non-secret hashes.

The packaged TOML is the canonical parameter catalog. An optional local TOML
selected by GWASMIND_PARAMETERS_FILE may supply an [overrides] table; overrides
must use existing dotted keys and satisfy the canonical type/range constraints.
Restart the process after changing a configuration. There is no live reload.
This module is the only reader of the parameter catalog; see config_files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config_files import PARAMETERS
from .config_files import freeze_value as _freeze
from .config_files import plain_value as _plain

PARAMETERS_RESOURCE = PARAMETERS.resource
PARAMETERS_FILE_ENV = "GWASMIND_PARAMETERS_FILE"
_KEY = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")


class ConfigurationError(ValueError):
    """Configuration is invalid; do not start a run with silently changed values."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _checked_value(key: str, value: Any, spec: Mapping[str, Any]) -> Any:
    kind = spec.get("type")
    valid = {
        "int": isinstance(value, int) and not isinstance(value, bool),
        "float": isinstance(value, (int, float)) and not isinstance(value, bool),
        "bool": isinstance(value, bool),
        "str": isinstance(value, str),
        "str_list": isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value),
        "int_list": isinstance(value, (list, tuple)) and all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        "float_list": isinstance(value, (list, tuple)) and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) for item in value),
    }
    if not valid.get(kind, False):
        raise ConfigurationError(f"Parameter {key!r} must have type {kind!r}")
    if kind in {"float", "int"}:
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ConfigurationError(f"Parameter {key!r} must be finite")
        if "minimum" in spec and value < spec["minimum"]:
            raise ConfigurationError(f"Parameter {key!r} is below its minimum")
        if "maximum" in spec and value > spec["maximum"]:
            raise ConfigurationError(f"Parameter {key!r} exceeds its maximum")
    if "choices" in spec and value not in spec["choices"]:
        raise ConfigurationError(f"Parameter {key!r} is not an allowed choice")
    if isinstance(value, (str, list, tuple)):
        if len(value) < spec.get("min_length", 0) or len(value) > spec.get("max_length", 1_000_000):
            raise ConfigurationError(f"Parameter {key!r} has an invalid length")
    if kind == "str" and key.endswith("_url"):
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ConfigurationError(f"Parameter {key!r} must be a non-secret HTTP(S) base URL without credentials or query fields")
    return _freeze(float(value) if kind == "float" else value)


def _cross_constraints(values: Mapping[str, Any]) -> None:
    """Named numeric invariants; configuration contains no executable rule text."""
    for key in ("ranking.ld_r2_threshold", "novelty.ld_r2_threshold"):
        if key in values and values[key] <= 0:
            raise ConfigurationError(f"{key} must be in (0, 1]")
    for larger, smaller in (
        ("transport.graphql_max_operation_bytes", "transport.graphql_max_response_bytes"),
        ("transport.ensembl_retry_timeout_seconds", "transport.ensembl_timeout_seconds"),
        ("execution.max_concurrent_runs", "execution.default_concurrent_runs"),
        ("execution.max_concurrent_runs", "execution.auto_default_concurrent_loci"),
        ("web.general_history_total_chars", "web.general_history_item_chars"),
        ("gates.question_max_chars", "web.chat_question_max_chars"),
        ("web.chat_history_item_max_chars", "web.general_history_item_chars"),
        ("web.chat_analysis_context_max_chars", "web.legacy_evidence_context_max_chars"),
        ("plot.max_total_points", "plot.background_points"),
        ("analysis.gwas_significance_p", "plot.p_value_floor"),
        ("v2g.ld.r2_core", "v2g.ld.r2_extended"),
        ("v2g.coloc.h4_core", "v2g.coloc.h4_support"),
    ):
        if larger in values and smaller in values and values[larger] < values[smaller]:
            raise ConfigurationError(f"Parameter {larger!r} must be at least {smaller!r}")
    if "execution.duckdb_memory_limit" in values and not re.fullmatch(r"[1-9][0-9]*(?:MB|GB)", values["execution.duckdb_memory_limit"]):
        raise ConfigurationError("execution.duckdb_memory_limit must be a positive MB or GB quantity")
    if "web.stream_reconnect_delays_ms" in values:
        delays = values["web.stream_reconnect_delays_ms"]
        if not delays or any(item < 0 or item > 300_000 for item in delays):
            raise ConfigurationError("Reconnect delays need a nonempty sequence between 0 and 300000 milliseconds")
    if "plot.suggestive_logp" in values and "plot.background_bin_logp_width" in values:
        if math.ceil(values["plot.suggestive_logp"] / values["plot.background_bin_logp_width"]) > 128:
            raise ConfigurationError("Configured background sampling exceeds the 128-band interchange limit")


@dataclass(frozen=True)
class ParameterCatalog:
    version: str
    values: Mapping[str, Any]
    metadata: Mapping[str, Mapping[str, Any]]
    file_sha256: str
    effective_sha256: str
    override_sha256: str | None = None


def parse_parameter_catalog(raw: bytes, *, overrides: bytes | None = None) -> ParameterCatalog:
    """Validate bytes without environment or I/O; useful for startup checks/tests."""
    try:
        document = tomllib.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError("Cannot parse the runtime parameter TOML") from exc
    if not isinstance(document.get("catalog"), dict) or set(document) != {"catalog", "parameters"}:
        raise ConfigurationError("Parameter TOML must contain catalog metadata and parameter entries only")
    version = document["catalog"].get("version")
    rows = document.get("parameters")
    if not isinstance(version, str) or not version.strip() or not isinstance(rows, list) or not rows:
        raise ConfigurationError("Parameter catalog needs a version and parameter entries")
    if document["catalog"].get("schema_version", 1) != 1:
        raise ConfigurationError("Unsupported parameter schema version")
    values: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(key := row.get("key"), str) or not _KEY.fullmatch(key):
            raise ConfigurationError("Parameter entry needs a valid dotted key")
        if key in values:
            raise ConfigurationError(f"Duplicate parameter {key!r}")
        for field in ("description", "unit", "impact"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ConfigurationError(f"Parameter {key!r} lacks {field}")
        if not isinstance(row.get("used_by"), list) or not row["used_by"] or not all(isinstance(item, str) and item for item in row["used_by"]):
            raise ConfigurationError(f"Parameter {key!r} needs usage locations")
        if "value" not in row or "default" not in row:
            raise ConfigurationError(f"Parameter {key!r} needs value and documented default")
        for bound in ("minimum", "maximum"):
            if bound in row and (isinstance(row[bound], bool) or not isinstance(row[bound], (int, float)) or not math.isfinite(row[bound])):
                raise ConfigurationError(f"Parameter {key!r} has an invalid {bound}")
        if "minimum" in row and "maximum" in row and row["minimum"] > row["maximum"]:
            raise ConfigurationError(f"Parameter {key!r} has reversed bounds")
        _checked_value(key, row["default"], row)
        values[key] = _checked_value(key, row["value"], row)
        metadata[key] = _freeze({name: value for name, value in row.items() if name != "value"})
    if overrides is not None:
        try:
            supplied = tomllib.loads(overrides.decode("utf-8-sig"))
        except (UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError("Cannot parse local parameter overrides") from exc
        if set(supplied) != {"overrides"} or not isinstance(supplied["overrides"], dict):
            raise ConfigurationError("Local configuration must contain only an [overrides] table with quoted dotted keys")
        for key, value in supplied["overrides"].items():
            if key not in metadata:
                raise ConfigurationError(f"Unknown parameter override {key!r}")
            values[key] = _checked_value(key, value, metadata[key])
    _cross_constraints(values)
    return ParameterCatalog(version, _freeze(values), _freeze(metadata), hashlib.sha256(raw).hexdigest(),
                            _digest(values), hashlib.sha256(overrides).hexdigest() if overrides is not None else None)


@lru_cache(maxsize=1)
def _parameter_catalog() -> ParameterCatalog:
    raw = files(PARAMETERS.package).joinpath(PARAMETERS_RESOURCE).read_bytes()
    selected = os.environ.get(PARAMETERS_FILE_ENV)
    try:
        overrides = Path(selected).expanduser().read_bytes() if selected else None
    except OSError as exc:
        raise ConfigurationError("Cannot read the selected local parameter overrides") from exc
    return parse_parameter_catalog(raw, overrides=overrides)


def load_parameters() -> Mapping[str, Any]:
    return _parameter_catalog().values


def parameter(key: str) -> Any:
    try:
        return load_parameters()[key]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown runtime parameter {key!r}") from exc


def parameter_metadata() -> Mapping[str, Mapping[str, Any]]:
    return _parameter_catalog().metadata


def parameter_provenance() -> dict[str, Any]:
    catalog = _parameter_catalog()
    return {"version": catalog.version, "resource": PARAMETERS.source_path.removeprefix("src/"),
            "file_sha256": catalog.file_sha256, "effective_sha256": catalog.effective_sha256,
            "override_sha256": catalog.override_sha256, "effective_values": _plain(catalog.values)}


def public_runtime_parameters() -> dict[str, Any]:
    """Only browser-used non-secret settings; generated from the loaded catalog."""
    names = {
        "analysis.gwas_significance_p", "execution.max_variants_per_query",
        "execution.max_concurrent_runs", "execution.auto_default_concurrent_loci",
        "web.max_json_bytes", "web.chat_history_limit", "web.chat_history_item_max_chars",
        "web.stream_reconnect_attempts", "web.stream_reconnect_delays_ms",
    }
    values = {key: _plain(value) for key, value in load_parameters().items()
              if key.startswith("plot.") or key in names}
    return {"values": values, "effective_sha256": _parameter_catalog().effective_sha256}


def configuration_provenance() -> dict[str, Any]:
    # Lazy imports keep parameter consumers independent from prompt and gate code.
    from . import config_freeze
    from .gate_rules import gate_rules_provenance
    from .prompting import prompt_catalog_provenance

    provenance = {"parameters": parameter_provenance(), "gate_rules": gate_rules_provenance(),
                  "prompts": prompt_catalog_provenance()}
    # Every recorded run passes through here, so the freeze list is checked
    # once per process before the first configuration is recorded.
    config_freeze.check_configuration_freeze()
    return provenance


def main() -> None:
    """Export effective non-secret values for launchers without parsing TOML twice."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", action="store_true")
    args = parser.parse_args()
    # ASCII escapes: launchers read this through a pipe in the console code page,
    # which cannot encode every multilingual alias (for example on cp1252).
    print(json.dumps(configuration_provenance() if args.provenance else _plain(load_parameters()),
                     ensure_ascii=True, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
