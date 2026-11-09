"""The three versioned configuration catalogs that fix GWASMind's behavior.

Every behavior-relevant setting comes from exactly one of these packaged files.
Each file has one loader module, which owns its parsing, validation and single
process-lifetime snapshot; no other module reads a catalog directly.

========== ==================================== =====================
Catalog    Packaged resource                    Loader
========== ==================================== =====================
parameters ``gwasmind.config/parameters.toml``  ``gwasmind.configuration``
prompts    ``gwasmind.prompts/llm_prompts.toml`` ``gwasmind.prompting``
gate_rules ``gwasmind.config/gate_rules.toml``  ``gwasmind.gate_rules``
========== ==================================== =====================

The freeze list (``docs/CONFIG_FREEZE_2026-09-28.json``) records the versions,
hashes and key thresholds of these files; ``gwasmind.config_freeze`` verifies
the loaded snapshots against it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class CatalogFile:
    name: str
    package: str
    resource: str
    loader: str

    @property
    def source_path(self) -> str:
        """Project-relative path, as recorded in the freeze list."""
        return "src/" + self.package.replace(".", "/") + "/" + self.resource


PARAMETERS = CatalogFile("parameters", "gwasmind.config", "parameters.toml", "gwasmind.configuration")
PROMPTS = CatalogFile("prompts", "gwasmind.prompts", "llm_prompts.toml", "gwasmind.prompting")
GATE_RULES = CatalogFile("gate_rules", "gwasmind.config", "gate_rules.toml", "gwasmind.gate_rules")
CATALOG_FILES = (PARAMETERS, PROMPTS, GATE_RULES)


def freeze_value(value: Any) -> Any:
    """Return a read-only copy of parsed TOML data (tables and arrays)."""
    if isinstance(value, dict):
        return MappingProxyType({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    return value


def plain_value(value: Any) -> Any:
    """Return a detached, JSON-safe copy of frozen catalog data."""
    if isinstance(value, Mapping):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_value(item) for item in value]
    return value
