"""Verify the loaded configuration against the recorded configuration freeze list.

The freeze list records, for each of the three catalogs, the raw file SHA-256,
the catalog version and the effective-content hash, plus the values of key
scientific thresholds. Verification compares those records with the snapshots
the process actually loaded, so it describes the configuration that ran, not
newer bytes on disk.

``configuration_provenance()`` runs this check once per process. Drift is
reported as a ``ConfigurationFreezeWarning``; with
``GWASMIND_REQUIRE_CONFIG_FREEZE=1`` it is a ``ConfigurationFreezeError``
instead. Local parameter overrides (``GWASMIND_PARAMETERS_FILE``) are a
supported feature: their effect on frozen thresholds is reported separately
and is not drift of the packaged catalogs.

The freeze list lives in the project's ``docs`` directory. A distribution that
ships only ``src`` reports the freeze list as unavailable.
"""
from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config_files import GATE_RULES, PARAMETERS, PROMPTS, plain_value

FREEZE_MANIFEST_NAME = "CONFIG_FREEZE_2026-09-28.json"
FREEZE_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "docs" / FREEZE_MANIFEST_NAME
REQUIRE_FREEZE_ENV = "GWASMIND_REQUIRE_CONFIG_FREEZE"
_MANIFEST_SECTIONS = ("files", "gate_rules", "parameters", "prompts", "frozen_thresholds")


class ConfigurationFreezeError(RuntimeError):
    """The loaded configuration differs from the freeze list and strict mode is on."""


class ConfigurationFreezeWarning(UserWarning):
    """The loaded configuration differs from the freeze list."""


@dataclass(frozen=True, slots=True)
class FreezeMismatch:
    field: str
    expected: Any
    actual: Any


@dataclass(frozen=True, slots=True)
class FreezeReport:
    status: str  # "verified", "drift" or "unavailable"
    manifest: str | None
    mismatches: tuple[FreezeMismatch, ...] = ()
    overrides_active: bool = False
    overridden_thresholds: tuple[FreezeMismatch, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "manifest": self.manifest, "overrides_active": self.overrides_active,
            "mismatches": [{"field": item.field, "expected": item.expected, "actual": item.actual}
                           for item in self.mismatches],
            "overridden_thresholds": [{"key": item.field, "frozen": item.expected, "effective": item.actual}
                                      for item in self.overridden_thresholds],
        }


def load_freeze_manifest(path: str | Path | None = None) -> dict[str, Any] | None:
    resolved = Path(path) if path is not None else FREEZE_MANIFEST_PATH
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def loaded_configuration_fingerprint() -> dict[str, Any]:
    """Versions and hashes of the snapshots this process loaded, keyed like the freeze list."""
    from .configuration import _parameter_catalog
    from .gate_rules import load_gate_rules
    from .prompting import prompt_catalog_provenance

    parameters = _parameter_catalog()
    rules = load_gate_rules()
    prompts = prompt_catalog_provenance()
    return {
        "files": {
            PARAMETERS.resource: {"path": PARAMETERS.source_path, "sha256": parameters.file_sha256},
            GATE_RULES.resource: {"path": GATE_RULES.source_path, "sha256": rules.raw_sha256},
            PROMPTS.resource: {"path": PROMPTS.source_path, "sha256": prompts["catalog_file_sha256"]},
        },
        "gate_rules": {"version": rules.version, "raw_sha256": rules.raw_sha256,
                       "effective_sha256": rules.effective_sha256},
        "parameters": {"effective_sha256": parameters.effective_sha256},
        "prompts": {key: prompts[key] for key in (
            "catalog_version", "catalog_schema_version", "catalog_resource",
            "catalog_file_sha256", "effective_content_sha256")},
        "overrides_active": parameters.override_sha256 is not None,
        "effective_values": plain_value(parameters.values),
    }


def verify_configuration_freeze(manifest: Mapping[str, Any] | None = None, *,
                                manifest_path: str | Path | None = None) -> FreezeReport:
    """Compare every recorded hash, version and threshold with the loaded snapshots."""
    location = str(manifest_path or FREEZE_MANIFEST_PATH) if manifest is None else None
    if manifest is None:
        manifest = load_freeze_manifest(manifest_path)
        if manifest is None:
            return FreezeReport("unavailable", location)
    malformed = [FreezeMismatch(f"manifest.{section}", "present", None)
                 for section in _MANIFEST_SECTIONS if not isinstance(manifest.get(section), Mapping)]
    if manifest.get("schema_version") != "1":
        malformed.append(FreezeMismatch("manifest.schema_version", "1", manifest.get("schema_version")))
    if malformed:
        return FreezeReport("drift", location, tuple(malformed))
    loaded = loaded_configuration_fingerprint()
    overrides_active = loaded["overrides_active"]
    mismatches: list[FreezeMismatch] = []

    def compare(field: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            mismatches.append(FreezeMismatch(field, expected, actual))

    for name, frozen in manifest["files"].items():
        current = loaded["files"].get(name)
        if current is None:
            mismatches.append(FreezeMismatch(f"files.{name}", frozen, None))
            continue
        compare(f"files.{name}.path", frozen["path"], current["path"])
        compare(f"files.{name}.sha256", frozen["sha256"], current["sha256"])
    for section in ("gate_rules", "prompts"):
        for key, expected in manifest[section].items():
            compare(f"{section}.{key}", expected, loaded[section].get(key))
    # Local overrides intentionally change effective values; the packaged file
    # itself is still verified through its raw SHA-256 above.
    if not overrides_active:
        compare("parameters.effective_sha256", manifest["parameters"]["effective_sha256"],
                loaded["parameters"]["effective_sha256"])
    overridden: list[FreezeMismatch] = []
    for key, expected in manifest["frozen_thresholds"].items():
        actual = loaded["effective_values"].get(key)
        if expected == actual and type(expected) is type(actual):
            continue
        if overrides_active:
            overridden.append(FreezeMismatch(key, expected, actual))
        else:
            mismatches.append(FreezeMismatch(f"frozen_thresholds.{key}", expected, actual))
    return FreezeReport("drift" if mismatches else "verified", location, tuple(mismatches),
                        overrides_active, tuple(overridden))


@lru_cache(maxsize=1)
def check_configuration_freeze() -> FreezeReport:
    """Run the freeze verification once per process; warn on drift or raise in strict mode."""
    report = verify_configuration_freeze()
    if report.status == "drift":
        fields = ", ".join(item.field for item in report.mismatches)
        message = f"Loaded configuration differs from {FREEZE_MANIFEST_NAME}: {fields}"
        if os.environ.get(REQUIRE_FREEZE_ENV) == "1":
            raise ConfigurationFreezeError(message)
        warnings.warn(message, ConfigurationFreezeWarning, stacklevel=3)
    if report.overridden_thresholds:
        keys = ", ".join(item.field for item in report.overridden_thresholds)
        warnings.warn(f"Local parameter overrides change frozen thresholds: {keys}",
                      ConfigurationFreezeWarning, stacklevel=3)
    return report


def main() -> None:
    """Print the freeze verification report; exit with status 1 on drift."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", help="freeze list to verify against (default: the project freeze list)")
    args = parser.parse_args()
    report = verify_configuration_freeze(manifest_path=args.manifest)
    # ASCII escapes keep the report printable through a non-UTF-8 pipe (for
    # example a Windows code page), whatever characters the manifest path has.
    print(json.dumps(report.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
    raise SystemExit(1 if report.status == "drift" else 0)


if __name__ == "__main__":
    main()
