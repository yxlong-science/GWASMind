"""Validated, immutable policy metadata for the two evidence gates.

TOML selects known checks, their severity, capability prerequisites, and the
validator registry. It cannot execute expressions or disable scientific checks.
This module is the only reader of gate_rules.toml. The packaged catalog has a
single process snapshot; an explicit path (used to test modified rules) is
parsed separately and never replaces it.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .config_files import GATE_RULES

GATE_RULES_PATH = Path(str(files(GATE_RULES.package).joinpath(GATE_RULES.resource)))
_IDENTITY_GATES = frozenset({"variant_integrity", "build_integrity", "allele_integrity", "evidence_availability"})
_RANKING_PREREQUISITES = frozenset({"locus_genes_1mb", "direct_locus_gene_evidence"})
_REQUIRED_VALIDATORS = {
    "ranking": frozenset({"ranking_schema_evidence", "ranking_scientific_assertions"}),
    "report": frozenset({"report_schema_evidence_consistency"}),
    "report_fragment": frozenset({"report_fragment_consistency"}),
}


class GateRulesError(ValueError):
    """Gate configuration is invalid or would bypass a scientific invariant."""


@dataclass(frozen=True, slots=True)
class GateRule:
    check_id: str
    gate: str
    severity: str
    display_group: str
    role: str
    code: str
    depends_on: tuple[str, ...]
    required_for: tuple[str, ...]
    allow_hard_stop: bool

    def metadata(self) -> dict[str, Any]:
        return {"gate": self.gate, "stage": "safety" if self.gate == "gate1" else "response",
                "display_group": self.display_group, "role": self.role,
                "code": self.code, "depends_on": list(self.depends_on)}


@dataclass(frozen=True, slots=True)
class GateValidator:
    validator_id: str
    target: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class GateRules:
    version: str
    rules: Mapping[str, GateRule]
    validators: tuple[GateValidator, ...]
    complete_module_statuses: tuple[str, ...]
    incomplete_module_statuses: tuple[str, ...]
    path: str
    raw_sha256: str
    effective_sha256: str

    def required_checks(self, capability: str) -> tuple[str, ...]:
        return tuple(rule.check_id for rule in self.rules.values() if capability in rule.required_for)

    def validator_ids(self, target: str) -> tuple[str, ...]:
        return tuple(item.validator_id for item in self.validators if item.target == target and item.enabled)


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise GateRulesError(f"{label} must be an array of nonempty strings")
    if len(set(value)) != len(value):
        raise GateRulesError(f"{label} contains duplicate values")
    return tuple(value)


def load_gate_rules(path: str | Path | None = None) -> GateRules:
    """Return the packaged rules snapshot, or the cached rules of an explicit file."""
    if path is None:
        return _packaged_gate_rules()
    resolved = Path(path).resolve()
    if resolved == GATE_RULES_PATH.resolve():
        return _packaged_gate_rules()
    return _gate_rules_file(resolved)


@lru_cache(maxsize=1)
def _packaged_gate_rules() -> GateRules:
    raw = files(GATE_RULES.package).joinpath(GATE_RULES.resource).read_bytes()
    return parse_gate_rules(raw, path=str(GATE_RULES_PATH.resolve()))


@lru_cache(maxsize=None)
def _gate_rules_file(resolved: Path) -> GateRules:
    return parse_gate_rules(resolved.read_bytes(), path=str(resolved))


def parse_gate_rules(raw: bytes, *, path: str) -> GateRules:
    """Validate rule bytes without I/O; ``path`` is recorded as provenance only."""
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise GateRulesError(f"invalid gate rules TOML: {path}") from exc
    if set(document) != {"version", "module_conditions", "rules", "validators"}:
        raise GateRulesError("gate rules require version, module_conditions, rules, and validators only")
    version = document["version"]
    if not isinstance(version, str) or not version.strip():
        raise GateRulesError("gate rules version is required")
    rules: dict[str, GateRule] = {}
    for check_id, value in document["rules"].items():
        if not isinstance(value, dict) or set(value) != {
            "gate", "severity", "display_group", "role", "code", "depends_on", "required_for", "allow_hard_stop",
        }:
            raise GateRulesError(f"invalid fields for gate rule {check_id}")
        if value["gate"] not in {"gate1", "gate2"} or value["severity"] not in {"hard_gate", "warning", "note"}:
            raise GateRulesError(f"invalid gate/severity for {check_id}")
        if value["role"] not in {"cause", "consequence", "output_validation"}:
            raise GateRulesError(f"invalid display role for {check_id}")
        if any(not isinstance(value[name], str) or not value[name].strip() for name in ("display_group", "code")):
            raise GateRulesError(f"display group and code are required for {check_id}")
        if not isinstance(value["allow_hard_stop"], bool):
            raise GateRulesError(f"allow_hard_stop must be boolean for {check_id}")
        rule = GateRule(check_id=check_id, **{**value,
            "depends_on": _strings(value["depends_on"], f"{check_id}.depends_on"),
            "required_for": _strings(value["required_for"], f"{check_id}.required_for")})
        if not set(rule.required_for).issubset({"report", "ranking"}):
            raise GateRulesError(f"unknown capability for {check_id}")
        if rule.severity == "hard_gate" and not rule.allow_hard_stop:
            raise GateRulesError(f"hard stop is not permitted for {check_id}")
        rules[check_id] = rule
    # Identity is enforced by the input/retrieval layer's explicit S1 receipt.
    # These observations remain mandatory audit aliases, but a partial source
    # failure or an empty successful search must not become an SSR hard stop.
    for check_id in _IDENTITY_GATES:
        rule = rules.get(check_id)
        if rule is None or rule.gate != "gate1":
            raise GateRulesError(f"mandatory input/data observation cannot be removed: {check_id}")
    for check_id in _RANKING_PREREQUISITES:
        rule = rules.get(check_id)
        if rule is None or rule.gate != "gate1":
            raise GateRulesError(f"mandatory ranking observation cannot be removed: {check_id}")
    for rule in rules.values():
        if rule.gate == "gate1" and (rule.severity == "hard_gate" or rule.required_for or rule.allow_hard_stop):
            raise GateRulesError(f"SSR observations cannot be configured as execution stops: {rule.check_id}")
        # Gate 2 corrects or removes unsupported model content; it never stops
        # an analysis or adds a scientific warning.
        if rule.gate == "gate2" and (rule.severity != "note" or rule.allow_hard_stop):
            raise GateRulesError(f"Gate 2 output checks must be non-blocking notes: {rule.check_id}")
    codes = [rule.code for rule in rules.values()]
    if len(set(codes)) != len(codes):
        raise GateRulesError("every gate rule requires its own code")
    for rule in rules.values():
        if not set(rule.depends_on).issubset(rules):
            raise GateRulesError(f"unknown prerequisite for {rule.check_id}")
    conditions = document["module_conditions"]
    if set(conditions) != {"complete", "incomplete"}:
        raise GateRulesError("module_conditions require complete and incomplete")
    complete = _strings(conditions["complete"], "module_conditions.complete")
    incomplete = _strings(conditions["incomplete"], "module_conditions.incomplete")
    if not {"success", "no_data"}.issubset(complete) or set(complete) - {"complete", "completed", "success", "no_data", "not_applicable"}:
        raise GateRulesError("failed or incomplete retrieval cannot be configured as completed")
    if set(complete).intersection(incomplete) or not {"missing", "unavailable", "failed", "error", "partial", "truncated", "blocked"}.issubset(incomplete):
        raise GateRulesError("incomplete retrieval statuses cannot be hidden")
    validators = []
    for validator_id, value in document["validators"].items():
        if set(value) != {"target", "enabled"} or not isinstance(value["enabled"], bool):
            raise GateRulesError(f"invalid validator configuration: {validator_id}")
        target = value["target"]
        if target not in _REQUIRED_VALIDATORS or validator_id not in _REQUIRED_VALIDATORS[target]:
            raise GateRulesError(f"unknown validator registry entry: {validator_id}")
        validators.append(GateValidator(validator_id, target, value["enabled"]))
    for target, required in _REQUIRED_VALIDATORS.items():
        enabled = {item.validator_id for item in validators if item.target == target and item.enabled}
        if enabled != required:
            raise GateRulesError(f"mandatory scientific validators cannot be disabled: {target}")
    effective = {"version": version, "rules": {key: asdict(value) for key, value in rules.items()},
        "validators": [asdict(item) for item in validators],
        "module_conditions": {"complete": complete, "incomplete": incomplete}}
    effective_raw = json.dumps(effective, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return GateRules(version, MappingProxyType(rules), tuple(validators), complete, incomplete,
        path, hashlib.sha256(raw).hexdigest(), hashlib.sha256(effective_raw).hexdigest())


def gate_rule(check_id: str) -> GateRule:
    return load_gate_rules().rules[check_id]


def gate_rules_provenance() -> dict[str, str]:
    loaded = load_gate_rules()
    return {"version": loaded.version, "path": loaded.path,
            "raw_sha256": loaded.raw_sha256, "effective_sha256": loaded.effective_sha256}


def group_gate_checks(checks: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    """Keep data causes, dependent execution outcomes, and output review separate."""
    nonpassing = {check["check_id"] for check in checks if check["outcome"] in {"warn", "fail"}}
    result: dict[str, Any] = {}
    for gate in ("gate1", "gate2"):
        selected = [check for check in checks if check.get("gate", "gate1") == gate]
        groups: dict[str, Any] = {}
        for check in selected:
            if check["outcome"] not in {"warn", "fail"}:
                continue
            group = groups.setdefault(check.get("display_group", "additional_review"),
                {"causes": [], "consequences": [], "output_validation": []})
            role = check.get("role", "cause")
            bucket = {"cause": "causes", "consequence": "consequences", "output_validation": "output_validation"}[role]
            group[bucket].append({"check_id": check["check_id"], "code": check.get("code", check["check_id"]),
                "caused_by": [identifier for identifier in check.get("depends_on", ()) if identifier in nonpassing]})
        result[gate] = {"check_ids": [check["check_id"] for check in selected],
            "status": "STOPPED" if any(check["severity"] == "hard_gate" and check["outcome"] == "fail" for check in selected)
            else "PASS_WITH_WARNINGS" if any(check["outcome"] in {"warn", "fail"} for check in selected)
            else "PASS" if selected else "NOT_RUN",
            "groups": groups}
    return result
