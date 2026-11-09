from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


SCHEMA_VERSION = "gwasmind-ot-tools/1"


class SemanticTool(StrEnum):
    """Stable GWASMind tools exposed above frozen or live Open Targets data."""

    GET_CAPABILITIES = "ot_get_capabilities"
    RESOLVE_ENTITIES = "ot_resolve_entities"
    GET_LOCUS_CONTEXT = "ot_get_locus_context"
    GET_LOCUS_GENE_EVIDENCE = "ot_get_locus_gene_evidence"
    GET_TARGET_PROFILES = "ot_get_target_profiles"
    GET_TARGET_DISEASE_EVIDENCE = "ot_get_target_disease_evidence"
    GET_DISEASE_PROFILES = "ot_get_disease_profiles"
    GET_DRUG_EVIDENCE = "ot_get_drug_evidence"
    ANALYZE_GENE_SET = "ot_analyze_gene_set"


class ToolStatus(StrEnum):
    """Unambiguous tool outcomes; ``no_data`` never means biological novelty."""

    SUCCESS = "success"
    PARTIAL = "partial"
    NO_DATA = "no_data"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"
    INVALID_INPUT = "invalid_input"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ToolCoverage:
    requested: tuple[str, ...] = ()
    completed: tuple[str, ...] = ()
    no_data: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    truncated: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = {
            "requested": self.requested,
            "completed": self.completed,
            "no_data": self.no_data,
            "unavailable": self.unavailable,
            "blocked": self.blocked,
            "failed": self.failed,
            "truncated": self.truncated,
        }
        for label, items in values.items():
            _validate_string_tuple(items, label=label)
        requested = set(self.requested)
        for label in (
            "completed",
            "no_data",
            "unavailable",
            "blocked",
            "failed",
            "truncated",
        ):
            unknown = set(values[label]).difference(requested)
            if unknown:
                raise ValueError(
                    f"coverage {label} contains unrequested module(s): "
                    + ", ".join(sorted(unknown))
                )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "requested": list(self.requested),
            "completed": list(self.completed),
            "no_data": list(self.no_data),
            "unavailable": list(self.unavailable),
            "blocked": list(self.blocked),
            "failed": list(self.failed),
            "truncated": list(self.truncated),
        }


@dataclass(frozen=True, slots=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False
    module: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("error code is required")
        if not self.message.strip():
            raise ValueError("error message is required")
        if self.module is not None and not self.module.strip():
            raise ValueError("error module cannot be empty")
        if self.details is not None:
            if not isinstance(self.details, Mapping):
                raise TypeError("error details must be an object")
            _to_json_value(self.details, path="$.error.details")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "module": self.module,
            **({"details": _to_json_value(self.details, path="$.error.details")} if self.details is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("artifact path is required")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.casefold()
        ):
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256.casefold()}


@dataclass(frozen=True, slots=True)
class ToolEnvelope:
    """Provider-neutral, strictly JSON-serializable semantic tool response."""

    tool_name: SemanticTool
    call_id: str
    status: ToolStatus
    provider: str
    coverage: ToolCoverage
    data_release: str = "unknown"
    snapshot_sha256: str = "unknown"
    policy_sha256: str = "unknown"
    input_sha256: str = "unknown"
    records: tuple[Mapping[str, Any], ...] = ()
    next_cursor: str | None = None
    warnings: tuple[str, ...] = ()
    error: ToolError | None = None
    raw_artifact: ArtifactReference | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("call_id", self.call_id),
            ("provider", self.provider),
            ("data_release", self.data_release),
            ("schema_version", self.schema_version),
        ):
            if not value.strip():
                raise ValueError(f"{label} is required")
        _validate_string_tuple(self.warnings, label="warnings", unique=False)
        if self.next_cursor is not None and not self.next_cursor.strip():
            raise ValueError("next_cursor cannot be empty")
        for index, record in enumerate(self.records):
            if not isinstance(record, Mapping):
                raise TypeError(f"records[{index}] must be a mapping")
            _to_json_value(record, path=f"$.records[{index}]")
        if self.status is ToolStatus.ERROR and self.error is None:
            raise ValueError("error status requires an error object")
        if self.status in {ToolStatus.SUCCESS, ToolStatus.NO_DATA} and self.error is not None:
            raise ValueError("successful or no-data status cannot contain an error object")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "tool_name": self.tool_name.value,
            "call_id": self.call_id,
            "status": self.status.value,
            "provider": self.provider,
            "data_release": self.data_release,
            "snapshot_sha256": self.snapshot_sha256,
            "policy_sha256": self.policy_sha256,
            "input_sha256": self.input_sha256,
            "coverage": self.coverage.to_dict(),
            "records": [_to_json_value(item, path="$.records") for item in self.records],
            "next_cursor": self.next_cursor,
            "warnings": list(self.warnings),
            "error": self.error.to_dict() if self.error else None,
            "raw_artifact": self.raw_artifact.to_dict() if self.raw_artifact else None,
        }
        return _to_json_value(value, path="$")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    tool: SemanticTool
    purpose: str
    input_schema: Mapping[str, Any]
    modules: tuple[str, ...]
    planner_stage: int
    call_when: str
    agent_callable: bool = True
    required_predecessors: tuple[SemanticTool, ...] = ()
    output_schema: Mapping[str, Any] = field(
        default_factory=lambda: TOOL_ENVELOPE_JSON_SCHEMA
    )

    def __post_init__(self) -> None:
        if not self.purpose.strip() or not self.call_when.strip():
            raise ValueError("tool purpose and call_when are required")
        if self.planner_stage < 0:
            raise ValueError("planner_stage cannot be negative")
        _validate_string_tuple(self.modules, label="modules")
        _to_json_value(self.input_schema, path="$.input_schema")
        _to_json_value(self.output_schema, path="$.output_schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.tool.value,
            "purpose": self.purpose,
            "input_schema": _to_json_value(self.input_schema, path="$.input_schema"),
            "output_schema": _to_json_value(self.output_schema, path="$.output_schema"),
            "modules": list(self.modules),
            "planner_stage": self.planner_stage,
            "call_when": self.call_when,
            "agent_callable": self.agent_callable,
            "required_predecessors": [item.value for item in self.required_predecessors],
        }


TOOL_ENVELOPE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "tool_name",
        "call_id",
        "status",
        "provider",
        "data_release",
        "snapshot_sha256",
        "policy_sha256",
        "input_sha256",
        "coverage",
        "records",
        "next_cursor",
        "warnings",
        "error",
        "raw_artifact",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "tool_name": {"enum": [item.value for item in SemanticTool]},
        "call_id": {"type": "string", "minLength": 1},
        "status": {"enum": [item.value for item in ToolStatus]},
        "provider": {"type": "string", "minLength": 1},
        "data_release": {"type": "string", "minLength": 1},
        "snapshot_sha256": {"type": "string", "minLength": 1},
        "policy_sha256": {"type": "string", "minLength": 1},
        "input_sha256": {"type": "string", "minLength": 1},
        "coverage": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "requested",
                "completed",
                "no_data",
                "unavailable",
                "blocked",
                "failed",
                "truncated",
            ],
            "properties": {
                name: {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
                for name in (
                    "requested",
                    "completed",
                    "no_data",
                    "unavailable",
                    "blocked",
                    "failed",
                    "truncated",
                )
            },
        },
        "records": {"type": "array", "items": {"type": "object"}},
        "next_cursor": {"type": ["string", "null"]},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "error": {
            "type": ["object", "null"],
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "retryable": {"type": "boolean"},
                "details": {"type": "object"},
                "module": {"type": ["string", "null"]},
            },
        },
        "raw_artifact": {
            "type": ["object", "null"],
            "properties": {
                "path": {"type": "string"},
                "sha256": {"type": "string"},
            },
        },
    },
}


def _validate_string_tuple(
    values: tuple[str, ...], *, label: str, unique: bool = True
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{label} must contain non-empty strings")
    if unique and len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")


def _to_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string object key")
            result[key] = _to_json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _to_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")
