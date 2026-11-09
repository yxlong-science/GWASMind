from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable, Mapping
from uuid import uuid4

from ..audit import sha256_json
from .catalog import DATASET_CATALOG, TOOL_SPECS, datasets_for_tool
from .contracts import (
    SemanticTool,
    ToolCoverage,
    ToolEnvelope,
    ToolError,
    ToolStatus,
    _to_json_value,
    _validate_string_tuple,
)


class ModuleStatus(StrEnum):
    SUCCESS = "success"
    NO_DATA = "no_data"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ModuleResult:
    module: str
    status: ModuleStatus
    records: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    truncated: bool = False
    error: ToolError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ModuleStatus):
            raise ValueError("module status must be a ModuleStatus value")
        _validate_string_tuple(self.warnings, label="warnings", unique=False)
        if not isinstance(self.truncated, bool):
            raise ValueError("module truncated flag must be boolean")
        if not self.module.strip():
            raise ValueError("module is required")
        if self.status is ModuleStatus.SUCCESS and not self.records:
            raise ValueError("successful module results require records")
        if self.status is ModuleStatus.NO_DATA and self.records:
            raise ValueError("no-data module results cannot contain records")
        if self.status is ModuleStatus.ERROR and self.error is None:
            raise ValueError("failed module results require an error")
        if self.status is not ModuleStatus.ERROR and self.error is not None:
            raise ValueError("only failed module results may contain an error")
        for index, record in enumerate(self.records):
            if not isinstance(record, Mapping):
                raise TypeError(f"records[{index}] must be a mapping")
            _to_json_value(record, path=f"$.records[{index}]")


ModuleHandler = Callable[
    [Mapping[str, Any]], ModuleResult | Awaitable[ModuleResult]
]


@dataclass(frozen=True, slots=True)
class ModuleAdapter:
    module: str
    tool: SemanticTool
    handler: ModuleHandler

    def __post_init__(self) -> None:
        if not self.module.strip():
            raise ValueError("adapter module is required")
        if not callable(self.handler):
            raise TypeError("adapter handler must be callable")


class SemanticToolService:
    """Sequential, provider-neutral execution of registered semantic modules.

    Module handlers run sequentially by design. Local adapters can therefore
    enforce conservative database limits without accidental fan-out from one
    model tool call.
    """

    def __init__(
        self,
        *,
        provider: str,
        data_release: str,
        snapshot_sha256: str = "unknown",
        schema_sha256: str = "unknown",
        policy_sha256: str = "unknown",
        adapters: tuple[ModuleAdapter, ...] = (),
        blocked_modules: frozenset[str] = frozenset(),
        limits: Mapping[str, Any] | None = None,
        dataset_capabilities: Mapping[str, Any] | None = None,
        upstream_metadata: Mapping[str, Any] | None = None,
    ):
        if not provider.strip() or not data_release.strip():
            raise ValueError("provider and data_release are required")
        registry: dict[tuple[SemanticTool, str], ModuleAdapter] = {}
        for adapter in adapters:
            key = (adapter.tool, adapter.module)
            if key in registry:
                raise ValueError(
                    f"duplicate module adapter: {adapter.tool.value}/{adapter.module}"
                )
            if adapter.module not in _allowed_modules(adapter.tool):
                raise ValueError(
                    f"module {adapter.module!r} is not assigned to {adapter.tool.value}"
                )
            registry[key] = adapter
        self.provider = provider
        self.data_release = data_release
        self.snapshot_sha256 = snapshot_sha256
        self.schema_sha256 = schema_sha256
        self.policy_sha256 = policy_sha256
        self.adapters = registry
        self.blocked_modules = blocked_modules
        self.limits = dict(limits or {"max_variants": 100, "max_targets": 200})
        self.dataset_capabilities = dict(dataset_capabilities or {})
        self.upstream_metadata = dict(upstream_metadata or {})

    async def call(
        self,
        tool: SemanticTool,
        arguments: Mapping[str, Any],
    ) -> ToolEnvelope:
        call_id = str(uuid4())
        try:
            if not isinstance(arguments, Mapping):
                raise TypeError("arguments must be an object")
            _to_json_value(arguments, path="$")
            input_sha256 = sha256_json(dict(arguments))
        except (TypeError, ValueError) as exc:
            return self._envelope(
                tool=tool, call_id=call_id, status=ToolStatus.INVALID_INPUT,
                coverage=ToolCoverage(), input_sha256="unknown",
                error=ToolError("invalid_input", str(exc)),
            )
        error = _validate_top_level(tool, arguments)
        if error is not None:
            return self._envelope(
                tool=tool,
                call_id=call_id,
                status=ToolStatus.INVALID_INPUT,
                coverage=ToolCoverage(),
                input_sha256=input_sha256,
                error=error,
            )
        if tool is SemanticTool.GET_CAPABILITIES:
            return self._capabilities(call_id=call_id, input_sha256=input_sha256)

        if tool is SemanticTool.ANALYZE_GENE_SET:
            analysis_modules = {
                "pathway_overrepresentation": "gene_set.pathway",
                "go_overrepresentation": "gene_set.go",
                "tissue_coverage": "gene_set.tissue_coverage",
                "evidence_coverage": "gene_set.evidence_coverage",
            }
            requested = tuple(
                analysis_modules.get(str(item), str(item))
                for item in arguments.get("analyses", ())
            )
            if len(requested) != len(set(requested)):
                return self._envelope(
                    tool=tool, call_id=call_id, status=ToolStatus.INVALID_INPUT,
                    coverage=ToolCoverage(), input_sha256=input_sha256,
                    error=ToolError("invalid_analyses", "analyses resolve to duplicate modules"),
                )
        else:
            requested = tuple(
                arguments.get("modules", TOOL_SPECS[tool].modules)
            )
        allowed = _allowed_modules(tool)
        invalid = tuple(module for module in requested if module not in allowed)
        if invalid:
            return self._envelope(
                tool=tool,
                call_id=call_id,
                status=ToolStatus.INVALID_INPUT,
                coverage=ToolCoverage(requested=requested),
                input_sha256=input_sha256,
                error=ToolError(
                    code="unknown_module",
                    message="module(s) are not assigned to this tool: "
                    + ", ".join(sorted(invalid)),
                ),
            )

        completed: list[str] = []
        no_data: list[str] = []
        unavailable: list[str] = []
        blocked: list[str] = []
        failed: list[str] = []
        truncated: list[str] = []
        records: list[Mapping[str, Any]] = []
        warnings: list[str] = []
        first_error: ToolError | None = None
        for module in requested:
            if module in self.blocked_modules:
                blocked.append(module)
                continue
            adapter = self.adapters.get((tool, module))
            if adapter is None:
                unavailable.append(module)
                continue
            try:
                value = adapter.handler(arguments)
                result = await value if inspect.isawaitable(value) else value
                if not isinstance(result, ModuleResult):
                    raise TypeError("module handler did not return ModuleResult")
                if result.module != module:
                    raise ValueError("module handler returned a mismatched module name")
                result.__post_init__()
                # Module results contain mutable mappings even though the wrapper
                # is frozen. Snapshot them before another adapter can mutate a
                # provider cache or reused result while this call is awaiting I/O.
                result_records = tuple(
                    _to_json_value(record, path=f"$.records[{index}]")
                    for index, record in enumerate(result.records)
                )
                if result.status is ModuleStatus.SUCCESS:
                    completed.append(module)
                    records.extend(result_records)
                    if result.truncated:
                        truncated.append(module)
                elif result.status is ModuleStatus.NO_DATA:
                    no_data.append(module)
                    # No rows in a retained subset is not complete no-data
                    # coverage. Preserve both facts so callers cannot turn a
                    # bounded live connection into evidence of absence.
                    if result.truncated:
                        truncated.append(module)
                elif result.status is ModuleStatus.BLOCKED:
                    blocked.append(module)
                else:
                    failed.append(module)
                    first_error = first_error or result.error
                warnings.extend(result.warnings)
            except Exception as exc:  # module boundary must fail closed
                from ..providers.retrieval_recovery import safe_transport_failure
                failure_details = safe_transport_failure(exc)
                failed.append(module)
                module_error = ToolError(
                    code="module_error",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=failure_details.get("retryable", False),
                    module=module,
                    details=failure_details or None,
                )
                first_error = first_error or module_error

        coverage = ToolCoverage(
            requested=requested,
            completed=tuple(completed),
            no_data=tuple(no_data),
            unavailable=tuple(unavailable),
            blocked=tuple(blocked),
            failed=tuple(failed),
            truncated=tuple(truncated),
        )
        status = _overall_status(coverage)
        error_value = first_error if status is ToolStatus.ERROR else None
        if status is ToolStatus.PARTIAL and first_error is not None:
            warnings.append(
                f"{first_error.module or 'module'} failed: {first_error.message}"
            )
        return self._envelope(
            tool=tool,
            call_id=call_id,
            status=status,
            coverage=coverage,
            input_sha256=input_sha256,
            records=tuple(records),
            warnings=tuple(dict.fromkeys(warnings)),
            error=error_value,
        )

    def _capabilities(self, *, call_id: str, input_sha256: str) -> ToolEnvelope:
        all_modules = tuple(
            sorted(
                {
                    module
                    for dataset in DATASET_CATALOG.values()
                    for module in dataset.modules
                }.union(
                    module for spec in TOOL_SPECS.values() for module in spec.modules
                )
            )
        )
        ready = tuple(sorted(
            {module for _, module in self.adapters}.difference(self.blocked_modules)
        ))
        blocked = tuple(sorted(set(all_modules).intersection(self.blocked_modules)))
        unavailable = tuple(
            module
            for module in all_modules
            if module not in ready and module not in self.blocked_modules
        )
        record = {
            "identity": "GWASMind unofficial semantic wrapper over Open Targets Platform",
            "schema_sha256": self.schema_sha256,
            "limits": self.limits,
            "datasets": self.dataset_capabilities,
            "upstream": self.upstream_metadata,
            "tools": [spec.to_dict() for spec in TOOL_SPECS.values()],
            "modules": {
                module: (
                    "blocked"
                    if module in self.blocked_modules
                    else "ready"
                    if any(key_module == module for _, key_module in self.adapters)
                    else "unavailable"
                )
                for module in all_modules
            },
        }
        coverage = ToolCoverage(
            requested=all_modules,
            completed=ready,
            unavailable=unavailable,
            blocked=blocked,
        )
        return self._envelope(
            tool=SemanticTool.GET_CAPABILITIES,
            call_id=call_id,
            status=ToolStatus.SUCCESS,
            coverage=coverage,
            input_sha256=input_sha256,
            records=(record,),
        )

    def _envelope(
        self,
        *,
        tool: SemanticTool,
        call_id: str,
        status: ToolStatus,
        coverage: ToolCoverage,
        input_sha256: str,
        records: tuple[Mapping[str, Any], ...] = (),
        warnings: tuple[str, ...] = (),
        error: ToolError | None = None,
    ) -> ToolEnvelope:
        return ToolEnvelope(
            tool_name=tool,
            call_id=call_id,
            status=status,
            provider=self.provider,
            data_release=self.data_release,
            snapshot_sha256=self.snapshot_sha256,
            policy_sha256=self.policy_sha256,
            input_sha256=input_sha256,
            coverage=coverage,
            records=records,
            warnings=warnings,
            error=error,
        )


def _allowed_modules(tool: SemanticTool) -> frozenset[str]:
    values = set(TOOL_SPECS[tool].modules)
    for dataset in datasets_for_tool(tool):
        values.update(dataset.modules)
    return frozenset(values)


def _validate_top_level(
    tool: SemanticTool, arguments: Mapping[str, Any]
) -> ToolError | None:
    if not isinstance(arguments, Mapping):
        return ToolError("invalid_input", "arguments must be an object")
    schema = TOOL_SPECS[tool].input_schema
    allowed = set(schema.get("properties", {}))
    unknown = set(arguments).difference(allowed)
    if unknown:
        return ToolError(
            "unknown_argument",
            "unknown argument(s): " + ", ".join(sorted(unknown)),
        )
    missing = set(schema.get("required", ())).difference(arguments)
    if missing:
        return ToolError(
            "missing_argument",
            "missing required argument(s): " + ", ".join(sorted(missing)),
        )
    schema_error = _validate_schema_value(arguments, schema, path="$")
    if schema_error is not None:
        return ToolError("invalid_argument", schema_error)
    modules = arguments.get("modules")
    if modules is not None:
        if not isinstance(modules, (list, tuple)) or not modules:
            return ToolError("invalid_modules", "modules must be a non-empty array")
        if any(not isinstance(module, str) or not module.strip() for module in modules):
            return ToolError("invalid_modules", "modules must contain non-empty strings")
        if len(modules) != len(set(modules)):
            return ToolError("invalid_modules", "modules must not contain duplicates")
    return None


def _validate_schema_value(
    value: Any, schema: Mapping[str, Any], *, path: str
) -> str | None:
    if "const" in schema and value != schema["const"]:
        return f"{path} must equal {schema['const']!r}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']!r}"
    expected = schema.get("type")
    if expected is not None:
        expected_types = (expected,) if isinstance(expected, str) else tuple(expected)
        if not any(_matches_json_type(value, item) for item in expected_types):
            return f"{path} must have JSON type {' or '.join(expected_types)}"
        if value is None:
            return None
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = set(schema.get("required", ()))
        missing = required.difference(value)
        if missing:
            return f"{path} is missing: {', '.join(sorted(missing))}"
        if schema.get("additionalProperties") is False:
            unknown = set(value).difference(properties)
            if unknown:
                return f"{path} contains unknown fields: {', '.join(sorted(unknown))}"
        alternatives = schema.get("anyOf")
        if alternatives:
            satisfied = False
            for alternative in alternatives:
                names = tuple(alternative.get("required", ()))
                if names and all(
                    name in value and value[name] not in (None, "", [], {}, ())
                    for name in names
                ):
                    satisfied = True
                    break
            if not satisfied:
                return f"{path} must provide one non-empty alternative filter"
        for key, nested in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, Mapping):
                error = _validate_schema_value(
                    nested, child_schema, path=f"{path}.{key}"
                )
                if error is not None:
                    return error
    elif isinstance(value, (list, tuple)):
        if len(value) < int(schema.get("minItems", 0)):
            return f"{path} contains too few items"
        maximum = schema.get("maxItems")
        if maximum is not None and len(value) > int(maximum):
            return f"{path} contains too many items"
        if schema.get("uniqueItems"):
            serialized = [repr(item) for item in value]
            if len(serialized) != len(set(serialized)):
                return f"{path} must not contain duplicate items"
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                error = _validate_schema_value(
                    item, item_schema, path=f"{path}[{index}]"
                )
                if error is not None:
                    return error
    elif isinstance(value, str):
        minimum = schema.get("minLength")
        if minimum is not None and len(value) < int(minimum):
            return f"{path} is too short"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            return f"{path} is below the minimum {minimum}"
        if maximum is not None and value > maximum:
            return f"{path} exceeds the maximum {maximum}"
    return None


def _matches_json_type(value: Any, expected: str) -> bool:
    return {
        "null": value is None,
        "object": isinstance(value, Mapping),
        "array": isinstance(value, (list, tuple)),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }.get(expected, True)


def _overall_status(coverage: ToolCoverage) -> ToolStatus:
    has_completed = bool(coverage.completed)
    has_checked = has_completed or bool(coverage.no_data)
    has_gap = bool(
        coverage.unavailable
        or coverage.blocked
        or coverage.failed
        or coverage.truncated
    )
    if has_checked and has_gap:
        return ToolStatus.PARTIAL
    if has_completed:
        return ToolStatus.SUCCESS
    if coverage.no_data:
        return ToolStatus.NO_DATA
    if coverage.failed:
        return ToolStatus.ERROR
    if coverage.blocked:
        return ToolStatus.BLOCKED
    return ToolStatus.UNAVAILABLE
