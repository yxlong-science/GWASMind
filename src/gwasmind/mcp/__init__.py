"""Stable semantic contracts above local or official Open Targets transports."""

from .catalog import (
    CATALOG_RELEASE,
    CATALOG_SOURCE,
    DATASET_CATALOG,
    LOCAL_DATASET_ALIASES,
    TOOL_SPECS,
    DatasetInspection,
    DatasetSpec,
    DatasetState,
    FrozenDatasetDiscovery,
    datasets_for_tool,
    discover_frozen_datasets,
)
from .contracts import (
    SCHEMA_VERSION,
    TOOL_ENVELOPE_JSON_SCHEMA,
    ArtifactReference,
    SemanticTool,
    ToolCoverage,
    ToolEnvelope,
    ToolError,
    ToolSpec,
    ToolStatus,
)
from .service import (
    ModuleAdapter,
    ModuleResult,
    ModuleStatus,
    SemanticToolService,
)
from .dispatcher import (
    SERVER_DESCRIPTION,
    SERVER_NAME,
    SemanticMCPDispatcher,
)

__all__ = [
    "ArtifactReference",
    "CATALOG_RELEASE",
    "CATALOG_SOURCE",
    "DATASET_CATALOG",
    "DatasetInspection",
    "DatasetSpec",
    "DatasetState",
    "FrozenDatasetDiscovery",
    "LOCAL_DATASET_ALIASES",
    "SCHEMA_VERSION",
    "SemanticTool",
    "TOOL_ENVELOPE_JSON_SCHEMA",
    "TOOL_SPECS",
    "ToolCoverage",
    "ToolEnvelope",
    "ToolError",
    "ToolSpec",
    "ToolStatus",
    "ModuleAdapter",
    "ModuleResult",
    "ModuleStatus",
    "SemanticToolService",
    "SERVER_DESCRIPTION",
    "SERVER_NAME",
    "SemanticMCPDispatcher",
    "datasets_for_tool",
    "discover_frozen_datasets",
]
