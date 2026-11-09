from __future__ import annotations

from typing import Any, Mapping

from .catalog import TOOL_SPECS
from .contracts import SemanticTool
from .service import SemanticToolService


SERVER_NAME = "gwasmind-open-targets-semantic"
SERVER_DESCRIPTION = (
    "Unofficial GWASMind semantic wrapper over Open Targets Platform; "
    "not an official Open Targets MCP server."
)


class SemanticMCPDispatcher:
    """Transport-neutral MCP tool listing and call dispatcher."""

    def __init__(self, service: SemanticToolService):
        self.service = service

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.value,
                "description": f"{SERVER_DESCRIPTION} {spec.purpose}",
                "inputSchema": dict(spec.input_schema),
                "outputSchema": dict(spec.output_schema),
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            }
            for tool, spec in TOOL_SPECS.items()
        ]

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        try:
            tool = SemanticTool(name)
        except ValueError as exc:
            raise ValueError(f"unknown GWASMind semantic tool: {name}") from exc
        value = {} if arguments is None else arguments
        if not isinstance(value, Mapping):
            raise TypeError("MCP tool arguments must be an object")
        envelope = await self.service.call(tool, value)
        return envelope.to_dict()
