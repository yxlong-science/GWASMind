from __future__ import annotations

import json
from typing import Any

from .. import __version__
from .dispatcher import SERVER_DESCRIPTION, SERVER_NAME, SemanticMCPDispatcher


SERVER_VERSION = __version__


def create_sdk_server(dispatcher: SemanticMCPDispatcher):
    """Create the low-level MCP SDK server with exact catalogued JSON schemas."""

    try:
        import mcp.types as types
        from mcp.server.lowlevel import Server
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('Install MCP server support with: pip install -e ".[mcp]"') from exc

    async def list_tools():
        values = []
        for item in dispatcher.list_tools():
            kwargs: dict[str, Any] = {
                "name": item["name"],
                "description": item["description"],
                "inputSchema": item["inputSchema"],
                "outputSchema": item["outputSchema"],
            }
            annotation_type = getattr(types, "ToolAnnotations", None)
            if annotation_type is not None:
                kwargs["annotations"] = annotation_type(**item["annotations"])
            try:
                values.append(types.Tool(**kwargs))
            except TypeError as exc:
                raise RuntimeError(
                    "installed MCP SDK does not support the required exact tool contract"
                ) from exc
        return values

    async def call_tool(name: str, arguments: dict[str, Any]):
        return await dispatcher.call_tool(name, arguments)

    if hasattr(Server, "list_tools") and hasattr(Server, "call_tool"):
        server = Server(SERVER_NAME)
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
        return server

    # MCP 2.x uses constructor handlers and explicit structured result objects.
    # The semantic envelopes and wire schemas remain identical to the 1.x API.
    async def handle_list_tools(_context, _parameters):
        return types.ListToolsResult(tools=await list_tools())

    async def handle_call_tool(_context, parameters):
        value = await call_tool(parameters.name, parameters.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(value, ensure_ascii=False))],
            structuredContent=value,
            isError=value.get("status") in {"error", "invalid_input", "blocked", "unavailable"},
        )

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
    )


async def run_stdio(dispatcher: SemanticMCPDispatcher) -> None:
    """Run the semantic tools over MCP stdio without writing protocol noise."""

    try:
        import mcp.server.stdio
        from mcp.server.lowlevel import NotificationOptions
        from mcp.server.models import InitializationOptions
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('Install MCP server support with: pip install -e ".[mcp]"') from exc

    server = create_sdk_server(dispatcher)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={
                        "gwasmind": {
                            "identity": SERVER_DESCRIPTION,
                            "semantic_schema": "gwasmind-ot-tools/1",
                        }
                    },
                ),
            ),
        )
