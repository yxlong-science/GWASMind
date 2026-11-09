"""Represent a failed live-service bootstrap without inventing capabilities."""
from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping
from itertools import islice
from typing import Any
from urllib.error import HTTPError, URLError
from uuid import uuid4

from ..audit import sha256_json
from ..progress import ProgressSink, emit_progress
from .contracts import SemanticTool, ToolCoverage, ToolError, ToolStatus
from .service import SemanticToolService

SCHEMA_UNAVAILABLE_MESSAGE = (
    "The external evidence service could not be initialized. This report is limited "
    "to the uploaded GWAS data; no external evidence was retrieved."
)

# Fixed diagnostic schema/allowlists, not request or scientific tuning values.
_DIAGNOSTIC_CHAIN_LIMIT = 16
_NETWORK_EXCEPTION_TYPES = frozenset({
    "TransportError", "RequestError", "ConnectError", "NetworkError", "ReadError",
    "WriteError", "CloseError", "TimeoutException", "ConnectTimeout", "ReadTimeout",
    "WriteTimeout", "PoolTimeout", "ProxyError", "ProtocolError", "RemoteProtocolError",
    "LocalProtocolError", "UnsupportedProtocol", "DecodingError",
})
_LOCAL_EXCEPTION_TYPES = {
    "builtins": frozenset({"TimeoutError", "ConnectionError", "ConnectionAbortedError",
        "ConnectionRefusedError", "ConnectionResetError", "BrokenPipeError", "OSError",
        "ValueError", "RuntimeError", "Exception", "ExceptionGroup", "BaseExceptionGroup"}),
    "ssl": frozenset({"SSLError", "SSLEOFError", "SSLZeroReturnError", "SSLCertVerificationError",
        "SSLWantReadError", "SSLWantWriteError"}),
    "socket": frozenset({"gaierror", "herror"}),
    "urllib.error": frozenset({"HTTPError", "URLError"}),
    "http.client": frozenset({"IncompleteRead", "BadStatusLine", "RemoteDisconnected",
        "HTTPException", "LineTooLong"}),
    "gwasmind.providers.live": frozenset({"LiveProviderError", "_RetryableGraphQLTransportError"}),
}
_SSL_DIAGNOSTIC_TOKENS = {
    "UNEXPECTED_EOF_WHILE_READING": "tls_unexpected_eof",
    "EOF occurred in violation of protocol": "tls_unexpected_eof",
    "CERTIFICATE_VERIFY_FAILED": "tls_certificate_verification_failed",
    "WRONG_VERSION_NUMBER": "tls_wrong_version",
    "TLSV1_ALERT": "tls_alert",
    "HANDSHAKE_FAILURE": "tls_handshake_failure",
}


def bootstrap_failure_diagnostic(error: BaseException, *, network_backend: str) -> dict[str, Any]:
    """Return bounded private diagnostics without retaining arbitrary exception text.

    Transport wrappers keep their underlying exception in a cause, context or
    URLError.reason. Inspect these relationships, but emit only fixed class names,
    known TLS categories, timeout categories and HTTP status codes. This payload
    is separate from the source-only admission sidecar and is never evidence.
    """
    rows: list[dict[str, Any]] = []
    pending = deque([("raised", error)])
    seen: set[int] = set()
    truncated = False
    while pending and len(rows) < _DIAGNOSTIC_CHAIN_LIMIT:
        relation, exc = pending.popleft()
        if id(exc) in seen:
            continue
        seen.add(id(exc))
        name, module = type(exc).__name__, type(exc).__module__
        library = module.split(".", 1)[0]
        network = library in {"httpx", "httpx2", "httpcore", "httpcore2"} and name in _NETWORK_EXCEPTION_TYPES
        local = name in _LOCAL_EXCEPTION_TYPES.get(module, ())
        row: dict[str, Any] = {"relation": relation, "exception_type": name if network or local else "other_exception"}
        if network or local:
            row["exception_module"] = library if network else module
        if network or (local and module in {"ssl", "socket", "http.client", "builtins"}):
            if "timeout" in name.lower():
                row["timeout_category"] = {
                    "ConnectTimeout": "connect", "ReadTimeout": "read",
                    "WriteTimeout": "write", "PoolTimeout": "pool",
                }.get(name, "unspecified")
        if network or (local and module == "ssl"):
            message = str(exc)[:4096].lower()
            codes = sorted({code for token, code in _SSL_DIAGNOSTIC_TOKENS.items() if token.lower() in message})
            if codes:
                row["ssl_categories"] = codes
        if isinstance(exc, HTTPError) and isinstance(exc.code, int) and not isinstance(exc.code, bool) and 100 <= exc.code <= 599:
            row["http_status"] = exc.code
        elif module == "gwasmind.providers.live" and local:
            match = re.match(r"^official GraphQL HTTP request returned ([1-5]\d{2})(?:$|:)", str(exc)[:80])
            if match:
                row["http_status"] = int(match.group(1))
        rows.append(row)
        children = [("cause", exc.__cause__), ("context", exc.__context__)]
        if isinstance(exc, URLError):
            children.append(("reason", exc.reason))
        if isinstance(exc, BaseExceptionGroup):
            children.extend(("group_child", item) for item in islice(exc.exceptions, _DIAGNOSTIC_CHAIN_LIMIT))
            truncated = truncated or len(exc.exceptions) > _DIAGNOSTIC_CHAIN_LIMIT
        pending.extend((kind, item) for kind, item in children if isinstance(item, BaseException) and id(item) not in seen)
    return {"schema_version": 1, "phase": "schema_discovery", "status": "unavailable",
            "network_backend": network_backend if network_backend in {"httpx2", "urllib", "mcp"} else "unknown",
            "exception_chain": rows, "chain_truncated": truncated or any(id(item) not in seen for _, item in pending)}


class UnavailableSemanticService(SemanticToolService):
    """No ready modules or schema; only the observed bootstrap failure is known."""

    def __init__(self, *, requested_release: str, policy_sha256: str,
                 error_type: str, progress_sink: ProgressSink | None = None):
        super().__init__(provider="official_graphql", data_release=requested_release,
            schema_sha256="unknown", policy_sha256=policy_sha256,
            upstream_metadata={"schema_bootstrap": {"status": "unavailable",
                "error_type": error_type, "requested_release": requested_release,
                "observed_release": None, "schema_verified": False}})
        self.progress_sink = progress_sink

    async def call(self, tool: SemanticTool, arguments: Mapping[str, Any]):
        if tool is not SemanticTool.GET_CAPABILITIES:
            return await super().call(tool, arguments)
        coverage = ToolCoverage(requested=("schema.discovery",), unavailable=("schema.discovery",))
        emit_progress(self.progress_sink, "semantic_tool_started", tool=tool.value,
                      modules=["schema.discovery"], required=True, progress_scope="workflow_step")
        result = self._envelope(tool=tool, call_id=str(uuid4()), status=ToolStatus.UNAVAILABLE,
            coverage=coverage, input_sha256=sha256_json(arguments),
            warnings=(SCHEMA_UNAVAILABLE_MESSAGE,),
            error=ToolError("schema_discovery_unavailable", SCHEMA_UNAVAILABLE_MESSAGE,
                            retryable=False, module="schema.discovery"))
        emit_progress(self.progress_sink, "semantic_tool_finished", tool=tool.value,
                      progress_scope="workflow_step",
                      status=result.status.value, coverage=coverage.to_dict(), records=0,
                      module_counts={}, warnings=list(result.warnings))
        return result
