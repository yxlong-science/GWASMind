from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import re
import threading
import weakref
from collections.abc import Iterable, Mapping
from collections import OrderedDict
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import date
from http.client import HTTPException
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from ..audit import sha256_json
from ..configuration import parameter
from ..models import DataMode, EvidenceKind, EvidenceRecord, LocusQuery
from ..qtl import native_qtl_study_type, qtl_study_type_from_payload
from .base import OpenTargetsProvider, ProviderCapabilities, ensure_supported
from .retrieval_recovery import (
    ENTITY_RECOVERY_ACTIVE, failure_category, recovery_progress, retry_after_seconds, safe_transport_failure,
)
from .query_cache import ACTIVE_REQUEST_KEY, RequestPageStore

OFFICIAL_OPEN_TARGETS_MCP_ENDPOINT = "https://mcp.platform.opentargets.org/mcp"
OFFICIAL_OPEN_TARGETS_GRAPHQL_ENDPOINT = (
    "https://api.platform.opentargets.org/api/v4/graphql"
)
MCP_SOURCE = "Open Targets Platform (official MCP)"
GRAPHQL_SOURCE = "Open Targets Platform (official GraphQL API)"
# Kept as a compatibility alias for callers that imported the original constant.
SOURCE = MCP_SOURCE

_DEFAULT_GRAPHQL_TIMEOUT_SECONDS = parameter("transport.graphql_timeout_seconds")
_DEFAULT_GRAPHQL_MAX_RESPONSE_BYTES = parameter("transport.graphql_max_response_bytes")
_DEFAULT_GRAPHQL_MAX_OPERATION_BYTES = parameter("transport.graphql_max_operation_bytes")
_DEFAULT_GRAPHQL_MAX_REQUEST_BYTES = parameter("transport.graphql_max_request_bytes")
_DEFAULT_GRAPHQL_MAX_BATCH_SIZE = parameter("retrieval.graphql_batch_size")
_EXPRESSION_PAGE_SIZE = parameter("retrieval.expression_page_size")
_EXPRESSION_MAX_ROWS = parameter("retrieval.expression_max_rows")
_EXPRESSION_ROW_FIELDS = (
    "median min q1 q3 max unit datasourceId datatypeId "
    "tissueBiosample { biosampleId biosampleName } "
    "celltypeBiosample { biosampleId biosampleName }"
)
_DEFAULT_GRAPHQL_MAX_SEARCH_QUERIES = parameter("retrieval.graphql_max_search_queries")
_DEFAULT_GRAPHQL_MAX_QUERY_CHARS = parameter("transport.graphql_max_query_chars")

_SCHEMA_INTROSPECTION_QUERY = """
query GWASMindSchema {
  __schema {
    queryType { name }
    types {
      kind
      name
      fields(includeDeprecated: true) { name }
      inputFields { name }
      enumValues(includeDeprecated: true) { name }
      interfaces { name }
      possibleTypes { name }
    }
  }
}
"""

_SEARCH_ENTITIES_QUERY = """
query GWASMindSearchEntity($queryString: String!) {
  search(queryString: $queryString) {
    hits { id name entity }
  }
}
"""

SUPPORTED_KINDS = frozenset(
    {
        EvidenceKind.GWAS,
        EvidenceKind.QTL,
        EvidenceKind.COLOCALIZATION,
        EvidenceKind.REGULATORY,
        EvidenceKind.EXPRESSION,
        EvidenceKind.FUNCTION,
        EvidenceKind.LITERATURE,
        EvidenceKind.L2G,
    }
)
_CREDIBLE_SET_KINDS = frozenset(
    {
        EvidenceKind.GWAS,
        EvidenceKind.QTL,
        EvidenceKind.COLOCALIZATION,
        EvidenceKind.L2G,
    }
)
# The 26.06 API accepts this full nested selection at 100 rows, while a static
# 500-row request is rejected by the server's query-complexity guard.  Keep the
# upstream page size small, but paginate large, real loci (for example the
# 865-row T1D locus).  The shared continuation-request budget, rather than an
# arbitrary per-variant row cap, bounds total work. Connections that exhaust
# that budget are returned as explicitly truncated coverage; they are never
# misreported as complete or as evidence absence.
_CREDIBLE_SET_PAGE_SIZE = parameter("retrieval.credible_set_page_size")
_CREDIBLE_SET_CONTINUATION_BATCH_SIZE = parameter("retrieval.credible_set_continuation_batch_size")
_CREDIBLE_SET_MAX_CONTINUATION_REQUESTS = parameter("retrieval.credible_set_max_continuation_requests")
# ``Variant.enhancerToGenes`` is a paginated connection as well.  Its default
# page contains only 25 rows while ``count`` reports the full result.  Request a
# larger but still bounded page explicitly and continue in round-robin order so
# one enhancer-rich variant cannot monopolize the run budget.
_ENHANCER_PAGE_SIZE = parameter("retrieval.enhancer_page_size")
_ENHANCER_CONTINUATION_BATCH_SIZE = parameter("retrieval.enhancer_continuation_batch_size")
_ENHANCER_MAX_CONTINUATION_REQUESTS = parameter("retrieval.enhancer_max_continuation_requests")
_COLOCALISATION_ROW_FIELDS = (
    "rightStudyType chromosome colocalisationMethod numberColocalisingVariants h3 h4 clpp "
    "betaRatioSignAverage otherStudyLocus { studyLocusId studyId studyType qtlGeneId "
    "study { id studyType traitFromSource pubmedId publicationDate publicationTitle "
    "publicationFirstAuthor publicationJournal target { id approvedSymbol } } }"
)
_VARIANT_ID = re.compile(r"^(?:[0-9]+|X|Y|MT)_\d+_[ACGTN]+_[ACGTN]+$", re.IGNORECASE)
_TARGET_ID = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)


def _canonical_variant_location(variant_id: str) -> tuple[str, int] | None:
    if not _VARIANT_ID.fullmatch(variant_id):
        return None
    chromosome, position, _, _ = variant_id.split("_", 3)
    try:
        parsed_position = int(position)
    except ValueError:  # guarded by the canonical-ID regex
        return None
    return (chromosome, parsed_position) if parsed_position > 0 else None


class LiveProviderError(RuntimeError):
    """An official live endpoint failed or returned data outside its contract."""


@dataclass(frozen=True, slots=True)
class CredibleSetCoverage:
    """Auditable coverage for one variant's paginated credible-set connection."""

    variant_id: str
    declared_count: int | None
    retrieved_count: int
    failure_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return bool(self.failure_reason) or self.declared_count is None or self.retrieved_count < self.declared_count


@dataclass(frozen=True, slots=True)
class EnhancerCoverage:
    """Auditable coverage for one variant's paginated E2G connection."""

    variant_id: str
    declared_count: int
    retrieved_count: int

    @property
    def truncated(self) -> bool:
        return self.retrieved_count < self.declared_count


@dataclass(frozen=True, slots=True)
class ConnectionCoverage:
    """Coverage of a bounded nested connection, scoped to its evidence kind."""

    kind: EvidenceKind
    entity_id: str
    connection: str
    declared_count: int
    retrieved_count: int
    cursor: str | None = None
    failure_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return bool(self.failure_reason) or self.retrieved_count < self.declared_count


@dataclass(frozen=True, slots=True)
class LiveProviderFetchResult:
    """Evidence rows plus coverage metadata that cannot fit in EvidenceRecord."""

    records: tuple[EvidenceRecord, ...]
    credible_set_coverage: tuple[CredibleSetCoverage, ...] = ()
    enhancer_coverage: tuple[EnhancerCoverage, ...] = ()
    warnings: tuple[str, ...] = ()
    connection_coverage: tuple[ConnectionCoverage, ...] = ()

    @property
    def truncated(self) -> bool:
        return any(item.truncated for item in self.credible_set_coverage) or any(
            item.truncated for item in self.enhancer_coverage
        ) or any(item.truncated for item in self.connection_coverage)


class _RetryableGraphQLTransportError(LiveProviderError):
    """A transient network failure eligible for the single bounded retry."""

    def __init__(self, message: str, *, http_client: Any | None = None,
                 retry_after: float | None = None, http_status: int | None = None):
        super().__init__(message)
        self.http_client = http_client
        self.retry_after = retry_after
        self.failure_details = ({"category": "rate_limited" if http_status == 429 else "service_error",
                                 "http_status": http_status, "retryable": True} if http_status is not None else
                                {"category": "transport_error", "retryable": True})


def _load_httpx2() -> Any | None:
    """Load the optional pooled HTTP client without making it a hard dependency."""

    try:
        return importlib.import_module("httpx2")
    except ImportError:
        return None


def _close_http_client(client: Any) -> None:
    """Best-effort finalizer callback; explicit ``close`` uses the same path."""

    try:
        client.close()
    except Exception:  # noqa: BLE001 - finalizers cannot surface shutdown errors
        # Finalizers must never turn interpreter shutdown into a noisy traceback.
        return


def _read_bounded_chunks(chunks: Iterable[bytes], limit: int) -> bytes:
    """Read at most ``limit + 1`` bytes so callers can detect truncation."""

    result = bytearray()
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise LiveProviderError("official GraphQL returned a non-bytes HTTP chunk")
        remaining = limit + 1 - len(result)
        if remaining <= 0:
            break
        result.extend(chunk[:remaining])
        if len(result) > limit:
            break
    return bytes(result)


def _bounded_graphql_error_body(raw: bytes, *, limit: int) -> str | None:
    if len(raw) > limit:
        return "error body exceeded the diagnostic size limit"
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(value, Mapping) and value.get("errors"):
        return _graphql_error_summary(value["errors"])
    return None


class MCPToolTransport(Protocol):
    """Small injectable boundary around the generic MCP ``call_tool`` operation."""

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


class OfficialOpenTargetsMCPTransport:
    """Transport for the hosted official Open Targets streamable-HTTP MCP."""

    source = MCP_SOURCE
    backend_name = "official_mcp"
    transport_description = "official hosted MCP"

    def __init__(self, endpoint: str = OFFICIAL_OPEN_TARGETS_MCP_ENDPOINT):
        self.endpoint = endpoint
        self._session = None

    @asynccontextmanager
    async def session_scope(self):
        """Reuse one official MCP session for every tool call in a run."""

        if self._session is not None:
            yield self
            return
        try:
            from mcp import ClientSession
            try:
                from mcp.client.streamable_http import streamable_http_client
            except ImportError:
                from mcp.client.streamable_http import (
                    streamablehttp_client as streamable_http_client,
                )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError('Install live support with: pip install -e ".[live]"') from exc

        async with streamable_http_client(self.endpoint) as streams:
            read_stream, write_stream = streams[:2]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                self._session = session
                try:
                    yield self
                finally:
                    self._session = None

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if self._session is None:
            async with self.session_scope():
                return await self.call_tool(name, arguments)
        result = await self._session.call_tool(name, dict(arguments))
        return _decode_mcp_result(result, tool=name)


class OfficialOpenTargetsGraphQLTransport:
    """Bounded adapter from the generic MCP tool contract to the official GraphQL API.

    The adapter is intentionally small: it implements only the four generic operations
    used by GWASMind.  It uses the standard-library HTTP stack so ``HTTP_PROXY``,
    ``HTTPS_PROXY`` and ``NO_PROXY`` continue to work without another dependency.
    Batch and search requests are sequential to keep client resource use predictable.
    """

    source = GRAPHQL_SOURCE
    backend_name = "official_graphql"
    transport_description = "official GraphQL API"

    def __init__(
        self,
        endpoint: str = OFFICIAL_OPEN_TARGETS_GRAPHQL_ENDPOINT,
        *,
        timeout_seconds: float = _DEFAULT_GRAPHQL_TIMEOUT_SECONDS,
        max_response_bytes: int = _DEFAULT_GRAPHQL_MAX_RESPONSE_BYTES,
        max_operation_bytes: int = _DEFAULT_GRAPHQL_MAX_OPERATION_BYTES,
        max_request_bytes: int = _DEFAULT_GRAPHQL_MAX_REQUEST_BYTES,
        max_batch_size: int = _DEFAULT_GRAPHQL_MAX_BATCH_SIZE,
        max_search_queries: int = _DEFAULT_GRAPHQL_MAX_SEARCH_QUERIES,
        max_transport_retries: int = parameter("transport.graphql_max_retries"),
        retry_backoff_seconds: float = parameter("transport.graphql_retry_backoff_seconds"),
    ):
        self.endpoint = _validated_http_endpoint(endpoint)
        if not 0 < timeout_seconds <= 60:
            raise ValueError("GraphQL timeout_seconds must be in (0, 60]")
        if not 1024 <= max_response_bytes <= 128 * 1024 * 1024:
            raise ValueError("GraphQL max_response_bytes is outside the safe range")
        if max_operation_bytes < max_response_bytes:
            raise ValueError("GraphQL max_operation_bytes must cover one response")
        if not 1024 <= max_request_bytes <= 16 * 1024 * 1024:
            raise ValueError("GraphQL max_request_bytes is outside the safe range")
        if not 1 <= max_batch_size <= 100:
            raise ValueError("GraphQL max_batch_size must be in [1, 100]")
        if not 1 <= max_search_queries <= 200:
            raise ValueError("GraphQL max_search_queries must be in [1, 200]")
        if max_transport_retries not in {0, 1}:
            raise ValueError("GraphQL max_transport_retries must be 0 or 1")
        if not 0 <= retry_backoff_seconds <= 1:
            raise ValueError("GraphQL retry_backoff_seconds must be in [0, 1]")
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.max_operation_bytes = int(max_operation_bytes)
        self.max_request_bytes = int(max_request_bytes)
        self.max_batch_size = int(max_batch_size)
        self.max_search_queries = int(max_search_queries)
        self.max_transport_retries = int(max_transport_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.request_cache = None
        self._request_locks = {}
        # build_opener installs ProxyHandler with environment-derived proxy settings.
        self._opener = build_opener()
        self._http_client_lock = threading.Lock()
        self._closed = False
        self._httpx2 = _load_httpx2()
        self._http_client = self._create_http_client()
        self._http_client_finalizer = (
            weakref.finalize(self, _close_http_client, self._http_client)
            if self._http_client is not None
            else None
        )

    @property
    def network_backend(self) -> str:
        with self._http_client_lock:
            return "httpx2" if self._http_client is not None else "urllib"

    def _create_http_client(self) -> Any | None:
        if self._httpx2 is None:
            return None
        limits = self._httpx2.Limits(
            max_connections=1,
            max_keepalive_connections=1,
            keepalive_expiry=30.0,
        )
        return self._httpx2.Client(
            verify=True,
            trust_env=True,
            http1=True,
            http2=False,
            timeout=self.timeout_seconds,
            follow_redirects=False,
            limits=limits,
        )

    def close(self) -> None:
        """Close the optional pooled client; safe to call more than once."""

        with self._http_client_lock:
            self._closed = True
            finalizer = self._http_client_finalizer
            if finalizer is not None and finalizer.alive:
                finalizer()

    def _replace_failed_http_client(self, failed_client: Any) -> None:
        """Give the existing retry a fresh pool without reviving a closed transport."""

        with self._http_client_lock:
            if self._closed:
                raise LiveProviderError("official GraphQL transport is closed")
            if failed_client is not self._http_client:
                # Another failed request already replaced this pool. Do not
                # close its healthy replacement because of a stale exception.
                return
            finalizer = self._http_client_finalizer
            if finalizer is not None and finalizer.alive:
                finalizer()
            self._http_client = None
            self._http_client_finalizer = None
            try:
                client = self._create_http_client()
                if client is None:
                    raise RuntimeError("pooled HTTP client is unavailable")
            except Exception as exc:  # noqa: BLE001 - preserve a bounded, credential-free failure
                self._closed = True
                raise LiveProviderError(
                    "official GraphQL retry could not recreate its HTTP client "
                    f"({type(exc).__name__})"
                ) from None
            self._http_client = client
            self._http_client_finalizer = weakref.finalize(self, _close_http_client, client)

    @asynccontextmanager
    async def session_scope(self):
        """Match the MCP transport lifecycle; GraphQL itself is stateless."""

        yield self

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        normalized = _json_value(arguments, label=f"{name} arguments")
        if not isinstance(normalized, dict):
            raise LiveProviderError(f"{name} arguments must be a JSON object")
        if name == "get_open_targets_graphql_schema":
            return await self._schema(normalized)
        if name == "query_open_targets_graphql":
            return await self._query(normalized)
        if name == "batch_query_open_targets_graphql":
            return await self._batch_query(normalized)
        if name == "search_entities":
            return await self._search_entities(normalized)
        raise LiveProviderError(
            f"official GraphQL fallback does not implement operation {name!r}"
        )

    async def _schema(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _reject_unknown_arguments(
            arguments, allowed={"categories"}, label="get_open_targets_graphql_schema"
        )
        categories = arguments.get("categories", [])
        if not isinstance(categories, list) or any(
            not isinstance(item, str) or not item.strip() for item in categories
        ):
            raise LiveProviderError("schema categories must be a list of non-empty strings")
        data, _ = await self._execute_graphql(_SCHEMA_INTROSPECTION_QUERY, {})
        schema = data.get("__schema")
        if not isinstance(schema, dict):
            raise LiveProviderError("official GraphQL introspection omitted __schema")
        # The negotiator recursively collects names, so a name-only introspection is an
        # exact, current feature inventory without shipping a hard-coded SDL snapshot.
        return {"schema": schema}

    async def _query(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _reject_unknown_arguments(
            arguments,
            allowed={"query_string", "variables"},
            label="query_open_targets_graphql",
        )
        query = _graphql_query_string(arguments.get("query_string"))
        variables = _graphql_variables(arguments.get("variables", {}))
        data, _ = await self._execute_graphql(query, variables)
        return {"status": "success", "data": data}

    async def _batch_query(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _reject_unknown_arguments(
            arguments,
            allowed={"query_string", "variables_list", "key_field"},
            label="batch_query_open_targets_graphql",
        )
        query = _graphql_query_string(arguments.get("query_string"))
        raw_variables = arguments.get("variables_list")
        if not isinstance(raw_variables, list) or not raw_variables:
            raise LiveProviderError("GraphQL variables_list must be a non-empty list")
        if len(raw_variables) > self.max_batch_size:
            raise LiveProviderError(
                f"GraphQL batch exceeds the {self.max_batch_size}-request limit"
            )
        key_field = arguments.get("key_field")
        if not isinstance(key_field, str) or not key_field.strip():
            raise LiveProviderError("GraphQL batch key_field must be a non-empty string")

        results: list[dict[str, Any]] = []
        operation_bytes = 0
        for index, raw in enumerate(raw_variables):
            variables = _graphql_variables(raw)
            identifier = variables.get(key_field)
            if not isinstance(identifier, (str, int, float)) or isinstance(
                identifier, bool
            ):
                raise LiveProviderError(
                    f"GraphQL batch variables[{index}] is missing scalar {key_field!r}"
                )
            data, response_bytes = await self._execute_graphql(query, variables)
            operation_bytes += response_bytes
            if operation_bytes > self.max_operation_bytes:
                raise LiveProviderError("GraphQL batch exceeded its total response-size limit")
            results.append(
                {
                    "index": index,
                    "id": str(identifier),
                    "result": {"status": "success", "data": data},
                }
            )
        return {
            "results": results,
            "status_counts": {"success": len(results)},
        }

    async def _search_entities(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _reject_unknown_arguments(
            arguments, allowed={"query_strings"}, label="search_entities"
        )
        queries = arguments.get("query_strings")
        if not isinstance(queries, list) or not queries:
            raise LiveProviderError("search_entities query_strings must be non-empty")
        if len(queries) > self.max_search_queries:
            raise LiveProviderError(
                f"entity search exceeds the {self.max_search_queries}-query limit"
            )
        if any(
            not isinstance(value, str) or not value.strip() or len(value) > 1000
            for value in queries
        ):
            raise LiveProviderError(
                "search_entities queries must be non-empty strings up to 1000 characters"
            )

        result: dict[str, list[dict[str, str]]] = {}
        operation_bytes = 0
        for value in queries:
            data, response_bytes = await self._execute_graphql(
                _SEARCH_ENTITIES_QUERY, {"queryString": value}
            )
            operation_bytes += response_bytes
            if operation_bytes > self.max_operation_bytes:
                raise LiveProviderError("entity search exceeded its total response-size limit")
            search = data.get("search")
            if not isinstance(search, dict):
                raise LiveProviderError("official GraphQL entity search omitted search")
            hits = search.get("hits")
            if not isinstance(hits, list):
                raise LiveProviderError("official GraphQL entity search omitted hits")
            normalized_hits: list[dict[str, str]] = []
            # The response is already bounded by the transport byte limits.
            # Do not hide later candidates before semantic type/identity checks:
            # a fourth hit can be another allele of the requested rsID.
            for index, hit in enumerate(hits):
                if not isinstance(hit, dict):
                    raise LiveProviderError(f"entity search hit {index} must be an object")
                identifier = _required_text(hit.get("id"), label="entity search hit id")
                entity = _required_text(hit.get("entity"), label="entity search hit type")
                normalized_hit = {"id": identifier, "type": entity}
                name = _optional_text(hit.get("name"))
                if name is not None:
                    normalized_hit["name"] = name
                normalized_hits.append(normalized_hit)
            result[value] = normalized_hits
        return result

    async def _execute_graphql(
        self, query: str, variables: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        if self._closed:
            raise LiveProviderError("official GraphQL transport is closed")
        if self.request_cache is None:
            return await self._execute_graphql_uncached(query, variables)
        key = self.request_cache.key(self.endpoint, query, variables)
        lock_key = (id(asyncio.get_running_loop()), key)
        lock = self._request_locks.setdefault(lock_key, asyncio.Lock())
        async with lock:
            return await self._execute_graphql_cached(query, variables)

    async def _execute_graphql_cached(self, query, variables):
        from .retrieval_recovery import record_transport_outcome
        cache = self.request_cache
        if cache is None:
            return await self._execute_graphql_uncached(query, variables)
        key = cache.key(self.endpoint, query, variables)
        request = {"endpoint": self.endpoint, "release": cache.release, "query": query, "variables": variables}
        try:
            cached = await asyncio.to_thread(cache.get, key)
        except (ValueError, OSError, KeyError) as exc:
            raise LiveProviderError("Saved request page failed integrity validation") from exc
        if cached is not None:
            if cached["status"] == "failed":
                failed = LiveProviderError(cached["error"])
                failed.failure_details = cached.get("failure_details", {})
                recovery_key, recovery_request = cache.recovery_request(request, key)
                try:
                    recovered = await asyncio.to_thread(cache.get, recovery_key)
                    if (recovered is None and ENTITY_RECOVERY_ACTIVE.get()
                            and parameter("retrieval.entity_recovery_attempts") > 0
                            and safe_transport_failure(failed).get("retryable") is True):
                        reserved = await asyncio.to_thread(cache.reserve_recovery, recovery_key, recovery_request)
                        if reserved:
                            key, request, cached = recovery_key, recovery_request, None
                        else:
                            recovered = await asyncio.to_thread(cache.get, recovery_key)
                    if recovered is not None:
                        key, request, cached = recovery_key, recovery_request, recovered
                except (ValueError, OSError, KeyError) as exc:
                    raise LiveProviderError("Saved recovery page failed integrity validation") from exc
        if cached is not None:
            recovery_progress("transport", "cached", **cache.summary())
            if cached["status"] == "failed":
                error = LiveProviderError(cached["error"])
                error.failure_details = cached.get("failure_details", {})
                error.entity_recovery_exhausted = "recovery_of" in request
                record_transport_outcome(query, variables, error.failure_details)
                raise error
            record_transport_outcome(query, variables)
            return cached["data"], cached["response_bytes"]
        token = ACTIVE_REQUEST_KEY.set(key)
        try:
            data, response_bytes = await self._execute_graphql_uncached(query, variables)
            await asyncio.to_thread(cache.save, key, data=data, response_bytes=response_bytes, request=request,
                                   recovery_status="completed" if "recovery_of" in request else None)
            recovery_progress("transport", "completed", **cache.summary())
            return data, response_bytes
        except LiveProviderError as exc:
            cache.note_failure(exc)
            await asyncio.to_thread(cache.save, key, error=exc, request=request,
                                   recovery_status="failed" if "recovery_of" in request else None)
            recovery_progress("transport", "incomplete", reason=failure_category(exc), **cache.summary())
            raise
        finally:
            ACTIVE_REQUEST_KEY.reset(token)

    async def _execute_graphql_uncached(self, query, variables):
        from .retrieval_recovery import record_transport_outcome
        retries = min(self.max_transport_retries, parameter("retrieval.shared_request_recovery_attempts"))
        for attempt in range(retries + 1):
            if self._closed:
                raise LiveProviderError("official GraphQL transport is closed")
            try:
                result = await asyncio.to_thread(
                    self._execute_graphql_sync, query, variables
                )
                record_transport_outcome(query, variables)
                return result
            except _RetryableGraphQLTransportError as exc:
                if self.request_cache is not None:
                    self.request_cache.note_failure(exc)
                if attempt >= retries:
                    error = LiveProviderError(
                        "official GraphQL transport failed after "
                        f"{attempt + 1} bounded attempt(s)"
                    )
                    error.failure_details = {**safe_transport_failure(exc), "attempts": attempt + 1,
                                             "recovery_exhausted": True}
                    record_transport_outcome(query, variables, error.failure_details)
                    raise error from exc
                if exc.http_client is not None:
                    # A failed connection/pool may remain unusable (including
                    # PoolTimeout). Reusing it can waste the only allowed retry.
                    await asyncio.to_thread(self._replace_failed_http_client, exc.http_client)
                delay = max(self.retry_backoff_seconds * (2 ** attempt), exc.retry_after or 0)
                # A long server cooldown is retained as a failure, never retried
                # prematurely or allowed to suspend a semantic request indefinitely.
                if delay > self.timeout_seconds:
                    error = LiveProviderError("official GraphQL rate-limit cooldown exceeds this request budget")
                    error.failure_details = {**safe_transport_failure(exc), "retryable": False,
                                             "attempts": attempt + 1, "recovery_exhausted": True}
                    record_transport_outcome(query, variables, error.failure_details)
                    raise error from exc
                recovery_progress("transport", "retrying", reason=failure_category(exc),
                                  attempt=attempt + 2, wait_seconds=delay)
                if delay:
                    await asyncio.sleep(delay)
        raise AssertionError("unreachable GraphQL retry state")

    def _execute_graphql_sync(
        self, query: str, variables: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        body = json.dumps(
            {"query": query, "variables": variables},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(body) > self.max_request_bytes:
            raise LiveProviderError("GraphQL request exceeds the configured size limit")
        with self._http_client_lock:
            if self._closed:
                raise LiveProviderError("official GraphQL transport is closed")
            client = self._http_client
        if self.request_cache is not None:
            self.request_cache.note_network_request()
        if client is not None:
            raw = self._execute_httpx2_request(body, client=client)
        else:
            raw = self._execute_urllib_request(body)
        return self._decode_graphql_response(raw, allow_partial_aliases="query GWASMindSharedEntities(" in query)

    def _execute_httpx2_request(self, body: bytes, *, client: Any) -> bytes:
        transport_error = self._httpx2.TransportError
        try:
            with client.stream(
                "POST",
                self.endpoint,
                content=body,
                headers=self._request_headers(),
            ) as response:
                status = int(response.status_code)
                if status < 200 or status >= 300:
                    limit = min(64 * 1024, self.max_response_bytes)
                    raw = _read_bounded_chunks(response.iter_bytes(), limit)
                    detail = _bounded_graphql_error_body(raw, limit=limit)
                    suffix = f": {detail}" if detail else ""
                    message = f"official GraphQL HTTP request returned {status}{suffix}"
                    if status == 429 or status in {500, 502, 503, 504}:
                        raise _RetryableGraphQLTransportError(
                            message, retry_after=retry_after_seconds(response.headers), http_status=status)
                    error = LiveProviderError(message)
                    error.failure_details = {"category": "http_client_error" if status < 500 else "service_error",
                                             "http_status": status, "retryable": False}
                    raise error
                self._validate_content_length(response.headers)
                raw = _read_bounded_chunks(
                    response.iter_bytes(), self.max_response_bytes
                )
        except transport_error as exc:
            raise _RetryableGraphQLTransportError(
                f"official GraphQL request failed ({type(exc).__name__})",
                http_client=client,
            ) from exc
        if len(raw) > self.max_response_bytes:
            raise LiveProviderError("official GraphQL response exceeds the configured size limit")
        return raw

    def _execute_urllib_request(self, body: bytes) -> bytes:
        request = Request(
            self.endpoint,
            data=body,
            headers=self._request_headers(),
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                self._validate_content_length(response.headers)
                raw = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            detail = _bounded_graphql_http_error(exc, limit=64 * 1024)
            suffix = f": {detail}" if detail else ""
            message = f"official GraphQL HTTP request returned {exc.code}{suffix}"
            if exc.code == 429 or exc.code in {500, 502, 503, 504}:
                raise _RetryableGraphQLTransportError(
                    message, retry_after=retry_after_seconds(exc.headers), http_status=exc.code) from exc
            raise LiveProviderError(message) from exc
        except (URLError, TimeoutError, OSError, HTTPException) as exc:
            raise _RetryableGraphQLTransportError(
                f"official GraphQL request failed ({type(exc).__name__})"
            ) from exc
        if len(raw) > self.max_response_bytes:
            raise LiveProviderError("official GraphQL response exceeds the configured size limit")
        return raw

    @staticmethod
    def _request_headers() -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "GWASMind/1.0 official-graphql-fallback",
        }

    def _validate_content_length(self, headers: Any) -> None:
        content_length = headers.get("Content-Length")
        if content_length is None:
            return
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError) as exc:
            raise LiveProviderError(
                "official GraphQL returned an invalid Content-Length"
            ) from exc
        if declared_length < 0:
            raise LiveProviderError(
                "official GraphQL returned an invalid Content-Length"
            )
        if declared_length > self.max_response_bytes:
            raise LiveProviderError(
                "official GraphQL response exceeds the configured size limit"
            )

    @staticmethod
    def _decode_graphql_response(raw: bytes, *, allow_partial_aliases: bool = False) -> tuple[dict[str, Any], int]:
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiveProviderError("official GraphQL returned malformed JSON") from exc
        normalized = _json_value(decoded, label="official GraphQL response")
        if not isinstance(normalized, dict):
            raise LiveProviderError("official GraphQL response must be a JSON object")
        errors = normalized.get("errors")
        data = normalized.get("data")
        if errors:
            # Only the shared-entity reader understands this explicit partial
            # contract. A located alias error cannot taint a different alias;
            # global/pathless errors still reject the entire response.
            if allow_partial_aliases and isinstance(errors, list) and isinstance(data, dict):
                located = {}
                for error in errors:
                    path = error.get("path") if isinstance(error, dict) else None
                    if (not isinstance(path, list) or not path or not isinstance(path[0], str)
                            or re.fullmatch(r"entity\d+", path[0]) is None or path[0] not in data):
                        break
                    located.setdefault(path[0], []).append(error)
                else:
                    if any(key not in located and value is not None for key, value in data.items()):
                        return {**data, "__gwasmind_alias_errors": located}, len(raw)
            raise LiveProviderError(
                "official GraphQL returned errors: " + _graphql_error_summary(errors)
            )
        if not isinstance(data, dict):
            raise LiveProviderError("official GraphQL response is missing its data object")
        return data, len(raw)


class OfficialOpenTargetsMCPFirstTransport:
    """Select hosted MCP once per run, falling back only on session-start failure.

    Once MCP has initialized, any later error fails closed.  This prevents a single
    result from silently mixing MCP and GraphQL provenance.  A failed MCP handshake
    disables further MCP attempts for this transport instance, avoiding repeated TLS
    delays on networks where only the official GraphQL endpoint is reachable.
    """

    def __init__(
        self,
        primary: MCPToolTransport | None = None,
        fallback: MCPToolTransport | None = None,
    ):
        self.primary = primary or OfficialOpenTargetsMCPTransport()
        self.fallback = fallback or OfficialOpenTargetsGraphQLTransport()
        self._active: MCPToolTransport | None = None
        self._last: MCPToolTransport | None = None
        self._primary_unavailable = False
        self.fallback_reason: str | None = None

    @property
    def endpoint(self) -> str | None:
        selected = self._active or self._last or self.primary
        value = getattr(selected, "endpoint", None)
        return str(value) if value is not None else None

    @property
    def source(self) -> str:
        selected = self._active or self._last or self.primary
        return str(getattr(selected, "source", MCP_SOURCE))

    @property
    def backend_name(self) -> str:
        selected = self._active or self._last or self.primary
        return str(getattr(selected, "backend_name", "official_mcp"))

    @property
    def transport_description(self) -> str:
        selected = self._active or self._last or self.primary
        return str(
            getattr(selected, "transport_description", "official hosted MCP")
        )

    @asynccontextmanager
    async def session_scope(self):
        if self._active is not None:
            yield self
            return

        if not self._primary_unavailable:
            stack = AsyncExitStack()
            try:
                await stack.enter_async_context(_transport_session(self.primary))
            except asyncio.CancelledError:
                await stack.aclose()
                raise
            except Exception as exc:  # noqa: BLE001 - MCP may raise ExceptionGroup
                await stack.aclose()
                self._primary_unavailable = True
                self.fallback_reason = type(exc).__name__
            else:
                self._activate(self.primary)
                try:
                    async with stack:
                        yield self
                finally:
                    self._active = None
                return

        async with _transport_session(self.fallback):
            self._activate(self.fallback)
            try:
                yield self
            finally:
                self._active = None

    def _activate(self, transport: MCPToolTransport) -> None:
        self._active = transport
        self._last = transport

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if self._active is None:
            async with self.session_scope():
                return await self.call_tool(name, arguments)
        return await self._active.call_tool(name, arguments)

    def close(self) -> None:
        """Close any pooled child transport without assuming one is active."""

        for transport in (self.primary, self.fallback):
            close = getattr(transport, "close", None)
            if callable(close):
                close()


class OfficialOpenTargetsMCPProvider(OpenTargetsProvider):
    """Normalize an official live Open Targets transport into ``EvidenceRecord`` values.

    This provider deliberately has no benchmark override. Its ``LIVE`` capability is
    the interface signal used by ``GWASToolGateway`` to reject benchmark access before
    any transport call is made. The lightweight default talks directly to the official
    GraphQL API. Callers that specifically want hosted MCP can inject
    ``OfficialOpenTargetsMCPTransport``; ``OfficialOpenTargetsMCPFirstTransport`` is
    available when MCP-first failover is desired.
    """

    def __init__(
        self,
        transport: MCPToolTransport | None = None,
        *,
        release: str = "live",
        cache_directory=None,
    ):
        self.transport = transport or OfficialOpenTargetsGraphQLTransport()
        self.request_cache = RequestPageStore(cache_directory, release=release)
        self._entity_loader = None
        if isinstance(self.transport, OfficialOpenTargetsGraphQLTransport):
            self.transport.request_cache = self.request_cache
        fallback = getattr(self.transport, "fallback", None)
        if isinstance(fallback, OfficialOpenTargetsGraphQLTransport):
            fallback.request_cache = self.request_cache
        self._colocalisation_cache = OrderedDict()
        # Query-capacity recovery is local to this provider/task and query shape.
        # Expression has a much larger nested selection than other target profiles.
        self._expression_request_limits: dict[tuple[str, str], dict[str, int]] = {}
        self._capabilities = ProviderCapabilities(
            mode=DataMode.LIVE,
            release=release,
            evidence_kinds=SUPPORTED_KINDS,
            snapshot_hash="live:official-open-targets-api",
            supports_l2g=True,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def endpoint(self) -> str | None:
        value = getattr(self.transport, "endpoint", None)
        return str(value) if value is not None else None

    @property
    def source(self) -> str:
        return str(getattr(self.transport, "source", MCP_SOURCE))

    @property
    def backend_name(self) -> str:
        return str(getattr(self.transport, "backend_name", "official_mcp"))

    @property
    def transport_description(self) -> str:
        return str(
            getattr(self.transport, "transport_description", "official hosted MCP")
        )

    def close(self) -> None:
        """Release a pooled live transport; safe when the transport is stateless."""

        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    async def fetch(
        self, query: LocusQuery, kinds: set[EvidenceKind]
    ) -> list[EvidenceRecord]:
        result = await self.fetch_with_coverage(query, kinds)
        if result.truncated:
            # The legacy provider interface has nowhere to carry coverage or
            # warnings. Failing closed here prevents a caller from silently
            # treating retained rows as a complete connection; the semantic
            # wrapper uses fetch_with_coverage and can preserve the partial rows.
            raise LiveProviderError(
                "credibleSets coverage was bounded; use the semantic live provider "
                "path so retained rows and truncation warnings remain auditable"
            )
        return list(result.records)

    async def fetch_with_coverage(
        self, query: LocusQuery, kinds: set[EvidenceKind]
    ) -> LiveProviderFetchResult:
        """Fetch evidence while retaining explicit live-connection coverage.

        The base provider contract intentionally returns evidence rows only.  The
        semantic wrapper needs the additional coverage object so a bounded live
        connection is reported as partial rather than being mistaken for either
        complete coverage or no evidence.
        """

        session_scope = getattr(self.transport, "session_scope", None)
        if callable(session_scope):
            async with session_scope():
                return await self._fetch_in_session(query, kinds)
        return await self._fetch_in_session(query, kinds)

    async def _fetch_in_session(
        self, query: LocusQuery, kinds: set[EvidenceKind]
    ) -> LiveProviderFetchResult:
        query.validate()
        ensure_supported(self, kinds)
        if not kinds:
            return LiveProviderFetchResult(())

        variant_kinds = {
            EvidenceKind.GWAS,
            EvidenceKind.QTL,
            EvidenceKind.COLOCALIZATION,
            EvidenceKind.REGULATORY,
            EvidenceKind.L2G,
        }
        target_kinds = {
            EvidenceKind.EXPRESSION,
            EvidenceKind.FUNCTION,
            EvidenceKind.LITERATURE,
        }

        variant_ids: list[str] = []
        if kinds.intersection(variant_kinds):
            variant_ids = await self._resolve_entities(
                [item.variant_id for item in query.variants], entity_type="variant"
            )

        target_ids: list[str] = []
        if kinds.intersection(target_kinds) and query.candidate_genes:
            target_ids = await self._resolve_entities(
                list(query.candidate_genes), entity_type="target"
            )

        evidence: list[EvidenceRecord] = []
        credible_set_coverage: tuple[CredibleSetCoverage, ...] = ()
        enhancer_coverage: tuple[EnhancerCoverage, ...] = ()
        connection_coverage: list[ConnectionCoverage] = []
        if variant_ids:
            variant_result = await self._query_variants(variant_ids, kinds)
            if (
                isinstance(variant_result, tuple)
                and len(variant_result) == 3
                and isinstance(variant_result[1], tuple)
                and isinstance(variant_result[2], tuple)
                and all(
                    isinstance(item, CredibleSetCoverage)
                    for item in variant_result[1]
                )
                and all(
                    isinstance(item, EnhancerCoverage)
                    for item in variant_result[2]
                )
            ):
                variants, credible_set_coverage, enhancer_coverage = variant_result
            elif (
                isinstance(variant_result, tuple)
                and len(variant_result) == 2
                and isinstance(variant_result[1], tuple)
                and all(
                    isinstance(item, CredibleSetCoverage)
                    for item in variant_result[1]
                )
            ):
                # Compatibility for tests or integrations that patch the old
                # two-item private helper contract.
                variants, credible_set_coverage = variant_result
            else:  # pragma: no cover - compatibility for a patched private helper
                variants = variant_result
            for result in variants:
                if result is not None:
                    if kinds.intersection({EvidenceKind.L2G, EvidenceKind.COLOCALIZATION}):
                        for credible in _connection_rows(result, "credibleSets"):
                            study = credible.get("study") or {}
                            for kind, field in (
                                (EvidenceKind.L2G, "l2GPredictions"),
                                (EvidenceKind.COLOCALIZATION, "colocalisation"),
                            ):
                                if kind not in kinds:
                                    continue
                                if kind is EvidenceKind.COLOCALIZATION:
                                    qtl_type = native_qtl_study_type(
                                        credible.get("studyType") or (
                                            study.get("studyType") if isinstance(study, dict) else None
                                        )
                                    )
                                    if qtl_type is None or (
                                        query.qtl_types and qtl_type not in query.qtl_types
                                    ):
                                        continue
                                connection_coverage.append(_bounded_connection_coverage(
                                    credible, field, kind,
                                    _required_text(credible.get("studyLocusId"), label="studyLocusId"),
                                ))
                    try:
                        evidence.extend(self._normalize_variant(result, kinds))
                    except LiveProviderError:
                        if self._entity_loader is None:
                            raise
                        # Apply the same identity/scientific normalizer to each
                        # shared object and channel. A rejected object must not
                        # erase unrelated, already validated evidence.
                        for kind in kinds:
                            if kind in _CREDIBLE_SET_KINDS:
                                views = [
                                    {"id": result["id"], "credibleSets": {"count": 1, "rows": [credible]}}
                                    for credible in _connection_rows(result, "credibleSets")
                                ]
                            else:
                                views = [result]
                            for view in views:
                                try:
                                    evidence.extend(self._normalize_variant(view, {kind}))
                                except LiveProviderError:
                                    connection_coverage.append(ConnectionCoverage(
                                        kind, result["id"], "validated_entity_evidence", 0, 0,
                                        failure_reason="invalid_entity_evidence",
                                    ))

        if target_ids:
            targets = (await self._query_target_profiles(target_ids, kinds, trait_ids=query.trait_ids)
                       if query.trait_ids else await self._query_target_profiles(target_ids, kinds))
            for result in targets:
                if result is not None:
                    if EvidenceKind.EXPRESSION in kinds:
                        value = (result.get("baselineExpression") or {}).get("coverage") or {}
                        if value.get("complete") is False:
                            connection_coverage.append(ConnectionCoverage(
                                EvidenceKind.EXPRESSION, result["id"], "baselineExpression",
                                value.get("declared_count") or 0, value.get("retrieved_count", 0),
                                failure_reason=value.get("reason", "incomplete_expression"),
                            ))
                    if EvidenceKind.LITERATURE in kinds:
                        connections = result.get('_literature_connections') or [
                            {'trait_ids': [], 'connection': result['literatureOcurrences']}]
                        for item in connections:
                            owner = _required_text(result.get("id"), label="target.id")
                            if item['trait_ids']:
                                owner += ':' + item['trait_ids'][0]
                            connection_coverage.append(_bounded_connection_coverage(
                                {'literatureOcurrences': item['connection']}, "literatureOcurrences", EvidenceKind.LITERATURE, owner))
                    evidence.extend(self._normalize_target(result, kinds))

        deduplicated: dict[str, EvidenceRecord] = {}
        for record in evidence:
            if (
                record.kind is EvidenceKind.GWAS
                and query.study_ids
                and record.study_id not in query.study_ids
            ):
                continue
            if record.kind in {
                EvidenceKind.QTL,
                EvidenceKind.COLOCALIZATION,
            }:
                qtl_type = qtl_study_type_from_payload(record.payload)
                if qtl_type is None or (
                    query.qtl_types and qtl_type not in query.qtl_types
                ):
                    continue
            deduplicated.setdefault(record.evidence_id, record)
        warnings = tuple(
            (
                f"Open Targets credible-set discovery is incomplete for {item.variant_id} "
                f"({item.failure_reason or 'unknown_coverage'}); retained {item.retrieved_count} memberships. "
                "Missing memberships are not evidence of absence."
                if item.declared_count is None else
                "Open Targets declared "
                f"{item.declared_count} credible-set rows for {item.variant_id}; "
                f"GWASMind retained {item.retrieved_count} rows in deterministic "
                "upstream page order under the bounded live-query budget. Coverage "
                "is incomplete, and an absent record in the retained subset is not "
                "evidence of absence."
            )
            for item in credible_set_coverage
            if item.truncated
        ) + tuple(
            (
                "Open Targets declared "
                f"{item.declared_count} enhancer-to-gene rows for {item.variant_id}; "
                f"GWASMind retained {item.retrieved_count} rows in deterministic "
                "upstream page order under the bounded live-query budget. Coverage "
                "is incomplete, and an absent regulatory link in the retained subset "
                "is not evidence of absence."
            )
            for item in enhancer_coverage
            if item.truncated
        ) + tuple(
            (f"Open Targets {item.connection} retrieval for {item.entity_id} remains incomplete "
             f"({item.failure_reason}); " +
             (f"declared {item.declared_count} rows; " if item.declared_count else "") +
             f"retained {item.retrieved_count} rows. " if item.failure_reason else
             f"Open Targets declared {item.declared_count} {item.connection} rows for "
             f"{item.entity_id}; the bounded response retained {item.retrieved_count}. ") +
            "Coverage is incomplete; absent rows are not evidence of absence."
            for item in connection_coverage
            if item.truncated
        )
        return LiveProviderFetchResult(
            records=tuple(deduplicated.values()),
            credible_set_coverage=credible_set_coverage,
            enhancer_coverage=enhancer_coverage,
            warnings=warnings,
            connection_coverage=tuple(connection_coverage),
        )

    async def _call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            value = await self.transport.call_tool(name, arguments)
        except LiveProviderError:
            raise
        except Exception as exc:
            raise LiveProviderError(
                f"official Open Targets {self.backend_name} operation {name!r} failed: {exc}"
            ) from exc
        normalized = _json_value(value, label=f"{name} response")
        if not isinstance(normalized, dict):
            raise LiveProviderError(f"{name} response must be a JSON object")
        return normalized

    async def _resolve_entities(self, values: list[str], *, entity_type: str) -> list[str]:
        resolved_by_input: dict[str, str] = {}
        unresolved: list[str] = []
        for value in values:
            direct = _direct_entity_id(value, entity_type=entity_type)
            if direct is None:
                unresolved.append(value)
            else:
                resolved_by_input[value] = direct

        if unresolved:
            response = await self._call_tool(
                "search_entities", {"query_strings": unresolved}
            )
            response = _unwrap_optional_success(response, label="search_entities")
            for value in unresolved:
                hits = _mapping_value_case_insensitive(response, value)
                if hits is _MISSING:
                    raise LiveProviderError(
                        f"search_entities omitted requested query {value!r}"
                    )
                if not isinstance(hits, list):
                    raise LiveProviderError(
                        f"search_entities results for {value!r} must be a list"
                    )
                accepted_hits: list[tuple[str, dict[str, Any]]] = []
                for index, hit in enumerate(hits):
                    if not isinstance(hit, dict):
                        raise LiveProviderError(
                            f"search_entities hit {value!r}[{index}] must be an object"
                        )
                    hit_type = _optional_text(hit.get("type"))
                    if hit_type is None:
                        raise LiveProviderError("search_entities hit is missing type")
                    normalized_type = hit_type.rsplit(".", 1)[-1].casefold()
                    accepted = {entity_type}
                    if entity_type == "target":
                        accepted.add("gene")
                    if normalized_type not in accepted:
                        continue
                    entity_id = _required_text(
                        hit.get("id"), label=f"search_entities {value!r} id"
                    )
                    if all(existing_id != entity_id for existing_id, _ in accepted_hits):
                        accepted_hits.append((entity_id, hit))
                exact_hits = [
                    entity_id
                    for entity_id, hit in accepted_hits
                    if _search_hit_matches_query(value, hit)
                ]
                selected_hits = exact_hits or [
                    entity_id for entity_id, _ in accepted_hits
                ]
                if len(selected_hits) > 1:
                    raise LiveProviderError(
                        f"search_entities returned an ambiguous {entity_type} mapping "
                        f"for {value!r}: {', '.join(selected_hits)}"
                    )
                if selected_hits:
                    resolved_by_input[value] = selected_hits[0]

        ordered: list[str] = []
        seen: set[str] = set()
        for value in values:
            entity_id = resolved_by_input.get(value)
            if entity_id is not None and entity_id not in seen:
                seen.add(entity_id)
                ordered.append(entity_id)
        return ordered

    async def _batch_query(self, *, query_string, variables_list, key_field, root_field):
        """Split only rejected aggregate requests; saved successful pages are reused."""
        try:
            return await self._batch_query_once(query_string=query_string, variables_list=variables_list,
                                               key_field=key_field, root_field=root_field)
        except LiveProviderError as exc:
            if len(variables_list) <= 1 or not (
                "batch exceeded its total response-size limit" in str(exc)
                or failure_category(exc) == "query_too_expensive"
            ):
                raise
            recovery_progress("transport", "splitting", reason="aggregate_request_too_large",
                              unit="requests", total_count=len(variables_list), completed_count=0)
            middle = len(variables_list) // 2
            left = await self._batch_query(query_string=query_string, variables_list=variables_list[:middle],
                                          key_field=key_field, root_field=root_field)
            right = await self._batch_query(query_string=query_string, variables_list=variables_list[middle:],
                                           key_field=key_field, root_field=root_field)
            return [*left, *right]

    async def _batch_query_once(
        self,
        *,
        query_string: str,
        variables_list: list[dict[str, Any]],
        key_field: str,
        root_field: str,
    ) -> list[dict[str, Any] | None]:
        response = await self._call_tool(
            "batch_query_open_targets_graphql",
            {
                "query_string": query_string,
                "variables_list": variables_list,
                "key_field": key_field,
            },
        )
        response = _unwrap_optional_success(
            response, label="batch_query_open_targets_graphql"
        )
        results = response.get("results")
        if not isinstance(results, list):
            raise LiveProviderError("batch GraphQL response is missing a results list")
        if len(results) != len(variables_list):
            raise LiveProviderError(
                "batch GraphQL response count does not match the requested variables"
            )

        ordered: list[dict[str, Any] | None] = [None] * len(variables_list)
        seen_indexes: set[int] = set()
        for fallback_index, item in enumerate(results):
            if not isinstance(item, dict):
                raise LiveProviderError("batch result entries must be objects")
            index = item.get("index", fallback_index)
            if not isinstance(index, int) or isinstance(index, bool):
                raise LiveProviderError("batch result index must be an integer")
            if index < 0 or index >= len(ordered) or index in seen_indexes:
                raise LiveProviderError("batch result contains an invalid or duplicate index")
            seen_indexes.add(index)

            expected_id = variables_list[index][key_field]
            returned_id = item.get("id")
            if returned_id is not None and str(returned_id) != expected_id:
                raise LiveProviderError("batch result id does not match its request")

            result = item.get("result")
            if not isinstance(result, dict):
                raise LiveProviderError("batch result is missing its query result object")
            data = _unwrap_query_result(result)
            ordered[index] = _requested_entity(data, root_field, str(expected_id))

        if len(seen_indexes) != len(ordered):
            raise LiveProviderError("batch GraphQL response omitted one or more indexes")
        return ordered

    async def _query_variant_profiles(
        self,
        variant_ids: list[str],
        *,
        include_transcript_consequences: bool = False,
    ) -> list[dict[str, Any] | None]:
        """Query canonical variants with bounded aliases to avoid N HTTP round trips."""

        if not variant_ids or len(variant_ids) > _DEFAULT_GRAPHQL_MAX_BATCH_SIZE:
            raise LiveProviderError("variant profile batch size must be between 1 and 100")
        canonical = []
        for value in variant_ids:
            normalized = _direct_entity_id(str(value), entity_type="variant")
            if normalized is None:
                raise LiveProviderError(
                    "variant profile aliases require canonical variant identifiers"
                )
            canonical.append(normalized)

        fields = "id chromosome position referenceAllele alternateAllele rsIds"
        batch_size = (parameter("retrieval.variant_consequence_batch_size")
                      if include_transcript_consequences
                      else parameter("retrieval.variant_core_batch_size"))
        if include_transcript_consequences:
            fields += (
                " transcriptConsequences { target { id approvedSymbol } "
                "consequenceScore distanceFromFootprint distanceFromTss "
                "siftPrediction polyphenPrediction }"
            )

        if self.backend_name != "official_graphql":
            query = (
                "query GWASMindVariantProfile($variantId: String!) { "
                f"variant(variantId: $variantId) {{ {fields} }} }}"
            )
            return await self._batch_query(
                query_string=query,
                variables_list=[{"variantId": item} for item in canonical],
                key_field="variantId",
                root_field="variant",
            )

        ordered: list[dict[str, Any] | None] = []
        for offset in range(0, len(canonical), batch_size):
            chunk = canonical[offset : offset + batch_size]
            definitions = ", ".join(
                f"$variant{index}: String!" for index in range(len(chunk))
            )
            selections = " ".join(
                f"variant{index}: variant(variantId: $variant{index}) {{ {fields} }}"
                for index in range(len(chunk))
            )
            query = (
                f"query GWASMindVariantProfiles({definitions}) {{ {selections} }}"
            )
            variables = {
                f"variant{index}": value for index, value in enumerate(chunk)
            }
            response = await self._call_tool(
                "query_open_targets_graphql",
                {"query_string": query, "variables": variables},
            )
            data = _unwrap_query_result(response)
            for index in range(len(chunk)):
                ordered.append(_requested_entity(data, f"variant{index}", chunk[index]))
        return ordered

    async def _query_target_profiles(
        self,
        target_ids: list[str],
        kinds: set[EvidenceKind],
        *, trait_ids: tuple[str, ...] = (),
    ) -> list[dict[str, Any] | None]:
        """Query target profiles in small GraphQL alias groups on the live API."""
        from .literature import resolved_literature_traits, prepare_literature_profiles

        traits = resolved_literature_traits(trait_ids) if EvidenceKind.LITERATURE in kinds else ()
        def finish(rows):
            return prepare_literature_profiles(rows, traits) if EvidenceKind.LITERATURE in kinds else rows

        if not target_ids or len(target_ids) > 200:
            raise LiveProviderError("target profile batch size must be between 1 and 200")
        canonical = []
        for value in target_ids:
            normalized = _direct_entity_id(str(value), entity_type="target")
            if normalized is None:
                raise LiveProviderError(
                    "target profile aliases require canonical Ensembl identifiers"
                )
            canonical.append(normalized)
        if EvidenceKind.EXPRESSION in kinds:
            return finish(await self._query_expression_targets(canonical, kinds, trait_ids=traits))
        if self.backend_name != "official_graphql":
            ordered = []
            for offset in range(0, len(canonical), _DEFAULT_GRAPHQL_MAX_BATCH_SIZE):
                ordered.extend(await self._batch_query(
                    query_string=_target_query(kinds, trait_ids=traits),
                    variables_list=[{"ensemblId": item} for item in canonical[offset:offset + _DEFAULT_GRAPHQL_MAX_BATCH_SIZE]],
                    key_field="ensemblId", root_field="target",
                ))
            if EvidenceKind.EXPRESSION in kinds:
                await self._complete_expression_profiles(ordered)
            return finish(ordered)

        fields = " ".join(_target_fields(kinds, trait_ids=traits))
        ordered: list[dict[str, Any] | None] = []
        for offset in range(0, len(canonical), 10):
            chunk = canonical[offset : offset + 10]
            definitions = ", ".join(
                f"$target{index}: String!" for index in range(len(chunk))
            )
            selections = " ".join(
                f"target{index}: target(ensemblId: $target{index}) {{ {fields} }}"
                for index in range(len(chunk))
            )
            query = f"query GWASMindTargetProfiles({definitions}) {{ {selections} }}"
            variables = {f"target{index}": value for index, value in enumerate(chunk)}
            response = await self._call_tool(
                "query_open_targets_graphql",
                {"query_string": query, "variables": variables},
            )
            data = _unwrap_query_result(response)
            for index in range(len(chunk)):
                ordered.append(_requested_entity(data, f"target{index}", chunk[index]))
        if EvidenceKind.EXPRESSION in kinds:
            await self._complete_expression_profiles(ordered)
        return finish(ordered)

    async def _query_expression_targets(self, canonical, kinds, *, trait_ids=()):
        """Recover only rejected alias groups; preserve each gene's validated pages."""
        remaining = [len(canonical) * parameter("retrieval.expression_requests_per_target")]
        fields_template = " ".join(_target_fields(kinds, trait_ids=trait_ids))
        limits = self._expression_request_limits.setdefault(
            (self.backend_name, fields_template), {"aliases": 10, "page_size": _EXPRESSION_PAGE_SIZE})
        completed = [0]
        async def first_page(chunk, size):
            # Do not rediscover the same rejected batch size for every new set of
            # genes. Consult the latest limit after each child: a child can reveal
            # a lower capacity, without discarding any successful sibling pages.
            if len(chunk) > limits["aliases"]:
                result, offset = [], 0
                while offset < len(chunk):
                    end = min(len(chunk), offset + limits["aliases"])
                    result.extend(await first_page(chunk[offset:end], size))
                    offset = end
                return result
            size = min(size, limits["page_size"])
            fields = fields_template.replace(
                f"size: {_EXPRESSION_PAGE_SIZE}", f"size: {size}")
            definitions = ", ".join(f"$target{i}: String!" for i in range(len(chunk)))
            selections = " ".join(
                f"target{i}: target(ensemblId: $target{i}) {{ {fields} }}" for i in range(len(chunk)))
            try:
                if remaining[0] <= 0:
                    raise LiveProviderError("expression continuation budget exhausted")
                remaining[0] -= 1
                if self.backend_name != "official_graphql":
                    result = await self._batch_query(
                        query_string=_target_query(kinds, trait_ids=trait_ids).replace(f"size: {_EXPRESSION_PAGE_SIZE}", f"size: {size}"),
                        variables_list=[{"ensemblId": gene} for gene in chunk],
                        key_field="ensemblId", root_field="target")
                else:
                    response = await self._call_tool("query_open_targets_graphql", {
                        "query_string": f"query GWASMindTargetProfiles({definitions}) {{ {selections} }}",
                        "variables": {f"target{i}": gene for i, gene in enumerate(chunk)},
                    })
                    data = _unwrap_query_result(response)
                    result = [_requested_entity(data, f"target{i}", gene) for i, gene in enumerate(chunk)]
                for target in result:
                    if target is not None:
                        target["_expression_page_size"] = size
                return result
            except LiveProviderError as exc:
                reason = failure_category(exc)
                if reason in {"query_too_expensive", "response_size_limit"} and (len(chunk) > 1 or size > 1):
                    recovery_progress("expression", "splitting", reason=reason, unit="genes",
                                      total_count=len(canonical), completed_count=completed[0],
                                      batch_size=len(chunk), page_size=size)
                    if len(chunk) > 1:
                        limits["aliases"] = min(limits["aliases"], (len(chunk) + 1) // 2)
                    else:
                        limits["page_size"] = min(limits["page_size"], max(1, size // 2))
                    return await first_page(chunk, size)
                # No second transport retry layer. The transport has exhausted
                # its own finite attempts; retain explicit unknown coverage.
                return [{"id": gene, "baselineExpression": {"count": None, "rows": [],
                    "coverage": {"complete": False, "declared_count": None,
                                 "retrieved_count": 0, "reason": reason}}} for gene in chunk]

        ordered = []
        offset = 0
        while offset < len(canonical):
            end = min(len(canonical), offset + limits["aliases"])
            chunk = await first_page(canonical[offset:end], _EXPRESSION_PAGE_SIZE)
            await self._complete_expression_profiles(chunk, retain_partial=True, request_budget=remaining)
            ordered.extend(chunk)
            offset = end
            completed[0] = len(ordered)
            recovery_progress("expression", "running", unit="genes",
                              completed_count=len(ordered), total_count=len(canonical))
        return ordered

    async def _complete_expression_profiles(self, targets, *, retain_partial=False, request_budget=None):
        """Validate count and fetch row-offset pages before any specificity calculation.

        baselineExpression uses row offsets for page.index in the archived 26.06
        contract. Count/identity/page-length changes fail closed instead of treating
        the default first 25 rows as the whole reference distribution.
        """
        query = (
            "query GWASMindExpressionPage($ensemblId: String!, $offset: Int!, $size: Int!) { "
            "target(ensemblId: $ensemblId) { id baselineExpression(page: {index: $offset, size: $size}) { count rows { "
            + _EXPRESSION_ROW_FIELDS + " } } } }"
        )
        for target in targets:
            if target is None:
                continue
            try:
                await self._complete_one_expression_profile(target, query, request_budget=request_budget)
            except LiveProviderError as exc:
                if not retain_partial:
                    raise
                connection = target.get("baselineExpression")
                if not isinstance(connection, dict):
                    connection = target["baselineExpression"] = {"rows": [], "count": None}
                rows = connection.get("rows")
                rows = rows if isinstance(rows, list) and all(isinstance(r, dict) for r in rows) else []
                connection["rows"] = rows
                count = connection.get("count")
                count = count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None
                connection["coverage"] = {"complete": False, "declared_count": count,
                    "retrieved_count": len(rows), "reason": failure_category(exc), "index_semantics": "row_offset"}
                recovery_progress("expression", "incomplete", entity_id=target["id"][:160],
                                  reason=failure_category(exc), unit="records",
                                  completed_count=len(rows), total_count=count)

    async def _complete_one_expression_profile(self, target, query, *, request_budget=None):
        # A failed first page is metadata only; it is not a zero-expression observation.
        if (target.get("baselineExpression") or {}).get("coverage", {}).get("complete") is False:
            return
        size = target.pop("_expression_page_size", _EXPRESSION_PAGE_SIZE)
        connection = target.get("baselineExpression")
        if not isinstance(connection, dict):
            raise LiveProviderError("baselineExpression must return count and rows")
        count, rows = connection.get("count"), connection.get("rows")
        if (not isinstance(count, int) or isinstance(count, bool) or count < 0
                or not isinstance(rows, list) or len(rows) != min(count, size)
                or any(not isinstance(row, dict) for row in rows)
                or count > _EXPRESSION_MAX_ROWS):
            raise LiveProviderError("baselineExpression count/page coverage is invalid or exceeds the bounded retrieval budget")
        identifier = _required_text(target.get("id"), label="target.id")
        # Retain validated pages for audit. Incomplete coverage excludes the
        # panel from expression qualification. Reject repeated page contents.
        rows = list(rows)
        previous_pages = {sha256_json(sorted(sha256_json(row) for row in rows))}
        while len(rows) < count:
            offset = len(rows)
            try:
                if request_budget is not None:
                    if request_budget[0] <= 0:
                        raise LiveProviderError("expression continuation budget exhausted")
                    request_budget[0] -= 1
                pages = await self._batch_query(
                    query_string=query,
                    variables_list=[{"ensemblId": identifier, "offset": offset, "size": size}],
                    key_field="ensemblId", root_field="target",
                )
            except LiveProviderError as exc:
                reason = failure_category(exc)
                if reason not in {"query_too_expensive", "response_size_limit"} or size <= 1:
                    raise
                size = max(1, size // 2)
                recovery_progress("expression", "splitting", entity_id=identifier[:160],
                                  reason=reason, page_size=size, offset=offset,
                                  unit="records", completed_count=offset, total_count=count)
                continue
            page = pages[0] if len(pages) == 1 else None
            if not isinstance(page, dict) or page.get("id") != identifier:
                raise LiveProviderError("baselineExpression continuation changed target identity")
            value = page.get("baselineExpression")
            value = value if isinstance(value, dict) else {}
            page_rows = value.get("rows")
            page_count = value.get("count")
            if (not isinstance(page_count, int) or isinstance(page_count, bool)
                    or page_count != count or not isinstance(page_rows, list)
                    or len(page_rows) != min(size, count - offset)
                    or any(not isinstance(row, dict) for row in page_rows)
                    or sha256_json(sorted(sha256_json(row) for row in page_rows)) in previous_pages):
                raise LiveProviderError("baselineExpression continuation is incomplete, repeated or changed its count")
            previous_pages.add(sha256_json(sorted(sha256_json(row) for row in page_rows)))
            rows.extend(page_rows)
            connection["rows"] = rows
            recovery_progress("expression", "running", entity_id=identifier[:160],
                              unit="records", completed_count=len(rows), total_count=count,
                              page_size=size, offset=offset)
        connection["rows"] = rows
        connection["coverage"] = {"complete": len(rows) == count, "declared_count": count,
                                  "retrieved_count": len(rows), "page_size": size,
                                  "index_semantics": "row_offset"}

    async def _query_variants(
        self, variant_ids: list[str], kinds: set[EvidenceKind]
    ) -> tuple[
        list[dict[str, Any] | None],
        tuple[CredibleSetCoverage, ...],
        tuple[EnhancerCoverage, ...],
    ]:
        query_string = _variant_query(kinds)
        credible_requested = bool(kinds.intersection(_CREDIBLE_SET_KINDS))
        enhancer_requested = EvidenceKind.REGULATORY in kinds
        if credible_requested and self.backend_name == "official_graphql":
            from .entity_retrieval import EntityEvidenceLoader
            if self._entity_loader is None:
                self._entity_loader = EntityEvidenceLoader(self)
            result = await self._entity_loader.variants(variant_ids, kinds)
            enhancer_coverage = ()
            if enhancer_requested:
                regulatory, _, enhancer_coverage = await self._query_variants(variant_ids, {EvidenceKind.REGULATORY})
                for variant, enrichment in zip(result.variants, regulatory, strict=True):
                    if enrichment is not None:
                        variant.update(enrichment)
                    else:
                        variant["enhancerToGenes"] = {"count": 0, "rows": []}
            return result.variants, tuple(CredibleSetCoverage(**item) for item in result.coverage), enhancer_coverage
        first_page_variables = [
            _variant_page_variables(
                item,
                credible_requested=credible_requested,
                enhancer_requested=enhancer_requested,
            )
            for item in variant_ids
        ]
        variants = await self._batch_query(
            query_string=query_string,
            variables_list=first_page_variables,
            key_field="variantId",
            root_field="variant",
        )
        credible_coverage = (
            await self._paginate_credible_sets(
                variants,
                variant_ids,
                query_string=query_string,
                enhancer_requested=enhancer_requested,
            )
            if credible_requested
            else ()
        )
        enhancer_coverage = (
            await self._paginate_enhancer_to_genes(
                variants,
                variant_ids,
                query_string=query_string,
                credible_requested=credible_requested,
            )
            if enhancer_requested
            else ()
        )
        if EvidenceKind.COLOCALIZATION in kinds:
            await self._paginate_colocalisation(variants)
        return variants, credible_coverage, enhancer_coverage

    async def _paginate_colocalisation(self, variants):
        """Continue unique QTL credible sets using the official page-number contract.

        Unlike expression, Backend.getColocalisations uses Pagination.offsetLimit
        (index * size). Reuse a validated connection across proxy SNP aliases.
        """
        query = (
            "query GWASMindColocalisationPage($studyLocusId: String!, $index: Int!, $size: Int!) { "
            "credibleSet(studyLocusId: $studyLocusId) { id: studyLocusId studyLocusId "
            "colocalisation(page: {index: $index, size: $size}) { count rows { "
            + _COLOCALISATION_ROW_FIELDS + " } } } }"
        )
        groups = {}
        for variant in variants:
            if variant is None:
                continue
            for credible in _connection_rows(variant, "credibleSets"):
                study = credible.get("study") or {}
                if native_qtl_study_type(credible.get("studyType") or study.get("studyType")) is None:
                    continue
                identifier = _required_text(credible.get("studyLocusId"), label="studyLocusId")
                groups.setdefault(identifier, []).append(credible)
        budget = parameter("retrieval.colocalisation_max_continuation_requests")
        for identifier, parents in groups.items():
            first = parents[0].get("colocalisation")
            rows = _connection_rows(parents[0], "colocalisation")
            count = first.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count < len(rows):
                raise LiveProviderError("colocalisation count is invalid")
            if count == len(rows):
                continue
            cache_key = (identifier, sha256_json(first))
            cached = self._colocalisation_cache.get(cache_key)
            if cached is not None and all(parent.get("colocalisation") == first for parent in parents[1:]):
                for parent in parents:
                    parent["colocalisation"] = cached
                self._colocalisation_cache.move_to_end(cache_key)
                recovery_progress("colocalisation", "cached", entity_id=identifier[:160],
                                  unit="records", completed_count=len(cached["rows"]), total_count=count)
                continue
            # Upstream default size is 25. Infer an explicitly supplied coherent
            # first page from its retained length, preserving compatibility.
            size = len(rows)
            if size == 0:
                first["coverage"] = {"reason": "empty_first_page", "complete": False}
                continue
            if any(parent.get("colocalisation") != first for parent in parents[1:]):
                for parent in parents:
                    parent["colocalisation"]["coverage"] = {"reason": "inconsistent_alias_pages", "complete": False}
                continue
            retained = list(rows)
            seen = {sha256_json(row) for row in rows}
            reason = None
            while len(retained) < count:
                if budget <= 0:
                    reason = "continuation_budget_exhausted"
                    break
                budget -= 1
                try:
                    pages = await self._batch_query(query_string=query,
                        variables_list=[{"studyLocusId": identifier, "index": len(retained) // size, "size": size}],
                        key_field="studyLocusId", root_field="credibleSet")
                    page = pages[0] if len(pages) == 1 else None
                    if not isinstance(page, dict) or page.get("studyLocusId") != identifier:
                        raise LiveProviderError("colocalisation continuation changed credible set identity")
                    connection = page.get("colocalisation")
                    page_rows = _connection_rows(page, "colocalisation")
                    hashes = [sha256_json(row) for row in page_rows]
                    if (connection.get("count") != count or isinstance(connection.get("count"), bool)
                            or len(page_rows) != min(size, count - len(retained))
                            or len(set(hashes)) != len(hashes) or seen.intersection(hashes)):
                        raise LiveProviderError("colocalisation continuation changed count or repeated rows")
                    retained.extend(page_rows)
                    seen.update(hashes)
                    recovery_progress("colocalisation", "running", entity_id=identifier[:160],
                                      unit="records", completed_count=len(retained), total_count=count)
                except LiveProviderError as exc:
                    reason = failure_category(exc)
                    break
            value = {**first, "rows": retained, "coverage": {
                "complete": len(retained) == count and reason is None,
                "declared_count": count, "retrieved_count": len(retained),
                "page_size": size, "index_semantics": "page_number", "reason": reason,
            }}
            for parent in parents:
                parent["colocalisation"] = value
            # Bound the cache by both connections and retained row count; only
            # coherent full pages can satisfy another proxy's identical query.
            if value["coverage"]["complete"]:
                self._colocalisation_cache[cache_key] = value
                while (len(self._colocalisation_cache) > 16 or
                       sum(len(item["rows"]) for item in self._colocalisation_cache.values()) > 10000):
                    self._colocalisation_cache.popitem(last=False)
            recovery_progress("colocalisation", "incomplete" if reason else "completed",
                              entity_id=identifier[:160], unit="records", completed_count=len(retained),
                              total_count=count, reason=reason)

    async def _paginate_credible_sets(
        self,
        variants: list[dict[str, Any] | None],
        variant_ids: list[str],
        *,
        query_string: str,
        enhancer_requested: bool,
    ) -> tuple[CredibleSetCoverage, ...]:
        continuation_pages: dict[int, int] = {}
        seen_study_locus_ids: dict[int, set[str]] = {}
        for variant_index, variant in enumerate(variants):
            if variant is None:
                continue
            expected_variant_id = variant_ids[variant_index]
            _validate_variant_page_identity(variant, expected_variant_id)
            count, rows = _credible_set_page(variant, page_index=0, expected_count=None)
            seen_study_locus_ids[variant_index] = _credible_set_row_ids(
                rows, variant_id=expected_variant_id
            )
            continuation_pages[variant_index] = max(
                math.ceil(count / _CREDIBLE_SET_PAGE_SIZE) - 1,
                0,
            )

        continuations = _round_robin_continuations(
            continuation_pages,
            limit=_CREDIBLE_SET_MAX_CONTINUATION_REQUESTS,
        )
        fetched_pages = {index: 0 for index in continuation_pages}
        for offset in range(0, len(continuations), _CREDIBLE_SET_CONTINUATION_BATCH_SIZE):
            chunk = continuations[offset : offset + _CREDIBLE_SET_CONTINUATION_BATCH_SIZE]
            page_results = await self._batch_query(
                query_string=query_string,
                variables_list=[
                    _variant_page_variables(
                        variant_ids[variant_index],
                        credible_requested=True,
                        enhancer_requested=enhancer_requested,
                        credible_page_index=page_index,
                    )
                    for variant_index, page_index in chunk
                ],
                key_field="variantId",
                root_field="variant",
            )
            for (variant_index, page_index), page_variant in zip(
                chunk, page_results, strict=True
            ):
                expected_variant_id = variant_ids[variant_index]
                if page_variant is None:
                    raise LiveProviderError(
                        "credibleSets continuation page unexpectedly returned a null variant"
                    )
                _validate_variant_page_identity(page_variant, expected_variant_id)
                first_variant = variants[variant_index]
                if first_variant is None:  # pragma: no cover - established above
                    raise LiveProviderError("credibleSets pagination lost its first page")
                first_connection = first_variant.get("credibleSets")
                if not isinstance(first_connection, dict):  # pragma: no cover
                    raise LiveProviderError("variant credibleSets must be an object")
                expected_count = first_connection.get("count")
                if not isinstance(expected_count, int) or isinstance(expected_count, bool):
                    raise LiveProviderError("credibleSets count must be an integer")
                _, page_rows = _credible_set_page(
                    page_variant,
                    page_index=page_index,
                    expected_count=expected_count,
                )
                page_ids = _credible_set_row_ids(page_rows, variant_id=expected_variant_id)
                if seen_study_locus_ids[variant_index].intersection(page_ids):
                    raise LiveProviderError(
                        "credibleSets pagination returned duplicate studyLocusId values"
                    )
                seen_study_locus_ids[variant_index].update(page_ids)
                first_rows = first_connection.get("rows")
                if not isinstance(first_rows, list):  # pragma: no cover
                    raise LiveProviderError("credibleSets rows must be a list")
                first_rows.extend(page_rows)
                fetched_pages[variant_index] += 1

        coverage: list[CredibleSetCoverage] = []
        for variant_index, variant in enumerate(variants):
            if variant is None:
                continue
            connection = variant.get("credibleSets")
            if not isinstance(connection, dict):  # pragma: no cover
                raise LiveProviderError("variant credibleSets must be an object")
            count = connection.get("count")
            rows = connection.get("rows")
            if not isinstance(count, int) or isinstance(count, bool):
                raise LiveProviderError("credibleSets count must be an integer")
            expected_rows = min(
                count,
                _CREDIBLE_SET_PAGE_SIZE * (1 + fetched_pages.get(variant_index, 0)),
            )
            if not isinstance(rows, list) or len(rows) != expected_rows:
                raise LiveProviderError(
                    "credibleSets pagination did not match the deterministic retained-page "
                    "budget"
                )
            if len(seen_study_locus_ids[variant_index]) != len(rows):
                raise LiveProviderError(
                    "credibleSets pagination did not yield one unique studyLocusId per row"
                )
            coverage.append(
                CredibleSetCoverage(variant_ids[variant_index], count, len(rows))
            )
        return tuple(coverage)

    async def _paginate_enhancer_to_genes(
        self,
        variants: list[dict[str, Any] | None],
        variant_ids: list[str],
        *,
        query_string: str,
        credible_requested: bool,
    ) -> tuple[EnhancerCoverage, ...]:
        continuation_pages: dict[int, int] = {}
        for variant_index, variant in enumerate(variants):
            if variant is None:
                continue
            expected_variant_id = variant_ids[variant_index]
            _validate_variant_page_identity(variant, expected_variant_id)
            count, _ = _enhancer_page(variant, page_index=0, expected_count=None)
            continuation_pages[variant_index] = max(
                math.ceil(count / _ENHANCER_PAGE_SIZE) - 1,
                0,
            )

        continuations = _round_robin_continuations(
            continuation_pages,
            limit=_ENHANCER_MAX_CONTINUATION_REQUESTS,
        )
        fetched_pages = {index: 0 for index in continuation_pages}
        for offset in range(0, len(continuations), _ENHANCER_CONTINUATION_BATCH_SIZE):
            chunk = continuations[offset : offset + _ENHANCER_CONTINUATION_BATCH_SIZE]
            page_results = await self._batch_query(
                query_string=query_string,
                variables_list=[
                    _variant_page_variables(
                        variant_ids[variant_index],
                        credible_requested=credible_requested,
                        enhancer_requested=True,
                        enhancer_page_index=page_index,
                    )
                    for variant_index, page_index in chunk
                ],
                key_field="variantId",
                root_field="variant",
            )
            for (variant_index, page_index), page_variant in zip(
                chunk, page_results, strict=True
            ):
                expected_variant_id = variant_ids[variant_index]
                if page_variant is None:
                    raise LiveProviderError(
                        "enhancerToGenes continuation page unexpectedly returned a null variant"
                    )
                _validate_variant_page_identity(page_variant, expected_variant_id)
                first_variant = variants[variant_index]
                if first_variant is None:  # pragma: no cover - established above
                    raise LiveProviderError("enhancerToGenes pagination lost its first page")
                first_connection = first_variant.get("enhancerToGenes")
                if not isinstance(first_connection, dict):  # pragma: no cover
                    raise LiveProviderError("variant enhancerToGenes must be an object")
                expected_count = first_connection.get("count")
                if not isinstance(expected_count, int) or isinstance(expected_count, bool):
                    raise LiveProviderError("enhancerToGenes count must be an integer")
                _, page_rows = _enhancer_page(
                    page_variant,
                    page_index=page_index,
                    expected_count=expected_count,
                )
                first_rows = first_connection.get("rows")
                if not isinstance(first_rows, list):  # pragma: no cover
                    raise LiveProviderError("enhancerToGenes rows must be a list")
                first_rows.extend(page_rows)
                fetched_pages[variant_index] += 1

        coverage: list[EnhancerCoverage] = []
        for variant_index, variant in enumerate(variants):
            if variant is None:
                continue
            connection = variant.get("enhancerToGenes")
            if not isinstance(connection, dict):  # pragma: no cover
                raise LiveProviderError("variant enhancerToGenes must be an object")
            count = connection.get("count")
            rows = connection.get("rows")
            if not isinstance(count, int) or isinstance(count, bool):
                raise LiveProviderError("enhancerToGenes count must be an integer")
            expected_rows = min(
                count,
                _ENHANCER_PAGE_SIZE * (1 + fetched_pages.get(variant_index, 0)),
            )
            if not isinstance(rows, list) or len(rows) != expected_rows:
                raise LiveProviderError(
                    "enhancerToGenes pagination did not match the deterministic retained-page "
                    "budget"
                )
            coverage.append(EnhancerCoverage(variant_ids[variant_index], count, len(rows)))
        return tuple(coverage)

    def _normalize_variant(
        self, variant: dict[str, Any], kinds: set[EvidenceKind]
    ) -> list[EvidenceRecord]:
        variant_id = _required_text(variant.get("id"), label="variant.id")
        records: list[EvidenceRecord] = []
        qtl_lineages: list[
            tuple[dict[str, Any], str | None, str | None, tuple[str, ...]]
        ] = []
        if kinds.intersection(_CREDIBLE_SET_KINDS):
            credible_connection = variant.get("credibleSets")
            if not isinstance(credible_connection, dict):
                raise LiveProviderError("variant credibleSets must be an object")
            credible_rows = _connection_rows(variant, "credibleSets")
            credible_count = credible_connection.get("count")
            if (
                not isinstance(credible_count, int)
                or isinstance(credible_count, bool)
                or credible_count < len(credible_rows)
            ):
                raise LiveProviderError(
                    "credibleSets retained rows violate the bounded live-query contract"
                )
            for credible in credible_rows:
                study_value = credible.get("study")
                if study_value is None:
                    study: dict[str, Any] = {}
                elif isinstance(study_value, dict):
                    study = study_value
                else:
                    raise LiveProviderError("credible set study must be an object or null")

                base = _credible_set_payload(variant_id, credible, study)
                study_id = _optional_text(base.get("studyId"))
                credible_set_id = _optional_text(base.get("studyLocusId"))
                pmid = _optional_text(base.get("pubmedId"))
                evidence_date = _parse_date(base.get("publicationDate"))
                lead_variant_id = _required_text(
                    base.get("leadVariantId"), label="credible set leadVariantId"
                )
                parents = tuple(dict.fromkeys((variant_id, lead_variant_id)))
                study_type = _required_text(
                    base.get("studyType"), label="credible set studyType"
                ).rsplit(".", 1)[-1].casefold()
                qtl_type = native_qtl_study_type(study_type)
                if qtl_type is not None:
                    qtl_lineages.append((base, study_id, credible_set_id, parents))

                if EvidenceKind.GWAS in kinds or EvidenceKind.QTL in kinds:
                    if study_type == "gwas":
                        kind: EvidenceKind | None = EvidenceKind.GWAS
                    elif qtl_type is not None:
                        kind = EvidenceKind.QTL
                    else:
                        kind = None
                    if kind is not None and kind in kinds:
                        records.append(
                            self._record(
                                kind=kind,
                                dataset="credible_set",
                                payload=base,
                                variant_id=variant_id,
                                gene_id=_optional_text(base.get("qtlGeneId")),
                                study_id=study_id,
                                pmid=pmid,
                                credible_set_id=credible_set_id,
                                parent_variant_ids=parents,
                                evidence_date=evidence_date,
                                tissue=_optional_text(base.get("biosampleName")),
                            )
                        )

                if EvidenceKind.L2G in kinds:
                    for prediction in _connection_rows(credible, "l2GPredictions"):
                        target = prediction.get("target")
                        if not isinstance(target, dict):
                            raise LiveProviderError("L2G prediction target must be an object")
                        gene_id = _required_text(target.get("id"), label="L2G target.id")
                        payload = {
                            "variantId": variant_id,
                            "queryVariantId": base.get("queryVariantId"),
                            "leadVariantId": lead_variant_id,
                            "queryVariantRole": base.get("queryVariantRole"),
                            "matchedLocusMember": base.get("matchedLocusMember"),
                            "studyId": base.get("studyId"),
                            "studyLocusId": base.get("studyLocusId"),
                            "traitFromSource": base.get("traitFromSource"),
                            "pubmedId": base.get("pubmedId"),
                            "publicationDate": base.get("publicationDate"),
                            "publicationTitle": base.get("publicationTitle"),
                            "publicationFirstAuthor": base.get(
                                "publicationFirstAuthor"
                            ),
                            "publicationJournal": base.get("publicationJournal"),
                            "geneId": gene_id,
                            "approvedSymbol": target.get("approvedSymbol"),
                            "l2g_score": prediction.get("score"),
                        }
                        records.append(
                            self._record(
                                kind=EvidenceKind.L2G,
                                dataset="l2g_prediction",
                                payload=payload,
                                variant_id=variant_id,
                                gene_id=gene_id,
                                study_id=study_id,
                                pmid=pmid,
                                credible_set_id=credible_set_id,
                                parent_variant_ids=parents,
                                evidence_date=evidence_date,
                            )
                        )

                qtl_gene_id = _optional_text(base.get("qtlGeneId"))
                if (
                    EvidenceKind.COLOCALIZATION in kinds
                    and qtl_type is not None
                    and qtl_gene_id is not None
                ):
                    for coloc in _connection_rows(credible, "colocalisation"):
                        other_value = coloc.get("otherStudyLocus")
                        if other_value is None:
                            other: dict[str, Any] = {}
                        elif isinstance(other_value, dict):
                            other = other_value
                        else:
                            raise LiveProviderError(
                                "colocalisation otherStudyLocus must be an object or null"
                            )
                        other_study_value = other.get("study")
                        if other_study_value is None:
                            other_study: dict[str, Any] = {}
                        elif isinstance(other_study_value, dict):
                            other_study = other_study_value
                        else:
                            raise LiveProviderError(
                                "colocalisation counterpart study must be an object or null"
                            )
                        _validate_study_lineage(other, other_study)
                        counterpart_type = other.get("studyType") or other_study.get("studyType")
                        _validate_study_type_agreement(
                            coloc.get("rightStudyType"), counterpart_type
                        )
                        other_target_value = other_study.get("target")
                        if other_target_value is None:
                            other_target: dict[str, Any] = {}
                        elif isinstance(other_target_value, dict):
                            other_target = other_target_value
                        else:
                            raise LiveProviderError(
                                "colocalisation counterpart target must be an object or null"
                            )
                        right_qtl_gene_id = _optional_text(other.get("qtlGeneId"))
                        right_study_target_id = _optional_text(other_target.get("id"))
                        if (
                            right_qtl_gene_id is not None
                            and right_study_target_id is not None
                            and right_qtl_gene_id != right_study_target_id
                        ):
                            raise LiveProviderError(
                                "colocalisation counterpart qtlGeneId conflicts with "
                                "its study target"
                            )
                        right_qtl_gene_id = right_qtl_gene_id or right_study_target_id
                        payload = {
                            "variantId": variant_id,
                            "queryVariantId": base.get("queryVariantId"),
                            "leadVariantId": lead_variant_id,
                            "queryVariantRole": base.get("queryVariantRole"),
                            "matchedLocusMember": base.get("matchedLocusMember"),
                            "studyId": base.get("studyId"),
                            "studyType": qtl_type,
                            "studyLocusId": base.get("studyLocusId"),
                            "qtlStudyLocusId": base.get("studyLocusId"),
                            "qtlGeneId": qtl_gene_id,
                            "leftStudyLocusId": base.get("studyLocusId"),
                            "leftStudyId": base.get("studyId"),
                            "leftStudyType": qtl_type,
                            "leftTraitFromSource": base.get("traitFromSource"),
                            "leftPubmedId": base.get("pubmedId"),
                            "leftPublicationDate": base.get("publicationDate"),
                            "leftPublicationTitle": base.get("publicationTitle"),
                            "leftPublicationFirstAuthor": base.get(
                                "publicationFirstAuthor"
                            ),
                            "leftPublicationJournal": base.get("publicationJournal"),
                            "rightStudyLocusId": other.get("studyLocusId"),
                            "rightStudyType": coloc.get("rightStudyType"),
                            "rightStudyId": other.get("studyId") or other_study.get("id"),
                            "rightQtlGeneId": right_qtl_gene_id,
                            "rightQtlGeneSymbol": other_target.get("approvedSymbol"),
                            "rightTraitFromSource": other_study.get("traitFromSource"),
                            **({"targetTraitRelation": other.get("targetTraitRelation"),
                                "rightDetailRetrievalState": other.get("detailRetrievalState")}
                               if "targetTraitRelation" in other else {}),
                            "rightPubmedId": other_study.get("pubmedId"),
                            "rightPublicationDate": other_study.get(
                                "publicationDate"
                            ),
                            "rightPublicationTitle": other_study.get(
                                "publicationTitle"
                            ),
                            "rightPublicationFirstAuthor": other_study.get(
                                "publicationFirstAuthor"
                            ),
                            "rightPublicationJournal": other_study.get(
                                "publicationJournal"
                            ),
                            # Generic study/publication aliases belong to the
                            # queried (left) molecular-QTL study, matching
                            # ``studyId``.  The colocalising phenotype study is
                            # always represented by explicit right-prefixed
                            # fields.  Keeping those lineages separate prevents
                            # a GTEx study from being mislabeled with a FinnGen
                            # or other GWAS publication.
                            "traitFromSource": base.get("traitFromSource"),
                            "pubmedId": base.get("pubmedId"),
                            "publicationDate": base.get("publicationDate"),
                            "publicationTitle": base.get("publicationTitle"),
                            "publicationFirstAuthor": base.get(
                                "publicationFirstAuthor"
                            ),
                            "publicationJournal": base.get("publicationJournal"),
                            "h3": coloc.get("h3"),
                            "h4": coloc.get("h4"),
                            "clpp": coloc.get("clpp"),
                            "colocalisationMethod": coloc.get("colocalisationMethod"),
                            "numberColocalisingVariants": coloc.get(
                                "numberColocalisingVariants"
                            ),
                        }
                        if "betaRatioSignAverage" in coloc:
                            payload["betaRatioSignAverage"] = coloc["betaRatioSignAverage"]
                        records.append(
                            self._record(
                                kind=EvidenceKind.COLOCALIZATION,
                                dataset="colocalisation_coloc",
                                payload=payload,
                                variant_id=variant_id,
                                gene_id=qtl_gene_id,
                                study_id=study_id,
                                pmid=_optional_text(base.get("pubmedId")),
                                credible_set_id=credible_set_id,
                                parent_variant_ids=parents,
                                evidence_date=_parse_date(base.get("publicationDate")),
                                tissue=_optional_text(base.get("biosampleName")),
                            )
                        )

        if EvidenceKind.REGULATORY in kinds:
            # Pagination completeness is validated by ``_query_variants`` and
            # represented separately as coverage metadata.  Normalization must
            # retain bounded partial pages instead of reclassifying them as a
            # malformed biological no-data response.
            for enhancer in _connection_rows(variant, "enhancerToGenes"):
                target = enhancer.get("target")
                if not isinstance(target, dict):
                    raise LiveProviderError("enhancer-to-gene target must be an object")
                gene_id = _required_text(
                    target.get("id"), label="enhancer-to-gene target.id"
                )
                location = _canonical_variant_location(variant_id)
                chromosome = _optional_text(enhancer.get("chromosome"))
                start = enhancer.get("start")
                end = enhancer.get("end")
                if (
                    location is None
                    or chromosome is None
                    or not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                ):
                    raise LiveProviderError(
                        "enhancer-to-gene row omitted a valid genomic interval"
                    )
                if start < 0 or end < start:
                    raise LiveProviderError(
                        "enhancer-to-gene row contains an invalid genomic interval"
                    )
                query_chromosome, query_position = location
                if (
                    chromosome.removeprefix("chr") != query_chromosome
                    or not start <= query_position <= end
                ):
                    raise LiveProviderError(
                        "enhancer-to-gene interval does not contain the query variant"
                    )
                regulatory_element_key = (
                    f"chr{chromosome.removeprefix('chr')}:{start}-{end}"
                )
                payload = {
                    "variantId": variant_id,
                    "queryVariantId": variant_id,
                    "overlapVariantId": variant_id,
                    "overlapVariantRole": "selected",
                    "overlapChromosome": query_chromosome,
                    "overlapPosition": query_position,
                    "exactVariantOverlap": True,
                    "coverageScope": "selected_variants_exact_overlap",
                    # The current live selection does not expose an upstream
                    # interval identifier.  This coordinate key is explicitly
                    # local/derived and is never presented as a source ID.
                    "regulatoryElementKey": regulatory_element_key,
                    "regulatoryElementKeySource": "derived_coordinate_key",
                    "geneId": gene_id,
                    "approvedSymbol": target.get("approvedSymbol"),
                    "chromosome": enhancer.get("chromosome"),
                    "start": start,
                    "end": end,
                    # Keep the upstream type verbatim.  In particular, an
                    # ``intergenic`` row is an E2G regulatory interval; it is
                    # not rewritten as a generic enhancer annotation.
                    "intervalType": enhancer.get("intervalType"),
                    "distanceToTss": enhancer.get("distanceToTss"),
                    "score": enhancer.get("score"),
                    "resourceScore": enhancer.get("resourceScore"),
                    "datasourceId": enhancer.get("datasourceId"),
                    "pmid": enhancer.get("pmid"),
                    "enhancerStudyId": enhancer.get("studyId"),
                    "biosampleName": enhancer.get("biosampleName"),
                    "biosampleFromSourceId": enhancer.get("biosampleFromSourceId"),
                    "qualityControls": enhancer.get("qualityControls"),
                }
                records.append(
                    self._record(
                        kind=EvidenceKind.REGULATORY,
                        dataset="enhancer_to_gene",
                        payload=payload,
                        variant_id=variant_id,
                        gene_id=gene_id,
                        study_id=_optional_text(enhancer.get("studyId")),
                        pmid=_optional_text(enhancer.get("pmid")),
                        parent_variant_ids=(variant_id,),
                        tissue=_optional_text(enhancer.get("biosampleName")),
                    )
                )
        return records

    def _normalize_target(
        self, target: dict[str, Any], kinds: set[EvidenceKind]
    ) -> list[EvidenceRecord]:
        gene_id = _required_text(target.get("id"), label="target.id")
        symbol = target.get("approvedSymbol")
        records: list[EvidenceRecord] = []

        if EvidenceKind.FUNCTION in kinds:
            payload = {
                "id": gene_id,
                "approvedSymbol": symbol,
                "approvedName": target.get("approvedName"),
                "biotype": target.get("biotype"),
                "functionDescriptions": target.get("functionDescriptions"),
                "pathways": target.get("pathways"),
            }
            if any(
                payload.get(field)
                for field in (
                    "approvedName",
                    "biotype",
                    "functionDescriptions",
                    "pathways",
                )
            ):
                records.append(
                    self._record(
                        kind=EvidenceKind.FUNCTION,
                        dataset="target",
                        payload=payload,
                        gene_id=gene_id,
                    )
                )

        if EvidenceKind.EXPRESSION in kinds:
            rows = _connection_rows(target, "baselineExpression")
            payload = {
                "geneId": gene_id,
                "approvedSymbol": symbol,
                "tissues": rows,
                "expressionCoverage": (target.get("baselineExpression") or {}).get("coverage"),
            }
            if rows or (payload["expressionCoverage"] or {}).get("complete") is False:
                records.append(
                    self._record(
                        kind=EvidenceKind.EXPRESSION,
                        dataset="expression",
                        payload=payload,
                        gene_id=gene_id,
                    )
                )

        if EvidenceKind.LITERATURE in kinds:
            for occurrence in _connection_rows(target, "literatureOcurrences"):
                payload = {
                    "geneId": gene_id,
                    "approvedSymbol": symbol,
                    "pmid": occurrence.get("pmid"),
                    "pmcid": occurrence.get("pmcid"),
                    "publicationDate": occurrence.get("publicationDate"),
                    "literatureScope": occurrence.get('literatureScope', {
                        'phenotype_filtered': False, 'trait_ids': [], 'mode': 'gene_background',
                        'filter_status': 'not_applied_no_resolved_trait'}),
                    "literatureConnection": {
                        "count": target["literatureOcurrences"].get("count"),
                        "cursor": target["literatureOcurrences"].get("cursor"),
                    },
                }
                records.append(
                    self._record(
                        kind=EvidenceKind.LITERATURE,
                        dataset="literature",
                        payload=payload,
                        gene_id=gene_id,
                        pmid=_optional_text(occurrence.get("pmid")),
                        evidence_date=_parse_date(occurrence.get("publicationDate")),
                    )
                )
        return records

    def _record(
        self,
        *,
        kind: EvidenceKind,
        dataset: str,
        payload: dict[str, Any],
        variant_id: str | None = None,
        gene_id: str | None = None,
        study_id: str | None = None,
        pmid: str | None = None,
        credible_set_id: str | None = None,
        parent_variant_ids: tuple[str, ...] = (),
        tissue: str | None = None,
        evidence_date: date | None = None,
    ) -> EvidenceRecord:
        normalized = _json_value(payload, label=f"{dataset} payload")
        if not isinstance(normalized, dict):  # pragma: no cover - payload is statically a dict
            raise LiveProviderError(f"{dataset} payload must be an object")
        return EvidenceRecord(
            evidence_id=_evidence_id(dataset, normalized),
            kind=kind,
            dataset=dataset,
            source=self.source,
            data_release=self.capabilities.release,
            payload=normalized,
            variant_id=variant_id,
            gene_id=gene_id,
            study_id=study_id,
            pmid=pmid,
            credible_set_id=credible_set_id,
            parent_variant_ids=parent_variant_ids,
            tissue=tissue,
            evidence_date=evidence_date,
        )


def _variant_query(kinds: set[EvidenceKind]) -> str:
    fields = [
        "id",
        "chromosome",
        "position",
        "referenceAllele",
        "alternateAllele",
        "rsIds",
    ]
    credible_sets_requested = bool(kinds.intersection(_CREDIBLE_SET_KINDS))
    if credible_sets_requested:
        credible_fields = [
            "studyLocusId",
            "studyId",
            "studyType",
            "chromosome",
            "position",
            "beta",
            "pValueMantissa",
            "pValueExponent",
            "finemappingMethod",
            "confidence",
            "qualityControls",
            "qtlGeneId",
            "variant { id }",
            (
                "locus(variantIds: [$variantId]) { count rows { "
                "variant { id } posteriorProbability logBF "
                "is95CredibleSet is99CredibleSet pValueMantissa pValueExponent "
                "beta standardError r2Overall } }"
            ),
            (
                "study { id studyType traitFromSource pubmedId publicationDate "
                "publicationTitle publicationFirstAuthor publicationJournal "
                "target { id approvedSymbol } "
                "biosample { biosampleId biosampleName } }"
            ),
        ]
        if EvidenceKind.L2G in kinds:
            credible_fields.append(
                "l2GPredictions { count rows { score target { id approvedSymbol } } }"
            )
        if EvidenceKind.COLOCALIZATION in kinds:
            credible_fields.append(
                "colocalisation { count rows { " + _COLOCALISATION_ROW_FIELDS + " } }"
            )
        fields.append(
            "credibleSets(page: { index: $crediblePageIndex, "
            "size: $crediblePageSize }) { count rows { "
            + " ".join(credible_fields)
            + " } }"
        )
    if EvidenceKind.REGULATORY in kinds:
        fields.append(
            "enhancerToGenes(page: { index: $enhancerPageIndex, "
            "size: $enhancerPageSize }) { count rows { chromosome start end "
            "intervalType distanceToTss "
            "score resourceScore { name value } datasourceId pmid studyId biosampleName "
            "biosampleFromSourceId qualityControls target { id approvedSymbol } } }"
        )
    variable_declaration = "$variantId: String!"
    if credible_sets_requested:
        variable_declaration += ", $crediblePageIndex: Int!, $crediblePageSize: Int!"
    if EvidenceKind.REGULATORY in kinds:
        variable_declaration += ", $enhancerPageIndex: Int!, $enhancerPageSize: Int!"
    return (
        f"query GWASMindVariant({variable_declaration}) {{ "
        "variant(variantId: $variantId) { "
        + " ".join(fields)
        + " } }"
    )


def _target_query(kinds: set[EvidenceKind], *, trait_ids=()) -> str:
    fields = _target_fields(kinds, trait_ids=trait_ids)
    return (
        "query GWASMindTarget($ensemblId: String!) { "
        "target(ensemblId: $ensemblId) { "
        + " ".join(fields)
        + " } }"
    )


def _target_fields(kinds: set[EvidenceKind], *, trait_ids=()) -> list[str]:
    fields = ["id", "approvedSymbol"]
    if EvidenceKind.FUNCTION in kinds:
        fields.extend(
            [
                "approvedName",
                "biotype",
                "functionDescriptions",
                "pathways { pathwayId pathway topLevelTerm }",
            ]
        )
    if EvidenceKind.EXPRESSION in kinds:
        fields.append(
            f"baselineExpression(page: {{index: 0, size: {_EXPRESSION_PAGE_SIZE}}}) {{ count rows {{ "
            + _EXPRESSION_ROW_FIELDS + " } }"
        )
    if EvidenceKind.LITERATURE in kinds:
        from .literature import literature_field
        fields.append(literature_field(trait_ids))
    return fields


def _credible_set_payload(
    variant_id: str, credible: dict[str, Any], study: dict[str, Any]
) -> dict[str, Any]:
    _validate_study_lineage(credible, study)
    lead = credible.get("variant")
    if not isinstance(lead, dict):
        raise LiveProviderError("credible set variant must be an object")
    lead_variant_id = _required_text(
        lead.get("id"), label="credible set lead variant.id"
    )
    locus = credible.get("locus")
    if not isinstance(locus, dict):
        raise LiveProviderError("credible set locus must be an object")
    members = _connection_rows(credible, "locus")
    locus_count = locus.get("count")
    if (
        not isinstance(locus_count, int)
        or isinstance(locus_count, bool)
        or locus_count != len(members)
    ):
        raise LiveProviderError(
            "filtered credible set locus count does not match its returned rows"
        )
    matched_members: list[dict[str, Any]] = []
    for index, member in enumerate(members):
        member_variant = member.get("variant")
        if not isinstance(member_variant, dict):
            raise LiveProviderError(
                f"credible set locus member {index} variant must be an object"
            )
        member_variant_id = _required_text(
            member_variant.get("id"),
            label=f"credible set locus member {index} variant.id",
        )
        if member_variant_id != variant_id:
            continue
        posterior_probability = member.get("posteriorProbability")
        if (
            not isinstance(posterior_probability, (int, float))
            or isinstance(posterior_probability, bool)
            or not math.isfinite(float(posterior_probability))
            or not 0 <= float(posterior_probability) <= 1
        ):
            raise LiveProviderError(
                "credible set locus member posteriorProbability must be in [0, 1]"
            )
        log_bf = member.get("logBF")
        if log_bf is not None and (
            not isinstance(log_bf, (int, float))
            or isinstance(log_bf, bool)
            or not math.isfinite(float(log_bf))
        ):
            raise LiveProviderError(
                "credible set locus member logBF must be numeric or null"
            )
        for flag in ("is95CredibleSet", "is99CredibleSet"):
            if not isinstance(member.get(flag), bool):
                raise LiveProviderError(
                    f"credible set locus member {flag} must be boolean"
                )
        matched_members.append(
            {
                "variantId": member_variant_id,
                **{
                    key: member.get(key)
                    for key in (
                        "posteriorProbability",
                        "logBF",
                        "is95CredibleSet",
                        "is99CredibleSet",
                        "pValueMantissa",
                        "pValueExponent",
                        "beta",
                        "standardError",
                        "r2Overall",
                    )
                },
            }
        )
    if len(matched_members) != 1:
        raise LiveProviderError(
            "credible set locus must contain the query variant exactly once"
        )
    study_target_value = study.get("target")
    if study_target_value is None:
        study_target: dict[str, Any] = {}
    elif isinstance(study_target_value, dict):
        study_target = study_target_value
    else:
        raise LiveProviderError("credible set study target must be an object or null")
    study_biosample_value = study.get("biosample")
    if study_biosample_value is None:
        study_biosample: dict[str, Any] = {}
    elif isinstance(study_biosample_value, dict):
        study_biosample = study_biosample_value
    else:
        raise LiveProviderError("credible set study biosample must be an object or null")
    credible_qtl_gene_id = _optional_text(credible.get("qtlGeneId"))
    study_target_id = _optional_text(study_target.get("id"))
    if (
        credible_qtl_gene_id is not None
        and study_target_id is not None
        and credible_qtl_gene_id != study_target_id
    ):
        raise LiveProviderError(
            "credible set qtlGeneId conflicts with the study target identifier"
        )
    qtl_gene_id = credible_qtl_gene_id or study_target_id
    qtl_gene_symbol = _optional_text(study_target.get("approvedSymbol"))

    return {
        # Keep ``variantId`` as the live-provider compatibility alias while exposing
        # the same explicit query/lead contract used by the frozen provider.
        "variantId": variant_id,
        "queryVariantId": variant_id,
        "leadVariantId": lead_variant_id,
        "queryVariantRole": (
            "lead" if variant_id == lead_variant_id else "credible_set_member"
        ),
        "matchedLocusMember": matched_members[0],
        "locusCount": locus_count,
        "studyLocusId": credible.get("studyLocusId"),
        "studyId": credible.get("studyId") or study.get("id"),
        "studyType": credible.get("studyType") or study.get("studyType"),
        "chromosome": credible.get("chromosome"),
        "position": credible.get("position"),
        "beta": credible.get("beta"),
        "pValueMantissa": credible.get("pValueMantissa"),
        "pValueExponent": credible.get("pValueExponent"),
        "finemappingMethod": credible.get("finemappingMethod"),
        "confidence": credible.get("confidence"),
        "qualityControls": credible.get("qualityControls"),
        # Open Targets 26.06 may leave CredibleSet.qtlGeneId null while the
        # canonical QTL target is present at Study.target.  Preserve the direct
        # field when supplied and otherwise use that official typed relation.
        "qtlGeneId": qtl_gene_id,
        "qtlGeneSymbol": qtl_gene_symbol,
        "approvedSymbol": qtl_gene_symbol,
        "biosampleId": _optional_text(study_biosample.get("biosampleId")),
        "biosampleName": _optional_text(study_biosample.get("biosampleName")),
        "traitFromSource": study.get("traitFromSource"),
        "pubmedId": study.get("pubmedId"),
        "publicationDate": study.get("publicationDate"),
        "publicationTitle": study.get("publicationTitle"),
        "publicationFirstAuthor": study.get("publicationFirstAuthor"),
        "publicationJournal": study.get("publicationJournal"),
    }


def _validate_study_lineage(locus: dict[str, Any], study: dict[str, Any]) -> None:
    locus_study_id = _optional_text(locus.get("studyId"))
    metadata_study_id = _optional_text(study.get("id"))
    if (
        locus_study_id is not None
        and metadata_study_id is not None
        and locus_study_id != metadata_study_id
    ):
        raise LiveProviderError("credible-set studyId conflicts with its study metadata")
    _validate_study_type_agreement(locus.get("studyType"), study.get("studyType"))


def _validate_study_type_agreement(left: Any, right: Any) -> None:
    left_type = _optional_text(left)
    right_type = _optional_text(right)
    if (
        left_type is not None
        and right_type is not None
        and left_type.rsplit(".", 1)[-1].casefold()
        != right_type.rsplit(".", 1)[-1].casefold()
    ):
        raise LiveProviderError("credible-set studyType conflicts with its study metadata")


def _connection_rows(parent: dict[str, Any], field: str) -> list[dict[str, Any]]:
    if field not in parent:
        raise LiveProviderError(f"GraphQL response omitted requested connection {field!r}")
    connection = parent[field]
    if connection is None:
        return []
    if not isinstance(connection, dict):
        raise LiveProviderError(f"GraphQL connection {field!r} must be an object or null")
    if "rows" not in connection:
        raise LiveProviderError(f"GraphQL connection {field!r} omitted requested rows")
    rows = connection.get("rows")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise LiveProviderError(f"GraphQL connection {field!r}.rows must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise LiveProviderError(f"GraphQL connection {field!r}.rows[{index}] must be an object")
    return rows


def _requested_entity(
    data: dict[str, Any], field: str, expected_id: str
) -> dict[str, Any] | None:
    """Distinguish an explicit absent entity from an incomplete or misrouted response."""

    if field not in data:
        raise LiveProviderError(f"GraphQL response omitted requested field {field!r}")
    entity = data[field]
    if entity is None:
        return None
    if not isinstance(entity, dict):
        raise LiveProviderError(f"GraphQL field {field!r} must be an object or null")
    if _required_text(entity.get("id"), label=f"{field}.id") != expected_id:
        raise LiveProviderError(f"GraphQL field {field!r} id does not match its request")
    return entity


def _bounded_connection_coverage(
    parent: dict[str, Any], field: str, kind: EvidenceKind, entity_id: str
) -> ConnectionCoverage:
    rows = _connection_rows(parent, field)
    connection = parent[field]
    count = connection.get("count") if isinstance(connection, dict) else None
    if not isinstance(count, int) or isinstance(count, bool) or count < len(rows):
        raise LiveProviderError(f"GraphQL connection {field!r} count must cover its returned rows")
    cursor = _optional_text(connection.get("cursor")) if isinstance(connection, dict) else None
    return ConnectionCoverage(kind, entity_id, field, count, len(rows), cursor,
                              (connection.get("coverage") or {}).get("reason"))


def _variant_page_variables(
    variant_id: str,
    *,
    credible_requested: bool,
    enhancer_requested: bool,
    credible_page_index: int = 0,
    enhancer_page_index: int = 0,
) -> dict[str, Any]:
    variables: dict[str, Any] = {"variantId": variant_id}
    if credible_requested:
        variables.update(
            {
                "crediblePageIndex": credible_page_index,
                "crediblePageSize": _CREDIBLE_SET_PAGE_SIZE,
            }
        )
    if enhancer_requested:
        variables.update(
            {
                "enhancerPageIndex": enhancer_page_index,
                "enhancerPageSize": _ENHANCER_PAGE_SIZE,
            }
        )
    return variables


def _round_robin_continuations(
    continuation_pages: Mapping[int, int], *, limit: int
) -> list[tuple[int, int]]:
    """Allocate a global page budget fairly across selected variants."""

    continuations: list[tuple[int, int]] = []
    for page_index in range(1, max(continuation_pages.values(), default=0) + 1):
        for variant_index in sorted(continuation_pages):
            if continuation_pages[variant_index] < page_index:
                continue
            continuations.append((variant_index, page_index))
            if len(continuations) >= limit:
                return continuations
    return continuations


def _enhancer_page(
    variant: dict[str, Any],
    *,
    page_index: int,
    expected_count: int | None,
) -> tuple[int, list[dict[str, Any]]]:
    connection = variant.get("enhancerToGenes")
    if not isinstance(connection, dict):
        raise LiveProviderError("variant enhancerToGenes must be an object")
    count = connection.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise LiveProviderError("enhancerToGenes count must be a non-negative integer")
    if expected_count is not None and count != expected_count:
        raise LiveProviderError(
            "enhancerToGenes declared count changed between pagination requests"
        )
    rows = _connection_rows(variant, "enhancerToGenes")
    remaining = max(count - page_index * _ENHANCER_PAGE_SIZE, 0)
    expected_page_rows = min(_ENHANCER_PAGE_SIZE, remaining)
    if len(rows) != expected_page_rows:
        raise LiveProviderError(
            "enhancerToGenes page row count does not match its declared total and page index"
        )
    return count, rows


def _validate_variant_page_identity(
    variant: dict[str, Any], expected_variant_id: str
) -> None:
    returned_variant_id = _required_text(variant.get("id"), label="variant.id")
    if returned_variant_id != expected_variant_id:
        raise LiveProviderError(
            "credibleSets page returned a variant that does not match its request"
        )


def _credible_set_page(
    variant: dict[str, Any],
    *,
    page_index: int,
    expected_count: int | None,
) -> tuple[int, list[dict[str, Any]]]:
    connection = variant.get("credibleSets")
    if not isinstance(connection, dict):
        raise LiveProviderError("variant credibleSets must be an object")
    count = connection.get("count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
    ):
        raise LiveProviderError("credibleSets count must be a non-negative integer")
    if expected_count is not None and count != expected_count:
        raise LiveProviderError(
            "credibleSets declared count changed between pagination requests"
        )
    rows = _connection_rows(variant, "credibleSets")
    remaining = max(count - page_index * _CREDIBLE_SET_PAGE_SIZE, 0)
    expected_page_rows = min(_CREDIBLE_SET_PAGE_SIZE, remaining)
    if len(rows) != expected_page_rows:
        raise LiveProviderError(
            "credibleSets page row count does not match its declared total and page index"
        )
    return count, rows


def _credible_set_row_ids(
    rows: list[dict[str, Any]], *, variant_id: str
) -> set[str]:
    row_ids = {
        _required_text(
            row.get("studyLocusId"),
            label=f"credibleSets row for {variant_id} studyLocusId",
        )
        for row in rows
    }
    if len(row_ids) != len(rows):
        raise LiveProviderError(
            "credibleSets page returned duplicate studyLocusId values"
        )
    return row_ids


def _unwrap_query_result(value: dict[str, Any]) -> dict[str, Any]:
    status = value.get("status")
    if not isinstance(status, str):
        raise LiveProviderError("GraphQL query result is missing status")
    if status.casefold() != "success":
        raise LiveProviderError(
            f"official Open Targets GraphQL query returned status {status!r}"
        )
    data = value.get("data")
    if not isinstance(data, dict):
        raise LiveProviderError("successful GraphQL query result is missing its data object")
    return data


def _unwrap_optional_success(value: dict[str, Any], *, label: str) -> dict[str, Any]:
    if "status" not in value:
        return value
    status = value.get("status")
    if not isinstance(status, str) or status.casefold() != "success":
        raise LiveProviderError(f"{label} returned non-success status {status!r}")
    data = value.get("data")
    if not isinstance(data, dict):
        raise LiveProviderError(f"successful {label} response is missing its data object")
    return data


def _direct_entity_id(value: str, *, entity_type: str) -> str | None:
    text = value.strip()
    if entity_type == "target" and _TARGET_ID.fullmatch(text):
        return text.split(".", 1)[0].upper()
    if entity_type == "variant":
        if text.casefold().startswith("chr"):
            text = text[3:]
        text = text.replace(":", "_")
        if _VARIANT_ID.fullmatch(text):
            return text.upper()
    return None


def _search_hit_matches_query(query: str, hit: Mapping[str, Any]) -> bool:
    expected = query.strip().casefold()
    for key in ("id", "name", "label", "approvedSymbol", "symbol", "rsId", "rsIds"):
        value = hit.get(key)
        candidates = value if isinstance(value, (list, tuple)) else (value,)
        if any(
            candidate is not None
            and str(candidate).strip().casefold() == expected
            for candidate in candidates
        ):
            return True
    return False


def _required_text(value: Any, *, label: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise LiveProviderError(f"{label} must be a non-empty scalar")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise LiveProviderError("identifier fields must be scalar values")
    text = str(value).strip()
    return text or None


@asynccontextmanager
async def _transport_session(transport: MCPToolTransport):
    scope = getattr(transport, "session_scope", None)
    if callable(scope):
        async with scope():
            yield transport
        return
    yield transport


def _validated_http_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("GraphQL endpoint must be a non-empty URL")
    value = endpoint.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("GraphQL endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("GraphQL endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("GraphQL endpoint must not contain a query or fragment")
    return value


def _reject_unknown_arguments(
    arguments: dict[str, Any], *, allowed: set[str], label: str
) -> None:
    unknown = set(arguments).difference(allowed)
    if unknown:
        raise LiveProviderError(
            f"{label} received unsupported arguments: {', '.join(sorted(unknown))}"
        )


def _graphql_query_string(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiveProviderError("GraphQL query_string must be a non-empty string")
    query = value.strip()
    if len(query) > _DEFAULT_GRAPHQL_MAX_QUERY_CHARS:
        raise LiveProviderError("GraphQL query_string exceeds the configured limit")
    if re.search(r"\b(?:mutation|subscription)\b", query, flags=re.IGNORECASE):
        raise LiveProviderError("official GraphQL fallback permits read-only queries only")
    return query


def _graphql_variables(value: Any) -> dict[str, Any]:
    normalized = _json_value(value, label="GraphQL variables")
    if not isinstance(normalized, dict):
        raise LiveProviderError("GraphQL variables must be a JSON object")
    return normalized


def _graphql_error_summary(value: Any) -> str:
    items = value if isinstance(value, list) else [value]
    messages: list[str] = []
    for item in items[:3]:
        if isinstance(item, Mapping):
            message = item.get("message")
        else:
            message = item
        text = " ".join(str(message or "unknown GraphQL error").split())[:300]
        messages.append(text)
    return "; ".join(messages) or "unknown GraphQL error"


def _bounded_graphql_http_error(error: HTTPError, *, limit: int) -> str | None:
    try:
        raw = error.read(limit + 1)
    except OSError:
        return None
    finally:
        error.close()
    if len(raw) > limit:
        return "error body exceeded the diagnostic size limit"
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(value, Mapping) and value.get("errors"):
        return _graphql_error_summary(value["errors"])
    return None


def _parse_date(value: Any) -> date | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise LiveProviderError(f"invalid ISO evidence date: {text!r}") from exc


def _evidence_id(dataset: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(f"{dataset}:{raw}".encode("utf-8")).hexdigest()[:20]
    return f"ot:{dataset}:{digest}"


def _json_value(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LiveProviderError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LiveProviderError(f"{label} contains a non-string object key")
            result[key] = _json_value(item, label=label)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label=label) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(by_alias=True), label=label)
    raise LiveProviderError(f"{label} contains non-JSON value {type(value).__name__}")


def _decode_mcp_result(value: Any, *, tool: str) -> Any:
    is_error = _object_field(value, "is_error", "isError")
    if is_error is not _MISSING and bool(is_error):
        raise LiveProviderError(f"official MCP tool {tool!r} returned an error result")

    structured = _object_field(value, "structured_content", "structuredContent")
    if structured is not _MISSING and structured is not None:
        return structured

    content = _object_field(value, "content")
    if not isinstance(content, list):
        raise LiveProviderError(f"official MCP tool {tool!r} returned no structured JSON")
    decoded: list[Any] = []
    for block in content:
        text = _object_field(block, "text")
        if isinstance(text, str):
            try:
                decoded.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise LiveProviderError(
                    f"official MCP tool {tool!r} returned non-JSON text"
                ) from exc
    if len(decoded) != 1:
        raise LiveProviderError(
            f"official MCP tool {tool!r} returned {len(decoded)} JSON payloads"
        )
    return decoded[0]


_MISSING = object()


def _object_field(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return _MISSING


def _mapping_value_case_insensitive(value: dict[str, Any], key: str) -> Any:
    if key in value:
        return value[key]
    folded = key.casefold()
    for candidate, item in value.items():
        if candidate.casefold() == folded:
            return item
    return _MISSING
