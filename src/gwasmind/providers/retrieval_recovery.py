"""Small, shared recovery state for one semantic request; no scientific policy."""
from __future__ import annotations

from contextvars import ContextVar
from http.client import HTTPException
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import re
from urllib.error import HTTPError, URLError

from ..progress import emit_progress

RECOVERY_SINK = ContextVar("retrieval_recovery_sink", default=None)
ANALYSIS_VARIANTS = ContextVar("retrieval_analysis_variants", default=())
ENTITY_RECOVERY_ACTIVE = ContextVar("entity_recovery_active", default=False)
RETRIEVAL_FAILURE_RECEIPTS = ContextVar("retrieval_failure_receipts", default=None)
ACTIVE_SEMANTIC_TOOL = ContextVar("active_semantic_tool", default=None)
ACTIVE_OPERATION_ID = ContextVar("active_retrieval_operation_id", default=None)


def record_transport_outcome(query, variables, details=None):
    """Retain terminal bounded failures; a verified success clears that request."""
    from ..audit import sha256_json
    receipts = RETRIEVAL_FAILURE_RECEIPTS.get()
    if receipts is None:
        return
    key = sha256_json({"query": query, "variables": variables})
    receipts[:] = [row for row in receipts if row["request_sha256"] != key]
    if isinstance(details, dict) and details.get("recovery_exhausted") is True:
        receipts.append({"code": "required_query_failed", "state": "FAILED",
                         "tool": ACTIVE_SEMANTIC_TOOL.get(), "request_sha256": key,
                         "details": dict(details)})


def safe_transport_failure(exc):
    """Only proven exception categories; never copy URLs, headers or error bodies."""
    allowed = {"request_timeout", "connection_error", "transport_error", "rate_limited",
               "service_error", "http_client_error", "recovery_interrupted"}
    pending, seen, generic = [exc], set(), None
    while pending and len(seen) < 8:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        value = getattr(current, "failure_details", None)
        if isinstance(value, dict) and value.get("category") in allowed and type(value.get("retryable")) is bool:
            clean = {"category": value["category"], "retryable": value["retryable"]}
            status = value.get("http_status")
            if type(status) is int and 100 <= status <= 599:
                clean["http_status"] = status
            attempts = value.get("attempts")
            if type(attempts) is int and attempts > 0:
                clean["attempts"] = attempts
            if type(value.get("recovery_exhausted")) is bool:
                clean["recovery_exhausted"] = value["recovery_exhausted"]
            if clean["category"] != "transport_error":
                return clean
            generic = clean
        if isinstance(current, HTTPError):
            status = current.code
            return {"category": "rate_limited" if status == 429 else "service_error" if status >= 500 else "http_client_error",
                    "http_status": status, "retryable": status in {429, 500, 502, 503, 504}}
        name = type(current).__name__
        module = type(current).__module__
        if isinstance(current, TimeoutError) or (module.startswith(("httpx", "httpcore"))
                and name in {"TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"}):
            return {"category": "request_timeout", "retryable": True}
        if isinstance(current, URLError):
            pending.insert(0, current.reason)
            generic = {"category": "connection_error", "retryable": True}
        elif isinstance(current, (ConnectionError, HTTPException)) or (module.startswith(("httpx", "httpcore"))
                and name in {"ConnectError", "ReadError", "WriteError", "RemoteProtocolError", "ProxyError"}):
            return {"category": "connection_error", "retryable": True}
        cause = getattr(current, "__cause__", None)
        if cause is not None:
            pending.append(cause)
    if generic is not None:
        return generic
    # Old cache receipts preserved this exact locally generated message but
    # discarded its cause. Recover the transport class, never invent a subtype.
    if re.fullmatch(r"official GraphQL transport failed after [1-9][0-9]* bounded attempt\(s\)", str(exc)):
        return {"category": "transport_error", "retryable": True}
    return {}


def recovery_progress(channel, status, **details):
    operation_id = ACTIVE_OPERATION_ID.get()
    if operation_id:
        details = {**details, "operation_id": operation_id, "tool": ACTIVE_SEMANTIC_TOOL.get()}
    emit_progress(RECOVERY_SINK.get(), "retrieval_recovery", channel=channel,
                  status=status, **details)


def failure_category(exc):
    detail = safe_transport_failure(exc)
    if detail and detail["category"] not in {"http_client_error", "transport_error"}:
        return detail["category"]
    parts = []
    current = exc
    for _ in range(4):
        if current is None:
            break
        parts.append(str(current))
        current = current.__cause__
    text = " ".join(parts).casefold()
    if "too expensive" in text or "complexity" in text:
        return "query_too_expensive"
    if "size limit" in text or "response-size limit" in text:
        return "response_size_limit"
    if "budget exhausted" in text:
        return "continuation_budget_exhausted"
    if "429" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "request_timeout"
    if "transport" in text or "request failed" in text:
        return "connection_error"
    return "invalid_or_unavailable_response"


def retry_after_seconds(headers):
    """Accept HTTP delta-seconds or date, never unbounded or negative waits."""
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(value))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            seconds = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None
