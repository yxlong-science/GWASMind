"""Run-scoped, content-bound request pages shared by semantic modules.

Evicting the small memory index never deletes completed pages on disk. Failed
requests retain their exhausted attempt budget so another module cannot restart
the same operation invisibly. A new run receives a new cache directory.
"""
from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import threading
from uuid import uuid4

from ..configuration import parameter
from .retrieval_recovery import safe_transport_failure

ACTIVE_REQUEST_KEY = ContextVar("active_open_targets_request_key", default=None)


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class RequestPageStore:
    def __init__(self, directory=None, *, release="live"):
        self.directory = Path(directory).resolve() if directory is not None else None
        self.release = release
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._memory = OrderedDict()
        self._lock = threading.RLock()
        self._attempts = {}
        self._failure_history = {}
        self._observed = {}
        self._stats = {"network_requests": 0, "cache_hits": 0, "failed_cache_hits": 0,
                       "completed_pages": 0, "failed_pages": 0, "unique_request_pages": 0}

    def key(self, endpoint, query, variables):
        return hashlib.sha256(_encoded({"endpoint": endpoint, "release": self.release,
                                       "query": query, "variables": variables})).hexdigest()

    def _remember(self, key, encoded):
        self._memory[key] = encoded
        self._memory.move_to_end(key)
        while len(self._memory) > parameter("retrieval.entity_cache_memory_items"):
            self._memory.popitem(last=False)

    def _observe(self, key, status):
        prior = self._observed.get(key)
        if prior == status:
            return
        if prior is not None:
            self._stats["completed_pages" if prior == "complete" else "failed_pages"] -= 1
        else:
            self._stats["unique_request_pages"] += 1
        self._observed[key] = status
        self._stats["completed_pages" if status == "complete" else "failed_pages"] += 1

    def get(self, key):
        with self._lock:
            encoded = self._memory.get(key)
            if encoded is None and self.directory is not None:
                path = self.directory / (key + ".json")
                if path.exists():
                    # Cache envelopes contain a bounded transport response plus
                    # small metadata; a damaged file is never loaded unbounded.
                    limit = (parameter("transport.graphql_max_response_bytes")
                             + parameter("transport.graphql_max_request_bytes"))
                    with path.open("rb") as source:
                        encoded = source.read(limit + 1)
                    if len(encoded) > limit:
                        raise ValueError("Cached request page exceeds the transport envelope limit")
            if encoded is None:
                return None
            value = json.loads(encoded)
            if value.get("key") != key or value.get("schema") != 1:
                raise ValueError("Cached request page identity is invalid")
            if value.get("status") not in {"complete", "failed"}:
                raise ValueError("Cached request page status is invalid")
            if "request" in value and hashlib.sha256(_encoded(value["request"])).hexdigest() != key:
                raise ValueError("Cached request metadata digest is invalid")
            if value["status"] == "complete" and hashlib.sha256(_encoded(value["data"])).hexdigest() != value.get("data_sha256"):
                raise ValueError("Cached request page content digest is invalid")
            self._remember(key, encoded)
            self._observe(key, value["status"])
            self._stats["cache_hits"] += 1
            if value["status"] == "failed":
                self._stats["failed_cache_hits"] += 1
            return value

    def note_network_request(self):
        key = ACTIVE_REQUEST_KEY.get()
        with self._lock:
            self._stats["network_requests"] += 1
            if key:
                self._attempts[key] = self._attempts.get(key, 0) + 1

    def note_failure(self, error):
        key = ACTIVE_REQUEST_KEY.get()
        detail = safe_transport_failure(error)
        if key is None or not detail:
            return
        with self._lock:
            attempt = self._attempts.get(key, 0)
            if not attempt:
                return
            history = self._failure_history.setdefault(key, [])
            value = {"network_attempt": attempt, **detail}
            if not history or history[-1].get("network_attempt") != attempt:
                history.append(value)

    def save(self, key, *, data=None, response_bytes=0, error=None, request=None, recovery_status=None):
        status = "complete" if error is None else "failed"
        with self._lock:
            value = {"schema": 1, "key": key, "status": status,
                     "network_attempts": self._attempts.get(key, 0), "response_bytes": response_bytes}
            if key in self._failure_history:
                value["failure_history"] = list(self._failure_history[key])
            if request is not None:
                if hashlib.sha256(_encoded(request)).hexdigest() != key:
                    raise ValueError("Request metadata does not match its page key")
                value["request"] = request
            if error is None:
                value.update(data=data, data_sha256=hashlib.sha256(_encoded(data)).hexdigest())
            else:
                value["error"] = str(error)[:2000]
                details = safe_transport_failure(error)
                if details:
                    value["failure_details"] = details
            if recovery_status is not None:
                value["recovery_status"] = recovery_status
            encoded = _encoded(value)
            if self.directory is not None:
                path = self.directory / (key + ".json")
                temporary = self.directory / (key + "." + uuid4().hex + ".tmp")
                try:
                    temporary.write_bytes(encoded)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            self._remember(key, encoded)
            self._observe(key, status)

    @staticmethod
    def recovery_request(request, original_key):
        value = {**request, "recovery_of": original_key, "recovery_attempt": 1}
        return hashlib.sha256(_encoded(value)).hexdigest(), value

    def reserve_recovery(self, key, request):
        """Persist the one allowed recovery before starting its transport work."""
        with self._lock:
            if self.directory is not None:
                # Exclusive creation also protects separate cache instances or
                # processes. A crash before saving the reservation cannot reset
                # the allowance or overwrite another recovery's successful page.
                claim = self.directory / (key + ".recovery-claim")
                try:
                    with claim.open("xb") as handle:
                        handle.write(_encoded({"key": key, "recovery_of": request["recovery_of"], "attempt": 1}))
                except FileExistsError:
                    return False
            # Claim before reading the reservation page. A competing Windows
            # process may still be atomically publishing that page; the claim
            # already proves the recovery allowance is owned, without reopening
            # a file in the middle of its replacement.
            if self.get(key) is not None:
                return False
            error = RuntimeError("The bounded entity recovery was interrupted before a complete response was saved.")
            error.failure_details = {"category": "recovery_interrupted", "retryable": False}
            self.save(key, error=error, request=request, recovery_status="reserved")
            return True

    def summary(self):
        with self._lock:
            # Rejected larger requests remain in diagnostics even when smaller
            # requests recovered every scientific record. These are historical
            # request failures, never a count of unresolved scientific gaps.
            return {**self._stats, "failed_request_pages": self._stats["failed_pages"]}
