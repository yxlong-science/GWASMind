"""Public, key-free GWAS Catalog / Ensembl LD transport.

Only variant identifiers, phenotype terms and reference-population codes leave
the process. No uploaded effect sizes, individual data, files or credentials are
sent. Every response is hashed and timestamped in the returned query ledger.
Catalog search uses exact ontology labels (no child-trait expansion), current
mapped GRCh38 positions, and complete pagination within configured limits.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import threading
import time
from collections.abc import Mapping
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from ..configuration import parameter
from ..progress import emit_progress
from .retrieval_recovery import retry_after_seconds
from .ld_cache import LDResponseCache

_RATE_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
_IDENTITY_ERRORS = (RuntimeError, ValueError, OSError, TypeError, KeyError, AttributeError)
RECOVERABLE_LD_REASONS = frozenset({"rate_limited", "service_error", "request_timeout", "connection_error", "total_budget_exhausted"})


def reference_failure_reason(exc):
    recorded = getattr(exc, "reason_code", None)
    if recorded:
        return recorded
    cause = exc
    while cause.__cause__ is not None:
        cause = cause.__cause__
    if isinstance(cause, HTTPError):
        return "rate_limited" if cause.code == 429 else "service_error" if cause.code >= 500 else "invalid_response"
    if isinstance(cause, TimeoutError) or isinstance(getattr(cause, "reason", None), TimeoutError):
        return "request_timeout"
    if isinstance(cause, URLError):
        return "connection_error"
    return "invalid_response"


def _reference_error(message, reason):
    error = RuntimeError(message)
    error.reason_code = reason
    return error


def _population_response_rows(value, requested_rsid):
    """Reject a contradictory named response; retain legacy unnamed row payloads."""
    if not isinstance(value, Mapping):
        raise ValueError("Population alleles are not verified: malformed reference response.")
    if "name" in value and value["name"] != requested_rsid:
        raise ValueError("Population response does not identify the requested rsID.")
    rows = value.get("populations")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Population alleles are not verified: missing or malformed population rows.")
    return rows


class LDReferenceProvider:
    """One screen's transport and provenance; test providers may replace this class."""

    def __init__(self, *, fetch=None, progress=None, progress_sink=None, cancel_event=None, cache_directory=None,
                 total_timeout_seconds=None):
        self._fetch = fetch
        self.supports_batch_identity = fetch is None
        if fetch is not None:
            try:
                inspect.signature(fetch).bind("https://example.invalid", {"ids": []})
                self.supports_batch_identity = True
            except (TypeError, ValueError):
                # Historical GET-only injected fixtures keep their contract.
                pass
        self.ledger = []
        self._cache = {}
        self._failed_requests, self._receipts = {}, {}
        self._variations, self._variation_failures = {}, {}
        self.identity_failure_details = {}
        self.source_identity_failure_details = {}
        self._source_candidate_cache = {}
        self._state_lock = threading.RLock()
        self._request_locks = {}
        self.response_cache = LDResponseCache(cache_directory) if cache_directory is not None else None
        self.request_summary = {"network_requests": 0, "cache_hits": 0, "failed_cache_hits": 0,
                                "batch_requests": 0, "identity_metadata_reuses": 0, "invalid_cache_entries": 0}
        self.progress = progress
        self.progress_sink = progress_sink
        self.cancel_event = cancel_event
        self.progress_context = {}
        self.owner_thread = threading.get_ident()
        timeout = parameter("novelty.total_timeout_seconds") if total_timeout_seconds is None else total_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("LD total timeout must be a positive finite number of seconds.")
        self.deadline = time.monotonic() + timeout
        self.ensembl = parameter("novelty.ensembl_rest_url").rstrip("/")
        self.catalog = parameter("novelty.catalog_rest_url").rstrip("/")

    def get(self, base, path, **params):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CancelledError()
        parsed = urlsplit(base)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("The public LD endpoint must be HTTPS without credentials.")
        query = {key: value for key, value in params.items() if key != "_body"}
        request = LDResponseCache.request(base + path + "?" + urlencode(query), params.get("_body"))
        key = LDResponseCache.key(request)
        with self._state_lock:
            lock = (self.response_cache.lock(key) if self.response_cache else
                    self._request_locks.setdefault(key, threading.RLock()))
        with lock:
            if key in self._failed_requests:
                self.request_summary["failed_cache_hits"] += 1
                raise self._failed_requests[key]
            if key in self._receipts:
                self.request_summary["cache_hits"] += 1
                self.ledger.append({**self._receipts[key], "from_cache": True,
                                    "observed_at": datetime.now(timezone.utc).isoformat()})
                return self._cache[key]
            try:
                if self.response_cache is not None:
                    try:
                        cached = self.response_cache.get(request)
                    except (OSError, ValueError, KeyError, TypeError):
                        # No cached scientific content is used after a failed
                        # integrity check. A fresh bounded request can repair it.
                        self.request_summary["invalid_cache_entries"] += 1
                        cached = None
                    if cached is not None:
                        value, receipt = cached
                        self.request_summary["cache_hits"] += 1
                        saved = {"url": request["url"], "method": request["method"],
                                 "request_body": request["body"], "status": "success", "from_cache": True,
                                 "observed_at": datetime.now(timezone.utc).isoformat(),
                                 **{name: receipt[name] for name in ("retrieved_at", "sha256", "bytes", "request_key", "response_path")}}
                        self.ledger.append(saved)
                        self._receipts[key], self._cache[key] = saved, value
                        return value
                value = self._get_uncached(base, path, _request=request, _request_key=key, **params)
                self._cache[key] = value
                return value
            except (RuntimeError, ValueError, OSError) as exc:
                # Failure reuse is per provider/request only, never persisted
                # as a negative result for another analysis.
                self._failed_requests[key] = exc
                raise

    def _get_uncached(self, base, path, *, _request, _request_key, **params):
        global _LAST_REQUEST
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CancelledError()
        body_value = params.pop("_body", None)
        request_body = json.dumps(body_value, separators=(",", ":")).encode() if body_value is not None else None
        url = base + path + "?" + urlencode(params)
        phase = "catalog_request" if base == self.catalog else "ld_reference_request"
        operation_id = uuid4().hex
        def notify(status, **details):
            if threading.get_ident() == self.owner_thread and not (self.cancel_event and self.cancel_event.is_set()):
                emit_progress(self.progress_sink, phase, **self.progress_context, status=status, operation_id=operation_id, **details)
        if self.progress and threading.get_ident() == self.owner_thread:
            label = "GWAS Catalog" if base == self.catalog else "LD reference"
            self.progress(f"{label}: checking retrieved associations and variant identity" +
                          (f" (page {params['page'] + 1})" if "page" in params else ""))
        # Endpoint configuration may choose a release archive, never credentials
        # in URLs. No redirects to arbitrary paging links are followed by callers.
        parsed = urlsplit(base)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("The public LD endpoint must be HTTPS without credentials.")
        for attempt in range(parameter("novelty.ld_max_retries") + 1):
            try:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise CancelledError()
                notify("started" if attempt == 0 else "retrying", attempt=attempt + 1)
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise _reference_error("The configured total LD screening time budget was reached.", "total_budget_exhausted")
                with _RATE_LOCK:
                    delay = parameter("novelty.http_min_interval_seconds") - (time.monotonic() - _LAST_REQUEST)
                    if delay > 0:
                        time.sleep(delay)
                    _LAST_REQUEST = time.monotonic()
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise CancelledError()
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise _reference_error("The configured total LD screening time budget was reached.", "total_budget_exhausted")
                started_at = datetime.now(timezone.utc).isoformat()
                started = time.monotonic()
                with self._state_lock:
                    self.request_summary["network_requests"] += 1
                    self.request_summary["batch_requests"] += int(body_value is not None)
                if self._fetch:
                    value = self._fetch(url, body_value) if body_value is not None else self._fetch(url)
                    body = json.dumps(value, sort_keys=True).encode()
                else:
                    request = Request(url, data=request_body, headers={"Accept": "application/json",
                                                    "Content-Type": "application/json",
                                                    "User-Agent": "GWASMind-LD/1.0"})
                    with urlopen(request, timeout=min(parameter("novelty.ld_timeout_seconds"), remaining)) as response:
                        # Reuse the canonical bounded-response transport limit.
                        cap = parameter("transport.graphql_max_response_bytes")
                        body = response.read(cap + 1)
                        if len(body) > cap:
                            raise ValueError("LD response exceeded the configured transport limit.")
                    value = json.loads(body)
                receipt = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                    "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                                    "attempt": attempt + 1, "status": "success",
                                    "method": "POST" if request_body else "GET",
                                    "request_body": body_value, "request_key": _request_key, "from_cache": False,
                                    "started_at": started_at, "elapsed_seconds": time.monotonic() - started}
                if self.response_cache is not None:
                    try:
                        stored = self.response_cache.save(_request, body, retrieved_at=receipt["retrieved_at"])
                    except OSError as exc:
                        raise RuntimeError("The local LD response artifact could not be saved.") from exc
                    receipt["response_path"] = stored["response_path"]
                self.ledger.append(receipt)
                self._receipts[_request_key] = receipt
                notify("completed", attempt=attempt + 1)
                return value
            except (OSError, ValueError, URLError) as exc:
                reason = ("rate_limited" if isinstance(exc, HTTPError) and exc.code == 429 else
                          "service_error" if isinstance(exc, HTTPError) else
                          "request_timeout" if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError) else
                          "invalid_response" if isinstance(exc, ValueError) else "connection_error")
                self.ledger.append({"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                    "attempt": attempt + 1, "status": "failed",
                                    "error_type": type(exc).__name__,
                                    "http_status": exc.code if isinstance(exc, HTTPError) else None,
                                    "method": "POST" if request_body else "GET", "request_body": body_value,
                                    "from_cache": False, "request_key": _request_key,
                                    "started_at": started_at, "elapsed_seconds": time.monotonic() - started})
                if (attempt == parameter("novelty.ld_max_retries")
                        or isinstance(exc, ValueError)
                        or isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code != 429):
                    notify("incomplete", attempt=attempt + 1, reason=reason)
                    raise _reference_error("A required public LD/catalog lookup did not complete.", reference_failure_reason(exc)) from exc
                delay = max(parameter("novelty.http_min_interval_seconds") * (2 ** attempt),
                            (retry_after_seconds(exc.headers) or 0) if isinstance(exc, HTTPError) else 0)
                remaining = self.deadline - time.monotonic()
                if delay >= remaining:
                    notify("incomplete", attempt=attempt + 1, reason="cooldown_exceeds_budget")
                    raise _reference_error("Public LD/catalog retry cooldown exceeds the remaining screening budget.", "total_budget_exhausted") from exc
                notify("retrying", attempt=attempt + 2, reason=reason, wait_seconds=delay)
                if self.cancel_event is not None:
                    if self.cancel_event.wait(delay):
                        raise CancelledError()
                else:
                    time.sleep(delay)
        raise RuntimeError("LD request failed.")

    def variation(self, rsid, *, populations=False):
        entry = self._variations.get(rsid)
        if entry is not None and (not populations or entry["populations_requested"]):
            self.request_summary["identity_metadata_reuses"] += 1
            return entry["value"]
        if (rsid, populations) in self._variation_failures:
            raise RuntimeError("The shared reference identity request did not complete.")
        return self.get(self.ensembl, f"/variation/human/{rsid}",
                        **({"pops": 1} if populations else {}), **{"content-type": "application/json"})

    def preload_variations(self, rsids, *, identity_scope, populations=True):
        """Reuse exact metadata at the required capability; upgrade only when needed."""
        def available(rsid):
            entry = self._variations.get(rsid)
            return entry is not None and (not populations or entry["populations_requested"])
        missing = sorted(rsid for rsid in set(rsids)
                         if not available(rsid) and (rsid, populations) not in self._variation_failures)
        if self.response_cache is not None:
            reused_parents = set()
            for rsid in missing:
                cached = None
                for cached_populations in ([True] if populations else [False, True]):
                    try:
                        cached = self.response_cache.get_variation(self.ensembl, rsid, populations=cached_populations)
                    except (OSError, ValueError, KeyError, TypeError):
                        self.request_summary["invalid_cache_entries"] += 1
                    if cached is not None:
                        break
                if cached is None:
                    continue
                value, receipt = cached
                parent = receipt["request"]
                saved = {"url": parent["url"], "method": "POST", "request_body": parent["body"],
                         "status": "success", "from_cache": True, "observed_at": datetime.now(timezone.utc).isoformat(),
                         **{key: receipt[key] for key in ("retrieved_at", "sha256", "bytes", "request_key", "response_path")}}
                if saved["request_key"] not in reused_parents:
                    self.ledger.append(saved)
                    reused_parents.add(saved["request_key"])
                self._variations[rsid] = {"value": value, "populations_requested": cached_populations, "parent_receipt": saved}
                self.request_summary["cache_hits"] += 1
            missing = [rsid for rsid in missing if not available(rsid)]
        size = parameter("retrieval.ld_population_batch_size" if populations else "retrieval.ld_identity_batch_size")
        params = {**({"pops": 1} if populations else {}), "content-type": "application/json"}
        completed = 0
        batch_progress_lock = threading.Lock()
        def group(ids, batch_index, batch_count):
            nonlocal completed
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise CancelledError()
            with batch_progress_lock:
                details = {"identity_scope": identity_scope, "populations_requested": populations,
                           "batch_index": batch_index, "batch_count": batch_count,
                           "variant_count": len(ids), "completed_count": completed, "total_count": len(missing)}
                emit_progress(self.progress_sink, "ld_identity_batch", status="started", **details, **self.request_summary)
            try:
                body = {"ids": ids}
                response = self.get(self.ensembl, "/variation/human", _body=body, **params)
                if not isinstance(response, Mapping):
                    raise ValueError("Reference identity batch was not an object.")
                request = LDResponseCache.request(self.ensembl + "/variation/human?" +
                    urlencode(params), body)
                receipt = self._receipts.get(LDResponseCache.key(request), {})
                for rsid in ids:
                    value = response.get(rsid)
                    if isinstance(value, Mapping) and value.get("name") == rsid:
                        self._variations[rsid] = {"value": value, "populations_requested": populations,
                                                 "parent_receipt": receipt}
                    else:
                        self._variation_failures[(rsid, populations)] = "missing_reference_identity"
                        self.identity_failure_details[(rsid, populations)] = {"identity_scope": identity_scope, "reason": "missing_reference_identity"}
            except (RuntimeError, ValueError, OSError) as exc:
                cause = exc
                while cause.__cause__ is not None:
                    cause = cause.__cause__
                status = cause.code if isinstance(cause, HTTPError) else None
                splittable = (status in {413, 429, 500, 502, 503, 504} or
                             isinstance(cause, (TimeoutError, URLError)) or
                             isinstance(cause, ValueError) and "transport limit" in str(cause))
                delay = retry_after_seconds(cause.headers) if status == 429 else 0
                remaining = self.deadline - time.monotonic()
                if splittable and len(ids) > 1 and remaining > (delay or 0):
                    if delay:
                        if self.cancel_event is not None:
                            if self.cancel_event.wait(delay): raise CancelledError()
                        else: time.sleep(delay)
                    emit_progress(self.progress_sink, "ld_identity_batch", status="splitting",
                                  reason="rate_limited" if status == 429 else "reference_batch_unavailable", **details)
                    middle = len(ids) // 2
                    group(ids[:middle], batch_index, batch_count)
                    group(ids[middle:], batch_index, batch_count)
                    return
                for rsid in ids:
                    self._variation_failures[(rsid, populations)] = "reference_identity_unavailable"
                    self.identity_failure_details[(rsid, populations)] = {"identity_scope": identity_scope, "reason": reference_failure_reason(exc)}
            with batch_progress_lock:
                completed += len(ids)
                emit_progress(self.progress_sink, "ld_identity_batch", status="completed" if all(available(rsid) for rsid in ids) else "incomplete",
                              **{**details, "completed_count": completed},
                              returned_count=sum(available(rsid) for rsid in missing), **self.request_summary)
        batches = [missing[start:start + size] for start in range(0, len(missing), size)]
        if identity_scope == "proxy" and populations and len(batches) > 1:
            # Fixed disjoint population batches share the existing request locks,
            # reference cache, rate limiter and total deadline. Recursive retry
            # splits stay sequential inside each worker, so they cannot multiply
            # the configured concurrency or alter the original request set.
            workers = min(len(batches), parameter("ranking.ld_identity_concurrency"))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = [pool.submit(group, ids, index + 1, len(batches)) for index, ids in enumerate(batches)]
                try:
                    for future in as_completed(pending):
                        if self.cancel_event is not None and self.cancel_event.is_set():
                            raise CancelledError()
                        future.result()
                finally:
                    for future in pending:
                        future.cancel()
        else:
            for index, ids in enumerate(batches):
                group(ids, index + 1, len(batches))

    def resolve_many(self, source_ids):
        """Discover candidates independently, then reuse batched exact mappings."""
        candidates, results = {}, {}
        self._prefetch_source_regions(source_ids)
        with ThreadPoolExecutor(max_workers=parameter("ranking.ld_identity_concurrency")) as pool:
            pending = {pool.submit(self._source_candidates, source_id): source_id for source_id in source_ids}
            for future in as_completed(pending):
                source_id = pending[future]
                try:
                    candidates[source_id] = future.result()[0]
                except _IDENTITY_ERRORS as exc:
                    results[source_id] = exc
                emit_progress(self.progress_sink, "ld_source_discovery", status="running",
                              completed_count=len(candidates) + len(results), total_count=len(source_ids),
                              **self.request_summary)
        emit_progress(self.progress_sink, "ld_source_discovery", status="incomplete" if results else "completed",
                      completed_count=len(source_ids), total_count=len(source_ids), **self.request_summary)
        self.preload_variations([rsid for ids in candidates.values() for rsid in ids], identity_scope="source")
        for source_id in source_ids:
            if source_id in results: continue
            try:
                if any(rsid not in self._variations for rsid in candidates[source_id]):
                    # An exhausted population batch also lacks its basic mapping.
                    # Do not restart its transport budget with individual GETs.
                    reasons = [self.identity_failure_details.get((rsid, True), {}).get("reason") for rsid in candidates[source_id]]
                    reason = next((value for value in reasons if value in RECOVERABLE_LD_REASONS), "invalid_response")
                    raise _reference_error("The shared source identity request did not complete.", reason)
                results[source_id] = self.resolve(source_id)
            except _IDENTITY_ERRORS as exc:
                results[source_id] = exc
        return results

    def _prefetch_source_regions(self, source_ids):
        """Share nearby coordinate discovery; original point filters stay exact."""
        by_chromosome = {}
        for source_id in source_ids:
            match = re.fullmatch(r"(?:chr)?([1-9]|1[0-9]|2[0-2]|X|Y)[_:](\d+)[_:]([ACGT])[_:]([ACGT])", source_id, re.I)
            if match:
                chrom, position, ref, alt = match.groups()
                by_chromosome.setdefault(chrom.upper(), []).append((int(position), source_id, ref.upper(), alt.upper()))
        groups = []
        for chrom, values in sorted(by_chromosome.items()):
            chunk = []
            for value in sorted(values):
                if chunk and value[0] - chunk[0][0] > parameter("ranking.ld_window_bp"):
                    groups.append((chrom, chunk)); chunk = []
                chunk.append(value)
            if chunk: groups.append((chrom, chunk))
        def discover(group):
            chrom, values = group
            try:
                rows = self.get(self.ensembl, f"/overlap/region/human/{chrom}:{values[0][0]}-{values[-1][0]}",
                                feature="variation", **{"content-type": "application/json"})
                if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
                    raise ValueError("Region discovery did not return a valid variation list.")
                for position, source_id, ref, alt in values:
                    candidates = sorted({row["id"] for row in rows if row.get("start") == position
                        and re.fullmatch(r"rs\d+", str(row.get("id", "")))
                        and isinstance(row.get("alleles"), list) and {ref, alt}.issubset(row["alleles"])})
                    if candidates:
                        self._source_candidate_cache[source_id] = candidates
            except _IDENTITY_ERRORS:
                pass  # Every unresolved point retains its original fallback.
            return len(values)
        completed = 0
        with ThreadPoolExecutor(max_workers=parameter("ranking.ld_identity_concurrency")) as pool:
            for count in pool.map(discover, groups):
                completed += count
                emit_progress(self.progress_sink, "ld_source_discovery", status="running",
                              completed_count=completed, total_count=len(source_ids), **self.request_summary)

    def pages(self, path, embedded_key, **params):
        """Require coherent pagination; never use a partial list as an exhaustive negative."""
        output, seen = [], set()
        complete = False
        expected_total = None
        first = self.get(self.catalog, path, **params, size=parameter("novelty.catalog_page_size"), page=0)
        total_pages = first.get("page", {}).get("totalPages")
        if not isinstance(total_pages, int) or isinstance(total_pages, bool) or total_pages < 0:
            raise ValueError("Missing catalog page count.")
        page_limit = min(max(total_pages, 1), parameter("novelty.catalog_max_pages"))
        values = {0: first}
        if page_limit > 1:
            with ThreadPoolExecutor(max_workers=parameter("novelty.catalog_concurrency")) as pool:
                pending = {pool.submit(self.get, self.catalog, path, **params,
                           size=parameter("novelty.catalog_page_size"), page=number): number
                           for number in range(1, page_limit)}
                for future in as_completed(pending):
                    number = pending[future]
                    values[number] = future.result()
                    if self.progress:
                        self.progress(f"GWAS Catalog: received {len(values)} of {total_pages} association pages")
        for page, value in sorted(values.items()):
            rows = value.get("_embedded", {}).get(embedded_key, [])
            meta = value.get("page", {})
            if not isinstance(rows, list) or meta.get("number") != page:
                raise ValueError("Invalid catalog pagination.")
            total = meta.get("totalElements")
            if not isinstance(total, int) or total < 0:
                raise ValueError("Missing catalog result count.")
            if expected_total is not None and expected_total != total:
                raise ValueError("Catalog changed during pagination; retry a coherent screen.")
            expected_total = total
            for row in rows:
                key = row.get("association_id") or row.get("efo_id") or row.get("accession_id")
                if not key or key in seen:
                    raise ValueError("Catalog pagination returned duplicated or missing identifiers.")
                seen.add(key)
                output.append(row)
            if page + 1 >= meta.get("totalPages", 1):
                complete = len(output) == total
                break
        return output, complete

    def _source_candidates(self, source_id):
        match = re.fullmatch(r"(?:chr)?([1-9]|1[0-9]|2[0-2]|X|Y)[_:](\d+)[_:]([ACGT])[_:]([ACGT])", source_id, re.I)
        if re.fullmatch(r"rs\d+", source_id, re.I):
            candidates = [source_id.lower()]
            expected = None
        elif match:
            chrom, position, ref, alt = match.groups()
            expected = (chrom.upper(), int(position), ref.upper(), alt.upper())
            if source_id in self._source_candidate_cache:
                return self._source_candidate_cache[source_id], expected
            rows = self.get(self.ensembl, f"/overlap/region/human/{chrom}:{position}-{position}",
                            feature="variation", **{"content-type": "application/json"})
            candidates = sorted({row["id"] for row in rows if row.get("start") == int(position)
                                 and re.fullmatch(r"rs\d+", row.get("id", ""))
                                 and {ref.upper(), alt.upper()}.issubset(row.get("alleles", []))})
        else:
            raise ValueError("LD requires a validated rsID or a biallelic canonical SNP identifier.")
        return candidates, expected

    def resolve(self, source_id):
        """Validate exact GRCh38 coordinate and alleles before assigning an rs identifier."""
        candidates, expected = self._source_candidates(source_id)
        resolved = []
        for candidate in candidates:
            value = self.variation(candidate)
            if value.get("name", candidate) != candidate:
                raise ValueError("Source reference metadata names a different requested rs identifier.")
            for row in value.get("mappings", []):
                alleles = row.get("allele_string", "").split("/")
                if (row.get("assembly_name") != parameter("novelty.reference_build")
                        or row.get("coord_system") != "chromosome" or row.get("strand") != 1
                        or row.get("start") != row.get("end")):
                    continue
                if expected and (str(row.get("seq_region_name")), row.get("start")) != expected[:2]:
                    continue
                if expected and not set(expected[2:]).issubset(alleles):
                    continue
                resolved.append({"source_variant_id": source_id, "rsid": value.get("name", candidate),
                                 "chromosome": str(row["seq_region_name"]), "position": row["start"],
                                 "alleles": alleles, "ld_alleles_unambiguous": len(alleles) == 2,
                                 "reference_allele": alleles[0],
                                 "source_allele_order_matches_reference": not expected or expected[2] == alleles[0],
                                 "selected_alleles": list(expected[2:]) if expected else alleles})
        unique = {(x["rsid"], x["chromosome"], x["position"]): x for x in resolved}
        if len(unique) != 1:
            raise ValueError("SNP identity is missing or ambiguous in the GRCh38 reference.")
        return next(iter(unique.values()))

    def pair(self, selected, prior, population):
        item = {"selected_variant_id": selected["source_variant_id"],
                "selected_rsid": selected["rsid"], "reported_rsid": prior["rsid"],
                "population": population, "r2": None, "status": "missing"}
        panel = parameter("novelty.reference_panel") + ":" + population
        panel_alleles = {}
        for variant in (selected, prior):
            if variant["ld_alleles_unambiguous"]:
                continue
            value = self.variation(variant['rsid'], populations=True)
            alleles = set()
            for row in _population_response_rows(value, variant["rsid"]):
                frequency = row.get("frequency")
                if (row.get("population") == panel and isinstance(frequency, (int, float))
                    and not isinstance(frequency, bool) and math.isfinite(frequency) and 0 < frequency <= 1):
                    alleles.add(str(row.get("allele")))
            # dbSNP can list extra rare alleles even when the requested 1000G
            # population is biallelic. Verify the actual panel instead of either
            # rejecting all such SNPs or borrowing LD for another allele pair.
            canonical_selection = not re.fullmatch(r"rs\d+", variant.get("source_variant_id", ""))
            if (len(alleles) != 2 or not alleles.issubset({"A", "C", "G", "T"}) or
                canonical_selection and set(variant.get("selected_alleles", [])) != alleles):
                return {**item, "reason": "Alleles in the requested LD reference population could not be uniquely verified."}
            panel_alleles[variant["rsid"]] = sorted(alleles)
        rows = self.get(self.ensembl, f"/ld/human/pairwise/{selected['rsid']}/{prior['rsid']}",
                        population_name=panel, r2=0, d_prime=0, **{"content-type": "application/json"})
        values = []
        for row in rows:
            if row.get("population_name") != panel or {row.get("variation1"), row.get("variation2")} != {selected["rsid"], prior["rsid"]}:
                continue
            try:
                value = float(row.get("r2"))
            except (TypeError, ValueError):
                continue
            if not isinstance(row.get("r2"), bool) and math.isfinite(value) and 0 <= value <= 1:
                values.append(value)
        if len(values) == 1:
            return {**item, "r2": values[0], "status": "computed", "verified_panel_alleles": panel_alleles}
        return {**item, "reason": "No unique numeric LD value returned for the requested pair and population."}

    def study_populations(self, study_id):
        value = self.get(self.catalog, f"/studies/{quote(study_id, safe='')}/ancestries")
        rows = value.get("_embedded", {}).get("ancestries", [])
        labels = {str(group.get("ancestral_group", "")) for row in rows
                  for group in row.get("ancestral_groups", [])}
        from ..novelty import choose_population
        return {choice.code for label in labels
                if (choice := choose_population("", population=label)).code}, sorted(labels)

    def validate_ld_population(self, variant, population):
        """The unchanged exact allele-pair check shared by both LD endpoints."""
        panel = parameter("novelty.reference_panel") + ":" + population
        observed = self.variation(variant['rsid'], populations=True)
        alleles = {str(row.get("allele")) for row in _population_response_rows(observed, variant['rsid'])
                   if row.get("population") == panel and isinstance(row.get("frequency"), (int, float))
                   and not isinstance(row.get("frequency"), bool) and math.isfinite(row["frequency"])
                   and 0 < row["frequency"] <= 1}
        if (len(alleles) != 2 or not alleles.issubset({"A", "C", "G", "T"}) or
            not re.fullmatch(r"rs\d+", variant["source_variant_id"]) and
            set(variant["selected_alleles"]) != alleles):
            raise ValueError("The selected allele pair is not verified in the requested LD population.")
        return True

    def proxies(self, variant, population, *, threshold=None, window_bp=None):
        """Enumerate the thresholded reference set; an empty set needs panel support."""
        panel = parameter("novelty.reference_panel") + ":" + population
        threshold = parameter("novelty.ld_r2_threshold") if threshold is None else threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("The LD proxy threshold must be finite and in (0, 1].")
        self.validate_ld_population(variant, population)
        flank = parameter("novelty.ld_window_bp") if window_bp is None else window_bp
        window_kb = math.ceil(2 * flank / 1000)
        if window_kb > 500:
            raise ValueError("The key-free Ensembl proxy endpoint supports at most 250 kb per side.")
        rows = self.get(self.ensembl, f"/ld/human/{variant['rsid']}/{quote(panel, safe='')}",
            window_size=window_kb, r2=threshold, d_prime=0, attribs=1,
            **{"content-type": "application/json"})
        if not isinstance(rows, list):
            raise ValueError("Proxy lookup did not return the expected complete list.")
        output = {}
        for row in rows:
            try:
                rsid, value = row["variation"], float(row["r2"])
                chrom, start, end = str(row["chr"]), int(row["start"]), int(row["end"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("A returned LD proxy lacks a verifiable identifier, position or LD value.") from None
            if (row.get("population_name") != panel or not re.fullmatch(r"rs\d+", rsid)
                or isinstance(row["r2"], bool) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("A returned LD proxy has inconsistent population or numeric identity.")
            if chrom != variant["chromosome"] or abs(start - variant["position"]) > flank:
                continue
            if value < threshold:
                continue
            pair = {"selected_variant_id": variant["source_variant_id"], "selected_rsid": variant["rsid"],
                    "reported_rsid": rsid, "population": population, "r2": value, "status": "computed",
                    "chromosome": chrom, "position": start, "end": end}
            if rsid in output and output[rsid] != pair:
                raise ValueError("Conflicting duplicate LD proxy values were returned.")
            output[rsid] = pair
        return list(output.values())

    def screen(self, source_ids, trait, population):
        """Selected SNPs + r2-threshold proxies, followed by prior-report set lookup.

        The Catalog is paginated by phenotype once, then intersected with the
        queried rs identifiers. This is equivalent to individual rs lookups but
        avoids hundreds of HTTP requests. Unrelated unlocated Catalog records
        do not determine coverage of the explicitly enumerated rs query set.
        """
        from ..novelty import _normalized, canonical_trait
        issues, selected, proxies, prior_rows = [], [], [], []
        identity_complete = True
        for identifier in source_ids:
            try:
                selected.append(self.resolve(identifier))
            except (RuntimeError, ValueError) as exc:
                identity_complete = False
                issues.append(f"Cannot verify selected SNP {identifier}: {exc}")
        if not selected:
            return {"pairs": [], "coverage_complete": False, "identity_complete": False,
                    "prior_variant_count": None, "required_pair_count": None, "issues": issues,
                    "query_ledger": self.ledger, "resolved_selection": [], "algorithm": "ld_proxies_then_reports"}
        traits, complete = self.pages("/efo-traits", "efo_traits", efo_trait=trait)
        matches = [x for x in traits if _normalized(canonical_trait(x.get("efo_trait", ""))) == _normalized(trait)]
        if not complete or len(matches) != 1:
            return {"pairs": [], "coverage_complete": False, "identity_complete": identity_complete,
                    "prior_variant_count": None, "required_pair_count": None,
                    "issues": issues + ["The exact phenotype could not be uniquely mapped in GWAS Catalog."],
                    "query_ledger": self.ledger, "resolved_selection": selected, "algorithm": "ld_proxies_then_reports"}
        efo_id = matches[0]["efo_id"]

        def matching_record(row, queried_ids):
            if efo_id not in {x.get("efo_id") for x in row.get("efo_traits", [])}:
                return None
            effect = row.get("snp_allele", [])
            if not any(x.get("rs_id") in queried_ids for x in effect):
                return None
            if row.get("multi_snp_haplotype") or row.get("snp_interaction") or len(effect) != 1:
                raise ValueError("A matching prior record describes a haplotype or SNP interaction; individual-variant novelty remains unresolved.")
            if not row.get("pubmed_id"):
                raise ValueError("A matching prior association lacks a verifiable publication reference.")
            labels = []
            if getattr(population, "reported_study_population_explicit", False):
                groups, labels = self.study_populations(str(row.get("accession_id", "")))
                if not groups:
                    raise ValueError("Population metadata for a matching prior SNP is unresolved.")
                if population.reported_study_population not in groups:
                    return None
            for variant in selected:
                if variant["rsid"] == effect[0]["rs_id"] and not variant["ld_alleles_unambiguous"]:
                    if effect[0].get("effect_allele") not in set(variant["selected_alleles"]) - {variant["reference_allele"]}:
                        raise ValueError("A selected multiallelic SNP's prior association cannot be uniquely assigned to the selected allele pair.")
            return {"rsid": effect[0]["rs_id"], "association_id": row["association_id"],
                    "study_id": row.get("accession_id"), "pmid": row.get("pubmed_id"),
                    "phenotype_id": efo_id, "population_labels": labels,
                    "effect_allele": effect[0].get("effect_allele")}

        # The selected SNP itself belongs to the query set. A verified exact
        # report is already sufficient and does not need proxy enumeration.
        for variant in selected:
            rows, _ = self.pages("/associations", "associations", rs_id=variant["rsid"],
                                  efo_id=efo_id, show_child_trait="false")
            for row in rows:
                record = matching_record(row, {variant["rsid"]})
                if record is None:
                    continue
                if (not variant["ld_alleles_unambiguous"] and record["effect_allele"]
                    not in set(variant["selected_alleles"]) - {variant["reference_allele"]}):
                    continue
                return {"pairs": [], "coverage_complete": False, "identity_complete": identity_complete,
                        "prior_variant_count": 1, "exact_matches": [variant["source_variant_id"]],
                        "basis": "The selected SNP itself has a verified same-phenotype prior report.",
                        "prior_associations": [{"rsid": variant["rsid"], "associations": [record]}],
                        "resolved_selection": selected, "phenotype_id": efo_id, "issues": issues,
                        "query_ledger": self.ledger, "algorithm": "exact_selected_snp_report"}
        for variant in selected:
            try:
                proxies.extend(self.proxies(variant, population.code))
            except (RuntimeError, ValueError) as exc:
                complete = False
                issues.append(str(exc))
        if len(proxies) > parameter("novelty.max_ld_pairs"):
            proxies = proxies[:parameter("novelty.max_ld_pairs")]
            complete = False
            issues.append("LD proxy budget reached; the query set is incomplete.")
        queried = {v["rsid"] for v in selected} | {pair["reported_rsid"] for pair in proxies}
        if self.progress:
            self.progress(f"Checking prior reports for {len(queried)} selected and LD-related SNPs")
        rows, catalog_complete = self.pages("/associations", "associations", efo_id=efo_id,
                                            show_child_trait="false")
        complete = complete and catalog_complete
        for row in rows:
            try:
                record = matching_record(row, queried)
            except (ValueError, RuntimeError) as exc:
                complete = False
                issues.append(str(exc))
                continue
            if record:
                prior_rows.append(record)
        reported_ids = {row["rsid"] for row in prior_rows}
        matched_pairs = [pair for pair in proxies if pair["reported_rsid"] in reported_ids]
        exact = [v["source_variant_id"] for v in selected if v["rsid"] in reported_ids]
        return {"pairs": matched_pairs, "required_pair_count": len(matched_pairs),
                "ld_proxies": proxies, "ld_proxy_count": len(proxies),
                "queried_snp_ids": sorted(queried), "queried_snp_count": len(queried),
                "catalog_associations_scanned": len(rows), "catalog_lookup_complete": catalog_complete,
                "coverage_complete": complete, "identity_complete": identity_complete,
                "prior_variant_count": len(reported_ids), "exact_matches": exact,
                "prior_associations": [{"rsid": rsid, "associations": [r for r in prior_rows if r["rsid"] == rsid]}
                                       for rsid in sorted(reported_ids)],
                "resolved_selection": selected, "phenotype_id": efo_id,
                "issues": list(dict.fromkeys(issues)), "query_ledger": self.ledger,
                "algorithm": "ld_proxies_then_reports"}
