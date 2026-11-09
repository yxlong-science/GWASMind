"""Bounded, exact Open Targets variant identities for LD reference checks.

Canonical IDs use variant(); rsIDs use paged Variant-only search and must occur
in the returned object's rsIds. Search rank/name/highlights never prove identity.
Only missing or failed lookups are eligible for a separately audited fallback.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError
from uuid import uuid4

from ..progress import emit_progress
from .ld_cache import LDResponseCache, encode
from .live import (OfficialOpenTargetsGraphQLTransport, OFFICIAL_OPEN_TARGETS_GRAPHQL_ENDPOINT,
                   _RetryableGraphQLTransportError)
from .ld import reference_failure_reason
from .retrieval_recovery import retry_after_seconds

_CANONICAL = re.compile(r"^([1-9]|1[0-9]|2[0-2]|X|Y|MT)_([1-9][0-9]*)_([ACGT]+)_([ACGT]+)$")
_RSID = re.compile(r"^rs[1-9][0-9]*$")
_FIELDS = "id chromosome position referenceAllele alternateAllele rsIds __typename"


def _normal(identifier):
    if not isinstance(identifier, str):
        return None
    value = identifier.strip()
    if _RSID.fullmatch(value.casefold()):
        return value.casefold()
    value = re.sub(r"^chr", "", value, flags=re.I).replace(":", "_").upper()
    match = _CANONICAL.fullmatch(value)
    return value if match and match[3] != match[4] else None


def _identity(value):
    if not isinstance(value, Mapping) or value.get("__typename", "Variant") != "Variant":
        raise ValueError("Invalid variant object")
    identifier = _normal(value.get("id"))
    match = _CANONICAL.fullmatch(identifier or "")
    if (not match or value.get("chromosome") != match[1] or type(value.get("position")) is not int
            or value["position"] != int(match[2]) or value.get("referenceAllele") != match[3]
            or value.get("alternateAllele") != match[4]):
        raise ValueError("Variant ID disagrees with its coordinate or alleles")
    rsids = value.get("rsIds")
    if rsids is None:
        rsids = []
    if not isinstance(rsids, list) or any(not isinstance(item, str) or not _RSID.fullmatch(item) for item in rsids):
        raise ValueError("Invalid variant aliases")
    return {"variant_id": identifier, "chromosome": match[1], "position": int(match[2]),
            "reference_allele": match[3], "alternate_allele": match[4], "rsids": sorted(set(rsids)),
            "genome_build": "GRCh38"}


class OTIdentityResolver:
    provider_name = "opentargets"
    provider_version = "1.0.0"

    def __init__(self, *, fetch=None, cache_directory=None, timeout_seconds=20.0,
                 total_timeout_seconds=180.0, deadline=None, cancel_event=None,
                 progress_sink=None, batch_size=20, search_page_size=100, max_search_pages=10,
                 max_retries=0):
        for value, maximum in ((batch_size, 100), (search_page_size, 100), (max_search_pages, 10)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("OT identity batch/page limits are outside their bounds")
        for value in (timeout_seconds, total_timeout_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("OT identity timeout must be finite and positive")
        if timeout_seconds > 60:
            raise ValueError("OT identity request timeout cannot exceed 60 seconds")
        if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
            raise ValueError("OT identity deadline must be finite")
        self.fetch, self.timeout_seconds = fetch, timeout_seconds
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("OT identity retries must be between zero and three")
        self.max_retries = max_retries
        self.deadline = time.monotonic() + total_timeout_seconds if deadline is None else deadline
        self.cancel_event, self.progress_sink = cancel_event, progress_sink
        self.batch_size, self.search_page_size, self.max_search_pages = batch_size, search_page_size, max_search_pages
        self.cache = LDResponseCache(Path(cache_directory) / "ot_identity") if cache_directory is not None else None
        self.endpoint = OFFICIAL_OPEN_TARGETS_GRAPHQL_ENDPOINT
        self.ledger = []
        self._pages, self._results = {}, {}
        self._reverse_checked = set()
        self._lock = threading.RLock()

    def _check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CancelledError()

    def _fetch_live(self, query, variables, remaining):
        transport = OfficialOpenTargetsGraphQLTransport(timeout_seconds=min(self.timeout_seconds, remaining),
            max_transport_retries=0, max_response_bytes=2 * 1024 * 1024,
            max_operation_bytes=2 * 1024 * 1024)
        try:
            data, _ = transport._execute_graphql_sync(query, dict(variables))
            return data
        finally:
            transport.close()

    @staticmethod
    def _transport_reason(error):
        if isinstance(error, _RetryableGraphQLTransportError):
            category = error.failure_details.get("category")
            return category if category in {"rate_limited", "service_error"} else "connection_error"
        return reference_failure_reason(error)

    def _fetch_with_retries(self, query, variables, receipt):
        for attempt in range(self.max_retries + 1):
            self._check_cancelled()
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                receipt["transport_reason"] = "total_budget_exhausted"
                raise TimeoutError("OT identity deadline exhausted")
            receipt.update(attempt=attempt + 1, started_at=datetime.now(timezone.utc).isoformat())
            started = time.monotonic()
            try:
                response = self.fetch(query, dict(variables)) if self.fetch is not None else self._fetch_live(query, variables, remaining)
                receipt["elapsed_seconds"] = time.monotonic() - started
                return response
            except CancelledError:
                raise
            except Exception as error:
                reason = self._transport_reason(error)
                receipt.update(elapsed_seconds=time.monotonic() - started, transport_reason=reason)
                if (reason not in {"rate_limited", "service_error", "request_timeout", "connection_error"}
                        or attempt >= self.max_retries):
                    raise
                self.ledger.append({**receipt, "status": "failed", "reason_code": "ot_identity_request_failed",
                                    "error_type": type(error).__name__})
                retry_after = retry_after_seconds(error.headers) if isinstance(error, HTTPError) else getattr(error, "retry_after", None)
                delay = max(float(retry_after or 0), 0.5 * 2 ** attempt)
                until = min(self.deadline, time.monotonic() + delay)
                while time.monotonic() < until:
                    self._check_cancelled()
                    time.sleep(min(.1, max(0, until - time.monotonic())))
                receipt.pop("transport_reason", None)

    def _request(self, query, variables):
        self._check_cancelled()
        request = {**LDResponseCache.request(self.endpoint, {"query": query, "variables": variables}),
                   "provider": self.provider_name, "provider_version": self.provider_version,
                   "response_contract": "ot-identity-decoded-data-v1", "reference_build": "GRCh38"}
        key = LDResponseCache.key(request)
        now = datetime.now(timezone.utc).isoformat()
        receipt = {"reference_provider": self.provider_name, "provider_version": self.provider_version,
                   "url": self.endpoint, "method": "POST", "request_key": key, "retrieved_at": now,
                   "representation": "canonical_decoded_graphql_data", "from_cache": False}
        if key in self._pages:
            data, previous = self._pages[key]
            receipt = {**previous, "from_cache": True, "observed_at": now}
            self.ledger.append(receipt)
            return deepcopy(data), receipt
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            receipt.update(status="failed", reason_code="ot_identity_deadline_exhausted")
            self.ledger.append(receipt)
            return None, receipt
        operation_id = uuid4().hex
        emit_progress(self.progress_sink, "ld_identity_batch", status="started", operation_id=operation_id,
                      reference_provider=self.provider_name, provider_version=self.provider_version,
                      total_count=len(variables))
        try:
            cached = None
            if self.cache is not None:
                try:
                    cached = self.cache.get(request)
                except (OSError, ValueError, TypeError, KeyError):
                    # A damaged cache is a recoverable storage gap, never an identity.
                    cached = None
            if cached is not None:
                data, saved = cached
                receipt.update({name: saved[name] for name in ("sha256", "bytes", "retrieved_at", "response_path")})
                receipt.update(status="success", from_cache=True, observed_at=now)
            else:
                response = self._fetch_with_retries(query, variables, receipt)
                if self.fetch is not None:
                    if not isinstance(response, Mapping):
                        raise ValueError("OT identity response must be an object")
                    if "data" in response or "errors" in response:
                        data, _ = OfficialOpenTargetsGraphQLTransport._decode_graphql_response(
                            encode(response), allow_partial_aliases=True)
                    else:
                        data = dict(response)
                else:
                    data = response
                # Located alias failures are independent; retain only their locations,
                # never provider free-text errors, in reusable scientific artifacts.
                if "__gwasmind_alias_errors" in data:
                    data = {**data, "__gwasmind_alias_errors": {alias: True for alias in data["__gwasmind_alias_errors"]}}
                raw = encode(data)
                if len(raw) > 2 * 1024 * 1024:
                    raise ValueError("OT identity response exceeded its bound")
                receipt.update(status="success", sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
                if self.cache is not None:
                    # Preserve the immutable partial-response artifact without
                    # making its failed aliases reusable across resolver runs.
                    saved = self.cache.save(request, raw, retrieved_at=now,
                        reusable=not bool(data.get("__gwasmind_alias_errors")))
                    receipt["response_path"] = saved["response_path"]
            self._check_cancelled()
        except CancelledError:
            raise
        except Exception as error:
            data = None
            receipt.update(status="failed", reason_code="ot_identity_request_failed", error_type=type(error).__name__)
        self._pages[key] = (deepcopy(data), receipt)
        self.ledger.append(receipt)
        emit_progress(self.progress_sink, "ld_identity_batch", status="completed" if data is not None else "incomplete", operation_id=operation_id,
                      reference_provider=self.provider_name, provider_version=self.provider_version,
                      completed_count=len(variables), total_count=len(variables))
        return data, receipt

    def _query(self, identifiers, page):
        definitions, selections, variables = [], [], {}
        for index, identifier in enumerate(identifiers):
            alias = f"entity{index}"
            variable = f"id{index}"
            definitions.append(f"${variable}: String!")
            variables[variable] = identifier
            if _RSID.fullmatch(identifier):
                selections.append(f'{alias}: search(queryString: ${variable}, entityNames: ["variant"], '
                    f'page: {{index: {page}, size: {self.search_page_size}}}) '
                    f'{{ total hits {{ id entity object {{ ... on Variant {{ {_FIELDS} }} }} }} }}')
            else:
                selections.append(f"{alias}: variant(variantId: ${variable}) {{ {_FIELDS} }}")
        # Reuse the existing transport's strict per-alias GraphQL error contract.
        return f"query GWASMindSharedEntities({', '.join(definitions)}) {{ {' '.join(selections)} }}", variables

    @staticmethod
    def _result(status, candidates, receipts, reason):
        ordered = sorted(candidates.values(), key=lambda row: row["variant_id"])
        if status == "failed" and len(ordered) > 1:
            # An unfinished page cannot make already-observed allelic ambiguity
            # eligible for silent replacement by a backup provider.
            status = "ambiguous"
        return {"status": status, "identity": deepcopy(ordered[0]) if status == "resolved" else None,
                "candidates": deepcopy(ordered), "receipts": deepcopy(receipts),
                "fallback_eligible": status in {"missing", "failed"}, "reason_code": reason}

    def resolve(self, identifier):
        return self.resolve_many([identifier])[identifier]

    def _check_missing_canonical_allele_order(self, identifiers):
        """Check the other allele order once; never repair the original input.

        This uses direct canonical lookups only. Reverse lookup results are not
        recursively submitted to this check, and existing primary results may
        satisfy the additional observation without a duplicate HTTP request.
        """
        targets = {}
        for identifier in identifiers:
            if (identifier is not None and _CANONICAL.fullmatch(identifier)
                    and identifier not in self._reverse_checked and self._results[identifier]["status"] == "missing"):
                chromosome, position, reference, alternate = identifier.split("_")
                targets[identifier] = f"{chromosome}_{position}_{alternate}_{reference}"
        missing = sorted({reverse for reverse in targets.values() if reverse not in self._results})
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            query, variables = self._query(batch, 0)
            data, receipt = self._request(query, variables)
            for index, identifier in enumerate(batch):
                alias, candidates = f"entity{index}", {}
                if data is None or alias not in data or alias in data.get("__gwasmind_alias_errors", {}):
                    status, reason = "failed", receipt.get("reason_code", "ot_alias_request_failed")
                elif data[alias] is None:
                    status, reason = "missing", "ot_variant_not_found"
                else:
                    try:
                        identity = _identity(data[alias])
                        if identity["variant_id"] != identifier:
                            raise ValueError("Wrong requested reverse identity")
                        candidates[identifier] = {**identity, "receipts": [receipt]}
                        status, reason = "resolved", "ot_exact_identity"
                    except ValueError:
                        status, reason = "conflicting", "ot_identity_fields_conflict"
                self._results[identifier] = self._result(status, candidates, [receipt], reason)
        for identifier, reverse in targets.items():
            original, checked = self._results[identifier], self._results[reverse]
            receipts, seen = [], set()
            for receipt in [*original["receipts"], *checked["receipts"]]:
                key = (receipt.get("request_key"), receipt.get("sha256"), receipt.get("status"))
                if key not in seen:
                    receipts.append(receipt)
                    seen.add(key)
            if checked["status"] == "resolved":
                status, reason = "conflicting", "ot_reference_allele_order_conflict"
            elif checked["status"] == "missing":
                status, reason = "missing", "ot_variant_not_found"
            elif checked["status"] == "failed":
                status, reason = "failed", "ot_reference_allele_order_check_failed"
            else:
                status, reason = "conflicting", "ot_identity_fields_conflict"
            result = self._result(status, {row["variant_id"]: row for row in checked["candidates"]}, receipts, reason)
            result["allele_order_check"] = {"original_variant_id": identifier, "reverse_variant_id": reverse,
                                              "reverse_status": checked["status"], "input_rewritten": False}
            self._results[identifier] = result
            self._reverse_checked.add(identifier)

    def resolve_many(self, identifiers: Sequence[str]):
        if not isinstance(identifiers, (list, tuple)) or len(identifiers) > 100000:
            raise ValueError("OT identity inputs must be a bounded identifier list")
        if any(not isinstance(item, str) for item in identifiers):
            raise ValueError("OT identity identifiers must be strings")
        with self._lock:
            self._check_cancelled()
            names = {item: _normal(item) for item in identifiers}
            needed = sorted({value for value in names.values() if value is not None and value not in self._results})
            for start in range(0, len(needed), self.batch_size):
                pending = needed[start:start + self.batch_size]
                state = {identifier: {"candidates": {}, "receipts": [], "seen": set(), "total": None} for identifier in pending}
                for page in range(self.max_search_pages):
                    if not pending:
                        break
                    query, variables = self._query(pending, page)
                    data, receipt = self._request(query, variables)
                    again = []
                    for index, identifier in enumerate(pending):
                        item = state[identifier]
                        item["receipts"].append(receipt)
                        alias = f"entity{index}"
                        status, reason = None, "ot_exact_identity"
                        if data is None or alias not in data or alias in data.get("__gwasmind_alias_errors", {}):
                            status, reason = "failed", receipt.get("reason_code", "ot_alias_request_failed")
                        elif not _RSID.fullmatch(identifier):
                            value = data[alias]
                            if value is None:
                                status, reason = "missing", "ot_variant_not_found"
                            else:
                                try:
                                    identity = _identity(value)
                                    if identity["variant_id"] != identifier:
                                        raise ValueError("Wrong requested identity")
                                    item["candidates"][identifier] = {**identity, "receipts": [receipt]}
                                    status = "resolved"
                                except ValueError:
                                    status, reason = "conflicting", "ot_identity_fields_conflict"
                        else:
                            value = data[alias]
                            hits = value.get("hits") if isinstance(value, Mapping) else None
                            total = value.get("total") if isinstance(value, Mapping) else None
                            if type(total) is not int or total < 0 or not isinstance(hits, list) or len(hits) > self.search_page_size:
                                status, reason = "failed", "ot_search_shape_invalid"
                            elif item["total"] is not None and item["total"] != total:
                                status, reason = "failed", "ot_search_scope_changed"
                            else:
                                item["total"] = total
                                for hit in hits:
                                    if not isinstance(hit, Mapping) or not isinstance(hit.get("id"), str):
                                        status, reason = "failed", "ot_search_shape_invalid"
                                        break
                                    if hit["id"] in item["seen"]:
                                        status, reason = "failed", "ot_search_page_repeated"
                                        break
                                    item["seen"].add(hit["id"])
                                    obj = hit.get("object")
                                    if str(hit.get("entity", "")).casefold() == "variant" and not isinstance(obj, Mapping):
                                        status, reason = "failed", "ot_search_object_incomplete"
                                        break
                                    if (str(hit.get("entity", "")).casefold() != "variant" or not isinstance(obj, Mapping)
                                            or not isinstance(obj.get("rsIds"), list) or identifier not in obj["rsIds"]):
                                        continue  # A fuzzy search hit proves no alias identity.
                                    try:
                                        identity = _identity(obj)
                                        if hit["id"] != identity["variant_id"]:
                                            raise ValueError("Search hit and object disagree")
                                    except ValueError:
                                        status, reason = "conflicting", "ot_identity_fields_conflict"
                                        break
                                    item["candidates"][identity["variant_id"]] = {**identity, "receipts": [receipt]}
                                if status is None:
                                    if len(item["seen"]) > total:
                                        status, reason = "failed", "ot_search_total_inconsistent"
                                    elif len(item["seen"]) == total:
                                        count = len(item["candidates"])
                                        status = "resolved" if count == 1 else "ambiguous" if count > 1 else "missing"
                                        reason = "ot_exact_identity" if count == 1 else "ot_multiple_exact_alleles" if count > 1 else "ot_exact_rsid_not_found"
                                    elif not hits or page + 1 >= self.max_search_pages:
                                        status, reason = "failed", "ot_search_incomplete"
                                    else:
                                        again.append(identifier)
                        if status is not None:
                            self._results[identifier] = self._result(status, item["candidates"], item["receipts"], reason)
                    pending = again
            self._check_missing_canonical_allele_order(names.values())
            return {name: deepcopy(self._results[value]) if value is not None else
                    self._result("failed", {}, [], "unsupported_identity_format") for name, value in names.items()}
