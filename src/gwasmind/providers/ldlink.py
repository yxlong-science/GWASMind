"""LDlink LDproxy/LDpair with separate authentication and scientific identity.

Contract: https://ldlink.nih.gov/apiaccess and the NCI LDlink server sources.
LDproxy Alleles are an unordered population contrast, never REF/ALT. Ensembl
identity methods remain inherited; callers must verify the GRCh38 reference.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import threading
import time
from collections.abc import Mapping
from concurrent.futures import CancelledError
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from ..configuration import parameter
from ..progress import emit_progress
from .ld import LDReferenceProvider, _reference_error
from .ld_cache import LDResponseCache, encode
from .ldlink_lock import serial_ldlink_request
from .retrieval_recovery import retry_after_seconds

_LDLINK_REQUEST_LOCK = threading.Lock()
_LAST_COMPLETED = 0.0
_POPULATIONS = frozenset("ALL AFR AMR EAS EUR SAS ACB ASW BEB CDX CEU CHB CHS CLM ESN FIN GBR GIH GWD IBS ITU JPT KHV LWK MSL MXL PEL PJL PUR STU TSI YRI".split())
_COORD = re.compile(r"chr([1-9]|1[0-9]|2[0-2]|X|Y):([1-9][0-9]*)", re.I)
_RSID = re.compile(r"rs[0-9]+")


def _failure(reason="invalid_response"):
    return _reference_error("The LDlink lookup did not return a verified complete result.", reason)


def _number(value, *, maximum=1.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise _failure() from None
    if isinstance(value, bool) or not math.isfinite(result) or not 0 <= result <= maximum:
        raise _failure()
    return result


def _alleles(value):
    if isinstance(value, str):
        value = value.strip().removeprefix("(").removesuffix(")").split("/")
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(x, str) and re.fullmatch(r"[ACGT]+|-", x) for x in value)
            or len(set(value)) != 2):
        raise _failure("population_alleles_ambiguous")
    return sorted(value)


def _coordinate(value):
    match = _COORD.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise _failure()
    return match[1].upper(), int(match[2])


def _assert_variant(observed, expected, *, allow_rsid_alias=False):
    if ((not allow_rsid_alias and observed["rsid"] != expected["rsid"]) or observed["chromosome"] != expected["chromosome"]
            or observed["position"] != expected["position"]):
        raise _failure("reference_coordinate_unverified")
    alleles = set(observed["alleles"])
    selected = expected.get("selected_alleles", expected.get("alleles", []))
    canonical = not _RSID.fullmatch(str(expected.get("source_variant_id", "")))
    if canonical and alleles != set(selected):
        raise _failure("population_alleles_ambiguous")
    if not alleles.issubset(set(expected.get("alleles", []))):
        raise _failure("population_alleles_ambiguous")


def _error_reason(message):
    value = str(message).lower()
    if "concurrent" in value or "quota" in value or "rate limit" in value:
        return "rate_limited"
    if "token" in value or "access" in value and any(x in value for x in ("blocked", "denied", "restrict")):
        return "authentication_failed"
    if any(x in value for x in ("monoallelic", "not polymorphic", "monomorphic")):
        return "monomorphic_in_selected_population"
    if any(x in value for x in ("not in dbsnp", "not in 1000", "not found")):
        return "not_found_in_queried_reference"
    return "invalid_response"


def _check_error(value):
    if isinstance(value, Mapping):
        if value.get("error"):
            raise _failure(_error_reason(value["error"]))
        if value.get("warning"):
            # Coordinate/allele warnings can change the scientific identity.
            raise _failure("reference_warning")


def _parse_proxy(text, variant, *, threshold=.01, window_bp=None):
    if text.lstrip().startswith(("{", "[")):
        try:
            _check_error(json.loads(text))
        except json.JSONDecodeError:
            pass
        raise _failure()
    if re.search(r"(?im)^\s*(?:warning|error)\b", text):
        raise _failure(_error_reason(text))
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")), delimiter="\t")
    required = {"RS_Number", "Coord", "Alleles", "MAF", "Distance", "Dprime", "R2"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise _failure(_error_reason(text))
    parsed, excluded = [], []
    raw_count = 0
    for line_number, raw in enumerate(reader, 2):
        raw_count += 1
        relevant, value = True, None
        try:
            value = _number(raw.get("R2"))
            # A verified value below threshold cannot hide an in-scope proxy,
            # even when its rsID is absent (e.g. an unnormalized indel).
            relevant = value >= threshold
            if None in raw or any(raw.get(key) is None for key in required):
                raise _failure("malformed_proxy_row")
            chrom, position = _coordinate(raw["Coord"])
            distance = int(raw["Distance"])
            if chrom != variant["chromosome"] or distance != position - variant["position"]:
                raise _failure("reference_coordinate_unverified")
            if window_bp is not None and abs(distance) > window_bp:
                relevant = False
            item = {"rsid": raw["RS_Number"], "chromosome": chrom, "position": position,
                    "alleles": _alleles(raw["Alleles"]), "r2": value,
                    "d_prime": _number(raw["Dprime"]), "maf": _number(raw["MAF"], maximum=.5),
                    "distance": distance}
            parsed.append((line_number, item, relevant))
        except (RuntimeError, ValueError, TypeError) as exc:
            excluded.append({"line": line_number, "reported_rsid": raw.get("RS_Number"), "r2": value,
                             "reason_code": getattr(exc, "reason_code", "malformed_proxy_row"),
                             "scope_relevant": relevant})
    expected = set(variant.get("selected_alleles", variant.get("alleles", [])))
    self_rows = [(line, row) for line, row, _ in parsed if row["distance"] == 0 and row["r2"] == 1
                 and row["chromosome"] == variant["chromosome"] and row["position"] == variant["position"]
                 and set(row["alleles"]) == expected]
    # rsIDs may be merged aliases; only one exact allelic self measurement
    # establishes the query identity. Its displayed allele order supplies no REF.
    if len(self_rows) != 1:
        raise _failure("source_identity_unverified")
    source_line, source = self_rows[0]
    if any(row["rsid"] == variant["rsid"] and line != source_line for line, row, _ in parsed):
        raise _failure("source_identity_unverified")
    if source["maf"] <= 0:
        raise _failure("monomorphic_in_selected_population")
    _assert_variant(source, variant, allow_rsid_alias=True)
    by_rsid = {}
    for line, row, relevant in parsed:
        if line == source_line:
            continue
        if not _RSID.fullmatch(row["rsid"]):
            excluded.append({"line": line, "reported_rsid": row["rsid"], "r2": row["r2"],
                             "reason_code": "proxy_identifier_missing", "scope_relevant": relevant})
            continue
        by_rsid.setdefault(row["rsid"], []).append((line, row, relevant))
    rows = []
    duplicate_count = 0
    for items in by_rsid.values():
        first = items[0][1]
        if any(row != first for _, row, _ in items):
            def contrast(row):
                return row["chromosome"], row["position"], tuple(row["alleles"])
            contrasts = {contrast(row) for _, row, _ in items}
            scoped = [(line, row, relevant) for line, row, relevant in items if relevant]
            selected = scoped[0][1] if scoped else None
            if (len(contrasts) > 1 and scoped and len({contrast(row) for _, row, _ in scoped}) == 1
                    and all(row == selected for _, row, _ in items if contrast(row) == contrast(selected))):
                # A second, out-of-scope alternate at the same rsID does not
                # invalidate the uniquely selected population contrast.
                rows.append(selected)
                excluded.extend({"line": line, "reported_rsid": row["rsid"], "r2": row["r2"],
                                 "reason_code": "other_allele_contrast_outside_scope", "scope_relevant": False}
                                for line, row, _ in items if contrast(row) != contrast(selected))
                continue
            reason = "multiple_allele_contrasts" if len(contrasts) > 1 else "conflicting_ld_records"
            excluded.extend({"line": line, "reported_rsid": row["rsid"], "r2": row["r2"],
                             "reason_code": reason, "scope_relevant": relevant}
                            for line, row, relevant in items)
        else:
            rows.append(first)
            duplicate_count += len(items) - 1
    return {"rows": rows, "source_self_row": source, "queried_rsid": variant["rsid"],
            "reported_self_rsid": source["rsid"], "excluded_rows": sorted(excluded, key=lambda row: row["line"]),
            "raw_row_count": raw_count, "duplicate_row_count": duplicate_count,
            "requested_scope_complete": not any(row["scope_relevant"] for row in excluded)}


def _parse_pair(text, selected, prior):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise _failure(_error_reason(text)) from None
    if isinstance(value, list):
        if len(value) != 1:
            raise _failure()
        value = value[0]
    if not isinstance(value, Mapping):
        raise _failure()
    _check_error(value)
    identities = []
    for field, expected in (("snp1", selected), ("snp2", prior)):
        row = value.get(field)
        if not isinstance(row, Mapping):
            raise _failure()
        chrom, pos = _coordinate(row.get("coord"))
        parts = [row.get("allele_1"), row.get("allele_2")]
        if any(not isinstance(x, Mapping) for x in parts):
            raise _failure()
        # Counts preserve rare observed alleles whose rounded frequency is zero.
        for part in parts:
            try:
                count = int(part["count"])
            except (KeyError, TypeError, ValueError):
                raise _failure() from None
            if count < 0 or isinstance(part["count"], bool):
                raise _failure()
            if count == 0:
                raise _failure("monomorphic_in_selected_population")
        identity = {"rsid": row.get("rsnum"), "chromosome": chrom, "position": pos,
                    "alleles": _alleles([x.get("allele") for x in parts])}
        _assert_variant(identity, expected)
        identities.append(identity)
    stats = value.get("statistics")
    if not isinstance(stats, Mapping):
        raise _failure()
    return {"r2": _number(stats.get("r2")), "d_prime": _number(stats.get("d_prime")), "identities": identities}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _failure("redirect_rejected")


class LDLinkReferenceProvider(LDReferenceProvider):
    provider_name = "ldlink"
    provider_version = "1.0.0"
    genome_build = "grch38"
    reference_panel = "1000GENOMES:phase_3"
    ldlink = "https://ldlink.nih.gov/LDlinkRest"
    supports_source_identity_batch = True

    def __init__(self, *, token, ldlink_fetch=None, identity_resolver=None, allow_ensembl_fallback=None, **kwargs):
        super().__init__(**kwargs)
        self._ldlink_token = token
        self._ldlink_fetch = ldlink_fetch
        self._proxy_metadata = {}
        self.proxy_lookup_issues = {}
        from .ldlink_identity import LDLinkIdentityBridge
        self._identity_bridge = LDLinkIdentityBridge(self, identity_resolver,
                                                   allow_ensembl_fallback=allow_ensembl_fallback)

    def resolve(self, source_id):
        result = self.resolve_many([source_id])[source_id]
        if isinstance(result, Exception):
            raise result
        return result

    def resolve_many(self, source_ids):
        return self._identity_bridge.source_many(source_ids)

    def proxy_reference_identities(self, rsids, population):
        return self._identity_bridge.proxy_many(rsids, population)

    def _remaining(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CancelledError()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise _failure("total_budget_exhausted")
        return remaining

    @contextmanager
    def _locked(self, lock):
        while not lock.acquire(timeout=min(.1, self._remaining())):
            pass
        try:
            self._remaining()
            yield
        finally:
            lock.release()

    def _delay(self, seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            delay = min(.1, self._remaining(), until - time.monotonic())
            if self.cancel_event is not None:
                self.cancel_event.wait(delay)
            else:
                time.sleep(delay)

    def _notify(self, status, **details):
        emit_progress(self.progress_sink, "ld_reference_request", reference_provider=self.provider_name,
                      provider_version=self.provider_version, status=status, **details)

    def _transport(self, url, *, timeout):
        """Final authentication boundary; unit tests require ldlink_fetch injection."""
        token = self._ldlink_token() if callable(self._ldlink_token) else self._ldlink_token
        if not isinstance(token, str) or not token.strip():
            raise _failure("authentication_failed")
        request = Request(url + "&" + urlencode({"token": token}), headers={
            "Accept": "application/json,text/plain", "User-Agent": "GWASMind-LDlink/1.0"})
        if self._ldlink_fetch is not None:
            body = self._ldlink_fetch(request, timeout)
            body = body.encode("utf-8") if isinstance(body, str) else body
        else:
            with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
                chunks, size = [], 0
                cap = parameter("transport.graphql_max_response_bytes")
                while True:
                    self._remaining()
                    chunk = response.read1(min(65536, cap + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > cap:
                        raise _failure("response_limit_exceeded")
                body = b"".join(chunks)
        if not isinstance(body, bytes) or len(body) > parameter("transport.graphql_max_response_bytes"):
            raise _failure("response_limit_exceeded")
        return self._sanitize_response(body.decode("utf-8-sig"), token)

    @staticmethod
    def _sanitize_response(text, token):
        redacted = False
        # JSON's request field is operational and can echo authentication.
        if text.lstrip().startswith(("{", "[")):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            def clean(value):
                nonlocal redacted
                if isinstance(value, dict):
                    result = {}
                    for key, item in value.items():
                        if key.lower() in {"request", "token", "api_key", "authorization"}:
                            redacted = True
                        else:
                            result[key] = clean(item)
                    return result
                if isinstance(value, list):
                    return [clean(item) for item in value]
                return value
            if data is not None:
                text = json.dumps(clean(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for secret in {token, quote(token, safe=""), json.dumps(token)[1:-1]}:
            if secret in text:
                # A reflected secret must never enter a raw scientific artifact.
                text = text.replace(secret, "[redacted]")
                redacted = True
        return text, redacted

    def _request_ldlink(self, endpoint, params, validate):
        global _LAST_COMPLETED
        self._remaining()
        url = self.ldlink + "/" + endpoint + "?" + urlencode(params)
        query_build = "GRCh37" if params.get("genome_build") == "grch37" else "GRCh38"
        request = {**LDResponseCache.request(url, None), "provider": self.provider_name,
                   "provider_version": self.provider_version, "reference_build": query_build,
                   "reference_panel": self.reference_panel, "response_contract": "ldlink-raw-envelope-v1"}
        key = LDResponseCache.key(request)
        operation_id = uuid4().hex
        with self._state_lock:
            lock = self.response_cache.lock(key) if self.response_cache else self._request_locks.setdefault(key, threading.RLock())
        with self._locked(lock):
            if key in self._failed_requests:
                self.request_summary["failed_cache_hits"] += 1
                raise self._failed_requests[key]
            cached = None
            if key in self._receipts:
                cached = self._cache[key], self._receipts[key]
            elif self.response_cache is not None:
                try:
                    cached = self.response_cache.get(request)
                except (OSError, ValueError, KeyError, TypeError):
                    self.request_summary["invalid_cache_entries"] += 1
            if cached is not None:
                value, prior = cached
                if not isinstance(value, Mapping) or value.get("format") != "ldlink-raw-envelope-v1":
                    raise _failure()
                parsed = validate(value["raw_response"])
                receipt = {**prior, "url": url, "method": "GET", "request_body": None, "status": "success",
                           "request_key": key, "from_cache": True, "observed_at": datetime.now(timezone.utc).isoformat(),
                           "reference_provider": self.provider_name, "provider_version": self.provider_version,
                           "query_build": query_build, "analysis_build": "GRCh38", "population": params.get("pop")}
                self.ledger.append(receipt)
                self.request_summary["cache_hits"] += 1
                self._cache[key], self._receipts[key] = value, receipt
                return parsed, receipt
            with self._locked(_LDLINK_REQUEST_LOCK), serial_ldlink_request(deadline=self.deadline, cancel_event=self.cancel_event):
                retries = parameter("ld.ldlink_max_retries")
                for attempt in range(retries + 1):
                    self._delay(max(0, parameter("ld.ldlink_min_interval_seconds") - (time.monotonic() - _LAST_COMPLETED)))
                    started = time.monotonic()
                    receipt = {"url": url, "method": "GET", "request_body": None, "request_key": key,
                               "attempt": attempt + 1, "from_cache": False, "reference_provider": self.provider_name,
                               "provider_version": self.provider_version, "started_at": datetime.now(timezone.utc).isoformat(),
                               "query_build": query_build, "analysis_build": "GRCh38", "population": params.get("pop")}
                    self._notify("started" if attempt == 0 else "retrying", attempt=attempt + 1, operation_id=operation_id)
                    failure = None
                    retry_after = 0
                    text, redacted = None, False
                    try:
                        self.request_summary["network_requests"] += 1
                        text, redacted = self._transport(url, timeout=min(parameter("ld.ldlink_timeout_seconds"), self._remaining()))
                        self._remaining()
                        parsed = validate(text)
                        value = {"format": "ldlink-raw-envelope-v1", "raw_response": text,
                                 "credential_fields_removed": redacted}
                        raw = encode(value)
                        receipt.update(status="success", retrieved_at=datetime.now(timezone.utc).isoformat(),
                                       elapsed_seconds=time.monotonic() - started, sha256=hashlib.sha256(raw).hexdigest(),
                                       bytes=len(raw), credential_fields_removed=redacted)
                        if self.response_cache is not None:
                            reusable = not isinstance(parsed, Mapping) or parsed.get("requested_scope_complete") is not False
                            saved = self.response_cache.save(request, raw, retrieved_at=receipt["retrieved_at"], reusable=reusable)
                            receipt["response_path"] = saved["response_path"]
                        self.ledger.append(receipt)
                        self._cache[key], self._receipts[key] = value, receipt
                        self._notify("completed", attempt=attempt + 1, operation_id=operation_id)
                        return parsed, receipt
                    except CancelledError:
                        raise
                    except Exception as exc:
                        reason = getattr(exc, "reason_code", None)
                        code = exc.code if isinstance(exc, HTTPError) else None
                        if not reason:
                            reason = ("authentication_failed" if code in {401, 403} else
                                      "rate_limited" if code == 429 else "service_error" if code and code >= 500 else
                                      "invalid_response" if code else "request_timeout" if isinstance(exc, TimeoutError)
                                      or isinstance(getattr(exc, "reason", None), TimeoutError) else
                                      "connection_error" if isinstance(exc, (URLError, OSError)) else "invalid_response")
                        retry_after = (retry_after_seconds(exc.headers) or 0) if isinstance(exc, HTTPError) else 0
                        failure = _failure(reason)
                        receipt.update(status="failed", reason_code=reason, http_status=code,
                                       error_type="LDlinkLookupError", retrieved_at=datetime.now(timezone.utc).isoformat(),
                                       elapsed_seconds=time.monotonic() - started)
                        if text is None and isinstance(exc, HTTPError):
                            try:
                                body = exc.read(parameter("transport.graphql_max_response_bytes") + 1)
                                token = self._ldlink_token() if callable(self._ldlink_token) else self._ldlink_token
                                if (isinstance(token, str) and token
                                        and len(body) <= parameter("transport.graphql_max_response_bytes")):
                                    text, redacted = self._sanitize_response(body.decode("utf-8-sig"), token)
                            except (OSError, UnicodeError, ValueError, TypeError):
                                pass
                        if text is not None:
                            raw = encode({"format": "ldlink-raw-envelope-v1", "raw_response": text,
                                          "credential_fields_removed": redacted})
                            receipt.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                                           credential_fields_removed=redacted, reusable=False,
                                           response_excerpt=text[:4096], response_excerpt_truncated=len(text) > 4096)
                            if self.response_cache is not None:
                                saved = self.response_cache.save(request, raw,
                                    retrieved_at=receipt["retrieved_at"], reusable=False)
                                receipt["response_path"] = saved["response_path"]
                        self.ledger.append(receipt)
                    finally:
                        _LAST_COMPLETED = time.monotonic()
                    # Raise outside the exception handler: no secret-bearing cause/context.
                    retryable = failure.reason_code in {"rate_limited", "service_error", "request_timeout", "connection_error"}
                    if not retryable or attempt == retries:
                        receipt["recovery_exhausted"] = True
                        self._failed_requests[key] = failure
                        self._notify("incomplete", attempt=attempt + 1, reason=failure.reason_code, operation_id=operation_id)
                        raise failure from None
                    self._delay(max(retry_after, parameter("ld.ldlink_min_interval_seconds") * 2 ** attempt))

    def _population(self, population):
        if not isinstance(population, str) or population not in _POPULATIONS:
            raise ValueError("LDlink requires one explicitly selected 1000 Genomes population code.")
        if parameter("novelty.reference_build") != "GRCh38" or parameter("novelty.reference_panel") != self.reference_panel:
            raise ValueError("LDlink adapter requires the recorded GRCh38 phase 3 reference panel.")
        return population

    @staticmethod
    def _provenance(receipt):
        return {"provider": "ldlink", "provider_version": "1.0.0", "genome_build": "GRCh38",
                "query_build": receipt.get("query_build", "GRCh38"), "analysis_build": "GRCh38",
                "reference_panel": "1000GENOMES:phase_3", "ld_request_key": receipt["request_key"],
                "ld_response_sha256": receipt["sha256"], "ld_response_path": receipt.get("response_path"),
                "ld_response_retrieved_at": receipt["retrieved_at"]}

    def proxy_metadata(self, rsid, population):
        with self._state_lock:
            return deepcopy(self._proxy_metadata.get((population, rsid)))

    def proxies(self, variant, population, *, threshold=None, window_bp=None):
        population = self._population(population)
        threshold = parameter("novelty.ld_r2_threshold") if threshold is None else threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not .01 < threshold <= 1:
            raise ValueError("LDlink proxy threshold must be finite and greater than its 0.01 output floor.")
        flank = parameter("novelty.ld_window_bp") if window_bp is None else window_bp
        if isinstance(flank, bool) or not isinstance(flank, int) or not 0 <= flank <= 1_000_000:
            raise ValueError("LDlink window must be an integer between zero and 1000000 bp per side.")
        if not _RSID.fullmatch(str(variant.get("rsid", ""))):
            raise ValueError("LDlink proxy queries require a reference-resolved rsID.")
        params = {"var": variant["rsid"], "pop": population, "r2_d": "r2", "window": flank, "genome_build": self.genome_build}
        try:
            parsed, receipt = self._request_ldlink("ldproxy", params,
                lambda text: _parse_proxy(text, variant, threshold=threshold, window_bp=flank))
        except RuntimeError as exc:
            if getattr(exc, "reason_code", None) != "not_found_in_queried_reference":
                raise
            from .ldlink_fallback import grch37_proxies
            parsed, receipt = grch37_proxies(self, variant, params, threshold=threshold, flank=flank)
        rows, source = parsed["rows"], parsed["source_self_row"]
        provenance = self._provenance(receipt)
        output = []
        for row in rows:
            if abs(row["distance"]) > flank or row["r2"] < threshold:
                continue
            metadata = {"alleles": row["alleles"], "chromosome": row["chromosome"], "position": row["position"],
                        "end": row["position"], "population": population, "genome_build": "GRCh38",
                        "reference_panel": self.reference_panel, "response_sha256": receipt["sha256"],
                        "request_key": receipt["request_key"], "response_path": receipt.get("response_path"),
                        "retrieved_at": receipt["retrieved_at"], "source_self_row": deepcopy(source),
                        "queried_rsid": variant["rsid"], "reported_self_rsid": source["rsid"]}
            mapping = {key: deepcopy(row[key]) for key in ("query_build", "analysis_build", "query_chromosome",
                       "query_position", "query_distance", "build_identity") if key in row}
            metadata.update(mapping)
            with self._state_lock:
                old = self._proxy_metadata.get((population, row["rsid"]))
                if old is not None and (old.get("conflict") or any(old.get(k) != metadata[k] for k in ("alleles", "chromosome", "position"))):
                    metadata["conflict"] = True
                self._proxy_metadata[(population, row["rsid"])] = metadata
            output.append({"selected_variant_id": variant["source_variant_id"], "selected_rsid": variant["rsid"],
                           "reported_rsid": row["rsid"], "population": population, "r2": row["r2"],
                           "d_prime": row["d_prime"], "status": "computed", "chromosome": row["chromosome"],
                           "position": row["position"], "end": row["position"], "population_alleles": row["alleles"],
                           "selected_population_alleles": source["alleles"], "queried_rsid": variant["rsid"],
                           "reported_self_rsid": source["rsid"], **provenance, **mapping})
        audit = {key: deepcopy(parsed[key]) for key in ("raw_row_count", "duplicate_row_count", "excluded_rows",
                 "requested_scope_complete", "queried_rsid", "reported_self_rsid")}
        audit.update(excluded_scope_count=sum(row["scope_relevant"] for row in parsed["excluded_rows"]),
                     retained_proxy_count=len(output), threshold=threshold, window_bp=flank,
                     request_key=receipt["request_key"], response_sha256=receipt["sha256"])
        audit.update({key: deepcopy(parsed[key]) for key in ("query_build", "analysis_build", "fallback_reason",
                      "query_source_self_row", "window_coverage_reason") if key in parsed})
        state = ("SUCCESS_DATA" if output else "SUCCESS_NO_DATA") if parsed["requested_scope_complete"] else "PARTIAL"
        receipt.update(state=state, complete=parsed["requested_scope_complete"], **audit)
        with self._state_lock:
            self.proxy_lookup_issues[(variant["source_variant_id"], population)] = {"state": state, **audit}
        return sorted(output, key=lambda row: (row["position"], row["reported_rsid"]))

    def pair(self, selected, prior, population):
        population = self._population(population)
        if any(not _RSID.fullmatch(str(row.get("rsid", ""))) for row in (selected, prior)):
            raise ValueError("LDlink pair queries require reference-resolved rsIDs.")
        params = {"var1": selected["rsid"], "var2": prior["rsid"], "pop": population,
                  "genome_build": self.genome_build, "json_out": "true"}
        parsed, receipt = self._request_ldlink("ldpair", params, lambda text: _parse_pair(text, selected, prior))
        receipt["state"] = "SUCCESS_DATA"
        alleles = {row["rsid"]: row["alleles"] for row in parsed["identities"]}
        return {"selected_variant_id": selected["source_variant_id"], "selected_rsid": selected["rsid"],
                "reported_rsid": prior["rsid"], "population": population, "r2": parsed["r2"],
                "d_prime": parsed["d_prime"], "status": "computed", "verified_panel_alleles": alleles,
                "selected_population_alleles": alleles[selected["rsid"]],
                "population_alleles": alleles[prior["rsid"]], **self._provenance(receipt)}
