"""OT-first reference identity for LDlink, with explicit basic-mapping fallback.

LDlink supplies the population allele contrast. OT or an independently checked
GRCh38 mapping supplies REF orientation. Neither is inferred from display order.
"""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import CancelledError
from copy import deepcopy
import re
import threading
import time

from ..configuration import parameter
from .ld import LDReferenceProvider, RECOVERABLE_LD_REASONS, _reference_error, reference_failure_reason

_RSID = re.compile(r"rs\d+")
_CANONICAL = re.compile(r"(?:chr)?([1-9]|1[0-9]|2[0-2]|X|Y)[_:](\d+)[_:]([ACGT]+)[_:]([ACGT]+)", re.I)
_STATUSES = {"resolved", "missing", "failed", "ambiguous", "conflicting"}


def _canonical(identifier):
    match = _CANONICAL.fullmatch(identifier)
    if match is None:
        return None
    chrom, pos, ref, alt = match.groups()
    return f"{chrom.upper()}_{int(pos)}_{ref.upper()}_{alt.upper()}"


def _valid_identity(value):
    if not isinstance(value, Mapping) or value.get("genome_build") != "GRCh38":
        return False
    chrom, pos = value.get("chromosome"), value.get("position")
    ref, alt = value.get("reference_allele"), value.get("alternate_allele")
    return (isinstance(chrom, str) and re.fullmatch(r"[1-9]|1[0-9]|2[0-2]|X|Y", chrom) is not None
            and isinstance(pos, int) and not isinstance(pos, bool) and pos > 0
            and isinstance(ref, str) and isinstance(alt, str)
            and re.fullmatch(r"[ACGT]+", ref) is not None and re.fullmatch(r"[ACGT]+", alt) is not None and ref != alt
            and value.get("variant_id") == f"{chrom}_{pos}_{ref}_{alt}"
            and isinstance(value.get("rsids"), list)
            and all(isinstance(x, str) and _RSID.fullmatch(x) for x in value["rsids"]))


def _minimal_edit(position, first, second):
    """Compare representations without inventing a reference base or left shift."""
    first, second = first.replace("-", ""), second.replace("-", "")
    while first and second and first[0] == second[0]:
        first, second, position = first[1:], second[1:], position + 1
    while first and second and first[-1] == second[-1]:
        first, second = first[:-1], second[:-1]
    return position, frozenset((first, second))


def _canonical_snv_ids(meta):
    """Return both explicit SNV orientations without guessing an indel anchor."""
    alleles = meta.get("alleles")
    if (meta.get("genome_build") != "GRCh38" or meta.get("conflict")
            or not isinstance(meta.get("chromosome"), str)
            or re.fullmatch(r"[1-9]|1[0-9]|2[0-2]|X|Y", meta["chromosome"]) is None
            or type(meta.get("position")) is not int or meta["position"] <= 0
            or meta.get("end", meta["position"]) != meta["position"]
            or not isinstance(alleles, (list, tuple)) or len(alleles) != 2
            or any(not isinstance(allele, str) or allele not in {"A", "C", "G", "T"} for allele in alleles)
            or alleles[0] == alleles[1]):
        return ()
    first, second = sorted(alleles)
    prefix = f'{meta["chromosome"]}_{meta["position"]}'
    return f"{prefix}_{first}_{second}", f"{prefix}_{second}_{first}"


def valid_canonical_binding(binding, identity, meta, rsid):
    """Validate an explicit coordinate proof while keeping provider rsIDs distinct."""
    if not isinstance(binding, Mapping) or not _valid_identity(identity):
        return False
    ids = _canonical_snv_ids(meta)
    if (len(ids) != 2 or binding.get("schema") != "ot_ldlink_snv_identity_v1"
            or binding.get("ldlink_rsid") != rsid
            or binding.get("ot_rsids") != identity["rsids"]
            or binding.get("variant_id") != identity["variant_id"]
            or identity["variant_id"] not in ids
            or binding.get("genome_build") != "GRCh38"
            or binding.get("population") != meta.get("population")
            or binding.get("population_request_key") != meta.get("request_key")
            or binding.get("population_response_sha256") != meta.get("response_sha256")
            or not meta.get("request_key") or not meta.get("response_sha256")):
        return False
    checks = binding.get("orientation_checks")
    if (not isinstance(checks, list) or len(checks) != 2
            or any(not isinstance(row, Mapping) for row in checks)
            or {row.get("requested_variant_id") for row in checks} != set(ids)):
        return False
    selected = [row for row in checks if row.get("status") == "resolved"]
    other = [row for row in checks if row.get("status") != "resolved"]
    if (len(selected) != 1 or len(other) != 1
            or selected[0].get("requested_variant_id") != identity["variant_id"]
            or selected[0].get("observed_variant_id") != identity["variant_id"]
            or other[0].get("status") not in {"missing", "reverse_orientation_absent"}
            or (other[0]["status"] == "reverse_orientation_absent"
                and other[0].get("observed_variant_id") != identity["variant_id"])):
        return False
    for check in checks:
        receipts = check.get("receipts")
        if (not isinstance(receipts, list) or not receipts
                or any(not isinstance(row, Mapping) or row.get("status") != "success"
                       or not row.get("request_key") or not row.get("sha256") for row in receipts)):
            return False
    return True


def matches_population_identity(identity, meta, rsid, *, canonical_binding=None):
    """Require the exact rsID or a separately proven canonical SNV identity.

    LDlink can represent a deletion as (C/-) at the deleted base while OT's
    reference-anchored ID is AC_A at the preceding base. Removing shared anchors
    proves that equivalence. Repeat-driven shifts and inferred anchors are not
    accepted: they need a separate reference-sequence normalization service.
    """
    if (not _valid_identity(identity)
            or (canonical_binding is not None and not valid_canonical_binding(canonical_binding, identity, meta, rsid))
            or (canonical_binding is None and rsid not in identity["rsids"])
            or identity["chromosome"] != meta.get("chromosome")
            or identity["genome_build"] != meta.get("genome_build", "GRCh38")):
        return False
    alleles = meta.get("alleles")
    if (not isinstance(alleles, (list, tuple)) or len(alleles) != 2
            or not all(isinstance(x, str) and re.fullmatch(r"[ACGT]+|-", x) for x in alleles)
            or len(set(alleles)) != 2
            or type(meta.get("position")) is not int):
        return False
    return _minimal_edit(identity["position"], identity["reference_allele"], identity["alternate_allele"]) == _minimal_edit(meta["position"], *alleles)


def _receipts(result):
    identity = result.get("identity")
    values = [*(result.get("receipts") or []),
              *((identity.get("receipts") or []) if isinstance(identity, Mapping) else [])]
    for candidate in result.get("candidates") or []:
        if isinstance(candidate, Mapping):
            values.extend(candidate.get("receipts") or [])
    unique = {}
    for value in values:
        if isinstance(value, Mapping):
            key = (value.get("request_key"), value.get("sha256"), value.get("status"), value.get("from_cache"))
            unique[key] = deepcopy(dict(value))
    return list(unique.values())


def _outcome(status, *, identity=None, provider="opentargets", receipts=(), reason=None):
    binding_receipts = identity.get("receipts") if isinstance(identity, Mapping) else None
    successful = [row for row in (binding_receipts or receipts) if row.get("status") == "success"]
    return {"status": status, "identity": deepcopy(identity), "provider": provider,
            "receipts": deepcopy(list(receipts)), "receipt": deepcopy(successful[-1]) if successful else None,
            "reason_code": reason}


class LDLinkIdentityBridge:
    def __init__(self, provider, resolver=None, *, allow_ensembl_fallback=None):
        self.provider = provider
        self.resolver = resolver
        self._lock = threading.RLock()
        self._source_results = {}
        self._fallback_enabled = (parameter("ld.ensembl_fallback_enabled")
                                  if allow_ensembl_fallback is None else allow_ensembl_fallback)
        if not isinstance(self._fallback_enabled, bool):
            raise ValueError("Ensembl fallback must be explicitly enabled or disabled")
        self._fallback_budget_seconds = parameter("ld.ensembl_fallback_budget_seconds")
        self._fallback_elapsed_seconds = 0.0

    def _fallback_call(self, operation, *args, **kwargs):
        """Share one supplement budget without shortening primary OT/LDlink work.

        The bridge lock serializes these scopes. Cached positive mappings remain
        readable at zero budget; the base transport checks this deadline before
        starting any new request, including batch retry/split requests.
        """
        started = time.monotonic()
        primary_deadline = self.provider.deadline
        remaining = max(0.0, self._fallback_budget_seconds - self._fallback_elapsed_seconds)
        self.provider.deadline = min(primary_deadline, started + remaining)
        try:
            return operation(*args, **kwargs)
        finally:
            self._fallback_elapsed_seconds += max(0.0, time.monotonic() - started)
            self.provider.deadline = primary_deadline
            if self._fallback_elapsed_seconds >= self._fallback_budget_seconds:
                # Batch failures are recorded below this adapter. They must not
                # reopen a fresh supplement budget through automatic LD recovery.
                # Original HTTP error receipts stay untouched in the ledger.
                for details in self.provider.identity_failure_details.values():
                    if details.get("reason") in RECOVERABLE_LD_REASONS:
                        details["reason"] = "ensembl_fallback_budget_exhausted"
                for details in self.provider.source_identity_failure_details.values():
                    if (details.get("reference_provider") == "ensembl"
                            and details.get("reason") in RECOVERABLE_LD_REASONS):
                        details["reason"] = "ensembl_fallback_budget_exhausted"

    def _fallback_failure_reason(self, exc):
        reason = reference_failure_reason(exc)
        if (self._fallback_elapsed_seconds >= self._fallback_budget_seconds
                and reason in RECOVERABLE_LD_REASONS):
            return "ensembl_fallback_budget_exhausted"
        return reason

    def _mark_exhausted_failures(self, outcomes):
        if self._fallback_elapsed_seconds >= self._fallback_budget_seconds:
            for outcome in outcomes.values():
                if (outcome.get("provider") == "ensembl" and outcome.get("status") == "failed"
                        and outcome.get("reason_code") in RECOVERABLE_LD_REASONS):
                    outcome["reason_code"] = "ensembl_fallback_budget_exhausted"

    def _resolver(self):
        if self.resolver is None:
            from .ot_identity import OTIdentityResolver
            p = self.provider
            self.resolver = OTIdentityResolver(deadline=p.deadline, cancel_event=p.cancel_event,
                progress_sink=p.progress_sink, cache_directory=p.response_cache.directory if p.response_cache else None,
                max_retries=parameter("ld.ot_identity_max_retries"))
        return self.resolver

    def _append_receipts(self, receipts, provider):
        parent = self.provider
        with parent._state_lock:
            known = {(x.get("reference_provider"), x.get("request_key"), x.get("sha256"),
                      x.get("status"), x.get("from_cache"), x.get("attempt"), x.get("started_at")) for x in parent.ledger}
            for receipt in receipts:
                row = {**receipt, "reference_provider": provider}
                identity = (provider, row.get("request_key"), row.get("sha256"), row.get("status"), row.get("from_cache"),
                            row.get("attempt"), row.get("started_at"))
                if identity not in known:
                    parent.ledger.append(row)
                    known.add(identity)
                    parent.request_summary["cache_hits" if row.get("from_cache") else "network_requests"] += 1

    def _lookup(self, ids):
        self.provider._remaining()
        try:
            resolver = self._resolver()
            results = resolver.resolve_many(ids)
            self._append_receipts(getattr(resolver, "ledger", ()), "opentargets")
        except CancelledError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            reason = reference_failure_reason(exc)
            return {item: {"status": "failed", "identity": None, "candidates": [], "receipts": [],
                           "fallback_eligible": True, "reason_code": reason} for item in ids}
        output = {}
        for identifier in ids:
            result = results.get(identifier) if isinstance(results, Mapping) else None
            if not isinstance(result, Mapping) or result.get("status") not in _STATUSES:
                result = {"status": "failed", "reason_code": "invalid_identity_response", "fallback_eligible": True}
            receipts = _receipts(result)
            self._append_receipts(receipts, "opentargets")
            if result.get("status") == "failed":
                transport = next((row.get("transport_reason") for row in reversed(receipts)
                                  if row.get("status") == "failed" and row.get("transport_reason")), None)
                if transport in RECOVERABLE_LD_REASONS:
                    result = {**result, "identity_reason_code": result.get("reason_code"), "reason_code": transport}
            output[identifier] = {**result, "receipts": receipts}
        return output

    @staticmethod
    def _candidates(result):
        values = [*(result.get("candidates") or [])]
        if result.get("identity") is not None:
            values.append(result["identity"])
        if any(not _valid_identity(value) for value in values):
            return None
        return list({value["variant_id"]: deepcopy(value) for value in values}.values())

    def _source_choice(self, identifier, result):
        receipts = result["receipts"]
        if result["status"] == "conflicting":
            from ..source_normalization import build_source_normalization

            proof = build_source_normalization(identifier, result)
            if proof is not None and proof["identity"].get("rsids") and all(
                any(row.get("status") == "success" and row.get("request_key") == receipt.get("request_key")
                    and row.get("sha256") == receipt.get("sha256") for row in self.provider.ledger)
                for receipt in proof["receipts"]
            ):
                return {**_outcome("resolved", identity=proof["identity"], receipts=receipts),
                        "source_normalization": proof}
            return _outcome("conflict", receipts=receipts, reason="reference_identity_conflict")
        candidates = self._candidates(result)
        if candidates is None:
            return _outcome("conflict", receipts=receipts, reason="invalid_reference_identity")
        expected = _canonical(identifier)
        selected = [row for row in candidates if row["variant_id"] == expected] if expected else [
            row for row in candidates if identifier.lower() in row["rsids"]]
        if result["status"] == "failed":
            if candidates and (len(selected) != 1 or len(candidates) != 1):
                return _outcome("conflict", receipts=receipts, reason="reference_identity_conflict")
            return _outcome("failed", identity=selected[0] if selected else None,
                            receipts=receipts, reason=result.get("reason_code"))
        if len(selected) == 1:
            identity = selected[0]
            if not identity["rsids"]:
                return _outcome("missing", identity=identity, receipts=receipts, reason="missing_rsid")
            return _outcome("resolved", identity=identity, receipts=receipts)
        if candidates or result["status"] in {"resolved", "ambiguous"}:
            return _outcome("conflict", receipts=receipts, reason="reference_identity_ambiguous")
        return _outcome(result["status"], receipts=receipts, reason=result.get("reason_code"))

    def _basic_preload(self, ids, scope):
        if not ids or not self.provider.supports_batch_identity:
            return
        self._fallback_call(LDReferenceProvider.preload_variations, self.provider, ids,
                            identity_scope=scope, populations=False)

    def _fallback_receipts(self, start):
        receipts = self.provider.ledger[start:]
        for receipt in receipts:
            # Only actual basic Ensembl requests occur in this scope.
            if not receipt.get("reference_provider"):
                receipt["reference_provider"] = "ensembl"
        return deepcopy(receipts)

    def _basic_binding(self, rsid):
        entry = self.provider._variations.get(rsid)
        if entry and entry.get("parent_receipt"):
            receipt = entry["parent_receipt"]
            receipt.setdefault("reference_provider", "ensembl")
            return [deepcopy(receipt)]
        matches = [row for row in self.provider.ledger if row.get("status") == "success"
                   and row.get("url", "").split("?")[0].endswith("/variation/human/" + rsid)]
        return deepcopy(matches[-1:])

    def _raise_basic_failure(self, rsids):
        for rsid in rsids:
            if rsid not in self.provider._variations and (rsid, False) in self.provider._variation_failures:
                reason = self.provider.identity_failure_details.get((rsid, False), {}).get("reason", "invalid_response")
                raise _reference_error("Fallback reference lookup did not complete.", reason)

    def source_many(self, source_ids):
        ids = list(dict.fromkeys(str(item) for item in source_ids))
        with self._lock:
            pending = [item for item in ids if item not in self._source_results]
            if not pending:
                return {item: deepcopy(self._source_results[item]) for item in ids}
            results = self._lookup(pending)
            choices = {item: self._source_choice(item, results[item]) for item in pending}
            fallback = [item for item in pending if self._fallback_enabled and choices[item]["status"] in {"missing", "failed"}
                        and (results[item].get("fallback_eligible") is True or choices[item]["reason_code"] == "missing_rsid")]
            # Discover only missing source identities, then batch BASIC mappings.
            start = len(self.provider.ledger)
            if fallback:
                self._fallback_call(LDReferenceProvider._prefetch_source_regions, self.provider, fallback)
            candidates = {}
            for item in fallback:
                try:
                    candidates[item] = self._fallback_call(LDReferenceProvider._source_candidates,
                                                          self.provider, item)[0]
                except CancelledError:
                    raise
                except (RuntimeError, ValueError, OSError) as exc:
                    choices[item] = _outcome("failed", provider="ensembl", receipts=choices[item]["receipts"],
                                             reason=self._fallback_failure_reason(exc))
            self._basic_preload(sorted({rsid for value in candidates.values() for rsid in value}), "source")
            fallback_receipts = self._fallback_receipts(start)
            for item in candidates:
                try:
                    self._raise_basic_failure(candidates[item])
                    base = self._fallback_call(LDReferenceProvider.resolve, self.provider, item)
                    ref = base.get("reference_allele")
                    selected = base.get("selected_alleles", [])
                    if len(selected) != 2 or ref not in selected:
                        raise _reference_error("No unique source reference allele contrast.", "reference_identity_ambiguous")
                    alt = next(x for x in selected if x != ref)
                    identity = {"variant_id": f'{base["chromosome"]}_{base["position"]}_{ref}_{alt}',
                        "chromosome": base["chromosome"], "position": base["position"],
                        "reference_allele": ref, "alternate_allele": alt, "rsids": [base["rsid"]], "genome_build": "GRCh38"}
                    expected = choices[item].get("identity")
                    if (not _valid_identity(identity) or (expected and identity["variant_id"] != expected["variant_id"])
                            or (_canonical(item) and identity["variant_id"] != _canonical(item))):
                        raise _reference_error("Reference providers disagree on the source identity.", "reference_identity_conflict")
                    identity["receipts"] = self._basic_binding(base["rsid"])
                    extra = self._fallback_receipts(start)
                    choices[item] = _outcome("resolved", identity=identity, provider="ensembl",
                        receipts=[*choices[item]["receipts"], *extra])
                except CancelledError:
                    raise
                except (RuntimeError, ValueError, OSError, StopIteration) as exc:
                    reason = self._fallback_failure_reason(exc)
                    choices[item] = _outcome("conflict" if "ambiguous" in reason or "conflict" in reason else "failed",
                        provider="ensembl", receipts=[*choices[item]["receipts"], *fallback_receipts], reason=reason)
            self._mark_exhausted_failures(choices)
            for item, choice in choices.items():
                if choice["status"] != "resolved":
                    reason = choice["reason_code"] or "reference_identity_unresolved"
                    self.provider.source_identity_failure_details[item] = {"reason": reason,
                        "identity_status": choice["status"], "reference_provider": choice["provider"]}
                    self._source_results[item] = _reference_error("Source reference identity remains unresolved.", reason)
                    continue
                identity = choice["identity"]
                rsid = item.lower() if item.lower() in identity["rsids"] else min(identity["rsids"], key=lambda x: (int(x[2:]), x))
                ref, alt = identity["reference_allele"], identity["alternate_allele"]
                self._source_results[item] = {"source_variant_id": item, "rsid": rsid,
                    "canonical_variant_id": identity["variant_id"],
                    "chromosome": identity["chromosome"], "position": identity["position"],
                    "alleles": [ref, alt], "selected_alleles": [ref, alt], "reference_allele": ref,
                    "ld_alleles_unambiguous": True, "source_allele_order_matches_reference": _canonical(item) in {None, identity["variant_id"]},
                    "identity_provider": choice["provider"], "identity_receipts": choice["receipts"],
                    "reference_identity": identity}
                if choice.get("source_normalization"):
                    self._source_results[item]["source_normalization"] = deepcopy(choice["source_normalization"])
            return {item: deepcopy(self._source_results[item]) for item in ids}

    def proxy_many(self, rsids, population):
        ids = sorted(set(rsids))
        with self._lock:
            metadata = {item: self.provider.proxy_metadata(item, population) for item in ids}
            output, eligible = {}, []
            for item, meta in metadata.items():
                if not isinstance(meta, Mapping) or meta.get("conflict") or meta.get("population") != population:
                    output[item] = _outcome("conflict", reason="population_identity_unverified")
                else:
                    eligible.append(item)
            results = self._lookup(eligible) if eligible else {}
            fallback = []
            for item in eligible:
                result, meta = results[item], metadata[item]
                candidates = self._candidates(result)
                if candidates is None or result["status"] == "conflicting":
                    output[item] = _outcome("conflict", receipts=result["receipts"], reason="reference_identity_conflict")
                    continue
                matches = [row for row in candidates if matches_population_identity(row, meta, item)]
                if result["status"] == "failed":
                    output[item] = _outcome("failed", identity=matches[0] if len(matches) == 1 else None,
                                           receipts=result["receipts"], reason=result.get("reason_code"))
                    if self._fallback_enabled and result.get("fallback_eligible") is True:
                        fallback.append(item)
                elif len(matches) == 1:
                    output[item] = _outcome("resolved", identity=matches[0], receipts=result["receipts"])
                elif candidates or result["status"] in {"resolved", "ambiguous"}:
                    output[item] = _outcome("conflict", receipts=result["receipts"], reason="reference_identity_conflict")
                else:
                    output[item] = _outcome(result["status"], receipts=result["receipts"], reason=result.get("reason_code"))
                    if self._fallback_enabled and result.get("fallback_eligible") is True and result["status"] in {"missing", "failed"}:
                        fallback.append(item)
            # OT and LDlink can name the same canonical SNV with different rsIDs.
            # Only a successfully exhausted, empty exact-alias search is eligible;
            # genuine conflicts and incomplete searches retain their prior state.
            canonical = {item: _canonical_snv_ids(metadata[item]) for item in eligible
                         if output[item]["status"] == "missing"
                         and results[item].get("reason_code") == "ot_exact_rsid_not_found"
                         and not self._candidates(results[item])}
            canonical = {item: ids for item, ids in canonical.items() if ids}
            self._resolve_canonical_proxies(canonical, metadata, output)
            fallback = [item for item in fallback if output[item]["status"] in {"missing", "failed"}]
            start = len(self.provider.ledger)
            self._basic_preload(fallback, "proxy")
            for item in fallback:
                meta = metadata[item]
                try:
                    self._raise_basic_failure([item])
                    record = self._fallback_call(LDReferenceProvider.variation, self.provider,
                                                 item, populations=False)
                    if not isinstance(record, Mapping) or record.get("name") != item:
                        raise _reference_error("Fallback names a different reference variant.", "reference_identity_conflict")
                    candidates = {}
                    for row in record.get("mappings", []):
                        if not isinstance(row, Mapping):
                            continue
                        if (row.get("assembly_name") != "GRCh38" or row.get("coord_system") != "chromosome"
                                or row.get("strand") != 1 or str(row.get("seq_region_name")) != meta["chromosome"]
                                or row.get("start") != meta["position"] or row.get("end") != meta["position"]):
                            continue
                        alleles = str(row.get("allele_string", "")).split("/")
                        ref = alleles[0]
                        observed = set(meta["alleles"])
                        if len(observed) != 2 or ref not in observed or not observed.issubset(alleles):
                            continue
                        alt = next(x for x in observed if x != ref)
                        identity = {"variant_id": f'{meta["chromosome"]}_{meta["position"]}_{ref}_{alt}',
                            "chromosome": meta["chromosome"], "position": meta["position"], "reference_allele": ref,
                            "alternate_allele": alt, "rsids": [item], "genome_build": "GRCh38"}
                        identity["receipts"] = self._basic_binding(item)
                        if _valid_identity(identity):
                            candidates[identity["variant_id"]] = identity
                    if len(candidates) != 1:
                        raise _reference_error("No unique fallback identity matches the LD population contrast.", "reference_identity_conflict")
                    chosen = next(iter(candidates.values()))
                    observed_ot = self._candidates(results[item]) or []
                    if any(any(chosen[key] != observed[key] for key in ("chromosome", "position", "reference_allele"))
                           for observed in observed_ot):
                        raise _reference_error("Fallback contradicts an already observed OT reference constraint.", "reference_identity_conflict")
                    receipts = self._fallback_receipts(start)
                    output[item] = _outcome("resolved", identity=chosen, provider="ensembl",
                                           receipts=[*output[item]["receipts"], *receipts])
                except CancelledError:
                    raise
                except (RuntimeError, ValueError, OSError, StopIteration) as exc:
                    reason = self._fallback_failure_reason(exc)
                    output[item] = _outcome("conflict" if "conflict" in reason else "failed", provider="ensembl",
                        receipts=[*output[item]["receipts"], *self._fallback_receipts(start)], reason=reason)
            self._mark_exhausted_failures(output)
            return output

    def _resolve_canonical_proxies(self, targets, metadata, output):
        if not targets:
            return
        results = self._lookup(sorted({identifier for ids in targets.values() for identifier in ids}))
        for rsid, ids in targets.items():
            checks, candidates = [], []
            for identifier in ids:
                result = results[identifier]
                values = self._candidates(result)
                check = {"requested_variant_id": identifier, "status": result["status"],
                         "receipts": result["receipts"]}
                if result["status"] == "resolved" and values and len(values) == 1:
                    check["observed_variant_id"] = values[0]["variant_id"]
                    candidates.append(values[0])
                elif (result["status"] == "conflicting"
                      and result.get("reason_code") == "ot_reference_allele_order_conflict"):
                    order = result.get("allele_order_check", {})
                    if (values and len(values) == 1 and order.get("original_variant_id") == identifier
                            and order.get("reverse_status") == "resolved"
                            and order.get("reverse_variant_id") == values[0]["variant_id"]):
                        check.update(status="reverse_orientation_absent", observed_variant_id=values[0]["variant_id"])
                checks.append(check)
            receipts = [*output[rsid]["receipts"], *(row for check in checks for row in check["receipts"])]
            if len(candidates) != 1:
                if len(candidates) > 1 or any(row["status"] in {"conflicting", "ambiguous"} for row in checks):
                    output[rsid] = _outcome("conflict", receipts=receipts, reason="reference_identity_conflict")
                elif any(row["status"] == "failed" for row in checks):
                    reason = next((results[identifier].get("reason_code") for identifier in ids
                                   if results[identifier]["status"] == "failed"), "ot_canonical_identity_incomplete")
                    output[rsid] = _outcome("failed", receipts=receipts, reason=reason)
                else:
                    output[rsid] = _outcome("missing", receipts=receipts, reason="ot_canonical_variant_not_found")
                continue
            identity, meta = candidates[0], metadata[rsid]
            binding = {"schema": "ot_ldlink_snv_identity_v1", "ldlink_rsid": rsid,
                       "ot_rsids": identity["rsids"], "variant_id": identity["variant_id"],
                       "genome_build": "GRCh38", "population": meta["population"],
                       "population_request_key": meta.get("request_key"),
                       "population_response_sha256": meta.get("response_sha256"), "orientation_checks": checks}
            if valid_canonical_binding(binding, identity, meta, rsid):
                output[rsid] = {**_outcome("resolved", identity=identity, receipts=receipts),
                               "canonical_identity_binding": binding}
            else:
                failure_reason = next((results[identifier].get("reason_code") for identifier in ids
                                       if results[identifier]["status"] == "failed"), None)
                output[rsid] = _outcome("failed" if any(row["status"] == "failed" for row in checks) else "conflict",
                                       receipts=receipts, reason=failure_reason or "ot_canonical_identity_incomplete")
