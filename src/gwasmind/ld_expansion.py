"""Expand gene-analysis variants without rewriting the user's GWAS observations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from copy import deepcopy
from math import isfinite
from functools import partial
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError

from .configuration import parameter
from .novelty import choose_population
from .progress import emit_progress
from .providers.ld import LDReferenceProvider, RECOVERABLE_LD_REASONS, reference_failure_reason
from .providers.ld_factory import (
    LDProviderConfigurationError, create_ld_provider, ld_configuration_issue, ld_provider_metadata,
)

_REFERENCE_ERRORS = (RuntimeError, ValueError, OSError, TypeError, KeyError, AttributeError)


def _reference_failure_reason(exc):
    if getattr(exc, "reason_code", None):
        return exc.reason_code
    value = exc
    while value.__cause__ is not None:
        value = value.__cause__
    if isinstance(value, HTTPError):
        return "rate_limited" if value.code == 429 else "service_error"
    if isinstance(value, TimeoutError) or isinstance(getattr(value, "reason", None), TimeoutError):
        return "request_timeout"
    if isinstance(value, URLError):
        return "connection_error"
    if "budget" in str(exc).casefold():
        return "total_budget_exhausted"
    if "allele pair" in str(exc).casefold() or "population alleles" in str(exc).casefold():
        return "population_alleles_unverified"
    return "invalid_response" if isinstance(value, (ValueError, TypeError, KeyError, AttributeError)) else "reference_unavailable"


def _reference_objects(value, label):
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"Invalid {label} returned by the LD reference service.")
    return value


def _biallelic_proxy_identity(provider, rsid, pair, population):
    """A unique reference contrast plus an observed population LD edge fixes the alleles.

    This shortcut establishes identity only. It does not establish allele frequency,
    effect direction, cohort-specific LD, or any molecular effect of the source SNP.
    Multiallelic identities require explicit population alleles in the slow path.
    """
    value = pair.get("r2")
    if (pair.get("status") != "computed" or pair.get("population") != population
        or not pair.get("selected_variant_id") or not pair.get("selected_rsid")
        or not isinstance(value, (int, float)) or isinstance(value, bool)
        or not isfinite(value) or not parameter("ranking.ld_r2_threshold") <= value <= 1):
        return None
    try:
        record = (provider.variation(rsid) if isinstance(provider, LDReferenceProvider) else
                  provider.get(provider.ensembl, f"/variation/human/{rsid}",
                               **{"content-type": "application/json"}))
        if not isinstance(record, Mapping) or record.get("name") != rsid:
            return None
        mappings = [row for row in _reference_objects(record.get("mappings"), "variant mappings")
                    if row.get("assembly_name") == parameter("novelty.reference_build")]
        if len(mappings) != 1:
            return None
        row = mappings[0]
        if (row.get("coord_system") != "chromosome" or row.get("strand") != 1
            or str(row.get("seq_region_name")) != pair["chromosome"]
            or not isinstance(row.get("start"), int) or isinstance(row.get("start"), bool)
            or not isinstance(row.get("end"), int) or isinstance(row.get("end"), bool)
            or row["start"] != pair["position"] or row["end"] != pair.get("end", pair["position"])
            or row["start"] != row["end"] or not isinstance(row.get("allele_string"), str)):
            return None
        alleles = row["allele_string"].split("/")
        if len(alleles) != 2 or len(set(alleles)) != 2 or not set(alleles).issubset({"A", "C", "G", "T"}):
            return None
        return {"variant_id": f"{pair['chromosome']}_{pair['position']}_{alleles[0]}_{alleles[1]}",
                "chromosome": pair["chromosome"], "position": pair["position"], "p_value": None}
    except _REFERENCE_ERRORS:
        return None


def _identity_verification(provider, rsid, method):
    metadata = getattr(provider, "_variations", {}).get(rsid)
    if metadata is not None:
        receipt = metadata.get("parent_receipt", {})
        return {"rsid": rsid, "method": method,
                "identity_response_sha256": receipt.get("sha256"),
                "identity_request_key": receipt.get("request_key"),
                "identity_response_method": receipt.get("method"),
                "identity_response_path": receipt.get("response_path"),
                "identity_response_retrieved_at": receipt.get("retrieved_at"),
                "identity_response_from_cache": receipt.get("from_cache", False)}
    records = [row for row in getattr(provider, "ledger", []) if row.get("status") == "success" and (
        method == "biallelic_mapping_and_population_ld" and row.get("url", "").split("?")[0].endswith(f"/variation/human/{rsid}")
        or method == "explicit_population_alleles" and row.get("method") == "POST"
        and isinstance(row.get("request_body"), Mapping) and rsid in row["request_body"].get("ids", []))]
    return {"rsid": rsid, "method": method,
            "identity_response_sha256": records[-1].get("sha256") if records else None}


def _ldlink_canonical_proxy_variants(provider, proxies, population, *, unresolved=None,
                                     verification=None, progress_sink=None):
    """Orient LDlink's population allele pair against independent GRCh38 REF.

    The unordered LDlink Alleles field supplies the observed biallelic contrast;
    only the reference mapping supplies REF. No population-frequency endpoint is
    needed, and extra dbSNP alleles do not override the measured population pair.
    """
    from .providers.ldlink_identity import matches_population_identity
    import re
    by_id, conflicts = {}, set()
    for pair in proxies:
        rsid = pair["reported_rsid"]
        previous = by_id.get(rsid)
        if previous is not None and any(previous.get(key) != pair.get(key)
                                        for key in ("chromosome", "position", "end")):
            conflicts.add(rsid)
        by_id[rsid] = pair
    converted, metadata, issues = {}, {}, []

    def reject(rsid, reason, message):
        issues.append(f"LD proxy {rsid}: {message}")
        if unresolved is not None:
            unresolved.append({"rsid": rsid, "reason_code": reason, "reason": message})

    for rsid in sorted(by_id):
        pair = by_id[rsid]
        try:
            value = provider.proxy_metadata(rsid, population)
            alleles = value.get("alleles") if isinstance(value, Mapping) else None
            receipts = [row for row in provider.ledger if isinstance(value, Mapping)
                        and row.get("status") == "success"
                        and value.get("request_key") and value.get("response_sha256")
                        and row.get("request_key") == value["request_key"]
                        and row.get("sha256") == value["response_sha256"]]
            if (rsid in conflicts or not isinstance(value, Mapping) or value.get("conflict")
                    or value.get("population") != population
                    or value.get("genome_build") != parameter("novelty.reference_build")
                    or value.get("reference_panel") != parameter("novelty.reference_panel")
                    or value.get("chromosome") != pair["chromosome"]
                    or value.get("position") != pair["position"]
                    or value.get("end", value.get("position")) != pair.get("end", pair["position"])
                    or not isinstance(alleles, (list, tuple)) or len(alleles) != 2
                    or not all(isinstance(allele, str) for allele in alleles)
                    or len(set(alleles)) != 2 or not all(re.fullmatch(r"[ACGT]+|-", allele) for allele in alleles)
                    or (not callable(getattr(provider, "proxy_reference_identities", None))
                        and not set(alleles).issubset({"A", "C", "G", "T"}))
                    or not receipts):
                raise ValueError()
            metadata[rsid] = value
        except _REFERENCE_ERRORS:
            reject(rsid, "ldlink_population_identity_unverified",
                   "The LDlink population allele pair, coordinates or response provenance is not verified.")
    ids = sorted(metadata)
    reference_lookup = getattr(provider, "proxy_reference_identities", None)
    references = reference_lookup(ids, population) if ids and callable(reference_lookup) else None
    if ids and references is None:
        provider.preload_variations(ids, identity_scope="proxy", populations=False)
    emit_progress(progress_sink, "ld_proxy_identity", status="started", total_count=len(by_id),
                  completed_count=len(by_id) - len(ids), validated_count=0)
    for index, rsid in enumerate(ids, start=1):
        if provider.cancel_event is not None and provider.cancel_event.is_set():
            raise CancelledError()
        pair, value = by_id[rsid], metadata[rsid]
        try:
            population_alleles = set(value["alleles"])
            if references is not None:
                result = references.get(rsid, {})
                identity, receipt = result.get("identity", {}), result.get("receipt", {})
                if (result.get("status") != "resolved" or result.get("provider") not in {"opentargets", "ensembl"}
                        or not isinstance(identity, Mapping) or not isinstance(receipt, Mapping)):
                    error = RuntimeError("The proxy reference identity was not resolved.")
                    error.reason_code = result.get("reason_code") or "reference_coordinate_unverified"
                    raise error
                ref, alt = identity.get("reference_allele"), identity.get("alternate_allele")
                binding = result.get("canonical_identity_binding")
                if (not matches_population_identity(identity, value, rsid, canonical_binding=binding)
                        or not receipt.get("request_key") or not receipt.get("sha256")
                        or not any(row.get("status") == "success" and row.get("request_key") == receipt["request_key"]
                                   and row.get("sha256") == receipt["sha256"] for row in provider.ledger)):
                    raise ValueError("The proxy identity and its reference provenance are inconsistent.")
                reference_check = {"rsid": rsid, "method": "reference_identity_and_ldlink_population_alleles",
                    "identity_provider": result["provider"], "identity_response_sha256": receipt["sha256"],
                    "identity_request_key": receipt["request_key"], "identity_response_path": receipt.get("response_path"),
                    "identity_response_retrieved_at": receipt.get("retrieved_at"),
                    "identity_receipts": result.get("receipts", [receipt])}
                if binding is not None:
                    proof_receipts = [row for check in binding["orientation_checks"] for row in check["receipts"]]
                    if not all(any(record.get("status") == "success"
                                   and record.get("reference_provider") == "opentargets"
                                   and record.get("request_key") == row["request_key"]
                                   and record.get("sha256") == row["sha256"] for record in provider.ledger)
                               for row in proof_receipts):
                        raise ValueError("Canonical proxy orientation lacks a successful reference receipt.")
                    reference_check.update(method="canonical_coordinate_and_ldlink_population_alleles",
                                           ldlink_rsid=rsid, reference_rsids=list(identity["rsids"]),
                                           canonical_identity_binding=deepcopy(binding))
            else:
                record = provider.variation(rsid, populations=False)
                if not isinstance(record, Mapping) or record.get("name") != rsid:
                    raise ValueError("The reference mapping does not identify this proxy.")
                mappings = [row for row in _reference_objects(record.get("mappings"), "variant mappings")
                            if row.get("assembly_name") == parameter("novelty.reference_build")
                            and row.get("coord_system") == "chromosome" and row.get("strand") == 1
                            and str(row.get("seq_region_name")) == pair["chromosome"]
                            and type(row.get("start")) is int and type(row.get("end")) is int
                            and row["start"] == row["end"] == pair["position"]]
                if len(mappings) != 1 or not isinstance(mappings[0].get("allele_string"), str):
                    raise ValueError("The GRCh38 reference identity is not unique.")
                reference_alleles = mappings[0]["allele_string"].split("/")
                if (not reference_alleles or reference_alleles[0] not in population_alleles
                        or not population_alleles.issubset(reference_alleles)):
                    raise ValueError("The LDlink allele pair conflicts with the reference mapping.")
                ref = reference_alleles[0]
                alt = next(iter(population_alleles - {ref}))
                reference_check = _identity_verification(provider, rsid, "reference_mapping_and_ldlink_population_alleles")
            reference_position = identity["position"] if references is not None else pair["position"]
            identifier = f"{pair['chromosome']}_{reference_position}_{ref}_{alt}"
            converted[rsid] = [{"variant_id": identifier, "chromosome": pair["chromosome"],
                               "position": reference_position, "p_value": None}]
            if verification is not None:
                verification.append({**reference_check,
                    "variant_id": identifier, "population": population,
                    "population_alleles": sorted(population_alleles),
                    "population_response_sha256": value["response_sha256"],
                    "population_request_key": value["request_key"], "reference_provider": "ldlink"})
        except _REFERENCE_ERRORS as exc:
            reason = reference_failure_reason(exc)
            reject(rsid, reason, "The independent GRCh38 reference identity could not be verified.")
            if reason in RECOVERABLE_LD_REASONS:
                provider.identity_failure_details[(rsid, False)] = {"identity_scope": "proxy", "reason": reason}
        emit_progress(progress_sink, "ld_proxy_identity", status="running", total_count=len(by_id),
                      completed_count=len(by_id) - len(ids) + index, validated_count=len(converted))
    emit_progress(progress_sink, "ld_proxy_identity", status="completed" if not issues else "incomplete",
                  total_count=len(by_id), completed_count=len(by_id), validated_count=len(converted))
    return converted, issues


def canonical_proxy_variants(provider: LDReferenceProvider, proxies, population, *, unresolved=None, verification=None, progress_sink=None):
    """Resolve a batch of reference-population proxies to verified GRCh38 SNP IDs.

    dbSNP identifiers may be multiallelic. The alleles present in the actual LD
    population, rather than every dbSNP alternate, define the proxy contrast.
    Complex alleles needing normalization are reported as unresolved.
    """
    if getattr(provider, "provider_name", None) == "ldlink":
        return _ldlink_canonical_proxy_variants(provider, proxies, population, unresolved=unresolved,
                                              verification=verification, progress_sink=progress_sink)
    by_id, conflicts = {}, set()
    for row in proxies:
        rsid = row["reported_rsid"]
        prior = by_id.get(rsid)
        if prior is not None and any(prior.get(key) != row.get(key) for key in ("chromosome", "position", "end")):
            conflicts.add(rsid)
        by_id[rsid] = row
    for rsid in conflicts:
        by_id.pop(rsid)
    ids = sorted(by_id)
    converted = {}
    batched_identity = isinstance(provider, LDReferenceProvider) and provider.supports_batch_identity
    issues = [f"LD proxy {rsid} has conflicting coordinates across selected SNPs." for rsid in sorted(conflicts)]
    if ids:
        if batched_identity:
            provider.preload_variations(ids, identity_scope="proxy", populations=False)
        completed, last_progress = 0, monotonic()
        emit_progress(progress_sink, "ld_proxy_identity", status="started", completed_count=0,
                      total_count=len(ids), validated_count=0)
        with ThreadPoolExecutor(max_workers=parameter("ranking.ld_identity_concurrency")) as pool:
            pending = {pool.submit(_biallelic_proxy_identity, provider, rsid, by_id[rsid], population): rsid for rsid in ids}
            for future in as_completed(pending):
                rsid = pending[future]
                value = future.result()
                completed += 1
                if getattr(provider, "progress", None) and (completed == len(ids) or monotonic() - last_progress >= parameter("execution.stream_progress_interval_seconds")):
                    provider.progress(f"LD reference: checked proxy identity {completed} of {len(ids)}.")
                    last_progress = monotonic()
                if value is not None:
                    converted[rsid] = [value]
                    if verification is not None:
                        verification.append(_identity_verification(provider, rsid, "biallelic_mapping_and_population_ld"))
                emit_progress(progress_sink, "ld_proxy_identity", status="running", completed_count=completed,
                              total_count=len(ids), validated_count=len(converted))
        emit_progress(progress_sink, "ld_proxy_identity", status="completed", completed_count=completed,
                      total_count=len(ids), validated_count=len(converted))
        ids = [rsid for rsid in ids if rsid not in converted]
    if batched_identity and ids:
        provider.preload_variations(ids, identity_scope="proxy", populations=True)
    batch_size = min(200, parameter("execution.max_variants_per_query"))
    panel = parameter("novelty.reference_panel") + ":" + population
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        emit_progress(progress_sink, "ld_population_identity", status="started",
                      batch_index=start // batch_size + 1, batch_count=(len(ids) + batch_size - 1) // batch_size,
                      completed_count=start, total_count=len(ids), variant_count=len(batch))
        try:
            if batched_identity:
                response = {rsid: provider._variations[rsid]["value"] for rsid in batch
                            if rsid in provider._variations and provider._variations[rsid]["populations_requested"]}
            else:
                response = provider.get(provider.ensembl, "/variation/human", _body={"ids": batch},
                                        pops=1, **{"content-type": "application/json"})
        except _REFERENCE_ERRORS as exc:
            issues.append(f"Proxy identity batch was not completed: {type(exc).__name__}.")
            emit_progress(progress_sink, "ld_population_identity", status="incomplete",
                          batch_index=start // batch_size + 1, batch_count=(len(ids) + batch_size - 1) // batch_size,
                          completed_count=start + len(batch), total_count=len(ids), variant_count=len(batch))
            continue
        if not isinstance(response, Mapping):
            issues.append("Proxy identity batch returned an invalid response.")
            emit_progress(progress_sink, "ld_population_identity", status="incomplete",
                          batch_index=start // batch_size + 1, batch_count=(len(ids) + batch_size - 1) // batch_size,
                          completed_count=start + len(batch), total_count=len(ids), variant_count=len(batch))
            continue
        for rsid in batch:
            record = response.get(rsid)
            if not isinstance(record, Mapping):
                issues.append(f"No reference identity returned for LD proxy {rsid}.")
                continue
            if record.get("name", rsid) != rsid:
                issues.append(f"The identity response for LD proxy {rsid} names a different rs identifier.")
                continue
            pair = by_id[rsid]
            try:
                reference_mappings = _reference_objects(record.get("mappings"), "variant mappings")
                reference_populations = _reference_objects(record.get("populations"), "population alleles")
            except ValueError as exc:
                issues.append(f"LD proxy {rsid} identity was not verified: {exc}")
                continue
            mappings = [row for row in reference_mappings
                        if row.get("assembly_name") == parameter("novelty.reference_build")
                        and row.get("coord_system") == "chromosome" and row.get("strand") == 1
                        and str(row.get("seq_region_name")) == pair["chromosome"]
                        and row.get("start") == pair["position"]
                        and row.get("end") == pair.get("end", pair["position"])]
            population_alleles = {str(row.get("allele")) for row in reference_populations
                if row.get("population") == panel and isinstance(row.get("frequency"), (int, float))
                and not isinstance(row.get("frequency"), bool) and isfinite(row["frequency"])
                and 0 < row["frequency"] <= 1}
            if (len(mappings) != 1 or len(population_alleles) != 2
                or not population_alleles.issubset({"A", "C", "G", "T"})
                or pair.get("end", pair["position"]) != pair["position"]):
                kind = record.get("var_class", "variant")
                kind = kind if isinstance(kind, str) else "variant"
                reason = (
                    "complex_variant_normalization_required" if pair.get("end", pair["position"]) != pair["position"]
                    or not population_alleles.issubset({"A", "C", "G", "T"}) else
                    "reference_coordinate_unverified" if len(mappings) != 1 else
                    "population_alleles_missing" if not population_alleles else
                    "population_alleles_ambiguous"
                )
                issues.append(f"LD proxy {rsid} ({kind}) needs allele or coordinate normalization or additional population metadata ({reason}); it was not assigned an invented SNP ID.")
                if unresolved is not None:
                    unresolved.append({"rsid": rsid, "variant_class": kind,
                                       "reason_code": reason,
                                       "reason": "Allele or coordinate normalization is required."})
                continue
            allele_string = mappings[0].get("allele_string")
            if not isinstance(allele_string, str):
                issues.append(f"No valid reference allele string returned for LD proxy {rsid}.")
                continue
            alleles = allele_string.split("/")
            ref = alleles[0]
            if ref not in population_alleles or not population_alleles.issubset(alleles):
                issues.append(f"Reference and population alleles disagree for LD proxy {rsid}.")
                continue
            converted[rsid] = [{"variant_id": f"{pair['chromosome']}_{pair['position']}_{ref}_{alt}",
                                "chromosome": pair["chromosome"], "position": pair["position"],
                                "p_value": None}
                               for alt in sorted(population_alleles) if alt != ref]
            if verification is not None:
                verification.append(_identity_verification(provider, rsid, "explicit_population_alleles"))
        emit_progress(progress_sink, "ld_population_identity", status="completed" if all(rsid in converted for rsid in batch) else "incomplete",
                      batch_index=start // batch_size + 1, batch_count=(len(ids) + batch_size - 1) // batch_size,
                      completed_count=start + len(batch), total_count=len(ids), variant_count=len(batch),
                      validated_count=sum(rsid in converted for rsid in ids[:start + len(batch)]))
    return converted, list(dict.fromkeys(issues))


def _resolve_ranking_sources(source, provider, progress_sink, cancel_event):
    """Resolve independent source identities, keeping the submitted order intact.

    The production provider already supports concurrent identity GETs. Unknown
    injected backends keep their previous sequential contract.
    """
    cancellation = cancel_event or getattr(provider, "cancel_event", None)
    if cancellation is not None and cancellation.is_set():
        raise CancelledError()
    bulk_results = None

    def resolve_one(index, row):
        if cancellation is not None and cancellation.is_set():
            raise CancelledError()
        if row.get("genome_build", parameter("novelty.reference_build")) != parameter("novelty.reference_build"):
            return index, None, f"Selected SNP {row.get('variant_id')} uses an incompatible genome build."
        try:
            resolved = (provider.resolve(str(row["variant_id"])) if bulk_results is None else
                        bulk_results.get(str(row["variant_id"])))
            if isinstance(resolved, Exception):
                raise resolved
            if (not isinstance(resolved, Mapping) or resolved.get("source_variant_id") != str(row["variant_id"])
                or not isinstance(resolved.get("rsid"), str) or not isinstance(resolved.get("chromosome"), str)
                or not isinstance(resolved.get("position"), int) or isinstance(resolved.get("position"), bool)):
                raise ValueError("Selected SNP resolution did not return a valid reference identity.")
            return index, resolved, None
        except _REFERENCE_ERRORS as exc:
            if isinstance(provider, LDReferenceProvider):
                provider.source_identity_failure_details[str(row["variant_id"])] = {"reason": reference_failure_reason(exc)}
            return index, None, f"Selected SNP identity could not be used for LD expansion: {reference_failure_reason(exc)}."

    resolved_by_index, errors_by_index = {}, {}
    completed = 0
    def retain(item):
        nonlocal completed
        index, resolved, issue = item
        completed += 1
        if resolved is not None:
            resolved_by_index[index] = resolved
        if issue:
            errors_by_index[index] = issue
        emit_progress(progress_sink, "ld_source_identity", status="running", source_count=len(source),
                      total_count=len(source), completed_count=completed, validated_count=len(resolved_by_index))

    if (isinstance(provider, LDReferenceProvider) and source
            and provider.supports_batch_identity
            and "resolve" not in provider.__dict__
            and (type(provider).resolve is LDReferenceProvider.resolve or
                 getattr(provider, "provider_name", None) == "ldlink"
                 and getattr(provider, "supports_source_identity_batch", False) is True)):
        compatible = [row for row in source if row.get("genome_build", parameter("novelty.reference_build")) == parameter("novelty.reference_build")]
        bulk_results = provider.resolve_many([str(row["variant_id"]) for row in compatible])
        for index, row in enumerate(source):
            retain(resolve_one(index, row))
    elif isinstance(provider, LDReferenceProvider) and len(source) > 1:
        with ThreadPoolExecutor(max_workers=min(len(source), parameter("ranking.ld_identity_concurrency"))) as pool:
            pending = [pool.submit(resolve_one, index, row) for index, row in enumerate(source)]
            try:
                for future in as_completed(pending):
                    if cancellation is not None and cancellation.is_set():
                        raise CancelledError()
                    retain(future.result())
            finally:
                # Running HTTP calls retain their existing timeout/cancel checks;
                # queued identities must not start after cancellation.
                for future in pending:
                    future.cancel()
    else:
        for index, row in enumerate(source):
            retain(resolve_one(index, row))
    if cancellation is not None and cancellation.is_set():
        raise CancelledError()
    return ([resolved_by_index[index] for index in sorted(resolved_by_index)],
            [errors_by_index[index] for index in sorted(errors_by_index)])


def _lookup_ranking_proxies(selected, provider, population, threshold, window, source_count, progress_sink, cancel_event):
    """Independent complete per-SNP windows; collect in original source order."""
    cancellation = cancel_event or getattr(provider, "cancel_event", None)
    def one(index, row):
        if cancellation is not None and cancellation.is_set():
            raise CancelledError()
        try:
            return index, provider.proxies(row, population, threshold=threshold, window_bp=window)
        except _REFERENCE_ERRORS as exc:
            return index, exc
    values = {}
    emit_progress(progress_sink, "ld_proxy_lookup", status="started", source_count=source_count,
                  completed_count=0, total_count=len(selected))
    def keep(item):
        index, value = item
        values[index] = value
        emit_progress(progress_sink, "ld_proxy_lookup", status="running", source_count=source_count,
                      completed_count=len(values), total_count=len(selected),
                      records=sum(len(v) for v in values.values() if isinstance(v, list)))
    if isinstance(provider, LDReferenceProvider) and len(selected) > 1:
        with ThreadPoolExecutor(max_workers=parameter("ranking.ld_identity_concurrency")) as pool:
            pending = [pool.submit(one, index, row) for index, row in enumerate(selected)]
            try:
                for future in as_completed(pending):
                    if cancellation is not None and cancellation.is_set():
                        raise CancelledError()
                    keep(future.result())
            finally:
                for future in pending: future.cancel()
    else:
        for index, row in enumerate(selected): keep(one(index, row))
    if cancellation is not None and cancellation.is_set():
        raise CancelledError()
    return [values[index] for index in range(len(selected))]


def _initial_ranking_ld_plan(observations, question, backend=None):
    """Retain source observations before any optional reference access."""
    source = deepcopy([dict(row) for row in observations])
    choice = choose_population(question)
    threshold = parameter("ranking.ld_r2_threshold")
    window = parameter("ranking.ld_window_bp")
    return {"source_observations": source, "analysis_variants": deepcopy(source),
            **ld_provider_metadata(backend),
            "ld_links": [], "complete": False, "issues": [], "population": choice.code,
            "population_assumption": choice.assumption, "threshold": threshold, "window_bp": window,
            "reference_panel": parameter("novelty.reference_panel"),
            "genome_build": parameter("novelty.reference_build"), "query_ledger": [],
            "source_variant_count": len(source), "added_variant_count": 0,
            "required_proxy_count": None, "validated_proxy_count": 0, "unresolved_proxies": [],
            "enumerated_ld_proxies": [], "proxy_identity_checks": []}


def _configuration_limited_ranking_plan(plan, progress_sink=None):
    issue = ld_configuration_issue(scope="ld_expansion")
    emit_progress(progress_sink, "ld_expansion", status="incomplete", reference_provider=plan["reference_provider"],
                  provider_version=plan["provider_version"], reason="ld_credential_unavailable",
                  source_count=plan["source_variant_count"], validated_count=0)
    return {**plan, "status": "incomplete", "issues": [issue["message"]],
            "configuration_issues": [issue], "operation_issues": [issue],
            "ld_lookup_performed": False, "source_identity_lookup_attempted": False,
            "request_summary": {"network_requests": 0}, "technical_gaps": [],
            "recovery_audit": {"attempt_count": 0, "recovery_attempted": False, "status": "configuration_unavailable"},
            "attempts": []}


def _expand_ranking_variants_once(*, observations: Sequence[Mapping[str, Any]], question: str,
                            backend=None, progress=None, progress_sink=None, cancel_event=None, cache_directory=None) -> dict[str, Any]:
    """Return an auditable retrieval plan with original and LD-added variants separate."""
    plan = _initial_ranking_ld_plan(observations, question, backend)
    source = plan["source_observations"]
    choice = choose_population(question)
    threshold, window = plan["threshold"], plan["window_bp"]
    if not parameter("ranking.ld_expansion_enabled"):
        return {**plan, "status": "disabled", "issues": ["LD expansion is disabled in the recorded configuration."]}
    if choice.code is None:
        return {**plan, "status": "incomplete", "issues": [choice.assumption]}
    if not source:
        return {**plan, "status": "incomplete", "issues": ["No selected SNP was supplied for LD expansion."]}
    try:
        provider = backend if backend is not None else create_ld_provider(progress=progress, progress_sink=progress_sink, cancel_event=cancel_event,
                                                 cache_directory=cache_directory,
                                                 total_timeout_seconds=parameter("ranking.ld_total_timeout_seconds"))
    except LDProviderConfigurationError:
        return _configuration_limited_ranking_plan(plan, progress_sink)
    plan.update(ld_provider_metadata(provider))
    selected, proxies, issues = [], [], []
    lookup_failures = []
    proxy_complete = True
    emit_progress(progress_sink, "ld_source_identity", status="started", source_count=len(source),
                  total_count=len(source), completed_count=0, validated_count=0)
    selected, source_issues = _resolve_ranking_sources(source, provider, progress_sink, cancel_event)
    issues.extend(source_issues)
    from .source_normalization import validate_source_normalization

    source_normalizations = [deepcopy(row["source_normalization"]) for row in selected
                             if validate_source_normalization(row.get("source_normalization"))
                             and row["source_normalization"]["original_id"] == row["source_variant_id"]]
    if source_normalizations:
        plan["source_identity_normalizations"] = source_normalizations
    emit_progress(progress_sink, "ld_source_identity", status="completed" if len(selected) == len(source) else "incomplete", source_count=len(source),
                  total_count=len(source), completed_count=len(source), validated_count=len(selected))
    if not selected:
        # No LD query ran: a failed prerequisite is not a successful empty
        # population query. Keep the receipts/gaps for the existing recovery.
        for phase in ("ld_proxy_lookup", "ld_proxy_identity"):
            emit_progress(progress_sink, phase, status="not_run", reason="source_identity_unresolved",
                          source_count=len(source), total_count=0, completed_count=0)
        emit_progress(progress_sink, "ld_expansion", status="incomplete", reason="source_identity_unresolved",
                      source_count=len(source), validated_count=0, total_analysis_variants=len(source))
        return {**plan, "status": "incomplete", "issues": list(dict.fromkeys(issues)),
                "query_ledger": list(getattr(provider, "ledger", [])),
                "request_summary": dict(getattr(provider, "request_summary", {})),
                "ld_lookup_performed": False, "source_identity_lookup_attempted": True,
                "ld_lookup_failures": [], "ld_lookup_audit": [],
                "technical_gaps": _technical_ld_gaps(provider, [])}
    selected_rsids = {row["rsid"] for row in selected}
    returned_by_source = _lookup_ranking_proxies(selected, provider, choice.code, threshold, window,
                                                len(source), progress_sink, cancel_event)
    for selected_index, row in enumerate(selected, start=1):
        try:
            returned = returned_by_source[selected_index - 1]
            if isinstance(returned, Exception):
                raise returned
            if not isinstance(returned, list):
                raise ValueError("LD proxies did not return an array.")
            lookup_audit = getattr(provider, "proxy_lookup_issues", {}).get(
                (row["source_variant_id"], choice.code), {})
            if lookup_audit.get("requested_scope_complete") is False:
                proxy_complete = False
                if lookup_audit.get("window_coverage_reason") == "grch37_window_not_proven_complete_in_grch38":
                    issues.append("LD proxies were recovered through GRCh37 and verified in GRCh38; "
                                  "complete coverage of the GRCh38 window is not established.")
                else:
                    issues.append("Some in-scope LD records could not be verified; the valid records were retained.")
            for pair in returned:
                if not isinstance(pair, Mapping):
                    proxy_complete = False
                    issues.append("An LD proxy was not a structured reference record.")
                    continue
                value = pair.get("r2")
                if (pair.get("status") != "computed" or not isinstance(value, (int, float))
                    or isinstance(value, bool) or not isfinite(value) or not 0 <= value <= 1
                    or pair.get("population") != choice.code
                    or pair.get("selected_variant_id") != row["source_variant_id"]):
                    proxy_complete = False
                    issues.append("An LD proxy had invalid source, population or numeric provenance.")
                    continue
                if (not isinstance(pair.get("reported_rsid"), str) or not isinstance(pair.get("chromosome"), str)
                    or not isinstance(pair.get("position"), int) or isinstance(pair.get("position"), bool)
                    or not isinstance(pair.get("end", pair["position"]), int) or isinstance(pair.get("end"), bool)):
                    proxy_complete = False
                    issues.append("An LD proxy had an invalid reference identifier or coordinate.")
                    continue
                if value < threshold or pair["reported_rsid"] in selected_rsids:
                    continue
                if pair["chromosome"] != row["chromosome"] or abs(pair["position"] - row["position"]) > window:
                    continue
                receipts = [item for item in getattr(provider, "ledger", [])
                            if item.get("status") == "success" and (
                                pair.get("ld_request_key") and item.get("request_key") == pair["ld_request_key"]
                                and item.get("sha256") == pair.get("ld_response_sha256")
                                or not pair.get("ld_request_key") and f"/ld/human/{row['rsid']}/" in item.get("url", ""))]
                if (plan["reference_provider"] == "ldlink" or pair.get("ld_request_key")) and not receipts:
                    proxy_complete = False
                    issues.append("An LD result did not match its successful reference-response receipt.")
                    continue
                proxies.append({**pair, "ld_response_sha256": receipts[-1].get("sha256") if receipts else None})
        except _REFERENCE_ERRORS as exc:
            proxy_complete = False
            issues.append(f"LD expansion is incomplete: {_reference_failure_reason(exc)}.")
            lookup_failures.append({"source_variant_id": row["source_variant_id"], "rsid": row["rsid"],
                                    "reason": _reference_failure_reason(exc), "error_type": type(exc).__name__})
    emit_progress(progress_sink, "ld_proxy_lookup", status="completed" if proxy_complete else "incomplete",
                  source_count=len(source), completed_count=len(selected), total_count=len(selected), records=len(proxies))
    required = len({pair["reported_rsid"] for pair in proxies})
    if len(proxies) > parameter("novelty.max_ld_pairs"):
        proxies = proxies[:parameter("novelty.max_ld_pairs")]
        issues.append("The recorded LD proxy budget was reached; expansion is incomplete.")
        proxy_complete = False
    plan["enumerated_ld_proxies"] = [dict(pair) for pair in proxies]
    if isinstance(provider, LDReferenceProvider):
        provider.progress_context = {}
    converted, identity_issues = canonical_proxy_variants(provider, proxies, choice.code,
        unresolved=plan["unresolved_proxies"], verification=plan["proxy_identity_checks"], progress_sink=progress_sink)
    issues.extend(identity_issues)
    identifiers = {str(row.get("variant_id")) for row in source}
    source_identities = identifiers | {proof["canonical_id"] for proof in source_normalizations}
    source_alias_proxies = {rsid for rsid, variants in converted.items()
                           if variants and all(row["variant_id"] in source_identities for row in variants)}
    if source_alias_proxies:
        # A different rs alias of a selected source is not an additional SNP.
        # Preserve its verification separately without creating a P-less copy.
        plan["excluded_source_alias_proxies"] = [dict(pair) for pair in proxies
                                                 if pair["reported_rsid"] in source_alias_proxies]
        proxies = [pair for pair in proxies if pair["reported_rsid"] not in source_alias_proxies]
        converted = {key: value for key, value in converted.items() if key not in source_alias_proxies}
        required -= len(source_alias_proxies)
        plan["enumerated_ld_proxies"] = [dict(pair) for pair in proxies]
    additions = []
    for pair in proxies:
        for variant in converted.get(pair["reported_rsid"], []):
            identifier = variant["variant_id"]
            if identifier not in identifiers:
                identifiers.add(identifier)
                additions.append(variant)
            coordinate = {"position": variant["position"], "end": variant["position"]}
            if variant["position"] != pair["position"]:
                coordinate.update(ld_reported_position=pair["position"], ld_reported_end=pair.get("end", pair["position"]))
            plan["ld_links"].append({**pair, **coordinate, "proxy_variant_id": identifier, "genome_build": plan["genome_build"],
                                     "evidence_relation": "via_population_ld", "source_gwas_p_copied": False})
    complete = not issues and len(selected) == len(source) and len(converted) == required
    emit_progress(progress_sink, "ld_expansion", status="completed" if complete else "incomplete",
                  source_count=len(source), total_count=required,
                  completed_count=len({pair["reported_rsid"] for pair in proxies}),
                  validated_count=len(converted), total_analysis_variants=len(source) + len(additions))
    return {**plan, "analysis_variants": [*source, *additions], "complete": complete,
            "status": "complete" if complete else "incomplete", "issues": list(dict.fromkeys(issues)),
            "ld_lookup_performed": True, "source_identity_lookup_attempted": True,
            "query_ledger": list(getattr(provider, "ledger", [])), "added_variant_count": len(additions),
            "request_summary": dict(getattr(provider, "request_summary", {})),
            "ld_lookup_failures": lookup_failures,
            "ld_lookup_audit": [{"source_variant_id": row["source_variant_id"], "population": choice.code,
                                 **deepcopy(getattr(provider, "proxy_lookup_issues", {}).get(
                                     (row["source_variant_id"], choice.code), {}))}
                                for row in selected
                                if (row["source_variant_id"], choice.code) in getattr(provider, "proxy_lookup_issues", {})],
            "required_proxy_count": required if proxy_complete and len(selected) == len(source) else None,
            "validated_proxy_count": len(converted),
            "technical_gaps": _technical_ld_gaps(provider, lookup_failures)}


def _technical_ld_gaps(provider, lookup_failures):
    gaps = {}
    for identifier, value in getattr(provider, "source_identity_failure_details", {}).items():
        if value.get("reason") in RECOVERABLE_LD_REASONS:
            gaps[("source_identity", identifier)] = {"phase": "source_identity", "entity_id": identifier, "reason": value["reason"]}
    for value in lookup_failures:
        if value.get("reason") in RECOVERABLE_LD_REASONS:
            gaps[("ld_window", value["source_variant_id"])] = {"phase": "ld_window", "entity_id": value["source_variant_id"], "reason": value["reason"]}
    for (rsid, populations), value in getattr(provider, "identity_failure_details", {}).items():
        entry = getattr(provider, "_variations", {}).get(rsid)
        available = entry is not None and (not populations or entry["populations_requested"])
        if value.get("identity_scope") == "proxy" and not available and value.get("reason") in RECOVERABLE_LD_REASONS:
            gaps[("proxy_identity", rsid)] = {"phase": "proxy_identity", "entity_id": rsid, "reason": value["reason"]}
    for (source_id, _population), audit in getattr(provider, "proxy_lookup_issues", {}).items():
        for row in audit.get("excluded_rows", []):
            reason, rsid = row.get("reason_code"), row.get("reported_rsid")
            if row.get("scope_relevant") is not True or reason not in RECOVERABLE_LD_REASONS:
                continue
            if rsid and rsid in getattr(provider, "_variations", {}):
                continue
            phase, entity = ("proxy_identity", rsid) if rsid else ("ld_window", source_id)
            gaps.setdefault((phase, entity), {"phase": phase, "entity_id": entity, "reason": reason})
    return [gaps[key] for key in sorted(gaps)]


def expand_ranking_variants(*, observations: Sequence[Mapping[str, Any]], question: str,
                            backend=None, progress=None, progress_sink=None, cancel_event=None, cache_directory=None) -> dict[str, Any]:
    from .ld_recovery import recover_ranking_ld
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError()
    arguments = {"observations": observations, "question": question, "progress": progress,
                 "progress_sink": progress_sink, "cancel_event": cancel_event, "cache_directory": cache_directory}
    if backend is None and (not parameter("ranking.ld_expansion_enabled") or not observations
                            or choose_population(question).code is None):
        first = _expand_ranking_variants_once(**arguments)
        return recover_ranking_ld(first, None, _expand_ranking_variants_once, create_ld_provider, arguments)
    try:
        provider = backend if backend is not None else create_ld_provider(progress=progress, progress_sink=progress_sink, cancel_event=cancel_event,
            cache_directory=cache_directory, total_timeout_seconds=parameter("ranking.ld_total_timeout_seconds"))
    except LDProviderConfigurationError:
        return _configuration_limited_ranking_plan(_initial_ranking_ld_plan(observations, question), progress_sink)
    first = _expand_ranking_variants_once(backend=provider, **arguments)
    if backend is not None:
        return first
    factory = partial(create_ld_provider, provider_name=ld_provider_metadata(provider)["reference_provider"])
    return recover_ranking_ld(first, provider, _expand_ranking_variants_once, factory, arguments)
