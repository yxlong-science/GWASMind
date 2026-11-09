"""One bounded LD recovery pass, retaining successful scientific provenance."""
from concurrent.futures import CancelledError
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .audit import sha256_json
from .configuration import parameter
from .progress import emit_progress
from .providers.ld import RECOVERABLE_LD_REASONS, reference_failure_reason
from .providers.ld_factory import ld_provider_metadata

_SCOPE_FIELDS = ("source_observations", "source_variant_count", "population", "threshold", "window_bp", "reference_panel", "genome_build", "reference_provider", "provider_version")
_LINK_FIELDS = ("selected_variant_id", "selected_rsid", "reported_rsid", "population", "r2", "status",
                "chromosome", "position", "end", "proxy_variant_id", "genome_build", "evidence_relation", "source_gwas_p_copied")


def _prime_successes(previous, current):
    if (ld_provider_metadata(previous) != ld_provider_metadata(current)
            or previous.ensembl != current.ensembl or previous.catalog != current.catalog
            or previous.response_cache.directory != current.response_cache.directory):
        raise ValueError("LD recovery cache scope changed")
    # A successfully received LDlink response can contain unresolved identities
    # or rows without rsIDs. Reuse the immutable response during this recovery;
    # its incomplete scientific scope is still re-derived by the parser. A new
    # connection failure must not discard LD measurements already received.
    reusable_keys = {key for key, receipt in previous._receipts.items()
                     if receipt.get("status") == "success" and (
                         receipt.get("requested_scope_complete") is not False
                         or receipt.get("reference_provider") == "ldlink")}
    current._cache = {key: deepcopy(value) for key, value in previous._cache.items() if key in reusable_keys}
    current._receipts = {key: deepcopy(value) for key, value in previous._receipts.items() if key in reusable_keys}
    current._variations = deepcopy(previous._variations)
    current._source_candidate_cache = deepcopy(previous._source_candidate_cache)
    # Technical failures may recover. Ordinary invalid/missing reference
    # identities retain their previous outcome without additional HTTP.
    current._failed_requests = {key: value for key, value in previous._failed_requests.items()
                                if reference_failure_reason(value) not in RECOVERABLE_LD_REASONS}
    for key, value in previous.identity_failure_details.items():
        if value.get("reason") not in RECOVERABLE_LD_REASONS:
            current.identity_failure_details[key] = deepcopy(value)
            if key in previous._variation_failures:
                current._variation_failures[key] = previous._variation_failures[key]
    parents = set()
    for value in current._variations.values():
        receipt = value.get("parent_receipt", {})
        receipt.update(from_cache=True, observed_at=datetime.now(timezone.utc).isoformat())
        if receipt.get("request_key") and receipt["request_key"] not in parents:
            current.ledger.append(deepcopy(receipt)); parents.add(receipt["request_key"])
        current.request_summary["cache_hits"] += 1


def _link_key(value):
    return value.get("selected_variant_id"), value.get("reported_rsid"), value.get("population")


def _scientific_link(value):
    return tuple(value.get(field) for field in _LINK_FIELDS)


def merge_ld_attempts(previous, current):
    """Keep prior positives; contradictory new links cannot silently replace them."""
    if any(previous.get(key) != current.get(key) for key in _SCOPE_FIELDS):
        return deepcopy(previous), [{"reason": "scope_mismatch"}]
    merged = deepcopy(current)
    for flag in ("ld_lookup_performed", "source_identity_lookup_attempted"):
        if flag in previous or flag in current:
            merged[flag] = any(plan.get(flag) is True for plan in (previous, current))
    links = {_link_key(row): deepcopy(row) for row in previous.get("ld_links", [])}
    conflicts = []
    for row in current.get("ld_links", []):
        key = _link_key(row)
        if key in links and _scientific_link(links[key]) != _scientific_link(row):
            conflicts.append({"reason": "conflicting_ld_link", "source_variant_id": key[0], "rsid": key[1]})
        else:
            links.setdefault(key, deepcopy(row))
    merged["ld_links"] = list(links.values())
    source = deepcopy(previous["source_observations"])
    variants = {row["variant_id"]: row for row in source}
    supported = {row.get("proxy_variant_id") for row in merged["ld_links"]}
    for plan in (previous, current):
        for row in plan.get("analysis_variants", [])[len(source):]:
            identifier = row.get("variant_id")
            if identifier not in supported: continue
            if row.get("p_value") is not None or row.get("beta") is not None:
                conflicts.append({"reason": "invalid_proxy_statistics", "variant_id": identifier})
                continue
            if identifier in variants and variants[identifier] != row:
                conflicts.append({"reason": "conflicting_proxy_identity", "variant_id": identifier})
                continue
            variants.setdefault(identifier, deepcopy(row))
    merged["source_observations"] = source
    merged["analysis_variants"] = list(variants.values())
    merged["ld_links"] = [row for row in merged["ld_links"] if row.get("proxy_variant_id") in variants]
    valid_rsids = {row["reported_rsid"] for row in merged["ld_links"]}
    checks = {}
    for plan in (previous, current):
        for row in plan.get("proxy_identity_checks", []):
            if row["rsid"] in valid_rsids: checks.setdefault(row["rsid"], deepcopy(row))
    merged["proxy_identity_checks"] = list(checks.values())
    merged["validated_proxy_count"] = len(valid_rsids)
    merged["added_variant_count"] = len(variants) - len(source)
    enumerated = {}
    for plan in (previous, current):
        for row in plan.get("enumerated_ld_proxies", []):
            enumerated.setdefault(_link_key(row), deepcopy(row))
    merged["enumerated_ld_proxies"] = list(enumerated.values())
    if merged.get("required_proxy_count") is not None:
        merged["required_proxy_count"] = max(merged["required_proxy_count"], len(valid_rsids),
                                             len({row.get("reported_rsid") for row in enumerated.values()}))
    merged["unresolved_proxies"] = [row for row in merged.get("unresolved_proxies", []) if row.get("rsid") not in valid_rsids]
    merged["issues"] = [issue for issue in merged.get("issues", [])
                        if not any(issue == f"No reference identity returned for LD proxy {rsid}." for rsid in valid_rsids)]
    merged["query_ledger"] = [*deepcopy(previous.get("query_ledger", [])), *deepcopy(current.get("query_ledger", []))]
    from .source_normalization import validate_source_normalization
    normalizations = {}
    for plan in (previous, current):
        for proof in plan.get("source_identity_normalizations", ()):
            if validate_source_normalization(proof):
                normalizations.setdefault(proof["original_id"], deepcopy(proof))
    if normalizations:
        merged["source_identity_normalizations"] = list(normalizations.values())
        raw_source_ids = {row["variant_id"] for row in source}
        canonical_source_ids = raw_source_ids | {proof["canonical_id"] for proof in normalizations.values()}
        excluded = [row for row in merged["ld_links"] if row["proxy_variant_id"] in canonical_source_ids]
        if excluded:
            # A later incomplete source lookup must not turn an identity already
            # verified as a source into a new, P-less proxy observation.
            merged["analysis_variants"] = [row for row in merged["analysis_variants"]
                if row["variant_id"] in raw_source_ids or row["variant_id"] not in canonical_source_ids]
            merged["ld_links"] = [row for row in merged["ld_links"] if row not in excluded]
            remaining_rsids = {row["reported_rsid"] for row in merged["ld_links"]}
            excluded_rsids = {row["reported_rsid"] for row in excluded} - remaining_rsids
            merged["enumerated_ld_proxies"] = [row for row in merged["enumerated_ld_proxies"]
                                                if row["reported_rsid"] not in excluded_rsids]
            merged["proxy_identity_checks"] = [row for row in merged["proxy_identity_checks"]
                                                if row["rsid"] in remaining_rsids]
            merged["unresolved_proxies"] = [row for row in merged["unresolved_proxies"]
                                            if row.get("rsid") not in excluded_rsids]
            merged["validated_proxy_count"] = len(remaining_rsids)
            merged["added_variant_count"] = len(merged["analysis_variants"]) - len(source)
            if merged.get("required_proxy_count") is not None:
                merged["required_proxy_count"] = max(len(remaining_rsids),
                    merged["required_proxy_count"] - len(excluded_rsids))
            excluded_audit = {}
            for row in [*previous.get("excluded_source_alias_proxies", []),
                        *current.get("excluded_source_alias_proxies", []), *excluded]:
                excluded_audit.setdefault(_link_key(row), deepcopy(row))
            merged["excluded_source_alias_proxies"] = list(excluded_audit.values())
    merged["request_summary"] = {key: previous.get("request_summary", {}).get(key, 0) + current.get("request_summary", {}).get(key, 0)
        for key in set(previous.get("request_summary", {})) | set(current.get("request_summary", {}))}
    if conflicts:
        merged["complete"] = False
        merged["issues"].append("LD recovery returned conflicting evidence; previous verified evidence was retained and coverage remains incomplete.")
    if merged.get("complete") and merged.get("required_proxy_count") != merged["validated_proxy_count"]:
        merged["complete"] = False
        merged["issues"].append("Some previously observed LD candidates remain unverified after recovery.")
    merged["status"] = "complete" if merged.get("complete") else "incomplete"
    return merged, conflicts


def recover_ranking_ld(first, provider, run_once, provider_factory, arguments):
    gaps = first.get("technical_gaps", [])
    cache = getattr(provider, "response_cache", None)
    positive = cache is not None and any(row.get("status") == "success" and row.get("response_path")
        and Path(row["response_path"]).is_file() for row in first.get("query_ledger", []))
    eligible = (bool(gaps) and parameter("ranking.ld_recovery_attempts") > 0
                and arguments.get("cache_directory") is not None and positive)
    if not eligible:
        return {**first, "recovery_audit": {"attempt_count": 1, "recovery_attempted": False,
            "status": "not_needed" if not gaps else "not_available"}, "attempts": []}
    cancellation = arguments.get("cancel_event")
    if cancellation is not None and cancellation.is_set(): raise CancelledError()
    emit_progress(arguments.get("progress_sink"), "ld_recovery", status="started",
        reason="transport_or_deadline_gap", attempt=2, completed_count=0, total_count=len(gaps),
        verified_proxy_count=first.get("validated_proxy_count", 0), previous_validated_count=first.get("validated_proxy_count", 0),
        newly_validated_count=0)
    recovery = None
    try:
        recovery = provider_factory(progress=arguments.get("progress"), progress_sink=arguments.get("progress_sink"),
            cancel_event=cancellation, cache_directory=arguments["cache_directory"],
            total_timeout_seconds=parameter("ranking.ld_total_timeout_seconds"))
        _prime_successes(provider, recovery)
        second = run_once(backend=recovery, **arguments)
    except (RuntimeError, ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
        second = {key: deepcopy(first.get(key)) for key in _SCOPE_FIELDS}
        second.update(complete=False, status="incomplete", analysis_variants=deepcopy(first["source_observations"]),
            added_variant_count=0, validated_proxy_count=0, required_proxy_count=None, ld_links=[], proxy_identity_checks=[],
            enumerated_ld_proxies=[], unresolved_proxies=[], technical_gaps=deepcopy(gaps),
            issues=["LD recovery did not complete; previously verified evidence was retained."],
            query_ledger=deepcopy(getattr(recovery, "ledger", [])), request_summary=dict(getattr(recovery, "request_summary", {})),
            recovery_failure={"error_type": type(exc).__name__, "reason": reference_failure_reason(exc)})
    if cancellation is not None and cancellation.is_set(): raise CancelledError()
    merged, conflicts = merge_ld_attempts(first, second)
    remaining = {(row["phase"], row["entity_id"]) for row in merged.get("technical_gaps", [])}
    completed = sum((row["phase"], row["entity_id"]) not in remaining for row in gaps)
    status = "completed" if not remaining and not conflicts else "incomplete"
    merged["attempts"] = [deepcopy(first), deepcopy(second)]
    merged["recovery_audit"] = {"attempt_count": 2, "recovery_attempted": True, "status": status,
        "initial_technical_gaps": deepcopy(gaps), "remaining_technical_gaps": deepcopy(merged.get("technical_gaps", [])),
        "previous_validated_count": first.get("validated_proxy_count", 0),
        "verified_proxy_count": merged.get("validated_proxy_count", 0),
        "newly_validated_count": max(0, merged.get("validated_proxy_count", 0) - first.get("validated_proxy_count", 0)),
        "conflicts": conflicts, "original_ledger_sha256": sha256_json(first.get("query_ledger", [])),
        "recovery_ledger_sha256": sha256_json(second.get("query_ledger", []))}
    emit_progress(arguments.get("progress_sink"), "ld_recovery", status=status, attempt=2,
        completed_count=completed, total_count=len(gaps), verified_proxy_count=merged.get("validated_proxy_count", 0),
        previous_validated_count=first.get("validated_proxy_count", 0),
        newly_validated_count=merged["recovery_audit"]["newly_validated_count"])
    return merged
