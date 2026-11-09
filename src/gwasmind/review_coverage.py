"""Pure review facts for final retrieval scopes; never retrieves or ranks evidence.

Final scope receipts take precedence over attempt logs. Success with no records
does not establish biological absence. These helpers do not execute Gate1.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_COMPLETE = {"success", "completed", "complete", "no_data", "success_data", "success_no_data"}
_NOT_APPLICABLE = {"not_applicable", "not_requested", "disabled"}
_LITERATURE = {"evidence.europepmc", "target.literature"}
_CAP = re.compile(
    r"Open Targets declared ([0-9]+) literatureOcurrences rows for [A-Za-z0-9_.:-]+; "
    r"the bounded response retained ([0-9]+)\. Coverage is incomplete; "
    r"absent rows are not evidence of absence\.")
_CAP_NOTICE = (
    "Open Targets target-literature lists were limited to the configured number of records "
    "per gene. Additional publications may be absent from this evidence view."
)
_EXPRESSION_ASSESSMENT_NOTE = (
    "Expression data retrieval is complete for the observed pool, but some tissue/cell-type "
    "comparisons remain unresolved; the retained first-tier set is provisional. See candidate_audit."
)
_REGIONAL_NOTE = re.compile(
    r"Exact selected-variant transcript annotations establish at least [0-9]+ gene\(s\) "
    r"within [0-9]+(?:\.[0-9]+)? Mb\. This is a lower bound, not a complete regional gene "
    r"census; proximity is not direct causal-gene evidence\.")
_CANDIDATE_CONTEXT_NOTE = re.compile(
    r"Candidate scope retained [0-9]+ gene\(s\) as unresolved context; missing or conflicting "
    r"locus coordinates do not establish trans status or absence of biological involvement\. "
    r"See the candidate audit\.")
_CANDIDATE_COMPONENTS = (
    "locus_discovery_complete", "colocalisation_detail_complete", "expression_retrieval_complete",
    "evidence_integrity_complete", "l2g_grouping_complete",
)


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> tuple:
    return tuple(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _ids(values: Any) -> list[str]:
    missing = {".", "-", "na", "n/a", "none", "null", "nan", "unknown"}
    return sorted({v.strip() for v in _rows(values) if isinstance(v, str) and v.strip()
                   and len(v) <= 256 and v.strip().casefold() not in missing})


def _state(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("final_status", value.get("status", value.get("state")))
    return value.strip().lower() if isinstance(value, str) else "unknown"


def _bounded_literature(warnings: Any, *, expression_retrieval_complete: bool = False) -> bool:
    values = _rows(warnings)
    if not values:
        return False
    cap_observed = False
    for value in values:
        # The current stage bridge attaches an entire tool's warning list to
        # each requested literature module. Remove only exact known messages
        # about other evidence scopes, never unknown errors or missing papers.
        if isinstance(value, str) and (_REGIONAL_NOTE.fullmatch(value) or _CANDIDATE_CONTEXT_NOTE.fullmatch(value)):
            continue
        if expression_retrieval_complete and value == _EXPRESSION_ASSESSMENT_NOTE:
            continue
        if value == _CAP_NOTICE:
            cap_observed = True
            continue
        match = _CAP.fullmatch(value) if isinstance(value, str) else None
        if not match or int(match[1]) <= int(match[2]):
            return False
        cap_observed = True
    return cap_observed


def _active_gap(value: Any) -> bool:
    """Only explicit scope/recovery labels dismiss a recorded unresolved item."""
    return (isinstance(value, Mapping) and value.get("scope_relevant") is not False
            and value.get("resolved") is not True and value.get("recovered") is not True
            and _state(value) not in {"resolved", "recovered"})


def assess_frozen_molqtl_coloc_coverage(value: Any) -> dict[str, Any] | None:
    """Validate completed coloc queries against their frozen molecular anchors.

    Returned row counts precede phenotype/H4 filtering. Neither a zero result
    nor unrelated variant/LD acquisition gaps make a completed query partial.
    Legacy variant-anchored receipts keep their original coverage semantics.
    """
    receipt = _mapping(value)
    if receipt.get("anchor_kind") != "frozen_molqtl_credible_set":
        return None
    reasons = []
    raw_ids = receipt.get("frozen_molqtl_ids", receipt.get("frozen_ids"))
    ids = _ids(raw_ids)
    ids_valid = (isinstance(raw_ids, (list, tuple)) and len(raw_ids) == len(ids)
                 and all(isinstance(item, str) and item in ids for item in raw_ids))
    if "frozen_molqtl_ids" in receipt and "frozen_ids" in receipt:
        ids_valid = ids_valid and receipt["frozen_molqtl_ids"] == receipt["frozen_ids"]
    if not ids_valid:
        reasons.append("frozen_molqtl_scope_unverified")
    anchors = _mapping(receipt.get("per_anchor"))
    if not isinstance(receipt.get("per_anchor"), Mapping):
        reasons.append("frozen_molqtl_anchor_receipts_unverified")
    missing = sorted(set(ids) - set(anchors))
    unexpected = sorted(str(key) for key in set(anchors) - set(ids))
    if missing:
        reasons.append("frozen_molqtl_anchor_missing")
    if unexpected:
        reasons.append("unfrozen_molqtl_anchor_present")
    completed, queried, failed, partial, incomplete = [], [], [], [], []
    for identifier in ids:
        anchor = _mapping(anchors.get(identifier))
        state = _state(anchor)
        if state in {"complete", "partial", "failed"}:
            queried.append(identifier)
        if state in {"failed", "partial"}:
            failed.append(identifier)
        if state == "partial":
            partial.append(identifier)
        returned = anchor.get("returned_rows")
        expected = anchor.get("expected_rows", anchor.get("count"))
        counts_valid = (type(returned) is int and type(expected) is int
                        and returned >= 0 and expected >= 0 and returned == expected)
        if "expected_rows" in anchor and "count" in anchor:
            counts_valid = counts_valid and anchor["expected_rows"] == anchor["count"]
        if state == "complete" and anchor.get("complete") is True and counts_valid:
            completed.append(identifier)
        else:
            incomplete.append(identifier)
    if incomplete:
        reasons.append("frozen_molqtl_queries_incomplete")
    counts = {"frozen_molqtl_count": len(ids), "queried_molqtl_count": len(queried),
              "completed_molqtl_count": len(completed), "failed_molqtl_count": len(failed)}
    if any(type(receipt.get(key)) is not int or receipt[key] != count for key, count in counts.items()):
        reasons.append("frozen_molqtl_counts_inconsistent")
    if receipt.get("complete") is not True:
        reasons.append("frozen_molqtl_receipt_incomplete")
    complete = not reasons
    status = "complete" if complete else "failed" if ids and len(failed) == len(ids) and not partial else "partial"
    fact = (f"Colocalization queries completed for {len(completed)}/{len(ids)} frozen molecular-QTL credible sets. "
            + ("Successful zero-result queries and results below the retention threshold are complete searches."
               if complete else "Failed, missing or incompletely paginated anchor queries remain unresolved."))
    if not ids_valid:
        fact = "Colocalization query coverage could not be verified against a valid frozen molecular-QTL anchor list."
    return {"anchor_kind": "frozen_molqtl_credible_set", "state": "pass" if complete else "warning",
            "complete": complete, "module_status": status, **counts, "reason_codes": reasons,
            "missing_anchor_ids": missing, "failed_anchor_ids": failed, "partial_anchor_ids": partial,
            "incomplete_anchor_ids": incomplete,
            "unexpected_anchor_ids": unexpected, "fact": fact,
            "scope_note": "Coverage applies to the frozen molecular-QTL anchors; upstream molecular-QTL acquisition is assessed separately."}


def assess_candidate_coloc_coverage(audit: Mapping) -> dict[str, Any] | None:
    """Read the native final scope receipt or its gateway progress copy."""
    nested = _mapping(_mapping(_mapping(audit.get("v2g")).get("coverage")).get("colocalisation"))
    return assess_frozen_molqtl_coloc_coverage(nested or audit.get("colocalisation_coverage"))


def assess_retrieval_coverage(
    *, module_statuses: Mapping, required_modules: Sequence,
    module_warnings: Mapping | None = None, candidate_audit: Mapping | None = None,
    require_exhaustive_literature: bool = False,
) -> dict[str, Any]:
    """Describe unresolved declared scopes, separating LD/context root causes.

    Status values can be final state strings or receipts with ``final_status``.
    A recovered module is listed only when its final receipt explicitly records
    ``recovered=True``; error text or record counts never establish recovery.
    """
    statuses, warnings, audit = map(_mapping, (module_statuses, module_warnings, candidate_audit))
    statuses, audit = dict(statuses), dict(audit)
    coloc = assess_candidate_coloc_coverage(audit)
    if coloc is not None:
        statuses["locus.colocalisation"] = coloc["module_status"]
        # This scope has its own immutable anchor universe. Legacy variant
        # limits and unrelated LD gaps cannot overturn its final query receipt.
        audit["colocalisation_detail_complete"] = coloc["complete"]
    required = set(_ids(required_modules)) - {"candidate_ranking"}
    incomplete_required, incomplete_optional, bounded, recovered = [], [], [], []
    for module in sorted(required | {key for key in statuses if isinstance(key, str)}):
        if module == "candidate_ranking":
            continue
        receipt = statuses.get(module)
        state = _state(receipt)
        if state in _COMPLETE:
            if _mapping(receipt).get("recovered") is True:
                recovered.append(module)
            continue
        if state in _NOT_APPLICABLE:
            continue
        if (state == "truncated" and module in _LITERATURE
                and _bounded_literature(warnings.get(module),
                    expression_retrieval_complete=audit.get("expression_retrieval_complete") is True)):
            bounded.append(module)
            if not require_exhaustive_literature:
                continue
        (incomplete_required if module in required else incomplete_optional).append(module)

    candidate_gaps = [key for key in _CANDIDATE_COMPONENTS if audit.get(key) is False
                      and not (coloc is not None and key == "colocalisation_detail_complete")]
    deferred = [key for key in ("ld_expansion_complete", "ld_proxy_retrieval_complete", "molecular_context_complete")
                if audit.get(key) is False]
    if (audit.get("expression_assessment_complete") is False
            and audit.get("expression_retrieval_complete") is True):
        deferred.append("expression_comparison_unresolved")
    unexplained = (audit.get("candidate_discovery_complete") is False
                   and not candidate_gaps and not deferred)
    reasons = []
    facts = []
    if incomplete_required:
        reasons.append("required_retrieval_scope_incomplete")
        facts.append("Required retrieval scope remains incomplete or unverified: " + ", ".join(incomplete_required) + ".")
    if incomplete_optional:
        reasons.append("optional_retrieval_scope_incomplete")
        facts.append("Additional retrieval scope remains incomplete or unverified: " + ", ".join(incomplete_optional) + ".")
    if candidate_gaps:
        reasons.append("candidate_comparison_scope_incomplete")
        facts.append("Candidate discovery or comparison has an unresolved scope component: " + ", ".join(candidate_gaps) + ".")
    return {"incomplete_required": incomplete_required, "incomplete_optional": incomplete_optional,
            "bounded_literature_modules": bounded, "recovered_modules": recovered,
            "candidate_gap_reasons": candidate_gaps, "deferred_candidate_causes": deferred,
            "unattributed_candidate_gap": unexplained, "facts": facts, "reason_codes": reasons,
            "colocalisation_coverage": coloc}


def _missing_proxy_rsid(row: Mapping) -> bool:
    """Exclude explicitly unaddressable proxies from Gate coverage only."""
    return (row.get("reason_code") == "proxy_identifier_missing"
            and not any(re.fullmatch(r"rs[0-9]+", str(row.get(key) or ""))
                        for key in ("reported_rsid", "rsid")))


def _missing_rsid_only_receipt(receipt: Mapping) -> bool:
    excluded = [row for row in _rows(receipt.get("excluded_rows")) if _active_gap(row)]
    in_scope = [row for row in excluded if row.get("scope_relevant") is True]
    count = receipt.get("excluded_scope_count")
    return (bool(in_scope) and all(_missing_proxy_rsid(row) for row in excluded)
            and (count is None or type(count) is int and count == len(in_scope))
            and _state(receipt) in {"unknown", "partial", *_COMPLETE})


def assess_ld_coverage(ld_scope: Mapping, ld_metadata: Mapping | None = None, *,
                       selected_source_identities_validated: bool = False) -> dict[str, Any]:
    """Review an already validated LDReportScope (e.g. dataclasses.asdict).

    Metadata is the *final* ld_expansion object. Historical attempts/query_ledger
    are deliberately ignored. No reference identities or LD pairs are invented.
    Optional proxy acquisition gaps are method notes once the selected source
    identities are independently established. An LD service's missing source rsID
    is an expansion gap, not a failure of the primary selected-variant identity.
    """
    scope, metadata = map(_mapping, (ld_scope, ld_metadata))
    affected = {"source_ids": [], "proxy_ids": [], "reported_proxy_ids": []}
    if (scope.get("provided") is False or scope.get("disabled") is True
            or _state(scope) == "disabled" or not scope and not metadata):
        return {"state": "not_applicable", "facts": [], "reason_codes": [], "affected": affected, "root_causes": []}
    reasons, facts = [], []

    def add(code: str, fact: str) -> None:
        if code not in reasons:
            reasons.append(code)
            facts.append(fact)

    if scope.get("valid") is False:
        add("ld_scope_identity_invalid", "The recorded LD scope did not pass source, proxy or reference identity validation.")
        affected["source_ids"].extend(_ids(scope.get("source_ids")))
    if scope.get("complete") is False:
        add("ld_reference_scope_incomplete", "The requested LD reference scope is incomplete; validated source and proxy evidence remains available.")
    if scope.get("proxy_evidence_complete") is False:
        add("ld_proxy_evidence_scope_incomplete", "Evidence retrieval for part of the verified proxy scope is incomplete.")
        affected["proxy_ids"].extend(_ids(scope.get("proxy_ids")))

    # These fields describe final unresolved entities, not historical failures.
    failures = [row for row in _rows(metadata.get("ld_lookup_failures")) if _active_gap(row)]
    omitted = _ids(metadata.get("omitted_source_variant_ids"))
    source_gaps = [row for row in _rows(metadata.get("technical_gaps")) if _active_gap(row)
                   and row.get("phase") in {"source_identity", "ld_window"}]
    if failures or omitted or source_gaps:
        add("ld_source_scope_incomplete", "LD identity or reference retrieval remains unresolved for part of the selected source scope.")
        affected["source_ids"].extend(omitted)
        affected["source_ids"].extend(_ids([r.get("source_variant_id") for r in failures]))
        affected["source_ids"].extend(_ids([r.get("entity_id") for r in source_gaps]))
    unresolved = [r for r in _rows(metadata.get("unresolved_proxies"))
                  if _active_gap(r) and not _missing_proxy_rsid(r)]
    if unresolved:
        add("ld_proxy_identity_unresolved", "Some in-scope LD proxies could not be verified and were excluded from the admitted proxy scope.")
        affected["reported_proxy_ids"].extend(_ids([r.get("rsid", r.get("reported_rsid")) for r in unresolved]))

    receipts = [row for row in _rows(metadata.get("ld_lookup_audit")) if _active_gap(row)]
    ignored_receipts = []
    for receipt in receipts:
        if _missing_rsid_only_receipt(receipt):
            ignored_receipts.append(receipt)
            continue
        excluded = [r for r in _rows(receipt.get("excluded_rows"))
                    if _active_gap(r) and r.get("scope_relevant") is True and not _missing_proxy_rsid(r)]
        if receipt.get("requested_scope_complete") is False or excluded:
            add("ld_in_scope_records_unresolved", "Some records inside the requested LD threshold and window were not verified or retained.")
            affected["source_ids"].extend(_ids([receipt.get("source_variant_id")]))
            affected["reported_proxy_ids"].extend(_ids([r.get("reported_rsid") for r in excluded]))

    # The acquired scope and its PARTIAL receipts remain immutable. Dismiss the
    # aggregate Gate cause only when the receipts explain it completely.
    source_aliases = {row[0]: row[1] for row in _rows(scope.get("source_aliases"))
                      if isinstance(row, (list, tuple)) and len(row) == 2
                      and all(isinstance(value, str) for value in row)}
    sources = {source_aliases.get(value, value) for value in _ids(scope.get("source_ids"))}
    audited_sources = {source_aliases.get(value, value)
                       for value in _ids([receipt.get("source_variant_id") for receipt in receipts])}
    missing_rsid_only = bool(ignored_receipts) and (
        scope.get("valid") is not False
        and not any(_active_gap(row) for row in _rows(metadata.get("technical_gaps")))
        and all(issue == "Some in-scope LD records could not be verified; the valid records were retained."
                for issue in _rows(metadata.get("issues")))
        and all(receipt.get("requested_scope_complete") is True or receipt in ignored_receipts
                for receipt in receipts)
        and sources.issubset(audited_sources)
        and all(reason == "ld_reference_scope_incomplete" for reason in reasons)
    )
    if missing_rsid_only:
        reasons.clear()
        facts.clear()
    for key in affected:
        affected[key] = _ids(affected[key])
    optional_only = selected_source_identities_validated
    state = (("note" if optional_only else "warning") if reasons else
             "pass" if scope.get("complete") is True or missing_rsid_only else "unknown")
    return {"state": state, "facts": facts, "reason_codes": reasons, "affected": affected,
            "optional_proxy_scope": optional_only,
            "missing_rsid_only": missing_rsid_only,
            "root_causes": [{"code": reason} for reason in reasons]}


def source_scoped_coverage_events(events: Sequence, ld_metadata: Mapping, *,
                                  source_ids: Sequence[str], proxy_ids: Sequence[str]) -> tuple[list, list[str]]:
    """Project complete source-batch receipts for review without changing the ledger.

    A failed merged source/proxy module is supplementary only when every selected
    source has a successful final batch receipt. Missing or mixed batch ownership,
    incomplete source receipts, and unrelated later module failures remain visible.
    """
    sources, proxies = set(source_ids), set(proxy_ids)
    batches = {}
    for row in _rows(_mapping(ld_metadata).get("evidence_batches")):
        if isinstance(row, Mapping):
            batches.setdefault((row.get("step"), row.get("tool")), []).append(row)
    projected, locus_modules = [], set()
    categories = ("unavailable", "blocked", "failed", "truncated")
    for event in events:
        if not isinstance(event, Mapping):
            projected.append(event)
            continue
        details = _mapping(event.get("details")) or event
        receipts = batches.get((details.get("step"), details.get("tool")), [])
        coverage = _mapping(details.get("coverage"))
        requested = set(_ids(coverage.get("requested")))
        failed = set().union(*[set(_ids(coverage.get(category))) for category in categories])
        eligible = (event.get("phase") == "semantic_tool_finished" and sources and receipts
                    and requested and failed and not coverage.get("next_cursor"))
        if not eligible:
            projected.append(event)
            continue
        counts = {row.get("batch_count") for row in receipts}
        indices = {row.get("batch_index") for row in receipts}
        if (len(counts) != 1 or not all(type(count) is int and count > 0 for count in counts)
                or indices != set(range(1, next(iter(counts)) + 1)) or len(indices) != len(receipts)):
            projected.append(event)
            continue
        source_rows, source_coverage, malformed = [], set(), False
        for row in receipts:
            identifiers = set(_ids(row.get("variant_ids")))
            if identifiers and identifiers.issubset(sources):
                source_rows.append(row)
                source_coverage.update(identifiers)
            elif not identifiers or not identifiers.issubset(proxies):
                malformed = True
        if malformed or source_coverage != sources:
            projected.append(event)
            continue
        source_complete = all(_state(row) in _COMPLETE
            and requested.issubset(set(_ids(_mapping(row.get("coverage")).get("completed")))
                                   | set(_ids(_mapping(row.get("coverage")).get("no_data"))))
            and not any(_mapping(row.get("coverage")).get(category) for category in categories)
            for row in source_rows)
        if not source_complete:
            projected.append(event)
            continue
        no_data = set.intersection(*[set(_ids(_mapping(row.get("coverage")).get("no_data"))) for row in source_rows])
        scoped_coverage = {**coverage, "completed": sorted(requested - no_data), "no_data": sorted(no_data),
                           **{category: [] for category in categories}}
        scoped = {**details, "coverage": scoped_coverage,
                  "status": "no_data" if requested.issubset(no_data) else "success"}
        projected.append({**event, "details": scoped} if isinstance(event.get("details"), Mapping) else scoped)
        if details.get("tool") == "ot_get_locus_gene_evidence":
            locus_modules.update(failed)
    return projected, sorted(locus_modules)
