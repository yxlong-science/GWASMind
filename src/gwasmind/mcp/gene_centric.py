"""Gene-centric V2G retrieval behind the semantic gateway boundary.

For live gene prioritization the lead-window genes replace candidate discovery and
the V2G gene-centric queries replace the variant-centric MCP modules. Everything
after the gateway (sufficiency gate, stage bridge and Gate1, deferred ranking,
report writer, Gate2 and rendering) receives the usual ``ToolCallResult``: the
evidence records, a complete ``candidate_audit`` and the usual progress events.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..audit import sha256_json
from ..biological_context import (
    ContextResolutionError,
    biological_context_from_dict,
    curated_biological_context,
    define_biological_context,
    standardize_biological_context,
)
from ..configuration import parameter
from ..models import EvidenceKind, EvidenceRecord, LocusQuery, analysis_variant_ids
from ..phenotype_scope import (
    PhenotypeScopeError,
    scope_ids,
    target_only_phenotype_scope,
    validate_phenotype_scope,
    verify_phenotype_scope_identities,
)
from ..progress import emit_progress
from ..review_coverage import assess_frozen_molqtl_coloc_coverage
from ..tools import ToolCallResult, ToolStatus
from ..v2g.ledger import remap_card_evidence_ids
from ..v2g.locus_set import proxies_from_ld_expansion
from ..v2g.models import GRCH38_VARIANT_ID, V2GRequest, V2GResult
from ..v2g.pipeline import run_v2g
from ..v2g.report import public_cards
from ..v2g.retrieval import graphql_transport_executor
from .normalization import semantic_records_to_evidence

POLICY_VERSION = "v2g-gene-window-1"
LOCUS_CONTEXT_MODULES = ("locus.variant",)
GENE_EVIDENCE_MODULES = ("locus.variant_consequence", "locus.qtl", "locus.colocalisation", "locus.enhancer_to_gene")
TARGET_MODULES = ("target.core", "target.expression", "target.go")
STEPS = (
    ("ot_get_locus_context", LOCUS_CONTEXT_MODULES, True),
    ("ot_get_locus_gene_evidence", GENE_EVIDENCE_MODULES, True),
    ("ot_get_target_profiles", TARGET_MODULES, False),
)


def resolve_gene_centric_context(query: LocusQuery) -> LocusQuery:
    """Bind every live entrypoint to a verified plan before any provider access.

    Explicit contexts and terms are authoritative. Missing context may use a
    recognized physiology plan; an explicit unresolved plan is never replaced.
    Persisted inputs are not mutated when a live request receives a frozen plan.
    """
    try:
        if query.biological_context is not None:
            context = biological_context_from_dict(query.biological_context, phenotype=query.trait)
            if not context.is_defined:
                raise ContextResolutionError("explicit_context_empty")
            if query.relevant_tissues and tuple(query.relevant_tissues) != context.relevant_terms:
                raise ContextResolutionError("explicit_context_terms_mismatch")
            context = standardize_biological_context(context)
        elif query.relevant_tissues:
            if any(not isinstance(term, str) or not term.strip() for term in query.relevant_tissues):
                raise ContextResolutionError("explicit_terms_invalid")
            context = standardize_biological_context(define_biological_context(
                query.trait, explicit_terms=tuple(query.relevant_tissues)))
        else:
            context = curated_biological_context(query.trait, reason="live_entrypoint_context_missing")
            if context is None:
                raise ContextResolutionError("phenotype_requires_explicit_context")
        if not context.is_defined or (context.biosample_mapping or {}).get("status") != "resolved":
            raise ContextResolutionError("context_identity_unresolved")
    except ContextResolutionError:
        raise
    except (TypeError, ValueError) as exc:
        raise ContextResolutionError("explicit_context_invalid") from exc
    return replace(query, biological_context=context.to_dict(), relevant_tissues=context.relevant_terms)


def lead_and_selected(query: LocusQuery) -> tuple[str, tuple[str, ...]]:
    """The most significant valid source variant leads; the other source variants are selected anchors."""

    valid = [row for row in query.variants if GRCH38_VARIANT_ID.fullmatch(row.variant_id)]
    if not valid:
        raise ValueError("No selected variant has a GRCh38 chr_pos_ref_alt identifier.")

    def p_value(row) -> float:
        value = row.p_value
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
        return float(value) if valid else math.inf

    ordered = sorted(valid, key=lambda row: (p_value(row), row.variant_id))
    lead = ordered[0].variant_id
    chromosome = lead.split("_", 1)[0]
    return lead, tuple(dict.fromkeys(row.variant_id for row in ordered[1:]
                                     if row.variant_id.split("_", 1)[0] == chromosome))


def _ld_status(query: LocusQuery) -> str:
    if query.ld_expansion is None:
        return "not_requested"
    return "partial" if query.ld_expansion.get("complete") is False else "complete"


def _module_statuses(result: V2GResult, counts: Mapping[str, int]) -> dict[str, str]:
    """Preserve channel completion independently of the number of retained records."""
    channels = {
        "locus.variant": "coding", "locus.variant_consequence": "coding",
        "locus.qtl": "qtl", "locus.colocalisation": "coloc",
        "locus.enhancer_to_gene": "e2g", "target.core": "function",
        "target.expression": "expression", "target.go": "function",
    }
    values = {}
    coloc = assess_frozen_molqtl_coloc_coverage(getattr(result, "coverage", {}).get("colocalisation"))
    for module, channel in channels.items():
        if module == "locus.colocalisation" and coloc is not None:
            values[module] = ("success" if counts.get(module) else "no_data") if coloc["complete"] else coloc["module_status"]
            continue
        states = {card.get("coverage", {}).get(channel, "not_requested") for card in result.genes}
        # A missing receipt cannot establish a completed negative search.
        if not states or states == {"not_requested"}:
            values[module] = "unavailable"
        elif states == {"failed"}:
            values[module] = "failed"
        elif states.intersection({"failed", "partial", "not_requested"}):
            values[module] = "partial"
        else:
            observed = (int(result.retrieval_status.get("variants_resolved") or 0) > 0
                        if module == "locus.variant" else bool(counts.get(module)))
            values[module] = "success" if observed else "no_data"
    return values


def _coverage(modules: Sequence[str], statuses: Mapping[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"requested": list(modules), "completed": [], "no_data": [],
                                    "unavailable": [], "blocked": [], "failed": [], "truncated": []}
    for module in modules:
        value = statuses.get(module, "no_data")
        key = {"success": "completed", "no_data": "no_data", "failed": "failed", "blocked": "blocked",
               "partial": "truncated", "unavailable": "unavailable"}[value]
        groups[key].append(module)
    return groups


def _step_status(coverage: Mapping[str, list[str]]) -> str:
    return "incomplete" if coverage["failed"] or coverage["truncated"] or coverage["unavailable"] else "success"


def candidate_audit(result: V2GResult, query: LocusQuery, cards: Sequence[Mapping[str, Any]], *,
                    records: Sequence[EvidenceRecord]) -> dict[str, Any]:
    """The candidate audit for the lead-window gene universe (every window gene is a candidate)."""

    by_gene = {card["gene_id"]: card for card in cards}
    from ..staged_reasoning import has_direct_locus_gene_evidence

    records_by_gene: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        if record.gene_id:
            records_by_gene.setdefault(record.gene_id, []).append(record)
    targets_ok = int(result.retrieval_status.get("targets_failed") or 0) == 0
    genes = []
    for gene in result.window.genes:
        card = by_gene.get(gene.gene_id, {})
        evidence = card.get("evidence", {})
        returned = {key: bool((evidence.get(category) or {}).get("records"))
                    for key, category in (("variant_consequence", "coding"), ("qtl", "qtl"),
                                          ("colocalisation", "coloc"), ("enhancer_to_gene", "e2g"))}
        returned["expression"] = bool((evidence.get("expression") or {}).get("evidence_ids"))
        expression_status = card.get("coverage", {}).get("expression", "not_requested")
        expression_complete = expression_status in {"observed", "not_observed"}
        gene_records = records_by_gene.get(gene.gene_id, [])
        coding_direct = has_direct_locus_gene_evidence(
            [record for record in gene_records if record.kind is EvidenceKind.VARIANT_ANNOTATION])
        auxiliary = card.get("auxiliary", {})
        genes.append({
            "gene_id": gene.gene_id, "symbol": gene.symbol,
            "sources": [{"distance_bp": auxiliary.get("body_distance_bp"), "module": "v2g.gene_window",
                         "scope": "local", "reason": "gene_body_overlaps_lead_window",
                         "record_key": f"window:{gene.gene_id}"}],
            "evidence_returned": returned,
            "expression_specificity": {"status": "not_used_for_admission", "qualified": False, "threshold": None,
                                       "fold_threshold": None, "method": "not_applicable_gene_window",
                                       "groups": []},
            "expression_retrieval_status": "complete" if expression_complete else expression_status,
            "direct_relation_verified": has_direct_locus_gene_evidence(gene_records),
            "coding_snp_relation_verified": coding_direct,
        })
    candidate_ids = [gene.gene_id for gene in result.window.genes]
    window_complete = result.window.status == "complete"
    expressions_complete = bool(genes) and all(gene["expression_retrieval_status"] == "complete" for gene in genes)
    policy = {"version": POLICY_VERSION, "window_rule": parameter("v2g.gene_window_rule"),
              "flank_bp": parameter("analysis.local_gene_window_bp"),
              "window_source": result.window.source}
    audit = {
        "mode": "gene_centric_window", "strategy": "gene_centric_window", "policy": policy,
        "candidate_ids": candidate_ids, "input_candidate_ids": list(query.candidate_genes),
        "variant_ids": list(analysis_variant_ids(query)),
        "candidate_pool_size": len(candidate_ids),
        "genes": genes,
        "candidate_discovery_complete": window_complete and targets_ok,
        "locus_discovery_complete": window_complete,
        "expression_discovery_complete": expressions_complete, "expression_retrieval_complete": expressions_complete,
        "expression_assessment_status": "not_applicable_gene_window",
        "direct_evidence_candidates": sum(gene["direct_relation_verified"] for gene in genes),
    }
    if query.ld_expansion is not None:
        audit["ld_expansion_complete"] = query.ld_expansion.get("complete") is not False
        # Proxy evidence is retrieved through the variant layer (consequences and E2G).
        audit["ld_proxy_retrieval_complete"] = all(
            card.get("coverage", {}).get(channel) in {"observed", "not_observed"}
            for card in cards for channel in ("coding", "e2g"))
    coloc_receipt = result.coverage.get("colocalisation")
    coloc = assess_frozen_molqtl_coloc_coverage(coloc_receipt)
    if coloc is not None:
        audit["colocalisation_coverage"] = deepcopy(coloc_receipt)
        audit["colocalisation_detail_complete"] = coloc["complete"]
    audit["candidate_fingerprint"] = sha256_json(candidate_ids)
    audit["audit_fingerprint"] = sha256_json({key: value for key, value in audit.items() if key != "genes"})
    return audit


def _progress_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if key not in {"genes", "v2g_cards", "v2g"}}


async def gene_centric_evidence(query: LocusQuery, *, progress_sink, executor=None,
                                http_get=None) -> tuple[V2GResult, list[str]]:
    """Run the V2G retrieval for one locus on the ranking LD expansion.

    ``executor`` (GraphQL) and ``http_get`` (gene-window REST lookup) are injectable for tests.
    """

    query = resolve_gene_centric_context(query)
    notes: list[str] = []
    transport = None
    if executor is None:
        from ..providers.live import OfficialOpenTargetsGraphQLTransport

        transport = OfficialOpenTargetsGraphQLTransport()
        executor = graphql_transport_executor(transport)
    try:
        lead, selected = lead_and_selected(query)
        phenotype_scope = (query.biological_context or {}).get("phenotype_scope")
        from ..v2g.progress import RetrievalProgress
        async with RetrievalProgress(progress_sink).operation("phenotype"):
            if phenotype_scope is not None:
                phenotype_scope = validate_phenotype_scope(phenotype_scope, phenotype=query.trait)
                if query.trait_ids and set(query.trait_ids) != set(scope_ids(phenotype_scope)):
                    raise PhenotypeScopeError("query_trait_ids_disagree_with_frozen_scope")
                phenotype_scope = await verify_phenotype_scope_identities(executor, phenotype_scope)
            else:
                phenotype_scope = await target_only_phenotype_scope(executor, query.trait,
                    target_ids=tuple(query.trait_ids))
        efos = scope_ids(phenotype_scope)
        context = biological_context_from_dict(query.biological_context, phenotype=query.trait)
        query = replace(query, trait_ids=efos,
            biological_context=replace(context, phenotype_scope=phenotype_scope).to_dict())
        if not efos:
            notes.append(f"The trait '{query.trait}' was not resolved to an Open Targets phenotype; "
                         "target-disease association was not requested.")
        elif phenotype_scope["status"] == "expansion_not_assessed":
            notes.append("Related phenotype expansion was not assessed; phenotype evidence is restricted to the verified target identity.")
        request = V2GRequest(locus_id=query.locus_id, lead_variant_id=lead, trait=query.trait,
                             selected_variants=selected, trait_efo_ids=efos,
                             context_terms=tuple(term for term in query.relevant_tissues if term.strip()),
                             biological_context=deepcopy(query.biological_context),
                             phenotype_scope=deepcopy(phenotype_scope),
                             explicit_candidates=tuple(query.candidate_genes))
        proxies = proxies_from_ld_expansion(query.ld_expansion, analysis_variant_ids(query))
        result = await run_v2g(request, executor=executor, http_get=http_get,
                               progress_sink=progress_sink,
                               precomputed_ld=(proxies, _ld_status(query)))
    finally:
        if transport is not None:
            transport.close()
    result.issues[:0] = notes
    return result, notes


async def gene_centric_tool_result(adapter, query: LocusQuery, kinds: set[EvidenceKind], *, call_id: str,
                                   requested: tuple[EvidenceKind, ...], request_pointer, executor=None,
                                   http_get=None) -> ToolCallResult:
    sink = adapter.progress_sink
    emit_progress(sink, "semantic_plan_ready", steps=len(STEPS), tools=[tool for tool, _, _ in STEPS])
    for index, (tool, modules, required) in enumerate(STEPS[:2], start=1):
        emit_progress(sink, "semantic_tool_started", progress_scope="workflow_step", step=index,
                      total_steps=len(STEPS), tool=tool, modules=list(modules), required=required)
    try:
        query = resolve_gene_centric_context(query)
        result, _ = await gene_centric_evidence(query, progress_sink=sink, executor=executor, http_get=http_get)
        query = replace(query, trait_ids=tuple(result.request.trait_efo_ids),
            biological_context=deepcopy(result.request.biological_context))
    except Exception as exc:  # noqa: BLE001 - preserve input or retrieval failure for downstream gates
        from ..v2g.capacity import EvidenceCapacityError
        invalid_context = isinstance(exc, ContextResolutionError) or (
            isinstance(exc, PhenotypeScopeError) and exc.reason != "frozen_identity_verification_unavailable")
        message = str(exc) if isinstance(exc, (ContextResolutionError, PhenotypeScopeError, EvidenceCapacityError)) else (
            f"gene-centric retrieval failed ({type(exc).__name__})")
        failure_raw = None
        capacity_issues = ()
        if isinstance(exc, EvidenceCapacityError):
            from ..academic_review import operation_issue
            capacity_issues = (operation_issue("SYSTEM_ERROR", category="processing_limit", scope="retrieval",
                message=message, blocking=True, reasons=("evidence_capacity_exceeded",)),)
            if adapter.artifacts and getattr(exc, "retrieval_snapshot", None):
                failure_raw = adapter.artifacts.write_json(f"tool_calls/{call_id}/raw_response.json",
                    {"pipeline": "gene_centric", "classification_completed": False,
                     "processing_error": message, **exc.retrieval_snapshot}, compact=True)
        for index, (tool, modules, required) in enumerate(STEPS[:2], start=1):
            emit_progress(sink, "semantic_tool_finished", progress_scope="workflow_step", step=index,
                          total_steps=len(STEPS), tool=tool, modules=list(modules), required=required,
                          status="error", coverage=_coverage(modules, dict.fromkeys(
                              modules, "blocked" if invalid_context else "failed")),
                          warnings=[], records=0, module_counts={}, unclassified_records=0)
        return ToolCallResult(call_id=call_id,
                              status=ToolStatus.INVALID_INPUT if invalid_context else ToolStatus.ERROR,
                              requested_kinds=requested, records=(),
                              policy_blocks=(), request_artifact=request_pointer, response_artifact=None,
                              response_sha256=sha256_json([]), error=message, resolved_query=query,
                              raw_response_artifact=failure_raw,
                              raw_response_sha256=failure_raw.sha256 if failure_raw else None,
                              operation_issues=capacity_issues)

    normalized = semantic_records_to_evidence(result.ledger, default_release=adapter.service.data_release)
    mapping = {record.payload["provenance"]["evidence_id"]: record.evidence_id for record in normalized}
    cards = remap_card_evidence_ids(list(result.genes), mapping)
    from ..variant_identity_registry import (
        build_variant_identity_registry, ot_identity_records, public_variant_labels,
    )
    display_variant_ids = set(result.locus.query_ids())
    display_variant_ids.update(row.get("cs_lead_variant_id") for card in cards
                               for row in card.get("evidence", {}).get("qtl", {}).get("records", []))
    identity_records = [row for row in ot_identity_records(result.raw_responses)
                        if row["variant"].get("id") in display_variant_ids]
    identity_registry = build_variant_identity_registry(
        variant_records=identity_records, ld_expansion=query.ld_expansion)
    if query.ld_expansion is not None:
        query = replace(query, ld_expansion={**query.ld_expansion,
                                           "variant_identity_registry": identity_registry})
    display_cards = remap_card_evidence_ids(list(result.display_genes or result.genes), mapping)
    emit_progress(sink, "gene_evidence_cards", cards=public_cards(display_cards,
        source_normalizations=(query.ld_expansion or {}).get("source_identity_normalizations", ()),
        ld_expansion=query.ld_expansion, biological_context=result.request.biological_context,
        phenotype_scope=result.request.phenotype_scope, variant_identity_registry=identity_registry),
        variant_labels=public_variant_labels(identity_registry))
    allowed, decisions = adapter.policy.filter_records(list(normalized))
    blocked = tuple(item.reason for item in decisions if not item.allowed)
    records = tuple(allowed)
    counts = Counter(str(record.payload.get("module")) for record in records)
    statuses = _module_statuses(result, counts)

    for index, (tool, modules, required) in enumerate(STEPS, start=1):
        if index == 3:
            emit_progress(sink, "semantic_tool_started", progress_scope="workflow_step", step=index,
                          total_steps=len(STEPS), tool=tool, modules=list(modules), required=required)
        coverage = _coverage(modules, statuses)
        emit_progress(sink, "semantic_tool_finished", progress_scope="workflow_step", step=index,
                      total_steps=len(STEPS), tool=tool, modules=list(modules), required=required,
                      status=_step_status(coverage), coverage=coverage, warnings=[],
                      records=sum(counts.get(module, 0) for module in modules),
                      module_counts={module: counts[module] for module in modules if counts.get(module)},
                      unclassified_records=0)
        if index == 2:
            audit = candidate_audit(result, query, cards, records=records)
            emit_progress(sink, "candidate_universe_discovered", targets=audit["candidate_ids"],
                          source_tool=tool, **_progress_summary(audit))

    audit = candidate_audit(result, query, cards, records=records)
    audit["v2g_cards"] = cards
    audit["v2g"] = {"retrieval_status": dict(result.retrieval_status), "issues": list(result.issues),
                    "locus": result.locus.to_dict(), "window": result.window.to_dict(),
                    "coverage": dict(result.coverage), "biological_context": deepcopy(result.request.biological_context),
                    "variant_identity_registry": identity_registry}
    raw_pointer = (adapter.artifacts.write_json(f"tool_calls/{call_id}/raw_response.json",
                                                {"pipeline": "gene_centric", **result.to_dict(include_raw=True),
                                                 "semantic_records": result.ledger}, compact=True)
                   if adapter.artifacts else None)
    incomplete = [module for module, value in statuses.items()
                  if value == "failed" and module in GENE_EVIDENCE_MODULES + LOCUS_CONTEXT_MODULES]
    status = (ToolStatus.NO_DATA if not records else
              ToolStatus.INCOMPLETE if incomplete or result.window.status != "complete" else ToolStatus.SUCCESS)
    error = None
    if result.window.status != "complete":
        error = "the lead-window gene census is incomplete"
    elif incomplete:
        error = "required gene-centric evidence modules are incomplete: " + ", ".join(sorted(incomplete))
    response_value = {"call_id": call_id, "status": status.value, "semantic_status": "gene_centric",
                      "policy_blocks": list(blocked), "warnings": list(result.issues)}
    response_pointer = (adapter.artifacts.write_record_response(f"tool_calls/{call_id}/response.json",
                                                                response_value, records)
                        if adapter.artifacts else None)
    from ..evidence_store import record_response_sha256

    resolved = replace(query, candidate_genes=tuple(audit["candidate_ids"]))
    return ToolCallResult(
        call_id=call_id, status=status, requested_kinds=requested, records=records, policy_blocks=blocked,
        request_artifact=request_pointer, response_artifact=response_pointer,
        response_sha256=(response_pointer.sha256 if response_pointer is not None
                         else record_response_sha256(response_value, records)),
        error=error,
        raw_response_artifact=raw_pointer,
        raw_response_sha256=raw_pointer.sha256 if raw_pointer is not None else sha256_json(result.ledger),
        warnings=tuple(result.issues), resolved_query=resolved, candidate_audit=audit,
        retrieval_statuses=statuses,
    )
