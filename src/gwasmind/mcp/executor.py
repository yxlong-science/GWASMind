from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from ..audit import sha256_json
from ..biological_context import (
    biological_context_from_dict,
    define_biological_context,
    standardize_biological_context,
)
from ..candidates import (
    MAX_DISCOVERED_L2G_TARGETS_PER_LOCUS,
    MAX_DISCOVERED_TARGETS,
    annotate_candidate_scope,
    annotate_molecular_ranking_scope,
    candidate_audit_summary,
    discover_candidates,
    discovery_locus_id,
    evidence_candidate_scope,
    record_target_identity_valid,
    target_ids,
)
from ..configuration import parameter
from ..expression_policy import DEFAULT_EXPRESSION_FOLD_THRESHOLD, expression_selectors_from_axes
from ..expression_specificity import DEFAULT_EXPRESSION_Z_THRESHOLD, expression_specificity
from ..models import EvidenceKind, LocusQuery, analysis_variant_ids
from ..molecular_context import (
    MOLECULAR_RETRIEVAL_SCOPE,
    MolecularRetrievalScope,
    freeze_molecular_context,
)
from ..progress import ProgressSink, emit_progress
from ..publications import normalize_pmid, parse_colocalization_publication_lineage
from ..supplementary_molecular import frozen_context_assessment, molecular_detail_anchor_verified
from .batching import annotate_analysis_origin, merge_variant_envelopes, variant_batches
from .contracts import SemanticTool, ToolEnvelope, ToolStatus
from .planner import (
    _TARGET_ID,
    _VARIANT_ID,
    SemanticEvidencePlan,
    SemanticEvidencePlanner,
    SemanticPlanStep,
)
from .regions import grch38_region_from_variant_ids
from .retrieval_state import RetrievalState, retrieval_state
from .service import SemanticToolService


class RetrievalStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT = "insufficient"
    BLOCKED = "blocked"
    INVALID_INPUT = "invalid_input"
    ERROR = "error"
    INCOMPLETE = "incomplete"


# Compatibility alias; limits and policy live in the shared candidate module.
MAX_DISCOVERED_L2G_TARGETS = MAX_DISCOVERED_L2G_TARGETS_PER_LOCUS


@dataclass(frozen=True, slots=True)
class SemanticRetrievalResult:
    status: RetrievalStatus
    query: LocusQuery
    plan: SemanticEvidencePlan | None
    calls: tuple[ToolEnvelope, ...]
    records: tuple[Mapping[str, Any], ...]
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    candidate_audit: Mapping[str, Any] | None = None
    operation_issues: tuple[Mapping[str, Any], ...] = ()
    source_identity_normalizations: tuple[Mapping[str, Any], ...] = ()


class SemanticRetrievalExecutor:
    """Execute the bounded semantic plan sequentially for one single agent."""

    async def _call_with_progress(self, tool, arguments, **details):
        from uuid import uuid4

        from ..providers.retrieval_recovery import (
            ACTIVE_OPERATION_ID,
            ACTIVE_SEMANTIC_TOOL,
            RECOVERY_SINK,
        )
        token = RECOVERY_SINK.set(self.progress_sink)
        tool_token = ACTIVE_SEMANTIC_TOOL.set(tool.value)
        details = {**details, "operation_id": uuid4().hex}
        operation_token = ACTIVE_OPERATION_ID.set(details["operation_id"])
        emit_progress(self.progress_sink, "semantic_call_started", tool=tool.value,
                      modules=list(arguments.get("modules", ())), **details)
        try:
            result = await self._call_observed(tool, arguments, **details)
            coverage = getattr(result, "coverage", None)
            records = getattr(result, "records", ())
            emit_progress(self.progress_sink, "semantic_call_finished", tool=tool.value,
                status=getattr(getattr(result, "status", None), "value", "success"),
                coverage=coverage.to_dict() if coverage is not None else {}, warnings=list(getattr(result, "warnings", ())),
                records=len(records), module_counts=_record_module_counts(records), **details)
            return result
        except BaseException:
            emit_progress(self.progress_sink, "semantic_call_finished", tool=tool.value, status="failed", **details)
            raise
        finally:
            ACTIVE_OPERATION_ID.reset(operation_token)
            RECOVERY_SINK.reset(token)
            ACTIVE_SEMANTIC_TOOL.reset(tool_token)

    async def _call_observed(self, tool, arguments, **details):
        """Keep a pending real request observable without changing its timeout."""
        if self.progress_sink is None:
            return await self.service.call(tool, arguments)
        task = asyncio.ensure_future(self.service.call(tool, arguments))
        started = time.monotonic()
        reported_wait = False
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=parameter("execution.stream_progress_interval_seconds"))
                if done:
                    result = task.result()
                    if reported_wait:
                        emit_progress(self.progress_sink, "semantic_request_finished", tool=tool.value,
                            status=getattr(getattr(result, "status", None), "value", "success"),
                            elapsed_seconds=max(0, int(time.monotonic() - started)), **details)
                    return result
                reported_wait = True
                emit_progress(self.progress_sink, "semantic_tool_waiting", tool=tool.value,
                              elapsed_seconds=max(0, int(time.monotonic() - started)), **details)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def __init__(
        self,
        service: SemanticToolService,
        *,
        planner: SemanticEvidencePlanner | None = None,
        page_size: int = parameter("retrieval.semantic_page_size"),
        progress_sink: ProgressSink | None = None,
        expression_z_threshold: float = DEFAULT_EXPRESSION_Z_THRESHOLD,
        expression_fold_threshold: float = DEFAULT_EXPRESSION_FOLD_THRESHOLD,
        expression_context_selectors=None,
        expression_context_groups=None,
        expression_detection_rules=None,
        expression_batch_size: int = parameter("retrieval.expression_target_batch_size"),
        source_identity_resolver=None,
    ):
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= 500
        ):
            raise ValueError("page_size must be between 1 and 500")
        self.service = service
        self.planner = planner or SemanticEvidencePlanner()
        self.page_size = page_size
        self.progress_sink = progress_sink
        self.source_identity_resolver = source_identity_resolver
        self.expression_z_threshold = expression_z_threshold
        self.expression_fold_threshold = expression_fold_threshold
        self.expression_context_selectors = expression_context_selectors
        self.expression_context_groups = expression_context_groups
        self.expression_detection_rules = expression_detection_rules
        # Validate numeric/source settings before making retrieval calls. Context
        # groups are validated against the resolved query context below.
        expression_specificity(
            [],
            relevant_terms=(),
            threshold=expression_z_threshold,
            fold_threshold=expression_fold_threshold,
            detection_rules=expression_detection_rules,
        )
        if (
            isinstance(expression_batch_size, bool)
            or not isinstance(expression_batch_size, int)
            or not 1 <= expression_batch_size <= 200
        ):
            raise ValueError("expression_batch_size must be between 1 and 200")
        self.expression_batch_size = expression_batch_size

    async def retrieve(
        self,
        query: LocusQuery,
        *,
        requested_kinds: frozenset[EvidenceKind] | None = None,
    ) -> SemanticRetrievalResult:
        from ..providers.retrieval_recovery import ANALYSIS_VARIANTS, RETRIEVAL_FAILURE_RECEIPTS
        # Task-local scope prevents simultaneous loci from borrowing each
        # other's member filters, including when an executor is reused.
        token = ANALYSIS_VARIANTS.set(())
        molecular_token = MOLECULAR_RETRIEVAL_SCOPE.set(None)
        failures = []
        from ..source_normalization import validate_source_normalization
        normalizations = [dict(row) for row in (query.ld_expansion or {}).get("source_identity_normalizations", ())
                          if validate_source_normalization(row)]
        failure_token = RETRIEVAL_FAILURE_RECEIPTS.set(failures)
        try:
            result = await self._retrieve(query, requested_kinds=requested_kinds,
                                          source_normalizations=normalizations)
            result = replace(result, source_identity_normalizations=tuple({
                row["original_id"]: row for row in normalizations}.values()))
            required = {SemanticTool.RESOLVE_ENTITIES.value, SemanticTool.GET_LOCUS_CONTEXT.value,
                        SemanticTool.GET_LOCUS_GENE_EVIDENCE.value}
            terminal = tuple(row for row in failures if row.get("tool") in required)
            return replace(result, operation_issues=(*result.operation_issues, *terminal)) if terminal else result
        finally:
            ANALYSIS_VARIANTS.reset(token)
            MOLECULAR_RETRIEVAL_SCOPE.reset(molecular_token)
            RETRIEVAL_FAILURE_RECEIPTS.reset(failure_token)

    async def _normalize_source_context(self, query, envelope, *, step, step_index):
        """Repair only proven source reversals, then retrieve their actual context.

        Successful empty catalogue results remain empty when no unique reverse
        identity is established. Existing valid source IDs incur no extra lookup.
        """
        from ..source_normalization import build_source_normalization, validate_source_normalization

        unchanged = (query, envelope, [], [], [])
        if (query.benchmark or str(query.genome_build or "").casefold() not in {"grch38", "hg38"}
                or "locus.variant" not in (*envelope.coverage.completed, *envelope.coverage.no_data)
                or "locus.variant" in (*envelope.coverage.failed, *envelope.coverage.truncated)):
            return unchanged
        observed = {(row.get("subject") or {}).get("id") for row in envelope.records
                    if row.get("module") == "locus.variant"}
        missing = [row.variant_id for row in query.variants
                   if _VARIANT_ID.fullmatch(row.variant_id) and row.variant_id not in observed]
        if not missing:
            return unchanged
        proofs = {row["original_id"]: dict(row)
                  for row in (query.ld_expansion or {}).get("source_identity_normalizations", ())
                  if validate_source_normalization(row) and row["original_id"] in missing}
        needed = [identifier for identifier in missing if identifier not in proofs]
        if needed and self.source_identity_resolver is not None:
            try:
                results = self.source_identity_resolver(needed)
                if inspect.isawaitable(results):
                    results = await results
                if isinstance(results, Mapping):
                    for identifier in needed:
                        proof = build_source_normalization(identifier, results.get(identifier), query.genome_build)
                        if proof is not None:
                            proofs[identifier] = proof
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - optional provider boundary
                # A supplemental check cannot reclassify an empty primary lookup
                # as invalid input, nor disclose provider error text or secrets.
                emit_progress(self.progress_sink, "source_identity_normalization_check",
                    status="incomplete", error_type=type(exc).__name__)
        if not proofs:
            return unchanged
        normalized, conflicts = _apply_source_normalizations(query, tuple(proofs.values()))
        source_ids = [row.get("variant_id") for row in (query.ld_expansion or {}).get("source_observations", ())]
        emit_progress(self.progress_sink, "source_identity_normalization_applied",
            source_variant_ids=source_ids or [row.variant_id for row in query.variants],
            normalizations=list(proofs.values()), conflicting_source_variant_ids=conflicts,
            source_p_preserved=True, effect_sign_changed=False)
        if normalized is None:
            return None, envelope, [], list(proofs.values()), conflicts
        from ..providers.retrieval_recovery import ANALYSIS_VARIANTS
        ANALYSIS_VARIANTS.set(tuple(analysis_variant_ids(normalized)))
        retained = set(analysis_variant_ids(normalized))
        records = tuple(row for row in envelope.records if _record_uses_retained_analysis_variant(row, retained))
        changed = sorted({proof["canonical_id"] for proof in proofs.values()} & retained)
        extras = []
        size = parameter("execution.max_variants_per_query")
        for start in range(0, len(changed), size):
            identifiers = changed[start:start + size]
            arguments = _arguments_for_step(step.tool, step.modules, normalized, self.page_size)
            region = grch38_region_from_variant_ids(identifiers)
            arguments.update(variant_ids=identifiers, region=region.to_dict() if region is not None else None)
            extra = await self._call_with_progress(step.tool, arguments, step=step_index,
                                                   source_identity_normalized=True)
            extras.append(extra)
        if extras:
            envelope = merge_variant_envelopes((replace(envelope, records=records), *extras))
        else:
            envelope = replace(envelope, records=records)
        envelope = replace(envelope, records=annotate_analysis_origin(envelope.records, normalized))
        return normalized, envelope, extras, list(proofs.values()), conflicts

    async def _retrieve(
        self,
        query: LocusQuery,
        *,
        requested_kinds: frozenset[EvidenceKind] | None = None,
        source_normalizations=None,
    ) -> SemanticRetrievalResult:
        emit_progress(self.progress_sink, "semantic_input_validation")
        initial_operation_issues = []
        try:
            query.validate()
            if source_normalizations:
                original_query = query
                normalized, conflicts = _apply_source_normalizations(query, source_normalizations)
                if conflicts:
                    initial_operation_issues.append({"code": "normalization_conflict",
                        "state": "STOP" if normalized is None else "PARTIAL",
                        "reason": "canonical_sources_have_conflicting_gwas_observations",
                        "source_variant_ids": conflicts,
                        "valid_source_count": 0 if normalized is None else len(normalized.variants)})
                emit_progress(self.progress_sink, "source_identity_normalization_applied",
                    source_variant_ids=[row.get("variant_id") for row in
                                        (query.ld_expansion or {}).get("source_observations", ())]
                                       or [row.variant_id for row in query.variants],
                    normalizations=list(source_normalizations), conflicting_source_variant_ids=conflicts,
                    source_p_preserved=True, effect_sign_changed=False)
                if normalized is None:
                    return SemanticRetrievalResult(RetrievalStatus.INVALID_INPUT, original_query, None, (), (),
                        ("Source normalization found conflicting GWAS observations for every source identity.",),
                        operation_issues=tuple(initial_operation_issues))
                query = normalized
            from ..providers.retrieval_recovery import ANALYSIS_VARIANTS
            ANALYSIS_VARIANTS.set(tuple(analysis_variant_ids(query)))
            if query.genome_build and query.genome_build.casefold() in {"grch37", "hg19"}:
                raise ValueError(
                    "semantic Open Targets retrieval requires GRCh38 coordinates; "
                    "GRCh37/hg19 input must be converted explicitly"
                )
            supplied_context = getattr(query, "biological_context", None)
            context = (
                biological_context_from_dict(supplied_context, phenotype=query.trait)
                if supplied_context is not None
                else define_biological_context(query.trait, explicit_terms=query.relevant_tissues)
            )
            if supplied_context is None and query.task == "prioritize_genes":
                context = standardize_biological_context(context)
                supplied_context = context.to_dict()
                query = replace(query, biological_context=supplied_context,
                                relevant_tissues=context.relevant_terms)
            expression_selectors = (
                self.expression_context_selectors
                if self.expression_context_selectors is not None
                else expression_selectors_from_axes(
                    organs=context.organs,
                    tissues=context.tissues,
                    cell_types=context.cell_types,
                )
            )
            molecular_context = (freeze_molecular_context(supplied_context)
                                 if supplied_context is not None and query.task == "prioritize_genes" else None)
            if molecular_context is not None:
                MOLECULAR_RETRIEVAL_SCOPE.set(MolecularRetrievalScope(
                    phase="discovery", context=molecular_context,
                    target_trait_ids=tuple(query.trait_ids), target_trait_labels=(query.trait,)))
            expression_context = expression_specificity(
                [],
                relevant_terms=context.relevant_terms,
                context_selectors=expression_selectors,
                context_groups=self.expression_context_groups,
                context_mapping=context.biosample_mapping,
            )["context_definition"]
        except ValueError as exc:
            return SemanticRetrievalResult(
                RetrievalStatus.INVALID_INPUT,
                query,
                None,
                (),
                (),
                (str(exc),),
            )
        emit_progress(self.progress_sink, "capability_check")
        capabilities = await self._call_with_progress(SemanticTool.GET_CAPABILITIES, {})
        if capabilities.status is not ToolStatus.SUCCESS or not capabilities.records:
            return SemanticRetrievalResult(
                RetrievalStatus.ERROR,
                query,
                None,
                (capabilities,),
                (),
                ((capabilities.error.message if capabilities.error else "semantic capability discovery failed"),),
                warnings=capabilities.warnings,
            )
        module_states = capabilities.records[0].get("modules")
        if not isinstance(module_states, Mapping):
            return SemanticRetrievalResult(
                RetrievalStatus.ERROR,
                query,
                None,
                (capabilities,),
                (),
                ("capability response omitted module states",),
            )
        ready = frozenset(
            str(module) for module, state in module_states.items() if state == "ready"
        )
        blocked = frozenset(
            str(module) for module, state in module_states.items() if state == "blocked"
        )
        try:
            plan = self.planner.plan(
                query,
                ready_modules=ready,
                blocked_modules=blocked,
                requested_kinds=requested_kinds,
            )
        except ValueError as exc:
            return SemanticRetrievalResult(
                RetrievalStatus.INCOMPLETE,
                query,
                None,
                (capabilities,),
                (),
                (str(exc),),
            )

        emit_progress(
            self.progress_sink,
            "semantic_plan_ready",
            steps=len(plan.steps),
            tools=[step.tool.value for step in plan.steps],
        )
        current = query
        candidate_audit = None
        expression_prefetched = False
        empty_no_data_shortlist = False
        calls: list[ToolEnvelope] = []
        records: list[Mapping[str, Any]] = []
        warnings: list[str] = []
        required_incomplete_reasons: list[str] = []
        operation_issues: list[Mapping[str, Any]] = list(initial_operation_issues)
        for index, step in enumerate(plan.steps, start=1):
            deferred_coloc = (molecular_context is not None and step.tool is SemanticTool.GET_LOCUS_GENE_EVIDENCE
                              and "locus.colocalisation" in step.modules)
            if deferred_coloc:
                modules = tuple(module for module in step.modules if module != "locus.colocalisation")
                if "locus.qtl" in ready and "locus.qtl" not in modules:
                    modules = (*modules, "locus.qtl")
                step = replace(step, modules=modules)
                # The recorded discovery plan describes the modules actually
                # requested. Coloc becomes a separate optional detail step only
                # if a frozen eligible scope makes that request applicable.
                planned = list(plan.steps)
                planned[index - 1] = step
                plan = replace(plan, steps=tuple(planned))
            if step.tool is SemanticTool.GET_TARGET_PROFILES and expression_prefetched:
                step = replace(
                    step, modules=tuple(m for m in step.modules if m != "target.expression")
                )
                if not step.modules:
                    continue
            if (
                step.tool
                in {
                    SemanticTool.GET_TARGET_PROFILES,
                    SemanticTool.GET_TARGET_DISEASE_EVIDENCE,
                    SemanticTool.GET_DRUG_EVIDENCE,
                }
                and not current.candidate_genes
            ):
                emit_progress(
                    self.progress_sink,
                    "semantic_tool_skipped",
                    step=index,
                    total_steps=len(plan.steps),
                    tool=step.tool.value,
                    reason="no candidate target was supplied or discovered",
                )
                continue
            arguments = _arguments_for_step(step.tool, step.modules, current, self.page_size)
            if (
                step.tool is SemanticTool.GET_TARGET_PROFILES
                and "evidence.europepmc" in step.modules
            ):
                # These PMIDs belong to this query's completed locus retrieval,
                # never a provider cache or a model-generated bibliography.
                arguments["publication_ids"] = list(_retrieved_study_pmids(records))
            if step.tool is SemanticTool.GET_LOCUS_GENE_EVIDENCE and molecular_context is not None:
                self._retrieval_phase_started("discovery", current, context=molecular_context)
            emit_progress(
                self.progress_sink,
                "semantic_tool_started",
                progress_scope="workflow_step",
                step=index,
                total_steps=len(plan.steps),
                tool=step.tool.value,
                modules=list(step.modules),
                required=step.required,
            )
            batches = (
                variant_batches(current, size=parameter("execution.max_variants_per_query"))
                if current.analysis_variants and step.tool in {
                    SemanticTool.GET_LOCUS_CONTEXT, SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                } else ()
            )
            step_calls = []
            if batches:
                batch_audit = []
                for batch_index, identifiers in enumerate(batches, start=1):
                    batch_arguments = {**arguments, "variant_ids": list(identifiers)}
                    if step.tool is SemanticTool.GET_LOCUS_CONTEXT:
                        region = grch38_region_from_variant_ids(identifiers)
                        batch_arguments["region"] = region.to_dict() if region is not None else None
                    elif batch_index > 1:
                        batch_arguments["allow_region_fallback"] = False
                    details = {"batch_index": batch_index, "batch_count": len(batches), "step": index,
                               "tool": step.tool.value,
                               "variant_count": len(identifiers),
                               "total_analysis_variants": len(analysis_variant_ids(current))}
                    emit_progress(self.progress_sink, "semantic_variant_batch_started", progress_scope="workflow_step", **details)
                    response = await self._call_with_progress(step.tool, batch_arguments,
                        **{key: value for key, value in details.items() if key not in {"tool", "progress_scope"}})
                    step_calls.append(response)
                    batch_audit.append({**details, "tool": step.tool.value,
                                        "variant_ids": list(identifiers), "status": response.status.value,
                                        "coverage": response.coverage.to_dict(), "input_sha256": response.input_sha256,
                                        "response_sha256": sha256_json(response.to_dict())})
                    emit_progress(self.progress_sink, "semantic_variant_batch_finished", progress_scope="workflow_step", **details,
                                  status=response.status.value, records=len(response.records))
                    if response.status in {ToolStatus.BLOCKED, ToolStatus.INVALID_INPUT}:
                        break
                plan = replace(plan, max_calls=plan.max_calls + len(step_calls) - 1)
                current = replace(current, ld_expansion={**current.ld_expansion,
                    "evidence_batches": [*current.ld_expansion.get("evidence_batches", []), *batch_audit]})
                envelope = merge_variant_envelopes(tuple(step_calls))
                envelope = replace(envelope, records=annotate_analysis_origin(envelope.records, current))
                if step.tool is SemanticTool.GET_LOCUS_CONTEXT and any(
                    _coverage_gap_modules(call) or call.next_cursor or call.status not in {ToolStatus.SUCCESS, ToolStatus.NO_DATA}
                    for call in step_calls[1:]
                ):
                    current = replace(current, ld_expansion={**current.ld_expansion,
                                                            "proxy_evidence_complete": False})
            else:
                envelope = await self._call_with_progress(step.tool, arguments, step=index)
                step_calls.append(envelope)
            if step.tool is SemanticTool.GET_LOCUS_CONTEXT:
                normalized, envelope, extra_calls, proofs, conflicts = await self._normalize_source_context(
                    current, envelope, step=step, step_index=index)
                if source_normalizations is not None:
                    source_normalizations.extend(proofs)
                step_calls.extend(extra_calls)
                if extra_calls:
                    plan = replace(plan, max_calls=plan.max_calls + len(extra_calls))
                if conflicts:
                    operation_issues.append({"code": "normalization_conflict",
                        "state": "STOP" if normalized is None else "PARTIAL",
                        "reason": "canonical_sources_have_conflicting_gwas_observations",
                        "source_variant_ids": conflicts,
                        "valid_source_count": 0 if normalized is None else len(normalized.variants)})
                if normalized is None:
                    return SemanticRetrievalResult(RetrievalStatus.INVALID_INPUT, current, plan,
                        (*calls, *step_calls), tuple(records),
                        ("Source normalization found conflicting GWAS observations for every source identity.",),
                        tuple(warnings), operation_issues=tuple(operation_issues))
                current = normalized
            module_counts = _record_module_counts(envelope.records)
            emit_progress(
                self.progress_sink,
                "semantic_tool_finished",
                progress_scope="workflow_step",
                step=index,
                total_steps=len(plan.steps),
                tool=step.tool.value,
                status=envelope.status.value,
                coverage=envelope.coverage.to_dict(),
                warnings=list(envelope.warnings),
                records=len(envelope.records),
                module_counts=module_counts,
                unclassified_records=(len(envelope.records) - sum(module_counts.values())),
            )
            calls.extend(step_calls)
            warnings.extend(envelope.warnings)
            if (step.tool is SemanticTool.GET_LOCUS_GENE_EVIDENCE and molecular_context is not None
                    and envelope.status in {ToolStatus.BLOCKED, ToolStatus.ERROR,
                                            ToolStatus.INVALID_INPUT, ToolStatus.UNAVAILABLE}):
                self._retrieval_phase("discovery", retrieval_state(envelope), current,
                    records=len(envelope.records), context=molecular_context)
            if step.tool is SemanticTool.GET_LOCUS_CONTEXT and molecular_context is not None:
                valid_query, excluded = _exclude_confirmed_source_conflicts(current, envelope.records)
                if excluded:
                    issue = {"code": "S1" if valid_query is None else "source_variants_excluded",
                             "state": "STOP" if valid_query is None else "PARTIAL",
                             "confirmed_invalid_sources": excluded,
                             "valid_source_count": 0 if valid_query is None else len(valid_query.variants),
                             "reason": "explicit_reference_identity_conflict"}
                    operation_issues.append(issue)
                    if valid_query is None:
                        return SemanticRetrievalResult(RetrievalStatus.INVALID_INPUT, current, plan,
                            tuple(calls), tuple(records), ("S1: all supplied sources have a confirmed reference identity conflict",),
                            tuple(warnings), operation_issues=tuple(operation_issues))
                    current = valid_query
                    from ..providers.retrieval_recovery import ANALYSIS_VARIANTS
                    ANALYSIS_VARIANTS.set(tuple(analysis_variant_ids(current)))
                    valid_ids = set(analysis_variant_ids(current))
                    envelope = replace(envelope, records=tuple(row for row in envelope.records
                        if _record_uses_retained_analysis_variant(row, valid_ids)))
                    warnings.append("Confirmed conflicting source variants and their exclusive LD links were excluded; other sources remain available.")
            may_continue_variant_gap = _may_continue_after_variant_coverage_gap(
                current,
                step.tool,
                envelope,
            )
            if may_continue_variant_gap:
                warnings.append(
                    "Open Targets returned no exact locus.variant record; continuing "
                    "this non-benchmark gene query so the bounded coverage-gap "
                    "retrieval path can run."
                )
            if step.tool is SemanticTool.RESOLVE_ENTITIES:
                if envelope.status in {
                    ToolStatus.ERROR,
                    ToolStatus.BLOCKED,
                    ToolStatus.UNAVAILABLE,
                }:
                    return SemanticRetrievalResult(
                        {
                            ToolStatus.ERROR: RetrievalStatus.ERROR,
                            ToolStatus.BLOCKED: RetrievalStatus.BLOCKED,
                            ToolStatus.UNAVAILABLE: RetrievalStatus.INCOMPLETE,
                        }[envelope.status],
                        current,
                        plan,
                        tuple(calls),
                        tuple(records),
                        (
                            envelope.error.message
                            if envelope.error
                            else (f"entity resolver returned {envelope.status.value}"),
                        ),
                        tuple(warnings),
                    )
                resolution_details: dict[str, Any] = {}
                resolved, reason, resolution_warnings = _apply_resolution(
                    current,
                    envelope,
                    allow_partial_variant_resolution=(
                        self.service.provider in {"official_mcp", "official_graphql"}
                        and not current.benchmark
                    ),
                    resolution_details=resolution_details,
                )
                if resolved is None:
                    if resolution_details.get("resolution_complete") and not resolution_details.get("resolved_aliases"):
                        operation_issues.append({"code": "source_identity_unresolved", "state": "STOP",
                            "resolution_complete": True, "valid_source_count": 0,
                            "omitted_sources": resolution_details.get("omitted_variants", []),
                            "reason": "completed_resolution_without_unique_allele_identity"})
                    return SemanticRetrievalResult(
                        RetrievalStatus.INVALID_INPUT,
                        current,
                        plan,
                        tuple(calls),
                        tuple(records),
                        (reason or "entity resolution failed",),
                        tuple(warnings),
                        operation_issues=tuple(operation_issues),
                    )
                emit_progress(
                    self.progress_sink,
                    "entity_resolution_applied",
                    allow_partial_variant_resolution=(
                        self.service.provider in {"official_mcp", "official_graphql"}
                        and not current.benchmark
                    ),
                    source_variant_ids=[item.variant_id for item in current.variants],
                    omitted_variant_ids=[
                        item["variant_id"] for item in resolution_details["omitted_variants"]
                    ],
                    omitted_variants=resolution_details["omitted_variants"],
                    resolved_aliases=resolution_details["resolved_aliases"],
                    resolved_variant_ids=[item.variant_id for item in resolved.variants],
                    warnings=list(resolution_warnings),
                )
                current = resolved
                warnings.extend(resolution_warnings)
                continue
            if envelope.status is ToolStatus.BLOCKED:
                return SemanticRetrievalResult(
                    RetrievalStatus.BLOCKED,
                    current,
                    plan,
                    tuple(calls),
                    tuple(records),
                    (f"required tool was blocked: {step.tool.value}",),
                    tuple(warnings),
                    candidate_audit,
                )
            if envelope.status in {ToolStatus.ERROR, ToolStatus.INVALID_INPUT}:
                if not step.required and envelope.status is ToolStatus.ERROR:
                    warnings.append(
                        f"Optional tool {step.tool.value} failed for "
                        + ", ".join(envelope.coverage.failed or step.modules)
                        + ": "
                        + (envelope.error.message if envelope.error else "service unavailable")
                        + "; completed evidence and independent later retrieval were retained."
                    )
                    records.extend(envelope.records)
                    continue
                records.extend(envelope.records)
                return SemanticRetrievalResult(
                    RetrievalStatus.ERROR
                    if envelope.status is ToolStatus.ERROR
                    else RetrievalStatus.INVALID_INPUT,
                    current,
                    plan,
                    tuple(calls),
                    tuple(records),
                    (envelope.error.message if envelope.error else envelope.status.value,),
                    tuple(warnings),
                    candidate_audit,
                )
            if (
                step.required
                and envelope.status
                in {
                    ToolStatus.NO_DATA,
                    ToolStatus.UNAVAILABLE,
                }
                and not (
                    step.tool is SemanticTool.GET_LOCUS_GENE_EVIDENCE
                    and envelope.status is ToolStatus.NO_DATA
                )
            ):
                if may_continue_variant_gap and envelope.status is ToolStatus.NO_DATA:
                    continue
                return SemanticRetrievalResult(
                    RetrievalStatus.INCOMPLETE
                    if envelope.status is ToolStatus.UNAVAILABLE
                    else RetrievalStatus.INSUFFICIENT,
                    current,
                    plan,
                    tuple(calls),
                    tuple(records),
                    (f"required tool returned {envelope.status.value}: {step.tool.value}",),
                    tuple(warnings),
                    candidate_audit,
                )
            if step.tool is SemanticTool.GET_LOCUS_GENE_EVIDENCE:
                # The same pure policy drives the backend, public audit and replay.
                # Explicit input constrains the pool and receives the same policy.
                pool = tuple(
                    sorted(
                        set(current.candidate_genes)
                        if current.candidate_genes
                        else {gene for row in envelope.records for gene in target_ids(row)}
                    )
                )
                (
                    expression_records,
                    expression_calls,
                    expression_status,
                ) = await self._discovery_expression(
                    pool,
                    expression_context["selectors"],
                    ready,
                    requested_kinds,
                )
                expression_prefetched = bool(expression_calls)
                calls.extend(expression_calls)
                # Account explicitly for the observed-pool prefetch: <=200 targets
                # per call. It does not fill or truncate the biological shortlist.
                plan = replace(plan, max_calls=plan.max_calls + len(expression_calls))
                warnings.extend(w for call in expression_calls for w in call.warnings)
                if molecular_context is not None:
                    # The public timeline changes phases at the actual boundary.
                    # The terminal scientific audit below retains its existing
                    # shape; progress reports the observed pool, not its later shortlist.
                    self._retrieval_phase("discovery", retrieval_state(envelope), current,
                        records=len(envelope.records), context=molecular_context,
                        candidate_count=len(pool))
                    self._retrieval_phase_started("shortlist", current, context=molecular_context)
                candidate_audit = discover_candidates(
                    (*envelope.records, *expression_records),
                    variant_ids=analysis_variant_ids(current),
                    explicit_candidates=current.candidate_genes,
                    relevant_terms=context.relevant_terms,
                    expression_z_threshold=self.expression_z_threshold,
                    expression_fold_threshold=self.expression_fold_threshold,
                    expression_context_selectors=expression_selectors,
                    expression_context_groups=self.expression_context_groups,
                    expression_context_mapping=context.biosample_mapping,
                    expression_detection_rules=self.expression_detection_rules,
                    expression_retrieval=expression_status,
                    molecular_context=molecular_context,
                )
                locus_complete = (
                    envelope.status in {ToolStatus.SUCCESS, ToolStatus.NO_DATA}
                    and not _coverage_gap_modules(envelope)
                    and envelope.next_cursor is None
                )
                candidate_audit["locus_discovery_complete"] = locus_complete
                ld_complete = current.ld_expansion is None or current.ld_expansion.get("status") == "disabled" or current.ld_expansion.get("complete") is True
                proxy_evidence_complete = current.ld_expansion is None or current.ld_expansion.get("proxy_evidence_complete", True)
                if current.ld_expansion is not None:
                    candidate_audit["ld_expansion_complete"] = ld_complete
                    candidate_audit["ld_proxy_retrieval_complete"] = proxy_evidence_complete
                candidate_audit["candidate_discovery_complete"] = (
                    locus_complete
                    and ld_complete
                    and proxy_evidence_complete
                    and candidate_audit["expression_discovery_complete"]
                    and candidate_audit["evidence_integrity_complete"]
                    and candidate_audit["l2g_grouping_complete"]
                    and candidate_audit.get("molecular_context_complete", True)
                )
                candidate_audit["locus_discovery_coverage"] = envelope.coverage.to_dict()
                candidate_audit["locus_discovery_next_cursor"] = envelope.next_cursor
                candidate_audit["audit_fingerprint"] = sha256_json(
                    {
                        key: value
                        for key, value in candidate_audit.items()
                        if key != "audit_fingerprint"
                    }
                )
                if not candidate_audit["candidate_discovery_complete"]:
                    gaps = []
                    if not locus_complete:
                        gaps.append("locus gene evidence")
                    if not candidate_audit["expression_discovery_complete"]:
                        gaps.append("expression for the observed pool")
                    if not candidate_audit["evidence_integrity_complete"]:
                        gaps.append("gene/variant/build identity conflicts")
                    if not candidate_audit["l2g_grouping_complete"]:
                        gaps.append("L2G study-locus/release identity")
                    if not ld_complete:
                        gaps.append("population LD expansion")
                    if not proxy_evidence_complete:
                        gaps.append("LD proxy evidence retrieval")
                    if not candidate_audit.get("molecular_context_complete", True):
                        gaps.append("molecular tissue/cell context")
                    required_incomplete_reasons.append(
                        "shortlist discovery coverage is incomplete: "
                        + ", ".join(gaps)
                        + "; retained first-tier candidates are provisional"
                    )
                discovered = tuple(candidate_audit["candidate_ids"])
                empty_no_data_shortlist = not discovered and envelope.status is ToolStatus.NO_DATA
                phase = (
                    "candidate_universe_explicit"
                    if current.candidate_genes
                    else "candidate_universe_discovered"
                )
                current = replace(current, candidate_genes=discovered)
                records.extend(annotate_candidate_scope(expression_records, candidate_audit))
                unresolved = sum(
                    not gene["selected"] and gene["coordinate_status"] == "unavailable"
                    for gene in candidate_audit["genes"]
                )
                if not candidate_audit["expression_discovery_complete"]:
                    warnings.append(
                        "Expression data retrieval is complete for the observed pool, but some tissue/cell-type comparisons remain unresolved; the retained first-tier set is provisional. See candidate_audit."
                        if candidate_audit.get("expression_retrieval_complete") is True
                        else "Expression retrieval is incomplete for the full observed pool; the retained first-tier set is provisional. See candidate_audit."
                    )
                if unresolved:
                    warnings.append(
                        f"Candidate scope retained {unresolved} gene(s) as unresolved context; "
                        "missing or conflicting locus coordinates do not establish trans status "
                        "or absence of biological involvement. See the candidate audit."
                    )
                emit_progress(
                    self.progress_sink,
                    phase,
                    targets=list(discovered),
                    source_tool=step.tool.value,
                    **candidate_audit_summary(candidate_audit),
                    complete_audit_reference={
                        "field": "candidate_audit",
                        "audit_fingerprint": candidate_audit["audit_fingerprint"],
                        "status": "pending_retrieval_artifact",
                    },
                )
                envelope = replace(
                    envelope, records=annotate_candidate_scope(envelope.records, candidate_audit)
                )
                if len(discovered) > MAX_DISCOVERED_TARGETS:
                    return SemanticRetrievalResult(
                        RetrievalStatus.INCOMPLETE,
                        current,
                        plan,
                        tuple(calls),
                        (*records, *envelope.records),
                        (
                            "First-tier union exceeds the 200-target transport limit; reduce the input locus scope. No genes were silently removed.",
                        ),
                        tuple(warnings),
                        candidate_audit,
                    )
                if molecular_context is not None:
                    phase = self._retrieval_phase("discovery", retrieval_state(envelope), current,
                                                  records=len(envelope.records), context=molecular_context,
                                                  publish=False)
                    candidate_audit["retrieval_phases"] = [phase]
                    candidate_audit["retrieval_phases"].append(self._retrieval_phase(
                        "shortlist", RetrievalState.SUCCESS_DATA if discovered else RetrievalState.SUCCESS_NO_DATA,
                        current, records=len(discovered), context=molecular_context))
                    detail_calls, detail_records, detail_phase = await self._molecular_details(
                        current, candidate_audit, molecular_context, requested=deferred_coloc)
                    calls.extend(detail_calls)
                    records.extend(detail_records)
                    plan = replace(plan, max_calls=plan.max_calls + len(detail_calls))
                    if detail_calls:
                        detail_step = SemanticPlanStep(step.stage + 1, SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                            ("locus.colocalisation",), False,
                            "Retrieve matched and frozen supplementary molecular colocalisation after shortlist freeze.",
                            "frozen candidate genes and exact local molecular study-locus identities")
                        position = plan.steps.index(step) + 1
                        plan = replace(plan, steps=(*plan.steps[:position], detail_step, *plan.steps[position:]))
                    candidate_audit["retrieval_phases"].append(detail_phase)
                    candidate_audit["colocalisation_detail_complete"] = detail_phase["state"] not in {"PARTIAL", "FAILED"}
                    if not candidate_audit["colocalisation_detail_complete"]:
                        required_incomplete_reasons.append("shortlist colocalisation detail retrieval is incomplete")
                    candidate_audit["audit_fingerprint"] = sha256_json(
                        {key: value for key, value in candidate_audit.items() if key != "audit_fingerprint"})
            hard_coverage_gaps = _hard_required_coverage_gaps(
                step.tool,
                envelope,
            )
            if (hard_coverage_gaps and batches and step.tool is SemanticTool.GET_LOCUS_CONTEXT
                    and step_calls[0].status in {ToolStatus.SUCCESS, ToolStatus.NO_DATA}
                    and not _hard_required_coverage_gaps(step.tool, step_calls[0])):
                # Only proxy batches failed. Source identity keeps its original
                # gate; the separately verified LD identities admit a limited
                # evidence search, with incomplete candidate coverage recorded.
                hard_coverage_gaps = ()
            if step.required and hard_coverage_gaps:
                # Retain evidence from every successfully completed module even
                # when a scientifically required module was truncated or failed.
                # Continue independent later modules for auditability, but keep
                # the final retrieval status INCOMPLETE so no causal ranking is
                # silently produced from partial direct evidence.
                records.extend(envelope.records)
                warnings.append(
                    f"Retained {len(envelope.records)} record(s) from completed "
                    f"modules before the required {step.tool.value} step was "
                    "marked incomplete; independent later modules will still run."
                )
                required_incomplete_reasons.append(
                    "required module coverage was incomplete: " + ", ".join(hard_coverage_gaps)
                )
                continue
            if step.required and (
                envelope.status is ToolStatus.PARTIAL
                or envelope.coverage.failed
                or envelope.coverage.truncated
            ):
                affected = _coverage_gap_modules(envelope)
                warnings.append(
                    "Optional coverage was incomplete for "
                    + ", ".join(affected)
                    + "; completed modules and later independent retrieval were retained."
                )
            records.extend(envelope.records)
        if candidate_audit is not None and molecular_context is not None:
            records = list(annotate_molecular_ranking_scope(records, candidate_audit))
            candidate_audit["audit_fingerprint"] = sha256_json(
                {key: value for key, value in candidate_audit.items() if key != "audit_fingerprint"})
        return SemanticRetrievalResult(
            (
                RetrievalStatus.INCOMPLETE
                if required_incomplete_reasons
                else RetrievalStatus.INSUFFICIENT
                if empty_no_data_shortlist
                else RetrievalStatus.COMPLETE
            ),
            current,
            plan,
            tuple(calls),
            tuple(records),
            tuple(required_incomplete_reasons)
            or (
                (
                    "locus gene evidence returned no_data and no expression-qualified candidate was found",
                )
                if empty_no_data_shortlist
                else ()
            ),
            warnings=tuple(warnings),
            candidate_audit=candidate_audit,
            operation_issues=tuple(operation_issues),
        )

    def _retrieval_phase_started(self, name, query, *, context, **extra):
        """Publish activity only; RUNNING is never a scientific retrieval outcome."""
        emit_progress(self.progress_sink, "retrieval_phase", name=name, state="RUNNING", status="running",
            module="locus.colocalisation" if name == "colocalisation_details" else "locus.qtl",
            source_count=len(query.variants),
            proxy_count=len(analysis_variant_ids(query)) - len(query.variants),
            context_sha256=context["context_sha256"], **extra)

    def _retrieval_phase(self, name, state, query, *, records, context, publish=True, **extra):
        value = {"name": name, "state": str(state),
                 "status": "incomplete" if state in {RetrievalState.PARTIAL, RetrievalState.FAILED} else "completed",
                 "module": "locus.colocalisation" if name == "colocalisation_details" else "locus.qtl",
                 "records": records, "source_count": len(query.variants),
                 "proxy_count": len(analysis_variant_ids(query)) - len(query.variants),
                 "candidate_count": len(query.candidate_genes),
                 "context_sha256": context["context_sha256"], **extra}
        if publish:
            emit_progress(self.progress_sink, "retrieval_phase", **value)
        return value

    async def _molecular_details(self, query, audit, context, *, requested):
        """Query only frozen matched or supplementary local QTL identities.

        Full returned records stay in call receipts. Only target/related-trait
        views join ranking records; unrelated counterpart facts are not erased.
        """
        matched = audit.get("molecular_detail_scope", ())
        supplementary = audit.get("supplementary_molecular_scope", ())
        selected = (*matched, *supplementary)
        ids = tuple(sorted({row["study_locus_id"] for row in selected}))
        genes = tuple(sorted({row["gene_id"] for row in selected}))
        variants = tuple(sorted({row["variant_id"] for row in selected}))
        linked = {(row["gene_id"], row["study_locus_id"], row["variant_id"]) for row in selected}
        supplementary_pairs = {(row["gene_id"], row["study_locus_id"], row["variant_id"]) for row in supplementary}
        matched_assessments = {(gene["gene_id"], source["locus_id"], source["variant_id"]):
            frozen_context_assessment(source.get("molecular_context", {}), context["context_sha256"])
            for gene in audit.get("genes", ()) if gene.get("selected")
            for source in gene.get("sources", ()) if source.get("module") == "locus.qtl" and source.get("tier") == "direct"}
        scope = MolecularRetrievalScope("details", context, genes, ids, tuple(query.trait_ids), (query.trait,),
            qualified_pairs=tuple(sorted(linked)), supplementary_pairs=tuple(sorted(supplementary_pairs)))
        scope_hash = sha256_json(scope.to_dict())
        if not requested or not ids or not context.get("defined"):
            state = RetrievalState.NOT_REQUESTED if not requested else RetrievalState.NOT_APPLICABLE
            return (), (), self._retrieval_phase("colocalisation_details", state, query, records=0,
                context=context, scope_sha256=scope_hash,
                reason="not_requested" if not requested else "no_qualified_local_molqtl")
        self._retrieval_phase_started("colocalisation_details", query, context=context,
            scope_sha256=scope_hash)
        token = MOLECULAR_RETRIEVAL_SCOPE.set(scope)
        calls = []
        try:
            size = parameter("execution.max_variants_per_query")
            for offset in range(0, len(variants), size):
                arguments = {"variant_ids": list(variants[offset:offset + size]),
                             "study_locus_ids": list(ids), "candidate_gene_ids": list(genes),
                             "only_candidate_genes": True, "modules": ["locus.colocalisation"],
                             "relevant_biosample_ids": list(query.relevant_tissues),
                             "qtl_types": list(query.qtl_types), "allow_region_fallback": False,
                             "page_size": self.page_size, "cursor": None}
                calls.append(await self._call_with_progress(SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                    arguments, retrieval_stage="colocalisation_details"))
        finally:
            MOLECULAR_RETRIEVAL_SCOPE.reset(token)
        envelope = merge_variant_envelopes(tuple(calls))
        state = retrieval_state(envelope)
        kept, excluded, invalid = [], Counter(), 0
        from ..expression_policy import context_matches
        for record in annotate_analysis_origin(envelope.records, query):
            raw = ((record.get("upstream") or {}).get("raw") or {})
            record_context = dict(record.get("context") or {})
            subject = record.get("subject") or {}
            gene_ids = target_ids(record)
            if (len(gene_ids) != 1 or not record_target_identity_valid(record)
                    or not molecular_detail_anchor_verified(record, discovery_locus_id(record))
                    or (gene_ids[0], discovery_locus_id(record), subject.get("id")) not in linked):
                invalid += 1
                continue
            relation = record_context.get("target_trait_relation")
            label = raw.get("rightTraitFromSource")
            observed_relation = ("unresolved" if not isinstance(label, str) or not label.strip()
                else "exact_label" if " ".join(label.casefold().split()) == " ".join(query.trait.casefold().split())
                else "related_label" if context_matches(query.trait, label) else "other_trait")
            if relation is not None and relation != observed_relation:
                invalid += 1
                excluded["contradictory_target_trait_relation"] += 1
                continue
            relation = observed_relation
            record_context.update(
                target_trait_relation=relation,
                phenotype_evidence_scope=(
                    "target_trait" if relation == "exact_label" else
                    "related_trait" if relation == "related_label" else
                    "other_trait" if relation == "other_trait" else "unresolved"
                ),
                rank_scope="target_trait" if relation == "exact_label" else
                "related_trait" if relation == "related_label" else
                "other_trait" if relation == "other_trait" else "unresolved",
                molecular_detail_record=True,
                context_sha256=context["context_sha256"])
            if relation == "unresolved":
                excluded[relation] += 1
                continue
            pair = (gene_ids[0], discovery_locus_id(record), subject.get("id"))
            other_context = pair in supplementary_pairs
            record_context["context_match_class"] = "other_reported_context" if other_context else "matched"
            record_context["molecular_evidence_tier"] = "supplementary" if other_context else "context_related"
            if pair in matched_assessments:
                record_context["molecular_context_assessment"] = matched_assessments[pair]
            if other_context:
                selection = audit.get("supplementary_molecular_selection", {})
                record_context.update(other_context_supplement=True, molecular_selection="supplementary_biosample",
                    selection_receipt_sha256=selection.get("receipt_sha256"))
                entry = next((row for row in selection.get("selected", ()) if
                    (row["gene_id"], row["study_locus_id"], row["variant_id"]) == pair), None)
                if entry is not None:
                    record_context["molecular_context_assessment"] = entry.get("context_assessment")
                    record_context["supplementary_selection"] = {key: entry[key] for key in
                        ("biosample", "qtl_type", "member_p", "selection_basis")}
            role = {"scope": "local", "reason": "supplementary_noncontext_molqtl_detail" if other_context
                    else "context_matched_molqtl_detail",
                    "direct_locus_support": True, "context_only": False}
            kept.append({**record, "context": record_context,
                         "derived": {**(record.get("derived") or {}), "candidate_scope": role}})
        if invalid:
            state = RetrievalState.PARTIAL if envelope.records else RetrievalState.FAILED
        audit["colocalisation_detail_audit"] = {
            "scope": scope.to_dict(), "scope_sha256": scope_hash,
            "returned_records": len(envelope.records), "ranking_view_records": len(kept),
            "context_only_records_retained_in_calls": dict(sorted(excluded.items())),
            "invalid_scope_records": invalid,
            "raw_call_ids": [call.call_id for call in calls],
            "raw_response_sha256": [sha256_json(call.to_dict()) for call in calls],
            "supplementary_pairs": [list(pair) for pair in sorted(supplementary_pairs)],
            "selection_basis": "frozen_shortlist_exact_molqtl_pairs_then_counterpart_trait_label"}
        return tuple(calls), tuple(kept), self._retrieval_phase("colocalisation_details", state, query,
            records=len(envelope.records), context=context, scope_sha256=scope_hash,
            ranking_records=len(kept), context_only_records=sum(excluded.values()), invalid_records=invalid)

    async def _discovery_expression(self, pool, context_terms, ready, requested_kinds):
        """Retrieve unprojected expression for every observed target before selection."""
        if not context_terms:
            return (), (), {gene: "context_undefined" for gene in pool}
        if "target.expression" not in ready:
            return (), (), {gene: "unavailable" for gene in pool}
        if requested_kinds is not None and EvidenceKind.EXPRESSION not in requested_kinds:
            return (), (), {gene: "not_requested" for gene in pool}
        records, calls, status = [], [], {}
        for offset in range(0, len(pool), self.expression_batch_size):
            genes = pool[offset : offset + self.expression_batch_size]
            emit_progress(
                self.progress_sink,
                "candidate_expression_started",
                targets=list(genes),
                tool=SemanticTool.GET_TARGET_PROFILES.value,
                modules=["target.expression"],
                completed_count=offset, total_count=len(pool),
            )
            call = await self._call_with_progress(
                SemanticTool.GET_TARGET_PROFILES,
                {
                    "target_ids": list(genes),
                    "modules": ["target.expression"],
                    "page_size": max(self.page_size, len(genes)),
                    "cursor": None,
                }, completed_count=offset, total_count=len(pool),
            )
            calls.append(call)
            rows = tuple(r for r in call.records if r.get("module") == "target.expression")
            valid_rows = tuple(r for r in rows if _expression_record_matches_request(r, genes))
            identity_complete = len(valid_rows) == len(call.records)
            if not identity_complete:
                call = replace(
                    call,
                    warnings=(
                        *call.warnings,
                        (
                            "Expression response contained an unexpected or inconsistent target identity; "
                            "the batch cannot establish complete expression discovery."
                        ),
                    ),
                )
                calls[-1] = call
            rows = valid_rows
            records.extend(rows)
            returned = {g for r in rows for g in target_ids(r)}
            profiles_complete = all(_expression_record_coverage_complete(r) for r in rows)
            complete = (
                call.status in {ToolStatus.SUCCESS, ToolStatus.NO_DATA}
                and not _coverage_gap_modules(call)
                and "target.expression" in (*call.coverage.completed, *call.coverage.no_data)
                and call.next_cursor is None
                and identity_complete
                and profiles_complete
                and (call.status is not ToolStatus.NO_DATA or not rows)
            )
            for gene in genes:
                gene_rows = [row for row in rows if gene in target_ids(row)]
                # A failed neighbour cannot erase an independently validated full
                # expression panel. A missing gene still needs whole-call coverage.
                explicit_complete = bool(gene_rows) and all(
                    _expression_record_has_complete_native_coverage(row)
                    for row in gene_rows
                )
                gene_complete = identity_complete and (complete or explicit_complete)
                status[gene] = ("returned" if gene in returned else "no_data") if gene_complete else "incomplete"
            emit_progress(
                self.progress_sink,
                "candidate_expression_finished",
                targets=list(genes),
                tool=SemanticTool.GET_TARGET_PROFILES.value,
                modules=["target.expression"],
                status=call.status.value,
                coverage=call.coverage.to_dict(),
                records=len(rows),
                module_counts={"target.expression": len(rows)},
                complete=complete,
                completed_count=offset + len(genes), total_count=len(pool),
                returned_count=sum(value == "returned" for value in status.values()),
                incomplete_count=sum(value == "incomplete" for value in status.values()),
            )
        return tuple(records), tuple(calls), status


def _expression_record_matches_request(record, genes) -> bool:
    """Bind normalized and native gene identities before assigning batch coverage."""
    identifiers = set(target_ids(record))
    return (
        record_target_identity_valid(record)
        and len(identifiers) == 1
        and next(iter(identifiers)) in genes
    )


def _expression_record_coverage_complete(record) -> bool:
    upstream = record.get("upstream")
    raw = upstream.get("raw") if isinstance(upstream, Mapping) else None
    if not isinstance(raw, Mapping) or "expressionCoverage" not in raw:
        # Frozen adapters may provide complete profiles without native paging
        # metadata. Their successful module contract supplies retrieval coverage.
        return True
    coverage = raw["expressionCoverage"]
    rows = raw.get("tissues")
    if not isinstance(coverage, Mapping) or coverage.get("complete") is not True:
        return False
    if not isinstance(rows, (list, tuple)):
        return False
    return all(
        isinstance(coverage[field], int)
        and not isinstance(coverage[field], bool)
        and coverage[field] == len(rows)
        for field in ("declared_count", "retrieved_count")
        if field in coverage
    )


def _expression_record_has_complete_native_coverage(record) -> bool:
    upstream = record.get("upstream")
    raw = upstream.get("raw") if isinstance(upstream, Mapping) else None
    return (isinstance(raw, Mapping)
            and isinstance(raw.get("expressionCoverage"), Mapping)
            and _expression_record_coverage_complete(record))


_DIRECT_LOCUS_GENE_MODULES = frozenset(
    {
        "locus.variant_consequence",
        "locus.qtl",
        "locus.colocalisation",
        "locus.enhancer_to_gene",
    }
)


def _coverage_gap_modules(envelope: ToolEnvelope) -> tuple[str, ...]:
    """Return deterministic module-level coverage gaps for one tool call."""

    return tuple(
        dict.fromkeys(
            (
                *envelope.coverage.unavailable,
                *envelope.coverage.blocked,
                *envelope.coverage.failed,
                *envelope.coverage.truncated,
            )
        )
    )


def _hard_required_coverage_gaps(
    tool: SemanticTool,
    envelope: ToolEnvelope,
) -> tuple[str, ...]:
    """Separate scientific hard gaps from optional modules sharing a tool call."""

    affected = _coverage_gap_modules(envelope)
    if tool is SemanticTool.GET_LOCUS_CONTEXT:
        required = frozenset({"locus.variant"})
    elif tool is SemanticTool.GET_LOCUS_GENE_EVIDENCE:
        # A failure in one independent evidence channel must not suppress a
        # defensible ranking when another channel returned an explicit
        # locus-to-gene link.  The incomplete channel is retained as a warning
        # and must be disclosed in the final limitations.  If no molecular
        # channel completed with evidence, keep the hard stop because absence
        # cannot be distinguished from failed retrieval.
        if _has_molecular_direct_record(envelope.records):
            return ()
        required = _DIRECT_LOCUS_GENE_MODULES
    else:
        required = frozenset(envelope.coverage.requested)
    return tuple(module for module in affected if module in required)


def _has_molecular_direct_record(records: tuple[Mapping[str, Any], ...]) -> bool:
    return any(
        evidence_candidate_scope(record).get("direct_locus_support", True)
        and str(record.get("module") or "").strip().casefold()
        in {
            "locus.qtl",
            "locus.colocalisation",
            "locus.enhancer_to_gene",
        }
        for record in records
    )


def _record_module_counts(
    records: tuple[Mapping[str, Any], ...],
) -> dict[str, int]:
    """Count only module names carried by the records actually returned.

    Planned modules and coverage metadata are deliberately not used here: a
    public retrieval count must describe real rows, including zero-row module
    outcomes, rather than the data the planner hoped to retrieve.
    """

    counts: Counter[str] = Counter()
    for record in records:
        module = record.get("module")
        if not isinstance(module, str):
            continue
        normalized = module.strip()
        if normalized:
            counts[normalized] += 1
    return dict(sorted(counts.items()))


def _arguments_for_step(
    tool: SemanticTool,
    modules: tuple[str, ...],
    query: LocusQuery,
    page_size: int,
) -> dict[str, Any]:
    variant_ids = [item.variant_id for item in query.variants]
    targets = list(query.candidate_genes)
    region = grch38_region_from_variant_ids(variant_ids)
    if tool is SemanticTool.RESOLVE_ENTITIES:
        values = [
            *(
                {"text": item, "types": ["variant"]}
                for item in variant_ids
                if not _VARIANT_ID.fullmatch(item)
            ),
            *(
                {"text": item, "types": ["target"]}
                for item in targets
                if not _TARGET_ID.fullmatch(item)
            ),
        ]
        return {"queries": values, "max_hits": 5}
    if tool is SemanticTool.GET_LOCUS_CONTEXT:
        return {
            "variant_ids": variant_ids,
            "study_ids": list(query.study_ids),
            "region": region.to_dict() if region is not None else None,
            "modules": list(modules),
            "qtl_types": list(query.qtl_types),
            "include_credible_set_members": True,
            "page_size": page_size,
            "cursor": None,
        }
    if tool is SemanticTool.GET_LOCUS_GENE_EVIDENCE:
        return {
            "variant_ids": variant_ids,
            "study_locus_ids": [],
            "candidate_gene_ids": targets,
            "modules": list(modules),
            "relevant_biosample_ids": list(query.relevant_tissues),
            "qtl_types": list(query.qtl_types),
            "only_candidate_genes": bool(targets),
            "allow_region_fallback": (query.task in {"gene_associations", "prioritize_genes"}),
            "page_size": page_size,
            "cursor": None,
        }
    if tool is SemanticTool.GET_TARGET_PROFILES:
        return {
            "target_ids": targets,
            "trait_ids": list(query.trait_ids),
            "modules": list(modules),
            "relevant_biosample_ids": list(query.relevant_tissues),
            "page_size": page_size,
            "cursor": None,
        }
    if tool is SemanticTool.GET_DISEASE_PROFILES:
        return {
            "disease_ids": list(query.trait_ids),
            "modules": list(modules),
            "page_size": page_size,
            "cursor": None,
        }
    if tool is SemanticTool.GET_TARGET_DISEASE_EVIDENCE:
        return {
            "target_ids": targets,
            "disease_ids": list(query.trait_ids),
            "association_scope": "direct",
            "modules": list(modules),
            "datatype_ids": [],
            "datasource_ids": [],
            "min_score": 0,
            "page_size": page_size,
            "cursor": None,
        }
    if tool is SemanticTool.GET_DRUG_EVIDENCE:
        return {
            "drug_ids": [],
            "target_ids": targets,
            "disease_ids": list(query.trait_ids),
            "modules": list(modules),
            "page_size": page_size,
            "cursor": None,
        }
    raise ValueError(f"unsupported planned tool: {tool.value}")


def _retrieved_study_pmids(records: list[Mapping[str, Any]]) -> tuple[str, ...]:
    """Prioritize metadata for explicit study citations in the current locus ledger."""

    module_priority = {
        "locus.qtl": 0,
        "locus.colocalisation": 1,
        "locus.enhancer_to_gene": 2,
        "locus.study": 3,
        "locus.credible_set": 4,
    }
    ordered = sorted(
        (record for record in records if record.get("module") in module_priority),
        key=lambda record: module_priority[str(record["module"])],
    )
    pmids: dict[str, None] = {}
    for record in ordered:
        if record.get("module") == "locus.colocalisation":
            for publication in parse_colocalization_publication_lineage(record).publications():
                pmids.setdefault(publication.pmid, None)
        context = record.get("context")
        context = context if isinstance(context, Mapping) else {}
        upstream = record.get("upstream")
        upstream = upstream if isinstance(upstream, Mapping) else {}
        raw = upstream.get("raw")
        raw = raw if isinstance(raw, Mapping) else {}
        for value in (
            context.get("pmid"),
            record.get("pmid"),
            raw.get("pubmedId"),
            raw.get("pmid"),
        ):
            pmid = normalize_pmid(value)
            if pmid:
                pmids.setdefault(pmid, None)
        if len(pmids) >= 100:
            break
    return tuple(pmids)[:100]


def _may_continue_after_variant_coverage_gap(
    query: LocusQuery,
    tool: SemanticTool,
    envelope: ToolEnvelope,
) -> bool:
    return (
        not query.benchmark
        and query.task in {"gene_associations", "prioritize_genes"}
        and tool is SemanticTool.GET_LOCUS_CONTEXT
        and "locus.variant" in envelope.coverage.no_data
    )


def _discovered_target_ids(
    records: tuple[Mapping[str, Any], ...],
) -> tuple[str, ...]:
    """Compatibility wrapper around the shared, auditable discovery policy."""
    return tuple(discover_candidates(records)["candidate_ids"])


def _discovery_locus_id(record: Mapping[str, Any]) -> str:
    return discovery_locus_id(record)


def _candidate_discovery_summary(
    records: tuple[Mapping[str, Any], ...],
    selected: tuple[str, ...],
) -> dict[str, Any]:
    audit = discover_candidates(records)
    if tuple(audit["candidate_ids"]) != selected:
        raise ValueError("candidate summary must describe the actual discovery set")
    return audit


def _is_direct_candidate_record(record: Mapping[str, Any], module: str) -> bool:
    audit = discover_candidates((record,))
    return any(source["tier"] == "direct" for gene in audit["genes"] for source in gene["sources"])


def _record_uses_retained_analysis_variant(record, retained_ids):
    subject = record.get("subject") or {}
    if subject.get("type") == "variant" and subject.get("id") not in retained_ids:
        return False
    raw = (record.get("upstream") or {}).get("raw") or {}
    member = raw.get("matchedLocusMember") or {}
    # Fine-mapping has a credible-set subject but still belongs to one exact
    # queried member. An external lead alone does not identify that member.
    for value in (raw.get("queryVariantId"), member.get("variantId")):
        if value is not None and value not in retained_ids:
            return False
    return True


def _apply_source_normalizations(query, proofs):
    """Canonicalise source identities without changing their GWAS observations."""
    from ..analysis_scope import resolved_ld_scope
    from ..source_normalization import validate_source_normalization

    mapping = {row["original_id"]: row["canonical_id"] for row in proofs
               if validate_source_normalization(row)}
    grouped = {}
    for row in query.variants:
        canonical = mapping.get(row.variant_id, row.variant_id)
        grouped.setdefault(canonical, []).append(row)
    conflicts, variants = [], []
    for canonical, rows in grouped.items():
        resolved = [replace(row, variant_id=canonical) for row in rows]
        if any(row != resolved[0] for row in resolved[1:]):
            conflicts.extend(row.variant_id for row in rows)
        else:
            variants.append(resolved[0])
    if not variants:
        return None, sorted(conflicts)
    plan = query.ld_expansion or {}
    aliases = dict(plan.get("resolved_source_aliases", {}))
    originals = [row.get("variant_id") for row in plan.get("source_observations", ())]
    for identifier in originals or [row.variant_id for row in query.variants]:
        current = aliases.get(identifier, identifier)
        if current in mapping:
            aliases[identifier] = mapping[current]
    omitted = sorted(set(plan.get("omitted_source_variant_ids", ())) | {
        identifier for identifier in originals if plan.get("resolved_source_aliases", {}).get(identifier, identifier) in conflicts})
    scope = resolved_ld_scope(query, tuple(variants), aliases, omitted)
    if scope.get("ld_expansion") is not None:
        previous = plan.get("source_identity_normalizations", ())
        scope["ld_expansion"]["source_identity_normalizations"] = list({
            row["original_id"]: dict(row) for row in (*previous, *proofs)
            if validate_source_normalization(row)}.values())
    result = replace(query, variants=tuple(variants), **scope)
    result.validate()
    return result, sorted(conflicts)


def _exclude_confirmed_source_conflicts(query, records):
    """A missing OT row is unknown. Only explicit source identity contradictions exclude."""
    sources = {row.variant_id for row in query.variants}
    excluded = {}
    for row in records:
        if row.get("module") != "locus.variant":
            continue
        identifier = (row.get("subject") or {}).get("id")
        if identifier not in sources or not _VARIANT_ID.fullmatch(identifier):
            continue
        chrom, position, ref, alt = identifier.replace(":", "_").split("_")
        chrom = chrom.upper().removeprefix("CHR")
        raw = (row.get("upstream") or {}).get("raw") or {}
        context = row.get("context") or {}
        reasons = set()
        for data in (raw, context):
            for key, expected in (("referenceAllele", ref), ("reference_allele", ref),
                                  ("alternateAllele", alt), ("alternate_allele", alt)):
                value = data.get(key)
                if isinstance(value, str) and re.fullmatch("[ACGTN]+", value, re.IGNORECASE) and value.upper() != expected.upper():
                    reasons.add("allele_conflict")
            observed = data.get("chromosome")
            if isinstance(observed, (str, int)) and not isinstance(observed, bool) and str(observed).upper().removeprefix("CHR") != chrom:
                reasons.add("chromosome_conflict")
            observed = data.get("position")
            if type(observed) is int and observed > 0 and observed != int(position):
                reasons.add("position_conflict")
            if data.get("genome_build") in {"GRCh37", "hg19"}:
                reasons.add("genome_build_conflict")
        if reasons:
            excluded[identifier] = {"variant_id": identifier, "reasons": sorted(reasons)}
    if not excluded:
        return query, []
    retained = tuple(row for row in query.variants if row.variant_id not in excluded)
    if not retained:
        return None, [excluded[key] for key in sorted(excluded)]
    from ..analysis_scope import resolved_ld_scope
    plan = query.ld_expansion or {}
    aliases = dict(plan.get("resolved_source_aliases", {}))
    original_ids = [row.get("variant_id") for row in plan.get("source_observations", ())] or list(sources)
    omitted = sorted(set(plan.get("omitted_source_variant_ids", ())) | {
        identifier for identifier in original_ids if aliases.get(identifier, identifier) in excluded})
    resolved_scope = resolved_ld_scope(query, retained, aliases, omitted)
    if resolved_scope.get("ld_expansion") is not None:
        combined = [*plan.get("excluded_ld_links", ()), *resolved_scope["ld_expansion"].get("excluded_ld_links", ())]
        resolved_scope["ld_expansion"]["excluded_ld_links"] = list({sha256_json(row): row for row in combined}.values())
    result = replace(query, variants=retained, **resolved_scope)
    result.validate()
    return result, [excluded[key] for key in sorted(excluded)]


def _apply_resolution(
    query: LocusQuery,
    envelope: ToolEnvelope,
    *,
    allow_partial_variant_resolution: bool = False,
    resolution_details: dict[str, Any] | None = None,
) -> tuple[LocusQuery | None, str | None, tuple[str, ...]]:
    if envelope.status not in {ToolStatus.SUCCESS, ToolStatus.PARTIAL}:
        return None, f"entity resolver returned {envelope.status.value}", ()

    variant_sources = {
        item.variant_id for item in query.variants if not _VARIANT_ID.fullmatch(item.variant_id)
    }
    target_sources = {item for item in query.candidate_genes if not _TARGET_ID.fullmatch(item)}
    expected_sources = variant_sources | target_sources
    records_by_source: dict[str, Mapping[str, Any]] = {}
    for record in envelope.records:
        text = str(record.get("query", "")).strip()
        if not text or text not in expected_sources:
            return None, f"entity resolver returned an unexpected query {text!r}", ()
        if text in records_by_source:
            return None, f"entity resolver returned duplicate results for {text!r}", ()
        if text in variant_sources and text in target_sources:
            return None, f"entity query type is ambiguous for {text!r}", ()
        records_by_source[text] = record

    omitted_variants: list[str] = []
    ambiguous_variants: list[str] = []
    omissions: list[dict[str, str]] = []
    resolved_variants = {
        item.variant_id: item.variant_id
        for item in query.variants
        if item.variant_id not in variant_sources
    }
    resolved_targets = {item: item for item in query.candidate_genes if item not in target_sources}
    for text in (
        *[item.variant_id for item in query.variants if item.variant_id in variant_sources],
        *[item for item in query.candidate_genes if item in target_sources],
    ):
        record = records_by_source.get(text)
        if record is None:
            return None, f"entity resolver omitted requested query {text!r}", ()
        hits = record.get("hits")
        if not isinstance(hits, list):
            return None, f"malformed entity hits for {text!r}", ()
        expected_type = "variant" if text in variant_sources else "target"
        if not hits:
            if expected_type == "variant":
                omitted_variants.append(text)
                omissions.append({"variant_id": text, "reason": "no_match"})
                continue
            return None, f"no canonical entity match for explicit target {text!r}", ()
        may_omit_ambiguous = (
            allow_partial_variant_resolution
            and not query.benchmark
            and expected_type == "variant"
            and re.fullmatch(r"rs[1-9]\d*", text, re.IGNORECASE)
        )
        if any(
            not isinstance(hit, Mapping)
            or hit.get("type") != expected_type
            or not (_VARIANT_ID if expected_type == "variant" else _TARGET_ID).fullmatch(
                str(hit.get("id") or "")
            )
            for hit in hits
        ):
            return None, f"malformed or wrong-type entity hits for {text!r}", ()
        coordinate_hit = (
            _coordinate_resolved_hit(query, text, record) if may_omit_ambiguous else None
        )
        if coordinate_hit is None and (record.get("auto_selectable") is not True or len(hits) != 1):
            if may_omit_ambiguous:
                ambiguous_variants.append(text)
                omissions.append({"variant_id": text, "reason": "ambiguous_search_candidates"})
                continue
            return None, f"ambiguous or unvalidated entity mapping for {text!r}", ()
        hit = coordinate_hit if coordinate_hit is not None else hits[0]
        if not isinstance(hit, Mapping):
            return None, f"malformed entity hit for {text!r}", ()
        hit_type = str(hit.get("type", "")).casefold()
        identifier = str(hit.get("id", "")).strip()
        if hit_type != expected_type:
            return None, f"entity resolver returned the wrong type for {text!r}", ()
        canonical_pattern = _VARIANT_ID if expected_type == "variant" else _TARGET_ID
        if not identifier or not canonical_pattern.fullmatch(identifier):
            return None, f"entity resolver returned a non-canonical ID for {text!r}", ()
        if expected_type == "variant":
            observation = next(item for item in query.variants if item.variant_id == text)
            if may_omit_ambiguous and not _hit_matches_source_coordinate(observation, identifier):
                ambiguous_variants.append(text)
                omissions.append(
                    {"variant_id": text, "reason": "no_search_hit_at_source_coordinate"}
                )
                continue
            resolved_variants[text] = identifier
        else:
            resolved_targets[text] = identifier

    if not resolved_variants:
        if resolution_details is not None:
            resolution_details.update({"resolved_aliases": {}, "omitted_variants": omissions,
                "resolution_complete": envelope.status is ToolStatus.SUCCESS
                and not _coverage_gap_modules(envelope)})
        return None, "no supplied variant has a canonical entity match", ()

    variants = []
    canonical_variant_sources: dict[str, list[str]] = {}
    for item in query.variants:
        canonical = resolved_variants.get(item.variant_id)
        if canonical is None:
            continue
        canonical_variant_sources.setdefault(canonical, []).append(item.variant_id)
        if any(existing.variant_id == canonical for existing in variants):
            continue
        variants.append(replace(item, variant_id=canonical))

    genes: list[str] = []
    canonical_target_sources: dict[str, list[str]] = {}
    for item in query.candidate_genes:
        canonical = resolved_targets[item]
        canonical_target_sources.setdefault(canonical, []).append(item)
        if canonical not in genes:
            genes.append(canonical)

    warnings: list[str] = []
    if omitted_variants:
        warnings.append(
            "Entity resolution omitted variant(s) with no canonical match: "
            + ", ".join(repr(item) for item in omitted_variants)
        )
    if ambiguous_variants:
        warnings.append(
            "Entity resolution could not uniquely verify variant(s) against the supplied "
            "coordinates; external retrieval excludes these observations: "
            + ", ".join(repr(item) for item in ambiguous_variants)
            + ". This incomplete coverage does not establish that they are unreported."
        )
    for canonical, sources in canonical_variant_sources.items():
        if len(sources) > 1:
            warnings.append(
                "Entity resolution collapsed variant aliases "
                + ", ".join(repr(item) for item in sources)
                + f" onto {canonical!r}."
            )
    for canonical, sources in canonical_target_sources.items():
        if len(sources) > 1:
            warnings.append(
                "Entity resolution collapsed target aliases "
                + ", ".join(repr(item) for item in sources)
                + f" onto {canonical!r}."
            )
    from ..analysis_scope import resolved_ld_scope

    resolved = replace(query, variants=tuple(variants), candidate_genes=tuple(genes),
                       **resolved_ld_scope(query, tuple(variants), resolved_variants,
                                           [item["variant_id"] for item in omissions]))
    try:
        resolved.validate()
    except ValueError as exc:
        return None, f"resolved identifier conflicts with supplied locus: {exc}", ()
    if resolution_details is not None:
        resolution_details.update(
            {"resolved_aliases": resolved_variants, "omitted_variants": omissions}
        )
    return (
        resolved,
        None,
        tuple(warnings),
    )


def _coordinate_resolved_hit(query, text, record) -> Mapping[str, Any] | None:
    """Use a supplied GRCh38 coordinate to constrain complete search candidates.

    This never invents an allele, changes coordinates, or chooses among multiple
    alleles at one position. Actual returned locus.variant identity must still be
    checked by the provider and runtime bridge before evidence is reported.
    """
    if str(query.genome_build or "").casefold() not in {"grch38", "hg38"}:
        return None
    if record.get("truncated") is not False or record.get("status") != "candidate_matches":
        return None
    observation = next((item for item in query.variants if item.variant_id == text), None)
    if observation is None or observation.chromosome is None or observation.position is None:
        return None
    chromosome = str(observation.chromosome).upper().removeprefix("CHR")
    matched = {}
    for hit in record.get("hits", ()):
        if (
            not isinstance(hit, Mapping)
            or str(hit.get("type")).casefold() != "variant"
            or not _VARIANT_ID.fullmatch(str(hit.get("id") or ""))
        ):
            return None
        identifier = hit["id"]
        parts = identifier.split("_")
        if parts[0].upper() == chromosome and int(parts[1]) == observation.position:
            matched[identifier] = hit
    return next(iter(matched.values())) if len(matched) == 1 else None


def _hit_matches_source_coordinate(observation, identifier: str) -> bool:
    chromosome, position, *_ = identifier.split("_")
    return (
        observation.chromosome is None
        or str(observation.chromosome).upper().removeprefix("CHR") == chromosome.upper()
    ) and (observation.position is None or observation.position == int(position))
