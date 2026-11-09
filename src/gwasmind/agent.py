from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any
from uuid import uuid4

from . import __version__
from .artifacts import ArtifactStore
from .audit import AuditLogger, RunManifest, sha256_json
from .candidates import candidate_metadata_summaries, candidate_pool_rows
from .coverage import ot_mhc_coverage_limitation, query_overlaps_ot_mhc
from .gate_rules import load_gate_rules
from .llm import (
    EvidenceSynthesizer,
    ModelIdentity,
    OpenAICompatibleSynthesizer,
    SynthesisOutput,
    SynthesisOutputError,
)
from .models import (
    AgentResult,
    DecisionStatus,
    EvidenceKind,
    EvidenceRecord,
    LocusQuery,
    SufficiencyReport,
)
from .progress import ProgressSink, emit_progress
from .prompting import PROMPT_VERSIONS, load_prompt
from .ranking_claims import assert_ranking_unchanged, repair_ranking_assertions
from .sufficiency import EvidenceSufficiencyGate, limited_discovery_ranking_scope
from .tools import GWASToolGateway, ToolCallResult, ToolStatus
from .validation import unverified_synthesis_fields, validate_synthesis

_RETRIEVAL_ONLY_TASKS = frozenset({"describe_variants", "gene_associations"})
_RESOLVED_METADATA_MAX_ID_CHARS = 256
_RESOLVED_METADATA_MAX_VARIANTS = 100
_RESOLVED_METADATA_MAX_CANDIDATES = 200


@dataclass(frozen=True, slots=True)
class EvidencePlan:
    kinds: tuple[EvidenceKind, ...]
    rationale: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kinds": [kind.value for kind in self.kinds],
            "rationale": list(self.rationale),
        }


class EvidencePlanner:
    """Small deterministic planner: one agent, explicit omics selection."""

    def plan(self, query: LocusQuery, available: frozenset[EvidenceKind]) -> EvidencePlan:
        selected: set[EvidenceKind] = set()
        reasons: list[str] = []
        model_requested = {
            EvidenceKind(item)
            for item in query.requested_evidence_kinds
            if item in {kind.value for kind in available}
        }
        if query.benchmark:
            model_requested.discard(EvidenceKind.L2G)
        if model_requested:
            selected.update(model_requested)
            reasons.append("validated evidence dimensions selected by the bounded query planner")
        if query.task == "describe_variants":
            selected.update(
                {
                    EvidenceKind.VARIANT_ANNOTATION,
                    EvidenceKind.STUDY,
                }.intersection(available)
            )
            reasons.append(
                "variant annotations, studies, credible sets, and fine-mapping "
                "context requested without a gene-ranking task"
            )
            return EvidencePlan(
                kinds=tuple(sorted(selected, key=lambda item: item.value)),
                rationale=tuple(reasons),
            )
        if query.task == "gene_associations":
            gene_association_kinds = {
                EvidenceKind.VARIANT_ANNOTATION,
                EvidenceKind.QTL,
                EvidenceKind.COLOCALIZATION,
                EvidenceKind.REGULATORY,
                EvidenceKind.EXPRESSION,
                EvidenceKind.FUNCTION,
                EvidenceKind.PATHWAY,
                EvidenceKind.LITERATURE,
            }
            if not query.benchmark:
                # L2G is a scored variant-to-gene evidence type.  It may be
                # reported for an association question without turning the
                # response into a causal ranking; frozen benchmark queries keep
                # the existing leakage-control exclusion.
                gene_association_kinds.add(EvidenceKind.L2G)
            selected.update(gene_association_kinds.intersection(available))
            reasons.append(
                "variant annotations, L2G, and direct molecular variant-to-gene "
                "evidence requested without a causal-gene ranking"
            )
            return EvidencePlan(
                kinds=tuple(sorted(selected, key=lambda item: item.value)),
                rationale=tuple(reasons),
            )
        variant_kinds = {
            EvidenceKind.QTL,
            EvidenceKind.COLOCALIZATION,
            EvidenceKind.REGULATORY,
        }
        if not query.benchmark and query.task == "prioritize_genes":
            # The existing direct-evidence gate also accepts explicit high-impact
            # coding/splice links; retrieve that dimension without an LLM plan.
            variant_kinds.add(EvidenceKind.VARIANT_ANNOTATION)
        selected.update(variant_kinds.intersection(available))
        if selected:
            reasons.append("variant-linked association and regulatory evidence requested")
        if query.study_ids and EvidenceKind.GWAS in available:
            selected.add(EvidenceKind.GWAS)
            reasons.append("GWAS records requested for explicit input study identifiers")
        if query.study_ids and EvidenceKind.STUDY in available:
            selected.add(EvidenceKind.STUDY)
            reasons.append("study and fine-mapping context requested for explicit studies")
        elif EvidenceKind.GWAS in available or EvidenceKind.STUDY in available:
            reasons.append("GWAS lookup skipped because no input study identifier was supplied")
        if query.candidate_genes or not query.benchmark:
            gene_kinds = {
                EvidenceKind.EXPRESSION,
                EvidenceKind.FUNCTION,
                EvidenceKind.PATHWAY,
            }
            selected.update(gene_kinds.intersection(available))
            reasons.append(
                "expression, function, and pathway evidence requested for supplied "
                "or locus-discovered candidates"
            )
        if EvidenceKind.LITERATURE in available:
            selected.add(EvidenceKind.LITERATURE)
            reasons.append("literature evidence requested when the provider exposes provenance")
        if not query.benchmark and EvidenceKind.L2G in available:
            selected.add(EvidenceKind.L2G)
            reasons.append("Open Targets L2G requested for exploratory, non-benchmark use")
        if not query.benchmark and query.task == "target_assessment":
            target_kinds = {
                EvidenceKind.SAFETY,
                EvidenceKind.TRACTABILITY,
                EvidenceKind.INTERACTION,
                EvidenceKind.PHENOTYPE,
            }
            selected.update(target_kinds.intersection(available))
            reasons.append(
                "target safety, tractability, interaction, and phenotype evidence requested"
            )
        if not query.benchmark and query.task == "disease_associations":
            selected.update(
                {EvidenceKind.TARGET_DISEASE, EvidenceKind.ONTOLOGY}.intersection(available)
            )
            reasons.append("target-disease and disease ontology evidence requested")
        if not query.benchmark and query.task == "drug_repurposing":
            selected.update(
                {
                    EvidenceKind.DRUG,
                    EvidenceKind.SAFETY,
                    EvidenceKind.TRACTABILITY,
                }.intersection(available)
            )
            if query.trait_ids and EvidenceKind.TARGET_DISEASE in available:
                selected.add(EvidenceKind.TARGET_DISEASE)
            reasons.append("drug and translational target evidence requested")
        # In a benchmark, L2G remains an external baseline and is never agent-visible.
        return EvidencePlan(
            kinds=tuple(sorted(selected, key=lambda item: item.value)),
            rationale=tuple(reasons),
        )


class GWASMindAgent:
    """Single-agent orchestration over one Open Targets tool contract."""

    def __init__(
        self,
        *,
        gateway: GWASToolGateway,
        synthesizer: EvidenceSynthesizer,
        gate: EvidenceSufficiencyGate | None = None,
        planner: EvidencePlanner | None = None,
        artifacts: ArtifactStore | None = None,
        audit: AuditLogger | None = None,
        code_version: str = __version__,
        progress_sink: ProgressSink | None = None,
    ):
        self.gateway = gateway
        self.synthesizer = synthesizer
        self.gate = gate or EvidenceSufficiencyGate()
        self.planner = planner or EvidencePlanner()
        self.artifacts = artifacts
        self.audit = audit or gateway.audit
        self.code_version = code_version
        self.progress_sink = progress_sink
        if isinstance(self.synthesizer, OpenAICompatibleSynthesizer):
            self.synthesizer.progress_sink = progress_sink
        self.system_prompt = load_prompt("agent_system")

    async def run(
        self,
        query: LocusQuery,
        *,
        run_id: str | None = None,
        defer_ranking: bool = False,
        retrieved: ToolCallResult | None = None,
    ) -> AgentResult:
        current_run = run_id or str(uuid4())
        if not isinstance(current_run, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", current_run):
            raise ValueError("run id must contain only letters, digits, dot, underscore, or hyphen")
        emit_progress(
            self.progress_sink,
            "validating_input",
            locus_id=query.locus_id,
        )
        self.audit.log(
            "agent_run_started",
            run_id=current_run,
            locus_id=query.locus_id,
            benchmark=query.benchmark,
        )
        preflight = self.gate.preflight(query)
        if preflight.reasons:
            emit_progress(self.progress_sink, "result_processing", status="started",
                          operation="organizing", records=0)
            status = _status_from_report(
                preflight,
                None,
                require_significant=(
                    query.task not in _RETRIEVAL_ONLY_TASKS
                    and self.gate.config.require_genome_wide_signal
                ),
            )
            emit_progress(
                self.progress_sink,
                "checking_sufficiency",
                stage="input",
                records=0,
                policy_blocks=preflight.policy_blocks,
            )
            emit_progress(self.progress_sink, "result_processing", status="completed",
                          operation="organizing", records=0)
            _emit_safety_gate_outcome(self.progress_sink, status, preflight)
            return self._finalize(
                query=query,
                run_id=current_run,
                status=status,
                sufficiency=preflight,
                summary=_status_summary(status, task=query.task),
                retrieval={"skipped": True, "reason": "input gate"},
            )

        if retrieved is None:
            emit_progress(self.progress_sink, "planning_evidence")
            plan = self.planner.plan(query, self.gateway.available_kinds)
            self.audit.log("evidence_plan_created", run_id=current_run, **plan.to_dict())
            emit_progress(self.progress_sink, "retrieving_evidence")
            tool_result = await self.gateway.fetch_evidence(query, set(plan.kinds))
        else:
            # Stage four consumes the exact Stage-two ledger. No MCP calls or
            # candidate discovery are repeated after the scientific review.
            tool_result = retrieved
            plan = EvidencePlan(retrieved.requested_kinds, ("reuse reviewed evidence",))
            self.audit.log(
                "reviewed_evidence_reused",
                run_id=current_run,
                records=len(retrieved.records),
                call_id=retrieved.call_id,
            )
        emit_progress(self.progress_sink, "result_processing", status="started",
                      operation="organizing", records=len(tool_result.records))
        resolved_query = tool_result.resolved_query or query
        emit_progress(
            self.progress_sink,
            "checking_sufficiency",
            tool_status=tool_result.status.value,
            records=len(tool_result.records),
            policy_blocks=len(tool_result.policy_blocks),
        )
        sufficiency = self.gate.evaluate(resolved_query, tool_result)
        limited_scope = limited_discovery_ranking_scope(resolved_query, tool_result)
        status = _status_from_report(
            sufficiency,
            tool_result.status,
            allow_limited_discovery=bool(limited_scope),
            require_significant=(
                resolved_query.task not in _RETRIEVAL_ONLY_TASKS
                and self.gate.config.require_genome_wide_signal
            ),
        )
        if (defer_ranking or query.task in _RETRIEVAL_ONLY_TASKS) and (
            status is DecisionStatus.PRIORITIZED
        ):
            status = DecisionStatus.EVIDENCE_RETRIEVED
        emit_progress(self.progress_sink, "result_processing", status="completed",
                      operation="organizing", records=len(tool_result.records))
        _emit_safety_gate_outcome(self.progress_sink, status, sufficiency)
        retrieval = {
            "plan": plan.to_dict(),
            "tool_call_id": tool_result.call_id,
            "status": tool_result.status.value,
            "response_sha256": tool_result.response_sha256,
            "raw_response_sha256": tool_result.raw_response_sha256,
            "raw_response_artifact": (
                tool_result.raw_response_artifact.path
                if tool_result.raw_response_artifact
                else None
            ),
            # Serialize records one at a time; keep the authoritative objects
            # available to review without another deep copy of all payloads.
            "records": tool_result.records,
            "policy_blocks": list(tool_result.policy_blocks),
            "warnings": list(tool_result.warnings),
            "error": tool_result.error,
            "candidate_audit": tool_result.candidate_audit,
            "operation_issues": list(tool_result.operation_issues),
            "candidate_pool": candidate_pool_rows(tool_result.candidate_audit)
            if tool_result.candidate_audit else [],
            "resolved_query": _bounded_resolved_query_metadata(resolved_query),
            "query": asdict(resolved_query),
            "input_query": asdict(query),
            "data_mode": self.gateway.provider.capabilities.mode.value,
            "data_release": self.gateway.provider.capabilities.release,
            "ranking_deferred": defer_ranking,
            "retrieval_statuses": dict(tool_result.retrieval_statuses),
        }
        if status is not DecisionStatus.PRIORITIZED:
            if status is DecisionStatus.EVIDENCE_RETRIEVED:
                if defer_ranking:
                    summary = (
                        f"Retrieved {len(tool_result.records)} evidence record(s); "
                        "candidate ranking is deferred until after the safety review."
                    )
                elif query.task == "gene_associations":
                    gene_linked = sum(
                        1
                        for record in tool_result.records
                        if isinstance(record.gene_id, str) and record.gene_id.strip()
                    )
                    summary = (
                        f"Retrieved {gene_linked} permitted gene-linked evidence record(s) "
                        "without causal-gene ranking."
                    )
                else:
                    summary = (
                        f"Retrieved {len(tool_result.records)} permitted Open Targets "
                        "variant/study record(s)."
                    )
            else:
                summary = _status_summary(status, task=query.task)
                if query.task == "prioritize_genes":
                    if status is DecisionStatus.INCOMPLETE:
                        summary = "Candidate ranking was not run because its evidence entry requirements were not met."
                    emit_progress(self.progress_sink, "ranking_not_attempted", ranking_attempted=False,
                        status=status.value, reason_codes=list(dict.fromkeys(reason.split(":", 1)[0] for reason in sufficiency.reasons)),
                        reasons=list(sufficiency.reasons), reason=summary)
            return self._finalize(
                query=query,
                run_id=current_run,
                status=status,
                sufficiency=sufficiency,
                summary=summary,
                retrieval=retrieval,
                evidence=tool_result.records,
                decision=({"skipped": True, "status": "ranking_not_attempted", "ranking_attempted": False,
                           "reason_codes": list(dict.fromkeys(reason.split(":", 1)[0] for reason in sufficiency.reasons)),
                           "reasons": list(sufficiency.reasons)}
                          if query.task == "prioritize_genes" and status is not DecisionStatus.EVIDENCE_RETRIEVED else None),
            )

        emit_progress(
            self.progress_sink,
            "synthesizing_decision",
            decision_records=sufficiency.decision_evidence_records,
            total_evidence_records=len(tool_result.records),
            candidate_genes=len(resolved_query.candidate_genes),
            ranking_scope="observed_admitted_candidates" if limited_scope else "retrieved_candidates",
            candidate_discovery_complete=(tool_result.candidate_audit or {}).get("candidate_discovery_complete"),
        )
        model_fact_pointer = None
        model_output_pointers = []
        try:
            if isinstance(self.synthesizer, OpenAICompatibleSynthesizer):
                self.synthesizer.ranking_retrieval_statuses = dict(tool_result.retrieval_statuses)
                from .card_store import cards_from_audit
                self.synthesizer.v2g_cards = cards_from_audit(tool_result.candidate_audit)
                def generation_audit(row):
                    self.audit.log("model_generation_attempt", run_id=current_run, **row)
                    emit_progress(self.progress_sink, "model_generation_attempt", **row)
                self.synthesizer.generation_audit_sink = generation_audit
                def store_model_fact_view(value):
                    nonlocal model_fact_pointer
                    scientific_hash = value.get("model_input_sha256")
                    if not isinstance(scientific_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", scientific_hash):
                        raise ValueError("Scientific fact view requires a valid content identity")
                    if self.artifacts is not None:
                        model_fact_pointer = self.artifacts.write_json(
                            f"runs/{current_run}/model_inputs/{scientific_hash}.json", value, compact=True,
                        )
                        if value.get("schema_version") == 2:
                            from .ranking_exports import store_ranking_exports

                            descriptor = store_ranking_exports(self.artifacts, current_run, value)
                            emit_progress(self.progress_sink, "ranking_input_frozen", **descriptor)
                    self.audit.log("model_fact_view_stored", run_id=current_run,
                        scientific_payload_sha256=scientific_hash,
                        evidence_alias_count=len(value.get("evidence_aliases", {})),
                        record_binding_count=len(value.get("record_bindings", [])),
                        artifact=asdict(model_fact_pointer) if model_fact_pointer is not None else None)
                self.synthesizer.model_fact_view_sink = store_model_fact_view
                def store_model_output(value):
                    output_hash = value.get("content_sha256")
                    attempt = value.get("attempt")
                    if (not isinstance(output_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", output_hash)
                            or not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1):
                        raise ValueError("Model output requires an attempt and valid content identity")
                    review = value.get("event") == "output_validation"
                    suffix = f"-review-{value['review_sequence']:02d}" if review else "-unvalidated"
                    payload = {**value, "run_id": current_run,
                        "model_fact_view_artifact": asdict(model_fact_pointer) if model_fact_pointer is not None else None}
                    pointer = self.artifacts.write_json(
                        f"runs/{current_run}/model_outputs/attempt-{attempt:02d}-{output_hash}{suffix}.json", payload, compact=True,
                    ) if self.artifacts is not None else None
                    receipt = {"attempt": attempt, "purpose": value.get("purpose"),
                        "content_sha256": output_hash, "event": value.get("event"),
                        "artifact": asdict(pointer) if pointer is not None else None}
                    model_output_pointers.append(receipt)
                    self.audit.log("model_output_validation_stored" if review else "model_visible_output_stored",
                        run_id=current_run, **{key: item for key, item in receipt.items() if key != "event"},
                        **({"validation_passed": value["validation_passed"],
                            "validation_category": value["validation_category"],
                            "validation_issues": value["validation_issues"]} if review else {"validation_status": "unvalidated"}))
                self.synthesizer.model_output_sink = store_model_output
            synthesis = await self.synthesizer.synthesize(
                resolved_query,
                tool_result.records,
                system_prompt=self.system_prompt,
            )
        except SynthesisOutputError as exc:
            self.audit.log("synthesis_structure_unavailable", run_id=current_run,
                           gate="gate2", target="candidate_ranking",
                           validation_attempts=exc.validation_attempts)
            synthesis = await _unavailable_model_ranking(
                resolved_query,
                tool_result.records,
                identity=self.synthesizer.identity,
                reason="candidate-ranking output did not satisfy the structure contract",
                category="json_structure",
            )
        except Exception as exc:  # noqa: BLE001 - preserve evidence without publishing a substitute ranking
            from .model_generation import classify_generation_failure
            self.audit.log(
                "synthesis_error",
                run_id=current_run,
                error_type=type(exc).__name__,
                **classify_generation_failure(exc),
            )
            synthesis = await _unavailable_model_ranking(
                resolved_query,
                tool_result.records,
                identity=self.synthesizer.identity,
                reason=f"candidate-ranking model unavailable ({type(exc).__name__})",
                category=getattr(exc, "category", None) if getattr(exc, "category", None) in _RANKING_RESOURCE_LIMITS
                else classify_generation_failure(exc)["category"],
            )
            synthesis = _retain_ranking_resource_limit(synthesis, exc)

        if not synthesis.usage.get("ranking_unavailable"):
            from .llm import _candidate_gene_universe
            from .shortlisting import selection_contract_errors
            if (tool_result.candidate_audit or {}).get("v2g_cards"):
                # The actual input contract owns this requirement, not the reply.
                synthesis = replace(synthesis, model_shortlisting_required=True)
            selection_errors = selection_contract_errors(synthesis,
                _candidate_gene_universe(resolved_query, tool_result.records),
                required=synthesis.model_shortlisting_required)
            if selection_errors:
                self.audit.log("model_selection_structure_unavailable", run_id=current_run,
                               validation_errors=list(selection_errors))
                synthesis = await _unavailable_model_ranking(resolved_query, tool_result.records,
                    identity=synthesis.identity,
                    reason="model candidate decisions did not satisfy the complete-pool contract",
                    category="candidate_contract")

        emit_progress(self.progress_sink, "ranking_validation", draft="initial",
                      evidence_records=len(tool_result.records))
        initial_errors = ((synthesis.usage["ranking_failure_reason"],)
            if synthesis.usage.get("ranking_unavailable") else
            validate_synthesis(resolved_query, tool_result.records, synthesis))
        if isinstance(self.synthesizer, OpenAICompatibleSynthesizer) and not synthesis.usage.get("ranking_unavailable"):
            self.synthesizer.mark_generation_validation(passed=not initial_errors, issues=initial_errors)
        self.audit.log("gate2_ranking_draft_review", run_id=current_run,
            draft="model_unavailable" if synthesis.usage.get("ranking_unavailable") else "initial",
            validation_passed=not initial_errors, issue_count=len(initial_errors), validation_errors=list(initial_errors),
            validator_ids=list(load_gate_rules().validator_ids("ranking")))
        hard_fact_review = None
        if not synthesis.usage.get("ranking_unavailable"):
            original_synthesis = synthesis
            from .gate2_hallucination import remove_ranking_hallucinations, remove_ranking_rsid_mismatches
            synthesis = remove_ranking_rsid_mismatches(resolved_query, synthesis)
            synthesis = repair_ranking_assertions(resolved_query, tool_result.records, synthesis, include_numeric_trace=True)
            synthesis = remove_ranking_hallucinations(resolved_query, tool_result.records, synthesis)
            assert_ranking_unchanged(original_synthesis, synthesis)
            validation_errors = validate_synthesis(resolved_query, tool_result.records, synthesis)
            unresolved = unverified_synthesis_fields(resolved_query, tool_result.records, synthesis, validation_errors)
            if unresolved:
                synthesis = replace(synthesis, usage={**synthesis.usage, "ranking_needs_review": 1,
                    "unverified_fields": [*synthesis.usage.get("unverified_fields", []), *unresolved]})
            hard_fact_review = _store_ranking_hard_fact_review(
                self.artifacts, self.audit, current_run, original_synthesis, synthesis, validation_errors)
            if synthesis.usage.get("hard_fact_corrections"):
                self.audit.log("ranking_local_assertion_repair_completed", run_id=current_run,
                    validation_passed=not validation_errors,
                    repaired_issue_count=len(synthesis.usage["hard_fact_corrections"]),
                    validator_ids=list(load_gate_rules().validator_ids("ranking")))
        else:
            validation_errors = initial_errors
        if validation_errors and not synthesis.usage.get("ranking_unavailable"):
            # A completed JSON draft is corrected only from a unique raw fact.
            # A parseable model ranking remains intact. Mark unresolved fields;
            # never replace or erase the gene order as a validation side effect.
            self.audit.log("synthesis_assertions_unresolved", run_id=current_run,
                           validation_errors=list(validation_errors), model_rewrite_requested=False)
        decision = _decision_artifact(synthesis, validation_errors)
        if hard_fact_review is not None:
            decision["hard_fact_review"] = hard_fact_review
            decision["status"] = hard_fact_review["status"]
        if model_fact_pointer is not None:
            decision["model_fact_view_artifact"] = asdict(model_fact_pointer)
        if model_output_pointers:
            decision["model_output_artifacts"] = model_output_pointers
        if isinstance(self.synthesizer, OpenAICompatibleSynthesizer):
            decision["generation_attempts"] = self.synthesizer.generation_attempts
        output_review = {
            "gate": "gate2", "target": "candidate_ranking", "check_id": "model_output_grounding",
            "validation_passed": not validation_errors and not synthesis.usage.get("unverified_fields"),
            "display_validation_passed": not validation_errors,
            "ranking_available": bool(synthesis.ranked_genes),
            "ranked_gene_count": len(synthesis.ranked_genes),
            "fully_verified": not validation_errors and not synthesis.usage.get("unverified_fields"),
            "unverified_field_count": len(synthesis.usage.get("unverified_fields", [])),
            "status": hard_fact_review["status"] if hard_fact_review else "ranking_unavailable",
            "not_assessed": hard_fact_review.get("not_assessed", []) if hard_fact_review else [],
            "fact_issues": hard_fact_review.get("fact_issues", {}) if hard_fact_review else {},
            "validator_ids": list(load_gate_rules().validator_ids("ranking")),
            "method": "model_unavailable" if synthesis.usage.get("ranking_unavailable") else
                "model_needs_review" if validation_errors or synthesis.usage.get("unverified_fields") else
                "deterministic" if synthesis.identity.offline else
                "model_fact_corrected" if synthesis.usage.get("hard_fact_corrections") else
                "model_repaired" if _ranking_repair_attempts(synthesis.usage) else "model",
            "retained_valid_rankings": synthesis.usage.get("retained_valid_rankings", 0),
            "projection_omitted_records": synthesis.evidence_projection.get("records_omitted_after_deduplication", 0),
            **({"resource_limit": synthesis.usage["ranking_resource_limit"]}
               if synthesis.usage.get("ranking_resource_limit") else {}),
            **({"failure_category": synthesis.usage["ranking_failure_category"]}
               if synthesis.usage.get("ranking_unavailable") and synthesis.usage.get("ranking_failure_category") else {}),
        }
        decision["gate2_review"] = output_review
        emit_progress(
            self.progress_sink,
            "synthesis_completed",
            ranked_genes=len(synthesis.ranked_genes),
            validation_passed=output_review["validation_passed"],
            repair_attempts=_ranking_repair_attempts(synthesis.usage),
            deterministic_fallback=False,
            ranking_unavailable=bool(synthesis.usage.get("ranking_unavailable")),
            gate2_review=output_review,
            **_public_token_usage(synthesis.usage),
        )
        explanation = {
            **({"candidate_decisions": [asdict(row) for row in synthesis.candidate_decisions]}
               if synthesis.candidate_decisions else {}),
            "claims": [asdict(claim) for claim in synthesis.claims],
            "model_summary": synthesis.summary,
            "notice": "Exploratory hypothesis generation; not proof of causality or mechanism.",
        }
        if synthesis.usage.get("ranking_unavailable"):
            summary = "Candidate ranking is unavailable. The retrieved evidence is retained for a partial report."
            failed = _add_warning(_add_reason(sufficiency,
                "INCOMPLETE: candidate ranking remains unavailable after verification; retrieved evidence was retained."),
                "Candidate-ranking warning: no ranking passed verification; interpretation is limited to the retained evidence.")
            return self._finalize(
                query=query,
                run_id=current_run,
                status=DecisionStatus.INCOMPLETE,
                sufficiency=failed,
                summary=summary,
                retrieval=retrieval,
                decision={**decision, "skipped": True, "status": "ranking_unavailable", "ranking_attempted": True},
                explanation={"claims": [], "model_summary": summary, "notice": "No unvalidated ranking was published."},
                evidence=tool_result.records,
            )

        if synthesis.usage.get("unverified_fields"):
            sufficiency = _add_warning(sufficiency,
                "Some model fields remain unverified. Unbound numerical values are marked in place; "
                "the original model ranking is retained for review.")
        recovery_warning = _ranking_recovery_warning(synthesis.usage)
        if recovery_warning:
            sufficiency = _add_warning(sufficiency, recovery_warning)
        omitted = synthesis.evidence_projection.get("records_omitted_after_deduplication", 0)
        if isinstance(omitted, int) and omitted > 0 and (
            not synthesis.usage.get("deterministic_fallback") or synthesis.usage.get("retained_valid_rankings")
        ):
            sufficiency = _add_warning(sufficiency, f"Model-input coverage warning: {omitted} deduplicated evidence records did not fit the model context. The full evidence ledger was retained and checked; the model ranking used its disclosed evidence subset.")

        summary = (
            f"Prioritized {len(synthesis.ranked_genes)} candidate gene(s) from "
            f"{sufficiency.decision_evidence_records} gene-linked decision record(s) "
            f"among {len(tool_result.records)} permitted Open Targets record(s). "
            "This is hypothesis generation, not proof of a causal gene or mechanism."
        )
        if synthesis.candidate_decisions and not synthesis.ranked_genes:
            summary = "The model did not select a candidate for prioritization. All candidate cards and selection reasons are retained."
        if limited_scope:
            summary += " Ranking is limited to observed admitted candidates; candidate discovery remains incomplete."
        return self._finalize(
            query=query,
            run_id=current_run,
            status=DecisionStatus.PRIORITIZED if synthesis.ranked_genes else DecisionStatus.INSUFFICIENT_EVIDENCE,
            sufficiency=sufficiency,
            summary=summary,
            retrieval=retrieval,
            decision=decision,
            explanation=explanation,
            evidence=tool_result.records,
            synthesis=synthesis,
        )

    def _finalize(
        self,
        *,
        query: LocusQuery,
        run_id: str,
        status: DecisionStatus,
        sufficiency: SufficiencyReport,
        summary: str,
        retrieval: dict[str, Any],
        decision: dict[str, Any] | None = None,
        explanation: dict[str, Any] | None = None,
        evidence: tuple = (),
        synthesis: SynthesisOutput | None = None,
    ) -> AgentResult:
        emit_progress(
            self.progress_sink,
            "writing_artifacts",
            status=status.value,
        )
        identity = synthesis.identity if synthesis else self.synthesizer.identity
        artifact_paths: dict[str, str] = {}
        # Only this same frozen-record tuple may reuse the first write. A list,
        # another tuple, or a same-sized but different sequence takes the normal
        # writer. There are no evidence mutations between these two writes.
        shared_records = evidence if (type(evidence) is tuple and evidence is retrieval.get("records")
                                      and all(type(row) is EvidenceRecord for row in evidence)) else None
        reusable_retrieval = None
        if self.artifacts:
            emit_progress(self.progress_sink, "result_processing", status="started",
                          operation="saving", records=len(retrieval.get("records", ())))
            values = {
                "retrieval": retrieval,
                "decision": decision or {"skipped": True, "status": status.value},
                "explanation": explanation
                or {"skipped": True, "status": status.value, "summary": summary},
            }
            for name, value in values.items():
                if name == "retrieval":
                    pointer = self.artifacts.write_evidence_document(
                        f"runs/{run_id}/{name}.json", value, field="records",
                        metadata_summaries=candidate_metadata_summaries(
                            value.get("candidate_audit"), value.get("candidate_pool", [])))
                    if shared_records is not None:
                        reusable_retrieval = pointer
                else:
                    pointer = self.artifacts.write_json(f"runs/{run_id}/{name}.json", value)
                artifact_paths[name] = pointer.path

            manifest = RunManifest.create(
                mode=self.gateway.provider.capabilities.mode.value,
                data_release=self.gateway.provider.capabilities.release,
                snapshot_hash=self.gateway.provider.capabilities.snapshot_hash,
                model_provider=identity.provider,
                model_id=identity.effective_model_id,
                model_parameters=self.synthesizer.parameters,
                prompt=self.system_prompt,
                policy=self.gateway.policy,
                code_version=self.code_version,
                run_id=run_id,
                input_value=asdict(query),
                tool_contract={
                    "available_kinds": sorted(kind.value for kind in self.gateway.available_kinds),
                    "provider_mode": self.gateway.provider.capabilities.mode.value,
                },
                requested_model_id=identity.requested_model_id,
                observed_model_id=identity.observed_model_id,
                prompt_version=PROMPT_VERSIONS["agent_system"],
                seed=self.synthesizer.parameters.get("seed"),
                benchmark=query.benchmark,
                offline=identity.offline,
            )
            manifest_pointer = self.artifacts.write_json(
                f"runs/{run_id}/run_manifest.json", manifest.to_dict()
            )
            artifact_paths["manifest"] = manifest_pointer.path

        result = AgentResult(
            locus_id=query.locus_id,
            status=status,
            ranked_genes=synthesis.ranked_genes
            if synthesis and status is DecisionStatus.PRIORITIZED
            else (),
            candidate_decisions=synthesis.candidate_decisions if synthesis else (),
            evidence=tuple(evidence),
            sufficiency=sufficiency,
            summary=summary,
            run_id=run_id,
            model_id=identity.effective_model_id,
            data_release=self.gateway.provider.capabilities.release,
            data_mode=self.gateway.provider.capabilities.mode.value,
            limitations=tuple(
                item
                for item in (
                    (
                        "Open Targets annotations supplement the uploaded GWAS "
                        "measurements; they do not replace the source summary statistics."
                    )
                    if query.task in _RETRIEVAL_ONLY_TASKS
                    else (
                        "Open Targets evidence supports prioritization but cannot establish "
                        "causality."
                    ),
                    *sufficiency.warnings,
                    (ot_mhc_coverage_limitation() if query_overlaps_ot_mhc(query) else None),
                    "Results require domain-expert review and independent validation.",
                )
                if item is not None
            ),
            artifacts=artifact_paths,
            ld_expansion=(retrieval.get("query") or {}).get("ld_expansion", query.ld_expansion),
            gate2_review=(decision or {}).get("gate2_review"),
            sampling_parameters=next(((row.get("token_budget") or {}).get("sampling_parameters")
                for row in reversed((decision or {}).get("generation_attempts", []))
                if isinstance((row.get("token_budget") or {}).get("sampling_parameters"), dict)), None),
        )
        if self.artifacts:
            result_document = replace(result, evidence=()).to_dict()
            result_document["evidence"] = result.evidence
            pointer = (
                self.artifacts._reuse_native_evidence(f"runs/{run_id}/result.json", result_document,
                                                      reusable_retrieval, original_records=shared_records)
                if reusable_retrieval is not None and result.evidence is shared_records else None
            )
            if pointer is None:
                pointer = self.artifacts.write_evidence_document(
                    f"runs/{run_id}/result.json", result_document, field="evidence")
            artifact_paths["result"] = pointer.path
            result = replace(result, artifacts=dict(artifact_paths))
            self.artifacts.write_json(
                f"runs/{run_id}/run_manifest.json",
                {**manifest.to_dict(), "result_artifact": {
                    "path": pointer.path,
                    "canonical_sha256": pointer.sha256,
                    # Compact artifacts contain exactly these canonical UTF-8
                    # bytes, without a BOM, indentation or trailing newline.
                    "byte_sha256": pointer.sha256,
                    "bytes": pointer.bytes,
                    "encoding": "canonical-json-utf8",
                }},
            )
            emit_progress(self.progress_sink, "result_processing", status="completed",
                          operation="saving", records=len(retrieval.get("records", ())))
        self.audit.log(
            "agent_run_finished",
            run_id=run_id,
            locus_id=query.locus_id,
            status=status.value,
            ranked_genes=len(result.ranked_genes),
        )
        emit_progress(
            self.progress_sink,
            "completed",
            status=status.value,
            ranked_genes=len(result.ranked_genes),
        )
        return result


def _status_from_report(
    report: SufficiencyReport,
    tool_status: ToolStatus | None,
    *,
    require_significant: bool = True,
    allow_limited_discovery: bool = False,
) -> DecisionStatus:
    if not report.valid_input or any(
        reason.startswith("INVALID_INPUT") for reason in report.reasons
    ):
        return DecisionStatus.INVALID_INPUT
    if require_significant and report.significant_signal is False:
        return DecisionStatus.NO_GWAS_SIGNAL
    if tool_status is ToolStatus.BLOCKED or report.policy_blocks:
        return DecisionStatus.POLICY_BLOCKED
    if tool_status is ToolStatus.INVALID_INPUT:
        return DecisionStatus.INVALID_INPUT
    if tool_status is ToolStatus.INCOMPLETE and not (allow_limited_discovery and report.passed):
        return DecisionStatus.INCOMPLETE
    if tool_status is ToolStatus.ERROR:
        return DecisionStatus.SERVICE_UNAVAILABLE
    if any(reason.startswith("INCOMPLETE") for reason in report.reasons):
        return DecisionStatus.INCOMPLETE
    if report.reasons:
        return DecisionStatus.INSUFFICIENT_EVIDENCE
    return DecisionStatus.PRIORITIZED


def _emit_safety_gate_outcome(
    sink: ProgressSink | None,
    status: DecisionStatus,
    report: SufficiencyReport,
) -> None:
    """Publish the fail-closed gate decision before any synthesis can start."""

    passed = report.passed and status in {
        DecisionStatus.PRIORITIZED,
        DecisionStatus.EVIDENCE_RETRIEVED,
    }
    emit_progress(
        sink,
        "safety_gate_passed" if passed else "safety_gate_stopped",
        status=status.value,
        evidence_records=report.evidence_records,
        decision_evidence_records=report.decision_evidence_records,
        evidence_sources=report.evidence_sources,
        locus_linked_genes=report.locus_linked_genes,
        region_gene_context_status=report.region_gene_context_status,
        nearby_annotated_gene_count=report.nearby_annotated_gene_count,
        region_gene_context_source=report.region_gene_context_source,
    )


def _public_token_usage(usage: Mapping[str, Any]) -> dict[str, int]:
    """Expose bounded counters only; never provider reasoning content."""

    values: dict[str, int] = {}
    aliases = {
        "completion_tokens": "output_tokens",
        "output_tokens": "output_tokens",
        "total_tokens": "total_tokens",
        "reasoning_tokens": "reasoning_tokens",
    }
    for source, target in aliases.items():
        raw = usage.get(source)
        if isinstance(raw, int) and not isinstance(raw, bool) and 0 <= raw <= 10_000_000:
            values[target] = max(values.get(target, 0), raw)
    return values


async def _unavailable_model_ranking(
    query: LocusQuery,
    evidence: tuple[EvidenceRecord, ...],
    *,
    reason: str,
    identity: ModelIdentity,
    prior: SynthesisOutput | None = None,
    category: str | None = None,
) -> SynthesisOutput:
    """Retain model provenance, never manufacture or splice a replacement ranking."""
    del query, evidence
    category_usage = {"ranking_failure_category": category} if category else {}
    if prior is not None and prior.ranked_genes:
        return replace(prior, usage={**prior.usage, "ranking_unavailable": 0,
            "ranking_needs_review": 1, "ranking_failure_reason": reason[:300],
            "deterministic_fallback": 0, "retained_valid_rankings": 0,
            "unverified_fields": [*prior.usage.get("unverified_fields", []),
                {"field": "output", "reason": reason[:300]}]})
    return SynthesisOutput(
        ranked_genes=(),
        claims=(),
        summary="Model gene ranking did not complete; retrieved evidence remains available.",
        identity=prior.identity if prior is not None else identity,
        raw_response_sha256=prior.raw_response_sha256 if prior is not None else
            sha256_json({"ranking_unavailable": reason}),
        evidence_projection=dict(prior.evidence_projection) if prior is not None else {},
        model_input_sha256=prior.model_input_sha256 if prior is not None else "",
        model_visible_evidence_ids=prior.model_visible_evidence_ids if prior is not None else None,
        model_candidate_genes=prior.model_candidate_genes if prior is not None else None,
        usage={
            **(prior.usage if prior is not None else {}),
            "ranking_unavailable": 1,
            "ranking_failure_reason": reason[:300],
            **category_usage,
            "deterministic_fallback": 0,
            "retained_valid_rankings": 0,
        },
    )


def _ranking_repair_attempts(usage: Mapping[str, Any]) -> int:
    return sum(
        int(value)
        for key in ("structure_repair_attempts", "validation_repair_attempts")
        for value in (usage.get(key),)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    )


_RANKING_RESOURCE_LIMITS = {
    "ranking_output_budget_exhausted": "its single-request output allowance",
    "ranking_total_output_budget_exhausted": "its cumulative output allowance",
    "ranking_deadline_exhausted": "its total time allowance",
    "ranking_provider_budget_violation": "the requested output allowance at the provider",
}


def _retain_ranking_resource_limit(output: SynthesisOutput, error: Exception) -> SynthesisOutput:
    category = getattr(error, "category", None)
    if category in _RANKING_RESOURCE_LIMITS:
        return replace(output, usage={**output.usage, "ranking_resource_limit": category})
    return output


def _ranking_recovery_warning(usage: Mapping[str, Any]) -> str | None:
    if usage.get("deterministic_fallback"):
        resource = _RANKING_RESOURCE_LIMITS.get(usage.get("ranking_resource_limit"))
        return (
            "Candidate-ranking warning: "
            + (f"The model ranking did not complete within {resource}; retrieved evidence is retained. " if resource else "")
            + "a deterministic evidence-based fallback was used. "
            + ("Verified original ranking items were retained; replacement candidates use "
               if usage.get("retained_valid_rankings") else "Direct-linked candidates were ordered using ")
            + "fixed evidence-type weights and record counts; "
            "it is not a calibrated probability of causality or a full model interpretation."
        )
    return None


def _status_summary(
    status: DecisionStatus,
    *,
    task: str = "prioritize_genes",
) -> str:
    if task == "describe_variants":
        retrieval_messages = {
            DecisionStatus.INVALID_INPUT: (
                "Input validation failed before variant and study evidence retrieval."
            ),
            DecisionStatus.EVIDENCE_RETRIEVED: ("Variant and study evidence retrieval completed."),
            DecisionStatus.NO_GWAS_SIGNAL: (
                "No supplied variant met the genome-wide significance gate."
            ),
            DecisionStatus.POLICY_BLOCKED: (
                "The leakage policy blocked variant and study evidence retrieval."
            ),
            DecisionStatus.SERVICE_UNAVAILABLE: (
                "A required service failed during variant and study evidence retrieval."
            ),
            DecisionStatus.CONFLICTING_EVIDENCE: (
                "The retrieved variant and study evidence contained unresolved conflicts."
            ),
            DecisionStatus.INSUFFICIENT_EVIDENCE: (
                "No permitted variant or study evidence was available for the selected variants."
            ),
            DecisionStatus.INCOMPLETE: ("Variant and study evidence retrieval did not complete."),
            DecisionStatus.PRIORITIZED: ("Variant and study evidence retrieval completed."),
        }
        return retrieval_messages[status]

    if task == "gene_associations":
        retrieval_messages = {
            DecisionStatus.INVALID_INPUT: (
                "Input validation failed before variant-to-gene evidence retrieval."
            ),
            DecisionStatus.EVIDENCE_RETRIEVED: (
                "Variant-to-gene evidence retrieval completed without causal-gene ranking."
            ),
            DecisionStatus.NO_GWAS_SIGNAL: (
                "No supplied variant met the genome-wide significance gate."
            ),
            DecisionStatus.POLICY_BLOCKED: (
                "The leakage policy blocked variant-to-gene evidence retrieval."
            ),
            DecisionStatus.SERVICE_UNAVAILABLE: (
                "A required service failed during variant-to-gene evidence retrieval."
            ),
            DecisionStatus.CONFLICTING_EVIDENCE: (
                "The retrieved variant-to-gene evidence contained unresolved conflicts."
            ),
            DecisionStatus.INSUFFICIENT_EVIDENCE: (
                "No permitted gene-linked evidence was available for the selected variants."
            ),
            DecisionStatus.INCOMPLETE: ("Variant-to-gene evidence retrieval did not complete."),
            DecisionStatus.PRIORITIZED: (
                "Variant-to-gene evidence retrieval completed without causal-gene ranking."
            ),
        }
        return retrieval_messages[status]

    messages = {
        DecisionStatus.INVALID_INPUT: "Input validation failed before evidence synthesis.",
        DecisionStatus.EVIDENCE_RETRIEVED: (
            "Variant and study evidence was retrieved without gene prioritization."
        ),
        DecisionStatus.NO_GWAS_SIGNAL: "No supplied variant met the genome-wide significance gate.",
        DecisionStatus.POLICY_BLOCKED: (
            "The leakage policy blocked this run; no ranking was produced."
        ),
        DecisionStatus.SERVICE_UNAVAILABLE: "A required service failed; no ranking was produced.",
        DecisionStatus.CONFLICTING_EVIDENCE: "Opposing evidence triggered abstention.",
        DecisionStatus.INSUFFICIENT_EVIDENCE: (
            "Permitted evidence was insufficient for prioritization."
        ),
        DecisionStatus.INCOMPLETE: "The model output did not pass deterministic validation.",
        DecisionStatus.PRIORITIZED: "Candidate prioritization completed.",
    }
    return messages[status]


def _add_reason(report: SufficiencyReport, reason: str) -> SufficiencyReport:
    return replace(report, reasons=(*report.reasons, reason))


def _add_warning(report: SufficiencyReport, warning: str) -> SufficiencyReport:
    return replace(report, warnings=tuple(dict.fromkeys((*report.warnings, warning))))


def _bounded_resolved_query_metadata(query: LocusQuery) -> dict[str, Any]:
    """Return the canonical retrieval scope without copying unbounded input text."""

    metadata = {
        "locus_id": str(query.locus_id)[:_RESOLVED_METADATA_MAX_ID_CHARS],
        "variant_ids": [
            str(item.variant_id)[:_RESOLVED_METADATA_MAX_ID_CHARS]
            for item in query.variants[:_RESOLVED_METADATA_MAX_VARIANTS]
        ],
        "candidate_genes": [
            str(item)[:_RESOLVED_METADATA_MAX_ID_CHARS]
            for item in query.candidate_genes[:_RESOLVED_METADATA_MAX_CANDIDATES]
        ],
        **({"biological_context": query.biological_context} if query.biological_context is not None else {}),
    }
    if query.ld_expansion is not None:
        metadata["analysis_variant_count"] = len(query.analysis_variants)
        metadata["ld_expansion_status"] = query.ld_expansion.get("status")
        metadata["analysis_scope_location"] = "query.analysis_variants and query.ld_expansion in this retrieval artifact"
    return metadata


def _store_ranking_hard_fact_review(artifacts, audit, run_id, original, corrected, errors):
    """Keep the complete JSON draft and exact corrections separate from raw output."""
    from .fact_review_issues import project_fact_issues
    from .llm import _synthesis_output_payload

    assert_ranking_unchanged(original, corrected)
    original_payload = _synthesis_output_payload(original)
    corrected_payload = _synthesis_output_payload(corrected)
    corrections = list(corrected.usage.get("hard_fact_corrections", []))
    unverified = list(corrected.usage.get("unverified_fields", []))
    status = "needs_review" if errors else "partially_verified" if unverified else "verified"
    receipt = {
        "format": "ranking_hard_fact_review_v1", "run_id": run_id,
        "review_scope": "hallucination_checks_only",
        "ranking_order_preserved": True,
        "ranking_order": [row.gene_id for row in original.ranked_genes],
        "status": status, "validation_passed": not errors and not unverified,
        "display_validation_passed": not errors, "fully_verified": not errors and not unverified,
        "ranking_available": bool(corrected.ranked_genes), "validation_errors": list(errors),
        "validator_ids": list(load_gate_rules().validator_ids("ranking")),
        "model_identity": original.identity.to_dict(),
        "model_input_sha256": original.model_input_sha256,
        "model_evidence_aliases": original.model_evidence_aliases,
        "model_candidate_genes": list(original.model_candidate_genes)
            if original.model_candidate_genes is not None else None,
        "model_visible_evidence_ids": list(original.model_visible_evidence_ids)
            if original.model_visible_evidence_ids is not None else None,
        "ranking_replay": dict(original.usage.get("ranking_replay") or {}),
        "original_response_sha256": original.raw_response_sha256,
        "original_output_sha256": sha256_json(original_payload),
        "corrected_output_sha256": sha256_json(corrected_payload),
        "original_output": original_payload, "corrected_output": corrected_payload,
        "corrections": corrections, "model_rewrite_requested": False,
        "hallucination_removals": list(corrected.usage.get("hallucination_removals", [])),
        "unverified_fields": unverified,
        "not_assessed": ["free_text_source_proxy_interpretation", "free_text_evidence_absence", "causal_interpretation"],
        "unverified_values_are_authoritative": False,
    }
    pointer = artifacts.write_json(
        f"runs/{run_id}/model_outputs/gate2-{sha256_json(receipt)}.json", receipt, compact=True,
    ) if artifacts is not None else None
    public = {key: receipt[key] for key in (
        "status", "validation_passed", "fully_verified", "ranking_available", "validation_errors", "original_response_sha256",
        "original_output_sha256", "corrected_output_sha256", "model_rewrite_requested",
        "review_scope", "ranking_order_preserved")}
    public.update(correction_count=len(corrections), unverified_field_count=len(unverified),
                  hallucination_removal_count=len(receipt["hallucination_removals"]),
                  not_assessed=receipt["not_assessed"],
                  fact_issues=project_fact_issues(receipt, target="candidate_ranking", visible_output=original_payload),
                  artifact=asdict(pointer) if pointer else None)
    audit.log("ranking_hard_fact_review_stored", run_id=run_id, **public)
    return public


def _decision_artifact(
    synthesis: SynthesisOutput, validation_errors: tuple[str, ...]
) -> dict[str, Any]:
    return {
        **({"candidate_decisions": [asdict(item) for item in synthesis.candidate_decisions]}
           if synthesis.candidate_decisions or synthesis.model_shortlisting_required else {}),
        **({"model_shortlisting_required": True} if synthesis.model_shortlisting_required else {}),
        **({"model_full_pool_ranking_required": True} if synthesis.model_full_pool_ranking_required else {}),
        "ranked_genes": [asdict(item) for item in synthesis.ranked_genes],
        "validation_errors": list(validation_errors),
        "model_identity": synthesis.identity.to_dict(),
        "model_input_sha256": synthesis.model_input_sha256,
        "model_evidence_aliases": synthesis.model_evidence_aliases,
        "model_candidate_genes": list(synthesis.model_candidate_genes)
            if synthesis.model_candidate_genes is not None else None,
        "raw_response_sha256": synthesis.raw_response_sha256,
        "usage": dict(synthesis.usage),
        "evidence_projection": dict(synthesis.evidence_projection),
        "model_visible_evidence_ids": (
            list(synthesis.model_visible_evidence_ids)
            if synthesis.model_visible_evidence_ids is not None
            else None
        ),
    }
