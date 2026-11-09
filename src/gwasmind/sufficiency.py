from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .academic_review import comparable_conflict_count
from .biological_context import context_term_matches, evidence_context_values
from .candidates import evidence_candidate_scope, record_target_identity_valid, target_ids
from .configuration import parameter
from .context_review import prepare_context_review, review_record_context
from .evidence_sources import source_diversity
from .expression_policy import context_matches
from .models import (
    EvidenceKind,
    EvidenceRecord,
    LocusQuery,
    SufficiencyReport,
    analysis_variant_ids,
)
from .projection import _frozen_molqtl_coloc_scope_allowed, _is_high_impact_coding
from .region_gate import (
    REGION_GENE_CONTEXT_DATASETS,
    local_gene_window_label,
    region_gene_context_from_records,
)
from .tools import ToolCallResult, ToolStatus

RANKING_LOCUS_LINK_KINDS = frozenset(
    {
        EvidenceKind.QTL,
        EvidenceKind.COLOCALIZATION,
        EvidenceKind.REGULATORY,
    }
)

def is_direct_locus_gene_evidence(record: EvidenceRecord) -> bool:
    """Return whether a record directly links the locus to a candidate gene.

    Molecular QTL, colocalisation and enhancer/regulatory links are direct by
    construction.  Variant annotations count only when they name a gene and
    carry an explicit high-impact coding or splice consequence.
    """

    if not record.gene_id:
        return False
    if _frozen_molqtl_coloc_scope_allowed(record) is False:
        return False
    scope = evidence_candidate_scope(record.payload)
    if scope and not scope.get("direct_locus_support"):
        return False
    if record.kind in RANKING_LOCUS_LINK_KINDS:
        return True
    if record.kind is not EvidenceKind.VARIANT_ANNOTATION:
        return False
    return _is_high_impact_coding(record)


def decision_evidence_for_query(
    query: LocusQuery,
    records: tuple[EvidenceRecord, ...] | list[EvidenceRecord],
) -> tuple[EvidenceRecord, ...]:
    """Return evidence allowed to influence the requested decision.

    Open Targets L2G is retained in the full retrieval ledger because exploratory
    runs use it to discover a bounded candidate-gene shortlist.  It is deliberately
    excluded from every ``prioritize_genes`` decision surface so neither repeated
    L2G rows nor an L2G score can affect sufficiency, synthesis, or ranking.
    """

    if query.task != "prioritize_genes":
        return tuple(records)
    retained = tuple(
        record
        for record in records
        if record.kind is not EvidenceKind.L2G
        and _frozen_molqtl_coloc_scope_allowed(record) is not False
        and record.dataset not in REGION_GENE_CONTEXT_DATASETS
        and not evidence_candidate_scope(record.payload).get("context_only")
        and not (
            isinstance(record.payload.get("context"), Mapping)
            and record.payload["context"].get("rank_scope") in {"context_only", "other_trait", "unresolved"}
            and not (
                record.kind is EvidenceKind.COLOCALIZATION
                and record.payload["context"].get("molecular_detail_record") is True
                and record.payload["context"].get("phenotype_evidence_scope")
                in {"target_trait", "related_trait", "other_trait", "target_phenotype", "related_phenotype"}
            )
        )
    )
    qtl_pairs = {_molecular_record_pair(record) for record in retained
                 if record.kind is EvidenceKind.QTL and is_direct_locus_gene_evidence(record)}
    retained_ids = {id(record) for record in retained}
    return tuple(record for record in records if id(record) in retained_ids or
                 _qualified_related_colocalization(query, record, qtl_pairs))


def _molecular_record_pair(record: EvidenceRecord):
    context = record.payload.get("context") or {}
    upstream = record.payload.get("upstream") or {}
    if not isinstance(context, Mapping) or not isinstance(upstream, Mapping):
        return None
    raw = upstream.get("raw") or {}
    if not isinstance(raw, Mapping):
        return None
    loci = {value for value in (record.credible_set_id, context.get("study_locus_id"),
             raw.get("qtlStudyLocusId"), raw.get("studyLocusId")) if isinstance(value, str) and value}
    if len(loci) != 1 or not record.gene_id or not record.variant_id:
        return None
    return record.gene_id, next(iter(loci)), record.variant_id


def _qualified_related_colocalization(query, record, retained_qtl_pairs):
    """Retain an explicitly classified phenotype colocalisation through its QTL tuple.

    The phenotype label is kept verbatim.  Relatedness is not defined by a fixed
    parenthetical-string format: the retrieval stage supplies ``related_label`` and
    this gate verifies that the source label contains the queried phenotype in a
    biologically readable form.  Other-trait records are retained when the detailed
    retrieval stage has already verified their exact molecular anchor; they remain
    explicitly labelled as other-trait evidence in the model card.
    """
    if record.kind is not EvidenceKind.COLOCALIZATION:
        return False
    payload = record.payload
    context = payload.get("context") or {}
    if not isinstance(context, Mapping):
        return False
    role = evidence_candidate_scope(payload)
    if (context.get("rank_scope") not in {"related_trait", "other_trait", "context_only"} or
            context.get("target_trait_relation") not in {"related_label", "other_trait"} or
            role.get("direct_locus_support") is not True or role.get("context_only") is not False):
        return False
    pair = _molecular_record_pair(record)
    if pair is None or pair not in retained_qtl_pairs or record.variant_id not in analysis_variant_ids(query):
        return False
    if query.candidate_genes and record.gene_id not in query.candidate_genes:
        return False
    if not record_target_identity_valid(payload):
        return False
    raw = (payload.get("upstream") or {}).get("raw") or {}
    label = raw.get("rightTraitFromSource")
    if not isinstance(label, str) or not context_matches(query.trait, label):
        return False
    if re.search(r"\bfamily\s+history\b", label, re.IGNORECASE):
        return False
    from .supplementary_molecular import molecular_detail_anchor_verified
    return molecular_detail_anchor_verified(payload, pair[1])


def limited_discovery_ranking_scope(query: LocusQuery, result: ToolCallResult) -> tuple[str, ...]:
    """Recognize a bounded discovery gap without relabeling retrieval complete.

    Only the executor's explicit shortlist-gap result is eligible. Independent
    required failures, identity conflicts, empty direct support, and benchmarks
    retain their existing entry gates. A partial candidate pool is never complete.
    """
    audit = result.candidate_audit
    if query.task != "prioritize_genes" or query.benchmark or result.status is not ToolStatus.INCOMPLETE:
        return ()
    if result.policy_blocks or not isinstance(audit, Mapping):
        return ()
    if (audit.get("candidate_discovery_complete") is not False
        or audit.get("evidence_integrity_complete") is not True
        or audit.get("l2g_grouping_complete") is not True
        or audit.get("evidence_integrity_issues") or audit.get("l2g_grouping_issues")):
        return ()
    match = re.fullmatch(
        r"shortlist discovery coverage is incomplete: (.+); retained first-tier candidates are provisional",
        result.error or "",
    )
    if match is None:
        return ()
    gaps = tuple(match.group(1).split(", "))
    if not gaps or len(gaps) != len(set(gaps)) or not set(gaps) <= {"expression for the observed pool", "locus gene evidence", "population LD expansion", "LD proxy evidence retrieval"}:
        return ()
    if "expression for the observed pool" in gaps and audit.get("expression_discovery_complete") is not False:
        return ()
    if "locus gene evidence" in gaps and audit.get("locus_discovery_complete") is not False:
        return ()
    if "population LD expansion" in gaps and (audit.get("ld_expansion_complete") is not False
            or query.ld_expansion is None or query.ld_expansion.get("complete") is not False):
        return ()
    if "LD proxy evidence retrieval" in gaps and (audit.get("ld_proxy_retrieval_complete") is not False
            or query.ld_expansion is None or query.ld_expansion.get("proxy_evidence_complete") is not False):
        return ()
    try:
        query.validate()
    except ValueError:
        return ()
    selected = audit.get("candidate_ids")
    if (not isinstance(selected, (list, tuple)) or not selected
        or any(not isinstance(gene, str) or not gene.strip() for gene in selected)
        or set(selected) != set(query.candidate_genes)):
        return ()
    identifiers = [record.evidence_id for record in result.records]
    if any(not isinstance(value, str) or not value.strip() for value in identifiers) or len(identifiers) != len(set(identifiers)):
        return ()
    # The regional census is a declared coverage observation. Candidate admission
    # is established by the audited linkage/context routes below, including remote
    # links; an unavailable census must not recreate the retired SSR hard gate.
    # Reuse the executor's first-tier admission receipt. This is not a second
    # shortlist/scoring policy: an HPA/L2G admission can supply target context,
    # while an arbitrary background paper cannot authorize this recovery path.
    admitted_context = set()
    policy = audit.get("policy") if isinstance(audit.get("policy"), Mapping) else {}
    l2g_limit = policy.get("max_l2g_per_locus")
    audited_genes = audit.get("genes")
    if isinstance(audited_genes, (list, tuple)):
        for gene in audited_genes:
            if (not isinstance(gene, Mapping) or gene.get("gene_id") not in selected
                    or gene.get("selected") is not True or gene.get("candidate_tier") != "first"):
                continue
            expression = gene.get("expression_specificity")
            sources = gene.get("sources")
            hpa_admitted = (isinstance(expression, Mapping) and expression.get("qualified") is True
                and expression.get("method") in {
                    "context_hpa_style_linear_median_v1", "context_hpa_style_linear_median_v2"
                }
                and policy.get("expression_method") == expression.get("method")
                and isinstance(sources, (list, tuple)) and any(isinstance(source, Mapping)
                    and source.get("tier") == "expression" and source.get("module") == "target.expression"
                    and source.get("reason") == "context_specific_expression" for source in sources))
            l2g_admitted = (gene.get("l2g_per_locus_selected") is True
                and type(l2g_limit) is int and l2g_limit > 0
                and isinstance(gene.get("l2g_top5_loci"), (list, tuple))
                and any(isinstance(row, Mapping) and type(row.get("discovery_rank")) is int
                    and 1 <= row["discovery_rank"] <= l2g_limit
                    for row in gene["l2g_top5_loci"]))
            if hpa_admitted or l2g_admitted:
                admitted_context.add(gene["gene_id"])
    qualifying = []
    analysis_membership = frozenset(analysis_variant_ids(query))
    for record in decision_evidence_for_query(query, result.records):
        if not isinstance(record.gene_id, str) or record.gene_id not in selected:
            continue
        payload = record.payload if isinstance(record.payload, Mapping) else {}
        scope = evidence_candidate_scope(payload)
        direct = is_direct_locus_gene_evidence(record) and scope.get("direct_locus_support") is True
        context_admitted = (record.gene_id in admitted_context and record.kind in {
            EvidenceKind.EXPRESSION, EvidenceKind.FUNCTION, EvidenceKind.PATHWAY,
            EvidenceKind.SAFETY, EvidenceKind.TRACTABILITY})
        if not direct and not context_admitted:
            continue
        subject = payload.get("subject") if isinstance(payload.get("subject"), Mapping) else {}
        if not record_target_identity_valid(payload) or target_ids(payload) != (record.gene_id,):
            return ()
        if direct and (record.variant_id not in analysis_membership
                or subject.get("type") != "variant" or subject.get("id") != record.variant_id):
            return ()
        if not direct and (subject.get("type") != "target" or subject.get("id") != record.gene_id):
            return ()
        context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
        upstream = payload.get("upstream") if isinstance(payload.get("upstream"), Mapping) else {}
        raw = upstream.get("raw") if isinstance(upstream.get("raw"), Mapping) else {}
        source_ids = [value for value in (record.study_id, context.get("study_id"), raw.get("studyId")) if value is not None]
        if any(not isinstance(value, str) or not value.strip() for value in source_ids) or len({value.strip() for value in source_ids}) > 1:
            return ()
        qualifying.append(record)
    return gaps if qualifying else ()


@dataclass(frozen=True, slots=True)
class SufficiencyConfig:
    min_evidence_records: int = parameter("ranking.sufficiency.min_evidence_records")
    min_independent_datasets: int = parameter("ranking.sufficiency.min_independent_datasets")
    min_genes_with_evidence: int = parameter("ranking.sufficiency.min_genes_with_evidence")
    # A weak source-GWAS signal is important context, but it is not a data-
    # integrity failure.  The report must disclose it instead of suppressing
    # otherwise valid external annotation.
    require_genome_wide_signal: bool = parameter("ranking.sufficiency.require_genome_wide_signal")
    require_relevant_tissue_if_declared: bool = parameter("ranking.sufficiency.require_relevant_tissue_if_declared")
    require_locus_linked_gene_evidence: bool = parameter("ranking.sufficiency.require_locus_linked_gene_evidence")


class EvidenceSufficiencyGate:
    def __init__(self, config: SufficiencyConfig | None = None):
        self.config = config or SufficiencyConfig()

    def preflight(self, query: LocusQuery) -> SufficiencyReport:
        reasons: list[str] = []
        warnings: list[str] = []
        valid_input = True
        try:
            query.validate()
        except ValueError as exc:
            valid_input = False
            reasons.append(f"INVALID_INPUT: {exc}")

        observed_significance = [
            item.is_genome_wide_significant
            for item in (query.variants if valid_input else ())
            if item.is_genome_wide_significant is not None
        ]
        significant_signal = (
            None if not valid_input else query.significance_verified
            if query.significance_verified is not None
            else any(observed_significance)
            if observed_significance
            else None
        )
        if valid_input and query.task not in {"describe_variants", "gene_associations"}:
            if significant_signal is False:
                threshold_text = f'{parameter("analysis.gwas_significance_p"):g}'.replace("e-0", "e-")
                message = f"No supplied variant reaches P <= {threshold_text} in the source GWAS."
                if self.config.require_genome_wide_signal:
                    reasons.append("NO_GWAS_SIGNAL: " + message)
                else:
                    warnings.append(
                        message
                        + " External evidence may still be described, but must not be "
                        "presented as significance in the source dataset."
                    )
            elif significant_signal is None:
                message = "Source-GWAS significance could not be verified from p-values."
                if self.config.require_genome_wide_signal:
                    reasons.append("INVALID_INPUT: " + message)
                    valid_input = False
                else:
                    warnings.append(message)

        return SufficiencyReport(
            valid_input=valid_input,
            significant_signal=significant_signal,
            evidence_sources=0,
            evidence_records=0,
            decision_evidence_records=0,
            locus_linked_records=0,
            locus_linked_genes=0,
            genes_with_evidence=0,
            conflicts=0,
            policy_blocks=0,
            warnings=tuple(warnings),
            reasons=tuple(reasons),
        )

    def evaluate(self, query: LocusQuery, tool_result: ToolCallResult) -> SufficiencyReport:
        preflight = self.preflight(query)
        if not preflight.valid_input:
            return preflight
        reasons = list(preflight.reasons)
        warnings = list(preflight.warnings)
        valid_input = preflight.valid_input
        significant_signal = preflight.significant_signal

        if tool_result.status is ToolStatus.BLOCKED:
            reasons.append("POLICY_BLOCKED: benchmark leakage policy rejected the query")
        elif tool_result.status is ToolStatus.INVALID_INPUT:
            reasons.append(
                "INVALID_INPUT: " + (tool_result.error or "semantic input validation failed")
            )
            valid_input = False
        elif tool_result.status is ToolStatus.INCOMPLETE:
            limited_scope = limited_discovery_ranking_scope(query, tool_result)
            if limited_scope:
                warnings.append(
                    "Candidate-discovery warning: ranking is limited to observed candidates with verified admission and scoped evidence. "
                    "Candidate discovery remains incomplete for " + ", ".join(limited_scope)
                    + "; additional eligible genes or supporting evidence may be missing. "
                    "The retained shortlist is provisional, not a complete candidate pool."
                )
            else:
                reasons.append(
                    "INCOMPLETE: " + (tool_result.error or "evidence retrieval was incomplete")
                )
        elif tool_result.status is ToolStatus.ERROR:
            reasons.append("SERVICE_UNAVAILABLE: evidence retrieval failed")
        elif tool_result.status is ToolStatus.NO_DATA:
            reasons.append("INSUFFICIENT_EVIDENCE: no permitted evidence was retrieved")

        records = list(tool_result.records)
        region_context = region_gene_context_from_records(records)
        if query.task == "prioritize_genes":
            if region_context.status in {"unavailable", "malformed"}:
                warnings.append(
                    f"Regional-coverage warning: the declared ±{local_gene_window_label()} annotated-gene context "
                    f"was {region_context.status}; verified direct and remote locus links remain usable."
                )
            elif (
                region_context.status == "complete"
                and region_context.annotated_gene_count == 0
            ):
                warnings.append(
                    f"Regional-evidence warning: no annotated gene was found within ±{local_gene_window_label()} "
                    "of the selected locus; this does not exclude verified remote locus links."
                )
        permitted_decision_records = decision_evidence_for_query(query, records)
        candidate_set = set(query.candidate_genes)
        decision_records = (
            [item for item in permitted_decision_records if item.gene_id in candidate_set]
            if query.task == "prioritize_genes" and candidate_set
            else list(permitted_decision_records)
        )
        datasets = source_diversity([item.to_dict() for item in decision_records]).source_ids
        genes = {item.gene_id for item in decision_records if item.gene_id}
        locus_linked_records = [
            item for item in decision_records if is_direct_locus_gene_evidence(item)
        ]
        locus_linked_genes = {item.gene_id for item in locus_linked_records if item.gene_id}
        direct_datasets = source_diversity([item.to_dict() for item in locus_linked_records]).source_ids
        diversity_datasets = direct_datasets if query.task == "prioritize_genes" else datasets
        if query.task == "describe_variants":
            if not records and tool_result.status is ToolStatus.SUCCESS:
                reasons.append(
                    "INSUFFICIENT_EVIDENCE: no variant or study annotation was retrieved"
                )
        elif query.task == "gene_associations":
            gene_linked_records = [
                item for item in records if isinstance(item.gene_id, str) and item.gene_id.strip()
            ]
            if not gene_linked_records and tool_result.status is ToolStatus.SUCCESS:
                reasons.append(
                    "INSUFFICIENT_EVIDENCE: no gene-linked record was retrieved for the "
                    "selected variants"
                )
        else:
            if len(decision_records) < self.config.min_evidence_records:
                warnings.append(
                    "Evidence-volume warning: only "
                    f"{len(decision_records)} gene-linked decision record(s) were retrieved; "
                    f"{self.config.min_evidence_records} are preferred."
                )
            if len(diversity_datasets) < self.config.min_independent_datasets:
                warnings.append(
                    "Evidence-source-diversity warning: "
                    + ("qualifying direct evidence" if query.task == "prioritize_genes" else "decision evidence")
                    + f" spans only {len(diversity_datasets)} distinct source study/dataset identifier(s); "
                    + f"{self.config.min_independent_datasets} are preferred. "
                    "Source identifiers do not establish independent cohorts or populations."
                )
            if (
                query.task == "prioritize_genes"
                and len(genes) < self.config.min_genes_with_evidence
            ):
                warnings.append(
                    "Candidate-coverage warning: only "
                    f"{len(genes)} candidate gene(s) have decision-eligible evidence; "
                    f"{self.config.min_genes_with_evidence} are preferred."
                )
            if (
                query.task == "prioritize_genes"
                and self.config.require_locus_linked_gene_evidence
                and not locus_linked_genes
            ):
                warnings.append("Direct-link warning: no candidate has a verified direct locus-to-gene link; "
                                "contextual support cannot be presented as a direct locus measurement.")
            if query.task == "prioritize_genes" and not genes:
                reasons.append("INSUFFICIENT_EVIDENCE: no candidate has decision-eligible gene evidence")

        relevant_tissue_coverage: bool | None = None
        if query.relevant_tissues:
            relevant_tissue_coverage = self._has_relevant_tissue(
                locus_linked_records if query.task == "prioritize_genes" else decision_records,
                query.relevant_tissues,
                biological_context=query.biological_context,
            )
            if (
                self.config.require_relevant_tissue_if_declared
                and query.task not in {"describe_variants", "gene_associations"}
                and not relevant_tissue_coverage
            ):
                warnings.append(
                    "Biological-context warning: disease-relevant tissue/cell-type "
                    + ("coverage could not be assessed from the qualifying direct locus-to-gene evidence."
                       if relevant_tissue_coverage is None else
                       "coverage is absent from the qualifying direct locus-to-gene evidence.")
                )

        detected_conflicts = self._count_conflicts(decision_records)
        conflicts = (
            0 if query.task in {"describe_variants", "gene_associations"} else detected_conflicts
        )
        if conflicts:
            warnings.append(
                "Evidence-conflict warning: "
                f"{conflicts} candidate gene(s) have opposing evidence that must be "
                "reported in the limitations."
            )

        evidence_dimensions = tuple(sorted({record.kind.value for record in decision_records}))
        missing_relevant_tissues: tuple[str, ...] = ()
        if query.relevant_tissues and not relevant_tissue_coverage:
            missing_relevant_tissues = tuple(query.relevant_tissues)

        return SufficiencyReport(
            valid_input=valid_input,
            significant_signal=significant_signal,
            evidence_sources=len(datasets),
            direct_evidence_sources=len(direct_datasets),
            context_evidence_scope="qualifying_direct_locus_gene_evidence" if query.task == "prioritize_genes" else "decision_evidence",
            evidence_records=len(records),
            decision_evidence_records=len(decision_records),
            locus_linked_records=len(locus_linked_records),
            locus_linked_genes=len(locus_linked_genes),
            genes_with_evidence=len(genes),
            conflicts=conflicts,
            policy_blocks=len(tool_result.policy_blocks),
            relevant_tissue_coverage=relevant_tissue_coverage,
            evidence_dimensions=evidence_dimensions,
            missing_relevant_tissues=missing_relevant_tissues,
            region_gene_context_status=region_context.status,
            nearby_annotated_gene_count=region_context.annotated_gene_count,
            region_gene_context_source=region_context.source,
            warnings=tuple(dict.fromkeys(warnings)),
            reasons=tuple(reasons),
        )

    @staticmethod
    def _has_relevant_tissue(
        records: list[EvidenceRecord], relevant_tissues: tuple[str, ...],
        *, biological_context: Mapping | None = None,
    ) -> bool | None:
        plan = prepare_context_review(biological_context)
        if plan is not None:
            measured, unresolved = False, False
            for record in records:
                if record.kind not in RANKING_LOCUS_LINK_KINDS:
                    continue
                classification = review_record_context(record.to_dict(), plan)["context_class"]
                if classification == "matched":
                    return True
                measured |= classification == "other_reported_context"
                unresolved |= classification == "context_unreported"
            return False if measured and not unresolved else None
        for record in records:
            observed = evidence_context_values(record.to_dict())
            if any(context_term_matches(term, value) for term in relevant_tissues for value in observed):
                return True
        return False

    @staticmethod
    def _count_conflicts(records: list[EvidenceRecord]) -> int:
        return comparable_conflict_count([{"gene_id": record.gene_id, "kind": record.kind.value,
            "variant_id": record.variant_id, "tissue": record.tissue,
            "supports": record.supports, "payload": record.payload} for record in records])
