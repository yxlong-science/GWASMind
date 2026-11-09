from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal, InvalidOperation
from typing import Any

from ..academic_review import (
    comparable_conflict_count,
    normalize_operation_issues,
    operation_issue,
    publication_count,
)
from ..biological_context import context_term_matches, evidence_context_values
from ..configuration import parameter
from ..context_review import prepare_context_review, review_record_context
from ..coverage import (
    COMPLEX_REGION_CONFIG_VERSION,
    COMPLEX_REGION_DEFINITIONS,
    mappings_overlap_complex_region,
)
from ..evidence_sources import source_diversity
from ..ld_support import (
    LD_MHC_WARNING,
    LD_PROXY_WARNING,
    LDReportScope,
    ld_report_scope,
)
from ..mcp.normalization import evidence_kind_for_module
from ..processing_scope import RESULT_PROCESSING_LIMIT_MESSAGE, is_source_only_processing_limit
from ..region_gate import REGION_GENE_CONTEXT_DATASETS, local_gene_window_label
from ..staged_reasoning import (
    SafetyFacts,
    SafetyOutcome,
    SafetySeverity,
    configured_safety_check,
    evaluate_safety_facts,
    has_direct_locus_gene_evidence,
)
from .ranking_policy import retained_model_review

GENOME_WIDE_SIGNIFICANCE = parameter("analysis.gwas_significance_p")
_CANONICAL_VARIANT = re.compile(
    r"^(?:chr)?(?P<chromosome>[0-9]+|X|Y|MT)[_:](?P<position>\d+)"
    r"[_:](?P<reference>[ACGTN]+)[_:](?P<alternate>[ACGTN]+)$",
    re.IGNORECASE,
)
_RSID = re.compile(r"^rs[1-9]\d*$", re.IGNORECASE)
_ALLELE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)
_VALID_CHROMOSOMES = frozenset({*(str(item) for item in range(1, 23)), "X", "Y", "MT"})
_BUILD_ALIASES = {
    "grch37": "GRCh37",
    "hg19": "GRCh37",
    "grch38": "GRCh38",
    "hg38": "GRCh38",
}
_TECHNICAL_STATUS_PRIORITY = {
    "success": 0,
    "no_data": 0,
    "not_applicable": 0,
    "missing": -1,
    "unavailable": 2,
    "blocked": 3,
    "failed": 4,
    "partial": 5,
    "truncated": 6,
}
_DIRECT_KINDS = frozenset({"qtl", "colocalization", "regulatory"})
_MANDATORY_TOOLS = frozenset({"ot_resolve_entities", "ot_get_locus_context"})
_HARD_CONTEXT_MODULES = frozenset({"locus.variant"})
_HARD_LOCUS_GENE_MODULES = frozenset(
    {
        "locus.variant_consequence",
        "locus.qtl",
        "locus.colocalisation",
        "locus.enhancer_to_gene",
    }
)


@dataclass(frozen=True, slots=True)
class SelectedVariantSignal:
    variant_id: str
    p_value: float | None
    genome_wide_significant: bool | None


@dataclass(frozen=True, slots=True)
class SelectedSignalStats:
    selected_variants: int
    variants_with_p_value: int
    genome_wide_significant_variants: int
    minimum_p_value: float | None
    maximum_p_value: float | None
    measurements: tuple[SelectedVariantSignal, ...]
    significance_threshold: float = GENOME_WIDE_SIGNIFICANCE

    @property
    def has_genome_wide_signal(self) -> bool | None:
        if self.variants_with_p_value == 0:
            return None
        return self.genome_wide_significant_variants > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "has_genome_wide_signal": self.has_genome_wide_signal,
        }


@dataclass(frozen=True, slots=True)
class RuntimeIntegrity:
    variant_integrity: bool
    build_integrity: bool
    allele_integrity: bool
    issues: tuple[str, ...] = ()
    unresolved_variant_ids: tuple[str, ...] = ()
    source_variant_count: int | None = None
    valid_source_count: int | None = None
    valid_source_variant_ids: tuple[str, ...] = ()
    excluded_source_variant_ids: tuple[str, ...] = ()
    duplicate_source_variant_ids: tuple[str, ...] = ()
    conflicting_source_variant_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeRegionGeneContext:
    status: str
    annotated_gene_count: int | None
    flank_bp: int | None
    truncated: bool
    source: str | None
    evidence_id: str | None
    count_is_lower_bound: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicRetrievalMetrics:
    evidence_records: int
    candidate_genes: int
    literature_records: int
    publication_abstracts: int
    independent_evidence_sources: int
    module_record_counts: tuple[tuple[str, int], ...]
    evidence_kind_counts: tuple[tuple[str, int], ...]
    required_modules: int
    completed_required_modules: int
    no_data_required_modules: int
    incomplete_required_modules: int
    independent_loci: int | None = None
    direct_evidence_candidates: int | None = None
    selected_direct_candidates: int | None = None
    selected_l2g_candidates: int | None = None
    selected_expression_candidates: int | None = None
    candidate_fingerprint: str | None = None
    candidate_policy_version: str | None = None
    candidate_mode: str | None = None
    context_gene_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_records": self.evidence_records,
            "candidate_genes": self.candidate_genes,
            "literature_records": self.literature_records,
            "publication_abstracts": self.publication_abstracts,
            "independent_evidence_sources": self.independent_evidence_sources,
            "source_study_count": self.independent_evidence_sources,
            "module_record_counts": dict(self.module_record_counts),
            "evidence_kind_counts": dict(self.evidence_kind_counts),
            "required_modules": self.required_modules,
            "completed_required_modules": self.completed_required_modules,
            "no_data_required_modules": self.no_data_required_modules,
            "incomplete_required_modules": self.incomplete_required_modules,
            "shortlist": {
                "candidate_fingerprint": self.candidate_fingerprint,
                "policy_version": self.candidate_policy_version,
                "mode": self.candidate_mode,
                "context_gene_count": self.context_gene_count,
                "independent_loci": self.independent_loci,
                "direct_evidence_candidates": self.direct_evidence_candidates,
                "selected_direct_candidates": self.selected_direct_candidates,
                "selected_l2g_candidates": self.selected_l2g_candidates,
                "selected_expression_candidates": self.selected_expression_candidates,
            },
        }

    @property
    def compact_summary(self) -> str:
        candidate_text = (
            f" · {self.candidate_genes} shortlisted genes" if self.candidate_genes else ""
        )
        return f"{self.evidence_records} evidence records{candidate_text}"


@dataclass(frozen=True, slots=True)
class RuntimeStageBridge:
    intent: str
    signal: SelectedSignalStats
    integrity: RuntimeIntegrity
    evidence: tuple[Mapping[str, Any], ...]
    required_modules: tuple[str, ...]
    module_statuses: tuple[tuple[str, str], ...]
    region_gene_context: RuntimeRegionGeneContext
    biological_context_defined: bool
    biological_context_coverage: bool | None
    independent_dataset_count: int
    conflict_count: int
    literature_count: int
    complex_mhc: bool
    retrieval_metrics: PublicRetrievalMetrics
    ranking_repair_attempts: int
    ranking_used_fallback: bool
    ranking_validation_failures: int
    ranking_not_attempted_status: str | None = None
    ranking_unavailable_reason: str | None = None
    limited_candidate_discovery: bool = False
    ranking_projection_omitted_records: int = 0
    ld_scope: LDReportScope = LDReportScope()
    result_processing_limited: bool = False
    ranking_retained_valid_rankings: int = 0
    model_review_unavailable: bool = False
    module_warnings: tuple[tuple[str, tuple[str, ...]], ...] = ()
    candidate_gene_ids: tuple[str, ...] | None = None
    operation_issues: tuple[dict[str, Any], ...] = ()
    source_observations: tuple[Mapping[str, Any], ...] | None = None
    review_requirements: Mapping[str, Any] = field(default_factory=dict)
    candidate_audit: Mapping[str, Any] = field(default_factory=dict)
    context_terms: tuple[str, ...] = ()
    biological_context: Mapping[str, Any] = field(default_factory=dict)
    ld_metadata: Mapping[str, Any] = field(default_factory=dict)
    review_genome_build: str | None = None
    complex_region_variants: tuple[str, ...] = ()
    complex_region_names: tuple[str, ...] = ()
    source_scoped_module_statuses: tuple[tuple[str, str], ...] = ()
    optional_proxy_locus_modules: tuple[str, ...] = ()
    _review_cache: dict[str, Any] = field(default_factory=dict, init=False, compare=False, repr=False)

    @property
    def selected_source_identities_validated(self) -> bool:
        integrity = self.integrity
        return bool(integrity.variant_integrity and integrity.build_integrity and integrity.allele_integrity
                    and integrity.source_variant_count and integrity.valid_source_count == integrity.source_variant_count
                    and not integrity.unresolved_variant_ids and not integrity.excluded_source_variant_ids)

    @property
    def safety_inputs(self) -> dict[str, Any]:
        return {
            "evidence": tuple(row for row in self.evidence if self._ld_record_in_scope(row)),
            "variant_integrity": self.integrity.variant_integrity,
            "build_integrity": self.integrity.build_integrity,
            "allele_integrity": self.integrity.allele_integrity,
            "source_observation_count": self.integrity.source_variant_count,
            "valid_source_variant_count": self.integrity.valid_source_count,
            "candidate_gene_ids": self.candidate_gene_ids,
            "source_observations": self.source_observations,
            "review_requirements": self.review_requirements,
            "candidate_audit": self.review_candidate_audit,
            "context_terms": self.context_terms,
            "biological_context": self.biological_context,
            "operation_issues": self.operation_issues,
            "required_modules": self.required_modules,
            "module_statuses": self.review_module_statuses,
            "module_warnings": dict(self.module_warnings),
            "nearby_gene_count": self.region_gene_context.annotated_gene_count,
            "biological_context_defined": self.biological_context_defined,
            "biological_context_coverage": self.biological_context_coverage,
            "independent_dataset_count": self.independent_dataset_count,
            "conflict_count": self.conflict_count,
            "complex_region": (
                ", ".join(self.complex_region_names)
                if self.complex_region_names else None
            ),
            "literature_count": self.literature_count,
            "intent": self.intent,
            "gwas_signal_present": self.signal.has_genome_wide_signal,
            "source_data_available": bool(
                self.signal.selected_variants and self.integrity.variant_integrity
                and self.integrity.build_integrity and self.integrity.allele_integrity
            ),
        }

    @property
    def review_module_statuses(self) -> dict[str, str]:
        statuses = {key: value for key, value in self.module_statuses if key != "candidate_ranking"}
        if self.selected_source_identities_validated:
            statuses.update(dict(self.source_scoped_module_statuses))
        # This final all-candidate receipt is stronger than historical request errors.
        # Context definition / expression comparisons do not establish retrieval completion.
        if self.candidate_audit.get("expression_retrieval_complete") is True:
            for module in ("target.expression", "target.baseline_expression"):
                if statuses.get(module) in {"failed", "error", "partial", "missing", "unavailable"}:
                    statuses[module] = "success"
        return statuses

    @property
    def review_candidate_audit(self) -> dict[str, Any]:
        audit = dict(self.candidate_audit)
        # Preserve an explicit LD search limitation even when candidate summaries
        # omit it. This qualifies an existing W1; it does not introduce a W7.
        if self.ld_metadata.get("complete") is False and str(self.ld_metadata.get("status", "")).lower() != "not_requested":
            audit["ld_expansion_complete"] = False
        coverage = audit.get("locus_discovery_coverage")
        if self.selected_source_identities_validated and self.optional_proxy_locus_modules and isinstance(coverage, Mapping):
            gaps = {module for category in ("failed", "truncated", "blocked", "unavailable")
                    for module in _sequence_value(coverage.get(category)) if isinstance(module, str)}
            if (gaps and gaps.issubset(self.optional_proxy_locus_modules)
                    and not audit.get("locus_discovery_next_cursor")):
                audit["locus_discovery_complete"] = True
                audit["ld_proxy_retrieval_complete"] = False
        return audit

    def _ld_record_in_scope(self, row: Mapping[str, Any]) -> bool:
        return self.ld_scope.permits_record(row)

    @property
    def retrieval_coverage(self) -> dict[str, Any]:
        statuses = {key: value for key, value in self.module_statuses if key != "candidate_ranking"}
        return {
            "module_statuses": statuses,
            "required_modules": list(self.required_modules),
            "complete": bool(statuses) and all(
                status in {"success", "no_data", "not_applicable"} for status in statuses.values()
            ),
        }

    @property
    def exploratory_ranking_allowed(self) -> bool:
        # A variant-description review does not evaluate gene-ranking gates.
        # Reuse the same observed evidence/identity/region with the ranking
        # policy instead of interpreting a description PASS as rank permission.
        if self.result_processing_limited:
            return False
        if "exploratory" not in self._review_cache:
            facts = (self.evaluate_safety() if self.intent == "prioritize_genes" else
                     evaluate_safety_facts(**{**self.safety_inputs, "intent": "prioritize_genes"}))
            # Contextual and functional gene evidence is eligible for
            # exploratory ranking when the task does not explicitly require
            # a direct variant-to-gene link.  Gate1 records that requirement
            # in ``facts.ranking_allowed``; requiring ``direct_evidence_ids``
            # here would silently reintroduce the old W1 behaviour.
            self._review_cache["exploratory"] = bool(facts.ranking_allowed)
        return self._review_cache["exploratory"]

    def evaluate_safety(self) -> SafetyFacts:
        # A bridge owns one frozen acquired scope. Subsequent stages reuse its
        # facts; a dataclass replacement starts with a fresh cache.
        if "facts" not in self._review_cache:
            facts = self._evaluate_safety_uncached()
            if facts.review_scope:
                scope = {key: value for key, value in facts.review_scope.items() if key != "sha256"}
                matched_regions = [dict(row) for row in COMPLEX_REGION_DEFINITIONS
                                   if row["name"] in self.complex_region_names]
                # Keep the established MHC scalar fields for saved-review
                # compatibility while exposing the complete versioned registry.
                mhc = next((row for row in matched_regions if row["name"] == "MHC/HLA"), None)
                scope["complex_region_definition"] = {
                    "version": COMPLEX_REGION_CONFIG_VERSION,
                    "name": (mhc["description"] if mhc else ", ".join(self.complex_region_names)),
                    "names": list(self.complex_region_names),
                    "regions": matched_regions,
                    "genome_build": "GRCh38", "chromosome": mhc["chromosome"] if mhc else None,
                    "start": mhc["start"] if mhc else None, "end": mhc["end"] if mhc else None,
                    "definition_source": mhc["source"] if mhc else None,
                    "analysis_genome_build": self.review_genome_build,
                    "overlap_count": len(self.complex_region_variants),
                    "overlap_examples": list(self.complex_region_variants[:32]),
                    "overlap_sha256": hashlib.sha256("\n".join(self.complex_region_variants).encode()).hexdigest(),
                }
                if self.ld_scope.provided:
                    scope["ld_reference"] = self.ld_scope.prompt_summary(link_limit=0)
                    scope["ld_review_policy"] = {
                        "proxy_expansion": "optional",
                        "selected_source_identities_validated": self.selected_source_identities_validated,
                        "optional_proxy_locus_modules": list(self.optional_proxy_locus_modules),
                    }
                    scope["ld_proxy_identity_binding"] = {
                        "count": len(self.ld_scope.proxy_ids),
                        "sha256": hashlib.sha256("\n".join(sorted(self.ld_scope.proxy_ids)).encode()).hexdigest(),
                    }
                scope["sha256"] = hashlib.sha256(json.dumps(scope, sort_keys=True, ensure_ascii=False,
                    separators=(",", ":")).encode()).hexdigest()
                facts = replace(facts, review_scope=scope)
            self._review_cache["facts"] = facts
        return self._review_cache["facts"]

    def _evaluate_safety_uncached(self) -> SafetyFacts:
        facts = evaluate_safety_facts(**self.safety_inputs)
        if self.review_genome_build != "GRCh38":
            facts = replace(facts, checks=tuple(replace(check,
                outcome=SafetyOutcome.NOT_ASSESSED,
                fact="The configured recognized complex-region registry is defined on GRCh38; compatible analyzed coordinates were not established for this region assessment.",
                reason_codes=("complex_region_build_unassessed",))
                if check.check_id == "complex_region" else check for check in facts.checks))
        elif self.complex_region_variants:
            proxy_overlap = bool(set(self.complex_region_variants).intersection(self.ld_scope.proxy_ids))
            source_overlap = bool(set(self.complex_region_variants).difference(self.ld_scope.proxy_ids))
            matched_regions = [row for row in COMPLEX_REGION_DEFINITIONS
                               if row["name"] in self.complex_region_names]
            region_text = "; ".join(
                f"{row['name']} chr{row['chromosome']}:{row['start']}-{row['end']}"
                for row in matched_regions
            ) or ", ".join(self.complex_region_names)
            detail = (f"The configured GRCh38 recognized complex region(s) {region_text} "
                      f"{len(self.complex_region_variants)} analyzed variant(s) overlap it; see the affected variants. "
                      "Missing records may reflect region-specific representation or database coverage.")
            facts = replace(facts, checks=tuple(replace(check,
                outcome=check.outcome if source_overlap else SafetyOutcome.NOTE,
                fact=((LD_MHC_WARNING + " ") if proxy_overlap and "MHC/HLA" in self.complex_region_names else "") + detail,
                affected={"variant_ids": self.complex_region_variants},
                refs=tuple({"type": "variant", "id": identifier} for identifier in self.complex_region_variants[:32]))
                if check.check_id == "complex_region" else check for check in facts.checks))
        if self.model_review_unavailable:
            facts = replace(facts, operation_issues=(*facts.operation_issues, operation_issue(
                "MODEL_REVIEW_UNAVAILABLE", category="review_unavailable", scope="review", blocking=False,
                message="Model-assisted review was unavailable; program-verified facts remain available.",
                reasons=("model_review_execution",))), checks=(*facts.checks, configured_safety_check(
                check_id="model_review_execution", label="Model-assisted review execution",
                severity=SafetySeverity.WARNING, outcome=SafetyOutcome.WARN,
                fact=("Program evidence checks completed, but the model-assisted review was unavailable. "
                      "The report uses the program-validated evidence and discloses the remaining limitations."),
            )))
        if self.result_processing_limited:
            # The empty processed ledger cannot characterize the retained full
            # result. Only input, operational scope, and source-GWAS checks apply.
            scientific_coverage = {
                "locus_genes_1mb", "direct_locus_gene_evidence",
                "biological_context_coverage", "evidence_independence",
                "evidence_conflicts", "literature_coverage", "optional_evidence_coverage",
            }
            return replace(facts, candidate_ranking_status="not_applicable",
                source_scope=None, checks=tuple(
                    replace(check, fact=RESULT_PROCESSING_LIMIT_MESSAGE)
                    if check.check_id == "required_modules" else
                    replace(check, fact="Uploaded GWAS observations are available for source-only interpretation.")
                    if check.check_id == "evidence_availability" and self.integrity.variant_integrity else
                    check for check in facts.checks if check.check_id not in scientific_coverage
                ))
        if self.integrity.unresolved_variant_ids:
            facts = replace(
                facts,
                checks=tuple(
                    replace(check, fact=(
                        "Reference and alternate alleles were validated for the resolved "
                        "variants used in external evidence retrieval; unresolved source "
                        "observations remain unvalidated and were excluded from that retrieval."
                    )) if check.check_id == "allele_integrity" and self.integrity.allele_integrity
                    else check for check in facts.checks
                ) + (configured_safety_check(
                    check_id="variant_resolution_coverage",
                    label="Variant resolution coverage",
                    severity=SafetySeverity.WARNING,
                    outcome=SafetyOutcome.WARN,
                    fact=(
                        "The completed entity resolver could not uniquely identify "
                        + ", ".join(self.integrity.unresolved_variant_ids)
                        + ". Their source observations are preserved, but absence of external "
                        "records for these variants is not evidence that they are unreported."
                    ),
                ),),
            )
        if self.region_gene_context.count_is_lower_bound:
            facts = replace(
                facts,
                checks=tuple(
                    replace(
                        check,
                        fact=(
                            "Exact selected-variant transcript annotations establish at least "
                            f"{self.region_gene_context.annotated_gene_count} gene(s) within {local_gene_window_label()}. "
                            "This proves nearby-gene existence; the count is a lower bound, "
                            "not a complete regional census."
                        ),
                    )
                    if check.check_id == "locus_genes_1mb"
                    else check
                    for check in facts.checks
                ),
            )
        if self.intent != "prioritize_genes":
            return facts
        ld_missing_rsid_only = False
        if self.ld_scope.provided:
            extra_checks = []
            if self.ld_scope.proxy_ids:
                extra_checks.append(configured_safety_check(check_id="ld_proxy_scope", label="LD-proxy support",
                    severity=SafetySeverity.WARNING, outcome=SafetyOutcome.PASS, fact=LD_PROXY_WARNING))
                # Preserve the ordered audit tuple, but index membership once:
                # repeated tuple searches become quadratic for large ledgers.
                direct_evidence_ids = frozenset(facts.direct_evidence_ids)
                proxy_evidence = sum(row.get("evidence_id") in direct_evidence_ids
                    and self.ld_scope.evidence_origin(row) == "ld_proxy" for row in self.evidence)
                if proxy_evidence:
                    facts = replace(facts, checks=tuple(replace(check, fact=(
                        f"{len(facts.direct_evidence_ids)} gene-linked molecular evidence record(s) were found, "
                        f"including {proxy_evidence} at verified LD proxies. Proxy-to-source support is "
                        "indirect through population-specific LD."
                    )) if check.check_id == "direct_locus_gene_evidence" else check for check in facts.checks))
            from ..review_coverage import assess_ld_coverage
            ld_assessment = assess_ld_coverage(asdict(self.ld_scope), self.ld_metadata,
                selected_source_identities_validated=self.selected_source_identities_validated)
            ld_missing_rsid_only = ld_assessment.get("missing_rsid_only") is True
            if ld_assessment["state"] in {"warning", "unknown", "note"}:
                affected = ld_assessment["affected"]
                ids = sorted(set((*affected["source_ids"], *affected["proxy_ids"], *affected["reported_proxy_ids"])))
                extra_checks.append(configured_safety_check(check_id="ld_proxy_coverage", label="LD reference and proxy coverage",
                    severity=SafetySeverity.WARNING,
                    outcome={"warning": SafetyOutcome.WARN, "note": SafetyOutcome.NOTE,
                             "unknown": SafetyOutcome.NOT_ASSESSED}[ld_assessment["state"]],
                    fact=" ".join(ld_assessment["facts"]) or "Completion of the declared LD reference scope could not be assessed.",
                    reason_codes=ld_assessment["reason_codes"] or ("ld_scope_unassessed",),
                    impact=("Proxy expansion is supplementary. Unverified proxies remain excluded; source observations and verified proxy evidence remain usable."
                            if ld_assessment.get("optional_proxy_scope") else
                            "Retained source and verified proxy evidence remains usable; additional linked candidates may be absent."),
                    affected={"variant_ids": ids}, refs=[{"type": "variant", "id": identifier} for identifier in ids[:24]]))
            facts = replace(facts, checks=(*facts.checks, *extra_checks))
        if self.limited_candidate_discovery and not self.candidate_audit and not any(
                check.check_id == "candidate_discovery_coverage" for check in facts.checks):
            facts = replace(facts, checks=(*facts.checks, configured_safety_check(
                check_id="candidate_discovery_coverage", label="Candidate-discovery coverage",
                severity=SafetySeverity.WARNING, outcome=SafetyOutcome.WARN,
                fact=("Candidate discovery is incomplete. Ranking is limited to the directly "
                      "supported candidates retrieved so far; additional candidates may be missing."),
            )))
            unrelated_gaps = any(check.nonpassing and check.check_id in {
                "required_modules", "optional_evidence_coverage", "variant_resolution_coverage"}
                for check in facts.checks)
            ld_only_gap = not unrelated_gaps and (ld_missing_rsid_only or any(check.check_id == "ld_proxy_coverage"
                and check.outcome in {SafetyOutcome.WARN, SafetyOutcome.NOTE} for check in facts.checks))
            if ld_only_gap and self.selected_source_identities_validated:
                facts = replace(facts, checks=tuple(replace(check, outcome=SafetyOutcome.NOTE,
                    fact="Candidate discovery retains the source and verified supplementary proxy evidence; unverified proxy additions are excluded.")
                    if check.check_id == "candidate_discovery_coverage" else check for check in facts.checks))
            facts = replace(facts, ld_only_discovery_gap=ld_only_gap)
        ranking_status = dict(self.module_statuses).get("candidate_ranking", "missing")
        facts = replace(facts, candidate_ranking_status=ranking_status)
        if self.ranking_not_attempted_status:
            return replace(facts, checks=(*facts.checks, configured_safety_check(
                check_id="ranking_not_attempted", label="Ranking execution",
                severity=SafetySeverity.WARNING, outcome=SafetyOutcome.WARN,
                fact=("Candidate ranking was not started because the available evidence did not "
                      "satisfy its requirements. Retrieved findings remain available; no ranking-model "
                      "response was generated or rejected."),
            )))
        if ranking_status in {"not_applicable", "deferred"}:
            # Upstream hard gates stopped the workflow before response
            # generation.  Do not mislabel an intentionally skipped model call
            # as a schema/evidence-link validation failure.
            return facts
        ranking_passed = ranking_status == "success"
        failed_twice = self.ranking_validation_failures >= 2
        if ranking_status == "unknown":
            ranking_fact = "Candidate-ranking execution status could not be verified; no validated ranking output is available."
        elif ranking_status == "partial":
            ranking_fact = (
                "Candidate ranking is available; Gate 2 corrected or removed unsupported statements "
                "and retained the model's original gene order."
            )
        elif ranking_passed and self.ranking_used_fallback and self.ranking_retained_valid_rankings:
            ranking_fact = (
                f"{self.ranking_retained_valid_rankings} validated model ranking entries and their "
                "original ranking positions were retained; evidence-based rules filled the remaining "
                "positions. This combined ranking orders available support; its scores are not "
                "causal probabilities."
            )
        elif ranking_passed and self.ranking_used_fallback:
            ranking_fact = (
                "Candidate genes were ordered with a deterministic evidence-based method "
                "after the ranking synthesis could not be validated. This method ranks "
                "available support; its scores are not causal probabilities."
            )
        elif ranking_passed and self.ranking_repair_attempts:
            ranking_fact = (
                "The first candidate-ranking response failed validation, then one bounded "
                "repair passed deterministic schema and evidence-link checks."
            )
        elif ranking_passed:
            ranking_fact = (
                "Candidate-ranking output passed deterministic schema and evidence-link validation."
            )
        elif failed_twice:
            ranking_fact = (
                "Candidate-ranking response validation failed twice (the initial response and "
                "one bounded repair)."
            )
        else:
            ranking_fact = (
                "Candidate-ranking execution did not produce a validated output; "
                "retrieved evidence remains available for deterministic reporting."
            )
        if self.ranking_projection_omitted_records and (
            not self.ranking_used_fallback or self.ranking_retained_valid_rankings
        ):
            ranking_fact += (f" Model context capacity omitted {self.ranking_projection_omitted_records} deduplicated evidence records. "
                "The full ledger was retained and checked; the model comparison used its disclosed evidence subset.")
        unavailable_reason = _RANKING_UNAVAILABLE_REASONS.get(self.ranking_unavailable_reason or "")
        unavailable = () if ranking_passed or ranking_status == "partial" else (configured_safety_check(
            check_id="ranking_unavailable", label="Ranking availability",
            severity=SafetySeverity.WARNING, outcome=SafetyOutcome.WARN,
            fact=(f"Candidate ranking is unavailable because {unavailable_reason}; retrieved evidence is retained."
                  if unavailable_reason else "Candidate ranking remains unavailable; retrieved evidence is retained."),
            reason_codes=(self.ranking_unavailable_reason,) if unavailable_reason else (),
        ),)
        return replace(
            facts,
            checks=(
                *facts.checks,
                *unavailable,
                configured_safety_check(
                    check_id="model_output_grounding",
                    label="Model-output grounding",
                    severity=SafetySeverity.WARNING,
                    outcome=(
                        SafetyOutcome.PASS
                        if ranking_passed and not self.ranking_used_fallback and not self.ranking_projection_omitted_records
                        else SafetyOutcome.WARN
                    ),
                    fact=ranking_fact,
                    reason_codes=("ranking_hallucination_controlled",) if ranking_status == "partial" else (),
                ),
            ),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "selected_signal": self.signal.to_dict(),
            "integrity": self.integrity.to_dict(),
            "region_gene_context": self.region_gene_context.to_dict(),
            "biological_context": {
                "defined": self.biological_context_defined,
                "covered": self.biological_context_coverage,
                "coverage_scope": "qualifying_direct_locus_gene_evidence" if self.intent == "prioritize_genes" else "retrieved_evidence",
            },
            "candidate_ranking_status": dict(self.module_statuses).get("candidate_ranking"),
            "ranking_not_attempted_status": self.ranking_not_attempted_status,
            "ranking_retained_valid_rankings": self.ranking_retained_valid_rankings,
            "limited_candidate_discovery": self.limited_candidate_discovery,
            "result_processing_limited": self.result_processing_limited,
            "complex_mhc": self.complex_mhc,
            "complex_region_names": list(self.complex_region_names),
            "retrieval": self.retrieval_metrics.to_dict(),
            "retrieval_coverage": self.retrieval_coverage,
            "exploratory_ranking_allowed": self.exploratory_ranking_allowed,
        }


def build_runtime_stage_bridge(
    *,
    observations: Sequence[Mapping[str, Any]],
    progress_events: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    intent: str | Any,
    biological_context: Mapping[str, Any] | Any,
    genome_build: str | None,
    required_modules: Sequence[str] | None = None,
    query_index: int | None = None,
    question: str = "",
    review_requirements: Mapping[str, Any] | None = None,
) -> RuntimeStageBridge:
    """Convert local runtime artifacts into Stage 2 metrics and Stage 3 facts.

    Progress coverage, not a fluent model description, determines whether required modules
    completed. A completed regional summary with zero genes is kept distinct from an absent,
    malformed, unavailable, or truncated regional query.
    """

    normalized_intent = _intent_value(intent)
    processing_limited = is_source_only_processing_limit(result)
    observation_rows = tuple(dict(item) for item in observations if isinstance(item, Mapping))
    evidence = _evidence_rows(result)
    signal = _selected_signal_stats(observation_rows)
    event_rows = tuple(
        item
        for item in (() if processing_limited else progress_events)
        if isinstance(item, Mapping)
        and (
            query_index is None
            or item.get("query_index") is None
            or item.get("query_index") == query_index
        )
    )
    derived_required, statuses, module_counts, shortlist = _coverage_from_progress(
        event_rows,
        intent=normalized_intent,
    )
    if processing_limited:
        derived_required, statuses = ("result.processing",), {"result.processing": "partial"}
        module_counts, shortlist = Counter(), {}
    resolution = result.get("variant_resolution")
    normalizations = resolution.get("normalizations", ()) if isinstance(resolution, Mapping) else ()
    identity_context = variant_identity_context(observations=observation_rows, evidence=evidence,
        genome_build=genome_build, progress_events=event_rows, normalizations=normalizations)
    aliases = identity_context["validated_aliases"]
    omitted = _explicitly_unresolved_variants(observation_rows, event_rows, statuses)
    # Every source observation that remains unresolved after the identity
    # context (including a completed resolver with no unique match) is a
    # selected-source coverage gap. Proxy-only identity failures are recorded
    # in the LD ledger and never enter this tuple.
    omitted = tuple(dict.fromkeys((*omitted, *(row["variant_id"] for row in identity_context["unresolved_variants"]))))
    # Another resolved source can retrieve a profile that lists the omitted
    # rsID as an alias. That does not resolve the original multiallelic choice.
    aliases = {source: canonical for source, canonical in aliases.items() if source not in omitted}
    ld_scope = LDReportScope() if processing_limited else ld_report_scope(result, observation_rows, normalized_intent)
    ld_scope = replace(ld_scope, source_ids=tuple(dict.fromkeys((*ld_scope.source_ids, *aliases.values()))))
    source_scoped_statuses, optional_proxy_locus_modules = {}, []
    if ld_scope.valid and ld_scope.provided:
        from ..review_coverage import source_scoped_coverage_events
        selected_ids = [aliases.get(row.get("variant_id"), row.get("variant_id")) for row in observation_rows]
        review_events, optional_proxy_locus_modules = source_scoped_coverage_events(event_rows,
            result.get("ld_expansion") or {}, source_ids=selected_ids, proxy_ids=ld_scope.proxy_ids)
        _, source_scoped_statuses, _, _ = _coverage_from_progress(review_events, intent=normalized_intent)
    integrity = _runtime_integrity(
        observation_rows, genome_build=genome_build,
        validated_aliases=aliases, unresolved_variant_ids=omitted,
    )
    retrieval = result.get("retrieval") if isinstance(result.get("retrieval"), Mapping) else {}
    raw_operations = retrieval.get("operation_issues", result.get("operation_issues"))
    operations = list(normalize_operation_issues(raw_operations))
    if integrity.conflicting_source_variant_ids:
        operations.append(operation_issue("SYSTEM_ERROR", category="system_error", scope="input", blocking=True,
            message="Different source observations assign conflicting measurements to the same variant identity.",
            reasons=("conflicting_source_observations",)))
    confirmed_excluded = set()
    if isinstance(raw_operations, (list, tuple)):
        for issue in raw_operations:
            if not isinstance(issue, Mapping) or issue.get("code") not in {"S1", "source_variants_excluded"}:
                continue
            excluded = issue.get("confirmed_invalid_sources")
            if isinstance(excluded, (list, tuple)):
                confirmed_excluded.update(row["variant_id"] for row in excluded
                    if isinstance(row, Mapping) and isinstance(row.get("variant_id"), str))
    if confirmed_excluded:
        valid_sources = tuple(identifier for identifier in integrity.valid_source_variant_ids
                              if aliases.get(identifier, identifier) not in confirmed_excluded)
        excluded_sources = tuple(identifier for identifier in integrity.valid_source_variant_ids
                                 if aliases.get(identifier, identifier) in confirmed_excluded)
        integrity = replace(integrity, valid_source_count=len(valid_sources), valid_source_variant_ids=valid_sources,
            excluded_source_variant_ids=tuple(dict.fromkeys((*integrity.excluded_source_variant_ids, *excluded_sources))),
            unresolved_variant_ids=tuple(dict.fromkeys((*integrity.unresolved_variant_ids, *excluded_sources))))
    excluded_ids = set(integrity.excluded_source_variant_ids)
    invalid_owned_records = any(row.get("variant_id") in excluded_ids for row in evidence)
    if invalid_owned_records:
        operations.append(operation_issue("SYSTEM_ERROR", category="system_error", scope="retrieval",
            blocking=True, message="Retrieved evidence is bound to an excluded source identity; the evidence context must be rebuilt.",
            reasons=("excluded_source_evidence_ownership",)))
    candidate_audit = retrieval.get("candidate_audit") if isinstance(retrieval.get("candidate_audit"), Mapping) else {}
    if not candidate_audit and isinstance(result.get("candidate_audit"), Mapping):
        candidate_audit = result["candidate_audit"]
    if isinstance(candidate_audit.get("summary"), Mapping):
        candidate_audit = candidate_audit["summary"]
    candidate_audit = dict(candidate_audit)
    for phase, details in map(_phase_details, event_rows):
        if phase in {"candidate_universe_discovered", "candidate_universe_explicit"}:
            for key in ("candidate_discovery_complete", "locus_discovery_complete", "colocalisation_detail_complete",
                        "expression_retrieval_complete", "expression_assessment_complete", "expression_discovery_complete",
                        "ld_expansion_complete", "ld_proxy_retrieval_complete", "molecular_context_complete"):
                if key not in candidate_audit and isinstance(details.get(key), bool):
                    candidate_audit[key] = details[key]
            if "locus_discovery_coverage" not in candidate_audit and isinstance(details.get("locus_discovery_coverage"), Mapping):
                candidate_audit["locus_discovery_coverage"] = dict(details["locus_discovery_coverage"])
                candidate_audit["locus_discovery_next_cursor"] = details.get("locus_discovery_next_cursor")
            # Stored gene-window card views keep only a summary; the window
            # universe remains authoritative for which genes can be ranked.
            event_ids = details.get("candidate_ids", details.get("targets"))
            if ("candidate_ids" not in candidate_audit and details.get("mode") == "gene_centric_window"
                    and isinstance(event_ids, (list, tuple))
                    and all(isinstance(value, str) and value.strip() for value in event_ids)):
                candidate_audit["candidate_ids"] = list(event_ids)
    from ..review_coverage import assess_candidate_coloc_coverage

    coloc_coverage = assess_candidate_coloc_coverage(candidate_audit)
    if coloc_coverage is not None:
        retained_coloc = any(row.get("kind") == "colocalization"
                            or (row.get("payload") or {}).get("module") == "locus.colocalisation"
                            for row in evidence)
        statuses["locus.colocalisation"] = (("success" if retained_coloc else "no_data")
            if coloc_coverage["complete"] else coloc_coverage["module_status"])
        candidate_audit["colocalisation_detail_complete"] = coloc_coverage["complete"]
    candidate_ids = candidate_audit.get("candidate_ids")
    if not isinstance(candidate_ids, (list, tuple)) or not all(isinstance(value, str) and value.strip() for value in candidate_ids):
        candidate_ids = None
    required = (
        tuple(dict.fromkeys(_module_identifier(item) for item in required_modules))
        if required_modules is not None and not processing_limited
        else derived_required
    )
    for module in required:
        statuses.setdefault(module, "missing")

    # Direct evidence channels are complementary.  Once at least one channel
    # returns locus-linked records, a failure in another channel becomes a
    # disclosed coverage warning instead of an analysis-wide hard stop.
    if normalized_intent == "prioritize_genes" and has_direct_locus_gene_evidence(evidence):
        required = tuple(
            module
            for module in required
            if not (
                module in _HARD_LOCUS_GENE_MODULES
                and statuses.get(module, "missing") not in {"success", "no_data", "not_applicable"}
            )
        )

    region_context = _region_gene_context(evidence)
    ranking_not_attempted_status = None
    if normalized_intent == "prioritize_genes" and not processing_limited:
        regional_module = "regional_gene_context"
        if regional_module not in required:
            required = (*required, regional_module)
        statuses[regional_module] = _region_technical_status(region_context)
        # Ranking validation is a post-retrieval hard gate, not an MCP module.
        # Keep its technical status for diagnostics without inflating Stage 2
        # retrieval-completeness metrics.
        retrieval_ready = all(
            statuses.get(module, "missing") in {"success", "no_data", "not_applicable"}
            for module in required
        )
        attempted = _ranking_was_attempted(result, event_rows)
        if not attempted:
            ranking_not_attempted_status = next((
                str(details.get("status")) for phase, details in map(_phase_details, reversed(event_rows))
                if phase == "ranking_not_attempted" and details.get("ranking_attempted") is False and details.get("status") in {
                    "INCOMPLETE", "INSUFFICIENT_EVIDENCE", "NO_GWAS_SIGNAL", "INVALID_INPUT",
                    "POLICY_BLOCKED", "SERVICE_UNAVAILABLE", "CONFLICTING_EVIDENCE",
                }
            ), None)
        statuses["candidate_ranking"] = (
            _candidate_ranking_technical_status(result, evidence) if attempted
            else "not_applicable" if ranking_not_attempted_status
            else "deferred" if str(result.get("status")) == "EVIDENCE_RETRIEVED" and retrieval_ready
            else "not_applicable" if not retrieval_ready or str(result.get("status")) in {
                "EVIDENCE_RETRIEVED", "INSUFFICIENT_EVIDENCE", "INVALID_INPUT", "NO_GWAS_SIGNAL", "SERVICE_UNAVAILABLE",
            } else "unknown"
        )

    context_value = _context_mapping(biological_context)
    relevant_terms = _relevant_context_terms(context_value)
    declared_context = context_value.get("defined")
    context_defined = (
        declared_context if isinstance(declared_context, bool) else bool(relevant_terms)
    )
    context_coverage = _context_coverage(
        result,
        evidence,
        context_defined=context_defined,
        relevant_terms=relevant_terms,
        direct_scope=normalized_intent == "prioritize_genes",
        biological_context=context_value,
    )
    independent_sources = _independent_dataset_count(result, evidence, direct_scope=normalized_intent == "prioritize_genes")
    conflicts = _conflict_count(result, evidence)
    literature = _literature_count(evidence)
    publication_abstracts = _literature_abstract_count(evidence)
    review_build = _normalize_build(genome_build)
    region_observations = [dict(row, variant_id=aliases.get(row.get("variant_id"), row.get("variant_id")))
                           for row in observation_rows if row.get("variant_id") in integrity.valid_source_variant_ids]
    if ld_scope.valid:
        region_observations.extend(ld_scope.proxy_observations)
    region_variants = tuple(sorted({str(row.get("variant_id")) for row in region_observations
                                    if row.get("variant_id") and _complex_mhc((row,))})) if review_build == "GRCh38" else ()
    complex_region_names = tuple(sorted({name for row in region_observations
                                         for name in _complex_region_names((row,), review_build)}))
    complex_mhc = "MHC/HLA" in complex_region_names
    ranking_repair_attempts, ranking_used_fallback = _ranking_recovery(
        event_rows,
        result=result,
    )
    ranking_validation_failures = _ranking_validation_failure_count(result)
    sufficiency = result.get("sufficiency")
    sufficiency = sufficiency if isinstance(sufficiency, Mapping) else {}
    metrics = _retrieval_metrics(
        evidence=evidence,
        module_counts=module_counts,
        shortlist=shortlist,
        required_modules=required,
        module_statuses=statuses,
        independent_sources=independent_sources,
        literature_count=literature,
        publication_abstracts=publication_abstracts,
    )
    ordered_statuses = tuple(sorted((module, status) for module, status in statuses.items()))
    from ..review_policy import review_requirements as make_review_requirements
    requirements = make_review_requirements(question, review_requirements)
    module_warnings = dict(_literature_module_warnings_from_progress(event_rows))
    retained_warnings = tuple(value for value in _sequence_value(retrieval.get("warnings")) if isinstance(value, str))
    for module in ("evidence.europepmc", "target.literature"):
        if module in statuses and retained_warnings:
            module_warnings[module] = tuple(dict.fromkeys((*module_warnings.get(module, ()), *retained_warnings)))
    return RuntimeStageBridge(
        intent=normalized_intent,
        signal=signal,
        integrity=integrity,
        candidate_gene_ids=tuple(candidate_ids) if candidate_ids is not None else None,
        operation_issues=tuple(operations),
        evidence=evidence,
        required_modules=required,
        module_statuses=ordered_statuses,
        module_warnings=tuple(sorted(module_warnings.items())),
        source_observations=observation_rows,
        review_requirements=requirements,
        candidate_audit=candidate_audit,
        context_terms=relevant_terms,
        biological_context=deepcopy(context_value),
        ld_metadata=dict(result.get("ld_expansion") or {}) if isinstance(result.get("ld_expansion"), Mapping) else {},
        region_gene_context=region_context,
        biological_context_defined=context_defined,
        biological_context_coverage=context_coverage,
        independent_dataset_count=independent_sources,
        conflict_count=conflicts,
        literature_count=literature,
        complex_mhc=complex_mhc,
        review_genome_build=review_build,
        complex_region_variants=region_variants,
        complex_region_names=complex_region_names,
        source_scoped_module_statuses=tuple(sorted(source_scoped_statuses.items())),
        optional_proxy_locus_modules=tuple(optional_proxy_locus_modules),
        retrieval_metrics=metrics,
        ranking_repair_attempts=ranking_repair_attempts,
        ranking_used_fallback=ranking_used_fallback,
        ranking_validation_failures=ranking_validation_failures,
        ranking_projection_omitted_records=max((
            _nonnegative_integer((details.get("gate2_review") or {}).get("projection_omitted_records")) or 0
            for phase, details in map(_phase_details, event_rows)
            if phase == "synthesis_completed" and isinstance(details.get("gate2_review"), Mapping)
        ), default=0),
        ranking_not_attempted_status=ranking_not_attempted_status,
        ranking_unavailable_reason=_ranking_unavailable_reason(result, event_rows),
        ld_scope=ld_scope,
        result_processing_limited=processing_limited,
        limited_candidate_discovery=any(
            isinstance(warning, str) and warning.startswith("Candidate-discovery warning:")
            for warning in _sequence_value(sufficiency.get("warnings"))
        ),
        ranking_retained_valid_rankings=_retained_model_ranking_count(event_rows),
        model_review_unavailable=(
            isinstance(result.get("safety_review_execution"), Mapping)
            and result["safety_review_execution"].get("model_review_status") == "unavailable"
        ),
    )


def build_runtime_safety_facts(**kwargs: Any) -> SafetyFacts:
    """One-call server convenience wrapper around :func:`build_runtime_stage_bridge`."""

    return build_runtime_stage_bridge(**kwargs).evaluate_safety()


def _selected_signal_stats(
    observations: Sequence[Mapping[str, Any]],
) -> SelectedSignalStats:
    values = tuple(
        value
        for item in observations
        for value in (_valid_p_value(item.get("p_value")),)
        if value is not None
    )
    return SelectedSignalStats(
        selected_variants=len(observations),
        variants_with_p_value=len(values),
        genome_wide_significant_variants=sum(value <= GENOME_WIDE_SIGNIFICANCE for value in values),
        minimum_p_value=min(values) if values else None,
        maximum_p_value=max(values) if values else None,
        measurements=tuple(
            SelectedVariantSignal(
                variant_id=str(item.get("variant_id") or "").strip(),
                p_value=_valid_p_value(item.get("p_value")),
                genome_wide_significant=(
                    _valid_p_value(item.get("p_value")) <= GENOME_WIDE_SIGNIFICANCE
                    if _valid_p_value(item.get("p_value")) is not None
                    else None
                ),
            )
            for item in observations
        ),
    )


def _runtime_integrity(
    observations: Sequence[Mapping[str, Any]],
    *,
    genome_build: str | None,
    validated_aliases: Mapping[str, str] | None = None,
    unresolved_variant_ids: tuple[str, ...] = (),
) -> RuntimeIntegrity:
    """Validate each source independently; a valid subset can proceed.

    Empty OT annotation is never consulted as proof of invalid source identity.
    Unresolved/invalid rows remain in the original source selection but cannot
    supply external evidence bindings. Global build errors invalidate every row.
    """
    legacy = _runtime_integrity_observations(observations, genome_build=genome_build,
        validated_aliases=validated_aliases, unresolved_variant_ids=unresolved_variant_ids)
    valid, excluded, duplicate_aliases = [], [], []
    seen_identities = {}
    conflicting = set()
    for row in observations:
        identifier = str(row.get("variant_id") or "")
        current = _runtime_integrity_observations((row,), genome_build=genome_build,
            validated_aliases=validated_aliases,
            unresolved_variant_ids=(identifier,) if identifier in unresolved_variant_ids else ())
        canonical = (validated_aliases or {}).get(identifier, identifier)
        match = _CANONICAL_VARIANT.fullmatch(canonical)
        identity = (f"{_normalize_chromosome(match.group('chromosome'))}_{int(match.group('position'))}_"
                    f"{match.group('reference').upper()}_{match.group('alternate').upper()}") if match else canonical.casefold()
        if (identifier not in unresolved_variant_ids
                and current.variant_integrity and current.build_integrity and current.allele_integrity):
            signature = tuple((key, type(row[key]).__name__, str(row[key])) for key in (
                "p_value", "beta", "standard_error", "odds_ratio", "effect_allele", "other_allele",
                "effect_allele_frequency", "sample_size") if key in row)
            if identity in seen_identities:
                first_id, first_signature = seen_identities[identity]
                if signature != first_signature:
                    conflicting.update((first_id, identifier))
                else:
                    duplicate_aliases.append(identifier)
            else:
                valid.append(identifier)
                seen_identities[identity] = (identifier, signature)
        else:
            excluded.append(identifier)
    valid = [identifier for identifier in valid if identifier not in conflicting]
    excluded.extend(sorted(conflicting))
    return replace(legacy, variant_integrity=legacy.variant_integrity or bool(valid),
        build_integrity=legacy.build_integrity or bool(valid), allele_integrity=legacy.allele_integrity or bool(valid),
        source_variant_count=len(observations), valid_source_count=len(valid),
        valid_source_variant_ids=tuple(valid), excluded_source_variant_ids=tuple(excluded),
        duplicate_source_variant_ids=tuple(duplicate_aliases),
        conflicting_source_variant_ids=tuple(sorted(conflicting)),
        unresolved_variant_ids=tuple(dict.fromkeys((*unresolved_variant_ids, *excluded))))


def _runtime_integrity_observations(
    observations: Sequence[Mapping[str, Any]], *, genome_build: str | None,
    validated_aliases: Mapping[str, str] | None = None,
    unresolved_variant_ids: tuple[str, ...] = (),
) -> RuntimeIntegrity:
    issues: list[str] = []
    if not observations:
        return RuntimeIntegrity(False, False, False, ("no_selected_variants",))
    identifiers: list[str] = []
    variant_valid = True
    allele_valid = True
    normalized_build = _normalize_build(genome_build)
    if normalized_build is None:
        issues.append("genome_build_unverified")
    for index, item in enumerate(observations):
        identifier = str(item.get("variant_id") or "").strip()
        match = _CANONICAL_VARIANT.fullmatch(identifier)
        rsid = bool(_RSID.fullmatch(identifier))
        if not identifier or (match is None and not rsid):
            variant_valid = False
            issues.append(f"variant_{index + 1}_identifier_invalid")
        chromosome = _normalize_chromosome(item.get("chromosome"))
        position = _positive_integer(item.get("position"))
        if item.get("chromosome") is not None and chromosome is None:
            variant_valid = False
            issues.append(f"variant_{index + 1}_chromosome_invalid")
        if item.get("position") is not None and position is None:
            variant_valid = False
            issues.append(f"variant_{index + 1}_position_invalid")
        if match is not None:
            canonical_chromosome = _normalize_chromosome(match.group("chromosome"))
            canonical_position = _positive_integer(match.group("position"))
            if canonical_chromosome is None or canonical_position is None:
                variant_valid = False
                issues.append(f"variant_{index + 1}_canonical_coordinate_invalid")
            if chromosome is not None and chromosome != canonical_chromosome:
                variant_valid = False
                issues.append(f"variant_{index + 1}_chromosome_mismatch")
            if position is not None and position != canonical_position:
                variant_valid = False
                issues.append(f"variant_{index + 1}_position_mismatch")
            reference = match.group("reference")
            alternate = match.group("alternate")
            identifiers.append(f"{canonical_chromosome}_{canonical_position}_{reference.upper()}_{alternate.upper()}")
            for keys, expected in (
                (("reference_allele", "referenceAllele", "ref", "REF"), reference),
                (("alternate_allele", "alternateAllele", "alt", "ALT"), alternate),
            ):
                for key in keys:
                    if key in item and item[key] is not None and str(item[key]).strip().casefold() != expected.casefold():
                        allele_valid = False
                        issues.append(f"variant_{index + 1}_allele_mismatch")
        else:
            identifiers.append(identifier.casefold())
            reference = _first_text(
                item,
                ("reference_allele", "referenceAllele", "ref", "REF"),
            )
            alternate = _first_text(
                item,
                ("alternate_allele", "alternateAllele", "alt", "ALT"),
            )
            verified = (validated_aliases or {}).get(identifier)
            verified_match = _CANONICAL_VARIANT.fullmatch(verified or "")
            if rsid and verified_match is not None:
                reference = reference or verified_match.group("reference")
                alternate = alternate or verified_match.group("alternate")
        explicitly_omitted = rsid and identifier in unresolved_variant_ids
        if rsid and validated_aliases is not None and identifier not in validated_aliases and not explicitly_omitted:
            allele_valid = False
            issues.append(f"variant_{index + 1}_identity_unverified")
        if not _valid_allele_pair(reference, alternate) and not explicitly_omitted:
            allele_valid = False
            issues.append(f"variant_{index + 1}_alleles_invalid_or_unverified")
        raw_item_build = item.get("genome_build") or item.get("genomeBuild") or item.get("build")
        item_build = _normalize_build(raw_item_build)
        if raw_item_build is not None and item_build is None:
            issues.append(f"variant_{index + 1}_genome_build_invalid")
            normalized_build = None
        if (
            item_build is not None
            and normalized_build is not None
            and item_build != normalized_build
        ):
            issues.append(f"variant_{index + 1}_genome_build_mismatch")
            normalized_build = None
        raw_p = item.get("p_value")
        if raw_p is not None and _valid_p_value(raw_p) is None:
            variant_valid = False
            issues.append(f"variant_{index + 1}_p_value_invalid")
    if len(identifiers) != len(set(identifiers)):
        variant_valid = False
        issues.append("duplicate_variant_identifiers")
    if unresolved_variant_ids and not validated_aliases:
        allele_valid = False
        issues.append("no_validated_variant_identity")
    return RuntimeIntegrity(
        variant_integrity=variant_valid,
        build_integrity=normalized_build is not None,
        allele_integrity=allele_valid,
        issues=tuple(dict.fromkeys(issues)),
        unresolved_variant_ids=unresolved_variant_ids,
    )


def validated_variant_aliases(
    *, observations: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]], genome_build: str | None,
    normalizations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, str]:
    """Match source IDs to actual variant identities without changing source data.

    Search candidates and positional proximity are insufficient. Each alias must
    occur in a provenance-complete locus.variant record, whose canonical ID,
    coordinates and alleles agree with the record and supplied GRCh38 observation.
    A multiallelic rsID remains unresolved unless source alleles disambiguate it.
    """
    if _normalize_build(genome_build) != "GRCh38":
        return {}
    identities: list[tuple[str, set[str], Mapping[str, Any]]] = []
    for record in evidence:
        payload = record.get("payload")
        if (record.get("kind") != "variant_annotation"
                or record.get("provenance_complete") is not True
                or not isinstance(payload, Mapping) or payload.get("module") != "locus.variant"):
            continue
        identifier = str(record.get("variant_id") or "")
        match = _CANONICAL_VARIANT.fullmatch(identifier)
        context = payload.get("context")
        subject = payload.get("subject")
        if (match is None or not isinstance(context, Mapping)
                or not isinstance(subject, Mapping) or subject.get("id") != identifier):
            continue
        expected = (match.group("chromosome"), int(match.group("position")),
                    match.group("reference").upper(), match.group("alternate").upper())
        actual = (_normalize_chromosome(context.get("chromosome")),
                  _positive_integer(context.get("position")),
                  str(context.get("reference_allele") or "").upper(),
                  str(context.get("alternate_allele") or "").upper())
        if actual != expected or not _valid_allele_pair(actual[2], actual[3]):
            continue
        aliases = context.get("rs_ids")
        aliases = {str(v).casefold() for v in aliases} if isinstance(aliases, (list, tuple)) else set()
        upstream = payload.get("upstream")
        raw = upstream.get("raw") if isinstance(upstream, Mapping) else None
        if isinstance(raw, Mapping):
            raw_expected = {"id": identifier, "chromosome": expected[0],
                            "position": expected[1], "referenceAllele": expected[2],
                            "alternateAllele": expected[3]}
            if any(key in raw and str(raw[key]).casefold() != str(val).casefold()
                   for key, val in raw_expected.items()):
                continue
            if "rsIds" in raw:
                values = raw["rsIds"]
                if not isinstance(values, (list, tuple)):
                    continue
                aliases.intersection_update(str(v).casefold() for v in values)
        identities.append((identifier, aliases, context))
    proofs = _validated_source_normalizations(observations, genome_build, normalizations)
    resolved = {row["original_id"]: row["canonical_id"] for row in proofs}
    for observation in observations:
        original = str(observation.get("variant_id") or "")
        candidates = set()
        for identifier, aliases, context in identities:
            if original != identifier and original.casefold() not in aliases:
                continue
            if observation.get("chromosome") is not None and (
                _normalize_chromosome(observation["chromosome"])
                != _normalize_chromosome(context["chromosome"])
            ):
                continue
            if observation.get("position") is not None and (
                _positive_integer(observation["position"]) != _positive_integer(context["position"])
            ):
                continue
            if any(
                observation.get(key) is not None
                and str(observation[key]).upper() != str(context[field]).upper()
                for field, keys in (
                    ("reference_allele", ("reference_allele", "referenceAllele", "ref", "REF")),
                    ("alternate_allele", ("alternate_allele", "alternateAllele", "alt", "ALT")),
                ) for key in keys
            ):
                continue
            candidates.add(identifier)
        if len(candidates) == 1:
            resolved[original] = next(iter(candidates))
    return resolved


def _validated_source_normalizations(observations, genome_build, normalizations=(), progress_events=()):
    from ..source_normalization import validate_source_normalization

    if _normalize_build(genome_build) != "GRCh38":
        return []
    sources = {str(row.get("variant_id") or ""): row for row in observations}
    values = list(normalizations) if isinstance(normalizations, (list, tuple)) else []
    for event in progress_events:
        phase, details = _phase_details(event)
        if phase != "source_identity_normalization_applied":
            continue
        scope = details.get("source_variant_ids")
        if not isinstance(scope, list) or not all(isinstance(value, str) for value in scope) or set(scope) != set(sources):
            continue
        values.extend(_sequence_value(details.get("normalizations")))
    validated = {}
    for proof in values:
        if not validate_source_normalization(proof) or proof["original_id"] not in sources:
            continue
        row = sources[proof["original_id"]]
        integrity = _runtime_integrity((row,), genome_build=genome_build)
        if integrity.variant_integrity and integrity.build_integrity and integrity.allele_integrity:
            validated[proof["original_id"]] = dict(proof)
    return [validated[key] for key in sorted(validated)]


def variant_identity_context(
    *, observations: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]],
    genome_build: str | None, progress_events: Sequence[Mapping[str, Any]] = (),
    normalizations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return actual validated aliases and explicit coverage gaps for reports.

    Callers must retain source observations separately. A resolver's candidate
    list never becomes a validated alias unless locus.variant confirms it.
    """
    proofs = _validated_source_normalizations(observations, genome_build, normalizations, progress_events)
    mapping = validated_variant_aliases(
        observations=observations, evidence=evidence, genome_build=genome_build,
        normalizations=proofs,
    )
    # Canonical source IDs already specify coordinate and alleles. A completed
    # catalogue lookup may legitimately return no annotation for that identity;
    # keep this distinct from an unresolved rsID without inventing an upstream
    # alias validation record.
    identified = {}
    for observation in observations:
        original = str(observation.get("variant_id") or "")
        if original in mapping:
            integrity = _runtime_integrity(
                (observation,), genome_build=genome_build, validated_aliases=mapping,
            )
            if integrity.variant_integrity and integrity.build_integrity and integrity.allele_integrity:
                identified[original] = mapping[original]
                continue
        match = _CANONICAL_VARIANT.fullmatch(original)
        if match is None or _normalize_build(genome_build) != "GRCh38":
            continue
        source_integrity = _runtime_integrity((observation,), genome_build=genome_build)
        if (source_integrity.variant_integrity and source_integrity.build_integrity
                and source_integrity.allele_integrity):
            identified[original] = (
                f"{match.group('chromosome').upper()}_{int(match.group('position'))}_"
                f"{match.group('reference').upper()}_{match.group('alternate').upper()}"
            )
    reasons = {}
    expected = {str(item.get("variant_id") or "") for item in observations}
    for event in progress_events:
        phase, details = _phase_details(event)
        sources = details.get("source_variant_ids")
        if (phase not in {"entity_resolution_applied", "source_identity_normalization_applied"} or not isinstance(sources, list)
                or not all(isinstance(v, str) for v in sources)
                or set(sources) != expected):
            continue
        if phase == "source_identity_normalization_applied":
            for identifier in _sequence_value(details.get("conflicting_source_variant_ids")):
                if isinstance(identifier, str) and identifier in expected:
                    reasons[identifier] = "normalization_conflict"
            continue
        for item in _sequence_value(details.get("omitted_variants")):
            if isinstance(item, Mapping) and item.get("variant_id") in expected:
                reasons[item["variant_id"]] = str(item.get("reason") or "unresolved")
    # Explicit omissions for this source selection take precedence over the
    # visible ledger's incidental aliases: only one competing allele may have
    # been retrieved through a different, already resolved source observation.
    for original in reasons:
        mapping.pop(original, None)
        identified.pop(original, None)
    unresolved = [
        {"variant_id": str(item.get("variant_id") or ""),
         "reason": reasons.get(str(item.get("variant_id") or ""), "no_validated_variant_identity")}
        for item in observations if str(item.get("variant_id") or "") not in identified
    ]
    return {"validated_aliases": mapping, "unresolved_variants": unresolved,
            "identified_variant_ids": list(dict.fromkeys(identified.values())),
            "source_to_identified_variant": identified,
            "normalizations": [row for row in proofs if row["original_id"] not in reasons],
            "source_variant_count": len(observations), "validated_variant_count": len(mapping),
            "identified_source_variant_count": len(identified),
            "complete": not unresolved}


def _explicitly_unresolved_variants(observations, events, statuses) -> tuple[str, ...]:
    if statuses.get("entity.resolve") != "success":
        return ()
    expected = {str(item.get("variant_id") or "") for item in observations}
    omitted = set()
    for event in events:
        phase, details = _phase_details(event)
        if phase != "entity_resolution_applied":
            continue
        if details.get("allow_partial_variant_resolution") is not True:
            continue
        supplied = details.get("source_variant_ids")
        if (not isinstance(supplied, list) or not all(isinstance(v, str) for v in supplied)
                or set(supplied) != expected):
            continue
        values = details.get("omitted_variant_ids")
        if isinstance(values, list) and all(isinstance(v, str) and v in expected for v in values):
            omitted.update(values)
    return tuple(str(item["variant_id"]) for item in observations if item.get("variant_id") in omitted)


def _literature_module_warnings_from_progress(
    events: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Bind recorded tool warnings only to explicitly requested literature modules."""
    result: dict[str, list[str]] = {}
    for event in events:
        phase, details = _phase_details(event)
        if phase != "semantic_tool_finished":
            continue
        coverage = details.get("coverage")
        if not isinstance(coverage, Mapping):
            continue
        warnings = [item for item in _sequence_value(details.get("warnings")) if isinstance(item, str)]
        for module in _sequence_value(coverage.get("requested")):
            if module not in ("evidence.europepmc", "target.literature", "publication.metadata"):
                continue
            result.setdefault(module, []).extend(warnings)
    return tuple((module, tuple(dict.fromkeys(warnings))) for module, warnings in sorted(result.items()))


def _coverage_from_progress(
    events: Sequence[Mapping[str, Any]],
    *,
    intent: str,
) -> tuple[tuple[str, ...], dict[str, str], Counter[str], dict[str, Any]]:
    required: list[str] = []
    statuses: dict[str, str] = {}
    module_counts: Counter[str] = Counter()
    starts: dict[tuple[Any, str], tuple[tuple[str, ...], bool]] = {}
    shortlist: dict[str, Any] = {
        "candidate_genes": None,
        "independent_loci": None,
        "direct_evidence_candidates": None,
        "selected_direct_candidates": None,
        "selected_l2g_candidates": None,
    }
    for event in events:
        phase, details = _phase_details(event)
        if phase == "candidate_expression_started":
            phase = "semantic_tool_started"
        elif phase == "candidate_expression_finished":
            phase = "semantic_tool_finished"
        tool = str(details.get("tool") or "").strip()
        step = _nonnegative_integer(details.get("step"))
        key = (step, tool)
        if phase == "semantic_tool_started":
            modules = tuple(
                _module_identifier(item)
                for item in _sequence_value(details.get("modules"))
                if isinstance(item, str) and item.strip()
            )
            inferred_required = tool in _MANDATORY_TOOLS or (
                tool == "ot_get_locus_gene_evidence" and intent != "describe_variants"
            )
            is_required = details.get("required") is True or (
                details.get("required") is None and inferred_required
            )
            if is_required and not modules:
                modules = ("locus.variant",) if tool == "ot_get_locus_context" else ("unrecognized_required_module",)
            starts[key] = (modules, is_required)
            required.extend(_required_modules_for_tool(tool, modules, is_required=is_required))
            for module in modules:
                statuses.setdefault(module, "missing")
            continue
        if phase == "semantic_tool_finished":
            modules, is_required = starts.get(key, ((), False))
            if key not in starts:
                is_required = tool in _MANDATORY_TOOLS or (
                    tool == "ot_get_locus_gene_evidence" and intent != "describe_variants"
                )
            coverage = details.get("coverage")
            coverage = coverage if isinstance(coverage, Mapping) else {}
            requested = tuple(
                _module_identifier(item)
                for item in _sequence_value(coverage.get("requested", modules))
                if isinstance(item, str) and item.strip()
            )
            if is_required and not requested:
                requested = modules or (("locus.variant",) if tool == "ot_get_locus_context" else ("unrecognized_required_module",))
            required.extend(_required_modules_for_tool(tool, requested, is_required=is_required))
            category_statuses = (
                ("completed", "success"),
                ("no_data", "no_data"),
                ("unavailable", "unavailable"),
                ("blocked", "blocked"),
                ("failed", "failed"),
                ("truncated", "truncated"),
            )
            observed_coverage = False
            for category, status in category_statuses:
                rows = coverage.get(category, ())
                if not isinstance(rows, (list, tuple)):
                    continue
                for item in rows:
                    if not isinstance(item, str) or not item.strip():
                        continue
                    observed_coverage = True
                    _update_status(statuses, _module_identifier(item), status)
            if not observed_coverage:
                fallback = _tool_status(str(details.get("status") or ""))
                for module in requested or modules:
                    _update_status(statuses, module, fallback)
            counts = details.get("module_counts")
            if isinstance(counts, Mapping):
                for module, raw_count in counts.items():
                    if not isinstance(module, str) or not _known_evidence_module(module):
                        continue
                    count = _nonnegative_integer(raw_count)
                    if count is not None:
                        module_counts[module] += count
            continue
        if phase in {"candidate_universe_discovered", "candidate_universe_explicit"}:
            shortlist["candidate_fingerprint"] = details.get("candidate_fingerprint")
            shortlist["candidate_policy_version"] = (details.get("policy") or {}).get("version")
            shortlist["candidate_mode"] = details.get("mode")
            shortlist["context_gene_count"] = _metric_integer(details.get("context_gene_count"))
            targets = details.get("targets")
            fallback_count = len(targets) if isinstance(targets, list) else None
            shortlist["candidate_genes"] = _metric_integer(
                details.get("shortlist_size"), fallback=fallback_count
            )
            for name in (
                "independent_loci",
                "direct_evidence_candidates",
                "selected_direct_candidates",
                "selected_l2g_candidates",
                "selected_expression_candidates",
            ):
                shortlist[name] = _metric_integer(details.get(name))
    return (
        tuple(dict.fromkeys(required)),
        statuses,
        module_counts,
        shortlist,
    )


def _required_modules_for_tool(
    tool: str,
    modules: Sequence[str],
    *,
    is_required: bool,
) -> tuple[str, ...]:
    """Return only the submodules whose technical completion is a hard gate.

    Semantic plan steps are intentionally coarse: one required locus-context call may
    contain optional study/credible-set/fine-mapping projections, and the required
    locus-to-gene call also carries optional L2G shortlist evidence.  Treating every
    projection as a hard gate would turn a bounded optional truncation into a false
    analysis stop.  Scientific sufficiency is evaluated separately from this technical
    completion check.
    """

    if not is_required:
        return ()
    normalized = tuple(dict.fromkeys(_module_identifier(item) for item in modules))
    if tool == "ot_get_locus_context":
        selected = tuple(item for item in normalized if item in _HARD_CONTEXT_MODULES)
        # The planner normally requests locus.variant.  Preserve fail-closed behavior
        # if a future provider exposes a different sole operational context module.
        return selected or normalized[:1]
    if tool == "ot_get_locus_gene_evidence":
        return tuple(item for item in normalized if item in _HARD_LOCUS_GENE_MODULES)
    return normalized


def _region_gene_context(
    evidence: Sequence[Mapping[str, Any]],
) -> RuntimeRegionGeneContext:
    rows = []
    for record in evidence:
        if str(record.get("dataset") or "") not in REGION_GENE_CONTEXT_DATASETS:
            continue
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if payload.get("relation") == "regional_gene_context_summary":
            rows.append((record, payload))
    if not rows:
        return RuntimeRegionGeneContext("absent", None, None, False, None, None)
    if len(rows) != 1:
        return RuntimeRegionGeneContext("malformed", None, None, False, None, None)
    record, payload = rows[0]
    derived = payload.get("derived")
    measurements = payload.get("measurements")
    context = payload.get("context")
    derived = derived if isinstance(derived, Mapping) else {}
    measurements = measurements if isinstance(measurements, Mapping) else {}
    context = context if isinstance(context, Mapping) else {}
    status = str(derived.get("status") or "malformed").strip().casefold()
    count = _nonnegative_integer(measurements.get("annotated_gene_count"))
    flank = _nonnegative_integer(context.get("flank_bp"))
    truncated = derived.get("truncated") is True
    if status == "complete" and count is None:
        status = "malformed"
    lower_bound = derived.get("count_is_lower_bound") is True
    if status == "nonempty" and (not lower_bound or count is None or count < 1):
        status = "malformed"
    return RuntimeRegionGeneContext(
        status=status,
        annotated_gene_count=count if status in {"complete", "nonempty"} else None,
        flank_bp=flank,
        truncated=truncated,
        source=str(record.get("source") or "").strip() or None,
        evidence_id=str(record.get("evidence_id") or "").strip() or None,
        count_is_lower_bound=lower_bound and status == "nonempty",
    )


def _region_technical_status(context: RuntimeRegionGeneContext) -> str:
    if context.truncated:
        return "truncated"
    return {
        "complete": "success",
        "nonempty": "success",
        "unavailable": "unavailable",
        "absent": "missing",
        "malformed": "failed",
    }.get(context.status, "failed")


def _ranking_was_attempted(result: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        result.get("ranked_genes") or str(result.get("status")) == "PRIORITIZED"
        or _ranking_validation_failure_count(result)
        or any(_ranking_recovery(events, result=result))
        or any(_phase_details(event)[0] in {"synthesizing_decision", "synthesis_completed"} for event in events)
    )


def _candidate_ranking_technical_status(
    result: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
) -> str:
    if "status" not in result or "ranked_genes" not in result:
        return "missing"
    if str(result.get("status") or "").strip().upper() != "PRIORITIZED":
        return "failed"
    rows = result.get("ranked_genes")
    if not isinstance(rows, list) or not rows:
        return "failed"
    evidence_by_id = {
        str(record.get("evidence_id") or "").strip(): record
        for record in evidence
        if str(record.get("evidence_id") or "").strip()
    }
    evidence_ids = set(evidence_by_id)
    retained_review = retained_model_review(result)
    genes: set[str] = set()
    for expected_rank, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            return "failed"
        gene_id = str(row.get("gene_id") or "").strip()
        cited = row.get("evidence_ids")
        rationale = row.get("rationale")
        limitations = row.get("limitations")
        if (
            not gene_id
            or (not retained_review and gene_id in genes)
            or isinstance(row.get("rank"), bool)
            or not isinstance(row.get("rank"), int)
            or row["rank"] < 1
            or (not retained_review and row.get("rank") != expected_rank)
            # Native AgentResult.to_dict() preserves these tuple fields; JSON
            # workers deserialize them as lists. Validate both representations.
            or not isinstance(cited, (list, tuple))
            or (not retained_review and not cited)
            or any(not isinstance(item, str) or not item.strip() for item in cited)
            or (not retained_review and len(cited) != len(set(cited)))
            or (not retained_review and not set(cited).issubset(evidence_ids))
            or not isinstance(rationale, str)
            or not rationale.strip()
            or not isinstance(limitations, (list, tuple))
            or any(not isinstance(item, str) for item in limitations)
        ):
            return "failed"
        if retained_review:
            continue
        cited_gene_ids = {
            str(evidence_by_id[evidence_id].get("gene_id") or "").strip()
            for evidence_id in cited
            if str(evidence_by_id[evidence_id].get("gene_id") or "").strip()
        }
        if gene_id not in cited_gene_ids or cited_gene_ids.difference({gene_id}):
            return "failed"
        genes.add(gene_id)
    return "partial" if retained_review else "success"


def _retained_model_ranking_count(events: Sequence[Mapping[str, Any]]) -> int:
    """Use the final completed ranking's count, never a stale earlier attempt."""
    for event in reversed(events):
        phase, details = _phase_details(event)
        if phase != "synthesis_completed":
            continue
        review = details.get("gate2_review")
        if not isinstance(review, Mapping) or review.get("validation_passed") is False:
            return 0
        count = _nonnegative_integer(review.get("retained_valid_rankings"))
        return count if count is not None else 0
    return 0


def _ranking_recovery(
    events: Sequence[Mapping[str, Any]],
    *,
    result: Mapping[str, Any],
) -> tuple[int, bool]:
    repair_attempts = 0
    used_fallback = False
    for event in events:
        phase, details = _phase_details(event)
        if phase != "synthesis_completed":
            continue
        repairs = _nonnegative_integer(details.get("repair_attempts"))
        if repairs is not None:
            repair_attempts = max(repair_attempts, repairs)
        used_fallback = used_fallback or details.get("deterministic_fallback") is True
    sufficiency = result.get("sufficiency")
    warnings = sufficiency.get("warnings") if isinstance(sufficiency, Mapping) else ()
    if isinstance(warnings, (list, tuple)):
        for value in warnings:
            warning = str(value or "").casefold()
            if not warning.startswith("candidate-ranking warning:"):
                continue
            if "deterministic evidence-based fallback" in warning:
                used_fallback = True
            elif "one bounded repair" in warning:
                repair_attempts = max(repair_attempts, 1)
    return repair_attempts, used_fallback


def _ranking_validation_failure_count(result: Mapping[str, Any]) -> int:
    """Recover the bounded validation-attempt count from public result metadata."""

    sufficiency = result.get("sufficiency")
    if not isinstance(sufficiency, Mapping):
        return 0
    reasons = sufficiency.get("reasons")
    if not isinstance(reasons, (list, tuple)):
        return 0
    count = 0
    for value in reasons:
        reason = str(value or "").strip().casefold()
        if not reason:
            continue
        if (
            "response validation failure after two attempts" in reason
            or "failed again after one repair attempt" in reason
            or "model output repair failed after one attempt" in reason
        ):
            return 2
        if "output validation failed" in reason or "response validation" in reason:
            count = max(count, 1)
    return count


_RANKING_UNAVAILABLE_REASONS = {
    "verified_ranking_model_unavailable": "no ranking model is configured",
    "ranking_worker_incomplete": "the ranking worker did not finish",
    "json_structure": "the model output did not satisfy the required structure after one repair",
    "candidate_contract": "the model did not return a decision for every candidate gene",
    "timeout": "the ranking model did not respond within the time limit after retries",
    "connection": "the ranking model service could not be reached after retries",
    "provider_unavailable": "the ranking model service was temporarily unavailable after retries",
    "rate_limit": "the ranking model service rate-limited the request after retries",
    "authentication": "the ranking model rejected the configured credentials",
    "context_capacity_rejected": "the evidence exceeded the ranking model's input capacity",
    "request_rejected": "the ranking model rejected the request",
    "model_identity": "the responding model did not match the configured model",
    "ranking_output_budget_exhausted": "the model reached its single-request output limit",
    "ranking_total_output_budget_exhausted": "the model reached its cumulative output limit",
    "ranking_deadline_exhausted": "the model reached its total time limit",
    "ranking_provider_budget_violation": "the provider did not honour the requested output limit",
    "unexpected_failure": "an unexpected ranking-model error occurred",
}


def _ranking_unavailable_reason(result: Mapping[str, Any], event_rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the classified cause of an unavailable ranking, never provider text."""
    decision = result.get("decision") if isinstance(result.get("decision"), Mapping) else {}
    reviews = [decision.get("gate2_review"), result.get("ranking_review"), *(
        details.get("gate2_review") for phase, details in map(_phase_details, reversed(event_rows))
        if phase == "synthesis_completed")]
    for review in reviews:
        if not isinstance(review, Mapping):
            continue
        for key in ("resource_limit", "failure_category"):
            if review.get(key) in _RANKING_UNAVAILABLE_REASONS:
                return str(review[key])
    for code in _sequence_value(decision.get("reason_codes")):
        if code in _RANKING_UNAVAILABLE_REASONS:
            return str(code)
    return None


def _context_coverage(
    result: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    *,
    context_defined: bool,
    relevant_terms: tuple[str, ...],
    direct_scope: bool = True,
    biological_context: Mapping[str, Any] | None = None,
) -> bool | None:
    if not context_defined:
        return None
    context_plan = prepare_context_review(biological_context)
    if context_plan is not None:
        measured, unresolved = False, False
        for record in evidence:
            if str(record.get("kind")) not in {"qtl", "colocalization", "regulatory"}:
                continue
            if direct_scope and not has_direct_locus_gene_evidence((record,)):
                continue
            match = review_record_context(record, context_plan)
            if match["context_class"] == "matched":
                return True
            measured |= match["context_class"] == "other_reported_context"
            unresolved |= match["context_class"] == "context_unreported"
        return False if measured and not unresolved else None
    if not relevant_terms:
        return None
    for record in evidence:
        if direct_scope and not has_direct_locus_gene_evidence((record,)):
            continue
        observed = _record_context_strings(record)
        if any(context_term_matches(term, value) for term in relevant_terms for value in observed):
            return True
    return False


def _independent_dataset_count(
    result: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]], *, direct_scope: bool = True,
) -> int:
    """Legacy name for anchored source identifiers, not cohorts or categories."""
    records = [record for record in evidence if not direct_scope or has_direct_locus_gene_evidence((record,))]
    return source_diversity(records).count


def _conflict_count(result: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> int:
    return comparable_conflict_count(evidence)


def _literature_count(evidence: Sequence[Mapping[str, Any]]) -> int:
    return publication_count(evidence)


def _literature_abstract_count(evidence: Sequence[Mapping[str, Any]]) -> int:
    values: set[str] = set()
    for index, record in enumerate(evidence):
        if str(record.get("kind") or "").casefold() != "literature":
            continue
        abstract = _nested_first_text(
            record.get("payload"),
            {"abstract", "abstracttext", "abstract_text"},
        )
        if not abstract:
            continue
        pmid = str(record.get("pmid") or "").strip()
        if not pmid:
            pmid = _nested_first_text(record.get("payload"), {"pmid", "pubmedid"})
        values.add("pmid:" + pmid if pmid else "record:" + str(index))
    return len(values)


def _retrieval_metrics(
    *,
    evidence: Sequence[Mapping[str, Any]],
    module_counts: Counter[str],
    shortlist: Mapping[str, Any],
    required_modules: Sequence[str],
    module_statuses: Mapping[str, str],
    independent_sources: int,
    literature_count: int,
    publication_abstracts: int,
) -> PublicRetrievalMetrics:
    kind_counts = Counter(
        str(record.get("kind") or "unknown").strip().casefold() for record in evidence
    )
    candidate_count = shortlist.get("candidate_genes")
    if candidate_count is None:
        candidate_count = len(
            {
                str(record.get("gene_id") or "").strip()
                for record in evidence
                if str(record.get("gene_id") or "").strip()
            }
        )
    required_statuses = [module_statuses.get(module, "missing") for module in required_modules]
    return PublicRetrievalMetrics(
        evidence_records=len(evidence),
        candidate_genes=int(candidate_count),
        literature_records=literature_count,
        publication_abstracts=publication_abstracts,
        independent_evidence_sources=independent_sources,
        module_record_counts=tuple(sorted(module_counts.items())),
        evidence_kind_counts=tuple(sorted(kind_counts.items())),
        required_modules=len(required_modules),
        completed_required_modules=sum(
            status in {"success", "no_data", "not_applicable"} for status in required_statuses
        ),
        no_data_required_modules=sum(status == "no_data" for status in required_statuses),
        incomplete_required_modules=sum(
            status not in {"success", "no_data", "not_applicable"} for status in required_statuses
        ),
        independent_loci=shortlist.get("independent_loci"),
        direct_evidence_candidates=shortlist.get("direct_evidence_candidates"),
        selected_direct_candidates=shortlist.get("selected_direct_candidates"),
        selected_l2g_candidates=shortlist.get("selected_l2g_candidates"),
        selected_expression_candidates=shortlist.get("selected_expression_candidates"),
        candidate_fingerprint=shortlist.get("candidate_fingerprint"),
        candidate_policy_version=shortlist.get("candidate_policy_version"),
        candidate_mode=shortlist.get("candidate_mode"),
        context_gene_count=shortlist.get("context_gene_count"),
    )


def _evidence_rows(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    for key in ("evidence", "evidence_ledger"):
        value = result.get(key)
        if isinstance(value, list):
            return tuple(dict(item) for item in value if isinstance(item, Mapping))
    return ()


def _phase_details(event: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    phase = str(event.get("phase") or "").strip()
    details = event.get("details")
    return phase, details if isinstance(details, Mapping) else event


def _update_status(statuses: dict[str, str], module: str, status: str) -> None:
    previous = statuses.get(module)
    if previous is None or _TECHNICAL_STATUS_PRIORITY.get(status, 10) > (
        _TECHNICAL_STATUS_PRIORITY.get(previous, 10)
    ):
        statuses[module] = status


def _tool_status(value: str) -> str:
    normalized = value.strip().casefold()
    return {
        "success": "success",
        "no_data": "no_data",
        "unavailable": "unavailable",
        "blocked": "blocked",
        "error": "failed",
        "invalid_input": "failed",
        "partial": "partial",
    }.get(normalized, "missing")


def _known_evidence_module(module: str) -> bool:
    try:
        evidence_kind_for_module(module)
    except ValueError:
        return False
    return True


def _module_identifier(value: Any) -> str:
    text = str(value).strip().casefold().replace(" ", "_")
    if re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,199}", text):
        return text
    return "unrecognized_required_module"


def _intent_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().casefold()


def _context_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    return {}


def _relevant_context_terms(value: Mapping[str, Any]) -> tuple[str, ...]:
    terms: list[str] = []
    for key in ("relevant_tissues", "organs", "tissues", "cell_types"):
        rows = value.get(key)
        if isinstance(rows, (list, tuple)):
            terms.extend(str(item).strip() for item in rows if str(item).strip())
    return tuple(dict.fromkeys(terms))


def _record_context_strings(record: Mapping[str, Any]) -> tuple[str, ...]:
    return evidence_context_values(record)


def _complex_region_names(
    observations: Sequence[Mapping[str, Any]], genome_build: str | None = "GRCh38"
) -> tuple[str, ...]:
    """Return names of configured recognized regions hit by analyzed mappings."""
    if genome_build != "GRCh38":
        return ()
    derived = []
    for item in observations:
        chromosome = item.get("chromosome")
        position = item.get("position")
        match = _CANONICAL_VARIANT.fullmatch(str(item.get("variant_id") or "").strip())
        if match is not None:
            chromosome = chromosome or match.group("chromosome")
            position = position or _positive_integer(match.group("position"))
        derived.append({"chromosome": chromosome, "position": position})
    return tuple(sorted({row["name"] for row in mappings_overlap_complex_region(
        derived, genome_build=genome_build
    )}))


def _complex_mhc(observations: Sequence[Mapping[str, Any]]) -> bool:
    """Backward-compatible predicate now covering the curated complex registry."""
    return bool(_complex_region_names(observations))


def _nested_text_values(value: Any, keys: set[str], *, depth: int = 0) -> tuple[str, ...]:
    if depth > 7:
        return ()
    values: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in keys:
                if isinstance(item, (list, tuple)):
                    values.extend(str(row).strip() for row in item if str(row).strip())
                elif item is not None and str(item).strip():
                    values.append(str(item).strip())
            values.extend(_nested_text_values(item, keys, depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for item in value[:1_000]:
            values.extend(_nested_text_values(item, keys, depth=depth + 1))
    return tuple(values)


def _nested_first_text(value: Any, keys: set[str], *, depth: int = 0) -> str:
    if depth > 7:
        return ""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in keys and item is not None and str(item).strip():
                return str(item).strip()
            nested = _nested_first_text(item, keys, depth=depth + 1)
            if nested:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value[:1_000]:
            nested = _nested_first_text(item, keys, depth=depth + 1)
            if nested:
                return nested
    return ""


def _normalize_build(value: Any) -> str | None:
    return _BUILD_ALIASES.get(str(value or "").strip().casefold())


def _normalize_chromosome(value: Any) -> str | None:
    text = str(value or "").strip().upper().removeprefix("CHR")
    return text if text in _VALID_CHROMOSOMES else None


def _positive_integer(value: Any) -> int | None:
    parsed = _nonnegative_integer(value)
    return parsed if parsed is not None and parsed > 0 else None


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
        if not parsed.is_finite() or not 0 <= parsed < 2**53 or parsed != parsed.to_integral_value():
            return None
    except (TypeError, ValueError, InvalidOperation, OverflowError):
        return None
    return int(parsed)


def _metric_integer(value: Any, *, fallback: int | None = None) -> int | None:
    parsed = _nonnegative_integer(value)
    return fallback if parsed is None else parsed


def _valid_p_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        exact = Decimal(str(value))
        if not exact.is_finite() or not 0 <= exact <= 1:
            return None
        parsed = float(exact)
    except (TypeError, ValueError, InvalidOperation, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and 0 <= parsed <= 1 else None


def _sequence_value(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _first_text(value: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        item = value.get(key)
        if item is not None and str(item).strip():
            return str(item).strip()
    return ""


def _valid_allele_pair(reference: str, alternate: str) -> bool:
    return bool(
        reference
        and alternate
        and _ALLELE.fullmatch(reference)
        and _ALLELE.fullmatch(alternate)
        and reference.casefold() != alternate.casefold()
    )
