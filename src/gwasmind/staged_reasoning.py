from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from decimal import ROUND_HALF_EVEN, Context, Decimal, InvalidOperation, localcontext
from enum import StrEnum
from typing import Any

from .academic_review import (
    operation_issue,
    publication_count,
    scientific_warnings,
)
from .candidates import evidence_candidate_scope
from .configuration import parameter
from .evidence_citations import inline_evidence_citations
from .evidence_sources import source_diversity
from .gate_rules import gate_rule, group_gate_checks, load_gate_rules
from .ld_support import (
    LD_COVERAGE_WARNING,
    LD_MHC_WARNING,
    LD_PROXY_EVIDENCE_WARNING,
    LD_PROXY_WARNING,
    ld_attribution_issues,
    ld_report_scope,
)
from .models import EvidenceKind, EvidenceRecord
from .progress import ProgressSink, emit_progress
from .projection import _has_high_impact_coding_payload
from .prompting import format_prompt, load_prompt
from .report_validation_cache import (
    memoized_report_derivation,
    report_validation_callback,
    report_validation_session,
)
from .source_gwas import summarize_source_gwas_observations


class SafetyStatus(StrEnum):
    """User-visible result of the stage-three safety and sufficiency review."""

    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    STOPPED = "STOPPED"


class SafetySeverity(StrEnum):
    HARD_GATE = "hard_gate"
    WARNING = "warning"


class SafetyOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    NOTE = "note"
    NOT_ASSESSED = "not_assessed"
    NOT_APPLICABLE = "not_applicable"


class StructuredOutputError(ValueError):
    """An LLM response did not satisfy a strict, grounded JSON contract."""


_REPORT_PROSE_LOCALLY_REVIEWED = ContextVar("report_prose_locally_reviewed", default=False)


_ALLOWED_HARD_STOP_CHECK_IDS = frozenset(
    rule.check_id for rule in load_gate_rules().rules.values() if rule.allow_hard_stop
) | frozenset({"variant_integrity", "build_integrity", "allele_integrity", "evidence_availability",
               "required_modules", "locus_genes_1mb", "direct_locus_gene_evidence",
               "model_output_grounding", "report_output_grounding"})
# Historical immutable review objects keep their original severity. New checks
# are created through configured_safety_check and the v2 loader forbids SSR stops.


SAFETY_AUDITOR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "explanation", "check_explanations", "limitations"],
    "properties": {
        "status": {
            "type": "string",
            "enum": [item.value for item in SafetyStatus],
        },
        "explanation": {"type": "string", "minLength": 1},
        "check_explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["check_id", "explanation"],
                "properties": {
                    "check_id": {"type": "string", "minLength": 1},
                    "explanation": {"type": "string", "minLength": 1},
                },
            },
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}


HYBRID_FINAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary_markdown",
        "evidence_sections",
        "limitations_uncertainty",
    ],
    "properties": {
        "summary_markdown": {"type": "string", "minLength": 1},
        "evidence_sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["section_id", "fact_ids"],
                "properties": {
                    "section_id": {"type": "string", "minLength": 1},
                    "fact_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "limitations_uncertainty": {
            "type": "object",
            "additionalProperties": False,
            "required": ["markdown"],
            "properties": {
                "markdown": {"type": "string", "minLength": 1},
                "safety_check_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
        "reference_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}


@dataclass(frozen=True, slots=True)
class SafetyCheck:
    check_id: str
    label: str
    severity: SafetySeverity
    outcome: SafetyOutcome
    fact: str
    reason_codes: tuple[str, ...] = ()
    impact: str = ""
    affected: Mapping[str, Any] = field(default_factory=dict)
    refs: tuple[Mapping[str, str], ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.check_id, label="safety check ID")
        if not self.label.strip() or not self.fact.strip():
            raise ValueError("safety check label and fact are required")
        if self.severity is SafetySeverity.HARD_GATE and self.outcome is SafetyOutcome.WARN:
            raise ValueError("a hard gate cannot have a warning outcome")
        if self.severity is SafetySeverity.WARNING and self.outcome is SafetyOutcome.FAIL:
            raise ValueError("a warning check cannot have a failure outcome")

    @property
    def nonpassing(self) -> bool:
        return self.outcome in {SafetyOutcome.WARN, SafetyOutcome.FAIL}

    def to_dict(self) -> dict[str, Any]:
        rule = load_gate_rules().rules.get(self.check_id)
        return {
            "check_id": self.check_id,
            "label": self.label,
            "severity": self.severity.value,
            "outcome": self.outcome.value,
            "fact": self.fact,
            **(rule.metadata() if rule else {}),
            **({"reason_codes": list(self.reason_codes), "impact": self.impact,
                "affected": {key: list(value) for key, value in self.affected.items()},
                "refs": [dict(ref) for ref in self.refs]} if self.reason_codes else {}),
        }


@dataclass(frozen=True, slots=True)
class SafetyFacts:
    """Program-computed facts that an LLM may explain but may never override."""

    checks: tuple[SafetyCheck, ...]
    evidence_record_count: int
    direct_evidence_ids: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    module_statuses: tuple[tuple[str, str], ...] = ()
    candidate_ranking_status: str | None = None
    source_ids: tuple[str, ...] = ()
    unidentified_source_records: int = 0
    source_scope: str | None = None
    intent: str = "prioritize_genes"
    source_observation_count: int | None = None
    valid_source_variant_count: int | None = None
    gene_evidence_ids: tuple[str, ...] = ()
    operation_issues: tuple[dict[str, Any], ...] = ()
    ld_only_discovery_gap: bool = False
    review_scope: Mapping[str, Any] = field(default_factory=dict)
    candidate_gene_count: int | None = None

    def __post_init__(self) -> None:
        if not self.checks:
            raise ValueError("at least one safety check is required")
        if self.evidence_record_count < 0:
            raise ValueError("evidence_record_count cannot be negative")
        _require_unique((item.check_id for item in self.checks), label="safety check IDs")
        unexpected_hard_stops = tuple(
            item.check_id
            for item in self.checks
            if item.severity is SafetySeverity.HARD_GATE
            and item.check_id not in _ALLOWED_HARD_STOP_CHECK_IDS
        )
        if unexpected_hard_stops:
            raise ValueError(
                "unreviewed safety checks cannot stop an analysis: "
                + ", ".join(unexpected_hard_stops)
            )
        _require_unique(self.direct_evidence_ids, label="direct evidence IDs")
        _require_unique(self.gene_evidence_ids, label="gene evidence IDs")
        for name in ("source_observation_count", "valid_source_variant_count"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or unknown")
        if (self.source_observation_count is not None and self.valid_source_variant_count is not None
                and self.valid_source_variant_count > self.source_observation_count):
            raise ValueError("valid source count cannot exceed the original source count")
        _require_unique(self.required_modules, label="required modules")
        module_names = tuple(item[0] for item in self.module_statuses)
        _require_unique(module_names, label="module status names")

    @property
    def status(self) -> SafetyStatus:
        if any(issue.get("blocking") is True for issue in self.operation_issues):
            return SafetyStatus.STOPPED
        if any(
            item.severity is SafetySeverity.HARD_GATE
            and item.outcome is SafetyOutcome.FAIL
            for item in self.checks
        ):
            return SafetyStatus.STOPPED
        if any(item.outcome is SafetyOutcome.WARN for item in self.checks):
            return SafetyStatus.PASS_WITH_WARNINGS
        return SafetyStatus.PASS

    @property
    def nonpassing_check_ids(self) -> tuple[str, ...]:
        return tuple(item.check_id for item in self.checks if item.nonpassing)

    @property
    def report_allowed(self) -> bool:
        """Coverage gaps restrict conclusions without discarding verified findings."""
        return self.status is not SafetyStatus.STOPPED

    @property
    def ranking_allowed(self) -> bool:
        # Ranking is unanswerable only when no candidate gene exists. Missing
        # locus links are disclosed as a warning; the model still ranks the pool.
        return self.report_allowed and bool(self.gene_evidence_ids or self.direct_evidence_ids
                                            or self.candidate_gene_count)

    @property
    def academic_warnings(self) -> list[dict[str, Any]]:
        return scientific_warnings(self.checks, ld_only_discovery_gap=self.ld_only_discovery_gap)

    @property
    def answerability(self) -> dict[str, Any]:
        missing = [check.check_id for check in self.checks if check.nonpassing
                   and check.check_id in {"task_evidence_requirements", "evidence_availability"}]
        if self.intent == "prioritize_genes" and not self.ranking_allowed:
            missing.append("no_eligible_gene_evidence")
        reasons = list(dict.fromkeys(missing or [reason for item in self.academic_warnings
                                                for reason in item["reason_codes"]]))
        level = "insufficient" if missing or not self.report_allowed else "limited" if reasons else "sufficient"
        return {"level": level, "reason_codes": reasons}

    @property
    def method_notes(self) -> list[dict[str, Any]]:
        from .academic_review import scientific_method_notes
        notes = scientific_method_notes(self.checks)
        if self.review_scope.get("bounded_literature_modules") and not any(item["code"] == "W4_METHOD" for item in notes):
            notes.append({"code": "W4_METHOD", "label": "Declared literature scope", "assessment_state": "note",
                "facts": ["Literature retrieval used the declared per-gene limit; this evidence view is not an exhaustive publication search."],
                "check_ids": [], "reason_codes": ["declared_literature_limit"]})
        if "expression_comparison_unresolved" in self.review_scope.get("deferred_candidate_causes", ()):
            existing = next((item for item in notes if item["code"] == "W3_METHOD"), None)
            if existing is None:
                existing = {"code": "W3_METHOD", "label": "Expression comparison scope",
                    "assessment_state": "not_assessed", "facts": [], "check_ids": [], "reason_codes": []}
                notes.append(existing)
            existing["facts"].append("Expression records were retrieved, but some expression-specificity comparisons could not be assessed. This is not a failed expression query.")
            existing["reason_codes"].append("expression_comparison_unresolved")
            existing["assessment_state"] = "not_assessed"
        return [*notes, *[{"code": "LD_PROXY_METHOD", "label": "Verified LD proxy evidence",
                  "facts": [check.fact], "check_ids": [check.check_id]}
                 for check in self.checks if check.check_id == "ld_proxy_scope"
                 and check.outcome is SafetyOutcome.PASS]]

    def to_dict(self) -> dict[str, Any]:
        dependency_assessment = self.review_scope.get("dependency_assessment", {})
        if not isinstance(dependency_assessment, Mapping):
            dependency_assessment = {}
        independence_state = dependency_assessment.get("independence_state") or "not_established"
        # Preserve the historical public field's binary meaning while
        # exposing the more detailed policy state separately.  A known shared
        # source or an unassessed relationship is not proof of independent
        # cohorts; only the explicitly established state passes.
        cohort_independence = "established" if independence_state == "established" else "not_established"
        independent_cohort_count = dependency_assessment.get("independent_cohort_count")
        return {
            "authoritative_status": self.status.value,
            "report_allowed": self.report_allowed,
            "ranking_allowed": self.ranking_allowed,
            "answerability": self.answerability,
            "academic_warnings": self.academic_warnings,
            "method_notes": self.method_notes,
            "review_scope": dict(self.review_scope),
            "operation_issues": [dict(issue) for issue in self.operation_issues],
            "source_identity": {"source_count": self.source_observation_count,
                                "valid_source_count": self.valid_source_variant_count},
            "evidence_record_count": self.evidence_record_count,
            "direct_evidence_ids": list(self.direct_evidence_ids),
            "required_modules": list(self.required_modules),
            "module_statuses": dict(self.module_statuses),
            "checks": [item.to_dict() for item in self.checks],
            "gate_reviews": group_gate_checks(tuple(item.to_dict() for item in self.checks)),
            **({"source_diversity": {
                "scope": self.source_scope, "source_ids": list(self.source_ids),
                "source_count": len(self.source_ids),
                "unidentified_records": self.unidentified_source_records,
                "cohort_independence": cohort_independence,
                "independent_cohort_count": independent_cohort_count,
            }} if self.source_scope is not None else {}),
            **({"candidate_ranking": {
                "status": self.candidate_ranking_status,
                "attempted": None if self.candidate_ranking_status == "unknown" else self.candidate_ranking_status not in {"deferred", "not_applicable"},
            }} if self.candidate_ranking_status is not None else {}),
        }

    def to_model_payload(self) -> dict[str, Any]:
        """Explain the authoritative checks without resending opaque ID inventories."""
        from .scientific_facts import identifier_set_summary

        payload = self.to_dict()
        payload["direct_evidence_binding"] = identifier_set_summary(payload.pop("direct_evidence_ids"))
        diversity = payload.get("source_diversity")
        if isinstance(diversity, dict):
            diversity["source_binding"] = identifier_set_summary(diversity.pop("source_ids", ()))
        payload.pop("gate_reviews", None)
        payload["checks"] = [check for check in payload["checks"] if check["outcome"] in {"warn", "fail"}]
        payload["check_explanation_scope"] = "nonpassing_observations"
        return payload


@dataclass(frozen=True, slots=True)
class SafetyReview:
    status: SafetyStatus
    explanation: str
    check_explanations: tuple[tuple[str, str], ...]
    limitations: tuple[str, ...]
    facts: SafetyFacts

    def __post_init__(self) -> None:
        if self.status is not self.facts.status:
            raise ValueError("the safety review cannot override program-computed status")
        if not self.explanation.strip():
            raise ValueError("a safety explanation is required")
        supplied_ids = tuple(item[0] for item in self.check_explanations)
        expected_ids = tuple(item.check_id for item in self.facts.checks)
        if supplied_ids != expected_ids:
            raise ValueError("check explanations must follow the authoritative check order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "explanation": self.explanation,
            "checks": [
                {
                    **check.to_dict(),
                    "explanation": explanation,
                }
                for check, (_, explanation) in zip(
                    self.facts.checks, self.check_explanations, strict=True
                )
            ],
            "limitations": list(self.limitations),
            "gate_reviews": group_gate_checks(tuple(check.to_dict() for check in self.facts.checks)),
            "answerability": self.facts.answerability,
            "academic_warnings": self.facts.academic_warnings,
            "method_notes": self.facts.method_notes,
            "review_policy_version": self.facts.review_scope.get("policy_version"),
            "review_scope_sha256": self.facts.review_scope.get("sha256"),
            "operation_issues": [dict(issue) for issue in self.facts.operation_issues],
        }


def configured_safety_check(
    *, check_id: str, label: str, severity: SafetySeverity, outcome: SafetyOutcome, fact: str,
    reason_codes: Sequence[str] = (), impact: str = "", affected: Mapping[str, Any] | None = None,
    refs: Sequence[Mapping[str, str]] = (),
) -> SafetyCheck:
    """Apply the loaded severity to a computed condition; never to model prose."""
    rule_severity = gate_rule(check_id).severity
    configured_severity = SafetySeverity.WARNING if rule_severity == "note" else SafetySeverity(rule_severity)
    configured_outcome = (outcome if outcome in {SafetyOutcome.PASS, SafetyOutcome.NOTE,
        SafetyOutcome.NOT_ASSESSED, SafetyOutcome.NOT_APPLICABLE} else
        SafetyOutcome.NOTE if rule_severity == "note" else
        SafetyOutcome.FAIL if configured_severity is SafetySeverity.HARD_GATE else SafetyOutcome.WARN)
    return SafetyCheck(check_id, label, configured_severity, configured_outcome, fact,
        tuple(reason_codes), impact, {key: tuple(value) for key, value in (affected or {}).items()},
        tuple(dict(ref) for ref in refs))


def preserve_gate1_warnings(current: SafetyFacts, prior: SafetyFacts) -> SafetyFacts:
    """Output recovery cannot erase previously established input/evidence limits."""
    checks = list(current.checks)
    indices = {check.check_id: index for index, check in enumerate(checks)}
    for check in prior.checks:
        rule = load_gate_rules().rules.get(check.check_id)
        if not check.nonpassing or rule is None or rule.gate != "gate1":
            continue
        index = indices.get(check.check_id)
        if index is None:
            indices[check.check_id] = len(checks)
            checks.append(check)
        elif checks[index].outcome is SafetyOutcome.PASS:
            checks[index] = check
    return replace(current, checks=tuple(checks))


_PUBLIC_OPTIONAL_MODULE_LABELS = {
    "locus.study": "GWAS study lookup",
    "locus.credible_set": "Credible-set lookup",
    "locus.fine_mapping": "Fine-mapping lookup",
    "locus.gene": "Regional gene lookup",
    "europepmc": "Europe PMC literature",
    "evidence.europepmc": "Open Targets target literature and publication metadata",
    "target.literature": "Open Targets target literature",
    "publication.metadata": "Europe PMC publication metadata",
    "locus.colocalisation": "Colocalization",
    "locus.qtl": "Molecular QTL",
    "locus.enhancer_to_gene": "Enhancer-to-gene links",
    "locus.variant": "Variant annotations",
    "locus.variant_consequence": "Variant consequences",
    "locus.expression": "Gene expression",
    "target.expression": "Gene expression",
    "locus.function": "Gene function",
    "target.function": "Gene function",
    "locus.gene_context": "Regional gene context",
    "locus.l2g": "Locus-to-gene evidence",
}

_TARGET_LITERATURE_LIMIT_WARNING = (
    "Open Targets target-literature lists were limited to the configured number of records per gene. "
    "Additional publications may be absent from this evidence view."
)
_PUBLICATION_METADATA_COVERAGE_WARNING = (
    "Europe PMC publication metadata was incomplete for some already identified PMIDs. "
    "Their identifiers and available metadata were retained."
)
_BIOLOGICAL_CONTEXT_COVERAGE_WARNING = (
    "The retrieved evidence did not establish coverage of the defined phenotype-relevant tissues or cell types."
)
_LEGACY_BIOLOGICAL_CONTEXT_COVERAGE_WARNING = (
    "No retrieved direct evidence matches the defined phenotype-relevant context."
)


def _literature_coverage_details(module_warnings: Mapping[str, Sequence[str]], modules: Sequence[str]) -> tuple[str, ...]:
    details = []
    for module in modules:
        if module not in {"evidence.europepmc", "target.literature", "publication.metadata"}:
            continue
        warnings = module_warnings.get(module, ())
        if not isinstance(warnings, (list, tuple)):
            continue
        for warning in warnings:
            if not isinstance(warning, str):
                continue
            limit = re.match(r"^Open Targets declared (\d{1,12}) literatureOcurrences rows for [A-Za-z0-9_.:-]+; the bounded response retained (\d{1,12})\.", warning)
            if limit and int(limit.group(1)) > int(limit.group(2)):
                details.append(_TARGET_LITERATURE_LIMIT_WARNING)
            if (warning.startswith("Publication metadata was unavailable for one batch; retained PMID provenance.")
                    or re.match(r"^Publication metadata was not returned for \d+ retrieved study PMID\(s\)", warning)
                    or warning.startswith("Evidence-linked publication metadata exceeded the retrieval bound; unresolved PMID provenance was retained.")):
                details.append(_PUBLICATION_METADATA_COVERAGE_WARNING)
    return tuple(dict.fromkeys(details))


def public_safety_warning_messages(review: SafetyReview) -> tuple[str, ...]:
    """Describe only immutable WARN checks, without model-authored explanations."""

    return tuple(dict.fromkeys(
        message for check in review.facts.checks if check.outcome is SafetyOutcome.WARN
        and check.check_id not in {"variant_integrity", "build_integrity", "allele_integrity"}
        for message in _public_check_warning_messages(check)
    ))


def public_safety_warning_items(review: SafetyReview) -> tuple[dict[str, Any], ...]:
    """Bind fixed public messages to their configured cause/consequence metadata."""
    nonpassing = set(review.facts.nonpassing_check_ids)
    items = []
    for check in review.facts.checks:
        if not check.nonpassing or check.check_id in {"variant_integrity", "build_integrity", "allele_integrity"}:
            continue
        rule = load_gate_rules().rules.get(check.check_id)
        metadata = rule.metadata() if rule else {
            "gate": "gate1", "stage": "safety", "display_group": "additional_review",
            "role": "cause", "code": check.check_id, "depends_on": [],
        }

        items.append({"check_id": check.check_id, **metadata,
            "severity": check.severity.value, "outcome": check.outcome.value,
            "caused_by": [identifier for identifier in metadata["depends_on"] if identifier in nonpassing],
            "messages": list(_public_check_warning_messages(check))})
    return tuple(items)


def _public_check_warning_messages(check: SafetyCheck) -> tuple[str, ...]:
    """Use computed check facts only; never an auditor's free explanation."""
    if not check.nonpassing:
        return ()
    if check.reason_codes:
        return (check.fact,)
    if check.check_id == "model_review_execution":
        return ("Program evidence checks completed, but the model-assisted review was unavailable. "
                "The report uses the program-validated evidence and discloses the remaining limitations.",)
    if check.outcome is SafetyOutcome.FAIL and check.check_id in {
        "variant_integrity", "build_integrity", "allele_integrity", "evidence_availability",
    }:
        return ({
            "variant_integrity": "Safety gate stopped: variant identifier integrity could not be verified.",
            "build_integrity": "Safety gate stopped: genome-build integrity could not be verified.",
            "allele_integrity": "Safety gate stopped: allele integrity could not be verified.",
            "evidence_availability": "Safety gate stopped: no verifiable retrieved evidence or uploaded observations are available.",
        }[check.check_id],)
    if check.check_id == "optional_evidence_coverage":
        match = re.search(r"incomplete for: (.+?)\. Missing", check.fact)
        names = match.group(1).split(", ") if match else ()
        details = tuple(message for message in (_TARGET_LITERATURE_LIMIT_WARNING, _PUBLICATION_METADATA_COVERAGE_WARNING) if message in check.fact)
        labels = tuple(dict.fromkeys(_PUBLIC_OPTIONAL_MODULE_LABELS.get(name, "Other optional evidence") for name in names
                       if not (details and name in {"evidence.europepmc", "target.literature", "publication.metadata"})))
        generic = tuple(f"Optional evidence coverage was incomplete: {label}. Missing records are not evidence of absence." for label in labels)
        return (*generic, *details) or ("Optional evidence coverage was incomplete: Other optional evidence. Missing records are not evidence of absence.",)
    if check.check_id == "evidence_independence":
        distinct_count = re.search(r"spans (\d+) distinct source study/dataset", check.fact)
        if distinct_count:
            scope = "Direct evidence" if check.fact.startswith("Qualifying direct") else "Retrieved evidence"
            return (f"{scope} spans only {int(distinct_count.group(1))} distinct source study/dataset identifier(s); "
                    "cohort independence is not established.",)
        count = re.search(r"spans (\d+) independent dataset", check.fact)
        message = (
            f"Evidence spans only {int(count.group(1))} independent dataset(s); independent support is limited."
            if count else "Evidence independence could not be verified."
        )
    elif check.check_id == "variant_resolution_coverage":
        message = (
            "Some selected variants could not be uniquely identified. Their uploaded "
            "values are preserved, but missing matches for them cannot establish novelty."
        )
    elif check.check_id == "evidence_conflicts":
        count = re.match(r"(\d+) evidence conflict", check.fact)
        message = (
            f"{int(count.group(1))} evidence conflict(s) require interpretation."
            if count else "Conflicting evidence requires interpretation."
        )
    elif check.check_id == "current_gwas_signal":
        message = (
            "No selected variant reaches the configured significance threshold in the loaded GWAS dataset."
            if check.fact.startswith("None of the selected variants")
            else "Significance in the loaded GWAS dataset could not be verified."
        )
    elif check.check_id == "biological_context_coverage":
        message = {
            "No explicit phenotype-specific tissue or cell context was defined.": "No phenotype-specific tissue or cell context was defined.",
            _LEGACY_BIOLOGICAL_CONTEXT_COVERAGE_WARNING: _BIOLOGICAL_CONTEXT_COVERAGE_WARNING,
            _BIOLOGICAL_CONTEXT_COVERAGE_WARNING: _BIOLOGICAL_CONTEXT_COVERAGE_WARNING,
        }.get(check.fact, "Phenotype-relevant tissue or cell coverage could not be verified.")
    elif check.check_id == "complex_region":
        message = (
            LD_MHC_WARNING if check.fact == LD_MHC_WARNING else
            "The selected locus overlaps the MHC region; correlated signals and study coverage require caution."
            if re.search(r"\b(?:MHC|HLA)\b", check.fact, re.I)
            else "The selected locus overlaps a configured complex region; association evidence requires caution."
        )
    elif check.check_id == "literature_coverage":
        message = check.fact
    elif check.check_id == "model_output_grounding":
        message = (
            check.fact
            if re.match(r"\d+ validated model ranking entries and their original ranking positions were retained;", check.fact)
            or "Model context capacity omitted" in check.fact
            else
            "Candidate ranking was corrected and validated after one repair."
            if "repair passed" in check.fact
            else "Candidate genes were ordered with a deterministic evidence-based method after the ranking synthesis could not be validated. This method ranks available support; its scores are not causal probabilities."
            if "ordered with a deterministic evidence-based method" in check.fact
            else "Candidate ranking used a validated evidence-based fallback after a model failure."
            if "passed validation" in check.fact
            else "Candidate ranking remains unavailable; retrieved evidence is retained."
        )
    elif check.check_id in {"ld_proxy_scope", "ld_proxy_coverage"}:
        if check.check_id == "ld_proxy_scope":
            return (LD_PROXY_WARNING,)
        return tuple(message for message in (LD_COVERAGE_WARNING, LD_PROXY_EVIDENCE_WARNING) if message in check.fact) or (LD_COVERAGE_WARNING,)
    elif check.check_id in {"required_modules", "locus_genes_1mb", "direct_locus_gene_evidence", "novelty_coverage",
                            "ranking_not_attempted", "ranking_unavailable", "candidate_discovery_coverage"}:
        message = check.fact
    else:
        message = "An additional evidence-review check requires caution; see the report limitations."
    return (message,)


def _writer_warning_messages(source: str | None, errors: Sequence[str]) -> tuple[str, ...]:
    # Gate 2 corrects or removes unsupported values and identifiers in place;
    # those edits stay in the fact-review audit and never become warnings.
    # Only an incomplete interpretation is disclosed to the reader.
    if source == "evidence_only_recovery":
        return ("The explanatory synthesis could not be completed reliably. Verified findings and references are provided, but interpretation remains incomplete.",)
    if source == "partial_recovery":
        return ("Some requested interpretation remains unresolved. The report retains verified findings and supported statements; unsupported conclusions were omitted.",)
    # A fully corrected draft has no remaining scientific limitation. Keep repair
    # diagnostics in the audit record rather than presenting them as a warning.
    return ()


def _report_warning_presentation(review: SafetyReview, source: str | None,
                                 errors: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Short opening for many warnings; every original cause remains available."""
    warnings = tuple(dict.fromkeys((*public_safety_warning_messages(review),
                                   *_writer_warning_messages(source, errors))))
    joined = " ".join(warnings)
    if len(warnings) <= 2 and len(joined) <= 360:
        return joined, ()
    labels_by_check = {
        "required_modules": "evidence retrieval", "optional_evidence_coverage": "optional evidence coverage",
        "variant_resolution_coverage": "variant identity coverage", "candidate_discovery_coverage": "candidate coverage",
        "locus_genes_1mb": "ranking requirements", "direct_locus_gene_evidence": "ranking requirements",
        "ranking_not_attempted": "ranking unavailable", "ranking_unavailable": "ranking unavailable",
        "current_gwas_signal": "uploaded GWAS signal",
        "biological_context_coverage": "tissue/cell relevance", "evidence_independence": "independent support",
        "evidence_conflicts": "conflicting evidence", "complex_region": "complex locus",
        "literature_coverage": "literature coverage", "novelty_coverage": "prior-report coverage (novelty remains unresolved)",
        "model_output_grounding": "ranking method", "model_review_execution": "additional quality review unavailable",
        "ld_proxy_scope": "LD evidence", "ld_proxy_coverage": "LD coverage",
    }
    labels = []
    for item in public_safety_warning_items(review):
        if item["outcome"] != SafetyOutcome.WARN.value or not item["messages"]:
            continue
        if item["check_id"] == "optional_evidence_coverage":
            for message in item["messages"]:
                label = "literature coverage" if ("Open Targets target-literature" in message
                    or "Europe PMC" in message or "Open Targets target literature" in message) else "optional evidence coverage"
                labels.append(label)
        else:
            labels.append(labels_by_check.get(item["check_id"], "additional review"))
    labels = list(dict.fromkeys(labels))
    opening = "Review limitations: " + ", ".join(labels) + "." if labels else ""
    if source == "evidence_only_recovery":
        opening += " The explanatory synthesis could not be completed reliably; interpretation remains incomplete."
    elif source == "partial_recovery":
        opening += " Some requested interpretation remains unresolved."
    return opening.strip(), warnings


@dataclass(frozen=True, slots=True)
class SafetyAuditResult:
    review: SafetyReview
    source: str
    attempts: int
    validation_errors: tuple[str, ...] = ()
    failure_diagnostics: Mapping[str, Any] | None = None

    @property
    def decision(self) -> SafetyStatus:
        return self.review.status

    @property
    def checks(self) -> tuple[dict[str, str], ...]:
        return tuple(item for item in self.review.to_dict()["checks"])

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            item.fact
            for item in self.review.facts.checks
            if item.severity is SafetySeverity.WARNING and item.outcome is SafetyOutcome.WARN
        )

    @property
    def limitations(self) -> tuple[str, ...]:
        return self.review.limitations

    @property
    def used_fallback(self) -> bool:
        return self.source == "deterministic_fallback"

    @property
    def validation_issues(self) -> tuple[str, ...]:
        return self.validation_errors

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.review.to_dict(),
            "decision": self.decision.value,
            "warnings": list(self.warnings),
            "source": self.source,
            "used_fallback": self.used_fallback,
            "attempts": self.attempts,
            "validation_issues": list(self.validation_errors),
            "review_execution": self.public_review_metadata,
        }

    @property
    def public_review_metadata(self) -> dict[str, Any]:
        completed = self.source in {"model", "model_repaired"}
        return {"method": "rules_with_model_explanation" if completed else "rules",
            "source": self.source, "model_review_status": "completed" if completed else "unavailable" if self.attempts else "not_requested",
            "model_attempts": self.attempts,
            **({"failure": dict(self.failure_diagnostics)} if self.failure_diagnostics else {})}


_COMPLETE_MODULE_STATUSES = frozenset(load_gate_rules().complete_module_statuses)
_INCOMPLETE_MODULE_STATUSES = frozenset(load_gate_rules().incomplete_module_statuses)
_DIRECT_KINDS = frozenset(
    {
        EvidenceKind.QTL.value,
        EvidenceKind.COLOCALIZATION.value,
        EvidenceKind.REGULATORY.value,
    }
)


def evaluate_safety_facts(
    *,
    evidence: Sequence[EvidenceRecord | Mapping[str, Any]],
    variant_integrity: bool | None,
    build_integrity: bool | None,
    allele_integrity: bool | None,
    required_modules: Sequence[str],
    module_statuses: Mapping[str, str],
    nearby_gene_count: int | None,
    biological_context_defined: bool,
    biological_context_coverage: bool | None,
    independent_dataset_count: int | None = None,
    conflict_count: int = 0,
    complex_region: str | bool | None = None,
    literature_count: int | None = None,
    intent: str | None = None,
    require_nearby_gene: bool | None = None,
    require_direct_evidence: bool | None = None,
    require_gwas_signal: bool = False,
    gwas_signal_present: bool | None = None,
    source_data_available: bool = False,
    module_warnings: Mapping[str, Sequence[str]] | None = None,
    source_observation_count: int | None = None,
    valid_source_variant_count: int | None = None,
    operation_issues: Sequence[Mapping[str, Any]] = (),
    candidate_gene_ids: Sequence[str] | None = None,
    source_observations: Sequence[Mapping[str, Any]] | None = None,
    review_requirements: Mapping[str, Any] | None = None,
    candidate_audit: Mapping[str, Any] | None = None,
    context_terms: Sequence[str] = (),
    biological_context: Mapping[str, Any] | None = None,
) -> SafetyFacts:
    """Compute task-scoped SSR observations after input/retrieval verification.

    S1/S2 receipts are operational. Academic insufficiency never stops a source
    report, and a successful empty query never implies invalid source identity.
    """

    records = tuple(evidence)
    normalized_intent = str(intent or "").strip().casefold()
    if require_nearby_gene is None:
        require_nearby_gene = normalized_intent in {"", "prioritize_genes"}
    if require_direct_evidence is None:
        # Ordinary prioritization can rank qualified contextual, functional,
        # or coding evidence.  A measured direct V2G link is required only
        # when the user explicitly asks for that mechanism.
        require_direct_evidence = False
    required = tuple(dict.fromkeys(_clean_identifier(item) for item in required_modules
                                   if _clean_identifier(item) != "candidate_ranking"))
    state_aliases = {"success_data": "success", "success_no_data": "no_data",
                     "not_requested": "not_requested", "not_applicable": "not_applicable"}
    normalized_statuses = {
        _clean_identifier(name): state_aliases.get(str(status).strip().casefold(), str(status).strip().casefold())
        for name, status in module_statuses.items()
    }
    checks: list[SafetyCheck] = []
    operational = [dict(issue) for issue in operation_issues]
    identity_required = normalized_intent in {"", "prioritize_genes", "describe_variants", "gene_associations"}
    identity_failed = (valid_source_variant_count == 0 if valid_source_variant_count is not None else
                       not all(value is True for value in (variant_integrity, build_integrity, allele_integrity)))
    if identity_required and identity_failed and not any(item.get("code") in {"S1", "S2", "SYSTEM_ERROR", "IDENTITY_UNRESOLVED"} for item in operational):
        reference_failed = any(normalized_statuses.get(module) in {
            "failed", "error", "unavailable", "blocked", "partial", "missing"}
            for module in ("entity.resolve", "schema.discovery"))
        operational.append(operation_issue("IDENTITY_UNRESOLVED" if reference_failed else "S1",
            category="service_unavailable" if reference_failed else "input_invalid",
            scope="retrieval" if reference_failed else "input", blocking=True,
            message=("Source identity verification could not finish because reference retrieval was incomplete."
                if reference_failed else "No source variant has a usable, verified identity for external analysis; original source observations are retained."),
            reasons=("reference_identity_unavailable" if reference_failed else "no_resolved_source_variant",)))

    checks.extend(
        (
            _integrity_check(
                "variant_integrity",
                "Variant identifier integrity",
                variant_integrity,
                "Selected variant identifiers were validated.",
                "Selected variant identifiers were invalid or could not be validated.",
            ),
            _integrity_check(
                "build_integrity",
                "Genome-build integrity",
                build_integrity,
                "The genome build was validated for the selected variants.",
                "The genome build was invalid or could not be validated.",
            ),
            _integrity_check(
                "allele_integrity",
                "Allele integrity",
                allele_integrity,
                "Reference and alternate alleles are well formed and consistent with the submitted variant identifiers; "
                "this is not a reference-database identity check.",
                "Reference and alternate alleles were invalid or could not be validated.",
            ),
        )
    )

    has_data = bool(records) or source_data_available is True
    checks.append(configured_safety_check(
        check_id="evidence_availability",
        label="Verifiable data availability",
        severity=SafetySeverity.HARD_GATE,
        outcome=SafetyOutcome.PASS if has_data else SafetyOutcome.FAIL,
        fact=("Retrieved evidence or validated uploaded observations are available."
              if has_data else "No verifiable retrieved evidence or uploaded observations are available."),
    ))

    from .review_policy import assess_scientific_evidence

    candidate_set = set(candidate_gene_ids) if candidate_gene_ids is not None else None
    candidate_records = tuple(record for record in records if candidate_set is None
        or _record_text(record, "gene_id") in candidate_set)
    direct_ids, direct_kinds, direct_datasets = _direct_evidence(candidate_records)
    eligible_genes = candidate_set if candidate_set is not None else {
        _record_text(record, "gene_id") for record in candidate_records
        if _record_text(record, "evidence_id") in set(direct_ids)}
    gene_evidence_ids = tuple(dict.fromkeys(_record_text(record, "evidence_id") for record in candidate_records
        if _record_text(record, "evidence_id") and _record_text(record, "gene_id") in eligible_genes
        and _record_kind(record) != EvidenceKind.L2G.value and _ranking_record_scope_allowed(record)))
    mappings = tuple(_review_record_mapping(record) for record in records)
    requirements = dict(review_requirements or {})
    if require_gwas_signal:
        requirements["require_gwas_significance"] = True
    if requirements.get("require_direct_evidence") is True:
        require_direct_evidence = True
    assessed, review_scope = assess_scientific_evidence(records=mappings,
        intent=normalized_intent or "prioritize_genes", direct_evidence_ids=direct_ids,
        gene_evidence_ids=gene_evidence_ids, candidate_gene_ids=candidate_gene_ids,
        required_modules=required, module_statuses=normalized_statuses, module_warnings=module_warnings,
        candidate_audit=candidate_audit, source_observations=source_observations,
        source_data_available=source_data_available, gwas_signal_present=gwas_signal_present,
        nearby_gene_count=nearby_gene_count, biological_context_defined=biological_context_defined,
        biological_context_coverage=biological_context_coverage, context_terms=context_terms,
        biological_context=biological_context,
        requirements=requirements, require_direct_evidence=require_direct_evidence,
        require_nearby_gene=require_nearby_gene, complex_region=complex_region)
    labels = {
        "task_evidence_requirements": "Task-specific evidence availability",
        "direct_locus_gene_evidence": "Direct variant-to-gene evidence",
        "locus_genes_1mb": "Annotated genes in the local window",
        "required_modules": "Required retrieval modules", "optional_evidence_coverage": "Requested evidence coverage",
        "novelty_coverage": "Prior-publication retrieval scope", "candidate_discovery_coverage": "Candidate-discovery coverage",
        "current_gwas_signal": "Current-dataset GWAS signal", "biological_context_coverage": "Biological-context coverage",
        "evidence_independence": "Original-study relationships", "evidence_conflicts": "Comparable evidence conflicts",
        "complex_region": "Complex genomic region", "literature_coverage": "Original-study provenance",
    }
    checks.extend(configured_safety_check(check_id=item["check_id"], label=labels[item["check_id"]],
        severity=SafetySeverity.WARNING, outcome=SafetyOutcome(item["outcome"]), fact=item["fact"],
        reason_codes=item["reason_codes"], impact=item["impact"], affected=item["affected"], refs=item["refs"])
        for item in assessed)
    direct_set = set(direct_ids)
    source_records = [record for record in mappings if not require_direct_evidence or record.get("evidence_id") in direct_set]
    diversity = source_diversity(source_records)
    ordered_statuses = tuple(
        (module, normalized_statuses.get(module, "missing")) for module in required
    )
    return SafetyFacts(
        checks=tuple(checks),
        evidence_record_count=len(records),
        direct_evidence_ids=direct_ids,
        required_modules=required,
        module_statuses=ordered_statuses,
        source_ids=diversity.source_ids,
        unidentified_source_records=diversity.unidentified_records,
        source_scope="qualifying_direct_locus_gene_evidence" if require_direct_evidence else "retrieved_evidence",
        intent=normalized_intent or "prioritize_genes",
        source_observation_count=source_observation_count,
        valid_source_variant_count=valid_source_variant_count,
        gene_evidence_ids=gene_evidence_ids,
        operation_issues=tuple(operational),
        review_scope=review_scope,
        candidate_gene_count=len(candidate_set) if candidate_set is not None else None,
    )


def safety_auditor_messages(facts: SafetyFacts) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_prompt("stage3_safety_auditor_system")},
        {
            "role": "user",
            "content": format_prompt(
                "stage3_safety_auditor_user",
                payload=_canonical_json(facts.to_model_payload()),
            ),
        },
    ]


def safety_auditor_repair_messages(
    facts: SafetyFacts,
    *,
    invalid_content: str,
    validation_error: str,
) -> list[dict[str, str]]:
    payload = {
        "authoritative_facts": facts.to_model_payload(),
        "validation_error": str(validation_error)[:parameter("gates.repair_error_max_chars")],
        "invalid_output": str(invalid_content)[:parameter("gates.safety_repair_input_max_chars")],
    }
    return [
        {"role": "system", "content": load_prompt("stage3_safety_repair_system")},
        {
            "role": "user",
            "content": format_prompt(
                "stage3_safety_repair_user",
                payload=_canonical_json(payload),
            ),
        },
    ]


def parse_safety_auditor_output(content: str, facts: SafetyFacts) -> SafetyReview:
    value = _strict_json_object(content, label="safety auditor output")
    _require_exact_keys(
        value,
        {"status", "explanation", "check_explanations", "limitations"},
        label="safety auditor output",
    )
    status = _parse_enum(value["status"], SafetyStatus, label="status")
    if status is not facts.status:
        raise StructuredOutputError(
            f"status must equal the authoritative program status {facts.status.value}"
        )
    explanation = _bounded_text(value["explanation"], label="explanation", limit=6_000)
    rows = value["check_explanations"]
    if not isinstance(rows, list):
        raise StructuredOutputError("check_explanations must be an array")
    parsed: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise StructuredOutputError(f"check_explanations[{index}] must be an object")
        _require_exact_keys(
            row,
            {"check_id", "explanation"},
            label=f"check_explanations[{index}]",
        )
        check_id = _bounded_text(row["check_id"], label="check_id", limit=100)
        if check_id in parsed:
            raise StructuredOutputError(f"duplicate check explanation: {check_id}")
        parsed[check_id] = _bounded_text(
            row["explanation"], label=f"explanation for {check_id}", limit=2_000
        )
    expected_ids = tuple(item.check_id for item in facts.checks)
    expected_model_ids = {item.check_id for item in facts.checks if item.nonpassing}
    if set(parsed) not in (set(expected_ids), expected_model_ids):
        missing = sorted(expected_model_ids.difference(parsed))
        unknown = sorted(set(parsed).difference(expected_ids))
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        raise StructuredOutputError("check explanation IDs do not match: " + "; ".join(detail))
    for check in facts.checks:
        parsed.setdefault(check.check_id, check.fact)
    limitations = _string_array(value["limitations"], label="limitations", limit=30)
    if facts.nonpassing_check_ids and not limitations:
        raise StructuredOutputError("limitations cannot be empty when a check did not pass")
    _validate_text_against_authority(
        "\n".join((explanation, *parsed.values(), *limitations)),
        authority=_canonical_json(facts.to_dict()),
        label="safety auditor explanation",
    )
    return SafetyReview(
        status=facts.status,
        explanation=explanation,
        check_explanations=tuple((check_id, parsed[check_id]) for check_id in expected_ids),
        limitations=limitations,
        facts=facts,
    )


def run_safety_audit(
    facts: SafetyFacts,
    completer: Callable[..., Any] | None,
    *,
    max_tokens: int = parameter("gates.safety_audit_max_tokens"),
) -> SafetyAuditResult:
    """Call the stage-three auditor, repair malformed JSON once, then fail safely."""

    if completer is None:
        return SafetyAuditResult(
            review=_fallback_safety_review(facts),
            source="deterministic_fallback",
            attempts=0,
            validation_errors=("model_not_configured",),
        )
    errors: list[str] = []
    try:
        draft = _complete_text(completer, safety_auditor_messages(facts), max_tokens=max_tokens)
    except Exception as exc:  # Provider failures must not crash the user response.
        from .model_generation import classify_generation_failure

        return SafetyAuditResult(
            review=_fallback_safety_review(facts),
            source="deterministic_fallback",
            attempts=1,
            validation_errors=(f"model_call_failed:{type(exc).__name__}",),
            failure_diagnostics=classify_generation_failure(exc),
        )
    try:
        review = parse_safety_auditor_output(draft, facts)
    except StructuredOutputError as exc:
        errors.append(f"draft:{exc}")
    else:
        return SafetyAuditResult(review=review, source="model", attempts=1)

    try:
        repaired = _complete_text(
            completer,
            safety_auditor_repair_messages(
                facts,
                invalid_content=draft,
                validation_error=errors[-1],
            ),
            max_tokens=max_tokens,
        )
        review = parse_safety_auditor_output(repaired, facts)
    except Exception as exc:
        from .model_generation import classify_generation_failure

        diagnostic = classify_generation_failure(exc)
        errors.append(f"repair:{exc}" if isinstance(exc, StructuredOutputError) else f"repair:{diagnostic['category']}")
        return SafetyAuditResult(
            review=_fallback_safety_review(facts),
            source="deterministic_fallback",
            attempts=2,
            validation_errors=tuple(errors),
            failure_diagnostics=diagnostic,
        )
    return SafetyAuditResult(
        review=review,
        source="model_repaired",
        attempts=2,
        validation_errors=tuple(errors),
    )


@dataclass(frozen=True, slots=True)
class FinalSection:
    section_id: str
    heading: str

    def __post_init__(self) -> None:
        _validate_identifier(self.section_id, label="section ID")
        _bounded_text(self.heading, label="section heading", limit=300)


@dataclass(frozen=True, slots=True)
class FinalEvidenceFact:
    """One deterministic display fact backed by zero or more MCP evidence records."""

    fact_id: str
    section_id: str
    evidence_ids: tuple[str, ...]
    markdown: str
    reference_ids: tuple[str, ...] = ()
    model_data: Mapping[str, Any] = field(default_factory=dict)
    required: bool = False

    def __post_init__(self) -> None:
        _validate_identifier(self.fact_id, label="fact ID")
        _validate_identifier(self.section_id, label="fact section ID")
        _require_unique(self.evidence_ids, label=f"evidence IDs for {self.fact_id}")
        _require_unique(self.reference_ids, label=f"reference IDs for {self.fact_id}")
        _bounded_text(
            self.markdown,
            label=f"display Markdown for {self.fact_id}",
            limit=parameter("gates.report_fact_max_chars"),
        )
        _json_value(self.model_data, path=f"facts.{self.fact_id}.model_data")


@dataclass(frozen=True, slots=True)
class FinalReference:
    reference_id: str
    markdown: str

    def __post_init__(self) -> None:
        _validate_identifier(self.reference_id, label="reference ID")
        _bounded_text(self.markdown, label=f"reference {self.reference_id}", limit=4_000)


@dataclass(frozen=True, slots=True)
class HybridFinalContext:
    question: str
    safety_review: SafetyReview
    sections: tuple[FinalSection, ...]
    facts: tuple[FinalEvidenceFact, ...]
    references: tuple[FinalReference, ...]
    fallback_summary: str
    intent: str = ""
    source_trait: str = ""
    source_observations: tuple[Mapping[str, Any], ...] = ()
    result: Mapping[str, Any] = field(default_factory=dict)
    deterministic_answer: str = ""
    biological_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _bounded_text(self.question, label="question", limit=parameter("gates.question_max_chars"))
        _bounded_text(self.fallback_summary, label="fallback summary", limit=12_000)
        if self.intent:
            _bounded_text(self.intent, label="intent", limit=100)
        if self.source_trait:
            _bounded_text(self.source_trait, label="source trait", limit=1_000)
        if self.deterministic_answer:
            _bounded_text(
                self.deterministic_answer,
                label="deterministic answer",
                limit=parameter("gates.report_fact_max_chars"),
            )
        _json_value(self.source_observations, path="source_observations")
        _json_value(self.biological_context, path="biological_context")
        _require_unique((item.section_id for item in self.sections), label="section IDs")
        _require_unique((item.fact_id for item in self.facts), label="fact IDs")
        _require_unique((item.reference_id for item in self.references), label="reference IDs")
        section_ids = {item.section_id for item in self.sections}
        reference_ids = {item.reference_id for item in self.references}
        for fact in self.facts:
            if fact.section_id not in section_ids:
                raise ValueError(f"fact {fact.fact_id} uses an unknown section ID")
            unknown_refs = set(fact.reference_ids).difference(reference_ids)
            if unknown_refs:
                raise ValueError(
                    f"fact {fact.fact_id} uses unknown reference IDs: "
                    + ", ".join(sorted(unknown_refs))
                )
        evidence_ids = [item for fact in self.facts for item in fact.evidence_ids]
        duplicates = sorted(
            evidence_id for evidence_id, count in Counter(evidence_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                "an evidence ID cannot be owned by more than one display fact: "
                + ", ".join(duplicates[:10])
            )

    def to_model_payload(self) -> dict[str, Any]:
        from .phenotype_scope import model_biological_context
        from .report_evidence_coverage import molecular_qtl_overlap_methods

        safety_payload = self.safety_review.to_dict()
        if self.intent == "describe_variants":
            safety_payload["limitations"] = [check.fact for check in self.safety_review.facts.checks if check.nonpassing]
        payload = {
            "question": self.question,
            "intent": self.intent,
            "source_trait": self.source_trait,
            "source_observations": _json_value(
                self.source_observations, path="source_observations"
            ),
            "biological_context": _json_value(
                model_biological_context(self.biological_context), path="biological_context"
            ),
            "validation_status": self.safety_review.status.value,
            "safety_review": safety_payload,
            "analysis_scope": _final_analysis_scope(self),
            "sections": [asdict(item) for item in self.sections],
            "facts": [
                {
                    "fact_id": item.fact_id,
                    "section_id": item.section_id,
                    "evidence_count": len(item.evidence_ids),
                    "reference_ids": list(item.reference_ids),
                    "data": _final_fact_model_data(item),
                    "required": item.required,
                }
                for item in self.facts
            ],
            "references": [
                {"reference_id": item.reference_id, "metadata": item.markdown}
                for item in self.references
            ],
            "required_limitation_check_ids": list(
                self.safety_review.facts.nonpassing_check_ids
            ),
        }
        overlap_methods = molecular_qtl_overlap_methods(self)
        if overlap_methods:
            payload["molecular_qtl_overlap_methods"] = overlap_methods
        if self.intent == "prioritize_genes":
            decisions = self.result.get("candidate_decisions")
            if isinstance(decisions, (list, tuple)) and decisions:
                payload["candidate_decisions"] = [
                    {key: _json_value(row[key], path=f"candidate_decisions.{key}")
                     for key in ("gene_id", "status", "reason", "evidence_ids") if key in row}
                    for row in decisions if isinstance(row, Mapping)]
                payload["candidate_selection"] = {
                    "method": "model", "assessment_completed": True,
                    "candidate_pool_count": len(decisions),
                    "selected_count": sum(row.get("status") == "selected" for row in decisions if isinstance(row, Mapping)),
                    "unresolved_count": sum(row.get("status") == "unresolved" for row in decisions if isinstance(row, Mapping)),
                    "meaning": "LLM shortlist annotations for deeper investigation, not ranking eligibility. Preserve the complete returned order and cite supplied facts."}
            full_ranking = _final_ranking_context(self.result)
            selection = {row['gene_id']: row['status'] for row in decisions or () if isinstance(row, Mapping) and row.get('gene_id')}
            for row in full_ranking:
                if row.get('gene_id') in selection:
                    row['shortlisted'] = selection[row['gene_id']] == 'selected'
                    row['investigation_status'] = selection[row['gene_id']]
            display_limit = parameter("ranking.max_ranked_genes")
            displayed = full_ranking[:display_limit]
            payload["ranked_candidates"] = displayed
            payload["ranking_display"] = {
                "display_limit": display_limit,
                "returned_ranked_count": len(full_ranking),
                "displayed_ranked_count": len(displayed),
                "displayed_ranked_gene_ids": [row["gene_id"] for row in displayed if "gene_id" in row],
                "complete_ranking_saved": True,
                "order_policy": "original_returned_ranking",
                "shortlist_is_ranking_filter": False,
                **({"tie_policy": "Candidates with tied=true share shared_rank because the ranking found no supplied "
                                  "observation that separates them. Describe them as tied, never as ordered."}
                   if any(row.get("tied") for row in full_ranking) else {}),
            }
            # This is a writer-only projection. Keep the complete result and
            # derived facts intact for auditing and final factual review.
            scope = dict(payload["analysis_scope"])
            computed = dict(scope["computed_facts"])
            computed.pop("ranked_gene_ids", None)
            computed.update(displayed_ranked_count=len(displayed),
                            displayed_ranked_gene_ids=payload["ranking_display"]["displayed_ranked_gene_ids"])
            scope["computed_facts"] = computed
            payload["analysis_scope"] = scope
            from .report_evidence_coverage import molecular_evidence_presence
            payload["displayed_molecular_evidence"] = molecular_evidence_presence(self)
        return payload


def _final_fact_model_data(fact: FinalEvidenceFact) -> Any:
    """Pass report facts without resending the complete record-to-fact audit mapping."""

    value = {
        key: item for key, item in fact.model_data.items()
        if key != "evidence_records"
    }
    return _json_value(value, path=fact.fact_id)


@memoized_report_derivation
def _final_ranking_context(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Carry already-validated ranking arguments into the final synthesis context."""

    from .models import shared_ranks

    ranking = result.get("ranked_genes")
    if not isinstance(ranking, (list, tuple)):
        return []
    rows = [row for row in ranking if isinstance(row, Mapping)]
    shared = shared_ranks(rows)
    context = []
    for row in rows:
        value = {key: _json_value(row[key], path=f"ranked_candidates.{key}")
                 for key in ("gene_id", "rank", "rationale", "limitations") if key in row}
        if shared.get(row.get("gene_id"), (0, False))[1]:
            value["shared_rank"], value["tied"] = shared[row["gene_id"]][0], True
        context.append(value)
    return context


@memoized_report_derivation
def _final_ld_report_scope(context: HybridFinalContext):
    return ld_report_scope(context.result, context.source_observations, context.intent)


@memoized_report_derivation
def _final_analysis_scope(context: HybridFinalContext) -> dict[str, Any]:
    from .report_facts import analysis_overview, computed_report_facts, ranking_provenance
    from .report_ld_projection import report_ld_projection
    rows = _runtime_evidence_rows(context.result)
    counts = Counter(str(row.get("kind") or "unknown") for row in rows)
    modules: Counter[str] = Counter()
    credible_sets = False
    measured_ld = False
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
        module = payload.get("module")
        if isinstance(module, str) and module:
            modules[module] += 1
        upstream = payload.get("upstream") if isinstance(payload.get("upstream"), Mapping) else {}
        raw = upstream.get("raw") if isinstance(upstream.get("raw"), Mapping) else payload
        member = raw.get("matchedLocusMember")
        credible_sets = credible_sets or module in {"locus.credible_set", "locus.fine_mapping"} or (
            row.get("dataset") == "credible_set" or payload.get("dataset") == "credible_set"
        ) or isinstance(member, Mapping) and bool(member.get("variantId"))
        for owner in (raw, member):
            if not isinstance(owner, Mapping):
                continue
            value = owner.get("r2Overall")
            if isinstance(value, (int, float, str, Decimal)) and not isinstance(value, bool):
                try:
                    r2 = Decimal(str(value))
                    measured_ld = measured_ld or r2.is_finite() and 0 <= r2 <= 1
                except InvalidOperation:
                    pass
    ld_scope = _final_ld_report_scope(context)
    return {
        "intent": context.intent,
        "retrieved_record_counts": dict(sorted(counts.items())),
        "retrieved_module_counts": dict(sorted(modules.items())),
        "external_credible_set_information_available": credible_sets,
        # Credible-set membership/its PIP alone is not a measured LD statistic.
        "external_ld_information_available": measured_ld or bool(ld_scope.valid and ld_scope.links),
        "required_retrieval_modules": list(context.safety_review.facts.required_modules),
        "gene_prioritization_requested": context.intent == "prioritize_genes",
        "validated_ranking_available": bool(_final_ranking_context(context.result)),
        "source_statistics_origin": "uploaded_source_observations",
        "external_statistics_origin": "individual_retrieved_studies",
        "computed_facts": computed_report_facts(context.source_observations, context.result),
        "analysis_overview": analysis_overview(context.source_observations, context.result,
            intent=context.intent, source_trait=context.source_trait),
        "ranking_provenance": ranking_provenance(context.result),
        **({"ld_expansion": report_ld_projection(ld_scope, (fact.markdown for fact in context.facts))}
           if ld_scope.provided else {}),
    }


def _validate_external_scope_absence(text: str, scope: Mapping[str, Any]) -> None:
    """Reject only unqualified availability denials contradicted by real records."""
    topic = r"(?:credible[- ]sets?|(?:measured\s+)?(?:linkage[- ]disequilibrium|LD))"
    topics = topic + r"(?:\s*(?:or|and|/)\s*" + topic + r")?"
    information = r"(?:\s+(?:information|data|evidence|measurements?|matri(?:x|ces)))?"
    availability = r"(?:provided|supplied|retrieved|available|included)"
    denial = re.compile(
        r"\b(?:no\s+(?:external\s+)?" + topics + information
        + r"\s+(?:(?:is|are|was|were|has\s+been|have\s+been)\s+)?" + availability
        + r"|" + topics + information + r"\s+(?:is|are|was|were|has\s+been|have\s+been)\s+"
        + r"(?:not\s+" + availability + r"|unavailable|absent))\b", re.I,
    )
    source = r"(?:(?:the|this)\s+)?(?:uploaded(?:\s+source)?|source)[- ](?:GWAS|dataset|data|file)"
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        # Another variant's retrieved record cannot refute a named-variant gap.
        if _REPORT_VARIANT.search(sentence):
            continue
        for match in denial.finditer(sentence):
            before = re.split(r";", sentence[:match.start()])[-1]
            after = re.split(r"[;,]", sentence[match.end():])[0]
            if (re.search(r"\b(?:in|from|with|for)\s+" + source + r"\s*,?\s*$", before, re.I)
                    or re.search(source + r"\s+(?:itself\s+)?(?:has|had|provides?|provided|supplies|supplied|includes?|included)\s*$", before, re.I)
                    or re.match(r"\s+(?:in|from|with|by|as\s+part\s+of)\s+" + source + r"\b", after, re.I)):
                # The external studies do not establish what the user's source
                # GWAS itself supplied or independently analysed.
                continue
            claim = match.group()
            if re.search(r"\bcredible[- ]sets?\b", claim, re.I) and scope["external_credible_set_information_available"]:
                raise StructuredOutputError("limitations credible-set availability contradicts retrieved external credible-set/fine-mapping records; distinguish external studies from analyses supplied by the uploaded source GWAS")
            if re.search(r"\b(?:linkage[- ]disequilibrium|LD)\b", claim, re.I) and scope["external_ld_information_available"]:
                raise StructuredOutputError("limitations LD availability contradicts retrieved external LD measurements; distinguish external studies from analyses supplied by the uploaded source GWAS")


def _validate_task_specific_limitations(text: str, *, context: HybridFinalContext) -> None:
    """Reject concrete scope/provenance contradictions, not general scientific prose."""

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        source = re.search(r"\bsource[- ]GWAS\s+P(?:[- ]values?|\s+values?)?\b(.{0,150}?)\bfrom\s+(?:(?:the|those)\s+)?(?:indicated|cited|external)\s+(?:publications?|studies)\b", sentence, re.I)
        attribution = source and re.search(
            r"\b(?:are|were|is|was|come|came|originate|originated|derive|derived|obtained|taken|sourced)(?:\s+(?:directly|originally|only|all|also))*\s*$",
            source.group(1), re.I,
        )
        if attribution and not re.search(r"\b(?:not|never)\b", source.group(1), re.I):
            raise StructuredOutputError("limitations source-GWAS provenance contradicts the uploaded source observations; external-study statistics belong to their individual publications")
    if context.intent != "describe_variants":
        return
    ranking_claim = re.search(r"\b(?:prioriti[sz]ed|ranked)\s+(?:candidate\s+)?genes?\b.{0,130}\b(?:hypothes\w*|caus\w*|proof|validat\w*)\b", text, re.I)
    pipeline_claim = any(
        not re.search(r"\b(?:no|without)\s*$", text[max(0, match.start() - 20):match.start()], re.I)
        and not re.match(r"\s+(?:was|is|has\s+been)\s+not\s+(?:performed|requested|conducted)\b", text[match.end():], re.I)
        for match in re.finditer(r"\bcandidate[- ]gene\s+prioriti[sz]ation\b", text, re.I)
    )
    if ranking_claim or pipeline_claim:
        raise StructuredOutputError("limitations task scope: this variant description did not prioritize genes; omit the gene-ranking caveat")
    scope = _final_analysis_scope(context)
    _validate_external_scope_absence(text, scope)
    counts = scope["retrieved_record_counts"]
    requested_modules = set(scope["required_retrieval_modules"])
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        for absence in re.finditer(
            r"\bno\s+(?:variant\s+)?annotations?\b(?:(?!\bno\b|[.;!?]).){0,70}?"
            r"\b(?:provided|supplied|retrieved|available)\b", sentence, re.I,
        ):
            # Negation governs only its immediately following proposition.
            # A display gap 'is not proof that no annotation was retrieved'
            # denies the absence inference; it does not assert absent data.
            # Never exempt a later, independent 'no annotations' in the sentence.
            denied_inference = re.search(
                r"\b(?:(?:is|are|was|were)\s+not\s+(?:proof|evidence)|"
                r"(?:does|do|did)\s+not\s+(?:prove|establish|show|demonstrate|mean))\s+that\s*$",
                sentence[:absence.start()], re.I,
            )
            if not denied_inference and not _REPORT_VARIANT.search(sentence) and counts.get("variant_annotation", 0):
                raise StructuredOutputError("limitations annotation availability contradicts the retrieved variant-annotation records; do not claim that annotations were absent")
        inference_object = r"(?:causal\s+genes?|(?:functional|regulatory)\s+annotations?)"
        if re.fullmatch(
            r"\s*No\s+" + inference_object + r"(?:\s+(?:or|and)\s+" + inference_object + r")?"
            r"\s+(?:is|are|was|were|has\s+been|have\s+been)\s+inferred"
            r"(?:\s+from\s+(?:(?:these|the|this|available|provided)\s+)?"
            r"(?:association|GWAS|observational)\s+(?:data|evidence|results))?[.!?]?\s*",
            sentence, re.IGNORECASE,
        ):
            # A complete non-inference statement describes the report's scope.
            # Do not exempt a sentence with an added availability gap or advice.
            continue
        if re.fullmatch(
            r"\s*(?:No\s+(?:functional|regulatory)\s+follow[- ]up\s+(?:was|is)\s+performed|"
            r"(?:Functional|Regulatory)\s+follow[- ]up\s+(?:was|is)\s+not\s+performed)"
            r"(?:\s+in\s+this\s+(?:analysis|description|report))?[.!?]?\s*",
            sentence, re.IGNORECASE,
        ):
            # A scope statement is not a recommendation to add experiments.
            continue
        domains = (
            (r"\b(?:[a-z-]*QTL|colocali[sz]ation)\b", {"qtl", "colocalization"}, {"locus.qtl", "locus.colocalisation"}),
            (r"\b(?:functional|regulatory)\s+(?:annotations?|evidence|follow[- ]up)\b", {"function", "regulatory"}, {"target.function", "locus.enhancer_to_gene"}),
        )
        for pattern, kinds, modules in domains:
            if not re.search(pattern, sentence, re.I):
                continue
            requested = bool(requested_modules & modules) or any(counts.get(kind, 0) for kind in kinds)
            if requested or re.search(pattern, context.question, re.I):
                continue
            gap = re.search(r"\b(?:no|missing|lack(?:s|ing)?|absence|without)\b", sentence, re.I)
            followup = re.search(r"\b(?:future\s+work|follow[- ]up|recommend\w*|could\s+include|should\s+include|would\s+be\s+needed)\b", sentence, re.I)
            if gap or followup:
                raise StructuredOutputError("limitations task scope: do not present unrequested functional/regulatory, gene, QTL or colocalization analyses as missing evidence or routine follow-up for this variant description")


def build_hybrid_final_context(
    *,
    question: str,
    intent: str,
    source_trait: str,
    source_observations: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    deterministic_answer: str,
    biological_context: Mapping[str, Any] | Any,
    safety_review: SafetyReview | SafetyAuditResult,
) -> HybridFinalContext:
    """Build auditable fact/reference registries from the current runtime answer.

    This is the server-facing convenience API. The existing deterministic answer remains
    authoritative for detailed wording and bibliography metadata. The model receives only
    bounded section facts and foreign keys, then application code rematerializes those
    deterministic details after validation.
    """

    review = safety_review.review if isinstance(safety_review, SafetyAuditResult) else safety_review
    context_value = _public_mapping(biological_context)
    evidence_rows = _runtime_evidence_rows(result)
    sections, facts, references = _registries_from_deterministic_answer(
        deterministic_answer,
        evidence_rows=evidence_rows,
    )
    fallback_summary = _deterministic_section(
        deterministic_answer, "Summary", next_heading="Evidence"
    )
    if not fallback_summary:
        fallback_summary = (
            "GWASMind completed the requested evidence retrieval. The detailed result below "
            "is rendered directly from validated evidence records."
        )
    return HybridFinalContext(
        question=str(question),
        safety_review=review,
        sections=sections,
        facts=facts,
        references=references,
        fallback_summary=fallback_summary,
        intent=str(intent),
        source_trait=str(source_trait),
        source_observations=tuple(dict(item) for item in source_observations),
        result=result,
        deterministic_answer=str(deterministic_answer),
        biological_context=context_value,
    )


@dataclass(frozen=True, slots=True)
class FinalEvidenceSelection:
    section_id: str
    fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HybridFinalDraft:
    summary_markdown: str
    evidence_sections: tuple[FinalEvidenceSelection, ...]
    limitations_uncertainty_markdown: str
    limitation_check_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HybridFinalResult:
    status: SafetyStatus
    markdown: str
    draft: HybridFinalDraft
    source: str
    attempts: int
    validation_errors: tuple[str, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    raw_json: str = ""
    corrected_json: str = ""
    raw_json_attempts: tuple[str, ...] = ()
    machine_audit: Mapping[str, Any] = field(default_factory=dict)

    @property
    def rendered_markdown(self) -> str:
        return self.markdown

    @property
    def validated_json(self) -> dict[str, Any]:
        return {
            "summary_markdown": self.draft.summary_markdown,
            "evidence_sections": [asdict(item) for item in self.draft.evidence_sections],
            "limitations_uncertainty": {
                "markdown": self.draft.limitations_uncertainty_markdown,
                "safety_check_ids": list(self.draft.limitation_check_ids),
            },
            "reference_ids": list(self.draft.reference_ids),
        }

    @property
    def issues(self) -> tuple[str, ...]:
        return self.validation_errors

    @property
    def gate2_review(self) -> dict[str, Any]:
        """Final-output review, separate from inherited Gate 1 evidence warnings."""
        from .fact_review_issues import project_fact_issues, visible_report_fact_issues

        validated_draft = self.source in {"model", "model_repaired", "deterministic_repair", "partial_recovery"}
        not_run = not validated_draft or "safety_stopped" in self.validation_errors or any(
            "model_generation_incomplete:" in error for error in self.validation_errors)
        fact_issues = visible_report_fact_issues(
            project_fact_issues(self.machine_audit, target="final_report", visible_output=self.raw_json),
            visible_output=self.validated_json, rendered_markdown=self.markdown,
        )
        recovered = self.source in {"partial_recovery", "evidence_only_recovery"} or bool(
            any(fact_issues.get(key) for key in ('unresolved_count', 'rejected_count', 'unexamined_count', 'omitted_count'))
            or self.machine_audit.get("not_assessed"))
        # Gate 2 removes or corrects unsupported content; it never reports a
        # warning status. fully_verified keeps the audit distinction.
        return {**gate_rule("report_output_grounding").metadata(),
            "check_id": "report_output_grounding", "target": "final_report",
            "status": "NOT_RUN" if not_run else "STOPPED" if self.status is SafetyStatus.STOPPED else "PASS",
            "method": "not_run" if not_run else self.source, "attempts": self.attempts,
            "generated_draft_validated": validated_draft,
            "fully_verified": validated_draft and not recovered and not not_run,
            "not_assessed": list(self.machine_audit.get("not_assessed", ())),
            "validator_ids": list(load_gate_rules().validator_ids("report")) if validated_draft else [],
            "warnings": list(_writer_warning_messages(self.source, self.validation_errors)) if recovered else [],
            "inherited_check_ids": list(self.draft.limitation_check_ids),
            "fact_issues": fact_issues,
            **({"machine_audit": dict(self.machine_audit)} if self.machine_audit else {})}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "markdown": self.markdown,
            "rendered_markdown": self.markdown,
            "validated_json": self.validated_json,
            "source": self.source,
            "attempts": self.attempts,
            "validation_errors": list(self.validation_errors),
            "usage": dict(self.usage),
            "raw_json": self.raw_json,
            "corrected_json": self.corrected_json,
            "raw_json_attempts": list(self.raw_json_attempts),
            "gate2_review": self.gate2_review,
        }


def hybrid_final_messages(context: HybridFinalContext) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_prompt("stage4_hybrid_final_system") + "\n\nUse analysis_scope.computed_facts for counts and the strongest uploaded SNP. Candidate pool, shortlist and returned TopN are different sets. Respect analysis_scope.ranking_provenance: when rules supplied the ranking, explain its available evidence without claiming that you made the ranking decision."},
        *_ld_policy_messages(context),
        {
            "role": "user",
            "content": format_prompt(
                "stage4_hybrid_final_user",
                payload=_canonical_json(context.to_model_payload()),
            ),
        },
    ]


def _ld_policy_messages(context: HybridFinalContext) -> list[dict[str, str]]:
    if _final_ld_report_scope(context).provided:
        return [{"role": "system", "content": load_prompt("ld_expansion_policy")}]
    return []


def hybrid_final_repair_messages(
    context: HybridFinalContext,
    *,
    invalid_content: str,
    validation_error: str,
) -> list[dict[str, str]]:
    payload = {
        "authoritative_context": context.to_model_payload(),
        "validation_error": str(validation_error)[:parameter("gates.repair_error_max_chars")],
        "invalid_output": str(invalid_content)[:parameter("gates.report_repair_input_max_chars")],
        "colocalization_fact_bindings": [
            {"fact_id": fact.fact_id, "subject": fact.model_data.get("subject"),
             "supported_statement": fact.markdown}
            for fact in context.facts if re.search(r"\bH4\b", fact.markdown, re.IGNORECASE)
        ][:50],
    }
    return [
        {"role": "system", "content": (
            "Recover the complete JSON structure of the supplied report. Return exactly one JSON object "
            "with summary_markdown (string), evidence_sections (array of section_id and fact_ids objects), "
            "limitations_uncertainty (object with markdown string and optional safety_check_ids array), "
            "and optional reference_ids array. Preserve the supplied scientific narrative and claims verbatim. "
            "Correct only malformed JSON syntax or missing structural fields. Do not rewrite, soften, expand, "
            "rerank, or repair scientific conclusions or numeric claims; the program will reconcile hard facts "
            "after the complete JSON object is received. Never supply reasoning text outside the JSON."
        )},
        *_ld_policy_messages(context),
        {
            "role": "user",
            "content": _canonical_json(payload),
        },
    ]


def parse_hybrid_final_output(content: str, context: HybridFinalContext) -> HybridFinalDraft:
    """Gate 2 entry shared by initial reports, repairs, and assembled recoveries."""
    validators = {"report_schema_evidence_consistency": _parse_hybrid_final_contract}
    draft = None
    for name in load_gate_rules().validator_ids("report"):
        draft = validators[name](content, context)
    if draft is None:
        raise StructuredOutputError("mandatory report validator was not run")
    return draft


def _parse_hybrid_final_contract(content: str, context: HybridFinalContext) -> HybridFinalDraft:
    value = _strict_json_object(content, label="hybrid final output")
    _require_exact_keys(
        value,
        {
            "summary_markdown",
            "evidence_sections",
            "limitations_uncertainty",
        } | ({"reference_ids"} if "reference_ids" in value else set()),
        label="hybrid final output",
    )
    summary = _bounded_markdown(
        value["summary_markdown"], label="summary_markdown", limit=parameter("gates.report_markdown_max_chars")
    )
    summary_fact_ids = tuple(_SUMMARY_FACT_CITATION_TOKEN.findall(summary))
    if not _REPORT_PROSE_LOCALLY_REVIEWED.get():
        summary_for_validation = _SUMMARY_FACT_CITATION_TOKEN.sub(
            "", _canonicalize_bound_tss_distances(summary, context=context)
        )
        _validate_free_scientific_markdown(summary_for_validation, context=context, label="summary_markdown")
        _validate_source_gwas_summary(summary_for_validation, context=context)
    limitation_value = value["limitations_uncertainty"]
    if not isinstance(limitation_value, dict):
        raise StructuredOutputError("limitations_uncertainty must be an object")
    _require_exact_keys(
        limitation_value,
        {"markdown"} | ({"safety_check_ids"} if "safety_check_ids" in limitation_value else set()),
        label="limitations_uncertainty",
    )
    limitations = _bounded_markdown(
        limitation_value["markdown"],
        label="limitations_uncertainty.markdown",
        limit=parameter("gates.report_markdown_max_chars"),
    )
    limitation_fact_ids = tuple(_SUMMARY_FACT_CITATION_TOKEN.findall(limitations))
    if not _REPORT_PROSE_LOCALLY_REVIEWED.get():
        _validate_task_specific_limitations(limitations, context=context)
        _validate_free_scientific_markdown(
            _SUMMARY_FACT_CITATION_TOKEN.sub("", _canonicalize_bound_tss_distances(limitations, context=context)),
            context=context, label="limitations_uncertainty.markdown")
    limitation_check_ids = _string_array(
        limitation_value.get("safety_check_ids", []),
        label="limitations_uncertainty.safety_check_ids",
        limit=50,
        deduplicate=True,
    )
    known_check_ids = {item.check_id for item in context.safety_review.facts.checks}
    unknown_check_ids = set(limitation_check_ids).difference(known_check_ids)
    if unknown_check_ids:
        raise StructuredOutputError(
            "unknown safety check IDs: " + ", ".join(sorted(unknown_check_ids))
        )
    # Applicable checks were already determined by the safety review. Asking the
    # model to recopy this list adds no grounding; preserve its prose and derive
    # this bookkeeping from the authoritative review instead.
    limitation_check_ids = context.safety_review.facts.nonpassing_check_ids

    rows = value["evidence_sections"]
    if not isinstance(rows, list):
        raise StructuredOutputError("evidence_sections must be an array")
    section_index = {item.section_id: item for item in context.sections}
    fact_index = {item.fact_id: item for item in context.facts}
    seen_sections: set[str] = set()
    seen_facts: set[str] = set()
    seen_evidence: set[str] = set()
    selections: list[FinalEvidenceSelection] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise StructuredOutputError(f"evidence_sections[{index}] must be an object")
        _require_exact_keys(
            row,
            {"section_id", "fact_ids"},
            label=f"evidence_sections[{index}]",
        )
        section_id = _bounded_text(row["section_id"], label="section_id", limit=200)
        if section_id not in section_index:
            raise StructuredOutputError(f"unknown section ID: {section_id}")
        if section_id in seen_sections:
            raise StructuredOutputError(f"duplicate section ID: {section_id}")
        seen_sections.add(section_id)
        fact_ids = _string_array(row["fact_ids"], label=f"{section_id}.fact_ids", limit=parameter("gates.report_fact_selection_limit"))
        if not fact_ids:
            raise StructuredOutputError(f"section {section_id} must select at least one fact")
        selected_facts: list[FinalEvidenceFact] = []
        for fact_id in fact_ids:
            fact = fact_index.get(fact_id)
            if fact is None:
                raise StructuredOutputError(f"unknown fact ID: {fact_id}")
            if fact.section_id != section_id:
                raise StructuredOutputError(
                    f"fact {fact_id} does not belong to section {section_id}"
                )
            if fact_id in seen_facts:
                raise StructuredOutputError(f"duplicate fact ID: {fact_id}")
            seen_facts.add(fact_id)
            selected_facts.append(fact)
        evidence_ids = tuple(
            evidence_id for fact in selected_facts for evidence_id in fact.evidence_ids
        )
        duplicates = set(evidence_ids).intersection(seen_evidence)
        if duplicates:
            raise StructuredOutputError(
                "evidence IDs cannot be repeated across sections: " + ", ".join(sorted(duplicates))
            )
        seen_evidence.update(evidence_ids)
        selections.append(
            FinalEvidenceSelection(
                section_id=section_id,
                fact_ids=fact_ids,
                evidence_ids=evidence_ids,
            )
        )

    required_facts = {item.fact_id for item in context.facts if item.required}
    if not required_facts.issubset(seen_facts):
        raise StructuredOutputError(
            "required facts were omitted: " + ", ".join(sorted(required_facts - seen_facts))
        )
    if context.facts and not selections:
        raise StructuredOutputError("at least one grounded evidence section is required")

    # The validated ranking and deterministic template own display order. The
    # final model selects foreign keys; harmless JSON array reordering must not
    # reorder ranked genes, their evidence modules, or reference numbering.
    section_order = {item.section_id: index for index, item in enumerate(context.sections)}
    fact_order = {item.fact_id: index for index, item in enumerate(context.facts)}
    selections = [
        FinalEvidenceSelection(
            section_id=selection.section_id,
            fact_ids=tuple(sorted(selection.fact_ids, key=fact_order.__getitem__)),
            evidence_ids=tuple(
                evidence_id
                for fact_id in sorted(selection.fact_ids, key=fact_order.__getitem__)
                for evidence_id in fact_index[fact_id].evidence_ids
            ),
        )
        for selection in sorted(selections, key=lambda item: section_order[item.section_id])
    ]

    unknown_summary_facts = set(summary_fact_ids).difference(fact_index)
    if unknown_summary_facts:
        raise StructuredOutputError(
            "summary uses unknown fact IDs: "
            + ", ".join(sorted(unknown_summary_facts))
        )
    unselected_summary_facts = set(summary_fact_ids).difference(seen_facts)
    if unselected_summary_facts:
        raise StructuredOutputError(
            "summary citation facts must also be selected for Evidence: "
            + ", ".join(sorted(unselected_summary_facts))
        )
    unknown_limitation_facts = set(limitation_fact_ids).difference(fact_index)
    if unknown_limitation_facts:
        raise StructuredOutputError("limitations uses unknown fact IDs: " + ", ".join(sorted(unknown_limitation_facts)))
    unselected_limitation_facts = set(limitation_fact_ids).difference(seen_facts)
    if unselected_limitation_facts:
        raise StructuredOutputError("limitations citation facts must also be selected for Evidence: " + ", ".join(sorted(unselected_limitation_facts)))
    if not _REPORT_PROSE_LOCALLY_REVIEWED.get():
        validate_report_fragment(summary, context, label="summary_markdown")
        validate_report_fragment(limitations, context, label="limitations_uncertainty.markdown")
    # A known context/source-data fact can be grounded without a publication.
    # Its anchor renders as empty text; it must not invent a bibliography entry
    # or invalidate an otherwise verified report.

    reference_ids = _string_array(
        value.get("reference_ids", []), label="reference_ids", limit=parameter("gates.report_reference_limit"), deduplicate=True
    )
    reference_index = {item.reference_id: item for item in context.references}
    unknown_references = set(reference_ids).difference(reference_index)
    if unknown_references:
        raise StructuredOutputError(
            "unknown reference IDs: " + ", ".join(sorted(unknown_references))
        )
    expected_references = tuple(
        reference_id
        for selection in selections
        for fact_id in selection.fact_ids
        for reference_id in fact_index[fact_id].reference_ids
    )
    expected_references = tuple(dict.fromkeys(expected_references))
    # Bibliography membership is a deterministic join, not a scientific decision.
    # Complete missing known references and remove unused known references without
    # spending another model request copying a foreign-key union. Unknown references
    # still fail above, and the rendered bibliography always matches actual facts.
    reference_ids = expected_references
    if reference_ids and not summary_fact_ids and not _REPORT_PROSE_LOCALLY_REVIEWED.get():
        raise StructuredOutputError(
            "summary must place at least one {{cite_fact:fact_id}} anchor after its "
            "supported claim"
        )
    return HybridFinalDraft(
        summary_markdown=summary,
        evidence_sections=tuple(selections),
        limitations_uncertainty_markdown=limitations,
        limitation_check_ids=limitation_check_ids,
        reference_ids=reference_ids,
    )


def _strip_outer_report_heading(text: str, section: str) -> str:
    """Remove duplicate field titles while preserving substantive prose."""
    if section == "summary":
        # A model may put the field's title and its paragraph on the same line.
        # Require a marked title or explicit punctuation, never a bare word.
        prefix = re.compile(r"^(?:#{1,6}\s+)?(?:(?P<mark>\*\*|__)Summary[.:]?(?P=mark)|Summary[.:])\s+", re.IGNORECASE)
        text = text.strip()
        while (match := prefix.match(text)) is not None:
            remainder = text[match.end():]
            if re.match(r"statistics?\b", remainder, re.IGNORECASE):
                break
            text = remainder
    expected = {"summary"} if section == "summary" else {
        "limitations", "limitations & uncertainty", "limitations and uncertainty",
    }
    lines = text.strip().splitlines()
    while lines:
        heading = re.sub(r"^#{1,6}\s+|\s+#+$", "", lines[0].strip()).strip("*_ ").rstrip(":." if section == "summary" else ":").strip().casefold()
        if heading not in expected:
            break
        lines.pop(0)
        if lines and re.fullmatch(r"\s*(?:={3,}|-{3,})\s*", lines[0]):
            lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def _public_program_limitations(text: str, review: SafetyReview) -> str:
    """Format exact program facts; leave other narrative and scientific claims intact."""
    replacements = {}
    context_messages = set()
    for check in review.facts.checks:
        if not check.nonpassing or check.check_id not in {"optional_evidence_coverage", "biological_context_coverage"}:
            continue
        messages = _public_check_warning_messages(check)
        replacements[check.fact] = " ".join(messages)
        if check.check_id == "biological_context_coverage":
            context_messages.update(messages)
            if check.fact in {_BIOLOGICAL_CONTEXT_COVERAGE_WARNING, _LEGACY_BIOLOGICAL_CONTEXT_COVERAGE_WARNING}:
                # This is an exact controlled sentence, not a rewrite of free prose.
                text = text.replace(_LEGACY_BIOLOGICAL_CONTEXT_COVERAGE_WARNING, _BIOLOGICAL_CONTEXT_COVERAGE_WARNING)
    lines = []
    seen_context = set()
    for line in text.splitlines():
        match = re.fullmatch(r"(\s*(?:[-*]\s+)?)(.*)", line)
        prefix, content = match.groups()
        content = replacements.get(content, content)
        if content in context_messages and content in seen_context:
            continue
        seen_context.update(message for message in context_messages if message in content)
        lines.append(prefix + content)
    return "\n".join(lines).strip()


def render_hybrid_final(
    context: HybridFinalContext,
    draft: HybridFinalDraft,
    *,
    status: SafetyStatus | None = None,
    response_source: str | None = None,
    validation_errors: Sequence[str] = (),
    _row_limit: int | None = None,
) -> str:
    """Render facts and bibliography deterministically after all foreign-key checks."""

    from .report_evidence_presentation import compact_evidence_fact, is_native_gene_report
    from .report_presentation import bibliography_presentation_text, llm_presentation_text
    from .web.gene_evidence import italicize_gene_symbols

    effective_status = status or context.safety_review.status
    row_limit = parameter("report.evidence.max_rows_per_category") if _row_limit is None else _row_limit
    if effective_status is SafetyStatus.STOPPED:
        return llm_presentation_text(_render_stopped_paragraph(context)) + "\n"
    fact_index = {item.fact_id: item for item in context.facts}
    section_index = {item.section_id: item for item in context.sections}
    reference_index = {item.reference_id: item for item in context.references}
    reference_numbers = {
        reference_id: index for index, reference_id in enumerate(draft.reference_ids, start=1)
    }
    summary, _ = _render_summary_fact_citation_tokens(
        _strip_outer_report_heading(draft.summary_markdown, "summary"),
        fact_index,
        reference_numbers,
        reference_index,
        source_observations_present=bool(context.source_observations),
    )
    # The completed model owns its Summary. Rendering may resolve citation
    # anchors and typography, but must not inject or delete scientific prose.
    limitations = _public_program_limitations(
        _strip_outer_report_heading(draft.limitations_uncertainty_markdown, "limitations"), context.safety_review,
    )
    limitations, _ = _render_summary_fact_citation_tokens(
        limitations, fact_index, reference_numbers, reference_index,
        source_observations_present=bool(context.source_observations),
    )
    summary = _link_numbered_citations(
        summary,
        draft.reference_ids,
        reference_index,
    )
    blocks = [f"**Report quality: {effective_status.value}**"]
    review_details: tuple[str, ...] = ()
    if effective_status is SafetyStatus.PASS_WITH_WARNINGS:
        opening, review_details = _report_warning_presentation(context.safety_review, response_source, validation_errors)
        if opening:
            blocks.append("**Warning:** " + opening)
        if review_details:
            # These exact program messages are retained once in Review details.
            # Preserve every other scientific paragraph, including paraphrases.
            repeated = set(review_details)
            limitations = "\n\n".join(paragraph for paragraph in re.split(r"\n\s*\n", limitations)
                                        if paragraph.strip() not in repeated)
    blocks.extend(["## **Summary**", summary])

    def section_body(selection: FinalEvidenceSelection) -> str:
        section_details: list[str] = []
        shown, omitted = {}, {}
        native_cards = is_native_gene_report(context.result)
        for fact_id in selection.fact_ids:
            fact = fact_index[fact_id]
            category = fact.model_data.get("evidence_module")
            if (native_cards and category and fact.markdown.startswith("*Datasource:")
                    and shown.get(category, 0) >= row_limit):
                continue
            if native_cards and category and fact.markdown.lstrip().startswith("- "):
                category = ("Colocalization" if " coloc " in fact.markdown else category)
                if shown.get(category, 0) >= row_limit:
                    omitted[category] = omitted.get(category, 0) + 1
                    continue
                shown[category] = shown.get(category, 0) + 1
            markdown, inline_reference_ids = _render_inline_reference_tokens(
                _normalize_evidence_category_heading(fact.markdown.strip()),
                reference_numbers,
                reference_index,
            )
            citations = " ".join(
                _reference_marker(
                    reference_id,
                    reference_numbers[reference_id],
                    reference_index,
                )
                for reference_id in fact.reference_ids
                if reference_id in reference_numbers
                and reference_id not in inline_reference_ids
            )
            if citations:
                markdown = markdown.rstrip(".") + f" {citations}."
            markdown = compact_evidence_fact(markdown, model_data=fact.model_data, result=context.result)
            if section_details:
                separator = fact.model_data.get("separator_before", "\n\n")
                section_details.append("\n" if separator == "\n" else "\n\n")
            section_details.append(markdown)
        if omitted:
            section_details.append("\n\nAdditional evidence is available in the gene evidence card and the complete archive.")
        return "".join(section_details)

    context_selections = tuple(selection for selection in draft.evidence_sections
        if _clean_heading(section_index[selection.section_id].heading).casefold() == "biological context")
    detail_selections = tuple(selection for selection in draft.evidence_sections if selection not in context_selections)
    if context_selections:
        blocks.extend(["## **Biological Context**", "\n\n".join(section_body(selection) for selection in context_selections)])
    blocks.append("## **Evidence**")
    if not detail_selections:
        blocks.append("No grounded evidence detail was selected for display.")
        if context.source_observations:
            blocks.extend(["### Uploaded GWAS observations", _source_gwas_status_sentence(context)])
    for selection in detail_selections:
        heading = section_index[selection.section_id].heading
        if context.intent == 'prioritize_genes' and context.result.get('candidate_decisions'):
            ids = set(re.findall(r'\bENSG\d+(?:\.\d+)?\b', heading))
            choices = [row for row in context.result['candidate_decisions']
                if isinstance(row, Mapping) and row.get('gene_id') in ids]
            if len(choices) == 1:
                heading += ' · Shortlist' if choices[0].get('status') == 'selected' else ' · Exploratory ranking'
        blocks.extend([f"### {heading}", section_body(selection)])
    if context.intent == 'prioritize_genes':
        from .selection_evidence import unranked_locus_evidence_markdown
        other_candidates = unranked_locus_evidence_markdown(context.result, _runtime_evidence_rows(context.result))
        if other_candidates:
            blocks.append(other_candidates)
    blocks.extend(("## **Limitations & Uncertainty**", limitations))
    if review_details:
        blocks.extend(("[Review details](#gwasmind-review-details)",
                       "\n".join("- " + message for message in review_details)))
    retained_references = draft.reference_ids
    if is_native_gene_report(context.result):
        displayed_numbers = set(re.findall(r"\[\[(\d+)\]\]\(", "\n".join(blocks)))
        retained_references = tuple(ref for ref in retained_references
            if str(reference_numbers[ref]) in displayed_numbers)
        renumber = {str(reference_numbers[ref]): str(index)
                    for index, ref in enumerate(retained_references, 1)}
        blocks = [re.sub(r"\[\[(\d+)\]\]\(", lambda match: "[[" + renumber.get(match[1], match[1]) + "]](", block)
                  for block in blocks]
        reference_numbers = {ref: index for index, ref in enumerate(retained_references, 1)}
    blocks.append("## **References**")
    if retained_references:
        blocks.extend(
            f"[{reference_numbers[reference_id]}] {reference_index[reference_id].markdown.strip()}"
            for reference_id in retained_references
        )
    else:
        from .processing_scope import is_source_only_processing_limit
        blocks.append(
            "External references in the retained result have not been reviewed for this response."
            if is_source_only_processing_limit(context.result) else
            "No traceable literature reference was available for the displayed evidence."
        )
    rendered = "\n\n".join(blocks).strip() + "\n"
    symbols = tuple(
        match.group(1)
        for section in context.sections
        if (match := re.match(r"^([A-Za-z][A-Za-z0-9-]*)\s+\(ENSG\d+\)", section.heading))
    )
    body, separator, bibliography = rendered.partition("**References**")
    from .report_evidence_presentation import source_orientation_presentation, retrieval_absence_citations
    body = source_orientation_presentation(body, context.result)
    body = retrieval_absence_citations(body, context.result)
    displayed = (italicize_gene_symbols(llm_presentation_text(body), symbols)
                 + separator + bibliography_presentation_text(bibliography))
    if (is_native_gene_report(context.result) and row_limit > 1
            and len(displayed) > parameter("report.evidence.target_max_chars")):
        # Apply the same smaller preview count to every candidate. Scientific
        # prose, rank order and full archived facts are never truncated.
        return render_hybrid_final(context, draft, status=status, response_source=response_source,
                                   validation_errors=validation_errors, _row_limit=row_limit - 1)
    return displayed


def _normalize_evidence_category_heading(markdown: str) -> str:
    """Promote legacy numbered evidence labels without changing their facts."""
    return re.sub(r"(?m)^\*\*(?:\d+[.)]\s+)([^\n]+?)\*\*:?[ \t]*$", r"#### \1", markdown)


def _render_stopped_paragraph(context: HybridFinalContext) -> str:
    """Explain an immutable hard stop in one deterministic, specific paragraph."""

    facts = context.safety_review.facts
    failed_checks = tuple(
        item
        for item in facts.checks
        if item.severity is SafetySeverity.HARD_GATE
        and item.outcome is SafetyOutcome.FAIL
    )
    status_by_module = dict(facts.module_statuses)
    incomplete_modules = tuple(
        (module, status_by_module.get(module, "missing"))
        for module in facts.required_modules
        if status_by_module.get(module, "missing") not in _COMPLETE_MODULE_STATUSES
    )
    validation_detail = _public_ranking_validation_failure(context.result)
    ranking_failed = any(
        item.check_id == "model_output_grounding" for item in failed_checks
    ) or any(module == "candidate_ranking" for module, _ in incomplete_modules)
    action = (
        "GWASMind could not produce a candidate-gene ranking."
        if context.intent.strip().casefold() == "prioritize_genes"
        else "GWASMind could not produce the requested evidence-backed result."
    )
    details: list[str] = [str(issue["message"]) for issue in facts.operation_issues
                        if issue.get("blocking") is True and isinstance(issue.get("message"), str)]
    if any(issue.get("code") == "S1" for issue in facts.operation_issues):
        details.extend(check.fact for check in facts.checks if check.nonpassing
                       and check.check_id in {"variant_integrity", "build_integrity", "allele_integrity"})
    if ranking_failed and validation_detail:
        details.append(
            f"Evidence retrieval returned {facts.evidence_record_count:,} record(s), but "
            f"the candidate-ranking response failed deterministic validation: "
            f"{validation_detail}."
        )
    else:
        reportable_modules = tuple(
            (module, status)
            for module, status in incomplete_modules
            if module != "candidate_ranking"
        )
        if reportable_modules:
            formatted = ", ".join(
                f"{module} ({status})" for module, status in reportable_modules
            )
            details.append(f"Required modules did not complete: {formatted}.")
        elif ranking_failed:
            details.append(
                "The candidate-ranking response did not pass deterministic schema and "
                "evidence-link validation."
            )
    details.extend(
        item.fact
        for item in failed_checks
        if item.check_id != "required_modules"
    )
    if not details:
        details.append(
            "A required validation check failed, but no more specific safe diagnostic "
            "was available."
        )
    if ranking_failed:
        conclusion = (
            "The locus evidence was not treated as absent; only the invalid structured "
            "ranking was rejected."
        )
    elif incomplete_modules:
        conclusion = (
            "This technical retrieval gap is not evidence that the locus lacks relevant "
            "genes or molecular links."
        )
    else:
        conclusion = (
            "Without those required inputs or evidence, a defensible result cannot be produced."
        )
    paragraph = " ".join(
        (
            _source_gwas_status_sentence(context),
            action,
            *details,
            conclusion,
        )
    )
    return re.sub(r"\s+", " ", paragraph).strip()


def _public_ranking_validation_failure(result: Mapping[str, Any]) -> str | None:
    """Return one bounded program-generated ranking error without provider details."""

    sufficiency = result.get("sufficiency")
    if not isinstance(sufficiency, Mapping):
        return None
    reasons = sufficiency.get("reasons")
    if not isinstance(reasons, (list, tuple)):
        return None
    for value in reasons:
        if not isinstance(value, str):
            continue
        reason = re.sub(r"\s+", " ", value).strip()
        if not reason.startswith("INCOMPLETE:"):
            continue
        reason = reason.removeprefix("INCOMPLETE:").strip()
        if reason.startswith("output validation failed:"):
            reason = reason.removeprefix("output validation failed:").strip()
        if not reason:
            continue
        return reason[:500].rstrip(" .")
    return None


def run_hybrid_final(
    context: HybridFinalContext,
    completer: Callable[..., Any] | None,
    *,
    max_tokens: int = parameter("gates.report_max_tokens"),
    progress_sink: ProgressSink | None = None,
) -> HybridFinalResult:
    """Review complete report JSON locally; retry malformed structure at most once."""

    with report_validation_session(context):
        return _run_hybrid_final(context, completer, max_tokens=max_tokens, progress_sink=progress_sink)


def _run_hybrid_final(
    context: HybridFinalContext, completer: Callable[..., Any] | None, *, max_tokens: int,
    progress_sink: ProgressSink | None = None,
) -> HybridFinalResult:

    if context.safety_review.status is SafetyStatus.STOPPED:
        draft = _stopped_final_draft(context)
        return HybridFinalResult(
            status=SafetyStatus.STOPPED,
            markdown=render_hybrid_final(context, draft, status=SafetyStatus.STOPPED),
            draft=draft,
            source="deterministic_fallback",
            attempts=0,
            validation_errors=("safety_stopped",),
        )
    if completer is None:
        return _fallback_final_result(context, attempts=0, errors=("model_not_configured",))
    errors: list[str] = []
    try:
        messages = hybrid_final_messages(context)
        emit_progress(progress_sink, "response_substage", substage="inference", status="running")
        draft_content, draft_usage = _complete_text_with_usage(
            completer, messages, max_tokens=max_tokens
        )
    except CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider failure must retain the evidence report
        incomplete = getattr(exc, "finish_reason", None) not in (None, "stop")
        result = _fallback_final_result(
            context,
            attempts=1,
            errors=((f"model_generation_incomplete:{exc.finish_reason}" if incomplete
                     else f"model_call_failed:{type(exc).__name__}"),),
            usage=getattr(exc, "usage", {}),
        )
        raw = getattr(exc, "raw_content", "")
        return _attach_final_machine_audit(result, (raw,)) if isinstance(raw, str) and raw else result
    emit_progress(progress_sink, "response_substage", substage="fact_verification", status="running")
    reviewed = _review_complete_final_candidate(
        context, draft_content, raw_contents=(draft_content,), errors=errors,
        usage=draft_usage, attempts=1,
    )
    if reviewed is not None:
        return reviewed

    repair_usage: dict[str, int] = {}
    try:
        messages = hybrid_final_repair_messages(
            context, invalid_content=draft_content, validation_error=errors[-1],
        )
        emit_progress(progress_sink, "response_substage", substage="inference", status="running")
        repaired_content, repair_usage = _complete_text_with_usage(
            completer, messages,
            max_tokens=max_tokens,
        )
    except CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider failure is not validation failure
        incomplete = getattr(exc, "finish_reason", None) not in (None, "stop")
        errors.append(f"model_generation_incomplete:{exc.finish_reason}" if incomplete
                      else f"repair_call:{type(exc).__name__}")
        repair_usage = getattr(exc, "usage", {})
        result = _fallback_final_result(
            context,
            attempts=2,
            errors=tuple(errors),
            usage=_merge_usage(draft_usage, repair_usage),
        )
        raw = getattr(exc, "raw_content", None)
        raw_contents = (draft_content, raw) if isinstance(raw, str) and raw else (draft_content,)
        return _attach_final_machine_audit(result, raw_contents)
    emit_progress(progress_sink, "response_substage", substage="fact_verification", status="running")
    reviewed = _review_complete_final_candidate(
        context, repaired_content, raw_contents=(draft_content, repaired_content), errors=errors,
        usage=_merge_usage(draft_usage, repair_usage), attempts=2,
    )
    if reviewed is not None:
        return reviewed
    result = _evidence_only_final_result(context, attempts=2, errors=tuple(errors),
                                        usage=_merge_usage(draft_usage, repair_usage))
    return _attach_final_machine_audit(result, (draft_content, repaired_content))


def _complete_report_structure(content: str) -> bool:
    """A complete narrative is never sent back for semantic model repair."""
    try:
        value = _strict_json_object(content, label="hybrid final output")
    except StructuredOutputError:
        return False
    limitations = value.get("limitations_uncertainty")
    rows = value.get("evidence_sections")
    return (isinstance(value.get("summary_markdown"), str)
            and isinstance(limitations, dict) and isinstance(limitations.get("markdown"), str)
            and isinstance(rows, list) and all(isinstance(row, dict)
                and isinstance(row.get("section_id"), str) and isinstance(row.get("fact_ids"), list)
                and all(isinstance(identifier, str) for identifier in row["fact_ids"]) for row in rows))


def _attach_final_machine_audit(
    result: HybridFinalResult, raw_contents: tuple[str, ...], *,
    corrected_content: str | None = None, audit: Mapping[str, Any] | None = None,
) -> HybridFinalResult:
    """Keep exact model bytes beside the final report and a content-addressed receipt."""
    if corrected_content is None:
        corrected_content = _canonical_json({
            "summary_markdown": result.draft.summary_markdown,
            "evidence_sections": [{"section_id": item.section_id, "fact_ids": list(item.fact_ids)}
                                  for item in result.draft.evidence_sections],
            "limitations_uncertainty": {"markdown": result.draft.limitations_uncertainty_markdown,
                                        "safety_check_ids": list(result.draft.limitation_check_ids)},
            "reference_ids": list(result.draft.reference_ids),
        })
    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    receipt = {"schema_version": "1", "corrections": [], **dict(audit or {})}
    if receipt.get("corrected_json_sha256"):
        receipt["reconciled_json_sha256"] = receipt["corrected_json_sha256"]
    receipt.update(raw_json_sha256=digest(raw_contents[0]),
                   corrected_json_sha256=digest(corrected_content),
                   model_attempt_sha256=[digest(content) for content in raw_contents],
                   recovery=result.source, semantic_repair_calls=0)
    return replace(result, raw_json=raw_contents[0], corrected_json=corrected_content,
                   raw_json_attempts=raw_contents, machine_audit=receipt)


def _review_complete_final_candidate(
    context: HybridFinalContext, content: str, *, raw_contents: tuple[str, ...],
    errors: list[str], usage: Mapping[str, int], attempts: int,
) -> HybridFinalResult | None:
    from .report_hard_facts import review_report_hard_facts
    from .report_recovery import repair_report_bookkeeping, normalize_nonrendered_bookkeeping

    if not _complete_report_structure(content):
        errors.append(("draft" if attempts == 1 else "repair") + ":incomplete_or_malformed_json_structure")
        return None
    reviewed_content, discarded_fields = normalize_nonrendered_bookkeeping(content)
    from .gate2_hallucination import remove_report_identifier_hallucinations, remove_report_rsid_mismatches
    # A wrong rsID beside a source variant must not block that variant's own values.
    result = context.result if isinstance(context.result, Mapping) else {}
    reviewed_content, rsid_removals = remove_report_rsid_mismatches(reviewed_content, result.get("ld_expansion"))
    corrected_content, audit = review_report_hard_facts(reviewed_content, context)
    if rsid_removals:
        audit["hallucination_removals"] = rsid_removals
    if discarded_fields:
        audit['discarded_nonrendered_fields'] = discarded_fields
        audit['normalized_input_sha256'] = hashlib.sha256(reviewed_content.encode('utf-8')).hexdigest()
    from .report_identifier_facts import report_identifier_findings
    identifier_findings = report_identifier_findings(corrected_content, context)
    corrected_content, removals = remove_report_identifier_hallucinations(corrected_content, identifier_findings)
    if removals:
        audit["hallucination_removals"] = [*audit.get("hallucination_removals", []), *removals]
        identifier_findings = report_identifier_findings(corrected_content, context)
    if identifier_findings:
        audit.setdefault("unverified_measurements", []).extend(identifier_findings)
    # Gate 2 findings are audit-only; report quality follows Gate 1.
    reviewed_status = context.safety_review.status
    prefix = "draft" if attempts == 1 else "repair"
    if audit.get("corrections"):
        errors.append(f"{prefix}:hard_fact_values_corrected:{len(audit['corrections'])}")
    if audit.get("unverified_measurements"):
        errors.append(f"unverified_report_measurements:{len(audit['unverified_measurements'])}")
    if identifier_findings:
        errors.append(f"unverified_report_identifiers:{len(identifier_findings)}")
    public_errors = tuple(error for error in errors
        if not error.startswith(("unverified_report_measurements:", "unverified_report_identifiers:")))
    review_token = _REPORT_PROSE_LOCALLY_REVIEWED.set(True)
    try:
        try:
            draft = parse_hybrid_final_output(corrected_content, context)
        except StructuredOutputError as exc:
            errors.append(f"{prefix}:{exc}")
            draft = repair_report_bookkeeping(corrected_content, context)
            if draft is not None:
                status = reviewed_status
                result = HybridFinalResult(status=status,
                    markdown=render_hybrid_final(context, draft, status=status, response_source="deterministic_repair", validation_errors=public_errors), draft=draft,
                    source="deterministic_repair" if attempts == 1 else "model_repaired", attempts=attempts,
                    validation_errors=tuple(errors), usage=usage)
                return _attach_final_machine_audit(result, raw_contents, audit=audit)
        else:
            source = "model_repaired" if attempts > 1 else "deterministic_repair" if audit.get("corrections") or audit.get("unverified_measurements") or discarded_fields else "model"
            status = reviewed_status
            result = HybridFinalResult(status=status,
                markdown=render_hybrid_final(context, draft, status=status, response_source=source, validation_errors=public_errors), draft=draft, source=source, attempts=attempts,
                validation_errors=tuple(errors), usage=usage)
            return _attach_final_machine_audit(result, raw_contents, corrected_content=corrected_content, audit=audit)
    finally:
        _REPORT_PROSE_LOCALLY_REVIEWED.reset(review_token)

    # A complete narrative survives even an unexpected registry/format problem.
    # Keep its prose verbatim and expose the unresolved structural review.
    value = _strict_json_object(corrected_content, label="complete report preservation")
    draft = replace(_fallback_final_draft(context), summary_markdown=value["summary_markdown"],
        limitations_uncertainty_markdown=value["limitations_uncertainty"]["markdown"])
    audit["not_assessed"] = ["report_registry_contract"]
    errors.append("report_registry_contract_not_assessed")
    status = context.safety_review.status
    result = HybridFinalResult(status=status,
        markdown=render_hybrid_final(context, draft, status=status,
            response_source="deterministic_repair", validation_errors=errors),
        draft=draft, source="deterministic_repair", attempts=attempts,
        validation_errors=tuple(errors), usage=usage)
    return _attach_final_machine_audit(result, raw_contents, audit=audit)


def _partial_final_result(
    context: HybridFinalContext, contents: Sequence[str], *, attempts: int,
    errors: tuple[str, ...], usage: Mapping[str, int],
) -> HybridFinalResult | None:
    from .report_recovery import recover_report_fragments
    for content in contents:
        draft = recover_report_fragments(content, context)
        if draft is None:
            continue
        warning = _writer_warning_messages("partial_recovery", errors)[0]
        draft = replace(draft, limitations_uncertainty_markdown=(
            draft.limitations_uncertainty_markdown + "\n\n" + warning
        ))
        status = context.safety_review.status
        return HybridFinalResult(
            status=status,
            markdown=render_hybrid_final(context, draft, status=status,
                                         response_source="partial_recovery", validation_errors=errors),
            draft=draft, source="partial_recovery", attempts=attempts,
            validation_errors=errors, usage=usage,
        )
    return None


def evidence_facts_from_records(
    records: Sequence[EvidenceRecord | Mapping[str, Any]],
    *,
    section_id: str = "retrieved_evidence",
) -> tuple[FinalEvidenceFact, ...]:
    """Create a conservative generic fact ledger for callers without a richer renderer.

    Gene-prioritization callers should normally supply richer deterministic facts. This
    helper deliberately exposes only identifiers, type, tissue, and source rather than
    asking the model to copy statistics out of arbitrary nested payloads.
    """

    facts: list[FinalEvidenceFact] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        evidence_id = _record_text(record, "evidence_id")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        kind = _record_kind(record)
        gene_id = _record_text(record, "gene_id")
        variant_id = _record_text(record, "variant_id")
        tissue = _record_text(record, "tissue")
        source = _record_text(record, "source")
        pmid = _record_text(record, "pmid")
        details = [kind or "evidence"]
        if gene_id:
            details.append(f"gene {gene_id}")
        if variant_id:
            details.append(f"variant {variant_id}")
        if tissue:
            details.append(f"context {tissue}")
        if source:
            details.append(f"source {source}")
        reference_ids = (f"pmid:{pmid}",) if pmid else ()
        facts.append(
            FinalEvidenceFact(
                fact_id=f"fact_{index}",
                section_id=section_id,
                evidence_ids=(evidence_id,),
                markdown="; ".join(details) + ".",
                reference_ids=reference_ids,
                model_data={
                    "kind": kind,
                    "gene_id": gene_id,
                    "variant_id": variant_id,
                    "tissue": tissue,
                    "source": source,
                },
            )
        )
    return tuple(facts)


def _registries_from_deterministic_answer(
    deterministic_answer: str,
    *,
    evidence_rows: tuple[Mapping[str, Any], ...],
) -> tuple[tuple[FinalSection, ...], tuple[FinalEvidenceFact, ...], tuple[FinalReference, ...]]:
    evidence_text = _deterministic_section(
        deterministic_answer, "Evidence", next_heading="References"
    )
    summary_text = _deterministic_section(
        deterministic_answer, "Summary", next_heading="Evidence"
    )
    reference_text = _deterministic_section(deterministic_answer, "References")
    references_by_number: dict[str, FinalReference] = {}
    for number, definition in re.findall(
        r"(?ms)^\[(\d+)\]\s+(.+?)(?=^\[\d+\]\s+|\Z)",
        reference_text.strip(),
    ):
        cleaned = definition.strip()
        if not cleaned:
            continue
        references_by_number[number] = FinalReference(
            reference_id=f"reference_{number}",
            markdown=cleaned,
        )

    parsed_sections = _split_evidence_sections(evidence_text)
    biological_context = _deterministic_section(deterministic_answer, "Biological Context", next_heading="Evidence")
    if biological_context:
        parsed_sections = (("Biological Context", biological_context), *parsed_sections)
    if not parsed_sections and evidence_text.strip():
        parsed_sections = (("Retrieved evidence", evidence_text.strip()),)
    row_terms = tuple((_runtime_record_terms(row), row) for row in evidence_rows)
    used_evidence_ids: set[str] = set()
    sections: list[FinalSection] = []
    facts: list[FinalEvidenceFact] = []
    for index, (heading, body) in enumerate(parsed_sections, start=1):
        section_id = f"section_{index:03d}"
        sections.append(FinalSection(section_id=section_id, heading=_clean_heading(heading)))
        searchable = _grounding_text(f"{heading} {body}")
        matched_ids: list[str] = []
        matched_data: list[dict[str, Any]] = []
        is_context_section = "biological context" in _grounding_text(heading)
        if not is_context_section:
            for terms, row in row_terms:
                evidence_id = str(row.get("evidence_id") or "").strip()
                if not evidence_id or evidence_id in used_evidence_ids:
                    continue
                gene_terms = tuple(term for term in terms if term)
                if gene_terms and any(
                    _term_in_text(term, searchable) for term in gene_terms
                ):
                    matched_ids.append(evidence_id)
                    used_evidence_ids.add(evidence_id)
                    matched_data.append(_bounded_runtime_record(row))
        def citation_token(match: re.Match[str]) -> str:
            reference = references_by_number.get(match.group(1))
            return f"{{{{cite:{reference.reference_id}}}}}" if reference else ""

        clean_body = _NUMBERED_CITATION.sub(citation_token, body)
        clean_body = re.sub(r"(?m)^\s*---+\s*$", "", clean_body).strip()
        claims = _split_display_claims(clean_body)
        # Keep every record in the section audit once. Citation ownership is narrower:
        # each displayed claim carries only the references originally attached to it.
        # This prevents one gene-profile anchor from citing unrelated expression,
        # function and QTL papers together in the Summary.
        claim_subject = _clean_heading(heading)
        for claim_index, (claim, separator, module) in enumerate(claims):
            # Describe uses variant subheadings inside one External annotations
            # section. Each subsequent bullet belongs to that variant, not to
            # every variant and measurement in the section.
            if re.match(r"^\*\*(?:rs\d+|`)", claim) and _REPORT_VARIANT.search(claim):
                claim_subject = " ".join(re.sub(r"[*`]", "", claim).split())[:300]
            fact_id = (
                f"fact_{index:03d}" if claim_index == 0
                else f"fact_{index:03d}_{claim_index + 1:02d}"
            )
            reference_ids = tuple(dict.fromkeys(_INLINE_REFERENCE_TOKEN.findall(claim)))
            facts.append(
                FinalEvidenceFact(
                    fact_id=fact_id,
                    section_id=section_id,
                    evidence_ids=tuple(matched_ids) if claim_index == 0 else (),
                    markdown=claim,
                    reference_ids=reference_ids,
                    model_data={
                        "subject": claim_subject,
                        "evidence_module": module,
                        "deterministic_detail": claim[:parameter("gates.report_fact_prompt_max_chars")],
                        "deterministic_detail_truncated": len(claim) > parameter("gates.report_fact_prompt_max_chars"),
                        "separator_before": separator,
                        "evidence_records": matched_data if claim_index == 0 else [],
                    },
                    required=True,
                )
            )

    # A deterministic answer without explicit Evidence detail still receives one bounded
    # fact. This keeps malformed model output recoverable without inventing scientific text.
    if not sections:
        sections.append(FinalSection(section_id="section_001", heading="Evidence"))
        facts.append(
            FinalEvidenceFact(
                fact_id="fact_001",
                section_id="section_001",
                evidence_ids=(),
                markdown="No deterministic evidence detail was available for display.",
                model_data={"deterministic_detail": ""},
                required=True,
            )
        )
    used_reference_ids = {
        reference_id for fact in facts for reference_id in fact.reference_ids
    }
    used_reference_ids.update(
        references_by_number[number].reference_id
        for number in dict.fromkeys(re.findall(r"\[(\d+)\]", summary_text))
        if number in references_by_number
    )
    references = tuple(
        reference
        for reference in references_by_number.values()
        if reference.reference_id in used_reference_ids
    )
    return tuple(sections), tuple(facts), references


def _split_display_claims(body: str) -> tuple[tuple[str, str, str], ...]:
    """Split trusted paragraphs/list items without changing their display layout."""

    chunks: list[tuple[str, str, str]] = []
    pending: list[str] = []
    separator = "\n\n"
    module = ""
    pending_module = ""

    def flush() -> None:
        if pending:
            chunks.append(("\n".join(pending).strip(), separator, pending_module))
            pending.clear()

    for line in body.splitlines():
        if not line.strip():
            flush()
            separator = "\n\n"
            continue
        if pending and re.match(r"^(?:[-*+] |\*\*\d+[.)]\s|####\s|\*Datasource:)", line):
            flush()
            separator = "\n"
        if re.match(r"^(?:\*\*\d+[.)]\s|####\s)", line):
            module = _clean_heading(line.rstrip(":"))
        if not pending:
            pending_module = module
        pending.append(line)
    flush()
    return tuple(chunks) or (("No deterministic detail was available for this section.", "", ""),)


def _split_evidence_sections(text: str) -> tuple[tuple[str, str], ...]:
    matches = list(re.finditer(r"(?m)^###\s+(.+?)\s*$", text))
    if not matches:
        return ()
    rows: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        rows.append(("Overview", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            rows.append((match.group(1).strip(), body))
    return tuple(rows)


def _deterministic_section(
    answer: str,
    heading: str,
    *,
    next_heading: str | None = None,
) -> str:
    def heading_pattern(label: str) -> str:
        return rf"(?mi)^(?:##[ \t]+)?(?:\*\*{re.escape(label)}\*\*|{re.escape(label)})[ \t]*$"

    start_match = re.search(heading_pattern(heading), str(answer))
    if not start_match:
        return ""
    start = start_match.end()
    if next_heading:
        end_match = re.search(
            heading_pattern(next_heading),
            str(answer)[start:],
        )
        end = start + end_match.start() if end_match else len(str(answer))
    else:
        end = len(str(answer))
    # Biological Context is a sibling of Summary, not part of its prose.
    if heading.casefold() == "summary":
        context_match = re.search(heading_pattern("Biological Context"), str(answer)[start:end])
        if context_match:
            end = start + context_match.start()
    return str(answer)[start:end].strip()


@memoized_report_derivation
def _runtime_evidence_rows(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    for key in ("evidence", "evidence_ledger"):
        value = result.get(key)
        if isinstance(value, list):
            return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _runtime_record_terms(row: Mapping[str, Any]) -> tuple[str, ...]:
    terms: list[str] = []
    gene_id = str(row.get("gene_id") or "").strip()
    if gene_id:
        terms.append(gene_id)
    payload = row.get("payload")
    if isinstance(payload, Mapping):
        for key, value in _walk_mapping_items(payload):
            if key.casefold() in {
                "approvedsymbol",
                "genesymbol",
                "symbol",
                "qtigenesymbol",
                "qtlgenesymbol",
                "rightqtlgenesymbol",
            }:
                text = str(value or "").strip()
                if text and len(text) <= 100:
                    terms.append(text)
            elif key.casefold() == "label" and re.fullmatch(
                r"[A-Za-z][A-Za-z0-9-]{1,30}", str(value or "")
            ):
                terms.append(str(value).strip())
    return tuple(dict.fromkeys(terms))


def _walk_mapping_items(value: Mapping[str, Any], *, depth: int = 0):
    if depth > 6:
        return
    for key, item in value.items():
        if isinstance(item, Mapping):
            yield from _walk_mapping_items(item, depth=depth + 1)
        elif isinstance(item, list):
            for nested in item[:100]:
                if isinstance(nested, Mapping):
                    yield from _walk_mapping_items(nested, depth=depth + 1)
        else:
            yield str(key), item


def _bounded_runtime_record(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "evidence_id",
        "kind",
        "dataset",
        "source",
        "variant_id",
        "gene_id",
        "study_id",
        "pmid",
        "tissue",
        "supports",
    ):
        value = row.get(key)
        if value is not None:
            result[key] = _json_value(value, path=f"record.{key}")
    return result


def _clean_heading(value: str) -> str:
    text = re.sub(r"[*_`]", "", str(value)).strip()
    text = re.sub(r"^#{1,6}\s+", "", text).strip()
    text = re.sub(r"^\d+[.)]\s*", "", text).strip()
    return text[:300] or "Evidence"


def _grounding_text(value: str) -> str:
    return " ".join(str(value).casefold().replace("-", " ").split())


def _term_in_text(term: str, text: str) -> bool:
    normalized = _grounding_text(term)
    pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
    return bool(normalized and re.search(pattern, text))


def _public_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    return {}


def _fallback_safety_review(facts: SafetyFacts) -> SafetyReview:
    explanations = tuple((item.check_id, item.fact) for item in facts.checks)
    limitations = tuple(item.fact for item in facts.checks if item.nonpassing)
    if not limitations:
        limitations = (
            "Passing these checks supports an evidence-grounded response but does not prove "
            "causality or eliminate uncertainty in the reported associations.",
        )
    explanation = {
        SafetyStatus.PASS: "The requested evidence review completed without a triggered warning. Method notes and unassessed questions remain disclosed.",
        SafetyStatus.PASS_WITH_WARNINGS: (
            "All hard gates passed, but one or more limitations require explicit caution."
        ),
        SafetyStatus.STOPPED: (
            "At least one required hard gate failed, so substantive inference was stopped."
        ),
    }[facts.status]
    return SafetyReview(
        status=facts.status,
        explanation=explanation,
        check_explanations=explanations,
        limitations=limitations,
        facts=facts,
    )


def _fallback_final_result(
    context: HybridFinalContext,
    *,
    attempts: int,
    errors: tuple[str, ...],
    usage: Mapping[str, int] | None = None,
) -> HybridFinalResult:
    if not context.facts and not context.source_observations:
        return _response_validation_stopped_result(context, attempts=attempts, errors=errors, usage=usage)
    if attempts:
        return _evidence_only_final_result(context, attempts=attempts, errors=errors, usage=usage)
    draft = _fallback_final_draft(context)
    # Stage-four JSON/wording validation is an implementation concern, not a
    # scientific safety finding.  When the stage-three gate has passed, the
    # deterministic renderer can still assemble a fully grounded answer from
    # prevalidated facts and references.  Preserve the authoritative safety
    # status and keep formatter diagnostics in ``validation_errors`` only.
    status = context.safety_review.status
    return HybridFinalResult(
        status=status,
        markdown=render_hybrid_final(context, draft, status=status),
        draft=draft,
        source="deterministic_fallback",
        attempts=attempts,
        validation_errors=errors,
        usage=dict(usage or {}),
    )


def _evidence_only_final_result(
    context: HybridFinalContext,
    *,
    attempts: int,
    errors: tuple[str, ...],
    usage: Mapping[str, int] | None = None,
) -> HybridFinalResult:
    """Recover verified findings when only the explanatory report failed."""

    if context.safety_review.status is SafetyStatus.STOPPED:
        draft = _stopped_final_draft(context)
        return HybridFinalResult(
            status=SafetyStatus.STOPPED,
            markdown=render_hybrid_final(context, draft, status=SafetyStatus.STOPPED),
            draft=draft, source="deterministic_fallback", attempts=attempts,
            validation_errors=errors, usage=dict(usage or {}),
        )
    if not context.facts and not context.source_observations:
        return _response_validation_stopped_result(context, attempts=attempts, errors=errors, usage=usage)
    draft = _fallback_final_draft(context)
    recovery_warning = _writer_warning_messages("evidence_only_recovery", errors)[0]
    draft = replace(draft, limitations_uncertainty_markdown=(
        draft.limitations_uncertainty_markdown + "\n\n" + recovery_warning
    ))
    # The writer did not deliver an interpretation (usually a technical
    # failure); this execution limit is disclosed, unlike Gate 2 edits.
    return HybridFinalResult(
        status=SafetyStatus.PASS_WITH_WARNINGS,
        markdown=render_hybrid_final(context, draft, status=SafetyStatus.PASS_WITH_WARNINGS, response_source="evidence_only_recovery", validation_errors=errors),
        draft=draft, source="evidence_only_recovery", attempts=attempts,
        validation_errors=errors, usage=dict(usage or {}),
    )


def _response_validation_stopped_result(
    context: HybridFinalContext,
    *,
    attempts: int,
    errors: tuple[str, ...],
    usage: Mapping[str, int] | None = None,
) -> HybridFinalResult:
    """Return one paragraph after two independently invalid response drafts."""

    draft = _stopped_final_draft(context)
    markdown = " ".join(
        (
            _source_gwas_status_sentence(context),
            "GWASMind could not prepare a verifiable report because no usable source "
            "observations or retrieved facts are available for this selection.",
            "Check the selected variants and uploaded dataset, then retry the analysis.",
        )
    )
    return HybridFinalResult(
        status=SafetyStatus.STOPPED,
        markdown=re.sub(r"\s+", " ", markdown).strip() + "\n",
        draft=draft,
        source="response_validation_stopped",
        attempts=attempts,
        validation_errors=errors,
        usage=dict(usage or {}),
    )


def _response_validation_failure_reason(errors: Sequence[str]) -> str:
    """Explain the actual failed check without echoing generated text or raw IDs."""

    error = errors[-1].casefold() if errors else ""
    if "source gwas phenotype" in error:
        return "The report mislabeled the source GWAS phenotype."
    if "selected-variant significance" in error or "significant signal" in error:
        return "The report did not correctly state the selected variants' source-GWAS significance."
    if "unsupported numbers" in error:
        return "The report contained a numerical value not supported by the retrieved evidence."
    if "unsupported ensembl" in error or "unsupported gene-like" in error:
        return (
            "The report named a gene or scientific identifier "
            "not present in the supplied evidence."
        )
    if "reference" in error or "citation" in error or "fact id" in error:
        return (
            "The report's evidence or citation links "
            "could not be matched to the retrieved records."
        )
    if "json" in error or "fields do not match" in error or "must be" in error:
        return "The model returned a report structure that could not be parsed or assembled."
    return "The model's report could not be reconciled with the validated evidence."


def _fallback_final_draft(context: HybridFinalContext) -> HybridFinalDraft:
    selections: list[FinalEvidenceSelection] = []
    for section in context.sections:
        facts = tuple(item for item in context.facts if item.section_id == section.section_id)
        if not facts:
            continue
        selections.append(
            FinalEvidenceSelection(
                section_id=section.section_id,
                fact_ids=tuple(item.fact_id for item in facts),
                evidence_ids=tuple(
                    evidence_id for item in facts for evidence_id in item.evidence_ids
                ),
            )
        )
    # No deterministic synthesis may substitute for a missing model Summary.
    # Only the selected facts own references in this evidence-only view.
    used_references = {reference_id for fact in context.facts for reference_id in fact.reference_ids}
    reference_ids = tuple(item.reference_id for item in context.references if item.reference_id in used_references)
    limitations = [check.fact for check in context.safety_review.facts.checks if check.nonpassing]
    if not limitations:
        limitations.append(
            "Candidate-gene prioritization is hypothesis generation and does not "
            "establish causality; independent validation remains required."
            if context.intent.strip().casefold() == "prioritize_genes"
            else "The displayed evidence supports interpretation but does not by itself "
            "establish causality; independent validation remains required."
        )
    return HybridFinalDraft(
        summary_markdown="A completed model Summary is unavailable. The available evidence is retained below.",
        evidence_sections=tuple(selections),
        limitations_uncertainty_markdown="\n".join(f"- {item}" for item in limitations),
        limitation_check_ids=context.safety_review.facts.nonpassing_check_ids,
        reference_ids=reference_ids,
    )


def _stopped_final_draft(context: HybridFinalContext) -> HybridFinalDraft:
    failures = tuple(
        item
        for item in context.safety_review.facts.checks
        if item.severity is SafetySeverity.HARD_GATE and item.outcome is SafetyOutcome.FAIL
    )
    limitations = context.safety_review.limitations or tuple(item.fact for item in failures)
    return HybridFinalDraft(
        summary_markdown=(
            _source_gwas_status_sentence(context)
            + " GWASMind stopped before substantive response generation because one or "
            "more required validation checks failed. No candidate-gene ranking or "
            "unsupported biological narrative was generated."
        ),
        evidence_sections=(),
        limitations_uncertainty_markdown="\n".join(f"- {item}" for item in limitations),
        limitation_check_ids=context.safety_review.facts.nonpassing_check_ids,
        reference_ids=(),
    )


def _source_gwas_status_sentence(context: HybridFinalContext) -> str:
    """Render the current-dataset observation before any stopped inference."""

    summary = summarize_source_gwas_observations(
        context.source_observations,
        phenotype=context.source_trait or "the loaded phenotype",
    )
    phenotype = summary.phenotype or "the loaded phenotype"
    if summary.variants_with_p_value == 0:
        return (
            f"In the uploaded **{phenotype}** GWAS, significance could not be verified "
            f"for the {summary.selected_variants} selected variant(s) because no valid "
            "P values were available."
        )
    range_text = (
        f"; selected P values ranged from {summary.minimum_p_value:.3e} to "
        f"{summary.maximum_p_value:.3e}"
        if summary.minimum_p_value is not None and summary.maximum_p_value is not None
        else ""
    )
    return (
        f"In the uploaded **{phenotype}** GWAS, "
        f"{summary.genome_wide_significant_variants} of {summary.selected_variants} "
        "selected variant(s) met the genome-wide significance threshold "
        f"(*P* <= {summary.significance_threshold:.0e}){range_text}."
    )


def _integrity_check(
    check_id: str,
    label: str,
    value: bool | None,
    passed_fact: str,
    failed_fact: str,
) -> SafetyCheck:
    return configured_safety_check(
        check_id=check_id,
        label=label,
        severity=SafetySeverity.HARD_GATE,
        outcome=SafetyOutcome.PASS if value is True else SafetyOutcome.FAIL,
        fact=passed_fact if value is True else failed_fact,
    )


def _direct_evidence(
    records: Sequence[EvidenceRecord | Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    evidence_ids: list[str] = []
    kinds: list[str] = []
    datasets: list[str] = []
    for record in records:
        evidence_id = _record_text(record, "evidence_id")
        gene_id = _record_text(record, "gene_id")
        kind = _record_kind(record)
        dataset = _record_text(record, "dataset")
        if not evidence_id or not gene_id:
            continue
        payload = record.payload if isinstance(record, EvidenceRecord) else record.get("payload", {})
        scope = evidence_candidate_scope(payload) if isinstance(payload, Mapping) else {}
        if scope and not scope.get("direct_locus_support"):
            continue
        if not _ranking_record_scope_allowed(record):
            continue
        direct_kind: str | None = None
        if kind in _DIRECT_KINDS:
            direct_kind = kind
        elif kind == EvidenceKind.VARIANT_ANNOTATION.value and _high_impact_coding(record):
            direct_kind = "high_impact_coding_or_splice"
        if direct_kind is None:
            continue
        evidence_ids.append(evidence_id)
        kinds.append(direct_kind)
        if dataset:
            datasets.append(dataset)
    return (
        tuple(dict.fromkeys(evidence_ids)),
        tuple(dict.fromkeys(kinds)),
        tuple(dict.fromkeys(datasets)),
    )


def has_direct_locus_gene_evidence(
    records: Sequence[EvidenceRecord | Mapping[str, Any]],
) -> bool:
    """Return whether any permitted direct locus-to-gene channel is represented."""

    return bool(_direct_evidence(records)[0])


def _ranking_record_scope_allowed(record: EvidenceRecord | Mapping[str, Any]) -> bool:
    from .projection import _frozen_molqtl_coloc_scope_allowed

    if _frozen_molqtl_coloc_scope_allowed(record) is False:
        return False
    payload = record.payload if isinstance(record, EvidenceRecord) else record.get("payload", {})
    if not isinstance(payload, Mapping):
        return False
    if evidence_candidate_scope(payload).get("context_only"):
        return False
    context = payload.get("context")
    return not (isinstance(context, Mapping) and context.get("rank_scope") in {
        "context_only", "other_trait", "unresolved"})


def _high_impact_coding(record: EvidenceRecord | Mapping[str, Any]) -> bool:
    payload = record.payload if isinstance(record, EvidenceRecord) else record.get("payload", {})
    if not isinstance(payload, Mapping):
        return False
    return _has_high_impact_coding_payload(payload)


def _literature_count(records: Sequence[EvidenceRecord | Mapping[str, Any]]) -> int:
    return publication_count([_review_record_mapping(record) for record in records])


def _review_record_mapping(record: EvidenceRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(record, EvidenceRecord):
        return record
    return {"kind": record.kind.value, "evidence_id": record.evidence_id,
            "source": record.source, "dataset": record.dataset, "data_release": record.data_release,
            "credible_set_id": record.credible_set_id, "parent_evidence_ids": record.parent_evidence_ids,
            "gene_id": record.gene_id, "variant_id": record.variant_id, "study_id": record.study_id,
            "pmid": record.pmid, "tissue": record.tissue, "supports": record.supports,
            "payload": record.payload}


def _record_kind(record: EvidenceRecord | Mapping[str, Any]) -> str:
    value = record.kind if isinstance(record, EvidenceRecord) else record.get("kind")
    return value.value if isinstance(value, EvidenceKind) else str(value or "").strip().casefold()


def _record_text(record: EvidenceRecord | Mapping[str, Any], key: str) -> str:
    value = getattr(record, key, None) if isinstance(record, EvidenceRecord) else record.get(key)
    if key in {"gene_id", "evidence_id"} and not isinstance(value, str):
        return ""
    return str(value).strip() if value is not None else ""


def _complete_text(
    completer: Callable[..., Any],
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> str:
    content, _usage = _complete_text_with_usage(
        completer,
        messages,
        max_tokens=max_tokens,
    )
    return content


def _complete_text_with_usage(
    completer: Callable[..., Any],
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> tuple[str, dict[str, int]]:
    with report_validation_callback():
        result = completer(messages, max_tokens=max_tokens)
    if isinstance(result, str):
        content = result
    elif isinstance(result, Mapping):
        content = result.get("content")
    else:
        content = getattr(result, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise StructuredOutputError("model returned an empty response")
    usage_value = (
        result.get("usage")
        if isinstance(result, Mapping)
        else getattr(result, "usage", {})
    )
    usage: dict[str, int] = {}
    if isinstance(usage_value, Mapping):
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
        ):
            value = usage_value.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10_000_000:
                usage[key] = value
    latency = (
        result.get("latency_ms")
        if isinstance(result, Mapping)
        else getattr(result, "latency_ms", None)
    )
    if isinstance(latency, int) and not isinstance(latency, bool) and 0 <= latency <= 3_600_000:
        usage["latency_ms"] = latency
    finish_reason = result.get("finish_reason") if isinstance(result, Mapping) else getattr(result, "finish_reason", None)
    if finish_reason not in (None, "stop"):
        from .model_generation import GenerationFailure
        reason = finish_reason if finish_reason in {"length", "content_filter", "tool_calls", "function_call"} else "invalid"
        failure = GenerationFailure("The report JSON generation did not complete.",
                                    category="output_length" if reason == "length" else "provider_finish",
                                    finish_reason=reason, retryable=False)
        failure.raw_content = content
        failure.usage = usage
        raise failure
    return content, usage


def _merge_usage(*values: Mapping[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for value in values:
        for key, count in value.items():
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                merged[key] = merged.get(key, 0) + count
    return merged


def _strict_json_object(content: str, *, label: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise StructuredOutputError(f"{label} is empty")
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        def unique_object(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate JSON field")
                result[key] = item
            return result

        def reject_constant(value):
            raise ValueError("non-finite JSON constant")

        value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise StructuredOutputError(f"{label} must be one JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
    details = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if unknown:
        details.append("unknown=" + ",".join(unknown))
    raise StructuredOutputError(f"{label} fields do not match the contract: " + "; ".join(details))


def _parse_enum(value: Any, enum_type: type[StrEnum], *, label: str):
    if not isinstance(value, str):
        raise StructuredOutputError(f"{label} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise StructuredOutputError(f"unsupported {label}: {value}") from exc


def _bounded_text(value: Any, *, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredOutputError(f"{label} must be a non-empty string")
    text = value.strip()
    if len(text) > limit:
        raise StructuredOutputError(f"{label} exceeds {limit} characters")
    if "\x00" in text:
        raise StructuredOutputError(f"{label} contains a null character")
    return text


_RESERVED_REPORT_HEADINGS = re.compile(
    r"(?:\*\*(?:Validation status|Report quality|Evidence|References|"
    r"Limitations\s*&\s*Uncertainty)\*\*|"
    r"^#{1,3}\s+(?:Evidence|References|Limitations\s*&\s*Uncertainty)\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def _bounded_markdown(value: Any, *, label: str, limit: int) -> str:
    text = _bounded_text(value, label=label, limit=limit)
    text = clean_report_control_characters(text)
    if not text.strip():
        raise StructuredOutputError(f"{label} is empty after removing display controls")
    if _RESERVED_REPORT_HEADINGS.search(text):
        raise StructuredOutputError(f"{label} contains a reserved report heading")
    if re.search(r"<\s*(?:script|iframe|object|embed)\b", text, re.IGNORECASE):
        raise StructuredOutputError(f"{label} contains unsafe HTML")
    if re.search(r"\[\d+(?:\s*,\s*\d+)*\]", text):
        raise StructuredOutputError(
            f"{label} must not generate citation numbers; use the reference_ids field"
        )
    return text


def clean_report_control_characters(text: str) -> str:
    """Remove terminal/display controls, preserving ordinary Markdown whitespace.

    Validate scientific content only after this operation: a hidden or visually
    reversed statistic must never acquire authority from its formatting.
    """

    text = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]", "", text)


def validate_report_fragment(
    text: str, context: HybridFinalContext, *, label: str = "summary_markdown",
) -> None:
    validators = {"report_fragment_consistency": _validate_report_fragment_contract}
    for name in load_gate_rules().validator_ids("report_fragment"):
        validators[name](text, context, label=label)


def _validate_report_fragment_contract(
    text: str, context: HybridFinalContext, *, label: str = "summary_markdown",
) -> None:
    """Validate a report fragment without imposing whole-report opening rules.

    Local recovery may retain only fragments passing this same scientific
    contract. Full parsing still checks selected/required facts and the source
    opening. Callers retaining text should apply clean_report_control_characters.
    """

    text = _bounded_markdown(text, label=label, limit=parameter("gates.report_markdown_max_chars"))
    known = {fact.fact_id for fact in context.facts}
    cited = set(_SUMMARY_FACT_CITATION_TOKEN.findall(text))
    if cited.difference(known):
        raise StructuredOutputError(f"{label} uses unknown fact IDs: " + ", ".join(sorted(cited - known)))
    residual = _SUMMARY_FACT_CITATION_TOKEN.sub("", text)
    if re.search(r"\{\{\s*cite(?:_fact)?\s*:", residual, re.IGNORECASE):
        raise StructuredOutputError(f"{label} contains a malformed or unsupported citation token")
    # Final model prose uses the supplied fact/reference namespace. Underlying
    # native record IDs, including IDs appearing only in lineage metadata,
    # cannot substitute for a visible fact citation or create a reference.
    raw_ids = {identifier for fact in context.facts for identifier in fact.evidence_ids}
    raw_ids.update(str(row["evidence_id"]) for row in context.result.get("evidence", ())
                   if isinstance(row, Mapping) and row.get("evidence_id"))
    reference_ids = {reference.reference_id for reference in context.references}
    allowed_ids = known | reference_ids | {str(index) for index in range(1, len(context.references) + 1)}
    inline_ids = inline_evidence_citations(residual, raw_ids | allowed_ids)
    invalid_inline = inline_ids.difference(allowed_ids)
    if invalid_inline:
        raise StructuredOutputError(
            f"{label} uses evidence IDs outside the final fact/reference citation namespace: "
            + ", ".join(sorted(invalid_inline))
            + "; cite a supplied fact instead"
        )
    if label.startswith("limitations"):
        _validate_task_specific_limitations(text, context=context)
    _validate_free_scientific_markdown(
        _SUMMARY_FACT_CITATION_TOKEN.sub("", _canonicalize_bound_tss_distances(text, context=context)),
        context=context, label=label,
    )
    _validate_fact_bound_measurements(text, context=context, label=label)
    _validate_source_effect_direction(text, context=context, label=label)


_SCIENTIFIC_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|"
    r"(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_SCIENTIFIC_PRODUCT = re.compile(
    r"(?<![\w.+-])(?P<mantissa>[-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:×|x|\*|·)\s*"
    r"10\s*(?:\^\s*(?P<plain>\{\s*[-+]?\d+\s*\}|[-+]?\d+)|(?P<bare>[-+]\s*\d+)|"
    r"(?P<super>[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))(?![\w}]|\.\d|\s*[×x*·])",
    re.IGNORECASE,
)
_PHENOTYPE_SPELLINGS = (
    (re.compile(r"\bMDD\b", re.IGNORECASE), "major depressive disorder"),
    (
        re.compile(
            r"\b(?:t1dm?|type[-\s]*(?:1|i)[-\s]+diabetes(?:\s+mellitus)?)\b", re.IGNORECASE
        ),
        "type 1 diabetes",
    ),
    (
        re.compile(
            r"\b(?:t2dm?|type[-\s]*(?:2|ii)[-\s]+diabetes(?:\s+mellitus)?)\b", re.IGNORECASE
        ),
        "type 2 diabetes",
    ),
)
_GENERIC_UPPERCASE_TOKENS = frozenset(
    {
        "API",
        "DNA",
        "EFFECT",
        "GO",
        "GWAS",
        "GWAS-QTL",
        "GWASMIND",
        "HLA",
        "LD",
        "L2G",
        "MCP",
        "MHC",
        "NO",
        "OR",
        "PASS",
        "PMID",
        "QTL",
        "RNA",
        "SNP",
        "SNPS",
        "STOPPED",
    }
)


def _validate_free_scientific_markdown(
    text: str,
    *,
    context: HybridFinalContext,
    label: str,
) -> None:
    """Reject readily machine-checkable inventions in otherwise free model prose."""
    from .report_facts import computed_report_facts, report_fact_issues
    factual_issues = report_fact_issues(text, computed_report_facts(context.source_observations, context.result))
    if factual_issues:
        raise StructuredOutputError(f"{label}: " + "; ".join(factual_issues))

    ld_scope = _final_ld_report_scope(context)
    ld_issues = ld_attribution_issues(text, scope=ld_scope, evidence=_runtime_evidence_rows(context.result),
                                    check_ld_relations=False)
    from .report_ld_assertions import report_ld_numeric_bindings
    ld_issues = (*ld_issues, *("LD r² assertion conflicts with its frozen evidence" for finding in
        report_ld_numeric_bindings(text, context) if finding["verification_status"] == "contradicted"))
    if ld_issues:
        raise StructuredOutputError(f"{label}: " + "; ".join(ld_issues))

    novelty_screen = context.result.get("novelty_screen")
    if isinstance(novelty_screen, Mapping):
        from .novelty import validate_novelty_claims

        try:
            validate_novelty_claims(text, novelty_screen)
        except ValueError as exc:
            raise StructuredOutputError(f"{label}: {exc}") from exc

    authority = "\n".join(
        (
            context.deterministic_answer,
            context.source_trait,
            _canonical_json(context.source_observations),
            _canonical_json(context.biological_context),
            _canonical_json(_final_analysis_scope(context)),
            _canonical_json(context.safety_review.to_dict()),
            _canonical_json([
                {key: row[key] for key in ("gene_id", "rank") if key in row}
                for row in _final_ranking_context(context.result)
            ]),
            _canonical_json(_validated_ranked_gene_symbols(context)),
            *(item.markdown for item in context.facts),
            *(_canonical_json(_final_fact_model_data(item)) for item in context.facts),
        )
    )
    validation_text = text
    if context.source_observations and _scientific_numbers(text).difference(_scientific_numbers(authority)):
        # Canonicalize only each verified source-measurement occurrence for the
        # generic numeric check. Other occurrences, including external claims,
        # do not inherit this value or a global rounding tolerance.
        replacements: list[tuple[int, int, str]] = []
        _validate_fact_bound_measurements(text, context=context, label=label,
                                         _source_number_replacements=replacements, _source_only=True)
        if replacements:
            validation_text = _measurement_prose(text)
            for start, end, exact_value in reversed(replacements):
                validation_text = validation_text[:start] + exact_value + validation_text[end:]
    validation_text = _canonicalize_bound_source_coordinates(validation_text, context=context)
    validation_text = _canonicalize_bound_reference_scope(validation_text, context=context)
    _validate_text_against_authority(
        _normalize_phenotype_spellings(validation_text),
        authority=_normalize_phenotype_spellings(authority),
        label=label,
        free_narrative=True,
    )
    _validate_colocalization_pairs(text, context=context, label=label)
    ranked_symbols = _validated_ranked_gene_symbols(context)
    for match in re.finditer(r"\b([A-Z][A-Z0-9-]{1,30})\s*\(\s*(ENSG\d{6,})\b", re.sub(r"[*`]", "", text)):
        if match[1] in ranked_symbols and ranked_symbols[match[1]] != match[2]:
            raise StructuredOutputError(f"{label} pairs a ranked gene symbol with an incorrect Ensembl gene ID")
    _validate_explicit_gene_ranks(text, context=context, label=label)
    _validate_member_pip_ownership(text, context=context, label=label)
    # Gate 2 compares objective facts and provenance. Causal wording is the
    # model's scientific narrative, not a reason to request another synthesis.
    for reference in context.references:
        if reference.reference_id in text:
            raise StructuredOutputError(
                f"{label} must not expose raw reference IDs; use the reference_ids field"
            )


def _canonicalize_bound_source_coordinates(text: str, *, context: HybridFinalContext) -> str:
    """Verify a selected-source span before normalizing its displayed units.

    Endpoints must be the source positions on the stated chromosome. Rounding
    applies only to this explicitly approximate range, at its written precision;
    the converted numbers never become authority for another claim.
    """
    if not context.source_observations or not re.search(r"\bchromosome\b", text, re.I):
        return text
    plain = _measurement_prose(text)
    number = r"\d[\d,]*(?:\.\d+)?"
    ranges = re.compile(
        r"\bchromosome\s+(?P<chromosome>\d{1,2}|X|Y|MT)\s*(?:region\s*)?"
        r"(?:\(\s*|[:=]\s*|(?:from|at|between)\s+)?"
        r"(?P<approx>about\s+|approximately\s+|approx\.?\s+|[≈~]\s*)?"
        r"(?P<start>" + number + r")\s*(?:[-–—]|to\b|and\b)\s*"
        r"(?P<end>" + number + r")\s*(?P<unit>bp|base\s+pairs?|kb|kilobases?|Mb|megabases?)\b", re.I,
    )
    source_subject = r"\b(?:source[- ]selected|selected|uploaded|source)\s+(?:variants?|SNPs?)\b"
    external_subject = r"\b(?:external|retrieved|studies|study|publications?|proxies|proxy|credible[- ]set)\b"
    replacements = []
    for match in ranges.finditer(plain):
        sentences = re.split(r"(?<=[.!?])\s+|[;\n]", plain[:match.start()])
        prefix = sentences[-1]
        bound = bool(re.search(source_subject, prefix, re.I))
        if not bound and re.fullmatch(r"\s*They\s+map\s+to\s+(?:a\s+single\s+|the\s+)?", prefix, re.I) and len(sentences) > 1:
            prefix = sentences[-2] + " " + prefix
            bound = bool(re.search(source_subject, sentences[-2], re.I))
        if not bound or re.search(external_subject, prefix, re.I):
            continue
        positions = []
        for row in context.source_observations:
            chromosome = str(row.get("chromosome", "")).removeprefix("chr").upper()
            position = row.get("position")
            if chromosome != match["chromosome"].upper() or type(position) is not int or position <= 0:
                raise StructuredOutputError("A selected-source chromosome span needs consistent source positions on the stated chromosome.")
            identifier = str(row.get("variant_id", "")).removeprefix("chr").replace(":", "_")
            if identifier and not identifier.upper().startswith(f"{chromosome}_{position}_"):
                raise StructuredOutputError("A selected-source chromosome span contradicts its source variant coordinates.")
            positions.append(position)
        unit = match["unit"].casefold()
        factor = Decimal(1000000 if unit.startswith("m") else 1000 if unit.startswith("k") else 1)
        for endpoint, actual in (("start", min(positions)), ("end", max(positions))):
            reported = Decimal(match[endpoint].replace(",", ""))
            with localcontext(Context(prec=64)):
                exact = reported * factor == actual
                approximate = bool(match["approx"]) and reported > 0 and (
                    (Decimal(actual) / factor).quantize(Decimal(1).scaleb(reported.as_tuple().exponent)) == reported)
            if not exact and not approximate:
                raise StructuredOutputError("A selected-source chromosome span contradicts its source endpoints, units or stated precision.")
            replacements.append((*match.span(endpoint), str(actual)))
    for start, end, value in reversed(replacements):
        plain = plain[:start] + value + plain[end:]
    return plain if replacements else text


def _canonicalize_bound_tss_distances(text: str, *, context: HybridFinalContext) -> str:
    """Bind a displayed E2G interval's TSS-distance conversion to its own fact.

    Only this occurrence is normalised for the generic numeric check. No value
    is added to global authority and an LD window cannot supply a TSS distance.
    """
    if not re.search(r"\bTSS\b", text, re.I):
        return text
    plain = _measurement_prose(text)
    quantity = re.compile(r"(?<![\w.])(?P<approx>about\s+|approximately\s+|approx\.?\s+|[≈~]\s*)?"
                          r"(?P<number>[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
                          r"\s*-?\s*(?P<unit>bp|base\s+pairs?|kb|kilobases?|Mb|megabases?)\b", re.I)
    facts = {}
    gene_aliases = {}
    for fact in context.facts:
        subject = str(fact.model_data.get('subject') or '')
        named_gene = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)\s*\((ENSG\d+)\)\s*", subject)
        if named_gene:
            gene_aliases[named_gene[1].casefold()] = named_gene[2]
            gene_aliases[named_gene[2].casefold()] = named_gene[2]
        detail = _measurement_prose(fact.markdown)
        distance = re.search(r"\bdistance\s+to\s+(?P<gene>[A-Za-z0-9_.-]+)\s+TSS\s*=\s*"
                             r"(?P<bp>\d[\d,]*)\s+bp\b", detail, re.I)
        variant = re.search(r"^\s*-?\s*(?:Source-selected variant|LD[- ]proxy(?: variant)?)\s+("
                            + _REPORT_VARIANT.pattern + r")\s+overlaps\b", detail, re.I)
        if not distance or not variant or 'regulatory interval' not in detail.casefold():
            continue
        identifier = re.search(r"\bENSG\d+\b", subject)
        gene = identifier.group() if identifier else distance['gene']
        if not re.search(r"(?<!\w)" + re.escape(distance['gene']) + r"(?!\w)", subject, re.I):
            continue
        gene_aliases[distance['gene'].casefold()] = gene
        gene_aliases[gene.casefold()] = gene
        facts[fact.fact_id] = (gene, variant[1].casefold().removeprefix('chr').replace(':', '_'),
                              Decimal(distance['bp'].replace(',', '')))
    replacements = []
    for match in quantity.finditer(plain):
        prefix = re.split(r"(?<=[.!?])\s+|[;\n]|\bwhereas\b", plain[:match.start()], flags=re.I)[-1]
        prior_citations = list(_SUMMARY_FACT_CITATION_TOKEN.finditer(prefix))
        if prior_citations:
            after_citation = prefix[prior_citations[-1].end():]
            # A completed cited QTL claim followed by a new regulatory-interval
            # subject does not lend its earlier SNP ID to the interval distance.
            if re.match(r"\s*,?\s*(?:and|plus)\s+(?:(?:a|an|the|reported)\s+)*"
                        r"regulatory\s+intervals?\b", after_citation, re.I):
                prefix = after_citation
        suffix = re.split(r"(?<=[.!?])\s+|[;\n]", plain[match.end():])[0]
        before_tss = re.search(r"\bTSS(?:\s+distance)?(?:\s+(?:is|of))?\s*[:=]?\s*$", prefix, re.I)
        after_tss = re.match(r"\s+(?:away\s+)?(?:from|to)\s+(?:(?:its|the)\s+)?"
                             r"(?:(?P<tss_gene>[A-Za-z0-9_.-]+)(?:'s)?\s+)?TSS\b", suffix, re.I)
        if not before_tss and not after_tss:
            continue
        anchors = list(_SUMMARY_FACT_CITATION_TOKEN.finditer(suffix))
        if not anchors:
            raise StructuredOutputError('A reported TSS distance needs its own cited regulatory fact.')
        cited = [anchors[0][1]]
        end = anchors[0].end()
        for anchor in anchors[1:]:
            if suffix[end:anchor.start()].strip():
                break
            cited.append(anchor[1])
            end = anchor.end()
        bindings = [facts.get(identifier) for identifier in cited]
        if any(binding is None for binding in bindings):
            raise StructuredOutputError('A reported TSS distance cites a fact without that interval-distance measurement.')
        owners = {binding[0] for binding in bindings}
        values = {binding[2] for binding in bindings}
        if len(owners) != 1 or len(values) != 1:
            raise StructuredOutputError('A reported TSS distance must identify one gene and one unambiguous cited distance.')
        if after_tss and after_tss['tss_gene'] and gene_aliases.get(after_tss['tss_gene'].casefold()) not in owners:
            raise StructuredOutputError('A reported TSS distance names a different gene from its cited fact.')
        local = prefix + suffix[:anchors[0].start()]
        mentioned = [(m.start(), gene) for alias, gene in gene_aliases.items()
                     for m in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", local, re.I)]
        if {gene for _, gene in mentioned} != owners:
            raise StructuredOutputError('A reported TSS distance does not identify the gene owning its cited fact.')
        variants = {m.group().casefold().removeprefix('chr').replace(':', '_') for m in _REPORT_VARIANT.finditer(local)}
        if not variants.issubset({binding[1] for binding in bindings}):
            raise StructuredOutputError('A reported TSS distance refers to a different variant from its cited fact.')
        # E2G distanceToTss belongs to the reported regulatory interval, not
        # proof that a SNP itself lies at that distance or overlaps the TSS.
        if not re.search(r"\bintervals?\b", local, re.I):
            raise StructuredOutputError('Use the reported regulatory interval as the owner of its TSS distance.')
        reported = Decimal(match['number'].replace(',', ''))
        unit = match['unit'].casefold()
        factor = Decimal(1000000 if unit.startswith('m') else 1000 if unit.startswith('k') else 1)
        actual = next(iter(values))
        with localcontext(Context(prec=64)):
            exact = reported * factor == actual
            approximate = bool(match['approx']) and reported > 0 and actual > 0 and (
                (actual / factor).quantize(Decimal(1).scaleb(reported.as_tuple().exponent)) == reported)
        if not exact and not approximate:
            raise StructuredOutputError('A reported TSS distance contradicts its cited interval measurement or stated precision.')
        replacements.append((match.start(), match.end(), f'{actual} bp'))
    for start, end, value in reversed(replacements):
        plain = plain[:start] + value + plain[end:]
    return plain if replacements else text


def _canonicalize_bound_reference_scope(text: str, *, context: HybridFinalContext) -> str:
    """Normalize only bound LD names/window occurrences for the numeric check.

    A reference name is not a sample size. Exact window-unit conversions are
    local to an explicit LD search statement; neither exemption adds a number
    to the authority or changes the text used by fact/measurement validators.
    """
    panels: set[str] = set()
    windows: set[int] = set()
    scope = _final_ld_report_scope(context)
    if scope.provided and scope.valid:
        if isinstance(scope.reference_panel, str):
            panels.add(scope.reference_panel)
        if type(scope.window_bp) is int and scope.window_bp > 0:
            windows.add(scope.window_bp)
    screen = context.result.get("novelty_screen")
    if isinstance(screen, Mapping):
        # The deterministic novelty fact must also expose the scope to the
        # writer. An unused result field is not model-visible authority.
        statement = "\n".join(fact.markdown for fact in context.facts
                              if fact.fact_id == "novelty_ld_result" and fact.required)
        panel = screen.get("reference_panel")
        if isinstance(panel, str) and panel in statement:
            panels.add(panel)
        window = screen.get("window_bp")
        if type(window) is int and window > 0 and re.search(
            rf"(?<![\d.]){window}\s+bp\b", statement.replace(",", ""), re.I
        ):
            windows.add(window)
    phases = {match[1] for panel in panels
              if (match := re.fullmatch(r"1000GENOMES:phase_(\d+)", panel, re.I))}
    if phases:
        name = re.compile(
            r"(?<![\w,])1000[ \t]*Genomes(?P<project>[ \t]+Project)?"
            r"(?:(?:[ \t]+phase[ \t]+|:phase_)(?P<phase>\d+))?(?![\w-]|\.\d)", re.I
        )

        def reference_name(match: re.Match[str]) -> str:
            phase = match["phase"]
            if phase is not None and phase not in phases:
                raise StructuredOutputError("Reference-panel phase contradicts the recorded reference panel.")
            if phase is None and re.match(r"(?:[ \t]+phase\b|:phase_)", text[match.end():], re.I):
                raise StructuredOutputError("Reference-panel phase cannot be matched to the recorded reference panel.")
            if phase is None and not match["project"] and not (
                re.search(r"\b(?:reference|panel)(?:\s+(?:is|was|from|of|the|used|uses|named))*\s+$",
                          text[:match.start()], re.I)
                or re.match(r"\s+(?:reference|panel|project)\b", text[match.end():], re.I)
            ):
                # The same words can mean a real count of sequenced genomes.
                return match[0]
            return "1000GENOMES" + (f":phase_{phase}" if phase is not None else "")

        text = name.sub(reference_name, text)
    if not windows:
        return text
    quantity = re.compile(
        r"(?<![\w.,])(?P<direction>(?:\u00b1|\+/-)[ \t]*)?"
        r"(?P<number>\d[\d,]{0,23}(?:\.\d{1,12})?|\.\d{1,12})"
        r"[ \t]*(?P<unit>bp|base\s+pairs?|kb|kilobases?|Mb|megabases?)(?![\w])", re.I
    )

    def window_quantity(match: re.Match[str]) -> str:
        unit = match["unit"].casefold()
        factor = 1_000_000 if unit.startswith("m") else 1000 if unit.startswith("k") else 1
        with localcontext(Context(prec=32)):
            value_bp = Decimal(match["number"].replace(",", "")) * factor
        raw_before = re.split(r"(?<=[.!?])\s+|[;\n]", text[:match.start()])[-1]
        # Markdown emphasis around a heading must not hide its full stop and
        # make a later gene-search distance inherit the heading's LD word.
        before = re.split(r"(?<=[.!?])\s+", re.sub(r"[*`]", "", raw_before))[-1]
        after = re.split(r"(?<=[.!?])\s+|[;\n]", text[match.end():])[0]
        ld_before = bool(re.search(r"\bLD\b", before, re.I))
        # A recorded LDlink search is an explicit LD owner even when the
        # writer names the provider after its quantity. Bind both the search
        # and the selected/source SNPs in this same clause; an LD heading or
        # another gene's interval cannot supply that ownership.
        ldlink_search = bool(
            scope.provided and scope.valid and scope.reference_provider == "ldlink"
            and re.search(r"\bsearch\s+(?:interval|window|radius)\b", before, re.I)
            and re.search(r"\b(?:each|every)\s+(?:source|selected)\s+(?:SNPs?|variants?)\b", after, re.I)
            and re.search(r"\b(?:using|through|via)\b[^;.!?]*\bLDlink\b", after, re.I)
        )
        ld_before = ld_before or ldlink_search
        # A direct sentence-level continuation can retain the immediately
        # preceding LD-expansion subject. Require an explicit search and its
        # per-side source-SNP scope; do not resolve general prose pronouns.
        if (not ld_before and scope.provided and scope.valid
                and re.match(r"\s*(?:It|This\s+expansion)\b", before, re.I)
                and re.search(r"\b(?:searching|search\s+window)\b", before, re.I)
                and re.match(r"\s+(?:(?:on|to)\s+)?(?:each|either)\s+side\s+of\s+"
                             r"(?:each|every)\s+(?:source|selected)\s+(?:SNPs?|variants?)\b", after, re.I)):
            sentences = re.split(r"(?<=[.!?])[ \t]+|\n", re.sub(r"[*`]", "", text[:match.start()]))
            if len(sentences) >= 2 and re.match(r"\s*(?:The\s+)?LD\s+expansion\b", sentences[-2], re.I):
                ld_before = True
        # Only a recorded search-distance continuation may inherit the
        # immediately preceding LD reference across a semicolon.
        previous = text[:match.start() - len(raw_before)]
        if not ld_before and previous.endswith(';') and re.match(
            r"\s*(?:the\s+)?recorded\s+(?:per[- ]side\s+)?search\s+distance\b", before, re.I
        ):
            antecedent = re.split(r"(?<=[.!?])\s+|[;\n]", previous[:-1])[-1]
            if re.search(r"\bLD\b", antecedent, re.I) and re.search(r"\b(?:reference|panel|expansion)\b", antecedent, re.I):
                ld_before = True
        # A semicolon-delimited LD configuration list keeps its enclosing
        # subject, e.g. "LD expansion (...; GRCh38; +/-250 kb)". Do not let
        # that context escape the parentheses or supply unrelated distances.
        opening = text.rfind("(", 0, match.start())
        if opening >= 0 and ")" not in text[opening:match.start()]:
            introduction = re.split(r"(?<=[.!?])\s+|[;\n]", text[:opening])[-1]
            closing = text.find(")", match.end())
            if (closing >= 0 and "\n" not in text[opening:closing]
                    and re.search(r"\bLD\b", introduction, re.I)
                    and re.search(r"\b(?:expansion|search|panel|scope|settings|proxies)\b", introduction, re.I)
                    and not re.search(r"\b(?:TSS|distances?|intervals?|size)\b", introduction, re.I)):
                ld_before = True
        # Do not borrow an LD window for a gene/TSS distance or another
        # measurement mentioned in the same sentence.
        explicit_ld_after = re.match(r"\s+LD\s+(?:search\s+)?(?:window|radius)\b", after, re.I)
        local_before = re.split(r"\bLD\b", before, flags=re.I)[-1] if ld_before else before
        if re.search(r"\b(?:gene|regulatory|enhancer|credible[- ]set)\s+(?:search\s+)?interval\b", local_before, re.I):
            return match[0]
        local_without_search_distance = re.sub(
            r"\b(?:per[- ]side\s+(?:search\s+)?distance|search\s+distance)\b", "search radius", local_before, flags=re.I)
        local_without_search_distance = re.sub(r"\bsearch\s+interval\b", "search window", local_without_search_distance, flags=re.I)
        if not (ld_before or explicit_ld_after) or re.search(
            r"\b(?:TSS|distances?|intervals?|size)\b", local_without_search_distance, re.I
        ):
            return match[0]
        window_before = re.search(
            r"\b(?:search\s+)?window(?:\s+(?:is|of|extends?|spans?|width|length|totals?))*"
            r"\s*[:=]?\s*$", before, re.I,
        )
        window_after = re.match(
            r"\s*(?:[- ]wide\s+)?(?:(?:total|full|overall|search|LD)\s+)*window\b", after, re.I,
        )
        each_side = re.match(
            r"\s+(?:(?:search\s+)?window\s+)?(?:(?:on|to)\s+)?(?:each|either)\s+side\b", after, re.I,
        )
        radius = re.search(r"\b(?:radius|half[- ]width)(?:\s+(?:is|of))?\s*[:=]?\s*$", before, re.I)
        radius = radius or re.match(r"\s+(?:radius|half[- ]width)\b", after, re.I)
        radius = radius or re.search(r"\bper[- ]side\s+(?:search\s+)?distance(?:\s+(?:is|of|was))?\s*[:=]?\s*$", before, re.I)
        within = re.search(r"\bwithin\s*$", before, re.I) and not window_after
        half_width = bool(match["direction"] or each_side or radius or within)
        total_width = bool(re.search(
            r"\b(?:total|full|overall)(?:\s+(?:LD|search|window|span|width|length|is|of))*"
            r"\s*[:=]?\s*$", before, re.I,
        ) or re.match(
            r"\s+(?:(?:in\s+)?(?:total|full|overall)\b|"
            r"(?:(?:search|LD)\s+)?window\s+(?:in\s+)?total\b)", after, re.I,
        ))
        if not (window_before or window_after or half_width or total_width):
            return match[0]
        if "," in match["number"] and not re.fullmatch(
            r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", match["number"]
        ):
            raise StructuredOutputError("LD search-window number has invalid thousands separators.")
        if half_width == total_width:
            raise StructuredOutputError(
                "LD search-window width is ambiguous: state the recorded distance on each side "
                "(or +/- radius), or explicitly label twice that distance as the total window width."
            )
        with localcontext(Context(prec=32)):
            radius_bp = value_bp / 2 if total_width else value_bp
        if radius_bp not in windows:
            raise StructuredOutputError(
                "LD search-window width contradicts the recorded per-side distance; "
                "a total window spans twice that distance."
            )
        # Compare a verified total-width conversion through its authoritative
        # radius only at this occurrence; other occurrences cannot borrow it.
        return f"{int(radius_bp)} bp"

    return quantity.sub(window_quantity, text)


def _validate_member_pip_ownership(text: str, *, context: HybridFinalContext, label: str) -> None:
    """A displayed member posterior must not be relabeled as a whole-set value."""

    member_fact = any(
        _REPORT_VARIANT.search(str(fact.model_data.get("subject", "")))
        and re.search(r"\bcredible[- ]set\s+member\b", fact.markdown, re.IGNORECASE)
        and re.search(r"\bPIP\s*(?:=|:)\s*\d", _measurement_prose(fact.markdown), re.IGNORECASE)
        for fact in context.facts
    )
    if not member_fact:
        return
    reassignment = re.compile(
        r"\bPIP(?:\s+value)?\s+(?:applies?|belongs?|refers?)\s+to\s+"
        r"(?:(?:the|this|that|a|an|entire)\s+){0,2}credible[- ]set"
        r"(?:\s+as\s+a\s+whole)?\s*[,;]?\s+(?:rather\s+than|instead\s+of)\s+"
        r"(?:to\s+)?(?:(?:this|that|the|a|an|selected|individual)\s+){0,2}variant\b",
        re.IGNORECASE,
    )
    for clause in _measurement_clauses(_measurement_prose(text)):
        for match in reassignment.finditer(clause):
            before = re.split(r"[;,]|\b(?:but|however|yet)\b", clause[:match.start()], flags=re.IGNORECASE)[-1]
            if re.search(r"\b(?:incorrect|false|misleading)\s+to\s+(?:say|claim|state)\s+(?:that\s+)?(?:the\s+)?$", before, re.IGNORECASE):
                continue
            raise StructuredOutputError(
                f"{label} changes credible-set member PIP ownership: the displayed posterior "
                "belongs to the selected variant within its credible set, not to the whole set instead of that variant."
            )


@memoized_report_derivation
def _reported_colocalization_pairs(context: HybridFinalContext) -> tuple[dict[str, Any], ...]:
    """Keep gene, phenotype and H4 together, instead of validating a bag of numbers."""

    pairs: list[dict[str, Any]] = []
    for record in _runtime_evidence_rows(context.result):
        if str(record.get("kind")) != EvidenceKind.COLOCALIZATION.value:
            continue
        payload = record.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        upstream = payload.get("upstream")
        upstream = upstream if isinstance(upstream, Mapping) else {}
        raw = upstream.get("raw")
        raw = raw if isinstance(raw, Mapping) else payload
        measurements = payload.get("measurements")
        measurements = measurements if isinstance(measurements, Mapping) else {}
        trait = next((str(raw[key]) for key in (
            "rightTraitFromSource", "traitFromSource", "colocTrait", "rightTrait", "leftTrait",
        ) if raw.get(key)), "")
        h4 = next((value.get("h4") for value in (measurements, raw, payload)
                   if value.get("h4") is not None), None)
        gene_id = str(record.get("gene_id") or "")
        if not gene_id or not trait or h4 is None or isinstance(h4, bool):
            continue
        try:
            number = Decimal(str(h4))
        except InvalidOperation:
            continue
        if not number.is_finite() or not 0 <= number <= 1:
            continue
        pairs.append({"gene_id": gene_id, "trait": trait, "h4": number})
    return tuple(pairs)


_REPORT_VARIANT = re.compile(
    r"\b(?:rs\d+|(?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+)\b", re.IGNORECASE
)
_REPORT_MEASUREMENT = re.compile(
    r"(?<!\w)(?P<metric>P(?:[- ]?value)?|Beta|β|PIP|H4|OR|odds[- ]ratio|log[- ]odds)\s*"
    r"(?:=|:|of\b|is\b|was\b|at\b|up\s+to\b|~|≈)\s*"
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?)(?![\w/:]|\.\d)",
    re.IGNORECASE,
)


def _measurement_prose(text: str) -> str:
    def scientific(match: re.Match[str]) -> str:
        exponent = match.group("plain") or match.group("bare") or match.group("super").translate(_SUPERSCRIPT_DIGITS)
        exponent = re.sub(r"\s+", "", exponent).strip("{}")
        return f"{match.group('mantissa')}e{exponent}"

    text = _SCIENTIFIC_PRODUCT.sub(scientific, text.replace("−", "-"))
    return _normalize_phenotype_spellings(re.sub(r"[*`]", "", text))


def _measurement_name(value: str) -> str:
    normalized = value.casefold().replace("-", "").replace(" ", "")
    return ("beta" if normalized == "β" else "p" if normalized in {"p", "pvalue"}
            else "or" if normalized == "oddsratio" else normalized)


def _reported_statistic(value: Any, *, label: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise StructuredOutputError(f"{label} contains an invalid reported statistic") from None
    if not number.is_finite():
        raise StructuredOutputError(f"{label} contains a nonfinite reported statistic")
    return number


def _measurement_clauses(text: str) -> tuple[str, ...]:
    """A semicolon retains sentence-level source context and variant identity."""

    return tuple(re.split(r"(?<=[.!?])\s+|\n+", text))


def _measurement_trait_phrase(text: str) -> str:
    # A display comma may become a parenthetical qualifier in prose. Keep the
    # same words, their order, and decimal study codes; only normalize layout.
    return " ".join(re.sub(r"[,()]", " ", text.replace("-", " ")).split())


def _measurement_trait_key(text: str) -> str:
    """Keep biological/transform qualifiers; study codes are checked separately."""

    text = re.sub(r"\(\s*(?:PheCode\s+[\d.]+|UKB\s+data\s+field\s+\d+)\s*\)", "", text, flags=re.IGNORECASE)
    return _measurement_trait_phrase(_normalize_phenotype_spellings(text)).casefold().strip()


def _fact_measurement_trait(fact: FinalEvidenceFact) -> str | None:
    heading = re.match(r"^-\s+\*\*(.+?)\*\*", fact.markdown)
    if heading:
        return heading.group(1).rstrip(":")
    # Molecular display rows have no bold heading. Their explicit phenotype
    # still belongs to the H4 in that same immutable display fact.
    coloc = re.search(r"\bcoloc\b.*?\bwith\s+(.+?)\s*\(\s*H4\s*(?:=|:)", fact.markdown, re.IGNORECASE)
    return coloc.group(1).strip() if coloc else None


def _joint_trait_study_phrases(
    heading: str, fact: FinalEvidenceFact, references: Mapping[str, str],
) -> set[str]:
    """Derive a narrow joint-study description from that fact's own paper.

    Both trait components and an explicit cross-trait meta-analysis must be
    present in the same cited publication metadata. These phrases describe the
    joint analysis; neither component becomes an alias of the joint phenotype.
    """

    components = re.split(r"\s+(?:or|and)\s+", _measurement_trait_key(heading))
    if len(components) != 2 or not all(components):
        return set()
    phrases = set()
    for reference_id in fact.reference_ids:
        publication = _measurement_trait_phrase(_normalize_phenotype_spellings(references.get(reference_id, ""))).casefold()
        if "cross trait meta analysis" not in publication:
            continue
        forms = []
        for component in components:
            # A paper may call circulating X levels simply X levels. Remove
            # only this nominal wrapper, and require the remaining complete
            # phrase in the very same publication before shortening "levels".
            core = re.sub(r"^circulating\s+", "", component)
            if not re.search(r"(?<!\w)" + re.escape(core) + r"(?!\w)", publication):
                break
            variants = {core}
            if core.endswith(" levels"):
                variants.add(core[:-7])
            if component in publication:
                variants.add(component)
            forms.append(variants)
        if len(forms) != 2:
            continue
        for left in forms[0]:
            for right in forms[1]:
                for first, second in ((left, right), (right, left)):
                    for separator in ("-", "–", "—", " and ", " or "):
                        phrases.add(_measurement_trait_phrase(
                            f"cross-trait {first}{separator}{second} meta-analysis"
                        ))
    return phrases


def _source_measurement_values(observation: Mapping[str, Any], name: str) -> tuple[Decimal, ...]:
    keys = {
        "p": ("p_value",), "beta": ("beta", "Beta"),
        "or": ("odds_ratio", "oddsRatio", "OR", "or"),
        "logodds": ("log_odds", "logOdds"),
    }.get(name, ())
    values = set()
    for key in keys:
        value = observation.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError):
            continue
        if (number.is_finite() and (name != "or" or number > 0)
                and (name != "p" or 0 < number <= 1)):
            values.add(number)
    # Conflicting aliases do not authorize choosing whichever value fits prose.
    return tuple(values) if len(values) == 1 else ()


def _source_effect_sign(
    observation: Mapping[str, Any], *, canonical_variant_id: str | None = None,
    stated_effect_allele: str | None = None,
) -> int | None:
    """Use the recorded EA without rewriting beta when genomic REF/ALT changes."""
    effect_allele = observation.get("effect_allele")
    if not isinstance(effect_allele, str) or not re.fullmatch(r"[ACGT]{1,32}", effect_allele.upper()):
        return None
    effect_allele = effect_allele.upper()
    identifier = str(canonical_variant_id or observation.get("variant_id") or "")
    match = re.fullmatch(r"(?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:]([ACGT]+)[_:]([ACGT]+)", identifier, re.I)
    if not match or effect_allele not in {match[1].upper(), match[2].upper()}:
        return None
    other = observation.get("other_allele")
    if other is not None and (not isinstance(other, str)
            or {effect_allele, other.upper()} != {match[1].upper(), match[2].upper()}):
        return None
    if stated_effect_allele is not None and stated_effect_allele != effect_allele:
        return None
    # A bare canonical-variant direction cannot borrow a REF-coded beta sign.
    # An explicit statement about the recorded EA can retain that original sign.
    if effect_allele != match[2].upper() and stated_effect_allele is None:
        return None
    signs = set()
    for name, keys in (("beta", ("beta", "Beta")), ("logodds", ("log_odds", "logOdds")),
                       ("or", ("odds_ratio", "oddsRatio", "OR", "or"))):
        for key in keys:
            for value in _source_measurement_values({key: observation.get(key)}, name):
                center = Decimal(1) if name == "or" else Decimal(0)
                signs.add(int(value > center) - int(value < center))
    labels = {
        "+": 1, "positive": 1, "increasing": 1, "risk_increasing": 1,
        "-": -1, "negative": -1, "decreasing": -1, "protective": -1,
        "risk_decreasing": -1, "neutral": 0, "zero": 0,
    }
    for key in ("direction", "effect_direction", "effectDirection"):
        value = str(observation.get(key, "")).casefold()
        if value in labels:
            signs.add(labels[value])
    return next(iter(signs)) if len(signs) == 1 else None


def _source_statistic_matches(reported: Decimal, actual: Decimal, name: str) -> bool:
    """Accept a source value at its stated precision without changing its meaning.

    This comparison is applied only after source and variant ownership are bound.
    Keep exact high-precision values exact. Inexact prose needs at least three
    significant digits and correct rounding, preserving significance and effect sign.
    """
    if not reported.is_finite() or not actual.is_finite():
        return False
    if name == "p" and not (0 < reported <= 1 and 0 < actual <= 1):
        return False
    if name == "or" and not (reported > 0 and actual > 0):
        return False
    if reported == actual:
        return True
    digits = len(reported.as_tuple().digits)
    exponent = reported.as_tuple().exponent
    if digits < 3 or digits > 1_000 or len(actual.as_tuple().digits) > 1_000 or abs(exponent) > 1_000 or abs(actual.adjusted()) > 1_000:
        return False
    if name == "p":
        threshold = Decimal(str(parameter("analysis.gwas_significance_p")))
        if (reported <= threshold) != (actual <= threshold):
            return False
    elif name in {"beta", "logodds", "or"}:
        center = Decimal(1) if name == "or" else Decimal(0)
        if (int(reported > center) - int(reported < center)) != (int(actual > center) - int(actual < center)):
            return False
    else:
        return False
    try:
        precision = max(28, digits + 2, len(actual.as_tuple().digits) + 2,
                        actual.adjusted() - exponent + 2)
        with localcontext(Context(prec=precision, rounding=ROUND_HALF_EVEN)):
            return actual.quantize(Decimal(1).scaleb(exponent), rounding=ROUND_HALF_EVEN) == reported
    except InvalidOperation:
        return False


def _validate_source_effect_direction(text: str, *, context: HybridFinalContext, label: str) -> None:
    """Check source direction against the corresponding uploaded effect sign.

    Source P alone supplies no sign. Cross-study agreement additionally needs
    harmonized effect-allele provenance, which free prose cannot manufacture.
    """

    source = r"\b(?:source|uploaded|loaded|current)[- ](?:GWAS|dataset|data)\b"
    direction = r"(?:effect\s+)?direction(?:\s+(?:reported|observed|seen))?"
    sign = r"(?:positive|negative|protective|risk[- ]increasing|risk[- ]reducing)"
    claims = re.compile(
        r"\b(?:consistent|concordant|aligned|inconsistent|discordant)\s+with\s+(?:the\s+)?"
        + direction + r"\s+(?:in|from|of)\s+(?:the\s+)?" + source
        + r"|" + source + r"(?:['’]s)?\s+(?:effect\s+)?direction\s+(?:is|was|appears)\s+"
        r"(?:(?:the|a)\s+)?(?:positive|negative|neutral|zero|same|opposite|consistent|concordant)\b"
        + r"|" + source + r"\s+(?:shows?|showed|has|had|reports?|reported|indicates?)\s+(?:a\s+)?"
        + sign + r"\s+(?:effect|association|beta|direction)\b"
        + r"|\b" + sign + r"\s+(?:effect|association|beta|direction)"
        r"(?:\s+(?:was|is|has\s+been))?(?:\s+(?:reported|observed|seen))?\s+(?:in|from|of)\s+(?:the\s+)?" + source
        + r"|\b(?:same|consistent|concordant|opposite|discordant)\s+(?:effect\s+)?direction\s+as\s+"
        r"(?:that\s+(?:in|from)\s+)?(?:the\s+)?" + source,
        re.IGNORECASE,
    )

    def canonical(identifier: str) -> str:
        return identifier.casefold().removeprefix("chr").replace(":", "_")

    aliases = {}
    for fact in context.facts:
        identifiers = [canonical(match.group()) for match in _REPORT_VARIANT.finditer(str(fact.model_data.get("subject", "")))]
        target = next((item for item in identifiers if not item.startswith("rs")), None)
        if target:
            aliases.update((item, target) for item in identifiers)
    resolution = context.result.get("variant_resolution")
    verified = resolution.get("validated_aliases") if isinstance(resolution, Mapping) else None
    originals = {str(row.get("variant_id") or "") for row in context.source_observations}
    if isinstance(verified, Mapping):
        for original, target in verified.items():
            if original in originals and isinstance(target, str) and _REPORT_VARIANT.fullmatch(target):
                aliases[canonical(original)] = canonical(target)

    observations: dict[str, list[Mapping[str, Any]]] = {}
    for row in context.source_observations:
        identifier = canonical(str(row.get("variant_id", "")))
        observations.setdefault(aliases.get(identifier, identifier), []).append(row)

    for clause in _measurement_clauses(_measurement_prose(text)):
        asserted = list(claims.finditer(clause))
        change_claims = re.compile(
            r"\b(?:increases?|increased|raises?|raised|decreases?|decreased|lowers?|lowered|reduces?|reduced)\s+"
            r"(?:the\s+)?(?:(?:risk|odds|levels?)\s+(?:of\s+|for\s+)?)?[A-Za-z][^.;{}]{0,100}"
            r"|\b(?:higher|lower|increased|decreased|elevated|reduced)\s+(?:risk|odds)\b[^.;{}]{0,80}", re.IGNORECASE,
        )
        source_mentions = tuple(re.finditer(source, clause, re.IGNORECASE))
        for change in change_claims.finditer(clause):
            prior_sources = [item for item in source_mentions if item.end() <= change.start()]
            nearest_source = prior_sources[-1] if prior_sources else None
            if nearest_source and not re.search(r"\b(?:external|retrieved|publication|published|study)\b", clause[nearest_source.end():change.start()], re.IGNORECASE):
                asserted.append(change)
            elif re.search(r"\b(?:in|from|of)\s+(?:the\s+)?" + source, clause[change.start():], re.IGNORECASE):
                asserted.append(change)
        for claim in asserted:
            before = clause[:claim.start()]
            unasserted = re.search(
                r"\b(?:(?:cannot|can't|could\s+not|unable\s+to)\s+(?:infer|determine|establish|assess|confirm|say)"
                r"|(?:unclear|unknown|not\s+known)\s+whether|no\s+evidence\s+(?:that|of)"
                r"|do(?:es)?\s+not\s+(?:establish|prove|show|imply))"
                r"(?:\s+(?!(?:but|however|yet)\b)\w+){0,10}\s*$", before, re.IGNORECASE,
            )
            if unasserted:
                continue
            identifiers = {aliases.get(canonical(match.group()), canonical(match.group()))
                           for match in _REPORT_VARIANT.finditer(clause)}
            selected = identifiers or set(observations)
            named_effect_alleles = {value.upper() for value in re.findall(
                r"\b(?:effect[- ]allele|EA)(?:\s+is\s+|\s*[:=]\s*|\s+)([ACGT]{1,32})\b", clause, re.I)}
            stated_effect_allele = next(iter(named_effect_alleles)) if len(named_effect_alleles) == 1 else None
            signs = {}
            for identifier in selected:
                row_signs = {_source_effect_sign(row, canonical_variant_id=identifier,
                    stated_effect_allele=stated_effect_allele) for row in observations.get(identifier, ())}
                # Standardizing two uploaded aliases must not silently choose
                # one conflicting effect observation by insertion order.
                signs[identifier] = next(iter(row_signs)) if len(row_signs) == 1 else None
            if not selected or len(named_effect_alleles) > 1 or any(signs.get(identifier) is None for identifier in selected):
                raise StructuredOutputError(
                    f"{label} asserts unsupported source-GWAS effect direction: the corresponding uploaded "
                    "observation needs a usable effect and an explicit effect allele bound to that variant. "
                    "A REF-coded effect must name its recorded effect allele; do not assign its sign to canonical ALT. "
                    "Retain original Beta, odds ratio and P values without changing their signs."
                )
            claim_text = claim.group().casefold()
            if re.search(r"\b(?:negative|protective|risk[- ]reducing|decreases?|decreased|lowers?|lowered|reduces?|reduced)\b", claim_text):
                asserted_sign = -1
            elif re.search(r"\b(?:positive|risk[- ]increasing|increases?|increased|raises?|raised|higher|elevated)\b", claim_text):
                asserted_sign = 1
            elif re.search(r"\b(?:neutral|zero)\b", claim_text):
                asserted_sign = 0
            else:
                raise StructuredOutputError(
                    f"{label} asserts unsupported source-GWAS effect direction agreement: "
                    "source effect availability alone does not establish a harmonized cross-study comparison."
                )
            if any(signs[identifier] != asserted_sign for identifier in selected):
                raise StructuredOutputError(
                    f"{label} asserts unsupported source-GWAS effect direction: the stated sign "
                    "contradicts the corresponding uploaded Beta, odds ratio or direction."
                )


def _uploaded_data_label(match: re.Match[str], clause: str, *, offset: int = 0) -> bool:
    """Exclude external qualifiers from the new adjacent-data provenance alias.

    Inspect the full sentence for a data label matched in a prefix or suffix so
    that truncation at a measurement/citation cannot hide its actual source.
    Source-record narration cannot absorb a subsequent external GWAS report.
    """

    if match.group("source_data") is None:
        if re.fullmatch(r"(?:loaded|uploaded|source|current)\s*", match.group(), re.I):
            # "Source P" is a local uploaded-statistic label, unless source
            # itself is explicitly owned by an external study/publication.
            return not re.search(
                r"\b(?:external|published|cited|reported|retrieved)"
                r"(?:\s+(?:study|publication|report|article|database|cohort))?"
                r"(?:['\u2019]s)?\s+$", clause[:offset + match.start()], re.I,
            )
        # A phrase such as "source record, the variant has been reported in
        # ... GWAS" crosses into another study. Do not consume that transition
        # inside the source label and then misattribute its external statistic.
        return not re.search(
            r"\b(?:has|have|had|is|are|was|were)\s+(?:also\s+)?(?:been\s+)?"
            r"(?:reported|retrieved|published|identified|observed)\b",
            match.group(), re.IGNORECASE,
        )
    start, end = match.span("source_data")
    external = r"(?:external|published|cited|reported|retrieved)"
    left_external = re.search(
        r"\b" + external + r"(?:\s+(?:study|publication|report|article|database|cohort))?"
        r"(?:['’]s)?\s+$", clause[:offset + start], re.IGNORECASE,
    )
    right_external = re.match(
        r"\s*[,()]?\s*(?:(?:from|of|in|by)|(?:reported|published|provided|retrieved|obtained)"
        r"\s+(?:by|from|in))\s+(?:(?:an?|the)\s+)?" + external + r"\b",
        clause[offset + end:], re.IGNORECASE,
    )
    return not (left_external or right_external)


def _validate_fact_bound_measurements(
    text: str, *, context: HybridFinalContext, label: str,
    _source_number_replacements: list[tuple[int, int, str]] | None = None,
    _source_only: bool = False,
) -> None:
    """Validate explicit reported statistics against their own displayed statement.

    This applies only to retrieval-backed report prose, never to the separate
    general-concept writer. A citation is not permission to borrow any statistic
    from its paper, another variant, or the uploaded source GWAS.
    """

    facts = {fact.fact_id: fact for fact in context.facts}
    fact_stats = {
        fact.fact_id: tuple(
            (_measurement_name(match.group("metric")), _reported_statistic(match.group("number"), label=label))
            for match in _REPORT_MEASUREMENT.finditer(_measurement_prose(fact.markdown))
        ) for fact in context.facts
    }
    owners: dict[str, set[str]] = {}
    source_statistic_owners: dict[str, set[str]] = {}
    fact_gene_owners: dict[str, set[str]] = {}
    from .report_facts import public_source_normalizations
    from .qtl_measurement_binding import credible_set_lead_binding
    source_ids = {str(row.get("variant_id") or "") for row in context.source_observations}
    normalized_sources = {
        row["original_id"].casefold().removeprefix("chr").replace(":", "_"):
            row["canonical_id"].casefold().removeprefix("chr").replace(":", "_")
        for row in public_source_normalizations(context.result) if row["original_id"] in source_ids
    }
    # The original observation still owns its statistic. Only a complete,
    # validated correction proof permits its canonical query ID as an alias;
    # a general report mapping cannot move a source P value to another SNP.
    aliases: dict[str, str] = dict(normalized_sources)
    traits: dict[str, str] = {}
    trait_bases: dict[str, str] = {}
    trait_phrases: dict[str, str] = {}
    joint_study_phrases: dict[str, set[str]] = {}
    reference_metadata = {reference.reference_id: reference.markdown for reference in context.references}
    phecodes: dict[str, set[str]] = {}
    ukb_traits: dict[str, tuple[str, str]] = {}
    gene_aliases = _reported_gene_aliases(context)
    ld_scope = _final_ld_report_scope(context)
    for fact in context.facts:
        subject = str(fact.model_data.get("subject") or fact.model_data.get("gene") or "")
        identifiers = tuple(match.group(0).casefold().removeprefix("chr").replace(":", "_")
                            for match in _REPORT_VARIANT.finditer(subject))
        canonical = next((item for item in identifiers if not item.startswith("rs")), None)
        for identifier in identifiers:
            target = canonical or identifier
            aliases[identifier] = normalized_sources.get(target, target)
        owner = {aliases[item] for item in identifiers}
        lead_binding = credible_set_lead_binding(fact.markdown)
        if lead_binding is not None:
            lead_identifier = lead_binding['variant_id'].casefold()
            owner = {aliases.get(lead_identifier, lead_identifier)}
        elif ld_scope.proxy_ids:
            owner.update(match.group(0).casefold() for match in _REPORT_VARIANT.finditer(fact.markdown)
                         if match.group(0) in ld_scope.proxy_ids)
        owner.update(gene for alias, gene in gene_aliases.items() if _term_in_text(alias, _grounding_text(subject)))
        fact_gene_owners[fact.fact_id] = {gene for alias, gene in gene_aliases.items()
                                        if _term_in_text(alias, _grounding_text(subject))}
        # Only the program-rendered member-statistic label establishes this
        # owner. An H4 study-pair lookup or an arbitrary variant mention does not.
        if not any(name in {"h4", "h3"} for name, _ in fact_stats[fact.fact_id]):
            representative = re.search(
                r"\(displayed statistics:\s*representative source-selected variant\s+"
                r"(?P<variant>(?:chr)?(?:[0-9]+|X|Y|MT)[_:][0-9]+[_:][ACGT]+[_:][ACGT]+)\s*;\s*study\b",
                _measurement_prose(fact.markdown), re.IGNORECASE,
            )
            selected = {aliases.get(value, value) for item in context.source_observations
                        if (value := str(item.get("variant_id", "")).casefold().removeprefix("chr").replace(":", "_"))}
            if representative or lead_binding is not None:
                identifier = (lead_binding['variant_id'].casefold() if lead_binding is not None else
                    representative.group("variant").casefold().removeprefix("chr").replace(":", "_"))
                identifier = aliases.get(identifier, identifier)
                if identifier in selected:
                    source_statistic_owners[fact.fact_id] = {identifier}
        owners[fact.fact_id] = owner
        heading = _fact_measurement_trait(fact)
        if heading:
            traits[fact.fact_id] = _measurement_trait_key(heading)
            trait_phrases[_measurement_trait_phrase(traits[fact.fact_id])] = traits[fact.fact_id]
            joint_phrases = _joint_trait_study_phrases(heading, fact, reference_metadata)
            for phrase in joint_phrases:
                trait_phrases[phrase] = traits[fact.fact_id]
                joint_study_phrases.setdefault(phrase, set()).add(fact.fact_id)
            if joint_phrases:
                for component in re.split(r"\s+(?:or|and)\s+", traits[fact.fact_id]):
                    core = re.sub(r"^circulating\s+", "", component)
                    # A component named alone is a different trait, not an
                    # unknown-phrase wildcard authorized by the joint P value.
                    for short in (component, core, core[:-7] if core.endswith(" levels") else core):
                        trait_phrases.setdefault(short, component)
            age_onset = re.fullmatch(r"age of onset of (.+)", traits[fact.fact_id])
            if age_onset:
                # Preserve the endpoint when prose names the same disease
                # before "GWAS for age of onset"; this is not a disease-risk
                # association and cannot borrow a plain disease's statistics.
                trait_phrases[f"{age_onset.group(1)} gwas for age of onset"] = traits[fact.fact_id]
                trait_phrases.setdefault(age_onset.group(1), age_onset.group(1))
            base = _measurement_trait_key(re.split(r"[,()]", heading, maxsplit=1)[0])
            trait_bases[fact.fact_id] = base
            if base and base != traits[fact.fact_id] and "UKB data field" not in heading:
                # A named bare phenotype must not become an unknown phrase
                # wildcard when its transformation/biological qualifier changes.
                trait_phrases.setdefault(base, base)
            phecodes[fact.fact_id] = set(re.findall(r"\bPheCode\s+(\d+(?:\.\d+)*)", heading, re.IGNORECASE))
            full_heading = _normalize_phenotype_spellings(heading).casefold().replace("-", " ")
            # Display parentheticals can continue with "; P=..." in prose.
            # Preserve their named phenotype scope even without the closing
            # parenthesis immediately after the label.
            trait_phrases[_measurement_trait_phrase(full_heading)] = traits[fact.fact_id]
            adjusted = re.search(r"\(\s*adjusted\s+for\s+([^()]+)\)", heading, re.IGNORECASE)
            if adjusted:
                covariate = _measurement_trait_phrase(adjusted.group(1)).casefold()
                # Preserve an explicit, named adjustment across its ordinary
                # adjectival form; never infer a covariate absent from the fact.
                for connector in ("", "association ", "signal "):
                    trait_phrases[f"{base} {connector}{covariate} adjusted"] = traits[fact.fact_id]
                trait_phrases[f"{covariate} adjusted {base}"] = traits[fact.fact_id]
            ukb = re.fullmatch(
                r"(.+?)\s*\(UKB\s+data\s+field\s+(\d+)\)", heading, re.IGNORECASE,
            )
            if ukb:
                # A UKB multi-condition question names the actual response
                # after " - ". Keep its exact field and response together;
                # the other conditions in that question are not aliases.
                response = ukb.group(1).rsplit(" - ", 1)[-1]
                ukb_traits[fact.fact_id] = (ukb.group(2), _coloc_trait_alias(response))

    def mentioned_owner(prefix: str, *, include_genes: bool = True) -> str | None:
        mentions = []
        for match in _REPORT_VARIANT.finditer(prefix):
            token = match.group(0).casefold().removeprefix("chr").replace(":", "_")
            before = prefix[:match.start()]
            after = prefix[match.end():]
            lead_label = re.search(r"\blead(?:\s+(?:variant|SNP))?(?:\s+is)?\s*$", before, re.IGNORECASE)
            member_statistic = re.search(r"\bmember\s*$", after, re.IGNORECASE)
            lead_aside = re.search(r"\bwith\s+(?:the\s+)?lead(?:\s+(?:variant|SNP))?\s*$", before, re.IGNORECASE)
            # A separately named lead is context for a member statistic. Keep
            # the member as subject; an explicit lead predicate remains a lead
            # assertion and must still fail if it borrows the member's value.
            aside = lead_label and (
                member_statistic or lead_aside and not re.search(
                    r"\b(?:has|had|shows?|reported|is|was|lead)\b", after, re.IGNORECASE,
                )
            )
            if aside and mentions:
                continue
            mentions.append((match.start(), aliases.get(token, token)))
        for alias, gene in (gene_aliases.items() if include_genes else ()):
            mentions.extend((match.start(), gene) for match in re.finditer(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)", prefix, re.IGNORECASE,
            ))
        return max(mentions, default=(0, None), key=lambda item: item[0])[1]

    def mentioned_trait(prefix: str) -> str | None:
        prefix = _measurement_trait_phrase(prefix)
        mentions = [
            (match.start(), match.end(), trait) for phrase, trait in trait_phrases.items() if phrase
            for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", prefix, re.IGNORECASE)
        ]
        # "Childhood onset asthma" owns its statistic; its contained word
        # "asthma" must not replace that more specific retrieved phenotype.
        mentions = [item for item in mentions if not any(
            other[0] <= item[0] and other[1] >= item[1]
            and other[1] - other[0] > item[1] - item[0] for other in mentions
        )]
        return max(mentions, default=(0, 0, None), key=lambda item: (item[0], item[1] - item[0]))[2]

    def source_owner(observation: Mapping[str, Any]) -> str:
        token = str(observation.get("variant_id", "")).casefold().removeprefix("chr").replace(":", "_")
        return aliases.get(token, token)

    def compatible_trait(fact_id: str, trait: str | None, prefix: str, suffix: str, name: str) -> bool:
        for region in (prefix, suffix):
            descriptions = tuple(re.finditer(
                r"\bcross[-\s]trait\s+[^.;:{}]{1,180}?\s+meta[-\s]analysis\b", region, re.IGNORECASE,
            ))
            if not descriptions:
                continue
            description = descriptions[-1]
            tail = _SUMMARY_FACT_CITATION_TOKEN.sub("", _REPORT_MEASUREMENT.sub("", region[description.end():]))
            if re.sub(r"\band\b", "", tail, flags=re.IGNORECASE).strip(" ,;()[]*`\t\r\n"):
                continue
            phrase = _measurement_trait_phrase(description.group()).casefold()
            if fact_id not in joint_study_phrases.get(phrase, set()):
                return False
        if trait is None or fact_id not in traits or traits[fact_id] == trait:
            return True
        if name == "h4" or trait_bases.get(fact_id) != trait:
            return False
        # A unique cited HbA1c/glucose phenotype may use its bare name. This
        # does not authorize replacing an explicitly stated transformation,
        # nor collapse raw and transformed facts when both were retrieved.
        alternatives = {traits[key] for key, base in trait_bases.items() if base == trait}
        if len(alternatives) != 1:
            return False
        region = suffix if mentioned_trait(suffix) == trait else prefix
        pattern = re.escape(trait).replace(r"\ ", r"[\s-]+")
        mentions = tuple(re.finditer(r"(?<!\w)" + pattern + r"(?!\w)", region, re.IGNORECASE))
        if not mentions:
            return False
        tail = region[mentions[-1].end():]
        qualifier = re.match(r"\s*\(\s*([^)]*)", tail)
        if qualifier:
            detail = _REPORT_VARIANT.sub("", _REPORT_MEASUREMENT.sub("", qualifier.group(1)))
            detail = re.sub(r"\be\.g\.|\bi\.e\.|\b(?:lead|member|selected|variant|SNP|for|with|and)\b", "", detail, flags=re.IGNORECASE)
            if re.search(r"[A-Za-z]", detail):
                return False
        before_name = region[max(0, mentions[-1].start() - 80):mentions[-1].start()]
        if re.search(r"\b(?:unnormalized|unadjusted|log[- ]transformed)\s*$", before_name, re.IGNORECASE):
            return False
        return True

    molecular_bindings = None
    signal_reference = re.compile(
        r"\b(?:that|this|the\s+same)\s+(?:[A-Za-z-]+\s+){0,3}"
        r"(?P<type>eQTL|sQTL)\s+signal\s+"
        r"(?:colocali[sz](?:es|ed)|(?:has|shows)\s+(?:a\s+)?colocali[sz]ation)\b", re.IGNORECASE)

    def unique_qtl_signal_owner(prefix: str, previous_anchors, cited_ids, gene_owner) -> str | None:
        """Resolve one explicit QTL-signal anaphor through its actual studies.

        This can replace an earlier lookup-SNP holder with a uniquely bound
        signal/gene owner. It never makes H4 a variant-level measurement.
        Ambiguous or absent provenance leaves the existing strict checks intact.
        """
        nonlocal molecular_bindings
        if len(cited_ids) != 1 or not previous_anchors:
            return None
        after_anchor = prefix[previous_anchors[-1].end():]
        signal = signal_reference.search(after_anchor)
        if signal is None or _REPORT_VARIANT.search(after_anchor):
            return None
        current = facts.get(cited_ids[0])
        if current is None:
            return None
        current_genes = fact_gene_owners.get(current.fact_id, set())
        if len(current_genes) != 1:
            return None
        gene = next(iter(current_genes))
        if gene_owner is not None and gene_owner != gene:
            return None
        current_text = _measurement_prose(current.markdown)
        coloc_type = re.search(r"\b(eQTL|sQTL)\s+coloc\b", current_text, re.IGNORECASE)
        pair = re.search(r"\(H4 study pair:\s*(\S+)\s+vs\s+([^\s;()]+)\s*;", current_text)
        if pair is None or coloc_type is None or coloc_type[1].casefold() != signal['type'].casefold():
            return None

        # Lazily scan retained typed evidence once per validation call, and
        # retain only the natural identities needed for this limited join.
        if molecular_bindings is None:
            qtls, colocs = {}, {}
            for row in _runtime_evidence_rows(context.result):
                kind = str(row.get("kind", ""))
                if kind not in {"qtl", "colocalization"}:
                    continue
                payload = row.get("payload")
                if not isinstance(payload, Mapping):
                    continue
                upstream, metadata = payload.get("upstream"), payload.get("context")
                if not isinstance(upstream, Mapping) or not isinstance(metadata, Mapping):
                    continue
                raw = upstream.get("raw")
                if not isinstance(raw, Mapping):
                    continue
                study_ids = {str(value) for value in (row.get("study_id"), raw.get("studyId"), metadata.get("study_id")) if value}
                loci = {str(value) for value in (row.get("credible_set_id"), raw.get("studyLocusId"), raw.get("leftStudyLocusId")) if value}
                if len(study_ids) != 1 or len(loci) != 1 or not row.get("gene_id"):
                    continue
                study, locus = next(iter(study_ids)), next(iter(loci))
                if kind == "qtl":
                    qtl_types = {str(value).casefold() for value in (raw.get("studyType"), metadata.get("study_type")) if value}
                    if len(qtl_types) != 1:
                        continue
                    qtl_type = next(iter(qtl_types))
                    key = (row["gene_id"], study, str(row.get("variant_id", "")), qtl_type)
                    qtls.setdefault(key, set()).add(locus)
                elif all(raw.get(key) for key in ("rightStudyId", "rightStudyLocusId", "colocalisationMethod")):
                    key = (row["gene_id"], study, str(raw["rightStudyId"]))
                    colocs.setdefault(key, set()).add((locus, str(raw["rightStudyLocusId"]), str(raw["colocalisationMethod"])))
            molecular_bindings = qtls, colocs
        qtls, colocs = molecular_bindings
        joined_qtls = set()
        last_was_qtl = False
        for anchor in previous_anchors:
            prior = facts.get(anchor.group(1))
            if prior is None:
                return None
            prior_text = _measurement_prose(prior.markdown)
            qtl_type = re.search(r"\b(eQTL|sQTL)\s*\(", prior_text, re.IGNORECASE)
            member = re.search(
                r"\(displayed statistics:\s*representative (?P<origin>source-selected variant|LD proxy)\s+"
                r"(?P<variant>(?:chr)?(?:[0-9]+|X|Y|MT)[_:][0-9]+[_:][ACGT]+[_:][ACGT]+)\s*;\s*study\s+"
                r"(?P<study>[^\s;)]+)", prior_text, re.IGNORECASE,
            )
            last_was_qtl = bool(qtl_type and member and not any(name in {"h4", "h3"} for name, _ in fact_stats[prior.fact_id]))
            if not last_was_qtl:
                continue
            prior_genes = fact_gene_owners.get(prior.fact_id, set())
            if prior_genes != {gene} or qtl_type[1].casefold() != signal['type'].casefold():
                return None
            variant = member['variant'].removeprefix('chr').replace(':', '_')
            allowed_variants = (ld_scope.proxy_ids if member['origin'].casefold() == 'ld proxy' else
                                tuple(str(item.get('variant_id', '')) for item in context.source_observations))
            if variant not in allowed_variants:
                return None
            possible = qtls.get((gene, member['study'], variant, qtl_type[1].casefold()), set())
            if len(possible) != 1:
                return None
            joined_qtls.add((member['study'], next(iter(possible))))
        if not last_was_qtl or len(joined_qtls) != 1:
            return None
        study, locus = next(iter(joined_qtls))
        pairs = colocs.get((gene, pair[1], pair[2]), set())
        if study != pair[1] or len(pairs) != 1 or next(iter(pairs))[0] != locus:
            return None
        return gene

    normalized_prose = _measurement_prose(text)
    next_clause_offset = 0
    for clause in _measurement_clauses(normalized_prose):
        clause_offset = normalized_prose.find(clause, next_clause_offset)
        next_clause_offset = clause_offset + len(clause)
        anchors = tuple(_SUMMARY_FACT_CITATION_TOKEN.finditer(clause))
        for metric in _REPORT_MEASUREMENT.finditer(clause):
            prefix = clause[:metric.start()]
            name = _measurement_name(metric.group("metric"))
            number = _reported_statistic(metric.group("number"), label=label)
            owner = mentioned_owner(prefix)
            variant_owner = mentioned_owner(prefix, include_genes=False)
            gene_mentions = [(match.start(), gene) for alias, gene in gene_aliases.items()
                             for match in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", prefix, re.IGNORECASE)]
            gene_owner = max(gene_mentions)[1] if gene_mentions else None
            # "P=... for rsID" binds as specifically as a preceding rsID. Do
            # not let a later, separately introduced variant steal this value.
            following_owner = _REPORT_VARIANT.search(clause, metric.end())
            if following_owner and re.fullmatch(
                r"\s*[,)]?\s*(?:for|at|in)\s+", clause[metric.end():following_owner.start()], re.IGNORECASE,
            ):
                owner = mentioned_owner(following_owner.group())
                variant_owner = owner
            # A current-statistic "for TRAIT" owns that statistic even when an
            # earlier statistic named another trait. Never cross the next
            # subject, measurement or citation into a later statement.
            trailing_end = len(clause)
            for pattern in (_REPORT_VARIANT, _REPORT_MEASUREMENT, _SUMMARY_FACT_CITATION_TOKEN):
                boundary = pattern.search(clause, metric.end())
                if boundary is not None:
                    trailing_end = min(trailing_end, boundary.start())
            suffix = clause[metric.end():trailing_end]
            trailing = re.match(r"\s*[,)]?\s*(?:for|with|in)\s+(.+)", suffix, re.IGNORECASE)
            trait = (mentioned_trait(trailing.group(1)) if trailing else None) or mentioned_trait(prefix)
            preceding_anchors = [anchor for anchor in anchors if anchor.end() <= metric.start()]
            local_prefix = prefix[preceding_anchors[-1].end():] if preceding_anchors else prefix
            claimed_codes = set(re.findall(r"\bPheCode\s+(\d+(?:\.\d+)*)", local_prefix + (trailing.group(1) if trailing else ""), re.IGNORECASE))
            ukb_mentions = tuple(re.finditer(
                r"\b(?:UK\s+Biobank|UKB)\s+(?:data\s+)?field\s+(\d+)\s+"
                r"([^.,;()\n{}]{1,160}?)\s+definition\b", prefix, re.IGNORECASE,
            ))
            ukb_scope = None
            if ukb_mentions:
                ukb = ukb_mentions[-1]
                after_ukb = prefix[ukb.end():]
                if mentioned_trait(after_ukb) is None and _REPORT_VARIANT.search(after_ukb) is None:
                    ukb_scope = (ukb.group(1), _coloc_trait_alias(ukb.group(2)))
            # Keep adjacent provenance labels separate. Otherwise "source P
            # value ... in the uploaded dataset" is consumed as one label that
            # straddles its statistic, so neither prefix nor suffix owns it.
            source_label = r"\b(?:loaded|uploaded|source|current)\b"
            # "Source data" also names uploaded observations, including a
            # selected variant that the external resolver could not identify.
            # Only the actual source phenotype may separate provenance from
            # "data"; general/external data is not the uploaded source GWAS.
            source_trait_label = re.escape(_measurement_prose(context.source_trait).strip())
            named_source = (source_trait_label + r"\s+(?:GWAS|dataset|data)\b|") if source_trait_label else ""
            source_pattern = (
                source_label + r"[-\s]+(?:" + named_source + r"(?:GWAS|dataset)\b)"
                r"|(?P<source_data>" + source_label + r"[-\s]+data\b)"
                r"|\b(?:source|uploaded)\s+(?:observations|(?:selected\s+)?(?:variants?|SNPs?))\b"
                r"|" + source_label + r"\s+$"
            )
            source_mentions = tuple(
                match for match in re.finditer(source_pattern, prefix, re.IGNORECASE)
                if _uploaded_data_label(match, clause)
            )
            source = source_mentions[-1] if source_mentions else None
            # The label may immediately follow the P value, inside or outside
            # its parentheses, instead of preceding the variant identifier.
            suffix_source = re.match(
                r"\s*[,)]?\s*(?:(?:in|from|for|of)\s+)?(?:the\s+)?(?:" + source_pattern + r")",
                suffix, re.IGNORECASE,
            )
            if suffix_source and not _uploaded_data_label(suffix_source, clause, offset=metric.end()):
                suffix_source = None
            sentence_sources = tuple(
                match for match in re.finditer(source_pattern, clause, re.IGNORECASE)
                if _uploaded_data_label(match, clause)
            )
            sentence_metrics = tuple(_REPORT_MEASUREMENT.finditer(clause))
            trailing_source = bool(sentence_sources) and sentence_sources[-1].start() >= sentence_metrics[-1].end()
            # A shared sentence-final source label covers the preceding pair of
            # selected variants, provided no explicit external claim intervenes.
            trailing_source = trailing_source and not re.search(
                r"\b(?:external|reported|credible|[a-z-]*QTL|coloc\w*)\b", clause[:sentence_sources[-1].start()], re.IGNORECASE,
            ) and (trait is None or trait == _measurement_trait_key(context.source_trait))
            source_context = source and not re.search(
                r"\b(?:external|reported|credible|[a-z-]*QTL|coloc\w*)\b", prefix[source.end():], re.IGNORECASE,
            ) and (trait is None or trait == _measurement_trait_key(context.source_trait))
            # A post-statistic attribution owns this same measurement. A
            # leading "source P" cannot borrow the uploaded value when the
            # sentence explicitly attributes it to an external publication.
            external_attribution = re.match(
                r"\s*[,)]?\s*(?:(?:is|was|are|were|has\s+been|had\s+been)\s+)?"
                r"(?:(?:reported|published|retrieved|obtained|taken)\s+(?:by|in|from)|from)\s+"
                r"(?:(?:an?|the)\s+)?external\s+(?:study|publication|report|article|database|cohort)\b",
                suffix, re.I,
            )
            if external_attribution:
                source_context = False
            if suffix_source or source_context or trailing_source:
                observations = tuple(item for item in context.source_observations
                                     if owner is None or source_owner(item) == owner)
                source_values = [_source_measurement_values(item, name) for item in observations]
                matches = [any(_source_statistic_matches(number, actual, name) for actual in values)
                           for values in source_values]
                if matches and (all(matches) if name != "p" and owner is None else any(matches)):
                    if _source_number_replacements is not None:
                        actual = next(actual for values in source_values for actual in values
                                      if _source_statistic_matches(number, actual, name))
                        if actual != number:
                            _source_number_replacements.append((
                                clause_offset + metric.start("number"),
                                clause_offset + metric.end("number"), str(actual),
                            ))
                    continue
                if _source_only:
                    continue
                raise StructuredOutputError(f"{label} assigns an incorrect source-GWAS {name} measurement to its variant")
            if _source_only:
                continue
            following = next((anchor for anchor in anchors if anchor.start() >= metric.end()), None)
            if name == "h4" and following is None:
                # The existing gene/phenotype/H4 validator handles unanchored
                # H4 narrative against its complete typed record pairs.
                continue
            candidates = list(facts)
            if following is not None:
                candidates = [following.group(1)]
                tail = clause[following.end():]
                previous_end = 0
                for extra in _SUMMARY_FACT_CITATION_TOKEN.finditer(tail):
                    if tail[previous_end:extra.start()].strip(" ,"):
                        break
                    candidates.append(extra.group(1))
                    previous_end = extra.end()
            if name == "h4" and owner is not None and owner == variant_owner:
                signal_owner = unique_qtl_signal_owner(prefix, preceding_anchors, candidates, gene_owner)
                if signal_owner is not None:
                    owner = signal_owner
                elif preceding_anchors and signal_reference.search(prefix[preceding_anchors[-1].end():]):
                    raise StructuredOutputError(
                        f"{label} has an H4 measurement with an ambiguous or unsupported QTL-signal reference. "
                        "Name the supported gene and study pair explicitly; do not assign H4 to a lookup SNP."
                    )
            candidates = [fact_id for fact_id in candidates if fact_id in facts
                          and (owner is None or not owners[fact_id] or owner in owners[fact_id]
                               or name not in {"h4", "h3"} and owner in source_statistic_owners.get(fact_id, set()))
                          and (name in {"h4", "h3"} or fact_id not in source_statistic_owners or (
                              (variant_owner is None or variant_owner in source_statistic_owners[fact_id])
                              and (gene_owner is None or gene_owner in fact_gene_owners[fact_id])))
                          and (not claimed_codes or claimed_codes == phecodes.get(fact_id, set()))
                          and (
                              (following is not None and ukb_traits.get(fact_id) == ukb_scope)
                              if ukb_scope is not None else
                              compatible_trait(fact_id, trait, prefix, suffix, name)
                          )]
            if any((name, number) in fact_stats[fact_id] for fact_id in candidates):
                continue
            raise StructuredOutputError(
                f"{label} mismatches a statement/variant/{name} measurement: {name}={number}. "
                "Use only that same cited display fact's statistic; missing member statistics "
                "cannot borrow lead-variant or source-GWAS values. Prefer qualitative synthesis."
            )

        if _source_only:
            continue
        negative = re.search(r"\b(?:no(?:\s+\w+){0,3}\s+risk[- ]increasing|protective|risk[- ]reducing|lower\s+risk)\b", clause, re.IGNORECASE)
        if negative and not re.search(r"\b(?:cannot|unclear|unknown|no\s+evidence|not\s+(?:known|established|possible)|(?:does|do)\s+not\s+(?:establish|prove|imply|mean|show|support)|(?:is|are)\s+not)\b", clause[:negative.start()], re.IGNORECASE):
            trait = mentioned_trait(clause[:negative.start()]) or mentioned_trait(clause)
            owner = mentioned_owner(clause[:negative.start()])
            betas = [number for fact_id, stats in fact_stats.items()
                     if trait is not None and traits.get(fact_id) == trait
                     and (owner is None or not owners[fact_id] or owner in owners[fact_id])
                     for name, number in stats if name == "beta"]
            if betas and all(number > 0 for number in betas):
                raise StructuredOutputError(
                    f"{label} contradicts the positive reported Beta for {trait}; "
                    "do not infer a protective or non-risk-increasing direction."
                )


@memoized_report_derivation
def _validated_ranked_gene_symbols(context: HybridFinalContext) -> dict[str, str]:
    """Bind ranked candidate names to their actual target annotations.

    A context-only ranked candidate can lack a displayed direct-evidence section.
    Its verified symbol is still an objective identity; this adds no biological
    assertion or measurement from the unshown annotation to report authority.
    """
    ranked = {row.get("gene_id") for row in _final_ranking_context(context.result)}
    bindings = {}
    for row in _runtime_evidence_rows(context.result):
        gene = row.get("gene_id")
        if not isinstance(gene, str) or gene not in ranked or row.get("provenance_complete") is not True:
            continue
        payload = row.get("payload") or {}
        if not isinstance(payload, Mapping):
            continue
        upstream = payload.get("upstream")
        raw = upstream.get("raw") if isinstance(upstream, Mapping) else {}
        raw = raw if isinstance(raw, Mapping) else {}
        context_values = payload.get("context")
        context_values = context_values if isinstance(context_values, Mapping) else {}
        entities = [entity for key in ("subject", "object") if isinstance(entity := payload.get(key), Mapping)]
        bound = (any(entity.get("type") == "target" and entity.get("id") == gene for entity in entities)
                 or any(raw.get(key) == gene for key in ("id", "geneId", "qtlGeneId", "targetId")))
        if not bound:
            continue
        values = [raw.get("approvedSymbol"), context_values.get("approved_symbol")]
        for entity in entities:
            if entity.get("type") == "target" and entity.get("id") == gene:
                values.extend((entity.get("approvedSymbol"), entity.get("label")))
        for value in values:
            if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9-]{1,30}", value):
                bindings.setdefault(value, set()).add(gene)
    return {symbol: next(iter(genes)) for symbol, genes in bindings.items() if len(genes) == 1}


@memoized_report_derivation
def _reported_gene_aliases(context: HybridFinalContext) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for row in _runtime_evidence_rows(context.result):
        gene_id = str(row.get("gene_id") or "")
        if not gene_id:
            continue
        aliases[gene_id] = gene_id
        for term in _runtime_record_terms(row):
            if re.fullmatch(r"[A-Z][A-Z0-9-]{1,30}", term):
                aliases[term] = gene_id
    return aliases


def _coloc_trait_alias(trait: str) -> str:
    # Parenthetical study/PheCode labels remain provenance, not a different trait.
    base = _normalize_phenotype_spellings(trait).split("(", 1)[0]
    return " ".join(base.casefold().replace("-", " ").split()).strip()


@memoized_report_derivation
def _reported_colocalization_groups(context: HybridFinalContext):
    """Index unchanged natural identities; retain all H4 values and their order."""
    groups = {}
    for pair in _reported_colocalization_pairs(context):
        key = pair['gene_id'], _measurement_trait_key(pair['trait'])
        codes = tuple(sorted(set(re.findall(r"\bPheCode\s+(\d+(?:\.\d+)*)", pair['trait'], re.IGNORECASE))))
        groups.setdefault(key, []).append((pair['h4'], codes))
    return {key: tuple(values) for key, values in groups.items()}


def _validate_colocalization_pairs(text: str, *, context: HybridFinalContext, label: str) -> None:
    if not re.search(r"\bH4\b", text, re.IGNORECASE):
        return
    groups = _reported_colocalization_groups(context)
    aliases = _reported_gene_aliases(context)
    if not groups or not aliases:
        return
    plain = _measurement_prose(text)
    gene_pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(item) for item in
                                         sorted(aliases, key=len, reverse=True))
        + r")(?![A-Za-z0-9_])", re.IGNORECASE,
    )
    gene_lookup = {key.casefold(): value for key, value in aliases.items()}
    trait_aliases = {trait for _gene, trait in groups}
    phenotype = str(context.biological_context.get("phenotype") or "")
    if phenotype:
        trait_aliases.add(_measurement_trait_key(phenotype))
    trait_aliases.discard("")
    trait_pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(
            re.escape(alias).replace(r"\ ", r"[\s-]+")
            for alias in sorted(trait_aliases, key=len, reverse=True)
        ) + r")(?![A-Za-z0-9])", re.IGNORECASE,
    )
    metrics = tuple(match for match in _REPORT_MEASUREMENT.finditer(plain)
                    if _measurement_name(match.group("metric")) == "h4")
    previous_end = 0
    for index, metric in enumerate(metrics):
        gene_mentions = tuple(gene_pattern.finditer(plain[:metric.start()]))
        if not gene_mentions:
            previous_end = metric.end()
            continue
        gene_mention = gene_mentions[-1]
        gene_id = gene_lookup[gene_mention.group(1).casefold()]
        boundaries = tuple(re.finditer(r"(?<=[.!?])\s+|\n+", plain[:metric.start()]))
        start = max(previous_end, boundaries[-1].end() if boundaries else 0,
                    gene_mention.end())
        before = _measurement_trait_phrase(plain[start:metric.start()])
        after_end = metrics[index + 1].start() if index + 1 < len(metrics) else len(plain)
        after = _measurement_trait_phrase(plain[metric.end():after_end])
        # Both 'T1D (H4=...)' and 'H4=... for T1D' are explicit assignments.
        trailing = re.match(r"\s*(?:for|with|in)\s+", after, re.IGNORECASE)
        mention = trait_pattern.match(after, trailing.end()) if trailing else None
        if mention is None:
            mentions = tuple(trait_pattern.finditer(before))
            mention = mentions[-1] if mentions else None
        previous_end = metric.end()
        if mention is None:
            # Qualitative/ambiguous prose is not converted into an invented pair.
            continue
        trait = _measurement_trait_key(mention.group(1))
        code_scope = after if trailing and trait_pattern.match(after, trailing.end()) else before
        reported_codes = tuple(sorted(set(re.findall(r"\bPheCode\s+(\d+(?:\.\d+)*)", code_scope, re.IGNORECASE))))
        allowed = tuple(
            h4 for h4, codes in groups.get((gene_id, trait), ())
            if not reported_codes or reported_codes == codes
        )
        reported = _reported_statistic(metric.group("number"), label=label)
        quantum = Decimal(1).scaleb(-max(2, -reported.as_tuple().exponent))
        with localcontext() as numeric_context:
            numeric_context.prec = max(28, -quantum.as_tuple().exponent + 2)
            if any(number.quantize(quantum) == reported for number in allowed):
                continue
        permitted = (
            ", ".join(dict.fromkeys(f"{value:.4g}" for value in allowed)) or "none retrieved"
        )
        raise StructuredOutputError(
            f"{label} mismatches a colocalization gene/trait/H4 pair: "
            f"{gene_mention.group(1)}, {trait}, H4={reported}. "
            f"Permitted H4 values for this gene and trait: {permitted}."
        )


@memoized_report_derivation
def _reported_rank_aliases(context: HybridFinalContext) -> tuple[dict[str, int], dict[str, str]]:
    ranking = _final_ranking_context(context.result)
    ranks = {row["gene_id"]: row["rank"] for row in ranking
             if isinstance(row.get("gene_id"), str) and row["gene_id"]
             and type(row.get("rank")) is int and row["rank"] > 0}
    if not ranks or len(ranks) != len(ranking) or sorted(ranks.values()) != list(range(1, len(ranks) + 1)):
        return {}, {}
    # Statements about tied candidates are checked against their shared rank.
    ranks = {gene: row.get("shared_rank", ranks[gene]) for gene, row in
             ((row["gene_id"], row) for row in ranking if row.get("gene_id") in ranks)}
    # Only this ranking check rebuilds alias ownership. A symbol shared by two
    # evidence genes cannot borrow whichever gene happened to occur last.
    owners: dict[str, set[str]] = {}
    for row in _runtime_evidence_rows(context.result):
        gene_id = row.get("gene_id")
        if not isinstance(gene_id, str) or not gene_id:
            continue
        for term in _runtime_record_terms(row):
            if re.fullmatch(r"[A-Z][A-Z0-9-]{1,30}", term):
                owners.setdefault(term.casefold(), set()).add(gene_id)
    aliases = {alias: next(iter(genes)) for alias, genes in owners.items()
               if len(genes) == 1 and next(iter(genes)) in ranks}
    aliases.update({gene.casefold(): gene for gene in ranks})
    return ranks, aliases


def _validate_explicit_gene_ranks(text: str, *, context: HybridFinalContext, label: str) -> None:
    from .ranking_order import explicit_ranking_order_issues, unavailable_ranking_issues

    ranks, aliases = _reported_rank_aliases(context)
    if not ranks:
        decision = context.result.get("decision")
        if isinstance(decision, Mapping) and decision.get("status") == "ranking_unavailable":
            owners: dict[str, set[str]] = {}
            for row in _runtime_evidence_rows(context.result):
                gene_id = row.get("gene_id")
                if not isinstance(gene_id, str) or not gene_id:
                    continue
                for term in (gene_id, *_runtime_record_terms(row)):
                    if term == gene_id or re.fullmatch(r"[A-Z][A-Z0-9-]{1,30}", term):
                        owners.setdefault(term.casefold(), set()).add(gene_id)
            known = {alias: next(iter(genes)) for alias, genes in owners.items() if len(genes) == 1}
            errors = unavailable_ranking_issues(text, aliases=known)
            if errors:
                raise StructuredOutputError(f"{label}: model ranking is unavailable; " + "; ".join(errors))
        return
    ordinals = {word: index for index, word in enumerate(
        ("first", "second", "third", "fourth", "fifth", "sixth",
         "seventh", "eighth", "ninth", "tenth"),
        start=1,
    )}
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(alias) for alias in
                                         sorted(aliases, key=len, reverse=True))
        + r")(?:\s*\(ENSG\d+\))?\s*,?\s*(?:ranks?|ranked)\s+(?:as\s+)?(?:the\s+)?"
        r"(" + "|".join(ordinals) + r"|\d+)(?![A-Za-z0-9_])", re.IGNORECASE,
    )
    lookup = {key.casefold(): gene for key, gene in aliases.items()}
    for match in pattern.finditer(re.sub(r"[*`]", "", text)):
        declared = match.group(2).casefold()
        number = ordinals.get(declared) or int(declared)
        expected = ranks[lookup[match.group(1).casefold()]]
        if number != expected:
            raise StructuredOutputError(
                f"{label} changes a validated gene rank: {match.group(1)} is rank "
                f"{expected}, not {number}."
            )
    order_errors = explicit_ranking_order_issues(text, ranks=ranks, aliases=aliases)
    if order_errors:
        raise StructuredOutputError(f"{label}: " + "; ".join(order_errors))


def _renumber_reference_markers(text: str, reference_ids: Sequence[str]) -> str:
    """Map trusted deterministic ``[N]`` markers onto a filtered bibliography."""

    numbers: dict[str, int] = {}
    for index, reference_id in enumerate(reference_ids, start=1):
        match = re.fullmatch(r"reference_(\d+)", reference_id)
        if match:
            numbers[match.group(1)] = index

    def replace(match: re.Match[str]) -> str:
        number = numbers.get(match.group(1))
        if number is None:
            return ""
        return f"[[{number}]]{match.group(2)}" if match.group(2) else f"[{number}]"

    return _NUMBERED_CITATION.sub(replace, text).strip()


_INLINE_REFERENCE_TOKEN = re.compile(
    r"\{\{cite:(reference_\d+)\}\}(?:\(https://[^)\s]+\))?"
)
_NUMBERED_CITATION = re.compile(r"\[?\[(\d+)\]\]?(\(https://[^)\s]+\))?")
_SUMMARY_FACT_CITATION_TOKEN = re.compile(
    r"\{\{\s*cite_fact\s*:\s*([A-Za-z0-9][A-Za-z0-9_.:-]{0,199})\s*\}\}"
)


def _render_inline_reference_tokens(
    text: str,
    reference_numbers: Mapping[str, int],
    reference_index: Mapping[str, FinalReference],
) -> tuple[str, frozenset[str]]:
    """Render trusted reference placeholders exactly where each fact used them."""

    used: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        reference_id = match.group(1)
        number = reference_numbers.get(reference_id)
        if number is None:
            return ""
        used.add(reference_id)
        return _reference_marker(
            reference_id,
            number,
            reference_index,
        )

    return _INLINE_REFERENCE_TOKEN.sub(replace, text), frozenset(used)


def _render_summary_fact_citation_tokens(
    text: str,
    fact_index: Mapping[str, FinalEvidenceFact],
    reference_numbers: Mapping[str, int],
    reference_index: Mapping[str, FinalReference],
    *,
    source_observations_present: bool = False,
) -> tuple[str, frozenset[str]]:
    """Attach each Summary citation to the exact fact anchor selected by the model."""

    used: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        fact = fact_index.get(match.group(1))
        if fact is None:
            return ""
        if source_observations_present and _source_only_citation_clause(text[:match.start()]):
            # The uploaded observations have no external publication ownership.
            # A misplaced display anchor is removed, not a scientific hard stop;
            # the unchanged external fact remains cited in its Evidence section.
            return ""
        markers: list[str] = []
        for reference_id in fact.reference_ids:
            number = reference_numbers.get(reference_id)
            if number is None:
                continue
            used.add(reference_id)
            markers.append(
                _reference_marker(
                    reference_id,
                    number,
                    reference_index,
                )
            )
        return " ".join(markers)

    rendered = _SUMMARY_FACT_CITATION_TOKEN.sub(replace, text)
    return re.sub(r"[ \t]+(?=[,.;:])", "", rendered), frozenset(used)


def _source_only_citation_clause(prefix: str) -> bool:
    """Recognize a source-data sentence, not a mixed biological interpretation."""

    plain = _SUMMARY_FACT_CITATION_TOKEN.sub("", prefix)
    plain = re.sub(r"[*`]", "", plain).rstrip()
    # Decimal points and scientific exponents are not sentence boundaries.
    clauses = re.split(r"(?<=[.!?;])\s+|\n+", plain)
    clause = clauses[-1].rstrip(".?!; ") if clauses else ""
    source = re.search(
        r"\b(?:loaded|uploaded|source|current)\b.{0,90}\b(?:GWAS|dataset)\b",
        clause, re.IGNORECASE,
    )
    observation = re.search(
        r"\b(?:significan\w*|signal|threshold|P\s*[=<]|P[ -]?values?|selected)\b",
        clause, re.IGNORECASE,
    )
    interpretation = re.search(
        r"\b(?:genes?|candidates?|eQTL|sQTL|pQTL|QTL|colocali[sz]\w*|coloc|"
        r"expression|enhancers?|regulatory|mechanis\w*|encod\w*|protein\w*)\b",
        clause, re.IGNORECASE,
    )
    return bool(source and observation and not interpretation)


def _compact_repeated_source_context(
    summary: str,
    limitations: str,
    *,
    context: HybridFinalContext,
) -> tuple[str, str]:
    """Remove repeated source-scope boilerplate, never the source-signal opening."""

    if not context.source_observations or not context.source_trait.strip():
        return summary, limitations
    source = _normalize_phenotype_spellings(context.source_trait).casefold()

    def plain(text: str) -> str:
        return _normalize_phenotype_spellings(re.sub(r"[*`]", "", text)).casefold()

    sentence_separator = r"((?<=[.!?])\s+(?=[A-Z*]))"

    def compact(text: str, repeated, *, keep_opening: bool = False) -> str:
        # Keep the original separators. Rejoining sentences with spaces turns
        # complete model paragraphs into one block even when nothing is removed.
        pieces = re.split(sentence_separator, text)
        for index in range(0, len(pieces), 2):
            if (keep_opening and index == 0) or not repeated(pieces[index]):
                continue
            pieces[index] = ""
            # Remove one adjacent horizontal separator with the duplicate,
            # while leaving every original paragraph/line boundary intact.
            if index and "\n" not in pieces[index - 1] and "\r" not in pieces[index - 1]:
                pieces[index - 1] = ""
            elif index + 1 < len(pieces) and "\n" not in pieces[index + 1] and "\r" not in pieces[index + 1]:
                pieces[index + 1] = ""
        return "".join(pieces)

    sentences = re.split(sentence_separator, summary)[::2]
    if not sentences or source not in plain(sentences[0]):
        return summary, limitations
    if not _source_only_citation_clause(sentences[0]):
        # A mixed source/interpretation opening is legitimate; do not rewrite it.
        return summary, limitations
    def repeated_summary(sentence: str) -> bool:
        normalized = plain(sentence)
        return bool(
            source in normalized
            and _source_only_citation_clause(sentence)
            and not _scientific_numbers(normalized.replace(source, ""))
            and "{{cite:" not in sentence
            and "[" not in sentence
        )
    summary = compact(summary, repeated_summary, keep_opening=True)

    def repeated_limitation(sentence: str) -> bool:
        normalized = plain(sentence)
        return bool(
            source in normalized
            and _source_only_citation_clause(sentence)
            and re.search(
                r"\b(?:cannot|must not|should not)\s+(?:be\s+)?"
                r"(?:interpreted|treated|used|taken)\s+as\s+evidence\b|"
                r"\bquery phenotype differs from the source\b",
                normalized,
            )
            and "[" not in sentence
        )
    compacted_limitations = compact(limitations, repeated_limitation)
    # If the model supplied nothing but that caveat, retaining it is preferable
    # to silently creating an empty Limitations section or inventing replacement prose.
    return summary, compacted_limitations if compacted_limitations.strip() else limitations


def _reference_marker(
    reference_id: str,
    number: int,
    reference_index: Mapping[str, FinalReference],
) -> str:
    reference = reference_index.get(reference_id)
    url = _reference_url(reference.markdown) if reference is not None else None
    return f"[[{number}]]({url})" if url else f"[{number}]"


def _reference_url(markdown: str) -> str | None:
    linked = re.search(r"\]\((https://[^)\s]+)\)", markdown)
    if linked:
        return linked.group(1)
    bare = re.search(r"https://[^\s<>]+", markdown)
    if bare:
        return bare.group(0).rstrip(".,;:)")
    pmid = re.search(r"\bPMID\s*:\s*(\d{1,12})\b", markdown, re.IGNORECASE)
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid.group(1)}/"
    return None


def _link_numbered_citations(
    text: str,
    reference_ids: Sequence[str],
    reference_index: Mapping[str, FinalReference],
) -> str:
    """Make trusted deterministic numeric citations clickable when a URL exists."""

    def replace(match: re.Match[str]) -> str:
        if match.group(2):
            return f"[[{match.group(1)}]]{match.group(2)}"
        number = int(match.group(1))
        if number < 1 or number > len(reference_ids):
            return match.group(0)
        reference_id = reference_ids[number - 1]
        return _reference_marker(reference_id, number, reference_index)

    linked = _NUMBERED_CITATION.sub(replace, text)
    # Repeated papers are useful at distinct claims, but two adjacent anchors
    # can yield the same paper twice in one citation group.
    marker_pattern = r"(?:\[\[\d+\]\]\(https://[^)\s]+\)|\[\d+\])"
    group_pattern = re.compile(marker_pattern + r"(?:[ \t]*(?:,[ \t]*)?" + marker_pattern + r")+")
    return group_pattern.sub(
        lambda match: " ".join(dict.fromkeys(re.findall(marker_pattern, match.group(0)))),
        linked,
    )


def _canonical_scientific_number(value: str) -> Decimal:
    """Compare equivalent numeric spellings without weakening value grounding."""

    cleaned = str(value).replace(",", "").casefold()
    try:
        number = Decimal(cleaned)
    except InvalidOperation as exc:  # The caller supplies regex-matched numbers only.
        raise StructuredOutputError(f"invalid numeric token: {value}") from exc
    # Decimal equality already equates scientific spellings. normalize() applies
    # the current 28-digit arithmetic context, which can round unsupported values
    # into allowed values or overflow on a model-supplied exponent.
    return Decimal(0) if number.is_zero() else number


def _normalize_phenotype_spellings(text: str) -> str:
    """Recognize narrow established phenotype aliases without conflating diagnoses."""

    for pattern, canonical in _PHENOTYPE_SPELLINGS:
        text = pattern.sub(canonical, text)
    return text


def _scientific_numbers(text: str) -> set[Decimal]:
    """Treat typographic powers as one number, preserving the mantissa and exponent."""

    def exponent_form(match: re.Match[str]) -> str:
        exponent = match.group("plain") or match.group("bare") or match.group("super").translate(_SUPERSCRIPT_DIGITS)
        exponent = re.sub(r"\s+", "", exponent).strip("{}")
        return f"{match.group('mantissa')}e{exponent}"

    normalized = _SCIENTIFIC_PRODUCT.sub(exponent_form, text.replace("−", "-"))
    return {
        _canonical_scientific_number(value)
        for value in _SCIENTIFIC_NUMBER.findall(normalized)
    }


def _explicit_gene_symbols(text: str) -> set[str]:
    """Find gene assertion slots, without classifying every method acronym as a gene.

    The exact identifiers in Evidence remain the primary contract. Free narrative is
    additionally checked for explicit candidate, gene-expression and prioritization
    assertions; generic experimental methods are not gene claims merely because they
    are capitalized.
    """

    # Match a complete symbol, never the capitalized prefix of a word such
    # as the O in "Gene Ontology". Single-letter symbols still need evidence.
    symbol = r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*(?![\w-])"
    symbols: set[str] = set()
    plain = re.sub(r"[*`]", "", text)
    prefix = (
        r"\b(?i:genes?|candidate\s+genes?|causal\s+genes?|prioritized\s+genes?|"
        r"expression\s+of|regulation\s+of|(?:eQTL|pQTL|sQTL)\s+for|"
        r"prioriti[sz](?:e|ed|ing)|rank(?:ed|ing)?)(?:\s*[:=]\s*|\s+)"
    )
    for match in re.finditer(
        prefix + rf"(?P<genes>{symbol}(?:(?:\s*,\s*|\s+(?i:and|or)\s+){symbol})*)",
        plain,
    ):
        symbols.update(re.findall(rf"\b{symbol}\b", match.group("genes")))
    predicates = (
        r"(?i:(?:is|was|remains|emerges\s+as|ranks\s+as)\s+"
        r"(?:(?:the|a|an)\s+)?(?:leading|top|strongest|prioriti[sz]ed|causal|candidate)\b|"
        r"(?:expression|eQTL|pQTL|sQTL|colocali[sz]ation)\b)"
    )
    for match in re.finditer(rf"\b(?P<gene>{symbol})\s+{predicates}", plain):
        symbols.add(match.group("gene"))
    for match in re.finditer(
        rf"\*\*(?P<gene>{symbol})\*\*\s+(?i:(?:(?:has|shows|receives|is|was)\s+)?"
        r"(?:(?:the|a|an|strongest|strong|most|direct|highest)\s+)*"
        r"(?:support|evidence|prioriti[sz]|rank|candidate))", text,
    ):
        symbols.add(match.group("gene"))
    for match in re.finditer(rf"\b(?P<gene>{symbol})\s*\(\s*ENSG\d+", plain):
        symbols.add(match.group("gene"))
    return symbols


def _validate_text_against_authority(
    text: str, *, authority: str, label: str, free_narrative: bool = False
) -> None:
    allowed_numbers = _scientific_numbers(authority)
    number_text = re.sub(r"(?m)^\s*(?:[-*]\s+)?\d+[.)]\s+", "", text)
    rendered_numbers = _scientific_numbers(number_text)
    unsupported_numbers = rendered_numbers.difference(allowed_numbers)
    if unsupported_numbers:
        raise StructuredOutputError(
            f"{label} contains unsupported numbers: "
            + ", ".join(str(item) for item in sorted(unsupported_numbers)[:10])
        )

    allowed_gene_ids = set(re.findall(r"\bENSG\d{6,}\b", authority, re.IGNORECASE))
    rendered_gene_ids = set(re.findall(r"\bENSG\d{6,}\b", text, re.IGNORECASE))
    if not {item.upper() for item in rendered_gene_ids}.issubset(
        {item.upper() for item in allowed_gene_ids}
    ):
        raise StructuredOutputError(f"{label} contains an unsupported Ensembl gene ID")

    uppercase_pattern = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\b")
    allowed_uppercase = set(uppercase_pattern.findall(authority)) | set(
        _GENERIC_UPPERCASE_TOKENS
    )
    rendered_uppercase = (
        _explicit_gene_symbols(text)
        if free_narrative
        else {item for item in uppercase_pattern.findall(text) if len(item) >= 3}
    )
    unsupported_uppercase = rendered_uppercase.difference(allowed_uppercase)
    unsupported_uppercase = {
        item
        for item in unsupported_uppercase
        if not re.search(
            rf"(?<![A-Za-z0-9]){re.escape(item)}(?![A-Za-z0-9])", authority, re.IGNORECASE
        )
    }
    if unsupported_uppercase:
        raise StructuredOutputError(
            f"{label} contains unsupported gene-like or scientific identifiers: "
            + ", ".join(sorted(unsupported_uppercase)[:10])
        )

    if free_narrative:
        variant_pattern = re.compile(
            r"\b(?:chr)?(?:[0-9]+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+\b",
            re.IGNORECASE,
        )
        def normalize_variant(value: str) -> str:
            return re.sub(r"^chr", "", value, flags=re.IGNORECASE).upper().replace(":", "_")

        allowed_variants = {normalize_variant(item) for item in variant_pattern.findall(authority)}
        asserted_variants = {normalize_variant(item) for item in variant_pattern.findall(text)}
        if asserted_variants.difference(allowed_variants):
            raise StructuredOutputError(f"{label} contains an unsupported variant ID")

    if re.search(r"https?://|\bPMID\s*:\s*\d+", text, re.IGNORECASE):
        raise StructuredOutputError(
            f"{label} must not generate a URL or bibliography entry; use reference IDs"
        )


def _validate_source_gwas_summary(text: str, *, context: HybridFinalContext) -> None:
    if not context.source_observations or context.intent.casefold() not in {
        "describe_variants",
        "gene_associations",
        "prioritize_genes",
    }:
        return
    # Sentence order is editorial. Validate actual source assertions wherever
    # they occur; do not reject a grounded report for starting with a gene.
    normalized = _normalize_phenotype_spellings(re.sub(r"[*`]", "", " ".join(text.strip().split()))).casefold()
    source_trait = _normalize_phenotype_spellings(
        " ".join(context.source_trait.strip().split())
    ).casefold()
    generic_traits = {
        "",
        "user-selected gwas locus",
        "the loaded phenotype",
        "loaded gwas phenotype",
    }
    for match in re.finditer(r"\b(?:loaded|uploaded|source|current)\s+([^.;,]{1,140}?)\s+gwas\b", normalized):
        stated_trait = match.group(1).strip()
        if source_trait not in generic_traits and stated_trait not in {"source", "current", "loaded", "uploaded"} and source_trait not in stated_trait:
            raise StructuredOutputError("summary_markdown contradicts the actual loaded source GWAS phenotype")
    signal_check = next(
        (
            item
            for item in context.safety_review.facts.checks
            if item.check_id in {"gwas_signal", "current_gwas_signal"}
        ),
        None,
    )
    source_signal = summarize_source_gwas_observations(context.source_observations, phenotype=context.source_trait)
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        if not re.search(r"\b(?:loaded|uploaded|source|current)\b.{0,160}\b(?:gwas|dataset)\b", sentence):
            continue
        _validate_source_significance_counts(sentence, context)
        named_sources = [measurement for measurement in source_signal.measurements
                         if measurement.variant_id and measurement.variant_id.casefold() in sentence]
        supported_subset = (bool(named_sources) and all(measurement.genome_wide_significant is True for measurement in named_sources)
            or source_signal.genome_wide_significant_variants > 0 and bool(re.search(
                r"\b\d+\s*(?:of|out of|/)\s*\d+\s+(?:selected\s+)?(?:variants?|snps?)\b|\bat least one\s+(?:selected\s+)?(?:variant|snp)\b",
                sentence)))
        if (signal_check is not None and signal_check.outcome is not SafetyOutcome.PASS
                and not supported_subset
                and re.search(r"\b(?:is|are|was|were|remain|remains)\s+(?:genome-wide\s+)?significant\b", sentence)
                and not re.search(r"\b(?:no|none|not|without|does not|did not)\b", sentence)):
            raise StructuredOutputError("summary_markdown overstates the source-GWAS significant signal")


def _validate_source_significance_counts(opening: str, context: HybridFinalContext) -> None:
    """Compare explicit source-GWAS count claims with source observations only."""

    from .report_source_counts import source_significance_count_corrections
    if source_significance_count_corrections(opening, context):
        raise StructuredOutputError("summary_markdown contains incorrect source-GWAS significance counts")

    source = summarize_source_gwas_observations(
        context.source_observations, phenotype=context.source_trait
    )
    sentence = re.split(r"\.\s+", re.sub(r"[*`]", "", opening), maxsplit=1)[0]
    fraction = re.search(
        r"\b(\d+)\s*(?:of|out of|/)\s*(\d+)\s+"
        r"(?:selected\s+)?(?:variants?(?:\(s\))?|snps?)\b[^.;]{0,90}"
        r"(?:significan\w*|threshold)",
        sentence,
    )
    if fraction and (
        int(fraction.group(1)) != source.genome_wide_significant_variants
        or int(fraction.group(2)) != source.selected_variants
    ):
        raise StructuredOutputError(
            "summary_markdown contains incorrect source-GWAS significance counts"
        )
    all_significant = re.search(
        r"\ball\s+(?:\d+\s+)?(?:selected\s+)?(?:variants?|snps?)\s+"
        r"(?:are|were|remain|reach|reached|met)\s+(?:(?:the|genome-wide|significance)\s+)*"
        r"(?:significant|threshold)",
        sentence,
    )
    if all_significant and source.all_measured_variants_are_significant is not True:
        raise StructuredOutputError(
            "summary_markdown overstates the source-GWAS significant signal"
        )


def _string_array(
    value: Any, *, label: str, limit: int, deduplicate: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise StructuredOutputError(f"{label} must be an array of strings")
    if len(value) > limit:
        raise StructuredOutputError(f"{label} exceeds {limit} items")
    parsed = tuple(
        _bounded_text(item, label=f"{label} item", limit=2_000) for item in value
    )
    if deduplicate:
        return tuple(dict.fromkeys(parsed))
    if len(parsed) != len(set(parsed)):
        raise StructuredOutputError(f"{label} must not contain duplicates")
    return parsed


def _validate_identifier(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", value):
        raise ValueError(f"{label} is invalid: {value!r}")


def _clean_identifier(value: Any) -> str:
    text = str(value).strip().casefold().replace(" ", "_")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,199}", text):
        raise ValueError(f"invalid module identifier: {value!r}")
    return text


def _require_unique(values, *, label: str) -> None:
    rows = tuple(values)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{label} must be unique")


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    raise ValueError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value, path="$"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _as_markdown_bullet(markdown: str) -> str:
    lines = markdown.strip().splitlines()
    if len(lines) == 1 and re.match(r"^[-*]\s+", lines[0]):
        return lines[0]
    if len(lines) == 1:
        return "- " + lines[0]
    return "- " + lines[0] + "\n" + "\n".join("  " + line for line in lines[1:])
