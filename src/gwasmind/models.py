from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from .configuration import parameter


class DataMode(StrEnum):
    FROZEN_SAFE = "frozen-safe"
    LIVE = "live"


class DecisionStatus(StrEnum):
    """Explicit terminal states; none of the abstentions imply gene novelty."""

    PRIORITIZED = "PRIORITIZED"
    EVIDENCE_RETRIEVED = "EVIDENCE_RETRIEVED"
    NO_GWAS_SIGNAL = "NO_GWAS_SIGNAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INVALID_INPUT = "INVALID_INPUT"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INCOMPLETE = "INCOMPLETE"

    # Backwards-compatible names for an earlier internal API.
    COMPLETED = "PRIORITIZED"
    ABSTAINED = "INSUFFICIENT_EVIDENCE"
    BLOCKED = "POLICY_BLOCKED"
    ERROR = "SERVICE_UNAVAILABLE"


class EvidenceKind(StrEnum):
    VARIANT_ANNOTATION = "variant_annotation"
    STUDY = "study"
    GWAS = "gwas"
    QTL = "qtl"
    COLOCALIZATION = "colocalization"
    REGULATORY = "regulatory"
    EXPRESSION = "expression"
    FUNCTION = "function"
    LITERATURE = "literature"
    L2G = "l2g"
    PATHWAY = "pathway"
    ONTOLOGY = "ontology"
    TARGET_DISEASE = "target_disease"
    DRUG = "drug"
    SAFETY = "safety"
    TRACTABILITY = "tractability"
    INTERACTION = "interaction"
    PHENOTYPE = "phenotype"
    GENE_SET = "gene_set"


class QTLStudyType(StrEnum):
    """Native Open Targets molecular-QTL ``studyType`` values we support.

    These values deliberately remain more specific than :class:`EvidenceKind`.
    ``EvidenceKind.QTL`` is the broad evidence dimension, while this enum is
    the auditable upstream subtype used to constrain a retrieval.
    """

    EQTL = "eqtl"
    PQTL = "pqtl"
    SQTL = "sqtl"
    TUQTL = "tuqtl"
    SC_EQTL = "sceqtl"
    SC_PQTL = "scpqtl"
    SC_SQTL = "scsqtl"
    SC_TUQTL = "sctuqtl"


QTL_STUDY_TYPES: tuple[str, ...] = tuple(item.value for item in QTLStudyType)
SINGLE_CELL_QTL_STUDY_TYPES: tuple[str, ...] = (
    QTLStudyType.SC_EQTL.value,
    QTLStudyType.SC_PQTL.value,
    QTLStudyType.SC_SQTL.value,
    QTLStudyType.SC_TUQTL.value,
)


@dataclass(frozen=True, slots=True)
class VariantObservation:
    variant_id: str
    p_value: float | None = None
    chromosome: str | None = None
    position: int | None = None

    @property
    def is_genome_wide_significant(self) -> bool | None:
        if self.p_value is None:
            return None
        return self.p_value <= parameter("analysis.gwas_significance_p")


def analysis_variant_ids(query: LocusQuery) -> tuple[str, ...]:
    """Effective retrieval identities; source statistics always use variants."""
    rows = query.analysis_variants if query.task == "prioritize_genes" and query.analysis_variants else query.variants
    return tuple(dict.fromkeys(row.variant_id for row in rows))


@dataclass(frozen=True, slots=True)
class LocusQuery:
    locus_id: str
    trait: str
    variants: tuple[VariantObservation, ...]
    source_trait: str | None = None
    candidate_genes: tuple[str, ...] = ()
    task: str = "prioritize_genes"
    ancestry: str | None = None
    relevant_tissues: tuple[str, ...] = ()
    genome_build: str | None = None
    ld_reference: str | None = None
    benchmark: bool = False
    significance_verified: bool | None = None
    significance_source: str | None = None
    study_ids: tuple[str, ...] = ()
    trait_ids: tuple[str, ...] = ()
    qtl_types: tuple[str, ...] = ()
    requested_evidence_kinds: tuple[str, ...] = ()
    # Source observations above remain the sole source-GWAS statistics. This
    # private scope adds verified LD proxies only for gene prioritization.
    analysis_variants: tuple[VariantObservation, ...] = ()
    ld_expansion: dict[str, Any] | None = None
    biological_context: dict[str, Any] | None = None

    def validate(self) -> None:
        from .analysis_scope import validate_analysis_scope

        validate_analysis_scope(self)
        if self.biological_context is not None:
            from .biological_context import biological_context_from_dict
            context = biological_context_from_dict(self.biological_context, phenotype=self.trait)
            if tuple(self.relevant_tissues) != context.relevant_terms:
                raise ValueError("relevant_tissues must match the fixed biological_context")
        for name in (
            "variants", "candidate_genes", "study_ids", "trait_ids", "qtl_types",
            "requested_evidence_kinds", "relevant_tissues",
        ):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ValueError(f"{name} must be an array")
            if name != "variants" and any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise ValueError(f"{name} must contain non-empty strings")
        for name in ("ancestry", "genome_build", "ld_reference", "significance_source"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
        if not isinstance(self.benchmark, bool):
            raise ValueError("benchmark must be boolean")
        if self.significance_verified is not None and not isinstance(
            self.significance_verified, bool
        ):
            raise ValueError("significance_verified must be boolean or None")
        if not isinstance(self.locus_id, str) or not self.locus_id.strip():
            raise ValueError("locus_id is required")
        if not isinstance(self.trait, str) or not self.trait.strip():
            raise ValueError("trait is required")
        if len(self.locus_id) > 256 or len(self.trait) > 1_000:
            raise ValueError("locus_id or trait exceeds the accepted length")
        if self.source_trait is not None and (
            not isinstance(self.source_trait, str)
            or not self.source_trait.strip()
            or len(self.source_trait) > 1_000
        ):
            raise ValueError("source_trait must be a non-empty string up to 1,000 characters")
        if not isinstance(self.task, str) or self.task not in {
            "describe_variants",
            "gene_associations",
            "prioritize_genes",
            "target_assessment",
            "disease_associations",
            "drug_repurposing",
        }:
            raise ValueError(f"unsupported analysis task: {self.task}")
        if not self.variants:
            raise ValueError("at least one variant is required")
        if len(self.variants) > parameter("execution.max_variants_per_query"):
            raise ValueError(f"a locus query accepts at most {parameter('execution.max_variants_per_query')} variants")
        if len(self.candidate_genes) > parameter("ranking.max_discovered_targets"):
            raise ValueError(f"a locus query accepts at most {parameter('ranking.max_discovered_targets')} candidate genes")
        if len(self.study_ids) > 100 or len(self.trait_ids) > 100:
            raise ValueError("a locus query accepts at most 100 study or trait IDs")
        if len(self.qtl_types) > len(QTL_STUDY_TYPES):
            raise ValueError("a locus query contains too many QTL study types")
        if len(self.requested_evidence_kinds) > len(EvidenceKind):
            raise ValueError("a locus query contains too many requested evidence kinds")
        if any(not isinstance(item, str) for item in self.qtl_types):
            raise ValueError("qtl_types must contain strings")
        if any(not isinstance(item, str) for item in self.requested_evidence_kinds):
            raise ValueError("requested_evidence_kinds must contain strings")
        if len(self.relevant_tissues) > 100:
            raise ValueError("a locus query accepts at most 100 relevant tissues")
        if any(not isinstance(variant, VariantObservation) for variant in self.variants):
            raise ValueError("variants must contain VariantObservation values")
        ids = [variant.variant_id for variant in self.variants]
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("variant_id cannot be empty")
        if len(ids) != len(set(ids)):
            raise ValueError("variant_id values must be unique within a locus")
        for variant in self.variants:
            if variant.p_value is not None and (
                isinstance(variant.p_value, bool)
                or not isinstance(variant.p_value, (int, float))
                or not 0 <= variant.p_value <= 1
                or not math.isfinite(variant.p_value)
            ):
                raise ValueError(f"p_value must be between 0 and 1: {variant.variant_id}")
            if variant.position is not None and (
                isinstance(variant.position, bool)
                or not isinstance(variant.position, int)
                or variant.position <= 0
            ):
                raise ValueError(f"position must be a positive integer: {variant.variant_id}")
            coordinate_id = re.fullmatch(
                r"(?:chr)?([0-9]+|X|Y|MT)[_:]([0-9]+)[_:][ACGTN-]+[_:][ACGTN-]+",
                variant.variant_id, re.IGNORECASE,
            )
            if coordinate_id is not None:
                chromosome, position = coordinate_id.groups()
                if int(position) < 1:
                    raise ValueError(f"variant coordinate must be positive: {variant.variant_id}")
                if variant.position is not None and variant.position != int(position):
                    raise ValueError(f"position conflicts with variant_id: {variant.variant_id}")
                if variant.chromosome is not None and (
                    str(variant.chromosome).upper().removeprefix("CHR") != chromosome.upper()
                ):
                    raise ValueError(f"chromosome conflicts with variant_id: {variant.variant_id}")
        if len(self.candidate_genes) != len(set(self.candidate_genes)):
            raise ValueError("candidate_genes must be unique within a locus")
        if any(not gene.strip() for gene in self.candidate_genes):
            raise ValueError("candidate_genes cannot contain an empty value")
        if self.genome_build and self.genome_build.casefold() not in {
            "grch37",
            "grch38",
            "hg19",
            "hg38",
        }:
            raise ValueError(f"unsupported genome build: {self.genome_build}")
        if self.benchmark and not self.candidate_genes:
            raise ValueError("benchmark loci require an explicit candidate-gene universe")
        if self.significance_verified is not None and not self.significance_source:
            raise ValueError("verified significance requires a provenance source")
        if any(not item.strip() for item in (*self.study_ids, *self.trait_ids)):
            raise ValueError("study_ids and trait_ids cannot contain empty values")
        if len(self.qtl_types) != len(set(self.qtl_types)):
            raise ValueError("qtl_types must be unique within a locus")
        unsupported_qtl_types = tuple(
            item for item in self.qtl_types if item not in QTL_STUDY_TYPES
        )
        if unsupported_qtl_types:
            raise ValueError(
                "unsupported QTL study type(s): "
                + ", ".join(unsupported_qtl_types)
            )
        if len(self.requested_evidence_kinds) != len(set(self.requested_evidence_kinds)):
            raise ValueError("requested_evidence_kinds must be unique within a locus")
        supported_evidence_kinds = {kind.value for kind in EvidenceKind}
        unsupported_evidence_kinds = tuple(
            item
            for item in self.requested_evidence_kinds
            if item not in supported_evidence_kinds
        )
        if unsupported_evidence_kinds:
            raise ValueError(
                "unsupported requested evidence kind(s): "
                + ", ".join(unsupported_evidence_kinds)
            )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    kind: EvidenceKind
    dataset: str
    source: str
    data_release: str
    payload: dict[str, Any]
    variant_id: str | None = None
    gene_id: str | None = None
    study_id: str | None = None
    pmid: str | None = None
    credible_set_id: str | None = None
    parent_variant_ids: tuple[str, ...] = ()
    parent_evidence_ids: tuple[str, ...] = ()
    tissue: str | None = None
    supports: bool | None = None
    provenance_complete: bool = True
    evidence_date: date | None = None
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        result["evidence_date"] = self.evidence_date.isoformat() if self.evidence_date else None
        result["retrieved_at"] = self.retrieved_at.isoformat()
        return result


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    """A model's explicit decision about one member of the complete candidate pool."""

    gene_id: str
    status: str
    reason: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RankedGene:
    gene_id: str
    rank: int
    evidence_ids: tuple[str, ...]
    rationale: str
    limitations: tuple[str, ...] = ()
    # The model found no supplied observation separating this gene from the previous one.
    tied_with_previous: bool = False


def shared_ranks(rows: Sequence[Any]) -> dict[str, tuple[int, bool]]:
    """Competition ranks (1, 1, 1, 4) and tie membership from ordered ranked rows or dicts."""
    groups: list[list[str]] = []
    for row in rows:
        gene = row.get("gene_id") if isinstance(row, Mapping) else getattr(row, "gene_id", None)
        tied = row.get("tied_with_previous") if isinstance(row, Mapping) else getattr(row, "tied_with_previous", False)
        if not isinstance(gene, str):
            continue
        if tied is True and groups:
            groups[-1].append(gene)
        else:
            groups.append([gene])
    result, position = {}, 1
    for group in groups:
        for gene in group:
            result[gene] = (position, len(group) > 1)
        position += len(group)
    return result


@dataclass(frozen=True, slots=True)
class SufficiencyReport:
    valid_input: bool
    significant_signal: bool | None
    evidence_sources: int
    evidence_records: int
    genes_with_evidence: int
    conflicts: int
    policy_blocks: int
    decision_evidence_records: int = 0
    locus_linked_records: int = 0
    locus_linked_genes: int = 0
    relevant_tissue_coverage: bool | None = None
    evidence_dimensions: tuple[str, ...] = ()
    missing_relevant_tissues: tuple[str, ...] = ()
    region_gene_context_status: str | None = None
    nearby_annotated_gene_count: int | None = None
    region_gene_context_source: str | None = None
    warnings: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    direct_evidence_sources: int = 0
    context_evidence_scope: str = "decision_evidence"

    @property
    def passed(self) -> bool:
        return not self.reasons


@dataclass(frozen=True, slots=True)
class AgentResult:
    locus_id: str
    status: DecisionStatus
    ranked_genes: tuple[RankedGene, ...]
    evidence: tuple[EvidenceRecord, ...]
    sufficiency: SufficiencyReport
    summary: str
    run_id: str
    model_id: str = "unknown"
    data_release: str = "unknown"
    data_mode: str = "unknown"
    limitations: tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    ld_expansion: dict[str, Any] | None = None
    gate2_review: dict[str, Any] | None = None
    candidate_decisions: tuple[CandidateDecision, ...] = ()
    sampling_parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **({"sampling_parameters": self.sampling_parameters} if self.sampling_parameters is not None else {}),
            **({"candidate_decisions": [asdict(item) for item in self.candidate_decisions]}
               if self.candidate_decisions else {}),
            **({"gate2_review": self.gate2_review} if self.gate2_review is not None else {}),
            **({"ld_expansion": self.ld_expansion} if self.ld_expansion is not None else {}),
            "locus_id": self.locus_id,
            "status": self.status.value,
            "ranked_genes": [asdict(item) for item in self.ranked_genes],
            "evidence": [item.to_dict() for item in self.evidence],
            "sufficiency": asdict(self.sufficiency),
            "summary": self.summary,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "data_release": self.data_release,
            "data_mode": self.data_mode,
            "limitations": list(self.limitations),
            "artifacts": dict(self.artifacts),
        }

    def to_compact_dict(self) -> dict[str, Any]:
        """User-facing view; full payloads remain in the evidence artifacts."""

        return {
            **({"sampling_parameters": self.sampling_parameters} if self.sampling_parameters is not None else {}),
            **({"candidate_decisions": [asdict(item) for item in self.candidate_decisions]}
               if self.candidate_decisions else {}),
            **({"gate2_review": self.gate2_review} if self.gate2_review is not None else {}),
            **({"ld_expansion": self.ld_expansion} if self.ld_expansion is not None else {}),
            "locus_id": self.locus_id,
            "status": self.status.value,
            "ranked_genes": [asdict(item) for item in self.ranked_genes],
            "evidence_ledger": [
                {
                    "evidence_id": item.evidence_id,
                    "kind": item.kind.value,
                    "dataset": item.dataset,
                    "source": item.source,
                    "data_release": item.data_release,
                    "variant_id": item.variant_id,
                    "gene_id": item.gene_id,
                    "study_id": item.study_id,
                    "pmid": item.pmid,
                    "credible_set_id": item.credible_set_id,
                    "tissue": item.tissue,
                }
                for item in self.evidence
            ],
            "sufficiency": asdict(self.sufficiency),
            "summary": self.summary,
            "limitations": list(self.limitations),
            "run_id": self.run_id,
            "model_id": self.model_id,
            "data_release": self.data_release,
            "data_mode": self.data_mode,
            "artifacts": dict(self.artifacts),
        }
