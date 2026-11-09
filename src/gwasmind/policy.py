from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date

from .models import DataMode, EvidenceKind, EvidenceRecord, LocusQuery


class PolicyViolation(RuntimeError):
    """Raised when a benchmark query attempts to access prohibited information."""


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str
    record: EvidenceRecord | None = None


@dataclass(frozen=True, slots=True)
class LeakagePolicy:
    """Deterministic query and evidence firewall for benchmark runs.

    A local snapshot makes leakage control possible, but does not provide it by
    itself. This policy isolates the evaluation labels from answer-bearing Open
    Targets records, source studies, coding signals, and post-cutoff evidence.
    """

    mode: DataMode
    release: str
    cutoff_date: date | None = None
    allowed_kinds: frozenset[EvidenceKind] = field(
        default_factory=lambda: frozenset(
            {
                EvidenceKind.GWAS,
                EvidenceKind.QTL,
                EvidenceKind.COLOCALIZATION,
                EvidenceKind.REGULATORY,
                EvidenceKind.EXPRESSION,
                EvidenceKind.FUNCTION,
                EvidenceKind.LITERATURE,
                EvidenceKind.VARIANT_ANNOTATION,
                EvidenceKind.STUDY,
                EvidenceKind.PATHWAY,
                EvidenceKind.ONTOLOGY,
                EvidenceKind.SAFETY,
                EvidenceKind.TRACTABILITY,
                EvidenceKind.INTERACTION,
                EvidenceKind.PHENOTYPE,
                EvidenceKind.GENE_SET,
            }
        )
    )
    forbidden_datasets: frozenset[str] = field(
        default_factory=lambda: frozenset({"l2g_prediction"})
    )
    forbidden_study_ids: frozenset[str] = field(default_factory=frozenset)
    forbidden_pmids: frozenset[str] = field(default_factory=frozenset)
    forbidden_variant_ids: frozenset[str] = field(default_factory=frozenset)
    forbidden_credible_set_ids: frozenset[str] = field(default_factory=frozenset)
    forbidden_evidence_ids: frozenset[str] = field(default_factory=frozenset)
    require_complete_provenance: bool = True
    require_release_match: bool = True
    block_on_any_denied_record: bool = True
    denied_payload_fields: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "l2g_score",
                "l2gScore",
                "predicted_causal_gene",
                "benchmark_label",
                "gold_gene",
            }
        )
    )

    def validate_query(self, query: LocusQuery, requested: set[EvidenceKind]) -> None:
        query.validate()
        if query.benchmark and self.mode is DataMode.LIVE:
            raise PolicyViolation("benchmark loci cannot use live Open Targets")
        if (
            self.mode is DataMode.FROZEN_SAFE
            and EvidenceKind.L2G in requested
            and (query.benchmark or "l2g_prediction" in self.forbidden_datasets)
        ):
            raise PolicyViolation("L2G is disabled in frozen-safe agent runs")
        unavailable = requested.difference(self.allowed_kinds)
        if unavailable:
            names = ", ".join(sorted(kind.value for kind in unavailable))
            raise PolicyViolation(f"evidence kinds are not permitted: {names}")
        denied_variants = {_variant_identity(value) for value in self.forbidden_variant_ids}
        overlap = {
            variant.variant_id for variant in query.variants
            if _variant_identity(variant.variant_id) in denied_variants
        }
        if overlap:
            raise PolicyViolation(
                "query contains a coding-label variant or LD proxy: " + ", ".join(sorted(overlap))
            )

    def evaluate_record(self, record: EvidenceRecord) -> PolicyDecision:
        if self.require_complete_provenance and not record.provenance_complete:
            return PolicyDecision(False, "record has incomplete provenance")
        if self.require_release_match and record.data_release != self.release:
            return PolicyDecision(
                False,
                f"record release {record.data_release!r} does not match frozen release {self.release!r}",
            )
        # Semantic wrappers can expose multiple study/variant/publication
        # lineages, especially both members of a colocalisation pair. Every
        # explicitly typed lineage is subject to the same existing embargo.
        for names, containers, denied, label in (
            (
                {"studyId", "study_id", "leftStudyId", "rightStudyId", "otherStudyId",
                 "qtlStudyId", "enhancerStudyId"},
                {"study"}, self.forbidden_study_ids, "benchmark study denied",
            ),
            (
                {"pmid", "pubmedId", "pubmed_id", "leftPubmedId", "rightPubmedId",
                 "otherPubmedId", "qtlPubmedId"},
                set(), self.forbidden_pmids, "benchmark publication denied",
            ),
            (
                {"credibleSetId", "credible_set_id", "studyLocusId", "study_locus_id",
                 "leftStudyLocusId", "rightStudyLocusId", "qtlStudyLocusId",
                 "queriedStudyLocusId", "otherStudyLocusId"},
                set(), self.forbidden_credible_set_ids, "benchmark credible set denied",
            ),
            (
                {"evidence_id", "source_evidence_id", "parent_evidence_ids"},
                set(), self.forbidden_evidence_ids, "derives from embargoed evidence",
            ),
        ):
            if not denied:
                continue
            overlap = _payload_identifiers(record.payload, names, containers).intersection(denied)
            if overlap:
                return PolicyDecision(False, label + ": " + ", ".join(sorted(overlap)))
        if self.forbidden_variant_ids:
            variants = _payload_identifiers(
                record.payload,
                {"variantId", "variant_id", "queryVariantId", "leadVariantId", "overlapVariantId",
                 "parent_variant_ids", "lead_variant_id", "qtl_member_variant_id"},
                {"variant"},
            )
            if record.variant_id:
                variants.add(record.variant_id)
            variants.update(record.parent_variant_ids)
            forbidden_variants = {_variant_identity(value) for value in self.forbidden_variant_ids}
            variant_overlap = {
                value for value in variants if _variant_identity(value) in forbidden_variants
            }
            if variant_overlap:
                return PolicyDecision(
                    False,
                    "record derives from a coding signal or LD proxy denied: "
                    + ", ".join(sorted(variant_overlap)),
                )
        if record.kind not in self.allowed_kinds:
            return PolicyDecision(False, f"kind denied: {record.kind.value}")
        if record.dataset in self.forbidden_datasets:
            return PolicyDecision(False, f"dataset denied: {record.dataset}")
        if record.study_id and record.study_id in self.forbidden_study_ids:
            return PolicyDecision(False, f"benchmark study denied: {record.study_id}")
        if record.pmid and record.pmid in self.forbidden_pmids:
            return PolicyDecision(False, f"benchmark publication denied: {record.pmid}")
        if record.variant_id and record.variant_id in self.forbidden_variant_ids:
            return PolicyDecision(False, f"coding signal or LD proxy denied: {record.variant_id}")
        parent_variant_overlap = set(record.parent_variant_ids).intersection(
            self.forbidden_variant_ids
        )
        if parent_variant_overlap:
            return PolicyDecision(
                False,
                "record derives from a coding signal or LD proxy: "
                + ", ".join(sorted(parent_variant_overlap)),
            )
        if self.forbidden_credible_set_ids:
            credible_set_id = record.credible_set_id or _first_payload_value(
                record.payload, {"credibleSetId", "credible_set_id", "studyLocusId"}
            )
            if credible_set_id and str(credible_set_id) in self.forbidden_credible_set_ids:
                return PolicyDecision(False, f"benchmark credible set denied: {credible_set_id}")
        if record.evidence_id in self.forbidden_evidence_ids:
            return PolicyDecision(False, f"benchmark evidence denied: {record.evidence_id}")
        parent_evidence_overlap = set(record.parent_evidence_ids).intersection(
            self.forbidden_evidence_ids
        )
        if parent_evidence_overlap:
            return PolicyDecision(
                False,
                "record derives from embargoed evidence: "
                + ", ".join(sorted(parent_evidence_overlap)),
            )
        if self.cutoff_date and record.evidence_date and record.evidence_date > self.cutoff_date:
            return PolicyDecision(False, f"post-cutoff evidence denied: {record.evidence_date}")
        if self.denied_payload_fields:
            leaked_fields = self.denied_payload_fields.intersection(_payload_keys(record.payload))
            if leaked_fields:
                return PolicyDecision(
                    False, "answer-bearing fields denied: " + ", ".join(sorted(leaked_fields))
                )
        return PolicyDecision(True, "allowed", record)

    def filter_records(
        self, records: Iterable[EvidenceRecord]
    ) -> tuple[list[EvidenceRecord], list[PolicyDecision]]:
        allowed: list[EvidenceRecord] = []
        decisions: list[PolicyDecision] = []
        for record in records:
            decision = self.evaluate_record(record)
            decisions.append(decision)
            if decision.allowed:
                allowed.append(record)
        return allowed, decisions


def live_policy(release: str = "live") -> LeakagePolicy:
    """Permissive policy for exploratory use; never use for primary benchmarks."""

    return LeakagePolicy(
        mode=DataMode.LIVE,
        release=release,
        allowed_kinds=frozenset(EvidenceKind),
        forbidden_datasets=frozenset(),
        denied_payload_fields=frozenset(),
        require_complete_provenance=False,
        require_release_match=False,
        block_on_any_denied_record=False,
    )


def frozen_exploratory_policy(release: str) -> LeakagePolicy:
    """Version-pinned but non-benchmark policy for local exploratory analysis."""

    return LeakagePolicy(
        mode=DataMode.FROZEN_SAFE,
        release=release,
        allowed_kinds=frozenset(EvidenceKind),
        forbidden_datasets=frozenset(),
        denied_payload_fields=frozenset(),
        require_complete_provenance=True,
        require_release_match=True,
        block_on_any_denied_record=True,
    )


def _payload_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_payload_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            keys.update(_payload_keys(nested))
    return keys


def _first_payload_value(value: object, names: set[str]) -> object | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in names and nested is not None:
                return nested
            result = _first_payload_value(nested, names)
            if result is not None:
                return result
    elif isinstance(value, (list, tuple)):
        for nested in value:
            result = _first_payload_value(nested, names)
            if result is not None:
                return result
    return None


def _payload_identifiers(
    value: object, names: set[str], containers: set[str]
) -> set[str]:
    identifiers: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in names:
                items = nested if isinstance(nested, (list, tuple)) else (nested,)
                identifiers.update(
                    str(item).strip() for item in items
                    if isinstance(item, (str, int)) and not isinstance(item, bool)
                )
            if key in containers and isinstance(nested, Mapping):
                identifier = nested.get("id")
                if isinstance(identifier, str):
                    identifiers.add(identifier.strip())
            identifiers.update(_payload_identifiers(nested, names, containers))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            identifiers.update(_payload_identifiers(nested, names, containers))
    return identifiers


def _variant_identity(value: str) -> str:
    text = value.strip()
    canonical = text.upper().removeprefix("CHR").replace(":", "_")
    if re.fullmatch(r"(?:[0-9]+|X|Y|MT)_[0-9]+_[ACGTN-]+_[ACGTN-]+", canonical):
        return canonical
    return text
