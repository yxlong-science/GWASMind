from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

GRCH38_VARIANT_ID = re.compile(r"^(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)_[1-9][0-9]*_[ACGT]+_[ACGT]+$")
ENSEMBL_GENE_ID = re.compile(r"^ENSG[0-9]{11}$")


class ChannelStatus(StrEnum):
    """Per-gene evidence channel state; ``not_observed`` never means biological absence."""

    OBSERVED = "observed"
    NOT_OBSERVED = "not_observed"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_REQUESTED = "not_requested"


class VariantSource(StrEnum):
    LEAD = "lead"
    SELECTED = "selected"
    CREDIBLE_SET = "credible_set"
    LD_PROXY = "ld_proxy"


class WeightBasis(StrEnum):
    PIP = "pip"
    LD_APPROXIMATION = "ld_approximation"


class LDStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    COMPLETE = "complete"
    PARTIAL = "partial"
    ANCHOR_NOT_IN_REFERENCE = "anchor_not_in_reference"
    FAILED = "failed"


def parse_variant_id(value: Any) -> tuple[str, int, str, str]:
    if not isinstance(value, str) or not GRCH38_VARIANT_ID.fullmatch(value):
        raise ValueError(f"invalid GRCh38 variant identifier: {value!r}")
    chromosome, position, reference, alternate = value.split("_")
    return chromosome, int(position), reference, alternate


@dataclass(frozen=True, slots=True)
class LocusVariant:
    variant_id: str
    sources: tuple[str, ...]
    pip: float | None
    r2_to_anchor: float | None
    weight: float
    rsid: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def position(self) -> int:
        return parse_variant_id(self.variant_id)[1]

    @property
    def query_ids(self) -> tuple[str, ...]:
        return (self.variant_id, *self.aliases)


@dataclass(frozen=True, slots=True)
class LocusVariantSet:
    lead_variant_id: str
    chromosome: str
    lead_position: int
    variants: tuple[LocusVariant, ...]
    weight_basis: str
    ld_status: str
    ld_anchor_variant_id: str | None = None
    issues: tuple[str, ...] = ()
    truncated: bool = False

    def by_query_id(self) -> dict[str, LocusVariant]:
        return {query_id: variant for variant in self.variants for query_id in variant.query_ids}

    def query_ids(self) -> tuple[str, ...]:
        return tuple(sorted({query_id for variant in self.variants for query_id in variant.query_ids}))

    def pip_centroid(self) -> float | None:
        weighted = [(variant.position, variant.pip) for variant in self.variants if variant.pip]
        total = sum(pip for _, pip in weighted)
        return sum(position * pip for position, pip in weighted) / total if total > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WindowGene:
    gene_id: str
    symbol: str | None
    chromosome: str
    start: int
    end: int
    strand: int
    biotype: str | None

    @property
    def tss(self) -> int:
        return self.start if self.strand >= 0 else self.end


@dataclass(frozen=True, slots=True)
class GeneWindow:
    chromosome: str
    start: int
    end: int
    genes: tuple[WindowGene, ...]
    status: str
    source: str
    explicit_missing: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class V2GRequest:
    locus_id: str
    lead_variant_id: str
    trait: str
    trait_efo_ids: tuple[str, ...] = ()
    credible_set: tuple[tuple[str, float], ...] = ()
    selected_variants: tuple[str, ...] = ()
    explicit_candidates: tuple[str, ...] = ()
    context_terms: tuple[str, ...] = ()
    biological_context: dict[str, Any] | None = None
    phenotype_scope: dict[str, Any] | None = None
    population: str = "EUR"
    use_ld: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class V2GResult:
    request: V2GRequest
    locus: LocusVariantSet
    window: GeneWindow
    genes: tuple[dict[str, Any], ...]
    retrieval_status: dict[str, str]
    raw_responses: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    # Presentation-only full observations. Never serialized into the scientific
    # input or raw result, which already archives the original provider records.
    display_genes: tuple[dict[str, Any], ...] = field(default_factory=tuple, repr=False)

    def gene_ids(self) -> tuple[str, ...]:
        return tuple(gene["gene_id"] for gene in self.genes)

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        document = {
            "request": self.request.to_dict(),
            "locus": self.locus.to_dict(),
            "window": self.window.to_dict(),
            "genes": list(self.genes),
            "retrieval_status": dict(self.retrieval_status),
            "issues": list(self.issues),
            "coverage": dict(self.coverage),
        }
        if include_raw:
            document["raw_responses"] = list(self.raw_responses)
        return document
