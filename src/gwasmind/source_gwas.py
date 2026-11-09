from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .configuration import parameter
from .models import LocusQuery

GENOME_WIDE_SIGNIFICANCE = parameter("analysis.gwas_significance_p")


@dataclass(frozen=True, slots=True)
class SelectedVariantMeasurement:
    variant_id: str
    p_value: float | None
    genome_wide_significant: bool | None


@dataclass(frozen=True, slots=True)
class SourceGWASSelectionSummary:
    """Deterministic facts about the current uploaded GWAS selection."""

    phenotype: str
    selected_variants: int
    variants_with_p_value: int
    genome_wide_significant_variants: int
    minimum_p_value: float | None
    maximum_p_value: float | None
    significance_threshold: float
    measurements: tuple[SelectedVariantMeasurement, ...]

    @property
    def has_genome_wide_signal(self) -> bool | None:
        if not self.variants_with_p_value:
            return None
        return self.genome_wide_significant_variants > 0

    @property
    def all_measured_variants_are_significant(self) -> bool | None:
        if not self.variants_with_p_value:
            return None
        return (
            self.variants_with_p_value == self.selected_variants
            and self.genome_wide_significant_variants == self.selected_variants
        )

    def to_dict(self) -> dict:
        value = asdict(self)
        value["has_genome_wide_signal"] = self.has_genome_wide_signal
        value["all_measured_variants_are_significant"] = (
            self.all_measured_variants_are_significant
        )
        return value


def summarize_source_gwas_selection(
    query: LocusQuery,
    *,
    threshold: float = GENOME_WIDE_SIGNIFICANCE,
) -> SourceGWASSelectionSummary:
    """Summarize current-GWAS measurements before any external annotation.

    This is intentionally independent of Open Targets or an LLM.  It prevents
    an association reported in an external study from being relabelled as a
    significant association in the user's uploaded GWAS.
    """

    observations = tuple(
        {"variant_id": item.variant_id, "p_value": item.p_value}
        for item in query.variants
    )
    return summarize_source_gwas_observations(
        observations,
        phenotype=query.source_trait or query.trait,
        threshold=threshold,
    )


def summarize_source_gwas_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    phenotype: str,
    threshold: float = GENOME_WIDE_SIGNIFICANCE,
) -> SourceGWASSelectionSummary:
    """Summarize browser-selected observations without requiring a ``LocusQuery``.

    The web layer already has the selected observations before it constructs a
    retrieval query.  Keeping this calculation here gives both paths the same
    definition of genome-wide significance and, crucially, keeps external
    database associations out of the current-GWAS signal calculation.
    """

    if not 0 < threshold < 1:
        raise ValueError("significance threshold must be between zero and one")

    measurements = tuple(
        _measurement_from_mapping(item, threshold=threshold) for item in observations
    )
    measured = tuple(
        item.p_value for item in measurements if item.p_value is not None
    )
    return SourceGWASSelectionSummary(
        phenotype=" ".join(str(phenotype or "").split()),
        selected_variants=len(measurements),
        variants_with_p_value=len(measured),
        genome_wide_significant_variants=sum(value <= threshold for value in measured),
        minimum_p_value=min(measured) if measured else None,
        maximum_p_value=max(measured) if measured else None,
        significance_threshold=threshold,
        measurements=measurements,
    )


def _measurement_from_mapping(
    observation: Mapping[str, Any],
    *,
    threshold: float,
) -> SelectedVariantMeasurement:
    variant_id = " ".join(str(observation.get("variant_id") or "").split())[:160]
    try:
        raw_p_value = observation.get("p_value")
        p_value = None if isinstance(raw_p_value, bool) else float(raw_p_value)
    except (TypeError, ValueError, OverflowError):
        p_value = None
    if p_value is not None and (
        not math.isfinite(p_value) or not 0 < p_value <= 1
    ):
        p_value = None
    return SelectedVariantMeasurement(
        variant_id=variant_id,
        p_value=p_value,
        genome_wide_significant=(
            p_value <= threshold if p_value is not None else None
        ),
    )
