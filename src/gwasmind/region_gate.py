from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .configuration import parameter
from .models import EvidenceRecord

REGION_GENE_CONTEXT_DATASETS = frozenset(
    {"ensembl_region_gene_context", "frozen_region_gene_context", "opentargets_region_gene_context"}
)
_GRCH38_VARIANT = re.compile(r"^([1-9]|1[0-9]|2[0-2]|X|Y|MT)_([1-9][0-9]*)_([ACGTN]+)_([ACGTN]+)$")
_ENSEMBL_GENE = re.compile(r"^ENSG[0-9]{11}(?:\.[0-9]+)?$")


def local_gene_window_label() -> str:
    """Display the configured flank using the existing megabase convention."""
    return f'{parameter("analysis.local_gene_window_bp") / 1_000_000:g} Mb'


def nearby_gene_witnesses_from_ot(
    profiles: Sequence[Mapping[str, Any]],
    *,
    variant_ids: Sequence[str],
    genome_build: str,
    flank_bp: int = parameter("analysis.local_gene_window_bp"),
) -> tuple[dict[str, Any], ...]:
    """Prove positive nearby-gene existence from exact selected-variant OT annotations.

    An empty result means no usable witness was available, never a gene desert.
    Footprint distance establishes proximity, not a functional variant-to-gene link.
    """

    if genome_build != "GRCh38":
        return ()
    selected = {value: _GRCH38_VARIANT.fullmatch(value) for value in variant_ids}
    witnesses: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        variant_id = str(profile.get("id") or "")
        variant = selected.get(variant_id)
        if variant is None:
            continue
        chromosome, position, reference, alternate = variant.groups()
        if (
            str(profile.get("chromosome") or "") != chromosome
            or isinstance(profile.get("position"), bool)
            or profile.get("position") != int(position)
            or profile.get("referenceAllele") != reference
            or profile.get("alternateAllele") != alternate
            or profile.get("genomeBuild", "GRCh38") != "GRCh38"
        ):
            continue
        consequences = profile.get("transcriptConsequences")
        if not isinstance(consequences, (list, tuple)):
            continue
        for consequence in consequences:
            if not isinstance(consequence, Mapping):
                continue
            target = consequence.get("target")
            if not isinstance(target, Mapping):
                continue
            gene_id = str(target.get("id") or "")
            distance = consequence.get("distanceFromFootprint")
            if (
                not _ENSEMBL_GENE.fullmatch(gene_id)
                or isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not 0 <= distance <= flank_bp
                or not float(distance).is_integer()
            ):
                continue
            gene_id = gene_id.split(".", 1)[0]
            witness = {
                "gene_id": gene_id,
                "variant_id": variant_id,
                "distance_from_footprint": int(distance),
            }
            previous = witnesses.get(gene_id)
            if previous is None or distance < previous["distance_from_footprint"]:
                witnesses[gene_id] = witness
    return tuple(witnesses[gene] for gene in sorted(witnesses))


@dataclass(frozen=True, slots=True)
class RegionGeneContext:
    status: str
    annotated_gene_count: int | None
    candidate_gene_ids: tuple[str, ...]
    source: str | None
    data_release: str | None
    flank_bp: int | None
    truncated: bool
    evidence_id: str | None
    count_is_lower_bound: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def region_gene_context_from_records(
    records: Sequence[EvidenceRecord],
) -> RegionGeneContext:
    """Read the single auditable ±1 Mb gene-context summary record."""

    summaries = []
    for record in records:
        if record.dataset not in REGION_GENE_CONTEXT_DATASETS:
            continue
        payload = record.payload if isinstance(record.payload, Mapping) else {}
        if payload.get("relation") != "regional_gene_context_summary":
            continue
        summaries.append(record)
    if not summaries:
        return RegionGeneContext("absent", None, (), None, None, None, False, None)
    if len(summaries) != 1:
        return RegionGeneContext("malformed", None, (), None, None, None, False, None)

    record = summaries[0]
    payload = record.payload
    derived = payload.get("derived")
    measurements = payload.get("measurements")
    context = payload.get("context")
    derived = derived if isinstance(derived, Mapping) else {}
    measurements = measurements if isinstance(measurements, Mapping) else {}
    context = context if isinstance(context, Mapping) else {}
    status = str(derived.get("status") or "malformed").strip().casefold()
    if status not in {"complete", "nonempty", "unavailable"}:
        status = "malformed"
    raw_count = measurements.get("annotated_gene_count")
    count = (
        int(raw_count)
        if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count >= 0
        else None
    )
    candidates = context.get("candidate_gene_ids")
    candidate_ids = (
        tuple(dict.fromkeys(str(item) for item in candidates if str(item).strip()))
        if isinstance(candidates, (list, tuple))
        else ()
    )
    raw_flank = context.get("flank_bp")
    flank = (
        int(raw_flank)
        if isinstance(raw_flank, int) and not isinstance(raw_flank, bool) and raw_flank >= 0
        else None
    )
    if status == "complete" and count is None:
        status = "malformed"
    lower_bound = derived.get("count_is_lower_bound") is True
    if status == "nonempty" and (not lower_bound or count is None or count < 1):
        status = "malformed"
    return RegionGeneContext(
        status=status,
        annotated_gene_count=count,
        candidate_gene_ids=candidate_ids,
        source=record.source,
        data_release=record.data_release,
        flank_bp=flank,
        truncated=bool(derived.get("truncated")),
        evidence_id=record.evidence_id,
        count_is_lower_bound=lower_bound,
    )
