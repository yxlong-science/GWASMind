"""Exact locus witnesses for complete, deduplicated enhancer-to-gene annotations.

The Open Targets variant and enhancer interval endpoints use GRCh38. A missing
build field therefore inherits that endpoint contract; a contradictory explicit
build is rejected. Locus admission, including LD validation, belongs upstream.

Inclusive overlap follows the actual Open Targets API contract: the variant
resolver passes its unchanged position as both query bounds, and the SQL tests
interval.start <= position <= interval.end. Source BED conventions must not be
applied a second time here.
https://github.com/opentargets/platform-api/blob/master/app/models/gql/Objects.scala
https://github.com/opentargets/platform-api/blob/master/app/models/db/EnhancerToGeneQuery.scala
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from ..configuration import parameter
from .models import LocusVariant, LocusVariantSet, VariantSource, parse_variant_id

if TYPE_CHECKING:
    from .retrieval import RetrievalBundle

_BUILD_KEYS = (
    "genomeBuild", "genome_build", "assembly", "assemblyName", "assembly_name",
    "referenceGenome", "reference_genome", "build",
)
_SELECTED_SOURCES = frozenset({
    VariantSource.LEAD.value, VariantSource.SELECTED.value, VariantSource.CREDIBLE_SET.value,
})


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _grch38(record: Mapping[str, Any]) -> bool:
    return all(
        value is None or str(value).casefold().strip() in {"grch38", "hg38", "38"}
        for key in _BUILD_KEYS if (value := record.get(key)) is not None
    )


def _chromosome(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    chromosome = str(value).strip()
    if chromosome.casefold().startswith("chr"):
        chromosome = chromosome[3:]
    return chromosome


def _coordinate(identifier: Any) -> tuple[str, int] | None:
    try:
        chromosome, position, _, _ = parse_variant_id(identifier)
    except ValueError:
        return None
    return chromosome, position


def _exact_interval(row: Mapping[str, Any], coordinate: tuple[str, int]) -> bool:
    start, end = row.get("start"), row.get("end")
    return (
        _grch38(row)
        and _chromosome(row.get("chromosome")) == coordinate[0]
        and isinstance(start, int) and not isinstance(start, bool)
        and isinstance(end, int) and not isinstance(end, bool)
        and 0 <= start <= coordinate[1] <= end
    )


def _witness_order(witness: Mapping[str, Any], lead: str) -> tuple[Any, ...]:
    variant = witness["variant"]
    return (
        not bool(_SELECTED_SOURCES.intersection(variant.sources)),
        variant.variant_id != lead,
        variant.variant_id,
        witness["query_variant_id"] != variant.variant_id,
        witness["query_variant_id"],
        witness["resolved_variant_id"],
    )


def _public_witness(witness: Mapping[str, Any]) -> dict[str, Any]:
    variant: LocusVariant = witness["variant"]
    return {
        "variant_id": variant.variant_id,
        "query_variant_id": witness["query_variant_id"],
        "resolved_variant_id": witness["resolved_variant_id"],
        "variant_sources": list(variant.sources),
        "gwas_pip": variant.pip,
        "r2_to_anchor": variant.r2_to_anchor,
    }


def _biosample_key(row: Mapping[str, Any], annotation: str) -> tuple[str, str]:
    biosample = row.get("biosample")
    biosample = biosample if isinstance(biosample, Mapping) else {}
    identifier = biosample.get("biosampleId")
    if isinstance(identifier, str) and identifier.strip():
        return "id", identifier.strip().casefold().replace(":", "_")
    name = biosample.get("biosampleName") or row.get("biosampleName")
    if isinstance(name, str) and name.strip():
        return "name", " ".join(name.casefold().split())
    # Missing identities do not establish that two observations share a sample.
    return "unresolved", annotation


def source_score_threshold(row):
    source = str(row.get("datasourceId") or "").casefold().replace("-", "_")
    if source in {"e2g", "encode_re2g", "re2g", "encode_e2g"}:
        return parameter("v2g.e2g.encode_re2g_min_score")
    # Unknown score calibrations are not silently assigned the ENCODE cutoff.
    return 0.0


def build_e2g_records(
    gene_id: str, locus: LocusVariantSet, bundle: RetrievalBundle,
) -> list[dict[str, Any]]:
    """Keep the strongest qualified link per variant, gene and biosample.

    Scores are compared at raw precision before display rounding. Equal scores
    use a deterministic complete-annotation tie break. Aliases share the same
    canonical variant group; one annotation winning for multiple variants is
    stored once with those witnesses. Provider pages remain untouched in raw.
    """
    grouped: dict[str, dict[str, Any]] = {}
    query_owners: dict[str, set[str]] = {}
    for variant in locus.variants:
        for query_id in variant.query_ids:
            query_owners.setdefault(query_id, set()).add(variant.variant_id)

    for variant in locus.variants:
        coordinate = _coordinate(variant.variant_id)
        if coordinate is None or coordinate[0] != _chromosome(locus.chromosome):
            continue
        for query_id in variant.query_ids:
            if query_owners[query_id] != {variant.variant_id} or _coordinate(query_id) != coordinate:
                continue
            record = bundle.variants.get(query_id)
            if not isinstance(record, Mapping) or not _grch38(record):
                continue
            resolved_id = record.get("id")
            if (resolved_id not in variant.query_ids or _coordinate(resolved_id) != coordinate
                    or query_owners.get(resolved_id) != {variant.variant_id}):
                continue
            enhancer_to_genes = record.get("enhancerToGenes")
            if not isinstance(enhancer_to_genes, Mapping):
                continue
            for row in enhancer_to_genes.get("rows") or []:
                if not isinstance(row, Mapping):
                    continue
                target = row.get("target")
                if not isinstance(target, Mapping) or target.get("id") != gene_id:
                    continue
                score = _finite(row.get("score"))
                if (score is None or not source_score_threshold(row) <= score <= 1
                        or not _exact_interval(row, coordinate)):
                    continue
                annotation = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                entry = grouped.setdefault(annotation, {"row": deepcopy(dict(row)), "witnesses": {}})
                identity = (variant.variant_id, query_id, resolved_id)
                entry["witnesses"].setdefault(identity, {
                    "variant": variant, "query_variant_id": query_id,
                    "resolved_variant_id": resolved_id,
                })

    best: dict[tuple[str, tuple[str, str]], str] = {}
    for annotation, entry in sorted(grouped.items(), key=lambda item: (-item[1]["row"]["score"], item[0])):
        biosample_key = _biosample_key(entry["row"], annotation)
        for witness in entry["witnesses"].values():
            best.setdefault((witness["variant"].variant_id, biosample_key), annotation)

    records = []
    for annotation, entry in grouped.items():
        row = entry["row"]
        biosample_key = _biosample_key(row, annotation)
        witnesses = sorted((w for w in entry["witnesses"].values()
                            if best[(w["variant"].variant_id, biosample_key)] == annotation),
                           key=lambda item: _witness_order(item, locus.lead_variant_id))
        if not witnesses:
            continue
        variant = witnesses[0]["variant"]
        biosample = row.get("biosample")
        biosample_name = row.get("biosampleName") or (
            biosample.get("biosampleName") if isinstance(biosample, Mapping) else None)
        records.append({
            "variant_id": variant.variant_id, "gwas_pip": variant.pip,
            "r2_to_anchor": variant.r2_to_anchor, "method": row.get("datasourceId"),
            "interval_type": row.get("intervalType"), "score": row["score"],
            "score_threshold": source_score_threshold(row),
            "biosample": biosample_name,
            "biosample_id": biosample.get("biosampleId") if isinstance(biosample, Mapping) else None,
            "chromosome": row.get("chromosome"), "start": row.get("start"), "end": row.get("end"),
            "study_id": row.get("studyId"), "pmid": row.get("pmid"),
            "distance_to_tss": row.get("distanceToTss"),
            "resource_scores": deepcopy(row.get("resourceScore") or []),
            "physical_annotation_sha256": hashlib.sha256(annotation.encode()).hexdigest(),
            "exact_variant_interval_overlap": True,
            "witnesses": [_public_witness(witness) for witness in witnesses],
            "_source": {"row": row, "variant": variant, "witnesses": witnesses},
        })
    records.sort(key=lambda row: (-row["score"], row["physical_annotation_sha256"]))
    bundle.coverage.setdefault("e2g_selection", {})[gene_id] = {
        "policy": "strongest_per_variant_gene_biosample_v1",
        "qualified_annotation_count": len(grouped), "variant_biosample_groups": len(best),
        "retained_annotation_count": len(records), "collapsed_annotation_count": len(grouped) - len(records),
    }
    return records
