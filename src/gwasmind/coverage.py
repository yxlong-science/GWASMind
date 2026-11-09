from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import LocusQuery

OT_MHC_CHROMOSOME = "6"
OT_MHC_START = 25_726_063
OT_MHC_END = 33_400_556
OT_CREDIBLE_SET_DOCS_URL = "https://platform-docs.opentargets.org/credible-set"
COMPLEX_REGION_CONFIG_VERSION = "2026-09-20.1"


def _load_complex_region_definitions() -> tuple[dict[str, Any], ...]:
    """Load the small, versioned set of recognized complex regions.

    This registry deliberately contains only regions with an explicit published
    or benchmark definition. It is not a generic low-mappability mask. A
    malformed local config fails closed to the established Open Targets MHC
    interval rather than inventing a broad region warning.
    """
    path = Path(__file__).with_name("config") / "complex_regions.json"
    fallback = ({
        "name": "MHC/HLA",
        "genome_build": "GRCh38",
        "chromosome": OT_MHC_CHROMOSOME,
        "start": OT_MHC_START,
        "end": OT_MHC_END,
        "source": OT_CREDIBLE_SET_DOCS_URL,
        "description": "Open Targets MHC credible-set exclusion interval",
    },)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("version") != COMPLEX_REGION_CONFIG_VERSION:
            return fallback
        rows = payload.get("regions")
        if not isinstance(rows, list):
            return fallback
        accepted = []
        for row in rows:
            if not isinstance(row, Mapping):
                return fallback
            name = str(row.get("name") or "").strip()
            build = str(row.get("genome_build") or "").strip()
            chromosome = str(row.get("chromosome") or "").strip().upper().removeprefix("CHR")
            start, end = row.get("start"), row.get("end")
            if (not name or not build or not chromosome or isinstance(start, bool)
                    or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int)
                    or start < 1 or end < start or not str(row.get("source") or "").strip()):
                return fallback
            accepted.append({
                "name": name, "genome_build": build, "chromosome": chromosome,
                "start": start, "end": end, "source": str(row["source"]),
                "description": str(row.get("description") or "").strip(),
            })
        return tuple(accepted) or fallback
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


COMPLEX_REGION_DEFINITIONS = _load_complex_region_definitions()


def is_ot_mhc_position(chromosome: Any, position: Any) -> bool:
    normalized = str(chromosome or "").upper().removeprefix("CHR")
    if isinstance(position, bool) or (
        isinstance(position, float) and not position.is_integer()
    ):
        return False
    try:
        coordinate = int(position)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        normalized == OT_MHC_CHROMOSOME
        and OT_MHC_START <= coordinate <= OT_MHC_END
    )


def recognized_complex_region_hits(
    chromosome: Any, position: Any, *, genome_build: str = "GRCh38"
) -> tuple[Mapping[str, Any], ...]:
    """Return configured recognized-region definitions containing a coordinate."""
    normalized = str(chromosome or "").upper().removeprefix("CHR")
    if isinstance(position, bool) or (
        isinstance(position, float) and not position.is_integer()
    ):
        return ()
    try:
        coordinate = int(position)
    except (TypeError, ValueError, OverflowError):
        return ()
    return tuple(
        row for row in COMPLEX_REGION_DEFINITIONS
        if row["genome_build"] == genome_build
        and row["chromosome"] == normalized
        and row["start"] <= coordinate <= row["end"]
    )


def recognized_complex_region_names(
    chromosome: Any, position: Any, *, genome_build: str = "GRCh38"
) -> tuple[str, ...]:
    return tuple(row["name"] for row in recognized_complex_region_hits(
        chromosome, position, genome_build=genome_build
    ))


def mappings_overlap_complex_region(
    items: Iterable[Mapping[str, Any]], *, genome_build: str = "GRCh38"
) -> tuple[dict[str, Any], ...]:
    """Return unique configured region hits for mapping-like variant rows."""
    hits: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for item in items:
        for row in recognized_complex_region_hits(
            item.get("chromosome"), item.get("position"), genome_build=genome_build
        ):
            key = (row["name"], row["chromosome"], row["start"], row["end"])
            hits[key] = dict(row)
    return tuple(hits.values())


def query_overlaps_ot_mhc(query: LocusQuery) -> bool:
    variants = query.analysis_variants if query.task == "prioritize_genes" and query.analysis_variants else query.variants
    for item in variants:
        if is_ot_mhc_position(item.chromosome, item.position):
            return True
        coordinate = re.fullmatch(
            r"(?:chr)?([0-9]+|X|Y|MT)[_:]([0-9]+)[_:][ACGTN-]+[_:][ACGTN-]+",
            item.variant_id, re.IGNORECASE,
        )
        if coordinate and is_ot_mhc_position(*coordinate.groups()):
            return True
    return False


def mappings_overlap_ot_mhc(items: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        is_ot_mhc_position(item.get("chromosome"), item.get("position"))
        for item in items
    )


def ot_mhc_coverage_limitation() -> str:
    return (
        "The selected locus overlaps the Open Targets MHC credible-set exclusion "
        f"interval chr6:{OT_MHC_START}-{OT_MHC_END}. Missing credible-set records may "
        "therefore reflect source coverage; the live API does not expose per-record "
        f"exclusion reasons. See {OT_CREDIBLE_SET_DOCS_URL}."
    )
