"""Build the locus variant set that anchors gene-centric evidence."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..configuration import parameter
from .models import (
    LDStatus,
    LocusVariant,
    LocusVariantSet,
    VariantSource,
    WeightBasis,
    parse_variant_id,
)

_NUCLEOTIDES = frozenset("ACGT")


def snv_orientations(chromosome: str, position: int, alleles: Sequence[str]) -> tuple[str, ...]:
    """Both REF/ALT orders of a biallelic SNV reported without a reference orientation."""

    bases = [str(item).strip().upper() for item in alleles]
    if len(bases) != 2 or bases[0] == bases[1] or any(len(item) != 1 or item not in _NUCLEOTIDES for item in bases):
        return ()
    return (f"{chromosome}_{position}_{bases[0]}_{bases[1]}", f"{chromosome}_{position}_{bases[1]}_{bases[0]}")


def orientation_aliases(variant_id: str) -> tuple[str, ...]:
    """The reversed REF/ALT identifier of a biallelic SNV; uploads may swap alleles."""

    chromosome, position, reference, alternate = parse_variant_id(variant_id)
    orientations = snv_orientations(chromosome, position, (reference, alternate))
    return tuple(value for value in orientations if value != variant_id)


def ld_proxies_from_rows(rows: Iterable[Mapping[str, Any]], *, chromosome: str,
                         min_r2: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert LD provider rows to orientation-ambiguous proxy records.

    Rows below ``min_r2``, on another chromosome, or not describing a biallelic
    SNV are skipped and counted; skipped rows are not treated as absent LD.
    """

    proxies: list[dict[str, Any]] = []
    skipped = {"below_threshold": 0, "other_chromosome": 0, "not_biallelic_snv": 0, "malformed": 0}
    for row in rows:
        try:
            r2 = float(row["r2"])
            position = int(row["position"])
            row_chromosome = str(row["chromosome"])
        except (KeyError, TypeError, ValueError):
            skipped["malformed"] += 1
            continue
        if not math.isfinite(r2) or not 0 <= r2 <= 1:
            skipped["malformed"] += 1
            continue
        if row_chromosome != chromosome:
            skipped["other_chromosome"] += 1
            continue
        if r2 < min_r2:
            skipped["below_threshold"] += 1
            continue
        orientations = snv_orientations(chromosome, position, row.get("population_alleles") or ())
        if not orientations:
            skipped["not_biallelic_snv"] += 1
            continue
        rsid = row.get("reported_rsid") or row.get("rsid")
        proxies.append({"variant_id": orientations[0], "aliases": orientations[1:], "r2": r2,
                        "rsid": str(rsid) if rsid else None})
    return proxies, skipped


def proxies_from_ld_expansion(ld_expansion: Mapping[str, Any] | None,
                              analysis_variant_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Proxy records from the ranking LD expansion (identity-verified ``ld_links``).

    Only proxies admitted to the analysis variant set are used; a proxy linked to
    several selected variants keeps its highest r2.
    """

    admitted = set(analysis_variant_ids)
    best: dict[str, dict[str, Any]] = {}
    for link in (ld_expansion or {}).get("ld_links") or []:
        variant_id = link.get("proxy_variant_id") if isinstance(link, Mapping) else None
        r2 = _valid_probability(link.get("r2")) if isinstance(link, Mapping) else None
        if not isinstance(variant_id, str) or variant_id not in admitted or r2 is None:
            continue
        current = best.get(variant_id)
        if current is None or r2 > current["r2"]:
            rsid = link.get("reported_rsid")
            best[variant_id] = {"variant_id": variant_id, "aliases": orientation_aliases(variant_id), "r2": r2,
                                "rsid": str(rsid) if rsid else None}
    return [best[key] for key in sorted(best)]


def _valid_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def build_locus_variant_set(
    lead_variant_id: str,
    *,
    credible_set: Sequence[tuple[str, float]] = (),
    ld_proxies: Sequence[Mapping[str, Any]] = (),
    ld_status: str = LDStatus.NOT_REQUESTED,
    ld_anchor_variant_id: str | None = None,
    issues: Sequence[str] = (),
    lead_rsid: str | None = None,
    selected_variants: Sequence[str] = (),
) -> LocusVariantSet:
    """Merge lead, selected variants, credible-set members and LD proxies into one weighted set.

    With supplied PIPs the weights are those PIPs (proxies outside the credible
    set weigh zero). Without PIPs the weights are r2-proportional and normalized
    to one; user-selected variants count as core anchors (r2 = 1). The basis is
    recorded because it is an approximation.
    """

    chromosome, lead_position, _, _ = parse_variant_id(lead_variant_id)
    notes = list(issues)
    entries: dict[str, dict[str, Any]] = {
        lead_variant_id: {"sources": {VariantSource.LEAD.value}, "pip": None, "r2": 1.0,
                          "rsid": lead_rsid, "aliases": set(orientation_aliases(lead_variant_id))}
    }
    for variant_id in selected_variants:
        if variant_id == lead_variant_id:
            continue
        try:
            selected_chromosome = parse_variant_id(variant_id)[0]
        except ValueError:
            notes.append(f"Selected variant {variant_id!r} is not a GRCh38 variant identifier and was skipped.")
            continue
        if selected_chromosome != chromosome:
            notes.append(f"Selected variant {variant_id} is on another chromosome and was skipped.")
            continue
        entries.setdefault(variant_id, {"sources": set(), "pip": None, "r2": 1.0, "rsid": None,
                                        "aliases": set(orientation_aliases(variant_id))})
        entries[variant_id]["sources"].add(VariantSource.SELECTED.value)
    for variant_id, pip in credible_set:
        try:
            member_chromosome = parse_variant_id(variant_id)[0]
        except ValueError:
            notes.append(f"Credible-set member {variant_id!r} is not a GRCh38 variant identifier and was skipped.")
            continue
        probability = _valid_probability(pip)
        if member_chromosome != chromosome or probability is None:
            notes.append(f"Credible-set member {variant_id} was skipped (other chromosome or invalid PIP).")
            continue
        entry = entries.setdefault(variant_id, {"sources": set(), "pip": None, "r2": None,
                                                "rsid": None, "aliases": set()})
        entry["sources"].add(VariantSource.CREDIBLE_SET.value)
        entry["pip"] = probability if entry["pip"] is None else max(entry["pip"], probability)

    r2_extended = parameter("v2g.ld.r2_extended")
    alias_owner = {alias: variant_id for variant_id, entry in entries.items()
                   for alias in (variant_id, *entry["aliases"])}
    for proxy in ld_proxies:
        r2 = _valid_probability(proxy.get("r2"))
        identifiers = [proxy.get("variant_id"), *(proxy.get("aliases") or ())]
        identifiers = [value for value in identifiers if isinstance(value, str)]
        if r2 is None or r2 < r2_extended or not identifiers:
            continue
        try:
            if parse_variant_id(identifiers[0])[0] != chromosome:
                continue
        except ValueError:
            continue
        owner = next((alias_owner[value] for value in identifiers if value in alias_owner), None)
        if owner is None:
            owner = identifiers[0]
            entries[owner] = {"sources": set(), "pip": None, "r2": None, "rsid": proxy.get("rsid"),
                              "aliases": set(identifiers[1:])}
        entry = entries[owner]
        entry["sources"].add(VariantSource.LD_PROXY.value)
        entry["r2"] = r2 if entry["r2"] is None else max(entry["r2"], r2)
        entry["rsid"] = entry["rsid"] or proxy.get("rsid")
        entry["aliases"].update(value for value in identifiers if value != owner)
        for value in identifiers:
            alias_owner.setdefault(value, owner)

    has_pip = any(entry["pip"] is not None for entry in entries.values())
    if has_pip:
        basis = WeightBasis.PIP.value
        weights = {variant_id: entry["pip"] or 0.0 for variant_id, entry in entries.items()}
    else:
        basis = WeightBasis.LD_APPROXIMATION.value
        total = sum(entry["r2"] or 0.0 for entry in entries.values())
        weights = {variant_id: (entry["r2"] or 0.0) / total for variant_id, entry in entries.items()}

    ordered = sorted(entries, key=lambda variant_id: (variant_id != lead_variant_id, -weights[variant_id],
                                                      -(entries[variant_id]["r2"] or 0.0), variant_id))
    limit = parameter("v2g.locus.max_variants")
    truncated = len(ordered) > limit
    if truncated:
        notes.append(f"The locus set was truncated to the {limit} highest-weight variants.")
    variants = tuple(
        LocusVariant(
            variant_id=variant_id,
            sources=tuple(sorted(entries[variant_id]["sources"])),
            pip=entries[variant_id]["pip"],
            r2_to_anchor=entries[variant_id]["r2"],
            weight=weights[variant_id],
            rsid=entries[variant_id]["rsid"],
            aliases=tuple(sorted(entries[variant_id]["aliases"])),
        )
        for variant_id in ordered[:limit]
    )
    return LocusVariantSet(
        lead_variant_id=lead_variant_id,
        chromosome=chromosome,
        lead_position=lead_position,
        variants=variants,
        weight_basis=basis,
        ld_status=str(ld_status),
        ld_anchor_variant_id=ld_anchor_variant_id,
        issues=tuple(notes),
        truncated=truncated,
    )


def ld_anchor_candidates(lead_variant_id: str, credible_set: Sequence[tuple[str, float]] = (),
                         selected_variants: Sequence[str] = ()) -> tuple[str, ...]:
    """The lead first, then selected variants in the given order, then credible-set members by PIP."""

    members = sorted(((pip, variant_id) for variant_id, pip in credible_set
                      if isinstance(variant_id, str) and variant_id != lead_variant_id
                      and _valid_probability(pip) is not None), key=lambda item: (-item[0], item[1]))
    ordered = (lead_variant_id, *selected_variants, *(variant_id for _, variant_id in members))
    return tuple(dict.fromkeys(ordered))
