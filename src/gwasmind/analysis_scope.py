"""Private, validated LD retrieval scope; never a replacement GWAS dataset."""
# Input gates intentionally use ValueError for every invalid query shape.
# ruff: noqa: TRY004
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace

from .configuration import parameter
from .models import LocusQuery, VariantObservation

_CANONICAL = re.compile(r"^([0-9]{1,2}|X|Y|MT)_([1-9][0-9]*)_([ACGT]+)_([ACGT]+)$")


def _rows(value) -> tuple[VariantObservation, ...]:
    from .serialization import locus_query_from_dict

    if not isinstance(value, list):
        raise ValueError("LD observations must be arrays")
    if len(value) > parameter("novelty.max_ld_pairs") + parameter("execution.max_variants_per_query"):
        raise ValueError("LD analysis scope exceeds the recorded proxy budget")
    clean = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("LD observations must contain objects")
        clean.append({key: row[key] for key in ("variant_id", "p_value", "chromosome", "position") if key in row})
    size = parameter("execution.max_variants_per_query")
    result = []
    for start in range(0, len(clean), size):
        result.extend(locus_query_from_dict({"locus_id": "ld-scope", "trait": "identity validation",
                                            "variants": clean[start:start + size]}).variants)
    return tuple(result)


def attach_ld_expansion(query: LocusQuery, plan: Mapping) -> LocusQuery:
    """Called only by the server-owned job submission path, after LD retrieval."""
    if not isinstance(plan, Mapping):
        raise ValueError("LD expansion must be an object")
    result = replace(query, analysis_variants=_rows(plan.get("analysis_variants")),
                     ld_expansion=deepcopy(dict(plan)))
    result.validate()
    return result


def validate_analysis_scope(query: LocusQuery) -> None:
    rows, plan = query.analysis_variants, query.ld_expansion
    if not isinstance(rows, (tuple, list)):
        raise ValueError("analysis_variants must be an array")
    if not rows and plan is None:
        return
    if query.task != "prioritize_genes" or query.benchmark:
        raise ValueError("LD analysis scope is limited to non-benchmark gene prioritization")
    if not isinstance(plan, dict) or not rows:
        raise ValueError("LD analysis variants require their recorded expansion plan")
    if any(not isinstance(row, VariantObservation) for row in rows):
        raise ValueError("analysis_variants must contain VariantObservation values")
    if len(rows) > parameter("novelty.max_ld_pairs") + parameter("execution.max_variants_per_query"):
        raise ValueError("LD analysis scope exceeds the recorded proxy budget")
    if len({row.variant_id for row in rows}) != len(rows):
        raise ValueError("LD analysis variant IDs must be unique")
    if plan.get("genome_build") != "GRCh38" or query.genome_build not in (None, "GRCh38", "hg38"):
        raise ValueError("LD analysis requires verified GRCh38 identities")
    window = plan.get("window_bp")
    if isinstance(window, bool) or not isinstance(window, int) or not 0 < window <= parameter("ranking.ld_window_bp"):
        raise ValueError("LD window must be positive and within the configured ranking window")
    if plan.get("reference_panel") != parameter("novelty.reference_panel"):
        raise ValueError("LD reference panel disagrees with the configured reference")
    populations = {value.split("|")[0] for value in parameter("novelty.population_aliases")}
    if plan.get("population") is not None and (not isinstance(plan["population"], str) or plan["population"] not in populations):
        raise ValueError("LD population is not configured")
    threshold = plan.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("LD threshold must be finite and in (0, 1]")
    if threshold < parameter("ranking.ld_r2_threshold"):
        raise ValueError("LD threshold is below the configured ranking threshold")
    if not isinstance(plan.get("complete"), bool) or not isinstance(plan.get("status"), str) or plan["status"] not in {"complete", "incomplete", "disabled"}:
        raise ValueError("LD coverage status is invalid")
    if plan["complete"] != (plan["status"] == "complete"):
        raise ValueError("LD coverage status contradicts completeness")
    issues = plan.get("issues")
    if not isinstance(issues, list) or any(not isinstance(issue, str) or not issue.strip() for issue in issues):
        raise ValueError("LD coverage issues must be an array of non-empty strings")
    if plan["complete"] and (issues or plan.get("population") is None):
        raise ValueError("Complete LD coverage cannot retain unresolved issues or population")
    originals = _rows(plan.get("source_observations"))
    original_by_id = {row.variant_id: row for row in originals}
    if len(original_by_id) != len(originals):
        raise ValueError("LD source observations must be unique")
    aliases = plan.get("resolved_source_aliases", {})
    omitted = plan.get("omitted_source_variant_ids", [])
    if (not isinstance(aliases, dict) or any(not isinstance(key, str) or not isinstance(value, str)
            or not _CANONICAL.fullmatch(value) for key, value in aliases.items())
            or not isinstance(omitted, list) or any(not isinstance(value, str) for value in omitted)):
        raise ValueError("LD source resolution metadata is invalid")
    if set(aliases).difference(original_by_id) or set(omitted).difference(original_by_id):
        raise ValueError("LD source resolution contains an unrelated observation")
    expected = []
    for row in originals:
        if row.variant_id in omitted:
            continue
        resolved = replace(row, variant_id=aliases.get(row.variant_id, row.variant_id))
        existing = next((other for other in expected if other.variant_id == resolved.variant_id), None)
        if existing is not None and existing != resolved:
            raise ValueError("LD source aliases have conflicting GWAS observations")
        if existing is None:
            expected.append(resolved)
    if tuple(expected) != query.variants:
        raise ValueError("LD source observations do not match the source query")
    source = {row.variant_id: row for row in query.variants}
    analysis = {row.variant_id: row for row in rows}
    if any(analysis.get(identifier) != row for identifier, row in source.items()):
        raise ValueError("LD expansion changed or omitted a source GWAS observation")
    proxy_ids = set(analysis).difference(source)
    recorded = _rows(plan.get("analysis_variants"))
    recorded_by_id = {row.variant_id: row for row in recorded}
    if len(recorded_by_id) != len(recorded) or any(recorded_by_id.get(identifier) != row for identifier, row in original_by_id.items()):
        raise ValueError("Recorded LD analysis changed its source observations or duplicated an identity")
    links = plan.get("ld_links")
    excluded_links = plan.get("excluded_ld_links", [])
    if not isinstance(links, list) or len(links) > parameter("novelty.max_ld_pairs"):
        raise ValueError("LD links must be an array within the proxy budget")
    if not isinstance(excluded_links, list) or len(links) + len(excluded_links) > parameter("novelty.max_ld_pairs"):
        raise ValueError("Excluded LD links exceed the proxy budget")
    linked = set()
    link_values = {}
    for link in [*links, *excluded_links]:
        if not isinstance(link, dict):
            raise ValueError("LD link must be an object")
        selected = link.get("selected_variant_id")
        canonical_selected = aliases.get(selected, selected)
        proxy = link.get("proxy_variant_id")
        r2 = link.get("r2")
        excluded = link in excluded_links
        if (selected not in original_by_id or (selected not in omitted if excluded else canonical_selected not in source)
                or proxy not in (recorded_by_id if excluded else analysis)):
            raise ValueError("LD link contains an unrelated source or proxy identity")
        coordinate = _CANONICAL.fullmatch(str(proxy))
        if (coordinate is None or coordinate[1] not in {*(str(value) for value in range(1, 23)), "X", "Y", "MT"}
                or str(link.get("chromosome")) != coordinate[1]
                or isinstance(link.get("position"), bool) or not isinstance(link.get("position"), int)
                or link.get("position") != int(coordinate[2])
                or isinstance(link.get("end", link.get("position")), bool)
                or not isinstance(link.get("end", link.get("position")), int)
                or link.get("end", link.get("position")) != int(coordinate[2])):
            raise ValueError("LD link coordinates disagree with the canonical proxy variant identity")
        source_row = original_by_id[selected]
        source_coordinate = _CANONICAL.fullmatch(aliases.get(selected, selected))
        source_chromosome = source_coordinate[1] if source_coordinate else source_row.chromosome
        source_position = int(source_coordinate[2]) if source_coordinate else source_row.position
        if source_chromosome is not None and str(source_chromosome).upper().removeprefix("CHR") != coordinate[1]:
            raise ValueError("LD link crosses source chromosomes")
        if source_position is not None and abs(int(coordinate[2]) - source_position) > window:
            raise ValueError("LD link exceeds its recorded source window")
        if isinstance(r2, bool) or not isinstance(r2, (int, float)) or not math.isfinite(r2) or not threshold <= r2 <= 1:
            raise ValueError("LD proxy does not meet its recorded r2 threshold")
        if (link.get("status") != "computed" or link.get("genome_build") != "GRCh38"
                or link.get("source_gwas_p_copied") is not False
                or link.get("evidence_relation") != "via_population_ld"
                or not isinstance(link.get("population"), str) or not link["population"]):
            raise ValueError("LD link lacks verified population/build/relation metadata")
        if link["population"] != plan.get("population"):
            raise ValueError("LD link population disagrees with its expansion plan")
        if not isinstance(link.get("reported_rsid"), str) or not re.fullmatch(r"rs[1-9][0-9]*", link["reported_rsid"]):
            raise ValueError("LD link lacks a valid reference proxy rsID")
        key = (selected, proxy, link["population"])
        if key in link_values and link_values[key] != link:
            raise ValueError("conflicting duplicate LD links")
        link_values[key] = link
        if not excluded:
            linked.add(proxy)
    if not proxy_ids.issubset(linked):
        raise ValueError("LD analysis includes a proxy without a validated LD link")
    for identifier in proxy_ids:
        row = analysis[identifier]
        match = _CANONICAL.fullmatch(identifier)
        if not match or match[3] == match[4] or row.p_value is not None:
            raise ValueError("LD proxy must have a canonical variant ID and no source GWAS P value")
        if row.chromosome not in {None, match[1]} or row.position not in {None, int(match[2])}:
            raise ValueError("LD proxy coordinates conflict with its canonical ID")
    original_source_ids = set(original_by_id)
    recorded_proxies = {row.variant_id: row for row in recorded if row.variant_id not in original_source_ids}
    if any(recorded_proxies.get(identifier) != analysis[identifier] for identifier in proxy_ids):
        raise ValueError("LD analysis proxies disagree with the recorded plan")
    expected_proxy_ids = set(recorded_proxies).difference(source)
    if omitted:
        expected_proxy_ids.intersection_update(linked)
    if proxy_ids != expected_proxy_ids:
        raise ValueError("LD analysis silently omitted a recorded proxy")
    counts = {key: plan.get(key) for key in ("source_variant_count", "added_variant_count", "required_proxy_count", "validated_proxy_count")}
    for key, value in counts.items():
        if key == "required_proxy_count" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("LD coverage counts must be non-negative integers")
    if counts["source_variant_count"] != len(originals) or counts["added_variant_count"] != len(recorded_proxies):
        raise ValueError("LD source or added-variant counts disagree with the recorded observations")
    verified_rsids = {link["reported_rsid"] for link in [*links, *excluded_links]}
    if counts["validated_proxy_count"] != len(verified_rsids):
        raise ValueError("LD validated-proxy count disagrees with verified identities")
    required = counts["required_proxy_count"]
    if required is not None and required < counts["validated_proxy_count"]:
        raise ValueError("LD required-proxy count is smaller than its verified proxy count")
    if plan["complete"] and (required != counts["validated_proxy_count"] or omitted):
        raise ValueError("Complete LD coverage requires every requested proxy and source identity")
    if plan["status"] == "disabled" and (links or excluded_links or recorded_proxies):
        raise ValueError("Disabled LD expansion cannot include proxy evidence")


def resolved_ld_scope(query: LocusQuery, variants, aliases, omitted):
    """Rebind source aliases after validated resolution without copying source P."""
    if query.ld_expansion is None:
        return {}
    plan = deepcopy(query.ld_expansion)
    plan["resolved_source_aliases"] = dict(aliases)
    plan["omitted_source_variant_ids"] = list(omitted)
    if omitted:
        plan["excluded_ld_links"] = [row for row in plan["ld_links"] if row["selected_variant_id"] in omitted]
        plan["ld_links"] = [row for row in plan["ld_links"] if row["selected_variant_id"] not in omitted]
        plan["complete"], plan["status"] = False, "incomplete"
        plan["issues"] = [*plan.get("issues", []), "Some source identities could not be resolved; their LD links were excluded."]
    source_ids = {row.variant_id for row in query.variants}
    retained_source_ids = {row.variant_id for row in variants}
    linked = {row["proxy_variant_id"] for row in plan["ld_links"]}
    proxies = tuple(row for row in query.analysis_variants if row.variant_id not in source_ids
                    and row.variant_id not in retained_source_ids and row.variant_id in linked)
    return {"analysis_variants": (*variants, *proxies), "ld_expansion": plan}


def private_query_dict(query: LocusQuery) -> dict:
    """JSON-shaped private worker input, including its reviewed LD plan."""
    import json
    return json.loads(json.dumps(asdict(query)))
