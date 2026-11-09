"""Deterministic first-tier discovery using L2G and context expression specificity.

This module performs no retrieval or inference. All callers, including offline
replays, use exactly the evidence available at the locus discovery boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .configuration import parameter
from .expression_policy import DEFAULT_EXPRESSION_FOLD_THRESHOLD
from .expression_specificity import DEFAULT_EXPRESSION_Z_THRESHOLD, expression_specificity
from .molecular_context import molecular_anchor_verified, molecular_context_match
from .supplementary_molecular import frozen_context_assessment, select_supplementary_molecular

POLICY_VERSION = "first-tier-context-hpa-v3.1"
LOCAL_WINDOW_BP = parameter("analysis.local_gene_window_bp")
MAX_DISCOVERED_TARGETS = parameter("ranking.max_discovered_targets")  # downstream transport limit; never a shortlist filling target
MAX_DISCOVERED_L2G_TARGETS_PER_LOCUS = parameter("ranking.max_l2g_targets_per_locus")
MOLECULAR_MODULES = frozenset({"locus.qtl", "locus.colocalisation", "locus.enhancer_to_gene"})
# Explicit gene-linked SO terms; a consequence score, biotype, SIFT score or
# generic HIGH flag alone is not proof of a coding/splice relationship.
CODING_TERMS = frozenset(
    {
        "transcript_ablation",
        "splice_acceptor_variant",
        "splice_donor_variant",
        "stop_gained",
        "frameshift_variant",
        "stop_lost",
        "start_lost",
        "inframe_insertion",
        "inframe_deletion",
        "missense_variant",
        "protein_altering_variant",
        "SO:0001893",
        "SO:0001574",
        "SO:0001575",
        "SO:0001587",
        "SO:0001589",
        "SO:0001578",
        "SO:0002012",
        "SO:0001821",
        "SO:0001822",
        "SO:0001583",
        "SO:0001818",
    }
)
_CONSEQUENCE_KEYS = frozenset(
    {
        "consequence",
        "consequenceterm",
        "consequenceterms",
        "consequenceids",
        "mostsevereconsequence",
        "transcriptconsequence",
    }
)


def _map(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any, *, signed: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        finite = math.isfinite(value)
    except OverflowError:
        return None
    if not finite or int(value) != value or (not signed and value < 0):
        return None
    return int(value)


def _chromosome(value: Any) -> str:
    text = str(value or "").upper().removeprefix("CHR")
    return text if text in {str(i) for i in range(1, 23)} | {"X", "Y", "MT"} else ""


def _finite_score(value: Any) -> bool:
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _variant(value: Any) -> tuple[str, int] | None:
    match = re.fullmatch(
        r"(?:chr)?([0-9]+|X|Y|MT)[_:]([0-9]+)[_:][ACGTN]+[_:][ACGTN]+", str(value), re.IGNORECASE
    )
    if match and _chromosome(match[1]):
        try:
            position = int(match[2])
        except ValueError:
            return None
        if position > 0:
            return _chromosome(match[1]), position
    return None


def target_ids(record: Mapping) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                entity["id"].strip()
                for entity in (_map(record.get("object")), _map(record.get("subject")))
                if entity.get("type") in {"target", "gene"}
                and isinstance(entity.get("id"), str)
                and entity["id"].strip()
            }
        )
    )


def record_target_identity_valid(record: Mapping) -> bool:
    """Do not borrow another gene's raw annotation, expression or coordinates."""
    targets = target_ids(record)
    if len(targets) != 1:
        return False
    expected = targets[0]
    raw = _map(_map(record.get("upstream")).get("raw"))
    identifiers = []
    for source in (raw, _map(record.get("context"))):
        identifiers.extend(
            source[key]
            for key in ("geneId", "targetId", "qtlGeneId", "gene_id", "target_id")
            if source.get(key) is not None
        )
    identifiers.extend(
        _map(raw.get(key))["id"]
        for key in ("gene", "target")
        if _map(raw.get(key)).get("id") is not None
    )
    return all(isinstance(value, str) and value.strip() == expected for value in identifiers)


def _record_build_compatible(record: Mapping) -> bool:
    """An explicitly incompatible build cannot become 'unknown' then bypass scope."""
    raw = _map(_map(record.get("upstream")).get("raw"))
    context = _map(record.get("context"))
    target = _map(raw.get("target"))
    entity = _map(record.get("object"))
    containers = (
        record,
        context,
        raw,
        target,
        entity,
        _map(raw.get("genomicLocation")),
        _map(target.get("genomicLocation")),
        _map(entity.get("genomicLocation")),
    )
    builds = [
        c[k]
        for c in containers
        for k in ("genome_build", "genomeBuild", "regulatory_genome_build")
        if c.get(k) not in (None, "")
    ]
    return all(isinstance(b, str) and b.strip().casefold() in {"grch38", "hg38"} for b in builds)


def _record_variant_identity_valid(record: Mapping) -> bool:
    """Verify explicit matched-member anchors; raw lead SNPs may legitimately differ."""
    if record.get("module") != "locus.qtl":
        return True
    context = _map(record.get("context"))
    raw = _map(_map(record.get("upstream")).get("raw"))
    subject = _map(record.get("subject")).get("id")
    member = _map(raw.get("matchedLocusMember"))
    explicit_ids = [
        x for x in (context.get("qtl_member_variant_id"), member.get("variantId")) if x is not None
    ]
    if any(not isinstance(x, str) or x != subject for x in explicit_ids):
        return False
    point = _variant(subject)
    if (
        point
        and context.get("qtl_member_chromosome") is not None
        and _chromosome(context["qtl_member_chromosome"]) != point[0]
    ):
        return False
    return not (
        point
        and context.get("qtl_member_position") is not None
        and _integer(context["qtl_member_position"]) != point[1]
    )


def discovery_locus_id(record: Mapping) -> str:
    context = _map(record.get("context"))
    raw = _map(_map(record.get("upstream")).get("raw"))
    for source, keys in (
        (context, ("study_locus_id", "credible_set_id")),
        (raw, ("studyLocusId", "credibleSetId")),
    ):
        for key in keys:
            if source.get(key):
                return str(source[key])
    return "variant:" + str(_map(record.get("subject")).get("id") or "unknown-locus")


def coding_terms(record: Mapping) -> list[str]:
    """Read gene-specific terms only, never another transcript's annotations."""
    if not record_target_identity_valid(record):
        return []
    raw = _map(_map(record.get("upstream")).get("raw"))
    found: set[str] = set()

    def scalars(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from scalars(item)
        elif isinstance(value, Mapping):
            for key in ("id", "label", "term"):
                yield from scalars(value.get(key))

    for source in (record, _map(record.get("context")), _map(record.get("measurements")), raw):
        for key, value in source.items():
            if str(key).replace("_", "").casefold() in _CONSEQUENCE_KEYS:
                found.update(term for term in scalars(value) if term in CODING_TERMS)
    return sorted(found)


def _locations(record: Mapping) -> list[tuple[str, int, int]]:
    if not record_target_identity_valid(record) or not _record_build_compatible(record):
        return []
    raw = _map(_map(record.get("upstream")).get("raw"))
    context = _map(record.get("context"))
    sources = [
        raw.get("genomicLocation"),
        _map(raw.get("target")).get("genomicLocation"),
        _map(record.get("object")).get("genomicLocation"),
    ]
    if context.get("gene_start") is not None:
        sources.append(
            {
                "chromosome": context.get("gene_chromosome", context.get("chromosome")),
                "start": context.get("gene_start"),
                "end": context.get("gene_end"),
            }
        )
    result = []
    for value in sources:
        location = _map(value)
        chrom = _chromosome(location.get("chromosome"))
        start, end = _integer(location.get("start")), _integer(location.get("end"))
        if chrom and start and end and end >= start:
            result.append((chrom, start, end))
    return result


def _record_key(record: Mapping) -> str:
    # Only our annotation is omitted; upstream derived flags affect eligibility.
    value = dict(record)
    value["derived"] = {
        key: val for key, val in _map(record.get("derived")).items() if key != "candidate_scope"
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def evidence_candidate_scope(payload: Mapping) -> Mapping:
    return _map(_map(payload.get("derived")).get("candidate_scope"))


def _exact_enhancer_overlap(record: Mapping, points: Mapping) -> bool:
    context = _map(record.get("context"))
    variant = points.get(_map(record.get("subject")).get("id"))
    start, end = _integer(context.get("regulatory_start")), _integer(context.get("regulatory_end"))
    return bool(
        variant
        and start
        and end
        and start <= variant[1] <= end
        and _chromosome(context.get("regulatory_chromosome")) == variant[0]
        and context.get("overlap_variant_id") == _map(record.get("subject")).get("id")
    )


def discover_candidates(
    records: Sequence[Mapping],
    *,
    variant_ids: Sequence[str] = (),
    explicit_candidates: Sequence[str] = (),
    window_bp: int = LOCAL_WINDOW_BP,
    relevant_terms: Sequence[str] = (),
    expression_z_threshold: float = DEFAULT_EXPRESSION_Z_THRESHOLD,
    expression_fold_threshold: float = DEFAULT_EXPRESSION_FOLD_THRESHOLD,
    expression_context_selectors=None,
    expression_context_groups=None,
    expression_context_mapping=None,
    expression_detection_rules=None,
    expression_retrieval: Mapping[str, str] | None = None,
    max_l2g_per_locus: int = MAX_DISCOVERED_L2G_TARGETS_PER_LOCUS,
    molecular_context: Mapping | None = None,
) -> dict[str, Any]:
    """Return only first-tier genes, without positional filling or a 40-gene cap.

    Explicit candidates constrain the observed pool, then receive the same
    first-tier eligibility check. Missing coordinates do not establish trans.
    """
    for name, value in (
        ("window_bp", window_bp),
        ("max_l2g_per_locus", max_l2g_per_locus),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if name == "window_bp" else 1)
        ):
            raise ValueError(f"{name} is invalid")
    expression_options = {
        "relevant_terms": relevant_terms,
        "threshold": expression_z_threshold,
        "fold_threshold": expression_fold_threshold,
        "context_selectors": expression_context_selectors,
        "context_groups": expression_context_groups,
        "context_mapping": (
            expression_context_mapping if expression_context_mapping is not None
            else (molecular_context or {}).get("biosample_mapping")
        ),
        "detection_rules": expression_detection_rules,
    }
    expression_policy = expression_specificity([], **expression_options)
    explicit_pool = {g.strip() for g in explicit_candidates if isinstance(g, str) and g.strip()}
    variants = tuple(
        sorted(
            set(
                variant_ids
                or (
                    str(_map(record.get("subject")).get("id"))
                    for record in records
                    if _map(record.get("subject")).get("type") == "variant"
                )
            )
        )
    )
    points = {v: _variant(v) for v in variants if _variant(v)}
    locations: dict[str, set] = defaultdict(set)
    distances: dict[tuple[str, str], list] = defaultdict(list)
    by_gene: dict[str, list] = defaultdict(list)
    expression_by_gene: dict[str, list] = defaultdict(list)
    invalid_expression_genes = set()
    for record in records:
        for gene in target_ids(record):
            if explicit_candidates and gene not in explicit_pool:
                continue
            locations[gene].update(_locations(record))
            if record.get("module") == "target.expression":
                if record_target_identity_valid(record):
                    expression_by_gene[gene].append(record)
                else:
                    invalid_expression_genes.add(gene)
            if str(record.get("module", "")).startswith("locus."):
                by_gene[gene].append(record)
            if (
                record.get("module") == "locus.variant_consequence"
                and record_target_identity_valid(record)
                and _record_build_compatible(record)
            ):
                variant = str(_map(record.get("subject")).get("id"))
                measurements = _map(record.get("measurements"))
                raw = _map(_map(record.get("upstream")).get("raw"))
                distance = _integer(
                    measurements.get("distance_from_footprint", raw.get("distanceFromFootprint"))
                )
                basis = "transcript_footprint"
                if distance is None:
                    tss = _integer(
                        measurements.get("distance_from_tss", raw.get("distanceFromTss")),
                        signed=True,
                    )
                    distance = abs(tss) if tss is not None else None
                    basis = "transcript_tss"
                if distance is not None and variant in points:
                    distances[gene, variant].append((distance, basis))

    def locality(gene, record):
        if not record_target_identity_valid(record):
            return "conflicting_target_identity", None, "record_target_mismatch"
        if not _record_build_compatible(record):
            return "incompatible_genome_build", None, "incompatible_genome_build"
        if not _record_variant_identity_valid(record):
            return "conflicting_variant_identity", None, "matched_member_mismatch"
        subject = _map(record.get("subject"))
        associated = str(subject.get("id")) if subject.get("type") == "variant" else None
        anchors = (
            {associated: points[associated]}
            if associated in points
            else (points if not associated else {})
        )
        if associated and associated not in points:
            return "unmatched_variant", None, "no_selected_variant_anchor"
        if len(locations[gene]) > 1:
            return "conflicting_coordinates", None, "inconsistent_gene_intervals"
        if locations[gene] and anchors:
            chrom, start, end = next(iter(locations[gene]))
            same = [max(start - pos, pos - end, 0) for c, pos in anchors.values() if c == chrom]
            if not same:
                return "cross_chromosome", None, "gene_interval"
            distance = min(same)
            return (
                ("local" if distance <= window_bp else "same_chromosome_distal"),
                distance,
                "gene_interval",
            )
        available = [item for variant in anchors for item in distances[gene, variant]]
        if available:
            distance, basis = min(available)
            return ("local" if distance <= window_bp else "same_chromosome_distal"), distance, basis
        return "unknown_coordinates", None, "no_gene_interval_or_transcript_distance"

    rows = {}
    for gene in set(by_gene) | explicit_pool:
        by_gene[gene].extend(expression_by_gene[gene])
    l2g: dict[str, dict[str, float]] = defaultdict(dict)
    l2g_identity_issues = []
    l2g_releases = defaultdict(set)
    for gene in sorted(set(by_gene) | explicit_pool):
        sources = []
        symbol = None
        expression = expression_specificity(expression_by_gene[gene], **expression_options)
        retrieval_status = (expression_retrieval or {}).get(gene)
        if gene in invalid_expression_genes:
            expression = {**expression, "qualified": False, "status": "conflicting_target_identity"}
            retrieval_status = "incomplete"
        if (
            expression_policy["context_definition"]["selectors"]
            and retrieval_status is not None
            and retrieval_status not in {"returned", "no_data"}
        ):
            expression = {**expression, "qualified": False, "status": "incomplete_retrieval"}
        for record in by_gene[gene]:
            module = str(record.get("module", ""))
            scope, distance, basis = locality(gene, record)
            molecular_match = (molecular_context_match(record, molecular_context)
                               if molecular_context is not None and module == "locus.qtl" else None)
            terms = coding_terms(record) if module == "locus.variant_consequence" else []
            derived = _map(record.get("derived"))
            veto_direct = bool(
                derived.get("not_direct_locus_to_gene_evidence") or derived.get("not_l2g_or_qtl")
            )
            # A remote coordinate contradicts a coding annotation. Splice terms
            # can be just outside a footprint; do not require distance == zero.
            coding = (
                bool(terms)
                and scope in {"local", "unknown_coordinates"}
                and not veto_direct
                and _map(record.get("subject")).get("id") in points
            )
            if coding and distance is not None:
                splice = bool(
                    set(terms)
                    & {
                        "splice_acceptor_variant",
                        "splice_donor_variant",
                        "SO:0001574",
                        "SO:0001575",
                    }
                )
                coding = distance <= (2 if splice else 0)
            enhancer = (
                module == "locus.enhancer_to_gene"
                and not veto_direct
                and record_target_identity_valid(record)
                and _record_build_compatible(record)
                and _exact_enhancer_overlap(record, points)
                and scope
                not in {"cross_chromosome", "conflicting_coordinates", "unmatched_variant"}
            )
            if coding:
                reason, tier = "explicit_coding_or_splice", "direct"
            elif enhancer:
                reason, tier = "exact_variant_enhancer_link", "direct"
            elif (
                scope == "local"
                and module in MOLECULAR_MODULES
                and not veto_direct
                and _map(record.get("subject")).get("id") in points
                and (molecular_context is None or module == "locus.enhancer_to_gene"
                     or module == "locus.qtl" and molecular_match["matched"]
                     and molecular_anchor_verified(record))
            ):
                reason, tier = "local_molecular_link", "direct"
            elif (
                module == "locus.l2g"
                and record_target_identity_valid(record)
                and _map(record.get("subject")).get("id") in points
            ):
                reason, tier = "l2g_top5_discovery", "l2g"
            elif module == "target.expression" and expression["qualified"]:
                reason, tier = "context_specific_expression", "expression"
            elif scope == "local" and module == "locus.variant_consequence":
                reason, tier = "local_positional_annotation_only", "positional"
            else:
                reason, tier = ("upstream_context_only" if veto_direct else scope), "context"
            key = _record_key(record)
            sources.append(
                {
                    "record_key": key,
                    "module": module,
                    "relation": record.get("relation"),
                    **({"molecular_context": molecular_match,
                        "molecular_anchor_verified": molecular_anchor_verified(record),
                        "molecular_supplementary_eligible": (
                            scope == "local" and not veto_direct
                            and _map(record.get("subject")).get("id") in points
                            and molecular_anchor_verified(record))}
                       if molecular_match is not None else {}),
                    "locus_id": discovery_locus_id(record),
                    "variant_id": _map(record.get("subject")).get("id")
                    if _map(record.get("subject")).get("type") == "variant"
                    else None,
                    "source": _map(record.get("provenance")).get("source"),
                    "data_release": _map(record.get("provenance")).get("data_release"),
                    "upstream_evidence_id": _map(record.get("provenance")).get("evidence_id"),
                    "source_artifacts": _map(record.get("provenance")).get("source_files", []),
                    "evidence_role": _map(record.get("provenance")).get("evidence_role"),
                    "scope": scope,
                    "distance_bp": distance,
                    "coordinate_basis": basis,
                    "coding_terms": terms,
                    "tier": tier,
                    "reason": reason,
                }
            )
            label = _map(record.get("object")).get("label")
            if label:
                symbol = min(symbol, str(label)) if symbol else str(label)
            if tier == "l2g":
                raw = _map(_map(record.get("upstream")).get("raw"))
                context = _map(record.get("context"))
                locus_ids = {
                    value.strip()
                    for source, keys in (
                        (context, ("study_locus_id", "credible_set_id")),
                        (raw, ("studyLocusId", "credibleSetId")),
                    )
                    for name in keys
                    if isinstance((value := source.get(name)), str) and value.strip()
                }
                if len(locus_ids) != 1:
                    l2g_identity_issues.append(
                        {
                            "gene_id": gene,
                            "record_key": key,
                            "issue": "missing_study_locus_identity"
                            if not locus_ids
                            else "conflicting_study_locus_identity",
                        }
                    )
                release = _map(record.get("provenance")).get("data_release")
                if isinstance(release, str) and release.strip():
                    l2g_releases[discovery_locus_id(record)].add(release.strip())
                value = _map(record.get("measurements")).get("l2g_score")
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and _finite_score(value)
                ):
                    group = l2g[discovery_locus_id(record)]
                    group[gene] = max(group.get(gene, -math.inf), value)
        unique = {s["record_key"]: s for s in sources}
        rows[gene] = {
            "gene_id": gene,
            "symbol": symbol,
            "sources": [unique[k] for k in sorted(unique)],
            "expression_specificity": expression,
            "expression_retrieval_status": (expression_retrieval or {}).get(
                gene, "returned" if expression_by_gene[gene] else "not_retrieved"
            )
            if gene not in invalid_expression_genes
            else "incomplete",
            "gene_intervals": [
                {"chromosome": chrom, "start": start, "end": end, "genome_build": "GRCh38"}
                for chrom, start, end in sorted(locations[gene])
            ],
        }

    selected_l2g = set()
    l2g_discovery_ranks = {}
    for locus in sorted(l2g):
        ranked_genes = sorted(l2g[locus], key=lambda g: (-l2g[locus][g], g))
        selected_l2g.update(ranked_genes[:max_l2g_per_locus])
        l2g_discovery_ranks[locus] = {g: rank for rank, g in enumerate(ranked_genes, 1)}

    def has_tier(gene, tier):
        return any(source["tier"] == tier for source in rows[gene]["sources"])

    direct = {g for g in rows if has_tier(g, "direct")}
    expression_genes = {g for g in rows if has_tier(g, "expression")}
    first_tier = direct | selected_l2g | expression_genes
    # Top5 has no distance/expression gate, and is never removed by a global cap.
    selected = tuple(sorted(first_tier))
    for gene, row in rows.items():
        kept = gene in selected
        eligible = gene in first_tier
        row["l2g_top5_loci"] = [
            {"study_locus_id": locus, "discovery_rank": ranks[gene]}
            for locus, ranks in l2g_discovery_ranks.items()
            if gene in ranks and ranks[gene] <= max_l2g_per_locus
        ]
        modules = {s["module"] for s in row["sources"]}
        scopes = {s["scope"] for s in row["sources"]}
        # Presence means returned in this exact discovery call, regardless of
        # eligibility. Missing annotation is not a biological absence claim.
        coding_snps = [
            s
            for s in row["sources"]
            if s["coding_terms"]
            and s["variant_id"] in points
            and re.fullmatch(
                r"(?:chr)?(?:[0-9]+|X|Y|MT)[_:][0-9]+[_:][ACGTN][_:][ACGTN]",
                str(s["variant_id"]),
                re.IGNORECASE,
            )
        ]
        row["evidence_returned"] = {
            "l2g": "locus.l2g" in modules,
            "molecular_qtl": bool(modules & {"locus.qtl", "locus.colocalisation"}),
            "e2g": "locus.enhancer_to_gene" in modules,
            "coding_snp": bool(coding_snps),
            "expression": bool(expression_by_gene[gene]),
            "context_specific_expression": row["expression_specificity"]["qualified"],
        }
        row["coding_snp_relation_verified"] = any(s["tier"] == "direct" for s in coding_snps)
        row["coordinate_status"] = (
            "conflicting_gene_intervals"
            if len(locations[gene]) > 1
            else "gene_interval_available"
            if locations[gene]
            else "transcript_distance_only"
            if any(s["coordinate_basis"].startswith("transcript_") for s in row["sources"])
            else "unavailable"
        )
        row["locality_verified"] = "local" in scopes
        row["direct_relation_verified"] = has_tier(gene, "direct")
        reasons = sorted(
            {
                source["reason"]
                for source in row["sources"]
                if source["tier"] in {"direct", "expression"}
            }
        )
        if gene in selected_l2g:
            reasons.append("l2g_top5_unconditional")
        row.update(
            {
                "selected": kept,
                "candidate_tier": "first" if kept else None,
                "eligible_before_global_cap": eligible,
                "l2g_per_locus_selected": gene in selected_l2g,
                "qualifying_reasons": reasons,
                "disposition": "effector_candidate"
                if kept
                else "downstream_context"
                if scopes & {"cross_chromosome", "same_chromosome_distal"}
                else "context_only",
                "selection_reason": ";".join(reasons) if kept else "no_first_tier_source",
            }
        )
    policy = {
        "version": "first-tier-context-molqtl-v4" if molecular_context is not None else POLICY_VERSION,
        "window_bp": window_bp,
        "max_targets": None,
        "only_first_tier": True,
        "expression_z_threshold": expression_z_threshold,
        "expression_z_role": "support_only_never_admits",
        "expression_tau_role": "support_only_never_admits",
        "expression_method": expression_policy["method"],
        "expression_fold_threshold": expression_fold_threshold,
        "expression_fold_comparison": ">=",
        "expression_detection_rules": expression_policy["detection_rules"],
        "expression_context": expression_policy["context_definition"],
        "l2g_top5_unconditional": True,
        "l2g_role": "shortlist_admission_only",
        "l2g_visible_to_final_ranker": False,
        "max_l2g_per_locus": max_l2g_per_locus,
        "distance_basis": "gene_footprint_then_tss",
        "selection": "union(direct_locus_links, per_study_locus_L2G_top5, context_specific_expression)",
        "unknown_coordinates": "not_a_veto_for_L2G_top5_or_context_expression_or_explicit_coding_or_exact_enhancer",
        **({"molecular_context": dict(molecular_context),
            "colocalisation_role": "post_shortlist_details_never_independent_admission"}
           if molecular_context is not None else {}),
    }
    fingerprint = hashlib.sha256(
        json.dumps(sorted(set(selected)), separators=(",", ":")).encode()
    ).hexdigest()
    audit = {
        "policy": policy,
        "mode": "explicit_pool_first_tier" if explicit_candidates else "discovery",
        "input_candidate_ids": list(explicit_candidates),
        "ignored_outside_explicit_pool": sorted(
            {
                g
                for record in records
                for g in target_ids(record)
                if explicit_candidates and g not in explicit_pool
            }
        ),
        "context_terms": sorted(set(relevant_terms)),
        "expression_retrieval_complete": not expression_policy["context_definition"]["selectors"]
        or all(
            g["expression_retrieval_status"] in {"returned", "no_data"}
            and g["expression_specificity"]["status"] != "incomplete_retrieval"
            for g in rows.values()
        ),
        "expression_assessment_complete": all(
            g["expression_specificity"].get(
                "assessment_complete",
                g["expression_specificity"]["status"] not in {
                    "incomplete_retrieval",
                    "unresolved_expression_comparisons",
                    "conflicting_target_identity",
                    "context_mapping_unresolved",
                },
            )
            for g in rows.values()
        ),
        "variant_ids": list(variants),
        "candidate_ids": list(selected),
        "candidate_fingerprint": fingerprint,
        "genes": list(rows.values()),
        "shortlist_size": len(selected),
        "strategy": POLICY_VERSION,
        "independent_loci": len(l2g),
        "direct_evidence_candidates": len(direct),
        "selected_direct_candidates": len(set(selected) & set(direct)),
        "selected_l2g_candidates": len(set(selected) & selected_l2g),
        "selected_expression_candidates": len(set(selected) & expression_genes),
        "positional_only_selected": 0,
        "positional_context_candidates": sum(has_tier(g, "positional") for g in selected),
        "context_gene_count": sum(not row["selected"] for row in rows.values()),
        "explicit_input_limits_pool_before_first_tier_selection": bool(explicit_candidates),
        "evidence_presence_definition": {
            "molecular_qtl": "locus.qtl or locus.colocalisation returned for this target",
            "coding_snp": "explicit gene-specific protein-altering or canonical splice SO term on an input SNP; scores and distance alone do not qualify",
            "absence": "no qualifying returned annotation, not proof of biological absence",
        },
    }
    # Authoritative empty retrieval completes source enumeration, not scientific
    # assessment. A no-data flag cannot conceal any returned unresolved profile.
    audit["expression_discovery_complete"] = (
        audit["expression_retrieval_complete"] and all(
            gene["expression_specificity"].get("assessment_complete", False)
            or (
                gene["expression_retrieval_status"] == "no_data"
                and not expression_by_gene[gene["gene_id"]]
            )
            for gene in rows.values()
        )
    )
    audit["evidence_integrity_issues"] = [
        {"gene_id": g["gene_id"], "record_key": s["record_key"], "issue": s["scope"]}
        for g in rows.values()
        for s in g["sources"]
        if s["scope"]
        in {
            "conflicting_target_identity",
            "conflicting_variant_identity",
            "incompatible_genome_build",
        }
    ]
    audit["evidence_integrity_complete"] = not audit["evidence_integrity_issues"]
    audit["l2g_grouping_issues"] = sorted(
        {json.dumps(issue, sort_keys=True): issue for issue in l2g_identity_issues}.values(),
        key=lambda issue: (issue["gene_id"], issue["record_key"]),
    ) + [
        {"study_locus_id": locus, "issue": "mixed_data_releases", "releases": sorted(releases)}
        for locus, releases in sorted(l2g_releases.items())
        if len(releases) > 1
    ]
    audit["l2g_grouping_complete"] = not audit["l2g_grouping_issues"]
    if molecular_context is not None:
        molecular_rows = [source for gene in rows.values() for source in gene["sources"]
                          if source["module"] == "locus.qtl" and source["scope"] == "local"]
        audit["molecular_context_complete"] = bool(molecular_context.get("defined")) and all(
            source.get("molecular_context", {}).get("status") in {"matched", "not_matched"}
            and source.get("molecular_anchor_verified") is True
            for source in molecular_rows)
        audit["molecular_context_status"] = ("complete" if audit["molecular_context_complete"]
                                              else "context_undefined" if not molecular_context.get("defined")
                                              else "molecular_anchor_unverified" if any(
                                                  not source.get("molecular_anchor_verified") for source in molecular_rows)
                                              else "biosample_unavailable")
        audit["molecular_detail_scope"] = [
            {"gene_id": gene["gene_id"], "study_locus_id": source["locus_id"],
             "variant_id": source["variant_id"]}
            for gene in rows.values() if gene["gene_id"] in selected
            for source in gene["sources"]
            if source["module"] == "locus.qtl" and source["tier"] == "direct"
            and not source["locus_id"].startswith("variant:")
        ]
        supplementary = select_supplementary_molecular(
            {_record_key(record): record for record in records}, audit["genes"],
            max_biosamples=parameter("ranking.supplementary_molecular_biosamples_per_gene"),
            context_sha256=molecular_context.get("context_sha256"))
        audit["supplementary_molecular_selection"] = supplementary
        audit["supplementary_molecular_scope"] = [
            {key: row[key] for key in ("gene_id", "study_locus_id", "variant_id")}
            for row in supplementary["selected"]]
    audit["audit_fingerprint"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return audit


def annotate_candidate_scope(records: Sequence[Mapping], audit: Mapping) -> tuple[Mapping, ...]:
    """Retain all evidence and attach deterministic, score-free evidence roles.

    Expression profiles are copied from the already evaluated candidate audit
    into the expression evidence row so the ranking model receives the same
    tissue/cell calculation that was used for shortlist assessment.  L2G
    remains audit/admission metadata and is deliberately not attached as a
    ranking score.
    """
    sources = {
        source["record_key"]: source for gene in audit["genes"] for source in gene["sources"]
    }
    genes = {gene["gene_id"]: gene for gene in audit.get("genes", ())}
    supplementary_keys = {row["record_key"] for row in audit.get(
        "supplementary_molecular_selection", {}).get("selected", ())}
    result = []
    for record in records:
        source = sources.get(_record_key(record))
        if source is None:
            if target_ids(record) and not record_target_identity_valid(record):
                source = {
                    "scope": "conflicting_target_identity",
                    "reason": "record_target_mismatch",
                    "tier": "context",
                    "module": record.get("module", ""),
                }
            elif (
                audit.get("input_candidate_ids")
                and target_ids(record)
                and not set(target_ids(record)).issubset(audit["input_candidate_ids"])
            ):
                source = {
                    "scope": "outside_explicit_pool",
                    "reason": "outside_explicit_pool",
                    "tier": "context",
                    "module": record.get("module", ""),
                }
            else:
                result.append(record)
                continue
        role = {
            "scope": source["scope"],
            "reason": source["reason"],
            "direct_locus_support": source["tier"] == "direct",
            "context_only": source["scope"]
            in {"conflicting_target_identity", "outside_explicit_pool"}
            or source["tier"] == "context"
            and source["module"].startswith("locus."),
        }
        if _record_key(record) in supplementary_keys:
            role.update(reason="supplementary_noncontext_molecular_link",
                        direct_locus_support=True, context_only=False)
        derived = {**_map(record.get("derived")), "candidate_scope": role}
        if record.get("module") == "target.expression":
            gene = next((gene_id for gene_id in target_ids(record) if gene_id in genes), None)
            expression = (genes.get(gene, {}).get("expression_specificity") or {}) if gene else {}
            profiles = expression.get("profiles")
            if isinstance(profiles, Mapping):
                # Global coverage and method definitions are scientific context,
                # not an admission score. Carry them beside the two measurement
                # axes so serialization cannot detach partial results from their
                # missingness, detection, and frozen-context limitations.
                assessment_keys = (
                    "method", "method_reference", "fold_threshold", "fold_comparison",
                    "z_role", "tau_role", "context_terms", "context_definition",
                    "detection_rules", "retrieval_complete", "assessment_complete",
                    "context_mapping_status", "context_coverage_complete",
                    "unassessed_context_selectors", "unobserved_context_selectors",
                    "evaluated_panel_count", "descriptive_panel_count",
                    "rejected_row_count", "invalid_metadata_patterns",
                )
                assessment = {key: expression[key] for key in assessment_keys if key in expression}
                if "threshold" in expression:
                    assessment["supporting_z_threshold"] = expression["threshold"]
                assessment["profile_contract"] = "expression-assessment-v1"
                derived.update(
                    expression_profiles={**profiles, "assessment": assessment},
                    expression_profile_contract="expression-profile-v1",
                )
        result.append(
            {**record, "derived": derived}
        )
    return tuple(result)


def annotate_molecular_ranking_scope(records: Sequence[Mapping], audit: Mapping) -> tuple[Mapping, ...]:
    """Apply one context/cis rule to positive and negative molecular facts alike.

    Initial locus context can wrap a QTL member as study/fine-mapping metadata.
    Only context-matched or explicitly frozen supplementary members may return
    through these aliases; candidate membership alone never authorizes a row.
    Coding, E2G, LD and genuine GWAS study records are outside this filter.
    """
    if "molecular_context" not in audit.get("policy", {}):
        return tuple(records)
    from .qtl import native_qtl_study_type
    eligible = {(gene["gene_id"], source["locus_id"], source["variant_id"])
                for gene in audit.get("genes", ()) if gene.get("selected")
                for source in gene.get("sources", ())
                if source.get("module") == "locus.qtl" and source.get("tier") == "direct"}
    matched_keys = {source["record_key"] for gene in audit.get("genes", ()) if gene.get("selected")
                    for source in gene.get("sources", ()) if source.get("module") == "locus.qtl"
                    and source.get("tier") == "direct"}
    source_assessments = {source["record_key"]: source.get("molecular_context", {})
                          for gene in audit.get("genes", ()) if gene.get("selected")
                          for source in gene.get("sources", ()) if source.get("module") == "locus.qtl"}
    receipt = audit.get("supplementary_molecular_selection", {})
    supplements = {(row["gene_id"], row["study_locus_id"], row["variant_id"]): row
                   for row in receipt.get("selected", ())}
    supplementary_keys = {row["record_key"] for row in supplements.values()}
    eligible.update(supplements)
    members = {(locus, variant) for _gene, locus, variant in eligible}
    output = []
    counts = defaultdict(int)
    for record in records:
        module = record.get("module")
        if module not in {"locus.qtl", "locus.study", "locus.credible_set", "locus.fine_mapping"}:
            output.append(record)
            continue
        raw = _map(_map(record.get("upstream")).get("raw"))
        context = _map(record.get("context"))
        if module != "locus.qtl" and native_qtl_study_type(raw.get("studyType") or context.get("study_type")) is None:
            output.append(record)
            continue
        subject = _map(record.get("subject"))
        variant = (subject.get("id") if subject.get("type") == "variant" else
                   _map(raw.get("matchedLocusMember")).get("variantId") or raw.get("queryVariantId"))
        locus = subject.get("id") if subject.get("type") == "credible_set" else discovery_locus_id(record)
        genes = target_ids(record)
        allowed = ((genes[0], locus, variant) in eligible if len(genes) == 1 else
                   (locus, variant) in members if not genes else False)
        supplement = supplements.get((genes[0], locus, variant)) if len(genes) == 1 else next(
            (row for (_gene, item_locus, item_variant), row in supplements.items()
             if item_locus == locus and item_variant == variant), None)
        if module == "locus.qtl" and _record_key(record) in matched_keys:
            supplement = None
        elif supplement is not None and module == "locus.qtl":
            allowed = allowed and _record_key(record) in supplementary_keys
        if allowed:
            plan_sha = audit.get("policy", {}).get("molecular_context", {}).get("context_sha256")
            assessment = (supplement.get("context_assessment") if supplement else
                frozen_context_assessment(source_assessments.get(_record_key(record), {}), plan_sha))
            selection_context = {
                "context_match_class": "other_reported_context" if supplement else "matched",
                "molecular_evidence_tier": "supplementary" if supplement else "context_related",
                "context_sha256": plan_sha,
            }
            if supplement is not None or _record_key(record) in source_assessments:
                selection_context["molecular_context_assessment"] = assessment
            if supplement is not None:
                selection_context.update(other_context_supplement=True,
                    molecular_selection="supplementary_biosample",
                    selection_receipt_sha256=receipt.get("receipt_sha256"),
                    supplementary_selection={key: supplement[key] for key in
                        ("biosample", "qtl_type", "member_p", "selection_basis")})
            output.append({**record, "context": {**context, **selection_context}})
            continue
        counts[str(module)] += 1
        output.append({**record, "context": {**context, "rank_scope": "context_only",
                      "molecular_scope_basis": "context_and_locality_required_before_ranking"}})
    # The caller owns the mutable audit; raw source rows remain unchanged.
    if isinstance(audit, dict):
        audit["molecular_context_only_record_counts"] = dict(sorted(counts.items()))
    return tuple(output)


def candidate_pool_rows(audit: Mapping) -> list[dict[str, Any]]:
    """Export the entire discovery pool, selected first then symbol/ID order.

    This is presentation of the executor's audit, never a second discovery pass.
    Evidence presence includes excluded/context rows and has no ranking column.
    """
    from .evidence_store import metadata_summary_related
    stored = metadata_summary_related(audit, "candidate_pool")
    if stored is not None:
        return stored
    if audit.get("mode") == "gene_centric_window":
        return _gene_window_pool_rows(audit)
    rows = []
    for gene in sorted(
        audit["genes"],
        key=lambda g: (
            not g["selected"],
            str(g["symbol"] or g["gene_id"]).casefold(),
            g["gene_id"],
        ),
    ):
        sources = gene["sources"]
        distances = [s["distance_bp"] for s in sources if s["distance_bp"] is not None]
        rows.append(
            {
                "gene_id": gene["gene_id"],
                "symbol": gene["symbol"] or gene["gene_id"],
                "selected": gene["selected"],
                "display_weight": "bold" if gene["selected"] else "normal",
                **{
                    f"{key}_returned": bool(value)
                    for key, value in gene["evidence_returned"].items()
                },
                "disposition": gene["disposition"],
                "selection_reason": gene["selection_reason"],
                "candidate_tier": gene["candidate_tier"],
                "l2g_top5_loci": ";".join(item["study_locus_id"] for item in gene["l2g_top5_loci"]),
                "expression_status": gene["expression_specificity"]["status"],
                "expression_threshold_pass": True
                if gene["expression_specificity"]["qualified"]
                else False
                if gene["expression_specificity"]["status"] == "evaluated_below_threshold"
                else None,
                "expression_retrieval_status": gene["expression_retrieval_status"],
                "context_expression_max_z": max(
                    (
                        sample["z"]
                        for group in gene["expression_specificity"]["groups"]
                        for sample in group["matched_biosamples"]
                        if sample["z"] is not None
                    ),
                    default=None,
                ),
                "expression_z_threshold": gene["expression_specificity"]["threshold"],
                "expression_z_role": "support_only_never_admits",
                "expression_fold_threshold": gene["expression_specificity"]["fold_threshold"],
                "expression_method": gene["expression_specificity"]["method"],
                "context_expression_max_fold": max(
                    [
                        s["fold_enrichment"]
                        for g in gene["expression_specificity"]["groups"]
                        for s in g["matched_biosamples"]
                        if s["fold_enrichment"] is not None
                    ]
                    + [
                        s["fold_enrichment"]
                        for g in gene["expression_specificity"]["groups"]
                        for s in g["group_enrichment"]
                        if s["fold_enrichment"] is not None
                    ],
                    default=None,
                ),
                "context_expression_zero_background": any(
                    s["zero_background"]
                    for g in gene["expression_specificity"]["groups"]
                    for s in (*g["matched_biosamples"], *g["group_enrichment"])
                ),
                "expression_tau_max": max(
                    (
                        g["tau"]
                        for g in gene["expression_specificity"]["groups"]
                        if g["tau"] is not None
                    ),
                    default=None,
                ),
                "expression_tau_context_peak_max": max(
                    (
                        g["tau"]
                        for g in gene["expression_specificity"]["groups"]
                        if g["tau"] is not None and g["tau_peak_matches_context"]
                    ),
                    default=None,
                ),
                "expression_qualifying_classes": ";".join(
                    sorted(
                        (
                            {"single_enriched"}
                            if any(
                                s["above_threshold"]
                                for g in gene["expression_specificity"]["groups"]
                                for s in g["matched_biosamples"]
                            )
                            else set()
                        )
                        | (
                            {"group_enriched"}
                            if any(
                                s["qualified"]
                                for g in gene["expression_specificity"]["groups"]
                                for s in g["group_enrichment"]
                            )
                            else set()
                        )
                    )
                )
                if gene["expression_specificity"]["qualified"]
                else "",
                "coordinate_status": gene["coordinate_status"],
                "locality_verified": gene["locality_verified"],
                "direct_relation_verified": gene["direct_relation_verified"],
                "coding_snp_relation_verified": gene["coding_snp_relation_verified"],
                "minimum_source_distance_bp": min(distances) if distances else None,
                "source_modules": ";".join(sorted({s["module"] for s in sources})),
                "source_scopes": ";".join(sorted({s["scope"] for s in sources})),
                "source_reasons": ";".join(sorted({s["reason"] for s in sources})),
                "source_record_keys": ";".join(s["record_key"] for s in sources),
                "candidate_fingerprint": audit["candidate_fingerprint"],
                "audit_fingerprint": audit["audit_fingerprint"],
                "policy_version": audit["policy"]["version"],
                "candidate_mode": audit["mode"],
            }
        )
    return rows


def _gene_window_pool_rows(audit: Mapping) -> list[dict[str, Any]]:
    """Every lead-window protein-coding gene is a candidate; no shortlist or L2G fields apply."""
    rows = []
    for gene in sorted(audit["genes"], key=lambda g: (str(g["symbol"] or g["gene_id"]).casefold(), g["gene_id"])):
        sources = gene["sources"]
        distances = [s["distance_bp"] for s in sources if s["distance_bp"] is not None]
        rows.append({
            "gene_id": gene["gene_id"],
            "symbol": gene["symbol"] or gene["gene_id"],
            **{f"{key}_returned": bool(value) for key, value in gene["evidence_returned"].items()},
            "expression_retrieval_status": gene["expression_retrieval_status"],
            "direct_relation_verified": gene["direct_relation_verified"],
            "coding_snp_relation_verified": gene["coding_snp_relation_verified"],
            "minimum_source_distance_bp": min(distances) if distances else None,
            "source_modules": ";".join(sorted({s["module"] for s in sources})),
            "source_scopes": ";".join(sorted({s["scope"] for s in sources})),
            "source_reasons": ";".join(sorted({s["reason"] for s in sources})),
            "source_record_keys": ";".join(s["record_key"] for s in sources),
            "candidate_fingerprint": audit["candidate_fingerprint"],
            "audit_fingerprint": audit["audit_fingerprint"],
            "policy_version": audit["policy"]["version"],
            "candidate_mode": audit["mode"],
        })
    return rows


def candidate_audit_summary(audit: Mapping | None) -> dict:
    """Reuse the stored audit view without building the full candidate-pool rows."""
    if not audit:
        return {}
    genes = []
    for gene in audit.get("genes", ()):
        expression = gene.get("expression_specificity") or {}
        genes.append({**{key: value for key, value in gene.items()
                         if key not in {"sources", "expression_specificity"}},
                      "source_count": len(gene.get("sources", ())),
                      "source_details_available": True,
                      "expression_specificity": {
                          **{key: value for key, value in expression.items() if key != "groups"},
                          "group_count": len(expression.get("groups", ())),
                          "group_details_available": True}})
    return {**{key: value for key, value in audit.items() if key != "genes"}, "genes": genes}


def candidate_metadata_summaries(audit: Mapping | None, pool: list) -> dict:
    """Small read views; complete sources and expression details stay in storage.

    Selection flags, policy, fingerprints and the already computed presentation
    fields are preserved. This does not repeat discovery or change its result.
    """
    from .evidence_store import SUMMARY_KEY
    if not audit or not pool or SUMMARY_KEY in audit:
        return {}
    audit_summary = candidate_audit_summary(audit)
    pool_summary = [{**{key: value for key, value in row.items() if key != "source_record_keys"},
                     "source_record_count": row["source_record_keys"].count(";") + 1
                     if row.get("source_record_keys") else 0,
                     "source_record_details_available": True} for row in pool]
    return {"candidate_audit": audit_summary, "candidate_pool": pool_summary or None}
