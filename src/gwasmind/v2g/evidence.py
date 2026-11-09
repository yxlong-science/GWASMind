"""Classify retrieved records into per-gene evidence cards.

Nothing here ranks genes. Each card carries precomputed auxiliary facts
(distances, gene-body overlap), the retrieved records grouped by evidence
category and per-category coverage. The LLM ranks; product cards have no tier.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..configuration import parameter
from .models import ChannelStatus, GeneWindow, LocusVariant, LocusVariantSet, WeightBasis
from .retrieval import RetrievalBundle, _unique_rows

CONSEQUENCE_CLASS = {
    "transcript_ablation": "lof", "splice_acceptor_variant": "splice_core", "splice_donor_variant": "splice_core",
    "stop_gained": "lof", "frameshift_variant": "lof", "stop_lost": "lof", "start_lost": "lof",
    "transcript_amplification": "lof", "missense_variant": "missense", "inframe_insertion": "missense",
    "inframe_deletion": "missense", "protein_altering_variant": "missense",
    "splice_region_variant": "splice_region", "splice_donor_5th_base_variant": "splice_region",
    "splice_donor_region_variant": "splice_region", "splice_polypyrimidine_tract_variant": "splice_region",
    "synonymous_variant": "synonymous", "stop_retained_variant": "synonymous",
    "start_retained_variant": "synonymous", "5_prime_UTR_variant": "utr", "3_prime_UTR_variant": "utr",
    "upstream_gene_variant": "flanking", "downstream_gene_variant": "flanking", "intron_variant": "intron",
    "non_coding_transcript_exon_variant": "noncoding_exon", "non_coding_transcript_variant": "noncoding_exon",
    "NMD_transcript_variant": "intron", "coding_sequence_variant": "missense",
    "incomplete_terminal_codon_variant": "missense", "mature_miRNA_variant": "noncoding_exon",
}
# Orders records within the coding category only; never a gene score.
CLASS_ORDER = {"lof": 0, "splice_core": 0, "missense": 1, "splice_region": 2, "synonymous": 3, "utr": 3,
               "noncoding_exon": 4, "flanking": 5, "intron": 5, "other": 6}
CODING_CLASSES = frozenset({"lof", "splice_core", "missense"})
DETAILED_CODING_CLASSES = frozenset({"lof", "splice_core", "missense", "splice_region", "synonymous", "utr"})
EVIDENCE_CATEGORIES = ("coding", "qtl", "coloc", "e2g", "expression", "constraint", "association", "function")
_STOPWORDS = frozenset({"of", "the", "and", "in", "a", "an"})


def _tokens(value: Any) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", str(value or "").casefold())
    return frozenset(word for word in words if word not in _STOPWORDS)


def context_matches(name: Any, terms: Sequence[str]) -> bool:
    """A biosample matches a context term when either token set contains the other.

    This links organ-level terms to substructures ("liver" and "right lobe of
    liver") without matching unrelated tissues that merely share a stopword.
    """

    tokens = _tokens(name)
    if not tokens:
        return False
    for term in terms:
        wanted = _tokens(term)
        if wanted and (wanted <= tokens or tokens <= wanted):
            return True
    return False


def _context_matcher(context_terms: Sequence[str], biological_context: Mapping[str, Any] | None):
    from ..biological_context import (
        BiologicalContext,
        compile_biosample_context,
        standardize_biological_context,
    )

    context = biological_context
    if context is None:
        context = standardize_biological_context(BiologicalContext(
            phenotype="", tissues=tuple(context_terms), source="explicit_context_terms")).to_dict()
    return compile_biosample_context(context)


def is_strong_anchor(variant: LocusVariant, basis: str) -> bool:
    if basis == WeightBasis.PIP.value:
        return variant.weight >= parameter("v2g.tier.min_variant_weight")
    return (variant.r2_to_anchor or 0.0) >= parameter("v2g.ld.r2_core")


def strip_private(value: Any) -> Any:
    """Drop classifier-internal ``_``-prefixed keys (raw sources) before cards leave the pipeline."""

    if isinstance(value, Mapping):
        return {key: strip_private(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [strip_private(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_private(item) for item in value)
    return value


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _abundance(value: Any) -> float | None:
    """Three significant digits, including very small nonzero measurements."""
    number = _float(value)
    return None if number is None else float(f"{number:.3g}")


def _specificity(value: Any) -> float | None:
    """Compact the value without moving it across the shortlist threshold."""
    number = _float(value)
    rounded = _abundance(value)
    threshold = parameter("v2g.expression.specificity_threshold")
    if number is not None and (rounded >= threshold) != (number >= threshold):
        return number
    return rounded


def _bounded(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    limit = parameter("v2g.card.max_records_per_category")
    return records[:limit], max(0, len(records) - limit)


def _aggregate_status(statuses: Iterable[str]) -> str:
    values = set(statuses)
    if not values:
        return ChannelStatus.NOT_REQUESTED.value
    if values == {ChannelStatus.FAILED.value}:
        return ChannelStatus.FAILED.value
    if ChannelStatus.FAILED.value in values or ChannelStatus.PARTIAL.value in values:
        return ChannelStatus.PARTIAL.value
    return ChannelStatus.OBSERVED.value


def _with_records(retrieval_status: str, records: list[dict[str, Any]]) -> str:
    """Observed when records exist; otherwise a completed lookup is not_observed."""

    if records:
        return retrieval_status if retrieval_status == ChannelStatus.PARTIAL.value else ChannelStatus.OBSERVED.value
    return ChannelStatus.NOT_OBSERVED.value if retrieval_status == ChannelStatus.OBSERVED.value else retrieval_status


def _variant_records(locus: LocusVariantSet, bundle: RetrievalBundle):
    """(locus variant, resolved OT variant record) pairs; unresolved aliases are ignored."""

    for variant in locus.variants:
        for query_id in variant.query_ids:
            record = bundle.variants.get(query_id)
            if record:
                yield variant, record
                break


def _variant_layer_status(locus: LocusVariantSet, bundle: RetrievalBundle, key: str) -> str:
    return _aggregate_status(bundle.variant_status.get(query_id, {}).get(key, ChannelStatus.NOT_REQUESTED.value)
                             for variant in locus.variants for query_id in variant.query_ids)


def _anchor(variant: LocusVariant) -> dict[str, Any]:
    return {"variant_id": variant.variant_id, "gwas_pip": _rounded(variant.pip),
            "r2_to_anchor": _rounded(variant.r2_to_anchor)}


def _identity(gene_id: str, window_symbol: str | None, target: Mapping[str, Any] | None) -> dict[str, Any]:
    target = target or {}

    def labels(key: str) -> list[str]:
        return sorted({str(item.get("label")) for item in (target.get(key) or [])
                       if isinstance(item, Mapping) and item.get("label")})

    return {
        "gene_id": gene_id,
        "symbol": target.get("approvedSymbol") or window_symbol,
        "name": target.get("approvedName"),
        "biotype": target.get("biotype"),
        "symbol_synonyms": labels("symbolSynonyms"),
        "name_synonyms": labels("nameSynonyms"),
        "obsolete_symbols": labels("obsoleteSymbols"),
        "obsolete_names": labels("obsoleteNames"),
        "transcript_ids": sorted({str(item) for item in (target.get("transcriptIds") or []) if item}),
        "protein_ids": sorted({str(item.get("id")) for item in (target.get("proteinIds") or [])
                               if isinstance(item, Mapping) and item.get("id")}),
    }


def _coding(gene_id: str, locus: LocusVariantSet, bundle: RetrievalBundle) -> dict[str, Any]:
    records = []
    for variant, record in _variant_records(locus, bundle):
        rows = [row for row in (record.get("transcriptConsequences") or [])
                if ((row.get("target") or {}).get("id")) == gene_id]
        canonical = [row for row in rows if row.get("isEnsemblCanonical")]
        labels = sorted({str(term.get("label")) for row in (canonical or rows)
                         for term in (row.get("variantConsequences") or []) if term.get("label")})
        if not labels:
            continue
        label = min(labels, key=lambda item: CLASS_ORDER.get(CONSEQUENCE_CLASS.get(item, "other"), 6))
        source = next(row for row in (canonical or rows)
                      if any(term.get("label") == label for term in (row.get("variantConsequences") or [])))
        records.append({**_anchor(variant), "consequence": label,
                        "canonical_transcript": bool(canonical),
                        "consequence_class": CONSEQUENCE_CLASS.get(label, "other"),
                        "strong_anchor": is_strong_anchor(variant, locus.weight_basis),
                        "_weight": variant.weight, "_source": {"row": source, "variant": variant}})
    records.sort(key=lambda row: (CLASS_ORDER.get(row["consequence_class"], 6), -row["_weight"],
                                  row["variant_id"]))
    class_counts: dict[str, int] = {}
    for row in records:
        class_counts[row["consequence_class"]] = class_counts.get(row["consequence_class"], 0) + 1
    # Flanking and intronic annotations are counted, not listed; they carry no gene-specific effect.
    detailed = [{key: value for key, value in row.items() if key != "_weight"} for row in records
                if row["consequence_class"] in DETAILED_CODING_CLASSES]
    shown, omitted = _bounded(detailed)
    status = _with_records(_variant_layer_status(locus, bundle, "consequence"), records)
    return {"status": status, "class_counts": class_counts, "records": shown, "omitted": omitted}


def _qtl(target: Mapping[str, Any] | None, locus: LocusVariantSet, status: str,
         context_terms: Sequence[str], biological_context: Mapping[str, Any] | None = None,
         context_matcher=None) -> dict[str, Any]:
    from .qtl_records import build_qtl_category

    return build_qtl_category(target, locus, status)


def _coloc_records(bundle: RetrievalBundle, context_terms: Sequence[str],
                   biological_context: Mapping[str, Any] | None = None, context_matcher=None) -> dict[str, list[dict[str, Any]]]:
    from .coloc_scope import POLICY, add_background, scope_classifier, coloc_sort_key
    classify = scope_classifier(bundle)
    scoped = bool(bundle.coverage.get("phenotype_scope") or bundle.coverage.get("phenotype_scope_ids"))
    cis_only = parameter("v2g.qtl.cis_only")
    genes: dict[str, list[dict[str, Any]]] = {}
    excluded = {"gene_identity_unavailable": 0, "trans_qtl": 0, "h4_unavailable": 0,
                "below_h4_threshold": 0}
    for gwas in bundle.gwas_credible_sets:
        gwas_study = gwas.get("study") or {}
        for row in gwas.get("colocalisation") or []:
            other = row.get("otherStudyLocus") or {}
            study = other.get("study") or {}
            gene = other.get("qtlGeneId") or ((study.get("target") or {}).get("id"))
            if not gene:
                excluded["gene_identity_unavailable"] += 1
                continue
            if cis_only and other.get("isTransQtl"):
                excluded["trans_qtl"] += 1
                continue
            h4 = _float(row.get("h4"))
            if h4 is None or not 0 <= h4 <= 1:
                excluded["h4_unavailable"] += 1
                continue
            phenotype_scope = row.get("_phenotype_scope") or classify(gwas_study)
            if scoped and phenotype_scope == "other_phenotype":
                add_background(bundle, {"gene_id": gene, "study_locus_id": other.get("studyLocusId"),
                    "qtl": other, "anchor_variants": gwas.get("anchor_variants", [])}, gwas_study, row)
                continue
            threshold = parameter("v2g.coloc.h4_in_scope" if scoped else "v2g.coloc.h4_core")
            if h4 < threshold:
                excluded["below_h4_threshold"] += 1
                continue
            biosample = ((study.get("biosample") or {}).get("biosampleName")) or None
            genes.setdefault(gene, []).append({
                "gwas_study_id": gwas_study.get("id"), "gwas_trait": gwas_study.get("traitFromSource"),
                "gwas_study_locus_id": gwas.get("studyLocusId"),
                "qtl_study_locus_id": other.get("studyLocusId"),
                "gwas_diseases": gwas_study.get("diseases") or [],
                "retrieval_policy": POLICY if scoped else "frozen_molqtl_all_phenotypes_v2",
                "phenotype_scope": phenotype_scope,
                "h4_inclusion_threshold": threshold,
                "qtl_type": other.get("studyType") or row.get("rightStudyType"), "qtl_study_id": study.get("id"),
                "biosample": biosample, "biosample_id": (study.get("biosample") or {}).get("biosampleId"),
                "method": row.get("colocalisationMethod"), "h4": h4,
                "clpp": _rounded(_float(row.get("clpp")), 5),
                "colocalising_variants": row.get("numberColocalisingVariants"),
                "_source": {"gwas": {key: value for key, value in gwas.items() if key != "colocalisation"},
                            "row": row},
            })
    for records in genes.values():
        records.sort(key=coloc_sort_key)
    bundle.coverage.setdefault("colocalisation", {}).update(
        excluded_by_reason=excluded, retained_rows=sum(len(rows) for rows in genes.values()),
        h4_threshold=parameter("v2g.coloc.h4_in_scope" if scoped else "v2g.coloc.h4_core"),
        other_phenotype_pair_count=sum(row["pair_count"] for row in bundle.coloc_background.values()))
    return genes


def _e2g(gene_id: str, locus: LocusVariantSet, bundle: RetrievalBundle,
         context_terms: Sequence[str], biological_context: Mapping[str, Any] | None = None,
         context_matcher=None) -> dict[str, Any]:
    from .e2g_records import build_e2g_records

    records = build_e2g_records(gene_id, locus, bundle)
    status = _with_records(_variant_layer_status(locus, bundle, "e2g"), records)
    return {"status": status, "linked_variants": len({w["variant_id"] for row in records for w in row["witnesses"]}),
            "selection": dict(bundle.coverage["e2g_selection"][gene_id]),
            "records": records, "omitted": 0}


def _expression(target: Mapping[str, Any] | None, status: str, context_terms: Sequence[str],
                biological_context: Mapping[str, Any] | None = None, context_matcher=None) -> dict[str, Any]:
    rows = _unique_rows(((target or {}).get("baselineExpression") or {}).get("rows") or [])
    match_context = context_matcher or _context_matcher(context_terms, biological_context)
    bulk = [row for row in rows if row.get("datasourceId") == "gtex" and _float(row.get("median")) is not None]
    top = sorted(bulk, key=lambda row: -_float(row["median"]))[:parameter("v2g.expression.top_tissues")]
    matched = []
    for row in rows:
        tissue = (row.get("tissueBiosample") or {}).get("biosampleName")
        cell = (row.get("celltypeBiosample") or {}).get("biosampleName")
        assessment = match_context(tissue=row.get("tissueBiosample"), cell_type=row.get("celltypeBiosample"))
        if assessment["matched"]:
            specificity = _float(row.get("specificity_score"))
            matched.append({"biosample": " / ".join(value for value in (tissue, cell) if value),
                            "tissue": tissue, "cell_type": cell,
                            "tissue_id": (row.get("tissueBiosample") or {}).get("biosampleId"),
                            "cell_type_id": (row.get("celltypeBiosample") or {}).get("biosampleId"),
                            "datasource": row.get("datasourceId"), "datatype": row.get("datatypeId"),
                            "median": _abundance(row.get("median")), "unit": row.get("unit"),
                            "specificity": _specificity(specificity), "specificity_high":
                            specificity >= parameter("v2g.expression.specificity_threshold") if specificity is not None else None,
                            "context_assessment": assessment, "_source_row": row})
    matched.sort(key=lambda row: (-(_float(row["_source_row"].get("specificity_score")) or 0.0), str(row["datasource"]),
                                  str(row["tissue_id"]), str(row["cell_type_id"])))
    most_specific = max((row for row in rows if _float(row.get("specificity_score")) is not None),
                        key=lambda row: _float(row["specificity_score"]), default=None)
    limit = parameter("v2g.card.max_expression_matches")
    shown, omitted = matched[:limit], max(0, len(matched) - limit)
    return {
        "status": _with_records(status, rows),
        "context_defined": bool(biological_context.get("defined")) if biological_context is not None else bool(context_terms),
        "specificity_method": "Open Targets CELLEX ranked specificity",
        "abundance_significant_digits": 3,
        "specificity_high_threshold": parameter("v2g.expression.specificity_threshold"),
        "specificity_is_causal_probability": False,
        "top_gtex_tissues": [{"tissue": (row.get("tissueBiosample") or {}).get("biosampleName"),
                              "tpm": _abundance(row.get("median"))} for row in top],
        "context_matched": shown, "context_matched_omitted": omitted,
        "_display_context_rows": matched,
        "context_high_specificity_count": sum(row["specificity_high"] is True for row in matched),
        "most_specific": None if most_specific is None else {
            "biosample": ((most_specific.get("celltypeBiosample") or {}).get("biosampleName")
                          or (most_specific.get("tissueBiosample") or {}).get("biosampleName")),
            "datasource": most_specific.get("datasourceId"),
            "specificity": _specificity(most_specific.get("specificity_score"))},
    }


def _constraint(target: Mapping[str, Any] | None, status: str) -> dict[str, Any]:
    rows = {str(row.get("constraintType")): row for row in ((target or {}).get("geneticConstraint") or [])}
    lof, mis = rows.get("lof") or {}, rows.get("mis") or {}
    return {"status": _with_records(status, list(rows.values())),
            "loeuf": _rounded(_float(lof.get("oeUpper"))), "lof_oe": _rounded(_float(lof.get("oe"))),
            "lof_score": _rounded(_float(lof.get("score"))), "missense_z": _rounded(_float(mis.get("score")))}


def _association(gene_id: str, bundle: RetrievalBundle) -> dict[str, Any]:
    def scores(row, key: str) -> dict[str, float | None]:
        return {str(item.get("id")): _rounded(_float(item.get("score")))
                for item in (row.get(key) or []) if item.get("id")}
    records = []
    entries = bundle.association_records.get(gene_id, [])
    if not entries and gene_id in bundle.associations:
        entries = [{"disease": bundle.association_disease, "association": bundle.associations[gene_id]}]
    for entry in entries:
        row = entry["association"]
        from ..phenotype_scope import classify_phenotype_ids

        phenotype_plan = bundle.coverage.get("phenotype_scope")
        phenotype_scope = (classify_phenotype_ids([entry["disease"]["id"]], phenotype_plan) if phenotype_plan else
                           "target_phenotype" if entry["disease"] == bundle.association_disease else "related_phenotype")
        datasource_scores = scores(row, "datasourceScores")
        excluded = sorted(set(datasource_scores) & set(parameter("v2g.association.excluded_datasources")))
        included = {key: value for key, value in datasource_scores.items() if key not in excluded}
        records.append({"disease": entry["disease"], "overall": None if excluded else _rounded(_float(row.get("score"))),
                        "phenotype_scope": phenotype_scope,
                        "datatypes": {} if excluded else scores(row, "datatypeScores"), "datasources": included,
                        "excluded_datasources": excluded,
                        "literature_interpretation": "Europe PMC literature co-occurrence; not locus evidence",
                        "_source": entry})
    primary = next((row for row in records if row["disease"] == bundle.association_disease), {})
    return {"status": _with_records(bundle.association_status, records), "records": records,
            "disease": bundle.association_disease, "overall": primary.get("overall"),
            "datatypes": primary.get("datatypes", {}), "datasources": primary.get("datasources", {}),
            "coverage_by_disease": dict(bundle.association_status_by_disease)}


def _function(target: Mapping[str, Any] | None, status: str) -> dict[str, Any]:
    target = target or {}
    terms, seen = [], set()
    for aspect in ("P", "F"):
        for row in target.get("geneOntology") or []:
            label = ((row.get("term") or {}).get("label"))
            if row.get("aspect") == aspect and label and label not in seen:
                seen.add(label)
                terms.append(label)
    pathways = list(dict.fromkeys(str(row.get("pathway")) for row in (target.get("pathways") or [])
                                  if row.get("pathway")))
    descriptions = target.get("functionDescriptions") or []
    first = str(descriptions[0]) if descriptions else ""
    sentence = re.split(r"(?<=[.!?])\s", first, maxsplit=1)[0][:400] if first else None
    present = bool(terms or pathways or sentence)
    return {"status": _with_records(status, [1] if present else []),
            "go_terms": terms[:parameter("v2g.function.go_terms")],
            "pathways": pathways[:parameter("v2g.function.pathways")], "description": sentence,
            "_display_go": target.get("geneOntology") or [],
            "_display_pathways": target.get("pathways") or [],
            "_display_descriptions": descriptions}


def evidence_tier(evidence: Mapping[str, Any]) -> tuple[str, list[str]]:
    """A classification label of the strongest locus-to-gene link; not a score."""

    core, support = [], []
    for row in evidence["coding"]["records"]:
        if row["consequence_class"] in CODING_CLASSES:
            (core if row["strong_anchor"] else support).append(f"coding:{row['consequence']}")
            break
    clpp_core = parameter("v2g.qtl.clpp_core")
    pip_basis = evidence["qtl"].get("clpp_basis") == WeightBasis.PIP.value
    for row in evidence["qtl"]["records"]:
        # With fine-mapping PIPs use CLPP; with r2-approximated weights (which dilute
        # CLPP across a large LD block) require a high QTL PIP at a core-LD variant.
        if pip_basis and (row["clpp"] or 0.0) >= clpp_core and row["strong_anchor_shared"]:
            core.append(f"qtl:{row['qtl_type']} clpp={row['clpp']}")
            break
        if not pip_basis and (row["max_qtl_pip_at_strong_anchor"] or 0.0) >= parameter("v2g.tier.min_variant_weight"):
            core.append(f"qtl:{row['qtl_type']} qtl_pip={row['max_qtl_pip_at_strong_anchor']}")
            break
    else:
        r2_extended = parameter("v2g.ld.r2_extended")
        if any((row["max_shared_r2"] or 0.0) >= r2_extended or (row["clpp"] or 0.0) > 0
               for row in evidence["qtl"]["records"]):
            support.append("qtl:overlap")
    for row in evidence["coloc"]["records"]:
        if (row["h4"] or 0.0) >= parameter("v2g.coloc.h4_core") or (row["clpp"] or 0.0) >= clpp_core:
            core.append(f"coloc:h4={row['h4']} clpp={row['clpp']}")
            break
        if (row["h4"] or 0.0) >= parameter("v2g.coloc.h4_support"):
            support.append(f"coloc:h4={row['h4']}")
            break
    e2g = evidence["e2g"]["records"]
    strong = [row for row in e2g if (row["score"] or 0.0) >= parameter("v2g.e2g.core_score")]
    if strong:
        core.append(f"e2g:{strong[0]['method']} score={strong[0]['score']}")
    elif e2g:
        support.append("e2g:link")
    if core:
        return "A", core + support
    if support:
        return "B", support
    return "C", ["position_only"]


def classify_gene_evidence(locus: LocusVariantSet, window: GeneWindow, bundle: RetrievalBundle, *,
                           context_terms: Sequence[str] = (),
                           biological_context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """One evidence card per window gene, in genomic order."""

    context_matcher = _context_matcher(context_terms, biological_context)
    coloc_by_gene = _coloc_records(bundle, context_terms, biological_context, context_matcher)
    centroid = locus.pip_centroid()
    cards: list[dict[str, Any]] = []
    for gene in window.genes:
        target = bundle.targets.get(gene.gene_id)
        status = bundle.target_status.get(gene.gene_id, {})
        target_status = status.get("target", ChannelStatus.NOT_REQUESTED.value)
        in_body = [variant for variant in locus.variants if gene.start <= variant.position <= gene.end]
        coloc_rows = coloc_by_gene.get(gene.gene_id, [])
        evidence = {
            "coding": _coding(gene.gene_id, locus, bundle),
            "qtl": _qtl(target, locus, status.get("qtl", ChannelStatus.NOT_REQUESTED.value),
                        context_terms, biological_context, context_matcher),
            "coloc": {"status": _with_records(bundle.coloc_status, coloc_rows), "records": coloc_rows,
                      "omitted": 0, "scope": "frozen_molqtl_all_phenotypes_v2",
                      "coverage": dict(bundle.coverage.get("colocalisation", {}))},
            "e2g": _e2g(gene.gene_id, locus, bundle, context_terms, biological_context, context_matcher),
            "expression": _expression(target, status.get("expression", ChannelStatus.NOT_REQUESTED.value),
                                      context_terms, biological_context, context_matcher),
            "constraint": _constraint(target, target_status),
            "association": _association(gene.gene_id, bundle),
            "function": _function(target, target_status),
        }
        background = [dict(row) for (owner, _), row in bundle.coloc_background.items() if owner == gene.gene_id]
        evidence["coloc"]["other_phenotypes"] = background
        evidence["coloc"]["scope"] = bundle.coverage.get("colocalisation", {}).get("retrieval_policy", evidence["coloc"]["scope"])
        cards.append({
            "gene_id": gene.gene_id,
            "identity": _identity(gene.gene_id, gene.symbol, target),
            "auxiliary": {
                "chromosome": gene.chromosome, "start": gene.start, "end": gene.end, "strand": gene.strand,
                "tss": gene.tss, "gene_length_bp": gene.end - gene.start + 1,
                "tss_distance_bp": abs(gene.tss - locus.lead_position),
                "body_distance_bp": max(gene.start - locus.lead_position, locus.lead_position - gene.end, 0),
                "centroid_tss_distance_bp": None if centroid is None else int(abs(gene.tss - centroid)),
                "locus_variants_in_gene_body": len(in_body),
                "gwas_pip_in_gene_body": _rounded(sum(variant.pip or 0.0 for variant in in_body)),
            },
            "evidence": evidence,
            "coverage": {category: evidence[category]["status"] for category in EVIDENCE_CATEGORIES},
        })
    by_distance = sorted(card["auxiliary"]["tss_distance_bp"] for card in cards)
    for card in cards:
        closer = sum(1 for value in by_distance if value < card["auxiliary"]["tss_distance_bp"])
        card["auxiliary"]["distance_rank"] = closer + 1
        card["auxiliary"]["closer_gene_count"] = closer
    return cards
