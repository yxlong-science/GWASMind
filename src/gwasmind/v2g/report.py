"""Browser-safe summaries of gene-centric evidence cards.

The scientific report is written by the report pipeline from the evidence
ledger; these summaries only feed the "Gene evidence cards" window.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .display_previews import (
    association_scope_source,
    coloc_preview,
    e2g_preview,
    expression_preview,
    function_preview,
    raw_number,
)
from .evidence import CODING_CLASSES, _float

PUBLIC_CARD_CATEGORIES = ("coding", "qtl", "coloc", "e2g", "expression", "association", "constraint", "function")
_TEXT_LIMIT = 300
_QTL_LABELS = {"eqtl": "eQTL", "pqtl": "pQTL", "sqtl": "sQTL", "tuqtl": "tuQTL", "sceqtl": "sc-eQTL",
               "scpqtl": "sc-pQTL", "scsqtl": "sc-sQTL", "sctuqtl": "sc-tuQTL"}


def _text(value: Any) -> str:
    return str(value)[:_TEXT_LIMIT]


def _qtl_type(value: Any) -> str:
    return _QTL_LABELS.get(str(value or "").casefold(), str(value or "QTL"))


def _anchor(row: Mapping[str, Any]) -> str:
    return str(row.get("variant_id"))


def _probability_text(value: Any) -> str | None:
    """Compact public display only; a rounded near-one value is not exact one."""
    number = _float(value)
    if number is None:
        return None
    label = f"{number:.4g}"
    return "~1" if number < 1 and label == "1" else label


def _qtl_display(row: Mapping[str, Any], source_aliases: Mapping[str, str], ld_scope=None) -> dict[str, Any]:
    if "cs_lead_variant_id" in row or "cs_lead_measurement" in row:
        return _lead_qtl_display(row, source_aliases, ld_scope)
    measurement = row.get("measurement") or {}
    locus_variant = measurement.get("locus_variant_id") or measurement.get("variant_id")
    measured_variant = measurement.get("measured_variant_id") or measurement.get("variant_id")
    sources = set(measurement.get("variant_sources") or ())
    origin = measurement.get("variant_origin", "unknown")
    if sources.intersection({"selected", "lead"}):
        origin = ("source_selected" if measured_variant == locus_variant
                  or source_aliases.get(locus_variant) == measured_variant else "unknown")
    elif "ld_proxy" in sources:
        origin = "ld_proxy" if measured_variant == locus_variant else "unknown"
    role = {"source_selected": "User-selected variant", "ld_proxy": "LD-linked variant"}.get(origin, "Analyzed variant")
    mantissa, exponent = measurement.get("pValueMantissa"), measurement.get("pValueExponent")
    finite = lambda value: isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
    p_value = (f"{mantissa:.4g}e{int(exponent):+d}" if finite(mantissa) and finite(exponent)
               and int(exponent) == exponent else None)
    beta = measurement.get("beta")
    beta_text = f"{beta:.4g}" if finite(beta) else None
    values = {"variant_id": measured_variant, "locus_variant_id": locus_variant, "variant_origin": origin,
              "variant_label": role, "qtl_type": _qtl_type(row.get("qtl_type")),
              "biosample": row.get("biosample") or "unspecified biosample", "biosample_id": row.get("biosample_id"),
              "p_value": p_value, "beta": beta_text, "study_id": row.get("study_id"),
              "project": row.get("project"),
              "study_locus_id": row.get("study_locus_id"), "evidence_id": row.get("evidence_id"),
              "context_matched": bool(row.get("context_matched")),
              "measurement": {key: measurement[key] for key in
                              ("pValueMantissa", "pValueExponent", "beta", "standardError")
                              if finite(measurement.get(key))}}
    if ld_scope is not None and ld_scope.valid:
        values["ld_relations"] = [dict(link) for link in ld_scope.links
                                  if link["proxy_variant_id"] == measured_variant]
        values["ld_reference_panel"] = ld_scope.reference_panel
    values["summary"] = " · ".join((str(measured_variant or "Variant unavailable"), role, values["qtl_type"],
        values["biosample"], f"P={p_value}" if p_value is not None else "P unavailable",
        f"beta={beta_text}" if beta_text is not None else "beta unavailable"))
    return values


def _lead_qtl_display(row: Mapping[str, Any], source_aliases: Mapping[str, str], ld_scope=None) -> dict[str, Any]:
    """Bind displayed statistics to the CS lead, separately from locus witnesses."""
    lead = row.get("cs_lead_variant_id")
    measurement = row.get("cs_lead_measurement") or {}
    if measurement.get("variant_id") != lead or not lead:
        measurement = {}
    finite = lambda value: isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
    mantissa, exponent = measurement.get("pValueMantissa"), measurement.get("pValueExponent")
    p_value = (f"{mantissa:.4g}e{int(exponent):+d}" if finite(mantissa) and finite(exponent)
               and int(exponent) == exponent else None)
    beta = measurement.get("beta")
    beta_text = f"{beta:.4g}" if finite(beta) else None
    witnesses = []
    for witness in row.get("locus_anchor_witnesses") or ():
        locus_variant = witness.get("variant_id")
        variant = witness.get("query_variant_id") or locus_variant
        sources = set(witness.get("source_roles") or witness.get("variant_sources") or ())
        origin = "unknown"
        if sources.intersection({"selected", "lead", "source_selected"}) and (
                locus_variant == variant or source_aliases.get(locus_variant) == variant):
            origin = "source_selected"
        elif sources.intersection({"ld_proxy", "ld_linked"}) and locus_variant == variant:
            origin = "ld_proxy"
        item = {"variant_id": variant, "locus_variant_id": locus_variant, "variant_origin": origin}
        if item not in witnesses:
            witnesses.append(item)
    lead_origins = {item["variant_origin"] for item in witnesses if item["variant_id"] == lead}
    origin = next(iter(lead_origins)) if len(lead_origins) == 1 else "unknown"
    values = {"variant_id": lead, "cs_lead_variant_id": lead, "measurement_scope": "credible_set_lead",
              "genome_build": "GRCh38",
              "variant_origin": origin, "variant_label": "Credible-set lead", "locus_anchor_witnesses": witnesses,
              "qtl_type": _qtl_type(row.get("qtl_type")), "biosample": row.get("biosample") or "unspecified biosample",
              "biosample_id": row.get("biosample_id"), "p_value": p_value, "beta": beta_text,
              "study_id": row.get("study_id"), "study_locus_id": row.get("study_locus_id"),
              "project": row.get("project"), "evidence_id": row.get("evidence_id"),
              "measurement": {key: measurement[key] for key in
                              ("pValueMantissa", "pValueExponent", "beta", "standardError")
                              if finite(measurement.get(key))}}
    if ld_scope is not None and ld_scope.valid:
        anchors = {item["variant_id"] for item in witnesses}
        values["ld_relations"] = [dict(link) for link in ld_scope.links if link["proxy_variant_id"] in anchors]
        values["ld_reference_panel"] = ld_scope.reference_panel
    roles = {"source_selected": "User-selected variant", "ld_proxy": "LD-linked variant", "unknown": "Analyzed variant"}
    overlap = " / ".join(dict.fromkeys(roles[item["variant_origin"]] for item in witnesses))
    values["summary"] = " · ".join((str(lead or "Lead unavailable"), "Credible-set lead", values["qtl_type"],
        values["biosample"], f"P={p_value}" if p_value is not None else "P unavailable",
        f"beta={beta_text}" if beta_text is not None else "beta unavailable", f"Locus overlap: {overlap}"))
    return values


def _qtl_summary(row: Mapping[str, Any], source_aliases: Mapping[str, str] | None = None) -> str:
    return _qtl_display(row, source_aliases or {})["summary"]


def _coloc_summary(row: Mapping[str, Any]) -> str:
    h4 = _probability_text(row.get("h4"))
    unlabelled = row.get("retrieval_policy") == "frozen_molqtl_all_phenotypes_v2"
    scope = "" if unlabelled else _phenotype_label(row)
    return " · ".join(item for item in (scope, f"PP.H4={h4}" if h4 is not None else "PP.H4 unavailable",
                       str(row.get("gwas_trait") or "GWAS phenotype unavailable"),
                       _qtl_type(row.get("qtl_type")), str(row.get("biosample") or "unspecified biosample"),
                       "" if unlabelled else "Context matched" if row.get("context_matched") is True else
                       "Context not matched" if row.get("context_matched") is False else "Context not established") if item)


def _phenotype_label(row: Mapping[str, Any]) -> str:
    return {"target_phenotype": "Target phenotype", "related_phenotype": "Related phenotype",
            "target_trait": "Target phenotype", "related_trait": "Related phenotype"}.get(row.get("phenotype_scope"), "")


def _association_summary(row: Mapping[str, Any]) -> str:
    disease = row.get("disease") or {}
    label = f"{disease.get('name') or 'Phenotype unavailable'} ({disease.get('id') or 'ID unavailable'})"
    if _phenotype_label(row):
        label = _phenotype_label(row) + " · " + label
    sources = sorted(((value, key) for key, value in row.get("datasources", {}).items()
                       if value is not None), reverse=True)
    names = {"gwas_credible_sets": "OT L2G-derived score", "ot_genetics_portal": "OT L2G-derived score",
             "europepmc": "Literature co-occurrence (not locus evidence)"}
    details = ([f"Overall {row['overall']}"] if row.get("overall") is not None else [])
    details += [f"{names.get(key, key)} {value}" for value, key in sources[:2]]
    return label + ": " + "; ".join(details)


def _expression_summary(row: Mapping[str, Any]) -> str:
    def usable(value):
        return _float(value) is not None

    median, specificity = row.get("median"), row.get("specificity")
    abundance = f"{median:g} {row.get('unit') or ''}" if usable(median) else "median unavailable"
    specificity_label = f"specificity {specificity:g}" if usable(specificity) else "specificity unavailable"
    return (f"{row['biosample']} · {row.get('datasource') or 'source unavailable'} · {abundance} · {specificity_label}"
            + (" (high)" if row.get("specificity_high") is True and usable(specificity) else ""))


def key_evidence(gene: Mapping[str, Any], source_aliases: Mapping[str, str] | None = None) -> list[str]:
    evidence = gene["evidence"]
    items: list[str] = []
    coding = evidence["coding"]["records"]
    if coding and (coding[0]["consequence_class"] in CODING_CLASSES or coding[0]["consequence_class"] == "splice_region"):
        items.append(f"{coding[0]['consequence'].replace('_', ' ')} ({_anchor(coding[0])})")
    qtl = evidence["qtl"]
    if qtl["records"]:
        top = qtl["records"][0]
        items.append(_qtl_summary(top, source_aliases))
    coloc = evidence["coloc"]["records"]
    if coloc:
        top = coloc[0]
        items.append("GWAS-QTL colocalisation: " + _coloc_summary(top))
    e2g = evidence["e2g"]["records"]
    if e2g:
        top = e2g[0]
        items.append(f"E2G score {raw_number(top['score'])} ({top['method']}, {top['biosample'] or 'unspecified biosample'})")
    association = evidence["association"]
    if association.get("overall") is not None:
        items.append(_association_summary(association))
    return items


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _category_summary(gene: Mapping[str, Any], category: str,
                      source_aliases: Mapping[str, str] | None = None, ld_scope=None, *,
                      biological_context=None, phenotype_scope=None, variant_identity_registry=None) -> dict[str, Any]:
    evidence = gene["evidence"][category]
    items: list[str] = []
    count: int | None = None
    preview = []

    def identity(identifier):
        if variant_identity_registry:
            from ..variant_identity_registry import variant_display
            return variant_display(identifier, variant_identity_registry)
        return {"display_label": identifier, "rsid": None}

    def variant_label(identifier):
        return identity(identifier)["display_label"]
    if category == "coding":
        count = len(evidence["records"])
        items = [f"{row['consequence'].replace('_', ' ')} ({_anchor(row)})" for row in evidence["records"][:3]]
    elif category == "qtl":
        count = evidence.get("overlapping", 0)
        items = [_qtl_summary(row, source_aliases) for row in evidence["records"][:3]]
        if evidence.get("credible_sets"):
            items.append(f"{count} of {evidence['credible_sets']} QTL credible sets overlap the locus")
    elif category == "coloc":
        from .coloc_scope import validate_background_rows

        validate_background_rows(evidence.get("other_phenotypes", ()))
        count = len(evidence["records"]) + int(evidence.get("omitted") or 0)
        preview = coloc_preview(evidence["records"], phenotype_scope=phenotype_scope,
                                biological_context=biological_context, summary=_coloc_summary)
        items = [row["summary"] for row in preview]
        if not items and evidence.get("scope"):
            items = ["No qualifying GWAS–QTL result retained."]
    elif category == "e2g":
        count = len(evidence["records"]) + int(evidence.get("omitted") or 0)
        preview = e2g_preview(evidence["records"], variant_label=variant_label,
                              biological_context=biological_context)
        items = [row["summary"] for row in preview]
    elif category == "expression":
        preview = expression_preview(evidence)
        items = [row["summary"] for row in preview]
        count = len(evidence.get("_display_context_rows", evidence.get("context_matched", [])))
        if "_display_context_rows" not in evidence:
            count += int(evidence.get("context_matched_omitted") or 0)
        if not items:
            items = ["No expression measurement matched the frozen context."]
    elif category == "association":
        records = evidence.get("records") or ([evidence] if evidence.get("overall") is not None else [])
        count = len(records)
        items = [_association_summary(row) for row in records[:3]]
    elif category == "constraint":
        items = [text for text in (f"LOEUF {evidence['loeuf']}" if evidence.get("loeuf") is not None else "",
                                   f"missense Z {evidence['missense_z']}" if evidence.get("missense_z") is not None
                                   else "") if text]
    elif category == "function":
        preview = function_preview(evidence)
        items = [row["summary"] for row in preview]
    result = {"status": evidence.get("status"), "count": count, "items": [_text(item) for item in items[:4]]}
    if category == "coding":
        classes = {row.get("consequence_class") for row in evidence["records"]}
        result["tags"] = (["Coding"] if classes & {"lof", "splice_core", "missense"} else [])
        if any("utr" in str(value).lower() for value in classes):
            result["tags"].append("UTR")
        if "synonymous" in classes:
            result["tags"].append("Synonymous")
    if category == "coloc":
        result["tags"] = (["Coloc (trait)"] if any(row.get("phenotype_scope") in
            {"target_phenotype", "related_phenotype"} for row in evidence["records"]) else [])
        if evidence.get("other_phenotypes"):
            result["tags"].append("Coloc (other)")
            if not evidence["records"]:
                phenotypes = {identity for row in evidence["other_phenotypes"] for identity in row["phenotype_ids"]}
                items = [f"Other-phenotype background: {len(phenotypes)} phenotypes; no target/related pair retained."]
                result["items"] = items
    if category == "association":
        source = association_scope_source(evidence, phenotype_scope)
        if source:
            result["scope_source"] = source
    if preview:
        result["preview"] = preview
        if category == "function" and result["status"] == "not_observed":
            result["status"] = "observed"
    if category == "qtl":
        result["rows"] = [_qtl_display(row, source_aliases or {}, ld_scope) for row in evidence["records"]]
        for row in result["rows"]:
            display = identity(row.get("variant_id"))
            row["variant_display"] = display["display_label"] or row.get("variant_id") or "Variant unavailable"
            if display.get("rsid"):
                row["rsid"] = display["rsid"]
    return result


def public_cards(genes: Sequence[Mapping[str, Any]], *,
                 source_normalizations: Sequence[Mapping[str, Any]] = (),
                 ld_expansion: Mapping[str, Any] | None = None,
                 biological_context: Mapping[str, Any] | None = None,
                 phenotype_scope: Mapping[str, Any] | None = None,
                 variant_identity_registry: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Unranked card summaries ordered by TSS distance to the lead variant."""

    from ..ld_support import ld_report_scope
    from ..source_normalization import validate_source_normalization

    aliases = {proof["original_id"]: proof["canonical_id"] for proof in source_normalizations
               if isinstance(proof, Mapping) and validate_source_normalization(proof)}
    ld_scope = (ld_report_scope({"ld_expansion": ld_expansion}, ld_expansion.get("source_observations", ()),
                                "prioritize_genes") if ld_expansion else None)
    ordered = sorted(genes, key=lambda gene: (gene["auxiliary"]["tss_distance_bp"], gene["gene_id"]))
    return [{
        "gene_id": gene["gene_id"],
        "symbol": _text(gene["identity"].get("symbol") or gene["gene_id"]),
        "rank": None,
        "tss_distance_bp": gene["auxiliary"]["tss_distance_bp"],
        "distance_rank": gene["auxiliary"]["distance_rank"],
        "key_evidence": [],
        "rationale": "",
        "categories": {category: _category_summary(gene, category, aliases, ld_scope,
                       biological_context=biological_context, phenotype_scope=phenotype_scope,
                       variant_identity_registry=variant_identity_registry) for category in PUBLIC_CARD_CATEGORIES},
    } for gene in ordered]


def rank_public_cards(cards: Sequence[Mapping[str, Any]],
                      ranked_genes: Sequence[Mapping[str, Any]],
                      candidate_decisions: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """Attach the validated model ranking and rationales; unranked cards follow in distance order."""

    from ..models import shared_ranks

    ranking = {}
    for position, row in enumerate(ranked_genes, 1):
        if isinstance(row, Mapping) and isinstance(row.get("gene_id"), str):
            rank = row.get("rank") if isinstance(row.get("rank"), int) and row["rank"] > 0 else position
            ranking[row["gene_id"]] = (rank, str(row.get("rationale") or "")[:1200])
    shared = shared_ranks([row for row in ranked_genes if isinstance(row, Mapping)])
    result = []
    decisions = {row["gene_id"]: row for row in candidate_decisions
                 if isinstance(row, Mapping) and row.get("gene_id") and row.get("status") in
                 {"selected", "not_selected", "unresolved"}}
    for card in cards:
        rank, rationale = ranking.get(card.get("gene_id"), (None, ""))
        value = {**card, "rank": rank, "rationale": rationale}
        if rank is not None and card.get("gene_id") in shared:
            value["shared_rank"], value["tied"] = shared[card["gene_id"]]
        if decision := decisions.get(card.get("gene_id")):
            reason = str(decision.get("reason") or "")[:1200]
            value.update(model_selection=decision["status"], selection_reason=reason,
                         selection_evidence_ids=list(decision.get("evidence_ids") or []),
                         rationale=rationale or reason)
        result.append(value)
    result.sort(key=lambda card: (card["rank"] is None, card["rank"] or 0, card.get("tss_distance_bp") or 0))
    return result
