"""Model evidence cards: complete QTL/E2G tables, bounded representative rows elsewhere."""

import json
import math
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy

from ..audit import sha256_json
from ..configuration import parameter
from .coloc_scope import coloc_sort_key, study_phenotype_ids, validate_background_rows


def _bytes(value):
    def scientific(item):
        if isinstance(item, Mapping):
            return {key: scientific(child) for key, child in item.items()
                    if not key.startswith("_") and key not in {"coverage", "coverage_by_disease"}}
        if isinstance(item, list):
            return [scientific(child) for child in item]
        return item
    # Per-channel acquisition receipts become one shared retrieval_summary in
    # the codec; charging their complete per-anchor ledger to every gene is wrong.
    return len(json.dumps(scientific(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def inline_ld(cards, context):
    table = context.get("ld_relations") or {}
    rows = [{**table.get("constant_fields", {}), **dict(zip(table.get("columns", []), row))}
            for row in table.get("rows", [])]
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["proxy_variant_id"], []).append(row)
    aliases = context.get("resolved_source_aliases", {})

    def walk(value):
        if isinstance(value, dict):
            for item in list(value.values()):
                walk(item)
            identifier = value.get("variant_id")
            if isinstance(identifier, str) and aliases.get(identifier, identifier) in by_variant:
                links = sorted(by_variant[aliases.get(identifier, identifier)],
                               key=lambda row: (-(row.get("r2") or 0), str(row.get("selected_variant_id"))))
                value["ld_links"] = deepcopy(links[:1])
                if len(links) > 1:
                    value["other_ld_pairs_archived"] = len(links) - 1
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(cards)
    context.pop("ld_relations", None)
    context["relations_inline"] = True
    context["relation_selection"] = "strongest_verified_edge_per_presented_variant"
    context["relation_selection_note"] = (
        "Each presented variant carries its strongest verified selected-source LD edge. "
        "Additional edges are counted by other_ld_pairs_archived and remain in the full ledger; "
        "an omitted edge is not evidence of absent LD.")
    context["card_scalar_semantics"]["pairwise_ld"] = (
        "Only the inline ld_links beside a variant authorize pair-specific LD claims; name both SNPs and population.")


def _summary(rows, category):
    summary = {"total_records": len(rows), "displayed_records": len(rows), "omitted_records": 0,
               "_complete_records_sha256": sha256_json(rows)}
    if category == "e2g":
        groups = {}
        for row in rows:
            key = str(row.get("interval_type") or "unreported")
            groups.setdefault(key, []).append(row)
        summary["interval_types"] = [{"interval_type": key, "records": len(group),
            "biosample_count": len({row.get("biosample_id") or row.get("biosample") for row in group}),
            "score_min": min(row["score"] for row in group), "score_max": max(row["score"] for row in group),
            "context_matched_biosample_count": len({row.get("biosample_id") or row.get("biosample")
                for row in group if row.get("_context_match")})} for key, group in sorted(groups.items())]
    if category == "qtl":
        summary["qtl_types"] = dict(Counter(row.get("qtl_type") or "unreported" for row in rows))
        summary["biosample_count"] = len({row.get("biosample_id") or row.get("biosample") for row in rows})
    if category == "coloc":
        summary["qtl_signal_count"] = len({row.get("qtl_study_locus_id") for row in rows})
        summary["max_h4"] = max((row.get("h4", 0) for row in rows), default=None)
    return summary


def _coloc_order(rows, phenotype_scope, biological_context):
    from .display_previews import coloc_preview
    chosen = coloc_preview(rows, phenotype_scope=phenotype_scope, biological_context=biological_context,
                           summary=lambda row: "")
    priority = {row["evidence_id"]: index for index, row in enumerate(chosen)}
    return sorted(rows, key=lambda row: (priority.get(row.get("evidence_id"), 3), coloc_sort_key(row)))


def _qtl_order(rows, displayed_partners, matcher):
    """The QTL behind a displayed target/related colocalization, then context-matched QTLs, come first."""
    def context_matched(row):
        biosample = {"biosampleId": row.get("biosample_id"), "biosampleName": row.get("biosample")}
        return matcher(tissue=biosample, cell_type=biosample).get("matched", False)

    order = {id(row): index for index, row in enumerate(rows)}
    return sorted(rows, key=lambda row: (row.get("study_locus_id") not in displayed_partners,
                                         not context_matched(row), order[id(row)]))


_UPLOADED_ROLES = frozenset({"selected", "lead"})
_COMPLETE_CATEGORIES = ("qtl", "e2g")


def _compact_membership(row, key):
    """Name the uploaded members; show one LD route only when no uploaded variant is a member."""
    witnesses = row.get(key, [])
    if not witnesses:
        return

    def roles(item):
        return set(item.get("source_roles", item.get("variant_sources", [])))

    uploaded = sorted({item["variant_id"] for item in witnesses if roles(item) & _UPLOADED_ROLES})
    kept = [] if uploaded else sorted(witnesses, key=lambda item: str(item.get("variant_id")))[:1]
    row[key] = kept
    row["uploaded_members"] = uploaded
    row["locus_witness_count"] = len(witnesses)
    row["locus_witnesses_omitted"] = len(witnesses) - len(kept)


def _scope_phenotypes(context):
    scope = context.get("phenotype_scope") if isinstance(context, Mapping) else None
    scope = scope if isinstance(scope, Mapping) else {}
    rows = ([scope["target"]] if isinstance(scope.get("target"), Mapping) else []) + list(scope.get("related") or ())
    return {row["id"]: row.get("name") for row in rows if isinstance(row, Mapping) and isinstance(row.get("id"), str)}


def _pair_phenotypes(row, scope_ids):
    return study_phenotype_ids(row.get("gwas_diseases"), row.get("gwas_trait"), scope_ids)


def _phenotype_table(rows, scope_ids):
    """(phenotype) -> distinct GWAS studies, QTL signals and maximum H4 over the supplied pairs."""
    table = {}
    for row in rows:
        for identity in _pair_phenotypes(row, scope_ids):
            entry = table.setdefault(identity, {"studies": set(), "signals": set(), "max_h4": None, "row": row})
            entry["studies"].add(row.get("gwas_study_id"))
            entry["signals"].add(row.get("qtl_study_locus_id"))
            if isinstance(row.get("h4"), (int, float)) and (entry["max_h4"] is None or row["h4"] > entry["max_h4"]):
                entry["max_h4"] = row["h4"]
    order = {identity: index for index, identity in enumerate(scope_ids)}
    return sorted(table.items(), key=lambda item: (order.get(item[0], len(order)), -(item[1]["max_h4"] or 0), item[0]))


def _attach_coloc_summaries(qtl_rows, coloc_rows, other_rows, scope_ids):
    """Every QTL row summarizes its colocalization per phenotype; the best pair stays citable."""
    pairs = {}
    for row in coloc_rows:
        pairs.setdefault(row.get("qtl_study_locus_id"), []).append(row)
    other = {row.get("qtl_study_locus_id"): row for row in other_rows}
    for qtl in qtl_rows:
        own = pairs.get(qtl.get("study_locus_id"), [])
        qtl["coloc_phenotypes"] = [{"phenotype_id": identity, "gwas_studies": len(entry["studies"]),
                                    "max_h4": _round(entry["max_h4"], 4)}
                                   for identity, entry in _phenotype_table(own, scope_ids)]
        best = min(own, key=coloc_sort_key) if own else None
        qtl["coloc_best_pair"] = None if best is None else {
            "evidence_id": best.get("evidence_id"), "gwas_trait": best.get("gwas_trait"),
            "gwas_study_id": best.get("gwas_study_id"), "h4": _round(best.get("h4"), 4),
            "gwas_study_locus_id": best.get("gwas_study_locus_id"), "method": best.get("method"),
            **({"phenotype_scope": best["phenotype_scope"]} if best.get("phenotype_scope") != "target_phenotype" else {})}
        background = other.get(qtl.get("study_locus_id")) or {}
        qtl["coloc_other_phenotypes"] = len(background["phenotype_ids"]) if background else 0
        qtl["coloc_other_max_h4"] = _round(background.get("max_h4"), 4)


def _gene_phenotype_summary(coloc_rows, scope):
    names = dict(scope)
    for row in coloc_rows:
        for disease in row.get("gwas_diseases") or ():
            if isinstance(disease, Mapping) and isinstance(disease.get("id"), str):
                names.setdefault(disease["id"], disease.get("name"))
    return [{"phenotype_id": identity, "phenotype": names.get(identity) or identity.removeprefix("unmapped:"),
             "phenotype_scope": entry["row"].get("phenotype_scope") if identity not in scope else
             ("target_phenotype" if identity == next(iter(scope)) else "related_phenotype"),
             "qtl_signals": len(entry["signals"]), "gwas_studies": len(entry["studies"]),
             "max_h4": _round(entry["max_h4"], 4)}
            for identity, entry in _phenotype_table(coloc_rows, tuple(scope))]


def _one_pair_per_signal_and_phenotype(rows, scope_ids):
    """Assign each signal/phenotype to its first representative, preserving original study mappings."""
    seen, kept = set(), []
    for row in rows:
        signal = row.get("qtl_study_locus_id")
        represented = [identity for identity in _pair_phenotypes(row, scope_ids)
                       if (signal, identity) not in seen]
        if represented:
            seen.update((signal, identity) for identity in represented)
            kept.append({**row, "represented_phenotype_ids": represented})
    return kept


def _round(value, digits):
    return round(value, digits) if isinstance(value, float) else value


def _significant(value, digits=4):
    return float(f"{value:.{digits}g}") if isinstance(value, float) else value


def _bounded_bytes(card):
    """Size of the representative parts; complete QTL and E2G tables are not budgeted per gene."""
    evidence = {key: ({**value, "records": []} if key in _COMPLETE_CATEGORIES else value)
                for key, value in card["evidence"].items()}
    return _bytes({**card, "evidence": evidence})


def representative_cards(cards, *, biological_context=None):
    """Complete QTL and E2G tables; bounded colocalization pairs, coding and expression rows.

    Every QTL credible set and E2G link that passed the scientific locus filters
    is shown. Each QTL row carries a summary of all its colocalization pairs.
    Missing evidence is never converted to absence.
    """
    from ..biological_context import compile_biosample_context
    context = biological_context or {}
    matcher = compile_biosample_context(context)
    scope = _scope_phenotypes(context)
    scope_ids = tuple(scope)
    result = deepcopy(cards)
    receipts = []
    budget = parameter("v2g.card.max_tokens_per_gene")
    ratio = parameter("model.token_estimate.deepseek_bytes_per_token")
    for card in result:
        from .compact_input import _clean
        evidence = card["evidence"] = {key: _clean(value, category=key) for key, value in card["evidence"].items()}
        for category in _COMPLETE_CATEGORIES:
            for row in evidence[category].get("records", []):
                _compact_membership(row, "locus_anchor_witnesses" if category == "qtl" else "witnesses")
                measurement = row.get("cs_lead_measurement")
                if isinstance(measurement, dict) and measurement.get("variant_id") == row.get("cs_lead_variant_id"):
                    # Both keys repeat the row: the scope is always the lead, the variant is cs_lead_variant_id.
                    measurement.pop("variant_id")
                    measurement.pop("measurement_scope", None)
                if isinstance(measurement, dict) and "pValueMantissa" in measurement:
                    measurement["pValueMantissa"] = _significant(measurement["pValueMantissa"])
        coding = evidence["coding"]
        coding_rows = coding.get("records", [])
        kept, counts = [], Counter()
        for row in coding_rows:
            category = row.get("consequence_class", "unknown")
            if counts[category] < parameter("v2g.card.max_variant_witnesses"):
                kept.append(row)
                counts[category] += 1
        coding["records"] = kept
        coding["summary"] = {"total_records": len(coding_rows), "displayed_records": len(kept),
                             "omitted_records": len(coding_rows) - len(kept)}
        other = evidence["coloc"].pop("other_phenotypes", [])
        validate_background_rows(other)
        _attach_coloc_summaries(evidence["qtl"].get("records", []), evidence["coloc"].get("records", []), other,
                                scope_ids)
        evidence["coloc"]["phenotype_summary"] = _gene_phenotype_summary(evidence["coloc"].get("records", []), scope)
        displayed_partners = set()
        for category in ("coloc", "qtl", "e2g"):
            section = evidence[category]
            rows = section.get("records", [])
            if category == "coloc":
                rows = _coloc_order(rows, context.get("phenotype_scope"), context)
                chosen = _one_pair_per_signal_and_phenotype(rows, scope_ids)[:parameter("v2g.card.max_coloc_records")]
                displayed_partners = {row.get("qtl_study_locus_id") for row in chosen}
            elif category == "qtl":
                rows = chosen = _qtl_order(rows, displayed_partners, matcher)
            else:
                from .display_previews import e2g_order
                rows = chosen = list(e2g_order(rows, context))
                for row in rows:
                    biosample = {"biosampleId": row.get("biosample_id"), "biosampleName": row.get("biosample")}
                    row["_context_match"] = matcher(tissue=biosample, cell_type=biosample).get("matched", False)
            section["summary"] = _summary(rows, category)
            for row in rows:
                row.pop("_context_match", None)
            section["records"] = chosen
            section["summary"].update(displayed_records=len(chosen), omitted_records=len(rows) - len(chosen))
            section["omitted"] = len(rows) - len(chosen)
        evidence["coloc"]["other_phenotype_summary"] = {
            "qtl_signal_count": len(other),
            "phenotype_count": len({identity for row in other for identity in row["phenotype_ids"]}),
            "max_h4": max((row["max_h4"] for row in other), default=None),
        }
        expression = evidence["expression"]
        matched = expression.get("context_matched", [])
        expression_limit = parameter("v2g.card.max_expression_records")
        expression["context_matched"] = matched[:expression_limit]
        expression["context_matched_omitted"] = expression.get("context_matched_omitted", 0) + max(0, len(matched) - expression_limit)
        # Only representative colocalization pairs are reduced; at least one stays.
        coloc_rows = evidence["coloc"]["records"]
        while _bounded_bytes(card) / ratio > budget and len(coloc_rows) > 1:
            coloc_rows.pop()
            section = evidence["coloc"]
            section["summary"].update(displayed_records=len(coloc_rows),
                                      omitted_records=section["summary"]["total_records"] - len(coloc_rows))
            section["omitted"] = section["summary"]["omitted_records"]
        if _bounded_bytes(card) / ratio > budget:
            raise ValueError(f"Locus evidence exceeds processing range: irreducible card for {card['gene_id']} "
                             f"requires {math.ceil(_bounded_bytes(card) / ratio)} estimated tokens outside its complete "
                             f"QTL/E2G tables (per-gene limit {budget}).")
        receipts.append({"gene_id": card["gene_id"], "estimated_tokens": math.ceil(_bytes(card) / ratio),
                         "bounded_estimated_tokens": math.ceil(_bounded_bytes(card) / ratio),
                         "budget": budget, "within_budget": True,
                         "category_counts": {key: evidence[key]["summary"] for key in ("qtl", "coloc", "e2g")}})
    return result, {"policy": "complete_molecular_tables_v3", "per_gene": receipts,
        "total_estimated_tokens": sum(row["estimated_tokens"] for row in receipts)}
