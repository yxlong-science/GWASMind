"""Narrow report corrections supported by named-gene, typed native witnesses.

This module never changes the selected genes, their order, or a mechanistic
preference. Ranking rationale is not an evidence source.
"""
from __future__ import annotations

import re
from collections.abc import Mapping


def _cards(context):
    audit = context.result.get("candidate_audit") or {}
    from .card_store import cards_from_audit
    rows = cards_from_audit(audit)
    return {row["gene_id"]: row for row in rows if isinstance(row, Mapping)
            and isinstance(row.get("identity"), Mapping) and row.get("gene_id")
            and row["identity"].get("gene_id") == row["gene_id"]}


def _symbols(context, cards):
    result = {gene: row["identity"]["symbol"] for gene, row in cards.items()
              if isinstance(row["identity"].get("symbol"), str)}
    for fact in context.facts:
        subject = re.fullmatch(r"([A-Za-z][A-Za-z0-9_.-]*)\s+\((ENSG\d+)\)",
                               str(fact.model_data.get("subject", "")))
        if subject:
            result.setdefault(subject[2], subject[1])
    return result


def _name(symbol):
    return r"(?<![\w-])[*`]*" + re.escape(symbol) + r"[*`]*(?![\w-])"


def _hypothetical(text, start):
    prefix = re.split(r"[.!?;\n]", text[:start])[-1]
    return bool(re.search(r"\b(?:if|whether|suppose|hypothes\w*|incorrect\w*|hypothetical|test)\b", prefix, re.I))


def _single_gene_clause(text, match, symbol, symbols):
    return (not _hypothetical(text, match.start())
            and not re.search(r"\b(?:may|might|could|would|whether|if|hypothes\w*)\b", match["prefix"], re.I)
            and not any(
        other != symbol and re.search(_name(other), match.group("prefix")) for other in symbols.values()))


def _utr_owner_corrections(text, context, cards, symbols):
    """A copied annotation can be corrected only through its actual native owner."""
    records = [row for row in context.result.get("evidence", ()) if isinstance(row, Mapping)]
    annotations = {}
    for row in records:
        payload = row.get("payload") or {}
        if not isinstance(payload, Mapping) or payload.get("module") != "locus.variant_consequence":
            continue
        upstream, target = payload.get("upstream"), payload.get("object")
        if not isinstance(upstream, Mapping) or not isinstance(target, Mapping):
            continue
        raw, owner = upstream.get("raw"), target.get("id")
        if (not isinstance(raw, Mapping) or not isinstance(raw.get("target"), Mapping)
                or raw.get("consequence") != "5_prime_UTR_variant"
                or raw.get("isEnsemblCanonical") is not True or owner != row.get("gene_id")
                or raw["target"].get("id") != owner or owner not in symbols
                or raw.get("variantId") != row.get("variant_id")):
            continue
        annotations[row.get("evidence_id")] = row
    edits = []
    for ranked in context.result.get("ranked_genes", ()):
        gene = ranked.get("gene_id") if isinstance(ranked, Mapping) else None
        if gene not in symbols:
            continue
        # A genuine same-gene annotation must never be contradicted merely
        # because another transcript or gene also owns this consequence.
        if any(row["gene_id"] == gene for row in annotations.values()):
            continue
        cited = [annotations[identifier] for identifier in ranked.get("evidence_ids", ())
                 if identifier in annotations and annotations[identifier]["gene_id"] != gene]
        owners = {row["gene_id"] for row in cited}
        if len(owners) != 1:
            continue
        owner = next(iter(owners))
        pattern = (_name(symbols[gene]) + r"(?P<prefix>[^.;!?\n]{0,400}?)\b"
            r"(?P<span>contains a canonical 5[′’']?[- ]?UTR variant in a (?:LD )?proxy)(?=[,.;]|$)")
        for match in re.finditer(pattern, text, re.I):
            if not _single_gene_clause(text, match, symbols[gene], symbols):
                continue
            source_ids = sorted(row["evidence_id"] for row in cited)
            fact_ids = [fact.fact_id for fact in context.facts if set(fact.evidence_ids).intersection(source_ids)]
            edits.append({"start": match.start("span"), "end": match.end("span"),
                "replacement": f"does not own the cited canonical 5′UTR annotation (the record is assigned to {symbols[owner]})",
                "metric": "variant_consequence_gene_owner", "source_type": "typed_native_evidence",
                "source_fact_ids": fact_ids, "source_evidence_ids": source_ids,
                "gene_id": gene, "owner_gene_id": owner,
                "witness_bindings": [{"evidence_id": row["evidence_id"], "gene_id": row["gene_id"],
                    "variant_id": row["variant_id"], "consequence": "5_prime_UTR_variant",
                    "canonical_transcript": True} for row in cited],
                "reason": "claimed_gene_borrowed_another_genes_cited_annotation"})
    return edits


def _nearest_metric_corrections(text, context, cards, symbols):
    audit = context.result.get("candidate_audit") or {}
    if not isinstance(audit, Mapping) or audit.get("candidate_discovery_complete") is not True:
        return []
    if set(audit.get("candidate_ids", ())) != set(cards):
        return []
    if any(not isinstance(row["identity"].get("biotype"), str) or not row["identity"]["biotype"] for row in cards.values()):
        return []
    coding = {gene: row for gene, row in cards.items() if row["identity"].get("biotype") == "protein_coding"}
    if len(coding) < 2 or len({row.get("auxiliary", {}).get("chromosome") for row in coding.values()}) != 1:
        return []
    distances = {}
    for gene, row in coding.items():
        auxiliary = row.get("auxiliary") or {}
        pair = tuple(auxiliary.get(key) for key in ("body_distance_bp", "tss_distance_bp"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
               or value == float("inf") or value != value for value in pair):
            return []
        distances[gene] = pair
    nearest_body = [gene for gene, pair in distances.items() if pair[0] == min(value[0] for value in distances.values())]
    nearest_tss = [gene for gene, pair in distances.items() if pair[1] == min(value[1] for value in distances.values())]
    if len(nearest_body) != 1 or len(nearest_tss) != 1 or nearest_body == nearest_tss:
        return []
    gene, tss_gene = nearest_body[0], nearest_tss[0]
    if gene not in symbols or tss_gene not in symbols:
        return []
    pattern = (_name(symbols[gene]) + r"(?P<prefix>[^.;!?\n]{0,300}?)\b"
        r"(?P<span>the nearest protein-coding gene)(?=\s+(?:and|but|because)\b|[,. ;]|$)")
    edits = []
    for match in re.finditer(pattern, text, re.I):
        if not _single_gene_clause(text, match, symbols[gene], symbols):
            continue
        suffix = text[match.end("span"):]
        if re.match(r"\s+(?:by|to|from|according|based|for|among|within|in|of)\b", suffix, re.I):
            continue
        edits.append({"start": match.start("span"), "end": match.end("span"),
            "replacement": "the nearest protein-coding gene by gene-body distance among retrieved candidates "
                f"({symbols[tss_gene]} is nearest by TSS distance)",
            "metric": "nearest_gene_distance_scope", "source_type": "native_candidate_coordinates",
            "source_fact_ids": [], "gene_id": gene,
            "witness_bindings": [{"gene_id": identity, "body_distance_bp": pair[0], "tss_distance_bp": pair[1]}
                for identity, pair in sorted(distances.items())],
            "reason": "gene_body_nearest_and_tss_nearest_are_distinct"})
    return edits


def _proxy_only_corrections(text, context, symbols):
    from .report_evidence_coverage import source_selected_qtl_witnesses

    witnesses = source_selected_qtl_witnesses(context)
    if not witnesses or not symbols:
        return []
    name = "(?:" + "|".join(_name(symbol) for symbol in sorted(set(symbols.values()), key=len, reverse=True)) + ")"
    pattern = (r"(?P<genes>" + name + r"(?:\s*,?\s+(?:and\s+)?" + name + r")*)\s+"
        r"(?P<span>rel(?:y|ies)\s+(?:(?:only|exclusively|entirely)\s+)?on regulatory or QTL evidence "
        r"mediated through LD proxies)(?=[.;]|$)")
    edits = []
    for match in re.finditer(pattern, text, re.I):
        if _hypothetical(text, match.start()):
            continue
        names = [symbol for symbol in symbols.values() if re.search(_name(symbol), match["genes"])]
        observed = [row for row in witnesses if row["gene"] in names]
        if not observed:
            continue
        # Do not erase a scope qualifier about a particular assay or biosample.
        prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1].rsplit("while", 1)[-1]
        if re.search(r"\b(?:in|within|among|under|for|restricted|filtered|context|tissue|cell|cohort)\b", prefix, re.I):
            continue
        anchors = {}
        for row in observed:
            anchors.setdefault(row["gene"], row["fact_id"])
        listed = " and ".join(gene + " {{cite_fact:" + identifier + "}}" for gene, identifier in anchors.items())
        edits.append({"start": match.start("span"), "end": match.end("span"),
            "replacement": ("has" if len(names) == 1 else "have") + " regulatory or QTL evidence involving LD proxies; "
                + "source-selected molecular QTL measurements are also displayed for " + listed,
            "metric": "gene_source_variant_measurement_presence", "source_type": "display_fact_registry",
            "source_fact_ids": [row["fact_id"] for row in observed], "witness_bindings": observed,
            "reason": "gene_level_proxy_only_support_contradicted_by_measured_source_qtl"})
    return edits


def gene_evidence_ownership_corrections(text, context):
    """Locally clarify verified ownership/scope without rewriting a ranking."""
    cards = _cards(context)
    symbols = _symbols(context, cards)
    return [*_utr_owner_corrections(text, context, cards, symbols),
            *_nearest_metric_corrections(text, context, cards, symbols),
            *_proxy_only_corrections(text, context, symbols)]
