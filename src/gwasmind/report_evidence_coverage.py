"""Explicit evidence presence in the report's immutable display facts.

Presence in a displayed fact is distinct from biological relevance, effect
direction, causality, and presence in a narrower earlier ranking view.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

from .report_gene_ownership import (
    gene_evidence_ownership_corrections as gene_evidence_ownership_corrections,
)


def molecular_qtl_overlap_methods(context):
    """Expose declared overlap types without inferring a method from model prose."""
    counts = {}
    for record in context.result.get("evidence", ()) or ():
        payload = record.get("payload") if isinstance(record, Mapping) else None
        if not isinstance(payload, Mapping) or payload.get("module") != "locus.qtl":
            continue
        binding = payload.get("context") or {}
        derived = payload.get("derived") or {}
        if not isinstance(binding, Mapping) or not isinstance(derived, Mapping):
            return []
        declaration = derived.get("molecular_overlap") or {}
        if not isinstance(declaration, Mapping):
            return []
        basis = binding.get("clpp_basis")
        if not isinstance(basis, str):
            return []
        scope = "query_variant_set_qtl_credible_set_overlap"
        kind = {"ld_approximation": "ld_weighted_qtl_overlap", "pip": "pip_weighted_qtl_overlap"}.get(basis)
        if (kind is None or binding.get("clpp_scope") != scope or declaration.get("basis") != basis
                or declaration.get("scope") != scope or declaration.get("statistic_type") != kind
                or (basis == "ld_approximation" and declaration.get("calibrated_colocalization_posterior") is not False)):
            return []
        counts[basis] = counts.get(basis, 0) + 1
    return [{"basis": basis, "statistic_type": "ld_weighted_qtl_overlap" if basis == "ld_approximation" else "pip_weighted_qtl_overlap",
             "scope": "query_variant_set_qtl_credible_set_overlap", "record_count": count,
             "computation": "sum(locus_variant_weight * QTL_member_PIP)",
             **({"calibrated_colocalization_posterior": False} if basis == "ld_approximation" else {})}
            for basis, count in sorted(counts.items())]


def clpp_method_label_corrections(text, context):
    """Correct an explicit method mislabel only with complete typed LD-overlap evidence."""
    methods = molecular_qtl_overlap_methods(context)
    if len(methods) != 1 or methods[0]["basis"] != "ld_approximation":
        return []
    records = [row for row in context.result.get("evidence", ()) if isinstance(row, Mapping)]
    if any((row.get("payload") or {}).get("module") == "locus.colocalisation"
           or row.get("kind") in {"colocalization", "colocalisation"} for row in records):
        return []
    pattern = re.compile(r"\b(?:many\s+)?(?:molQTL|molecular[- ]QTL)\s+overlaps\s+(?:rest|rely)\s+on\s+"
        r"(?P<span>(?:small\s+|low\s+)?colocali[sz]ation\s+posterior\s+probabilities\s+and\s+LD\s+approximation)"
        r"(?=\s+rather\s+than\s+direct\s+colocali[sz]ation[.;]|[.;]|$)", re.I)
    source_ids = sorted({row["evidence_id"] for row in records if row.get("evidence_id")
                         and (row.get("payload") or {}).get("module") == "locus.qtl"})
    fact_ids = [fact.fact_id for fact in context.facts if set(fact.evidence_ids).intersection(source_ids)]
    edits = []
    for match in pattern.finditer(text):
        prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
        if re.search(r"\b(?:if|whether|hypothes\w*|suppose|incorrect|external|another|different)\b", prefix, re.I):
            continue
        edits.append({"start": match.start("span"), "end": match.end("span"),
            "replacement": "LD-weighted molecular-QTL overlap scores", "metric": "qtl_overlap_method",
            "source_type": "typed_qtl_overlap_method", "source_fact_ids": fact_ids,
            "source_evidence_ids": source_ids, "method_binding": methods[0],
            "reason": "ld_weighted_qtl_overlap_is_not_a_calibrated_colocalization_posterior"})
    return edits


def molecular_evidence_presence(context):
    present = {}
    for fact in context.facts:
        data = fact.model_data
        subject = str(data.get("subject", ""))
        gene = re.match(r"^([A-Za-z][A-Za-z0-9_.-]*)\s+\(ENSG\d+\)$", subject)
        if not gene or data.get("evidence_module") != "Molecular QTL/Colocalization":
            continue
        if not re.match(r"^\s*[-*]?\s*(?:sc-)?(?:eQTL|sQTL|pQTL|molQTL)\s*\(", fact.markdown, re.IGNORECASE):
            continue
        if not re.search(r"\bP\s*=", fact.markdown):
            continue
        present.setdefault(gene.group(1), []).append(fact.fact_id)
    return present


def exclusive_molqtl_corrections(text, context):
    """Correct only unqualified exclusivity contradicted by supplied QTL rows.

    Qualifiers about a particular cell type, source variant, replication,
    colocalisation or strongest evidence never satisfy these exact patterns.
    No ranking or mechanistic interpretation is changed.
    """
    presence = molecular_evidence_presence(context)
    if len(presence) < 2:
        return []
    result = getattr(context, "result", {})
    ranked_ids = {row.get("gene_id") for row in result.get("ranked_genes", ()) if isinstance(row, Mapping)} if isinstance(result, Mapping) else set()
    ranked_presence = {}
    for fact in context.facts:
        subject = re.fullmatch(r"([A-Za-z][A-Za-z0-9_.-]*)\s+\((ENSG\d+)\)", str(fact.model_data.get("subject", "")))
        if subject and subject[2] in ranked_ids and fact.fact_id in presence.get(subject[1], ()):
            ranked_presence.setdefault(subject[1], []).append(fact.fact_id)
    edits = []
    def hypothetical(start):
        prefix = re.split(r"[.!?;\n]", text[:start])[-1]
        return bool(re.search(r"\b(?:if|whether|suppose|hypothesis|hypothetical|test)\b", prefix, re.IGNORECASE))

    source_qtl_witnesses = None
    for gene, facts in presence.items():
        name = r"(?<![\w-])[*`]*" + re.escape(gene) + r"[*`]*"
        if gene in ranked_presence and len(ranked_presence) > 1:
            pattern = (name + r"\s+is\s+(?P<span>the only ranked candidate)\s+"
                       r"with a retained molecular QTL\s*(?=[:.;])")
            for match in re.finditer(pattern, text, re.I):
                prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
                if hypothetical(match.start()) or re.search(
                    r"\b(?:in|within|among|under|for|restricted|filtered|context|tissue|cell|cohort)\b", prefix, re.I
                ):
                    continue
                witnesses = {other: identifiers for other, identifiers in ranked_presence.items() if other != gene}
                edits.append({"start": match.start("span"), "end": match.end("span"),
                    "replacement": "a ranked candidate",
                    "source_fact_ids": sorted(identifier for identifiers in witnesses.values() for identifier in identifiers),
                    "ranked_gene_witnesses": witnesses,
                    "reason": "unqualified_ranked_qtl_exclusivity_contradicted_by_displayed_facts"})
        pattern = (name + r"\s+is\s+(?P<span>the only candidate\s+with local molecular[- ]QTL evidence "
                   r"(?P<scope>linked to the locus at all|in the displayed set))\s*(?=[:.;])")
        for match in re.finditer(pattern, text):
            prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
            if hypothetical(match.start()) or re.search(
                r"\b(?:in|within|among|under|for|restricted|filtered|context|tissue|cell|cohort)\b", prefix, re.I
            ):
                continue
            if source_qtl_witnesses is None:
                source_qtl_witnesses = source_selected_qtl_witnesses(context)
            # A generic gene QTL cannot contradict the narrower local claim.
            # Require another gene's actual measured source-selected variant.
            witnesses = [row for row in source_qtl_witnesses if row["gene"] != gene]
            if not witnesses:
                continue
            scope = ("in the displayed set" if match.group("scope") == "in the displayed set"
                     else "linked to the locus")
            edits.append({"start": match.start("span"), "end": match.end("span"),
                "replacement": "a candidate with local molecular-QTL evidence " + scope,
                "source_fact_ids": sorted({row["fact_id"] for row in witnesses}),
                "witness_bindings": witnesses,
                "reason": "unqualified_exclusive_local_qtl_presence_contradicted"})
        pattern = name + r"\s+is\s+(?P<span>the only candidate)\s+with a measured molecular QTL link\s*(?=[:.;])"
        for match in re.finditer(pattern, text):
            if hypothetical(match.start()):
                continue
            edits.append({"start": match.start("span"), "end": match.end("span"),
                "replacement": "a candidate", "source_fact_ids": sorted({identifier
                    for rows in presence.values() for identifier in rows}),
                "reason": "unqualified_exclusive_qtl_presence_contradicted"})
        pattern = name + r"\s+is\s+(?P<span>supported only by)\s+proxy regulatory overlaps\b"
        for match in re.finditer(pattern, text):
            if hypothetical(match.start()):
                continue
            edits.append({"start": match.start("span"), "end": match.end("span"),
                "replacement": "supported by", "source_fact_ids": facts,
                "reason": "exclusive_regulatory_support_contradicted_by_displayed_qtl"})
        pattern = name + r"[^;\n]{0,600}?\b(?P<span>its displayed evidence lacks a measured variant-to-expression effect)(?=[;.,])"
        for match in re.finditer(pattern, text):
            scope = re.split(r"[.;\n]", text[:match.start()])[-1] + match.group()
            if hypothetical(match.start()) or any(other != gene and re.search(
                r"(?<![\w-])" + re.escape(other) + r"(?![\w-])", scope) for other in presence):
                continue
            edits.append({"start": match.start("span"), "end": match.end("span"),
                "replacement": "its displayed evidence includes a molecular QTL association "
                               + "{{cite_fact:" + facts[0] + "}}",
                "source_fact_ids": facts,
                "reason": "unqualified_expression_association_absence_contradicted"})
    return edits


def ranking_validation_claim_corrections(text, context):
    """Distinguish completed model generation from complete fact verification."""
    from .report_facts import ranking_provenance

    result = context.result
    if not isinstance(result, Mapping) or ranking_provenance(result).get("method") != "model":
        return []
    review = result.get("ranking_review", result.get("gate2_review"))
    if not isinstance(review, Mapping) or not (
        review.get("target") == "candidate_ranking"
        and review.get("status") in {"partially_verified", "needs_review"}
        and review.get("ranking_available") is True
        and review.get("fully_verified") is False
        and result.get("ranked_genes")
    ):
        return []
    edits = []
    pattern = (r"\b(?:the|this) ranking(?: below)? (?:is|was) generated by a "
               r"(?P<span>validated model run) over (?:this|the selected) locus(?=[;,.])")
    for match in re.finditer(pattern, text, re.I):
        if re.split(r"[.!?;\n]", text[:match.start()])[-1].strip():
            continue
        edits.append({"start": match.start("span"), "end": match.end("span"),
            "replacement": "model run", "metric": "ranking_validation_status",
            "source_type": "ranking_review", "source_fact_ids": [],
            "review_binding": {key: review[key] for key in (
                "target", "status", "ranking_available", "fully_verified")},
            "reason": "generation_completed_but_ranking_facts_partially_verified"})
    pattern = r"\breturned as ranked priorities by the (?P<span>validated model)(?=\s*:)"
    for match in re.finditer(pattern, text, re.I):
        prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
        if re.search(r"\b(?:if|whether|suppose|hypothesis|hypothetical|test)\b", prefix, re.I):
            continue
        edits.append({"start": match.start("span"), "end": match.end("span"),
            "replacement": "model", "metric": "ranking_validation_status",
            "source_type": "ranking_review", "source_fact_ids": [],
            "review_binding": {key: review[key] for key in (
                "target", "status", "ranking_available", "fully_verified")},
            "reason": "generation_completed_but_ranking_facts_partially_verified"})
    return edits


def source_selected_qtl_witnesses(context):
    """Bind a displayed QTL statistic to a selected identity, including proven swaps.

    The aggregate 'Source + LD' badge alone is insufficient: the displayed
    representative carrying the statistic must itself be source-selected.
    """
    from .report_facts import public_source_normalizations
    from .report_hard_facts import _SURFACE_MEASUREMENT, _metric_key, _surface_value
    from .source_normalization import validate_source_normalization
    from .qtl_measurement_binding import credible_set_lead_binding

    def canonical(value):
        return re.sub(r"^chr", "", str(value), flags=re.I).upper().replace(":", "_")

    originals = {canonical(row["variant_id"]): row["variant_id"] for row in context.source_observations
                 if isinstance(row, Mapping) and row.get("variant_id")}
    aliases = {canonical(row["original_id"]): canonical(row["canonical_id"])
               for row in public_source_normalizations(context.result)
               if canonical(row["original_id"]) in originals}
    ld = context.result.get("ld_expansion")
    if isinstance(ld, Mapping):
        for row in ld.get("source_identity_normalizations", ()):
            if isinstance(row, Mapping) and validate_source_normalization(row) and canonical(row["original_id"]) in originals:
                aliases[canonical(row["original_id"])] = canonical(row["canonical_id"])
    source_ids = {aliases.get(identifier, identifier) for identifier in originals}
    witnesses = []
    for fact in context.facts:
        data = fact.model_data
        if data.get("evidence_module") != "Molecular QTL/Colocalization" or data.get("deterministic_detail_truncated") is True:
            continue
        gene = re.match(r"^([A-Za-z][A-Za-z0-9_.-]*)\s+\(ENSG\d+\)$", str(data.get("subject", "")))
        if not gene or not re.match(r"^\s*[-*]?\s*(?:sc[-]?)?(?:e|s|p|tu|mol)QTL\s*\(", fact.markdown, re.I):
            continue
        measured_p = [item for item in _SURFACE_MEASUREMENT.finditer(fact.markdown) if _metric_key(item["metric"]) == "p"]
        if len(measured_p) != 1:
            continue
        try:
            value = _surface_value(measured_p[0])
        except (ArithmeticError, ValueError):
            continue
        if not value.is_finite() or not 0 < value <= 1:
            continue
        variant = re.search(r"displayed statistics:\s*representative source-selected variant\s+[`*]*"
            r"((?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+)", fact.markdown, re.I)
        study = re.search(r";\s*study\s+`([^`]+)`", fact.markdown)
        lead = credible_set_lead_binding(fact.markdown)
        displayed_variant = lead['variant_id'] if lead is not None else canonical(variant[1]) if variant is not None else None
        resolved_variant = aliases.get(displayed_variant, displayed_variant)
        if (variant is None and lead is None) or study is None or resolved_variant not in source_ids:
            continue
        witnesses.append({"fact_id": fact.fact_id, "gene": gene[1], "variant_id": resolved_variant,
            "displayed_variant_id": displayed_variant,
            "study_id": study[1], "source_observation_ids": [originals[key] for key in originals
                if aliases.get(key, key) == resolved_variant],
            "fact_sha256": hashlib.sha256(json.dumps({"fact_id": fact.fact_id, "markdown": fact.markdown,
                "subject": data.get("subject")}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
    return witnesses


def source_variant_measurement_absence_corrections(text, context):
    """Correct explicit source-measurement denial, not cohort or mechanism claims."""
    trait = str(context.source_trait or "").strip()
    if not trait:
        return []
    pattern = re.compile(r"\bnone of the displayed variant-to-gene or variant-to-expression links "
        r"are measurements on the uploaded " + re.escape(trait)
        + r" SNPs themselves(?=[.]|$)", re.I)
    exclusive = re.compile(r"\bonly\s+(?P<claim>the\s+(?:missense|coding)\s+annotation\s+for\s+"
        r"[*`]*[A-Za-z][A-Za-z0-9_.-]*[*`]*\s+is\s+tied\s+directly\s+to\s+"
        r"a\s+(?:selected|source-selected|uploaded)\s+SNP)(?=[.;]|$)", re.I)
    matches = [*pattern.finditer(text), *exclusive.finditer(text)]
    if not matches:
        return []
    witnesses = source_selected_qtl_witnesses(context)
    if not witnesses:
        return []
    edits = []
    for match in matches:
        sentence_prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
        # A tissue/gene/study/cohort qualifier can narrow the claim. The exact
        # absolute sentence, or its clause after a colon, is the supported case.
        clause_prefix = sentence_prefix.rsplit(":", 1)[-1].strip()
        if clause_prefix or sentence_prefix.strip().casefold() in {"hypothesis:", "hypothetical:"} or re.search(r"\b(?:if|whether|suppose|incorrect|cohort|participants|samples|dataset|for|in|from|within)\b",
                                      sentence_prefix, re.I):
            continue
        first = witnesses[0]
        replacement = ("some displayed molecular QTL measurements refer to source-selected SNPs, including "
            f"{first['gene']} at `{first['variant_id']}` " + "{{cite_fact:" + first["fact_id"] + "}}"
            "; these retrieved QTL-study measurements do not establish that molecular traits were measured in the uploaded GWAS")
        if match.re is exclusive:
            replacement = match["claim"] + "; " + replacement
        edits.append({"start": match.start(), "end": match.end(), "replacement": replacement,
            "metric": "source_variant_measurement_presence", "source_type": "display_fact_registry",
            "source_fact_ids": [row["fact_id"] for row in witnesses], "witness_bindings": witnesses,
            "reason": "absolute_source_variant_measurement_absence_contradicted_by_displayed_qtl"})
    return edits
