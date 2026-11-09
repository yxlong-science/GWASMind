"""Bounded, user-facing explanations of existing Gate 2 review findings.

This module projects review receipts. It never decides whether a scientific
claim is valid, changes a ranking, or reads model reasoning or prompt context.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from .gate2_hallucination import REMOVED_VALUE_LABEL

MAX_ITEMS = 64
MAX_EXCERPT = 240
_MAX_INPUT_ITEMS = 4096
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,159}\Z")
_GENE = re.compile(r"ENSG\d{11}(?:\.\d+)?\Z")
_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9.-]{0,39}\Z")
_VARIANT = re.compile(r"(?:chr)?(?:\d{1,2}|X|Y|MT)[_:]\d+[_:][ACGTN-]+[_:][ACGTN-]+\Z", re.I)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?\Z")
_LD_RANGE = re.compile(r"(?:0(?:\.\d+)?|1(?:\.0+)?)\s*(?:[-–—]|to)\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\Z")
_LD_UNPARSED = re.compile(r"[+-]?(?:\d[\d.eE+–—-]*|\.\d[\d.eE+–—-]*)\Z")
_NONFINITE_LITERAL = re.compile(r"[+-]?(?:NaN|Infinity|inf|∞)%?\Z", re.I)
_COUNT_LITERAL = re.compile(r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\Z", re.I)
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻−", "0123456789+--")
_SENSITIVE = re.compile(r"(?:api[_ -]?key|authorization|bearer\s|password|<think>|reasoning_content)", re.I)
_METRICS = {
    "r2": "LD r²", "p": "P value", "beta": "Beta", "or": "Odds ratio",
    "h3": "Colocalization H3", "h4": "Colocalization H4", "pip": "Posterior probability",
    "clpp": "CLPP", "score": "Score", "se": "Standard error", "z": "Z score",
    "logodds": "Log odds", "dnase": "DNase", "hic": "Hi-C", "tss": "TSS distance",
    "chromosome": "Chromosome", "position": "Position", "coordinate": "Coordinate",
    "gene_id": "Gene identifier", "variant_id": "Variant identifier", "citation": "Evidence reference",
    "evidence_presence": "Evidence coverage statement", "ranking_validation_status": "Ranking review statement",
    "source_significance_count": "Source GWAS count", "source_variant_count": "Selected SNP count", "rank": "Gene rank",
    "candidate_pool_count": "Candidate gene count", "shortlist_count": "Shortlisted gene count", "ranked_count": "Prioritized gene count",
    "variant_consequence_gene_owner": "Variant annotation owner",
    "nearest_gene_distance_scope": "Nearest-gene distance definition",
    "gene_source_variant_measurement_presence": "Source-SNP QTL evidence",
}
_REASONS = {
    "candidate_evidence_absence_contradicted": "The LLM's stated absence conflicts with retained evidence for this candidate. Its selection status and ranking were not changed.",
    "numeric_domain_error": "The value lies outside the valid range of this statistic.",
    "numeric_value_mismatch": "The stated value conflicts with the uniquely identified numeric source.",
    "configured_threshold_mismatch": "The stated inclusion threshold differs from the threshold frozen with the evidence.",
    "threshold_source_unavailable": "The saved evidence does not identify the stated inclusion threshold.",
    "measurement_relation_ambiguous": "Several source measurements remain possible; their ownership must be clarified before judging the value.",
    "ld_binding_unclear": "The LD pair or reference is not uniquely specified. The original value is retained; it is not classified as numerically wrong.",
    "ld_evidence_conflict": "The complete LD assertion conflicts with the frozen source-to-proxy LD evidence for its identified scope.",
    "ld_data_unavailable": "The saved evidence does not contain enough LD information to verify this assertion. The original statement is retained.",
    "measurement_not_uniquely_bound": "The statement cannot be linked to one unambiguous measurement in the saved evidence.",
    "measurement_source_not_verified": "The stated measurement could not be verified against its cited or identified source.",
    "measurement_unit_not_verified": "The stated measurement unit does not match the unit recorded in its identified source.",
    "measurement owner or source is not uniquely verified": "The measurement owner or source could not be uniquely identified.",
    "LD pair or cited record owner is not uniquely verified": "The source SNP–proxy SNP pair or cited record could not be uniquely verified for this LD value.",
    "unknown_fact_id": "The cited evidence reference is not in the saved evidence registry.",
    "gene outside the candidate universe": "The gene identifier is not in the retrieved candidate set.",
    "unverified evidence citation or owner": "The evidence reference is unavailable or does not belong to the stated gene.",
    "unverified evidence citation": "The evidence reference is not available in the evidence shown to the model.",
    "input_precision_normalized": "The number was normalized to the precision recorded in its source.",
    "source_precision_normalized": "The number was normalized to the uploaded source measurement.",
    "unique_declared_metric_value": "The number was corrected from its uniquely identified evidence source.",
    "unique_source_variant_measurement": "The number was corrected from the matching uploaded source variant.",
    "unique_cited_fact_measurement": "The number was corrected from the uniquely cited evidence record.",
    "unique_raw_measurement_numeric_correction_v1": "The number was corrected from its uniquely identified evidence source.",
    "bound_ld_edge_numeric_correction_v1": "The LD value was corrected from the uniquely identified source SNP–proxy SNP pair.",
    "unqualified_exclusive_local_qtl_presence_contradicted": "Other displayed genes also have molecular QTL evidence, so the exclusive statement was corrected.",
    "unqualified_exclusive_qtl_presence_contradicted": "Other displayed genes also have molecular QTL evidence, so the exclusive statement was corrected.",
    "generation_completed_but_ranking_facts_partially_verified": "Model generation completed, but some ranking facts remain unverified; the review wording was corrected.",
    "unverified_assertion_removed": "The unsupported value was removed; the final report does not assert it.",
    "selected_source_total_contradicted_by_uploaded_selection": "The selected SNP total was corrected from the uploaded selection; LD proxies were not included.",
    "candidate_scope_total_contradicted_by_recorded_counts": "The gene total was corrected from its recorded candidate, shortlist or returned-ranking scope.",
    "report_gene_identifier_not_in_evidence": "This gene identifier could not be matched to the saved evidence used for this report.",
    "report_variant_identifier_not_in_evidence": "This SNP identifier could not be matched to the saved variants or evidence used for this report.",
    "report_proxy_exclusion_not_in_recorded_scope": "The saved LD records do not support describing this SNP as an unverified, excluded proxy.",
    "report_gene_symbol_identifier_conflict": "The stated gene symbol and Ensembl identifier do not match the saved gene identity.",
    "claimed_gene_borrowed_another_genes_cited_annotation": "The cited variant annotation belongs to another gene; its gene attribution was corrected.",
    "gene_body_nearest_and_tss_nearest_are_distinct": "Gene-body distance and TSS distance identify different nearest genes; the distance definition was clarified.",
    "gene_level_proxy_only_support_contradicted_by_measured_source_qtl": "Displayed QTL measurements also involve selected source SNPs; the proxy-only description was corrected.",
}

_TYPED_SEMANTIC = {
    "variant_consequence_gene_owner": ("typed_native_evidence", "claimed_gene_borrowed_another_genes_cited_annotation"),
    "nearest_gene_distance_scope": ("native_candidate_coordinates", "gene_body_nearest_and_tss_nearest_are_distinct"),
    "gene_source_variant_measurement_presence": ("display_fact_registry", "gene_level_proxy_only_support_contradicted_by_measured_source_qtl"),
}

_OBJECTIVE_METRICS = frozenset(_METRICS) - {"evidence_presence", "ranking_validation_status", "rank"}
_VALUE_TOKEN = re.compile(r"(?<![A-Za-z0-9_.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?(?![A-Za-z0-9_])")


def _text(value: Any, limit: int = 240) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ""
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value)).strip()
    if _SENSITIVE.search(value):
        return ""
    return value[:limit]


def _payload(value: Any) -> Mapping:
    if isinstance(value, str) and len(value) <= 2_000_000:
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return {}
    return value if isinstance(value, Mapping) else {}


def _visible_field(payload: Mapping, field: str, target: str) -> tuple[Any, str, Mapping] | None:
    if target == "final_report":
        if field == "summary_markdown":
            return payload.get(field), "Report summary", {}
        if field == "limitations_uncertainty.markdown":
            limits = payload.get("limitations_uncertainty")
            return (limits.get("markdown") if isinstance(limits, Mapping) else None), "Report limitations", {}
        return None
    if field == "summary":
        return payload.get(field), "Ranking summary", {}
    match = re.fullmatch(r"(ranked_genes|claims|candidate_decisions)\[(\d{1,4})\]\.(rationale|reason|gene_id|evidence_ids|text|limitations(?:\[\d{1,4}\])?)", field)
    if not match:
        return None
    group, index, member = match.groups()
    rows = payload.get(group)
    row = rows[int(index)] if isinstance(rows, (list, tuple)) and int(index) < len(rows) else {}
    if not isinstance(row, Mapping):
        row = {}
    label = (f"Gene ranking {int(index) + 1}" if group == "ranked_genes" else
             f"Candidate selection {int(index) + 1}" if group == 'candidate_decisions' else f"Ranking claim {int(index) + 1}")
    if limit := re.fullmatch(r"limitations\[(\d+)\]", member):
        values = row.get("limitations")
        at = int(limit[1])
        value = values[at] if isinstance(values, (list, tuple)) and at < len(values) else None
        return value, label + " · Limitation", row
    if member == "limitations":
        return None
    return row.get(member), label + " · " + {"rationale": "Rationale", "gene_id": "Gene identifier",
        "evidence_ids": "Evidence references", "text": "Statement", "reason": "Selection reason"}[member], row


def _finding_span(text: Any, finding: Mapping, original: str) -> tuple[int, int] | None:
    span = finding.get("span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        span = [finding.get("start"), finding.get("end")]
    start, end = span
    numeric_span = isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool) and 0 <= start < end <= 2_000_000
    if not isinstance(text, str):
        return (start, end) if numeric_span else None
    valid = numeric_span and end <= len(text) and original in text[start:end]
    if not valid:
        needle = finding.get("original_span")
        if not isinstance(needle, str) or original not in needle or text.count(needle) != 1:
            needle = original
        if text.count(needle) != 1:
            return None
        start, end = text.index(needle), text.index(needle) + len(needle)
    return start, end


def _excerpt(text: Any, finding: Mapping, original: str) -> str:
    if not isinstance(text, str) or not original or _SENSITIVE.search(text):
        return ""
    span = _finding_span(text, finding, original)
    if span is None:
        return ""
    start, end = span
    boundary = re.compile(r"[.!?](?=\s|\{\{cite_fact:)|\n")
    preceding = list(boundary.finditer(text[:start]))
    left = preceding[-1].end() if preceding else 0
    following = boundary.search(text, end)
    right = following.end() if following else len(text)
    clean = lambda value: re.sub(r"\{\{cite_fact:[^}]+\}\}", "", value)
    value = clean(text[left:right])
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > MAX_EXCERPT:
        # Preserve the subject and the original measurement, with explicit
        # elision when the visible sentence does not fit the compact receipt.
        offset = value.find(original)
        if offset >= 0 and offset + len(original) <= MAX_EXCERPT - 4:
            value = value[:MAX_EXCERPT - 2].rstrip() + " …"
        elif offset >= 0:
            head = value[:65].rstrip()
            tail = value[max(0, offset - 65):offset + len(original) + 60].strip()
            value = head + " … " + tail + (" …" if offset + len(original) + 60 < len(value) else "")
        else:
            return ""
    return _text(value, MAX_EXCERPT)


def _identifiers(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and _ID.fullmatch(item)))[:8]


def _typed_semantic_receipt(finding: Mapping, metric: str, *, projected: bool) -> bool:
    """Validate a receipt's declared type/bindings, without publishing raw witnesses."""
    expected_source, expected_reason = _TYPED_SEMANTIC[metric]
    if finding.get("source_type") != expected_source or finding.get("reason") != expected_reason:
        return False
    if projected:
        # Public projections keep only the finite type/reason and stable issue
        # identity. Raw native records remain in the original machine receipt.
        return True
    rows = finding.get("witness_bindings")
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 200 or any(not isinstance(row, Mapping) for row in rows):
        return False
    valid_id = lambda value: isinstance(value, str) and bool(_ID.fullmatch(value))
    valid_gene = lambda value: isinstance(value, str) and bool(_GENE.fullmatch(value))
    valid_variant = lambda value: isinstance(value, str) and bool(_VARIANT.fullmatch(value))
    ids = finding.get("source_fact_ids", ())
    if not isinstance(ids, (list, tuple)) or len(ids) > 200 or not all(valid_id(value) for value in ids):
        return False
    if metric == "variant_consequence_gene_owner":
        gene, owner = finding.get("gene_id"), finding.get("owner_gene_id")
        evidence = finding.get("source_evidence_ids")
        return (valid_gene(gene) and valid_gene(owner) and gene != owner
            and isinstance(evidence, (list, tuple)) and 1 <= len(evidence) <= 200 and all(valid_id(value) for value in evidence)
            and all(valid_id(row.get("evidence_id")) for row in rows)
            and {row.get("evidence_id") for row in rows} == set(evidence)
            and all(row.get("gene_id") == owner and valid_variant(row.get("variant_id"))
                and row.get("consequence") == "5_prime_UTR_variant" and row.get("canonical_transcript") is True for row in rows))
    if metric == "nearest_gene_distance_scope":
        gene = finding.get("gene_id")
        return (valid_gene(gene) and len(rows) >= 2
            and all(valid_gene(row.get("gene_id")) for row in rows)
            and len({row["gene_id"] for row in rows}) == len(rows)
            and gene in {row["gene_id"] for row in rows}
            and all(isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)
                and 0 <= row[key] <= 1e12 and math.isfinite(row[key])
                for row in rows for key in ("body_distance_bp", "tss_distance_bp")))
    return (bool(ids) and all(valid_id(row.get("fact_id")) for row in rows)
        and {row.get("fact_id") for row in rows} == set(ids)
        and all(isinstance(row.get("gene"), str) and _SYMBOL.fullmatch(row["gene"])
            and valid_variant(row.get("variant_id")) and valid_variant(row.get("displayed_variant_id"))
            and valid_id(row.get("study_id")) and isinstance(row.get("source_observation_ids"), (list, tuple))
            and 1 <= len(row["source_observation_ids"]) <= 100
            and all(valid_variant(value) for value in row["source_observation_ids"])
            and isinstance(row.get("fact_sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", row["fact_sha256"])
            for row in rows))


def _project_finding(finding: Any, *, target: str, disposition: str,
                     visible: Mapping, gene_labels: Mapping, projected: bool = False) -> dict | None:
    if not isinstance(finding, Mapping):
        return None
    field = finding.get("field")
    if not isinstance(field, str):
        return None
    field_data = _visible_field(visible, field, target)
    if field_data is None:
        return None
    raw_text, location, owner = field_data
    metric = finding.get("metric") or ("gene_id" if field.endswith(".gene_id") else
        "citation" if field.endswith(".evidence_ids") else "")
    if not isinstance(metric, str) or metric not in _METRICS:
        return None
    semantic = metric in _TYPED_SEMANTIC
    if semantic and (target != "final_report" or disposition != "corrected"
                     or not _typed_semantic_receipt(finding, metric, projected=projected)):
        return None
    raw_original = finding.get("original_value")
    if metric == "citation" and isinstance(raw_original, (list, tuple)):
        original = ", ".join(_identifiers(raw_original))
    else:
        original = _text(raw_original)
    if not original:
        return None
    numeric = metric not in {"gene_id", "variant_id", "citation", "evidence_presence", "ranking_validation_status"} and not semantic
    normalized_number = re.sub(r"\s+", "", original).translate(_SUPERSCRIPT)
    normalized_number = re.sub(r"(?:×|x|\*)10", "e", normalized_number).replace("^", "").replace("{", "").replace("}", "")
    word_count = metric in {"source_variant_count", "candidate_pool_count", "shortlist_count", "ranked_count"} and _COUNT_LITERAL.fullmatch(normalized_number)
    ld_range = metric == "r2" and _LD_RANGE.fullmatch(original) and (projected
        or finding.get("rule") == "typed_ld_assertion_v1")
    ld_unparsed = metric == "r2" and _LD_UNPARSED.fullmatch(original) and (projected
        or finding.get("rule") == "typed_ld_assertion_v1")
    if numeric and not (_NUMBER.fullmatch(normalized_number) or _NONFINITE_LITERAL.fullmatch(normalized_number) or word_count or ld_range or ld_unparsed):
        return None
    if (numeric and _NONFINITE_LITERAL.fullmatch(normalized_number) and not projected
            and (not isinstance(raw_text, str) or _finding_span(raw_text, finding, original) is None)):
        # Invalid scalar literals explain a real displayed assertion. An
        # unbound NaN/Inf in an arbitrary audit object is not a public finding.
        return None
    if not numeric and metric in {"gene_id", "variant_id", "citation"}:
        ids = original.split(", ")
        if any(not _ID.fullmatch(item) for item in ids):
            return None
    if not projected and metric in {"evidence_presence", "ranking_validation_status"} and (
            not isinstance(raw_text, str) or original not in raw_text):
        return None
    if semantic and not projected and (not isinstance(raw_text, str)
            or _finding_span(raw_text, finding, original) is None):
        return None
    code = finding.get("reason") or finding.get("rule")
    reason = _REASONS.get(code) if isinstance(code, str) else None
    if metric == "r2" and disposition == "unverified" and code not in {"ld_binding_unclear", "ld_evidence_conflict", "ld_data_unavailable"}:
        code = "ld_pair_not_uniquely_verified"
        reason = _REASONS["LD pair or cited record owner is not uniquely verified"]
    elif reason is None:
        code = "source_value_corrected" if disposition == "corrected" else "source_not_verified"
        reason = "The stated fact was corrected using the saved evidence." if disposition == "corrected" else "The stated fact could not be verified against the saved evidence."
    raw_replacement = finding.get("replacement") or finding.get("display_replacement") or finding.get("corrected_value")
    if semantic and isinstance(raw_replacement, str):
        raw_replacement = re.sub(r"\{\{cite_fact:[^}]+\}\}", "", raw_replacement)
        raw_replacement = re.sub(r"\s+", " ", raw_replacement).strip()
    replacement = _text(raw_replacement)
    if disposition == "unverified" and not replacement:
        replacement = "Reference retained for review" if metric in {"gene_id", "variant_id", "citation"} else _METRICS[metric] + " not verified"
    if not replacement:
        return None
    item = {"target": target, "disposition": disposition, "location": location,
        "field": field, "metric": metric, "metric_label": _METRICS[metric],
        "original_value": original, "replacement": replacement,
        "reason_code": code, "reason": reason}
    from .numeric_assertions import finding_category
    item["finding_category"] = finding_category(code, disposition=disposition)
    if code == 'candidate_evidence_absence_contradicted':
        item['finding_category'] = 'evidence_contradiction'
    numeric_ids = [identifier for identifier in _identifiers(finding.get("numeric_fact_ids"))
                   if re.fullmatch(r"nf:[a-f0-9]{64}", identifier)]
    if numeric_ids:
        item["numeric_fact_ids"] = numeric_ids
    if semantic:
        item["source_type"] = _TYPED_SEMANTIC[metric][0]
    occurrence = _finding_span(raw_text, finding, original)
    if occurrence is not None:
        item["occurrence"] = {"start": occurrence[0], "end": occurrence[1]}
    gene_id = owner.get("gene_id")
    if metric != "gene_id" and isinstance(gene_id, str) and _GENE.fullmatch(gene_id):
        item["gene_id"] = gene_id
        symbol = gene_labels.get(gene_id) or owner.get("gene_symbol")
        if isinstance(symbol, str) and _SYMBOL.fullmatch(symbol):
            item["gene_symbol"] = symbol
            item["location"] = re.sub(r"Gene ranking \d+", f"Gene ranking · {symbol}", location, count=1)
    excerpt = _excerpt(raw_text, finding, original)
    if excerpt:
        item["excerpt"] = excerpt
    ids = _identifiers(finding.get("source_fact_ids") or finding.get("evidence_ids"))
    if ids:
        item["evidence_ids"] = ids
    variant = finding.get("variant_id")
    if disposition == "corrected" and isinstance(variant, str) and _VARIANT.fullmatch(variant):
        item["variant_ids"] = [variant]
    stable = {key: item[key] for key in ("target", "disposition", "field", "metric", "original_value", "replacement", "reason_code")}
    if "occurrence" in item:
        stable["occurrence"] = item["occurrence"]
    item["issue_id"] = "fact-" + hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]
    return item


def _envelope(items: list[dict], *, rejected: int = 0, unexamined: int = 0) -> dict:
    unique = {item["issue_id"]: item for item in items}
    rows = sorted(unique.values(), key=lambda item: (item["disposition"] != "unverified", item["target"], item["field"], item["issue_id"]))
    return {"schema_version": "1", "unresolved_count": sum(item["disposition"] == "unverified" for item in rows),
        "corrected_count": sum(item["disposition"] == "corrected" for item in rows),
        "items": rows[:MAX_ITEMS], "omitted_count": max(0, len(rows) - MAX_ITEMS),
        "rejected_count": rejected, "unexamined_count": unexamined}


def project_fact_issues(audit: Any, *, target: str, visible_output: Any = None,
                        gene_labels: Mapping | None = None) -> dict:
    """Project audit fields only; visible output is used solely for short quotes."""
    if target not in {"final_report", "candidate_ranking"}:
        raise ValueError("Unknown fact-review target")
    audit = _payload(audit)
    visible = _payload(visible_output)
    groups = [("unverified_measurements" if target == "final_report" else "unverified_fields", "unverified"),
              ("corrections", "corrected")]
    items, rejected, unexamined = [], 0, 0
    for key, disposition in groups:
        rows = audit.get(key, ())
        if not isinstance(rows, (list, tuple)):
            continue
        unexamined += max(0, len(rows) - _MAX_INPUT_ITEMS)
        for finding in rows[:_MAX_INPUT_ITEMS]:
            removed = disposition == "unverified" and isinstance(finding, Mapping) and finding.get("removed") is True
            if removed:
                finding = {**finding, "reason": "unverified_assertion_removed"}
            projected = _project_finding(finding, target=target, disposition="corrected" if removed else disposition,
                visible=visible, gene_labels=gene_labels if isinstance(gene_labels, Mapping) else {})
            if projected is None:
                rejected += 1
            else:
                items.append(projected)
    return _envelope(items, rejected=rejected, unexamined=unexamined)


def combine_fact_issues(*projections: Mapping) -> dict:
    """Combine generated public projections without repeating the same finding."""
    items, rejected, unexamined, omitted = [], 0, 0, 0
    hidden_unresolved = hidden_corrected = 0
    seen = set()
    for projection in projections:
        if not isinstance(projection, Mapping) or projection.get("schema_version") != "1":
            continue
        rows = projection.get("items", ())
        if not isinstance(rows, (list, tuple)):
            continue
        fingerprint = tuple(str(item.get("issue_id", "")) for item in rows[:MAX_ITEMS] if isinstance(item, Mapping))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        counts = {"unverified": 0, "corrected": 0}
        for item in rows[:MAX_ITEMS]:
            if not isinstance(item, Mapping) or item.get("disposition") not in counts or item.get("target") not in {"final_report", "candidate_ranking"}:
                rejected += 1
                continue
            occurrence = item.get("occurrence")
            position = occurrence if isinstance(occurrence, Mapping) else {}
            safe = _project_finding({**dict(item), **position, "reason": item.get("reason_code"),
                "corrected_value": item.get("replacement")}, target=item["target"],
                disposition=item["disposition"], visible={}, gene_labels={}, projected=True)
            if safe is None or safe["issue_id"] != item.get("issue_id"):
                rejected += 1
                continue
            for key in ("excerpt", "gene_id", "gene_symbol"):
                value = _text(item.get(key), MAX_EXCERPT if key == "excerpt" else 40)
                if value and (key == "excerpt" or (key == "gene_id" and _GENE.fullmatch(value)) or (key == "gene_symbol" and _SYMBOL.fullmatch(value))):
                    safe[key] = value
            if safe.get("gene_symbol") and safe["target"] == "candidate_ranking":
                safe["location"] = re.sub(r"Gene ranking \d+", "Gene ranking · " + safe["gene_symbol"], safe["location"], count=1)
            for key in ("evidence_ids", "variant_ids", "numeric_fact_ids"):
                ids = _identifiers(item.get(key))
                if ids:
                    safe[key] = ids
            items.append(safe)
            counts[safe["disposition"]] += 1
        for key, disposition in (("unresolved_count", "unverified"), ("corrected_count", "corrected")):
            value = projection.get(key, 0)
            if isinstance(value, int) and 0 <= value <= _MAX_INPUT_ITEMS * 2:
                if disposition == "unverified":
                    hidden_unresolved += max(0, value - counts[disposition])
                else:
                    hidden_corrected += max(0, value - counts[disposition])
        for key in ("rejected_count", "unexamined_count", "omitted_count"):
            value = projection.get(key, 0)
            if isinstance(value, int) and 0 <= value <= 1_000_000:
                if key == "rejected_count":
                    rejected += value
                elif key == "unexamined_count":
                    unexamined += value
                else:
                    omitted += value
    result = _envelope(items, rejected=rejected, unexamined=unexamined)
    result["omitted_count"] += omitted
    result["unresolved_count"] += hidden_unresolved
    result["corrected_count"] += hidden_corrected
    return result


merge_fact_issues = combine_fact_issues


def _plain_statement(value: str) -> str:
    """Normalize presentation markup without reading link targets as assertions."""
    value = html.unescape(value)
    value = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    value = re.sub(r"\{\{cite_fact:[^}]+\}\}", "", value)
    value = re.sub(r"\[([^\]]*)\]\([^\n]*?\)", r"\1", value)
    value = re.sub(r"<[^>]*>", "", value)
    value = value.translate(_SUPERSCRIPT)
    value = re.sub(r"(?:×|x|\*)\s*10\s*\^?\s*([+-]?\d+)", r"e\1", value)
    return re.sub(r"\s+", " ", value.replace("*", "").replace("`", "")).strip().casefold()


def _value_spans(value: str, original: str, *, numeric: bool) -> list[tuple[int, int]]:
    if not numeric or _NONFINITE_LITERAL.fullmatch(original) or _COUNT_LITERAL.fullmatch(original) or _LD_RANGE.fullmatch(original) or (_LD_UNPARSED.fullmatch(original) and not _NUMBER.fullmatch(original)):
        return [(match.start(), match.end()) for match in re.finditer(
            r"(?<![\w])" + re.escape(original) + r"(?![\w])", value)][:256]
    try:
        expected = Decimal(original.rstrip("%"))
    except InvalidOperation:
        return []
    spans = []
    for token in _VALUE_TOKEN.finditer(value):
        if len(spans) >= 256:
            break
        if token.group().endswith("%") != original.endswith("%"):
            continue
        try:
            if Decimal(token.group().rstrip("%")) == expected:
                spans.append(token.span())
        except InvalidOperation:
            continue
    return spans


def _assertion_survives(item: Mapping, visible: str) -> bool:
    """Require the reviewed value in the same local statement, not elsewhere."""
    original = _plain_statement(str(item.get("original_value", "")))
    text = _plain_statement(visible)
    numeric = item.get("metric") not in {"gene_id", "variant_id", "citation"}
    occurrences = _value_spans(text, original, numeric=numeric)
    if not occurrences:
        return False
    if not numeric and item.get("reason_code") != "report_gene_symbol_identifier_conflict":
        # A retained unknown identifier is still unknown when nearby numeric
        # corrections change the surrounding sentence.
        return True
    excerpt = _plain_statement(str(item.get("excerpt", "")))
    if not numeric and item.get("reason_code") == "report_gene_symbol_identifier_conflict":
        pair = re.search(r"\b([a-z][a-z0-9.-]*)\s*\(\s*" + re.escape(original), excerpt)
        return bool(re.search(r"\b" + re.escape(pair[1]) + r"\s*\(\s*" + re.escape(original), text)) if pair else True
    original_spans = _value_spans(excerpt, original, numeric=numeric)
    if not original_spans:
        # Older receipts may lack an excerpt. Only the known output field is
        # searched; retaining an actual unmatched value is the honest result.
        return True
    for left, right in original_spans:
        before = excerpt[:left].split("…")[-1][-60:]
        after = excerpt[right:].split("…")[0][:60]
        check_before, check_after = len(before.strip()) >= 12, len(after.strip()) >= 12
        for start, end in occurrences:
            if ((not check_before or text[:start].endswith(before))
                    and (not check_after or text[end:].startswith(after))):
                return True
    return False


def _rendered_field(markdown: str, field: str) -> str:
    labels = {"summary_markdown": "summary", "limitations_uncertainty.markdown": "limitations & uncertainty"}
    heading = labels.get(field)
    if not heading:
        return ""
    lines = markdown.splitlines()
    collected, active, has_headings = [], False, False
    for line in lines:
        match = re.fullmatch(r"\s*(?:#{1,2}\s+)?\*{0,2}(Summary|Biological Context|Evidence|Limitations & Uncertainty|References)\*{0,2}\s*", line, re.I)
        if match:
            has_headings = True
            if active:
                break
            active = match[1].casefold() == heading
        elif active:
            collected.append(line)
    # A plain reply can have no section headings. Its text is still visible.
    return "\n".join(collected) if active or collected else "" if has_headings else markdown


def visible_report_fact_issues(projection: Mapping, *, visible_output: Any = None,
                              rendered_markdown: str | None = None) -> dict:
    """Project objective facts that remain in the final answer after correction.

    Internal ranking findings and prose-style audits are retained in their
    original receipts, never promoted to report warnings. A removed assertion
    is a neutral local correction when its replacement is present in the same
    report field. This function neither changes the answer nor verifies new facts.
    """
    clean = combine_fact_issues(projection)
    visible = _payload(visible_output)
    markdown = rendered_markdown if isinstance(rendered_markdown, str) and len(rendered_markdown) <= 2_000_000 else None
    can_inspect = bool(visible) or markdown is not None
    items = []
    for item in clean["items"]:
        if item["target"] != "final_report" or item["metric"] not in _OBJECTIVE_METRICS:
            continue
        if not can_inspect:
            items.append(item)
            continue
        field = _visible_field(visible, item["field"], "final_report")
        value = field[0] if field is not None and isinstance(field[0], str) else _rendered_field(markdown or "", item["field"])
        if len(value) > 2_000_000:
            items.append(item)
            continue
        if item["metric"] in _TYPED_SEMANTIC:
            # A factual clarification can retain the original words inside its
            # longer corrected phrase (e.g. 'nearest protein-coding gene').
            # Show its receipt only when the corrected wording is now visible.
            replacement = _plain_statement(item["replacement"])
            if item["disposition"] == "corrected" and replacement and replacement in _plain_statement(value):
                items.append(item)
            continue
        if _assertion_survives(item, value):
            if item["disposition"] == "unverified":
                items.append(item)
            continue
        replacement = _plain_statement(item["replacement"])
        removed = item["replacement"] == REMOVED_VALUE_LABEL
        if not removed and (not replacement or replacement not in _plain_statement(value)):
            continue
        if item["disposition"] == "corrected":
            items.append(item)
            continue
        occurrence = item.get("occurrence", {})
        corrected = _project_finding({**item, **occurrence,
            "reason": "unverified_assertion_removed"}, target="final_report", disposition="corrected",
            visible={}, gene_labels={}, projected=True)
        if corrected is not None:
            for key in ("excerpt", "gene_id", "gene_symbol", "variant_ids", "evidence_ids"):
                if key in item:
                    corrected[key] = item[key]
            items.append(corrected)
    result = _envelope(items, rejected=clean["rejected_count"], unexamined=clean["unexamined_count"])
    result["omitted_count"] += clean["omitted_count"]
    # Do not infer a clean review when a bounded receipt omitted unresolved rows.
    shown_unresolved = sum(item["disposition"] == "unverified" for item in clean["items"])
    result["unresolved_count"] += max(0, clean["unresolved_count"] - shown_unresolved)
    result["scope"] = "final_visible_answer"
    result["visibility_checked"] = can_inspect and isinstance(projection, Mapping) and projection.get("schema_version") == "1"
    return result
