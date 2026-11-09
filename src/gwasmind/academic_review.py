"""Task-scoped scientific warnings, separate from execution and output failures.

The original check identifiers remain stable audit aliases.  This module groups
observed limitations for display; it neither changes evidence nor invents a
coverage denominator, an independent cohort, or a publication.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .gate_rules import load_gate_rules


WARNING_CATALOG = {
    "W1": ("Task-critical evidence is missing", ("answerability",)),
    "W2": ("Source GWAS significance is not established", ("source_gwas",)),
    "W3": ("Biological context is not matched", ("biological_context",)),
    "W4": ("Declared retrieval scope is incomplete", ("evidence_scope",)),
    "W5": ("Evidence independence is not established", ("independent_support",)),
    "W6": ("Comparable evidence conflicts", ("interpretation",)),
    "W7": ("A selected source SNP identity could not be resolved", ("variant_scope", "input")),
    "W8": ("Complex genomic region", ("locus_interpretation",)),
    "W9": ("Original-study provenance is incomplete", ("provenance",)),
}

# Each Gate 1 rule whose configured code is a W code (W1.1, W7, ...) belongs to
# that warning family. W7 is limited to the identities the user selected:
# LD-expanded proxy lookup, proxy identity, and optional proxy coverage carry
# their own non-W codes and remain method notes.
CHECK_WARNING_CODES = MappingProxyType({
    check_id: rule.code.split(".")[0]
    for check_id, rule in load_gate_rules().rules.items()
    if rule.gate == "gate1" and re.fullmatch(r"W\d+(?:\.\d+)?", rule.code)
})


def scientific_warnings(checks: Sequence[Any], *, ld_only_discovery_gap: bool = False) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for check in checks:
        if str(check.outcome) not in {"warn", "fail"}:
            continue
        code = CHECK_WARNING_CODES.get(check.check_id)
        if code is None:
            continue  # Identity/execution failures and Gate 2 are not SSR warnings.
        if ld_only_discovery_gap and check.check_id == "candidate_discovery_coverage":
            continue  # The LD assessment already records this supplementary scope.
        label, domains = WARNING_CATALOG[code]
        item = grouped.setdefault(code, {"code": code, "label": label,
            "reason_codes": [], "facts": [], "check_ids": [],
            "impact_domains": list(domains), "refs": []})
        item["check_ids"].append(check.check_id)
        item["reason_codes"].extend(getattr(check, "reason_codes", ()) or (check.check_id,))
        if check.fact not in item["facts"]:
            item["facts"].append(check.fact)
        _merge_review_details(item, check)
    return _bounded_items([grouped[code] for code in WARNING_CATALOG if code in grouped])


def _merge_review_details(item: dict[str, Any], check: Any) -> None:
    """Project bounded, program-bound facts for live and saved review cards."""
    rule = load_gate_rules().rules.get(check.check_id)
    if rule is not None and rule.code not in item.setdefault("rule_codes", []):
        item["rule_codes"].append(rule.code)
    item["reason_codes"] = list(dict.fromkeys(item.get("reason_codes", ())))
    impact = getattr(check, "impact", "")
    if impact:
        existing = item.get("impact", "")
        if impact not in existing:
            item["impact"] = " ".join(filter(None, (existing, impact)))
    affected = item.setdefault("affected", {})
    for key, values in getattr(check, "affected", {}).items():
        if key not in {"variant_ids", "gene_ids", "study_ids", "modules"}:
            continue
        affected[key] = list(dict.fromkeys([*affected.get(key, ()), *values]))
    refs = item.setdefault("refs", [])
    for value in getattr(check, "refs", ()):
        ref = {"type": str(value.get("type", "evidence")), "id": str(value.get("id", ""))}
        if ref["id"] and ref not in refs and len(refs) < 32:
            refs.append(ref)


def scientific_method_notes(checks: Sequence[Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for check in checks:
        state = str(check.outcome)
        code = CHECK_WARNING_CODES.get(check.check_id)
        if code is None or state not in {"note", "not_assessed"}:
            continue
        item = grouped.setdefault(code, {"code": code + "_METHOD", "label": WARNING_CATALOG[code][0],
            "facts": [], "reason_codes": [], "check_ids": [], "refs": [], "assessment_state": state})
        if state == "not_assessed":
            item["assessment_state"] = state
        item["check_ids"].append(check.check_id)
        item["reason_codes"].extend(getattr(check, "reason_codes", ()) or (check.check_id,))
        if check.fact not in item["facts"]:
            item["facts"].append(check.fact)
        _merge_review_details(item, check)
    return _bounded_items([grouped[code] for code in WARNING_CATALOG if code in grouped])


def _bounded_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in items:
        affected = item.get("affected", {})
        item["affected_count"] = {key: len(values) for key, values in affected.items()}
        item["affected"] = {key: sorted(values)[:32] for key, values in affected.items()}
    return items


def operation_issue(code: str, *, category: str, scope: str, message: str,
                    blocking: bool, reasons: Sequence[str] = ()) -> dict[str, Any]:
    return {"code": code, "category": category, "scope": scope, "message": message,
            "blocking": blocking, "reason_codes": list(dict.fromkeys(reasons))}


def normalize_operation_issues(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    defaults = {
        "S1": ("input_invalid", "input", True, "No source variant could be resolved to a usable identity."),
        "S2": ("service_unavailable", "retrieval", True, "A required data service remained unavailable after bounded recovery."),
        "SYSTEM_ERROR": ("system_error", "retrieval", True, "The evidence context failed an internal integrity check."),
        "source_variants_excluded": ("input_excluded", "input", False, "Some source variants were excluded after reference identity checks."),
        "IDENTITY_UNRESOLVED": ("service_unavailable", "retrieval", True, "Source identity verification could not finish because reference retrieval was incomplete."),
    }
    output = []
    for row in value:
        if not isinstance(row, Mapping) or row.get("code") not in defaults:
            continue
        code = row["code"]
        category, scope, blocking, message = defaults[code]
        reasons = row.get("reason_codes", ())
        reasons = tuple(reason[:200] for reason in reasons if isinstance(reason, str)) if isinstance(reasons, (list, tuple)) else ()
        supplied = row.get("message")
        output.append(operation_issue(code, category=category, scope=scope, blocking=blocking,
            message=supplied[:2000] if isinstance(supplied, str) and supplied.strip() else message,
            reasons=reasons or (code,)))
    return tuple(output)


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def original_publication_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Read only this record's explicit source provenance, not unrelated papers.

    DOI is a valid alternative to PMID.  A generic literature evidence ID or a
    database home URL does not establish the original study's publication.
    """
    payload = _mapping(record.get("payload"))
    upstream = _mapping(payload.get("upstream"))
    raw = _mapping(upstream.get("raw"))
    mappings = [record, payload, _mapping(payload.get("context")),
                _mapping(payload.get("provenance")), upstream, raw,
                _mapping(raw.get("study")), _mapping(raw.get("publication"))]
    found = set()
    for item in mappings:
        for key, value in item.items():
            name = str(key).casefold().replace("_", "")
            values = value if isinstance(value, (list, tuple)) else (value,)
            for raw_value in values:
                if not isinstance(raw_value, (str, int)) or isinstance(raw_value, bool):
                    continue
                text = str(raw_value).strip()
                if name in {"pmid", "pmids", "pubmedid", "pubmedids"}:
                    match = re.fullmatch(r"(?:PMID\s*:\s*)?([1-9]\d*)", text, re.I)
                    if match:
                        found.add("pmid:" + match.group(1))
                elif name in {"doi", "dois", "publicationdoi"}:
                    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", text, flags=re.I)
                    if re.fullmatch(r"10\.\d{4,9}/\S+", text, re.I):
                        found.add("doi:" + text.casefold())
    return tuple(sorted(found))


def publication_count(records: Sequence[Mapping[str, Any]]) -> int:
    return len({identifier for record in records for identifier in original_publication_ids(record)})


def unreferenced_original_study_count(records: Sequence[Mapping[str, Any]]) -> int:
    from .evidence_sources import record_source_identifier
    source_links: dict[str, set[str]] = {}
    unidentified = 0
    for record in records:
        source = record_source_identifier(record)
        identifiers = original_publication_ids(record)
        if source is not None:
            source_links.setdefault(source, set()).update(identifiers)
        elif str(record.get("kind")) in {"qtl", "colocalization", "gwas", "study", "literature"} and not identifiers:
            unidentified += 1
    return sum(not links for links in source_links.values()) + unidentified


def comparable_conflict_count(records: Sequence[Mapping[str, Any]]) -> int:
    """Opposition is a conflict only within an observed comparable scope.

    Missing scope metadata is not evidence that two different studies measured
    the same thing.  Effect directions are never inferred from a support flag.
    """
    groups: dict[tuple[str, ...], set[bool]] = {}
    for record in records:
        supports = record.get("supports")
        gene = record.get("gene_id")
        if not isinstance(supports, bool) or not isinstance(gene, str) or not gene.strip():
            continue
        payload = _mapping(record.get("payload"))
        context = _mapping(payload.get("context"))
        raw = _mapping(_mapping(payload.get("upstream")).get("raw"))
        kind = str(record.get("kind") or "")
        trait = next((str(value).strip().casefold() for value in (
            context.get("trait_id"), context.get("trait"), raw.get("traitFromSource"),
            raw.get("diseaseId"), payload.get("trait")) if isinstance(value, str) and value.strip()), "")
        tissue = next((str(value).strip().casefold() for value in (
            record.get("tissue"), context.get("tissue"), context.get("biosample"))
            if isinstance(value, str) and value.strip()), "")
        variant = str(record.get("variant_id") or "")
        if not kind or not trait:
            continue
        if kind in {"qtl", "colocalization", "regulatory", "variant_annotation"} and (not variant or not tissue):
            continue
        # An explicit comparison key can narrow the scope further, never broaden it.
        comparison = str(context.get("comparison_id") or "")
        groups.setdefault((gene, kind, variant, trait, tissue, comparison), set()).add(supports)
    return sum(values == {True, False} for values in groups.values())
