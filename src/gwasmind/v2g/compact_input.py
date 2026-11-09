"""Versioned, reversible scientific cards for the native V2G model request.

The core projection removes operational metadata, never scientific rows. Named
tables and a shared dictionary encode that core without loss. Original cards,
provider responses and the full evidence ledger remain outside this projection.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..audit import canonical_json, sha256_json
from ..configuration import parameter

SCHEMA = "gwasmind_v2g_scientific_core_v1"
TABLE = "gwasmind_v2g_table_v1"
_MISSING = {"$missing": True}
_REFERENCE = "$ref"
_RESERVED = frozenset({_REFERENCE, "$missing"})
_MOLECULAR = frozenset({"qtl", "coloc", "e2g"})
_AUDIT_FIELDS = frozenset({
    "context_assessment", "request_pointer", "response_pointer", "raw_response",
    "raw_payload", "audit", "audit_trail", "retrieval_trace", "page_receipts",
    "physical_annotation_sha256",
})
_QTL_INTERNAL_FIELDS = frozenset({
    "clpp", "clpp_basis", "shared_variants", "shared_variant_count",
    "max_shared_r2", "max_qtl_pip_at_strong_anchor", "strong_anchor_shared",
    "matched_tissue_nonoverlapping",
})
_COVERAGE_FIELDS = frozenset({
    "anchor_kind", "complete", "frozen_molqtl_count", "queried_molqtl_count",
    "completed_molqtl_count", "failed_molqtl_count", "invalid_molqtl_identity_count",
    "raw_rows", "returned_rows", "retained_rows", "h4_threshold", "scope",
    "status", "excluded_by_reason",
})


def _clean(value: Any, *, category: str | None = None) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key in _RESERVED:
                raise ValueError("Scientific data contains a reserved compact-input marker")
            if str(key).startswith("_") or key in _AUDIT_FIELDS:
                continue
            if category == "qtl" and key in _QTL_INTERNAL_FIELDS:
                continue
            if category == "coloc" and key == "clpp":
                continue
            result[key] = _clean(item, category=category)
        return result
    if isinstance(value, (list, tuple)):
        return [_clean(item, category=category) for item in value]
    # Reject NaN/Infinity and non-JSON values before they can become ambiguous
    # nulls or invalid evidence during the provider's JSON serialization.
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def _context_core(context: Mapping[str, Any]) -> dict[str, Any]:
    result = _clean(context)
    mapping = result.pop("biosample_mapping", None)
    if isinstance(mapping, Mapping):
        result["selected_biosamples"] = [
            {key: item[key] for key in ("axis", "input", "status", "matches") if key in item}
            for item in mapping.get("concepts", ()) if isinstance(item, Mapping)
        ]
    return result


def scientific_core_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicitly defined scientific core, without mutating input.

    Unknown scientific fields are retained. Only named operational fields and
    deprecated molecular context/overlap scores are removed. This keeps future
    provider measurements from being silently lost by a narrow allowlist.
    """
    result = _clean(payload)
    comparison = result["gene_comparison"]
    if comparison.get("format") != "gwasmind_v2g_gene_cards_v1":
        raise ValueError("Compact V2G encoding requires native gene cards")
    if "encoding" in comparison:
        raise ValueError("A compact V2G payload must be decoded before projection")
    context = result.get("locus", {}).get("biological_context")
    if isinstance(context, Mapping):
        result["locus"]["biological_context"] = _context_core(context)
    coverage = {}
    cards = []
    for source_card in comparison["comparison_cards"]:
        card = deepcopy(source_card)
        # Legacy tiers depend on shared-PIP/CLPP heuristics and must not become
        # an implicit programmatic preference in the new model-only comparison.
        card.pop("tier", None)
        card.pop("tier_basis", None)
        for category, original in card.get("evidence", {}).items():
            section = _clean(original, category=category)
            receipt = section.pop("coverage", None)
            if receipt is not None:
                brief = {key: value for key, value in receipt.items() if key in _COVERAGE_FIELDS}
                coverage.setdefault(category, {})[canonical_json(brief)] = brief
            section.pop("coverage_by_disease", None)
            if category == "association" and section.get("records"):
                # These root fields repeat the target-phenotype row verbatim.
                for key in ("overall", "disease", "datasources", "datatypes"):
                    section.pop(key, None)
            card["evidence"][category] = section
        cards.append(card)
    comparison["comparison_cards"] = cards
    scalar_semantics = comparison.get("ld_context", {}).get("card_scalar_semantics", {})
    scalar_semantics.pop("max_shared_r2", None)
    if coverage:
        comparison["retrieval_summary"] = {
            category: list(variants.values()) for category, variants in coverage.items()
        }
    return result


def _same(left: Any, right: Any) -> bool:
    return type(left) is type(right) and canonical_json(left) == canonical_json(right)


def _table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = sorted({key for row in rows for key in row}, key=lambda key: (key != "evidence_id", key))
    constants = {key: rows[0][key] for key in columns if key in rows[0]
                 and all(key in row and _same(row[key], rows[0][key]) for row in rows)}
    columns = [key for key in columns if key not in constants]
    return {"format": TABLE, "columns": columns, "constant_fields": constants,
            "rows": [[row[key] if key in row else deepcopy(_MISSING) for key in columns] for row in rows]}


def encode_v2g_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Encode all core rows; a verified round trip is required before returning."""
    core = scientific_core_payload(payload)
    counts: Counter[str] = Counter()

    def candidate(value, path=()):
        if isinstance(value, str) and len(value) >= 24:
            return canonical_json(value)
        # An interval is often returned through the same group of LD-linked
        # variants in hundreds of biosamples. Share the complete witness or
        # resource list, preserving every identity, measurement and ordering.
        # Evidence rows and citation lists themselves remain individual rows.
        if (isinstance(value, list) and path
                and path[-1] in {"witnesses", "locus_anchor_witnesses", "resource_scores"}
                and 0 < len(value) <= 64):
            key = canonical_json(value)
            if 96 <= len(key) <= 32768:
                return key
        if isinstance(value, Mapping):
            # Only small leaf metadata is interned. This bounds reference depth
            # and avoids serializing every large evidence subtree while merely
            # counting repeated definitions on very large loci.
            if (len(value) > 16 or "evidence_id" in value or "evidence_ids" in value
                    or any(isinstance(child, (Mapping, list, tuple)) for child in value.values())):
                return None
            key = canonical_json(value)
            if len(key) >= 96 and not value.get("format"):
                return key
        return None

    def count(value, path=()):
        key = candidate(value, path)
        if key:
            counts[key] += 1
        if isinstance(value, Mapping):
            for name, child in value.items():
                count(child, (*path, name))
        elif isinstance(value, list):
            for child in value:
                count(child, (*path, "[]"))

    count(core["gene_comparison"]["comparison_cards"])
    reusable = {key for key, number in counts.items() if number > 1
                and (len(key) - 20) * (number - 1) > 40}
    if not parameter("v2g.card.shared_references"):
        reusable = set()
    definitions: dict[str, Any] = {}
    references: dict[str, str] = {}

    def encode(value, *, path=(), intern=True):
        key = candidate(value, path) if intern else None
        if key in reusable:
            if key not in references:
                identifier = f"D{len(references) + 1}"
                references[key] = identifier
                definitions[identifier] = encode(value, path=path, intern=False)
            return {_REFERENCE: references[key]}
        if isinstance(value, Mapping):
            return {name: encode(child, path=(*path, name)) for name, child in value.items()}
        if isinstance(value, list):
            children = [encode(child, path=(*path, "[]")) for child in value]
            if (len(children) >= 2 and all(isinstance(child, dict) and _REFERENCE not in child for child in children)
                    and path not in {("comparison_cards",)}):
                packed = _table(children)
                if len(canonical_json(packed)) < len(canonical_json(children)):
                    return packed
            return children
        return value

    result = deepcopy(core)
    comparison = result["gene_comparison"]
    # Keep card identities and the existing LD pair table directly readable;
    # explicit shared references encode only scientific card content.
    comparison["comparison_cards"] = [
        {**{key: deepcopy(value) for key, value in card.items() if key != "evidence"},
         "evidence": encode(card.get("evidence", {}), path=("evidence",))}
        for card in core["gene_comparison"]["comparison_cards"]
    ]
    comparison["encoding"] = SCHEMA
    comparison["shared_definitions"] = definitions
    decoded = decode_v2g_payload(result)
    if canonical_json(decoded) != canonical_json(core):
        raise ValueError("Compact V2G scientific core failed exact round-trip verification")
    original_chars, core_chars, encoded_chars = map(lambda item: len(canonical_json(item)), (payload, core, result))
    receipt = {
        "encoding": SCHEMA, "source_payload_chars": original_chars,
        "scientific_core_chars": core_chars, "encoded_payload_chars": encoded_chars,
        "scientific_core_sha256": sha256_json(core), "core_round_trip_verified": True,
        "shared_definition_count": len(definitions),
        "candidate_count": len(comparison["comparison_cards"]),
        "scientific_rows_by_category": _row_counts(core), "scientific_rows_omitted_by_encoding": 0,
        "core_citation_bindings": _citation_owners(core),
    }
    return result, receipt


def decode_v2g_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the new schema or return an untouched copy of a legacy payload."""
    result = deepcopy(payload)
    comparison = result.get("gene_comparison") or {}
    encoding = comparison.get("encoding")
    if encoding is None:
        return result
    if encoding != SCHEMA:
        raise ValueError(f"Unsupported V2G scientific input encoding: {encoding}")
    definitions = comparison.get("shared_definitions")
    if not isinstance(definitions, Mapping):
        raise ValueError("Compact V2G input has no shared definitions")
    cache = {}
    used = set()

    def decode(value, *, stack=()):
        if isinstance(value, Mapping):
            if _REFERENCE in value:
                identifier = value[_REFERENCE]
                if set(value) != {_REFERENCE} or not isinstance(identifier, str) or identifier not in definitions:
                    raise ValueError("Compact V2G input has an invalid shared reference")
                if identifier in stack:
                    raise ValueError("Compact V2G input has a cyclic shared reference")
                used.add(identifier)
                if identifier not in cache:
                    cache[identifier] = decode(definitions[identifier], stack=(*stack, identifier))
                return deepcopy(cache[identifier])
            if value.get("format") == TABLE:
                columns = value.get("columns")
                constants = value.get("constant_fields")
                rows = value.get("rows")
                if (set(value) != {"format", "columns", "constant_fields", "rows"}
                        or not isinstance(columns, list) or not all(isinstance(key, str) for key in columns)
                        or len(columns) != len(set(columns)) or not isinstance(constants, Mapping)
                        or set(columns) & set(constants) or not isinstance(rows, list)):
                    raise ValueError("Compact V2G input has an invalid scientific table")
                result_rows = []
                for row in rows:
                    if not isinstance(row, list) or len(row) != len(columns):
                        raise ValueError("Compact V2G table has a misaligned row")
                    restored = decode(constants, stack=stack)
                    for name, item in zip(columns, row, strict=True):
                        if item != _MISSING:
                            restored[name] = decode(item, stack=stack)
                    result_rows.append(restored)
                return result_rows
            if "$missing" in value:
                raise ValueError("Compact V2G input has a missing marker outside a table cell")
            return {key: decode(item, stack=stack) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item, stack=stack) for item in value]
        return value

    comparison["comparison_cards"] = decode(comparison["comparison_cards"])
    if used != set(definitions):
        raise ValueError("Compact V2G input contains unused shared definitions")
    comparison.pop("encoding")
    comparison.pop("shared_definitions")
    return result


def _row_counts(core: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for card in core["gene_comparison"]["comparison_cards"]:
        for category, section in card.get("evidence", {}).items():
            counts[category] += len(section.get("records", ()))
            if category == "expression":
                counts[category] += len(section.get("context_matched", ()))
    return dict(sorted(counts.items()))


def _citation_owners(core: Mapping[str, Any]) -> dict[str, list[str]]:
    owners: dict[str, set[str]] = {}

    def collect(value, gene):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "evidence_id" and isinstance(child, str):
                    owners.setdefault(child, set()).add(gene)
                elif key == "evidence_ids" and isinstance(child, list):
                    for identifier in child:
                        owners.setdefault(identifier, set()).add(gene)
                else:
                    collect(child, gene)
        elif isinstance(value, list):
            for child in value:
                collect(child, gene)

    for card in core["gene_comparison"]["comparison_cards"]:
        collect(card.get("evidence", {}), card["gene_id"])
    return {key: sorted(values) for key, values in sorted(owners.items())}


def validate_card_source_bindings(cards, records) -> dict[str, Any]:
    """Bind native card citations to typed inference records before aliasing.

    This guard catches wrong-gene or unknown original references before the
    alias projection could discard them or make their identity less obvious.
    Records without citations remain uncitable background, as in older cards.
    """
    owners = {}
    for record in records:
        if record.evidence_id in owners and owners[record.evidence_id] != record.gene_id:
            raise ValueError("Native V2G inference ledger has conflicting evidence owners")
        owners[record.evidence_id] = record.gene_id
    checked = set()

    def reference(identifier, gene):
        if not isinstance(identifier, str) or identifier not in owners:
            raise ValueError("Native V2G card cites an unknown record outside its frozen inference ledger")
        if owners[identifier] != gene:
            raise ValueError("Native V2G card citation belongs to a different gene in the frozen inference ledger")
        checked.add(identifier)

    def walk(value, gene):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).startswith("_"):
                    continue
                if key == "evidence_id":
                    reference(child, gene)
                elif key == "evidence_ids":
                    if not isinstance(child, (list, tuple)):
                        raise ValueError("Native V2G card has invalid evidence references")
                    for identifier in child:
                        reference(identifier, gene)
                else:
                    walk(child, gene)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child, gene)

    for card in cards:
        walk(card.get("evidence", {}), card["gene_id"])
    return {"typed_source_binding_verified": True, "checked_original_evidence_ids": len(checked)}


def validate_v2g_payload(payload: Mapping[str, Any], original: Mapping[str, Any], *,
                         alias_mapping: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Prove post-presentation tables, values and citation owners are unchanged."""
    actual, expected = decode_v2g_payload(payload), decode_v2g_payload(original)
    mapping = alias_mapping or {}

    def rewrite(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, Mapping):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    expected_cards = {card["gene_id"]: rewrite(card) for card in expected["gene_comparison"]["comparison_cards"]}
    actual_cards = {card["gene_id"]: card for card in actual["gene_comparison"]["comparison_cards"]}
    if (len(actual_cards) != len(actual["gene_comparison"]["comparison_cards"])
            or canonical_json(actual_cards) != canonical_json(expected_cards)):
        raise ValueError("Compact V2G scientific cards changed during presentation or alias rebinding")
    expected_owners = {mapping.get(key, key): value for key, value in _citation_owners(expected).items()}
    if _citation_owners(actual) != expected_owners:
        raise ValueError("Compact V2G citation ownership changed during presentation")
    from ..ranking_input import clean_and_order_model_payload

    presentation = {"candidate_order": actual["locus"]["candidate_genes"],
                    "algorithm": actual["locus"].get("input_order_policy", "validation-only")}
    normalized_actual = clean_and_order_model_payload(actual, presentation)
    normalized_expected = clean_and_order_model_payload(rewrite(expected), presentation)
    if canonical_json(normalized_actual) != canonical_json(normalized_expected):
        raise ValueError("Compact V2G global scientific context changed during presentation")
    return {"encoding": SCHEMA, "core_round_trip_verified": True, "citation_binding_verified": True,
            "global_context_verified": True,
            "candidate_count": len(actual_cards), "scientific_rows_by_category": _row_counts(actual),
            "scientific_rows_omitted_by_encoding": 0}
