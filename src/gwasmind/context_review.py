"""Recompute scientific context matching from evidence and a verified plan."""
from __future__ import annotations

from collections.abc import Mapping

from .audit import sha256_json
from .molecular_context import freeze_molecular_context, molecular_context_match


def prepare_context_review(value):
    """Return a modern mapping plan, or None for the unchanged legacy path."""
    if not isinstance(value, Mapping) or value.get("biosample_mapping") is None:
        return None
    mapping = value.get("biosample_mapping")
    mapping = mapping if isinstance(mapping, Mapping) else {}
    catalog = mapping.get("catalog")
    catalog = catalog if isinstance(catalog, Mapping) else {}
    identity = {"method": "recomputed_ot_biosample_identity_and_hierarchy",
                "context_sha256": value.get("context_sha256"),
                "mapping_sha256": sha256_json(mapping), "catalog_sha256": catalog.get("catalog_sha256"),
                "catalog_source_sha256": catalog.get("source_sha256"),
                "derived_record_tags_are_authoritative": False}
    try:
        frozen = freeze_molecular_context(value)
    except (TypeError, ValueError):
        return {"frozen": None, "identity": {**identity, "validation_status": "invalid_context_mapping"}}
    return {"frozen": frozen, "identity": {**identity, "context_sha256": frozen["context_sha256"],
            "frozen_context_sha256": sha256_json(frozen), "validation_status": "verified"}}


def review_record_context(record, plan):
    """Unwrap normalized EvidenceRecord payloads without trusting derived tags."""
    if plan["frozen"] is None:
        return {"matched": False, "context_class": "context_unreported", "mapping_status": "invalid_context_mapping",
                "observed_biosamples": [], "match_basis": []}
    payload = record.get("payload")
    payload = payload if isinstance(payload, Mapping) else record
    upstream = payload.get("upstream")
    raw = upstream.get("raw") if isinstance(upstream, Mapping) else payload
    raw = raw if isinstance(raw, Mapping) else {}
    context = payload.get("context")
    context = dict(context) if isinstance(context, Mapping) else {}
    # The normalized top-level tissue is an observed provider field. It is not
    # the requested relevant_tissues list or an earlier derived match verdict.
    tissue = record.get("tissue")
    if isinstance(tissue, str) and tissue.strip() and not context.get("biosample"):
        context["biosample"] = tissue
    return molecular_context_match({"upstream": {"raw": raw}, "context": context}, plan["frozen"])
