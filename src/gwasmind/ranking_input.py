"""Immutable, replayable model-visible ranking input and presentation order."""
from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass

from .audit import canonical_json, sha256_json
from .configuration import parameter

INPUT_SCHEMA = "ranking-input-v3"
ORDER_ALGORITHM = "sha256-seeded-candidate-order-v1"
_AUDIT_ONLY_FIELDS = frozenset({
    "candidate_scope", "direct_evidence_gene_ids", "decision_scope", "candidate_audit",
    "candidate_discovery", "shortlist_reason", "shortlist_policy", "candidate_admission",
    "molecular_selection", "selection_receipt_sha256", "supplementary_selection", "rank_scope",
    "l2g_score", "l2gScore", "locus_to_gene_score", "l2gFeatures",
})


def presentation_fingerprint(query):
    """Bind presentation to scientific input, excluding run IDs and retrieval clocks."""
    return sha256_json({"trait": query.trait, "trait_ids": sorted(query.trait_ids),
        "variants": sorted((row.variant_id, row.p_value) for row in query.variants),
        "analysis_variants": sorted(row.variant_id for row in query.analysis_variants),
        "genome_build": query.genome_build, "ancestry": query.ancestry,
        "ld_reference": query.ld_reference, "tissues": sorted(query.relevant_tissues)})


def candidate_presentation(candidate_genes, *, seed=None, input_fingerprint=None):
    """Randomize presentation reproducibly; never score scientific evidence."""
    policy = "explicit" if seed is not None else parameter("ranking.presentation_seed_policy")
    genes = sorted(set(candidate_genes))
    if seed is None:
        seed = (secrets.token_hex(16) if policy == "random" else
                sha256_json({"candidate_genes": genes, "input_fingerprint": input_fingerprint}))
    seed = str(seed)
    if not seed or len(seed) > 256:
        raise ValueError("Ranking presentation seed must contain 1 to 256 characters")
    enabled = parameter("ranking.presentation_randomize_candidates")
    if enabled:
        genes.sort(key=lambda gene: (hashlib.sha256((seed + "\0" + gene).encode()).digest(), gene))
    return {"seed": seed, "seed_policy": policy, "algorithm": ORDER_ALGORITHM if enabled else "canonical-identities-v1",
            "candidate_order": genes, "candidate_order_sha256": sha256_json(genes),
            "interpretation": "Presentation order only; randomization does not eliminate all model biases."}


def _field_is_audit_only(key):
    name = str(key)
    lowered = name.casefold()
    return (any(part in _AUDIT_ONLY_FIELDS for part in name.split("."))
            or lowered.endswith(("sha256", "_hash")))


def clean_and_order_model_payload(value, presentation):
    """Remove admission hints and reorder only explicit presentation collections.

    A catalog is scientific data, even when its objects contain ``gene_id``.
    Reordering its entries can silently change positional foreign keys. Never
    infer presentation semantics from the values of an arbitrary collection.
    """
    positions = {gene: index for index, gene in enumerate(presentation["candidate_order"])}

    presentation_lists = {
        ("locus", "candidate_genes"),
        ("gene_comparison", "comparison_cards"),
        ("gene_comparison", "citation_records"),
    }

    def transform(item, path=()):
        if isinstance(item, Mapping):
            if {"columns", "constant_fields", "rows"}.issubset(item):
                keep = [index for index, name in enumerate(item["columns"]) if not _field_is_audit_only(name)]
                columns = [item["columns"][index] for index in keep]
                constants = transform(item["constant_fields"], path + ("constant_fields",))
                rows = [[transform(row[index], path + ("rows",)) for index in keep] for row in item["rows"]]
                if len(path) == 2 and path[0] == "scientific_facts" and "gene_id" in columns:
                    gene_index = columns.index("gene_id")
                    rows.sort(key=lambda row: positions.get(row[gene_index], len(positions)))
                return {**{name: transform(content, path + (name,)) for name, content in item.items()
                           if name not in {"columns", "constant_fields", "rows"} and not _field_is_audit_only(name)},
                        "columns": columns, "constant_fields": constants, "rows": rows}
            return {name: transform(content, path + (name,)) for name, content in item.items() if not _field_is_audit_only(name)}
        if isinstance(item, (tuple, list)):
            result = [transform(row, path + ("[]",)) for row in item]
            if path not in presentation_lists:
                return result
            if result and all(isinstance(row, str) and row in positions for row in result):
                result.sort(key=positions.__getitem__)
            elif result and all(isinstance(row, Mapping) and row.get("gene_id") in positions for row in result):
                result.sort(key=lambda row: positions[row["gene_id"]])
            return result
        return item

    payload = transform(value)
    payload["locus"]["candidate_genes"] = list(presentation["candidate_order"])
    payload["locus"]["input_order_policy"] = presentation["algorithm"]
    payload["input_schema_version"] = INPUT_SCHEMA
    return payload


def rebind_presentation_aliases(payload, aliases):
    """Make citation numbering follow the frozen presentation, not hidden ID order."""
    encountered = []

    def walk(value):
        if isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value in aliases and value not in encountered:
            encountered.append(value)

    walk(payload.get("gene_comparison", payload.get("scientific_facts", payload)))
    # Invisible audit-only aliases are never added to the prompt.
    mapping = {old: f"E{index + 1}" for index, old in enumerate(encountered)}

    def rewrite(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, Mapping):
            return {mapping.get(key, key): rewrite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [rewrite(item) for item in value]
        return value

    visible_aliases = {mapping[key]: aliases[key] for key in encountered}
    return rewrite(payload), visible_aliases, mapping


@dataclass(frozen=True)
class RankingInputSnapshot:
    """Canonical bytes own the freeze; returned dictionaries are fresh copies."""
    encoded_audit: str

    @classmethod
    def from_audit(cls, value):
        if value["model_input_sha256"] != sha256_json(value["model_payload"]):
            raise ValueError("Ranking snapshot payload hash is inconsistent")
        if value["messages_sha256"] != sha256_json(value["messages"]):
            raise ValueError("Ranking snapshot message hash is inconsistent")
        return cls(canonical_json(value))

    def to_audit_payload(self):
        return json.loads(self.encoded_audit)
