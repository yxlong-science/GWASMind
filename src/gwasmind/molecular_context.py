"""Frozen molecular discovery context; no inferred biology or network access."""
from __future__ import annotations

import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass

from .audit import sha256_json
from .expression_policy import expression_context, matches_selector

MOLECULAR_POLICY_VERSION = "context-local-molqtl-v1"
_CELL = re.compile(r"cells?\b|lymphocyte|monocyte|macrophage|eosinophil|neutrophil", re.I)


def _map(value):
    return value if isinstance(value, Mapping) else {}


def freeze_molecular_context(value):
    """Use caller-declared tissue/cell axes; preserve undefined context explicitly."""
    if not isinstance(value, Mapping):
        raise ValueError("molecular context must be an explicit object")
    for key in ("organs", "tissues", "cell_types", "relevant_tissues"):
        terms = value.get(key, ())
        if (not isinstance(terms, (list, tuple))
                or any(not isinstance(term, str) or not term.strip() for term in terms)):
            raise ValueError(f"molecular context {key} must be an array of nonempty terms")
    selectors = [{"tissue": term} for term in (*value.get("organs", ()), *value.get("tissues", ()))]
    # The shared selector constructor retains organ-qualified cell restrictions.
    selectors.extend(expression_context(value.get("cell_types", ()))["selectors"])
    if not selectors and value.get("relevant_tissues") and value.get("defined") is not False:
        selectors = expression_context(value["relevant_tissues"])["selectors"]
    selected = expression_context((), selectors=selectors)["selectors"]
    if value.get("defined") is False:
        selected = []
    result = {"version": MOLECULAR_POLICY_VERSION, "selectors": selected,
              "defined": bool(selected), "source": value.get("source", "unresolved"),
              "ruleset_version": value.get("ruleset_version"),
              "context_sha256": value.get("context_sha256") or sha256_json(dict(value))}
    if value.get("biosample_mapping") is not None:
        from copy import deepcopy

        from .biosample_catalog import map_context_concepts
        expected = map_context_concepts(value)
        if value["biosample_mapping"] != expected:
            raise ValueError("Frozen context mapping differs from its declared concepts")
        result.update(version="context-local-molqtl-v2", biosample_mapping=deepcopy(expected))
    return result


def molecular_context_match(record, context):
    """Read source biosample fields, never requested context or counterpart traits."""
    if not context.get("defined") or not context.get("selectors"):
        return {"status": "context_undefined", "matched": False, "observed": [],
                "context_class": "context_unreported", "mapping_status": "context_undefined",
                "observed_biosamples": [], "match_basis": [], "context_source": context.get("source")}
    if context.get("biosample_mapping") is not None:
        from .biosample_catalog import match_catalog_context
        return {**match_catalog_context(record, context["biosample_mapping"]),
                "context_source": context.get("source")}
    raw = _map(_map(record.get("upstream")).get("raw"))
    observed = _map(record.get("context"))
    tissue, cell, samples = [], [], []
    for source in (raw, observed):
        for key in ("tissue", "tissue_name", "tissueName"):
            if isinstance(source.get(key), str) and source[key].strip(): tissue.append(source[key].strip())
        for key in ("cell_type", "cellType", "cell_type_name"):
            if isinstance(source.get(key), str) and source[key].strip(): cell.append(source[key].strip())
        for key in ("biosample", "biosampleName", "biosample_name", "biosampleId", "biosample_id"):
            value = source.get(key)
            if isinstance(value, str) and value.strip(): samples.append(value.strip())
    for label in samples:
        (cell if _CELL.search(label) else tissue).append(label)
    # Explicit tissue/cell metadata may form a pair. A cell-only label cannot
    # invent the missing organ required by an organ-qualified selector.
    sample = ("", "", "; ".join(sorted(set(tissue))), "; ".join(sorted(set(cell))))
    labels = sorted(set((*tissue, *cell)))
    matched = any(matches_selector(selector, sample) for selector in context["selectors"])
    return {"status": "matched" if matched else "not_matched" if labels else "biosample_unavailable",
            "matched": matched, "observed": labels,
            "context_class": "matched" if matched else "other_reported_context" if labels else "context_unreported",
            "mapping_status": "legacy_label_only", "observed_biosamples": [],
            "match_basis": ["legacy_phrase_match"] if matched else [], "context_source": context.get("source")}


def molecular_anchor_verified(record):
    """A credible-set member must name the actual analysis variant explicitly."""
    subject = _map(record.get("subject")).get("id")
    raw = _map(_map(record.get("upstream")).get("raw"))
    context = _map(record.get("context"))
    member = _map(raw.get("matchedLocusMember"))
    ids = [v for v in (member.get("variantId"), context.get("qtl_member_variant_id")) if v is not None]
    if ids:
        return all(v == subject for v in ids) and raw.get("queryVariantId", subject) == subject
    if raw.get("queryVariantRole") == "credible_set_member":
        return False
    return bool(subject and raw.get("variantId") == subject and raw.get("queryVariantId", subject) == subject)


@dataclass(frozen=True, slots=True)
class MolecularRetrievalScope:
    """Internal task-local contract; semantic tool names remain unchanged."""
    phase: str
    context: Mapping
    candidate_genes: tuple[str, ...] = ()
    study_locus_ids: tuple[str, ...] = ()
    target_trait_ids: tuple[str, ...] = ()
    target_trait_labels: tuple[str, ...] = ()
    qualified_pairs: tuple[tuple[str, str, str], ...] = ()
    supplementary_pairs: tuple[tuple[str, str, str], ...] = ()

    def to_dict(self):
        return {"version": MOLECULAR_POLICY_VERSION, "phase": self.phase,
                "context": dict(self.context), "candidate_genes": list(self.candidate_genes),
                "study_locus_ids": list(self.study_locus_ids), "target_trait_ids": list(self.target_trait_ids),
                "target_trait_labels": list(self.target_trait_labels),
                "qualified_pairs": [list(row) for row in self.qualified_pairs],
                "supplementary_pairs": [list(row) for row in self.supplementary_pairs]}


MOLECULAR_RETRIEVAL_SCOPE = ContextVar("molecular_retrieval_scope", default=None)
