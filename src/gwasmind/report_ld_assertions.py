"""Adapt frozen report facts to the same typed LD review used for ranking."""
from __future__ import annotations

import re
from types import SimpleNamespace

from .ld_assertions import _VARIANT, identity_resolver, record_ld_witnesses
from .models import EvidenceKind
from .native_source_assertions import native_ld_numeric_bindings


def report_ld_numeric_bindings(text, context):
    if not re.search(r"\br\s*(?:²|\^\s*2|2|[- ]squared)\b", text, re.I):
        return ()
    from .staged_reasoning import _final_ld_report_scope, _reported_gene_aliases, _runtime_evidence_rows

    scope = _final_ld_report_scope(context)
    plan = dict(context.result.get("ld_expansion") or {})
    plan["ld_links"] = list(scope.links) if scope.valid else []
    if not scope.valid or not scope.provided or scope.disabled:
        plan["threshold"] = None
    plan["resolved_source_aliases"] = dict(scope.source_aliases)
    canonical = identity_resolver(plan)
    aliases = _reported_gene_aliases(context)
    records = []
    for row in _runtime_evidence_rows(context.result):
        try:
            kind = EvidenceKind(row.get("kind"))
        except ValueError:
            continue
        records.append(SimpleNamespace(evidence_id=row.get("evidence_id"), gene_id=row.get("gene_id"),
            variant_id=row.get("variant_id"), kind=kind, payload=row.get("payload") or {}))
    by_id = {row.evidence_id: row for row in records}
    fact_records = []
    for fact in context.facts:
        subject = str(fact.model_data.get("subject") or fact.model_data.get("gene") or "")
        genes = {gene for alias, gene in aliases.items()
                 if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", subject, re.I)}
        owned = [by_id[identifier] for identifier in fact.evidence_ids if identifier in by_id]
        # These are program-owned display facts, not newly generated model
        # prose. Explicit SNPs displayed by a fact narrow its local witnesses.
        named = {canonical(match.group()) for match in _VARIANT.finditer(fact.markdown)}
        witnesses = set().union(*(record_ld_witnesses(row, canonical) for row in owned)) if owned else set()
        if named:
            witnesses = witnesses & named if witnesses else named
        if not genes:
            genes = {row.gene_id for row in owned if row.gene_id}
        fact_records.append(SimpleNamespace(evidence_id=fact.fact_id,
            gene_id=next(iter(genes)) if len(genes) == 1 else None,
            variant_id=next(iter(witnesses)) if len(witnesses) == 1 else None,
            kind=owned[0].kind if owned and all(row.kind == owned[0].kind for row in owned) else EvidenceKind.FUNCTION,
            payload={"context": {"locus_anchor_witnesses": [{"variant_id": identifier} for identifier in sorted(witnesses)]}}))
    query = SimpleNamespace(ld_expansion=plan,
        variants=tuple(SimpleNamespace(variant_id=row["variant_id"]) for row in context.source_observations if row.get("variant_id")))
    return native_ld_numeric_bindings(text, query=query, records=tuple(fact_records),
        aliases={fact.fact_id: fact.fact_id for fact in context.facts}, gene_aliases=aliases,
        all_records=tuple(records + fact_records))
