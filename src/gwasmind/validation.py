from __future__ import annotations

import re

from .evidence_citations import inline_evidence_citations
from .gate_rules import load_gate_rules
from .llm import SynthesisOutput, _candidate_gene_universe
from .models import EvidenceRecord, LocusQuery
from .ranking_claims import ranking_assertion_issues
from .shortlisting import selection_contract_errors
from .sufficiency import decision_evidence_for_query


def validate_synthesis(
    query: LocusQuery,
    evidence: tuple[EvidenceRecord, ...],
    output: SynthesisOutput,
) -> tuple[str, ...]:
    """Gate 2: check output structure, evidence ownership and verifiable facts."""
    validators = {
        "ranking_schema_evidence": _ranking_schema_evidence_issues,
        "ranking_scientific_assertions": _ranking_scientific_issues,
    }
    errors = []
    for name in load_gate_rules().validator_ids("ranking"):
        errors.extend(validators[name](query, evidence, output))
    return tuple(dict.fromkeys(errors))


def _ranking_scientific_issues(query, evidence, output):
    # Hard facts are numerical bindings plus explicit entity/citation ownership.
    # Free prose about presence, absence, causality and source/proxy interpretation
    # is outside Gate 2's mechanical remit and cannot remove a model ranking.
    return ranking_assertion_issues(query, decision_evidence_for_query(query, evidence), output)


def unverified_synthesis_fields(query, evidence, output, errors):
    """Describe unresolved structured owners without deleting parseable rows."""
    expected = set(_candidate_gene_universe(query, evidence))
    known = {row.evidence_id: row for row in decision_evidence_for_query(query, evidence)}
    visible = set(output.model_visible_evidence_ids) if output.model_visible_evidence_ids is not None else set(known)
    fields = []
    for index, row in enumerate(output.ranked_genes):
        prefix = f"ranked_genes[{index}]"
        if row.gene_id not in expected:
            fields.append({"field": prefix + ".gene_id", "reason": "gene outside the candidate universe", "original_value": row.gene_id})
        unknown = [value for value in row.evidence_ids if value not in known or value not in visible]
        foreign = [value for value in row.evidence_ids if value in known and known[value].gene_id and known[value].gene_id != row.gene_id]
        if unknown or foreign or not row.evidence_ids and not query.benchmark:
            fields.append({"field": prefix + ".evidence_ids", "reason": "unverified evidence citation or owner",
                           "original_value": list(row.evidence_ids), "unverified_ids": unknown + foreign})
    for index, row in enumerate(output.claims):
        if not row.evidence_ids or any(value not in known or value not in visible for value in row.evidence_ids):
            fields.append({"field": f"claims[{index}].evidence_ids", "reason": "unverified evidence citation",
                           "original_value": list(row.evidence_ids)})
    for error in errors:
        field = "output"
        for index, row in enumerate(output.ranked_genes):
            if error.startswith((f"rationale for {row.gene_id}", f"limitation for {row.gene_id}", row.gene_id + " ")):
                field = f"ranked_genes[{index}].rationale" if error.startswith("rationale") else f"ranked_genes[{index}]"
                break
        if match := re.match(r"claim (\d+)", error):
            field = f"claims[{int(match[1]) - 1}].text"
        elif error.startswith("summary"):
            field = "summary"
        if not any(item["field"] == field and item["reason"] == error for item in fields):
            fields.append({"field": field, "reason": error})
    return fields


def _ranking_schema_evidence_issues(
    query: LocusQuery, evidence: tuple[EvidenceRecord, ...], output: SynthesisOutput,
) -> tuple[str, ...]:

    errors: list[str] = []
    # The full retrieval ledger may contain L2G rows used to form the candidate
    # shortlist.  They are intentionally absent from the evidence namespace that
    # can support or cite a ``prioritize_genes`` decision.
    inference_evidence = decision_evidence_for_query(query, evidence)
    evidence_by_id: dict[str, EvidenceRecord] = {}
    duplicate_evidence_ids: set[str] = set()
    for record in inference_evidence:
        if record.evidence_id in evidence_by_id:
            duplicate_evidence_ids.add(record.evidence_id)
            continue
        evidence_by_id[record.evidence_id] = record
    if duplicate_evidence_ids:
        errors.append(
            "evidence ledger contains duplicate evidence_id values: "
            + ", ".join(sorted(duplicate_evidence_ids))
        )
    visible_evidence_ids = (
        set(evidence_by_id)
        if output.model_visible_evidence_ids is None
        else set(output.model_visible_evidence_ids)
    )
    projection_unknown = visible_evidence_ids.difference(evidence_by_id)
    if projection_unknown:
        errors.append(
            "model-visible projection contains unknown evidence: "
            + ", ".join(sorted(projection_unknown))
        )
    # Structured evidence_ids alone do not cover citations placed in prose.
    # Every model draft, repair and retained ranking uses this same boundary.
    full_ledger_ids = {record.evidence_id for record in evidence}
    # Frozen presentation aliases are references to the same evidence rows,
    # not additional evidence. Keep inline markers in the same namespace as
    # structured citations and the numerical ownership reviewer.
    evidence_aliases = {alias: canonical for alias, canonical in (output.model_evidence_aliases or {}).items()
                        if re.fullmatch(r'E[1-9][0-9]*', alias) and canonical in full_ledger_ids}
    prose = [("summary", output.summary)]
    for candidate in output.ranked_genes:
        prose.append((f"{candidate.gene_id} rationale", candidate.rationale))
        prose.extend((f"{candidate.gene_id} limitation", text) for text in candidate.limitations)
    prose.extend((f"claim {index} text", claim.text) for index, claim in enumerate(output.claims, start=1))
    prose.extend((f"candidate decision for {row.gene_id}", row.reason) for row in output.candidate_decisions)
    for location, text in prose:
        cited = {evidence_aliases.get(identifier, identifier) for identifier in
                 inline_evidence_citations(text, full_ledger_ids | visible_evidence_ids | evidence_aliases.keys())}
        unknown = cited.difference(evidence_by_id)
        hidden = cited.intersection(evidence_by_id).difference(visible_evidence_ids)
        if unknown:
            errors.append(f"{location} cites unknown or decision-ineligible evidence: " + ", ".join(sorted(unknown)))
        if hidden:
            errors.append(f"{location} cites evidence omitted from the model projection: " + ", ".join(sorted(hidden)))
    genes = [item.gene_id for item in output.ranked_genes]
    if len(genes) != len(set(genes)):
        errors.append("ranked genes contain duplicates")
    expected = set(_candidate_gene_universe(query, evidence))
    received = set(genes)
    model_selection = bool(output.candidate_decisions) or output.model_shortlisting_required
    if model_selection:
        errors.extend(selection_contract_errors(output, expected, required=True))
        if query.benchmark:
            errors.append("model shortlisting is not enabled for benchmark queries")
    # Gene prioritization always ranks the frozen shortlist in full.  The
    # interface may display only the first configured entries, but truncating
    # the model response would make the displayed top list look like a
    # complete ranking while silently dropping shortlisted genes.  Keep the
    # benchmark wording for compatibility, and use an explicit message for
    # ordinary prioritization.
    if not received.issubset(expected):
        errors.append("output contains a gene outside the candidate universe")
    # The hosted/native synthesizer attaches model_candidate_genes after the
    # frozen input is built; that metadata is the authoritative production
    # marker for the complete-shortlist contract.  Keep direct legacy helper
    # callers compatible when they construct an unbound SynthesisOutput.
    require_complete_shortlist = query.benchmark or output.model_candidate_genes is not None
    if received != expected and require_complete_shortlist and output.model_candidate_genes is None and not model_selection:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        if query.benchmark:
            errors.append("benchmark output must preserve the complete candidate universe")
        else:
            detail = []
            if missing:
                detail.append("missing: " + ", ".join(missing))
            if extra:
                detail.append("unexpected: " + ", ".join(extra))
            errors.append("output must rank every candidate exactly once (" + "; ".join(detail) + ")")
    if output.model_candidate_genes is not None:
        frozen = output.model_candidate_genes
        if not isinstance(frozen, tuple) or any(not isinstance(gene, str) or not gene.strip() for gene in frozen):
            errors.append("model candidate universe metadata must contain gene identifiers")
        else:
            frozen_set = set(frozen)
            if len(frozen) != len(frozen_set):
                errors.append("model candidate universe metadata contains duplicate genes")
            if frozen_set != expected:
                errors.append("model candidate universe does not match the query candidate universe")
            if received != frozen_set and not model_selection:
                missing = sorted(frozen_set - received)
                extra = sorted(received - frozen_set)
                detail = []
                if missing:
                    detail.append("missing: " + ", ".join(missing))
                if extra:
                    detail.append("unexpected: " + ", ".join(extra))
                errors.append("output must rank every frozen candidate exactly once (" + "; ".join(detail) + ")")
    if not query.benchmark and not genes and not model_selection:
        errors.append("exploratory output must rank at least one gene")
    if [item.rank for item in output.ranked_genes] != list(range(1, len(output.ranked_genes) + 1)):
        errors.append("ranks must be consecutive and start at one")

    for item in output.ranked_genes:
        no_available_gene_evidence = output.model_full_pool_ranking_required and not any(
            record.gene_id == item.gene_id and record.evidence_id in visible_evidence_ids for record in evidence)
        if not query.benchmark and not item.evidence_ids and not no_available_gene_evidence:
            errors.append(f"{item.gene_id} has no model-visible evidence citation")
        missing = set(item.evidence_ids).difference(evidence_by_id)
        if missing:
            errors.append(f"{item.gene_id} cites unknown evidence: " + ", ".join(sorted(missing)))
        hidden = set(item.evidence_ids).intersection(evidence_by_id) - visible_evidence_ids
        if hidden:
            errors.append(
                f"{item.gene_id} cites evidence omitted from the model projection: "
                + ", ".join(sorted(hidden))
            )
        linked = [evidence_by_id[value] for value in item.evidence_ids if value in evidence_by_id]
        foreign_genes = {
            str(record.gene_id)
            for record in linked
            if record.gene_id and record.gene_id != item.gene_id
        }
        if foreign_genes:
            errors.append(
                f"{item.gene_id} cites evidence linked to another gene: "
                + ", ".join(sorted(foreign_genes))
            )
        if not query.benchmark and not no_available_gene_evidence and not any(
            record.gene_id == item.gene_id and record.evidence_id in visible_evidence_ids
            for record in linked
        ):
            errors.append(f"{item.gene_id} lacks a model-visible gene-specific evidence citation")
        # Lack of a direct locus link is an SSR scientific limitation, not a
        # malformed ranking. The model may compare eligible contextual evidence;
        # owner, visibility and source/proxy claim checks remain mandatory.

    for decision in output.candidate_decisions:
        aliases = output.model_evidence_aliases or {}
        inline = inline_evidence_citations(decision.reason, full_ledger_ids | visible_evidence_ids | set(aliases))
        inline.update(match.group() for match in re.finditer(r"\bE[1-9][0-9]*\b", decision.reason)
                      if match.group() in aliases)
        cited_ids = set(decision.evidence_ids) | {aliases.get(value, value) for value in inline}
        missing = cited_ids.difference(evidence_by_id)
        hidden = cited_ids.intersection(evidence_by_id) - visible_evidence_ids
        foreign = {evidence_by_id[value].gene_id for value in cited_ids
                   if value in evidence_by_id and evidence_by_id[value].gene_id
                   and evidence_by_id[value].gene_id != decision.gene_id}
        if missing or hidden or foreign:
            errors.append(f"candidate decision for {decision.gene_id} cites unknown, hidden or other-gene evidence")

    for index, claim in enumerate(output.claims, start=1):
        if not claim.evidence_ids:
            errors.append(f"claim {index} has no evidence citation")
        missing = set(claim.evidence_ids).difference(evidence_by_id)
        if missing:
            errors.append(f"claim {index} cites unknown evidence: " + ", ".join(sorted(missing)))
        hidden = set(claim.evidence_ids).intersection(evidence_by_id) - visible_evidence_ids
        if hidden:
            errors.append(
                f"claim {index} cites evidence omitted from the model projection: "
                + ", ".join(sorted(hidden))
            )
    return tuple(dict.fromkeys(errors))


