"""Trace retained locus evidence for unranked candidates without selecting them."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import html
import re

from .audit import sha256_json
from .ld_assertions import identity_resolver

_CHANNEL = r"(?:molQTL|eQTL|sQTL|QTL|E2G|coding|coloc(?:ali[sz]ation)?|regulatory)"
_ABSENCE = re.compile(r"\b(?:has|have|shows?)\s+no\s+"
    r"(?:(?:supplied|retained|retrieved|recorded|overlapping|locus-specific)\s+)*"
    r"(?P<channels>" + _CHANNEL + r"(?:\s*(?:,\s*(?:or\s+|and\s+)?|or\s+|and\s+|/)\s*" + _CHANNEL + r")*)"
    r"\s+(?:links?|evidence|records?|annotations?)(?P<scope>[^.;]*)", re.I)


def _value(record, key, default=None):
    return record.get(key, default) if isinstance(record, Mapping) else getattr(record, key, default)


def locus_rows(records):
    """Read only admitted, explicit locus relationships; background is excluded."""
    from .candidates import evidence_candidate_scope
    for record in records:
        gene, payload = _value(record, 'gene_id'), _value(record, 'payload') or {}
        if not gene:
            continue
        scope = evidence_candidate_scope(payload)
        if scope and (scope.get('context_only') or not scope.get('direct_locus_support')):
            continue
        kind = str(_value(record, 'kind'))
        channel = {'qtl': 'QTL', 'colocalization': 'colocalisation', 'regulatory': 'E2G',
                   'variant_annotation': 'variant annotation'}.get(kind)
        if not channel:
            continue
        context = payload.get('context') or {}
        raw = (payload.get('upstream') or {}).get('raw') or {}
        if channel == 'E2G' and not (context.get('exact_variant_overlap') is True or raw.get('exactVariantOverlap') is True):
            continue
        if channel == 'variant annotation' and not context.get('consequence') and not raw.get('consequence'):
            continue
        obj = payload.get('object') or {}
        symbol = obj.get('label') if obj.get('id') == gene else None
        yield {'gene_id': gene, 'symbol': symbol or gene, 'channel': channel,
            'evidence_id': _value(record, 'evidence_id'), 'variant_id': _value(record, 'variant_id'),
            'context': context, 'raw': raw}


def selection_absence_findings(query, records, output):
    """Flag explicit false absence claims, never an LLM's choice or ranking."""
    visible = set(output.model_visible_evidence_ids) if output.model_visible_evidence_ids is not None else None
    canonical = identity_resolver(query.ld_expansion or {})
    selected = {canonical(row.variant_id) for row in query.variants}
    by_gene = defaultdict(list)
    for row in locus_rows(records):
        if visible is None or row['evidence_id'] in visible:
            by_gene[row['gene_id']].append(row)
    findings = []
    for index, decision in enumerate(output.candidate_decisions):
        for match in _ABSENCE.finditer(decision.reason):
            scope = match['scope'].strip()
            # Tissue-, phenotype-, effect- and assay-qualified negatives require
            # a richer binding. Do not replace them with generic locus presence.
            if scope and not re.fullmatch(r"(?:to|at|for) (?:the )?(?:source|selected|uploaded|locus|LD[- ]linked|proxy) (?:variants?|SNPs?|locus)", scope, re.I):
                continue
            names = re.findall(_CHANNEL, match['channels'], re.I)
            channels = {'E2G' if name.lower() in {'e2g', 'regulatory'} else
                        'QTL' if 'qtl' in name.lower() else 'variant annotation' if name.lower() == 'coding'
                        else 'colocalisation' for name in names}
            rows = [row for row in by_gene[decision.gene_id] if row['channel'] in channels]
            if re.search(r'\b(?:source|selected|uploaded)\b', scope, re.I):
                rows = [row for row in rows if canonical(row['variant_id']) in selected or any(
                    canonical(w.get('variant_id')) in selected for w in row['context'].get('locus_anchor_witnesses', []) if isinstance(w, Mapping))]
            elif re.search(r'\b(?:proxy|LD[- ]linked)\b', scope, re.I):
                rows = [row for row in rows if canonical(row['variant_id']) not in selected]
            if not rows:
                continue
            findings.append({'field': f'candidate_decisions[{index}].reason', 'metric': 'evidence_presence',
                'original_value': match.group(), 'span': [match.start(), match.end()],
                'display_replacement': 'Original LLM selection retained; the stated evidence absence requires review.',
                'reason': 'candidate_evidence_absence_contradicted', 'rule': 'frozen_candidate_evidence_presence_v1',
                'gene_id': decision.gene_id, 'evidence_ids': sorted({row['evidence_id'] for row in rows}),
                'retained_channels': sorted({row['channel'] for row in rows}),
                'original_field_sha256': sha256_json(decision.reason)})
    return findings


def unranked_locus_evidence_markdown(result, records):
    """A separate, unranked coverage view; no candidate is inserted into TopN."""
    ranked = {row.get('gene_id') for row in result.get('ranked_genes', ()) if isinstance(row, Mapping)}
    decisions = {row['gene_id']: row for row in result.get('candidate_decisions', ())
        if isinstance(row, Mapping) and row.get('status') in {'not_selected', 'unresolved'} and row.get('gene_id')}
    decisions = {gene: row for gene, row in decisions.items() if gene not in ranked}
    by_gene = defaultdict(list)
    for row in locus_rows(records):
        if row['gene_id'] in decisions:
            by_gene[row['gene_id']].append(row)
    if not by_gene:
        return ''
    lines = ['<details><summary>Other candidates with locus evidence</summary>', '',
        'These candidates remain outside the LLM ranking. Retained links do not establish causality.', '']
    for gene, rows in sorted(by_gene.items(), key=lambda item: (item[1][0]['symbol'], item[0])):
        counts = {name: sum(row['channel'] == name for row in rows) for name in sorted({row['channel'] for row in rows})}
        evidence = '; '.join(f'{name}: {count:,} retained record' + ('s' if count != 1 else '') for name, count in counts.items())
        decision = decisions[gene]
        status = 'Not selected' if decision['status'] == 'not_selected' else 'Unresolved'
        items = ((result.get('gate2_review') or {}).get('fact_issues') or {}).get('items') or []
        contradicted = any(item.get('gene_id') == gene and item.get('reason_code') ==
            'candidate_evidence_absence_contradicted' for item in items if isinstance(item, Mapping))
        lines.extend([f"**{html.escape(rows[0]['symbol'])}** — {status} by the LLM. {evidence}.", '',
            *(['The selection rationale conflicts with retained evidence and requires factual review.', ''] if contradicted else []),
            'LLM selection reason: ' + html.escape(str(decision.get('reason') or 'No explanation recorded.')), '',
            '<details><summary>Source records</summary>', '',
            'Gene: `' + html.escape(gene) + '`. Example retained evidence IDs: ' + ', '.join(
                '`' + html.escape(identifier) + '`' for identifier in sorted({row['evidence_id'] for row in rows})[:3]) + '.',
            '', '</details>', ''])
    return '\n'.join([*lines, '</details>'])
