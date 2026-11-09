"""Gene-specific bibliography and honest contextual-selection metadata."""
from __future__ import annotations

import re
from datetime import date


def resolved_literature_traits(values=()):
    """Retain only already-mapped ontology IDs as contextual metadata."""
    if not isinstance(values, (list, tuple)) or len(values) > 100:
        raise ValueError('trait_ids must be an array of at most 100 mapped identifiers')
    return tuple(sorted({value.strip() for value in values if isinstance(value, str)
        and re.fullmatch(r'(?:EFO|MONDO|HP|DOID|Orphanet|OTAR)_[0-9]+', value.strip())}))


def literature_field(trait_ids=()):
    resolved_literature_traits(trait_ids)
    # OT LiteratureQuery uses keywordId IN(ids): additionalIds is a union,
    # not gene AND phenotype. Sending a disease would admit papers that never
    # mention this target. Keep gene-specific rows, then use retrieved titles
    # and abstracts for local phenotype priority in the publication projection.
    return 'literatureOcurrences { count cursor rows { pmid pmcid publicationDate } }'


def _order(row):
    value = str(row.get('publicationDate') or '')
    try:
        published = date.fromisoformat(value[:10]).toordinal()
    except ValueError:
        published = 0
    pmid = str(row.get('pmid') or '')
    return (-published, -int(pmid) if pmid.isdigit() else 0, pmid)


def prepare_literature_profiles(targets, trait_ids=()):
    from .live import EvidenceKind, _bounded_connection_coverage, _connection_rows

    traits = resolved_literature_traits(trait_ids)
    result = []
    for target in targets:
        if target is None:
            result.append(None)
            continue
        _bounded_connection_coverage(target, 'literatureOcurrences', EvidenceKind.LITERATURE, target['id'])
        scope = {'phenotype_filtered': False, 'trait_ids': list(traits),
            'mode': 'gene_background',
            'selection_method': 'local_title_abstract_phenotype_priority',
            'filter_status': 'not_applied_upstream_union_semantics' if traits else 'not_applied_no_resolved_trait'}
        rows = [{**row, 'literatureScope': dict(scope)} for row in _connection_rows(target, 'literatureOcurrences')]
        connection = target['literatureOcurrences']
        result.append({**target, 'literatureOcurrences': {**connection, 'rows': sorted(rows, key=_order)},
            '_literature_connections': [{'trait_ids': [], 'connection': connection}]})
    return result
