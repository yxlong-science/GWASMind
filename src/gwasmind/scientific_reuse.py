"""Natural table joins for values already proved identical in the fact view.

This module changes representation only. It never supplies an absent value,
changes evidence identity, combines conflicting measurements, or filters rows.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from collections import Counter
from typing import Any

from .scientific_capacity import restore_column_values, restore_member_columns


_MEMBER_FIELDS = tuple('statistics.' + name for name in (
    'beta', 'pValueExponent', 'pValueMantissa', 'posteriorProbability', 'standardError'))
_OTHER_PREFIX = 'other_reported_statistics.'


def _identity(value: Any) -> str:
    # JSON spelling distinguishes bool/int/float and nested null/missing keys.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _rows(table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Preserve all encoded columns, including explicit null cells."""
    table = restore_column_values(table)
    return [{**table['constant_fields'], **dict(zip(table['columns'], row, strict=True))}
            for row in table['rows']]


def _pack(rows: list[dict[str, Any]], *, null_constants=()) -> dict[str, Any]:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    # The public reader distinguishes explicit constant nulls from missing
    # values encoded as null column cells. Do not promote the latter.
    constants = {key: rows[0][key] for key in columns if rows and key in rows[0]
        and (rows[0][key] is not None or key in null_constants)
        and all(key in row and _identity(row[key]) == _identity(rows[0][key]) for row in rows)}
    columns = [key for key in columns if key not in constants]
    return {'columns': columns, 'constant_fields': constants,
            'rows': [[row.get(key) for key in columns] for row in rows]}


def _same_member_binding(row, member, variant, study) -> bool:
    return (member is not None and variant is not None and study is not None
        and bool(row.get('gene_id')) and bool(row.get('study_ref'))
        and member.get('study_ref') == row['study_ref']
        and study.get('gene_id') == row['gene_id']
        and member.get('variant_id') == variant.get('variant_id')
        and bool(member.get('evidence_ref')) and row.get('evidence_refs') == [member['evidence_ref']])


def share_scientific_measurements(payload: dict[str, Any], *, reuse_identities: bool = True) -> None:
    """Factor exact repeats after conventional tables have been constructed."""
    annotations = _rows(payload['gene_annotations'])
    # Generated A identifiers preserve the original table order without a
    # second model-visible list of every annotation ID. Unknown legacy ID
    # schemes simply keep their original table.
    if annotations and all(row.get('id') == f'A{index + 1}' for index, row in enumerate(annotations)):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in annotations:
            groups.setdefault(str(row['kind']), []).append(row)
        null_constants = {key for key, value in payload['gene_annotations']['constant_fields'].items() if value is None}
        grouped = {'groups': {kind: _pack(rows, null_constants=null_constants) for kind, rows in groups.items()}}
        if len(_identity(grouped)) < len(_identity(payload['gene_annotations'])):
            payload['gene_annotations'] = grouped

    members = {row['id']: row for row in _rows(payload['credible_set_members'])}
    variants = {row['id']: row for row in _rows(payload['variants'])}
    studies = {row['id']: row for row in _rows(payload['studies'])}
    qtls = _rows(payload['molecular_qtl'])
    profiles: list[dict[str, Any]] = []
    profile_ids: dict[str, str] = {}
    profile_values = []
    for row in qtls:
        other = {key: value for key, value in row.items() if key.startswith(_OTHER_PREFIX)}
        profile_values.append({**{key: row[key] for key in ('source_id', 'study_ref', 'gene_id')}, **other}
            if other and any(value is not None for value in other.values())
            and all(row.get(key) for key in ('source_id', 'study_ref', 'gene_id')) else None)
    profile_counts = Counter(_identity(value) for value in profile_values if value is not None)
    changed = False
    for row, original_profile in zip(qtls, profile_values, strict=True):
        member = members.get(row.get('member_ref'))
        variant = variants.get(row.get('variant_ref'))
        study = studies.get(row.get('study_ref'))
        same_binding = _same_member_binding(row, member, variant, study)
        if same_binding and all(key in row and key in member
            and type(row[key]) in (int, float) and type(row[key]) is type(member[key])
            and _identity(row[key]) == _identity(member[key]) for key in _MEMBER_FIELDS):
            for key in _MEMBER_FIELDS:
                del row[key]
            row['member_statistics_ref'] = member['id']
            changed = True
        # These are explicitly the original OTHER reported values. Sharing a
        # value profile must never replace the variant member's measurements.
        if original_profile is not None and profile_counts[_identity(original_profile)] > 1:
            profile = original_profile
            key = _identity(profile)
            if key not in profile_ids:
                profile_ids[key] = f'OS{len(profiles) + 1}'
                profiles.append({'id': profile_ids[key], **profile})
            for field in tuple(row):
                if not field.startswith(_OTHER_PREFIX):
                    continue
                del row[field]
            row['other_statistics_ref'] = profile_ids[key]
            changed = True
    if changed:
        null_constants = {key for key, value in payload['molecular_qtl']['constant_fields'].items() if value is None}
        payload['molecular_qtl'] = _pack(qtls, null_constants=null_constants)
    if profiles:
        payload['reported_statistics_profiles'] = _pack(profiles)
    if reuse_identities:
        share_scientific_identities(payload)


def _null_constants(table):
    return {key for key, value in table['constant_fields'].items() if value is None}


def _annotation_entities(row, variants):
    variant = variants.get(row.get('variant_ref'), {})
    identifier = variant.get('variant_id')
    subject = {'id': identifier, 'type': 'variant'} if isinstance(identifier, str) and identifier else None
    gene = row.get('gene_id')
    context = row.get('context')
    symbol = context.get('approved_symbol') if isinstance(context, Mapping) else None
    obj = {'id': gene, 'label': symbol, 'type': 'target'} if (
        isinstance(gene, str) and gene and isinstance(symbol, str) and symbol) else None
    return subject, obj


def share_scientific_identities(payload: dict[str, Any]) -> None:
    """Reuse only the exact identity already present on a referenced entity."""
    studies = {row['id']: row for row in _rows(payload['studies'])}
    variants = {row['id']: row for row in _rows(payload['variants'])}
    old_qtl = payload['molecular_qtl']
    rows = _rows(old_qtl)
    for row in rows:
        gene = row.get('gene_id')
        study_gene = studies.get(row.get('study_ref'), {}).get('gene_id')
        if isinstance(gene, str) and gene and type(study_gene) is type(gene) and gene == study_gene:
            del row['gene_id']
            row['gene_from_study_ref'] = True
    packed = _pack(rows, null_constants=_null_constants(old_qtl))
    if len(_identity(packed)) < len(_identity(old_qtl)):
        payload['molecular_qtl'] = packed

    annotations = payload['gene_annotations']
    tables = annotations['groups'] if 'groups' in annotations else {'all': annotations}
    for kind, table in list(tables.items()):
        rows = _rows(table)
        for row in rows:
            if row.get('kind') != 'variant_annotation':
                continue
            subject, obj = _annotation_entities(row, variants)
            for field, expected, flag in (('subject', subject, 'subject_from_variant_ref'),
                                         ('object', obj, 'object_from_gene_context')):
                if expected is not None and _identity(row.get(field)) == _identity(expected):
                    del row[field]
                    row[flag] = True
        packed = _pack(rows, null_constants=_null_constants(table))
        if len(_identity(packed)) < len(_identity(table)):
            if 'groups' in annotations:
                annotations['groups'][kind] = packed
            else:
                payload['gene_annotations'] = packed


def _restore_annotation_entities(rows, payload):
    variants = {row['id']: row for row in _rows(payload['variants'])}
    for row in rows:
        subject, obj = _annotation_entities(row, variants)
        for field, expected, flag in (('subject', subject, 'subject_from_variant_ref'),
                                     ('object', obj, 'object_from_gene_context')):
            shared = row.pop(flag, None)
            if shared is not None:
                if shared is not True or expected is None or (row.get(field) is not None
                    and _identity(row[field]) != _identity(expected)):
                    raise ValueError('Shared annotation identity does not match its explicit referenced owner')
                row[field] = expected


def restored_scientific_table(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return the original logical table for existing program consumers.

    The model sees the named groups/member/profile references. This join is
    deterministic and also permits independent saved-payload roundtrip audits.
    """
    table = restore_column_values(payload[name])
    if name == 'molecular_qtl':
        table = restore_member_columns(payload, table)
    if name == 'gene_annotations':
        if 'groups' in table:
            rows = [row for group in table['groups'].values() for row in _rows(group)]
            rows.sort(key=lambda row: int(row['id'][1:]))
            null_constants = {key for group in table['groups'].values()
                              for key, value in group['constant_fields'].items() if value is None}
        elif any(key in table['columns'] or key in table['constant_fields']
                 for key in ('subject_from_variant_ref', 'object_from_gene_context')):
            rows = _rows(table)
            null_constants = _null_constants(table)
        else:
            return table
        _restore_annotation_entities(rows, payload)
        return _pack(rows, null_constants=null_constants)
    if name != 'molecular_qtl' or not any(key in table['columns'] or key in table['constant_fields']
        for key in ('member_statistics_ref', 'other_statistics_ref', 'gene_from_study_ref')):
        return table
    rows = _rows(table)
    members = {row['id']: row for row in _rows(payload['credible_set_members'])}
    variants = {row['id']: row for row in _rows(payload['variants'])}
    studies = {row['id']: row for row in _rows(payload['studies'])}
    profiles = {row['id']: row for row in _rows(payload['reported_statistics_profiles'])} if 'reported_statistics_profiles' in payload else {}
    for row in rows:
        gene_from_study = row.pop('gene_from_study_ref', None)
        if gene_from_study is not None:
            gene = studies.get(row.get('study_ref'), {}).get('gene_id')
            if gene_from_study is not True or not isinstance(gene, str) or not gene or (
                row.get('gene_id') is not None and _identity(row['gene_id']) != _identity(gene)):
                raise ValueError('Shared QTL gene identity does not match its referenced study owner')
            row['gene_id'] = gene
        member_ref = row.pop('member_statistics_ref', None)
        if member_ref is not None:
            if member_ref != row.get('member_ref') or not _same_member_binding(
                row, members.get(member_ref), variants.get(row.get('variant_ref')), studies.get(row.get('study_ref'))
            ):
                raise ValueError('Shared member statistics do not match the QTL scientific owner')
            for key in _MEMBER_FIELDS:
                row[key] = deepcopy(members[member_ref][key])
        profile_ref = row.pop('other_statistics_ref', None)
        if profile_ref is not None:
            if any(profiles[profile_ref].get(key) != row.get(key) for key in ('source_id','study_ref','gene_id')):
                raise ValueError('Shared reported-statistics profile does not match its source, study and gene')
            row.update({key: deepcopy(value) for key, value in profiles[profile_ref].items()
                        if key.startswith(_OTHER_PREFIX)})
    null_constants = {key for key, value in table['constant_fields'].items() if value is None}
    return _pack(rows, null_constants=null_constants)
