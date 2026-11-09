"""Lossless named value references and already-proved QTL ownership joins.

Numeric values, including numbers nested in objects, stay directly visible.
Sharing changes representation only; original row order, null cells and field
presence are restored before existing scientific readers and validators run.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

VALUE_ENCODING = "named-column-value-references-v1"
VALUE_PREFIX = "value_ref:"
MEMBER_RELATIONSHIPS = {
    "study_ref": "member.study_ref",
    "member_statistics_ref": "member.id",
    "evidence_refs": "singleton(member.evidence_ref)",
    "variant_ref": "unique_variant(member.variant_id)",
    "qtl_type": "member_study.study_type",
    "support_trace_ref": "unique_single_member_trace",
}


def _identity(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _size(value: Any) -> int:
    return len(_identity(value).encode("utf-8"))


def _contains_number(value: Any) -> bool:
    if type(value) in (int, float):
        return True
    if isinstance(value, Mapping):
        return any(_contains_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_number(item) for item in value)
    return False


def share_column_values(table: Mapping[str, Any]) -> Mapping[str, Any]:
    """Use explicit, column-scoped references only when actual JSON gets smaller."""
    if "groups" in table:
        return {**table, "groups": {name: share_column_values(group) for name, group in table["groups"].items()}}
    if "encoding" in table or not {"columns", "rows", "constant_fields"}.issubset(table):
        return table
    columns, original_rows = table["columns"], table["rows"]
    if not original_rows:
        return table
    current_header = {key: value for key, value in table.items() if key != "rows"}
    current_header = {**current_header, "columns": list(columns)}
    header_size = _size(current_header)
    replacements: dict[int, list[int]] = {}
    for index, column in enumerate(columns):
        values = [row[index] for row in original_rows]
        if any(_contains_number(value) for value in values):
            continue
        reference = VALUE_PREFIX + column
        # A literal field with the same spelling is never silently reinterpreted.
        if reference in columns or reference in table["constant_fields"]:
            continue
        identities: dict[str, int] = {}
        dictionary, references = [], []
        original_cell_bytes = 0
        for value in values:
            identity = _identity(value)
            original_cell_bytes += len(identity.encode("utf-8"))
            if identity not in identities:
                identities[identity] = len(dictionary) + 1
                dictionary.append(value)
            references.append(identities[identity])
        if len(dictionary) == len(values):
            continue
        proposal = {**current_header, "columns": list(current_header["columns"]), "encoding": VALUE_ENCODING,
                    "column_values": {**current_header.get("column_values", {}),
                                      reference: {"column": column, "values": dictionary}}}
        proposal["columns"][index] = reference
        proposed_header_size = _size(proposal)
        # Row punctuation is unchanged: only the encoded cell lengths differ.
        change = proposed_header_size - header_size + sum(len(str(value)) for value in references) - original_cell_bytes
        if change < 0:
            current_header, header_size = proposal, proposed_header_size
            replacements[index] = references
    if not replacements:
        return table
    result = deepcopy(current_header)
    result["rows"] = [[replacements[index][row_index] if index in replacements else deepcopy(value)
                       for index, value in enumerate(row)] for row_index, row in enumerate(original_rows)]
    return result


def restore_column_values(table: Mapping[str, Any]) -> Mapping[str, Any]:
    """Decode declared references; ordinary legacy column/value strings are literal."""
    if "encoding" not in table:
        return table
    if table["encoding"] != VALUE_ENCODING:
        raise ValueError("Unknown scientific table value encoding")
    expected = {"columns", "constant_fields", "rows", "encoding", "column_values"}
    if not expected.issubset(table) or set(table) - expected - {"member_derived_columns"}:
        raise ValueError("Malformed scientific value-reference table")
    columns, dictionaries = table["columns"], table["column_values"]
    if (not isinstance(columns, list) or not all(isinstance(column, str) for column in columns)
            or len(columns) != len(set(columns)) or not isinstance(dictionaries, Mapping) or not dictionaries
            or not isinstance(table["rows"], list) or not isinstance(table["constant_fields"], Mapping)
            or any(not isinstance(row, list) or len(row) != len(columns) for row in table["rows"])):
        raise ValueError("Malformed scientific value-reference rows")
    result = {"columns": list(columns), "constant_fields": deepcopy(table["constant_fields"]),
              "rows": deepcopy(table["rows"])}
    if "member_derived_columns" in table:
        result["member_derived_columns"] = deepcopy(table["member_derived_columns"])
    restored_names: set[str] = set()
    for reference, definition in dictionaries.items():
        if reference not in columns or not isinstance(definition, Mapping) or set(definition) != {"column", "values"}:
            raise ValueError("Unknown scientific value-reference column")
        original, values = definition["column"], definition["values"]
        if (not isinstance(original, str) or reference != VALUE_PREFIX + original
                or original in columns or original in result["constant_fields"] or original in restored_names
                or not isinstance(values, list) or not values):
            raise ValueError("Ambiguous scientific value-reference column")
        restored_names.add(original)
        index = columns.index(reference)
        for row in result["rows"]:
            value = row[index]
            if type(value) is not int or value < 1 or value > len(values):
                raise ValueError("Invalid scientific value reference")
            row[index] = deepcopy(values[value - 1])
        result["columns"][index] = original
    return result


def _rows(table):
    table = restore_column_values(table)
    return [{**table["constant_fields"], **dict(zip(table["columns"], row, strict=True))} for row in table["rows"]]


def _unique_index(rows):
    result = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError("Missing or duplicate scientific owner ID")
        result[identifier] = row
    return result


def _member_indexes(payload):
    members = _unique_index(_rows(payload["credible_set_members"]))
    studies = _unique_index(_rows(payload["studies"]))
    variants: dict[str, list[str]] = {}
    for row in _unique_index(_rows(payload["variants"])).values():
        variants.setdefault(_identity(row.get("variant_id")), []).append(row["id"])
    traces: dict[str, list[str]] = {}
    for row in _unique_index(_rows(payload["support_traces"])).values():
        body = {key: value for key, value in row.items() if key != "id" and value is not None}
        traces.setdefault(_identity(body), []).append(row["id"])
    return members, studies, variants, traces


def _member_values(row, indexes):
    members, studies, variants, traces = indexes
    member = members.get(row.get("member_ref"))
    if member is None or not all(isinstance(member.get(key), str) and member[key]
                                 for key in ("id", "study_ref", "variant_id", "evidence_ref")):
        return None
    study = studies.get(member["study_ref"])
    if study is None or not isinstance(study.get("gene_id"), str) or not study["gene_id"]:
        return None
    variant = variants.get(_identity(member["variant_id"]), [])
    if len(variant) != 1:
        return None
    trace = traces.get(_identity({"variant_refs": variant, "member_refs": [member["id"]]}), [])
    result = {"study_ref": member["study_ref"], "member_statistics_ref": member["id"],
              "evidence_refs": [member["evidence_ref"]], "variant_ref": variant[0],
              "qtl_type": study.get("study_type")}
    if len(trace) == 1:
        result["support_trace_ref"] = trace[0]
    return result


def _same_owner(row, values, indexes):
    if values is None:
        return False
    for field in ("study_ref", "variant_ref", "evidence_refs"):
        if field in row and _identity(row[field]) != _identity(values[field]):
            return False
    gene = indexes[1][values["study_ref"]]["gene_id"]
    if "gene_id" in row and _identity(row["gene_id"]) != _identity(gene):
        return False
    return row.get("gene_from_study_ref") is True or row.get("gene_id") == gene


def share_member_columns(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove a column only when every original value equals its explicit join."""
    table = payload["molecular_qtl"]
    if "member_derived_columns" in table or not table["rows"]:
        return table
    indexes = _member_indexes(payload)
    rows = _rows(table)
    expected = [_member_values(row, indexes) for row in rows]
    if not all(_same_owner(row, value, indexes) for row, value in zip(rows, expected, strict=True)):
        return table
    declarations = []
    for field, relationship in MEMBER_RELATIONSHIPS.items():
        if field not in table["columns"]:
            continue
        if all(field in value and field in row and _identity(row[field]) == _identity(value[field])
               for row, value in zip(rows, expected, strict=True)):
            declarations.append({"column": field, "position": table["columns"].index(field), "relationship": relationship})
    if not declarations:
        return table
    removed = {item["column"] for item in declarations}
    indices = [index for index, column in enumerate(table["columns"]) if column not in removed]
    result = {**table, "columns": [table["columns"][index] for index in indices],
              "rows": [[row[index] for index in indices] for row in table["rows"]],
              "member_derived_columns": declarations}
    return result if _size(result) < _size(table) else table


def restore_member_columns(payload: Mapping[str, Any], table: Mapping[str, Any]) -> Mapping[str, Any]:
    if "member_derived_columns" not in table:
        return table
    declarations = table["member_derived_columns"]
    if (not isinstance(declarations, list) or not declarations
            or any(not isinstance(item, Mapping) or set(item) != {"column", "position", "relationship"}
                   or item.get("column") not in MEMBER_RELATIONSHIPS
                   or item.get("relationship") != MEMBER_RELATIONSHIPS[item["column"]]
                   or type(item.get("position")) is not int for item in declarations)
            or len({item["column"] for item in declarations}) != len(declarations)
            or len({item["position"] for item in declarations}) != len(declarations)):
        raise ValueError("Malformed scientific member-column relationships")
    indexes = _member_indexes(payload)
    rows = _rows(table)
    values = [_member_values(row, indexes) for row in rows]
    if not all(_same_owner(row, value, indexes) for row, value in zip(rows, values, strict=True)):
        raise ValueError("Conflicting scientific member owner")
    result = {"columns": list(table["columns"]), "constant_fields": deepcopy(table["constant_fields"]),
              "rows": deepcopy(table["rows"])}
    for item in sorted(declarations, key=lambda value: value["position"]):
        field, position = item["column"], item["position"]
        if (field in result["columns"] or field in result["constant_fields"] or position < 0
                or position > len(result["columns"]) or any(field not in value for value in values)):
            raise ValueError("Missing or ambiguous scientific member relationship")
        result["columns"].insert(position, field)
        for row, value in zip(result["rows"], values, strict=True):
            row.insert(position, deepcopy(value[field]))
    return result


def share_scientific_capacity(payload: dict[str, Any]) -> None:
    """Called after existing scientific grouping; aliases/bindings are external."""
    payload["molecular_qtl"] = share_member_columns(payload)
    for name, table in list(payload.items()):
        if isinstance(table, Mapping) and ("groups" in table or {"columns", "rows", "constant_fields"}.issubset(table)):
            payload[name] = share_column_values(table)
