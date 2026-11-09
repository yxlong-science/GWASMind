"""Versioned, reversible scientific result encodings with named foreign keys.

Catalog identifiers are immutable within a frozen input. They are dictionary
keys, never list positions, and are assigned before presentation randomization.
The decoder is also the validation boundary for the actual provider payload.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy


def _signature(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class _Catalog:
    def __init__(self, prefix):
        self.prefix = prefix
        self.values = {}
        self.lookup = {}

    def add(self, value):
        signature = _signature(value)
        if signature not in self.lookup:
            key = f"{self.prefix}{len(self.values) + 1}"
            self.lookup[signature] = key
            self.values[key] = deepcopy(value)
        return self.lookup[signature]


def _lookup(catalog, reference, description):
    try:
        if isinstance(catalog, list) and (type(reference) is not int or reference < 0):
            raise KeyError(reference)
        if not isinstance(catalog, (list, Mapping)):
            raise KeyError(reference)
        return catalog[reference]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unknown {description} reference: {reference!r}") from exc


def decode_colocalization_record(comparison, index):
    """Decode historical positional v1 or named-key v2 and verify the index join."""
    if index.get("kind") != "colocalizations" or "record_ref" not in index:
        return deepcopy(index)
    catalog = comparison.get("colocalization_record_catalog", {})
    if catalog.get("format") not in {"colocalization_record_catalog_v1", "colocalization_record_catalog_v2"}:
        raise ValueError("Unsupported colocalization catalog format")
    row = _lookup(catalog.get("rows", {}), index["record_ref"], "colocalization record")
    columns = catalog.get("columns", [])
    if len(columns) != len(set(columns)) or len(columns) != len(row):
        raise ValueError("Colocalization table columns and values do not align")
    common = catalog.get("common_fields", {})
    if set(common).intersection(columns):
        raise ValueError("Colocalization common fields overwrite row values")
    decoded = {**deepcopy(common), **dict(zip(columns, deepcopy(row), strict=True))}
    for field, objects in catalog.get("objects", {}).items():
        reference = decoded.pop(field + "_ref", None)
        decoded[field] = deepcopy(_lookup(objects, reference, field))
    for field in ("evidence_id", "evidence_ref", "fact_id", "gene_id", "kind"):
        if decoded.get(field) != index.get(field):
            raise ValueError(f"Colocalization index identity mismatch: {field}")
    return decoded


_BIOSAMPLE_COLUMNS = ("tissue_id", "cell_type_id", "tissue", "cell_type")
_GROUP_COLUMNS = ("id", "rationale", "members", "resolved_biosamples")
_METHOD_FIELDS = ("comparison_axis", "comparison_tissue", "comparison_tissue_id", "unit",
                  "data_release", "datasource", "datatype", "detection_rule", "tau_scale")


def compact_expression_results(profiles):
    """Share biosamples, group definitions, methods and schemas across all genes.

    Every computed result, missingness reason and matched measurement survives.
    Table constants replace repeated scalar cells; zero, null and false remain
    distinct. Reference-panel summarization happens explicitly upstream.
    """
    catalogs = {name: _Catalog(prefix) for name, prefix in (
        ("biosamples", "B"), ("group_definitions", "G"), ("methods", "M"),
        ("selectors", "X"), ("schemas", "S"), ("table_constants", "K"))}

    def encode(value, field=None):
        if isinstance(value, Mapping):
            if set(value) == {"columns", "rows"}:
                return table(value["columns"], value["rows"], field=field)
            if field == "group" and any(name in value for name in _METHOD_FIELDS):
                method = {name: deepcopy(value[name]) for name in _METHOD_FIELDS if name in value}
                return {"method_ref": catalogs["methods"].add(method),
                        **{name: encode(item, name) for name, item in value.items() if name not in method}}
            return {name: encode(item, name) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            if field in {"tau_peak_biosamples", "resolved_biosamples", "unavailable_biosamples"} and all(
                    isinstance(row, (list, tuple)) and len(row) == 4 for row in value):
                return {"biosample_refs": [catalogs["biosamples"].add(dict(zip(_BIOSAMPLE_COLUMNS, row, strict=True)))
                                           for row in value]}
            if field in {"matched_context_selectors", "members", "context_selectors"} and value:
                return {"selector_ref": catalogs["selectors"].add(value)}
            if field == "groups" and value and all(isinstance(row, Mapping) for row in value):
                encoded = [encode(row, "group") for row in value]
                # Preserve absent keys as absent, rather than silently changing
                # missing fields into explicitly reported null measurements.
                if all(set(row) == set(encoded[0]) for row in encoded):
                    columns = sorted(encoded[0])
                    return table(columns, [[row[name] for name in columns] for row in encoded],
                                 field="group_results", mapping_rows=True, encoded=True)
                return encoded
            return [encode(item) for item in value]
        return value

    def table(columns, rows, *, field, mapping_rows=False, encoded=False):
        columns = list(columns)
        if len(set(columns)) != len(columns) or any(len(row) != len(columns) for row in rows):
            raise ValueError("Expression table columns and values do not align")
        original_columns = list(columns)
        nested_columns = {}
        if mapping_rows and "reference_panel_summary" in columns and rows:
            position = columns.index("reference_panel_summary")
            summaries = [row[position] for row in rows]
            if all(isinstance(summary, Mapping) and set(summary) == set(summaries[0]) for summary in summaries):
                nested_columns["reference_panel_summary"] = sorted(summaries[0])
                suffixes = nested_columns["reference_panel_summary"]
                columns = [*columns[:position], *(f"reference_panel_summary.{key}" for key in suffixes), *columns[position + 1:]]
                rows = [[*row[:position], *(row[position][key] for key in suffixes), *row[position + 1:]] for row in rows]
        identity_columns = [name for name in (_GROUP_COLUMNS if field == "group_enrichment" else _BIOSAMPLE_COLUMNS)
                            if name in columns]
        catalog_name = "group_definitions" if field == "group_enrichment" else "biosamples"
        ref_name = "group_ref" if field == "group_enrichment" else "biosample_ref"
        if not identity_columns:
            ref_name = None
        encoded_columns = ([ref_name] if ref_name else []) + [name for name in columns if name not in identity_columns]
        encoded_rows = []
        for row in rows:
            values = dict(zip(columns, row, strict=True))
            result = {}
            if ref_name:
                definition = {name: encode(values[name], name) for name in identity_columns}
                result[ref_name] = catalogs[catalog_name].add(definition)
            result.update({name: values[name] if encoded else encode(values[name], name)
                           for name in columns if name not in identity_columns})
            encoded_rows.append(result)
        constants = {name: encoded_rows[0][name] for name in encoded_columns
                     if len(encoded_rows) > 1 and all(_signature(row[name]) == _signature(encoded_rows[0][name])
                                                     for row in encoded_rows)}
        variable = [name for name in encoded_columns if name not in constants]
        schema = {"columns": encoded_columns, "original_columns": original_columns}
        if nested_columns:
            schema["nested_columns"] = nested_columns
        if ref_name:
            schema["identity_columns"] = identity_columns
        if mapping_rows:
            schema["mapping_rows"] = True
        return {"schema_ref": catalogs["schemas"].add(schema),
                "constant_ref": catalogs["table_constants"].add(constants),
                "rows": [[row[name] for name in variable] for row in encoded_rows]}

    encoded_profiles = {name: encode(profile) for name, profile in profiles.items()}
    return {"format": "expression_profile_catalog_v4", "profiles": encoded_profiles,
            **{name: catalog.values for name, catalog in catalogs.items()},
            "encoding": (
                "Named dictionary references are immutable. For each schema_ref table, read schema.columns "
                "in order excluding keys in table_constants[constant_ref], zip remaining columns with each "
                "row, and add those constants. Dotted reference_panel_summary columns belong to that summary. "
                "Expand biosample_ref/group_ref/method_ref/selector_ref from their named "
                "dictionaries; biosample_refs lists retain their order. Every computed result, matched "
                "measurement and unavailable/partial reason is retained. Null is unknown, never zero. "
                "Full raw expression and full reference panels remain in the local scientific archive."
            )}


def expand_expression_results(catalog, profile_ref, *, _used=None):
    """Decode a v4 scientific expression result without relying on list order."""
    if catalog.get("format") != "expression_profile_catalog_v4":
        raise ValueError("Unsupported expression result catalog format")

    def lookup(name, reference):
        value = _lookup(catalog.get(name, {}), reference, "expression " + name)
        if _used is not None:
            _used.setdefault(name, set()).add(reference)
        return value

    def decode(value):
        if isinstance(value, Mapping):
            if set(value) == {"biosample_refs"}:
                return [[lookup("biosamples", reference).get(key)
                         for key in _BIOSAMPLE_COLUMNS] for reference in value["biosample_refs"]]
            if set(value) == {"selector_ref"}:
                return deepcopy(lookup("selectors", value["selector_ref"]))
            if "schema_ref" in value:
                schema = lookup("schemas", value["schema_ref"])
                columns = schema["columns"]
                constants = lookup("table_constants", value["constant_ref"])
                if not set(constants).issubset(columns) or len(columns) != len(set(columns)):
                    raise ValueError("Expression table constant fields do not match its schema")
                variable = [name for name in columns if name not in constants]
                rows = []
                for row in value["rows"]:
                    if len(row) != len(variable):
                        raise ValueError("Expression table columns and values do not align")
                    expanded = {name: decode(item) for name, item in {**constants, **dict(zip(variable, row, strict=True))}.items()}
                    for ref_name, catalog_name in (("biosample_ref", "biosamples"), ("group_ref", "group_definitions")):
                        if ref_name in expanded:
                            identity = decode(lookup(catalog_name, expanded.pop(ref_name)))
                            if set(identity).intersection(expanded):
                                raise ValueError("Expression identity columns overwrite measurements")
                            expanded.update(identity)
                    original = schema["original_columns"]
                    for name, fields in schema.get("nested_columns", {}).items():
                        expanded[name] = {key: expanded.pop(f"{name}.{key}") for key in fields}
                    if set(expanded) != set(original):
                        raise ValueError("Expression result does not reconstruct its original columns")
                    rows.append(decode(expanded) if schema.get("mapping_rows") else [expanded[name] for name in original])
                return rows if schema.get("mapping_rows") else {"columns": list(schema["original_columns"]), "rows": rows}
            decoded = {name: decode(item) for name, item in value.items() if name != "method_ref"}
            if "method_ref" in value:
                method = deepcopy(lookup("methods", value["method_ref"]))
                if set(method).intersection(decoded):
                    raise ValueError("Expression method fields overwrite results")
                decoded.update(method)
            return decoded
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value

    return decode(lookup("profiles", profile_ref))


def _validate_colocalization_catalog_membership(comparison):
    """Everything exposed in a scientific catalog must belong to a visible row."""
    catalog = comparison.get("colocalization_record_catalog", {})
    indexes = [row for row in comparison.get("citation_records", [])
               if row.get("kind") == "colocalizations" and "record_ref" in row]
    rows = catalog.get("rows", {})
    expected_rows = set(range(len(rows))) if isinstance(rows, list) else set(rows)
    references = [row["record_ref"] for row in indexes]
    if len(set(references)) != len(references) or set(references) != expected_rows:
        raise ValueError("Colocalization catalog contains orphan or duplicated row references")
    allowed = {"context", "credible_set_member", "right_study", "study", "statistics", "other_reported_statistics", "variant"}
    objects = catalog.get("objects", {})
    if objects and set(objects) != allowed:
        raise ValueError("Colocalization catalog object families do not match the schema")
    for field, values in objects.items():
        position = catalog.get("columns", []).index(field + "_ref")
        used = {_lookup(rows, reference, "colocalization record")[position] for reference in references}
        keys = set(range(len(values))) if isinstance(values, list) else set(values)
        if used != keys:
            raise ValueError(f"Colocalization catalog contains orphan {field} objects")


def _strip_audit(value):
    from .ranking_input import _field_is_audit_only

    if isinstance(value, Mapping):
        if {"columns", "rows"}.issubset(value):
            keep = [i for i, name in enumerate(value["columns"]) if not _field_is_audit_only(name)]
            result = {name: _strip_audit(item) for name, item in value.items() if name not in {"columns", "rows"}
                      and not _field_is_audit_only(name)}
            result.update(columns=[value["columns"][i] for i in keep],
                          rows=[[_strip_audit(row[i]) for i in keep] for row in value["rows"]])
            return result
        return {name: _strip_audit(item) for name, item in value.items() if not _field_is_audit_only(name)}
    if isinstance(value, (tuple, list)):
        return [_strip_audit(item) for item in value]
    return value


def _rewrite_aliases(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, Mapping):
        return {mapping.get(name, name): _rewrite_aliases(item, mapping) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_rewrite_aliases(item, mapping) for item in value]
    return value


def validate_model_gene_comparison(payload, source_comparison, *, alias_mapping=None):
    """Fail closed if final model science differs from the expanded card source.

    This must run after cleaning and alias rebinding. The identity-index join is
    insufficient: all decoded study, gene, phenotype, member, variant, tissue and
    statistic fields are compared, as are every expression result and card fact.
    Returns a small receipt for local audit storage, never the scientific prompt.
    """
    from .ranking_cards import _summarize_expression_profile, expand_expression_profile_catalog

    comparison = payload.get("gene_comparison", payload)
    source = _strip_audit(_rewrite_aliases(source_comparison, alias_mapping or {}))
    generated = {"expression_profile_catalog", "colocalization_record_catalog", "model_compaction"}
    if set(comparison) != set(source) | generated:
        raise ValueError("Scientific comparison fields do not match the encoded schema")
    for field in set(source) - {"comparison_cards", "citation_records"}:
        if _signature(_strip_audit(comparison.get(field))) != _signature(source[field]):
            raise ValueError(f"Scientific comparison metadata changed after encoding: {field}")
    _validate_colocalization_catalog_membership(comparison)
    decoded = [decode_colocalization_record(comparison, row) for row in comparison.get("citation_records", [])]
    expected_records = source.get("citation_records", [])
    actual_by_id = {row["evidence_id"]: row for row in decoded}
    expected_by_id = {row["evidence_id"]: row for row in expected_records}
    if len(actual_by_id) != len(decoded) or len(expected_by_id) != len(expected_records):
        raise ValueError("Scientific payload contains duplicated citation identities")
    if set(actual_by_id) != set(expected_by_id):
        raise ValueError("Scientific payload citation membership changed")
    for identifier, expected in expected_by_id.items():
        expected = deepcopy(expected)
        if expected.get("kind") == "colocalizations":
            expected.pop("context_assessment", None)
            # Legacy expanded cards omit unavailable nested fields while the
            # stable catalog represents their absence explicitly as null.
            for field in comparison.get("colocalization_record_catalog", {}).get("objects", {}):
                expected.setdefault(field, None)
        actual = actual_by_id[identifier]
        if _signature(actual) != _signature(expected):
            changed = sorted(key for key in set(expected) | set(actual)
                             if _signature(expected.get(key)) != _signature(actual.get(key)))
            raise ValueError(f"Scientific citation changed after encoding: {identifier}; fields={changed}")
    actual_cards = {row["gene_id"]: row for row in comparison.get("comparison_cards", [])}
    expected_cards = {row["gene_id"]: row for row in source.get("comparison_cards", [])}
    if len(actual_cards) != len(comparison.get("comparison_cards", [])) or set(actual_cards) != set(expected_cards):
        raise ValueError("Scientific payload candidate membership changed")
    profiles_checked = 0
    expression_used = {}
    expression_catalog = comparison.get("expression_profile_catalog", {})
    for gene, expected in expected_cards.items():
        actual = deepcopy(actual_cards[gene])
        expected = deepcopy(expected)
        profile = expected.get("expression_profiles")
        if isinstance(profile, Mapping):
            reference = actual.get("expression_profiles", {}).get("profile_ref")
            actual["expression_profiles"] = (
                expand_expression_results(expression_catalog, reference, _used=expression_used)
                if expression_catalog.get("format") == "expression_profile_catalog_v4"
                else expand_expression_profile_catalog(expression_catalog, reference))
            expected["expression_profiles"] = _summarize_expression_profile(profile)
            profiles_checked += 1
        if _signature(actual) != _signature(expected):
            changed = sorted(key for key in set(expected) | set(actual)
                             if _signature(expected.get(key)) != _signature(actual.get(key)))
            raise ValueError(f"Scientific candidate facts changed after encoding: {gene}; fields={changed}")
    if expression_catalog.get("format") == "expression_profile_catalog_v4":
        families = {"profiles", "biosamples", "group_definitions", "methods", "selectors", "schemas", "table_constants"}
        if set(expression_catalog) != families | {"format", "encoding"}:
            raise ValueError("Expression catalog fields do not match the schema")
        for family in families:
            if set(expression_catalog[family]) != expression_used.get(family, set()):
                raise ValueError(f"Expression catalog contains orphan {family} objects")
    return {"format": "ranking_payload_integrity_v1", "status": "passed",
            "citation_records_checked": len(decoded),
            "colocalization_records_checked": sum(row.get("kind") == "colocalizations" for row in decoded),
            "expression_profiles_checked": profiles_checked, "candidate_cards_checked": len(actual_cards)}
