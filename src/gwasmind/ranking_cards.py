"""Deterministic gene comparison summaries over the complete scientific ledger.

These cards select evidence for interpretation, never assign gene scores/ranks.
All direct molecular facts remain visible. Large regulatory/background tables
are explicitly summarized, with auditable membership outside the model prompt.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from statistics import median

from .configuration import parameter
from .scientific_facts import ScientificFactView


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _compact_expression_profiles(profiles):
    """Encode native tissue/cell profiles losslessly as columnar model-card data.

    The full native profile remains in the scientific fact view.  Model cards use
    columns plus arrays so repeated keys across thousands of reference samples do
    not consume the provider context window.  No measurement, status, selector,
    reference-panel value, or enrichment field is dropped.
    """
    if not isinstance(profiles, Mapping):
        return profiles

    def table(rows, columns):
        # New computed coverage/partial-result fields must survive without a
        # synchronized whitelist update in the prompt serializer.
        extra = sorted({key for row in rows if isinstance(row, Mapping) for key in row} - set(columns))
        columns = [*columns, *extra]
        return {
            "columns": columns,
            "rows": [[row.get(column) for column in columns] for row in rows if isinstance(row, Mapping)],
        }

    def compact_group(group):
        if not isinstance(group, Mapping):
            return group
        panel_columns = ["tissue_id", "cell_type_id", "tissue", "cell_type", "expression_value"]
        matched_columns = [
            "tissue_id", "cell_type_id", "tissue", "cell_type", "expression_value",
            "matched_context_selectors", "detection_pass", "background_max", "background_mean",
            "fold_enrichment", "zero_background", "tissue_enriched_fold_pass",
            "tissue_enhanced_fold_pass", "z", "z_above_support_threshold", "above_threshold",
        ]
        metadata = {
            key: value for key, value in group.items()
            if key not in {"reference_panel", "matched_biosamples", "group_enrichment"}
        }
        compact = {"metadata": metadata}
        compact["reference_panel"] = table(group.get("reference_panel", []), panel_columns)
        compact["matched_biosamples"] = table(group.get("matched_biosamples", []), matched_columns)
        enrichment_columns = [
            "id", "rationale", "members", "resolved_biosamples", "status", "group_mean",
            "outside_max", "all_members_detected", "fold_enrichment", "zero_background", "qualified",
        ]
        compact["group_enrichment"] = table(group.get("group_enrichment", []), enrichment_columns)
        return compact

    return {
        axis: {
            key: value for key, value in profile.items() if key != "groups"
        } | {"groups": [compact_group(group) for group in profile.get("groups", [])]}
        for axis, profile in profiles.items()
        if isinstance(profile, Mapping)
    }


def compact_model_gene_comparison(payload):
    """Reduce repeated model-prompt scaffolding without dropping card facts.

    The scientific card view remains fully expanded for audit and validation.
    The provider-facing copy interns repeated expression profiles and removes
    the verbose ontology-mapping trace from colocalization citations; the
    compact context summary and the complete scientific fact view retain that
    information outside the model prompt.
    """
    result = deepcopy(payload)
    cards = result.get("comparison_cards", [])
    profile_catalog = {}
    profile_ids = {}
    for card in cards:
        profile = card.get("expression_profiles")
        if not isinstance(profile, Mapping):
            continue
        profile_key = _hash(profile)
        profile_id = profile_ids.setdefault(profile_key, f"EP{len(profile_ids) + 1}")
        profile_catalog.setdefault(profile_id, _summarize_expression_profile(profile))
        card["expression_profiles"] = {"profile_ref": profile_id}
    result["citation_records"], coloc_catalog = _compact_colocalization_records(
        result.get("citation_records", [])
    )
    from .ranking_payload_codec import compact_expression_results

    result["expression_profile_catalog"] = compact_expression_results(profile_catalog)
    result["colocalization_record_catalog"] = coloc_catalog
    result["model_compaction"] = {
        "format": "gene_comparison_model_compaction_v2",
        "expression_profiles": (
            "Cards reference a profile-result summary by profile_ref. It includes both tissue and cell axes, "
            "all context groups, context-matched measurements, specificity statistics, and reference-panel "
            "summaries; complete reference rows remain in the scientific fact view."
        ),
        "colocalization_context_assessment": (
            "The verbose ontology-mapping trace is retained in full scientific facts; "
            "each citation retains the resolved context summary."
        ),
    }
    return result


def _summarize_expression_profile(profile):
    """Send computed expression results, not an entire repeated reference panel."""
    if "groups" not in profile:
        return {
            axis: _summarize_expression_profile(axis_profile)
            for axis, axis_profile in profile.items()
            if isinstance(axis_profile, Mapping)
        }
    result = {key: value for key, value in profile.items() if key != "groups"}
    groups = []
    for group in profile.get("groups", []):
        metadata = group.get("metadata", {})
        row_table = group.get("reference_panel", {})
        columns = list(row_table.get("columns", []))
        rows = list(row_table.get("rows", []))
        try:
            value_index = columns.index("expression_value")
        except ValueError:
            value_index = None
        numeric = [
            row[value_index] for row in rows
            if value_index is not None and len(row) > value_index
            and _number(row[value_index])
        ]
        top_rows = sorted(
            (row for row in rows if value_index is not None and len(row) > value_index
             and _number(row[value_index])),
            key=lambda row: row[value_index],
            reverse=True,
        )[:3]
        summary = {
            **deepcopy(metadata),
            "comparison_axis": metadata.get("comparison_axis"),
            "comparison_tissue": metadata.get("comparison_tissue"),
            "comparison_tissue_id": metadata.get("comparison_tissue_id"),
            "status": metadata.get("status"),
            "tau": metadata.get("tau"),
            "tau_peak_biosamples": metadata.get("tau_peak_biosamples"),
            "tau_peak_matches_context": metadata.get("tau_peak_matches_context"),
            "tau_scale": metadata.get("tau_scale"),
            "log1p_mean": metadata.get("log1p_mean"),
            "log1p_sd": metadata.get("log1p_sd"),
            "unit": metadata.get("unit"),
            "data_release": metadata.get("data_release"),
            "datasource": metadata.get("datasource"),
            "datatype": metadata.get("datatype"),
            "reference_biosamples": metadata.get("reference_biosamples"),
            "detection_rule": deepcopy(metadata.get("detection_rule")),
            "reference_panel_summary": {
                "n": len(rows),
                "measured_count": len(numeric),
                "missing_count": len(rows) - len(numeric),
                "nonzero_count": sum(value > 0 for value in numeric),
                "maximum": max(numeric) if numeric else None,
                "mean": sum(numeric) / len(numeric) if numeric else None,
                "median": median(numeric) if numeric else None,
                "top_expression": {"columns": columns, "rows": top_rows},
            },
            "matched_biosamples": deepcopy(group.get("matched_biosamples", {})),
            "group_enrichment": deepcopy(group.get("group_enrichment", {})),
        }
        groups.append(summary)
    result["groups"] = groups
    return result


def _compact_colocalization_records(records):
    """Intern repeated nested objects while retaining every displayed fact.

    The index remains the citation allowlist.  Colocalization details are kept
    in a row table plus object catalogs, so a large locus does not repeat the
    same study, context, or credible-set field names hundreds of times.
    """
    nested = (
        "context", "credible_set_member", "right_study", "study", "statistics",
        "other_reported_statistics", "variant",
    )
    coloc = [row for row in records if isinstance(row, Mapping) and row.get("kind") == "colocalizations"]
    if not coloc:
        return list(records), {"format": "colocalization_record_catalog_v2", "columns": [], "rows": {}, "objects": {}}
    objects = {field: {} for field in nested}
    object_ids = {field: {} for field in nested}
    prefixes = dict(zip(nested, ("CX", "M", "RS", "ST", "N", "OS", "V"), strict=True))
    rows = {}
    row_keys = sorted({key for row in coloc for key in row if key != "context_assessment" and key not in nested})
    identities = {"evidence_id", "evidence_ref", "fact_id", "gene_id", "kind"}
    common_fields = {key: deepcopy(coloc[0][key]) for key in row_keys if key not in identities
                     and all(key in row and _json(row[key]) == _json(coloc[0].get(key)) for row in coloc)}
    row_keys = [key for key in row_keys if key not in common_fields]
    columns = [*row_keys, *(field + "_ref" for field in nested)]
    for record in coloc:
        row = []
        for key in row_keys:
            row.append(deepcopy(record.get(key)))
        for field in nested:
            value = record.get(field)
            signature = _json(value)
            reference = object_ids[field].get(signature)
            if reference is None:
                reference = f"{prefixes[field]}{len(objects[field]) + 1}"
                object_ids[field][signature] = reference
                objects[field][reference] = deepcopy(value)
            row.append(reference)
        rows[f"CR{len(rows) + 1}"] = row
    compact_records = []
    coloc_index = 0
    for record in records:
        if not isinstance(record, Mapping) or record.get("kind") != "colocalizations":
            compact_records.append(record)
            continue
        compact_records.append({
            "evidence_id": record.get("evidence_id"),
            "evidence_ref": record.get("evidence_ref"),
            "fact_id": record.get("fact_id"),
            "gene_id": record.get("gene_id"),
            "kind": record.get("kind"),
            "record_ref": f"CR{coloc_index + 1}",
        })
        coloc_index += 1
    return compact_records, {
        "format": "colocalization_record_catalog_v2",
        "columns": columns,
        "common_fields": common_fields,
        "rows": rows,
        "objects": objects,
        "omitted_from_model_rows": ["context_assessment"],
        "omitted_field_semantics": (
            "The verbose ontology-mapping trace is retained in the complete scientific facts; "
            "context and molecular_context_assessment summaries remain on the card."
        ),
    }


def _compact_expression_profile_catalog(profiles):
    """Share biosample identities across gene-specific expression profiles."""
    dynamic_columns = {
        "reference_panel": {"expression_value"},
        "matched_biosamples": {
            "expression_value", "detection_pass", "background_max", "background_mean",
            "fold_enrichment", "zero_background", "tissue_enriched_fold_pass",
            "tissue_enhanced_fold_pass", "z", "z_above_support_threshold", "above_threshold",
        },
        "group_enrichment": {
            "status", "group_mean", "outside_max", "all_members_detected", "fold_enrichment",
            "zero_background", "qualified",
        },
    }

    def split_table(table, dynamic):
        columns = list(table.get("columns", [])) if isinstance(table, Mapping) else []
        rows = table.get("rows", []) if isinstance(table, Mapping) else []
        dynamic_names = [column for column in columns if column in dynamic]
        static_names = [column for column in columns if column not in dynamic]
        static_index = [columns.index(column) for column in static_names]
        dynamic_index = [columns.index(column) for column in dynamic_names]
        return (
            {"columns": static_names, "rows": [[row[index] for index in static_index] for row in rows]},
            {"columns": dynamic_names, "rows": [[row[index] for index in dynamic_index] for row in rows]},
        )

    shared_groups = {}
    group_ids = {}
    normalized_profiles = {}
    for profile_id, profile in profiles.items():
        normalized = {key: value for key, value in profile.items() if key != "groups"}
        normalized["groups"] = []
        for group in profile.get("groups", []):
            shared = {
                "reference_panel": split_table(group.get("reference_panel", {}), dynamic_columns["reference_panel"])[0],
                "matched_biosamples": split_table(group.get("matched_biosamples", {}), dynamic_columns["matched_biosamples"])[0],
                "group_enrichment": split_table(group.get("group_enrichment", {}), dynamic_columns["group_enrichment"])[0],
            }
            # Stable metadata identifies the group, while the full metadata is
            # retained in the gene-specific profile for exact reconstruction.
            shared["stable_metadata"] = {
                key: value for key, value in group.get("metadata", {}).items()
                if key not in {"log1p_mean", "log1p_sd", "tau", "tau_peak_biosamples", "tau_peak_matches_context"}
            }
            signature = _hash(shared)
            group_ref = group_ids.setdefault(signature, f"G{len(group_ids) + 1}")
            shared_groups.setdefault(group_ref, shared)
            reference_static, reference_dynamic = split_table(
                group.get("reference_panel", {}), dynamic_columns["reference_panel"]
            )
            matched_static, matched_dynamic = split_table(
                group.get("matched_biosamples", {}), dynamic_columns["matched_biosamples"]
            )
            enrichment_static, enrichment_dynamic = split_table(
                group.get("group_enrichment", {}), dynamic_columns["group_enrichment"]
            )
            normalized["groups"].append({
                "group_ref": group_ref,
                "metadata": deepcopy(group.get("metadata", {})),
                "reference_panel_values": reference_dynamic,
                "matched_biosamples_values": matched_dynamic,
                "group_enrichment_values": enrichment_dynamic,
                "shared_table_integrity": {
                    "reference_panel": reference_static == shared["reference_panel"],
                    "matched_biosamples": matched_static == shared["matched_biosamples"],
                    "group_enrichment": enrichment_static == shared["group_enrichment"],
                },
            })
        normalized_profiles[profile_id] = normalized
    return {
        "format": "expression_profile_catalog_v2",
        "profiles": normalized_profiles,
        "shared_groups": shared_groups,
        "encoding": (
            "Each profile group joins its group_ref shared identity tables with the "
            "gene-specific dynamic value tables by row order."
        ),
    }


def expand_expression_profile_catalog(catalog, profile_ref):
    """Reconstruct historical profiles or current shared scientific results."""
    if isinstance(catalog, Mapping) and catalog.get("format") == "expression_profile_catalog_v4":
        from .ranking_payload_codec import expand_expression_results

        return expand_expression_results(catalog, profile_ref)
    if not isinstance(catalog, Mapping) or catalog.get("format") != "expression_profile_catalog_v2":
        return (catalog.get("profiles", {}).get(profile_ref) if isinstance(catalog, Mapping) else None)
    profile = deepcopy(catalog.get("profiles", {}).get(profile_ref))
    if not isinstance(profile, Mapping):
        return None
    groups = []
    for compact_group in profile.get("groups", []):
        shared = catalog.get("shared_groups", {}).get(compact_group.get("group_ref"), {})
        group = {"metadata": deepcopy(compact_group.get("metadata", {}))}
        for name in ("reference_panel", "matched_biosamples", "group_enrichment"):
            static = shared.get(name, {"columns": [], "rows": []})
            dynamic = compact_group.get(name + "_values", {"columns": [], "rows": []})
            columns = [*static.get("columns", []), *dynamic.get("columns", [])]
            rows = [
                [*static_row, *dynamic_row]
                for static_row, dynamic_row in zip(static.get("rows", []), dynamic.get("rows", []), strict=True)
            ]
            group[name] = {"columns": columns, "rows": rows}
        groups.append(group)
    profile["groups"] = groups
    return profile


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _canonical_table_order(value):
    if isinstance(value, Mapping):
        return {key: sorted((_canonical_table_order(row) for row in item), key=_json)
                if key == "rows" and isinstance(item, list) else _canonical_table_order(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_table_order(item) for item in value]
    return value


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _range(values):
    values = list(values)
    known = [value for value in values if _number(value)]
    return {"reported_count": len(known), "missing_count": len(values) - len(known),
            "minimum": min(known) if known else None, "maximum": max(known) if known else None}


def _matching(text, terms):
    text = re.sub(r"[\s_-]+", " ", str(text).casefold())
    return [term for term in terms if re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"s?(?!\w)", text)]


def _coding(row):
    from .candidates import coding_terms

    for field in ("subject", "object"):
        entity = row.get(field) or {}
        if entity.get("type") in {"target", "gene"} and entity.get("id") != row.get("gene_id"):
            return False

    def normalize(value):
        if isinstance(value, Mapping):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value.replace("SO_", "SO:", 1) if isinstance(value, str) and value.startswith("SO_") else value

    return bool(coding_terms({"object": {"id": row.get("gene_id"), "type": "target"},
        "context": normalize(row.get("context", {})), "measurements": normalize(row.get("statistics", {})),
        "upstream": {"raw": normalize(row.get("annotation", {}))}}))


def _flagged(row):
    return (row.get("supports") is False or row.get("provenance_complete") is False
            or bool(row.get("quality_controls"))
            or bool((row.get("regulatory_annotation") or {}).get("qualityControls")))


def _verified_model_source_aliases(query):
    """Rebind display names only when the exact source correction is proved."""
    from .analysis_scope import validate_analysis_scope
    from .source_normalization import validate_source_normalization

    plan = query.ld_expansion if isinstance(query.ld_expansion, Mapping) else {}
    aliases = plan.get("resolved_source_aliases", {})
    if not isinstance(aliases, Mapping):
        return {}
    sources = {row.variant_id for row in query.variants}
    original = {row.get("variant_id") for row in plan.get("source_observations", []) if isinstance(row, Mapping)}
    proofs = {row["original_id"]: row for row in plan.get("source_identity_normalizations", [])
              if validate_source_normalization(row)}
    scope_valid = None
    result = {}
    for raw, canonical in sorted(aliases.items()):
        if raw not in original or canonical not in sources or raw == canonical:
            continue
        proof = proofs.get(raw)
        if proof and proof["canonical_id"] == canonical:
            result[raw] = {"original_id": raw, "canonical_id": canonical,
                           "basis": "validated_ot_ref_alt_normalization", "proof_sha256": proof["proof_sha256"]}
        elif isinstance(raw, str) and re.fullmatch(r"rs[1-9][0-9]*", raw, re.IGNORECASE):
            if scope_valid is None:
                try:
                    validate_analysis_scope(query)
                    scope_valid = True
                except (TypeError, ValueError, KeyError):
                    scope_valid = False
            if scope_valid:
                result[raw] = {"original_id": raw, "canonical_id": canonical,
                               "basis": "validated_rsid_analysis_scope"}
    return result


@dataclass(frozen=True)
class RankingCardView:
    payload: dict
    evidence_aliases: dict[str, str]
    aggregation_receipt: dict

    @property
    def evidence_ids(self):
        return tuple(sorted(set(self.evidence_aliases.values())))


def build_ranking_cards(query, view: ScientificFactView, *, retrieval_statuses=None) -> RankingCardView:
    """Build stable cards and exact citation bindings without changing input facts."""
    names = ("sources", "studies", "variants", "ld_relations", "credible_set_members", "molecular_qtl",
             "colocalizations", "gene_annotations", "regulatory_evidence", "regulatory_variant_links", "support_traces")
    tables = {name: view.table(name) if name in view.payload else [] for name in names}
    lookup = {name: {row["id"]: row for row in tables[name]} for name in
              ("sources", "studies", "variants", "ld_relations", "credible_set_members", "support_traces")}
    source_ids = {row.variant_id for row in query.variants}
    model_source_aliases = _verified_model_source_aliases(query)
    applied_source_aliases = set()
    terms = sorted({str(value).strip() for value in (query.trait, *query.relevant_tissues)
                    if isinstance(value, str) and len(value.strip()) >= 3})
    candidates = sorted(set(view.payload.get("candidate_genes", query.candidate_genes)))
    from .molecular_context import freeze_molecular_context, molecular_context_match
    biological_context = query.biological_context or {"relevant_tissues": list(query.relevant_tissues),
        "source": "explicit query context", "defined": bool(query.relevant_tissues)}
    frozen_context = freeze_molecular_context(biological_context)
    retrieval_statuses = dict(retrieval_statuses or {})
    citations, aliases, receipt_groups = {}, {}, []
    bindings = {row["evidence_id"]: row for row in view.record_bindings}
    bindings_by_fact = defaultdict(list)
    for binding in view.record_bindings:
        bindings_by_fact[binding.get("fact_id")].append(binding)
    available_aliases = dict(view.evidence_aliases)
    alias_by_evidence = {identifier: alias for alias, identifier in available_aliases.items()}
    next_alias = max((int(alias[1:]) for alias in available_aliases if re.fullmatch(r"E[0-9]+", alias)), default=0) + 1
    links = defaultdict(list)
    for row in tables["regulatory_variant_links"]:
        links[row["regulatory_ref"]].append(row)

    def variant_detail(reference):
        row = lookup["variants"].get(reference)
        if row is None:
            return None
        result = {key: deepcopy(value) for key, value in row.items() if key not in {"id", "ld_relation_ids"}}
        result["origin"] = "source_selected" if row["variant_id"] in source_ids else row.get("origin", "unreported")
        if result["origin"] != "source_selected":
            result.pop("source_gwas_p_value", None)
        result["ld_relations"] = [deepcopy(lookup["ld_relations"][key]) for key in row.get("ld_relation_ids", [])
                                  if key in lookup["ld_relations"]]
        for relation in result["ld_relations"]:
            original = relation.get("selected_variant_id")
            if original in model_source_aliases:
                relation["selected_variant_id"] = model_source_aliases[original]["canonical_id"]
                applied_source_aliases.add(original)
        return result

    def coloc_member_refs(row):
        """Retain actual member/query-variant pairs, never their Cartesian product."""
        nonlocal next_alias
        trace = lookup["support_traces"].get(row.get("support_trace_ref"))
        if trace is None:
            return None
        row_refs = set(row.get("evidence_refs", []))
        chosen = {}
        for binding in sorted(bindings_by_fact[row["id"]], key=lambda item: (
                alias_by_evidence.get(item["evidence_id"]) not in row_refs,
                item["evidence_id"] not in alias_by_evidence, item["evidence_id"])):
            pair = (binding.get("variant_ref"), binding.get("member_id"))
            chosen.setdefault(pair, binding)
        if ({pair[0] for pair in chosen if pair[0]} != set(trace.get("variant_refs", []))
                or {pair[1] for pair in chosen if pair[1]} != set(trace.get("member_refs", []))):
            raise ValueError("Colocalization member trace does not match its original evidence bindings")
        refs = set(row_refs)
        for pair, binding in sorted(chosen.items(), key=lambda item: _json(item[0])):
            if pair[0] and pair[0] not in lookup["variants"]:
                raise ValueError("Colocalization query-variant binding is unavailable")
            if pair[1] and pair[1] not in lookup["credible_set_members"]:
                raise ValueError("Colocalization member measurement is unavailable")
            identifier = binding["evidence_id"]
            alias = alias_by_evidence.get(identifier)
            if alias is None:
                alias = f"E{next_alias}"
                next_alias += 1
                available_aliases[alias] = identifier
                alias_by_evidence[identifier] = alias
            refs.add(alias)
        return sorted(refs)

    def cite(row, kind, *, link=None, background=None, member_refs=None, background_assessment=None):
        refs = member_refs if member_refs is not None else (link or row).get("evidence_refs", [])
        visible = []
        for alias in sorted(refs):
            identifier = available_aliases.get(alias)
            if identifier is None:
                raise ValueError("Ranking card citation has no exact evidence binding")
            binding = bindings.get(identifier, {})
            if binding.get("fact_id", row["id"]) != row["id"]:
                raise ValueError("Ranking card citation belongs to a different scientific fact")
            aliases[alias] = identifier
            visible.append(alias)
            if alias not in citations:
                result = {"evidence_ref": alias, "evidence_id": identifier, "fact_id": row["id"],
                          "gene_id": row.get("gene_id"), "kind": kind,
                          "source_ref": row.get("source_id"),
                          "study": deepcopy(lookup["studies"].get(row.get("study_ref"), {}))}
                if background is not None and kind == "expression":
                    # A profile's joined tissue list describes all returned
                    # measurements. Selected measurements carry their exact
                    # tissue/cell identity; do not imply that list is their scope.
                    result["study"] = {key: value for key, value in result["study"].items()
                                       if key not in {"tissue", "biosample_name", "biosample_id", "biosample_ref"}}
                if background is None:
                    for key in ("statistics", "other_reported_statistics", "quality_controls", "supports",
                                "provenance_complete", "confidence", "qtl_type", "finemapping_method", "method",
                                "regulatory_annotation", "annotation", "context", "query_variant_role"):
                        if key in row:
                            result[key] = deepcopy(row[key])
                    if row.get("member_ref"):
                        result["credible_set_member"] = deepcopy(lookup["credible_set_members"].get(row["member_ref"], {}))
                    elif member_refs is not None and binding.get("member_id"):
                        member = lookup["credible_set_members"][binding["member_id"]]
                        # The shared member table's first evidence_ref may
                        # belong to a QTL fact. Cite this coloc record's own
                        # binding while retaining the member's exact study,
                        # variant and measurements.
                        result["credible_set_member"] = deepcopy({key: value for key, value in member.items()
                                                                  if key != "evidence_ref"})
                    if row.get("right_study_ref"):
                        result["right_study"] = deepcopy(lookup["studies"].get(row["right_study_ref"], {}))
                    reference = (link or row).get("variant_ref") or binding.get("variant_ref")
                    detail = variant_detail(reference)
                    if detail is not None:
                        result["variant"] = detail
                    if link:
                        result["variant_overlap"] = deepcopy(link.get("variant_specific", {}))
                result["context_assessment"] = ({"scope": "selected_background_items", "item_assessments": []}
                    if background is not None and kind == "expression" else context_assessment(row))
                for field, empty in (("statistics", {}), ("quality_controls", []), ("supports", None),
                                     ("confidence", None), ("provenance_complete", None), ("context", {})):
                    result.setdefault(field, deepcopy(empty))
                result["unreported_fields"] = [field for field in
                    ("statistics", "quality_controls", "supports", "confidence", "provenance_complete", "context")
                    if result[field] is None or result[field] == {} or result[field] == []]
                citations[alias] = result
            if background is not None:
                observations = citations[alias].setdefault("selected_background", [])
                if background not in observations:
                    observations.append(deepcopy(background))
                    if kind == "expression":
                        citations[alias]["context_assessment"]["item_assessments"].append({
                            "selected_background_index": len(observations),
                            "assessment": deepcopy(background_assessment)})
        return {"fact_id": row["id"], "evidence_refs": visible}

    def membership(kind, gene, rows, selected):
        fact_ids = sorted(row["id"] for row in rows)
        fact_set = set(fact_ids)
        member_ids = sorted({binding["evidence_id"] for binding in view.record_bindings if binding.get("fact_id") in fact_set})
        receipt = {"kind": kind, "gene_id": gene, "fact_count": len(fact_ids), "fact_ids": fact_ids,
                   "member_evidence_count": len(member_ids), "member_evidence_ids": member_ids,
                   "membership_sha256": _hash(member_ids), "facts_sha256": _hash(sorted(rows, key=lambda row: row["id"])),
                   "representative_evidence_refs": sorted(set(selected))}
        receipt_groups.append(receipt)
        return {key: receipt[key] for key in ("fact_count", "member_evidence_count", "membership_sha256")}

    def origin(link):
        row = lookup["variants"].get(link.get("variant_ref"), {})
        if row.get("variant_id") in source_ids:
            return "source_selected"
        return "ld_proxy" if row.get("origin") == "ld_proxy" else "other_variant_context"

    def link_key(link):
        row = lookup["variants"].get(link.get("variant_ref"), {})
        relation = [lookup["ld_relations"].get(key, {}) for key in row.get("ld_relation_ids", [])]
        strengths = [item.get("r2") for item in relation if _number(item.get("r2"))]
        return (-max(strengths, default=-1), row.get("variant_id", ""), link["id"])

    def context_assessment(row):
        study = lookup["studies"].get(row.get("study_ref"), {})
        context = dict(row.get("context") or {})
        raw = {**study, **(row.get("regulatory_annotation") or {})}
        if study.get("biosample_id"):
            raw.setdefault("biosampleId", study["biosample_id"])
        if study.get("biosample_name"):
            raw.setdefault("biosampleName", study["biosample_name"])
        if study.get("tissue"):
            raw.setdefault("tissue", study["tissue"])
        return molecular_context_match({"context": context, "upstream": {"raw": raw}}, frozen_context)

    def context_of(row):
        study = lookup["studies"].get(row.get("study_ref"), {})
        labels = [str(study[key]) for key in ("tissue", "biosample_name", "biosample_id") if study.get(key)]
        text = "; ".join(dict.fromkeys(labels))
        assessed = context_assessment(row)
        classification = assessed.get("context_class") or (
            "matched" if assessed.get("matched") else "other_reported_context" if text else "context_unreported")
        return classification, text

    def availability(modules, rows, visible_count):
        statuses = {module: retrieval_statuses.get(module, "unknown") for module in modules}
        values = [value.get("status", "unknown") if isinstance(value, Mapping) else value for value in statuses.values()]
        state = "completed" if values and all(value in {"success", "no_data"} for value in values) else (
            "not_requested" if values and all(value == "not_requested" for value in values) else
            "incomplete" if any(value not in {"success", "no_data", "unknown", "not_requested"} for value in values) else "unknown")
        return {"query_status": state, "module_statuses": statuses, "retained_fact_count": len(rows),
            "displayed_item_count": visible_count, "evidence_status": "available" if rows else "no_retained_evidence",
            "missing_semantics": "An empty retained set is not a negative biological result; query coverage and evidence filtering remain distinct."}

    cards = []
    for gene in candidates:
        annotations = [row for row in tables["gene_annotations"] if row.get("gene_id") == gene]
        card = {"gene_id": gene, "gene_symbol": None, "direct_molecular": {},
                "expression_profiles": None, "expression_profile_contract": None,
                "regulatory_groups": [], "positional_summary": {},
                "availability": {}, "background_summary": {"role": "context_only",
                "selection_basis": "frozen biological context; stable measurement identity tie-break"}}
        for family in ("molecular_qtl", "colocalizations"):
            rows = sorted((row for row in tables[family] if row.get("gene_id") == gene), key=lambda row: row["id"])
            retained = []
            for row in rows:
                value = cite(
                    row,
                    family,
                    member_refs=coloc_member_refs(row) if family == "colocalizations" else None,
                )
                if family == "colocalizations":
                    context = row.get("context") if isinstance(row.get("context"), Mapping) else {}
                    value.update(
                        phenotype_evidence_scope=context.get(
                            "phenotype_evidence_scope",
                            context.get("target_trait_relation", "unresolved"),
                        ),
                        target_trait_relation=context.get("target_trait_relation"),
                        molecular_evidence_tier=context.get("molecular_evidence_tier"),
                        source_phenotype=(
                            lookup["studies"].get(row.get("right_study_ref"), {}).get("trait_from_source")
                            or lookup["studies"].get(row.get("right_study_ref"), {}).get("traitFromSource")
                            or lookup["studies"].get(row.get("right_study_ref"), {}).get("trait")
                        ),
                    )
                retained.append(value)
            card["direct_molecular"][family] = retained
            card["availability"][family] = availability(("locus.qtl",) if family == "molecular_qtl" else ("locus.colocalisation",), rows, len(retained))
            membership(family, gene, rows, [ref for row in retained for ref in row["evidence_refs"]])
        coding = [row for row in annotations if row.get("kind") == "variant_annotation" and _coding(row)]
        card["direct_molecular"]["coding_splicing"] = [cite(row, "variant_annotation") for row in sorted(coding, key=lambda row: row["id"])]
        card["availability"]["coding_splicing"] = availability(("locus.variant_consequence",), coding, len(coding))
        membership("coding_splicing", gene, coding, [ref for row in coding for ref in row.get("evidence_refs", [])])

        regulatory = [row for row in tables["regulatory_evidence"] if row.get("gene_id") == gene]
        grouped = defaultdict(list)
        for row in regulatory:
            annotation = row.get("regulatory_annotation", {})
            context_class, _ = context_of(row)
            key = (row.get("source_id"), annotation.get("datasourceId"), annotation.get("method") or row.get("method"),
                   annotation.get("model"), annotation.get("modelVersion"), context_class, annotation.get("intervalType"))
            grouped[_json(key)].append(row)
        small_limit = parameter("ranking.cards.regulatory_keep_all_max_relationships")
        group_limit = parameter("ranking.cards.regulatory_representatives_per_group")
        gene_limit = parameter("ranking.cards.regulatory_representatives_per_gene")
        all_relationships = [(row, link) for row in regulatory for link in (links[row["id"]] or [None])]
        keep_all = len(all_relationships) <= small_limit

        def pair_origin(pair):
            return origin(pair[1]) if pair[1] else "unlinked_regulatory_context"

        def pair_priority(pair, *, comparable):
            row, link = pair
            confidence = row.get("confidence")
            score = row.get("statistics", {}).get("score")
            # Numeric priorities apply only inside a declared identical method group.
            return ({"source_selected": 0, "ld_proxy": 1}.get(pair_origin(pair), 2),
                    -confidence if comparable and _number(confidence) else 0,
                    -score if comparable and _number(score) else 0,
                    link_key(link) if link else (0, "", ""), row["id"])

        prepared = []
        for key, rows in sorted(grouped.items()):
            source, dataset, method, model, version, context_class, interval_type = json.loads(key)
            comparable = bool(source and dataset and (method or model and version))
            pairs = [(row, link) for row in rows for link in (links[row["id"]] or [None])]
            pairs.sort(key=lambda pair: pair_priority(pair, comparable=comparable))
            ordinary = [pair for pair in pairs if not _flagged(pair[0])]
            flagged = [pair for pair in pairs if _flagged(pair[0])]
            prepared.append({"key": key, "rows": rows, "pairs": pairs,
                "ordinary": ordinary, "flagged": flagged, "selected": [], "comparable": comparable,
                "context_class": context_class})
        if keep_all:
            for item in prepared:
                item["selected"] = list(item["pairs"])
        else:
            # Round-robin declared groups. Never compare scores across methods.
            prioritized = sorted(prepared, key=lambda item: (
                {"matched": 0, "other_reported_context": 1, "context_unreported": 2}.get(item["context_class"], 2),
                item["key"]))
            remaining = gene_limit
            for offset in range(group_limit):
                for item in prioritized:
                    if remaining and offset < len(item["ordinary"]):
                        item["selected"].append(item["ordinary"][offset])
                        remaining -= 1
            for item in prepared:
                item["selected"].extend(item["flagged"])

        shown_regulatory = 0
        for item in prepared:
            key, rows = item["key"], item["rows"]
            source, dataset, method, model, version, context_class, interval_type = json.loads(key)
            group = {"source_id": source, "dataset": dataset, "method": method, "model": model,
                     "model_version": version, "context_match_class": context_class,
                     "regulatory_element_type": interval_type,
                     "display_mode": "all_relationships" if keep_all else "representatives",
                     "representative_selection": "exact source/proxy links; frozen context; explicit within-method confidence; stable identity",
                     "score_comparison_within_group_supported": item["comparable"],
                     "relationship_count": sum(len(links[row["id"]]) for row in rows),
                     "other_variant_relationship_count": sum(origin(link) == "other_variant_context"
                         for row in rows for link in links[row["id"]]),
                     "independent_experiment_count": None,
                     "score_range": _range(row.get("statistics", {}).get("score") for row in rows),
                     "distance_to_tss_range": _range(row.get("other_reported_statistics", {}).get("distanceToTss") for row in rows),
                     "context_distribution": dict(sorted(Counter(context_of(row)[1] or "unreported" for row in rows).items())),
                     "interval_type_counts": dict(sorted(Counter(row.get("regulatory_annotation", {}).get("intervalType", "unreported") for row in rows).items())),
                     "unique_interval_count": len({row.get("interval_ref") for row in rows}),
                     "selected_relationships": [], "representatives": {}, "qualified_or_conflicting_evidence": [],
                     "selection_limits": {"small_set_all_relationships": small_limit,
                         "ordinary_per_group": group_limit, "ordinary_per_gene": gene_limit,
                         "flagged_evidence_exempt_from_limits": True}}
            chosen = []
            for scope in ("source_selected", "ld_proxy"):
                group[scope + "_relationship_count"] = sum(pair_origin(pair) == scope for pair in item["pairs"])
            for row, link in item["selected"]:
                value = {**cite(row, "regulatory", link=link),
                         "relationship_scope": pair_origin((row, link)),
                         "interval_ref": row.get("interval_ref"),
                         "variant_ref": link.get("variant_ref") if link else None}
                group["selected_relationships"].append(value)
                if value["relationship_scope"] in {"source_selected", "ld_proxy"}:
                    group["representatives"].setdefault(value["relationship_scope"], value)
                if _flagged(row):
                    group["qualified_or_conflicting_evidence"].append(value)
                chosen.extend(value["evidence_refs"])
            group["displayed_relationship_count"] = len(group["selected_relationships"])
            group["summarized_relationship_count"] = len(item["pairs"]) - len(item["selected"])
            group["flagged_relationship_count"] = len(item["flagged"])
            group["budget_exceeded_to_preserve_flags"] = not keep_all and bool(item["flagged"]) and (
                len(item["selected"]) > group_limit or sum(len(value["selected"]) for value in prepared) > gene_limit)
            shown_regulatory += len(item["selected"])
            group.update(membership("regulatory:" + key, gene, rows, chosen))
            card["regulatory_groups"].append(group)
        card["availability"]["regulatory"] = availability(("locus.enhancer_to_gene",), regulatory, shown_regulatory)

        positional = [row for row in annotations if row.get("kind") == "variant_annotation"]
        for scope in ("source_selected", "ld_proxy"):
            rows = [row for row in positional if origin(row) == scope]
            measured = [row for row in rows if _number(row.get("statistics", {}).get("distance_from_footprint"))]
            entry = {"record_count": len(rows), "distance_from_footprint_range": _range(
                row.get("statistics", {}).get("distance_from_footprint") for row in rows)}
            if measured:
                nearest = min(measured, key=lambda row: (abs(row["statistics"]["distance_from_footprint"]), row["id"]))
                entry["nearest"] = cite(nearest, "variant_annotation")
            card["positional_summary"][scope] = entry
        card["positional_summary"]["explicit_coding_splicing_fact_count"] = len(coding)
        card["positional_summary"]["role"] = "distance is positional context, not proof of a causal gene"
        membership("positional_context", gene, positional, [ref for entry in card["positional_summary"].values()
                   if isinstance(entry, dict) for ref in entry.get("nearest", {}).get("evidence_refs", [])])

        for kind in ("function", "expression", "pathway", "literature"):
            rows = sorted((row for row in annotations if row.get("kind") == kind), key=lambda row: row["id"])
            items = []
            for row in rows:
                annotation = row.get("annotation", {})
                symbol = annotation.get("approvedSymbol") or row.get("context", {}).get("approved_symbol")
                if symbol and not card["gene_symbol"]:
                    card["gene_symbol"] = symbol
                if (
                    kind == "expression"
                    and isinstance(annotation.get("expression_profiles"), Mapping)
                    and card["expression_profiles"] is None
                ):
                    card["expression_profiles"] = _compact_expression_profiles(annotation["expression_profiles"])
                    card["expression_profile_contract"] = annotation.get(
                        "expression_profile_contract", "expression-profile-v1"
                    )
                if kind == "function":
                    descriptions = annotation.get("functionDescriptions", [])
                    if descriptions:
                        sentence = re.split(r"(?<=[.!?])\s+", str(descriptions[0]), maxsplit=1)[0]
                        items.append((row, {"function_description": sentence, "approved_name": annotation.get("approvedName")}, [], None))
                else:
                    values = (annotation.get("tissues", []) if kind == "expression" else
                              [*annotation.get("geneOntology", []), *annotation.get("pathways", [])] if kind == "pathway" else
                              [{**annotation, "publication": lookup["studies"].get(row.get("study_ref"), {})}])
                    for item in values:
                        assessment = None
                        if kind == "expression":
                            raw = {key: value for key, value in item.items()
                                   if key in {"tissue", "tissue_name", "tissueId", "tissueLabel",
                                              "cell_type", "cellType", "cellLabel", "cellTypeId"}}
                            # A measurement can report both an organ and a cell
                            # identity. Preserve those separate axes rather than
                            # replacing one generic biosample ID with the other.
                            for field, identity_key, label_key in (
                                    ("tissueBiosample", "tissueId", "tissue"),
                                    ("celltypeBiosample", "cellTypeId", "cell_type")):
                                sample = item.get(field)
                                if isinstance(sample, Mapping):
                                    if sample.get("biosampleId"):
                                        raw[identity_key] = sample["biosampleId"]
                                    if sample.get("biosampleName"):
                                        raw[label_key] = sample["biosampleName"]
                            assessment = molecular_context_match({"upstream": {"raw": raw}}, frozen_context)
                            matches = ((assessment.get("observed") or
                                [entry["label"] for entry in assessment.get("observed_biosamples", [])])
                                if assessment.get("matched") else [])
                        else:
                            matches = _matching(_json(item), terms)
                        if matches:
                            items.append((row, item, matches, assessment))
            items.sort(key=lambda item: (-len(item[2]), _json(item[1]), item[0]["id"]))
            limit = parameter("ranking.cards." + kind + "_items")
            chosen, seen = [], set()
            for row, item, matches, assessment in items:
                signature = _json(item)
                if signature in seen:
                    continue
                seen.add(signature)
                if len(chosen) < limit:
                    selected_item = {key: value for key, value in item.items() if key != "publication"} if kind == "literature" else item
                    value = cite(row, kind, background=selected_item, background_assessment=assessment)
                    chosen.append({**value, "matched_context_terms": matches,
                                   **({"context_assessment": deepcopy(assessment)} if assessment is not None else {})})
            summary = {"reported_record_count": len(rows), "context_matching_item_count": len(items),
                       "selected_item_count": len(chosen), "selected": chosen,
                       "selection_limit": limit, "unselected_content_retained_in_ledger": True}
            summary["context_selection_basis"] = ("frozen biosample identity and context mapping" if kind == "expression"
                else "text relevance only; not a biosample identity match" if kind in {"pathway", "literature"}
                else "reported function description; not molecular causal evidence")
            if kind == "expression":
                summary["reported_measurement_count"] = sum(len(row.get("annotation", {}).get("tissues", [])) for row in rows)
            card["background_summary"][kind] = summary
            modules = {"function": ("target.core",), "expression": ("target.expression",),
                       "pathway": ("target.pathway", "target.go"), "literature": ("evidence.europepmc",)}[kind]
            card["availability"][kind] = availability(modules, rows, len(chosen))
            membership("background:" + kind, gene, rows, [ref for item in chosen for ref in item["evidence_refs"]])
        card["coverage"] = {"candidate_annotations": len(annotations), "regulatory_facts": len(regulatory),
                            "direct_facts_summarized_away": 0, "background_is_causal_evidence": False}
        cards.append(card)

    receipt = {"format": "ranking_card_aggregation_receipt_v1", "complete_scientific_view_sha256": _hash(_canonical_table_order(view.payload)),
               "groups": receipt_groups, "visible_evidence_ids": sorted(set(aliases.values())),
               "model_ld_source_alias_normalizations": [model_source_aliases[key] for key in sorted(applied_source_aliases)],
               "visible_evidence_ids_sha256": _hash(sorted(set(aliases.values())))}
    payload = {"format": "ranking_gene_cards_v2", "comparison_cards": cards,
               "citation_records": [citations[key] for key in sorted(citations)],
               "sources": {key: deepcopy(lookup["sources"][key]) for key in sorted(
                   {row["source_ref"] for row in citations.values() if row.get("source_ref") in lookup["sources"]})},
               "context_terms": terms,
               "coverage": {"candidate_gene_count": len(candidates), "visible_citation_record_count": len(citations),
                            "full_scientific_fact_count": view.payload.get("coverage", {}).get("scientific_fact_count"),
                            "all_candidate_genes_retained": True, "all_direct_molecular_facts_retained": True,
                            "regulatory_and_background_tables_summarized": any(
                                group["summarized_relationship_count"] for card in cards for group in card["regulatory_groups"])
                                or any(section["reported_record_count"] > section["selected_item_count"]
                                    for card in cards for section in card["background_summary"].values()
                                    if isinstance(section, Mapping) and "reported_record_count" in section),
                            "aggregation_receipt_sha256": _hash(receipt)},
               "interpretation_rules": ["Candidate ID order is not a priority order.",
                    "Correlated proxy links and overlapping regulatory assays are not independent support votes.",
                    "Small regulatory sets retain all relationships. Large sets disclose per-method representatives and retain flagged evidence outside ordinary budgets; scores from different methods are not comparable.",
                    "Reported regulatory scores are not calibrated causal probabilities, and group record counts are not independent experiment counts.",
                    "Promoter overlap, expression, pathway and literature context do not establish gene causality.",
                    "No colocalization record means unavailable evidence, not evidence against colocalization."]}
    return RankingCardView(payload, dict(sorted(aliases.items())), receipt)
