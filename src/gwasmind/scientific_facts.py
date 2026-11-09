"""Natural scientific tables sharing study, member and provenance relationships.

Different measured values remain distinct rows; repeated lookups are provenance,
not additional votes. Full record bindings stay outside the model prompt.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .models import EvidenceKind, EvidenceRecord, LocusQuery


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identifier_set_summary(values: Sequence[str]) -> dict[str, Any]:
    ordered = sorted(set(values))
    return {"count": len(ordered), "sha256": hashlib.sha256(_json(ordered).encode()).hexdigest()}


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != {} and value != [] and value != ()


def _clean(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(item) for key, item in value.items() if _present(item)}


class _Table:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.rows: list[dict[str, Any]] = []
        self.keys: dict[str, str] = {}

    def add(self, values: Mapping[str, Any]) -> str:
        clean = _clean(values)
        key = _json(clean)
        if key not in self.keys:
            identifier = f"{self.prefix}{len(self.rows) + 1}"
            self.keys[key] = identifier
            self.rows.append({"id": identifier, **clean})
        return self.keys[key]


class _StudyTable(_Table):
    def coalesce_complementary_metadata(self) -> None:
        """Merge a known identity only when its entire metadata bucket agrees.

        Considering the whole bucket before assigning references avoids letting
        an initially missing field bridge two later conflicting observations.
        Unidentified studies retain their exact-content identity.
        """
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in self.rows:
            values = {key: value for key, value in row.items() if key != "id"}
            identity_field = "credible_set_id" if values.get("credible_set_id") else "study_id"
            known = all(values.get(key) for key in ("source", "data_release", identity_field))
            identity = _json([values["source"], values["data_release"], identity_field, values[identity_field]]) if known else _json(values)
            buckets.setdefault(identity, []).append(values)
        replacements = {}
        for rows in buckets.values():
            merged: dict[str, Any] = {}
            conflict = False
            for row in rows:
                for key, value in row.items():
                    if key in merged and (type(merged[key]) is not type(value) or merged[key] != value):
                        conflict = True
                    merged[key] = value
            for row in rows:
                replacements[_json(row)] = row if conflict else merged
        table = _Table(self.prefix)
        remapped_keys = {key: table.add(replacements[key]) for key in self.keys}
        self.rows = table.rows
        self.keys = {**table.keys, **remapped_keys}


def _study_descriptions(record: EvidenceRecord) -> tuple[dict[str, Any], dict[str, Any] | None]:
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    upstream = payload.get("upstream")
    raw = upstream.get("raw") if isinstance(upstream, Mapping) else {}
    raw = raw if isinstance(raw, Mapping) else {}

    def value(key: str) -> Any:
        result = raw[key] if raw.get(key) is not None else payload.get(key)
        return result if isinstance(result, (str, int, float, bool)) else None

    source = {"source": record.source, "data_release": record.data_release}
    left = {**source, "study_id": value("studyId") or record.study_id,
        "credible_set_id": value("leftStudyLocusId") or value("studyLocusId") or record.credible_set_id,
        # A regulatory assay studies an interval/sample. Its target gene belongs
        # to the regulatory annotation, not to the identity of the assay.
        "gene_id": record.gene_id if record.kind is not EvidenceKind.REGULATORY else None,
        "study_type": value("studyType"), "tissue": record.tissue,
        "biosample_id": value("biosampleId"), "biosample_name": value("biosampleName"),
        "biosample_from_source_id": value("biosampleFromSourceId"),
        "trait": value("traitFromSource"), "lead_variant_id": value("leadVariantId"),
        "pmid": value("pubmedId") or record.pmid,
        **{key: value(key) for key in ("publicationDate", "publicationFirstAuthor", "publicationJournal", "publicationTitle")}}
    right = None
    if record.kind is EvidenceKind.COLOCALIZATION:
        right = {**source, "study_id": value("rightStudyId"), "credible_set_id": value("rightStudyLocusId"),
            "study_type": value("rightStudyType"), "trait": value("rightTraitFromSource"),
            "gene_id": value("rightQtlGeneId"), "pmid": value("rightPubmedId"),
            **{key.removeprefix("right"): value(key) for key in (
                "rightPublicationDate", "rightPublicationFirstAuthor", "rightPublicationJournal", "rightPublicationTitle")}}
    return left, right


@dataclass(frozen=True)
class ScientificFactView:
    payload: dict[str, Any]
    evidence_aliases: dict[str, str]
    evidence_ids: tuple[str, ...]
    record_bindings: tuple[dict[str, str], ...]

    def table(self, name: str) -> list[dict[str, Any]]:
        from .scientific_reuse import restored_scientific_table

        table = restored_scientific_table(self.payload, name)
        rows = []
        for values in table["rows"]:
            flat = {**table["constant_fields"], **{key: item for key, item in zip(table["columns"], values, strict=True) if item is not None}}
            record: dict[str, Any] = {}
            resource_names = flat.pop("resource_measurement_names", None)
            resource_values = {key.removeprefix("resource_measurement."): flat.pop(key)
                               for key in list(flat) if key.startswith("resource_measurement.")}
            for key, value in flat.items():
                if key.startswith(("statistics.", "other_reported_statistics.", "regulatory_annotation.", "variant_specific.")):
                    family, field = key.split(".", 1)
                    record.setdefault(family, {})[field] = value
                else:
                    record[key] = value
            if resource_names is not None:
                record.setdefault("statistics", {})["resource_scores"] = [
                    {"name": resource, "value": resource_values.get(resource)} for resource in resource_names]
            rows.append(record)
        # Return a conventional joined scientific view to callers. The model
        # receives the explicit named tables and references in ``payload``.
        joins = {
            "studies": (("biosample_ref", "biosamples"), ("publication_ref", "publications")),
            "regulatory_evidence": (("interval_ref", "regulatory_intervals"),),
            "regulatory_variant_links": (("overlap_metrics_ref", "variant_overlap_metrics"),),
        }
        for reference, target in joins.get(name, ()):
            if target not in self.payload:
                continue  # Original v1 artifacts remain readable.
            targets = {row["id"]: row for row in self.table(target)}
            for row in rows:
                if reference not in row:
                    continue
                for key, value in targets[row[reference]].items():
                    if key in {"id", "source_id"}:
                        continue
                    if isinstance(value, Mapping) and isinstance(row.get(key), Mapping):
                        row[key] = {**value, **row[key]}
                    else:
                        row.setdefault(key, value)
        return rows


def _scientific_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Conventional named columns; constants apply equally to every table row."""
    flat_rows = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if key in {"statistics", "other_reported_statistics", "regulatory_annotation", "variant_specific"}:
                for field, item in value.items():
                    # Named assay measurements are ordinary numeric columns,
                    # rather than a repeated list of {name, value} objects.
                    if key == "statistics" and field == "resource_scores" and _named_resource_scores(item):
                        flat["resource_measurement_names"] = [entry["name"] for entry in item]
                        flat.update({f"resource_measurement.{entry['name']}": entry["value"] for entry in item})
                    else:
                        flat[f"{key}.{field}"] = item
            else:
                flat[key] = value
        flat_rows.append(flat)
    # Upstream field sets may iterate differently across Python hash seeds.
    # Fix presentation order without changing row order or scientific values.
    columns = sorted({key for row in flat_rows for key in row}, key=lambda key: (key != "id", key))
    constants = {key: flat_rows[0][key] for key in columns if flat_rows and key in flat_rows[0]
        and all(key in row and type(row[key]) is type(flat_rows[0][key]) and row[key] == flat_rows[0][key] for row in flat_rows)}
    columns = [key for key in columns if key not in constants]
    return {"columns": columns, "constant_fields": constants,
            "rows": [[row.get(key) for key in columns] for row in flat_rows]}


def _named_resource_scores(value: Any) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(entry, Mapping) and set(entry) == {"name", "value"}
                    and isinstance(entry["name"], str) and entry["name"] for entry in value)
            and len({entry["name"] for entry in value}) == len(value))


def _share_scientific_context(payload: dict[str, Any]) -> None:
    """Factor exact repeated sample, publication, interval and overlap values.

    Every reference remains on its original study/variant relationship. Shared
    metadata is neither a new experiment nor an additional independent vote.
    No missing value bridges conflicting metadata: tables use exact identities.
    """
    biosamples, publications, intervals, overlaps = (_Table(prefix) for prefix in ("B", "P", "I", "O"))
    for row in payload["studies"]:
        sample = {key: row.pop(key) for key in ("tissue", "biosample_id", "biosample_name", "biosample_from_source_id") if key in row}
        if sample:
            row["biosample_ref"] = biosamples.add(sample)
        publication = {key: row.pop(key) for key in list(row)
                       if key == "pmid" or key.lower().startswith("publication")}
        if publication:
            row["publication_ref"] = publications.add(publication)
    for row in payload["regulatory_evidence"]:
        interval = {"source_id": row["source_id"]}
        for family, fields in (
            ("regulatory_annotation", {"chromosome", "start", "end", "intervalType", "regulatoryElementKey", "regulatoryElementKeySource"}),
            ("other_reported_statistics", {"start", "end"}),
            ("statistics", {"start", "end"}),
        ):
            values = row.get(family, {})
            selected = {key: values.pop(key) for key in fields if key in values}
            if selected:
                interval[family] = selected
        if len(interval) > 1:
            row["interval_ref"] = intervals.add(interval)
    for row in payload["regulatory_variant_links"]:
        values = row.pop("variant_specific", {})
        if values:
            row["overlap_metrics_ref"] = overlaps.add({"variant_specific": values})
    payload.update(biosamples=biosamples.rows, publications=publications.rows,
                   regulatory_intervals=intervals.rows, variant_overlap_metrics=overlaps.rows)


def _regulatory_annotation(raw: Mapping[str, Any], *, base: Mapping[str, Any],
                           study: Mapping[str, Any]) -> dict[str, Any]:
    """Keep each reported value once, in explicit science columns/relationships.

    Only exact duplicate copies move out of the annotation envelope. Distinct
    scores, coordinates, tissue labels and quality fields remain available.
    """
    annotation = {key: value for key, value in raw.items() if key not in {
        "studyId", "studyLocusId", "qtlGeneId", "geneId", "variantId", "queryVariantId", "matchedLocusMember"}}

    def remove_same(key: str, destination: Mapping[str, Any], destination_key: str) -> None:
        if (key in annotation and destination_key in destination and _present(destination[destination_key])
                and _json(annotation[key]) == _json(destination[destination_key])):
            annotation.pop(key)

    for family in ("statistics", "other_reported_statistics"):
        values = base.get(family, {})
        for key in list(annotation):
            remove_same(key, values, key)
    remove_same("resourceScore", base.get("statistics", {}), "resource_scores")
    for key, destination_key in (("biosampleId", "biosample_id"), ("biosampleName", "biosample_name"),
                                 ("biosampleFromSourceId", "biosample_from_source_id")):
        remove_same(key, study, destination_key)
    for key, destination_key in (("coverageScope", "coverage_scope"), ("qualityControls", "quality_controls")):
        remove_same(key, base, destination_key)
    return annotation


def _regulatory_entity_and_relationship(base: Mapping[str, Any], annotation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate an exact annotation from the variant that queried its interval."""
    entity = deepcopy({key: value for key, value in base.items() if key not in {"candidate_scope", "coverage_scope"}})
    entity["regulatory_annotation"] = dict(annotation)
    variant_fields = {}
    query_fields = {"overlapVariantId", "overlapVariantRole", "overlapPosition", "overlapChromosome",
                    "queryVariantId", "queryVariantRole", "coverageScope", "exactVariantOverlap"}
    for family in ("regulatory_annotation", "other_reported_statistics", "statistics"):
        fields = entity.get(family, {})
        for key in query_fields.intersection(fields):
            # Distinct names for conflicting raw/normalized observations keep
            # both original values rather than silently preferring one copy.
            name = key if key not in variant_fields or _json(variant_fields[key]) == _json(fields[key]) else f"{family}_{key}"
            variant_fields[name] = fields.pop(key)
    return entity, {"variant_specific": variant_fields, "candidate_scope": base.get("candidate_scope"),
                    "coverage_scope": base.get("coverage_scope")}


def build_scientific_fact_view(query: LocusQuery, evidence: Sequence[EvidenceRecord], *,
                               candidate_genes: Sequence[str]) -> ScientificFactView:
    """Retain all naturally distinct facts and all actual member statistics.

    Members are keyed by study, variant and actual statistics; conflicting
    measurements therefore remain separate. H4 stays bound to its study pair.
    """
    from .projection import compact_record_for_synthesis

    candidates = set(candidate_genes)
    records = [record for record in evidence if record.gene_id in candidates]
    if len({row.evidence_id for row in records}) != len(records):
        raise ValueError("Scientific fact view requires unique evidence identifiers")
    sources, variants, members = (_Table(prefix) for prefix in ("S", "V", "M"))
    studies = _StudyTable("ST")
    for record in records:
        for description in _study_descriptions(record):
            if description is not None:
                studies.add(description)
    studies.coalesce_complementary_metadata()
    ld, qtl, coloc, regulatory, annotations = (_Table(prefix) for prefix in ("LD", "Q", "C", "R", "A"))
    regulatory_links = _Table("RL")
    regulatory_link_references: dict[str, dict[str, str]] = {}
    aliases: dict[str, str] = {}
    alias_by_evidence: dict[str, str] = {}
    references: dict[str, dict[str, str]] = {}
    fact_traces: dict[str, dict[str, set[str]]] = {}
    member_evidence: dict[str, str] = {}
    bindings = []
    selected = {row.variant_id: row.p_value for row in query.variants}

    def alias(record: EvidenceRecord) -> str:
        if record.evidence_id not in alias_by_evidence:
            name = f"E{len(aliases) + 1}"
            aliases[name] = record.evidence_id
            alias_by_evidence[record.evidence_id] = name
        return alias_by_evidence[record.evidence_id]

    for record in records:
        compact = compact_record_for_synthesis(record, relevant_tissues=query.relevant_tissues, complete_scientific_lists=True)
        raw = compact.get("raw_summary", {})
        source = sources.add({key: compact.get(key) for key in ("source", "dataset", "data_release", "module")})
        variant_id = record.variant_id
        origin = compact.get("analysis_origin") or ("source_selected" if variant_id in selected else "other_variant_context")
        variant = variants.add({"variant_id": variant_id, "origin": origin,
            **({"source_gwas_p_value": selected[variant_id]} if variant_id in selected else {})}) if variant_id else None
        links = [ld.add(link) for link in compact.get("ld_proxy_links", ())]
        if variant and links:
            row = variants.rows[int(variant[1:]) - 1]
            row["ld_relation_ids"] = list(dict.fromkeys((*row.get("ld_relation_ids", ()), *links)))
        left_values, right_values = _study_descriptions(record)
        left = studies.add(left_values)
        right = studies.add(right_values) if right_values is not None else None
        matched = deepcopy(raw.get("matchedLocusMember") or {})
        member_id = None
        if matched:
            member_variant = matched.pop("variantId", variant_id)
            member_id = members.add({"study_ref": left, "variant_id": member_variant, "statistics": matched})
            if member_id not in member_evidence:
                member_evidence[member_id] = alias(record)
        statistics = deepcopy(compact.get("measurements", {}))
        raw_statistics = {key: value for key, value in raw.items() if isinstance(value, (int, float)) and not isinstance(value, bool)
            and not (key in statistics and type(statistics[key]) is type(value) and statistics[key] == value)}
        base = {"gene_id": record.gene_id, "source_id": source, "study_ref": left,
                "statistics": statistics, "other_reported_statistics": raw_statistics,
                "supports": record.supports, "provenance_complete": record.provenance_complete,
                "candidate_scope": compact.get("candidate_scope"),
                "context": {key: deepcopy(value) for key, value in compact.get("context", {}).items()
                            if key not in {"analysis_origin", "ld_proxy_links"}},
                "confidence": raw.get("confidence"), "quality_controls": raw.get("qualityControls"),
                "coverage_scope": raw.get("coverageScope")}
        regulatory_link_id = None
        if record.kind is EvidenceKind.COLOCALIZATION:
            fact_id = coloc.add({**base, "right_study_ref": right,
                "method": raw.get("colocalisationMethod"), "query_variant_role": raw.get("queryVariantRole")})
        elif record.kind is EvidenceKind.QTL:
            fact_id = qtl.add({**base, "variant_ref": variant, "member_ref": member_id,
                "qtl_type": raw.get("studyType"), "finemapping_method": raw.get("finemappingMethod")})
        elif record.kind is EvidenceKind.REGULATORY:
            entity, relationship = _regulatory_entity_and_relationship(base,
                _regulatory_annotation(raw, base=base, study=studies.rows[int(left[2:]) - 1]))
            fact_id = regulatory.add(entity)
            regulatory_link_id = regulatory_links.add({"regulatory_ref": fact_id, "variant_ref": variant, **relationship})
            link_origins = regulatory_link_references.setdefault(regulatory_link_id, {})
            if origin not in link_origins:
                link_origins[origin] = alias(record)
        else:
            annotation = raw
            if record.kind is EvidenceKind.EXPRESSION and compact.get("expression_profiles"):
                annotation = {
                    **raw,
                    "expression_profiles": deepcopy(compact["expression_profiles"]),
                    "expression_profile_contract": compact.get(
                        "expression_profile_contract", "expression-profile-v1"
                    ),
                }
            fact_id = annotations.add({**base, "kind": record.kind.value, "variant_ref": variant,
                "annotation": annotation, "context": {key: value for key, value in compact.get("context", {}).items()
                    if key not in {"analysis_origin", "ld_proxy_links"}},
                "subject": compact.get("subject"), "object": compact.get("object"), "relation": compact.get("relation")})
        origin_refs = references.setdefault(fact_id, {})
        if origin not in origin_refs:
            origin_refs[origin] = alias(record)
        trace = fact_traces.setdefault(fact_id, {"variant_refs": set(), "member_refs": set()})
        if variant:
            trace["variant_refs"].add(variant)
        if member_id:
            trace["member_refs"].add(member_id)
        bindings.append(_clean({"evidence_id": record.evidence_id, "fact_id": fact_id,
            "member_id": member_id, "variant_ref": variant, "regulatory_link_id": regulatory_link_id}))
    traces = _Table("T")
    for table in (qtl, coloc, regulatory, annotations):
        for row in table.rows:
            fact_id = row["id"]
            row["evidence_refs"] = list(dict.fromkeys(references[fact_id].values()))
            if table is not regulatory:
                row["support_trace_ref"] = traces.add({key: sorted(value) for key, value in fact_traces[fact_id].items()})
    for row in regulatory_links.rows:
        row["evidence_refs"] = list(dict.fromkeys(regulatory_link_references[row["id"]].values()))
    for row in members.rows:
        row["evidence_ref"] = member_evidence[row["id"]]
    payload = {"format": "scientific_fact_view_v1", "candidate_genes": list(candidate_genes),
        "sources": sources.rows, "studies": studies.rows, "variants": variants.rows,
        "credible_set_members": members.rows, "molecular_qtl": qtl.rows, "colocalizations": coloc.rows,
        "regulatory_evidence": regulatory.rows, "gene_annotations": annotations.rows,
        "regulatory_variant_links": regulatory_links.rows,
        "ld_relations": ld.rows, "support_traces": traces.rows,
        "coverage": {"candidate_count": len(candidate_genes), "retained_record_count": len(evidence),
            "candidate_record_count": len(records), "scientific_fact_count": len(references),
            "regulatory_relationship_count": len(regulatory_links.rows), "regulatory_relationships_omitted": 0,
            "member_measurement_count": len(members.rows), "scientific_facts_omitted": 0,
            "member_measurements_omitted": 0}}
    _share_scientific_context(payload)
    for name in ("sources", "studies", "variants", "credible_set_members", "molecular_qtl", "colocalizations",
                 "regulatory_evidence", "regulatory_variant_links", "gene_annotations", "ld_relations", "support_traces",
                 "biosamples", "publications", "regulatory_intervals", "variant_overlap_metrics"):
        payload[name] = _scientific_table(payload[name])
    from .scientific_reuse import share_scientific_measurements

    share_scientific_measurements(payload)
    from .scientific_capacity import share_scientific_capacity

    share_scientific_capacity(payload)
    return ScientificFactView(payload, aliases, tuple(aliases.values()), tuple(bindings))
