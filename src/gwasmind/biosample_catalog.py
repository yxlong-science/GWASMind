"""Offline, provenance-bound Open Targets biosample identity and hierarchy lookup."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files

from .configuration import parameter

# The OT snapshot contains is_a ancestry but omits anatomical part_of edges.
# These bounded assertions connect common provider compartments to their organs;
# no cell-to-organ inference or lexical expansion is allowed.
_ANATOMICAL_SOURCE = {
    "source": "Uberon", "source_version": "releases/2026-06-19",
    "source_url": "https://raw.githubusercontent.com/obophenotype/uberon/master/uberon.obo",
    "source_sha256": "7f06d8e8442008a67132a1599b652e86fe0c52d75c8d6bc5b0cc36a0031e6b3f",
}
_ANATOMICAL_RELATIONS = tuple({"subject_id": subject, "relation": "part_of", "object_id": target,
    **_ANATOMICAL_SOURCE, "source_assertion": f"relationship: part_of {target.replace('_', ':')} ! {label}"}
    for subject, target, label in (
        ("UBERON_0001113", "UBERON_0002107", "liver"),
        ("UBERON_0004151", "UBERON_0000948", "heart"),
        ("UBERON_0006618", "UBERON_0002081", "cardiac atrium"),
        ("UBERON_0001225", "UBERON_0002113", "kidney"),
        ("UBERON_0000006", "UBERON_0000016", "endocrine pancreas"),
        ("UBERON_0000016", "UBERON_0001264", "pancreas"),
        ("UBERON_0000014", "UBERON_0002097", "skin of body"),
        ("UBERON_0002116", "UBERON_0002108", "small intestine"),
        ("UBERON_0001621", "UBERON_0000948", "heart"),
    ))


def _anatomical_path(catalog, observed, selected, allowed):
    if not observed.startswith("UBERON_") or not selected.startswith("UBERON_"):
        return None
    edges = [row for row in allowed if row in _ANATOMICAL_RELATIONS
             and row["subject_id"] in catalog.rows and row["object_id"] in catalog.rows]
    pending, visited = [(observed, [])], set()
    while pending:
        node, path = pending.pop(0)
        if node in visited:
            continue
        visited.add(node)
        ancestry = {node, *catalog.rows.get(node, {}).get("ancestors", ())}
        if path and selected in ancestry:
            return path
        for row in edges:
            if row["subject_id"] in ancestry and row["object_id"] not in visited:
                pending.append((row["object_id"], [*path, row]))
    return None


def normalized_label(value):
    """Normalize orthography only; biological aliases come from the source catalog."""
    text = " ".join(str(value).casefold().replace("-", " ").split())
    return re.sub(r"\bcells$", "cell", text)


def normalized_id(value):
    text = str(value).strip()
    match = re.search(r"(?:^|/)([A-Za-z]+)[_:]([0-9]+)$", text)
    return f"{match[1].upper()}_{match[2]}" if match else text


# Fields that may differ between equivalent catalog contents: the content hash and
# the provenance path of the Parquet source. Rows and all other metadata are identical.
_EQUIVALENCE_FIELDS = ("equivalent_catalog_sha256", "equivalence_note")
_PROVENANCE_ONLY_FIELDS = ("catalog_sha256", "source_path")


class BiosampleCatalog:
    def __init__(self, payload, *, content_sha256):
        self.metadata = {key: deepcopy(value) for key, value in payload.items()
                         if key != "rows" and key not in _EQUIVALENCE_FIELDS}
        self.metadata["catalog_sha256"] = content_sha256
        # Earlier content with identical rows (only provenance fields differed) stays
        # a valid identity, so mappings saved with it are not reported as mismatched.
        self._identities = frozenset((content_sha256, *payload.get("equivalent_catalog_sha256", ())))
        self.rows = {}
        self.labels, self.synonyms = defaultdict(set), defaultdict(set)
        for row in payload["rows"]:
            value = dict(zip(payload["columns"], row))
            identity = value["id"]
            if identity in self.rows:
                raise ValueError("Biosample catalog contains duplicate identities")
            self.rows[identity] = value
            self.labels[normalized_label(value["label"])].add(identity)
            for synonym in value.get("synonyms", ()):
                self.synonyms[normalized_label(synonym)].add(identity)

    def recorded_with(self, catalog_sha256):
        """Whether a mapping recorded with this catalog hash used these exact rows."""
        return catalog_sha256 in self._identities

    def recorded_metadata(self, recorded):
        """The catalog metadata a frozen mapping must carry to validate.

        A mapping recorded with an equivalent earlier content keeps its recorded
        metadata, provided it differs from the current metadata only in the
        provenance-only fields; anything else is still a mismatch.
        """
        if (isinstance(recorded, Mapping) and recorded.get("catalog_sha256") != self.metadata["catalog_sha256"]
                and self.recorded_with(recorded.get("catalog_sha256"))):
            def comparable(value):
                return {key: item for key, item in value.items() if key not in _PROVENANCE_ONLY_FIELDS}
            if comparable(recorded) == comparable(self.metadata):
                return deepcopy(dict(recorded))
        return deepcopy(self.metadata)

    def axis(self, identity):
        row = self.rows.get(identity, {})
        ancestors = set(row.get("ancestors", ())) | {identity}
        if "CL_0000000" in ancestors or identity.startswith("CL_"):
            return "cell_types"
        if "UBERON_0000062" in ancestors:
            return "organs"
        if identity.startswith("UBERON_"):
            return "tissues"
        return None

    def resolve(self, value, *, axis=None):
        raw_identity = str(value).strip()
        identity = raw_identity if raw_identity in self.rows else normalized_id(value)
        if identity in self.rows:
            found, basis = {identity}, "catalog_id"
        elif re.search(r"(?:^|/)[A-Za-z]+[_:][0-9]+$", str(value)):
            found, basis = set(), "unrecognized_id"
        else:
            key = normalized_label(value)
            found, basis = set(self.labels.get(key, ())), "catalog_label"
            if not found:
                found, basis = set(self.synonyms.get(key, ())), "catalog_synonym"
        if axis and len(found) > 1:
            scoped = {item for item in found if self.axis(item) == axis}
            if scoped:
                found = scoped
        status = "resolved" if len(found) == 1 else "ambiguous" if found else "unresolved"
        matches = [{"id": item, "label": self.rows[item]["label"]} for item in sorted(found)]
        return {"input": str(value), "status": status, "basis": basis, "matches": matches}

    def relation(self, observed_id, selected_id, *, allow_descendants=True):
        if observed_id == selected_id:
            return "exact_biosample_id"
        if (allow_descendants and selected_id in
                self.rows.get(observed_id, {}).get("ancestors", ())):
            return "catalog_ancestor"
        return None

    def search(self, value, *, limit=8):
        """Return bounded vocabulary candidates, never infer disease relevance.

        Exact labels, source synonyms and IDs take precedence. Otherwise all
        query tokens must occur in one source label/synonym; no semantic or
        fuzzy replacement is used. Native catalog axes are authoritative.
        """
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Biosample search limit must be between 1 and 20")
        resolved = self.resolve(value)
        if resolved["basis"] == "unrecognized_id":
            return []
        if resolved["matches"]:
            identities = [(0, row["id"], resolved["basis"]) for row in resolved["matches"]]
        else:
            tokens = set(re.findall(r"[^\W_]+", normalized_label(value)))
            if not tokens - {"cell", "tissue", "organ", "human", "of", "the", "and", "in"}:
                return []
            identities = []
            for identity, row in self.rows.items():
                best = None
                for label in (row["label"], *row.get("synonyms", ())):
                    observed = set(re.findall(r"[^\W_]+", normalized_label(label)))
                    if tokens.issubset(observed):
                        extra = len(observed - tokens)
                        best = extra if best is None else min(best, extra)
                if best is not None:
                    identities.append((best, identity, "catalog_token_match"))
        rows = []
        for score, identity, basis in identities:
            axis = self.axis(identity)
            if axis is not None:
                row = self.rows[identity]
                rows.append((score, row["label"].casefold(), identity, {
                    "id": identity, "label": row["label"], "axis": axis,
                    "match_basis": basis,
                }))
        return [row[-1] for row in sorted(rows, key=lambda row: row[:3])[:limit]]


@lru_cache(maxsize=1)
def biosample_catalog():
    resource = parameter("context.biosample_catalog_resource")
    if (not isinstance(resource, str) or resource.startswith("/") or "\\" in resource or
            ":" in resource or ".." in resource.split("/")):
        raise ValueError("Biosample catalog must be a packaged relative resource")
    content = gzip.decompress(files("gwasmind").joinpath(resource).read_bytes())
    payload = json.loads(content)
    if payload.get("format") != "opentargets_biosample_catalog_v1":
        raise ValueError("Unsupported biosample catalog format")
    return BiosampleCatalog(payload, content_sha256=hashlib.sha256(content).hexdigest())


def map_context_concepts(axes):
    """Freeze mappings of declared concepts; do not infer phenotype biology."""
    catalog = biosample_catalog()
    limit = parameter("context.max_concepts_per_level")
    concepts = []
    for axis in ("organs", "tissues", "cell_types"):
        values = tuple(dict.fromkeys(axes.get(axis, ())))
        if len(values) > limit:
            raise ValueError(f"Biological context {axis} exceeds the configured concept limit")
        for value in values:
            resolved = catalog.resolve(value, axis=axis)
            if resolved["status"] == "resolved":
                native_axis = catalog.axis(resolved["matches"][0]["id"])
                if native_axis is not None and native_axis != axis:
                    resolved = {**resolved, "status": "axis_mismatch", "catalog_axis": native_axis}
            concepts.append({"axis": axis, **resolved})
    # Persisted v1 plans retain their exact historical serialization and policy.
    existing = axes.get("biosample_mapping") or {}
    legacy = existing.get("version") == "ot-biosample-context-v1"
    result = {"version": "ot-biosample-context-v1" if legacy else "ot-biosample-context-v2",
            "catalog": catalog.recorded_metadata(existing.get("catalog")),
            "concepts": concepts, "status": ("unresolved" if not concepts and not legacy else
                "resolved" if all(row["status"] == "resolved" for row in concepts) else "partial"),
            "relation_policy": "exact_or_reported_catalog_ancestor",
            "allow_catalog_descendants": parameter("context.allow_catalog_descendants"),
            "organ_inference": "none"}
    if not legacy:
        result["anatomical_relations"] = deepcopy(list(_ANATOMICAL_RELATIONS))
        result["relation_policy"] = "exact_or_catalog_ancestor_or_declared_anatomical_part"
    return result


def _observed_fields(record):
    upstream = record.get("upstream") or {}
    raw = upstream.get("raw", {}) if isinstance(upstream, Mapping) else {}
    sources = [raw, record.get("context", {})]
    entries = []
    fields = {
        "biosample": None, "biosampleName": None, "biosample_name": None,
        "biosampleId": None, "biosample_id": None,
        "tissue": "tissues", "tissue_name": "tissues", "tissueName": "tissues",
        "tissueId": "tissues", "tissue_id": "tissues", "tissueLabel": "tissues",
        "cell_type": "cell_types", "cellType": "cell_types", "cell_type_name": "cell_types",
        "cellTypeId": "cell_types", "cell_type_id": "cell_types", "cellLabel": "cell_types",
    }
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key, axis in fields.items():
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                entries.append((key, axis, value.strip()))
            elif key == "biosample" and isinstance(value, Mapping):
                for nested in ("biosampleId", "biosampleName", "id", "label", "name"):
                    if isinstance(value.get(nested), str) and value[nested].strip():
                        entries.append(("biosample." + nested, None, value[nested].strip()))
    return list(dict.fromkeys(entries))


def match_catalog_context(record, mapping):
    """Resolve observed identity before comparing frozen concepts; unknown is not other."""
    catalog = biosample_catalog()
    fields = _observed_fields(record)
    result = {"matched": False, "observed": sorted({row[2] for row in fields}),
              "observed_biosamples": [], "mapping_status": "unresolved", "match_basis": [],
              "context_class": "context_unreported", "status": "biosample_unavailable",
              "catalog_sha256": catalog.metadata["catalog_sha256"]}
    if not catalog.recorded_with(mapping.get("catalog", {}).get("catalog_sha256")):
        return {**result, "mapping_status": "catalog_mismatch"}
    resolutions = [{"field": key, "axis": axis, **catalog.resolve(value, axis=axis)}
                   for key, axis, value in fields]
    result["observed_mappings"] = resolutions
    identities = {match["id"] for row in resolutions if row["status"] == "resolved" for match in row["matches"]}
    result["observed_biosamples"] = [{"id": item, "label": catalog.rows[item]["label"]} for item in sorted(identities)]
    # A biosample ID paired with a contradictory label must not be silently reinterpreted.
    general = [row for row in resolutions if row["axis"] is None]
    general_ids = {match["id"] for row in general if row["status"] == "resolved" for match in row["matches"]}
    if len(general_ids) > 1:
        return {**result, "mapping_status": "conflicting"}
    unresolved_ids = any(row["basis"] == "unrecognized_id" for row in resolutions)
    explicit_ids = {match["id"] for row in resolutions if row["basis"] == "catalog_id"
                    for match in row["matches"]}
    unresolved = [row for row in resolutions if row["status"] != "resolved"]
    # A verified ID disambiguates its own optional label; an unknown ID never does.
    uncertain = [row for row in unresolved if not (len(explicit_ids) == 1 and
        (row["status"] == "unresolved" and row["basis"] != "unrecognized_id" or
         row["status"] == "ambiguous" and explicit_ids.issubset({m["id"] for m in row["matches"]})))]
    if unresolved_ids or uncertain:
        return {**result, "mapping_status": "ambiguous" if any(row["status"] == "ambiguous" for row in resolutions) else "unresolved"}
    if not identities:
        return result
    result["mapping_status"] = "resolved"
    concepts = mapping.get("concepts", ())
    for selected in concepts:
        if selected.get("status") != "resolved":
            continue
        for target in selected.get("matches", ()):
            for observed in sorted(identities):
                relation = catalog.relation(observed, target["id"],
                    allow_descendants=mapping.get("allow_catalog_descendants") is True)
                relation_path = None
                if relation is None and mapping.get("allow_catalog_descendants") is True:
                    relation_path = _anatomical_path(catalog, observed, target["id"],
                        mapping.get("anatomical_relations", ()))
                    if relation_path:
                        relation = "declared_anatomical_part"
                if relation:
                    binding = {"observed_id": observed, "selected_id": target["id"],
                        "selected_axis": selected["axis"], "relation": relation}
                    if relation_path:
                        binding["relation_provenance"] = deepcopy(_ANATOMICAL_SOURCE)
                        binding["relation_path"] = deepcopy(relation_path)
                    result["match_basis"].append(binding)
    if result["match_basis"]:
        result.update(matched=True, status="matched", context_class="matched")
    elif all(row.get("status") == "resolved" for row in concepts):
        result.update(status="not_matched", context_class="other_reported_context")
    else:
        result["mapping_status"] = "context_mapping_incomplete"
    return result
