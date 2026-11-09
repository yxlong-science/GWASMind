"""One frozen Open Targets phenotype identity scope shared by V2G channels."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from copy import deepcopy

from .audit import sha256_json

PHENOTYPE_SCOPE_VERSION = "ot-phenotype-scope-v1"
MAX_RELATED_PHENOTYPES = 8
MAX_PHENOTYPE_CANDIDATES = 150
EFO_RELATIONS = ("descendant", "sibling", "sibling_descendant")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]{0,30}_[A-Za-z0-9]{1,100}")


class PhenotypeScopeError(ValueError):
    """A bounded phenotype scope failed identity or frozen-contract validation."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f"Phenotype scope could not be resolved ({reason}).")


def normalized_phenotype_label(value):
    return " ".join(str(value or "").casefold().split())


def _identity(row):
    if not isinstance(row, Mapping) or not isinstance(row.get("id"), str) or not _IDENTIFIER.fullmatch(row["id"]):
        raise PhenotypeScopeError("invalid_phenotype_identifier")
    if not isinstance(row.get("name"), str) or not row["name"].strip() or len(row["name"]) > 500:
        raise PhenotypeScopeError("invalid_phenotype_name")
    return {"id": row["id"], "name": row["name"]}


def freeze_phenotype_scope(phenotype, *, target=None, related=(), candidates=(), source,
                           status="resolved", issues=(), target_reason=""):
    manifest = sorted((deepcopy(dict(row)) for row in candidates), key=lambda row: row["id"])
    result = {"version": PHENOTYPE_SCOPE_VERSION, "phenotype": phenotype,
        "target": None if target is None else _identity(target),
        "related": [deepcopy(dict(row)) for row in related], "status": status, "source": source,
        "target_reason": target_reason, "candidate_manifest": manifest,
        "selection_manifest_sha256": sha256_json(manifest), "issues": list(issues)}
    result["scope_sha256"] = sha256_json(result)
    return validate_phenotype_scope(result, phenotype=phenotype)


def validate_phenotype_scope(value, *, phenotype):
    if not isinstance(value, Mapping):
        raise PhenotypeScopeError("scope_must_be_an_object")
    allowed = {"version", "phenotype", "target", "related", "status", "source", "target_reason",
               "candidate_manifest", "selection_manifest_sha256", "issues", "scope_sha256"}
    if set(value) != allowed or value.get("version") != PHENOTYPE_SCOPE_VERSION:
        raise PhenotypeScopeError("unsupported_scope_contract")
    if normalized_phenotype_label(value.get("phenotype")) != normalized_phenotype_label(phenotype):
        raise PhenotypeScopeError("scope_phenotype_mismatch")
    if value.get("status") not in {"resolved", "expansion_not_assessed", "target_unresolved"}:
        raise PhenotypeScopeError("invalid_scope_status")
    for key in ("source", "target_reason"):
        if not isinstance(value.get(key), str) or len(value[key]) > 500:
            raise PhenotypeScopeError("invalid_scope_provenance")
    issues = value.get("issues")
    if not isinstance(issues, list) or len(issues) > 20 or any(not isinstance(item, str) or len(item) > 200 for item in issues):
        raise PhenotypeScopeError("invalid_scope_issues")
    rows = value.get("candidate_manifest")
    if not isinstance(rows, list) or len(rows) > MAX_PHENOTYPE_CANDIDATES:
        raise PhenotypeScopeError("invalid_candidate_manifest")
    manifest = {}
    for row in rows:
        identity = _identity(row)
        if set(row) != {"id", "name", "exact_synonyms"} or identity["id"] in manifest:
            raise PhenotypeScopeError("ambiguous_candidate_identity")
        synonyms = row.get("exact_synonyms")
        if not isinstance(synonyms, list) or len(synonyms) > 100 or any(
            not isinstance(term, str) or len(term) > 500 for term in synonyms
        ):
            raise PhenotypeScopeError("invalid_exact_synonyms")
        manifest[identity["id"]] = identity
    if value["selection_manifest_sha256"] != sha256_json(rows):
        raise PhenotypeScopeError("candidate_manifest_hash_mismatch")
    related = value.get("related")
    if not isinstance(related, list) or len(related) > MAX_RELATED_PHENOTYPES:
        raise PhenotypeScopeError("invalid_related_phenotypes")
    target = value.get("target")
    selected = set()
    if target is not None:
        identity = _identity(target)
        if set(target) != {"id", "name"} or manifest.get(identity["id"]) != identity:
            raise PhenotypeScopeError("target_outside_verified_manifest")
        selected.add(identity["id"])
    for row in related:
        identity = _identity(row)
        if set(row) != {"id", "name", "reason"} or manifest.get(identity["id"]) != identity:
            raise PhenotypeScopeError("related_phenotype_outside_verified_manifest")
        if identity["id"] in selected:
            raise PhenotypeScopeError("duplicate_phenotype_identity")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip() or len(row["reason"]) > 500:
            raise PhenotypeScopeError("related_phenotype_reason_required")
        selected.add(identity["id"])
    if (target is None) != (value["status"] == "target_unresolved") or (target is None and related):
        raise PhenotypeScopeError("target_status_mismatch")
    if value["status"] == "expansion_not_assessed" and related:
        raise PhenotypeScopeError("unassessed_expansion_has_related_phenotypes")
    if value.get("scope_sha256") != sha256_json({key: item for key, item in value.items() if key != "scope_sha256"}):
        raise PhenotypeScopeError("scope_hash_mismatch")
    return deepcopy(dict(value))


def rebind_phenotype_scope(value, phenotype):
    result = validate_phenotype_scope(value, phenotype=phenotype)
    result["phenotype"] = phenotype
    result["scope_sha256"] = sha256_json({key: item for key, item in result.items() if key != "scope_sha256"})
    return result


def scope_ids(scope):
    if not isinstance(scope, Mapping):
        return ()
    value = validate_phenotype_scope(scope, phenotype=scope.get("phenotype", ""))
    return tuple([value["target"]["id"]] if value["target"] else []) + tuple(row["id"] for row in value["related"])


def model_biological_context(context):
    """Expose selected identities to inference; keep the full selection manifest in audit data."""
    if not isinstance(context, Mapping):
        return context
    result = deepcopy(dict(context))
    scope = result.get("phenotype_scope")
    if scope is not None:
        value = validate_phenotype_scope(scope, phenotype=result.get("phenotype", ""))
        result["phenotype_scope"] = {key: value[key] for key in (
            "version", "phenotype", "target", "related", "status", "source", "target_reason", "issues", "scope_sha256")}
    return result


def classify_phenotype_ids(observed_ids, scope):
    if not isinstance(scope, Mapping):
        return None
    value = validate_phenotype_scope(scope, phenotype=scope.get("phenotype", ""))
    observed = {item for item in observed_ids if isinstance(item, str)}
    if value["target"] and value["target"]["id"] in observed:
        return "target_phenotype"
    if observed.intersection(row["id"] for row in value["related"]):
        return "related_phenotype"
    return None


def seeded_target_id(term):
    from importlib.resources import files
    seeds = json.loads(files("gwasmind").joinpath("data/phenotype_context_seeds.json").read_text(encoding="utf-8"))
    aliases = {normalized_phenotype_label(key): value for key, value in seeds.get("aliases", {}).items()}
    label = normalized_phenotype_label(term)
    canonical = normalized_phenotype_label(aliases.get(label, term))
    return next((identity for name, identity in seeds.get("target_ids", {}).items()
                 if normalized_phenotype_label(name) == canonical), None)


async def phenotype_search(executor, term, *, expand=True):
    query = ('query V2GTraitSearch($q:String!){ search(queryString:$q, entityNames:["disease"], '
             'page:{index:0,size:10}){ hits{ id name } } }')
    primary = term.strip()[:180]
    terms = [primary, primary + " measurement", primary + " levels"] if expand else [primary]
    # A parenthetical qualifier such as "(females)" or "(cystatin C)" can push the measured
    # entity out of the search results; the full label still decides any exact match.
    base = re.sub(r"\s*\([^()]*\)", "", primary).strip()
    if expand and base and base != primary:
        terms += [base, base + " measurement"]
    terms = list(dict.fromkeys(terms))
    identifiers = [seeded_target_id(term)] if seeded_target_id(term) else []
    failures = []
    for search in terms:
        try:
            data = await executor(query, {"q": search})
            identifiers.extend(row["id"] for row in (data.get("search") or {}).get("hits", ())
                if isinstance(row, Mapping) and isinstance(row.get("id"), str) and _IDENTIFIER.fullmatch(row["id"]))
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve alternate verified identity lookups
            failures.append(exc)
    if not identifiers and len(failures) == len(terms):
        raise failures[0]
    return tuple(dict.fromkeys(identifiers))[:MAX_PHENOTYPE_CANDIDATES]


async def phenotype_identities(executor, identifiers):
    identifiers = tuple(dict.fromkeys(identifiers))
    if len(identifiers) > MAX_PHENOTYPE_CANDIDATES or any(
        not isinstance(item, str) or not _IDENTIFIER.fullmatch(item) for item in identifiers
    ):
        raise PhenotypeScopeError("identity_query_out_of_bounds")
    found = {}
    for offset in range(0, len(identifiers), 50):
        batch = identifiers[offset:offset + 50]
        fields = "id name synonyms{ relation terms }"
        query = "query V2GScopeIdentities{ " + " ".join(
            f'd{index}: disease(efoId:{json.dumps(identity)}){{ {fields} }}'
            for index, identity in enumerate(batch)) + " }"
        data = await executor(query, {})
        for index, identity in enumerate(batch):
            row = data.get(f"d{index}")
            if not isinstance(row, Mapping) or row.get("id") != identity:
                continue
            normalized = _identity(row)
            synonyms = sorted({term for item in row.get("synonyms") or [] if isinstance(item, Mapping)
                and item.get("relation") == "hasExactSynonym" for term in item.get("terms") or []
                if isinstance(term, str) and term.strip() and len(term) <= 500})[:100]
            found[identity] = {**normalized, "exact_synonyms": synonyms}
    return found


def _identifier_list(values):
    return [item for item in values or () if isinstance(item, str) and _IDENTIFIER.fullmatch(item)]


async def efo_neighbourhood(executor, target_id):
    """Related-phenotype candidates read from the ontology graph, never from model proposals.

    Candidates are the target's descendants, the siblings under its most specific parent (the
    parent with the fewest children) and those siblings' descendants, in that order.
    """
    if not isinstance(target_id, str) or not _IDENTIFIER.fullmatch(target_id):
        raise PhenotypeScopeError("invalid_phenotype_identifier")
    query = ("query V2GEfoNeighbourhood($id:String!){ meta{ dataVersion{ year month } } "
             "disease(efoId:$id){ id descendants parents{ id children{ id } } } }")
    data = await executor(query, {"id": target_id})
    disease = data.get("disease") if isinstance(data, Mapping) else None
    if not isinstance(disease, Mapping) or disease.get("id") != target_id:
        raise PhenotypeScopeError("efo_structure_unavailable")
    version = (data.get("meta") or {}).get("dataVersion") or {}
    release = ".".join(str(version[key]) for key in ("year", "month") if version.get(key)) or "unknown"
    parents = [row for row in disease.get("parents") or () if isinstance(row, Mapping)
               and isinstance(row.get("id"), str) and _IDENTIFIER.fullmatch(row["id"])]
    descendants = _identifier_list(disease.get("descendants"))
    siblings = []
    if parents:
        parent = min(parents, key=lambda row: (len(row.get("children") or ()), row["id"]))
        siblings = _identifier_list(child.get("id") for child in parent.get("children") or ()
                                    if isinstance(child, Mapping))
    sibling_descendants = []
    if siblings:
        rows = await executor("query V2GEfoSiblingDescendants($ids:[String!]!){ diseases(efoIds:$ids)"
                              "{ id descendants } }", {"ids": siblings})
        by_id = {row.get("id"): row for row in (rows.get("diseases") or ()) if isinstance(row, Mapping)}
        sibling_descendants = [item for sibling in siblings
                               for item in _identifier_list((by_id.get(sibling) or {}).get("descendants"))]
    ordered = {}
    for relation, values in zip(EFO_RELATIONS, (descendants, siblings, sibling_descendants)):
        for identity in values:
            if identity != target_id:
                ordered.setdefault(identity, relation)
    return release, list(ordered.items())


def exact_target(phenotype, candidates):
    candidates = list(candidates)
    seeded = seeded_target_id(phenotype)
    if seeded:
        return next((row for row in candidates if row["id"] == seeded), None)
    wanted = normalized_phenotype_label(phenotype)
    exact = [row for row in candidates if wanted in {
        normalized_phenotype_label(item) for item in (row["id"], row["name"], *row["exact_synonyms"])}]
    return exact[0] if len(exact) == 1 else None


async def target_only_phenotype_scope(executor, phenotype, *, target_ids: Sequence[str] = (),
                                     reason="model_expansion_not_requested"):
    if len(tuple(dict.fromkeys(target_ids))) > 1:
        raise PhenotypeScopeError("multiple_trait_ids_require_grouped_phenotype_scope")
    try:
        identifiers = tuple(target_ids) or await phenotype_search(executor, phenotype)
        candidates = await phenotype_identities(executor, identifiers)
    except CancelledError:
        raise
    except Exception:  # noqa: BLE001 - identity unavailability must remain explicit
        return freeze_phenotype_scope(phenotype, source="Open Targets phenotype identity unavailable",
            status="target_unresolved", issues=("phenotype_identity_lookup_unavailable",))
    target = candidates.get(target_ids[0]) if target_ids else exact_target(phenotype, candidates.values())
    return freeze_phenotype_scope(phenotype, target=target, candidates=candidates.values(),
        source="Verified target identity; related phenotype expansion not assessed",
        status="expansion_not_assessed" if target else "target_unresolved", issues=(reason,))


async def verify_phenotype_scope_identities(executor, scope):
    value = validate_phenotype_scope(scope, phenotype=scope["phenotype"])
    identifiers = scope_ids(value)
    if not identifiers:
        return value
    try:
        observed = await phenotype_identities(executor, identifiers)
    except CancelledError:
        raise
    except Exception as exc:
        raise PhenotypeScopeError("frozen_identity_verification_unavailable") from exc
    selected = [value["target"], *value["related"]]
    if any(row["id"] not in observed or observed[row["id"]]["name"] != row["name"] for row in selected):
        raise PhenotypeScopeError("frozen_identity_verification_failed")
    return value
