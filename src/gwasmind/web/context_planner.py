"""One bounded, versioned phenotype-context contract for Chat and Auto."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace

from ..biological_context import (
    BiologicalContext,
    ContextResolutionError,
    biological_context_from_dict,
    curated_biological_context,
    define_biological_context,
    standardize_biological_context,
)
from ..biosample_catalog import biosample_catalog
from ..configuration import parameter
from ..prompting import load_prompt

PLANNER_VERSION = "phenotype-context-v5-nonempty-ot-discovery"
_AXES = ("organs", "tissues", "cell_types")
_CANDIDATES_PER_SEARCH_TERM = 8
MAX_CONTEXT_MODEL_CALLS = 3


def _candidate_manifest(context: BiologicalContext, *, search_terms=None) -> dict[str, object]:
    """Build a bounded selection catalog from fixed seeds or full-vocabulary searches."""
    catalog = biosample_catalog()
    identities = {}
    searches = []
    mapping = context.biosample_mapping or {}
    for concept in mapping.get("concepts", ()):
        if concept.get("status") not in {"resolved", "ambiguous"}:
            continue
        matches = concept.get("matches", ())
        for match in matches:
            identity = match.get("id")
            source = catalog.rows.get(identity)
            axis = catalog.axis(identity)
            if source is not None and axis in _AXES:
                identities[identity] = axis
    if search_terms is not None:
        for axis in _AXES:
            for term in search_terms[axis]:
                matches = catalog.search(term, limit=_CANDIDATES_PER_SEARCH_TERM)
                searches.append({"query": term, "proposed_axis": axis,
                                 "matches": matches, "status": "matched" if matches else "unresolved"})
                for match in matches:
                    identities[match["id"]] = match["axis"]
    rows = []
    for identity, axis in identities.items():
        source = catalog.rows[identity]
        rows.append({
            "axis": axis,
            "id": identity,
            "label": source["label"],
            "parents": list(source.get("parents", ())),
            "ancestors": list(source.get("ancestors", ())),
        })
    rows.sort(key=lambda row: (row["axis"], row["label"].casefold(), row["id"]))
    return {
        "version": "ot-biosample-candidate-manifest-v2",
        "catalog_sha256": catalog.metadata["catalog_sha256"],
        "catalog_release": catalog.metadata.get("release"),
        "discovery": "bounded_full_catalog_search" if search_terms is not None else "curated_context_seeds",
        "searches": searches,
        "candidates": rows,
    }


def _normalize_selection(value: dict, manifest: dict[str, object]) -> dict[str, list[str]]:
    """Convert model-selected OT IDs into canonical labels before context freezing."""
    candidates = {row["id"]: row for row in manifest.get("candidates", ())}
    selected = value.get("selected_biosample_ids")
    if selected is None:
        selected = value.get("selected_biosamples")
    if selected is None:
        raise ValueError("context planner requires selected_biosample_ids from its candidate manifest")
    if not isinstance(selected, dict):
        raise ValueError("context planner selected_biosample_ids must be an object")  # noqa: TRY004 - model schema validation
    if set(selected) != set(_AXES):
        raise ValueError("context planner selected_biosample_ids must contain all three axes")
    result = {axis: [] for axis in _AXES}
    for axis in _AXES:
        ids = selected.get(axis, [])
        if (not isinstance(ids, list) or len(ids) > parameter("context.max_concepts_per_level") or
                any(not isinstance(identity, str) for identity in ids)):
            raise ValueError(f"context planner selected {axis} must contain OT IDs")
        for identity in dict.fromkeys(ids):
            row = candidates.get(identity)
            if row is None or row["axis"] != axis:
                raise ValueError("context planner selected an OT biosample outside the candidate manifest")
            # Preserve the selected identity when a catalog label is ambiguous.
            resolved = biosample_catalog().resolve(row["label"], axis=axis)
            term = row["label"] if resolved["matches"] == [{"id": identity, "label": row["label"]}] else identity
            result[axis].append(term)
    return result


def _completion_json(completion) -> dict:
    raw = str(completion.content).strip()
    if raw.startswith("```"):
        if "\n" not in raw:
            raise ValueError("context planner returned a malformed JSON fence")
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("defined"), bool):
        raise ValueError("context planner returned an invalid schema")  # noqa: TRY004 - model schema validation
    return value


def _proposal_terms(value: dict) -> dict[str, list[str]]:
    terms = value.get("search_terms")
    if not isinstance(terms, dict) or set(terms) != set(_AXES):
        raise ValueError("context proposal search_terms must contain all three axes")
    limit = parameter("context.max_concepts_per_level")
    for axis in _AXES:
        if (not isinstance(terms[axis], list) or len(terms[axis]) > limit or
                any(not isinstance(term, str) or not term.strip() or len(term) > 120
                    for term in terms[axis])):
            raise ValueError("context proposal must contain bounded non-empty search terms")
    result = {axis: list(dict.fromkeys(term.strip() for term in terms[axis])) for axis in _AXES}
    if value["defined"] != any(result.values()):
        raise ValueError("context proposal defined contradicts its search terms")
    return result


def _detached_plan(context: BiologicalContext, phenotype: str) -> BiologicalContext:
    from ..phenotype_scope import rebind_phenotype_scope
    scope = (rebind_phenotype_scope(context.phenotype_scope, phenotype)
             if context.phenotype_scope is not None else None)
    return replace(context, phenotype=phenotype, biosample_mapping=deepcopy(context.biosample_mapping),
                   phenotype_scope=scope)


def _validated_plan(phenotype: str, value: dict, manifest: dict) -> BiologicalContext:
    terms = _normalize_selection(value, manifest)
    if value["defined"] is not True or not any(terms.values()):
        raise ContextResolutionError("empty_selection")
    context = biological_context_from_dict({**terms, "phenotype": phenotype,
        "defined": True, "rule_id": PLANNER_VERSION, "ruleset_version": PLANNER_VERSION,
        "source": "bounded LLM OT biosample selection"}, phenotype=phenotype)
    context = standardize_biological_context(context)
    if context.biosample_mapping["status"] != "resolved":
        raise ContextResolutionError("selection_identity_unresolved")
    return context


def _resolve_model_plan(phenotype: str, fallback: BiologicalContext, complete: Callable) -> BiologicalContext:
    """Bound logical proposal/selection calls; transport failures do not trigger a new call."""
    manifest = _candidate_manifest(fallback)
    propose = not manifest["candidates"]
    proposed = False
    failure = None
    for _ in range(MAX_CONTEXT_MODEL_CALLS):
        request = {"phenotype": phenotype}
        if not propose:
            request["candidate_biosamples"] = manifest
        if failure is not None:
            request["previous_resolution_error"] = failure
        prompt = "biological_context_proposal_system" if propose else "biological_context_system"
        try:
            completion = complete([{"role": "system", "content": load_prompt(prompt)},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)}])
        except Exception as exc:
            from ..model_generation import classify_generation_failure
            reason = "model_" + classify_generation_failure(exc)["category"]
            recovery = curated_biological_context(phenotype, reason=reason)
            if recovery is not None:
                return recovery
            raise ContextResolutionError(reason) from exc
        try:
            value = _completion_json(completion)
            if propose:
                proposed = True
                terms = _proposal_terms(value)
                if not any(terms.values()):
                    raise ContextResolutionError("empty_proposal")
                manifest = _candidate_manifest(fallback, search_terms=terms)
                if not manifest["candidates"]:
                    raise ContextResolutionError("no_catalog_candidates")
                propose = False
            else:
                return _validated_plan(phenotype, value, manifest)
        except (ValueError, TypeError, AttributeError) as exc:
            failure = exc.reason if isinstance(exc, ContextResolutionError) else "invalid_model_structure_or_identity"
            # A curated seed selection can be widened once by independent
            # phenotype-only proposals; an unseeded selection gets one repair.
            if not propose and not proposed:
                propose = True
    recovery = curated_biological_context(phenotype, reason=failure or "model_call_budget")
    if recovery is not None:
        return recovery
    raise ContextResolutionError(failure or "model_call_budget")


class ContextPlanner:
    """Coalesce concurrent requests; retain immutable plans, never model credentials.

    Explicit user terms take precedence. Curated phenotypes supply fixed seeds;
    other phenotypes propose search terms, search the full local catalog, then
    select only offered IDs. Both workflows are independent of SNP count.
    Contexts are saved with each run for replay; this cache only saves requests.
    """

    def __init__(self, max_entries: int = 256):
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, Future] = OrderedDict()
        self._max_entries = max_entries

    def resolve(self, phenotype: str, *, model_id: str = "", endpoint: str = "",
                complete: Callable | None = None, explicit_terms: tuple[str, ...] = ()) -> BiologicalContext:
        if explicit_terms:
            fallback = standardize_biological_context(
                define_biological_context(phenotype, explicit_terms=explicit_terms))
            if not fallback.is_defined or fallback.biosample_mapping["status"] != "resolved":
                raise ContextResolutionError("explicit_identity_unresolved")
            return fallback
        fallback = curated_biological_context(phenotype)
        if fallback is None:
            fallback = standardize_biological_context(BiologicalContext(phenotype=phenotype))
        if phenotype.strip().casefold() in {
            "", "user-selected gwas locus", "the loaded phenotype", "loaded gwas phenotype",
        }:
            raise ContextResolutionError("phenotype_unspecified")
        if complete is None:
            if not fallback.is_defined:
                raise ContextResolutionError("model_unavailable_no_curated_context")
            return fallback
        catalog = biosample_catalog()
        key = (PLANNER_VERSION, phenotype.strip().casefold(), model_id, endpoint,
               catalog.metadata["catalog_sha256"], load_prompt("biological_context_system"),
               load_prompt("biological_context_proposal_system"))
        with self._lock:
            pending = self._entries.get(key)
            owner = pending is None
            if owner:
                pending = Future()
                self._entries[key] = pending
            self._entries.move_to_end(key)
        if not owner:
            # The cache matches phenotype aliases case-insensitively; its public
            # snapshot must still bind the exact current query string and hash.
            return _detached_plan(pending.result(), phenotype)
        try:
            result = _resolve_model_plan(phenotype, fallback, complete)
        except Exception as exc:
            # A planner failure is unresolved context, never an all-tissue match.
            # Do not cache failures: a later request may recover normally.
            pending.set_exception(exc)
            with self._lock:
                self._entries.pop(key, None)
            raise
        pending.set_result(result)
        with self._lock:
            for old_key in list(self._entries):
                if len(self._entries) <= self._max_entries:
                    break
                if self._entries[old_key].done():
                    self._entries.pop(old_key)
        return _detached_plan(result, phenotype)
