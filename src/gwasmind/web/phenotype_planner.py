"""Phenotype scope with an EFO backbone: exact target match, ontology candidates, one saved selection."""
from __future__ import annotations

import asyncio
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import CancelledError, Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import nullcontext

from ..audit import sha256_json
from ..configuration import parameter
from ..model_generation import classify_generation_failure
from ..phenotype_scope import (
    MAX_PHENOTYPE_CANDIDATES,
    MAX_RELATED_PHENOTYPES,
    PhenotypeScopeError,
    efo_neighbourhood,
    exact_target,
    freeze_phenotype_scope,
    phenotype_identities,
    phenotype_search,
    rebind_phenotype_scope,
)
from ..prompting import load_prompt
from .phenotype_scope_store import _check_cancel, saved_scope_record, scope_from_saved

PLANNER_VERSION = "phenotype-scope-planner-v4-atomic-efo"
MAX_SELECTION_ATTEMPTS = 2
EXACT_TARGET_REASON = "The requested phenotype exactly matches this EFO name or exact synonym."


def _json(completion):
    text = str(completion.content).strip()
    if text.startswith("```") and "\n" in text:
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise PhenotypeScopeError("model_scope_object_required")
    return value


def _reason(value, key="target_reason"):
    reason = value.get(key)
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise PhenotypeScopeError("target_selection_reason_required")
    return reason


def _target_selection(value, primary_ids):
    """A null target_id records that no offered identity is equivalent to the requested phenotype."""
    if set(value) != {"target_id", "target_reason", "related"} or (
            value.get("target_id") is not None and value.get("target_id") not in primary_ids):
        raise PhenotypeScopeError("invalid_target_selection")
    return value["target_id"], _reason(value)


def _related_selection(value, target_id, candidates):
    if set(value) != {"target_id", "target_reason", "related"} or value.get("target_id") != target_id:
        raise PhenotypeScopeError("invalid_target_selection")
    rows = value.get("related")
    if not isinstance(rows, list) or len(rows) > MAX_RELATED_PHENOTYPES:
        raise PhenotypeScopeError("invalid_related_selection")
    related, seen = [], {target_id}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "reason"} or row.get("id") not in candidates:
            raise PhenotypeScopeError("related_selection_outside_manifest")
        if row["id"] in seen:
            raise PhenotypeScopeError("duplicate_selected_identity")
        if not isinstance(row["reason"], str) or not row["reason"].strip() or len(row["reason"]) > 500:
            raise PhenotypeScopeError("related_phenotype_reason_required")
        seen.add(row["id"])
        related.append({"id": row["id"], "name": candidates[row["id"]]["name"], "reason": row["reason"]})
    return related


async def _cancellable(awaitable, cancel_event):
    task = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            _check_cancel(cancel_event)
            await asyncio.wait({task}, timeout=0.05)
        _check_cancel(cancel_event)
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _scope_executor(executor, *, cancel_event=None, managed_retries=False):
    """Use one bounded retry layer; official OT transport already owns its retries."""
    async def execute(query, variables):
        retries = 0 if managed_retries else parameter("transport.graphql_max_retries")
        for attempt in range(retries + 1):
            _check_cancel(cancel_event)
            try:
                return await _cancellable(executor(query, variables), cancel_event)
            except CancelledError:
                raise
            except Exception as exc:
                details = getattr(exc, "failure_details", {})
                details = details if isinstance(details, Mapping) else {}
                if (attempt >= retries or details.get("recovery_exhausted")
                        or details.get("retryable") is False or not classify_generation_failure(exc)["retryable"]):
                    raise
                delay = parameter("transport.graphql_retry_backoff_seconds") * 2 ** attempt
                await _cancellable(asyncio.sleep(delay), cancel_event)
        raise AssertionError("Unreachable phenotype lookup retry state")
    return execute


async def _plan(phenotype, biosample_context, *, executor, complete, store=None, cancel_event=None):
    issues = []

    def selector(prompt_name, payload, parse):
        failure = None
        for _ in range(MAX_SELECTION_ATTEMPTS):
            _check_cancel(cancel_event)
            request = {**payload, **({"previous_resolution_error": failure} if failure else {})}
            try:
                response = complete([{"role": "system", "content": load_prompt(prompt_name)},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)}])
                _check_cancel(cancel_event)
            except CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - return bounded provider failure categories only
                return None, "model_" + classify_generation_failure(exc)["category"]
            try:
                return parse(_json(response)), None
            except (ValueError, TypeError, AttributeError):
                failure = "invalid_scope_identity_selection"
        return None, failure

    saved_fallback = store.for_phenotype(phenotype, cancel_event=cancel_event) if store is not None else None
    try:
        primary_ids = await phenotype_search(executor, phenotype)
        primary = await phenotype_identities(executor, primary_ids)
    except CancelledError:
        raise
    except Exception:  # noqa: BLE001 - recover only an existing verified saved identity
        if saved_fallback is not None:
            current = store.get(saved_fallback["target"]["id"], cancel_event=cancel_event)
            if current is not None:
                return scope_from_saved(phenotype, current, issues=("saved_scope_not_reverified",))
        return freeze_phenotype_scope(phenotype, source="Open Targets phenotype identity unavailable",
            status="target_unresolved", issues=("phenotype_identity_lookup_unavailable",))
    primary_ids = tuple(identity for identity in primary_ids if identity in primary)
    target, target_reason = exact_target(phenotype, primary.values()), EXACT_TARGET_REASON
    if target is None:
        with store.alias_selection(phenotype, cancel_event) if store is not None else nullcontext():
            alias = store.alias(phenotype, cancel_event=cancel_event) if store is not None else None
            if alias is not None:
                try:
                    known = primary if alias["target_id"] in primary else await phenotype_identities(executor, [alias["target_id"]])
                    target, target_reason = known.get(alias["target_id"]), alias["target_reason"]
                except CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - an unverified alias cannot establish target identity
                    target = None
            if target is None:
                if not primary_ids or complete is None:
                    return freeze_phenotype_scope(phenotype, candidates=primary.values(),
                        source="Verified target identity; related phenotype expansion not assessed", status="target_unresolved",
                        issues=("target_identity_unresolved" if not primary_ids else "model_expansion_not_requested",))
                chosen, failure = selector("phenotype_scope_selection_system", {"phenotype": phenotype,
                    "biosample_context": biosample_context, "selection_phase": "target_identity",
                    "target_candidate_ids": list(primary_ids), "exact_target_id": None,
                    "target_scope_rule": "Choose the exact broad phenotype when available; do not substitute particle-size subfractions, ratios or combined definitions for the requested measurement.",
                    "candidate_phenotypes": [primary[key] for key in primary_ids]},
                    lambda value: _target_selection(value, primary_ids))
                if chosen is None or chosen[0] is None:
                    return freeze_phenotype_scope(phenotype, candidates=primary.values(),
                        source="Verified target identity; related phenotype expansion not assessed",
                        status="target_unresolved",
                        issues=(failure if chosen is None else "no_equivalent_target_identity",))
                target, target_reason = primary[chosen[0]], chosen[1]
                if store is not None:
                    store.save_alias(phenotype, target["id"], target_reason, cancel_event=cancel_event)

    def target_only(reason, candidates=()):
        return freeze_phenotype_scope(phenotype, target=target, candidates=[target, *candidates],
            source="Verified target identity; related phenotype expansion not assessed",
            status="expansion_not_assessed", target_reason=target_reason, issues=(*issues, reason))

    # The same verified OT identity owns one selection even across aliases and processes.
    with store.selection(target["id"], cancel_event) if store is not None else nullcontext():
        saved, revision = store.snapshot(target["id"], cancel_event) if store is not None else (None, None)
        if saved is not None:
            try:
                observed = await phenotype_identities(executor, [target["id"], *(row["id"] for row in saved["related"])])
            except CancelledError:
                raise
            except Exception:  # noqa: BLE001 - retain the saved scope with explicit revalidation failure
                return scope_from_saved(phenotype, saved, target_reason=target_reason,
                                        issues=("saved_scope_not_reverified",))
            if all(observed.get(row["id"], {}).get("name") == row["name"] for row in (saved["target"], *saved["related"])):
                return scope_from_saved(phenotype, saved, target_reason=target_reason)
            issues.append("saved_scope_identity_changed")
        if complete is None:
            return target_only("model_expansion_not_requested")
        try:
            release, neighbourhood = await efo_neighbourhood(executor, target["id"])
            if len(neighbourhood) >= MAX_PHENOTYPE_CANDIDATES:
                neighbourhood = neighbourhood[:MAX_PHENOTYPE_CANDIDATES - 1]
                issues.append("efo_candidate_list_truncated")
            identities = await phenotype_identities(executor, [identity for identity, _ in neighbourhood])
        except CancelledError:
            raise
        except Exception:  # noqa: BLE001 - preserve the verified target if expansion fails
            return target_only("efo_structure_unavailable")
        candidates = [{**identities[identity], "relation": relation}
                      for identity, relation in neighbourhood if identity in identities]
        manifest_rows = [{key: row[key] for key in ("id", "name", "exact_synonyms")} for row in candidates]
        related = []
        if candidates:
            by_id = {row["id"]: row for row in candidates}
            related, failure = selector("phenotype_scope_selection_system", {"phenotype": phenotype,
                "biosample_context": biosample_context, "selection_phase": "related_phenotypes",
                "target_candidate_ids": [target["id"]], "exact_target_id": target["id"],
                "target_phenotype": target, "candidate_phenotypes": candidates},
                lambda value: _related_selection(value, target["id"], by_id))
            if related is None:
                return target_only(failure, manifest_rows)
        record = saved_scope_record(target=target, target_reason=target_reason, related=related,
            candidates=candidates, ontology_release=release,
            selection_issues=[issue for issue in issues if issue == "efo_candidate_list_truncated"])
        if store is not None:
            record, committed = store.put_if_current(record, revision, cancel_event)
            if record is None:
                return target_only("saved_scope_changed_during_planning", manifest_rows)
            if not committed:
                return scope_from_saved(phenotype, record, target_reason=target_reason)
        return scope_from_saved(phenotype, record, target_reason=target_reason, issues=issues)


class PhenotypeScopePlanner:
    """Coalesce phenotype plans without credentials, variant data or gene evidence."""

    def __init__(self, max_entries=256, store=None):
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._max_entries = max_entries
        self.store = store
        self._epoch = 0

    def forget(self):
        """Invalidate pending and completed plans for every request started after an edit."""
        with self._lock:
            self._epoch += 1
            self._entries.clear()

    def resolve(self, phenotype, *, biological_context, complete=None, model_id="", endpoint="", executor=None,
                cancel_event=None):
        _check_cancel(cancel_event)
        axes = {axis: list(biological_context.get(axis, ())) for axis in ("organs", "tissues", "cell_types")}
        key = (PLANNER_VERSION, phenotype.strip().casefold(), model_id, endpoint, complete is not None, sha256_json(axes),
               load_prompt("phenotype_scope_selection_system"))
        with self._lock:
            # Saved scopes coalesce at the per-target OS transaction. Sharing a
            # request Future would also share its owner's cancellation event.
            key = (self._epoch, object() if self.store is not None else None, *key)
            pending = self._entries.get(key)
            owner = pending is None
            if owner:
                pending = Future()
                self._entries[key] = pending
            self._entries.move_to_end(key)
        if not owner:
            while True:
                _check_cancel(cancel_event)
                try:
                    return rebind_phenotype_scope(pending.result(timeout=0.05), phenotype)
                except FutureTimeoutError:
                    if pending.done():
                        return rebind_phenotype_scope(pending.result(), phenotype)
                    continue

        async def run():
            if executor is not None:
                return await _plan(phenotype, axes, executor=_scope_executor(executor, cancel_event=cancel_event),
                                   complete=complete, store=self.store, cancel_event=cancel_event)
            from ..providers.live import OfficialOpenTargetsGraphQLTransport
            from ..v2g.retrieval import graphql_transport_executor
            transport = OfficialOpenTargetsGraphQLTransport()
            try:
                bounded = _scope_executor(graphql_transport_executor(transport), cancel_event=cancel_event,
                                          managed_retries=True)
                return await _plan(phenotype, axes, executor=bounded, complete=complete, store=self.store,
                                   cancel_event=cancel_event)
            finally:
                transport.close()

        try:
            result = asyncio.run(run())
        except BaseException as exc:
            pending.set_exception(exc)
            with self._lock:
                if self._entries.get(key) is pending:
                    self._entries.pop(key, None)
            raise
        pending.set_result(result)
        with self._lock:
            cacheable = ((result["status"] == "resolved" and not result["issues"]) or
                (result["status"] == "expansion_not_assessed" and
                 result["issues"] == ["model_expansion_not_requested"]))
            if (self.store is not None or not cacheable) and self._entries.get(key) is pending:
                self._entries.pop(key, None)
            for old_key in list(self._entries):
                if len(self._entries) <= self._max_entries:
                    break
                if self._entries[old_key].done():
                    self._entries.pop(old_key)
        return rebind_phenotype_scope(result, phenotype)
