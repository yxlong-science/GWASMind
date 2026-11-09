"""Saved EFO phenotype scopes, one file per target identity, reused by every later analysis."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..configuration import parameter
from ..file_lock import FileLockTimeout, exclusive_file_lock
from ..phenotype_scope import (
    _IDENTIFIER,
    EFO_RELATIONS,
    MAX_PHENOTYPE_CANDIDATES,
    MAX_RELATED_PHENOTYPES,
    PhenotypeScopeError,
    freeze_phenotype_scope,
    normalized_phenotype_label,
    seeded_target_id,
)

STORE_VERSION = "efo-phenotype-scope-store-v2"
SELECTORS = ("model", "user")
ALIAS_POLICY = "equivalent-identity-or-abstain-v1"
USER_ADDED_REASON = "Added by the user from the EFO neighbourhood ({relation})."


def _now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def _text(value, limit=500):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _row(row, keys):
    if not isinstance(row, Mapping) or set(row) != set(keys):
        raise PhenotypeScopeError("invalid_saved_scope_row")
    if not isinstance(row["id"], str) or not _IDENTIFIER.fullmatch(row["id"]) or not _text(row["name"]):
        raise PhenotypeScopeError("invalid_saved_scope_identity")
    if "exact_synonyms" in row and (not isinstance(row["exact_synonyms"], list) or len(row["exact_synonyms"]) > 100 or any(
            not isinstance(term, str) or len(term) > 500 for term in row["exact_synonyms"])):
        raise PhenotypeScopeError("invalid_saved_scope_synonyms")
    if "relation" in row and row["relation"] not in EFO_RELATIONS:
        raise PhenotypeScopeError("invalid_saved_scope_relation")
    if "reason" in row and not _text(row["reason"]):
        raise PhenotypeScopeError("invalid_saved_scope_reason")
    return deepcopy(dict(row))


def validate_saved_scope(record):
    fields = {"version", "target", "target_reason", "related", "candidates", "ontology_release",
              "selected_by", "created_at", "updated_at", "selection_issues", "revision"}
    if not isinstance(record, Mapping) or set(record) != fields or record["version"] != STORE_VERSION:
        raise PhenotypeScopeError("unsupported_saved_scope")
    target = _row(record["target"], ("id", "name", "exact_synonyms"))
    if not isinstance(record["candidates"], list) or not isinstance(record["related"], list):
        raise PhenotypeScopeError("invalid_saved_scope_candidates")
    candidates = [_row(row, ("id", "name", "exact_synonyms", "relation")) for row in record["candidates"]]
    by_id = {row["id"]: row for row in candidates}
    if len(by_id) != len(candidates) or target["id"] in by_id or len(candidates) >= MAX_PHENOTYPE_CANDIDATES:
        raise PhenotypeScopeError("invalid_saved_scope_candidates")
    related = [_row(row, ("id", "name", "reason")) for row in record["related"]]
    if len(related) > MAX_RELATED_PHENOTYPES or len({row["id"] for row in related}) != len(related):
        raise PhenotypeScopeError("invalid_saved_scope_related")
    if any(row["id"] not in by_id or by_id[row["id"]]["name"] != row["name"] for row in related):
        raise PhenotypeScopeError("saved_related_outside_candidates")
    if (not _text(record["target_reason"]) or record["selected_by"] not in SELECTORS
            or not _text(record["ontology_release"], 40)
            or not all(_text(record[key], 40) for key in ("created_at", "updated_at"))):
        raise PhenotypeScopeError("invalid_saved_scope_provenance")
    if (not isinstance(record["selection_issues"], list) or len(record["selection_issues"]) > 10
            or any(not _text(issue, 200) for issue in record["selection_issues"])
            or not isinstance(record["revision"], str) or len(record["revision"]) != 32
            or any(char not in "0123456789abcdef" for char in record["revision"])):
        raise PhenotypeScopeError("invalid_saved_scope_revision")
    return deepcopy(dict(record))


def saved_scope_record(*, target, target_reason, related, candidates, ontology_release, selected_by="model",
                       selection_issues=()):
    stamp = _now()
    return validate_saved_scope({"version": STORE_VERSION, "target": dict(target), "target_reason": target_reason,
        "related": [dict(row) for row in related], "candidates": [dict(row) for row in candidates],
        "ontology_release": ontology_release, "selected_by": selected_by, "created_at": stamp, "updated_at": stamp,
        "selection_issues": list(dict.fromkeys(selection_issues)), "revision": uuid4().hex})


def scope_from_saved(phenotype, record, *, target_reason=None, issues=()):
    """Freeze one analysis scope; the provenance text depends only on the saved record."""
    value = validate_saved_scope(record)
    selector = "the user" if value["selected_by"] == "user" else "the model"
    source = (f"Saved EFO phenotype scope for {value['target']['id']} (Open Targets {value['ontology_release']}); "
              f"related phenotypes selected by {selector} from the EFO neighbourhood")
    manifest = [value["target"], *({key: row[key] for key in ("id", "name", "exact_synonyms")}
                                   for row in value["candidates"])]
    return freeze_phenotype_scope(phenotype, target=value["target"], related=value["related"],
        candidates=manifest, source=source, target_reason=target_reason or value["target_reason"],
        issues=tuple(dict.fromkeys((*value["selection_issues"], *issues))))


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Phenotype scope planning was cancelled.")


def _retry_file_access(operation, cancel_event=None):
    """Briefly recover Windows sharing/permission races; persistent I/O errors remain explicit."""
    for attempt in range(3):
        _check_cancel(cancel_event)
        try:
            return operation()
        except PermissionError:
            if attempt == 2:
                raise
            delay = 0.05 * 2 ** attempt
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    raise CancelledError("Phenotype scope storage recovery was cancelled.")
            else:
                time.sleep(delay)
    raise AssertionError("Unreachable phenotype storage retry state")


class PhenotypeScopeStore:
    """Atomic JSON files under one directory; edits apply to the next analysis of that target."""

    def __init__(self, root, *, lock_timeout_seconds=None):
        self.root = Path(root).resolve()
        timeout = parameter("web.phenotype_scope_lock_timeout_seconds") if lock_timeout_seconds is None else lock_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Scope lock timeout must be a finite positive number.")
        self.lock_timeout_seconds = float(timeout)

    @contextmanager
    def _transaction(self, name, cancel_event=None):
        """OS locks release after a process exits; lock files must never be unlinked."""
        _check_cancel(cancel_event)
        directory = self.root / ".locks"
        path = directory / (hashlib.sha256(name.encode()).hexdigest() + ".lock")
        try:
            with exclusive_file_lock(path, self.lock_timeout_seconds, cancel_event):
                yield
        except FileLockTimeout as exc:
            raise PhenotypeScopeError("saved_scope_lock_timeout") from exc

    def selection(self, target_id, cancel_event=None):
        self._path(target_id)
        return self._transaction("selection:" + target_id, cancel_event)

    def alias_selection(self, phenotype, cancel_event=None):
        return self._transaction("alias-selection:" + normalized_phenotype_label(phenotype), cancel_event)

    def _path(self, target_id):
        if not isinstance(target_id, str) or not _IDENTIFIER.fullmatch(target_id):
            raise PhenotypeScopeError("invalid_phenotype_identifier")
        return self.root / f"{target_id}.json"

    def _write(self, path, value, cancel_event=None):
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=self.root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            _retry_file_access(lambda: os.replace(temporary, path), cancel_event)
        finally:
            temporary.unlink(missing_ok=True)

    def _read(self, path, cancel_event=None):
        try:
            return json.loads(_retry_file_access(lambda: path.read_text(encoding="utf-8"), cancel_event))
        except (FileNotFoundError, ValueError):
            return None

    @staticmethod
    def _record(value, target_id):
        try:
            record = validate_saved_scope(value)
        except PhenotypeScopeError:
            return None
        return record if record["target"]["id"] == target_id else None

    def snapshot(self, target_id, cancel_event=None):
        path = self._path(target_id)
        with self._transaction("state:" + target_id, cancel_event):
            value = self._read(path, cancel_event)
        revision = value.get("revision") if isinstance(value, Mapping) else None
        return self._record(value, target_id), revision

    def get(self, target_id, *, cancel_event=None):
        return self.snapshot(target_id, cancel_event)[0]

    def put(self, record):
        value = validate_saved_scope(record)
        value["revision"] = uuid4().hex
        with self._transaction("state:" + value["target"]["id"]):
            self._write(self._path(value["target"]["id"]), value)
        return value

    def put_if_current(self, record, expected_revision, cancel_event=None):
        """An in-flight model selection must never overwrite a newer Settings edit/reset."""
        value = validate_saved_scope(record)
        target_id = value["target"]["id"]
        with self._transaction("state:" + target_id, cancel_event):
            current = self._read(self._path(target_id), cancel_event)
            revision = current.get("revision") if isinstance(current, Mapping) else None
            if revision != expected_revision:
                return self._record(current, target_id), False
            value["revision"] = uuid4().hex
            self._write(self._path(target_id), value, cancel_event)
        return value, True

    def delete(self, target_id):
        path = self._path(target_id)
        with self._transaction("state:" + target_id):
            existed = self._record(self._read(path), target_id) is not None
            # A tombstone revision prevents an older absent-record selection from recreating it.
            self._write(path, {"version": STORE_VERSION, "deleted": True, "target_id": target_id,
                               "revision": uuid4().hex})
        with self._transaction("aliases"):
            aliases = self._read(self.root / "aliases.json")
            aliases = aliases if isinstance(aliases, Mapping) else {}
            kept = {label: row for label, row in aliases.items()
                    if not isinstance(row, Mapping) or row.get("target_id") != target_id}
            if kept != aliases:
                self._write(self.root / "aliases.json", kept)
        return existed

    def list(self, *, cancel_event=None):
        records = []
        if self.root.is_dir():
            for path in sorted(self.root.glob("*.json")):
                if _IDENTIFIER.fullmatch(path.stem) and (record := self.get(path.stem, cancel_event=cancel_event)) is not None:
                    records.append(record)
        return records

    def update_related(self, target_id, related_ids: Sequence[str]):
        if not isinstance(related_ids, list) or len(related_ids) > MAX_RELATED_PHENOTYPES or any(
                not isinstance(item, str) for item in related_ids) or len(set(related_ids)) != len(related_ids):
            raise PhenotypeScopeError("invalid_related_selection")
        path = self._path(target_id)
        with self._transaction("state:" + target_id):
            record = self._record(self._read(path), target_id)
            if record is None:
                raise PhenotypeScopeError("saved_scope_not_found")
            candidates = {row["id"]: row for row in record["candidates"]}
            if any(item not in candidates for item in related_ids):
                raise PhenotypeScopeError("related_selection_outside_manifest")
            previous = {row["id"]: row["reason"] for row in record["related"]}
            record["related"] = [{"id": item, "name": candidates[item]["name"], "reason": previous.get(item)
                                  or USER_ADDED_REASON.format(relation=candidates[item]["relation"].replace("_", " "))}
                                 for item in related_ids]
            record.update(selected_by="user", updated_at=_now(), revision=uuid4().hex)
            self._write(path, record)
        return record

    def for_phenotype(self, phenotype, *, cancel_event=None):
        """Only an exact saved identity/synonym, curated ID or verified saved alias may recover a scope."""
        alias = self.alias(phenotype, cancel_event=cancel_event)
        identity = seeded_target_id(phenotype) or (alias or {}).get("target_id")
        if identity:
            return self.get(identity, cancel_event=cancel_event)
        label = normalized_phenotype_label(phenotype)
        matches = [record for record in self.list(cancel_event=cancel_event) if label in {
            normalized_phenotype_label(term) for term in (record["target"]["id"], record["target"]["name"],
                                                         *record["target"]["exact_synonyms"])}]
        return matches[0] if len(matches) == 1 else None

    def alias(self, phenotype, *, cancel_event=None):
        with self._transaction("aliases", cancel_event):
            aliases = self._read(self.root / "aliases.json", cancel_event) or {}
        row = aliases.get(normalized_phenotype_label(phenotype)) if isinstance(aliases, Mapping) else None
        # Aliases saved before abstention was allowed were forced choices and are selected again.
        if (not isinstance(row, Mapping) or set(row) != {"target_id", "target_reason", "policy"}
                or row["policy"] != ALIAS_POLICY
                or not isinstance(row["target_id"], str) or not _IDENTIFIER.fullmatch(row["target_id"])
                or not _text(row["target_reason"])):
            return None
        return {"target_id": row["target_id"], "target_reason": row["target_reason"]}

    def save_alias(self, phenotype, target_id, target_reason, *, cancel_event=None):
        self._path(target_id)
        if not _text(target_reason):
            raise PhenotypeScopeError("invalid_saved_scope_reason")
        with self._transaction("aliases", cancel_event):
            aliases = self._read(self.root / "aliases.json", cancel_event)
            aliases = dict(aliases) if isinstance(aliases, Mapping) else {}
            aliases[normalized_phenotype_label(phenotype)] = {"target_id": target_id, "target_reason": target_reason,
                                                              "policy": ALIAS_POLICY}
            self._write(self.root / "aliases.json", aliases, cancel_event)
