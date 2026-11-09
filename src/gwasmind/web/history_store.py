"""Local, restart-safe dataset and report history without model credentials.

The database belongs in ``<software directory>/data/history``.  Compressed
payloads and their listing metadata commit in one SQLite transaction, so an
interrupted write cannot leave a visible entry pointing at a missing file.
Only compact plot data and explicit report fields are retained; API settings,
request headers, and provider responses are never accepted as history fields.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sqlite3
import stat
import threading
from collections.abc import Mapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .ui_state import UiDataError, UiState

_IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")
_KINDS = frozenset({"datasets", "chats", "analyses"})
MAX_TITLE_WORDS = 10
MAX_TITLE_CHARS = 200
DATASET_IDENTITY_VERSION = "dataset-v2"


def _dataset_fingerprint(payload: Mapping[str, Any]) -> str:
    """Keep scientific and sampling metadata, excluding parser wall time."""
    metadata = dict(payload.get("metadata", {}))
    metadata.pop("processing_time", None)
    canonical = {**payload, "metadata": metadata}
    return DATASET_IDENTITY_VERSION + ":" + hashlib.sha256(_encode(canonical)).hexdigest()


def clean_title(value: str) -> str:
    """Fold a stored title onto one line and at most MAX_TITLE_WORDS words.

    The title is what a user scans in the history list, so it is capped by
    words rather than characters; the character cap only bounds a single
    pathological token.
    """
    words = value.split()
    if not words:
        return "Untitled"
    return " ".join(words[:MAX_TITLE_WORDS])[:MAX_TITLE_CHARS] or "Untitled"


_MAX_PAYLOAD_BYTES = 32 * 1024 * 1024
_MAX_REPORT_TEXT = 1_000_000
_MAX_MESSAGES = 1_000
_MAX_NESTING = 20
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|secret|"
    r"access[_-]?token|refresh[_-]?token|provider[_-]?settings|"
    r"model[_-]?settings|request[_-]?headers|environment|reasoning_content)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"\b(?:sk-[a-zA-Z0-9_-]{12,}|Bearer\s+[a-zA-Z0-9_.~-]{12,})",
    re.IGNORECASE,
)
_MESSAGE_FIELDS = frozenset({
    "role", "content", "timestamp", "selected_variants", "selected_region",
    "selected_regions", "run_id", "report_quality", "progress", "analysis_context", "kind",
    "analysis_status", "computed_facts", "variant_normalizations", "ranking_evidence",
    "gene_evidence_cards", "variant_labels",
})
_GENE_CARD_FIELDS = frozenset({
    "gene_id", "symbol", "rank", "shared_rank", "tied", "tier", "tier_basis", "tss_distance_bp", "distance_rank",
    "key_evidence", "rationale", "categories", "model_selection", "selection_reason", "selection_evidence_ids",
})
_GENE_CARD_CATEGORY_FIELDS = frozenset({"status", "count", "items", "rows", "preview", "scope_source", "tags"})
_QTL_ROW_TEXT_LIMITS = {
    "variant_id": 120, "locus_variant_id": 120, "variant_origin": 30, "variant_label": 30,
    "qtl_type": 30, "biosample": 500, "biosample_id": 200, "p_value": 80, "beta": 80,
    "study_id": 2000, "study_locus_id": 2000, "evidence_id": 2000, "project": 200,
    "summary": 2000, "ld_reference_panel": 200,
    "cs_lead_variant_id": 120, "measurement_scope": 30,
    "variant_display": 300, "rsid": 80, "genome_build": 30,
}
_QTL_ROW_FIELDS = frozenset(_QTL_ROW_TEXT_LIMITS) | {
    "context_matched", "legacy_display", "measurement", "ld_relations", "locus_anchor_witnesses",
}
_QTL_VARIANT_ID = re.compile(r"(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)_[1-9][0-9]{0,11}_[ACGT]+_[ACGT]+")
_CHAT_FIELDS = frozenset({
    "title", "dataset_id", "source_phenotype", "messages", "selected_variants",
    "selected_region", "selected_regions", "run_id", "status", "analysis_context",
})
_ANALYSIS_FIELDS = frozenset({
    "title", "dataset_id", "source_phenotype", "target_phenotype", "task_id",
    "run_id", "filename", "queries", "results", "selected_variants",
    "selected_region", "status", "warnings", "snp_positions", "snp_count",
    "original_snp_count", "locus_count", "progress", "summary", "report",
    "chat_messages", "concurrency", "loci", "analysis_mode", "timing",
})


class HistoryError(ValueError):
    """History input is invalid, or an existing record cannot be read safely."""


class HistoryNotFoundError(HistoryError):
    """The requested opaque history identifier does not exist."""


def default_history_root(project_root: str | Path | None = None) -> Path:
    """Resolve storage beside the installed project, not the caller's cwd."""

    project = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    return project.resolve() / "data" / "history"


class LocalHistoryStore:
    """Thread-safe, atomic local snapshots with bounded chronological listing.

    Instantiate once per server; separate instances/processes can also read the
    same database.  Deleting data from the current UI does not delete history.
    No retention policy silently removes a user's saved datasets or reports.
    """

    def __init__(self, root: str | Path) -> None:
        requested_root = Path(root).absolute()
        requested_root.mkdir(parents=True, exist_ok=True)
        self.root = requested_root.resolve()
        self.database_path = self.root / "history.sqlite3"
        self._lock = threading.RLock()
        self._check_database_paths()
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS history ("
                "id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, title TEXT NOT NULL, "
                "dataset_id TEXT, source_phenotype TEXT NOT NULL DEFAULT '', "
                "item_count INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT '', "
                "fingerprint TEXT, payload BLOB NOT NULL, payload_size INTEGER NOT NULL, "
                "schema_version INTEGER NOT NULL DEFAULT 1)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS history_recent "
                "ON history(kind, updated_at DESC, id DESC)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS history_dataset_fingerprint "
                "ON history(kind, fingerprint) WHERE kind = 'datasets'"
            )

    def save_dataset(
        self, payload: Mapping[str, Any], *, data_source: str = "user_upload",
    ) -> dict[str, Any]:
        """Save validated, downsampled plot rows, never the original large file.

        Re-saving exactly the same normalized dataset reuses its identifier and
        updates its last-used timestamp. Distinct filenames/phenotypes remain
        distinct history entries. Restoring does not create a duplicate.
        """

        if not isinstance(payload, Mapping):
            raise HistoryError("dataset history must be an object")
        if data_source not in {"user_upload", "asthma", "mdd"}:
            raise HistoryError("dataset history source is unsupported")
        state = UiState()
        try:
            state.load_processed(payload)
        except UiDataError as exc:
            raise HistoryError(str(exc)) from exc
        normalized = state.plot_payload()
        normalized["metadata"] = _safe_value(normalized["metadata"])
        metadata = normalized["metadata"]
        if data_source != "user_upload":
            metadata.update({"data_source": data_source, "demo": True, "synthetic": False})
            source_file = payload.get("metadata", {}).get("source_file")
            if isinstance(source_file, str):
                metadata["source_file"] = _safe_value(source_file[:240])
        raw = _encode(normalized)
        fingerprint = _dataset_fingerprint(normalized)
        return self._write(
            "datasets", normalized,
            title=metadata.get("filename") or metadata["source_phenotype"],
            source_phenotype=metadata["source_phenotype"],
            item_count=normalized["total_snps"], fingerprint=fingerprint, raw=raw,
        )

    def get_dataset(self, history_id: str) -> dict[str, Any]:
        value = self.get("datasets", history_id)
        return {
            **value["payload"],
            "history_id": value["id"],
            "created_at": value["created_at"],
            "updated_at": value["updated_at"],
        }

    def list_datasets(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return self.list("datasets", limit=limit, offset=offset)

    def list_datasets_with_activity(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List uploaded datasets together with the conversations they own.

        A dataset is the unit a user recognises, so the upload list carries the
        number of saved chats and saved analyses that point at that upload. The
        counts are read from the same rows the flat lists use and never rewrite
        an existing snapshot.
        """
        return self._attach_dataset_children(self.list("datasets", limit=limit, offset=offset))

    def _attach_dataset_children(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return items
        placeholders = ", ".join("?" for _ in items)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT dataset_id, kind, COUNT(*) AS total, MAX(updated_at) AS last_activity "
                "FROM history WHERE kind IN ('chats', 'analyses') "
                f"AND dataset_id IN ({placeholders}) GROUP BY dataset_id, kind",
                [item["id"] for item in items],
            ).fetchall()
        counts: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = counts.setdefault(row["dataset_id"], {"chat_count": 0, "analysis_count": 0})
            entry["chat_count" if row["kind"] == "chats" else "analysis_count"] = int(row["total"])
            last = row["last_activity"]
            if isinstance(last, str) and (entry.get("last_activity") is None or last > entry["last_activity"]):
                entry["last_activity"] = last
        for item in items:
            entry = counts.get(item["id"], {})
            item["chat_count"] = int(entry.get("chat_count", 0))
            item["analysis_count"] = int(entry.get("analysis_count", 0))
            item["last_activity"] = entry.get("last_activity")
        return items

    def save_chat(
        self, payload: Mapping[str, Any], *, history_id: str | None = None,
    ) -> dict[str, Any]:
        normalized = _project(payload, _CHAT_FIELDS)
        messages = normalized.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HistoryError("chat history requires at least one message")
        if len(messages) > _MAX_MESSAGES:
            raise HistoryError(f"chat history accepts at most {_MAX_MESSAGES:,} messages")
        normalized_messages = []
        for message in messages:
            item = _project(message, _MESSAGE_FIELDS)
            if item.get("role") not in {"user", "assistant", "system"}:
                raise HistoryError("chat history contains an unsupported message role")
            content = item.get("content")
            if not isinstance(content, str) or len(content) > _MAX_REPORT_TEXT:
                raise HistoryError("chat history contains an invalid message")
            normalized_messages.append(item)
        normalized["messages"] = normalized_messages
        first_question = next(
            (item["content"] for item in normalized_messages if item["role"] == "user"),
            "Saved conversation",
        )
        # The opening question names the conversation once; a later save that
        # carries no title keeps whatever the record already says.
        return self._write_snapshot(
            "chats", normalized, history_id=history_id,
            title=normalized.get("title"), fallback_title=first_question,
            item_count=len(normalized_messages),
        )

    def get_chat(self, history_id: str) -> dict[str, Any]:
        return self._get_snapshot("chats", history_id)

    def list_chats(self, *, limit: int = 50, offset: int = 0,
                    dataset_id: str | None = None) -> list[dict[str, Any]]:
        return self.list("chats", limit=limit, offset=offset, dataset_id=dataset_id)

    def retitle_chats(self, titles: Mapping[str, str], *, restore_exact: bool = False) -> list[dict[str, Any]]:
        """Rename saved conversations without touching their content or order.

        The row title and the copy stored inside the snapshot payload are written
        together, so a later read cannot resurrect the previous name. Created and
        updated timestamps are preserved: renaming is not conversation activity.
        """
        if not isinstance(titles, Mapping) or not titles:
            raise HistoryError("a rename request needs at least one conversation")
        if len(titles) > 500:
            raise HistoryError("a rename request is limited to 500 conversations")
        renamed = []
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            for history_id, title in titles.items():
                _identifier(history_id)
                if not isinstance(title, str):
                    raise HistoryError("history title must be text")
                if restore_exact and (not title or len(title) > MAX_TITLE_CHARS or _safe_value(title) != title):
                    raise HistoryError("backed-up title is invalid")
                clean = title if restore_exact else clean_title(title)
                row = connection.execute(
                    "SELECT created_at, updated_at, title, payload, payload_size, schema_version "
                    "FROM history WHERE kind = 'chats' AND id = ?",
                    (history_id,),
                ).fetchone()
                if row is None:
                    raise HistoryNotFoundError("saved history was not found")
                if row["schema_version"] != 1:
                    raise HistoryError("saved history requires a newer GWASMind version")
                if row["title"] != clean:
                    payload = _decode(row["payload"], row["payload_size"])
                    if isinstance(payload, dict) and "title" in payload:
                        payload["title"] = clean
                        raw = _encode(payload)
                        connection.execute(
                            "UPDATE history SET title = ?, payload = ?, payload_size = ? WHERE id = ?",
                            (clean, gzip.compress(raw, compresslevel=1, mtime=0), len(raw), history_id),
                        )
                    else:
                        connection.execute(
                            "UPDATE history SET title = ? WHERE id = ?", (clean, history_id),
                        )
                renamed.append({
                    "id": history_id, "history_id": history_id, "kind": "chats",
                    "created_at": row["created_at"], "updated_at": row["updated_at"],
                    "title": clean, "previous_title": row["title"],
                })
        return renamed

    def save_analysis(
        self, payload: Mapping[str, Any], *, history_id: str | None = None,
        preserve_discussions: bool = False,
    ) -> dict[str, Any]:
        normalized = _project(payload, _ANALYSIS_FIELDS)
        queries = normalized.get("queries", [])
        results = normalized.get("results", [])
        if not isinstance(queries, list) or not isinstance(results, list):
            raise HistoryError("analysis history queries and results must be arrays")
        if not queries and not results:
            raise HistoryError("analysis history requires queries or results")
        discussions = normalized.get("chat_messages")
        if discussions is not None:
            if not isinstance(discussions, dict):
                raise HistoryError("analysis conversations must be indexed by locus number")
            clean_discussions = {}
            for locus, messages in discussions.items():
                if not re.fullmatch(r"[1-9][0-9]{0,3}", str(locus)):
                    raise HistoryError("analysis conversation locus number is invalid")
                clean_discussions[str(locus)] = _analysis_messages(messages)
            normalized["chat_messages"] = clean_discussions
        # An analysis record keeps the name it was created with; a later
        # progress update without a title cannot rename it.
        return self._write_snapshot(
            "analyses", normalized, history_id=history_id,
            title=normalized.get("title"),
            fallback_title=normalized.get("filename") or "Auto Analysis",
            item_count=max(len(queries), len(results)),
            preserve_discussions=preserve_discussions,
        )

    def update_analysis_discussion(
        self, history_id: str, locus_no: int, messages: Any,
    ) -> dict[str, Any]:
        """Replace one locus discussion without rewriting an older analysis snapshot."""
        _identifier(history_id)
        if isinstance(locus_no, bool) or not isinstance(locus_no, int) or not 1 <= locus_no <= 9999:
            raise HistoryError("analysis conversation locus number is invalid")
        clean_messages = _analysis_messages(messages)
        timestamp = _now()
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM history WHERE kind = 'analyses' AND id = ?", (history_id,),
            ).fetchone()
            if existing is None:
                raise HistoryNotFoundError("saved history was not found")
            if existing["schema_version"] != 1:
                raise HistoryError("saved history requires a newer GWASMind version")
            payload = _decode(existing["payload"], existing["payload_size"])
            discussions = payload.get("chat_messages", {})
            if not isinstance(discussions, dict):
                raise HistoryError("saved analysis conversations are invalid")
            payload["chat_messages"] = {**discussions, str(locus_no): clean_messages}
            raw = _encode(payload)
            compressed = gzip.compress(raw, compresslevel=1, mtime=0)
            connection.execute(
                "UPDATE history SET updated_at = ?, payload = ?, payload_size = ? "
                "WHERE kind = 'analyses' AND id = ?",
                (timestamp, compressed, len(raw), history_id),
            )
        return {
            "id": history_id, "history_id": history_id, "kind": "analyses",
            "created_at": existing["created_at"], "updated_at": timestamp,
            "title": existing["title"], "dataset_id": existing["dataset_id"],
            "source_phenotype": existing["source_phenotype"],
            "item_count": existing["item_count"], "status": existing["status"],
        }

    def get_analysis(self, history_id: str) -> dict[str, Any]:
        value = self._get_snapshot("analyses", history_id)
        # Read-time compatibility only: storage update timestamps describe saves
        # and discussions, so they can never supply an analysis duration.
        records = [value]
        for field in ("loci", "results"):
            if isinstance(value.get(field), list):
                records.extend(value[field])
        for record in records:
            if isinstance(record, dict) and "timing" not in record:
                timing = _recorded_timing(record)
                if timing is not None:
                    record["timing"] = timing
        return value

    def list_analyses(self, *, limit: int = 50, offset: int = 0,
                      dataset_id: str | None = None) -> list[dict[str, Any]]:
        return self.list("analyses", limit=limit, offset=offset, dataset_id=dataset_id)

    def list(
        self, kind: str, *, limit: int = 50, offset: int = 0, dataset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        _kind(kind)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise HistoryError("history list limit must be between 1 and 100")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 100_000:
            raise HistoryError("history list offset is invalid")
        clause, parameters = "", [kind]
        if dataset_id is not None:
            try:
                _identifier(dataset_id)
            except HistoryError as exc:
                raise HistoryError("history dataset filter is invalid") from exc
            clause, parameters = " AND dataset_id = ?", [kind, dataset_id]
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, kind, created_at, updated_at, title, dataset_id, "
                "source_phenotype, item_count, status FROM history WHERE kind = ?" + clause +
                " ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, kind: str, history_id: str) -> dict[str, Any]:
        _kind(kind)
        _identifier(history_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM history WHERE kind = ? AND id = ?", (kind, history_id),
            ).fetchone()
        if row is None:
            raise HistoryNotFoundError("saved history was not found")
        if row["schema_version"] != 1:
            raise HistoryError("saved history requires a newer GWASMind version")
        value = dict(row)
        value["payload"] = _decode(value["payload"], value["payload_size"])
        value.pop("fingerprint", None)
        value.pop("payload_size", None)
        return value

    def _get_snapshot(self, kind: str, history_id: str) -> dict[str, Any]:
        value = self.get(kind, history_id)
        return {
            **value["payload"], "history_id": value["id"], "title": value["title"],
            "created_at": value["created_at"], "updated_at": value["updated_at"],
        }

    def _write_snapshot(
        self, kind: str, payload: dict[str, Any], *, history_id: str | None,
        title: Any, item_count: int, preserve_discussions: bool = False,
        fallback_title: Any = None,
    ) -> dict[str, Any]:
        dataset_id = payload.get("dataset_id") or None
        if dataset_id is not None:
            _identifier(dataset_id)
            with closing(self._connect()) as connection:
                exists = connection.execute(
                    "SELECT source_phenotype FROM history WHERE kind = 'datasets' AND id = ?",
                    (dataset_id,),
                ).fetchone()
            if exists is None:
                raise HistoryNotFoundError("the conversation's saved dataset was not found")
            # The saved source phenotype is immutable even when a follow-up
            # question defines a different target phenotype.
            payload["source_phenotype"] = exists["source_phenotype"]
        phenotype = payload.get("source_phenotype", "")
        if not isinstance(phenotype, str) or len(phenotype) > 500:
            raise HistoryError("history source phenotype is invalid")
        return self._write(
            kind, payload, history_id=history_id, title=title, dataset_id=dataset_id,
            source_phenotype=phenotype, item_count=item_count,
            status=str(payload.get("status", ""))[:80],
            preserve_discussions=preserve_discussions,
            fallback_title=fallback_title,
        )

    def _write(
        self, kind: str, payload: dict[str, Any], *, title: Any,
        source_phenotype: str, item_count: int, history_id: str | None = None,
        dataset_id: str | None = None, status: str = "", fingerprint: str | None = None,
        raw: bytes | None = None, preserve_discussions: bool = False,
        fallback_title: Any = None,
    ) -> dict[str, Any]:
        if history_id is not None:
            _identifier(history_id)
        if title is not None and not isinstance(title, str):
            raise HistoryError("history title must be text")
        raw = raw if raw is not None else _encode(payload)
        compressed = gzip.compress(raw, compresslevel=1, mtime=0)
        timestamp = _now()
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = None
            if fingerprint is not None:
                existing = connection.execute(
                    "SELECT id, created_at, title FROM history WHERE kind = ? AND fingerprint = ?",
                    (kind, fingerprint),
                ).fetchone()
                if existing is None and fingerprint.startswith(DATASET_IDENTITY_VERSION + ":"):
                    # Adopt one matching legacy identity without merging rows or
                    # moving any existing conversation to another dataset.
                    legacy = connection.execute(
                        "SELECT id, created_at, title, payload, payload_size FROM history "
                        "WHERE kind = 'datasets' AND fingerprint NOT LIKE ? ORDER BY created_at, id",
                        (DATASET_IDENTITY_VERSION + ":%",),
                    )
                    for candidate in legacy:
                        if _dataset_fingerprint(_decode(candidate["payload"], candidate["payload_size"])) == fingerprint:
                            existing = candidate
                            connection.execute("UPDATE history SET fingerprint = ? WHERE id = ?", (fingerprint, candidate["id"]))
                            break
            elif history_id is not None:
                existing = connection.execute(
                    "SELECT id, created_at, title, dataset_id, payload, payload_size, schema_version "
                    "FROM history WHERE kind = ? AND id = ?",
                    (kind, history_id),
                ).fetchone()
                if existing is None:
                    raise HistoryNotFoundError("saved history was not found")
                if existing["dataset_id"] is not None and existing["dataset_id"] != dataset_id:
                    raise HistoryError("saved conversation cannot change its source dataset")
                if kind == "analyses" and preserve_discussions:
                    # Read under the same SQLite write transaction as the
                    # progress update, including writers using another store.
                    if existing["schema_version"] != 1:
                        raise HistoryError("saved history requires a newer GWASMind version")
                    previous = _decode(existing["payload"], existing["payload_size"])
                    payload = dict(payload)
                    if "chat_messages" in previous:
                        payload["chat_messages"] = previous["chat_messages"]
                    else:
                        payload.pop("chat_messages", None)
                    raw = _encode(payload)
                    compressed = gzip.compress(raw, compresslevel=1, mtime=0)
            # A record is named once. An update that carries no title keeps the
            # stored one instead of renaming a conversation the user may have
            # labelled or that was already named after its opening question.
            resolved = title if title else (existing["title"] if existing is not None else fallback_title)
            if not isinstance(resolved, str):
                raise HistoryError("history title must be text")
            title = existing["title"] if not title and existing is not None else clean_title(resolved)
            identifier = existing["id"] if existing is not None else uuid4().hex
            created_at = existing["created_at"] if existing is not None else timestamp
            connection.execute(
                "INSERT INTO history (id, kind, created_at, updated_at, title, dataset_id, "
                "source_phenotype, item_count, status, fingerprint, payload, payload_size) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at, "
                "title = excluded.title, dataset_id = excluded.dataset_id, "
                "source_phenotype = excluded.source_phenotype, item_count = excluded.item_count, "
                "status = excluded.status, payload = excluded.payload, "
                "payload_size = excluded.payload_size",
                (identifier, kind, created_at, timestamp, title, dataset_id,
                 source_phenotype, item_count, status, fingerprint, compressed, len(raw)),
            )
        return {
            "id": identifier, "history_id": identifier, "kind": kind,
            "created_at": created_at, "updated_at": timestamp, "title": title,
            "dataset_id": dataset_id, "source_phenotype": source_phenotype,
            "item_count": item_count, "status": status,
        }

    def _check_database_paths(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            candidate = self.root / ("history.sqlite3" + suffix)
            # One lstat per path: SQLite creates and removes its sidecar files while
            # other connections open and close, so separate exists() and is_file()
            # calls could observe a file that vanished in between.
            try:
                mode = candidate.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise HistoryError("history database must not be a symbolic link")
            if not stat.S_ISREG(mode):
                raise HistoryError("history database path is not a regular file")

    def _connect(self) -> sqlite3.Connection:
        self._check_database_paths()
        connection = sqlite3.connect(str(self.database_path), timeout=20.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 20000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection


def _analysis_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or len(messages) > _MAX_MESSAGES:
        raise HistoryError("analysis conversation exceeds its message limit")
    cleaned = []
    for message in messages:
        item = _project(message, _MESSAGE_FIELDS)
        if item.get("role") not in {"user", "assistant", "system"}:
            raise HistoryError("analysis conversation contains an invalid message role")
        if not isinstance(item.get("content"), str) or len(item["content"]) > _MAX_REPORT_TEXT:
            raise HistoryError("analysis conversation contains an invalid message")
        cleaned.append(item)
    return cleaned


def _identifier(value: Any) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise HistoryError("history identifier must be an opaque 32-character ID")


def _kind(value: Any) -> None:
    if not isinstance(value, str) or value not in _KINDS:
        raise HistoryError("history kind is not supported")


def _project(payload: Any, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise HistoryError("history payload must be an object")
    result = {key: _safe_value(value) for key, value in payload.items() if key in fields}
    if "selected_regions" in result:
        _validate_selected_regions(result["selected_regions"])
    if "variant_normalizations" in result:
        _validate_variant_normalizations(result["variant_normalizations"])
    if "ranking_evidence" in result:
        _validate_ranking_evidence(result["ranking_evidence"])
    if "gene_evidence_cards" in result:
        _validate_gene_evidence_cards(result["gene_evidence_cards"])
    if "variant_labels" in result:
        _validate_variant_labels(result["variant_labels"])
    return result


def _recorded_timing(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Derive only from explicit, ordered execution timestamps in older records."""
    started, finished = record.get("started_at"), record.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        if start.tzinfo is None or finish.tzinfo is None or finish < start:
            return None
        elapsed = round((finish - start).total_seconds(), 3)
    except (ValueError, OverflowError):
        return None
    return {"started_at": started, "finished_at": finished, "elapsed_seconds": elapsed,
            "observed_at": finished, "timing_basis": "recorded_timestamps"}


def _validate_variant_labels(value):
    """Display metadata cannot become an arbitrary destination or identity proof."""
    fields = {"variant_id", "canonical_variant_id", "rsid", "genome_build"}
    if not isinstance(value, list):
        raise HistoryError("variant_labels must be an array")
    seen = {}
    for row in value:
        if (not isinstance(row, dict) or set(row) != fields or row.get("genome_build") != "GRCh38"
                or not isinstance(row.get("rsid"), str) or not re.fullmatch(r"rs[1-9][0-9]*", row["rsid"])
                or not isinstance(row.get("variant_id"), str) or not _QTL_VARIANT_ID.fullmatch(row["variant_id"])
                or not isinstance(row.get("canonical_variant_id"), str)
                or not _QTL_VARIANT_ID.fullmatch(row["canonical_variant_id"])):
            raise HistoryError("variant_labels contains an invalid variant label")
        original, canonical = row["variant_id"].split("_"), row["canonical_variant_id"].split("_")
        if original[:2] != canonical[:2] or set(original[2:]) != set(canonical[2:]):
            raise HistoryError("variant_labels contains inconsistent variant alleles")
        previous = seen.setdefault(row["variant_id"], row)
        if previous != row:
            raise HistoryError("variant_labels contains conflicting display labels")


def _validate_gene_evidence_cards(value: Any) -> None:
    """Persist only the bounded public card summaries emitted by the V2G workflow."""
    if not isinstance(value, list):
        raise HistoryError("saved gene evidence cards must be a list")
    for card in value:
        if not isinstance(card, dict) or not set(card) <= _GENE_CARD_FIELDS:
            raise HistoryError("saved gene evidence card contains unsupported fields")
        if not isinstance(card.get("gene_id"), str) or not re.fullmatch(r"ENSG[0-9]{11}", card["gene_id"]):
            raise HistoryError("saved gene evidence card has an invalid gene identifier")
        if ("tier" in card and card["tier"] not in {"A", "B", "C"}) or not (card.get("rank") is None or type(card["rank"]) is int) \
                or not (card.get("shared_rank") is None or type(card["shared_rank"]) is int) \
                or not (card.get("tied") is None or type(card["tied"]) is bool):
            raise HistoryError("saved gene evidence card has an invalid tier or rank")
        if card.get("model_selection") not in {None, "selected", "not_selected", "unresolved"}:
            raise HistoryError("saved gene evidence card has an invalid model selection")
        references = card.get("selection_evidence_ids", [])
        if (not isinstance(references, list) or len(references) > 256
                or any(not isinstance(value, str) or len(value) > 300 for value in references)):
            raise HistoryError("saved gene evidence card selection references are invalid")
        texts = [card.get("symbol", ""), card.get("rationale", ""), card.get("selection_reason", ""), *card.get("tier_basis", []),
                 *card.get("key_evidence", [])]
        categories = card.get("categories", {})
        if not isinstance(categories, dict) or len(categories) > 12:
            raise HistoryError("saved gene evidence card has invalid categories")
        for category, summary in categories.items():
            if not isinstance(summary, dict) or not set(summary) <= _GENE_CARD_CATEGORY_FIELDS:
                raise HistoryError("saved gene evidence card category has unsupported fields")
            if "rows" in summary:
                if category != "qtl":
                    raise HistoryError("saved structured evidence rows are supported only for molecular QTL")
                _validate_qtl_rows(summary["rows"])
            if "preview" in summary:
                _validate_evidence_preview(summary["preview"], category)
            if "tags" in summary:
                tags = summary["tags"]
                allowed = {"Coding", "UTR", "Synonymous"} if category == "coding" else {
                    "Coloc (trait)", "Coloc (other)"} if category == "coloc" else set()
                if not isinstance(tags, list) or len(tags) > 3 or any(not isinstance(tag, str) or tag not in allowed for tag in tags):
                    raise HistoryError("saved locus-link labels are invalid")
            if "scope_source" in summary:
                source = summary["scope_source"]
                if category != "association" or not isinstance(source, list) or len(source) > 9 or any(
                        not isinstance(item, str) or len(item) > 1200 for item in source):
                    raise HistoryError("saved phenotype query scope is invalid")
            texts.extend(summary.get("items", []))
        if any(not isinstance(text, str) or len(text) > 1200 for text in texts):
            raise HistoryError("saved gene evidence card text is invalid")


def _validate_evidence_preview(value: Any, category: str) -> None:
    """Persist compact typed previews without admitting arbitrary provider payloads."""
    if category not in {"coloc", "e2g", "expression", "function"} or not isinstance(value, list) or len(value) > 3:
        raise HistoryError("saved evidence preview has an unsupported category or size")
    fields = {"kind", "summary", "source", "evidence_id", "source_ref", "group", "detail"}
    for row in value:
        if not isinstance(row, dict) or not set(row) <= fields or row.get("kind") != category:
            raise HistoryError("saved evidence preview contains unsupported fields")
        for key, limit in {"summary": 1200, "evidence_id": 300, "group": 300, "detail": 1200}.items():
            if key in row and (not isinstance(row[key], str) or len(row[key]) > limit):
                raise HistoryError("saved evidence preview text is invalid")
        source = row.get("source")
        if not isinstance(source, list) or len(source) > 32 or any(
                not isinstance(item, str) or len(item) > 1200 for item in source):
            raise HistoryError("saved evidence preview source is invalid")
        if not isinstance(row.get("source_ref"), str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", row["source_ref"]):
            raise HistoryError("saved evidence preview reference is invalid")


def _validate_qtl_rows(value: Any) -> None:
    """Keep typed lead/member identities without arbitrary provider payloads."""
    if not isinstance(value, list):
        raise HistoryError("saved QTL rows must be a list")
    labels = {"source_selected": "User-selected variant", "ld_proxy": "LD-linked variant", "unknown": "Analyzed variant"}
    measurement_fields = {"pValueMantissa", "pValueExponent", "beta", "standardError"}
    link_fields = {"selected_variant_id", "proxy_variant_id", "r2", "population", "genome_build"}
    for row in value:
        if not isinstance(row, dict) or not set(row) <= _QTL_ROW_FIELDS:
            raise HistoryError("saved QTL row contains unsupported fields")
        for key, limit in _QTL_ROW_TEXT_LIMITS.items():
            if key in row and (not isinstance(row[key], str) or len(row[key]) > limit):
                raise HistoryError("saved QTL row text is invalid")
        if "rsid" in row and not re.fullmatch(r"rs[0-9]+", row["rsid"]):
            raise HistoryError("saved QTL rsID is invalid")
        if "genome_build" in row and row["genome_build"] != "GRCh38":
            raise HistoryError("saved QTL genome build is invalid")
        origin = row.get("variant_origin")
        lead_scope = row.get("measurement_scope") == "credible_set_lead"
        expected_label = "Credible-set lead" if lead_scope else labels.get(origin)
        if origin not in labels or row.get("variant_label", expected_label) != expected_label:
            raise HistoryError("saved QTL variant role is invalid")
        if "measurement_scope" in row and not lead_scope:
            raise HistoryError("saved QTL measurement scope is invalid")
        if lead_scope and (not row.get("cs_lead_variant_id") or row.get("variant_id") != row["cs_lead_variant_id"]):
            raise HistoryError("saved QTL lead identity is inconsistent")
        if not lead_scope and ("cs_lead_variant_id" in row or "locus_anchor_witnesses" in row):
            raise HistoryError("saved QTL lead fields require a typed measurement scope")
        for key in ("variant_id", "locus_variant_id", "cs_lead_variant_id"):
            identifier = row.get(key, "")
            if identifier and not _QTL_VARIANT_ID.fullmatch(identifier):
                raise HistoryError("saved QTL variant identifier is invalid")
        for key in ("context_matched", "legacy_display"):
            if key in row and type(row[key]) is not bool:
                raise HistoryError("saved QTL row flag is invalid")
        measurement = row.get("measurement", {})
        if not isinstance(measurement, dict) or not set(measurement) <= measurement_fields:
            raise HistoryError("saved QTL measurement contains unsupported fields")
        if any(type(number) not in (int, float) or not -1.7976931348623157e308 <= number <= 1.7976931348623157e308
               for number in measurement.values()):
            raise HistoryError("saved QTL measurement must contain finite numbers")
        witnesses = row.get("locus_anchor_witnesses", [])
        if not isinstance(witnesses, list):
            raise HistoryError("saved QTL locus witnesses must be a list")
        for witness in witnesses:
            if not isinstance(witness, dict) or set(witness) != {"variant_id", "locus_variant_id", "variant_origin"}:
                raise HistoryError("saved QTL locus witness contains unsupported fields")
            if not isinstance(witness["variant_origin"], str) or witness["variant_origin"] not in labels:
                raise HistoryError("saved QTL locus witness role is invalid")
            for key in ("variant_id", "locus_variant_id"):
                if not isinstance(witness[key], str) or len(witness[key]) > 120 or not _QTL_VARIANT_ID.fullmatch(witness[key]):
                    raise HistoryError("saved QTL locus witness identifier is invalid")
        if lead_scope and origin != "unknown" and not any(witness["variant_id"] == row["variant_id"]
                                                         and witness["variant_origin"] == origin for witness in witnesses):
            raise HistoryError("saved QTL lead role has no matching locus witness")
        links = row.get("ld_relations", [])
        if not isinstance(links, list):
            raise HistoryError("saved QTL LD relations must be a list")
        for link in links:
            if not isinstance(link, dict) or set(link) != link_fields:
                raise HistoryError("saved QTL LD relation contains unsupported fields")
            for key in ("selected_variant_id", "proxy_variant_id"):
                identifier = link[key]
                if not isinstance(identifier, str) or len(identifier) > 120 or not _QTL_VARIANT_ID.fullmatch(identifier):
                    raise HistoryError("saved QTL LD relation has an invalid variant identifier")
            linked_variants = {witness["variant_id"] for witness in witnesses} if lead_scope else {row.get("variant_id")}
            if link["proxy_variant_id"] not in linked_variants:
                raise HistoryError("saved QTL LD relation does not name the displayed measured variant")
            if type(link["r2"]) not in (int, float) or not 0 <= link["r2"] <= 1:
                raise HistoryError("saved QTL LD relation has an invalid r2")
            if (not isinstance(link["population"], str) or link["population"] not in {"AFR", "AMR", "EAS", "EUR", "SAS"}
                    or link["genome_build"] != "GRCh38"):
                raise HistoryError("saved QTL LD relation has an invalid population or genome build")


def _validate_ranking_evidence(value: Any) -> None:
    """Persist only bounded local snapshot identifiers, never arbitrary URLs."""
    if not isinstance(value, list) or len(value) > 256:
        raise HistoryError("saved ranking evidence must be a bounded list")
    fields = {"run_id", "query_index", "snapshot_id", "model_input_sha256", "messages_sha256", "candidate_count"}
    for row in value:
        if not isinstance(row, dict) or set(row) != fields:
            raise HistoryError("saved ranking evidence contains unsupported fields")
        _identifier(row["run_id"])
        for key in ("snapshot_id", "model_input_sha256", "messages_sha256"):
            if not isinstance(row[key], str) or not re.fullmatch(r"[a-f0-9]{64}", row[key]):
                raise HistoryError("saved ranking evidence has an invalid hash")
        for key in ("query_index", "candidate_count"):
            if type(row[key]) is not int or row[key] < 0:
                raise HistoryError("saved ranking evidence has an invalid count")


def _validate_variant_normalizations(value: Any) -> None:
    """Retain the small public identity receipt, never provider responses."""

    if not isinstance(value, list) or len(value) > 2_000:
        raise HistoryError("saved variant normalizations exceed the identity receipt limit")
    identifier = re.compile(r"([1-9]|1[0-9]|2[0-2]|X|Y|MT)_([1-9][0-9]{0,11})_([ACGT])_([ACGT])")
    seen = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "original_id", "canonical_id", "genome_build", "reason_code",
            "source_p_preserved", "effect_sign_changed",
        }:
            raise HistoryError("saved variant normalization has invalid fields")
        original, canonical = item["original_id"], item["canonical_id"]
        before = identifier.fullmatch(original) if isinstance(original, str) else None
        after = identifier.fullmatch(canonical) if isinstance(canonical, str) else None
        if (not before or not after or original == canonical
                or before.group(1, 2) != after.group(1, 2)
                or before.group(3, 4) != after.group(4, 3)
                or original in seen):
            raise HistoryError("saved variant normalization is not a unique same-coordinate allele reversal")
        if (item["genome_build"] != "GRCh38"
                or item["reason_code"] != "ot_reference_allele_order_conflict"
                or item["source_p_preserved"] is not True or item["effect_sign_changed"] is not False):
            raise HistoryError("saved variant normalization has invalid provenance")
        seen.add(original)


def _validate_selected_regions(value: Any) -> None:
    """Keep exact multi-selection boundaries without storing arbitrary SNP objects."""

    if not isinstance(value, list) or len(value) > 500:
        raise HistoryError("saved selection accepts at most 500 regions")
    total_variants = 0
    for region in value:
        if not isinstance(region, dict) or set(region) != {
            "chromosome", "start", "end", "selected_variants",
        }:
            raise HistoryError("saved selection region has invalid fields")
        chromosome = region["chromosome"]
        if not isinstance(chromosome, str) or not re.fullmatch(
            r"(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)", chromosome, re.IGNORECASE,
        ):
            raise HistoryError("saved selection region has an invalid chromosome")
        start, end = region["start"], region["end"]
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end)):
            raise HistoryError("saved selection region coordinates must be integers")
        if not 0 < start <= end < 2**53:
            raise HistoryError("saved selection region has invalid coordinates")
        identifiers = region["selected_variants"]
        if not isinstance(identifiers, list):
            raise HistoryError("saved selection variants must be an array")
        total_variants += len(identifiers)
        if total_variants > 300_000:
            raise HistoryError("saved selection exceeds the plot variant limit")
        if any(not isinstance(item, str) or not 1 <= len(item) <= 120 for item in identifiers):
            raise HistoryError("saved selection has an invalid variant identifier")


def _safe_value(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_NESTING:
        raise HistoryError("history payload nesting is too deep")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HistoryError("history values must be finite")
        return value
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[credential redacted]", value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item, depth + 1)
            for key, item in value.items() if not _SECRET_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth + 1) for item in value]
    raise HistoryError("history payload contains a non-JSON value")


def _encode(payload: Mapping[str, Any]) -> bytes:
    try:
        raw = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise HistoryError("history payload is not valid JSON") from exc
    if len(raw) > _MAX_PAYLOAD_BYTES:
        raise HistoryError("history entry exceeds 32 MB; start a new conversation")
    return raw


def _decode(payload: bytes, expected_size: int) -> dict[str, Any]:
    if not isinstance(expected_size, int) or not 0 <= expected_size <= _MAX_PAYLOAD_BYTES:
        raise HistoryError("saved history exceeds its size limit")
    # A bounded stream read also protects against a damaged/replaced compressed blob.
    import io

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as handle:
            raw = handle.read(expected_size + 1)
        if len(raw) != expected_size:
            raise HistoryError("saved history failed its size check")
        value = json.loads(raw)
    except (OSError, EOFError, UnicodeError, ValueError) as exc:
        raise HistoryError("saved history could not be decoded") from exc
    if not isinstance(value, dict):
        raise HistoryError("saved history is not an object")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
