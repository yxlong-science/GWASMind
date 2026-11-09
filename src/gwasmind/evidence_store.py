"""Lossless, bounded I/O for complete evidence; model projections are separate.

Large evidence arrays live in immutable, content-addressed JSONL chunks. A small
JSON document binds the chunks to the unchanged run metadata. Readers verify
every byte and restore every record before scientific review. This module does
not select, merge, score, or discard scientific evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audit import canonical_json

FORMAT = "gwasmind-evidence-jsonl-v1"
STORE_KEY = "_evidence_store"
METADATA_KEY = "_metadata_fields"
METADATA_FORMAT = "gwasmind-metadata-field-v1"
SUMMARY_KEY = "_metadata_summary"
INLINE_METADATA_BYTES = 64 * 1024
CHUNK_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_CHUNKS = 16384
MAX_METADATA_BYTES = MAX_CHUNKS * CHUNK_BYTES
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class StoredMetadataDict(dict):
    """Explicit metadata summary retaining its immutable complete-field reference."""
    def __init__(self, value, *, document, field, reference):
        super().__init__(value)
        self.document, self.field, self.reference = document, field, reference
        self.document_identity = _identity(document)
        self[SUMMARY_KEY] = _summary_marker(document, field, reference)

    def __getitem__(self, key):
        if key == "v2g_cards" and getattr(self, "_card_view", None) is not None:
            return self._card_view
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def related(self, field):
        _require(_identity(self.document) == self.document_identity, "Metadata source document changed")
        document = _read_manifest(self.document, MAX_MANIFEST_BYTES)
        references = document.get(METADATA_KEY, {})
        if field in references:
            return _read_metadata_field(self.document, field, references[field], mode="summary")
        _require(field in document, "Related metadata field is missing")
        return document[field]


class StoredMetadataList(list):
    """Display rows whose full source-key details remain in the referenced field."""
    def __init__(self, value, *, document, field, reference):
        super().__init__({**row, SUMMARY_KEY: _summary_marker(document, field, reference)}
                         if isinstance(row, Mapping) else row for row in value)
        self.document, self.field, self.reference = document, field, reference


def _summary_marker(document, field, reference):
    return {"field": field, "complete_value": dict(reference),
            "document": str(Path(document).resolve()), "document_sha256": _hash_file(document)}


class EvidenceStoreError(ValueError):
    """An evidence document is malformed, incomplete, or no longer authentic."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceStoreError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        # Every JSONL row is decoded separately. Share immutable, repeated field
        # names and short labels instead of retaining millions of equal strings.
        key = sys.intern(key)
        if isinstance(value, str) and len(value) <= 256:
            value = sys.intern(value)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                if isinstance(nested, str) and len(nested) <= 256:
                    value[index] = sys.intern(nested)
        _require(key not in result, "Duplicate JSON field in evidence artifact")
        result[key] = value
    return result


def _reject_constant(value):
    raise EvidenceStoreError("Non-finite value in evidence artifact")


def _finite_float(value):
    result = float(value)
    _require(math.isfinite(result), "Non-finite value in evidence artifact")
    return result


def _loads(text):
    return json.loads(text, object_pairs_hook=_unique, parse_constant=_reject_constant,
                      parse_float=_finite_float)


def iter_canonical(value: Any) -> Iterator[str]:
    """Canonical JSON after JSON-key normalization, without copying the tree."""
    from .audit import _json_default
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"),
                              allow_nan=False, default=_json_default, sort_keys=True)
    def bounded_piece(item):
        pending, nodes, chars, normalize = [item], 0, 0, False
        while pending:
            value = pending.pop()
            nodes += 1
            if nodes > 512:
                return None
            if isinstance(value, dict):
                if len(value) + len(pending) > 512:
                    return None
                normalize = normalize or any(not isinstance(key, str) for key in value)
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                if len(value) + len(pending) > 512:
                    return None
                pending.extend(value)
            elif isinstance(value, str):
                chars += len(value)
                if chars > CHUNK_BYTES:
                    return None
        text = encoder.encode(item)
        return encoder.encode(json.loads(text)) if normalize else text
    def parts(item):
        # Let the C encoder handle bounded subtrees in one piece. Iterating one
        # token at a time across millions of fields is unnecessarily slow.
        bounded = bounded_piece(item)
        if bounded is not None:
            yield bounded
            return
        if isinstance(item, dict):
            normalized = {}
            for key, nested in item.items():
                if not isinstance(key, str):
                    key = next(iter(json.loads(encoder.encode({key: None}))))
                _require(key not in normalized, "Conflicting normalized JSON keys")
                normalized[key] = nested
            yield "{"
            for index, key in enumerate(sorted(normalized)):
                if index:
                    yield ","
                yield encoder.encode(key)
                yield ":"
                yield from parts(normalized[key])
            yield "}"
        elif isinstance(item, (list, tuple)):
            yield "["
            for index, nested in enumerate(item):
                if index:
                    yield ","
                yield from parts(nested)
            yield "]"
        else:
            yield encoder.encode(item)
    yield from parts(value)


def _identity(path: Path) -> tuple[int, ...]:
    value = path.stat()
    _require(stat.S_ISREG(value.st_mode), "Evidence artifact must be a regular file")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def atomic_json(path: Path, value: Any, *, compact: bool = True,
                byte_progress: Callable[[int], None] | None = None) -> tuple[int, str]:
    """Serialize without constructing a second complete JSON string in memory."""
    from .audit import _json_default
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, allow_nan=False,
                              indent=2, default=_json_default)
    parts = iter_canonical(value) if compact else encoder.iterencode(value)
    encoded = (part.encode("utf-8") for part in parts)
    return (_atomic_parts(path, encoded, byte_progress=byte_progress) if byte_progress is not None
            else _atomic_parts(path, encoded))


def _atomic_parts(path: Path, parts: Iterable[bytes], *,
                  byte_progress: Callable[[int], None] | None = None) -> tuple[int, str]:
    temporary = None
    digest, size, reported = hashlib.sha256(), 0, 0
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            for part in parts:
                f.write(part)
                digest.update(part)
                size += len(part)
                if byte_progress is not None and size - reported >= CHUNK_BYTES:
                    byte_progress(size)
                    reported = size
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
    return size, digest.hexdigest()


def iter_record_response(value: dict[str, Any], records: tuple | list, *,
                         progress: Callable[[int, int], None] | None = None) -> Iterator[str]:
    """Encode the existing standalone response with one detached record at a time.

    This is a narrow evidence-response entry point, not a JSON default hook.
    It accepts only replayable native EvidenceRecord sequences. Scientific row
    order and the public to_dict deep-copy contract remain unchanged.
    """
    from .models import EvidenceRecord
    _require(type(value) is dict and "records" not in value
             and all(type(key) is str for key in value), "Invalid record response envelope")
    _require(type(records) in (tuple, list), "Record response requires a replayable native sequence")
    yield "{"
    for ordinal, key in enumerate(sorted((*value, "records"))):
        if ordinal:
            yield ","
        yield from iter_canonical(key)
        yield ":"
        if key != "records":
            yield from iter_canonical(value[key])
            continue
        yield "["
        for index, record in enumerate(records):
            _require(type(record) is EvidenceRecord, "Record response contains a non-native evidence row")
            if index:
                yield ","
            yield from iter_canonical(record.to_dict())
            if progress is not None:
                progress(index + 1, len(records))
        yield "]"
    yield "}"


def write_record_response(path: Path, value: dict[str, Any], records: tuple | list, *,
                          progress: Callable[[int, int], None] | None = None,
                          byte_progress: Callable[[int], None] | None = None) -> tuple[int, str]:
    """Atomically save unchanged standalone JSON without a whole-row deep-copy list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _atomic_parts(path, (part.encode("utf-8") for part in
        iter_record_response(value, records, progress=progress)), byte_progress=byte_progress)


def record_response_sha256(value: dict[str, Any], records: tuple | list, *,
                           progress: Callable[[int, int], None] | None = None) -> str:
    """Artifact-free callers receive the same response digest, without materializing it."""
    digest = hashlib.sha256()
    for part in iter_record_response(value, records, progress=progress):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


class LedgerWriter:
    """Write one normalized record at a time, with at most one chunk buffered."""
    def __init__(self, document: Path):
        self.document = document
        self.directory = document.parent / "evidence_chunks"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pending = bytearray()
        self.pending_count = 0
        self.count = 0
        self.digest = hashlib.sha256()
        self.chunks: list[dict[str, Any]] = []

    def append(self, record: Any) -> None:
        if hasattr(record, "to_dict"):
            record = record.to_dict()
        _require(isinstance(record, Mapping), "Evidence row must be a JSON object")
        encoded = (canonical_json(record) + "\n").encode("utf-8")
        _require(len(encoded) <= MAX_RECORD_BYTES, "Single evidence row exceeds processing limit")
        # Reject non-finite/duplicate normalized fields before publishing a chunk.
        _loads(encoded)
        if self.pending and len(self.pending) + len(encoded) > CHUNK_BYTES:
            self.flush()
        self.pending.extend(encoded)
        self.pending_count += 1
        self.count += 1
        self.digest.update(encoded)

    def flush(self) -> None:
        if not self.pending:
            return
        _require(len(self.chunks) < MAX_CHUNKS, "Evidence ledger exceeds chunk processing limit")
        digest = hashlib.sha256(self.pending).hexdigest()
        target = self.directory / (digest + ".jsonl")
        if target.exists():
            _require(_hash_file(target) == digest, "Existing evidence chunk failed verification")
        else:
            _atomic_parts(target, [self.pending])
        self.chunks.append({"path": "evidence_chunks/" + target.name,
            "bytes": len(self.pending), "sha256": digest, "records": self.pending_count})
        self.pending = bytearray()
        self.pending_count = 0

    def finish(self, field: str) -> dict[str, Any]:
        self.flush()
        return {"format": FORMAT, "field": field, "records": self.count,
                "sha256": self.digest.hexdigest(), "chunks": self.chunks}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    before = _identity(path)
    with path.open("rb") as f:
        remaining = before[2]
        while remaining:
            chunk = f.read(min(1024 * 1024, remaining))
            _require(bool(chunk), "Evidence artifact was truncated while reading")
            remaining -= len(chunk)
            digest.update(chunk)
        _require(f.read(1) == b"", "Evidence artifact grew while reading")
    _require(_identity(path) == before, "Evidence artifact changed while reading")
    return digest.hexdigest()


def _read_manifest(path: Path, max_bytes: int) -> dict:
    before = _identity(path)
    _require(before[2] <= max_bytes, "Evidence document metadata exceeds processing limit")
    with path.open("rb") as handle:
        # The ceiling is a validation limit, not an allocation request. Card
        # archives share the much larger metadata ceiling; reading that entire
        # allowance can allocate tens of GB for a small verified file on Windows.
        raw = handle.read(before[2] + 1)
    _require(len(raw) == before[2] and _identity(path) == before,
             "Evidence document changed while reading")
    value = _loads(raw)
    _require(isinstance(value, dict), "Evidence document must be a JSON object")
    return value


def _metadata_path(document: Path, descriptor: Any) -> Path:
    _require(isinstance(descriptor, dict) and set(descriptor) ==
             {"format", "path", "bytes", "sha256"}, "Invalid metadata field descriptor")
    sha = descriptor["sha256"]
    _require(descriptor["format"] == METADATA_FORMAT and isinstance(sha, str)
             and _HEX.fullmatch(sha), "Invalid metadata field format/digest")
    _require(descriptor["path"] == "metadata_fields/" + sha + ".json",
             "Invalid metadata field path")
    root = document.parent.resolve()
    path = (root / descriptor["path"]).resolve()
    _require(path.parent == root / "metadata_fields", "Metadata field path escapes its run")
    _require(type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] <= MAX_METADATA_BYTES,
             "Invalid metadata field byte count")
    _require(_identity(path)[2] == descriptor["bytes"], "Metadata field byte count mismatch")
    _require(_hash_file(path) == sha, "Metadata field digest mismatch")
    return path


def verify_metadata_fields(document: Path, value: Mapping) -> None:
    references = value.get(METADATA_KEY, {})
    _require(isinstance(references, dict), "Invalid metadata field index")
    for field, reference in references.items():
        _require(isinstance(field, str) and field not in {STORE_KEY, METADATA_KEY, "records", "evidence"}
                 and field not in value, "Conflicting metadata field")
        _metadata_path(document, reference)


def _metadata_prefix(path: Path, *, full: bool, reference: Mapping):
    before = _identity(path)
    with path.open(encoding="utf-8") as handle:
        stream = _JSONStream(handle)
        stream.take("{")
        _require(stream.value() == "summary", "Malformed metadata field summary")
        stream.take(":")
        summary = stream.value()
        stream.take(",")
        _require(stream.value() == "value", "Malformed metadata field value")
        stream.take(":")
        value = stream.tree() if full or summary is None else None
        if full or summary is None:
            stream.take("}")
            _require(stream.peek() == "", "Trailing data in metadata field")
    _require(_hash_file(path) == reference["sha256"] and _identity(path) == before,
             "Metadata field changed while reading")
    return summary, value


def _read_metadata_field(document: Path, field: str, reference: Mapping, *, mode: str):
    path = _metadata_path(document, reference)
    summary, value = _metadata_prefix(path, full=mode == "full", reference=reference)
    if mode == "full" or summary is None:
        return value
    if isinstance(summary, dict):
        return StoredMetadataDict(summary, document=document, field=field, reference=reference)
    _require(isinstance(summary, list) and summary and all(isinstance(row, dict) for row in summary),
             "Unsupported metadata summary shape")
    return StoredMetadataList(summary, document=document, field=field, reference=reference)


def metadata_summary_related(value: Mapping, field: str):
    """Resolve sibling summaries even after an ordinary JSON round trip."""
    marker = value.get(SUMMARY_KEY)
    if marker is None:
        return None
    document, _reference, _summary = _validated_summary(value)
    manifest = _read_manifest(document, MAX_MANIFEST_BYTES)
    references = manifest.get(METADATA_KEY, {})
    if field in references:
        return _read_metadata_field(document, field, references[field], mode="summary")
    _require(field in manifest, "Related metadata field is missing")
    return manifest[field]


def _validated_summary(value):
    if isinstance(value, Mapping):
        marker = value.get(SUMMARY_KEY)
        summary = {key: item for key, item in value.items() if key != SUMMARY_KEY}
    elif isinstance(value, list) and value and isinstance(value[0], Mapping):
        marker = value[0].get(SUMMARY_KEY)
        _require(all(isinstance(row, Mapping) and row.get(SUMMARY_KEY) == marker for row in value),
                 "Conflicting metadata summary references")
        summary = [{key: item for key, item in row.items() if key != SUMMARY_KEY} for row in value]
    else:
        raise EvidenceStoreError("Invalid detached metadata summary")
    _require(isinstance(marker, dict) and set(marker) ==
             {"field", "complete_value", "document", "document_sha256"}, "Invalid metadata summary reference")
    _require(isinstance(marker["document"], str) and Path(marker["document"]).is_absolute(),
             "Invalid metadata source document")
    document = Path(marker["document"])
    _require(_hash_file(document) == marker["document_sha256"], "Metadata source manifest digest mismatch")
    manifest = _read_manifest(document, MAX_MANIFEST_BYTES)
    field, reference = marker["field"], marker["complete_value"]
    _require(isinstance(field, str) and manifest.get(METADATA_KEY, {}).get(field) == reference,
             "Metadata summary does not match source manifest")
    path = _metadata_path(document, reference)
    expected, _unused = _metadata_prefix(path, full=False, reference=reference)
    _require(expected is not None and canonical_json(summary) == canonical_json(expected),
             "Metadata summary changed; complete details cannot be silently rewritten")
    return document, reference, summary


def _contains_summary(value) -> bool:
    if isinstance(value, Mapping):
        return SUMMARY_KEY in value
    return isinstance(value, (list, tuple)) and any(isinstance(row, Mapping) and SUMMARY_KEY in row for row in value)


def _known_summary_without_reference(field, value) -> bool:
    if field == "candidate_audit" and isinstance(value, Mapping):
        for gene in value.get("genes", ()):
            if not isinstance(gene, Mapping):
                continue
            if gene.get("source_details_available") is True and "sources" not in gene:
                return True
            expression = gene.get("expression_specificity")
            if (isinstance(expression, Mapping) and expression.get("group_details_available") is True
                    and "groups" not in expression):
                return True
    if field == "candidate_pool" and isinstance(value, (list, tuple)):
        return any(isinstance(row, Mapping) and row.get("source_record_details_available") is True
                   and "source_record_keys" not in row for row in value)
    return False


def _copy_metadata_reference(document: Path, value) -> dict:
    source_document, reference, _summary = _validated_summary(value)
    from .card_store import KEY as CARD_KEY, StoredCardList, store_cards
    if isinstance(_summary, dict) and CARD_KEY in _summary:
        store_cards(document, StoredCardList(source_document, _summary[CARD_KEY]))
    source = _metadata_path(source_document, reference)
    target = document.parent / reference["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != source.resolve():
        if target.exists():
            _metadata_path(document, reference)
        else:
            # Immutable content may be reused on the same volume. Copy otherwise.
            try:
                os.link(source, target)
            except OSError:
                with source.open("rb") as handle:
                    _atomic_parts(target, iter(lambda: handle.read(1024 * 1024), b""))
            _metadata_path(document, reference)
    return dict(reference)


def _prepare_metadata(document: Path, value: Mapping, *, field: str, summaries: Mapping | None = None) -> dict:
    _require(METADATA_KEY not in value, "Unexpanded metadata fields cannot be rewritten")
    result, references = {}, {}
    summaries = summaries or {}
    for key, item in value.items():
        if key == field:
            continue
        if _contains_summary(item):
            references[key] = _copy_metadata_reference(document, item)
            # Prevent attaching a verified summary under another semantic field.
            marker = item.get(SUMMARY_KEY) if isinstance(item, Mapping) else item[0][SUMMARY_KEY]
            _require(marker["field"] == key, "Metadata summary field mismatch")
            continue
        if key == "candidate_audit":
            from .card_store import archive_audit, KEY as CARD_KEY
            item = archive_audit(document, item)
            if key in summaries and CARD_KEY in item:
                summaries = {**summaries, key: {
                    **{name: content for name, content in summaries[key].items() if name != "v2g_cards"},
                    CARD_KEY: item[CARD_KEY]}}
        _require(not _known_summary_without_reference(key, item),
                 "Metadata summary lost its complete-field reference")
        # Stop measuring once a value belongs in a sidecar; never allocate a
        # second full representation just to choose its storage encoding.
        size = 0
        for part in iter_canonical(item):
            size += len(part.encode("utf-8"))
            if size > INLINE_METADATA_BYTES:
                break
        if size <= INLINE_METADATA_BYTES:
            result[key] = item
            continue
        summary = summaries.get(key)
        if summary is not None:
            _require(isinstance(summary, dict) or isinstance(summary, list) and summary
                     and all(isinstance(row, dict) for row in summary), "Unsupported metadata summary shape")
            _require(not _contains_summary(summary), "Nested metadata summary reference")
            _require(len(canonical_json(summary).encode("utf-8")) <= MAX_MANIFEST_BYTES,
                     "Metadata summary exceeds processing limit")
        directory = document.parent / "metadata_fields"
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / (uuid4().hex + ".staging")
        try:
            size, sha = atomic_json(temporary, {"summary": summary, "value": item})
            _require(size <= MAX_METADATA_BYTES, "Metadata field exceeds processing limit")
            target = directory / (sha + ".json")
            if target.exists():
                _require(_hash_file(target) == sha, "Existing metadata field failed verification")
            else:
                os.replace(temporary, target)
            references[key] = {"format": METADATA_FORMAT, "path": "metadata_fields/" + target.name,
                               "bytes": size, "sha256": sha}
        finally:
            temporary.unlink(missing_ok=True)
    if references:
        result[METADATA_KEY] = references
    return result


def _write_manifest(path: Path, document: Mapping) -> tuple[int, str]:
    encoded = canonical_json(document).encode("utf-8")
    _require(len(encoded) <= MAX_MANIFEST_BYTES, "Evidence document metadata exceeds processing limit")
    return _atomic_parts(path, [encoded])


def write_document(path: Path, value: Mapping[str, Any], *, field: str,
                   externalize: bool = True, metadata_summaries: Mapping | None = None) -> tuple[int, str]:
    """Store complete rows separately; small legacy-shaped documents stay inline."""
    _require(field in {"evidence", "records"} and STORE_KEY not in value,
             "Invalid evidence document field")
    rows = value.get(field, ())
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _prepare_metadata(path, value, field=field, summaries=metadata_summaries)
    # Ordinary small analyses keep the established standalone JSON contract.
    # The count only selects an encoding. It never limits the evidence scope.
    pending_rows, encoded_bytes = [], 0
    iterator = iter(rows)
    if not externalize or (hasattr(rows, "__len__") and len(rows) < 1000):
        for row in iterator:
            row = row.to_dict() if hasattr(row, "to_dict") else row
            row_size = len(canonical_json(row).encode("utf-8"))
            _require(row_size <= MAX_RECORD_BYTES, "Single evidence row exceeds processing limit")
            encoded_bytes += row_size
            pending_rows.append(row)
            if externalize and encoded_bytes > CHUNK_BYTES:
                break
        else:
            return _write_manifest(path, {**document, field: pending_rows})
    writer = LedgerWriter(path)
    for row in pending_rows:
        writer.append(row)
    for row in iterator:
        writer.append(row)
    descriptor = writer.finish(field)
    document[STORE_KEY] = descriptor
    return _write_manifest(path, document)


def _reuse_verified_ledger(path: Path, value: Mapping[str, Any], *, field: str,
                           source: Path, source_sha256: str,
                           original_records: tuple) -> tuple[int, str] | None:
    """Rebind a just-written ledger after the Agent proves identical row objects.

    This private fast path does not infer equality from record counts. Its sole
    caller is the synchronous finalizer, using the very same frozen-record tuple
    for retrieval and result. Hash every retained byte, but do not re-encode or
    re-parse rows already validated by their first write. Inline sources return
    None so the caller retains the ordinary writer.
    """
    from .models import EvidenceRecord
    rows = value.get(field)
    _require(type(rows) is tuple and rows is original_records
             and all(type(row) is EvidenceRecord for row in rows),
             "Evidence reuse requires the same frozen-record tuple")
    path, source = Path(path), Path(source)
    _require(field == "evidence" and path.parent.resolve() == source.parent.resolve()
             and path.resolve() != source.resolve(), "Evidence reuse escaped its native run")
    _require(_hash_file(source) == source_sha256, "Evidence source manifest changed")
    original = _read_manifest(source, MAX_MANIFEST_BYTES)
    if STORE_KEY not in original:
        return None
    _require("records" not in original, "Reusable source contains conflicting ledgers")
    descriptor = original[STORE_KEY]
    _require(isinstance(descriptor, dict) and set(descriptor) ==
             {"format", "field", "records", "sha256", "chunks"}
             and descriptor["format"] == FORMAT and descriptor["field"] == "records",
             "Invalid reusable evidence ledger")
    _require(type(descriptor["records"]) is int
             and descriptor["records"] == len(rows), "Reusable evidence count mismatch")
    _require(isinstance(descriptor["sha256"], str) and _HEX.fullmatch(descriptor["sha256"]),
             "Invalid reusable evidence digest")
    chunks = descriptor["chunks"]
    _require(isinstance(chunks, list) and len(chunks) <= MAX_CHUNKS, "Invalid reusable chunks")
    verify_metadata_fields(source, original)
    root = source.parent.resolve()
    digest, total = hashlib.sha256(), 0
    for chunk in chunks:
        _require(isinstance(chunk, dict) and set(chunk) == {"path", "bytes", "sha256", "records"},
                 "Invalid reusable chunk descriptor")
        sha, size, count = chunk["sha256"], chunk["bytes"], chunk["records"]
        _require(isinstance(sha, str) and _HEX.fullmatch(sha)
                 and chunk["path"] == "evidence_chunks/" + sha + ".jsonl",
                 "Invalid reusable chunk identity")
        retained = (root / chunk["path"]).resolve()
        _require(retained.parent == root / "evidence_chunks", "Reusable chunk escaped its run")
        _require(type(size) is int and 0 < size <= max(CHUNK_BYTES, MAX_RECORD_BYTES)
                 and type(count) is int and count > 0, "Invalid reusable chunk bounds")
        before = _identity(retained)
        _require(before[2] == size, "Reusable chunk byte count mismatch")
        with retained.open("rb") as handle:
            encoded = handle.read(size + 1)
        _require(len(encoded) == size and _identity(retained) == before
                 and hashlib.sha256(encoded).hexdigest() == sha and encoded.endswith(b"\n")
                 and encoded.count(b"\n") == count, "Reusable chunk verification failed")
        digest.update(encoded)
        total += count
    _require(total == descriptor["records"] and digest.hexdigest() == descriptor["sha256"],
             "Reusable ledger count/digest mismatch")
    _require(_hash_file(source) == source_sha256, "Evidence source manifest changed")
    document = _prepare_metadata(path, value, field=field, summaries=None)
    document[STORE_KEY] = {**descriptor, "field": field}
    return _write_manifest(path, document)


def read_document(path: Path, *, field: str, materialize: bool = True,
                  max_bytes: int = MAX_MANIFEST_BYTES, progress=None,
                  metadata_mode: str = "full") -> dict[str, Any]:
    _require(metadata_mode in {"full", "summary"}, "Invalid metadata read mode")
    document = _read_manifest(path, max_bytes)
    references = document.get(METADATA_KEY, {})
    _require(isinstance(references, dict), "Invalid metadata field index")
    for key, reference in references.items():
        _require(key not in document and key not in {STORE_KEY, METADATA_KEY, "records", "evidence"},
                 "Conflicting metadata field")
        document[key] = _read_metadata_field(path, key, reference, mode=metadata_mode)
    document.pop(METADATA_KEY, None)
    from .card_store import restore_cards
    restore_cards(path, document.get("candidate_audit"))
    if STORE_KEY in document:
        _require(field not in document, "Evidence document contains two conflicting ledgers")
        rows = iter_records(path, document[STORE_KEY], expected_field=field, progress=progress)
        document[field] = list(rows) if materialize else rows
        if materialize:
            document.pop(STORE_KEY)
    return document


def iter_records(document_path: Path, descriptor: Any, *, expected_field: str,
                 progress=None) -> Iterator[dict]:
    _require(isinstance(descriptor, dict) and set(descriptor) ==
             {"format", "field", "records", "sha256", "chunks"}, "Invalid evidence ledger descriptor")
    _require(descriptor["format"] == FORMAT and descriptor["field"] == expected_field,
             "Evidence ledger format/field mismatch")
    expected_count = descriptor["records"]
    _require(type(expected_count) is int and expected_count >= 0, "Invalid evidence record count")
    _require(isinstance(descriptor["sha256"], str) and _HEX.fullmatch(descriptor["sha256"]),
             "Invalid evidence ledger digest")
    chunks = descriptor["chunks"]
    _require(isinstance(chunks, list) and len(chunks) <= MAX_CHUNKS,
             "Invalid evidence chunk list")
    root = document_path.parent.resolve()
    total, digest = 0, hashlib.sha256()
    if progress:
        progress({"status": "loading", "operation": "loading", "records": 0, "total_records": expected_count,
                  "chunks_completed": 0, "total_chunks": len(chunks)})
    for chunk_index, chunk in enumerate(chunks):
        _require(isinstance(chunk, dict) and set(chunk) == {"path", "bytes", "sha256", "records"},
                 "Invalid evidence chunk descriptor")
        relative, sha = chunk["path"], chunk["sha256"]
        _require(isinstance(sha, str) and _HEX.fullmatch(sha), "Invalid evidence chunk digest")
        _require(relative == "evidence_chunks/" + sha + ".jsonl",
                 "Invalid evidence chunk path")
        path = (root / relative).resolve()
        _require(path.parent == root / "evidence_chunks", "Evidence chunk path escapes its run")
        size, count = chunk["bytes"], chunk["records"]
        _require(type(size) is int and 0 < size <= max(CHUNK_BYTES, MAX_RECORD_BYTES)
                 and type(count) is int and count > 0, "Invalid evidence chunk size/count")
        before = _identity(path)
        _require(before[2] == size, "Evidence chunk byte count mismatch")
        with path.open("rb") as f:
            data = f.read(size + 1)
        _require(len(data) == size and _identity(path) == before,
                 "Evidence chunk changed while reading")
        _require(hashlib.sha256(data).hexdigest() == sha and data.endswith(b"\n"),
                 "Evidence chunk digest or framing mismatch")
        digest.update(data)
        rows = data.splitlines()
        _require(len(rows) == count, "Evidence chunk record count mismatch")
        for encoded in rows:
            _require(len(encoded) + 1 <= MAX_RECORD_BYTES, "Evidence row exceeds processing limit")
            row = _loads(encoded)
            _require(isinstance(row, dict), "Evidence row must be a JSON object")
            total += 1
            yield row
        if progress:
            progress({"status": "loading", "operation": "loading", "records": total, "total_records": expected_count,
                      "chunks_completed": chunk_index + 1, "total_chunks": len(chunks)})
    _require(total == expected_count and digest.hexdigest() == descriptor["sha256"],
             "Evidence ledger count/digest mismatch")
    if progress:
        # Reading and verifying finish together only after every chunk and the
        # aggregate count/digest have passed. Keep both public operation rows
        # explicit so a generic completion cannot strand "loading" as running.
        for operation in ("loading", "verifying"):
            progress({"status": "completed", "operation": operation,
                      "records": total, "total_records": expected_count,
                      "chunks_completed": len(chunks), "total_chunks": len(chunks)})


class _JSONStream:
    """Incremental standard JSON decoder for legacy top-level evidence arrays."""
    def __init__(self, handle):
        self.handle, self.buffer, self.pos, self.eof = handle, "", 0, False
        self.decoder = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_reject_constant,
                                        parse_float=_finite_float)

    def fill(self):
        chunk = self.handle.read(1024 * 1024)
        self.buffer = self.buffer[self.pos:] + chunk
        self.pos = 0
        self.eof = not chunk

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos] in " \t\r\n":
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ""
            self.fill()

    def take(self, expected):
        _require(self.peek() == expected, "Malformed legacy evidence JSON")
        self.pos += 1

    def value(self, maximum=MAX_MANIFEST_BYTES):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                # A number at a buffer boundary may continue in the next chunk.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                if (type(value) in {int, float} and end < len(self.buffer)
                        and self.buffer[end] in ".eE+-0123456789" and not self.eof):
                    self.fill()
                    continue
                _require(len(self.buffer[self.pos:end].encode("utf-8")) <= maximum,
                         "Legacy evidence value exceeds processing limit")
                self.pos = end
                return value
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise EvidenceStoreError("Malformed legacy evidence JSON") from exc
                _require(len(self.buffer[self.pos:]) <= maximum, "Legacy evidence value exceeds processing limit")
                self.fill()

    def tree(self):
        """Decode a complete metadata value without buffering its complete JSON.

        Small objects use the standard decoder; only containers crossing the
        current read buffer are traversed. Every member and original scalar is
        retained, including fields that the everyday summary does not expose.
        """
        first = self.peek()
        if first not in {"{", "["}:
            return self.value(MAX_METADATA_BYTES)
        try:
            value, end = self.decoder.raw_decode(self.buffer, self.pos)
        except json.JSONDecodeError:
            pass
        else:
            self.pos = end
            return value
        opening, closing = ("{", "}") if first == "{" else ("[", "]")
        self.take(opening)
        result = {} if first == "{" else []
        while self.peek() != closing:
            if first == "{":
                key = self.value()
                _require(isinstance(key, str) and key not in result, "Duplicate metadata field")
                self.take(":")
                result[sys.intern(key)] = self.tree()
            else:
                result.append(self.tree())
            if self.peek() == closing:
                break
            self.take(",")
            _require(self.peek() != closing, "Trailing comma in metadata field")
        self.take(closing)
        return result


def scan_legacy_document(path: Path, *, field: str,
                         on_record: Callable[[dict], None]) -> dict[str, Any]:
    """Read all legacy bytes, preserving every field without a giant JSON string."""
    before = _identity(path)
    metadata: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        stream = _JSONStream(handle)
        stream.take("{")
        while stream.peek() != "}":
            key = stream.value()
            _require(isinstance(key, str) and key not in metadata, "Duplicate legacy evidence field")
            stream.take(":")
            if key == field:
                metadata[key] = None
                stream.take("[")
                while stream.peek() != "]":
                    row = stream.value(MAX_RECORD_BYTES)
                    _require(isinstance(row, dict), "Legacy evidence row must be an object")
                    on_record(row)
                    if stream.peek() == "]":
                        break
                    stream.take(",")
                    _require(stream.peek() != "]", "Trailing comma in evidence array")
                stream.take("]")
            else:
                metadata[key] = stream.tree()
            if stream.peek() == "}":
                break
            stream.take(",")
            _require(stream.peek() != "}", "Trailing comma in evidence object")
        stream.take("}")
        _require(stream.peek() == "", "Trailing data in evidence document")
    _require(_identity(path) == before, "Legacy evidence artifact changed while reading")
    _require(field in metadata and STORE_KEY not in metadata and METADATA_KEY not in metadata,
             "Missing legacy evidence array")
    metadata.pop(field)
    return metadata


def import_legacy_document(source: Path, destination: Path, *, field: str, progress=None) -> dict[str, Any]:
    """Create a separate recoverable copy. Never alter the historical artifact."""
    _require(source.resolve() != destination.resolve() and not destination.exists(),
             "Imported evidence destination must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = _identity(source)
    source_sha = _hash_file(source)
    writer = LedgerWriter(destination)
    if progress:
        progress({"status": "importing", "operation": "organizing", "records": 0, "total_records": None})
    def append(row):
        writer.append(row)
        if progress and writer.count % 1024 == 0:
            progress({"status": "importing", "operation": "organizing", "records": writer.count, "total_records": None})
    metadata = scan_legacy_document(source, field=field, on_record=append)
    descriptor = writer.finish(field)
    _require(_identity(source) == before, "Legacy evidence artifact changed during import")
    from .candidates import candidate_metadata_summaries
    summaries = candidate_metadata_summaries(metadata.get("candidate_audit"), metadata.get("candidate_pool", []))
    metadata = _prepare_metadata(destination, metadata, field=field, summaries=summaries)
    metadata[STORE_KEY] = descriptor
    encoded = canonical_json(metadata).encode("utf-8")
    _require(len(encoded) <= MAX_MANIFEST_BYTES, "Imported evidence metadata exceeds limit")
    _atomic_parts(destination, [encoded])
    receipt = {"source": str(source.resolve()), "source_bytes": before[2],
        "source_sha256": source_sha, "destination": str(destination.resolve()),
        "destination_sha256": hashlib.sha256(encoded).hexdigest(),
        "records": descriptor["records"], "ledger_sha256": descriptor["sha256"]}
    atomic_json(destination.with_suffix(".import.json"), receipt)
    if progress:
        progress({"status": "completed", "operation": "organizing", "records": writer.count,
                  "total_records": writer.count})
    return receipt


def export_document(source: Path, destination: Path, *, field: str) -> Path:
    """Create a standalone full JSON download, without loading all rows."""
    raw_manifest = _read_manifest(source, MAX_MANIFEST_BYTES)
    document = read_document(source, field=field, materialize=False)
    descriptor = document.pop(STORE_KEY, None)
    if descriptor is None and METADATA_KEY not in raw_manifest:
        return source
    rows = document.pop(field)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def parts():
        yield b"{"
        for key, value in sorted(document.items()):
            yield (canonical_json(key) + ":").encode("utf-8")
            for part in iter_canonical(value):
                yield part.encode("utf-8")
            yield b"," 
        yield (canonical_json(field) + ":[").encode("utf-8")
        first = True
        for row in rows:
            if not first:
                yield b","
            first = False
            yield canonical_json(row).encode("utf-8")
        yield b"]}"
    _atomic_parts(destination, parts())
    return destination
