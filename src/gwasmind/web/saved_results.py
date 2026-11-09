"""Small private snapshots referring to immutable, fully verified native results.

HTTP results and standalone exports retain their full scientific contents. These
receipts avoid serializing the same evidence ledger again for job bookkeeping.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..configuration import parameter
from ..evidence_store import METADATA_KEY, STORE_KEY, read_document
from ..evidence_store import _read_manifest as _read_evidence_manifest
from .result_processing import (
    ResultProcessingReceiptError,
    _verify_result_bytes,
    verify_chunked_result_receipt,
)

FORMAT_KEY = "_saved_result_format"
REFERENCE_KEY = "_saved_agent_result"
FORMAT_VERSION = "native-result-references-v1"


def _require(condition):
    if not condition:
        raise ResultProcessingReceiptError("invalid saved-result reference")


def _relative(root, path):
    relative = Path(path).resolve().relative_to(Path(root).resolve())
    _require(bool(relative.parts))
    return relative.as_posix()


def _bound_file(root, value):
    _require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"})
    raw = value["path"]
    _require(isinstance(raw, str) and not Path(raw).is_absolute() and ".." not in Path(raw).parts)
    path = (root / raw).resolve()
    _require(root in path.parents and type(value["bytes"]) is int and value["bytes"] > 0)
    _require(isinstance(value["sha256"], str) and len(value["sha256"]) == 64)
    _verify_result_bytes(path, expected_bytes=value["bytes"], expected_sha256=value["sha256"])
    return path


def make_result_reference(*, root, result_path, input_path, value, expected_run_id,
                          expected_mode, expected_release):
    """Reference an already-validated small native manifest; never its expansion."""
    def receipt(path):
        path = Path(path)
        size = path.stat().st_size
        _require(0 < size <= parameter("execution.max_result_json_bytes"))
        with path.open("rb") as handle:
            raw = handle.read(size + 1)
        _require(len(raw) == size)
        return {"path": _relative(root, path), "bytes": size,
                "sha256": hashlib.sha256(raw).hexdigest()}
    return {REFERENCE_KEY: {
        "result": receipt(result_path), "input": receipt(input_path),
        "run_id": expected_run_id, "locus_id": value["locus_id"],
        "mode": expected_mode, "release": expected_release,
        "data_release": value["data_release"], "status": value["status"],
        "evidence_records": len(value["evidence"]),
    }, "status": value["status"]}


def read_saved_results(path, *, metadata_mode="summary"):
    """Restore every record from new receipts, or read legacy inline snapshots."""
    path = Path(path).resolve()
    snapshot = _read_evidence_manifest(path, parameter("execution.max_result_json_bytes"))
    return restore_result_snapshot(snapshot, root=path.parent, metadata_mode=metadata_mode)


def restore_result_snapshot(snapshot, *, root, metadata_mode="summary"):
    """Resolve an owned snapshot using the same verified on-disk references.

    The caller owns the snapshot; this function does not retain restored records.
    Capturing it before leaving the manager lock keeps retrieval/ranking snapshots
    separate even when ranking resumes while the earlier result is being read.
    """
    root = Path(root).resolve()
    if FORMAT_KEY not in snapshot:
        rows = snapshot.get("results")
        _require(not isinstance(rows, list) or not any(
            isinstance(row, dict) and REFERENCE_KEY in row for row in rows
        ))
        return snapshot
    _require(snapshot.pop(FORMAT_KEY) == FORMAT_VERSION)
    _require(isinstance(snapshot.get("results"), list))
    restored = []
    for index, row in enumerate(snapshot["results"]):
        _require(isinstance(row, dict))
        if REFERENCE_KEY not in row:
            # Explicitly inline legacy/source-only results remain inline.
            restored.append(row)
            continue
        _require(set(row) == {REFERENCE_KEY, "status"})
        ref = row[REFERENCE_KEY]
        _require(isinstance(ref, dict) and set(ref) == {
            "result", "input", "run_id", "locus_id", "mode", "release",
            "data_release", "status", "evidence_records",
        })
        _require(all(isinstance(ref.get(key), str) and ref[key]
                     for key in ("run_id", "locus_id", "mode", "release", "data_release", "status")))
        result_path = _bound_file(root, ref.get("result"))
        input_path = _bound_file(root, ref.get("input"))
        query = _read_evidence_manifest(input_path, parameter("execution.max_result_json_bytes"))
        document = _read_evidence_manifest(result_path, parameter("execution.max_result_json_bytes"))
        _require(ref["run_id"] == f"{snapshot.get('run_id')}-q{index + 1:03d}")
        _require(ref.get("mode") == snapshot.get("mode") and ref.get("release") == snapshot.get("release"))
        _require(query.get("locus_id") == ref.get("locus_id"))
        if STORE_KEY in document or METADATA_KEY in document:
            verify_chunked_result_receipt(result_path, query_value=query,
                expected_run_id=ref["run_id"], expected_mode=ref["mode"],
                expected_release=ref["data_release"])
        value = read_document(result_path, field="evidence", metadata_mode=metadata_mode)
        _require(value.get("run_id") == ref.get("run_id") and value.get("locus_id") == ref.get("locus_id"))
        _require(value.get("data_mode") == ref.get("mode") and value.get("data_release") == ref.get("data_release"))
        _require(value.get("status") == ref.get("status") == row["status"])
        _require(type(ref.get("evidence_records")) is int
                 and len(value.get("evidence", [])) == ref["evidence_records"])
        restored.append(value)
    snapshot["results"] = restored
    return snapshot
