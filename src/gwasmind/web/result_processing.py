"""Verify retained oversized results without loading their scientific contents."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..audit import sha256_json
from ..configuration import parameter
from ..processing_scope import RESULT_PROCESSING_LIMIT_MESSAGE
from ..serialization import locus_query_from_dict

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ResultProcessingReceiptError(ValueError):
    """The retained result could not be bound to this exact completed input."""


def _require(condition: bool) -> None:
    if not condition:
        raise ResultProcessingReceiptError("invalid retained-result receipt")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value)
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ResultProcessingReceiptError("non-finite retained-result receipt")


def _finite_float(value: str) -> float:
    result = float(value)
    _require(math.isfinite(result))
    return result


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    _require(stat.S_ISREG(value.st_mode))
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_manifest(path: Path) -> dict[str, Any]:
    maximum = parameter("execution.max_result_manifest_json_bytes")
    before = _file_identity(path.stat())
    _require(before[2] <= maximum)
    with path.open("rb") as handle:
        opened = _file_identity(os.fstat(handle.fileno()))
        # Windows stat/fstat can report different creation times after an
        # atomic replacement. Bind inode/device/size/mtime across APIs, then
        # require each API's complete identity (including ctime) to stay fixed.
        _require(opened[:4] == before[:4])
        encoded = handle.read(maximum + 1)
        _require(len(encoded) <= maximum and len(encoded) == before[2])
        _require(_file_identity(os.fstat(handle.fileno())) == opened)
    _require(_file_identity(path.stat()) == before)
    value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant, parse_float=_finite_float)
    _require(isinstance(value, dict))
    return value


def _verify_result_bytes(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    """Read at most the verified file size plus one byte, using a bounded buffer."""
    chunk_bytes = parameter("execution.result_hash_chunk_bytes")
    before = _file_identity(path.stat())
    _require(before[2] == expected_bytes)
    digest = hashlib.sha256()
    remaining = expected_bytes
    with path.open("rb") as handle:
        opened = _file_identity(os.fstat(handle.fileno()))
        _require(opened[:4] == before[:4])
        while remaining:
            chunk = handle.read(min(chunk_bytes, remaining))
            _require(bool(chunk))
            remaining -= len(chunk)
            digest.update(chunk)
        _require(handle.read(1) == b"")
        _require(_file_identity(os.fstat(handle.fileno())) == opened)
    _require(_file_identity(path.stat()) == before)
    _require(digest.hexdigest() == expected_sha256)


def verify_chunked_result_receipt(
    result_path: Path, *, query_value: Mapping[str, Any], expected_run_id: str,
    expected_mode: str, expected_release: str, source_only_bootstrap_failure: bool = False,
) -> None:
    """Bind a small evidence manifest to this completed worker's exact input."""
    query = locus_query_from_dict(dict(query_value), allow_analysis_scope=True)
    query.validate()
    root = result_path.parent.resolve()
    _require(result_path.name == "result.json" and root.name == expected_run_id)
    manifest = _read_manifest(root / "run_manifest.json")
    _require(manifest.get("run_id") == expected_run_id)
    _require(manifest.get("input_hash") == sha256_json(asdict(query)))
    _require(manifest.get("mode") == expected_mode)
    release = manifest.get("data_release")
    if source_only_bootstrap_failure:
        # The caller has already verified the exit-4 bootstrap receipt against
        # this exact input. Schema discovery failed before a release could be
        # observed; only this empty failure artifact may retain the requested
        # release placeholder. All input, path and byte bindings still apply.
        _require(expected_mode == "live" and release == expected_release)
        result = _read_manifest(result_path)
        _require(result.get("status") == "SERVICE_UNAVAILABLE"
                 and result.get("evidence") == [] and result.get("ranked_genes") == []
                 and result.get("run_id") == expected_run_id
                 and result.get("data_mode") == expected_mode
                 and result.get("data_release") == expected_release)
    elif expected_mode == "live" and expected_release.casefold() == "live":
        _require(isinstance(release, str) and bool(release.strip())
                 and release.casefold() not in {"live", "unknown", "unverified"})
    else:
        _require(release == expected_release)
    artifact = manifest.get("result_artifact")
    _require(isinstance(artifact, dict) and artifact.get("encoding") == "canonical-json-utf8")
    _require(isinstance(artifact.get("path"), str)
             and Path(artifact["path"]).resolve() == result_path.resolve())
    size, digest = artifact.get("bytes"), artifact.get("byte_sha256")
    _require(type(size) is int and size > 0 and isinstance(digest, str)
             and _SHA256.fullmatch(digest) is not None and artifact.get("canonical_sha256") == digest)
    _verify_result_bytes(result_path, expected_bytes=size, expected_sha256=digest)


def source_only_oversized_result(
    result_path: Path,
    *,
    query_value: Mapping[str, Any],
    expected_run_id: str,
    expected_mode: str,
    expected_release: str,
    return_code: int,
    defer_ranking: bool,
    limit_bytes: int,
) -> dict[str, Any]:
    """Return a processing limitation, never a scientific review of the big result.

    The worker writes its completed manifest only after the complete result is
    atomically stored. Its input and byte hashes bind that artifact to this run.
    Only the shared deferred workflow may use this source-only result; ordinary
    native callers still require their complete result to fit the parsing cap.
    """
    _require(type(return_code) is int and return_code == 0 and defer_ranking is True)
    _require(type(limit_bytes) is int and limit_bytes > 0)
    query = locus_query_from_dict(dict(query_value), allow_analysis_scope=True)
    query.validate()
    run_root = result_path.parent.resolve()
    _require(result_path.name == "result.json" and run_root.name == expected_run_id)
    resolved_result = result_path.resolve()
    manifest_path = result_path.with_name("run_manifest.json").resolve()
    _require(resolved_result.parent == run_root and manifest_path.parent == run_root)
    manifest = _read_manifest(manifest_path)
    _require(manifest.get("run_id") == expected_run_id)
    _require(manifest.get("input_hash") == sha256_json(asdict(query)))
    _require(expected_mode in {"live", "frozen-safe"} and manifest.get("mode") == expected_mode)
    release = manifest.get("data_release")
    _require(isinstance(release, str) and bool(release) and release == release.strip())
    if expected_mode == "live" and expected_release.casefold() == "live":
        _require(release.casefold() not in {"live", "unknown", "unverified"})
    else:
        _require(release == expected_release)
    artifact = manifest.get("result_artifact")
    _require(isinstance(artifact, dict))
    _require(artifact.get("encoding") == "canonical-json-utf8")
    artifact_path = artifact.get("path")
    _require(isinstance(artifact_path, str) and Path(artifact_path).is_absolute())
    _require(Path(artifact_path).resolve() == resolved_result)
    size = artifact.get("bytes")
    _require(type(size) is int and size > limit_bytes)
    digest = artifact.get("byte_sha256")
    _require(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None)
    _require(artifact.get("canonical_sha256") == digest)
    _verify_result_bytes(resolved_result, expected_bytes=size, expected_sha256=digest)

    source_p = [variant.p_value for variant in query.variants]
    significant = (
        True if any(p is not None and p <= parameter("analysis.gwas_significance_p") for p in source_p)
        else False if source_p and all(p is not None for p in source_p) else None
    )
    limitation = {
        "phase": "result_processing", "status": "size_limit", "source_only": True,
        "external_evidence_processed": False, "complete_result_retained": True,
        "ranking_skipped": True, "result_bytes": size, "limit_bytes": limit_bytes,
    }
    # These are counts of records loaded for this interactive view, not a claim
    # that the preserved full result is empty. Its scientific fields are unread.
    sufficiency = {
        "valid_input": True, "significant_signal": significant,
        "evidence_sources": 0, "evidence_records": 0, "genes_with_evidence": 0,
        "conflicts": 0, "policy_blocks": 0, "decision_evidence_records": 0,
        "locus_linked_records": 0, "locus_linked_genes": 0, "direct_evidence_sources": 0,
        "relevant_tissue_coverage": None, "evidence_dimensions": [],
        "missing_relevant_tissues": [], "region_gene_context_status": None,
        "nearby_annotated_gene_count": None, "region_gene_context_source": None,
        "context_evidence_scope": "unprocessed_retained_result",
        "warnings": [RESULT_PROCESSING_LIMIT_MESSAGE], "reasons": [RESULT_PROCESSING_LIMIT_MESSAGE],
    }
    return {
        "locus_id": query.locus_id, "run_id": expected_run_id, "status": "INCOMPLETE",
        "ranked_genes": [], "evidence": [], "sufficiency": sufficiency,
        "summary": RESULT_PROCESSING_LIMIT_MESSAGE,
        "model_id": "deterministic-source-only-result-limit",
        "data_mode": expected_mode, "data_release": release,
        "limitations": [RESULT_PROCESSING_LIMIT_MESSAGE],
        "artifacts": {"result": str(resolved_result), "manifest": str(manifest_path)},
        "external_evidence_processed": False, "complete_result_retained": True,
        "result_processing_limit": limitation,
    }
