"""Preserve scientific artifacts while deduplicating and compressing completed workers.

All existing paths retain identical bytes. A durable journal precedes replacement
of a duplicate with a verified hard link. Raw responses, full fact views and
stdout are never deleted; interrupted receipt writes are recoverable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from .configuration import parameter
from .file_lock import exclusive_file_lock

RECEIPT_NAME = "retention_receipt.json"
JOURNAL_NAME = "retention_journal.json"
POLICY_VERSION = "3"
_FORMAT = "gwasmind_retention_receipt_v2"
_CONTENT_NAME = re.compile(r"([0-9a-f]{64})\.(?:json|jsonl)")
_MIN_COMPRESS_BYTES = 64 * 1024


def _local_file(root: Path, path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)


def compress_stage(stage: Path) -> int:
    """Apply byte-preserving Windows compression; count successful submissions."""
    compact = shutil.which("compact") if os.name == "nt" else None
    if compact is None:
        return 0
    stage = stage.resolve()
    files = [path for path in stage.rglob("*") if _local_file(stage, path)
             and not any(part.startswith(".wof-") for part in path.relative_to(stage).parts)
             and path.stat().st_size >= _MIN_COMPRESS_BYTES]
    successful = 0
    with tempfile.TemporaryDirectory(prefix=".wof-", dir=stage) as directory:
        work = Path(directory).resolve()
        if not work.is_relative_to(stage):
            raise ValueError("Compression workspace escaped its stage")
        names = []
        for index, path in enumerate(files):
            try:
                os.link(path, work / str(index))
            except OSError:
                continue
            names.append(str(index))
        for start in range(0, len(names), 100):
            batch = names[start:start + 100]
            result = subprocess.run([compact, "/c", "/exe:lzx", "/q", "/i", *batch], cwd=work,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode == 0:
                successful += len(batch)
    return successful


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Storage optimization receipt is too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("format") != _FORMAT
            or value.get("policy_version") != POLICY_VERSION or not isinstance(value.get("files"), list)):
        raise ValueError("Unsupported storage optimization receipt version")
    return value


def _immutable_files(stage: Path) -> list[Path]:
    return sorted(path for pattern in ("output/tool_calls/*/raw_response.json", "output/tool_calls/*/response.json",
        "output/runs/*/evidence_chunks/*.jsonl", "output/runs/*/model_inputs/*.json")
        for path in stage.glob(pattern) if _local_file(stage, path))


def _link_plan(stage: Path, source_stage: Path | None) -> list[dict[str, Any]]:
    canonical: dict[tuple[int, str], Path] = {}
    candidates = [(path, False) for path in _immutable_files(source_stage)] if source_stage else []
    candidates += [(path, True) for path in _immutable_files(stage)]
    sizes: dict[int, int] = {}
    for path, _ in candidates:
        sizes[path.stat().st_size] = sizes.get(path.stat().st_size, 0) + 1
    planned = []
    for path, replaceable in candidates:
        size = path.stat().st_size
        if not size or sizes[size] < 2:
            continue
        digest = _sha256(path)
        addressed = _CONTENT_NAME.fullmatch(path.name)
        if addressed and addressed[1] != digest:
            continue
        source = canonical.setdefault((size, digest), path)
        if not replaceable or source == path or os.path.samefile(source, path):
            continue
        planned.append({"path": path.relative_to(stage).as_posix(), "bytes": size, "sha256": digest,
            "action": "hard_linked", "status": "pending",
            "linked_to": os.path.relpath(source, stage).replace(os.sep, "/")})
    return planned


def _paths(stage: Path, source_stage: Path | None, entry: dict[str, Any]) -> tuple[Path, Path]:
    if (not isinstance(entry, dict) or entry.get("action") != "hard_linked"
            or entry.get("status") not in {"pending", "completed", "skipped"}
            or type(entry.get("bytes")) is not int or entry["bytes"] < 0
            or not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or not isinstance(entry.get("path"), str) or not isinstance(entry.get("linked_to"), str)):
        raise ValueError("Invalid storage optimization journal entry")
    target = (stage / entry["path"]).resolve()
    source = (stage / entry["linked_to"]).resolve()
    if not target.is_relative_to(stage) or not any(source.is_relative_to(root) for root in (stage, source_stage) if root):
        raise ValueError("Storage optimization journal escaped its stage")
    return target, source


def _completed_stage(stage: Path) -> bool:
    runs = stage / "output" / "runs"
    return runs.is_dir() and any((child / "result.json").is_file() and (child / "run_manifest.json").is_file()
                                 for child in runs.iterdir())


def compact_saved_runs(artifact_root: Path, *, compress: bool = True) -> dict[str, Any]:
    """Optimize completed jobs only; preserve failed/cancelled workers unchanged."""
    summary = {"runs": 0, "skipped_runs": 0, "freed_bytes": 0}
    for job_file in sorted(Path(artifact_root).glob("*/job.json")):
        try:
            state = json.loads(job_file.read_text(encoding="utf-8")).get("state")
        except (OSError, ValueError):
            state = None
        if state != "completed":
            summary["skipped_runs"] += 1
            continue
        for query in sorted((job_file.parent / "queries").glob("[0-9][0-9][0-9]")):
            if _completed_stage(query):
                summary["freed_bytes"] += compact_completed_stage(query, compress=compress)["freed_bytes_this_pass"]
            ranking = query / "ranking"
            if _completed_stage(ranking):
                summary["freed_bytes"] += compact_completed_stage(
                    ranking, source_stage=query, compress=compress)["freed_bytes_this_pass"]
        summary["runs"] += 1
    readable = Path(artifact_root) / "_readable_exports"
    if compress and readable.is_dir():
        compress_stage(readable)
    return summary


def compact_completed_stage(stage: Path, *, source_stage: Path | None = None,
                            compress: bool = False) -> dict[str, Any]:
    """Preserve all artifact bytes and paths, with recoverable deduplication receipts."""
    stage = Path(stage).resolve()
    source_stage = Path(source_stage).resolve() if source_stage is not None else None
    if source_stage == stage:
        source_stage = None
    with exclusive_file_lock(stage / ".retention.lock", parameter("artifacts.compaction_lock_timeout_seconds")):
        return _compact(stage, source_stage, compress)


def _compact(stage: Path, source_stage: Path | None, compress: bool) -> dict[str, Any]:
    receipt_path, journal_path = stage / RECEIPT_NAME, stage / JOURNAL_NAME
    previous = _read_record(receipt_path)
    journal = _read_record(journal_path)
    entries: dict[str, dict[str, Any]] = {}
    for record in (previous, journal):
        for entry in (record or {}).get("files", []):
            _paths(stage, source_stage, entry)
            entries[entry["path"]] = dict(entry)
    for entry in _link_plan(stage, source_stage):
        entries[entry["path"]] = entry
    receipt = {"format": _FORMAT, "policy_version": POLICY_VERSION,
               "scientific_artifacts_preserved": True, "files": list(entries.values()), "status": "running",
               "byte_accounting": "logical_bytes_relinked_not_physical_disk_recovery"}
    _atomic_json(journal_path, receipt)
    freed_now = 0
    for entry in receipt["files"]:
        if entry["status"] != "pending":
            continue
        target, source = _paths(stage, source_stage, entry)
        temporary = target.with_name(target.name + "." + uuid4().hex + ".link")
        try:
            if not (target.is_file() and source.is_file()
                    and target.stat().st_size == source.stat().st_size == entry["bytes"]
                    and _sha256(target) == _sha256(source) == entry["sha256"]):
                entry.update(status="skipped", reason="artifact_changed")
            else:
                if not os.path.samefile(target, source):
                    os.link(source, temporary)
                    os.replace(temporary, target)
                    freed_now += entry["bytes"]
                entry["status"] = "completed"
        except OSError as exc:
            entry.update(status="skipped", reason=type(exc).__name__)
        finally:
            temporary.unlink(missing_ok=True)
        _atomic_json(journal_path, receipt)
    receipt.update(status="completed", freed_bytes_this_pass=freed_now,
        freed_bytes=sum(entry["bytes"] for entry in receipt["files"] if entry["status"] == "completed"),
        transparent_compression="none")
    _atomic_json(journal_path, receipt)
    _atomic_json(receipt_path, receipt)
    if compress:
        try:
            if compress_stage(stage):
                receipt["transparent_compression"] = "wof_lzx"
        except (OSError, subprocess.SubprocessError) as exc:
            receipt["compression_issue"] = type(exc).__name__
        _atomic_json(journal_path, receipt)
        _atomic_json(receipt_path, receipt)
    return receipt
