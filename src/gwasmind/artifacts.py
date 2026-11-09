from __future__ import annotations

import hashlib
import json
import os  # noqa: F401 -- public atomic-replace seam retained for integrations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import sha256_json
from .evidence_store import (
    STORE_KEY,
    atomic_json,
    iter_canonical,
    iter_records,
    verify_metadata_fields,
    write_document,
    write_record_response,
)


@dataclass(frozen=True, slots=True)
class ArtifactPointer:
    path: str
    sha256: str
    bytes: int


class ArtifactStore:
    """Atomic, content-addressed storage for tool requests and responses."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: str, value: Any, *, compact: bool = False,
                   byte_progress=None) -> ArtifactPointer:
        target = (self.root / relative_path).resolve()
        if self.root != target and self.root not in target.parents:
            raise ValueError(f"artifact path escapes store: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        size, byte_digest = (atomic_json(target, value, compact=compact, byte_progress=byte_progress)
                             if byte_progress is not None else atomic_json(target, value, compact=compact))
        if compact:
            digest = byte_digest
        else:
            digest_state = hashlib.sha256()
            for part in iter_canonical(value):
                digest_state.update(part.encode("utf-8"))
            digest = digest_state.hexdigest()
        return ArtifactPointer(path=str(target), sha256=digest, bytes=size)

    def write_record_response(self, relative_path: str, value: dict[str, Any], records: tuple | list,
                              *, progress=None, byte_progress=None) -> ArtifactPointer:
        """Preserve the standalone tool response while streaming detached records."""
        target = (self.root / relative_path).resolve()
        if self.root != target and self.root not in target.parents:
            raise ValueError(f"artifact path escapes store: {target}")
        size, digest = write_record_response(target, value, records,
            progress=progress, byte_progress=byte_progress)
        return ArtifactPointer(path=str(target), sha256=digest, bytes=size)

    def write_evidence_document(self, relative_path: str, value: Any, *, field: str,
                                metadata_summaries=None) -> ArtifactPointer:
        """Write complete evidence with a small, portable manifest for large runs."""
        target = (self.root / relative_path).resolve()
        if self.root != target and self.root not in target.parents:
            raise ValueError(f"artifact path escapes store: {target}")
        size, digest = write_document(target, value, field=field, metadata_summaries=metadata_summaries)
        return ArtifactPointer(path=str(target), sha256=digest, bytes=size)

    def _reuse_native_evidence(self, relative_path: str, value: Any,
                               source: ArtifactPointer, *, original_records: tuple) -> ArtifactPointer | None:
        """Private Agent-finalization path; source and target share exact records."""
        from .evidence_store import _reuse_verified_ledger
        target = (self.root / relative_path).resolve()
        source_path = Path(source.path).resolve()
        if self.root not in target.parents or self.root not in source_path.parents:
            raise ValueError("evidence reuse escaped artifact store")
        result = _reuse_verified_ledger(target, value, field="evidence",
                                        source=source_path, source_sha256=source.sha256,
                                        original_records=original_records)
        return ArtifactPointer(path=str(target), bytes=result[0], sha256=result[1]) if result else None

    def verify(self, pointer: ArtifactPointer) -> bool:
        try:
            target = Path(pointer.path).resolve()
            if self.root not in target.parents or not target.is_file():
                return False
            value = json.loads(target.read_text(encoding="utf-8"))
            if sha256_json(value) != pointer.sha256:
                return False
            if isinstance(value, dict):
                verify_metadata_fields(target, value)
            if isinstance(value, dict) and STORE_KEY in value:
                descriptor = value[STORE_KEY]
                field = descriptor.get("field") if isinstance(descriptor, dict) else None
                if field not in {"evidence", "records"} or field in value:
                    return False
                for _row in iter_records(target, descriptor, expected_field=field):
                    pass
            return True
        except (OSError, UnicodeError, ValueError, TypeError):
            return False
