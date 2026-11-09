from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .audit import sha256_json


SNAPSHOT_MANIFEST_FILENAME = "gwasmind_snapshot_manifest.json"
MAX_SNAPSHOT_MANIFEST_BYTES = 1024 * 1024
_HASH = re.compile(r"^[a-f0-9]{64}$")
_RELEASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class SnapshotManifestError(ValueError):
    """The frozen snapshot manifest is absent, stale, or internally invalid."""


@dataclass(frozen=True, slots=True)
class SnapshotInspection:
    data_root: str
    release: str
    datasets: dict[str, dict[str, int]]
    manifest_sha256: str
    content_sha256: str | None

    def to_dict(self) -> dict:
        return {
            "data_root": self.data_root,
            "release": self.release,
            "datasets": self.datasets,
            "manifest_sha256": self.manifest_sha256,
            "content_sha256": self.content_sha256,
        }


def inspect_snapshot(
    data_root: str | Path, *, release: str, full_hash: bool = False
) -> SnapshotInspection:
    if (
        not isinstance(release, str)
        or not _RELEASE.fullmatch(release)
        or release.casefold() in {"live", "unknown", "unverified"}
    ):
        raise ValueError("snapshot inspection requires one explicit frozen release")
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"snapshot root does not exist: {root}")
    datasets: dict[str, dict[str, int]] = {}
    all_files: list[Path] = []
    for dataset in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(dataset.rglob("*.parquet"))
        if not files:
            continue
        all_files.extend(files)
        datasets[dataset.name] = {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }
    if not datasets:
        raise ValueError("snapshot contains no Parquet datasets")
    content_hash: str | None = None
    if full_hash:
        digest = hashlib.sha256()
        for path in all_files:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        content_hash = digest.hexdigest()
    manifest_value: dict[str, Any] = {"release": release, "datasets": datasets}
    if content_hash is not None:
        manifest_value["content_sha256"] = content_hash
    return SnapshotInspection(
        data_root=str(root),
        release=release,
        datasets=datasets,
        manifest_sha256=sha256_json(manifest_value),
        content_sha256=content_hash,
    )


def damaged_parquet_files(data_root: str | Path) -> list[str]:
    """Parquet files whose 4-byte "PAR1" header or footer is missing.

    A cheap provisioning check (8 bytes per file) that catches truncated or
    zero-filled copies before a manifest certifies them.
    """
    root = Path(data_root).resolve()
    damaged: list[str] = []
    for path in sorted(root.glob("*/**/*.parquet")):
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(4)
            handle.seek(max(0, size - 4))
            tail = handle.read(4)
        if head != b"PAR1" or tail != b"PAR1":
            damaged.append(path.relative_to(root).as_posix())
    return damaged


def snapshot_manifest_path(data_root: str | Path) -> Path:
    return Path(data_root).resolve() / SNAPSHOT_MANIFEST_FILENAME


def load_snapshot_manifest(
    data_root: str | Path,
    *,
    expected_release: str | None = None,
) -> SnapshotInspection:
    """Validate the conventional frozen-snapshot manifest without reading Parquet.

    The manifest hash binds the declared release to the dataset directory names,
    Parquet file counts, and byte sizes returned by :func:`inspect_snapshot`.
    This intentionally does not compute a full content hash during interactive
    startup.  Creating the manifest is an explicit provisioning operation;
    runtime callers only read and verify it.
    """

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise SnapshotManifestError(f"snapshot root does not exist: {root}")
    path = snapshot_manifest_path(root)
    if not path.is_file():
        raise SnapshotManifestError(
            f"frozen snapshot manifest is missing: {path.name}; create it with "
            "gwasmind inspect-snapshot --release <release> --output <manifest>"
        )
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SnapshotManifestError("frozen snapshot manifest cannot be inspected") from exc
    if not 1 <= size <= MAX_SNAPSHOT_MANIFEST_BYTES:
        raise SnapshotManifestError("frozen snapshot manifest has an invalid size")
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_SNAPSHOT_MANIFEST_BYTES + 1)
        if len(raw) > MAX_SNAPSHOT_MANIFEST_BYTES:
            raise SnapshotManifestError("frozen snapshot manifest has an invalid size")
        value: Any = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SnapshotManifestError("frozen snapshot manifest is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise SnapshotManifestError("frozen snapshot manifest must be a JSON object")

    allowed = {
        "data_root",
        "release",
        "datasets",
        "manifest_sha256",
        "content_sha256",
    }
    unknown = set(value).difference(allowed)
    required = {"release", "datasets", "manifest_sha256"}
    if unknown or required.difference(value):
        raise SnapshotManifestError(
            "frozen snapshot manifest fields do not match the supported contract"
        )
    release = value.get("release")
    if (
        not isinstance(release, str)
        or not _RELEASE.fullmatch(release)
        or release.casefold() in {"live", "unknown", "unverified"}
    ):
        raise SnapshotManifestError(
            "frozen snapshot manifest must declare one explicit release"
        )
    if expected_release is not None and release != expected_release:
        raise SnapshotManifestError(
            f"frozen snapshot release {release!r} does not match requested release "
            f"{expected_release!r}"
        )

    datasets = _validated_dataset_manifest(value.get("datasets"))
    declared_hash = value.get("manifest_sha256")
    if (
        not isinstance(declared_hash, str)
        or declared_hash.casefold() == "unverified"
        or not _HASH.fullmatch(declared_hash)
    ):
        raise SnapshotManifestError(
            "frozen snapshot manifest_sha256 must be a verified SHA-256 value"
        )
    content_hash = value.get("content_sha256")
    if content_hash is not None and (
        not isinstance(content_hash, str) or not _HASH.fullmatch(content_hash)
    ):
        raise SnapshotManifestError(
            "frozen snapshot content_sha256 must be null or a SHA-256 value"
        )

    bound_fields: dict[str, Any] = {"release": release, "datasets": datasets}
    if content_hash is not None:
        bound_fields["content_sha256"] = content_hash
    if sha256_json(bound_fields) != declared_hash:
        raise SnapshotManifestError(
            "frozen snapshot manifest hash does not match its declared fields"
        )

    try:
        current = inspect_snapshot(root, release=release, full_hash=False)
    except (OSError, ValueError) as exc:
        raise SnapshotManifestError("frozen snapshot contents cannot be verified") from exc
    if current.datasets != datasets:
        raise SnapshotManifestError(
            "frozen snapshot contents do not match the provisioned manifest; "
            "the snapshot may be incomplete, mixed, or modified"
        )
    return SnapshotInspection(
        data_root=str(root),
        release=release,
        datasets=datasets,
        manifest_sha256=declared_hash,
        content_sha256=content_hash,
    )


def _validated_dataset_manifest(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or not value:
        raise SnapshotManifestError(
            "frozen snapshot manifest must contain at least one dataset"
        )
    datasets: dict[str, dict[str, int]] = {}
    for name, details in value.items():
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or not isinstance(details, Mapping)
            or set(details) != {"files", "bytes"}
        ):
            raise SnapshotManifestError(
                "frozen snapshot manifest contains an invalid dataset entry"
            )
        files = details.get("files")
        byte_count = details.get("bytes")
        if (
            not isinstance(files, int)
            or isinstance(files, bool)
            or files <= 0
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
        ):
            raise SnapshotManifestError(
                "frozen snapshot manifest contains invalid dataset counts"
            )
        datasets[name] = {"files": files, "bytes": byte_count}
    return dict(sorted(datasets.items()))
