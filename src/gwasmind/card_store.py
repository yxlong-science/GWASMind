"""Immutable card archives kept outside evidence metadata and loaded on demand."""
from pathlib import Path
import os
from uuid import uuid4

FORMAT = "gwasmind-card-archive-v1"
KEY = "v2g_cards_artifact"


class StoredCardList(list):
    """List-compatible lazy read view; old in-memory consumers need no schema fork."""

    def __init__(self, document, reference):
        super().__init__()
        self.document = Path(document)
        self.reference = dict(reference)
        self._loaded = None

    def _rows(self):
        if self._loaded is None:
            from .evidence_store import _read_manifest, _require, MAX_METADATA_BYTES
            path = archive_path(self.document, self.reference)
            value = _read_manifest(path, MAX_METADATA_BYTES)
            _require(isinstance(value.get("cards"), list), "Invalid gene evidence card archive")
            _require(len(value["cards"]) == self.reference["count"], "Gene evidence card count mismatch")
            self._loaded = value["cards"]
        return self._loaded

    def __len__(self):
        return self.reference["count"]

    def __iter__(self):
        return iter(self._rows())

    def __getitem__(self, key):
        return self._rows()[key]

    def __eq__(self, other):
        return self._rows() == (list(other) if isinstance(other, StoredCardList) else other)

    def __ne__(self, other):
        return not self == other

    def __deepcopy__(self, memo):
        # Each read view has its own cache. The file remains content addressed.
        return type(self)(self.document, self.reference)


def archive_path(document, reference):
    from .evidence_store import _require, _HEX, _identity, _hash_file, MAX_METADATA_BYTES
    _require(isinstance(reference, dict) and set(reference) == {"format", "path", "sha256", "bytes", "count"},
             "Invalid gene evidence card reference")
    sha = reference["sha256"]
    _require(reference["format"] == FORMAT and isinstance(sha, str) and _HEX.fullmatch(sha),
             "Invalid gene evidence card digest")
    _require(reference["path"] == f"gene_cards/{sha}.json", "Invalid gene evidence card path")
    root = Path(document).parent.resolve()
    path = (root / reference["path"]).resolve()
    _require(path.parent == root / "gene_cards", "Gene evidence card path escapes its run")
    _require(type(reference["count"]) is int and reference["count"] >= 0, "Invalid gene evidence card count")
    _require(type(reference["bytes"]) is int and 0 < reference["bytes"] <= MAX_METADATA_BYTES,
             "Gene evidence cards exceed processing range")
    _require(_identity(path)[2] == reference["bytes"] and _hash_file(path) == sha,
             "Gene evidence card archive failed integrity verification")
    return path


def store_cards(document, cards):
    from .evidence_store import atomic_json, _require, MAX_METADATA_BYTES
    directory = Path(document).parent / "gene_cards"
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(cards, StoredCardList):
        source = archive_path(cards.document, cards.reference)
        target = directory / source.name
        if source.resolve() != target.resolve() and not target.exists():
            import shutil
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        archive_path(document, cards.reference)
        return dict(cards.reference)
    temporary = directory / (uuid4().hex + ".staging")
    try:
        size, sha = atomic_json(temporary, {"cards": cards})
        _require(size <= MAX_METADATA_BYTES, "Locus evidence exceeds processing range: gene evidence card archive")
        target = directory / (sha + ".json")
        if not target.exists():
            os.replace(temporary, target)
        reference = {"format": FORMAT, "path": "gene_cards/" + target.name,
                     "bytes": size, "sha256": sha, "count": len(cards)}
        archive_path(document, reference)
        return reference
    finally:
        temporary.unlink(missing_ok=True)


def archive_audit(document, audit):
    if not isinstance(audit, dict) or "v2g_cards" not in audit:
        return audit
    return {**{key: value for key, value in audit.items() if key != "v2g_cards"},
            KEY: store_cards(document, audit["v2g_cards"])}


def restore_cards(document, audit):
    if isinstance(audit, dict) and KEY in audit and "v2g_cards" not in audit:
        from .evidence_store import StoredMetadataDict
        view = StoredCardList(document, audit[KEY])
        if isinstance(audit, StoredMetadataDict):
            audit._card_view = view
        else:
            audit["v2g_cards"] = view
    return audit


def cards_from_audit(audit):
    """Resolve cards after a metadata-summary JSON round trip as well as live reads."""
    from collections.abc import Mapping
    if not isinstance(audit, Mapping):
        return ()
    if (cards := audit.get("v2g_cards")) is not None:
        return cards
    marker = audit.get("_metadata_summary") or {}
    if KEY in audit and marker.get("document"):
        from .evidence_store import _validated_summary
        document, _, _ = _validated_summary(audit)
        return StoredCardList(document, audit[KEY])
    return ()
