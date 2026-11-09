from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import QTL_STUDY_TYPES


QTL_STUDY_TYPE_LABELS: dict[str, str] = {
    "eqtl": "eQTL",
    "pqtl": "pQTL",
    "sqtl": "sQTL",
    "tuqtl": "tuQTL",
    "sceqtl": "sc-eQTL",
    "scpqtl": "sc-pQTL",
    "scsqtl": "sc-sQTL",
    "sctuqtl": "sc-tuQTL",
}


def native_qtl_study_type(value: Any) -> str | None:
    """Return a supported native Open Targets QTL study type, if present.

    Open Targets responses can expose enum-like values with a namespace
    prefix.  We remove only that prefix and case-normalise; biological aliases
    are intentionally not guessed at this data boundary.
    """

    if value is None:
        return None
    normalized = str(value).strip().rsplit(".", 1)[-1].casefold()
    return normalized if normalized in QTL_STUDY_TYPES else None


def qtl_study_type_from_payload(value: Mapping[str, Any]) -> str | None:
    """Read the native study type without converting unrelated studies to QTL."""

    for key in ("studyType", "study_type"):
        if key in value:
            return native_qtl_study_type(value.get(key))
    study = value.get("study")
    if isinstance(study, Mapping):
        for key in ("studyType", "study_type"):
            if key in study:
                return native_qtl_study_type(study.get(key))
    return None


def qtl_study_type_from_semantic_payload(
    value: Mapping[str, Any],
) -> str | None:
    """Find a native subtype in a normalized semantic evidence payload."""

    direct = qtl_study_type_from_payload(value)
    if direct is not None:
        return direct
    context = value.get("context")
    if isinstance(context, Mapping):
        direct = qtl_study_type_from_payload(context)
        if direct is not None:
            return direct
    upstream = value.get("upstream")
    if isinstance(upstream, Mapping):
        raw = upstream.get("raw")
        if isinstance(raw, Mapping):
            return qtl_study_type_from_payload(raw)
    return None


def requested_qtl_study_types(values: Any) -> tuple[str, ...]:
    """Validate a caller-provided native subtype list without inventing aliases."""

    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("qtl_types must be an array")
    result = tuple(str(item) for item in values)
    if len(result) != len(set(result)):
        raise ValueError("qtl_types must not contain duplicates")
    unsupported = tuple(item for item in result if item not in QTL_STUDY_TYPES)
    if unsupported:
        raise ValueError(
            "unsupported QTL study type(s): " + ", ".join(unsupported)
        )
    return result
