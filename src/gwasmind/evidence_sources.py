"""Trace source-study diversity without inferring cohort independence."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _identifier(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def record_source_identifier(record: Mapping) -> str | None:
    """Use the anchored study, not evidence categories or coloc partner counts.

    Top-level/context/raw studyId are aliases of the same source record. A
    conflicting identity is unresolved. A typed molecular coloc side can fill
    an absent primary ID; arbitrary GWAS partners cannot inflate direct support.
    """
    payload = _mapping(record.get("payload"))
    context = _mapping(payload.get("context"))
    raw = _mapping(_mapping(payload.get("upstream")).get("raw"))
    identifiers = {value for value in (
        _identifier(record.get("study_id")), _identifier(context.get("study_id")),
        _identifier(raw.get("studyId")),
    ) if value}
    if identifiers:
        return "study:" + next(iter(identifiers)) if len(identifiers) == 1 else None
    if str(record.get("kind")) == "colocalization":
        molecular = {
            identifier for side in ("left", "right")
            if str(raw.get(side + "StudyType", "")).casefold().endswith("qtl")
            for identifier in [_identifier(raw.get(side + "StudyId"))] if identifier
        }
        if molecular:
            return "study:" + next(iter(molecular)) if len(molecular) == 1 else None
    datasets = {value for value in (
        _identifier(record.get("source_dataset_id")), _identifier(context.get("source_dataset_id")),
        _identifier(raw.get("sourceDatasetId")),
    ) if value}
    return "dataset:" + next(iter(datasets)) if len(datasets) == 1 else None


@dataclass(frozen=True, slots=True)
class SourceDiversity:
    source_ids: tuple[str, ...]
    unidentified_records: int

    @property
    def count(self) -> int:
        return len(self.source_ids)


def source_diversity(records: Sequence[Mapping]) -> SourceDiversity:
    values = [record_source_identifier(record) for record in records]
    return SourceDiversity(tuple(sorted({value for value in values if value})), values.count(None))
