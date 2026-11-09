"""Restore only the GWAS data supported by an analysis history snapshot."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

from .history_store import HistoryError
from .ui_state import UiDataError, UiState, _position, _validated_row


def _source_rows(snapshot):
    top = snapshot.get("selected_variants")
    queries, results = snapshot.get("queries"), snapshot.get("results")
    queries = queries if isinstance(queries, list) else []
    results = results if isinstance(results, list) else []
    by_locus = {row.get("locus_no", index + 1): row for index, row in enumerate(results)
                if isinstance(row, Mapping)}
    output = []
    loci = set(range(1, len(queries) + 1)) | {key for key in by_locus if type(key) is int and key > 0}
    for number in sorted(loci):
        index = number - 1
        selected = by_locus.get(index + 1, {}).get("selected_variants")
        query = queries[index] if index < len(queries) and isinstance(queries[index], Mapping) else {}
        original = query.get("variants")
        selected = original if isinstance(original, list) and original else selected
        output.extend(selected if isinstance(selected, list) else [])
    return output or (top if isinstance(top, list) else [])


def _plot_row(value, *, exact_identity=False):
    if not isinstance(value, Mapping):
        return None
    identifier = str(value.get("variant_id") or "").strip()
    canonical = re.fullmatch(r"(?:chr)?([0-9]{1,2}|X|Y|MT)[_:](\d+)[_:]([ACGTN-]+)[_:]([ACGTN-]+)", identifier, re.I)
    if not canonical and not re.fullmatch(r"rs\d+", identifier, re.I):
        return None
    chromosome = value.get("chromosome") or (canonical[1] if canonical else None)
    position = value.get("position") if value.get("position") is not None else canonical[2] if canonical else None
    ref, alt = (canonical[3], canonical[4]) if canonical else ("N", "N")
    try:
        p_value = value.get("p_value")
        row = _validated_row([chromosome, position, -math.log10(float(p_value)), identifier, p_value, ref, alt], 0)
        if exact_identity and canonical and row[5:7] != [ref.upper(), alt.upper()]:
            return None
        return row
    except (UiDataError, ValueError, TypeError, OverflowError):
        return None


def _point_key(row):
    return str(row[3]), str(row[0]), int(row[1]), float(row[4])


def matching_analysis_dataset(store, snapshot: Mapping) -> str | None:
    """Find one dataset containing every saved source ID, coordinate and P value."""
    sources = _source_rows(snapshot)
    points = [_plot_row(row, exact_identity=True) for row in sources]
    if not points or any(row is None for row in points):
        return None
    required = {_point_key(row) for row in points}
    coordinates = set()
    positions = snapshot.get("snp_positions", [])
    if not isinstance(positions, list):
        return None
    try:
        for row in positions:
            chrom = re.sub(r"^chr", "", str(row["chromosome"]), flags=re.I).upper()
            coordinates.add((chrom, _position(row["position"])))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    found = None
    for offset in range(0, 100001, 100):
        page = store.list_datasets(limit=100, offset=offset)
        for item in page:
            try:
                rows = store.get_dataset(item["id"])["data"]
                keys = {_point_key(row) for row in rows}
            except (HistoryError, OSError, KeyError, TypeError, ValueError, IndexError):
                continue
            if required.issubset(keys) and coordinates.issubset({(key[1], key[2]) for key in keys}):
                if found is not None:
                    return None
                found = item["id"]
        if len(page) < 100:
            return found
    return None


def restore_analysis_plot(store, snapshot: Mapping, history_id: str) -> tuple[UiState, str | None]:
    """Prefer a recorded binding, then a unique full match, then saved SNPs only."""
    state = UiState()
    dataset_id = snapshot.get("dataset_id") or matching_analysis_dataset(store, snapshot)
    if dataset_id:
        state.load_history(store.get_dataset(dataset_id), dataset_id)
        state.set_analysis_history_context(history_id)
        return state, dataset_id
    points = {_point_key(point): point for row in _source_rows(snapshot)
              if (point := _plot_row(row)) is not None}
    if points:
        state.load_processed({"format": "array", "data": list(points.values()), "metadata": {
            "filename": snapshot.get("filename") or "Saved analysis SNPs",
            "source_phenotype": snapshot.get("source_phenotype") or "Saved analysis",
            "original_snps": len(points), "sampling_strategy": "saved-analysis-selection"}})
    state.set_analysis_history_context(history_id, selection_only=True)
    return state, None
