from __future__ import annotations

import gzip
import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Mapping

from ..configuration import parameter

MAX_PLOT_POINTS = parameter("plot.max_total_points")
_DEFAULT_PRIORITY_THRESHOLD = parameter("plot.suggestive_logp")
_DEFAULT_GENOME_WIDE_THRESHOLD = round(-math.log10(parameter("analysis.gwas_significance_p")), parameter("plot.threshold_decimal_places"))
_DEFAULT_BACKGROUND_BIN_SIZE = parameter("plot.background_bin_logp_width")
# Interchange bound, retained for saved data produced under older display settings.
_MAX_NEG_LOG_P = 300.0
_MAX_BACKGROUND_BINS = 128
_CHROMOSOME = re.compile(r"^(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)$", re.IGNORECASE)
_VARIANT_ID = re.compile(
    r"^(?:rs\d+|(?:chr)?(?:[0-9]{1,2}|X|Y|MT)(?:_|:)\d+(?:_|:)[ACGTN-]+(?:_|:)[ACGTN-]+)$",
    re.IGNORECASE,
)


class UiDataError(ValueError):
    pass


class UiState:
    """Bounded in-memory state for the original visualization workflow."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: list[list[Any]] = []
        self._metadata: dict[str, Any] = {}
        self._variant_index: dict[str, list[Any]] = {}

    def clear(self) -> None:
        with self._lock:
            self._data = []
            self._metadata = {}
            self._variant_index = {}

    def has_data(self) -> dict[str, Any]:
        with self._lock:
            return {
                "has_data": bool(self._data),
                "total_snps": len(self._data),
                "data_source": str(self._metadata.get("data_source", "")),
            }

    def data_source(self) -> str | None:
        with self._lock:
            value = self._metadata.get("data_source")
            return str(value) if value else None

    def history_dataset_id(self) -> str | None:
        with self._lock:
            value = self._metadata.get("history_dataset_id")
            return str(value) if value else None

    def set_history_dataset_id(self, history_id: str) -> None:
        """Link the active plot to a persisted dataset without copying its rows."""

        if not isinstance(history_id, str) or not re.fullmatch(r"[a-f0-9]{32}", history_id):
            raise UiDataError("dataset history identifier is invalid")
        with self._lock:
            if not self._data:
                raise UiDataError("no dataset is loaded")
            self._metadata["history_dataset_id"] = history_id

    def set_analysis_history_context(self, history_id: str, *, selection_only: bool = False) -> None:
        """Identify the restored analysis without changing its scientific rows."""
        if not isinstance(history_id, str) or not re.fullmatch(r"[a-f0-9]{32}", history_id):
            raise UiDataError("analysis history identifier is invalid")
        with self._lock:
            self._metadata["history_analysis_id"] = history_id
            self._metadata["history_plot_scope"] = "analysis_selection" if selection_only else "dataset"
            if selection_only:
                self._metadata["history_plot_notice"] = (
                    "Showing saved analysis SNPs with recorded P values only; the original GWAS background is unavailable.")
            else:
                self._metadata.pop("history_plot_notice", None)

    def load_history(self, value: Mapping[str, Any], history_id: str) -> dict[str, Any]:
        """Restore checked plot rows and the original uploaded/demo provenance."""

        if not isinstance(history_id, str) or not re.fullmatch(r"[a-f0-9]{32}", history_id):
            raise UiDataError("dataset history identifier is invalid")
        self.load_processed(value)
        metadata = value.get("metadata", {})
        source = metadata.get("data_source", "user_upload")
        with self._lock:
            if source in {"asthma", "mdd"}:
                self._metadata.update({"data_source": source, "demo": True, "synthetic": False})
                source_file = metadata.get("source_file")
                if isinstance(source_file, str):
                    self._metadata["source_file"] = source_file[:240]
            self._metadata["history_dataset_id"] = history_id
        return self.has_data()

    def trait_hint(self) -> str:
        with self._lock:
            phenotype = str(self._metadata.get("source_phenotype", "")).strip()
            source = str(self._metadata.get("data_source", "")).casefold()
        if phenotype:
            return phenotype
        return {
            "asthma": "asthma",
            "mdd": "major depressive disorder",
        }.get(source, "user-selected GWAS locus")

    def variant_observations(
        self,
        variant_ids: list[str],
        *,
        region: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if len(variant_ids) > parameter("execution.max_variants_per_query"):
            raise UiDataError(f"chat accepts at most {parameter('execution.max_variants_per_query')} variants")
        with self._lock:
            selected_rows: list[list[Any]] = []
            seen: set[str] = set()
            for raw_identifier in variant_ids:
                identifier = str(raw_identifier).strip()
                if not _VARIANT_ID.fullmatch(identifier):
                    raise UiDataError(f"invalid variant identifier: {identifier!r}")
                row = self._variant_index.get(identifier.casefold())
                if row is None:
                    row = _observation_row_from_identifier(identifier)
                    row = self._variant_index.get(_query_variant_id(row).casefold(), row)
                identity = _query_variant_id(row).casefold()
                if identity not in seen:
                    selected_rows.append(list(row))
                    seen.add(identity)
            if not selected_rows and region:
                chromosome = str(region.get("chromosome", "")).removeprefix("chr").upper()
                try:
                    start = _position(region.get("start"))
                    end = _position(region.get("end"))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise UiDataError("chat region has invalid coordinates") from exc
                if not _CHROMOSOME.fullmatch(chromosome) or start <= 0 or end < start:
                    raise UiDataError("chat region has invalid coordinates")
                selected_rows = [
                    list(row)
                    for row in self._data
                    if str(row[0]).upper() == chromosome and start <= int(row[1]) <= end
                ][:parameter("execution.max_variants_per_query")]
        return [
            {
                "variant_id": _query_variant_id(row),
                "p_value": row[4],
                "chromosome": str(row[0]) if row[0] else None,
                "position": int(row[1]) if row[1] else None,
                **(dict(row[7]) if len(row) > 7 and isinstance(row[7], Mapping) else {}),
            }
            for row in selected_rows
        ]

    def plot_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "data": [list(row) for row in self._data],
                "format": "array",
                "total_snps": len(self._data),
                "metadata": dict(self._metadata),
            }

    def load_processed(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if value.get("format") != "array":
            raise UiDataError("processed GWAS data must use compact array format")
        rows = value.get("data")
        if not isinstance(rows, list) or not rows:
            raise UiDataError("processed GWAS data cannot be empty")
        if len(rows) > MAX_PLOT_POINTS:
            raise UiDataError(f"processed GWAS data exceeds {MAX_PLOT_POINTS:,} points")

        metadata_value = value.get("metadata", {})
        if not isinstance(metadata_value, Mapping):
            raise UiDataError("metadata must be an object")
        source_phenotype = _source_phenotype(metadata_value.get("source_phenotype"))
        normalized = [_validated_row(row, index) for index, row in enumerate(rows)]
        original_snps = max(
            len(normalized),
            _safe_nonnegative_int(metadata_value.get("original_snps"), len(normalized)),
        )
        sampling_metadata = _normalize_sampling_metadata(
            metadata_value,
            normalized,
            original_snps=original_snps,
        )
        metadata = {
            "filename": _bounded_text(metadata_value.get("filename"), 240),
            "data_source": "user_upload",
            "source_phenotype": source_phenotype,
            "total_snps": len(normalized),
            "original_snps": original_snps,
            "chromosomes": sorted(
                {str(row[0]) for row in normalized}, key=_chromosome_sort_key
            ),
            "sampled": original_snps > len(normalized),
            "processing_time": _bounded_text(metadata_value.get("processing_time"), 32),
            "rejected_rows": _safe_nonnegative_int(
                metadata_value.get("rejected_rows"), 0
            ),
            "repeated_headers": _safe_nonnegative_int(
                metadata_value.get("repeated_headers"), 0
            ),
            "zero_pvalues_capped": _safe_nonnegative_int(
                metadata_value.get("zero_pvalues_capped"), 0
            ),
            "sampling_strategy": _bounded_text(
                metadata_value.get("sampling_strategy"), 96
            ),
            "sampling_seed": _safe_nonnegative_int(
                metadata_value.get("sampling_seed"), 0
            ),
            **sampling_metadata,
        }
        with self._lock:
            self._data = normalized
            self._metadata = metadata
            self._variant_index = _build_variant_index(normalized)
        return self.has_data()

    def load_demo(self, name: str) -> dict[str, Any]:
        if name not in {"asthma", "mdd"}:
            raise UiDataError("demo dataset is not available")
        data, asset_metadata = _load_demo_asset(name)
        with self._lock:
            self._data = data
            self._metadata = {
                "filename": "Asthma demo" if name == "asthma" else "Depression demo",
                "data_source": name,
                "source_phenotype": (
                    "asthma" if name == "asthma" else "major depressive disorder"
                ),
                "total_snps": len(data),
                "original_snps": int(asset_metadata["original_count"]),
                "upstream_processed_snps": int(
                    asset_metadata["upstream_processed_count"]
                ),
                "chromosomes": sorted(
                    {str(row[0]) for row in data}, key=_chromosome_sort_key
                ),
                "sampled": True,
                "demo": True,
                "synthetic": False,
                "source_file": str(asset_metadata["source_file"]),
                "sampling_strategy": str(asset_metadata["sampling_strategy"]),
                "sampling_seed": int(asset_metadata["sampling_seed"]),
                "significant_count": int(asset_metadata["significant_count"]),
            }
            self._variant_index = _build_variant_index(data)
        return self.has_data()


def _validated_row(value: Any, index: int) -> list[Any]:
    if not isinstance(value, list) or len(value) < 7:
        raise UiDataError(f"data[{index}] must contain seven fields")
    chromosome = str(value[0]).removeprefix("chr").removeprefix("CHR").strip().upper()
    if not _CHROMOSOME.fullmatch(chromosome):
        raise UiDataError(f"data[{index}] has an invalid chromosome")
    try:
        position = _position(value[1])
        if isinstance(value[2], bool) or isinstance(value[4], bool):
            raise ValueError("boolean values are not measurements")
        neg_log_p = float(value[2])
        p_value = float(value[4])
    except (TypeError, ValueError, OverflowError) as exc:
        raise UiDataError(f"data[{index}] has invalid position or numeric fields") from exc
    if position <= 0 or not math.isfinite(neg_log_p) or neg_log_p < 0:
        raise UiDataError(f"data[{index}] has an invalid position or -log10(p)")
    if not math.isfinite(p_value) or not 0 < p_value <= 1:
        raise UiDataError(f"data[{index}] has an invalid p-value")
    ref = _allele(value[5])
    alt = _allele(value[6])
    supplied_id = _bounded_text(value[3], 120)
    snp_id = _canonical_variant_id(
        supplied_id,
        chromosome=chromosome,
        position=position,
        ref=ref,
        alt=alt,
    )
    # P is the scientific observation; never allow an independently supplied
    # display height to turn a nonsignificant row into a visually strong signal.
    row = [chromosome, position, round(-math.log10(p_value), parameter("plot.threshold_decimal_places")), snp_id, p_value, ref, alt]
    if len(value) > 7 and isinstance(value[7], Mapping):
        effects = {}
        for field in ("beta", "odds_ratio", "se"):
            raw = value[7].get(field)
            if raw is None:
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise UiDataError(f"data[{index}] has an invalid {field}")
            try:
                number = float(raw)
            except (ValueError, OverflowError) as exc:
                raise UiDataError(f"data[{index}] has an invalid {field}") from exc
            if not math.isfinite(number) or (field == "odds_ratio" and number <= 0) or (field == "se" and number < 0):
                raise UiDataError(f"data[{index}] has an invalid {field}")
            effects[field] = number
        for field in ("effect_allele", "other_allele", "allele1", "allele2"):
            raw = value[7].get(field)
            if raw is None or isinstance(raw, str) and raw.strip().upper() in {"", "N", "NA", "NAN", "NONE", "NULL", "."}:
                continue
            if not isinstance(raw, str) or not re.fullmatch(r"[ACGT]{1,32}", raw.strip().upper()):
                raise UiDataError(f"data[{index}] has an invalid {field}")
            effects[field] = raw.strip().upper()
        if effects:
            row.append(effects)
    return row


def _load_demo_asset(name: str) -> tuple[list[list[Any]], dict[str, Any]]:
    path = (
        Path(__file__).with_name("static")
        / "demo_data"
        / f"{name}.real.json.gz"
    )
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise UiDataError(f"demo asset is unavailable or invalid: {name}") from exc
    if not isinstance(value, Mapping):
        raise UiDataError(f"demo asset is invalid: {name}")
    rows = value.get("data")
    metadata = value.get("metadata")
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) > MAX_PLOT_POINTS
        or not isinstance(metadata, Mapping)
        or metadata.get("synthetic") is not False
    ):
        raise UiDataError(f"demo asset failed its integrity contract: {name}")
    required_metadata = {
        "source_file",
        "original_count",
        "upstream_processed_count",
        "sampling_strategy",
        "sampling_seed",
        "significant_count",
    }
    if required_metadata.difference(metadata):
        raise UiDataError(f"demo asset lacks provenance metadata: {name}")
    normalized = [_validated_row(row, index) for index, row in enumerate(rows)]
    normalized.sort(
        key=lambda item: (_chromosome_sort_key(str(item[0])), int(item[1]))
    )
    return normalized, dict(metadata)


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _source_phenotype(value: Any) -> str:
    """Validate the immutable phenotype attached to an uploaded GWAS dataset."""

    if not isinstance(value, str):
        raise UiDataError("source GWAS phenotype is required")
    phenotype = " ".join(value.split())
    if not phenotype:
        raise UiDataError("source GWAS phenotype is required")
    if len(phenotype) > 500:
        raise UiDataError("source GWAS phenotype is too long")
    if any(character in phenotype for character in "\x00"):
        raise UiDataError("source GWAS phenotype contains an invalid character")
    return phenotype


def _normalize_sampling_metadata(
    metadata: Mapping[str, Any],
    rows: list[list[Any]],
    *,
    original_snps: int,
) -> dict[str, Any]:
    """Normalize browser sampling disclosure against the accepted plot rows.

    Retained counts are derived from the validated rows. Seen counts remain
    client-supplied lower-resolution provenance, but are bounded so every
    retained category is a subset of its seen category and all category totals
    agree with ``original_snps`` and the accepted row count.
    """

    row_count = len(rows)
    hard_total_cap = min(
        MAX_PLOT_POINTS,
        max(
            row_count,
            _safe_nonnegative_int(metadata.get("hard_total_cap"), MAX_PLOT_POINTS),
        ),
    )
    priority_threshold = _safe_bounded_float(
        metadata.get("priority_threshold"),
        _DEFAULT_PRIORITY_THRESHOLD,
        minimum=0.0,
        maximum=_MAX_NEG_LOG_P,
    )
    genome_wide_threshold = _DEFAULT_GENOME_WIDE_THRESHOLD
    background_bin_size = _safe_bounded_float(
        metadata.get("background_bin_size"),
        _DEFAULT_BACKGROUND_BIN_SIZE,
        minimum=0.001,
        maximum=_MAX_NEG_LOG_P,
    )
    bin_count = max(
        1,
        min(
            _MAX_BACKGROUND_BINS,
            math.ceil(priority_threshold / background_bin_size),
        ),
    )

    significance_p = parameter("analysis.gwas_significance_p")
    priority_retained = sum(
        float(row[2]) >= priority_threshold or float(row[4]) <= significance_p for row in rows
    )
    genome_wide_retained = sum(
        float(row[4]) <= significance_p for row in rows
    )
    genome_wide_retained = min(genome_wide_retained, priority_retained)
    moderate_retained = priority_retained - genome_wide_retained
    background_retained = row_count - priority_retained

    declared_priority_seen = _first_nonnegative_int(
        metadata,
        ("priority_seen", "significant_seen"),
    )
    if declared_priority_seen is None:
        declared_genome_seen = _optional_nonnegative_int(
            metadata.get("genome_wide_seen")
        )
        declared_moderate_seen = _optional_nonnegative_int(
            metadata.get("moderate_seen")
        )
        if declared_genome_seen is not None and declared_moderate_seen is not None:
            declared_priority_seen = declared_genome_seen + declared_moderate_seen
        else:
            declared_priority_seen = priority_retained

    # Seen counts describe the input stream and may legitimately exceed the
    # in-memory hard cap. Only category partitioning constrains provenance.
    priority_seen_upper = original_snps - background_retained
    priority_seen = min(
        max(priority_retained, declared_priority_seen),
        max(priority_retained, priority_seen_upper),
    )

    declared_genome_wide_seen = _optional_nonnegative_int(
        metadata.get("genome_wide_seen")
    )
    if declared_genome_wide_seen is None:
        declared_moderate_seen = _optional_nonnegative_int(
            metadata.get("moderate_seen")
        )
        if declared_moderate_seen is None:
            declared_genome_wide_seen = genome_wide_retained
        else:
            declared_genome_wide_seen = priority_seen - min(
                priority_seen,
                declared_moderate_seen,
            )
    genome_wide_seen = min(
        max(genome_wide_retained, declared_genome_wide_seen),
        priority_seen - moderate_retained,
    )
    moderate_seen = priority_seen - genome_wide_seen

    priority_omitted = priority_seen - priority_retained
    background_seen = original_snps - priority_seen
    priority_label = (
        _bounded_text(
            metadata.get("priority_label") or metadata.get("significant_label"),
            64,
        )
        or "priority_or_suggestive"
    )

    actual_background_bins = [0] * bin_count
    for row in rows:
        neg_log_p = float(row[2])
        if neg_log_p >= priority_threshold or float(row[4]) <= significance_p:
            continue
        bin_index = min(
            bin_count - 1,
            max(0, math.floor(neg_log_p / background_bin_size)),
        )
        actual_background_bins[bin_index] += 1

    declared_per_bin_cap = _safe_nonnegative_int(
        metadata.get("background_per_bin_cap"),
        0,
    )
    background_per_bin_cap = min(
        hard_total_cap,
        max(declared_per_bin_cap, max(actual_background_bins, default=0)),
    )
    background_capacity_budget = min(
        background_per_bin_cap * bin_count,
        max(0, hard_total_cap - priority_retained),
    )
    background_capacity_budget = max(
        background_retained,
        background_capacity_budget,
    )
    background_bin_capacity = _bounded_count_vector(
        metadata.get("background_bin_capacity"),
        minima=actual_background_bins,
        item_cap=background_per_bin_cap,
        total_cap=background_capacity_budget,
    )
    background_bin_seen = _bounded_count_vector(
        metadata.get("background_bin_seen"),
        minima=actual_background_bins,
        item_cap=background_seen,
        total_cap=background_seen,
    )

    return {
        "priority_threshold": priority_threshold,
        "priority_seen": priority_seen,
        "priority_retained": priority_retained,
        "priority_omitted": priority_omitted,
        "priority_truncated": priority_omitted > 0,
        "priority_label": priority_label,
        # Keep the legacy names as exact aliases so older UI consumers cannot
        # disclose a contradictory interpretation of the same sample.
        "significant_seen": priority_seen,
        "significant_retained": priority_retained,
        "significant_omitted": priority_omitted,
        "significant_truncated": priority_omitted > 0,
        "significant_label": priority_label,
        "genome_wide_threshold": genome_wide_threshold,
        "genome_wide_seen": genome_wide_seen,
        "genome_wide_retained": genome_wide_retained,
        "moderate_seen": moderate_seen,
        "moderate_retained": moderate_retained,
        "background_seen": background_seen,
        "background_retained": background_retained,
        "background_bin_size": background_bin_size,
        "background_per_bin_cap": background_per_bin_cap,
        "background_bin_capacity": background_bin_capacity,
        "background_bin_seen": background_bin_seen,
        "background_bin_retained": actual_background_bins,
        "hard_total_cap": hard_total_cap,
    }


def _build_variant_index(rows: list[list[Any]]) -> dict[str, list[Any]]:
    index: dict[str, list[Any]] = {}
    for row in rows:
        identifier = str(row[3]).casefold()
        index.setdefault(identifier, row)
        coordinate = f"{row[0]}_{row[1]}_{row[5]}_{row[6]}".casefold()
        index.setdefault(coordinate, row)
    return index


def _observation_row_from_identifier(identifier: str) -> list[Any]:
    if identifier.casefold().startswith("rs"):
        return [None, None, None, identifier, None, None, None]
    parts = re.split(r"[_:]", re.sub(r"^chr", "", identifier, flags=re.IGNORECASE))
    if len(parts) != 4:
        raise UiDataError(f"invalid coordinate variant identifier: {identifier!r}")
    chromosome, position, ref, alt = parts
    chromosome = chromosome.upper()
    if not _CHROMOSOME.fullmatch(chromosome):
        raise UiDataError("coordinate variant identifier has an invalid chromosome")
    try:
        parsed_position = _position(position)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UiDataError("coordinate variant identifier has an invalid position") from exc
    return [chromosome, parsed_position, None, identifier, None, ref.upper(), alt.upper()]


def _query_variant_id(row: list[Any]) -> str:
    if (
        row[0]
        and row[1]
        and row[5]
        and row[6]
        and _CHROMOSOME.fullmatch(str(row[0]))
        and re.fullmatch(r"[ACGT]+", str(row[5]))
        and re.fullmatch(r"[ACGT]+", str(row[6]))
    ):
        return f"{str(row[0]).upper()}_{int(row[1])}_{row[5]}_{row[6]}"
    return str(row[3])


def _allele(value: Any) -> str:
    allele = str(value or "").strip().upper()
    return (
        allele
        if allele and len(allele) <= 32 and re.fullmatch(r"[ACGTN]+", allele)
        else "N"
    )


def _canonical_variant_id(
    value: str, *, chromosome: str, position: int, ref: str, alt: str
) -> str:
    if re.fullmatch(r"rs\d+", value, flags=re.IGNORECASE):
        return "rs" + value[2:]
    if _VARIANT_ID.fullmatch(value):
        coordinate = re.split(
            r"[_:]", re.sub(r"^chr", "", value, flags=re.IGNORECASE)
        )
        if len(coordinate) == 4:
            candidate_chromosome, candidate_position, candidate_ref, candidate_alt = (
                coordinate
            )
            candidate_chromosome = candidate_chromosome.upper()
            try:
                parsed_position = int(candidate_position)
            except ValueError:
                parsed_position = 0
            if _CHROMOSOME.fullmatch(candidate_chromosome) and parsed_position > 0:
                if (
                    candidate_chromosome != chromosome
                    or parsed_position != position
                    or _allele(candidate_ref) != ref
                    or _allele(candidate_alt) != alt
                ):
                    raise UiDataError("coordinate variant identifier disagrees with row fields")
                return (
                    f"{candidate_chromosome}_{parsed_position}_"
                    f"{_allele(candidate_ref)}_{_allele(candidate_alt)}"
                )
    return f"{chromosome}_{position}_{ref}_{alt}"


def _position(value: Any) -> int:
    """Genomic coordinates must survive a Python-to-browser round trip exactly."""

    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("position must be an integer")
    result = int(value)
    if isinstance(value, float) and result != value:
        raise ValueError("position must not have a fractional part")
    if not 0 < result < 2**53:
        raise ValueError("position must be a positive browser-safe integer")
    return result


def _safe_nonnegative_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if result >= 0 else default


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _first_nonnegative_int(
    metadata: Mapping[str, Any],
    names: tuple[str, ...],
) -> int | None:
    for name in names:
        result = _optional_nonnegative_int(metadata.get(name))
        if result is not None:
            return result
    return None


def _safe_bounded_float(
    value: Any,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(result):
        return default
    return min(maximum, max(minimum, result))


def _bounded_count_vector(
    value: Any,
    *,
    minima: list[int],
    item_cap: int,
    total_cap: int,
) -> list[int]:
    source = value if isinstance(value, list) else []
    result = [min(item_cap, max(0, int(minimum))) for minimum in minima]
    remaining = max(0, total_cap - sum(result))
    for index in range(min(len(source), len(result))):
        requested = _optional_nonnegative_int(source[index])
        if requested is None:
            continue
        target = min(item_cap, requested)
        if target <= result[index] or remaining <= 0:
            continue
        increase = min(target - result[index], remaining)
        result[index] += increase
        remaining -= increase
    return result


def _chromosome_sort_key(value: str) -> tuple[int, str]:
    normalized = value.upper()
    if normalized.isdigit():
        return int(normalized), normalized
    return {"X": 23, "Y": 24, "MT": 25}.get(normalized, 99), normalized
