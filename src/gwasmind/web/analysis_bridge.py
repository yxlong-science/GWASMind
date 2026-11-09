from __future__ import annotations

import csv
import io
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping
from uuid import uuid4

from ..configuration import parameter
from .ranking_policy import live_model_only_result, model_ranking_unavailable
from .jobs import (
    MAX_QUERIES,
    MAX_VARIANTS_PER_QUERY,
    JobConflictError,
    JobNotFoundError,
    JobValidationError,
)

MAX_LOCUS_TEXT_BYTES = parameter("web.max_json_bytes")
MAX_UPLOAD_SESSIONS = parameter("execution.max_upload_sessions")
MAX_LOCUS_ID_LENGTH = 256
_VARIANT_ID = re.compile(
    r"^(?:rs\d+|(?:chr)?(?:[0-9]{1,2}|X|Y|MT)(?:_|:)\d+(?:_|:)[ACGTN-]+(?:_|:)[ACGTN-]+)$",
    re.IGNORECASE,
)
_TASK_ID = re.compile(r"^[a-f0-9]{32}$")


@dataclass(slots=True)
class AnalysisUpload:
    task_id: str
    filename: str
    queries: list[dict[str, Any]]
    snp_positions: list[dict[str, Any]]
    original_snp_count: int
    warnings: list[str]
    run_id: str | None = None
    source_observations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    dataset_id: str | None = None

    @property
    def snp_count(self) -> int:
        return sum(len(query["variants"]) for query in self.queries)


class AnalysisBridge:
    """Parse the Auto Analysis panel input and project it onto the job API."""

    def __init__(
        self, *, run_is_terminal: Callable[[str], bool] | None = None
    ) -> None:
        self._lock = threading.RLock()
        self._uploads: OrderedDict[str, AnalysisUpload] = OrderedDict()
        self._run_is_terminal = run_is_terminal

    def set_run_terminal_resolver(
        self, resolver: Callable[[str], bool] | None
    ) -> None:
        """Supply a fail-closed activity check for capacity-safe LRU eviction."""

        with self._lock:
            self._run_is_terminal = resolver

    def upload_text(self, *, filename: Any, text: Any, dataset_id: str | None = None) -> dict[str, Any]:
        if dataset_id is not None and (not isinstance(dataset_id, str) or not _TASK_ID.fullmatch(dataset_id)):
            raise JobValidationError("plot dataset identifier is invalid")
        if not isinstance(filename, str) or not filename.strip():
            raise JobValidationError("locus filename is required")
        if not isinstance(text, str) or not text.strip():
            raise JobValidationError("locus file cannot be empty")
        if len(text.encode("utf-8")) > MAX_LOCUS_TEXT_BYTES:
            raise JobValidationError(f"locus file exceeds {MAX_LOCUS_TEXT_BYTES / 1024 / 1024:g} MB")
        source_observations: dict[str, list[dict[str, Any]]] = {}
        queries, positions, original_snp_count, warnings = _parse_locus_text(text, source_observations=source_observations)
        task_id = uuid4().hex
        upload = AnalysisUpload(
            task_id=task_id,
            filename=filename.strip()[:240],
            queries=queries,
            snp_positions=positions,
            original_snp_count=original_snp_count,
            warnings=warnings,
            source_observations=source_observations,
            dataset_id=dataset_id,
        )
        with self._lock:
            while len(self._uploads) >= MAX_UPLOAD_SESSIONS:
                evictable_task_id = next(
                    (
                        retained_task_id
                        for retained_task_id, retained in self._uploads.items()
                        if self._evictable(retained)
                    ),
                    None,
                )
                if evictable_task_id is None:
                    raise JobConflictError(
                        "analysis upload capacity is full; all retained uploads are "
                        "attached to runs"
                    )
                self._uploads.pop(evictable_task_id)
            self._uploads[task_id] = upload
            self._uploads.move_to_end(task_id)
        return {
            "task_id": task_id,
            "snp_count": upload.snp_count,
            "original_snp_count": upload.original_snp_count,
            "locus_count": len(upload.queries),
            "snp_positions": list(upload.snp_positions),
            "warnings": list(upload.warnings),
        }

    def queries_for_run(self, task_id: str, phenotype: str) -> list[dict[str, Any]]:
        if not isinstance(phenotype, str) or not phenotype.strip():
            raise JobValidationError("target phenotype is required")
        if len(phenotype.strip()) > 1_000:
            raise JobValidationError("target phenotype is too long")
        upload = self._get(task_id)
        return [
            {
                **query,
                "trait": phenotype.strip(),
                "task": "prioritize_genes",
                "benchmark": False,
                "genome_build": "GRCh38",
            }
            for query in upload.queries
        ]

    def source_observations_for_locus(self, task_id: str, locus_id: str) -> list[dict[str, Any]]:
        """Return source values from this exact upload, outside native query fields."""
        upload = self._get(task_id)
        return [dict(item) for item in upload.source_observations.get(locus_id, [])]

    def attach_run(self, task_id: str, run_id: str) -> None:
        if not _TASK_ID.fullmatch(task_id):
            raise JobNotFoundError("analysis upload was not found")
        with self._lock:
            upload = self._uploads.get(task_id)
            if upload is None:
                raise JobNotFoundError("analysis upload was not found")
            self._uploads.move_to_end(task_id)
            upload.run_id = run_id

    def ensure_run(self, task_id: str, create: Callable[[], str]) -> str:
        """Atomically create at most one analysis run for a legacy upload."""

        if not _TASK_ID.fullmatch(task_id):
            raise JobNotFoundError("analysis upload was not found")
        with self._lock:
            upload = self._uploads.get(task_id)
            if upload is None:
                raise JobNotFoundError("analysis upload was not found")
            self._uploads.move_to_end(task_id)
            if upload.run_id is None:
                upload.run_id = create()
            return upload.run_id

    def run_id(self, task_id: str) -> str:
        upload = self._get(task_id)
        if upload.run_id is None:
            raise JobNotFoundError("analysis run is not available")
        return upload.run_id

    def upload(self, task_id: str) -> AnalysisUpload:
        return self._get(task_id)

    def _get(self, task_id: str) -> AnalysisUpload:
        if not _TASK_ID.fullmatch(task_id):
            raise JobNotFoundError("analysis upload was not found")
        with self._lock:
            upload = self._uploads.get(task_id)
            if upload is None:
                raise JobNotFoundError("analysis upload was not found")
            self._uploads.move_to_end(task_id)
            return upload

    def _evictable(self, upload: AnalysisUpload) -> bool:
        if upload.run_id is None:
            return True
        resolver = self._run_is_terminal
        if resolver is None:
            return False
        try:
            return bool(resolver(upload.run_id))
        except Exception:
            # Never discard a possibly active upload when the activity check is
            # unavailable. The caller receives the explicit capacity error.
            return False


def project_results(
    results: list[Mapping[str, Any]], queries: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if len(results) != len(queries):
        raise JobValidationError(
            "analysis result count does not match the submitted locus count"
        )
    projected: list[dict[str, Any]] = []
    for index, (result, query) in enumerate(zip(results, queries, strict=False), start=1):
        result = live_model_only_result(result)
        variants = query.get("variants", [])
        if not isinstance(variants, list):
            variants = []
        first_variant = _lead_variant(variants)
        ranked = result.get("ranked_genes", [])
        ranked = [item for item in ranked if isinstance(item, Mapping)] if isinstance(ranked, (list, tuple)) else []
        displayed = ranked[:parameter("ranking.max_ranked_genes")]
        gene_ids = [
            str(item.get("gene_id"))
            for item in displayed if item.get("gene_id")
        ]
        evidence = result.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        symbol_map = _gene_symbol_map(evidence)
        gene_labels = [_display_gene(gene_id, symbol_map) for gene_id in gene_ids]
        gene_labels_with_ids = [
            _display_gene(gene_id, symbol_map, include_id=True)
            for gene_id in gene_ids
        ]
        pmids = sorted(
            {
                str(item.get("pmid"))
                for item in evidence
                if isinstance(item, Mapping) and item.get("pmid")
            }
        )
        raw_status = str(result.get("status", "INSUFFICIENT_EVIDENCE")).upper()
        display_status = "Prioritized" if raw_status == "PRIORITIZED" else "Needs review"
        if model_ranking_unavailable(result):
            display_status = "Model ranking incomplete"
            raw_status = "RANKING_UNAVAILABLE"
        p_value = first_variant.get("p_value")
        neg_log_p = None
        try:
            numeric_p = float(p_value)
            if numeric_p > 0:
                neg_log_p = round(-math.log10(numeric_p), parameter("plot.threshold_decimal_places"))
        except (TypeError, ValueError):
            numeric_p = None
        rationales = [
            str(item.get("rationale"))
            for item in displayed
            if isinstance(item, Mapping) and item.get("rationale")
        ]
        summary = str(result.get("summary") or "No evidence summary was returned.")
        evidence_summary = summary
        if rationales:
            evidence_summary += "\n\n**Ranked gene evidence**\n\n" + "\n".join(
                f"- {text}" for text in rationales
            )
        kinds = [str(item.get("kind", "")) for item in evidence if isinstance(item, Mapping)]
        datasets = _unique_text(
            str(item.get("dataset", ""))
            for item in evidence
            if isinstance(item, Mapping)
        )
        if kinds or datasets:
            kind_counts = {
                kind: kinds.count(kind) for kind in sorted(set(kinds)) if kind
            }
            coverage = ", ".join(
                f"{kind} ({count})" for kind, count in kind_counts.items()
            )
            evidence_summary += (
                "\n\n**Evidence coverage**\n\n"
                f"- Records: {len(evidence)}\n"
                f"- Kinds: {coverage or 'None'}\n"
                f"- Datasets: {', '.join(datasets) if datasets else 'None'}\n"
                f"- Mode/release: {result.get('data_mode', 'unknown')} / "
                f"{result.get('data_release', 'unknown')}"
            )
        l2g_ids = _gene_ids_for_kinds(evidence, {"l2g"})
        enhancer_ids = _gene_ids_for_kinds(evidence, {"regulatory"})
        go_terms = _evidence_values(
            evidence,
            kinds={"pathway"},
            dataset_markers={"go"},
            keys={"go", "goTerms", "go_terms", "term"},
        )
        pathways = _evidence_values(
            evidence,
            kinds={"pathway"},
            dataset_markers={"pathway", "reactome"},
            keys={"pathways", "pathway", "pathwayName", "pathway_name"},
        )
        expression_tissues = _evidence_values(
            evidence,
            kinds={"expression"},
            keys={"tissues", "tissue", "biosample", "biosampleName"},
        )
        drugs = _evidence_values(
            evidence,
            kinds={"drug"},
            entity_type="drug",
            keys={"drugName", "drug_name", "prefName"},
        )
        association_labels = _evidence_values(
            evidence,
            kinds={"gwas", "study", "target_disease", "ontology"},
            entity_type="disease",
            keys={
                "trait",
                "traitReported",
                "traitFromSource",
                "diseaseName",
                "disease_name",
            },
        )
        positions = [
            {
                "chromosome": str(item.get("chromosome")),
                "position": int(item.get("position")),
            }
            for item in variants
            if isinstance(item, Mapping)
            and item.get("chromosome")
            and item.get("position") is not None
        ]
        position_values = [int(item["position"]) for item in positions]
        limitations_value = result.get("limitations", [])
        limitations = (
            [str(item) for item in limitations_value if str(item).strip()]
            if isinstance(limitations_value, list)
            else []
        )
        projected.append(
            {
                "locus_no": index,
                "locus_id": str(query.get("locus_id", index)),
                "status": display_status,
                "decision_status": raw_status,
                "lead_rsid": str(first_variant.get("variant_id") or "-"),
                "chr": first_variant.get("chromosome") or "-",
                "pos": first_variant.get("position") or "-",
                "lead_pos": first_variant.get("position") or "-",
                "start_pos": min(position_values) if position_values else "-",
                "end_pos": max(position_values) if position_values else "-",
                "p_value": numeric_p,
                "neg_log_p": neg_log_p,
                "snp_count": len(variants),
                "snp_positions": positions,
                "top_gene": gene_labels[0] if gene_labels else "-",
                "top_gene_id": gene_ids[0] if gene_ids else "-",
                "candidate_genes": (
                    ", ".join(gene_labels_with_ids) if gene_ids else
                    "Not ranked · evidence retained" if model_ranking_unavailable(result) else "None"
                ),
                "candidate_gene_ids": gene_ids,
                "ranked_count": len(ranked),
                "displayed_ranked_count": len(displayed),
                "ranked_gene_ids": [str(item["gene_id"]) for item in ranked if item.get("gene_id")],
                "l2g_genes": ", ".join(
                    _display_gene(value, symbol_map, include_id=True)
                    for value in l2g_ids
                ) or "None",
                "enhancer_genes": ", ".join(
                    _display_gene(value, symbol_map, include_id=True)
                    for value in enhancer_ids
                ) or "None",
                "go_terms": "; ".join(go_terms) or "None",
                "pathways": "; ".join(pathways) or "None",
                "expression_tissues": "; ".join(expression_tissues) or "None",
                "drugs": "; ".join(drugs) or "None",
                "gwas_traits": (
                    "; ".join(association_labels)
                    if association_labels
                    else "No direct GWAS or target-disease associations were returned."
                ),
                "gwas_associations": association_labels,
                "llm_interpretation": evidence_summary,
                "references": "; ".join(f"PMID:{pmid}" for pmid in pmids),
                "key_references": [f"PMID:{pmid}" for pmid in pmids],
                "direct_associations": association_labels,
                "evidence_count": len(evidence),
                "evidence_datasets": datasets,
                "data_mode": str(result.get("data_mode", "unknown")),
                "data_release": str(result.get("data_release", "unknown")),
                "limitations": limitations,
            }
        )
    return projected


def _lead_variant(variants: list[Any]) -> Mapping[str, Any]:
    candidates = [item for item in variants if isinstance(item, Mapping)]
    if not candidates:
        return {}

    def key(item: Mapping[str, Any]) -> tuple[bool, float, str]:
        raw = item.get("p_value")
        try:
            p_value = float(raw)
        except (TypeError, ValueError):
            p_value = math.inf
        return raw is None or not math.isfinite(p_value), p_value, str(
            item.get("variant_id", "")
        )

    return min(candidates, key=key)


def _gene_symbol_map(evidence: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        gene_id = str(item.get("gene_id") or "").strip()
        payload = item.get("payload")
        if not gene_id or not isinstance(payload, Mapping):
            continue
        labels = _nested_values(
            payload,
            {"approvedSymbol", "approved_symbol", "geneSymbol", "gene_symbol"},
            limit=4,
        )
        labels.extend(_entity_labels(payload, "target", identifier=gene_id, limit=4))
        symbol = next(
            (
                value
                for value in labels
                if value and value.casefold() != gene_id.casefold()
            ),
            None,
        )
        if symbol:
            result.setdefault(gene_id, symbol)
    return result


def _display_gene(
    gene_id: str, symbol_map: Mapping[str, str], *, include_id: bool = False
) -> str:
    symbol = symbol_map.get(gene_id)
    if not symbol:
        return gene_id
    return f"{symbol} ({gene_id})" if include_id else symbol


def _gene_ids_for_kinds(evidence: list[Any], kinds: set[str]) -> list[str]:
    return _unique_text(
        str(item.get("gene_id") or "")
        for item in evidence
        if isinstance(item, Mapping)
        and str(item.get("kind", "")).casefold() in kinds
    )


def _evidence_values(
    evidence: list[Any],
    *,
    kinds: set[str],
    keys: set[str],
    dataset_markers: set[str] | None = None,
    entity_type: str | None = None,
    limit: int = 24,
) -> list[str]:
    values: list[str] = []
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind", "")).casefold()
        dataset = str(item.get("dataset", "")).casefold()
        if kind not in kinds:
            continue
        if dataset_markers and not any(marker in dataset for marker in dataset_markers):
            continue
        payload = item.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if entity_type:
            values.extend(_entity_labels(payload, entity_type, limit=limit))
        values.extend(_nested_values(payload, keys, limit=limit))
        values = _unique_text(values)[:limit]
        if len(values) >= limit:
            break
    return values


def _entity_labels(
    value: Any,
    entity_type: str,
    *,
    identifier: str | None = None,
    limit: int = 24,
) -> list[str]:
    result: list[str] = []
    stack = [value]
    visited = 0
    while stack and visited < 2_000 and len(result) < limit:
        current = stack.pop()
        visited += 1
        if isinstance(current, Mapping):
            current_type = str(current.get("type", "")).casefold()
            current_id = str(current.get("id", "")).strip()
            if current_type in {entity_type, "gene" if entity_type == "target" else entity_type}:
                if not identifier or current_id.casefold() == identifier.casefold():
                    label = current.get("label") or current.get("name") or current_id
                    if isinstance(label, (str, int, float)) and str(label).strip():
                        result.append(str(label).strip())
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return _unique_text(result)[:limit]


def _nested_values(value: Any, keys: set[str], *, limit: int) -> list[str]:
    result: list[str] = []
    stack = [value]
    visited = 0
    while stack and visited < 2_000 and len(result) < limit:
        current = stack.pop()
        visited += 1
        if isinstance(current, Mapping):
            for key, nested in current.items():
                if str(key) in keys:
                    result.extend(_human_values(nested, limit=limit - len(result)))
                stack.append(nested)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return _unique_text(result)[:limit]


def _human_values(value: Any, *, limit: int) -> list[str]:
    if limit <= 0 or value is None or isinstance(value, bool):
        return []
    if isinstance(value, (str, int, float)):
        text = " ".join(str(value).split())[:300]
        return [text] if text else []
    if isinstance(value, Mapping):
        preferred = (
            value.get("label")
            or value.get("name")
            or value.get("approvedSymbol")
            or value.get("id")
            or value.get("term")
        )
        if preferred is not None:
            return _human_values(preferred, limit=limit)
        result: list[str] = []
        for nested in value.values():
            result.extend(_human_values(nested, limit=limit - len(result)))
            if len(result) >= limit:
                break
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for nested in value:
            result.extend(_human_values(nested, limit=limit - len(result)))
            if len(result) >= limit:
                break
        return result
    return []


def _unique_text(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split())[:300]
        normalized = text.casefold()
        if text and normalized not in seen:
            seen.add(normalized)
            result.append(text)
    return result


def _parse_locus_text(
    text: str,
    *, source_observations: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[str]]:
    text = _strip_preamble(text)
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    delimiter = _detect_delimiter(first_line)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise JobValidationError("locus file has no header")
    normalized_fields = [_normalize_header(value) for value in reader.fieldnames]
    if any(not value for value in normalized_fields):
        raise JobValidationError("locus file contains an empty column name")
    if len(normalized_fields) != len(set(normalized_fields)):
        raise JobValidationError("locus file contains duplicate column names")
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    positions: list[dict[str, Any]] = []
    original_snp_count = 0
    zero_pvalues_capped = 0
    missing_pvalues = 0
    coordinate_variant_count = 0
    for row_index, raw_row in enumerate(reader, start=2):
        if raw_row.get(None):
            raise JobValidationError(f"row {row_index} contains too many columns")
        row = {
            _normalize_header(key): str(value or "").strip()
            for key, value in raw_row.items()
            if key
        }
        if not any(row.values()):
            continue
        if original_snp_count >= MAX_QUERIES * MAX_VARIANTS_PER_QUERY:
            raise JobValidationError("locus file exceeds the 5,000-variant limit")
        variant_id = _first(row, "variant_id", "rsid", "snp_id")
        chromosome = _clean_chromosome(_first(row, "chromosome", "chr"))
        position = _optional_positive_int(_first(row, "position", "pos"), row_index)
        ref = _first(row, "ref", "reference_allele").upper()
        alt = _first(row, "alt", "alternate_allele").upper()
        if not variant_id and chromosome and position and ref and alt:
            variant_id = f"{chromosome}_{position}_{ref}_{alt}"
        if not variant_id or not _VARIANT_ID.fullmatch(variant_id):
            raise JobValidationError(f"row {row_index} has no valid rsID or chr/pos/ref/alt")
        if variant_id.casefold().startswith("rs"):
            variant_id = "rs" + variant_id[2:]
        else:
            id_chromosome, id_position, id_ref, id_alt = re.split(
                r"[_:]", re.sub(r"^chr", "", variant_id, flags=re.IGNORECASE)
            )
            id_chromosome = _clean_chromosome(id_chromosome)
            parsed_id_position = _optional_positive_int(id_position, row_index)
            id_ref, id_alt = id_ref.upper(), id_alt.upper()
            if (
                (chromosome and chromosome != id_chromosome)
                or (position is not None and position != parsed_id_position)
                or (ref and ref != id_ref)
                or (alt and alt != id_alt)
            ):
                raise JobValidationError(
                    f"row {row_index} variant identifier disagrees with coordinate/allele columns"
                )
            chromosome, position = id_chromosome, parsed_id_position
            variant_id = f"{chromosome}_{position}_{id_ref}_{id_alt}"
            coordinate_variant_count += 1
        p_value = _optional_probability(_first(row, "p", "pvalue", "p_value"), row_index)
        if p_value is None:
            missing_pvalues += 1
        if p_value == 0:
            p_value = parameter("plot.p_value_floor")
            zero_pvalues_capped += 1
        locus_id = _first(row, "locus_id", "locus_no") or str(row_index - 1)
        if len(locus_id) > MAX_LOCUS_ID_LENGTH:
            raise JobValidationError(
                f"row {row_index} locus_id exceeds the "
                f"{MAX_LOCUS_ID_LENGTH}-character limit"
            )
        variant = {
            "variant_id": variant_id,
            "p_value": p_value,
            "chromosome": chromosome or None,
            "position": position,
        }
        effects = {}
        for field_name, aliases in (("beta", ("beta",)), ("odds_ratio", ("odds_ratio", "or")),
                                    ("se", ("se", "standard_error"))):
            raw = _first(row, *aliases)
            if raw.casefold() in {"", "na", "nan", "none", "null", "."}:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError):
                raise JobValidationError(f"row {row_index} has an invalid {field_name}") from None
            if not math.isfinite(value) or field_name == "odds_ratio" and value <= 0 or field_name == "se" and value < 0:
                raise JobValidationError(f"row {row_index} has an invalid {field_name}")
            effects[field_name] = value
        for field_name, aliases in (
            ("effect_allele", ("effect_allele", "effectallele", "ea")),
            ("other_allele", ("other_allele", "otherallele", "non_effect_allele", "nea")),
            ("allele1", ("allele1", "a1")), ("allele2", ("allele2", "a2")),
        ):
            raw = _first(row, *aliases).upper()
            if raw in {"", "N", "NA", "NAN", "NONE", "NULL", "."}:
                continue
            if not re.fullmatch(r"[ACGT]{1,32}", raw):
                raise JobValidationError(f"row {row_index} has an invalid {field_name}")
            effects[field_name] = raw
        if effects:
            variant["_source_effects"] = effects
        grouped.setdefault(locus_id, []).append(variant)
        original_snp_count += 1
        if chromosome and position:
            positions.append({"chromosome": chromosome, "position": position})

    if not grouped:
        raise JobValidationError("locus file contains no data rows")
    if len(grouped) > MAX_QUERIES:
        raise JobValidationError(f"locus file exceeds the {MAX_QUERIES}-locus limit")
    warnings: list[str] = []
    if zero_pvalues_capped:
        warnings.append(
            f"Capped {zero_pvalues_capped} zero or underflowed p-value(s) at {parameter('plot.p_value_floor'):g}."
        )
    if missing_pvalues:
        warnings.append(
            f"{missing_pvalues} row(s) have no p-value; loci without any supplied "
            "p-value will stop at Needs review before evidence retrieval."
        )
    if coordinate_variant_count:
        warnings.append(
            f"Interpreting {coordinate_variant_count} coordinate variant(s) as "
            "GRCh38/hg38; no implicit liftover was performed."
        )
    queries: list[dict[str, Any]] = []
    for locus_id, variants in grouped.items():
        deduplicated: OrderedDict[str, dict[str, Any]] = OrderedDict()
        duplicate_count = 0
        for variant in variants:
            key = str(variant["variant_id"]).casefold()
            existing = deduplicated.get(key)
            if existing is None:
                deduplicated[key] = variant
                continue
            duplicate_count += 1
            current_p = variant.get("p_value")
            existing_p = existing.get("p_value")
            if current_p is not None and (
                existing_p is None or float(current_p) < float(existing_p)
            ):
                deduplicated[key] = variant
        variants = list(deduplicated.values())
        if duplicate_count:
            warnings.append(
                f"Locus {locus_id}: merged {duplicate_count} duplicate variant row(s), "
                "retaining the smallest supplied p-value."
            )
        if len(variants) > MAX_VARIANTS_PER_QUERY:
            original_count = len(variants)
            variants = sorted(
                variants,
                key=lambda item: (
                    item.get("p_value") is None,
                    float(item["p_value"])
                    if item.get("p_value") is not None
                    else math.inf,
                    str(item["variant_id"]),
                ),
            )[:MAX_VARIANTS_PER_QUERY]
            warnings.append(
                f"Locus {locus_id}: retained the {MAX_VARIANTS_PER_QUERY} variants "
                f"with the smallest supplied p-values from {original_count}."
            )
        queries.append(
            {
                "locus_id": locus_id,
                "trait": "pending phenotype",
                "variants": [{key: value for key, value in item.items() if key != "_source_effects"} for item in variants],
                "candidate_genes": [],
            }
        )
        if source_observations is not None:
            source_observations[locus_id] = [
                {**{key: value for key, value in item.items() if key != "_source_effects"}, **item.get("_source_effects", {})}
                for item in variants
            ]
    return queries, positions, original_snp_count, warnings


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _strip_preamble(text: str) -> str:
    lines = text.splitlines(keepends=True)
    first_data_index = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("##") or re.match(r"^#\s", stripped):
            first_data_index = index + 1
            continue
        break
    return "".join(lines[first_data_index:])


def _detect_delimiter(header: str) -> str:
    counts = {",": 0, "\t": 0}
    in_quotes = False
    index = 0
    while index < len(header):
        character = header[index]
        if character == '"':
            if in_quotes and index + 1 < len(header) and header[index + 1] == '"':
                index += 1
            else:
                in_quotes = not in_quotes
        elif not in_quotes and character in counts:
            counts[character] += 1
        index += 1
    if in_quotes:
        raise JobValidationError("locus header contains an unterminated quoted field")
    if counts[","] == counts["\t"]:
        raise JobValidationError("could not determine whether the locus file is CSV or TSV")
    return "\t" if counts["\t"] > counts[","] else ","


def _first(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def _clean_chromosome(value: str) -> str:
    normalized = re.sub(r"^chr", "", value, flags=re.IGNORECASE).upper()
    if not normalized:
        return ""
    if normalized not in {*(str(value) for value in range(1, 23)), "X", "Y", "MT"}:
        raise JobValidationError(f"invalid chromosome: {value}")
    return normalized


def _optional_positive_int(value: str, row_index: int) -> int | None:
    if not value:
        return None
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise JobValidationError(f"row {row_index} has an invalid position") from exc
    if not decimal_value.is_finite() or decimal_value != decimal_value.to_integral_value():
        raise JobValidationError(f"row {row_index} has a non-integer position")
    if not 0 < decimal_value < 2**53:
        raise JobValidationError(f"row {row_index} position is not a positive browser-safe integer")
    return int(decimal_value)


def _optional_probability(value: str, row_index: int) -> float | None:
    if not value:
        return None
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise JobValidationError(f"row {row_index} has an invalid p-value") from exc
    if not decimal_value.is_finite() or not 0 <= decimal_value <= 1:
        raise JobValidationError(f"row {row_index} has a p-value outside 0..1")
    return float(decimal_value)
