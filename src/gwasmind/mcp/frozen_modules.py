from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..audit import sha256_json
from ..configuration import parameter
from ..models import EvidenceKind, LocusQuery, VariantObservation
from ..providers.frozen import DuckDBFrozenOpenTargetsProvider, _normalized_variant_id
from ..qtl import qtl_study_type_from_payload, requested_qtl_study_types
from ..region_gate import local_gene_window_label
from .contracts import SemanticTool
from .regions import grch38_region_from_variant_ids
from .service import ModuleAdapter, ModuleResult, ModuleStatus, SemanticToolService

SOURCE = "Open Targets Platform frozen download"

MODULE_SOURCES: dict[tuple[SemanticTool, str], frozenset[str]] = {
    (SemanticTool.GET_LOCUS_CONTEXT, "locus.variant"): frozenset({"variant"}),
    (SemanticTool.GET_LOCUS_CONTEXT, "locus.study"): frozenset(
        {"variant", "credible_set", "study"}
    ),
    (SemanticTool.GET_LOCUS_CONTEXT, "locus.credible_set"): frozenset(
        {"variant", "credible_set", "study"}
    ),
    (SemanticTool.GET_LOCUS_CONTEXT, "locus.fine_mapping"): frozenset(
        {"variant", "credible_set", "study"}
    ),
    (SemanticTool.GET_LOCUS_GENE_EVIDENCE, "locus.variant_consequence"): frozenset(
        {"variant"}
    ),
    (SemanticTool.GET_LOCUS_GENE_EVIDENCE, "locus.qtl"): frozenset(
        {"variant", "credible_set", "study"}
    ),
    (SemanticTool.GET_LOCUS_GENE_EVIDENCE, "locus.colocalisation"): frozenset(
        {"variant", "credible_set", "study", "colocalisation_coloc"}
    ),
    (SemanticTool.GET_LOCUS_GENE_EVIDENCE, "locus.enhancer_to_gene"): frozenset(
        {"variant", "enhancer_to_gene"}
    ),
    (SemanticTool.GET_LOCUS_GENE_EVIDENCE, "locus.l2g"): frozenset(
        {"variant", "credible_set", "l2g_prediction"}
    ),
    (SemanticTool.GET_TARGET_PROFILES, "target.expression"): frozenset(
        {"expression"}
    ),
    (SemanticTool.GET_TARGET_PROFILES, "context.tissue"): frozenset({"biosample"}),
    (SemanticTool.GET_DISEASE_PROFILES, "disease.core"): frozenset({"disease"}),
    (SemanticTool.GET_DISEASE_PROFILES, "disease.ontology"): frozenset({"disease"}),
    (SemanticTool.GET_DRUG_EVIDENCE, "drug.clinical_target"): frozenset(
        {"known_drug"}
    ),
    (SemanticTool.ANALYZE_GENE_SET, "gene_set.go"): frozenset(
        {"target", "go"}
    ),
    (SemanticTool.ANALYZE_GENE_SET, "gene_set.pathway"): frozenset(
        {"target", "reactome"}
    ),
    (SemanticTool.ANALYZE_GENE_SET, "gene_set.tissue_coverage"): frozenset(
        {"expression"}
    ),
    (SemanticTool.ANALYZE_GENE_SET, "gene_set.evidence_coverage"): frozenset(
        {"target", "expression"}
    ),
}


class FrozenSemanticModules:
    """Bounded semantic adapters for the local Open Targets Parquet snapshot."""

    def __init__(self, provider: DuckDBFrozenOpenTargetsProvider):
        self.provider = provider
        self._present_datasets = set(
            provider.frozen_dataset_discovery.recognized_source_names
        )
        for dataset in provider.frozen_dataset_discovery.datasets:
            if not dataset.ready:
                self._present_datasets.difference_update(dataset.source_names)
        self._target_cache: OrderedDict[str, tuple[list[dict[str, Any]], bool]] = (
            OrderedDict()
        )
        self._locus_cache: OrderedDict[
            str, dict[EvidenceKind, tuple[list[dict[str, Any]], bool]]
        ] = OrderedDict()

    def adapters(self) -> tuple[ModuleAdapter, ...]:
        adapters: list[ModuleAdapter] = [
            ModuleAdapter(
                "entity.resolve",
                SemanticTool.RESOLVE_ENTITIES,
                self.resolve_entities,
            ),
            ModuleAdapter(
                "locus.variant",
                SemanticTool.GET_LOCUS_CONTEXT,
                self.locus_variant,
            ),
            ModuleAdapter(
                "locus.study",
                SemanticTool.GET_LOCUS_CONTEXT,
                self.locus_study,
            ),
            ModuleAdapter(
                "locus.credible_set",
                SemanticTool.GET_LOCUS_CONTEXT,
                self.locus_credible_set,
            ),
            ModuleAdapter(
                "locus.fine_mapping",
                SemanticTool.GET_LOCUS_CONTEXT,
                self.locus_fine_mapping,
            ),
            ModuleAdapter(
                "locus.variant_consequence",
                SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                self.variant_consequences,
            ),
            ModuleAdapter(
                "locus.qtl",
                SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                self.locus_qtl,
            ),
            ModuleAdapter(
                "locus.colocalisation",
                SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                self.locus_colocalisation,
            ),
            ModuleAdapter(
                "locus.enhancer_to_gene",
                SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                self.locus_enhancer_to_gene,
            ),
            ModuleAdapter(
                "locus.l2g",
                SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                self.locus_l2g,
            ),
            ModuleAdapter(
                "target.expression",
                SemanticTool.GET_TARGET_PROFILES,
                self.target_expression,
            ),
            ModuleAdapter(
                "context.tissue",
                SemanticTool.GET_TARGET_PROFILES,
                self.biosample_context,
            ),
            ModuleAdapter(
                "disease.core",
                SemanticTool.GET_DISEASE_PROFILES,
                self.disease_core,
            ),
            ModuleAdapter(
                "disease.ontology",
                SemanticTool.GET_DISEASE_PROFILES,
                self.disease_ontology,
            ),
            ModuleAdapter(
                "drug.clinical_target",
                SemanticTool.GET_DRUG_EVIDENCE,
                self.known_drug,
            ),
            ModuleAdapter(
                "gene_set.go",
                SemanticTool.ANALYZE_GENE_SET,
                self.gene_set_go,
            ),
            ModuleAdapter(
                "gene_set.pathway",
                SemanticTool.ANALYZE_GENE_SET,
                self.gene_set_pathway,
            ),
            ModuleAdapter(
                "gene_set.tissue_coverage",
                SemanticTool.ANALYZE_GENE_SET,
                self.gene_set_tissue_coverage,
            ),
            ModuleAdapter(
                "gene_set.evidence_coverage",
                SemanticTool.ANALYZE_GENE_SET,
                self.gene_set_evidence_coverage,
            ),
        ]
        target_modules = (
            "target.core",
            "target.annotation",
            "target.function",
            "target.subcellular_location",
            "target.constraint",
            "target.homologues",
            "target.tractability",
            "target.safety",
            "target.chemical_probes",
            "target.tep",
            "target.hallmarks",
            "target.go",
            "target.pathway",
        )
        adapters.extend(
            ModuleAdapter(
                module,
                SemanticTool.GET_TARGET_PROFILES,
                self._target_handler(module),
            )
            for module in target_modules
        )
        present = self._present_datasets
        filtered: list[ModuleAdapter] = []
        for adapter in adapters:
            if adapter.tool is SemanticTool.RESOLVE_ENTITIES:
                if present.intersection(
                    {
                        "target",
                        "variant",
                        "disease",
                        "known_drug",
                        "study",
                        "biosample",
                        "go",
                        "reactome",
                    }
                ):
                    filtered.append(adapter)
                continue
            required = MODULE_SOURCES.get((adapter.tool, adapter.module))
            if required is None and adapter.module.startswith("target."):
                required = frozenset(
                    {"target", "go"}
                    if adapter.module == "target.go"
                    else {"target", "reactome"}
                    if adapter.module == "target.pathway"
                    else {"target"}
                )
            if required is not None and required.issubset(present):
                filtered.append(adapter)
        return tuple(filtered)

    def service(
        self,
        *,
        policy_sha256: str = "unknown",
        blocked_modules: frozenset[str] = frozenset(),
    ) -> SemanticToolService:
        return SemanticToolService(
            provider="frozen",
            data_release=self.provider.capabilities.release,
            snapshot_sha256=self.provider.capabilities.snapshot_hash,
            schema_sha256=self.provider.capabilities.schema_hash,
            policy_sha256=policy_sha256,
            adapters=self.adapters(),
            blocked_modules=blocked_modules,
            limits={
                "max_variants": 100,
                "max_targets": 200,
                "max_discovered_targets": 40,
                "max_discovered_l2g_targets_per_locus": 10,
                "max_page_size": 500,
                "duckdb_threads": self.provider.max_threads,
                "duckdb_memory": self.provider.memory_limit,
            },
            dataset_capabilities={
                item.canonical_name or item.source_names[0]: item.to_dict()
                for item in (
                    *self.provider.frozen_dataset_discovery.datasets,
                    *self.provider.frozen_dataset_discovery.unknown,
                )
            },
            upstream_metadata={
                "source": SOURCE,
                "catalog_release": self.provider.frozen_dataset_discovery.catalog_release,
                "catalog_source": self.provider.frozen_dataset_discovery.catalog_source,
            },
        )

    def resolve_entities(self, arguments: Mapping[str, Any]) -> ModuleResult:
        max_hits = _bounded_int(arguments.get("max_hits", 5), 1, 20, "max_hits")
        records: list[dict[str, Any]] = []
        any_hit = False
        for item in arguments.get("queries", ()):
            if not isinstance(item, Mapping):
                raise ValueError("each entity query must be an object")
            text = _required_text(item.get("text"), "query text")
            requested_types = tuple(
                item.get(
                    "types",
                    ("target", "variant", "disease", "drug", "study", "biosample", "pathway"),
                )
            )
            hits: list[dict[str, Any]] = []
            for entity_type in requested_types:
                hits.extend(self._resolve_type(text, str(entity_type), max_hits + 1))
            unique_hits = _deduplicate_hits(hits)
            truncated = len(unique_hits) > max_hits
            hits = unique_hits[:max_hits]
            any_hit = any_hit or bool(hits)
            records.append(
                {
                    "query": text,
                    "status": (
                        "no_match"
                        if not hits
                        else "resolved"
                        if len(hits) == 1 and hits[0]["match"] == "exact" and not truncated
                        else "ambiguous"
                    ),
                    "hits": hits,
                    "auto_selectable": bool(
                        len(hits) == 1 and hits[0]["match"] == "exact" and not truncated
                    ),
                    "truncated": truncated,
                }
            )
        return ModuleResult(
            "entity.resolve",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=tuple(records),
            warnings=()
            if any_hit
            else ("No exact or normalized entity match was found.",),
        )

    def locus_variant(self, arguments: Mapping[str, Any]) -> ModuleResult:
        rows, truncated = self._variant_rows(arguments)
        records = tuple(
            _semantic_record(
                module="locus.variant",
                dataset="variant",
                relation="variant_annotation",
                subject={"type": "variant", "id": row["variantId"]},
                raw=row,
                release=self.provider.capabilities.release,
                measurements={"allele_frequencies": row.get("alleleFrequencies")},
            )
            for row in rows
        )
        return _module_result("locus.variant", records, truncated=truncated)

    def variant_consequences(self, arguments: Mapping[str, Any]) -> ModuleResult:
        rows, parent_truncated = self._variant_rows(arguments)
        candidates = set(_string_list(arguments.get("candidate_gene_ids", ())))
        page_size = _page_size(arguments)
        records: list[dict[str, Any]] = []
        truncated = parent_truncated
        for row in rows:
            for consequence in row.get("transcriptConsequences") or ():
                if not isinstance(consequence, Mapping):
                    continue
                gene_id = _optional_text(consequence.get("targetId"))
                symbol = _optional_text(consequence.get("approvedSymbol"))
                if candidates and not ({gene_id, symbol} - {None}).intersection(candidates):
                    continue
                records.append(
                    _semantic_record(
                        module="locus.variant_consequence",
                        dataset="variant",
                        relation="variant_transcript_consequence",
                        subject={"type": "variant", "id": row["variantId"]},
                        object_value={"type": "target", "id": gene_id, "label": symbol},
                        raw=dict(consequence),
                        release=self.provider.capabilities.release,
                        measurements={
                            "consequence_score": consequence.get("consequenceScore"),
                            "distance_from_tss": consequence.get("distanceFromTss"),
                            "sift": consequence.get("siftPrediction"),
                            "polyphen": consequence.get("polyphenPrediction"),
                        },
                    )
                )
                if len(records) > page_size:
                    truncated = True
                    records = records[:page_size]
                    break
            if len(records) >= page_size and truncated:
                break
        warnings: list[str] = []
        include_region_context = arguments.get("allow_region_fallback", False)
        if not isinstance(include_region_context, bool):
            raise TypeError("allow_region_fallback must be boolean")
        if include_region_context:
            region_records, region_warnings = self._region_gene_context(arguments)
            records.extend(region_records)
            warnings.extend(region_warnings)
        return _module_result(
            "locus.variant_consequence",
            tuple(records),
            truncated=truncated,
            warnings=tuple(warnings),
        )

    def _region_gene_context(
        self,
        arguments: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Read the ±1 Mb annotated-gene gate input from the local snapshot."""

        region = grch38_region_from_variant_ids(
            _string_list(arguments.get("variant_ids", ()))
        )
        if region is None:
            return [self._region_gene_summary(None, status="unavailable")], [
                (
                    f"The frozen ±{local_gene_window_label()} annotated-gene check was unavailable because one "
                    "same-chromosome GRCh38 interval could not be derived."
                )
            ]
        if "target" not in self._present_datasets:
            return [self._region_gene_summary(region, status="unavailable")], [
                (
                    "The frozen snapshot has no target genomic-location dataset for the "
                    f"±{local_gene_window_label()} annotated-gene check."
                )
            ]

        query_start = max(1, region.start - parameter("analysis.local_gene_window_bp"))
        query_end = region.end + parameter("analysis.local_gene_window_bp")
        connection = None
        try:
            path = self.provider._dataset_glob("target")
            connection = self.provider._connect()
            sql = f"""
                SELECT id, approvedSymbol, biotype, genomicLocation,
                       COUNT(*) OVER () AS total_gene_count
                FROM read_parquet('{path}')
                WHERE genomicLocation IS NOT NULL
                  AND replace(lower(genomicLocation.chromosome), 'chr', '') = lower(?)
                  AND genomicLocation."start" <= ?
                  AND genomicLocation."end" >= ?
                ORDER BY
                  CASE WHEN genomicLocation."end" >= ?
                            AND genomicLocation."start" <= ? THEN 0 ELSE 1 END,
                  CASE
                    WHEN genomicLocation."end" < ? THEN ? - genomicLocation."end"
                    WHEN genomicLocation."start" > ? THEN genomicLocation."start" - ?
                    ELSE 0
                  END,
                  genomicLocation."start", id
                LIMIT 10
            """
            result = connection.execute(
                sql,
                [
                    region.chromosome,
                    query_end,
                    query_start,
                    region.start,
                    region.end,
                    region.start,
                    region.start,
                    region.end,
                    region.end,
                ],
            )
            columns = [item[0] for item in result.description]
            rows = [_row(columns, row) for row in result.fetchall()]
            for row in rows:
                location = row.get("genomicLocation")
                gene_id = row.get("id")
                count = row.get("total_gene_count")
                if (
                    not isinstance(gene_id, str) or not gene_id.strip()
                    or isinstance(count, bool) or not isinstance(count, int)
                    or count < len(rows) or not isinstance(location, Mapping)
                ):
                    raise ValueError("malformed regional gene identity or count")
                start, end = location.get("start"), location.get("end")
                if (
                    isinstance(start, bool) or not isinstance(start, int)
                    or isinstance(end, bool) or not isinstance(end, int)
                    or not 1 <= start <= end
                    or str(location.get("chromosome")).upper().removeprefix("CHR") != region.chromosome
                    or start > query_end or end < query_start
                ):
                    raise ValueError("malformed regional gene coordinates")
            if len({row["id"] for row in rows}) != len(rows) or len({row["total_gene_count"] for row in rows}) > 1:
                raise ValueError("inconsistent regional gene census")
        except Exception as exc:  # noqa: BLE001 - optional snapshot schema boundary
            return [self._region_gene_summary(region, status="unavailable")], [
                (
                    f"The frozen ±{local_gene_window_label()} annotated-gene query was unavailable: "
                    f"{type(exc).__name__}."
                )
            ]
        finally:
            if connection is not None:
                connection.close()

        total = int(rows[0]["total_gene_count"]) if rows else 0
        gene_ids = tuple(str(row["id"]) for row in rows)
        records = [
            self._region_gene_summary(
                region,
                status="complete",
                annotated_gene_count=total,
                candidate_gene_ids=gene_ids,
                query_start=query_start,
                query_end=query_end,
                truncated=total > len(rows),
            )
        ]
        for row in rows:
            location = row.get("genomicLocation")
            location = location if isinstance(location, Mapping) else {}
            start = int(location.get("start"))
            end = int(location.get("end"))
            distance = (
                region.start - end
                if end < region.start
                else start - region.end
                if start > region.end
                else 0
            )
            value = _semantic_record(
                module="locus.variant_consequence",
                dataset="frozen_region_gene_context",
                relation="region_overlap" if distance == 0 else "nearby",
                subject={
                    "type": "region",
                    "id": (
                        f"{region.genome_build}:chr{region.chromosome}:"
                        f"{region.start}-{region.end}"
                    ),
                },
                object_value={
                    "type": "target",
                    "id": row["id"],
                    "label": row.get("approvedSymbol"),
                },
                raw={
                    "id": row["id"],
                    "approvedSymbol": row.get("approvedSymbol"),
                    "biotype": row.get("biotype"),
                    "genomicLocation": location,
                },
                release=self.provider.capabilities.release,
                measurements={"distance_to_region": distance},
                context={
                    **region.to_dict(),
                    "query_start": query_start,
                    "query_end": query_end,
                    "flank_bp": parameter("analysis.local_gene_window_bp"),
                    "gene_start": start,
                    "gene_end": end,
                    "strand": location.get("strand"),
                    "biotype": row.get("biotype"),
                },
            )
            value["derived"] = {
                "candidate_basis": "region_overlap" if distance == 0 else "nearby",
                "region_gene_context": True,
                "unranked": True,
                "not_l2g_or_qtl": True,
                "not_direct_locus_to_gene_evidence": True,
            }
            records.append(value)
        return records, [
            (
                f"The frozen ±{local_gene_window_label()} gene context is an unranked locus-informativeness "
                "check; positional proximity is not direct causal-gene evidence."
            )
        ]

    def _region_gene_summary(
        self,
        region,
        *,
        status: str,
        annotated_gene_count: int | None = None,
        candidate_gene_ids: tuple[str, ...] = (),
        query_start: int | None = None,
        query_end: int | None = None,
        truncated: bool = False,
    ) -> dict[str, Any]:
        value = _semantic_record(
            module="locus.variant_consequence",
            dataset="frozen_region_gene_context",
            relation="regional_gene_context_summary",
            subject={
                "type": "region",
                "id": (
                    f"{region.genome_build}:chr{region.chromosome}:"
                    f"{region.start}-{region.end}"
                    if region is not None
                    else "GRCh38:unresolved"
                ),
            },
            raw={},
            release=self.provider.capabilities.release,
            measurements={
                "annotated_gene_count": annotated_gene_count,
                "returned_candidate_count": len(candidate_gene_ids),
            },
            context={
                **(region.to_dict() if region is not None else {}),
                "query_start": query_start,
                "query_end": query_end,
                "flank_bp": parameter("analysis.local_gene_window_bp"),
                "candidate_gene_ids": list(candidate_gene_ids),
            },
        )
        value["derived"] = {
            "region_gene_context": True,
            "hard_gate_input": True,
            "status": status,
            "truncated": truncated,
            "unranked": True,
            "not_direct_locus_to_gene_evidence": True,
        }
        return value

    def locus_study(self, arguments: Mapping[str, Any]) -> ModuleResult:
        records_by_id: dict[str, dict[str, Any]] = {}
        study_ids = _string_list(arguments.get("study_ids", ()))
        if study_ids:
            placeholders = ",".join("?" for _ in study_ids)
            rows, truncated = self._read_rows(
                "study",
                "*",
                f"studyId IN ({placeholders})",
                list(study_ids),
                _page_size(arguments),
            )
            for row in rows:
                records_by_id[str(row["studyId"])] = row
        else:
            locus, truncated = self._locus_records(
                arguments, {EvidenceKind.GWAS, EvidenceKind.QTL}
            )
            for kind in (EvidenceKind.GWAS, EvidenceKind.QTL):
                for row in locus.get(kind, ([], False))[0]:
                    study_id = _optional_text(row.get("studyId"))
                    if study_id:
                        records_by_id[study_id] = {
                            key: row.get(key)
                            for key in (
                                "studyId",
                                "studyType",
                                "traitFromSource",
                                "biosampleId",
                                "pubmedId",
                                "publicationDate",
                                "publicationFirstAuthor",
                                "publicationJournal",
                            )
                        }
        records = tuple(
            _semantic_record(
                module="locus.study",
                dataset="study",
                relation="study_metadata",
                subject={"type": "study", "id": study_id},
                raw=row,
                release=self.provider.capabilities.release,
                context={
                    "study_id": study_id,
                    "study_type": row.get("studyType"),
                    "biosample_id": row.get("biosampleId"),
                    "disease_ids": row.get("diseaseIds"),
                    "publication_date": row.get("publicationDate"),
                    "pmid": row.get("pubmedId"),
                },
            )
            for study_id, row in records_by_id.items()
        )
        return _module_result("locus.study", records, truncated=truncated)

    def locus_credible_set(self, arguments: Mapping[str, Any]) -> ModuleResult:
        rows, truncated = self._combined_locus_rows(arguments)
        records = tuple(
            _semantic_record(
                module="locus.credible_set",
                dataset="credible_set",
                relation="credible_set_membership",
                subject={"type": "variant", "id": row.get("queryVariantId")},
                object_value={"type": "credible_set", "id": row.get("studyLocusId")},
                raw=row,
                release=self.provider.capabilities.release,
                measurements={
                    "member_posterior_probability": (
                        (row.get("matchedLocusMember") or {}).get("posteriorProbability")
                        if isinstance(row.get("matchedLocusMember"), Mapping)
                        else None
                    ),
                    "p_value_mantissa": row.get("pValueMantissa"),
                    "p_value_exponent": row.get("pValueExponent"),
                },
                context={
                    "lead_variant_id": row.get("leadVariantId"),
                    "query_variant_role": row.get("queryVariantRole"),
                    "study_id": row.get("studyId"),
                    "study_type": row.get("studyType"),
                    "study_locus_id": row.get("studyLocusId"),
                    "pmid": row.get("pubmedId"),
                    "publication_date": row.get("publicationDate"),
                },
            )
            for row in rows
        )
        return _module_result("locus.credible_set", records, truncated=truncated)

    def locus_fine_mapping(self, arguments: Mapping[str, Any]) -> ModuleResult:
        rows, truncated = self._combined_locus_rows(arguments)
        records = tuple(
            _semantic_record(
                module="locus.fine_mapping",
                dataset="credible_set",
                relation="fine_mapping_result",
                subject={"type": "credible_set", "id": row.get("studyLocusId")},
                raw={
                    key: row.get(key)
                    for key in (
                        "finemappingMethod",
                        "confidence",
                        "matchedLocusMember",
                        "pValueMantissa",
                        "pValueExponent",
                        "beta",
                    )
                },
                release=self.provider.capabilities.release,
                measurements={
                    "posterior_probability": (
                        (row.get("matchedLocusMember") or {}).get("posteriorProbability")
                        if isinstance(row.get("matchedLocusMember"), Mapping)
                        else None
                    ),
                    "log_bayes_factor": (
                        (row.get("matchedLocusMember") or {}).get("logBF")
                        if isinstance(row.get("matchedLocusMember"), Mapping)
                        else None
                    ),
                },
                context={
                    "fine_mapping_method": row.get("finemappingMethod"),
                    "fine_mapping_confidence_label": row.get("confidence"),
                    "study_id": row.get("studyId"),
                    "study_locus_id": row.get("studyLocusId"),
                    "pmid": row.get("pubmedId"),
                    "publication_date": row.get("publicationDate"),
                },
            )
            for row in rows
        )
        return _module_result("locus.fine_mapping", records, truncated=truncated)

    def locus_qtl(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return self._evidence_module(arguments, EvidenceKind.QTL, "locus.qtl")

    def locus_colocalisation(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return self._evidence_module(
            arguments, EvidenceKind.COLOCALIZATION, "locus.colocalisation"
        )

    def locus_enhancer_to_gene(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return self._evidence_module(
            arguments, EvidenceKind.REGULATORY, "locus.enhancer_to_gene"
        )

    def locus_l2g(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return self._evidence_module(arguments, EvidenceKind.L2G, "locus.l2g")

    def target_expression(self, arguments: Mapping[str, Any]) -> ModuleResult:
        target_ids = _string_list(arguments.get("target_ids", ()))
        if not target_ids:
            raise ValueError("target_ids are required")
        placeholders = ",".join("?" for _ in target_ids)
        rows, truncated = self._read_rows(
            "expression",
            "id, tissues",
            f"id IN ({placeholders})",
            list(target_ids),
            _page_size(arguments),
        )
        records = tuple(
            _semantic_record(
                module="target.expression",
                dataset="expression",
                relation="baseline_expression",
                subject={"type": "target", "id": row["id"]},
                raw=row,
                release=self.provider.capabilities.release,
            )
            for row in rows
            if row.get("tissues")
        )
        return _module_result("target.expression", records, truncated=truncated)

    def biosample_context(self, arguments: Mapping[str, Any]) -> ModuleResult:
        biosample_ids = _string_list(arguments.get("relevant_biosample_ids", ()))
        if not biosample_ids:
            return ModuleResult("context.tissue", ModuleStatus.NO_DATA)
        placeholders = ",".join("?" for _ in biosample_ids)
        rows, truncated = self._read_rows(
            "biosample",
            "*",
            f"biosampleId IN ({placeholders})",
            list(biosample_ids),
            _page_size(arguments),
        )
        records = tuple(
            _semantic_record(
                module="context.tissue",
                dataset="biosample",
                relation="biosample_ontology",
                subject={"type": "biosample", "id": row["biosampleId"]},
                raw=row,
                release=self.provider.capabilities.release,
            )
            for row in rows
        )
        return _module_result("context.tissue", records, truncated=truncated)

    def disease_core(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return self._disease_module(arguments, "disease.core")

    def disease_ontology(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return self._disease_module(arguments, "disease.ontology")

    def known_drug(self, arguments: Mapping[str, Any]) -> ModuleResult:
        conditions: list[str] = []
        parameters: list[str] = []
        for argument, column in (
            ("drug_ids", "drugId"),
            ("target_ids", "targetId"),
            ("disease_ids", "diseaseId"),
        ):
            values = _string_list(arguments.get(argument, ()))
            if values:
                conditions.append(f"{column} IN ({','.join('?' for _ in values)})")
                parameters.extend(values)
        if not conditions:
            raise ValueError("at least one drug, target, or disease identifier is required")
        rows, truncated = self._read_rows(
            "known_drug",
            "*",
            " AND ".join(conditions),
            parameters,
            _page_size(arguments),
        )
        records = tuple(
            _semantic_record(
                module="drug.clinical_target",
                dataset="known_drug",
                relation="legacy_drug_target_disease",
                subject={"type": "drug", "id": row.get("drugId"), "label": row.get("prefName")},
                object_value={
                    "type": "target",
                    "id": row.get("targetId"),
                    "label": row.get("approvedSymbol"),
                },
                raw=row,
                release=self.provider.capabilities.release,
                measurements={"clinical_phase": row.get("phase")},
                context={
                    "disease_id": row.get("diseaseId"),
                    "status": row.get("status"),
                    "mechanism_of_action": row.get("mechanismOfAction"),
                    "legacy_schema": True,
                },
            )
            for row in rows
        )
        return _module_result(
            "drug.clinical_target",
            records,
            truncated=truncated,
            warnings=(
                "The 25.09 known_drug aggregate is a legacy schema and is not "
                "equivalent to the live clinical_target type.",
            ),
        )

    def gene_set_go(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return self._gene_set_enrichment(arguments, kind="go")

    def gene_set_pathway(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return self._gene_set_enrichment(arguments, kind="pathway")

    def _gene_set_enrichment(
        self, arguments: Mapping[str, Any], *, kind: str
    ) -> ModuleResult:
        selected = _string_list(arguments.get("target_ids", ()))
        if len(selected) < 3:
            raise ValueError("gene-set enrichment requires at least three targets")
        background = arguments.get("background")
        if not isinstance(background, Mapping):
            raise ValueError("background is required")
        background_type = str(background.get("type"))
        explicit = _string_list(background.get("target_ids", ()))
        if background_type == "explicit":
            if not explicit:
                raise ValueError("explicit background requires target_ids")
            if not set(selected).issubset(explicit):
                raise ValueError("explicit background must contain every selected target")
        elif background_type == "candidate_universe":
            explicit = selected
        elif background_type != "all_protein_coding":
            raise ValueError("unsupported background type")
        minimum = _bounded_int(
            arguments.get("min_set_size", 3), 1, 1000, "min_set_size"
        )
        target_path = self.provider._dataset_glob("target")
        lookup_dataset = "go" if kind == "go" else "reactome"
        lookup_path = self.provider._dataset_glob(lookup_dataset)
        selected_placeholders = ",".join("?" for _ in selected)
        selected_predicate = f"id IN ({selected_placeholders})"
        if explicit:
            background_predicate = (
                "id IN (" + ",".join("?" for _ in explicit) + ")"
            )
            background_parameters = list(explicit)
        else:
            background_predicate = "biotype = 'protein_coding'"
            background_parameters = []
            selected_predicate += " AND biotype = 'protein_coding'"
        if kind == "go":
            term_expression = "annotation.id"
            unnest_expression = "UNNEST(t.go) AS u(annotation)"
            label_column = "name"
        else:
            term_expression = "annotation.pathwayId"
            unnest_expression = "UNNEST(t.pathways) AS u(annotation)"
            label_column = "label"
        connection = self.provider._connect()
        try:
            universe_size = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet('{target_path}') "
                    f"WHERE {background_predicate}",
                    background_parameters,
                ).fetchone()[0]
            )
            selected_size = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet('{target_path}') "
                    f"WHERE {selected_predicate}",
                    list(selected),
                ).fetchone()[0]
            )
            if selected_size != len(selected):
                raise ValueError(
                    "every selected target must resolve inside the chosen background"
                )
            sql = f"""
                WITH selected AS (
                    SELECT id, go, pathways
                    FROM read_parquet('{target_path}')
                    WHERE {selected_predicate}
                ), universe AS (
                    SELECT id, go, pathways
                    FROM read_parquet('{target_path}')
                    WHERE {background_predicate}
                ), selected_terms AS (
                    SELECT {term_expression} AS termId,
                           count(DISTINCT t.id) AS overlapCount,
                           list(DISTINCT t.id) AS overlapGenes
                    FROM selected t, {unnest_expression}
                    GROUP BY termId
                ), universe_terms AS (
                    SELECT {term_expression} AS termId,
                           count(DISTINCT t.id) AS universeTermCount
                    FROM universe t, {unnest_expression}
                    GROUP BY termId
                )
                SELECT u.termId, lookup.{label_column} AS termLabel,
                       COALESCE(s.overlapCount, 0) AS overlapCount,
                       COALESCE(s.overlapGenes, []) AS overlapGenes,
                       u.universeTermCount
                FROM universe_terms u
                LEFT JOIN selected_terms s ON s.termId = u.termId
                LEFT JOIN read_parquet('{lookup_path}') lookup
                  ON u.termId = lookup.id
                WHERE u.universeTermCount >= ?
            """
            parameters = [*selected, *background_parameters, minimum]
            result = connection.execute(sql, parameters)
            columns = [item[0] for item in result.description]
            rows = [_row(columns, row) for row in result.fetchall()]
        finally:
            connection.close()
        enriched: list[dict[str, Any]] = []
        for row in rows:
            overlap = int(row["overlapCount"])
            universe_term = int(row["universeTermCount"])
            p_value = _hypergeometric_survival(
                overlap,
                population_size=universe_size,
                success_states=universe_term,
                draws=selected_size,
            )
            enriched.append({**row, "pValue": p_value})
        q_values = _benjamini_hochberg([row["pValue"] for row in enriched])
        for row, q_value in zip(enriched, q_values, strict=True):
            row["qValue"] = q_value
        enriched.sort(key=lambda row: (row["qValue"], row["pValue"], row["termId"]))
        truncated = len(enriched) > 500
        module = "gene_set.go" if kind == "go" else "gene_set.pathway"
        records = tuple(
            {
                "module": module,
                "dataset": lookup_dataset,
                "relation": "overrepresentation",
                "subject": {"type": "gene_set", "id": sha256_json(selected)},
                "object": {
                    "type": "go_term" if kind == "go" else "pathway",
                    "id": row["termId"],
                    "label": row.get("termLabel"),
                },
                "measurements": {
                    "overlap_count": row["overlapCount"],
                    "overlap_genes": row["overlapGenes"],
                    "term_size_in_universe": row["universeTermCount"],
                    "selected_size": selected_size,
                    "universe_size": universe_size,
                    "p_value": row["pValue"],
                    "q_value_bh": row["qValue"],
                    "tested_terms": len(enriched),
                },
                "context": {
                    "background_type": background_type,
                    "multiple_testing": "BH",
                    "min_set_size": minimum,
                },
                "provenance": {
                    "source": SOURCE,
                    "data_release": self.provider.capabilities.release,
                },
                "upstream": {"raw": row},
                "derived": {"method": "hypergeometric_overrepresentation"},
            }
            for row in enriched[:500]
        )
        return _module_result(module, records, truncated=truncated)

    def gene_set_tissue_coverage(
        self, arguments: Mapping[str, Any]
    ) -> ModuleResult:
        targets = _string_list(arguments.get("target_ids", ()))
        placeholders = ",".join("?" for _ in targets)
        rows, truncated = self._read_rows(
            "expression",
            "id, tissues",
            f"id IN ({placeholders})",
            list(targets),
            min(len(targets), 500),
        )
        coverage: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            for tissue in row.get("tissues") or ():
                if not isinstance(tissue, Mapping):
                    continue
                identifier = str(tissue.get("efo_code") or "unknown")
                label = str(tissue.get("label") or identifier)
                rna = tissue.get("rna") if isinstance(tissue.get("rna"), Mapping) else {}
                protein = (
                    tissue.get("protein")
                    if isinstance(tissue.get("protein"), Mapping)
                    else {}
                )
                signal_values = (rna.get("value"), rna.get("level"), protein.get("level"))
                if any(
                    isinstance(value, bool)
                    or (isinstance(value, (int, float)) and not math.isfinite(value))
                    for value in signal_values
                ):
                    raise ValueError("expression signal contains an invalid numeric measurement")
                has_signal = any(
                    isinstance(value, (int, float)) and value > 0 for value in signal_values
                )
                if has_signal:
                    coverage.setdefault((identifier, label), set()).add(str(row["id"]))
        records = tuple(
            {
                "module": "gene_set.tissue_coverage",
                "dataset": "expression",
                "relation": "gene_set_tissue_coverage",
                "subject": {"type": "gene_set", "id": sha256_json(targets)},
                "object": {"type": "biosample", "id": key[0], "label": key[1]},
                "measurements": {
                    "covered_genes": sorted(genes),
                    "covered_count": len(genes),
                    "requested_count": len(targets),
                    "coverage_fraction": len(genes) / len(targets),
                },
                "context": {},
                "provenance": {
                    "source": SOURCE,
                    "data_release": self.provider.capabilities.release,
                },
                "upstream": {"raw": {}},
                "derived": {"rule": "positive RNA or protein signal"},
            }
            for key, genes in sorted(
                coverage.items(), key=lambda item: (-len(item[1]), item[0])
            )
        )
        return _module_result(
            "gene_set.tissue_coverage", records, truncated=truncated
        )

    def gene_set_evidence_coverage(
        self, arguments: Mapping[str, Any]
    ) -> ModuleResult:
        targets = _string_list(arguments.get("target_ids", ()))
        target_rows, target_truncated = self._read_rows(
            "target",
            "id, functionDescriptions, go, pathways, tractability, safetyLiabilities",
            f"id IN ({','.join('?' for _ in targets)})",
            list(targets),
            min(len(targets), 500),
        )
        expression_rows, expression_truncated = self._read_rows(
            "expression",
            "id",
            f"id IN ({','.join('?' for _ in targets)})",
            list(targets),
            min(len(targets), 500),
        )
        expression_ids = {str(row["id"]) for row in expression_rows}
        records = tuple(
            {
                "module": "gene_set.evidence_coverage",
                "dataset": "target+expression",
                "relation": "target_annotation_coverage",
                "subject": {"type": "target", "id": row["id"]},
                "object": None,
                "measurements": {
                    "function": bool(row.get("functionDescriptions")),
                    "go": bool(row.get("go")),
                    "pathway": bool(row.get("pathways")),
                    "expression": str(row["id"]) in expression_ids,
                    "tractability": bool(row.get("tractability")),
                    "safety": bool(row.get("safetyLiabilities")),
                },
                "context": {},
                "provenance": {
                    "source": SOURCE,
                    "data_release": self.provider.capabilities.release,
                },
                "upstream": {"raw": {}},
                "derived": {},
            }
            for row in target_rows
        )
        return _module_result(
            "gene_set.evidence_coverage",
            records,
            truncated=target_truncated or expression_truncated,
        )

    def _disease_module(
        self, arguments: Mapping[str, Any], module: str
    ) -> ModuleResult:
        disease_ids = _string_list(arguments.get("disease_ids", ()))
        if not disease_ids:
            raise ValueError("disease_ids are required")
        placeholders = ",".join("?" for _ in disease_ids)
        rows, truncated = self._read_rows(
            "disease",
            "*",
            f"id IN ({placeholders})",
            list(disease_ids),
            _page_size(arguments),
        )
        fields = (
            ("id", "code", "name", "description", "dbXRefs")
            if module == "disease.core"
            else (
                "id",
                "parents",
                "children",
                "ancestors",
                "descendants",
                "therapeuticAreas",
                "ontology",
                "synonyms",
            )
        )
        records = tuple(
            _semantic_record(
                module=module,
                dataset="disease",
                relation="disease_profile" if module == "disease.core" else "disease_ontology",
                subject={"type": "disease", "id": row["id"], "label": row.get("name")},
                raw={field: row.get(field) for field in fields},
                release=self.provider.capabilities.release,
            )
            for row in rows
        )
        return _module_result(module, records, truncated=truncated)

    def _target_handler(self, module: str):
        def handler(arguments: Mapping[str, Any]) -> ModuleResult:
            return self._target_module(arguments, module)

        return handler

    def _target_module(
        self, arguments: Mapping[str, Any], module: str
    ) -> ModuleResult:
        rows, truncated = self._target_rows(arguments)
        fields = {
            "target.core": (
                "id",
                "approvedSymbol",
                "approvedName",
                "biotype",
                "genomicLocation",
                "canonicalTranscript",
                "proteinIds",
                "dbXrefs",
            ),
            "target.annotation": (
                "id",
                "synonyms",
                "symbolSynonyms",
                "nameSynonyms",
                "targetClass",
                "alternativeGenes",
            ),
            "target.function": ("id", "functionDescriptions"),
            "target.subcellular_location": ("id", "subcellularLocations"),
            "target.constraint": ("id", "constraint"),
            "target.homologues": ("id", "homologues"),
            "target.tractability": ("id", "tractability"),
            "target.safety": ("id", "safetyLiabilities"),
            "target.chemical_probes": ("id", "chemicalProbes"),
            "target.tep": ("id", "tep"),
            "target.hallmarks": ("id", "hallmarks"),
            "target.go": ("id", "go"),
            "target.pathway": ("id", "pathways"),
        }[module]
        records: list[dict[str, Any]] = []
        go_lookup: dict[str, dict[str, Any]] = {}
        reactome_lookup: dict[str, dict[str, Any]] = {}
        if module == "target.go":
            identifiers = tuple(
                sorted(
                    {
                        str(annotation["id"])
                        for row in rows
                        for annotation in row.get("go") or ()
                        if isinstance(annotation, Mapping) and annotation.get("id")
                    }
                )
            )
            if len(identifiers) > 500:
                identifiers = identifiers[:500]
                truncated = True
            go_lookup = {
                str(item["id"]): item for item in self._go_terms(identifiers)
            }
        elif module == "target.pathway":
            identifiers = tuple(
                sorted(
                    {
                        str(pathway["pathwayId"])
                        for row in rows
                        for pathway in row.get("pathways") or ()
                        if isinstance(pathway, Mapping) and pathway.get("pathwayId")
                    }
                )
            )
            if len(identifiers) > 500:
                identifiers = identifiers[:500]
                truncated = True
            reactome_lookup = {
                str(item["id"]): item for item in self._reactome_terms(identifiers)
            }
        for row in rows:
            if module != "target.core" and not any(
                row.get(field) not in (None, [], {}, ())
                for field in fields
                if field != "id"
            ):
                continue
            raw = {field: row.get(field) for field in fields}
            if module == "target.go":
                raw["terms"] = [
                    go_lookup[str(annotation["id"])]
                    for annotation in row.get("go") or ()
                    if isinstance(annotation, Mapping)
                    and str(annotation.get("id")) in go_lookup
                ]
            elif module == "target.pathway":
                raw["pathwayHierarchy"] = [
                    reactome_lookup[str(pathway["pathwayId"])]
                    for pathway in row.get("pathways") or ()
                    if isinstance(pathway, Mapping)
                    and str(pathway.get("pathwayId")) in reactome_lookup
                ]
            records.append(
                _semantic_record(
                    module=module,
                    dataset=(
                        "go"
                        if module == "target.go"
                        else "reactome"
                        if module == "target.pathway"
                        else "target"
                    ),
                    relation=module.replace("target.", "target_"),
                    subject={
                        "type": "target",
                        "id": row["id"],
                        "label": row.get("approvedSymbol"),
                    },
                    raw=raw,
                    release=self.provider.capabilities.release,
                )
            )
        return _module_result(module, tuple(records), truncated=truncated)

    def _target_rows(
        self, arguments: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], bool]:
        target_ids = _string_list(arguments.get("target_ids", ()))
        if not target_ids:
            raise ValueError("target_ids are required")
        key = sha256_json({"target_ids": target_ids, "page_size": _page_size(arguments)})
        if key in self._target_cache:
            self._target_cache.move_to_end(key)
            return self._target_cache[key]
        placeholders = ",".join("?" for _ in target_ids)
        rows, truncated = self._read_rows(
            "target",
            "*",
            f"id IN ({placeholders}) OR approvedSymbol IN ({placeholders})",
            [*target_ids, *target_ids],
            _page_size(arguments),
        )
        self._cache_put(self._target_cache, key, (rows, truncated))
        return rows, truncated

    def _variant_rows(
        self, arguments: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], bool]:
        variant_ids = _string_list(arguments.get("variant_ids", ()))
        if not variant_ids:
            raise ValueError("variant_ids are required")
        rows: list[dict[str, Any]] = []
        truncated = False
        page_size = _page_size(arguments)
        path = self.provider._dataset_glob("variant")
        connection = self.provider._connect()
        try:
            sql = f"""
                SELECT variantId, chromosome, position, referenceAllele,
                       alternateAllele, variantEffect, mostSevereConsequenceId,
                       transcriptConsequences, rsIds, hgvsId, alleleFrequencies,
                       dbXrefs, variantDescription
                FROM read_parquet('{path}')
                WHERE variantId = ? OR list_contains(rsIds, ?)
                LIMIT {page_size + 1}
            """
            seen: set[str] = set()
            for variant_id in variant_ids:
                canonical_query = _normalized_variant_id(variant_id)
                result = connection.execute(sql, [canonical_query, canonical_query])
                columns = [item[0] for item in result.description]
                for row in result.fetchall():
                    value = _row(columns, row)
                    canonical = str(value["variantId"])
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                    rows.append(value)
                    if len(rows) > page_size:
                        truncated = True
                        rows = rows[:page_size]
                        return rows, truncated
        finally:
            connection.close()
        return rows, truncated

    def _locus_records(
        self,
        arguments: Mapping[str, Any],
        requested_kinds: set[EvidenceKind],
    ) -> tuple[dict[EvidenceKind, tuple[list[dict[str, Any]], bool]], bool]:
        variant_ids = _string_list(arguments.get("variant_ids", ()))
        if not variant_ids:
            raise ValueError("variant_ids are required")
        cache_key = sha256_json(
            {
                "variant_ids": variant_ids,
                "study_ids": _string_list(arguments.get("study_ids", ())),
                "qtl_types": requested_qtl_study_types(
                    arguments.get("qtl_types", ())
                ),
                "page_size": _page_size(arguments),
                "kinds": sorted(kind.value for kind in requested_kinds),
            }
        )
        if cache_key in self._locus_cache:
            self._locus_cache.move_to_end(cache_key)
            value = self._locus_cache[cache_key]
            return value, any(item[1] for item in value.values())
        query = LocusQuery(
            locus_id="semantic-mcp-query",
            trait="not-used-for-retrieval",
            variants=tuple(
                VariantObservation(variant_id, p_value=1e-9)
                for variant_id in variant_ids
            ),
            study_ids=_string_list(arguments.get("study_ids", ())),
            qtl_types=requested_qtl_study_types(arguments.get("qtl_types", ())),
        )
        connection = self.provider._connect()
        try:
            resolved = self.provider._resolve_variants(connection, query)
            canonical = list(resolved)
            page_size = _page_size(arguments)
            values: dict[EvidenceKind, tuple[list[dict[str, Any]], bool]] = {}
            if canonical:
                needs_qtl_lineage = bool(
                    requested_kinds.intersection(
                        {
                            EvidenceKind.QTL,
                            EvidenceKind.COLOCALIZATION,
                        }
                    )
                )
                qtl_records = (
                    self.provider._fetch_associations(
                        connection,
                        canonical,
                        gwas=False,
                        **(
                            {"qtl_types": query.qtl_types}
                            if query.qtl_types
                            else {}
                        ),
                    )
                    if needs_qtl_lineage
                    else []
                )
                fetchers = {
                    EvidenceKind.GWAS: lambda: self.provider._fetch_associations(
                        connection,
                        canonical,
                        gwas=True,
                        study_ids=query.study_ids,
                    ),
                    EvidenceKind.QTL: lambda: qtl_records,
                    EvidenceKind.L2G: lambda: self.provider._fetch_l2g(
                        connection, canonical
                    ),
                    EvidenceKind.COLOCALIZATION: lambda: self.provider._fetch_colocalization(
                        connection, qtl_records
                    ),
                    EvidenceKind.REGULATORY: lambda: self.provider._fetch_regulatory(
                        connection, canonical
                    ),
                }
                for kind, fetcher in fetchers.items():
                    if kind not in requested_kinds:
                        continue
                    if kind not in self.provider.capabilities.evidence_kinds:
                        continue
                    records = fetcher()
                    filtered = [
                        record.to_dict()
                        for record in records
                        if not query.study_ids
                        or not record.study_id
                        or record.study_id in query.study_ids
                    ]
                    is_truncated = (
                        len(filtered) > page_size
                        or getattr(records, "truncated", len(records) in {200, 1000})
                        or (
                            kind is EvidenceKind.COLOCALIZATION
                            and getattr(qtl_records, "truncated", False)
                        )
                    )
                    values[kind] = (filtered[:page_size], is_truncated)
            self._cache_put(self._locus_cache, cache_key, values)
            return values, any(item[1] for item in values.values())
        finally:
            connection.close()

    def _combined_locus_rows(
        self, arguments: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], bool]:
        values, _ = self._locus_records(
            arguments, {EvidenceKind.GWAS, EvidenceKind.QTL}
        )
        rows: list[dict[str, Any]] = []
        truncated = False
        seen: set[str] = set()
        for kind in (EvidenceKind.GWAS, EvidenceKind.QTL):
            kind_rows, kind_truncated = values.get(kind, ([], False))
            truncated = truncated or kind_truncated
            for record in kind_rows:
                payload = record.get("payload")
                if not isinstance(payload, Mapping):
                    continue
                evidence_id = str(record.get("evidence_id"))
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                rows.append(dict(payload))
        return rows, truncated

    def _evidence_module(
        self,
        arguments: Mapping[str, Any],
        kind: EvidenceKind,
        module: str,
    ) -> ModuleResult:
        # Each semantic module is an independent failure boundary. Fetching every
        # sibling kind here made a QTL/schema failure also break unrelated E2G.
        values, _ = self._locus_records(arguments, {kind})
        rows, truncated = values.get(kind, ([], False))
        candidates = set(_string_list(arguments.get("candidate_gene_ids", ())))
        selected_qtl_types = set(
            requested_qtl_study_types(arguments.get("qtl_types", ()))
        )
        records: list[dict[str, Any]] = []
        for row in rows:
            gene_id = _optional_text(row.get("gene_id"))
            payload = (
                row.get("payload")
                if isinstance(row.get("payload"), Mapping)
                else {}
            )
            native_qtl_type = qtl_study_type_from_payload(payload)
            if kind is EvidenceKind.QTL and (
                native_qtl_type is None
                or (
                    selected_qtl_types
                    and native_qtl_type not in selected_qtl_types
                )
            ):
                continue
            symbol = _optional_text(payload.get("approvedSymbol"))
            if candidates and not ({gene_id, symbol} - {None}).intersection(candidates):
                continue
            measurements = {
                key: payload.get(key)
                for key in (
                    "posteriorProbability",
                    "h0",
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "clpp",
                    "score",
                    "resourceScore",
                    "l2g_score",
                    "distanceToTss",
                )
                if key in payload
            }
            qtl_member_context: dict[str, Any] = {}
            if kind is EvidenceKind.QTL:
                member = payload.get("matchedLocusMember")
                member_id = (
                    _optional_text(member.get("variantId"))
                    if isinstance(member, Mapping)
                    else None
                )
                chromosome, position = _canonical_variant_coordinates(member_id)
                qtl_member_context = {
                    "qtl_member_variant_id": member_id,
                    "qtl_member_chromosome": chromosome,
                    "qtl_member_position": position,
                }
            records.append(
                _semantic_record(
                    module=module,
                    dataset=str(row.get("dataset")),
                    relation={
                        EvidenceKind.QTL: "variant_qtl_gene",
                        EvidenceKind.COLOCALIZATION: "gwas_qtl_colocalisation",
                        EvidenceKind.REGULATORY: (
                            "variant_overlaps_regulatory_interval_linked_to_target"
                        ),
                        EvidenceKind.L2G: "open_targets_l2g",
                    }[kind],
                    subject={"type": "variant", "id": row.get("variant_id")},
                    object_value={"type": "target", "id": gene_id, "label": symbol},
                    raw=payload,
                    release=self.provider.capabilities.release,
                    measurements=measurements,
                    context={
                        "study_id": row.get("study_id"),
                        "study_locus_id": row.get("credible_set_id"),
                        "pmid": row.get("pmid"),
                        "publication_date": row.get("evidence_date"),
                        "source_evidence_id": row.get("evidence_id"),
                        "parent_evidence_ids": row.get("parent_evidence_ids"),
                        "biosample": row.get("tissue"),
                        "parent_variant_ids": row.get("parent_variant_ids"),
                        "study_type": native_qtl_type,
                        **qtl_member_context,
                        **(
                            {
                                "overlap_variant_id": payload.get("overlapVariantId")
                                or row.get("variant_id"),
                                "overlap_variant_role": payload.get(
                                    "overlapVariantRole"
                                ),
                                "overlap_chromosome": payload.get(
                                    "overlapChromosome"
                                ),
                                "overlap_position": payload.get("overlapPosition"),
                                "exact_variant_overlap": payload.get(
                                    "exactVariantOverlap"
                                ),
                                "coverage_scope": payload.get("coverageScope"),
                                "regulatory_element_key": payload.get(
                                    "regulatoryElementKey"
                                ),
                                "regulatory_element_key_source": payload.get(
                                    "regulatoryElementKeySource"
                                ),
                                "regulatory_chromosome": payload.get("chromosome"),
                                "regulatory_start": payload.get("start"),
                                "regulatory_end": payload.get("end"),
                                "interval_type": payload.get("intervalType"),
                                "distance_to_tss": payload.get("distanceToTss"),
                                "datasource_id": payload.get("datasourceId"),
                                "biosample_id": payload.get("biosampleId"),
                                "enhancer_study_id": payload.get("studyId"),
                            }
                            if kind is EvidenceKind.REGULATORY
                            else {}
                        ),
                    },
                    derived=(
                        {
                            "exact_variant_interval_overlap": (
                                payload.get("exactVariantOverlap") is True
                            ),
                            "coverage_scope": payload.get("coverageScope"),
                            "regulatory_element_key_is_derived": (
                                payload.get("regulatoryElementKeySource")
                                == "derived_coordinate_key"
                            ),
                        }
                        if kind is EvidenceKind.REGULATORY
                        else None
                    ),
                )
            )
        return _module_result(module, tuple(records), truncated=truncated)

    def _go_terms(self, ids: tuple[str, ...]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows, _ = self._read_rows(
            "go", "id, name", f"id IN ({placeholders})", list(ids), len(ids)
        )
        return rows

    def _reactome_terms(self, ids: tuple[str, ...]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows, _ = self._read_rows(
            "reactome", "*", f"id IN ({placeholders})", list(ids), len(ids)
        )
        return rows

    def _resolve_type(
        self, text: str, entity_type: str, max_hits: int
    ) -> list[dict[str, Any]]:
        queries: dict[str, tuple[str, str, str, list[Any]]] = {
            "target": (
                "target",
                "id, approvedSymbol, approvedName",
                "lower(id) = lower(?) OR lower(approvedSymbol) = lower(?)",
                [text, text],
            ),
            "variant": (
                "variant",
                "variantId AS id, variantId AS label, rsIds",
                "lower(variantId) = lower(?) OR list_contains(rsIds, ?)",
                [text, text],
            ),
            "disease": (
                "disease",
                "id, name AS label, code",
                "lower(id) = lower(?) OR lower(code) = lower(?) OR lower(name) = lower(?)",
                [text, text, text],
            ),
            "study": (
                "study",
                "studyId AS id, traitFromSource AS label, studyType",
                "lower(studyId) = lower(?)",
                [text],
            ),
            "biosample": (
                "biosample",
                "biosampleId AS id, biosampleName AS label, synonyms",
                "lower(biosampleId) = lower(?) OR lower(biosampleName) = lower(?)",
                [text, text],
            ),
            "drug": (
                "known_drug",
                "DISTINCT drugId AS id, prefName AS label, drugType",
                "lower(drugId) = lower(?) OR lower(prefName) = lower(?)",
                [text, text],
            ),
        }
        if entity_type == "pathway":
            hits: list[dict[str, Any]] = []
            for dataset, label_column in (("go", "name"), ("reactome", "label")):
                if dataset not in self._present_datasets:
                    continue
                rows, _ = self._read_rows(
                    dataset,
                    f"id, {label_column} AS label",
                    f"lower(id) = lower(?) OR lower({label_column}) = lower(?)",
                    [text, text],
                    max_hits,
                )
                hits.extend(
                    {
                        "id": row["id"],
                        "type": "pathway",
                        "label": row.get("label"),
                        "match": "exact",
                        "matched_field": "id_or_label",
                        "source_dataset": dataset,
                    }
                    for row in rows
                )
            return hits[:max_hits]
        if entity_type not in queries:
            return []
        dataset, columns, predicate, parameters = queries[entity_type]
        if dataset not in self._present_datasets:
            return []
        rows, _ = self._read_rows(
            dataset, columns, predicate, parameters, max_hits
        )
        return [
            {
                "id": row.get("id") or row.get("approvedSymbol"),
                "type": entity_type,
                "label": row.get("label") or row.get("approvedSymbol") or row.get("approvedName"),
                "match": "exact",
                "matched_field": "canonical_id_or_exact_label",
                "source_dataset": dataset,
            }
            for row in rows
        ]

    def _read_rows(
        self,
        dataset: str,
        columns: str,
        predicate: str,
        parameters: list[Any],
        page_size: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        page_size = _bounded_int(page_size, 1, 500, "page_size")
        path = self.provider._dataset_glob(dataset)
        connection = self.provider._connect()
        try:
            result = connection.execute(
                f"SELECT {columns} FROM read_parquet('{path}') "
                f"WHERE {predicate} LIMIT {page_size + 1}",
                parameters,
            )
            names = [item[0] for item in result.description]
            rows = [_row(names, row) for row in result.fetchall()]
        finally:
            connection.close()
        return rows[:page_size], len(rows) > page_size

    @staticmethod
    def _cache_put(cache: OrderedDict, key: str, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > 4:
            cache.popitem(last=False)


def create_frozen_semantic_service(
    data_root: str | Path,
    *,
    release: str,
    snapshot_sha256: str,
    max_threads: int = parameter("execution.duckdb_threads"),
    memory_limit: str = parameter("execution.duckdb_memory_limit"),
    policy_sha256: str = "unknown",
    blocked_modules: frozenset[str] = frozenset(),
) -> SemanticToolService:
    provider = DuckDBFrozenOpenTargetsProvider(
        data_root,
        release=release,
        snapshot_hash=snapshot_sha256,
        max_threads=max_threads,
        memory_limit=memory_limit,
    )
    return FrozenSemanticModules(provider).service(
        policy_sha256=policy_sha256,
        blocked_modules=blocked_modules,
    )


def _module_result(
    module: str,
    records: tuple[Mapping[str, Any], ...],
    *,
    truncated: bool = False,
    warnings: tuple[str, ...] = (),
) -> ModuleResult:
    return ModuleResult(
        module,
        ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
        records=records,
        truncated=truncated,
        warnings=warnings,
    )


def _semantic_record(
    *,
    module: str,
    dataset: str,
    relation: str,
    subject: Mapping[str, Any],
    raw: Mapping[str, Any],
    release: str,
    object_value: Mapping[str, Any] | None = None,
    measurements: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    derived: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "module": module,
        "dataset": dataset,
        "relation": relation,
        "subject": dict(subject),
        "object": dict(object_value) if object_value else None,
        "measurements": dict(measurements or {}),
        "context": dict(context or {}),
        "provenance": {"source": SOURCE, "data_release": release},
        "upstream": {"raw": dict(raw)},
        "derived": dict(derived or {}),
    }
    return _json_safe(value)


def _row(columns: list[str], row: tuple[Any, ...]) -> dict[str, Any]:
    return _json_safe(dict(zip(columns, row, strict=True)))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("Open Targets record contains a non-finite number")
        return value
    if isinstance(value, Decimal):
        return _json_safe(float(value))
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _page_size(arguments: Mapping[str, Any]) -> int:
    return _bounded_int(
        arguments.get("page_size", parameter("retrieval.semantic_page_size")), 1, 500, "page_size"
    )


def _bounded_int(value: Any, minimum: int, maximum: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("identifier collection must be an array")
    result = tuple(_required_text(item, "identifier") for item in value)
    if len(result) != len(set(result)):
        raise ValueError("identifier collection must not contain duplicates")
    return result


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_variant_coordinates(
    variant_id: str | None,
) -> tuple[str | None, int | None]:
    if not variant_id:
        return None, None
    parts = variant_id.split("_")
    if len(parts) != 4:
        return None, None
    try:
        position = int(parts[1])
    except ValueError:
        return None, None
    chromosome = parts[0][3:] if parts[0].casefold().startswith("chr") else parts[0]
    if not chromosome or position < 1:
        return None, None
    return chromosome, position


def _deduplicate_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        key = (str(hit.get("type")), str(hit.get("id")))
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)
    return result


def _hypergeometric_survival(
    observed: int,
    *,
    population_size: int,
    success_states: int,
    draws: int,
) -> float:
    if not 0 <= observed <= draws <= population_size:
        raise ValueError("invalid hypergeometric parameters")
    if not 0 <= success_states <= population_size:
        raise ValueError("invalid hypergeometric success-state count")
    maximum = min(success_states, draws)
    minimum = max(observed, draws - (population_size - success_states))
    if minimum > maximum:
        return 0.0

    def log_choose(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float("-inf")
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    denominator = log_choose(population_size, draws)
    probabilities = [
        math.exp(
            log_choose(success_states, value)
            + log_choose(population_size - success_states, draws - value)
            - denominator
        )
        for value in range(minimum, maximum + 1)
    ]
    return min(1.0, max(0.0, math.fsum(probabilities)))


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[original_index] * count / rank)
        adjusted[original_index] = min(1.0, max(0.0, running))
    return adjusted
