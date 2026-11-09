from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..audit import sha256_json
from ..configuration import parameter
from ..models import EvidenceKind, EvidenceRecord, LocusQuery, VariantObservation
from ..providers.live import MCP_SOURCE, OfficialOpenTargetsMCPProvider
from ..publications import normalize_pmid, publication_from_metadata
from ..qtl import qtl_study_type_from_payload, requested_qtl_study_types
from ..region_gate import local_gene_window_label, nearby_gene_witnesses_from_ot
from .contracts import SemanticTool
from .ensembl import (
    ENSEMBL_REST_ENDPOINT,
    ENSEMBL_REST_SOURCE,
    EnsemblRegionGeneLookup,
    EnsemblRegionLookupError,
    RegionGeneCandidate,
    RegionGeneLookup,
)
from .europe_pmc import EuropePMCError, EuropePMCPublicationClient
from .live_schema import LiveSchemaSnapshot, OfficialSchemaNegotiator
from .regions import GRCh38Region, grch38_region_from_variant_ids
from .service import ModuleAdapter, ModuleResult, ModuleStatus, SemanticToolService

_TARGET_ID = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)
_DRUG_ID = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)
_VARIANT_ID = re.compile(
    r"^(?:chr)?(?:[0-9]+|X|Y|MT)(?:_|:)\d+(?:_|:)[ACGTN]+(?:_|:)[ACGTN]+$",
    re.IGNORECASE,
)
_DIRECT_VARIANT_ID = re.compile(r"^(?:[0-9]+|X|Y|MT)_\d+_[ACGTN]+_[ACGTN]+$", re.IGNORECASE)

# Transcript consequences are nested inside each exact Open Targets variant
# response rather than exposed as a separately pageable connection. Treating
# the semantic tool page_size as one global cap made a normal multi-variant
# locus appear incomplete as soon as the aggregate crossed 500 rows. Keep two
# explicit bounds instead: no selected variant may contribute more than the
# requested page size, and the complete locus response is globally bounded.
# We scan all returned consequences so truncated means an observed budget
# overrun rather than merely reaching the old global cap.
MAX_VARIANT_CONSEQUENCE_RECORDS = parameter("retrieval.max_variant_consequence_records")
MAX_PUBLICATION_ENRICHMENT_PMIDS = parameter("retrieval.publication_enrichment_pmids")
# Bibliographic citations already attached to evidence must not compete with
# the optional recent-literature/abstract sample for the same 100 slots.
MAX_CITED_PUBLICATION_ENRICHMENT_PMIDS = parameter("retrieval.cited_publication_enrichment_pmids")


@dataclass(frozen=True, slots=True)
class _LiveFetchBundle:
    records: tuple[EvidenceRecord, ...]
    warnings: tuple[str, ...] = ()
    credible_sets_truncated: bool = False
    truncated_kinds: frozenset[EvidenceKind] = frozenset()


def _provider_source(provider: Any) -> str:
    return str(getattr(provider, "source", MCP_SOURCE))


def _provider_backend_name(provider: Any) -> str:
    return str(getattr(provider, "backend_name", "official_mcp"))


def _provider_transport_description(provider: Any) -> str:
    return str(getattr(provider, "transport_description", "official hosted MCP"))


class LiveSemanticModules:
    """Semantic adapters backed by the official Open Targets MCP/API.

    Only modules normalized by the current wrapper are registered. Other
    catalogued Open Targets domains remain explicitly unavailable until their
    schema-discovered adapter and contract tests exist.
    """

    def __init__(
        self,
        provider: OfficialOpenTargetsMCPProvider,
        *,
        schema: LiveSchemaSnapshot | None = None,
        region_gene_lookup: RegionGeneLookup | None = None,
        publication_enricher: EuropePMCPublicationClient | None = None,
    ):
        self.provider = provider
        self.schema = schema
        self.region_gene_lookup = region_gene_lookup
        self.publication_enricher = publication_enricher
        self._cache: OrderedDict[str, _LiveFetchBundle] = OrderedDict()
        self._phenotype_resolution_cache: OrderedDict[str, ModuleResult] = OrderedDict()
        self._extended_cache: OrderedDict[str, tuple[dict[str, Any], ...]] = OrderedDict()
        self._disease_cache: OrderedDict[str, tuple[dict[str, Any], ...]] = OrderedDict()
        self._association_cache: OrderedDict[str, tuple[tuple[dict[str, Any], ...], bool]] = (
            OrderedDict()
        )
        self._clinical_cache: OrderedDict[str, tuple[tuple[dict[str, Any], ...], bool]] = (
            OrderedDict()
        )
        self._drug_cache: OrderedDict[str, tuple[tuple[dict[str, Any], ...], bool]] = OrderedDict()
        self._variant_cache: OrderedDict[str, tuple[dict[str, Any], ...]] = OrderedDict()
        self._variant_consequence_cache: OrderedDict[str, tuple[dict[str, Any], ...]] = (
            OrderedDict()
        )

    def adapters(self) -> tuple[ModuleAdapter, ...]:
        adapters = (
            ModuleAdapter("entity.resolve", SemanticTool.RESOLVE_ENTITIES, self.resolve_entities),
            ModuleAdapter("locus.variant", SemanticTool.GET_LOCUS_CONTEXT, self.locus_variant),
            ModuleAdapter(
                "locus.variant_consequence",
                SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                self.locus_variant_consequence,
            ),
            ModuleAdapter("locus.study", SemanticTool.GET_LOCUS_CONTEXT, self.locus_study),
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
            ModuleAdapter("locus.qtl", SemanticTool.GET_LOCUS_GENE_EVIDENCE, self.locus_qtl),
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
            ModuleAdapter("locus.l2g", SemanticTool.GET_LOCUS_GENE_EVIDENCE, self.locus_l2g),
            ModuleAdapter("target.core", SemanticTool.GET_TARGET_PROFILES, self.target_core),
            ModuleAdapter(
                "target.function",
                SemanticTool.GET_TARGET_PROFILES,
                self.target_function,
            ),
            ModuleAdapter(
                "target.pathway",
                SemanticTool.GET_TARGET_PROFILES,
                self.target_pathway,
            ),
            ModuleAdapter(
                "target.expression",
                SemanticTool.GET_TARGET_PROFILES,
                self.target_expression,
            ),
            ModuleAdapter(
                "evidence.europepmc",
                SemanticTool.GET_TARGET_PROFILES,
                self.target_literature,
            ),
        )
        if self.schema is None:
            return adapters
        extended = list(adapters)
        if self.schema.module_support.get("target.safety", False):
            extended.append(
                ModuleAdapter(
                    "target.safety",
                    SemanticTool.GET_TARGET_PROFILES,
                    self.target_safety,
                )
            )
        if self.schema.module_support.get("target.tractability", False):
            extended.append(
                ModuleAdapter(
                    "target.tractability",
                    SemanticTool.GET_TARGET_PROFILES,
                    self.target_tractability,
                )
            )
        if self.schema.module_support.get("target.go", False):
            extended.append(
                ModuleAdapter(
                    "target.go",
                    SemanticTool.GET_TARGET_PROFILES,
                    self.target_go,
                )
            )
        if self.schema.module_support.get("target.interaction", False):
            extended.append(
                ModuleAdapter(
                    "target.interaction",
                    SemanticTool.GET_TARGET_PROFILES,
                    self.target_interaction,
                )
            )
        if self.schema.module_support.get("disease.core", False):
            extended.extend(
                (
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
                )
            )
        if self.schema.module_support.get("association.datasource_direct", False):
            for module in (
                "association.overall_direct",
                "association.overall_indirect",
                "association.datasource_direct",
                "association.datasource_indirect",
                "association.datatype_direct",
                "association.datatype_indirect",
            ):
                extended.append(
                    ModuleAdapter(
                        module,
                        SemanticTool.GET_TARGET_DISEASE_EVIDENCE,
                        self._association_handler(module),
                    )
                )
        if self.schema.module_support.get("drug.clinical_target", False):
            extended.append(
                ModuleAdapter(
                    "drug.clinical_target",
                    SemanticTool.GET_DRUG_EVIDENCE,
                    self.drug_clinical_target,
                )
            )
        drug_handlers = (
            ("drug.core", self.drug_core),
            ("drug.mechanism", self.drug_mechanism),
            ("drug.indication", self.drug_indication),
            ("drug.warning", self.drug_warning),
            ("drug.adverse_event", self.drug_adverse_event),
            ("drug.pharmacogenomics", self.drug_pharmacogenomics),
        )
        for module, handler in drug_handlers:
            if self.schema.module_support.get(module, False):
                extended.append(ModuleAdapter(module, SemanticTool.GET_DRUG_EVIDENCE, handler))
        requirements = {
            "locus.study": "locus.credible_set",
            "locus.variant": "locus.variant",
            "locus.variant_consequence": "locus.variant_consequence",
            "locus.credible_set": "locus.credible_set",
            "locus.fine_mapping": "locus.fine_mapping",
            "locus.qtl": "locus.credible_set",
            "locus.colocalisation": "locus.colocalisation",
            "locus.enhancer_to_gene": "locus.enhancer_to_gene",
            "locus.l2g": "locus.l2g",
            "target.core": "target.core",
            "target.function": "target.core",
            "target.pathway": "target.core",
            "target.expression": "target.expression",
            "target.go": "target.go",
            "target.interaction": "target.interaction",
            "evidence.europepmc": "target.literature",
            "disease.core": "disease.core",
            "disease.ontology": "disease.core",
            "association.overall_direct": "association.datasource_direct",
            "association.overall_indirect": "association.datasource_direct",
            "association.datasource_direct": "association.datasource_direct",
            "association.datasource_indirect": "association.datasource_direct",
            "association.datatype_direct": "association.datasource_direct",
            "association.datatype_indirect": "association.datasource_direct",
            "drug.clinical_target": "drug.clinical_target",
            "drug.core": "drug.core",
            "drug.mechanism": "drug.mechanism",
            "drug.indication": "drug.indication",
            "drug.warning": "drug.warning",
            "drug.adverse_event": "drug.adverse_event",
            "drug.pharmacogenomics": "drug.pharmacogenomics",
        }
        return tuple(
            adapter
            for adapter in extended
            if adapter.module == "entity.resolve"
            or self.schema.module_support.get(
                requirements.get(adapter.module, adapter.module), False
            )
        )

    def service(
        self,
        *,
        policy_sha256: str = "unknown",
        blocked_modules: frozenset[str] = frozenset(),
    ) -> SemanticToolService:
        backend_name = _provider_backend_name(self.provider)
        return SemanticToolService(
            provider=backend_name,
            data_release=(
                self.schema.data_version
                if self.schema is not None
                else self.provider.capabilities.release
            ),
            snapshot_sha256="live:not-a-frozen-snapshot",
            schema_sha256=(
                self.schema.schema_sha256
                if self.schema is not None
                else self.provider.capabilities.schema_hash
            ),
            policy_sha256=policy_sha256,
            adapters=self.adapters(),
            blocked_modules=blocked_modules,
            limits={
                "max_variants": 100,
                "max_targets": 200,
                "max_discovered_targets": parameter("ranking.max_discovered_targets"),
                "max_discovered_l2g_targets_per_locus": parameter("ranking.max_l2g_targets_per_locus"),
                "max_page_size": 500,
                "network_source": self.provider.endpoint,
                "coverage_gap_region_lookup": self.region_gene_lookup is not None,
            },
            upstream_metadata={
                "source": "Open Targets Platform",
                "transport": _provider_transport_description(self.provider),
                "endpoint": self.provider.endpoint,
                "official_mcp_tools_are_generic": (backend_name == "official_mcp"),
                "generic_operations_emulated_over_graphql": (backend_name == "official_graphql"),
                "api_version": self.schema.api_version if self.schema else "unknown",
                "data_version": self.schema.data_version if self.schema else "unknown",
                "schema_discovered": self.schema is not None,
                "ensembl_region_fallback_enabled": self.region_gene_lookup is not None,
                "module_support": (dict(self.schema.module_support) if self.schema else {}),
            },
        )

    async def resolve_entities(self, arguments: Mapping[str, Any]) -> ModuleResult:
        queries = arguments.get("queries", ())
        phenotype_only = bool(queries) and all(
            isinstance(item, Mapping) and set(item.get("types", ())) == {"disease"}
            for item in queries
        )
        cache_key = sha256_json(dict(arguments)) if phenotype_only else None
        if cache_key in self._phenotype_resolution_cache:
            self._phenotype_resolution_cache.move_to_end(cache_key)
            return self._phenotype_resolution_cache[cache_key]
        values = [
            str(item.get("text", "")).strip()
            for item in queries
            if isinstance(item, Mapping) and str(item.get("text", "")).strip()
        ]
        if not values:
            return ModuleResult("entity.resolve", ModuleStatus.NO_DATA)
        # A supported locus can contain 100 variants plus 200 candidate genes.
        # Keep each upstream request bounded, while resolving that same finite
        # input inside one semantic plan step. Any failed batch invalidates the
        # resolution step before an incomplete mapping can be used.
        responses = []
        for offset in range(0, len(queries), 100):
            batch = queries[offset:offset + 100]
            response = await self.provider._call_tool("search_entities", {
                "query_strings": [str(item["text"]).strip() for item in batch],
            })
            response = _unwrap_optional_success_mapping(response, "search_entities")
            responses.extend((item, response) for item in batch)
        max_hits = int(arguments.get("max_hits", 5))
        records: list[dict[str, Any]] = []
        for item, response in responses:
            text = str(item["text"]).strip()
            hits = _mapping_value_case_insensitive(response, text)
            if hits is None:
                raise ValueError(f"official entity resolver omitted requested query {text!r}")
            if not isinstance(hits, list):
                raise ValueError(f"official entity results for {text!r} are not a list")
            allowed = {str(value).casefold() for value in item.get("types", ())}
            normalized = []
            for hit in hits:
                if not isinstance(hit, Mapping):
                    continue
                entity_type = str(hit.get("type", "")).rsplit(".", 1)[-1].casefold()
                if (
                    allowed
                    and entity_type not in allowed
                    and not (entity_type == "gene" and "target" in allowed)
                ):
                    continue
                identifier = str(hit.get("id", "")).strip()
                normalized_type = "target" if entity_type == "gene" else entity_type
                if not _canonical_entity_id(identifier, normalized_type):
                    continue
                normalized.append(
                    {
                        "id": identifier,
                        "type": normalized_type,
                        **({"label": str(hit.get("name") or hit.get("label") or "")}
                           if normalized_type == "disease" else {}),
                        "match": (
                            "exact_label"
                            if (_disease_hit_matches_query(hit, text, identifier)
                                if normalized_type == "disease"
                                else _hit_matches_query(hit, text, identifier))
                            else "official_search_candidate"
                        ),
                    }
                )
            unique = _deduplicate(normalized)
            exact = [hit for hit in unique if hit["match"] == "exact_label"]
            selected = exact if len(exact) == 1 else unique
            truncated = len(selected) > max_hits
            visible = selected[:max_hits]
            auto_selectable = len(selected) == 1 and not truncated and (
                selected[0]["type"] != "disease" or len(exact) == 1
            )
            records.append(
                {
                    "query": text,
                    "status": (
                        "no_match"
                        if not unique
                        else "resolved"
                        if auto_selectable
                        else "candidate_matches"
                    ),
                    "hits": visible,
                    "auto_selectable": auto_selectable,
                    "truncated": truncated,
                    "warning": (
                        "A unique exact or type-compatible canonical ID was returned "
                        "by the official resolver."
                        if auto_selectable
                        else "No unique exact phenotype match was established."
                        if allowed == {"disease"}
                        else "Official search returned multiple candidates; GWASMind "
                        "will not choose among ambiguous mappings."
                    ),
                }
            )
        result = ModuleResult(
            "entity.resolve",
            ModuleStatus.SUCCESS,
            records=tuple(records),
        )
        if cache_key is not None:
            self._phenotype_resolution_cache[cache_key] = result
            while len(self._phenotype_resolution_cache) > 64:
                self._phenotype_resolution_cache.popitem(last=False)
        return result

    async def locus_study(self, arguments: Mapping[str, Any]) -> ModuleResult:
        bundle = await self._locus_base(arguments)
        return _with_live_coverage(
            _records_result(
                "locus.study", bundle.records, relation="study_metadata"
            ),
            bundle,
        )

    async def locus_variant(self, arguments: Mapping[str, Any]) -> ModuleResult:
        variants = await self._variant_profiles(arguments)
        records = tuple(
            {
                "module": "locus.variant",
                "dataset": "variant",
                "relation": "variant_annotation",
                "subject": {"type": "variant", "id": variant.get("id")},
                "object": None,
                "measurements": {},
                "context": {
                    "chromosome": variant.get("chromosome"),
                    "position": variant.get("position"),
                    "reference_allele": variant.get("referenceAllele"),
                    "alternate_allele": variant.get("alternateAllele"),
                    "rs_ids": variant.get("rsIds"),
                },
                "provenance": {
                    "source": _provider_source(self.provider),
                    "data_release": self.schema.data_version if self.schema else "live",
                },
                "upstream": {"raw": variant},
                "derived": {},
            }
            for variant in variants
        )
        return ModuleResult(
            "locus.variant",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=records,
        )

    async def locus_variant_consequence(self, arguments: Mapping[str, Any]) -> ModuleResult:
        """Return exact OT transcript consequences or unranked regional candidates."""

        profiles = await self._variant_consequence_profiles(arguments)
        candidates = {
            str(item).strip()
            for item in arguments.get("candidate_gene_ids", ())
            if str(item).strip()
        }
        per_variant_budget = min(max(int(arguments.get("page_size", parameter("retrieval.semantic_page_size"))), 1), 500)
        global_budget = MAX_VARIANT_CONSEQUENCE_RECORDS
        records: list[dict[str, Any]] = []
        truncated = False
        for variant in profiles:
            variant_id = str(variant.get("id") or "").strip()
            if not variant_id:
                continue
            consequences = variant.get("transcriptConsequences") or ()
            if not isinstance(consequences, (list, tuple)):
                raise ValueError("variant transcriptConsequences must be a list")
            retained_for_variant = 0
            for consequence in consequences:
                if not isinstance(consequence, Mapping):
                    continue
                target = consequence.get("target")
                target = target if isinstance(target, Mapping) else {}
                gene_id = str(target.get("id") or consequence.get("targetId") or "").strip()
                symbol = str(
                    target.get("approvedSymbol") or consequence.get("approvedSymbol") or ""
                ).strip()
                if not gene_id:
                    continue
                if candidates and not ({gene_id, symbol} - {""}).intersection(candidates):
                    continue
                if (
                    retained_for_variant >= per_variant_budget
                    or len(records) >= global_budget
                ):
                    truncated = True
                    # Continue scanning without retaining rows. This proves a
                    # true budget overrun while keeping memory bounded.
                    continue
                records.append(
                    {
                        "module": "locus.variant_consequence",
                        "dataset": "variant",
                        "relation": "variant_transcript_consequence",
                        "subject": {"type": "variant", "id": variant_id},
                        "object": {
                            "type": "target",
                            "id": gene_id,
                            "label": symbol or None,
                        },
                        "measurements": {
                            "consequence_score": consequence.get("consequenceScore"),
                            "distance_from_footprint": consequence.get("distanceFromFootprint"),
                            "distance_from_tss": consequence.get("distanceFromTss"),
                            "sift": consequence.get("siftPrediction"),
                            "polyphen": consequence.get("polyphenPrediction"),
                        },
                        "context": {"approved_symbol": symbol or None},
                        "provenance": self._live_provenance(),
                        "upstream": {"raw": dict(consequence)},
                        "derived": {
                            "candidate_basis": "variant_transcript_consequence",
                            "unranked": True,
                        },
                    }
                )
                retained_for_variant += 1
        warnings: list[str] = []
        if profiles:
            if not records:
                warnings.append(
                    "Open Targets returned the exact variant record(s), but no matching "
                    "transcript consequences were available."
                )
        else:
            warnings.append(
                "Open Targets returned no exact variant annotation record for the "
                "requested canonical variant IDs."
            )
        include_region_context = arguments.get("allow_region_fallback", False)
        if not isinstance(include_region_context, bool):
            raise TypeError("allow_region_fallback must be boolean")
        if include_region_context:
            region_records, region_warnings = await self._region_gene_context(
                arguments,
                candidates=candidates,
                consequence_profiles=profiles,
            )
            records.extend(region_records)
            warnings.extend(region_warnings)
        return ModuleResult(
            "locus.variant_consequence",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=tuple(records),
            truncated=truncated,
            warnings=tuple(warnings),
        )

    async def _region_gene_context(
        self,
        arguments: Mapping[str, Any],
        *,
        candidates: set[str],
        consequence_profiles: tuple[Mapping[str, Any], ...] = (),
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Return the auditable ±1 Mb annotated-gene gate input."""

        region = grch38_region_from_variant_ids(
            str(item) for item in arguments.get("variant_ids", ())
        )
        if region is None:
            return [
                _region_gene_summary_record(
                    region=None,
                    status="unavailable",
                    source=ENSEMBL_REST_SOURCE,
                    data_release="GRCh38",
                )
            ], [
                (
                    f"The ±{local_gene_window_label()} annotated-gene check was unavailable because one "
                    "same-chromosome GRCh38 interval could not be derived."
                )
            ]
        witnesses = nearby_gene_witnesses_from_ot(
            consequence_profiles,
            variant_ids=tuple(str(item) for item in arguments.get("variant_ids", ())),
            genome_build=region.genome_build,
        )
        if witnesses:
            summary = _region_gene_summary_record(
                region=region,
                status="nonempty",
                source=_provider_source(self.provider),
                data_release=self.schema.data_version if self.schema else "live",
                annotated_gene_count=len(witnesses),
                candidate_gene_ids=tuple(item["gene_id"] for item in witnesses),
                query_start=max(1, region.start - parameter("analysis.local_gene_window_bp")),
                query_end=region.end + parameter("analysis.local_gene_window_bp"),
                dataset="opentargets_region_gene_context",
                count_is_lower_bound=True,
            )
            summary["upstream"]["raw"] = {"nearby_gene_witnesses": list(witnesses)}
            summary["derived"]["existence_basis"] = "selected_variant_transcript_footprint_distance"
            return [summary], [
                "Exact selected-variant transcript annotations establish at least "
                f"{len(witnesses)} gene(s) within {local_gene_window_label()}. This is a lower bound, not a complete "
                "regional gene census; proximity is not direct causal-gene evidence."
            ]
        if self.region_gene_lookup is None:
            return [
                _region_gene_summary_record(
                    region=region,
                    status="unavailable",
                    source=ENSEMBL_REST_SOURCE,
                    data_release=region.genome_build,
                )
            ], [f"The ±{local_gene_window_label()} Ensembl annotated-gene check is not configured."]
        try:
            lookup_with_metadata = getattr(
                self.region_gene_lookup,
                "lookup_with_metadata",
                None,
            )
            if callable(lookup_with_metadata):
                result = await lookup_with_metadata(region)
                regional = tuple(result.candidates)
                total_gene_count = int(result.total_gene_count)
                query_start = int(result.query_start)
                query_end = int(result.query_end)
                flank_bp = int(result.flank_bp)
                lookup_truncated = bool(result.coverage_truncated)
                candidate_list_truncated = bool(result.candidate_list_truncated)
            else:
                regional = await self.region_gene_lookup.lookup(region)
                total_gene_count = len(regional)
                flank_bp = int(getattr(self.region_gene_lookup, "flank_bp", parameter("analysis.local_gene_window_bp")))
                query_start = max(1, region.start - flank_bp)
                query_end = region.end + flank_bp
                lookup_truncated = False
                candidate_list_truncated = False
        except EnsemblRegionLookupError as exc:
            return [
                _region_gene_summary_record(
                    region=region,
                    status="unavailable",
                    source=ENSEMBL_REST_SOURCE,
                    data_release=region.genome_build,
                )
            ], [f"The ±{local_gene_window_label()} Ensembl annotated-gene check was unavailable: {exc}"]

        records = [
            _region_gene_summary_record(
                region=region,
                status="complete",
                source=ENSEMBL_REST_SOURCE,
                data_release=region.genome_build,
                annotated_gene_count=total_gene_count,
                candidate_gene_ids=tuple(item.gene_id for item in regional),
                query_start=query_start,
                query_end=query_end,
                flank_bp=flank_bp,
                truncated=lookup_truncated,
                candidate_list_truncated=candidate_list_truncated,
            )
        ]
        for candidate in regional:
            if candidates and candidate.gene_id not in candidates:
                continue
            records.append(
                _region_gene_candidate_record(
                    region,
                    candidate,
                    query_start=query_start,
                    query_end=query_end,
                    flank_bp=flank_bp,
                    source=ENSEMBL_REST_SOURCE,
                    data_release=region.genome_build,
                )
            )
        return records, [
            (
                f"The ±{local_gene_window_label()} Ensembl gene context is an unranked locus-informativeness "
                "check; positional proximity is not direct causal-gene evidence."
            )
        ]

    async def _variant_consequence_profiles(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        requested = tuple(str(item) for item in arguments.get("variant_ids", ()))
        if not requested:
            raise ValueError("variant_ids are required")
        key = sha256_json({"variant_consequence_ids": requested})
        if key in self._variant_consequence_cache:
            self._variant_consequence_cache.move_to_end(key)
            return self._variant_consequence_cache[key]
        variant_profile_key = sha256_json({"variant_ids": requested})
        if (
            variant_profile_key in self._variant_cache
            and not self._variant_cache[variant_profile_key]
        ):
            # The mandatory locus.variant stage already established that every
            # exact Open Targets variant is absent. Avoid a second upstream
            # batch before the single bounded regional fallback.
            self._variant_consequence_cache[key] = ()
            return ()
        canonical = (
            list(requested)
            if all(_DIRECT_VARIANT_ID.fullmatch(item) for item in requested)
            else await self.provider._resolve_entities(list(requested), entity_type="variant")
        )
        aliased_query = getattr(self.provider, "_query_variant_profiles", None)
        if canonical and callable(aliased_query):
            values = await aliased_query(canonical, include_transcript_consequences=True)
        else:
            query = """
                query GWASMindVariantConsequences($variantId: String!) {
                  variant(variantId: $variantId) {
                    id chromosome position referenceAllele alternateAllele rsIds
                    transcriptConsequences {
                      target { id approvedSymbol }
                      consequenceScore distanceFromFootprint distanceFromTss
                      siftPrediction polyphenPrediction
                    }
                  }
                }
            """
            values = (
                await self.provider._batch_query(
                    query_string=query,
                    variables_list=[{"variantId": item} for item in canonical],
                    key_field="variantId",
                    root_field="variant",
                )
                if canonical
                else []
            )
        result = tuple(dict(item) for item in values if isinstance(item, Mapping))
        self._variant_consequence_cache[key] = result
        self._variant_consequence_cache.move_to_end(key)
        while len(self._variant_consequence_cache) > 8:
            self._variant_consequence_cache.popitem(last=False)
        return result

    async def _variant_profiles(self, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        requested = tuple(str(item) for item in arguments.get("variant_ids", ()))
        if not requested:
            raise ValueError("variant_ids are required")
        key = sha256_json({"variant_ids": requested})
        if key in self._variant_cache:
            self._variant_cache.move_to_end(key)
            return self._variant_cache[key]
        canonical = (
            list(requested)
            if all(_DIRECT_VARIANT_ID.fullmatch(item) for item in requested)
            else await self.provider._resolve_entities(list(requested), entity_type="variant")
        )
        aliased_query = getattr(self.provider, "_query_variant_profiles", None)
        if canonical and callable(aliased_query):
            values = await aliased_query(canonical)
        else:
            query = """
                query GWASMindVariantProfile($variantId: String!) {
                  variant(variantId: $variantId) {
                    id chromosome position referenceAllele alternateAllele rsIds
                  }
                }
            """
            values = (
                await self.provider._batch_query(
                    query_string=query,
                    variables_list=[{"variantId": item} for item in canonical],
                    key_field="variantId",
                    root_field="variant",
                )
                if canonical
                else []
            )
        result = tuple(item for item in values if item is not None)
        self._variant_cache[key] = result
        self._variant_cache.move_to_end(key)
        while len(self._variant_cache) > 8:
            self._variant_cache.popitem(last=False)
        return result

    async def locus_credible_set(self, arguments: Mapping[str, Any]) -> ModuleResult:
        bundle = await self._locus_base(arguments)
        return _with_live_coverage(
            _credible_set_membership_result(bundle.records), bundle
        )

    async def locus_fine_mapping(self, arguments: Mapping[str, Any]) -> ModuleResult:
        bundle = await self._locus_base(arguments)
        return _with_live_coverage(_fine_mapping_result(bundle.records), bundle)

    async def locus_qtl(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._kind_result(arguments, EvidenceKind.QTL, "locus.qtl")

    async def locus_colocalisation(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._kind_result(
            arguments, EvidenceKind.COLOCALIZATION, "locus.colocalisation"
        )

    async def locus_enhancer_to_gene(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._kind_result(arguments, EvidenceKind.REGULATORY, "locus.enhancer_to_gene")

    async def locus_l2g(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._kind_result(arguments, EvidenceKind.L2G, "locus.l2g")

    async def target_function(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._target_kind_result(arguments, EvidenceKind.FUNCTION, "target.function")

    async def target_core(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._target_kind_result(arguments, EvidenceKind.FUNCTION, "target.core")

    async def target_pathway(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._target_kind_result(arguments, EvidenceKind.FUNCTION, "target.pathway")

    async def target_expression(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._target_kind_result(
            arguments, EvidenceKind.EXPRESSION, "target.expression"
        )

    async def target_literature(self, arguments: Mapping[str, Any]) -> ModuleResult:
        requested_publications = arguments.get("publication_ids", ())
        if not isinstance(requested_publications, (list, tuple)):
            raise ValueError("publication_ids must be an array of PMIDs")
        if len(requested_publications) > MAX_PUBLICATION_ENRICHMENT_PMIDS:
            raise ValueError("publication_ids exceeds the supported metadata budget")
        explicit_pmids = tuple(normalize_pmid(value) for value in requested_publications)
        if any(pmid is None for pmid in explicit_pmids):
            raise ValueError("publication_ids must contain valid PMIDs")
        explicit_pmids = tuple(dict.fromkeys(explicit_pmids))
        bundle = await self._fetch_bundle(arguments, {EvidenceKind.LITERATURE}, target_mode=True)
        records = tuple(replace(record, payload={**record.payload, 'pmid_origin': 'current_locus_evidence'})
            if record.pmid in explicit_pmids else record for record in bundle.records)
        existing_pmids = {record.pmid for record in records}
        records = (
            *records,
            *(EvidenceRecord(
                evidence_id=f"study-publication:{pmid}",
                kind=EvidenceKind.LITERATURE,
                dataset="retrieved_study_publication",
                source=_provider_source(self.provider),
                data_release=self.schema.data_version if self.schema else "live",
                pmid=pmid,
                payload={"pmid_origin": "current_locus_evidence", "metadata_only": True},
            ) for pmid in explicit_pmids if pmid not in existing_pmids),
        )
        embedded: tuple[EvidenceRecord, ...] = ()
        warnings: tuple[str, ...] = bundle.warnings
        if {"target.core", "target.function"}.intersection(arguments.get("modules", ())):
            # Target profiles are normally already cached by the preceding modules.
            # Embedded function citations are evidence-linked priorities, even when
            # they are old or missing from the target-literature occurrence feed.
            try:
                function_records = await self._fetch(
                    arguments, {EvidenceKind.FUNCTION}, target_mode=True
                )
            except Exception:
                function_records = ()
                warnings = (*warnings,
                    "Function-citation metadata was unavailable; retained target literature.",
                )
            embedded = _function_publication_records(function_records)
            existing = {(record.gene_id, record.pmid) for record in records}
            records = (
                *records,
                *(row for row in embedded if (row.gene_id, row.pmid) not in existing),
            )
        if self.publication_enricher is not None and records:
            cited_pmids = tuple(dict.fromkeys((
                *explicit_pmids,
                *_publication_enrichment_pmids(
                    embedded, limit=MAX_CITED_PUBLICATION_ENRICHMENT_PMIDS
                ),
            )))
            pmids = tuple(dict.fromkeys((
                *cited_pmids,
                *_publication_enrichment_pmids(records, limit=MAX_PUBLICATION_ENRICHMENT_PMIDS),
            )))[:min(
                MAX_CITED_PUBLICATION_ENRICHMENT_PMIDS,
                max(MAX_PUBLICATION_ENRICHMENT_PMIDS, len(cited_pmids)),
            )]
            if len(cited_pmids) > len(pmids):
                warnings = (*warnings,
                    ("Evidence-linked publication metadata exceeded the retrieval bound; "
                     "unresolved PMID provenance was retained."),
                )
            # At most four requests, each with the provider's established 100-PMID
            # bound. These are only PMIDs in this call's ledger, not cached genes
            # from another locus or publications proposed by the language model.
            responses = await asyncio.gather(*(
                self.publication_enricher.fetch(pmids[start:start + 100])
                for start in range(0, len(pmids), 100)
            ), return_exceptions=True)
            metadata: dict[str, dict[str, Any]] = {}
            for response in responses:
                if isinstance(response, EuropePMCError):
                    warnings = (*warnings,
                        ("Publication metadata was unavailable for one batch; "
                         "retained PMID provenance."),
                    )
                elif isinstance(response, BaseException):
                    raise response
                else:
                    metadata.update(response)
            missing_pmids = set()
            for pmid in cited_pmids:
                value = metadata.get(pmid, {})
                publication = publication_from_metadata(
                    pmid, authors=value.get("authors"), title=value.get("publicationTitle"),
                    journal=value.get("publicationJournal"),
                    publication_date=value.get("publicationDate"),
                )
                if publication is None or not publication.metadata_complete:
                    missing_pmids.add(pmid)
            if missing_pmids:
                warnings = (*warnings,
                    (f"Publication metadata was not returned for {len(missing_pmids)} "
                     "retrieved study PMID(s), including evidence-linked function citations; "
                     "their original identifiers were retained."),
                )
            if metadata:
                records = tuple(
                    replace(
                        record,
                        payload={
                            **record.payload,
                            **metadata.get(str(record.pmid), {}),
                            **({"metadata_source": "Europe PMC"}
                               if str(record.pmid) in metadata else {}),
                        },
                    )
                    for record in records
                )
        result = _records_result(
            "evidence.europepmc",
            records,
            relation="evidence_europepmc",
        )
        return _with_live_coverage(replace(result, warnings=warnings), bundle)

    async def target_safety(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._target_extended_result(arguments, "target.safety", "safetyLiabilities")

    async def target_tractability(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._target_extended_result(arguments, "target.tractability", "tractability")

    async def target_go(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._target_extended_result(arguments, "target.go", "geneOntology")

    async def target_interaction(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._target_extended_result(arguments, "target.interaction", "interactions")

    async def _target_extended_result(
        self,
        arguments: Mapping[str, Any],
        module: str,
        field: str,
    ) -> ModuleResult:
        targets = await self._target_extended(arguments)
        truncated = False
        if field == "interactions":
            visible_targets = []
            for target in targets:
                if field not in target:
                    raise ValueError("target response omitted requested interactions")
                connection = target[field]
                if connection is None:
                    continue
                if not isinstance(connection, Mapping):
                    raise TypeError("interactions must be an object or null")
                rows = connection.get("rows")
                count = connection.get("count")
                if not isinstance(rows, list) or any(
                    not isinstance(row, Mapping) for row in rows
                ):
                    raise ValueError("interactions.rows must be a list of objects")
                if (
                    not isinstance(count, int)
                    or isinstance(count, bool)
                    or count < len(rows)
                ):
                    raise ValueError("interactions.count must cover the returned rows")
                truncated = truncated or count > len(rows)
                if rows:
                    visible_targets.append(target)
            targets = tuple(visible_targets)
        records = tuple(
            {
                "module": module,
                "dataset": "target",
                "relation": module.replace(".", "_"),
                "subject": {
                    "type": "target",
                    "id": target.get("id"),
                    "label": target.get("approvedSymbol"),
                },
                "object": None,
                "measurements": {},
                "context": {},
                "provenance": {
                    "source": _provider_source(self.provider),
                    "data_release": (
                        self.schema.data_version
                        if self.schema
                        else self.provider.capabilities.release
                    ),
                },
                "upstream": {"raw": {field: target.get(field)}},
                "derived": {},
            }
            for target in targets
            if target.get(field) not in (None, [], {}, ())
        )
        return ModuleResult(
            module,
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=records,
            truncated=truncated,
        )

    async def _target_extended(self, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        target_ids = tuple(str(item) for item in arguments.get("target_ids", ()))
        if not target_ids:
            raise ValueError("target_ids are required")
        requested = set(arguments.get("modules", ()))
        include_safety = "target.safety" in requested
        include_tractability = "target.tractability" in requested
        include_go = "target.go" in requested
        include_interactions = "target.interaction" in requested
        fields = ["id", "approvedSymbol"]
        if include_safety:
            fields.append(
                "safetyLiabilities { event eventId datasource literature url "
                "effects { direction dosing } "
                "biosamples { cellFormat cellLabel tissueId tissueLabel } "
                "studies { description name type } }"
            )
        if include_tractability:
            fields.append("tractability { label modality value }")
        if include_go:
            fields.append("geneOntology { term { id label } aspect evidence geneProduct source }")
        if include_interactions:
            fields.append(
                "interactions { count rows { intA intB intABiologicalRole "
                "intBBiologicalRole count sourceDatabase score "
                "targetA { id approvedSymbol } targetB { id approvedSymbol } } }"
            )
        key = sha256_json({"target_ids": target_ids, "fields": fields})
        if key in self._extended_cache:
            self._extended_cache.move_to_end(key)
            return self._extended_cache[key]
        query = (
            "query GWASMindTargetAssessment($ensemblId: String!) { "
            "target(ensemblId: $ensemblId) { " + " ".join(fields) + " } }"
        )
        values = await self.provider._batch_query(
            query_string=query,
            variables_list=[{"ensemblId": item} for item in target_ids],
            key_field="ensemblId",
            root_field="target",
        )
        result = tuple(item for item in values if item is not None)
        self._extended_cache[key] = result
        self._extended_cache.move_to_end(key)
        while len(self._extended_cache) > 4:
            self._extended_cache.popitem(last=False)
        return result

    async def disease_core(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._disease_result(arguments, "disease.core", "core")

    async def disease_ontology(self, arguments: Mapping[str, Any]) -> ModuleResult:
        return await self._disease_result(arguments, "disease.ontology", "ontology")

    async def _disease_result(
        self, arguments: Mapping[str, Any], module: str, field_group: str
    ) -> ModuleResult:
        diseases = await self._disease_profiles(arguments)
        core_fields = ("id", "name", "description", "dbXRefs", "synonyms")
        ontology_fields = (
            "id",
            "parents",
            "children",
            "ancestors",
            "descendants",
            "therapeuticAreas",
            "ontology",
        )
        fields = core_fields if field_group == "core" else ontology_fields
        records = tuple(
            {
                "module": module,
                "dataset": "disease",
                "relation": module.replace(".", "_"),
                "subject": {
                    "type": "disease",
                    "id": disease.get("id"),
                    "label": disease.get("name"),
                },
                "object": None,
                "measurements": {},
                "context": {},
                "provenance": {
                    "source": _provider_source(self.provider),
                    "data_release": self.schema.data_version if self.schema else "live",
                },
                "upstream": {"raw": {field: disease.get(field) for field in fields}},
                "derived": {},
            }
            for disease in diseases
        )
        return ModuleResult(
            module,
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=records,
        )

    async def _disease_profiles(self, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        disease_ids = tuple(str(item) for item in arguments.get("disease_ids", ()))
        if not disease_ids:
            raise ValueError("disease_ids are required")
        key = sha256_json({"disease_ids": disease_ids})
        if key in self._disease_cache:
            self._disease_cache.move_to_end(key)
            return self._disease_cache[key]
        query = """
            query GWASMindDiseaseProfile($efoId: String!) {
              disease(efoId: $efoId) {
                id name description dbXRefs
                synonyms { relation terms }
                parents { id name }
                children { id name }
                ancestors descendants
                therapeuticAreas { id name }
                ontology { isTherapeuticArea leaf }
              }
            }
        """
        values = await self.provider._batch_query(
            query_string=query,
            variables_list=[{"efoId": item} for item in disease_ids],
            key_field="efoId",
            root_field="disease",
        )
        result = tuple(item for item in values if item is not None)
        self._disease_cache[key] = result
        self._disease_cache.move_to_end(key)
        while len(self._disease_cache) > 4:
            self._disease_cache.popitem(last=False)
        return result

    def _association_handler(self, module: str):
        async def handler(arguments: Mapping[str, Any]) -> ModuleResult:
            return await self._association_result(arguments, module)

        return handler

    async def _association_result(self, arguments: Mapping[str, Any], module: str) -> ModuleResult:
        rows, truncated = await self._association_rows(arguments)
        required_scope = "indirect" if module.endswith("_indirect") else "direct"
        selected = [row for row in rows if row["association_scope"] == required_scope]
        if ".overall_" in module:
            records = tuple(
                _association_record(
                    module,
                    row,
                    source=_provider_source(self.provider),
                    measurements={"association_score": row.get("score")},
                )
                for row in selected
            )
        elif ".datasource_" in module:
            records = tuple(
                _association_record(
                    module,
                    row,
                    source=_provider_source(self.provider),
                    measurements={
                        "association_score": row.get("score"),
                        "datasource_scores": row.get("datasourceScores"),
                    },
                )
                for row in selected
            )
        else:
            records = tuple(
                _association_record(
                    module,
                    row,
                    source=_provider_source(self.provider),
                    measurements={
                        "association_score": row.get("score"),
                        "datatype_scores": row.get("datatypeScores"),
                    },
                )
                for row in selected
            )
        return ModuleResult(
            module,
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=records,
            truncated=truncated,
            warnings=(
                "Open Targets association scores are heuristic ranking scores, "
                "not probabilities or confidence scores.",
            ),
        )

    async def _association_rows(
        self, arguments: Mapping[str, Any]
    ) -> tuple[tuple[dict[str, Any], ...], bool]:
        target_ids = tuple(str(item) for item in arguments.get("target_ids", ()))
        disease_ids = tuple(str(item) for item in arguments.get("disease_ids", ()))
        if not target_ids:
            raise ValueError("target_ids are required")
        if len(disease_ids) > 20:
            raise ValueError("live association lookup accepts at most 20 disease IDs")
        scope_value = str(arguments.get("association_scope", "direct"))
        if not disease_ids and scope_value != "direct":
            raise ValueError("unfiltered target association lookup supports direct scope only")
        scopes = (
            (
                ("direct", False),
                ("indirect", True),
            )
            if scope_value == "both"
            else ((scope_value, scope_value == "indirect"),)
        )
        page_size = min(int(arguments.get("page_size", parameter("retrieval.semantic_page_size"))), 200)
        if page_size < 1:
            raise ValueError("page_size must be positive")
        key = sha256_json(
            {
                "target_ids": target_ids,
                "disease_ids": disease_ids,
                "scopes": scopes,
                "page_size": page_size,
            }
        )
        if key in self._association_cache:
            self._association_cache.move_to_end(key)
            return self._association_cache[key]
        if not disease_ids:
            result = await self._target_association_rows(target_ids, page_size)
            self._association_cache[key] = result
            self._association_cache.move_to_end(key)
            while len(self._association_cache) > 4:
                self._association_cache.popitem(last=False)
            return result
        query = """
            query GWASMindDiseaseAssociations(
              $efoId: String!, $pageIndex: Int!, $pageSize: Int!,
              $enableIndirect: Boolean!
            ) {
              disease(efoId: $efoId) {
                id name
                associatedTargets(
                  enableIndirect: $enableIndirect,
                  page: {index: $pageIndex, size: $pageSize}
                ) {
                  count
                  rows {
                    score
                    target { id approvedSymbol approvedName }
                    datasourceScores { id score }
                    datatypeScores { id score }
                  }
                }
              }
            }
        """
        target_set = set(target_ids)
        output: list[dict[str, Any]] = []
        truncated = False
        for scope_name, enable_indirect in scopes:
            active = set(disease_ids)
            found: dict[str, set[str]] = {item: set() for item in disease_ids}
            for page_index in range(parameter("retrieval.association_max_pages")):
                if not active:
                    break
                ordered_active = [item for item in disease_ids if item in active]
                values = await self.provider._batch_query(
                    query_string=query,
                    variables_list=[
                        {
                            "efoId": disease_id,
                            "pageIndex": page_index,
                            "pageSize": page_size,
                            "enableIndirect": enable_indirect,
                        }
                        for disease_id in ordered_active
                    ],
                    key_field="efoId",
                    root_field="disease",
                )
                for disease_id, disease in zip(ordered_active, values, strict=True):
                    if disease is None:
                        active.discard(disease_id)
                        continue
                    connection = disease.get("associatedTargets")
                    if not isinstance(connection, Mapping):
                        raise ValueError("associatedTargets must be an object")
                    rows = connection.get("rows") or []
                    if not isinstance(rows, list):
                        raise ValueError("associatedTargets.rows must be a list")
                    count = int(connection.get("count") or 0)
                    for row in rows:
                        if not isinstance(row, Mapping):
                            continue
                        target = row.get("target")
                        if not isinstance(target, Mapping):
                            continue
                        target_id = str(target.get("id") or "")
                        if target_id not in target_set:
                            continue
                        found[disease_id].add(target_id)
                        output.append(
                            {
                                **dict(row),
                                "disease": {
                                    "id": disease.get("id") or disease_id,
                                    "name": disease.get("name"),
                                },
                                "association_scope": scope_name,
                                "page_index": page_index,
                                "total_associated_targets": count,
                            }
                        )
                    exhausted = (page_index + 1) * page_size >= count or not rows
                    complete = target_set.issubset(found[disease_id])
                    if exhausted or complete:
                        active.discard(disease_id)
            if active:
                truncated = True
        result = (tuple(output), truncated)
        self._association_cache[key] = result
        self._association_cache.move_to_end(key)
        while len(self._association_cache) > 4:
            self._association_cache.popitem(last=False)
        return result

    async def _target_association_rows(
        self, target_ids: tuple[str, ...], page_size: int
    ) -> tuple[tuple[dict[str, Any], ...], bool]:
        """Retrieve top direct diseases for targets without guessing an EFO ID."""

        query = """
            query GWASMindTargetAssociations(
              $ensemblId: String!, $pageSize: Int!
            ) {
              target(ensemblId: $ensemblId) {
                id approvedSymbol approvedName
                associatedDiseases(
                  enableIndirect: false,
                  page: {index: 0, size: $pageSize}
                ) {
                  count
                  rows {
                    score
                    disease { id name }
                    datasourceScores { id score }
                    datatypeScores { id score }
                  }
                }
              }
            }
        """
        requested_targets = target_ids[:page_size]
        per_target_size = min(
            100,
            max(1, (page_size + len(requested_targets) - 1) // len(requested_targets)),
        )
        values = await self.provider._batch_query(
            query_string=query,
            variables_list=[
                {"ensemblId": target_id, "pageSize": per_target_size}
                for target_id in requested_targets
            ],
            key_field="ensemblId",
            root_field="target",
        )
        output: list[dict[str, Any]] = []
        truncated = len(requested_targets) < len(target_ids)
        for target_id, target in zip(requested_targets, values, strict=True):
            if target is None:
                continue
            connection = target.get("associatedDiseases")
            if not isinstance(connection, Mapping):
                raise ValueError("associatedDiseases must be an object")
            rows = connection.get("rows") or []
            if not isinstance(rows, list):
                raise ValueError("associatedDiseases.rows must be a list")
            count = int(connection.get("count") or 0)
            if count > len(rows):
                truncated = True
            target_value = {
                "id": target.get("id") or target_id,
                "approvedSymbol": target.get("approvedSymbol"),
                "approvedName": target.get("approvedName"),
            }
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                if len(output) >= page_size:
                    truncated = True
                    break
                output.append(
                    {
                        **dict(row),
                        "target": target_value,
                        "association_scope": "direct",
                        "page_index": 0,
                        "total_associated_diseases": count,
                    }
                )
        return tuple(output), truncated

    async def drug_core(self, arguments: Mapping[str, Any]) -> ModuleResult:
        drugs, truncated = await self._drug_profiles(arguments)
        records = tuple(
            {
                "module": "drug.core",
                "dataset": "drug_molecule",
                "relation": "drug_annotation",
                "subject": _drug_subject(drug),
                "object": None,
                "measurements": _present_values(
                    maximum_clinical_stage=drug.get("maximumClinicalStage")
                ),
                "context": _present_values(
                    drug_type=drug.get("drugType"),
                    description=drug.get("description"),
                    synonyms=drug.get("synonyms"),
                    trade_names=drug.get("tradeNames"),
                    cross_references=drug.get("crossReferences"),
                    parent_molecule=drug.get("parentMolecule"),
                    child_molecules=drug.get("childMolecules"),
                ),
                "provenance": self._live_provenance(),
                "upstream": {"raw": drug},
                "derived": {},
            }
            for drug in drugs
            if drug.get("id")
        )
        return ModuleResult(
            "drug.core",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=records,
            truncated=truncated,
        )

    async def drug_mechanism(self, arguments: Mapping[str, Any]) -> ModuleResult:
        drugs, truncated = await self._drug_profiles(arguments)
        target_filter = {str(item).casefold() for item in arguments.get("target_ids", ())}
        records: list[dict[str, Any]] = []
        for drug in drugs:
            connection = drug.get("mechanismsOfAction")
            rows, _ = _connection_rows(connection, "mechanismsOfAction")
            unique_actions = (
                connection.get("uniqueActionTypes") if isinstance(connection, Mapping) else None
            )
            unique_targets = (
                connection.get("uniqueTargetTypes") if isinstance(connection, Mapping) else None
            )
            for row in rows:
                if not _has_content(row):
                    continue
                targets = _mapping_list(row.get("targets"), "mechanism targets")
                if target_filter:
                    targets = tuple(
                        target
                        for target in targets
                        if str(target.get("id", "")).casefold() in target_filter
                    )
                    if not targets:
                        continue
                selected_targets: tuple[Mapping[str, Any] | None, ...] = (
                    targets if targets else (None,)
                )
                for target in selected_targets:
                    records.append(
                        {
                            "module": "drug.mechanism",
                            "dataset": "drug_mechanism_of_action",
                            "relation": "drug_target_mechanism",
                            "subject": _drug_subject(drug),
                            "object": (
                                {
                                    "type": "target",
                                    "id": target.get("id"),
                                    "label": target.get("approvedSymbol")
                                    or target.get("approvedName"),
                                }
                                if target is not None
                                else None
                            ),
                            "measurements": {},
                            "context": _present_values(
                                mechanism_of_action=row.get("mechanismOfAction"),
                                action_type=row.get("actionType"),
                                target_name=row.get("targetName"),
                                references=row.get("references"),
                                unique_action_types=unique_actions,
                                unique_target_types=unique_targets,
                            ),
                            "provenance": self._live_provenance(),
                            "upstream": {"raw": row},
                            "derived": {},
                        }
                    )
        return ModuleResult(
            "drug.mechanism",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=tuple(records),
            truncated=truncated,
        )

    async def drug_indication(self, arguments: Mapping[str, Any]) -> ModuleResult:
        drugs, truncated = await self._drug_profiles(arguments)
        disease_filter = {str(item).casefold() for item in arguments.get("disease_ids", ())}
        records: list[dict[str, Any]] = []
        for drug in drugs:
            rows, count = _connection_rows(drug.get("indications"), "indications")
            truncated = truncated or count > len(rows)
            for row in rows:
                if not _has_content(row):
                    continue
                disease = row.get("disease")
                disease = disease if isinstance(disease, Mapping) else {}
                disease_id = str(disease.get("id") or row.get("id") or "")
                if disease_filter and disease_id.casefold() not in disease_filter:
                    continue
                records.append(
                    {
                        "module": "drug.indication",
                        "dataset": "clinical_indication",
                        "relation": "drug_disease_clinical_indication",
                        "subject": _drug_subject(drug),
                        "object": {
                            "type": "disease",
                            "id": disease_id or None,
                            "label": disease.get("name"),
                        },
                        "measurements": _present_values(
                            max_clinical_stage=row.get("maxClinicalStage")
                        ),
                        "context": _present_values(
                            clinical_reports=row.get("clinicalReports"),
                            total_indications=count,
                        ),
                        "provenance": self._live_provenance(),
                        "upstream": {"raw": row},
                        "derived": {},
                    }
                )
        return ModuleResult(
            "drug.indication",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=tuple(records),
            truncated=truncated,
            warnings=(
                "Clinical stage is a source-native development category, not a "
                "probability of efficacy or genetic causality.",
            ),
        )

    async def drug_warning(self, arguments: Mapping[str, Any]) -> ModuleResult:
        drugs, truncated = await self._drug_profiles(arguments)
        disease_filter = {str(item).casefold() for item in arguments.get("disease_ids", ())}
        records: list[dict[str, Any]] = []
        for drug in drugs:
            warnings = _mapping_list(drug.get("drugWarnings"), "drugWarnings")
            for warning in warnings:
                if not _has_content(warning):
                    continue
                warning_ids = {
                    str(warning.get("efoId") or "").casefold(),
                    str(warning.get("efoIdForWarningClass") or "").casefold(),
                }
                if disease_filter and not disease_filter.intersection(warning_ids):
                    continue
                records.append(
                    {
                        "module": "drug.warning",
                        "dataset": "drug_warning",
                        "relation": "drug_safety_warning",
                        "subject": _drug_subject(drug),
                        "object": (
                            {
                                "type": "disease",
                                "id": warning.get("efoId"),
                                "label": warning.get("efoTerm"),
                            }
                            if warning.get("efoId") or warning.get("efoTerm")
                            else None
                        ),
                        "measurements": _present_values(year=warning.get("year")),
                        "context": _present_values(
                            warning_type=warning.get("warningType"),
                            toxicity_class=warning.get("toxicityClass"),
                            country=warning.get("country"),
                            description=warning.get("description"),
                            references=warning.get("references"),
                            chembl_ids=warning.get("chemblIds"),
                            warning_class_efo_id=warning.get("efoIdForWarningClass"),
                        ),
                        "provenance": self._live_provenance(),
                        "upstream": {"raw": warning},
                        "derived": {},
                    }
                )
        return ModuleResult(
            "drug.warning",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=tuple(records),
            truncated=truncated,
            warnings=(
                "Warnings are source annotations and do not quantify event "
                "probability for an individual patient.",
            ),
        )

    async def drug_adverse_event(self, arguments: Mapping[str, Any]) -> ModuleResult:
        drugs, truncated = await self._drug_profiles(arguments)
        records: list[dict[str, Any]] = []
        for drug in drugs:
            connection = drug.get("adverseEvents")
            rows, count = _connection_rows(connection, "adverseEvents")
            truncated = truncated or count > len(rows)
            critical_value = (
                connection.get("criticalValue") if isinstance(connection, Mapping) else None
            )
            for row in rows:
                if not _has_content(row):
                    continue
                records.append(
                    {
                        "module": "drug.adverse_event",
                        "dataset": "openfda_significant_adverse_drug_reactions",
                        "relation": "drug_adverse_event_signal",
                        "subject": _drug_subject(drug),
                        "object": {
                            "type": "adverse_event",
                            "id": row.get("meddraCode"),
                            "label": row.get("name"),
                        },
                        "measurements": _present_values(
                            report_count=row.get("count"),
                            log_likelihood_ratio=row.get("logLR"),
                            critical_value=critical_value,
                        ),
                        "context": _present_values(total_signals=count),
                        "provenance": self._live_provenance(),
                        "upstream": {"raw": row},
                        "derived": {},
                    }
                )
        return ModuleResult(
            "drug.adverse_event",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=tuple(records),
            truncated=truncated,
            warnings=(
                "OpenFDA/FAERS log-likelihood-ratio signals are not incidence "
                "rates, causal estimates, or calibrated probabilities.",
            ),
        )

    async def drug_pharmacogenomics(self, arguments: Mapping[str, Any]) -> ModuleResult:
        drugs, truncated = await self._drug_profiles(arguments)
        target_filter = {str(item).casefold() for item in arguments.get("target_ids", ())}
        records: list[dict[str, Any]] = []
        for drug in drugs:
            rows = _mapping_list(drug.get("pharmacogenomics"), "pharmacogenomics")
            for row in rows:
                if not _has_content(row):
                    continue
                target = row.get("target")
                target = target if isinstance(target, Mapping) else {}
                target_id = str(target.get("id") or row.get("targetFromSourceId") or "")
                if target_filter and target_id.casefold() not in target_filter:
                    continue
                records.append(
                    {
                        "module": "drug.pharmacogenomics",
                        "dataset": "pharmacogenomics",
                        "relation": "drug_genotype_phenotype_annotation",
                        "subject": _drug_subject(drug),
                        "object": (
                            {
                                "type": "target",
                                "id": target_id,
                                "label": target.get("approvedSymbol") or target.get("approvedName"),
                            }
                            if target_id
                            else None
                        ),
                        "measurements": {},
                        "context": _present_values(
                            evidence_level=row.get("evidenceLevel"),
                            genotype=row.get("genotype"),
                            genotype_annotation_text=row.get("genotypeAnnotationText"),
                            pgx_category=row.get("pgxCategory"),
                            phenotype_text=row.get("phenotypeText"),
                            variant_rs_id=row.get("variantRsId"),
                            variant_id=row.get("variantId"),
                            variant_functional_consequence_id=row.get(
                                "variantFunctionalConsequenceId"
                            ),
                            literature=row.get("literature"),
                            study_id=row.get("studyId"),
                            is_direct_target=row.get("isDirectTarget"),
                            drugs=row.get("drugs"),
                            variant_annotation=row.get("variantAnnotation"),
                        ),
                        "provenance": self._live_provenance(),
                        "upstream": {"raw": row},
                        "derived": {},
                    }
                )
        return ModuleResult(
            "drug.pharmacogenomics",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=tuple(records),
            truncated=truncated,
            warnings=(
                "Pharmacogenomics evidenceLevel is a source-native category, "
                "not a calibrated probability or GWAS gene-prioritisation score.",
            ),
        )

    async def drug_clinical_target(self, arguments: Mapping[str, Any]) -> ModuleResult:
        rows, truncated = await self._clinical_target_rows(arguments)
        drug_filter = {str(item) for item in arguments.get("drug_ids", ())}
        disease_filter = {str(item) for item in arguments.get("disease_ids", ())}
        records: list[dict[str, Any]] = []
        for row in rows:
            drug = row.get("drug") if isinstance(row.get("drug"), Mapping) else {}
            diseases = row.get("diseases") if isinstance(row.get("diseases"), list) else []
            disease_ids = {
                str(nested.get("disease", {}).get("id"))
                for nested in diseases
                if isinstance(nested, Mapping) and isinstance(nested.get("disease"), Mapping)
            }
            if drug_filter and str(drug.get("id")) not in drug_filter:
                continue
            if disease_filter and not disease_filter.intersection(disease_ids):
                continue
            records.append(
                {
                    "module": "drug.clinical_target",
                    "dataset": "clinical_target",
                    "relation": "drug_target_disease_clinical_candidate",
                    "subject": {
                        "type": "drug",
                        "id": drug.get("id"),
                        "label": drug.get("name"),
                    },
                    "object": {
                        "type": "target",
                        "id": row.get("targetId"),
                        "label": row.get("targetSymbol"),
                    },
                    "measurements": {"max_clinical_stage": row.get("maxClinicalStage")},
                    "context": {
                        "diseases": diseases,
                        "total_clinical_candidates": row.get("totalCount"),
                    },
                    "provenance": {
                        "source": _provider_source(self.provider),
                        "data_release": self.schema.data_version if self.schema else "live",
                    },
                    "upstream": {"raw": row},
                    "derived": {},
                }
            )
        return ModuleResult(
            "drug.clinical_target",
            ModuleStatus.SUCCESS if records else ModuleStatus.NO_DATA,
            records=tuple(records),
            truncated=truncated,
            warnings=(
                "Clinical stage is translational context and is not genetic "
                "evidence for locus-to-gene causality.",
            ),
        )

    async def _clinical_target_rows(
        self, arguments: Mapping[str, Any]
    ) -> tuple[tuple[dict[str, Any], ...], bool]:
        target_ids = tuple(str(item) for item in arguments.get("target_ids", ()))
        if not target_ids:
            raise ValueError(
                "the current clinical-target adapter requires target_ids; "
                "resolve targets before requesting drug evidence"
            )
        key = sha256_json({"target_ids": target_ids})
        if key in self._clinical_cache:
            self._clinical_cache.move_to_end(key)
            return self._clinical_cache[key]
        query = """
            query GWASMindClinicalTargets($ensemblId: String!) {
              target(ensemblId: $ensemblId) {
                id approvedSymbol
                drugAndClinicalCandidates {
                  count
                  rows {
                    maxClinicalStage
                    drug { id name drugType maximumClinicalStage }
                    diseases { disease { id name } }
                  }
                }
              }
            }
        """
        values = await self.provider._batch_query(
            query_string=query,
            variables_list=[{"ensemblId": item} for item in target_ids],
            key_field="ensemblId",
            root_field="target",
        )
        rows: list[dict[str, Any]] = []
        truncated = False
        for target in values:
            if target is None:
                continue
            connection = target.get("drugAndClinicalCandidates")
            if not isinstance(connection, Mapping):
                raise ValueError("drugAndClinicalCandidates must be an object")
            candidates = connection.get("rows") or []
            if not isinstance(candidates, list):
                raise ValueError("drugAndClinicalCandidates.rows must be a list")
            count = int(connection.get("count") or 0)
            truncated = truncated or count > len(candidates)
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                rows.append(
                    {
                        **dict(candidate),
                        "targetId": target.get("id"),
                        "targetSymbol": target.get("approvedSymbol"),
                        "totalCount": count,
                    }
                )
        result = (tuple(rows), truncated)
        self._clinical_cache[key] = result
        self._clinical_cache.move_to_end(key)
        while len(self._clinical_cache) > 4:
            self._clinical_cache.popitem(last=False)
        return result

    async def _drug_profiles(
        self, arguments: Mapping[str, Any]
    ) -> tuple[tuple[dict[str, Any], ...], bool]:
        drug_ids = _ordered_drug_ids(arguments.get("drug_ids", ()))
        discovery_truncated = False
        if not drug_ids and arguments.get("target_ids"):
            candidates, discovery_truncated = await self._clinical_target_rows(arguments)
            disease_filter = {str(item).casefold() for item in arguments.get("disease_ids", ())}
            discovered: list[str] = []
            for candidate in candidates:
                diseases = candidate.get("diseases")
                diseases = diseases if isinstance(diseases, list) else []
                disease_ids = {
                    str(nested.get("disease", {}).get("id", "")).casefold()
                    for nested in diseases
                    if isinstance(nested, Mapping) and isinstance(nested.get("disease"), Mapping)
                }
                if disease_filter and not disease_filter.intersection(disease_ids):
                    continue
                drug = candidate.get("drug")
                if isinstance(drug, Mapping) and drug.get("id"):
                    discovered.append(str(drug["id"]))
            drug_ids = _ordered_drug_ids(discovered)
        if not drug_ids:
            return (), discovery_truncated

        page_size = min(max(int(arguments.get("page_size", parameter("retrieval.semantic_page_size"))), 1), 100)
        selected_ids = drug_ids[:page_size]
        truncated = discovery_truncated or len(selected_ids) < len(drug_ids)
        requested = {str(item) for item in arguments.get("modules", ())} or {
            "drug.core",
            "drug.mechanism",
            "drug.indication",
            "drug.warning",
            "drug.adverse_event",
            "drug.pharmacogenomics",
        }
        fields = ["id name"]
        if "drug.core" in requested:
            fields.append(
                """
                drugType maximumClinicalStage description
                synonyms { label source }
                tradeNames { label source }
                crossReferences { source ids }
                parentMolecule { id name }
                childMolecules { id name }
                """
            )
        if "drug.mechanism" in requested:
            fields.append(
                """
                mechanismsOfAction {
                  uniqueActionTypes uniqueTargetTypes
                  rows {
                    mechanismOfAction actionType targetName
                    references { ids source urls }
                    targets { id approvedSymbol approvedName }
                  }
                }
                """
            )
        if "drug.indication" in requested:
            fields.append(
                """
                indications {
                  count
                  rows {
                    id maxClinicalStage
                    disease { id name }
                    clinicalReports {
                      id title trialPhase source clinicalStage url
                      phaseFromSource type trialOverallStatus year
                      qualityControls hasExpertReview
                    }
                  }
                }
                """
            )
        if "drug.warning" in requested:
            fields.append(
                """
                drugWarnings {
                  id chemblIds toxicityClass country description
                  references { id source url }
                  warningType efoTerm efoId efoIdForWarningClass year
                }
                """
            )
        include_adverse_events = "drug.adverse_event" in requested
        if include_adverse_events:
            fields.append(
                """
                adverseEvents(page: {index: 0, size: $pageSize}) {
                  count criticalValue
                  rows { name meddraCode count logLR }
                }
                """
            )
        if "drug.pharmacogenomics" in requested:
            fields.append(
                """
                pharmacogenomics {
                  datasourceId datatypeId evidenceLevel genotype
                  genotypeAnnotationText genotypeId
                  haplotypeFromSourceId haplotypeId literature pgxCategory
                  phenotypeFromSourceId phenotypeText studyId
                  variantRsId variantId variantFunctionalConsequenceId
                  targetFromSourceId isDirectTarget
                  variantAnnotation {
                    literature effectDescription effectType
                    baseAlleleOrGenotype comparisonAlleleOrGenotype
                    directionality effect entity
                  }
                  target { id approvedSymbol approvedName }
                  drugs { drugId drugFromSource drug { id name } }
                }
                """
            )
        declarations = "$chemblId: String!"
        if include_adverse_events:
            declarations += ", $pageSize: Int!"
        query = f"""
            query GWASMindDrugProfile({declarations}) {{
              drug(chemblId: $chemblId) {{
                {"".join(fields)}
              }}
            }}
        """
        key = sha256_json(
            {
                "drug_ids": selected_ids,
                "fields": tuple(sorted(requested)),
                "adverse_page_size": page_size if include_adverse_events else None,
            }
        )
        if key in self._drug_cache:
            self._drug_cache.move_to_end(key)
            cached, query_truncated = self._drug_cache[key]
            return cached, truncated or query_truncated

        variables_list: list[dict[str, Any]] = []
        for drug_id in selected_ids:
            variables: dict[str, Any] = {"chemblId": drug_id}
            if include_adverse_events:
                variables["pageSize"] = page_size
            variables_list.append(variables)
        values = await self.provider._batch_query(
            query_string=query,
            variables_list=variables_list,
            key_field="chemblId",
            root_field="drug",
        )
        profiles: list[dict[str, Any]] = []
        for drug_id, drug in zip(selected_ids, values, strict=True):
            if drug is None:
                continue
            if not isinstance(drug, Mapping):
                raise ValueError("drug must be an object")
            profiles.append({"id": drug.get("id") or drug_id, **dict(drug)})
        result = (tuple(profiles), len(selected_ids) < len(drug_ids))
        self._drug_cache[key] = result
        self._drug_cache.move_to_end(key)
        while len(self._drug_cache) > 4:
            self._drug_cache.popitem(last=False)
        return result[0], truncated or result[1]

    def _live_provenance(self) -> dict[str, str]:
        return {
            "source": _provider_source(self.provider),
            "data_release": self.schema.data_version if self.schema else "live",
        }

    async def _locus_base(self, arguments: Mapping[str, Any]) -> _LiveFetchBundle:
        return await self._fetch_bundle(
            arguments, {EvidenceKind.GWAS, EvidenceKind.QTL}, target_mode=False
        )

    async def _kind_result(
        self, arguments: Mapping[str, Any], kind: EvidenceKind, module: str
    ) -> ModuleResult:
        module_kinds = {
            "locus.qtl": EvidenceKind.QTL,
            "locus.colocalisation": EvidenceKind.COLOCALIZATION,
            "locus.enhancer_to_gene": EvidenceKind.REGULATORY,
            "locus.l2g": EvidenceKind.L2G,
        }
        requested_modules = arguments.get("modules", ())
        if kind is EvidenceKind.REGULATORY:
            # E2G is a separate, independently paginated Variant connection.
            # Keeping it out of the credible-set bundle prevents an E2G schema,
            # transport, or pagination failure from erasing valid L2G/QTL/coloc
            # evidence (and vice versa).
            requested_kinds = {EvidenceKind.REGULATORY}
        else:
            requested_kinds = {
                module_kinds[str(name)]
                for name in requested_modules
                if str(name) in module_kinds
                and module_kinds[str(name)] is not EvidenceKind.REGULATORY
            }
            requested_kinds.add(kind)
        bundle = await self._fetch_bundle(
            arguments, requested_kinds, target_mode=False
        )
        records = tuple(
            record
            for record in bundle.records
            if record.kind is kind
        )
        return _with_live_coverage(
            _records_result(
                module,
                records,
                relation=(
                    "variant_overlaps_regulatory_interval_linked_to_target"
                    if kind is EvidenceKind.REGULATORY
                    else module.replace(".", "_")
                ),
            ),
            bundle,
        )

    async def _target_kind_result(
        self, arguments: Mapping[str, Any], kind: EvidenceKind, module: str
    ) -> ModuleResult:
        bundle = await self._fetch_bundle(arguments, {kind}, target_mode=True)
        records = bundle.records
        if module == "target.pathway":
            records = tuple(record for record in records if record.payload.get("pathways"))
        elif module == "target.expression":
            records = tuple(
                record for record in records if record.tissue or record.payload.get("tissues")
                or (record.payload.get("expressionCoverage") or {}).get("complete") is False
            )
        elif module == "target.function":
            records = tuple(
                record
                for record in records
                if record.payload.get("functionDescriptions")
                or record.payload.get("approvedName")
                or record.payload.get("biotype")
            )
        return _with_live_coverage(
            _records_result(module, records, relation=module.replace(".", "_")), bundle
        )

    async def _fetch(
        self,
        arguments: Mapping[str, Any],
        kinds: set[EvidenceKind],
        *,
        target_mode: bool,
    ) -> tuple[EvidenceRecord, ...]:
        bundle = await self._fetch_bundle(
            arguments,
            kinds,
            target_mode=target_mode,
        )
        return bundle.records

    async def _fetch_bundle(
        self,
        arguments: Mapping[str, Any],
        kinds: set[EvidenceKind],
        *,
        target_mode: bool,
    ) -> _LiveFetchBundle:
        from ..molecular_context import MOLECULAR_RETRIEVAL_SCOPE
        molecular_scope = MOLECULAR_RETRIEVAL_SCOPE.get()
        variant_ids = tuple(str(item) for item in arguments.get("variant_ids", ()))
        target_ids = tuple(str(item) for item in arguments.get("target_ids", ()))
        requested_traits = arguments.get('trait_ids', ())
        if (not isinstance(requested_traits, (list, tuple)) or len(requested_traits) > 100
                or any(not isinstance(item, str) for item in requested_traits)):
            raise ValueError('trait_ids must be an array of at most 100 strings')
        trait_ids = tuple(sorted(set(requested_traits)))
        candidate_gene_ids = tuple(str(item) for item in arguments.get("candidate_gene_ids", ()))
        qtl_types = requested_qtl_study_types(arguments.get("qtl_types", ()))
        only_candidate_genes = arguments.get("only_candidate_genes", False)
        if not isinstance(only_candidate_genes, bool):
            raise ValueError("only_candidate_genes must be boolean")
        if target_mode and not target_ids:
            raise ValueError("target_ids are required")
        if not target_mode and not variant_ids:
            raise ValueError("variant_ids are required")
        if only_candidate_genes and not candidate_gene_ids:
            raise ValueError("candidate_gene_ids are required when only_candidate_genes is true")
        key = sha256_json(
            {
                "variant_ids": variant_ids,
                "target_ids": target_ids,
                "candidate_gene_ids": candidate_gene_ids,
                "only_candidate_genes": only_candidate_genes,
                "trait_ids": trait_ids,
                "study_ids": arguments.get("study_ids", ()),
                "qtl_types": qtl_types,
                "kinds": sorted(item.value for item in kinds),
                "molecular_scope": molecular_scope.to_dict() if molecular_scope is not None else None,
            }
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        query = LocusQuery(
            locus_id="live-semantic-mcp-query",
            trait="not-used-for-retrieval",
            trait_ids=trait_ids,
            variants=tuple(VariantObservation(item, p_value=1e-9) for item in variant_ids)
            or (VariantObservation("1_1_A_G", p_value=1e-9),),
            candidate_genes=(target_ids if target_mode else candidate_gene_ids),
            relevant_tissues=tuple(
                str(item) for item in arguments.get("relevant_biosample_ids", ())
            ),
            study_ids=tuple(str(item) for item in arguments.get("study_ids", ())),
            qtl_types=qtl_types,
        )
        fetch_with_coverage = getattr(self.provider, "fetch_with_coverage", None)
        truncated_kinds: frozenset[EvidenceKind] = frozenset()
        if callable(fetch_with_coverage):
            provider_result = await fetch_with_coverage(query, kinds)
            value = tuple(provider_result.records)
            coverage_warnings = tuple(provider_result.warnings)
            credible_sets_truncated = bool(provider_result.truncated)
            if hasattr(provider_result, "connection_coverage"):
                truncated_kinds = frozenset(
                    item.kind for item in provider_result.connection_coverage if item.truncated
                )
                credible_sets_truncated = any(
                    item.truncated
                    for item in (*provider_result.credible_set_coverage, *provider_result.enhancer_coverage)
                )
        else:  # pragma: no cover - compatibility for a provider-like test double
            value = tuple(await self.provider.fetch(query, kinds))
            coverage_warnings = ()
            credible_sets_truncated = False
        if only_candidate_genes:
            candidates = set(candidate_gene_ids)
            value = tuple(
                record
                for record in value
                if candidates.intersection(
                    {record.gene_id, record.payload.get("approvedSymbol")} - {None}
                )
            )
        value = tuple(
            record
            for record in value
            if record.kind is not EvidenceKind.QTL
            or (
                (qtl_type := qtl_study_type_from_payload(record.payload)) is not None
                and (not qtl_types or qtl_type in qtl_types)
            )
        )
        if self.schema is not None:
            value = tuple(
                replace(record, data_release=self.schema.data_version)
                if record.data_release.strip().casefold() == "live"
                else record
                for record in value
            )
        bundle = _LiveFetchBundle(
            records=value,
            warnings=coverage_warnings,
            credible_sets_truncated=credible_sets_truncated,
            truncated_kinds=truncated_kinds,
        )
        self._cache[key] = bundle
        self._cache.move_to_end(key)
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return bundle


def create_live_semantic_service(
    *,
    release: str = "live",
    provider: OfficialOpenTargetsMCPProvider | None = None,
    policy_sha256: str = "unknown",
    blocked_modules: frozenset[str] = frozenset(),
    region_gene_lookup: RegionGeneLookup | None = None,
    enable_region_gene_fallback: bool = True,
    publication_enricher: EuropePMCPublicationClient | None = None,
) -> SemanticToolService:
    selected = provider or OfficialOpenTargetsMCPProvider(release=release)
    lookup = region_gene_lookup
    if lookup is None and enable_region_gene_fallback:
        lookup = EnsemblRegionGeneLookup(flank_bp=parameter("analysis.local_gene_window_bp"))
    return LiveSemanticModules(
        selected,
        region_gene_lookup=lookup,
        publication_enricher=(
            publication_enricher or EuropePMCPublicationClient()
        ),
    ).service(
        policy_sha256=policy_sha256,
        blocked_modules=blocked_modules,
    )


async def create_discovered_live_semantic_service(
    *,
    release: str = "live",
    provider: OfficialOpenTargetsMCPProvider | None = None,
    policy_sha256: str = "unknown",
    blocked_modules: frozenset[str] = frozenset(),
    region_gene_lookup: RegionGeneLookup | None = None,
    enable_region_gene_fallback: bool = True,
    publication_enricher: EuropePMCPublicationClient | None = None,
) -> SemanticToolService:
    selected = provider or OfficialOpenTargetsMCPProvider(release=release)
    snapshot = await OfficialSchemaNegotiator(selected).inspect()
    lookup = region_gene_lookup
    if lookup is None and enable_region_gene_fallback:
        lookup = EnsemblRegionGeneLookup(flank_bp=parameter("analysis.local_gene_window_bp"))
    return LiveSemanticModules(
        selected,
        schema=snapshot,
        region_gene_lookup=lookup,
        publication_enricher=(
            publication_enricher or EuropePMCPublicationClient()
        ),
    ).service(
        policy_sha256=policy_sha256,
        blocked_modules=blocked_modules,
    )


def _credible_set_membership_result(
    records: tuple[EvidenceRecord, ...],
) -> ModuleResult:
    values: list[dict[str, Any]] = []
    for record in records:
        member = _matched_locus_member(record)
        query_variant_id = _required_record_text(
            record.payload.get("queryVariantId") or record.variant_id,
            "credible-set query variant",
        )
        credible_set_id = _required_record_text(
            record.credible_set_id or record.payload.get("studyLocusId"),
            "credible-set ID",
        )
        values.append(
            {
                "module": "locus.credible_set",
                "dataset": record.dataset,
                "relation": "credible_set_membership",
                "subject": {"type": "variant", "id": query_variant_id},
                "object": {"type": "credible_set", "id": credible_set_id},
                "measurements": {
                    "member_posterior_probability": member.get("posteriorProbability"),
                    "p_value_mantissa": record.payload.get("pValueMantissa"),
                    "p_value_exponent": record.payload.get("pValueExponent"),
                },
                "context": {
                    "lead_variant_id": record.payload.get("leadVariantId"),
                    "query_variant_role": record.payload.get("queryVariantRole"),
                    "study_id": record.study_id,
                    "study_type": record.payload.get("studyType"),
                },
                "provenance": {
                    "source": record.source,
                    "data_release": record.data_release,
                    "evidence_id": record.evidence_id,
                },
                "upstream": {"raw": record.payload},
                "derived": {},
            }
        )
    return ModuleResult(
        "locus.credible_set",
        ModuleStatus.SUCCESS if values else ModuleStatus.NO_DATA,
        records=tuple(values),
    )


def _fine_mapping_result(records: tuple[EvidenceRecord, ...]) -> ModuleResult:
    values: list[dict[str, Any]] = []
    for record in records:
        member = _matched_locus_member(record)
        credible_set_id = _required_record_text(
            record.credible_set_id or record.payload.get("studyLocusId"),
            "credible-set ID",
        )
        values.append(
            {
                "module": "locus.fine_mapping",
                "dataset": record.dataset,
                "relation": "fine_mapping_result",
                "subject": {"type": "credible_set", "id": credible_set_id},
                "object": None,
                "measurements": {
                    "posterior_probability": member.get("posteriorProbability"),
                    "log_bayes_factor": member.get("logBF"),
                },
                "context": {
                    "fine_mapping_method": record.payload.get("finemappingMethod"),
                    "fine_mapping_confidence_label": record.payload.get("confidence"),
                    "study_id": record.study_id,
                    "study_type": record.payload.get("studyType"),
                },
                "provenance": {
                    "source": record.source,
                    "data_release": record.data_release,
                    "evidence_id": record.evidence_id,
                },
                "upstream": {
                    "raw": {
                        key: record.payload.get(key)
                        for key in (
                            "finemappingMethod",
                            "confidence",
                            "qualityControls",
                            "matchedLocusMember",
                            "pValueMantissa",
                            "pValueExponent",
                            "beta",
                        )
                    }
                },
                "derived": {},
            }
        )
    return ModuleResult(
        "locus.fine_mapping",
        ModuleStatus.SUCCESS if values else ModuleStatus.NO_DATA,
        records=tuple(values),
    )


def _region_gene_summary_record(
    *,
    region: GRCh38Region | None,
    status: str,
    source: str,
    data_release: str,
    annotated_gene_count: int | None = None,
    candidate_gene_ids: tuple[str, ...] = (),
    query_start: int | None = None,
    query_end: int | None = None,
    flank_bp: int = parameter("analysis.local_gene_window_bp"),
    truncated: bool = False,
    candidate_list_truncated: bool = False,
    dataset: str = "ensembl_region_gene_context",
    count_is_lower_bound: bool = False,
) -> dict[str, Any]:
    region_id = (
        f"{region.genome_build}:chr{region.chromosome}:{region.start}-{region.end}"
        if region is not None
        else "GRCh38:unresolved"
    )
    return {
        "module": "locus.variant_consequence",
        "dataset": dataset,
        "relation": "regional_gene_context_summary",
        "subject": {"type": "region", "id": region_id},
        "object": None,
        "measurements": {
            "annotated_gene_count": annotated_gene_count,
            "returned_candidate_count": len(candidate_gene_ids),
        },
        "context": {
            **(region.to_dict() if region is not None else {}),
            "query_start": query_start,
            "query_end": query_end,
            "flank_bp": flank_bp,
            "candidate_gene_ids": list(candidate_gene_ids),
        },
        "provenance": {
            "source": source,
            "data_release": data_release,
            "endpoint": ENSEMBL_REST_ENDPOINT if source == ENSEMBL_REST_SOURCE else None,
        },
        "upstream": {"raw": {}},
        "derived": {
            "region_gene_context": True,
            "hard_gate_input": True,
            "status": status,
            "truncated": truncated,
            "candidate_list_truncated": candidate_list_truncated,
            "count_is_lower_bound": count_is_lower_bound,
            "unranked": True,
            "not_direct_locus_to_gene_evidence": True,
        },
    }


def _region_gene_candidate_record(
    region: GRCh38Region,
    candidate: RegionGeneCandidate,
    *,
    query_start: int,
    query_end: int,
    flank_bp: int,
    source: str,
    data_release: str,
) -> dict[str, Any]:
    return {
        "module": "locus.variant_consequence",
        "dataset": "ensembl_region_gene_context",
        "relation": candidate.relation,
        "subject": {
            "type": "region",
            "id": (
                f"{region.genome_build}:chr{region.chromosome}:"
                f"{region.start}-{region.end}"
            ),
        },
        "object": {
            "type": "target",
            "id": candidate.gene_id,
            "label": candidate.symbol,
        },
        "measurements": {"distance_to_region": candidate.distance_to_region},
        "context": {
            **region.to_dict(),
            "query_start": query_start,
            "query_end": query_end,
            "flank_bp": flank_bp,
            "gene_start": candidate.start,
            "gene_end": candidate.end,
            "strand": candidate.strand,
            "biotype": candidate.biotype,
        },
        "provenance": {
            "source": source,
            "data_release": data_release,
            "endpoint": ENSEMBL_REST_ENDPOINT if source == ENSEMBL_REST_SOURCE else None,
        },
        "upstream": {"raw": dict(candidate.raw)},
        "derived": {
            "candidate_basis": candidate.relation,
            "region_gene_context": True,
            "unranked": True,
            "not_l2g_or_qtl": True,
            "not_direct_locus_to_gene_evidence": True,
        },
    }


def _matched_locus_member(record: EvidenceRecord) -> Mapping[str, Any]:
    member = record.payload.get("matchedLocusMember")
    if not isinstance(member, Mapping):
        raise ValueError("credible-set evidence omitted matchedLocusMember")
    variant_id = str(member.get("variantId") or "").strip()
    if not variant_id or variant_id != str(record.variant_id or "").strip():
        raise ValueError("credible-set member does not match the query variant")
    return member


def _required_record_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _function_publication_records(
    records: tuple[EvidenceRecord, ...],
) -> tuple[EvidenceRecord, ...]:
    """Retain the real PMIDs explicitly cited by returned target function descriptions."""

    result: dict[tuple[str | None, str], EvidenceRecord] = {}
    for record in records:
        descriptions = record.payload.get("functionDescriptions") or ()
        if isinstance(descriptions, str):
            descriptions = (descriptions,)
        if not isinstance(descriptions, (list, tuple)):
            continue
        for description in descriptions:
            text = description if isinstance(description, str) else ""
            for pmid in re.findall(r"\bPubMed\s*:\s*([1-9]\d{0,11})\b", text, re.IGNORECASE):
                key = (record.gene_id, pmid)
                result.setdefault(key, EvidenceRecord(
                    evidence_id="function-publication:" + sha256_json(key)[:24],
                    kind=EvidenceKind.LITERATURE,
                    dataset="function_publication",
                    source=record.source,
                    data_release=record.data_release,
                    gene_id=record.gene_id,
                    pmid=pmid,
                    payload={
                        "pmid_origin": "function_description",
                        "source_evidence_id": record.evidence_id,
                    },
                ))
    return tuple(result.values())


def _publication_enrichment_pmids(
    records: tuple[EvidenceRecord, ...],
    *,
    limit: int,
) -> tuple[str, ...]:
    """Choose a candidate-balanced, recent PMID set for abstract enrichment.

    Target literature can contain hundreds of occurrence rows. Europe PMC
    enrichment is intentionally bounded, so taking the provider's first rows
    would spend the entire budget on the first few candidate genes. Round-robin
    selection gives each candidate a chance to contribute an abstract while the
    complete PMID ledger remains untouched.
    """

    if limit < 1:
        raise ValueError("publication enrichment limit must be positive")
    by_gene: OrderedDict[str, dict[str, EvidenceRecord]] = OrderedDict()
    all_records: dict[str, EvidenceRecord] = {}
    for record in records:
        pmid = normalize_pmid(record.pmid)
        if pmid is None:
            continue
        all_records.setdefault(pmid, record)
        gene_id = str(record.gene_id or "unassigned")
        by_gene.setdefault(gene_id, {}).setdefault(pmid, record)

    def priority(item: tuple[str, EvidenceRecord]) -> tuple[int, int]:
        pmid, record = item
        published = record.evidence_date.toordinal() if record.evidence_date else 0
        return (-published, -int(pmid))

    ordered_by_gene = {
        gene_id: [pmid for pmid, _record in sorted(values.items(), key=priority)]
        for gene_id, values in by_gene.items()
    }
    selected: list[str] = []
    selected_set: set[str] = set()
    max_depth = max((len(values) for values in ordered_by_gene.values()), default=0)
    for depth in range(max_depth):
        for values in ordered_by_gene.values():
            if depth >= len(values):
                continue
            pmid = values[depth]
            if pmid in selected_set:
                continue
            selected.append(pmid)
            selected_set.add(pmid)
            if len(selected) >= limit:
                return tuple(selected)

    for pmid, _record in sorted(all_records.items(), key=priority):
        if pmid in selected_set:
            continue
        selected.append(pmid)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _records_result(
    module: str,
    records: tuple[EvidenceRecord, ...],
    *,
    relation: str,
) -> ModuleResult:
    values = tuple(
        {
            "module": module,
            "dataset": record.dataset,
            "relation": relation,
            "subject": {
                "type": "variant" if record.variant_id else "target",
                "id": record.variant_id or record.gene_id,
            },
            "object": (
                {
                    "type": "target",
                    "id": record.gene_id,
                    "label": record.payload.get("approvedSymbol")
                    or record.payload.get("qtlGeneSymbol"),
                }
                if record.variant_id and record.gene_id
                else None
            ),
            "measurements": _semantic_measurements(record),
            "context": {
                "study_id": record.study_id,
                "study_locus_id": record.credible_set_id,
                "biosample": record.tissue,
                "pmid": record.pmid,
                "publication_date": (
                    record.evidence_date.isoformat() if record.evidence_date is not None else None
                ),
                "study_type": qtl_study_type_from_payload(record.payload),
                **({"target_trait_relation": record.payload.get("targetTraitRelation"),
                    "detail_retrieval_state": record.payload.get("rightDetailRetrievalState"),
                    "rank_scope": "target_trait" if record.payload.get("targetTraitRelation") == "exact_label"
                    else "unresolved" if record.payload.get("targetTraitRelation") == "unresolved" else "context_only"}
                   if "targetTraitRelation" in record.payload else {}),
                **_qtl_member_context(record),
                **_regulatory_context(record),
            },
            "provenance": {
                "source": record.source,
                "data_release": record.data_release,
                "evidence_id": record.evidence_id,
            },
            "upstream": {"raw": record.payload},
            "derived": _regulatory_derived(record),
        }
        for record in records
    )
    return ModuleResult(
        module,
        ModuleStatus.SUCCESS if values else ModuleStatus.NO_DATA,
        records=values,
    )


def _with_live_coverage(
    result: ModuleResult,
    bundle: _LiveFetchBundle,
) -> ModuleResult:
    """Attach bounded credible-set coverage to only the dependent module."""

    module_kind = {
        "locus.qtl": EvidenceKind.QTL,
        "locus.colocalisation": EvidenceKind.COLOCALIZATION,
        "locus.l2g": EvidenceKind.L2G,
        "evidence.europepmc": EvidenceKind.LITERATURE,
    }.get(result.module)
    truncated = bundle.credible_sets_truncated or module_kind in bundle.truncated_kinds
    if not truncated and not bundle.warnings:
        return result
    return replace(
        result,
        truncated=result.truncated or truncated,
        warnings=tuple(dict.fromkeys((*result.warnings, *bundle.warnings))),
    )


def _semantic_measurements(record: EvidenceRecord) -> dict[str, Any]:
    values = {
        key: record.payload.get(key)
        for key in (
            "posteriorProbability",
            "h3",
            "h4",
            "clpp",
            "betaRatioSignAverage",
            "l2g_score",
            "score",
        )
        if key in record.payload
    }
    if record.kind is EvidenceKind.QTL:
        member = _matched_locus_member(record)
        values.update(
            {
                key: member.get(key)
                for key in (
                    "posteriorProbability",
                    "pValueMantissa",
                    "pValueExponent",
                    "beta",
                    "standardError",
                )
                if member.get(key) is not None
            }
        )
    if record.kind is EvidenceKind.REGULATORY:
        values["resource_scores"] = record.payload.get("resourceScore")
    return values


def _regulatory_context(record: EvidenceRecord) -> dict[str, Any]:
    if record.kind is not EvidenceKind.REGULATORY:
        return {}
    payload = record.payload
    return {
        "overlap_variant_id": payload.get("overlapVariantId") or record.variant_id,
        "overlap_variant_role": payload.get("overlapVariantRole"),
        "overlap_chromosome": payload.get("overlapChromosome"),
        "overlap_position": payload.get("overlapPosition"),
        "exact_variant_overlap": payload.get("exactVariantOverlap"),
        "coverage_scope": payload.get("coverageScope"),
        "regulatory_element_key": payload.get("regulatoryElementKey"),
        "regulatory_element_key_source": payload.get("regulatoryElementKeySource"),
        "regulatory_chromosome": payload.get("chromosome"),
        "regulatory_start": payload.get("start"),
        "regulatory_end": payload.get("end"),
        "interval_type": payload.get("intervalType"),
        "distance_to_tss": payload.get("distanceToTss"),
        "datasource_id": payload.get("datasourceId"),
        "biosample_id": payload.get("biosampleFromSourceId"),
        "enhancer_study_id": payload.get("enhancerStudyId"),
    }


def _regulatory_derived(record: EvidenceRecord) -> dict[str, Any]:
    if record.kind is not EvidenceKind.REGULATORY:
        return {}
    payload = record.payload
    return {
        "exact_variant_interval_overlap": payload.get("exactVariantOverlap") is True,
        "coverage_scope": payload.get("coverageScope"),
        "regulatory_element_key_is_derived": (
            payload.get("regulatoryElementKeySource") == "derived_coordinate_key"
        ),
    }


def _qtl_member_context(record: EvidenceRecord) -> dict[str, Any]:
    if record.kind is not EvidenceKind.QTL:
        return {}
    member = _matched_locus_member(record)
    variant_id = _required_record_text(member.get("variantId"), "QTL member variant")
    parts = variant_id.split("_")
    if len(parts) != 4:
        raise ValueError("QTL member variant must be a canonical Open Targets ID")
    try:
        position = int(parts[1])
    except ValueError as exc:
        raise ValueError("QTL member variant position must be an integer") from exc
    if not parts[0] or position < 1:
        raise ValueError("QTL member variant position is invalid")
    return {
        "qtl_member_variant_id": variant_id,
        "qtl_member_chromosome": parts[0],
        "qtl_member_position": position,
    }


def _canonical_entity_id(identifier: str, entity_type: str) -> bool:
    if entity_type == "target":
        return bool(_TARGET_ID.fullmatch(identifier))
    if entity_type == "variant":
        return bool(_VARIANT_ID.fullmatch(identifier))
    if entity_type == "disease":
        return bool(identifier)
    if entity_type == "drug":
        return bool(_DRUG_ID.fullmatch(identifier))
    return False


def _ordered_drug_ids(values: Any) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, (list, tuple)):
        return ()
    for value in values:
        identifier = str(value).strip().upper()
        if not _DRUG_ID.fullmatch(identifier) or identifier in seen:
            continue
        seen.add(identifier)
        output.append(identifier)
    return tuple(output)


def _drug_subject(drug: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "drug",
        "id": drug.get("id"),
        "label": drug.get("name"),
    }


def _present_values(**values: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def _has_content(value: Mapping[str, Any]) -> bool:
    return bool(_present_values(**dict(value)))


def _mapping_list(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return tuple(item for item in value if isinstance(item, Mapping))


def _connection_rows(value: Any, label: str) -> tuple[tuple[Mapping[str, Any], ...], int]:
    if value is None:
        return (), 0
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    rows = _mapping_list(value.get("rows"), f"{label}.rows")
    count_value = value.get("count")
    count = len(rows) if count_value is None else int(count_value)
    return rows, count


def _deduplicate(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (str(value.get("type")), str(value.get("id")))
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _unwrap_optional_success_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if "status" not in value:
        return value
    status = value.get("status")
    if not isinstance(status, str) or status.casefold() != "success":
        raise ValueError(f"{label} returned non-success status {status!r}")
    data = value.get("data")
    if not isinstance(data, Mapping):
        raise ValueError(f"successful {label} response omitted its data object")
    return data


def _mapping_value_case_insensitive(value: Mapping[str, Any], key: str) -> Any | None:
    if key in value:
        return value[key]
    folded = key.casefold()
    for candidate, item in value.items():
        if str(candidate).casefold() == folded:
            return item
    return None


def _hit_matches_query(hit: Mapping[str, Any], query: str, identifier: str) -> bool:
    expected = query.casefold()
    if identifier.casefold() == expected:
        return True
    for key in (
        "query",
        "label",
        "name",
        "approvedSymbol",
        "symbol",
        "rsId",
        "rsIds",
    ):
        value = hit.get(key)
        candidates = value if isinstance(value, (list, tuple)) else (value,)
        if any(
            candidate is not None and str(candidate).strip().casefold() == expected
            for candidate in candidates
        ):
            return True
    return False


def _disease_hit_matches_query(hit: Mapping[str, Any], query: str, identifier: str) -> bool:
    """Require an ontology identity/label or explicitly supplied exact synonym.

    An echoed search query and broad/related ontology synonyms are not proof
    that the returned disease denotes the user's phenotype.
    """
    expected = query.strip().casefold()
    values = [identifier, hit.get("label"), hit.get("name")]
    for key in ("exactSynonyms", "hasExactSynonym"):
        value = hit.get(key, ())
        values.extend(value if isinstance(value, (list, tuple)) else (value,))
    synonyms = hit.get("synonyms")
    for synonym in synonyms if isinstance(synonyms, (list, tuple)) else ():
        if isinstance(synonym, Mapping) and synonym.get("relation") == "hasExactSynonym":
            terms = synonym.get("terms")
            values.extend(terms if isinstance(terms, (list, tuple)) else (terms,))
    return any(isinstance(value, str) and value.strip().casefold() == expected for value in values)


def _association_record(
    module: str,
    row: Mapping[str, Any],
    *,
    source: str,
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
    disease = row.get("disease") if isinstance(row.get("disease"), Mapping) else {}
    return {
        "module": module,
        "dataset": module.replace("association.", "association_by_"),
        "relation": "target_disease_association",
        "subject": {
            "type": "target",
            "id": target.get("id"),
            "label": target.get("approvedSymbol"),
        },
        "object": {
            "type": "disease",
            "id": disease.get("id"),
            "label": disease.get("name"),
        },
        "measurements": dict(measurements),
        "context": {
            "association_scope": row.get("association_scope"),
            "page_index": row.get("page_index"),
            "total_associated_targets": row.get("total_associated_targets"),
            "total_associated_diseases": row.get("total_associated_diseases"),
        },
        "provenance": {"source": source},
        "upstream": {"raw": dict(row)},
        "derived": {},
    }
