from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..configuration import parameter
from ..models import QTL_STUDY_TYPES
from .contracts import SemanticTool, ToolSpec

CATALOG_RELEASE = "26.06"
CATALOG_SOURCE = (
    "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.06/output/"
)


class DatasetState(StrEnum):
    READY = "ready"
    PRESENT_UNIMPLEMENTED = "present_unimplemented"
    EMPTY = "empty"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    modules: tuple[str, ...]
    tools: tuple[SemanticTool, ...]
    local_optional: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset name is required")
        if not self.modules or len(self.modules) != len(set(self.modules)):
            raise ValueError(f"dataset {self.name} needs unique modules")
        if not self.tools or len(self.tools) != len(set(self.tools)):
            raise ValueError(f"dataset {self.name} needs unique tools")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modules": list(self.modules),
            "tools": [item.value for item in self.tools],
            "local_optional": self.local_optional,
        }


@dataclass(frozen=True, slots=True)
class LocalDatasetAlias:
    source_name: str
    canonical_name: str
    schema_compatible: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "canonical_name": self.canonical_name,
            "schema_compatible": self.schema_compatible,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class DatasetInspection:
    canonical_name: str | None
    source_names: tuple[str, ...]
    state: DatasetState
    parquet_files: int
    bytes: int
    local_optional: bool
    modules: tuple[str, ...] = ()
    tools: tuple[SemanticTool, ...] = ()
    note: str = ""

    @property
    def recognized(self) -> bool:
        return self.canonical_name is not None and self.state not in {
            DatasetState.MISSING,
            DatasetState.UNKNOWN,
        }

    @property
    def ready(self) -> bool:
        return self.state is DatasetState.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "source_names": list(self.source_names),
            "state": self.state.value,
            "parquet_files": self.parquet_files,
            "bytes": self.bytes,
            "local_optional": self.local_optional,
            "modules": list(self.modules),
            "tools": [item.value for item in self.tools],
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class FrozenDatasetDiscovery:
    root: str
    root_exists: bool
    catalog_release: str
    catalog_source: str
    datasets: tuple[DatasetInspection, ...]
    unknown: tuple[DatasetInspection, ...]

    @property
    def present(self) -> tuple[DatasetInspection, ...]:
        return tuple(item for item in self.datasets if item.recognized)

    @property
    def missing(self) -> tuple[DatasetInspection, ...]:
        return tuple(item for item in self.datasets if item.state is DatasetState.MISSING)

    @property
    def ready(self) -> tuple[DatasetInspection, ...]:
        return tuple(item for item in self.datasets if item.ready)

    @property
    def recognized_source_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for item in self.present for name in item.source_names)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "root_exists": self.root_exists,
            "catalog_release": self.catalog_release,
            "catalog_source": self.catalog_source,
            "summary": {
                "catalogued": len(self.datasets),
                "present": len(self.present),
                "ready": len(self.ready),
                "missing": len(self.missing),
                "unknown": len(self.unknown),
            },
            "datasets": [item.to_dict() for item in self.datasets],
            "unknown": [item.to_dict() for item in self.unknown],
        }


def _ds(
    name: str,
    modules: tuple[str, ...],
    tools: tuple[SemanticTool, ...],
    *,
    local_optional: bool = True,
) -> DatasetSpec:
    return DatasetSpec(name, modules, tools, local_optional)


C = SemanticTool

# Exact directory names from the Open Targets Platform 26.06 download index.
_DATASETS = (
    _ds("association_by_datasource_direct", ("association.datasource_direct",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("association_by_datasource_indirect", ("association.datasource_indirect",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("association_by_datatype_direct", ("association.datatype_direct",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("association_by_datatype_indirect", ("association.datatype_indirect",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("association_overall_direct", ("association.overall_direct",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("association_overall_indirect", ("association.overall_indirect",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("baseline_expression", ("target.expression", "gene_set.tissue_coverage"), (C.GET_TARGET_PROFILES, C.ANALYZE_GENE_SET)),
    _ds("biosample", ("entity.biosample", "context.tissue"), (C.RESOLVE_ENTITIES, C.GET_LOCUS_CONTEXT, C.GET_LOCUS_GENE_EVIDENCE, C.GET_TARGET_PROFILES)),
    _ds("clinical_indication", ("drug.indication", "drug.clinical_indication"), (C.GET_DRUG_EVIDENCE,)),
    _ds("clinical_report", ("drug.clinical_report",), (C.GET_DRUG_EVIDENCE,)),
    _ds("clinical_target", ("drug.clinical_target",), (C.GET_DRUG_EVIDENCE,)),
    _ds("colocalisation", ("locus.colocalisation",), (C.GET_LOCUS_GENE_EVIDENCE,)),
    _ds("credible_set", ("locus.credible_set", "locus.fine_mapping"), (C.GET_LOCUS_CONTEXT, C.GET_LOCUS_GENE_EVIDENCE), local_optional=False),
    _ds("disease", ("disease.core", "disease.ontology"), (C.RESOLVE_ENTITIES, C.GET_DISEASE_PROFILES, C.GET_TARGET_DISEASE_EVIDENCE, C.GET_DRUG_EVIDENCE)),
    _ds("disease_hpo", ("disease.hpo",), (C.GET_DISEASE_PROFILES,)),
    _ds("disease_phenotype", ("disease.phenotype",), (C.GET_DISEASE_PROFILES,)),
    _ds("drug_mechanism_of_action", ("drug.mechanism",), (C.GET_DRUG_EVIDENCE,)),
    _ds("drug_molecule", ("drug.core",), (C.RESOLVE_ENTITIES, C.GET_DRUG_EVIDENCE)),
    _ds("drug_warning", ("drug.warning",), (C.GET_DRUG_EVIDENCE,)),
    _ds("enhancer_to_gene", ("locus.enhancer_to_gene",), (C.GET_LOCUS_GENE_EVIDENCE,)),
    _ds("evidence_cancer_biomarkers", ("evidence.cancer_biomarkers",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_cancer_gene_census", ("evidence.cancer_gene_census",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_clingen", ("evidence.clingen",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_clinical_precedence", ("evidence.clinical_precedence",), (C.GET_TARGET_DISEASE_EVIDENCE, C.GET_DRUG_EVIDENCE)),
    _ds("evidence_crispr", ("evidence.crispr",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_crispr_screen", ("evidence.crispr_screen",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_europepmc", ("evidence.europepmc",), (C.GET_TARGET_DISEASE_EVIDENCE, C.GET_TARGET_PROFILES)),
    _ds("evidence_eva", ("evidence.eva",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_eva_somatic", ("evidence.eva_somatic",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_expression_atlas", ("evidence.expression_atlas",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_gene2phenotype", ("evidence.gene2phenotype",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_gene_burden", ("evidence.gene_burden",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_genomics_england", ("evidence.genomics_england",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_gwas_credible_sets", ("evidence.gwas_credible_sets",), (C.GET_TARGET_DISEASE_EVIDENCE, C.GET_LOCUS_GENE_EVIDENCE)),
    _ds("evidence_impc", ("evidence.impc",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_intogen", ("evidence.intogen",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_orphanet", ("evidence.orphanet",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_reactome", ("evidence.reactome",), (C.GET_TARGET_DISEASE_EVIDENCE,)),
    _ds("evidence_uniprot_literature", ("evidence.uniprot_literature",), (C.GET_TARGET_DISEASE_EVIDENCE, C.GET_TARGET_PROFILES)),
    _ds("evidence_uniprot_variants", ("evidence.uniprot_variants",), (C.GET_TARGET_DISEASE_EVIDENCE, C.GET_LOCUS_GENE_EVIDENCE)),
    _ds("go", ("target.go", "gene_set.go"), (C.RESOLVE_ENTITIES, C.GET_TARGET_PROFILES, C.ANALYZE_GENE_SET)),
    _ds("interaction", ("target.interaction",), (C.GET_TARGET_PROFILES,)),
    _ds("interaction_evidence", ("target.interaction_evidence",), (C.GET_TARGET_PROFILES,)),
    _ds("l2g_prediction", ("locus.l2g",), (C.GET_LOCUS_GENE_EVIDENCE,)),
    _ds("literature_entity_lut", ("literature.entity_lookup",), (C.RESOLVE_ENTITIES, C.GET_TARGET_PROFILES, C.GET_DISEASE_PROFILES)),
    _ds("literature_vector", ("literature.vector",), (C.GET_TARGET_PROFILES, C.GET_DISEASE_PROFILES)),
    _ds("mouse_phenotype", ("target.mouse_phenotype",), (C.GET_TARGET_PROFILES, C.GET_DISEASE_PROFILES)),
    _ds("openfda_significant_adverse_drug_reactions", ("drug.adverse_event",), (C.GET_DRUG_EVIDENCE,)),
    _ds("pharmacogenomics", ("drug.pharmacogenomics",), (C.GET_DRUG_EVIDENCE, C.GET_TARGET_PROFILES)),
    _ds("reactome", ("target.pathway", "gene_set.pathway"), (C.RESOLVE_ENTITIES, C.GET_TARGET_PROFILES, C.ANALYZE_GENE_SET)),
    _ds("so", ("locus.sequence_ontology",), (C.GET_LOCUS_CONTEXT, C.GET_LOCUS_GENE_EVIDENCE)),
    _ds("study", ("locus.study",), (C.RESOLVE_ENTITIES, C.GET_LOCUS_CONTEXT, C.GET_LOCUS_GENE_EVIDENCE), local_optional=False),
    _ds(
        "target",
        (
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
        ),
        (
            C.RESOLVE_ENTITIES,
            C.GET_TARGET_PROFILES,
            C.GET_LOCUS_GENE_EVIDENCE,
            C.GET_TARGET_DISEASE_EVIDENCE,
        ),
        local_optional=False,
    ),
    _ds("target_essentiality", ("target.essentiality",), (C.GET_TARGET_PROFILES,)),
    _ds("target_prioritisation", ("target.prioritisation",), (C.GET_TARGET_PROFILES,)),
    _ds("variant", ("locus.variant", "locus.variant_consequence"), (C.RESOLVE_ENTITIES, C.GET_LOCUS_CONTEXT, C.GET_LOCUS_GENE_EVIDENCE), local_optional=False),
)

DATASET_CATALOG: Mapping[str, DatasetSpec] = MappingProxyType(
    {item.name: item for item in _DATASETS}
)


LOCAL_DATASET_ALIASES: Mapping[str, LocalDatasetAlias] = MappingProxyType(
    {
        "colocalisation_coloc": LocalDatasetAlias(
            "colocalisation_coloc",
            "colocalisation",
            False,
            "Legacy pre-26.06 coloc-only directory; schema must be adapted explicitly.",
        ),
        "expression": LocalDatasetAlias(
            "expression",
            "baseline_expression",
            False,
            "Legacy aggregated expression directory; schema must be adapted explicitly.",
        ),
        "known_drug": LocalDatasetAlias(
            "known_drug",
            "clinical_target",
            False,
            "Legacy target-disease-drug aggregate; not schema-equivalent to clinical_target.",
        ),
    }
)


def _array_of_strings(*, max_items: int = 200, min_items: int = 1) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": min_items,
        "maxItems": max_items,
        "uniqueItems": True,
    }


def _qtl_type_array() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"enum": list(QTL_STUDY_TYPES)},
        "minItems": 0,
        "maxItems": len(QTL_STUDY_TYPES),
        "uniqueItems": True,
        "description": (
            "Optional exact native Open Targets QTL studyType filter; an empty "
            "array requests all supported molecular-QTL subtypes."
        ),
    }


def _object(
    properties: Mapping[str, Any],
    *,
    required: tuple[str, ...] = (),
    any_of: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
    }
    if required:
        value["required"] = list(required)
    if any_of:
        value["anyOf"] = [{"required": list(group)} for group in any_of]
    return value


_PAGE = {
    "page_size": {"type": "integer", "minimum": 1, "maximum": 500, "default": parameter("retrieval.semantic_page_size")},
    "cursor": {"type": ["string", "null"], "minLength": 1, "default": None},
}


TOOL_SPECS: Mapping[SemanticTool, ToolSpec] = MappingProxyType(
    {
        C.GET_CAPABILITIES: ToolSpec(
            C.GET_CAPABILITIES,
            "Report release-bound provider, dataset, module, schema, and limit capabilities.",
            _object({}),
            ("runtime.capabilities",),
            0,
            "Call once at startup; the controller consumes it before the agent runs.",
            agent_callable=False,
        ),
        C.RESOLVE_ENTITIES: ToolSpec(
            C.RESOLVE_ENTITIES,
            "Resolve names and aliases to unambiguous Open Targets entity identifiers.",
            _object(
                {
                    "queries": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 300,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text"],
                            "properties": {
                                "text": {"type": "string", "minLength": 1},
                                "types": {
                                    "type": "array",
                                    "items": {
                                        "enum": [
                                            "target",
                                            "variant",
                                            "disease",
                                            "drug",
                                            "study",
                                            "biosample",
                                            "pathway",
                                        ]
                                    },
                                    "uniqueItems": True,
                                },
                            },
                        },
                    },
                    "max_hits": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                required=("queries",),
            ),
            ("entity.resolve",),
            1,
            "Call only for non-canonical identifiers; never auto-select an ambiguous hit.",
        ),
        C.GET_LOCUS_CONTEXT: ToolSpec(
            C.GET_LOCUS_CONTEXT,
            "Retrieve variant, study, credible-set, and fine-mapping context for a locus.",
            _object(
                {
                    "variant_ids": _array_of_strings(max_items=100, min_items=0),
                    "study_ids": _array_of_strings(max_items=100, min_items=0),
                    "region": {
                        "type": ["object", "null"],
                        "additionalProperties": False,
                        "required": ["chromosome", "start", "end", "genome_build"],
                        "properties": {
                            "chromosome": {"type": "string", "minLength": 1},
                            "start": {"type": "integer", "minimum": 1},
                            "end": {"type": "integer", "minimum": 1},
                            "genome_build": {"enum": ["GRCh38"]},
                        },
                    },
                    "modules": _array_of_strings(max_items=8),
                    "qtl_types": _qtl_type_array(),
                    "include_credible_set_members": {"type": "boolean", "default": True},
                    **_PAGE,
                },
                required=("modules",),
                any_of=(("variant_ids",), ("study_ids",), ("region",)),
            ),
            ("locus.variant", "locus.study", "locus.credible_set", "locus.fine_mapping"),
            2,
            "Mandatory first evidence call for a validated locus.",
            required_predecessors=(C.RESOLVE_ENTITIES,),
        ),
        C.GET_LOCUS_GENE_EVIDENCE: ToolSpec(
            C.GET_LOCUS_GENE_EVIDENCE,
            "Retrieve typed variant-to-gene links without collapsing native metrics.",
            _object(
                {
                    "variant_ids": _array_of_strings(max_items=100, min_items=0),
                    "study_locus_ids": _array_of_strings(max_items=200, min_items=0),
                    "candidate_gene_ids": _array_of_strings(
                        max_items=500, min_items=0
                    ),
                    "modules": _array_of_strings(max_items=12),
                    "relevant_biosample_ids": _array_of_strings(max_items=100, min_items=0),
                    "qtl_types": _qtl_type_array(),
                    "only_candidate_genes": {"type": "boolean"},
                    "allow_region_fallback": {
                        "type": "boolean",
                        "default": False,
                    },
                    **_PAGE,
                },
                required=("candidate_gene_ids", "modules", "only_candidate_genes"),
                any_of=(("variant_ids",), ("study_locus_ids",)),
            ),
            ("locus.variant_consequence", "locus.qtl", "locus.colocalisation", "locus.enhancer_to_gene", "locus.l2g"),
            3,
            (
                "Mandatory after locus context; restrict benchmark queries to the "
                "fixed candidate universe or discover an exploratory universe from "
                "typed locus-to-gene evidence."
            ),
            required_predecessors=(C.GET_LOCUS_CONTEXT,),
        ),
        C.GET_TARGET_PROFILES: ToolSpec(
            C.GET_TARGET_PROFILES,
            "Retrieve target function, expression, interaction, safety, and tractability annotations.",
            _object(
                {
                    "target_ids": _array_of_strings(max_items=200),
                    "modules": _array_of_strings(max_items=32),
                    "publication_ids": _array_of_strings(max_items=100, min_items=0),
                    "trait_ids": _array_of_strings(max_items=100, min_items=0),
                    "relevant_biosample_ids": _array_of_strings(max_items=100, min_items=0),
                    **_PAGE,
                },
                required=("target_ids", "modules"),
            ),
            ("target.core", "target.function", "target.expression", "target.safety", "target.tractability"),
            4,
            "Call after gene-link retrieval when biological plausibility or translation is needed.",
            required_predecessors=(C.GET_LOCUS_GENE_EVIDENCE,),
        ),
        C.GET_TARGET_DISEASE_EVIDENCE: ToolSpec(
            C.GET_TARGET_DISEASE_EVIDENCE,
            "Retrieve direct or ontology-propagated target-disease associations and evidence.",
            _object(
                {
                    "target_ids": _array_of_strings(max_items=200),
                    "disease_ids": _array_of_strings(max_items=100, min_items=0),
                    "association_scope": {"enum": ["direct", "indirect", "both"], "default": "direct"},
                    "modules": _array_of_strings(max_items=8),
                    "datatype_ids": _array_of_strings(max_items=100, min_items=0),
                    "datasource_ids": _array_of_strings(max_items=100, min_items=0),
                    "min_score": {"type": "number", "minimum": 0, "maximum": 1, "default": 0},
                    **_PAGE,
                },
                required=("target_ids", "disease_ids", "modules"),
            ),
            ("association.overall", "association.datasource", "association.raw_evidence"),
            4,
            "Optional phenotype-alignment call; policy may block it in genetic benchmarks.",
            required_predecessors=(C.GET_LOCUS_GENE_EVIDENCE,),
        ),
        C.GET_DISEASE_PROFILES: ToolSpec(
            C.GET_DISEASE_PROFILES,
            "Retrieve disease ontology, HPO, phenotype, therapeutic-area, and literature context.",
            _object(
                {"disease_ids": _array_of_strings(max_items=100), "modules": _array_of_strings(max_items=16), **_PAGE},
                required=("disease_ids", "modules"),
            ),
            ("disease.core", "disease.ontology", "disease.hpo", "disease.phenotype"),
            4,
            "Call when trait-to-disease mapping or direct-versus-indirect scope needs clarification.",
            required_predecessors=(C.RESOLVE_ENTITIES,),
        ),
        C.GET_DRUG_EVIDENCE: ToolSpec(
            C.GET_DRUG_EVIDENCE,
            "Retrieve drug mechanisms, indications, clinical reports, warnings, and pharmacogenomics.",
            _object(
                {
                    "drug_ids": _array_of_strings(max_items=100, min_items=0),
                    "target_ids": _array_of_strings(max_items=200, min_items=0),
                    "disease_ids": _array_of_strings(max_items=100, min_items=0),
                    "modules": _array_of_strings(max_items=16),
                    **_PAGE,
                },
                required=("modules",),
                any_of=(("drug_ids",), ("target_ids",), ("disease_ids",)),
            ),
            (
                "drug.core",
                "drug.mechanism",
                "drug.indication",
                "drug.clinical_target",
                "drug.warning",
                "drug.adverse_event",
                "drug.pharmacogenomics",
            ),
            5,
            "Use only for translational questions after genetic prioritization, never as a substitute for it.",
            required_predecessors=(C.GET_LOCUS_GENE_EVIDENCE,),
        ),
        C.ANALYZE_GENE_SET: ToolSpec(
            C.ANALYZE_GENE_SET,
            "Run deterministic pathway, GO, tissue, or evidence-coverage analyses.",
            _object(
                {
                    "target_ids": _array_of_strings(max_items=500, min_items=3),
                    "analyses": _array_of_strings(max_items=8),
                    "background": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type"],
                        "properties": {
                            "type": {"enum": ["candidate_universe", "all_protein_coding", "explicit"]},
                            "target_ids": _array_of_strings(max_items=10000, min_items=0),
                        },
                    },
                    "multiple_testing": {"const": "BH"},
                    "min_set_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 3,
                    },
                },
                required=("target_ids", "analyses", "background", "multiple_testing"),
            ),
            ("gene_set.pathway", "gene_set.go", "gene_set.tissue_coverage", "gene_set.evidence_coverage"),
            5,
            "Optional for three or more genes; all statistics remain deterministic code.",
            required_predecessors=(C.GET_LOCUS_GENE_EVIDENCE,),
        ),
    }
)


def discover_frozen_datasets(
    root: str | Path,
    *,
    implemented_datasets: Iterable[str] = (),
) -> FrozenDatasetDiscovery:
    """Discover local data without equating directory presence with implementation.

    A dataset is ``ready`` only when a non-empty directory is present *and* its
    canonical name is supplied through ``implemented_datasets`` by an adapter.
    """

    root_path = Path(root).resolve()
    root_exists = root_path.is_dir()
    directories = (
        {item.name: item for item in root_path.iterdir() if item.is_dir()}
        if root_exists
        else {}
    )
    implemented = {
        _canonical_dataset_name(item) for item in implemented_datasets
    }
    by_canonical: dict[str, list[tuple[str, Path, LocalDatasetAlias | None]]] = {}
    unknown_directories: list[tuple[str, Path]] = []
    for source_name, path in sorted(directories.items()):
        if source_name in DATASET_CATALOG:
            by_canonical.setdefault(source_name, []).append((source_name, path, None))
            continue
        alias = LOCAL_DATASET_ALIASES.get(source_name)
        if alias is not None:
            by_canonical.setdefault(alias.canonical_name, []).append(
                (source_name, path, alias)
            )
        else:
            unknown_directories.append((source_name, path))

    inspections: list[DatasetInspection] = []
    for canonical_name, spec in DATASET_CATALOG.items():
        sources = by_canonical.get(canonical_name, [])
        if not sources:
            inspections.append(
                DatasetInspection(
                    canonical_name,
                    (),
                    DatasetState.MISSING,
                    0,
                    0,
                    spec.local_optional,
                    spec.modules,
                    spec.tools,
                    "optional local dataset is not installed"
                    if spec.local_optional
                    else "required core local dataset is not installed",
                )
            )
            continue
        files = [file for _, path, _ in sources for file in path.rglob("*.parquet")]
        aliases = [alias for _, _, alias in sources if alias is not None]
        if not files:
            state = DatasetState.EMPTY
            note = "dataset directory exists but contains no Parquet files"
        elif canonical_name in implemented:
            state = DatasetState.READY
            note = "present and explicitly registered by an adapter"
        else:
            state = DatasetState.PRESENT_UNIMPLEMENTED
            note = "present, but no adapter was registered; capability is not claimed"
        if aliases:
            alias_notes = "; ".join(alias.note for alias in aliases)
            note = f"{note}; {alias_notes}"
        inspections.append(
            DatasetInspection(
                canonical_name,
                tuple(sorted(source_name for source_name, _, _ in sources)),
                state,
                len(files),
                sum(file.stat().st_size for file in files),
                spec.local_optional,
                spec.modules,
                spec.tools,
                note,
            )
        )

    unknown = tuple(
        _unknown_inspection(source_name, path)
        for source_name, path in unknown_directories
    )
    return FrozenDatasetDiscovery(
        str(root_path),
        root_exists,
        CATALOG_RELEASE,
        CATALOG_SOURCE,
        tuple(inspections),
        unknown,
    )


def datasets_for_tool(tool: SemanticTool) -> tuple[DatasetSpec, ...]:
    return tuple(spec for spec in DATASET_CATALOG.values() if tool in spec.tools)


def _canonical_dataset_name(name: str) -> str:
    if name in DATASET_CATALOG:
        return name
    alias = LOCAL_DATASET_ALIASES.get(name)
    if alias is not None:
        return alias.canonical_name
    return name


def _unknown_inspection(source_name: str, path: Path) -> DatasetInspection:
    files = list(path.rglob("*.parquet"))
    return DatasetInspection(
        None,
        (source_name,),
        DatasetState.UNKNOWN,
        len(files),
        sum(file.stat().st_size for file in files),
        True,
        (),
        (),
        "directory is not in the pinned Open Targets catalog",
    )
