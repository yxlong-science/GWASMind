from __future__ import annotations

import re
from dataclasses import dataclass

from ..configuration import parameter
from ..models import EvidenceKind, LocusQuery
from .contracts import SemanticTool
from .normalization import evidence_kind_for_module

_TARGET_ID = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)
_VARIANT_ID = re.compile(
    r"^(?:chr)?(?:[0-9]+|X|Y|MT)(?:_|:)\d+(?:_|:)[ACGTN]+(?:_|:)[ACGTN]+$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SemanticPlanStep:
    stage: int
    tool: SemanticTool
    modules: tuple[str, ...]
    required: bool
    purpose: str
    input_source: str

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "tool": self.tool.value,
            "modules": list(self.modules),
            "required": self.required,
            "purpose": self.purpose,
            "input_source": self.input_source,
        }


@dataclass(frozen=True, slots=True)
class SemanticEvidencePlan:
    steps: tuple[SemanticPlanStep, ...]
    max_calls: int = parameter("retrieval.semantic_max_calls")
    max_follow_up_rounds: int = 2  # Compatibility metadata; executor does not consume this field.

    def __post_init__(self) -> None:
        if len(self.steps) > self.max_calls:
            raise ValueError("semantic plan exceeds the MCP call budget")
        if any(step.stage < 1 for step in self.steps):
            raise ValueError("semantic plan stages begin at one")

    def to_dict(self) -> dict:
        return {
            "max_calls": self.max_calls,
            "max_follow_up_rounds": self.max_follow_up_rounds,
            "steps": [step.to_dict() for step in self.steps],
        }


class SemanticEvidencePlanner:
    """Deterministic guardrails around the single agent's tool choices."""

    def plan(
        self,
        query: LocusQuery,
        *,
        ready_modules: frozenset[str],
        blocked_modules: frozenset[str] = frozenset(),
        requested_kinds: frozenset[EvidenceKind] | None = None,
    ) -> SemanticEvidencePlan:
        available = ready_modules.difference(blocked_modules)
        steps: list[SemanticPlanStep] = []
        needs_resolution = any(
            not _VARIANT_ID.fullmatch(item.variant_id) for item in query.variants
        ) or any(not _TARGET_ID.fullmatch(item) for item in query.candidate_genes)
        if needs_resolution and "entity.resolve" not in available:
            raise ValueError("required semantic entity resolver is unavailable")
        if needs_resolution and "entity.resolve" in available:
            steps.append(
                SemanticPlanStep(
                    1,
                    SemanticTool.RESOLVE_ENTITIES,
                    ("entity.resolve",),
                    True,
                    "Resolve rsIDs and gene symbols without choosing ambiguous candidates.",
                    "locus input identifiers",
                )
            )

        context_candidates = (
            "locus.variant",
            "locus.study",
            "locus.credible_set",
            "locus.fine_mapping",
        )
        if requested_kinds is None:
            locus_context = _select(available, context_candidates)
        else:
            # One locus module is mandatory for canonical-ID/input validation.
            # Additional study/fine-mapping modules run only when requested.
            operational = _select(available, ("locus.variant",))
            if not operational:
                operational = _select(available, context_candidates)[:1]
            requested_context = tuple(
                module
                for module in _select(available, context_candidates)
                if _module_requested(module, requested_kinds)
                or (
                    module.startswith("locus.")
                    and EvidenceKind.GWAS in requested_kinds
                    and module != "locus.variant"
                )
            )
            locus_context = tuple(dict.fromkeys((*operational, *requested_context)))
        steps.append(
            SemanticPlanStep(
                2,
                SemanticTool.GET_LOCUS_CONTEXT,
                locus_context,
                True,
                "Validate the locus and retain native fine-mapping measurements.",
                "resolved variants, study IDs, and locus region",
            )
        )

        gene_modules = [
            "locus.variant_consequence",
            "locus.l2g",
            "locus.qtl",
            "locus.colocalisation",
            "locus.enhancer_to_gene",
        ]
        if query.benchmark:
            gene_modules.remove("locus.l2g")
        locus_gene = _select(available, tuple(gene_modules))
        if requested_kinds is not None:
            locus_gene = tuple(
                module
                for module in locus_gene
                if _module_requested(module, requested_kinds)
                or (
                    module == "locus.variant_consequence"
                    and query.task in {"gene_associations", "prioritize_genes"}
                )
            )
        if query.task != "describe_variants":
            steps.append(
                SemanticPlanStep(
                    3,
                    SemanticTool.GET_LOCUS_GENE_EVIDENCE,
                    locus_gene,
                    True,
                    (
                        "Build typed variant-to-gene links inside the fixed candidate "
                        "universe, or discover candidates for exploratory runs."
                    ),
                    "validated locus and optional resolved candidate target IDs",
                )
            )

        target_modules: tuple[str, ...] = ()
        profile_task = "prioritize_genes" if query.benchmark else query.task
        if profile_task in {"gene_associations", "prioritize_genes"}:
            target_modules = (
                "target.core",
                "target.function",
                "target.expression",
                "target.go",
                "target.pathway",
                "evidence.europepmc",
            )
        elif profile_task == "target_assessment":
            target_modules = (
                "target.core",
                "target.function",
                "target.expression",
                "target.go",
                "target.pathway",
                "target.constraint",
                "target.tractability",
                "target.safety",
                "target.interaction",
                "target.mouse_phenotype",
                "evidence.europepmc",
            )
        elif profile_task == "drug_repurposing":
            target_modules = (
                "target.core",
                "target.constraint",
                "target.tractability",
                "target.safety",
                "evidence.europepmc",
            )
        selected_targets = _select(available, target_modules)
        if requested_kinds is not None:
            selected_targets = tuple(
                module
                for module in selected_targets
                if _module_requested(module, requested_kinds)
            )
        if selected_targets and (query.candidate_genes or not query.benchmark):
            steps.append(
                SemanticPlanStep(
                    4,
                    SemanticTool.GET_TARGET_PROFILES,
                    selected_targets,
                    False,
                    "Add biological and tissue context after locus-to-gene retrieval.",
                    "candidate or discovered targets retained after the mandatory gene-link step",
                )
            )

        disease_modules = _select(
            available,
            (
                "disease.core",
                "disease.ontology",
                "disease.hpo",
                "disease.therapeutic_area",
            ),
        )
        if (
            requested_kinds is not None
            and EvidenceKind.ONTOLOGY not in requested_kinds
        ):
            disease_modules = ()
        if query.trait_ids and not query.benchmark and disease_modules:
            steps.append(
                SemanticPlanStep(
                    5,
                    SemanticTool.GET_DISEASE_PROFILES,
                    disease_modules,
                    False,
                    "Add explicit disease ontology and phenotype context.",
                    "caller-supplied disease identifiers",
                )
            )

        association_modules = _select(
            available,
            (
                "association.overall_direct",
                "association.datasource_direct",
                "association.raw_evidence",
            ),
        )
        if requested_kinds is not None and EvidenceKind.TARGET_DISEASE not in requested_kinds:
            association_modules = ()
        needs_associations = query.task == "disease_associations" or (
            query.task == "drug_repurposing" and bool(query.trait_ids)
        )
        if needs_associations and not query.benchmark and association_modules:
            steps.append(
                SemanticPlanStep(
                    5,
                    SemanticTool.GET_TARGET_DISEASE_EVIDENCE,
                    association_modules,
                    False,
                    (
                        "Retrieve direct target-disease associations without treating "
                        "association scores as probabilities. An empty disease-ID list "
                        "means all direct associations for the discovered targets."
                    ),
                    "resolved targets and explicit disease IDs, if supplied",
                )
            )

        requested_drug_modules: tuple[str, ...] = ()
        if query.task == "prioritize_genes" and not query.benchmark:
            # Auto Analysis includes a bounded related-drugs panel.  Only the
            # direct clinical-target relation is needed for that panel.
            requested_drug_modules = ("drug.clinical_target",)
        elif query.task == "drug_repurposing" and not query.benchmark:
            requested_drug_modules = (
                "drug.core",
                "drug.mechanism",
                "drug.indication",
                "drug.clinical_target",
                "drug.warning",
                "drug.adverse_event",
                "drug.pharmacogenomics",
            )
        drug_modules = _select(available, requested_drug_modules)
        if requested_kinds is not None and EvidenceKind.DRUG not in requested_kinds:
            drug_modules = ()
        if drug_modules:
            steps.append(
                SemanticPlanStep(
                    6,
                    SemanticTool.GET_DRUG_EVIDENCE,
                    drug_modules,
                    False,
                    "Add translational evidence only after genetic prioritization.",
                    "prioritized targets and resolved disease IDs",
                )
            )

        required_tools = [SemanticTool.GET_LOCUS_CONTEXT]
        if query.task != "describe_variants":
            required_tools.append(SemanticTool.GET_LOCUS_GENE_EVIDENCE)
        for required_tool in required_tools:
            step = next(item for item in steps if item.tool is required_tool)
            if not step.modules:
                raise ValueError(
                    f"required semantic modules are unavailable for {required_tool.value}"
                )
        if len(steps) > parameter("retrieval.semantic_max_calls"):
            # With resolution, drug queries can request seven calls when both
            # ontology and associations are installed. Keep the existing six-
            # call bound, retaining the task-specific associations and drugs.
            steps = [step for step in steps if step.tool is not SemanticTool.GET_DISEASE_PROFILES]
        return SemanticEvidencePlan(tuple(steps))


def _select(available: frozenset[str], requested: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(module for module in requested if module in available)


def _module_requested(
    module: str, requested_kinds: frozenset[EvidenceKind]
) -> bool:
    try:
        return evidence_kind_for_module(module) in requested_kinds
    except ValueError:
        return False
