from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..biological_context import (
    BiologicalContext,
    define_biological_context,
)
from ..configuration import parameter
from ..coverage import (
    OT_CREDIBLE_SET_DOCS_URL,
    OT_MHC_END,
    OT_MHC_START,
    mappings_overlap_ot_mhc,
)
from ..ld_support import ld_report_scope, ld_scope_summary, proxy_trace_label
from ..models import DecisionStatus, EvidenceKind, EvidenceRecord
from ..prompting import format_prompt, load_prompt
from ..publications import CitationRegistry as PublicationCitationRegistry
from ..publications import (
    Publication,
    parse_colocalization_publication_lineage,
    publication_from_metadata,
    resolve_datasource_publication,
    resolve_publication,
)
from ..source_gwas import (
    SourceGWASSelectionSummary,
    summarize_source_gwas_observations,
)
from ..staged_reasoning import has_direct_locus_gene_evidence
from ..processing_scope import RESULT_PROCESSING_LIMIT_MESSAGE, is_source_only_processing_limit
from .chat_intent import ChatIntent
from .gene_evidence import (
    EvidenceTrace,
    GeneEvidence,
    aggregate_gene_evidence,
    format_probability,
    render_gene_evidence_markdown,
)

ENSEMBL_OVERLAP_DOCS_URL = "https://rest.ensembl.org/documentation/info/overlap_region"
OPEN_TARGETS_VARIANT_DOCS_URL = "https://platform-docs.opentargets.org/variant"
MAX_GENE_REFERENCES = parameter("web.gene_reference_limit")
MAX_REFERENCES_PER_GENE = parameter("web.references_per_gene")


def _complete_reference_registry(
    registry: PublicationCitationRegistry,
    publication_enricher: Callable[[PublicationCitationRegistry], object] | None,
) -> None:
    registry.fill_known_metadata()
    if publication_enricher is not None and registry.entries:
        try:
            publication_enricher(registry)
        except (RuntimeError, OSError, ValueError, TypeError):
            # A bibliography service failure cannot discard verified science.
            # The runtime callback records its own bounded enrichment audit.
            pass


# Trait classes are deliberately broad and deterministic. They are used only to
# summarize Open Targets traitFromSource values already present in the evidence;
# they never add an ontology term or infer a new association.
_TRAIT_CATEGORY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "metabolic/glycemic",
        re.compile(
            r"\b(?:diabet\w*|glucos\w*|glyc(?:a?em|os)\w*|hba1c|"
            r"(?:glycated )?(?:ha?emoglobin) a1c|insulin\w*|"
            r"metabol\w*|lipid\w*|cholesterol|triglyceride\w*|obes\w*|adipos\w*|"
            r"body mass index|bmi|urate|uric acid|gout|fatty acid\w*|amino acid\w*|"
            r"histidine\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "respiratory",
        re.compile(
            r"\b(?:asthma\w*|copd|chronic obstructive|bronch\w*|respirat\w*|"
            r"pulmonary\w*|spirometr\w*|fev1|fvc|lung (?:function|capacity|volume|"
            r"disease)|emphysema\w*|sleep apn(?:ea|oea))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "immune/inflammatory",
        re.compile(
            r"\b(?:autoimmun\w*|inflamm\w*|allerg\w*|atop\w*|rheumatoid\w*|"
            r"lupus|psoriasis|crohn\w*|ulcerative colitis|celiac|coeliac|immun\w*|"
            r"ige|c[- ]reactive protein|cytokine\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cardiovascular",
        re.compile(
            r"\b(?:coronary|cardiovascular|cardiac|heart|myocard\w*|blood pressure|"
            r"hypertension|atrial|vascular|arterial|artery|stroke|thrombo\w*|"
            r"aneurysm\w*|atherosclero\w*|pulse)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hematologic",
        re.compile(
            r"\b(?:hemat\w*|haemat\w*|hemoglobin|haemoglobin|anemia|anaemia|"
            r"erythrocyte\w*|red blood cell\w*|leukocyte\w*|white blood cell\w*|"
            r"platelet\w*|neutrophil\w*|eosinophil\w*|basophil\w*|monocyte\w*|"
            r"lymphocyte\w*|coagulat\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "neoplastic",
        re.compile(
            r"\b(?:cancer\w*|carcinoma\w*|tumou?r\w*|neoplasm\w*|malignan\w*|"
            r"lymphoma\w*|leukemia\w*|leukaemia\w*|melanoma\w*|myeloma\w*|"
            r"glioma\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reproductive",
        re.compile(
            r"\b(?:uter\w*|ovari\w*|endometri\w*|leiomyoma\w*|fibroid\w*|"
            r"pregnan\w*|fertil\w*|infertil\w*|menopaus\w*|menstru\w*|prostat\w*|"
            r"polycystic ovary|erectile)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "neurologic/psychiatric",
        re.compile(
            r"\b(?:neurolog\w*|alzheimer\w*|parkinson\w*|dementia\w*|cognit\w*|"
            r"epilep\w*|seizure\w*|migraine\w*|depress\w*|schizophren\w*|"
            r"bipolar\w*|anxiety|insomnia\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "renal",
        re.compile(
            r"\b(?:renal|kidney|nephr\w*|egfr|creatinine|albuminuria)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "gastrointestinal/hepatic",
        re.compile(
            r"\b(?:liver|hepatic|cirrhosis|bowel|intestin\w*|gastric|gastro\w*|"
            r"bilirubin|aminotransferase|gallbladder|pancrea\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "musculoskeletal",
        re.compile(
            r"\b(?:bone|fracture\w*|osteo\w*|arthritis|muscle|musculoskeletal|"
            r"sarcopenia\w*|joint|back pain)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dermatologic",
        re.compile(
            r"\b(?:skin|dermat\w*|eczema\w*|acne|alopecia\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "infectious",
        re.compile(
            r"\b(?:infect\w*|viral|virus|bacterial|covid(?:-19)?|influenza|"
            r"tuberculosis|sepsis)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "anthropometric",
        re.compile(
            r"\b(?:height|weight|waist|hip circumference|body shape|birth size)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ophthalmic",
        re.compile(
            r"\b(?:ocular|retina\w*|glaucoma\w*|cataract\w*|macular|eyesight|eye)\b",
            re.IGNORECASE,
        ),
    ),
)

_TRAIT_LABEL_NOISE = re.compile(
    r"\b(?:uk biobank|ukb|phecode|field (?:id|code)|self[- ]reported|"
    r"inverse[- ]normal(?:ized)?|inv[- ]norm(?:alized)?|maximum|minimum|"
    r"more control exclusions|adjusted for)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    key: str
    markdown: str


@dataclass(frozen=True, slots=True)
class DatabaseAnswerValidation:
    """Validation result for one model-rendered, retrieval-backed answer."""

    valid: bool
    issues: tuple[str, ...] = ()


def database_response_contract(intent: ChatIntent) -> str:
    """Return the fixed user-facing contract for every retrieval-backed answer."""

    detail_name = {
        ChatIntent.VARIANT_DESCRIPTION: "database_intent_describe_variants",
        ChatIntent.GENE_PRIORITIZATION: "database_intent_prioritize_genes",
        ChatIntent.GENE_ASSOCIATIONS: "database_intent_gene_associations",
        ChatIntent.TARGET_ASSESSMENT: "database_intent_target_assessment",
        ChatIntent.DISEASE_ASSOCIATIONS: "database_intent_disease_associations",
        ChatIntent.DRUG_REPURPOSING: "database_intent_drug_repurposing",
        ChatIntent.GENERAL: "database_intent_general",
    }[intent]
    return format_prompt(
        "database_response_contract",
        intent_detail=load_prompt(detail_name),
    )


def selected_variant_trait_answer(
    observations: Sequence[Mapping[str, Any]],
    *,
    trait: str,
    query_phenotype: str | None = None,
    region: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    publication_enricher: Callable[[PublicationCitationRegistry], object] | None = None,
) -> str:
    """Render the compact, deterministic Describe-SNP evidence contract.

    Open Targets may return hundreds of credible-set rows.  The display contract mirrors
    the useful part of the original interface without delegating evidence or reference selection
    to the model: choose at most five trait-diverse variants, retain at most five distinct
    associations for each, and register publications only after that bounded selection.
    """

    source_only = _source_only_retrieval_answer(observations, trait=trait, result=result)
    if source_only is not None:
        return source_only
    normalized = [_normalize_observation(item) for item in observations]
    normalized = [item for item in normalized if item["variant_id"]]
    evidence = _evidence_rows(result)
    studies = [item for item in _study_associations(evidence) if _is_trait_study(item)]
    source_trait = _plain(trait, 500) or "the loaded phenotype"
    requested_phenotype = _plain(query_phenotype, 500)
    focused_query = bool(requested_phenotype)
    focused_studies = (
        [
            item
            for item in studies
            if _phenotypes_equivalent(requested_phenotype, _plain(item.get("trait"), 500))
        ]
        if focused_query
        else studies
    )
    representative = _representative_trait_associations(focused_studies)
    displayed_studies = [item for _, items in representative for item in items]
    release = _plain((result or {}).get("data_release"), 80) or "not reported"
    citations, citation_numbers = _citation_registry(
        displayed_studies,
        include_mhc=False,
        publication_enricher=publication_enricher,
    )
    rs_ids = _variant_rs_ids(evidence)

    selected_count = len(normalized)
    external_variant_count = len({item["variant_id"] for item in focused_studies})
    external_traits = sorted(
        {item["trait"] for item in focused_studies if item["trait"]},
        key=str.casefold,
    )
    region_text = _region_text(normalized, region)
    location = f" in {region_text}" if region_text else ""
    source_gwas = summarize_source_gwas_observations(
        normalized,
        phenotype=source_trait,
    )
    source_summary = _uploaded_gwas_signal_summary(source_gwas)

    if focused_studies:
        # Summarize the same representative associations shown in Evidence, so
        # every named trait has its own traceable fact and publication anchor.
        phenotype_overview = _phenotype_overview(
            displayed_studies, citation_numbers=citation_numbers
        )
        external_summary = (
            f"{phenotype_overview} In Open Targets release {release}, "
            f"{external_variant_count} of the {selected_count} user-selected "
            f"SNP{'s' if selected_count != 1 else ''}{location} "
            f"{'has' if external_variant_count == 1 else 'have'} exact credible-set trait "
            f"associations spanning {len(external_traits)} "
            f"reported trait{'s' if len(external_traits) != 1 else ''}. "
            f"Key findings from "
            f"{len(representative)} representative variant"
            f"{'s are' if len(representative) != 1 else ' is'} shown below."
        )
        if focused_query:
            external_summary = (
                f"The lookup found exact selected-variant records matching "
                f"**{requested_phenotype}**. " + external_summary
            )
    else:
        if focused_query:
            external_summary = (
                f"Open Targets release {release} returned no exact selected-variant "
                f"record matching **{requested_phenotype}**."
            )
            if not _phenotypes_equivalent(source_trait, requested_phenotype):
                external_summary += (
                    f" The uploaded *P* values measure association with **{source_trait}**, "
                    f"not **{requested_phenotype}**, so they cannot establish the queried association."
                )
        else:
            external_summary = (
                f"Open Targets release {release} returned no exact credible-set trait "
                f"associations for the {selected_count} user-selected SNP"
                f"{'s' if selected_count != 1 else ''}{location}; this is a database-coverage "
                "result, not evidence that the variants have no phenotypic associations."
            )
    if not focused_studies and selection_overlaps_ot_mhc(normalized):
        external_summary += (
            " The selection overlaps the MHC interval for which Open Targets excludes "
            f"credible sets whose lead variant is in chr6:{OT_MHC_START:,}-{OT_MHC_END:,} "
            f"([coverage documentation]({OT_CREDIBLE_SET_DOCS_URL}))."
        )
    summary = f"{source_summary}\n\n**External annotations:** {external_summary}"

    parts = [
        "**Summary**",
        "",
        summary,
        "",
        "**Evidence**",
        "",
        "### External database annotations",
        "",
    ]
    if focused_studies:
        for variant_id, items in representative:
            rs_id = rs_ids.get(variant_id, "")
            heading = (
                f"**{_plain(rs_id, 80)}** (`{_md_code(variant_id)}`)"
                if rs_id
                else f"**`{_md_code(variant_id)}`**"
            )
            parts.extend([heading, ""])
            for item in items:
                details = []
                if item["member_p"] is not None:
                    details.append(f"*P*={_format_p_display(item['member_p'])}")
                if item["member_beta"] is not None:
                    details.append(f"Beta={item['member_beta']:.4g}")
                if not details:
                    if item["query_variant_role"] == "credible_set_member":
                        membership = "credible-set member"
                        if item["lead_variant_id"]:
                            membership += f"; lead `{_md_code(item['lead_variant_id'])}`"
                        if item["posterior_probability"] is not None:
                            membership += f"; PIP={format_probability(item['posterior_probability'])}"
                        details.append(membership)
                    elif item["query_variant_role"] == "lead":
                        details.append(f"lead variant for study `{_md_code(item['study_id'])}`")
                    else:
                        details.append(f"study `{_md_code(item['study_id'])}`")
                citation = citation_numbers.get(item["citation_key"])
                if citation is None and not any(
                    f"`{_md_code(item['study_id'])}`" in detail for detail in details
                ):
                    details.append(f"study `{_md_code(item['study_id'])}`")
                suffix = f" [{citation}]" if citation is not None else ""
                parts.append(f"- **{_plain(item['trait'], 500)}:** {', '.join(details)}{suffix}")
            parts.append("")
    else:
        if focused_query:
            parts.append(
                "No exact selected-variant trait/study record matching "
                f"**{_md_inline(requested_phenotype)}** was returned."
            )
        else:
            parts.append("No exact selected-variant trait/study records were returned.")
        if selection_overlaps_ot_mhc(normalized):
            parts.extend(
                [
                    "",
                    (
                        "- **Coverage limitation:** This selection overlaps the documented MHC "
                        "[credible-set exclusion interval]"
                        f"({OT_CREDIBLE_SET_DOCS_URL}). The live API does not expose "
                        "per-record exclusion reasons, so GWASMind reports this as a scope "
                        "warning, not as proof that each absent record was individually excluded."
                    ),
                ]
            )

    parts.extend(["", "**References**", ""])
    if citations:
        parts.extend(citation.markdown for citation in citations)
    else:
        parts.append(
            "No publication metadata was returned for the displayed associations; no "
            "literature citation was invented."
        )
    return "\n".join(parts)


def _uploaded_gwas_signal_summary(
    summary: SourceGWASSelectionSummary, *, external_evidence_reported: bool = True,
) -> str:
    """Render current-GWAS measurements before any external annotation.

    This wording deliberately separates significance in the user's uploaded
    analysis from associations reported by external studies.  It is especially
    explicit for negative controls so external database hits cannot be narrated
    as a signal in the current phenotype.
    """

    phenotype = _summary_emphasis(summary.phenotype or "the loaded phenotype")
    selected = summary.selected_variants
    significant = summary.genome_wide_significant_variants
    variant_word = "variant" if selected == 1 else "variants"
    threshold = _format_p_display(summary.significance_threshold)

    if summary.variants_with_p_value:
        p_range = (
            f"{_format_p_display(summary.minimum_p_value)} to "
            f"{_format_p_display(summary.maximum_p_value)}"
        )
        measurement_note = ""
        if summary.variants_with_p_value != selected:
            measurement_note = (
                f" Usable *P* values were available for "
                f"{summary.variants_with_p_value} of {selected} selected {variant_word}."
            )
        status = (
            f"In the uploaded {phenotype} GWAS, **{significant} of {selected} selected "
            f"{variant_word} {'is' if selected == 1 else 'are'} genome-wide significant** "
            f"at *P* ≤ {threshold} (selected *P*-value range: {p_range})."
            f"{measurement_note}"
        )
    else:
        status = (
            f"In the uploaded {phenotype} GWAS, none of the {selected} selected "
            f"{variant_word} has a usable *P* value, so genome-wide significance and the "
            "selected *P*-value range cannot be assessed."
        )

    if summary.has_genome_wide_signal is True:
        if not external_evidence_reported:
            return status + " The current uploaded GWAS supports a genome-wide-significant signal among this selection."
        return (
            status
            + " The current uploaded GWAS therefore supports a genome-wide-significant "
            "signal among this selection. External annotations are reported separately "
            "and do not change the current-GWAS measurements."
        )
    if summary.has_genome_wide_signal is False:
        if not external_evidence_reported:
            return status + " **The current uploaded GWAS does not support a genome-wide-significant signal for this selection.**"
        return (
            status
            + " **The current uploaded GWAS does not support a genome-wide-significant "
            "signal for this selection. External associations reported below come from "
            "other studies and must not be interpreted as evidence that these variants "
            f"are significant for {phenotype} in the uploaded analysis.**"
        )
    return (
        status
        + " **External associations cannot substitute for the missing current-GWAS "
        "significance measurements.**"
    )


def _source_only_retrieval_answer(
    observations: Sequence[Mapping[str, Any]],
    *,
    trait: str,
    result: Mapping[str, Any] | None,
    prioritize: bool = False,
) -> str | None:
    """Describe source observations without converting a technical gap to no hit."""
    value = result or {}
    if is_source_only_processing_limit(value):
        normalized = [_normalize_observation(item) for item in observations]
        normalized = [item for item in normalized if item["variant_id"]]
        summary = _uploaded_gwas_signal_summary(summarize_source_gwas_observations(
            normalized, phenotype=_plain(trait, 500) or "the loaded phenotype",
        ), external_evidence_reported=False)
        return "\n\n".join((
            "**Summary**", summary + " " + RESULT_PROCESSING_LIMIT_MESSAGE,
            "**Evidence**", "### Uploaded GWAS data",
            "The selected variants and their original P values were retained.",
            "### Retained external result",
            "The complete external result remains stored. It was too large to review in this "
            "interactive analysis, so no external findings, gene ranking, or novelty conclusion "
            "are presented. A smaller retrieval scope or a workflow able to process the full "
            "result is needed for further interpretation; simply repeating this question may "
            "reach the same limit.",
            "**References**", "External references in the retained result have not been reviewed for this response.",
        ))
    failure = value.get("retrieval_failure")
    if not (
        value.get("external_evidence_available") is False
        and value.get("data_release_verified") is False
        and isinstance(failure, Mapping) and failure.get("source_only") is True
        and failure.get("phase") == "schema_discovery"
        and failure.get("status") == "unavailable"
        and not _evidence_rows(value) and not value.get("ranked_genes")
    ):
        return None
    normalized = [_normalize_observation(item) for item in observations]
    normalized = [item for item in normalized if item["variant_id"]]
    summary = _uploaded_gwas_signal_summary(summarize_source_gwas_observations(
        normalized, phenotype=_plain(trait, 500) or "the loaded phenotype",
    ), external_evidence_reported=False)
    gap = (
        "The Open Targets evidence service could not be initialized, so its database "
        "retrieval could not be completed. This part of the report describes the uploaded GWAS data."
    )
    next_step = (
        "Candidate-gene ranking is unavailable until external gene evidence can be retrieved."
        if prioritize else
        "Open Targets annotations and associations could not be assessed."
    )
    if isinstance(value.get("novelty_screen"), Mapping):
        gap += " The separately requested prior-report and LD search is described in its own section."
    return "\n\n".join((
        "**Summary**", summary, "**Evidence**", "### Uploaded GWAS data",
        "The selected variants and their original P values were retained.",
        "### External database annotations", gap + " " + next_step,
        "The selected input is retained; retry this question when the data service is available.",
        "**References**", "No Open Targets references were retrieved for this report.",
    ))


def selected_variant_gene_answer(
    observations: Sequence[Mapping[str, Any]],
    *,
    trait: str,
    query_phenotype: str | None = None,
    region: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    prioritize: bool,
    biological_context: BiologicalContext | None = None,
    publication_enricher: Callable[[PublicationCitationRegistry], object] | None = None,
) -> str:
    """Render the script-governed four-module candidate-gene answer.

    The ranking order comes from the validated synthesis result.  Formatting,
    scientific de-duplication, evidence selection, and bibliography construction
    remain deterministic so a second model call cannot delete modules, expose an
    L2G score, inflate record counts, or relabel a source publication.
    """

    source_only = _source_only_retrieval_answer(observations, trait=trait, result=result, prioritize=prioritize)
    if source_only is not None:
        return source_only
    normalized = [_normalize_observation(item) for item in observations]
    normalized = [item for item in normalized if item["variant_id"]]
    rows = _evidence_rows(result)
    records = _mapping_evidence_records(rows)
    ranked_value = (result or {}).get("ranked_genes")
    ranked = ranked_value if isinstance(ranked_value, list) else []
    ld_scope = ld_report_scope(result or {}, observations, "prioritize_genes" if prioritize else "gene_associations")
    if ld_scope.provided:
        records = tuple(record for record in records if ld_scope.permits_record(record.to_dict()))
    aggregated = aggregate_gene_evidence(
        records,
        ranked_genes=ranked if prioritize else (),
        variant_origins=(
            {record.variant_id: ld_scope.evidence_origin(record.to_dict())
             for record in records if record.variant_id}
            if ld_scope.provided else
            {item["variant_id"]: "source_selected" for item in normalized}
        ),
        colocalization_phenotype_match=(
            lambda observed: _phenotypes_equivalent(query_phenotype, observed)
        ) if query_phenotype else None,
    )

    locus_link_kinds = {
        EvidenceKind.VARIANT_ANNOTATION,
        EvidenceKind.QTL,
        EvidenceKind.COLOCALIZATION,
        EvidenceKind.REGULATORY,
    }
    linked_gene_ids = {
        record.gene_id
        for record in records
        if record.gene_id and record.kind in locus_link_kinds
    }
    rankable_gene_ids = {
        record.gene_id
        for record in records
        if record.gene_id and has_direct_locus_gene_evidence((record,))
    }
    display_limit = parameter("ranking.max_ranked_genes")
    if prioritize and ranked:
        # A display limit is not another candidate-admission decision. Preserve
        # the original returned order, including entries whose evidence has a
        # review limitation; never promote rank six when rank two is sparse.
        displayed = tuple(
            gene
            for gene in aggregated
            if gene.rank is not None
        )[:display_limit]
    elif prioritize:
        displayed = tuple(
            gene
            for gene in aggregated
            if gene.gene_id in rankable_gene_ids
            and (gene.molecular_qtl or gene.colocalizations or gene.enhancer_gene or gene.coding)
        )[:display_limit]
    else:
        displayed = tuple(
            gene for gene in aggregated if gene.gene_id in linked_gene_ids
        )[:display_limit]

    source_trait = _plain(trait, 500) or "the loaded GWAS phenotype"
    interpretation_phenotype = _plain(query_phenotype, 500) or source_trait
    resolved_biological_context = (
        biological_context or define_biological_context(interpretation_phenotype)
        if prioritize
        else None
    )
    biological_context_markdown = (
        _biological_context_markdown(resolved_biological_context, displayed)
        if resolved_biological_context is not None
        else None
    )
    ensembl_positional_gene_ids = {
        str(record.gene_id)
        for record in records
        if record.gene_id
        and "ensembl" in f"{record.source} {record.dataset}".casefold()
        and record.payload.get("relation") != "regional_gene_context_summary"
    }
    has_ensembl = any(
        gene.gene_id in ensembl_positional_gene_ids
        and not (gene.molecular_qtl or gene.colocalizations or gene.enhancer_gene or gene.coding)
        for gene in displayed
    )
    include_mhc = selection_overlaps_ot_mhc([*normalized, *ld_scope.proxy_observations])
    registry = PublicationCitationRegistry()
    citation_resolver = _gene_publication_resolver(records, registry)
    if ld_scope.proxy_ids:
        base_resolver = citation_resolver
        by_id = {record.evidence_id: record.to_dict() for record in records}

        def citation_resolver(trace):
            label = proxy_trace_label(ld_scope, [by_id[identifier] for identifier in trace.evidence_ids if identifier in by_id])
            return " ".join(part for part in (label, base_resolver(trace)) if part) or None
    gene_summary = _structured_gene_summary(
        displayed,
        trait=interpretation_phenotype,
        prioritize=prioritize,
        ranking_unavailable=prioritize and not bool(ranked),
        positional_only=bool(displayed) and not any(
            gene.molecular_qtl or gene.colocalizations or gene.enhancer_gene or gene.coding
            for gene in displayed
        ),
        citation_resolver=citation_resolver,
    )
    if ld_scope.proxy_ids:
        gene_summary = gene_summary.replace("direct locus-linked evidence", "locus-linked evidence")
    if prioritize and ranked:
        gene_summary += (
            f" This report displays {len(displayed)} of {len(ranked)} returned ranked genes "
            "in their original order; the complete ranking is retained in the saved result."
        )
    source_summary = _uploaded_gwas_signal_summary(
        summarize_source_gwas_observations(
            normalized,
            phenotype=source_trait,
        )
    )
    summary = f"{source_summary}\n\n**Integrated gene interpretation:** {gene_summary}"
    if ld_scope.provided:
        summary = f"{source_summary}\n\n**LD evidence scope:** {ld_scope_summary(ld_scope)}\n\n**Integrated gene interpretation:** {gene_summary}"

    # First pass registers only publications actually used by displayed lines.
    render_gene_evidence_markdown(
        displayed,
        summary=summary,
        biological_context=biological_context_markdown,
        citation_resolver=citation_resolver,
    )
    _complete_reference_registry(registry, publication_enricher)
    references = [entry.markdown() for entry in registry.entries]

    documentation_sentences: list[str] = []
    if has_ensembl:
        number = len(references) + 1
        references.append(
            f"[{number}] Ensembl REST API. [Overlap region endpoint]"
            f"({ENSEMBL_OVERLAP_DOCS_URL})."
        )
        documentation_sentences.append(
            "The positional candidates come from an Ensembl GRCh38 interval overlap "
            "rather than molecular variant-to-gene evidence "
            f"[{number}]({ENSEMBL_OVERLAP_DOCS_URL})."
        )
    if include_mhc:
        number = len(references) + 1
        references.append(
            f"[{number}] Open Targets Platform. [Credible Set exclusion criteria]"
            f"({OT_CREDIBLE_SET_DOCS_URL})."
        )
        documentation_sentences.append(
            ("This interval overlaps the MHC credible-set exclusion interval "
             if selection_overlaps_ot_mhc(normalized) else
             "The LD-expanded retrieval includes a proxy in the MHC credible-set exclusion interval ")
            + f"chr6:{OT_MHC_START:,}-{OT_MHC_END:,}, so missing fine-mapping evidence is "
            f"coverage-sensitive [{number}]({OT_CREDIBLE_SET_DOCS_URL})."
        )
    if documentation_sentences:
        summary = " ".join((summary, *documentation_sentences))
    if not references:
        references = [
            (
                "No publication metadata was attached to the displayed evidence; no "
                "reference was invented."
            )
        ]

    return render_gene_evidence_markdown(
        displayed,
        summary=summary,
        biological_context=biological_context_markdown,
        citation_resolver=citation_resolver,
        references=references,
    ).rstrip()


def task_specific_retrieval_answer(
    intent: ChatIntent,
    *,
    result: Mapping[str, Any] | None,
    observations: Sequence[Mapping[str, Any]] = (),
    trait: str = "",
    query_phenotype: str | None = None,
    region: Mapping[str, Any] | None = None,
    biological_context: BiologicalContext | None = None,
    publication_enricher: Callable[[PublicationCitationRegistry], object] | None = None,
) -> str:
    """Render the deterministic authority for every retrieval-backed chat task.

    This dispatcher is intentionally independent of question wording.  The route has
    already been validated before it reaches this layer, so each task receives a
    scientific view tailored to its evidence rather than falling back to a generic
    selected-variant summary.  The hybrid response layer may rewrite the Summary and
    add ``Limitations & Uncertainty``; Evidence and References remain grounded here.
    """

    source_only = _source_only_retrieval_answer(
        observations, trait=trait, result=result, prioritize=intent is ChatIntent.GENE_PRIORITIZATION,
    )
    if source_only is not None:
        return source_only
    if intent is ChatIntent.VARIANT_DESCRIPTION:
        return selected_variant_trait_answer(
            observations,
            trait=trait,
            query_phenotype=query_phenotype,
            region=region,
            result=result,
            publication_enricher=publication_enricher,
        )
    if intent is ChatIntent.GENE_ASSOCIATIONS:
        return selected_variant_gene_answer(
            observations,
            trait=trait,
            query_phenotype=query_phenotype,
            region=region,
            result=result,
            prioritize=False,
            biological_context=biological_context,
            publication_enricher=publication_enricher,
        )
    if intent is ChatIntent.GENE_PRIORITIZATION:
        return selected_variant_gene_answer(
            observations,
            trait=trait,
            query_phenotype=query_phenotype,
            region=region,
            result=result,
            prioritize=True,
            biological_context=biological_context,
            publication_enricher=publication_enricher,
        )
    if intent is ChatIntent.TARGET_ASSESSMENT:
        return target_assessment_answer(
            result=result,
            observations=observations,
            trait=trait,
            publication_enricher=publication_enricher,
        )
    if intent is ChatIntent.DISEASE_ASSOCIATIONS:
        return disease_association_answer(
            result=result,
            observations=observations,
            trait=trait,
            publication_enricher=publication_enricher,
        )
    if intent is ChatIntent.DRUG_REPURPOSING:
        return drug_repurposing_answer(
            result=result,
            observations=observations,
            trait=trait,
            publication_enricher=publication_enricher,
        )
    raise ValueError("general chat does not have a retrieval-backed answer")


def target_assessment_answer(
    *,
    result: Mapping[str, Any] | None,
    observations: Sequence[Mapping[str, Any]] = (),
    trait: str = "",
    publication_enricher: Callable[[PublicationCitationRegistry], object] | None = None,
) -> str:
    """Render target biology and translational annotations without implying validation."""

    rows = [
        row
        for row in _evidence_rows(result)
        if _row_kind(row)
        in {
            EvidenceKind.FUNCTION.value,
            EvidenceKind.EXPRESSION.value,
            EvidenceKind.PATHWAY.value,
            EvidenceKind.SAFETY.value,
            EvidenceKind.TRACTABILITY.value,
            EvidenceKind.INTERACTION.value,
            EvidenceKind.PHENOTYPE.value,
        }
    ]
    rows = _deduplicate_semantic_rows(rows)
    targets = _target_coverage(rows)
    target_names = [item[1] for item in targets]
    module_names = sorted({_target_module_label(row) for row in rows}, key=str.casefold)
    if rows:
        summary = (
            "GWASMind's integrated target assessment retrieved "
            f"{len(rows)} biological or translational annotation"
            f"{'s' if len(rows) != 1 else ''} for {len(targets)} target"
            f"{'s' if len(targets) != 1 else ''}: "
            f"{_natural_list([_summary_emphasis(item) for item in target_names])}. "
            "The available profile spans "
            f"{_natural_list([_summary_emphasis(item) for item in module_names])}. "
            "These annotations characterize the targets; they do not by themselves "
            "validate a therapeutic target or establish clinical safety."
        )
    else:
        summary = (
            "GWASMind found no target-profile evidence specific to the requested "
            "assessment. This is a retrieval-coverage result, not evidence that a "
            "target lacks biological function, expression, safety liabilities, or "
            "therapeutic tractability."
        )
    summary = _prepend_source_gwas_summary(summary, observations, trait)

    registry = PublicationCitationRegistry()
    overview_lines: list[str] = []
    for target_id, label, target_rows in targets:
        dimensions = sorted(
            {_target_module_label(row) for row in target_rows}, key=str.casefold
        )
        identifier = f" (`{_md_code(target_id)}`)" if target_id and target_id != label else ""
        overview_lines.append(
            f"- **{_md_inline(label)}**{identifier}: "
            f"{_natural_list([_md_inline(item) for item in dimensions])}."
        )
    if not overview_lines:
        overview_lines.append(
            "No target identity with task-specific profile evidence was returned."
        )

    biology_rows = [
        row
        for row in rows
        if _row_kind(row)
        in {
            EvidenceKind.FUNCTION.value,
            EvidenceKind.EXPRESSION.value,
            EvidenceKind.PATHWAY.value,
        }
    ]
    translational_rows = [row for row in rows if row not in biology_rows]
    biology_lines = [
        _target_profile_bullet(row, registry) for row in biology_rows[:12]
    ] or ["No function, expression, or pathway annotation was returned."]
    translational_lines = [
        _target_profile_bullet(row, registry) for row in translational_rows[:12]
    ] or [
        "No safety, tractability, interaction, or model-organism phenotype annotation "
        "was returned."
    ]
    return _assemble_retrieval_answer(
        summary,
        (
            ("Targets and evidence coverage", overview_lines),
            ("Function, expression, and pathways", biology_lines),
            ("Safety, tractability, interactions, and phenotypes", translational_lines),
        ),
        registry,
        publication_enricher=publication_enricher,
    )


def disease_association_answer(
    *,
    result: Mapping[str, Any] | None,
    observations: Sequence[Mapping[str, Any]] = (),
    trait: str = "",
    publication_enricher: Callable[[PublicationCitationRegistry], object] | None = None,
) -> str:
    """Render direct target-disease associations and their score composition."""

    rows = _deduplicate_semantic_rows(
        [
            row
            for row in _evidence_rows(result)
            if _row_kind(row) == EvidenceKind.TARGET_DISEASE.value
        ]
    )
    associations: list[tuple[float | None, Mapping[str, Any]]] = [
        (_association_score(row), row) for row in rows
    ]
    associations.sort(
        key=lambda item: (
            item[0] is None,
            -(item[0] or 0.0),
            _disease_label(item[1]).casefold(),
            _target_label(item[1])[1].casefold(),
        )
    )
    disease_names = sorted(
        {_disease_label(row) for _, row in associations if _disease_label(row)},
        key=str.casefold,
    )
    target_names = sorted(
        {_target_label(row)[1] for _, row in associations if _target_label(row)[1]},
        key=str.casefold,
    )
    if rows:
        representative = disease_names[:5]
        summary = (
            "GWASMind's integrated disease-association view found "
            f"{len(rows)} direct target-disease association"
            f"{'s' if len(rows) != 1 else ''} linking {len(target_names)} target"
            f"{'s' if len(target_names) != 1 else ''} to {len(disease_names)} disease"
            f"{'s' if len(disease_names) != 1 else ''}. Representative diseases are "
            f"{_natural_list([_summary_emphasis(item) for item in representative])}. "
            "Association scores are source-native evidence-ranking summaries, not "
            "probabilities of causality or clinical effect."
        )
    else:
        summary = (
            "GWASMind retrieved no direct target-disease association specific to the "
            "requested targets. A missing database match is a coverage result and does "
            "not prove that the targets have no disease relevance."
        )
    summary = _prepend_source_gwas_summary(summary, observations, trait)

    registry = PublicationCitationRegistry()
    association_lines = [
        _disease_association_bullet(row, registry) for _, row in associations[:20]
    ] or ["No direct target-disease association record was returned."]
    composition_lines = _association_composition_lines(
        [row for _, row in associations], registry
    ) or [
        "No datasource- or datatype-level score composition was returned for the "
        "requested associations."
    ]
    return _assemble_retrieval_answer(
        summary,
        (
            ("Direct target-disease associations", association_lines),
            ("Evidence composition", composition_lines),
        ),
        registry,
        publication_enricher=publication_enricher,
    )


def drug_repurposing_answer(
    *,
    result: Mapping[str, Any] | None,
    observations: Sequence[Mapping[str, Any]] = (),
    trait: str = "",
    publication_enricher: Callable[[PublicationCitationRegistry], object] | None = None,
) -> str:
    """Render drug evidence as repurposing context, never as an efficacy claim."""

    rows = _deduplicate_semantic_rows(
        [
            row
            for row in _evidence_rows(result)
            if _row_kind(row)
            in {
                EvidenceKind.DRUG.value,
                EvidenceKind.SAFETY.value,
                EvidenceKind.TRACTABILITY.value,
            }
        ]
    )
    drug_rows = [row for row in rows if _row_kind(row) == EvidenceKind.DRUG.value]
    drug_names = sorted(
        {_drug_label(row) for row in drug_rows if _drug_label(row)}, key=str.casefold
    )
    mechanism_rows = [row for row in drug_rows if "mechanism" in _row_module(row)]
    indication_rows = [
        row
        for row in drug_rows
        if any(token in _row_module(row) for token in ("indication", "clinical_target"))
    ]
    safety_rows = [
        row
        for row in rows
        if _row_kind(row) in {EvidenceKind.SAFETY.value, EvidenceKind.TRACTABILITY.value}
        or any(
            token in _row_module(row)
            for token in ("warning", "adverse", "pharmacogen", "safety")
        )
    ]
    core_rows = [
        row
        for row in drug_rows
        if row not in mechanism_rows and row not in indication_rows and row not in safety_rows
    ]
    if drug_rows:
        summary = (
            "GWASMind's integrated drug-evidence view identified "
            f"{len(drug_names)} drug or clinical candidate"
            f"{'s' if len(drug_names) != 1 else ''}: "
            f"{_natural_list([_summary_emphasis(item) for item in drug_names[:8]])}. "
            f"The retrieved evidence contains {len(mechanism_rows)} mechanism, "
            f"{len(indication_rows)} indication or clinical-stage, and "
            f"{len(safety_rows)} safety or tractability annotation"
            f"{'s' if len(safety_rows) != 1 else ''}. These are candidates for expert "
            "review, not evidence of efficacy, safety, or a validated new indication."
        )
    else:
        summary = (
            "GWASMind retrieved no drug or clinical-candidate record specific to the "
            "requested target context. This is a database-coverage result and does not "
            "establish that the targets are undruggable or that repurposing is impossible."
        )
    summary = _prepend_source_gwas_summary(summary, observations, trait)

    registry = PublicationCitationRegistry()
    identity_lines = [
        _drug_evidence_bullet(row, registry) for row in core_rows[:10]
    ] or [
        f"- **{_md_inline(name)}**: a drug or clinical-candidate record was returned."
        for name in drug_names[:10]
    ] or ["No drug identity or clinical-candidate record was returned."]
    mechanism_lines = [
        _drug_evidence_bullet(row, registry) for row in mechanism_rows[:12]
    ] or ["No target-specific mechanism-of-action record was returned."]
    indication_lines = [
        _drug_evidence_bullet(row, registry) for row in indication_rows[:12]
    ] or ["No indication or clinical-stage record was returned."]
    safety_lines = [
        _drug_evidence_bullet(row, registry) for row in safety_rows[:12]
    ] or [
        "No drug warning, adverse-event, pharmacogenomic, target-safety, or "
        "tractability annotation was returned."
    ]
    return _assemble_retrieval_answer(
        summary,
        (
            ("Drugs and clinical candidates", identity_lines),
            ("Mechanisms and linked targets", mechanism_lines),
            ("Indications and clinical development", indication_lines),
            ("Safety, adverse events, pharmacogenomics, and tractability", safety_lines),
        ),
        registry,
        publication_enricher=publication_enricher,
    )


def _assemble_retrieval_answer(
    summary: str,
    sections: Sequence[tuple[str, Sequence[str]]],
    registry: PublicationCitationRegistry,
    *, publication_enricher: Callable[[PublicationCitationRegistry], object] | None = None,
) -> str:
    """Build the shared total/parts shell and a one-publication-per-line ledger."""

    _complete_reference_registry(registry, publication_enricher)
    parts = ["**Summary**", "", summary, "", "**Evidence**", ""]
    for heading, lines in sections:
        parts.extend([f"### {heading}", ""])
        parts.extend(str(line).strip() for line in lines if str(line).strip())
        parts.append("")
    parts.extend(["**References**", ""])
    if registry.entries:
        # A bibliography entry must remain exactly one physical line.  This also
        # prevents an upstream title containing a newline from corrupting the report.
        parts.extend(" ".join(entry.markdown().split()) for entry in registry.entries)
    else:
        parts.append(
            "No publication metadata was attached to the displayed evidence; no "
            "reference was invented."
        )
    return "\n".join(parts).rstrip()


def _prepend_source_gwas_summary(
    summary: str,
    observations: Sequence[Mapping[str, Any]],
    trait: str,
) -> str:
    normalized = [_normalize_observation(item) for item in observations]
    normalized = [item for item in normalized if item["variant_id"]]
    if not normalized:
        return summary
    source_summary = _uploaded_gwas_signal_summary(
        summarize_source_gwas_observations(
            normalized,
            phenotype=_plain(trait, 500) or "the loaded phenotype",
        )
    )
    return f"{source_summary}\n\n**Requested analysis:** {summary}"


def _row_kind(row: Mapping[str, Any]) -> str:
    return _plain(row.get("kind"), 80).casefold()


def _row_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("payload")
    return payload if isinstance(payload, Mapping) else {}


def _row_raw(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _row_payload(row)
    upstream = payload.get("upstream")
    upstream = upstream if isinstance(upstream, Mapping) else {}
    raw = upstream.get("raw")
    return raw if isinstance(raw, Mapping) else payload


def _row_module(row: Mapping[str, Any]) -> str:
    payload = _row_payload(row)
    return _plain(payload.get("module") or row.get("dataset"), 160).casefold()


def _row_mapping(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = _row_payload(row).get(field)
    return value if isinstance(value, Mapping) else {}


def _entity(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = _row_payload(row).get(field)
    return value if isinstance(value, Mapping) else {}


def _target_label(row: Mapping[str, Any]) -> tuple[str, str]:
    payload = _row_payload(row)
    raw = _row_raw(row)
    subject = _entity(row, "subject")
    object_value = _entity(row, "object")
    target = (
        object_value
        if _plain(object_value.get("type"), 40).casefold() == "target"
        else subject
        if _plain(subject.get("type"), 40).casefold() == "target"
        else {}
    )
    target_id = _plain(
        row.get("gene_id")
        or target.get("id")
        or raw.get("id")
        or raw.get("targetId")
        or payload.get("geneId"),
        160,
    )
    label = _plain(
        target.get("label")
        or raw.get("approvedSymbol")
        or raw.get("geneSymbol")
        or target_id,
        160,
    )
    return target_id or label, label or target_id or "target not identified"


def _disease_label(row: Mapping[str, Any]) -> str:
    subject = _entity(row, "subject")
    object_value = _entity(row, "object")
    disease = (
        object_value
        if _plain(object_value.get("type"), 40).casefold() == "disease"
        else subject
        if _plain(subject.get("type"), 40).casefold() == "disease"
        else {}
    )
    raw = _row_raw(row)
    raw_disease = raw.get("disease")
    raw_disease = raw_disease if isinstance(raw_disease, Mapping) else {}
    return _plain(
        disease.get("label")
        or disease.get("id")
        or raw_disease.get("name")
        or raw_disease.get("id"),
        240,
    )


def _drug_label(row: Mapping[str, Any]) -> str:
    subject = _entity(row, "subject")
    object_value = _entity(row, "object")
    drug = (
        subject
        if _plain(subject.get("type"), 40).casefold() == "drug"
        else object_value
        if _plain(object_value.get("type"), 40).casefold() == "drug"
        else {}
    )
    raw = _row_raw(row)
    raw_drug = raw.get("drug")
    raw_drug = raw_drug if isinstance(raw_drug, Mapping) else {}
    return _plain(
        drug.get("label")
        or drug.get("id")
        or raw.get("name")
        or raw_drug.get("name")
        or raw_drug.get("id"),
        240,
    )


def _deduplicate_semantic_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        evidence_id = _plain(row.get("evidence_id"), 300)
        if evidence_id:
            key = f"id:{evidence_id}"
        else:
            target_id, target_label = _target_label(row)
            key = repr(
                (
                    _row_kind(row),
                    _row_module(row),
                    target_id,
                    target_label,
                    _disease_label(row),
                    _drug_label(row),
                    _row_mapping(row, "measurements"),
                    _row_mapping(row, "context"),
                )
            )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _target_coverage(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, list[Mapping[str, Any]]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        target_id, label = _target_label(row)
        grouped[(target_id, label)].append(row)
    return [
        (target_id, label, grouped[(target_id, label)])
        for target_id, label in sorted(grouped, key=lambda item: item[1].casefold())
    ]


def _target_module_label(row: Mapping[str, Any]) -> str:
    module = _row_module(row)
    if "pathway" in module or module.endswith(".go") or module.endswith("_go"):
        return "pathways and ontology"
    return {
        EvidenceKind.FUNCTION.value: "function",
        EvidenceKind.EXPRESSION.value: "expression",
        EvidenceKind.PATHWAY.value: "pathways",
        EvidenceKind.SAFETY.value: "safety liabilities",
        EvidenceKind.TRACTABILITY.value: "tractability",
        EvidenceKind.INTERACTION.value: "interactions",
        EvidenceKind.PHENOTYPE.value: "model-organism phenotypes",
    }.get(_row_kind(row), _row_kind(row).replace("_", " ") or "annotation")


def _target_profile_bullet(
    row: Mapping[str, Any],
    registry: PublicationCitationRegistry,
) -> str:
    target_id, label = _target_label(row)
    kind = _row_kind(row)
    module = _row_module(row)
    raw = _row_raw(row)
    measurements = _row_mapping(row, "measurements")
    context = _row_mapping(row, "context")
    detail = ""
    if "pathway" in module or module.endswith(".go") or module.endswith("_go"):
        detail = _pathway_detail(raw)
    elif kind == EvidenceKind.FUNCTION.value:
        descriptions = _text_values(raw.get("functionDescriptions"), limit=2)
        if descriptions:
            detail = "; ".join(descriptions)
        else:
            detail = _mapping_detail(raw, ("approvedName", "description"))
    elif kind == EvidenceKind.EXPRESSION.value:
        detail = _expression_detail(raw, measurements, context)
    elif kind == EvidenceKind.PATHWAY.value:
        detail = _pathway_detail(raw)
    elif kind == EvidenceKind.SAFETY.value:
        liabilities = raw.get("safetyLiabilities")
        detail = _nested_collection_detail(
            liabilities,
            ("event", "eventId", "datasource", "url"),
            fallback="safety-liability annotation returned",
        )
    elif kind == EvidenceKind.TRACTABILITY.value:
        detail = _nested_collection_detail(
            raw.get("tractability"),
            ("label", "modality", "value"),
            fallback="tractability annotation returned",
        )
    elif kind == EvidenceKind.INTERACTION.value:
        interactions = raw.get("interactions")
        if isinstance(interactions, Mapping):
            interactions = interactions.get("rows")
        detail = _nested_collection_detail(
            interactions,
            ("intA", "intB", "sourceDatabase", "score"),
            fallback="interaction annotation returned",
        )
    elif kind == EvidenceKind.PHENOTYPE.value:
        detail = _nested_collection_detail(
            raw.get("mousePhenotypes") or raw.get("phenotypes") or raw,
            ("label", "phenotype", "model", "effect"),
            fallback="model-organism phenotype annotation returned",
        )
    detail = detail or f"{_target_module_label(row)} annotation returned"
    identifier = f" (`{_md_code(target_id)}`)" if target_id and target_id != label else ""
    return (
        f"- **{_md_inline(label)}**{identifier} — "
        f"**{_md_inline(_target_module_label(row))}:** {_md_inline(detail)}"
        f"{_row_citation_suffix(row, registry)}"
    )


def _association_score(row: Mapping[str, Any]) -> float | None:
    measurements = _row_mapping(row, "measurements")
    raw = _row_raw(row)
    return _finite_float(measurements.get("association_score") or raw.get("score"))


def _disease_association_bullet(
    row: Mapping[str, Any],
    registry: PublicationCitationRegistry,
) -> str:
    target_id, target = _target_label(row)
    disease = _disease_label(row) or "disease not identified"
    score = _association_score(row)
    context = _row_mapping(row, "context")
    scope = _plain(context.get("association_scope"), 80)
    if not scope:
        module = _row_module(row)
        scope = "indirect" if module.endswith("indirect") else "direct"
    details = [f"{scope} association"]
    if score is not None:
        details.append(f"source association score={score:.4g}")
    identifier = f" (`{_md_code(target_id)}`)" if target_id and target_id != target else ""
    return (
        f"- **{_md_inline(target)}**{identifier} → **{_md_inline(disease)}**: "
        f"{', '.join(details)}{_row_citation_suffix(row, registry)}"
    )


def _association_composition_lines(
    rows: Sequence[Mapping[str, Any]],
    registry: PublicationCitationRegistry,
) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        measurements = _row_mapping(row, "measurements")
        target = _target_label(row)[1]
        disease = _disease_label(row)
        for field, label in (
            ("datasource_scores", "datasource scores"),
            ("datatype_scores", "datatype scores"),
        ):
            values = measurements.get(field)
            detail = _score_collection_detail(values)
            if not detail or (target, field) in seen:
                continue
            seen.add((target, field))
            pair = f"{target}–{disease}" if disease else target
            lines.append(
                f"- **{_md_inline(pair)} — {label}:** {_md_inline(detail)}"
                f"{_row_citation_suffix(row, registry)}"
            )
    return lines[:12]


def _drug_evidence_bullet(
    row: Mapping[str, Any],
    registry: PublicationCitationRegistry,
) -> str:
    drug = _drug_label(row) or "drug not identified"
    display_entity = drug
    module = _row_module(row)
    context = _row_mapping(row, "context")
    measurements = _row_mapping(row, "measurements")
    object_value = _entity(row, "object")
    other_label = _plain(object_value.get("label") or object_value.get("id"), 240)
    details: list[str] = []
    if "mechanism" in module:
        details.extend(
            _present_detail_values(
                context,
                (
                    ("action_type", "action"),
                    ("mechanism_of_action", "mechanism"),
                    ("target_name", "target"),
                ),
            )
        )
        category = "mechanism"
    elif "indication" in module or "clinical_target" in module:
        if other_label:
            details.append(f"disease={other_label}")
        details.extend(
            _present_detail_values(
                measurements,
                (("max_clinical_stage", "maximum clinical stage"),),
            )
        )
        category = "indication"
    elif "warning" in module:
        details.extend(
            _present_detail_values(
                context,
                (
                    ("warning_type", "warning"),
                    ("toxicity_class", "toxicity class"),
                    ("description", "description"),
                ),
            )
        )
        category = "warning"
    elif "adverse" in module:
        if other_label:
            details.append(f"event={other_label}")
        details.extend(
            _present_detail_values(
                measurements,
                (("count", "count"), ("log_lr", "log likelihood ratio")),
            )
        )
        category = "adverse-event signal"
    elif "pharmacogen" in module:
        details.extend(
            _present_detail_values(
                context,
                (
                    ("evidence_level", "evidence level"),
                    ("phenotype_text", "phenotype"),
                    ("pgx_category", "category"),
                ),
            )
        )
        category = "pharmacogenomics"
    elif _row_kind(row) == EvidenceKind.SAFETY.value:
        _target_id, display_entity = _target_label(row)
        details.append(
            _nested_collection_detail(
                _row_raw(row).get("safetyLiabilities"),
                ("event", "eventId", "datasource", "url"),
                fallback="safety-liability annotation returned",
            )
        )
        category = "target safety"
    elif _row_kind(row) == EvidenceKind.TRACTABILITY.value:
        _target_id, display_entity = _target_label(row)
        details.append(
            _nested_collection_detail(
                _row_raw(row).get("tractability"),
                ("label", "modality", "value"),
                fallback="tractability annotation returned",
            )
        )
        category = "target tractability"
    else:
        details.extend(
            _present_detail_values(
                measurements,
                (("maximum_clinical_stage", "maximum clinical stage"),),
            )
        )
        details.extend(
            _present_detail_values(context, (("drug_type", "drug type"),))
        )
        category = "drug profile"
    detail = "; ".join(details) or "annotation returned"
    return (
        f"- **{_md_inline(display_entity)} — {_md_inline(category)}:** {_md_inline(detail)}"
        f"{_row_citation_suffix(row, registry)}"
    )


def _row_citation_suffix(
    row: Mapping[str, Any],
    registry: PublicationCitationRegistry,
) -> str:
    payload = _row_payload(row)
    raw = _row_raw(row)
    context = _row_mapping(row, "context")
    pmids: list[str] = []
    for candidate in (
        row.get("pmid"),
        context.get("pmid"),
        raw.get("pubmedId"),
        raw.get("pmid"),
    ):
        pmid = _pmid(candidate)
        if pmid and pmid not in pmids:
            pmids.append(pmid)
    for mapping in (raw, payload):
        for key in ("literature", "pmids", "pubmedIds"):
            for candidate in _scalar_values(mapping.get(key), limit=8):
                pmid = _pmid(candidate)
                if pmid and pmid not in pmids:
                    pmids.append(pmid)
    for pmid in _nested_pmid_values(raw):
        if pmid not in pmids:
            pmids.append(pmid)
    numbers: list[int] = []
    for pmid in pmids[:3]:
        publication = publication_from_metadata(
            pmid,
            authors=raw.get("publicationFirstAuthor") or raw.get("authors"),
            title=raw.get("publicationTitle") or raw.get("title"),
            journal=raw.get("publicationJournal") or raw.get("journal"),
            publication_date=raw.get("publicationDate") or raw.get("publication_date"),
            doi=raw.get("doi"),
        )
        number = registry.register(publication) if publication is not None else None
        if number is not None and number not in numbers:
            numbers.append(number)
    return "" if not numbers else " " + " ".join(f"[{item}]" for item in numbers)


def _expression_detail(
    raw: Mapping[str, Any],
    measurements: Mapping[str, Any],
    context: Mapping[str, Any],
) -> str:
    tissues = raw.get("tissues")
    items: list[str] = []
    if isinstance(tissues, Sequence) and not isinstance(tissues, (str, bytes)):
        for tissue in tissues[:5]:
            if not isinstance(tissue, Mapping):
                continue
            label = _plain(
                tissue.get("label") or tissue.get("tissueLabel") or tissue.get("name"),
                160,
            )
            rna = tissue.get("rna")
            rna = rna if isinstance(rna, Mapping) else {}
            value = rna.get("value")
            unit = _plain(rna.get("unit"), 60)
            if label:
                items.append(
                    f"{label}{f'={_plain(value, 60)} {unit}'.rstrip() if value is not None else ''}"
                )
    biosample = _plain(context.get("biosample") or context.get("tissue"), 160)
    if not items and biosample:
        value = next(
            (
                measurements.get(key)
                for key in ("value", "expression", "rna_value")
                if measurements.get(key) is not None
            ),
            None,
        )
        items.append(f"{biosample}{f'={_plain(value, 60)}' if value is not None else ''}")
    return "; ".join(items) or "expression annotation returned"


def _pathway_detail(raw: Mapping[str, Any]) -> str:
    values = raw.get("pathways") or raw.get("geneOntology") or raw.get("go")
    return _nested_collection_detail(
        values,
        ("name", "label", "pathwayId", "id", "aspect"),
        fallback="pathway or ontology annotation returned",
    )


def _nested_collection_detail(
    value: Any,
    keys: Sequence[str],
    *,
    fallback: str,
) -> str:
    if isinstance(value, Mapping):
        values = value.get("rows")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        return fallback
    items: list[str] = []
    for item in values[:3]:
        if isinstance(item, Mapping):
            detail = _mapping_detail(item, keys)
            if detail:
                items.append(detail)
        elif item not in (None, ""):
            items.append(_plain(item, 240))
    return "; ".join(items) or fallback


def _mapping_detail(value: Mapping[str, Any], keys: Sequence[str]) -> str:
    parts: list[str] = []
    for key in keys:
        item = value.get(key)
        if item in (None, "", [], {}):
            continue
        if isinstance(item, bool):
            text = "yes" if item else "no"
        elif isinstance(item, Mapping):
            text = _plain(item.get("label") or item.get("name") or item.get("id"), 240)
        else:
            text = _plain(item, 240)
        if text:
            parts.append(f"{key.replace('_', ' ')}={text}")
    return ", ".join(parts)


def _score_collection_detail(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ""
    parts: list[str] = []
    for item in value[:6]:
        if not isinstance(item, Mapping):
            continue
        label = _plain(item.get("id") or item.get("label"), 160)
        score = _finite_float(item.get("score"))
        if label:
            parts.append(f"{label}{f'={score:.4g}' if score is not None else ''}")
    return ", ".join(parts)


def _present_detail_values(
    mapping: Mapping[str, Any],
    fields: Sequence[tuple[str, str]],
) -> list[str]:
    values: list[str] = []
    for key, label in fields:
        value = mapping.get(key)
        if value in (None, "", [], {}):
            continue
        values.append(f"{label}={_plain(value, 300)}")
    return values


def _text_values(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [_plain(item, 500) for item in value[:limit] if _plain(item, 500)]


def _scalar_values(value: Any, *, limit: int) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value[:limit])
    return [] if value in (None, "") else [value]


def _nested_pmid_values(value: Any, *, depth: int = 0) -> list[str]:
    """Collect only values carried by explicit PMID/literature fields."""

    if depth > 5:
        return []
    output: list[str] = []
    if isinstance(value, Mapping):
        for key, item in list(value.items())[:100]:
            normalized_key = re.sub(r"[^a-z]", "", str(key).casefold())
            if normalized_key in {"pmid", "pubmedid", "pmids", "pubmedids", "literature"}:
                for candidate in _scalar_values(item, limit=12):
                    pmid = _pmid(candidate)
                    if pmid and pmid not in output:
                        output.append(pmid)
            if isinstance(item, (Mapping, list, tuple)):
                for pmid in _nested_pmid_values(item, depth=depth + 1):
                    if pmid not in output:
                        output.append(pmid)
            if len(output) >= 12:
                break
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value[:30]:
            for pmid in _nested_pmid_values(item, depth=depth + 1):
                if pmid not in output:
                    output.append(pmid)
            if len(output) >= 12:
                break
    return output


def _md_inline(value: Any) -> str:
    text = _plain(value, 1_000)
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|])", r"\\\1", text)


def _mapping_evidence_records(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[EvidenceRecord, ...]:
    """Rehydrate serialized ledger rows for the scientific display aggregator."""

    records: list[EvidenceRecord] = []
    for index, row in enumerate(rows):
        try:
            kind = EvidenceKind(str(row.get("kind") or ""))
        except ValueError:
            continue
        payload = row.get("payload")
        payload = dict(payload) if isinstance(payload, Mapping) else {}
        parent_variants = row.get("parent_variant_ids")
        parent_evidence = row.get("parent_evidence_ids")
        supports = row.get("supports")
        records.append(
            EvidenceRecord(
                evidence_id=_plain(row.get("evidence_id"), 500)
                or f"display-row-{index}",
                kind=kind,
                dataset=_plain(row.get("dataset"), 300) or kind.value,
                source=_plain(row.get("source"), 500) or "Open Targets Platform",
                data_release=_plain(row.get("data_release"), 120) or "not reported",
                payload=payload,
                variant_id=_plain(row.get("variant_id"), 200) or None,
                gene_id=_plain(row.get("gene_id"), 200) or None,
                study_id=_plain(row.get("study_id"), 300) or None,
                pmid=_pmid(row.get("pmid")) or None,
                credible_set_id=_plain(row.get("credible_set_id"), 300) or None,
                parent_variant_ids=(
                    tuple(str(item) for item in parent_variants)
                    if isinstance(parent_variants, (list, tuple))
                    else ()
                ),
                parent_evidence_ids=(
                    tuple(str(item) for item in parent_evidence)
                    if isinstance(parent_evidence, (list, tuple))
                    else ()
                ),
                tissue=_plain(row.get("tissue"), 300) or None,
                supports=supports if isinstance(supports, bool) else None,
                provenance_complete=row.get("provenance_complete") is not False,
            )
        )
    return tuple(records)


def _biological_context_markdown(
    context: BiologicalContext,
    genes: Sequence[GeneEvidence],
) -> str:
    """Display only the phenotype context users need to interpret a ranking."""

    if not context.is_defined:
        return (
            "- Key tissues: Not defined for this phenotype.\n"
            "- Key cell types: Not defined for this phenotype."
        )

    tissues = _natural_list(list(dict.fromkeys(_plain(item, 200) for item in (*context.organs, *context.tissues))))
    cells = _natural_list([_plain(item, 200) for item in context.cell_types])
    lines: list[str] = []
    if tissues:
        lines.append(f"- Key tissues: {tissues}.")
    else:
        lines.append("- Key tissues: Not defined for this phenotype.")
    if cells:
        lines.append(f"- Key cell types: {cells}.")
    else:
        lines.append("- Key cell types: Not defined for this phenotype.")
    return "\n".join(lines)


def _structured_gene_summary(
    genes: Sequence[GeneEvidence],
    *,
    trait: str,
    prioritize: bool,
    ranking_unavailable: bool,
    positional_only: bool,
    citation_resolver: Any = None,
) -> str:
    """Build a concise total-before-details comparison from visible evidence."""

    if not genes:
        task = "candidate-gene ranking" if prioritize else "variant-to-gene association"
        return (
            f"GWASMind's integrated assessment found no evidence-backed {task} at the "
            f"selected locus for **{trait}**. This is a database-coverage result, not evidence "
            "that the locus has no biologically relevant gene."
        )

    names = [f"**{gene.label}**" for gene in genes]
    if positional_only:
        if prioritize and not ranking_unavailable:
            return (
                f"The returned ranking retains {_natural_list(names)} in that order for **{trait}**. "
                "No molecular variant-to-gene or high-impact coding/splice support is available "
                "in their displayed evidence; the original order is retained with this limitation."
            )
        return (
            "GWASMind's integrated assessment found no molecular variant-to-gene "
            f"link at the selected locus for **{trait}**; {_natural_list(names)} are reported "
            "as positional candidates rather than as associated or causal genes."
        )

    if ranking_unavailable:
        citations = _summary_gene_citations(genes, citation_resolver)
        return (
            "GWASMind's integrated assessment identified direct locus-linked evidence for "
            f"{_natural_list(names)} at the selected locus for **{trait}**, but the evidence "
            f"did not support a validated candidate-gene ordering{citations}."
        )

    lead = genes[0]
    support: list[str] = []
    if lead.coding:
        support.append("high-impact coding/splice annotation")
    if lead.molecular_qtl:
        qtl_types = tuple(
            dict.fromkeys(_display_qtl_label(item.subtype) for item in lead.molecular_qtl)
        )
        tissues = tuple(
            dict.fromkeys(item.tissue for item in lead.molecular_qtl if item.tissue)
        )
        qtl_text = _natural_list(list(qtl_types))
        if tissues:
            qtl_text += " in " + _natural_list([str(item) for item in tissues[:3]])
        support.append(qtl_text)
    if lead.colocalizations:
        support.append("phenotype-linked colocalisation")
    if lead.enhancer_gene:
        support.append("enhancer-to-gene evidence")
    if lead.expression:
        support.append("measured expression")
    if lead.function.descriptions:
        support.append("functional annotation")
    if lead.function.gene_ontology:
        support.append("Gene Ontology annotation")
    if lead.function.pathways:
        support.append("pathway annotation")
    support_text = _natural_list(support) or "the displayed gene-level annotations"
    opening = (
        f"GWASMind's integrated assessment prioritizes **{lead.label}** as the leading "
        f"candidate at the selected locus for **{trait}**, supported by {support_text}"
        if prioritize
        else "GWASMind's integrated assessment links the selected locus for "
        f"**{trait}** most directly to **{lead.label}** through {support_text}"
    )
    if prioritize and not support:
        opening = (
            f"The returned ranking places **{lead.label}** first at the selected locus for **{trait}**; "
            "no gene-level support is available in its displayed evidence"
        )
    opening += _summary_gene_citations((lead,), citation_resolver) + "."

    coloc_studies = [
        {
            "trait": item.trait,
            "variant_id": item.qtl_study_locus_id or lead.gene_id,
            "study_id": item.other_study_id,
            "member_p": None,
        }
        for item in lead.colocalizations
        if item.trait
    ]
    phenotype_sentence = _phenotype_overview(coloc_studies) if coloc_studies else ""
    if phenotype_sentence:
        phenotype_sentence = phenotype_sentence.rstrip(".")
        phenotype_sentence += _summary_trace_citations(
            (item.trace for item in lead.colocalizations), citation_resolver
        ) + "."

    comparison = ""
    if len(genes) > 1:
        alternatives = genes[1:]
        direct = [
            gene.label
            for gene in alternatives
            if gene.molecular_qtl or gene.colocalizations or gene.enhancer_gene or gene.coding
        ]
        contextual = [gene.label for gene in alternatives if gene.label not in direct]
        clauses: list[str] = []
        if direct:
            verb = "has" if len(direct) == 1 else "have"
            convergence_verb = "shows" if len(direct) == 1 else "show"
            clauses.append(
                f"{_natural_list([f'**{item}**' for item in direct])} also {verb} direct "
                f"locus-linked evidence but {convergence_verb} less convergence in the "
                "displayed modules"
                + _summary_gene_citations(
                    tuple(gene for gene in alternatives if gene.label in direct),
                    citation_resolver,
                    direct_only=True,
                )
            )
        if contextual:
            verb = "has" if len(contextual) == 1 else "have"
            clauses.append(
                f"{_natural_list([f'**{item}**' for item in contextual])} {verb} no molecular "
                "variant-to-gene or high-impact coding/splice support in the displayed evidence"
            )
        if clauses:
            comparison = "; ".join(clauses) + "."

    return " ".join(item for item in (opening, phenotype_sentence, comparison) if item)


def _summary_gene_citations(
    genes: Sequence[GeneEvidence],
    citation_resolver: Any,
    *,
    direct_only: bool = False,
) -> str:
    """Cite representative support for gene-level claims in the Summary."""

    traces: list[EvidenceTrace] = []
    for gene in genes:
        if gene.coding:
            traces.append(gene.coding[0].trace)
        if gene.molecular_qtl:
            traces.append(gene.molecular_qtl[0].trace)
        if gene.colocalizations:
            traces.append(gene.colocalizations[0].trace)
        if gene.enhancer_gene:
            traces.append(gene.enhancer_gene[0].trace)
        if not direct_only and gene.expression and gene.expression[0].measurements:
            traces.append(gene.expression[0].measurements[0].trace)
        if not direct_only and gene.function.descriptions:
            traces.append(gene.function.descriptions[0].trace)
        elif not direct_only and gene.function.pathways:
            traces.append(gene.function.pathways[0].trace)
    return _summary_trace_citations(traces, citation_resolver)


def _summary_trace_citations(traces: Any, citation_resolver: Any) -> str:
    if citation_resolver is None:
        return ""
    markers: list[str] = []
    for trace in traces:
        resolved = citation_resolver(trace)
        for marker in re.findall(
            r"\[?\[\d+\]\]?(?:\(https://[^)\s]+\))?", str(resolved or "")
        ):
            if marker not in markers:
                markers.append(marker)
            if len(markers) >= 5:
                break
        if len(markers) >= 5:
            break
    return (" " + " ".join(markers)) if markers else ""


def _display_qtl_label(value: str) -> str:
    normalized = str(value or "qtl").casefold()
    return {
        "eqtl": "eQTL",
        "pqtl": "pQTL",
        "sqtl": "sQTL",
        "tuqtl": "tuQTL",
        "sceqtl": "single-cell eQTL",
        "scpqtl": "single-cell pQTL",
        "scsqtl": "single-cell sQTL",
        "sctuqtl": "single-cell tuQTL",
    }.get(normalized, str(value or "QTL"))


def _gene_publication_resolver(
    records: Sequence[EvidenceRecord],
    registry: PublicationCitationRegistry,
):
    """Return a trace resolver with role-aware QTL/coloc provenance."""

    by_id = {record.evidence_id: record for record in records}
    enriched_publications = {}

    def remember(publication):
        if publication is None:
            return
        previous = enriched_publications.get(publication.pmid)
        def completeness(item):
            return sum(bool(getattr(item, key, None))
                       for key in ("authors", "title", "journal", "year"))
        if completeness(publication) > completeness(previous):
            enriched_publications[publication.pmid] = publication

    for record in records:
        if record.kind is EvidenceKind.COLOCALIZATION:
            # A coloc record has two publication owners. Generic right-study
            # metadata must never fill the left QTL PMID just because it occurs
            # on the same row.
            for publication in parse_colocalization_publication_lineage(record.to_dict()).publications():
                remember(publication)
            continue
        raw = _evidence_upstream_raw(record)
        # Literature enrichment is attached to the normalized payload. It must
        # not be hidden by an older, sparse upstream.raw record for the same PMID.
        metadata = {**raw, **{key: value for key, value in record.payload.items()
                            if key in {"publicationFirstAuthor", "authors", "publicationTitle", "title",
                                       "publicationJournal", "journal", "publicationDate", "year", "doi"}
                            and value not in (None, "")}}
        publication = publication_from_metadata(
            record.pmid or raw.get("pubmedId") or raw.get("pmid"),
            authors=metadata.get("publicationFirstAuthor") or metadata.get("authors"),
            title=metadata.get("publicationTitle") or metadata.get("title"),
            journal=metadata.get("publicationJournal") or metadata.get("journal"),
            publication_date=metadata.get("publicationDate") or metadata.get("year"),
            doi=metadata.get("doi"),
        )
        remember(publication)

    def register(publication: Any, numbers: list[int]) -> None:
        if len(numbers) >= 3:
            return
        if publication is not None:
            publication = enriched_publications.get(publication.pmid, publication)
        number = registry.register(publication)
        if number is not None and number not in numbers:
            numbers.append(number)

    def resolve(trace: EvidenceTrace) -> str | None:
        numbers: list[int] = []
        for evidence_id in trace.evidence_ids:
            record = by_id.get(evidence_id)
            if record is None:
                continue
            raw = _evidence_upstream_raw(record)
            if record.kind is EvidenceKind.COLOCALIZATION:
                lineage = parse_colocalization_publication_lineage(record.to_dict())
                for publication in lineage.publications():
                    register(publication, numbers)
                continue
            pmid = (
                record.pmid
                or raw.get("pubmedId")
                or raw.get("pmid")
            )
            publication = publication_from_metadata(
                pmid,
                authors=raw.get("publicationFirstAuthor") or raw.get("authors"),
                title=raw.get("publicationTitle"),
                journal=raw.get("publicationJournal"),
                publication_date=raw.get("publicationDate"),
            )
            if publication is None:
                publication = resolve_publication(
                    study_id=record.study_id or raw.get("studyId"),
                )
            register(publication, numbers)

        for study_id in trace.study_ids:
            register(resolve_publication(study_id=study_id), numbers)
        for pmid in trace.pmids:
            register(resolve_publication(pmid=pmid), numbers)
        # Datasource papers are provenance fallbacks.  Do not pile a generic
        # database paper onto a fact that already has a study-specific citation.
        if not numbers:
            for datasource_id in trace.datasource_ids:
                register(resolve_datasource_publication(datasource_id), numbers)
        markers: list[str] = []
        entries = registry.entries
        for number in numbers[:3]:
            publication = entries[number - 1].publication
            markers.append(f"[[{number}]]({publication.pubmed_url})")
        return " ".join(markers) or None

    return resolve


def _evidence_upstream_raw(record: EvidenceRecord) -> Mapping[str, Any]:
    upstream = record.payload.get("upstream")
    if isinstance(upstream, Mapping):
        raw = upstream.get("raw")
        if isinstance(raw, Mapping):
            return raw
    return record.payload


def validate_database_answer(
    content: str,
    deterministic_answer: str,
    *,
    intent: ChatIntent = ChatIntent.GENERAL,
) -> bool:
    """Return whether a model answer satisfies the grounded display contract."""

    return database_answer_validation(
        content,
        deterministic_answer,
        intent=intent,
    ).valid


def database_answer_validation(
    content: str,
    deterministic_answer: str,
    *,
    intent: ChatIntent = ChatIntent.GENERAL,
) -> DatabaseAnswerValidation:
    """Validate a model rendering against the deterministic evidence answer.

    The deterministic answer is the authority for facts and references.  This
    validator intentionally prefers a deterministic fallback over accepting a
    fluent answer that adds a gene, number, publication, URL, or internal tool
    status that was not present in that authority.
    """

    text = str(content or "").strip()
    issues: list[str] = []
    headings = ["**Summary**", "**Evidence**", "**References**"]
    positions = [text.find(heading) for heading in headings]
    if any(position < 0 for position in positions):
        issues.append("missing_required_heading")
    elif positions != sorted(positions):
        issues.append("heading_order")

    body, reference_text = _answer_sections(text)
    deterministic_body, deterministic_reference_text = _answer_sections(deterministic_answer)
    supplied_references = _reference_definitions(deterministic_reference_text)
    rendered_references = _reference_definitions(reference_text)
    rendered_reference_numbers = re.findall(r"(?m)^\s*\[(\d+)\]\s+", reference_text)
    if len(rendered_reference_numbers) != len(set(rendered_reference_numbers)):
        issues.append("duplicate_reference_definition")
    allowed = set(supplied_references)
    used = set(re.findall(r"\[(\d+)\]", body))
    required_inline = set(re.findall(r"\[(\d+)\]", deterministic_body))
    if not used.issubset(allowed):
        issues.append("unsupported_reference")
    if required_inline and not required_inline.issubset(used):
        issues.append("missing_inline_reference")
    supplied_links = dict(re.findall(r"\[(\d+)\]\(([^\s)]+)\)", deterministic_body))
    for number, url in re.findall(r"\[(\d+)\]\(([^\s)]+)\)", body):
        if number in supplied_links and url != supplied_links[number]:
            issues.append("reference_link_changed")
    if rendered_references != supplied_references:
        issues.append("reference_definition_changed")

    allowed_urls = set(_answer_urls(deterministic_answer))
    rendered_urls = set(_answer_urls(text))
    if not rendered_urls.issubset(allowed_urls):
        issues.append("unsupported_reference_url")
    allowed_pmids = set(_answer_pmids(deterministic_answer))
    rendered_pmids = set(_answer_pmids(text))
    if not rendered_pmids.issubset(allowed_pmids):
        issues.append("unsupported_pmid")

    internal_statuses = {item.value for item in DecisionStatus}
    internal_statuses.update(
        {
            "NO_EXTERNAL_MATCH",
            "NO_DATA",
            "POLICY_BLOCKED",
            "SERVICE_UNAVAILABLE",
        }
    )
    forbidden_status = re.compile(
        r"(?:\b(?:"
        + "|".join(re.escape(item) for item in sorted(internal_statuses))
        + r")\b|\*\*Run status:\*\*|\*\*Question-specific evidence:\*\*)",
        re.IGNORECASE,
    )
    if forbidden_status.search(text):
        issues.append("internal_status_exposed")

    allowed_numbers = set(_answer_numbers(deterministic_answer))
    rendered_numbers = set(_answer_numbers(text))
    if not rendered_numbers.issubset(allowed_numbers):
        issues.append("unsupported_numeric_claim")

    allowed_gene_ids = set(re.findall(r"\bENSG\d{6,}\b", deterministic_answer))
    rendered_gene_ids = set(re.findall(r"\bENSG\d{6,}\b", text))
    if not rendered_gene_ids.issubset(allowed_gene_ids):
        issues.append("unsupported_gene_id")
    if allowed_gene_ids and not allowed_gene_ids.issubset(rendered_gene_ids):
        issues.append("required_gene_omitted")

    if intent in {ChatIntent.GENE_ASSOCIATIONS, ChatIntent.GENE_PRIORITIZATION}:
        allowed_symbols = _uppercase_scientific_tokens(deterministic_answer)
        rendered_symbols = _uppercase_scientific_tokens(text)
        generic_symbols = {
            "API",
            "DNA",
            "GWAS",
            "L2G",
            "MHC",
            "PMID",
            "QTL",
            "REST",
            "RNA",
            "SNP",
        }
        if not rendered_symbols.issubset(allowed_symbols | generic_symbols):
            issues.append("unsupported_gene_or_symbol")

    if intent is ChatIntent.GENE_ASSOCIATIONS:
        positive_ranking = re.compile(
            r"(?:\b(?:evidence|analysis|we)\s+(?:prioriti[sz]es?|ranks?)\b|"
            r"\b(?:is|are)\s+(?:the\s+)?(?:causal|prioritized|top-ranked)\s+gene\b|"
            r"\bthe\s+(?:causal|top-ranked)\s+gene\s+(?:is|are)\b)",
            re.IGNORECASE,
        )
        if positive_ranking.search(text):
            issues.append("association_conflated_with_ranking")

    if "Source GWAS" in deterministic_answer:
        phenotype_match = re.search(r"(?m)^- \*\*Phenotype:\*\*\s+(.+?)\s*$", deterministic_answer)
        if phenotype_match and phenotype_match.group(1).casefold() not in text.casefold():
            issues.append("source_phenotype_omitted")
        unsupported_absence = re.compile(
            r"\b(?:no\s+(?:specific\s+)?traits?\s+(?:can\s+be|are|were)\s+"
            r"associated|cannot\s+(?:associate|link)\s+(?:these\s+)?variants?\s+"
            r"(?:with|to)\s+(?:any\s+)?traits?)\b",
            re.IGNORECASE,
        )
        if unsupported_absence.search(text):
            issues.append("source_gwas_contradicted")

    if intent is ChatIntent.VARIANT_DESCRIPTION:
        legacy_sections = re.compile(
            r"(?:\*\*Source GWAS\*\*|\*\*Representative selected variants\*\*|"
            r"\*\*Open Targets trait lookup\*\*)",
            re.IGNORECASE,
        )
        if legacy_sections.search(text):
            issues.append("verbose_variant_template")
        source_requirements = _uploaded_gwas_requirements(deterministic_body)
        if source_requirements is not None:
            phenotype, significant, selected, p_range, negative_control = (
                source_requirements
            )
            rendered_grounded = _grounded_text(body)
            if _grounded_text(phenotype) not in rendered_grounded:
                issues.append("source_phenotype_omitted")
            if (
                f"{significant} of {selected}" not in rendered_grounded
                or "genome-wide significant" not in rendered_grounded
            ):
                issues.append("source_gwas_signal_omitted")
            if p_range and _grounded_text(p_range) not in rendered_grounded:
                issues.append("source_p_value_range_omitted")
            if "external annotations" not in rendered_grounded:
                issues.append("external_annotations_not_separated")
            if negative_control and not all(
                term in rendered_grounded
                for term in (
                    "does not support",
                    "external associations",
                    "uploaded analysis",
                )
            ):
                issues.append("negative_control_warning_omitted")
        required_variants = set(
            re.findall(
                r"\b(?:[1-9]|1\d|2[0-2]|X|Y|MT)_\d+_[ACGT]+_[ACGT]+\b",
                deterministic_body,
                re.IGNORECASE,
            )
        )
        rendered_variants = set(
            re.findall(
                r"\b(?:[1-9]|1\d|2[0-2]|X|Y|MT)_\d+_[ACGT]+_[ACGT]+\b",
                body,
                re.IGNORECASE,
            )
        )
        if not required_variants.issubset(rendered_variants):
            issues.append("displayed_variant_omitted")
        required_categories, required_traits = _variant_phenotype_requirements(deterministic_body)
        rendered_summary = body.split("**Evidence**", 1)[0]
        grounded_summary = _grounded_text(rendered_summary)
        category_terms_present = all(
            all(_grounded_text(term) in grounded_summary for term in category.split("/"))
            for category in required_categories
        )
        trait_labels_present = all(
            _grounded_text(trait) in grounded_summary for trait in required_traits
        )
        if (required_categories or required_traits) and (
            (required_categories and "phenotyp" not in grounded_summary)
            or not category_terms_present
            or not trait_labels_present
        ):
            issues.append("phenotype_overview_omitted")

    return DatabaseAnswerValidation(not issues, tuple(dict.fromkeys(issues)))


def _answer_sections(text: str) -> tuple[str, str]:
    marker = "**References**"
    if marker not in text:
        return text, ""
    body, references = text.split(marker, 1)
    return body, references


def _variant_phenotype_requirements(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract stable grounded labels while allowing a model to reword the sentences."""

    normalized = " ".join(str(text or "").split())
    category_match = re.search(r"The retrieved associations span (.+?) phenotypes\.", normalized)
    categories = (
        tuple(re.findall(r"\*\*(.+?)\*\*", category_match.group(1))) if category_match else ()
    )
    trait_match = re.search(
        r"Representative reported traits include (.+?)\.",
        normalized,
    )
    traits = tuple(re.findall(r"\*\*(.+?)\*\*", trait_match.group(1))) if trait_match else ()
    return categories, traits


def _uploaded_gwas_requirements(
    text: str,
) -> tuple[str, str, str, str, bool] | None:
    """Extract the source-first facts a model rendering may reword but not omit."""

    normalized = " ".join(str(text or "").split())
    match = re.search(
        r"In the uploaded \*\*(.+?)\*\* GWAS, \*\*(\d+) of (\d+) selected "
        r"variants? (?:is|are) genome-wide significant\*\* at \*P\* ≤ .+? "
        r"\(selected \*P\*-value range: (.+?)\)\.",
        normalized,
    )
    if match is None:
        return None
    phenotype, significant, selected, p_range = match.groups()
    return (
        phenotype,
        significant,
        selected,
        p_range,
        int(significant) == 0,
    )


def _grounded_text(value: str) -> str:
    """Normalize prose and Markdown for grounded-label presence checks."""

    return " ".join(re.sub(r"[*_`]", "", str(value or "")).casefold().split())


def _reference_definitions(text: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*\[(\d+)\]\s+(.+?)\s*$", text):
        definitions[match.group(1)] = " ".join(match.group(2).split())
    return definitions


def _answer_urls(text: str) -> tuple[str, ...]:
    return tuple(
        value.rstrip(".,;)")
        for value in re.findall(r"https?://[^\s)>]+(?:\([^\s)>]*\)[^\s)>]*)?", text)
    )


def _answer_pmids(text: str) -> tuple[str, ...]:
    values = re.findall(r"\bPMID\s*:\s*([1-9]\d{0,11})\b", text, re.IGNORECASE)
    values.extend(re.findall(r"pubmed\.ncbi\.nlm\.nih\.gov/([1-9]\d{0,11})/", text))
    return tuple(values)


def _answer_numbers(text: str) -> tuple[str, ...]:
    values = re.findall(
        r"(?<![A-Za-z0-9_])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?![A-Za-z0-9_])",
        text,
        re.IGNORECASE,
    )
    return tuple(value.replace(",", "").casefold() for value in values)


def _uppercase_scientific_tokens(text: str) -> set[str]:
    return {
        value for value in re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\b", text) if len(value) >= 3
    }


def database_answer_repair_messages(
    *,
    question: str,
    invalid_answer: str,
    deterministic_answer: str,
    intent: ChatIntent,
) -> list[dict[str, str]]:
    """Build one bounded formatting repair request with no new fact generation."""

    del invalid_answer
    allowed = sorted(set(re.findall(r"(?m)^\[(\d+)\]\s", deterministic_answer)))
    intent_rule = (
        load_prompt("database_repair_variant_rule")
        if intent is ChatIntent.VARIANT_DESCRIPTION
        else ""
    )
    system = format_prompt(
        "database_repair_system",
        allowed_reference_numbers=", ".join(allowed) if allowed else "none",
        intent_rule=intent_rule,
        response_contract=database_response_contract(intent),
    )
    payload = format_prompt(
        "database_repair_user",
        question=str(question)[:4_000],
        deterministic_answer=str(deterministic_answer)[:18_000],
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": payload},
    ]


def selection_overlaps_ot_mhc(observations: Sequence[Mapping[str, Any]]) -> bool:
    return mappings_overlap_ot_mhc(observations)


_VARIANT_GENE_KINDS = frozenset(
    {"variant_annotation", "qtl", "colocalization", "regulatory", "l2g"}
)


def _gene_rows(
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        payload = row.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        subject = payload.get("subject")
        subject = subject if isinstance(subject, Mapping) else {}
        object_value = payload.get("object")
        object_value = object_value if isinstance(object_value, Mapping) else {}
        gene_id = _plain(
            row.get("gene_id")
            or (
                object_value.get("id")
                if str(object_value.get("type", "")).casefold() in {"target", "gene"}
                else None
            )
            or (
                subject.get("id")
                if str(subject.get("type", "")).casefold() in {"target", "gene"}
                else None
            ),
            120,
        )
        if not gene_id:
            continue
        label = _plain(
            object_value.get("label")
            or subject.get("label")
            or _nested_value(payload, "upstream", "raw", "approvedSymbol"),
            120,
        )
        measurements = payload.get("measurements")
        measurements = measurements if isinstance(measurements, Mapping) else {}
        context = payload.get("context")
        context = context if isinstance(context, Mapping) else {}
        grouped[gene_id].append(
            {
                "gene_id": gene_id,
                "label": label,
                "kind": _plain(row.get("kind"), 80),
                "dataset": _plain(row.get("dataset"), 160),
                "source": _plain(row.get("source"), 300),
                "variant_id": _plain(row.get("variant_id") or subject.get("id"), 160),
                "tissue": _plain(row.get("tissue") or context.get("biosample"), 300),
                "pmid": _pmid(row.get("pmid") or context.get("pmid")),
                "relation": _plain(payload.get("relation"), 120),
                "measurements": measurements,
                "payload": payload,
            }
        )
    return grouped


def _nested_value(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _is_variant_gene_link(row: Mapping[str, Any]) -> bool:
    return str(row.get("kind", "")) in _VARIANT_GENE_KINDS


def _is_ensembl_row(row: Mapping[str, Any]) -> bool:
    text = f"{row.get('source', '')} {row.get('dataset', '')}".casefold()
    return "ensembl" in text


def _ranked_gene_ids(result: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(result, Mapping):
        return ()
    ranked = result.get("ranked_genes")
    if not isinstance(ranked, list):
        return ()
    values: list[tuple[int, str]] = []
    for index, item in enumerate(ranked):
        if not isinstance(item, Mapping):
            continue
        gene_id = _plain(item.get("gene_id"), 120)
        if not gene_id:
            continue
        try:
            rank = int(item.get("rank", index + 1))
        except (TypeError, ValueError):
            rank = index + 1
        values.append((rank, gene_id))
    return tuple(gene_id for _, gene_id in sorted(values))


def _ordered_gene_ids(
    linked: Mapping[str, Sequence[Mapping[str, Any]]],
    ranked: Sequence[str],
) -> list[str]:
    values = [gene_id for gene_id in ranked if gene_id in linked]
    remaining = sorted(
        (gene_id for gene_id in linked if gene_id not in values),
        key=lambda gene_id: (
            -sum(
                str(row.get("kind", "")) in {"qtl", "colocalization", "regulatory", "l2g"}
                for row in linked[gene_id]
            ),
            -len(linked[gene_id]),
            _gene_display(gene_id, linked[gene_id]).casefold(),
        ),
    )
    values.extend(remaining)
    return values


def _gene_display(gene_id: str, rows: Sequence[Mapping[str, Any]]) -> str:
    labels = [_plain(row.get("label"), 120) for row in rows]
    label = next((item for item in labels if item and item != gene_id), "")
    return f"{label} (`{_md_code(gene_id)}`)" if label else f"`{_md_code(gene_id)}`"


def _gene_evidence_bullets(
    rows: Sequence[Mapping[str, Any]], citation_numbers: Mapping[str, int]
) -> list[str]:
    output: list[str] = []
    positional = [row for row in rows if _is_ensembl_row(row)]
    if positional:
        relation = (
            "overlaps"
            if any(row.get("relation") == "region_overlap" for row in positional)
            else "is near"
        )
        reference = citation_numbers.get("ensembl-overlap")
        suffix = f" [{reference}]" if reference is not None else ""
        output.append(
            f"- **Positional context:** the selected GRCh38 interval {relation} this gene; "
            f"this is candidate discovery, not association evidence.{suffix}"
        )

    consequences = [
        row for row in rows if row.get("kind") == "variant_annotation" and not _is_ensembl_row(row)
    ]
    if consequences:
        variants = sorted({row["variant_id"] for row in consequences if row.get("variant_id")})
        reference = citation_numbers.get("ot-variant")
        suffix = f" [{reference}]" if reference is not None else ""
        output.append(
            f"- **Variant consequence:** {len(consequences)} transcript-consequence record(s) "
            f"covering {len(variants)} selected variant(s).{suffix}"
        )

    labels = {
        "qtl": "Molecular QTL",
        "colocalization": "Colocalisation",
        "regulatory": "Regulatory / enhancer-to-gene",
    }
    for kind, label in labels.items():
        selected = [row for row in rows if row.get("kind") == kind]
        if not selected:
            continue
        tissues = sorted({row["tissue"] for row in selected if row.get("tissue")})
        variants = sorted({row["variant_id"] for row in selected if row.get("variant_id")})
        detail = [f"{len(selected)} record(s)"]
        if variants:
            detail.append(f"{len(variants)} selected variant(s)")
        if tissues:
            detail.append("tissues/cell types: " + ", ".join(tissues[:5]))
        metric = _first_measurement(selected)
        if metric:
            detail.append(metric)
        references = _ordered_reference_numbers(selected, citation_numbers)
        suffix = " " + " ".join(f"[{number}]" for number in references) if references else ""
        output.append(f"- **{label}:** {'; '.join(detail)}.{suffix}")

    profile_kinds = {
        "expression": "Expression",
        "function": "Function",
        "pathway": "Pathway / ontology",
        "literature": "Literature",
    }
    for kind, label in profile_kinds.items():
        selected = [row for row in rows if row.get("kind") == kind]
        if selected:
            references = []
            if kind == "literature":
                references = _ordered_reference_numbers(
                    selected,
                    citation_numbers,
                    limit=3,
                )
            suffix = " " + " ".join(f"[{number}]" for number in references) if references else ""
            output.append(
                f"- **{label} context:** {len(selected)} Open Targets record(s); these "
                "annotations contextualize a linked gene but do not independently link the "
                f"selected variants to it.{suffix}"
            )
    return output or [
        "- A gene identifier was returned, but no interpretable evidence field was available."
    ]


def _ordered_reference_numbers(
    rows: Sequence[Mapping[str, Any]],
    citation_numbers: Mapping[str, int],
    *,
    limit: int | None = None,
) -> list[int]:
    values: list[int] = []
    for row in rows:
        pmid = _pmid(row.get("pmid"))
        number = citation_numbers.get(f"pmid:{pmid}") if pmid else None
        if number is None or number in values:
            continue
        values.append(number)
        if limit is not None and len(values) >= limit:
            break
    return values


def _first_measurement(rows: Sequence[Mapping[str, Any]]) -> str:
    preferred = (
        ("posteriorProbability", "posterior probability"),
        ("h4", "H4"),
        ("clpp", "CLPP"),
        ("score", "score"),
        ("beta", "beta"),
        ("p_value", "P"),
    )
    for row in rows:
        measurements = row.get("measurements")
        if not isinstance(measurements, Mapping):
            continue
        for key, label in preferred:
            value = _finite_float(measurements.get(key))
            if value is not None:
                displayed = format_probability(value) if key in {"posteriorProbability", "h4", "clpp"} else f"{value:.4g}"
                return f"representative {label}={displayed}"
    return ""


def _gene_citation_registry(
    evidence: Sequence[Mapping[str, Any]],
    *,
    include_ensembl: bool,
    include_ot_variant: bool,
    include_mhc: bool,
) -> tuple[list[Citation], dict[str, int]]:
    citations: list[Citation] = []
    numbers: dict[str, int] = {}

    def add(key: str, markdown: str) -> None:
        if key in numbers:
            return
        number = len(citations) + 1
        numbers[key] = number
        citations.append(Citation(number, key, f"[{number}] {markdown}"))

    if include_ensembl:
        add(
            "ensembl-overlap",
            f"Ensembl REST API. [Overlap region endpoint]({ENSEMBL_OVERLAP_DOCS_URL}).",
        )
    if include_ot_variant:
        add(
            "ot-variant",
            "Open Targets Platform. "
            f"[Variant annotation and transcript consequences]({OPEN_TARGETS_VARIANT_DOCS_URL}).",
        )
    if include_mhc:
        add(
            "ot-mhc",
            "Open Targets Platform. "
            f"[Credible Set exclusion criteria]({OT_CREDIBLE_SET_DOCS_URL}).",
        )
    rows_by_gene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unscoped_rows: list[Mapping[str, Any]] = []
    for row in evidence:
        if not _pmid(row.get("pmid")):
            continue
        gene_id = _plain(row.get("gene_id"), 120)
        (rows_by_gene[gene_id] if gene_id else unscoped_rows).append(row)

    citation_kind_order = (
        "colocalization",
        "qtl",
        "regulatory",
        "variant_annotation",
    )

    def representative_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """Keep one distinct publication per evidence layer before filling spare slots."""

        selected: list[Mapping[str, Any]] = []
        seen_pmids: set[str] = set()
        for kind in citation_kind_order:
            candidates = [row for row in rows if str(row.get("kind", "")) == kind]
            candidates.sort(key=_gene_citation_row_key)
            for row in candidates:
                pmid = _pmid(row.get("pmid"))
                if pmid and pmid not in seen_pmids:
                    selected.append(row)
                    seen_pmids.add(pmid)
                    break
        if len(selected) < MAX_REFERENCES_PER_GENE:
            eligible = [row for row in rows if str(row.get("kind", "")) in citation_kind_order]
            for row in sorted(eligible, key=_gene_citation_row_key):
                pmid = _pmid(row.get("pmid"))
                if not pmid or pmid in seen_pmids:
                    continue
                selected.append(row)
                seen_pmids.add(pmid)
                if len(selected) >= MAX_REFERENCES_PER_GENE:
                    break
        return selected[:MAX_REFERENCES_PER_GENE]

    rows_by_gene = {gene_id: representative_rows(rows) for gene_id, rows in rows_by_gene.items()}
    unscoped_rows = representative_rows(unscoped_rows)

    balanced_rows: list[Mapping[str, Any]] = []
    grouped = list(rows_by_gene.values())
    index = 0
    while grouped and index < MAX_REFERENCES_PER_GENE and len(balanced_rows) < MAX_GENE_REFERENCES:
        retained: list[list[Mapping[str, Any]]] = []
        for rows in grouped:
            if index < len(rows):
                balanced_rows.append(rows[index])
                retained.append(rows)
            if len(balanced_rows) >= MAX_GENE_REFERENCES:
                break
        grouped = retained
        index += 1
    balanced_rows.extend(unscoped_rows)

    for row in balanced_rows:
        if len(citations) >= MAX_GENE_REFERENCES:
            break
        pmid = _pmid(row.get("pmid"))
        if not pmid:
            continue
        payload = row.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        raw = _nested_value(payload, "upstream", "raw")
        raw = raw if isinstance(raw, Mapping) else {}
        author = _plain(raw.get("publicationFirstAuthor"), 300)
        title = _plain(raw.get("publicationTitle"), 1_000)
        journal = _plain(raw.get("publicationJournal"), 300)
        publication_date = _plain(raw.get("publicationDate"), 40)
        pieces = [item.rstrip(". ") for item in (author, title, journal) if item]
        label = ". ".join(pieces)
        if not label:
            label = "Source-study metadata unavailable"
            if publication_date:
                label += f" ({publication_date[:4]})"
        add(
            f"pmid:{pmid}",
            f"{label}. [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)",
        )
    return citations, numbers


def _gene_citation_row_key(row: Mapping[str, Any]) -> tuple[bool, str, str]:
    payload = row.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    raw = _nested_value(payload, "upstream", "raw")
    raw = raw if isinstance(raw, Mapping) else {}
    has_metadata = any(
        _plain(raw.get(key), 1_000)
        for key in (
            "publicationTitle",
            "publicationFirstAuthor",
            "publicationJournal",
        )
    )
    return (
        not has_metadata,
        _plain(raw.get("traitFromSource"), 500).casefold(),
        _pmid(row.get("pmid")),
    )


def _normalize_observation(item: Mapping[str, Any]) -> dict[str, Any]:
    identifier = _plain(item.get("variant_id"), 160)
    chromosome = _plain(item.get("chromosome"), 16).upper().removeprefix("CHR")
    try:
        position = int(item.get("position"))
    except (TypeError, ValueError, OverflowError):
        position = None
    try:
        raw_p_value = item.get("p_value")
        p_value = None if isinstance(raw_p_value, bool) else float(raw_p_value)
    except (TypeError, ValueError, OverflowError):
        p_value = None
    if p_value is not None and (not math.isfinite(p_value) or not 0 < p_value <= 1):
        p_value = None
    return {
        "variant_id": identifier,
        "chromosome": chromosome,
        "position": position,
        "p_value": p_value,
    }


def _evidence_rows(result: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if result is None:
        return []
    rows = result.get("evidence")
    if not isinstance(rows, list):
        rows = result.get("evidence_ledger")
    if not isinstance(rows, list):
        return []
    return [item for item in rows if isinstance(item, Mapping)]


def _study_associations(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in evidence:
        if str(row.get("kind") or "") not in {"study", "gwas"}:
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        upstream = payload.get("upstream")
        raw = upstream.get("raw") if isinstance(upstream, Mapping) else None
        if not isinstance(raw, Mapping):
            raw = {}
        variant_id = _plain(
            row.get("variant_id") or raw.get("queryVariantId") or raw.get("variantId"),
            160,
        )
        trait = _plain(raw.get("traitFromSource"), 500)
        study_id = _plain(row.get("study_id") or raw.get("studyId"), 200)
        if not variant_id or not trait or not study_id:
            continue
        key = (variant_id, trait.casefold(), study_id.casefold())
        if key in seen:
            continue
        seen.add(key)
        member = raw.get("matchedLocusMember")
        member = member if isinstance(member, Mapping) else {}
        pmid = _pmid(row.get("pmid") or raw.get("pubmedId"))
        output.append(
            {
                "variant_id": variant_id,
                "trait": trait,
                "study_id": study_id,
                "study_type": _plain(raw.get("studyType"), 100),
                "lead_variant_id": _plain(raw.get("leadVariantId"), 160),
                "query_variant_role": _plain(raw.get("queryVariantRole"), 80),
                "posterior_probability": _finite_float(member.get("posteriorProbability")),
                "member_p": _mantissa_exponent(
                    member.get("pValueMantissa"), member.get("pValueExponent")
                ),
                "member_beta": _finite_float(member.get("beta")),
                "pmid": pmid,
                "citation_key": f"pmid:{pmid}" if pmid else "",
                "publication_title": _plain(raw.get("publicationTitle"), 1_000),
                "publication_author": _plain(raw.get("publicationFirstAuthor"), 300),
                "publication_journal": _plain(raw.get("publicationJournal"), 300),
                "publication_date": _plain(raw.get("publicationDate"), 40),
            }
        )
    return output


def _is_trait_study(item: Mapping[str, Any]) -> bool:
    """Keep phenotypic GWAS studies out of molecular-QTL display noise."""

    study_type = _plain(item.get("study_type"), 100).casefold()
    study_id = _plain(item.get("study_id"), 200).casefold()
    trait = _plain(item.get("trait"), 500)
    if "qtl" in study_type:
        return False
    if study_id.startswith(
        (
            "eqtl_",
            "pqtl_",
            "sqtl_",
            "tuqtl_",
            "scpqtl_",
            "sceqtl_",
            "fusion_",
        )
    ):
        return False
    return not re.search(r"\bENSG\d{6,}\b", trait, re.IGNORECASE)


def _phenotype_overview(
    studies: Sequence[Mapping[str, Any]],
    *,
    max_categories: int = 4,
    max_traits: int = 5,
    citation_numbers: Mapping[str, int] | None = None,
) -> str:
    """Summarize only retrieved trait labels, with stable category diversity.

    Category names are descriptive buckets rather than new Open Targets evidence. The
    representative labels are copied verbatim from ``traitFromSource``. A round-robin
    across categories prevents a large family of near-identical biobank endpoints from
    crowding all other phenotype classes out of the short summary.
    """

    records: dict[str, dict[str, Any]] = {}
    for item in studies:
        trait = _plain(item.get("trait"), 500)
        if not trait:
            continue
        key = trait.casefold()
        record = records.setdefault(
            key,
            {
                "trait": trait,
                "category": _trait_category(trait),
                "variants": set(),
                "studies": set(),
                "min_p": math.inf,
                "citations": set(),
            },
        )
        variant_id = _plain(item.get("variant_id"), 160)
        study_id = _plain(item.get("study_id"), 200)
        if variant_id:
            record["variants"].add(variant_id)
        if study_id:
            record["studies"].add(study_id)
        citation = (citation_numbers or {}).get(item.get("citation_key"))
        if citation is not None:
            record["citations"].add(citation)
        p_value = _finite_float(item.get("member_p"))
        if p_value is not None and p_value > 0:
            record["min_p"] = min(record["min_p"], p_value)
    if not records:
        return ""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records.values():
        grouped[str(record["category"])].append(record)

    category_priority = {label: index for index, (label, _) in enumerate(_TRAIT_CATEGORY_RULES)}

    def record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        trait = _plain(record.get("trait"), 500)
        noise = len(_TRAIT_LABEL_NOISE.findall(trait))
        min_p = record.get("min_p")
        return (
            noise,
            -len(record.get("variants") or ()),
            -len(record.get("studies") or ()),
            float(min_p) if isinstance(min_p, (int, float)) else math.inf,
            len(trait),
            trait.casefold(),
        )

    for values in grouped.values():
        values.sort(key=record_key)

    ordered_categories = sorted(
        grouped,
        key=lambda category: (
            category == "other",
            -len(grouped[category]),
            category_priority.get(category, len(category_priority)),
            category,
        ),
    )
    named_categories = [item for item in ordered_categories if item != "other"]
    category_sentence = ""
    if named_categories:
        labels = [_summary_emphasis(item) for item in named_categories[:max_categories]]
        category_sentence = f"The retrieved associations span {_natural_list(labels)} phenotypes. "
    elif len(records) > 1:
        category_sentence = "The retrieved associations span heterogeneous phenotypes. "

    target_count = min(max_traits, len(records))
    representatives: list[str] = []
    offsets = {category: 0 for category in ordered_categories}
    while len(representatives) < target_count:
        added = False
        for category in ordered_categories:
            offset = offsets[category]
            values = grouped[category]
            if offset >= len(values):
                continue
            representative = values[offset]
            label = _summary_emphasis(representative["trait"])
            if representative["citations"]:
                label += f" [{min(representative['citations'])}]"
            representatives.append(label)
            offsets[category] = offset + 1
            added = True
            if len(representatives) >= target_count:
                break
        if not added:
            break

    return (
        category_sentence
        + "Representative reported traits include "
        + _natural_list(representatives, separator="; ")
        + "."
    )


def _trait_category(trait: str) -> str:
    for label, pattern in _TRAIT_CATEGORY_RULES:
        if pattern.search(trait):
            return label
    return "other"


_PHENOTYPE_EQUIVALENCE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "coronary heart disease",
            "coronary artery disease",
            "chd",
            "cad",
            "ischaemic heart disease",
            "ischemic heart disease",
            "coronary atherosclerosis",
            "myocardial infarction",
            "acute myocardial infarction",
        }
    ),
    frozenset({"type 1 diabetes", "type 1 diabetes mellitus", "t1d"}),
    frozenset({"type 2 diabetes", "type 2 diabetes mellitus", "t2d"}),
    frozenset({"major depressive disorder", "major depression", "mdd"}),
)


def _phenotype_key(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", _plain(value, 500).casefold())
    return " ".join(text.split())


def _phenotypes_equivalent(left: str, right: str) -> bool:
    """Conservatively match a query phenotype to a returned study label.

    This intentionally avoids broad category matching: blood pressure or stroke must
    never be treated as a coronary-heart-disease record merely because both are
    cardiovascular traits.
    """

    left_key = _phenotype_key(left)
    right_key = _phenotype_key(right)
    if not left_key or not right_key:
        return False
    for group in _PHENOTYPE_EQUIVALENCE_GROUPS:
        if left_key in group:
            return right_key in group or any(
                alias in right_key for alias in group if len(alias) >= 5
            )
        if right_key in group:
            return left_key in group or any(
                alias in left_key for alias in group if len(alias) >= 5
            )
    return left_key == right_key or (
        len(left_key) >= 5
        and len(right_key) >= 5
        and (left_key in right_key or right_key in left_key)
    )


def _natural_list(values: Sequence[str], *, separator: str = ", ") -> str:
    items = [item for item in values if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{separator.join(items[:-1])}{separator}and {items[-1]}"


def _summary_emphasis(value: Any) -> str:
    return f"**{_plain(value, 500).replace('*', '')}**"


def _representative_trait_associations(
    studies: Sequence[Mapping[str, Any]],
    *,
    max_variants: int = 5,
    max_associations: int = 5,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Choose stable, trait-diverse variants before references are registered."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in studies:
        variant_id = _plain(item.get("variant_id"), 160)
        if variant_id:
            grouped[variant_id].append(dict(item))
    if not grouped:
        return []

    def association_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        p_value = item.get("member_p")
        return (
            p_value is None,
            float(p_value) if p_value is not None else math.inf,
            not bool(_pmid(item.get("pmid"))),
            _plain(item.get("trait"), 500).casefold(),
            _plain(item.get("study_id"), 200).casefold(),
        )

    candidate_traits = {
        variant_id: {
            _plain(item.get("trait"), 500).casefold()
            for item in items
            if _plain(item.get("trait"), 500)
        }
        for variant_id, items in grouped.items()
    }
    selected_ids: list[str] = []
    covered_traits: set[str] = set()
    remaining = set(grouped)
    while remaining and len(selected_ids) < max_variants:

        def variant_key(variant_id: str) -> tuple[Any, ...]:
            items = grouped[variant_id]
            traits = candidate_traits[variant_id]
            p_values = [
                float(item["member_p"]) for item in items if item.get("member_p") is not None
            ]
            return (
                len(traits - covered_traits),
                len(traits),
                sum(bool(_pmid(item.get("pmid"))) for item in items),
                -(min(p_values) if p_values else math.inf),
                variant_id,
            )

        chosen = max(remaining, key=variant_key)
        remaining.remove(chosen)
        selected_ids.append(chosen)
        covered_traits.update(candidate_traits[chosen])

    output: list[tuple[str, list[dict[str, Any]]]] = []
    for variant_id in selected_ids:
        ordered = sorted(grouped[variant_id], key=association_key)
        selected: list[dict[str, Any]] = []
        seen_traits: set[str] = set()
        seen_publications: set[str] = set()
        # First spread the display across independent studies/publications.  This
        # avoids filling a SNP block with five near-identical phenotypes from one
        # biobank paper when cross-trait evidence is available.
        for item in ordered:
            trait_key = _plain(item.get("trait"), 500).casefold()
            publication_key = (
                _pmid(item.get("pmid")) or _plain(item.get("study_id"), 200).casefold()
            )
            if not trait_key or trait_key in seen_traits or publication_key in seen_publications:
                continue
            selected.append(item)
            seen_traits.add(trait_key)
            seen_publications.add(publication_key)
            if len(selected) >= max_associations:
                break
        # When fewer than five independent publications exist, fill the remaining
        # slots with distinct traits from the strongest already represented studies.
        if len(selected) < max_associations:
            selected_keys = {
                (
                    _plain(item.get("trait"), 500).casefold(),
                    _plain(item.get("study_id"), 200).casefold(),
                )
                for item in selected
            }
            for item in ordered:
                trait_key = _plain(item.get("trait"), 500).casefold()
                key = (trait_key, _plain(item.get("study_id"), 200).casefold())
                if not trait_key or trait_key in seen_traits or key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(key)
                seen_traits.add(trait_key)
                if len(selected) >= max_associations:
                    break
        output.append((variant_id, selected))
    return output


def _variant_rs_ids(evidence: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in evidence:
        if str(row.get("kind") or "") != "variant_annotation":
            continue
        payload = row.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        raw = _nested_value(payload, "upstream", "raw")
        raw = raw if isinstance(raw, Mapping) else {}
        variant_id = _plain(row.get("variant_id") or raw.get("id"), 160)
        rs_ids = raw.get("rsIds")
        if not variant_id or not isinstance(rs_ids, Sequence) or isinstance(rs_ids, str):
            continue
        for value in rs_ids:
            rs_id = _plain(value, 80)
            if re.fullmatch(r"rs\d+", rs_id, re.IGNORECASE):
                values.setdefault(variant_id, rs_id)
                break
    return values


def _citation_registry(
    studies: Sequence[Mapping[str, Any]], *, include_mhc: bool,
    publication_enricher: Callable[[PublicationCitationRegistry], object] | None = None,
) -> tuple[list[Citation], dict[str, int]]:
    citations: list[Citation] = []
    numbers: dict[str, int] = {}
    registry = PublicationCitationRegistry()
    for item in studies:
        pmid = _pmid(item.get("pmid"))
        if not pmid:
            continue
        author = _plain(item.get("publication_author"), 300)
        title = _plain(item.get("publication_title"), 1_000)
        journal = _plain(item.get("publication_journal"), 300)
        publication_date = _plain(item.get("publication_date"), 40)
        year_match = re.search(r"\b(?:19|20)\d{2}\b", publication_date)
        registry.register(Publication(pmid,
            authors=(author if author.endswith("et al.") else f"{author} et al.") if author else None,
            title=title or None, journal=journal or None,
            year=int(year_match.group(0)) if year_match else None))
    _complete_reference_registry(registry, publication_enricher)
    for entry in registry.entries:
        publication = entry.publication
        numbers[publication.key] = entry.number
        prefix = "Publication metadata unavailable. " if not any((publication.authors, publication.title, publication.journal, publication.year)) else ""
        citations.append(Citation(entry.number, publication.key,
            f"[{entry.number}] {prefix}{publication.markdown()}"))
    if include_mhc:
        key = "ot-mhc"
        number = len(citations) + 1
        numbers[key] = number
        citations.append(
            Citation(
                number,
                key,
                f"[{number}] Open Targets Platform. "
                f"[Credible Set exclusion criteria]({OT_CREDIBLE_SET_DOCS_URL}).",
            )
        )
    return citations, numbers


def _mantissa_exponent(mantissa: Any, exponent: Any) -> float | None:
    value = _finite_float(mantissa)
    exponent_value = _finite_float(exponent)
    if value is None or exponent_value is None or not exponent_value.is_integer():
        return None
    try:
        result = value * (10.0**exponent_value)
    except OverflowError:
        return None
    return result if math.isfinite(result) and 0 < result <= 1 else None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _pmid(value: Any) -> str:
    match = re.fullmatch(
        r"(?:PMID\s*:\s*)?([1-9]\d{0,11})",
        str(value or "").strip(),
        re.IGNORECASE,
    )
    return match.group(1) if match else ""


def _region_text(
    observations: Sequence[Mapping[str, Any]], region: Mapping[str, Any] | None
) -> str:
    coordinates = [
        (str(item.get("chromosome") or ""), item.get("position"))
        for item in observations
        if item.get("chromosome") and isinstance(item.get("position"), int)
    ]
    chromosomes = {chromosome for chromosome, _ in coordinates}
    if len(chromosomes) == 1 and coordinates:
        chromosome = next(iter(chromosomes))
        positions = [int(position) for _, position in coordinates]
        return f"chr{chromosome}:{min(positions):,}-{max(positions):,}"
    if isinstance(region, Mapping):
        chromosome = _plain(region.get("chromosome"), 16).removeprefix("chr")
        try:
            start = int(region.get("start"))
            end = int(region.get("end"))
        except (TypeError, ValueError, OverflowError):
            return ""
        if chromosome and start > 0 and end >= start:
            return f"chr{chromosome}:{start:,}-{end:,}"
    return ""


def _format_p(value: float) -> str:
    return f"{value:.3e}"


_SUPERSCRIPT_DIGITS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _format_p_display(value: float) -> str:
    mantissa, exponent = f"{value:.3e}".split("e", 1)
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}×10{str(int(exponent)).translate(_SUPERSCRIPT_DIGITS)}"


def _plain(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _md_code(value: Any) -> str:
    return _plain(value, 200).replace("`", "")
