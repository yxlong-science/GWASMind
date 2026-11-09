from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .candidates import evidence_candidate_scope
from .configuration import parameter
from .models import EvidenceKind, EvidenceRecord, LocusQuery, analysis_variant_ids
from .prompting import load_prompt
from .publications import normalize_pmid

MAX_SYNTHESIS_EVIDENCE_RECORDS = parameter("projection.ranking_max_records")
MAX_SYNTHESIS_EVIDENCE_CHARS = parameter("projection.ranking_max_chars")
LARGE_CONTEXT_SYNTHESIS_EVIDENCE_RECORDS = parameter("projection.ranking_large_max_records")
LARGE_CONTEXT_SYNTHESIS_EVIDENCE_CHARS = parameter("projection.ranking_large_max_chars")
MAX_MODEL_PUBLICATIONS = parameter("projection.max_publications")
MAX_PUBLICATION_ABSTRACT_CHARS = parameter("projection.publication_abstract_chars")

# The retrieval ledger remains complete in local artifacts. These limits apply only to
# the compact ledger sent to the ranking model. Direct locus-to-gene channels receive a
# larger allowance than target-profile context, while still preventing thousands of E2G
# intervals or generic target papers from dominating prompt size and latency.
_DIRECT_RECORD_LIMITS_PER_CANDIDATE = {
    EvidenceKind.QTL: parameter("projection.direct_qtl_records_per_candidate"),
    EvidenceKind.COLOCALIZATION: parameter("projection.direct_colocalization_records_per_candidate"),
    EvidenceKind.REGULATORY: parameter("projection.direct_regulatory_records_per_candidate"),
}
_CONTEXT_RECORD_LIMITS_PER_CANDIDATE = {
    EvidenceKind.VARIANT_ANNOTATION: parameter("projection.context_variant_annotation_records_per_candidate"),
    EvidenceKind.EXPRESSION: parameter("projection.context_expression_records_per_candidate"),
    EvidenceKind.FUNCTION: parameter("projection.context_function_records_per_candidate"),
    EvidenceKind.PATHWAY: parameter("projection.context_pathway_records_per_candidate"),
    EvidenceKind.LITERATURE: parameter("projection.context_literature_records_per_candidate"),
    EvidenceKind.STUDY: parameter("projection.context_study_records_per_candidate"),
    EvidenceKind.GWAS: parameter("projection.context_gwas_records_per_candidate"),
    EvidenceKind.ONTOLOGY: parameter("projection.context_ontology_records_per_candidate"),
    EvidenceKind.TARGET_DISEASE: parameter("projection.context_target_disease_records_per_candidate"),
    EvidenceKind.DRUG: parameter("projection.context_drug_records_per_candidate"),
    EvidenceKind.SAFETY: parameter("projection.context_safety_records_per_candidate"),
    EvidenceKind.TRACTABILITY: parameter("projection.context_tractability_records_per_candidate"),
    EvidenceKind.INTERACTION: parameter("projection.context_interaction_records_per_candidate"),
    EvidenceKind.PHENOTYPE: parameter("projection.context_phenotype_records_per_candidate"),
    EvidenceKind.GENE_SET: parameter("projection.context_gene_set_records_per_candidate"),
}
_DEFAULT_CONTEXT_RECORD_LIMIT_PER_CANDIDATE = parameter("projection.default_context_records_per_candidate")
_SEVERE_CONSEQUENCES = frozenset(
    {
        "transcript_ablation",
        "splice_acceptor_variant",
        "splice_donor_variant",
        "stop_gained",
        "frameshift_variant",
        "stop_lost",
        "start_lost",
    }
)
_DIRECT_SCOPED_CODING_CONSEQUENCES = frozenset({"missense_variant"})


class SynthesisProjectionError(ValueError):
    """The evidence ledger cannot be projected without losing required coverage."""


_SYNTHESIS_KIND_ORDER = {
    kind: index
    for index, kind in enumerate(
        (
            EvidenceKind.COLOCALIZATION,
            EvidenceKind.QTL,
            EvidenceKind.L2G,
            EvidenceKind.REGULATORY,
            EvidenceKind.VARIANT_ANNOTATION,
            EvidenceKind.EXPRESSION,
            EvidenceKind.FUNCTION,
            EvidenceKind.PATHWAY,
            EvidenceKind.LITERATURE,
            EvidenceKind.GWAS,
            EvidenceKind.STUDY,
        )
    )
}
_RAW_SCALAR_FIELDS = (
    "approvedName",
    "approvedSymbol",
    "biotype",
    "geneId",
    "id",
    "qtlGeneId",
    "qtlGeneSymbol",
    "rightQtlGeneId",
    "rightQtlGeneSymbol",
    "traitFromSource",
    "rightTraitFromSource",
    "studyId",
    "studyLocusId",
    "studyType",
    "rightStudyId",
    "rightStudyLocusId",
    "rightStudyType",
    "leftStudyLocusId",
    "qtlStudyLocusId",
    "queriedStudyLocusId",
    "otherStudyLocusId",
    "leadVariantId",
    "queryVariantId",
    "queryVariantRole",
    "variantId",
    "overlapVariantId",
    "overlapVariantRole",
    "overlapChromosome",
    "overlapPosition",
    "exactVariantOverlap",
    "coverageScope",
    "regulatoryElementKey",
    "regulatoryElementKeySource",
    "chromosome",
    "start",
    "end",
    "biosampleId",
    "biosampleName",
    "biosampleFromSourceId",
    "datasourceId",
    "enhancerStudyId",
    "intervalId",
    "intervalType",
    "distanceToTss",
    "pubmedId",
    "rightPubmedId",
    "publicationDate",
    "publicationFirstAuthor",
    "publicationJournal",
    "publicationTitle",
    "rightPublicationDate",
    "rightPublicationFirstAuthor",
    "rightPublicationJournal",
    "rightPublicationTitle",
    "colocalisationMethod",
    "betaRatioSignAverage",
    "finemappingMethod",
    "confidence",
    "consequence",
    "transcriptConsequence",
    "mostSevereConsequence",
    "impact",
    "vepImpact",
    "high_impact",
    "is_high_impact",
    "isHighImpact",
    "numberColocalisingVariants",
    "l2g_score",
    "h3",
    "h4",
    "clpp",
    "score",
    "beta",
    "pValueMantissa",
    "pValueExponent",
)
_RAW_LIST_LIMITS = {
    "functionDescriptions": 3,
    "pathways": 5,
    "go": 5,
    "geneOntology": 5,
    "tissues": 5,
    "qualityControls": 5,
    "resourceScore": 8,
    "consequenceTerms": 8,
    "consequence_terms": 8,
    "transcriptConsequences": 3,
}
_MATCHED_MEMBER_FIELDS = (
    "variantId",
    "posteriorProbability",
    "pValueMantissa",
    "pValueExponent",
    "beta",
    "standardError",
    "logBF",
    "r2Overall",
    "is95CredibleSet",
    "is99CredibleSet",
)
def project_record_for_model(
    record: EvidenceRecord,
    *,
    relevant_tissues: tuple[str, ...] = (),
    expression_limit: int = parameter("projection.expression_rows"),
) -> EvidenceRecord:
    """Bound model context while preserving the full record in raw artifacts."""

    payload = dict(record.payload)
    tissue = record.tissue
    expression_rows = payload.get("tissues")
    expression_location = "payload"
    if not isinstance(expression_rows, list):
        upstream = payload.get("upstream")
        raw = upstream.get("raw") if isinstance(upstream, Mapping) else None
        expression_rows = raw.get("tissues") if isinstance(raw, Mapping) else None
        expression_location = "upstream.raw"
    if record.kind is EvidenceKind.EXPRESSION and isinstance(expression_rows, list):
        rows = list(expression_rows)
        kept, matched_labels = _compact_expression_rows(
            rows, relevant_tissues=relevant_tissues, limit=expression_limit
        )
        projection_metadata = {
            "tissues": kept,
            "tissueRecordCount": len(rows),
            "omittedTissueRecordCount": len(rows) - len(kept),
            "relevantTissuesRequested": list(relevant_tissues),
            "relevantTissueMatches": list(matched_labels),
        }
        if expression_location == "payload":
            payload.update(projection_metadata)
        else:
            upstream = dict(payload.get("upstream") or {})
            raw = dict(upstream.get("raw") or {})
            raw.update(projection_metadata)
            upstream["raw"] = raw
            payload["upstream"] = upstream
        if matched_labels:
            tissue = "; ".join(matched_labels)
    if record.kind is EvidenceKind.FUNCTION:
        for key, limit in (
            ("go", parameter("projection.function_go_rows")),
            ("pathways", parameter("projection.function_pathway_rows")),
            ("functionDescriptions", parameter("projection.function_description_rows")),
        ):
            value = payload.get(key)
            if isinstance(value, list) and len(value) > limit:
                payload[f"{key}RecordCount"] = len(value)
                payload[f"omitted{key[0].upper() + key[1:]}RecordCount"] = len(value) - limit
                payload[key] = value[:limit]
    return replace(record, payload=payload, tissue=tissue)


def build_synthesis_evidence_projection(
    query: LocusQuery,
    evidence: Sequence[EvidenceRecord],
    *,
    candidate_genes: Sequence[str],
    max_records: int = MAX_SYNTHESIS_EVIDENCE_RECORDS,
    max_chars: int = MAX_SYNTHESIS_EVIDENCE_CHARS,
    use_model_context: bool = False,
    allowed_evidence_ids: frozenset[str] | None = None,
    allowed_publication_pmids: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a diverse, traceable model ledger while retaining full artifacts elsewhere.

    The full permitted ``EvidenceRecord`` values remain available to deterministic
    validation and artifact storage.  This projection prevents a locus with thousands of
    repetitive credible-set rows from overflowing a provider context window.  Selection is
    round-robin across candidate genes and evidence kinds, phenotype/tissue relevant rows
    sort first, and every retained row keeps its exact ``evidence_id``.
    """

    if max_records < 1 or max_chars < 1:
        raise ValueError("synthesis projection limits must be positive")
    evidence_ids = [record.evidence_id for record in evidence]
    duplicate_ids = sorted(
        evidence_id for evidence_id, count in Counter(evidence_ids).items() if count > 1
    )
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise SynthesisProjectionError(
            "evidence ledger contains duplicate evidence_id values: " + preview
        )
    candidates = tuple(dict.fromkeys(str(item) for item in candidate_genes if str(item)))
    candidate_order = {gene_id: index for index, gene_id in enumerate(candidates)}
    eligible = [record for record in evidence if record.gene_id in candidate_order]
    publication_occurrences = [
        record for record in eligible if record.kind is EvidenceKind.LITERATURE
    ]
    unique_publication_pmids = {
        pmid
        for record in publication_occurrences
        if (pmid := _publication_record_pmid(record)) is not None
    }
    publication_ledger = build_publication_projection(
        eligible if allowed_publication_pmids is None else [record for record in eligible
            if _publication_record_pmid(record) in allowed_publication_pmids],
        query=query,
        max_publications=max(1, len(unique_publication_pmids)) if use_model_context else MAX_MODEL_PUBLICATIONS,
        max_chars=max(2_000, max_chars // 5),
    )
    publication_chars = len(
        json.dumps(
            publication_ledger,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    )
    selected_publication_pmids = {
        str(item["pmid"]) for item in publication_ledger if item.get("pmid")
    }
    record_char_budget = max_chars - publication_chars
    if record_char_budget < 1:
        raise SynthesisProjectionError(
            "publication projection consumed the complete model character budget"
        )
    grouped: dict[tuple[str, EvidenceKind], list[EvidenceRecord]] = defaultdict(list)
    for record in eligible:
        grouped[(str(record.gene_id), record.kind)].append(record)

    deduplication: dict[str, dict[str, Any]] = {}
    deduplicated_count = 0
    deduplicated_by_kind: Counter[str] = Counter()
    deduplicated_by_gene: dict[str, Counter[str]] = defaultdict(Counter)
    bounded_groups: list[tuple[tuple[str, EvidenceKind], list[EvidenceRecord]]] = []
    for key, records in grouped.items():
        unique_records, unique_metadata = _deduplicate_synthesis_records(
            records,
            query=query,
        )
        deduplicated_count += len(unique_records)
        deduplicated_by_kind[key[1].value] += len(unique_records)
        deduplicated_by_gene[key[0]][key[1].value] += len(unique_records)
        deduplication.update(unique_metadata)
        # A capacity retry may only remove previously visible scientific rows;
        # swapping in different, denser rows could exceed the context again.
        # Full-ledger deduplication and omission counts above stay unchanged.
        if allowed_evidence_ids is not None:
            unique_records = [record for record in unique_records if record.evidence_id in allowed_evidence_ids]
        if key[1] is EvidenceKind.LITERATURE:
            unique_records = [
                record
                for record in unique_records
                if (pmid := _publication_record_pmid(record)) is None
                or pmid in selected_publication_pmids
            ]
        # Retain a bounded, phenotype/tissue-aware sample per candidate and
        # evidence kind. The complete deduplicated inventory remains reported
        # in the summary and the full records remain in local artifacts.
        selected = (sorted(unique_records, key=lambda record: _synthesis_record_key(record, query))
                    if use_model_context else _bounded_candidate_kind_records(
                        unique_records, query=query, kind=key[1]))
        if selected:
            bounded_groups.append((key, selected))
    # Evidence-kind-major order prevents one candidate's many modules from
    # consuming the global budget before other candidates are considered.
    bounded_groups.sort(
        key=lambda item: (
            _SYNTHESIS_KIND_ORDER.get(item[0][1], len(_SYNTHESIS_KIND_ORDER)),
            candidate_order.get(item[0][0], len(candidate_order)),
        )
    )

    projected: list[dict[str, Any]] = []
    projected_ids: set[str] = set()
    used_chars = 0

    def append_record(record: EvidenceRecord, *, required: bool) -> bool:
        nonlocal used_chars
        if record.evidence_id in projected_ids:
            return True
        if len(projected) >= max_records:
            if required:
                raise SynthesisProjectionError(
                    "projection record limit cannot preserve one evidence row per "
                    "evidence-supported candidate gene"
                )
            return False
        compact = compact_record_for_synthesis(
            record,
            relevant_tissues=query.relevant_tissues,
            complete_scientific_lists=use_model_context,
        )
        metadata = deduplication.get(record.evidence_id)
        if metadata and int(metadata.get("represented_record_count", 1)) > 1:
            compact["deduplication"] = metadata
        encoded = json.dumps(
            compact,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        if used_chars + len(encoded) > record_char_budget:
            if required:
                raise SynthesisProjectionError(
                    "projection character budget cannot preserve one evidence row per "
                    "evidence-supported candidate gene"
                )
            return False
        projected.append(compact)
        projected_ids.add(record.evidence_id)
        used_chars += len(encoded)
        return True

    # Coverage pass 1: preserve one row from every direct locus-to-gene dimension
    # available for each candidate. Generic expression/function/literature rows
    # cannot crowd out QTL, colocalisation, E2G, or high-impact coding/splice
    # evidence.
    records_by_candidate: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for (gene_id, _kind), records in bounded_groups:
        records_by_candidate[gene_id].extend(records)
    for gene_id in candidates:
        candidate_records = records_by_candidate.get(gene_id, [])
        direct_by_kind: dict[EvidenceKind, list[EvidenceRecord]] = defaultdict(list)
        for record in candidate_records:
            if _is_direct_synthesis_record(record):
                direct_by_kind[record.kind].append(record)
        for kind in sorted(
            direct_by_kind,
            key=lambda item: _SYNTHESIS_KIND_ORDER.get(item, len(_SYNTHESIS_KIND_ORDER)),
        ):
            if kind is EvidenceKind.COLOCALIZATION and use_model_context:
                required = [
                    record
                    for record in direct_by_kind[kind]
                    if _is_model_required_colocalization(record)
                ]
                for record in required:
                    append_record(record, required=True)
                if required:
                    continue
            strongest = min(
                direct_by_kind[kind],
                key=lambda record: _synthesis_record_key(record, query),
            )
            append_record(strongest, required=True)

    # Coverage pass 2: a context-only candidate still receives one row so the
    # model can explicitly leave it unranked instead of confusing projection
    # truncation with missing database evidence.
    for gene_id in candidates:
        candidate_records = records_by_candidate.get(gene_id, [])
        if not candidate_records:
            continue
        if any(_is_direct_synthesis_record(record) for record in candidate_records):
            continue
        strongest = min(
            candidate_records,
            key=lambda record: _synthesis_record_key(record, query),
        )
        append_record(strongest, required=True)

    max_depth = max((len(records) for _, records in bounded_groups), default=0)
    for depth in range(max_depth):
        for _key, records in bounded_groups:
            if depth >= len(records) or len(projected) >= max_records:
                continue
            append_record(records[depth], required=False)
        if len(projected) >= max_records:
            break

    total_by_kind = Counter(record.kind.value for record in eligible)
    visible_by_kind = Counter(str(record["kind"]) for record in projected)
    total_by_gene: dict[str, Counter[str]] = defaultdict(Counter)
    visible_by_gene: dict[str, Counter[str]] = defaultdict(Counter)
    for record in eligible:
        total_by_gene[str(record.gene_id)][record.kind.value] += 1
    for record in projected:
        visible_by_gene[str(record["gene_id"])][str(record["kind"])] += 1

    summary = {
        "full_record_count": len(evidence),
        "candidate_gene_record_count": len(eligible),
        "deduplicated_candidate_gene_record_count": deduplicated_count,
        "records_collapsed_by_deduplication": max(0, len(eligible) - deduplicated_count),
        "model_visible_record_count": len(projected),
        "model_visible_json_chars": used_chars + publication_chars,
        "omitted_candidate_gene_record_count": max(0, len(eligible) - len(projected)),
        "records_omitted_after_deduplication": max(
            0,
            deduplicated_count - len(projected),
        ),
        "candidate_gene_count": len(candidates),
        "candidate_genes_with_eligible_evidence": sum(
            bool(records_by_candidate.get(gene_id)) for gene_id in candidates
        ),
        "candidate_genes_with_model_visible_evidence": sum(
            bool(visible_by_gene.get(gene_id)) for gene_id in candidates
        ),
        "candidate_genes_without_eligible_evidence": [
            gene_id for gene_id in candidates if not records_by_candidate.get(gene_id)
        ],
        "candidate_genes_without_model_visible_evidence": [
            gene_id
            for gene_id in candidates
            if records_by_candidate.get(gene_id) and not visible_by_gene.get(gene_id)
        ],
        "full_counts_by_kind": dict(sorted(total_by_kind.items())),
        "deduplicated_counts_by_kind": dict(sorted(deduplicated_by_kind.items())),
        "model_visible_counts_by_kind": dict(sorted(visible_by_kind.items())),
        "full_counts_by_gene_and_kind": {
            gene_id: dict(sorted(total_by_gene.get(gene_id, {}).items())) for gene_id in candidates
        },
        "model_visible_counts_by_gene_and_kind": {
            gene_id: dict(sorted(visible_by_gene.get(gene_id, {}).items()))
            for gene_id in candidates
        },
        "deduplicated_counts_by_gene_and_kind": {
            gene_id: dict(sorted(deduplicated_by_gene.get(gene_id, {}).items()))
            for gene_id in candidates
        },
        "selection_policy": load_prompt("synthesis_projection_policy"),
        "publication_count": len(publication_ledger),
        "retrieved_publication_occurrence_count": len(publication_occurrences),
        "unique_retrieved_publication_count": len(unique_publication_pmids),
        "publication_ledger": publication_ledger,
        "per_candidate_record_limits": {
            "direct": {
                kind.value: limit
                for kind, limit in _DIRECT_RECORD_LIMITS_PER_CANDIDATE.items()
            },
            "context": {
                kind.value: limit
                for kind, limit in _CONTEXT_RECORD_LIMITS_PER_CANDIDATE.items()
            },
        },
        "limits": {"records": max_records, "json_chars": max_chars},
        "capacity_policy": "model_context" if use_model_context else "configured_projection",
    }
    if use_model_context:
        required_colocalizations = [
            record for record in eligible if _is_model_required_colocalization(record)
        ]
        visible_ids = {row["evidence_id"] for row in projected}
        summary["required_model_visible_colocalization_count"] = len(required_colocalizations)
        summary["required_model_visible_colocalization_ids"] = [
            record.evidence_id for record in required_colocalizations
        ]
        summary["missing_required_model_colocalization_ids"] = [
            record.evidence_id
            for record in required_colocalizations
            if record.evidence_id not in visible_ids
        ]
        summary["per_candidate_record_limits"] = None
    if allowed_evidence_ids is not None:
        summary["retry_selection_policy"] = "subset_of_previous_model_visible_evidence"
    if query.ld_expansion is not None:
        visible_variants = sorted({row["variant_id"] for row in projected if row.get("variant_id")})
        all_variants = analysis_variant_ids(query)
        scope = query.ld_expansion
        summary["analysis_scope"] = {
            "source_variant_ids": [row.variant_id for row in query.variants],
            "source_observations": [{"variant_id": row.variant_id, "p_value": row.p_value} for row in query.variants],
            "analysis_variant_count": len(all_variants),
            "ld_proxy_count": len(set(all_variants).difference(row.variant_id for row in query.variants)),
            "model_visible_variant_ids": visible_variants,
            "analysis_variants_without_visible_records": len(set(all_variants).difference(visible_variants)),
            **{key: scope.get(key) for key in ("population", "threshold", "window_bp", "reference_panel", "genome_build", "complete", "status")},
            "window_bp_basis": "per_side",
            "proxy_evidence_complete": scope.get("proxy_evidence_complete", True),
        }
    return projected, summary


def build_publication_projection(
    evidence: Sequence[EvidenceRecord],
    *,
    query: LocusQuery | None = None,
    max_publications: int = MAX_MODEL_PUBLICATIONS,
    max_chars: int = parameter("projection.publications_max_chars"),
) -> list[dict[str, Any]]:
    """Return one bounded model-visible publication row per PMID.

    Open Targets target-literature rows can repeat the same paper for several
    candidates.  This projection deduplicates those rows before the LLM call,
    preserves every linked gene/evidence ID, and carries an upstream abstract
    when one is actually available.  It never fabricates missing metadata.
    """

    if max_publications < 1 or max_chars < 1:
        raise ValueError("publication projection limits must be positive")
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in evidence:
        if record.kind is not EvidenceKind.LITERATURE:
            continue
        pmid = _publication_record_pmid(record)
        if pmid is not None:
            grouped[pmid].append(record)

    result: list[dict[str, Any]] = []
    used_chars = 2  # JSON list brackets
    for pmid in _select_publication_pmids(
        grouped,
        query=query,
        max_publications=max_publications,
    ):
        records = grouped[pmid]
        row: dict[str, Any] = {
            "pmid": pmid,
            "linked_gene_ids": sorted(
                {str(record.gene_id) for record in records if record.gene_id}
            ),
            "supporting_evidence_ids": sorted(
                {record.evidence_id for record in records}
            ),
        }
        for output_key, source_keys in (
            ("title", ("publicationTitle", "title")),
            ("authors", ("authors", "authorString", "publicationFirstAuthor")),
            ("journal", ("publicationJournal", "journal", "journalTitle")),
            ("publication_date", ("publicationDate", "publication_date")),
            ("doi", ("doi",)),
        ):
            value = _first_publication_value(records, source_keys)
            if value:
                row[output_key] = value
        abstract = _first_publication_value(
            records,
            ("abstractText", "abstract", "abstract_text"),
        )
        if abstract:
            row["abstract"] = abstract[:MAX_PUBLICATION_ABSTRACT_CHARS]
            if len(abstract) > MAX_PUBLICATION_ABSTRACT_CHARS:
                row["abstract_truncated"] = True
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        if result and used_chars + len(encoded) + 1 > max_chars:
            continue
        if not result and len(encoded) + 2 > max_chars:
            # Preserve provenance even when one unusually long abstract exceeds
            # the whole publication budget.
            row.pop("abstract", None)
            row["abstract_omitted_for_budget"] = True
            encoded = json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        if used_chars + len(encoded) + (1 if result else 0) > max_chars:
            # Publication metadata itself can exceed the budget. Skip this
            # optional row while preserving exact metadata in the full ledger.
            continue
        result.append(row)
        used_chars += len(encoded) + (1 if len(result) > 1 else 0)
        if len(result) >= max_publications:
            break
    return result


def _select_publication_pmids(
    grouped: Mapping[str, Sequence[EvidenceRecord]],
    *,
    query: LocusQuery | None,
    max_publications: int,
) -> list[str]:
    """Select a relevance-ordered, candidate-balanced publication set."""

    globally_ordered = sorted(
        grouped,
        key=lambda value: _publication_priority(value, grouped[value], query=query),
    )
    gene_order = tuple(
        dict.fromkeys(
            (
                *(query.candidate_genes if query is not None else ()),
                *(
                    str(record.gene_id)
                    for records in grouped.values()
                    for record in records
                    if record.gene_id
                ),
            )
        )
    )
    by_gene = {
        gene_id: [
            pmid
            for pmid in globally_ordered
            if any(str(record.gene_id or "") == gene_id for record in grouped[pmid])
        ]
        for gene_id in gene_order
    }
    # Citations for actual retrieved observations must survive the context
    # budget before candidate-balanced background papers, even without a gene.
    selected = [pmid for pmid in globally_ordered if any(
        _publication_field(record, 'pmid_origin') in {'current_locus_evidence', 'function_description'}
        or record.dataset in {'retrieved_study_publication', 'function_publication'}
        for record in grouped[pmid])][:max_publications]
    selected_set = set(selected)
    if len(selected) >= max_publications:
        return selected
    depth = 0
    while len(selected) < max_publications:
        added = False
        for gene_id in gene_order:
            candidates = by_gene[gene_id]
            if depth >= len(candidates):
                continue
            pmid = candidates[depth]
            if pmid in selected_set:
                continue
            selected.append(pmid)
            selected_set.add(pmid)
            added = True
            if len(selected) >= max_publications:
                return selected
        if not added and all(depth >= len(items) - 1 for items in by_gene.values()):
            break
        depth += 1

    for pmid in globally_ordered:
        if pmid in selected_set:
            continue
        selected.append(pmid)
        if len(selected) >= max_publications:
            break
    return selected


def _publication_priority(
    pmid: str,
    records: Sequence[EvidenceRecord],
    *,
    query: LocusQuery | None,
) -> tuple[Any, ...]:
    """Prefer phenotype-relevant, enriched, recent papers deterministically.

    PMID numeric order is not a relevance ranking. The former ascending sort
    systematically selected the oldest papers when a target returned hundreds.
    This key uses only metadata already retrieved by GWASMind.
    """

    searchable_fields = (
        "publicationTitle",
        "title",
        "abstractText",
        "abstract",
        "abstract_text",
    )
    searchable = " ".join(
        [
            *(
                value
                for record in records
                for key in searchable_fields
                if (value := _publication_field(record, key))
            ),
        ]
    ).casefold()
    trait_matches = int(query is not None and _trait_phrase_matches(query.trait, searchable))
    has_abstract = bool(
        _first_publication_value(records, ("abstractText", "abstract", "abstract_text"))
    )
    newest_ordinal = max(
        (record.evidence_date.toordinal() for record in records if record.evidence_date),
        default=0,
    )
    return (-trait_matches, -int(has_abstract), -newest_ordinal, -int(pmid))


def _first_publication_value(
    records: Sequence[EvidenceRecord],
    keys: tuple[str, ...],
) -> str | None:
    for record in records:
        for key in keys:
            value = _publication_field(record, key)
            if value:
                return value
    return None


def _publication_field(record: EvidenceRecord, key: str) -> str | None:
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    upstream = payload.get("upstream")
    raw = upstream.get("raw") if isinstance(upstream, Mapping) else {}
    context = payload.get("context")
    for mapping in (payload, raw, context):
        if not isinstance(mapping, Mapping):
            continue
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _publication_record_pmid(record: EvidenceRecord) -> str | None:
    return normalize_pmid(
        record.pmid
        or _publication_field(record, "pmid")
        or _publication_field(record, "pubmedId")
    )


def _deduplicate_synthesis_records(
    records: Sequence[EvidenceRecord],
    *,
    query: LocusQuery,
    preserve_analysis_origin: bool = True,
) -> tuple[list[EvidenceRecord], dict[str, dict[str, Any]]]:
    """Represent one scientific group without pooling its member measurements.

    Shared QTL credible sets or colocalisation pairs define grouping units, not
    equality of every member-specific PIP or effect. Only the representative's
    measurements enter its display row; original IDs/hashes retain the lineage.
    """

    groups: dict[tuple[str, ...], list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        groups[_synthesis_deduplication_key(
            record, preserve_analysis_origin=preserve_analysis_origin
        )].append(record)
    unique: list[EvidenceRecord] = []
    metadata: dict[str, dict[str, Any]] = {}
    for grouped_records in groups.values():
        ordered = sorted(
            grouped_records,
            key=lambda record: _synthesis_record_key(record, query),
        )
        representative = ordered[0]
        unique.append(representative)
        variant_ids = tuple(
            dict.fromkeys(
                str(record.variant_id) for record in ordered if str(record.variant_id or "")
            )
        )
        represented_evidence_ids = tuple(record.evidence_id for record in ordered)
        represented_pmids = tuple(
            dict.fromkeys(str(record.pmid) for record in ordered if str(record.pmid or ""))
        )
        metadata[representative.evidence_id] = {
            "represented_record_count": len(ordered),
            "represented_evidence_ids": list(represented_evidence_ids[:20]),
            "additional_evidence_ids_omitted": max(0, len(represented_evidence_ids) - 20),
            "represented_evidence_ids_sha256": hashlib.sha256(
                json.dumps(
                    sorted(represented_evidence_ids),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "represented_variant_count": len(variant_ids),
            "represented_variant_ids": list(variant_ids[:20]),
            "additional_variant_ids_omitted": max(0, len(variant_ids) - 20),
            "represented_pmids": list(represented_pmids[:20]),
            "additional_pmids_omitted": max(0, len(represented_pmids) - 20),
            "deduplication_key_type": _deduplication_key_type(representative.kind),
            "measurement_binding": "representative_evidence_id",
            "member_measurements_pooled": False,
        }
    unique.sort(key=lambda record: _synthesis_record_key(record, query))
    return unique, metadata


def _synthesis_deduplication_key(
    record: EvidenceRecord, *, preserve_analysis_origin: bool = True
) -> tuple[str, ...]:
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    raw = _nested_mapping(payload, "upstream", "raw")
    common = (
        str(record.gene_id or ""),
        record.kind.value,
        "support_state",
        "opposes" if record.supports is False else "supports_or_unspecified",
    )
    if record.kind is not EvidenceKind.LITERATURE:
        # Scientific identifiers belong to their source dataset and release.
        # Do not silently equate the same identifier from different references.
        # PMID is the intentional exception: one publication can be retrieved
        # through multiple channels without becoming several publications.
        common = (*common, "reference", record.source, record.dataset, record.data_release)
    origin = _nested_mapping(payload, "context").get("analysis_origin")
    if preserve_analysis_origin and origin in {"source_selected", "ld_proxy"}:
        # Identical study assertions can be shared, but a proxy representative
        # must never substitute for evidence on the original selected SNP.
        common = (*common, "analysis_origin", origin)

    def unique_record_key() -> tuple[str, ...]:
        return common + ("evidence_id", record.evidence_id)

    if record.kind is EvidenceKind.L2G:
        locus_id = _first_nonempty(
            record.credible_set_id,
            _scientific_value(payload, raw, "studyLocusId"),
        )
        if not locus_id:
            return unique_record_key()
        return common + (
            "credible_set",
            locus_id,
            _first_nonempty(
                record.study_id,
                _scientific_value(payload, raw, "studyId"),
            ),
        )
    if record.kind is EvidenceKind.QTL:
        locus_id = _first_nonempty(
            record.credible_set_id,
            _scientific_value(payload, raw, "studyLocusId"),
        )
        biosample = _first_nonempty(
            record.tissue,
            _scientific_value(payload, raw, "biosampleId"),
            _scientific_value(payload, raw, "biosampleName"),
        )
        if not locus_id or not biosample:
            return unique_record_key()
        return common + (
            "qtl_credible_set",
            locus_id,
            _first_nonempty(
                record.study_id,
                _scientific_value(payload, raw, "studyId"),
            ),
            biosample,
        )
    if record.kind is EvidenceKind.COLOCALIZATION:
        qtl_locus = _first_nonempty(
            _scientific_value(payload, raw, "qtlStudyLocusId"),
            _scientific_value(payload, raw, "queriedStudyLocusId"),
            record.credible_set_id,
        )
        other_locus = _first_nonempty(
            _scientific_value(payload, raw, "otherStudyLocusId"),
        )
        left_locus = _first_nonempty(_scientific_value(payload, raw, "leftStudyLocusId"))
        right_locus = _first_nonempty(_scientific_value(payload, raw, "rightStudyLocusId"))
        if not other_locus and qtl_locus:
            other_locus = right_locus if qtl_locus != right_locus else left_locus
        if not qtl_locus or not other_locus:
            return unique_record_key()
        return common + (
            "colocalisation_pair",
            qtl_locus,
            other_locus,
            _first_nonempty(
                _scientific_value(payload, raw, "traitFromSource"),
                _scientific_value(payload, raw, "rightTraitFromSource"),
            ),
        )
    if record.kind is EvidenceKind.LITERATURE:
        pmid = _first_nonempty(
            record.pmid,
            _scientific_value(payload, raw, "pubmedId"),
            _scientific_value(payload, raw, "pmid"),
        )
        return common + ("pmid", pmid) if pmid else unique_record_key()
    if record.kind in {
        EvidenceKind.EXPRESSION,
        EvidenceKind.FUNCTION,
        EvidenceKind.PATHWAY,
    }:
        module = _first_nonempty(payload.get("module"), record.dataset, record.source)
        fingerprint = _scientific_profile_fingerprint(record)
        return common + ("profile", module, record.dataset, record.source, fingerprint)
    if record.kind is EvidenceKind.REGULATORY:
        regulatory_element = _first_nonempty(
            _scientific_value(payload, raw, "intervalId"),
            _scientific_value(payload, raw, "regulatoryElementKey"),
        )
        if not regulatory_element:
            chromosome = _first_nonempty(_scientific_value(payload, raw, "chromosome"))
            start = _first_nonempty(_scientific_value(payload, raw, "start"))
            end = _first_nonempty(_scientific_value(payload, raw, "end"))
            if chromosome and start and end:
                regulatory_element = f"chr{chromosome.removeprefix('chr')}:{start}-{end}"
        relation = _first_nonempty(payload.get("relation"))
        if not regulatory_element or not relation:
            return unique_record_key()
        return common + (
            "regulatory_assertion",
            regulatory_element,
            _first_nonempty(_scientific_value(payload, raw, "intervalType")),
            relation,
            _first_nonempty(record.study_id, _scientific_value(payload, raw, "studyId")),
            _first_nonempty(record.credible_set_id),
            _first_nonempty(
                record.tissue,
                _scientific_value(payload, raw, "biosampleId"),
                _scientific_value(payload, raw, "biosampleName"),
            ),
            record.dataset,
            record.source,
            _scientific_profile_fingerprint(record),
        )
    # Other evidence kinds are normally bounded and heterogeneous.  Avoid
    # speculative merging: an exact evidence ID remains the safe identity.
    return unique_record_key()


def _deduplication_key_type(kind: EvidenceKind) -> str:
    return {
        EvidenceKind.L2G: "gene+credible_set",
        EvidenceKind.QTL: "gene+qtl_credible_set+biosample",
        EvidenceKind.COLOCALIZATION: "gene+qtl_locus+other_locus+trait",
        EvidenceKind.LITERATURE: "gene+pmid",
        EvidenceKind.EXPRESSION: "gene+module+content_fingerprint",
        EvidenceKind.FUNCTION: "gene+module+content_fingerprint",
        EvidenceKind.PATHWAY: "gene+module+content_fingerprint",
        EvidenceKind.REGULATORY: "gene+regulatory_interval+study+tissue+assertion",
    }.get(kind, "exact_evidence_id")


def compact_record_for_synthesis(
    record: EvidenceRecord,
    *,
    relevant_tissues: tuple[str, ...] = (),
    complete_scientific_lists: bool = False,
) -> dict[str, Any]:
    """Strip redundant provider envelopes while retaining scientific fields and lineage."""

    payload = record.payload if isinstance(record.payload, Mapping) else {}
    raw = _nested_mapping(payload, "upstream", "raw")
    raw_summary: dict[str, Any] = {}
    for key in _RAW_SCALAR_FIELDS:
        value = _scientific_value(payload, raw, key)
        if _is_compact_scalar(value):
            raw_summary[key] = value
    member = _scientific_value(payload, raw, "matchedLocusMember")
    if isinstance(member, Mapping):
        raw_summary["matchedLocusMember"] = dict(member) if complete_scientific_lists else {
            key: member[key]
            for key in _MATCHED_MEMBER_FIELDS
            if key in member and _is_compact_scalar(member[key])
        }
    for key, limit in _RAW_LIST_LIMITS.items():
        value = _scientific_value(payload, raw, key)
        if isinstance(value, list):
            if complete_scientific_lists:
                raw_summary[key] = value
                continue
            if key == "tissues" and record.kind is EvidenceKind.EXPRESSION:
                kept, matched_labels = _compact_expression_rows(
                    list(value),
                    relevant_tissues=relevant_tissues,
                    limit=limit,
                )
                raw_summary[key] = [_bounded_nested_value(item) for item in kept]
                raw_summary["relevantTissuesRequested"] = list(relevant_tissues)
                raw_summary["relevantTissueMatches"] = list(matched_labels)
            else:
                raw_summary[key] = [_bounded_nested_value(item) for item in value[:limit]]
            if len(value) > limit:
                raw_summary[f"{key}RecordCount"] = len(value)
                raw_summary[f"omitted{key[0].upper() + key[1:]}RecordCount"] = len(value) - limit

    result: dict[str, Any] = {
        "evidence_id": record.evidence_id,
        "kind": record.kind.value,
        "dataset": record.dataset,
        "source": record.source,
        "data_release": record.data_release,
        "module": payload.get("module"),
        "gene_id": record.gene_id,
        "variant_id": record.variant_id,
        "study_id": record.study_id,
        "pmid": record.pmid,
        "credible_set_id": record.credible_set_id,
        "tissue": record.tissue,
        "supports": record.supports,
        "parent_variant_ids": list(record.parent_variant_ids if complete_scientific_lists else record.parent_variant_ids[:20]),
        "parent_evidence_ids": list(record.parent_evidence_ids if complete_scientific_lists else record.parent_evidence_ids[:20]),
        "provenance_complete": record.provenance_complete,
        "evidence_date": record.evidence_date.isoformat() if record.evidence_date else None,
        "relation": payload.get("relation"),
        "measurements": dict(_nested_mapping(payload, "measurements")) if complete_scientific_lists else _bounded_mapping(payload.get("measurements"), limit=20),
        "context": dict(_nested_mapping(payload, "context")) if complete_scientific_lists else _bounded_mapping(payload.get("context"), limit=20),
        "subject": dict(_nested_mapping(payload, "subject")) if complete_scientific_lists else _bounded_mapping(payload.get("subject"), limit=10),
        "object": dict(_nested_mapping(payload, "object")) if complete_scientific_lists else _bounded_mapping(payload.get("object"), limit=10),
        "raw_summary": raw_summary,
        "candidate_scope": dict(evidence_candidate_scope(payload)),
    }
    derived = payload.get("derived") if isinstance(payload.get("derived"), Mapping) else {}
    if isinstance(derived.get("expression_profiles"), Mapping):
        result["expression_profiles"] = (
            dict(derived["expression_profiles"])
            if complete_scientific_lists
            else _bounded_mapping(derived["expression_profiles"], limit=8)
        )
        result["expression_profile_contract"] = derived.get(
            "expression_profile_contract", "expression-profile-v1"
        )
    context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
    if context.get("analysis_origin"):
        # The top-level relation below is complete and counted in the record
        # budget; omit the generic nested copy, which otherwise truncates lists.
        without_links = {key: value for key, value in context.items() if key != "ld_proxy_links"}
        result["context"] = without_links if complete_scientific_lists else _bounded_mapping(without_links, limit=20)
        result["analysis_origin"] = context["analysis_origin"]
        links = context.get("ld_proxy_links", [])
        if isinstance(links, list):
            # Every source link for this model-visible proxy is retained. The
            # normal record character budget includes these compact relations.
            result["ld_proxy_links"] = [{key: link.get(key) for key in
                ("selected_variant_id", "proxy_variant_id", "population", "r2", "genome_build", "evidence_relation")}
                for link in links if isinstance(link, Mapping)]
            result["ld_proxy_links_omitted"] = 0
            result["ld_proxy_link_count"] = len(result["ld_proxy_links"])
    return {key: value for key, value in result.items() if value not in (None, "", [], {}, ())}


def _bounded_candidate_kind_records(
    records: Sequence[EvidenceRecord],
    *,
    query: LocusQuery,
    kind: EvidenceKind,
) -> list[EvidenceRecord]:
    """Bound one candidate/kind group without losing its strongest science."""

    ordered = _diverse_synthesis_records(records, query=query, limit=len(records))
    if kind is EvidenceKind.LITERATURE:
        ordered = sorted(
            ordered,
            key=lambda record: _publication_priority(
                _publication_record_pmid(record) or "0",
                (record,),
                query=query,
            ),
        )
    if kind is EvidenceKind.VARIANT_ANNOTATION:
        high_impact = [record for record in ordered if _is_high_impact_coding(record)]
        selected_ids = {record.evidence_id for record in high_impact}
        limit = max(
            len(high_impact),
            _CONTEXT_RECORD_LIMITS_PER_CANDIDATE[EvidenceKind.VARIANT_ANNOTATION],
        )
        context = [record for record in ordered if record.evidence_id not in selected_ids]
        return [*high_impact, *context[: max(0, limit - len(high_impact))]]

    limit = _DIRECT_RECORD_LIMITS_PER_CANDIDATE.get(
        kind,
        _CONTEXT_RECORD_LIMITS_PER_CANDIDATE.get(
            kind,
            _DEFAULT_CONTEXT_RECORD_LIMIT_PER_CANDIDATE,
        ),
    )
    return ordered[:limit]


def _is_direct_synthesis_record(record: EvidenceRecord) -> bool:
    return record.kind in _DIRECT_RECORD_LIMITS_PER_CANDIDATE or (
        record.kind is EvidenceKind.VARIANT_ANNOTATION and _is_high_impact_coding(record)
    )


def _frozen_molqtl_coloc_scope_allowed(record: EvidenceRecord | Mapping[str, Any]) -> bool | None:
    """Validate frozen molecular-query lineage without relabelling old records.

    A frozen molQTL query identity is required for the new EFO relations. Related
    phenotype support remains related, even if its numerical support is strong.
    """
    kind = record.kind.value if isinstance(record, EvidenceRecord) else record.get("kind")
    if kind != EvidenceKind.COLOCALIZATION.value:
        return None
    payload = record.payload if isinstance(record, EvidenceRecord) else record.get("payload", {})
    payload = payload if isinstance(payload, Mapping) else {}
    context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
    relation = context.get("target_trait_relation")
    phenotype = context.get("phenotype_evidence_scope")
    rank_scope = context.get("rank_scope")
    unclassified = context.get("retrieval_policy") == "frozen_molqtl_all_phenotypes_v2"
    if unclassified and (rank_scope != "molecular_locus_link" or relation is not None or phenotype is not None):
        return False
    if (not unclassified and relation not in {"same_disease_efo", "allowed_related_disease_efo"}
            and phenotype not in {"target_phenotype", "related_phenotype"}
            and rank_scope not in {"target_phenotype", "related_phenotype"}):
        if rank_scope == "molecular_locus_link":
            return False
        return None
    scopes = {"target_trait": "target", "target_phenotype": "target",
              "related_trait": "related", "related_phenotype": "related"}
    group = {"same_disease_efo": "target", "allowed_related_disease_efo": "related"}.get(relation)
    if not unclassified and (group is None or scopes.get(phenotype) != group or scopes.get(rank_scope) != group):
        return False
    role = evidence_candidate_scope(payload)
    upstream = payload.get("upstream") if isinstance(payload.get("upstream"), Mapping) else {}
    raw = upstream.get("raw") if isinstance(upstream.get("raw"), Mapping) else {}
    if (not unclassified and phenotype not in {"target_phenotype", "related_phenotype"}
            and rank_scope not in {"target_phenotype", "related_phenotype"}
            and "queriedStudyLocusId" not in raw):
        # Saved variant-anchored V2G records used the same EFO relation names
        # with legacy scope enums. Keep that explicit historical contract;
        # never invent a new frozen-query lineage for those records.
        return None
    queried = raw.get("queriedStudyLocusId")
    gene = record.gene_id if isinstance(record, EvidenceRecord) else record.get("gene_id")
    variant = record.variant_id if isinstance(record, EvidenceRecord) else record.get("variant_id")
    credible_set = record.credible_set_id if isinstance(record, EvidenceRecord) else record.get("credible_set_id")
    target = payload.get("object") if isinstance(payload.get("object"), Mapping) else {}
    subject = payload.get("subject") if isinstance(payload.get("subject"), Mapping) else {}
    return (role.get("direct_locus_support") is True and role.get("context_only") is not True
            and isinstance(queried, str) and bool(queried)
            and raw.get("qtlStudyLocusId") == queried and context.get("study_locus_id") == queried
            and (not credible_set or credible_set == queried)
            and bool(gene) and raw.get("qtlGeneId") == gene and target.get("id") == gene
            and bool(variant) and subject.get("id") == variant and raw.get("queryVariantId") == variant)


def _is_model_required_colocalization(record: EvidenceRecord) -> bool:
    """Return whether a coloc must be visible in the model evidence cards."""
    if record.kind is not EvidenceKind.COLOCALIZATION:
        return False
    frozen_scope = _frozen_molqtl_coloc_scope_allowed(record)
    if frozen_scope is not None:
        return frozen_scope
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
    role = evidence_candidate_scope(payload)
    return (
        context.get("target_trait_relation") in {"exact_label", "related_label", "other_trait",
                                                "same_disease_efo", "allowed_related_disease_efo"}
        and role.get("direct_locus_support") is True
        and role.get("context_only") is not True
    )


def _is_high_impact_coding(record: EvidenceRecord) -> bool:
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    scope = evidence_candidate_scope(payload)
    if scope and not scope.get("direct_locus_support"):
        return False
    return _has_high_impact_coding_payload(payload)


def _has_high_impact_coding_payload(payload: Mapping[str, Any]) -> bool:
    # Missense counts only when retrieval verified the variant belongs to the locus
    # set; positional or unscoped missense annotations remain background.
    explicit_direct = evidence_candidate_scope(payload).get("direct_locus_support") is True
    consequences = _SEVERE_CONSEQUENCES | (_DIRECT_SCOPED_CODING_CONSEQUENCES if explicit_direct else frozenset())

    def has_consequence(value: Any, depth: int = 0) -> bool:
        if depth > 7:
            return False
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized = str(key).replace("_", "").casefold()
                if normalized in {"highimpact", "ishighimpact"} and nested is True:
                    return True
                if normalized in {"impact", "vepimpact"} and (
                    isinstance(nested, str) and nested.strip().casefold() == "high"
                ):
                    return True
                if normalized in {
                    "consequence", "consequenceterm", "consequenceterms",
                    "transcriptconsequence", "mostsevereconsequence",
                } and any(
                    isinstance(item, str) and item.strip().casefold() in consequences
                    for item in _walk_projection_scalars(nested)
                ):
                    return True
                if has_consequence(nested, depth + 1):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(has_consequence(item, depth + 1) for item in value[:1_000])
        return False

    return has_consequence(payload)


def _walk_projection_scalars(value: Any, *, depth: int = 0):
    if depth > 7:
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_projection_scalars(item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value[:1_000]:
            yield from _walk_projection_scalars(item, depth=depth + 1)
    elif isinstance(value, (str, int, float, bool)):
        yield value


def _diverse_synthesis_records(
    records: Sequence[EvidenceRecord],
    *,
    query: LocusQuery,
    limit: int,
) -> list[EvidenceRecord]:
    ordered = sorted(records, key=lambda record: _synthesis_record_key(record, query))
    selected: list[EvidenceRecord] = []
    selected_ids: set[str] = set()

    # Preserve both sides of an explicit evidence conflict when the per-kind
    # budget permits it.  The remaining slots still favor supportive records.
    for is_opposing in (False, True):
        matching = [record for record in ordered if (record.supports is False) is is_opposing]
        if not matching:
            continue
        record = matching[0]
        selected.append(record)
        selected_ids.add(record.evidence_id)
        if len(selected) >= limit:
            return selected

    seen_traits: set[str] = set()
    for record in ordered:
        if record.evidence_id in selected_ids:
            continue
        trait = _record_trait_label(record).casefold()
        if not trait or trait in seen_traits:
            continue
        selected.append(record)
        selected_ids.add(record.evidence_id)
        seen_traits.add(trait)
        if len(selected) >= limit:
            return selected
    seen_context: set[tuple[str, str, str, str]] = set()
    for record in ordered:
        if record.evidence_id in selected_ids:
            continue
        context_key = (
            str(record.study_id or ""),
            str(record.variant_id or ""),
            str(record.tissue or ""),
            str(record.pmid or ""),
        )
        if context_key in seen_context:
            continue
        seen_context.add(context_key)
        selected.append(record)
        selected_ids.add(record.evidence_id)
        if len(selected) >= limit:
            return selected
    for record in ordered:
        if record.evidence_id in selected_ids:
            continue
        selected.append(record)
        selected_ids.add(record.evidence_id)
        if len(selected) >= limit:
            break
    return selected


def _synthesis_record_key(record: EvidenceRecord, query: LocusQuery) -> tuple[Any, ...]:
    trait_text = _record_trait_label(record).casefold()
    trait_match = _trait_phrase_matches(query.trait, trait_text)
    tissue_text = str(record.tissue or "").casefold()
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    raw = _nested_mapping(payload, "upstream", "raw")
    biosample_text = " ".join(
        _first_nonempty(_scientific_value(payload, raw, key))
        for key in ("biosampleId", "biosampleName")
    ).casefold()
    tissues = _scientific_value(payload, raw, "tissues")
    if isinstance(tissues, list):
        biosample_text += " " + json.dumps(tissues[:20], ensure_ascii=False, default=str).casefold()
    tissue_match = bool(
        query.relevant_tissues
        and any(
            _whole_phrase_matches(item, f"{tissue_text} {biosample_text}")
            for item in query.relevant_tissues
        )
    )
    return (
        record.supports is False,
        not trait_match,
        not tissue_match,
        -_record_strength(record),
        record.evidence_id,
    )


def _normalized_phrase(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def _whole_phrase_matches(phrase: str, text: str) -> bool:
    phrase = _normalized_phrase(phrase)
    return bool(phrase and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
                                     _normalized_phrase(text)))


def _trait_phrase_matches(trait: str, text: str) -> bool:
    """Relevance only: match complete phrases, never shared generic disease words.

    A configured longer subtype alias owns its span before a shorter parent alias
    is considered (eoMDD is not MDD). This prioritizes retrieved rows; it neither
    discards other phenotypes nor establishes a same-phenotype association.
    """
    key, text = _normalized_phrase(trait), _normalized_phrase(text)
    if not key or not text:
        return False
    aliases: dict[str, str] = {}
    for group in parameter("novelty.trait_alias_groups"):
        parts = [_normalized_phrase(part) for part in group.split("|")]
        aliases.update({part: parts[0] for part in parts if part})
    target = aliases.get(key, key)
    aliases.setdefault(key, target)
    # Nonoverlapping longest-first matches prevent a subtype mention from being
    # reinterpreted as its parent. A separate parent mention can still match.
    spans: list[tuple[int, int]] = []
    for alias in sorted(aliases, key=lambda value: (-len(value), value)):
        for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text):
            if any(match.start() < end and start < match.end() for start, end in spans):
                continue
            spans.append(match.span())
            if aliases[alias] == target:
                return True
    return False


def scientific_scoring_records(
    records: Sequence[EvidenceRecord], *, query: LocusQuery
) -> list[EvidenceRecord]:
    """Count each scientific assertion once in the deterministic backup ranker.

    The LLM ledger preserves selected/proxy origins in separate display rows to
    prevent false variant attribution. Those rows are not extra scoring votes.
    This helper is unbounded by model budgets; full records and citation lineage
    stay with the caller. Distinct studies, pairs, tissues and opposing assertions
    remain distinct under the same scientific keys used by model projection.
    """
    unique, _metadata = _deduplicate_synthesis_records(
        records, query=query, preserve_analysis_origin=False
    )
    return unique


def _record_strength(record: EvidenceRecord) -> float:
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    raw = _nested_mapping(payload, "upstream", "raw")
    measurements = payload.get("measurements")
    values: list[float] = []
    member = _scientific_value(payload, raw, "matchedLocusMember")
    for mapping in (measurements, payload, raw, member):
        if not isinstance(mapping, Mapping):
            continue
        for key in (
            "h4",
            "clpp",
            "l2g_score",
            "posteriorProbability",
            "score",
        ):
            value = mapping.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                values.append(float(value))
    return max(values, default=float("-inf"))


def _record_trait_label(record: EvidenceRecord) -> str:
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    raw = _nested_mapping(payload, "upstream", "raw")
    return " | ".join(
        str(_scientific_value(payload, raw, key) or "").strip()
        for key in ("traitFromSource", "rightTraitFromSource")
        if str(_scientific_value(payload, raw, key) or "").strip()
    )


def _scientific_value(
    payload: Mapping[str, Any],
    raw: Mapping[str, Any],
    key: str,
) -> Any:
    """Read one allow-listed science field from live or frozen record shapes."""

    if key in raw and raw.get(key) is not None:
        return raw.get(key)
    return payload.get(key)


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _scientific_profile_fingerprint(record: EvidenceRecord) -> str:
    """Hash only allow-listed scientific content, never a provider envelope."""

    payload = record.payload if isinstance(record.payload, Mapping) else {}
    raw = _nested_mapping(payload, "upstream", "raw")
    keys = (
        "approvedName",
        "approvedSymbol",
        "biotype",
        "geneId",
        "functionDescriptions",
        "tissues",
        "pathways",
        "go",
        "geneOntology",
        "datasourceId",
        "intervalId",
        "biosampleId",
        "biosampleName",
        "score",
    )
    content = {
        key: _scientific_value(payload, raw, key)
        for key in keys
        if _scientific_value(payload, raw, key) not in (None, "", [], {}, ())
    }
    content.update(
        {
            "module": payload.get("module"),
            "relation": payload.get("relation"),
            "dataset": record.dataset,
            "source": record.source,
        }
    )
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _nested_mapping(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _bounded_mapping(value: Any, *, limit: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _bounded_nested_value(item)
        for key, item in list(value.items())[:limit]
        if item not in (None, "", [], {}, ())
    }


def _bounded_nested_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return str(value)[:500]
    if _is_compact_scalar(value):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_nested_value(item, depth=depth + 1)
            for key, item in list(value.items())[:12]
            if item not in (None, "", [], {}, ())
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_nested_value(item, depth=depth + 1) for item in value[:12]]
    return str(value)[:500]


def _is_compact_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and (
        not isinstance(value, str) or len(value) <= 2_000
    )


def _compact_expression_rows(
    rows: list[Any],
    *,
    relevant_tissues: tuple[str, ...],
    limit: int,
) -> tuple[list[Any], tuple[str, ...]]:
    if limit < 1:
        raise ValueError("expression projection limit must be positive")
    relevant: list[Any] = []
    labels: list[str] = []
    for row in rows:
        text = json.dumps(row, ensure_ascii=False, default=str).casefold()
        if any(_whole_phrase_matches(term, text) for term in relevant_tissues):
            relevant.append(row)
            label = _expression_label(row)
            if label and label not in labels:
                labels.append(label)
    ranked = sorted(rows, key=_expression_score, reverse=True)
    selected: list[Any] = []
    seen: set[str] = set()
    for row in [*relevant, *ranked]:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected, tuple(labels)


def _expression_score(row: Any) -> float:
    if not isinstance(row, dict):
        return float("-inf")
    candidates = [row.get("median"), row.get("value")]
    rna = row.get("rna")
    if isinstance(rna, dict):
        candidates.extend([rna.get("value"), rna.get("level")])
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return float("-inf")


def _expression_label(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    for key in ("label", "biosampleName"):
        if row.get(key):
            return str(row[key])
    for key in ("tissueBiosample", "celltypeBiosample"):
        nested = row.get(key)
        if isinstance(nested, dict) and nested.get("biosampleName"):
            return str(nested["biosampleName"])
    return None
