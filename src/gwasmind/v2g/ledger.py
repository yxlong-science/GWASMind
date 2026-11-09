"""Express gene-centric V2G evidence as semantic evidence records.

Each card record becomes one semantic record (``module/subject/object/context/
measurements/derived/upstream.raw``) that the evidence normalization turns into an
``EvidenceRecord``, so the sufficiency gate, safety review, ranking validation,
report writer and renderer consume it unchanged. The card record receives the
record's upstream identifier (``provenance.evidence_id``); after normalization
the caller remaps it to the ledger ``evidence_id`` the model cites.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any
from ..configuration import parameter

from .evidence import CODING_CLASSES, DETAILED_CODING_CLASSES
from .models import GeneWindow, LocusVariant, LocusVariantSet
from .retrieval import RetrievalBundle, _unique_rows

SOURCE = "Open Targets Platform (official GraphQL API)"
REGION_DATASET = "opentargets_region_gene_context"


def upstream_id(kind: str, *parts: Any) -> str:
    digest = hashlib.sha256(json.dumps([kind, *parts], sort_keys=True, default=str).encode()).hexdigest()
    return f"v2g:{kind}:{digest[:24]}"


def _scope(direct: bool, reason: str) -> dict[str, Any]:
    return {"candidate_scope": {"context_only": False, "direct_locus_support": direct,
                                "reason": reason, "scope": "local"}}


def _record(*, dataset: str, module: str, relation: str, identifier: str, subject: Mapping[str, Any],
            gene_id: str | None = None, symbol: str | None = None, context: Mapping[str, Any] | None = None,
            measurements: Mapping[str, Any] | None = None, derived: Mapping[str, Any] | None = None,
            raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    variant_subject = subject.get("type") == "variant"
    return {
        "module": module, "relation": relation, "dataset": dataset,
        "subject": dict(subject),
        "object": {"type": "target", "id": gene_id, "label": symbol} if variant_subject and gene_id else None,
        "context": {key: value for key, value in (context or {}).items() if value is not None},
        "measurements": {key: value for key, value in (measurements or {}).items() if value is not None},
        "derived": dict(derived or {}),
        "provenance": {"source": SOURCE, "evidence_id": identifier, "pipeline": "gwasmind_v2g_gene_centric"},
        "upstream": {"raw": dict(raw or {})},
    }


def _text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _variant_subject(variant: LocusVariant) -> dict[str, Any]:
    return {"type": "variant", "id": variant.variant_id}


def _coding(gene_id: str, symbol: str | None, category: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for card_row in category["records"]:
        source = card_row.get("_source") or {}
        row, variant = source.get("row") or {}, source.get("variant")
        if card_row["consequence_class"] not in DETAILED_CODING_CLASSES or variant is None:
            continue
        identifier = upstream_id("variant_annotation", gene_id, variant.variant_id, card_row["consequence"])
        card_row["evidence_id"] = identifier
        records.append(_record(
            dataset="variant", module="locus.variant_consequence", relation="variant_transcript_consequence",
            identifier=identifier, subject=_variant_subject(variant), gene_id=gene_id, symbol=symbol,
            context={"approved_symbol": symbol, "consequence_class": card_row["consequence_class"],
                     "r2_to_anchor": card_row.get("r2_to_anchor")},
            measurements={"distance_from_tss": row.get("distanceFromTss"),
                          "distance_from_footprint": row.get("distanceFromFootprint")},
            derived=_scope(card_row["consequence_class"] in CODING_CLASSES, "variant_transcript_consequence"),
            raw={"target": {"id": gene_id, "approvedSymbol": symbol}, "variantId": variant.variant_id,
                 "consequence": card_row["consequence"], "impact": row.get("impact"),
                 "isEnsemblCanonical": row.get("isEnsemblCanonical"),
                 "variantConsequences": row.get("variantConsequences") or [],
                 "distanceFromTss": row.get("distanceFromTss"),
                 "distanceFromFootprint": row.get("distanceFromFootprint")}))
    return records


def _qtl(gene_id: str, symbol: str | None, category: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    basis = category.get("clpp_basis")
    overlap = ({"basis": basis, "scope": "query_variant_set_qtl_credible_set_overlap",
                "statistic_type": "ld_weighted_qtl_overlap" if basis == "ld_approximation" else "pip_weighted_qtl_overlap",
                **({"calibrated_colocalization_posterior": False} if basis == "ld_approximation" else {})}
               if basis in {"ld_approximation", "pip"} else None)
    for card_row in category["records"]:
        source = card_row.get("_source") or {}
        row, member, variant = source.get("row") or {}, source.get("member") or {}, source.get("variant")
        if variant is None:
            continue
        study = row.get("study") or {}
        biosample = study.get("biosample") or {}
        locus_id = row.get("studyLocusId")
        identifier = upstream_id("qtl", gene_id, locus_id, "credible_set_lead")
        lead = card_row["cs_lead_measurement"]
        card_row["evidence_id"] = identifier
        matched = {"variantId": (member.get("variant") or {}).get("id"),
                   "posteriorProbability": member.get("posteriorProbability"),
                   "pValueMantissa": member.get("pValueMantissa"), "pValueExponent": member.get("pValueExponent"),
                   "beta": member.get("beta"), "standardError": member.get("standardError")}
        records.append(_record(
            dataset="credible_set", module="locus.qtl", relation="locus_qtl", identifier=identifier,
            subject=_variant_subject(variant), gene_id=gene_id, symbol=symbol,
            context={"biosample": biosample.get("biosampleName"), "biosample_id": biosample.get("biosampleId"),
                     "study_id": study.get("id"),
                     "study_locus_id": locus_id, "study_type": row.get("studyType"),
                     "pmid": _text(study.get("pubmedId")), "qtl_member_variant_id": variant.variant_id,
                     "clpp_basis": basis if overlap else None,
                     "clpp_scope": overlap["scope"] if overlap else None,
                     "measurement_scope": "credible_set_lead", "measured_variant_id": lead["variant_id"],
                     "lead_pvalue_max_exclusive": category.get("lead_pvalue_max_exclusive"),
                     "cs_lead_variant_id": lead["variant_id"],
                     "locus_anchor_witnesses": card_row["locus_anchor_witnesses"],
                     "locus_anchor_roles": card_row["locus_anchor_roles"],
                     "shared_variant_count": card_row.get("shared_variant_count")},
            measurements={key: value for key, value in lead.items()
                          if key in {"pValueMantissa", "pValueExponent", "beta", "standardError"}},
            derived={**_scope(True, "gene_centric_locus_overlap"),
                     **({"molecular_overlap": overlap} if overlap else {})},
            raw={"studyId": study.get("id"), "studyLocusId": locus_id, "studyType": row.get("studyType"),
                 "projectId": study.get("projectId"), "biosampleId": biosample.get("biosampleId"),
                 "biosampleName": biosample.get("biosampleName"), "qtlGeneId": gene_id, "qtlGeneSymbol": symbol,
                 "approvedSymbol": symbol, "pubmedId": study.get("pubmedId"),
                 "publicationFirstAuthor": study.get("publicationFirstAuthor"),
                 "publicationDate": study.get("publicationDate"), "isTransQtl": row.get("isTransQtl"),
                 "matchedLocusMember": matched, "variantId": variant.variant_id,
                 "csLeadVariantId": lead["variant_id"], "csLeadMeasurement": dict(lead),
                 "measuredVariantId": lead["variant_id"], "measurementScope": "credible_set_lead",
                 "finemappingMethod": card_row.get("finemapping_method"),
                 "confidence": card_row.get("confidence"), "qualityControls": card_row.get("quality_controls"),
                 "locusAnchorWitnesses": card_row["locus_anchor_witnesses"],
                 "queryVariantId": matched["variantId"], "queryVariantRole": "credible_set_member"}))
    return records


def _coloc(gene_id: str, symbol: str | None, category: Mapping[str, Any],
           by_query: Mapping[str, LocusVariant]) -> list[dict[str, Any]]:
    records = []
    for card_row in category["records"]:
        source = card_row.get("_source") or {}
        gwas, row = source.get("gwas") or {}, source.get("row") or {}
        other = row.get("otherStudyLocus") or {}
        qtl_study, gwas_study = other.get("study") or {}, gwas.get("study") or {}
        anchors = [by_query[value] for value in row.get("_anchor_variants", gwas.get("anchor_variants")) or []
                   if value in by_query]
        if not anchors:
            continue
        variant = anchors[0]
        biosample = (qtl_study.get("biosample") or {}).get("biosampleName")
        biosample_id = (qtl_study.get("biosample") or {}).get("biosampleId")
        disease_ids = [disease["id"] for disease in gwas_study.get("diseases") or [] if disease.get("id")]
        identifier = upstream_id("colocalization", gene_id, other.get("studyLocusId"), gwas.get("studyLocusId"),
                                 row.get("colocalisationMethod"))
        card_row["evidence_id"] = identifier
        classified_scope = card_row.get("phenotype_scope") if card_row.get("retrieval_policy") != "frozen_molqtl_all_phenotypes_v2" else None
        records.append(_record(
            dataset="colocalisation_coloc", module="locus.colocalisation", relation="locus_colocalisation",
            identifier=identifier, subject=_variant_subject(variant), gene_id=gene_id, symbol=symbol,
            context={"biosample": biosample, "biosample_id": biosample_id, "study_id": qtl_study.get("id"),
                     "study_locus_id": other.get("studyLocusId"), "study_type": other.get("studyType"),
                     "pmid": _text(qtl_study.get("pubmedId")), "rank_scope": classified_scope or "molecular_locus_link",
                     "phenotype_evidence_scope": classified_scope,
                     "target_trait_relation": {"target_phenotype": "same_disease_efo",
                         "related_phenotype": "allowed_related_disease_efo"}.get(classified_scope),
                     "retrieval_policy": card_row.get("retrieval_policy"), "phenotype_ids": disease_ids,
                     "phenotype_scope": card_row.get("phenotype_scope"),
                     "h4_inclusion_threshold": card_row.get("h4_inclusion_threshold")},
            measurements={"h3": row.get("h3"), "h4": row.get("h4"), "clpp": row.get("clpp"),
                          "betaRatioSignAverage": row.get("betaRatioSignAverage")},
            derived=_scope(True, "frozen_molqtl_all_phenotypes_colocalisation"),
            raw={"h3": row.get("h3"), "h4": row.get("h4"), "clpp": row.get("clpp"),
                 "betaRatioSignAverage": row.get("betaRatioSignAverage"),
                 "colocalisationMethod": row.get("colocalisationMethod"),
                 "numberColocalisingVariants": row.get("numberColocalisingVariants"),
                 "qtlGeneId": gene_id, "qtlStudyLocusId": other.get("studyLocusId"),
                 "studyId": qtl_study.get("id"), "studyLocusId": other.get("studyLocusId"),
                 "leftStudyId": qtl_study.get("id"), "leftStudyLocusId": other.get("studyLocusId"),
                 "leftStudyType": other.get("studyType"), "leftPubmedId": qtl_study.get("pubmedId"),
                 "pubmedId": qtl_study.get("pubmedId"), "biosampleName": biosample, "biosampleId": biosample_id,
                 "rightStudyId": gwas_study.get("id"), "rightStudyLocusId": gwas.get("studyLocusId"),
                 "rightStudyType": "gwas", "rightTraitFromSource": gwas_study.get("traitFromSource"),
                 "rightDiseaseIds": disease_ids, "rightDiseases": gwas_study.get("diseases") or [],
                 "rightPubmedId": gwas_study.get("pubmedId"),
                 "rightPublicationFirstAuthor": gwas_study.get("publicationFirstAuthor"),
                 "rightPublicationDate": gwas_study.get("publicationDate"), "queryVariantId": variant.variant_id,
                 "queriedStudyLocusId": row.get("_queried_molqtl_id"),
                 "retrievalPolicy": card_row.get("retrieval_policy"),
                 "phenotypeScope": card_row.get("phenotype_scope"),
                 "upstreamColocalisation": row.get("_upstream_row")}))
    from .coloc_scope import validate_background_rows

    background_rows = category.get("other_phenotypes", ())
    validate_background_rows(background_rows)
    for summary in background_rows:
        identifier = upstream_id("colocalization_background", gene_id, summary["qtl_study_locus_id"])
        summary["evidence_id"] = identifier
        records.append(_record(dataset="colocalisation_background_summary", module="locus.colocalisation",
            relation="other_phenotype_colocalisation_summary", identifier=identifier,
            subject={"type": "target", "id": gene_id, "label": symbol}, gene_id=gene_id, symbol=symbol,
            context={"phenotype_scope": "other_phenotype", "study_locus_id": summary["qtl_study_locus_id"],
                     "retrieval_policy": summary["retrieval_policy"]},
            measurements={"pair_count": summary["pair_count"], "max_h4": summary["max_h4"],
                          "phenotype_count": len(summary["phenotype_ids"])},
            derived=_scope(False, "other_phenotype_background_summary"), raw=dict(summary)))
    return records


def _e2g(gene_id: str, symbol: str | None, category: Mapping[str, Any],
         selected: frozenset[str]) -> list[dict[str, Any]]:
    records = []
    for card_row in category["records"]:
        source = card_row.get("_source") or {}
        row, variant = source.get("row") or {}, source.get("variant")
        if variant is None:
            continue
        biosample = row.get("biosampleName") or ((row.get("biosample") or {}).get("biosampleName"))
        biosample_id = (row.get("biosample") or {}).get("biosampleId")
        chromosome, start, end = row.get("chromosome"), row.get("start"), row.get("end")
        key = f"chr{chromosome}:{start}-{end}" if chromosome and start is not None and end is not None else None
        identifier = upstream_id("regulatory", gene_id, card_row["physical_annotation_sha256"])
        card_row["evidence_id"] = identifier
        role = "selected" if variant.variant_id in selected else "ld_proxy"
        records.append(_record(
            dataset="enhancer_to_gene", module="locus.enhancer_to_gene",
            relation="variant_overlaps_regulatory_interval_linked_to_target", identifier=identifier,
            subject=_variant_subject(variant), gene_id=gene_id, symbol=symbol,
            context={"biosample": biosample, "biosample_id": biosample_id, "datasource_id": row.get("datasourceId"),
                     "distance_to_tss": row.get("distanceToTss"), "enhancer_study_id": row.get("studyId"),
                     "exact_variant_overlap": True, "interval_type": row.get("intervalType"),
                     "overlap_variant_id": variant.variant_id, "overlap_variant_role": role,
                     "overlap_chromosome": chromosome, "overlap_position": variant.position,
                     "regulatory_chromosome": chromosome, "regulatory_start": start, "regulatory_end": end,
                     "regulatory_element_key": key, "study_id": row.get("studyId"), "pmid": _text(row.get("pmid")),
                     "locus_anchor_witnesses": card_row["witnesses"],
                     "physical_annotation_sha256": card_row["physical_annotation_sha256"]},
            measurements={"score": row.get("score"), "resource_scores": row.get("resourceScore") or []},
            derived={**_scope(True, "exact_variant_enhancer_link"), "exact_variant_interval_overlap": True},
            raw={"geneId": gene_id, "approvedSymbol": symbol, "biosampleName": biosample,
                 "biosampleFromSourceId": biosample_id, "chromosome": chromosome, "start": start, "end": end,
                 "datasourceId": row.get("datasourceId"), "distanceToTss": row.get("distanceToTss"),
                 "enhancerStudyId": row.get("studyId"), "studyId": row.get("studyId"),
                 "intervalType": row.get("intervalType"), "exactVariantOverlap": True,
                 "overlapVariantId": variant.variant_id, "overlapVariantRole": role, "pmid": row.get("pmid"),
                 "resourceScore": row.get("resourceScore") or [], "score": row.get("score"),
                 "variantId": variant.variant_id, "queryVariantId": variant.variant_id,
                 "regulatoryElementKey": key, "locusAnchorWitnesses": card_row["witnesses"],
                 "physicalAnnotationSha256": card_row["physical_annotation_sha256"],
                 "upstreamE2G": dict(row)}))
    return records


def _expression_rows(target: Mapping[str, Any], preferred: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """Bound complete, distinct observations; abundance and specificity are independently usable."""

    def finite(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            number = float(value)
        except (OverflowError, ValueError):
            return None
        return number if math.isfinite(number) else None

    preferred_order = {json.dumps(row, sort_keys=True): index for index, row in reversed(list(enumerate(preferred)))}
    rows = _unique_rows([*preferred, *((target.get("baselineExpression") or {}).get("rows") or [])])
    rows = [row for row in rows if finite(row.get("median")) is not None
            or finite(row.get("specificity_score")) is not None]

    def order(row):
        key = json.dumps(row, sort_keys=True)
        median, specificity = finite(row.get("median")), finite(row.get("specificity_score"))
        return (preferred_order.get(key, len(preferred)), str(row.get("datasourceId")),
                str(row.get("datatypeId")), str(row.get("unit")), median is None,
                -(median if median is not None else 0.0), specificity is None,
                -(specificity if specificity is not None else 0.0))

    rows.sort(key=order)
    return rows[:parameter("v2g.ledger.expression_rows")]


def _gene_level(gene_id: str, symbol: str | None, card: dict[str, Any], target: Mapping[str, Any] | None,
                association: Mapping[str, Any] | None, disease: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    evidence = card["evidence"]
    subject = {"type": "target", "id": gene_id, "label": symbol}
    records = []

    def add(category: str, dataset: str, module: str, relation: str, raw: Mapping[str, Any],
            context: Mapping[str, Any] | None = None, identity: str | None = None) -> str:
        identifier = upstream_id(module, gene_id, dataset, identity) if identity else upstream_id(module, gene_id, dataset)
        evidence[category].setdefault("evidence_ids", []).append(identifier)
        records.append(_record(dataset=dataset, module=module, relation=relation, identifier=identifier,
                               subject=subject, context=context, derived=_scope(False, "gene_level_context"),
                               raw=raw))
        return identifier

    target = target or {}
    expression = _expression_rows(target, [row["_source_row"] for row in evidence["expression"].get("context_matched", [])
                                          if row.get("_source_row")])
    if expression:
        add("expression", "expression", "target.expression", "target_expression",
            {"approvedSymbol": symbol, "tissues": {"rows": expression}})
    if target.get("functionDescriptions"):
        add("function", "target", "target.core", "target_core",
            {"id": gene_id, "approvedSymbol": symbol, "approvedName": target.get("approvedName"),
             "biotype": target.get("biotype"), "functionDescriptions": target.get("functionDescriptions")})
    if target.get("geneOntology") or target.get("pathways"):
        add("function", "target", "target.go", "target_go",
            {"geneOntology": target.get("geneOntology") or [], "pathways": target.get("pathways") or []})
    if target.get("geneticConstraint"):
        add("constraint", "genetic_constraint", "target.constraint", "target_genetic_constraint",
            {"geneticConstraint": target.get("geneticConstraint")})
    association_rows = evidence["association"].get("records", [])
    if not association_rows and association:
        association_rows = [{"disease": disease, "_source": {"association": association, "disease": disease}}]
    for card_row in association_rows:
        source = card_row.get("_source") or {}
        observed_disease = source.get("disease") or card_row.get("disease") or {}
        identifier = add("association", "association_by_datasource", "association.target_disease", "target_disease_association",
            {"target": {"id": gene_id, "approvedSymbol": symbol}, "disease": dict(observed_disease),
             "score": card_row.get("overall"),
             "datatypeScores": [{"id": key, "score": value} for key, value in card_row.get("datatypes", {}).items()],
             "datasourceScores": [{"id": key, "score": value} for key, value in card_row.get("datasources", {}).items()],
             "excludedDatasourceIds": card_row.get("excluded_datasources", []),
             "literatureInterpretation": "Literature co-occurrence; not locus evidence"},
            context={"disease_id": observed_disease.get("id"), "disease_name": observed_disease.get("name"),
                     "phenotype_evidence_scope": card_row.get("phenotype_scope")},
            identity=observed_disease.get("id"))
        card_row["evidence_id"] = identifier
    return records


def region_summary(locus: LocusVariantSet, window: GeneWindow, flank_bp: int) -> dict[str, Any]:
    """The single auditable lead-window gene census the region gate reads."""

    status = "complete" if window.status == "complete" else "unavailable"
    return _record(
        dataset=REGION_DATASET, module="locus.variant", relation="regional_gene_context_summary",
        identifier=upstream_id("region", locus.lead_variant_id, flank_bp),
        subject={"type": "variant", "id": locus.lead_variant_id},
        context={"candidate_gene_ids": [gene.gene_id for gene in window.genes], "flank_bp": flank_bp,
                 "source": window.source},
        measurements={"annotated_gene_count": len(window.genes)},
        derived={"status": status, "count_is_lower_bound": False, "truncated": False})


def semantic_records(cards: Sequence[dict[str, Any]], locus: LocusVariantSet,
                     bundle: RetrievalBundle) -> list[dict[str, Any]]:
    """Semantic records for every card record; stamps the upstream identifier onto the cards."""

    by_query = locus.by_query_id()
    selected = frozenset(variant.variant_id for variant in locus.variants
                         if "selected" in variant.sources or "lead" in variant.sources)
    records: list[dict[str, Any]] = []
    for card in cards:
        gene_id, symbol, evidence = card["gene_id"], card["identity"].get("symbol"), card["evidence"]
        records += _coding(gene_id, symbol, evidence["coding"])
        records += _qtl(gene_id, symbol, evidence["qtl"])
        records += _coloc(gene_id, symbol, evidence["coloc"], by_query)
        records += _e2g(gene_id, symbol, evidence["e2g"], selected)
        records += _gene_level(gene_id, symbol, card, bundle.targets.get(gene_id),
                               bundle.associations.get(gene_id), bundle.association_disease)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["provenance"]["evidence_id"], record)
    return list(unique.values())


def remap_card_evidence_ids(cards: Sequence[dict[str, Any]], mapping: Mapping[str, str]) -> list[dict[str, Any]]:
    """Replace upstream identifiers on card records with ledger evidence IDs; unknown ones are dropped."""

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "evidence_id":
                    if item in mapping:
                        result[key] = mapping[item]
                elif key == "evidence_ids":
                    result[key] = [mapping[entry] for entry in item if entry in mapping]
                else:
                    result[key] = visit(item)
            return result
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return [visit(card) for card in cards]
