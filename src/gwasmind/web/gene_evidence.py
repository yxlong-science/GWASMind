from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..models import EvidenceKind, EvidenceRecord, RankedGene
from ..staged_reasoning import has_direct_locus_gene_evidence


@dataclass(frozen=True, slots=True)
class EvidenceTrace:
    """Stable links back to the full evidence ledger and its publications."""

    evidence_ids: tuple[str, ...]
    pmids: tuple[str, ...] = ()
    study_ids: tuple[str, ...] = ()
    datasource_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MolecularQTLEvidence:
    subtype: str
    tissue: str | None
    p_value: float | None
    beta: float | None
    variant_id: str | None
    study_id: str | None
    credible_set_id: str | None
    trace: EvidenceTrace
    representative_evidence_id: str | None = None
    variant_origin: str = "unknown"
    measurement_scope: str = "representative_member"
    locus_anchor_witnesses: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ColocalizationEvidence:
    qtl_subtype: str
    tissue: str | None
    trait: str | None
    h4: float | None
    qtl_study_locus_id: str | None
    other_study_locus_id: str | None
    other_study_id: str | None
    trace: EvidenceTrace
    representative_evidence_id: str | None = None
    query_variant_id: str | None = None
    qtl_study_id: str | None = None
    variant_origin: str = "unknown"


@dataclass(frozen=True, slots=True)
class EnhancerGeneEvidence:
    regulatory_element_key: str | None
    location: str | None
    interval_type: str | None
    overlap_variant_id: str | None
    overlap_variant_role: str | None
    exact_variant_overlap: bool
    tissue: str | None
    biosample_id: str | None
    score: float | None
    resource_scores: tuple[tuple[str, float], ...]
    distance_to_tss: int | None
    datasource_id: str | None
    study_id: str | None
    trace: EvidenceTrace
    variant_origin: str = "unknown"


@dataclass(frozen=True, slots=True)
class ExpressionMeasurement:
    biosample: str
    biosample_id: str | None
    value: float | str
    metric: str
    trace: EvidenceTrace


@dataclass(frozen=True, slots=True)
class ExpressionGroup:
    """Comparable expression values from one datasource, datatype, and unit only."""

    datasource_id: str
    datatype_id: str | None
    unit: str | None
    measurements: tuple[ExpressionMeasurement, ...]


@dataclass(frozen=True, slots=True)
class FunctionDescription:
    text: str
    trace: EvidenceTrace


@dataclass(frozen=True, slots=True)
class GeneOntologyEvidence:
    go_id: str | None
    label: str
    aspect: str | None
    source: str | None
    trace: EvidenceTrace


@dataclass(frozen=True, slots=True)
class ReactomePathwayEvidence:
    pathway_id: str | None
    name: str
    category: str | None
    trace: EvidenceTrace


@dataclass(frozen=True, slots=True)
class GeneFunctionEvidence:
    descriptions: tuple[FunctionDescription, ...] = ()
    gene_ontology: tuple[GeneOntologyEvidence, ...] = ()
    pathways: tuple[ReactomePathwayEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class GeneEvidence:
    gene_id: str
    symbol: str | None
    rank: int | None
    shortlist_sources: tuple[str, ...]
    molecular_qtl: tuple[MolecularQTLEvidence, ...] = ()
    colocalizations: tuple[ColocalizationEvidence, ...] = ()
    enhancer_gene: tuple[EnhancerGeneEvidence, ...] = ()
    expression: tuple[ExpressionGroup, ...] = ()
    function: GeneFunctionEvidence = GeneFunctionEvidence()
    coding: tuple[FunctionDescription, ...] = ()

    @property
    def label(self) -> str:
        return self.symbol or self.gene_id


@dataclass(slots=True)
class _MutableTrace:
    evidence_ids: set[str]
    pmids: set[str]
    study_ids: set[str]
    datasource_ids: set[str]


@dataclass(slots=True)
class _Candidate:
    gene_id: str
    symbol: str | None
    rank: int | None
    sources: set[str]


def aggregate_gene_evidence(
    records: Sequence[EvidenceRecord],
    *,
    ranked_genes: Sequence[str | RankedGene | Mapping[str, Any]] = (),
    shortlisted_genes: Sequence[str | Mapping[str, Any]] = (),
    colocalization_phenotype_match: Callable[[str], bool] | None = None,
    variant_origins: Mapping[str, str] | None = None,
) -> tuple[GeneEvidence, ...]:
    """Build a deduplicated, per-gene evidence view for the user-facing answer.

    L2G records are deliberately used only to discover an unordered shortlist.  Their
    score, publication, and record contents never enter any visible evidence dimension.
    Explicit ``ranked_genes`` remain authoritative; L2G-only genes are appended in
    alphabetical order so upstream L2G ordering cannot leak into the result.
    """

    identities, symbols = _gene_identities(records)
    candidates: dict[str, _Candidate] = {}
    order: list[str] = []

    ranked_items = sorted(
        enumerate(ranked_genes),
        key=lambda item: (_declared_rank(item[1], item[0] + 1), item[0]),
    )
    for index, item in ranked_items:
        token, declared_rank = _candidate_token(item)
        gene_id, symbol = _resolve_candidate(token, identities, symbols)
        _add_candidate(
            candidates,
            order,
            gene_id,
            symbol,
            source="ranked",
            rank=declared_rank or index + 1,
        )

    for item in shortlisted_genes:
        token, _ = _candidate_token(item)
        gene_id, symbol = _resolve_candidate(token, identities, symbols)
        _add_candidate(
            candidates,
            order,
            gene_id,
            symbol,
            source="explicit_shortlist",
            rank=None,
        )

    l2g_genes: dict[str, str | None] = {}
    evidence_genes: dict[str, str | None] = {}
    for record in records:
        gene_id, symbol = _record_gene(record)
        if gene_id is None:
            continue
        if record.kind is EvidenceKind.L2G:
            l2g_genes.setdefault(gene_id, symbol)
        else:
            evidence_genes.setdefault(gene_id, symbol)

    for gene_id in sorted(l2g_genes, key=lambda value: (l2g_genes[value] or value).casefold()):
        _add_candidate(
            candidates,
            order,
            gene_id,
            l2g_genes[gene_id],
            source="l2g_shortlist",
            rank=None,
        )
    for gene_id in sorted(
        evidence_genes, key=lambda value: (evidence_genes[value] or value).casefold()
    ):
        _add_candidate(
            candidates,
            order,
            gene_id,
            evidence_genes[gene_id],
            source="direct_evidence",
            rank=None,
        )

    qtls: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    colocs: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    enhancers: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    expression: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    descriptions: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    coding: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    go_terms: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    pathways: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)

    for record in records:
        if record.kind is EvidenceKind.L2G:
            continue
        gene_id, _ = _record_gene(record)
        if gene_id is None:
            continue
        origin = _analysis_origin(record, variant_origins)
        if record.kind is EvidenceKind.QTL:
            _collect_qtl(qtls[gene_id], record, origin=origin)
        elif record.kind is EvidenceKind.COLOCALIZATION:
            _collect_colocalization(colocs[gene_id], record, origin=origin)
        elif record.kind is EvidenceKind.REGULATORY:
            _collect_enhancer(enhancers[gene_id], record, origin=origin)
        elif record.kind is EvidenceKind.EXPRESSION:
            _collect_expression(expression[gene_id], record)
        elif record.kind in {EvidenceKind.FUNCTION, EvidenceKind.PATHWAY}:
            _collect_function(descriptions[gene_id], go_terms[gene_id], pathways[gene_id], record)
        elif (
            record.kind is EvidenceKind.VARIANT_ANNOTATION
            and has_direct_locus_gene_evidence((record,))
        ):
            raw = _raw_payload(record)
            consequence = _first_text((raw, record.payload), (
                "consequence", "consequenceLabel", "consequence_label", "consequenceId",
            ))
            if consequence is None:
                for key in ("mostSevereConsequence", "variantFunctionalConsequence"):
                    value = raw.get(key)
                    if isinstance(value, Mapping):
                        consequence = _first_text((value,), ("label", "term", "id"))
                    elif isinstance(value, str):
                        consequence = value
                    if consequence:
                        break
            annotation = consequence or "high-impact annotation (subtype not reported)"
            variant = record.variant_id or _first_text((raw,), ("variantId",))
            text = (
                f"{_variant_role_label(origin)} `{variant}`: {annotation}; source: {record.source}."
                if variant else f"{annotation}; source: {record.source}."
            )
            _merge_item(coding[gene_id], (variant, annotation), {"text": text}, record)

    result: list[GeneEvidence] = []
    for gene_id in order:
        candidate = candidates[gene_id]
        result.append(
            GeneEvidence(
                gene_id=gene_id,
                symbol=candidate.symbol,
                rank=candidate.rank,
                shortlist_sources=tuple(sorted(candidate.sources)),
                molecular_qtl=_finalize_qtls(qtls[gene_id]),
                colocalizations=_finalize_colocalizations(
                    colocs[gene_id], phenotype_matches=colocalization_phenotype_match,
                ),
                enhancer_gene=_finalize_enhancers(enhancers[gene_id]),
                expression=_finalize_expression(expression[gene_id]),
                function=GeneFunctionEvidence(
                    descriptions=_finalize_descriptions(descriptions[gene_id]),
                    gene_ontology=_finalize_go(go_terms[gene_id]),
                    pathways=_finalize_pathways(pathways[gene_id]),
                ),
                coding=_finalize_descriptions(coding[gene_id]),
            )
        )
    return tuple(result)


CitationResolver = Callable[[EvidenceTrace], str | int | None]


def render_gene_evidence_markdown(
    genes: Sequence[GeneEvidence],
    *,
    summary: str | None = None,
    biological_context: str | None = None,
    citation_resolver: CitationResolver | None = None,
    references: Sequence[str] = (),
) -> str:
    """Render a consistent report hierarchy from verified evidence fields.

    Publication ownership stays outside this module.  ``citation_resolver`` can be wired
    to the publication registry using ``EvidenceTrace``; until then, no citation is
    invented.  The default HTML comments are safe placeholders for ``chat_response.py``.
    """

    summary_text = summary or (
        "<!-- Summary: add one phenotype-aware comparative paragraph here; "
        "use only the visible biological evidence. -->"
    )
    lines = [
        "## **Summary**",
        "",
        summary_text,
        "",
    ]
    if biological_context:
        lines.extend(["## **Biological Context**", "", biological_context, ""])
    lines.extend(["## **Evidence**", ""])

    if not genes:
        lines.extend(["No gene-specific evidence details are displayed in this report.", ""])

    for gene in genes:
        identifier = (
            f" (`{_markdown(gene.gene_id)}`)" if gene.label != gene.gene_id else ""
        )
        lines.extend([f"### {_markdown(gene.label)}{identifier}", ""])
        lines.extend(["#### Molecular QTL/Colocalization", ""])
        if not gene.molecular_qtl and not gene.colocalizations:
            lines.extend(["No direct molecular QTL or colocalization evidence was reported.", ""])
        else:
            for item in gene.molecular_qtl[:3]:
                tissue = _markdown(item.tissue or "tissue not reported")
                details = []
                if item.p_value is not None:
                    details.append(f"P={_scientific(item.p_value)}")
                if item.beta is not None:
                    details.append(f"Beta={_number(item.beta)}")
                detail_text = f": {', '.join(details)}" if details else ""
                representative = _qtl_measurement_label(item) if details else ""
                citation = _citation_suffix(item.trace, citation_resolver)
                lines.append(
                    f"- {_markdown(_display_qtl_type(item.subtype))} ({tissue})"
                    f"{detail_text}{representative}{citation}"
                )
            lines.extend(_render_colocalizations(gene.colocalizations, citation_resolver))
            lines.append("")

        lines.extend(["#### Regulatory Region-to-Gene (E2G)", ""])
        if not gene.enhancer_gene:
            lines.extend(
                [
                    (
                        "No exact selected-variant overlap with an E2G regulatory "
                        "interval was returned."
                    ),
                    "",
                ]
            )
        else:
            for item in gene.enhancer_gene[:3]:
                location = _markdown(item.location or "location not reported")
                attributes = []
                if item.distance_to_tss is not None:
                    attributes.append(
                        f"distance to {_markdown(gene.label)} TSS={item.distance_to_tss:,} bp"
                    )
                for display_name, aliases in (
                    ("DNase", ("dnase",)),
                    ("HiC", ("hic", "hic_contacts", "hic contacts")),
                ):
                    score = _named_resource_score(item.resource_scores, aliases)
                    if score is not None:
                        attributes.append(f"{display_name}={_number(score)}")
                attribute_text = f"; {', '.join(attributes)}" if attributes else ""
                variant = _markdown(item.overlap_variant_id or "variant not reported")
                role = _variant_role_label(item.variant_origin)
                overlap = " overlaps " if item.exact_variant_overlap else " is linked to "
                citation = _citation_suffix(item.trace, citation_resolver)
                lines.append(
                    f"- {role} `{variant}`{overlap}the reported regulatory interval "
                    f"`{location}`, linked to "
                    f"**{_markdown(gene.label)}**"
                    f"{attribute_text}{citation}"
                )
            lines.append("")

        lines.extend(["#### Expression", ""])
        if not gene.expression:
            lines.extend(["No expression data was reported.", ""])
        else:
            for group in gene.expression:
                unit = group.unit or "unit not reported"
                datatype = f"; datatype={group.datatype_id}" if group.datatype_id else ""
                lines.append(
                    f"*Datasource: {_markdown(group.datasource_id)}; "
                    f"unit={_markdown(unit)}{_markdown(datatype)}*"
                )
                for item in group.measurements:
                    value = _number(item.value) if isinstance(item.value, float) else item.value
                    citation = _citation_suffix(item.trace, citation_resolver)
                    lines.append(
                        f"- {_markdown(item.biosample)}: {_markdown(str(value))}"
                        f"{citation}"
                    )
                lines.append("")

        lines.extend(["#### Function", ""])
        function = gene.function
        for item in gene.coding[:3]:
            citation = _citation_suffix(item.trace, citation_resolver)
            chunks = re.split(r"(`[^`\n]*`)", item.text)
            annotation = "".join(
                chunk if index % 2 else _markdown(chunk).replace("_", "\\_")
                for index, chunk in enumerate(chunks)
            )
            lines.append(f"- Coding/splice annotation: {annotation}{citation}")
        if (
            not gene.coding and not function.descriptions
            and not function.gene_ontology and not function.pathways
        ):
            lines.extend(
                [
                    "No function, Gene Ontology, or Reactome pathway annotation was reported.",
                    "",
                ]
            )
        else:
            for item in function.descriptions[:1]:
                citation = _citation_suffix(item.trace, citation_resolver)
                description = _without_inline_pubmed(item.text)
                if len(description) > 320:
                    sentence_end = description.rfind(". ", 0, 320)
                    description = (
                        description[: sentence_end + 1]
                        if sentence_end >= 100
                        else description[:317].rstrip() + "..."
                    )
                lines.append(f"- Function: {_markdown(description)}{citation}")
            for item in function.gene_ontology[:3]:
                aspect = f" ({_markdown(_display_go_aspect(item.aspect))})" if item.aspect else ""
                go_id = f" [{_markdown(item.go_id)}]" if item.go_id else ""
                citation = _citation_suffix(item.trace, citation_resolver)
                lines.append(f"- GO: {_markdown(item.label)}{aspect}{go_id}{citation}")
            for item in function.pathways[:2]:
                category = f" ({_markdown(item.category)})" if item.category else ""
                pathway_id = f" [{_markdown(item.pathway_id)}]" if item.pathway_id else ""
                citation = _citation_suffix(item.trace, citation_resolver)
                lines.append(f"- Pathway: {_markdown(item.name)}{category}{pathway_id}{citation}")
            lines.append("")

    lines.extend(["## **References**", ""])
    if references:
        for reference in references:
            # A reference is always exactly one physical Markdown line.  This is
            # enforced here even if a future resolver accidentally supplies
            # wrapped or multi-line metadata.
            lines.extend([" ".join(reference.split()), ""])
    else:
        lines.append(
            "<!-- References: supplied by the publication registry after citation use. -->"
        )
    rendered = "\n".join(lines).rstrip() + "\n"
    body, separator, bibliography = rendered.partition("**References**")
    return (
        italicize_gene_symbols(body, (gene.symbol for gene in genes if gene.symbol))
        + separator + bibliography
    )


def italicize_gene_symbols(markdown: str, symbols) -> str:
    """Emphasize human gene symbols in bold italics, preserving IDs and protein labels.

    Keep the existing helper name for callers; the extra weight makes gene names
    readable in narrow chat panels and carries through Markdown exports.
    """

    known = sorted(
        {str(symbol) for symbol in symbols
         if symbol and not str(symbol).upper().startswith("ENSG")},
        key=len, reverse=True,
    )
    if not known:
        return markdown
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?P<emphasis>\*{1,3}|_{1,3})?(?P<symbol>"
        + "|".join(re.escape(symbol) for symbol in known)
        + r")(?(emphasis)(?P=emphasis))(?![A-Za-z0-9_])"
    )
    # Keep literal code and link destinations unchanged. Link labels remain text
    # and may contain a biological gene symbol that needs italic formatting.
    chunks = re.split(r"(```[\s\S]*?```|`[^`\n]*`|\]\([^\n)]*\))", markdown)
    for index in range(0, len(chunks), 2):
        text = chunks[index]

        def replace(match: re.Match[str]) -> str:
            before = text[max(0, match.start() - 25):match.start()]
            after = text[match.end():match.end() + 25]
            if re.search(r"\bprotein\s+$", before, re.IGNORECASE) or re.match(
                r"\s+(?:protein|polypeptide)\b", after, re.IGNORECASE
            ):
                return match.group("symbol")
            return f"***{match.group('symbol')}***"

        chunks[index] = pattern.sub(replace, text)
    return "".join(chunks)


def _gene_identities(
    records: Sequence[EvidenceRecord],
) -> tuple[dict[str, str], dict[str, str | None]]:
    identities: dict[str, str] = {}
    symbols: dict[str, str | None] = {}
    for record in records:
        gene_id, symbol = _record_gene(record)
        if gene_id is None:
            continue
        identities.setdefault(gene_id.casefold(), gene_id)
        if symbol:
            identities.setdefault(symbol.casefold(), gene_id)
        symbols[gene_id] = symbols.get(gene_id) or symbol
    return identities, symbols


def _record_gene(record: EvidenceRecord) -> tuple[str | None, str | None]:
    payload = record.payload
    raw = _raw_payload(record)
    subject = payload.get("subject") if isinstance(payload.get("subject"), Mapping) else {}
    object_value = payload.get("object") if isinstance(payload.get("object"), Mapping) else {}
    gene_id = _text(record.gene_id)
    if gene_id is None and object_value.get("type") == "target":
        gene_id = _text(object_value.get("id"))
    if gene_id is None and subject.get("type") == "target":
        gene_id = _text(subject.get("id"))
    gene_id = gene_id or _first_text((raw, payload), ("geneId", "qtlGeneId", "id", "targetId"))
    symbol = _first_text(
        (raw, object_value, subject, payload),
        ("approvedSymbol", "qtlGeneSymbol", "geneSymbol", "label"),
    )
    return gene_id or symbol, symbol


def _candidate_token(item: str | RankedGene | Mapping[str, Any]) -> tuple[str, int | None]:
    if isinstance(item, RankedGene):
        return item.gene_id, item.rank
    if isinstance(item, Mapping):
        token = _first_text((item,), ("gene_id", "geneId", "id", "symbol", "gene"))
        if token is None:
            raise ValueError("candidate gene mapping requires an identifier")
        rank = _integer(item.get("rank"))
        return token, rank
    token = _text(item)
    if token is None:
        raise ValueError("candidate gene identifier cannot be empty")
    return token, None


def _declared_rank(item: str | RankedGene | Mapping[str, Any], fallback: int) -> int:
    _, rank = _candidate_token(item)
    return rank or fallback


def _resolve_candidate(
    token: str, identities: Mapping[str, str], symbols: Mapping[str, str | None]
) -> tuple[str, str | None]:
    gene_id = identities.get(token.casefold(), token)
    symbol = symbols.get(gene_id)
    if symbol is None and not token.upper().startswith("ENSG"):
        symbol = token
    return gene_id, symbol


def _add_candidate(
    candidates: dict[str, _Candidate],
    order: list[str],
    gene_id: str,
    symbol: str | None,
    *,
    source: str,
    rank: int | None,
) -> None:
    if gene_id not in candidates:
        candidates[gene_id] = _Candidate(gene_id, symbol, rank, {source})
        order.append(gene_id)
        return
    candidate = candidates[gene_id]
    candidate.symbol = candidate.symbol or symbol
    candidate.rank = candidate.rank if candidate.rank is not None else rank
    candidate.sources.add(source)


def _analysis_origin(record: EvidenceRecord, overrides: Mapping[str, str] | None) -> str:
    if overrides is not None:
        value = overrides.get(record.variant_id or "", "unknown")
    else:
        _payload, _raw, _measurements, _member, context = _record_views(record)
        value = context.get("analysis_origin")
    return value if value in {"source_selected", "ld_proxy"} else "unknown"


def _variant_role_label(origin: str) -> str:
    return {"source_selected": "Source-selected variant", "ld_proxy": "LD proxy"}.get(origin, "Analyzed variant")


def _qtl_measurement_label(item: MolecularQTLEvidence) -> str:
    if not item.variant_id:
        return ""
    role = "LD proxy" if item.variant_origin == "ld_proxy" else _variant_role_label(item.variant_origin).lower()
    study = f"; study `{_markdown(item.study_id)}`" if item.study_id else ""
    if item.measurement_scope == "credible_set_lead":
        roles = {"source_selected": "user-selected variant", "ld_proxy": "LD-linked variant"}
        witnesses = " / ".join(f"{roles.get(origin, 'analyzed variant')} `{_markdown(variant)}`"
                               for variant, origin in item.locus_anchor_witnesses)
        overlap = f"; locus overlap: {witnesses}" if witnesses else ""
        return f" (displayed statistics: credible-set lead variant `{_markdown(item.variant_id)}`{study}{overlap})"
    return f" (displayed statistics: representative {role} `{_markdown(item.variant_id)}`{study})"


def _colocalization_measurement_label(item: ColocalizationEvidence) -> str:
    details = []
    if item.qtl_study_id and item.other_study_id:
        details.append(f"H4 study pair: `{_markdown(item.qtl_study_id)}` vs `{_markdown(item.other_study_id)}`")
    if item.query_variant_id:
        role = "LD proxy" if item.variant_origin == "ld_proxy" else _variant_role_label(item.variant_origin).lower()
        details.append(f"representative lookup through {role} `{_markdown(item.query_variant_id)}`")
    return " (" + "; ".join(details) + ")" if details else ""


def _collect_qtl(values: dict[tuple[Any, ...], dict[str, Any]], record: EvidenceRecord, *, origin: str = "unknown") -> None:
    payload, raw, measurements, member, context = _record_views(record)
    lead_scope = context.get("measurement_scope") == "credible_set_lead"
    subtype = (
        _first_text(
            (context, measurements, member, raw, payload),
            ("study_type", "studyType", "qtlStudyType"),
        )
        or "qtl"
    )
    tissue = record.tissue or _first_text(
        (context, raw, payload), ("biosample", "biosampleName", "tissue", "biosampleId")
    )
    measured_views = (measurements,) if lead_scope else (measurements, member, raw, payload)
    mantissa = _first_number(measured_views, ("pValueMantissa",))
    exponent = _first_number(measured_views, ("pValueExponent",))
    p_value = _first_number(measured_views, ("pValue", "p_value"))
    if p_value is None and mantissa is not None and exponent is not None and exponent.is_integer():
        try:
            p_value = mantissa * (10.0 ** exponent)
        except OverflowError:
            p_value = None
    if p_value is not None and (not math.isfinite(p_value) or not 0 < p_value <= 1):
        p_value = None
    beta = _first_number(measured_views, ("beta",))
    variant_id = (_first_text((context,), ("measured_variant_id", "cs_lead_variant_id")) if lead_scope else
                  record.variant_id or _first_text(
        (member, raw, payload), ("variantId", "queryVariantId", "qtlVariantId")
    ))
    witnesses: tuple[tuple[str, str], ...] = ()
    if lead_scope:
        # EvidenceRecord.variant_id owns the local overlap; it need not own the statistics.
        witness = (_first_text((member,), ("variantId",))
                   or _first_text((member.get("variant", {}) if isinstance(member.get("variant"), Mapping) else {},), ("id",))
                   or record.variant_id)
        anchors = []
        for item in context.get("locus_anchor_witnesses") or ():
            if not isinstance(item, Mapping) or not (anchor := _text(item.get("variant_id"))):
                continue
            sources = set(item.get("variant_sources") or ())
            role = ("source_selected" if sources.intersection({"selected", "lead"}) else
                    "ld_proxy" if "ld_proxy" in sources else "unknown")
            anchors.append((anchor, role))
        witnesses = tuple(sorted(set(anchors))) if anchors else ((witness, origin),) if witness else ()
        lead_origins = {role for anchor, role in witnesses if anchor == variant_id}
        origin = next(iter(lead_origins)) if len(lead_origins) == 1 else "unknown"
        if not variant_id:
            p_value = beta = None
    study_id = record.study_id or _first_text((context, raw, payload), ("study_id", "studyId"))
    credible_set_id = record.credible_set_id or _first_text(
        (context, raw, payload), ("study_locus_id", "studyLocusId", "qtlStudyLocusId")
    )
    # The scientific unit is the molecular-QTL credible set, not every selected
    # member variant that happened to retrieve the same set.  Keep one
    # representative member (the smallest member P value) while merging every
    # traceable evidence id into that unit.
    key = (subtype.casefold(), _fold(tissue), study_id, credible_set_id,
           "credible_set_lead" if lead_scope else "representative_member", variant_id if lead_scope else None)
    current = _merge_item(
        values,
        key,
        {
            "subtype": subtype,
            "tissue": tissue,
            "p_value": p_value,
            "beta": beta,
            "variant_id": variant_id,
            "study_id": study_id,
            "credible_set_id": credible_set_id,
            "representative_evidence_id": record.evidence_id,
            "variant_origin": origin,
            "measurement_scope": "credible_set_lead" if lead_scope else "representative_member",
            "locus_anchor_witnesses": witnesses,
        },
        record,
        study_ids=(study_id,),
    )
    if lead_scope:
        current["locus_anchor_witnesses"] = tuple(sorted(set(current["locus_anchor_witnesses"]) | set(witnesses)))
    if p_value is not None and (
        current.get("p_value") is None or p_value < current["p_value"]
    ):
        current.update(
            {
                "p_value": p_value,
                "beta": beta,
                "variant_id": variant_id,
                "representative_evidence_id": record.evidence_id,
                "variant_origin": origin,
                "measurement_scope": "credible_set_lead" if lead_scope else "representative_member",
            }
        )


def _collect_colocalization(
    values: dict[tuple[Any, ...], dict[str, Any]], record: EvidenceRecord, *, origin: str = "unknown"
) -> None:
    payload, raw, measurements, _member, context = _record_views(record)
    if payload.get("relation") == "other_phenotype_colocalisation_summary":
        return  # Background counts are not a new individual GWAS-QTL study pair.
    subtype = (
        _first_text(
            (context, raw, payload),
            ("study_type", "studyType", "qtlStudyType", "leftStudyType"),
        )
        or "qtl"
    )
    tissue = record.tissue or _first_text(
        (context, raw, payload), ("biosample", "biosampleName", "tissue", "biosampleId")
    )
    trait = _first_text(
        (raw, payload),
        (
            "rightTraitFromSource",
            "traitFromSource",
            "colocTrait",
            "rightTrait",
            "leftTrait",
        ),
    )
    h4 = _first_number((measurements, raw, payload), ("h4",))
    if h4 is not None and not 0 <= h4 <= 1:
        h4 = None
    qtl_locus = (
        _first_text(
            (raw, payload, context),
            ("qtlStudyLocusId", "queriedStudyLocusId", "study_locus_id"),
        )
        or record.credible_set_id
    )
    other_locus = _first_text((raw, payload), ("otherStudyLocusId", "rightStudyLocusId"))
    other_study = _first_text((raw, payload), ("rightStudyId", "otherStudyId"))
    query_variant = record.variant_id or _first_text((raw, payload), ("queryVariantId", "variantId"))
    qtl_study = record.study_id or _first_text((context, raw, payload), ("study_id", "leftStudyId", "studyId"))
    counterpart = other_locus or other_study or trait
    key = (subtype.casefold(), _fold(tissue), qtl_locus, counterpart)
    existing = _merge_item(
        values,
        key,
        {
            "qtl_subtype": subtype,
            "tissue": tissue,
            "trait": trait,
            "h4": h4,
            "qtl_study_locus_id": qtl_locus,
            "other_study_locus_id": other_locus,
            "other_study_id": other_study,
            "representative_evidence_id": record.evidence_id,
            "query_variant_id": query_variant,
            "qtl_study_id": qtl_study,
            "variant_origin": origin,
        },
        record,
        study_ids=tuple(
            value
            for value in (
                _first_text((raw, payload), ("leftStudyId", "studyId")),
                other_study,
            )
            if value
        ),
        datasource_ids=("open_targets_colocalisation",),
    )
    if h4 is not None and (existing.get("h4") is None or h4 > existing["h4"]):
        existing.update(h4=h4, representative_evidence_id=record.evidence_id,
                        query_variant_id=query_variant, qtl_study_id=qtl_study, variant_origin=origin)


def _collect_enhancer(
    values: dict[tuple[Any, ...], dict[str, Any]], record: EvidenceRecord, *, origin: str = "unknown"
) -> None:
    payload, raw, measurements, _, context = _record_views(record)
    chromosome = _first_text((raw, payload), ("chromosome", "chrom"))
    start = _first_integer((raw, payload), ("start",))
    end = _first_integer((raw, payload), ("end",))
    location = None
    if chromosome and start is not None and end is not None:
        location = f"chr{chromosome.removeprefix('chr')}:{start}-{end}"
    regulatory_element_key = _first_text(
        (context, raw, payload),
        ("regulatory_element_key", "regulatoryElementKey", "intervalId"),
    ) or location
    interval_type = _first_text(
        (context, raw, payload), ("interval_type", "intervalType")
    )
    tissue = record.tissue or _first_text(
        (context, raw, payload), ("biosample", "biosampleName", "tissue", "biosampleId")
    )
    biosample_id = _first_text(
        (context, raw, payload),
        ("biosample_id", "biosampleFromSourceId", "biosampleId"),
    )
    score = _first_number((raw, payload), ("score",))
    datasource = _first_text((raw, payload), ("datasourceId", "source"))
    overlap_variant = (
        _first_text(
            (context, raw, payload),
            (
                "overlap_variant_id",
                "overlapVariantId",
                "variantId",
                "queryVariantId",
                "qtlVariantId",
            ),
        )
        or record.variant_id
    )
    overlap_role = _first_text(
        (context, raw, payload), ("overlap_variant_role", "overlapVariantRole")
    )
    if record.variant_id and overlap_variant and record.variant_id != overlap_variant:
        origin = "unknown"
    exact_overlap = _first_boolean(
        (context, raw, payload),
        ("exact_variant_overlap", "exactVariantOverlap"),
    )
    resource_scores = _regulatory_resource_scores(
        measurements.get("resource_scores") or raw.get("resourceScore")
    )
    distance_to_tss = _first_integer(
        (context, raw, payload), ("distance_to_tss", "distanceToTss")
    )
    study_id = _first_text(
        (context, raw, payload),
        ("enhancer_study_id", "enhancerStudyId", "studyId"),
    ) or record.study_id
    key = (
        chromosome,
        start,
        end,
        _fold(tissue),
        datasource,
        overlap_variant,
        study_id,
    )
    _merge_item(
        values,
        key,
        {
            "regulatory_element_key": regulatory_element_key,
            "location": location,
            "interval_type": interval_type,
            "overlap_variant_id": overlap_variant,
            "overlap_variant_role": overlap_role,
            "variant_origin": origin,
            "exact_variant_overlap": exact_overlap is True,
            "tissue": tissue,
            "biosample_id": biosample_id,
            "score": score,
            "resource_scores": resource_scores,
            "distance_to_tss": distance_to_tss,
            "datasource_id": datasource,
            "study_id": study_id,
        },
        record,
        study_ids=(study_id,),
        datasource_ids=tuple(
            value for value in (datasource, "enhancer_to_gene") if value
        ),
    )


def _collect_expression(
    values: dict[tuple[Any, ...], dict[str, Any]], record: EvidenceRecord
) -> None:
    payload, raw, _, _, _ = _record_views(record)
    rows = raw.get("tissues") or payload.get("tissues")
    if isinstance(rows, Mapping):
        rows = rows.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cell = row.get("celltypeBiosample")
        tissue = row.get("tissueBiosample")
        cell = cell if isinstance(cell, Mapping) else {}
        tissue = tissue if isinstance(tissue, Mapping) else {}
        cell_name = _first_text((cell,), ("biosampleName", "label", "name"))
        tissue_name = _first_text(
            (tissue, row), ("biosampleName", "label", "tissue", "name")
        )
        biosample = (
            f"{cell_name} — {tissue_name}"
            if cell_name and tissue_name
            else cell_name or tissue_name
        )
        if biosample is None:
            continue
        biosample_id = _first_text((cell, tissue, row), ("biosampleId", "efo_code", "id"))
        rna = row.get("rna") if isinstance(row.get("rna"), Mapping) else {}
        protein = row.get("protein") if isinstance(row.get("protein"), Mapping) else {}
        if _number_value(row.get("median")) is not None:
            value: float | str | None = _number_value(row.get("median"))
            metric = "median"
        elif _number_value(row.get("value")) is not None:
            value = _number_value(row.get("value"))
            metric = _text(row.get("metric")) or "value"
        elif _number_value(rna.get("value")) is not None:
            value = _number_value(rna.get("value"))
            metric = "rna"
        elif _number_value(protein.get("level")) is not None:
            value = _number_value(protein.get("level"))
            metric = "protein level"
        else:
            text_value = _text(row.get("value") or rna.get("value") or protein.get("level"))
            if text_value is None:
                continue
            value = text_value
            metric = _text(row.get("metric")) or "value"
        datasource = _first_text((row,), ("datasourceId", "datasource")) or record.dataset
        datatype = _first_text((row,), ("datatypeId", "datatype"))
        if datatype is None:
            datatype = "rna" if rna else "protein" if protein else None
        unit = _first_text((row, rna, protein), ("unit",))
        key = (
            datasource.casefold(),
            _fold(datatype),
            _fold(unit),
            _fold(biosample_id or biosample),
            metric.casefold(),
            value,
        )
        _merge_item(
            values,
            key,
            {
                "datasource_id": datasource,
                "datatype_id": datatype,
                "unit": unit,
                "biosample": biosample,
                "biosample_id": biosample_id,
                "value": value,
                "metric": metric,
            },
            record,
            datasource_ids=(datasource,),
        )


def _collect_function(
    descriptions: dict[tuple[Any, ...], dict[str, Any]],
    go_terms: dict[tuple[Any, ...], dict[str, Any]],
    pathways: dict[tuple[Any, ...], dict[str, Any]],
    record: EvidenceRecord,
) -> None:
    payload, raw, _, _, _ = _record_views(record)
    description_values = raw.get("functionDescriptions") or payload.get("functionDescriptions")
    if isinstance(description_values, (str, Mapping)):
        description_values = [description_values]
    if isinstance(description_values, Sequence):
        for item in description_values:
            if isinstance(item, Mapping):
                text = _first_text((item,), ("description", "text", "value", "label"))
            else:
                text = _text(item)
            if text:
                embedded_pmids = tuple(
                    dict.fromkeys(
                        re.findall(r"\bPubMed\s*:\s*([1-9]\d{0,11})\b", text, re.IGNORECASE)
                    )
                )
                _merge_item(
                    descriptions,
                    (text.casefold(),),
                    {"text": text},
                    record,
                    pmids=embedded_pmids,
                    datasource_ids=("uniprot",),
                )

    go_values = raw.get("geneOntology") or raw.get("go") or payload.get("go")
    if isinstance(go_values, Mapping):
        go_values = [go_values]
    if isinstance(go_values, Sequence) and not isinstance(go_values, (str, bytes)):
        for item in go_values:
            if not isinstance(item, Mapping):
                continue
            term = item.get("term") if isinstance(item.get("term"), Mapping) else item
            go_id = _first_text((term, item), ("id", "goId"))
            label = _first_text((term, item), ("label", "name", "goName")) or go_id
            if label is None:
                continue
            aspect = _first_text((item, term), ("aspect",))
            source = _first_text((item,), ("source",))
            _merge_item(
                go_terms,
                (go_id or label.casefold(), aspect),
                {
                    "go_id": go_id,
                    "label": label,
                    "aspect": aspect,
                    "source": source,
                },
                record,
                datasource_ids=("gene_ontology",),
            )

    pathway_values = raw.get("pathways") or payload.get("pathways")
    if isinstance(pathway_values, Mapping):
        pathway_values = [pathway_values]
    if isinstance(pathway_values, Sequence) and not isinstance(pathway_values, (str, bytes)):
        for item in pathway_values:
            if not isinstance(item, Mapping):
                continue
            pathway_id = _first_text((item,), ("pathwayId", "id"))
            name = _first_text((item,), ("pathway", "pathwayName", "label", "name"))
            if name is None:
                continue
            category = _first_text((item,), ("topLevelTerm", "category"))
            _merge_item(
                pathways,
                (pathway_id or name.casefold(), category),
                {"pathway_id": pathway_id, "name": name, "category": category},
                record,
                datasource_ids=("reactome",),
            )


def _merge_item(
    values: dict[tuple[Any, ...], dict[str, Any]],
    key: tuple[Any, ...],
    item: dict[str, Any],
    record: EvidenceRecord,
    *,
    pmids: Sequence[str | None] = (),
    study_ids: Sequence[str | None] = (),
    datasource_ids: Sequence[str | None] = (),
) -> dict[str, Any]:
    current = values.get(key)
    if current is None:
        current = {**item, "_trace": _MutableTrace(set(), set(), set(), set())}
        values[key] = current
    trace: _MutableTrace = current["_trace"]
    trace.evidence_ids.add(record.evidence_id)
    trace.pmids.update(_record_pmids(record))
    trace.pmids.update(value for value in pmids if value)
    trace.study_ids.update(value for value in study_ids if value)
    trace.study_ids.update(_record_study_ids(record))
    trace.datasource_ids.update(value for value in datasource_ids if value)
    return current


def _record_pmids(record: EvidenceRecord) -> set[str]:
    payload, raw, _, _, context = _record_views(record)
    values = {
        _text(value)
        for value in (
            record.pmid,
            context.get("pmid"),
            raw.get("pmid"),
            raw.get("pubmedId"),
            raw.get("rightPubmedId"),
            raw.get("leftPubmedId"),
            payload.get("pmid"),
        )
    }
    return {value for value in values if value}


def _record_study_ids(record: EvidenceRecord) -> set[str]:
    payload, raw, _, _, context = _record_views(record)
    values = {
        _text(value)
        for value in (
            record.study_id,
            context.get("study_id"),
            raw.get("studyId"),
            raw.get("leftStudyId"),
            raw.get("rightStudyId"),
            payload.get("studyId"),
        )
    }
    return {value for value in values if value}


def _trace(item: Mapping[str, Any]) -> EvidenceTrace:
    value: _MutableTrace = item["_trace"]
    return EvidenceTrace(
        tuple(sorted(value.evidence_ids)),
        tuple(sorted(value.pmids)),
        tuple(sorted(value.study_ids)),
        tuple(sorted(value.datasource_ids)),
    )


def _finalize_qtls(
    values: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[MolecularQTLEvidence, ...]:
    result = [
        MolecularQTLEvidence(
            item["subtype"],
            item["tissue"],
            item["p_value"],
            item["beta"],
            item["variant_id"],
            item["study_id"],
            item["credible_set_id"],
            _trace(item),
            item["representative_evidence_id"],
            item["variant_origin"],
            item["measurement_scope"],
            item["locus_anchor_witnesses"],
        )
        for item in values.values()
    ]
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.subtype.casefold(),
                item.p_value is None,
                item.p_value if item.p_value is not None else math.inf,
                _fold(item.tissue),
            ),
        )[:5]
    )


def _finalize_colocalizations(
    values: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    phenotype_matches: Callable[[str], bool] | None = None,
) -> tuple[ColocalizationEvidence, ...]:
    result = [
        ColocalizationEvidence(
            item["qtl_subtype"],
            item["tissue"],
            item["trait"],
            item["h4"],
            item["qtl_study_locus_id"],
            item["other_study_locus_id"],
            item["other_study_id"],
            _trace(item),
            item["representative_evidence_id"],
            item["query_variant_id"],
            item["qtl_study_id"],
            item["variant_origin"],
        )
        for item in values.values()
    ]
    return tuple(
        sorted(
            result,
            key=lambda item: (
                not (phenotype_matches and phenotype_matches(item.trait or "")),
                item.qtl_subtype.casefold(),
                -(item.h4 if item.h4 is not None else -1),
                _fold(item.trait),
            ),
        )[:3]
    )


def _finalize_enhancers(
    values: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[EnhancerGeneEvidence, ...]:
    result = [
        EnhancerGeneEvidence(
            regulatory_element_key=item["regulatory_element_key"],
            location=item["location"],
            interval_type=item["interval_type"],
            overlap_variant_id=item["overlap_variant_id"],
            overlap_variant_role=item["overlap_variant_role"],
            exact_variant_overlap=item["exact_variant_overlap"],
            tissue=item["tissue"],
            biosample_id=item["biosample_id"],
            score=item["score"],
            resource_scores=item["resource_scores"],
            distance_to_tss=item["distance_to_tss"],
            datasource_id=item["datasource_id"],
            study_id=item["study_id"],
            trace=_trace(item),
            variant_origin=item["variant_origin"],
        )
        for item in values.values()
    ]
    return tuple(
        sorted(
            result,
            key=lambda item: (
                -(item.score if item.score is not None else -1),
                _fold(item.tissue),
                item.location or "",
            ),
        )[:5]
    )


def _finalize_expression(
    values: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[ExpressionGroup, ...]:
    groups: dict[tuple[str, str | None, str | None], list[ExpressionMeasurement]] = defaultdict(
        list
    )
    for item in values.values():
        group_key = (item["datasource_id"], item["datatype_id"], item["unit"])
        groups[group_key].append(
            ExpressionMeasurement(
                item["biosample"],
                item["biosample_id"],
                item["value"],
                item["metric"],
                _trace(item),
            )
        )
    result = []
    for (datasource, datatype, unit), measurements in groups.items():
        measurements.sort(key=_expression_sort_key)
        result.append(ExpressionGroup(datasource, datatype, unit, tuple(measurements[:3])))
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.datasource_id.casefold(),
                _fold(item.datatype_id),
                _fold(item.unit),
            ),
        )[:3]
    )


def _expression_sort_key(item: ExpressionMeasurement) -> tuple[Any, ...]:
    if isinstance(item.value, float):
        return (0, -item.value, item.biosample.casefold())
    return (1, str(item.value).casefold(), item.biosample.casefold())


def _finalize_descriptions(
    values: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[FunctionDescription, ...]:
    return tuple(
        FunctionDescription(item["text"], _trace(item))
        for item in sorted(values.values(), key=lambda value: value["text"].casefold())[:2]
    )


def _finalize_go(
    values: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[GeneOntologyEvidence, ...]:
    return tuple(
        GeneOntologyEvidence(
            item["go_id"], item["label"], item["aspect"], item["source"], _trace(item)
        )
        for item in sorted(values.values(), key=_functional_annotation_key)[:5]
    )


def _finalize_pathways(
    values: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[ReactomePathwayEvidence, ...]:
    return tuple(
        ReactomePathwayEvidence(item["pathway_id"], item["name"], item["category"], _trace(item))
        for item in sorted(
            values.values(),
            key=lambda value: _functional_annotation_key(
                {"label": value["name"], "go_id": value["pathway_id"]}
            ),
        )[:3]
    )


def _render_colocalizations(
    values: Sequence[ColocalizationEvidence], resolver: CitationResolver | None
) -> list[str]:
    grouped: dict[tuple[str, str | None], list[ColocalizationEvidence]] = defaultdict(list)
    for item in values:
        grouped[(item.qtl_subtype, item.tissue)].append(item)
    result = []
    for (subtype, tissue), items in grouped.items():
        # Multiple studies can report the same human-readable trait.  Display
        # the strongest H4 once and merge its provenance instead of repeating
        # an indistinguishable label in the same sentence.
        by_trait: dict[str, ColocalizationEvidence] = {}
        for item in items:
            key = (item.trait or "trait not reported").casefold()
            current = by_trait.get(key)
            if current is None or (item.h4 or -1) > (current.h4 or -1):
                by_trait[key] = item
        items = list(by_trait.values())
        tissue_text = f" ({_markdown(tissue)})" if tissue else ""
        for item in items:
            trait = _markdown(item.trait or "trait not reported")
            h4 = f" (H4={format_probability(item.h4)})" if item.h4 is not None else ""
            representative = _colocalization_measurement_label(item) if item.h4 is not None else ""
            # A citation belongs to this trait/H4 pair, not every trait in the group.
            citation = _citation_suffix(item.trace, resolver)
            result.append(
                f"- {_markdown(_display_qtl_type(subtype))} coloc{tissue_text} "
                f"with {trait}{h4}{representative}{citation}"
            )
    return result


def _citation_suffix(trace: EvidenceTrace, resolver: CitationResolver | None) -> str:
    if resolver is None:
        return ""
    marker = resolver(trace)
    if marker is None or marker == "":
        return ""
    if isinstance(marker, int):
        return f" [{marker}]"
    text = str(marker).strip()
    return f" {text}" if text.startswith("[") else f" [{text}]"


_BIOLOGICAL_PRIORITY_TERMS = (
    "asthma",
    "allerg",
    "immune",
    "inflamm",
    "interleukin",
    "cytokine",
    "t cell",
    "t-cell",
    "lymphocyte",
    "eosinophil",
    "macrophage",
    "mast cell",
    "th2",
)


def _functional_annotation_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    label = str(value.get("label") or "")
    lowered = label.casefold()
    priority = next(
        (index for index, term in enumerate(_BIOLOGICAL_PRIORITY_TERMS) if term in lowered),
        len(_BIOLOGICAL_PRIORITY_TERMS),
    )
    return (priority, label.casefold(), str(value.get("go_id") or ""))


def _record_views(
    record: EvidenceRecord,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    payload = record.payload
    raw = _raw_payload(record)
    measurements = payload.get("measurements")
    measurements = measurements if isinstance(measurements, Mapping) else {}
    context = payload.get("context")
    context = context if isinstance(context, Mapping) else {}
    member = raw.get("matchedLocusMember") or raw.get("qtlMatchedLocusMember")
    member = member if isinstance(member, Mapping) else {}
    return payload, raw, measurements, member, context


def _raw_payload(record: EvidenceRecord) -> Mapping[str, Any]:
    upstream = record.payload.get("upstream")
    if isinstance(upstream, Mapping) and isinstance(upstream.get("raw"), Mapping):
        return upstream["raw"]
    return record.payload


def _first_text(views: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str | None:
    for view in views:
        for key in keys:
            value = _text(view.get(key))
            if value is not None:
                return value
    return None


def _first_number(views: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float | None:
    for view in views:
        for key in keys:
            value = _number_value(view.get(key))
            if value is not None:
                return value
    return None


def _first_integer(views: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> int | None:
    for view in views:
        for key in keys:
            value = _integer(view.get(key))
            if value is not None:
                return value
    return None


def _first_boolean(
    views: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> bool | None:
    for view in views:
        for key in keys:
            value = view.get(key)
            if isinstance(value, bool):
                return value
    return None


def _regulatory_resource_scores(value: Any) -> tuple[tuple[str, float], ...]:
    if isinstance(value, Mapping):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    scores: dict[str, float] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _text(item.get("name"))
        score = _number_value(item.get("value"))
        if name is not None and score is not None:
            scores.setdefault(name, score)
    return tuple(scores.items())


def _named_resource_score(
    scores: Sequence[tuple[str, float]], aliases: Sequence[str]
) -> float | None:
    """Return one E2G component without exposing unrelated implementation scores."""

    normalized_aliases = {
        "".join(character for character in alias.casefold() if character.isalnum())
        for alias in aliases
    }
    for name, value in scores:
        normalized_name = "".join(
            character for character in str(name).casefold() if character.isalnum()
        )
        if normalized_name in normalized_aliases:
            return value
    return None


def _without_inline_pubmed(value: str) -> str:
    """Remove source-style PubMed tokens after converting them to bibliography links."""

    text = re.sub(
        r"\s*\((?:PubMed\s*:\s*[1-9]\d{0,11})(?:\s*,\s*PubMed\s*:\s*[1-9]\d{0,11})*\)",
        "",
        str(value),
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bPubMed\s*:\s*[1-9]\d{0,11}\b(?:\s*,\s*PubMed\s*:\s*[1-9]\d{0,11}\b)*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s{2,}", " ", text)).strip(" .") + "."


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, (Mapping, Sequence)) and not isinstance(value, str):
        return None
    text = str(value).strip()
    return text or None


def _number_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number_value(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _fold(value: str | None) -> str:
    return value.casefold() if value else ""


def _display_qtl_type(value: str | None) -> str:
    mapping = {
        "eqtl": "eQTL",
        "pqtl": "pQTL",
        "sqtl": "sQTL",
        "tuqtl": "tuQTL",
        "sceqtl": "sc-eQTL",
        "scpqtl": "sc-pQTL",
        "scsqtl": "sc-sQTL",
        "sctuqtl": "sc-tuQTL",
        "qtl": "QTL",
    }
    normalized = (value or "qtl").rsplit(".", 1)[-1].casefold()
    return mapping.get(normalized, value or "QTL")


def _display_go_aspect(value: str | None) -> str:
    return {
        "P": "Process",
        "F": "Function",
        "C": "Component",
    }.get((value or "").upper(), value or "")


def _scientific(value: float) -> str:
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}×10{_superscript(int(exponent))}"


def _superscript(value: int) -> str:
    table = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(value).translate(table)


def format_probability(value: float) -> str:
    """Keep an interior probability visibly distinct from exact zero and one."""
    rendered = f"{value:.4g}"
    if 0 < value < 1 and float(rendered) in (0, 1):
        for precision in range(5, 18):
            rendered = format(value, f".{precision}g")
            if 0 < float(rendered) < 1:
                break
    return rendered


def _number(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.4g}"


def _markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("*", "\\*")
