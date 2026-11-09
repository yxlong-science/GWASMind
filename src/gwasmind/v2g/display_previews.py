"""Small, traceable browser previews; never used to select or rank genes."""
from __future__ import annotations

import re
from collections.abc import Mapping

from ..audit import sha256_json
from ..display_numbers import three_significant_figures
from .evidence import _context_matcher, _float

def _display_matcher(biological_context):
    """Use the same frozen OT hierarchy as model-card relevance summaries."""
    return _context_matcher((), biological_context) if biological_context else None


def raw_number(value):
    """Round-trip a finite provider number without turning a near-one score into one."""
    return str(value).removesuffix(".0") if _float(value) is not None else "unavailable"


def _display_number(value):
    number = _float(value)
    return f"{number:.4g}" if number is not None else "unavailable"


def _source(*pairs):
    return [f"{label}: {value}" for label, value in pairs if value is not None and value != ""]


def _record(category, row, summary, source, *, group=None, detail=None):
    raw = row.get("_source_row") or {key: value for key, value in row.items() if not key.startswith("_")}
    result = {"kind": category, "summary": str(summary)[:1200], "source": source,
              "evidence_id": str(row.get("evidence_id") or ""),
              "source_ref": "sha256:" + sha256_json(raw)}
    if group:
        result["group"] = str(group)[:300]
    if detail:
        result["detail"] = str(detail)[:1200]
    return result


def _context_priority(row, matcher):
    if matcher is None:
        return 1
    biosample = {"biosampleId": row.get("biosample_id"), "biosampleName": row.get("biosample")}
    return 0 if matcher(tissue=biosample, cell_type=biosample).get("matched") else 1


def _phenotype_priority(row, scope):
    observed = {item.get("id") if isinstance(item, Mapping) else item for item in row.get("gwas_diseases") or []}
    target_id = (scope.get("target") or {}).get("id")
    if target_id and target_id in observed:
        return 0
    if observed.intersection(item.get("id") for item in scope.get("related") or []):
        return 1
    return 2


def coloc_preview(rows, *, phenotype_scope=None, biological_context=None, summary):
    """Relevance first, then distinct molecular signals and assays within that priority."""
    scope = phenotype_scope or {}
    matcher = _display_matcher(biological_context)
    candidates = sorted(rows, key=lambda row: (_phenotype_priority(row, scope),
        _context_priority(row, matcher), -(_float(row.get("h4")) or 0), str(row.get("evidence_id"))))
    chosen, signals, assays = [], set(), set()
    while candidates and len(chosen) < 3:
        # Relevance is primary. Diversity breaks ties before the strength of another
        # result from a molecular signal already represented in the preview.
        index = min(range(len(candidates)), key=lambda i: (
            _phenotype_priority(candidates[i], scope), _context_priority(candidates[i], matcher),
            candidates[i].get("qtl_study_locus_id") in signals,
            candidates[i].get("qtl_type") in assays,
            -(_float(candidates[i].get("h4")) or 0), i))
        row = candidates.pop(index)
        chosen.append(row)
        if row.get("qtl_study_locus_id"):
            signals.add(row["qtl_study_locus_id"])
        if row.get("qtl_type"):
            assays.add(row["qtl_type"])
    return [_record("coloc", row, summary(row), _source(
        ("GWAS study", row.get("gwas_study_id")), ("GWAS credible set", row.get("gwas_study_locus_id")),
        ("Molecular QTL study", row.get("qtl_study_id")), ("Molecular QTL credible set", row.get("qtl_study_locus_id")),
        ("PP.H4 as reported", raw_number(row.get("h4"))), ("Biosample", row.get("biosample_id")),
        ("Method", row.get("method")),
        ("Phenotype identifiers", ", ".join(str(item.get("id") or "") for item in row.get("gwas_diseases") or []
                                          if isinstance(item, Mapping))))) for row in chosen]


def e2g_order(rows, biological_context=None):
    matcher = _display_matcher(biological_context)
    return sorted(rows, key=lambda row: (_context_priority(row, matcher),
        -(_float(row.get("score")) or 0), str(row.get("evidence_id"))))


def e2g_preview(rows, *, variant_label, biological_context=None):
    ordered = e2g_order(rows, biological_context)
    result = []
    for row in ordered[:3]:
        witnesses = row.get("witnesses") or [{"variant_id": row.get("variant_id")}]
        variants = list(dict.fromkeys(item.get("variant_id") for item in witnesses if item.get("variant_id")))
        labels = [variant_label(identifier) for identifier in variants]
        resources = [item for item in row.get("resource_scores") or []
                     if isinstance(item, Mapping) and item.get("name")]
        scores = [f"{item['name']}={three_significant_figures(item.get('value'))}" for item in resources]
        source = _source(("Study", row.get("study_id")), ("Biosample", row.get("biosample_id")),
            ("E2G score as reported", raw_number(row.get("score"))),
            ("Regulatory interval", f"chr{row['chromosome']}:{row['start']}-{row['end']} (GRCh38)"
             if all(row.get(key) is not None for key in ("chromosome", "start", "end")) else None),
            ("Interval type", row.get("interval_type")), ("Distance to TSS", row.get("distance_to_tss")),
            ("PMID", row.get("pmid")), ("Overlapping variants", ", ".join(variants)))
        source += [f"{item['name']} as reported: {raw_number(item.get('value'))}" for item in resources]
        for witness in witnesses:
            roles = witness.get("source_roles") or witness.get("variant_sources") or []
            if roles:
                source += _source(("Variant origin", f"{witness.get('variant_id')}: {', '.join(roles)}"))
        result.append(_record("e2g", row,
            f"{', '.join(labels) or 'Variant unavailable'} · {row.get('biosample') or 'Biosample unavailable'} · "
            f"{row.get('method') or 'E2G'} score={three_significant_figures(row.get('score'))}", source, detail=" · ".join(scores)))
    return result


def expression_preview(evidence):
    rows = evidence.get("_display_context_rows", evidence.get("context_matched", []))

    def order(row):
        axes = (row.get("context_assessment") or {}).get("axis_assessments") or {}
        tissue = bool((axes.get("tissue") or {}).get("matched"))
        cell = bool((axes.get("cell_type") or {}).get("matched"))
        # Specificity is comparable on the source scale; CPM and TPM abundance is
        # never used together to order rows from different assays.
        specificity = (row.get("_source_row") or {}).get("specificity_score", row.get("specificity"))
        return (not (tissue and cell), not tissue, not cell, -(_float(specificity) or 0),
                str(row.get("datasource")), str(row.get("tissue_id")), str(row.get("cell_type_id")))

    result = []
    for row in sorted(rows, key=order)[:3]:
        raw = row.get("_source_row") or {}
        median = raw.get("median", row.get("median"))
        specificity = raw.get("specificity_score", row.get("specificity"))
        label = " / ".join(str(value) for value in (row.get("tissue"), row.get("cell_type")) if value)
        label = label or row.get("biosample") or "Biosample unavailable"
        unit = row.get("unit") or "unit unavailable"
        datasource = {"tabula_sapiens": "Tabula Sapiens", "gtex": "GTEx"}.get(row.get("datasource"), row.get("datasource"))
        datatype = {"scrna-seq": "scRNA-seq", "bulk rna-seq": "bulk RNA-seq"}.get(row.get("datatype"), row.get("datatype"))
        group = f"{datasource or 'Source unavailable'} · {datatype or 'assay unavailable'} · {unit}"
        abundance = f"{_float(median):.3g}" if _float(median) is not None else "median unavailable"
        summary = f"{label}: {abundance}"
        reference = {**row, "evidence_id": (evidence.get("evidence_ids") or [""])[0]}
        result.append(_record("expression", reference, summary, _source(
            ("Tissue", row.get("tissue_id")), ("Cell type", row.get("cell_type_id")),
            ("Median as reported", raw_number(median)),
            ("Specificity as reported", raw_number(specificity)), ("Specificity method", evidence.get("specificity_method"))),
            group=group))
    return result


def function_preview(evidence):
    """Provider annotations and referenced function text; no new literature claims."""
    result = []
    references = evidence.get("evidence_ids") or [""]
    go = evidence.get("_display_go") or [{"term": {"label": label}} for label in evidence.get("go_terms", [])]
    go = sorted(go, key=lambda row: {"P": 0, "F": 1, "C": 2}.get(row.get("aspect"), 3))
    pathways = evidence.get("_display_pathways") or [{"pathway": label} for label in evidence.get("pathways", [])]
    for group, rows, label_key in (("GO", go, "term"), ("Pathway", pathways, "pathway")):
        labels, source = [], []
        for row in rows:
            term = row.get(label_key)
            label = (term.get("label") if isinstance(term, Mapping) else term) or ""
            if not label:
                continue
            if label not in labels:
                labels.append(label)
            identifier = (term.get("id") if isinstance(term, Mapping) else row.get("pathwayId")) or ""
            aspect = {"P": "biological process", "F": "molecular function", "C": "cellular component"}.get(row.get("aspect"))
            source.append(f"{identifier} {label}".strip() + (f" ({row['aspect']}: {aspect})" if aspect else ""))
        if labels:
            row = {"annotations": rows, "evidence_id": references[-1]}
            excerpt = source[:31] + [f"Annotations: {len(source)}; full source retained"] if len(source) > 32 else source
            result.append(_record("function", row, "; ".join(labels[:2]), excerpt, group=group))
    descriptions = evidence.get("_display_descriptions") or ([evidence["description"]] if evidence.get("description") else [])
    referenced = [str(value) for value in descriptions if re.search(r"(?:PubMed|PMID)\s*:?\s*\d+", str(value), re.IGNORECASE)]
    if referenced:
        full = referenced[0]
        pmids = list(dict.fromkeys(re.findall(r"(?:PubMed|PMID)\s*:?\s*(\d+)", full, re.IGNORECASE)))
        # The summary is an excerpt of provider text; its complete selected source
        # description stays inspectable without pretending to have read a paper.
        excerpt = re.split(r"(?<=[.!?])\s", full, maxsplit=1)[0]
        source = [full[index:index + 1200] for index in range(0, len(full), 1200)]
        source.insert(0, "PMID: " + ", ".join(pmids))
        result.append(_record("function", {"description": full, "evidence_id": references[0]},
                              excerpt[:1100], source[:32], group="Literature · referenced function description"))
    return result[:3]


def association_scope_source(evidence, phenotype_scope):
    """Separate successful retrieval from a returned gene association."""
    scope = phenotype_scope or {}
    identities = [("Target phenotype", scope["target"])] if scope.get("target") else []
    identities += [("Related phenotype", row) for row in scope.get("related") or []]
    if not identities and evidence.get("disease"):
        identities = [("Target phenotype", evidence["disease"])]
    statuses = evidence.get("coverage_by_disease") or {}
    returned = {(row.get("disease") or {}).get("id") for row in evidence.get("records") or []}
    result = []
    for role, identity in identities:
        identifier = identity.get("id")
        state = statuses.get(identifier)
        if state in {"observed", "not_observed"}:
            status = "Query completed"
            finding = "association returned for this gene" if identifier in returned else "no association returned for this gene"
        else:
            status = {"failed": "Query failed", "partial": "Query incomplete", "not_requested": "Not queried"}.get(
                state, "Query status unavailable")
            finding = "retained association for this gene" if identifier in returned else "no retained association for this gene"
        result.append(f"{role}: {identity.get('name') or identifier} ({identifier}). {status}; {finding}.")
    return result
