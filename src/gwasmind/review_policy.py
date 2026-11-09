"""Read-only, task-scoped scientific review of already acquired evidence.

This module never retrieves evidence, admits candidates, ranks genes, or uses a
model's later answer to decide which limitations to disclose.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .biological_context import context_term_matches, evidence_context_values
from .context_review import prepare_context_review, review_record_context
from .region_gate import local_gene_window_label
from .source_gwas import summarize_source_gwas_observations

REVIEW_POLICY_VERSION = "2026-10-10.1"
_TASK_KINDS = {
    "target_assessment": {"safety", "tractability", "interaction", "phenotype", "function"},
    "disease_associations": {"target_disease", "ontology", "gwas", "study"},
    "drug_repurposing": {"drug", "target_disease", "safety", "tractability"},
    "gene_associations": {"qtl", "colocalization", "regulatory", "target_disease", "function", "gwas"},
}
_MOLECULAR_KINDS = {"qtl", "colocalization", "regulatory"}
_CORE_LOCUS_MODULES = frozenset({"locus.qtl", "locus.colocalisation", "locus.enhancer_to_gene",
                                 "locus.variant_consequence"})
_REQUIREMENTS = {"require_gwas_significance", "require_independent_support",
                 "require_original_publication", "require_exhaustive_literature", "require_tissue_mechanism",
                 "require_direct_evidence"}


def review_requirements(question: str = "", explicit: Mapping[str, Any] | None = None) -> dict[str, bool]:
    """Record explicit question requirements before inference, without rerouting it."""
    text = str(question or "")[:12000]
    patterns = {
        "require_gwas_significance": r"(?:\u662f\u5426|\u6709\u6ca1\u6709|\u5224\u65ad).{0,15}(?:\u663e\u8457|\u663e\u8457\u6027)|(?:is|are|test|assess|check).{0,35}(?:genome.wide significant|significance in (?:my|this|the uploaded))",
        "require_independent_support": r"\u72ec\u7acb(?:\u961f\u5217|\u91cd\u590d|\u590d\u73b0|\u9a8c\u8bc1)|independent(?:ly)? (?:replicat|validat|cohort)",
        "require_original_publication": r"\u539f\u59cb(?:\u8bba\u6587|\u6587\u732e|\u7814\u7a76).{0,15}(?:\u65b9\u6cd5|\u5b9e\u9a8c|\u6838\u67e5|\u6838\u5bf9)|(?:check|verify|audit).{0,30}(?:original (?:paper|publication|study))",
        # Prior GWAS association screening has a variant/study/trait scope; it
        # does not request every publication mentioning each candidate gene.
        "require_exhaustive_literature": r"(?:\u7a77\u5c3d|\u5168\u9762|\u5b8c\u6574|\u6240\u6709|\u5168\u90e8).{0,8}(?:\u6587\u732e|\u8bba\u6587)|(?:\u6587\u732e|\u8bba\u6587).{0,8}(?:\u7a77\u5c3d|\u5168\u90e8)|\b(?:exhaustive|comprehensive|complete|all)\s+(?:(?:published|prior|available)\s+)?(?:literature|publications|papers)\b|\b(?:literature|publication)\s+(?:search|review)\s+(?:must be\s+)?(?:exhaustive|complete)\b",
        "require_tissue_mechanism": r"\u7ec4\u7ec7\u7279\u5f02.{0,8}\u673a\u5236|\u7ec6\u80de\u7279\u5f02.{0,8}\u673a\u5236|(?:tissue|cell).specific mechanism",
        # A routine gene-prioritization request may be answered with contextual,
        # coding, or other qualified gene evidence.  Require a direct
        # variant-to-gene mechanism only when the user asks for that claim.
        "require_direct_evidence": r"(?:direct|measured|mechanistic)\s+(?:variant[- ]to[- ]gene|V2G)|\bV2G\b|\u76f4\u63a5.{0,8}\u53d8\u5f02.{0,8}\u57fa\u56e0|\u5fc5\u987b.{0,8}\u8bc1\u660e.{0,8}\u53d8\u5f02.{0,8}\u57fa\u56e0",
    }
    result = {}
    for name, pattern in patterns.items():
        matches = [m for m in re.finditer(pattern, text, re.I)
                   if not re.search(r"(?:(?:\u4e0d\u9700\u8981|\u4e0d\u8981\u6c42|\u65e0\u9700|\u4e0d\u7528)|(?:no|not|do\s+not|don't|without)(?:\s+\w+){0,4})\s*$", text[max(0, m.start()-60):m.start()], re.I)]
        result[name] = bool(matches)
    for key, value in (explicit or {}).items():
        if key in _REQUIREMENTS and isinstance(value, bool):
            result[key] = value
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _ids(values) -> list[str]:
    return sorted({str(v).strip() for v in values if isinstance(v, str) and v.strip()})


def _binding(values) -> dict[str, Any]:
    ids = _ids(values)
    return {"count": len(ids), "sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "examples": ids[:12]}


def _check(check_id, outcome, fact, reason, impact="", *, rows=(), variants=(), genes=(), studies=(), modules=()):
    affected = {"variant_ids": _ids([*variants, *(r.get("variant_id") for r in rows)]),
                "gene_ids": _ids([*genes, *(r.get("gene_id") for r in rows)]),
                "study_ids": _ids([*studies, *(r.get("study_id") for r in rows)]),
                "modules": _ids(modules)}
    refs = [{"type": "evidence", "id": value} for value in _ids(r.get("evidence_id") for r in rows)[:24]]
    refs.extend({"type": "module", "id": value} for value in affected["modules"][:8])
    return {"check_id": check_id, "outcome": outcome, "fact": fact,
            "reason_codes": [reason] if isinstance(reason, str) else list(reason),
            "impact": impact, "affected": affected, "refs": refs[:32]}


_FIELD_ALIASES = {
    "trait": {"trait", "traitid", "traitfromsource", "diseaseid", "phenotype", "phenotypeid"},
    "tissue": {"tissue", "biosample", "biosampleid", "biosamplename", "celltype", "celltypeid"},
    "endpoint": {"endpoint", "molecularphenotypeid", "moleculartraitid", "transcriptid", "phenotypeid", "assayid"},
    "method": {"method", "comparisonmethod", "colocalisationmethod", "colocalizationmethod", "model", "modeltype", "statisticalmodel", "assay"},
    "scale": {"effectscale", "scale", "betaunit", "units", "unit", "measurementscale"},
    "population": {"population", "populationid", "ancestry", "ancestryid", "ancestrylabel", "studypopulation", "cohortpopulation"},
    "effect_allele": {"effectallele", "effectalleles", "testedallele", "testedalleles", "betaeffectallele"},
}
_BETA_ALIASES = {"beta", "effectsize", "effect", "estimate", "betahat", "betavalue"}
_VARIANT_ALIASES = {"variantid", "variant", "snp", "snpid", "rsid", "rsnumber"}


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _record_views(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    payload = _mapping(record.get("payload"))
    context = _mapping(payload.get("context"))
    measurements = _mapping(payload.get("measurements"))
    upstream = _mapping(payload.get("upstream"))
    raw = _mapping(upstream.get("raw"))
    return (record, payload, context, measurements, upstream, raw,
            _mapping(payload.get("statistics")), _mapping(raw.get("measurements")),
            _mapping(raw.get("association")), _mapping(raw.get("study")),
            _mapping(raw.get("population")), _mapping(payload.get("provenance")))


def _field_value(views: Sequence[Mapping[str, Any]], aliases: set[str]) -> str:
    normalized = {_normal_key(alias) for alias in aliases}
    for view in views:
        for key, value in view.items():
            if _normal_key(key) not in normalized or isinstance(value, bool):
                continue
            if isinstance(value, Mapping):
                for nested_key in ("id", "name", "label", "code"):
                    nested = value.get(nested_key)
                    if isinstance(nested, (str, int, float)) and str(nested).strip():
                        return str(nested).strip()
                continue
            if isinstance(value, (str, int, float)):
                text = str(value).strip()
                if text:
                    return text
    return ""


def _numeric_field(views: Sequence[Mapping[str, Any]], aliases: set[str]) -> float | None:
    normalized = {_normal_key(alias) for alias in aliases}
    for view in views:
        for key, value in view.items():
            if _normal_key(key) not in normalized or isinstance(value, bool):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(parsed):
                return parsed
    return None


def _normalize_variant(value: str) -> str:
    text = value.strip().casefold()
    if not text:
        return ""
    text = re.sub(r"^chr", "", text)
    match = re.fullmatch(r"([0-9]+|x|y|m|mt)[_:](\d+)[_:]([acgtn]+)[_:]([acgtn]+)", text)
    if match:
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}_{match.group(4)}"
    rs = re.fullmatch(r"rs\d+", text)
    return rs.group(0) if rs else text


def _variant_value(record: Mapping[str, Any], views: Sequence[Mapping[str, Any]]) -> str:
    # Prefer the canonical record identity.  Proxy labels and free-text claims
    # are never treated as the measured variant for a direction comparison.
    value = str(record.get("variant_id") or "").strip()
    if not value:
        value = _field_value(views, _VARIANT_ALIASES)
    return _normalize_variant(value)


def _field_values(views: Sequence[Mapping[str, Any]], aliases: set[str]) -> set[str]:
    """Keep conflicting declarations visible instead of selecting the first one."""
    normalized = {_normal_key(alias) for alias in aliases}
    return {value for view in views for key, item in view.items()
            if _normal_key(key) in normalized
            and (value := _field_value(({key: item},), normalized))}


def _bound_effect_values(record, views, reasons):
    """Bind effect fields to one measurement, including context-governed projections."""
    _, payload, context, measurements, upstream, raw = views[:6]
    lead_measurement = _mapping(raw.get("csLeadMeasurement"))
    measured_aliases = {"measured_variant_id"}
    lead_aliases = {"cs_lead_variant_id"}
    scopes = {_normal_key(value) for value in
              _field_values((*views, lead_measurement), {"measurement_scope"})}
    lead_scope = "crediblesetlead" in scopes or bool(lead_measurement)
    owners = {_normalize_variant(value) for value in _field_values(views, measured_aliases)}
    if lead_scope:
        owners.update(_normalize_variant(value) for value in _field_values(views, lead_aliases))
        owners.update(_normalize_variant(value) for value in
                      _field_values((lead_measurement,), measured_aliases | _VARIANT_ALIASES))
    if len(scopes) > 1 or lead_scope and scopes and scopes != {"crediblesetlead"}:
        reasons.add("ambiguous_measurement_scope")
    if len(owners) > 1:
        reasons.add("ambiguous_measurement_owner")
    variant = next(iter(owners)) if len(owners) == 1 else ""
    if not owners and not lead_scope:
        variant = _variant_value(record, views)
    if not variant:
        reasons.add("missing_measurement_owner" if not owners else "ambiguous_measurement_owner")

    def owner(view, inherited):
        declared = _field_values((view,), measured_aliases)
        if lead_scope:
            declared |= _field_values((view,), lead_aliases)
        # In a lead record, generic record/raw variant IDs can identify the
        # queried member. Explicit measured IDs take precedence over them.
        if not declared:
            declared = _field_values((view,), _VARIANT_ALIASES)
        normalized = {_normalize_variant(value) for value in declared}
        return next(iter(normalized)) if len(normalized) == 1 else "" if normalized else inherited

    payload_owner = owner(payload, variant)
    context_owner = owner(context, payload_owner)
    upstream_owner = owner(upstream, context_owner)
    raw_owner = owner(raw, upstream_owner)
    bound_views = [(record, owner(record, variant)), (payload, payload_owner),
                   (context, context_owner), (measurements, owner(measurements, context_owner)),
                   (upstream, upstream_owner), (raw, raw_owner)]
    for view, inherited in ((views[6], context_owner), (views[7], raw_owner),
                            (views[8], raw_owner), (views[9], raw_owner),
                            (views[10], raw_owner), (views[11], context_owner),
                            (lead_measurement, variant)):
        bound_views.append((view, owner(view, inherited)))
    betas, alleles = set(), set()
    for view, field_owner in bound_views:
        beta_values = _field_values((view,), _BETA_ALIASES)
        allele_values = _field_values((view,), _FIELD_ALIASES["effect_allele"])
        if (beta_values or allele_values) and (not variant or field_owner != variant):
            reasons.add("effect_measurement_owner_mismatch")
            continue
        for value in beta_values:
            parsed = _numeric_field(({"beta": value},), {"beta"})
            if parsed is None:
                reasons.add("invalid_beta")
            else:
                betas.add(parsed)
        alleles.update(value.upper() for value in allele_values)
    if len(betas) > 1:
        reasons.add("ambiguous_beta")
    if len(alleles) > 1:
        reasons.add("ambiguous_effect_allele")
    return (variant, next(iter(betas)) if len(betas) == 1 else None,
            next(iter(alleles)) if len(alleles) == 1 else "")


def _comparable_measurement(record: Mapping[str, Any], *, exclusion_reasons=None) -> dict[str, Any] | None:
    views = _record_views(record)
    # Most functional and colocalization records do not contain an effect
    # estimate. They need no measurement-owner resolution or alignment work.
    lead_measurement = _mapping(views[5].get("csLeadMeasurement"))
    if str(record.get("kind") or "").casefold() != "qtl" and not _field_values(
            (*views, lead_measurement), _BETA_ALIASES):
        if exclusion_reasons is not None:
            exclusion_reasons.add("missing_beta")
        return None
    reasons = set()
    variant, beta, effect_allele = _bound_effect_values(record, views, reasons)
    trait = _field_value(views, _FIELD_ALIASES["trait"]).casefold()
    tissue = _field_value(views, _FIELD_ALIASES["tissue"]).casefold()
    population = _field_value(views, _FIELD_ALIASES["population"]).casefold()
    endpoint = _field_value(views, _FIELD_ALIASES["endpoint"]).casefold()
    method = _field_value(views, _FIELD_ALIASES["method"]).casefold()
    scale = _field_value(views, _FIELD_ALIASES["scale"]).casefold()
    # Unknown endpoint, model, or scale cannot be treated as equality.  This
    # avoids converting unrelated provider assays into a false conflict.
    for name, available in (("effect_allele", effect_allele), ("beta", beta is not None),
                            ("trait", trait), ("endpoint", endpoint), ("tissue", tissue),
                            ("method", method), ("scale", scale), ("population", population)):
        if not available:
            reasons.add("missing_" + name)
    # An allele string such as "A/G" is not an aligned effect allele.  Keep
    # the raw provider value untouched, but exclude it from this comparison.
    if effect_allele and not re.fullmatch(r"[ACGT]+", effect_allele):
        reasons.add("invalid_effect_allele")
    variant_parts = variant.upper().split("_")
    if effect_allele and len(variant_parts) == 4 and effect_allele not in variant_parts[2:]:
        reasons.add("effect_allele_not_in_measured_variant")
    if reasons:
        if exclusion_reasons is not None:
            exclusion_reasons.update(reasons)
        return None
    return {"key": (str(record.get("gene_id") or ""), str(record.get("kind") or "").casefold(),
                     variant, effect_allele, trait, endpoint, tissue, method, scale, population),
            "beta": beta, "record": record}


def assess_comparable_conflicts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Detect only conflicts that can be compared from explicit metadata.

    Support/opposition flags remain eligible when they describe the same
    measured hypothesis.  Effect-direction conflicts additionally require a
    normalized exact variant, aligned effect allele, trait/endpoint, tissue,
    model, scale, and population.  Missing metadata, proxies, or allele flips
    are not conflicts; their raw provider records remain untouched.
    """
    support_groups = defaultdict(list)
    beta_groups = defaultdict(list)
    scoped_records: dict[str, Mapping[str, Any]] = {}
    direction_exclusions = Counter()
    comparable_groups = 0
    for record in records:
        if not isinstance(record, Mapping) or not record.get("gene_id"):
            continue
        payload = _mapping(record.get("payload"))
        context = _mapping(payload.get("context"))
        views = _record_views(record)
        trait = _field_value(views, _FIELD_ALIASES["trait"]).casefold()
        tissue = _field_value(views, _FIELD_ALIASES["tissue"]).casefold()
        kind, variant = str(record.get("kind") or "").casefold(), _variant_value(record, views)
        if context.get("comparison_eligible") is False:
            direction_exclusions["comparison_ineligible"] += 1
            continue
        exclusion_reasons = set()
        measurement = _comparable_measurement(record, exclusion_reasons=exclusion_reasons)
        direction_exclusions.update(exclusion_reasons)
        if not kind or not trait or kind in _MOLECULAR_KINDS and (not variant or not tissue):
            continue
        endpoint = _field_value(views, _FIELD_ALIASES["endpoint"]).casefold()
        method = _field_value(views, _FIELD_ALIASES["method"]).casefold()
        comparison_id = str(context.get("comparison_id") or "")
        support_key = (str(record["gene_id"]), kind, variant, trait, tissue,
                       comparison_id, endpoint, method)
        # A support flag is comparable only when the provider explicitly
        # supplies a measured hypothesis, endpoint, and method.  Otherwise it
        # is a broad interpretation label rather than a technical conflict.
        if isinstance(record.get("supports"), bool) and comparison_id and endpoint and method:
            support_groups[support_key].append(record)
            scoped_records[str(record.get("evidence_id") or id(record))] = record
        if measurement is not None:
            beta_groups[measurement["key"]].append(measurement)
            scoped_records[str(record.get("evidence_id") or id(record))] = record
    support_conflicts = [rows for rows in support_groups.values() if {r["supports"] for r in rows} == {True, False}]
    beta_conflicts = [rows for rows in beta_groups.values()
                      if len(rows) >= 2 and any((first["beta"] < 0 < second["beta"]
                                                or second["beta"] < 0 < first["beta"])
                                                for index, first in enumerate(rows)
                                                for second in rows[index + 1:])]
    conflict_rows: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for rows in [*support_conflicts, *[[item["record"] for item in group] for group in beta_conflicts]]:
        for row in rows:
            identifier = str(row.get("evidence_id") or id(row))
            if identifier not in seen_ids:
                seen_ids.add(identifier)
                conflict_rows.append(row)
    comparable_groups = sum(len(rows) >= 2 for rows in support_groups.values()) + sum(
        len(rows) >= 2 for rows in beta_groups.values())
    return {"count": len(support_conflicts) + len(beta_conflicts),
            "support_conflict_count": len(support_conflicts),
            "beta_conflict_count": len(beta_conflicts),
            "comparable_groups": comparable_groups,
            "explicit_comparable_records": len(scoped_records),
            "unknown_records": len(records) - len(scoped_records),
            "effect_direction_exclusion_counts": dict(sorted(direction_exclusions.items())),
            "conflicting_records": conflict_rows}


def _locus_link(record: Mapping[str, Any]) -> bool:
    """A molecular record, or a variant consequence not marked as positional-only.

    UTR, synonymous and splice-region annotations carry direct_locus_support=False:
    they place the variant inside the gene, which is position, not a locus-to-gene link.
    """
    kind = record.get("kind")
    if kind in _MOLECULAR_KINDS:
        return True
    if kind != "variant_annotation":
        return False
    payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
    derived = payload.get("derived") if isinstance(payload.get("derived"), Mapping) else {}
    scope = derived.get("candidate_scope")
    return not isinstance(scope, Mapping) or scope.get("direct_locus_support") is not False


def assess_scientific_evidence(
    *, records: Sequence[Mapping[str, Any]], intent: str, direct_evidence_ids: Sequence[str],
    gene_evidence_ids: Sequence[str], candidate_gene_ids: Sequence[str] | None,
    required_modules: Sequence[str], module_statuses: Mapping[str, str],
    module_warnings: Mapping[str, Sequence[str]] | None = None,
    candidate_audit: Mapping[str, Any] | None = None,
    source_observations: Sequence[Mapping[str, Any]] | None = None,
    source_data_available: bool = False, gwas_signal_present: bool | None = None,
    nearby_gene_count: int | None = None, biological_context_defined: bool = False,
    biological_context_coverage: bool | None = None, context_terms: Sequence[str] = (),
    biological_context: Mapping[str, Any] | None = None,
    requirements: Mapping[str, Any] | None = None, require_direct_evidence: bool = False,
    require_nearby_gene: bool = False, complex_region: str | bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .review_coverage import assess_retrieval_coverage
    from .review_provenance import review_dependencies, review_provenance

    intent = intent or "prioritize_genes"
    required = review_requirements(explicit=requirements)
    # Direct V2G evidence is an explicit task requirement, not the default
    # admission criterion for ordinary gene prioritization.  Contextual and
    # qualified functional/coding evidence can still support a provisional
    # ranking when the user did not ask for a measured direct link.
    require_direct_evidence = bool(require_direct_evidence or required.get("require_direct_evidence"))
    audit = dict(candidate_audit or {})
    candidate_set = set(candidate_gene_ids) if candidate_gene_ids is not None else None
    direct_set, gene_set = set(direct_evidence_ids), set(gene_evidence_ids)
    # Candidate selection is frozen upstream. An incomplete candidate remains in scope.
    scoped = [r for r in records if intent != "prioritize_genes" or candidate_set is None
              or not r.get("gene_id") or r.get("gene_id") in candidate_set]
    direct = [r for r in scoped if r.get("evidence_id") in direct_set]
    genes = [r for r in scoped if r.get("evidence_id") in gene_set]
    coverage = assess_retrieval_coverage(module_statuses=module_statuses, required_modules=required_modules,
        module_warnings=module_warnings, candidate_audit=audit,
        require_exhaustive_literature=required["require_exhaustive_literature"])
    coloc_coverage = coverage.get("colocalisation_coverage")
    if coloc_coverage is not None:
        audit["colocalisation_detail_complete"] = coloc_coverage["complete"]
    incomplete = [*coverage["incomplete_required"], *coverage["incomplete_optional"]]
    phenotype_scope = _mapping(_mapping(biological_context).get("phenotype_scope"))
    phenotype_required = intent in {"prioritize_genes", *_TASK_KINDS} and bool(
        str(phenotype_scope.get("phenotype") or _mapping(biological_context).get("phenotype") or "").strip())
    phenotype_unresolved = phenotype_required and phenotype_scope.get("status") == "target_unresolved"
    ld_incomplete = any(audit.get(key) is False for key in ("ld_expansion_complete", "ld_proxy_retrieval_complete"))
    checks = []
    if intent != "general":
        # A non-empty candidate window is enough to answer a prioritization
        # question; absent locus links are reported by direct_locus_gene_evidence.
        task_supported = (bool(genes) or bool(candidate_set)) if intent == "prioritize_genes" else (
            any(r.get("kind") in _TASK_KINDS[intent] for r in scoped) if intent in _TASK_KINDS
            else bool(scoped) or source_data_available)
        no_candidate = candidate_set == set() and intent == "prioritize_genes"
        task_supported = task_supported and not no_candidate
        checks.append(_check("task_evidence_requirements", "pass" if task_supported else "warn",
            "Evidence relevant to the requested task is available." if task_supported else
            ("No eligible candidate was retained in the declared discovery scope. Available variant and source observations remain reportable."
             if no_candidate else "Task-critical evidence was not available in the retained results. Existing verified observations remain reportable."),
            "task_evidence_available" if task_supported else "no_eligible_candidate" if no_candidate else "task_evidence_missing",
            "Report only the candidates and conclusions supported by the retained evidence; do not fill missing ranking positions.", rows=genes))
        if phenotype_unresolved:
            check = checks[-1]
            check["outcome"] = "warn"
            check["fact"] = ("" if task_supported else check["fact"] + " ") + (
                "The target phenotype identity could not be resolved. Phenotype-specific association and "
                "colocalization support remain unassessed; available locus-to-gene evidence is retained.")
            check["reason_codes"] = ([] if task_supported else check["reason_codes"]) + ["phenotype_identity_unresolved"]
            check["impact"] += " Treat phenotype-specific interpretation as exploratory; do not substitute a related phenotype for the target."
    if require_direct_evidence:
        checks.append(_check("direct_locus_gene_evidence", "pass" if direct else "warn",
            f"{len(direct)} qualifying direct variant-to-gene evidence record(s) support the in-scope candidates." if direct else
            ("Direct variant-to-gene evidence coverage could not be established because relevant retrieval remains incomplete. No qualifying direct link is available in the retained results; this is not evidence of absence."
             if incomplete or coverage["candidate_gap_reasons"] or coverage["deferred_candidate_causes"] else
             "The completed declared search retained no qualifying direct variant-to-gene link. Contextual or prediction-based candidates remain exploratory."),
            "direct_link_available" if direct else "direct_link_not_established",
            "Expression and L2G recall do not establish a measured variant-to-gene mechanism; complementary molecular, coding, and regulatory routes remain valid.",
            rows=direct, genes=candidate_set or ()))
    elif intent == "prioritize_genes" and candidate_set and not direct and not any(
            r.get("gene_id") in candidate_set and _locus_link(r) for r in scoped):
        # No candidate has any locus link, so the ranking itself rests only on
        # position and background biology: this changes the main conclusion.
        checks.append(_check("direct_locus_gene_evidence", "warn",
            "No candidate gene has a retained locus-to-gene link (coding consequence, molecular QTL, "
            "colocalization or enhancer-to-gene prediction). The ranking relies on position and background "
            "biology and is exploratory."
            + (" Relevant retrieval remains incomplete; this is not evidence of absence." if incomplete else ""),
            "direct_link_not_established",
            "Present the ranking as exploratory; position and background evidence do not establish a locus mechanism.",
            genes=candidate_set))
    elif intent in {"prioritize_genes", "gene_associations"}:
        # Keep the check in the authoritative receipt so consumers have one
        # stable record, but do not turn an unrequested direct-link absence
        # into W1 or a user-facing limitation.
        checks.append(_check("direct_locus_gene_evidence", "pass",
            f"{len(direct)} qualifying direct variant-to-gene evidence record(s) were retained; a direct link was not required for this task." if direct else
            "A direct variant-to-gene link was not required for this task; qualified contextual, functional, or coding evidence remains eligible for ranking."
            + (" Relevant retrieval remains incomplete; this is not evidence of absence." if incomplete else ""),
            "direct_link_available" if direct else "direct_link_not_established",
            "Direct-link evidence is reported when available; its absence does not invalidate an otherwise supported contextual ranking.",
            rows=direct, genes=candidate_set or ()))
    if ld_incomplete:
        for check in checks:
            if check["check_id"] in {"task_evidence_requirements", "direct_locus_gene_evidence"} and check["outcome"] == "warn":
                check["fact"] += (" LD expansion was incomplete, so additional linked variants and "
                                  "gene links may remain unexamined. This is not evidence that such links are absent.")
    if require_nearby_gene:
        known_count = type(nearby_gene_count) is int and nearby_gene_count >= 0
        checks.append(_check("locus_genes_1mb", "pass" if known_count and nearby_gene_count else "note" if known_count else "not_assessed",
            f"{nearby_gene_count} annotated gene(s) were found within the selected locus expanded by {local_gene_window_label()} on each side." if known_count and nearby_gene_count else
            f"No annotated gene was found within the selected locus expanded by {local_gene_window_label()} on each side. This does not invalidate a separately verified variant-to-gene link." if known_count else
            f"Annotated-gene coverage within the selected locus expanded by {local_gene_window_label()} on each side could not be verified. This is unassessed coverage, not a confirmed gene desert; independently verified gene links remain usable.",
            "local_gene_context" if known_count and nearby_gene_count else "local_gene_view_empty" if known_count else "local_gene_census_unassessed"))

    required_fact = ("Declared required evidence retrieval remains incomplete for: " + ", ".join(coverage["incomplete_required"]) + "."
        if coverage["incomplete_required"] else
        "Required retrieval finished within its declared scope, including any successful zero-result queries.")
    if coloc_coverage is not None and (coloc_coverage["complete"] or "locus.colocalisation" in coverage["incomplete_required"]):
        required_fact += " " + coloc_coverage["fact"]
    # A required channel that returned data but stopped early narrows the
    # scope; only a channel that produced nothing can change the conclusion.
    effective_statuses = dict(module_statuses)
    if coloc_coverage is not None:
        effective_statuses["locus.colocalisation"] = coloc_coverage["module_status"]
    unusable_required = [module for module in coverage["incomplete_required"]
                         if str(effective_statuses.get(module, "missing")).casefold() not in {"partial", "truncated"}]
    checks.append(_check("required_modules",
        "warn" if unusable_required else "note" if coverage["incomplete_required"] else "pass",
        required_fact,
        "required_retrieval_incomplete" if coverage["incomplete_required"] else "declared_required_scope_complete",
        "Missing results from unfinished queries cannot establish absence of an association.", modules=coverage["incomplete_required"]))
    if coverage["incomplete_optional"]:
        literature_requested = required["require_exhaustive_literature"] and bool(
            set(coverage["incomplete_optional"]) & {"evidence.europepmc", "target.literature", "publication.metadata"})
        # A core locus-to-gene channel that produced nothing leaves the candidate
        # comparison incomplete even when another channel supplied a link.
        core_channel_lost = any(
            module in _CORE_LOCUS_MODULES
            and str(effective_statuses.get(module, "missing")).casefold() not in {"partial", "truncated"}
            for module in coverage["incomplete_optional"])
        checks.append(_check("optional_evidence_coverage",
            "warn" if literature_requested or core_channel_lost else "note",
            "Requested evidence remains incomplete for: " + ", ".join(coverage["incomplete_optional"]) + "."
            + (" " + coloc_coverage["fact"] if coloc_coverage is not None
               and "locus.colocalisation" in coverage["incomplete_optional"] else ""),
            "requested_evidence_scope_incomplete", "Limit comparisons to the retained scope; an unfinished lookup is not a negative finding.",
            modules=coverage["incomplete_optional"]))
    if coverage["bounded_literature_modules"] and not coverage["incomplete_optional"]:
        checks.append(_check("optional_evidence_coverage", "note",
            "Literature retrieval used the declared per-gene limit. This is a bounded evidence view, not an exhaustive search for all prior publications.",
            "declared_literature_limit", "Do not infer that a gene or locus has never been reported from an absent entry in this bounded view.",
            modules=coverage["bounded_literature_modules"]))
    if coverage["candidate_gap_reasons"] or coverage["unattributed_candidate_gap"]:
        reasons = coverage["candidate_gap_reasons"] or ["candidate_scope_unresolved"]
        checks.append(_check("candidate_discovery_coverage", "warn",
            "Candidate comparison has an unresolved acquisition scope: " + ", ".join(reasons) + ".",
            reasons, "Additional candidate or molecular-detail evidence may be absent; retain the supported candidates.", genes=candidate_set or ()))

    signal_relevant = intent in {"prioritize_genes", "describe_variants", "gene_associations"} or required["require_gwas_significance"]
    source_summary = None
    if signal_relevant:
        if source_observations is not None:
            source_summary = summarize_source_gwas_observations(source_observations, phenotype="")
            measured = source_summary.measurements
            below = [m.variant_id for m in measured if m.genome_wide_significant is False]
            unknown = [m.variant_id for m in measured if m.genome_wide_significant is None]
            # Explicit source groups define loci. Distance or a shared query ID
            # cannot establish that selected variants are independent signals.
            groups = defaultdict(list)
            for observation, measurement in zip(source_observations, measured, strict=True):
                group = observation.get("locus_id") or observation.get("signal_id")
                if isinstance(group, str) and group.strip():
                    groups[group].append(measurement.genome_wide_significant)
            unsupported_groups = [key for key, values in groups.items() if values and all(value is False for value in values)]
            all_groups_supported = bool(groups) and sum(len(values) for values in groups.values()) == len(measured) and all(
                any(value is True for value in values) for values in groups.values())
            all_below = bool(measured) and len(below) == len(measured)
            warn = bool(unsupported_groups) or all_below or required["require_gwas_significance"] and bool(below or unknown or not measured)
            outcome = "warn" if warn else "not_assessed" if unknown or not measured or below and not all_groups_supported else "pass"
            fact = (f"Source GWAS P values were supplied for {source_summary.variants_with_p_value}/{len(measured)} selected variants; "
                    f"{source_summary.genome_wide_significant_variants} meet the configured threshold ({source_summary.significance_threshold:g}). "
                    f"{len(below)} measured variants do not meet it; {len(unknown)} remain unassessed. "
                    + (f"{len(unsupported_groups)} explicitly identified locus/signal group(s) have no significant supplied variant. " if unsupported_groups else "")
                    + ("Selected-variant significance does not establish significance for every locus; a locus partition was not supplied for all variants."
                       if below and len(groups) == 0 and not all_below else ""))
            variants = [*below, *unknown]
        else:
            outcome = "warn" if gwas_signal_present is False or gwas_signal_present is None and required["require_gwas_significance"] else "not_assessed" if gwas_signal_present is None else "pass"
            fact = ("None of the selected variants reaches the configured GWAS significance threshold in the supplied source observations." if gwas_signal_present is False else
                    "The supplied source summary reports a significant variant; per-variant source measurements were not provided to this review." if gwas_signal_present else
                    "Source GWAS P values were not supplied to this review. Functional interpretation is conditional on the nominated variants; source significance is unassessed.")
            variants = []
        checks.append(_check("current_gwas_signal", outcome, fact,
            "source_significance_limited" if outcome == "warn" else "source_significance_unassessed" if outcome == "not_assessed" else "source_significance_observed",
            "External study P values must not be relabelled as significance in the uploaded GWAS.", variants=variants))

    molecular_scope = direct if require_direct_evidence or intent in {"prioritize_genes", "gene_associations"} else scoped
    molecular = [r for r in molecular_scope if r.get("kind") in _MOLECULAR_KINDS]
    # These annotation IDs already passed the caller's qualified coding/splice
    # check. Background annotations and general gene descriptions cannot exempt a gene.
    coding_genes = {r.get("gene_id") for r in direct if r.get("kind") == "variant_annotation"}
    coding_exempt = coding_genes if not required["require_tissue_mechanism"] else set()
    required_molecular = [r for r in molecular if r.get("gene_id") not in coding_exempt]
    coding_only = bool(coding_genes) and not required_molecular
    empty_candidates = intent == "prioritize_genes" and candidate_set == set()
    context_needed = required["require_tissue_mechanism"] or empty_candidates or (intent in {"prioritize_genes", "gene_associations"} and not coding_only and bool(scoped))
    context_plan = prepare_context_review(biological_context)
    context_explicit = bool(context_terms) or context_plan is not None
    context_profile = {"defined": biological_context_defined, "terms": _ids(context_terms)}
    if context_plan is not None:
        context_profile["mapping_identity"] = context_plan["identity"]
    if context_needed:
        matching, measured, missing_context = [], [], []
        for row in molecular:
            if context_plan is not None:
                assessment = review_record_context(row, context_plan)
                classification = assessment["context_class"]
                if classification in {"matched", "other_reported_context"}:
                    measured.append(row)
                if classification == "matched":
                    matching.append(row)
                elif classification == "context_unreported" and row.get("gene_id") not in coding_exempt:
                    missing_context.append(row)
                continue
            observed = evidence_context_values(row)
            if observed:
                measured.append(row)
                if any(context_term_matches(term, value) for term in context_terms for value in observed):
                    matching.append(row)
            elif row.get("gene_id") not in coding_exempt:
                missing_context.append(row)
        matched_genes = _ids(r.get("gene_id") for r in matching)
        measured_genes = {r.get("gene_id") for r in measured}
        unresolved_genes = {r.get("gene_id") for r in missing_context}
        mismatched_genes = measured_genes - set(matched_genes) - coding_exempt if context_explicit else set()
        # An unresolved biosample might match the plan; do not claim that every
        # direct molecular record of this gene was measured elsewhere.
        mismatched_genes -= unresolved_genes
        mismatched = [r for r in measured if r.get("gene_id") in mismatched_genes]
        unknown_genes = unresolved_genes - set(matched_genes) - coding_exempt
        context_candidates = set(_ids(candidate_set if candidate_set is not None else
                                      (r.get("gene_id") for r in scoped)))
        unmeasured_genes = context_candidates - measured_genes - unresolved_genes - coding_exempt
        unassessed = [r for r in missing_context if r.get("gene_id") in unknown_genes]
        covered = bool(matching) if context_explicit else biological_context_coverage
        context_retrieval_unfinished = (
            any(name in incomplete for name in ("locus.qtl", "locus.colocalisation", "locus.enhancer_to_gene", "target.expression"))
            or any(audit.get(name) is False for name in ("molecular_context_complete",
                "colocalisation_detail_complete", "locus_discovery_complete", "evidence_integrity_complete")))
        all_candidates_mismatched = bool(context_candidates) and context_candidates <= mismatched_genes
        context_profile.update({"matched_gene_ids": matched_genes, "matching_evidence": _binding(r.get("evidence_id") for r in matching),
            "mismatched_gene_ids": _ids(mismatched_genes), "mismatching_evidence": _binding(r.get("evidence_id") for r in mismatched),
            "unassessed_gene_ids": _ids(unknown_genes), "missing_context_evidence": _binding(r.get("evidence_id") for r in missing_context),
            "coding_exempt_gene_ids": _ids(coding_exempt), "candidate_gene_ids": _ids(context_candidates),
            "genes_without_direct_context_measurements": _ids(unmeasured_genes),
            "all_candidates_measured_without_match": all_candidates_mismatched,
            "retrieval_unfinished": context_retrieval_unfinished})
        affected_rows = matching if covered else required_molecular
        if empty_candidates:
            state, reason, fact = "not_assessed", "no_candidate_context_to_assess", "No eligible candidate was retained, so candidate-specific tissue or cell matching could not be assessed. The missing candidate evidence is reported under W1."
            affected_rows = []
        elif context_plan is not None and context_plan["frozen"] is None:
            state, reason, fact = "not_assessed", "context_mapping_unverified", "The frozen biological-context mapping could not be verified. Biosample concordance remains unassessed; this does not establish a tissue mismatch."
        elif not biological_context_defined:
            state, reason, fact = "not_assessed", "context_undefined", "No explicit phenotype-relevant tissue or cell context was established, so candidate-specific context coverage remains unassessed. This is not a database finding that relevant tissue evidence is absent."
        elif not molecular and not required["require_tissue_mechanism"]:
            state, reason, fact = "not_assessed", "context_match_unassessed", "No qualifying molecular measurement was retained for candidate-specific context matching. Background-only candidates remain exploratory; missing molecular evidence is not a measured tissue mismatch."
        elif covered and mismatched_genes:
            state, reason, fact = "note", "partial_context_coverage", f"Context-matched direct molecular or regulatory evidence is available for {len(matched_genes)} candidate gene(s). Evidence for {len(mismatched_genes)} other candidate gene(s) was measured only in other reported contexts within the retained results. These candidate-specific differences do not trigger an overall biological-context warning or change the candidate order."
            if context_retrieval_unfinished:
                fact += " Retrieval remains incomplete, so missing matching evidence for individual candidates does not establish absence."
            affected_rows = mismatched
        elif context_explicit and covered and unknown_genes:
            state, reason, fact = "note", "context_metadata_incomplete", f"Matching molecular evidence remains usable for {len(matched_genes)} gene(s). Tissue or cell identities are missing or unresolved for the retained molecular evidence of {len(unknown_genes)} other gene(s), whose context match remains unassessed."
            affected_rows = unassessed
        elif covered:
            state, reason, fact = "pass", "measured_context_match", "Qualifying measured evidence covers at least one declared phenotype-relevant context; coverage is not asserted for every candidate or tissue."
        elif coding_exempt:
            state, reason, fact = "note", "coding_route_context_optional", "A qualified coding or splice consequence supplies a direct candidate mechanism without requiring a tissue measurement. Tissue-specific evidence for other candidates is recorded separately and does not trigger an overall biological-context warning."
            affected_rows = mismatched or unassessed
        elif context_retrieval_unfinished:
            state, reason, fact = "not_assessed", "context_retrieval_unfinished", "Matching measured context could not be assessed because relevant evidence retrieval remains incomplete."
        elif not context_explicit or covered is None or not all_candidates_mismatched:
            state, reason, fact = "not_assessed", "context_match_unassessed", "The supplied context or retained direct measurements do not establish whether every candidate lacks a context match. Unresolved biosamples and candidates without qualifying direct measurements are unassessed, not confirmed mismatches."
            affected_rows = unassessed or required_molecular
        else:
            # Tissue mismatch limits mechanism claims, not the gene ranking,
            # unless the user explicitly asked for a tissue-specific mechanism.
            state = "warn" if required["require_tissue_mechanism"] else "note"
            reason, fact = "no_measured_context_match", "No candidate has a context-matched direct molecular or regulatory measurement within the completed declared search. All assessed candidates have qualifying measurements only in other reported contexts; this does not establish biological absence outside that search."
            affected_rows = mismatched if context_explicit else required_molecular
        checks.append(_check("biological_context_coverage", state, fact, reason,
            "Limit tissue-specific mechanism claims to the contexts actually measured. Request parameters are not measured tissue evidence.",
            rows=affected_rows))

    dependence_records = direct if require_direct_evidence else scoped
    dependency = review_dependencies(dependence_records, include_groups=False)
    # Independence is meaningful only for a qualifying evidence scope.  Do not
    # manufacture an independence check from auxiliary/background records when
    # no direct evidence was retained and the user did not request replication.
    if (intent not in {"general", "describe_variants"} and dependence_records
            and (require_direct_evidence or direct or required["require_independent_support"])):
        established = dependency.get("independence_state") == "established"
        state = "pass" if established else "warn" if required["require_independent_support"] else "not_assessed"
        checks.append(_check("evidence_independence", state,
            "Explicit recorded cohort relationships establish the requested independent support." if established else
            "Original study and derivation relationships were grouped; independent cohort replication is not established by study IDs, publications, evidence rows, or LD proxies alone.",
            "independence_established" if established else "independence_not_established",
            "Complementary evidence may inform different mechanistic links without constituting independent replication. Shared-source records must not be counted as independent votes.", rows=dependence_records))

    conflicts = assess_comparable_conflicts(dependence_records)
    if intent != "general":
        state = "warn" if conflicts["count"] else "pass" if conflicts["comparable_groups"] and not conflicts["unknown_records"] else "not_assessed"
        conflict_fact = (
            f'{conflicts["count"]} comparable evidence conflict(s) require interpretation '
            f'({conflicts["support_conflict_count"]} explicit support/opposition and '
            f'{conflicts["beta_conflict_count"]} aligned effect-direction difference(s)).'
            " Opposite point estimates do not by themselves establish statistical heterogeneity."
            if conflicts["count"] else
            "No explicit opposition or aligned effect-direction conflict was detected within the recorded comparable hypotheses."
            if state == "pass" else
            "Conflict assessment is limited by missing explicit opposition or comparable measurement information. Evidence consistency has not been established."
        )
        checks.append(_check("evidence_conflicts", state,
            conflict_fact,
            "comparable_conflict" if conflicts["count"] else "no_explicit_comparable_conflict" if state == "pass" else "conflicts_not_assessed",
            "Effect-direction differences are flagged only after exact-variant, effect-allele, trait, endpoint, tissue, model, scale, and population alignment. Allele flips, proxies, missing metadata, and cross-context heterogeneity are not treated as conflicts.",
            rows=conflicts["conflicting_records"]))
    checks.append(_check("complex_region", "warn" if complex_region else "pass",
        f"An analyzed variant overlaps {complex_region if isinstance(complex_region, str) else 'a configured complex genomic region'}; regional LD and database coverage limit interpretation." if complex_region else
        "No analyzed variant was flagged by the configured complex-region definition; this is not an assessment of every possible genomic complexity.",
        "configured_complex_region_overlap" if complex_region else "no_configured_region_overlap",
        "Absence of records from a region excluded by a database is not absence of genetic association."))

    provenance = review_provenance(scoped, require_publication=required["require_original_publication"],
                                  include_records=False, include_groups=False)
    if intent != "general" and scoped:
        untraceable = provenance.get("untraceable_study_count", 0)
        untraceable_records = provenance.get("untraceable_record_count", 0)
        conflict = provenance.get("conflicted_record_count", 0)
        missing_papers = provenance.get("missing_publication_count", 0)
        missing_paper_records = provenance.get("missing_publication_record_count", 0)
        # Provenance bookkeeping does not change the main conclusion; it is a
        # warning only when the user asked for original-publication checks.
        gaps = bool(untraceable or untraceable_records or conflict or missing_papers or missing_paper_records)
        state = ("warn" if gaps and required["require_original_publication"] else "note" if gaps else "pass")
        fact = (f"{untraceable} identified source group(s) and {untraceable_records} ungrouped evidence record(s) cannot be traced; "
                f"{conflict} record(s) have conflicting source identifiers. "
                f"Original publications were not located for {missing_papers} source group(s) and {missing_paper_records} ungrouped record(s). "
                "Data-record traceability and original-paper availability were assessed separately.")
        traceability_gap = bool(untraceable or untraceable_records or conflict)
        affected_ids = set(provenance.get("affected_evidence_ids", ()))
        if state == "note" and not traceability_gap:
            affected_ids.update(provenance.get("missing_publication_evidence_ids", ()))
        # A pass over zero study records checked nothing, so it must not read as a verified source.
        pass_reason = "original_source_verified" if provenance.get("study_record_count") else "no_study_records_to_trace"
        provenance_check = _check("literature_coverage", state, fact,
            provenance.get("reason_codes") or [pass_reason if state == "pass" else "original_publication_unavailable"],
            "A database record can support its recorded facts without establishing that the original paper was checked. Each colocalization study side requires its own provenance.",
            rows=[r for r in scoped if r.get("evidence_id") in affected_ids])
        # The affected source may be the right side of a colocalization pair;
        # its record's top-level study_id commonly identifies only the left side.
        provenance_check["affected"]["study_ids"] = _ids(provenance.get(
            "missing_publication_study_ids" if state == "note" and not traceability_gap else "affected_study_ids", ()))
        checks.append(provenance_check)

    scope = {"policy_version": REVIEW_POLICY_VERSION, "intent": intent, "requirements": required,
        "candidate_scope": ("retained_task_evidence" if candidate_set is None else
                            "complete_gene_window" if audit.get("mode") == "gene_centric_window" else "upstream_shortlist"),
        "candidate_binding": _binding(candidate_set or ()), "evidence_binding": _binding(r.get("evidence_id") for r in scoped),
        "direct_evidence_binding": _binding(r.get("evidence_id") for r in direct),
        "context": context_profile, "module_statuses": dict(sorted(module_statuses.items())),
        "source_significance": {"source_observation_count": len(source_observations) if source_observations is not None else None,
            "source_p_count": source_summary.variants_with_p_value if source_summary else None,
            "source_measurements_sha256": hashlib.sha256(json.dumps(sorted((
                str(row.get("variant_id") or ""), str(row.get("p_value")),
                str(row.get("locus_id") or row.get("signal_id") or "")) for row in source_observations or ()),
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()},
        "dependency_assessment": {"independence_state": dependency.get("independence_state"),
            "independent_cohort_count": dependency.get("independent_cohort_count"),
            "unidentified_source_binding": _binding(dependency.get("unidentified_records", ()))},
        "conflict_assessment": {key: value for key, value in conflicts.items() if key != "conflicting_records"},
        "provenance_assessment": {key: value for key, value in provenance.items() if key.endswith("count")},
        "candidate_gap_reasons": coverage["candidate_gap_reasons"], "deferred_candidate_causes": coverage["deferred_candidate_causes"],
        "colocalisation_coverage": coloc_coverage,
        "bounded_literature_modules": coverage["bounded_literature_modules"], "model_output_used": False}
    if phenotype_scope:
        scope["phenotype_identity"] = {"status": phenotype_scope.get("status"),
            "scope_sha256": phenotype_scope.get("scope_sha256"), "required_for_task": phenotype_required,
            "issues": list(phenotype_scope.get("issues") or ())}
    scope["sha256"] = hashlib.sha256(json.dumps(scope, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return checks, scope
