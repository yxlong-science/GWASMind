"""Source-bound CS-lead admission with separate local overlap witnesses."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation

from ..configuration import parameter
from .models import parse_variant_id
from .retrieval import _unique_rows


def cs_lead_measurement(row):
    """Never substitute a member measurement for the source CS lead."""
    from .evidence import _float

    identifier = (row.get("variant") or {}).get("id")
    try:
        parse_variant_id(identifier)
    except ValueError:
        return None, "lead_identity_unavailable"
    mantissa, exponent = row.get("pValueMantissa"), row.get("pValueExponent")
    if (_float(mantissa) is None or _float(exponent) is None or exponent != int(exponent)
            or mantissa <= 0 or abs(exponent) > 100000):
        return None, "lead_p_unavailable"
    try:
        value = Decimal(str(mantissa)).scaleb(int(exponent))
    except InvalidOperation:
        return None, "lead_p_unavailable"
    if not value.is_finite() or not 0 < value <= 1:
        return None, "lead_p_unavailable"
    if value >= Decimal(str(parameter("v2g.qtl.lead_pvalue_max"))):
        return None, "lead_p_above_threshold"
    measurement = {"variant_id": identifier, "measurement_scope": "credible_set_lead",
                   "pValueMantissa": mantissa, "pValueExponent": exponent}
    if _float(row.get("beta")) is not None:
        measurement["beta"] = row["beta"]
    return measurement, None


def build_qtl_category(target, locus, status):
    from .evidence import _anchor, _float, _rounded, _with_records, is_strong_anchor

    if target is None:
        return {"status": status, "credible_sets": 0, "overlapping": 0, "records": [], "omitted": 0}
    by_query = locus.by_query_id()
    types = set(parameter("v2g.qtl.study_types"))
    cis_only = parameter("v2g.qtl.cis_only")
    records, excluded = [], Counter()
    total = overlapping_before_lead_gate = 0
    for row in _unique_rows((target.get("credibleSets") or {}).get("rows") or []):
        if row.get("studyType") not in types:
            excluded["unsupported_study_type"] += 1
            continue
        if cis_only and row.get("isTransQtl") is not False:
            excluded["trans_qtl" if row.get("isTransQtl") is True else "cis_status_unavailable"] += 1
            continue
        total += 1
        shared = []
        for member in (row.get("locus") or {}).get("rows") or []:
            query_id = (member.get("variant") or {}).get("id")
            variant = by_query.get(query_id)
            if variant is not None:
                probability = _float(member.get("posteriorProbability"))
                probability = probability if probability is not None and 0 <= probability <= 1 else None
                shared.append((variant, probability, member, query_id))
        if not shared:
            excluded["no_local_member_overlap"] += 1
            continue
        overlapping_before_lead_gate += 1
        lead, reason = cs_lead_measurement(row)
        if lead is None:
            excluded[reason] += 1
            continue
        # The measured lead can lie outside S. Local witnesses establish this
        # signal's relationship to S; they never acquire the lead's P or beta.
        ordered = sorted(shared, key=lambda item: (-(item[1] if item[1] is not None else -1),
                                                   item[0].variant_id, item[3]))
        representative, _, member, _ = ordered[0]
        usable = [(v, p) for v, p, _, _ in ordered if p is not None]
        strong = [p for v, p in usable if is_strong_anchor(v, locus.weight_basis)]
        study = row.get("study") or {}
        biosample = study.get("biosample") or {}
        witnesses = [{"variant_id": v.variant_id, "query_variant_id": query_id,
                      "variant_sources": list(v.sources)} for v, _, _, query_id in ordered]
        records.append({
            "qtl_type": row["studyType"], "study_id": study.get("id"), "project": study.get("projectId"),
            "study_locus_id": row.get("studyLocusId"), "biosample": biosample.get("biosampleName"),
            "biosample_id": biosample.get("biosampleId"), "trans": row.get("isTransQtl"),
            "cs_lead_variant_id": lead["variant_id"], "cs_lead_measurement": lead,
            "finemapping_method": row.get("finemappingMethod"), "confidence": row.get("confidence"),
            "quality_controls": row.get("qualityControls") or [],
            "locus_anchor_witnesses": witnesses,
            "locus_anchor_roles": sorted({role for v, _, _, _ in ordered for role in v.sources}),
            "clpp": _rounded(sum(v.weight * p for v, p in usable), 5) if usable else None,
            "shared_variant_count": len(ordered),
            "max_shared_r2": max((v.r2_to_anchor or 0.0 for v, _, _, _ in ordered), default=None),
            "max_qtl_pip_at_strong_anchor": max(strong, default=None),
            "shared_variants": [{**_anchor(v), "qtl_pip": p} for v, p, _, _ in ordered],
            "strong_anchor_shared": bool(strong),
            "_source": {"row": row, "variant": representative, "member": member},
        })
    records.sort(key=lambda row: (str(row["study_id"]), str(row["study_locus_id"]), row["cs_lead_variant_id"]))
    return {"status": _with_records(status, records), "clpp_basis": locus.weight_basis,
            "credible_sets": total, "overlapping": len(records),
            "overlapping_before_lead_gate": overlapping_before_lead_gate,
            "trans_excluded": excluded["trans_qtl"], "excluded_by_reason": dict(excluded),
            "lead_pvalue_max_exclusive": parameter("v2g.qtl.lead_pvalue_max"),
            "records": records, "omitted": 0}
