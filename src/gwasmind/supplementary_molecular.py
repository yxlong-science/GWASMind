"""Select bounded non-context molecular observations after candidate freeze."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy

from .audit import sha256_json
from .qtl import qtl_study_type_from_semantic_payload


def _map(value):
    return value if isinstance(value, Mapping) else {}


def _finite_number(value):
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def molecular_context_class(match):
    """Preserve explicit unknown context instead of relabelling it as unrelated."""
    value = match.get("context_class")
    if value in {"matched", "other_reported_context", "context_unreported"}:
        return value
    return {"matched": "matched", "not_matched": "other_reported_context"}.get(
        match.get("status"), "context_unreported")


def frozen_context_assessment(match, context_sha256):
    """Expose the computed assessment; downstream gates must verify its plan."""
    return {"context_sha256": context_sha256, "context_class": molecular_context_class(match),
            "mapping_status": match.get("mapping_status", "legacy_label_only"),
            "observed_biosamples": deepcopy(match.get("observed_biosamples", [])),
            "match_basis": deepcopy(match.get("match_basis", [])),
            "source": "frozen_molecular_discovery_context"}


def _biosample(match):
    """Use one resolved biological sample identity, not a study or assay ID."""
    observed = match.get("observed_biosamples")
    if observed is not None and match.get("mapping_status") != "legacy_label_only":
        identities = {}
        for row in observed:
            if not isinstance(row, Mapping):
                continue
            identifier = row.get("id") or row.get("biosample_id") or row.get("biosampleId")
            label = row.get("label") or row.get("biosample_name") or row.get("biosampleName")
            if isinstance(identifier, str) and identifier.strip():
                identities[identifier.strip()] = str(label or identifier).strip()
        if len(identities) != 1:
            return None
        identifier, label = next(iter(identities.items()))
        return {"id": identifier, "label": label, "identity_basis": "resolved_biosample_id"}
    # Compatibility for explicit older contexts that have no ontology mapping.
    # A modern unresolved/ambiguous mapping is never authorized by this branch.
    labels = {" ".join(str(value).split()) for value in match.get("observed", ()) if str(value).strip()}
    if len(labels) != 1:
        return None
    label = next(iter(labels))
    return {"id": "label:" + label.casefold(), "label": label, "identity_basis": "reported_label_legacy_context"}


def actual_member_p(record, variant_id):
    """Read only the exact analyzed member's P; never fall back to lead/GWAS P."""
    raw = _map(_map(record.get("upstream")).get("raw"))
    member = _map(raw.get("matchedLocusMember"))
    if member.get("variantId") != variant_id:
        return {"available": False, "reason": "exact_member_p_unavailable"}
    mantissa, exponent = member.get("pValueMantissa"), member.get("pValueExponent")
    if (_finite_number(mantissa) and _finite_number(exponent) and mantissa > 0
            and int(exponent) == exponent):
        logp = math.log10(mantissa) + exponent
        if logp <= 0:
            return {"available": True, "log10_p": logp, "mantissa": mantissa,
                    "exponent": exponent, "source": "matchedLocusMember.pValueMantissa/pValueExponent"}
    for key in ("pValue", "p_value"):
        value = member.get(key)
        if _finite_number(value) and 0 < value <= 1:
            return {"available": True, "log10_p": math.log10(value), "p_value": value,
                    "source": "matchedLocusMember." + key}
    return {"available": False, "reason": "exact_member_p_unavailable"}


def molecular_detail_anchor_verified(record, study_locus_id):
    """Accept live or frozen explicit QTL-member lineage for a detail record."""
    from .molecular_context import molecular_anchor_verified

    raw = _map(_map(record.get("upstream")).get("raw"))
    member = _map(raw.get("qtlMatchedLocusMember"))
    if not member:
        return molecular_anchor_verified(record)
    subject = _map(record.get("subject")).get("id")
    live_member = _map(raw.get("matchedLocusMember"))
    return (record.get("module") == "locus.colocalisation"
            and raw.get("qtlStudyLocusId") == study_locus_id
            and bool(subject) and member.get("variantId") == subject
            and (not live_member or live_member.get("variantId") == subject)
            and raw.get("queryVariantId", subject) == subject)


def select_supplementary_molecular(records_by_key, gene_rows, *, max_biosamples, context_sha256):
    """Keep one member-P representative per non-context biosample and frozen gene.

    This changes evidence representation only. Candidate admission and all already
    eligible context-matched molecular observations remain unchanged.
    """
    if type(max_biosamples) is not int or max_biosamples < 0:
        raise ValueError("Supplementary molecular biosample limit must be a non-negative integer")
    selected, excluded, missing_p, gene_summaries = [], Counter(), [], []
    eligible_count = 0
    for gene in sorted(gene_rows, key=lambda item: item["gene_id"]):
        if not gene.get("selected"):
            continue
        by_biosample = defaultdict(list)
        for source in gene.get("sources", ()):
            if source.get("module") != "locus.qtl":
                continue
            match = source.get("molecular_context") or {}
            context_class = molecular_context_class(match)
            if context_class == "matched":
                continue
            if context_class != "other_reported_context":
                excluded["context_unreported"] += 1
                continue
            if not source.get("molecular_supplementary_eligible"):
                excluded["local_anchor_or_identity_ineligible"] += 1
                continue
            row = records_by_key.get(source["record_key"])
            if row is None or qtl_study_type_from_semantic_payload(row) is None:
                excluded["unsupported_or_unreported_qtl_type"] += 1
                continue
            sample = _biosample(match)
            if sample is None:
                excluded["biosample_identity_unresolved"] += 1
                continue
            if source["locus_id"].startswith("variant:"):
                excluded["study_locus_identity_unavailable"] += 1
                continue
            measure = actual_member_p(row, source["variant_id"])
            if not measure["available"]:
                excluded["missing_member_p"] += 1
                missing_p.append({"gene_id": gene["gene_id"], "study_locus_id": source["locus_id"],
                                  "variant_id": source["variant_id"], "record_key": source["record_key"],
                                  "biosample": sample, "reason": measure["reason"]})
                continue
            candidate = {"gene_id": gene["gene_id"], "study_locus_id": source["locus_id"],
                         "variant_id": source["variant_id"], "record_key": source["record_key"],
                         "biosample": sample, "qtl_type": qtl_study_type_from_semantic_payload(row),
                         "context_class": context_class, "member_p": measure,
                         "context_assessment": frozen_context_assessment(match, context_sha256),
                         "selection_basis": "smallest_actual_member_p_within_biosample"}
            by_biosample[sample["id"]].append(candidate)
            eligible_count += 1

        def order(item):
            measure = item["member_p"]
            return (not measure["available"], measure.get("log10_p", 0), item["biosample"]["id"],
                    item["study_locus_id"], item["variant_id"], item["record_key"])

        representatives = [min(rows, key=order) for rows in by_biosample.values()]
        gene_selected = sorted(representatives, key=order)[:max_biosamples]
        selected.extend(gene_selected)
        gene_summaries.append({"gene_id": gene["gene_id"], "verified_member_p_biosamples": len(representatives),
            "selected_biosamples": len(gene_selected), "unused_capacity": max_biosamples - len(gene_selected),
            "unfilled_reason": "supplement_disabled" if max_biosamples == 0 else
            "fewer_than_limit_biosamples_with_verified_member_p" if len(gene_selected) < max_biosamples else None})
        excluded["same_biosample_nonrepresentative"] += sum(len(rows) - 1 for rows in by_biosample.values())
        excluded["biosample_limit"] += max(0, len(representatives) - max_biosamples)
    receipt = {"version": "supplementary-molecular-biosamples-v1", "context_sha256": context_sha256,
               "max_biosamples_per_gene": max_biosamples, "matched_molecular_records_limited": False,
               "candidate_ids": sorted(gene["gene_id"] for gene in gene_rows if gene.get("selected")),
               "eligible_noncontext_records": eligible_count, "selected": selected,
               "gene_summaries": gene_summaries,
               "excluded_missing_member_p": sorted(missing_p, key=lambda row: (row["gene_id"], row["record_key"])),
               "unfilled_slots_are_not_imputed": True,
               "excluded_counts": dict(sorted((key, count) for key, count in excluded.items() if count)),
               "selection_scope": "already_selected_gene_local_verified_analysis_member_only"}
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt
