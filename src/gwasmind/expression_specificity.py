"""HPA-style context enrichment. Z and Tau never independently admit a gene."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from statistics import pstdev
from typing import Any

from .biosample_catalog import biosample_catalog
from .configuration import parameter
from .expression_policy import (
    DEFAULT_DETECTION_RULES,
    DEFAULT_EXPRESSION_FOLD_THRESHOLD,
    EXPRESSION_METHOD,
    HPA_METHOD_URL,
    expression_context,
    matches_selector,
    resolved_sample_identity,
)

DEFAULT_EXPRESSION_Z_THRESHOLD = parameter("expression.supporting_z_threshold")  # Backward-compatible supporting annotation only.


def _map(value):
    return value if isinstance(value, Mapping) else {}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _text(value):
    return " ".join(value.split()).casefold() if isinstance(value, str) else ""


def _biosample(value):
    value = _map(value)
    name = _text(value.get("biosampleName") or value.get("label"))
    identifier = _text(value.get("biosampleId") or value.get("efo_code")) or name
    return identifier, name


def _valid_biosample(value):
    if value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    fields = ("biosampleId", "efo_code", "biosampleName", "label")
    return all(value.get(field) is None or isinstance(value[field], str) for field in fields)


def _ratio(numerator, denominator):
    # JSON has no infinity; zero background is recorded separately.
    if denominator <= 0:
        return None
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else None


def _mean(values):
    # Sum before division so exact fold boundaries (e.g. mean(7, 17) == 12)
    # are not shifted by per-value scaling. The rare overflowing sum uses an
    # exact rational fallback: its average still fits in the finite input range.
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        return float(sum(Fraction(value) for value in values) / len(values))


def _unique_median(values):
    values = sorted(set(values))
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else _mean(values[middle - 1 : middle + 1])


def _compatible(pattern, key):
    return all(expected is None or expected == actual for expected, actual in zip(pattern, key))


def _coverage_complete(raw, rows):
    if "expressionCoverage" not in raw:
        # Historical complete adapters did not attach native paging metadata.
        return True
    coverage = raw["expressionCoverage"]
    if not isinstance(coverage, Mapping) or coverage.get("complete") is not True:
        return False
    if not isinstance(rows, (list, tuple)):
        return False
    return all(
        isinstance(coverage[field], int)
        and not isinstance(coverage[field], bool)
        and coverage[field] == len(rows)
        for field in ("declared_count", "retrieved_count")
        if field in coverage
    )


def _axis_profile(groups, *, axis, selectors):
    """Return the explicit model-facing profile for one expression axis."""
    selected = [group for group in groups if group.get("comparison_axis") == axis]
    terms = sorted({
        term
        for selector in selectors
        if axis == "tissue" and "tissue" in selector
        or axis == "cell_type" and "cell_type" in selector
        for term in selector.get("tissue" if axis == "tissue" else "cell_type", ())
    })
    unresolved_statuses = {
        "incomplete_retrieval",
        "invalid_reference_metadata",
        "invalid_reference_values",
        "insufficient_reference_biosamples",
        "detection_rule_unavailable",
        "missing_or_ambiguous_group_members",
        "no_outside_reference",
    }
    assessed = sum(group.get("status") == "evaluated" for group in selected)
    unresolved = sum(
        group.get("status") in unresolved_statuses
        or any(test.get("status") in unresolved_statuses for test in group.get("group_enrichment", ()))
        for group in selected
    )
    if not selected:
        status = "no_observed_records"
    elif assessed and unresolved:
        status = "partially_evaluated"
    elif unresolved:
        status = "unresolved"
    elif any(
        group.get("status") == "evaluated"
        or any(test.get("status") == "evaluated" for test in group.get("group_enrichment", ()))
        for group in selected
    ):
        status = "evaluated"
    else:
        status = "observed_not_evaluable"
    return {
        "axis": axis,
        "context_terms": terms,
        "groups": selected,
        "group_count": len(selected),
        "evaluated_panel_count": assessed,
        "unresolved_panel_count": unresolved,
        "descriptive_panel_count": sum(group.get("descriptive_status") == "evaluated" for group in selected),
        "status": status,
        "profile_contract": "expression-profile-v1",
    }


def expression_specificity(
    records: Sequence[Mapping],
    *,
    relevant_terms: Sequence[str],
    threshold: float = DEFAULT_EXPRESSION_Z_THRESHOLD,
    fold_threshold: float = DEFAULT_EXPRESSION_FOLD_THRESHOLD,
    context_selectors=None,
    context_groups=None,
    detection_rules=None,
    context_mapping=None,
) -> dict[str, Any]:
    if _number(threshold) is None:
        raise ValueError("expression supporting Z threshold must be a finite nonnegative number")
    if _number(fold_threshold) is None or fold_threshold <= 1:
        raise ValueError("expression fold threshold must be finite and greater than one")
    context = expression_context(
        relevant_terms, selectors=context_selectors, context_groups=context_groups
    )
    def matches(selector, sample):
        return matches_selector(selector, sample, context_mapping=context_mapping)

    context["match_policy"] = (
        "frozen_ot_id_or_catalog_ancestor" if context_mapping is not None
        else "ot_id_or_catalog_ancestor_with_legacy_literal_fallback"
    )
    mapping_status = "legacy" if context_mapping is None else (
        "catalog_mismatch"
        if not isinstance(context_mapping, Mapping) or
        not biosample_catalog().recorded_with(context_mapping.get("catalog", {}).get("catalog_sha256"))
        else "resolved" if all(item.get("status") == "resolved"
                               for item in context_mapping.get("concepts", ()))
        else "partial"
    )
    context["mapping_status"] = mapping_status
    rules = tuple(DEFAULT_DETECTION_RULES if detection_rules is None else detection_rules)
    floors = {}
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise TypeError("expression detection rules must be mappings")
        minimum = _number(rule.get("minimum"))
        if (
            minimum is None
            or minimum <= 0
            or not _text(rule.get("basis"))
            or not _text(rule.get("reference"))
        ):
            raise ValueError(
                "expression detection rules require a positive minimum, basis and reference"
            )
        key = tuple(_text(rule.get(k)) for k in ("source", "datatype", "unit"))
        if not all(key) or key in floors:
            raise ValueError("duplicate or incomplete expression detection rule")
        floors[key] = dict(rule)
    grouped = defaultdict(lambda: defaultdict(list))
    observed = defaultdict(set)
    metadata_errors = set()
    incomplete_panels = set()
    rejected = 0
    incomplete = False
    for record in records:
        if not isinstance(record, Mapping):
            rejected += 1
            metadata_errors.add((None,) * 7)
            continue
        raw = _map(_map(record.get("upstream")).get("raw"))
        release = str(_map(record.get("provenance")).get("data_release") or "unspecified")
        rows = raw.get("tissues", record.get("tissues", ()))
        record_complete = _coverage_complete(raw, rows)
        if not record_complete:
            incomplete = True
        if not isinstance(rows, (list, tuple)):
            rejected += 1
            metadata_errors.add((None, None, None, release, None, None, None))
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                rejected += 1
                metadata_errors.add((None, None, None, release, None, None, None))
                continue
            value = _number(row.get("median"))
            source, dtype, unit = (
                _text(row.get(k)) for k in ("datasourceId", "datatypeId", "unit")
            )
            tissue, tissue_name = _biosample(row.get("tissueBiosample"))
            cell, cell_name = _biosample(row.get("celltypeBiosample"))
            valid_biosamples = all(
                _valid_biosample(row.get(field))
                for field in ("tissueBiosample", "celltypeBiosample")
            )
            if "rna" in row and "median" not in row:
                rna = _map(row.get("rna"))
                value = _number(rna.get("value"))
                source = _text(
                    _map(record.get("provenance")).get("source") or "Open Targets frozen expression"
                )
                dtype, unit = "rna", _text(rna.get("unit"))
                tissue, tissue_name = _biosample(row.get("tissue", row))
                valid_biosamples = _valid_biosample(row.get("tissue", row))
            if not source or not dtype or not unit or not (tissue or cell) or not valid_biosamples:
                rejected += 1
                # The missing field must not silently discard a potentially
                # higher background. Block every compatible comparison panel.
                metadata_errors.add(
                    (source or None, dtype or None, unit or None, release, None, None, None)
                )
                continue
            # Separate tissue summaries, global cell summaries and cell types
            # within each organ. These cannot serve as each other's background.
            axis = "cell_type" if cell else "tissue"
            stratum = (tissue, tissue_name) if cell and tissue else ("", "")
            key = (source, dtype, unit, release, axis, *stratum)
            sample = (tissue, cell, tissue_name, cell_name)
            observed[key].add(sample)
            if not record_complete:
                incomplete_panels.add(key)
            if value is None:
                rejected += 1
                continue
            # Preserve differently labelled conditions even when ontology IDs
            # coincide. Exact duplicate rows never change expression weights.
            grouped[key][sample].append(value)
    groups = []
    for key in sorted(observed):
        samples = grouped.get(key, {})
        values = {s: _unique_median(v) for s, v in sorted(samples.items())}
        missing = observed[key] - values.keys()
        transformed = {s: math.log1p(v) for s, v in values.items()}
        mean = _mean(list(transformed.values())) if transformed else None
        sd = (
            pstdev(transformed.values())
            if len(values) > 1 else 0.0
        )
        maximum = max(values.values(), default=0)
        tau = (
            math.fsum(1 - v / maximum for v in values.values()) / (len(values) - 1)
            if maximum > 0 and len(values) > 1
            else None
        )
        peaks = [s for s, v in values.items() if v == maximum] if maximum > 0 else []
        rule = floors.get(tuple(v.casefold() for v in key[:3]))
        status = (
            "incomplete_retrieval"
            if key in incomplete_panels
            else "invalid_reference_metadata"
            if any(_compatible(pattern, key) for pattern in metadata_errors)
            else "invalid_reference_values"
            if missing
            else "insufficient_reference_biosamples"
            if len(values) < parameter("expression.min_comparison_biosamples")
            else "detection_rule_unavailable"
            if rule is None
            else "evaluated"
        )
        evaluable = status == "evaluated"
        minimum = rule["minimum"] if rule else None
        matched = []
        for sample in sorted(values):
            selectors = [s for s in context["selectors"] if matches(s, sample)]
            if not selectors:
                continue
            value = values[sample]
            others = [v for s, v in values.items() if s != sample]
            background = max(others) if others else None
            background_mean = _mean(others) if others else None
            detected = value >= minimum if minimum is not None else None
            enriched = background is not None and value > 0 and value / fold_threshold >= background
            enhanced = (
                background_mean is not None
                and value > 0
                and value / fold_threshold >= background_mean
            )
            z = (transformed[sample] - mean) / sd if sd > 1e-12 else None
            qualified = evaluable and detected is True and enriched
            matched.append(
                {
                    "tissue_id": sample[0],
                    "cell_type_id": sample[1],
                    "tissue": sample[2],
                    "cell_type": sample[3],
                    "matched_context_selectors": selectors,
                    "expression_value": value,
                    "assessment_status": status,
                    "assessed_background_count": len(others),
                    "missing_background_count": len(missing),
                    "comparison_scope": "available_reference_biosamples",
                    "detection_pass": detected,
                    "background_max": background,
                    "background_mean": background_mean,
                    "fold_enrichment": _ratio(value, background)
                    if background is not None
                    else None,
                    "zero_background": background == 0 and value > 0,
                    "tissue_enriched_fold_pass": enriched,
                    "tissue_enhanced_fold_pass": enhanced,
                    "z": z,
                    "z_above_support_threshold": z is not None and z > threshold,
                    "above_threshold": qualified,
                }
            )
        tests = []
        for definition in context["predefined_groups"]:
            if definition["axis"] != key[4]:
                continue
            membership = [
                [s for s in values if matches(member, s)]
                for member in definition["members"]
            ]
            members = {s for hits in membership for s in hits}
            # A declared lineage can have multiple catalog-resolved subtypes.
            # Use all of them, with equal weight per declared lineage so richer
            # sampling of T cells cannot outweigh a sparsely sampled B lineage.
            # Duplicate identities with different condition labels remain
            # ambiguous for group admission; their measurements stay separate.
            identities = [[resolved_sample_identity(s, key[4]) for s in hits]
                          for hits in membership]
            ambiguous = any(
                len(hits) > 1 and (None in ids or len(set(ids)) != len(ids))
                for hits, ids in zip(membership, identities)
            ) or len(members) != sum(map(len, membership))
            resolved = all(membership) and not ambiguous
            outside = [v for s, v in values.items() if s not in members]
            group_status = (
                "not_observed_in_panel"
                if not members
                else status
                if not evaluable
                else "missing_or_ambiguous_group_members"
                if not resolved
                else "no_outside_reference"
                if not outside
                else "evaluated"
            )
            member_means = [_mean([values[s] for s in sorted(hits)]) if hits else None
                            for hits in membership]
            available_means = [value for value in member_means if value is not None]
            avg = _mean(available_means) if available_means else None
            bg = max(outside) if outside else None
            detected = (
                all(values[s] >= minimum for s in members)
                if members and minimum is not None
                else None
            )
            fold_pass = (
                avg is not None and bg is not None and avg > 0 and avg / fold_threshold >= bg
            )
            tests.append(
                {
                    "id": definition["id"],
                    "rationale": definition["rationale"],
                    "members": definition["members"],
                    "resolved_biosamples": [list(s) for s in sorted(members)],
                    "declared_member_count": len(membership),
                    "assessed_member_count": len(available_means),
                    "missing_member_count": len(membership) - len(available_means),
                    "member_biosample_counts": [len(hits) for hits in membership],
                    "member_means": member_means,
                    "ambiguous_membership": ambiguous,
                    "assessed_outside_count": len(outside),
                    "missing_reference_count": len(missing),
                    "group_mean_scope": "all_declared_members" if resolved else "available_members_only",
                    "status": group_status,
                    "group_mean": avg,
                    "outside_max": bg,
                    "all_members_detected": detected if resolved else None,
                    "observed_members_detected": detected,
                    "observed_fold_pass": fold_pass,
                    "fold_enrichment": _ratio(avg, bg)
                    if avg is not None and bg is not None
                    else None,
                    "zero_background": bg == 0 and avg is not None and avg > 0,
                    "qualified": group_status == "evaluated" and detected is True and fold_pass,
                }
            )
        groups.append(
            {
                "datasource": key[0],
                "datatype": key[1],
                "unit": key[2],
                "data_release": key[3],
                "comparison_axis": key[4],
                "comparison_tissue_id": key[5],
                "comparison_tissue": key[6],
                "reference_biosamples": len(values),
                "observed_biosample_count": len(observed[key]),
                "missing_biosample_count": len(missing),
                "unavailable_biosamples": [list(s) for s in sorted(missing)],
                "retrieval_complete": key not in incomplete_panels,
                "reference_complete": not missing and key not in incomplete_panels and not any(
                    _compatible(pattern, key) for pattern in metadata_errors
                ),
                "descriptive_status": "evaluated" if len(values) > 1 else "insufficient_finite_values",
                "statistics_scope": "finite_observed_biosamples_no_imputation",
                "reference_panel": [
                    {
                        "tissue_id": s[0],
                        "cell_type_id": s[1],
                        "tissue": s[2],
                        "cell_type": s[3],
                        "expression_value": v,
                    }
                    for s, v in sorted(values.items())
                ],
                "log1p_mean": mean,
                "log1p_sd": sd,
                "tau": tau,
                "tau_scale": "linear_median",
                "tau_peak_biosamples": [list(s) for s in peaks],
                "tau_peak_matches_context": any(
                    matches(selector, s)
                    for s in peaks
                    for selector in context["selectors"]
                ),
                "detection_rule": rule,
                "status": status,
                "matched_biosamples": matched,
                "group_enrichment": tests,
            }
        )
    qualified = any(s["above_threshold"] for g in groups for s in g["matched_biosamples"]) or any(
        t["qualified"] for g in groups for t in g["group_enrichment"]
    )
    evaluated = any(g["status"] == "evaluated" and g["matched_biosamples"] for g in groups)
    # A resolved negative in one lineage cannot establish a negative for another
    # requested lineage seen only in an assay without a detection rule. Keep
    # observed-data calculations distinct from coverage of the frozen context.
    unassessed_selectors = [selector for selector in context["selectors"] if not any(
        selector in sample["matched_context_selectors"]
        for group in groups if group["status"] == "evaluated"
        for sample in group["matched_biosamples"]
    )]
    unobserved_selectors = [selector for selector in context["selectors"] if not any(
        matches(selector, sample) for samples in observed.values() for sample in samples
    )]
    # False means all observed, supported context comparisons are resolved.
    # Another complete panel's negative cannot hide an invalid or small panel,
    # nor a predefined group that could not resolve every requested member.
    unresolved = (
        any(
            key[:3] in floors
            and any(
                matches(selector, sample)
                for sample in samples
                for selector in context["selectors"]
            )
            and (bool(samples - grouped.get(key, {}).keys()) or
                 len(grouped.get(key, {})) < parameter("expression.min_comparison_biosamples"))
            for key, samples in observed.items()
        )
        or any(
            group["detection_rule"] is not None
            and group["matched_biosamples"]
            and (
                group["status"] != "evaluated"
                or any(test["status"] not in {"evaluated", "not_observed_in_panel"}
                       for test in group["group_enrichment"])
            )
            for group in groups
        )
        or any(any(_compatible(pattern[:3], key) for key in floors) for pattern in metadata_errors)
    )
    return {
        "method": EXPRESSION_METHOD,
        "method_reference": HPA_METHOD_URL,
        "fold_threshold": fold_threshold,
        "fold_comparison": ">=",
        "threshold": threshold,
        "z_role": "support_only_never_admits",
        "tau_role": "support_only_never_admits",
        "context_terms": context["terms"],
        "context_definition": context,
        "profiles": {
            "tissue": _axis_profile(groups, axis="tissue", selectors=context["selectors"]),
            "cell": _axis_profile(groups, axis="cell_type", selectors=context["selectors"]),
        },
        "detection_rules": list(rules),
        "qualified": qualified,
        "retrieval_complete": not incomplete,
        "assessment_complete": not incomplete and not unresolved and
        not unassessed_selectors and mapping_status in {"resolved", "legacy"},
        "context_mapping_status": mapping_status,
        "context_coverage_complete": not unassessed_selectors,
        "unassessed_context_selectors": unassessed_selectors,
        "unobserved_context_selectors": unobserved_selectors,
        "evaluated_panel_count": sum(g["status"] == "evaluated" and bool(g["matched_biosamples"])
                                     for g in groups),
        "descriptive_panel_count": sum(g["descriptive_status"] == "evaluated" for g in groups),
        "status": "context_undefined"
        if not context["selectors"]
        else "qualified"
        if qualified
        else "incomplete_retrieval"
        if incomplete
        else "context_mapping_unresolved"
        if mapping_status not in {"resolved", "legacy"}
        else "unresolved_expression_comparisons"
        if unresolved or (evaluated and unassessed_selectors)
        else "evaluated_below_threshold"
        if evaluated
        else "context_not_observed"
        if groups and not any(g["matched_biosamples"] for g in groups)
        else "insufficient_expression_data",
        "groups": groups,
        "rejected_row_count": rejected,
        "invalid_metadata_patterns": [
            list(pattern)
            for pattern in sorted(metadata_errors, key=lambda p: tuple(v or "" for v in p))
        ],
    }
