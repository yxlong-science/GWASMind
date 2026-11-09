"""Local selection and candidate-scope counts; no significance inference for LD."""
from __future__ import annotations

import re

from .source_gwas import summarize_source_gwas_observations


_WORDS = {word: number for number, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"))}
_COUNT = r"(?:\d+|" + "|".join(_WORDS) + r")"
_SOURCE_NOUN = r"(?:(?:source[- ]selected|user[- ]selected|uploaded|source|selected|original)\s+)+(?:SNPs?|variants?)"
_MEET = r"(?:meeting|meet|met|reaching|reach|reached|passing|pass|passed)"
_THRESHOLD = r"(?P<threshold>[^.;\n()]{0,90}?\b(?:GWAS|genome[- ]wide)\s+(?:significance\s+)?threshold)"
_COUNTED = re.compile(
    rf"\b(?P<all>all\s+)?(?P<count>{_COUNT})\s+(?P<noun>{_SOURCE_NOUN})\s+"
    rf"(?:(?:are|were)\s+)?(?P<verb>{_MEET})\s+{_THRESHOLD}", re.IGNORECASE)
_FRACTION = re.compile(
    rf"\b(?P<count>{_COUNT})\s+(?:of|out of|/)\s+(?:the\s+)?(?P<total>{_COUNT})\s+"
    rf"(?P<noun>{_SOURCE_NOUN})\s+(?:(?:are|were)\s+)?(?P<verb>{_MEET})\s+{_THRESHOLD}", re.IGNORECASE)


def _number(value):
    return int(value) if value.isdigit() else _WORDS[value.casefold()]


def source_selection_count_corrections(text, context):
    return _scope_count_corrections(text, context, include_gene_counts=False)


def report_scope_count_corrections(text, context):
    return _scope_count_corrections(text, context, include_gene_counts=True)


def _scope_count_corrections(text, context, *, include_gene_counts):
    """Correct explicit selection totals without relabelling selected subsets.

    This count is independent of phenotype significance, LD expansion and task
    intent. Bare counts of variants with a feature, external-study selections,
    conditionals, quoted claims and comparisons do not assert this total.
    """
    from .report_facts import computed_report_facts
    facts = computed_report_facts(context.source_observations, context.result)
    patterns = [
        ("source_variant_count", rf"\b(?P<scope>there\s+(?:are|were)|a\s+total\s+of|all)\s+(?P<count>{_COUNT})\s+{_SOURCE_NOUN}\b"),
        ("source_variant_count", rf"\b(?P<scope>(?:this|the|current)\s+(?:analysis|report|selection|request)\s+"
        rf"(?:includes|contains|covers|comprises))\s+(?P<count>{_COUNT})\s+"
        rf"(?:(?:selected|uploaded|source)\s+)?(?:SNPs?|variants?)\b"),
        ("source_variant_count", rf"\b(?P<count>{_COUNT})\s+(?:SNPs?|variants?)\s+(?P<scope>(?:were|are|have\s+been)\s+selected)\b"),
    ]
    if include_gene_counts:
        patterns.extend((
            ("candidate_pool_count", rf"\b(?P<scope>candidate\s+pool\s+(?:contains|includes|has|comprises))\s+(?P<count>{_COUNT})\s+(?:candidate\s+)?genes?\b"),
            ("shortlist_count", rf"\b(?P<scope>shortlist\s+(?:contains|includes|has|comprises))\s+(?P<count>{_COUNT})\s+(?:candidate\s+)?genes?\b"),
            ("shortlist_count", rf"\b(?P<scope>shortlist\s+of)\s+(?P<count>{_COUNT})\s+(?:candidate\s+genes?|candidates?|genes?)\b"),
            ("candidate_pool_count", rf"\b(?P<scope>candidate\s+pool\s+of)\s+(?P<count>{_COUNT})\s+(?:candidate\s+genes?|candidates?|genes?)\b"),
            ("shortlist_count", rf"\b(?P<scope>there\s+(?:are|were)|a\s+total\s+of)\s+(?P<count>{_COUNT})\s+(?:shortlisted|short-listed)\s+(?:candidate\s+)?genes?\b"),
            ("ranked_count", rf"\b(?P<scope>(?:this|the|current)\s+(?:analysis|report)\s+(?:returned|ranked|prioritized))\s+(?P<count>{_COUNT})\s+(?:(?:ranked|prioritized)\s+)?genes?\b"),
        ))
    significance_edits = source_significance_count_corrections(text, context)
    edits = []
    for metric, pattern in patterns:
        expected = facts.get(metric)
        if type(expected) is not int or (metric == "source_variant_count" and not context.source_observations):
            continue
        if metric == "ranked_count" and not isinstance(context.result.get("ranked_genes"), (list, tuple)):
            continue
        for match in re.finditer(pattern, text, re.I):
            if _number(match["count"]) == expected:
                continue
            prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
            suffix = re.split(r"[.!?;\n]", text[match.end():], maxsplit=1)[0]
            line_prefix = text[text.rfind("\n", 0, match.start()) + 1:match.start()]
            if re.match(r"\s*>", line_prefix):
                continue
            if any(opening in prefix and closing in suffix for opening, closing in (("\"", "\""), ("\u201c", "\u201d"))):
                continue
            if re.search(r"\b(?:if|whether|suppose|hypothesis|hypothetical|incorrect|unsupported|not|no|claim|"
                         r"external|published|previous|prior|another|possibly|perhaps|maybe|might|may|could|"
                         r"likely|uncertain|between|rather\s+than)\b", prefix, re.I):
                continue
            if re.search(r"\b(?:if|unless|would|could|might|hypothetical)\b", suffix, re.I):
                continue
            if re.search(r"\b(?:among|of|out of|at least|at most|up to)\s*$|/\s*$", prefix, re.I):
                continue
            if re.search(r"\b(?:external|published|previous|prior|another)\s+(?:study|GWAS|selection)\b", suffix, re.I):
                continue
            if re.match(r"\s+(?:by|for|in|from)\b[^.;\n]{0,160}\b(?:study|cohort|biobank|consortium|publication)\b", suffix, re.I):
                continue
            if match["scope"].casefold() != "all" and re.match(
                    r"\s+(?:with|without|that|which|having|overlapping|within|carrying|showing|associated)\b", suffix, re.I):
                continue
            start, end = match.span("count")
            if any(edit["start"] < end and start < edit["end"] for edit in (*edits, *significance_edits)):
                continue
            edits.append({"start": start, "end": end, "replacement": str(expected),
                "metric": metric, metric: expected,
                "reason": "selected_source_total_contradicted_by_uploaded_selection" if metric == "source_variant_count"
                    else "candidate_scope_total_contradicted_by_recorded_counts"})
    return edits


def source_significance_count_corrections(text, context):
    """Correct explicit impossible source counts, preserving the surrounding prose.

    A smaller significant subset can be legitimate. An affirmative count above
    the number measured significant in the uploaded selection cannot be. The
    pattern requires a source scope and a GWAS/genome-wide threshold; QTL and
    nominal thresholds, negative statements and hypothetical claims are excluded.
    """
    if not context.source_observations:
        return []
    source = summarize_source_gwas_observations(context.source_observations, phenotype=context.source_trait)
    if source.variants_with_p_value == 0:
        return []
    edits = []

    def asserted(match):
        prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
        if re.search(r"\b(?:if|whether|suppose|hypothesis|hypothetical|incorrect|unsupported|not|no|claim)\b", prefix, re.I):
            return False
        if re.search(r"\b(?:among|of|out of|at least|at most|up to)\s*$|/\s*$", prefix, re.I):
            return False
        return not re.search(r"\b(?:eQTL|sQTL|pQTL|molQTL|QTL|nominal|FDR|suggestive)\b", match.group("threshold"), re.I)

    def add(match, replacement):
        edits.append({"start": match.start(), "end": match.end(), "replacement": replacement,
            "metric": "source_significance_count", "source_variant_count": source.selected_variants,
            "source_variants_with_p_value": source.variants_with_p_value,
            "source_significant_variant_count": source.genome_wide_significant_variants,
            "significance_threshold": source.significance_threshold,
            "reason": "source_gwas_significance_count_contradicted_by_uploaded_p_values"})

    for match in _FRACTION.finditer(text):
        if not asserted(match):
            continue
        if (_number(match["count"]) == source.genome_wide_significant_variants
                and _number(match["total"]) == source.selected_variants):
            continue
        add(match, f"{source.genome_wide_significant_variants} of {source.selected_variants} "
            f"source variants meet {match['threshold']}")
    for match in _COUNTED.finditer(text):
        if not asserted(match) or any(edit["start"] <= match.start() < edit["end"] for edit in edits):
            continue
        if _number(match["count"]) <= source.genome_wide_significant_variants:
            continue
        # Keep total selection and the significant subset distinct. The rest of
        # the sentence (including a separately stated LD-proxy count) is untouched.
        verb = "has" if source.genome_wide_significant_variants == 1 else "have"
        add(match, f"{source.selected_variants} source variants "
            f"({source.genome_wide_significant_variants} {verb} measured P values meeting {match['threshold']})")
    return edits
