"""Population-specific, bounded prior-association screening.

This is a reproducible project definition, not a test of causal independence.
An r-squared value equal to the configured threshold belongs to the known side.
Unknown identity, missing LD, or incomplete retrieval never becomes zero LD.
The original selection is retained separately from reference-panel rs identifiers.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .configuration import parameter


@dataclass(frozen=True)
class PopulationChoice:
    code: str | None
    explicit: bool
    assumption: str = ""


@dataclass(frozen=True)
class NoveltyPopulationChoice(PopulationChoice):
    reported_study_population: str | None = None
    reported_study_population_explicit: bool = False
    reported_study_population_assumption: str = ""


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def _contains(text: str, term: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(term.casefold())
                          + r"(?![a-z0-9])", text.casefold()))


def _negated_reference_prefix(text: str) -> bool:
    """Bind exclusion to the immediately following choice, not an entire sentence."""
    return bool(re.search(
        r"(?:\b(?:not|never|without|except|excluding)|\b(?:do\s+not|don't)\s+(?:use|select|choose)|"
        r"\u4e0d\u7528|\u4e0d\u8981(?:\u7528|\u4f7f\u7528|\u91c7\u7528)?|\u4e0d\u4f7f\u7528|\u4e0d\u91c7\u7528|\u4e0d\u9009(?:\u62e9)?|\u6392\u9664)\s*(?:the\s+)?$", text, re.I))


def _reference_population_mentions(question, mentions, default_pattern):
    """Locate panel instructions without importing source-study background."""
    requested = []
    command = r"\b(?:use|using|select|choose|apply|prefer)\s+(?:the\s+)?|\u4f7f\u7528|\u91c7\u7528|\u9009\u62e9|\u7528"
    source = r"(?:source|uploaded|original)\s+(?:GWAS|study|data)|\u6e90(?:\u7814\u7a76|\u6570\u636e)|\u4e0a\u4f20"
    for start, end, code in mentions:
        prefix = re.split(r"[;.!?\n\uff1b\u3002\uff1f\uff01]", question[:start])[-1]
        suffix = question[end:]
        commands = list(re.finditer(command, prefix, re.I))
        backgrounds = list(re.finditer(source, prefix, re.I))
        direct = commands and not prefix[commands[-1].end():].strip()
        qualified = re.match(r"\s+(?:LD|reference)(?!\w)", suffix, re.I)
        coordinated = (re.search(r"(?:\b(?:and|or|plus)|[,\uff0c]|\u548c|\u6216|\u4ee5\u53ca)\s*$", prefix, re.I)
            and (re.search(default_pattern, prefix, re.I) or commands)
            and (not backgrounds or commands and commands[-1].start() > backgrounds[-1].start()))
        if direct or qualified or coordinated:
            requested.append((start, end, code))
    return requested


def _question_population_choice(question: str, aliases: Mapping[str, Sequence[str]]) -> PopulationChoice | None:
    """Recognize explicit default-reference and local exclusion instructions."""
    mentions = [(match.start(), match.end(), code) for code, words in aliases.items() for word in words
                for match in re.finditer(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", question, re.I)]
    affirmative = {code for start, _, code in mentions if not _negated_reference_prefix(question[:start])}
    excluded = {code for start, _, code in mentions if _negated_reference_prefix(question[:start])}
    default_pattern = (
        r"\b(?:use|using|select|choose|apply|prefer|and|or)\s+(?:the\s+)?(?:configured\s+)?default\s+"
        r"(?:LD(?:\s+reference|\s+panel)?|reference(?:\s+panel)?)\b"
        r"|(?:\u4f7f\u7528|\u91c7\u7528|\u9009\u62e9|\u7528|\u6309|\u548c|\u6216|\u4ee5\u53ca)(?:\u7cfb\u7edf|\u914d\u7f6e\u7684|\u8bbe\u7f6e\u7684)?\u9ed8\u8ba4(?:\u7684)?(?:LD(?:\u53c2\u8003)?|\u53c2\u8003(?:\u4eba\u7fa4|\u9762\u677f)?)"
    )
    defaults = tuple(re.finditer(default_pattern, question, re.I))
    requested_defaults = [match for match in defaults if not _negated_reference_prefix(question[:match.start()])
                          and not re.search(r"(?:\b(?:don't|do\s+not)|\u4e0d|\u4e0d\u8981)\s*$", question[:match.start()], re.I)]
    ambiguity = PopulationChoice(None, True, "Multiple or contradictory LD references requested; specify one LD reference.")
    reference_mentions = _reference_population_mentions(question, mentions, default_pattern)
    if reference_mentions:
        # A direct instruction such as "Use EUR LD" selects the reference.
        # Earlier source ancestry describes the uploaded cohort, not a second panel.
        affirmative = {code for start, _, code in reference_mentions
                       if not _negated_reference_prefix(question[:start])}
    if requested_defaults:
        default = parameter("novelty.default_population")
        # Source ancestry is background to a default-reference instruction.
        # An additional affirmative reference instruction still conflicts.
        requested = {code for start, _, code in reference_mentions
                     if not _negated_reference_prefix(question[:start])}
        if excluded.intersection({default, *requested}) or len({default, *requested}) > 1:
            return ambiguity
        return PopulationChoice(default, False, "The configured default LD reference was explicitly requested.")
    if excluded:
        if affirmative.intersection(excluded) or len(affirmative) != 1:
            return ambiguity
        return PopulationChoice(next(iter(affirmative)), True)
    if defaults and not affirmative:
        return PopulationChoice(None, True, "The default LD reference was excluded; specify one LD reference.")
    if reference_mentions:
        return PopulationChoice(next(iter(affirmative)), True) if len(affirmative) == 1 else ambiguity
    return None


def choose_population(question: str, population: str | None = None) -> PopulationChoice:
    """Use explicit request ancestry; source-GWAS ancestry is not a default override."""
    aliases = {parts[0]: parts for value in parameter("novelty.population_aliases")
               if (parts := value.split("|"))}
    if population is not None:
        matches = {code for code, words in aliases.items()
                   if any(_normalized(population) == _normalized(word) for word in words)}
    else:
        instruction = _question_population_choice(question, aliases)
        if instruction is not None:
            return instruction
        matches = {code for code, words in aliases.items()
                   if any(_contains(question, word) for word in words)}
    if len(matches) == 1:
        return PopulationChoice(next(iter(matches)), True)
    if len(matches) > 1:
        return PopulationChoice(None, True, "Multiple populations requested; specify one LD reference.")
    if population is not None:
        return PopulationChoice(None, True, "The requested LD population is not configured.")
    if re.search(r"\basian\b|\u4e9a\u6d32", question, re.I):
        code = parameter("novelty.asian_default_population")
        return PopulationChoice(code, True, f"Unqualified Asian ancestry is interpreted as {code}.")
    # Recognize unsupported explicit ancestry instead of silently declaring EUR.
    if re.search(r"\b(?:Oceanian|Oceanic|Middle Eastern|Arab|Pacific Islander|Aboriginal)\b|"
                 r"\u5927\u6d0b\u6d32|\u4e2d\u4e1c|\u592a\u5e73\u6d0b\u5c9b|\u539f\u4f4f\u6c11", question, re.I):
        return PopulationChoice(None, True, "The requested LD population is not configured.")
    return PopulationChoice(parameter("novelty.default_population"), False,
                            "No population specified; the configured default LD reference is used.")


def choose_novelty_populations(question: str) -> NoveltyPopulationChoice:
    """Separate LD reference choice from an explicitly scoped prior-study search."""
    terms = [word for group in parameter("novelty.population_aliases") for word in group.split("|")]
    terms.extend(("Asian", "\u4e9a\u6d32"))
    term_pattern = "(?:" + "|".join(re.escape(word) for word in sorted(set(terms), key=len, reverse=True)) + ")"
    spans = []
    for clause in re.finditer(r"[^;,.!?\n\u3002\uff1b\uff0c\uff1f\uff01]+", question):
        text = clause.group()
        if not (re.search(r"\b(?:reported|published|studies|reports|associations|novel|new|known)\b|"
                          r"\u62a5\u9053|\u53d1\u8868|\u65e2\u5f80|\u4ee5\u524d|\u4e4b\u524d", text, re.I)
                or is_prior_report_question(text)):
            continue
        patterns = (
            r"\b(?:in|among|from)\s+(?P<population>[^;,.!?\n]{1,100}?)\s+(?:populations?|cohorts?|ancestr(?:y|ies)|people)\b",
            r"\b(?:in|among)\s+(?P<population>" + term_pattern + r")\s*$",
            r"(?P<population>" + term_pattern + r")\s+(?:prior\s+|published\s+)?(?:studies|cohorts|reports)\b",
            r"(?:\u5728|\u9488\u5bf9|\u4e8e)(?P<population>[\u3400-\u9fffA-Za-z -]{1,40}?)(?:\u4eba\u7fa4|\u7956\u6e90|\u961f\u5217)",
            r"(?P<population>" + term_pattern + r")(?:\u4eba\u7fa4|\u4eba|\u7956\u6e90)(?:\u4e2d|\u7684)?",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                if (re.search(r"\bLD\b|\breference\b", match.group("population"), re.I)
                        or re.match(r"\s*(?:LD\b|reference\b|\u53c2\u8003)", text[match.end():], re.I)):
                    continue
                spans.append((clause.start() + match.start(), clause.start() + match.end(), match.group("population")))
        # Configured Chinese aliases such as "Chinese people" already contain
        # their population noun and need no second noun after the alias.
        for match in re.finditer(term_pattern, text, re.I):
            if (match.group().endswith("\u4eba")
                    and not re.match(r"\s*(?:LD\b|reference\b|\u53c2\u8003)", text[match.end():], re.I)):
                spans.append((clause.start() + match.start(), clause.start() + match.end(), match.group()))
    reported = PopulationChoice(None, False, "Prior reports are searched across all study populations.")
    if spans:
        choice = choose_population(" ".join(value for _, _, value in spans))
        reported = PopulationChoice(choice.code if choice.explicit else None, True,
            choice.assumption if choice.explicit else "The requested prior-study population is not configured.")
    remaining = list(question)
    for start, end, _ in spans:
        remaining[start:end] = " " * (end - start)
    reference_text = "".join(remaining)
    reference = choose_population(reference_text)
    if (spans and not reference.explicit
            and "explicitly requested" not in reference.assumption):
        # Preserve the established same-population reference when the user only
        # specifies an ancestry for prior reporting, without a separate panel.
        # Merely asking to include LD-linked variants does not select a panel.
        reference = choose_population(" ".join(value for _, _, value in spans))
    return NoveltyPopulationChoice(reference.code, reference.explicit, reference.assumption,
        reported.code, reported.explicit, reported.assumption)


def canonical_trait(trait: str) -> str:
    """Exact, documented aliases only; subtypes and related diseases stay separate."""
    key = _normalized(trait)
    for group in parameter("novelty.trait_alias_groups"):
        words = group.split("|")
        if any(key == _normalized(word) for word in words):
            return words[0]
    return trait.strip()


def _reported_trait_key(trait: str) -> str:
    """Ignore only a trailing explicit PheCode identifier, never a subtype."""
    label = re.sub(r"\s*\(\s*PheCode\s+\d+(?:\.\d+)?\s*\)\s*$", "", trait, flags=re.I)
    return _normalized(canonical_trait(label))


_PRIOR_ASSOCIATION_REQUEST = re.compile(
    r"\b(?:has|have)\s+(?:(?:this|that|these|those|the|selected|current|my|our)\s+){0,3}"
    r"(?:\d+\s+)?(?:locus|loci|region|snps?|variants?|rs\d+|it|they)\s+"
    r"(?:(?:ever|already|previously|not)\s+)?been\s+(?:associated|linked)\s+(?:with|to)\b",
    re.IGNORECASE,
)
_PRIOR_REPORT_REQUEST = re.compile(r"\bprior[-\s]+(?:reports?|reporting|studies|publications?)\b", re.IGNORECASE)
_NO_PRIOR_REPORT_SCREEN = re.compile(
    r"\b(?:do\s+not|don't|without|skip|omit)\s+(?:(?:any|a|the)\s+)?"
    r"(?:(?:check(?:ing)?|assess(?:ing)?|screen(?:ing)?|search(?:ing)?)\s+)?"
    r"(?:(?:any|a|the)\s+)?"
    r"(?:novelty|prior[-\s]+(?:reports?|reporting|studies|publications?))\b", re.IGNORECASE,
)


def is_prior_report_question(question: str) -> bool:
    """Recognize explicit prior-report requests, not ordinary association prose."""
    return bool(not _NO_PRIOR_REPORT_SCREEN.search(question)
                and (_PRIOR_ASSOCIATION_REQUEST.search(question) or _PRIOR_REPORT_REQUEST.search(question)))


def strip_report_population_suffix(phenotype: str) -> str:
    """Remove a trailing configured ancestry scope without erasing disease subtypes."""
    match = re.search(
        r"\s+(?:in|among|from)\s+(?P<scope>.+?)\s+"
        r"(?:populations?|cohorts?|people|participants?|ancestr(?:y|ies))$",
        phenotype, re.IGNORECASE,
    )
    if match is None:
        return phenotype
    terms = [word for group in parameter("novelty.population_aliases") for word in group.split("|")]
    terms.append("Asian")
    names = "(?:" + "|".join(re.escape(word) for word in sorted(set(terms), key=len, reverse=True)) + ")"
    scope = r"(?:(?:the|a|an)\s+)?" + names + r"(?:[- ]ancestry)?"
    scope += r"(?:\s+(?:and|or)\s+" + names + r"(?:[- ]ancestry)?)*"
    if not re.fullmatch(scope, match["scope"], re.IGNORECASE):
        return phenotype
    return phenotype[:match.start()].rstrip()


def is_novelty_question(question: str) -> bool:
    """Applied only after the chat router selected an actual retrieval task."""
    if _NO_PRIOR_REPORT_SCREEN.search(question):
        return False
    return bool(re.search(
        r"\b(?:novel|novelty|new locus|new association|known|previously|reported|published)\b|"
        r"\u65b0(?:\u7684)?\u4f4d\u70b9|\u65e2\u5f80|\u5df2\u77e5|\u4ee5\u524d|\u4e4b\u524d|\u62a5\u9053|\u53d1\u8868", question, re.I,
    ) or is_prior_report_question(question))


def classify_ld_screen(*, pairs: Sequence[Mapping[str, Any]], coverage_complete: bool,
                       identity_complete: bool, threshold: float,
                       prior_variant_count: int | None, exact_matches: Sequence[str] = ()) -> dict[str, Any]:
    """Pure decision kernel; callers must enumerate every required pair in scope.

    A verified positive can establish a known connection even if another query
    failed. A negative claim requires complete retrieval, identity and pair coverage.
    """
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("LD threshold must be finite and in (0, 1].")
    numeric = []
    for pair in pairs:
        value = pair.get("r2")
        if (isinstance(value, (float, int)) and not isinstance(value, bool)
                and math.isfinite(value) and 0 <= value <= 1
                and pair.get("status") == "computed"):
            numeric.append(float(value))
    if exact_matches or any(value >= threshold for value in numeric):
        status, label = "PREVIOUSLY_REPORTED", "Previously reported or linked to a reported SNP"
    elif (prior_variant_count is None or not coverage_complete or not identity_complete or len(numeric) != len(pairs)
          or (prior_variant_count > 0 and not pairs)):
        status, label = "UNRESOLVED", "Novelty unresolved"
    else:
        # Keep the historical machine state for Auto Analysis compatibility;
        # a complete negative database screen is not proof of global novelty.
        status, label = "NOVEL_IN_SCOPE", "No prior association found in the queried database version and search scope"
    return {"status": status, "label": label, "r2_threshold": threshold,
            "max_r2": max(numeric) if numeric else None,
            "computed_pairs": len(numeric), "required_pairs": len(pairs),
            "coverage_complete": coverage_complete, "identity_complete": identity_complete,
            "prior_variant_count": prior_variant_count,
            "exact_matches": list(exact_matches), "pairs": [dict(pair) for pair in pairs]}


def _native_prior_candidates(native: Mapping[str, Any], source_ids: Sequence[str], trait: str,
                             *, all_phenotypes: bool = False):
    """Bind a native positive to one coherent original record and this selection.

    Display flattening may choose one of several duplicate metadata fields. A
    scientific shortcut must first reject conflicts rather than inherit that
    preference. Missing legacy fields are tolerated when the remaining explicit
    lead identity and publication are consistent.
    """
    from .web.chat_response import _evidence_rows, _is_trait_study, _pmid, _study_associations

    def identifier(value):
        value = str(value or "").strip()
        if re.fullmatch(r"rs\d+", value, re.I):
            return value.casefold()
        match = re.fullmatch(r"(?:chr)?(\d+|X|Y|MT)[_:](\d+)[_:]([ACGTN]+)[_:]([ACGTN]+)", value, re.I)
        return (f"{match[1].upper()}_{int(match[2])}_{match[3].upper()}_{match[4].upper()}"
                if match else value)

    source_keys = {identifier(value) for value in source_ids}
    resolution = native.get("variant_resolution") or {}
    resolution = resolution if isinstance(resolution, Mapping) else {}
    source_mapping = resolution.get("source_to_identified_variant") or {}
    source_mapping = source_mapping if isinstance(source_mapping, Mapping) else {}
    aliases = {identifier(source): identifier(target) for source, target in source_mapping.items()
               if identifier(source) in source_keys and isinstance(target, str) and target.strip()}
    selected = source_keys | set(aliases.values())
    matches, binding_issues = [], []
    seen = set()
    for record in _evidence_rows(native):
        if str(record.get("kind") or "") not in {"study", "gwas"}:
            continue
        payload = record.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        upstream = payload.get("upstream")
        raw = upstream.get("raw") if isinstance(upstream, Mapping) else {}
        raw = raw if isinstance(raw, Mapping) else {}
        context = payload.get("context")
        context = context if isinstance(context, Mapping) else {}
        subject = payload.get("subject")
        subject = subject if isinstance(subject, Mapping) else {}
        member = raw.get("matchedLocusMember")
        member = member if isinstance(member, Mapping) else {}
        if not all_phenotypes and _reported_trait_key(str(raw.get("traitFromSource") or "")) != _reported_trait_key(trait):
            continue
        query_fields = {key: identifier(value) for key, value in (
            ("variant_id", record.get("variant_id")), ("subject.id", subject.get("id")),
            ("queryVariantId", raw.get("queryVariantId")), ("variantId", raw.get("variantId")),
            ("matchedLocusMember.variantId", member.get("variantId")),
        ) if value is not None and str(value).strip()}
        lead = identifier(raw.get("leadVariantId"))
        if not selected.intersection((*query_fields.values(), lead)):
            # A coherent record for another variant cannot affect this source.
            continue
        errors = []
        query_ids = {aliases.get(value, value) for value in query_fields.values()}
        if len(query_ids) > 1:
            errors.append("query-variant identity fields")
        types = {str(value).strip().casefold() for value in
                 (raw.get("studyType"), context.get("study_type")) if value is not None and str(value).strip()}
        if len(types) > 1:
            errors.append("study-type fields")
        elif types and types != {"gwas"}:
            continue
        study_ids = {str(value).strip().casefold() for value in
                     (record.get("study_id"), raw.get("studyId"), context.get("study_id"))
                     if value is not None and str(value).strip()}
        if len(study_ids) > 1:
            errors.append("study identifiers")
        pmid_values = [value for value in (record.get("pmid"), raw.get("pubmedId"), context.get("pmid"))
                       if value is not None and str(value).strip()]
        pmids = {_pmid(value) for value in pmid_values}
        if len(pmids) > 1:
            errors.append("publication identifiers")
        role = str(raw.get("queryVariantRole") or "").strip().casefold()
        query_id = next(iter(query_ids), "")
        lead = aliases.get(lead, lead)
        if role and role not in {"lead", "credible_set_member"}:
            errors.append("query-variant role")
        elif role == "lead" and lead and query_id != lead:
            errors.append("query/lead role and identity")
        elif role == "credible_set_member" and lead and query_id == lead:
            errors.append("query/lead role and identity")
        if errors:
            binding_issues.append({"evidence_id": str(record.get("evidence_id") or ""),
                                  "fields": list(dict.fromkeys(errors))})
            continue
        if not lead or query_id != lead or role == "credible_set_member" or not pmids or "" in pmids:
            continue
        rows = _study_associations([record])
        if not rows or not rows[0].get("pmid") or not _is_trait_study(rows[0]):
            continue
        row = rows[0]
        row["variant_id"] = query_id
        row["lead_variant_id"] = lead
        row["query_variant_role"] = role
        row["evidence_ids"] = [str(record["evidence_id"])] if record.get("evidence_id") else []
        key = (query_id, _reported_trait_key(row["trait"]), row["study_id"].casefold(), row["pmid"])
        if key not in seen:
            seen.add(key)
            matches.append(row)
    return matches, binding_issues


def prior_screen_precondition(*, observations: Sequence[Mapping[str, Any]], phenotype: str,
                              native_result: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Return a local not-run receipt when the preceding acquisition cannot support a screen."""
    native = native_result or {}
    sufficiency = native.get("sufficiency") or {}
    failure = native.get("retrieval_failure") or {}
    resolution = native.get("variant_resolution") or {}
    status = str(native.get("status") or "")
    source_ids = [str(row.get("variant_id") or "").strip() for row in observations]
    code = message = None
    if not source_ids or not any(source_ids):
        code, message = "source_selection_missing", "No selected source variant is available."
    elif status == "INVALID_INPUT" or sufficiency.get("valid_input") is False:
        code, message = "source_input_invalid", "The selected source input could not be validated."
    elif status == "POLICY_BLOCKED":
        code, message = "acquisition_blocked", "The requested external evidence acquisition was blocked."
    elif not native.get("evidence") and (
        status == "SERVICE_UNAVAILABLE" or native.get("external_evidence_available") is False
        or failure.get("status") in {"unavailable", "failed", "error"}
    ):
        code, message = "external_acquisition_unavailable", "External evidence retrieval failed; only uploaded observations are retained."
    elif (
        resolution.get("complete") is False
        and type(resolution.get("identified_source_variant_count")) is int
        and resolution["identified_source_variant_count"] == 0
        and resolution.get("source_variant_count") == len(source_ids)
        and not resolution.get("source_to_identified_variant")
        and {row.get("variant_id") for row in resolution.get("unresolved_variants", [])
             if isinstance(row, Mapping)} == set(source_ids)
    ):
        code, message = "source_identity_unresolved", "No selected source SNP identity was resolved during the completed acquisition."
    if code is None:
        return None
    return {"phenotype": phenotype, "status": "UNRESOLVED", "label": "Novelty unresolved",
            "execution_status": "not_run", "precondition_reason": code, "precondition_message": message,
            "selected_variant_ids": source_ids, "coverage_complete": False, "identity_complete": False,
            "prior_variant_count": None, "required_pairs": None, "computed_pairs": 0, "max_r2": None,
            "query_ledger": [], "lookup_performed": False, "ld_lookup_performed": False,
            "issues": [message + " Prior-report and LD screening was not run; prior associations and their absence remain unassessed."]}


def _review_prior_ld_coverage(provider, details, source_ids, population):
    """Consume final LD receipts without changing the frozen provider's records."""
    from .review_coverage import _missing_proxy_rsid

    audit = getattr(provider, "proxy_lookup_issues", {})
    if not isinstance(audit, Mapping):
        return details
    reviews, receipts = [], []
    for source in dict.fromkeys(source_ids):
        key = (source, population)
        if key not in audit:
            continue
        row = audit[key]
        row = row if isinstance(row, Mapping) else {}
        excluded = row.get("excluded_rows", [])
        valid_rows = isinstance(excluded, list) and all(
            isinstance(item, Mapping) and type(item.get("scope_relevant")) is bool for item in excluded)
        in_scope = [item for item in excluded if item["scope_relevant"]] if valid_rows else []
        ignored = [item for item in in_scope if _missing_proxy_rsid(item)]
        count = row.get("excluded_scope_count")
        valid_count = count is None or type(count) is int and count == len(in_scope)
        state = row.get("state")
        window_gap = bool(row.get("window_coverage_reason")) or row.get("query_build") not in (
            None, parameter("novelty.reference_build"))
        missing_only = bool(in_scope) and len(ignored) == len(in_scope)
        complete = (valid_rows and valid_count and not window_gap
            and state in {"SUCCESS_DATA", "SUCCESS_NO_DATA", "PARTIAL"}
            and ((row.get("requested_scope_complete") is True and state != "PARTIAL" and not in_scope)
                 or missing_only))
        receipts.append({"source_variant_id": source, **deepcopy(dict(row))})
        reviews.append({"source_variant_id": source, "complete": complete,
                        "ignored_missing_rsid_rows": len(ignored)})
    if not reviews:
        return details
    incomplete = [row["source_variant_id"] for row in reviews if not row["complete"]]
    return {**details, "ld_lookup_audit": receipts, "ld_coverage_review": reviews,
            "coverage_complete": details.get("coverage_complete") is True and not incomplete,
            "prior_variant_count": None if incomplete else details.get("prior_variant_count"),
            "issues": [*details.get("issues", []), *([
                "LD reference retrieval remains incomplete for part of the selected scope."
            ] if incomplete else [])]}


def screen_prior_associations(*, observations: Sequence[Mapping[str, Any]], phenotype: str,
                              question: str, genome_build: str = "GRCh38",
                              backend=None, native_result: Mapping[str, Any] | None = None,
                              progress=None, require_model_phenotype: bool = False,
                              phenotype_completer=None, phenotype_model_id: str = "",
                              cancel_event=None) -> dict[str, Any]:
    """Perform a scoped screen using a transport provider or an explicit test provider."""
    if cancel_event is not None and cancel_event.is_set():
        from concurrent.futures import CancelledError
        raise CancelledError("Prior-report screening was cancelled.")
    not_run = prior_screen_precondition(observations=observations, phenotype=phenotype, native_result=native_result)
    if not_run is not None:
        return not_run
    from .providers.ld_factory import (
        LDProviderConfigurationError, create_ld_provider, ld_configuration_issue, ld_provider_metadata,
    )
    from .processing_scope import RESULT_PROCESSING_LIMIT_MESSAGE, is_source_only_processing_limit
    if is_source_only_processing_limit(native_result or {}):
        return {"phenotype": phenotype, "status": "UNRESOLVED",
                "label": "Novelty unresolved", "source_only_processing_limit": True,
                "coverage_complete": False, "identity_complete": False,
                "prior_variant_count": None, "required_pairs": None,
                "computed_pairs": 0, "max_r2": None,
                "issues": [RESULT_PROCESSING_LIMIT_MESSAGE]}
    population = choose_novelty_populations(question)
    source_ids = [str(row.get("variant_id") or "") for row in observations]
    base = {"phenotype": phenotype, "queried_trait": canonical_trait(phenotype),
            **ld_provider_metadata(backend),
            "association_source": "GWAS Catalog", "association_data_release": None,
            "association_version_status": "not_exposed",
            "selected_variant_ids": source_ids, "population": population.code,
            "population_explicit": population.explicit,
            "population_assumption": population.assumption,
            "reported_study_population": population.reported_study_population,
            "reported_study_population_explicit": population.reported_study_population_explicit,
            "reported_study_population_assumption": population.reported_study_population_assumption,
            "genome_build": genome_build,
            "reference_panel": parameter("novelty.reference_panel"),
            "window_bp": parameter("novelty.ld_window_bp"),
            "scope": "GWAS Catalog exact-phenotype reports for the selected SNPs and their reference-population LD proxies within the configured window",
            "interpretation_limit": "Database coverage and the LD rule do not prove causal independence or worldwide novelty."}
    if (not population.code or not phenotype.strip() or not source_ids
            or population.reported_study_population_explicit and population.reported_study_population is None):
        return {**base, **classify_ld_screen(pairs=[], coverage_complete=False,
                identity_complete=False, threshold=parameter("novelty.ld_r2_threshold"),
                prior_variant_count=None), "required_pairs": None, "issues": [
                    population.reported_study_population_assumption if population.reported_study_population_explicit and population.reported_study_population is None
                    else population.assumption or "Missing phenotype or selection."]}
    if genome_build != parameter("novelty.reference_build"):
        return {**base, **classify_ld_screen(pairs=[], coverage_complete=False,
                identity_complete=False, threshold=parameter("novelty.ld_r2_threshold"),
                prior_variant_count=None), "required_pairs": None, "issues": ["Reference-build mismatch; no implicit coordinate conversion."]}
    native = native_result or {}
    resolution = native.get("variant_resolution") or {}
    native_matches, native_binding_issues = _native_prior_candidates(
        native, source_ids, base["queried_trait"], all_phenotypes=require_model_phenotype)
    native_relevance_complete = True
    phenotype_decisions = None
    provider = backend
    native_scope_checks = []
    native_scope_issues = ([
        "A native record related to the selected SNP has conflicting "
        "variant, study, role, or publication fields. That record cannot establish a "
        "prior report, and an empty Catalog result cannot establish novelty while "
        "this coverage remains unresolved."
    ] if native_binding_issues else [])

    def configuration_limited_screen():
        issue = ld_configuration_issue(scope="novelty_ld")
        return {**base, **classify_ld_screen(pairs=[], coverage_complete=False, identity_complete=False,
                    threshold=parameter("novelty.ld_r2_threshold"), prior_variant_count=None),
                "pairs": [], "required_pairs": None, "required_pair_count": None,
                "query_ledger": [], "ld_lookup_performed": False,
                "configuration_issues": [issue], "operation_issues": [issue],
                "issues": [*native_scope_issues, issue["message"]],
                "native_report_scope_checks": native_scope_checks,
                "native_report_binding_issues": native_binding_issues,
                "study_lookup_status": (native.get("retrieval_coverage") or {}).get("module_statuses", {}).get("locus.study", "not_available")}

    checked = {}

    def scoped_native_matches(matches):
        nonlocal provider
        if not matches or not population.reported_study_population_explicit:
            return matches
        provider = provider if provider is not None else create_ld_provider(progress=progress)
        base.update(ld_provider_metadata(provider))
        accepted = []
        for row in matches:
            study = str(row.get("study_id") or "")
            if study not in checked:
                groups = set()
                try:
                    if study:
                        groups, _ = provider.study_populations(study)
                        groups = set(groups).intersection({"EUR", "EAS", "SAS", "AFR", "AMR"})
                except (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError):
                    groups = set()
                checked[study] = groups
                native_scope_checks.append({"study_id": study, "populations": sorted(groups),
                    "status": "matched" if population.reported_study_population in groups
                    else "outside_scope" if groups else "unresolved"})
                if not groups:
                    native_scope_issues.append(
                        "A published prior report for an exact selected lead SNP has "
                        "unresolved study-population metadata. Novelty in the requested population "
                        "cannot be established from an empty Catalog result."
                    )
            if population.reported_study_population in checked[study]:
                accepted.append(row)
                # A verified positive settles the question; do not issue extra
                # ancestry requests merely to enumerate more positive reports.
                break
        return accepted

    native_candidates = native_matches
    explicit_matches = [row for row in native_candidates
                        if _reported_trait_key(row["trait"]) == _reported_trait_key(base["queried_trait"])]
    try:
        native_matches = scoped_native_matches(explicit_matches)
    except LDProviderConfigurationError:
        return configuration_limited_screen()
    if require_model_phenotype:
        from .phenotype_relevance import assess_phenotype_relevance, accepted_phenotype_records, PhenotypeDecisionCache
        phenotype_decisions = PhenotypeDecisionCache()
        base["phenotype_relevance"] = {}
        base["phenotype_matching_method"] = "explicit_reported_phenotype"
        if not native_matches:
            if progress:
                progress("Reviewing retrieved GWAS records for additional phenotype matches...")
            assessment = assess_phenotype_relevance(phenotype, native_candidates,
                complete=phenotype_completer, model_id=phenotype_model_id, cancel_event=cancel_event,
                decision_cache=phenotype_decisions,
                stop_on_relevant=not population.reported_study_population_explicit)
            semantic_matches, native_relevance_complete = accepted_phenotype_records(phenotype, native_candidates, assessment)
            base["phenotype_relevance"]["native"] = assessment
            base["phenotype_matching_method"] = "model_assessed_retrieved_labels"
            try:
                native_matches = scoped_native_matches(semantic_matches)
            except LDProviderConfigurationError:
                return configuration_limited_screen()
            if (not native_matches and not native_relevance_complete
                    and assessment.get("reason") != "phenotype_relevance_not_exhaustive"):
                return {**base, **classify_ld_screen(pairs=[], coverage_complete=False,
                    identity_complete=resolution.get("complete") is True,
                    threshold=parameter("novelty.ld_r2_threshold"), prior_variant_count=None),
                    "required_pairs": None, "query_ledger": list(getattr(provider, "ledger", [])), "ld_lookup_performed": False,
                    "lookup_performed": False, "scope": "Retrieved native GWAS records; phenotype relevance remains unresolved",
                    "native_report_binding_issues": native_binding_issues,
                    "issues": ["Phenotype relevance could not be completely assessed from the retrieved records. "
                               "Prior-report status remains unresolved; an incomplete assessment cannot establish novelty."]}
    if native_matches:
        exact_ids = list(dict.fromkeys(row["variant_id"] for row in native_matches))
        decision = classify_ld_screen(pairs=[], coverage_complete=False,
            identity_complete=resolution.get("complete") is True,
            threshold=parameter("novelty.ld_r2_threshold"), prior_variant_count=len(exact_ids),
            exact_matches=exact_ids)
        return {**base, **decision, "native_exact_report_matches": native_matches,
                "association_source": "Open Targets", "association_data_release": native.get("data_release"),
                "association_version_status": "recorded" if native.get("data_release") not in {None, "", "live"} else "not_exposed",
                "scope": ("Published GWAS evidence with model-assessed phenotype relevance and an exact selected lead-SNP match in Open Targets"
                          if base.get("phenotype_matching_method") == "model_assessed_retrieved_labels" else "Published same-phenotype GWAS evidence with an exact selected lead-SNP match in Open Targets"),
                "basis": ("A selected SNP is explicitly the lead SNP of a retrieved published GWAS whose recorded phenotype was assessed as relevant to the query."
                          if base.get("phenotype_matching_method") == "model_assessed_retrieved_labels" else "A selected SNP is explicitly the lead SNP of a retrieved published same-phenotype GWAS."),
                "matched_phenotypes": list(dict.fromkeys(row["trait"] for row in native_matches)),
                "native_report_scope_checks": native_scope_checks,
                "native_report_binding_issues": native_binding_issues,
                "query_ledger": list(getattr(provider, "ledger", [])),
                "issues": list(dict.fromkeys(native_scope_issues)),
                "prior_associations": [{"rsid": None, "associations": native_matches}]}
    try:
        provider = provider if provider is not None else create_ld_provider(progress=progress)
    except LDProviderConfigurationError:
        return configuration_limited_screen()
    base.update(ld_provider_metadata(provider))
    try:
        details = provider.screen(source_ids, base["queried_trait"], population)
        if (not isinstance(details, Mapping) or not isinstance(details.get("pairs", []), list)
            or any(not isinstance(row, Mapping) for row in details.get("pairs", []))
            or not isinstance(details.get("exact_matches", []), list)):
            raise ValueError("The LD screening service returned an invalid result shape.")
        count = details.get("prior_variant_count")
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
            raise ValueError("The prior-report count is not a verified non-negative integer.")
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError) as exc:
        details = {"pairs": [], "coverage_complete": False, "identity_complete": False,
                   "prior_variant_count": None, "required_pair_count": None,
                   "issues": [str(exc) if isinstance(exc, RuntimeError) else f"LD screening failed: {type(exc).__name__}."],
                   "query_ledger": list(getattr(provider, "ledger", []))}
    details = _review_prior_ld_coverage(provider, details, source_ids, population.code)
    if require_model_phenotype:
        from .phenotype_relevance import assess_phenotype_relevance, accepted_phenotype_records
        from .publications import normalize_pmid
        catalog_records = []
        invalid_catalog_record = False
        for group in details.get("prior_associations", ()):
            for row in group.get("associations", ()) if isinstance(group, Mapping) else ():
                if not isinstance(row, Mapping) or not normalize_pmid(row.get("pmid")) or not row.get("study_id"):
                    invalid_catalog_record = True
                    continue
                if row.get("rsid") and group.get("rsid") and row["rsid"] != group["rsid"]:
                    invalid_catalog_record = True
                    continue
                label = row.get("trait") or row.get("traitFromSource")
                catalog_records.append({**row, "rsid": row.get("rsid") or group.get("rsid"),
                    "trait": str(label or base["queried_trait"]),
                    "phenotype_label_kind": "reported_trait" if label else "catalog_query_scope",
                    "phenotype_id": row.get("phenotype_id") or details.get("phenotype_id")})
        selected_rsids = {row.get("source_variant_id"): row.get("rsid")
                          for row in details.get("resolved_selection", ())
                          if isinstance(row, Mapping) and row.get("source_variant_id") in source_ids}

        def retained_connections(records):
            rsids = {row["rsid"] for row in records if row.get("rsid")}
            return (rsids,
                [row for row in details.get("pairs", ()) if row.get("reported_rsid") in rsids
                 and row.get("selected_variant_id") in selected_rsids and row.get("selected_rsid")
                 and selected_rsids[row["selected_variant_id"]] == row["selected_rsid"]],
                [identifier for identifier in details.get("exact_matches", ()) if selected_rsids.get(identifier) in rsids])

        accepted = [row for row in catalog_records if row["phenotype_label_kind"] == "reported_trait"
                    and _reported_trait_key(row["trait"]) == _reported_trait_key(base["queried_trait"])]
        accepted_rsids, retained_pairs, retained_exact = retained_connections(accepted)
        explicit_positive = classify_ld_screen(pairs=retained_pairs, exact_matches=retained_exact,
            coverage_complete=False, identity_complete=False, prior_variant_count=None,
            threshold=parameter("novelty.ld_r2_threshold"))["status"] == "PREVIOUSLY_REPORTED"
        catalog_complete = False
        if explicit_positive:
            base["phenotype_matching_method"] = "explicit_reported_phenotype"
        else:
            assessment = assess_phenotype_relevance(phenotype, catalog_records,
                complete=phenotype_completer, model_id=phenotype_model_id, cancel_event=cancel_event,
                decision_cache=phenotype_decisions)
            accepted, catalog_complete = accepted_phenotype_records(phenotype, catalog_records, assessment)
            base["phenotype_relevance"]["catalog"] = assessment
            accepted_rsids, retained_pairs, retained_exact = retained_connections(accepted)
        positive_without_records = bool((details.get("exact_matches") or details.get("pairs")
                                        or details.get("prior_variant_count")) and not catalog_records)
        semantic_complete = (native_relevance_complete and catalog_complete
                             and not invalid_catalog_record and not positive_without_records)
        details = {**details,
            "retrieved_prior_associations": details.get("prior_associations", []),
            "prior_associations": [{"rsid": rsid, "associations": [row for row in accepted if row.get("rsid") == rsid]}
                                   for rsid in sorted(accepted_rsids)],
            "pairs": retained_pairs, "exact_matches": retained_exact,
            "required_pair_count": len(retained_pairs),
            "prior_variant_count": len(accepted_rsids) if semantic_complete else None,
            "coverage_complete": details.get("coverage_complete") is True and semantic_complete,
            "issues": list(details.get("issues", ())) + ([] if semantic_complete or explicit_positive else [
                "Phenotype relevance or its recorded association binding is incomplete; novelty remains unresolved."])}
    if native_scope_issues:
        details = {**details, "coverage_complete": False, "prior_variant_count": None,
                   "issues": list(dict.fromkeys([*details.get("issues", []), *native_scope_issues]))}
    decision = classify_ld_screen(
        pairs=details.get("pairs", []), coverage_complete=details.get("coverage_complete") is True,
        identity_complete=details.get("identity_complete") is True,
        threshold=parameter("novelty.ld_r2_threshold"),
        prior_variant_count=details.get("prior_variant_count"),
        exact_matches=details.get("exact_matches", []),
    )
    return {**base, **details, **decision,
            "native_report_scope_checks": native_scope_checks,
            "native_report_binding_issues": native_binding_issues,
            "required_pairs": details.get("required_pair_count", decision["required_pairs"]),
            "study_lookup_status": (native.get("retrieval_coverage") or {}).get("module_statuses", {}).get("locus.study", "not_available")}


def novelty_summary(screen: Mapping[str, Any]) -> str:
    """Readable authoritative text, with missing values rendered as missing."""
    if screen.get("execution_status") == "not_run" and screen.get("precondition_reason"):
        return "Novelty remains unresolved. " + " ".join(screen.get("issues", []))
    if screen.get("source_only_processing_limit") is True:
        return ("Novelty remains unresolved. The retained external result exceeded the "
                "interactive processing limit and was not reviewed. No additional prior-report "
                "search was run for this response; neither a prior association nor its absence "
                "can be concluded from this processing limit.")
    if (screen.get("phenotype_matching_method") == "model_assessed_retrieved_labels"
            and screen.get("lookup_performed") is False and not screen.get("native_exact_report_matches")):
        return ("Novelty remains unresolved. " + " ".join(screen.get("issues", []))
                + " No Catalog or LD prior-report query was run for this incomplete assessment.")
    title = ("No prior association found in the queried database version and search scope"
             if screen.get("status") == "NOVEL_IN_SCOPE" else str(screen.get("label", "Novelty unresolved")))
    if screen.get("native_exact_report_matches"):
        scope = (
            f"The report matches the requested {screen['reported_study_population']} study-population scope. "
            if screen.get("reported_study_population_explicit") else
            "Prior reports were considered across study populations. "
        )
        semantic = screen.get("phenotype_matching_method") == "model_assessed_retrieved_labels"
        matched_labels = list(dict.fromkeys(str(row.get("trait")) for row in screen["native_exact_report_matches"]
                                           if row.get("trait")))
        relevance = (f"GWAS whose recorded phenotype was assessed by the model as relevant to {screen.get('phenotype')}. "
            + "Matched reported phenotype labels: " + "; ".join(matched_labels) + ". "
            if semantic else f"same-phenotype GWAS for {screen.get('phenotype')}. ")
        return (f"{title}. A selected SNP is explicitly the lead SNP of a retrieved, published "
                + relevance + scope
                + "This exact lead-SNP evidence establishes a prior report without requiring an additional LD proxy search. "
                + " ".join(str(item) for item in screen.get("issues", []))
                + " " + str(screen.get("interpretation_limit", "")))
    summary = (f"{title}. LD reference: {screen.get('population') or 'unresolved population'}, "
               f"{screen.get('reference_panel')}, {screen.get('genome_build')}. "
               f"Prior-report search set: selected SNPs and LD variants with r\u00b2 >= {screen.get('r2_threshold')}. "
               f"Search: {screen.get('queried_trait')}, within {screen.get('window_bp'):,} bp "
               f"of each selected variant. ")
    if screen.get("association_source"):
        summary += f"Association source: {screen['association_source']}. "
        if screen.get("association_version_status") == "recorded" and screen.get("association_data_release"):
            summary += f"Database release: {screen['association_data_release']}. "
        elif screen.get("association_version_status") == "not_exposed":
            summary += "The association API did not expose a version identifier; this finding is limited to the recorded query. "
    if screen.get("reported_study_population_explicit"):
        summary += f"Prior-study population scope: {screen.get('reported_study_population') or 'unresolved'}. "
    elif "reported_study_population_explicit" in screen:
        summary += "Prior-study population scope: all study populations; the LD reference does not restrict reported-study ancestry. "
    if screen.get('required_pairs') is None:
        summary += "LD comparison coverage could not be established."
    elif screen.get("algorithm") != "ld_proxies_then_reports" and not screen.get("exact_matches"):
        summary += (f"{screen.get('computed_pairs', 0)} of {screen['required_pairs']} "
                    "required LD pairs computed.")
    if screen.get("max_r2") is not None:
        summary += f" Maximum observed r\u00b2: {screen['max_r2']:.6g}."
    if screen.get("algorithm") == "ld_proxies_then_reports":
        if screen.get('queried_snp_count') is not None:
            summary += (f" Prior-report lookup covered {screen['queried_snp_count']} selected and LD-related variants; "
                        f"{screen.get('ld_proxy_count', 0)} LD proxy pairs were returned.")
        else:
            summary += " The queried LD variant set could not be fully established."
    if screen.get("exact_matches"):
        summary += " The selected SNP itself appears in matching prior associations."
    if screen.get("prior_variant_count") == 0 and screen.get("status") == "NOVEL_IN_SCOPE":
        summary += " No matching prior association was retrieved for the selected and LD-related variant set."
    if screen.get("population_assumption"):
        summary += " " + str(screen["population_assumption"])
    if screen.get("issues"):
        summary += " " + " ".join(str(x) for x in screen["issues"])
    if screen.get("phenotype_matching_method") == "model_assessed_retrieved_labels":
        summary += (" Phenotype relevance was assessed for the retrieved records. The Catalog finding is limited "
                    "to its recorded exact-phenotype/EFO query; it does not establish that every related, subtype, "
                    "or composite phenotype was searched.")
    return summary + " " + str(screen.get("interpretation_limit", ""))


def novelty_resource_references(reference_provider):
    """Cite resource publications separately from individual association studies."""
    from .staged_reasoning import FinalReference

    ld_reference = (
        "Machiela MJ, Chanock SJ. LDlink: a web-based application for exploring population-specific "
        "haplotype structure and linking correlated alleles of possible functional variants. "
        "Bioinformatics. 2015;31(21):3555-3557. "
        "[PMID: 26139635](https://pubmed.ncbi.nlm.nih.gov/26139635/). "
        "[DOI: 10.1093/bioinformatics/btv402](https://doi.org/10.1093/bioinformatics/btv402)."
        if reference_provider == "ldlink" else
        "Ensembl REST, population-specific LD proxy lookup. https://rest.ensembl.org/documentation/info/ld_id_get"
    )
    return (
        FinalReference("ld_reference", ld_reference),
        FinalReference("prior_catalog", "Cerezo M et al. The NHGRI-EBI GWAS Catalog: standards for "
                       "reusability, sustainability and diversity. Nucleic Acids Research. "
                       "2025;53(D1):D998-D1005. "
                       "[PMID: 39530240](https://pubmed.ncbi.nlm.nih.gov/39530240/). "
                       "[DOI: 10.1093/nar/gkae1070](https://doi.org/10.1093/nar/gkae1070)."),
    )


def add_novelty_context(context):
    """Add the recorded prior-association facts to the model's evidence registry."""
    from dataclasses import replace

    from .staged_reasoning import FinalEvidenceFact, FinalSection
    from .novelty_references import prior_report_references
    from .ld_support import ld_provider_metadata
    screen = context.result.get("novelty_screen")
    if not isinstance(screen, Mapping):
        return context
    statement = novelty_summary(screen)
    provider_metadata = ld_provider_metadata(screen)
    refs = novelty_resource_references(provider_metadata.get("reference_provider", "ensembl"))
    if screen.get("source_only_processing_limit") is True or screen.get("execution_status") == "not_run":
        refs = ()
    if screen.get("native_exact_report_matches"):
        refs = ()
    if screen.get("lookup_performed") is False:
        refs = ()
    if screen.get("source_only_processing_limit") is True or screen.get("execution_status") == "not_run":
        reference_ids = ()
    else:
        context, reference_ids = prior_report_references(context, screen, refs)
    fact = FinalEvidenceFact("novelty_ld_result", "novelty_ld", (), statement,
                            reference_ids,
                            {"status": screen["status"], "supported_statement": statement,
                             "population": screen.get("population"),
                             "reported_study_population": screen.get("reported_study_population"),
                             "reported_study_population_explicit": screen.get("reported_study_population_explicit", False),
                             "association_source": screen.get("association_source"),
                             "association_data_release": screen.get("association_data_release"),
                             "association_version_status": screen.get("association_version_status"),
                             "phenotype_matching_method": screen.get("phenotype_matching_method"),
                             "matched_phenotypes": screen.get("matched_phenotypes", []),
                             "r2_threshold": screen.get("r2_threshold"),
                             "max_r2": screen.get("max_r2"), **provider_metadata}, True)
    return replace(context, sections=(*context.sections, FinalSection("novelty_ld", "Prior reports and LD")),
                   facts=(*context.facts, fact),
                   fallback_summary=context.fallback_summary + " " + statement)


def novelty_context_for_report(context):
    """Keep prior-report analysis separate from a gene-centric prioritization report.

    Explicit novelty and legacy report tasks keep their existing presentation.
    The internal novelty result is retained in every case; only its separate
    V2G report section and unreferenced resource citations are omitted.
    """
    from dataclasses import replace

    audit = context.result.get("candidate_audit") or {}
    gene_centric = isinstance(audit, Mapping) and audit.get("mode") == "gene_centric_window"
    if context.intent != "prioritize_genes" or not gene_centric:
        return add_novelty_context(context)
    removed = tuple(fact for fact in context.facts if fact.section_id == "novelty_ld")
    facts = tuple(fact for fact in context.facts if fact.section_id != "novelty_ld")
    used = {reference for fact in facts for reference in fact.reference_ids}
    summary = context.fallback_summary
    for fact in reversed(removed):
        suffix = " " + fact.markdown
        if summary.endswith(suffix):
            summary = summary[:-len(suffix)]
    return replace(context, sections=tuple(section for section in context.sections
                                           if section.section_id != "novelty_ld"),
                   facts=facts, fallback_summary=summary,
                   references=tuple(reference for reference in context.references
                                    if reference.reference_id not in {"ld_reference", "prior_catalog"}
                                    or reference.reference_id in used))


def novelty_coverage_check(facts, screen):
    """An unresolved novelty task is one required-retrieval warning, not a new gate."""
    if not isinstance(screen, Mapping) or screen.get("status") != "UNRESOLVED":
        return facts
    from dataclasses import replace

    from .staged_reasoning import SafetyOutcome, SafetySeverity, configured_safety_check
    if any(check.check_id == "novelty_coverage" for check in facts.checks):
        return facts
    check = configured_safety_check(check_id="novelty_coverage", label="Prior-report and LD coverage",
                                    severity=SafetySeverity.WARNING, outcome=SafetyOutcome.WARN,
                                    fact="Prior-report or LD coverage is incomplete; novelty remains unresolved. "
                                    "Available evidence can still be interpreted within its supported scope.")
    return replace(facts, checks=(*facts.checks, check))


_AFFIRMATIVE_NOVELTY = re.compile(
        r"\b(?:is|are|represents?|constitutes?|confirms?|establishes?|identified as|classified as|considered)\s+"
        r"(?:(?:a|an|the|truly|entirely|potentially|likely|completely)\s+)*"
        r"(?:novel|new|previously unreported)\s+(?:locus|loci|signals?|associations?)\b|"
        r"(?:\u8fd9\u662f|\u786e\u8ba4\u4e3a|\u786e\u5b9a\u4e3a|\u5c5e\u4e8e|\u662f\u4e00\u4e2a|\u662f\u65b0\u7684)(?:\u65b0\u4f4d\u70b9|\u65b0\u7684\u4f4d\u70b9|\u65b0\u7684\u5173\u8054)", re.I)


def _affirmative_novelty_claims(text):
    for match in _AFFIRMATIVE_NOVELTY.finditer(text):
        prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
        if not re.search(r"\b(?:if|whether|suppose|hypothetical|incorrect|not|cannot)\b|\u4e0d\u80fd|\u4e0d\u53ef|\u5047\u5982|\u662f\u5426|\u5e76\u975e", prefix, re.I):
            yield match


def validate_novelty_claims(text: str, screen: Mapping[str, Any]) -> None:
    """An exhaustive in-scope negative still cannot establish global novelty."""
    if next(_affirmative_novelty_claims(text), None) is not None:
        raise ValueError("The novelty claim contradicts the bounded prior-association result: a database screen does not establish global novelty.")


def novelty_claim_corrections(text, context):
    """Replace only affirmative novelty predicates; retain ranking and narrative."""
    screen = context.result.get("novelty_screen")
    if not isinstance(screen, Mapping) or screen.get("status") not in {"NOVEL_IN_SCOPE", "UNRESOLVED", "PREVIOUSLY_REPORTED"}:
        return []
    edits = []
    for match in _affirmative_novelty_claims(text):
        if screen["status"] == "NOVEL_IN_SCOPE":
            replacement = "has no prior association found in the queried database version and search scope"
        elif screen["status"] == "PREVIOUSLY_REPORTED":
            replacement = ("has a prior association with a phenotype assessed as relevant to the query, or a verified LD link to one"
                           if screen.get("phenotype_matching_method") == "model_assessed_retrieved_labels"
                           else "has a prior same-phenotype association or a verified LD link to one in the queried scope")
        else:
            replacement = "has unresolved prior-association status in the queried scope"
        if re.match(r"are\b|(?:represent|constitute|confirm|establish)\b", match.group(), re.I):
            replacement = replacement.replace("has ", "have ", 1)
        edits.append({"start": match.start(), "end": match.end(), "replacement": replacement,
            "metric": "prior_association_scope", "source_type": "prior_association_receipt",
            "source_id": "novelty_ld_result", "screen_status": screen["status"],
            "reason": "bounded_database_screen_cannot_establish_global_novelty"})
    return edits
