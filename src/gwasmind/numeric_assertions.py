"""Semantic categories for numeric assertions, independent of their typography."""
import re
from .numeric_provenance import finite_decimal


def is_selection_threshold(sentence, match, metric):
    if metric not in {"h3", "h4", "p", "pip", "clpp"} or match["relation"] not in {"<", "<=", "\u2264", ">", ">=", "\u2265"}:
        return False
    before, after = sentence[:match.start()], sentence[match.end():]
    if re.search(r"\b(?:inclusion|retrieval|filtering|selection)\s+(?:threshold|cutoff)\b[^.!?]{0,40}$", before, re.I):
        return True
    # "H4 >= .9 pairs" selects a subset. "This pair has H4 >= .9" asserts
    # a measured bound and must continue through source-owned verification.
    if re.match(r"\s+(?:pairs|records|signals|colocali[sz]ations)\b", after, re.I):
        return True
    if re.search(r"\b(?:every|any|each)\s*$", before, re.I) and re.match(r"\s+(?:pair|record|signal)\b", after, re.I):
        return True
    return bool(re.search(r"\bNo\s*$", before, re.I)
                and re.match(r"\s+colocali[sz]ation\b", after, re.I))


def finding_category(reason, *, disposition="unverified", candidate_count=None):
    if reason in {"input_precision_normalized", "source_precision_normalized"}:
        return "representation_normalization"
    if disposition == "corrected" or reason in {"ld_evidence_conflict", "numeric_value_mismatch", "numeric_domain_error", "configured_threshold_mismatch"}:
        return "numeric_error"
    if reason in {"ld_binding_unclear", "measurement_not_uniquely_bound", "measurement_relation_ambiguous"}:
        return "relation_ambiguity"
    if candidate_count is not None and candidate_count > 1:
        return "relation_ambiguity"
    return "not_verifiable"


def selection_threshold_assessment(sentence, match, metric, ledger):
    if not is_selection_threshold(sentence, match, metric):
        return None
    value = finite_decimal(match["number"])
    if match.groupdict().get("percent") and value is not None:
        value /= 100
    facts = [fact for fact in ledger["facts"] if fact["metric"] == metric and fact["scope"] == "inclusion_threshold"]
    result = {"assertion_type": "selection_threshold", "operator": match["relation"],
              "numeric_fact_ids": [fact["fact_id"] for fact in facts], "status": "not_a_point_measurement"}
    if value is None or not 0 <= value <= 1:
        return {**result, "status": "numeric_error", "reason": "numeric_domain_error"}
    explicit_policy = re.search(r"\b(?:inclusion|retrieval|filtering|selection)\s+(?:threshold|cutoff)\b[^.!?]{0,40}$", sentence[:match.start()], re.I)
    if explicit_policy:
        if not facts:
            return {**result, "status": "not_verifiable", "reason": "threshold_source_unavailable"}
        expected = {finite_decimal(fact["value"]) for fact in facts}
        operator = match['relation'].replace('\u2265', '>=').replace('\u2264', '<=')
        if expected != {value} or {fact['provenance']['operator'] for fact in facts} != {operator}:
            return {**result, "status": "numeric_error" if len(expected) == 1 else "relation_ambiguity",
                    "reason": "configured_threshold_mismatch" if len(expected) == 1 else "measurement_relation_ambiguous"}
        result["status"] = "verified"
    return result
