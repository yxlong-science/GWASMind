"""Expose final-answer fact findings without changing producer audit receipts."""
from collections.abc import Mapping
from copy import deepcopy

from ..fact_review_issues import combine_fact_issues, visible_report_fact_issues


def public_ranking_review(review):
    """Keep the ranking receipt separate from final-report certification."""
    if not isinstance(review, Mapping) or review.get("target") == "final_report":
        return None
    state = review.get("status")
    if state not in {"needs_review", "partially_verified", "verified", "PASS", "PASS_WITH_WARNINGS"}:
        return None
    return {"status": state, "fact_issues": combine_fact_issues(review.get("fact_issues", {}))}

_REPLACED_MEASUREMENT_WARNING = (
    "Some reported measurements could not be verified and are labelled in place. "
    "The model's interpretation and verified evidence are retained."
)


def combined_output_review(report_review: Mapping, ranking_review: Mapping | None = None, *,
                           visible_output=None, rendered_markdown: str | None = None) -> dict:
    """Review only facts asserted in the answer the user receives.

    The ranking argument remains accepted for older callers, but its audit is
    intentionally not merged into the final-answer findings.
    """
    report = deepcopy(dict(report_review))
    original_issues = report.get("fact_issues", {})
    issues = visible_report_fact_issues(original_issues, visible_output=visible_output,
                                       rendered_markdown=rendered_markdown)
    report["fact_issues"] = issues
    report["scope"] = "final_visible_answer"
    if issues["unresolved_count"]:
        # Unresolved items stay in the audit; Gate 2 never reports a warning.
        report["fully_verified"] = False
    elif (rendered_markdown is not None or visible_output is not None) and isinstance(original_issues, Mapping) and original_issues.get("schema_version") == "1":
        warnings = [warning for warning in report.get("warnings", ())
                    if warning != _REPLACED_MEASUREMENT_WARNING]
        report["warnings"] = warnings
        complete = not any((issues["omitted_count"], issues["unexamined_count"], issues["rejected_count"],
                            report.get("not_assessed"), warnings,
                            report.get("generated_draft_validated") is False,
                            report.get("method") in {"partial_recovery", "evidence_only_recovery"}))
        if complete and report.get("status") == "PASS_WITH_WARNINGS":
            report["status"] = "PASS"
            report["fully_verified"] = True
    return report


def visible_writer_warning_messages(warnings, output_review: Mapping) -> tuple[str, ...]:
    """Keep genuine output limits while dropping notices for replaced values."""
    issues = output_review.get("fact_issues", {})
    resolved = isinstance(issues, Mapping) and issues.get("visibility_checked") and not issues.get("unresolved_count")
    return tuple(warning for warning in warnings
                 if not (resolved and warning == _REPLACED_MEASUREMENT_WARNING))


def visible_report_quality(gate1_status: str, generated_status: str, output_review: Mapping) -> str:
    """Report quality is the Gate 1 evidence status; Gate 2 never downgrades it.

    Gate 2 edits keep the Gate 1 status, so a stricter generated status only
    reflects a writer that failed to deliver an interpretation.
    """
    if "STOPPED" in {gate1_status, generated_status}:
        return "STOPPED"
    if "PASS_WITH_WARNINGS" in {gate1_status, generated_status}:
        return "PASS_WITH_WARNINGS"
    if gate1_status == "PASS":
        return "PASS"
    return generated_status
