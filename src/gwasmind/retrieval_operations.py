"""Operational retrieval stops require positive failure receipts, never empty data."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .academic_review import operation_issue


def retrieval_operation_issues(receipts: Sequence[Mapping], *, has_usable_result: bool) -> tuple[dict, ...]:
    issues = [dict(row) for row in receipts]
    for issue in issues:
        if issue.get("code") == "S1":
            issue.update(operation_issue("S1", category="input_invalid", scope="input", blocking=True,
                message="No source variant could be assigned a usable identity for external analysis.",
                reasons=(str(issue.get("reason") or "no_valid_source_variant"),)))
    exhausted = [row for row in receipts if row.get("code") == "required_query_failed"
        and row.get("state") == "FAILED" and isinstance(row.get("details"), Mapping)
        and row["details"].get("recovery_exhausted") is True
        and row["details"].get("category") in {
            "request_timeout", "connection_error", "transport_error", "rate_limited", "service_error"
        }]
    if exhausted and not has_usable_result and not any(row.get("code") == "S2" for row in issues):
        issues.append({**operation_issue("S2", category="service_unavailable", scope="retrieval", blocking=True,
            message="Required Open Targets queries remained unavailable after bounded recovery; source observations are retained.",
            reasons=tuple(sorted({row["details"]["category"] for row in exhausted}))),
            "failed_query_count": len(exhausted),
            "request_sha256": sorted({row["request_sha256"] for row in exhausted if isinstance(row.get("request_sha256"), str)}),
        })
    return tuple(issues)
