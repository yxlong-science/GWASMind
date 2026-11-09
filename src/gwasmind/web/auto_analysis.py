from __future__ import annotations

import sqlite3
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic as _monotonic
from typing import Any

from ..configuration import parameter
from ..novelty import novelty_summary
from ..prompting import format_prompt
from ..report_facts import public_source_normalizations
from .analysis_bridge import AnalysisBridge, project_results
from .chat_intent import ChatIntent
from .chat_response import (
    _evidence_rows,
    _is_trait_study,
    _phenotypes_equivalent,
    _study_associations,
)
from .history_store import (
    HistoryError,
    _validate_ranking_evidence,
    _validate_variant_normalizations,
)
from .jobs import JobConflictError, JobError, JobNotFoundError, JobValidationError
from .stage_bridge import variant_identity_context

_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_MAX_TASKS = parameter("execution.auto_max_tasks")
_MAX_EVENTS = parameter("execution.auto_max_events")
DEFAULT_LOCUS_CONCURRENCY = parameter("execution.auto_default_concurrent_loci")
MAX_LOCUS_CONCURRENCY = parameter("execution.max_concurrent_runs")
_STAGES = ("plan", "retrieve", "safety", "response")
_WORKFLOWS = ("screening",)
_ANALYSIS_MODE = "prior_report_screening"
_MAX_STAGE_DETAILS = 24
_MAX_STAGE_SUBSTEPS = 128
_PROGRESS_METRICS = frozenset({
    "variant_annotations", "credible_sets", "variant_consequences", "molecular_qtl",
    "colocalization", "enhancer_links", "l2g_predictions", "candidate_genes",
    "gene_expression", "gene_function", "pathways", "publications", "publication_abstracts",
    "total_records", "elapsed_seconds", "output_tokens", "total_tokens", "received_chars",
    "ranking_elapsed_seconds", "stage_elapsed_seconds", "estimated_output_tokens",
    "source_variant_count", "verified_proxy_count", "analysis_variant_count",
    "candidate_pool_count", "shortlist_count", "ranked_count", "displayed_ranked_count",
})
_RANKING_EVIDENCE_FIELDS = frozenset({
    "run_id", "query_index", "snapshot_id", "model_input_sha256", "messages_sha256", "candidate_count",
})


def _public_ranking_evidence(value: Any) -> dict[str, Any] | None:
    """Reuse the public snapshot descriptor contract; never forward arbitrary URLs."""
    if not isinstance(value, Mapping):
        return None
    descriptor = {key: value[key] for key in _RANKING_EVIDENCE_FIELDS if key in value}
    try:
        _validate_ranking_evidence([descriptor])
    except HistoryError:
        return None
    return descriptor


def _ranking_evidence_rows(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    retained = {}
    for value in values:
        descriptor = _public_ranking_evidence(value)
        if descriptor is None:
            continue
        key = (descriptor["run_id"], descriptor["query_index"], descriptor["snapshot_id"])
        if key in retained:
            continue
        try:
            _validate_ranking_evidence([*retained.values(), descriptor])
        except HistoryError:
            break
        retained[key] = descriptor
    return list(retained.values())


def _public_fields(value: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project only the already-public review contract, including nested objects."""
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key, shape in schema.items():
        item = value.get(key)
        if shape is str and isinstance(item, str) and len(item) <= 1600:
            result[key] = item
        elif shape is bool and type(item) is bool:
            result[key] = item
        elif isinstance(shape, dict) and isinstance(item, Mapping):
            result[key] = _public_fields(item, shape)
        elif isinstance(shape, tuple) and isinstance(item, (list, tuple)):
            member, limit = shape
            if len(item) > limit:
                continue
            if member is str:
                if all(isinstance(row, str) and len(row) <= 1600 for row in item):
                    result[key] = list(item)
            elif all(isinstance(row, Mapping) for row in item):
                result[key] = [_public_fields(row, member) for row in item]
    return result


def _public_workflow_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Persist no model/tool payloads; reuse the live retrieval/identity validators."""
    event_type = event.get("type")
    if event_type == "variant_normalization":
        rows = event.get("normalizations")
        try:
            _validate_variant_normalizations(rows)
        except HistoryError:
            return {}
        return {"type": event_type, "normalizations": deepcopy(rows)}
    if event_type not in {"stage", "review_update"} or event.get("stage") not in _STAGES:
        return {}
    if event.get("state") not in {"pending", "running", "passed", "warning", "stopped", "failed"}:
        return {}
    result = {"type": event_type, "stage": event["stage"], "state": event["state"]}
    for key, limit in (("summary", 240), ("detail", 500)):
        if isinstance(event.get(key), str):
            result[key] = event[key][:limit]
    if event.get("requested_task") in {intent.value for intent in ChatIntent}:
        result["requested_task"] = event["requested_task"]
    if event.get("stage") == "response" and isinstance(event.get("substage"), str) and event["substage"] in {"inference", "fact_verification"}:
        result["substage"] = event["substage"]
    if event.get("stage") == "response" and event.get("ranking_unavailable") is True:
        result["ranking_unavailable"] = True
    if event.get("stage") == "response" and isinstance(event.get("model_usage"), Mapping):
        from .model_usage import public_model_usage
        usage = public_model_usage(event["model_usage"])
        if usage:
            result["model_usage"] = usage
    if isinstance(event.get("metrics"), Mapping):
        result["metrics"] = {key: number for key, number in event["metrics"].items()
                             if key in _PROGRESS_METRICS and type(number) is int
                             and 0 <= number <= 10_000_000}
    if isinstance(event.get("progress"), Mapping):
        # Delayed import avoids the server -> manager import cycle. This is the
        # same projection used before publishing the shared Chat workflow.
        from .server import _public_retrieval_progress

        progress = _public_retrieval_progress(event["progress"])
        if progress:
            result["progress"] = progress
    if event_type == "review_update":
        result.update(_public_fields(event, {
            "warnings": (str, 32),
            "answerability": {"level": str, "reason_codes": (str, 24)},
            "academic_warnings": ({
                "code": str, "label": str, "rule_codes": (str, 8), "facts": (str, 24), "reason_codes": (str, 24),
                "check_ids": (str, 24), "impact_domains": (str, 24),
                "refs": ({"type": str, "id": str}, 32),
                "impact": str,
                "affected": {"variant_ids": (str, 32), "gene_ids": (str, 32), "study_ids": (str, 32), "modules": (str, 32)},
                "affected_count": {"variant_ids": int, "gene_ids": int, "study_ids": int, "modules": int},
            }, 32),
            "operation_issues": ({
                "code": str, "label": str, "scope": str, "message": str, "blocking": bool,
                "category": str, "reason_codes": (str, 24),
            }, 16),
            "method_notes": ({
                "code": str, "label": str, "rule_codes": (str, 8), "facts": (str, 24), "check_ids": (str, 24),
                "assessment_state": str, "reason_codes": (str, 24), "impact": str,
                "refs": ({"type": str, "id": str}, 32),
                "affected": {"variant_ids": (str, 32), "gene_ids": (str, 32), "study_ids": (str, 32), "modules": (str, 32)},
                "affected_count": {"variant_ids": int, "gene_ids": int, "study_ids": int, "modules": int},
            }, 32),
            "review_policy_version": str,
            "review_scope_sha256": str,
            "gate_reviews": {"gate1": {"status": str}, "gate2": {"status": str}},
            "gate_warning_items": ({
                "check_id": str, "gate": str, "role": str, "display_group": str,
                "caused_by": (str, 24), "messages": (str, 16),
            }, 32),
            "output_review": {"status": str, "warnings": (str, 8)},
        }))
    return result


def _substep_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    if value.get("operation_id"):
        return (value["operation_id"], value.get("phase"), value.get("channel"))
    # Batch/page/elapsed counters update one operation. They must not add a new
    # retained row on every poll or every LD proxy.
    return tuple(value.get(key) for key in (
        "phase", "name", "tool", "channel", "reference_provider", "identity_scope",
        "populations_requested", "operation",
    ))


def _progress_snapshot(batch: _Batch) -> list[dict[str, Any]]:
    return [deepcopy(batch.progress[key]) for key in sorted(
        batch.progress, key=lambda item: (item[0], _WORKFLOWS.index(item[1])),
    )]


class _AnalysisCancelled(Exception):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(slots=True)
class _RunTiming:
    started_at: str | None = None
    finished_at: str | None = None
    started_clock: float | None = None
    elapsed_seconds: float | None = None

    def start(self) -> None:
        if self.started_clock is None:
            self.started_clock = _monotonic()
            self.started_at = _utc_now()

    def finish(self) -> None:
        if self.started_clock is not None and self.finished_at is None:
            self.elapsed_seconds = round(max(0.0, _monotonic() - self.started_clock), 3)
            self.finished_at = _utc_now()

    def snapshot(self) -> dict[str, Any]:
        elapsed = self.elapsed_seconds
        if self.started_clock is not None and self.finished_at is None:
            elapsed = round(max(0.0, _monotonic() - self.started_clock), 3)
        return {"started_at": self.started_at, "finished_at": self.finished_at,
                "elapsed_seconds": elapsed,
                "observed_at": (self.finished_at or _utc_now()) if self.started_clock is not None else None,
                "timing_basis": "backend_monotonic"}


@dataclass(slots=True)
class _Batch:
    task_id: str
    phenotype: str
    queries: list[dict[str, Any]]
    concurrency: int = DEFAULT_LOCUS_CONCURRENCY
    state: str = "running"
    active_run_ids: set[str] = field(default_factory=set)
    locus_states: list[dict[str, Any]] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[tuple[int, str], dict[str, Any]] = field(default_factory=dict)
    cancelled: threading.Event = field(default_factory=threading.Event)
    sequence: int = 0
    error: str | None = None
    history_id: str | None = None
    timing: _RunTiming = field(default_factory=_RunTiming)
    locus_timers: dict[int, _RunTiming] = field(default_factory=dict)


def _locus_snapshot(batch: _Batch, index: int) -> dict[str, Any]:
    return {**batch.locus_states[index],
            "timing": batch.locus_timers.get(index, _RunTiming()).snapshot()}


def _loci_snapshot(batch: _Batch) -> list[dict[str, Any]]:
    return [_locus_snapshot(batch, index) for index in range(len(batch.locus_states))]


def finalize_auto_screen(
    screen: Mapping[str, Any], *, native_result: Mapping[str, Any],
    observations: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project Auto's evaluated scope after the unchanged shared reviews.

    Explicit identity omissions are warnings about excluded SNPs. They are not
    failed association requests, unknown LD windows, or unresolved phenotype
    judgments. This consumer performs no retrieval and never mutates a receipt.
    """
    from ..novelty import classify_ld_screen

    result = deepcopy(dict(screen))
    if result.get("auto_scope_review", {}).get("policy") == "auto-resolved-scope-v1":
        return result
    source_ids = [str(row.get("variant_id") or "") for row in observations]
    source_set = set(source_ids)
    resolution = native_result.get("variant_resolution") or {}
    omissions = resolution.get("unresolved_variants", [])
    allowed_source_reasons = {
        "no_match", "ambiguous_search_candidates", "no_search_hit_at_source_coordinate",
        "normalization_conflict", "no_validated_variant_identity",
    }
    excluded, blockers = [], []
    omitted = {}
    for row in omissions if isinstance(omissions, list) else ():
        if not isinstance(row, Mapping):
            blockers.append("Malformed source identity omission.")
            continue
        identifier, reason = row.get("variant_id"), row.get("reason")
        if identifier not in source_set or identifier in omitted or reason not in allowed_source_reasons:
            blockers.append("A source identity omission is not bound to the selected scope.")
            continue
        omitted[identifier] = reason
    selected = result.get("resolved_selection", [])
    selected = selected if isinstance(selected, list) else []
    resolved = {}
    for row in selected:
        if (not isinstance(row, Mapping) or row.get("source_variant_id") not in source_set
                or row.get("source_variant_id") in resolved
                or not re.fullmatch(r"rs[1-9]\d*", str(row.get("rsid") or ""))):
            blockers.append("The evaluated source identity set is invalid.")
            continue
        resolved[row["source_variant_id"]] = row["rsid"]
    if result.get("native_exact_report_matches") and not selected:
        resolved = {identifier: canonical for identifier, canonical in
                    (resolution.get("source_to_identified_variant") or {}).items()
                    if identifier in source_set and identifier not in omitted}
    # The provider separately records identity completeness and the actual
    # association/LD coverage of its resolved subset. Do not infer an error's
    # cause; accept an enumerated omission only after that subset is complete.
    semantic = result.get("phenotype_relevance") or {}
    request_rows = result.get("query_ledger", [])
    final_requests = {row.get("request_key"): row for row in request_rows if isinstance(row, Mapping)}
    if (result.get("identity_complete") is False and result.get("coverage_complete") is True
            and result.get("catalog_lookup_complete") is True and resolved
            and set(result.get("selected_variant_ids", [])) == source_set
            and all(isinstance(semantic.get(stage), Mapping) and semantic[stage].get("complete") is True
                    for stage in ("native", "catalog"))
            and request_rows and all(isinstance(row, Mapping) for row in request_rows)
            and all(key and row.get("status") == "success" for key, row in final_requests.items())):
        for identifier in source_set - set(resolved):
            if any(isinstance(issue, str) and issue.startswith(f"Cannot verify selected SNP {identifier}:")
                   for issue in result.get("issues", [])):
                omitted.setdefault(identifier, "ld_reference_identity_unresolved")
    for identifier, reason in omitted.items():
        if identifier not in resolved:
            excluded.append({"scope": "source", "variant_id": identifier, "reason": reason})

    audits = result.get("ld_lookup_audit", [])
    audits = audits if isinstance(audits, list) else []
    audit_sources, reviewed_audits = set(), []
    ledger = result.get("query_ledger", [])
    ledger = ledger if isinstance(ledger, list) else []
    for audit in audits:
        if not isinstance(audit, Mapping):
            blockers.append("Malformed LD retrieval receipt.")
            continue
        source = audit.get("source_variant_id")
        valid = source in resolved and source not in audit_sources
        audit_sources.add(source)
        rows = audit.get("excluded_rows", [])
        valid = valid and isinstance(rows, list) and all(
            isinstance(row, Mapping) and type(row.get("scope_relevant")) is bool for row in rows)
        in_scope = [row for row in rows if row.get("scope_relevant") is True] if valid else []
        count = audit.get("excluded_scope_count")
        valid = valid and type(count) is int and count == len(in_scope)
        identity_rows = []
        for row in in_scope:
            rsid = str(row.get("reported_rsid") or "")
            reason = row.get("reason_code")
            missing = reason == "proxy_identifier_missing" and not re.fullmatch(r"rs[1-9]\d*", rsid)
            ambiguous = reason == "multiple_allele_contrasts" and bool(re.fullmatch(r"rs[1-9]\d*", rsid))
            if not (missing or ambiguous):
                valid = False
            if ambiguous:
                identity_rows.append({"scope": "ld_proxy", "variant_id": rsid,
                    "source_variant_id": source, "reason": reason, "line": row.get("line")})
            elif missing:
                identity_rows.append({"scope": "ld_proxy_row", "variant_id": None,
                    "source_variant_id": source, "reason": reason, "line": row.get("line"),
                    "request_key": audit.get("request_key")})
        excluded.extend(identity_rows)
        bound = [row for row in ledger if isinstance(row, Mapping)
                 and row.get("request_key") == audit.get("request_key")
                 and row.get("status") == "success"
                 and row.get("sha256") == audit.get("response_sha256")]
        valid = (valid and bool(audit.get("request_key")) and bool(audit.get("response_sha256"))
                 and bool(bound) and not audit.get("window_coverage_reason")
                 and audit.get("query_build", "GRCh38") == "GRCh38"
                 and audit.get("analysis_build", "GRCh38") == "GRCh38"
                 and audit.get("threshold") == result.get("r2_threshold")
                 and audit.get("window_bp") == result.get("window_bp")
                 and audit.get("state") in {"SUCCESS_DATA", "SUCCESS_NO_DATA", "PARTIAL"}
                 and ((not in_scope and audit.get("requested_scope_complete") is True
                       and audit.get("state") != "PARTIAL") or bool(in_scope)))
        reviewed_audits.append({"source_variant_id": source, "complete": bool(valid)})
        if not valid:
            blockers.append("An LD request, window, or non-identity exclusion remains incomplete.")

    if not excluded:
        return result
    unique_excluded = {(row["scope"], row["variant_id"]) for row in excluded if row["variant_id"]}
    source_exclusions = {row["variant_id"] for row in excluded if row["scope"] == "source"}
    proxy_exclusions = {(row["source_variant_id"], row["variant_id"])
                        for row in excluded if row["scope"] == "ld_proxy"}
    unidentified_rows = {(row["source_variant_id"], row["request_key"], row["line"])
                         for row in excluded if row["scope"] == "ld_proxy_row"}
    warning = (f"Warning: {len(source_exclusions)} source SNP identity/identities, "
               f"{len(proxy_exclusions)} source-specific LD allele contrast(s), and {len(unidentified_rows)} "
               "LD response row(s) without a usable rsID could not be uniquely resolved "
               "and are excluded from Auto Analysis's evaluated scope. A proxy verified for another "
               "source remains evaluated through that verified connection. Known/Novel refers only to "
               "the resolved SNPs that were evaluated; exclusion is not evidence of no association.")
    raw_issues = list(result.get("issues", []))
    waived_issues = []
    for issue in raw_issues:
        source_issue = isinstance(issue, str) and any(
            issue.startswith(f"Cannot verify selected SNP {identifier}:") for identifier in omitted)
        ld_issue = issue == "LD reference retrieval remains incomplete for part of the selected scope."
        if source_issue or ld_issue and audits and all(row["complete"] for row in reviewed_audits):
            waived_issues.append(issue)
        else:
            blockers.append(str(issue))
    if (not source_ids or len(source_set) != len(source_ids)
            or set(result.get("selected_variant_ids", [])) != source_set):
        blockers.append("The screening receipt does not describe this source selection.")
    if not resolved or set(resolved) | set(omitted) != source_set:
        blockers.append("A non-empty, fully enumerated resolved source scope is unavailable.")
    latest_requests = {row.get("request_key"): row for row in ledger if isinstance(row, Mapping)}
    if (any(not isinstance(row, Mapping) for row in ledger) or None in latest_requests
            or any(row.get("status") != "success" for row in latest_requests.values())):
        blockers.append("A recorded request did not complete successfully.")
    if not ledger:
        blockers.append("Retrieval receipts are unavailable.")
    if audits and audit_sources != set(resolved):
        blockers.append("LD coverage receipts do not cover every evaluated source SNP.")
    if result.get("catalog_lookup_complete") is not True:
        blockers.append("The prior-association query did not establish complete coverage.")
    semantics = result.get("phenotype_relevance") or {}
    if not all(isinstance(semantics.get(stage), Mapping) and semantics[stage].get("complete") is True
               for stage in ("native", "catalog")):
        blockers.append("Phenotype relevance is not completely resolved.")
    if (result.get("native_report_binding_issues") or result.get("operation_issues")
            or result.get("configuration_issues") or result.get("source_only_processing_limit")
            or result.get("execution_status") == "not_run"):
        blockers.append("The shared screening has an unresolved non-identity requirement.")
    if result.get("coverage_complete") is not True and not (
            audits and all(row["complete"] for row in reviewed_audits)
            and "LD reference retrieval remains incomplete for part of the selected scope." in waived_issues):
        blockers.append("Complete retrieval within the retained scope was not established.")
    known = result.get("status") == "PREVIOUSLY_REPORTED"
    if not known and result.get("status") == "UNRESOLVED" and not blockers:
        associations = [row for group in result.get("prior_associations", [])
                        for row in group.get("associations", [])]
        count = len({row.get("rsid") or group.get("rsid")
                     for group in result.get("prior_associations", [])
                     for row in group.get("associations", [])})
        if any(not row.get("pmid") or not row.get("study_id") for row in associations):
            blockers.append("A prior-association record lacks its publication or study binding.")
        else:
            decision = classify_ld_screen(pairs=result.get("pairs", []),
                exact_matches=result.get("exact_matches", []), coverage_complete=True,
                identity_complete=True, threshold=result["r2_threshold"], prior_variant_count=count)
            result.update(decision)
            result["selected_variant_ids"] = [identifier for identifier in source_ids if identifier in resolved]
            result["scope"] = "Resolved selected SNPs and their evaluated reference-LD proxies; unresolved SNP identities are excluded"
            result["issues"] = [issue for issue in raw_issues if issue not in waived_issues]
    result["raw_screening_status"] = screen.get("status")
    result["analysis_scope"] = "resolved_variants_only"
    result["auto_scope_review"] = {
        "policy": "auto-resolved-scope-v1", "original_status": screen.get("status"),
        "original_coverage_complete": screen.get("coverage_complete"),
        "original_identity_complete": screen.get("identity_complete"),
        "original_selected_variant_ids": source_ids,
        "evaluated_source_variant_ids": [identifier for identifier in source_ids if identifier in resolved],
        "evaluated_snp_ids": list(result.get("queried_snp_ids", resolved.values())),
        "excluded_variants": excluded, "excluded_variant_count": len(unique_excluded),
        "excluded_source_variant_count": len(source_exclusions),
        "excluded_proxy_query_count": len(proxy_exclusions),
        "excluded_unidentified_row_count": len(unidentified_rows),
        "retained_scope_complete": None if known else not blockers,
        "blocking_reasons": [] if known else list(dict.fromkeys(blockers)),
        "warnings": [warning], "ld_receipt_review": reviewed_audits,
    }
    result["warnings"] = list(dict.fromkeys([*result.get("warnings", []), warning]))
    result["issues"] = list(dict.fromkeys([*result.get("issues", []), warning]))
    return result


def prior_report_screen(
    outcome: Mapping[str, Any], *, phenotype: str, selected_variant_ids: set[str],
    selected_observations: list[Mapping[str, Any]] | None = None,
    require_model_phenotype: bool = False,
) -> dict[str, Any]:
    """Project the shared screening receipt; Auto requires that receipt in every data mode.

    The optional legacy branch matches retrieved trait records, including
    credible-set members, for direct helper compatibility only. New Auto runs
    never use it. Neither incomplete lookup nor missing receipt establishes novelty.
    """

    if outcome.get("error") or outcome.get("safety_review") not in {
        "PASS", "PASS_WITH_WARNINGS"
    }:
        return {
            "status": "UNRESOLVED",
            "label": "Screening incomplete",
            "reason": (
                "The phenotype-specific evidence lookup did not complete its required "
                "checks. Reporting status remains unresolved."
            ),
            "matching_studies": 0,
            "matching_variants": 0,
            "publications": [],
        }
    result = outcome.get("result")
    result = result if isinstance(result, Mapping) else {}
    from ..processing_scope import RESULT_PROCESSING_LIMIT_MESSAGE, is_source_only_processing_limit
    if is_source_only_processing_limit(result):
        return {"status": "UNRESOLVED", "label": "Novelty unresolved",
                "reason": RESULT_PROCESSING_LIMIT_MESSAGE,
                "source_only_processing_limit": True,
                "matching_studies": None, "matching_variants": None, "publications": []}
    ld_screen = result.get("auto_screening")
    if not isinstance(ld_screen, Mapping):
        ld_screen = result.get("novelty_screen")
    if isinstance(ld_screen, Mapping):
        associations = [row for prior in ld_screen.get("prior_associations", [])
                        for row in prior.get("associations", [])]
        return {**dict(ld_screen), "reason": novelty_summary(ld_screen),
                "matching_studies": len({row.get("study_id") for row in associations}),
                "matching_variants": ld_screen.get("prior_variant_count", 0),
                "publications": sorted({str(row["pmid"]) for row in associations if row.get("pmid")})}
    if require_model_phenotype:
        return {"status": "UNRESOLVED", "label": "Novelty unresolved",
                "reason": "The model-assessed prior-report screening receipt is unavailable. "
                          "Prior-report status remains unresolved.",
                "matching_studies": None, "matching_variants": None, "publications": []}
    evidence = _evidence_rows(result)
    resolution = result.get("variant_resolution")
    if not isinstance(resolution, Mapping):
        resolution = variant_identity_context(
            observations=selected_observations or [
                {"variant_id": identifier} for identifier in sorted(selected_variant_ids)
            ],
            evidence=evidence, genome_build="GRCh38",
        )
    unresolved_ids = {
        str(item.get("variant_id") or "")
        for item in resolution.get("unresolved_variants", []) if isinstance(item, Mapping)
    }
    matched_identifiers = (selected_variant_ids - unresolved_ids) | set(
        resolution.get("identified_variant_ids", [])
    )
    matches = [
        item
        for item in _study_associations(evidence)
        if _is_trait_study(item)
        and item["variant_id"] in matched_identifiers
        and _phenotypes_equivalent(phenotype, str(item.get("trait") or ""))
    ]
    if matches:
        return {
            "status": "PREVIOUSLY_REPORTED",
            "label": "Previously reported",
            "reason": (
                f"Retrieved {len({row['study_id'] for row in matches})} matching GWAS "
                f"study record(s) for {phenotype}, linked to selected variants."
            ),
            "matching_studies": len({row["study_id"] for row in matches}),
            "matching_variants": len({row["variant_id"] for row in matches}),
            "publications": sorted({row["pmid"] for row in matches if row.get("pmid")}),
        }
    if resolution.get("complete") is not True:
        return {
            "status": "UNRESOLVED", "label": "Screening incomplete",
            "reason": (
                "No matching prior report was found among the identified variants, but "
                "some source variants could not be uniquely identified. This incomplete "
                "lookup cannot establish novelty; retained evidence remains available."
            ),
            "matching_studies": 0, "matching_variants": 0, "publications": [],
        }
    coverage = result.get("retrieval_coverage")
    if isinstance(coverage, Mapping):
        statuses = coverage.get("module_statuses")
        statuses = statuses if isinstance(statuses, Mapping) else {}
        study_status = statuses.get("locus.study", "missing")
        if study_status not in {"success", "no_data"}:
            return {
                "status": "UNRESOLVED", "label": "Novelty unresolved",
                "reason": (
                    "The prior-report study lookup is incomplete. No matching report was "
                    "found in the available records, but novelty remains unresolved."
                ),
                "study_lookup_status": study_status,
                "matching_studies": 0, "matching_variants": 0, "publications": [],
            }
    return {
        "status": "NO_PRIOR_REPORT_RETRIEVED",
        "label": "No prior report retrieved",
        "reason": (
            f"No selected-variant GWAS record matching {phenotype} was retrieved. "
            "This is not proof of novelty; "
            "nearby or LD-related published signals may require additional review."
        ),
        "matching_studies": 0,
        "matching_variants": 0,
        "publications": [],
    }


class AutoAnalysisManager:
    """Screen each uploaded locus once through Chat's prior-report workflow.

    Observers attach to one background batch. Disconnecting an SSE observer does
    not restart, duplicate or cancel expensive model work. Explicit cancellation
    stops the current JobManager run and prevents any next locus from starting.
    """

    def __init__(
        self,
        *,
        bridge: AnalysisBridge,
        workflow_factory: Callable[..., Any],
        cancel_run: Callable[[str], Any],
        save_result: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.bridge = bridge
        self.workflow_factory = workflow_factory
        self.cancel_run = cancel_run
        self.save_result = save_result
        self._condition = threading.Condition(threading.RLock())
        self._tasks: OrderedDict[str, _Batch] = OrderedDict()

    def has_task(self, task_id: str) -> bool:
        with self._condition:
            return task_id in self._tasks

    def begin(self, task_id: str, phenotype: str, concurrency: int | None = None) -> str:
        if concurrency is not None and (
            isinstance(concurrency, bool) or not isinstance(concurrency, int)
            or not 1 <= concurrency <= MAX_LOCUS_CONCURRENCY
        ):
            raise JobValidationError(f"concurrency must be an integer from 1 to {MAX_LOCUS_CONCURRENCY}")
        queries = self.bridge.queries_for_run(task_id, phenotype)

        def create() -> str:
            # Consistent lock order: upload -> batch. Upload capacity checks use
            # this same order when they ask whether an existing batch is active.
            with self._condition:
                while len(self._tasks) >= _MAX_TASKS:
                    removable = next(
                        (key for key, item in self._tasks.items() if item.state in _TERMINAL),
                        None,
                    )
                    if removable is None:
                        raise JobConflictError("All Auto Analysis slots are currently active.")
                    self._tasks.pop(removable)
                batch = _Batch(
                    task_id, phenotype.strip(), queries,
                    concurrency=concurrency or DEFAULT_LOCUS_CONCURRENCY,
                    locus_states=[{
                        "locus_index": index, "locus_no": index + 1,
                        "locus_id": query["locus_id"], "state": "queued",
                    } for index, query in enumerate(queries)],
                )
                self._tasks[task_id] = batch
            threading.Thread(
                target=self._execute,
                args=(batch,),
                name=f"gwasmind-auto-{task_id[:8]}",
                daemon=True,
            ).start()
            return task_id

        self.bridge.ensure_run(task_id, create)
        with self._condition:
            existing = self._get(task_id)
            if existing.phenotype != phenotype.strip():
                raise JobConflictError(
                    "This upload already has an analysis for another phenotype; "
                    "upload it again to start a different analysis."
                )
            if concurrency is not None and existing.concurrency != concurrency:
                raise JobConflictError(
                    "This analysis already has a concurrency setting; "
                    "upload it again to change the setting."
                )
        return task_id

    def snapshot(self, task_id: str) -> dict[str, Any]:
        with self._condition:
            batch = self._get(task_id)
            return {
                "task_id": task_id,
                "run_id": task_id,
                "run_ids": list(batch.run_ids),
                "state": batch.state,
                "phenotype": batch.phenotype,
                "analysis_mode": _ANALYSIS_MODE,
                "concurrency": batch.concurrency,
                "loci": _loci_snapshot(batch),
                "timing": batch.timing.snapshot(),
                "progress": _progress_snapshot(batch),
                "history_id": batch.history_id,
                "results": deepcopy(batch.results),
                "error": batch.error,
            }

    def wait_for_events(
        self, task_id: str, *, after_sequence: int = 0,
        timeout: float = parameter("execution.auto_event_wait_seconds"),
    ) -> tuple[list[dict[str, Any]], bool]:
        with self._condition:
            batch = self._get(task_id)
            self._condition.wait_for(
                lambda: batch.sequence > after_sequence or batch.state in _TERMINAL,
                timeout=max(0, timeout),
            )
            # A fresh measurement follows replayed events without becoming a
            # retained event: observers must not wake one another in a loop.
            timing = {"type": "timing", "seq": batch.sequence, "state": batch.state,
                      "timing": batch.timing.snapshot(), "loci": _loci_snapshot(batch)}
            return (
                [*[dict(item) for item in batch.events if item["seq"] > after_sequence], timing],
                batch.state in _TERMINAL,
            )

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._condition:
            batch = self._get(task_id)
            if batch.state in _TERMINAL:
                return self.snapshot(task_id)
            batch.cancelled.set()
            batch.state = "cancelling"
            run_ids = list(batch.active_run_ids)
        for run_id in run_ids:
            try:
                self.cancel_run(run_id)
            except JobError:
                pass
        return self.snapshot(task_id)

    def close(self) -> None:
        with self._condition:
            task_ids = [key for key, batch in self._tasks.items() if batch.state not in _TERMINAL]
        for task_id in task_ids:
            self.cancel(task_id)

    def _get(self, task_id: str) -> _Batch:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise JobNotFoundError("Auto Analysis task was not found") from exc

    def _publish(self, batch: _Batch, event: Mapping[str, Any]) -> None:
        with self._condition:
            batch.sequence += 1
            batch.events.append({**event, "seq": batch.sequence})
            if len(batch.events) > _MAX_EVENTS:
                del batch.events[: len(batch.events) - _MAX_EVENTS]
            self._condition.notify_all()

    def _check_cancelled(self, batch: _Batch) -> None:
        if batch.cancelled.is_set():
            raise _AnalysisCancelled("Auto Analysis was cancelled.")

    def _remember_progress(
        self, batch: _Batch, index: int, workflow: str, event: Mapping[str, Any],
    ) -> None:
        """Retain a compact current snapshot independently of the bounded SSE log."""
        with self._condition:
            record = batch.progress.setdefault((index, workflow), {
                "locus_index": index, "workflow": workflow,
                "stages": [], "variant_normalizations": [],
            })
            if event["type"] == "variant_normalization":
                existing = {item["original_id"]: item for item in record["variant_normalizations"]}
                for item in event["normalizations"]:
                    # The public receipt validator already requires a same-coordinate
                    # GRCh38 reversal. A contradictory later receipt cannot replace it.
                    previous = existing.get(item["original_id"])
                    if previous is None and len(existing) < 2000:
                        existing[item["original_id"]] = deepcopy(item)
                record["variant_normalizations"] = list(existing.values())
                return
            stages = record["stages"]
            stage = next((item for item in stages if item["stage"] == event["stage"]), None)
            if stage is None:
                stage = {"stage": event["stage"], "details": [], "substeps": []}
                stages.append(stage)
                stages.sort(key=lambda item: _STAGES.index(item["stage"]))
            late_activity = (
                stage.get("state") in {"passed", "warning", "stopped", "failed"}
                and event["state"] in {"pending", "running"}
            )
            for key in ("state", "summary", "requested_task", "substage", "ranking_unavailable"):
                if key in event and not late_activity:
                    stage[key] = event[key]
            # A delayed operation callback must not turn recorded completion into
            # an unfinished History stage or replace its measured final duration.
            # Later terminal review decisions are still applied normally.
            if "metrics" in event and not late_activity:
                stage.setdefault("metrics", {}).update(event["metrics"])
            if event["type"] == "review_update" and not late_activity:
                stage["review_update"] = deepcopy(dict(event))
            progress = event.get("progress")
            if progress:
                steps = stage["substeps"]
                position = next((n for n, item in enumerate(steps)
                                 if _substep_key(item) == _substep_key(progress)), None)
                if position is not None:
                    terminal = {"completed", "incomplete", "failed", "unavailable", "blocked", "not_run", "cached"}
                    stale = (progress.get("operation_id") and steps[position].get("status") in terminal
                             and progress.get("status") not in terminal)
                    if not stale:
                        steps[position] = {**steps[position], **deepcopy(progress)}
                else:
                    if len(steps) >= _MAX_STAGE_SUBSTEPS:
                        steps.pop(0)
                        stage["substeps_omitted"] = stage.get("substeps_omitted", 0) + 1
                    steps.append(deepcopy(progress))
            # Progress summaries and changing token/time counters already live in
            # structured fields; repeating their text would grow history per poll.
            detail = event.get("detail")
            metrics = event.get("metrics", {})
            telemetry = bool(set(metrics) & {
                "ranking_elapsed_seconds", "output_tokens", "received_chars", "estimated_output_tokens",
            }) and event["state"] == "running"
            if detail and not progress and not telemetry and detail not in stage["details"]:
                if len(stage["details"]) >= _MAX_STAGE_DETAILS:
                    stage["details"].pop(0)
                    stage["details_omitted"] = stage.get("details_omitted", 0) + 1
                stage["details"].append(detail)

    def _workflow(
        self, batch: _Batch, query: Mapping[str, Any], index: int,
    ) -> dict[str, Any]:
        phase = "screening"
        workflow_run_ids: set[str] = set()

        def on_run(run_id: str) -> None:
            with self._condition:
                batch.active_run_ids.add(run_id)
                workflow_run_ids.add(run_id)
                batch.run_ids.append(run_id)
            if batch.cancelled.is_set():
                self.cancel_run(run_id)
                raise _AnalysisCancelled("Auto Analysis was cancelled.")

        def emit(event: Mapping[str, Any]) -> None:
            self._check_cancelled(batch)
            if event.get("type") == "ranking_evidence":
                return
            elif event.get("type") in {"stage", "review_update", "variant_normalization"}:
                public = _public_workflow_event(event)
                if not public:
                    return
                if public["type"] != "variant_normalization":
                    public["requested_task"] = ChatIntent.VARIANT_DESCRIPTION.value
                self._remember_progress(batch, index, phase, public)
                self._publish(batch, {
                    **public,
                    "type": {"stage": "workflow_stage", "review_update": "workflow_review_update",
                             "variant_normalization": "workflow_variant_normalization"}[event["type"]],
                    "workflow": phase,
                    "locus_index": index,
                    "locus_id": query["locus_id"],
                })
            elif event.get("type") == "error":
                self._publish(batch, {
                    "type": "log", "level": "error", "locus_index": index,
                    "message": str(event.get("content") or "Analysis did not complete."),
                })

        variants = self.bridge.source_observations_for_locus(batch.task_id, query["locus_id"])
        if not variants:  # Older retained tasks never borrow from another upload or the plot.
            variants = [dict(item) for item in query["variants"]]
        workflow = self.workflow_factory(
            emit=emit,
            source_observations=[dict(item) for item in variants],
            source_trait=batch.phenotype,
            query_phenotype=batch.phenotype,
            required_intent=ChatIntent.VARIANT_DESCRIPTION,
            on_run=on_run,
            cancel_event=batch.cancelled,
            auto_context={"analysis_mode": _ANALYSIS_MODE},
        )
        question = format_prompt(
            "web.auto.screen_question",
            phenotype=batch.phenotype,
        )
        self._check_cancelled(batch)
        try:
            outcome = workflow.run({
                "question": question,
                "variants": [str(item["variant_id"]) for item in variants],
                "history": [],
            })
        finally:
            with self._condition:
                batch.active_run_ids.difference_update(workflow_run_ids)
        self._check_cancelled(batch)
        return {**outcome, "source_observations": [dict(item) for item in variants],
                "ranking_evidence": []}

    def _stats(self, batch: _Batch) -> None:
        with self._condition:
            rows = list(batch.results)
            states = [item["state"] for item in batch.locus_states]
            self._publish(batch, {
                "type": "stats",
                "snps": sum(len(query["variants"]) for query in batch.queries),
                "loci": len(batch.queries),
                "concurrency": batch.concurrency,
                "timing": batch.timing.snapshot(),
                "running": states.count("running"),
                "waiting": states.count("queued"),
                "completed": len(rows),
                "failed": states.count("failed"),
                "cancelled": states.count("cancelled"),
                "known": sum(row["screening_status"] == "PREVIOUSLY_REPORTED" for row in rows),
                "novel": sum(row["screening_status"] == "NOVEL_IN_SCOPE" for row in rows),
                "unresolved": sum(row["status"] == "Unresolved" for row in rows),
                "prioritized": sum(row["decision_status"] == "PRIORITIZED" for row in rows),
                "exploratory_rankings": sum(row["decision_status"] == "EXPLORATORY_RANKING" for row in rows),
                "novelty_unresolved": sum(row["screening_status"] == "UNRESOLVED" for row in rows),
                "needs_review": sum(row["status"] == "Unresolved" for row in rows),
                "previously_reported": sum(
                    row["screening_status"] == "PREVIOUSLY_REPORTED" for row in rows
                ),
                "no_prior_report": sum(
                    row["screening_status"] == "NO_PRIOR_REPORT_RETRIEVED" for row in rows
                ),
            })

    def _locus_state(self, batch: _Batch, index: int, state: str) -> None:
        with self._condition:
            if state == "running":
                batch.locus_timers.setdefault(index, _RunTiming()).start()
            elif state in _TERMINAL and index in batch.locus_timers:
                batch.locus_timers[index].finish()
            batch.locus_states[index]["state"] = state
            event = {"type": "locus_state", **_locus_snapshot(batch, index)}
        self._publish(batch, event)
        self._stats(batch)

    def _execute_locus(self, batch: _Batch, index: int) -> dict[str, Any]:
        self._check_cancelled(batch)
        query = batch.queries[index]
        self._locus_state(batch, index, "running")
        try:
            return self._run_locus(batch, query, index)
        finally:
            # Freeze at the worker's actual exit, not when the batch scheduler
            # next collects this future or finishes saving another locus.
            with self._condition:
                batch.locus_timers[index].finish()

    def _run_locus(self, batch: _Batch, query: Mapping[str, Any], index: int) -> dict[str, Any]:
        self._publish(batch, {
            "type": "log", "level": "info", "locus_index": index,
            "message": f"Screening locus {query['locus_id']} for prior "
            f"{batch.phenotype} GWAS reports.",
        })
        screening = {}
        screen = prior_report_screen(
            {"error": "Screening has not completed."}, phenotype=batch.phenotype,
            selected_variant_ids=set(), require_model_phenotype=True,
        )
        try:
            screening = self._workflow(batch, query, index)
            screen = prior_report_screen(
                screening, phenotype=batch.phenotype,
                selected_variant_ids={str(item["variant_id"]) for item in query["variants"]},
                selected_observations=[dict(item) for item in query["variants"]],
                require_model_phenotype=True,
            )
            outcome = screening
        except _AnalysisCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - a locus failure must not stop its peers
            outcome = {
                "error": str(exc) if isinstance(exc, JobError) else (
                    "This locus could not finish because of a local processing error. "
                    "Other loci continue; this locus can be run again separately."
                ),
                "report_quality": "STOPPED", "safety_review": "STOPPED",
            }
        self._check_cancelled(batch)
        return self._result_row(batch, query, index, screening, screen, outcome)

    @staticmethod
    def _result_row(batch, query, index, screening, screen, outcome) -> dict[str, Any]:
        result = outcome.get("result") or {}
        # Keep the shared locus/evidence projection and the legacy row shape,
        # while ensuring a screening result never acquires gene-ranking output.
        row = project_results([{**result, "ranked_genes": []}], [query])[0]
        quality = str(outcome.get("report_quality") or "STOPPED")
        row.update({
            "analysis_mode": _ANALYSIS_MODE,
            "status": {"PREVIOUSLY_REPORTED": "Known", "NOVEL_IN_SCOPE": "Novel (in scope)"}.get(
                screen["status"], "Unresolved"),
            "decision_status": screen["status"],
            "top_gene": "-", "top_gene_id": "-", "candidate_genes": "Not requested",
            "candidate_gene_ids": [], "ranked_gene_ids": [],
            "ranked_count": 0, "displayed_ranked_count": 0,
        })
        report = str(outcome.get("report_markdown") or outcome.get("error") or "")
        normalizations = {item["original_id"]: dict(item)
                          for workflow in (screening, outcome)
                          for item in public_source_normalizations(workflow.get("result") or {})}
        row.update({
            "locus_no": index + 1,
            "screening_status": screen["status"],
            "novelty_status": "PREVIOUSLY_REPORTED" if screen["status"] == "PREVIOUSLY_REPORTED" else "UNRESOLVED"
                if screen["status"] == "UNRESOLVED" else "NOVEL_IN_SCOPE"
                if screen["status"] == "NOVEL_IN_SCOPE" else "NOT_ESTABLISHED",
            "ranking_mode": "not_started",
            "novelty_notice": (
                "Novelty unresolved · available evidence only" if screen["status"] == "UNRESOLVED"
                else "No prior report retrieved · novelty not established" if screen["status"] == "NO_PRIOR_REPORT_RETRIEVED"
                else "No prior association found in the queried database version and search scope" if screen["status"] == "NOVEL_IN_SCOPE"
                else "Previously reported"
            ),
            "screening_label": screen["label"],
            "screening_reason": screen["reason"],
            "screening": screen,
            "original_screening": deepcopy((screening.get("result") or {}).get("novelty_screen", {})),
            "auto_scope_review": deepcopy(screen.get("auto_scope_review", {})),
            "original_screening_status": screen.get("raw_screening_status", screen["status"]),
            "warnings": list(dict.fromkeys([*row.get("warnings", []), *screen.get("warnings", [])])),
            "variant_resolution": (screening.get("result") or {}).get("variant_resolution", {}),
            "screening_report_markdown": str(screening.get("report_markdown") or ""),
            "report_markdown": report,
            "report_quality": quality,
            "analysis_status": outcome.get("analysis_status", {}),
            "computed_facts": outcome.get("computed_facts", {}),
            "ranking_evidence": [],
            "gene_evidence_cards": [],
            "variant_normalizations": list(normalizations.values()),
            "llm_interpretation": report or screen["reason"],
            "source_phenotype": batch.phenotype,
            "query_phenotype": batch.phenotype,
            "selected_variants": [dict(item) for item in screening.get("source_observations", query["variants"])],
            "selected_region": {
                "chromosome": query["variants"][0]["chromosome"],
                "start": min(item["position"] for item in query["variants"]),
                "end": max(item["position"] for item in query["variants"]),
            } if (
                all(item.get("position") is not None for item in query["variants"])
                and len({item.get("chromosome") for item in query["variants"]}) == 1
                and query["variants"][0].get("chromosome")
            ) else None,
            "run_id": outcome.get("run_id"),
            "screening_run_id": screening.get("run_id"),
            "error": outcome.get("error"),
        })
        return row

    def _execute(self, batch: _Batch) -> None:
        try:
            with self._condition:
                batch.timing.start()
            self._publish(batch, {
                "type": "loci", "concurrency": batch.concurrency,
                "data": _loci_snapshot(batch), "timing": batch.timing.snapshot(),
            })
            self._stats(batch)
            next_index = 0
            pending = {}
            with ThreadPoolExecutor(
                max_workers=batch.concurrency,
                thread_name_prefix=f"gwasmind-locus-{batch.task_id[:8]}",
            ) as pool:
                while pending or next_index < len(batch.queries):
                    if batch.cancelled.is_set():
                        for future in pending:
                            future.cancel()
                        break
                    while next_index < len(batch.queries) and len(pending) < batch.concurrency:
                        self._check_cancelled(batch)
                        future = pool.submit(self._execute_locus, batch, next_index)
                        pending[future] = next_index
                        next_index += 1
                    finished, _ = wait(pending, timeout=parameter("execution.auto_poll_seconds"),
                                       return_when=FIRST_COMPLETED)
                    for future in finished:
                        index = pending.pop(future)
                        try:
                            row = future.result()
                        except _AnalysisCancelled:
                            continue
                        except Exception:  # noqa: BLE001 - malformed output is local to this locus
                            outcome = {
                                "error": "This locus could not finish because its result could not be read. "
                                "Other loci continue; this locus can be run again separately.",
                                "report_quality": "STOPPED", "safety_review": "STOPPED",
                            }
                            screen = prior_report_screen(outcome, phenotype=batch.phenotype,
                                                         selected_variant_ids=set(), require_model_phenotype=True)
                            row = self._result_row(batch, batch.queries[index], index, {}, screen, outcome)
                        with self._condition:
                            if index in batch.locus_timers:
                                batch.locus_timers[index].finish()
                            row["timing"] = batch.locus_timers.get(index, _RunTiming()).snapshot()
                            batch.results.append(row)
                            batch.results.sort(key=lambda item: item["locus_no"])
                            state = "failed" if row.get("error") or row["report_quality"] == "STOPPED" else "completed"
                            batch.locus_states[index]["state"] = state
                            results = list(batch.results)
                        self._publish(batch, {"type": "locus_state", **_locus_snapshot(batch, index)})
                        self._stats(batch)
                        self._publish(batch, {
                            "type": "progress", "percent": round(100 * len(results) / len(batch.queries)),
                        })
                        self._save(batch, status="running")
                        self._publish(batch, {
                            "type": "results", "data": results, "history_id": batch.history_id,
                        })
            self._check_cancelled(batch)
            if batch.results and all(row.get("error") for row in batch.results):
                raise JobValidationError(str(batch.results[0]["error"]))
            with self._condition:
                batch.timing.finish()
            self._save(batch, status="completed")
            with self._condition:
                self._check_cancelled(batch)
                batch.state = "completed"
                self._publish(batch, {
                    "type": "complete",
                    "download_url": f"/api/analysis/quick-download/{batch.task_id}",
                    "history_id": batch.history_id,
                    "timing": batch.timing.snapshot(),
                })
        except _AnalysisCancelled:
            batch.error = "Auto Analysis was cancelled."
            with self._condition:
                batch.state = "cancelled"
                batch.timing.finish()
                for index, item in enumerate(batch.locus_states):
                    if item["state"] in {"queued", "running"}:
                        if index in batch.locus_timers:
                            batch.locus_timers[index].finish()
                        item["state"] = "cancelled"
                        self._publish(batch, {"type": "locus_state", **_locus_snapshot(batch, index)})
            self._stats(batch)
            self._publish(batch, {"type": "error", "message": batch.error,
                                  "timing": batch.timing.snapshot()})
            self._save(batch, status="cancelled")
        except Exception as exc:  # noqa: BLE001 - isolate one background batch failure
            batch.error = str(exc) if isinstance(exc, JobError) else (
                "The local Auto Analysis workflow did not complete."
            )
            with self._condition:
                batch.state = "failed"
                batch.timing.finish()
                for index, item in enumerate(batch.locus_states):
                    if item["state"] in {"queued", "running"}:
                        if index in batch.locus_timers:
                            batch.locus_timers[index].finish()
                        item["state"] = "failed"
                        self._publish(batch, {"type": "locus_state", **_locus_snapshot(batch, index)})
            self._publish(batch, {"type": "error", "message": batch.error,
                                  "timing": batch.timing.snapshot()})
            self._save(batch, status="failed")
        finally:
            with self._condition:
                batch.active_run_ids.clear()
                self._condition.notify_all()

    def _save(self, batch: _Batch, *, status: str) -> None:
        if self.save_result is None:
            return
        try:
            # Only the existing completed-locus/terminal checkpoints write history.
            # Snapshot under the lock; perform SQLite/file I/O after releasing it.
            with self._condition:
                payload = {
                    "title": f"{batch.phenotype} · {len(batch.queries)} loci",
                    "source_phenotype": batch.phenotype,
                    "analysis_mode": _ANALYSIS_MODE,
                    "task_id": batch.task_id,
                    "run_id": batch.task_id,
                    "queries": batch.queries,
                    "results": list(batch.results),
                    "concurrency": batch.concurrency,
                    "loci": _loci_snapshot(batch),
                    "timing": batch.timing.snapshot(),
                    "progress": _progress_snapshot(batch),
                    "status": status,
                }
            saved = self.save_result(payload)
            if isinstance(saved, Mapping) and saved.get("history_id"):
                batch.history_id = str(saved["history_id"])
        except (OSError, ValueError, sqlite3.Error):
            self._publish(batch, {
                "type": "log", "level": "warning",
                "message": "This analysis is available in this session, but local history "
                "could not be updated.",
            })
