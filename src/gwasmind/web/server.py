from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import mimetypes
import os
import queue
import re
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Mapping
from concurrent.futures import CancelledError, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextvars import copy_context
from dataclasses import replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .. import __version__
from ..biological_context import (
    BiologicalContext,
    phenotype_from_question,
)
from ..configuration import configuration_provenance, parameter
from ..ld_expansion import expand_ranking_variants
from ..mcp.europe_pmc import enrich_citation_registry
from ..model_options import (
    DEFAULT_MODEL_BASE_URL,
    DEFAULT_MODEL_ID,
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORTS,
    THINKING_DISABLED,
    THINKING_ENABLED,
    THINKING_MODES,
)
from ..novelty import (
    novelty_context_for_report,
    is_novelty_question,
    novelty_coverage_check,
    prior_screen_precondition,
    screen_prior_associations,
)
from ..processing_scope import (
    RESULT_PROCESSING_LIMIT_MESSAGE,
    RESULT_PROCESSING_SUMMARY,
    is_source_only_processing_limit,
)
from ..prompting import format_prompt, load_prompt
from ..provider_catalog import (
    ProviderCatalogError,
    infer_provider_id,
    validate_provider_configuration,
)
from ..public_model_catalog import PublicModelCatalog
from ..qtl import QTL_STUDY_TYPE_LABELS, qtl_study_type_from_semantic_payload
from ..region_gate import local_gene_window_label
from ..source_gwas import summarize_source_gwas_observations
from ..staged_reasoning import (
    SafetyOutcome,
    SafetySeverity,
    SafetyStatus,
    _writer_warning_messages,
    build_hybrid_final_context,
    configured_safety_check,
    preserve_gate1_warnings,
    public_safety_warning_items,
    public_safety_warning_messages,
    run_hybrid_final,
    run_safety_audit,
)
from ..v2g.report import rank_public_cards
from .analysis_bridge import AnalysisBridge, project_results
from .analysis_history import restore_analysis_plot
from .auto_analysis import AutoAnalysisManager, finalize_auto_screen
from .chat_intent import (
    ChatIntent,
    ChatRoute,
    QTLRoutingError,
    is_explicit_general_chat,
    model_chat_router_messages,
    parse_model_chat_route,
    reconcile_chat_route,
    route_chat_intent,
    route_chat_request,
)
from .chat_response import (
    database_response_contract,
    selected_variant_trait_answer,
    task_specific_retrieval_answer,
)
from .general_response import (
    NoRetrievalContext,
    NoRetrievalMode,
    authoritative_gwasmind_software_facts,
    bound_no_retrieval_history,
    classify_no_retrieval_mode,
    run_no_retrieval_response,
)
from .history_store import (
    HistoryError,
    HistoryNotFoundError,
    LocalHistoryStore,
    default_history_root,
)
from .jobs import (
    ARTIFACT_NAMES,
    DUCKDB_MEMORY_LIMIT,
    MAX_CANDIDATE_GENES_PER_QUERY,
    MAX_QUERIES,
    MAX_VARIANTS_PER_QUERY,
    TERMINAL_STATES,
    WORKER_THREADS,
    JobError,
    JobManager,
    JobNotFoundError,
    JobValidationError,
)
from .ranking_policy import (
    RANKING_NOT_CONFIGURED_DETAIL,
    RANKING_UNAVAILABLE_DETAIL,
    RANKING_UNAVAILABLE_SUMMARY,
    live_model_only_result,
    model_ranking_unavailable,
)
from .stage_bridge import build_runtime_stage_bridge, variant_identity_context
from .templates import render_page
from .ui_state import UiDataError, UiState
from .xlsx_report import build_xlsx_report

MAX_JSON_BYTES = parameter("web.max_json_bytes")
MAX_UPLOAD_JSON_BYTES = parameter("web.max_upload_json_bytes")
API_PREFIX = "/api/v1"
_RUN_ID = re.compile(r"^[a-f0-9]{32}$")
_CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
)
# On Windows, mimetypes lets registry entries override its defaults; a common
# mis-registration maps .js to text/plain, and with nosniff the browser then
# refuses to run the page scripts. These are the standard library defaults.
_STATIC_CONTENT_TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
                         ".html": "text/html", ".json": "application/json", ".svg": "image/svg+xml"}


def _auto_report_warning_scope(answer, screen, review):
    """Label a retained full-input warning without changing its Gate record."""
    scope = screen.get("auto_scope_review", {}) if isinstance(screen, Mapping) else {}
    if not (scope.get("original_status") == "UNRESOLVED"
            and screen.get("status") == "NOVEL_IN_SCOPE"
            and scope.get("retained_scope_complete") is True):
        return answer
    opening, boundary, body = answer.partition("\n\n## ")
    prefix = "Original full-input review (including excluded SNPs): "
    if prefix in opening:
        return answer
    for check in review.facts.checks:
        if check.check_id == "novelty_coverage":
            opening = opening.replace(check.fact, prefix + check.fact, 1)
    opening = opening.replace("prior-report coverage (novelty remains unresolved)",
                              "original full-input prior-report coverage (including excluded SNPs)", 1)
    return opening + boundary + body


class _LDProgressMailbox:
    """Coalesce adjacent live counters without dropping operation boundaries.

    Producers never wait for a browser to read. Starts, completions, failures,
    retries and changes of scope stay ordered; the final LD result is separate.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._pending = deque()

    _counter_fields = frozenset({
        "phase", "status", "reference_provider", "provider_version", "identity_scope",
        "name", "operation", "tool", "channel", "unit", "batch_index", "batch_count", "attempt",
        "scope_sha256", "context_sha256", "shortlist_sha256", "source_count", "completed_count",
        "total_count", "validated_count", "pending_count", "variant_count", "total_analysis_variants",
        "elapsed_seconds", "step", "page_size", "offset", "wait_seconds", "network_requests",
        "cache_hits", "failed_cache_hits", "completed_pages", "failed_pages", "batch_size",
        "returned_count", "verified_proxy_count", "previous_validated_count", "newly_validated_count",
        "proxy_count", "candidate_count", "populations_requested", "operation_id",
    })

    @classmethod
    def _is_live_counter(cls, value):
        # A new/unknown payload may contain scientific observations. Retain it.
        return (value.get("status") in {"running", "waiting"}
                and value.keys() <= cls._counter_fields
                and all(item is None or isinstance(item, (str, int, float, bool)) for item in value.values()))

    @staticmethod
    def _scope(value):
        return tuple(value.get(key) for key in (
            "phase", "reference_provider", "provider_version", "identity_scope",
            "name", "operation", "tool", "channel", "batch_index", "attempt",
            "scope_sha256", "context_sha256", "shortlist_sha256", "operation_id",
        ))

    def put(self, value):
        with self._condition:
            previous = self._pending[-1] if self._pending else None
            if (previous is not None and self._is_live_counter(value)
                    and self._is_live_counter(previous)
                    and self._scope(previous) == self._scope(value)):
                self._pending[-1] = value
            else:
                self._pending.append(value)
            self._condition.notify()

    def get(self, timeout):
        with self._condition:
            if not self._condition.wait_for(lambda: bool(self._pending), timeout=timeout):
                raise queue.Empty
            return self._pending.popleft()


def _retain_worker_event(events, event):
    """Keep audit inputs, excluding only explicitly transient keepalives."""
    if not (event.get("type") == "heartbeat" and event.get("transient") is True):
        events.append(dict(event))


def _validate_model_provider_request(value: Mapping[str, Any]) -> str:
    """Validate a Settings provider selection before touching session state."""

    base_url = value.get("api_base_url")
    model_id = value.get("model")
    requested_provider = value.get("provider_id")
    provider_id = (
        requested_provider
        if isinstance(requested_provider, str) and requested_provider.strip()
        else infer_provider_id(str(base_url or ""), str(model_id or ""))
    )
    try:
        validated = validate_provider_configuration(
            provider_id=provider_id,
            base_url=base_url,
            model_id=model_id,
        )
    except ProviderCatalogError as exc:
        raise JobValidationError(str(exc)) from exc
    return validated.provider_id


_PUBLIC_EVIDENCE_LABELS = {
    "variant": "variant annotations",
    "credible_set": "credible sets",
    "study": "GWAS study metadata",
    "fine_mapping": "fine-mapping data",
    "l2g": "L2G predictions",
    "molecular_qtl": "molecular QTL evidence",
    "colocalisation": "colocalization evidence",
    "regulatory": "regulatory links",
    "expression": "gene expression",
    "function": "gene function",
    "pathway": "pathways",
    "literature": "supporting literature",
    "target_profile": "target profiles",
    "constraint": "genetic constraint",
    "tractability": "target tractability",
    "safety": "target safety",
    "interaction": "molecular interactions",
    "phenotype": "model-organism phenotypes",
    "target_disease": "target-disease evidence",
    "disease_profile": "disease profiles",
    "known_drug": "known drugs",
    "clinical_candidate": "clinical candidates",
    "indication": "drug indications",
    "mechanism": "mechanisms of action",
    "pharmacogenomics": "pharmacogenomic evidence",
}

_PUBLIC_MODULE_LABELS = {
    "schema.discovery": "Open Targets service availability",
    "locus.variant": "variant annotations",
    "locus.study": "GWAS study metadata",
    "locus.credible_set": "credible sets",
    "locus.fine_mapping": "fine-mapping data",
    "locus.variant_consequence": "variant consequences",
    "locus.qtl": "molecular QTL evidence",
    "locus.colocalisation": "colocalization evidence",
    "locus.enhancer_to_gene": "enhancer-to-gene links",
    "locus.l2g": "L2G predictions",
    "target.core": "candidate-gene identities",
    "target.function": "gene function",
    "target.expression": "gene expression",
    "target.go": "Gene Ontology annotations",
    "target.pathway": "biological pathways",
    "evidence.europepmc": "supporting literature",
}

_PUBLIC_MODULE_COUNT_LABELS = {
    "locus.variant": ("variant annotation", "variant annotations"),
    "locus.study": ("GWAS study record", "GWAS study records"),
    "locus.credible_set": ("credible set", "credible sets"),
    "locus.fine_mapping": ("fine-mapping record", "fine-mapping records"),
    "locus.variant_consequence": ("variant consequence", "variant consequences"),
    "locus.qtl": ("molecular QTL record", "molecular QTL records"),
    "locus.colocalisation": ("colocalization record", "colocalization records"),
    "locus.enhancer_to_gene": ("enhancer-to-gene link", "enhancer-to-gene links"),
    "locus.l2g": ("L2G prediction", "L2G predictions"),
    "target.core": ("candidate-gene identity", "candidate-gene identities"),
    "target.function": ("gene-function record", "gene-function records"),
    "target.expression": ("gene-expression record", "gene-expression records"),
    "target.go": ("Gene Ontology annotation", "Gene Ontology annotations"),
    "target.pathway": ("biological-pathway record", "biological-pathway records"),
    "target.constraint": ("genetic-constraint record", "genetic-constraint records"),
    "target.tractability": ("target-tractability record", "target-tractability records"),
    "target.safety": ("target-safety record", "target-safety records"),
    "target.interaction": ("molecular-interaction record", "molecular-interaction records"),
    "target.mouse_phenotype": (
        "model-organism phenotype record",
        "model-organism phenotype records",
    ),
    "evidence.europepmc": ("publication record", "publication records"),
    "drug.core": ("drug record", "drug records"),
    "drug.mechanism": ("mechanism-of-action record", "mechanism-of-action records"),
    "drug.indication": ("drug-indication record", "drug-indication records"),
    "drug.warning": ("drug-warning record", "drug-warning records"),
    "drug.adverse_event": ("adverse-event record", "adverse-event records"),
    "drug.pharmacogenomics": (
        "pharmacogenomic record",
        "pharmacogenomic records",
    ),
    "drug.clinical_target": ("clinical-target record", "clinical-target records"),
}

_PUBLIC_TOOL_DATA = {
    "ot_resolve_entities": ("selected variant and gene identifiers",),
    "ot_get_locus_context": (
        "variant annotations",
        "GWAS study metadata",
        "credible sets",
        "fine-mapping data",
    ),
    "ot_get_locus_gene_evidence": (
        "variant consequences",
        "molecular QTL evidence",
        "colocalization evidence",
        "enhancer-to-gene links",
        "L2G predictions",
    ),
    "ot_get_target_profiles": ("target expression, function, safety, and tractability data",),
    "ot_get_target_disease_evidence": ("target-disease evidence",),
    "ot_get_disease_profiles": ("disease profile data",),
    "ot_get_drug_evidence": ("drug, indication, and clinical evidence",),
    "ot_analyze_gene_set": ("gene-set enrichment data",),
}


def _public_list(items: list[str] | tuple[str, ...]) -> str:
    unique = list(dict.fromkeys(item for item in items if item))
    if not unique:
        return "the requested evidence"
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def _public_retrieval_bundle(
    intent: ChatIntent,
    *,
    qtl_types: tuple[str, ...] = (),
) -> str | None:
    labels = [
        _PUBLIC_EVIDENCE_LABELS[item]
        for item in intent.retrieval_bundle
        if item in _PUBLIC_EVIDENCE_LABELS
    ]
    if qtl_types and "molecular QTL evidence" in labels:
        qtl_labels = [QTL_STUDY_TYPE_LABELS.get(item, item) for item in qtl_types]
        qtl_index = labels.index("molecular QTL evidence")
        labels[qtl_index] = f"molecular QTL evidence ({_public_list(qtl_labels)})"
    if not labels:
        return None
    return f"Planned evidence: {_public_list(labels)}."


def _public_semantic_record_counts(details: Mapping[str, Any]) -> str | None:
    """Render bounded counts from actual returned record modules only."""

    if details.get("tool") == "ot_get_capabilities":
        return None
    counts_value = details.get("module_counts")
    counts = counts_value if isinstance(counts_value, Mapping) else {}
    phrases: list[str] = []
    recognized_total = 0
    for module, forms in _PUBLIC_MODULE_COUNT_LABELS.items():
        count = counts.get(module)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            continue
        recognized_total += count
        phrases.append(f"{count:,} {forms[0] if count == 1 else forms[1]}")

    total_value = details.get("records")
    total = (
        total_value
        if isinstance(total_value, int) and not isinstance(total_value, bool) and total_value >= 0
        else None
    )
    if not counts and total is not None and details.get("tool") == "ot_resolve_entities":
        return f"Resolved {total:,} selected {'identifier' if total == 1 else 'identifiers'}."
    if total is not None and total > recognized_total:
        other = total - recognized_total
        phrases.append(f"{other:,} other evidence {'record' if other == 1 else 'records'}")
    if phrases:
        return f"Retrieved {_public_list(phrases)}."
    if total is not None:
        return f"Retrieved {total:,} evidence {'record' if total == 1 else 'records'}."
    return None


def _public_semantic_record_count_lines(details: Mapping[str, Any]) -> tuple[str, ...]:
    """Return one public, actual-count line per recognized evidence module.

    Counts come only from ``semantic_tool_finished.module_counts``, which the executor
    derives from returned records rather than from the requested module list.
    """

    if details.get("tool") == "ot_get_capabilities":
        return ()
    counts_value = details.get("module_counts")
    counts = counts_value if isinstance(counts_value, Mapping) else {}
    lines: list[str] = []
    recognized_total = 0
    for module, forms in _PUBLIC_MODULE_COUNT_LABELS.items():
        count = counts.get(module)
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or count > 10_000_000
        ):
            continue
        recognized_total += count
        lines.append(f"Retrieved {count:,} {forms[0] if count == 1 else forms[1]}.")

    total_value = details.get("records")
    total = (
        total_value
        if isinstance(total_value, int)
        and not isinstance(total_value, bool)
        and 0 <= total_value <= 10_000_000
        else None
    )
    tool = str(details.get("tool", ""))
    if not lines and total is not None and tool == "ot_resolve_entities":
        return (f"Resolved {total:,} selected {'identifier' if total == 1 else 'identifiers'}.",)
    if total is not None and total > recognized_total:
        other = total - recognized_total
        lines.append(f"Retrieved {other:,} other evidence {'record' if other == 1 else 'records'}.")
    if not lines and total is not None:
        lines.append(f"Retrieved {total:,} evidence {'record' if total == 1 else 'records'}.")
    return tuple(lines)


def _public_retrieval_progress(value: Any) -> dict[str, Any]:
    """A bounded public progress contract; no URL, variant payload or error text."""
    if not isinstance(value, Mapping):
        return {}
    phases = {"ld_source_identity", "ld_source_discovery", "ld_identity_batch", "ld_proxy_lookup", "ld_proxy_identity", "ld_population_identity",
              "ld_expansion", "ld_recovery", "ld_reference_request", "catalog_request", "semantic_variant_batch", "semantic_tool", "retrieval_recovery", "result_processing", "retrieval_phase", "v2g_retrieval"}
    if not isinstance(value.get("phase"), str) or not isinstance(value.get("status"), str):
        return {}
    if value.get("phase") not in phases or value.get("status") not in {
        "started", "running", "waiting", "completed", "incomplete", "failed", "retrying", "cached", "splitting", "blocked", "not_run", "unavailable"}:
        return {}
    result = {"phase": value["phase"], "status": value["status"]}
    if value["phase"] == "v2g_retrieval":
        if value.get("name") not in {"phenotype", "candidates", "gene_profiles", "variants", "colocalisation", "associations", "cards"}:
            return {}
        result["name"] = value["name"]
        if value.get("timing_basis") == "backend_monotonic":
            result["timing_basis"] = "backend_monotonic"
    if isinstance(value.get("operation_id"), str) and re.fullmatch(r"[a-f0-9]{32}", value["operation_id"]):
        result["operation_id"] = value["operation_id"]
    if (value.get("phase") == "semantic_tool" and value.get("status") == "completed"
            and value.get("tool") == "ot_get_target_profiles"
            and value.get("coverage_note") == "bounded_literature"):
        result["coverage_note"] = "bounded_literature"
    from ..mcp.catalog import TOOL_SPECS
    public_modules = {module for spec in TOOL_SPECS.values() for module in spec.modules}
    if isinstance(value.get("module"), str) and value["module"] in public_modules:
        result["module"] = value["module"]
    if isinstance(value.get("modules"), (list, tuple)):
        result["modules"] = list(dict.fromkeys(module for module in value["modules"][:64]
            if isinstance(module, str) and module in public_modules))
    if isinstance(value.get("module_statuses"), Mapping):
        allowed_states = {"RUNNING", "SUCCESS_DATA", "SUCCESS_NO_DATA", "PARTIAL", "FAILED",
                          "NOT_REQUESTED", "NOT_APPLICABLE"}
        result["module_statuses"] = {module: state for module, state in value["module_statuses"].items()
            if module in public_modules and isinstance(state, str) and state in allowed_states}
    if isinstance(value.get("module_counts"), Mapping):
        result["module_counts"] = {module: count for module, count in value["module_counts"].items()
            if module in public_modules and type(count) is int and 0 <= count <= 10_000_000}
    if "reference_provider" in value:
        if not isinstance(value["reference_provider"], str) or value["reference_provider"] not in {"ldlink", "ensembl", "opentargets"}:
            return {}
        if value["reference_provider"] == "opentargets" and value["phase"] not in {
            "ld_reference_request", "ld_identity_batch", "ld_source_identity", "ld_source_discovery",
            "ld_proxy_identity", "ld_population_identity",
        }:
            return {}
        result["reference_provider"] = value["reference_provider"]
    if "provider_version" in value:
        if ("reference_provider" not in result or not isinstance(value["provider_version"], str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", value["provider_version"])):
            return {}
        result["provider_version"] = value["provider_version"]
    if value.get("name") in {"discovery", "shortlist", "colocalisation_details"}:
        result["name"] = value["name"]
    if value.get("state") in {"RUNNING", "SUCCESS_DATA", "SUCCESS_WITH_DATA", "SUCCESS_NO_DATA", "PARTIAL", "FAILED", "NOT_REQUESTED", "NOT_APPLICABLE"}:
        result["state"] = value["state"]
    for key in ("context_sha256", "shortlist_sha256", "scope_sha256"):
        if isinstance(value.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", value[key]):
            result[key] = value[key]
    if value.get("identity_scope") in {"source", "proxy"}:
        result["identity_scope"] = value["identity_scope"]
    if type(value.get("populations_requested")) is bool:
        result["populations_requested"] = value["populations_requested"]
    if value.get("operation") in {"organizing", "saving", "verifying", "loading", "saving_raw",
                                  "normalizing", "checking_policy", "projecting", "saving_response"}:
        result["operation"] = value["operation"]
    if type(value.get("bytes_written")) is int and 0 <= value["bytes_written"] <= 2**53 - 1:
        result["bytes_written"] = value["bytes_written"]
    for key in ("source_count", "completed_count", "total_count", "validated_count", "pending_count",
                "batch_index", "batch_count", "variant_count", "total_analysis_variants", "records", "elapsed_seconds", "step", "attempt", "page_size", "offset", "wait_seconds",
                "network_requests", "cache_hits", "failed_cache_hits", "completed_pages", "failed_pages",
                "successful_requests", "failed_items", "partial_items",
                "failed_request_pages", "unique_request_pages", "unique_credible_sets", "unique_studies",
                "unique_connections", "batch_size", "returned_count", "verified_proxy_count",
                "previous_validated_count", "newly_validated_count", "proxy_count", "candidate_count"):
        number = value.get(key)
        if isinstance(number, int) and not isinstance(number, bool) and 0 <= number <= 10_000_000:
            result[key] = number
    tool = value.get("tool")
    reason = value.get("reason")
    if isinstance(reason, str) and reason in {"ld_credential_unavailable", "service_error", "rate_limited", "request_timeout", "invalid_response", "connection_error", "query_too_expensive", "partial_page", "request_budget", "continuation_budget_exhausted", "invalid_or_unavailable_response", "aggregate_request_too_large", "response_size_limit", "reference_batch_unavailable", "cooldown_exceeds_budget", "transport_or_deadline_gap", "source_identity_unavailable", "source_identity_unresolved", "source_input_invalid", "source_selection_missing", "external_acquisition_unavailable"}:
        result["reason"] = reason
    for key, allowed in (("channel", {"expression", "colocalisation", "transport", "entities", "credible_sets"}), ("unit", {"genes", "records", "pages", "entities", "requests", "variants", "phenotypes", "credible_sets"})):
        if value.get(key) in allowed:
            result[key] = value[key]
    if isinstance(tool, str) and tool in {"ot_get_capabilities", "ot_resolve_entities", "ot_get_locus_context", "ot_get_locus_gene_evidence",
                "ot_get_target_profiles", "ot_get_target_disease_evidence", "ot_get_disease_profiles",
                "ot_get_drug_evidence", "ot_analyze_gene_set"}:
        result["tool"] = tool
    for count, total in (("completed_count", "total_count"), ("validated_count", "total_count"),
                         ("batch_index", "batch_count"), ("variant_count", "total_analysis_variants")):
        if count in result and total in result and result[count] > result[total]:
            return {}
    return result


def _completed_bounded_literature_tool(details: Mapping[str, Any]) -> bool:
    """Recognize a completed bounded request without erasing real retrieval gaps."""
    from ..review_coverage import _bounded_literature

    if details.get("tool") != "ot_get_target_profiles" or details.get("status") != "partial":
        return False
    coverage = details.get("coverage")
    if not isinstance(coverage, Mapping):
        return False
    groups = {}
    for name in ("requested", "completed", "no_data", "failed", "blocked", "unavailable", "truncated"):
        values = coverage.get(name)
        if (not isinstance(values, (list, tuple))
                or any(not isinstance(value, str) or not value for value in values)):
            return False
        groups[name] = set(values)
    if (not groups["requested"] or groups["failed"] or groups["blocked"] or groups["unavailable"]
            or not groups["truncated"] or not groups["truncated"] <= {"evidence.europepmc", "target.literature"}
            or not groups["truncated"] <= groups["requested"]
            or not groups["requested"] <= groups["completed"] | groups["no_data"]):
        return False
    return _bounded_literature(details.get("warnings"))


def _worker_retrieval_progress(event: Mapping[str, Any]) -> dict[str, Any]:
    if event.get("type") != "progress" or not isinstance(event.get("details"), Mapping):
        return {}
    phase, details = event.get("phase"), event["details"]
    if (details.get("progress_scope") == "workflow_step"
            or phase in {"candidate_expression_started", "candidate_expression_finished"}
            or phase == "semantic_request_finished" and details.get("operation_id")):
        return {}
    if phase in {"retrieval_recovery", "retrieval_phase", "v2g_retrieval"}:
        if phase == "retrieval_phase" and not details.get("status"):
            details = {**details, "status": "running" if details.get("state") == "RUNNING"
                       else "incomplete" if details.get("state") in {"FAILED", "PARTIAL"} else "completed"}
        return _public_retrieval_progress({**details, "phase": phase})
    if not isinstance(phase, str) or phase not in {"semantic_call_started", "semantic_call_finished", "semantic_tool_started", "semantic_tool_finished", "semantic_tool_waiting",
                     "semantic_variant_batch_started", "semantic_variant_batch_finished", "semantic_request_finished",
                     "candidate_expression_started", "candidate_expression_finished"}:
        return {}
    batch = phase.startswith("semantic_variant_batch") or bool(details.get("batch_index")) and not phase.startswith("semantic_call")
    status = "waiting" if phase.endswith("waiting") else "started" if phase.endswith("started") else (
        "completed" if details.get("status") in {"success", "no_data"} else
        "failed" if details.get("status") in {"error", "failed"} else
        "unavailable" if details.get("status") == "unavailable" else
        "blocked" if details.get("status") == "blocked" else "incomplete")
    coverage = details.get("coverage")
    extra = {}
    if phase.endswith("finished") and isinstance(coverage, Mapping):
        extra["modules"] = coverage.get("requested", [])
        module_states = {}
        for name, state in (("completed", "SUCCESS_DATA"), ("no_data", "SUCCESS_NO_DATA"),
                            ("unavailable", "FAILED"), ("blocked", "FAILED"),
                            ("failed", "FAILED"), ("truncated", "PARTIAL")):
            members = coverage.get(name)
            if isinstance(members, (list, tuple)):
                module_states.update({module: state for module in members if isinstance(module, str)})
        extra["module_statuses"] = module_states
    if phase in {"semantic_tool_finished", "semantic_call_finished"} and _completed_bounded_literature_tool(details):
        status = "completed"
        extra["coverage_note"] = "bounded_literature"
        # Public completion describes the configured retrieval scope. The
        # original receipt still records the full database list as truncated.
        for module in coverage["truncated"]:
            extra["module_statuses"][module] = "SUCCESS_NO_DATA" if module in coverage["no_data"] else "SUCCESS_DATA"
    return _public_retrieval_progress({**details, **extra,
        "phase": "semantic_variant_batch" if batch else "semantic_tool", "status": status})


def _worker_result_processing_progress(event: Mapping[str, Any]) -> dict[str, Any]:
    """Expose local evidence work separately from waiting for a data provider."""
    phase = event.get("phase")
    details = event.get("details")
    details = details if isinstance(details, Mapping) else {}
    operation = {"writing_artifacts": "saving"}.get(phase)
    if phase == "result_processing":
        status = details.get("status")
        if status not in {"started", "running", "completed", "organizing", "saving", "verifying", "importing", "loading"}:
            return {}
        operation = details.get("operation") or {"importing": "organizing", "completed": "verifying"}.get(status, status)
        if operation not in {"organizing", "saving", "verifying", "loading", "saving_raw",
                              "normalizing", "checking_policy", "projecting", "saving_response"}:
            operation = "organizing"
    elif operation is None:
        return {}
    value = {"phase": "result_processing", "status": "completed" if phase == "result_processing"
        and details.get("status") == "completed" else "running", "operation": operation,
        "records": details.get("records"), "bytes_written": details.get("bytes_written")}
    if "total_records" in details:
        value.update(completed_count=details.get("records"), total_count=details.get("total_records"), unit="records")
    return _public_retrieval_progress(value)


def _worker_sufficiency_seconds(events: list[Mapping[str, Any]]) -> float | None:
    """Measure paired worker checks, never polling delay or missing timestamps.

    These checks precede saved-artifact loading, so they cannot move the live
    workflow past retrieval yet. Account for their observed duration once when
    the completed evidence enters the full scientific review.
    """
    pending: dict[tuple[str, int], datetime] = {}
    intervals: set[tuple[datetime, datetime]] = set()
    for event in events:
        phase = event.get("phase")
        if phase not in {"checking_sufficiency", "safety_gate_passed", "safety_gate_stopped"}:
            continue
        run_id, query_index = event.get("child_run_id"), event.get("query_index")
        stamp = event.get("child_timestamp")
        if (not isinstance(run_id, str) or not run_id or type(query_index) is not int
                or query_index < 0 or not isinstance(stamp, str) or len(stamp) > 80):
            continue
        try:
            timestamp = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            continue
        key = (run_id, query_index)
        if phase == "checking_sufficiency":
            # Input-preflight emits its notification after evaluating input;
            # that notification cannot establish the check's start time.
            details = event.get("details")
            if isinstance(details, Mapping) and details.get("stage") == "input":
                continue
            pending.setdefault(key, timestamp)
        else:
            start = pending.pop(key, None)
            if start is not None and timestamp >= start:
                intervals.add((start, timestamp))
    if not intervals:
        return None
    # Duplicate delivery and overlapping worker intervals must not inflate wall
    # time. The ordinary Chat workflow has a single retrieval worker.
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum((end - start).total_seconds() for start, end in merged)


def _retrieval_progress_detail(progress: Mapping[str, Any]) -> str:
    if progress.get("status") in {"blocked", "not_run"}:
        return {
            "source_identity_unresolved": "Requested LD lookup could not start because no source SNP identity was resolved.",
            "source_identity_unavailable": "Requested LD lookup could not start because source SNP identity is unavailable.",
            "source_input_invalid": "Requested LD lookup could not start because the source input is invalid.",
            "source_selection_missing": "Requested LD lookup could not start because no source SNP was selected.",
        }.get(progress.get("reason"), "This requested retrieval operation could not start.")
    if progress.get("phase") == "ld_expansion" and progress.get("reason") == "ld_credential_unavailable":
        return "LD expansion was not performed: configure the LDlink API connection in Settings."
    labels = {"ld_source_identity": "Checking selected SNP identities", "ld_proxy_lookup": "Retrieving population LD proxies",
              "ld_source_discovery": "Matching selected SNP coordinates to the reference",
              "ld_identity_batch": "Loading reference alleles and population metadata",
              "ld_proxy_identity": "Verifying LD proxy identities", "ld_population_identity": "Checking population-specific proxy alleles",
              "ld_expansion": "LD expansion", "ld_reference_request": "Querying the Ensembl reference",
              "ld_recovery": "Completing unfinished LD reference checks",
              "catalog_request": "Querying GWAS Catalog", "semantic_variant_batch": "Retrieving evidence batch",
              "semantic_tool": "Retrieving evidence", "retrieval_recovery": "Recovering evidence queries", "result_processing": "Loading and verifying the complete evidence ledger"}
    text = labels.get(progress.get("phase"), "Retrieving evidence")
    if progress.get("phase") == "ld_reference_request" and progress.get("reference_provider") == "ldlink":
        text = "Querying the LDlink reference"
    if progress.get("phase") == "retrieval_phase":
        text = {"discovery": "Discovering candidate evidence", "shortlist": "Fixing the candidate shortlist",
                "colocalisation_details": "Retrieving shortlisted molecular colocalisation"}.get(progress.get("name"), text)
    if progress.get("phase") == "result_processing":
        text = {"organizing": "Organizing retained evidence", "saving": "Saving the complete analysis record",
                "verifying": "Verifying the retained evidence", "loading": "Loading retained evidence",
                "saving_raw": "Saving original evidence", "normalizing": "Organizing evidence records",
                "checking_policy": "Checking evidence eligibility", "projecting": "Preparing analysis evidence",
                "saving_response": "Saving evidence for review"}.get(progress.get("operation"), text)
        if "bytes_written" in progress:
            text += f" · {progress['bytes_written'] / (1024 * 1024):,.1f} MiB written"
    if progress.get("phase") == "ld_identity_batch" and type(progress.get("populations_requested")) is bool:
        if progress["populations_requested"]:
            text = ("Resolving remaining proxy alleles in the reference population" if progress.get("identity_scope") == "proxy"
                    else "Checking reference-population data for selected SNPs")
        else:
            text = "Checking LD proxy identity records" if progress.get("identity_scope") == "proxy" else "Checking selected SNP identity records"
    if progress.get("reference_provider") == "opentargets":
        text = "Open Targets variant identity"
    if progress.get("phase") == "retrieval_recovery":
        text = {"expression": "Retrieving gene expression", "colocalisation": "Completing colocalization pages",
                "transport": "Retrieving shared evidence requests", "entities": "Retrieving shared study evidence",
                "credible_sets": "Identifying credible sets linked to the locus"}.get(progress.get("channel"), text)
    if progress.get("phase") == "semantic_tool" and progress.get("tool") == "ot_get_capabilities":
        if progress.get("status") in {"failed", "unavailable", "incomplete"}:
            return "Open Targets connection could not be verified; evidence availability remains unknown."
        return "Open Targets connection and available interfaces checked." if progress.get("status") == "completed" else "Connecting to Open Targets and checking available interfaces..."
    if progress.get("batch_count"):
        text += f" {progress.get('batch_index', 0)}/{progress['batch_count']}"
    if "completed_count" in progress and "total_count" in progress:
        text += f": {progress['completed_count']}/{progress['total_count']}" + (f" {progress['unit']}" if progress.get("unit") else "") + " checked"
    if "validated_count" in progress:
        text += f", {progress['validated_count']} validated"
    if progress.get("phase") == "ld_recovery" and "verified_proxy_count" in progress:
        text += f", {progress['verified_proxy_count']} verified LD proxies retained"
    if "variant_count" in progress:
        text += f" ({progress['variant_count']} variants)"
    if "records" in progress:
        text += f", {progress['records']} records returned"
    if "attempt" in progress:
        text += f", request attempt {progress['attempt']}"
    suffix = {"waiting": "waiting for the external service", "retrying": "retrying the request",
              "completed": "completed", "incomplete": "finished with incomplete coverage",
              "failed": "request failed", "unavailable": "external service unavailable",
              "cached": "reusing a recorded reference response", "splitting": "reducing the request size"}.get(progress.get("status"), "in progress")
    reason = {"service_error": "reference service error", "rate_limited": "reference service rate limit",
              "request_timeout": "request timed out", "invalid_response": "invalid reference response",
              "connection_error": "connection error", "query_too_expensive": "query complexity limit; splitting into smaller requests",
              "partial_page": "incomplete page", "request_budget": "recovery limit reached", "continuation_budget_exhausted": "recovery limit reached",
              "invalid_or_unavailable_response": "an incomplete service response",
              "aggregate_request_too_large": "splitting the request while retaining completed results",
              "response_size_limit": "reducing the page size while retaining completed results",
              "reference_batch_unavailable": "the reference batch did not complete",
              "cooldown_exceeds_budget": "the requested service wait exceeded the remaining retrieval time",
              "transport_or_deadline_gap": "an unfinished reference request; completed results are reused"}.get(progress.get("reason"))
    if reason:
        suffix += " after " + reason
    return text + " — " + suffix + (f" ({progress['elapsed_seconds']}s elapsed)." if "elapsed_seconds" in progress else ".")


def _public_chat_job_statuses(event: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    """Map one worker event to one or more append-only public status lines."""

    if event.get("type") == "progress" and event.get("phase") == "semantic_variant_batch_started":
        details = event.get("details")
        if not isinstance(details, Mapping):
            return ()
        names = ("batch_index", "batch_count", "variant_count", "total_analysis_variants")
        values = [details.get(name) for name in names]
        if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1_000_000 for value in values):
            return ()
        index, count, size, total = values
        if index > count or size > total:
            return ()
        return ((55, f"Retrieving evidence batch {index}/{count}: {size} variants ({total} analysis variants in total)."),)
    if str(event.get("type", "")) == "progress" and str(event.get("phase", "")) == (
        "semantic_tool_finished"
    ):
        details = event.get("details")
        details = details if isinstance(details, Mapping) else {}
        step = details.get("step")
        step_number = step if isinstance(step, int) and not isinstance(step, bool) else 0
        stage = 50 + min(max(step_number, 0) * 5, 20)
        return tuple((stage, line) for line in _public_semantic_record_count_lines(details))
    status = _public_chat_job_status(event)
    return (status,) if status is not None else ()


def _public_stage_metrics(metrics: Any) -> dict[str, int]:
    """Project runtime retrieval counts onto the browser's strict public keys.

    L2G is deliberately omitted here because Stage 2 already names it once when
    explaining shortlist construction. It remains fully recorded in local run
    artifacts and the bridge audit payload.
    """

    module_counts = dict(getattr(metrics, "module_record_counts", ()))
    mapping = {
        "locus.variant": "variant_annotations",
        "locus.credible_set": "credible_sets",
        "locus.variant_consequence": "variant_consequences",
        "locus.qtl": "molecular_qtl",
        "locus.colocalisation": "colocalization",
        "locus.enhancer_to_gene": "enhancer_links",
        "target.expression": "gene_expression",
        "target.function": "gene_function",
        "target.pathway": "pathways",
    }
    public: dict[str, int] = {}
    for module, key in mapping.items():
        count = module_counts.get(module)
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            public[key] = count
    total = getattr(metrics, "evidence_records", None)
    candidates = getattr(metrics, "candidate_genes", None)
    publications = getattr(metrics, "literature_records", None)
    publication_abstracts = getattr(metrics, "publication_abstracts", None)
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        public["total_records"] = total
    if isinstance(candidates, int) and not isinstance(candidates, bool) and candidates >= 0:
        public["candidate_genes"] = candidates
    if isinstance(publications, int) and not isinstance(publications, bool) and publications > 0:
        public["publications"] = publications
    if (
        isinstance(publication_abstracts, int)
        and not isinstance(publication_abstracts, bool)
        and publication_abstracts > 0
    ):
        public["publication_abstracts"] = publication_abstracts
    return public


def _safe_gate_status_message(status: str) -> str:
    messages = {
        "NO_GWAS_SIGNAL": "the selected variants did not pass the GWAS-signal requirement",
        "INSUFFICIENT_EVIDENCE": "the permitted evidence was insufficient",
        "CONFLICTING_EVIDENCE": "the retrieved evidence contained unresolved conflicts",
        "INVALID_INPUT": "the analysis input did not pass validation",
        "POLICY_BLOCKED": "the configured data-safety policy blocked this analysis",
        "SERVICE_UNAVAILABLE": "a required evidence service was unavailable",
        "INCOMPLETE": "the analysis did not complete its required validation",
    }
    return messages.get(status, "the evidence requirements were not satisfied")


def _public_analysis_scope_metrics(outcome: Mapping[str, Any]) -> dict[str, int]:
    """Publish known scientific scopes; unchecked proxy counts are not evidence."""
    observations = outcome.get("source_observations")
    if not isinstance(observations, (list, tuple)):
        return {}
    from ..report_facts import computed_report_facts
    value = outcome.get("result")
    result = dict(value) if isinstance(value, Mapping) else {}
    plan = outcome.get("ld_expansion")
    if isinstance(plan, Mapping) and "ld_expansion" not in result:
        result["ld_expansion"] = plan
    facts = computed_report_facts(observations, result)
    keys = {"source_variant_count", "verified_proxy_count", "analysis_variant_count",
            "candidate_pool_count", "shortlist_count"}
    ld_plan = result.get("ld_expansion")
    if isinstance(ld_plan, Mapping) and ld_plan.get("ld_lookup_performed") is False:
        keys.discard("verified_proxy_count")
    if isinstance(result.get("ranked_genes"), (list, tuple)):
        keys.update({"ranked_count", "displayed_ranked_count"})
    if outcome.get("requested_task") == ChatIntent.VARIANT_DESCRIPTION.value:
        keys.difference_update({"candidate_pool_count", "shortlist_count", "ranked_count", "displayed_ranked_count"})
    return {key: facts[key] for key in keys if type(facts.get(key)) is int
            and 0 <= facts[key] <= 10_000_000}


def _public_safety_stop_detail(safety: Any) -> str:
    """Map the first immutable hard-gate failure to a fixed public message."""

    review = getattr(safety, "review", None)
    facts = getattr(review, "facts", None)
    checks = getattr(facts, "checks", ())
    for check in checks:
        severity = getattr(getattr(check, "severity", None), "value", "")
        outcome = getattr(getattr(check, "outcome", None), "value", "")
        check_id = str(getattr(check, "check_id", "")).strip()
        if severity != "hard_gate" or outcome != "fail":
            continue
        messages = {
            "variant_integrity": (
                "Safety gate stopped: variant identifier integrity could not be verified."
            ),
            "build_integrity": (
                "Safety gate stopped: genome-build integrity could not be verified."
            ),
            "allele_integrity": ("Safety gate stopped: allele integrity could not be verified."),
            "required_modules": (
                "Safety gate stopped: required retrieval modules did not complete."
            ),
            "gwas_signal": (
                "Safety gate stopped: selected variants did not pass the GWAS-signal requirement."
            ),
            "locus_genes_1mb": (
                f"Safety gate stopped: annotated-gene coverage within {local_gene_window_label()} was insufficient."
            ),
            "direct_locus_gene_evidence": (
                "Safety gate stopped: no direct locus-to-gene evidence was found."
            ),
            "model_output_grounding": (
                "Safety gate stopped: the candidate-ranking output failed validation."
            ),
        }
        return messages.get(
            check_id,
            "Safety gate stopped: a required validation check failed.",
        )
    return "Safety gate stopped: a required validation check failed."


def _public_chat_job_status(event: Mapping[str, Any]) -> tuple[int, str] | None:
    """Map private worker telemetry onto a small, monotonic public vocabulary."""

    event_type = str(event.get("type", ""))
    if event_type in {"queued", "started", "query_started"}:
        return 30, "Planning the evidence retrieval..."
    if event_type != "progress":
        return None

    phase = str(event.get("phase", ""))
    details = event.get("details")
    if not isinstance(details, Mapping):
        details = {}
    if phase in {"validating_input", "semantic_input_validation", "capability_check"}:
        return 32, "Validating the selected variants and data access..."
    if phase == "planning_evidence":
        return 35, "Planning the evidence retrieval..."
    if phase in {"retrieving_evidence", "semantic_plan_ready"}:
        return 40, "Preparing the requested Open Targets evidence queries..."
    if phase == "semantic_tool_started":
        modules_value = details.get("modules")
        modules = modules_value if isinstance(modules_value, list) else []
        labels = [
            _PUBLIC_MODULE_LABELS[str(module)]
            for module in modules
            if str(module) in _PUBLIC_MODULE_LABELS
        ]
        tool = str(details.get("tool", ""))
        if not labels:
            labels = list(_PUBLIC_TOOL_DATA.get(tool, ("the requested evidence",)))
        step = details.get("step")
        step_number = step if isinstance(step, int) and not isinstance(step, bool) else 0
        return 50 + min(max(step_number, 0) * 5, 20), f"Fetching {_public_list(labels)}..."
    if phase == "semantic_tool_finished":
        message = _public_semantic_record_counts(details)
        if message is None:
            return None
        step = details.get("step")
        step_number = step if isinstance(step, int) and not isinstance(step, bool) else 0
        return 50 + min(max(step_number, 0) * 5, 20), message
    if phase == "candidate_universe_discovered":
        targets = details.get("targets")
        count = len(targets) if isinstance(targets, list) else 0
        if 0 < count <= MAX_CANDIDATE_GENES_PER_QUERY:
            return 62, f"Identified {count} candidate genes; preparing target profiles..."
        return 62, "Preparing the candidate-gene target profiles..."
    if phase == "checking_sufficiency":
        return 75, "Organizing and validating the retrieved evidence..."
    if phase == "safety_gate_passed":
        decision_records = details.get("decision_evidence_records")
        sources = details.get("evidence_sources")
        if (
            isinstance(decision_records, int)
            and not isinstance(decision_records, bool)
            and decision_records >= 0
            and isinstance(sources, int)
            and not isinstance(sources, bool)
            and sources >= 0
        ):
            return (
                78,
                "Safety gate passed: "
                f"{decision_records:,} decision-eligible evidence "
                f"{'record' if decision_records == 1 else 'records'} across "
                f"{sources:,} traceable source {'identifier' if sources == 1 else 'identifiers'}.",
            )
        return 78, "Safety gate passed; preparing the evidence-grounded response."
    if phase == "safety_gate_stopped":
        status = str(details.get("status", "")).upper()
        return 78, f"Safety gate stopped: {_safe_gate_status_message(status)}."
    if phase == "synthesizing_decision":
        total = details.get("total_evidence_records")
        candidates = details.get("candidate_genes")
        if (
            isinstance(total, int)
            and not isinstance(total, bool)
            and total >= 0
            and isinstance(candidates, int)
            and not isinstance(candidates, bool)
            and candidates >= 0
        ):
            return (
                80,
                f"Evidence ready: {total:,} audited records. Prioritizing "
                f"{candidates:,} shortlisted {'gene' if candidates == 1 else 'genes'} "
                "from a compact deduplicated set...",
            )
        return 80, "Evidence ready. Prioritizing shortlisted genes..."
    if phase == "synthesis_completed":
        genes = details.get("ranked_genes")
        output_tokens = details.get("output_tokens")
        if details.get("ranking_unavailable") is True or genes == 0:
            return 82, RANKING_UNAVAILABLE_DETAIL
        message = (
            f"Candidate ranking completed for {genes:,} {'gene' if genes == 1 else 'genes'}"
            if isinstance(genes, int) and not isinstance(genes, bool) and genes >= 0
            else "Candidate ranking completed"
        )
        if (
            isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens >= 0
        ):
            message += f" · {output_tokens:,} model output tokens"
        if details.get("validation_passed") is False:
            message += " · some facts need review"
        return 82, message + "."
    if phase == "writing_artifacts":
        status = str(details.get("status", "")).upper()
        if status not in {"PRIORITIZED", "EVIDENCE_RETRIEVED"}:
            return 85, "Preparing the safe abstention response..."
        return 85, "Preparing the final evidence-grounded response..."
    return None


def _is_loopback_name(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.casefold().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_bind_host(host: str) -> None:
    if not _is_loopback_name(host):
        raise ValueError("the web server may bind only to a loopback address")


class GWASMindHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: JobManager,
        *,
        static_root: str | Path | None = None,
        owns_manager: bool = False,
        heartbeat_interval: float = parameter("web.heartbeat_interval_seconds"),
        ui_state: UiState | None = None,
        analysis_bridge: AnalysisBridge | None = None,
        history_store: LocalHistoryStore | None = None,
        default_analysis_mode: str = "live",
        model_catalog: PublicModelCatalog | None = None,
        refresh_public_models: bool = False,
    ):
        _validate_bind_host(server_address[0])
        self.configuration_provenance = configuration_provenance()
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if default_analysis_mode not in {"live", "frozen-safe"}:
            raise ValueError("default_analysis_mode must be live or frozen-safe")
        if default_analysis_mode == "frozen-safe":
            try:
                manager.frozen_snapshot_defaults()
            except JobValidationError as exc:
                raise ValueError(
                    f"frozen-safe graphical mode requires a verified snapshot manifest: {exc}"
                ) from exc
        self.manager = manager
        from .context_planner import ContextPlanner
        self.context_planner = ContextPlanner()
        from .phenotype_planner import PhenotypeScopePlanner
        from .phenotype_scope_store import PhenotypeScopeStore
        artifact_root = getattr(manager, "artifact_root", None)
        self.model_catalog = model_catalog or PublicModelCatalog(
            cache_path=Path(artifact_root).parent / "public_model_catalog.json"
            if artifact_root is not None else None
        )
        self.phenotype_scope_store = (PhenotypeScopeStore(Path(artifact_root).parent / "phenotype_scopes")
                                      if artifact_root is not None else None)
        self.phenotype_scope_planner = PhenotypeScopePlanner(store=self.phenotype_scope_store)
        from .ldlink_settings import LDLinkSettings
        self.ldlink_settings = LDLinkSettings()
        # Set only by main() on Windows: where a verified Settings configuration is
        # kept with user-scoped DPAPI encryption so it survives a restart.
        self.model_configuration_path: Path | None = None
        self.static_root = (
            Path(static_root).resolve()
            if static_root is not None
            else Path(__file__).with_name("static").resolve()
        )
        self.owns_manager = owns_manager
        self.heartbeat_interval = heartbeat_interval
        self._ui_state_lock = threading.Lock()
        self.ui_state = ui_state or UiState()
        self._history_store = history_store
        self._history_store_lock = threading.Lock()
        self.analysis_bridge = analysis_bridge or AnalysisBridge()
        self._auto_history_ids: dict[str, str] = {}
        self.auto_analysis = AutoAnalysisManager(
            bridge=self.analysis_bridge,
            workflow_factory=lambda **kwargs: AnalysisWorkflow(self, **kwargs),
            cancel_run=self.manager.cancel,
            save_result=self._save_auto_history,
        )
        self.analysis_bridge.set_run_terminal_resolver(self._run_is_terminal)
        self.default_analysis_mode = default_analysis_mode
        super().__init__(server_address, GWASMindRequestHandler)
        if refresh_public_models:
            self.model_catalog.start()

    @property
    def history_store(self) -> LocalHistoryStore:
        with self._history_store_lock:
            if self._history_store is None:
                self._history_store = LocalHistoryStore(
                    default_history_root(getattr(self.manager, "project_root", None))
                )
        return self._history_store

    def publish_ui_state(self, state: UiState, *, expected_empty: UiState | None = None) -> bool:
        """Publish complete plot state without overwriting a concurrent explicit load."""
        with self._ui_state_lock:
            if expected_empty is not None and (
                self.ui_state is not expected_empty or self.ui_state.has_data()["has_data"]
            ):
                return False
            self.ui_state = state
            return True

    def _run_is_terminal(self, run_id: str) -> bool:
        if self.auto_analysis.has_task(run_id):
            return self.auto_analysis.snapshot(run_id)["state"] in TERMINAL_STATES
        try:
            return self.manager.snapshot(run_id)["state"] in TERMINAL_STATES
        except JobNotFoundError:
            # JobManager has already evicted the history entry, so it cannot be
            # an active child and the legacy task mapping is no longer useful.
            return True

    def _save_auto_history(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        history = getattr(self, "history_store", None)
        if history is not None:
            task_id = str(payload["task_id"])
            upload = self.analysis_bridge.upload(task_id)
            saved = history.save_analysis(
                {**payload, "filename": upload.filename,
                 "snp_count": upload.snp_count, "locus_count": len(upload.queries),
                 "snp_positions": upload.snp_positions, "warnings": upload.warnings,
                 "target_phenotype": payload.get("source_phenotype", ""),
                 **({"dataset_id": upload.dataset_id} if upload.dataset_id else {})},
                history_id=self._auto_history_ids.get(task_id),
                preserve_discussions=True,
            )
            self._auto_history_ids[task_id] = saved["history_id"]
            return saved
        return None

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Ignore ordinary browser disconnects without printing a traceback."""

        if isinstance(sys.exception(), _CLIENT_DISCONNECT_ERRORS):
            return
        super().handle_error(request, client_address)

    def analysis_request_defaults(self) -> dict[str, Any]:
        if self.default_analysis_mode == "live":
            return {"mode": "live"}
        return {
            "mode": "frozen-safe",
            **self.manager.frozen_snapshot_defaults(),
        }

    def server_close(self) -> None:
        try:
            self.model_catalog.close()
            self.auto_analysis.close()
            super().server_close()
        finally:
            if self.owns_manager:
                self.manager.close()


class GWASMindRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "GWASMindWeb/1.0"
    sys_version = ""

    @property
    def app_server(self) -> GWASMindHTTPServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:  # noqa: N802
        if not self._request_is_local():
            return
        path = urlsplit(self.path).path
        try:
            if path == f"{API_PREFIX}/ldlink-settings":
                self._send_json(HTTPStatus.OK, {"success": True, "ldlink": self.app_server.ldlink_settings.status()})
                return
            if path == f"{API_PREFIX}/phenotype-scopes":
                self._handle_phenotype_scopes("list", None, {})
                return
            if path.startswith("/api/history/"):
                self._get_history(path)
                return
            if path == f"{API_PREFIX}/health":
                health = self.app_server.manager.health()
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": bool(
                            health.get("worker_alive")
                            and health.get("accepting_jobs")
                            and health.get("artifact_root_ready")
                        ),
                        "service": "gwasmind-web",
                        "version": __version__,
                        **health,
                    },
                )
                return
            if path == f"{API_PREFIX}/config":
                self._send_json(HTTPStatus.OK, self._public_config())
                return
            if path == f"{API_PREFIX}/model-status":
                self._send_json(HTTPStatus.OK, self.app_server.manager.model_status())
                return
            if path == f"{API_PREFIX}/model-catalog":
                self._send_json(HTTPStatus.OK, self.app_server.model_catalog.snapshot())
                return
            if path == f"{API_PREFIX}/model-settings":
                self._send_json(HTTPStatus.OK, self.app_server.manager.model_status())
                return
            if path == f"{API_PREFIX}/runs":
                self._send_json(HTTPStatus.OK, {"runs": self.app_server.manager.list_runs()})
                return
            if path == "/api/has-data":
                self._send_json(HTTPStatus.OK, self.app_server.ui_state.has_data())
                return
            if path == "/api/db-status":
                selected_mode = self.app_server.default_analysis_mode
                frozen_defaults = (
                    self.app_server.manager.frozen_snapshot_defaults()
                    if selected_mode == "frozen-safe"
                    else None
                )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        # This endpoint reports the installed local bridge, not
                        # remote reachability.  Actual upstream errors are
                        # surfaced by each bounded analysis request.
                        "available": True,
                        "bridge_available": True,
                        "scope": "local_bridge",
                        "upstream_reachability": "not_checked",
                        "upstream_checked": False,
                        "mode": selected_mode,
                        "source": (
                            "official Open Targets API/MCP"
                            if selected_mode == "live"
                            else "local versioned Open Targets snapshot"
                        ),
                        "download_required": False,
                        "release": (frozen_defaults["release"] if frozen_defaults else "live"),
                        "snapshot_verified": bool(frozen_defaults),
                        "snapshot_manifest_verified": bool(frozen_defaults),
                        "snapshot_content_rehashed": False,
                        "snapshot_sha256": (
                            frozen_defaults["snapshot_hash"] if frozen_defaults else None
                        ),
                    },
                )
                return
            if path == "/api/gwas-data":
                self._send_json(HTTPStatus.OK, self.app_server.ui_state.plot_payload())
                return
            if path == "/demo_data/chr1_all_loci.csv":
                self._send_static("demo_data/chr1_all_loci.csv")
                return
            if path.startswith("/api/analysis/quick-stream/"):
                task_id = unquote(path.rsplit("/", 1)[-1])
                parameters = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                phenotype = parameters.get("phenotype", [""])[0]
                concurrency_values = parameters.get("concurrency")
                max_concurrency = parameter("execution.max_concurrent_runs")
                if concurrency_values is not None and (
                    len(concurrency_values) != 1
                    or concurrency_values[0] not in {str(i) for i in range(1, max_concurrency + 1)}
                ):
                    raise JobValidationError(f"concurrency must be an integer from 1 to {max_concurrency}")
                self._stream_legacy_analysis(
                    task_id, phenotype,
                    concurrency=int(concurrency_values[0]) if concurrency_values else None,
                )
                return
            if path.startswith("/api/analysis/quick-results/"):
                task_id = unquote(path.rsplit("/", 1)[-1])
                self._send_legacy_analysis_result(task_id)
                return
            if path.startswith("/api/analysis/quick-download/"):
                task_id = unquote(path.rsplit("/", 1)[-1])
                self._send_legacy_analysis_download(task_id)
                return

            parts = _route_parts(path)
            if len(parts) >= 2 and parts[0] == "runs":
                run_id = self._validated_run_id(parts[1])
                if len(parts) == 2 or (len(parts) == 3 and parts[2] == "status"):
                    self._send_json(HTTPStatus.OK, self.app_server.manager.snapshot(run_id))
                    return
                if len(parts) == 3 and parts[2] == "events":
                    self._stream_events(run_id)
                    return
                if len(parts) == 3 and parts[2] == "result":
                    self._send_json(HTTPStatus.OK, self.app_server.manager.result(run_id))
                    return
                if len(parts) == 3 and parts[2] == "artifact":
                    parameters = parse_qs(urlsplit(self.path).query)
                    try:
                        query_index = int(parameters.get("query", [""])[0])
                    except ValueError as exc:
                        raise JobValidationError("artifact query index must be an integer") from exc
                    name = parameters.get("name", [""])[0]
                    self._send_artifact(run_id, query_index, name)
                    return
                if len(parts) == 5 and parts[2] == "artifact":
                    try:
                        query_index = int(parts[3])
                    except ValueError as exc:
                        raise JobValidationError("artifact query index must be an integer") from exc
                    self._send_artifact(run_id, query_index, parts[4])
                    return

            if path == "/" or path == "/index.html":
                if path == "/" and self.app_server.ui_state.has_data()["has_data"]:
                    self._redirect("/app")
                    return
                self._send_page("index.html")
                return
            if path == "/app":
                initial_state = self.app_server.ui_state
                if not initial_state.has_data()["has_data"]:
                    parameters = parse_qs(urlsplit(self.path).query)
                    kind = parameters.get("history_kind", [""])[0]
                    history_id = parameters.get("history_id", [""])[0]
                    if kind not in {"chats", "analyses"} or not _RUN_ID.fullmatch(history_id):
                        self._redirect("/")
                        return
                    # A saved report may have no linked plotting dataset. It
                    # still needs the workspace to reopen without inventing data.
                    snapshot = (self.app_server.history_store.get_analysis(history_id) if kind == "analyses"
                                else self.app_server.history_store.get(kind, history_id))
                    if kind == "analyses" or snapshot.get("dataset_id"):
                        next_state, _ = self._history_plot_state(kind, history_id, snapshot)
                        self.app_server.publish_ui_state(next_state, expected_empty=initial_state)
                self._send_page("app.html", data_source=self.app_server.ui_state.data_source())
                return
            if path == "/analysis":
                self._send_page("analysis.html")
                return
            if path == "/tutorial":
                self._send_page("tutorial.html")
                return
            if path == "/contact":
                self._send_page("contact.html")
                return
            if path.startswith("/static/"):
                self._send_static(unquote(path.removeprefix("/static/")))
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
        except JobError as exc:
            self._send_error_safely(exc.status_code, exc.code, str(exc))
        except UiDataError as exc:
            self._send_error_safely(HTTPStatus.BAD_REQUEST, "invalid_gwas_data", str(exc))
        except HistoryNotFoundError as exc:
            self._send_error_safely(HTTPStatus.NOT_FOUND, "history_not_found", str(exc))
        except HistoryError as exc:
            self._send_error_safely(HTTPStatus.BAD_REQUEST, "invalid_history", str(exc))
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
        except Exception:
            self._send_error_safely(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "The local web service encountered an unexpected error.",
            )

    def do_POST(self) -> None:  # noqa: N802
        if not self._request_is_local():
            return
        path = urlsplit(self.path).path
        try:
            if path in {f"{API_PREFIX}/ldlink-settings", f"{API_PREFIX}/ldlink-test"}:
                self._handle_ldlink_settings(path, self._read_json_body(max_bytes=8192))
                return
            if path.startswith(f"{API_PREFIX}/phenotype-scopes/"):
                self._handle_phenotype_scopes("update", unquote(path.rsplit("/", 1)[-1]),
                                              self._read_json_body(max_bytes=16384))
                return
            if path.startswith("/api/history/"):
                self._post_history(path)
                return
            if path == "/api/chat/stream":
                self._stream_chat(self._read_json_body())
                return
            if path == f"{API_PREFIX}/model-list":
                request_value = self._read_json_body()
                allowed = {"provider_id", "api_base_url", "api_key"}
                unknown = set(request_value).difference(allowed)
                if unknown:
                    raise JobValidationError(
                        "unknown model-list fields: " + ", ".join(sorted(unknown))
                    )
                result = self.app_server.manager.list_provider_models(
                    provider_id=request_value.get("provider_id"),
                    base_url=request_value.get("api_base_url"),
                    api_key=request_value.get("api_key"),
                )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "success": True,
                        **result,
                    },
                )
                return
            if path == f"{API_PREFIX}/model-reasoning":
                request_value = self._read_json_body()
                if set(request_value).difference({"provider_id", "api_base_url", "model"}):
                    raise JobValidationError("unknown model-reasoning fields")
                result = self.app_server.manager.model_reasoning(
                    provider_id=request_value.get("provider_id"), base_url=request_value.get("api_base_url"),
                    model_id=request_value.get("model"),
                )
                self._send_json(HTTPStatus.OK, {"success": True, **result})
                return
            if path == f"{API_PREFIX}/model-settings":
                request_value = self._read_json_body()
                allowed = {
                    "provider_id",
                    "api_base_url",
                    "api_key",
                    "model",
                    "thinking_mode",
                    "reasoning_effort",
                }
                unknown = set(request_value).difference(allowed)
                if unknown:
                    raise JobValidationError(
                        "unknown model setting fields: " + ", ".join(sorted(unknown))
                    )
                provider_id = _validate_model_provider_request(request_value)
                status = self.app_server.manager.configure_runtime_model(
                    api_key=request_value.get("api_key"),
                    base_url=request_value.get("api_base_url"),
                    model_id=request_value.get("model"),
                    provider_id=provider_id,
                    thinking_mode=request_value.get("thinking_mode"),
                    reasoning_effort=request_value.get("reasoning_effort"),
                )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "success": True,
                        "response": (
                            "Configuration saved for this local session. "
                            "Use Save & Test before AI analysis."
                        ),
                        "model_status": status,
                    },
                )
                return
            if path == "/api/test-llm-connection":
                request_value = self._read_json_body()
                allowed = {
                    "provider_id",
                    "api_base_url",
                    "api_key",
                    "model",
                    "thinking_mode",
                    "reasoning_effort",
                }
                unknown = set(request_value).difference(allowed)
                if unknown:
                    raise JobValidationError(
                        "unknown model setting fields: " + ", ".join(sorted(unknown))
                    )
                provider_id = _validate_model_provider_request(request_value)
                result = self.app_server.manager.test_runtime_model_connection(
                    api_key=request_value.get("api_key"),
                    base_url=request_value.get("api_base_url"),
                    model_id=request_value.get("model"),
                    provider_id=provider_id,
                    thinking_mode=request_value.get("thinking_mode"),
                    reasoning_effort=request_value.get("reasoning_effort"),
                )
                _remember_model_configuration(self.app_server)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "success": True,
                        "response": (
                            "Connection verified with "
                            f"{result['observed_model']} ({result['latency_ms']} ms). "
                            "Reasoning depth was not verified by this connection test."
                        ),
                        **result,
                    },
                )
                return
            if path == f"{API_PREFIX}/runs":
                request_value = self._read_json_body()
                snapshot = self.app_server.manager.submit(request_value)
                run_id = snapshot["run_id"]
                self._send_json(
                    HTTPStatus.ACCEPTED,
                    {
                        "run_id": run_id,
                        "state": snapshot["state"],
                        "mode": snapshot["mode"],
                        "release": snapshot["release"],
                        "query_count": snapshot["query_count"],
                        "status_url": f"{API_PREFIX}/runs/{run_id}/status",
                        "events_url": f"{API_PREFIX}/runs/{run_id}/events",
                        "result_url": f"{API_PREFIX}/runs/{run_id}/result",
                        "cancel_url": f"{API_PREFIX}/runs/{run_id}/cancel",
                    },
                )
                return
            if path == "/upload-processed":
                request_value = self._read_json_body(max_bytes=MAX_UPLOAD_JSON_BYTES)
                next_state = UiState()
                status = next_state.load_processed(request_value)
                saved = self.app_server.history_store.save_dataset(
                    next_state.plot_payload()
                )
                next_state.set_history_dataset_id(saved["id"])
                # Publish rows, source phenotype and history identity together.
                # A failed history write must not replace the user's active data.
                self.app_server.publish_ui_state(next_state)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "success": True,
                        "message": f"Loaded {status['total_snps']:,} SNPs",
                        "history_id": saved["id"],
                        "redirect": "/app",
                    },
                )
                return
            if path in {"/load-demo/asthma", "/load-demo/mdd"}:
                self._reject_nonempty_body()
                name = path.rsplit("/", 1)[-1]
                next_state = UiState()
                status = next_state.load_demo(name)
                saved = self.app_server.history_store.save_dataset(
                    next_state.plot_payload(), data_source=name,
                )
                next_state.set_history_dataset_id(saved["id"])
                self.app_server.publish_ui_state(next_state)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "success": True,
                        "message": f"Loaded {status['total_snps']:,} demo SNPs",
                        "history_id": saved["id"],
                        "redirect": "/app",
                    },
                )
                return
            if path == "/clear-data":
                self._reject_nonempty_body()
                self.app_server.publish_ui_state(UiState())
                self._send_json(
                    HTTPStatus.OK,
                    {"success": True, "message": "Data cleared", "redirect": "/"},
                )
                return
            if path == "/api/analysis/quick-upload":
                request_value = self._read_json_body()
                allowed = {"filename", "text", "dataset_id"}
                unknown = set(request_value).difference(allowed)
                if unknown:
                    raise JobValidationError(
                        "unknown locus upload fields: " + ", ".join(sorted(unknown))
                    )
                dataset_id = request_value.get("dataset_id")
                if dataset_id is not None:
                    # Bind the browser's plotting context before asynchronous
                    # analysis can outlive a later dataset switch.
                    self.app_server.history_store.get_dataset(dataset_id)
                result = self.app_server.analysis_bridge.upload_text(
                    filename=request_value.get("filename"),
                    text=request_value.get("text"),
                    dataset_id=dataset_id,
                )
                self._send_json(HTTPStatus.OK, result)
                return
            if path.startswith("/api/analysis/quick-cancel/"):
                self._reject_nonempty_body()
                task_id = unquote(path.rsplit("/", 1)[-1])
                if self.app_server.auto_analysis.has_task(task_id):
                    snapshot = self.app_server.auto_analysis.cancel(task_id)
                    self._send_json(HTTPStatus.OK, snapshot)
                    return
                run_id = self.app_server.analysis_bridge.run_id(task_id)
                snapshot = self.app_server.manager.snapshot(run_id)
                if snapshot["state"] not in TERMINAL_STATES:
                    snapshot = self.app_server.manager.cancel(run_id)
                self._send_json(
                    HTTPStatus.OK,
                    {"task_id": task_id, "run_id": run_id, "state": snapshot["state"]},
                )
                return

            parts = _route_parts(path)
            if len(parts) == 3 and parts[0] == "runs" and parts[2] == "cancel":
                self._reject_nonempty_body()
                run_id = self._validated_run_id(parts[1])
                snapshot = self.app_server.manager.cancel(run_id)
                status = HTTPStatus.OK if snapshot["state"] == "cancelled" else HTTPStatus.ACCEPTED
                self._send_json(
                    status,
                    {"run_id": run_id, "state": snapshot["state"]},
                )
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
        except JobError as exc:
            self._send_error_safely(exc.status_code, exc.code, str(exc))
        except UiDataError as exc:
            self._send_error_safely(HTTPStatus.BAD_REQUEST, "invalid_gwas_data", str(exc))
        except HistoryNotFoundError as exc:
            self._send_error_safely(HTTPStatus.NOT_FOUND, "history_not_found", str(exc))
        except HistoryError as exc:
            self._send_error_safely(HTTPStatus.BAD_REQUEST, "invalid_history", str(exc))
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
        except Exception:
            self._send_error_safely(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "The local web service encountered an unexpected error.",
            )

    def _get_history(self, path: str) -> None:
        parts = path.removeprefix("/api/history/").split("/")
        if not parts or parts[0] not in {"datasets", "chats", "analyses"}:
            raise HistoryNotFoundError("history route was not found")
        store = self.app_server.history_store
        kind = parts[0]
        if len(parts) == 1:
            parameters = parse_qs(urlsplit(self.path).query)
            try:
                limit = int(parameters.get("limit", ["50"])[0])
                offset = int(parameters.get("offset", ["0"])[0])
            except (ValueError, IndexError) as exc:
                raise HistoryError("history pagination is invalid") from exc
            dataset_id = parameters.get("dataset_id", [None])[0] or None
            # The upload list is the entry point for everything saved under one
            # dataset, so it carries that dataset's conversation counts.
            items = (store.list_datasets_with_activity(limit=limit, offset=offset) if kind == "datasets"
                     else store.list(kind, limit=limit, offset=offset, dataset_id=dataset_id))
            self._send_json(HTTPStatus.OK, {
                "kind": kind, "items": items, "offset": offset,
                "has_more": len(items) == limit,
            })
            return
        if len(parts) == 2:
            read = {
                "datasets": store.get_dataset,
                "chats": store.get_chat,
                "analyses": store.get_analysis,
            }[kind]
            snapshot = read(parts[1])
            if kind == "chats" and parse_qs(urlsplit(self.path).query).get("include_ranking_review") == ["1"]:
                for message in snapshot.get("messages", []):
                    descriptors = message.get("ranking_evidence", [])
                    if message.get("role") != "assistant" or len(descriptors) != 1:
                        continue
                    try:
                        ranking_review = self.app_server.manager.ranking_review_receipt(descriptors[0])
                    except (JobValidationError, JobNotFoundError, OSError, ValueError, TypeError, AttributeError):
                        ranking_review = None
                    if ranking_review:
                        for stage in message.get("progress", []):
                            if stage.get("stage") == "safety" and isinstance(stage.get("review_update"), dict):
                                stage["review_update"]["ranking_review"] = ranking_review
            self._send_json(HTTPStatus.OK, snapshot)
            return
        if len(parts) == 3 and kind == "analyses" and parts[2] == "download":
            snapshot = store.get_analysis(parts[1])
            results = snapshot.get("results", [])
            if not isinstance(results, list) or any(
                not isinstance(row, Mapping) for row in results
            ):
                raise HistoryError("saved analysis results cannot be exported")
            self._send_analysis_xlsx(results, analysis_mode=snapshot.get("analysis_mode"))
            return
        raise HistoryNotFoundError("history route was not found")

    def _history_plot_state(self, kind: str, history_id: str, snapshot: Mapping[str, Any]) -> tuple[UiState, str | None]:
        """Restore the saved plotting context for menus and cold history URLs."""
        if kind == "analyses":
            return restore_analysis_plot(self.app_server.history_store, snapshot, history_id)
        dataset_id = history_id if kind == "datasets" else snapshot.get("dataset_id")
        next_state = UiState()
        if dataset_id:
            dataset = snapshot if kind == "datasets" else self.app_server.history_store.get_dataset(dataset_id)
            next_state.load_history(dataset, dataset_id)
        return next_state, dataset_id

    def _post_history(self, path: str) -> None:
        parts = path.removeprefix("/api/history/").split("/")
        if not parts or parts[0] not in {"datasets", "chats", "analyses"}:
            raise HistoryNotFoundError("history route was not found")
        store = self.app_server.history_store
        kind = parts[0]
        if len(parts) == 3 and kind == "analyses" and parts[2] == "discussion":
            payload = self._read_json_body(max_bytes=MAX_UPLOAD_JSON_BYTES)
            if set(payload) != {"locus_no", "messages"}:
                raise HistoryError("analysis discussion requires only locus_no and messages")
            saved = store.update_analysis_discussion(
                parts[1], payload["locus_no"], payload["messages"],
            )
            self._send_json(HTTPStatus.OK, {"success": True, **saved})
            return
        if len(parts) == 3 and parts[2] == "restore":
            self._reject_nonempty_body()
            read = {
                "datasets": store.get_dataset,
                "chats": store.get_chat,
                "analyses": store.get_analysis,
            }[kind]
            snapshot = read(parts[1])
            # Report-only sessions get an empty state, preserving their original scope.
            next_state, dataset_id = self._history_plot_state(kind, parts[1], snapshot)
            self.app_server.publish_ui_state(next_state)
            self._send_json(HTTPStatus.OK, {
                "success": True, "history_id": parts[1], "dataset_id": dataset_id,
                "redirect": "/app", "snapshot": snapshot if kind != "datasets" else None,
            })
            return
        if kind in {"chats", "analyses"} and len(parts) in {1, 2}:
            payload = self._read_json_body(max_bytes=MAX_UPLOAD_JSON_BYTES)
            history_id = parts[1] if len(parts) == 2 else payload.pop("history_id", None)
            write = store.save_chat if kind == "chats" else store.save_analysis
            saved = write(payload, history_id=history_id)
            self._send_json(HTTPStatus.OK, {"success": True, **saved})
            return
        raise HistoryNotFoundError("history route was not found")

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._request_is_local():
            return
        self._send_error_safely(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "Cross-origin preflight is not supported; use the same-origin UI.",
        )

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._request_is_local():
            return
        self.close_connection = True
        path = urlsplit(self.path).path
        if path.startswith(f"{API_PREFIX}/phenotype-scopes/"):
            self._handle_phenotype_scopes("delete", unquote(path.rsplit("/", 1)[-1]), {})
            return
        if path != f"{API_PREFIX}/ldlink-settings":
            self._send_error_safely(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
            return
        self._handle_ldlink_settings("delete", {})

    def _handle_phenotype_scopes(self, action: str, target_id: str | None, value: Mapping[str, Any]) -> None:
        from ..phenotype_scope import PhenotypeScopeError
        store = self.app_server.phenotype_scope_store
        try:
            if store is None:
                raise PhenotypeScopeError("phenotype_scope_store_unavailable")
            if action == "update":
                if set(value) != {"related_ids"}:
                    raise PhenotypeScopeError("invalid_related_selection")
                store.update_related(target_id, value["related_ids"])
            elif action == "delete":
                store.delete(target_id)
            if action != "list":
                self.app_server.phenotype_scope_planner.forget()
            self._send_json(HTTPStatus.OK, {"success": True, "scopes": store.list()})
        except PhenotypeScopeError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"success": False, "reason": exc.reason,
                "scopes": store.list() if store is not None else []})
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
        except Exception:
            self._send_error_safely(HTTPStatus.INTERNAL_SERVER_ERROR, "phenotype_scope_unavailable",
                                   "Saved phenotype scopes could not be updated.")

    def _handle_ldlink_settings(self, path: str, value: Mapping[str, Any]) -> None:
        from ..credentials import CredentialError
        try:
            if set(value).difference({"api_key"}):
                raise CredentialError("Unknown LDlink settings fields.")
            settings = self.app_server.ldlink_settings
            result = (settings.clear() if path == "delete" else
                      settings.test(value.get("api_key")) if path.endswith("ldlink-test") else
                      settings.save(value.get("api_key")))
            self._send_json(HTTPStatus.OK, result)
        except CredentialError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"success": False, "response": str(exc),
                "ldlink": self.app_server.ldlink_settings.status()})
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
        except Exception:
            self._send_error_safely(HTTPStatus.INTERNAL_SERVER_ERROR, "ld_settings_unavailable",
                                   "LDlink settings could not be updated.")

    def _request_is_local(self) -> bool:
        host_header = self.headers.get("Host")
        try:
            parsed_host = urlsplit(f"//{host_header}") if host_header else None
            host_name = parsed_host.hostname if parsed_host else None
            host_port = parsed_host.port if parsed_host else None
        except ValueError:
            host_name = None
            host_port = None
        expected_port = int(self.app_server.server_address[1])
        if not _is_loopback_name(host_name) or host_port not in {None, expected_port}:
            self._send_error_safely(
                HTTPStatus.FORBIDDEN,
                "invalid_host",
                "The Host header must identify a loopback address.",
            )
            return False

        origin = self.headers.get("Origin")
        if origin:
            try:
                parsed = urlsplit(origin)
                origin_port = parsed.port
            except ValueError:
                parsed = None
                origin_port = None
            if (
                parsed is None
                or parsed.scheme != "http"
                or not _is_loopback_name(parsed.hostname)
                or origin_port != expected_port
            ):
                self._send_error_safely(
                    HTTPStatus.FORBIDDEN,
                    "invalid_origin",
                    "Cross-origin requests are not permitted.",
                )
                return False
        return True

    def _read_json_body(self, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
        # On any parse failure, unread bytes must not become another request on
        # this HTTP/1.1 connection. Successful reads restore its prior state.
        previous_close = self.close_connection
        self.close_connection = True
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise JobValidationError("Content-Type must be application/json")
        length = self._content_length()
        if length is None:
            raise JobValidationError("Content-Length is required")
        if length <= 0:
            raise JobValidationError("request JSON cannot be empty")
        if length > max_bytes:
            self.close_connection = True
            raise _PayloadTooLarge(f"request JSON exceeds {max_bytes // (1024 * 1024)} MB")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise JobValidationError("request body ended before Content-Length")
        try:
            value = json.loads(raw.decode("utf-8"))
        # Python's integer-string guard and custom JSON decoders may raise a
        # plain ValueError rather than JSONDecodeError.  It is still malformed
        # client input, never an internal server failure.
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise JobValidationError("request body is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise JobValidationError("request body must be a JSON object")
        self.close_connection = previous_close
        return value

    def _content_length(self) -> int | None:
        if self.headers.get("Transfer-Encoding") is not None:
            self.close_connection = True
            raise JobValidationError("Transfer-Encoding is not supported")
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            return None
        raw = lengths[0].strip()
        if len(lengths) != 1 or not raw or not raw.isascii() or not raw.isdecimal():
            self.close_connection = True
            raise JobValidationError("Content-Length is invalid")
        try:
            return int(raw)
        except ValueError as exc:
            self.close_connection = True
            raise JobValidationError("Content-Length is invalid") from exc

    def _reject_nonempty_body(self) -> None:
        length = self._content_length()
        if length is None:
            return
        if length > 0:
            self.close_connection = True
            if length > MAX_JSON_BYTES:
                self.close_connection = True
                raise _PayloadTooLarge("request body exceeds 2 MB")
            raise JobValidationError("cancel does not accept a request body")

    def _stream_events(self, run_id: str) -> None:
        last_id = self.headers.get("Last-Event-ID", "0")
        try:
            after_sequence = max(0, int(last_id))
        except ValueError as exc:
            raise JobValidationError("Last-Event-ID must be an integer") from exc

        # Resolve the run before committing SSE response headers so a missing
        # run can still return a normal structured 404 response.
        self.app_server.manager.snapshot(run_id)

        self.send_response(HTTPStatus.OK)
        self._common_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                events, terminal = self.app_server.manager.wait_for_events(
                    run_id,
                    after_sequence=after_sequence,
                    timeout=self.app_server.heartbeat_interval,
                )
                for event in events:
                    event_type = str(event.get("type", "status"))
                    if event_type == "heartbeat" and event.get("transient") is True:
                        self.wfile.write(b": heartbeat\n\n")
                        continue
                    sequence = int(event["seq"])
                    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    block = (f"id: {sequence}\nevent: {event_type}\ndata: {payload}\n\n").encode(
                        "utf-8"
                    )
                    self.wfile.write(block)
                    after_sequence = sequence
                self.wfile.flush()
                if terminal:
                    self.close_connection = True
                    return
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True

    def _stream_legacy_analysis(
        self, task_id: str, phenotype: str, *, concurrency: int | None = None,
    ) -> None:
        auto = self.app_server.auto_analysis
        auto.begin(task_id, phenotype, concurrency=concurrency)
        self.send_response(HTTPStatus.OK)
        self._common_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        sequence = 0
        try:
            while True:
                events, terminal = auto.wait_for_events(
                    task_id, after_sequence=sequence,
                    timeout=self.app_server.heartbeat_interval,
                )
                for event in events:
                    sequence = int(event["seq"])
                    self._write_legacy_event(event)
                if terminal:
                    break
                if not events:
                    self.wfile.write(b": gwasmind-keepalive\n\n")
                    self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except _CLIENT_DISCONNECT_ERRORS:
            # Auto work belongs to the task, not to this observer connection.
            pass
        except JobError as exc:
            self._finish_legacy_stream_error(str(exc))
        finally:
            self.close_connection = True

    def _finish_legacy_stream_error(self, message: str) -> None:
        try:
            self._write_legacy_event({"type": "error", "message": message})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except _CLIENT_DISCONNECT_ERRORS:
            pass
        self.close_connection = True

    def _stream_chat(self, value: Mapping[str, Any]) -> None:
        def start() -> None:
            self.send_response(HTTPStatus.OK)
            self._common_headers()
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

        value = dict(value)
        auto_reference = value.pop("auto_analysis_ref", None)
        source_options: dict[str, Any] = {}
        if auto_reference is not None:
            source_options = resolve_auto_followup(self.app_server, auto_reference)
            previous_report = source_options.pop("previous_report", "")
            # The immutable uploaded observations, not current plot selection or
            # browser-provided coordinates, define the source data for this turn.
            value["variants"] = [
                str(item["variant_id"]) for item in source_options["source_observations"]
            ]
            value.pop("region", None)
            # A saved report is conversational evidence, not a command to bypass
            # the planner. Fresh phenotype questions must be allowed to retrieve.
            value.pop("analysis_context", None)
            history = value.get("history", [])
            history_limit = parameter("web.chat_history_limit")
            if isinstance(history, list) and len(history) <= history_limit and previous_report and history_limit > 0:
                value["history"] = [
                    {"role": "assistant", "content": previous_report[:parameter("web.chat_history_item_max_chars")]},
                    *(history[-(history_limit - 1):] if history_limit > 1 else []),
                ]
        workflow = AnalysisWorkflow(
            self.app_server, emit=self._write_legacy_event, start=start,
            on_run=lambda run_id: self._write_chat_event("run", run_id),
            **source_options,
        )
        workflow.run(value)
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except _CLIENT_DISCONNECT_ERRORS:
            pass
        self.close_connection = True

    def _finish_chat_stream_error(self, message: str) -> None:
        try:
            self._write_chat_event("error", message)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except _CLIENT_DISCONNECT_ERRORS:
            pass
        self.close_connection = True

    def _write_chat_event(self, event_type: str, content: str) -> None:
        self._write_legacy_event({"type": event_type, "content": content})

    def _write_legacy_event(self, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _send_legacy_analysis_result(self, task_id: str) -> None:
        if self.app_server.auto_analysis.has_task(task_id):
            self._send_json(HTTPStatus.OK, self.app_server.auto_analysis.snapshot(task_id))
            return
        bridge = self.app_server.analysis_bridge
        run_id = bridge.run_id(task_id)
        upload = bridge.upload(task_id)
        result = self.app_server.manager.result(run_id)
        projected = project_results(result["results"], upload.queries)
        self._send_json(
            HTTPStatus.OK,
            {"task_id": task_id, "run_id": run_id, "results": projected},
        )

    def _send_legacy_analysis_download(self, task_id: str) -> None:
        if self.app_server.auto_analysis.has_task(task_id):
            result = self.app_server.auto_analysis.snapshot(task_id)
            projected = result["results"]
        else:
            bridge = self.app_server.analysis_bridge
            run_id = bridge.run_id(task_id)
            upload = bridge.upload(task_id)
            result = self.app_server.manager.result(run_id)
            projected = project_results(result["results"], upload.queries)
        self._send_analysis_xlsx(projected, analysis_mode=result.get("analysis_mode"))

    def _send_analysis_xlsx(self, results: list[Mapping[str, Any]], *, analysis_mode: str | None = None) -> None:
        """Export active or persisted results without depending on a live run."""

        screening = (all(row.get("analysis_mode") == "prior_report_screening" for row in results)
                     if results else analysis_mode == "prior_report_screening")
        serializer = _screening_xlsx_row if screening else _legacy_xlsx_row
        export_rows = [serializer(row) for row in results]
        columns = tuple(export_rows[0]) if export_rows else tuple(serializer({}))
        payload = build_xlsx_report(export_rows, columns, sheet_name="GWASMind Results")
        self.send_response(HTTPStatus.OK)
        self._common_headers()
        self.send_header(
            "Content-Type",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.send_header("Content-Disposition", 'attachment; filename="gwasmind_results.xlsx"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_artifact(self, run_id: str, query_index: int, name: str) -> None:
        formats = parse_qs(urlsplit(self.path).query, keep_blank_values=True).get("format", ["raw"])
        if len(formats) != 1 or formats[0] not in {"raw", "html"}:
            raise JobValidationError("artifact format must be raw or html")
        if formats[0] == "html":
            path = self.app_server.manager.readable_ranking_export_path(run_id, query_index, name)
        else:
            exporter = getattr(self.app_server.manager, "export_artifact_path", self.app_server.manager.artifact_path)
            path = exporter(run_id, query_index, name)
        size = path.stat().st_size
        content_type = (
            "application/x-ndjson; charset=utf-8"
            if path.suffix == ".jsonl"
            else "text/html; charset=utf-8" if path.suffix == ".html"
            else "text/markdown; charset=utf-8" if path.suffix == ".md"
            else "application/json; charset=utf-8"
        )
        self.send_response(HTTPStatus.OK)
        self._common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

    def _send_static(self, relative: str) -> None:
        root = self.app_server.static_root
        target = (root / relative).resolve()
        if root != target and root not in target.parents:
            self._send_error(
                HTTPStatus.FORBIDDEN,
                "invalid_static_path",
                "Static path escaped its root.",
            )
            return
        if not target.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "Static resource not found.")
            return
        payload = target.read_bytes()
        content_type = (_STATIC_CONTENT_TYPES.get(target.suffix.lower())
                        or mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self._common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_page(self, name: str, *, data_source: str | None = None) -> None:
        payload = render_page(name, data_source=data_source)
        self.send_response(HTTPStatus.OK)
        self._common_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self._common_headers()
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _public_config(self) -> dict[str, Any]:
        manager = self.app_server.manager
        return {
            "api_version": "v1",
            "default_mode": self.app_server.default_analysis_mode,
            "default_ranker": "openai",
            "modes": ["live", "frozen-safe"],
            "frozen_data_root": (
                str(manager.default_frozen_data_root) if manager.default_frozen_data_root else None
            ),
            "default_frozen_release": manager.default_frozen_release,
            "default_frozen_snapshot_hash": manager.default_frozen_snapshot_hash,
            "limits": {
                "max_request_bytes": MAX_JSON_BYTES,
                "max_upload_bytes": MAX_UPLOAD_JSON_BYTES,
                "max_queries": MAX_QUERIES,
                "max_variants_per_query": MAX_VARIANTS_PER_QUERY,
                "max_candidate_genes_per_query": MAX_CANDIDATE_GENES_PER_QUERY,
                "analysis_concurrency": 1,
                "worker_threads": WORKER_THREADS,
                "duckdb_memory_limit": DUCKDB_MEMORY_LIMIT,
            },
            "artifact_names": sorted(ARTIFACT_NAMES),
        }

    def _send_json(self, status: int | HTTPStatus, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self._common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status: int | HTTPStatus, code: str, message: str) -> None:
        self._send_json(status, {"error": message, "code": code})

    def _send_error_safely(self, status: int | HTTPStatus, code: str, message: str) -> None:
        try:
            self._send_error(status, code, message)
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True

    def _common_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )

    @staticmethod
    def _validated_run_id(run_id: str) -> str:
        if not _RUN_ID.fullmatch(run_id):
            raise JobNotFoundError("run was not found")
        return run_id

    def log_message(self, format: str, *args: Any) -> None:
        # The scientific audit and job event streams are structured; avoid a
        # second, potentially noisy request log in the desktop process.
        return


def resolve_auto_followup(
    app_server: GWASMindHTTPServer, reference: Any,
) -> dict[str, Any]:
    """Resolve original uploaded locus observations without changing the plot."""
    if not isinstance(reference, Mapping) or set(reference) - {
        "history_id", "task_id", "locus_no"
    }:
        raise JobValidationError("Auto Analysis reference is invalid")
    locus_no = reference.get("locus_no")
    if isinstance(locus_no, bool) or not isinstance(locus_no, int) or locus_no < 1:
        raise JobValidationError("Auto Analysis locus number is invalid")
    history_id, task_id = reference.get("history_id"), reference.get("task_id")
    if history_id:
        snapshot = app_server.history_store.get_analysis(history_id)
        queries = snapshot.get("queries", [])
        phenotype = snapshot.get("target_phenotype") or snapshot.get("source_phenotype")
    elif isinstance(task_id, str) and _RUN_ID.fullmatch(task_id):
        snapshot = app_server.auto_analysis.snapshot(task_id)
        phenotype = snapshot["phenotype"]
        queries = app_server.analysis_bridge.queries_for_run(task_id, phenotype)
    else:
        raise JobValidationError("A saved analysis or active Auto Analysis task is required")
    if locus_no > len(queries):
        raise JobValidationError("The selected locus was not found in this analysis")
    observations = queries[locus_no - 1].get("variants")
    if not isinstance(observations, list) or not observations or not phenotype:
        raise JobValidationError("The saved locus does not contain its original GWAS observations")
    saved_row = next((row for row in snapshot.get("results", [])
                      if isinstance(row, Mapping) and row.get("locus_no") == locus_no), {})
    retained = saved_row.get("selected_variants")
    if isinstance(retained, list) and len(retained) == len(observations) and all(
        isinstance(row, Mapping) and all(row.get(key) == original.get(key)
            for key in ("variant_id", "p_value", "chromosome", "position"))
        for row, original in zip(retained, observations, strict=True)
    ):
        observations = retained
    previous_report = str(saved_row.get("report_markdown") or "")
    if saved_row.get("analysis_mode") == "prior_report_screening":
        classification = {
            "PREVIOUSLY_REPORTED": "Known",
            "NOVEL_IN_SCOPE": "Novel (in scope)",
        }.get(saved_row.get("screening_status"), "Unresolved")
        conclusion = f"## Prior-association screening: {classification}"
        reason = str(saved_row.get("screening_reason") or "").strip()
        previous_report = "\n\n".join(
            part for part in (conclusion, reason, previous_report) if part
        )
    return {
        "source_observations": [dict(item) for item in observations],
        "source_trait": str(phenotype),
        "previous_report": previous_report,
    }


class _OfflineRankingLDReference:
    """Keep frozen retrieval offline and disclose unavailable LD reference coverage."""

    @staticmethod
    def resolve(_identifier):
        raise RuntimeError("Population LD reference is unavailable in frozen mode.")


class _OfflineNoveltyReference:
    """An explicit unavailable reference for frozen runs; never opens a socket."""

    @staticmethod
    def screen(source_ids, phenotype, population):
        return {
            "pairs": [], "coverage_complete": False, "identity_complete": True,
            "prior_variant_count": 0,
            "issues": ["Live prior-association and LD reference queries are disabled in frozen mode."],
        }


def _phenotype_scope_status(scope: Mapping[str, Any]) -> str:
    if scope.get("status") == "target_unresolved":
        return "Phenotype scope: target phenotype not resolved in EFO."
    if scope.get("status") == "expansion_not_assessed":
        return "Phenotype scope: target phenotype only; related phenotypes not assessed."
    return f"Phenotype scope: target plus {len(scope.get('related') or ())} related phenotypes from the saved EFO scope."


def _input_retrieval_precondition(observations, *, needs_retrieval):
    """Stop impossible requests locally; an unresolved rsID may still be looked up."""
    if not needs_retrieval:
        return None
    if not observations:
        return {"reason_code": "source_selection_missing", "message": "No source SNP was selected."}
    from .stage_bridge import _runtime_integrity, _runtime_integrity_observations
    integrity = _runtime_integrity(observations, genome_build="GRCh38")
    if integrity.valid_source_count:
        return None
    for row in observations:
        # Reference alleles for an rsID are supplied by identity resolution.
        # Their initial absence does not prove malformed input.
        if re.fullmatch(r"rs[0-9]+", str(row.get("variant_id") or "").strip(), re.I):
            current = _runtime_integrity_observations((row,), genome_build="GRCh38")
            allele_fields = ("reference_allele", "referenceAllele", "ref", "REF",
                             "alternate_allele", "alternateAllele", "alt", "ALT")
            if current.variant_integrity and current.build_integrity and not any(row.get(key) for key in allele_fields):
                return None
    return {"reason_code": "source_input_invalid", "message": "None of the selected source variants passed the local input checks."}


class AnalysisWorkflow:
    """One transport-neutral four-stage pipeline shared by Chat and Auto Analysis.

    Only public stage events and a grounded final report leave the engine. An Auto
    upload supplies immutable observations explicitly, independent of the plot's
    currently loaded dataset. Credentials stay inside the existing JobManager.
    """

    def __init__(
        self,
        app_server: GWASMindHTTPServer,
        *,
        emit: Any,
        start: Any = None,
        source_observations: list[Mapping[str, Any]] | None = None,
        source_trait: str | None = None,
        query_phenotype: str | None = None,
        required_intent: ChatIntent | None = None,
        on_run: Any = None,
        cancel_event: threading.Event | None = None,
        auto_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.app_server = app_server
        self.emit = emit
        self.start = start or (lambda: None)
        self.source_observations = source_observations
        self.source_trait = source_trait
        self.query_phenotype = query_phenotype
        self.required_intent = required_intent
        self.on_run = on_run or (lambda run_id: None)
        self.cancel_event = cancel_event
        self.auto_context = dict(auto_context or {})
        self._stage_started: dict[str, float] = {}
        self._stage_elapsed: dict[str, int] = {}
        self._stage_time_offsets: dict[str, float] = {}
        self._active_public_stage: str | None = None
        self._response_stream_lock = threading.Lock()
        self._stage_event_lock = threading.RLock()
        self._last_stage_payload = None
        self._response_stream_metrics: dict[str, int] = {}
        from .model_usage import ModelCallUsage
        self._model_usage = ModelCallUsage()
        self._response_completed_tokens = 0
        self._response_completed_chars = 0
        self._response_completed_tokens_exact = True
        self._response_owner_thread = threading.get_ident()
        self.outcome: dict[str, Any] = {
            "configuration_provenance": configuration_provenance(),
            "report_markdown": "",
            "report_quality": "PASS",
            "run_id": None,
            "result": {},
            "error": None,
        }

    def run(self, value: Mapping[str, Any]) -> dict[str, Any]:
        self._execute(value)
        return dict(self.outcome)

    def _auto_safety_context(self, facts):
        facts = novelty_coverage_check(
            facts, self.outcome.get("result", {}).get("novelty_screen"),
        )
        if self.auto_context.get("ranking_mode") != "exploratory":
            return facts
        if any(check.check_id == "novelty_coverage" for check in facts.checks):
            return facts
        return replace(facts, checks=(*facts.checks, configured_safety_check(
            check_id="novelty_coverage", label="Prior-report coverage",
            severity=SafetySeverity.WARNING, outcome=SafetyOutcome.WARN,
            fact=("Prior-report lookup is incomplete; novelty remains unresolved. "
                  + ("Gene ranking is exploratory and uses only the available direct evidence."
                     if facts.ranking_allowed else
                     "Gene evidence was assessed, but no supported gene ranking is available.")),
        )))

    def _execute(self, value: Mapping[str, Any]) -> None:
        allowed = {
            "question",
            "message",
            "region",
            "variants",
            "history",
            "analysis_context",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise JobValidationError("unknown chat fields: " + ", ".join(sorted(unknown)))
        question = value.get("question") or value.get("message")
        if not isinstance(question, str) or not question.strip():
            raise JobValidationError("chat question is required")
        question = question.strip()
        if len(question) > parameter("web.chat_question_max_chars"):
            raise JobValidationError(f"chat question exceeds {parameter('web.chat_question_max_chars'):,} characters")
        variants_value = value.get("variants", [])
        if not isinstance(variants_value, list) or len(variants_value) > parameter("execution.max_variants_per_query"):
            raise JobValidationError(f"chat variants must be an array of at most {parameter('execution.max_variants_per_query')} IDs")
        if any(not isinstance(item, str) for item in variants_value):
            raise JobValidationError("chat variant IDs must be strings")
        region = value.get("region")
        if region is not None and not isinstance(region, Mapping):
            raise JobValidationError("chat region must be an object")
        history = value.get("history", [])
        if not isinstance(history, list) or len(history) > parameter("web.chat_history_limit"):
            raise JobValidationError(f"chat history must contain at most {parameter('web.chat_history_limit')} messages")
        for index, message in enumerate(history):
            if not isinstance(message, Mapping):
                raise JobValidationError(f"chat history[{index}] must be an object")
            if set(message).difference({"role", "content"}):
                raise JobValidationError(f"chat history[{index}] has unknown fields")
            if not isinstance(message.get("role"), str) or message["role"] not in {
                "user", "assistant"
            }:
                raise JobValidationError(f"chat history[{index}] has an invalid role")
            if not isinstance(message.get("content"), str) or len(message["content"]) > parameter("web.chat_history_item_max_chars"):
                raise JobValidationError(f"chat history[{index}] has invalid content")
        context = value.get("analysis_context")
        if context is not None and (not isinstance(context, str) or len(context) > parameter("web.chat_analysis_context_max_chars")):
            raise JobValidationError(f"analysis_context must be a string up to {parameter('web.chat_analysis_context_max_chars'):,} characters")

        # Keep one dataset generation for both rows and source phenotype even if
        # another browser tab uploads/restores/clears data during this request.
        plot_state = self.app_server.ui_state
        observations = (
            [dict(item) for item in self.source_observations]
            if self.source_observations is not None
            else plot_state.variant_observations(
                [str(item) for item in variants_value], region=region
            )
        )
        source_trait = self.source_trait or plot_state.trait_hint()
        self.outcome["source_observations"] = [dict(item) for item in observations]
        fixed_workflow = self.required_intent is not None
        if fixed_workflow:
            fallback_route = ChatRoute(
                intent=self.required_intent,
                needs_retrieval=True,
                phenotype=self.query_phenotype or source_trait,
                planning_source="fixed_auto_workflow",
            )
        else:
            try:
                fallback_route = route_chat_request(
                    question, selected_variant_count=len(observations),
                    has_selected_region=bool(region), source_trait=source_trait,
                )
            except QTLRoutingError as exc:
                raise JobValidationError(str(exc)) from exc
        query_phenotype: str | None = None
        if isinstance(context, str):
            match = re.search(
                r"##\s+Target Phenotype\s+\*\*([^*\r\n]{1,500})\*\*",
                context,
                flags=re.IGNORECASE,
            )
            if match:
                query_phenotype = match.group(1).strip()

        source_signal = summarize_source_gwas_observations(
            observations,
            phenotype=source_trait,
        )
        model_status = self.app_server.manager.model_status()
        model = model_status.get("models", {}).get("openai", {})
        use_model = bool(model.get("available"))
        run_id: str | None = None
        from ..model_generation import GenerationAttempts
        generation_by_stage = {}
        writer_outputs: list[Mapping[str, Any]] = []
        writer_output_audit_errors: list[str] = []
        writer_call_count = 0
        phenotype_relevance_calls = 0

        def structural_completer(
            messages: list[dict[str, str]], *, max_tokens: int = parameter("model.chat_default_max_tokens"),
            json_output: bool = True,
            generation_stage: str | None = None,
        ) -> Any:
            nonlocal writer_call_count
            in_response = self._active_public_stage == "response"
            stage_key = generation_stage or self._active_public_stage or "plan"
            allowance = generation_by_stage.setdefault(stage_key, GenerationAttempts(
                transport_retries=parameter("model.writer_transport_retries")) if stage_key == "response"
                else GenerationAttempts())
            prior_tokens = self._response_completed_tokens
            prior_chars = self._response_completed_chars
            prior_exact = self._response_completed_tokens_exact
            saw_stream_progress = False
            if in_response:
                from ..model_options import model_token_budget
                output_allowance = model_token_budget(base_url=model.get("base_url"),
                    model_id=model.get("model_id"), messages=messages)["effective_max_output_tokens"]
                with self._response_stream_lock:
                    self._model_usage.start(f"report-{writer_call_count + 1}",
                        "report_generation" if writer_call_count == 0 else "report_repair",
                        writer_call_count + 1, output_allowance)
                writer_call_count += 1

            def stream_progress(metrics: dict[str, int]) -> None:
                nonlocal saw_stream_progress
                saw_stream_progress = True
                public = {"received_chars": prior_chars + metrics.get("received_chars", 0)}
                source_key = (
                    "output_tokens" if "output_tokens" in metrics else "estimated_output_tokens"
                )
                if source_key in metrics:
                    key = "output_tokens" if prior_exact and source_key == "output_tokens" \
                        else "estimated_output_tokens"
                    public[key] = prior_tokens + metrics[source_key]
                with self._response_stream_lock:
                    self._response_stream_metrics = public
                    self._model_usage.stream(metrics)
                if threading.get_ident() == self._response_owner_thread:
                    self._write_chat_stage_event(
                        "response", "running", "Generating response", metrics=public
                    )

            try:
                result = self.app_server.manager.complete_runtime_chat(
                    messages,
                    max_tokens=max_tokens,
                    # Gene ranking owns the scientific reasoning; descriptive writers
                    # format retrieved facts. Only configured intents keep thinking.
                    thinking_mode_override=(
                        None if in_response and getattr(intent, "value", intent)
                        in parameter("model.writer_reasoning_intents") else THINKING_DISABLED),
                    json_output=json_output,
                    generation_attempts=allowance,
                    **({"progress_callback": stream_progress} if in_response else {}),
                    **({"cancel_event": self.cancel_event} if self.cancel_event is not None else {}),
                )
            except Exception as exc:
                if in_response:
                    with self._response_stream_lock:
                        self._model_usage.finish(getattr(exc, "usage", {}), status="failed")
                raise
            if in_response:
                if run_id is not None:
                    try:
                        writer_outputs.append(self.app_server.manager.record_chat_writer_output(
                            run_id, writer_call=writer_call_count, messages=messages, completion=result,
                        ))
                    except (JobError, OSError, ValueError) as exc:
                        writer_output_audit_errors.append(type(exc).__name__)
                usage = getattr(result, "usage", {})
                count = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                    self._response_completed_tokens += count
                else:
                    estimated_tokens = (len(getattr(result, "content", "")) + 3) // 4
                    self._response_completed_tokens += estimated_tokens
                    self._response_completed_tokens_exact = False
                self._response_completed_chars += len(getattr(result, "content", ""))
                with self._response_stream_lock:
                    self._model_usage.finish(usage)
                    public = dict(self._response_stream_metrics)
                    if not saw_stream_progress:
                        # A non-stream provider's original content is observed;
                        # program-rendered evidence/reference text is not counted.
                        public["received_chars"] = self._response_completed_chars
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        key = "output_tokens" if self._response_completed_tokens_exact \
                            else "estimated_output_tokens"
                        previous_key = (
                            "estimated_output_tokens" if key == "output_tokens" else "output_tokens"
                        )
                        public.pop(previous_key, None)
                        public[key] = self._response_completed_tokens
                    self._response_stream_metrics = public
            return result

        def phenotype_completer(messages):
            nonlocal phenotype_relevance_calls
            phenotype_relevance_calls += 1
            count = len(json.loads(messages[-1]["content"])["candidates"])
            stage = ("phenotype_relevance" if phenotype_relevance_calls == 1 else
                     f"phenotype_relevance_batch_{phenotype_relevance_calls}")
            return self._run_stage_operation(lambda: structural_completer(messages,
                max_tokens=max(1200, min(16384, 384 * count)), generation_stage=stage),
                stage="safety", detail="Assessing phenotype relevance of the retrieved GWAS records...")

        self.start()

        try:
            self._write_chat_stage_event(
                "plan",
                "running",
                "Preparing fixed workflow" if fixed_workflow else "Understanding request",
                detail=("Preparing the fixed Auto Analysis workflow..." if fixed_workflow
                        else "Analyzing the question and selected variants..."),
            )
            chat_route = fallback_route
            if fixed_workflow:
                # Auto already owns the screening -> prioritization decision.
                # Do not ask a question router to rediscover or change that plan.
                pass
            elif isinstance(context, str) and context.strip():
                # Auto Analysis has already supplied a bounded evidence context.
                # Follow-up chat reasons over that context and must not silently
                # launch a second locus retrieval.
                chat_route = ChatRoute(
                    intent=ChatIntent.GENERAL,
                    qtl_types=(),
                    needs_retrieval=False,
                    planning_source="supplied_analysis_context",
                )
            elif use_model:
                self._write_chat_stage_event(
                    "plan",
                    "running",
                    "Understanding request",
                    detail="Identifying the requested analysis workflow...",
                )
                try:
                    routing_completion = structural_completer(
                        model_chat_router_messages(
                            question,
                            selected_variant_count=len(observations),
                            source_trait=(
                                source_trait
                                if source_trait.casefold()
                                not in {
                                    "user-selected gwas locus",
                                    "the loaded phenotype",
                                    "loaded gwas phenotype",
                                }
                                else None
                            ),
                            selected_significant_count=(
                                source_signal.genome_wide_significant_variants
                            ),
                            conversation_history=history,
                        ),
                        max_tokens=parameter("model.router_max_tokens"),
                    )
                    model_route = reconcile_chat_route(
                        question, parse_model_chat_route(routing_completion.content),
                        selected_variant_count=len(observations),
                        has_selected_region=bool(region), source_trait=source_trait,
                    )
                    if is_explicit_general_chat(question) and model_route.needs_retrieval:
                        chat_route = fallback_route
                    else:
                        # A user-named native QTL subtype is a hard query constraint;
                        # the planner may choose the workflow but cannot broaden it.
                        chat_route = ChatRoute(
                            intent=model_route.intent,
                            qtl_types=(
                                fallback_route.qtl_types
                                if fallback_route.qtl_types
                                else model_route.qtl_types
                            ),
                            needs_retrieval=model_route.needs_retrieval,
                            phenotype=model_route.phenotype,
                            context_tissues=model_route.context_tissues,
                            context_cell_types=model_route.context_cell_types,
                            evidence_kinds=model_route.evidence_kinds,
                            planning_source=model_route.planning_source,
                        )
                except (JobError, ValueError):
                    # Invalid planner JSON cannot invent or broaden a tool call.
                    chat_route = fallback_route

            intent = chat_route.intent
            self.outcome["requested_task"] = intent.value
            if chat_route.needs_retrieval:
                query_phenotype = (
                    str(chat_route.phenotype).strip()
                    if chat_route.phenotype
                    else phenotype_from_question(question, fallback="") or None
                )
            ranking_context: BiologicalContext | None = None
            if chat_route.needs_retrieval and (
                query_phenotype or intent is ChatIntent.GENE_PRIORITIZATION
            ) and (intent is not ChatIntent.GENE_PRIORITIZATION or observations):
                from ..biological_context import context_term_matches
                explicit_terms = tuple(term for term in (
                    *chat_route.context_tissues, *chat_route.context_cell_types
                ) if context_term_matches(term, question))
                try:
                    ranking_context = self.app_server.context_planner.resolve(
                        query_phenotype or source_trait,
                        explicit_terms=explicit_terms,
                        model_id=str(model.get("model_id", "")), endpoint=str(model.get("base_url", "")),
                        complete=(lambda messages: structural_completer(messages,
                            max_tokens=parameter("model.router_max_tokens"), generation_stage="context")) if use_model else None,
                    )
                    if intent is ChatIntent.GENE_PRIORITIZATION and (
                        not ranking_context.is_defined or
                        (ranking_context.biosample_mapping or {}).get("status") != "resolved"
                    ):
                        from ..biological_context import ContextResolutionError
                        raise ContextResolutionError("verified_nonempty_context_required")
                except (JobError, ValueError, RuntimeError, OSError) as exc:
                    from ..biological_context import ContextResolutionError, curated_biological_context
                    from ..model_generation import classify_generation_failure
                    reason = exc.reason if isinstance(exc, ContextResolutionError) else (
                        "context_" + classify_generation_failure(exc)["category"])
                    try:
                        ranking_context = (None if explicit_terms else curated_biological_context(
                            query_phenotype or source_trait, reason=reason))
                    except (ValueError, RuntimeError, OSError):
                        ranking_context = None
                    if ranking_context is None and intent is ChatIntent.GENE_PRIORITIZATION:
                        self.outcome["context_resolution"] = {"status": "unavailable", "reason": reason}
                        self._write_chat_stage_event("plan", "stopped", "Biological context needs clarification")
                        raise JobError("A verified biological context could not be established for gene "
                            "prioritization. Please clarify the phenotype or provide a specific relevant "
                            "tissue or cell type, then try again.") from exc
                    if ranking_context is None:
                        # Variant description can proceed without tissue claims.
                        ranking_context = BiologicalContext(
                            phenotype=query_phenotype or source_trait,
                            source="context resolution unavailable",
                            ruleset_version="2.0.0")
                if intent is ChatIntent.GENE_PRIORITIZATION and ranking_context.phenotype_scope is None:
                    from dataclasses import replace as replace_context

                    from ..phenotype_scope import freeze_phenotype_scope
                    from .phenotype_planner import PhenotypeScopePlanner
                    scope_planner = getattr(self.app_server, "phenotype_scope_planner", None) or PhenotypeScopePlanner()
                    try:
                        phenotype_scope = scope_planner.resolve(
                            query_phenotype or source_trait, biological_context=ranking_context.to_dict(),
                            cancel_event=self.cancel_event,
                            model_id=str(model.get("model_id", "")), endpoint=str(model.get("base_url", "")),
                            complete=(lambda messages: structural_completer(messages,
                                max_tokens=parameter("model.router_max_tokens"), generation_stage="phenotype_scope"))
                                if use_model else None)
                    except CancelledError:
                        raise
                    except (ValueError, RuntimeError, OSError):
                        phenotype_scope = freeze_phenotype_scope(query_phenotype or source_trait,
                            source="Phenotype scope planning unavailable", status="target_unresolved",
                            issues=("phenotype_scope_planning_unavailable",))
                    ranking_context = replace_context(ranking_context, phenotype_scope=phenotype_scope)
                self.outcome["biological_context"] = ranking_context.to_dict()

            if fixed_workflow:
                plan_summary = (
                    "Workflow ready · prior-report screening"
                    if intent is ChatIntent.VARIANT_DESCRIPTION
                    else "Workflow ready · gene prioritization"
                )
            else:
                plan_summary = f"Request understood · {intent.label}"
                if ranking_context is not None and ranking_context.is_defined:
                    plan_summary += " · phenotype context defined"
                elif chat_route.needs_retrieval:
                    plan_summary += " · no phenotype context"
            self._write_chat_stage_event(
                "plan",
                "running",
                plan_summary,
                detail=(
                    "Fixed workflow: checking prior phenotype-specific GWAS reports."
                    if fixed_workflow and intent is ChatIntent.VARIANT_DESCRIPTION
                    else "Fixed workflow: prioritizing genes for loci with no prior report retrieved."
                    if fixed_workflow else f"Intent identified: {intent.label}."
                ),
            )
            if ranking_context is not None and ranking_context.is_defined:
                self._write_chat_stage_event(
                    "plan",
                    "running",
                    plan_summary,
                    detail="Defining phenotype-relevant tissues and cell types...",
                )
                if ranking_context.phenotype_scope is not None:
                    self._write_chat_stage_event("plan", "running", plan_summary,
                        detail=_phenotype_scope_status(ranking_context.phenotype_scope))
                self._write_chat_stage_event(
                    "plan",
                    "passed",
                    plan_summary,
                    detail=(
                        "Biological context defined: "
                        f"{len(ranking_context.relevant_terms)} relevant tissues/cell types."
                    ),
                )
            else:
                self._write_chat_stage_event("plan", "passed", plan_summary)

            input_issue = _input_retrieval_precondition(observations, needs_retrieval=chat_route.needs_retrieval)
            should_submit = bool(observations and chat_route.needs_retrieval and input_issue is None)
            if not should_submit:
                response_state = "stopped" if input_issue else "passed"
                if input_issue:
                    use_model = False
                    self.outcome.update(report_quality="STOPPED", acquisition_issue=input_issue,
                        result={"status": "INVALID_INPUT", "ranked_genes": [], "evidence": [],
                                "sufficiency": {"valid_input": False}, "acquisition_issue": input_issue})
                self._write_chat_stage_event(
                    "retrieve",
                    "stopped" if input_issue else "passed",
                    "Data acquisition not started" if input_issue else "Evidence retrieval not required",
                    detail=input_issue["message"] if input_issue else "No external evidence retrieval was required.",
                    progress=({"phase": "ld_expansion", "status": "blocked", "reason": input_issue["reason_code"],
                               "source_count": len(observations)}
                              if input_issue and intent is ChatIntent.GENE_PRIORITIZATION else None),
                )
                self._write_chat_stage_event(
                    "safety",
                    "stopped" if input_issue else "passed",
                    "Safety review · STOPPED" if input_issue else "Safety review · PASS",
                )
                self._write_chat_stage_event(
                    "response",
                    "running",
                    "Generating response",
                    detail="Preparing the final response...",
                )
                if input_issue:
                    answer = (_missing_selection_answer() if not observations else
                        "**Summary**\n\nThe selected variants could not be validated. No external evidence retrieval or LD query was run.\n\n"
                        "**Evidence**\n\nNo external evidence was retrieved.\n\n"
                        "**Limitations & Uncertainty**\n\nCheck the selected variant identifiers, coordinates, alleles, genome build and uploaded measurements.\n\n"
                        "**References**\n\nNo external references were retrieved.")
                elif isinstance(context, str) and context.strip() and use_model:
                    completion = structural_completer(
                        _model_chat_messages(
                            question=question,
                            history=history,
                            analysis_context=context,
                            result=None,
                            deterministic_summary=None,
                            intent=intent,
                            source_observations=observations,
                            source_trait=source_trait,
                        ),
                        json_output=False,
                    )
                    answer = completion.content
                elif intent is ChatIntent.GENERAL:
                    bounded_history = bound_no_retrieval_history(history)
                    response_mode = classify_no_retrieval_mode(
                        question,
                        bounded_history,
                    )
                    software_facts = (
                        authoritative_gwasmind_software_facts(
                            question,
                            bounded_history,
                        )
                        if response_mode is NoRetrievalMode.SOFTWARE_GUIDANCE
                        else ()
                    )
                    no_retrieval = run_no_retrieval_response(
                        NoRetrievalContext(
                            question=question,
                            mode=response_mode,
                            selected_variant_count=len(observations),
                            authoritative_facts=software_facts,
                            conversation_history=bounded_history,
                        ),
                        (structural_completer if use_model else None),
                    )
                    answer = no_retrieval.rendered_markdown
                    if no_retrieval.used_fallback and (
                        response_mode is not NoRetrievalMode.CASUAL_GREETING
                        and not software_facts
                    ):
                        self.outcome["report_quality"] = "STOPPED"
                        response_state = "stopped"
                    elif no_retrieval.validation_errors:
                        self.outcome["report_quality"] = "PASS_WITH_WARNINGS"
                        response_state = "warning"
                elif chat_route.needs_retrieval and not observations:
                    answer = _missing_selection_answer()
                else:
                    answer = (
                        "Configure and test an LLM connection in Settings to use general "
                        "AI Chat. For locus-specific evidence retrieval, also select one or "
                        "more variants on the Manhattan plot."
                    )
                self._write_chat_stage_event(
                    "response",
                    response_state,
                    "Response stopped · reason reported"
                    if response_state == "stopped" else "Response ready",
                    metrics=self._response_stream_metrics,
                )
            else:
                shortlist_needed = intent in {
                    ChatIntent.GENE_ASSOCIATIONS,
                    ChatIntent.GENE_PRIORITIZATION,
                }
                self._write_chat_stage_event(
                    "retrieve",
                    "running",
                    "Retrieving evidence",
                    detail=(
                        ("Defining the +/-1 Mb candidate-gene window and retrieving gene-centric evidence..."
                         if parameter("v2g.pipeline") == "gene_centric" else
                         "Building the candidate-gene shortlist from the selected variants' credible-set/LD context and L2G evidence...")
                        if shortlist_needed
                        else "Planning the evidence retrieval..."
                    ),
                )
                chromosome = observations[0].get("chromosome") or "selected"
                positions = [
                    int(item["position"])
                    for item in observations
                    if item.get("position") is not None
                ]
                locus_id = (
                    f"chr{chromosome}:{min(positions)}-{max(positions)}"
                    if positions
                    else f"interactive-{observations[0]['variant_id']}"
                )
                rank_with_model = bool(use_model and intent is ChatIntent.GENE_PRIORITIZATION)
                request_value: dict[str, Any] = {
                    **self.app_server.analysis_request_defaults(),
                    "ranker": "openai" if rank_with_model else "deterministic",
                    "defer_ranking": True,
                    "queries": [
                        {
                            "locus_id": locus_id,
                            "trait": query_phenotype or source_trait,
                            "source_trait": source_trait,
                            "variants": [
                                {key: item[key] for key in ("variant_id", "p_value", "chromosome", "position") if key in item}
                                for item in observations
                            ],
                            "candidate_genes": [],
                            "task": intent.value,
                            "relevant_tissues": (
                                list(ranking_context.relevant_terms)
                                if ranking_context is not None
                                else []
                            ),
                            "biological_context": ranking_context.to_dict() if ranking_context else None,
                            "qtl_types": list(chat_route.qtl_types),
                            "requested_evidence_kinds": list(chat_route.evidence_kinds),
                            "genome_build": "GRCh38",
                            "benchmark": False,
                        }
                    ],
                }
                if rank_with_model:
                    request_value.update(
                        {
                            "model_id": str(model.get("model_id") or DEFAULT_MODEL_ID),
                            "base_url": str(model.get("base_url") or DEFAULT_MODEL_BASE_URL),
                            "thinking_mode": str(model.get("thinking_mode") or THINKING_ENABLED),
                            "reasoning_effort": str(
                                model.get("reasoning_effort") or DEFAULT_REASONING_EFFORT
                            ),
                            "model_provider": str(model.get("provider_id") or "openai-compatible"),
                        }
                    )
                ld_expansion = None
                if intent is ChatIntent.GENE_PRIORITIZATION:
                    self._write_chat_stage_event(
                        "retrieve", "running", "Retrieving evidence",
                        detail="Checking population LD proxies for gene prioritization...",
                    )
                    ld_expansion = self._expand_ld_with_progress(
                        observations=observations, question=question,
                        backend=None if request_value.get("mode") == "live" else _OfflineRankingLDReference(),
                        cache_directory=self.app_server.manager.artifact_root / "public_evidence_cache" / "ld_reference",
                    )
                    self.outcome["ld_expansion"] = ld_expansion
                    self._write_chat_stage_event(
                        "retrieve", "running", "Retrieving evidence",
                        detail=(
                            f"Retained {len(observations)} selected SNP(s) and added "
                            f"{ld_expansion['added_variant_count']} verified LD proxy variant(s) "
                            f"for gene prioritization (r² ≥ {ld_expansion['threshold']:g}, "
                            f"{ld_expansion['population'] or 'population unresolved'}, "
                            f"±{ld_expansion['window_bp'] / 1000:g} kb)."
                        ),
                    )
                submit_options = {"ld_expansion": ld_expansion} if ld_expansion is not None else {}
                snapshot = self.app_server.manager.submit(request_value, **submit_options)
                run_id = str(snapshot["run_id"])
                self.outcome["run_id"] = run_id
                self.on_run(run_id)

                sequence = 0
                worker_events: list[Mapping[str, Any]] = []
                l2g_progress_shown = shortlist_needed
                while True:
                    events, terminal = self._run_stage_operation(
                        lambda: self.app_server.manager.wait_for_events(
                            run_id,
                            after_sequence=sequence,
                            timeout=self.app_server.heartbeat_interval,
                        ), stage="retrieve",
                        detail="Waiting for the current analysis and its saved artifacts...", announce=False,
                    )
                    for event in events:
                        sequence = int(event["seq"])
                        _retain_worker_event(worker_events, event)
                        phase = str(event.get("phase", ""))
                        if phase == "gene_evidence_cards":
                            cards = (event.get("details") or {}).get("cards")
                            if isinstance(cards, list):
                                self.outcome["gene_evidence_cards"] = cards
                                labels = (event.get("details") or {}).get("variant_labels")
                                if isinstance(labels, list):
                                    self.outcome["variant_labels"] = labels
                                self.emit({"type": "gene_evidence_cards", "cards": cards,
                                           "variant_labels": self.outcome.get("variant_labels", [])})
                            continue
                        processing_progress = _worker_result_processing_progress(event)
                        if processing_progress:
                            self._write_chat_stage_event("retrieve", "running", "Preparing retained evidence",
                                detail=_retrieval_progress_detail(processing_progress), progress=processing_progress)
                            continue
                        if phase == "result_processing":
                            detail = event.get("details") or {}
                            if detail.get("status") in {"importing", "loading", "completed"} and "records" in detail:
                                public = _public_retrieval_progress({"phase": "result_processing",
                                    "status": "completed" if detail["status"] == "completed" else "running",
                                    "completed_count": detail.get("records"), "total_count": detail.get("total_records"),
                                    "records": detail.get("records"), "unit": "records"})
                                self._write_chat_stage_event("retrieve", "running", "Retrieving evidence",
                                    detail=_retrieval_progress_detail(public), progress=public)
                                continue
                            retained = detail.get("complete_result_retained") is True
                            self._write_chat_stage_event("retrieve", "running", "Retrieving evidence",
                                detail=("The complete result was preserved; preparing a source-only response."
                                        if retained else "Checking the retained result before source-only reporting..."))
                            continue
                        if phase in {
                            "checking_sufficiency",
                            "safety_gate_passed",
                            "safety_gate_stopped",
                            "writing_artifacts",
                            "completed",
                            "synthesizing_decision",
                            "synthesis_completed",
                        }:
                            continue
                        public_progress = _worker_retrieval_progress(event)
                        if public_progress:
                            self._write_chat_stage_event("retrieve", "running", "Retrieving evidence",
                                detail=_retrieval_progress_detail(public_progress), progress=public_progress)
                            if phase != "semantic_tool_finished":
                                continue
                        for _stage, progress_content in _public_chat_job_statuses(event):
                            if l2g_progress_shown and "L2G" in progress_content:
                                continue
                            self._write_chat_stage_event(
                                "retrieve",
                                "running",
                                "Retrieving evidence",
                                detail=progress_content,
                            )
                    if terminal:
                        break
                    self.emit({"type": "heartbeat"})

                result = self._run_stage_operation(
                    lambda: self.app_server.manager.result(run_id), stage="retrieve",
                    detail="Preparing the saved evidence for review...",
                )
                if result["state"] != "completed":
                    self._write_chat_stage_event(
                        "retrieve",
                        "failed",
                        "Retrieving evidence",
                    )
                    self._write_chat_event("error", str(result.get("error") or result["state"]))
                    return
                result_value = result["results"][0]
                processing_limited = is_source_only_processing_limit(result_value)
                if processing_limited:
                    # The loader verified the retained bytes, not their scientific
                    # contents. Explain that known limit without another model call.
                    use_model = False
                    self.outcome["result_processing_limit"] = result_value["result_processing_limit"]
                if ld_expansion is not None and not processing_limited:
                    result_value = {**result_value, "ld_expansion": result_value.get("ld_expansion", ld_expansion)}
                    self.outcome["ld_expansion"] = result_value["ld_expansion"]
                self.outcome["result"] = result_value
                prior_not_run = prior_screen_precondition(
                    observations=observations, phenotype=query_phenotype or source_trait,
                    native_result=result_value,
                )
                if prior_not_run is not None:
                    self.outcome["acquisition_issue"] = {
                        "reason_code": prior_not_run["precondition_reason"],
                        "message": prior_not_run["precondition_message"],
                        "source_only": not result_value.get("evidence"),
                    }
                self._begin_scientific_review(worker_events, processing_limited=processing_limited)
                bridge = self._run_stage_operation(
                    lambda: build_runtime_stage_bridge(
                        observations=observations,
                        progress_events=worker_events,
                        result=result_value,
                        intent=intent.value,
                        biological_context=ranking_context or {},
                        genome_build="GRCh38",
                        question=question,
                    ), stage="safety",
                    detail="Organizing evidence relationships and checking retrieval coverage...",
                )
                result_value = {
                    **result_value,
                    "retrieval_coverage": bridge.retrieval_coverage,
                    "retrieval_metrics": bridge.retrieval_metrics.to_dict(),
                    "exploratory_ranking_allowed": self._run_stage_operation(
                        lambda: bridge.exploratory_ranking_allowed, stage="safety",
                        detail="Checking evidence sufficiency and reviewing the supported conclusions...",
                    ),
                    "variant_resolution": self._run_stage_operation(
                        lambda: variant_identity_context(
                            observations=observations, evidence=bridge.evidence,
                            genome_build="GRCh38", progress_events=() if processing_limited else worker_events,
                            normalizations=(ld_expansion or {}).get("source_identity_normalizations", []),
                        ), stage="safety",
                        detail="Checking selected and linked variant identities against the saved evidence...",
                    ),
                }
                self.outcome["result"] = result_value
                public_metrics = {} if processing_limited else _public_stage_metrics(bridge.retrieval_metrics)
                operational_facts = self._run_stage_operation(bridge.evaluate_safety, stage="safety",
                    detail="Checking evidence sufficiency and reviewing the supported conclusions...")
                result_value["ld_review_policy"] = dict(operational_facts.review_scope.get("ld_review_policy", {}))
                result_value["retrieval_review_policy"] = {
                    "bounded_literature_modules": list(operational_facts.review_scope.get("bounded_literature_modules", []))}
                if (
                    query_phenotype and is_novelty_question(question)
                    and self.auto_context.get("ranking_mode") != "exploratory"
                    and operational_facts.status is not SafetyStatus.STOPPED
                ):
                    if not processing_limited and prior_not_run is None:
                        self._write_chat_stage_event(
                            "safety", "running", "Safety review",
                            detail="Checking prior phenotype associations and reference LD...",
                        )
                    result_value["novelty_screen"] = prior_not_run or screen_prior_associations(
                        observations=observations, phenotype=query_phenotype,
                        question=question, genome_build="GRCh38",
                        native_result=result_value,
                        progress=lambda detail: self._write_chat_stage_event(
                            "safety", "running", "Safety review", detail=detail,
                        ),
                        backend=None if request_value.get("mode") == "live" else _OfflineNoveltyReference(),
                        require_model_phenotype=True,
                        phenotype_completer=phenotype_completer if use_model else None,
                        phenotype_model_id=str(model.get("model_id") or ""),
                        cancel_event=self.cancel_event,
                    )
                    self.outcome["result"] = result_value
                blocking_operations = tuple(issue for issue in operational_facts.operation_issues
                    if issue.get("blocking") is True)
                if blocking_operations:
                    use_model = False
                self._write_chat_stage_event(
                    "retrieve",
                    "stopped" if blocking_operations else "warning" if processing_limited or prior_not_run else "passed",
                    "Data acquisition stopped" if blocking_operations else
                    "External evidence unavailable · source data retained" if prior_not_run else
                    RESULT_PROCESSING_SUMMARY if processing_limited else "Retrieving evidence",
                    detail=" ".join(str(issue.get("message", "")) for issue in blocking_operations)
                    if blocking_operations else prior_not_run["precondition_message"] if prior_not_run else
                    RESULT_PROCESSING_LIMIT_MESSAGE if processing_limited else None,
                    metrics=public_metrics,
                )

                safety = self._run_stage_operation(
                    lambda: run_safety_audit(
                        self._auto_safety_context(bridge.evaluate_safety()),
                        # SSR is a fixed scientific rule set. Its result and
                        # explanation are reproducible without another model
                        # classification; the model reasons over these facts
                        # when ranking and writing the final interpretation.
                        None,
                    ), stage="safety",
                    detail="Checking evidence sufficiency and reviewing the supported conclusions...",
                )
                review_execution = dict(safety.public_review_metadata)
                # Retain the first model review's diagnostics before a rule-only
                # review replaces it; these go only to the local answer audit.
                safety_validation_issues = (() if review_execution.get("model_review_status") == "not_requested"
                                            else tuple(safety.validation_errors))
                self.outcome["safety_review_execution"] = review_execution
                result_value["safety_review_execution"] = review_execution
                if review_execution.get("model_review_status") == "unavailable":
                    # Keep the failed model attempt visible through the later
                    # rule-only ranking review instead of silently overwriting it.
                    from dataclasses import replace
                    bridge = replace(bridge, model_review_unavailable=True)
                    safety = self._run_stage_operation(
                        lambda: run_safety_audit(
                            self._auto_safety_context(bridge.evaluate_safety()), None,
                        ), stage="safety", detail="Completing the evidence checks after the review attempt...",
                    )
                self.outcome["safety_review"] = safety.decision.value
                self.outcome["report_quality"] = safety.decision.value
                result_value["analysis_capabilities"] = {
                    "report_allowed": safety.review.facts.report_allowed,
                    "ranking_allowed": safety.review.facts.ranking_allowed,
                }
                self.outcome["ranking_mode"] = (
                    self.auto_context.get("ranking_mode", "standard")
                    if intent is ChatIntent.GENE_PRIORITIZATION else "not_started"
                )
                public_warnings = public_safety_warning_messages(safety.review)
                warning_count = len(safety.review.facts.academic_warnings)
                safety_state = {
                    SafetyStatus.PASS: "passed",
                    SafetyStatus.PASS_WITH_WARNINGS: "warning",
                    SafetyStatus.STOPPED: "stopped",
                }[safety.decision]
                safety_summary = f"Safety review · {safety.decision.value}"
                if warning_count:
                    noun = "warning" if warning_count == 1 else "warnings"
                    safety_summary += f" · {warning_count} {noun}"
                if safety.decision is SafetyStatus.PASS_WITH_WARNINGS:
                    safety_details = tuple(f"Safety warning: {message}" for message in public_warnings)
                elif safety.decision is SafetyStatus.STOPPED:
                    safety_details = (_public_safety_stop_detail(safety),)
                else:
                    safety_details = ("Scientific review completed without a triggered warning; method notes and unassessed questions remain available.",)
                for safety_detail in safety_details:
                    self._write_chat_stage_event(
                        "safety", safety_state, safety_summary, detail=safety_detail,
                    )
                self.emit({"type": "review_update", "stage": "safety", "state": safety_state,
                    "summary": safety_summary, "warnings": list(public_warnings),
                    "ld_review_policy": result_value.get("ld_review_policy", {}),
                    "requested_task": intent.value,
                    **{key: safety.review.to_dict()[key] for key in (
                        "answerability", "academic_warnings", "operation_issues", "method_notes",
                        "review_policy_version", "review_scope_sha256"
                    ) if key in safety.review.to_dict()},
                    "gate_reviews": safety.review.to_dict()["gate_reviews"],
                    "gate_warning_items": list(public_safety_warning_items(safety.review))})

                self._write_chat_stage_event(
                    "response",
                    "running",
                    "Generating response",
                    detail=(
                        "Preparing the safe abstention response..."
                        if safety.decision is SafetyStatus.STOPPED
                        else "Generating the evidence-grounded response..."
                    ),
                )
                if (
                    intent is ChatIntent.GENE_PRIORITIZATION
                    and rank_with_model
                    and safety.review.facts.ranking_allowed
                    and not processing_limited
                ):
                    self._run_stage_operation(
                        lambda: self.app_server.manager.resume_ranking(
                            run_id, safety_status=safety.decision.value,
                        ), stage="response", detail="Saving the reviewed evidence and preparing candidate ranking...",
                    )
                    ranking_started = time.monotonic()
                    ranking_activity = "Candidate ranking in progress..."
                    while True:
                        events, terminal = self._run_stage_operation(
                            lambda: self.app_server.manager.wait_for_events(
                                run_id,
                                after_sequence=sequence,
                                timeout=self.app_server.heartbeat_interval,
                            ), stage="response",
                            detail="Waiting for the current analysis and its saved artifacts...", announce=False,
                        )
                        for event in events:
                            sequence = int(event["seq"])
                            _retain_worker_event(worker_events, event)
                            if event.get("phase") == "ranking_input_frozen":
                                descriptor = self.app_server.manager.ranking_evidence_downloads(
                                    run_id, int(event.get("query_index", 0)),
                                )
                                snapshots = self.outcome.setdefault("ranking_evidence", [])
                                if descriptor not in snapshots:
                                    snapshots.append(descriptor)
                                self.emit({"type": "ranking_evidence", **descriptor})
                                continue
                            if event.get("phase") == "ranking_validation":
                                ranking_activity = "Checking ranking claims against the retrieved evidence..."
                                self._write_chat_stage_event("response", "running", "Generating response",
                                    detail=ranking_activity, substage="fact_verification")
                                continue
                            if event.get("phase") == "model_generation_attempt":
                                with self._response_stream_lock:
                                    self._model_usage.ranking_event(event.get("details") or {})
                                if (event.get("details") or {}).get("status") == "started":
                                    ranking_activity = "Candidate ranking in progress..."
                                    self._write_chat_stage_event("response", "running", "Generating response",
                                        detail=ranking_activity, substage="inference")
                                else:
                                    self._write_chat_stage_event("response", "running", "Generating response",
                                        metrics=self._response_stream_metrics)
                                continue
                            if event.get("phase") == "synthesis_retry":
                                details = event.get("details") or {}
                                count = details.get("attempt")
                                if type(count) is int and 1 <= count <= 3:
                                    ranking_activity = f"Retrying candidate ranking from saved evidence (attempt {count}/3)."
                                    self._write_chat_stage_event("response", "running", "Generating response",
                                        detail=ranking_activity)
                                continue
                            if event.get("phase") not in {
                                "synthesizing_decision",
                                "synthesis_completed",
                                "synthesis_stream",
                            }:
                                continue
                            if event.get("phase") == "synthesis_stream":
                                details = event.get("details", {})
                                if isinstance(details, Mapping):
                                    with self._response_stream_lock:
                                        self._model_usage.stream(details, cumulative=True)
                                    self._response_stream_metrics = {
                                        key: val for key, val in details.items()
                                        if key in {
                                            "output_tokens", "estimated_output_tokens",
                                            "received_chars",
                                        }
                                        and isinstance(val, int) and not isinstance(val, bool)
                                        and 0 <= val <= 10_000_000
                                    }
                                self._write_chat_stage_event(
                                    "response", "running", "Generating response",
                                    metrics=self._response_stream_metrics,
                                )
                                continue
                            for _stage, content in _public_chat_job_statuses(event):
                                details = event.get("details", {})
                                tokens = details.get("output_tokens")
                                if event.get("phase") == "synthesis_completed" and (
                                    isinstance(tokens, int) and not isinstance(tokens, bool)
                                    and 0 <= tokens <= 10_000_000
                                ):
                                    self._response_completed_tokens = tokens
                                    self._response_completed_tokens_exact = True
                                if event.get("phase") == "synthesis_completed":
                                    with self._response_stream_lock:
                                        self._model_usage.ranking_review(event.get("details") or {})
                                    prior = self._response_stream_metrics
                                    self._response_completed_chars = prior.get("received_chars", 0)
                                    if type(tokens) is not int and "estimated_output_tokens" in prior:
                                        self._response_completed_tokens = prior[
                                            "estimated_output_tokens"
                                        ]
                                        self._response_completed_tokens_exact = False
                                self._write_chat_stage_event(
                                    "response",
                                    "running",
                                    "Generating response",
                                    detail=(
                                        "Prioritizing candidate genes from reviewed evidence..."
                                        if event.get("phase") == "synthesizing_decision"
                                        else content
                                    ),
                                    metrics=(
                                        {
                                            ("output_tokens"
                                             if self._response_completed_tokens_exact
                                             else "estimated_output_tokens"):
                                                self._response_completed_tokens
                                        }
                                        if event.get("phase") == "synthesis_completed"
                                        and isinstance(tokens, int) else None
                                    ),
                                )
                        if terminal:
                            break
                        self._write_chat_stage_event(
                            "response",
                            "running",
                            "Generating response",
                            detail=ranking_activity,
                            metrics={
                                "candidate_genes": bridge.retrieval_metrics.candidate_genes,
                                "ranking_elapsed_seconds": max(
                                    1,
                                    round(time.monotonic() - ranking_started),
                                ),
                            },
                        )
                    ranked_result = self._run_stage_operation(
                        lambda: self.app_server.manager.result(run_id), stage="response",
                        detail="Preparing the saved ranking for final review...",
                    )
                    if ranked_result["state"] != "completed":
                        if ranked_result["state"] == "cancelled":
                            raise CancelledError("Analysis was cancelled.")
                        # The completed retrieval remains available even when
                        # the separate ranking worker cannot produce a result.
                        result_value = {**result_value, "status": "INCOMPLETE", "ranked_genes": [],
                            "ranking_unavailable": True,
                            "decision": {"status": "ranking_unavailable", "ranking_attempted": True,
                                         "reason_codes": ["ranking_worker_incomplete"]},
                            "ranking_review": {"method": "model_unavailable", "validation_passed": False},
                            "summary": RANKING_UNAVAILABLE_DETAIL}
                    else:
                        # The ranking worker owns rankings, not the earlier
                        # completed prior-association screen. Preserve that
                        # receipt for Gate1, the final writer and Auto Analysis.
                        prior_association_screen = result_value.get("novelty_screen")
                        result_value = live_model_only_result(ranked_result["results"][0])
                        if isinstance(prior_association_screen, Mapping):
                            result_value["novelty_screen"] = dict(prior_association_screen)
                    result_value["safety_review_execution"] = dict(
                        self.outcome.get("safety_review_execution", {}))
                    if ld_expansion is not None:
                        result_value = {**result_value, "ld_expansion": result_value.get("ld_expansion", ld_expansion)}
                        self.outcome["ld_expansion"] = result_value["ld_expansion"]
                    self.outcome["result"] = result_value
                    if self.outcome.get("gene_evidence_cards"):
                        ranked_cards = rank_public_cards(self.outcome["gene_evidence_cards"],
                                                         result_value.get("ranked_genes") or [],
                                                         result_value.get("candidate_decisions") or [])
                        self.outcome["gene_evidence_cards"] = ranked_cards
                        self.emit({"type": "gene_evidence_cards", "cards": ranked_cards,
                                   "variant_labels": self.outcome.get("variant_labels", [])})
                    ranking_bridge = self._run_stage_operation(
                        lambda: build_runtime_stage_bridge(
                            observations=observations,
                            progress_events=worker_events,
                            result=result_value,
                            intent=intent.value,
                            biological_context=ranking_context or {},
                            genome_build="GRCh38",
                            question=question,
                        ), stage="response",
                        detail="Organizing evidence relationships and checking retrieval coverage...",
                    )
                    result_value = {
                        **result_value,
                        "retrieval_coverage": ranking_bridge.retrieval_coverage,
                        "ld_review_policy": dict(ranking_bridge.evaluate_safety().review_scope.get("ld_review_policy", {})),
                        "retrieval_review_policy": {
                            "bounded_literature_modules": list(ranking_bridge.evaluate_safety().review_scope.get("bounded_literature_modules", []))},
                        "retrieval_metrics": ranking_bridge.retrieval_metrics.to_dict(),
                        "exploratory_ranking_allowed": ranking_bridge.exploratory_ranking_allowed,
                        "variant_resolution": self._run_stage_operation(
                            lambda: variant_identity_context(
                                observations=observations, evidence=ranking_bridge.evidence,
                                genome_build="GRCh38", progress_events=worker_events,
                                normalizations=(ld_expansion or {}).get("source_identity_normalizations", []),
                            ), stage="response",
                            detail="Checking selected and linked variant identities against the saved evidence...",
                        ),
                    }
                    self.outcome["result"] = result_value
                    # Output validation belongs to Stage four. Carry recoveries
                    from ..report_facts import ranking_provenance
                    result_value["ranking_review"] = result_value.get("ranking_review") or next((dict((event.get("details") or {})["gate2_review"])
                        for event in reversed(worker_events) if event.get("phase") == "synthesis_completed"
                        and isinstance((event.get("details") or {}).get("gate2_review"), Mapping)), {})
                    result_value["ranking_provenance"] = ranking_provenance(
                        result_value, fallback=ranking_bridge.ranking_used_fallback)
                    # or a failed ranking into report quality without re-running
                    # the data review or pretending to retrieve evidence again.
                    if (
                        ranking_bridge.ranking_repair_attempts
                        or ranking_bridge.ranking_used_fallback
                        or ranking_bridge.ranking_validation_failures
                        or ranking_bridge.ranking_projection_omitted_records
                        or str(result_value.get("status")) != "PRIORITIZED"
                    ):
                        safety = self._run_stage_operation(
                            lambda: run_safety_audit(self._auto_safety_context(preserve_gate1_warnings(
                                ranking_bridge.evaluate_safety(), safety.review.facts,
                            )), None), stage="response",
                            detail="Checking the ranking outcome while preserving the evidence limitations...",
                        )

                if (intent is ChatIntent.GENE_PRIORITIZATION and not rank_with_model
                        and safety.review.facts.ranking_allowed and not processing_limited):
                    result_value = {**result_value, "status": "INCOMPLETE", "ranked_genes": [],
                        "ranking_unavailable": True,
                        "decision": {"status": "ranking_unavailable", "ranking_attempted": False,
                                     "reason_codes": ["verified_ranking_model_unavailable"]},
                        "summary": RANKING_NOT_CONFIGURED_DETAIL}
                unavailable_ranking = model_ranking_unavailable(result_value)
                self.outcome["result"] = result_value
                if unavailable_ranking:
                    self.outcome["ranking_mode"] = "unavailable"
                    self.outcome["ranking_unavailable"] = True
                    self._write_chat_stage_event("response", "running", "Generating response",
                        detail=RANKING_UNAVAILABLE_DETAIL if rank_with_model else RANKING_NOT_CONFIGURED_DETAIL,
                        ranking_unavailable=True)
                if intent is ChatIntent.GENE_PRIORITIZATION and not safety.review.facts.ranking_allowed:
                    # Deferred retrieval is already terminal in JobManager and
                    # holds no worker/model slot. Keep that usable completed
                    # evidence instead of relabelling it as user cancellation.
                    if not unavailable_ranking:
                        self.outcome["ranking_mode"] = "not_started"
                self.outcome["safety_review"] = safety.decision.value
                result_value["analysis_capabilities"] = {
                    "report_allowed": safety.review.facts.report_allowed,
                    "ranking_allowed": safety.review.facts.ranking_allowed,
                }
                if self.auto_context:
                    result_value["auto_analysis_context"] = dict(self.auto_context)
                # Auto has a narrower final decision scope; preserve the shared
                # receipt and the completed Gate review exactly as evaluated.
                if (self.auto_context.get("analysis_mode") == "prior_report_screening"
                        and safety.decision is not SafetyStatus.STOPPED
                        and isinstance(result_value.get("novelty_screen"), Mapping)):
                    result_value["auto_screening"] = finalize_auto_screen(
                        result_value["novelty_screen"], native_result=result_value,
                        observations=observations,
                    )

                identified = result_value["variant_resolution"]["source_to_identified_variant"]
                display_observations = [
                    {**item, "variant_id": identified.get(str(item["variant_id"]), item["variant_id"])}
                    for item in observations
                ]
                publication_enrichment: dict[str, Any] = {}
                final_context = self._prepare_hybrid_final_with_progress(
                    intent=intent,
                    result=result_value,
                    display_observations=display_observations,
                    source_trait=source_trait,
                    query_phenotype=query_phenotype,
                    region=region,
                    biological_context=ranking_context,
                    question=question,
                    source_observations=observations,
                    safety_review=safety,
                    publication_enrichment=publication_enrichment,
                    live=request_value.get("mode") == "live",
                )
                final_result, response_metrics = self._run_hybrid_final_with_progress(
                    final_context,
                    structural_completer if use_model else None,
                )
                from .output_review import (
                    combined_output_review,
                    public_ranking_review,
                    visible_report_quality,
                    visible_writer_warning_messages,
                )
                report_review = final_result.gate2_review
                answer = final_result.rendered_markdown
                answer = _auto_report_warning_scope(answer, result_value.get("auto_screening"), safety.review)
                output_review = combined_output_review(report_review,
                    visible_output=final_result.validated_json, rendered_markdown=answer)
                with self._response_stream_lock:
                    self._model_usage.report_review(report_review)
                self.outcome["gate2_review"] = output_review
                result_value["gate2_review"] = output_review
                self.outcome["report_quality"] = visible_report_quality(
                    safety.decision.value, final_result.status.value, output_review)
                if self.outcome["report_quality"] != final_result.status.value:
                    answer = answer.replace(f"**Report quality: {final_result.status.value}**",
                                            f"**Report quality: {self.outcome['report_quality']}**", 1)
                recovered_evidence = final_result.source in {"evidence_only_recovery", "partial_recovery"}
                fallback_used = final_result.source in {"deterministic_fallback", "evidence_only_recovery", "partial_recovery"}
                response_stopped = final_result.status is SafetyStatus.STOPPED
                self.outcome["response_source"] = final_result.source
                self.outcome["interpretation_status"] = (
                    "stopped" if response_stopped else "incomplete" if recovered_evidence or processing_limited else "complete"
                )
                final_warnings = tuple(dict.fromkeys((
                    *public_safety_warning_messages(safety.review),
                    *("Response limitation: " + warning for warning in visible_writer_warning_messages(
                        _writer_warning_messages(final_result.source, final_result.validation_errors), output_review)),
                )))
                # This replaces the displayed review conclusions; it does not
                # start another safety/model call or move the active stage back.
                self.emit({
                    "type": "review_update", "stage": "safety",
                    "state": "stopped" if response_stopped else "warning" if final_warnings else "passed",
                    "summary": f"Safety review · {final_result.status.value}",
                    "warnings": list(final_warnings),
                    "ld_review_policy": result_value.get("ld_review_policy", {}),
                    "requested_task": intent.value,
                    **{key: safety.review.to_dict()[key] for key in (
                        "answerability", "academic_warnings", "operation_issues", "method_notes",
                        "review_policy_version", "review_scope_sha256"
                    ) if key in safety.review.to_dict()},
                    "gate_reviews": safety.review.to_dict()["gate_reviews"],
                    "gate_warning_items": list(public_safety_warning_items(safety.review)),
                    "output_review": output_review,
                    "ranking_review": public_ranking_review(result_value.get("ranking_review")),
                })
                self._write_chat_stage_event(
                    "response",
                    "stopped"
                    if response_stopped
                    else "warning"
                    if unavailable_ranking or output_review.get("status") == "PASS_WITH_WARNINGS"
                    else "passed",
                    (
                        "Response stopped · reason reported"
                        if response_stopped
                        else "Source GWAS summary ready · external review incomplete"
                        if processing_limited
                        else "Verified evidence ready · interpretation incomplete"
                        if recovered_evidence
                        else RANKING_UNAVAILABLE_SUMMARY
                        if unavailable_ranking
                        else "Response ready"
                    ),
                    detail=(
                        "Formatting the response and references..." if not fallback_used else None
                    ),
                    metrics=response_metrics,
                    ranking_unavailable=unavailable_ranking,
                )
                fact_review_output = None
                if final_result.raw_json or final_result.raw_json_attempts:
                    try:
                        fact_review_output = self.app_server.manager.record_chat_fact_review(
                            run_id, raw_json=final_result.raw_json,
                            corrected_json=final_result.corrected_json,
                            raw_json_attempts=final_result.raw_json_attempts,
                            machine_audit=final_result.machine_audit,
                        )
                    except (JobError, OSError, ValueError) as exc:
                        writer_output_audit_errors.append("fact_review:" + type(exc).__name__)
                try:
                    self._run_stage_operation(lambda: self.app_server.manager.record_chat_answer_outcome(
                        run_id,
                        response_source=(
                            "deterministic_safety_stop"
                            if safety.decision is SafetyStatus.STOPPED
                            else "response_validation_stopped"
                            if response_stopped
                            else "evidence_only_recovery"
                            if final_result.source == "evidence_only_recovery"
                            else "partial_recovery"
                            if final_result.source == "partial_recovery"
                            else "deterministic_repair"
                            if final_result.source == "deterministic_repair"
                            else "model"
                            if final_result.source == "model"
                            else "model_repaired"
                            if final_result.source == "model_repaired"
                            else "deterministic_invalid_model_output"
                            if use_model
                            else "deterministic_no_model"
                        ),
                        validation_issues=tuple(final_result.validation_errors),
                        output_review=output_review,
                        generation_attempts={key: value.public() for key, value in generation_by_stage.items()},
                        safety_review_execution={**self.outcome.get("safety_review_execution", {}),
                            "validation_issues": list(safety_validation_issues)},
                        publication_enrichment=publication_enrichment,
                        model_outputs=tuple(writer_outputs),
                        model_output_audit_errors=tuple(writer_output_audit_errors),
                        fact_review_output=fact_review_output,
                    ), stage="response", announce=False,
                        detail="Saving the completed report audit...")
                except (JobError, OSError, ValueError):
                    pass

            if isinstance(self.outcome.get("result"), Mapping):
                from ..report_facts import (
                    analysis_status,
                    computed_report_facts,
                    retrieval_incomplete_for_report,
                    status_markdown,
                )
                final_value = self.outcome["result"]
                # Preserve the native ranking method before report Gate 2 can replace its review.
                incomplete = retrieval_incomplete_for_report(final_value)
                status = analysis_status(final_value,
                    report_source=self.outcome.get("response_source", "evidence_summary"),
                    report_quality=self.outcome.get("report_quality", "STOPPED"),
                    explanation_model=str(model.get("model_id", "")), retrieval_incomplete=incomplete,
                    ranking_requested=intent is ChatIntent.GENE_PRIORITIZATION)
                facts = computed_report_facts(observations, final_value)
                self.outcome["analysis_status"] = status
                self.outcome["computed_facts"] = facts
                self.outcome["generation_attempts"] = {key: value.public() for key, value in generation_by_stage.items()}
                final_value["analysis_status"] = status
                final_value["computed_facts"] = facts
                status_text = status_markdown(status, facts)
                answer = answer.replace("**Summary**", status_text + "**Summary**", 1) if "**Summary**" in answer else answer + "\n\n" + status_text
                self.emit({"type": "analysis_status", "analysis_status": status, "computed_facts": facts})
                from ..report_facts import public_source_normalizations
                normalization_notice = public_source_normalizations(final_value)
                if normalization_notice:
                    self.outcome["variant_normalizations"] = normalization_notice
                    self.emit({"type": "variant_normalization", "normalizations": normalization_notice})
            self.outcome["report_markdown"] = answer
            self.outcome["source_trait"] = source_trait
            self.outcome["query_phenotype"] = query_phenotype
            for start in range(0, len(answer), 320):
                self._write_chat_event("answer_chunk", answer[start : start + 320])
            self._write_chat_event("complete", "Evidence summary complete.")

        except _CLIENT_DISCONNECT_ERRORS:
            if run_id is not None:
                try:
                    snapshot = self.app_server.manager.snapshot(run_id)
                    if snapshot["state"] not in TERMINAL_STATES:
                        self.app_server.manager.cancel(run_id)
                except JobError:
                    pass

        except CancelledError:
            self.outcome["cancelled"] = True
            self.outcome["interpretation_status"] = "stopped"
            self._write_chat_stage_event("response", "stopped", "Response cancelled")
            self._finish_chat_stream_error("Analysis was cancelled.")
        except JobError as exc:
            self._finish_chat_stream_error(str(exc))
        except Exception:
            self._finish_chat_stream_error(
                "The local analysis service failed before producing a response."
            )

    def _finish_chat_stream_error(self, message: str) -> None:
        self.outcome["error"] = message
        self.outcome["report_quality"] = "STOPPED"
        self._write_chat_event("error", message)

    def _begin_scientific_review(self, worker_events, *, processing_limited: bool) -> None:
        """Close data loading before building and evaluating scientific facts."""
        if "safety" not in self._stage_started:
            early_seconds = _worker_sufficiency_seconds(worker_events)
            retrieval_started = self._stage_started.get("retrieve")
            available = max(0, time.monotonic() - retrieval_started) if retrieval_started is not None else 0
            # An interval outside the observed workflow duration is unusable;
            # never guess or stretch a duration from inconsistent old receipts.
            if early_seconds is not None and early_seconds > available:
                early_seconds = None
            if early_seconds is not None:
                self._stage_time_offsets.update(retrieve=-early_seconds, safety=early_seconds)
            self.outcome["stage_timing"] = {
                "early_sufficiency_seconds": early_seconds,
                "early_sufficiency_basis": "paired_worker_phase_timestamps",
            }
        acquisition_issue = self.outcome.get("acquisition_issue") or {}
        self._write_chat_stage_event(
            "retrieve", "warning" if processing_limited or acquisition_issue else "passed",
            "External evidence unavailable · source data retained" if acquisition_issue else
            RESULT_PROCESSING_SUMMARY if processing_limited else "Retrieving evidence",
            detail=acquisition_issue.get("message") or (RESULT_PROCESSING_LIMIT_MESSAGE if processing_limited else None),
        )
        self._write_chat_stage_event(
            "safety", "running", "Safety review",
            detail="Organizing and validating the retrieved evidence...",
        )

    def _run_stage_operation(self, operation, *, stage, detail, announce=True):
        """Keep this request live while an isolated operation computes or waits.

        Only the owner thread writes SSE. The worker retains the request context;
        cancellation/disconnection stops further waiting messages and is visible
        to an in-flight model call through the workflow's cancellation event.
        """
        if self.cancel_event is None:
            self.cancel_event = threading.Event()
        if self.cancel_event.is_set():
            raise CancelledError("Analysis was cancelled.")
        summary = {"retrieve": "Retrieving evidence", "safety": "Safety review",
                   "response": "Generating response"}[stage]
        if announce:
            self._write_chat_stage_event(stage, "running", summary, detail=detail)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gwasmind-stage-operation")
        future = executor.submit(copy_context().run, operation)
        try:
            while True:
                if self.cancel_event.is_set():
                    raise CancelledError("Analysis was cancelled.")
                try:
                    result = future.result(timeout=parameter("execution.stream_progress_interval_seconds"))
                except FutureTimeoutError:
                    # A TimeoutError raised by the operation is its real failure,
                    # not a signal to wait indefinitely for an already-done task.
                    if future.done():
                        result = future.result()
                    else:
                        # Event polling preserves the last concrete operation;
                        # only an explicitly announced operation owns a stage row.
                        self.emit({"type": "heartbeat"})
                        continue
                if self.cancel_event.is_set():
                    raise CancelledError("Analysis was cancelled.")
                return result
        except (CancelledError, *_CLIENT_DISCONNECT_ERRORS):
            self.cancel_event.set()
            future.cancel()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _expand_ld_with_progress(self, **arguments):
        """Forward callbacks on the owner thread and report an actually pending lookup."""
        pending = _LDProgressMailbox()
        stopped = threading.Event()
        def progress_sink(phase, details):
            if not stopped.is_set():
                pending.put({"phase": phase, **details})
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gwasmind-ld-progress")
        future = executor.submit(expand_ranking_variants, **arguments, progress_sink=progress_sink,
                                 cancel_event=stopped)
        current = None
        started = time.monotonic()
        try:
            while True:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise CancelledError()
                try:
                    value = pending.get(timeout=parameter("execution.stream_progress_interval_seconds")
                                        if not future.done() else 0)
                except queue.Empty:
                    if future.done():
                        return future.result()
                    if current and current["status"] in {"started", "running", "waiting", "retrying"}:
                        public = {**current, "status": "waiting", "elapsed_seconds": max(0, int(time.monotonic() - started))}
                        self._write_chat_stage_event("retrieve", "running", "Retrieving evidence",
                            detail=_retrieval_progress_detail(public), progress=public)
                    continue
                public = _public_retrieval_progress(value)
                if not public:
                    continue
                if (not current or public["phase"] != current["phase"] or public["status"] == "started"):
                    started = time.monotonic()
                current = public
                self._write_chat_stage_event("retrieve", "running", "Retrieving evidence",
                    detail=_retrieval_progress_detail(public), progress=public)
        finally:
            stopped.set()
            executor.shutdown(wait=False, cancel_futures=True)

    def _prepare_hybrid_final_with_progress(
        self, *, intent, result, display_observations, source_trait,
        query_phenotype, region, biological_context, question,
        source_observations, safety_review, publication_enrichment, live,
    ):
        """Prepare report facts/references off-thread without worker SSE writes."""
        def check_cancelled():
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise CancelledError("Report preparation was cancelled.")

        def complete_publication_metadata(registry):
            check_cancelled()
            # The owner announces this entire preparation operation. This
            # renderer callback must never write to the response stream.
            try:
                publication_enrichment.update(asyncio.run(enrich_citation_registry(registry)))
            except (RuntimeError, OSError, ValueError, TypeError) as exc:
                publication_enrichment.update({"errors": [type(exc).__name__],
                    "unresolved_pmids": [entry.publication.pmid for entry in registry.entries
                                         if not entry.publication.metadata_complete]})
            check_cancelled()
            result["publication_enrichment"] = dict(publication_enrichment)
            return publication_enrichment

        def prepare():
            check_cancelled()
            report_result = result
            auto_screen = result.get("auto_screening")
            if (self.auto_context.get("analysis_mode") == "prior_report_screening"
                    and isinstance(auto_screen, Mapping)):
                report_result = {**result, "novelty_screen": auto_screen}
            answer = task_specific_retrieval_answer(
                # LD receipts bind the original uploaded observations. The
                # result's verified aliases supply canonical evidence labels;
                # replacing source IDs here would invalidate that contract.
                intent, result=report_result, observations=source_observations,
                trait=source_trait, query_phenotype=query_phenotype, region=region,
                biological_context=biological_context,
                publication_enricher=complete_publication_metadata if live else None,
            )
            check_cancelled()
            context = build_hybrid_final_context(
                question=question, intent=intent.value, source_trait=source_trait,
                source_observations=source_observations, result=report_result,
                deterministic_answer=answer, biological_context=biological_context or {},
                safety_review=safety_review,
            )
            check_cancelled()
            context = novelty_context_for_report(context)
            scope = auto_screen.get("auto_scope_review") if isinstance(auto_screen, Mapping) else None
            if (report_result is not result and isinstance(scope, Mapping)
                    and scope.get("warnings")):
                from ..staged_reasoning import FinalEvidenceFact

                statement = (
                    "Auto Analysis excludes unresolved SNP identities from its final evaluated scope. "
                    f"Final screening status: {auto_screen['status']}. "
                    + " ".join(str(warning) for warning in scope["warnings"])
                    + " The unchanged Gate review describes the original full input, including excluded SNPs; "
                    "its identity-coverage warning does not override this evaluated-scope decision. "
                    "Excluded SNPs are unassessed, not negative evidence."
                )
                fact = FinalEvidenceFact("auto_evaluated_scope", "novelty_ld", (), statement, (),
                    {"supported_statement": statement, "status": auto_screen["status"],
                     "scope_review": dict(scope)}, True)
                context = replace(context, facts=(*context.facts, fact),
                                  fallback_summary=context.fallback_summary + " " + statement)
            return context

        return self._run_stage_operation(prepare, stage="response",
            detail="Preparing the evidence report and references...")

    def _run_hybrid_final_with_progress(
        self,
        context: Any,
        completer: Any,
    ) -> tuple[Any, dict[str, int]]:
        """Keep one response-stage row alive without exposing hidden reasoning."""

        started = time.monotonic()
        if completer is None or context.safety_review.status is SafetyStatus.STOPPED:
            result = self._run_stage_operation(lambda: run_hybrid_final(context, completer),
                stage="response", detail="Formatting the response and references...")
        else:
            if self.cancel_event is None:
                self.cancel_event = threading.Event()
            if self.cancel_event.is_set():
                raise CancelledError("Response generation was cancelled.")
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="gwasmind-final-response",
            )
            pending_substages: deque[str] = deque(maxlen=16)
            substage_lock = threading.Lock()
            queued_substage = None

            def progress_sink(phase, details):
                nonlocal queued_substage
                if (phase != "response_substage" or not isinstance(details, Mapping)
                        or details.get("status") != "running"
                        or not isinstance(details.get("substage"), str)
                        or details.get("substage") not in {"inference", "fact_verification"}):
                    return
                substage = details["substage"]
                with substage_lock:
                    if substage != queued_substage:
                        pending_substages.append(substage)
                        queued_substage = substage

            def drain_substages():
                while True:
                    with substage_lock:
                        if not pending_substages:
                            return
                        substage = pending_substages.popleft()
                    self._write_chat_stage_event(
                        "response", "running",
                        "Generating response" if substage == "inference" else "Validating response",
                        substage=substage,
                    )

            future = executor.submit(copy_context().run, run_hybrid_final, context, completer,
                                     progress_sink=progress_sink)
            wait_seconds = 1.0
            try:
                while True:
                    try:
                        if self.cancel_event is not None and self.cancel_event.is_set():
                            raise CancelledError("Response generation was cancelled.")
                        drain_substages()
                        result = future.result(timeout=wait_seconds)
                        if self.cancel_event.is_set():
                            raise CancelledError("Response generation was cancelled.")
                        drain_substages()
                        break
                    except FutureTimeoutError:
                        if future.done():
                            result = future.result()
                            if self.cancel_event.is_set():
                                raise CancelledError("Response generation was cancelled.")
                            drain_substages()
                            break
                        drain_substages()
                        elapsed = max(1, round(time.monotonic() - started))
                        with self._response_stream_lock:
                            streaming_metrics = dict(self._response_stream_metrics)
                        self._write_chat_stage_event(
                            "response",
                            "running",
                            "Generating response",
                            metrics={"elapsed_seconds": elapsed, **streaming_metrics},
                        )
            except (CancelledError, *_CLIENT_DISCONNECT_ERRORS):
                self.cancel_event.set()
                future.cancel()
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        elapsed = max(0, round(time.monotonic() - started))
        usage = getattr(result, "usage", {})
        metrics: dict[str, int] = {
            "elapsed_seconds": elapsed,
        }
        with self._response_stream_lock:
            if "received_chars" in self._response_stream_metrics:
                metrics["received_chars"] = self._response_stream_metrics["received_chars"]
        if isinstance(usage, Mapping):
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
                # Includes stage-4 ranking plus initial/repair report completions.
                key = "output_tokens" if self._response_completed_tokens_exact \
                    else "estimated_output_tokens"
                metrics[key] = max(
                    completion_tokens, self._response_completed_tokens
                )
            if (isinstance(total_tokens, int) and not isinstance(total_tokens, bool)
                    and self._response_completed_tokens <= (completion_tokens or 0)):
                metrics["total_tokens"] = total_tokens
        return result, metrics

    def _write_chat_event(self, event_type: str, content: str) -> None:
        if event_type == "error":
            self.outcome["error"] = content
            self.outcome["report_quality"] = "STOPPED"
        self.emit({"type": event_type, "content": content})

    def _write_chat_stage_event(
        self,
        stage: str,
        state: str,
        summary: str,
        *,
        detail: str | None = None,
        metrics: Mapping[str, int] | None = None,
        progress: Mapping[str, Any] | None = None,
        substage: str | None = None,
        ranking_unavailable: bool = False,
    ) -> None:
        """Write one bounded public update for the fixed four-stage UI."""

        if stage not in {"plan", "retrieve", "safety", "response"}:
            raise ValueError("unsupported public chat stage")
        if state not in {"pending", "running", "passed", "warning", "stopped", "failed"}:
            raise ValueError("unsupported public chat stage state")
        if substage is not None and (stage != "response" or not isinstance(substage, str)
                                     or substage not in {"inference", "fact_verification"}):
            raise ValueError("unsupported public response substage")
        now = time.monotonic()
        if state != "pending":
            self._stage_started.setdefault(stage, now)
            self._active_public_stage = stage
        duration = self._stage_elapsed.get(stage)
        if duration is None:
            duration = max(0, int(now - self._stage_started.get(stage, now)
                                  + self._stage_time_offsets.get(stage, 0)))
            if state in {"passed", "warning", "stopped", "failed"}:
                self._stage_elapsed[stage] = duration
        metrics = {**(metrics or {}), "stage_elapsed_seconds": duration}
        if stage in {"retrieve", "response"}:
            metrics = {**_public_analysis_scope_metrics(getattr(self, "outcome", {})), **metrics}
            if getattr(self, "outcome", {}).get("requested_task") == ChatIntent.VARIANT_DESCRIPTION.value:
                for key in ("candidate_pool_count", "shortlist_count", "ranked_count", "displayed_ranked_count", "candidate_genes"):
                    metrics.pop(key, None)
        payload: dict[str, Any] = {
            "type": "stage",
            "stage": stage,
            "state": state,
            "summary": str(summary)[:240],
        }
        if substage is not None:
            payload["substage"] = substage
        if stage == "response" and hasattr(self, "_model_usage"):
            with self._response_stream_lock:
                usage = self._model_usage.public()
            if usage:
                payload["model_usage"] = usage
        if stage == "response" and ranking_unavailable is True:
            payload["ranking_unavailable"] = True
        if detail:
            payload["detail"] = str(detail)[:500]
        public_progress = _public_retrieval_progress(progress)
        if public_progress:
            payload["progress"] = public_progress
        if metrics:
            allowed_metrics = {
                "variant_annotations",
                "credible_sets",
                "variant_consequences",
                "molecular_qtl",
                "colocalization",
                "enhancer_links",
                "l2g_predictions",
                "candidate_genes",
                "gene_expression",
                "gene_function",
                "pathways",
                "publications",
                "publication_abstracts",
                "total_records",
                "elapsed_seconds",
                "output_tokens",
                "total_tokens",
                "received_chars",
                "ranking_elapsed_seconds",
                "stage_elapsed_seconds",
                "estimated_output_tokens",
                "source_variant_count",
                "verified_proxy_count",
                "analysis_variant_count",
                "candidate_pool_count",
                "shortlist_count",
                "ranked_count",
                "displayed_ranked_count",
            }
            payload["metrics"] = {
                key: value
                for key, value in metrics.items()
                if key in allowed_metrics
                and isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= 10_000_000
            }
        with self._stage_event_lock:
            if payload == self._last_stage_payload:
                return
            self.emit(payload)
            self._last_stage_payload = payload


class _PayloadTooLarge(JobValidationError):
    status_code = 413
    code = "payload_too_large"


_LEGACY_XLSX_COLUMNS = (
    "Locus",
    "Locus ID",
    "Status",
    "Report Quality",
    "Prior Report Screen",
    "Screening Scope",
    "Source GWAS Phenotype",
    "Native Decision",
    "Lead SNP",
    "Chromosome",
    "Position",
    "P-value",
    "-log10(P)",
    "SNP Count",
    "Top Prioritized Candidate",
    "Candidate Genes",
    "Direct Associations",
    "Evidence Datasets",
    "Evidence Records",
    "Evidence Summary",
    "References",
    "Data Mode",
    "Data Release",
    "Limitations",
)


def _screening_xlsx_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Export bounded prior-association results without gene-ranking columns."""
    screen = row.get("screening")
    screen = screen if isinstance(screen, Mapping) else {}
    publications = screen.get("publications")
    publications = publications if isinstance(publications, (list, tuple)) else ()
    source = _legacy_xlsx_row(row)
    references = [str(source["References"])] if source["References"] else []
    for value in dict.fromkeys(str(value) for value in publications if isinstance(value, (str, int))):
        if not re.search(r"(?<!\d)" + re.escape(value) + r"(?!\d)", "; ".join(references)):
            references.append("PMID: " + value)
    result = {key: source[key] for key in (
        "Locus", "Locus ID", "Lead SNP", "Chromosome", "Position", "P-value", "-log10(P)", "SNP Count")}
    result.update({
        "Classification": row.get("status"),
        "Screening Status": row.get("screening_status"),
        "Screening Reason": row.get("screening_reason"),
        "Source GWAS Phenotype": row.get("source_phenotype"),
        "Query Phenotype": row.get("query_phenotype"),
        "LD Population": screen.get("population"),
        "LD r² Threshold": screen.get("r2_threshold"),
        "Search Window (bp per side)": screen.get("window_bp"),
        "Matching Studies": screen.get("matching_studies"),
        "Matching Variants": screen.get("matching_variants"),
        "Report Quality": row.get("report_quality"),
        "Evidence Summary": source["Evidence Summary"],
        "References": "; ".join(references),
        "Data Mode": source["Data Mode"],
        "Data Release": source["Data Release"],
        "Limitations": source["Limitations"],
    })
    return result


def _legacy_xlsx_row(row: Mapping[str, Any]) -> dict[str, Any]:
    def joined(value: Any) -> str:
        if isinstance(value, str):
            return value
        if not isinstance(value, (list, tuple)):
            return ""
        return "; ".join(item for item in value if isinstance(item, str))

    return {
        "Locus": row.get("locus_no"),
        "Locus ID": row.get("locus_id"),
        "Status": row.get("status"),
        "Report Quality": row.get("report_quality"),
        "Prior Report Screen": row.get("screening_label"),
        "Screening Scope": row.get("screening_reason"),
        "Source GWAS Phenotype": row.get("source_phenotype"),
        "Native Decision": row.get("decision_status"),
        "Lead SNP": row.get("lead_rsid"),
        "Chromosome": row.get("chr"),
        "Position": row.get("pos"),
        "P-value": row.get("p_value"),
        "-log10(P)": row.get("neg_log_p"),
        "SNP Count": row.get("snp_count"),
        "Top Prioritized Candidate": row.get("top_gene"),
        "Candidate Genes": row.get("candidate_genes"),
        "Direct Associations": joined(row.get("direct_associations")),
        "Evidence Datasets": joined(row.get("evidence_datasets")),
        "Evidence Records": row.get("evidence_count"),
        "Evidence Summary": (
            row.get("report_markdown") or row.get("llm_interpretation") or row.get("report")
        ),
        "References": row.get("references"),
        "Data Mode": row.get("data_mode"),
        "Data Release": row.get("data_release"),
        "Limitations": joined(row.get("limitations")),
    }


def _route_parts(path: str) -> list[str]:
    if not path.startswith(f"{API_PREFIX}/"):
        return []
    return [unquote(item) for item in path.removeprefix(f"{API_PREFIX}/").split("/")]


def _model_chat_messages(
    *,
    question: str,
    history: list[Any],
    analysis_context: Any,
    result: Mapping[str, Any] | None,
    deterministic_summary: str | None,
    intent: ChatIntent = ChatIntent.GENERAL,
    source_observations: list[Mapping[str, Any]] | None = None,
    source_trait: str | None = None,
) -> list[dict[str, str]]:
    """Build a bounded conversational prompt from evidence, not hidden reasoning."""

    system = load_prompt("chat_answer_system")
    if result is not None:
        system += "\n\n" + database_response_contract(intent)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    history_budget = parameter("web.general_history_total_chars")
    history_turns = parameter("web.general_history_turns")
    retained: list[dict[str, str]] = []
    for item in reversed(history[-history_turns:] if history_turns else []):
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        bounded = content[:parameter("web.general_history_item_chars")]
        if len(bounded) > history_budget:
            bounded = bounded[:history_budget]
        if not bounded:
            continue
        retained.append({"role": str(role), "content": bounded})
        history_budget -= len(bounded)
        if history_budget <= 0:
            break
    messages.extend(reversed(retained))

    payload: dict[str, Any] = {
        "question": question,
        "analysis_context": (
            analysis_context[:parameter("web.legacy_evidence_context_max_chars")] if isinstance(analysis_context, str) else None
        ),
    }
    if source_observations and intent is not ChatIntent.VARIANT_DESCRIPTION:
        payload["source_gwas"] = {
            "trait": str(source_trait or "loaded GWAS phenotype")[:500],
            "selected_variant_count": len(source_observations),
            "observations": [
                {key: item.get(key) for key in ("variant_id", "chromosome", "position", "p_value")}
                for item in source_observations[:parameter("execution.max_variants_per_query")]
            ],
            "interpretation": load_prompt("chat_source_gwas_interpretation"),
        }
    if result is None:
        payload["retrieval_note"] = load_prompt("chat_no_retrieval_note")
    else:
        bounded_result = _bounded_chat_result(result, intent=intent)
        if intent is ChatIntent.VARIANT_DESCRIPTION:
            # The deterministic renderer has already performed the reproducible
            # five-variant diversity selection and reference pruning.  Giving the
            # writer hundreds of raw credible-set rows would let it silently replace
            # those choices and recreate the earlier verbose failure mode.
            bounded_result["evidence"] = []
            bounded_result["evidence_projection"] = load_prompt("chat_evidence_projection_note")
        payload["gwasmind_result"] = bounded_result
        payload["deterministic_evidence_summary"] = (deterministic_summary or "")[:parameter("web.legacy_evidence_context_max_chars")]
    messages.append(
        {
            "role": "user",
            "content": format_prompt(
                "chat_answer_user",
                payload=json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        }
    )
    return messages


def _bounded_chat_result(
    result: Mapping[str, Any], *, intent: ChatIntent = ChatIntent.GENERAL
) -> dict[str, Any]:
    # Compatibility projection: the current contextual-followup caller supplies
    # result=None. Retrieved reports use build_hybrid_final_context instead.
    ranked = result.get("ranked_genes")
    limitations = result.get("limitations")
    evidence = _chat_evidence_rows(result)
    summary = str(result.get("summary", ""))[:8_000]
    for status in (
        "PRIORITIZED",
        "EVIDENCE_RETRIEVED",
        "NO_GWAS_SIGNAL",
        "INSUFFICIENT_EVIDENCE",
        "CONFLICTING_EVIDENCE",
        "INVALID_INPUT",
        "POLICY_BLOCKED",
        "SERVICE_UNAVAILABLE",
        "INCOMPLETE",
        "NO_EXTERNAL_MATCH",
    ):
        summary = re.sub(rf"\b{status}\b", "the recorded retrieval outcome", summary)
    value: dict[str, Any] = {
        "retrieval_scope": (
            "external Open Targets lookup"
            if intent is ChatIntent.VARIANT_DESCRIPTION
            else "GWASMind analysis task"
        ),
        "retrieved_evidence_record_count": len(evidence),
        "has_retrieved_evidence": bool(evidence),
        "summary": summary,
        "data_mode": str(result.get("data_mode", "live"))[:80],
        "data_release": str(result.get("data_release", ""))[:120],
        "limitations": limitations[:12] if isinstance(limitations, list) else [],
        "evidence": [],
    }
    if intent is ChatIntent.GENE_PRIORITIZATION:
        value["ranked_genes"] = ranked[:10] if isinstance(ranked, list) else []
    evidence_budget = 32_000
    for item in evidence[:120]:
        compact = {
            key: item.get(key)
            for key in (
                "evidence_id",
                "kind",
                "source",
                "dataset",
                "data_release",
                "pmid",
                "variant_id",
                "gene_id",
                "study_id",
                "tissue",
                "subject",
                "object",
                "payload",
                "limitations",
            )
            if item.get(key) is not None
        }
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > 4_000:
            compact.pop("payload", None)
            compact["payload_note"] = load_prompt("chat_large_payload_note")
            encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > evidence_budget:
            break
        value["evidence"].append(compact)
        evidence_budget -= len(encoded)
    return value


def _describe_selected_variants(
    observations: list[Mapping[str, Any]],
    *,
    trait: str,
    region: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None = None,
) -> str:
    """Render source and external trait evidence without invoking gene ranking."""

    return selected_variant_trait_answer(
        observations,
        trait=trait,
        region=region,
        result=result,
    )


def _missing_selection_answer() -> str:
    """Return the fixed template for a locus query without a selected locus."""

    return "\n".join(
        (
            "**Summary**",
            "",
            "This request requires locus-specific evidence, but no variants are selected.",
            "",
            "**Details**",
            "",
            "### Required input",
            "",
            "- Select one or more variants on the Manhattan plot, then submit the question again.",
            "- GWASMind will use the selected variants and their uploaded-GWAS "
            "measurements as the analysis scope.",
            "",
            "**Limitations & Uncertainty**",
            "",
            "No external database query or locus interpretation was performed for this response.",
            "",
            "**References**",
            "",
            "No external references were retrieved for this response.",
        )
    )


def _result_passes_safety_gate(result: Mapping[str, Any]) -> bool:
    """Fail closed if the persisted gate outcome is absent or inconsistent."""

    status = str(result.get("status", "")).upper()
    if status not in {"PRIORITIZED", "EVIDENCE_RETRIEVED"}:
        return False
    sufficiency = result.get("sufficiency")
    if not isinstance(sufficiency, Mapping):
        return False
    reasons = sufficiency.get("reasons")
    if not isinstance(reasons, (list, tuple)) or reasons:
        return False
    if sufficiency.get("valid_input") is not True:
        return False
    for field in ("policy_blocks", "conflicts"):
        value = sufficiency.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            return False
    return True


def _safe_scripted_abstention(result: Mapping[str, Any]) -> str:
    """Return one fixed explanatory paragraph without invoking a prose model."""

    status = str(result.get("status", "INSUFFICIENT_EVIDENCE")).upper()
    sufficiency = result.get("sufficiency")
    values = sufficiency if isinstance(sufficiency, Mapping) else {}

    reasons = values.get("reasons")
    reason = ""
    if isinstance(reasons, (list, tuple)):
        reason = next(
            (
                re.sub(r"\s+", " ", item).strip()[:500]
                for item in reasons
                if isinstance(item, str) and item.strip()
            ),
            "",
        )
    detail = f" The recorded reason was: {reason}." if reason else ""
    return (
        "GWASMind stopped before substantive response generation because "
        f"{_safe_gate_status_message(status)}.{detail} No candidate-gene ranking or "
        "unsupported biological claim was generated; this result does not show that "
        "the locus lacks relevant genes, traits, or mechanisms."
    )


def _chat_answer(
    question: str,
    result: Mapping[str, Any],
    intent: ChatIntent | None = None,
) -> str:
    routed = intent or route_chat_intent(question)
    status = str(result.get("status", "INSUFFICIENT_EVIDENCE")).upper()
    ranked_value = result.get("ranked_genes", [])
    ranked = ranked_value if isinstance(ranked_value, list) else []
    genes = [
        _md_text(item.get("gene_id"), 80)
        for item in ranked[:10]
        if isinstance(item, Mapping) and item.get("gene_id")
    ]
    evidence = _chat_evidence_rows(result)
    relevant_kinds = {
        ChatIntent.GENE_ASSOCIATIONS: {
            "variant_annotation",
            "qtl",
            "colocalization",
            "regulatory",
            "expression",
            "function",
            "pathway",
            "literature",
        },
        ChatIntent.GENE_PRIORITIZATION: {
            "variant_annotation",
            "study",
            "gwas",
            "qtl",
            "colocalization",
            "regulatory",
            "l2g",
            "expression",
            "function",
            "pathway",
        },
        ChatIntent.TARGET_ASSESSMENT: {
            "function",
            "expression",
            "pathway",
            "safety",
            "tractability",
            "interaction",
            "phenotype",
        },
        ChatIntent.DISEASE_ASSOCIATIONS: {"target_disease"},
        ChatIntent.DRUG_REPURPOSING: {"drug", "safety", "tractability"},
    }[routed]
    selected = [item for item in evidence if str(item.get("kind", "")) in relevant_kinds]
    kind_counts = Counter(str(item.get("kind", "unknown")) for item in selected)
    datasets = _unique_chat_values(selected, "dataset", limit=12)
    sources = _unique_chat_values(selected or evidence, "source", limit=8)
    releases = _unique_chat_values(selected or evidence, "data_release", limit=8)
    if not releases:
        fallback_release = _md_text(result.get("data_release"), 80)
        releases = [fallback_release] if fallback_release else []

    limitations_value = result.get("limitations", [])
    limitations = (
        [_md_text(item, 500) for item in limitations_value[:8]]
        if isinstance(limitations_value, (list, tuple))
        else []
    )
    parts = [
        "**Summary**",
        "",
        f"The retrieved Open Targets evidence was evaluated for {routed.label}; "
        "the answer below distinguishes supported findings from missing database coverage.",
    ]

    if routed is ChatIntent.GENE_ASSOCIATIONS:
        linked_genes = sorted(
            {_md_text(item.get("gene_id"), 80) for item in selected if item.get("gene_id")}
        )
        parts.append(
            "**Variant-linked genes:** "
            + (", ".join(linked_genes) if linked_genes else "None returned")
        )
    elif routed is ChatIntent.GENE_PRIORITIZATION:
        if status == "PRIORITIZED" and genes:
            parts.append(f"**Prioritized genes:** {', '.join(genes)}")
        else:
            parts.append(
                "**Prioritized genes:** No reliable ranking was produced from the "
                "retrieved evidence."
            )
    elif routed is ChatIntent.TARGET_ASSESSMENT:
        targets = _chat_entities(evidence, "target")
        parts.append(
            "**Targets represented:** "
            + (", ".join(targets) if targets else ", ".join(genes) or "None reported")
        )
    elif routed is ChatIntent.DISEASE_ASSOCIATIONS:
        diseases = _chat_entities(selected, "disease")
        parts.append(f"**Direct target-disease records:** {len(selected)}")
        parts.append(
            "**Diseases represented:** "
            + (", ".join(diseases) if diseases else "None identifiable in returned records")
        )
    else:
        drugs = _chat_entities(selected, "drug")
        drug_records = sum(count for kind, count in kind_counts.items() if kind == "drug")
        parts.append(f"**Drug records:** {drug_records}")
        parts.append(
            "**Drugs represented:** "
            + (", ".join(drugs) if drugs else "None identifiable in returned records")
        )

    parts.extend(["", "**Evidence**", "", "### Retrieved evidence", ""])
    coverage = ", ".join(
        f"{_md_text(kind, 80)} ({count})" for kind, count in sorted(kind_counts.items())
    )
    qtl_type_counts: Counter[str] = Counter()
    for item in selected:
        if str(item.get("kind", "")) != "qtl":
            continue
        payload = item.get("payload")
        if not isinstance(payload, Mapping):
            continue
        qtl_type = qtl_study_type_from_semantic_payload(payload)
        if qtl_type is not None:
            qtl_type_counts[qtl_type] += 1
    parts.extend(
        [
            f"- **Evidence types:** {coverage or 'None returned'}",
            f"- **Datasets:** {', '.join(datasets) if datasets else 'None returned'}",
            f"- **Sources:** {', '.join(sources) if sources else 'None reported'}",
            f"- **Data mode/release:** {_md_text(result.get('data_mode', 'live'), 80)} / "
            f"{', '.join(releases) if releases else 'not reported'}",
        ]
    )
    if qtl_type_counts:
        parts.append(
            "- **QTL study types:** "
            + ", ".join(
                f"{QTL_STUDY_TYPE_LABELS[qtl_type]} ({count})"
                for qtl_type, count in qtl_type_counts.items()
            )
        )

    if not selected:
        parts.extend(
            [
                "",
                "No evidence records specific to this question were returned, so "
                "GWASMind cannot support a substantive answer from this run.",
            ]
        )

    route_limitations = {
        ChatIntent.GENE_ASSOCIATIONS: (
            "Variant-to-gene links are evidence-type specific and do not establish a causal gene.",
        ),
        ChatIntent.GENE_PRIORITIZATION: (
            "A ranking is hypothesis generation and does not establish a causal gene.",
        ),
        ChatIntent.TARGET_ASSESSMENT: (
            "Retrieved target profiles do not by themselves validate a therapeutic target.",
        ),
        ChatIntent.DISEASE_ASSOCIATIONS: (
            "No disease identifier was inferred from free text; the route requests "
            "direct associations for the locus-linked targets.",
            "Association scores are evidence aggregates, not probabilities of causality.",
        ),
        ChatIntent.DRUG_REPURPOSING: (
            "Drug or clinical-candidate evidence does not establish efficacy, safety, "
            "or a new indication.",
        ),
    }[routed]
    limitations = [*limitations, *route_limitations]
    if limitations:
        parts.extend(
            ["", "### Coverage and limitations", "", *[f"- {item}" for item in limitations]]
        )
    parts.extend(
        [
            "",
            "A missing database match is reported only as no retrieved evidence; it is "
            "not proof of novelty.",
        ]
    )
    pmids = _chat_pmids(selected)
    parts.extend(["", "**References**", ""])
    if pmids:
        parts.extend(
            f"[{index}] [PubMed PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
            for index, pmid in enumerate(pmids, 1)
        )
    else:
        parts.append(
            "No publication metadata was returned for the question-specific evidence; "
            "no reference was invented."
        )
    return "\n".join(parts)


def _chat_evidence_rows(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    # Legacy formatter bound, not the retrieval ledger or current Gate 2 input.
    value = result.get("evidence")
    if not isinstance(value, list):
        value = result.get("evidence_ledger")
    if not isinstance(value, list):
        return []
    return [item for item in value[:10_000] if isinstance(item, Mapping)]


def _unique_chat_values(rows: list[Mapping[str, Any]], key: str, *, limit: int) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = _md_text(row.get(key), 160)
        normalized = value.casefold()
        if value and normalized not in seen:
            seen.add(normalized)
            values.append(value)
            if len(values) >= limit:
                break
    return values


def _chat_entities(rows: list[Mapping[str, Any]], entity_type: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        for key in ("subject", "object"):
            entity = payload.get(key)
            if not isinstance(entity, Mapping):
                continue
            if str(entity.get("type", "")).casefold() != entity_type:
                continue
            value = _md_text(entity.get("label") or entity.get("id"), 120)
            normalized = value.casefold()
            if value and normalized not in seen:
                seen.add(normalized)
                values.append(value)
                if len(values) >= 8:
                    return values
    return values


def _chat_pmids(rows: list[Mapping[str, Any]], *, limit: int = 20) -> list[str]:
    """Return bounded, question-specific PubMed identifiers without guessing."""

    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        payload = row.get("payload")
        candidates: list[Any] = [row.get("pmid")]
        if isinstance(payload, Mapping):
            candidates.append(payload.get("pmid"))
            context = payload.get("context")
            if isinstance(context, Mapping):
                candidates.append(context.get("pmid"))
            upstream = payload.get("upstream")
            if isinstance(upstream, Mapping):
                raw = upstream.get("raw")
                if isinstance(raw, Mapping):
                    candidates.extend((raw.get("pmid"), raw.get("pubmedId")))
        for candidate in candidates:
            if candidate is None:
                continue
            match = re.fullmatch(
                r"(?:PMID\s*:\s*)?([1-9]\d{0,11})",
                str(candidate).strip(),
                flags=re.IGNORECASE,
            )
            if match is None:
                continue
            pmid = match.group(1)
            if pmid in seen:
                continue
            seen.add(pmid)
            values.append(pmid)
            if len(values) >= limit:
                return values
    return values


def _md_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())[:limit]
    return re.sub(r"([\\`*_[\]{}()<>#+.!|-])", r"\\\1", text)


def _remember_model_configuration(app_server: GWASMindHTTPServer) -> None:
    """Keep a just-verified Settings configuration for the next start (best effort)."""
    path = app_server.model_configuration_path
    if path is None:
        return
    configuration = app_server.manager.verified_runtime_model_configuration()
    if configuration is None:
        return
    from ..credentials import CredentialError
    from ..model_credentials import save_model_configuration
    try:
        save_model_configuration(configuration, path)
    except (CredentialError, OSError):
        # The session stays usable; only the next start will ask again.
        pass


def _restore_model_configuration(app_server: GWASMindHTTPServer, path: Path) -> None:
    """Load a saved configuration as configured but not yet verified."""
    from ..credentials import CredentialError
    from ..model_credentials import load_model_configuration
    if not path.is_file():
        return
    try:
        saved = load_model_configuration(path)
        app_server.manager.configure_runtime_model(
            api_key=saved["api_key"],
            base_url=saved["base_url"],
            model_id=saved["model_id"],
            provider_id=saved["model_provider"],
            thinking_mode=saved["thinking_mode"],
            reasoning_effort=saved["reasoning_effort"],
        )
    except (CredentialError, JobValidationError, ValueError):
        # Unreadable, or no longer valid for this catalog: Settings asks again.
        return


def create_server(
    *,
    host: str = parameter("web.host"),
    port: int = parameter("web.launcher_port"),
    artifact_root: str | Path | None = None,
    static_root: str | Path | None = None,
    default_frozen_data_root: str | Path | None = None,
    model_api_key_env: str | None = None,
    default_model_base_url: str | None = None,
    default_model_id: str | None = None,
    default_model_thinking_mode: str | None = None,
    default_model_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    manager: JobManager | None = None,
    heartbeat_interval: float = parameter("web.heartbeat_interval_seconds"),
    ui_state: UiState | None = None,
    analysis_bridge: AnalysisBridge | None = None,
    history_store: LocalHistoryStore | None = None,
    default_analysis_mode: str = "live",
    model_catalog: PublicModelCatalog | None = None,
    refresh_public_models: bool = False,
) -> GWASMindHTTPServer:
    _validate_bind_host(host)
    owns_manager = manager is None
    if manager is None:
        project_root = Path(__file__).resolve().parents[3]
        resolved_artifacts = artifact_root or project_root / "artifacts" / "web_runs"
        manager = JobManager(
            resolved_artifacts,
            max_concurrent_runs=parameter("execution.max_concurrent_runs"),
            default_frozen_data_root=default_frozen_data_root,
            model_api_key_env=model_api_key_env,
            default_model_base_url=default_model_base_url,
            default_model_id=default_model_id,
            default_model_thinking_mode=default_model_thinking_mode,
            default_model_reasoning_effort=default_model_reasoning_effort,
        )
    try:
        return GWASMindHTTPServer(
            (host, port),
            manager,
            static_root=static_root,
            owns_manager=owns_manager,
            heartbeat_interval=heartbeat_interval,
            ui_state=ui_state,
            analysis_bridge=analysis_bridge,
            history_store=history_store,
            default_analysis_mode=default_analysis_mode,
            model_catalog=model_catalog,
            refresh_public_models=refresh_public_models,
        )
    except Exception:
        if owns_manager:
            manager.close()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gwasmind.web.server")
    parser.add_argument("--host", default=parameter("web.host"))
    parser.add_argument("--port", type=int, default=parameter("web.launcher_port"))
    parser.add_argument("--artifact-root")
    parser.add_argument("--frozen-data-root")
    parser.add_argument(
        "--default-mode",
        choices=("live", "frozen-safe"),
        default="live",
        help="evidence mode used by the preserved graphical Chat/Auto Analysis UI",
    )
    parser.add_argument("--model-api-key-env")
    parser.add_argument("--model-base-url")
    parser.add_argument("--model-id")
    parser.add_argument("--model-thinking-mode", choices=sorted(THINKING_MODES))
    parser.add_argument(
        "--model-reasoning-effort",
        choices=sorted(REASONING_EFFORTS),
    )
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--no-model-catalog-refresh", action="store_true",
        help="use cached or bundled public LLM lists without checking for updates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_bind_host(args.host)
    default_frozen = args.frozen_data_root
    model_api_key_env = args.model_api_key_env or os.environ.get("GWASMIND_MODEL_API_KEY_ENV")
    default_model_base_url = args.model_base_url or os.environ.get("GWASMIND_MODEL_BASE_URL")
    default_model_id = args.model_id or os.environ.get("GWASMIND_MODEL_ID")
    default_model_thinking_mode = args.model_thinking_mode or os.environ.get(
        "GWASMIND_MODEL_THINKING_MODE"
    )
    default_model_reasoning_effort = (
        args.model_reasoning_effort
        or os.environ.get("GWASMIND_MODEL_REASONING_EFFORT")
        or DEFAULT_REASONING_EFFORT
    )
    configured_model_values = (
        model_api_key_env,
        default_model_base_url,
        default_model_id,
    )
    if any(configured_model_values) and not all(configured_model_values):
        parser.error(
            "environment-backed model configuration requires --model-api-key-env, "
            "--model-base-url, and --model-id together"
        )

    server = create_server(
        host=args.host,
        port=args.port,
        artifact_root=args.artifact_root,
        default_frozen_data_root=default_frozen,
        default_analysis_mode=args.default_mode,
        model_api_key_env=model_api_key_env,
        default_model_base_url=default_model_base_url,
        default_model_id=default_model_id,
        default_model_thinking_mode=default_model_thinking_mode,
        default_model_reasoning_effort=default_model_reasoning_effort,
        refresh_public_models=not args.no_model_catalog_refresh and args.default_mode == "live",
    )
    if os.name == "nt" and not model_api_key_env:
        from ..model_credentials import model_configuration_path
        server.model_configuration_path = model_configuration_path()
        _restore_model_configuration(server, server.model_configuration_path)
    address, port = server.server_address[:2]
    url = f"http://{address}:{port}/"
    print(
        json.dumps(
            {
                "event": "gwasmind_web_ready",
                "host": address,
                "port": port,
                "url": url,
                "mode": args.default_mode,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
