from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from ..audit import sha256_json
from ..configuration import configuration_provenance, parameter
from ..evidence_store import (
    METADATA_KEY,
    STORE_KEY,
    export_document,
    import_legacy_document,
    read_document,
)
from ..evidence_store import (
    _read_manifest as _read_evidence_manifest,
)
from ..evidence_store import (
    atomic_json as _stream_atomic_json,
)
from ..model_options import (
    DEFAULT_MODEL_BASE_URL,
    DEFAULT_MODEL_ID,
    DEFAULT_REASONING_EFFORT,
    THINKING_DISABLED,
    THINKING_ENABLED,
    default_thinking_mode,
    normalize_reasoning_effort,
    normalize_thinking_mode,
)
from ..models import DecisionStatus, EvidenceKind
from ..provider_catalog import (
    CUSTOM_PROVIDER_ID,
    ProviderCatalogError,
    get_provider,
    infer_provider_id,
    validate_provider_configuration,
    validate_provider_endpoint,
)
from ..provider_reasoning import reasoning_for_endpoint, reasoning_request_body
from ..serialization import locus_query_from_dict
from ..snapshot import SnapshotManifestError, load_snapshot_manifest
from .model_connection import (
    ModelChatResult,
    ModelProbeError,
    complete_openai_compatible,
    list_openai_compatible_models,
    probe_openai_compatible,
)
from .ranking_policy import live_model_only_result, retained_model_review
from .result_processing import source_only_oversized_result, verify_chunked_result_receipt
from .saved_results import (
    FORMAT_KEY,
    FORMAT_VERSION,
    make_result_reference,
    restore_result_snapshot,
)

MAX_QUERIES = parameter("execution.max_queries")
MAX_VARIANTS_PER_QUERY = parameter("execution.max_variants_per_query")
MAX_CANDIDATE_GENES_PER_QUERY = parameter("ranking.max_discovered_targets")
WORKER_THREADS = parameter("execution.duckdb_threads")
DUCKDB_MEMORY_LIMIT = parameter("execution.duckdb_memory_limit")
MAX_EVENTS_PER_JOB = parameter("execution.max_events_per_job")
MAX_JOB_HISTORY = parameter("execution.max_job_history")
MAX_PENDING_RUNS = parameter("execution.max_pending_runs")
MAX_RESULT_JSON_BYTES = parameter("execution.max_result_json_bytes")
TERMINAL_STATES = frozenset({"completed", "cancelled", "failed"})
ARTIFACT_NAMES = frozenset(
    {
        "result",
        "retrieval",
        "decision",
        "explanation",
        "manifest",
        "effective-config",
        "audit",
        "ranking-input",
        "ranking-evidence",
    }
)

_SAFE_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,55}_API_KEY$")
_RUNTIME_API_KEY_ENV = "GWASMIND_RUNTIME_API_KEY"


class JobError(RuntimeError):
    """Base class for web job errors with an HTTP-friendly status."""

    status_code = 400
    code = "job_error"


class JobValidationError(JobError):
    code = "invalid_request"


class JobNotFoundError(JobError):
    status_code = 404
    code = "run_not_found"


class JobConflictError(JobError):
    status_code = 409
    code = "run_conflict"


class ModelConnectionError(JobError):
    status_code = 502
    code = "model_connection_failed"


class _PublicJobFailure(RuntimeError):
    """A deliberately sanitized failure message safe for HTTP/SSE clients."""


ProcessFactory = Callable[..., Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    _stream_atomic_json(path, value, compact=False)


@dataclass(slots=True)
class _Job:
    run_id: str
    request: dict[str, Any]
    root: Path
    model_api_key: str | None = field(default=None, repr=False)
    model_api_key_env: str | None = None
    state: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    current_query: int | None = None
    completed_queries: int = 0
    query_fraction: float = 0.0
    cancel_requested: bool = False
    error: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    reviewed_retrieval: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    next_sequence: int = 1
    process: Any | None = None
    result_references: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def query_count(self) -> int:
        return len(self.request["queries"])


class JobManager:
    """A bounded-concurrency supervisor around the existing GWASMind CLI.

    HTTP handler threads enqueue independent runs. Each private worker launches
    at most one child at a time; queries within each run remain sequential.
    The default preserves single-run behavior; the web server uses the configured
    worker limit for independent native runs and model requests.
    """

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        python_executable: str | None = None,
        process_factory: ProcessFactory | None = None,
        project_root: str | Path | None = None,
        default_frozen_data_root: str | Path | None = None,
        default_frozen_release: str | None = None,
        model_api_key_env: str | None = None,
        default_model_base_url: str | None = None,
        default_model_id: str | None = None,
        default_model_thinking_mode: str | None = None,
        default_model_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        poll_interval: float = parameter("execution.worker_poll_seconds"),
        max_concurrent_runs: int = parameter("execution.default_concurrent_runs"),
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if (
            isinstance(max_concurrent_runs, bool)
            or not isinstance(max_concurrent_runs, int)
            or not 1 <= max_concurrent_runs <= parameter("execution.max_concurrent_runs")
        ):
            raise ValueError(f"max_concurrent_runs must be an integer from 1 to {parameter('execution.max_concurrent_runs')}")
        if model_api_key_env is not None and (
            not _SAFE_ENV_NAME.fullmatch(model_api_key_env)
            or not model_api_key_env.endswith("_API_KEY")
        ):
            raise ValueError("model_api_key_env must be an uppercase *_API_KEY environment name")
        if bool(default_model_base_url) != bool(default_model_id):
            raise ValueError(
                "default_model_base_url and default_model_id must be supplied together"
            )
        if model_api_key_env is not None and not default_model_base_url:
            raise ValueError(
                "model_api_key_env requires explicit default_model_base_url and "
                "default_model_id values"
            )
        if default_model_base_url is not None and not _safe_model_base_url(default_model_base_url):
            raise ValueError("default_model_base_url must be HTTPS or a loopback HTTP URL")
        if default_model_id is not None and not _safe_model_id(default_model_id):
            raise ValueError("default_model_id is invalid")
        if default_model_thinking_mode is not None:
            try:
                normalize_thinking_mode(default_model_thinking_mode)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        try:
            normalize_reasoning_effort(default_model_reasoning_effort)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        resolved_frozen_root = (
            Path(default_frozen_data_root).resolve()
            if default_frozen_data_root is not None
            else None
        )
        frozen_inspection = (
            load_snapshot_manifest(
                resolved_frozen_root,
                expected_release=default_frozen_release,
            )
            if resolved_frozen_root is not None
            else None
        )
        self.artifact_root = Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.python_executable = python_executable or sys.executable
        self.process_factory = process_factory or subprocess.Popen
        self._explicit_project_root = project_root is not None
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[3]
        )
        self.default_frozen_data_root = resolved_frozen_root
        self.default_frozen_release = (
            frozen_inspection.release if frozen_inspection is not None else None
        )
        self.default_frozen_snapshot_hash = (
            frozen_inspection.manifest_sha256 if frozen_inspection is not None else None
        )
        self.model_api_key_env = model_api_key_env
        self._runtime_api_key: str | None = None
        self._runtime_base_url = (default_model_base_url or DEFAULT_MODEL_BASE_URL).rstrip("/")
        self._runtime_model_id = (default_model_id or DEFAULT_MODEL_ID).strip()
        self._runtime_provider_id = infer_provider_id(
            self._runtime_base_url,
            self._runtime_model_id,
        )
        self._runtime_thinking_mode = (
            normalize_thinking_mode(default_model_thinking_mode)
            if default_model_thinking_mode is not None
            else default_thinking_mode(
                model_id=self._runtime_model_id,
                base_url=self._runtime_base_url,
            )
        )
        self._runtime_reasoning_effort = normalize_reasoning_effort(default_model_reasoning_effort)
        self._runtime_reasoning_profile: dict[str, Any] | None = None
        self._provider_reasoning_capabilities: dict[tuple[str, str], dict[str, Any]] = {}
        self._verified_model_fingerprint: str | None = None
        self._model_config_revision = 0
        self.poll_interval = poll_interval
        self.max_concurrent_runs = max_concurrent_runs

        self._jobs: dict[str, _Job] = {}
        self._active_run_id: str | None = None
        self._active_jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._model_request_lock = threading.BoundedSemaphore(max_concurrent_runs)
        self._active_model_requests = 0
        self._model_probe_lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._closing = False
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                name=("gwasmind-web-job-worker" if max_concurrent_runs == 1
                      else f"gwasmind-web-job-worker-{index + 1}"),
                daemon=True,
            ) for index in range(max_concurrent_runs)
        ]
        self._worker = self._workers[0]  # Compatibility for single-worker callers.
        try:
            for worker in self._workers:
                worker.start()
        except Exception:
            self._closing = True
            for worker in self._workers:
                if worker.ident is not None:
                    self._queue.put(None)
            self._join_workers(5.0)
            raise

    def frozen_snapshot_defaults(self) -> dict[str, str]:
        """Revalidate and return the configured frozen snapshot identity."""

        root = self.default_frozen_data_root
        if root is None:
            raise JobValidationError("data_root is required for frozen-safe mode")
        try:
            inspection = load_snapshot_manifest(
                root,
                expected_release=self.default_frozen_release,
            )
        except SnapshotManifestError as exc:
            raise JobValidationError(str(exc)) from exc
        return {
            "data_root": inspection.data_root,
            "release": inspection.release,
            "snapshot_hash": inspection.manifest_sha256,
        }

    def submit(self, value: Mapping[str, Any], *, ld_expansion: Mapping[str, Any] | None = None) -> dict[str, Any]:
        request, model_api_key, model_api_key_env = self._validate_request(value)
        if ld_expansion is not None:
            from ..analysis_scope import attach_ld_expansion, private_query_dict

            if len(request["queries"]) != 1:
                raise JobValidationError("LD ranking expansion belongs to exactly one source query")
            try:
                query = locus_query_from_dict(request["queries"][0])
                request["queries"][0] = private_query_dict(attach_ld_expansion(query, ld_expansion))
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
        run_id = uuid4().hex
        root = (self.artifact_root / run_id).resolve()
        if self.artifact_root not in root.parents:
            raise JobValidationError("generated run path escaped the artifact root")
        with self._changed:
            if self._closing:
                raise JobConflictError("the job manager is shutting down")
            pending = sum(job.state not in TERMINAL_STATES for job in self._jobs.values())
            if pending >= MAX_PENDING_RUNS:
                raise JobConflictError(
                    f"the local queue already contains {MAX_PENDING_RUNS} pending runs"
                )
            root.mkdir(parents=True, exist_ok=False)
            job = _Job(
                run_id=run_id,
                request=request,
                root=root,
                model_api_key=model_api_key,
                model_api_key_env=model_api_key_env,
            )
            try:
                self._emit_locked(job, "queued", "Analysis queued.")
                self._persist_locked(job)
            except Exception:
                # The run has not been published or queued yet. Roll back its
                # private directory so callers never observe a ghost job that
                # no worker can execute.
                shutil.rmtree(root, ignore_errors=True)
                raise
            self._evict_terminal_jobs_locked()
            self._jobs[run_id] = job
            snapshot = self._snapshot_locked(job)
            self._queue.put(run_id)
        return snapshot

    def snapshot(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._get_locked(run_id)
            return self._snapshot_locked(job)

    def resume_ranking(self, run_id: str, *, safety_status: str) -> dict[str, Any]:
        """Queue inference over this run's completed, reviewed retrieval only."""
        if safety_status not in {"PASS", "PASS_WITH_WARNINGS"}:
            raise JobValidationError("ranking requires a passed evidence review")
        with self._changed:
            job = self._get_locked(run_id)
            if job.request.get("mode") == "live" and job.request.get("ranker") != "openai":
                raise JobValidationError("Live gene ranking requires a verified model; rule-based ranking is disabled.")
            if (
                job.state != "completed"
                or not job.request.get("defer_ranking")
                or job.reviewed_retrieval is not None
                or job.query_count != 1
                or self._closing
                or any(result.get("result_processing_limit") for result in job.results)
            ):
                raise JobConflictError("this run is not ready for deferred ranking")
            pending = sum(item.state not in TERMINAL_STATES for item in self._jobs.values())
            if pending >= MAX_PENDING_RUNS:
                raise JobConflictError(
                    f"the local queue already contains {MAX_PENDING_RUNS} pending runs"
                )
            path = self.artifact_path(run_id, 0, "retrieval")
            request, key, key_env = self._validate_request(
                {**job.request, "defer_ranking": False}, allow_analysis_scope=True,
            )
            # Keep the pre-review result and all raw artifacts intact. Ranking
            # writes into a separate subdirectory of the same private run.
            _atomic_json(job.root / "retrieved_result.json", self._saved_result_snapshot(job))
            staged = replace(
                job, reviewed_retrieval=path, request=request,
                model_api_key=key, model_api_key_env=key_env, results=[], result_references={},
                completed_queries=0, query_fraction=0.0, finished_at=None,
                state="queued", events=list(job.events),
            )
            self._emit_locked(
                staged,
                "ranking_queued",
                "Candidate ranking queued after review.",
                safety_status=safety_status,
            )
            self._persist_locked(staged)
            self._jobs[run_id] = staged
            self._queue.put(run_id)
            return self._snapshot_locked(staged)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at)
            return [self._snapshot_locked(job) for job in jobs]

    def health(self) -> dict[str, Any]:
        with self._lock:
            queued = sum(job.state == "queued" for job in self._jobs.values())
            workers_alive = sum(worker.is_alive() for worker in self._workers)
            worker_alive = workers_alive > 0
            accepting_jobs = worker_alive and not self._closing
            return {
                "active_run_id": self._active_run_id,
                "active_run_ids": list(self._active_jobs),
                "active_runs": len(self._active_jobs),
                "max_concurrent_runs": self.max_concurrent_runs,
                "available_run_slots": (
                    max(0, workers_alive - len(self._active_jobs)) if accepting_jobs else 0
                ),
                "workers_alive": workers_alive,
                "active_model_requests": self._active_model_requests,
                "available_model_slots": (
                    max(0, self.max_concurrent_runs - self._active_model_requests)
                    if accepting_jobs else 0
                ),
                "queued_runs": queued,
                "worker_alive": worker_alive,
                "accepting_jobs": accepting_jobs,
                "artifact_root_ready": (
                    self.artifact_root.is_dir() and os.access(self.artifact_root, os.W_OK)
                ),
            }

    def model_status(self) -> dict[str, Any]:
        with self._lock:
            runtime_key = self._runtime_api_key or ""
            runtime_base_url = self._runtime_base_url
            runtime_model_id = self._runtime_model_id
            runtime_provider_id = self._runtime_provider_id
            runtime_thinking_mode = self._runtime_thinking_mode
            runtime_reasoning_effort = self._runtime_reasoning_effort
            reasoning_profile = copy.deepcopy(self._runtime_reasoning_profile)
            verified_fingerprint = self._verified_model_fingerprint
        configured = self.model_api_key_env
        environment_key = os.environ.get(configured, "").strip() if configured else ""
        effective_key = runtime_key or environment_key
        is_configured = bool(effective_key)
        provider_id = runtime_provider_id
        provider = get_provider(provider_id)
        verified = bool(
            effective_key
            and verified_fingerprint
            == _model_config_fingerprint(
                effective_key,
                runtime_base_url,
                runtime_model_id,
                runtime_thinking_mode,
                runtime_reasoning_effort,
                reasoning_profile,
            )
        )
        return {
            "default_ranker": "openai",
            "models": {
                "deterministic": {
                    "available": True,
                    "model_id": "evidence-count-ranker-v1",
                },
                "openai": {
                    "available": verified,
                    "configured": is_configured,
                    "verified": verified,
                    "credentials_source": (
                        "runtime-memory"
                        if runtime_key
                        else "environment"
                        if environment_key
                        else None
                    ),
                    "api_key_env": configured,
                    "base_url": runtime_base_url,
                    "model_id": runtime_model_id,
                    "provider_id": provider_id,
                    "provider_name": provider.name if provider is not None else "Custom",
                    "provider_category": (provider.category if provider is not None else "custom"),
                    "protocol": (
                        provider.protocol if provider is not None else "openai-chat-completions"
                    ),
                    "thinking_mode": runtime_thinking_mode,
                    "reasoning_effort": runtime_reasoning_effort,
                    "reasoning": reasoning_for_endpoint(runtime_base_url, runtime_model_id, profile=reasoning_profile),
                },
            },
        }

    def configure_runtime_model(
        self,
        *,
        api_key: Any,
        base_url: Any,
        model_id: Any,
        provider_id: Any = None,
        thinking_mode: Any = None,
        reasoning_effort: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            (
                key,
                replace_key,
                normalized_base,
                normalized_model,
                normalized_thinking,
                normalized_effort,
            ) = self._resolve_runtime_model_settings(
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )
            normalized_provider = _normalize_runtime_provider_id(
                provider_id, normalized_base, normalized_model,
            )
            provider_changed = normalized_provider != self._runtime_provider_id
            if replace_key:
                self._runtime_api_key = key
            self._runtime_base_url = normalized_base
            self._runtime_model_id = normalized_model
            self._runtime_provider_id = normalized_provider
            self._runtime_thinking_mode = normalized_thinking
            self._runtime_reasoning_effort = normalized_effort
            self._runtime_reasoning_profile = self._reasoning_profile_for(normalized_base, normalized_model)
            self._model_config_revision += 1
            if provider_changed:
                self._verified_model_fingerprint = None
        return self.model_status()

    def verified_runtime_model_configuration(self) -> dict[str, str] | None:
        """The verified Settings configuration, for local encrypted storage only.

        An environment credential is never returned; it stays with the launch
        environment. The result must never be sent to the browser or logged.
        """
        if not self.model_status()["models"]["openai"]["verified"]:
            return None
        with self._lock:
            if not self._runtime_api_key:
                return None
            return {
                "api_key": self._runtime_api_key,
                "base_url": self._runtime_base_url,
                "model_id": self._runtime_model_id,
                "thinking_mode": self._runtime_thinking_mode,
                "reasoning_effort": self._runtime_reasoning_effort,
                "model_provider": self._runtime_provider_id or CUSTOM_PROVIDER_ID,
            }

    def list_provider_models(
        self,
        *,
        api_key: Any,
        base_url: Any,
        provider_id: Any,
    ) -> dict[str, Any]:
        """Fetch current model IDs without changing or verifying session settings."""

        try:
            endpoint = validate_provider_endpoint(
                provider_id=provider_id,
                base_url=base_url,
            )
        except ProviderCatalogError as exc:
            raise JobValidationError(str(exc)) from exc

        if api_key is not None and not isinstance(api_key, str):
            raise JobValidationError("API key must be a string")
        supplied_key = api_key.strip() if isinstance(api_key, str) else ""
        if supplied_key in {"\u2022" * 8, "\u25cf" * 8}:
            raise JobValidationError("A masked key placeholder is not an API key")
        with self._lock:
            retained_key = self._runtime_api_key or ""
            retained_base_url = self._runtime_base_url
            retained_provider_id = self._runtime_provider_id
            key_environment = self.model_api_key_env

        retained_endpoint_matches = False
        try:
            retained_endpoint = validate_provider_endpoint(
                provider_id=retained_provider_id,
                base_url=retained_base_url,
            )
        except ProviderCatalogError:
            retained_endpoint = None
        if retained_endpoint is not None:
            retained_endpoint_matches = (
                endpoint.provider_id == retained_endpoint.provider_id
                and endpoint.base_url == retained_endpoint.base_url
            )

        key = supplied_key
        if not key and retained_endpoint_matches:
            key = retained_key
            if not key and key_environment:
                key = os.environ.get(key_environment, "").strip()
        if not key:
            raise JobValidationError(
                "API key is required to refresh models for this provider and API Base URL."
            )
        if not 8 <= len(key) <= 512:
            raise JobValidationError("API key must contain between 8 and 512 characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in key):
            raise JobValidationError("API key contains invalid control characters")

        try:
            with self._model_probe_lock:
                result = list_openai_compatible_models(
                    api_key=key,
                    base_url=endpoint.base_url,
                )
        except ModelProbeError as exc:
            message = str(exc)
            if key in message:
                message = "The provider model-list request failed."
            # Do not retain an upstream exception chain that a future logger
            # could serialize alongside request-scoped credentials or details.
            raise ModelConnectionError(message) from None
        capabilities = {}
        for model_id in result.models:
            cap = result.reasoning_capabilities.get(model_id)
            if endpoint.provider_id == "openrouter" and isinstance(cap, Mapping):
                profile = {"provider_id": "openrouter", "base_url": endpoint.base_url,
                           "model_id": model_id, "reasoning": dict(cap)}
                try:
                    cap = reasoning_for_endpoint(endpoint.base_url, model_id, profile=profile)
                except ValueError:
                    cap = reasoning_for_endpoint(endpoint.base_url, model_id)
                with self._lock:
                    self._provider_reasoning_capabilities[(endpoint.base_url, model_id)] = copy.deepcopy(cap)
            else:
                cap = reasoning_for_endpoint(endpoint.base_url, model_id)
            capabilities[model_id] = cap
        return {
            "provider_id": endpoint.provider_id,
            "api_base_url": endpoint.base_url,
            "models": list(result.models),
            "count": len(result.models),
            "truncated": result.truncated,
            "latency_ms": result.latency_ms,
            "reasoning_capabilities": capabilities,
        }

    def _reasoning_profile_for(self, base_url: str, model_id: str) -> dict[str, Any] | None:
        if infer_provider_id(base_url) != "openrouter":
            return None
        endpoint = validate_provider_endpoint(provider_id="openrouter", base_url=base_url)
        with self._lock:
            cap = copy.deepcopy(self._provider_reasoning_capabilities.get((endpoint.base_url, model_id)))
        if cap is None:
            return None
        return {"provider_id": "openrouter", "base_url": base_url.rstrip("/"),
                "model_id": model_id, "reasoning": cap}

    def model_reasoning(self, *, provider_id: Any, base_url: Any, model_id: Any) -> dict[str, Any]:
        """Resolve public capabilities locally without touching credentials."""
        try:
            validated = validate_provider_configuration(provider_id=provider_id, base_url=base_url, model_id=model_id)
            profile = self._reasoning_profile_for(validated.base_url, validated.model_id)
            cap = reasoning_for_endpoint(validated.base_url, validated.model_id, profile=profile)
        except ValueError as exc:
            raise JobValidationError(str(exc)) from exc
        return {"model_id": validated.model_id, "reasoning": cap}

    def test_runtime_model_connection(
        self,
        *,
        api_key: Any,
        base_url: Any,
        model_id: Any,
        provider_id: Any = None,
        thinking_mode: Any = None,
        reasoning_effort: Any = None,
    ) -> dict[str, Any]:
        """Probe a candidate configuration and activate it only after success."""

        with self._lock:
            revision = self._model_config_revision
            (
                key,
                replace_key,
                normalized_base,
                normalized_model,
                normalized_thinking,
                normalized_effort,
            ) = self._resolve_runtime_model_settings(
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )
            reasoning_profile = self._reasoning_profile_for(normalized_base, normalized_model)
        normalized_provider = _normalize_runtime_provider_id(
            provider_id,
            normalized_base,
            normalized_model,
        )
        try:
            with self._model_probe_lock:
                probe = probe_openai_compatible(
                    api_key=key,
                    base_url=normalized_base,
                    model_id=normalized_model,
                    thinking_mode=normalized_thinking,
                    reasoning_effort=normalized_effort,
                )
        except ModelProbeError as exc:
            raise ModelConnectionError(str(exc)) from exc
        with self._lock:
            if revision != self._model_config_revision or self._closing:
                raise JobConflictError("Model settings changed while the connection was tested.")
            if replace_key:
                self._runtime_api_key = key
            self._runtime_base_url = normalized_base
            self._runtime_model_id = normalized_model
            self._runtime_provider_id = normalized_provider
            self._runtime_thinking_mode = normalized_thinking
            self._runtime_reasoning_effort = normalized_effort
            self._runtime_reasoning_profile = reasoning_profile
            self._model_config_revision += 1
            self._verified_model_fingerprint = _model_config_fingerprint(
                key,
                normalized_base,
                normalized_model,
                normalized_thinking,
                normalized_effort,
                reasoning_profile,
            )
        return {
            "requested_model": probe.requested_model,
            "observed_model": probe.observed_model,
            "latency_ms": probe.latency_ms,
            "probe_thinking_mode": probe.thinking_mode,
            "probe_reasoning_effort": probe.reasoning_effort,
            "model_status": self.model_status(),
        }

    def complete_runtime_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = parameter("model.chat_default_max_tokens"),
        thinking_mode_override: str | None = None,
        reasoning_effort_override: str | None = None,
        progress_callback: Callable[[dict[str, int]], None] | None = None,
        json_output: bool = False,
        cancel_event: threading.Event | None = None,
        generation_attempts: Any = None,
    ) -> ModelChatResult:
        """Use the configured in-memory credential for a bounded UI chat turn."""

        with self._lock:
            if self._closing:
                raise JobConflictError("the job manager is shutting down")
            key = self._runtime_api_key or ""
            base_url = self._runtime_base_url
            model_id = self._runtime_model_id
            thinking_mode = self._runtime_thinking_mode
            reasoning_effort = self._runtime_reasoning_effort
            reasoning_profile = copy.deepcopy(self._runtime_reasoning_profile)
            verified_fingerprint = self._verified_model_fingerprint
        if not key and self.model_api_key_env:
            key = os.environ.get(self.model_api_key_env, "").strip()
        if not key:
            raise JobValidationError(
                "Configure and test an LLM connection in Settings before using AI Chat."
            )
        if verified_fingerprint != _model_config_fingerprint(
            key,
            base_url,
            model_id,
            thinking_mode,
            reasoning_effort,
            reasoning_profile,
        ):
            raise JobValidationError(
                "Test the saved LLM connection in Settings before using AI Chat."
            )
        effective_thinking = (
            normalize_thinking_mode(thinking_mode_override)
            if thinking_mode_override is not None
            else thinking_mode
        )
        effective_effort = (
            normalize_reasoning_effort(reasoning_effort_override)
            if reasoning_effort_override is not None
            else reasoning_effort
        )
        if thinking_mode_override == THINKING_DISABLED:
            capability = reasoning_for_endpoint(base_url, model_id, profile=reasoning_profile)
            if not capability["can_disable"] and capability["kind"] not in {"none", "unknown"}:
                levels = capability["efforts"]
                effective_thinking = THINKING_ENABLED if levels else "auto"
                effective_effort = levels[0] if levels else "default"
        request_messages = copy.deepcopy(messages)
        try:
            self._acquire_model_request_slot(cancel_event)
            try:
                return complete_openai_compatible(
                    api_key=key,
                    base_url=base_url,
                    model_id=model_id,
                    messages=request_messages,
                    max_tokens=max_tokens,
                    thinking_mode=effective_thinking,
                    reasoning_effort=effective_effort,
                    **({"reasoning_profile": reasoning_profile} if reasoning_profile is not None else {}),
                    **({"progress_callback": progress_callback} if progress_callback else {}),
                    **({"json_output": True} if json_output else {}),
                    **({"generation_attempts": generation_attempts} if generation_attempts is not None else {}),
                    **({"cancel_event": cancel_event} if cancel_event is not None else {}),
                )
            finally:
                with self._lock:
                    self._active_model_requests -= 1
                self._model_request_lock.release()
        except ModelProbeError as exc:
            raise ModelConnectionError(str(exc)) from exc

    def _acquire_model_request_slot(self, cancel_event: threading.Event | None) -> None:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("the model request was cancelled before it started")
            with self._lock:
                if self._closing:
                    raise JobConflictError("the job manager is shutting down")
            if not self._model_request_lock.acquire(timeout=0.1):
                continue
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError("the model request was cancelled before it started")
                with self._lock:
                    if self._closing:
                        raise JobConflictError("the job manager is shutting down")
                    self._active_model_requests += 1
            except Exception:
                self._model_request_lock.release()
                raise
            return

    def _resolve_runtime_model_settings(
        self,
        *,
        api_key: Any,
        base_url: Any,
        model_id: Any,
        thinking_mode: Any,
        reasoning_effort: Any,
    ) -> tuple[str, bool, str, str, str, str]:
        if api_key is not None and not isinstance(api_key, str):
            raise JobValidationError("API key must be a string")
        supplied_key = api_key.strip() if isinstance(api_key, str) else ""
        if supplied_key in {"\u2022" * 8, "\u25cf" * 8}:
            raise JobValidationError("A masked key placeholder is not an API key")
        replace_key = bool(supplied_key)
        with self._lock:
            retained_key = self._runtime_api_key or ""
            current_base = self._runtime_base_url
            current_model = self._runtime_model_id
            current_thinking = self._runtime_thinking_mode
            current_effort = self._runtime_reasoning_effort
        if replace_key:
            key = supplied_key
        else:
            key = retained_key
            if not key and self.model_api_key_env:
                key = os.environ.get(self.model_api_key_env, "").strip()
        if not 8 <= len(key) <= 512:
            raise JobValidationError("API key must contain between 8 and 512 characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in key):
            raise JobValidationError("API key contains invalid control characters")
        if not _safe_model_base_url(base_url):
            raise JobValidationError("base_url must be HTTPS or a loopback HTTP URL")
        if not _safe_model_id(model_id):
            raise JobValidationError("model_id must contain between 1 and 120 valid characters")
        normalized_base = str(base_url).rstrip("/")
        normalized_model = model_id.strip()
        if not replace_key and normalized_base != current_base.rstrip("/"):
            raise JobValidationError("API key is required when changing the API Base URL")
        if thinking_mode is None:
            normalized_thinking = (
                current_thinking
                if (normalized_base, normalized_model) == (current_base, current_model)
                else default_thinking_mode(
                    model_id=normalized_model,
                    base_url=normalized_base,
                )
            )
        else:
            try:
                normalized_thinking = normalize_thinking_mode(thinking_mode)
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
        if reasoning_effort is None:
            normalized_effort = (
                current_effort
                if (normalized_base, normalized_model) == (current_base, current_model)
                else "default"
            )
        else:
            try:
                normalized_effort = normalize_reasoning_effort(reasoning_effort)
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
        try:
            reasoning_request_body(base_url=normalized_base, model_id=normalized_model,
                thinking_mode=normalized_thinking, reasoning_effort=normalized_effort,
                profile=self._reasoning_profile_for(normalized_base, normalized_model))
        except ValueError as exc:
            raise JobValidationError(str(exc)) from exc
        return (
            key,
            replace_key,
            normalized_base,
            normalized_model,
            normalized_thinking,
            normalized_effort,
        )

    def wait_for_events(
        self, run_id: str, *, after_sequence: int = 0, timeout: float = 15.0
    ) -> tuple[list[dict[str, Any]], bool]:
        if after_sequence < 0:
            raise JobValidationError("event sequence cannot be negative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._changed:
            while True:
                job = self._get_locked(run_id)
                events = [dict(event) for event in job.events if int(event["seq"]) > after_sequence]
                if events or job.state in TERMINAL_STATES:
                    return events, job.state in TERMINAL_STATES
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return [self._heartbeat_locked(job)], False
                # Notifications are global to the manager. Loop until this
                # particular job changes or its own timeout expires so activity
                # in unrelated runs cannot manufacture heartbeat storms.
                self._changed.wait(remaining)

    def cancel(self, run_id: str) -> dict[str, Any]:
        process = None
        with self._changed:
            job = self._get_locked(run_id)
            if job.state == "cancelled":
                return self._snapshot_locked(job)
            if (
                job.state == "completed"
                and job.request.get("defer_ranking")
                and job.reviewed_retrieval is None
            ):
                job.cancel_requested = True
                self._finish_cancelled_locked(job)
                try:
                    self._persist_locked(job)
                except Exception:
                    self._changed.notify_all()
                return self._snapshot_locked(job)
            if job.state in {"completed", "failed"}:
                raise JobConflictError(f"run is already {job.state}")
            job.cancel_requested = True
            if job.state == "queued":
                self._finish_cancelled_locked(job)
                try:
                    self._persist_locked(job)
                except Exception:
                    # Cancellation is an in-memory safety action first. A full
                    # artifact disk must not keep a queued run executable.
                    self._changed.notify_all()
                return self._snapshot_locked(job)
            if job.state != "cancelling":
                job.state = "cancelling"
                self._emit_locked(job, "cancelling", "Cancellation requested.")
                try:
                    self._persist_locked(job)
                except Exception:
                    # Continue to terminate the exact process even when the
                    # status snapshot cannot be refreshed on disk.
                    self._changed.notify_all()
            process = job.process

        if process is not None:
            self._terminate_if_running(process)
        return self.snapshot(run_id)

    def result(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._get_locked(run_id)
            if job.state not in TERMINAL_STATES:
                raise JobConflictError("analysis has not reached a terminal state")
            # Capture the coherent terminal snapshot under the lock. Completed
            # native results retain only receipts; restore full independent data
            # outside the lock without caching it back onto the job.
            snapshot = self._saved_result_snapshot(job)
            snapshot["results"] = list(snapshot["results"])
            root = job.root
        return restore_result_snapshot(copy.deepcopy(snapshot), root=root)

    @staticmethod
    def _saved_result_snapshot(job: _Job) -> dict[str, Any]:
        snapshot = JobManager._result_snapshot(job, copy_results=False)
        snapshot["results"] = [job.result_references.get(index, result)
                               for index, result in enumerate(job.results)]
        if job.result_references:
            snapshot[FORMAT_KEY] = FORMAT_VERSION
        return snapshot

    @staticmethod
    def _result_snapshot(job: _Job, *, copy_results: bool = True) -> dict[str, Any]:
        return {
            "run_id": job.run_id,
            "state": job.state,
            "mode": job.request["mode"],
            "release": job.request["release"],
            "query_count": job.query_count,
            "completed_queries": job.completed_queries,
            "results": copy.deepcopy(job.results) if copy_results else job.results,
            "error": job.error,
        }

    def record_chat_writer_output(
        self, run_id: str, *, writer_call: int,
        messages: list[dict[str, str]], completion: ModelChatResult,
    ) -> dict[str, Any]:
        """Keep visible candidate output and its scientific input for local review.

        These are unvalidated candidates, never report content or public artifacts.
        Provider headers, credentials and hidden reasoning are not accepted here.
        """
        if type(writer_call) is not int or writer_call < 1:
            raise ValueError("writer_call must be a positive integer")
        content = getattr(completion, "content", None)
        if not isinstance(content, str):
            raise ValueError("writer content must be text")
        safe_messages = [{"role": row["role"], "content": row["content"]} for row in messages
                         if isinstance(row, Mapping) and isinstance(row.get("role"), str)
                         and isinstance(row.get("content"), str)]
        if len(safe_messages) != len(messages):
            raise ValueError("writer messages must contain text roles and content")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._lock:
            root = self._get_locked(run_id).root
        path = (root / "chat_writer_outputs" / f"{writer_call:03d}-{content_hash}.json").resolve()
        if root not in path.parents:
            raise JobValidationError("writer audit path escaped the run directory")
        usage = getattr(completion, "usage", {})
        usage = usage if isinstance(usage, Mapping) else {}
        payload = {"recorded_at": _utc_now(), "run_id": run_id, "writer_call": writer_call,
            "purpose": "report_draft" if writer_call == 1 else "report_repair",
            "validation_state": "candidate_only_see_chat_answer_outcome",
            "messages": safe_messages, "messages_sha256": sha256_json(safe_messages),
            "visible_content": content, "content_sha256": content_hash,
            "model": {key: getattr(completion, key, None) for key in
                      ("requested_model", "observed_model", "thinking_mode", "reasoning_effort", "latency_ms")},
            "usage": {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                      if type(usage.get(key)) is int and usage[key] >= 0}}
        _atomic_json(path, payload)
        return {"writer_call": writer_call, "purpose": payload["purpose"],
            "path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content_sha256": content_hash, "messages_sha256": payload["messages_sha256"]}

    def record_chat_fact_review(
        self, run_id: str, *, raw_json: str, corrected_json: str,
        raw_json_attempts: tuple[str, ...], machine_audit: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Save visible JSON before/after Gate 2 separately from the public report.

        Raw drafts can contain unverified assertions. They are audit material,
        and this method does not make them downloadable report artifacts.
        """
        if not isinstance(raw_json, str) or not isinstance(corrected_json, str):
            raise ValueError("Gate 2 JSON audit must contain text")
        if not all(isinstance(item, str) for item in raw_json_attempts):
            raise ValueError("Gate 2 raw attempts must contain text")
        digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        raw_hash, corrected_hash = digest(raw_json), digest(corrected_json)
        payload = {"schema_version": "report-hard-fact-audit-v1", "run_id": run_id,
            "recorded_at": _utc_now(), "purpose": "objective_fact_review",
            "raw_validation_state": "unverified_model_draft",
            "corrected_validation_state": "see_machine_audit_and_chat_answer_outcome",
            "raw_json": raw_json, "raw_json_sha256": raw_hash,
            "corrected_json": corrected_json, "corrected_json_sha256": corrected_hash,
            "raw_json_attempts": [{"content": item, "sha256": digest(item)} for item in raw_json_attempts],
            "machine_audit": copy.deepcopy(dict(machine_audit))}
        with self._lock:
            root = self._get_locked(run_id).root
        path = (root / "chat_fact_reviews" / f"{raw_hash}-{corrected_hash}.json").resolve()
        if root not in path.parents:
            raise JobValidationError("fact review audit path escaped the run directory")
        _atomic_json(path, payload)
        return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_json_sha256": raw_hash, "corrected_json_sha256": corrected_hash}

    def record_chat_answer_outcome(
        self,
        run_id: str,
        *,
        response_source: str,
        validation_issues: tuple[str, ...] = (),
        model_content: str | None = None,
        repair_content: str | None = None,
        draft_result: ModelChatResult | None = None,
        repair_result: ModelChatResult | None = None,
        output_review: Mapping[str, Any] | None = None,
        generation_attempts: Mapping[str, Any] | None = None,
        safety_review_execution: Mapping[str, Any] | None = None,
        publication_enrichment: Mapping[str, Any] | None = None,
        model_outputs: tuple[Mapping[str, Any], ...] = (),
        model_output_audit_errors: tuple[str, ...] = (),
        fact_review_output: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a content-free audit record for final chat answer generation."""

        allowed_sources = {
            "model",
            "model_repaired",
            "deterministic_repair",
            "partial_recovery",
            "deterministic_no_model",
            "deterministic_invalid_model_output",
            "deterministic_model_error",
            "deterministic_safety_stop",
            "response_validation_stopped",
            "evidence_only_recovery",
        }
        if response_source not in allowed_sources:
            raise ValueError("unsupported chat response source")
        with self._lock:
            job = self._get_locked(run_id)
            path = (job.root / "chat_answer_outcome.json").resolve()
            if job.root not in path.parents:
                raise JobValidationError("chat audit path escaped the run directory")
            model_status = self.model_status().get("models", {}).get("openai", {})

        def model_metadata(value: ModelChatResult | None) -> dict[str, Any] | None:
            if value is None:
                return None
            return {
                "requested_model": value.requested_model,
                "observed_model": value.observed_model,
                "latency_ms": value.latency_ms,
                "thinking_mode": value.thinking_mode,
                "reasoning_effort": value.reasoning_effort,
            }

        draft_metadata = model_metadata(draft_result)
        repair_metadata = model_metadata(repair_result)
        payload = {
            "recorded_at": _utc_now(),
            "response_source": response_source,
            "configuration_provenance": configuration_provenance(),
            "output_review": copy.deepcopy(dict(output_review)) if output_review is not None else None,
            "generation_attempts": copy.deepcopy(dict(generation_attempts)) if generation_attempts is not None else {},
            "safety_review_execution": copy.deepcopy(dict(safety_review_execution)) if safety_review_execution is not None else None,
            "publication_enrichment": copy.deepcopy(dict(publication_enrichment)) if publication_enrichment is not None else None,
            "model_outputs": [{**dict(pointer), "validation_issues": [issue for issue in validation_issues
                if issue.startswith("draft:" if pointer.get("writer_call") == 1 else "repair:")]}
                for pointer in model_outputs],
            "model_output_audit_errors": list(model_output_audit_errors),
            "fact_review_output": dict(fact_review_output) if fact_review_output is not None else None,
            "validation_issues": list(validation_issues),
            "model_id": (
                (draft_metadata or {}).get("observed_model") or model_status.get("model_id")
            ),
            "model_verified": bool(model_status.get("verified")),
            "draft_model": draft_metadata,
            "repair_model": repair_metadata,
            "draft_rejected_sha256": (
                hashlib.sha256(model_content.encode("utf-8")).hexdigest()
                if model_content is not None
                else None
            ),
            "repair_rejected_sha256": (
                hashlib.sha256(repair_content.encode("utf-8")).hexdigest()
                if repair_content is not None
                else None
            ),
            # Backwards-compatible alias for alpha run artifacts.
            "rejected_content_sha256": (
                hashlib.sha256(model_content.encode("utf-8")).hexdigest()
                if model_content is not None
                else None
            ),
        }
        _atomic_json(path, payload)

    def artifact_path(self, run_id: str, query_index: int, name: str) -> Path:
        if name not in ARTIFACT_NAMES:
            raise JobValidationError("artifact name is not allowed")
        if name in {"ranking-input", "ranking-evidence"}:
            return self._ranking_export(run_id, query_index, name)[0]
        with self._lock:
            job = self._get_locked(run_id)
            if not 0 <= query_index < job.query_count:
                raise JobValidationError("query index is outside this run")
            query_root = self._query_root(job, query_index)
            child_run_id = self._child_run_id(job, query_index)
            output = query_root / "output"
            mapping = {
                "result": output / "runs" / child_run_id / "result.json",
                "retrieval": output / "runs" / child_run_id / "retrieval.json",
                "decision": output / "runs" / child_run_id / "decision.json",
                "explanation": output / "runs" / child_run_id / "explanation.json",
                "manifest": output / "runs" / child_run_id / "run_manifest.json",
                "effective-config": output / "effective_config.json",
                "audit": output / "audit.jsonl",
            }
            target = mapping[name].resolve()
            if job.root != target and job.root not in target.parents:
                raise JobValidationError("artifact path escaped the run directory")
            if not target.is_file():
                raise JobNotFoundError("artifact is not available")
            return target

    def _ranking_export(self, run_id: str, query_index: int, name: str):
        """Resolve a committed ranking export without reconstructing model input."""
        from ..ranking_exports import RECEIPT_NAME, verified_export

        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise JobValidationError("ranking export run ID is invalid")
        if type(query_index) is not int or query_index < 0:
            raise JobValidationError("ranking export query index is invalid")
        root = (self.artifact_root / run_id).resolve()
        if self.artifact_root not in root.parents:
            raise JobValidationError("ranking export escaped the artifact root")
        # Saved history can still download a frozen input after process restart.
        with self._lock:
            job = self._jobs.get(run_id)
            count = job.query_count if job is not None else None
        if count is None:
            saved = root / "job.json"
            if not saved.is_file() or saved.is_symlink():
                raise JobNotFoundError("run was not found")
            try:
                if saved.stat().st_size > MAX_RESULT_JSON_BYTES:
                    raise ValueError("saved run metadata is too large")
                value = json.loads(saved.read_text(encoding="utf-8"))
                if not isinstance(value, dict) or value.get("run_id") != run_id:
                    raise ValueError("saved run identity does not match")
                count = value.get("query_count")
            except (OSError, ValueError, TypeError) as exc:
                raise JobValidationError("saved run metadata is invalid") from exc
        if type(count) is not int or not 0 <= query_index < count:
            raise JobValidationError("query index is outside this run")
        query_root = root / "queries" / f"{query_index:03d}"
        child = f"{run_id}-q{query_index + 1:03d}"
        for path in (query_root / "ranking", query_root):
            base = (path / "output" / "runs" / child).resolve()
            if root not in base.parents:
                raise JobValidationError("ranking export escaped the run")
            if not (base / RECEIPT_NAME).exists():
                continue
            try:
                target, descriptor = verified_export(base, name, max_receipt_bytes=MAX_RESULT_JSON_BYTES)
                return target, {"run_id": run_id, "query_index": query_index, **descriptor}
            except FileNotFoundError as exc:
                raise JobNotFoundError("ranking evidence is not available") from exc
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise JobValidationError("frozen ranking evidence could not be verified") from exc
        raise JobNotFoundError("ranking evidence has not been frozen for this query")

    def ranking_evidence_downloads(self, run_id: str, query_index: int = 0) -> dict:
        """Public identifiers only; clients construct local artifact URLs."""
        _, descriptor = self._ranking_export(run_id, query_index, "ranking-input")
        self._ranking_export(run_id, query_index, "ranking-evidence")
        return descriptor

    def ranking_review_receipt(self, descriptor: Mapping[str, Any]) -> dict | None:
        """Read a saved ranking audit only after its exact frozen input is verified."""
        from .output_review import public_ranking_review
        source, verified = self._ranking_export(descriptor.get("run_id"), descriptor.get("query_index"), "ranking-input")
        if any(descriptor.get(key) != verified.get(key) for key in ("snapshot_id", "model_input_sha256", "messages_sha256")):
            raise JobValidationError("ranking review input identity does not match")
        base = source.parent.parent
        path = (base / "decision.json").resolve()
        if path.parent != base or path.is_symlink() or not path.is_file():
            return None
        if path.stat().st_size > MAX_RESULT_JSON_BYTES:
            raise JobValidationError("ranking review is too large")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("model_input_sha256") != descriptor.get("model_input_sha256"):
            raise JobValidationError("ranking review belongs to a different input")
        return public_ranking_review(value.get("gate2_review"))

    def export_artifact_path(self, run_id: str, query_index: int, name: str) -> Path:
        """Downloads remain standalone JSON even when runtime evidence is chunked."""
        path = self.artifact_path(run_id, query_index, name)
        if name not in {"result", "retrieval"} or path.stat().st_size > MAX_RESULT_JSON_BYTES:
            return path
        return export_document(path, path.parent / "exports" / path.name,
            field="evidence" if name == "result" else "records")

    def readable_ranking_export_path(self, run_id: str, query_index: int, name: str) -> Path:
        """Derive a readable document from verified bytes, outside the frozen run."""
        from ..ranking_export_presentation import render_ranking_evidence_html, render_ranking_input_html
        from ..ranking_exports import _write_bytes_once

        if name not in {"ranking-input", "ranking-evidence"}:
            raise JobValidationError("HTML presentation is available only for ranking evidence and LLM input")
        source, descriptor = self._ranking_export(run_id, query_index, "ranking-input")
        if source.stat().st_size > MAX_RESULT_JSON_BYTES:
            raise JobValidationError("Frozen ranking input exceeds the readable export size limit")
        try:
            raw = source.read_bytes().decode("utf-8")
            render = render_ranking_input_html if name == "ranking-input" else render_ranking_evidence_html
            content = render(raw).encode("utf-8")
            digest = hashlib.sha256(content).hexdigest()
            label = "llm-input" if name == "ranking-input" else "gene-evidence-card"
            filename = f"gwasmind-{label}-{descriptor['snapshot_id'][:12]}.html"
            cache = self.artifact_root / "_readable_exports"
            folder = cache / run_id / f"{query_index:03d}" / digest
            if Path(filename).name != filename:
                raise ValueError("Readable ranking export name is not a plain file name")
            with self._lock:
                # Create, then resolve. On Windows, a non-strict resolve() of a folder
                # that a concurrent request creates mid-lookup keeps its \\?\ prefix
                # and would fail the containment check below.
                folder.mkdir(parents=True, exist_ok=True)
                cache, folder = cache.resolve(), folder.resolve()
                if self.artifact_root not in cache.parents or cache not in folder.parents:
                    raise ValueError("Readable ranking export escaped its cache")
                target = folder / filename
                _write_bytes_once(target, content)
            return target
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            raise JobValidationError("The frozen ranking input could not be formatted") from exc

    def close(self, *, timeout: float = 5.0) -> None:
        with self._changed:
            if self._closing:
                already_closing = True
                run_ids: list[str] = []
            else:
                already_closing = False
                self._closing = True
                run_ids = [
                    job.run_id for job in self._jobs.values() if job.state not in TERMINAL_STATES
                ]
                # Signal every run before waiting for any one child to exit.
                # Otherwise another active batch could launch its next query
                # while shutdown is still reaping the first process.
                for run_id in run_ids:
                    self._jobs[run_id].cancel_requested = True
                self._changed.notify_all()
        if already_closing:
            self._join_workers(timeout)
            with self._lock:
                self._runtime_api_key = None
                self._verified_model_fingerprint = None
                for job in self._jobs.values():
                    job.model_api_key = None
            return

        for run_id in run_ids:
            process = None
            try:
                self.cancel(run_id)
            except Exception:
                # Shutdown must remain best-effort even if job.json can no
                # longer be updated. Preserve the cancellation flag and reap
                # the exact child outside the manager lock.
                with self._changed:
                    job = self._jobs.get(run_id)
                    if job is not None:
                        job.cancel_requested = True
                        process = job.process
                if process is not None:
                    self._terminate_if_running(process)

        for _worker in self._workers:
            self._queue.put(None)
        self._join_workers(timeout)
        with self._lock:
            self._runtime_api_key = None
            self._verified_model_fingerprint = None
            for job in self._jobs.values():
                job.model_api_key = None

    def _join_workers(self, timeout: float) -> None:
        deadline = time.monotonic() + max(timeout, 0.0)
        for worker in self._workers:
            if worker.ident is not None and worker is not threading.current_thread():
                worker.join(timeout=max(0.0, deadline - time.monotonic()))

    def _validate_request(
        self, value: Mapping[str, Any], *, allow_analysis_scope: bool = False,
    ) -> tuple[dict[str, Any], str | None, str | None]:
        if not isinstance(value, Mapping):
            raise JobValidationError("request body must be a JSON object")
        allowed = {
            "mode",
            "release",
            "data_root",
            "snapshot_hash",
            "ranker",
            "defer_ranking",
            "model_id",
            "model_provider",
            "base_url",
            "thinking_mode",
            "reasoning_effort",
            "allow_model_alias",
            "offline",
            "queries",
        }
        if allow_analysis_scope:
            allowed.add("reasoning_profile")
        unknown = set(value).difference(allowed)
        if unknown:
            raise JobValidationError("unknown run fields: " + ", ".join(sorted(unknown)))
        for field_name in ("allow_model_alias", "offline", "defer_ranking"):
            if field_name in value and not isinstance(value[field_name], bool):
                raise JobValidationError(f"{field_name} must be a JSON boolean")
        if "thinking_mode" in value:
            try:
                normalize_thinking_mode(value["thinking_mode"])
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
        if "reasoning_effort" in value:
            try:
                normalize_reasoning_effort(value["reasoning_effort"])
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
        mode = value.get("mode")
        if not isinstance(mode, str) or mode not in {"live", "frozen-safe"}:
            raise JobValidationError("mode must be live or frozen-safe")
        queries = value.get("queries")
        if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES:
            raise JobValidationError("queries must contain between 1 and 50 loci")

        normalized_queries: list[dict[str, Any]] = []
        for index, query_value in enumerate(queries):
            if not isinstance(query_value, Mapping):
                raise JobValidationError(f"queries[{index}] must be an object")
            genes = query_value.get("candidate_genes", [])
            if not isinstance(genes, list):
                raise JobValidationError(
                    f"queries[{index}].candidate_genes must be an array when supplied"
                )
            if len(genes) > MAX_CANDIDATE_GENES_PER_QUERY:
                raise JobValidationError(
                    f"queries[{index}] exceeds {MAX_CANDIDATE_GENES_PER_QUERY} candidate genes"
                )
            try:
                parsed_query = locus_query_from_dict(dict(query_value), allow_analysis_scope=allow_analysis_scope)
                parsed_query.validate()
            except (KeyError, TypeError, ValueError) as exc:
                raise JobValidationError(f"queries[{index}] is invalid: {exc}") from exc
            if parsed_query.benchmark:
                raise JobValidationError("benchmark runs are not enabled in the web tester")
            normalized_queries.append(copy.deepcopy(dict(query_value)))

        ranker = value.get("ranker", "openai" if mode == "live" and not value.get("defer_ranking") else "deterministic")
        if not isinstance(ranker, str) or ranker not in {"deterministic", "openai"}:
            raise JobValidationError("ranker must be deterministic or openai")
        if mode == "live" and ranker != "openai" and value.get("defer_ranking") is not True:
            raise JobValidationError("Live gene ranking requires a verified model; rule-based ranking is disabled.")
        model_api_key: str | None = None
        model_api_key_env: str | None = None
        reasoning_profile = None
        if ranker == "openai":
            with self._lock:
                runtime_api_key = self._runtime_api_key
                runtime_model_id = self._runtime_model_id
                runtime_base_url = self._runtime_base_url
                runtime_provider_id = self._runtime_provider_id
                runtime_thinking_mode = self._runtime_thinking_mode
                runtime_reasoning_effort = self._runtime_reasoning_effort
                runtime_reasoning_profile = copy.deepcopy(self._runtime_reasoning_profile)
                verified_fingerprint = self._verified_model_fingerprint
            environment_api_key = (
                os.environ.get(self.model_api_key_env) if self.model_api_key_env else None
            )
            if runtime_api_key:
                model_api_key = runtime_api_key
                model_api_key_env = _RUNTIME_API_KEY_ENV
            elif environment_api_key:
                model_api_key = environment_api_key
                model_api_key_env = self.model_api_key_env
            else:
                raise JobValidationError(
                    "OpenAI-compatible ranking is unavailable; configure its environment key"
                )
            model_id = value.get("model_id") or runtime_model_id
            if not _safe_model_id(model_id):
                raise JobValidationError("model_id is required for openai ranking")
            verified_base_url = value.get("base_url") or runtime_base_url
            if not _safe_model_base_url(verified_base_url):
                raise JobValidationError("base_url must be HTTPS or a loopback HTTP URL")
            try:
                model_thinking_mode = normalize_thinking_mode(
                    value.get("thinking_mode", runtime_thinking_mode)
                )
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
            try:
                model_reasoning_effort = normalize_reasoning_effort(
                    value.get("reasoning_effort", runtime_reasoning_effort)
                )
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
            if verified_fingerprint != _model_config_fingerprint(
                model_api_key,
                str(verified_base_url).rstrip("/"),
                model_id.strip(),
                model_thinking_mode,
                model_reasoning_effort,
                runtime_reasoning_profile,
            ):
                raise JobValidationError(
                    "OpenAI-compatible ranking is unverified; use Save & Test in Settings"
                )
            reasoning_profile = copy.deepcopy(value.get("reasoning_profile", runtime_reasoning_profile))
            try:
                reasoning_request_body(base_url=str(verified_base_url), model_id=model_id.strip(),
                    thinking_mode=model_thinking_mode, reasoning_effort=model_reasoning_effort,
                    profile=reasoning_profile)
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
        else:
            runtime_model_id = "evidence-count-ranker-v1"
            runtime_base_url = None
            runtime_provider_id = "deterministic"
            model_thinking_mode = THINKING_DISABLED
            model_reasoning_effort = DEFAULT_REASONING_EFFORT

        release = value.get("release")
        if release is not None and (not isinstance(release, str) or not release.strip()):
            raise JobValidationError("release must be a non-empty string")
        data_root: str | None = None
        snapshot_hash: str
        if mode == "live":
            release = release or "live"
            if value.get("offline") is True:
                raise JobValidationError("live mode cannot be offline")
            snapshot_hash = "live:not-a-frozen-snapshot"
        else:
            requested_root = value.get("data_root")
            if requested_root is not None and (
                not isinstance(requested_root, str) or not requested_root.strip()
            ):
                raise JobValidationError("data_root must be a non-empty path string")
            configured_root = requested_root or self.default_frozen_data_root
            if not configured_root:
                raise JobValidationError("data_root is required for frozen-safe mode")
            root_path = Path(configured_root).resolve()
            requested_release = str(release).strip() if release is not None else None
            if requested_release == "live":
                raise JobValidationError("frozen-safe mode requires a frozen release")
            try:
                inspection = load_snapshot_manifest(
                    root_path,
                    expected_release=requested_release,
                )
            except SnapshotManifestError as exc:
                raise JobValidationError(str(exc)) from exc
            supplied_hash = value.get("snapshot_hash")
            if supplied_hash is not None and (
                not isinstance(supplied_hash, str)
                or supplied_hash.casefold() == "unverified"
                or supplied_hash != inspection.manifest_sha256
            ):
                raise JobValidationError(
                    "snapshot_hash must match the verified frozen snapshot manifest"
                )
            data_root = inspection.data_root
            release = inspection.release
            snapshot_hash = inspection.manifest_sha256

        base_url = value.get("base_url") or (runtime_base_url if ranker == "openai" else None)
        if base_url is not None and not _safe_model_base_url(base_url):
            raise JobValidationError("base_url must be HTTPS or a loopback HTTP URL")

        return (
            {
                "mode": mode,
                "release": str(release),
                "data_root": data_root,
                "snapshot_hash": snapshot_hash,
                "ranker": ranker,
                "defer_ranking": bool(value.get("defer_ranking", False)),
                "model_id": str(value.get("model_id") or runtime_model_id),
                "model_provider": str(
                    value.get("model_provider")
                    or (runtime_provider_id if ranker == "openai" else "deterministic")
                ),
                "base_url": base_url,
                "thinking_mode": model_thinking_mode,
                "reasoning_effort": model_reasoning_effort,
                **({"reasoning_profile": reasoning_profile} if reasoning_profile is not None else {}),
                "allow_model_alias": bool(value.get("allow_model_alias", False)),
                "offline": bool(
                    value.get(
                        "offline",
                        mode == "frozen-safe" and ranker == "deterministic",
                    )
                ),
                "queries": normalized_queries,
            },
            model_api_key,
            model_api_key_env,
        )

    def _worker_loop(self) -> None:
        while True:
            run_id = self._queue.get()
            if run_id is None:
                return
            with self._changed:
                job = self._jobs.get(run_id)
                if self._closing or job is None or job.state in TERMINAL_STATES:
                    continue
                self._active_jobs[run_id] = job
                self._active_run_id = next(iter(self._active_jobs), None)
            try:
                self._run_job(job)
            except Exception:
                with self._changed:
                    process = job.process
                if process is not None:
                    self._terminate_if_running(process)
                with self._changed:
                    if self._jobs.get(run_id) is not job:
                        continue
                    if job.state not in TERMINAL_STATES:
                        self._finish_failed_locked(
                            job,
                            "The analysis worker encountered an unexpected local error. "
                            "See the private local run artifacts for diagnostic details.",
                        )
                    try:
                        self._persist_locked(job)
                    except Exception:
                        # Preserve worker availability even when the artifact
                        # directory itself has become unavailable.
                        self._changed.notify_all()
            finally:
                with self._changed:
                    # A completed retrieval may have been resumed on another
                    # worker under the same public run ID. Do not clear that
                    # newer job's ownership when this worker finishes cleanup.
                    if self._active_jobs.get(run_id) is job:
                        del self._active_jobs[run_id]
                    self._active_run_id = next(iter(self._active_jobs), None)
                    self._changed.notify_all()

    def _run_job(self, job: _Job) -> None:
        with self._changed:
            if self._closing or job.cancel_requested:
                job.cancel_requested = True
                self._finish_cancelled_locked(job)
                self._persist_locked(job)
                return
            job.state = "running"
            if job.started_at is None:
                job.started_at = _utc_now()
            self._emit_locked(job, "started", "Analysis started.")
            self._persist_locked(job)

        for query_index, query_value in enumerate(job.request["queries"]):
            with self._changed:
                if job.cancel_requested:
                    self._finish_cancelled_locked(job)
                    self._persist_locked(job)
                    return
                job.current_query = query_index
                job.query_fraction = 0.0
                self._emit_locked(
                    job,
                    "query_started",
                    f"Locus {query_index + 1}/{job.query_count} started.",
                    query_index=query_index,
                )
                self._persist_locked(job)

            try:
                result = self._run_query(job, query_index, query_value)
            except Exception as exc:
                with self._changed:
                    if job.cancel_requested:
                        self._finish_cancelled_locked(job)
                    else:
                        message = (
                            str(exc)
                            if isinstance(exc, _PublicJobFailure)
                            else "The local analysis process failed. See the private "
                            "local run artifacts for diagnostic details."
                        )
                        self._finish_failed_locked(job, message)
                    self._persist_locked(job)
                return

            with self._changed:
                if job.cancel_requested:
                    self._finish_cancelled_locked(job)
                    self._persist_locked(job)
                    return
                job.results.append(result)
                job.completed_queries += 1
                job.query_fraction = 0.0
                decision_status = str(result.get("status", "unknown"))
                self._emit_locked(
                    job,
                    "query_completed",
                    f"Locus {query_index + 1}/{job.query_count} completed.",
                    query_index=query_index,
                    decision_status=decision_status,
                )
                self._persist_locked(job)

        with self._changed:
            completed = replace(
                job, state="completed", current_query=None,
                finished_at=_utc_now(), model_api_key=None,
            )
            # A completed retrieval can immediately be resumed by another
            # thread. Publish that state only after all completion writes have
            # succeeded, so failed finalization cannot overwrite a resumed job.
            _atomic_json(job.root / "web_result.json", self._saved_result_snapshot(completed))
            self._persist_locked(completed)
            # Publish small verified references only after both completion writes
            # succeed. Active consumers keep their independent full snapshots;
            # legacy/source-only rows without references retain their old form.
            job.results = self._saved_result_snapshot(completed)["results"]
            job.state = completed.state
            job.current_query = None
            job.finished_at = completed.finished_at
            job.model_api_key = None
            self._emit_locked(job, "completed", "All loci completed.")

    def _run_query(
        self, job: _Job, query_index: int, query_value: Mapping[str, Any]
    ) -> dict[str, Any]:
        query_root = self._query_root(job, query_index)
        query_root.mkdir(parents=True, exist_ok=True)
        input_path = query_root / "input.json"
        output_path = query_root / "output"
        progress_path = query_root / "progress.jsonl"
        stdout_path = query_root / "stdout.log"
        stderr_path = query_root / "stderr.log"
        child_run_id = self._child_run_id(job, query_index)
        _atomic_json(input_path, dict(query_value))
        output_path.mkdir(parents=True, exist_ok=True)

        command = self._command(
            job,
            input_path=input_path,
            output_path=output_path,
            progress_path=progress_path,
            child_run_id=child_run_id,
        )
        environment = self._child_environment(job)
        creation_flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        )
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            with self._changed:
                if self._closing or job.cancel_requested:
                    job.cancel_requested = True
                    raise JobConflictError("analysis was cancelled before its child started")
            process = self.process_factory(
                command,
                cwd=str(self.project_root),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                shell=False,
                creationflags=creation_flags,
                start_new_session=os.name != "nt",
            )
            with self._changed:
                job.process = process
                cancel_on_start = self._closing or job.cancel_requested
                if cancel_on_start:
                    job.cancel_requested = True
            if cancel_on_start:
                self._terminate_if_running(process)

            progress_offset = 0
            reaped = False
            try:
                while process.poll() is None:
                    progress_offset = self._drain_progress(
                        job, query_index, progress_path, progress_offset
                    )
                    time.sleep(self.poll_interval)
                return_code = int(process.wait())
                reaped = True
                self._drain_progress(job, query_index, progress_path, progress_offset, final=True)
            finally:
                # Monitoring, filesystem, and decode failures must never leave
                # the exact CLI child running in the background. Process waits
                # deliberately happen outside the manager lock.
                if not reaped:
                    self._terminate_if_running(process)
                with self._changed:
                    if job.process is process:
                        job.process = None

        if job.cancel_requested:
            raise JobConflictError("analysis was cancelled")
        source_only_failure = (
            return_code == 4
            and job.request.get("mode") == "live"
            and job.request.get("defer_ranking") is True
            and _verified_source_only_bootstrap_failure(
                output_path,
                query_value=query_value,
                expected_run_id=child_run_id,
                expected_release=str(job.request["release"]),
            )
        )
        if return_code == 4 and not source_only_failure:
            # The CLI uses exit 4 exclusively for SERVICE_UNAVAILABLE.  This is
            # an operational failure, not a completed scientific abstention:
            # keeping it terminal-failed prevents legacy clients from emitting
            # a misleading ``complete`` event while preserving private logs and
            # any CLI artifacts for local diagnosis.
            raise _PublicJobFailure(
                "A required service was unavailable; the analysis did not complete. "
                "Your selected input was retained. You can run this locus again "
                "when the service is available."
            )
        if return_code not in {0, 2, 3} and not source_only_failure:
            # stderr remains in the private local artifact directory. Do not
            # reflect provider responses, filesystem paths, or credentials into
            # public HTTP/SSE error payloads, and do not load an unbounded log.
            raise _PublicJobFailure(
                "A technical problem interrupted this analysis. Your selected input "
                "was retained, so you can run this locus again."
            )

        result_path = output_path / "runs" / child_run_id / "result.json"
        if not result_path.is_file():
            raise _PublicJobFailure(
                "The analysis could not prepare its result. Your selected input "
                "was retained, so you can run this locus again."
            )
        def evidence_progress(details):
            status = details.get("status")
            if status == "importing":
                message = "Organizing retained evidence into smaller blocks."
            elif status == "completed":
                message = "Complete evidence ledger verified and ready for review."
            else:
                message = "Loading and verifying the complete evidence ledger."
            count, total = details.get("records"), details.get("total_records")
            if isinstance(count, int):
                message += f" {count:,}" + (f"/{total:,}" if isinstance(total, int) else "") + " records."
            with self._changed:
                if job.cancel_requested:
                    raise JobConflictError("analysis was cancelled")
                self._emit_locked(job, "progress", message, phase="result_processing", details=details)
        try:
            if result_path.stat().st_size > MAX_RESULT_JSON_BYTES:
                if return_code == 0 and job.request.get("defer_ranking") is True:
                    with self._changed:
                        self._emit_locked(
                            job, "progress", "Verifying the retained result after reaching the processing limit.",
                            phase="result_processing", details={"status": "verifying"},
                        )
                try:
                    source_result = source_only_oversized_result(
                        result_path, query_value=query_value,
                        expected_run_id=child_run_id, expected_mode=str(job.request["mode"]),
                        expected_release=str(job.request["release"]), return_code=return_code,
                        defer_ranking=job.request.get("defer_ranking") is True,
                        limit_bytes=MAX_RESULT_JSON_BYTES,
                    )
                except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
                    raise _PublicJobFailure(
                        "GWASMind result artifact exceeds the local safety limit."
                    ) from exc
                with self._changed:
                    if job.cancel_requested:
                        raise JobConflictError("analysis was cancelled")
                    self._emit_locked(job, "progress",
                        "Organizing the retained evidence in smaller blocks for complete review.",
                        phase="result_processing", details={"status": "importing", "records": 0,
                                                            "total_records": None})
                try:
                    recovered = result_path.parent / "recovered_results" / uuid4().hex / "result.json"
                    import_legacy_document(result_path, recovered, field="evidence", progress=evidence_progress)
                    value = read_document(recovered, field="evidence", max_bytes=MAX_RESULT_JSON_BYTES, metadata_mode="summary",
                                          progress=evidence_progress)
                except (OSError, UnicodeError, ValueError, OverflowError, RecursionError):
                    # A failed import cannot turn retained bytes into reviewed
                    # evidence. Keep the verified historical source-only option.
                    with self._changed:
                        self._emit_locked(job, "progress",
                            "The complete result was preserved; preparing a source-only response.",
                            phase="result_processing", details=source_result["result_processing_limit"])
                    return source_result
            else:
                # read_document expands and removes metadata descriptors. Bind
                # both external formats to this worker's exact input before
                # loading their contents; legacy standalone results stay valid.
                document = _read_evidence_manifest(result_path, MAX_RESULT_JSON_BYTES)
                external_fields = STORE_KEY in document or METADATA_KEY in document
                del document
                if external_fields:
                    verify_chunked_result_receipt(result_path, query_value=query_value,
                        expected_run_id=child_run_id, expected_mode=str(job.request["mode"]),
                        expected_release=str(job.request["release"]),
                        source_only_bootstrap_failure=source_only_failure)
                value = read_document(result_path, field="evidence", materialize=False, metadata_mode="summary",
                    max_bytes=MAX_RESULT_JSON_BYTES, progress=evidence_progress)
                if STORE_KEY in value:
                    with self._changed:
                        self._emit_locked(job, "progress",
                            "Checking the complete evidence ledger before review.",
                            phase="result_processing", details={"status": "loading", "records": 0,
                                "total_records": value[STORE_KEY].get("records")})
                    value["evidence"] = list(value["evidence"])
                    value.pop(STORE_KEY)
        except (OSError, UnicodeError, ValueError) as exc:
            raise _PublicJobFailure(
                "GWASMind produced an unreadable result artifact. "
                "See the private local query logs for diagnostic details."
            ) from exc
        if not isinstance(value, dict):
            raise _PublicJobFailure("GWASMind result artifact is not a JSON object.")
        validated = _validate_result_artifact(
            value,
            expected_locus_id=str(query_value.get("locus_id", "")),
            expected_run_id=child_run_id,
            expected_mode=str(job.request["mode"]),
            expected_release=str(job.request["release"]),
            return_code=return_code,
            source_only_bootstrap_failure=source_only_failure,
        )
        validated = live_model_only_result(validated)
        if source_only_failure:
            # The shared report workflow can still describe validated source
            # observations. Completion here means the local retrieval attempt
            # ended; it does not claim external coverage or a scientific result.
            validated["external_evidence_available"] = False
            validated["data_release_verified"] = False
            validated["retrieval_failure"] = {
                "phase": "schema_discovery", "status": "unavailable",
                "source_only": True, "observed_release": None,
            }
        # Native manifests already bind the full evidence/metadata blocks. Keep
        # bookkeeping small; legacy oversized recovery and source-only overlays
        # retain their existing inline behavior instead of inventing a receipt.
        if not source_only_failure and result_path.stat().st_size <= MAX_RESULT_JSON_BYTES:
            reference = make_result_reference(root=job.root, result_path=result_path,
                input_path=input_path, value=validated, expected_run_id=child_run_id,
                expected_mode=str(job.request["mode"]), expected_release=str(job.request["release"]))
            with self._changed:
                job.result_references[query_index] = reference
        if not source_only_failure and not parameter("artifacts.retain_intermediate_files"):
            from ..artifact_retention import compact_completed_stage
            try:
                compact_completed_stage(
                    query_root, source_stage=query_root.parent if query_root.name == "ranking" else None,
                    compress=parameter("artifacts.compress_completed_runs"))
            except (OSError, ValueError):
                pass  # Compaction only frees disk; a completed result never depends on it.
        return validated

    def _command(
        self,
        job: _Job,
        *,
        input_path: Path,
        output_path: Path,
        progress_path: Path,
        child_run_id: str,
    ) -> list[str]:
        request = job.request
        command = [
            self.python_executable,
            str(Path(__file__).resolve().parents[1] / "worker_bootstrap.py"),
            str(self._worker_package_directory()),
            "--source-binding-receipt",
            str(progress_path.with_name("source_binding.json")),
            "analyze",
            "--input",
            str(input_path),
            "--mode",
            request["mode"],
            "--release",
            request["release"],
            "--ranker",
            request["ranker"],
            "--output-dir",
            str(output_path),
            "--run-id",
            child_run_id,
            "--event-jsonl",
            str(progress_path),
        ]
        # The complete result is read from its bounded, validated artifact.
        # CLI stdout only needs the summary; duplicating the full evidence
        # ledger here increases disk and serialization costs for LD batches.
        if job.reviewed_retrieval is not None:
            command.extend(["--reviewed-retrieval", str(job.reviewed_retrieval)])
        elif request.get("defer_ranking"):
            command.append("--defer-ranking")
        if request["mode"] == "frozen-safe":
            command.extend(
                [
                    "--data-root",
                    request["data_root"],
                    "--snapshot-hash",
                    request["snapshot_hash"],
                ]
            )
        if request["offline"]:
            command.append("--offline")
        if request["ranker"] == "openai":
            command.extend(
                [
                    "--model-provider",
                    request["model_provider"],
                    "--model-id",
                    request["model_id"],
                    "--thinking-mode",
                    request["thinking_mode"],
                    "--reasoning-effort",
                    request["reasoning_effort"],
                    "--api-key-env",
                    job.model_api_key_env or "",
                ]
            )
            if request["base_url"]:
                command.extend(["--base-url", request["base_url"]])
            if request.get("reasoning_profile") is not None:
                command.extend(["--reasoning-profile", json.dumps(request["reasoning_profile"], separators=(",", ":"))])
            if request["allow_model_alias"]:
                command.append("--allow-model-alias")
        return command

    def _worker_package_directory(self) -> Path:
        """Select one source tree, including the ordinary installed-package case."""
        loaded_package = Path(__file__).resolve().parents[1]
        if loaded_package.parent.name != "src" and not self._explicit_project_root:
            # A wheel installation has no project/src tree. Bind the package
            # that is actually running the supervisor, not another sys.path hit.
            return loaded_package
        return self.project_root / "src" / "gwasmind"

    def _child_environment(self, job: _Job) -> dict[str, str]:
        environment = dict(os.environ)
        source_root = str(self._worker_package_directory().parent)
        inherited_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = source_root + (
            os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
        )
        environment.pop(_RUNTIME_API_KEY_ENV, None)
        if self.model_api_key_env:
            environment.pop(self.model_api_key_env, None)
        if job.model_api_key and job.model_api_key_env:
            environment[job.model_api_key_env] = job.model_api_key
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": str(WORKER_THREADS),
                "OPENBLAS_NUM_THREADS": str(WORKER_THREADS),
                "MKL_NUM_THREADS": str(WORKER_THREADS),
                "NUMEXPR_NUM_THREADS": str(WORKER_THREADS),
                "DUCKDB_THREADS": str(WORKER_THREADS),
            }
        )
        return environment

    def _drain_progress(
        self,
        job: _Job,
        query_index: int,
        path: Path,
        offset: int,
        *,
        final: bool = False,
    ) -> int:
        if not path.is_file():
            return offset
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not final and not line.endswith("\n"):
                    offset = line_start
                    break
                offset = handle.tell()
                try:
                    child_event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(child_event, Mapping):
                    continue
                phase = str(child_event.get("phase", "unknown"))
                details = child_event.get("details")
                if not isinstance(details, Mapping):
                    details = {}
                with self._changed:
                    self._update_query_fraction_locked(job, phase, details)
                    self._emit_locked(
                        job,
                        "progress",
                        _phase_message(phase),
                        query_index=query_index,
                        phase=phase,
                        details=dict(details),
                        child_run_id=str(child_event.get("run_id", "")),
                        child_timestamp=child_event.get("timestamp"),
                    )
        return offset

    def _update_query_fraction_locked(
        self, job: _Job, phase: str, details: Mapping[str, Any]
    ) -> None:
        if phase == "semantic_tool_started":
            step = _positive_int(details.get("step"))
            total = _positive_int(details.get("total_steps"))
            if step and total:
                job.query_fraction = max(job.query_fraction, (step - 1) / total)
        elif phase == "semantic_tool_finished":
            step = _positive_int(details.get("step"))
            total = _positive_int(details.get("total_steps"))
            if step and total:
                job.query_fraction = max(job.query_fraction, min(step / total, 0.95))
        elif phase == "completed":
            job.query_fraction = 1.0

    def _emit_locked(
        self,
        job: _Job,
        event_type: str,
        message: str,
        **extra: Any,
    ) -> dict[str, Any]:
        completed = job.completed_queries + (
            job.query_fraction if job.current_query is not None else 0.0
        )
        percent = int(min(100.0, max(0.0, completed / job.query_count * 100)))
        event = {
            "seq": job.next_sequence,
            "run_id": job.run_id,
            "type": event_type,
            "state": job.state,
            "timestamp": _utc_now(),
            "message": message,
            "query_index": job.current_query,
            "completed_queries": job.completed_queries,
            "query_count": job.query_count,
            "percent": percent,
            **extra,
        }
        job.next_sequence += 1
        job.events.append(event)
        if len(job.events) > MAX_EVENTS_PER_JOB:
            del job.events[: len(job.events) - MAX_EVENTS_PER_JOB]
        self._changed.notify_all()
        return event

    def _heartbeat_locked(self, job: _Job) -> dict[str, Any]:
        """Build a keepalive without consuming event history or sequence IDs."""

        completed = job.completed_queries + (
            job.query_fraction if job.current_query is not None else 0.0
        )
        percent = int(min(100.0, max(0.0, completed / job.query_count * 100)))
        return {
            "seq": max(0, job.next_sequence - 1),
            "run_id": job.run_id,
            "type": "heartbeat",
            "state": job.state,
            "timestamp": _utc_now(),
            "message": "Analysis is active.",
            "query_index": job.current_query,
            "completed_queries": job.completed_queries,
            "query_count": job.query_count,
            "percent": percent,
            "transient": True,
        }

    def _evict_terminal_jobs_locked(self) -> None:
        overflow = len(self._jobs) - MAX_JOB_HISTORY + 1
        if overflow <= 0:
            return
        terminal = sorted(
            (job for job in self._jobs.values() if job.state in TERMINAL_STATES),
            key=lambda item: item.finished_at or item.created_at,
        )
        for job in terminal[:overflow]:
            self._jobs.pop(job.run_id, None)

    def _snapshot_locked(self, job: _Job) -> dict[str, Any]:
        completed = job.completed_queries + (
            job.query_fraction if job.current_query is not None else 0.0
        )
        percent = int(min(100.0, max(0.0, completed / job.query_count * 100)))
        return {
            "run_id": job.run_id,
            "state": job.state,
            "mode": job.request["mode"],
            "release": job.request["release"],
            "query_count": job.query_count,
            "current_query": job.current_query,
            "completed_queries": job.completed_queries,
            "progress": {
                "completed": job.completed_queries,
                "total": job.query_count,
                "percent": percent,
            },
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "cancel_requested": job.cancel_requested,
            "error": job.error,
            "decision_statuses": [str(result.get("status", "unknown")) for result in job.results],
        }

    def _persist_locked(self, job: _Job) -> None:
        _atomic_json(job.root / "job.json", self._snapshot_locked(job))

    def _finish_cancelled_locked(self, job: _Job) -> None:
        if job.state == "cancelled":
            return
        job.state = "cancelled"
        job.current_query = None
        job.finished_at = _utc_now()
        job.process = None
        job.model_api_key = None
        self._emit_locked(job, "cancelled", "Analysis cancelled.")

    def _finish_failed_locked(self, job: _Job, message: str) -> None:
        job.state = "failed"
        job.current_query = None
        job.finished_at = _utc_now()
        job.process = None
        job.model_api_key = None
        job.error = message[:4000]
        self._emit_locked(job, "error", "Analysis failed.", error=job.error)

    def _get_locked(self, run_id: str) -> _Job:
        job = self._jobs.get(run_id)
        if job is None:
            raise JobNotFoundError("run was not found")
        return job

    @staticmethod
    def _query_root(job: _Job, query_index: int) -> Path:
        root = job.root / "queries" / f"{query_index:03d}"
        return root / "ranking" if job.reviewed_retrieval is not None else root

    @staticmethod
    def _child_run_id(job: _Job, query_index: int) -> str:
        return f"{job.run_id}-q{query_index + 1:03d}"

    @staticmethod
    def _terminate_exact_process(process: Any) -> None:
        try:
            process.terminate()
        except (OSError, ProcessLookupError):
            return
        try:
            process.wait(timeout=2.0)
        except (subprocess.TimeoutExpired, TimeoutError):
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                return
            try:
                process.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, TimeoutError, OSError, ProcessLookupError):
                return
        except (OSError, ProcessLookupError):
            return

    @classmethod
    def _terminate_if_running(cls, process: Any) -> None:
        try:
            running = process.poll() is None
        except (OSError, ProcessLookupError):
            return
        if running:
            cls._terminate_exact_process(process)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def _phase_message(phase: str) -> str:
    messages = {
        "validating_input": "Validating locus input.",
        "planning_evidence": "Planning evidence retrieval.",
        "retrieving_evidence": "Retrieving Open Targets evidence.",
        "semantic_input_validation": "Validating semantic-tool input.",
        "capability_check": "Checking provider capabilities.",
        "semantic_plan_ready": "Semantic evidence plan is ready.",
        "semantic_tool_started": "Semantic evidence tool started.",
        "semantic_tool_finished": "Semantic evidence tool finished.",
        "checking_sufficiency": "Checking evidence sufficiency.",
        "synthesizing_decision": "Synthesizing the gene-prioritization result.",
        "writing_artifacts": "Writing reproducibility artifacts.",
        "completed": "Locus analysis completed.",
    }
    return messages.get(phase, phase.replace("_", " ").capitalize() + ".")


def _verified_source_only_bootstrap_failure(
    output_path: Path,
    *,
    query_value: Mapping[str, Any],
    expected_run_id: str,
    expected_release: str,
) -> bool:
    """Bind a bounded bootstrap failure receipt to this exact worker input."""
    path = output_path / "schema_bootstrap.json"
    try:
        if not path.is_file() or path.stat().st_size > 16_384:
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(value, Mapping)
        and type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("run_id") == expected_run_id
        and value.get("source_input_sha256") == sha256_json(query_value)
        and value.get("requested_release") == expected_release
        and "observed_release" in value and value["observed_release"] is None
        and value.get("source_only") is True
        and value.get("schema_verified") is False
        and value.get("status") == "unavailable"
        and value.get("error_type") in {"LiveProviderError", "OSError", "ValueError"}
    )


def _validate_result_artifact(
    value: Mapping[str, Any],
    *,
    expected_locus_id: str,
    expected_run_id: str,
    expected_mode: str,
    expected_release: str,
    return_code: int,
    source_only_bootstrap_failure: bool = False,
) -> dict[str, Any]:
    required = {
        "locus_id",
        "status",
        "ranked_genes",
        "evidence",
        "sufficiency",
        "summary",
        "run_id",
        "model_id",
        "data_release",
        "data_mode",
        "limitations",
        "artifacts",
    }
    if required.difference(value):
        raise _PublicJobFailure("GWASMind result artifact does not match the AgentResult contract.")
    if value.get("locus_id") != expected_locus_id or value.get("run_id") != expected_run_id:
        raise _PublicJobFailure(
            "GWASMind result artifact identifiers do not match the submitted locus."
        )
    status = value.get("status")
    allowed_statuses = {item.value for item in DecisionStatus}
    if status not in allowed_statuses:
        raise _PublicJobFailure("GWASMind result artifact has an invalid decision status.")
    expected_by_exit = {
        0: {
            DecisionStatus.PRIORITIZED.value,
            DecisionStatus.EVIDENCE_RETRIEVED.value,
        },
        2: {
            DecisionStatus.INVALID_INPUT.value,
            DecisionStatus.NO_GWAS_SIGNAL.value,
        },
        3: {
            DecisionStatus.INSUFFICIENT_EVIDENCE.value,
            DecisionStatus.CONFLICTING_EVIDENCE.value,
            DecisionStatus.POLICY_BLOCKED.value,
            DecisionStatus.INCOMPLETE.value,
        },
    }
    if source_only_bootstrap_failure:
        expected_by_exit[4] = {DecisionStatus.SERVICE_UNAVAILABLE.value}
    if status not in expected_by_exit.get(return_code, set()):
        raise _PublicJobFailure("GWASMind result status is inconsistent with the CLI exit code.")
    actual_release = value.get("data_release")
    release_matches = actual_release == expected_release
    if expected_mode == "live" and expected_release.casefold() == "live":
        release_matches = (
            isinstance(actual_release, str)
            and bool(actual_release.strip())
            and actual_release.casefold() not in {"live", "unknown", "unverified"}
        )
        if source_only_bootstrap_failure:
            # No upstream release was observed. Keep the requested value only
            # for this explicitly empty external-evidence result.
            release_matches = actual_release == expected_release
    if value.get("data_mode") != expected_mode or not release_matches:
        raise _PublicJobFailure(
            "GWASMind result data mode or release differs from the requested run."
        )
    if not isinstance(value.get("summary"), str) or not value["summary"].strip():
        raise _PublicJobFailure("GWASMind result artifact has no summary.")
    if not isinstance(value.get("model_id"), str) or not value["model_id"].strip():
        raise _PublicJobFailure("GWASMind result artifact has no model identity.")
    if not isinstance(value.get("limitations"), list) or any(
        not isinstance(item, str) for item in value["limitations"]
    ):
        raise _PublicJobFailure("GWASMind result limitations are malformed.")
    if not isinstance(value.get("artifacts"), Mapping):
        raise _PublicJobFailure("GWASMind result artifact pointers are malformed.")

    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        raise _PublicJobFailure("GWASMind evidence ledger is not an array.")
    evidence_ids: set[str] = set()
    evidence_kinds = {item.value for item in EvidenceKind}
    for record in evidence:
        if not isinstance(record, Mapping):
            raise _PublicJobFailure("GWASMind evidence ledger contains a malformed row.")
        if not isinstance(record.get("evidence_id"), str) or not record["evidence_id"]:
            raise _PublicJobFailure("GWASMind evidence ledger contains an invalid ID.")
        if record["evidence_id"] in evidence_ids:
            raise _PublicJobFailure("GWASMind evidence ledger contains duplicate IDs.")
        evidence_ids.add(record["evidence_id"])
        if record.get("kind") not in evidence_kinds:
            raise _PublicJobFailure("GWASMind evidence ledger contains an invalid kind.")
        for field_name in ("dataset", "source", "data_release"):
            if not isinstance(record.get(field_name), str) or not record[field_name]:
                raise _PublicJobFailure("GWASMind evidence ledger has incomplete provenance.")
        if not isinstance(record.get("payload"), Mapping):
            raise _PublicJobFailure("GWASMind evidence ledger payload is malformed.")

    ranked = value.get("ranked_genes")
    if not isinstance(ranked, list):
        raise _PublicJobFailure("GWASMind ranked_genes is not an array.")
    retained_review = retained_model_review(value)
    for index, row in enumerate(ranked, start=1):
        if not isinstance(row, Mapping):
            raise _PublicJobFailure("GWASMind ranked_genes contains a malformed row.")
        if (
            not isinstance(row.get("gene_id"), str)
            or not row["gene_id"].strip()
            or not isinstance(row.get("rank"), int)
            or isinstance(row.get("rank"), bool)
            or row["rank"] < 1
            or (not retained_review and row.get("rank") != index)
            or not isinstance(row.get("evidence_ids"), list)
            or any(not isinstance(item, str) for item in row["evidence_ids"])
            or (not retained_review and not set(row["evidence_ids"]).issubset(evidence_ids))
            or not isinstance(row.get("rationale"), str)
            or not isinstance(row.get("limitations"), list)
            or any(not isinstance(item, str) for item in row["limitations"])
        ):
            raise _PublicJobFailure("GWASMind ranked gene contract is invalid.")

    sufficiency = value.get("sufficiency")
    sufficiency_fields = {
        "valid_input",
        "significant_signal",
        "evidence_sources",
        "evidence_records",
        "genes_with_evidence",
        "conflicts",
        "policy_blocks",
        "decision_evidence_records",
        "locus_linked_records",
        "locus_linked_genes",
        "relevant_tissue_coverage",
        "evidence_dimensions",
        "missing_relevant_tissues",
        "reasons",
    }
    if not isinstance(sufficiency, Mapping) or sufficiency_fields.difference(sufficiency):
        raise _PublicJobFailure("GWASMind sufficiency report is incomplete.")
    if not isinstance(sufficiency.get("valid_input"), bool) or (
        sufficiency.get("significant_signal") is not None
        and not isinstance(sufficiency.get("significant_signal"), bool)
    ):
        raise _PublicJobFailure("GWASMind sufficiency flags are malformed.")
    for field_name in (
        "evidence_sources",
        "evidence_records",
        "genes_with_evidence",
        "conflicts",
        "policy_blocks",
        "decision_evidence_records",
        "locus_linked_records",
        "locus_linked_genes",
    ):
        field_value = sufficiency.get(field_name)
        if not isinstance(field_value, int) or isinstance(field_value, bool) or field_value < 0:
            raise _PublicJobFailure("GWASMind sufficiency counts are malformed.")
    for field_name in ("evidence_dimensions", "missing_relevant_tissues", "reasons"):
        field_value = sufficiency.get(field_name)
        if not isinstance(field_value, list) or any(
            not isinstance(item, str) for item in field_value
        ):
            raise _PublicJobFailure("GWASMind sufficiency details are malformed.")
    tissue_coverage = sufficiency.get("relevant_tissue_coverage")
    if tissue_coverage is not None and not isinstance(tissue_coverage, bool):
        raise _PublicJobFailure("GWASMind tissue-coverage flag is malformed.")
    if sufficiency["evidence_records"] != len(evidence):
        raise _PublicJobFailure(
            "GWASMind sufficiency evidence count disagrees with the evidence ledger."
        )
    if source_only_bootstrap_failure and (
        return_code != 4 or expected_mode != "live"
        or status != DecisionStatus.SERVICE_UNAVAILABLE.value
        or evidence or ranked or sufficiency["valid_input"] is not True
        or any(sufficiency[name] != 0 for name in (
            "evidence_sources", "evidence_records", "genes_with_evidence", "conflicts",
            "policy_blocks", "decision_evidence_records", "locus_linked_records", "locus_linked_genes",
        ))
    ):
        raise _PublicJobFailure("GWASMind source-only recovery contains an invalid evidence contract.")
    if status == DecisionStatus.PRIORITIZED.value:
        if (
            not ranked
            or not evidence
            or sufficiency["reasons"]
            or sufficiency["valid_input"] is not True
            or sufficiency["significant_signal"] is not True
        ):
            raise _PublicJobFailure(
                "GWASMind prioritized result lacks the required evidence contract."
            )
    elif status == DecisionStatus.EVIDENCE_RETRIEVED.value:
        if (
            ranked
            or not evidence
            or sufficiency["reasons"]
            or sufficiency["valid_input"] is not True
        ):
            raise _PublicJobFailure(
                "GWASMind evidence-retrieval result lacks the required evidence contract."
            )
    elif ranked:
        raise _PublicJobFailure("GWASMind abstention result must not contain a gene ranking.")
    return dict(value)


def _model_config_fingerprint(
    api_key: str,
    base_url: str,
    model_id: str,
    thinking_mode: str,
    reasoning_effort: str,
    reasoning_profile: Mapping[str, Any] | None = None,
) -> str:
    value = "\0".join(
        (
            api_key,
            base_url.rstrip("/"),
            model_id.strip(),
            normalize_thinking_mode(thinking_mode),
            normalize_reasoning_effort(reasoning_effort),
        )
    )
    if reasoning_profile is not None:
        value += "\0" + json.dumps(reasoning_profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_runtime_provider_id(value: Any, base_url: str, model_id: str) -> str:
    if value is None:
        return infer_provider_id(base_url, model_id)
    if not isinstance(value, str) or not value.strip():
        raise JobValidationError("provider_id must be a non-empty string")
    provider = get_provider(value.strip())
    if provider is None:
        raise JobValidationError("unknown model provider")
    if not provider.configurable:
        raise JobValidationError(
            provider.disabled_reason or "the selected model provider is not configurable"
        )
    return provider.id


def _safe_model_id(value: Any) -> bool:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 120:
        return False
    return not any(ord(character) < 32 or ord(character) == 127 for character in value.strip())


def _safe_model_base_url(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
    ):
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    if port is not None and port <= 0:
        return False
    if not parsed.hostname or parsed.username or parsed.password or parsed.fragment or parsed.query:
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    normalized = parsed.hostname.casefold().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False
