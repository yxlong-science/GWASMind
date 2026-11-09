from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from time import monotonic
from uuid import uuid4

from ..artifacts import ArtifactStore
from ..audit import AuditLogger, sha256_json
from ..configuration import parameter
from ..evidence_store import record_response_sha256
from ..models import DataMode, EvidenceKind, LocusQuery
from ..policy import LeakagePolicy, PolicyViolation
from ..progress import ProgressSink, emit_progress
from ..projection import project_record_for_model
from ..providers.base import ProviderCapabilities
from ..region_gate import REGION_GENE_CONTEXT_DATASETS
from ..retrieval_operations import retrieval_operation_issues
from ..retrieval_status import module_retrieval_statuses
from ..tools import ToolCallResult, ToolStatus
from ..v2g.pipeline import gene_centric_enabled
from .executor import RetrievalStatus, SemanticRetrievalExecutor
from .gene_centric import gene_centric_tool_result
from .normalization import evidence_kind_for_module, semantic_records_to_evidence
from .service import ModuleAdapter, SemanticToolService

INTERACTIVE_SEMANTIC_PAGE_SIZE = parameter("retrieval.interactive_page_size")


@dataclass(frozen=True, slots=True)
class _SemanticProviderView:
    capabilities: ProviderCapabilities


class SemanticGatewayAdapter:
    """Adapt staged semantic MCP retrieval to the existing agent boundary."""

    def __init__(
        self,
        service: SemanticToolService,
        policy: LeakagePolicy,
        *,
        artifacts: ArtifactStore | None = None,
        audit: AuditLogger | None = None,
        executor: SemanticRetrievalExecutor | None = None,
        progress_sink: ProgressSink | None = None,
    ):
        expected_providers = (
            {"official_graphql", "official_mcp"} if policy.mode is DataMode.LIVE else {"frozen"}
        )
        if service.provider not in expected_providers:
            raise ValueError(
                "semantic service/policy mode mismatch: "
                f"{service.provider} not in {sorted(expected_providers)}"
            )
        self.service = service
        self.policy = policy
        self.artifacts = artifacts
        self.audit = audit or AuditLogger()
        self.progress_sink = progress_sink
        self.executor = executor or SemanticRetrievalExecutor(
            service,
            page_size=INTERACTIVE_SEMANTIC_PAGE_SIZE,
            progress_sink=progress_sink,
            source_identity_resolver=(self._resolve_source_identity
                                      if policy.mode is DataMode.LIVE else None),
        )
        self.provider = _SemanticProviderView(
            ProviderCapabilities(
                mode=policy.mode,
                release=service.data_release,
                evidence_kinds=_service_evidence_kinds(service),
                snapshot_hash=service.snapshot_sha256,
                schema_hash=service.schema_sha256,
                modules=frozenset(module for _, module in service.adapters),
                supports_l2g=(
                    "locus.l2g" in {module for _, module in service.adapters}
                    and "locus.l2g" not in service.blocked_modules
                ),
            )
        )

    @property
    def available_kinds(self) -> frozenset[EvidenceKind]:
        return self.policy.allowed_kinds.intersection(self.provider.capabilities.evidence_kinds)

    async def fetch_evidence(self, query: LocusQuery, kinds: set[EvidenceKind]) -> ToolCallResult:
        call_id = str(uuid4())
        requested = tuple(sorted(kinds, key=lambda item: item.value))
        request_value = {
            "call_id": call_id,
            "locus_id": query.locus_id,
            "requested_kinds": [item.value for item in requested],
            "qtl_types": list(query.qtl_types),
            "semantic_service": self.service.provider,
            "data_release": self.service.data_release,
        }
        request_pointer = (
            self.artifacts.write_json(f"tool_calls/{call_id}/request.json", request_value)
            if self.artifacts
            else None
        )
        self.audit.log("semantic_retrieval_started", **request_value)
        try:
            self.policy.validate_query(query, kinds)
        except (PolicyViolation, ValueError) as exc:
            policy_blocked = isinstance(exc, PolicyViolation)
            return ToolCallResult(
                call_id=call_id,
                status=ToolStatus.BLOCKED if policy_blocked else ToolStatus.INVALID_INPUT,
                requested_kinds=requested,
                records=(),
                policy_blocks=(str(exc),) if policy_blocked else (),
                request_artifact=request_pointer,
                response_artifact=None,
                response_sha256=sha256_json([]),
                error=str(exc),
                resolved_query=query,
            )

        if (query.task == "prioritize_genes" and not query.benchmark
                and self.policy.mode is DataMode.LIVE and gene_centric_enabled()):
            return await gene_centric_tool_result(self, query, kinds, call_id=call_id, requested=requested,
                                                  request_pointer=request_pointer)

        retrieval = await self.executor.retrieve(query, requested_kinds=frozenset(kinds))
        emit_progress(self.progress_sink, "result_processing", status="started",
                      operation="saving_raw", bytes_written=0)
        _, raw_progress = self._processing_progress("saving_raw")
        raw_value = {
            "status": retrieval.status.value,
            "candidate_audit": retrieval.candidate_audit,
            "operation_issues": list(retrieval.operation_issues),
            "source_identity_normalizations": list(retrieval.source_identity_normalizations),
            "resolved_query": {
                "locus_id": retrieval.query.locus_id,
                "variant_ids": [item.variant_id for item in retrieval.query.variants],
                "candidate_genes": list(retrieval.query.candidate_genes),
                "qtl_types": list(retrieval.query.qtl_types),
                "biological_context": retrieval.query.biological_context,
            },
            "plan": retrieval.plan.to_dict() if retrieval.plan else None,
            "calls": [item.to_dict() for item in retrieval.calls],
            "records": list(retrieval.records),
            "reasons": list(retrieval.reasons),
            "warnings": list(retrieval.warnings),
        }
        raw_pointer = (
            self.artifacts.write_json(f"tool_calls/{call_id}/raw_response.json", raw_value, compact=True,
                                      byte_progress=raw_progress)
            if self.artifacts
            else None
        )
        raw_sha256 = raw_pointer.sha256 if raw_pointer is not None else sha256_json(raw_value)
        emit_progress(self.progress_sink, "result_processing", status="completed",
                      operation="saving_raw", records=len(retrieval.records),
                      **({"bytes_written": raw_pointer.bytes} if raw_pointer is not None else {}))
        if retrieval.status is RetrievalStatus.BLOCKED:
            status = ToolStatus.BLOCKED
        elif retrieval.status is RetrievalStatus.INVALID_INPUT:
            status = ToolStatus.INVALID_INPUT
        elif retrieval.status is RetrievalStatus.INCOMPLETE:
            status = ToolStatus.INCOMPLETE
        elif retrieval.status is RetrievalStatus.ERROR:
            status = ToolStatus.ERROR
        else:
            status = ToolStatus.SUCCESS

        records = semantic_records_to_evidence(
            self._processing_records(retrieval.records, "normalizing"),
            default_release=self.service.data_release,
        )
        # Mandatory locus-validation calls are operational context.  They are
        # retained in the raw artifact but never become model-visible evidence
        # unless the outer evidence planner explicitly requested their kind.
        records = tuple(
            record
            for record in records
            if record.kind in kinds or record.dataset in REGION_GENE_CONTEXT_DATASETS
        )
        allowed, decisions = self.policy.filter_records(self._processing_records(records, "checking_policy"))
        blocked = tuple(item.reason for item in decisions if not item.allowed)
        hard_block = bool(blocked and self.policy.block_on_any_denied_record)
        if hard_block:
            status = ToolStatus.BLOCKED
            allowed = []
        projected = tuple(
            project_record_for_model(record, relevant_tissues=retrieval.query.relevant_tissues)
            for record in self._processing_records(allowed, "projecting")
        )
        from ..sufficiency import decision_evidence_for_query
        usable = decision_evidence_for_query(retrieval.query, projected)
        if query.task in {"prioritize_genes", "gene_associations", "target_assessment"}:
            usable = tuple(record for record in usable if record.gene_id)
        retrieval = replace(retrieval, operation_issues=retrieval_operation_issues(
            retrieval.operation_issues, has_usable_result=bool(usable)))
        if status is ToolStatus.SUCCESS and not projected:
            status = ToolStatus.NO_DATA
        response_value = {
            "call_id": call_id,
            "status": status.value,
            "semantic_status": retrieval.status.value,
            "policy_blocks": list(blocked),
            "raw_response_sha256": raw_sha256,
            "reasons": list(retrieval.reasons),
            "warnings": list(retrieval.warnings),
            **({"source_identity_normalizations": list(retrieval.source_identity_normalizations)}
               if retrieval.source_identity_normalizations else {}),
            **({"operation_issues": list(retrieval.operation_issues)} if retrieval.operation_issues else {}),
        }
        emit_progress(self.progress_sink, "result_processing", status="started",
                      operation="saving_response", records=0, total_records=len(projected), bytes_written=0)
        record_progress, byte_progress = self._processing_progress("saving_response", len(projected))
        response_pointer = (
            self.artifacts.write_record_response(f"tool_calls/{call_id}/response.json", response_value, projected,
                progress=record_progress, byte_progress=byte_progress)
            if self.artifacts
            else None
        )
        response_sha256 = response_pointer.sha256 if response_pointer is not None else record_response_sha256(
            response_value, projected, progress=record_progress)
        emit_progress(self.progress_sink, "result_processing", status="completed",
                      operation="saving_response", records=len(projected), total_records=len(projected),
                      **({"bytes_written": response_pointer.bytes} if response_pointer is not None else {}))
        self.audit.log(
            "semantic_retrieval_finished",
            call_id=call_id,
            status=status.value,
            semantic_status=retrieval.status.value,
            records=len(projected),
            response_sha256=response_sha256,
        )
        return ToolCallResult(
            call_id=call_id,
            status=status,
            requested_kinds=requested,
            records=projected,
            policy_blocks=blocked,
            request_artifact=request_pointer,
            response_artifact=response_pointer,
            response_sha256=response_sha256,
            error="; ".join(retrieval.reasons) or None,
            raw_response_artifact=raw_pointer,
            raw_response_sha256=raw_sha256,
            warnings=retrieval.warnings,
            resolved_query=retrieval.query,
            candidate_audit=retrieval.candidate_audit,
            operation_issues=tuple(dict(item) for item in retrieval.operation_issues),
            retrieval_statuses=module_retrieval_statuses(retrieval.calls, retrieval.plan),
        )

    async def _resolve_source_identity(self, identifiers):
        """One bounded OT supplement for exact source IDs missing in context."""
        from ..providers.ot_identity import OTIdentityResolver

        resolver = OTIdentityResolver(
            cache_directory=self.artifacts.root / "source_identity" if self.artifacts is not None else None,
            timeout_seconds=15.0, total_timeout_seconds=60.0, progress_sink=self.progress_sink)
        return await asyncio.to_thread(resolver.resolve_many, identifiers)

    def _processing_progress(self, operation, total_records=None):
        last = monotonic()
        interval = parameter("execution.stream_progress_interval_seconds")
        counts = {} if total_records is None else {"records": 0, "total_records": total_records}

        def update(**values):
            nonlocal last
            counts.update(values)
            now = monotonic()
            if now - last >= interval:
                emit_progress(self.progress_sink, "result_processing", status="running",
                              operation=operation, **counts)
                last = now
        return (lambda completed, total: update(records=completed, total_records=total),
                lambda size: update(bytes_written=size))

    def _processing_records(self, records, operation):
        """Count consumed records; an interrupted consumer never reports completion."""
        total = len(records)
        emit_progress(self.progress_sink, "result_processing", status="started",
                      operation=operation, records=0, total_records=total)
        progress, _ = self._processing_progress(operation, total)
        for index, record in enumerate(records, start=1):
            yield record
            progress(index, total)
        emit_progress(self.progress_sink, "result_processing", status="completed",
                      operation=operation, records=total, total_records=total)


def blocked_modules_for_policy(
    adapters: tuple[ModuleAdapter, ...],
    policy: LeakagePolicy,
    *,
    benchmark: bool,
) -> frozenset[str]:
    blocked: set[str] = set()
    for adapter in adapters:
        if adapter.module == "entity.resolve":
            continue
        try:
            kind = evidence_kind_for_module(adapter.module)
        except ValueError:
            blocked.add(adapter.module)
            continue
        if kind not in policy.allowed_kinds:
            blocked.add(adapter.module)
        if adapter.module == "locus.l2g" and (
            benchmark or "l2g_prediction" in policy.forbidden_datasets
        ):
            blocked.add(adapter.module)
    return frozenset(blocked)


def _service_evidence_kinds(
    service: SemanticToolService,
) -> frozenset[EvidenceKind]:
    kinds: set[EvidenceKind] = set()
    for _, module in service.adapters:
        if module == "entity.resolve" or module in service.blocked_modules:
            continue
        try:
            kinds.add(evidence_kind_for_module(module))
        except ValueError:
            continue
    return frozenset(kinds)
