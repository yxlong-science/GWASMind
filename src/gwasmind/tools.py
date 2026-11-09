from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import StrEnum
from uuid import uuid4

from .artifacts import ArtifactPointer, ArtifactStore
from .audit import AuditLogger, sha256_json
from .models import DataMode, EvidenceKind, EvidenceRecord, LocusQuery
from .policy import LeakagePolicy, PolicyViolation
from .projection import project_record_for_model
from .providers.base import OpenTargetsProvider


class ToolStatus(StrEnum):
    SUCCESS = "success"
    NO_DATA = "no_data"
    BLOCKED = "blocked"
    INVALID_INPUT = "invalid_input"
    INCOMPLETE = "incomplete"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    call_id: str
    status: ToolStatus
    requested_kinds: tuple[EvidenceKind, ...]
    records: tuple[EvidenceRecord, ...]
    policy_blocks: tuple[str, ...]
    request_artifact: ArtifactPointer | None
    response_artifact: ArtifactPointer | None
    response_sha256: str
    error: str | None = None
    raw_response_artifact: ArtifactPointer | None = None
    raw_response_sha256: str = ""
    warnings: tuple[str, ...] = ()
    resolved_query: LocusQuery | None = None
    candidate_audit: dict | None = None
    operation_issues: tuple[dict, ...] = ()
    retrieval_statuses: dict[str, str] = field(default_factory=dict)


def reviewed_tool_result(value: dict, query: LocusQuery, *, record_iterator=None) -> ToolCallResult:
    """Restore a private retrieval artifact for inference without another lookup."""
    from .serialization import locus_query_from_dict

    saved_input = value.get("input_query")
    if isinstance(saved_input, dict):
        saved_input = {"analysis_variants": [], "ld_expansion": None, "biological_context": None, **saved_input}
    # JSON arrays and dataclass tuples have different Python representations.
    if saved_input != asdict(query) and sha256_json(saved_input) != sha256_json(asdict(query)):
        raise ValueError("reviewed retrieval does not belong to this input query")
    resolved = locus_query_from_dict(value["query"], allow_analysis_scope=True)
    if query.ld_expansion is None and resolved.ld_expansion is not None:
        raise ValueError("reviewed retrieval introduced an unrequested LD expansion")
    if query.ld_expansion is not None:
        # Resolution may add aliases/omissions, but may not replace the reviewed
        # reference queries, input observations or LD measurements.
        for key in ("source_observations", "analysis_variants", "query_ledger", "threshold",
                    "window_bp", "reference_panel", "population", "genome_build"):
            if sha256_json(query.ld_expansion.get(key)) != sha256_json((resolved.ld_expansion or {}).get(key)):
                raise ValueError("reviewed retrieval changed the recorded LD expansion")
        original_links = query.ld_expansion.get("ld_links", [])
        if any(link not in original_links for link in [*(resolved.ld_expansion or {}).get("ld_links", []),
                                                       *(resolved.ld_expansion or {}).get("excluded_ld_links", [])]):
            raise ValueError("reviewed retrieval introduced an unreviewed LD link")
    rows = value.get("records") if record_iterator is None else record_iterator
    if record_iterator is None and not isinstance(rows, list):
        raise ValueError("reviewed retrieval is missing its evidence ledger")
    records = []
    for item in rows:
        row = dict(item)
        row["kind"] = EvidenceKind(row["kind"])
        row["evidence_date"] = (
            date.fromisoformat(row["evidence_date"]) if row.get("evidence_date") else None
        )
        if row.get("retrieved_at"):
            row["retrieved_at"] = datetime.fromisoformat(row["retrieved_at"])
        for key in ("parent_variant_ids", "parent_evidence_ids"):
            row[key] = tuple(row.get(key, ()))
        records.append(EvidenceRecord(**row))
    return ToolCallResult(
        call_id=value["tool_call_id"],
        status=ToolStatus(value["status"]),
        requested_kinds=tuple(EvidenceKind(kind) for kind in value["plan"]["kinds"]),
        records=tuple(records),
        policy_blocks=tuple(value.get("policy_blocks", ())),
        request_artifact=None,
        response_artifact=None,
        response_sha256=value.get("response_sha256", ""),
        raw_response_sha256=value.get("raw_response_sha256", ""),
        retrieval_statuses=dict(value.get("retrieval_statuses", {})),
        error=value.get("error"),
        warnings=tuple(value.get("warnings", ())),
        resolved_query=resolved,
        candidate_audit=value.get("candidate_audit"),
        operation_issues=tuple(value.get("operation_issues", ())),
    )


class GWASToolGateway:
    """Stable tool contract shared by the frozen and official-live providers."""

    def __init__(
        self,
        provider: OpenTargetsProvider,
        policy: LeakagePolicy,
        *,
        audit: AuditLogger | None = None,
        artifacts: ArtifactStore | None = None,
    ):
        self.provider = provider
        self.policy = policy
        self.audit = audit or AuditLogger()
        self.artifacts = artifacts
        if provider.capabilities.mode is not policy.mode:
            raise ValueError(
                "provider/policy mode mismatch: "
                f"{provider.capabilities.mode.value} != {policy.mode.value}"
            )

    @property
    def available_kinds(self) -> frozenset[EvidenceKind]:
        return self.provider.capabilities.evidence_kinds.intersection(self.policy.allowed_kinds)

    async def fetch_evidence(self, query: LocusQuery, kinds: set[EvidenceKind]) -> ToolCallResult:
        call_id = str(uuid4())
        requested = tuple(sorted(kinds, key=lambda item: item.value))
        request_value = {
            "call_id": call_id,
            "locus_id": query.locus_id,
            "trait": query.trait,
            "source_trait": query.source_trait or query.trait,
            "variant_ids": [item.variant_id for item in query.variants],
            "candidate_genes": list(query.candidate_genes),
            "qtl_types": list(query.qtl_types),
            "requested_kinds": [item.value for item in requested],
            "provider_mode": self.provider.capabilities.mode.value,
            "data_release": self.provider.capabilities.release,
        }
        request_pointer = (
            self.artifacts.write_json(f"tool_calls/{call_id}/request.json", request_value)
            if self.artifacts
            else None
        )
        self.audit.log("tool_call_started", **request_value)
        try:
            if query.analysis_variants:
                raise PolicyViolation("LD-expanded retrieval requires the semantic tool contract")
            if query.benchmark and self.provider.capabilities.mode is DataMode.LIVE:
                raise PolicyViolation("benchmark loci cannot use the live Open Targets provider")
            self.policy.validate_query(query, kinds)
        except PolicyViolation as exc:
            self.audit.log("tool_call_blocked", call_id=call_id, reason=str(exc))
            return ToolCallResult(
                call_id=call_id,
                status=ToolStatus.BLOCKED,
                requested_kinds=requested,
                records=(),
                policy_blocks=(str(exc),),
                request_artifact=request_pointer,
                response_artifact=None,
                response_sha256=sha256_json([]),
                error=str(exc),
                resolved_query=query,
            )

        try:
            raw = await self.provider.fetch(query, kinds)
            unexpected = [record.kind for record in raw if record.kind not in kinds]
            if unexpected:
                names = ", ".join(sorted({item.value for item in unexpected}))
                raise PolicyViolation(f"provider returned unrequested evidence kinds: {names}")
            allowed, decisions = self.policy.filter_records(raw)
            blocked = tuple(item.reason for item in decisions if not item.allowed)
            hard_block = bool(blocked and self.policy.block_on_any_denied_record)
            raw_response_value = {
                "call_id": call_id,
                "records": [record.to_dict() for record in allowed],
            }
            raw_response_pointer = (
                self.artifacts.write_json(
                    f"tool_calls/{call_id}/raw_response.json", raw_response_value
                )
                if self.artifacts and not hard_block
                else None
            )
            projected = [
                project_record_for_model(record, relevant_tissues=query.relevant_tissues)
                for record in allowed
            ]
            response_value = {
                "call_id": call_id,
                "status": ("blocked" if hard_block else "success" if allowed else "no_data"),
                "records": [] if hard_block else [record.to_dict() for record in projected],
                "policy_blocks": list(blocked),
                "raw_response_sha256": sha256_json(raw_response_value),
            }
            response_pointer = (
                self.artifacts.write_json(f"tool_calls/{call_id}/response.json", response_value)
                if self.artifacts
                else None
            )
            status = (
                ToolStatus.BLOCKED
                if hard_block
                else ToolStatus.SUCCESS
                if allowed
                else ToolStatus.NO_DATA
            )
            self.audit.log(
                "tool_call_finished",
                call_id=call_id,
                status=status.value,
                raw_records=len(raw),
                allowed_records=len(allowed),
                policy_blocks=list(blocked),
                response_sha256=sha256_json(response_value),
            )
            return ToolCallResult(
                call_id=call_id,
                status=status,
                requested_kinds=requested,
                records=() if hard_block else tuple(projected),
                policy_blocks=blocked,
                request_artifact=request_pointer,
                response_artifact=response_pointer,
                response_sha256=sha256_json(response_value),
                raw_response_artifact=raw_response_pointer,
                raw_response_sha256=sha256_json(raw_response_value),
                resolved_query=query,
            )
        except PolicyViolation as exc:
            self.audit.log("tool_call_blocked", call_id=call_id, reason=str(exc))
            return ToolCallResult(
                call_id=call_id,
                status=ToolStatus.BLOCKED,
                requested_kinds=requested,
                records=(),
                policy_blocks=(str(exc),),
                request_artifact=request_pointer,
                response_artifact=None,
                response_sha256=sha256_json([]),
                error=str(exc),
                resolved_query=query,
            )
        except Exception as exc:  # fail closed at the boundary
            self.audit.log("tool_call_error", call_id=call_id, error=type(exc).__name__)
            return ToolCallResult(
                call_id=call_id,
                status=ToolStatus.ERROR,
                requested_kinds=requested,
                records=(),
                policy_blocks=(),
                request_artifact=request_pointer,
                response_artifact=None,
                response_sha256=sha256_json([]),
                error=f"Evidence retrieval failed ({type(exc).__name__}).",
                resolved_query=query,
            )
