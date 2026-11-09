from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace
from typing import Any, ClassVar
from urllib.parse import urlsplit

from .audit import canonical_json, sha256_json
from .configuration import parameter
from .model_generation import (
    GenerationAttempts,
    GenerationFailure,
    classify_generation_failure,
    provider_context_recovery,
    provider_output_limit,
)
from .model_options import (
    DEFAULT_REASONING_EFFORT,
    is_deepseek_model_endpoint,
    model_token_budget,
    normalize_reasoning_effort,
    normalize_thinking_mode,
)
from .model_stream import CompletionStream, CompletionStreamError, bounded_token_usage
from .models import (
    CandidateDecision,
    EvidenceKind,
    EvidenceRecord,
    LocusQuery,
    RankedGene,
    analysis_variant_ids,
)
from .progress import ProgressSink, emit_progress
from .projection import (
    SynthesisProjectionError,
    build_synthesis_evidence_projection,
    scientific_scoring_records,
)
from .prompting import format_prompt, load_prompt
from .provider_catalog import infer_provider_id, model_for_endpoint
from .provider_reasoning import reasoning_for_endpoint, reasoning_request_body
from .ranking_budget import RankingBudget, RankingBudgetPolicy
from .sufficiency import decision_evidence_for_query, is_direct_locus_gene_evidence


class ModelIdentityError(RuntimeError):
    pass


class SynthesisOutputError(ValueError):
    """A structured ranking failed validation.

    ``validation_attempts`` lets the orchestrator distinguish a recoverable
    first malformed draft from two independently invalid responses.
    """

    def __init__(self, message: str, *, validation_attempts: int = 1):
        super().__init__(message)
        self.validation_attempts = validation_attempts


MAX_SYNTHESIS_REPAIR_ERROR_CHARS = parameter("model.ranking_repair_error_chars")
MAX_SYNTHESIS_REPAIR_DRAFT_CHARS = parameter("model.ranking_repair_draft_chars")


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    requested_model_id: str
    observed_model_id: str | None = None
    request_id: str | None = None
    endpoint: str | None = None
    offline: bool = False

    @property
    def effective_model_id(self) -> str:
        return self.observed_model_id or self.requested_model_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CitedClaim:
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SynthesisOutput:
    ranked_genes: tuple[RankedGene, ...]
    claims: tuple[CitedClaim, ...]
    summary: str
    identity: ModelIdentity
    raw_response_sha256: str
    usage: dict[str, Any] = field(default_factory=dict)
    model_visible_evidence_ids: tuple[str, ...] | None = None
    evidence_projection: dict[str, Any] = field(default_factory=dict)
    model_input_sha256: str = ""
    # The actual frozen alias map; None denotes historical canonical numbering.
    model_evidence_aliases: dict[str, str] | None = None
    # Runtime-bound input candidates, never inferred from the model's reply.
    # None preserves the contract for historical and non-SDK adapters.
    model_candidate_genes: tuple[str, ...] | None = None
    candidate_decisions: tuple[CandidateDecision, ...] = ()
    model_shortlisting_required: bool = False
    model_full_pool_ranking_required: bool = False


class EvidenceSynthesizer(ABC):
    @property
    @abstractmethod
    def identity(self) -> ModelIdentity:
        raise NotImplementedError

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    async def synthesize(
        self,
        query: LocusQuery,
        evidence: tuple[EvidenceRecord, ...],
        *,
        system_prompt: str,
    ) -> SynthesisOutput:
        raise NotImplementedError

    async def repair_validation_failure(
        self,
        query: LocusQuery,
        evidence: tuple[EvidenceRecord, ...],
        output: SynthesisOutput,
        validation_errors: tuple[str, ...],
        *,
        system_prompt: str,
    ) -> SynthesisOutput | None:
        """Optionally repair one parsed output that failed grounding validation."""

        del query, evidence, output, validation_errors, system_prompt
        return None


class DeterministicEvidenceSynthesizer(EvidenceSynthesizer):
    """A non-LLM ablation using the exact same normalized evidence ledger."""

    WEIGHTS: ClassVar[dict[EvidenceKind, float]] = {
        EvidenceKind.COLOCALIZATION: parameter("ranking.deterministic_weight_colocalization"),
        EvidenceKind.QTL: parameter("ranking.deterministic_weight_qtl"),
        EvidenceKind.REGULATORY: parameter("ranking.deterministic_weight_regulatory"),
        EvidenceKind.GWAS: parameter("ranking.deterministic_weight_gwas"),
        EvidenceKind.EXPRESSION: parameter("ranking.deterministic_weight_expression"),
        EvidenceKind.FUNCTION: parameter("ranking.deterministic_weight_function"),
        EvidenceKind.LITERATURE: parameter("ranking.deterministic_weight_literature"),
        # L2G is candidate-discovery metadata only.  ``decision_evidence_for_query``
        # removes it before scoring; the zero value is an additional fail-closed
        # guard if a future caller bypasses that boundary.
        EvidenceKind.L2G: 0.0,
        EvidenceKind.VARIANT_ANNOTATION: 0.0,
        EvidenceKind.STUDY: 0.0,
        EvidenceKind.PATHWAY: 0.0,
        EvidenceKind.ONTOLOGY: 0.0,
        EvidenceKind.TARGET_DISEASE: 0.0,
        EvidenceKind.DRUG: 0.0,
        EvidenceKind.SAFETY: 0.0,
        EvidenceKind.TRACTABILITY: 0.0,
        EvidenceKind.INTERACTION: 0.0,
        EvidenceKind.PHENOTYPE: 0.0,
        EvidenceKind.GENE_SET: 0.0,
    }

    def __init__(self) -> None:
        self._identity = ModelIdentity(
            provider="deterministic",
            requested_model_id="evidence-count-ranker-v1",
            observed_model_id="evidence-count-ranker-v1",
            offline=True,
        )

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    async def synthesize(
        self,
        query: LocusQuery,
        evidence: tuple[EvidenceRecord, ...],
        *,
        system_prompt: str,
    ) -> SynthesisOutput:
        del system_prompt
        gene_universe = _candidate_gene_universe(query, evidence)
        inference_evidence = decision_evidence_for_query(query, evidence)
        by_gene: dict[str, list[EvidenceRecord]] = {gene: [] for gene in gene_universe}
        for record in inference_evidence:
            if record.gene_id in by_gene:
                by_gene[record.gene_id].append(record)

        # All permitted records remain available for attribution. Repeated hits
        # on one scientific assertion, including selected/proxy duplicates, must
        # not multiply its weight or the tie-breaking count.
        scoring_by_gene = {
            gene: scientific_scoring_records(records, query=query)
            for gene, records in by_gene.items()
        }

        def score(gene: str) -> tuple[float, int, str]:
            records = scoring_by_gene[gene]
            total = 0.0
            for record in records:
                direction = -1.0 if record.supports is False else 1.0
                total += self.WEIGHTS[record.kind] * direction
            return (-total, -len(records), gene)

        ordered = sorted(gene_universe, key=score)
        if not query.benchmark:
            # Target profiles are biological context, not a locus-to-gene link. A
            # shortlisted gene must have its own QTL, colocalisation, or regulatory
            # record before expression/function/pathway/literature can inform rank.
            ordered = [
                gene
                for gene in ordered
                if any(is_direct_locus_gene_evidence(record) for record in by_gene[gene])
            ]
        ranked: list[RankedGene] = []
        claims: list[CitedClaim] = []
        for rank, gene in enumerate(ordered, start=1):
            records = by_gene[gene]
            ids = tuple(sorted(record.evidence_id for record in records))
            if ids:
                rationale = (
                    f"Assessed using {len(scoring_by_gene[gene])} distinct scientific evidence "
                    f"group(s) from {len(ids)} permitted record(s)."
                )
                claims.append(
                    CitedClaim(text=f"Evidence was retrieved for {gene}.", evidence_ids=ids)
                )
                limitations: tuple[str, ...] = ()
            else:
                rationale = (
                    "No gene-linked record was retrieved; retained to preserve the "
                    "candidate universe."
                )
                limitations = ("No permitted gene-linked evidence was retrieved.",)
            ranked.append(
                RankedGene(
                    gene_id=gene,
                    rank=rank,
                    evidence_ids=ids,
                    rationale=rationale,
                    limitations=limitations,
                )
            )
        raw = {
            "ranked_genes": [asdict(item) for item in ranked],
            "claims": [asdict(item) for item in claims],
        }
        return SynthesisOutput(
            ranked_genes=tuple(ranked),
            claims=tuple(claims),
            summary="Deterministic evidence-ranking ablation completed.",
            identity=self.identity,
            raw_response_sha256=sha256_json(raw),
            usage={"scientific_grouping": {
                gene: {"retrieved_records": len(by_gene[gene]),
                       "scoring_groups": len(scoring_by_gene[gene]),
                       "weighted_group_score": -score(gene)[0]}
                for gene in gene_universe
            }},
        )


class OpenAICompatibleSynthesizer(EvidenceSynthesizer):
    """Strict JSON adapter for hosted or local OpenAI-compatible model endpoints."""

    def __init__(
        self,
        *,
        model_id: str,
        provider: str,
        api_key_env: str,
        base_url: str | None = None,
        temperature: float = parameter("model.ranking_temperature"),
        seed: int | None = parameter("model.ranking_seed"),
        timeout_seconds: float | None = None,
        max_retries: int = parameter("model.sdk_max_retries"),
        require_exact_model_id: bool = True,
        offline: bool = False,
        thinking_mode: str = parameter("model.ranking_thinking_mode"),
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        reasoning_profile: Mapping[str, Any] | None = None,
    ):
        self.model_id = model_id
        self.provider = provider
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.temperature = temperature
        self.seed = seed
        self.timeout_seconds = (
            parameter("model.deepseek_timeout_seconds")
            if timeout_seconds is None
            and is_deepseek_model_endpoint(model_id=model_id, base_url=base_url)
            else parameter("model.compatible_timeout_seconds")
            if timeout_seconds is None
            else timeout_seconds
        )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self.max_retries = max_retries
        self.require_exact_model_id = require_exact_model_id
        self.offline = offline
        self.thinking_mode = normalize_thinking_mode(thinking_mode)
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        if reasoning_profile is not None and not isinstance(reasoning_profile, Mapping):
            raise ValueError("reasoning_profile must be a mapping")
        self.reasoning_profile = deepcopy(dict(reasoning_profile)) if reasoning_profile is not None else None
        self.progress_sink: ProgressSink | None = None
        self._stream_output_tokens = 0
        self._stream_received_chars = 0
        self._stream_usage_exact = True
        self.generation_audit_sink = None
        self.model_fact_view_sink = None
        self.ranking_retrieval_statuses = {}
        self.v2g_cards = None
        self._ranking_input_snapshot = None
        self._ranking_snapshot_fingerprint = None
        self._ranking_snapshot_pending = False
        self.model_output_sink = None
        self._visible_output_receipts: dict[int, dict[str, Any]] = {}
        self.cancel_event = None
        self._generation_attempts = GenerationAttempts()
        self._ranking_budget: RankingBudget | None = None
        self._generation_purpose = "initial"
        self._provider_output_limit = None
        self._context_retry_builder = None
        self._active_projection = None
        self._active_context_calibration = None
        self._evidence_aliases: dict[str, str] = {}
        self._model_citation_records: dict[str, dict[str, Any]] = {}
        self._full_model_citation_records: dict[str, dict[str, Any]] = {}
        self._preserve_scientific_input = False
        self._ranking_input_active = False
        self._provider_limit_basis = None
        self._capacity_bound_messages_sha256 = None
        reasoning_request_body(
            model_id=self.model_id, base_url=self.base_url or "",
            thinking_mode=self.thinking_mode, reasoning_effort=self.reasoning_effort,
            profile=self.reasoning_profile,
        )
        self._identity = ModelIdentity(
            provider=provider,
            requested_model_id=model_id,
            endpoint=base_url,
            offline=offline,
        )

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def _completion_budget(self) -> int | None:
        catalog_model = model_for_endpoint(self.base_url or "", self.model_id)
        if catalog_model is not None and catalog_model.max_output_tokens is not None:
            return catalog_model.max_output_tokens
        return None

    @property
    def generation_attempts(self) -> list[dict[str, Any]]:
        return self._generation_attempts.public()

    def mark_generation_validation(self, *, passed: bool, category: str = "grounding",
                                   issues: tuple[str, ...] = ()) -> None:
        self._generation_attempts.validation(passed=passed, category=category)
        if self._generation_attempts.rows:
            self._record_visible_output_review(self._generation_attempts.rows[-1], passed=passed,
                                               category=category, issues=issues)

    def _capture_visible_output(self, content: Any, row: Mapping[str, Any], *, finish_reason: Any = None) -> None:
        """Archive only assistant answer text; never provider reasoning or headers."""
        if self.model_output_sink is None or not isinstance(content, str):
            return
        attempt = row["attempt"]
        if attempt in self._visible_output_receipts:
            return
        receipt = {"attempt": attempt, "purpose": row["purpose"], "model_id": self.model_id,
            "model_input_sha256": row.get("model_input_sha256"),
            "message_input_sha256": row.get("token_budget", {}).get("message_input_sha256"),
            "scientific_payload_sha256": (self._active_projection or {}).get("model_evidence_payload_sha256"),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "finish_reason": finish_reason if finish_reason in {None, "stop", "length", "content_filter", "tool_calls", "function_call"} else "unrecognized"}
        try:
            self.model_output_sink({"format": "model_visible_output_v1", "event": "visible_output",
                "validation_status": "unvalidated", **receipt, "content": content})
        except Exception as exc:
            raise GenerationFailure("The model answer could not be archived.", category="output_archive_failed") from exc
        self._visible_output_receipts[attempt] = {**receipt, "review_count": 0}

    def _record_visible_output_review(self, row: Mapping[str, Any], *, passed: bool,
                                      category: str, issues: tuple[str, ...]) -> None:
        receipt = self._visible_output_receipts.get(row["attempt"])
        if self.model_output_sink is None or receipt is None:
            return
        receipt["review_count"] += 1
        value = {key: item for key, item in receipt.items() if key != "review_count"}
        try:
            self.model_output_sink({"format": "model_output_validation_v1", "event": "output_validation", **value,
                "review_sequence": receipt["review_count"], "validation_passed": passed,
                "validation_category": category, "validation_issues": list(issues)})
        except Exception as exc:
            raise GenerationFailure("The model answer review could not be archived.", category="output_archive_failed") from exc

    @property
    def parameters(self) -> dict[str, Any]:
        # Configuration describes the initial scientific request, even after a
        # repair has used its own endpoint-verified reasoning controls.
        sampling = self._sampling_options(thinking_mode=self.thinking_mode)
        return {
            "temperature": sampling.get("temperature"),
            "seed": sampling.get("seed"),
            "sampling_omission_reason": ("Temperature and seed were not sent: unsupported in the active provider reasoning mode."
                                         if not sampling else None),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": 0,
            "maximum_generation_attempts": self._generation_attempts.maximum,
            "require_exact_model_id": self.require_exact_model_id,
            "thinking_mode": self.thinking_mode,
            "reasoning_effort": self.reasoning_effort,
            "provider_max_tokens": self._completion_budget(),
            **({"ranking_resource_policy": self._ranking_budget.policy.public()}
               if self._ranking_budget is not None else {}),
            **({"reasoning_profile": deepcopy(self.reasoning_profile)} if self.reasoning_profile is not None else {}),
        }

    def _effective_reasoning_controls(self) -> tuple[str, str]:
        """Use focused repair controls only when the endpoint supports them."""
        configured = (self.thinking_mode, self.reasoning_effort)
        if self._generation_purpose not in {"json_structure_repair", "grounding_repair"}:
            return configured
        capability = reasoning_for_endpoint(self.base_url or "", self.model_id, self.reasoning_profile)
        if capability["kind"] in {"unknown", "fixed", "none"}:
            return configured
        if capability["can_disable"]:
            return "disabled", DEFAULT_REASONING_EFFORT
        supported = capability["efforts"]
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
            if effort in supported:
                return "enabled", effort
        return configured

    def _sampling_options(self, *, thinking_mode: str | None = None) -> dict[str, Any]:
        if thinking_mode is None:
            thinking_mode, _ = self._effective_reasoning_controls()
        catalog_model = model_for_endpoint(self.base_url or "", self.model_id)
        capability = reasoning_for_endpoint(self.base_url or "", self.model_id, self.reasoning_profile)
        deepseek = is_deepseek_model_endpoint(model_id=self.model_id, base_url=self.base_url)
        if deepseek:
            return {"temperature": self.temperature, "seed": self.seed} if thinking_mode == "disabled" else {}
        native_reasoning = bool(catalog_model and catalog_model.native_reasoning)
        active_controls = capability["kind"] in {"toggle_effort", "effort", "toggle", "fixed"} and thinking_mode != "disabled"
        if native_reasoning or active_controls:
            return {}
        result: dict[str, Any] = {"temperature": self.temperature}
        hostname = (urlsplit(self.base_url or "").hostname or "").casefold().rstrip(".")
        if hostname == "api.openai.com" or hostname.endswith(".openai.com"):
            result["seed"] = self.seed
        return result

    def _request_options(self) -> dict[str, Any]:
        request_options: dict[str, Any] = {}
        hostname = (urlsplit(self.base_url or "").hostname or "").casefold().rstrip(".")
        official_json_mode = hostname.endswith(
            ("api.deepseek.com", ".deepseek.com", "api.openai.com", ".openai.com")
        )
        if official_json_mode:
            request_options["response_format"] = {"type": "json_object"}
        provider_max_tokens = self._completion_budget()
        thinking_mode, reasoning_effort = self._effective_reasoning_controls()
        thinking_body = reasoning_request_body(
            model_id=self.model_id,
            base_url=self.base_url or "",
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            profile=self.reasoning_profile,
        )
        if "reasoning_effort" in thinking_body:
            request_options["reasoning_effort"] = thinking_body.pop("reasoning_effort")
        if thinking_body:
            request_options["extra_body"] = thinking_body
        if provider_max_tokens is None:
            pass  # Unknown models use their actual provider default.
        elif hostname == "api.openai.com" or hostname.endswith(".openai.com"):
            request_options["max_completion_tokens"] = provider_max_tokens
        else:
            request_options["max_tokens"] = provider_max_tokens
        request_options.update(self._sampling_options(thinking_mode=thinking_mode))
        return request_options

    def prepare_ranking_input(self, query, evidence, *, system_prompt, presentation_seed=None):
        """Freeze before credentials/SDK use; retries retain this exact input."""
        from .ranking_input import (
            RankingInputSnapshot,
            candidate_presentation,
            clean_and_order_model_payload,
            rebind_presentation_aliases,
        )

        fingerprint = sha256_json({"query": asdict(query),
            "evidence": [row.to_dict() for row in evidence], "system_prompt": system_prompt,
            "model_id": self.model_id, "base_url": self.base_url,
            "retrieval_statuses": self.ranking_retrieval_statuses})
        if (self._ranking_snapshot_pending and self._ranking_snapshot_fingerprint == fingerprint
                and self._ranking_input_snapshot is not None and presentation_seed is None):
            return self._ranking_input_snapshot
        candidates = _candidate_gene_universe(query, evidence)
        inference = decision_evidence_for_query(query, evidence)
        payload, messages, view = _natural_ranking_request(query, inference,
            candidate_genes=candidates, system_prompt=system_prompt,
            model_id=self.model_id, base_url=self.base_url)
        comparison = None
        v2g_view = None
        if query.task == "prioritize_genes" and self.v2g_cards:
            from .v2g.compact_input import validate_card_source_bindings

            source_bindings = validate_card_source_bindings(self.v2g_cards, inference)
            payload, messages, v2g_view = _v2g_card_ranking_request(query, payload, view,
                v2g_cards=self.v2g_cards, system_prompt=system_prompt, model_id=self.model_id,
                base_url=self.base_url)
            v2g_view.aggregation_receipt["compact_input"]["source_bindings"] = source_bindings
            self._full_model_citation_records = {}
        elif query.task == "prioritize_genes" and parameter("ranking.model_evidence_view") == "gene_comparison":
            payload, messages, comparison = _gene_comparison_ranking_request(query, payload, view,
                system_prompt=system_prompt, model_id=self.model_id, base_url=self.base_url,
                retrieval_statuses=self.ranking_retrieval_statuses)
            self._full_model_citation_records = {
                row["evidence_id"]: deepcopy(row)
                for row in comparison.payload["citation_records"]
            }
        visible = v2g_view or comparison or view
        retained = {row.evidence_id for row in inference}
        excluded = [row.evidence_id for row in evidence if row.evidence_id not in retained]
        decision_scope = {"retained_ledger_records": len(evidence), "decision_eligible_records": len(inference),
            "context_or_discovery_records_excluded": len(excluded),
            "excluded_evidence_ids_sha256": sha256_json(sorted(excluded)),
            "policy": "task-context-and-shortlist-v1"}
        from .ranking_input import presentation_fingerprint
        presentation = candidate_presentation(candidates, seed=presentation_seed,
            input_fingerprint=presentation_fingerprint(query))
        payload = clean_and_order_model_payload(payload, presentation)
        payload, aliases, alias_mapping = rebind_presentation_aliases(payload, visible.evidence_aliases)
        # Alias integrity is retained in the local audit snapshot/receipt, not
        # in the scientific content sent to the model.
        payload["evidence_projection"].pop("evidence_aliases_sha256", None)
        payload["locus"]["input_order_policy"] = (
            "Presentation order only; it is not a scientific ranking signal."
        )
        payload_validation = {}
        if comparison is not None:
            from .ranking_payload_codec import validate_model_gene_comparison

            payload_validation = validate_model_gene_comparison(
                payload, comparison.payload, alias_mapping=alias_mapping)
        elif v2g_view is not None:
            from .v2g.compact_input import validate_v2g_payload

            payload_validation = validate_v2g_payload(
                payload, v2g_view.payload, alias_mapping=alias_mapping)
            payload_validation.update(v2g_view.aggregation_receipt["compact_input"]["source_bindings"])
        messages[1]["content"] = format_prompt("agent_user", payload=canonical_json(payload))
        if v2g_view is not None:
            capacity = model_token_budget(base_url=self.base_url, model_id=self.model_id, messages=messages,
                task_max_output_tokens=parameter("model.ranking_request_output_tokens"))
            v2g_view.aggregation_receipt["compact_input"]["request_capacity"] = capacity
            limit = capacity["input_token_budget"] or capacity["context_window"]
            if limit is not None and capacity["prompt_tokens_estimate"] >= limit:
                raise SynthesisOutputError(
                    "The complete compact V2G evidence exceeds the known model input capacity "
                    f"({capacity['prompt_tokens_estimate']} estimated tokens; limit {limit}). "
                    "No evidence was truncated and no ranking request was sent. "
                    "Original evidence remains in the analysis artifacts; use a model with a larger context.")
        from .numeric_provenance import build_numeric_ledger
        numeric_ledger = build_numeric_ledger(query, evidence)
        audit = {"format": "scientific_fact_bindings_v1", "schema_version": 2,
            "numeric_fact_ledger": numeric_ledger,
            "input_schema_version": payload["input_schema_version"],
            "model_input_sha256": sha256_json(payload), "model_payload": payload,
            "messages": messages, "messages_sha256": sha256_json(messages),
            "presentation": presentation, "presentation_alias_map": alias_mapping,
            "evidence_aliases": aliases, "record_bindings": list(view.record_bindings),
            "full_scientific_facts": view.payload, "full_evidence_aliases": dict(view.evidence_aliases),
            "aggregation_receipt": (v2g_view or comparison).aggregation_receipt if (v2g_view or comparison) else {},
            "model_payload_validation": payload_validation,
            "audit_only_decision_scope": decision_scope,
            "scientific_input_fingerprint": fingerprint}
        snapshot = RankingInputSnapshot.from_audit(audit)
        self._ranking_input_snapshot = snapshot
        self._ranking_snapshot_fingerprint = fingerprint
        self._ranking_snapshot_pending = True
        return snapshot

    async def synthesize(
        self,
        query: LocusQuery,
        evidence: tuple[EvidenceRecord, ...],
        *,
        system_prompt: str,
    ) -> SynthesisOutput:
        # One fresh allowance per decision; scientific repair below shares it.
        self._ranking_budget = (RankingBudget(RankingBudgetPolicy.for_model(self.base_url, self.model_id))
                                if query.task == "prioritize_genes" else None)
        self._generation_attempts = GenerationAttempts(sink=self.generation_audit_sink,
                                                       resource_budget=self._ranking_budget)
        self._visible_output_receipts = {}
        self._generation_purpose = "initial"
        self._provider_output_limit = None
        self._context_retry_builder = None
        self._active_projection = None
        self._active_context_calibration = None
        self._evidence_aliases = {}
        self._model_citation_records = {}
        self._full_model_citation_records = {}
        self._preserve_scientific_input = False
        self._ranking_input_active = False
        self._provider_limit_basis = None
        self._capacity_bound_messages_sha256 = None
        snapshot = self.prepare_ranking_input(query, evidence, system_prompt=system_prompt)
        frozen = snapshot.to_audit_payload()
        payload, messages = frozen["model_payload"], frozen["messages"]
        candidate_universe = tuple(frozen["presentation"]["candidate_order"])
        self._evidence_aliases = dict(frozen["evidence_aliases"])
        comparison_payload = payload.get("gene_comparison")
        shortlisting_required = bool(comparison_payload and
                                    comparison_payload.get("format") == "gwasmind_v2g_gene_cards_v1")
        self._model_citation_records = {}
        if comparison_payload is not None and "citation_records" in comparison_payload:
            inverse_aliases = {original: alias for alias, original in self._evidence_aliases.items()}
            full_records = {
                row["evidence_id"]: _translate_evidence_references(deepcopy(row), inverse_aliases)
                for row in (self._full_model_citation_records.values()
                            if self._full_model_citation_records else comparison_payload["citation_records"])
            }
            self._model_citation_records = {}
            for compact in comparison_payload["citation_records"]:
                original_id = self._evidence_aliases.get(compact["evidence_id"], compact["evidence_id"])
                translated = full_records.get(original_id, {})
                merged = {**translated, **deepcopy(compact)}
                for key, value in translated.items():
                    merged.setdefault(key, value)
                merged["source"] = deepcopy(comparison_payload["sources"].get(compact.get("source_ref"), {}))
                self._model_citation_records[original_id] = merged
        self._preserve_scientific_input = True
        self._ranking_input_active = True
        self._active_projection = {**_projection_attempt_audit(payload),
            "frozen_messages_sha256": frozen["messages_sha256"],
            "presentation_seed": frozen["presentation"]["seed"],
            "candidate_order_sha256": frozen["presentation"]["candidate_order_sha256"]}
        if self.model_fact_view_sink is not None:
            self.model_fact_view_sink(snapshot.to_audit_payload())
        model_evidence = [{"evidence_id": identifier} for identifier in sorted(set(self._evidence_aliases.values()))]
        projection_summary = payload["evidence_projection"]
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('Install model support with: pip install -e ".[llm]"') from exc

        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"missing credential environment variable: {self.api_key_env}")
        async with _closing_model_client(AsyncOpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        ), resource_budget=self._ranking_budget) as client:
            self._stream_output_tokens = 0
            self._stream_received_chars = 0
            self._stream_usage_exact = True
            request_options = self._request_options()
            response = await self._create_completion(client,
                model=self.model_id,
                messages=messages,
                **request_options,
            )
            self._context_retry_builder = None
            self._ranking_input_active = False
            identity, content, usage = self._completion_parts(response)
            try:
                parsed = _parse_synthesis_content(content, identity=identity, usage=usage)
                parsed = replace(parsed, model_full_pool_ranking_required=shortlisting_required)
                _validate_model_selection(parsed, candidate_universe, required=shortlisting_required)
            except SynthesisOutputError as draft_error:
                self.mark_generation_validation(passed=False, category="json_structure", issues=(str(draft_error),))
                if not self._generation_attempts.remaining:
                    raise
                self._generation_purpose = "json_structure_repair"
                initial_usage = usage
                repair_messages = _synthesis_repair_messages(
                    validation_error=draft_error,
                    malformed_draft=content,
                    allowed_genes=candidate_universe,
                    model_visible_evidence_ids=tuple(self._evidence_aliases),
                )
                repair_messages[0]["content"] = _ranking_system_prompt(repair_messages[0]["content"], query)
                # Provider/transport failures are not malformed model output.  Let
                # them propagate unchanged so the caller can report a service
                # failure instead of mislabelling it as a response-validation
                # failure.
                repaired_response = await self._create_completion(client,
                    model=self.model_id,
                    messages=repair_messages,
                    **self._request_options(),
                )
                try:
                    identity, repaired_content, repair_usage = self._completion_parts(
                        repaired_response
                    )
                    usage = _combined_usage(initial_usage, repair_usage)
                    usage["structure_repair_attempts"] = 1
                    parsed = _parse_synthesis_content(
                        repaired_content,
                        identity=identity,
                        usage=usage,
                    )
                    parsed = replace(parsed, model_full_pool_ranking_required=shortlisting_required)
                    _validate_model_selection(parsed, candidate_universe, required=shortlisting_required)
                    _validate_repaired_identifiers(
                        _resolve_evidence_aliases(parsed, self._evidence_aliases),
                        malformed_draft=_draft_aliases_to_original(content, self._evidence_aliases),
                    )
                except ModelIdentityError:
                    raise
                except SynthesisOutputError as repair_error:
                    self.mark_generation_validation(passed=False, category="json_structure", issues=(str(repair_error),))
                    initial_detail = str(draft_error)[:MAX_SYNTHESIS_REPAIR_ERROR_CHARS]
                    repair_detail = str(repair_error)[:MAX_SYNTHESIS_REPAIR_ERROR_CHARS]
                    raise SynthesisOutputError(
                        "model output repair failed after one attempt "
                        f"(initial: {initial_detail}; repair: {repair_detail})",
                        validation_attempts=2,
                    ) from None
            self._ranking_snapshot_pending = False
            return replace(
                _resolve_evidence_aliases(parsed, self._evidence_aliases),
                model_visible_evidence_ids=tuple(
                    str(record["evidence_id"]) for record in model_evidence
                ),
                evidence_projection=projection_summary,
                model_input_sha256=sha256_json(payload),
                model_evidence_aliases=dict(self._evidence_aliases),
                model_candidate_genes=candidate_universe,
                model_shortlisting_required=shortlisting_required,
                model_full_pool_ranking_required=shortlisting_required,
            )

    async def repair_validation_failure(
        self,
        query: LocusQuery,
        evidence: tuple[EvidenceRecord, ...],
        output: SynthesisOutput,
        validation_errors: tuple[str, ...],
        *,
        system_prompt: str,
    ) -> SynthesisOutput | None:
        """Run one bounded repair for a parsed but deterministically invalid ranking."""

        del system_prompt
        if query.task == "prioritize_genes" and self._ranking_budget is None:
            self._ranking_budget = RankingBudget(RankingBudgetPolicy.for_model(self.base_url, self.model_id))
            self._generation_attempts.resource_budget = self._ranking_budget
        self.mark_generation_validation(passed=False, category="grounding", issues=validation_errors)
        if not self._generation_attempts.remaining:
            return None
        self._generation_purpose = "grounding_repair"
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('Install model support with: pip install -e ".[llm]"') from exc

        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"missing credential environment variable: {self.api_key_env}")
        candidate_universe = (output.model_candidate_genes if output.model_candidate_genes is not None
                              else _candidate_gene_universe(query, evidence))
        visible_ids = tuple(output.model_visible_evidence_ids or ())
        inverse_aliases = {identifier: alias for alias, identifier in self._evidence_aliases.items()}
        draft = canonical_json(_translate_evidence_references(_synthesis_output_payload(output), inverse_aliases))
        cited_ids = {identifier for row in (*output.ranked_genes, *output.claims, *output.candidate_decisions)
                     for identifier in row.evidence_ids}
        messages = _synthesis_validation_repair_messages(
            validation_errors=validation_errors,
            invalid_draft=draft,
            allowed_genes=candidate_universe,
            model_visible_evidence_ids=tuple(inverse_aliases.get(identifier, identifier) for identifier in visible_ids),
            citation_bindings=_translate_evidence_references(_citation_bindings(
                evidence, cited_ids.intersection(visible_ids), shown_records=self._model_citation_records), inverse_aliases),
        )
        messages[0]["content"] = _ranking_system_prompt(messages[0]["content"], query)
        async with _closing_model_client(AsyncOpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        ), resource_budget=self._ranking_budget) as client:
            response = await self._create_completion(client,
                model=self.model_id,
                messages=messages,
                **self._request_options(),
            )
            identity, content, repair_usage = self._completion_parts(response)
            try:
                parsed = _parse_synthesis_content(content, identity=identity, usage=repair_usage)
                _validate_repaired_identifiers(_resolve_evidence_aliases(parsed, self._evidence_aliases),
                    malformed_draft=_draft_aliases_to_original(draft, self._evidence_aliases))
                from .ranking_claims import assert_ranking_unchanged

                try:
                    assert_ranking_unchanged(output, parsed)
                except ValueError as exc:
                    raise SynthesisOutputError(str(exc)) from exc
            except SynthesisOutputError as exc:
                self.mark_generation_validation(passed=False, category="json_structure", issues=(str(exc),))
                raise
            return replace(
                _resolve_evidence_aliases(parsed, self._evidence_aliases),
                model_visible_evidence_ids=visible_ids,
                evidence_projection=output.evidence_projection,
                model_input_sha256=output.model_input_sha256,
                model_evidence_aliases=dict(self._evidence_aliases),
                model_candidate_genes=output.model_candidate_genes,
                model_shortlisting_required=output.model_shortlisting_required,
                model_full_pool_ranking_required=output.model_full_pool_ranking_required,
                usage=_merged_usage(output.usage, repair_usage),
            )

    def _active_token_budget(self, messages: Any) -> dict[str, Any]:
        options = {}
        if self._ranking_budget is not None:
            options["task_max_output_tokens"] = self._ranking_budget.request_limit()
            if self._provider_limit_basis == "joint_context" and self._active_context_calibration:
                options["confirmed_context_remaining_tokens"] = max(0,
                    self._active_context_calibration["context_window"] - self._active_context_calibration["actual_prompt_tokens"])
        budget = model_token_budget(base_url=self.base_url, model_id=self.model_id, messages=messages, **options)
        if self._ranking_budget is not None:
            budget["ranking_task_budget"] = self._ranking_budget.snapshot()
        return budget

    async def _create_completion_with_deadline(self, client: Any, **kwargs: Any) -> Any:
        if self._ranking_budget is None:
            return await self._create_completion_once(client, **kwargs)
        # The allowance was reserved when this attempt began. Check time here,
        # not remaining tokens, because its own reservation may consume the rest.
        deadline = asyncio.timeout(self._ranking_budget.remaining_seconds)
        try:
            async with deadline:
                return await self._create_completion_once(client, **kwargs)
        except TimeoutError:
            if deadline.expired():
                raise GenerationFailure("The gene-ranking task reached its total wall-clock deadline; scientific input is retained.",
                    category="ranking_deadline_exhausted", retryable=False) from None
            raise  # An ordinary SDK/network timeout retains its own diagnostics.

    async def _create_completion(self, client: Any, **kwargs: Any) -> Any:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise asyncio.CancelledError()
        if (self._provider_limit_basis == "joint_context" and
                self._capacity_bound_messages_sha256 != sha256_json(kwargs.get("messages", []))):
            self._provider_output_limit = None
            self._provider_limit_basis = None
            self._active_context_calibration = None
        budget = self._active_token_budget(kwargs.get("messages"))
        token_key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
        if self._ranking_budget is not None and not ({"max_completion_tokens", "max_tokens"} & kwargs.keys()):
            hostname = (urlsplit(self.base_url or "").hostname or "").casefold().rstrip(".")
            if hostname == "api.openai.com" or hostname.endswith(".openai.com"):
                token_key = "max_completion_tokens"
        kwargs.pop("max_tokens", None)
        kwargs.pop("max_completion_tokens", None)
        if budget["effective_max_output_tokens"] is not None:
            kwargs[token_key] = budget["effective_max_output_tokens"]
        if self._provider_output_limit is not None:
            kwargs[token_key] = min(kwargs.get(token_key, self._provider_output_limit), self._provider_output_limit)
        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise asyncio.CancelledError()
            budget = self._active_token_budget(kwargs.get("messages"))
            if self._ranking_budget is not None:
                kwargs[token_key] = min(kwargs.get(token_key, budget["effective_max_output_tokens"]),
                                        budget["effective_max_output_tokens"])
            budget["effective_max_output_tokens"] = kwargs.get(token_key)
            options = {key: value for key, value in kwargs.items() if key not in {"messages", "model"}}
            # Preserve the actual wire controls, including retries that retain
            # their original request options. No prompt or credentials are logged.
            budget["request_controls"] = {key: deepcopy(options[key]) for key in
                ("reasoning_effort", "extra_body", "temperature", "seed") if key in options}
            budget["sampling_parameters"] = {"temperature": options.get("temperature"), "seed": options.get("seed"),
                "omission_reason": ("Unsupported in the active provider reasoning mode"
                                    if "temperature" not in options and "seed" not in options else None)}
            budget["message_input_sha256"] = sha256_json(kwargs.get("messages", []))
            if self._provider_output_limit is not None:
                budget["provider_reported_output_limit_tokens"] = self._provider_output_limit
                budget["budget_reason"] = "provider_joint_capacity_maximum_preserving_facts" if self._provider_limit_basis == "joint_context" else "explicit_provider_output_limit_correction"
            if self._active_context_calibration is not None:
                budget["provider_context_calibration"] = dict(self._active_context_calibration)
            row = self._generation_attempts.begin(model_id=self.model_id, provider=self.provider,
                messages=kwargs.get("messages", []), options=options, purpose=self._generation_purpose, token_budget=budget,
                projection=({**self._active_projection, "use_in_attempt": "model_input_ledger" if self._ranking_input_active or callable(self._context_retry_builder)
                             else "originating_ranking_projection_for_repair_bindings"} if self._active_projection is not None else None))
            try:
                if self._ranking_input_active:
                    expected_hash = (self._active_projection or {}).get("frozen_messages_sha256")
                    if not expected_hash or sha256_json(kwargs.get("messages", [])) != expected_hash:
                        raise GenerationFailure(
                            "The gene-ranking request changed after scientific validation; the provider request was blocked.",
                            category="ranking_input_integrity_failure", retryable=False)
                response = await self._create_completion_with_deadline(client, **kwargs)
                choices = getattr(response, "choices", None)
                if isinstance(choices, list) and choices:
                    self._capture_visible_output(getattr(getattr(choices[0], "message", None), "content", None), row,
                                                 finish_reason=getattr(choices[0], "finish_reason", None))
                # Validate model and finish_reason before deciding that an attempt
                # completed. In particular 8192 tokens + stop is successful.
                _, _, usage = self._completion_parts(response)
                reason = getattr(response.choices[0], "finish_reason", None)
                self._generation_attempts.finish(row, finish_reason=reason, usage=usage)
                if (self._ranking_budget is not None and
                        row.get("task_resource_usage", {}).get("provider_exceeded_requested_output")):
                    failure = GenerationFailure("The provider exceeded the requested gene-ranking output allowance.",
                        category="ranking_provider_budget_violation", retryable=False)
                    failure.usage = usage
                    raise failure
                return response
            except asyncio.CancelledError:
                self._generation_attempts.finish(row, failure=GenerationFailure("Generation cancelled.", category="cancelled"))
                raise
            except Exception as exc:
                usage = getattr(exc, "usage", {})
                failure_details = classify_generation_failure(exc)
                self._record_visible_output_review(row, passed=False, category=failure_details["category"],
                    issues=("The completion failed before scientific validation: " + failure_details["category"],))
                if (classify_generation_failure(exc)["category"] == "output_length"
                        and isinstance(usage, Mapping) and budget["context_window"] and budget["input_token_budget"]
                        and isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int)
                        and usage["prompt_tokens"] > budget["input_token_budget"]
                        and usage["prompt_tokens"] + usage["completion_tokens"] >= budget["context_window"]):
                    if self._preserve_scientific_input and parameter("model.preserve_scientific_input_on_capacity_feedback"):
                        available_output = budget["context_window"] - usage["prompt_tokens"]
                        failure = GenerationFailure("The provider's joint context was exhausted; scientific input is retained.",
                            category="context_exhausted" if available_output > 0 else "input_exceeds_model_context",
                            finish_reason="length", retryable=False)
                        # At the actual context boundary the same evidence and
                        # output allowance cannot create more generation space.
                        # Unlike a network retry or a provider capacity 400,
                        # repeating this request spends tokens without recovery.
                        row["token_budget"]["recovery_decision"] = "no_retry_identical_scientific_input_at_context_capacity"
                        self._generation_attempts.finish(row, failure=failure, usage=usage)
                        raise failure from None
                    failure = GenerationFailure("The provider's joint input/output context was exhausted.",
                        category="context_exhausted", finish_reason="length", retryable=callable(self._context_retry_builder))
                    failure.usage = bounded_token_usage(usage)
                    exc = failure
                    self._generation_attempts.finish(row, failure=exc)
                    if callable(self._context_retry_builder) and self._generation_attempts.remaining:
                        kwargs["messages"] = self._context_retry_builder(usage)
                        self._generation_purpose = "context_exhausted_reprojection"
                        continue
                    raise failure
                if self._ranking_budget is not None and failure_details["category"] == "output_length":
                    failure = GenerationFailure("The gene-ranking completion reached its output limit; the same scientific request will not be regenerated automatically.",
                        category="ranking_output_budget_exhausted", finish_reason="length", retryable=False)
                    failure.usage = usage
                    row["token_budget"]["recovery_decision"] = "no_retry_after_ranking_output_budget_exhaustion"
                    self._generation_attempts.finish(row, failure=failure, usage=usage)
                    raise failure from None
                status = getattr(exc, "status_code", None)
                context_recovery = provider_context_recovery(getattr(exc, "body", None),
                    context_window=budget["context_window"], requested_output_tokens=kwargs.get(token_key)) if status in {400, 422} else None
                if (context_recovery is not None and self._preserve_scientific_input
                        and parameter("model.preserve_scientific_input_on_capacity_feedback")):
                    available_output = context_recovery["context_window"] - context_recovery["prompt_tokens"]
                    corrected_output = min(budget["provider_max_output_tokens"], available_output,
                        kwargs.get(token_key, budget["provider_max_output_tokens"]))
                    can_retry = corrected_output > 0 and corrected_output < kwargs.get(token_key, 0) and bool(self._generation_attempts.remaining)
                    failure = GenerationFailure("The provider specified the usable joint capacity; scientific input is retained.",
                        category="context_capacity_rejected" if available_output > 0 else "input_exceeds_model_context",
                        http_status=status, retryable=can_retry, provider_error=getattr(exc, "body", None))
                    self._generation_attempts.finish(row, failure=failure)
                    if not can_retry:
                        raise failure from None
                    self._provider_output_limit = corrected_output
                    self._provider_limit_basis = "joint_context"
                    self._capacity_bound_messages_sha256 = sha256_json(kwargs.get("messages", []))
                    self._active_context_calibration = {"actual_prompt_tokens": context_recovery["prompt_tokens"],
                        "context_window": context_recovery["context_window"],
                        "actual_maximum_output_tokens": min(budget["provider_max_output_tokens"], available_output),
                        "effective_requested_output_tokens": corrected_output,
                        "scientific_input_preserved": True}
                    kwargs[token_key] = corrected_output
                    self._generation_purpose = "provider_joint_capacity_retry"
                    emit_progress(self.progress_sink, "synthesis_retry", attempt=len(self._generation_attempts.rows) + 1,
                        maximum_attempts=self._generation_attempts.maximum, reason="context_capacity_rejected")
                    continue
                if context_recovery is not None and callable(self._context_retry_builder):
                    can_retry = bool(self._generation_attempts.remaining)
                    failure = GenerationFailure("The provider rejected the joint input/output capacity.",
                        category="context_capacity_rejected", http_status=status, retryable=can_retry,
                        provider_error=getattr(exc, "body", None))
                    self._generation_attempts.finish(row, failure=failure)
                    if can_retry:
                        retry_messages = self._context_retry_builder(context_recovery)
                        retry_budget = model_token_budget(base_url=self.base_url, model_id=self.model_id, messages=retry_messages)
                        if retry_budget["prompt_utf8_bytes"] >= budget["prompt_utf8_bytes"]:
                            raise GenerationFailure("Capacity recovery would repeat an unchanged or larger request.",
                                category="context_reprojection_stalled", provider_error=getattr(exc, "body", None)) from None
                        kwargs["messages"] = retry_messages
                        self._generation_purpose = "provider_context_reprojection"
                        emit_progress(self.progress_sink, "synthesis_retry", attempt=len(self._generation_attempts.rows) + 1,
                            maximum_attempts=self._generation_attempts.maximum, reason="context_capacity_rejected")
                        continue
                    raise failure from None
                suggested = provider_output_limit(getattr(exc, "body", None)) if status in {400, 422} else None
                if (suggested is not None and self._provider_output_limit is None
                        and (kwargs.get(token_key) is None or suggested < kwargs[token_key])):
                    self._provider_output_limit = suggested
                    kwargs[token_key] = suggested
                    failure = GenerationFailure("The provider specified its output-token ceiling.",
                        category="provider_output_limit", http_status=status, retryable=True,
                        provider_error=getattr(exc, "body", None))
                    self._generation_attempts.finish(row, failure=failure)
                    if self._generation_attempts.remaining:
                        self._generation_purpose = "provider_output_limit_retry"
                        continue
                self._generation_attempts.finish(row, failure=exc)
                failure = classify_generation_failure(exc)
                if not failure["retryable"] or not self._generation_attempts.remaining:
                    if failure["category"] == "context_capacity_rejected":
                        raise GenerationFailure("The provider rejected the joint context without usable capacity feedback.",
                            category="context_capacity_rejected", http_status=failure["http_status"],
                            provider_error=getattr(exc, "body", None)) from None
                    raise
                if self._ranking_budget is not None:
                    try:
                        self._ranking_budget.require_available()
                    except GenerationFailure as resource_failure:
                        row["preceding_failure"] = failure
                        self._generation_attempts.finish(row, failure=resource_failure, usage=usage)
                        raise resource_failure from None
                self._generation_purpose = failure["category"] + "_retry"
                if failure["category"] == "output_length":
                    kwargs["messages"] = [*kwargs["messages"], {"role": "user", "content":
                        "The previous response reached the provider output limit. Regenerate the complete required JSON concisely, preserving every ranked gene, evidence reference and uncertainty; omit repetitive prose. Do not change the ranking task or evidence."}]
                emit_progress(self.progress_sink, "synthesis_retry", attempt=len(self._generation_attempts.rows) + 1,
                              maximum_attempts=self._generation_attempts.maximum, reason=failure["category"])
                await asyncio.sleep(min(0.25, self._ranking_budget.remaining_seconds)
                                    if self._ranking_budget is not None else 0.25)

    async def _create_completion_once(self, client: Any, **kwargs: Any) -> Any:
        if self.progress_sink is None or not is_deepseek_model_endpoint(
            model_id=self.model_id, base_url=self.base_url
        ):
            return await client.chat.completions.create(**kwargs)

        prior_tokens = self._stream_output_tokens
        prior_chars = self._stream_received_chars
        prior_exact = self._stream_usage_exact

        def on_progress(metrics: dict[str, int]) -> None:
            public = {"received_chars": prior_chars + metrics["received_chars"]}
            source_key = (
                "output_tokens" if "output_tokens" in metrics else "estimated_output_tokens"
            )
            if source_key in metrics:
                token_key = "output_tokens" if prior_exact and source_key == "output_tokens" \
                    else "estimated_output_tokens"
                public[token_key] = prior_tokens + metrics[source_key]
            emit_progress(self.progress_sink, "synthesis_stream", **public)

        # Transport memory protection follows the actual model allowance and
        # counts discarded reasoning characters without imposing an 8192 cap.
        maximum = self._completion_budget()
        accumulator = CompletionStream(max_chars=max(16 * 1024 * 1024, (maximum or 0) * 32), on_progress=on_progress)
        stream = await client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True}
        )
        try:
            async for chunk in stream:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise asyncio.CancelledError()
                value = chunk if isinstance(chunk, Mapping) else chunk.model_dump()
                accumulator.add(value)
            value = accumulator.completion()
        except CompletionStreamError as exc:
            exc.usage = bounded_token_usage(accumulator.usage)
            raise
        finally:
            if self._generation_attempts.rows:
                self._capture_visible_output("".join(accumulator.parts), self._generation_attempts.rows[-1],
                                             finish_reason=accumulator.finish_reason)
            with suppress(Exception):
                if self._ranking_budget is None:
                    await stream.close()
                else:
                    async with asyncio.timeout(max(0.01, min(1.0, self._ranking_budget.remaining_seconds))):
                        await stream.close()
        self._stream_received_chars += accumulator.received_chars
        tokens = value["usage"].get("completion_tokens")
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
            self._stream_output_tokens += tokens
        else:
            self._stream_output_tokens += (accumulator.received_chars + 3) // 4
            self._stream_usage_exact = False
        return SimpleNamespace(
            model=value["model"], id=value["id"],
            choices=[SimpleNamespace(finish_reason=value["choices"][0]["finish_reason"],
                       message=SimpleNamespace(content=value["choices"][0]["message"]["content"]))],
            usage=SimpleNamespace(model_dump=lambda: value["usage"]),
        )

    def _completion_parts(
        self,
        response: Any,
    ) -> tuple[ModelIdentity, Any, dict[str, Any]]:
        """Validate the provider envelope without weakening structured-output checks."""

        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise SynthesisOutputError("model response did not contain a chat-completion result")
        observed = getattr(response, "model", None)
        if not isinstance(observed, str) or not observed.strip():
            raise ModelIdentityError("provider response did not identify the model used")
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason not in (None, "stop"):
            safe_reason = finish_reason if finish_reason in ("length", "content_filter", "tool_calls", "function_call") else "invalid"
            failure = GenerationFailure(
                "model did not complete the JSON response safely "
                f"(finish_reason={safe_reason})",
                category="output_length" if safe_reason == "length" else "provider_finish",
                finish_reason=safe_reason, retryable=safe_reason == "length",
            )
            failure.usage = bounded_token_usage(response.usage.model_dump() if response.usage else {})
            raise failure
        request_id = getattr(response, "id", None)
        identity = ModelIdentity(
            provider=self.provider,
            requested_model_id=self.model_id,
            observed_model_id=observed,
            request_id=request_id,
            endpoint=self.base_url,
            offline=self.offline,
        )
        if self.require_exact_model_id and observed != self.model_id:
            raise ModelIdentityError(
                "provider served a different model identifier than requested"
            )
        self._identity = identity
        try:
            content = visible_completion_text(
                response.choices[0].message.content,
                allow_blocks=infer_provider_id(self.base_url or "") == "mistral",
            )
        except ValueError as exc:
            raise SynthesisOutputError("model returned an unsupported content format") from exc
        usage = bounded_token_usage(response.usage.model_dump() if response.usage else {})
        return identity, content, usage


def visible_completion_text(content: Any, *, allow_blocks: bool = False) -> str:
    """Extract only visible text, without exposing a provider's thinking chunks."""

    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not allow_blocks or not isinstance(content, list):
        raise ValueError("unsupported completion content")
    visible: list[str] = []
    for chunk in content:
        if not isinstance(chunk, Mapping):
            raise ValueError("unsupported completion content")  # noqa: TRY004 - provider response validation
        if chunk.get("type") == "text" and isinstance(chunk.get("text"), str):
            visible.append(chunk["text"])
        elif chunk.get("type") == "thinking" and isinstance(chunk.get("thinking"), list):
            continue
        else:
            raise ValueError("unsupported completion content")
    return "".join(visible)


@asynccontextmanager
async def _closing_model_client(client: Any, *, resource_budget: RankingBudget | None = None):
    """Close each SDK client on success and failure, including pre-request errors."""

    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            # A cleanup failure must not overwrite the original inference error.
            with suppress(Exception):
                if resource_budget is None:
                    await close()
                else:
                    async with asyncio.timeout(max(0.01, min(1.0, resource_budget.remaining_seconds))):
                        await close()


def _parse_synthesis_content(
    content: Any,
    *,
    identity: ModelIdentity,
    usage: dict[str, Any],
) -> SynthesisOutput:
    if not isinstance(content, str) or not content.strip():
        raise SynthesisOutputError("model returned an empty response")
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SynthesisOutputError("model response is not valid JSON") from exc
    return _parse_synthesis(value, identity, sha256_json(value), usage)


def _ranking_system_prompt(system_prompt: str, query: LocusQuery) -> str:
    """Add the catalogued provenance policy only for a recorded LD scope."""
    if query.ld_expansion is None:
        return system_prompt
    return system_prompt + "\n\n" + load_prompt("ld_expansion_policy")


def _initial_ranking_system_prompt(system_prompt: str, *, evidence_view: str) -> str:
    """Assemble four catalogued sections for an initial ranking request.

    The field guide owns source/proxy and coverage interpretation. Repair and
    report requests retain their separate contracts and conditional LD policy.
    Caller-provided concept text is preserved; scientific instructions otherwise
    come only from the versioned prompt catalog.
    """
    field_guides = {
        "gene_comparison": "agent_gene_comparison_contract",
        "complete_facts": "agent_fact_view_contract",
        "legacy_projection": "agent_projected_evidence_contract",
        "v2g_cards": "agent_v2g_card_contract",
    }
    if evidence_view not in field_guides:
        raise ValueError(f"Unsupported ranking evidence view: {evidence_view}")
    if evidence_view == "v2g_cards" and system_prompt == load_prompt("agent_system"):
        system_prompt = load_prompt("agent_v2g_system")
    return "\n\n".join((
        system_prompt,
        load_prompt("agent_v2g_reasoning_process" if evidence_view == "v2g_cards" else "agent_reasoning_process"),
        load_prompt("agent_v2g_output_contract" if evidence_view == "v2g_cards" else "agent_output_contract"),
        load_prompt(field_guides[evidence_view]),
    ))


def _complete_repair_draft(value: Any) -> str:
    """Supply the complete existing draft or decline the repair request."""
    draft = str(value or "")
    if len(draft) > MAX_SYNTHESIS_REPAIR_DRAFT_CHARS:
        raise SynthesisOutputError(
            f"complete draft ({len(draft)} characters) exceeds the repair limit "
            f"({MAX_SYNTHESIS_REPAIR_DRAFT_CHARS} characters); no repair request was sent"
        )
    return draft


def _synthesis_repair_messages(
    *,
    validation_error: Exception,
    malformed_draft: Any,
    allowed_genes: tuple[str, ...],
    model_visible_evidence_ids: tuple[str, ...],
) -> list[dict[str, str]]:
    payload = {
        "validation_error": str(validation_error)[:MAX_SYNTHESIS_REPAIR_ERROR_CHARS],
        "malformed_draft": _complete_repair_draft(malformed_draft),
        "allowed_genes": list(allowed_genes),
        "model_visible_evidence_ids": list(model_visible_evidence_ids),
    }
    return [
        {"role": "system", "content": load_prompt("agent_repair_system")},
        {
            "role": "user",
            "content": format_prompt(
                "agent_repair_user",
                payload=canonical_json(payload),
            ),
        },
    ]


def _synthesis_validation_repair_messages(
    *,
    validation_errors: tuple[str, ...],
    invalid_draft: str,
    allowed_genes: tuple[str, ...],
    model_visible_evidence_ids: tuple[str, ...],
    citation_bindings: tuple[dict[str, Any], ...] = (),
) -> list[dict[str, str]]:
    payload = {
        "validation_errors": [
            str(error)[:MAX_SYNTHESIS_REPAIR_ERROR_CHARS]
            for error in validation_errors[:parameter("model.ranking_repair_max_errors")]
        ],
        "invalid_draft": _complete_repair_draft(invalid_draft),
        "allowed_genes": list(allowed_genes),
        "model_visible_evidence_ids": list(model_visible_evidence_ids),
        "citation_bindings": list(citation_bindings),
    }
    return [
        {
            "role": "system",
            "content": load_prompt("agent_validation_repair_system"),
        },
        {
            "role": "user",
            "content": format_prompt(
                "agent_validation_repair_user",
                payload=canonical_json(payload),
            ),
        },
    ]


def _citation_bindings(evidence: tuple[EvidenceRecord, ...], identifiers: set[str], *,
                       shown_records: Mapping[str, dict[str, Any]] | None = None) -> tuple[dict[str, Any], ...]:
    """Describe only already-cited visible identifiers; do not expand repair scope."""
    from .ranking_claims import raw_measurements

    return tuple({"evidence_id": record.evidence_id, "gene_id": record.gene_id,
                  "direct_locus_gene_link": is_direct_locus_gene_evidence(record),
                  **({"shown_scientific_record": deepcopy(shown_records[record.evidence_id])}
                     if shown_records and record.evidence_id in shown_records else {}),
                  **({"ld_proxy_links": deepcopy(record.payload["context"]["ld_proxy_links"])}
                     if not (shown_records and record.evidence_id in shown_records)
                     and isinstance(record.payload.get("context"), dict)
                     and record.payload["context"].get("ld_proxy_links") else {}),
                  **({"raw_measurements": [{"metric": item.metric, "value": str(item.value),
                                        "gene_id": item.gene_id, "variant_id": item.variant_id,
                                        "trait": item.trait} for item in raw_measurements(record)]}
                     if raw_measurements(record) else {})}
                 for record in evidence if record.evidence_id in identifiers)


def _synthesis_output_payload(output: SynthesisOutput) -> dict[str, Any]:
    return {
        **({"candidate_decisions": [asdict(item) for item in output.candidate_decisions]}
           if output.candidate_decisions or output.model_shortlisting_required else {}),
        "ranked_genes": [
            {
                "gene_id": item.gene_id,
                "evidence_ids": list(item.evidence_ids),
                "rationale": item.rationale,
                "limitations": list(item.limitations),
                **({"tied_with_previous": True} if item.tied_with_previous else {}),
            }
            for item in output.ranked_genes
        ],
        "claims": [
            {"text": item.text, "evidence_ids": list(item.evidence_ids)}
            for item in output.claims
        ],
        "summary": output.summary,
    }


def _merged_usage(
    initial: Mapping[str, Any], repair: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep auditable total token counts across the initial and repair calls."""

    merged = _combined_usage(initial, repair)
    merged["validation_repair_attempts"] = 1
    return merged


def _combined_usage(
    initial: Mapping[str, Any], repair: Mapping[str, Any]
) -> dict[str, Any]:
    """Combine provider counters without assigning a repair category."""

    merged: dict[str, Any] = dict(initial)
    for key, value in repair.items():
        prior = merged.get(key)
        if isinstance(prior, Mapping) and isinstance(value, Mapping):
            merged[key] = _combined_usage(prior, value)
        elif (
            isinstance(prior, (int, float))
            and not isinstance(prior, bool)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            merged[key] = prior + value
        else:
            merged[key] = value
    return merged


def _validate_repaired_identifiers(
    output: SynthesisOutput,
    *,
    malformed_draft: Any,
) -> None:
    """Reject identifiers that a structure-only repair added to the draft.

    The normal grounding validator checks identifiers against the retrieved
    evidence ledger.  This additional check enforces the narrower repair
    contract: a repair cannot introduce an identifier for the first time.
    A parseable original candidate list also owns its exact order and count.
    """

    # The outgoing request is bounded separately. Alias expansion can lengthen
    # this authority text; never turn a valid tail identifier into an invention.
    draft = str(malformed_draft or "")
    identifiers = {
        *(item.gene_id for item in output.candidate_decisions),
        *(evidence_id for item in output.candidate_decisions for evidence_id in item.evidence_ids),
        *(item.gene_id for item in output.ranked_genes),
        *(evidence_id for item in output.ranked_genes for evidence_id in item.evidence_ids),
        *(evidence_id for claim in output.claims for evidence_id in claim.evidence_ids),
    }
    introduced = sorted(
        identifier for identifier in identifiers
        if not re.search(r"(?<![\w.:-])" + re.escape(identifier) + r"(?![\w.:-])", draft)
    )
    if introduced:
        bounded = ", ".join(item[:160] for item in introduced[:10])
        raise SynthesisOutputError(
            "model output repair introduced identifier(s) absent from the malformed draft: "
            + bounded
        )
    selection_required = bool(output.candidate_decisions) or output.model_shortlisting_required
    authority = draft
    if selection_required:
        # A complete outer fence is only a wrapper. Do not guess syntax or
        # missing decisions: a structure repair cannot choose on their behalf.
        fenced = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", authority, re.I | re.S)
        if fenced:
            authority = fenced[1]
    duplicate_fields = False

    def unique_fields(pairs):
        nonlocal duplicate_fields
        result = {}
        for key, value in pairs:
            duplicate_fields = duplicate_fields or key in result
            result[key] = value
        return result

    try:
        original = json.loads(authority, object_pairs_hook=unique_fields)
    except (ValueError, TypeError, RecursionError) as exc:
        if selection_required:
            raise SynthesisOutputError(
                "model output repair cannot establish the original candidate decisions and ranking"
            ) from exc
        return
    selection_required = selection_required or (isinstance(original, dict) and "candidate_decisions" in original)
    if selection_required and duplicate_fields:
        raise SynthesisOutputError("model output repair cannot establish unambiguous original candidate decisions")
    decisions = original.get("candidate_decisions") if isinstance(original, dict) else None
    if selection_required:
        if not (isinstance(decisions, list) and decisions
            and all(isinstance(row, dict) and isinstance(row.get("gene_id"), str)
                    and row["gene_id"].strip()
                    and isinstance(row.get("status"), str)
                    and row.get("status") in {"selected", "not_selected", "unresolved"} for row in decisions)
            and len({row["gene_id"] for row in decisions}) == len(decisions)):
            raise SynthesisOutputError("model output repair cannot establish every original candidate decision")
        expected_decisions = tuple((row["gene_id"], row["status"]) for row in decisions)
        if tuple((row.gene_id, row.status) for row in output.candidate_decisions) != expected_decisions:
            raise SynthesisOutputError("model output repair must preserve every original candidate decision")
    rows = original.get("ranked_genes") if isinstance(original, dict) else None
    if (not isinstance(rows, list)
            or any(not isinstance(row, dict) or not isinstance(row.get("gene_id"), str)
                   or not row["gene_id"].strip() for row in rows)):
        if selection_required:
            raise SynthesisOutputError("model output repair cannot establish the exact original model ranking")
        return
    genes = tuple(row["gene_id"] for row in rows)
    if len(set(genes)) != len(genes):
        if selection_required:
            raise SynthesisOutputError("model output repair cannot establish the exact original model ranking")
        return
    expected = tuple((gene, rank) for rank, gene in enumerate(genes, start=1))
    actual = tuple((row.gene_id, row.rank) for row in output.ranked_genes)
    if actual != expected:
        raise SynthesisOutputError("model output repair must preserve the exact original model ranking; "
                                   "candidate genes cannot be added, removed, reordered or renumbered")


def _candidate_gene_universe(
    query: LocusQuery,
    evidence: tuple[EvidenceRecord, ...],
) -> tuple[str, ...]:
    """Return the exact bounded universe visible to every synthesizer.

    Interactive locus queries normally begin without user-supplied genes; the
    semantic retrieval executor discovers them from L2G/QTL/consequence records.
    That executor does not mutate the immutable outer ``LocusQuery``.  Deriving
    the same universe here prevents the hosted model from receiving an empty
    candidate list while the deterministic ablation sees discovered genes.
    """

    if query.candidate_genes:
        return query.candidate_genes
    discovered = tuple(sorted({record.gene_id for record in evidence if record.gene_id}))
    return discovered


def _projection_attempt_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Retain the attempted model scope even when final fallback uses a full ledger."""
    summary = payload["evidence_projection"]
    keys = ("full_record_count", "candidate_gene_record_count", "deduplicated_candidate_gene_record_count",
            "records_collapsed_by_deduplication", "model_visible_record_count", "records_omitted_after_deduplication",
            "candidate_gene_count", "candidate_genes_with_model_visible_evidence",
            "member_measurements_omitted", "model_visible_member_measurements",
            "model_visible_regulatory_relationships", "regulatory_relationships_omitted")
    return {**{key: summary[key] for key in keys if type(summary.get(key)) is int and summary[key] >= 0},
            "projection_sha256": sha256_json(summary),
            "model_visible_evidence_ids_sha256": summary.get("model_visible_evidence_ids_sha256") or sha256_json([row["evidence_id"] for row in payload.get("evidence_ledger", ()) if isinstance(row, Mapping) and row.get("evidence_id")]),
            "model_evidence_payload_sha256": sha256_json(payload),
            **{key: summary[key] for key in ("evidence_aliases_sha256", "record_bindings_sha256") if key in summary}}


def _resolve_evidence_aliases(output: SynthesisOutput, aliases: Mapping[str, str]) -> SynthesisOutput:
    def prose(text: str) -> str:
        # A known alias explicitly labelled as a record is a citation, not a
        # gene symbol (e.g. "E2G record E225 shows ..."). Resolve it before
        # entity and inline-citation validation. Unknown aliases and unlabelled
        # gene assertions stay unchanged and must still pass those validators.
        return re.sub(
            r"\b(?:evidence|records?|citations?)(?:\s+(?:id|identifier))?"
            r"\s*[:=]?\s+[`\[]?(?P<alias>E[0-9]+)\b",
            lambda match: match.group(0)[:match.start("alias") - match.start()]
            + aliases.get(match["alias"], match["alias"]),
            text, flags=re.IGNORECASE,
        )

    def resolve(row):
        fields = {"evidence_ids": tuple(dict.fromkeys(aliases.get(identifier, identifier) for identifier in row.evidence_ids))}
        if hasattr(row, "rationale"):
            fields.update(rationale=prose(row.rationale), limitations=tuple(prose(text) for text in row.limitations))
        elif hasattr(row, "reason"):
            fields["reason"] = prose(row.reason)
        else:
            fields["text"] = prose(row.text)
        return replace(row, **fields)
    return replace(output, ranked_genes=tuple(resolve(row) for row in output.ranked_genes),
                   claims=tuple(resolve(row) for row in output.claims), summary=prose(output.summary),
                   candidate_decisions=tuple(resolve(row) for row in output.candidate_decisions))


def _draft_aliases_to_original(draft: str, aliases: Mapping[str, str]) -> str:
    return re.sub(r'"(E[0-9]+)"', lambda match: json.dumps(aliases.get(match[1], match[1])), draft)


def _translate_evidence_references(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, dict):
        return {mapping.get(key, key): _translate_evidence_references(item, mapping) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_translate_evidence_references(item, mapping) for item in value]
    return value


def _natural_ranking_request(query: LocusQuery, evidence: tuple[EvidenceRecord, ...], *,
    candidate_genes: tuple[str, ...], system_prompt: str, model_id: str, base_url: str | None):
    from .scientific_facts import build_scientific_fact_view

    # Stable presentation prevents parallel request completion order from becoming
    # an accidental cue to the model. This orders inputs, never gene priorities.
    evidence = tuple(sorted(evidence, key=lambda row: (
        row.gene_id or "", row.kind.value, row.study_id or "", row.credible_set_id or "",
        row.variant_id or "", row.evidence_id)))
    candidate_genes = tuple(sorted(candidate_genes))
    view = build_scientific_fact_view(query, evidence, candidate_genes=candidate_genes)
    counts = view.payload["coverage"]
    supported = {row.gene_id for row in evidence if row.gene_id in candidate_genes}
    summary = {"capacity_policy": "complete_scientific_facts", "full_record_count": len(evidence),
        "candidate_gene_record_count": counts["candidate_record_count"],
        "deduplicated_candidate_gene_record_count": counts["scientific_fact_count"],
        "records_collapsed_by_deduplication": counts["candidate_record_count"] - counts["scientific_fact_count"],
        "model_visible_record_count": counts["scientific_fact_count"], "records_omitted_after_deduplication": 0,
        "member_measurements_omitted": 0, "model_visible_member_measurements": counts["member_measurement_count"],
        "model_visible_regulatory_relationships": counts["regulatory_relationship_count"],
        "regulatory_relationships_omitted": counts["regulatory_relationships_omitted"],
        "candidate_gene_count": len(candidate_genes), "candidate_genes_with_model_visible_evidence": len(supported),
        "candidate_genes_without_eligible_evidence": [gene for gene in candidate_genes if gene not in supported],
        "model_visible_evidence_ids_sha256": sha256_json(view.evidence_ids),
        "evidence_aliases_sha256": sha256_json(view.evidence_aliases),
        "record_bindings_sha256": sha256_json(view.record_bindings)}
    if query.ld_expansion is not None:
        visible_variants = sorted({row["variant_id"] for row in view.table("variants")})
        all_variants = analysis_variant_ids(query)
        scope = query.ld_expansion
        summary["analysis_scope"] = {
            "source_variant_ids": [row.variant_id for row in query.variants],
            "source_observations": [{"variant_id": row.variant_id, "p_value": row.p_value} for row in query.variants],
            "analysis_variant_count": len(all_variants),
            "ld_proxy_count": len(set(all_variants).difference(row.variant_id for row in query.variants)),
            "model_visible_variant_ids": visible_variants,
            "analysis_variants_without_visible_records": len(set(all_variants).difference(visible_variants)),
            **{key: scope.get(key) for key in ("population", "threshold", "window_bp", "reference_panel", "genome_build", "complete", "status")},
            "window_bp_basis": "per_side",
            "proxy_evidence_complete": scope.get("proxy_evidence_complete", True),
        }
    from .phenotype_scope import model_biological_context

    payload = {"locus": {"locus_id": query.locus_id, "trait": query.trait,
        "source_trait": query.source_trait or query.trait, "candidate_genes": list(candidate_genes),
        "relevant_tissues": list(query.relevant_tissues), "task": query.task,
        "biological_context": model_biological_context(query.biological_context),
        "input_order_policy": "canonical-identities-v1",
        "source_observations": [{"variant_id": row.variant_id, "p_value": row.p_value} for row in query.variants]},
        "scientific_facts": view.payload, "evidence_projection": summary,
        "direct_evidence_gene_ids": sorted({row.gene_id for row in evidence if row.gene_id in candidate_genes
            and is_direct_locus_gene_evidence(row)})}
    messages = [{"role": "system", "content": _initial_ranking_system_prompt(system_prompt, evidence_view="complete_facts")},
                {"role": "user", "content": format_prompt("agent_user", payload=canonical_json(payload))}]
    summary["request_capacity"] = {"reason": "complete_scientific_facts_preserved",
        **{key: value for key, value in model_token_budget(base_url=base_url, model_id=model_id, messages=messages).items()
           if key in {"context_window", "provider_max_output_tokens", "input_token_budget"}}}
    messages[1]["content"] = format_prompt("agent_user", payload=canonical_json(payload))
    return payload, messages, view


def _gene_comparison_ranking_request(query, original_payload, full_view, *, system_prompt, model_id, base_url,
                                     retrieval_statuses=None):
    """Give the model traceable gene comparisons while retaining the full audit.

    Aggregation is an explicit representation policy, not emergency context
    truncation. Only references actually shown on the cards may be cited.
    """
    from .ranking_cards import build_ranking_cards, compact_model_gene_comparison

    cards = (build_ranking_cards(query, full_view, retrieval_statuses=retrieval_statuses)
             if retrieval_statuses else build_ranking_cards(query, full_view))
    model_gene_comparison = compact_model_gene_comparison(cards.payload)
    expected = set(original_payload["locus"]["candidate_genes"])
    actual = [row["gene_id"] for row in cards.payload["comparison_cards"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Gene comparison must retain every shortlisted candidate exactly once")
    full_evidence_ids = set(full_view.evidence_ids) | {row["evidence_id"] for row in full_view.record_bindings}
    if not set(cards.evidence_ids).issubset(full_evidence_ids):
        raise ValueError("Gene comparison contains an unknown evidence reference")
    visible_ids = set(cards.evidence_ids)
    citation_records = cards.payload["citation_records"]
    visible_members = {row["credible_set_member"]["id"] for row in citation_records
                       if row.get("credible_set_member", {}).get("id")}
    visible_variants = sorted({row["variant"]["variant_id"] for row in citation_records
                               if row.get("variant", {}).get("variant_id")})
    visible_regulatory_links = {row["regulatory_link_id"] for row in full_view.record_bindings
                                if row["evidence_id"] in visible_ids and row.get("regulatory_link_id")}
    summary = deepcopy(original_payload["evidence_projection"])
    summary.update(capacity_policy="gene_comparison_cards_v2",
        model_visible_record_count=len(cards.evidence_ids),
        model_visible_member_measurements=len(visible_members),
        model_visible_regulatory_relationships=len(visible_regulatory_links),
        model_visible_evidence_ids_sha256=sha256_json(cards.evidence_ids),
        evidence_aliases_sha256=sha256_json(cards.evidence_aliases),
        evidence_aggregation={"policy": "gene_comparison_cards_v2", "candidate_count": len(actual),
            "full_evidence_record_count": len(full_evidence_ids),
            "directly_citable_record_count": len(cards.evidence_ids),
            "records_summarized": max(0, len(full_evidence_ids) - len(cards.evidence_ids)),
            "individually_cited_variant_ids": visible_variants,
            "analysis_variants_without_individual_citations": len(set(analysis_variant_ids(query)).difference(visible_variants)),
            "citation_visibility_is_retrieval_coverage": False,
            "member_measurements_summarized": max(0,
                original_payload["evidence_projection"]["model_visible_member_measurements"] - len(visible_members)),
            "regulatory_relationships_summarized": max(0,
                original_payload["evidence_projection"]["model_visible_regulatory_relationships"] - len(visible_regulatory_links)),
            "full_ledger_retained": True, "individual_records_all_shown": False,
            "aggregation_receipt_sha256": sha256_json(cards.aggregation_receipt),
            "complete_scientific_facts_sha256": sha256_json(full_view.payload)})
    if "analysis_scope" in summary:
        # Individual card citations are a presentation choice, not failed LD
        # lookups or variants without retrieved evidence.
        summary["analysis_scope"].pop("model_visible_variant_ids", None)
        summary["analysis_scope"].pop("analysis_variants_without_visible_records", None)
    model_projection = {
        key: deepcopy(summary[key])
        for key in (
            "analysis_scope", "candidate_gene_count", "candidate_gene_record_count",
            "candidate_genes_with_model_visible_evidence", "candidate_genes_without_eligible_evidence",
            "full_record_count", "model_visible_record_count", "model_visible_member_measurements",
            "model_visible_regulatory_relationships", "records_summarized", "records_omitted_after_deduplication",
            "regulatory_relationships_omitted", "member_measurements_omitted", "records_collapsed_by_deduplication",
        )
        if key in summary
    }
    if isinstance(summary.get("evidence_aggregation"), Mapping):
        model_projection["evidence_aggregation"] = {
            key: deepcopy(summary["evidence_aggregation"][key])
            for key in (
                "analysis_variants_without_individual_citations", "candidate_count",
                "citation_visibility_is_retrieval_coverage", "directly_citable_record_count",
                "full_evidence_record_count", "full_ledger_retained", "individual_records_all_shown",
                "individually_cited_variant_ids", "member_measurements_summarized",
                "records_summarized", "regulatory_relationships_summarized",
            )
            if key in summary["evidence_aggregation"]
        }
    payload = {"locus": deepcopy(original_payload["locus"]), "gene_comparison": model_gene_comparison,
               "evidence_projection": model_projection,
               "direct_evidence_gene_ids": list(original_payload["direct_evidence_gene_ids"])}
    messages = [{"role": "system", "content": _initial_ranking_system_prompt(system_prompt, evidence_view="gene_comparison")},
                {"role": "user", "content": format_prompt("agent_user", payload=canonical_json(payload))}]
    summary["request_capacity"] = {"reason": "audited_gene_comparison_representation",
        **{key: value for key, value in model_token_budget(base_url=base_url, model_id=model_id, messages=messages).items()
           if key in {"context_window", "provider_max_output_tokens", "input_token_budget"}}}
    messages[1]["content"] = format_prompt("agent_user", payload=canonical_json(payload))
    return payload, messages, cards


@dataclass(frozen=True, slots=True)
class V2GCardView:
    payload: dict[str, Any]
    evidence_aliases: dict[str, str]
    aggregation_receipt: dict[str, Any]


def _v2g_card_ld_context(query, cards):
    """Keep measured source-proxy pairs separate from the cards' scalar weights."""
    from .analysis_scope import validate_analysis_scope
    from .ld_support import ld_provider_metadata
    from .scientific_facts import _scientific_table

    context = {
        "provided": query.ld_expansion is not None,
        "card_scalar_semantics": {
            "selected_source_r2_to_anchor": "Identity convention for locus weighting (1), not measured LD to another SNP.",
            "proxy_r2_to_anchor": "Maximum recorded r2 across selected-source links; the scalar does not name its partner.",
            "max_shared_r2": "Maximum card scalar among shared variants; selected-source identity values may be included.",
            "pairwise_ld": "Only the recorded ld_relations authorize pair-specific LD claims; name both SNPs and population.",
            "maximum_ld_wording": "For a proxy scalar, say maximum recorded r2 with selected SNPs; it is not necessarily r2 with the GWAS lead.",
        },
        "ld_relations": _scientific_table([]),
    }
    if query.ld_expansion is None:
        return context
    validate_analysis_scope(query)
    plan = query.ld_expansion
    named = set()

    def collect(value):
        if isinstance(value, Mapping):
            if isinstance(value.get("variant_id"), str):
                named.add(value["variant_id"])
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(cards)
    aliases = plan.get("resolved_source_aliases", {})
    named = {aliases.get(identifier, identifier) for identifier in named}
    links = [link for link in plan["ld_links"]
             if aliases.get(link["proxy_variant_id"], link["proxy_variant_id"]) in named]
    context.update(
        **{key: plan.get(key) for key in ("status", "complete", "population", "reference_panel", "genome_build",
                                         "threshold", "window_bp")},
        **ld_provider_metadata(plan),
        resolved_source_aliases=dict(aliases),
        window_bp_basis="per_side",
        relation_selection="all_verified_edges_for_card_variants",
        relation_selection_note=("Every recorded selected-source edge for a variant named in the cards is retained. "
                                 "Other edges remain in the full ledger; omitted edges are not evidence of absent LD."),
        ld_relations=_scientific_table([
            {key: link[key] for key in ("selected_variant_id", "proxy_variant_id", "r2", "population", "genome_build")}
            for link in links]),
    )
    return context


def _v2g_card_ranking_request(query, original_payload, full_view, *, v2g_cards, system_prompt, model_id,
                              base_url):
    """Replace the scientific body with the gene-centric cards; prompts and contract stay unchanged.

    Card evidence IDs become the fact view's E aliases. Records outside the decision
    ledger have no alias and are shown as context without a citable reference.
    """
    from .v2g.cards import build_gene_cards
    from .v2g.compact_input import encode_v2g_payload

    candidates = list(original_payload["locus"]["candidate_genes"])
    by_gene = {card["gene_id"]: card for card in v2g_cards}
    if set(by_gene) != set(candidates):
        raise ValueError("Gene-centric cards must cover the candidate universe exactly once")
    alias_by_id = {evidence_id: alias for alias, evidence_id in full_view.evidence_aliases.items()}
    shown: dict[str, str] = {}

    def aliased(value):
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if key == "evidence_id":
                    if item in alias_by_id:
                        result[key] = alias_by_id[item]
                        shown[alias_by_id[item]] = item
                elif key == "evidence_ids":
                    refs = [alias_by_id[entry] for entry in item if entry in alias_by_id]
                    shown.update({alias_by_id[entry]: entry for entry in item if entry in alias_by_id})
                    if refs:
                        result[key] = refs
                else:
                    result[key] = aliased(item)
            return result
        if isinstance(value, list):
            return [aliased(item) for item in value]
        return value

    card_set = build_gene_cards([by_gene[gene] for gene in sorted(candidates)], seed=query.locus_id, anonymize=False)
    cards = []
    for card in card_set.cards:
        cards.append({"gene_id": card.pop("id"), **card})
    from .v2g.model_view import inline_ld, representative_cards
    ld_context = _v2g_card_ld_context(query, cards)
    inline_ld(cards, ld_context)
    cards, view_receipt = representative_cards(cards, biological_context=query.biological_context)
    cards = [aliased(card) for card in cards]
    summary = deepcopy(original_payload["evidence_projection"])
    summary.update(capacity_policy="gwasmind_v2g_gene_cards_v1", model_visible_record_count=len(shown),
                   model_visible_evidence_ids_sha256=sha256_json(sorted(shown.values())),
                   evidence_aliases_sha256=sha256_json(shown),
                   records_summarized=max(0, len(full_view.evidence_ids) - len(shown)))
    model_projection = {key: deepcopy(summary[key]) for key in (
        "analysis_scope", "candidate_gene_count", "candidate_genes_with_model_visible_evidence",
        "candidate_genes_without_eligible_evidence", "full_record_count", "model_visible_record_count",
        "records_summarized", "records_omitted_after_deduplication") if key in summary}
    if "analysis_scope" in model_projection:
        model_projection["analysis_scope"].pop("model_visible_variant_ids", None)
        model_projection["analysis_scope"].pop("analysis_variants_without_visible_records", None)
    payload = {"locus": deepcopy(original_payload["locus"]),
               "gene_comparison": {"format": "gwasmind_v2g_gene_cards_v1", "comparison_cards": cards,
                                   "ld_context": ld_context,
                                   "shortlisting_guidance": {
                                       "selection_owner": "LLM",
                                       "candidate_pool_size": len(cards),
                                       "fixed_shortlist_size": None,
                                       "note": "Shortlist from locus links and position; expression and associations are background. No candidate is automatically selected or removed."}},
               "evidence_projection": model_projection,
               "direct_evidence_gene_ids": list(original_payload["direct_evidence_gene_ids"])}
    payload, compact_receipt = encode_v2g_payload(payload)
    messages = [{"role": "system", "content": _initial_ranking_system_prompt(system_prompt, evidence_view="v2g_cards")},
                {"role": "user", "content": format_prompt("agent_user", payload=canonical_json(payload))}]
    summary["request_capacity"] = {"reason": "gene_centric_evidence_cards",
        **{key: value for key, value in model_token_budget(base_url=base_url, model_id=model_id, messages=messages).items()
           if key in {"context_window", "provider_max_output_tokens", "input_token_budget"}}}
    receipt = {"policy": "gwasmind_v2g_gene_cards_v1", "candidate_count": len(cards),
               "citable_record_count": len(shown), "full_evidence_record_count": len(full_view.evidence_ids),
               "cards_sha256": sha256_json(cards), "compact_input": compact_receipt,
               "representative_view": view_receipt}
    return payload, messages, V2GCardView(payload=payload, evidence_aliases=dict(sorted(shown.items())),
                                          aggregation_receipt=receipt)


def _projected_ranking_request(
    query: LocusQuery, evidence: tuple[EvidenceRecord, ...], *, candidate_genes: tuple[str, ...],
    system_prompt: str, model_id: str, base_url: str | None, input_token_limit: int | None = None,
    provider_capacity: Mapping[str, int] | None = None,
    allowed_evidence_ids: frozenset[str] | None = None,
    allowed_publication_pmids: frozenset[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Fit the complete model-visible request into the model's joint capacity.

    Output reservation comes only from the provider's maximum, never a separate
    application quota. Unknown limits remain unknown. Each local packing pass
    removes only measured overage and reports the exact scientific omissions.
    """
    budget = model_token_budget(base_url=base_url, model_id=model_id)
    token_limit = input_token_limit if input_token_limit is not None else budget["input_token_budget"]
    max_chars = (token_limit * 3 if token_limit else
                 max(1, sum(len(canonical_json(asdict(row))) for row in evidence) * 2))
    system = _initial_ranking_system_prompt(system_prompt, evidence_view="legacy_projection")
    for packing_pass in range(3):
        try:
            records, summary = build_synthesis_evidence_projection(query, evidence,
                candidate_genes=candidate_genes, max_records=max(1, len(evidence)),
                max_chars=max_chars, use_model_context=True, allowed_evidence_ids=allowed_evidence_ids,
                allowed_publication_pmids=allowed_publication_pmids)
        except SynthesisProjectionError as exc:
            raise SynthesisOutputError(str(exc)) from exc
        summary["request_capacity"] = {
            "context_window": budget["context_window"],
            "reserved_maximum_output_tokens": budget["provider_max_output_tokens"],
            "input_token_budget": budget["input_token_budget"],
            "packing_input_token_budget": token_limit,
            "reason": budget["input_budget_reason"],
            "includes_system_prompt_and_evidence_json": True,
            "provider_tokenizer_calibrated": input_token_limit is not None,
            "packing_passes": packing_pass + 1,
        }
        if provider_capacity is not None:
            summary["request_capacity"]["provider_observed_capacity"] = dict(provider_capacity)
        visible = {str(row["evidence_id"]) for row in records}
        payload = {
            "locus": {"locus_id": query.locus_id, "trait": query.trait,
                "source_trait": query.source_trait or query.trait, "candidate_genes": list(candidate_genes),
                "candidate_universe_source": "retrieval_resolved_query", "relevant_tissues": list(query.relevant_tissues),
                "task": query.task, "benchmark": query.benchmark},
            "evidence_projection": summary, "evidence_ledger": records,
            "direct_evidence_gene_ids": sorted({record.gene_id for record in evidence
                if record.evidence_id in visible and is_direct_locus_gene_evidence(record)
                and record.gene_id in candidate_genes}),
        }
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": format_prompt("agent_user", payload=canonical_json(payload))}]
        measured = model_token_budget(base_url=base_url, model_id=model_id, messages=messages)
        # These are complete-message estimates; actual provider counts are kept in
        # the attempt ledger and can calibrate a context-exhaustion regeneration.
        excess = measured["prompt_tokens_estimate"] - token_limit if token_limit else 0
        if excess <= 0 or packing_pass == 2:
            return payload, messages
        # Once retries restrict selection to a prior subset, unused allocation
        # can hide the entire small overage. Remove measured excess from actual
        # occupied projection size so another pass really removes content.
        max_chars = max(1, min(max_chars, summary["model_visible_json_chars"]) - excess * 3)
    raise AssertionError("unreachable projection packing state")


def _parse_synthesis(
    value: Any,
    identity: ModelIdentity,
    raw_response_sha256: str,
    usage: dict[str, Any],
) -> SynthesisOutput:
    if not isinstance(value, dict):
        raise SynthesisOutputError("model output must be a JSON object")
    required_top = {"ranked_genes", "claims", "summary"}
    allowed_top = required_top | {"candidate_decisions"}
    unknown = set(value).difference(allowed_top)
    if unknown:
        raise SynthesisOutputError("unexpected output fields: " + ", ".join(sorted(unknown)))
    missing = required_top.difference(value)
    if missing:
        raise SynthesisOutputError("missing output fields: " + ", ".join(sorted(missing)))
    rows = value.get("ranked_genes")
    claims_value = value.get("claims")
    if not isinstance(rows, list) or not isinstance(claims_value, list):
        raise SynthesisOutputError("ranked_genes and claims must be arrays")
    decisions = []
    decision_values = value.get("candidate_decisions", [])
    if not isinstance(decision_values, list):
        raise SynthesisOutputError("candidate_decisions must be an array")
    for index, row in enumerate(decision_values):
        if not isinstance(row, dict) or set(row) != {"gene_id", "status", "reason", "evidence_ids"}:
            raise SynthesisOutputError("candidate decision fields do not match the contract")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in ("gene_id", "status", "reason")):
            raise SynthesisOutputError("candidate decisions require non-empty gene_id, status and reason")
        if row["status"] not in {"selected", "not_selected", "unresolved"}:
            raise SynthesisOutputError("candidate decision status must be selected, not_selected or unresolved")
        decisions.append(CandidateDecision(row["gene_id"], row["status"], row["reason"],
            _strict_string_list(row["evidence_ids"], label=f"candidate_decisions[{index}].evidence_ids")))
    ranked: list[RankedGene] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise SynthesisOutputError("each ranked gene must be an object")
        if not {"gene_id", "evidence_ids", "rationale", "limitations"} <= set(row) <= {
                "gene_id", "evidence_ids", "rationale", "limitations", "tied_with_previous"}:
            raise SynthesisOutputError("ranked gene fields do not match the contract")
        tied = row.get("tied_with_previous", False)
        if type(tied) is not bool:
            raise SynthesisOutputError(f"ranked_genes[{index - 1}].tied_with_previous must be a boolean")
        if tied and index == 1:
            raise SynthesisOutputError("the first ranked gene cannot be tied with a previous gene")
        if not isinstance(row["gene_id"], str):
            raise SynthesisOutputError(f"ranked_genes[{index - 1}].gene_id must be a string")
        if not isinstance(row["rationale"], str):
            raise SynthesisOutputError(f"ranked_genes[{index - 1}].rationale must be a string")
        evidence_ids = _strict_string_list(
            row["evidence_ids"],
            label=f"ranked_genes[{index - 1}].evidence_ids",
        )
        limitations_value = row["limitations"]
        if isinstance(limitations_value, str):
            # A single limitation is semantically identical to a one-item JSON
            # array. Normalize this harmless model variation locally while
            # keeping evidence IDs and every grounding-sensitive field strict.
            limitations_value = [limitations_value]
        limitations = _strict_string_list(
            limitations_value,
            label=f"ranked_genes[{index - 1}].limitations",
        )
        ranked.append(
            RankedGene(
                gene_id=row["gene_id"],
                rank=index,
                evidence_ids=evidence_ids,
                rationale=row["rationale"],
                limitations=limitations,
                tied_with_previous=tied,
            )
        )
    claims: list[CitedClaim] = []
    for row in claims_value:
        if not isinstance(row, dict) or set(row) != {"text", "evidence_ids"}:
            raise SynthesisOutputError("claim fields do not match the contract")
        if not isinstance(row["text"], str):
            raise SynthesisOutputError(f"claims[{len(claims)}].text must be a string")
        claim_evidence_ids = _strict_string_list(
            row["evidence_ids"],
            label=f"claims[{len(claims)}].evidence_ids",
        )
        claims.append(
            CitedClaim(
                text=row["text"],
                evidence_ids=claim_evidence_ids,
            )
        )
    if not isinstance(value["summary"], str):
        raise SynthesisOutputError("summary must be a string")
    return SynthesisOutput(
        ranked_genes=tuple(ranked),
        claims=tuple(claims),
        summary=value["summary"],
        identity=identity,
        raw_response_sha256=raw_response_sha256,
        usage=usage,
        candidate_decisions=tuple(decisions),
    )


def _validate_model_selection(output, candidates, *, required):
    from .shortlisting import selection_contract_errors

    errors = selection_contract_errors(output, candidates, required=required)
    if errors:
        raise SynthesisOutputError("; ".join(errors))


def _strict_string_list(value: Any, *, label: str) -> tuple[str, ...]:
    """Parse a JSON string array without silently iterating over a scalar string."""

    if not isinstance(value, list):
        raise SynthesisOutputError(f"{label} must be an array of strings")
    if any(not isinstance(item, str) for item in value):
        raise SynthesisOutputError(f"{label} must contain only strings")
    return tuple(value)
