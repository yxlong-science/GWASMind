from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import GWASMindAgent
from .artifacts import ArtifactStore
from .audit import AuditLogger, sha256_json
from .configuration import parameter
from .evidence_store import (
    MAX_MANIFEST_BYTES,
    STORE_KEY,
    import_legacy_document,
    read_document,
)
from .llm import DeterministicEvidenceSynthesizer, OpenAICompatibleSynthesizer
from .mcp.bootstrap import (
    SCHEMA_UNAVAILABLE_MESSAGE,
    UnavailableSemanticService,
    bootstrap_failure_diagnostic,
)
from .mcp.contracts import SemanticTool, ToolCoverage
from .mcp.dispatcher import SemanticMCPDispatcher
from .mcp.ensembl import EnsemblRegionGeneLookup
from .mcp.europe_pmc import EuropePMCPublicationClient
from .mcp.frozen_modules import FrozenSemanticModules, create_frozen_semantic_service
from .mcp.gateway import SemanticGatewayAdapter, blocked_modules_for_policy
from .mcp.live_modules import (
    LiveSemanticModules,
    create_discovered_live_semantic_service,
)
from .mcp.live_schema import OfficialSchemaNegotiator
from .mcp.sdk_server import run_stdio
from .model_options import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORTS,
    THINKING_MODES,
)
from .models import DataMode, DecisionStatus
from .policy import LeakagePolicy, frozen_exploratory_policy, live_policy
from .progress import JsonlProgressSink, emit_progress
from .providers.frozen import DuckDBFrozenOpenTargetsProvider
from .providers.live import LiveProviderError, OfficialOpenTargetsMCPProvider
from .runtime import RuntimeConfig
from .serialization import (
    leakage_policy_from_dict,
    locus_query_from_dict,
    validate_benchmark_policy,
)
from .snapshot import damaged_parquet_files, inspect_snapshot, load_snapshot_manifest
from .tools import GWASToolGateway, reviewed_tool_result

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _run_id_argument(value: str) -> str:
    if not _SAFE_RUN_ID.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run id must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _reasoning_profile_argument(value: str) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = item
        return result
    try:
        if len(value.encode("utf-8")) > 8192:
            raise ValueError("profile too large")
        profile = json.loads(value, object_pairs_hook=unique_object)
        if not isinstance(profile, dict):
            raise ValueError("profile must be an object")
        return profile
    except (ValueError, RecursionError) as exc:
        raise argparse.ArgumentTypeError("Invalid bounded model reasoning profile") from exc


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", delete=False,
            dir=path.parent, suffix=".tmp",
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _verified_frozen_snapshot(
    data_root: str | Path,
    *,
    release: str | None,
    snapshot_hash: str | None,
):
    if release == "live":
        raise ValueError("frozen-safe mode cannot use the live release")
    inspection = load_snapshot_manifest(data_root, expected_release=release)
    if snapshot_hash is not None and (
        not isinstance(snapshot_hash, str)
        or snapshot_hash.casefold() == "unverified"
        or snapshot_hash != inspection.manifest_sha256
    ):
        raise ValueError("--snapshot-hash must match the verified frozen snapshot manifest")
    return inspection


def _live_analysis_modules(
    provider: OfficialOpenTargetsMCPProvider,
    schema: Any,
) -> LiveSemanticModules:
    """Wire the live CLI path with the same bounded publication enrichment as MCP mode."""

    return LiveSemanticModules(
        provider,
        schema=schema,
        region_gene_lookup=EnsemblRegionGeneLookup(),
        publication_enricher=EuropePMCPublicationClient(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gwasmind")
    commands = parser.add_subparsers(dest="command", required=True)

    import_evidence = commands.add_parser("import-evidence-artifact",
        help="Losslessly import a retained large result/retrieval without external queries")
    import_evidence.add_argument("--input", required=True)
    import_evidence.add_argument("--output", required=True,
        help="New manifest file; the historical input is never overwritten")
    import_evidence.add_argument("--field", choices=("evidence", "records"), required=True)

    inspect = commands.add_parser("inspect-snapshot")
    inspect.add_argument("--data-root", required=True)
    inspect.add_argument(
        "--release",
        help=(
            "provision a manifest for this frozen release; omit to verify the "
            "existing conventional manifest"
        ),
    )
    inspect.add_argument("--full-hash", action="store_true")
    inspect.add_argument("--output")

    inspect_mcp = commands.add_parser("inspect-mcp-capabilities")
    inspect_mcp.add_argument("--data-root", required=True)
    inspect_mcp.add_argument("--release")
    inspect_mcp.add_argument("--snapshot-hash")
    inspect_mcp.add_argument("--duckdb-threads", type=int, default=parameter("execution.duckdb_threads"))
    inspect_mcp.add_argument("--duckdb-memory-limit", default=parameter("execution.duckdb_memory_limit"))
    inspect_mcp.add_argument("--output")

    serve_mcp = commands.add_parser("serve-mcp")
    serve_mcp.add_argument("--mode", choices=[item.value for item in DataMode], required=True)
    serve_mcp.add_argument("--data-root")
    serve_mcp.add_argument("--release")
    serve_mcp.add_argument("--snapshot-hash")
    serve_mcp.add_argument("--duckdb-threads", type=int, default=parameter("execution.duckdb_threads"))
    serve_mcp.add_argument("--duckdb-memory-limit", default=parameter("execution.duckdb_memory_limit"))

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--mode", choices=[item.value for item in DataMode], required=True)
    analyze.add_argument("--data-root")
    analyze.add_argument("--release")
    analyze.add_argument("--snapshot-hash")
    analyze.add_argument("--duckdb-threads", type=int, default=parameter("execution.duckdb_threads"))
    analyze.add_argument("--duckdb-memory-limit", default=parameter("execution.duckdb_memory_limit"))
    analyze.add_argument("--policy")
    analyze.add_argument("--ranker", choices=["deterministic", "openai"], default=None,
                         help="model ranking; deterministic is restricted to explicit frozen-data comparisons or retrieval only")
    analyze.add_argument("--defer-ranking", action="store_true")
    analyze.add_argument("--reviewed-retrieval")
    analyze.add_argument("--model-provider", default="deterministic")
    analyze.add_argument(
        "--model-id",
        help="requested model; defaults to the configured model for openai or the deterministic ranker ID",
    )
    analyze.add_argument("--api-key-env")
    analyze.add_argument("--base-url")
    analyze.add_argument("--reasoning-profile", type=_reasoning_profile_argument, help=argparse.SUPPRESS)
    analyze.add_argument(
        "--thinking-mode",
        choices=sorted(THINKING_MODES),
        default=parameter("model.ranking_thinking_mode"),
    )
    analyze.add_argument(
        "--reasoning-effort",
        choices=sorted(REASONING_EFFORTS),
        default=DEFAULT_REASONING_EFFORT,
    )
    analyze.add_argument("--allow-model-alias", action="store_true")
    analyze.add_argument("--offline", action="store_true")
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--run-id", type=_run_id_argument)
    analyze.add_argument(
        "--event-jsonl",
        help="Append structured, non-chain-of-thought progress events for a local UI.",
    )
    analyze.add_argument(
        "--tool-contract",
        choices=["semantic", "legacy"],
        default="semantic",
        help="Use staged semantic MCP tools by default; legacy is retained for ablation.",
    )
    analyze.add_argument(
        "--full-json",
        action="store_true",
        help="Print full evidence payloads; they are always saved to artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "import-evidence-artifact":
        receipt = import_legacy_document(Path(args.input), Path(args.output), field=args.field)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    if args.command == "inspect-snapshot":
        if args.release:
            damaged = damaged_parquet_files(args.data_root)
            if damaged:
                raise ValueError(
                    f"{len(damaged)} Parquet file(s) are truncated or damaged (missing the "
                    f"PAR1 header or footer), for example {', '.join(damaged[:3])}; "
                    "re-copy or re-download them before provisioning a manifest"
                )
            result = inspect_snapshot(
                args.data_root, release=args.release, full_hash=args.full_hash
            ).to_dict()
        else:
            if args.full_hash:
                raise ValueError("--full-hash is a provisioning option and requires --release")
            result = load_snapshot_manifest(args.data_root).to_dict()
        if args.output:
            _atomic_json(Path(args.output), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "inspect-mcp-capabilities":
        inspection = _verified_frozen_snapshot(
            args.data_root,
            release=args.release,
            snapshot_hash=args.snapshot_hash,
        )
        service = create_frozen_semantic_service(
            args.data_root,
            release=inspection.release,
            snapshot_sha256=inspection.manifest_sha256,
            max_threads=args.duckdb_threads,
            memory_limit=args.duckdb_memory_limit,
        )
        result = asyncio.run(service.call(SemanticTool.GET_CAPABILITIES, {})).to_dict()
        if args.output:
            _atomic_json(Path(args.output), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "serve-mcp":

        async def serve() -> None:
            mode = DataMode(args.mode)
            if mode is DataMode.FROZEN_SAFE:
                if not args.data_root:
                    raise ValueError("--data-root is required for frozen-safe MCP mode")
                inspection = _verified_frozen_snapshot(
                    args.data_root,
                    release=args.release,
                    snapshot_hash=args.snapshot_hash,
                )
                service = create_frozen_semantic_service(
                    args.data_root,
                    release=inspection.release,
                    snapshot_sha256=inspection.manifest_sha256,
                    max_threads=args.duckdb_threads,
                    memory_limit=args.duckdb_memory_limit,
                )
            else:
                service = await create_discovered_live_semantic_service(
                    release=args.release or "live"
                )
            await run_stdio(SemanticMCPDispatcher(service))

        asyncio.run(serve())
        return 0

    if args.command == "analyze":
        # Retrieval-only work requires no model. Actual prioritization defaults
        # to a model and can never silently select the evidence-count baseline.
        input_value = _read_json(Path(args.input))
        query = locus_query_from_dict(input_value, allow_analysis_scope=True)
        args.ranker = args.ranker or ("deterministic" if args.defer_ranking or
            query.task in {"describe_variants", "gene_associations"} else "openai")
        if args.model_id is None:
            args.model_id = (
                parameter("model.default_id")
                if args.ranker == "openai" else "evidence-count-ranker-v1"
            )
        if args.ranker == "openai" and args.base_url is None:
            args.base_url = parameter("model.default_base_url")
        # Local worker files can carry the server-validated retrieval scope;
        # HTTP JSON is validated separately and cannot supply this private data.
        mode = DataMode(args.mode)
        if (mode is DataMode.LIVE and query.task == "prioritize_genes"
                and not args.defer_ranking and args.ranker != "openai"):
            raise ValueError("Live gene prioritization requires model ranking; evidence-count ranking is disabled")
        reviewed = None
        if args.reviewed_retrieval:
            if args.defer_ranking:
                raise ValueError("cannot defer a resumed ranking")
            retrieval_path = Path(args.reviewed_retrieval)
            if retrieval_path.stat().st_size > MAX_MANIFEST_BYTES:
                # Recover the historical evidence in a separate local directory;
                # no lookup is repeated and the original artifact is unchanged.
                recovered_path = Path(args.output_dir) / "imported_retrieval" / uuid4().hex / "retrieval.json"
                import_legacy_document(retrieval_path, recovered_path, field="records")
                retrieval_path = recovered_path
            saved = read_document(retrieval_path, field="records", materialize=False, metadata_mode="summary")
            if saved.get("data_mode") != mode.value:
                raise ValueError("reviewed retrieval data mode does not match")
            saved_release = saved.get("data_release")
            if args.release not in {None, "live", saved_release}:
                raise ValueError("reviewed retrieval release does not match")
            args.release = saved_release
            reviewed = reviewed_tool_result(saved, query,
                record_iterator=saved["records"] if STORE_KEY in saved else None)
        runtime = RuntimeConfig(
            data_mode=mode,
            model_provider=(
                "deterministic" if args.ranker == "deterministic"
                else "openai-compatible" if args.model_provider == "deterministic"
                else args.model_provider
            ),
            model_id=args.model_id,
            benchmark=query.benchmark,
            offline=args.offline,
            model_base_url=args.base_url,
            thinking_mode=args.thinking_mode,
            reasoning_effort=args.reasoning_effort,
            reasoning_profile=getattr(args, "reasoning_profile", None),
        )
        runtime.validate()
        if mode is DataMode.FROZEN_SAFE:
            if not args.data_root:
                raise ValueError("--data-root is required for frozen-safe mode")
            inspection = _verified_frozen_snapshot(
                args.data_root,
                release=args.release,
                snapshot_hash=args.snapshot_hash,
            )
            args.release = inspection.release
            snapshot_hash = inspection.manifest_sha256
            provider = DuckDBFrozenOpenTargetsProvider(
                args.data_root,
                release=args.release,
                snapshot_hash=snapshot_hash,
                max_threads=args.duckdb_threads,
                memory_limit=args.duckdb_memory_limit,
            )
            if args.policy:
                policy = leakage_policy_from_dict(_read_json(Path(args.policy)), expected_mode=mode)
            else:
                policy = (
                    LeakagePolicy(mode=mode, release=args.release)
                    if query.benchmark
                    else frozen_exploratory_policy(args.release)
                )
            if query.benchmark:
                if not args.policy:
                    raise ValueError("benchmark runs require an explicit leakage policy file")
                validate_benchmark_policy(policy)
        else:
            live_release = args.release or "live"
            provider = OfficialOpenTargetsMCPProvider(
                release=live_release,
                cache_directory=Path(args.output_dir).resolve() / "request_cache",
            )
            policy = live_policy(live_release)

        if args.ranker == "deterministic":
            synthesizer = DeterministicEvidenceSynthesizer()
        else:
            if not args.api_key_env:
                raise ValueError("--api-key-env is required for an OpenAI-compatible model")
            synthesizer = OpenAICompatibleSynthesizer(
                model_id=args.model_id,
                provider=runtime.model_provider,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                require_exact_model_id=not args.allow_model_alias,
                offline=args.offline,
                thinking_mode=args.thinking_mode,
                reasoning_effort=args.reasoning_effort,
                **({"reasoning_profile": args.reasoning_profile} if getattr(args, "reasoning_profile", None) is not None else {}),
            )
        output = Path(args.output_dir).resolve()
        store = ArtifactStore(output)
        audit = AuditLogger(output / "audit.jsonl")
        current_run_id = args.run_id
        if args.event_jsonl and not current_run_id:
            raise ValueError("--event-jsonl requires --run-id")
        progress_sink = (
            JsonlProgressSink(args.event_jsonl, run_id=current_run_id) if args.event_jsonl else None
        )
        store.write_json("effective_config.json", runtime.public_dict())
        if args.tool_contract == "semantic" and reviewed is None:
            service = None
            if mode is DataMode.FROZEN_SAFE:
                modules = FrozenSemanticModules(provider)
            else:
                bootstrap_network_backend = getattr(provider.transport, "network_backend", "unknown")
                emit_progress(progress_sink, "semantic_tool_started", tool=SemanticTool.GET_CAPABILITIES.value,
                    modules=["schema.discovery"], required=True, stage="bootstrap")
                try:
                    schema = asyncio.run(OfficialSchemaNegotiator(provider).inspect())
                except (LiveProviderError, OSError, ValueError) as exc:
                    # Transport/schema discovery failed before retrieval. Preserve
                    # the normal agent/artifact path with explicitly unavailable
                    # capabilities; neither a schema nor empty biological results
                    # can be inferred from this failure.
                    error_type = ("LiveProviderError" if isinstance(exc, LiveProviderError)
                                  else "OSError" if isinstance(exc, OSError) else "ValueError")
                    if current_run_id is None:
                        current_run_id = str(uuid4())
                    service = UnavailableSemanticService(requested_release=live_release,
                        policy_sha256=sha256_json(asdict(policy)),
                        error_type=error_type)
                    failure = {**service.upstream_metadata["schema_bootstrap"], "schema_version": 1,
                        "run_id": current_run_id, "source_only": True,
                        "source_input_sha256": sha256_json(input_value)}
                    store.write_json("schema_bootstrap.json", failure)
                    diagnostic = bootstrap_failure_diagnostic(
                        exc, network_backend=bootstrap_network_backend
                    )
                    store.write_json("schema_bootstrap_diagnostic.json", {
                        **diagnostic, "run_id": current_run_id,
                        "source_input_sha256": failure["source_input_sha256"],
                    })
                    audit.log("schema_bootstrap_unavailable", **failure)
                    emit_progress(progress_sink, "semantic_tool_finished", tool=SemanticTool.GET_CAPABILITIES.value,
                        stage="bootstrap", status="unavailable", records=0, module_counts={},
                        coverage=ToolCoverage(requested=("schema.discovery",), unavailable=("schema.discovery",)).to_dict(),
                        warnings=[SCHEMA_UNAVAILABLE_MESSAGE])
                else:
                    emit_progress(progress_sink, "semantic_tool_finished", tool=SemanticTool.GET_CAPABILITIES.value,
                        stage="bootstrap", status="success", records=0, module_counts={},
                        coverage=ToolCoverage(requested=("schema.discovery",), completed=("schema.discovery",)).to_dict())
                    modules = _live_analysis_modules(provider, schema)
            if service is None:
                blocked_modules = blocked_modules_for_policy(
                    modules.adapters(), policy, benchmark=query.benchmark
                )
                service = modules.service(
                    policy_sha256=sha256_json(asdict(policy)),
                    blocked_modules=blocked_modules,
                )
            gateway = SemanticGatewayAdapter(
                service,
                policy,
                artifacts=store,
                audit=audit,
                progress_sink=progress_sink,
            )
        else:
            gateway = GWASToolGateway(provider, policy, artifacts=store, audit=audit)
        result = asyncio.run(
            GWASMindAgent(
                gateway=gateway,
                synthesizer=synthesizer,
                artifacts=store,
                audit=audit,
                progress_sink=progress_sink,
            ).run(
                query,
                run_id=current_run_id,
                defer_ranking=args.defer_ranking,
                retrieved=reviewed,
            )
        )
        printed = result.to_dict() if args.full_json else result.to_compact_dict()
        print(json.dumps(printed, ensure_ascii=False, indent=2, sort_keys=True))
        if result.status in {
            DecisionStatus.PRIORITIZED,
            DecisionStatus.EVIDENCE_RETRIEVED,
        }:
            return 0
        if result.status in {
            DecisionStatus.INVALID_INPUT,
            DecisionStatus.NO_GWAS_SIGNAL,
        }:
            return 2
        if result.status is DecisionStatus.SERVICE_UNAVAILABLE:
            return 4
        return 3
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
