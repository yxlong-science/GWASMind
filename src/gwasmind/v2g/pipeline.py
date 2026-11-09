"""Orchestrate locus set, gene window, retrieval and evidence-card classification.

The pipeline never ranks genes; ranking is produced by the model from the cards.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..configuration import parameter
from .evidence import classify_gene_evidence, strip_private
from .gene_window import HttpGet, fetch_gene_window
from .ledger import region_summary, semantic_records
from .locus_set import (
    build_locus_variant_set,
    ld_anchor_candidates,
    ld_proxies_from_rows,
    orientation_aliases,
)
from .models import LDStatus, V2GRequest, V2GResult, parse_variant_id
from .retrieval import GraphQLExecutor, V2GRetriever

LDProxyFetch = Callable[[Mapping[str, Any], str, float, int], Sequence[Mapping[str, Any]]]
MAX_LD_ANCHOR_ATTEMPTS = 5
# Reasons that describe the anchor itself; another anchor may still succeed.
ANCHOR_SPECIFIC_LD_FAILURES = frozenset({"variant_not_in_reference", "not_found_in_queried_reference",
                                         "monomorphic_in_selected_population", "source_identity_unverified",
                                         "population_alleles_ambiguous", "reference_coordinate_unverified",
                                         "reference_warning", "missing_rsid"})


def gene_centric_enabled() -> bool:
    return parameter("v2g.pipeline") == "gene_centric"


def ldlink_proxy_fetcher(provider: Any) -> LDProxyFetch:
    """Adapt an LDLinkReferenceProvider; its serial lock, cache and retries still apply."""

    def fetch(anchor: Mapping[str, Any], population: str, threshold: float, window_bp: int):
        chromosome, position, reference, alternate = parse_variant_id(anchor["variant_id"])
        variant = {"rsid": anchor["rsid"], "source_variant_id": anchor["variant_id"], "chromosome": chromosome,
                   "position": position, "alleles": [reference, alternate],
                   "reference_allele": reference, "alternate_allele": alternate,
                   "canonical_variant_id": anchor["variant_id"]}
        return provider.proxies(variant, population, threshold=threshold, window_bp=window_bp)

    fetch.proxy_lookup_issues = getattr(provider, "proxy_lookup_issues", {})
    return fetch


async def _anchor_identities(executor: GraphQLExecutor, anchors: Sequence[str]
                             ) -> tuple[dict[str, tuple[str, str | None] | None], str | None]:
    """Resolve each anchor to its Open Targets orientation and first rsID.

    Both REF/ALT orders of an SNV are queried because uploads may swap alleles.
    """

    queried = [(anchor, variant_id) for anchor in anchors for variant_id in (anchor, *orientation_aliases(anchor))]
    body = " ".join(f'v{index}: variant(variantId:"{variant}"){{ id rsIds }}'
                    for index, (_, variant) in enumerate(queried))
    try:
        data = await executor("query V2GAnchors{ " + body + " }", {})
    except Exception as exc:  # noqa: BLE001 - LD is then reported as failed, not the whole locus
        return {anchor: None for anchor in anchors}, type(exc).__name__
    result: dict[str, tuple[str, str | None] | None] = {anchor: None for anchor in anchors}
    for index, (anchor, variant_id) in enumerate(queried):
        record = data.get(f"v{index}")
        if not isinstance(record, Mapping) or result[anchor] is not None:
            continue
        rsids = [value for value in (record.get("rsIds") or []) if isinstance(value, str) and value.startswith("rs")]
        result[anchor] = (variant_id, rsids[0] if rsids else None)
    return result, None


async def _ld_expansion(request: V2GRequest, executor: GraphQLExecutor, ld_fetch: LDProxyFetch | None):
    if not request.use_ld or ld_fetch is None:
        return [], LDStatus.NOT_REQUESTED.value, None, [], None
    # With user-selected variants each independent selection anchors its own LD
    # block; otherwise (e.g. a supplied credible set) one usable anchor suffices.
    wanted = parameter("v2g.ld.max_anchors") if request.selected_variants else 1
    anchors = list(ld_anchor_candidates(request.lead_variant_id, request.credible_set,
                                        request.selected_variants))[:wanted + MAX_LD_ANCHOR_ATTEMPTS]
    identities, lookup_failure = await _anchor_identities(executor, anchors)
    if lookup_failure:
        return [], LDStatus.FAILED.value, None, [f"LD anchor rsID lookup failed ({lookup_failure})."], None
    lead_identity = identities.get(request.lead_variant_id)
    lead_rsid = lead_identity[1] if lead_identity else None
    chromosome = parse_variant_id(request.lead_variant_id)[0]
    r2_core, r2_extended = parameter("v2g.ld.r2_core"), parameter("v2g.ld.r2_extended")
    issues: list[str] = []
    collected: list[dict[str, Any]] = []
    covered: set[str] = set()
    used: list[str] = []
    anchor_failures = non_snv_rows = malformed_rows = 0
    stopped_reason = None
    scope_incomplete = False
    for anchor in anchors:
        if len(used) >= wanted:
            break
        if request.selected_variants and {anchor, *orientation_aliases(anchor)} & covered:
            continue
        identity = identities.get(anchor)
        if not identity or not identity[1]:
            anchor_failures += 1
            issues.append(f"LD anchor {anchor} has no rsID in Open Targets.")
            continue
        resolved_id, rsid = identity
        try:
            rows = await asyncio.to_thread(ld_fetch, {"variant_id": resolved_id, "rsid": rsid}, request.population,
                                           r2_extended, parameter("v2g.ld.window_bp"))
        except Exception as exc:  # noqa: BLE001 - classify by the provider's reason code only
            reason = getattr(exc, "reason_code", None) or type(exc).__name__
            if reason in ANCHOR_SPECIFIC_LD_FAILURES:
                anchor_failures += 1
                issues.append(f"LD anchor {anchor} could not be used ({reason}).")
                continue
            stopped_reason = reason
            issues.append(f"LD expansion stopped ({reason}).")
            break
        audit = getattr(ld_fetch, "proxy_lookup_issues", {}).get((resolved_id, request.population), {})
        if audit.get("requested_scope_complete") is False:
            scope_incomplete = True
            if audit.get("query_build") == "GRCh37":
                issues.append("LD proxies were recovered through GRCh37 and verified in GRCh38; "
                              "complete coverage of the GRCh38 window is not established.")
            else:
                issues.append("Some in-scope LD records could not be verified; the valid records were retained.")
        proxies, skipped = ld_proxies_from_rows(rows, chromosome=chromosome, min_r2=r2_extended)
        non_snv_rows += skipped["not_biallelic_snv"]
        malformed_rows += skipped["malformed"]
        collected.extend(proxies)
        used.append(anchor)
        covered.update({anchor, *orientation_aliases(anchor)})
        covered.update(value for proxy in proxies if proxy["r2"] >= r2_core
                       for value in (proxy["variant_id"], *proxy.get("aliases", ())))
    if non_snv_rows:
        issues.append(f"{non_snv_rows} LD proxies that are not biallelic SNVs (indels or multi-allelic sites) "
                      "were excluded by design.")
    if malformed_rows:
        issues.append(f"{malformed_rows} LD rows were malformed and could not be used.")
    if used and used[0] != request.lead_variant_id:
        issues.append(f"The lead is not usable in the LD reference; variant {used[0]} was the first LD anchor.")
    if not used:
        status = (LDStatus.FAILED.value if stopped_reason or not anchor_failures
                  else LDStatus.ANCHOR_NOT_IN_REFERENCE.value)
        return [], status, None, issues, lead_rsid
    status = LDStatus.PARTIAL.value if stopped_reason or malformed_rows or scope_incomplete else LDStatus.COMPLETE.value
    return collected, status, used[0], issues, lead_rsid


async def run_v2g(request: V2GRequest, *, executor: GraphQLExecutor, ld_fetch: LDProxyFetch | None = None,
                  http_get: HttpGet | None = None, progress: Callable[[str], None] | None = None,
                  progress_sink=None,
                  precomputed_ld: tuple[Sequence[Mapping[str, Any]], str] | None = None) -> V2GResult:
    """Run the gene-centric pipeline for one locus; every window gene receives an evidence card.

    ``precomputed_ld`` supplies ``(proxies, ld_status)`` from an LD expansion already
    performed upstream (the ranking LD step); no LD requests are then made here.
    """

    notify = progress or (lambda message: None)
    from .progress import RetrievalProgress
    activity = RetrievalProgress(progress_sink)
    parse_variant_id(request.lead_variant_id)
    if precomputed_ld is not None:
        proxies, ld_status = list(precomputed_ld[0]), precomputed_ld[1]
        anchor, ld_issues, lead_rsid = request.lead_variant_id, [], None
    else:
        notify("Building the locus variant set (population LD expansion)...")
        proxies, ld_status, anchor, ld_issues, lead_rsid = await _ld_expansion(request, executor, ld_fetch)
    locus = build_locus_variant_set(request.lead_variant_id, credible_set=request.credible_set, ld_proxies=proxies,
                                    ld_status=ld_status, ld_anchor_variant_id=anchor, issues=ld_issues,
                                    lead_rsid=lead_rsid, selected_variants=request.selected_variants)
    notify(f"Locus variant set ready: {len(locus.variants)} variants. Collecting genes within the lead window...")
    async with activity.operation("candidates"):
        window = await asyncio.to_thread(fetch_gene_window, locus, explicit_candidates=request.explicit_candidates,
                                         http_get=http_get)
        activity.update(candidate_count=len(window.genes))
    notify(f"Retrieving gene-centric evidence for {len(window.genes)} genes...")
    retriever = V2GRetriever(executor, progress=activity)
    bundle = await retriever.retrieve(locus=locus, gene_ids=[gene.gene_id for gene in window.genes],
                                      trait_efo_ids=request.trait_efo_ids,
                                      phenotype_scope=request.phenotype_scope)
    notify("Classifying evidence into gene cards...")
    from .capacity import check_channel_capacity, EvidenceCapacityError
    try:
        check_channel_capacity(bundle)
    except EvidenceCapacityError as exc:
        from dataclasses import asdict
        exc.retrieval_snapshot = {"request": asdict(request), "locus": locus.to_dict(),
            "window": window.to_dict(), "raw_responses": bundle.raw, "coverage": bundle.coverage}
        raise
    async with activity.operation("cards"):
        classified = classify_gene_evidence(locus, window, bundle, context_terms=request.context_terms,
                                            biological_context=request.biological_context)
        ledger = [region_summary(locus, window, parameter("analysis.local_gene_window_bp")),
                  *semantic_records(classified, locus, bundle)]
        genes = strip_private(classified)
        activity.update(records=len(ledger), candidate_count=len(genes))
    retrieval_status = {
        "window": window.status,
        "ld": locus.ld_status,
        "weight_basis": locus.weight_basis,
        "coloc": bundle.coloc_status,
        "association": bundle.association_status,
        "targets_failed": str(sum(1 for status in bundle.target_status.values()
                                  if status.get("target") == "failed")),
        "variants_resolved": str(sum(1 for record in bundle.variants.values() if record)),
    }
    issues = [*locus.issues, *window.issues, *bundle.issues]
    return V2GResult(request=request, locus=locus, window=window, genes=tuple(genes),
                     retrieval_status=retrieval_status, raw_responses=bundle.raw, issues=issues, ledger=ledger,
                     coverage=bundle.coverage, display_genes=tuple(classified))
