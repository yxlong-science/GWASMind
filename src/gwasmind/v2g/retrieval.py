"""Batched Open Targets GraphQL retrieval for gene-centric V2G evidence.

A failed request marks only the affected genes/variants and channels as failed;
the remaining evidence is still retrieved. E2G pagination exhausts each variant
connection; bounded per-gene channels report any truncation as partial.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..configuration import parameter
from .models import ChannelStatus, LocusVariantSet

GraphQLExecutor = Callable[[str, Mapping[str, Any]], Awaitable[Mapping[str, Any]]]


_QTL_ROW = ("studyLocusId studyType isTransQtl variant{ id rsIds } "
            "pValueMantissa pValueExponent beta finemappingMethod confidence qualityControls "
            "study{ id projectId pubmedId publicationFirstAuthor publicationDate "
            "biosample{ biosampleId biosampleName } } "
            "locus(variantIds:$vids){ rows{ variant{ id } posteriorProbability "
            "pValueMantissa pValueExponent beta standardError } }")
_QTL_OVERLAP_ROW = _QTL_ROW.replace("biosample{ biosampleId biosampleName } } ",
                                    "biosample{ biosampleId biosampleName } target{ id } } ", 1)
assert _QTL_OVERLAP_ROW != _QTL_ROW
_EXPRESSION_ROW = ("datasourceId datatypeId median unit specificity_score "
                   "tissueBiosample{ biosampleId biosampleName } celltypeBiosample{ biosampleId biosampleName }")
_TARGET_FIELDS = (
    "id approvedSymbol approvedName biotype "
    "symbolSynonyms{ label } nameSynonyms{ label } obsoleteSymbols{ label } obsoleteNames{ label } "
    "proteinIds{ id source } transcriptIds "
    "genomicLocation{ chromosome start end strand } canonicalTranscript{ id start end strand } "
    "geneticConstraint{ constraintType oe oeUpper score } functionDescriptions "
    "geneOntology{ aspect term{ id label } } pathways{ pathway pathwayId topLevelTerm } "
    f"credibleSets(page:{{index:0,size:$qsize}}){{ count rows{{ {_QTL_ROW} }} }} "
    f"baselineExpression(page:{{index:0,size:$esize}}){{ count rows{{ {_EXPRESSION_ROW} }} }}"
)
_E2G_ROW = ("target{ id } score datasourceId intervalType biosampleName "
            "biosample{ biosampleId biosampleName } chromosome start end distanceToTss "
            "resourceScore{ name value } studyId pmid")
_VARIANT_FIELDS = (
    "id rsIds "
    "transcriptConsequences{ target{ id } isEnsemblCanonical impact distanceFromTss distanceFromFootprint "
    "variantConsequences{ id label } } "
    f"enhancerToGenes(page:{{index:0,size:$e2gsize}}){{ count rows{{ {_E2G_ROW} }} }}"
)


class _OverlapSupplementFailed(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


class _OverlapSupplementInvalid(_OverlapSupplementFailed):
    def __init__(self, detail: str):
        super().__init__("overlap_supplement_invalid", detail)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _chunks(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _gql_string(value: str) -> str:
    return json.dumps(value)


def _unique_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate complete returned observations without merging their measurements."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            unique.setdefault(_digest(row), dict(row))
    return list(unique.values())


@dataclass(slots=True)
class RetrievalBundle:
    coloc_background: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    frozen_molqtl: dict[str, dict[str, Any]] = field(default_factory=dict)
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    target_status: dict[str, dict[str, str]] = field(default_factory=dict)
    variants: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    variant_status: dict[str, dict[str, str]] = field(default_factory=dict)
    gwas_credible_sets: list[dict[str, Any]] = field(default_factory=list)
    coloc_status: str = ChannelStatus.NOT_REQUESTED.value
    associations: dict[str, dict[str, Any]] = field(default_factory=dict)
    association_records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    association_status_by_disease: dict[str, str] = field(default_factory=dict)
    association_disease: dict[str, Any] | None = None
    association_status: str = ChannelStatus.NOT_REQUESTED.value
    coverage: dict[str, Any] = field(default_factory=dict)
    raw: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


class V2GRetriever:
    def __init__(self, executor: GraphQLExecutor, *, progress=None):
        from .progress import RetrievalProgress
        self._execute_graphql = executor
        self.progress = progress or RetrievalProgress()

    async def _run(self, bundle: RetrievalBundle, operation: str, query: str,
                   variables: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            data = await self._execute_graphql(query, variables)
        except Exception:
            self.progress.request_finished(success=False)
            raise
        self.progress.request_finished(success=True)
        bundle.raw.append({"operation": operation, "variables_sha256": _digest(variables),
                           "query_sha256": hashlib.sha256(query.encode()).hexdigest(), "data": data})
        self.progress.update(completed_pages=len(bundle.raw))
        return data

    async def retrieve(self, *, locus: LocusVariantSet, gene_ids: Sequence[str],
                       trait_efo_ids: Sequence[str] = (), phenotype_scope: Mapping[str, Any] | None = None) -> RetrievalBundle:
        bundle = RetrievalBundle()
        if phenotype_scope:
            bundle.coverage["phenotype_scope"] = dict(phenotype_scope)
        bundle.coverage["phenotype_scope_ids"] = list(trait_efo_ids)
        query_ids = list(locus.query_ids())
        async with self.progress.operation("gene_profiles", completed_count=0, total_count=len(gene_ids), unit="genes"):
            await self._targets(bundle, list(gene_ids), query_ids)
            self.progress.update(completed_count=len(bundle.target_status),
                failed_items=sum("failed" in s.values() for s in bundle.target_status.values()),
                partial_items=sum("partial" in s.values() for s in bundle.target_status.values()))
        async with self.progress.operation("variants", completed_count=0, total_count=len(query_ids), unit="variants"):
            await self._variants(bundle, query_ids)
            self.progress.update(completed_count=len(bundle.variant_status),
                failed_items=sum("failed" in s.values() for s in bundle.variant_status.values()),
                partial_items=sum("partial" in s.values() for s in bundle.variant_status.values()))
        async with self.progress.operation("colocalisation", unit="credible_sets"):
            await self._colocalisation(bundle, locus, trait_efo_ids)
            self.progress.update(failed_items=bundle.coverage.get("colocalisation", {}).get("failed_molqtl_count", 0))
        async with self.progress.operation("associations", completed_count=0, total_count=len(set(trait_efo_ids)), unit="phenotypes"):
            await self._associations(bundle, list(gene_ids), trait_efo_ids)
            self.progress.update(completed_count=len(bundle.association_status_by_disease), failed_items=sum(
                state == "failed" for state in bundle.association_status_by_disease.values()))
        return bundle

    async def _targets(self, bundle: RetrievalBundle, gene_ids: list[str], query_ids: list[str]) -> None:
        qsize = parameter("v2g.qtl.page_size")
        expression_page_size = parameter("v2g.expression.page_size")
        variables = {"vids": query_ids, "qsize": qsize, "esize": expression_page_size}
        header = "query V2GTargets($vids:[String!]!,$qsize:Int!,$esize:Int!){ "
        pending = list(_chunks(gene_ids, parameter("v2g.retrieval.gene_batch_size")))
        while pending:
            batch = pending.pop(0)
            body = " ".join(f"g{index}: target(ensemblId:{_gql_string(gene)}){{ {_TARGET_FIELDS} }}"
                            for index, gene in enumerate(batch))
            try:
                data = await self._run(bundle, "targets", header + body + " }", variables)
            except Exception as exc:  # noqa: BLE001 - one failed batch must not hide other genes
                if len(batch) > 1:
                    # The server's query-cost limit scales with the variant set; retry genes singly.
                    pending[:0] = [[gene] for gene in batch]
                    continue
                bundle.issues.append(f"Target lookup for {batch[0]} failed ({type(exc).__name__}).")
                bundle.target_status[batch[0]] = {"target": ChannelStatus.FAILED.value,
                                                  "qtl": ChannelStatus.FAILED.value,
                                                  "expression": ChannelStatus.FAILED.value}
                continue
            for index, gene in enumerate(batch):
                target = data.get(f"g{index}")
                if not isinstance(target, Mapping):
                    bundle.target_status[gene] = {"target": ChannelStatus.NOT_OBSERVED.value,
                                                  "qtl": ChannelStatus.NOT_OBSERVED.value,
                                                  "expression": ChannelStatus.NOT_OBSERVED.value}
                    continue
                target = dict(target)
                status = {"target": ChannelStatus.OBSERVED.value}
                status["qtl"] = await self._complete_connection(
                    bundle, target, gene, "credibleSets", _QTL_ROW, qsize,
                    parameter("v2g.qtl.max_credible_sets_per_gene"), query_ids)
                status["expression"] = await self._complete_connection(
                    bundle, target, gene, "baselineExpression", _EXPRESSION_ROW, expression_page_size,
                    parameter("v2g.expression.max_rows_per_gene"), query_ids)
                bundle.targets[gene] = target
                bundle.target_status[gene] = status
                self.progress.update(completed_count=len(bundle.target_status))
        truncated = [gene for gene, coverage in bundle.coverage.get("targets", {}).items()
                     if coverage.get("credibleSets", {}).get("stop_reason") == "row_limit"]
        await self._qtl_overlap_supplement(bundle, truncated, query_ids)

    async def _qtl_overlap_supplement(self, bundle: RetrievalBundle, genes: list[str], query_ids: list[str]) -> None:
        """A per-gene row limit must not hide credible sets that contain an analysis variant."""
        if not genes or not query_ids:
            return
        size = parameter("v2g.qtl.page_size")
        query = ("query V2GQtlOverlap($vids:[String!]!,$types:[StudyTypeEnum!],$index:Int!,$size:Int!){ "
                 "credibleSets(variantIds:$vids, studyTypes:$types, page:{index:$index,size:$size}){ count rows{ "
                 f"{_QTL_OVERLAP_ROW} }} }} }}")
        rows, count, page, pages_seen = [], None, 0, set()

        def invalid(code):
            raise _OverlapSupplementInvalid(code)

        try:
            while count is None or len(rows) < count:
                try:
                    data = await self._run(bundle, "qtl_overlap_supplement", query, {
                        "vids": query_ids, "types": list(parameter("v2g.qtl.study_types")), "index": page, "size": size})
                except Exception as exc:  # noqa: BLE001 - a transport failure is a retryable gap, not a budget limit
                    raise _OverlapSupplementFailed("request_failed", type(exc).__name__) from exc
                connection = data.get("credibleSets") if isinstance(data, Mapping) else None
                reported = connection.get("count") if isinstance(connection, Mapping) else None
                chunk = connection.get("rows") if isinstance(connection, Mapping) else None
                if type(reported) is not int or reported < 0 or not isinstance(chunk, list):
                    invalid("malformed_connection")
                if count is not None and reported != count:
                    invalid("reported_count_changed")
                count = reported
                if not chunk and len(rows) < count:
                    invalid("pagination_ended_early")
                digest = _digest(chunk)
                if chunk and digest in pages_seen:
                    invalid("repeated_page")
                pages_seen.add(digest)
                rows.extend(chunk)
                page += 1
            if len(rows) != count:
                invalid("row_count_mismatch")
            identifiers = [row.get("studyLocusId") if isinstance(row, Mapping) else None for row in rows]
            if any(not isinstance(item, str) or not item for item in identifiers):
                invalid("credible_set_identity_missing")
            if len(set(identifiers)) != len(identifiers):
                invalid("duplicate_credible_set")
            by_gene: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                study = dict(row.get("study") or {})
                gene = (study.pop("target", None) or {}).get("id")
                if not isinstance(gene, str) or not gene:
                    invalid("gene_attribution_missing")
                if gene in genes:
                    by_gene.setdefault(gene, []).append({**row, "study": study})
            for gene in genes:
                known = {row.get("studyLocusId"): row for row in bundle.targets[gene]["credibleSets"]["rows"]}
                if any(row["studyLocusId"] in known and _digest(known[row["studyLocusId"]]) != _digest(row)
                       for row in by_gene.get(gene, [])):
                    invalid("conflicting_credible_set")
        except _OverlapSupplementFailed as failure:
            bundle.issues.append(f"QTL overlap supplement for row-limited genes failed ({failure.detail}); "
                                 "their QTL channels remain incomplete.")
            for gene in genes:
                bundle.coverage["targets"][gene]["credibleSets"].update(
                    complete=False, stop_reason=failure.reason,
                    overlap_supplement={"status": "failed", "reason": failure.reason, "detail": failure.detail,
                                        "pages_received": page, "rows_received": len(rows), "reported_count": count})
            return
        for gene in genes:
            connection = bundle.targets[gene]["credibleSets"]
            known = {row.get("studyLocusId") for row in connection["rows"]}
            added = [row for row in by_gene.get(gene, []) if row.get("studyLocusId") not in known]
            connection["rows"] = [*connection["rows"], *added]
            # Complete for the analysis-variant scope only; the gene's other credible sets were not all read.
            bundle.coverage["targets"][gene]["credibleSets"].update(
                complete=True, stop_reason=None, completion_scope="analysis_variant_overlap", gene_rows_complete=False,
                overlap_supplement={"status": "complete", "reported_count": count, "pages_received": page,
                                    "overlapping_rows": len(by_gene.get(gene, [])), "added_rows": len(added)})
            bundle.target_status[gene]["qtl"] = (ChannelStatus.OBSERVED.value if connection["rows"]
                                                 else ChannelStatus.NOT_OBSERVED.value)

    async def _complete_connection(self, bundle: RetrievalBundle, target: dict[str, Any], gene: str,
                                   name: str, row_fields: str, page_size: int, limit: int,
                                   query_ids: list[str]) -> str:
        connection = target.get(name)
        if not isinstance(connection, Mapping):
            return ChannelStatus.NOT_OBSERVED.value
        initial_rows = list(connection.get("rows") or [])
        rows = _unique_rows(initial_rows)
        count = connection.get("count")
        count = count if isinstance(count, int) and count >= 0 else len(rows)
        page = 1
        # The baselineExpression resolver passes page.index directly to SQL
        # LIMIT(offset, size); other target connections use a page number.
        offset = len(initial_rows)
        complete = True
        stop_reason = None
        uses_variants = "$vids" in row_fields
        declarations = "$id:String!,$index:Int!,$size:Int!" + (",$vids:[String!]!" if uses_variants else "")
        # The provider count is before deduplication. Consume raw row offsets
        # while retaining each distinct observation only once.
        while offset < count and len(rows) < limit:
            query = (f"query V2GTargetPage({declarations}){{ "
                     f"target(ensemblId:$id){{ {name}(page:{{index:$index,size:$size}}){{ count rows{{ {row_fields} }} }} }} }}")
            variables = {"id": gene, "index": offset if name == "baselineExpression" else page,
                         "size": page_size, **({"vids": query_ids} if uses_variants else {})}
            try:
                data = await self._run(bundle, f"target_{name}_page", query, variables)
            except Exception as exc:  # noqa: BLE001
                bundle.issues.append(f"{name} continuation for {gene} failed ({type(exc).__name__}).")
                complete = False
                stop_reason = "request_failed"
                break
            continuation = (((data.get("target") or {}).get(name)) or {})
            reported_count = continuation.get("count")
            if isinstance(reported_count, int) and reported_count != count:
                bundle.issues.append(f"{name} continuation for {gene} changed its reported count.")
                complete = False
                stop_reason = "reported_count_changed"
                break
            chunk = continuation.get("rows") or []
            if not chunk:
                complete = False
                stop_reason = "unexpected_empty_page"
                break
            offset += len(chunk)
            previous = len(rows)
            rows = _unique_rows([*rows, *chunk])
            if len(rows) == previous:
                bundle.issues.append(f"{name} continuation for {gene} repeated previously returned rows.")
                complete = False
                stop_reason = "repeated_page"
                break
            page += 1
        target[name] = {"count": count, "rows": rows[:limit]}
        bundle.coverage.setdefault("targets", {}).setdefault(gene, {})[name] = {
            "reported_count": count, "unique_rows": len(rows[:limit]),
            "received_rows": offset, "duplicate_rows": offset - len(rows),
            "complete": complete and offset == count and len(rows) <= limit,
            "stop_reason": stop_reason or ("row_limit" if offset < count or len(rows) > limit else
                "count_mismatch" if offset != count else None),
            "pagination": "row_offset" if name == "baselineExpression" else "page_number",
        }
        if not bundle.coverage["targets"][gene][name]["complete"]:
            return ChannelStatus.PARTIAL.value
        return ChannelStatus.OBSERVED.value if rows else ChannelStatus.NOT_OBSERVED.value

    async def _variants(self, bundle: RetrievalBundle, query_ids: list[str]) -> None:
        e2g_size = parameter("v2g.e2g.page_size")
        header = "query V2GVariants($e2gsize:Int!){ "
        pending = list(_chunks(query_ids, parameter("v2g.retrieval.variant_batch_size")))
        while pending:
            batch = pending.pop(0)
            body = " ".join(f"v{index}: variant(variantId:{_gql_string(variant)}){{ {_VARIANT_FIELDS} }}"
                            for index, variant in enumerate(batch))
            try:
                data = await self._run(bundle, "variants", header + body + " }", {"e2gsize": e2g_size})
            except Exception as exc:  # noqa: BLE001
                if len(batch) > 1:
                    pending[0:0] = [[variant] for variant in batch]
                    continue
                bundle.issues.append(f"Variant batch failed ({type(exc).__name__}).")
                for variant in batch:
                    bundle.variant_status[variant] = {"consequence": ChannelStatus.FAILED.value,
                                                      "e2g": ChannelStatus.FAILED.value}
                continue
            for index, variant_id in enumerate(batch):
                variant = data.get(f"v{index}")
                if not isinstance(variant, Mapping):
                    bundle.variants[variant_id] = None
                    bundle.variant_status[variant_id] = {"consequence": ChannelStatus.NOT_OBSERVED.value,
                                                         "e2g": ChannelStatus.NOT_OBSERVED.value}
                    continue
                variant = dict(variant)
                e2g_status = await self._complete_e2g(bundle, variant_id, variant, e2g_size)
                bundle.variants[variant_id] = variant
                bundle.variant_status[variant_id] = {"consequence": ChannelStatus.OBSERVED.value, "e2g": e2g_status}
                self.progress.update(completed_count=len(bundle.variant_status))

    async def _complete_e2g(self, bundle: RetrievalBundle, variant_id: str,
                            variant: dict[str, Any], page_size: int) -> str:
        """Count provider rows before deduplication, with no total-row cutoff.

        Raw pages remain in bundle.raw. Identical annotations may occur in
        different pages; they do not create a false coverage gap. A repeated
        whole page or a changing count cannot establish complete retrieval.
        """
        connection = variant.get("enhancerToGenes")
        count = connection.get("count") if isinstance(connection, Mapping) else None
        valid_count = isinstance(count, int) and not isinstance(count, bool) and count >= 0
        rows = connection.get("rows") if isinstance(connection, Mapping) else None
        valid_rows = isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows)
        stop_reason = None if valid_count and valid_rows else "invalid_connection"
        raw_rows = list(rows) if valid_rows else []
        seen_pages = {_digest(raw_rows)} if raw_rows else set()
        page = 1
        while not stop_reason and len(raw_rows) < count:
            query = ("query V2GE2GPage($id:String!,$index:Int!,$size:Int!){ variant(variantId:$id){ "
                     f"id enhancerToGenes(page:{{index:$index,size:$size}}){{ count rows{{ {_E2G_ROW} }} }} }} }}")
            try:
                data = await self._run(bundle, "variant_e2g_page", query,
                                       {"id": variant_id, "index": page, "size": page_size})
            except Exception as exc:  # noqa: BLE001
                bundle.issues.append(f"E2G continuation for {variant_id} failed ({type(exc).__name__}).")
                stop_reason = "request_failed"
                break
            returned = data.get("variant")
            continuation = returned.get("enhancerToGenes") if isinstance(returned, Mapping) else None
            if not isinstance(continuation, Mapping):
                stop_reason = "invalid_connection"
                break
            if returned.get("id") is not None and returned["id"] != variant.get("id"):
                stop_reason = "variant_identity_mismatch"
                break
            if type(continuation.get("count")) is not int or continuation["count"] != count:
                bundle.issues.append(f"E2G continuation for {variant_id} changed its reported count.")
                stop_reason = "count_changed"
                break
            chunk = continuation.get("rows")
            if not isinstance(chunk, list) or not all(isinstance(row, Mapping) for row in chunk):
                stop_reason = "invalid_rows"
                break
            if not chunk:
                stop_reason = "empty_page"
                break
            page_hash = _digest(chunk)
            if page_hash in seen_pages:
                bundle.issues.append(f"E2G continuation for {variant_id} repeated previously returned rows.")
                stop_reason = "repeated_page"
                break
            seen_pages.add(page_hash)
            raw_rows.extend(chunk)
            page += 1
        if not stop_reason and len(raw_rows) != count:
            stop_reason = "count_mismatch"
        unique = _unique_rows(raw_rows)
        variant["enhancerToGenes"] = {"count": count, "rows": unique}
        bundle.coverage.setdefault("variants", {}).setdefault(variant_id, {})["enhancerToGenes"] = {
            "reported_count": count, "received_rows": len(raw_rows), "unique_rows": len(unique),
            "duplicate_rows": len(raw_rows) - len(unique), "complete": stop_reason is None,
            "stop_reason": stop_reason, "pagination": "page_number",
        }
        if stop_reason:
            return ChannelStatus.PARTIAL.value
        return ChannelStatus.OBSERVED.value if unique else ChannelStatus.NOT_OBSERVED.value

    async def _colocalisation(self, bundle: RetrievalBundle, locus: LocusVariantSet,
                              trait_efo_ids: Sequence[str]) -> None:
        from .colocalisation import retrieve_frozen_colocalisation

        await retrieve_frozen_colocalisation(self, bundle, locus, trait_efo_ids)

    async def _associations(self, bundle: RetrievalBundle, gene_ids: list[str],
                            trait_efo_ids: Sequence[str]) -> None:
        efos = list(dict.fromkeys(value for value in trait_efo_ids if value))
        if not efos or not gene_ids:
            bundle.association_status = ChannelStatus.NOT_REQUESTED.value
            return
        query = ("query V2GAssociations($efo:String!,$genes:[String!]!,$size:Int!){ disease(efoId:$efo){ id name "
                 "associatedTargets(Bs:$genes, page:{index:0,size:$size}){ count rows{ target{ id } score "
                 "datatypeScores{ id score } datasourceScores{ id score } } } } }")
        for efo in efos:
            try:
                data = await self._run(bundle, "associations", query,
                                       {"efo": efo, "genes": gene_ids, "size": len(gene_ids)})
            except Exception as exc:  # noqa: BLE001
                bundle.issues.append(f"Target-disease association lookup for {efo} failed ({type(exc).__name__}).")
                bundle.association_status_by_disease[efo] = ChannelStatus.FAILED.value
                continue
            disease = data.get("disease")
            if not isinstance(disease, Mapping):
                bundle.association_status_by_disease[efo] = ChannelStatus.NOT_OBSERVED.value
                continue
            if disease.get("id") != efo:
                bundle.issues.append(f"Target-disease association lookup for {efo} returned another disease identity.")
                bundle.association_status_by_disease[efo] = ChannelStatus.FAILED.value
                continue
            identity = {"id": disease.get("id"), "name": disease.get("name")}
            if efo == efos[0]:
                bundle.association_disease = identity
            rows = _unique_rows((disease.get("associatedTargets") or {}).get("rows") or [])
            for row in rows:
                gene = (row.get("target") or {}).get("id")
                if gene in gene_ids:
                    bundle.association_records.setdefault(gene, []).append({"disease": identity, "association": row})
                    if efo == efos[0]:
                        bundle.associations[gene] = dict(row)
            bundle.association_status_by_disease[efo] = ChannelStatus.OBSERVED.value
            self.progress.update(completed_count=len(bundle.association_status_by_disease))
        states = set(bundle.association_status_by_disease.values())
        bundle.association_status = (ChannelStatus.FAILED.value if states == {ChannelStatus.FAILED.value}
                                     else ChannelStatus.PARTIAL.value if ChannelStatus.FAILED.value in states
                                     else ChannelStatus.OBSERVED.value if bundle.association_records
                                     else ChannelStatus.NOT_OBSERVED.value)


async def resolve_trait_efo_ids(executor: GraphQLExecutor, trait: str) -> tuple[str, ...]:
    """Resolve an exact label or a verified disease synonym; never trust search order."""

    if not trait or not trait.strip():
        return ()
    query = ('query V2GTraitSearch($q:String!){ search(queryString:$q, entityNames:["disease"], '
             'page:{index:0,size:10}){ hits{ id name } } }')
    try:
        data = await executor(query, {"q": trait.strip()[:200]})
    except Exception:  # noqa: BLE001 - an unresolved trait only disables trait-matched channels
        return ()
    hits = ((data.get("search") or {}).get("hits")) or []
    def label(value: Any) -> str:
        return " ".join(str(value or "").casefold().split())

    wanted = label(trait)
    exact = {str(hit["id"]) for hit in hits if isinstance(hit, Mapping) and hit.get("id")
             and (label(hit.get("name")) == wanted or label(hit.get("id")) == wanted)}
    if len(exact) == 1:
        return tuple(exact)
    if exact:
        return ()
    matches = set()
    alias_lookup_incomplete = False
    for hit in hits:
        identifier = hit.get("id") if isinstance(hit, Mapping) else None
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z]+_[0-9]+", identifier):
            continue
        try:
            result = await executor("query V2GTraitAliases($id:String!){ disease(efoId:$id){ id name synonyms{ relation terms } } }",
                                    {"id": identifier})
            disease = result.get("disease") or {}
            aliases = [disease.get("name"), *(term for row in disease.get("synonyms") or []
                                               if isinstance(row, Mapping) and row.get("relation") == "hasExactSynonym"
                                               for term in row.get("terms") or [])]
            if disease.get("id") == identifier and wanted in {label(alias) for alias in aliases}:
                matches.add(identifier)
        except Exception:  # noqa: BLE001 - unverified aliases cannot establish a phenotype identity
            alias_lookup_incomplete = True
    return tuple(matches) if len(matches) == 1 and not alias_lookup_incomplete else ()


def graphql_transport_executor(transport: Any) -> GraphQLExecutor:
    """Adapt the existing bounded official GraphQL transport to the retriever."""

    async def execute(query: str, variables: Mapping[str, Any]) -> Mapping[str, Any]:
        result = await transport.call_tool("query_open_targets_graphql",
                                           {"query_string": query, "variables": dict(variables)})
        if isinstance(result, Mapping) and result.get("errors"):
            raise RuntimeError("GraphQL response contains retrieval errors")
        data = result.get("data") if isinstance(result, Mapping) else None
        if not isinstance(data, Mapping):
            raise TypeError("GraphQL response has no data object")
        return data

    return execute
