from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from ..configuration import parameter
from ..mcp.catalog import FrozenDatasetDiscovery, discover_frozen_datasets
from ..models import (
    QTL_STUDY_TYPES,
    DataMode,
    EvidenceKind,
    EvidenceRecord,
    LocusQuery,
)
from ..qtl import requested_qtl_study_types
from .base import OpenTargetsProvider, ProviderCapabilities, ensure_supported

SUPPORTED_KINDS = frozenset(
    {
        EvidenceKind.GWAS,
        EvidenceKind.QTL,
        EvidenceKind.COLOCALIZATION,
        EvidenceKind.REGULATORY,
        EvidenceKind.EXPRESSION,
        EvidenceKind.FUNCTION,
        EvidenceKind.L2G,
    }
)

IMPLEMENTED_SOURCE_DATASETS = frozenset(
    {
        "biosample",
        "disease",
        "go",
        "known_drug",
        "reactome",
        "credible_set",
        "study",
        "variant",
        "l2g_prediction",
        "colocalisation_coloc",
        "enhancer_to_gene",
        "expression",
        "target",
    }
)


class _BoundedEvidenceRecords(list[EvidenceRecord]):
    """Keep the existing list API while carrying observed source truncation."""

    def __init__(self, records=(), *, truncated: bool = False):
        super().__init__(records)
        self.truncated = truncated

    def extend(self, records) -> None:
        self.truncated = self.truncated or getattr(records, "truncated", False)
        super().extend(records)


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _evidence_id(dataset: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(f"{dataset}:{raw}".encode("utf-8")).hexdigest()[:20]
    return f"ot:{dataset}:{digest}"


def _variant_chromosome(variant_id: str) -> str:
    """Return the chromosome component of a normalized Open Targets variant ID."""

    chromosome = variant_id.split("_", 1)[0]
    if chromosome.casefold().startswith("chr"):
        chromosome = chromosome[3:]
    return chromosome


def _canonical_variant_location(variant_id: Any) -> tuple[str, int] | None:
    """Parse one canonical Open Targets variant ID without inventing alleles/positions."""

    if not isinstance(variant_id, str):
        return None
    canonical = _normalized_variant_id(variant_id)
    if not re.fullmatch(r"(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)_[0-9]+_[ACGTN]+_[ACGTN]+", canonical):
        return None
    parts = canonical.split("_")
    try:
        position = int(parts[1])
    except ValueError:
        return None
    if position < 1:
        return None
    chromosome = parts[0][3:] if parts[0].casefold().startswith("chr") else parts[0]
    return (chromosome, position) if chromosome else None


def _normalized_variant_id(value: str) -> str:
    canonical = value.strip().upper().removeprefix("CHR").replace(":", "_")
    if re.fullmatch(r"(?:[0-9]+|X|Y|MT)_[0-9]+_[ACGTN]+_[ACGTN]+", canonical):
        return canonical
    return value.casefold() if re.fullmatch(r"rs[0-9]+", value, re.IGNORECASE) else value


class DuckDBFrozenOpenTargetsProvider(OpenTargetsProvider):
    """Read-only adapter for a downloaded Open Targets Parquet release.

    Dataset paths are fixed by configuration. All locus, variant, gene, and
    study values use DuckDB parameter binding; they are never interpolated into
    SQL strings.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        release: str,
        snapshot_hash: str,
        max_threads: int = parameter("execution.duckdb_threads"),
        memory_limit: str = parameter("execution.duckdb_memory_limit"),
    ):
        self.data_root = Path(data_root).resolve()
        if not self.data_root.is_dir():
            raise ValueError(f"Open Targets data root does not exist: {self.data_root}")
        if (
            not isinstance(release, str)
            or not release.strip()
            or release.casefold() in {"live", "unknown", "unverified"}
        ):
            raise ValueError("frozen provider requires one explicit frozen release")
        if (
            not isinstance(snapshot_hash, str)
            or not snapshot_hash.strip()
            or snapshot_hash.casefold() in {"live", "unknown", "unverified"}
        ):
            raise ValueError("frozen provider requires a verified snapshot hash")
        if isinstance(max_threads, bool) or not isinstance(max_threads, int) or not 1 <= max_threads <= 8:
            raise ValueError("max_threads must be between 1 and 8")
        if not isinstance(memory_limit, str) or not re.fullmatch(r"[1-9][0-9]*(?:MB|GB)", memory_limit.upper()):
            raise ValueError("memory_limit must use a positive MB or GB value")
        self.max_threads = max_threads
        self.memory_limit = memory_limit.upper()
        self.dataset_discovery = discover_frozen_datasets(
            self.data_root,
            implemented_datasets=IMPLEMENTED_SOURCE_DATASETS,
        )
        ready_datasets = frozenset(
            item.canonical_name
            for item in self.dataset_discovery.ready
            if item.canonical_name is not None
        )
        ready_modules = frozenset(
            module for item in self.dataset_discovery.ready for module in item.modules
        )
        self._capabilities = ProviderCapabilities(
            mode=DataMode.FROZEN_SAFE,
            release=release,
            evidence_kinds=SUPPORTED_KINDS,
            snapshot_hash=snapshot_hash,
            datasets=ready_datasets,
            modules=ready_modules,
            supports_l2g="l2g_prediction" in ready_datasets,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def frozen_dataset_discovery(self) -> FrozenDatasetDiscovery:
        return self.dataset_discovery

    def _dataset_glob(self, dataset: str) -> str:
        if dataset not in {
            "credible_set",
            "study",
            "variant",
            "l2g_prediction",
            "colocalisation_coloc",
            "enhancer_to_gene",
            "expression",
            "target",
            "disease",
            "known_drug",
            "biosample",
            "go",
            "reactome",
            "association_by_datasource_direct",
            "association_by_datasource_indirect",
        }:
            raise ValueError(f"unsupported dataset: {dataset}")
        path = self.data_root / dataset
        if not path.is_dir():
            raise ValueError(f"missing Open Targets dataset: {path}")
        return (path / "**" / "*.parquet").as_posix().replace("'", "''")

    def _connect(self):
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError('Install local support with: pip install -e ".[local]"') from exc
        connection = duckdb.connect(":memory:")
        # Conservative defaults keep interactive runs responsive. Values are
        # validated above before being interpolated into DuckDB settings.
        try:
            connection.execute(f"SET threads = {self.max_threads}")
            connection.execute(f"SET memory_limit = '{self.memory_limit}'")
            connection.execute("SET preserve_insertion_order = false")
        except Exception:
            connection.close()
            raise
        return connection

    def _resolve_variants(self, conn, query: LocusQuery) -> dict[str, dict[str, Any]]:
        path = self._dataset_glob("variant")
        resolved: dict[str, dict[str, Any]] = {}
        sql = f"""
            SELECT variantId, chromosome, position, rsIds,
                   mostSevereConsequenceId, transcriptConsequences
            FROM read_parquet('{path}')
            WHERE variantId = ? OR list_contains(rsIds, ?)
            LIMIT 5
        """
        for observation in query.variants:
            variant_id = _normalized_variant_id(observation.variant_id)
            rows = conn.execute(sql, [variant_id, variant_id]).fetchall()
            columns = [item[0] for item in conn.description]
            matches: dict[str, dict[str, Any]] = {}
            for row in rows:
                record = dict(zip(columns, row, strict=True))
                matches[str(record["variantId"])] = record
            if len(matches) > 1:
                raise ValueError(
                    f"ambiguous canonical variant mapping for {observation.variant_id!r}"
                )
            resolved.update(matches)
        return resolved

    @staticmethod
    def _is_lead_variant(conn, credible_path: str, variant_id: str) -> bool:
        chromosome = _variant_chromosome(variant_id)
        row = conn.execute(
            f"SELECT 1 FROM read_parquet('{credible_path}') "
            "WHERE chromosome = ? AND variantId = ? LIMIT 1",
            [chromosome, variant_id],
        ).fetchone()
        return row is not None

    def _credible_match_predicate(
        self, conn, credible_path: str, variant_id: str
    ) -> str:
        # A variant may lead one study and be a member of another. Checking only
        # its lead rows drops legitimate QTL/GWAS and L2G evidence in that case.
        # DuckDB 2.0 rejects the old "x -> ..." lambda form; this syntax needs 1.3+.
        return (
            "list_contains("
            "list_concat([cs.variantId], "
            "list_transform(cs.locus, lambda member: member.variantId)), ?"
            ")"
        )

    async def fetch(
        self, query: LocusQuery, kinds: set[EvidenceKind]
    ) -> list[EvidenceRecord]:
        query.validate()
        if query.genome_build and query.genome_build.casefold() in {"grch37", "hg19"}:
            raise ValueError("frozen Open Targets retrieval requires GRCh38 coordinates")
        ensure_supported(self, kinds)
        conn = self._connect()
        try:
            resolved = self._resolve_variants(conn, query)
            variant_ids = list(resolved)
            evidence = _BoundedEvidenceRecords()
            if EvidenceKind.GWAS in kinds and variant_ids:
                evidence.extend(
                    self._fetch_associations(
                        conn, variant_ids, gwas=True, study_ids=query.study_ids
                    )
                )
            if EvidenceKind.L2G in kinds and variant_ids:
                evidence.extend(self._fetch_l2g(conn, variant_ids))
            qtl_records: list[EvidenceRecord] = []
            if variant_ids and kinds.intersection(
                {EvidenceKind.QTL, EvidenceKind.COLOCALIZATION}
            ):
                # Colocalisation requires an exact native-QTL lineage.  E2G
                # regulatory evidence is independent: the interval must contain
                # the selected variant coordinate and supplies its own gene link.
                qtl_records = self._fetch_associations(
                    conn,
                    variant_ids,
                    gwas=False,
                    qtl_types=query.qtl_types,
                )
            if EvidenceKind.QTL in kinds:
                evidence.extend(qtl_records)
            if EvidenceKind.COLOCALIZATION in kinds:
                evidence.extend(self._fetch_colocalization(conn, qtl_records))
                evidence.truncated = evidence.truncated or getattr(qtl_records, "truncated", False)
            if EvidenceKind.REGULATORY in kinds:
                evidence.extend(self._fetch_regulatory(conn, variant_ids))
            if EvidenceKind.EXPRESSION in kinds and query.candidate_genes:
                evidence.extend(self._fetch_expression(conn, query.candidate_genes))
            if EvidenceKind.FUNCTION in kinds and query.candidate_genes:
                evidence.extend(self._fetch_functions(conn, query.candidate_genes))
            if evidence.truncated:
                raise RuntimeError(
                    "frozen source evidence exceeded its retrieval bound; "
                    "use semantic retrieval to retain explicit partial coverage"
                )
            return evidence
        finally:
            conn.close()

    def _fetch_associations(
        self,
        conn,
        variant_ids: list[str],
        *,
        gwas: bool,
        study_ids: tuple[str, ...] = (),
        qtl_types: tuple[str, ...] = (),
    ) -> list[EvidenceRecord]:
        credible = self._dataset_glob("credible_set")
        study = self._dataset_glob("study")
        selected_qtl_types = (
            requested_qtl_study_types(qtl_types) or QTL_STUDY_TYPES
        )
        if gwas:
            comparator = "= 'gwas'"
            qtl_parameters: tuple[str, ...] = ()
        else:
            qtl_placeholders = ",".join("?" for _ in selected_qtl_types)
            comparator = f"IN ({qtl_placeholders})"
            qtl_parameters = selected_qtl_types
        study_filter = ""
        if study_ids:
            study_placeholders = ",".join("?" for _ in study_ids)
            study_filter = f" AND s.studyId IN ({study_placeholders})"
        sql_template = """
            SELECT ? AS queryVariantId,
                   cs.studyLocusId, cs.studyId,
                   cs.variantId AS leadVariantId, cs.chromosome,
                   cs.position, cs.beta, cs.pValueMantissa, cs.pValueExponent,
                   cs.finemappingMethod, cs.confidence, cs.studyType,
                   list_extract(
                       list_filter(cs.locus, lambda member: member.variantId = ?), 1
                   ) AS matchedLocusMember,
                   s.geneId, s.traitFromSource, s.biosampleId, s.pubmedId,
                   s.studyType AS sourceStudyType,
                   s.publicationDate, s.publicationFirstAuthor,
                   s.publicationJournal
            FROM read_parquet('{credible}') cs
            JOIN read_parquet('{study}') s ON cs.studyId = s.studyId
            WHERE cs.chromosome = ?
                  AND ({match_predicate})
                  AND s.studyType {comparator}
                  {study_filter}
            LIMIT 1001
        """
        kind = EvidenceKind.GWAS if gwas else EvidenceKind.QTL
        records = _BoundedEvidenceRecords()
        seen: set[tuple[str, str]] = set()
        for query_variant_id in variant_ids:
            match_predicate = self._credible_match_predicate(
                conn, credible, query_variant_id
            )
            sql = sql_template.format(
                credible=credible,
                study=study,
                match_predicate=match_predicate,
                comparator=comparator,
                study_filter=study_filter,
            )
            parameters = [
                query_variant_id,
                query_variant_id,
                _variant_chromosome(query_variant_id),
                query_variant_id,
                *qtl_parameters,
                *study_ids,
            ]
            rows = conn.execute(sql, parameters).fetchall()
            records.truncated = records.truncated or len(rows) > 1000
            columns = [item[0] for item in conn.description]
            for row in rows[:1000]:
                payload = dict(zip(columns, row, strict=True))
                source_study_type = payload.pop("sourceStudyType", None)
                if source_study_type is not None:
                    if payload.get("studyType") not in (None, source_study_type):
                        raise ValueError("credible-set study type conflicts with source study metadata")
                    payload["studyType"] = source_study_type
                if not gwas and (
                    not isinstance(payload.get("geneId"), str)
                    or not payload["geneId"].strip()
                ):
                    raise ValueError("molecular QTL study omitted its gene identity")
                key = (query_variant_id, str(payload["studyLocusId"]))
                if key in seen:
                    continue
                seen.add(key)
                lead_variant_id = str(payload["leadVariantId"])
                payload["queryVariantRole"] = (
                    "lead" if query_variant_id == lead_variant_id else "credible_set_member"
                )
                parent_variants = tuple(dict.fromkeys((query_variant_id, lead_variant_id)))
                records.append(
                    EvidenceRecord(
                        evidence_id=_evidence_id("credible_set", payload),
                        kind=kind,
                        dataset="credible_set",
                        source="Open Targets Platform",
                        data_release=self.capabilities.release,
                        payload=payload,
                        variant_id=query_variant_id,
                        gene_id=payload.get("geneId"),
                        study_id=payload.get("studyId"),
                        pmid=(
                            str(payload["pubmedId"])
                            if payload.get("pubmedId")
                            else None
                        ),
                        credible_set_id=payload.get("studyLocusId"),
                        parent_variant_ids=parent_variants,
                        tissue=payload.get("biosampleId"),
                        evidence_date=_parse_date(payload.get("publicationDate")),
                    )
                )
        return records

    def _fetch_l2g(self, conn, variant_ids: list[str]) -> list[EvidenceRecord]:
        credible = self._dataset_glob("credible_set")
        l2g = self._dataset_glob("l2g_prediction")
        sql_template = """
            WITH loci AS (
                SELECT ? AS queryVariantId,
                       cs.variantId AS leadVariantId,
                       cs.studyId,
                       cs.studyLocusId,
                       list_extract(
                           list_filter(cs.locus, lambda member: member.variantId = ?), 1
                       ) AS matchedLocusMember
                FROM read_parquet('{credible}') cs
                WHERE cs.chromosome = ?
                  AND ({match_predicate})
            )
            SELECT loci.queryVariantId, loci.leadVariantId,
                   loci.studyId, loci.studyLocusId, loci.matchedLocusMember,
                   l2g.geneId, l2g.score, l2g.features
            FROM loci
            JOIN read_parquet('{l2g}') l2g
              ON loci.studyLocusId = l2g.studyLocusId
            ORDER BY l2g.score DESC
            LIMIT 1001
        """
        records = _BoundedEvidenceRecords()
        seen: set[tuple[str, str, str]] = set()
        for query_variant_id in variant_ids:
            sql = sql_template.format(
                credible=credible,
                l2g=l2g,
                match_predicate=self._credible_match_predicate(
                    conn, credible, query_variant_id
                ),
            )
            rows = conn.execute(
                sql,
                [
                    query_variant_id,
                    query_variant_id,
                    _variant_chromosome(query_variant_id),
                    query_variant_id,
                ],
            ).fetchall()
            records.truncated = records.truncated or len(rows) > 1000
            columns = [item[0] for item in conn.description]
            for row in rows[:1000]:
                payload = dict(zip(columns, row, strict=True))
                payload["l2g_score"] = payload.pop("score")
                lead_variant_id = str(payload["leadVariantId"])
                payload["queryVariantRole"] = (
                    "lead" if query_variant_id == lead_variant_id else "credible_set_member"
                )
                key = (
                    query_variant_id,
                    str(payload["studyLocusId"]),
                    str(payload["geneId"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    EvidenceRecord(
                        evidence_id=_evidence_id("l2g_prediction", payload),
                        kind=EvidenceKind.L2G,
                        dataset="l2g_prediction",
                        source="Open Targets Platform L2G",
                        data_release=self.capabilities.release,
                        payload=payload,
                        variant_id=query_variant_id,
                        gene_id=payload.get("geneId"),
                        study_id=payload.get("studyId"),
                        credible_set_id=payload.get("studyLocusId"),
                        parent_variant_ids=tuple(
                            dict.fromkeys((query_variant_id, lead_variant_id))
                        ),
                    )
                )
        return records

    def _fetch_colocalization(
        self, conn, qtl_records: list[EvidenceRecord]
    ) -> list[EvidenceRecord]:
        """Retrieve coloc only for the exact QTL studyLocusIds found upstream."""

        qtl_by_locus: dict[str, list[EvidenceRecord]] = {}
        for record in qtl_records:
            member = record.payload.get("matchedLocusMember")
            member_id = member.get("variantId") if isinstance(member, dict) else None
            if (
                record.kind is not EvidenceKind.QTL
                or not record.credible_set_id
                or not record.gene_id
                or member_id != record.variant_id
                or _canonical_variant_location(member_id) is None
            ):
                continue
            qtl_by_locus.setdefault(str(record.credible_set_id), []).append(record)
        if not qtl_by_locus:
            return []
        credible = self._dataset_glob("credible_set")
        study = self._dataset_glob("study")
        coloc = self._dataset_glob("colocalisation_coloc")
        placeholders = ",".join("?" for _ in qtl_by_locus)
        sql = f"""
            WITH colocated AS (
                SELECT c.*,
                       c.leftStudyLocusId AS queriedStudyLocusId,
                       c.rightStudyLocusId AS otherStudyLocusId
                FROM read_parquet('{coloc}') c
                WHERE c.leftStudyLocusId IN ({placeholders})
                UNION ALL
                SELECT c.*,
                       c.rightStudyLocusId AS queriedStudyLocusId,
                       c.leftStudyLocusId AS otherStudyLocusId
                FROM read_parquet('{coloc}') c
                WHERE c.rightStudyLocusId IN ({placeholders})
            )
            SELECT colocated.queriedStudyLocusId,
                   colocated.leftStudyLocusId, colocated.rightStudyLocusId,
                   colocated.otherStudyLocusId, colocated.rightStudyType,
                   colocated.h3, colocated.h4,
                   colocated.colocalisationMethod,
                   colocated.numberColocalisingVariants,
                   other_cs.studyId AS otherStudyId,
                   other_study.studyType AS otherStudyType,
                   other_study.geneId AS otherGeneId,
                   other_study.biosampleId AS otherBiosampleId,
                   other_study.traitFromSource AS otherTraitFromSource,
                   other_study.pubmedId AS otherPubmedId,
                   other_study.publicationDate AS otherPublicationDate,
                   other_study.publicationFirstAuthor AS otherPublicationFirstAuthor,
                   other_study.publicationJournal AS otherPublicationJournal
            FROM colocated
            LEFT JOIN read_parquet('{credible}') other_cs
              ON colocated.otherStudyLocusId = other_cs.studyLocusId
            LEFT JOIN read_parquet('{study}') other_study
              ON other_cs.studyId = other_study.studyId
            ORDER BY colocated.h4 DESC
            LIMIT 1001
        """
        locus_ids = list(qtl_by_locus)
        parameters = [*locus_ids, *locus_ids]
        rows = conn.execute(sql, parameters).fetchall()
        columns = [item[0] for item in conn.description]
        records = _BoundedEvidenceRecords(truncated=len(rows) > 1000)
        seen: set[tuple[str, str, str]] = set()
        for row in rows[:1000]:
            payload = dict(zip(columns, row, strict=True))
            queried_locus_id = str(payload.get("queriedStudyLocusId") or "")
            upstream_qtls = qtl_by_locus.get(queried_locus_id, [])
            if not upstream_qtls:  # fail closed if upstream lineage is not exact
                continue
            for qtl in upstream_qtls:
                lineage_payload = {
                    **payload,
                    "upstreamLeftStudyLocusId": payload.get("leftStudyLocusId"),
                    "upstreamRightStudyLocusId": payload.get("rightStudyLocusId"),
                    "leftStudyLocusId": queried_locus_id,
                    "rightStudyLocusId": payload.get("otherStudyLocusId"),
                    "leftStudyId": qtl.study_id,
                    "rightStudyId": payload.get("otherStudyId"),
                    "leftStudyType": qtl.payload.get("studyType"),
                    "rightStudyType": payload.get("otherStudyType"),
                    "leftPubmedId": qtl.pmid,
                    "rightPubmedId": payload.get("otherPubmedId"),
                    "leftTraitFromSource": qtl.payload.get("traitFromSource"),
                    "rightTraitFromSource": payload.get("otherTraitFromSource"),
                    "leftPublicationDate": qtl.payload.get("publicationDate"),
                    "rightPublicationDate": payload.get("otherPublicationDate"),
                    "leftPublicationFirstAuthor": qtl.payload.get("publicationFirstAuthor"),
                    "rightPublicationFirstAuthor": payload.get("otherPublicationFirstAuthor"),
                    "leftPublicationJournal": qtl.payload.get("publicationJournal"),
                    "rightPublicationJournal": payload.get("otherPublicationJournal"),
                    "qtlStudyLocusId": queried_locus_id,
                    "qtlGeneId": qtl.gene_id,
                    "qtlStudyType": qtl.payload.get("studyType"),
                    "qtlMatchedLocusMember": qtl.payload.get("matchedLocusMember"),
                }
                key = (
                    queried_locus_id,
                    str(payload.get("otherStudyLocusId")),
                    str(qtl.variant_id),
                )
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    EvidenceRecord(
                        evidence_id=_evidence_id(
                            "colocalisation_coloc", lineage_payload
                        ),
                        kind=EvidenceKind.COLOCALIZATION,
                        dataset="colocalisation_coloc",
                        source="Open Targets Platform",
                        data_release=self.capabilities.release,
                        payload=lineage_payload,
                        variant_id=qtl.variant_id,
                        gene_id=qtl.gene_id,
                        study_id=qtl.study_id,
                        pmid=qtl.pmid,
                        credible_set_id=queried_locus_id,
                        parent_variant_ids=qtl.parent_variant_ids,
                        parent_evidence_ids=(qtl.evidence_id,),
                        tissue=qtl.tissue,
                        evidence_date=max(
                            (value for value in (
                                qtl.evidence_date, _parse_date(payload.get("otherPublicationDate")),
                            ) if value is not None), default=None,
                        ),
                    )
                )
        return records

    def _fetch_regulatory(
        self, conn, variant_ids: list[str]
    ) -> list[EvidenceRecord]:
        """Return E2G intervals that contain each selected variant exactly."""

        if not variant_ids:
            return []
        enhancer = self._dataset_glob("enhancer_to_gene")
        sql = f"""
            SELECT chromosome, start, "end", geneId, biosampleName,
                   intervalType, distanceToTss, score, resourceScore, datasourceId, pmid,
                   studyId, biosampleId, intervalId
            FROM read_parquet('{enhancer}')
            WHERE chromosome = ? AND start <= ? AND "end" >= ?
            ORDER BY score DESC, geneId, start, "end"
            LIMIT 201
        """
        seen: set[str] = set()
        records = _BoundedEvidenceRecords()
        for variant_id in variant_ids:
            location = _canonical_variant_location(variant_id)
            if location is None:
                raise ValueError(
                    "enhancer lookup requires a canonical Open Targets variant ID"
                )
            chromosome, position = location
            rows = conn.execute(
                sql,
                [chromosome, position, position],
            ).fetchall()
            records.truncated = records.truncated or len(rows) > 200
            columns = [item[0] for item in conn.description]
            for row in rows[:200]:
                payload = dict(zip(columns, row, strict=True))
                gene_id = str(payload.get("geneId") or "").strip()
                start = payload.get("start")
                end = payload.get("end")
                row_chromosome = str(payload.get("chromosome") or "").removeprefix(
                    "chr"
                )
                if not gene_id:
                    raise ValueError("enhancer-to-gene row omitted geneId")
                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or start < 0
                    or end < start
                ):
                    raise ValueError(
                        "enhancer-to-gene row contains an invalid genomic interval"
                    )
                if row_chromosome != chromosome or not start <= position <= end:
                    raise ValueError(
                        "enhancer-to-gene interval does not contain the query variant"
                    )
                interval_id = str(payload.get("intervalId") or "").strip() or None
                payload.update(
                    {
                        "variantId": variant_id,
                        "queryVariantId": variant_id,
                        "overlapVariantId": variant_id,
                        "overlapVariantRole": "selected",
                        "overlapChromosome": chromosome,
                        "overlapPosition": position,
                        "exactVariantOverlap": True,
                        "coverageScope": "selected_variants_exact_overlap",
                        "regulatoryElementKey": interval_id
                        or f"chr{chromosome}:{start}-{end}",
                        "regulatoryElementKeySource": (
                            "upstream_interval_id"
                            if interval_id
                            else "derived_coordinate_key"
                        ),
                    }
                )
                key = _evidence_id("enhancer_to_gene", payload)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    EvidenceRecord(
                        evidence_id=key,
                        kind=EvidenceKind.REGULATORY,
                        dataset="enhancer_to_gene",
                        source="Open Targets Platform",
                        data_release=self.capabilities.release,
                        payload=payload,
                        variant_id=variant_id,
                        gene_id=gene_id,
                        study_id=payload.get("studyId"),
                        pmid=str(payload["pmid"]) if payload.get("pmid") else None,
                        parent_variant_ids=(variant_id,),
                        tissue=payload.get("biosampleName"),
                    )
                )
        return records

    def _fetch_expression(self, conn, genes: tuple[str, ...]) -> list[EvidenceRecord]:
        expression = self._dataset_glob("expression")
        placeholders = ",".join("?" for _ in genes)
        sql = f"SELECT id, tissues FROM read_parquet('{expression}') WHERE id IN ({placeholders})"
        rows = conn.execute(sql, list(genes)).fetchall()
        records: list[EvidenceRecord] = []
        for gene_id, tissues in rows:
            payload = {"geneId": gene_id, "tissues": tissues}
            records.append(
                EvidenceRecord(
                    evidence_id=_evidence_id("expression", payload),
                    kind=EvidenceKind.EXPRESSION,
                    dataset="expression",
                    source="Open Targets Platform",
                    data_release=self.capabilities.release,
                    payload=payload,
                    gene_id=gene_id,
                )
            )
        return records

    def _fetch_functions(self, conn, genes: tuple[str, ...]) -> list[EvidenceRecord]:
        target = self._dataset_glob("target")
        placeholders = ",".join("?" for _ in genes)
        sql = f"""
            SELECT id, approvedSymbol, approvedName, functionDescriptions,
                   go, pathways, genomicLocation
            FROM read_parquet('{target}')
            WHERE id IN ({placeholders}) OR approvedSymbol IN ({placeholders})
        """
        parameters = list(genes) + list(genes)
        rows = conn.execute(sql, parameters).fetchall()
        columns = [item[0] for item in conn.description]
        records: list[EvidenceRecord] = []
        for row in rows:
            payload = dict(zip(columns, row, strict=True))
            records.append(
                EvidenceRecord(
                    evidence_id=_evidence_id("target", payload),
                    kind=EvidenceKind.FUNCTION,
                    dataset="target",
                    source="Open Targets Platform",
                    data_release=self.capabilities.release,
                    payload=payload,
                    gene_id=payload.get("id"),
                )
            )
        return records
