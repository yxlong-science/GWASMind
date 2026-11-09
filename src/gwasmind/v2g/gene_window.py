"""Lead-centered gene window from the packaged coordinate table or one Ensembl REST request."""
from __future__ import annotations

import bisect
import gzip
import io
import json
import time
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from importlib.resources import files
from typing import Any
from urllib.request import Request, build_opener

from ..configuration import parameter
from .models import ENSEMBL_GENE_ID, GeneWindow, LocusVariantSet, WindowGene

ENSEMBL_REST_ENDPOINT = "https://rest.ensembl.org"
ENSEMBL_SOURCE = "Ensembl REST GRCh38 overlap"
GENE_TABLE_RESOURCE = "data/gene_coordinates_grch38.tsv.gz"
HttpGet = Callable[[str, float, int], bytes]


@lru_cache(maxsize=1)
def packaged_gene_table() -> tuple[str, dict[str, tuple[list[int], list[dict[str, Any]]]]]:
    """(provenance line, per-chromosome genes sorted by start) from the packaged table."""

    raw = files("gwasmind").joinpath(GENE_TABLE_RESOURCE).read_bytes()
    with io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding="utf-8") as handle:
        provenance = handle.readline().lstrip("# ").strip()
        header = handle.readline().rstrip("\n").split("\t")
        by_chromosome: dict[str, list[dict[str, Any]]] = {}
        for line in handle:
            row = dict(zip(header, line.rstrip("\n").split("\t"), strict=False))
            by_chromosome.setdefault(row["chromosome"], []).append(
                {"id": row["gene_id"], "external_name": row["symbol"] or None, "start": int(row["start"]),
                 "end": int(row["end"]), "strand": int(row["strand"] or 1), "biotype": row["biotype"] or None,
                 "seq_region_name": row["chromosome"], "feature_type": "gene"})
    table = {}
    for chromosome, genes in by_chromosome.items():
        genes.sort(key=lambda gene: (gene["start"], gene["id"]))
        table[chromosome] = ([gene["start"] for gene in genes], genes)
    return provenance, table


def _packaged_overlap(chromosome: str, start: int, end: int) -> tuple[list[dict[str, Any]], str]:
    provenance, table = packaged_gene_table()
    starts, genes = table.get(chromosome, ([], []))
    # Genes are sorted by start; any overlapping gene starts at or before `end`.
    candidates = genes[:bisect.bisect_right(starts, end)]
    return [gene for gene in candidates if gene["end"] >= start], provenance


def _default_http_get(url: str, timeout: float, max_bytes: int) -> bytes:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "GWASMind-V2G/1.0"})
    with build_opener().open(request, timeout=timeout) as response:
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError("Ensembl overlap response exceeds its byte limit")
    return body


def window_bounds(lead_position: int, flank_bp: int) -> tuple[int, int]:
    return max(1, lead_position - flank_bp), lead_position + flank_bp


def parse_ensembl_genes(payload: Any, *, chromosome: str, start: int, end: int) -> list[WindowGene]:
    """Validated gene features inside [start, end] under the configured window rule."""

    if not isinstance(payload, list):
        raise TypeError("Ensembl overlap payload must be a list")
    rule = parameter("v2g.gene_window_rule")
    genes: dict[str, WindowGene] = {}
    for item in payload:
        if not isinstance(item, Mapping) or item.get("feature_type", "gene") != "gene":
            continue
        gene_id = str(item.get("id") or item.get("gene_id") or "").split(".", 1)[0]
        try:
            gene_start, gene_end = int(item["start"]), int(item["end"])
            strand = int(item.get("strand") or 1)
        except (KeyError, TypeError, ValueError):
            continue
        if not ENSEMBL_GENE_ID.fullmatch(gene_id) or gene_end < gene_start:
            continue
        if str(item.get("seq_region_name", chromosome)) != chromosome:
            continue
        gene = WindowGene(gene_id=gene_id, symbol=item.get("external_name") or None, chromosome=chromosome,
                          start=gene_start, end=gene_end, strand=1 if strand >= 0 else -1,
                          biotype=item.get("biotype") or None)
        inside = start <= gene.tss <= end if rule == "tss" else gene_start <= end and gene_end >= start
        if inside:
            genes[gene_id] = gene
    return sorted(genes.values(), key=lambda gene: (gene.start, gene.gene_id))


def fetch_gene_window(locus: LocusVariantSet, *, explicit_candidates: Sequence[str] = (),
                      http_get: HttpGet | None = None, endpoint: str = ENSEMBL_REST_ENDPOINT) -> GeneWindow:
    """Genes around the lead; explicit candidates restrict the window and override the biotype filter."""

    flank = parameter("analysis.local_gene_window_bp")
    start, end = window_bounds(locus.lead_position, flank)
    source = ENSEMBL_SOURCE
    if parameter("v2g.gene_window_source") == "packaged_table" and http_get is None:
        payload, provenance = _packaged_overlap(locus.chromosome, start, end)
        source = f"Packaged GRCh38 gene table ({provenance})"
    else:
        url = (f"{endpoint.rstrip('/')}/overlap/region/human/{locus.chromosome}:{start}-{end}"
               "?feature=gene;content-type=application/json")
        fetch = http_get or _default_http_get
        timeout = parameter("transport.ensembl_retry_timeout_seconds")
        max_bytes = 4 * 1024 * 1024
        payload, failure = None, None
        for attempt in range(3):
            try:
                payload = json.loads(fetch(url, timeout, max_bytes).decode("utf-8"))
                failure = None
                break
            except Exception as exc:  # noqa: BLE001 - report the class, never a credential
                failure = type(exc).__name__
                time.sleep(0.5 * (attempt + 1))
        if payload is None:
            return GeneWindow(locus.chromosome, start, end, (), "failed", ENSEMBL_SOURCE,
                              tuple(explicit_candidates), (f"Ensembl gene window lookup failed ({failure}).",))
    try:
        genes = parse_ensembl_genes(payload, chromosome=locus.chromosome, start=start, end=end)
    except (TypeError, ValueError) as exc:
        return GeneWindow(locus.chromosome, start, end, (), "failed", source,
                          tuple(explicit_candidates), (str(exc),))
    explicit = tuple(dict.fromkeys(str(item).split(".", 1)[0] for item in explicit_candidates if item))
    issues: list[str] = []
    if explicit:
        wanted = set(explicit)
        genes = [gene for gene in genes if gene.gene_id in wanted]
        found = {gene.gene_id for gene in genes}
        missing = tuple(gene_id for gene_id in explicit if gene_id not in found)
        if missing:
            issues.append(f"{len(missing)} explicit candidate(s) were not found in the current Ensembl window.")
    else:
        biotypes = set(parameter("v2g.gene_biotypes"))
        genes = [gene for gene in genes if gene.biotype in biotypes]
        missing = ()
    status = "complete"
    limit = parameter("v2g.max_window_genes")
    if len(genes) > limit:
        genes = sorted(genes, key=lambda gene: (abs(gene.tss - locus.lead_position), gene.gene_id))[:limit]
        genes = sorted(genes, key=lambda gene: (gene.start, gene.gene_id))
        status = "truncated"
        issues.append(f"The gene window exceeded {limit} genes; the nearest {limit} were retained.")
    return GeneWindow(locus.chromosome, start, end, tuple(genes), status, source, missing, tuple(issues))
