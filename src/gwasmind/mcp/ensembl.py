from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, build_opener

from ..configuration import parameter
from .regions import GRCh38Region

ENSEMBL_REST_ENDPOINT = "https://rest.ensembl.org"
ENSEMBL_REST_SOURCE = "Ensembl REST"
DEFAULT_TIMEOUT_SECONDS = parameter("transport.ensembl_timeout_seconds")
DEFAULT_RETRY_TIMEOUT_SECONDS = parameter("transport.ensembl_retry_timeout_seconds")
DEFAULT_MAX_RESPONSE_BYTES = parameter("transport.ensembl_max_response_bytes")
DEFAULT_FLANK_BP = parameter("analysis.local_gene_window_bp")
DEFAULT_MAX_INPUT_SPAN_BP = parameter("retrieval.ensembl_max_input_span_bp")
DEFAULT_MAX_GENES = parameter("retrieval.ensembl_max_genes")
_ENSEMBL_GENE_ID = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)


class EnsemblRegionLookupError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RegionGeneCandidate:
    gene_id: str
    symbol: str | None
    chromosome: str
    start: int
    end: int
    strand: int | None
    biotype: str | None
    relation: str
    distance_to_region: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RegionGeneLookupResult:
    candidates: tuple[RegionGeneCandidate, ...]
    total_gene_count: int
    query_start: int
    query_end: int
    flank_bp: int
    candidate_list_truncated: bool
    coverage_truncated: bool = False

    @property
    def truncated(self) -> bool:
        """Compatibility view: only incomplete interval coverage is a hard gap."""

        return self.coverage_truncated


class RegionGeneLookup(Protocol):
    async def lookup(self, region: GRCh38Region) -> tuple[RegionGeneCandidate, ...]: ...


BoundedHttpGet = Callable[[str, float, int], bytes]


class EnsemblRegionGeneLookup:
    """At most two bounded Ensembl requests for complete ±1 Mb gene context."""

    def __init__(
        self,
        *,
        endpoint: str = ENSEMBL_REST_ENDPOINT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retry_timeout_seconds: float | None = DEFAULT_RETRY_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        flank_bp: int = DEFAULT_FLANK_BP,
        max_input_span_bp: int = DEFAULT_MAX_INPUT_SPAN_BP,
        max_genes: int = DEFAULT_MAX_GENES,
        http_get: BoundedHttpGet | None = None,
    ):
        if not endpoint.startswith("https://"):
            raise ValueError("Ensembl endpoint must use HTTPS")
        if not 0 < timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be in (0, 30]")
        if retry_timeout_seconds is not None and not (
            timeout_seconds <= retry_timeout_seconds <= 30
        ):
            raise ValueError(
                "retry_timeout_seconds must be None or between timeout_seconds and 30"
            )
        if not 1 <= max_response_bytes <= 4 * 1024 * 1024:
            raise ValueError("max_response_bytes must be between 1 and 4 MiB")
        if not 0 <= flank_bp <= 1_000_000:
            raise ValueError("flank_bp must be between 0 and 1,000,000")
        if not 1 <= max_input_span_bp <= 10_000_000:
            raise ValueError("max_input_span_bp must be between 1 and 10,000,000")
        if not 1 <= max_genes <= DEFAULT_MAX_GENES:
            raise ValueError(f"max_genes must be between 1 and {DEFAULT_MAX_GENES}")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.retry_timeout_seconds = (
            float(retry_timeout_seconds)
            if retry_timeout_seconds is not None
            else None
        )
        self.max_response_bytes = int(max_response_bytes)
        self.flank_bp = int(flank_bp)
        self.max_input_span_bp = int(max_input_span_bp)
        self.max_genes = int(max_genes)
        self._http_get = http_get or _bounded_http_get

    async def lookup(self, region: GRCh38Region) -> tuple[RegionGeneCandidate, ...]:
        return (await self.lookup_with_metadata(region)).candidates

    async def lookup_with_metadata(
        self, region: GRCh38Region
    ) -> RegionGeneLookupResult:
        if region.end - region.start + 1 > self.max_input_span_bp:
            raise EnsemblRegionLookupError(
                "derived locus is too wide for the bounded Ensembl lookup"
            )
        query_start = max(1, region.start - self.flank_bp)
        query_end = region.end + self.flank_bp
        coordinate = quote(
            f"{region.chromosome}:{query_start}-{query_end}", safe=":-"
        )
        # Ensembl documents ``content-type`` as an explicit query parameter for
        # this endpoint.  Supplying only an Accept header is intermittently
        # answered with HTTP 500 or a stalled response by some regional/proxy
        # paths, so keep both signals while retaining a single bounded request.
        url = (
            f"{self.endpoint}/overlap/region/human/{coordinate}"
            "?feature=gene;content-type=application/json"
        )
        try:
            raw = await asyncio.to_thread(
                self._http_get,
                url,
                self.timeout_seconds,
                self.max_response_bytes,
            )
        except EnsemblRegionLookupError:
            if (
                self.retry_timeout_seconds is None
                or self.retry_timeout_seconds <= self.timeout_seconds
            ):
                raise
            # Retry the same authoritative complete interval once with a larger
            # but still bounded timeout. Candidate evidence is never substituted
            # for a failed regional response.
            raw = await asyncio.to_thread(
                self._http_get,
                url,
                self.retry_timeout_seconds,
                self.max_response_bytes,
            )
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EnsemblRegionLookupError(
                "Ensembl returned malformed JSON"
            ) from exc
        if not isinstance(decoded, list):
            raise EnsemblRegionLookupError("Ensembl region response must be a list")

        candidates: dict[str, RegionGeneCandidate] = {}
        for value in decoded:
            if isinstance(value, Mapping):
                assembly = value.get("assembly_name")
                if assembly is not None and assembly != region.genome_build:
                    raise EnsemblRegionLookupError(
                        "Ensembl gene assembly does not match the requested GRCh38 region"
                    )
            candidate = _candidate_from_row(value, region)
            if candidate is None:
                # A skipped malformed gene must not turn into a complete empty
                # regional census, which feeds the no-nearby-gene hard gate.
                raise EnsemblRegionLookupError("Ensembl returned an invalid regional gene row")
            if candidate.end < query_start or candidate.start > query_end:
                raise EnsemblRegionLookupError(
                    "Ensembl returned a gene outside the requested interval"
                )
            existing = candidates.get(candidate.gene_id)
            if existing is None or _candidate_sort_key(candidate) < _candidate_sort_key(
                existing
            ):
                candidates[candidate.gene_id] = candidate
        ordered = sorted(candidates.values(), key=_candidate_sort_key)
        overlapping = [item for item in ordered if item.relation == "region_overlap"]
        # Exact interval overlaps are both faster to enrich downstream and less
        # ambiguous than a mixed overlap/nearby candidate list. Nearby genes are
        # used only when the selected interval does not overlap any gene.
        selected = overlapping if overlapping else ordered
        bounded = tuple(selected[: self.max_genes])
        return RegionGeneLookupResult(
            candidates=bounded,
            total_gene_count=len(ordered),
            query_start=query_start,
            query_end=query_end,
            flank_bp=self.flank_bp,
            candidate_list_truncated=len(selected) > len(bounded),
            # The complete decoded interval established total_gene_count.
            # Bounding only the displayed candidate list does not truncate
            # coverage for the regional hard gate.
            coverage_truncated=False,
        )


def _candidate_from_row(
    value: Any, region: GRCh38Region
) -> RegionGeneCandidate | None:
    if not isinstance(value, Mapping):
        return None
    feature_type = str(value.get("feature_type") or "gene").casefold()
    if feature_type != "gene":
        return None
    gene_id = str(value.get("id") or "").strip()
    if _ENSEMBL_GENE_ID.fullmatch(gene_id) is None:
        return None
    chromosome = str(value.get("seq_region_name") or "").strip()
    if chromosome.removeprefix("chr").upper() != region.chromosome.upper():
        return None
    start = _positive_integer(value.get("start"))
    end = _positive_integer(value.get("end"))
    if start is None or end is None or end < start:
        return None
    if end < region.start:
        distance = region.start - end
    elif start > region.end:
        distance = start - region.end
    else:
        distance = 0
    strand_value = value.get("strand")
    strand = (
        int(strand_value)
        if isinstance(strand_value, int)
        and not isinstance(strand_value, bool)
        and strand_value in {-1, 1}
        else None
    )
    symbol = str(value.get("external_name") or "").strip() or None
    biotype = str(value.get("biotype") or "").strip() or None
    return RegionGeneCandidate(
        gene_id=gene_id,
        symbol=symbol,
        chromosome=region.chromosome,
        start=start,
        end=end,
        strand=strand,
        biotype=biotype,
        relation="region_overlap" if distance == 0 else "nearby",
        distance_to_region=distance,
        raw=dict(value),
    )


def _candidate_sort_key(candidate: RegionGeneCandidate) -> tuple[int, int, int, str]:
    return (
        0 if candidate.relation == "region_overlap" else 1,
        candidate.distance_to_region,
        candidate.start,
        candidate.gene_id,
    )


def _positive_integer(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return None
    return value


def _bounded_http_get(url: str, timeout_seconds: float, max_bytes: int) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "GWASMind/1.0 Ensembl-region-gene-context",
        },
        method="GET",
    )
    try:
        with build_opener().open(request, timeout=timeout_seconds) as response:
            content_type = str(response.headers.get("Content-Type") or "").casefold()
            if "json" not in content_type:
                raise EnsemblRegionLookupError(
                    "Ensembl returned a non-JSON content type"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    if int(declared) > max_bytes:
                        raise EnsemblRegionLookupError(
                            "Ensembl response exceeds the configured byte limit"
                        )
                except ValueError as exc:
                    raise EnsemblRegionLookupError(
                        "Ensembl returned an invalid Content-Length"
                    ) from exc
            value = response.read(max_bytes + 1)
    except EnsemblRegionLookupError:
        raise
    except HTTPError as exc:
        raise EnsemblRegionLookupError(
            f"Ensembl returned HTTP {exc.code}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise EnsemblRegionLookupError("Ensembl request failed") from exc
    if len(value) > max_bytes:
        raise EnsemblRegionLookupError(
            "Ensembl response exceeds the configured byte limit"
        )
    return value
