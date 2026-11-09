from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..configuration import parameter
from ..publications import (
    CitationRegistry,
    Publication,
    normalize_pmid,
    publication_metadata_conflicts,
)

EUROPE_PMC_SEARCH_ENDPOINT = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
)
EUROPE_PMC_SOURCE = "Europe PMC"
NCBI_PUBMED_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_PUBMED_SOURCE = "NCBI PubMed (ESummary)"
_MAX_PMIDS = parameter("retrieval.publication_max_pmids")
_MAX_RESPONSE_BYTES = parameter("transport.publication_max_response_bytes")
_MAX_ABSTRACT_CHARS = parameter("retrieval.publication_abstract_chars")
_PUBLICATION_CACHE: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_PUBLICATION_CACHE_LOCK = threading.Lock()
_METADATA_BATCH_LIMIT = 25
_METADATA_RETRY_LIMIT = 5
_NCBI_REQUEST_LOCK = threading.Lock()
_NCBI_LAST_REQUEST = 0.0


class EuropePMCError(RuntimeError):
    """A bounded Europe PMC metadata request failed."""

    def __init__(self, message: str, *, category: str = "provider_error",
                 http_status: int | None = None, cause_type: str | None = None):
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.cause_type = cause_type


class EuropePMCPublicationClient:
    """Fetch publication metadata/abstracts for an explicit PMID set.

    The request never performs a free-text literature search.  It enriches only
    PMIDs already returned by the evidence provider, which keeps the retrieval
    traceable and bounded.
    """

    def __init__(self, *, timeout_seconds: float = parameter("transport.publication_timeout_seconds")):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self._attempts: list[dict[str, Any]] = []
        self._fetch_receipts: list[dict[str, Any]] = []
        self._conflicts: list[dict[str, Any]] = []

    @property
    def diagnostics(self) -> dict[str, Any]:
        return deepcopy({"attempts": self._attempts, "fetches": self._fetch_receipts,
                         "provider_conflicts": self._conflicts})

    async def fetch(self, pmids: Sequence[str]) -> dict[str, dict[str, Any]]:
        normalized = tuple(
            dict.fromkeys(
                value
                for item in pmids
                if (value := normalize_pmid(item)) is not None
            )
        )[:_MAX_PMIDS]
        if not normalized:
            return {}
        now = time.monotonic()
        cached: dict[str, dict[str, Any]] = {}
        ttl = parameter("retrieval.publication_cache_ttl_seconds")
        with _PUBLICATION_CACHE_LOCK:
            for pmid in normalized:
                entry = _PUBLICATION_CACHE.get(pmid)
                if entry is not None and now - entry[0] < ttl:
                    cached[pmid] = dict(entry[1])
                    _PUBLICATION_CACHE.move_to_end(pmid)
                elif entry is not None:
                    del _PUBLICATION_CACHE[pmid]
        missing = tuple(pmid for pmid in normalized if pmid not in cached)
        receipt = {"requested_pmids": list(normalized), "cache_hit_pmids": list(cached)}
        self._fetch_receipts.append(receipt)
        if not missing:
            receipt.update(returned_pmids=list(cached), complete_pmids=list(cached))
            return cached
        try:
            fetched = await self._fetch_uncached(missing)
        except EuropePMCError:
            receipt.update(returned_pmids=list(cached), complete_pmids=list(cached))
            if cached:
                return cached
            raise
        # A server response may only enrich the explicitly requested identifiers.
        fetched = {pmid: dict(value) for pmid, value in fetched.items()
                   if pmid in missing and isinstance(value, Mapping)}
        with _PUBLICATION_CACHE_LOCK:
            for pmid, metadata in fetched.items():
                if _has_bibliography(metadata):
                    _PUBLICATION_CACHE[pmid] = (time.monotonic(), dict(metadata))
                    _PUBLICATION_CACHE.move_to_end(pmid)
            while len(_PUBLICATION_CACHE) > parameter("retrieval.publication_cache_max_entries"):
                _PUBLICATION_CACHE.popitem(last=False)
        result = {**cached, **fetched}
        receipt.update(returned_pmids=list(result),
                       complete_pmids=[pmid for pmid, row in result.items() if _has_bibliography(row)])
        return result

    async def _attempt(self, provider: str, phase: str, pmids: tuple[str, ...], fetch):
        started = time.monotonic()
        receipt = {"attempt": len(self._attempts) + 1, "provider": provider, "phase": phase,
                   "requested_pmids": list(pmids), "status": "running"}
        self._attempts.append(receipt)
        try:
            result = await asyncio.to_thread(fetch, pmids)
            result = {pmid: dict(row) for pmid, row in result.items()
                      if pmid in pmids and isinstance(row, Mapping)} if isinstance(result, Mapping) else {}
            complete = [pmid for pmid, row in result.items() if _has_bibliography(row)]
            receipt.update(status="complete" if len(complete) == len(pmids) else "partial" if result else "no_data",
                           returned_pmids=list(result), complete_pmids=complete,
                           missing_pmids=[pmid for pmid in pmids if pmid not in complete])
            return result, None
        except (EuropePMCError, OSError, ValueError, TypeError) as exc:
            metadata_error = exc if isinstance(exc, EuropePMCError) else None
            receipt.update(status="failed", returned_pmids=[], complete_pmids=[], missing_pmids=list(pmids),
                           error_category=metadata_error.category if metadata_error else
                           "transport_error" if isinstance(exc, OSError) else "provider_error",
                           cause_type=(metadata_error.cause_type if metadata_error else None) or type(exc).__name__)
            status = metadata_error.http_status if metadata_error else None
            if type(status) is int and 100 <= status <= 599:
                receipt["http_status"] = status
            return {}, exc
        finally:
            receipt["elapsed_ms"] = round(max(0.0, time.monotonic() - started) * 1000, 3)

    def _merge(self, result, incoming):
        for pmid, metadata in incoming.items():
            previous = result.get(pmid, {})
            if previous and publication_metadata_conflicts(_metadata_publication(pmid, previous),
                                                           _metadata_publication(pmid, metadata)):
                self._conflicts.append({"pmid": pmid, "retained_source": previous.get("metadata_source"),
                                        "incoming_source": metadata.get("metadata_source"),
                                        "status": "not_merged"})
                continue
            result[pmid] = {**previous, **{key: value for key, value in metadata.items() if value is not None}}

    async def _fetch_uncached(self, normalized: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """Keep failed/partial lookups retryable; cache only verified complete metadata."""
        result: dict[str, dict[str, Any]] = {}
        first_error = None
        semaphore = asyncio.Semaphore(parameter("execution.publication_retry_concurrency"))
        async def run_batch(provider, phase, batch, fetch):
            async with semaphore:
                return await self._attempt(provider, phase, batch, fetch)

        batch_size = min(_MAX_PMIDS, _METADATA_BATCH_LIMIT, parameter("retrieval.publication_retry_batch_size"))
        for provider, phase, size, fetch in (
            (EUROPE_PMC_SOURCE, "initial", batch_size, self._fetch_sync),
            (EUROPE_PMC_SOURCE, "retry", min(_METADATA_RETRY_LIMIT, batch_size), self._fetch_sync),
            (NCBI_PUBMED_SOURCE, "fallback", batch_size, self._fetch_ncbi_sync),
        ):
            missing = tuple(pmid for pmid in normalized if not _has_bibliography(result.get(pmid, {})))
            if not missing:
                break
            batches = [missing[start:start + size] for start in range(0, len(missing), size)]
            responses = await asyncio.gather(*(run_batch(provider, phase, batch, fetch) for batch in batches))
            for batch, error in responses:
                if error is not None and first_error is None:
                    first_error = error
                self._merge(result, batch)
        if first_error is not None and not result:
            if isinstance(first_error, EuropePMCError):
                raise first_error
            raise EuropePMCError("Exact-identifier publication metadata was unavailable.",
                                 cause_type=type(first_error).__name__) from first_error
        return result

    def _request_json(self, request):
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            status = exc.code if isinstance(exc, HTTPError) else None
            reason = exc.reason if isinstance(exc, URLError) else exc
            category = "http_error" if status is not None else "timeout" if isinstance(reason, TimeoutError) else "transport_error"
            cause = type(reason).__name__ if isinstance(reason, BaseException) and status is None else type(exc).__name__
            raise EuropePMCError("Exact-identifier publication request failed.", category=category,
                                 http_status=status, cause_type=cause) from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise EuropePMCError("Publication response exceeded the size limit", category="response_too_large")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EuropePMCError("Publication response returned invalid JSON", category="invalid_json",
                                 cause_type=type(exc).__name__) from exc
        if not isinstance(value, Mapping):
            raise EuropePMCError("Publication response must be an object", category="invalid_schema")
        return value, raw

    def _fetch_sync(self, pmids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        query = " OR ".join(f"EXT_ID:{pmid}" for pmid in pmids)
        url = EUROPE_PMC_SEARCH_ENDPOINT + "?" + urlencode(
            {
                "query": f"({query}) AND SRC:MED",
                "format": "json",
                "resultType": "core",
                "pageSize": len(pmids),
            }
        )
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "GWASMind/1.0 (publication metadata retrieval)",
            },
        )
        value, raw = self._request_json(request)
        result_list = value.get("resultList")
        rows = result_list.get("result") if isinstance(result_list, Mapping) else None
        if not isinstance(rows, list):
            raise EuropePMCError("Europe PMC response omitted resultList.result", category="invalid_schema")

        requested = set(pmids)
        result: dict[str, dict[str, Any]] = {}
        retrieved_at = datetime.now(UTC).isoformat()
        response_hash = hashlib.sha256(raw).hexdigest()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if row.get("source", "MED") != "MED":
                continue
            pmid = normalize_pmid(row.get("pmid") or row.get("id"))
            if pmid is None or pmid not in requested or pmid in result:
                continue
            if row.get("pmid") is not None and row.get("id") is not None and normalize_pmid(row["id"]) != pmid:
                continue
            metadata: dict[str, Any] = {}
            for output, source in (
                ("publicationTitle", "title"),
                ("authors", "authorString"),
                ("publicationJournal", "journalTitle"),
                ("publicationDate", "firstPublicationDate"),
                ("doi", "doi"),
            ):
                text = str(row.get(source) or "").strip()
                if text:
                    metadata[output] = text
            journal_info = row.get("journalInfo")
            journal_info = journal_info if isinstance(journal_info, Mapping) else {}
            journal = journal_info.get("journal")
            journal = journal if isinstance(journal, Mapping) else {}
            journal_title = journal.get("title") or journal.get("medlineAbbreviation")
            if journal_title and "publicationJournal" not in metadata:
                metadata["publicationJournal"] = str(journal_title).strip()
            # Cite the journal publication year rather than an earlier online
            # release year. Preserve the first-online date separately for audit.
            year = journal_info.get("yearOfPublication") or row.get("pubYear")
            if year:
                if metadata.get("publicationDate"):
                    metadata["firstPublicationDate"] = metadata["publicationDate"]
                metadata["publicationDate"] = str(year).strip()
            abstract = str(row.get("abstractText") or "").strip()
            if abstract:
                metadata["abstractText"] = abstract[:_MAX_ABSTRACT_CHARS]
                if len(abstract) > _MAX_ABSTRACT_CHARS:
                    metadata["abstractTruncated"] = True
            metadata.update(metadata_source=EUROPE_PMC_SOURCE,
                            metadata_retrieved_at=retrieved_at,
                            metadata_response_sha256=response_hash)
            result[pmid] = metadata
        return result

    def _fetch_ncbi_sync(self, pmids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """Retrieve existing PubMed identifiers; no search or article discovery."""
        global _NCBI_LAST_REQUEST
        url = NCBI_PUBMED_ENDPOINT + "?" + urlencode({
            "db": "pubmed", "id": ",".join(pmids), "retmode": "json", "tool": "GWASMind",
        })
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "GWASMind/1.0"})
        # NCBI's no-key limit is three requests per second across E-utilities.
        with _NCBI_REQUEST_LOCK:
            remaining = 0.34 - (time.monotonic() - _NCBI_LAST_REQUEST)
            if remaining > 0:
                time.sleep(remaining)
            _NCBI_LAST_REQUEST = time.monotonic()
            value, raw = self._request_json(request)
        rows = value.get("result")
        if not isinstance(rows, Mapping) or not isinstance(rows.get("uids"), list):
            raise EuropePMCError("NCBI PubMed response omitted result.uids", category="invalid_schema")
        result = {}
        for identifier in rows["uids"]:
            pmid = normalize_pmid(identifier)
            row = rows.get(str(identifier))
            if pmid not in pmids or not isinstance(row, Mapping) or normalize_pmid(row.get("uid")) != pmid:
                continue
            author_rows = row.get("authors")
            author_rows = author_rows if isinstance(author_rows, list) else []
            authors = ", ".join(str(item["name"]).strip() for item in author_rows
                                if isinstance(item, Mapping) and isinstance(item.get("name"), str) and item["name"].strip())
            fields = {"publicationTitle": row.get("title"), "authors": authors,
                      "publicationJournal": row.get("fulljournalname") or row.get("source"),
                      "publicationDate": row.get("pubdate")}
            metadata = {key: value.strip() for key, value in fields.items() if isinstance(value, str) and value.strip()}
            for item in row.get("articleids") or []:
                if isinstance(item, Mapping) and item.get("idtype") == "doi" and isinstance(item.get("value"), str):
                    metadata["doi"] = item["value"].strip()
                    break
            metadata.update(metadata_source=NCBI_PUBMED_SOURCE,
                            metadata_retrieved_at=datetime.now(UTC).isoformat(),
                            metadata_response_sha256=hashlib.sha256(raw).hexdigest())
            result[pmid] = metadata
        return result


def _has_bibliography(metadata: Mapping[str, Any]) -> bool:
    """A PMID or a publication year alone is not a completed citation."""

    if not all(isinstance(metadata.get(field), str) and metadata[field].strip() for field in (
        "publicationTitle", "authors", "publicationJournal", "publicationDate",
    )):
        return False
    year = re.match(r"^(\d{4})(?:\b|[-/])", metadata["publicationDate"].strip())
    return bool(year and 1800 <= int(year.group(1)) <= 3000)


def _metadata_publication(pmid: str, metadata: Mapping[str, Any]) -> Publication:
    def text(key):
        value = metadata.get(key)
        return " ".join(value.split()) if isinstance(value, str) and value.strip() else None
    date = text("publicationDate")
    match = re.match(r"^(\d{4})(?:\b|[-/])", date) if date else None
    year = int(match.group(1)) if match and 1800 <= int(match.group(1)) <= 3000 else None
    return Publication(pmid, authors=text("authors"), title=text("publicationTitle"),
                       journal=text("publicationJournal"), year=year, doi=text("doi"))


async def enrich_citation_registry(
    registry: CitationRegistry, *, client: EuropePMCPublicationClient | None = None,
) -> dict[str, Any]:
    """Enrich only incomplete, already numbered citations; never discover papers.

    The renderer invokes this through an explicit runtime callback. Pure rendering
    and offline callers remain network-free. Unavailable metadata does not change
    evidence, ranking, citation membership or numbering.
    """
    registry.fill_known_metadata()
    missing = [entry.publication.pmid for entry in registry.entries if not entry.publication.metadata_complete]
    requested = missing[:parameter("retrieval.cited_publication_enrichment_pmids")]
    audit: dict[str, Any] = {"requested_pmids": requested, "resolved_pmids": [],
        "unresolved_pmids": missing, "conflicting_pmids": [], "errors": [], "source": EUROPE_PMC_SOURCE,
        "bibliography": {entry.publication.pmid: {**asdict(entry.publication), "source": "provided evidence or exact reference registry"}
                         for entry in registry.entries}, "conflicts": {}, "attempts": [], "coverage": {}}
    if not requested:
        audit["coverage"] = {entry.publication.pmid: {"status": "complete", "requested": False,
                                                    "source": "provided evidence or exact reference registry"}
                             for entry in registry.entries}
        return audit
    client = client or EuropePMCPublicationClient()
    previous_diagnostics = client.diagnostics if isinstance(client, EuropePMCPublicationClient) else {}
    diagnostic_offsets = {key: len(previous_diagnostics.get(key, []))
                          for key in ("attempts", "fetches", "provider_conflicts")}
    semaphore = asyncio.Semaphore(parameter("execution.publication_retry_concurrency"))

    async def batch(ids):
        async with semaphore:
            try:
                value = await client.fetch(ids)
                return value if isinstance(value, Mapping) else {}
            except (EuropePMCError, OSError, ValueError, TypeError) as exc:
                audit["errors"].append(type(exc).__name__)
                return {}

    responses = await asyncio.gather(*(batch(requested[start:start + _MAX_PMIDS])
        for start in range(0, len(requested), _MAX_PMIDS)))
    diagnostics = client.diagnostics if isinstance(client, EuropePMCPublicationClient) else {}
    audit["attempts"] = diagnostics.get("attempts", [])[diagnostic_offsets["attempts"]:]
    audit["fetches"] = diagnostics.get("fetches", [])[diagnostic_offsets["fetches"]:]
    audit["provider_conflicts"] = diagnostics.get("provider_conflicts", [])[diagnostic_offsets["provider_conflicts"]:]
    audit["providers_used"] = list(dict.fromkeys(attempt["provider"] for attempt in audit["attempts"]))
    by_pmid = {entry.publication.pmid: entry.publication for entry in registry.entries}
    for response in responses:
        for pmid, metadata in response.items():
            if pmid not in requested or not isinstance(metadata, Mapping):
                continue
            def text(key, fields=metadata):
                value = fields.get(key)
                return " ".join(value.split()) if isinstance(value, str) and value.strip() else None
            publication = _metadata_publication(pmid, metadata)
            if publication_metadata_conflicts(by_pmid[pmid], publication):
                audit["conflicting_pmids"].append(pmid)
                audit["conflicts"][pmid] = {"retained": asdict(by_pmid[pmid]), "incoming": asdict(publication),
                    "source": text("metadata_source") or EUROPE_PMC_SOURCE,
                    "retrieved_at": text("metadata_retrieved_at"), "response_sha256": text("metadata_response_sha256")}
                continue
            registry.register(publication)
            audit["bibliography"][pmid] = {**asdict(registry.entries[registry.number_for(pmid) - 1].publication),
                "source": "provided evidence supplemented by " + (text("metadata_source") or EUROPE_PMC_SOURCE),
                "retrieved_at": text("metadata_retrieved_at"), "response_sha256": text("metadata_response_sha256")}
    audit["unresolved_pmids"] = [entry.publication.pmid for entry in registry.entries if not entry.publication.metadata_complete]
    audit["resolved_pmids"] = [pmid for pmid in requested if pmid not in audit["unresolved_pmids"]]
    cached = {pmid for fetch in audit["fetches"] for pmid in fetch.get("cache_hit_pmids", [])}
    for entry in registry.entries:
        pmid, publication = entry.publication.pmid, entry.publication
        present = [name for name in ("authors", "title", "journal", "year", "doi") if getattr(publication, name)]
        status = ("complete" if publication.metadata_complete else "budget_excluded" if pmid not in requested
                  else "partial" if present else "unavailable")
        audit["coverage"][pmid] = {"status": status, "requested": pmid in requested,
                                  "cache_hit": pmid in cached, "available_fields": present,
                                  "attempts": [attempt["attempt"] for attempt in audit["attempts"]
                                               if pmid in attempt["requested_pmids"]]}
    return audit
