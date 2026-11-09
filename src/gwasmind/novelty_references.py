"""Project recorded prior-report publications without querying a provider."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace

from .publications import (
    CitationRegistry, Publication, normalize_pmid, publication_from_metadata,
    publication_metadata_conflicts,
)

_PUBMED = re.compile(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/([1-9]\d{0,11})(?:/|(?=[)\s]|$))")
_PMID_LABEL = re.compile(r"\bPMID\s*:\s*([1-9]\d{0,11})\b", re.I)
_CITATION = re.compile(r"\{\{\s*cite\s*:\s*([A-Za-z0-9_.-]+)\s*\}\}")
_LEGACY_PREFIX = "Published GWAS supporting the exact selected lead-SNP association, retrieved through Open Targets."


def _reference_pmid(reference):
    identifiers = set(_PUBMED.findall(reference.markdown)) | set(_PMID_LABEL.findall(reference.markdown))
    return next(iter(identifiers)) if len(identifiers) == 1 else None


def _has_bibliography(markdown):
    text = markdown.replace(_LEGACY_PREFIX, "").replace("Publication metadata unavailable.", "")
    text = re.sub(r"\[PMID\s*:\s*[1-9]\d{0,11}\]\([^)]*\)", "", text, flags=re.I)
    text = _PUBMED.sub("", text)
    text = _PMID_LABEL.sub("", text)
    return bool(text.strip(" .\n\t[]()"))


def _screen_rows(screen):
    for row in screen.get("native_exact_report_matches") or ():
        if isinstance(row, Mapping):
            yield row
    for group in screen.get("prior_associations") or ():
        if isinstance(group, Mapping):
            yield from (row for row in group.get("associations") or () if isinstance(row, Mapping))
    for publication in screen.get("publications") or ():
        yield publication if isinstance(publication, Mapping) else {"pmid": publication}


def _record_publication(row):
    """Metadata must name the same PMID as its record before it can be used."""
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    upstream = payload.get("upstream") if isinstance(payload.get("upstream"), Mapping) else {}
    raw = upstream.get("raw") if isinstance(upstream.get("raw"), Mapping) else {}
    sources = (row, payload, raw)
    identifiers = {normalize_pmid(source.get(key)) for source in sources
                   for key in ("pmid", "pubmedId", "pubmed_id") if source.get(key) not in (None, "")}
    if len(identifiers) != 1 or None in identifiers:
        return None

    def field(keys, limit):
        for source in sources:
            for key in keys:
                value = source.get(key)
                if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                    return str(value).strip()[:limit]
        return None

    return publication_from_metadata(next(iter(identifiers)),
        authors=field(("publication_author", "publicationFirstAuthor", "authors"), 300),
        title=field(("publication_title", "publicationTitle", "title"), 1000),
        journal=field(("publication_journal", "publicationJournal", "journal"), 300),
        publication_date=field(("publication_date", "publicationDate", "year"), 40),
        doi=field(("doi",), 200))


def prior_report_references(context, screen, resource_references):
    """Return a deduplicated context and reference IDs for its prior-report fact.

    Existing publication IDs stay stable. Duplicate IDs in display facts are
    redirected before the final renderer assigns citation numbers. Only derived
    reference text changes; the result records and model outputs stay untouched.
    """
    from .staged_reasoning import FinalReference

    screen_rows = tuple(_screen_rows(screen))
    pmids = tuple(dict.fromkeys(p.pmid for row in screen_rows if (p := _record_publication(row))))
    registry = CitationRegistry()
    evidence = context.result.get("evidence") or context.result.get("evidence_ledger") or ()
    for row in (*screen_rows, *(row for row in evidence if isinstance(row, Mapping))):
        publication = _record_publication(row)
        if publication is None or publication.pmid not in pmids:
            continue
        number = registry.number_for(publication.pmid)
        if number is not None and publication_metadata_conflicts(registry.entries[number - 1].publication, publication):
            continue
        registry.register(publication)
    publications = {entry.publication.pmid: entry.publication for entry in registry.entries}
    references, by_pmid, by_id, remap = [], {}, {}, {}

    def retain(reference):
        pmid = _reference_pmid(reference)
        previous = by_pmid.get(pmid) if pmid else by_id.get(reference.reference_id)
        if previous is not None:
            canonical = references[previous]
            remap[reference.reference_id] = canonical.reference_id
            if not _has_bibliography(canonical.markdown) and _has_bibliography(reference.markdown):
                references[previous] = replace(canonical, markdown=reference.markdown)
            return canonical.reference_id
        if reference.reference_id in by_id:
            suffix = 2
            while reference.reference_id + "_" + str(suffix) in by_id:
                suffix += 1
            reference = replace(reference, reference_id=reference.reference_id + "_" + str(suffix))
        by_id[reference.reference_id] = len(references)
        if pmid:
            by_pmid[pmid] = len(references)
        references.append(reference)
        return reference.reference_id

    for reference in context.references:
        retain(reference)
    selected = []
    for pmid in pmids:
        selected.append(retain(FinalReference("prior_native_pmid_" + pmid,
            publications.get(pmid, Publication(pmid)).markdown())))
    selected.extend(retain(reference) for reference in resource_references)
    references = [replace(reference, markdown=publications.get(pmid, Publication(pmid)).markdown())
                  if (pmid := _reference_pmid(reference)) and not _has_bibliography(reference.markdown)
                  else reference for reference in references]

    def citations(text):
        return _CITATION.sub(lambda match: "{{cite:" + remap.get(match[1], match[1]) + "}}", text)

    facts = tuple(replace(fact,
        reference_ids=tuple(dict.fromkeys(remap.get(identifier, identifier) for identifier in fact.reference_ids)),
        markdown=citations(fact.markdown),
        model_data={**fact.model_data, **({"deterministic_detail": citations(fact.model_data["deterministic_detail"])}
            if isinstance(fact.model_data.get("deterministic_detail"), str) else {})})
        for fact in context.facts) if remap else context.facts
    return replace(context, references=tuple(references), facts=facts), tuple(dict.fromkeys(selected))
