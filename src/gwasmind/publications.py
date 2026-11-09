from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_PMID_PATTERN = re.compile(r"(?:PMID\s*:\s*)?([1-9]\d{0,11})", re.IGNORECASE)
_NON_IDENTIFIER = re.compile(r"[^a-z0-9]+")


def normalize_pmid(value: Any) -> str | None:
    """Normalize a PMID without accepting URLs, free text, zero, or signs."""

    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    match = _PMID_PATTERN.fullmatch(text)
    return match.group(1) if match else None


@dataclass(frozen=True, slots=True)
class Publication:
    """A canonical publication, or a PMID-only safe fallback.

    Unknown PMIDs deliberately carry no inferred author, title, journal, or year.
    This keeps an incomplete Open Targets study record traceable without assigning
    the publication to Open Targets or fabricating bibliographic metadata.
    """

    pmid: str
    authors: str | None = None
    title: str | None = None
    journal: str | None = None
    year: int | None = None
    doi: str | None = None

    def __post_init__(self) -> None:
        normalized = normalize_pmid(self.pmid)
        if normalized is None:
            raise ValueError("pmid must be a positive PubMed identifier")
        object.__setattr__(self, "pmid", normalized)
        if self.year is not None and not 1800 <= self.year <= 3000:
            raise ValueError("publication year is outside the supported range")

    @property
    def key(self) -> str:
        return f"pmid:{self.pmid}"

    @property
    def pubmed_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def metadata_complete(self) -> bool:
        return all((self.authors, self.title, self.journal, self.year))

    def markdown(self) -> str:
        """Return a reference body without a leading citation number."""

        pieces = tuple(value for value in (
            _compact_authors(str(self.authors)).rstrip(". ") if self.authors else "",
            str(self.title).rstrip(". ") if self.title else "",
            str(self.journal).rstrip(". ") if self.journal else "",
            str(self.year) if self.year else "",
        ) if value)
        prefix = ". ".join(pieces) + ". " if pieces else ""
        return prefix + f"[PMID: {self.pmid}]({self.pubmed_url})"


def _compact_authors(authors: str) -> str:
    """Display the first author of an explicit author list; retain full metadata."""

    text = " ".join(authors.split()).strip()
    if re.search(r"\bet\s+al\.?\b", text, re.IGNORECASE):
        return text
    # Europe PMC uses 'Surname AB, Surname CD, ...'. Do not mistake a single
    # inverted full name ('Surname, Given name') for multiple authors.
    names = re.split(r"\s*;\s*|,\s+(?=[^,;]+?\s+[A-Z]{1,8}(?:[,;.]|$))", text)
    if len(names) > 1 and names[0].strip():
        return names[0].strip().rstrip(".") + " et al."
    return text


_PUBLICATIONS = {
    "32913098": Publication(
        pmid="32913098",
        authors="GTEx Consortium",
        title="The GTEx Consortium atlas of genetic regulatory effects across human tissues",
        journal="Science",
        year=2020,
        doi="10.1126/science.aaz1776",
    ),
    "36653562": Publication(
        pmid="36653562",
        authors="Kurki MI et al.",
        title="FinnGen provides genetic insights from a well-phenotyped isolated population",
        journal="Nature",
        year=2023,
        doi="10.1038/s41586-022-05473-8",
    ),
    "34711957": Publication(
        pmid="34711957",
        authors="Mountjoy E et al.",
        title=(
            "An open approach to systematically prioritize causal variants and genes at all "
            "published human GWAS trait-associated loci"
        ),
        journal="Nature Genetics",
        year=2021,
        doi="10.1038/s41588-021-00945-5",
    ),
    "35549404": Publication(
        pmid="35549404",
        authors="Tabula Sapiens Consortium et al.",
        title="The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans",
        journal="Science",
        year=2022,
        doi="10.1126/science.abl4896",
    ),
    # Verified against the PubMed record; a frequently used single-cell eQTL source.
    "35545678": Publication(
        pmid="35545678",
        authors="Nathan A et al.",
        title="Single-cell eQTL models reveal dynamic T cell state dependence of disease loci",
        journal="Nature",
        year=2022,
        doi="10.1038/s41586-022-04713-1",
    ),
    "30449622": Publication(
        pmid="30449622",
        authors="Schmiedel BJ et al.",
        title="Impact of Genetic Polymorphisms on Human Immune Cell Gene Expression",
        journal="Cell",
        year=2018,
        doi="10.1016/j.cell.2018.10.022",
    ),
    "39494541": Publication(
        pmid="39494541",
        authors="Perez-Riverol Y et al.",
        title="The PRIDE database at 20 years: 2025 update",
        journal="Nucleic Acids Research",
        year=2025,
        doi="10.1093/nar/gkae1011",
    ),
    "33290552": Publication(
        pmid="33290552",
        authors="Gene Ontology Consortium",
        title="The Gene Ontology resource: enriching a GOld mine",
        journal="Nucleic Acids Research",
        year=2021,
        doi="10.1093/nar/gkaa1113",
    ),
    "34788843": Publication(
        pmid="34788843",
        authors="Gillespie M et al.",
        title="The Reactome pathway knowledgebase 2022",
        journal="Nucleic Acids Research",
        year=2022,
        doi="10.1093/nar/gkab1028",
    ),
    "36408920": Publication(
        pmid="36408920",
        authors="UniProt Consortium",
        title="UniProt: the Universal Protein Knowledgebase in 2023",
        journal="Nucleic Acids Research",
        year=2023,
        doi="10.1093/nar/gkac1052",
    ),
    "38014075": Publication(
        pmid="38014075",
        authors="Gschwind AR et al.",
        title="An encyclopedia of enhancer-gene regulatory interactions in the human genome",
        journal="bioRxiv",
        year=2023,
        doi="10.1101/2023.11.09.563812",
    ),
}


PUBLICATION_REGISTRY: Mapping[str, Publication] = MappingProxyType(_PUBLICATIONS)


# Prefixes are matched case-insensitively, longest first. They are used only when
# an explicit PMID is absent; an upstream PMID always wins, even when unknown.
_STUDY_PREFIX_PMIDS: tuple[tuple[str, str], ...] = (
    ("finngen_", "36653562"),
    ("gtex_", "32913098"),
    ("gtex-", "32913098"),
)


_DATASOURCE_PMIDS: Mapping[str, str] = MappingProxyType(
    {
        "gtex": "32913098",
        "finngen": "36653562",
        "open_targets_l2g": "34711957",
        "l2g": "34711957",
        "open_targets_colocalisation": "34711957",
        "open_targets_colocalization": "34711957",
        "colocalisation": "34711957",
        "colocalization": "34711957",
        "tabula_sapiens": "35549404",
        "dice": "30449622",
        "pride": "39494541",
        "pride_otar3091": "39494541",
        "otar3091": "39494541",
        "gene_ontology": "33290552",
        "go": "33290552",
        "reactome": "34788843",
        "uniprot": "36408920",
        "enhancer_to_gene": "38014075",
        "encode_re2g": "38014075",
        "e2g": "38014075",
    }
)


def canonical_publication(pmid: Any) -> Publication | None:
    """Resolve a PMID, returning a PMID-only publication when it is unknown."""

    normalized = normalize_pmid(pmid)
    if normalized is None:
        return None
    return PUBLICATION_REGISTRY.get(normalized, Publication(normalized))


def publication_from_metadata(
    pmid: Any,
    *,
    authors: Any = None,
    title: Any = None,
    journal: Any = None,
    publication_date: Any = None,
    doi: Any = None,
) -> Publication | None:
    """Resolve a PMID while preserving complete upstream bibliography metadata.

    The curated registry remains authoritative for known resources.  For an
    otherwise unknown PMID, study metadata returned by the provider is preferable
    to an unlabelled PMID. Partial metadata is retained as supplied; absent fields
    remain absent rather than being inferred or replaced with invented text.
    """

    normalized = normalize_pmid(pmid)
    if normalized is None:
        return None
    canonical = PUBLICATION_REGISTRY.get(normalized)
    if canonical is not None:
        return canonical
    author_text = _first_text(authors)
    title_text = _first_text(title)
    journal_text = _first_text(journal)
    date_text = _first_text(publication_date)
    year = None
    if date_text is not None:
        match = re.match(r"^(\d{4})", date_text)
        if match and 1800 <= int(match.group(1)) <= 3000:
            year = int(match.group(1))
    return Publication(
        normalized,
        authors=author_text,
        title=title_text,
        journal=journal_text,
        year=year,
        doi=_first_text(doi),
    )


def resolve_study_publication(
    study_id: Any,
    *,
    pmid: Any = None,
) -> Publication | None:
    """Resolve an explicit study PMID or, when absent, a known study prefix."""

    explicit = normalize_pmid(pmid)
    if explicit is not None:
        return canonical_publication(explicit)
    study = str(study_id or "").strip().casefold()
    for prefix, mapped_pmid in sorted(
        _STUDY_PREFIX_PMIDS,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if study.startswith(prefix):
            return PUBLICATION_REGISTRY[mapped_pmid]
    return None


def resolve_datasource_publication(
    datasource_id: Any,
    *,
    pmid: Any = None,
) -> Publication | None:
    """Resolve an explicit PMID or a versioned datasource alias."""

    explicit = normalize_pmid(pmid)
    if explicit is not None:
        return canonical_publication(explicit)
    key = _normalize_identifier(datasource_id)
    mapped_pmid = _DATASOURCE_PMIDS.get(key)
    return PUBLICATION_REGISTRY[mapped_pmid] if mapped_pmid else None


def resolve_publication(
    *,
    pmid: Any = None,
    study_id: Any = None,
    datasource_id: Any = None,
) -> Publication | None:
    """Resolve publication provenance with explicit PMID taking precedence."""

    explicit = normalize_pmid(pmid)
    if explicit is not None:
        return canonical_publication(explicit)
    study = resolve_study_publication(study_id)
    if study is not None:
        return study
    return resolve_datasource_publication(datasource_id)


@dataclass(frozen=True, slots=True)
class PublicationLineageSide:
    role: str
    study_id: str | None
    study_locus_id: str | None
    study_type: str | None
    trait: str | None
    publication: Publication | None


@dataclass(frozen=True, slots=True)
class ColocalizationPublicationLineage:
    left: PublicationLineageSide
    right: PublicationLineageSide
    method: str | None

    def publications(self) -> tuple[Publication, ...]:
        """Return left then right publications, deduplicated by PMID."""

        values: list[Publication] = []
        seen: set[str] = set()
        for publication in (self.left.publication, self.right.publication):
            if publication is None or publication.pmid in seen:
                continue
            seen.add(publication.pmid)
            values.append(publication)
        return tuple(values)


def parse_colocalization_publication_lineage(
    record: Mapping[str, Any],
) -> ColocalizationPublicationLineage:
    """Extract the two publication lineages from a semantic or raw coloc record.

    Current Open Targets-backed GWASMind records use the left side for the queried
    molecular-QTL credible set and the right side for the colocalising study. Some
    historical records copy the right publication into generic ``pubmedId`` and
    ``context.pmid`` fields. Those generic fields are therefore considered right-side
    aliases whenever a right study is present; they are never assigned to the left
    study implicitly.
    """

    payload = _mapping(record.get("payload"))
    context = _mapping(payload.get("context")) or _mapping(record.get("context"))
    raw = _raw_mapping(record, payload)

    left_study_id = _first_text(
        raw.get("leftStudyId"),
        raw.get("qtlStudyId"),
        raw.get("studyId"),
        context.get("study_id"),
    )
    left_pmid = _first_pmid(
        raw.get("leftPubmedId"),
        raw.get("leftPmid"),
        raw.get("qtlPubmedId"),
    )
    left_publication = publication_from_metadata(
        left_pmid,
        authors=raw.get("leftPublicationFirstAuthor"),
        title=raw.get("leftPublicationTitle"),
        journal=raw.get("leftPublicationJournal"),
        publication_date=raw.get("leftPublicationDate"),
    ) or resolve_study_publication(left_study_id)
    left = PublicationLineageSide(
        role="left",
        study_id=left_study_id,
        study_locus_id=_first_text(
            raw.get("leftStudyLocusId"),
            raw.get("qtlStudyLocusId"),
            raw.get("queriedStudyLocusId"),
            raw.get("studyLocusId"),
            context.get("study_locus_id"),
        ),
        study_type=_first_text(
            raw.get("leftStudyType"),
            raw.get("qtlStudyType"),
            raw.get("studyType"),
            context.get("study_type"),
        ),
        trait=_first_text(
            raw.get("leftTraitFromSource"),
            raw.get("qtlTraitFromSource"),
        ),
        publication=left_publication,
    )

    right_study_id = _first_text(
        raw.get("rightStudyId"),
        raw.get("otherStudyId"),
    )
    right_pmid = _first_pmid(
        raw.get("rightPubmedId"),
        raw.get("rightPmid"),
        raw.get("otherPubmedId"),
    )
    explicit_left_lineage = any(
        key in raw
        for key in (
            "leftStudyId",
            "leftStudyLocusId",
            "leftPubmedId",
            "leftPublicationTitle",
        )
    )
    if right_study_id is not None and right_pmid is None and not explicit_left_lineage:
        right_pmid = _first_pmid(
            raw.get("pubmedId"),
            context.get("pmid"),
            record.get("pmid"),
        )
    generic_pmid = _first_pmid(raw.get("pubmedId"), context.get("pmid"), record.get("pmid"))
    generic_right_metadata = (
        not explicit_left_lineage or generic_pmid is not None and generic_pmid == right_pmid
    )
    right_publication = publication_from_metadata(
        right_pmid,
        authors=raw.get("rightPublicationFirstAuthor") or (
            raw.get("publicationFirstAuthor") if generic_right_metadata else None
        ),
        title=raw.get("rightPublicationTitle") or (
            raw.get("publicationTitle") if generic_right_metadata else None
        ),
        journal=raw.get("rightPublicationJournal") or (
            raw.get("publicationJournal") if generic_right_metadata else None
        ),
        publication_date=raw.get("rightPublicationDate") or (
            raw.get("publicationDate") if generic_right_metadata else None
        ),
    ) or resolve_study_publication(right_study_id)
    right = PublicationLineageSide(
        role="right",
        study_id=right_study_id,
        study_locus_id=_first_text(
            raw.get("rightStudyLocusId"),
            raw.get("otherStudyLocusId"),
        ),
        study_type=_first_text(
            raw.get("rightStudyType"),
            raw.get("otherStudyType"),
        ),
        trait=_first_text(
            raw.get("rightTraitFromSource"),
            raw.get("otherTraitFromSource"),
            raw.get("traitFromSource") if right_study_id is not None else None,
        ),
        publication=right_publication,
    )
    return ColocalizationPublicationLineage(
        left=left,
        right=right,
        method=_first_text(raw.get("colocalisationMethod"), raw.get("colocalizationMethod")),
    )


@dataclass(frozen=True, slots=True)
class NumberedPublication:
    number: int
    publication: Publication

    def markdown(self) -> str:
        return f"[{self.number}] {self.publication.markdown()}"


class CitationRegistry:
    """First-use, PMID-deduplicated citation numbering."""

    def __init__(self) -> None:
        self._entries: list[NumberedPublication] = []
        self._numbers: dict[str, int] = {}

    def register(self, publication: Publication | Any) -> int | None:
        resolved = (
            publication
            if isinstance(publication, Publication)
            else canonical_publication(publication)
        )
        if resolved is None:
            return None
        if resolved.pmid in self._numbers:
            number = self._numbers[resolved.pmid]
            previous = self._entries[number - 1].publication
            # Enrichment can arrive after a PMID-only trace was registered.
            # Fill missing metadata for that same PMID without renumbering it or
            # borrowing information from a different study/publication.
            merged = Publication(
                resolved.pmid, authors=previous.authors or resolved.authors,
                title=previous.title or resolved.title, journal=previous.journal or resolved.journal,
                year=previous.year or resolved.year, doi=previous.doi or resolved.doi,
            )
            if merged is not None:
                self._entries[number - 1] = NumberedPublication(number, merged)
            return number
        number = len(self._entries) + 1
        self._numbers[resolved.pmid] = number
        self._entries.append(NumberedPublication(number, resolved))
        return number

    def register_lineage(
        self,
        lineage: ColocalizationPublicationLineage,
    ) -> tuple[int, ...]:
        return tuple(
            number
            for publication in lineage.publications()
            if (number := self.register(publication)) is not None
        )

    def number_for(self, pmid: Any) -> int | None:
        normalized = normalize_pmid(pmid)
        return self._numbers.get(normalized) if normalized else None

    @property
    def entries(self) -> tuple[NumberedPublication, ...]:
        return tuple(self._entries)

    def markdown(self) -> str:
        return "\n".join(entry.markdown() for entry in self._entries)

    def fill_known_metadata(self) -> None:
        """Fill registered PMIDs from the existing exact reference registry only."""
        for entry in self.entries:
            publication = PUBLICATION_REGISTRY.get(entry.publication.pmid)
            if publication is not None and not publication_metadata_conflicts(entry.publication, publication):
                self.register(publication)


def publication_metadata_conflicts(previous: Publication, incoming: Publication) -> bool:
    """Do not combine contradictory nonempty metadata into a synthetic citation."""
    def normalize(value):
        return " ".join(str(value).casefold().strip().rstrip('.').split())

    if previous.pmid != incoming.pmid:
        return True
    for field in ("title", "journal", "year", "doi"):
        before, after = getattr(previous, field), getattr(incoming, field)
        if before and after and normalize(before) != normalize(after):
            return True
    # Provider records often retain only the first author; a fuller author list
    # for that same first author is enrichment, not a conflicting identity.
    if previous.authors and incoming.authors:
        def first_author(value):
            return re.sub(r"\bet\s+al\.?", "", _compact_authors(value), flags=re.IGNORECASE).strip().casefold().rstrip('.')
        if first_author(previous.authors) != first_author(incoming.authors):
            return True
    return False


def _normalize_identifier(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return _NON_IDENTIFIER.sub("_", text).strip("_")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _raw_mapping(
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload_upstream = _mapping(payload.get("upstream"))
    payload_raw = _mapping(payload_upstream.get("raw"))
    if payload_raw:
        return payload_raw
    record_upstream = _mapping(record.get("upstream"))
    record_raw = _mapping(record_upstream.get("raw"))
    if record_raw:
        return record_raw
    return payload or record


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None or isinstance(value, (bool, Mapping, list, tuple, set, frozenset)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_pmid(*values: Any) -> str | None:
    for value in values:
        if (pmid := normalize_pmid(value)) is not None:
            return pmid
    return None


__all__ = [
    "PUBLICATION_REGISTRY",
    "CitationRegistry",
    "ColocalizationPublicationLineage",
    "NumberedPublication",
    "Publication",
    "PublicationLineageSide",
    "canonical_publication",
    "normalize_pmid",
    "parse_colocalization_publication_lineage",
    "publication_from_metadata",
    "resolve_datasource_publication",
    "resolve_publication",
    "resolve_study_publication",
]
