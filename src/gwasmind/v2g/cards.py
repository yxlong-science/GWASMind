"""Model-facing gene evidence cards and candidate-identity anonymization."""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..configuration import parameter
from .evidence import EVIDENCE_CATEGORIES, strip_private

CARD_SECTIONS = ("auxiliary", *EVIDENCE_CATEGORIES)
LOCUS_LINK_CATEGORIES = frozenset({"coding", "qtl", "coloc", "e2g"})
AMBIGUOUS_GENE_TOKEN = "[candidate gene]"
_CASE_INSENSITIVE_ID = re.compile(r"^(?:ENS[GTP]\d{11}(?:\.\d+)?|[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$",
                                  re.IGNORECASE)
_MIN_CASE_INSENSITIVE_SYMBOL = 5
_MIN_NAME_LENGTH = 6


class ResidualIdentifierError(ValueError):
    """Anonymized model input still contains a candidate-gene identifier."""


def assign_codes(gene_ids: Sequence[str], seed: str) -> dict[str, str]:
    """Stable, seed-dependent random codes G01..Gnn."""

    ordered = sorted(gene_ids, key=lambda gene: hashlib.sha256(f"{seed}:{gene}".encode()).hexdigest())
    width = max(2, len(str(len(ordered))))
    return {gene: f"G{index:0{width}d}" for index, gene in enumerate(ordered, 1)}


def _identifiers(identity: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """(case-insensitive IDs, symbols, names) of one gene."""

    ids = [identity.get("gene_id"), *identity.get("transcript_ids", ()), *identity.get("protein_ids", ())]
    symbols = [identity.get("symbol"), *identity.get("symbol_synonyms", ()), *identity.get("obsolete_symbols", ())]
    names = [identity.get("name"), *identity.get("name_synonyms", ()), *identity.get("obsolete_names", ())]

    def clean(values: Iterable[Any]) -> list[str]:
        return [str(value).strip() for value in values if value and len(str(value).strip()) >= 2]

    return clean(ids), clean(symbols), [value for value in clean(names) if len(value) >= _MIN_NAME_LENGTH]


def _pattern(terms: Iterable[str], *, ignore_case: bool) -> re.Pattern[str] | None:
    unique = sorted(set(terms), key=lambda term: (-len(term), term))
    if not unique:
        return None
    body = "|".join(re.escape(term) for term in unique)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", re.IGNORECASE if ignore_case else 0)


@dataclass(slots=True)
class IdentityScrubber:
    code_by_gene: Mapping[str, str]
    replacements_ci: dict[str, str] = field(default_factory=dict)
    replacements_cs: dict[str, str] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)

    @classmethod
    def from_identities(cls, identities: Sequence[Mapping[str, Any]], code_by_gene: Mapping[str, str]):
        scrubber = cls(code_by_gene)
        owners_ci: dict[str, set[str]] = {}
        owners_cs: dict[str, set[str]] = {}
        for identity in identities:
            gene = str(identity["gene_id"])
            ids, symbols, names = _identifiers(identity)
            for term in ids + names:
                owners_ci.setdefault(term.casefold(), set()).add(gene)
            for symbol in symbols:
                if len(symbol) >= _MIN_CASE_INSENSITIVE_SYMBOL or _CASE_INSENSITIVE_ID.fullmatch(symbol):
                    owners_ci.setdefault(symbol.casefold(), set()).add(gene)
                else:
                    owners_cs.setdefault(symbol, set()).add(gene)
        for owners, target in ((owners_ci, scrubber.replacements_ci), (owners_cs, scrubber.replacements_cs)):
            for term, genes in owners.items():
                if len(genes) == 1:
                    target[term] = code_by_gene[next(iter(genes))]
                else:
                    target[term] = AMBIGUOUS_GENE_TOKEN
                    scrubber.ambiguous.add(term)
        return scrubber

    def _patterns(self):
        return (_pattern(self.replacements_ci, ignore_case=True), _pattern(self.replacements_cs, ignore_case=False))

    def scrub_text(self, text: str) -> str:
        ci, cs = self._patterns()
        if ci is not None:
            text = ci.sub(lambda match: self.replacements_ci[match.group(0).casefold()], text)
        if cs is not None:
            text = cs.sub(lambda match: self.replacements_cs[match.group(0)], text)
        return text

    def scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, Mapping):
            return {key: self.scrub(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.scrub(item) for item in value]
        return value

    def residuals(self, value: Any) -> list[str]:
        ci, cs = self._patterns()
        found: list[str] = []

        def walk(item: Any) -> None:
            if isinstance(item, str):
                for pattern in (ci, cs):
                    if pattern is not None:
                        found.extend(match.group(0) for match in pattern.finditer(item))
            elif isinstance(item, Mapping):
                for key, child in item.items():
                    walk(key)
                    walk(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    walk(child)

        walk(value)
        return sorted(set(found))


@dataclass(slots=True)
class CardSet:
    cards: list[dict[str, Any]]
    anonymized: bool
    seed: str
    order: str
    code_by_gene: dict[str, str]
    channels: tuple[str, ...]
    ambiguous_identifiers: int = 0
    residual_identifiers: int = 0

    def gene_for_card(self) -> dict[str, str]:
        return {code: gene for gene, code in self.code_by_gene.items()}

    def audit(self) -> dict[str, Any]:
        return {"anonymized": self.anonymized, "seed": self.seed, "order": self.order,
                "code_by_gene": dict(self.code_by_gene), "channels": list(self.channels),
                "ambiguous_identifiers": self.ambiguous_identifiers,
                "residual_identifiers": self.residual_identifiers}


def _card(gene: Mapping[str, Any], code: str, sections: Sequence[str], *, named: bool) -> dict[str, Any]:
    card: dict[str, Any] = {"id": code}
    if named:
        card["symbol"] = gene["identity"].get("symbol")
        card["name"] = gene["identity"].get("name")
    if "auxiliary" in sections:
        card["auxiliary"] = gene["auxiliary"]
    categories = [section for section in sections if section in EVIDENCE_CATEGORIES]
    if categories:
        card["evidence"] = {category: gene["evidence"][category] for category in categories}
    return card


def build_gene_cards(genes: Sequence[Mapping[str, Any]], *, seed: str, anonymize: bool | None = None,
                     channels: Sequence[str] = CARD_SECTIONS, order: str = "genomic") -> CardSet:
    """Model-facing cards for every gene; ``channels`` selects card sections for ablation arms.

    Anonymized cards never leave this function with a residual identifier.
    """

    anonymize = parameter("v2g.anonymize_gene_identity") if anonymize is None else anonymize
    unknown = set(channels) - set(CARD_SECTIONS)
    if unknown:
        raise ValueError(f"unknown card sections: {sorted(unknown)}")
    genes = [strip_private(gene) for gene in genes]
    gene_ids = [gene["gene_id"] for gene in genes]
    codes = assign_codes(gene_ids, seed) if anonymize else {gene: gene for gene in gene_ids}
    ordered = list(genes)
    if order == "shuffled":
        random.Random(hashlib.sha256(f"order:{seed}".encode()).hexdigest()).shuffle(ordered)
    elif order != "genomic":
        raise ValueError("order must be genomic or shuffled")
    cards = [_card(gene, codes[gene["gene_id"]], channels, named=not anonymize) for gene in ordered]
    card_set = CardSet(cards=cards, anonymized=anonymize, seed=seed, order=order, code_by_gene=codes,
                       channels=tuple(channels))
    if anonymize:
        scrubber = IdentityScrubber.from_identities([gene["identity"] for gene in genes], codes)
        card_set.cards = scrubber.scrub(cards)
        card_set.ambiguous_identifiers = len(scrubber.ambiguous)
        residuals = scrubber.residuals(card_set.cards)
        card_set.residual_identifiers = len(residuals)
        if residuals:
            raise ResidualIdentifierError(f"{len(residuals)} candidate identifiers remain after anonymization")
    return card_set
