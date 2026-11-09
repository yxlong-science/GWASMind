"""Reconcile explicit ranking statements after combining recovered candidates.

This checks positions in the returned list, not comparative scientific support.
Only names bound to typed evidence (or the ranked gene IDs) are interpreted.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .models import EvidenceRecord, RankedGene
from .ranking_claims import _gene_aliases

_ORDINALS = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
    "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
)
_POSITION = r"(?:" + "|".join(_ORDINALS) + r"|(?:number\s+|#)?\d{1,3}(?:st|nd|rd|th)?)"
_RELATION = r"(?:ahead\s+of|higher\s+than|lower\s+than|before|after|above|below|behind)"
_VERB = r"(?:(?:is\s+)?(?:currently\s+)?(?:ranked|placed|prioriti[sz]ed)|ranks)"
_RELATIVE_VERB = r"(?:(?:is\s+)?(?:currently\s+)?ranked|ranks)"
_QUALIFIED = re.compile(
    r"\b(?:whether|if|might|may|could|would|previously|formerly|historically)\b[^;.!?]*$|"
    r"\b(?:original|previous|prior)\s+(?:ranking|ordering|list)\b[^;.!?]*$", re.IGNORECASE,
)
_OLD_SCOPE = re.compile(
    r"\s+in\s+(?:the\s+)?(?:original|previous|prior)\s+(?:ranking|ordering|list)\b", re.IGNORECASE,
)
_DENIED = re.compile(
    r"\b(?:cannot|can't|unable\s+to|do\s+not|does\s+not)\s+"
    r"(?:conclude|say|establish|determine|assert|claim)\s+(?:that\s+)?$|"
    r"\b(?:not\s+established|no\s+evidence)\s+that\s+$", re.IGNORECASE,
)
_HISTORICAL_CONTEXT = re.compile(
    r'\b(?:in|according\s+to)\s+(?:the\s+)?(?:literature|'
    r'(?:published|reported|external|historical)\s+(?:study|ranking|analysis|list))\s*,?\s*$', re.I)


class _OrderChecker:
    def __init__(self, ranks: Mapping[str, int], aliases: Mapping[str, str] | Sequence[tuple[str, str]]):
        self.positions = dict(ranks)
        owners: dict[str, set[str]] = {}
        for alias, gene in (aliases.items() if isinstance(aliases, Mapping) else aliases):
            if isinstance(alias, str) and alias and isinstance(gene, str) and gene:
                owners.setdefault(alias.casefold(), set()).add(gene)
        self.aliases = {alias: next(iter(genes)) for alias, genes in owners.items() if len(genes) == 1}
        self.aliases.update({gene.casefold(): gene for gene in self.positions})
        names = "|".join(re.escape(name) for name in sorted(self.aliases, key=len, reverse=True))
        name = r"(?<!\w)(?:" + names + r")(?!\w)"
        predicate = (
            r"(?:" + _VERB + r"\s+(?P<position>" + _POSITION + r")(?!\w)"
            r"(?:\s*,?\s+(?P<relation>" + _RELATION + r")\s+(?P<other>" + name + r"))?"
            r"|" + _RELATIVE_VERB + r"\s+(?P<relative>" + _RELATION + r")\s+(?P<peer>" + name + r"))"
        )
        self.explicit = re.compile(r"(?P<subject>" + name + r")\s+" + predicate, re.IGNORECASE)
        self.implicit = re.compile(
            r"^\s*(?:[-*]\s+)?(?:(?:it|this\s+(?:candidate|gene))\s+)?" + predicate, re.IGNORECASE,
        )

    def issues(self, text: str, *, owner: str | None = None) -> tuple[str, ...]:
        # Formatting does not change the identity or order being asserted.
        plain = re.sub(r"[*`]", "", text)
        issues = []
        matches = [(match, self.aliases[match["subject"].casefold()])
                   for match in self.explicit.finditer(plain)]
        implicit = self.implicit.match(plain)
        if implicit and owner in self.positions:
            matches.append((implicit, owner))
        for match, gene in matches:
            prefix = re.split(r"[;,]|\b(?:but|however|yet|whereas)\b", plain[:match.start()], flags=re.IGNORECASE)[-1]
            if (_QUALIFIED.search(prefix) or _DENIED.search(prefix)
                    or _HISTORICAL_CONTEXT.search(plain[:match.start()])
                    or _OLD_SCOPE.match(plain[match.end():])):
                continue
            position = self.positions.get(gene)
            if position is None:
                issues.append(f"ranking statement names an unranked candidate: {gene}")
                continue
            stated = match["position"]
            if stated:
                ordinal = stated.casefold()
                number = _ORDINALS.index(ordinal) + 1 if ordinal in _ORDINALS else int(re.search(r"\d+", ordinal)[0])
                if number != position:
                    issues.append(f"{gene} is ranked {position}, not {number}, in the validated list")
            relation = match["relation"] or match["relative"]
            other = match["other"] or match["peer"]
            if not relation or not other:
                continue
            peer = self.aliases[other.casefold()]
            peer_position = self.positions.get(peer)
            if peer_position is not None and peer_position == position and peer != gene:
                issues.append(f"{gene} {relation} {peer} conflicts with the validated tie")
                continue
            earlier = " ".join(relation.casefold().split()) in {"before", "above", "ahead of", "higher than"}
            valid = peer_position is not None and (
                position < peer_position if earlier else position > peer_position
            )
            if not valid:
                issues.append(f"{gene} {relation} {peer} conflicts with the validated candidate order")
        return tuple(dict.fromkeys(issues))


def _competition_ranks(values) -> bool:
    """Consecutive positions, or shared ranks such as 1, 1, 1, 4 for tied candidates."""
    ordered = sorted(values)
    return bool(ordered) and all(rank == ordered.index(rank) + 1 for rank in ordered)


def explicit_ranking_order_issues(
    text: str, *, ranks: Mapping[str, int],
    aliases: Mapping[str, str] | Sequence[tuple[str, str]] = (), owner: str | None = None,
) -> tuple[str, ...]:
    """Check explicit present ranking prose against a complete, validated order.

    The function has no query/model state. A sequence of alias pairs preserves
    conflicting symbol ownership; ambiguous symbols are never assigned by row
    order. Without valid consecutive ranks there is no order to compare against.
    """
    if not ranks or any(not isinstance(gene, str) or not gene or type(rank) is not int or rank < 1
                        for gene, rank in ranks.items()):
        return ()
    if not _competition_ranks(ranks.values()):
        return ()
    checker = _OrderChecker(ranks, aliases)
    return tuple(dict.fromkeys(issue for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
                              for issue in checker.issues(sentence, owner=owner)))


def tied_order_sentence_edits(text, *, ranks, aliases=()):
    """Remove explicit current ordering assertions that contradict a saved tie.

    A complete sentence is the minimum safe unit: deleting only a relation can
    turn its remaining scientific predicate into a different assertion.
    """
    checker = _OrderChecker(ranks, aliases)
    edits = []
    for sentence in re.finditer(r'.+?(?:(?<=[.!?])(?=\s)|(?=\n)|$)', text, re.DOTALL):
        issues = checker.issues(sentence.group())
        if any(issue.endswith('conflicts with the validated tie') for issue in issues):
            edits.append({'start': sentence.start(), 'end': sentence.end(), 'replacement': '',
                          'removed_text': sentence.group(), 'reason': 'strict_order_conflicts_with_saved_tie'})
    return edits


def unavailable_ranking_issues(text: str, *, aliases: Mapping[str, str]) -> tuple[str, ...]:
    """Reject an asserted present ranking after an explicitly failed model run."""
    if not aliases:
        return ()
    checker = _OrderChecker({}, aliases)
    names = "|".join(re.escape(name) for name in sorted(checker.aliases, key=len, reverse=True))
    name = r"(?<!\w)(?:" + names + r")(?!\w)"
    top = re.compile(
        name + r"\s+(?:is|remains)\s+(?:the\s+)?(?:top|leading|first[- ]ranked|highest[- ]ranked)"
        r"\s+(?:candidate|gene)\b|\b(?:top|leading|first[- ]ranked)\s+(?:candidate|gene)\s*:\s*" + name,
        re.IGNORECASE,
    )
    issues = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"[*`]", "", text)):
        issues.extend(checker.issues(sentence))
        for match in top.finditer(sentence):
            prefix = re.split(r"[;,]|\b(?:but|however|yet|whereas)\b", sentence[:match.start()], flags=re.IGNORECASE)[-1]
            if _QUALIFIED.search(prefix) or _DENIED.search(prefix) or _OLD_SCOPE.match(sentence[match.end():]):
                continue
            issues.append("a top candidate is asserted although model ranking is unavailable")
    return tuple(dict.fromkeys(issues))


def reconcile_recovered_ranking_order(
    ranked: Sequence[RankedGene], evidence: Sequence[EvidenceRecord],
) -> tuple[tuple[RankedGene, ...], int]:
    """Remove only sentences with an explicit, now-false ranking statement.

    Other sentences and still-valid comparisons remain byte-for-byte intact.
    Evidence, scientific scores, and rank positions are never changed. An empty
    rationale receives only its actual position, without a new scientific claim.
    """
    from .models import shared_ranks

    shared = shared_ranks(ranked)
    checker = _OrderChecker({row.gene_id: shared.get(row.gene_id, (row.rank, False))[0] for row in ranked},
                            _gene_aliases(evidence))
    removed = 0

    def clean(text: str, owner: str) -> str:
        nonlocal removed
        changed = False
        pieces = re.split(r"((?<=[.!?])\s+|\n+)", text)
        for index in range(0, len(pieces), 2):
            if checker.issues(pieces[index], owner=owner):
                pieces[index] = ""
                if index + 1 < len(pieces):
                    pieces[index + 1] = ""
                removed += 1
                changed = True
        return "".join(pieces).strip() if changed else text

    repaired = []
    for row in ranked:
        rationale = clean(row.rationale, row.gene_id)
        if not rationale and row.rationale:
            rationale = f"{row.gene_id} is ranked {row.rank} in the recovered candidate list."
        limitations = tuple(value for text in row.limitations for value in (clean(text, row.gene_id),) if value or not text)
        repaired.append(replace(row, rationale=rationale, limitations=limitations))
    return tuple(repaired), removed
