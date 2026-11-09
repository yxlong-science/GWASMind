"""Correct explicit report references to a saved ranking, never the ranking itself."""
from __future__ import annotations

from collections.abc import Mapping
import re

from .audit import sha256_json
from .ranking_order import _ORDINALS, _RELATION, _OrderChecker, tied_order_sentence_edits


_SENTENCE = re.compile(r".+?(?:(?<=[.!?])(?=\s)|(?=\n)|$)", re.DOTALL)
_NONCURRENT = re.compile(
    r"\b(?:if|whether|might|may|could|would|suppose|hypothes\w*|hypothetical|"
    r"previous\w*|prior|original|historical\w*|formerly|published|publication|"
    r"reported|study|literature|external|not|no|never|cannot|can't|uncertain|"
    r"unresolved|unconfirmed|incorrect|false|claim\w*|among|within)\b", re.IGNORECASE)
_OPPOSITE = {"ahead of": "after", "higher than": "lower than", "before": "after", "above": "below",
             "after": "before", "behind": "ahead of", "lower than": "higher than", "below": "above"}


def _position_label(original, rank):
    if original.casefold() in _ORDINALS:
        value = _ORDINALS[rank - 1] if rank <= len(_ORDINALS) else str(rank)
    else:
        suffix = "th" if 10 <= rank % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
        value = re.sub(r"\d+(?:st|nd|rd|th)?", str(rank) + (suffix if re.search(r"\d(?:st|nd|rd|th)$", original, re.I) else ""), original, flags=re.I)
    return value.upper() if original.isupper() else value.capitalize() if original.istitle() else value


def saved_rank_reference_corrections(text, context):
    """Replace only explicit current rank numbers or pairwise ordering words.

    Biological support, hypotheses, historical rankings, subsets and unbound
    subjects are not factual references to this list. No evidence is scored and
    no candidate is inserted, removed, reordered or rewritten by this function.
    """
    from .staged_reasoning import _reported_rank_aliases

    saved = context.result.get("ranked_genes")
    if not isinstance(saved, (list, tuple)) or not saved or any(not isinstance(row, Mapping) for row in saved):
        return []
    ranks, aliases = _reported_rank_aliases(context)
    if len(ranks) != len(saved):
        return []
    decision = context.result.get("decision")
    if isinstance(decision, Mapping) and decision.get("status") == "ranking_unavailable":
        return []
    checker = _OrderChecker(ranks, aliases)
    source = {"source_type": "validated_ranking", "source_id": "ranked_genes",
              "source_sha256": sha256_json(saved),
              "source_ranks": [{"gene_id": gene, "rank": rank} for gene, rank in ranks.items()]}
    edits = [{**edit, "metric": "ranking_order", **source}
             for edit in tied_order_sentence_edits(text, ranks=ranks, aliases=aliases)]
    for sentence in _SENTENCE.finditer(text):
        if any(edit['start'] <= sentence.start() < edit['end'] for edit in edits):
            continue
        original = sentence.group()
        # Strip ordinary emphasis only for matching. Map edits back to exact
        # source characters so JSON/prose formatting outside each token survives.
        offsets = [index for index, char in enumerate(original) if char not in "*`"]
        plain = "".join(original[index] for index in offsets)
        if _NONCURRENT.search(plain) or re.search(r'["“”‘’]|(?<!\w)\'|\'(?!\w)', plain):
            continue
        for match in checker.explicit.finditer(plain):
            gene = checker.aliases[match["subject"].casefold()]
            rank = ranks[gene]
            tail = plain[match.end():]
            # An unbound comparison peer or an extra scope cannot be resolved
            # by silently assuming this report's complete candidate list.
            if re.match(r"\s+(?:" + _RELATION + r")\b", tail, re.I):
                continue
            if re.match(r"\s+(?:for|in|under|across)\b", tail, re.I) and not re.match(
                r"\s+in\s+(?:the\s+)?(?:current|returned|saved|validated|model)\s+(?:ranking|list|order)\b", tail, re.I):
                continue

            def add(group, replacement, metric, **details):
                start, end = offsets[match.start(group)], offsets[match.end(group) - 1] + 1
                # Do not cut across an emphasis delimiter inside a phrase.
                if any(char in original[start:end] for char in "*`"):
                    return
                if original[start:end] == replacement:
                    return
                edits.append({"start": sentence.start() + start, "end": sentence.start() + end,
                    "replacement": replacement, "metric": metric, "gene_id": gene,
                    "expected_rank": rank, "reason": "report_reference_conflicts_with_saved_ranking",
                    **source, **details})

            stated = match["position"]
            if stated:
                number = _ORDINALS.index(stated.casefold()) + 1 if stated.casefold() in _ORDINALS else int(re.search(r"\d+", stated)[0])
                if number != rank:
                    add("position", _position_label(stated, rank), "ranking_position")
            relation = match["relation"] or match["relative"]
            peer = match["other"] or match["peer"]
            if not relation or not peer:
                continue
            peer_id = checker.aliases[peer.casefold()]
            peer_rank = ranks[peer_id]
            if gene == peer_id or rank == peer_rank:
                continue
            key = " ".join(relation.casefold().split())
            earlier = key in {"ahead of", "higher than", "before", "above"}
            if earlier != (rank < peer_rank):
                replacement = _OPPOSITE[key]
                if relation.isupper():
                    replacement = replacement.upper()
                elif relation.istitle():
                    replacement = replacement.capitalize()
                add("relation" if match["relation"] else "relative", replacement, "ranking_order",
                    peer_gene_id=peer_id, peer_expected_rank=peer_rank)
    return edits
