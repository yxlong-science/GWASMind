"""Model-only ranking of V2G evidence cards with a strict output contract.

The model returns card identifiers; the application maps them to genes. An
invalid ranking gets one bounded correction. If it is still invalid it is
recorded as invalid: the application never fills, reorders or replaces it.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..prompting import format_prompt, load_prompt
from .cards import CardSet

Complete = Callable[[list[dict[str, str]], int], str]
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass(slots=True)
class RankingParse:
    order: list[str]
    rationales: dict[str, dict[str, Any]]
    summary: str
    ranking_errors: list[str]
    explanation_errors: list[str]

    @property
    def ranking_valid(self) -> bool:
        return not self.ranking_errors


@dataclass(slots=True)
class RankingOutcome:
    status: str  # valid | repaired | invalid | unavailable
    gene_order: list[str]
    rationales: dict[str, dict[str, Any]]
    summary: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    explanation_errors: list[str] = field(default_factory=list)

    @property
    def ranking_valid(self) -> bool:
        return self.status in {"valid", "repaired"}

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "gene_order": list(self.gene_order), "rationales": dict(self.rationales),
                "summary": self.summary, "attempts": list(self.attempts),
                "explanation_errors": list(self.explanation_errors)}


def build_ranking_messages(card_set: CardSet, *, trait: str, locus_summary: Mapping[str, Any],
                           reasoning: bool = True) -> list[dict[str, str]]:
    system = load_prompt("v2g_ranking_system")
    if reasoning:
        system += "\n\n" + load_prompt("v2g_ranking_reasoning")
    if card_set.anonymized:
        system += "\n\n" + load_prompt("v2g_anonymized_note")
    payload = json.dumps({"trait": trait, "locus": dict(locus_summary), "candidates": card_set.cards},
                         ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system", "content": system},
            {"role": "user", "content": format_prompt("v2g_ranking_user", payload=payload)}]


def parse_ranking(content: str, valid_ids: Sequence[str]) -> RankingParse:
    valid = list(valid_ids)
    known = set(valid)
    try:
        value = json.loads(_FENCE.sub("", str(content or "")))
    except json.JSONDecodeError:
        return RankingParse([], {}, "", ["The answer is not a JSON object."], [])
    if not isinstance(value, Mapping):
        return RankingParse([], {}, "", ["The answer is not a JSON object."], [])
    ranking = value.get("ranking")
    errors: list[str] = []
    order: list[str] = []
    if not isinstance(ranking, list) or not all(isinstance(item, str) for item in ranking):
        errors.append("ranking must be an array of candidate id strings.")
    else:
        order = [item.strip() for item in ranking]
        duplicates = sorted({item for item in order if order.count(item) > 1})
        unknown = sorted({item for item in order if item not in known})
        missing = [item for item in valid if item not in set(order)]
        if duplicates:
            errors.append(f"Duplicate ids: {', '.join(duplicates)}.")
        if unknown:
            errors.append(f"Unknown ids: {', '.join(unknown)}.")
        if missing:
            errors.append(f"Missing ids: {', '.join(missing)}.")
    rationales: dict[str, dict[str, Any]] = {}
    explanation_errors: list[str] = []
    raw_rationales = value.get("rationales")
    if raw_rationales is not None and not isinstance(raw_rationales, list):
        explanation_errors.append("rationales must be an array.")
        raw_rationales = []
    for item in raw_rationales or []:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            explanation_errors.append("A rationale has no candidate id.")
            continue
        identifier = item["id"].strip()
        text = str(item.get("text") or "").strip()
        if identifier not in known:
            explanation_errors.append(f"Rationale for unknown id {identifier}.")
            continue
        if not text:
            explanation_errors.append(f"Rationale for {identifier} is empty.")
            continue
        categories = item.get("categories")
        rationales[identifier] = {"text": text[:2000], "categories": [str(value)[:40] for value in categories][:12]
                                  if isinstance(categories, list) else []}
    summary = str(value.get("summary") or "").strip()[:3000]
    return RankingParse(order, rationales, summary, errors, explanation_errors)


def rank_with_model(card_set: CardSet, complete: Complete, *, trait: str, locus_summary: Mapping[str, Any],
                    reasoning: bool = True) -> RankingOutcome:
    """Call the model once, and once more with the validation errors if the ranking is invalid."""

    valid_ids = [card["id"] for card in card_set.cards]
    gene_for = card_set.gene_for_card()
    messages = build_ranking_messages(card_set, trait=trait, locus_summary=locus_summary, reasoning=reasoning)
    attempts: list[dict[str, Any]] = []
    parsed = RankingParse([], {}, "", ["No answer was generated."], [])
    for attempt in (1, 2):
        content = complete(messages, attempt)
        parsed = parse_ranking(content, valid_ids)
        attempts.append({"attempt": attempt, "raw_output": content, "ranking_errors": list(parsed.ranking_errors),
                         "explanation_errors": list(parsed.explanation_errors)})
        if parsed.ranking_valid:
            break
        messages = [*messages, {"role": "assistant", "content": str(content or "")},
                    {"role": "user", "content": format_prompt(
                        "v2g_ranking_repair_user", errors=" ".join(parsed.ranking_errors),
                        ids=", ".join(valid_ids))}]
    if not parsed.ranking_valid:
        return RankingOutcome("invalid", [], {}, "", attempts, parsed.explanation_errors)
    return RankingOutcome(
        status="valid" if len(attempts) == 1 else "repaired",
        gene_order=[gene_for[identifier] for identifier in parsed.order],
        rationales={gene_for[identifier]: value for identifier, value in parsed.rationales.items()},
        summary=parsed.summary,
        attempts=attempts,
        explanation_errors=parsed.explanation_errors,
    )
