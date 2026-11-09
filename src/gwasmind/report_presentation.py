"""Presentation vocabulary for explicit language-model report metadata only."""
from __future__ import annotations

import html
import re

_LLM_PHRASES = {
    "model ranking": "LLM ranking",
    "validated model items": "Validated LLM items",
    "model-based order": "LLM-based order",
    "model-assisted review": "LLM-assisted review",
    "the model's complete interpretation": "The LLM's complete interpretation",
    "the model's interpretation": "The LLM's interpretation",
    "a completed model summary": "A completed LLM Summary",
    "the model returned a report structure": "The LLM returned a report structure",
    "the model's report could not be reconciled": "The LLM's report could not be reconciled",
}
_PHRASE = re.compile(r"\b(?:" + "|".join(re.escape(key) for key in _LLM_PHRASES) + r")\b", re.IGNORECASE)


def llm_presentation_text(text: str) -> str:
    """Do not rename statistical models, protocol values, source quotes or code.

    Call this after validation, and before appending the untouched bibliography.
    The input string and raw draft/audit records remain unchanged.
    """
    lines, fenced = [], False
    for line in text.splitlines(keepends=True):
        if re.match(r"\s*(?:```|~~~)", line):
            fenced = not fenced
            lines.append(line)
            continue
        if fenced or re.match(r"\s*>", line):
            lines.append(line)
            continue
        pieces = re.split(r"(`[^`\n]*`)", line)
        lines.append("".join(piece if index % 2 else _PHRASE.sub(
            lambda match: _LLM_PHRASES[match[0].casefold()], piece) for index, piece in enumerate(pieces)))
    return "".join(lines)


_INLINE_BIBLIOGRAPHY_TAG = re.compile(r"</?(?:i|em|b|strong|italic)\b[^>]*>", re.IGNORECASE)
_SCRIPT_BIBLIOGRAPHY_TAG = re.compile(r"<(sup|sub)\b[^>]*>(.*?)</\1\s*>", re.IGNORECASE | re.DOTALL)
_SUPERSCRIPT = str.maketrans("0123456789+-=()in", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿ")
_SUBSCRIPT = str.maketrans("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")


def bibliography_presentation_text(text: str) -> str:
    """Display publisher inline markup as text; never mutate citation metadata.

    This runs only on the final bibliography after report validation. Citation
    numbers, membership, URLs, provider fields and writer inputs stay unchanged.
    Known superscript/subscript notation remains scientific notation, while
    unknown markup stays escaped rather than becoming executable HTML.
    """
    pieces = re.split(r"(`[^`\n]*`|\]\([^\n)]*\))", text)
    for index in range(0, len(pieces), 2):
        value = pieces[index]
        for _ in range(2):
            decoded = html.unescape(value)
            if decoded == value:
                break
            value = decoded
        value = _INLINE_BIBLIOGRAPHY_TAG.sub("", value)

        def script(match: re.Match[str]) -> str:
            content = match[2]
            table = _SUPERSCRIPT if match[1].casefold() == "sup" else _SUBSCRIPT
            if content and all(ord(char) in table for char in content):
                return content.translate(table)
            return ("^(" if match[1].casefold() == "sup" else "_(") + content + ")"

        value = _SCRIPT_BIBLIOGRAPHY_TAG.sub(script, value)
        pieces[index] = value.replace("<", "&lt;").replace(">", "&gt;").replace("\xa0", " ")
    return "".join(pieces)
