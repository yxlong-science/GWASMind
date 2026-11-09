"""Recognize inline evidence references without treating ordinary labels as IDs."""
from __future__ import annotations

import re
from collections.abc import Set as AbstractSet

_INLINE_TOKENS = re.compile(r"[\w][\w.:/@+\-]*")
_EVIDENCE_LABEL = r"(?<![A-Za-z0-9])(?:\*\*|__)?(?:evidence|records?|citations?|\u8bc1\u636e|\u5f15\u7528)"
_LABELED_EVIDENCE = re.compile(
    _EVIDENCE_LABEL +
    r"(?:\*\*|__)?(?:\s+(?:ids?|identifiers?)\s*[:：=]?|\s*[:：=])"
    r"(?:\*\*|__)?\s*(?:\[([^\]\n]+)\]|`([^`\n]+)`)", re.IGNORECASE,
)
_BARE_LABELED_EVIDENCE = re.compile(
    _EVIDENCE_LABEL + r"(?:\*\*|__)?"
    r"\s*(?:\[([^\]\n]+)\]|`([^`\n]+)`)", re.IGNORECASE,
)


def inline_evidence_citations(text: str, known_ids: AbstractSet[str]) -> set[str]:
    identifiers = set()
    for token in _INLINE_TOKENS.findall(text):
        if token in known_ids:
            identifiers.add(token)
        elif token.rstrip(".:") in known_ids:
            identifiers.add(token.rstrip(".:"))
    namespaces = {identifier.split(":", 1)[0] for identifier in known_ids if ":" in identifier}
    namespaces = {value for value in namespaces if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value)}
    namespaces.update({"semantic", "semantic-upstream"})
    namespace_pattern = re.compile(
        r"(?<![\w-])(?:" + "|".join(re.escape(value) for value in sorted(namespaces)) + r"):[\w.:/-]+",
        re.IGNORECASE,
    )
    identifiers.update(value.rstrip(".:") for value in namespace_pattern.findall(text))
    for match in _LABELED_EVIDENCE.finditer(text):
        identifiers.update(token.rstrip(".:") for token in _INLINE_TOKENS.findall(
            match.group(1) or match.group(2)
        ))
    for match in _BARE_LABELED_EVIDENCE.finditer(text):
        # Without an explicit separator, require identifier-shaped content;
        # prose such as "evidence [not assessed]" remains ordinary prose.
        label = (match.group(1) or match.group(2)).strip()
        for part in re.split(r"[,;]\s*", label):
            if _INLINE_TOKENS.fullmatch(part) and (
                any(marker in part for marker in (":", "_", "-", "/", "@"))
                or (re.search(r"[A-Za-z]", part) and re.search(r"\d", part))
            ):
                identifiers.add(part.rstrip(".:"))
    return identifiers
