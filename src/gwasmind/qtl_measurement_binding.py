"""Read variant ownership from program-rendered credible-set lead QTL rows."""
from __future__ import annotations

import re


def credible_set_lead_binding(markdown: str) -> dict[str, str] | None:
    """Bind one displayed molecular row; overlap witnesses do not own its values.

    Call only with an immutable display fact, never with model-written prose.
    Multirow or ambiguous facts require row-level handling by the caller.
    """
    plain = re.sub(r"[*`]", "", markdown).strip()
    if "\n" in plain or not re.match(r"^[-]?\s*(?:sc[-]?)?(?:e|s|p|tu|mol)QTL\s*\(", plain, re.I):
        return None
    pattern = (r"\(displayed statistics:\s*credible-set lead variant\s+"
        r"(?P<variant>(?:chr)?(?:[0-9]+|X|Y|MT)[_:][0-9]+[_:][ACGTN]+[_:][ACGTN]+)"
        r"\s*;\s*study\s+(?P<study>[^;()\s]+)(?=\s*[;)])")
    matches = list(re.finditer(pattern, plain, re.I))
    if len(matches) != 1 or len(re.findall(r"credible-set lead variant", plain, re.I)) != 1:
        return None
    return {"variant_id": re.sub(r"^chr", "", matches[0]["variant"], flags=re.I).upper().replace(":", "_"),
            "study_id": matches[0]["study"], "measurement_scope": "credible_set_lead"}
