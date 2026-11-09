"""Display verified source-variant identities without ambiguous alias pairs."""
from __future__ import annotations

import re

from .report_facts import public_source_normalizations


def source_variant_name_corrections(text, context):
    """Collapse a model-written alias pair only with a validated identity proof.

    Standalone uploaded identifiers and statements describing input corrections
    remain intact. This changes no observation, P value, allele effect or rank.
    """
    edits = []
    for proof in public_source_normalizations(context.result):
        original, canonical = proof["original_id"], proof["canonical_id"]
        for first, second in ((original, canonical), (canonical, original)):
            pattern = re.compile(
                r"(?<![\w])`?" + re.escape(first) + r"`?\s*\(\s*alias\s+`?"
                + re.escape(second) + r"`?\s*\)", re.I,
            )
            for match in pattern.finditer(text):
                edits.append({"start": match.start(), "end": match.end(),
                              "replacement": "`" + canonical + "`",
                              "metric": "variant_id", "source_type": "validated_source_identity",
                              "reason": "verified_reference_allele_order_normalized",
                              "original_id": original, "canonical_id": canonical})
    return edits
