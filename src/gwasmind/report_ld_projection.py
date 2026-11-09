"""Compact final-report LD provenance selected from the validated full plan."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .ld_support import LDReportScope
from .scientific_facts import _scientific_table

_VARIANT = re.compile(r"\b(?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+\b", re.I)


def report_ld_projection(scope: LDReportScope, displayed_facts: Iterable[str]) -> dict[str, Any]:
    """Keep every source edge for each proxy in authoritative displayed facts.

    The scope has already validated identities, alleles, population and r².
    Selection is by report relevance, never by array order, link count or r².
    Full fact text includes the expandable source/proxy attribution trace.
    """
    named = {
        re.sub(r"^chr", "", match, flags=re.I).replace(":", "_").upper()
        for text in displayed_facts for match in _VARIANT.findall(text)
    }
    proxies = set(scope.proxy_ids).intersection(named)
    links = [dict(link) for link in scope.links if link["proxy_variant_id"] in proxies]
    # Optional provider/version provenance describes the recorded LD service;
    # it is retained independently of which proxy edges are displayed.
    summary = scope.prompt_summary(link_limit=0)
    summary.pop("ld_links")
    summary.pop("omitted_link_count")
    summary.pop("total_link_count")
    summary.update({
        "ld_relations": _scientific_table(links),
        "relation_selection": "all_verified_edges_for_reported_proxies",
        "relation_selection_note": (
            "All verified source-proxy edges for proxies named in the displayed facts are included. "
            "Other recorded edges remain in the full ledger. This report-relevance selection is "
            "not a statement of complete regional retrieval and does not change the recorded "
            "expansion completeness."
        ),
    })
    return summary
