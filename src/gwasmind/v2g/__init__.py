"""Gene-centric variant-to-gene (V2G) evidence retrieval and deterministic scoring.

Every gene in the lead-centered window is retained. Evidence tiers summarize
the strongest locus-to-gene link but never remove a gene from the ranking.
"""
from __future__ import annotations

from .models import (
    ChannelStatus,
    GeneWindow,
    LocusVariant,
    LocusVariantSet,
    V2GRequest,
    V2GResult,
    WindowGene,
)

__all__ = [
    "ChannelStatus",
    "GeneWindow",
    "LocusVariant",
    "LocusVariantSet",
    "V2GRequest",
    "V2GResult",
    "WindowGene",
]
