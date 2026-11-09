"""GWASMind scientific core."""

# Defined before the submodule imports so they can read it during package import.
__version__ = "1.0.0"

from .models import (  # noqa: E402
    AgentResult,
    DecisionStatus,
    EvidenceRecord,
    LocusQuery,
    RankedGene,
)
from .agent import GWASMindAgent  # noqa: E402

__all__ = [
    "AgentResult",
    "DecisionStatus",
    "EvidenceRecord",
    "LocusQuery",
    "GWASMindAgent",
    "RankedGene",
]
