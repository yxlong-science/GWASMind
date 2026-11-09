from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..models import DataMode, EvidenceKind, EvidenceRecord, LocusQuery


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    mode: DataMode
    release: str
    evidence_kinds: frozenset[EvidenceKind]
    snapshot_hash: str = "unknown"
    schema_hash: str = "unknown"
    datasets: frozenset[str] = frozenset()
    modules: frozenset[str] = frozenset()
    supports_l2g: bool = False


class OpenTargetsProvider(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    @abstractmethod
    async def fetch(
        self, query: LocusQuery, kinds: set[EvidenceKind]
    ) -> list[EvidenceRecord]:
        raise NotImplementedError


def ensure_supported(provider: OpenTargetsProvider, kinds: set[EvidenceKind]) -> None:
    missing = kinds.difference(provider.capabilities.evidence_kinds)
    if missing:
        names = ", ".join(sorted(kind.value for kind in missing))
        raise ValueError(f"provider does not support: {names}")
