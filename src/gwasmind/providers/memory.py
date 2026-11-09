from __future__ import annotations

from copy import deepcopy

from ..models import DataMode, EvidenceKind, EvidenceRecord, LocusQuery
from ..qtl import qtl_study_type_from_semantic_payload
from .base import OpenTargetsProvider, ProviderCapabilities, ensure_supported


class MemoryOpenTargetsProvider(OpenTargetsProvider):
    """Deterministic provider used for unit tests and recorded fixtures."""

    def __init__(
        self,
        records: list[EvidenceRecord],
        *,
        release: str = "fixture",
        mode: DataMode = DataMode.FROZEN_SAFE,
        snapshot_hash: str = "fixture",
    ):
        self._records = deepcopy(list(records))
        self._capabilities = ProviderCapabilities(
            mode=mode,
            release=release,
            evidence_kinds=frozenset(record.kind for record in records),
            snapshot_hash=snapshot_hash,
            supports_l2g=any(record.kind is EvidenceKind.L2G for record in records),
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def fetch(
        self, query: LocusQuery, kinds: set[EvidenceKind]
    ) -> list[EvidenceRecord]:
        query.validate()
        ensure_supported(self, kinds)
        variant_ids = {variant.variant_id for variant in query.variants}
        genes = set(query.candidate_genes)
        return deepcopy([
            record
            for record in self._records
            if record.kind in kinds
            and (
                record.variant_id is None
                or record.variant_id in variant_ids
            )
            and (not genes or record.gene_id is None or record.gene_id in genes)
            and (
                record.kind is not EvidenceKind.QTL
                or not query.qtl_types
                or qtl_study_type_from_semantic_payload(record.payload) in query.qtl_types
            )
        ])
