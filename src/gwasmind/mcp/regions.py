from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_CANONICAL_VARIANT = re.compile(
    r"^(?:chr)?(?P<chromosome>[0-9]+|X|Y|MT)(?:_|:)(?P<position>[0-9]+)"
    r"(?:_|:)[ACGTN]+(?:_|:)[ACGTN]+$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GRCh38Region:
    chromosome: str
    start: int
    end: int
    genome_build: str = "GRCh38"

    def __post_init__(self) -> None:
        if not self.chromosome.strip():
            raise ValueError("chromosome is required")
        if self.start < 1 or self.end < self.start:
            raise ValueError("region coordinates are invalid")
        if self.genome_build != "GRCh38":
            raise ValueError("only GRCh38 regions are supported")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "chromosome": self.chromosome,
            "start": self.start,
            "end": self.end,
            "genome_build": self.genome_build,
        }


def grch38_region_from_variant_ids(
    variant_ids: Iterable[str],
) -> GRCh38Region | None:
    """Derive one exact same-chromosome GRCh38 interval from canonical variants."""

    chromosome: str | None = None
    positions: list[int] = []
    for value in variant_ids:
        match = _CANONICAL_VARIANT.fullmatch(str(value).strip())
        if match is None:
            return None
        current = match.group("chromosome").upper()
        if chromosome is None:
            chromosome = current
        elif current != chromosome:
            return None
        position = int(match.group("position"))
        if position < 1:
            return None
        positions.append(position)
    if chromosome is None or not positions:
        return None
    return GRCh38Region(chromosome, min(positions), max(positions))
