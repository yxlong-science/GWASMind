from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

GENOME_WIDE_THRESHOLD = 5e-8
PRIORITY_THRESHOLD = 2.5
BACKGROUND_BIN_SIZE = 0.25
BACKGROUND_POINTS_PER_BIN = 10_000
MAX_PROCESSED_POINTS = 300_000
DATASETS = {
    "asthma": "asthma.json.gz",
    "mdd": "mdd.json.gz",
}


def build_asset(source: Path, destination: Path) -> dict[str, Any]:
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    rows = value.get("data")
    metadata = value.get("metadata")
    if not isinstance(rows, list) or not isinstance(metadata, dict):
        raise ValueError(f"invalid processed demo asset: {source}")

    selected: list[dict[str, Any]] = []
    genome_wide_count = 0
    priority_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            p_value = float(row["pvalue"])
            neg_log_p = float(row["neg_log_p"])
        except (KeyError, TypeError, ValueError):
            continue
        selected.append(row)
        if p_value <= GENOME_WIDE_THRESHOLD:
            genome_wide_count += 1
        if neg_log_p >= PRIORITY_THRESHOLD:
            priority_count += 1
    if len(selected) > MAX_PROCESSED_POINTS:
        raise ValueError(
            f"processed source demo exceeds the {MAX_PROCESSED_POINTS:,}-point safety cap: "
            f"{source} has {len(selected):,} rows"
        )
    selected.sort(
        key=lambda row: (
            chromosome_sort_key(str(row.get("chromosome", ""))),
            int(row.get("position", 0)),
        )
    )
    compact = [
        [
            str(row.get("chromosome", "")),
            int(row.get("position", 0)),
            float(row.get("neg_log_p", 0)),
            str(row.get("snp_id", "")),
            float(row.get("pvalue", 1)),
            str(row.get("ref", "N")),
            str(row.get("alt", "N")),
        ]
        for row in selected
    ]
    output = {
        "data": compact,
        "metadata": {
            "source_file": str(metadata.get("source_file", source.name)),
            "original_count": int(metadata.get("original_count", len(rows))),
            "upstream_processed_count": len(rows),
            "visualization_count": len(compact),
            "significant_count": genome_wide_count,
            "priority_count": priority_count,
            "sampling_strategy": "gwasmind-2.0-priority-plus-stratified-background",
            "sampling_seed": 42,
            "genome_wide_threshold": GENOME_WIDE_THRESHOLD,
            "priority_threshold": PRIORITY_THRESHOLD,
            "background_bin_size": BACKGROUND_BIN_SIZE,
            "background_points_per_bin": BACKGROUND_POINTS_PER_BIN,
            "synthetic": False,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        output, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    with destination.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            compresslevel=9,
            mtime=0,
        ) as gzip_handle:
            gzip_handle.write(serialized)
    return output["metadata"]


def chromosome_sort_key(value: str) -> tuple[int, str]:
    normalized = value.removeprefix("chr").upper()
    if normalized.isdigit():
        return int(normalized), normalized
    return {"X": 23, "Y": 24, "MT": 25}.get(normalized, 99), normalized


def _locus_match_key(
    chromosome: Any, position: Any, p_value: Any,
) -> tuple[str, int, Decimal]:
    chromosome = re.sub(r"^chr", "", str(chromosome).strip(), flags=re.IGNORECASE).upper()
    try:
        coordinate = Decimal(str(position))
        probability = Decimal(str(p_value))
        if (
            chromosome not in {*(str(number) for number in range(1, 23)), "X", "Y", "MT"}
            or not coordinate.is_finite()
            or not 0 < coordinate < 2**53
            or coordinate != coordinate.to_integral_value()
            or not probability.is_finite()
            or not 0 <= probability <= 1
        ):
            raise ValueError("invalid coordinate or p-value in locus source")
    except InvalidOperation as exc:
        raise ValueError("invalid coordinate or p-value in locus source") from exc
    return chromosome, int(coordinate), probability


def build_locus_asset(
    source: Path, destination: Path, *, variant_source: Path,
) -> int:
    """Preserve locus rows while recovering alleles from the same GWAS asset.

    Coordinates alone cannot establish variant identity. Require an exact,
    unique chromosome/position/p-value match in the supplied compact GWAS
    asset, and retain the original rsID only as a separate provenance column.
    """
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if str(row.get("chr", "")) == "1"]
    if not rows:
        raise ValueError(f"locus source contains no chromosome 1 rows: {source}")
    with gzip.open(variant_source, "rt", encoding="utf-8") as handle:
        value = json.load(handle, parse_float=Decimal)
    if not isinstance(value, dict) or not isinstance(value.get("data"), list):
        raise ValueError(f"invalid compact variant source: {variant_source}")  # noqa: TRY004
    variants: dict[tuple[str, int, Decimal], list[list[Any]]] = {}
    for variant in value["data"]:
        if not isinstance(variant, list) or len(variant) < 7:
            raise ValueError(f"invalid compact variant row: {variant_source}")
        key = _locus_match_key(variant[0], variant[1], variant[4])
        variants.setdefault(key, []).append(variant)
    output_rows = []
    for row in rows:
        key = _locus_match_key(row["chr"], row["pos"], row["P"])
        matches = variants.get(key, [])
        if len(matches) != 1:
            raise ValueError(
                f"locus row {key[0]}:{key[1]} requires exactly one same-source "
                f"coordinate/p-value match; found {len(matches)}"
            )
        ref, alt = (str(allele).upper() for allele in matches[0][5:7])
        if not re.fullmatch(r"[ACGT]+", ref) or not re.fullmatch(r"[ACGT]+", alt) or ref == alt:
            raise ValueError(f"locus row {key[0]}:{key[1]} has no valid source ref/alt")
        output_rows.append(
            {
                "chromosome": row["chr"],
                "position": row["pos"],
                "p_value": row["P"],
                "variant_id": f"{key[0]}_{key[1]}_{ref}_{alt}",
                "source_rsid": row["rsID"],
                "locus_id": row["locus_no"],
                "r2": row.get("r2", ""),
            }
        )
    # Validate every match before opening an existing generated destination.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "chromosome",
                "position",
                "p_value",
                "variant_id",
                "source_rsid",
                "locus_id",
                "r2",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--locus-source", type=Path)
    args = parser.parse_args()
    for name, filename in DATASETS.items():
        metadata = build_asset(
            args.source_dir / filename,
            args.output_dir / f"{name}.real.json.gz",
        )
        print(name, json.dumps(metadata, sort_keys=True))
    if args.locus_source:
        count = build_locus_asset(
            args.locus_source,
            args.output_dir / "chr1_all_loci.csv",
            variant_source=args.output_dir / "asthma.real.json.gz",
        )
        print("auto-analysis", json.dumps({"rows": count, "loci": 8}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
