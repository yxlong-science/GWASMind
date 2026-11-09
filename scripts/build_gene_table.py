"""Build the packaged GRCh38 gene-coordinate table used for the V2G gene window.

Reads an Open Targets `target` Parquet dataset and writes a compact gzip TSV:
gene_id, symbol, chromosome, start, end, strand, biotype. The first line records
the source dataset and release so the window is reproducible.

Usage: python scripts/build_gene_table.py <target_parquet_dir> <release>
"""
from __future__ import annotations

import gzip
import hashlib
import sys
from pathlib import Path

import duckdb

OUTPUT = Path(__file__).resolve().parents[1] / "src" / "gwasmind" / "data" / "gene_coordinates_grch38.tsv.gz"
CHROMOSOMES = {*(str(value) for value in range(1, 23)), "X", "Y", "MT"}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    source, release = Path(sys.argv[1]), sys.argv[2]
    rows = duckdb.connect().execute(
        "select id, approvedSymbol, genomicLocation.chromosome, genomicLocation.start, genomicLocation.end, "
        "genomicLocation.strand, biotype from read_parquet(?) where genomicLocation is not null order by 3, 4, 1",
        [str(source / "**" / "*.parquet")],
    ).fetchall()
    kept = [row for row in rows if str(row[2]) in CHROMOSOMES and row[3] and row[4]]
    body = "".join("\t".join("" if value is None else str(value) for value in row) + "\n" for row in kept)
    digest = hashlib.sha256(body.encode()).hexdigest()
    header = (f"# source=Open Targets Platform target dataset; release={release}; genome_build=GRCh38; "
              f"genes={len(kept)}; content_sha256={digest}\n"
              "gene_id\tsymbol\tchromosome\tstart\tend\tstrand\tbiotype\n")
    with gzip.open(OUTPUT, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(header + body)
    print(f"{OUTPUT} genes={len(kept)} sha256={digest}")


if __name__ == "__main__":
    main()
