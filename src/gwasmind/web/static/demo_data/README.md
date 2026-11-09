# Demo data provenance

These are real, bounded visualization/test assets derived from the original
GWASMind demo files. They are not randomly generated GWAS results.

## Manhattan demos

- `asthma.real.json.gz` is derived from `demo_data/processed/asthma.json.gz`
  (source SHA-256 `3c2a5f8695a5d5991d532ad2761893e3b0218e55ff8d5378db6d3ec117d44179`).
  It preserves all 247,098 original processed rows: every point with
  `-log10(p) >= 2.5` plus the ten seeded, 0.25-wide background strata (up to
  10,000 points per stratum), from 6,887,773 original rows. The file includes
  7,517 genome-wide-significant points. Asset SHA-256:
  `bc10189007039607322a75ccd5eb8af83949301142ab43453bb711902be005ff`.
- `mdd.real.json.gz` is derived from `demo_data/processed/mdd.json.gz`
  (source SHA-256 `232099949603e63a4b49851e462d6b40f82cfcfafd807a69fc7ba58538f88630`).
  It preserves all 176,596 original processed rows using the same
  stratified algorithm, from 8,848,589 original rows, including 332
  genome-wide-significant points. Asset SHA-256:
  `e8f07d05c06ab111eba1a2c00f46b2400b3f14a3344ea7d5cb75889f8e201ae7`.

## Auto Analysis demo

`chr1_all_loci.csv` contains 503 real chromosome-1 locus rows with canonical
`variant_id` (chromosome, position, reference and alternate alleles), independent
`source_rsid`, coordinate, p-value, locus assignment, and LD r2. It is derived by exact
chromosome filtering from the original GWASMind `flask_app/all_loci_ld_final.csv`
(source SHA-256 `bb42f397742759ad523a2693bcb8dfebb2c5a4143af141b473135b4e68963bc8`).
All 503 rows match exactly one row in the unchanged `asthma.real.json.gz` by
chromosome, position, and decimal p-value. Their reference and alternate alleles
come from those same-source rows; none are inferred from rsIDs. The generator
rejects missing or ambiguous matches and unknown alleles before writing the CSV.
The original coordinates, p-values, rsIDs, locus assignments, r2 values, and row
order are preserved. `source_rsid` is provenance and does not compete with the
parser's `variant_id` column. Existing Auto Analysis limits retain 453 variants
across the eight loci, with the same selection and order as before this identity
repair. The 503 original rows remain in the downloadable file.
Every one of its eight loci contains at least one p-value below `5e-8`, so the
demo can exercise evidence retrieval rather than a predetermined abstention.
Asset SHA-256:
`c98a862e5d717c54c5994bd93c57296bd73dd4d18231d785549cff776509cedf`.

This Auto Analysis demo is specific to asthma. MDD has no bundled Auto Analysis
locus CSV; its unchanged Manhattan asset already includes its source alleles.

Rebuild all three files with `scripts/build_demo_assets.py`. The gzip output
uses a zero timestamp, fixed seeds, stable ordering, and compact JSON so hashes
are reproducible.
When `--locus-source` is supplied, the generator uses the newly built asthma
asset for the exact allele match. To rebuild only the CSV without rewriting the
Manhattan assets, call `build_locus_asset(source, destination,
variant_source=Path("asthma.real.json.gz"))` with the existing same-source asset.
