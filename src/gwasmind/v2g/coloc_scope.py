"""Frozen phenotype ownership and compact background colocalisation receipts."""

from bisect import bisect_left
from collections.abc import Mapping

from ..configuration import parameter

POLICY = "frozen_molqtl_phenotype_stratified_v4"


def validate_background_rows(rows):
    """Reject earlier background shapes instead of converting unknown identities to zero."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Unsupported colocalisation background schema: background rows must be an array.")  # noqa: TRY004 -- schema validation consistently raises ValueError
    for row in rows:
        if (not isinstance(row, Mapping) or row.get("retrieval_policy") != POLICY
                or "phenotype_counts" in row
                or any(not isinstance(row.get(key), list)
                       or not all(isinstance(value, str) and value for value in row[key])
                       or len(set(row[key])) != len(row[key])
                       for key in ("phenotype_ids", "gwas_study_ids"))):
            raise ValueError(
                "Unsupported colocalisation background schema; expected "
                f"{POLICY} with explicit phenotype_ids and gwas_study_ids. "
                "Retrieve the evidence again with the current version.")


def scope_classifier(bundle):
    scope = bundle.coverage.get("phenotype_scope") or {}
    ids = bundle.coverage.get("phenotype_scope_ids", ())
    target = (scope.get("target") or {}).get("id") or (ids[0] if ids else None)
    related = {row["id"] for row in scope.get("related", ())} | set(ids[1:])

    def classify(study):
        observed = {row.get("id") for row in study.get("diseases", ()) if isinstance(row, Mapping)}
        if target and target in observed:
            return "target_phenotype"
        if observed & related:
            return "related_phenotype"
        return "other_phenotype"

    return classify


def study_phenotype_ids(diseases, trait=None, scope_ids=()):
    """Phenotype identities of one GWAS study: its EFO IDs, restricted to the scope when one is given.

    Different studies of one phenotype share an identity, so they are never counted as more phenotypes.
    """
    observed = [row.get("id") for row in diseases or () if isinstance(row, Mapping) and isinstance(row.get("id"), str)]
    matched = [identity for identity in scope_ids if identity in observed] if scope_ids else sorted(set(observed))
    return matched or [f"unmapped:{trait or 'unreported phenotype'}"]


def _insert(values, value):
    index = bisect_left(values, value)
    if index == len(values) or values[index] != value:
        values.insert(index, value)


def add_background(bundle, qtl, study, row):
    h4 = row.get("h4")
    if isinstance(h4, bool) or not isinstance(h4, (int, float)) or not 0 <= h4 <= 1:
        return
    key = (qtl["gene_id"], qtl["study_locus_id"])
    source = qtl["qtl"]
    item = bundle.coloc_background.setdefault(key, {
        "gene_id": qtl["gene_id"], "qtl_study_locus_id": qtl["study_locus_id"],
        "qtl_study_id": (source.get("study") or {}).get("id"), "qtl_type": source.get("studyType"),
        "biosample": ((source.get("study") or {}).get("biosample") or {}).get("biosampleName"),
        "phenotype_scope": "other_phenotype", "retrieval_policy": POLICY,
        "pair_count": 0, "max_h4": 0.0, "high_h4_pair_count": 0,
        "high_h4_threshold": parameter("v2g.coloc.h4_other"), "phenotype_ids": [], "gwas_study_ids": [],
        "locus_anchor_variant_ids": qtl["anchor_variants"],
    })
    item["pair_count"] += 1
    item["max_h4"] = max(item["max_h4"], h4)
    item["high_h4_pair_count"] += h4 >= item["high_h4_threshold"]
    for identity in study_phenotype_ids(study.get("diseases"), study.get("traitFromSource")):
        _insert(item["phenotype_ids"], identity)
    if isinstance(study.get("id"), str):
        _insert(item["gwas_study_ids"], study["id"])


def coloc_sort_key(row):
    return ({"target_phenotype": 0, "related_phenotype": 1}.get(row.get("phenotype_scope"), 2),
            -(row.get("h4") or 0), str(row.get("qtl_study_locus_id")), str(row.get("gwas_study_locus_id")))
