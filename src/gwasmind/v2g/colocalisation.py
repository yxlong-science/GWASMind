"""Query every frozen molecular-QTL credible set, independently of its yield."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from ..configuration import parameter
from .coloc_scope import POLICY, add_background, scope_classifier
FIELDS = (
    "h3 h4 clpp betaRatioSignAverage colocalisationMethod numberColocalisingVariants "
    "otherStudyLocus{ studyLocusId studyType study{ id traitFromSource projectId pubmedId "
    "publicationFirstAuthor publicationDate diseases{ id name } } }"
)


def freeze_molqtl(bundle, locus):
    """Use the same qualifying QTL records as the cards, before any display limits."""
    from .evidence import _qtl

    frozen, invalid = {}, 0
    by_query = locus.by_query_id()
    for gene_id, target in sorted(bundle.targets.items()):
        status = bundle.target_status.get(gene_id, {}).get("qtl", "not_requested")
        category = _qtl(target, locus, status, ())
        for card in category["records"]:
            identifier = card.get("study_locus_id")
            if not isinstance(identifier, str) or not identifier:
                invalid += 1
                continue
            source = card["_source"]["row"]
            shared = sorted({(member.get("variant") or {}).get("id")
                             for member in (source.get("locus") or {}).get("rows") or []
                             if (member.get("variant") or {}).get("id") in by_query})
            entry = {"gene_id": gene_id, "study_locus_id": identifier,
                     "anchor_variants": shared, "qtl": copy.deepcopy(source)}
            if identifier in frozen and frozen[identifier] != entry:
                invalid += 1
                continue
            frozen[identifier] = entry
    return frozen, invalid


def _connection(data, identifier):
    root = data.get("credibleSets") if isinstance(data, Mapping) else None
    rows = root.get("rows") if isinstance(root, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("Missing credible-set response")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("studyLocusId") == identifier]
    if len(matches) != 1 or not isinstance(matches[0].get("colocalisation"), Mapping):
        raise ValueError("Frozen molecular-QTL anchor not returned uniquely")
    connection = matches[0]["colocalisation"]
    count, values = connection.get("count"), connection.get("rows")
    if type(count) is not int or count < 0 or not isinstance(values, list):
        raise ValueError("Invalid colocalization connection")
    if any(not isinstance(row, Mapping) or not isinstance(row.get("otherStudyLocus"), Mapping)
           or not row["otherStudyLocus"].get("studyLocusId") for row in values):
        raise ValueError("Incomplete colocalization record identity")
    return count, values


async def retrieve_frozen_colocalisation(retriever, bundle, locus, trait_ids):
    from .retrieval import _unique_rows

    bundle.coverage.setdefault("phenotype_scope_ids", list(trait_ids))
    frozen, invalid = freeze_molqtl(bundle, locus)
    page_size = parameter("v2g.coloc.page_size")
    batch_size = parameter("v2g.coloc.batch_size")
    classify = scope_classifier(bundle)
    bundle.frozen_molqtl = frozen
    ids = sorted(frozen)
    retriever.progress.update(completed_count=0, total_count=len(ids))
    manifest = [{"study_locus_id": identifier, "gene_id": frozen[identifier]["gene_id"],
                 "anchor_variants": frozen[identifier]["anchor_variants"]} for identifier in ids]
    coverage = {
        "anchor_kind": "frozen_molqtl_credible_set", "frozen_molqtl_ids": ids,
        "frozen_molqtl_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "frozen_molqtl_count": len(ids), "queried_molqtl_count": 0,
        "completed_molqtl_count": 0, "failed_molqtl_count": 0,
        "invalid_molqtl_identity_count": invalid, "per_anchor": {}, "complete": False,
        "retrieval_policy": POLICY,
        "raw_colocalisation_rows": 0, "gwas_counterpart_rows": 0, "excluded_non_gwas_rows": 0,
    }
    bundle.coverage["colocalisation"] = coverage
    initial_query = ("query V2GMolQtlColoc($ids:[String!]!,$size:Int!){ "
        "credibleSets(studyLocusIds:$ids,page:{index:0,size:$size}){ rows{ studyLocusId "
        f"colocalisation(studyTypes:[gwas],page:{{index:0,size:{page_size}}}){{ count rows{{ {FIELDS} }} }} }} }} }}")
    page_query = ("query V2GMolQtlColocPage($id:String!,$index:Int!,$size:Int!){ "
        "credibleSets(studyLocusIds:[$id],page:{index:0,size:1}){ rows{ studyLocusId "
        f"colocalisation(studyTypes:[gwas],page:{{index:$index,size:$size}}){{ count rows{{ {FIELDS} }} }} }} }} }}")
    gwas_by_id = {}
    for offset in range(0, len(ids), batch_size):
        batch = ids[offset:offset + batch_size]
        try:
            initial = await retriever._run(bundle, "molqtl_colocalisation", initial_query,
                                          {"ids": batch, "size": len(batch)})
        except Exception:  # A failed batch is retried per anchor so one cannot hide the others.
            initial = None
        for identifier in batch:
            coverage["queried_molqtl_count"] += 1
            receipt = {"status": "failed", "complete": False, "returned_rows": 0, "expected_rows": None}
            coverage["per_anchor"][identifier] = receipt
            values = []
            try:
                data = initial
                if data is None:
                    data = await retriever._run(bundle, "molqtl_colocalisation_retry", initial_query,
                                               {"ids": [identifier], "size": 1})
                expected, values = _connection(data, identifier)
                values = _unique_rows(values)
                receipt.update(status="partial", expected_rows=expected)
                page = 1
                while len(values) < expected:
                    data = await retriever._run(bundle, "molqtl_colocalisation_page", page_query,
                                               {"id": identifier, "index": page, "size": page_size})
                    count, chunk = _connection(data, identifier)
                    if count != expected:
                        raise ValueError("Colocalization count changed during pagination")
                    expanded = _unique_rows([*values, *chunk])
                    if len(expanded) == len(values):
                        raise ValueError("Colocalization pagination made no progress")
                    values = expanded
                    page += 1
                if len(values) != expected:
                    raise ValueError("Colocalization count does not match unique rows")
                receipt.update(status="complete", complete=True)
                coverage["completed_molqtl_count"] += 1
            except Exception as error:
                receipt["error_type"] = type(error).__name__
                bundle.issues.append(f"Colocalization query for molecular-QTL credible set {identifier} "
                                     f"is incomplete ({type(error).__name__}).")
                coverage["failed_molqtl_count"] += 1
            receipt["returned_rows"] = len(values)
            coverage["raw_colocalisation_rows"] += len(values)
            retriever.progress.update(completed_count=coverage["completed_molqtl_count"],
                                      records=coverage["raw_colocalisation_rows"])
            qtl = frozen[identifier]
            for result in values:
                other = result.get("otherStudyLocus") or {}
                study = other.get("study") or {}
                if other.get("studyType") != "gwas":
                    coverage["excluded_non_gwas_rows"] += 1
                    continue
                coverage["gwas_counterpart_rows"] += 1
                phenotype_scope = classify(study)
                if phenotype_scope == "other_phenotype":
                    add_background(bundle, qtl, study, result)
                    continue
                gwas_id = other["studyLocusId"]
                gwas = gwas_by_id.setdefault(gwas_id, {"studyLocusId": gwas_id,
                    "study": copy.deepcopy(study), "anchor_variants": [], "colocalisation": []})
                gwas["anchor_variants"] = sorted(set(gwas["anchor_variants"]) | set(qtl["anchor_variants"]))
                # This is a derived gene-facing orientation; the untouched provider row is retained.
                gwas["colocalisation"].append({**result,
                    "_phenotype_scope": phenotype_scope,
                    "otherStudyLocus": {**qtl["qtl"], "qtlGeneId": qtl["gene_id"]},
                    "_queried_molqtl_id": identifier, "_anchor_variants": qtl["anchor_variants"],
                    "_upstream_row": copy.deepcopy(result)})
    coverage["complete"] = coverage["completed_molqtl_count"] == len(ids) and invalid == 0
    coverage["gwas_credible_sets_returned"] = len(gwas_by_id)
    bundle.gwas_credible_sets = [gwas_by_id[key] for key in sorted(gwas_by_id)]
    bundle.coloc_status = ("observed" if coverage["complete"] and (gwas_by_id or bundle.coloc_background) else
                          "not_observed" if coverage["complete"] else "partial")
