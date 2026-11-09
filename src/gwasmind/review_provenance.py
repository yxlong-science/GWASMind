"""Read-only source tracing and dependence facts for pre-generation W5/W9 review.

Identifiers are interpreted, never externally verified. These functions do not
decide whether a task requires independent replication or a particular paper.
Callers supply the complete, predeclared review scope and decide applicability.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

_STUDY_KINDS = frozenset({"qtl", "colocalization", "gwas", "study", "literature", "regulatory"})
_PUB_FIELDS = {"pmid": "pmid", "pmids": "pmid", "pubmedid": "pmid", "pubmedids": "pmid",
    "doi": "doi", "dois": "doi", "publicationdoi": "doi", "pmcid": "pmcid", "pmcids": "pmcid",
    "ppr": "ppr", "pprid": "ppr", "preprintid": "ppr"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _name(value: Any) -> str:
    return str(value).casefold().replace("_", "")


def _values(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def _hypothesis(record: Mapping[str, Any]) -> str:
    context = _mapping(_mapping(record.get("payload")).get("context"))
    gene = _text(record.get("gene_id")) or "unspecified_gene"
    explicit = _text(context.get("hypothesis_id") or context.get("comparison_id"))
    return f"{gene}|{explicit}" if explicit else gene


def publication_identifier(value: Any, *, declared_type: str | None = None) -> dict[str, Any]:
    """Recognize only explicit PMID, DOI, PMCID or Europe PMC preprint syntax.

    A PPR in a pmid field retains its PPR namespace. A DOI or PMID by itself
    does not establish peer review, publication status or a successful lookup.
    """
    text = str(value).strip() if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
    declared = _PUB_FIELDS.get(_name(declared_type), declared_type)
    kind, identifier = None, None
    pmid = re.fullmatch(r"(?:PMID\s*:\s*|https?://pubmed\.ncbi\.nlm\.nih\.gov/|"
                       r"https?://europepmc\.org/article/MED/)([1-9]\d*)/?", text, re.I)
    pmcid = re.fullmatch(r"(?:PMCID\s*:\s*|https?://(?:pmc\.ncbi\.nlm\.nih\.gov|"
                        r"www\.ncbi\.nlm\.nih\.gov/pmc)/articles/|https?://europepmc\.org/article/PMC/)?"
                        r"(PMC[1-9]\d*)/?", text, re.I)
    ppr = re.fullmatch(r"(?:https?://europepmc\.org/article/PPR/)?(PPR[1-9]\d*)/?", text, re.I)
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", text, flags=re.I)
    if pmid:
        kind, identifier = "pmid", pmid[1]
    elif declared == "pmid" and re.fullmatch(r"[1-9]\d*", text):
        kind, identifier = "pmid", text
    elif pmcid:
        kind, identifier = "pmcid", pmcid[1].upper()
    elif ppr:
        kind, identifier = "ppr", ppr[1].upper()
    elif re.fullmatch(r"10\.\d{4,9}/\S+", doi, re.I):
        kind, identifier = "doi", doi.casefold()
    return {"kind": kind, "identifier": identifier,
        "canonical": f"{kind}:{identifier}" if kind else None,
        "declared_type": declared, "field_type_mismatch": bool(kind and declared and kind != declared),
        "publication_status": "preprint" if kind == "ppr" else "not_established",
        "externally_verified": False}


def _publications(views: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict], list[str]]:
    found, scalar_slots, notes = set(), defaultdict(set), set()
    for view in views:
        for key, value in view.items():
            declared = _PUB_FIELDS.get(_name(key))
            if declared is None:
                continue
            for item in _values(value):
                if item is None or item == "":
                    continue
                parsed = publication_identifier(item, declared_type=declared)
                if parsed["canonical"]:
                    found.add(parsed["canonical"])
                    if not isinstance(value, (list, tuple)) and not _name(key).endswith("s"):
                        scalar_slots[parsed["kind"]].add(parsed["canonical"])
                    if parsed["field_type_mismatch"]:
                        notes.add(f"publication_identifier_retyped:{declared}->{parsed['kind']}")
                else:
                    notes.add(f"unrecognized_publication_identifier:{declared}")
    conflicts = [{"reason_code": "publication_identifier_conflict", "namespace": namespace,
                  "identifiers": sorted(values)} for namespace, values in sorted(scalar_slots.items()) if len(values) > 1]
    return sorted(found), conflicts, sorted(notes)


def _identifiers(views: Sequence[Mapping[str, Any]], fields: set[str]) -> list[str]:
    return sorted({_text(value) for view in views for key, item in view.items() if _name(key) in fields
                   for value in _values(item) if _text(value)})


def _role(role: str, views: Sequence[Mapping[str, Any]], *, source: str, release: str,
          extra_record_ids: Sequence[str] = ()) -> dict[str, Any]:
    studies = _identifiers(views, {"studyid"})
    datasets = _identifiers(views, {"sourcedatasetid"})
    record_ids = sorted(set(extra_record_ids) | set(_identifiers(views,
        {"studylocusid", "recordid", "sourcerecordid", "enhancerstudyid", "regulatoryelementkey"})))
    publication_ids, conflicts, notes = _publications(views)
    for namespace, ids in (("study", studies), ("dataset", datasets)):
        if len(ids) > 1:
            conflicts.append({"reason_code": f"{namespace}_identifier_conflict", "identifiers": ids})
    source_conflict = any(item["reason_code"] in {"study_identifier_conflict", "dataset_identifier_conflict"} for item in conflicts)
    # A named original study is an anchor even when its release is unavailable.
    # An arbitrary/generated ledger ID is never an original-record fallback.
    traceable = bool(source and (studies or datasets or (release and record_ids)) and not source_conflict)
    if traceable and not release:
        notes.append("data_release_not_supplied")
    return {"role":role, "source":source or None, "data_release":release or None,
        "study_ids":studies, "source_dataset_ids":datasets, "original_record_ids":record_ids,
        "publication_ids":publication_ids, "data_traceable":traceable,
        "publication_locatable":bool(publication_ids) and not any(item["reason_code"] == "publication_identifier_conflict" for item in conflicts),
        "publication_status":"preprint_identifier_present" if any(value.startswith("ppr:") for value in publication_ids) else "not_established",
        "identifier_conflicts":conflicts, "notes":sorted(set(notes)), "externally_verified":False}


def _is_background_summary(record: Mapping[str, Any], raw: Mapping[str, Any]) -> bool:
    return (_text(record.get("dataset")) == "colocalisation_background_summary"
            or ("qtl_study_id" in raw and "pair_count" in raw and raw.get("phenotype_scope") == "other_phenotype"))


def _aggregate_role(raw: Mapping[str, Any], *, source: str, release: str) -> dict[str, Any]:
    studies = sorted({_text(value) for value in _values(raw.get("gwas_study_ids")) if _text(value)})
    traceable = bool(source and studies)
    notes = [] if studies else ["aggregate_gwas_study_ids_not_recorded"]
    if traceable and not release:
        notes.append("data_release_not_supplied")
    return {"role": "gwas_aggregate", "source": source or None, "data_release": release or None,
        "study_ids": studies, "source_dataset_ids": [], "original_record_ids": [], "publication_ids": [],
        "data_traceable": traceable, "publication_locatable": False, "publication_status": "not_established",
        "identifier_conflicts": [], "notes": sorted(set(notes)), "externally_verified": False}


def review_record_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep data tracing, publication location and coloc endpoints distinct."""
    payload = _mapping(record.get("payload")); context = _mapping(payload.get("context"))
    provenance = _mapping(payload.get("provenance")); upstream = _mapping(payload.get("upstream"))
    raw = _mapping(upstream.get("raw"))
    kind = str(record.get("kind") or "")
    source = next((_text(value) for value in (record.get("source"), provenance.get("source")) if _text(value)), "")
    release = next((_text(value) for value in (record.get("data_release"), provenance.get("data_release")) if _text(value)), "")
    views = [record, payload, context, provenance, upstream, raw, _mapping(raw.get("study")), _mapping(raw.get("publication"))]
    generic_ids = _identifiers(views, {"studyid"})
    roles, conflicts = [], []
    releases = {_text(value) for value in (record.get("data_release"), provenance.get("data_release")) if _text(value)}
    if len(releases) > 1:
        conflicts.append({"reason_code":"data_release_conflict", "identifiers":sorted(releases)})
    if kind == "colocalization" and _is_background_summary(record, raw):
        # One QTL signal summarized against many GWAS studies: its GWAS side is a set
        # of study IDs, not a second pair endpoint, so several IDs are not a conflict.
        roles.append(_role("qtl", [{"studyId": raw.get("qtl_study_id")}, {"studyLocusId": raw.get("qtl_study_locus_id")}],
                           source=source, release=release))
        roles.append(_aggregate_role(raw, source=source, release=release))
    elif kind == "colocalization":
        side_views = {}
        for side in ("left", "right"):
            values = {key[len(side):]:value for key,value in raw.items() if str(key).startswith(side)}
            nested = _mapping(raw.get(side + "Study"))
            if _text(nested.get("id")):
                nested = {**nested, "studyId":nested["id"]}
            side_views[side] = [values, nested, _mapping(raw.get(side + "Publication"))]
        candidates = [side for side, items in side_views.items()
                      if len(generic_ids) == 1 and generic_ids[0] in _identifiers(items, {"studyid"})]
        if len(candidates) == 1:
            side_views[candidates[0]].extend(views)
        elif generic_ids:
            roles.append(_role("unassigned_primary", views, source=source, release=release))
            endpoints = {value for items in side_views.values() for value in _identifiers(items, {"studyid"})}
            if len(endpoints) == 2 and not set(generic_ids).intersection(endpoints):
                conflicts.append({"reason_code":"primary_study_not_a_colocalization_endpoint", "identifiers":generic_ids,
                                  "endpoint_study_ids":sorted(endpoints)})
        for side, items in side_views.items():
            roles.append(_role(side, items, source=source, release=release))
    else:
        # Raw source IDs are allowed only as versioned original records, never
        # the generated top-level evidence_id or an unrelated target gene ID.
        raw_id = _text(raw.get("id"))
        original = [raw_id] if raw_id and kind in _STUDY_KINDS and raw_id not in {
            _text(record.get("gene_id")), _text(record.get("variant_id"))} else []
        roles.append(_role("primary", views, source=source, release=release, extra_record_ids=original))
    required_roles = [value for value in roles if value["role"] != "unassigned_primary"]
    for value in roles:
        conflicts.extend({**item, "role":value["role"]} for item in value["identifier_conflicts"])
    relevant = kind in _STUDY_KINDS
    return {"evidence_id":_text(record.get("evidence_id")) or None, "kind":kind,
        "gene_id":_text(record.get("gene_id")) or None, "variant_id":_text(record.get("variant_id")) or None,
        "study_evidence":relevant, "source_roles":roles,
        "data_traceable":bool(required_roles) and all(value["data_traceable"] for value in required_roles),
        "publication_locatable":bool(required_roles) and all(value["publication_locatable"] for value in required_roles),
        "identifier_conflicts":conflicts, "notes":sorted({note for value in roles for note in value["notes"]}),
        "externally_verified":False}


def review_provenance(records: Sequence[Mapping[str, Any]], *, require_publication: bool = False,
                      include_records: bool = True, include_groups: bool = True) -> dict[str, Any]:
    """Count identified original-study groups separately from isolated records.

    Only an identical source/release/single-study tuple can share publication
    metadata. An endpoint, publication, gene, proxy or generated evidence ID
    never supplies a missing study key. Isolated counts count input records,
    even when both unidentifiable endpoints of one colocalization are missing.
    include_records=False omits only expanded row profiles; include_groups=False
    additionally omits expanded normal study groups. All counts, gap bindings,
    affected identifiers and notes remain complete in either mode.
    """
    # Profiles are transient in compact mode. Group members need only their
    # source binding, never the complete role/profile graph. A located paper
    # permanently resolves that group's pending publication gaps.
    reviewed, groups = [], {}
    gaps, missing_publications, affected, affected_studies, reasons = [], [], set(), set(), set()
    isolated_gaps, isolated_missing = [], []
    notes = set()
    record_count = study_record_count = conflicted_records = 0
    ungrouped_records = untraceable_records = missing_publication_records = 0

    def binding(member):
        ref, gene, variant, role_name, studies = member
        return {"evidence_id":ref, "gene_id":gene, "variant_id":variant,
                "role":role_name, "study_ids":list(studies)}

    def mark(members, *, unavailable=False, missing_paper=False, orphan=False):
        for member in members:
            if unavailable:
                (isolated_gaps if orphan else gaps).append(binding(member))
            if missing_paper:
                (isolated_missing if orphan else missing_publications).append(binding(member))
            if unavailable or missing_paper and require_publication:
                affected.add(member[0])
                affected_studies.update(member[4])
        if unavailable:
            reasons.add("original_source_untraceable")
        if missing_paper and require_publication:
            reasons.add("required_original_publication_unlocated")

    for index, record in enumerate(records):
        item = review_record_provenance(record)
        record_count += 1
        notes.update(item["notes"])
        if include_records:
            reviewed.append(item)
        if not item["study_evidence"]:
            continue
        study_record_count += 1
        ref = item["evidence_id"]
        ungrouped = untraceable = publication_missing = False
        for role in item["source_roles"]:
            if role["role"] == "unassigned_primary":
                continue
            stable_study = (role["source"] and role["data_release"] and len(role["study_ids"]) == 1
                and not any(conflict["reason_code"] in {"data_release_conflict", "primary_study_not_a_colocalization_endpoint"}
                            for conflict in item["identifier_conflicts"]))
            member = (ref, item["gene_id"], item["variant_id"], role["role"], tuple(role["study_ids"]))
            if stable_study:
                key = (role["source"], role["data_release"], role["study_ids"][0])
                group = groups.get(key)
                if group is None:
                    group = groups[key] = {"publications":set(), "data_traceable":False,
                        "missing_seen":False, "members":[], "record_count":0, "last_index":None,
                        "evidence_ids":set() if include_groups else None}
                group["data_traceable"] |= role["data_traceable"]
                group["missing_seen"] |= not role["publication_locatable"]
                if role["publication_locatable"]:
                    group["publications"].update(role["publication_ids"])
                if group["last_index"] != index:
                    group["record_count"] += 1
                    group["last_index"] = index
                if include_groups:
                    group["members"].append(member)
                    if ref:
                        group["evidence_ids"].add(ref)
                elif group["publications"]:
                    group["members"].clear()
                else:
                    group["members"].append(member)
            else:
                ungrouped = True
                unavailable = not (role["data_traceable"] or role["publication_locatable"])
                missing_paper = not role["publication_locatable"]
                untraceable |= unavailable
                publication_missing |= missing_paper
                if unavailable or missing_paper:
                    mark((member,), unavailable=unavailable, missing_paper=missing_paper, orphan=True)
        ungrouped_records += ungrouped
        untraceable_records += untraceable
        missing_publication_records += publication_missing
        if item["identifier_conflicts"]:
            conflicted_records += 1
            affected.add(ref); reasons.add("source_identifier_conflict")
            affected_studies.update(sid for role in item["source_roles"] for sid in role["study_ids"])

    study_groups, untraceable_studies, missing_studies, shared_publications = [], 0, 0, 0
    for (source, release, study), group in sorted(groups.items()):
        publication_ids = sorted(group["publications"])
        data_traceable = group["data_traceable"]
        unavailable, missing_paper = not (data_traceable or publication_ids), not publication_ids
        untraceable_studies += unavailable
        missing_studies += missing_paper
        if publication_ids and group["missing_seen"]:
            shared_publications += 1
        if unavailable or missing_paper:
            mark(group["members"], unavailable=unavailable, missing_paper=missing_paper)
        if include_groups:
            study_groups.append({"source":source, "data_release":release, "study_id":study,
                "data_traceable":data_traceable, "publication_locatable":bool(publication_ids),
                "publication_ids":publication_ids, "record_count":group["record_count"],
                "evidence_ids":sorted(group["evidence_ids"]),
                "source_roles":[binding(member) for member in group["members"]]})
    gaps.extend(isolated_gaps)
    missing_publications.extend(isolated_missing)
    return {"records":reviewed, "record_count":record_count,
        "study_record_count":study_record_count,
        "original_study_group_count":len(groups), "study_groups":study_groups, "group_details_included":include_groups,
        "untraceable_study_count":untraceable_studies, "missing_publication_count":missing_studies,
        "ungrouped_record_count":ungrouped_records,
        "untraceable_record_count":untraceable_records,
        "missing_publication_record_count":missing_publication_records,
        "conflicted_record_count":conflicted_records, "untraceable_source_roles":gaps,
        "missing_publication_roles":missing_publications,
        "missing_publication_evidence_ids":sorted({row["evidence_id"] for row in missing_publications if row["evidence_id"]}),
        "missing_publication_study_ids":sorted({sid for row in missing_publications for sid in row["study_ids"]}),
        "affected_evidence_ids":sorted(value for value in affected if value),
        "affected_study_ids":sorted(affected_studies),
        "reason_codes":sorted(reasons), "require_publication":require_publication,
        "notes":sorted(notes
                       | ({"some_original_publications_are_not_located"} if missing_publications else set())
                       | ({"publication_metadata_shared_within_identical_source_release_study"} if shared_publications else set())),
        "count_scope":"Study counts use identical source/release/study-ID groups. Ungrouped counts use distinct input records; neither counts independent cohorts or distinct papers.",
        "externally_verified":False}


def review_dependencies(records: Sequence[Mapping[str, Any]], *, include_groups: bool = True) -> dict[str, Any]:
    """Describe source relationships; distinct labels never prove independence.

    Shared publication/datasource labels remain separate from known reuse of a
    study or original record. Even an explicit cohort label does not prove that
    participant sets are identical. No source or evidence row is discarded.
    include_groups=False retains exact counts and assessments without allocating
    expanded per-group evidence inventories; list fields remain present but empty.
    """
    source_buckets, families, publications, dependency = ({} for _ in range(4))
    refs, unidentified, steps, hypothesis_counts = {}, [], defaultdict(set), defaultdict(int)
    missing = object()

    def observe(buckets, key, ref):
        if include_groups:
            buckets.setdefault(key, set()).add(ref)
        else:
            # A first exact ID or None (= at least two distinct IDs) suffices
            # for the exact group counts and dependence state. No sampling.
            prior = buckets.get(key, missing)
            if prior is missing:
                buckets[key] = ref
            elif prior is not None and prior != ref:
                buckets[key] = None

    def multiple_count(buckets):
        return sum(len(value) > 1 for value in buckets.values()) if include_groups else sum(value is None for value in buckets.values())
    for index, record in enumerate(records):
        result = review_record_provenance(record)
        if not result["study_evidence"]:
            continue
        ref = result["evidence_id"] or f"unidentified-record:{index + 1}"
        payload = _mapping(record.get("payload")); context = _mapping(payload.get("context"))
        raw = _mapping(_mapping(payload.get("upstream")).get("raw"))
        # Different candidates are never pooled as repetitions of one claim.
        hypothesis = _hypothesis(record)
        if ref in refs:
            prior_hypothesis = _hypothesis(refs[ref])
            hypothesis_counts[prior_hypothesis] -= 1
            if not hypothesis_counts[prior_hypothesis]:
                del hypothesis_counts[prior_hypothesis]
        refs[ref] = record
        hypothesis_counts[hypothesis] += 1
        if include_groups:
            steps[ref].add(result["kind"])
        anchored = False
        for role in result["source_roles"]:
            if result["identifier_conflicts"] or role["role"] == "unassigned_primary":
                continue
            for sid in role["study_ids"]:
                key = (hypothesis, "study", role["source"] or "unspecified_source", sid)
                observe(source_buckets, key, ref); observe(dependency, key, ref); anchored = True
            for dataset in role["source_dataset_ids"]:
                observe(families, (hypothesis, "source_dataset", dataset), ref)
                anchored = True
            for original in role["original_record_ids"]:
                if role["source"]:
                    key = (hypothesis, "original_record", role["source"], original)
                    observe(dependency, key, ref); anchored = True
            for publication in role["publication_ids"]:
                observe(publications, (hypothesis, publication), ref)
        for field, label in (("source_family", "source_family"), ("source_dataset_id", "source_dataset"),
                             ("cohort_id", "cohort_label"), ("sample_set_id", "sample_set"), ("datasource_id", "datasource")):
            aliases = {_name(field)}
            values = _identifiers([record, context, raw], aliases)
            for value in values:
                observe(families, (hypothesis, label, value), ref)
                if label == "sample_set":
                    observe(dependency, (hypothesis, "sample_set", "explicit_sample_set", value), ref)
        if not anchored:
            unidentified.append(ref)
    for ref, record in refs.items():
        for parent in _values(record.get("parent_evidence_ids", ())):
            if isinstance(parent, str) and parent in refs and parent != ref and _hypothesis(refs[parent]) == _hypothesis(record):
                key = (_hypothesis(record), "derived_from_record", "explicit_parent", parent)
                observe(dependency, key, ref)
                observe(dependency, key, parent)

    def groups(buckets, *, multiple=True):
        result = []
        if not include_groups:
            return result
        for key, values in sorted(buckets.items()):
            if multiple and len(values) < 2:
                continue
            result.append({"hypothesis":key[0], "basis":list(key[1:]), "evidence_ids":sorted(values),
                "record_count":len(values), "mechanism_steps":sorted({step for ref in values for step in steps[ref]})})
        return result

    known = groups(dependency)
    dependency_count = multiple_count(dependency)
    independent = _independent_comparisons(refs)
    established = bool(refs) and {item["hypothesis"] for item in independent} == set(hypothesis_counts)
    independent_cohorts = sorted({cohort for comparison in independent
                                  for cohort in comparison.get("cohort_ids", ())})
    return {"source_groups":groups(source_buckets, multiple=False), "dependency_groups":known,
        "source_family_groups":groups(families), "shared_publication_groups":groups(publications),
        "unidentified_records":sorted(set(unidentified)), "record_count":len(records),
        "group_details_included":include_groups, "source_group_count":len(source_buckets),
        "dependency_group_count":dependency_count, "source_family_group_count":multiple_count(families),
        "shared_publication_group_count":multiple_count(publications),
        "study_record_count":len(refs), "independence_state":"established" if established else "known_dependency" if dependency_count else "unknown" if refs else "not_applicable",
        # Report every cohort represented by a verified disjoint comparison;
        # an arbitrary constant of two would undercount multiple independent
        # cohorts.  Unknown participant overlap never contributes here.
        "independent_cohort_count":len(independent_cohorts) if established else None,
        "independence_established":established, "independent_comparisons":independent,
        "notes":["Source and publication identifiers are not independent-study votes.",
                 "Shared publications or datasource labels do not establish identical cohorts.",
                 "Different source labels do not establish disjoint participants.",
                 "Complementary mechanism steps may share an original experiment."]}


def _independent_comparisons(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Verify a deliberately strict stored participant-membership contract.

    provenance.sample_membership requires: hypothesis_id matching an explicit
    context hypothesis_id/comparison_id, cohort_id, participant_id_namespace,
    complete=True, nonempty participant_ids, their SHA256 (sorted unique IDs,
    UTF-8 JSON, ensure_ascii=False, compact separators), source_record_id bound
    to this record's original study/record IDs, and the same source/data_release.
    Identifiers and a boolean claim of independence alone never suffice. The
    result contains hashes/counts only, never participant IDs. No external
    verification is implied: this checks the complete stored memberships.
    """
    groups = defaultdict(list)
    for ref, record in records.items():
        payload = _mapping(record.get("payload")); context = _mapping(payload.get("context"))
        membership = _mapping(_mapping(payload.get("provenance")).get("sample_membership"))
        hypothesis = _text(context.get("hypothesis_id") or context.get("comparison_id"))
        if not hypothesis or membership.get("hypothesis_id") != hypothesis or membership.get("complete") is not True:
            continue
        result = review_record_provenance(record)
        if result["identifier_conflicts"]:
            continue
        participants = membership.get("participant_ids")
        if not isinstance(participants, (list, tuple)) or not participants or any(not _text(value) for value in participants):
            continue
        participants = frozenset(value.strip() for value in participants)
        digest = hashlib.sha256(json.dumps(sorted(participants), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        namespace, cohort, source_record = (_text(membership.get(field)) for field in
            ("participant_id_namespace", "cohort_id", "source_record_id"))
        if not namespace or not cohort or not source_record or membership.get("participant_ids_sha256") != digest:
            continue
        if not any(role["data_traceable"] and source_record in role["study_ids"] + role["original_record_ids"]
                   and membership.get("source") == role["source"] and membership.get("data_release") == role["data_release"]
                   and role["data_release"] for role in result["source_roles"]):
            continue
        groups[(_hypothesis(record), namespace)].append((ref, cohort, source_record, participants, digest))
    output = []
    for (hypothesis, namespace), rows in sorted(groups.items()):
        # Build a verified disjoint cohort set rather than stopping after the
        # first pair.  Every emitted pair is still auditable, while the union
        # of cohort IDs gives an accurate count when three or more cohorts are
        # available for one hypothesis.
        selected: list[tuple[Any, ...]] = []
        for candidate in sorted(rows):
            parent_ids = set(_values(records[candidate[0]].get("parent_evidence_ids", ())))
            if candidate[0] in parent_ids:
                continue
            if any(candidate[1] == prior[1] or candidate[2] == prior[2]
                   or not candidate[3].isdisjoint(prior[3])
                   or candidate[0] in set(_values(records[prior[0]].get("parent_evidence_ids", ())))
                   or prior[0] in parent_ids for prior in selected):
                continue
            selected.append(candidate)
        if len(selected) >= 2:
            anchor = selected[0]
            for cohort in selected[1:]:
                pair = (anchor, cohort)
                output.append({"hypothesis":hypothesis, "basis":"source_bound_complete_disjoint_participant_sets",
                    "participant_id_namespace":namespace, "evidence_ids":[row[0] for row in pair],
                    "cohort_ids":[row[1] for row in pair], "participant_counts":[len(row[3]) for row in pair],
                    "participant_set_sha256":[row[4] for row in pair], "externally_verified":False})
    return output
