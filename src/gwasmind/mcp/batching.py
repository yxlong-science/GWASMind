"""Combine bounded variant retrieval calls before candidate discovery."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from ..audit import sha256_json
from ..models import LocusQuery, analysis_variant_ids
from .contracts import ToolCoverage, ToolEnvelope, ToolError, ToolStatus
from .ensembl import ENSEMBL_REST_ENDPOINT, ENSEMBL_REST_SOURCE
from .normalization import semantic_record_to_evidence


def variant_batches(query: LocusQuery, *, size: int) -> tuple[tuple[str, ...], ...]:
    # The first call owns the source-region census. Proxy calls cannot replace
    # it with a larger interval, or turn missing proxy data into a gene desert.
    source = tuple(dict.fromkeys(row.variant_id for row in query.variants))
    proxies = tuple(identifier for identifier in analysis_variant_ids(query) if identifier not in source)
    return (source, *(proxies[start:start + size] for start in range(0, len(proxies), size)))


def annotate_analysis_origin(records, query: LocusQuery):
    if query.ld_expansion is None:
        return records
    selected = {row.variant_id for row in query.variants}
    analysis = set(analysis_variant_ids(query))
    links = query.ld_expansion.get("ld_links", [])
    links_by_proxy = {}
    for link in links:
        links_by_proxy.setdefault(link["proxy_variant_id"], []).append(link)
    result = []
    for row in records:
        subject = row.get("subject")
        subject = subject if isinstance(subject, Mapping) else {}
        identifier = subject.get("id") if subject.get("type") == "variant" else None
        if identifier is None:
            result.append(row)
            continue
        context = deepcopy(row.get("context") or {})
        context["analysis_origin"] = "source_selected" if identifier in selected else "ld_proxy" if identifier in analysis else "other_variant_context"
        if identifier in analysis and identifier not in selected:
            context["ld_proxy_links"] = deepcopy(links_by_proxy.get(identifier, []))
        result.append({**row, "context": context})
    return tuple(result)


def _compatible_values(left, right) -> bool:
    # Numeric 1 and 1.0 carry the same measurement; booleans do not.
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return left == right
    return sha256_json(left) == sha256_json(right)


def _complementary_mapping(left, right):
    """Missing values add no assertion; two present assertions must agree."""
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return None
    result = dict(left)
    for key, value in right.items():
        previous = result.get(key)
        if previous is not None and value is not None and not _compatible_values(previous, value):
            return None
        if previous is None:
            result[key] = deepcopy(value)
    return result


def _native_projection_valid(row, normalized) -> bool:
    """Check explicit source bindings before comparing duplicate projection views."""
    upstream = row.get("upstream") or {}
    raw = upstream.get("raw") or {}
    if not isinstance(raw, Mapping):
        return False
    context = row.get("context") or {}
    if not isinstance(context, Mapping):
        return False
    # These semantic locus calls operate in GRCh38. An altered explicit build
    # cannot escape the identity check merely by changing the normalized ID.
    for source in (row, context, raw):
        for key in ("genome_build", "genomeBuild", "assembly", "assembly_name"):
            if source.get(key) not in (None, "", "GRCh38", "hg38"):
                return False
    if normalized.variant_id and raw.get("queryVariantId") not in (None, normalized.variant_id):
        return False
    if normalized.gene_id:
        for key in ("geneId", "targetId", "qtlGeneId"):
            if raw.get(key) not in (None, normalized.gene_id):
                return False
    if normalized.study_id and raw.get("studyId") not in (None, normalized.study_id):
        return False
    module = row.get("module")
    if module == "locus.credible_set":
        member = raw.get("matchedLocusMember") or {}
        object_value = row.get("object") or {}
        if (row.get("relation") != "credible_set_membership" or not isinstance(member, Mapping)
                or not isinstance(object_value, Mapping)
                or object_value.get("type") != "credible_set"
                or object_value.get("id") != raw.get("studyLocusId")
                or member.get("variantId") not in (None, normalized.variant_id)):
            return False
        expected = {"member_posterior_probability": member.get("posteriorProbability"),
                    "p_value_mantissa": raw.get("pValueMantissa"),
                    "p_value_exponent": raw.get("pValueExponent")}
        measurements = row.get("measurements") or {}
        if not isinstance(measurements, Mapping):
            return False
        if any(value is not None and (key not in expected or not _compatible_values(value, expected[key]))
               for key, value in measurements.items()):
            return False
    return not (module == "locus.study" and raw.get("studyType") == "gwas" and normalized.gene_id)


def _merge_projection_views(left, right):
    """Retain complementary fields without changing the scientific assertion role."""
    modules = {left.get("module"), right.get("module")}
    if len(modules) == 1:
        if left.get("relation") != right.get("relation") or left.get("object") != right.get("object"):
            return None
        primary = min((left, right), key=sha256_json)
    elif modules == {"locus.study", "locus.credible_set"}:
        study = left if left.get("module") == "locus.study" else right
        if study.get("relation") != "study_metadata":
            return None
        # Keep the membership relation and its lead/member context, while the
        # complementary study view contributes publication/study metadata.
        primary = left if left.get("module") == "locus.credible_set" else right
    else:
        return None
    if left.get("dataset") != right.get("dataset") or left.get("subject") != right.get("subject"):
        return None
    measurements = _complementary_mapping(left.get("measurements") or {}, right.get("measurements") or {})
    context = _complementary_mapping(left.get("context") or {}, right.get("context") or {})
    if measurements is None or context is None:
        return None
    return {**primary, "measurements": measurements, "context": context}


def _record_release_valid(row, normalized, envelope: ToolEnvelope) -> bool:
    """An envelope's OT release does not rename its independent Ensembl context."""
    if (normalized.source != ENSEMBL_REST_SOURCE
            and envelope.data_release in {"unknown", "live", normalized.data_release}):
        return True
    if normalized.dataset != "ensembl_region_gene_context":
        return False
    provenance = row.get("provenance") or {}
    context = row.get("context") or {}
    subject = row.get("subject") or {}
    derived = row.get("derived") or {}
    object_value = row.get("object")
    if not all(isinstance(value, Mapping) for value in (provenance, context, subject, derived)):
        return False
    summary = row.get("relation") == "regional_gene_context_summary" and object_value is None
    nearby = (row.get("relation") in {"region_overlap", "nearby"}
              and isinstance(object_value, Mapping) and object_value.get("type") == "target")
    return (
        row.get("module") == "locus.variant_consequence"
        and normalized.source == ENSEMBL_REST_SOURCE
        and normalized.data_release == "GRCh38"
        and provenance.get("endpoint") == ENSEMBL_REST_ENDPOINT
        and subject.get("type") == "region"
        and isinstance(subject.get("id"), str) and subject["id"].startswith("GRCh38:")
        and (context.get("genome_build") == "GRCh38" or (
            subject["id"] == "GRCh38:unresolved" and context.get("genome_build") is None))
        and derived.get("region_gene_context") is True
        and derived.get("unranked") is True
        and derived.get("not_direct_locus_to_gene_evidence") is True
        and (summary or nearby)
    )


def merge_variant_envelopes(envelopes: tuple[ToolEnvelope, ...]) -> ToolEnvelope:
    """Completeness is an all-batches fact; a later success cannot erase gaps."""
    first = envelopes[0]
    requested = tuple(dict.fromkeys(module for envelope in envelopes for module in envelope.coverage.requested))
    coverage = {}
    for field in ("unavailable", "blocked", "failed", "truncated"):
        coverage[field] = tuple(module for module in requested if any(
            module in getattr(envelope.coverage, field) for envelope in envelopes))
    # Malformed or unsuccessful whole-call outcomes must not advertise complete
    # coverage just because their module lists were absent or optimistic.
    failed = set(coverage["failed"])
    for envelope in envelopes:
        if envelope.status in {ToolStatus.ERROR, ToolStatus.INVALID_INPUT}:
            failed.update(envelope.coverage.requested or requested)
    coverage["failed"] = tuple(module for module in requested if module in failed)
    incomplete = set().union(*(set(coverage[field]) for field in coverage))
    coverage["completed"] = tuple(module for module in requested if module not in incomplete and all(
        module in (*envelope.coverage.completed, *envelope.coverage.no_data) for envelope in envelopes))
    coverage["no_data"] = tuple(module for module in requested if module not in incomplete and all(
        module in envelope.coverage.no_data for envelope in envelopes))
    missing = set(requested).difference(coverage["completed"], coverage["no_data"], incomplete)
    if missing:
        coverage["failed"] = tuple(module for module in requested if module in failed | missing)
    records, fingerprints, conflicts = {}, {}, set()
    invalid_record = False
    for envelope in envelopes:
        for row in envelope.records:
            normalized = semantic_record_to_evidence(row, default_release=envelope.data_release)
            identity = normalized.evidence_id
            if (row.get("module") not in envelope.coverage.requested or not _native_projection_valid(row, normalized)
                    or not _record_release_valid(row, normalized, envelope)):
                conflicts.add(identity)
                invalid_record = True
                continue
            # Native metrics and raw source identity must agree for repeated IDs.
            # Harmless wrapper/provenance text may differ across semantic modules.
            upstream = row.get("upstream") or {}
            signature = sha256_json({"raw": upstream.get("raw"),
                                     "variant_id": normalized.variant_id, "gene_id": normalized.gene_id,
                                     "study_id": normalized.study_id, "release": normalized.data_release})
            if identity in fingerprints and fingerprints[identity] != signature:
                conflicts.add(identity)
            elif identity in records:
                combined = _merge_projection_views(records[identity], row)
                if combined is None:
                    conflicts.add(identity)
                else:
                    records[identity] = combined
            else:
                fingerprints[identity] = signature
                records.setdefault(identity, row)
    for identity in conflicts:
        records.pop(identity, None)
    statuses = {envelope.status for envelope in envelopes}
    error = None
    if conflicts or invalid_record or ToolStatus.INVALID_INPUT in statuses:
        status = ToolStatus.INVALID_INPUT
        error = ToolError("batch_evidence_identity_conflict", "Batched evidence contained conflicting or invalid identities.")
    elif ToolStatus.BLOCKED in statuses:
        status = ToolStatus.BLOCKED
    elif all(envelope.status is ToolStatus.NO_DATA for envelope in envelopes):
        status = ToolStatus.NO_DATA
    elif any(coverage[field] for field in ("unavailable", "blocked", "failed", "truncated")) or any(envelope.next_cursor for envelope in envelopes) or ToolStatus.PARTIAL in statuses:
        status = ToolStatus.PARTIAL
    else:
        status = ToolStatus.SUCCESS
    warnings = tuple(dict.fromkeys(warning for envelope in envelopes for warning in envelope.warnings))
    if status is ToolStatus.PARTIAL:
        warnings = (*warnings, "Some variant retrieval batches were incomplete; verified records were retained and the combined scope remains incomplete.")
    return replace(first, call_id="variant-batches:" + sha256_json([item.call_id for item in envelopes])[:24],
                   input_sha256=sha256_json([item.input_sha256 for item in envelopes]), status=status,
                   coverage=ToolCoverage(requested=requested, **coverage), records=tuple(records.values()),
                   next_cursor="incomplete-variant-batches" if any(item.next_cursor for item in envelopes) else None,
                   warnings=warnings, error=error, raw_artifact=None)
