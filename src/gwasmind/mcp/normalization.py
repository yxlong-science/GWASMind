from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from ..audit import sha256_json
from ..models import EvidenceKind, EvidenceRecord


def semantic_record_to_evidence(
    record: Mapping[str, Any],
    *,
    default_release: str,
) -> EvidenceRecord:
    module = _required(record.get("module"), "module")
    dataset = _required(record.get("dataset"), "dataset")
    subject = record.get("subject") if isinstance(record.get("subject"), Mapping) else {}
    object_value = record.get("object") if isinstance(record.get("object"), Mapping) else {}
    context = record.get("context") if isinstance(record.get("context"), Mapping) else {}
    provenance = (
        record.get("provenance")
        if isinstance(record.get("provenance"), Mapping)
        else {}
    )
    subject_type = str(subject.get("type") or "")
    object_type = str(object_value.get("type") or "")
    variant_id = (
        _optional(subject.get("id")) if subject_type == "variant" else None
    )
    gene_id = (
        _optional(object_value.get("id"))
        if object_type == "target"
        else _optional(subject.get("id"))
        if subject_type == "target"
        else None
    )
    tissue = _optional(context.get("biosample"))
    if tissue is None and object_type == "biosample":
        tissue = _optional(object_value.get("label") or object_value.get("id"))
    payload = dict(record)
    parent_variants = context.get("parent_variant_ids")
    if not isinstance(parent_variants, (list, tuple)):
        parent_variants = ()
    release = _optional(provenance.get("data_release")) or default_release
    source = _optional(provenance.get("source")) or "Open Targets Platform"
    upstream_id = _optional(provenance.get("evidence_id"))
    upstream = record.get("upstream") if isinstance(record.get("upstream"), Mapping) else {}
    raw_upstream = upstream.get("raw")
    if upstream_id:
        identity = {
            "source": source,
            "release": release,
            "upstream_id": upstream_id,
            "kind": evidence_kind_for_module(module).value,
            # Upstream row IDs need not be globally unique across biological
            # anchors. Never let the first gene/variant silently erase another.
            "variant_id": variant_id,
            "gene_id": gene_id,
        }
        evidence_id = f"semantic-upstream:{sha256_json(identity)[:24]}"
    elif isinstance(raw_upstream, Mapping) and raw_upstream:
        # Several semantic modules can wrap the same Open Targets row.  A
        # module-independent upstream identity prevents those aliases from
        # inflating evidence counts or apparent dataset independence.
        identity = {
            "source": source,
            "release": release,
            "kind": evidence_kind_for_module(module).value,
            "subject": subject,
            "object": object_value,
            "raw": raw_upstream,
        }
        evidence_id = f"semantic-upstream:{sha256_json(identity)[:24]}"
    else:
        evidence_id = f"semantic:{sha256_json(payload)[:24]}"
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind=evidence_kind_for_module(module),
        dataset=dataset,
        source=source,
        data_release=release,
        payload=payload,
        variant_id=variant_id,
        gene_id=gene_id,
        study_id=_optional(context.get("study_id")),
        pmid=_optional(context.get("pmid")),
        credible_set_id=_optional(context.get("study_locus_id")),
        parent_variant_ids=tuple(str(item) for item in parent_variants),
        tissue=tissue,
        evidence_date=_parse_date(
            provenance.get("publication_date")
            or context.get("publication_date")
        ),
    )


def semantic_records_to_evidence(
    records: Iterable[Mapping[str, Any]],
    *,
    default_release: str,
) -> tuple[EvidenceRecord, ...]:
    values: dict[str, EvidenceRecord] = {}
    for record in records:
        normalized = semantic_record_to_evidence(
            record, default_release=default_release
        )
        values.setdefault(normalized.evidence_id, normalized)
    return tuple(values.values())


def evidence_kind_for_module(module: str) -> EvidenceKind:
    if module == "evidence.europepmc":
        return EvidenceKind.LITERATURE
    if module == "locus.qtl":
        return EvidenceKind.QTL
    if module == "locus.colocalisation":
        return EvidenceKind.COLOCALIZATION
    if module == "locus.enhancer_to_gene":
        return EvidenceKind.REGULATORY
    if module == "locus.l2g":
        return EvidenceKind.L2G
    if module in {"locus.variant", "locus.variant_consequence"}:
        return EvidenceKind.VARIANT_ANNOTATION
    if module.startswith("locus."):
        return EvidenceKind.STUDY
    if module == "target.expression":
        return EvidenceKind.EXPRESSION
    if module in {"target.go", "target.pathway"}:
        return EvidenceKind.PATHWAY
    if module == "target.safety":
        return EvidenceKind.SAFETY
    if module == "target.tractability":
        return EvidenceKind.TRACTABILITY
    if module.startswith("target.interaction"):
        return EvidenceKind.INTERACTION
    if module.startswith("target.mouse"):
        return EvidenceKind.PHENOTYPE
    if module.startswith("target."):
        return EvidenceKind.FUNCTION
    if module.startswith("disease.") or module.startswith("context."):
        return EvidenceKind.ONTOLOGY
    if module.startswith("association.") or module.startswith("evidence."):
        return EvidenceKind.TARGET_DISEASE
    if module.startswith("drug."):
        return EvidenceKind.DRUG
    if module.startswith("gene_set."):
        return EvidenceKind.GENE_SET
    raise ValueError(f"unmapped semantic evidence module: {module}")


def _required(value: Any, label: str) -> str:
    text = _optional(value)
    if text is None:
        raise ValueError(f"semantic record {label} is required")
    return text


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_date(value: Any) -> date | None:
    text = _optional(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None
