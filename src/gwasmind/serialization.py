from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from .biological_context import (
    biological_context_from_dict,
    curated_biological_context,
    define_biological_context,
)
from .models import DataMode, EvidenceKind, LocusQuery, VariantObservation
from .policy import LeakagePolicy

_VARIANT_ID = re.compile(
    r"^(?:rs\d+|(?:chr)?(?:[0-9]{1,2}|X|Y|MT)(?:_|:)\d+(?:_|:)[ACGTN-]+(?:_|:)[ACGTN-]+)$",
    re.IGNORECASE,
)
_GENE_ID = re.compile(r"^(?:ENSG\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9._-]{0,63})$")


def locus_query_from_dict(
    value: Any, *, strict_identifiers: bool = True, allow_analysis_scope: bool = False,
) -> LocusQuery:
    if not isinstance(value, dict):
        raise ValueError("locus input must be a JSON object")
    allowed = {
        "locus_id",
        "trait",
        "source_trait",
        "variants",
        "candidate_genes",
        "task",
        "ancestry",
        "relevant_tissues",
        "genome_build",
        "ld_reference",
        "benchmark",
        "significance_verified",
        "significance_source",
        "study_ids",
        "trait_ids",
        "qtl_types",
        "requested_evidence_kinds",
        "analysis_variants",
        "ld_expansion",
        "biological_context",
    }
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError("unknown locus fields: " + ", ".join(sorted(unknown)))
    if "analysis_variants" in value and not isinstance(value["analysis_variants"], list):
        raise ValueError("analysis_variants must be an array")
    if not allow_analysis_scope and (value.get("analysis_variants") or value.get("ld_expansion") is not None):
        raise ValueError("LD analysis scope may only be supplied by the private retrieval workflow")
    for name in ("locus_id", "trait", "task"):
        if name in value and not isinstance(value[name], str):
            raise ValueError(f"{name} must be a string")
    for name in ("ancestry", "genome_build", "ld_reference", "significance_source"):
        if value.get(name) is not None and not isinstance(value[name], str):
            raise ValueError(f"{name} must be a string or null")
    variant_rows = value.get("variants")
    if not isinstance(variant_rows, list):
        raise ValueError("variants must be an array")
    variants: list[VariantObservation] = []
    for index, row in enumerate(variant_rows):
        if not isinstance(row, dict):
            raise ValueError(f"variants[{index}] must be an object")
        unknown_variant = set(row).difference(
            {"variant_id", "p_value", "chromosome", "position"}
        )
        if unknown_variant:
            raise ValueError(
                f"variants[{index}] has unknown fields: "
                + ", ".join(sorted(unknown_variant))
            )
        if not isinstance(row.get("variant_id", ""), str):
            raise ValueError(f"variants[{index}].variant_id must be a string")
        variant_id = row.get("variant_id", "").strip()
        if strict_identifiers and not _VARIANT_ID.fullmatch(variant_id):
            raise ValueError(f"invalid variant identifier: {variant_id!r}")
        p_value = row.get("p_value")
        if isinstance(p_value, bool):
            raise ValueError(f"variants[{index}].p_value must be numeric, not boolean")
        if p_value is not None:
            try:
                precise_p = Decimal(str(p_value))
                if not precise_p.is_finite() or not 0 <= precise_p <= 1:
                    raise ValueError("out of range")
                p_value = float(precise_p)
            except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"variants[{index}].p_value must be between 0 and 1") from exc
        position = row.get("position")
        if isinstance(position, bool) or (
            isinstance(position, float) and not position.is_integer()
        ):
            raise ValueError(f"variants[{index}].position must be an integer")
        if position is not None:
            try:
                position = int(position)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"variants[{index}].position must be an integer") from exc
        variants.append(
            VariantObservation(
                variant_id=variant_id,
                p_value=p_value,
                chromosome=str(row["chromosome"]) if row.get("chromosome") else None,
                position=position,
            )
        )
    candidates_raw = value.get("candidate_genes", [])
    if not isinstance(candidates_raw, list):
        raise ValueError("candidate_genes must be an array")
    if any(not isinstance(gene, str) for gene in candidates_raw):
        raise ValueError("candidate_genes must contain strings")
    candidates = tuple(str(gene).strip() for gene in candidates_raw)
    if strict_identifiers:
        bad = [gene for gene in candidates if not _GENE_ID.fullmatch(gene)]
        if bad:
            raise ValueError("invalid gene identifiers: " + ", ".join(bad))
    tissues_raw = value.get("relevant_tissues", [])
    if not isinstance(tissues_raw, list):
        raise ValueError("relevant_tissues must be an array")
    if any(not isinstance(item, str) or not item.strip() for item in tissues_raw):
        raise ValueError("relevant_tissues must contain non-empty strings")
    task = str(value.get("task", "prioritize_genes"))
    trait = str(value.get("trait", ""))
    source_trait_value = value.get("source_trait")
    if source_trait_value is not None and not isinstance(source_trait_value, str):
        raise ValueError("source_trait must be a string or null")
    # Gene prioritization must define its biological context before evidence
    # projection and model inference.  Explicit caller terms remain authoritative;
    # New nonbenchmark requests may use an authored, versioned physiology plan.
    # Historical null/explicit contexts and benchmark defaults retain their contract.
    # Unknown phenotypes remain unresolved; the live V2G boundary requires clarification.
    biological_context = None
    if value.get("biological_context") is not None:
        biological_context = biological_context_from_dict(value["biological_context"], phenotype=trait)
        relevant_tissues = biological_context.relevant_terms
        if "relevant_tissues" in value and tuple(tissues_raw) != relevant_tissues:
            raise ValueError("relevant_tissues must match biological_context")
    elif "biological_context" in value:
        # Legacy private artifacts explicitly store null. Preserve their exact
        # replay contract; new entrypoints provide a fixed context explicitly.
        relevant_tissues = tuple(str(item) for item in tissues_raw)
    elif task == "prioritize_genes":
        biological_context = (curated_biological_context(trait, reason="new_query_context_default")
            if not tissues_raw and value.get("benchmark") is not True else None)
        biological_context = biological_context or define_biological_context(
            trait,
            explicit_terms=tuple(str(item) for item in tissues_raw),
        )
        relevant_tissues = biological_context.relevant_terms
    else:
        relevant_tissues = tuple(str(item) for item in tissues_raw)
    study_ids_raw = value.get("study_ids", [])
    trait_ids_raw = value.get("trait_ids", [])
    qtl_types_raw = value.get("qtl_types", [])
    requested_evidence_kinds_raw = value.get("requested_evidence_kinds", [])
    if (
        not isinstance(study_ids_raw, list)
        or not isinstance(trait_ids_raw, list)
        or not isinstance(qtl_types_raw, list)
        or not isinstance(requested_evidence_kinds_raw, list)
    ):
        raise ValueError(
            "study_ids, trait_ids, qtl_types, and requested_evidence_kinds must be arrays"
        )
    if any(not isinstance(item, str) for item in (
        *study_ids_raw, *trait_ids_raw, *qtl_types_raw, *requested_evidence_kinds_raw,
    )):
        raise ValueError("study_ids, trait_ids, qtl_types, and requested_evidence_kinds must contain strings")
    benchmark_value = value.get("benchmark", False)
    if not isinstance(benchmark_value, bool):
        raise ValueError("benchmark must be a JSON boolean")
    significance_verified_value = value.get("significance_verified")
    if significance_verified_value is not None and not isinstance(
        significance_verified_value, bool
    ):
        raise ValueError("significance_verified must be a JSON boolean or null")
    query = LocusQuery(
        locus_id=str(value.get("locus_id", "")),
        trait=trait,
        variants=tuple(variants),
        source_trait=(
            source_trait_value.strip()
            if isinstance(source_trait_value, str)
            else None
        ),
        candidate_genes=candidates,
        task=task,
        ancestry=str(value["ancestry"]) if value.get("ancestry") else None,
        relevant_tissues=relevant_tissues,
        biological_context=biological_context.to_dict() if biological_context is not None else None,
        genome_build=str(value["genome_build"]) if value.get("genome_build") else None,
        ld_reference=str(value["ld_reference"]) if value.get("ld_reference") else None,
        benchmark=benchmark_value,
        significance_verified=(
            significance_verified_value
            if significance_verified_value is not None
            else None
        ),
        significance_source=(
            str(value["significance_source"])
            if value.get("significance_source")
            else None
        ),
        study_ids=tuple(str(item) for item in study_ids_raw),
        trait_ids=tuple(str(item) for item in trait_ids_raw),
        qtl_types=tuple(str(item) for item in qtl_types_raw),
        requested_evidence_kinds=tuple(
            str(item) for item in requested_evidence_kinds_raw
        ),
    )
    if allow_analysis_scope and (value.get("analysis_variants") or value.get("ld_expansion") is not None):
        from copy import deepcopy
        from dataclasses import replace

        from .analysis_scope import _rows

        query = replace(query, analysis_variants=_rows(value.get("analysis_variants")),
                        ld_expansion=deepcopy(value.get("ld_expansion")))
    query.validate()
    return query


def leakage_policy_from_dict(value: Any, *, expected_mode: DataMode) -> LeakagePolicy:
    if not isinstance(value, dict):
        raise ValueError("policy must be a JSON object")
    allowed = {
        "mode",
        "release",
        "cutoff_date",
        "allowed_kinds",
        "forbidden_datasets",
        "forbidden_study_ids",
        "forbidden_pmids",
        "forbidden_variant_ids",
        "forbidden_credible_set_ids",
        "forbidden_evidence_ids",
        "denied_payload_fields",
        "require_complete_provenance",
        "require_release_match",
        "block_on_any_denied_record",
    }
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError("unknown policy fields: " + ", ".join(sorted(unknown)))
    mode = DataMode(value.get("mode", expected_mode.value))
    if mode is not expected_mode:
        raise ValueError(f"policy mode {mode.value} does not match {expected_mode.value}")
    cutoff = value.get("cutoff_date")
    kinds = value.get("allowed_kinds")
    kwargs: dict[str, Any] = {
        "mode": mode,
        "release": str(value.get("release", "")),
        "cutoff_date": date.fromisoformat(cutoff) if cutoff else None,
    }
    set_fields = {
        "forbidden_datasets": str,
        "forbidden_study_ids": str,
        "forbidden_pmids": str,
        "forbidden_variant_ids": str,
        "forbidden_credible_set_ids": str,
        "forbidden_evidence_ids": str,
        "denied_payload_fields": str,
    }
    for field_name, converter in set_fields.items():
        if field_name in value:
            raw = value[field_name]
            if not isinstance(raw, list):
                raise ValueError(f"{field_name} must be an array")
            kwargs[field_name] = frozenset(converter(item) for item in raw)
    if kinds is not None:
        if not isinstance(kinds, list):
            raise ValueError("allowed_kinds must be an array")
        kwargs["allowed_kinds"] = frozenset(EvidenceKind(item) for item in kinds)
    for field_name in (
        "require_complete_provenance",
        "require_release_match",
        "block_on_any_denied_record",
    ):
        if field_name in value:
            if not isinstance(value[field_name], bool):
                raise ValueError(f"{field_name} must be boolean")
            kwargs[field_name] = value[field_name]
    policy = LeakagePolicy(**kwargs)
    if not policy.release:
        raise ValueError("policy release is required")
    return policy


def validate_benchmark_policy(policy: LeakagePolicy) -> None:
    if policy.mode is not DataMode.FROZEN_SAFE:
        raise ValueError("benchmark policy must use frozen-safe mode")
    if not policy.forbidden_variant_ids:
        raise ValueError("benchmark policy requires coding-signal/LD blocked variants")
    if not (
        policy.forbidden_study_ids
        or policy.forbidden_credible_set_ids
        or policy.forbidden_evidence_ids
    ):
        raise ValueError(
            "benchmark policy requires source study, credible-set, or evidence embargo IDs"
        )
