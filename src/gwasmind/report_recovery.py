"""Bounded report repair over immutable fact registries.

This module never changes evidence or guesses the intended scientific association.
Bookkeeping can be derived uniquely; unresolved factual fragments are omitted
locally, followed by validation of the complete assembled report. Scientific
narrative is never returned to a model merely to change its wording.
"""
from __future__ import annotations

import json
import re
import hashlib
from typing import TYPE_CHECKING

from .configuration import parameter

if TYPE_CHECKING:
    from .staged_reasoning import HybridFinalContext, HybridFinalDraft


def normalize_nonrendered_bookkeeping(content: str) -> tuple[str, list[dict]]:
    """Discard only known non-rendered extras from an already complete draft.

    These model-supplied aliases/warnings never supply scientific authority and
    are not rendered. Required fields, reference selections and unknown extras
    remain subject to the full contract; original model bytes are kept upstream.
    """
    from .staged_reasoning import StructuredOutputError, _strict_json_object
    try:
        value = _strict_json_object(content, label='non-rendered bookkeeping')
    except StructuredOutputError:
        return content, []
    limits = value.get('limitations_uncertainty')
    if not isinstance(limits, dict):
        return content, []
    discarded = []
    for name, expected in (('source_variant_aliases', dict), ('warnings', list)):
        if name not in limits or not isinstance(limits[name], expected):
            continue
        original = limits.pop(name)
        encoded = json.dumps(original, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
        discarded.append({'path': 'limitations_uncertainty.' + name,
            'reason': 'nonrendered_model_bookkeeping_not_scientific_authority',
            'value_sha256': hashlib.sha256(encoded.encode('utf-8')).hexdigest()})
    return (json.dumps(value, ensure_ascii=True) if discarded else content), discarded


def _payload(content: str) -> dict | None:
    from .staged_reasoning import StructuredOutputError, _strict_json_object
    try:
        value = _strict_json_object(content, label="report recovery")
    except StructuredOutputError:
        return None
    # Unknown substantive fields are not silently laundered into a report.
    if set(value).difference({"summary_markdown", "evidence_sections", "limitations_uncertainty", "reference_ids"}):
        return None
    if not isinstance(value.get("summary_markdown"), str):
        return None
    limitations = value.get("limitations_uncertainty")
    if not isinstance(limitations, dict) or not isinstance(limitations.get("markdown"), str):
        return None
    if set(limitations).difference({"markdown", "safety_check_ids"}):
        return None
    if len(value["summary_markdown"]) > parameter("gates.report_markdown_max_chars") or len(limitations["markdown"]) > parameter("gates.report_markdown_max_chars"):
        return None
    rows = value.get("evidence_sections")
    if not isinstance(rows, list) or any(not isinstance(row, dict) or set(row).difference({"section_id", "fact_ids"}) for row in rows):
        return None
    return value


def _fact_selections(context: HybridFinalContext) -> list[dict]:
    return [{"section_id": section.section_id, "fact_ids": ids}
            for section in context.sections
            if (ids := [fact.fact_id for fact in context.facts if fact.section_id == section.section_id])]


def repair_report_bookkeeping(content: str, context: HybridFinalContext) -> HybridFinalDraft | None:
    """Repair only uniquely derived registry selections; validate all prose again."""
    from .staged_reasoning import StructuredOutputError, parse_hybrid_final_output
    value = _payload(content)
    if value is None:
        return None
    value.pop("reference_ids", None)
    value["limitations_uncertainty"].pop("safety_check_ids", None)
    value["evidence_sections"] = _fact_selections(context)
    try:
        return parse_hybrid_final_output(json.dumps(value), context)
    except StructuredOutputError:
        return None


def _verified_fragments(text: str, context: HybridFinalContext, label: str) -> tuple[list[str], int]:
    from .staged_reasoning import StructuredOutputError, validate_report_fragment
    kept: list[str] = []
    omitted = 0
    attempts_left = parameter("gates.recovery_fragment_attempts")
    # Keep citations with their preceding claim and never split numeric decimals.
    paragraphs = re.split(r"\n\s*\n|\n(?=\s*[-*] )", text)
    for paragraph in paragraphs[:parameter("gates.recovery_split_limit")]:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if attempts_left <= 0:
            omitted += 1
            continue
        attempts_left -= 1
        try:
            validate_report_fragment(paragraph, context, label=label)
        except StructuredOutputError:
            sentences = re.split(r"(?<=[.!?])\s+(?!\{\{cite_fact:)", paragraph)
            for sentence in sentences[:parameter("gates.recovery_split_limit")]:
                if not sentence.strip():
                    continue
                if attempts_left <= 0:
                    omitted += 1
                    continue
                attempts_left -= 1
                try:
                    validate_report_fragment(sentence, context, label=label)
                except StructuredOutputError:
                    omitted += 1
                else:
                    kept.append(sentence.strip())
        else:
            kept.append(paragraph)
    return kept, omitted


def recover_report_fragments(content: str, context: HybridFinalContext) -> HybridFinalDraft | None:
    """Retain independent valid prose without requesting another scientific draft."""
    from .staged_reasoning import (
        StructuredOutputError,
        _source_gwas_status_sentence,
        parse_hybrid_final_output,
    )
    value = _payload(content)
    if value is None:
        return None
    summary, summary_omitted = _verified_fragments(value["summary_markdown"], context, "summary_markdown")
    limitations, limitations_omitted = _verified_fragments(
        value["limitations_uncertainty"]["markdown"], context, "limitations_uncertainty.markdown",
    )
    if not summary or not (summary_omitted + limitations_omitted):
        return None
    # Source observations remain authoritative even if the model's opening failed.
    source_opening = _source_gwas_status_sentence(context) if context.source_observations else ""
    candidates = ["\n\n".join(summary)]
    if source_opening:
        candidates.append(source_opening + "\n\n" + candidates[0])
    if not limitations:
        limitations = ["Interpretation is limited to the verified evidence shown here."]
    for check in context.safety_review.facts.checks:
        if check.nonpassing and check.fact not in limitations:
            limitations.append(check.fact)
    for summary_text in candidates:
        candidate = {
            "summary_markdown": summary_text,
            "evidence_sections": _fact_selections(context),
            "limitations_uncertainty": {"markdown": "\n\n".join(limitations)},
        }
        try:
            return parse_hybrid_final_output(json.dumps(candidate), context)
        except StructuredOutputError:
            continue
    return None
