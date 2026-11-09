"""Compact final evidence display without changing facts or measurement ownership.

This module is called only after report validation. Source disclosures retain
measurement identities and witnesses without repeating the entire prose line.
The writer and factual checks see unchanged canonical facts. Original records
remain in the evidence archive.
"""
from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping
from typing import Any

from .display_numbers import three_significant_figures

_VARIANT = r"[0-9XYMT]+_\d+_[ACGT]+_[ACGT]+"
_QTL = re.compile(
    rf"^(?P<head>-\s+(?:sc-)?(?:e|s|p|tu|mol)?QTL\s+\(.+?\)):\s*"
    rf"(?P<stats>.*?)\s+\(displayed statistics: credible-set lead variant\s+"
    rf"`(?P<lead>{_VARIANT})`[^)]*\)(?P<tail>.*)$", re.IGNORECASE,
)
_COLOC = re.compile(
    r"^(?P<head>-\s+(?:sc-)?(?:e|s|p|tu|mol)?QTL coloc .+?\(H4=[^)]+\))"
    r"\s+\(H4 study pair:[^)]*\)(?P<tail>.*)$", re.IGNORECASE,
)
_E2G = re.compile(
    rf"^(?P<head>-\s+(?:LD proxy|Source-selected variant|Analyzed variant) "
    rf"`(?P<variant>{_VARIANT})` overlaps the reported regulatory interval "
    r"`chr(?P<chromosome>[0-9XYMT]+):(?P<start>\d+)-(?P<end>\d+)`, "
    r"linked to \*{2,3}(?P<gene>[A-Za-z0-9-]+)\*{2,3})(?P<tail>.*)$",
)
_ORIGIN = re.compile(
    r'\s*\[(?:Indirect LD proxy evidence|Source \+ LD proxy evidence)\]'
    r'\(#gwasmind-evidence-origin "[^"]*"\)',
)
_CITATION = re.compile(r"\[\[\d+\]\]\(https?://[^\s)]+\)|\{\{cite:[^}]+\}\}")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def is_native_gene_report(result: Mapping[str, Any]) -> bool:
    """Recognize native records even when cards live in a separate artifact."""
    audit = result.get("candidate_audit")
    if isinstance(audit, Mapping) and audit.get("mode") == "gene_centric_window":
        return True
    return any(isinstance(row, Mapping) and isinstance(row.get("payload"), Mapping)
               and (row["payload"].get("provenance") or {}).get("pipeline") == "gwasmind_v2g_gene_centric"
               for row in result.get("evidence", ()))


def _source(line: str, extra: str = "") -> str:
    # Quotes and HTML are escaped before entering the Markdown title. No raw
    # provider HTML or script can become an executable disclosure body.
    detail = html.escape(line + ("; " + extra if extra else ""), quote=True)
    return f'[Source](#gwasmind-evidence-source "{detail}")'


def _with_source(head: str, original: str, tail: str, extra: str = "") -> str:
    remainder = _ORIGIN.sub("", tail).strip()
    return head + (" " + remainder if remainder else "") + " " + _source(original, extra)


def source_orientation_presentation(text, result):
    """Display uploaded orientation first; measurement values and signs remain untouched."""
    from .report_facts import public_source_normalizations
    for proof in public_source_normalizations(result):
        original, canonical = proof["original_id"], proof["canonical_id"]
        if original == canonical:
            continue
        # Labels explicitly name both representations, including in disclosures.
        text = "\n".join(line if line.lstrip().startswith("- ") and (
            "credible-set lead" in line or "#gwasmind-evidence-source" in line) else
            re.sub(r"(?<![A-Za-z0-9_])" + re.escape(canonical) + r"(?![A-Za-z0-9_])",
                   lambda _: f"{original} (OT-normalized: {canonical})", line)
            for line in text.split("\n"))
    return text


def retrieval_absence_citations(text, result):
    """A returned-record absence is a retrieval outcome, never a paper's finding.

    Only bibliography markers immediately following explicit retrieval-absence
    wording are replaced. The scientific prose and positive citations stay intact.
    """
    pattern = re.compile(
        r"(\bno\b[^.!?\n]{0,220}\b(?:colocali[sz]ation|(?:molecular\s+)?QTL|eQTL|sQTL|E2G)"
        r"[^.!?\n]{0,100}\b(?:supplied|returned|retained|retrieved|reported))"
        r"((?:\s*\[\[\d+\]\]\(https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?\))+)",
        re.IGNORECASE)
    run_id = str(result.get("run_id") or "this analysis")
    receipt = _source(f"Retrieval receipt: {run_id}. This statement concerns the frozen query scope and retained evidence, "
                      "not a negative experimental finding in a cited publication. Full query completion and retained counts are archived.")
    return pattern.sub(lambda match: match[1] + " " + receipt, text)


def _native_e2g(match: re.Match[str], subject: str, result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    audit = result.get("candidate_audit")
    from .card_store import cards_from_audit
    cards = cards_from_audit(audit)
    gene_id = re.search(r"\bENSG\d+\b", subject)
    if gene_id is None:
        return None
    observations = []
    for card in cards:
        if not isinstance(card, Mapping) or card.get("gene_id") != gene_id.group():
            continue
        identity = card.get("identity", {})
        if not isinstance(identity, Mapping) or identity.get("symbol") != match["gene"]:
            continue
        evidence = card.get("evidence", {})
        rows = evidence.get("e2g", {}).get("records", ()) if isinstance(evidence, Mapping) else ()
        observations.extend(rows)
    if not observations:
        # Native result manifests deliberately do not duplicate the card file.
        # The same typed observations are available in their evidence ledger.
        for evidence in result.get("evidence", ()):
            if not isinstance(evidence, Mapping) or evidence.get("gene_id") != gene_id.group():
                continue
            payload = evidence.get("payload") or {}
            if (payload.get("module") != "locus.enhancer_to_gene"
                    or (payload.get("provenance") or {}).get("pipeline") != "gwasmind_v2g_gene_centric"
                    or (payload.get("object") or {}).get("label") != match["gene"]):
                continue
            context, values = payload.get("context") or {}, payload.get("measurements") or {}
            observations.append({"variant_id": evidence.get("variant_id"),
                "chromosome": context.get("regulatory_chromosome"), "start": context.get("regulatory_start"),
                "end": context.get("regulatory_end"), "exact_variant_interval_overlap": context.get("exact_variant_overlap"),
                "evidence_id": evidence.get("evidence_id"), "study_id": evidence.get("study_id"),
                "biosample": context.get("biosample"), "biosample_id": context.get("biosample_id"),
                "distance_to_tss": context.get("distance_to_tss"), "witnesses": context.get("locus_anchor_witnesses", []),
                "score": values.get("score"), "resource_scores": values.get("resource_scores", [])})
    matched = []
    for row in observations:
        if not isinstance(row, Mapping) or row.get("exact_variant_interval_overlap") is not True:
            continue
        if (str(row.get("chromosome")), str(row.get("start")), str(row.get("end"))) != (
            match["chromosome"], match["start"], match["end"],
        ):
            continue
        variants = {row.get("variant_id")}
        for witness in row.get("witnesses", ()):
            if isinstance(witness, Mapping):
                variants.update(witness.get(key) for key in ("variant_id", "resolved_variant_id", "query_variant_id"))
        if match["variant"] not in variants or not row.get("evidence_id"):
            continue
        resources = {
            str(item.get("name", "")).casefold(): _number(item.get("value"))
            for item in row.get("resource_scores", ()) if isinstance(item, Mapping)
        }
        compatible = True
        for label, names in (("DNase", ("dnase",)), ("HiC", ("hic", "hic_contacts", "hic contacts"))):
            rendered = re.search(rf"\b{label}=([^\s,]+)", match["tail"])
            if rendered:
                expected = next((resources[name] for name in names if resources.get(name) is not None), None)
                if expected is None or _number(rendered[1]) != _number(f"{expected:.4g}"):
                    compatible = False
        distance = re.search(r"TSS=([\d,]+) bp", match["tail"])
        if distance and _number(distance[1].replace(",", "")) != _number(row.get("distance_to_tss")):
            compatible = False
        if compatible:
            matched.append(row)
    return matched[0] if len(matched) == 1 else None


def compact_evidence_fact(markdown: str, *, model_data: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """Reformat only controlled molecular facts from the new gene-centric path."""
    if not is_native_gene_report(result):
        return markdown
    module = model_data.get("evidence_module")
    if module not in {"Molecular QTL/Colocalization", "Regulatory Region-to-Gene (E2G)"}:
        return markdown
    if "\n" in markdown or "#gwasmind-evidence-source" in markdown:
        return markdown
    if module == "Molecular QTL/Colocalization":
        if match := _QTL.fullmatch(markdown):
            head = f"{match['head']}: credible-set lead `{match['lead']}`; {match['stats']}"
            overlap = re.search(r"; locus overlap: (.*?)\)(?:\s|$)", markdown)
            if overlap:
                variants = re.findall(r"(?:user-selected variant|LD-linked variant|analyzed variant) `[^`]+`", overlap[1])
                selected = [value for value in variants if value.startswith("user-selected")]
                head += f"; {len(variants)} overlapping locus variants"
                if selected:
                    head += "; " + ", ".join(selected)
            # Preserve every witness once, grouped by its role. The unabridged
            # canonical fact remains in the archive; repeating its citation and
            # origin tooltip here used to double the report size.
            groups = []
            if overlap:
                for role in ("user-selected variant", "LD-linked variant", "analyzed variant"):
                    identifiers = re.findall(re.escape(role) + r" `([^`]+)`", overlap[1])
                    if identifiers:
                        groups.append(role + "s: " + ", ".join(dict.fromkeys(identifiers)))
            study = re.search(r"; study `([^`]+)`", markdown)
            source = (f"{match['head']}: {match['stats']} (displayed statistics: credible-set lead variant "
                      f"`{match['lead']}`" + (f"; study `{study[1]}`" if study else "") + ")")
            extra = "; ".join([*groups, "Beta: OT lead orientation; no sign conversion; effect allele unreported."])
            return _with_source(head, source, match["tail"], extra)
        if match := _COLOC.fullmatch(markdown):
            original = _ORIGIN.sub("", _CITATION.sub("", markdown)).strip()
            return _with_source(match["head"], original, match["tail"])
    elif match := _E2G.fullmatch(markdown):
        row = _native_e2g(match, str(model_data.get("subject", "")), result)
        if row is None:
            return markdown
        attributes = []
        if row.get("biosample"):
            # Escape display text without allowing provider strings to add links.
            biosample = html.escape(str(row["biosample"])).replace("[", "\\[").replace("]", "\\]")
            attributes.append(biosample)
        score = _number(row.get("score"))
        if score is not None and 0 <= score <= 1:
            attributes.append("E2G score=" + three_significant_figures(row["score"]))
        if not attributes:
            return markdown
        extra = "; ".join(f"{label}: {row[key]}" for key, label in (
            ("evidence_id", "Evidence record"), ("study_id", "Study"),
            ("biosample_id", "Biosample ID"), ("biosample", "Biosample"), ("score", "Raw E2G score"),
        ) if row.get(key) is not None)
        raw_resources = "; ".join(f"{item['name']} as reported: {item['value']}"
            for item in row.get("resource_scores", ()) if isinstance(item, Mapping)
            and item.get("name") and _number(item.get("value")) is not None)
        if raw_resources:
            extra += "; " + raw_resources
        citations = " ".join(_CITATION.findall(match["tail"]))
        concise = f"- `{match['variant']}`; " + "; ".join(attributes)
        origin = "LD-linked variant" if "LD proxy" in match["head"] else "User-selected variant" if "Source-selected" in match["head"] else "Analyzed variant"
        source = (f"{origin}: {match['variant']}; GRCh38 interval: chr{match['chromosome']}:{match['start']}-{match['end']}; "
                  f"Gene: {match['gene']}; distance to TSS: {row.get('distance_to_tss')} bp")
        return _with_source(concise, source, citations, extra)
    return markdown
