"""Gate 2 hallucination control for a parsed model ranking.

Citations that were not shown to the model or belong to another gene are
removed, claims left without any citation are dropped, and sentences naming a
variant identifier that occurs nowhere in the frozen query or evidence are
removed. Gene order, candidate decisions and every other sentence are kept as
written. The removals are recorded for audit; they are not user warnings.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from typing import Any

from .models import EvidenceRecord, LocusQuery

_VARIANT = re.compile(r"(?<![\w-])(?:chr)?([0-9]{1,2}|X|Y|MT|M)[_:]([0-9]{1,10})[_:]([ACGTN]+)[_:]([ACGTN]+)(?![\w-])",
                      re.IGNORECASE)
_RSID = re.compile(r"(?<![\w-])rs[0-9]{2,12}(?![\w-])", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_PLACEHOLDER = "[identifier removed]"

# Marks the span of an unverifiable value until every edit of a field is applied.
REMOVED_VALUE = "\ue000"
REMOVED_VALUE_LABEL = "Removed from the report"
_CONNECTOR_BEFORE_REMOVED = re.compile(
    r"\s*\b(?:with|at|of|and|or|by)\s+" + REMOVED_VALUE + r"(?=\s*(?:[,.;:)]|$))", re.IGNORECASE)


def invalid_probability_sentence_edits(text, findings):
    """Remove only a simple affirmative assertion of an impossible probability.

    Unbound but numerically admissible measurements, thresholds, negatives,
    hypotheses, examples, compound claims and ambiguous boundaries are kept.
    A bare number beside a panel label is not a scientific measurement.
    """
    from decimal import Decimal, InvalidOperation

    typed = {'p': r'\b(?:GWAS|QTL|association|p[- ]value)\b',
             'h4': r'\b(?:coloc|colocali[sz]ation|posterior probability)\b',
             'h3': r'\b(?:coloc|colocali[sz]ation|posterior probability)\b',
             'pip': r'\b(?:fine[- ]mapping|posterior inclusion probability)\b',
             'clpp': r'\b(?:coloc|colocali[sz]ation|posterior probability)\b',
             'r2': r'\b(?:LD|linkage disequilibrium)\b'}
    edits = {}
    for finding in findings:
        metric = finding.get('metric')
        if metric not in typed:
            continue
        try:
            value = Decimal(str(finding.get('original_value')))
        except (InvalidOperation, ValueError):
            continue
        if value.is_finite() and Decimal(0) <= value <= Decimal(1):
            continue
        span = finding.get('span') or (finding.get('start'), finding.get('end'))
        if len(span) != 2 or any(type(value) is not int for value in span):
            continue
        sentence = next((item for item in re.finditer(r'.+?(?:(?<=[.!?])(?=\s)|(?=\n)|$)', text, re.DOTALL)
                         if item.start() <= span[0] < span[1] <= item.end()), None)
        if sentence is None:
            continue
        plain = re.sub(r'[*`]', '', sentence.group())
        prefix = re.sub(r'[*`]', '', text[sentence.start():span[0]])
        if (not re.search(typed[metric], prefix, re.I)
                or not re.search(r'\b(?:has|shows|reports|observed|estimated|measured|yields)\b', prefix, re.I)
                or re.search(r'[;\n]|\b(?:not|no|never|without|if|whether|example|hypothetical|would|could|'
                             r'threshold|cutoff|cut-off|and|but|or|whereas)\b', plain, re.I)):
            continue
        # Only one labelled numeric assertion can be deleted as a unit.
        from .ranking_claims import _METRIC
        if len(list(_METRIC.finditer(plain))) != 1:
            continue
        finding.update(removed=True, display_replacement=REMOVED_VALUE_LABEL,
                       replacement='', reason='numeric_domain_error', removed_text=sentence.group())
        edits[sentence.span()] = {'start': sentence.start(), 'end': sentence.end(), 'replacement': ''}
    return list(edits.values())


def _tidy_line(line: str) -> str:
    mark = REMOVED_VALUE
    line = re.sub(r"[(\[]\s*" + mark + r"\s*[)\]]", "", line)
    line = re.sub(r"\(\s*" + mark + r"\s*[;,]\s*", "(", line)
    line = re.sub(r"\s*[;,]\s*" + mark + r"\s*\)", ")", line)
    line = re.sub(r"[;,]\s*" + mark + r"(?=\s*[;,.)])", "", line)
    line = _CONNECTOR_BEFORE_REMOVED.sub("", line)
    line = line.replace(mark, "")
    line = re.sub(r"\(\s*\)", "", line)
    line = re.sub(r"[ \t]*[,;][ \t]*(?=[,;.])", "", line)
    line = re.sub(r"^(\s*(?:[-*+]|\d+\.)?[ \t]*)[,;][ \t]*", r"\1", line)
    line = re.sub(r"(?<=\S)[ \t]{2,}", " ", line)
    line = re.sub(r"[ \t]+([,.;:)])", r"\1", line)
    line = re.sub(r"\([ \t]+", "(", line)
    return line.rstrip() if re.search(r"\w", line) else ""


def tidy_removed_values(text: str) -> str:
    """Drop removal marks and repair the punctuation of the lines they touched."""
    if REMOVED_VALUE not in text:
        return text
    return "\n".join(_tidy_line(line) if REMOVED_VALUE in line else line for line in text.split("\n"))


_MARK = r"[*`]*"
_VARIANT_TOKEN = r"(?:chr)?(?:[0-9]{1,2}|X|Y|MT|M)[_:][0-9]{1,10}[_:][ACGTN]+[_:][ACGTN]+"
_RSID_TOKEN = r"rs[0-9]{2,12}"
# Only explicit pairings are checked: "VARIANT (rsID)", "rsID (VARIANT)" and "VARIANT/rsID".
_PAIRINGS = (
    ("variant_then_rsid", re.compile(
        r"(?<![\w-])(?P<variant>" + _VARIANT_TOKEN + r")" + _MARK + r"(?P<drop>\s*\(\s*" + _MARK
        + r"(?P<rsid>" + _RSID_TOKEN + r")" + _MARK + r"(?:\s*\)|\s*[;,]\s*))", re.IGNORECASE)),
    ("rsid_then_variant", re.compile(
        r"(?<![\w-])(?P<drop>" + _MARK + r"(?P<rsid>" + _RSID_TOKEN + r")" + _MARK + r"\s*\(\s*)" + _MARK
        + r"(?P<variant>" + _VARIANT_TOKEN + r")" + _MARK + r"(?P<close>\s*\))", re.IGNORECASE)),
    ("variant_slash_rsid", re.compile(
        r"(?<![\w-])(?P<variant>" + _VARIANT_TOKEN + r")" + _MARK + r"(?P<drop>\s*/\s*" + _MARK
        + r"(?P<rsid>" + _RSID_TOKEN + r")" + _MARK + r")(?![\w-])", re.IGNORECASE)),
    ("rsid_slash_variant", re.compile(
        r"(?<![\w-])(?P<drop>" + _MARK + r"(?P<rsid>" + _RSID_TOKEN + r")" + _MARK + r"\s*/\s*)" + _MARK
        + r"(?P<variant>" + _VARIANT_TOKEN + r")(?![\w-])", re.IGNORECASE)),
)


def rsid_locations(ld_expansion: Any) -> dict[str, set[tuple[str, int, int]]]:
    """GRCh38 positions of rsIDs from the validated identity registry and LDlink rows."""
    from .variant_identity_registry import validate_variant_identity_registry

    plan = ld_expansion if isinstance(ld_expansion, dict) else {}
    locations: dict[str, set[tuple[str, int, int]]] = {}

    def add(rsid: Any, chromosome: Any, start: Any, end: Any = None) -> None:
        if not isinstance(rsid, str) or not re.fullmatch(_RSID_TOKEN, rsid, re.IGNORECASE):
            return
        try:
            start, end = int(start), int(end if end is not None else start)
        except (TypeError, ValueError):
            return
        chromosome = str(chromosome or "").upper().removeprefix("CHR")
        if chromosome and start > 0 and end >= start:
            locations.setdefault(rsid.casefold(), set()).add((chromosome, start, end))

    registry = plan.get("variant_identity_registry")
    if validate_variant_identity_registry(registry):
        for identifier, row in registry["variants"].items():
            chromosome, position = identifier.split("_")[:2]
            for rsid in row["rsids"]:
                add(rsid, chromosome, position)
    for link in (*(plan.get("ld_links") or ()), *(plan.get("enumerated_ld_proxies") or ())):
        if not isinstance(link, dict) or link.get("genome_build") != "GRCh38" or link.get("status") != "computed":
            continue
        add(link.get("reported_rsid"), link.get("chromosome"), link.get("position"), link.get("end"))
        selected = re.fullmatch(_VARIANT_TOKEN, str(link.get("selected_variant_id") or ""), re.IGNORECASE)
        if selected:
            chromosome, position = re.split(r"[_:]", selected.group())[:2]
            add(link.get("selected_rsid"), chromosome.removeprefix("chr"), position)
    return locations


def _rsid_conflict(variant: str, rsid: str, locations: dict[str, set[tuple[str, int, int]]]) -> list[list[Any]] | None:
    """Recorded positions that place ``rsid`` away from ``variant``; None when not proven."""
    found = locations.get(rsid.casefold())
    if not found:
        return None
    chromosome, position = re.split(r"[_:]", variant.upper().removeprefix("CHR"))[:2]
    position = int(position)
    if any(chrom == chromosome and start - 1 <= position <= end + 1 for chrom, start, end in found):
        return None
    return [list(item) for item in sorted(found)]


def remove_mismatched_rsid_labels(text: str, field: str, locations: dict[str, set[tuple[str, int, int]]],
                                  removals: list[dict[str, Any]]) -> str:
    """Delete an rsID written beside a variant it provably does not name; keep the variant and the sentence."""
    if not locations or "rs" not in text.casefold():
        return text
    for form, pattern in _PAIRINGS:
        edits = []
        for match in pattern.finditer(text):
            evidence = _rsid_conflict(match["variant"], match["rsid"], locations)
            if evidence is None:
                continue
            drop_start, drop_end = match.span("drop")
            replacement = ""
            if form == "variant_then_rsid" and not text[drop_start:drop_end].rstrip().endswith(")"):
                replacement = " ("
            if form == "rsid_then_variant":
                edits.append((match.start("close"), match.end("close"), ""))
            edits.append((drop_start, drop_end, replacement))
            removals.append({"field": field, "kind": "rsid_label", "reason": "rsid_recorded_at_another_position",
                             "variant_id": match["variant"], "rsid": match["rsid"], "rsid_locations": evidence,
                             "removed_text": match.group()})
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
    return text


def remove_report_rsid_mismatches(content: str, ld_expansion: Any) -> tuple[str, list[dict[str, Any]]]:
    """Apply the rsID pairing check to the two rendered report fields."""
    locations = rsid_locations(ld_expansion)
    if not locations:
        return content, []
    value = json.loads(content)
    removals: list[dict[str, Any]] = []
    limitations = value.get("limitations_uncertainty")
    for field, owner, key in (("summary_markdown", value, "summary_markdown"),
                              ("limitations_uncertainty.markdown", limitations, "markdown")):
        if isinstance(owner, dict) and isinstance(owner.get(key), str):
            owner[key] = remove_mismatched_rsid_labels(owner[key], field, locations, removals)
    if not removals:
        return content, []
    return json.dumps(value, ensure_ascii=True, indent=2), removals


def remove_ranking_rsid_mismatches(query: LocusQuery, output):
    """Apply the rsID pairing check to every ranking prose field; order and decisions are unchanged."""
    locations = rsid_locations(query.ld_expansion)
    if not locations:
        return output
    removals: list[dict[str, Any]] = []

    def clean(text: str, field: str) -> str:
        return remove_mismatched_rsid_labels(text, field, locations, removals)

    ranked = tuple(replace(row, rationale=clean(row.rationale, f"ranked_genes[{index}].rationale"),
                           limitations=tuple(clean(item, f"ranked_genes[{index}].limitations[{position}]")
                                             for position, item in enumerate(row.limitations)))
                   for index, row in enumerate(output.ranked_genes))
    decisions = tuple(replace(row, reason=clean(row.reason, f"candidate_decisions[{index}].reason"))
                      for index, row in enumerate(output.candidate_decisions))
    claims = tuple(replace(row, text=clean(row.text, f"claims[{index}].text")) for index, row in enumerate(output.claims))
    summary = clean(output.summary, "summary")
    if not removals:
        return output
    usage = {**output.usage, "hallucination_removals": [*output.usage.get("hallucination_removals", []), *removals]}
    return replace(output, ranked_genes=ranked, candidate_decisions=decisions, claims=claims, summary=summary, usage=usage)


def _variant_forms(match: re.Match) -> tuple[str, str]:
    chromosome, position, reference, alternate = match.groups()
    chromosome = chromosome.upper().removeprefix("CHR")
    return (f"{chromosome}_{position}_{reference.upper()}_{alternate.upper()}",
            f"{chromosome}_{position}_{alternate.upper()}_{reference.upper()}")


def _known_identifiers(query: LocusQuery, records: tuple[EvidenceRecord, ...]) -> tuple[set[str], set[str]]:
    text = json.dumps([asdict(query), [record.to_dict() for record in records]], default=str, ensure_ascii=False)
    variants = {form for match in _VARIANT.finditer(text) for form in _variant_forms(match)}
    rsids = {match.group().casefold() for match in _RSID.finditer(text)}
    return variants, rsids


def _fabricated(text: str, variants: set[str], rsids: set[str]) -> list[str]:
    found = [match.group() for match in _VARIANT.finditer(text)
             if not set(_variant_forms(match)) & variants]
    found += [match.group() for match in _RSID.finditer(text) if match.group().casefold() not in rsids]
    return found


def _clean_text(text: str, field: str, variants: set[str], rsids: set[str], removals: list[dict[str, Any]]) -> str:
    if not _fabricated(text, variants, rsids):
        return text
    sentences = _SENTENCE_BREAK.split(text)
    kept = [sentence for sentence in sentences if not _fabricated(sentence, variants, rsids)]
    if kept:
        for sentence in sentences:
            if sentence not in kept:
                removals.append({"field": field, "kind": "sentence", "reason": "identifier_absent_from_input",
                                 "identifiers": _fabricated(sentence, variants, rsids), "removed_text": sentence})
        return " ".join(kept)
    # Every sentence names an absent identifier: keep the reasoning, drop the identifiers.
    cleaned = text
    for identifier in dict.fromkeys(_fabricated(text, variants, rsids)):
        cleaned = re.sub(r"(?<![\w-])" + re.escape(identifier) + r"(?![\w-])", _PLACEHOLDER, cleaned)
        removals.append({"field": field, "kind": "identifier", "reason": "identifier_absent_from_input",
                         "identifiers": [identifier], "removed_text": identifier})
    return cleaned


def _owned(ids: tuple[str, ...], gene_id: str | None, field: str, owners: dict[str, str | None],
           visible: set[str], removals: list[dict[str, Any]]) -> tuple[str, ...]:
    kept = []
    for identifier in ids:
        owner = owners.get(identifier, "")
        if identifier not in visible or owner == "":
            reason = "citation_not_in_model_input"
        elif gene_id is not None and owner is not None and owner != gene_id:
            reason = "citation_owned_by_another_gene"
        else:
            kept.append(identifier)
            continue
        removals.append({"field": field, "kind": "citation", "reason": reason, "removed_text": identifier})
    return tuple(kept)


def _clean_inline_citations(text, field, owners, visible, gene_id, removals):
    """Remove invalid IDs only from explicit evidence/citation marker syntax."""
    from .evidence_citations import _LABELED_EVIDENCE, _BARE_LABELED_EVIDENCE, _INLINE_TOKENS

    edits, seen = [], set()
    for pattern in (_LABELED_EVIDENCE, _BARE_LABELED_EVIDENCE):
        for match in pattern.finditer(text):
            if match.span() in seen:
                continue
            seen.add(match.span())
            group = 1 if match.group(1) is not None else 2
            contents = match.group(group)
            token_edits = []
            for token in _INLINE_TOKENS.finditer(contents):
                identifier = token.group().rstrip('.:')
                # Bibliographic references have a separate authority and are
                # never deleted as unknown molecular-evidence identifiers.
                if re.match(r'(?:PMID:|PMC\d|doi:|https?:|ref(?:erence)?[_-]?\d)', identifier, re.I):
                    continue
                shaped = (identifier in owners or re.fullmatch(r'E\d+', identifier, re.I)
                          or any(char in identifier for char in ':_/-@'))
                if not shaped:
                    continue
                if identifier not in owners or identifier not in visible:
                    reason = 'citation_not_in_model_input'
                elif gene_id is not None and owners[identifier] not in {None, gene_id}:
                    reason = 'citation_owned_by_another_gene'
                else:
                    continue
                start, end = token.span()
                if start and end < len(contents) and contents[start - 1] == contents[end] == '`':
                    start, end = start - 1, end + 1
                token_edits.append((start, end))
                removals.append({'field': field, 'kind': 'inline_citation', 'reason': reason,
                                 'removed_text': identifier})
            if not token_edits:
                continue
            for start, end in reversed(token_edits):
                contents = contents[:start] + contents[end:]
            contents = re.sub(r'([,;])(?:\s*[,;])+', r'\1', contents).strip(' \t,;')
            if not contents.strip('` '):
                start, end = match.span()
                if start and end < len(text) and text[start - 1] == '(' and text[end] == ')':
                    start, end = start - 1, end + 1
                    if start and text[start - 1] == ' ':
                        start -= 1
                edits.append((start, end, ''))
            else:
                edits.append((*match.span(group), contents))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _removable_report_identifier(finding: dict[str, Any]) -> bool:
    # All-letter symbols may be ordinary acronyms; only identifier-shaped
    # tokens are treated as fabricated and removed.
    value = str(finding.get("original_value") or "")
    return finding.get("metric") == "variant_id" or (
        finding.get("metric") == "gene_id" and any(character.isdigit() for character in value))


def remove_report_identifier_hallucinations(content: str, findings: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Remove report sentences that assert identifiers absent from the evidence."""
    removable = [finding for finding in findings if _removable_report_identifier(finding)]
    if not removable:
        return content, []
    value = json.loads(content)
    limitations = value.get("limitations_uncertainty")
    fields = {"summary_markdown": (value, "summary_markdown"),
              "limitations_uncertainty.markdown": (limitations, "markdown") if isinstance(limitations, dict) else None}
    removals: list[dict[str, Any]] = []
    for field, location in fields.items():
        if location is None or not isinstance(location[0].get(location[1]), str):
            continue
        text = location[0][location[1]]
        identifiers = {finding["original_value"] for finding in removable if finding.get("field") == field}
        if not identifiers:
            continue
        pattern = re.compile("|".join(r"(?<![\w-])" + re.escape(item) + r"(?![\w-])"
                                      for item in sorted(identifiers, key=len, reverse=True)))
        paragraphs, field_removals = [], []
        for paragraph in text.split("\n\n"):
            sentences = _SENTENCE_BREAK.split(paragraph)
            kept = [sentence for sentence in sentences if not pattern.search(sentence)]
            field_removals.extend({"field": field, "kind": "sentence", "reason": "identifier_absent_from_evidence",
                                   "identifiers": sorted(set(pattern.findall(sentence))), "removed_text": sentence}
                                  for sentence in sentences if sentence not in kept)
            paragraphs.append(" ".join(kept))
        cleaned = "\n\n".join(item for item in paragraphs if item.strip())
        if not cleaned.strip():
            # Every sentence names an absent identifier: keep the reasoning, drop the identifiers.
            cleaned = pattern.sub(_PLACEHOLDER, text)
            field_removals = [{"field": field, "kind": "identifier", "reason": "identifier_absent_from_evidence",
                               "identifiers": sorted(set(pattern.findall(text))), "removed_text": text}]
        removals.extend(field_removals)
        location[0][location[1]] = cleaned
    if not removals:
        return content, []
    return json.dumps(value, ensure_ascii=True, indent=2), removals


def remove_ranking_hallucinations(query: LocusQuery, records: tuple[EvidenceRecord, ...], output):
    """Return the ranking with unsupported citations and identifiers removed."""
    owners = {record.evidence_id: record.gene_id for record in records}
    visible = set(owners) if output.model_visible_evidence_ids is None else set(output.model_visible_evidence_ids)
    variants, rsids = _known_identifiers(query, records)
    removals: list[dict[str, Any]] = []
    from .models import shared_ranks
    from .ranking_claims import _gene_aliases
    from .ranking_order import tied_order_sentence_edits
    ranks = {gene: shared for gene, (shared, _tied) in shared_ranks(output.ranked_genes).items()}
    aliases = _gene_aliases(records)
    inline_owners, inline_visible = dict(owners), set(visible)
    evidence_aliases = output.model_evidence_aliases
    if evidence_aliases is None and any(re.search(r'\bE\d+\b', text) for text in (
            output.summary, *(row.rationale for row in output.ranked_genes),
            *(value for row in output.ranked_genes for value in row.limitations),
            *(row.reason for row in output.candidate_decisions), *(row.text for row in output.claims))):
        # Legacy outputs do not save a presentation manifest. Reuse the same
        # bounded compatibility mapping as the validator, not a new ordering.
        from .ranking_claims import _assertion_context
        evidence_aliases = _assertion_context(query, records, output).citation_aliases
    for alias, canonical in (evidence_aliases or {}).items():
        if re.fullmatch(r'E[1-9][0-9]*', alias) and canonical in owners:
            inline_owners[alias] = owners[canonical]
            if canonical in visible:
                inline_visible.add(alias)

    def clean(text, field, owner=None):
        cleaned = _clean_inline_citations(text, field, inline_owners, inline_visible, owner, removals)
        cleaned = _clean_text(cleaned, field, variants, rsids, removals)
        edits = tied_order_sentence_edits(cleaned, ranks=ranks, aliases=aliases)
        for edit in reversed(edits):
            removals.append({'field': field, 'kind': 'sentence', **edit})
            cleaned = cleaned[:edit['start']] + edit['replacement'] + cleaned[edit['end']:]
        return cleaned.strip() if edits else cleaned

    ranked = []
    for index, row in enumerate(output.ranked_genes):
        prefix = f"ranked_genes[{index}]"
        ranked.append(replace(row,
            evidence_ids=_owned(row.evidence_ids, row.gene_id, prefix + ".evidence_ids", owners, visible, removals),
            rationale=clean(row.rationale, prefix + ".rationale", row.gene_id),
            limitations=tuple(clean(item, f"{prefix}.limitations[{position}]", row.gene_id)
                              for position, item in enumerate(row.limitations))))
    decisions = []
    for index, row in enumerate(output.candidate_decisions):
        prefix = f"candidate_decisions[{index}]"
        decisions.append(replace(row,
            evidence_ids=_owned(row.evidence_ids, row.gene_id, prefix + ".evidence_ids", owners, visible, removals),
            reason=clean(row.reason, prefix + ".reason", row.gene_id)))
    claims = []
    for index, row in enumerate(output.claims):
        prefix = f"claims[{index}]"
        cited = _owned(row.evidence_ids, None, prefix + ".evidence_ids", owners, visible, removals)
        if not cited:
            removals.append({"field": prefix, "kind": "claim", "reason": "claim_without_supplied_citation",
                             "removed_text": row.text})
            continue
        claims.append(replace(row, evidence_ids=cited,
                              text=clean(row.text, prefix + ".text")))
    summary = clean(output.summary, "summary")
    if not removals:
        return output
    usage = {**output.usage, "hallucination_removals": [*output.usage.get("hallucination_removals", []), *removals]}
    return replace(output, ranked_genes=tuple(ranked), candidate_decisions=tuple(decisions),
                   claims=tuple(claims), summary=summary, usage=usage)
