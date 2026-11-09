"""Check explicit identifiers in final prose against the saved evidence context."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .staged_reasoning import HybridFinalContext

_ENSEMBL = re.compile(r"\bENSG\d{6,}(?:\.\d+)?\b", re.I)
_VARIANT = re.compile(r"\b(?:chr)?(?:\d{1,2}|X|Y|MT)[_:]\d+[_:][ACGTN-]+[_:][ACGTN-]+\b", re.I)
_RSID = re.compile(r"\brs\d+\b", re.I)
_SUBJECT = re.compile(
    r"\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*)\s+"
    r"(?i:has|shows|cures|regulates|controls|encodes|mediates|affects|increases|decreases|drives)\b"
)
_PROXY_EXCLUSION = re.compile(
    r'(?:^|(?<=[.!?])\s+)\s*(?:(?:The\s+)?LD\s+expansion\s+was\s+incomplete,\s+and\s+|The\s+)?'
    r'\b(?:unverified|unresolved)\s+(?:in[- ]scope\s+)?(?:LD\s+)?prox(?:y|ies)\s*'
    r'(?:\(\s*(?:e\.g\.,?\s*)?)?'
    r'rs\d+(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)rs\d+)*\s*\)?\s+'
    r'(?:was|were)\s+excluded'
    r'(?:\s+from\s+(?:the\s+)?(?:admitted\s+)?(?:proxy|analysis|LD)\s+scope)?'
    r'(?=\s*(?:[.!?](?:\s|$)|$|\n))', re.I | re.M,
)


def _variant_key(value: str) -> str:
    return value.casefold().removeprefix("chr").replace(":", "_")


def _unverified_proxy_authority(context):
    """Separate recorded exclusion metadata from verified variant identities."""
    expansion = context.result.get('ld_expansion')
    expansion = expansion if isinstance(expansion, Mapping) else {}
    rows = expansion.get('unresolved_proxies')
    unresolved, recovered = set(), set()
    for row in rows if isinstance(rows, (list, tuple)) else ():
        if not isinstance(row, Mapping) or not row.get('reason_code'):
            continue
        rsid = row.get('rsid')
        if isinstance(rsid, str) and _RSID.fullmatch(rsid):
            if (row.get('resolved') is True or row.get('recovered') is True
                    or row.get('status') in {'resolved', 'recovered', 'verified'}):
                recovered.add(rsid.casefold())
            else:
                unresolved.add(rsid.casefold())
    # Conflicting admitted identities veto an exclusion claim. These receipts
    # are not added to general identifier authority by this metadata exception.
    for key, rsid_key, variant_key in (('ld_links', 'reported_rsid', 'proxy_variant_id'),
                                       ('proxy_identity_checks', 'rsid', 'variant_id')):
        entries = expansion.get(key)
        for row in entries if isinstance(entries, (list, tuple)) else ():
            if (isinstance(row, Mapping) and isinstance(row.get(rsid_key), str)
                    and _RSID.fullmatch(row[rsid_key]) and isinstance(row.get(variant_key), str)
                    and _VARIANT.fullmatch(row[variant_key])):
                recovered.add(row[rsid_key].casefold())
    disclosed = set()
    for check in context.safety_review.facts.checks:
        if check.check_id != 'ld_proxy_coverage' or 'ld_proxy_identity_unresolved' not in check.reason_codes:
            continue
        for identifier in check.affected.get('variant_ids', ()):
            if isinstance(identifier, str) and _RSID.fullmatch(identifier):
                disclosed.add(identifier.casefold())
    return unresolved.difference(recovered).intersection(disclosed)


def report_identifier_findings(content: str, context: HybridFinalContext) -> list[dict]:
    """Return bounded findings for unsupported identifiers still asserted.

    A gene may be part of a mechanism without being a shortlisted candidate.
    Saved gene-function and pathway evidence therefore also supply identifier
    authority. Model prose and internal ranking rationale never supply it.
    """
    from .staged_reasoning import (
        _GENERIC_UPPERCASE_TOKENS, _explicit_gene_symbols, _reported_gene_aliases,
        _validated_ranked_gene_symbols, _normalize_phenotype_spellings,
    )

    if not isinstance(content, str) or len(content) > 2_000_000:
        return []
    try:
        payload = json.loads(content)
    except (ValueError, RecursionError):
        return []
    if not isinstance(payload, dict):
        return []
    fields = {"summary_markdown": payload.get("summary_markdown")}
    limits = payload.get("limitations_uncertainty")
    fields["limitations_uncertainty.markdown"] = limits.get("markdown") if isinstance(limits, dict) else None
    aliases = _reported_gene_aliases(context)
    ranked = _validated_ranked_gene_symbols(context)
    authority = "\n".join((
        context.deterministic_answer,
        context.source_trait,
        json.dumps(context.source_observations, ensure_ascii=True),
        json.dumps(context.biological_context, ensure_ascii=True),
        json.dumps(context.result.get("variant_resolution", {}), ensure_ascii=True),
        json.dumps(aliases, ensure_ascii=True),
        json.dumps(ranked, ensure_ascii=True),
        *(item.markdown for item in context.facts),
        *(json.dumps(item.model_data, ensure_ascii=True) for item in context.facts),
    ))
    known_gene_ids = {match.group().casefold().split(".")[0] for match in _ENSEMBL.finditer(authority)}
    known_variants = {_variant_key(match.group()) for match in _VARIANT.finditer(authority)}
    # Mentioning an excluded rsID in a limitation never verifies its identity.
    identity_authority = _PROXY_EXCLUSION.sub('', re.sub(r'[*`]', ' ', authority))
    known_rsids = {match.group().casefold() for match in _RSID.finditer(identity_authority)}
    # The pairing check and the existence check must use the same frozen,
    # validated authority. A verified identity need not be repeated in a
    # displayed molecular fact to remain a valid report identifier.
    from .variant_identity_registry import validate_variant_identity_registry
    expansion = context.result.get('ld_expansion')
    registry = expansion.get('variant_identity_registry') if isinstance(expansion, Mapping) else None
    if validate_variant_identity_registry(registry):
        known_variants.update(_variant_key(value) for value in registry['variants'])
        known_variants.update(_variant_key(value) for value in registry['source_aliases'])
        known_rsids.update(rsid.casefold() for row in registry['variants'].values() for rsid in row['rsids'])
    disclosed_exclusions = _unverified_proxy_authority(context).difference(known_rsids)
    known_terms = {word.casefold() for word in re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", authority)}
    known_terms.update(term.casefold() for term in _GENERIC_UPPERCASE_TOKENS)
    known_terms.update({"h0", "h1", "h2", "h3", "h4", "r2", "clpp", "pip", "tss", "e2g", "v2g", "l2g"})
    normalized_authority = _normalize_phenotype_spellings(authority).casefold()
    findings = []
    for field, text in fields.items():
        if not isinstance(text, str) or len(text) > 1_000_000:
            continue
        plain = re.sub(r"[*`]", "", text)
        # Preserve source offsets while recognizing optional Markdown emphasis.
        exclusion_spans = [match.span() for match in _PROXY_EXCLUSION.finditer(re.sub(r'[*`]', ' ', text))]

        def excluded_metadata(match):
            return (match.group().casefold() in disclosed_exclusions
                    and any(start <= match.start() and match.end() <= end for start, end in exclusion_spans))

        def exclusion_claim(match):
            return any(start <= match.start() and match.end() <= end for start, end in exclusion_spans)

        unsupported = {}
        for match in _ENSEMBL.finditer(text):
            if match.group().casefold().split(".")[0] not in known_gene_ids:
                unsupported[match.group()] = ("gene_id", "report_gene_identifier_not_in_evidence")
        for pattern, known in ((_VARIANT, known_variants), (_RSID, known_rsids)):
            for match in pattern.finditer(text):
                identifier = _variant_key(match.group())
                if pattern is _RSID and exclusion_claim(match) and not excluded_metadata(match):
                    unsupported[match.group()] = ("variant_id", "report_proxy_exclusion_not_in_recorded_scope")
                elif identifier not in known:
                    if pattern is _RSID and excluded_metadata(match):
                        continue
                    unsupported[match.group()] = ("variant_id", "report_variant_identifier_not_in_evidence")
        symbols = _explicit_gene_symbols(text)
        symbols.update(match[1] for match in _SUBJECT.finditer(plain)
                       if any(character.isdigit() for character in match[1]))
        for symbol in symbols:
            phenotype_alias = _normalize_phenotype_spellings(symbol).casefold()
            known_phenotype = phenotype_alias != symbol.casefold() and re.search(
                r"(?<!\w)" + re.escape(phenotype_alias) + r"(?!\w)", normalized_authority)
            if symbol.casefold() not in known_terms and not known_phenotype and not _ENSEMBL.fullmatch(symbol):
                unsupported[symbol] = ("gene_id", "report_gene_identifier_not_in_evidence")
        for symbol, identifier in ranked.items():
            for match in re.finditer(r"\b" + re.escape(symbol) + r"\s*\(\s*(ENSG\d{6,}(?:\.\d+)?)\b", plain):
                if match[1].casefold().split(".")[0] != identifier.casefold().split(".")[0]:
                    unsupported[match[1]] = ("gene_id", "report_gene_symbol_identifier_conflict")
        for original, (metric, reason) in sorted(unsupported.items()):
            for match in re.finditer(r"(?<![\w])" + re.escape(original) + r"(?![\w])", text):
                if metric == 'variant_id' and excluded_metadata(match):
                    continue
                occurrence_reason = reason
                if metric == 'variant_id' and _RSID.fullmatch(original):
                    if exclusion_claim(match):
                        occurrence_reason = 'report_proxy_exclusion_not_in_recorded_scope'
                    elif original.casefold() in known_rsids:
                        continue
                    else:
                        occurrence_reason = 'report_variant_identifier_not_in_evidence'
                findings.append({"field": field, "metric": metric,
                    "original_value": original, "replacement": "Identifier retained for review",
                    "reason": occurrence_reason, "span": [match.start(), match.end()]})
                if len(findings) >= 256:
                    return findings
    return findings
