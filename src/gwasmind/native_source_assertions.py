"""Bind native ranking source-absence claims to the model-visible raw ledger."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .ld_support import (
    _R2,
    LDReportScope,
    _has_bound_source_statistic,
    _source_measurement_absence_issues,
    ld_attribution_issues,
    ld_relation_issues,
    ld_report_scope,
)
from .models import EvidenceKind, EvidenceRecord, LocusQuery
from .ranking_claims import (
    _METRIC,
    _NUMBER,
    _explicit_parenthetical_source_p_owner,
    _gene_aliases,
    _measurement_matches,
    _number,
    _parenthetical_bounds,
    _variant,
)

if TYPE_CHECKING:
    from .llm import SynthesisOutput

_SOURCE = re.compile(r"\b(?:source(?:[- ]selected)?|uploaded|selected|original|index)[- ]+(?:SNP|variant)s?\b"
                     r"|\b(?:source|uploaded)[- ]+(?:molecular|(?:e|s|p|tu)?QTL|colocali[sz]ation)\b", re.I)
_VARIANT = re.compile(r"\b(?:rs\d+|(?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+)\b", re.I)
_MOLECULAR = re.compile(r"\b(?:molecular|(?:e|s|p|tu)?QTL|colocali[sz]ation)\b", re.I)
_QTL = re.compile(r"\b(?:e|s|p|tu)?QTL\b", re.I)
_COLOC = re.compile(r"\bcolocali[sz]ation\b", re.I)
_ALL_PROXY = re.compile(
    r"\b(?:all|every)\s+(?:(?:the|retrieved|visible|available|direct|molecular|external|"
    r"(?:e|s|p|tu)?QTL|colocali[sz]ation|and|or)\s+){0,8}(?:evidence|records?)\s+"
    r"(?:is|are|was|were)\s+(?:(?:entirely|solely|exclusively|only)\s+)?"
    r"(?:on|at|from|through|via)\s+(?:the\s+)?(?:LD[- ]?)?prox(?:y|ies)\b"
    r"|\b(?:molecular|(?:e|s|p|tu)?QTL|colocali[sz]ation)\s+evidence\s+(?:is|are)\s+"
    r"(?:entirely|solely|exclusively)\s+(?:proxy[- ]mediated|(?:on|from)\s+(?:LD\s+)?prox(?:y|ies))\b"
    r"|\bonly\s+(?:LD\s+)?prox(?:y|ies)\s+(?:have|has|show|shows)\s+(?:direct\s+)?molecular\s+evidence\b", re.I,
)


def _normal(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mentions(text: str, value: str) -> bool:
    return bool(value and re.search(r"(?<!\w)" + re.escape(_normal(value)) + r"(?!\w)", _normal(text)))


def _raw(record: EvidenceRecord) -> Mapping[str, Any]:
    return _mapping(_mapping(_mapping(record.payload).get("upstream")).get("raw"))


def _clauses(text: str) -> list[str]:
    # An LD relation and its citation may share one parenthesis. Preserve that
    # complete grammatical scope when splitting independent statements.
    text = re.sub(r"\([^()]*\)",
        lambda match: match.group().replace(";", " ") if _R2.search(match.group()) else match.group(), text)
    return re.split(r"(?<=[.!?])\s+|[;\n]|\b(?:but|however|whereas|although)\b|"
        r",?\s+and\s+(?=(?:no|none|all)\b|(?:the\s+)?"
        r"(?:uploaded|source(?:[- ]selected)?|selected|original|index)[- ]+(?:SNP|variant)s?\s+"
        r"(?:itself\s+)?(?:has|have|is|are|shows?|directly)\b)", re.sub(r"[*`]", "", text), flags=re.I)


def native_ld_attribution_issues(text: str, *, scope: LDReportScope, evidence,
                                 gene_id: str | None = None, query: LocusQuery | None = None) -> tuple[str, ...]:
    """Do not reinterpret a checked negative clause as a positive measurement.

    All numerical LD assertions and every remaining affirmative source claim
    still use the existing shared validator. Final-report behavior is unchanged.
    """
    aliases = dict(scope.source_aliases)
    def canonical(value):
        value = str(value or "").removeprefix("chr").replace(":", "_")
        return aliases.get(value, value)
    source_ids = {canonical(value) for value in scope.source_ids}
    # Native ranking uses the typed numeric reviewer once. A second scalar
    # parser cannot contradict its complete range or resolved discourse scope.
    issues = list(ld_relation_issues(text, scope)) if query is None else []
    if query is not None:
        # The full numeric validator checks source GWAS facts independently.
        # Only a verified source/trait/value triple is excluded from this second
        # pass over external molecular statistics; all other metrics remain.
        spans = []
        for match in _METRIC.finditer(text):
            identifier = _explicit_parenthetical_source_p_owner(text, match, query)
            value = _number(match.group("number"))
            if identifier is None or value is None:
                continue
            if match.group("percent"):
                value /= 100
            if any(_variant(row.variant_id) == identifier and actual is not None
                   and _measurement_matches(match, value, actual)
                   for row in query.variants for actual in (_number(row.p_value),)):
                spans.append((match.start(), match.end()))
        for begin, end in reversed(spans):
            text = text[:begin] + " " * (end - begin) + text[end:]
    for clause in _clauses(text):
        if _absence_kinds(clause, source_ids, canonical):
            continue
        issues.extend(ld_attribution_issues(clause, scope=scope, evidence=evidence,
                                            gene_id=gene_id, check_source_absence=False,
                                            check_ld_relations=False))
    return tuple(dict.fromkeys(issues))


def _legacy_native_ld_numeric_bindings(text: str, *, query: LocusQuery, records: tuple[EvidenceRecord, ...],
                               aliases: Mapping[str, str], gene_aliases: Mapping[str, str],
                               owner: str | None = None) -> tuple[dict[str, Any], ...]:
    """Resolve r² from explicit pairs or cited record owners, never prose polarity.

    A named source elsewhere in a comparison is not an implicit LD partner.
    Unresolved ownership is returned as an empty binding for local redaction.
    """
    plan = _mapping(query.ld_expansion)
    from .ld_assertions import identity_resolver

    canonical = identity_resolver(plan)

    source_ids = {canonical(row.variant_id) for row in query.variants}
    edges = tuple({**edge, "source": canonical(edge.get("selected_variant_id")),
                   "proxy": canonical(edge.get("proxy_variant_id"))}
                  for edge in plan.get("ld_links", []) if isinstance(edge, Mapping)
                  and _number(edge.get("r2")) is not None)
    proxy_ids = {edge["proxy"] for edge in edges}
    by_id = {record.evidence_id: record for record in records}
    findings = []
    # These are explicit punctuation/contrast boundaries, not an inference of
    # which gene is biologically favored or whether a sentence is affirmative.
    spans = [span for span in re.finditer(r"(?<=[.!?])\s+|;|\b(?:whereas|while|but|however)\b", text, re.I)
             if _parenthetical_bounds(text, span.start()) is None]
    for metric in _R2.finditer(text):
        begin = max((span.end() for span in spans if span.end() <= metric.start()), default=0)
        end = min((span.start() for span in spans if span.start() >= metric.end()), default=len(text))
        clause = text[begin:end]
        before, after = text[begin:metric.start()], text[metric.end():end]
        reference_tokens = []
        opening = before.rfind("(")
        if opening > before.rfind(")"):
            closing = after.find(")")
            reference_tokens = re.findall(r"\bE\d+\b", before[opening:] + (after[:closing] if closing >= 0 else ""))
        adjacent = re.match(r"\s*\(([^()]*)\)", after)
        if adjacent:
            reference_tokens.extend(re.findall(r"\bE\d+\b", adjacent[1]))
        if not reference_tokens and re.search(r"\b(?:that|this|the same)\s+variant\b", before, re.I):
            groups = list(re.finditer(r"\([^()]*\bE\d+\b[^()]*\)", text[:metric.start()]))
            if groups:
                reference_tokens = re.findall(r"\bE\d+\b", groups[-1].group())
        explicit_ids = {aliases[token] for token in reference_tokens if token in aliases}
        direct_ids = {identifier for identifier in by_id if identifier in clause}
        if not explicit_ids and direct_ids:
            explicit_ids = direct_ids
        cited = tuple(by_id[identifier] for identifier in explicit_ids if identifier in by_id)
        selected = set()
        named_proxy = set()
        # Only syntax explicitly joining two identifiers creates a pair.
        pair_body = (r"(?P<a>" + _VARIANT.pattern + r")\s*(?:and|with|vs\.?|versus|[-–])\s*"
                     r"(?:(?:the|source|selected|uploaded|LD|proxy|SNP|variant)\s+)*"
                     r"(?P<b>" + _VARIANT.pattern + r")")
        # A relation stated beside this number outranks a preceding list of
        # proxy identifiers that does not describe the measured pair.
        pair = re.compile(r"\s+between\s+(?:(?:the|source|selected|uploaded|LD|proxy|SNP|variant)\s+)*"
                          + pair_body, re.I).match(clause, metric.end() - begin)
        bounds = _parenthetical_bounds(text, metric.start())
        enclosed_owner = enclosed_partner = None
        enclosed_populations = set()
        if bounds:
            opening, closing = bounds
            enclosed_owner = re.search(r"(?P<v>" + _VARIANT.pattern + r")\s*$", text[:opening], re.I)
            if not re.fullmatch(r"\s*(?:LD[- ]?proxy\s*,\s*)?"
                                r"(?:(?:EUR|AFR|AMR|EAS|SAS|ALL)\s*[;,]?\s*)?",
                                text[opening + 1:metric.start()], re.I):
                enclosed_owner = None
            enclosed_partner = re.match(r"\s+with\s+(?:(?:the|source|selected|uploaded|SNP|variant)\s+)*"
                                         r"(?P<v>" + _VARIANT.pattern + r")", after, re.I)
            enclosed_populations = set(re.findall(r"\b(?:EUR|AFR|AMR|EAS|SAS|ALL)\b",
                                                  text[opening:closing].upper()))
        enclosed_pair = enclosed_owner is not None and enclosed_partner is not None
        if enclosed_pair:
            # A variant immediately owning this parenthesis and its stated
            # partner outrank an earlier list of source variants in the clause.
            pair = None
            named_proxy = {canonical(enclosed_owner["v"])}
            selected = {canonical(enclosed_partner["v"])}
        elif pair is None:
            pair = re.search(pair_body, clause, re.I)
        if pair:
            pair_ids = {canonical(pair["a"]), canonical(pair["b"])}
            selected, named_proxy = pair_ids & source_ids, pair_ids & proxy_ids
            source_list_prefix = clause[metric.end() - begin:pair.start()]
            if pair_ids <= source_ids and re.match(
                r"\s+with\s+(?:(?:[A-Za-z]+|\d+)\s+){0,6}source\s+(?:SNPs?|variants?)\s+(?:including\s+)?$",
                source_list_prefix, re.I,
            ):
                # A list of two sources is not a source/proxy pair. The proxy
                # must be named independently beside this numeric field.
                selected, named_proxy, pair = set(), set(), None
        explicit_pair_valid = (named_proxy <= proxy_ids and selected <= source_ids
                               and len(enclosed_populations) <= 1
                               and not re.match(r"\s*(?:,\s*(?:and\s+)?|and\s+)" + _VARIANT.pattern,
                                                after[enclosed_partner.end():], re.I)) if enclosed_pair else (
                                   pair is None or len(selected) == len(named_proxy) == 1)
        leading = re.search(r"(?P<v>" + _VARIANT.pattern + r")\s*(?:(?:has|have|with|at)\s+(?:LD\s+)?)?(?:[,\(]\s*)?"
                            r"(?:(?:EUR|AFR|AMR|EAS|SAS|ALL)\s*[;,]?\s*)?$", before, re.I)
        # The exact proxy can introduce its LD value within a longer
        # parenthesis: "X at r2 ... with source Y", or "X at QTL PIP ...,
        # r2 ... with source Y". Require that adjacent grammar and the stated
        # partner below; other nearby variants never become an implicit pair.
        inline_owner = re.search(
            r"(?P<v>" + _VARIANT.pattern + r")\s+at\s+"
            r"(?:QTL\s+PIP\s*(?:=|:|is\s+|of\s+)?\s*" + _NUMBER + r"\s*,\s*)?$",
            before, re.IGNORECASE,
        )
        inline_partner = re.match(
            r"\s+(?:with|to)\s+(?:(?:the|source|selected|uploaded|SNP|variant)\s+)*"
            r"(?P<v>" + _VARIANT.pattern + r")", after, re.IGNORECASE,
        )
        if bounds and inline_owner and inline_partner:
            named_proxy = {canonical(inline_owner["v"])}
            selected = {canonical(inline_partner["v"])}
            # Preserve population and mixed-pair constraints even when the
            # LD pair starts inside, rather than immediately before, brackets.
            enclosed_pair = True
            enclosed_partner = inline_partner
            explicit_pair_valid = (named_proxy <= proxy_ids and selected <= source_ids
                and len(enclosed_populations) <= 1
                and not re.match(r"\s*(?:,\s*(?:and\s+)?|and\s+)" + _VARIANT.pattern,
                                 after[inline_partner.end():], re.IGNORECASE))
        # An explicit apposition can place the population between the number
        # and the partner: "X, an r2=1.0 (EUR) proxy of source variant Y".
        # Require both adjacent endpoints; this does not authorize a later
        # source-free restatement or borrow a variant from another clause.
        apposition_owner = re.search(r"(?P<v>" + _VARIANT.pattern + r")\s*,\s*an?\s*$", before, re.I)
        apposition_partner = re.match(
            r"\s*(?:\((?P<population>[^()]{1,40})\)\s*)?(?:LD[- ]?)?proxy\s+of\s+"
            r"(?:(?:the|source|selected|uploaded|SNP|variant)\s+)*(?P<v>" + _VARIANT.pattern + r")",
            after, re.I,
        )
        if apposition_owner and apposition_partner:
            named_proxy = {canonical(apposition_owner["v"])}
            selected = {canonical(apposition_partner["v"])}
            population = apposition_partner["population"]
            enclosed_populations = {population.strip().upper()} if population else set()
            enclosed_pair = True
            explicit_pair_valid = (named_proxy <= proxy_ids and selected <= source_ids
                and not re.match(r"\s*(?:,\s*(?:and\s+)?|and\s+)" + _VARIANT.pattern,
                                 after[apposition_partner.end():], re.I))
        trailing = re.match(r"\s*(?:(?:on|at|for)\s+|(?:LD[- ]?)?proxy\s+)"
                            r"(?P<v>" + _VARIANT.pattern + r")", after, re.I)
        if not named_proxy:
            if any(canonical(match["v"]) not in proxy_ids for match in (leading, trailing) if match):
                # An unresolved rsID or unrecorded owner cannot disappear and
                # then borrow the only nearby cited proxy's correct number.
                explicit_pair_valid = False
            named_proxy = {canonical(match["v"]) for match in (leading, trailing) if match
                           and canonical(match["v"]) in proxy_ids}
        # A proxy(...r²...with source SNP...) is an explicitly named pair.
        partner = re.match(r"\s+(?:with|to)\s+(?:(?:the|source|selected|uploaded|SNP|variant)\s+)*"
                           r"(?P<v>" + _VARIANT.pattern + r")", after, re.I)
        if named_proxy and partner:
            selected = {canonical(partner["v"])}
        source_list = re.match(r"\s+with\s+(?:(?:[A-Za-z]+|\d+)\s+){0,6}source\s+(?:SNPs?|variants?)\s+(?:including\s+)?"
                              r"(?P<ids>" + _VARIANT.pattern + r"(?:\s*(?:,|and)\s*" + _VARIANT.pattern + r")*)", after, re.I)
        if named_proxy and source_list:
            selected = {canonical(item.group()) for item in _VARIANT.finditer(source_list["ids"])}
        same_source = re.match(r"\s+with\s+the\s+same\s+(?:selected\s+)?source\s+(?:SNP|variant)\b", after, re.I)
        if named_proxy and same_source:
            sentence_start = max((span.end() for span in re.finditer(r"(?<=[.!?])\s+", text[:begin])), default=0)
            prefix = text[sentence_start:begin]
            antecedents = {canonical(match["v"]) for match in re.finditer(
                r"\b(?:source|selected|uploaded)\s+(?:(?:source|selected|SNP|variant)\s+)*"
                r"(?P<v>" + _VARIANT.pattern + r")", prefix, re.I)}
            earlier_sources = {canonical(match.group()) for match in _VARIANT.finditer(prefix)} & source_ids
            proxy_label = re.search(r"\bproxy\s+(?:variant\s+)?" + _VARIANT.pattern, before, re.I)
            selected = antecedents if proxy_label and len(antecedents) == 1 and len(antecedents | earlier_sources) == 1 else {"unresolved_previous_source"}
        if cited and not selected:
            record_proxies = {canonical(row.variant_id) for row in cited if canonical(row.variant_id) in proxy_ids}
            named_proxy = named_proxy & record_proxies if named_proxy else record_proxies
        elif not named_proxy:
            mentions = [(match.start(), gene) for alias, gene in gene_aliases.items()
                        for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", before, re.I)]
            gene = max(mentions, default=(0, owner))[1]
            eligible = tuple(row for row in records if gene is None or row.gene_id == gene)
            if re.search(r"\b(?:cis[- ]?)?(?:e|s|p|tu)?QTLs?\b", before, re.I):
                eligible = tuple(row for row in eligible if row.kind is EvidenceKind.QTL)
            owned = {canonical(row.variant_id) for row in eligible if canonical(row.variant_id) in proxy_ids}
            if len(owned) == 1:
                named_proxy = owned
                cited = eligible
        if re.match(r"\s+with\s+(?:an?\s+)?(?:unspecified|unknown)\s+source\b", after, re.I):
            explicit_pair_valid = False
        bound = tuple(edge for edge in edges if explicit_pair_valid and edge["proxy"] in named_proxy
                      and (not selected or edge["source"] in selected)
                      and (not enclosed_pair or not enclosed_populations
                           or str(edge.get("population") or plan.get("population") or "").upper()
                           in enclosed_populations))
        all_sources_bound = not selected or selected <= {edge["source"] for edge in bound}
        threshold = bool(re.search(r"\b(?:threshold|cutoff|minimum)\b", clause, re.I))
        if not named_proxy and threshold and _number(plan.get("threshold")) is not None:
            bound = ({"r2": plan["threshold"], "source": "configured_threshold", "proxy": ""},)
        values = {_number(edge["r2"]) for edge in bound}
        findings.append({"span": [metric.start(), metric.end()],
                         "number_span": [metric.start("number"), metric.end("number")],
                         "original_value": metric["number"], "operator": metric["operator"] or "=",
                         "values": tuple(sorted(value for value in values if value is not None)),
                         "bindings": bound, "evidence_ids": sorted(row.evidence_id for row in cited),
                         "owner_resolved": bool(bound) and all_sources_bound and (len(named_proxy) == 1 or threshold),
                         "explicit_source_ids": tuple(sorted(selected)), "all_bindings_required": bool(selected),
                         "metric": "r2"})
    return tuple(findings)


def native_ld_numeric_bindings(text: str, *, query: LocusQuery, records: tuple[EvidenceRecord, ...],
                               aliases: Mapping[str, str], gene_aliases: Mapping[str, str],
                               owner: str | None = None, all_records=None) -> tuple[dict[str, Any], ...]:
    """Review typed LD claims without treating parser uncertainty as a false value."""
    from .ld_assertions import legacy_parse_text, parse_ld_assertions, review_ld_assertions

    if not parse_ld_assertions(text) and not _R2.search(text):
        return ()

    findings = _legacy_native_ld_numeric_bindings(legacy_parse_text(text), query=query, records=records,
                                                  aliases=aliases, gene_aliases=gene_aliases, owner=owner)
    return review_ld_assertions(text, legacy_findings=findings, query=query, records=records,
                               aliases=aliases, gene_aliases=gene_aliases, owner=owner, all_records=all_records)


def _source_witness(record: EvidenceRecord, source_ids: set[str], canonical) -> bool:
    if record.kind not in {EvidenceKind.QTL, EvidenceKind.COLOCALIZATION}:
        return False
    if canonical(record.variant_id) not in source_ids or not record.gene_id:
        return False
    raw = _raw(record)
    if not any(raw.get(key) == record.gene_id for key in ("qtlGeneId", "geneId")):
        return False
    if not re.fullmatch(r"(?:e|s|p|tu)?qtl", str(raw.get("studyType") or ""), re.I):
        return False
    # A credible-set member needs its own measured statistic. PIP or pair-level
    # H4 alone cannot establish a direct molecular measurement of the source.
    if not _has_bound_source_statistic(raw, canonical(record.variant_id), record.gene_id, canonical):
        return False
    if record.kind is EvidenceKind.COLOCALIZATION:
        if not (raw.get("qtlStudyLocusId") or raw.get("leftStudyLocusId")) or not raw.get("rightStudyLocusId"):
            return False
        if not any(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) and 0 <= value <= 1
                   for value in (raw.get("h4"), raw.get("clpp"))):
            return False
    return True


def _absence_kinds(clause: str, source_ids: set[str], canonical) -> set[EvidenceKind]:
    molecular = _MOLECULAR.search(clause)
    exclusive = _ALL_PROXY.search(clause)
    if exclusive and (re.search(r"\b(?:not|whether)\s*$", clause[:exclusive.start()], re.I)
                      or re.search(r"\b(?:except|apart from|other than)\b", clause, re.I)):
        exclusive = None
    if not exclusive:
        named_source = _SOURCE.search(clause) or any(canonical(match.group()) in source_ids for match in _VARIANT.finditer(clause))
        if not named_source or molecular is None:
            return set()
        # Negation must deny the evidence, not its ability to establish causality.
        prefix = clause[:molecular.start()]
        negative = re.search(r"\b(?:no|neither|without)\b[^,;.!?]{0,120}$", prefix, re.I)
        # A comma can coordinate noun phrases under one negative determiner:
        # "no source-selected variant link, QTL or colocalization is retained".
        # Only a complete noun list with one shared passive predicate qualifies;
        # a following affirmative clause cannot inherit this exemption.
        noun = (r"(?:(?:direct|retained|visible|retrieved|available|molecular|source[- ]selected|"
                r"source|uploaded|selected|original|index)\s+)*"
                r"(?:(?:SNP|variant)s?|(?:e|s|p|tu)?QTL|colocali[sz]ation)"
                r"(?:\s+(?:evidence|links?|records?|associations?))?")
        coordinated_negative = re.search(
            r"\b(?:no|neither)\s+" + noun
            + r"(?:(?:\s*,\s*(?:(?:and|or|nor)\s+)?|\s+(?:and|or|nor)\s+)" + noun + r")+"
            r"\s+(?:is|are|was|were)\s+(?:retained|retrieved|available|visible|reported|provided)"
            r"\s*[.!?]?\s*$", clause, re.I)
        absent = re.search(r"\b(?:is|are|was|were)\s+(?:absent|missing|unavailable|not\s+(?:visible|retrieved|reported|provided))\b", clause[molecular.end():], re.I)
        if not (negative or coordinated_negative or absent):
            return set()
    if re.search(r"\b(?:source|uploaded)[- ]GWAS\b", clause, re.I):
        return set()
    if re.search(r"\b(?:establish|prove|demonstrate|confirm|show)\w*\b.{0,70}\bcausal(?:ity)?\b", clause, re.I):
        return set()
    kinds = set()
    if _QTL.search(clause):
        kinds.add(EvidenceKind.QTL)
    if _COLOC.search(clause):
        kinds.add(EvidenceKind.COLOCALIZATION)
    return kinds or {EvidenceKind.QTL, EvidenceKind.COLOCALIZATION}


def _matching_scope(clause, query, records, witnesses, owner, source_ids, canonical):
    aliases = _gene_aliases(records)
    aliases.update({gene.casefold(): gene for gene in query.candidate_genes})
    named_genes = {gene for name, gene in aliases.items() if _mentions(clause, name)}
    named_genes.update(re.findall(r"\bENSG\d+\b", clause))
    named_variants = {canonical(match.group()) for match in _VARIANT.finditer(clause)}
    if _ALL_PROXY.search(clause) and not named_variants.intersection(source_ids):
        # Proxy examples in an "all evidence is on proxies" claim do not
        # restrict its implicit denial of all uploaded-source measurements.
        named_variants = set()
    query_scoped = re.search(r"\b(?:query|queried|requested|target)[- ](?:trait|phenotype)(?:[- ]specific)?\b", clause, re.I)
    traits = {query.trait, query.source_trait or query.trait}
    tissues = set(query.relevant_tissues)
    studies, sources = set(), set()
    for record in records:
        raw, payload = _raw(record), _mapping(record.payload)
        context = _mapping(payload.get("context"))
        traits.update(value for key in ("rightTraitFromSource", "traitFromSource")
                      if isinstance(value := raw.get(key), str) and value)
        tissues.update(value for value in (record.tissue, context.get("biosample")) if isinstance(value, str) and value)
        studies.update(value for value in (record.study_id, raw.get("rightStudyId"), raw.get("leftStudyId")) if isinstance(value, str) and value)
        if record.source:
            sources.add(record.source)
    named_traits = {_normal(query.trait)} if query_scoped else {_normal(value) for value in traits if _mentions(clause, value)}
    # Gene expression trait IDs name the gene rather than a disease restriction.
    named_traits.difference_update(_normal(name) for name in aliases)
    named_tissues = {_normal(value) for value in tissues if _mentions(clause, value)}
    named_studies = {value for value in studies if _mentions(clause, value)}
    named_sources = {value for value in sources if _mentions(clause, value)}
    qualified_names = [*aliases, *traits, *tissues, *studies, *sources]
    for match in re.finditer(r"\b(?:for|in|from|with)\s+(?:the\s+)?", clause, re.I):
        tail = clause[match.end():]
        if re.match(r"(?:this gene|source[- ](?:selected[- ])?(?:SNP|variant)|uploaded (?:SNP|variant)|"
                    r"(?:query|queried|requested|target)[- ](?:trait|phenotype)|"
                    r"(?:any|all|this|these|selected)\s+(?:candidate|gene|locus|loci|SNP|variant|tissue|study|studies)s?|"
                    r"(?:model[- ]visible |visible |retrieved |provided )?(?:ledger|projection|evidence|records))\b", tail, re.I):
            continue
        if not any(re.match(re.escape(_normal(name)) + r"(?!\w)", _normal(tail)) for name in qualified_names if name):
            # Unknown phenotype/tissue/study qualifiers cannot be contradicted
            # by an unrelated source record. Do not guess their meaning.
            return ()
    matched = []
    for record in witnesses:
        raw, context = _raw(record), _mapping(_mapping(record.payload).get("context"))
        if (owner and record.gene_id != owner) or (named_genes and record.gene_id not in named_genes):
            continue
        if named_variants and canonical(record.variant_id) not in named_variants:
            continue
        trait = raw.get("rightTraitFromSource") if record.kind is EvidenceKind.COLOCALIZATION else raw.get("traitFromSource")
        if named_traits and _normal(str(trait or "")) not in named_traits:
            continue
        if named_tissues and _normal(str(record.tissue or context.get("biosample") or "")) not in named_tissues:
            continue
        if named_studies and not named_studies.intersection({record.study_id, raw.get("leftStudyId"), raw.get("rightStudyId")}):
            continue
        if named_sources and record.source not in named_sources:
            continue
        qtl_types = {match.group().casefold() for match in _QTL.finditer(clause)}
        if record.kind is EvidenceKind.QTL and qtl_types and "qtl" not in qtl_types and str(raw.get("studyType")).casefold() not in qtl_types:
            continue
        matched.append(record)
    return tuple(matched)


def native_source_assertion_issues(query: LocusQuery, evidence: tuple[EvidenceRecord, ...],
                                   output: SynthesisOutput) -> tuple[str, ...]:
    """A broad absence claim must agree with all evidence the model could see."""
    visible = None if output.model_visible_evidence_ids is None else set(output.model_visible_evidence_ids)
    records = tuple(record for record in evidence if visible is None or record.evidence_id in visible)
    aliases = _mapping(query.ld_expansion).get("resolved_source_aliases", {})
    aliases = {str(key).casefold(): str(value).casefold() for key, value in _mapping(aliases).items()}
    def canonical(value):
        value = str(value or "").casefold().removeprefix("chr").replace(":", "_")
        return aliases.get(value, value)
    source_ids = {canonical(row.variant_id) for row in query.variants}
    witnesses = tuple(record for record in records if _source_witness(record, source_ids, canonical))
    if not records:
        return ()
    plan = _mapping(query.ld_expansion)
    observations = plan.get("source_observations", [asdict(row) for row in query.variants])
    scope = ld_report_scope({"ld_expansion": query.ld_expansion}, observations, query.task)
    prose = [("summary", output.summary, None)]
    for candidate in output.ranked_genes:
        prose.append((f"rationale for {candidate.gene_id}", candidate.rationale, candidate.gene_id))
        prose.extend((f"limitation for {candidate.gene_id}", text, candidate.gene_id) for text in candidate.limitations)
    prose.extend((f"claim {index}", claim.text, None) for index, claim in enumerate(output.claims, 1))
    issues = []
    for label, text, owner in prose:
        for clause in _clauses(text):
            if scope.provided:
                scoped_records = _matching_scope(clause, query, records, records, owner, source_ids, canonical)
                # Preserve the established finite member-statistic protection;
                # only apply its biological qualifiers before testing absence.
                issues.extend(_source_measurement_absence_issues(
                    clause, scope=scope, evidence=[record.to_dict() for record in scoped_records], gene_id=owner))
            kinds = _absence_kinds(clause, source_ids, canonical)
            if not kinds:
                continue
            matched = _matching_scope(clause, query, records,
                tuple(record for record in witnesses if record.kind in kinds), owner, source_ids, canonical)
            if matched:
                record = matched[0]
                issues.append(f"{label} denies model-visible source-SNP {record.kind.value} evidence "
                              f"({record.evidence_id}); distinguish trait-specific gaps from absence of source evidence")
    return tuple(dict.fromkeys(issues))
