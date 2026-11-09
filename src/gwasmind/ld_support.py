"""Offline evidence/report semantics for an already retrieved LD expansion plan."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from .analysis_scope import validate_analysis_scope
from .configuration import parameter
from .models import EvidenceKind, EvidenceRecord, LocusQuery, VariantObservation

LD_PROXY_WARNING = (
    "Candidate-gene assessment includes verified LD proxies. Evidence measured at a proxy "
    "supports the selected locus indirectly through population-specific LD; it is not a "
    "direct measurement of the uploaded SNPs. Uploaded SNP counts and P values are unchanged."
)
LD_COVERAGE_WARNING = (
    "LD proxy coverage is incomplete. Available source-SNP and verified proxy evidence is "
    "retained, but additional linked variants and candidate genes may be missing."
)
LD_PROXY_EVIDENCE_WARNING = (
    "LD proxies were identified, but evidence retrieval for some proxies was incomplete. "
    "Validated source-SNP and proxy evidence is retained; candidate-gene coverage may be limited."
)
LD_MHC_WARNING = (
    "The LD-expanded analysis includes a proxy in the Open Targets MHC credible-set "
    "exclusion interval; correlated signals and study coverage require caution."
)
LD_DISABLED_NOTICE = (
    "LD proxy expansion is disabled in the recorded configuration; candidate-gene "
    "assessment uses only the uploaded SNPs."
)


def ld_provider_metadata(value: Mapping[str, Any]) -> dict[str, str]:
    """Retain recorded provider identity without inventing it for old ledgers."""
    provider, version = value.get("reference_provider"), value.get("provider_version")
    if provider is None and version is None:
        return {}
    if not isinstance(provider, str) or provider not in {"ensembl", "ldlink"}:
        raise ValueError("LD reference provider is not supported.")
    result = {"reference_provider": provider}
    if version is not None:
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", version):
            raise ValueError("LD provider version is invalid.")
        result["provider_version"] = version
    return result


@dataclass(frozen=True, slots=True)
class LDReportScope:
    provided: bool = False
    valid: bool = True
    disabled: bool = False
    complete: bool = True
    proxy_evidence_complete: bool | None = None
    source_count: int = 0
    source_ids: tuple[str, ...] = ()
    source_aliases: tuple[tuple[str, str], ...] = ()
    proxy_ids: tuple[str, ...] = ()
    links: tuple[Mapping[str, Any], ...] = ()
    population: str | None = None
    reference_panel: str | None = None
    threshold: float | None = None
    window_bp: int | None = None
    reference_provider: str | None = None
    provider_version: str | None = None

    def __post_init__(self) -> None:
        ld_provider_metadata({"reference_provider": self.reference_provider, "provider_version": self.provider_version})

    def evidence_origin(self, record: Mapping[str, Any]) -> str:
        identifier = str(record.get("variant_id") or "")
        if identifier in self.source_ids:
            return "source_selected"
        if identifier in self.proxy_ids:
            return "ld_proxy"
        return "unbound" if identifier else "gene_context"

    def permits_record(self, record: Mapping[str, Any]) -> bool:
        if not self.provided:
            return True
        if record.get("variant_id"):
            return self.evidence_origin(record) in {"source_selected", "ld_proxy"}
        payload = record.get("payload")
        context = payload.get("context") if isinstance(payload, Mapping) else {}
        return not isinstance(context, Mapping) or context.get("analysis_origin") != "ld_proxy"

    def prompt_summary(self, *, link_limit: int) -> dict[str, Any]:
        selected = self.links[:link_limit]
        return {"status": "disabled" if self.disabled else "complete" if self.complete else "incomplete",
            "source_variant_count": self.source_count, "added_variant_count": len(self.proxy_ids),
            "population": self.population, "reference_panel": self.reference_panel, "genome_build": "GRCh38",
            "threshold": self.threshold, "window_bp": self.window_bp, "window_bp_basis": "per_side",
            "expansion_complete": self.complete, "proxy_evidence_complete": self.proxy_evidence_complete,
            "evidence_relation": "via_population_ld", "source_gwas_p_copied": False,
            "ld_links": [dict(link) for link in selected], "total_link_count": len(self.links),
            "omitted_link_count": len(self.links) - len(selected), "query_ledger_included": False,
            **ld_provider_metadata({"reference_provider": self.reference_provider, "provider_version": self.provider_version})}

    @property
    def proxy_observations(self) -> tuple[dict[str, Any], ...]:
        return tuple({"variant_id": identifier, "chromosome": identifier.split("_")[0],
                      "position": int(identifier.split("_")[1]), "p_value": None} for identifier in self.proxy_ids)


def ld_report_scope(result: Mapping[str, Any], observations: Sequence[Mapping[str, Any]], intent: str) -> LDReportScope:
    """Reuse native scope validation; an invalid proxy plan never rewrites source data."""
    source_ids = tuple(dict.fromkeys(str(row.get("variant_id") or "") for row in observations))
    empty = LDReportScope(source_count=len(observations), source_ids=source_ids)
    if intent != "prioritize_genes" or result.get("ld_expansion") is None:
        return empty
    plan = result.get("ld_expansion")
    if not isinstance(plan, Mapping):
        return replace(empty, provided=True, valid=False, complete=False)
    try:
        provider_metadata = ld_provider_metadata(plan)
        def observation(row):
            return VariantObservation(**{key: row[key] for key in ("variant_id", "p_value", "chromosome", "position") if key in row})
        originals = tuple(observation(row) for row in observations)
        if originals != tuple(observation(row) for row in plan["source_observations"]):
            raise ValueError("LD plan changed the uploaded source observations")
        aliases = plan.get("resolved_source_aliases", {})
        omitted = set(plan.get("omitted_source_variant_ids", ()))
        source = tuple(dict.fromkeys(replace(row, variant_id=aliases.get(row.variant_id, row.variant_id))
                                    for row in originals if row.variant_id not in omitted))
        resolved_source_ids = {row.variant_id for row in source}
        linked_proxy_ids = {link.get("proxy_variant_id") for link in plan["ld_links"]}
        proxies = tuple(observation(row) for row in plan["analysis_variants"]
                        if row["variant_id"] not in source_ids and row["variant_id"] not in resolved_source_ids
                        and (not omitted or row["variant_id"] in linked_proxy_ids))
        analysis = (*source, *proxies)
        query = LocusQuery(locus_id="ld-report-scope", trait="scope validation", variants=source,
            task="prioritize_genes", genome_build="GRCh38", analysis_variants=analysis, ld_expansion=dict(plan))
        validate_analysis_scope(query)
    except (KeyError, TypeError, ValueError, AttributeError):
        return replace(empty, provided=True, valid=False, complete=False)
    bound_source_ids = tuple(dict.fromkeys((*source_ids, *(row.variant_id for row in source))))
    proxy_ids = tuple(row.variant_id for row in analysis if row.variant_id not in bound_source_ids)
    links = tuple({key: link[key] for key in ("selected_variant_id", "proxy_variant_id", "r2", "population", "genome_build")}
                  for link in plan["ld_links"] if link["proxy_variant_id"] in proxy_ids)
    return LDReportScope(provided=True, valid=True, disabled=plan["status"] == "disabled",
        complete=plan["complete"], proxy_evidence_complete=plan.get("proxy_evidence_complete"),
        source_count=len(observations), source_ids=bound_source_ids,
        source_aliases=tuple(aliases.items()),
        proxy_ids=proxy_ids, links=links, population=plan.get("population"),
        reference_panel=plan.get("reference_panel"), threshold=plan.get("threshold"), window_bp=plan.get("window_bp"),
        **provider_metadata)


def proxy_trace_label(scope: LDReportScope, records: Sequence[Mapping[str, Any]]) -> str:
    """Label the exact evidence trace, including mixed source/proxy traces."""
    if not scope.proxy_ids:
        return ""
    proxies = {str(row.get("variant_id")) for row in records if scope.evidence_origin(row) == "ld_proxy"}
    if not proxies:
        return ""
    sources = {str(row.get("variant_id")) for row in records
               if scope.evidence_origin(row) == "source_selected"}
    label = "Source + LD proxy evidence" if sources else "Indirect LD proxy evidence"
    # The short link remains readable in Markdown exports. Its exact, validated
    # provenance is carried in the link title and expanded by the safe DOM layer.
    detail = "Source SNPs: " + (", ".join(sorted(sources)) or "none in this evidence trace")
    detail += "; Verified LD proxies: " + ", ".join(sorted(proxies))
    detail += "; Proxy evidence is indirect support through population-specific LD."
    return f'[{label}](#gwasmind-evidence-origin "{detail}")'


def ld_scope_summary(scope: LDReportScope) -> str:
    if not scope.provided:
        return ""
    if scope.disabled:
        return LD_DISABLED_NOTICE
    text = (f"Gene-evidence retrieval uses {scope.source_count} uploaded SNP(s) and "
            f"{len(scope.proxy_ids)} verified LD proxy variant(s). Uploaded source P values "
            "remain attached only to the original SNPs.")
    if scope.proxy_ids:
        provider = {"ldlink": "LDlink", "ensembl": "Ensembl"}.get(scope.reference_provider)
        if provider:
            text += f" LD estimates were retrieved through {provider}."
        text += (f" The proxy reference is {scope.population}, {scope.reference_panel}, GRCh38, "
                 f"with r² ≥ {scope.threshold:g} and a {scope.window_bp:,} bp search window on each side. "
                 "Proxy evidence supports the selected locus indirectly.")
    if not scope.complete or scope.proxy_evidence_complete is False:
        # Optional expansion does not make valid selected SNPs incomplete. Keep
        # the full coverage receipt in the audit, without adding report warnings.
        text += " Only validated proxy variants and their retrieved evidence are included."
    return text


_VARIANT = re.compile(r"\b(?:rs\d+|(?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+)\b", re.I)
_DIRECT_MEASUREMENT = re.compile(r"\b(?:(?:e|s|p|tu)?QTL|colocali[sz]ation|regulat(?:ory|es?|ion)|enhancer|coding|splice)\b", re.I)
_SOURCE_SUBJECT = re.compile(r"\b(?:uploaded|source(?:[- ]selected)?|selected|original|index)\s+(?:GWAS\s+)?(?:SNP|variant)s?\b", re.I)
_SOURCE_DIRECT = re.compile(r"\b(?:direct(?:ly)?|itself|overlaps?)\b", re.I)
_VIA_LD = re.compile(r"\b(?:indirect(?:ly)?|via\s+(?:population[- ]specific\s+)?LD|LD[- ]prox(?:y|ies)|proxy[- ](?:measured|mediated)|through\s+LD)\b", re.I)
_R2 = re.compile(r"\br\s*(?:²|\^\s*2|2|[- ]squared)\s*(?P<operator>>=|<=|[=≥≤<>≈~]|(?:is|of|was|equals)\b)?\s*(?P<number>(?:\d+(?:\.\d+)?|\.\d+))", re.I)


def ld_relation_issues(text: str, scope: LDReportScope) -> tuple[str, ...]:
    """Bind explicit LD measurements to their pair; unrelated facts cannot lend r²."""
    if not scope.provided or not scope.valid or scope.disabled:
        return ()
    aliases = dict(scope.source_aliases)
    populations = {value.split("|")[0] for value in parameter("novelty.population_aliases")}
    issues = []
    plain = re.sub(r"[*`]", "", text)
    # A parenthesized LD assertion may separate its proxy, value and sources
    # with semicolons. Keep that one assertion together for identity binding.
    depth = 0
    enclosed = []
    for char in plain:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        enclosed.append(" " if char == ";" and depth else char)
    plain = "".join(enclosed)
    # A new proxy-overlap noun phrase owns its prefixed r². A preceding
    # source-E2G interval is not an implicit partner of that separate claim.
    r2_expression = re.sub(r"\(\?P<\w+>", "(?:", _R2.pattern)
    for clause in re.split(r"(?<=[.!?])\s+|[;\n]|\b(?:but|whereas)\b|,\s+while\s+(?="
                          r"(?:(?:the\s+)?(?:[A-Z]{3}\s+)?(?:LD[- ]?)?proxy(?:\s+variant)?\s+"
                          + _VARIANT.pattern + r"|(?:[\w-]+\s+){0,5}(?:records?|evidence|links?)\s+"
                          r"(?:is|are|was|were)\s+(?:LD[- ]?)?proxy[- ]mediated\b))"
                          r"|\b(?:plus|and)\s+(?=(?:[A-Z]{3}\s+)?" + r2_expression
                          + r"\s+(?:LD[- ]?)?proxy(?:[- ]mediated)?\s+(?:overlaps?|links?|records?|evidence)\b)",
                          plain, flags=re.I):
        matches = tuple(_R2.finditer(clause))
        # In a list of measurements, the next explicitly named proxy starts
        # its own item before its r² token. Do not lend that proxy identifier
        # backwards to the preceding item's source-proxy assertion.
        item_starts = {}
        for index, match in enumerate(matches):
            if not index:
                continue
            owner = re.search(r"(" + _VARIANT.pattern + r")\s*(?:(?:has|with)\s+)?$",
                              clause[matches[index - 1].end():match.start()], re.I)
            if owner and aliases.get(owner[1], owner[1]) in scope.proxy_ids:
                item_starts[index] = matches[index - 1].end() + owner.start()
        for index, match in enumerate(matches):
            start = item_starts.get(index, matches[index - 1].end() if index else 0)
            end = item_starts.get(index + 1, matches[index + 1].start() if index + 1 < len(matches) else len(clause))
            local = clause[start:end]
            named = {aliases.get(value, value) for value in _VARIANT.findall(local)}
            # An explicit pair belongs to this r² assertion. Another proxy in
            # the preceding explanation must not turn it into a three-way LD
            # claim. Coordinated lists with a third variant remain ambiguous.
            pair = re.match(r"\s+between\s+(" + _VARIANT.pattern + r")\s+and\s+("
                            + _VARIANT.pattern + r")", clause[match.end():end], re.I)
            if pair and not re.match(r"\s*(?:,\s*(?:and\s+)?|and\s+)" + _VARIANT.pattern,
                                     clause[match.end() + pair.end():end], re.I):
                named = {aliases.get(value.removeprefix("chr").replace(":", "_"),
                                     value.removeprefix("chr").replace(":", "_"))
                         for value in pair.groups()}
            else:
                # The immediately enclosing proxy owns "proxy A (r2 ...
                # with source B)". Earlier source-member lists are not part
                # of that explicitly two-variant assertion.
                proxy = re.search(r"\b(?:LD[- ]?)?proxy(?:\s+variant)?\s+("
                                  + _VARIANT.pattern + r")\s*\(\s*$", clause[:match.start()], re.I)
                if proxy is None:
                    proxy = re.search(r"(" + _VARIANT.pattern + r")\s*\(\s*(?:LD[- ]?)?proxy\s*,\s*$",
                                      clause[:match.start()], re.I)
                source = re.match(r"\s+with\s+(?:(?:the\s+)?(?:source(?:[- ]selected)?|selected|uploaded)"
                                  r"\s+(?:SNP|variant)\s+)?(" + _VARIANT.pattern + r")",
                                  clause[match.end():end], re.I)
                if proxy and source and not re.match(
                    r"\s*(?:,\s*(?:and\s+)?|and\s+)" + _VARIANT.pattern,
                    clause[match.end() + source.end():end], re.I,
                ):
                    proxy_id, source_id = (aliases.get(value.removeprefix("chr").replace(":", "_"),
                        value.removeprefix("chr").replace(":", "_")) for value in (proxy[1], source[1]))
                    if proxy_id not in scope.proxy_ids or source_id not in scope.source_ids:
                        issues.append("LD proxy and source identities must match their recorded roles")
                    named = {proxy_id, source_id}
            # Naming concrete variants makes this a measured relation claim.
            # Nearby prose about the retrieval cutoff cannot lend its value to
            # a missing or contradictory source-proxy edge.
            threshold = not named and bool(re.search(r"\b(?:threshold|cutoff|minimum)\b", local, re.I))
            candidates = [link for link in scope.links if named.issubset({
                aliases.get(link["selected_variant_id"], link["selected_variant_id"]), link["proxy_variant_id"]})]
            # Explicit one-to-many/many-to-many wording asserts each listed
            # source-proxy edge, not a nonexistent three-way LD statistic.
            # Require the complete Cartesian set and verify every value below.
            grouped_expected = None
            grouped_sources = re.search(
                r"\bwith\s+(?:(?:the\s+)?source\s+(?:variants|SNPs)\b|"
                r"each\s+of\s+(?:(?:the\s+)?source\s+(?:variants|SNPs)\s+)?"
                + _VARIANT.pattern + r")", local, re.I)
            # A bare, explicit list after "with" has the same one-to-many
            # meaning as "with each of". Every named source must have the
            # reported edge; a missing or discordant member still fails.
            bare_sources = re.match(
                r"\s+with\s+(?P<sources>" + _VARIANT.pattern
                + r"(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)" + _VARIANT.pattern + r")+)",
                clause[match.end():end], re.I)
            if bare_sources and all(aliases.get(value, value) in
                    {aliases.get(source, source) for source in scope.source_ids}
                    for value in _VARIANT.findall(bare_sources["sources"])):
                grouped_sources = grouped_sources or bare_sources
            if len(named) > 2 and grouped_sources:
                sources = named.intersection(aliases.get(value, value) for value in scope.source_ids)
                proxies = named.intersection(scope.proxy_ids)
                if sources and proxies and sources.isdisjoint(proxies) and sources | proxies == named:
                    edges = {(aliases.get(link["selected_variant_id"], link["selected_variant_id"]), link["proxy_variant_id"]):
                             link["r2"] for link in scope.links}
                    required_edges = {(source, proxy) for source in sources for proxy in proxies}
                    grouped_expected = [edges[edge] for edge in sorted(required_edges)] if required_edges <= edges.keys() else []
            # An unqualified lower-bound without a pair denotes the configured retrieval cutoff.
            threshold = threshold or (not named and match["operator"] in {">=", "≥"})
            expected = [scope.threshold] if threshold else [link["r2"] for link in candidates]
            value = Decimal(match["number"])
            tolerance = Decimal(1).scaleb(value.as_tuple().exponent) / 2
            def agrees(actual):
                if actual is None:
                    return False
                actual = Decimal(str(actual))
                if not threshold and match["operator"] in {">=", "≥", ">", "<=", "≤", "<"}:
                    return {">=": actual >= value, "≥": actual >= value, ">": actual > value,
                            "<=": actual <= value, "≤": actual <= value, "<": actual < value}[match["operator"]]
                # Integer rounding must not turn a strong correlation into
                # a claim of perfect LD, or rewrite the configured cutoff.
                if value in {Decimal(0), Decimal(1)}:
                    return actual == value
                return actual == value or abs(actual - value) < tolerance
            supported = (bool(grouped_expected) and all(agrees(actual) for actual in grouped_expected)) \
                if grouped_expected is not None and not threshold else any(agrees(actual) for actual in expected)
            if not supported:
                issues.append("LD r² must match the recorded source-proxy pair or configured threshold")
            # Only affirmative local population labels are bound, not caveats about transferability.
            mentioned = re.findall(r"\b(?:in|for|population(?:\s+is)?|ancestry(?:\s+is)?)\s+([A-Z]{3})\b", local)
            if any(population in populations and population != scope.population for population in mentioned):
                issues.append("LD population must match the recorded source-proxy reference")
    for match in re.finditer(r"\b(\d+)\s+(?:uploaded|source)\s+SNPs?\b", plain, re.I):
        if int(match[1]) != scope.source_count:
            issues.append("Uploaded SNP counts must use the original source observations, excluding LD proxies")
    for match in re.finditer(r"\b(\d+)\s+verified\s+LD\s+prox(?:y|ies)\b", plain, re.I):
        if int(match[1]) != len(scope.proxy_ids):
            issues.append("Verified LD proxy counts must match the validated analysis scope")
    return tuple(dict.fromkeys(issues))


def _claims_direct_source(clause: str, scope: LDReportScope) -> bool:
    # In "proxy A (r2=.9 with uploaded variants) overlaps an interval", the
    # LD partners inside parentheses do not replace proxy A as the subject.
    # Parentheses containing their own measurement/direct assertion are not
    # excluded; actual source reassignment must still be checked.
    relationship_spans = []
    for match in re.finditer(r"\b(?:LD[- ]?)?proxy(?:\s+variant)?\s+" + _VARIANT.pattern
                             + r"\s*\((?P<relationship>[^()]*)\)", clause, re.I):
        body = match["relationship"]
        if _R2.search(body) and not _DIRECT_MEASUREMENT.search(body) and not _SOURCE_DIRECT.search(body):
            relationship_spans.append(match.span("relationship"))
    def is_subject(position):
        return not any(start <= position < end for start, end in relationship_spans)
    subjects = [(match.start(), "source") for match in _SOURCE_SUBJECT.finditer(clause) if is_subject(match.start())]
    subjects.extend((match.start(), "proxy") for match in re.finditer(r"\b(?:LD[- ]?)?proxy\b", clause, re.I)
                    if is_subject(match.start()))
    subjects.extend((match.start(), "source" if match.group(0) in scope.source_ids else "proxy")
                    for match in _VARIANT.finditer(clause)
                    if match.group(0) in {*scope.source_ids, *scope.proxy_ids} and is_subject(match.start()))
    for match in _SOURCE_DIRECT.finditer(clause):
        if _negated_measurement(clause, match):
            continue
        preceding = [subject for subject in subjects if subject[0] < match.start()]
        subject = max(preceding, default=None) or min(subjects, default=None)
        if subject and subject[1] == "source":
            return True
    return False


def _without_ld_partner_parentheses(text: str, scope: LDReportScope) -> str:
    """An identified proxy's pure LD partners do not become molecular subjects.

    The caller checks LD numbers on the original text separately. Parentheses
    with their own molecular or direct-source assertion stay fully visible.
    """
    aliases = dict(scope.source_aliases)
    proxy_ids = frozenset(scope.proxy_ids)
    spans = []
    patterns = (
        r"(?P<variant>" + _VARIANT.pattern
        + r")\s*(?:,\s*(?:an?\s+)?(?:LD[- ]?)?proxy\s*)?\((?P<relationship>[^()]*)\)",
        r"\b(?:LD[- ]?)?proxy(?:\s+variant)?\s*\(\s*(?P<variant>" + _VARIANT.pattern
        + r")\s*,\s*(?P<relationship>[^()]*)\)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            identifier = match["variant"].removeprefix("chr").replace(":", "_")
            body = match["relationship"]
            if (aliases.get(identifier, identifier) in proxy_ids and _R2.search(body)
                    and not _DIRECT_MEASUREMENT.search(body) and not _SOURCE_DIRECT.search(body)):
                spans.append(match.span("relationship"))
    for start, end in reversed(spans):
        text = text[:start] + " " * (end - start) + text[end:]
    return text


def _negated_measurement(clause: str, match: re.Match[str]) -> bool:
    prefix = re.split(r"[,;]|\b(?:but|however|yet|whereas)\b", clause[:match.start()], flags=re.I)[-1]
    # In "no high-impact coding/splice annotation", the slash joins two
    # negated consequence nouns; it does not begin a positive source claim.
    if match.group().casefold() in {"coding", "splice"}:
        prefix = re.sub(r"\b(coding|splice)\s*/\s*$", r"\1 ", prefix, flags=re.I)
    qualifiers = r"(?:(?:establish|establishes|established|provide|provides|show|shows|support|supports|demonstrate|demonstrates|a|an|the|any|direct|molecular|functional|regulatory|measured|high[- ]impact|evidence|for|of|enough|to|and|or|(?:e|s|p|tu)?QTL|colocali[sz]ation|enhancer|coding|splice)\s+){0,12}"
    # "Whether the eQTL and sQTL share a causal variant" asks an unresolved
    # question; naming the assays does not assert a measurement on a source SNP
    # mentioned earlier. Scope this to adjacent noun-list qualifiers. A later
    # source subject/predicate cannot inherit the exemption, even in one sentence.
    unresolved = re.search(r"\bwhether\s+(?:(?:a|an|the|both|and|or|"
                           r"(?:(?:cis|trans)[- ]?)?(?:e|s|p|tu)?QTL)\s+)*"
                           r"(?:(?:cis|trans)[- ])?$", prefix, re.I)
    return bool(unresolved
                or re.search(r"\b(?:no|not|without|never|cannot|can't)\s+" + qualifiers + r"$", prefix, re.I)
                or re.search(r"\b(?:does|do|did)\s+not\s+(?:establish|prove|show|demonstrate|mean)\s+that\s+"
                             r"(?:the\s+)?(?:uploaded|source|selected|original|index)\s+(?:SNP|variant)s?\s+"
                             r"(?:(?:itself|themselves)\s+)?(?:directly\s+)?$", prefix, re.I)
                or re.search(r"\bno\s+(?:uploaded\s+)?" + _SOURCE_SUBJECT.pattern
                             + r"\s+(?:has|have|carries|carry|shows|show|provides|provide)\s+"
                             r"(?:(?:its|their)\s+own\s+)?(?:(?:a|an|the)\s+)?"
                             r"(?:(?-i:[A-Z][A-Z0-9_-]{1,40})\s+)?" + qualifiers + r"$", prefix, re.I)
                or re.search(r"\bnone\s+of\s+(?:the\s+)?(?:(?:retrieved|available|reported|supplied|"
                             r"molecular|functional|regulatory|or|(?:e|s|p|tu)?QTL)\s+){0,8}$", prefix, re.I))


def _measurement_kind(measurement: re.Match[str]) -> str:
    """Classify this mention, not an unrelated (possibly negated) later word."""
    term = measurement.group().casefold()
    if term.endswith("qtl"):
        return "qtl"
    if term.startswith("colocali"):
        return "colocalization"
    if term.startswith("regulat") or term == "enhancer":
        return "regulatory"
    return "variant_annotation"


def _measurement_spans(clause: str):
    mentions = []
    for mention in _DIRECT_MEASUREMENT.finditer(clause):
        if (mention.group().casefold() == "regulatory"
                and re.match(r"\s+T[- ]cells?\b", clause[mention.end():], re.I)):
            # The biosample name is not a second enhancer/regulatory assay.
            continue
        if mention.group().casefold() == "coding":
            prefix, suffix = clause[:mention.start()], clause[mention.end():]
            # Gene biotype and "non-coding" are not positive claims that the
            # uploaded variant has a measured coding consequence.
            if (re.search(r"\bprotein(?:-|\s+)$", prefix, re.I)
                    and re.match(r"\s+(?:gene|transcript|target)s?\b", suffix, re.I)
                    or re.search(r"\bnon(?:-|\s+)$", prefix, re.I)):
                continue
        mentions.append(mention)
    first_start = 0
    if mentions:
        # A preceding parenthetical statistic can remain after a semicolon
        # split. A closing parenthesis and explicit new assertion must not
        # transfer that H4/P value to the following measurement.
        previous_assertions = list(re.finditer(r"\)[^();]*?,\s*plus\b\s*",
                                               clause[:mentions[0].start()], re.I))
        if previous_assertions:
            first_start = previous_assertions[-1].end()
    boundaries = [first_start]
    for before, after in zip(mentions, mentions[1:]):
        connectors = list(re.finditer(r",?\s+\b(?:and|or|plus|with|including)\b\s+",
                                     clause[before.end():after.start()], re.I))
        boundaries.append(before.end() + connectors[-1].end() if connectors else after.start())
    boundaries.append(len(clause))
    for index, mention in enumerate(mentions):
        if not _negated_measurement(clause, mention):
            yield mention, clause[boundaries[index]:boundaries[index + 1]]


def ld_attribution_issues(text: str, *, scope: LDReportScope, evidence: Sequence[Mapping[str, Any]],
                          gene_id: str | None = None, check_source_absence: bool = True,
                          check_ld_relations: bool = True) -> tuple[str, ...]:
    """Reject explicit proxy-to-source reassignment, without inferring effect direction."""
    issues = list(ld_relation_issues(text, scope)) if check_ld_relations else []
    issues.extend(_ld_metadata_absence_issues(text, scope=scope, evidence=evidence, gene_id=gene_id))
    if check_source_absence:
        issues.extend(_source_measurement_absence_issues(text, scope=scope, evidence=evidence, gene_id=gene_id))
    if not scope.provided or not scope.proxy_ids:
        return tuple(issues)
    relevant = [row for row in evidence if not gene_id or row.get("gene_id") == gene_id]
    # Membership is boolean; retain original row order and source precedence.
    # Build the two small ID lookups once, not once per retained evidence row.
    source_ids, proxy_ids = frozenset(scope.source_ids), frozenset(scope.proxy_ids)
    def origin(row):
        identifier = str(row.get("variant_id") or "")
        if identifier in source_ids:
            return "source_selected"
        if identifier in proxy_ids:
            return "ld_proxy"
        return "unbound" if identifier else "gene_context"
    proxy_rows = [row for row in relevant if origin(row) == "ld_proxy"]
    if not proxy_rows:
        return tuple(issues)
    independent_subject = r"(?:(?:the\s+)?" + _SOURCE_SUBJECT.pattern + r"|" + _VARIANT.pattern + r")"
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"[*`]", "", text)):
        sentence = _without_ld_partner_parentheses(sentence, scope)
        # A new regulatory subject with its own predicate and source owner is
        # independent of a preceding proxy-QTL assertion. Do not split a bare
        # comma or coordinated "QTL and regulatory at the source SNP" list.
        for clause in re.split(
            r";|\b(?:but|however|whereas)\b|(?:,?\s+(?:and|plus)\s+|,\s+)(?="
            + independent_subject + r"\s+"
            r"(?:(?:itself|themselves)\s+)?(?:has|have|shows?|carr(?:y|ies)|is|are|regulates?|directly|overlaps?)\b)"
            r"|,\s+(?=(?:and\s+)?(?:the\s+)?"
            r"(?:regulatory|enhancer)\s+(?:links?|relationships?|records?|evidence)\b[^,;()]*?\b"
            r"(?:sit|sits|are|is|were|was|lie|lies)\s+(?:on|at|from)\s+(?:the\s+)?"
            r"(?:source(?:[- ]selected)?|selected|uploaded)\s+(?:SNP|variant)s?\b)",
            sentence, flags=re.I,
        ):
            measurements = list(_measurement_spans(clause))
            if not measurements:
                continue
            identifiers = {match.group(0).removeprefix("chr").replace(":", "_") for match in _VARIANT.finditer(clause)}
            names_source = bool(identifiers.intersection(scope.source_ids)) or bool(_SOURCE_SUBJECT.search(clause))
            if not names_source:
                continue
            if _VIA_LD.search(clause) and not _claims_direct_source(clause, scope):
                continue
            named_genes = set()
            for row in relevant:
                payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
                obj = payload.get("object") if isinstance(payload.get("object"), Mapping) else {}
                names = [row.get("gene_id"), payload.get("approvedSymbol"), obj.get("label")]
                if any(isinstance(name, str) and name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", clause, re.I) for name in names):
                    named_genes.add(row.get("gene_id"))
            for measurement, statement in measurements:
                kind = _measurement_kind(measurement)
                source_rows = [row for row in relevant if origin(row) == "source_selected"
                               and row.get("kind") == kind
                               and (not named_genes or row.get("gene_id") in named_genes)
                               and (not identifiers.intersection(scope.source_ids) or row.get("variant_id") in identifiers)]
                if not source_rows or _source_statistic_mismatch(statement, source_rows):
                    issues.append("LD-proxy evidence cannot be presented as a direct molecular measurement of the uploaded source SNP")
    return tuple(dict.fromkeys(issues))


def _ld_metadata_absence_issues(text: str, *, scope: LDReportScope,
                              evidence: Sequence[Mapping[str, Any]], gene_id: str | None) -> tuple[str, ...]:
    """Check claimed absence of LD metadata on the cited proxy records only.

    A reference-panel edge is not uploaded-GWAS or cohort-specific LD. A field
    must be present on the record's own link and agree with the verified scope
    before it can refute a statement that the cited records lack that field.
    """
    if not scope.provided or not scope.valid or scope.disabled:
        return ()
    aliases = dict(scope.source_aliases)
    def canonical(value):
        value = str(value or "").removeprefix("chr").replace(":", "_")
        return aliases.get(value, value)
    edges = relevant = proxy_ids = source_ids = None
    field = r"(?:r\s*(?:²|\^\s*2|2|[- ]squared)(?:\s+values?)?|population|genome[- ]build|genome\s+build|LD\s+(?:statistics?|measurements?|correlations?))"
    fields = field + r"(?:\s*(?:,\s*(?:(?:or|and)\s+)?|(?:or|and)\s+)" + field + r")*"
    cited = r"(?:the\s+)?(?:cited|supplied|retained|retrieved|displayed)(?:\s+LD[- ]proxy|\s+proxy)?\s+(?:records?|evidence|data)"
    missing = re.compile(
        cited + r"\s+(?:do(?:es)?|did)\s+not\s+(?:provide|report|include|contain)\s+(?:the\s+|any\s+)?(?P<provided_fields>" + fields + r")"
        + r"|\bno\s+(?P<no_fields>" + fields + r")[^.;!?]{0,160}\b(?:is|are|was|were)\s+(?:available|provided|reported)\s+(?:among|in|from)\s+" + cited
        + r"|\b(?P<unavailable_fields>" + fields + r")[^.;!?]{0,120}\b(?:is|are|was|were)\s+(?:missing|unavailable|absent|not\s+(?:available|provided|reported))\s+(?:among|in|from)\s+" + cited,
        re.I,
    )
    for clause in re.split(r"(?<=[.!?])\s+|[;\n]|\b(?:but|however|whereas)\b", re.sub(r"[*`]", "", text), flags=re.I):
        denial = missing.search(clause)
        if not denial or not re.search(r"\bLD\b|linkage[- ]disequilibrium", clause, re.I):
            continue
        if re.search(r"\b(?:cannot|can't)\s+(?:say|claim|conclude|establish)(?:\s+that)?\s*$|"
                     r"\b(?:false|incorrect|not\s+true)\s+that\s*$", clause[:denial.start()], re.I):
            continue
        # These describe a different measurement source, not the attached
        # population reference links. Do not transfer reference LD to a GWAS.
        if re.search(r"\b(?:uploaded|source)[- ]GWAS\b|\b(?:cohort[- ]specific|in[- ]sample)\s+LD\b", clause, re.I):
            continue
        if relevant is None:
            # Most prose does not deny LD metadata. Build the evidence/edge
            # indexes only for a real absence assertion and reuse hash-based
            # membership across its records and any later clauses.
            proxy_ids = frozenset(scope.proxy_ids)
            source_ids = frozenset(canonical(value) for value in scope.source_ids)
            edges = {(canonical(link["selected_variant_id"]), canonical(link["proxy_variant_id"])): link
                     for link in scope.links}
            relevant = [row for row in evidence if (not gene_id or row.get("gene_id") == gene_id)
                        and canonical(row.get("variant_id")) in proxy_ids]
        named = {canonical(match.group()) for match in _VARIANT.finditer(clause)}
        named_proxies = named.intersection(proxy_ids)
        named_sources = named.intersection(source_ids)
        named_genes = set(re.findall(r"\bENSG\d+\b", clause, re.I))
        for row in relevant:
            payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
            obj = payload.get("object") if isinstance(payload.get("object"), Mapping) else {}
            if any(isinstance(name, str) and name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", clause, re.I)
                   for name in (row.get("gene_id"), obj.get("label"))):
                named_genes.add(row.get("gene_id"))
        metrics = {"r2" if re.match(r"r\s*(?:²|\^\s*2|2|[- ]squared)|LD\b", match.group(), re.I)
                   else "population" if match.group().casefold() == "population" else "genome_build"
                   for match in re.finditer(field, next(value for value in denial.groupdict().values() if value is not None), re.I)}
        for row in relevant:
            if named_genes and row.get("gene_id") not in named_genes:
                continue
            proxy = canonical(row.get("variant_id"))
            if named_proxies and proxy not in named_proxies:
                continue
            if named and not named_proxies and not named_sources:
                continue
            payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
            context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
            links = context.get("ld_proxy_links", ())
            if not isinstance(links, (list, tuple)):
                continue
            for link in links:
                if not isinstance(link, Mapping) or canonical(link.get("proxy_variant_id")) != proxy:
                    continue
                source = canonical(link.get("selected_variant_id"))
                recorded = edges.get((source, proxy))
                if not recorded or (named_sources and source not in named_sources):
                    continue
                supplied = {key: link[key] for key in ("r2", "population", "genome_build") if link.get(key) is not None}
                if "r2" in supplied and (isinstance(supplied["r2"], bool)
                        or not isinstance(supplied["r2"], (float, int)) or not math.isfinite(supplied["r2"])):
                    continue
                if any(value != recorded.get(key) for key, value in supplied.items()):
                    continue
                if metrics.intersection(supplied):
                    return ("Cited proxy records provide verified LD metadata; do not describe their recorded r², population or genome build as unavailable",)
    return ()


def _source_measurement_absence_issues(text: str, *, scope: LDReportScope,
                                     evidence: Sequence[Mapping[str, Any]],
                                     gene_id: str | None) -> tuple[str, ...]:
    """A measured source member cannot be erased by an absolute absence claim.

    The proof is a statistic on that variant in the external raw record. Set
    membership, PIP and a pair-level colocalization probability are insufficient.
    This does not infer uploaded-GWAS effects or establish a causal mechanism.
    """
    if not scope.valid or not scope.source_ids:
        return ()
    relevant = [row for row in evidence if not gene_id or row.get("gene_id") == gene_id]
    aliases = dict(scope.source_aliases)
    def canonical(value):
        value = str(value or "").removeprefix("chr").replace(":", "_")
        return aliases.get(value, value)
    source_ids = {canonical(value) for value in scope.source_ids}
    # Splitting independent assertions keeps a true missing source-GWAS effect
    # statement from masking a false absence claim later in the same sentence.
    independent_negative = (
        r"(?:none|no|without)\b|(?:(?:the\s+)?(?:uploaded|source|selected|original|index)\s+"
        r"(?:GWAS\s+)?(?:SNP|variant)s?|[A-Z][A-Z0-9_-]+)\s+(?:has|have|is|are)\s+(?:no|without)\b"
    )
    clauses = re.split(r"(?<=[.!?])\s+|[;\n]|\b(?:but|however|whereas)\b"
                       r"|,?\s+\band\s+(?=" + independent_negative + r")",
                       re.sub(r"[*`]", "", text), flags=re.I)
    for clause in clauses:
        named_variants = {canonical(match.group()) for match in _VARIANT.finditer(clause)}
        if not (_SOURCE_SUBJECT.search(clause) or named_variants.intersection(source_ids)):
            continue
        if not (_DIRECT_MEASUREMENT.search(clause) or re.search(r"\bmolecular\b", clause, re.I)):
            continue
        negative = re.search(
            r"\bnone\s+of\b.{0,160}\b(?:evidence|measurements?|statistics?)\b.{0,90}"
            r"\b(?:is|are|was|were|provides?|represents?)\b.{0,40}\b(?:measurement|measured|direct)\b"
            r"|\b(?:no|without)\s+(?:(?:direct|retrieved|reported|available|supplied|external|"
            r"molecular|functional|regulatory|colocali[sz]ation|(?:e|s|p|tu)?QTL|study)\s+){0,8}"
            r"(?:evidence|measurements?|statistics?|associations?)\b", clause, re.I)
        if not negative:
            continue
        if re.search(r"\b(?:source|uploaded)[ -]GWAS\s+(?:effect|beta|statistic|data)\b", clause, re.I):
            continue
        if re.search(r"\b(?:evidence|measurements?|statistics?)\s+(?:(?:can|could|does|is|are|sufficient|enough|to|conclusively|definitively)\s+)*(?:establish|prove|demonstrate|confirm|show)\w*\b",
                     clause, re.I) and re.search(r"\bcausal(?:ity)?\b", clause, re.I):
            continue
        if re.search(r"\bproxy\s+evidence\b", clause, re.I):
            # Proxy evidence cannot itself establish a source measurement, even
            # when separate direct source measurements also happen to exist.
            continue
        qtl_types = {match.group().casefold() for match in re.finditer(r"\b(?:e|s|p|tu)?QTL\b", clause, re.I)}
        kinds = set()
        if qtl_types or re.search(r"\bmolecular\b", clause, re.I):
            kinds.add("qtl")
        if re.search(r"\bcolocali[sz]ation\b", clause, re.I):
            kinds.add("colocalization")
        if re.search(r"\b(?:regulatory|enhancer)\b", clause, re.I):
            kinds.add("regulatory")
        named_genes = set(re.findall(r"\bENSG\d+\b", clause, re.I))
        named_studies = set()
        named_sources = set()
        named_tissues = set()
        qualified_aliases = set()
        for row in relevant:
            payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
            obj = payload.get("object") if isinstance(payload.get("object"), Mapping) else {}
            context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
            names = (row.get("gene_id"), payload.get("approvedSymbol"), obj.get("label"))
            if any(isinstance(name, str) and name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", clause, re.I) for name in names):
                named_genes.add(row.get("gene_id"))
            study = row.get("study_id")
            if isinstance(study, str) and study and re.search(r"(?<!\w)" + re.escape(study) + r"(?!\w)", clause, re.I):
                named_studies.add(study)
            source, tissue = row.get("source"), row.get("tissue") or context.get("biosample")
            for value, named in ((source, named_sources), (tissue, named_tissues)):
                if isinstance(value, str) and value:
                    qualified_aliases.add(value)
                    if re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", clause, re.I):
                        named.add(value)
            if isinstance(study, str) and study:
                qualified_aliases.update((study, "study " + study))
        # A tissue/study-qualified absence needs a matching scope. Unknown
        # qualifiers are not disproved by unrelated measured source records.
        if any(not any(re.match(re.escape(alias) + r"(?!\w)", clause[match.end():], re.I)
                       for alias in qualified_aliases)
               for match in re.finditer(r"\b(?:in|from)\s+(?:the\s+)?", clause, re.I)):
            continue
        for row in relevant:
            if (canonical(row.get("variant_id")) not in source_ids or row.get("kind") not in kinds
                    or (named_variants and canonical(row.get("variant_id")) not in named_variants)
                    or (named_genes and row.get("gene_id") not in named_genes)
                    or (named_studies and row.get("study_id") not in named_studies)
                    or (named_sources and row.get("source") not in named_sources)):
                continue
            payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
            context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
            if named_tissues and (row.get("tissue") or context.get("biosample")) not in named_tissues:
                continue
            upstream = payload.get("upstream") if isinstance(payload.get("upstream"), Mapping) else {}
            raw = upstream.get("raw") if isinstance(upstream.get("raw"), Mapping) else {}
            study_type = str(raw.get("studyType") or context.get("study_type") or "").casefold()
            if qtl_types and "qtl" not in qtl_types and row.get("kind") == "qtl" and study_type not in qtl_types:
                continue
            if row.get("kind") in {"qtl", "colocalization"} and _has_bound_source_statistic(
                    raw, canonical(row.get("variant_id")), row.get("gene_id"), canonical):
                return ("Retrieved external-study measurements on the uploaded source SNP cannot be described as absent; keep uploaded-GWAS effects and causality separate",)
    return ()


def _has_bound_source_statistic(raw: Mapping[str, Any], variant_id: str, gene_id: str | None, canonical) -> bool:
    """Require raw variant-specific statistics; never borrow a different lead."""
    for key in ("queryVariantId", "variantId"):
        if raw.get(key) and canonical(raw[key]) != variant_id:
            return False
    if any(raw.get(key) and raw[key] != gene_id for key in ("geneId", "qtlGeneId")):
        return False
    member = raw.get("matchedLocusMember")
    if isinstance(member, Mapping) and member:
        if canonical(member.get("variantId")) != variant_id:
            return False
        measured = member
    else:
        if (raw.get("queryVariantRole") == "credible_set_member"
                or canonical(raw.get("variantId")) != variant_id
                or (raw.get("leadVariantId") and canonical(raw["leadVariantId"]) != variant_id)):
            return False
        measured = raw
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
            return None
        try:
            value = float(value)
        except (ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None
    for key in ("beta", "standardError", "oddsRatio", "p_value", "pValue", "pvalue"):
        value = number(measured.get(key))
        if value is not None and (key == "beta" or key == "standardError" and value >= 0
                or key == "oddsRatio" and value > 0 or key in {"p_value", "pValue", "pvalue"} and 0 <= value <= 1):
            return True
    mantissa, exponent = number(measured.get("pValueMantissa")), number(measured.get("pValueExponent"))
    return (mantissa is not None and exponent is not None and 0 < mantissa <= 10
            and exponent.is_integer() and -1000 <= exponent <= 0
            and (exponent < 0 or mantissa <= 1))


def _source_statistic_mismatch(text: str, source_rows: Sequence[Mapping[str, Any]]) -> bool:
    from .ranking_claims import (
        _METRIC,
        _measurement_matches,
        _metric_name,
        _number,
        raw_measurements,
    )

    records = tuple(EvidenceRecord(evidence_id=str(row.get("evidence_id") or ""),
        kind=EvidenceKind(row["kind"]), dataset=str(row.get("dataset") or ""),
        source=str(row.get("source") or ""), data_release=str(row.get("data_release") or ""),
        payload=dict(row.get("payload") or {}), gene_id=row.get("gene_id"), variant_id=row.get("variant_id"))
        for row in source_rows)
    measurements = tuple(item for record in records for item in raw_measurements(record))
    for match in _METRIC.finditer(text):
        value = _number(match.group("number"))
        if value is None:
            return True
        if match.group("percent"):
            value /= 100
        if not any(item.metric == _metric_name(match.group("metric"))
                   and _measurement_matches(match, value, item.value) for item in measurements):
            return True
    return False
