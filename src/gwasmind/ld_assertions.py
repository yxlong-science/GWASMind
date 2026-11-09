"""Typed LD assertions shared by ranking and report fact review.

Parsing uncertainty is not a numerical contradiction. The complete expression
is retained unless one explicitly owned scalar can be corrected from one edge.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
_EXPRESSION = re.compile(
    r"\br\s*(?:²|\^\s*2|2|[- ]squared)\s*"
    r"(?P<operator>>=|<=|[=≥≤<>≈~]|(?:is|of|was|equals|approximately|approx\.?|about|around|"
    r"at\s+least|at\s+most|up\s+to|below|above|under|over)\b)?\s*"
    r"(?P<number>" + NUMBER + r")"
    r"(?:\s*(?:[-–—]|to)\s*(?P<upper>" + NUMBER + r"))?(?!\w|\.\d|\s*(?:[-–—]|to)\s*\d)", re.I,
)
_VARIANT = re.compile(r"\b(?:rs\d+|(?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+)\b", re.I)
_CITATION = re.compile(r"\bE\d+\b|\{\{cite_fact:([^}]+)\}\}")
_APPROX = re.compile(r"\b(?:approximately|approx\.?|about|around)\s*(?=" + NUMBER + r")", re.I)
_UNPARSED = re.compile(r"\br\s*(?:²|\^\s*2|2|[- ]squared)\s*(?:[=≥≤<>≈~]+\s*)?"
    r"(?P<number>[+-]?(?:\d[\d.eE+–—-]*|\.\d[\d.eE+–—-]*))", re.I)


def _number(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _rounded_equal(reported, actual):
    if reported is None or actual is None:
        return False
    if reported == actual:
        return True
    if reported == reported.to_integral() and -1 <= reported <= 1:
        return False
    tolerance = Decimal(1).scaleb(reported.as_tuple().exponent) / 2
    return abs(reported - actual) <= tolerance


def parse_ld_assertions(text):
    """Return scalar/range/bound assertions with complete, stable text spans."""
    assertions = []
    for match in _EXPRESSION.finditer(text):
        operator = (match["operator"] or "=").casefold()
        prefix = text[max(0, match.start() - 18):match.start()]
        preceding = re.search(r"\b(below|under|above|over|at\s+least|at\s+most)\s*$", prefix, re.I)
        if preceding and operator == "=":
            operator = preceding[1].casefold()
        operator = {"below": "<", "under": "<", "above": ">", "over": ">", "at least": ">=",
                    "at most": "<=", "up to": "<=", "≥": ">=", "≤": "<="}.get(operator, operator)
        approximate = operator in {"approximately", "approx", "approx.", "about", "around", "≈", "~"}
        if operator not in {"<", "<=", ">", ">="}:
            operator = "="
        upper = match["upper"]
        assertions.append({"span": [match.start(), match.end()],
            "number_span": [match.start("number"), match.end("number")],
            "original_value": text[match.start("number"):match.end("upper") if upper else match.end("number")],
            "original_span": match.group(), "lower": match["number"], "upper": upper,
            "operator": operator, "approximate": approximate,
            "assertion_type": "range" if upper is not None else "bound" if operator != "=" else "scalar"})
    return tuple(assertions)


def legacy_parse_text(text):
    """Expose approximate scalar ownership to the legacy syntax binder in place."""
    return _APPROX.sub(lambda match: " " * len(match.group()), re.sub(r"[*`]", " ", text))


def identity_resolver(plan):
    """Canonicalize through frozen proofs, never through coordinate proximity."""
    aliases = {str(key).casefold().removeprefix("chr").replace(":", "_"):
               str(value).casefold().removeprefix("chr").replace(":", "_")
               for key, value in (plan.get("resolved_source_aliases") or {}).items()}
    registry = plan.get("variant_identity_registry")
    try:
        from .variant_identity_registry import build_variant_identity_registry, variant_aliases
    except ImportError:
        registry_aliases = {}
    else:
        if not registry:
            registry = build_variant_identity_registry(
                source_normalizations=plan.get("source_identity_normalizations", ()), ld_expansion=plan,
                genome_build=plan.get("genome_build") or "GRCh38")
        registry_aliases = {str(key).casefold(): str(value).casefold()
                            for key, value in variant_aliases(registry).items()}

    def canonical(value):
        token = str(value or "").casefold().removeprefix("chr").replace(":", "_")
        if token in registry_aliases:
            return registry_aliases[token]
        if token.startswith("rs"):
            return "unresolved:" + token
        return aliases.get(token, token)
    return canonical


def _field(record, key, default=None):
    return record.get(key, default) if isinstance(record, Mapping) else getattr(record, key, default)


def record_ld_witnesses(record, canonical):
    """QTL lead statistics do not transfer LD ownership away from local witnesses."""
    payload = _field(record, "payload", {}) or {}
    context = payload.get("context") or {}
    raw = (payload.get("upstream") or {}).get("raw") or {}
    witnesses = context.get("locus_anchor_witnesses") or raw.get("locusAnchorWitnesses") or []
    variants = {canonical(item.get("variant_id")) for item in witnesses if isinstance(item, Mapping) and item.get("variant_id")}
    if not variants and _field(record, "variant_id"):
        variants.add(canonical(_field(record, "variant_id")))
    return variants


def _clause(text, assertion):
    # Citation parentheses may contain semicolons; only sentence/paragraph
    # boundaries restrict discourse resolution here. The syntax binder keeps
    # its finer independent-pair boundaries.
    boundaries = list(re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text))
    start = max((item.end() for item in boundaries if item.end() <= assertion["span"][0]), default=0)
    end = min((item.start() for item in boundaries if item.start() >= assertion["span"][1]), default=len(text))
    return start, end, text[start:end]


def _local_refs(text, assertion, end):
    tail = text[assertion["span"][1]:end]
    groups = list(re.finditer(r"\([^()]*\bE\d+\b[^()]*\)|(?:\{\{cite_fact:[^}]+\}\}\s*)+", tail))
    if not groups:
        return set()
    first = groups[0]
    # A second quantitative assertion starts another measurement scope.
    if _EXPRESSION.search(tail[:first.start()]):
        return set()
    return {match[1] or match.group() for match in _CITATION.finditer(first.group())}


def _retrieval_threshold_scope(clause, assertion):
    """Recognize a declared inclusion policy, not observed pairwise LD bounds."""
    if _VARIANT.search(clause):
        return False
    if re.search(r"\b(?:threshold|cutoff|minimum)\b", clause, re.I):
        return True
    return (assertion["assertion_type"] == "bound" and assertion["operator"] == ">="
        and bool(re.search(r"\b(?:added\s+(?:analysis\s+)?|LD[- ]expanded\s+|expanded\s+analysis\s+)(?:variants|proxies)\b", clause, re.I))
        and bool(re.search(r"\b(?:entered|included|selected|added|retained|admitted)\s+(?:through|using|via|under|by|with)\b[^.!?]{0,90}\bLD\b", clause, re.I)))


def review_ld_assertions(text, *, legacy_findings, query, records, aliases, gene_aliases,
                         owner=None, all_records=None):
    """Classify mathematical validity separately from binding completeness."""
    plan = query.ld_expansion or {}
    from .numeric_provenance import ld_numeric_facts
    fact_rows = ld_numeric_facts(query)
    canonical = identity_resolver(plan)
    edges = tuple({**edge, "source": canonical(edge.get("selected_variant_id")),
                   "proxy": canonical(edge.get("proxy_variant_id"))}
                  for edge in plan.get("ld_links", ()) if isinstance(edge, Mapping)
                  and _number(edge.get("r2")) is not None)
    by_id = {_field(record, "evidence_id"): record for record in (all_records or records)}
    local_records = {_field(record, "evidence_id"): record for record in records}
    results = []
    assertions = list(parse_ld_assertions(text))
    for match in _UNPARSED.finditer(text):
        if not any(left <= match.start() < right for left, right in (item["span"] for item in assertions)):
            value = match["number"].rstrip(".")
            finish = match.start("number") + len(value)
            assertions.append({"span": [match.start(), finish], "number_span": [match.start("number"), finish],
                "original_value": value, "original_span": text[match.start():finish], "lower": None,
                "upper": None, "operator": "=", "approximate": False, "assertion_type": "unparsed"})
    for assertion in sorted(assertions, key=lambda item: item["span"][0]):
        inherited = next((item for item in legacy_findings
                          if item["number_span"][0] == assertion["number_span"][0]), {})
        finding = {**inherited, **assertion, "metric": "r2"}
        finding.setdefault("bindings", ())
        finding.setdefault("values", ())
        finding.setdefault("evidence_ids", [])
        finding.setdefault("explicit_source_ids", ())
        finding.setdefault("owner_resolved", False)
        finding.setdefault("all_bindings_required", False)
        start, end, clause = _clause(text, assertion)
        refs = _local_refs(text, assertion, end)
        ids = {aliases.get(ref, ref) for ref in refs}
        cited = tuple(by_id[identifier] for identifier in ids if identifier in by_id)
        complete_citation_scope = not ids or ids <= by_id.keys()
        before = text[start:assertion["span"][0]]
        mentions = [(match.start(), gene) for alias, gene in gene_aliases.items()
                    for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", before, re.I)]
        current_owner = owner if owner is not None else (max(mentions)[1] if mentions else None)
        aggregate = assertion["assertion_type"] in {"range", "bound"} and bool(re.search(r"\b(?:all|every|proxies|links)\b", clause, re.I))
        aggregate = aggregate or bool(re.search(r"\b(?:all|every)\s+(?:variant\s+|local\s+|LD\s+)?(?:proxies|links)\b", before, re.I)
            or re.match(r"\s+for\s+directly\s+named\s+proxies\b", text[assertion["span"][1]:end], re.I))
        mode = "explicit_pair" if finding["explicit_source_ids"] and finding["owner_resolved"] else "record_scope"
        if _retrieval_threshold_scope(clause, assertion) and _number(plan.get("threshold")) is not None:
            finding.update(bindings=({"r2": plan["threshold"], "source": "configured_threshold", "proxy": "",
                "population": plan.get("population"), "genome_build": plan.get("genome_build")},),
                values=(_number(plan["threshold"]),), owner_resolved=True, all_bindings_required=True)
        if any(edge.get("source") == "configured_threshold" for edge in finding["bindings"]):
            mode = "configured_threshold"
            aggregate = False
        selected_sources = {canonical(row.variant_id) for row in query.variants}
        named_variants = {canonical(match.group()) for match in _VARIANT.finditer(clause)}
        maximum = (assertion['assertion_type'] == 'scalar'
            and bool(re.search(r"\bmax(?:imum)?(?:\s+recorded)?\s*$", before, re.I))
            and bool(re.search(r"\b(?:selected|source)\s+(?:SNPs|variants)\b", clause, re.I))
            and not re.search(r"\b(?:lead|index)\s+(?:SNP|variant)\b", clause, re.I))
        if maximum and mode == 'record_scope':
            eligible = cited or tuple(record for record in local_records.values()
                                      if current_owner is None or _field(record, 'gene_id') == current_owner)
            witnesses = set().union(*(record_ld_witnesses(record, canonical) for record in eligible)) if eligible else set()
            named_proxies = named_variants - selected_sources
            proxies = witnesses & named_proxies if named_proxies else witnesses
            if len(proxies) == 1 and complete_citation_scope:
                proxy = next(iter(proxies))
                maxima = [fact for fact in fact_rows if fact['scope'] == 'maximum_recorded_selected_source_ld'
                          and fact['owner']['variant_id'] == proxy]
                populations = {item.upper() for item in re.findall(r"\b(?:EUR|AFR|AMR|EAS|SAS|ALL)\b", clause, re.I)}
                builds = {item.upper() for item in re.findall(r"\bGRCh(?:37|38)\b", clause, re.I)}
                maxima = [fact for fact in maxima if (not populations or populations == {fact['population'].upper()})
                          and (not builds or builds == {fact['genome_build'].upper()})]
                if len(maxima) == 1:
                    fact = maxima[0]
                    bound = tuple(edge for edge in edges if edge['proxy'] == proxy and edge['source'] in selected_sources
                                  and str(edge.get('population') or plan.get('population') or 'not_reported') == fact['population']
                                  and str(edge.get('genome_build') or plan.get('genome_build') or 'not_reported') == fact['genome_build'])
                    finding.update(bindings=bound, values=(_number(fact['value']),), owner_resolved=bool(bound),
                                   all_bindings_required=True, numeric_fact_ids=[fact['fact_id']])
                    mode = 'maximum_recorded_selected_source_ld'
                    aggregate = False
        if mode == "record_scope" and finding["owner_resolved"] and len(finding["bindings"]) == 1 and len(selected_sources) == 1:
            edge = finding["bindings"][0]
            if edge["source"] in selected_sources and (edge["proxy"] in named_variants or finding["evidence_ids"]):
                mode = "single_selected_source"
                finding["explicit_source_ids"] = tuple(sorted(selected_sources))
        if aggregate:
            eligible = cited or tuple(record for record in local_records.values()
                                      if current_owner is None or _field(record, "gene_id") == current_owner)
            if re.search(r"\b(?:all|every)\b", clause, re.I) and current_owner and not refs:
                eligible = tuple(record for record in by_id.values() if _field(record, "gene_id") == current_owner)
            witnesses = set().union(*(record_ld_witnesses(record, canonical) for record in eligible)) if eligible else set()
            populations = {value.upper() for value in re.findall(r"\b(?:(?i:EUR|AFR|AMR|EAS|SAS)|ALL)\b", clause)}
            bound = tuple(edge for edge in edges if edge["proxy"] in witnesses
                          and (not populations or populations == {str(edge.get("population", "")).upper()}))
            finding.update(bindings=bound, values=tuple(sorted({_number(edge["r2"]) for edge in bound})),
                           evidence_ids=sorted(_field(record, "evidence_id") for record in eligible),
                           owner_resolved=bool(bound), all_bindings_required=True)
            mode = "scoped_aggregate"
            if not current_owner and not refs and re.search(r"\ball\s+regional\b", clause, re.I):
                complete_citation_scope = False
        qualifier_text = clause
        if not aggregate:
            separators = list(re.finditer(r";|\b(?:whereas|while|but|however)\b", clause, re.I))
            left = max((item.end() for item in separators if start + item.end() <= assertion["span"][0]), default=0)
            right = min((item.start() for item in separators if start + item.start() >= assertion["span"][1]), default=len(clause))
            qualifier_text = clause[left:right]
        # Population/build qualifiers belong to this assertion, including
        # those outside parentheses. They cannot borrow a different panel.
        populations = {value.upper() for value in re.findall(r"\b(?:(?i:EUR|AFR|AMR|EAS|SAS)|ALL)\b", qualifier_text)}
        builds = set(value.upper() for value in re.findall(r"\bGRCh(?:37|38)\b", qualifier_text, re.I))
        if (len(populations) > 1 or len(builds) > 1 or any(
                (populations and str(edge.get("population") or plan.get("population") or "").upper() not in populations)
                or (builds and str(edge.get("genome_build") or plan.get("genome_build") or "").upper() not in builds)
                for edge in finding["bindings"])):
            finding.update(bindings=(), values=(), owner_resolved=False, numeric_fact_ids=[])
        # The definite placement/proxy in the same paragraph may refer back
        # to one already explicit pair. Never pick its antecedent by number.
        if mode == "record_scope" and not refs and not _VARIANT.search(clause):
            definite = re.search(r"\b(?:the\s+(?:same\s+)?(?:promoter[- ]proximal\s+placement|promoter\s+proxy|regulatory\s+(?:placement|link)|proxy|variant)|this\s+(?:proxy|variant|link))\b", clause, re.I)
            prior = [item for item in results if item.get("binding_mode") == "explicit_pair"
                     and item.get("verification_status") == "verified"
                     and assertion["span"][0] - item["span"][1] <= 1600
                     and "\n\n" not in text[item["span"][1]:assertion["span"][0]]]
            keys = {(edge["source"], edge["proxy"], edge.get("population"), edge.get("genome_build"))
                    for item in prior for edge in item["bindings"]}
            intervening_genes = {gene for alias, gene in gene_aliases.items()
                                if prior and re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                                                       text[prior[-1]["span"][1]:assertion["span"][0]], re.I)}
            intervening_variants = {canonical(match.group()) for match in _VARIANT.finditer(
                text[prior[-1]["span"][1]:assertion["span"][0]])} if prior else set()
            antecedent_variants = {identifier for key in keys for identifier in key[:2]}
            if definite and len(keys) == 1 and intervening_variants <= antecedent_variants and (not intervening_genes or intervening_genes <= {current_owner}) and all(
                    (not populations or str(edge.get("population") or plan.get("population") or "").upper() in populations)
                    and (not builds or str(edge.get("genome_build") or plan.get("genome_build") or "").upper() in builds)
                    for item in prior for edge in item["bindings"]):
                base = prior[-1]
                finding.update({key: base[key] for key in ("bindings", "values", "explicit_source_ids", "evidence_ids")})
                finding.update(owner_resolved=True, all_bindings_required=True, antecedent_span=base["span"])
                mode = "unique_same_paragraph_reference"
        values = finding["values"]
        lower, upper = _number(assertion["lower"]), _number(assertion["upper"])
        valid_range = lower is not None and 0 <= lower <= 1 and (upper is None or lower <= upper <= 1)

        def satisfies(value):
            if not valid_range:
                return False
            if mode == "configured_threshold":
                # The inclusion cutoff itself must match the saved setting;
                # a looser inequality cannot impersonate that same protocol.
                return upper is None and assertion["operator"] in {"=", ">="} and _rounded_equal(lower, value)
            if upper is not None:
                return (lower <= value or _rounded_equal(lower, value)) and (value <= upper or _rounded_equal(upper, value))
            return {"<": value < lower, "<=": value <= lower, ">": value > lower, ">=": value >= lower}.get(
                assertion["operator"], _rounded_equal(lower, value))

        # No bound authority means unknown, not a numerical mismatch.
        consistent = ((all if finding["all_bindings_required"] else any)(satisfies(value) for value in values)
                      if values and assertion["assertion_type"] != "unparsed" else None)
        explicit = mode in {"explicit_pair", "single_selected_source", "unique_same_paragraph_reference", "scoped_aggregate", "configured_threshold", "maximum_recorded_selected_source_ld"}
        unique_edge = len(finding["bindings"]) == 1
        if assertion["assertion_type"] == "unparsed":
            status = "unclear_binding"
        elif not edges and not values:
            status = "insufficient_data"
        elif finding["owner_resolved"] and not consistent and (explicit or unique_edge) and complete_citation_scope:
            status = "contradicted"
        elif finding["owner_resolved"] and consistent and complete_citation_scope and (explicit or unique_edge):
            status = "verified"
        else:
            status = "unclear_binding"
        finding.update(verification_status=status, numeric_consistent=consistent, binding_mode=mode,
                       can_correct=status == "contradicted" and ((mode in {"explicit_pair", "single_selected_source", "unique_same_paragraph_reference"} and unique_edge)
                           or mode == 'maximum_recorded_selected_source_ld')
                       and assertion["assertion_type"] == "scalar" and not assertion["approximate"],
                       reason={"verified": "ld_assertion_verified", "unclear_binding": "ld_binding_unclear",
                               "contradicted": "ld_evidence_conflict", "insufficient_data": "ld_data_unavailable"}[status])
        if 'numeric_fact_ids' not in finding:
            ids = []
            for fact in fact_rows:
                if fact['scope'] == 'inclusion_threshold' and mode == 'configured_threshold':
                    ids.append(fact['fact_id'])
                if fact['scope'] != 'ld_pair':
                    continue
                source = fact['provenance']
                if any(source['selected_variant_id'] == edge['source'] and source['proxy_variant_id'] == edge['proxy']
                       and _number(fact['value']) == _number(edge['r2'])
                       and source['population'] == str(edge.get('population') or plan.get('population') or 'not_reported')
                       and source['genome_build'] == str(edge.get('genome_build') or plan.get('genome_build') or 'not_reported')
                       for edge in finding['bindings']):
                    ids.append(fact['fact_id'])
            finding['numeric_fact_ids'] = sorted(ids)
        finding['finding_category'] = {'verified': 'verified', 'contradicted': 'numeric_error',
            'unclear_binding': 'relation_ambiguity', 'insufficient_data': 'not_verifiable'}[status]
        results.append(finding)
    return tuple(results)
