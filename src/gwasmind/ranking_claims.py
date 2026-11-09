"""Check ranking assertions against typed retrieved records, never model prose."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation, localcontext
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .audit import sha256_json
from .evidence_citations import inline_evidence_citations
from .models import EvidenceKind, EvidenceRecord, LocusQuery
from .numeric_provenance import NumericFact as RawMeasurement, build_numeric_ledger, source_p_facts

if TYPE_CHECKING:
    from .llm import SynthesisOutput

_MANTISSA = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_EXPONENT = r"(?:\^?\s*\{?[−+-]?\d+\}?|[⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
_NUMBER = _MANTISSA + r"(?:e[+-]?\d+|\s*[×x]\s*10\s*" + _EXPONENT + r")?"
_METRIC = re.compile(
    r"(?<!\w)(?P<metric>P(?:[- ]?value)?|Beta|β|H[34]|PIP|CLPP|odds\s+ratio|OR)\s*"
    r"(?P<relation><=|>=|[<>=≤≥]|:|of\b|is\b|was\b|at\b|~|≈)\s*"
    r"(?P<number>" + _NUMBER + r")(?![\w/:]|\.\d)(?P<percent>\s*%)?", re.I,
)
_COMPANION_EFFECT = re.compile(
    r"(?<!\w)(?P<metric>Beta|β|OR|odds\s+ratio)\s*(?P<relation>=|:|~|≈)?\s*"
    r"(?P<number>" + _NUMBER + r")(?![\w/:]|\.\d)", re.I,
)
_VARIANT = re.compile(r"\b(?:rs\d+|(?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+)\b", re.I)
_CERTAIN_CAUSAL = re.compile(
    r"\b(?:(?:proven|validated|definitive|confirmed)\s+causal|"
    r"(?:conclusively|definitively|definitely|unequivocally|certainly)\s+"
    r"(?:(?:established|demonstrated|identified|confirmed|shown|proven|validated)\s+(?:as\s+)?)?"
    r"(?:the\s+|a\s+)?causal(?:\s+gene)?|"
    r"(?:established|demonstrated|confirmed|proven|validated)\s+as\s+(?:the\s+|a\s+)?causal(?:\s+gene)?|"
    r"(?:demonstrates?|proves?)\s+(?:the\s+)?(?:mechanism|causality)|"
    r"(?:establishes?|confirms?)\s+(?:the\s+)?(?:causal\s+(?:gene|mechanism)|causality)|"
    r"(?:is|are|was|were)\s+(?:the\s+|a\s+)(?:causal|causative)\s+(?:gene|variant))\b", re.I,
)
_UNCERTAIN = re.compile(
    r"\b(?:cannot|can't|not(?!\s+only\b)|never|without|uncertain|unproven|unconfirmed|"
    r"may|might|could|whether|hypothes\w*|potential\w*|putative|possible)\b[^,;.!?]{0,55}$|"
    r"\bno\b.{0,35}\b(?:evidence|proof|basis|support|indication)\b[^,;.!?]{0,55}$", re.I,
)
_EXCLUDED_CAUSAL_BASIS = re.compile(
    r"\brather\s+than\s+by\s+(?:(?:a|an|the)\s+)?$", re.I,
)


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip())


def causal_claim_issues(text: str, *, label: str) -> tuple[str, ...]:
    """Reject asserted causal certainty; keep explicit negation and uncertainty."""
    issues = []
    for sentence in _sentences(re.sub(r"[*`]", "", text)):
        for match in _CERTAIN_CAUSAL.finditer(sentence):
            # A prior, separately contrasted assertion cannot negate this one.
            prefix = re.split(
                r"[;,]|\b(?:but|however|yet|whereas)\b|"
                r"\b(?:and|while)\s+(?=[A-Za-z0-9_-]+\s+(?:is|are|was|were|has|shows)\b)",
                sentence[:match.start()], flags=re.I,
            )[-1]
            if _UNCERTAIN.search(prefix):
                continue
            # "QTL records rather than by a validated causal inference"
            # explicitly excludes that basis. Anchor to this exact predicate;
            # an earlier exclusion cannot excuse a later affirmative claim.
            if _EXCLUDED_CAUSAL_BASIS.search(prefix):
                continue
            # "...; neither establishes causality" negates this predicate.
            # Require adjacency: an earlier neither cannot excuse a later
            # affirmative assertion in the same sentence.
            if re.search(r"\bneither(?:\s+(?:candidate|gene|variant)\s+(?:is|was|has\s+been))?\s*$", prefix, re.I):
                continue
            # A negative quantified subject negates only its own adjacent
            # predicate: "No prioritized gene is proven causal." It cannot
            # excuse an affirmative claim later in the sentence.
            if re.search(
                r"\bno\s+(?:(?:prioriti[sz]ed|candidate|selected|ranked|individual|single)\s+){0,3}"
                r"(?:genes?|variants?|candidates?)(?:\s+(?:is|are|was|were|has\s+been|have\s+been))?\s*$",
                prefix, re.I,
            ):
                continue
            # An abstract assignment that would require further evidence is
            # a future requirement, not a causal assignment already made.
            # Bind both sides to this exact predicate so later affirmative
            # claims cannot inherit the conditional qualification.
            if (
                re.fullmatch(r"definitive\s+causal", match.group(0), re.I)
                and re.fullmatch(r"\s*(?:(?:a|an|the)\s+)?", prefix, re.I)
                and re.match(
                    r"\s+assignment\s+would\s+require(?!\s*(?:no|nothing|neither)\b)\s+\S",
                    sentence[match.end():], re.I,
                )
            ):
                continue
            issues.append(f"{label} makes a disallowed causal-strength claim: {match.group(0).casefold()}")
    return tuple(dict.fromkeys(issues))


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        text = str(value).replace("−", "-")
        product = re.fullmatch(r"(" + _MANTISSA + r")\s*[×x]\s*10\s*(" + _EXPONENT + r")", text, re.I)
        if product:
            exponent = product.group(2).translate(str.maketrans("⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹", "-+0123456789")).strip("^{} ")
            text = product.group(1) + "e" + exponent
        number = Decimal(text)
        return number if number.is_finite() and abs(number.adjusted()) <= 1000 else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _normal(text: str) -> str:
    return " ".join(text.casefold().replace("−", "-").replace("-", " ").split())


def _trait_key(text: str) -> str:
    value = _normal(text)
    # A terminal PheCode identifier is label metadata, not a phenotype subtype.
    # Keep every other parenthetical qualifier (onset, adjustment, etc.) intact.
    value = re.sub(r"\s+\(phecode\s+\d+(?:\.\d+)?\)\Z", "", value)
    onset = re.fullmatch(r"asthma\s*\((adult|childhood)\s+onset\)", value)
    return onset.group(1) + " onset asthma" if onset else value


def _trait_label_aliases(text: str) -> set[str]:
    """Equivalent label punctuation retains every scientific qualifier."""
    normal = _normal(text)
    aliases = {normal, _trait_key(text)}
    qualified = re.fullmatch(r"(.+?),\s+([^,()]+)", normal)
    if qualified:
        aliases.add(qualified[1] + " (" + qualified[2] + ")")
    return aliases


def _declared_trait_reference(prefix: str, trait: str | None,
                              mentions: Sequence[tuple[int, int, str]],
                              bindings: Sequence[RawMeasurement]) -> str | None:
    """Resolve 'the <base trait> record' only to a uniquely declared full label.

    A shortened reference does not remove a subtype qualifier. Its complete
    label must already be explicitly present in this sentence and in the cited
    evidence. Multiple declared subtypes remain unresolved.
    """
    if trait is None or not mentions:
        return trait
    latest = max(mentions, key=lambda item: item[0])
    if not re.match(r"\s+(?:records?|pairs?|colocali[sz]ation)\b", prefix[latest[1]:], re.I):
        return trait
    declared = set()
    for item in bindings:
        key = _trait_key(item.trait)
        if not (key.startswith(trait + ", ") or key.startswith(trait + " (")):
            continue
        if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", prefix[:latest[0]])
               for alias in _trait_label_aliases(item.trait)):
            declared.add(key)
    if len(declared) == 1:
        return next(iter(declared))
    return "unresolved multiple declared trait subtypes" if declared else trait


def _variant(value: str) -> str:
    return value.casefold().removeprefix("chr").replace(":", "_")


def _metric_name(value: str) -> str:
    name = value.casefold().replace("-", "").replace(" ", "")
    return "p" if name in {"pvalue", "p"} else "beta" if name == "β" else "or" if name == "oddsratio" else name


def _parenthetical_bounds(text: str, position: int) -> tuple[int, int] | None:
    """Find the balanced parenthesis containing this token without splitting its fields."""
    stack = []
    for index, character in enumerate(text):
        if character == "(":
            stack.append(index)
        elif character == ")" and stack:
            opening = stack.pop()
            if opening < position < index:
                return opening, index + 1
    return None


def _parenthetical_measurement_scope(sentence: str, match: re.Match[str]):
    bounds = _parenthetical_bounds(sentence, match.start())
    if bounds is None:
        return sentence, match, ""
    opening, end = bounds
    # An earlier complete parenthesis is a separate record/LD annotation. Its
    # source partner or assay trait cannot become this statistic's owner.
    begin = sentence.rfind(")", 0, opening) + 1
    local = sentence[begin:end]
    local_match = _METRIC.match(local, match.start() - begin)
    return local, local_match, sentence[opening + 1:end - 1]


def _cited_qtl_pip_scope(sentence: str, match: re.Match[str], *, owner: str | None):
    """A new, terminal QTL-PIP clause cannot inherit an earlier regulatory SNP.

    Require an identified gene, a separate clause explicitly naming QTL PIP,
    and an immediately adjacent citation-only parenthesis. Declared tissues,
    variants or other qualifications are never removed to obtain a match.
    """
    if not owner or _metric_name(match.group("metric")) != "pip":
        return None
    boundary = re.search(
        r"[,;]\s*(?:and\s+)?(?:its|[A-Za-z][A-Za-z0-9_-]*['\u2019]s)\s+"
        r"(?:(?:top|highest|maximum|recorded|reported)\s+)*QTL\s*$",
        sentence[:match.start()], re.I,
    )
    citation = re.fullmatch(r"\s*\((E[1-9][0-9]*(?:\s*[,;]\s*E[1-9][0-9]*)*)\)\s*[.!?]?\s*",
                            sentence[match.end():])
    if boundary is None or citation is None:
        return None
    begin = boundary.start() + 1
    local = sentence[begin:]
    return local, _METRIC.match(local, match.start() - begin), citation[1]


def _cited_qtl_measurement_after_ld_scope(sentence: str, match: re.Match[str], *,
        owner: str | None, query: LocusQuery, records: Sequence[EvidenceRecord], context):
    """Keep a cited QTL measurement on its member after a separate LD tuple.

    Only a closed LD parenthesis followed by ``measured at Beta, P (E...)``
    qualifies. A verified explicit pair identifies the secondary source SNP;
    its identifier is masked for measurement ownership, never for the LD check.
    The cited QTL must independently match the gene, member and stated tissue.
    """
    if not owner or _metric_name(match["metric"]) not in {"p", "beta"}:
        return None
    markers = list(re.finditer(r"\)\s*,\s*measured\s+at\s+", sentence[:match.start()], re.IGNORECASE))
    if not markers:
        return None
    marker = markers[-1]
    bounds = _parenthetical_bounds(sentence, marker.start() - 1)
    if bounds is None or bounds[1] != marker.start() + 1:
        return None
    cursor, measurements = marker.end(), []
    while (measurement := _METRIC.match(sentence, cursor)) is not None:
        if _metric_name(measurement["metric"]) not in {"p", "beta"}:
            return None
        measurements.append(measurement)
        cursor = measurement.end()
        citation = re.match(r"\s*\((E[1-9][0-9]*(?:\s*[,;]\s*E[1-9][0-9]*)*)\)", sentence[cursor:])
        if citation:
            break
        separator = re.match(r"\s*[,;]\s*", sentence[cursor:])
        if separator is None:
            return None
        cursor += separator.end()
    else:
        return None
    if not any(item.start() == match.start() for item in measurements):
        return None
    # A qualifier after the citation still constrains this measurement; it
    # must not disappear merely because the citation follows the numeric tuple.
    if re.match(r"\s*(?:[,;]\s*)?(?:for|at|in|with)\b",
                sentence[cursor + citation.end():], re.IGNORECASE):
        return None
    tokens = re.findall(r"\bE\d+\b", citation[1])
    if any(token not in context.citation_aliases for token in tokens):
        return None
    cited_ids = {context.citation_aliases[token] for token in tokens}
    cited = tuple(row for row in records if row.evidence_id in cited_ids)
    if len(cited) != 1 or cited[0].kind is not EvidenceKind.QTL or cited[0].gene_id != owner:
        return None
    row = cited[0]
    tissue = row.tissue or _text(_map(row.payload.get("context")).get("biosample"))
    if not tissue or not re.search(r"(?<!\w)" + re.escape(tissue)
            + r"\s+(?:cis[- ]?)?(?:e|s|p|tu)?QTL\b", sentence[:bounds[0]], re.IGNORECASE):
        return None
    from .native_source_assertions import native_ld_numeric_bindings
    findings = native_ld_numeric_bindings(sentence[bounds[0]:bounds[1]], query=query,
        records=tuple(records), aliases=context.citation_aliases, gene_aliases=context.aliases, owner=owner)
    if len(findings) != 1 or not findings[0]["owner_resolved"] or len(findings[0]["explicit_source_ids"]) != 1:
        return None
    normalizations = {_variant(str(key)): _variant(str(value)) for key, value in
                      _map(_map(query.ld_expansion).get("resolved_source_aliases")).items()}
    def canonical(value):
        identifier = _variant(value)
        return normalizations.get(identifier, identifier)
    proxies = {edge["proxy"] for edge in findings[0]["bindings"]}
    if proxies != {canonical(row.variant_id or "")}:
        return None
    source = findings[0]["explicit_source_ids"][0]
    local = sentence
    for identifier in reversed(list(_VARIANT.finditer(sentence, bounds[0], bounds[1]))):
        if canonical(identifier.group()) == source:
            local = local[:identifier.start()] + " " * len(identifier.group()) + local[identifier.end():]
    return local, _METRIC.match(local, match.start()), citation[1]


def raw_measurements(record: EvidenceRecord) -> tuple[RawMeasurement, ...]:
    """Read the declared measurement owner without borrowing lead/member values."""
    payload = _map(record.payload)
    measurements = _map(payload.get("measurements"))
    context = _map(payload.get("context"))
    raw = _map(_map(payload.get("upstream")).get("raw"))
    member = _map(raw.get("matchedLocusMember"))
    lead_scope = (context.get("measurement_scope") == "credible_set_lead"
                  or raw.get("measurementScope") == "credible_set_lead")
    lead = _map(raw.get("csLeadMeasurement"))
    lead_ids = {str(value) for value in (context.get("measured_variant_id"), context.get("cs_lead_variant_id"),
        raw.get("measuredVariantId"), raw.get("csLeadVariantId"), lead.get("variant_id")) if value}
    valid_lead = lead_scope and len(lead_ids) == 1 and lead.get("measurement_scope") == "credible_set_lead"
    is_member = raw.get("queryVariantRole") == "credible_set_member" or bool(
        member and member.get("variantId") == record.variant_id
        and raw.get("leadVariantId") != record.variant_id
    )
    trait = next((_text(obj.get(key)) for obj, key in (
        (raw, "rightTraitFromSource") if record.kind.value == "colocalization" else (raw, "traitFromSource"),
        (context, "trait"), (context, "trait_from_source"), (payload, "trait"),
    ) if _text(obj.get(key))), "")
    result = []
    keys = {"p": ("p_value", "pValue", "pvalue"), "beta": ("beta",),
            "h4": ("h4", "H4"), "h3": ("h3", "H3"),
            "pip": ("posteriorProbability", "pip", "posterior_probability"),
            "clpp": ("clpp",), "or": ("oddsRatio", "odds_ratio")}
    for metric, aliases in keys.items():
        if lead_scope and metric in {"p", "beta", "pip", "or"}:
            containers = (("payload.upstream.raw.csLeadMeasurement", lead),) if valid_lead else ()
        elif is_member and metric in {"p", "beta", "pip", "or"}:
            containers = (("payload.upstream.raw.matchedLocusMember", member), ("payload.measurements", measurements))
        else:
            containers = (("payload.upstream.raw", raw), ("payload.measurements", measurements), ("payload", payload))
        value = None
        source_field = ""
        for path, container in containers:
            candidates = [_number(container[key]) for key in aliases if key in container]
            candidates = [number for number in candidates if number is not None]
            source_field = path + "." + "+".join(key for key in aliases if key in container)
            if metric == "p" and not candidates:
                mantissa, exponent = _number(container.get("pValueMantissa")), _number(container.get("pValueExponent"))
                if mantissa is not None and exponent is not None and exponent == int(exponent) and -1000 <= exponent <= 0:
                    candidates = [mantissa * Decimal(10) ** int(exponent)]
                    source_field = path + ".pValueMantissa*p10(pValueExponent)"
            if candidates:
                if len(set(candidates)) == 1:
                    value = candidates[0]
                break
        if value is None or metric in {"p", "h4", "h3", "pip", "clpp"} and not 0 <= value <= 1:
            continue
        measured_variant = next(iter(lead_ids)) if valid_lead and metric in {"p", "beta", "pip", "or"} else record.variant_id
        scope = "credible_set_lead" if valid_lead and metric in {"p", "beta", "pip", "or"} else (
            "credible_set_member" if is_member and metric in {"p", "beta", "pip", "or"} else "colocalisation_pair" if metric in {"h3", "h4"} else "measurement")
        provenance = {"source_field": source_field, "source_record_sha256": sha256_json(record.to_dict()),
            "source": record.source, "dataset": record.dataset, "data_release": record.data_release,
            "study_id": raw.get("studyId") or context.get("study_id") or record.study_id,
            "study_locus_id": raw.get("studyLocusId") or context.get("study_locus_id"),
            "other_study_id": raw.get("rightStudyId"), "other_study_locus_id": raw.get("rightStudyLocusId"),
            "method": raw.get("colocalisationMethod"),
            "biosample_id": raw.get("biosampleId") or context.get("biosample_id"),
            "genome_build": context.get("genome_build") or raw.get("genomeBuild")}
        result.append(RawMeasurement(record.evidence_id, _text(record.gene_id), _variant(_text(measured_variant)), trait, metric, value,
            scope, tuple(sorted((key, str(item)) for key, item in provenance.items() if item is not None))))
    return tuple(result)


def _gene_aliases(records: Sequence[EvidenceRecord]) -> dict[str, str]:
    aliases: dict[str, set[str]] = {}
    for record in records:
        gene = _text(record.gene_id)
        if not gene:
            continue
        payload = _map(record.payload)
        raw = _map(_map(payload.get("upstream")).get("raw"))
        names = [gene]
        for key in ("subject", "object"):
            entity = _map(payload.get(key))
            if entity.get("type") == "target" and entity.get("id") == gene:
                names.append(_text(entity.get("label")))
        for id_key, symbol_key in (("id", "approvedSymbol"), ("geneId", "approvedSymbol"), ("qtlGeneId", "qtlGeneSymbol"), ("rightQtlGeneId", "rightQtlGeneSymbol")):
            if raw.get(id_key) == gene:
                names.append(_text(raw.get(symbol_key)))
        for name in names:
            if name:
                aliases.setdefault(name.casefold(), set()).add(gene)
    return {name: next(iter(genes)) for name, genes in aliases.items() if len(genes) == 1}


def _mentions(text: str, aliases: Mapping[str, str]) -> list[tuple[int, int, str]]:
    matches = [(match.start(), match.end(), owner) for alias, owner in aliases.items()
               for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I)]
    return [item for item in matches if not any(other[0] <= item[0] and other[1] >= item[1]
            and other[1]-other[0] > item[1]-item[0] for other in matches)]


def _rounded_equal(reported: Decimal, actual: Decimal) -> bool:
    if reported == actual:
        return True
    if reported == reported.to_integral() and -1 <= reported <= 1:
        return False
    try:
        with localcontext() as ctx:
            ctx.prec = 40
            return actual.quantize(Decimal(1).scaleb(reported.as_tuple().exponent)) == reported
    except InvalidOperation:
        return False


def _measurement_matches(match: re.Match[str], reported: Decimal, actual: Decimal) -> bool:
    relation = match.group("relation")
    if relation in {"<", "<=", "≤", ">", ">=", "≥"}:
        return {"<":actual < reported, "<=":actual <= reported, "≤":actual <= reported,
                ">":actual > reported, ">=":actual >= reported, "≥":actual >= reported}[relation]
    return _rounded_equal(reported, actual)


def _measurement_candidates(sentence: str, match: re.Match[str], *, bindings: Sequence[RawMeasurement],
                            aliases: Mapping[str, str], traits: Mapping[str, str], owner: str | None) -> tuple[RawMeasurement, ...]:
    prefix, suffix = sentence[:match.start()], sentence[match.end():]
    before_genes = _mentions(prefix, aliases)
    gene = max(before_genes, default=(0, 0, owner), key=lambda item:item[0])[2]
    variants = list(_VARIANT.finditer(prefix))
    variant = _variant(variants[-1].group()) if variants else None
    following_variant = _VARIANT.match(suffix.lstrip())
    trailing_variant = re.match(r"\s+(?:for|at|in)\s+(" + _VARIANT.pattern + r")", suffix, re.I)
    if following_variant or trailing_variant:
        variant = _variant((following_variant or trailing_variant).group(0).split()[-1])
    normalized_prefix = _normal(prefix)
    local_traits = dict(traits)
    # A complete qualified label can introduce its own later short reference,
    # even when no unrelated record happens to provide the unqualified alias.
    # Resolution below still requires that one full label to be unique.
    for label in {item.trait for item in bindings}:
        key = _trait_key(label)
        qualified = re.fullmatch(r"(.+?)(?:,\s+[^,()]+|\s+\([^()]+\))", key)
        if qualified and any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized_prefix)
                             for alias in _trait_label_aliases(label)):
            local_traits.setdefault(qualified[1], qualified[1])
    before_traits = _mentions(normalized_prefix, local_traits)
    trait = max(before_traits, default=(0, 0, None), key=lambda item:item[0])[2]
    trait = _declared_trait_reference(normalized_prefix, trait, before_traits, bindings)
    if trait is None:
        # In the explicit "has <phenotype> [QTL/coloc] P/H4" slot, an unknown
        # phenotype is a constraint, not permission to borrow any cited trait.
        # Restrict this to the adjacent predicate; never infer a phenotype from
        # arbitrary narrative words or a source-GWAS parenthesis.
        stated = re.search(r"\b(?:has|shows|exhibits)\s+([a-z][a-z0-9 -]{0,100})\s*$", normalized_prefix)
        if stated:
            qualifiers = {"a", "an", "the", "its", "external", "retrieved", "reported", "published",
                "qtl", "eqtl", "sqtl", "coloc", "colocalization", "colocalisation", "gwas",
                "association", "associations", "signal", "evidence", "support", "with", "and", "record", "records",
                "strong", "stronger", "weak", "weaker", "moderate", "high", "higher", "low", "lower",
                "clear", "consistent", "robust", "significant", "nominal", "independent", "direct",
                "molecular", "genetic", "convincing", "compelling", "genome", "wide"}
            description = re.sub(r"\bsame trait\b", "", stated[1])
            label = " ".join(word for word in description.split() if word not in qualifiers)
            if label:
                trait = "unbound declared trait: " + label
    trailing = re.match(r"\s*(?:for|in|with)\s+(.+)", suffix, re.I)
    if trailing:
        # Do not use a subsequent numeric assertion to label this value.
        tail = _METRIC.split(trailing.group(1))[0]
        normalized_tail = _normal(tail)
        following_traits = [item for item in _mentions(normalized_tail, traits)
                            if re.fullmatch(r"(?:(?:the|trait|phenotype)\s+)*",
                                            normalized_tail[:item[0]])]
        if following_traits:
            trait = min(following_traits, key=lambda item:item[0])[2]
    trailing_gwas = re.match(r"\s*,\s*([^();,]+?)\s+GWAS\b", suffix, re.I)
    if trailing_gwas:
        # A post-value GWAS label is still an explicit phenotype constraint.
        # Unknown labels must not fall through to an unrestricted source P.
        trait = _trait_key(trailing_gwas[1])
    labelled = re.search(r"\busing\s+(?:the\s+)?GWAS\s+label\s+(['\"])([^'\"]+)\1", suffix, re.I)
    if labelled and not (_mentions(suffix[:labelled.start()], aliases)
                         or _VARIANT.search(suffix[:labelled.start()])
                         or re.search(r"[;]|\b(?:but|whereas|however)\b", suffix[:labelled.start()], re.I)):
        explicit = _trait_key(labelled[2])
        # The full quoted label may follow a joint H4/CLPP tuple. It does not
        # override a conflicting phenotype explicitly declared before it.
        trait = explicit if trait is None or trait == explicit else "conflicting explicit trait labels"
    metric = _metric_name(match.group("metric"))
    return tuple(item for item in bindings if item.metric == metric
                 and (gene is None or item.gene_id == gene)
                 and (variant is None or item.variant_id == variant)
                 and (trait is None or _trait_key(item.trait) == trait))


@dataclass(frozen=True)
class _AssertionContext:
    aliases: Mapping[str, str]
    traits: Mapping[str, str]
    measurements: Mapping[int, tuple[RawMeasurement, ...]]
    citation_aliases: Mapping[str, str]
    variant_ids: frozenset[str]
    numeric_ledger: Mapping


def _assertion_context(query: LocusQuery, records: Sequence[EvidenceRecord], output=None) -> _AssertionContext:
    """One complete, read-only context per validation/repair call, never per run."""
    aliases = _gene_aliases(records)
    aliases.update({gene.casefold(): gene for gene in query.candidate_genes if isinstance(gene, str)})
    measurements = {id(record): raw_measurements(record) for record in records}
    trait_names = {item.trait for values in measurements.values() for item in values if item.trait} | {
        query.trait, query.source_trait or query.trait}
    traits = {alias: _trait_key(trait) for trait in trait_names for alias in _trait_label_aliases(trait)}
    # QTL target IDs sometimes occupy an upstream trait-label field. An
    # explicitly recognised gene ID/symbol is a gene owner in report prose,
    # not a phenotype constraint on a later colocalization measurement.
    for alias in aliases:
        traits.pop(_normal(alias), None)
    # These conventional abbreviations denote the unqualified disease only;
    # they never erase onset or other scientific subtype qualifiers.
    for number in (1, 2):
        base = f"type {number} diabetes"
        if traits.get(base) == base:
            traits[f"t{number}d"] = base
    references = {}
    if output is not None:
        prose = " ".join((output.summary, *(row.rationale for row in output.ranked_genes),
                          *(text for row in output.ranked_genes for text in row.limitations),
                          *(row.reason for row in output.candidate_decisions),
                          *(row.text for row in output.claims)))
        if re.search(r"\bE\d+\b", prose):
            visible = set((row.evidence_id for row in records) if output.model_visible_evidence_ids is None
                          else output.model_visible_evidence_ids)
            known = {row.evidence_id for row in records}
            if output.model_evidence_aliases is not None:
                # Presentation aliases are frozen before inference. Rebuilding a
                # canonical order here would validate against different records.
                references = {key: value for key, value in output.model_evidence_aliases.items()
                              if re.fullmatch(r"E[1-9][0-9]*", key) and value in visible and value in known}
            else:
                from .llm import _candidate_gene_universe
                from .scientific_facts import build_scientific_fact_view
                ordered = tuple(sorted(records, key=lambda row: (row.gene_id or "", row.kind.value,
                    row.study_id or "", row.credible_set_id or "", row.variant_id or "", row.evidence_id)))
                view = build_scientific_fact_view(query, ordered,
                    candidate_genes=tuple(sorted(_candidate_gene_universe(query, tuple(records)))))
                references = {key: value for key, value in view.evidence_aliases.items() if value in visible}
    variant_ids = {_variant(row.variant_id) for row in (*query.variants, *query.analysis_variants)}
    for row in records:
        if row.variant_id:
            variant_ids.add(_variant(row.variant_id))
        raw = _map(_map(row.payload.get("upstream")).get("raw"))
        variant_ids.update(_variant(str(raw[key])) for key in ("variantId", "queryVariantId", "leadVariantId") if raw.get(key))
    plan = _map(query.ld_expansion)
    normalizations = _map(plan.get("resolved_source_aliases"))
    variant_ids.update(_variant(str(value)) for value in (*normalizations, *normalizations.values()))
    for edge in plan.get("ld_links", []):
        if isinstance(edge, Mapping):
            variant_ids.update(_variant(str(edge[key])) for key in ("selected_variant_id", "proxy_variant_id") if edge.get(key))
    return _AssertionContext(MappingProxyType(aliases), MappingProxyType(traits),
                             MappingProxyType(measurements), MappingProxyType(references), frozenset(variant_ids),
                             build_numeric_ledger(query, records, measurements=measurements))


def _explicit_parenthetical_source_p_owner(sentence: str, match: re.Match[str], query: LocusQuery) -> str | None:
    """Bind a single source-P parenthesis before later external annotations."""
    bounds = _parenthetical_bounds(sentence, match.start())
    if bounds is not None and _metric_name(match.group("metric")) == "p":
        opening, end = bounds
        prefix = sentence[:opening]
        # A short P field of an explicitly identified source variant is the
        # uploaded observation. Molecular/external statistics remain separate.
        source_marker = re.search(r"\b(?:source(?:[- ]selected)?|uploaded|loaded|selected|original)"
                                  r"[- ]+(?:genome[- ]wide[- ]significant\s+)?"
                                  r"(?:(?P<trait>[A-Za-z][A-Za-z0-9 ()-]{0,100}?)\s+)?"
                                  r"(?:variant|SNP)s?\b", prefix, re.I)
        if source_marker is None:
            source_marker = re.search(r"\b(?:uploaded|loaded|source|current)\s+"
                r"(?:(?P<trait>[A-Za-z][A-Za-z0-9 -]{0,80}?)\s+)?(?:GWAS|signal|scan)\s*:\s*"
                + _VARIANT.pattern + r"\s*$", prefix, re.I)
        external = re.search(r"\b(?:[a-z-]*QTL|colocali[sz]\w*|external|published)\b", sentence[:end], re.I)
        identifiers = {_variant(item.group()) for item in _VARIANT.finditer(sentence[:match.start()])}
        tail = re.split(r"[;,]", sentence[opening + 1:match.start()])[-1].strip()
        tail = _VARIANT.sub("", tail).strip()
        source_traits = {_trait_key(value) for value in _trait_label_aliases(query.source_trait or query.trait)}
        # A phenotype-qualified source label still identifies the uploaded
        # observation: "source-selected asthma variant (ID, P=...)". Require
        # the exact uploaded phenotype; a subtype or unrelated trait cannot
        # borrow that observation merely because the variant/value matches.
        source_qualifier = source_marker['trait'] if source_marker else None
        if source_qualifier and re.fullmatch(r'(?:source(?:[- ]selected)?|selected|original)', source_qualifier, re.I):
            source_qualifier = None
        elif source_qualifier and _trait_key(source_qualifier) not in source_traits:
            # "Uploaded asthma source variant" names the same uploaded
            # phenotype as "source-selected asthma variant". Remove only
            # the terminal identity qualifier; the remaining phenotype must
            # still match exactly, including any onset/subtype qualifiers.
            source_qualifier = re.sub(r'\s+(?:source(?:[- ]selected)?|selected)$', '',
                                      source_qualifier, flags=re.I)
        if source_marker and source_qualifier and _trait_key(source_qualifier) not in source_traits:
            source_marker = None
        declared = re.sub(r"^(?:uploaded|loaded|source|current)\s+|\s*GWAS$", "", tail, flags=re.I).strip()
        after = re.match(r"\s*,\s*([^();,]+?)\s+GWAS\b", sentence[match.end():end], re.I)
        trait_ok = (not declared or declared.casefold() in {"uploaded", "loaded", "source", "current"}
                    or _trait_key(declared) in source_traits)
        trait_ok = trait_ok and (after is None or _trait_key(after[1]) in source_traits)
        source_ids = {_variant(row.variant_id) for row in query.variants}
        if source_marker and not external and trait_ok and len(identifiers) == 1 and identifiers <= source_ids:
            return next(iter(identifiers))
    if _metric_name(match.group("metric")) != "p" or not re.match(r"\s*\)", sentence[match.end():]):
        return None
    prefix = re.search(
        r"\b(?:loaded|uploaded|source|current)[- ]GWAS\s+"
        r"(?:(?:source[- ]selected|selected)\s+)?(?:variant|SNP)\s*,?\s+(?P<variant>"
        + _VARIANT.pattern + r")\s*\(\s*(?P<trait>[^();]{0,120})$",
        sentence[:match.start()], re.I,
    )
    if prefix is None:
        # A source-selected identifier can introduce the same explicit source
        # statistic with the GWAS label inside its parenthesis. Keep the
        # phenotype and identifier checks below; external-study P values must
        # never borrow a matching number from another uploaded variant.
        prefix = re.search(
            r"\b(?:loaded|uploaded|source(?:[- ]selected)?|selected|original|index)\s+"
            r"(?:variant|SNP)\s*,?\s+(?P<variant>" + _VARIANT.pattern
            + r")\s*\(\s*(?:(?:loaded|uploaded|source|current)[- ]+)?"
            r"(?P<trait>[^();]{0,120}?)\bGWAS\s*$",
            sentence[:match.start()], re.I,
        )
    if prefix is None:
        # A selected source identifier followed by the exact uploaded trait
        # explicitly owns its parenthesized P, even without repeating "GWAS".
        prefix = re.search(
            r"\b(?:source(?:[- ]selected)?|selected(?:[- ]source)?|uploaded|original|index)\s+"
            r"(?:variant|SNP)\s*,?\s+(?P<variant>" + _VARIANT.pattern
            + r")\s*\(\s*(?P<trait>[^();]{1,120})$",
            sentence[:match.start()], re.I,
        )
    if prefix is None:
        prefix = re.search(r"(?P<variant>" + _VARIANT.pattern + r")\s*\(\s*"
                           r"(?P<trait>uploaded|loaded|source|current)\s*$",
                           sentence[:match.start()], re.I)
    if prefix is None:
        return None
    declared_trait = _normal(prefix.group("trait")).strip()
    if declared_trait in {"uploaded", "loaded", "source", "current"}:
        declared_trait = ""
    source_traits = {_normal(value) for value in _trait_label_aliases(query.source_trait or query.trait)}
    if declared_trait and declared_trait not in source_traits:
        return None
    identifier = _variant(prefix.group("variant"))
    return identifier if any(_variant(row.variant_id) == identifier for row in query.variants) else None


def _explicit_variant_link_issues(sentence: str, records: Sequence[EvidenceRecord],
                                  query: LocusQuery, *, label: str) -> tuple[str, ...]:
    """An asserted exact overlap must use its cited record's actual variant."""
    selected_variant = (r"(?:(?:the|strongest|source[- ]selected|selected|source|uploaded)\s+)*"
                        r"(?:variant|SNP)\s*,?\s+(?P<variant>" + _VARIANT.pattern + r")")
    patterns = [r"\bexact[- ]overlap\s+(?:of|with)\s+" + selected_variant]
    # A proxy-mediated link is not a claim of direct overlap at the source SNP.
    # Keep those explanations under their existing LD/proxy validators.
    if not re.search(r"\b(?:LD|proxy|proxies|indirect|correlated)\b", sentence, re.I):
        patterns.append(r"\bdirect\s+regulatory\s+link\s+(?:at|to|with)\s+" + selected_variant)
    aliases = _map(_map(query.ld_expansion).get("resolved_source_aliases"))
    canonical_aliases = {_variant(str(key)): _variant(str(value)) for key, value in aliases.items()}
    def canonical(identifier):
        value = _variant(identifier)
        return canonical_aliases.get(value, value)

    known_ids = {row.evidence_id for row in records}
    regulatory = bool(re.search(r"\b(?:regulatory|E2G)\b", sentence, re.I))
    issues = []
    selected_ids = {canonical(row.variant_id) for row in query.variants}
    for match in re.finditer(r"\b(?:LD[- ]proxy|proxy)\s+(?:variant|SNP)\s+(`?)("
                             + _VARIANT.pattern + r")", sentence, re.I):
        if canonical(match[2]) in selected_ids:
            issues.append(f"{label} labels selected-source variant {match[2]} as an LD proxy")
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, re.I):
            cited = inline_evidence_citations(sentence[:match.start()], known_ids)
            linked = tuple(row for row in records if not cited or row.evidence_id in cited)
            owners = {canonical(row.variant_id) for row in linked if row.variant_id
                      and (not regulatory or row.kind is EvidenceKind.REGULATORY)}
            if canonical(match["variant"]) not in owners:
                issues.append(f"{label} asserts an exact/direct variant link for {match['variant']} "
                              "without a matching cited variant record")
    return tuple(dict.fromkeys(issues))


def _prose_check(text: str, *, label: str, records: Sequence[EvidenceRecord], all_records: Sequence[EvidenceRecord],
                 query: LocusQuery, owner: str | None = None, repair: bool = False,
                 _context: _AssertionContext | None = None,
                 corrections: list[dict[str, Any]] | None = None,
                 unverified: list[dict[str, Any]] | None = None,
                 numeric_trace: list[dict[str, Any]] | None = None,
                 correction_authorized: bool = True) -> tuple[str, tuple[str, ...]]:
    context = _context or _assertion_context(query, all_records)
    aliases, traits = context.aliases, context.traits
    bindings = tuple(item for record in records for item in (
        context.measurements[id(record)] if id(record) in context.measurements else raw_measurements(record)))
    # Gate 2 checks record identities and measurements, not the model's tone or
    # strength of interpretation. Never rewrite or delete explanatory prose.
    errors = []
    edits: list[tuple[int, int, str]] = []
    offset = 0
    for sentence in _sentences(text):
        start = text.find(sentence, offset)
        offset = start + len(sentence)
        variant_link_issues = _explicit_variant_link_issues(sentence, records, query, label=label)
        errors.extend(variant_link_issues)
        entity_issue = bool(variant_link_issues)
        for reference in re.findall(r"\bE\d+\b", sentence):
            if reference not in context.citation_aliases and reference.casefold() not in aliases:
                errors.append(f"{label} cites an unknown or hidden model-visible evidence alias: {reference}")
                entity_issue = True
        candidates_genes = {_text(record.gene_id) for record in records if record.gene_id}
        for begin, end, gene in _mentions(sentence, aliases):
            assertion = re.match(r"\s+(?:has|shows|exhibits|is\s+(?:associated|linked)|colocali[sz]es)\b", sentence[end:], re.I) or re.search(
                r"\b(?:(?:linked|assigned|connected)\s+to|(?:association|regulation|expression)\s+of|regulates|targets)\s*$",
                sentence[:begin], re.I,
            )
            if records and gene not in candidates_genes and assertion:
                errors.append(f"{label} asserts evidence for {gene} without a matching cited gene record")
                entity_issue = True
        for unknown in re.finditer(r"\b([A-Z][A-Z0-9_-]{1,40})\s+(?:has|shows|exhibits|is\s+(?:associated|linked))\b", sentence):
            gene = unknown.group(1)
            if gene.casefold() not in aliases and gene not in {"GWAS", "QTL", "LD", "DNA", "RNA", "SNP", "H4", "PIP", "CLPP"}:
                errors.append(f"{label} asserts evidence for an unbound gene symbol: {gene}")
                entity_issue = True
        for identifier in re.findall(r"\bENSG\d{6,}(?:\.\d+)?\b", sentence, re.I):
            if identifier.casefold() not in aliases:
                errors.append(f"{label} contains an unbound Ensembl gene ID: {identifier}")
                entity_issue = True
        for variant in _VARIANT.finditer(sentence):
            if _variant(variant.group()) not in context.variant_ids:
                errors.append(f"{label} contains an unbound variant ID: {variant.group()}")
                entity_issue = True
        for match in _METRIC.finditer(sentence):
            reported = _number(match.group("number"))
            percent = bool(match.group("percent"))
            percent_allowed = _metric_name(match.group("metric")) in {"p", "pip", "h3", "h4", "clpp"}
            if reported is not None and percent:
                reported = reported / 100 if percent_allowed else None
            from .numeric_assertions import selection_threshold_assessment
            metric = _metric_name(match.group("metric"))
            threshold = selection_threshold_assessment(sentence, match, metric, context.numeric_ledger)
            if threshold is not None and not entity_issue:
                span = [start + match.start(), start + match.end()]
                if numeric_trace is not None:
                    numeric_trace.append({"field": label, "metric": metric, "value": match["number"],
                        "span": span, **threshold})
                if threshold["status"] not in {"verified", "not_a_point_measurement"}:
                    errors.append(f"{label} has an unsupported {metric} selection threshold")
                    if unverified is not None:
                        unverified.append({"field": label, "metric": metric, "original_value": match["number"],
                            "display_replacement": "Original statement retained for review", "span": span,
                            "reason": threshold["reason"], "numeric_fact_ids": threshold["numeric_fact_ids"],
                            "rule": "typed_selection_threshold_v1", "original_field_sha256": sha256_json(text)})
                continue
            preceding_genes = _mentions(sentence[:match.start()], aliases)
            measurement_owner = owner if owner is not None else (
                max(preceding_genes, key=lambda item: item[0])[2] if preceding_genes else None)
            cited_qtl_scope = _cited_qtl_pip_scope(sentence, match, owner=measurement_owner)
            if cited_qtl_scope is None:
                cited_qtl_scope = _cited_qtl_measurement_after_ld_scope(sentence, match,
                    owner=measurement_owner, query=query, records=records, context=context)
            local_sentence, local_match, parenthetical = (
                cited_qtl_scope
                or _parenthetical_measurement_scope(sentence, match))
            cited_ids = {context.citation_aliases[alias] for alias in re.findall(r"\bE\d+\b", parenthetical)
                         if alias in context.citation_aliases}
            cited_ids.update(inline_evidence_citations(parenthetical, {row.evidence_id for row in records}))
            local_bindings = tuple(item for item in bindings
                                   if (not cited_ids and cited_qtl_scope is None) or item.evidence_id in cited_ids)
            # A later parenthesis may abbreviate a gene named before an
            # earlier parenthesis. Do not turn that scoped statistic into a
            # search across every gene cited by a global summary.
            allowed = _measurement_candidates(local_sentence, local_match, bindings=local_bindings,
                                              aliases=aliases, traits=traits, owner=measurement_owner)
            # P values from different evidence kinds remain separate even if
            # their gene and numeric value happen to coincide.
            local_prefix = local_sentence[:local_match.start()]
            declared_kind = (EvidenceKind.QTL if re.search(r"\b(?:[es]?QTL)\s*$", local_prefix, re.I) else
                             EvidenceKind.COLOCALIZATION if re.search(r"\bcolocali[sz]ation\s*$", local_prefix, re.I) else
                             EvidenceKind.GWAS if re.search(r"\bGWAS\s*$", local_prefix, re.I) else None)
            if declared_kind is not None:
                kind_ids = {row.evidence_id for row in records if row.kind is declared_kind}
                allowed = tuple(item for item in allowed if item.evidence_id in kind_ids)
            declared_studies = set(re.findall(
                r"\b(?:study|dataset)\s+(GCST\d+|[A-Z][A-Z0-9]*(?:[-_][A-Za-z0-9]+)+)\b", local_sentence))
            if declared_studies:
                study_ids = {row.evidence_id for row in records
                             if row.study_id in declared_studies or row.dataset in declared_studies}
                allowed = tuple(item for item in allowed if item.evidence_id in study_ids)
            source_only = re.search(r"\b(?:loaded|uploaded|source|current)\b.{0,90}?\b(?:GWAS|dataset)\b", local_sentence, re.I) and not re.search(
                r"\b(?:external|published|reported|credible[- ]set|[a-z-]*QTL|colocali[sz]\w*|regulatory)\b", local_sentence, re.I,
            )
            explicit_source = _explicit_parenthetical_source_p_owner(local_sentence, local_match, query)
            # Selection identity does not turn a measured eQTL P into the
            # uploaded GWAS P. Preserve the cited, same-owner molecular binding
            # when the surrounding clause explicitly describes that assay.
            # Explicit source-GWAS or source-phenotype statistics retain their
            # separate uploaded-observation namespace.
            molecular_ids = {row.evidence_id for row in records
                             if row.kind in {EvidenceKind.QTL, EvidenceKind.COLOCALIZATION}}
            molecular_context = re.search(r"\b(?:cis[- ]?)?(?:e|s|p|tu)?QTL\b|\bcolocali[sz]ation\b",
                                          sentence[:match.start()], re.I)
            # Sentence segmentation at abbreviations such as 'e.g.' can leave
            # a selected-SNP (Beta, P) tuple without its earlier assay noun.
            # A companion effect binds that tuple only when the same cited
            # molecular record supplies the matching gene/variant measurement.
            molecular_companion = any(
                _metric_name(companion['metric']) in {'beta', 'or'}
                and any(binding.evidence_id in molecular_ids
                        and _measurement_matches(companion, value, binding.value)
                        for binding in _measurement_candidates(local_sentence, companion,
                            bindings=local_bindings, aliases=aliases, traits=traits, owner=measurement_owner))
                for companion in _COMPANION_EFFECT.finditer(local_sentence)
                if (value := _number(companion['number'])) is not None)
            explicit_statistic = re.search(r"\b(?:source|uploaded|loaded|current)[- ]+(?:GWAS|dataset|data)\b",
                                           local_sentence, re.I) or any(
                re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", _normal(parenthetical))
                for alias in _trait_label_aliases(query.source_trait or query.trait))
            if ((molecular_context or molecular_companion) and not explicit_statistic
                    and any(item.evidence_id in molecular_ids and item.metric == 'p' for item in allowed)):
                explicit_source = None
                source_only = False
            # A named variant's source-P parenthesis has its own phenotype and
            # source/proxy binding. Failure to establish it must not fall back
            # to an earlier generic 'uploaded GWAS' label in the sentence.
            if parenthetical and explicit_source is None and _VARIANT.search(local_prefix):
                bounds = _parenthetical_bounds(local_sentence, local_match.start())
                local_label = local_sentence[bounds[0] + 1:local_match.start()] if bounds else ''
                local_label = re.split(r'[;,]', local_label)[-1].strip()
                # An explicit uploaded-GWAS label can own a (Beta, P) tuple;
                # the preceding Beta is not a new phenotype. A nonempty
                # unmatched trait label cannot borrow any molecular statistic.
                if not (source_only and explicit_statistic and not local_label):
                    source_only = False
                    if explicit_statistic and local_label and _trait_key(local_label) not in {
                            _trait_key(value) for value in _trait_label_aliases(query.source_trait or query.trait)}:
                        allowed = ()
            # An explicit molecular citation still owns its own P statistic,
            # even when the surrounding prose also calls its SNP a source SNP.
            if cited_ids and any(item.metric == "p" for item in local_bindings):
                explicit_source = None
                source_only = False
            if (source_only or explicit_source is not None) and _metric_name(match.group("metric")) == "p":
                source_labels = list(re.finditer(
                    r"\b(?:loaded|uploaded|source|current)[- ]+(?P<trait>[A-Za-z][A-Za-z0-9 ()-]{0,90}?)\s+GWAS\b",
                    local_prefix, re.I))
                source_trait_matches = not source_labels or _trait_key(source_labels[-1]["trait"]) in {
                    _trait_key(value) for value in _trait_label_aliases(query.source_trait or query.trait)}
                source = tuple(fact for fact in source_p_facts(query) if source_trait_matches
                               and (explicit_source is None or _variant(fact.variant_id) == explicit_source))
                allowed = _measurement_candidates(local_sentence, local_match, bindings=source, aliases={}, traits=traits, owner=None)
            if reported is not None and any(_measurement_matches(match, reported, item.value) for item in allowed) and (not repair or correction_authorized):
                if numeric_trace is not None:
                    numeric_trace.append({"field": label, "metric": metric, "value": match["number"],
                        "span": [start + match.start(), start + match.end()], "assertion_type": "measurement",
                        "status": "verified", "numeric_fact_ids": [item.fact_id for item in allowed
                            if _measurement_matches(match, reported, item.value)]})
                continue
            errors.append(f"{label} has an unsupported gene/variant/trait/{_metric_name(match.group('metric'))} measurement")
            # Equal numbers in different records do not establish a unique
            # source. Keep ambiguous/unknown assertions unchanged and unverified.
            unique = set(allowed)
            can_correct = (repair and correction_authorized and not entity_issue and len(unique) == 1
                    and match.group("relation") not in {"<", "<=", "≤", ">", ">=", "≥"}
                    and (not percent or percent_allowed))
            if can_correct:
                binding = next(iter(unique))
                if binding.trait and not any(
                    re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", _normal(sentence))
                    for alias in _trait_label_aliases(binding.trait)
                ) and binding.evidence_id != "uploaded_source":
                    # An unrecognised or omitted phenotype must not be silently
                    # replaced with the sole record's different phenotype.
                    can_correct = False
            if can_correct:
                value = binding.value * (100 if percent else 1)
                replacement = str(value)
                edits.append((start + match.start("number"), start + match.end("number"), replacement))
                if corrections is not None:
                    record = next((row for row in records if row.evidence_id == binding.evidence_id), None)
                    origin = ({"kind": "uploaded_source", "variant_id": binding.variant_id,
                               "trait": binding.trait, "query_sha256": sha256_json(asdict(query))}
                              if record is None else
                              {"kind": record.kind.value, "source": record.source, "dataset": record.dataset,
                               "data_release": record.data_release, "study_id": record.study_id,
                               "evidence_sha256": sha256_json(record.to_dict())})
                    corrections.append({"field": label, "original_value": match.group("number"),
                        "numeric_fact_ids": [binding.fact_id],
                        "corrected_value": replacement, "metric": binding.metric,
                        "relation": match.group("relation"), "percent": percent,
                        "evidence_ids": [binding.evidence_id], "gene_id": binding.gene_id,
                        "variant_id": binding.variant_id, "trait": binding.trait, "origin": origin,
                        "original_field_sha256": sha256_json(text),
                        "span": [start + match.start("number"), start + match.end("number")],
                        "rule": "unique_raw_measurement_numeric_correction_v1"})
            elif repair:
                metric_name = _metric_name(match.group("metric"))
                # A missing or ambiguous binding is not proof of a false
                # number. Keep the whole proposition, including qualifiers.
                replacement = "Original statement retained for review"
                if unverified is not None:
                    domain_error = (reported is None or not reported.is_finite() or
                        metric_name in {'p', 'h4', 'h3', 'pip', 'clpp'} and not 0 <= reported <= 1)
                    reason = ('numeric_domain_error' if domain_error else
                        'measurement_relation_ambiguous' if len(unique) > 1 else
                        'numeric_value_mismatch' if len(unique) == 1 and not entity_issue and correction_authorized
                            and (not next(iter(unique)).trait or any(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', _normal(sentence))
                                for alias in _trait_label_aliases(next(iter(unique)).trait))) else
                        'measurement owner or source is not uniquely verified')
                    unverified.append({"field": label, "metric": metric_name,
                        "original_value": match.group("number") + ("%" if percent else ""), "display_replacement": replacement,
                        "span": [start + match.start(), start + match.end()],
                        "evidence_ids": [row.evidence_id for row in records],
                        "numeric_fact_ids": sorted(item.fact_id for item in unique),
                        "candidate_bindings": [{"numeric_fact_id": item.fact_id, "evidence_id": item.evidence_id, "gene_id": item.gene_id,
                            "variant_id": item.variant_id, "trait": item.trait, "value": str(item.value)}
                            for item in allowed],
                        "reason": reason,
                        "original_field_sha256": sha256_json(text),
                        "rule": "unverified_numeric_span_v1"})
    from .native_source_assertions import native_ld_numeric_bindings
    for finding in native_ld_numeric_bindings(text, query=query, records=tuple(records),
            aliases=context.citation_aliases, gene_aliases=context.aliases, owner=owner, all_records=all_records):
        status = finding["verification_status"]
        if numeric_trace is not None:
            numeric_trace.append({"field": label, "metric": "r2", "value": finding['original_value'],
                "span": finding['span'], "assertion_type": finding['binding_mode'],
                "status": finding['finding_category'], "numeric_fact_ids": finding['numeric_fact_ids']})
        if status == "verified" and (not repair or correction_authorized):
            continue
        errors.append(f"{label} has an LD r² assertion requiring review ({status})")
        if not repair:
            continue
        if correction_authorized and finding["can_correct"]:
            replacement = str(finding["values"][0])
            edits.append((*finding["number_span"], replacement))
            if corrections is not None:
                corrections.append({"field": label, "metric": "r2", "original_value": finding["original_value"],
                    "numeric_fact_ids": finding['numeric_fact_ids'],
                    "corrected_value": replacement, "span": finding["number_span"],
                    "evidence_ids": finding["evidence_ids"], "origin": {"ld_edges": list(finding["bindings"])},
                    "original_field_sha256": sha256_json(text), "rule": "bound_ld_edge_numeric_correction_v1"})
        elif unverified is not None:
            # An unclear reference is not a false number. Keep the complete
            # expression visible, including both range endpoints, for review.
            unverified.append({"field": label, "metric": "r2", "original_value": finding["original_value"],
                "numeric_fact_ids": finding['numeric_fact_ids'],
                "display_replacement": "Original statement retained for review", "span": finding["span"],
                "original_span": finding["original_span"],
                "evidence_ids": finding["evidence_ids"], "candidate_bindings": list(finding["bindings"]),
                "reason": finding["reason"], "verification_status": status,
                "numeric_consistent": finding["numeric_consistent"], "binding_mode": finding["binding_mode"],
                "original_field_sha256": sha256_json(text), "rule": "typed_ld_assertion_v1"})
    from .gate2_hallucination import invalid_probability_sentence_edits
    for edit in invalid_probability_sentence_edits(text, [item for item in (unverified or ())
                                                          if item.get('field') == label]):
        edits = [item for item in edits if not (edit['start'] <= item[0] and item[1] <= edit['end'])]
        edits.append((edit['start'], edit['end'], ''))
    for begin, end, replacement in sorted(edits, reverse=True):
        text = text[:begin] + replacement + text[end:]
    from .gate2_hallucination import tidy_removed_values
    return tidy_removed_values(text), tuple(dict.fromkeys(errors))


def ranking_assertion_issues(query: LocusQuery, evidence: tuple[EvidenceRecord, ...], output: SynthesisOutput) -> tuple[str, ...]:
    context = _assertion_context(query, evidence, output)
    by_id = {record.evidence_id: record for record in evidence}
    visible = set(by_id) if output.model_visible_evidence_ids is None else set(output.model_visible_evidence_ids)
    referenced = set()
    errors = []
    for index, row in enumerate((*output.ranked_genes, *output.claims)):
        ids = set(row.evidence_ids) & visible
        referenced.update(ids)
        records = tuple(by_id[value] for value in ids if value in by_id)
        if hasattr(row, "gene_id"):
            pieces = [(f"rationale for {row.gene_id}", row.rationale), *((f"limitation for {row.gene_id}", text) for text in row.limitations)]
            owner = row.gene_id
        else:
            pieces, owner = [(f"claim {index + 1 - len(output.ranked_genes)}", row.text)], None
        for label, text in pieces:
            errors.extend(_prose_check(text, label=label, records=records, all_records=evidence, query=query, owner=owner, _context=context)[1])
    for row in output.candidate_decisions:
        ids = set(row.evidence_ids) & visible
        referenced.update(ids)
        records = tuple(by_id[value] for value in ids if value in by_id)
        errors.extend(_prose_check(row.reason, label=f"candidate decision for {row.gene_id}", records=records,
            all_records=evidence, query=query, owner=row.gene_id, _context=context)[1])
    errors.extend(_prose_check(output.summary, label="summary", records=tuple(by_id[value] for value in referenced if value in by_id), all_records=evidence, query=query, _context=context)[1])
    return tuple(dict.fromkeys(errors))


def assert_ranking_unchanged(original: SynthesisOutput, reviewed: SynthesisOutput) -> None:
    """A fact review has no authority to add, drop, reorder or renumber genes."""
    before = tuple((row.gene_id, row.rank, row.tied_with_previous) for row in original.ranked_genes)
    after = tuple((row.gene_id, row.rank, row.tied_with_previous) for row in reviewed.ranked_genes)
    if before != after:
        raise ValueError("Gate2 fact review must preserve the exact original model ranking")
    from .shortlisting import assert_selection_unchanged
    assert_selection_unchanged(original, reviewed)


def repair_ranking_assertions(query: LocusQuery, evidence: tuple[EvidenceRecord, ...], output: SynthesisOutput, *,
                              include_numeric_trace: bool = False) -> SynthesisOutput:
    """Correct a uniquely bound raw number; preserve interpretation, IDs and raw hash."""
    from .sufficiency import decision_evidence_for_query

    evidence = decision_evidence_for_query(query, evidence)
    by_id = {record.evidence_id: record for record in evidence}
    if len(by_id) != len(evidence):
        return output
    visible = set(by_id) if output.model_visible_evidence_ids is None else set(output.model_visible_evidence_ids)
    all_cited = {value for row in (*output.ranked_genes, *output.claims, *output.candidate_decisions)
                 for value in row.evidence_ids}
    context = _assertion_context(query, evidence, output)
    ranked, claims, decisions, corrections, unverified, numeric_trace = [], [], [], [], [], []
    def correct(text, *, field, records, owner=None, cited_ids=()):
        before = len(corrections)
        unverified_before = len(unverified)
        fixed = _prose_check(text, label=field, records=records, all_records=evidence,
            query=query, owner=owner, repair=True, _context=context, corrections=corrections, unverified=unverified, numeric_trace=numeric_trace,
            correction_authorized=set(cited_ids) <= visible & set(by_id)
                and not any(row.gene_id and owner and row.gene_id != owner for row in records))[0]
        for correction in corrections[before:]:
            correction["corrected_field_sha256"] = sha256_json(fixed)
        for finding in unverified[unverified_before:]:
            finding["corrected_field_sha256"] = sha256_json(fixed)
        return fixed

    for index, row in enumerate(output.ranked_genes):
        records = tuple(by_id[value] for value in row.evidence_ids if value in by_id and value in visible)
        rationale = correct(row.rationale, field=f"ranked_genes[{index}].rationale", records=records, owner=row.gene_id, cited_ids=row.evidence_ids)
        limitations = tuple(correct(text, field=f"ranked_genes[{index}].limitations[{position}]",
            records=records, owner=row.gene_id, cited_ids=row.evidence_ids) for position, text in enumerate(row.limitations))
        ranked.append(replace(row, rationale=rationale, limitations=limitations))
    for index, row in enumerate(output.claims):
        text = correct(row.text, field=f"claims[{index}].text", records=tuple(by_id[value] for value in row.evidence_ids if value in by_id and value in visible), cited_ids=row.evidence_ids)
        claims.append(replace(row, text=text))
    for index, row in enumerate(output.candidate_decisions):
        reason = correct(row.reason, field=f"candidate_decisions[{index}].reason", owner=row.gene_id,
            records=tuple(by_id[value] for value in row.evidence_ids if value in by_id and value in visible),
            cited_ids=row.evidence_ids)
        decisions.append(replace(row, reason=reason))
    summary = correct(output.summary, field="summary", records=tuple(by_id[value] for value in sorted(all_cited) if value in by_id and value in visible), cited_ids=all_cited)
    from .selection_evidence import selection_absence_findings
    unverified.extend(selection_absence_findings(query, evidence, output))
    changed = (tuple(ranked) != output.ranked_genes or tuple(claims) != output.claims
               or tuple(decisions) != output.candidate_decisions or summary != output.summary)
    return replace(output, ranked_genes=tuple(ranked), claims=tuple(claims), summary=summary,
                   candidate_decisions=tuple(decisions),
                   usage={**output.usage, "local_assertion_repairs": 1,
                          "numeric_fact_ledger_sha256": context.numeric_ledger["ledger_sha256"],
                          "numeric_assertion_trace": numeric_trace,
                          "hard_fact_corrections": [*output.usage.get("hard_fact_corrections", []), *corrections],
                          "unverified_fields": [*output.usage.get("unverified_fields", []), *unverified]}) if changed or corrections or unverified or (include_numeric_trace and numeric_trace) else output
