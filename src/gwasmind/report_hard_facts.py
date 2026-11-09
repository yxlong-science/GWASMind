"""Conservative numeric reconciliation against uniquely identified report sources.

No inference or model call is made here. The caller must validate the complete
corrected report; a correction receipt is not a scientific validation result.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from .gate2_hallucination import REMOVED_VALUE

if TYPE_CHECKING:
    from .staged_reasoning import HybridFinalContext

_JSON_STRING = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
_JSON_CHARACTER = re.compile(
    r'\\u[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}'
    r'|\\u[0-9a-fA-F]{4}|\\.|[^\\]', re.DOTALL,
)
_CLAUSE = re.compile(r'.+?(?:(?<=[.!?])(?=\s)|(?=\n)|$)', re.DOTALL)
_COORDINATE = re.compile(
    r'\b(?P<metric>chromosome|(?:base[- ]pair\s+)?position|coordinate)'
    r'\s*(?:=|:|is\b|was\b|at\b)?\s*(?P<number>\d+)(?![\w.]|[-–]\d)', re.IGNORECASE,
)
_EXTERNAL = re.compile(r'\b(?:external|published|retrieved|reported|eQTL|sQTL|QTL|coloc\w*)\b', re.IGNORECASE)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def _source_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False))


def _canonical(identifier: str) -> str:
    return identifier.casefold().removeprefix('chr').replace(':', '_')


def _strict_payload(content: str) -> dict | None:
    from .configuration import parameter
    from .staged_reasoning import StructuredOutputError, _strict_json_object

    if not isinstance(content, str) or not content.lstrip().startswith('{'):
        return None
    try:
        value = _strict_json_object(content, label='hard-fact reconciliation')
    except (StructuredOutputError, RecursionError):
        return None
    required = {'summary_markdown', 'evidence_sections', 'limitations_uncertainty'}
    if not required.issubset(value):
        return None
    limits = value['limitations_uncertainty']
    if not isinstance(limits, dict):
        return None
    for markdown in (value['summary_markdown'], limits.get('markdown')):
        if not isinstance(markdown, str) or len(markdown) > parameter('gates.report_markdown_max_chars'):
            return None
    def strings(items):
        return isinstance(items, list) and all(isinstance(item, str) for item in items)
    if 'reference_ids' in value and not strings(value['reference_ids']):
        return None
    if 'safety_check_ids' in limits and not strings(limits['safety_check_ids']):
        return None
    rows = value['evidence_sections']
    if not isinstance(rows, list) or any(not isinstance(row, dict)
        or not isinstance(row.get('section_id'), str)
        or not strings(row.get('fact_ids')) for row in rows):
        return None
    return value


def _field_tokens(content: str) -> dict[str, re.Match]:
    """Locate the two rendered fields by JSON path, ignoring extra metadata.

    A complete report can carry unrecognized bookkeeping fields. Those fields
    must neither disable fact review nor shadow a rendered Markdown field.
    """
    decoder = json.JSONDecoder()
    def positions(opening):
        result, index = {}, opening + 1
        while index < len(content):
            while index < len(content) and content[index].isspace():
                index += 1
            if index >= len(content) or content[index] == '}':
                return result
            key, index = decoder.raw_decode(content, index)
            while index < len(content) and content[index].isspace():
                index += 1
            if index >= len(content) or content[index] != ':':
                return {}
            index += 1
            while index < len(content) and content[index].isspace():
                index += 1
            result[key] = index
            _, index = decoder.raw_decode(content, index)
            while index < len(content) and content[index].isspace():
                index += 1
            if index < len(content) and content[index] == ',':
                index += 1
            else:
                return result
        return result
    try:
        top = positions(len(content) - len(content.lstrip()))
        limits = positions(top['limitations_uncertainty'])
        locations = {'summary_markdown': top['summary_markdown'], 'markdown': limits['markdown']}
        return {key: token for key, position in locations.items()
                if (token := _JSON_STRING.match(content, position)) is not None}
    except (KeyError, TypeError, ValueError, RecursionError):
        return {}


def _configured_gwas_threshold(clause: str, measurement) -> bool:
    """Recognize the configured decision threshold, not an observed SNP P."""
    from decimal import Decimal, InvalidOperation
    from .configuration import parameter
    if _metric_key(measurement['metric']) != 'p' or not re.search(r'\b(?:GWAS|genome[- ]wide)\b', clause, re.I):
        return False
    before, after = clause[:measurement.start()], clause[measurement.end():]
    named = (re.search(r'\b(?:threshold|cut[- ]?off)\s*(?::|of|at|is|was|\()?\s*$', before, re.I)
             or re.match(r'\s*\)?\s*(?:significance\s+)?(?:threshold|cut[- ]?off)\b', after, re.I))
    if not named:
        return False
    try:
        return _surface_value(measurement) == Decimal(str(parameter('analysis.gwas_significance_p')))
    except (ArithmeticError, InvalidOperation, ValueError):
        return False


def _tidy_removed_fields(content: str) -> str:
    """Finish value removals in the rendered fields once all spans are edited."""
    from .gate2_hallucination import tidy_removed_values

    if json.dumps(REMOVED_VALUE, ensure_ascii=True)[1:-1] not in content:
        return content
    tokens = sorted(_field_tokens(content).values(), key=lambda token: token.start(), reverse=True)
    for token in tokens:
        text = json.loads(token.group())
        if REMOVED_VALUE in text:
            content = (content[:token.start()] + json.dumps(tidy_removed_values(text), ensure_ascii=True)
                       + content[token.end():])
    return content


def _raw_boundaries(token: str) -> list[int]:
    boundaries = [1]
    for piece in _JSON_CHARACTER.finditer(token[1:-1]):
        boundaries.append(piece.end() + 1)
    return boundaries


def _source_binding(clause: str, context: HybridFinalContext):
    from .staged_reasoning import _REPORT_VARIANT, _measurement_trait_key

    mentions = {_canonical(match.group()) for match in _REPORT_VARIANT.finditer(clause)}
    if len(mentions) != 1 or _EXTERNAL.search(clause):
        return None
    # An explicit, complete source phenotype is mandatory. A generic "source
    # P" alone cannot prove that a different named phenotype was not intended.
    plain = re.sub(r'[*`]', '', clause)
    labels = re.findall(r'\b(?:loaded|uploaded|source|current)[-\s]+([^.;{}]{1,140}?)\s+(?:GWAS|dataset|data)\b', plain, re.IGNORECASE)
    if not labels or any(_measurement_trait_key(label) != _measurement_trait_key(context.source_trait) for label in labels):
        return None
    if re.search(r'\b(?:associated|association|for|with|and|but|while|compared|vs)\b', plain, re.IGNORECASE):
        return None
    rows = [row for row in context.source_observations if _canonical(str(row.get('variant_id', ''))) in mentions]
    if len(rows) != 1:
        return None
    row = rows[0]
    # Conflicting explicit row phenotypes must never be repaired using context.
    if any(row.get(key) and _measurement_trait_key(str(row[key])) != _measurement_trait_key(context.source_trait)
           for key in ('trait', 'phenotype', 'source_trait', 'source_phenotype')):
        return None
    return row


def _fact_binding(clause: str, fact, context: HybridFinalContext) -> bool:
    from .staged_reasoning import (
        _GENERIC_UPPERCASE_TOKENS,
        _REPORT_MEASUREMENT,
        _REPORT_VARIANT,
        _fact_measurement_trait,
        _measurement_trait_key,
        _reported_gene_aliases,
    )

    subject = str(fact.model_data.get('subject') or fact.model_data.get('gene') or '')
    owner_variants = {_canonical(match.group()) for match in _REPORT_VARIANT.finditer(subject)}
    mentioned = {_canonical(match.group()) for match in _REPORT_VARIANT.finditer(clause)}
    if mentioned and (not owner_variants or not mentioned.issubset(owner_variants)):
        return False
    gene_aliases = dict(_reported_gene_aliases(context))
    for candidate in context.facts:
        gene = str(candidate.model_data.get('gene') or candidate.model_data.get('subject') or '')
        if re.fullmatch(r'[A-Z][A-Z0-9-]{1,30}', gene) and not _REPORT_VARIANT.fullmatch(gene):
            gene_aliases.setdefault(gene, gene)
    metric = _REPORT_MEASUREMENT.search(clause)
    prefix = clause[:metric.start()] if metric else ''
    preceding_variants = {_canonical(match.group()) for match in _REPORT_VARIANT.finditer(prefix)}
    gene_mentions = {gene for alias, gene in gene_aliases.items()
                     if re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', prefix, re.IGNORECASE)}
    gene_owners = {gene for alias, gene in gene_aliases.items()
                   if re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', subject, re.IGNORECASE)}
    if gene_mentions and (not gene_owners or not gene_mentions.issubset(gene_owners)):
        return False
    if not preceding_variants and not gene_mentions:
        return False
    trait = _fact_measurement_trait(fact)
    if trait:
        normalized = _measurement_trait_key(re.sub(r'[*`]', '', clause))
        if not re.search(r'(?<!\w)' + re.escape(_measurement_trait_key(trait)) + r'(?!\w)', normalized):
            return False
    if gene_mentions:
        known = set(_GENERIC_UPPERCASE_TOKENS) | {'PIP', 'BETA', 'H4'} | set(gene_aliases)
        # A known gene mentioned earlier cannot authorize an unknown gene's
        # later statistic. Unknown uppercase owners remain for full validation.
        if any(token not in known for token in re.findall(r'\b[A-Z][A-Z0-9-]{1,30}\b', prefix)):
            return False
    return True


def reconcile_report_hard_facts(content: str, context: HybridFinalContext) -> tuple[str, dict]:
    """Replace only uniquely bound numbers; malformed/ambiguous input is unchanged.

    Audit hashes use UTF-8 JSON bytes; source hashes use canonical JSON of the
    original observation or complete immutable display-fact snapshot. All JSON
    formatting and all prose outside corrected numeric spans remain unchanged.
    """
    from .staged_reasoning import (
        _REPORT_MEASUREMENT,
        _SUMMARY_FACT_CITATION_TOKEN,
        StructuredOutputError,
        _measurement_name,
        _measurement_prose,
        _reported_statistic,
        _source_measurement_values,
        _validate_fact_bound_measurements,
    )

    audit = {'schema_version': '1', 'raw_json_sha256': _sha(content),
             'corrected_json_sha256': _sha(content), 'corrections': []}
    payload = _strict_payload(content)
    if payload is None:
        audit['unresolved'] = [{'reason': 'invalid_or_incomplete_json'}]
        return content, audit
    tokens = _field_tokens(content)
    edits = []
    facts = {fact.fact_id: fact for fact in context.facts}
    for field, key in (('summary_markdown', 'summary_markdown'), ('limitations_uncertainty.markdown', 'markdown')):
        token = tokens.get(key)
        if token is None:
            continue
        text = json.loads(token.group())
        boundaries = _raw_boundaries(token.group())
        if len(boundaries) != len(text) + 1:
            continue
        for sentence in _CLAUSE.finditer(text):
            clause = sentence.group()
            if len(clause) > 2000 or ';' in clause or re.search(r'\b(?:whereas|versus|instead|rather\s+than)\b', clause, re.IGNORECASE):
                continue
            measurements = [metric for metric in _REPORT_MEASUREMENT.finditer(clause)
                            if not _configured_gwas_threshold(clause, metric)]
            # A decimal followed by typographic scientific notation is not a
            # complete number span in the original text; never replace only
            # its mantissa or silently change a percentage's units.
            if any(re.match(r'\s*(?:[%×^]|[xX*·]\s*10|e\s*[+-]|(?:k|M|m)b\b)', clause[metric.end():]) for metric in measurements):
                continue
            anchors = list(_SUMMARY_FACT_CITATION_TOKEN.finditer(clause))
            source = _source_binding(clause, context) if not anchors else None
            candidates = []
            if source is not None:
                if len(measurements) == 1:
                    metric = measurements[0]
                    name = _measurement_name(metric['metric'])
                    # Effect signs require allele harmonization, so this pass
                    # only corrects the original source P value.
                    values = _source_measurement_values(source, name) if name == 'p' else ()
                    if len(values) == 1 and 0 < values[0] <= 1:
                        candidates.append((metric, str(values[0]), 'source_observation', str(source['variant_id']), dict(source)))
                identifier = _canonical(str(source['variant_id']))
                parts = identifier.split('_')
                chromosome, position = source.get('chromosome'), source.get('position')
                coherent = len(parts) == 4 and str(chromosome).casefold().removeprefix('chr') == parts[0] and str(position) == parts[1]
                if coherent:
                    for coordinate in _COORDINATE.finditer(clause):
                        if re.match(r'\s*(?:[kKmM]b|cM)\b', clause[coordinate.end():]):
                            continue
                        expected = str(chromosome) if coordinate['metric'].casefold() == 'chromosome' else str(position)
                        if re.fullmatch(r'\d+', expected):
                            candidates.append((coordinate, expected, 'source_observation', str(source['variant_id']), dict(source)))
            elif len(anchors) == 1 and len(measurements) == 1 and not re.search(r'\b(?:loaded|uploaded|source|current)\b', clause, re.IGNORECASE):
                fact = facts.get(anchors[0].group(1))
                metric = measurements[0]
                if fact is not None and metric.end() < anchors[0].start() and _fact_binding(clause, fact, context):
                    name = _measurement_name(metric['metric'])
                    try:
                        values = {_reported_statistic(match['number'], label=field)
                                  for match in _REPORT_MEASUREMENT.finditer(_measurement_prose(fact.markdown))
                                  if _measurement_name(match['metric']) == name}
                    except StructuredOutputError:
                        continue
                    if len(values) == 1:
                        expected = str(next(iter(values)))
                        patched = clause[:metric.start('number')] + expected + clause[metric.end('number'):]
                        try:
                            _validate_fact_bound_measurements(patched, context=context, label=field)
                        except StructuredOutputError:
                            continue
                        snapshot = {key: getattr(fact, key) for key in ('fact_id', 'section_id', 'evidence_ids', 'markdown', 'reference_ids', 'required')}
                        snapshot['model_data'] = dict(fact.model_data)
                        candidates.append((metric, expected, 'display_fact', fact.fact_id, snapshot))
            for match, expected, source_type, source_id, snapshot in candidates:
                original = match['number']
                try:
                    if _reported_statistic(original, label=field) == _reported_statistic(expected, label=field):
                        continue
                except StructuredOutputError:
                    continue
                start, end = sentence.start() + match.start('number'), sentence.start() + match.end('number')
                edits.append((token.start() + boundaries[start], token.start() + boundaries[end], expected))
                audit['corrections'].append({'field': field, 'metric': _metric_key(match['metric']),
                    'start': start, 'end': end, 'original_value': original, 'corrected_value': expected,
                    'source_type': source_type, 'source_id': source_id, 'source_sha256': _source_sha(snapshot),
                    'reason': 'unique_source_variant_measurement' if source_type == 'source_observation' else 'unique_cited_fact_measurement'})
    corrected = content
    for start, end, replacement in sorted(edits, reverse=True):
        corrected = corrected[:start] + replacement + corrected[end:]
    audit['corrected_json_sha256'] = _sha(corrected)
    return corrected, audit


# This finite metric vocabulary describes data fields, not scientific tone.
_SURFACE_MEASUREMENT = re.compile(
    r"(?<!\w)(?P<metric>P(?:[- ]?value)?|Beta|β|PIP|CLPP|H[34]|OR|odds[- ]ratio|log[- ]odds|r(?:²|\^?2)|SE|DNase|HiC|TSS)\s*"
    r"(?:=|:|of\b|is\b|was\b|at\b|up\s+to\b|~|≈)\s*"
    r"(?P<number>[+−-]?(?:(?:\d+(?:\.\d+)?|\.\d+)(?:e[+−-]?\d+|"
    r"\s*(?:×|x|\*|·)\s*10\s*(?:\^\s*(?:\{\s*[+−-]?\d+\s*\}|[+−-]?\d+)|[+−-]\s*\d+|[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?|NaN|Infinity|inf|∞))"
    r"(?P<percent>\s*%)?(?![\w/:}]|\.\d)", re.IGNORECASE,
)
_PROBABILITY_METRICS = frozenset({"p", "pip", "clpp", "h3", "h4", "r2"})


def _unfinished_scientific_end(clause, measurement):
    """Do not reconcile a mantissa when an unparsed scientific suffix follows."""
    suffix = re.match(r'\s*(?:[×xX*·]\s*10|[eE](?=\s*[+−-]))'
                      r'(?:[^\s,;)]|\s(?=[\d.+−\-^{}⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]))*', clause[measurement.end():])
    if suffix is None:
        return None
    return measurement.end() + len(suffix.group().rstrip(' .\t\r\n'))


def _metric_key(value):
    from .staged_reasoning import _measurement_name
    return _measurement_name(value).replace("²", "2").replace("^", "")


def _surface_value(match):
    from decimal import Decimal

    from .staged_reasoning import _measurement_prose
    number = Decimal(_measurement_prose(match["number"]).replace("∞", "Infinity"))
    if match.groupdict().get("percent") and _metric_key(match["metric"]) in _PROBABILITY_METRICS:
        number /= 100
    return number


def _rounds_to_authority(value, authoritative):
    from decimal import ROUND_HALF_EVEN, localcontext
    if value is None or not value.is_finite() or not authoritative.is_finite():
        return False
    if value == authoritative:
        return True
    # Only discard unsupported extra precision; never relax the declared
    # source precision to fit a less precise or unrelated number.
    if value.as_tuple().exponent >= authoritative.as_tuple().exponent:
        return False
    try:
        with localcontext() as decimal_context:
            decimal_context.prec = 80
            return value.quantize(authoritative, rounding=ROUND_HALF_EVEN) == authoritative
    except ArithmeticError:
        return False


def _valid_metric_authority(name, value):
    """A cited value must itself be a finite, dimensionally admissible number."""
    if not value.is_finite():
        return False
    if name == "p":
        return 0 < value <= 1
    if name in {"pip", "clpp", "h3", "h4", "r2"}:
        return 0 <= value <= 1
    if name == "or":
        return value > 0
    if name == "se":
        return value >= 0
    return True


def _source_identity_resolver(context, text=""):
    from .report_facts import public_source_normalizations
    from .source_normalization import validate_source_normalization
    aliases = {}
    rs_owners = {}
    for row in public_source_normalizations(context.result):
        aliases[_canonical(row["original_id"])] = _canonical(row["canonical_id"])
    resolution = context.result.get("variant_resolution", {})
    proofs = list(resolution.get("normalizations", ())) if isinstance(resolution, dict) else []
    ld = context.result.get("ld_expansion", {})
    if isinstance(ld, dict):
        proofs.extend(ld.get("source_identity_normalizations", ()))
        from .variant_identity_registry import validate_variant_identity_registry
        registry = ld.get("variant_identity_registry")
        if validate_variant_identity_registry(registry):
            aliases.update({_canonical(key): _canonical(value)
                            for key, value in registry["source_aliases"].items()})
            for identifier, identity in registry["variants"].items():
                for rsid in identity["rsids"]:
                    rs_owners.setdefault(rsid.casefold(), set()).add(_canonical(identifier))
    for row in proofs:
        if isinstance(row, dict) and validate_source_normalization(row):
            canonical = _canonical(row["canonical_id"])
            aliases[_canonical(row["original_id"])] = canonical
            for rsid in row["identity"].get("rsids", ()):
                if isinstance(rsid, str) and re.fullmatch(r"rs\d+", rsid, re.I):
                    rs_owners.setdefault(rsid.casefold(), set()).add(canonical)
    requested = set(re.findall(r"\brs\d+\b", text.casefold()))
    if requested:
        # A display label is not an identity proof. Reuse the identity gate's
        # coordinate/allele/provenance checks on authoritative variant records.
        from .web.stage_bridge import validated_variant_aliases
        observed_names = tuple({"variant_id": rsid} for rsid in sorted(requested))
        for record in context.result.get("evidence", ()):
            # Preserve all individually validated owners. An aggregate resolver
            # omits ambiguous rsIDs; that omission must not erase a conflict
            # with a separately saved normalization proof.
            if (not isinstance(record, dict) or not isinstance(record.get("payload"), dict)
                    or record["payload"].get("module") != "locus.variant"):
                continue
            resolved = validated_variant_aliases(
                observations=observed_names, evidence=(record,), genome_build="GRCh38")
            for rsid, identifier in resolved.items():
                rs_owners.setdefault(rsid.casefold(), set()).add(_canonical(identifier))
    aliases.update({rsid: next(iter(owners)) for rsid, owners in rs_owners.items() if len(owners) == 1})
    return lambda value: aliases.get(_canonical(value), _canonical(value))


def _source_owner_rows(clause, context):
    from .staged_reasoning import _REPORT_VARIANT, _measurement_trait_key
    if not _source_phenotype_labels_match(clause, context):
        return []
    resolve = _source_identity_resolver(context, clause)
    owners = {resolve(match.group()) for match in _REPORT_VARIANT.finditer(clause)}
    implicit = _unique_implicit_source_p(clause, context, resolve=resolve)
    if implicit is not None:
        return [implicit]
    if _EXTERNAL.search(clause):
        return []
    # Reports sometimes refer to the lead as “the selected variant” and omit
    # its normalized identifier.  A source P can still be authoritative when
    # the frozen source has exactly one observation; with multiple selected
    # variants there is no safe owner and the value remains unverified.
    if not owners and len(context.source_observations) == 1 and re.search(
            r"\b(?:selected|lead|index|uploaded|source)\s+(?:source\s+)?(?:variant|SNP)\b",
            clause, re.IGNORECASE):
        owners = {resolve(str(context.source_observations[0].get("variant_id", "")))}
    if len(owners) != 1:
        return []
    normalized = _measurement_trait_key(re.sub(r"[*\x60]", "", clause))
    trait = _measurement_trait_key(context.source_trait)
    if not trait or not re.search(r"(?<!\w)" + re.escape(trait) + r"(?!\w)", normalized):
        # A named selected SNP followed immediately by its source-data P
        # parenthesis also has an explicit owner, even among several sources.
        # The provenance suffix must end locally; another trait or a proxy
        # cannot be introduced between the statistic and its source label.
        plain = re.sub(r'[*`]', '', clause)
        metrics = list(_SURFACE_MEASUREMENT.finditer(plain))
        explicit_source = (bool(trait) and len(metrics) == 1 and _metric_key(metrics[0]['metric']) == 'p'
            and not re.search(r'\b(?:proxy|proxies)\b', plain, re.I)
            and re.search(_REPORT_VARIANT.pattern + r'\s*\(\s*$', plain[:metrics[0].start()], re.I)
            and re.match(r'\s+in\s+(?:the\s+)?(?:source|uploaded|loaded|current)\s+(?:GWAS|data|dataset)\s*(?:[),.;]|$)',
                         plain[metrics[0].end():], re.I))
        if not explicit_source:
            return []
    if not re.search(r"\b(?:source|uploaded|loaded|current)\b", clause, re.IGNORECASE):
        return []
    return [row for row in context.source_observations if resolve(str(row.get("variant_id", ""))) in owners]


def _source_phenotype_aliases(context):
    """Exact source labels and aliases from its verified target identity only."""
    from .phenotype_scope import validate_phenotype_scope, PhenotypeScopeError

    def key(label):
        return " ".join(str(label or "").casefold().split())

    source = key(context.source_trait)
    aliases = {source} if source else set()
    biological = context.biological_context
    scope = biological.get("phenotype_scope") if isinstance(biological, dict) else None
    if not isinstance(scope, dict):
        return aliases
    try:
        verified = validate_phenotype_scope(scope, phenotype=scope.get("phenotype", ""))
    except (PhenotypeScopeError, TypeError, ValueError):
        return aliases
    target = verified.get("target")
    if not target:
        return aliases
    row = next((row for row in verified["candidate_manifest"] if row["id"] == target["id"]), None)
    if row is None:
        return aliases
    supplied = {key(row["name"]), key(row["id"]), *(key(item) for item in row["exact_synonyms"])}
    # The frozen target must belong to this uploaded phenotype, not merely to
    # another phenotype mentioned somewhere in the biological context.
    if source == key(verified["phenotype"]) or source in supplied:
        aliases.update(supplied)
    return aliases


def _source_phenotype_labels_match(clause, context):
    """Source qualifiers may surround a phenotype without becoming its name."""
    from .staged_reasoning import _REPORT_VARIANT
    aliases = _source_phenotype_aliases(context)
    if not aliases:
        return False
    labels = re.findall(r'\b(?:loaded|uploaded|source|current)\s+([^,.;{}]{1,100}?)\s+(?:GWAS|dataset)\b',
                        clause, re.I)
    labels += re.findall(r'\b(?:concerns|in|from|for)\s+(?:(?:a|an|the|that)\s+)?'
                         r'([^,.;{}]{1,100}?)\s+(?:source|uploaded|loaded|current)\s+GWAS\b', clause, re.I)
    provenance = r'(?:source|uploaded|loaded|current|the)'
    for label in labels:
        measured = _SURFACE_MEASUREMENT.match(label)
        if measured is not None:
            # "source P=... in the uploaded dataset" describes a statistic,
            # not an intervening phenotype. Its source grammar is checked by
            # the measurement owner resolver, including explicit trait tails.
            continue
        if (_REPORT_VARIANT.search(label) and re.match(
                r'(?:lead\s+variant\b|source[- ]observation\s+ledger\b)', label, re.I)):
            # A bounded variant/normalization description is not a phenotype
            # label. Its identity is checked separately against the source.
            continue
        label = re.sub(r'^(?:' + provenance + r'\s+)+|(?:\s+' + provenance + r')+$', '', label, flags=re.I)
        key = " ".join(label.casefold().split())
        if key in {'source', 'uploaded', 'loaded', 'current', 'the'} | aliases:
            continue
        # "for <trait> at the chromosome 1 locus selected from the uploaded
        # GWAS" names the source trait and then the locus, not another phenotype.
        location = re.fullmatch(
            r'(?P<label>.+?)\s+(?:at|near|around|in)\s+(?:the\s+)?'
            r'(?:chromosome\s+(?:[1-9]|1[0-9]|2[0-2]|x|y|mt)\s+(?:locus|region|interval)'
            r'|(?:genomic\s+)?(?:locus|region|interval)\s+on\s+chromosome\s+(?:[1-9]|1[0-9]|2[0-2]|x|y|mt))'
            r'(?:\s+selected\s+from(?:\s+the)?)?', key)
        if location and location['label'] in aliases:
            continue
        return False
    return True


def _explicit_selected_source_p_rows(clause, measurement, context):
    """Bind an explicitly named uploaded SNP's own P despite a nearby citation.

    A report citation may support another fact in the same sentence. It cannot
    suppress a separately declared source observation, or supply a missing one.
    None means this narrow syntax is absent; an empty list means it was stated
    but its source identity, phenotype or ownership could not be verified.
    """
    from .staged_reasoning import _REPORT_VARIANT, _measurement_trait_key
    if _metric_key(measurement['metric']) != 'p':
        return None
    prefix = re.sub(r'[*`]', '', clause[:measurement.start()])
    declared = re.search(
        r'\b(?:the\s+)?(?:uploaded|loaded|source(?:[- ]selected)?|user[- ]selected|selected|input|current)\s+'
        r'(?:(?:(?:only|unique)\s+)?(?:lead|index)\s+)?(?:variant|SNP)\s+(?:is\s+)?(?P<variant>' + _REPORT_VARIANT.pattern + r')'
        r'\s+(?:with|has|had|carries)\s+(?:(?:a|the|its)\s+)?(?:source|uploaded|loaded|current)\s*$',
        prefix, re.I)
    if declared is None:
        # The source label belongs to the SNP, so "uploaded variant A at
        # P=..." need not repeat "source" immediately before its statistic.
        source_at = re.search(
            r'\b(?:the\s+)?(?P<strongest>strongest\s+)?(?:uploaded|loaded|current|source)\s+'
            r'(?:source\s+)?(?:variant|SNP)\s+(?:is\s+)?'
            r'(?P<variant>' + _REPORT_VARIANT.pattern + r')'
            r'(?:\s*\(\s*(?:normalized\s+(?:as|to)\s+)?(?P<alias>' + _REPORT_VARIANT.pattern + r')\s*\))?'
            r'\s+at\s+(?:(?:source|uploaded|loaded|current)\s+)?$', prefix, re.I)
        if source_at is not None:
            resolve = _source_identity_resolver(context, clause)
            if source_at['alias'] and resolve(source_at['variant']) != resolve(source_at['alias']):
                return []
            if source_at['strongest']:
                from .report_facts import computed_report_facts
                facts = computed_report_facts(context.source_observations, context.result)
                if resolve(source_at['variant']) not in {resolve(item) for item in facts['source_lead_variant_ids']}:
                    return []
            declared = source_at
    if declared is None:
        # A source statistic can be owned by a named SNP followed by its
        # verified alternative notation and a relative/possessive clause.
        # Both names must resolve to one source row; a correct number alone
        # never establishes that identity.
        named_source = re.search(
            r'\b(?:(?:the|a)\s+)?(?:(?:selected|source|uploaded|loaded|input|current|user[- ]selected)\s+)+'
            r'(?:(?:lead|index)\s+)?(?:variant|SNP)\s+(?:is\s+)?'
            r'(?P<variant>' + _REPORT_VARIANT.pattern + r')'
            r'(?:\s*\(\s*(?:normalized\s+(?:as|to)\s+)?(?P<alias>' + _REPORT_VARIANT.pattern + r')\s*\))?'
            r"(?:\s*,?\s+whose|\s*(?:'s|’s)|\s+(?:has|had|carries|with|showing))\s+"
            r'(?:(?:a|an|the|its)\s+)?(?:(?:uploaded|loaded|current)\s+)?'
            r'(?:source(?:[- ]GWAS)?|uploaded(?:[- ]GWAS)?|loaded(?:[- ]GWAS)?)\s*$', prefix, re.I)
        if named_source is not None:
            resolve = _source_identity_resolver(context, clause)
            if named_source['alias'] and resolve(named_source['variant']) != resolve(named_source['alias']):
                return []
            declared = named_source
    if declared is None:
        # A named lead can own an appositive source P, an "at source P"
        # statement or a parenthetical uploaded-data significance statement.
        # Keep the relation local: no intervening owner, external study,
        # proxy, phenotype label or second measurement can enter this form.
        lead_source = re.search(
            r'\b(?:(?:the|a)\s+)?(?:(?:selected|source|uploaded|loaded|input|current|user[- ]selected)\s+)*'
            r'(?:lead|index)(?:\s+(?:variant|SNP))?\s*,?\s+(?:(?:is|of)\s+)?'
            r'(?P<variant>' + _REPORT_VARIANT.pattern + r')'
            r'(?:\s*\(\s*(?:normalized\s+(?:as|to)\s+)?(?P<alias>' + _REPORT_VARIANT.pattern + r')\s*\))?'
            r'(?:\s*,\s*(?:its\s+)?(?:uploaded\s+|loaded\s+|current\s+)?source(?:[- ]GWAS)?'
            r'|\s+at\s+(?:uploaded\s+|loaded\s+|current\s+)?source(?:[- ]GWAS)?'
            r'|\s*,?\s+(?:has|had|carries|with)\s+(?:(?:a|the|its)\s+)?'
            r'(?:uploaded\s+|loaded\s+|current\s+)?source(?:[- ]GWAS)?'
            r'|\s*,?\s+is\s+(?:genome[- ]wide\s+)?significant\s+in\s+(?:the\s+)?'
            r'(?:uploaded|loaded|current)\s+(?:source\s+)?(?:data|dataset|GWAS)\s*\(\s*source)\s*$',
            prefix, re.I)
        if lead_source is not None:
            resolve = _source_identity_resolver(context, clause)
            if lead_source['alias'] and resolve(lead_source['variant']) != resolve(lead_source['alias']):
                return []
            declared = lead_source
    if declared is None:
        # A selected-source list can use one immediate parenthesis per SNP.
        # The shared uploaded phenotype header authorizes omitted repetitions
        # of "source", but each number still belongs to its own named SNP.
        parenthetical = re.search(
            r'(?P<variant>' + _REPORT_VARIANT.pattern + r')\s*\(\s*'
            r'(?:(?:lead|index)\s*;\s*)?(?P<origin>(?:source|uploaded|loaded|current)\s+)?$',
            prefix, re.I)
        if parenthetical is not None and re.match(r'\s*\)', clause[measurement.end():]):
            first_variant = _REPORT_VARIANT.search(prefix)
            header = prefix[:first_variant.start()] if first_variant else ''
            trait = _measurement_trait_key(context.source_trait)
            normalized_header = _measurement_trait_key(header)
            shared_source_list = (bool(trait)
                and re.search(r'\b(?:uploaded|loaded|source|current)\b', header, re.I)
                and re.search(r'\b(?:variants|SNPs)\b', header, re.I)
                and re.search(r'(?<!\w)' + re.escape(trait) + r'(?!\w)', normalized_header)
                and not _EXTERNAL.search(header)
                and not re.search(r'\b(?:proxy|proxies)\b', header, re.I))
            if parenthetical['origin'] or shared_source_list:
                declared = parenthetical
    minimum_claim = False
    if declared is None:
        # Source minimum followed by its owner is the same fact as an
        # owner-first source-P statement, not an external study statistic.
        minimum_claim = bool(re.search(r'\bminimum\s+(?:source|uploaded|loaded|current)\s*$', prefix, re.IGNORECASE))
        if minimum_claim:
            declared = re.match(r'\s+at\s+(?:the\s+)?lead\s+(?:variant|SNP)\s+[`*]*(?P<variant>'
                + _REPORT_VARIANT.pattern + r')[`*]*\s*'
                r'(?:\(\s*(?:canonical|normalized)\s+identifier\s+(?:for|of)\s+(?:the\s+)?'
                r'(?:uploaded|loaded|source|selected)\s+[`*]*(?P<alias>' + _REPORT_VARIANT.pattern
                + r')[`*]*\s*\)\s*)?(?:[).,;]|$)', clause[measurement.end():], re.IGNORECASE)
            if declared is None:
                declared = re.search(r'\blead\s+(?:canonical\s+)?(?:(?:variant|SNP)\s+)?'
                    r'(?P<variant>' + _REPORT_VARIANT.pattern + r')\s*;\s*minimum\s+'
                    r'(?:source|uploaded|loaded|current)\s*$', prefix, re.I)
            if declared is None or _EXTERNAL.search(clause) or re.search(r'\b(?:proxy|proxies)\b', clause, re.IGNORECASE):
                return []
            from .report_facts import computed_report_facts
            resolve = _source_identity_resolver(context, clause)
            alias = declared.groupdict().get('alias')
            if alias and resolve(declared['variant']) != resolve(alias):
                return []
            facts = computed_report_facts(context.source_observations, context.result)
            if resolve(declared['variant']) not in {resolve(item) for item in facts['source_lead_variant_ids']}:
                return []
    if declared is None:
        source_phrase = bool(re.search(r'source[- ]data\s*$', prefix.rstrip(), re.IGNORECASE))
        lead_matches = list(re.finditer(_REPORT_VARIANT.pattern, prefix))
        lead = lead_matches[-1] if lead_matches and re.search(r'lead\s+variant', prefix[:lead_matches[-1].start()], re.IGNORECASE) else None
        uploaded_tail = bool(re.search(r'uploaded\s+measurement\s+from\s+.*(?:GWAS|dataset)',
                                       clause[measurement.end():], re.IGNORECASE))
        if not source_phrase or not lead or not uploaded_tail:
            # No explicit selected-source syntax was present.  Let the
            # ordinary source-owner resolver handle an implicit lead/source
            # statement (including one that shares a citation).
            return None
        if _EXTERNAL.search(clause):
            return []
        resolve = _source_identity_resolver(context, clause)
        identifier = resolve(lead.group())
        rows = [row for row in context.source_observations
                if resolve(str(row.get('variant_id', ''))) == identifier]
        trait = _measurement_trait_key(context.source_trait)
        if len(rows) != 1 or any(row.get(key) and _measurement_trait_key(str(row[key])) != trait
                                  for row in rows for key in ('trait', 'phenotype', 'source_trait', 'source_phenotype')):
            return []
        return rows
    trait = _measurement_trait_key(context.source_trait)
    if not trait:
        return []
    # An explicitly named different phenotype cannot borrow the uploaded P.
    # Inspect both source-first and phenotype-first dataset descriptions.
    if not _source_phenotype_labels_match(clause, context):
        return []
    local_prefix = re.split(r'[;\n]|\}\}', prefix if minimum_claim else prefix[:declared.start()])[-1]
    if _EXTERNAL.search(local_prefix) or re.search(r'\b(?:proxy|proxies)\b', local_prefix, re.I):
        return []
    tail = clause[measurement.end():]
    if re.match(r'\s*(?:for|with|in)\b', tail, re.I):
        suffix = re.match(r'\s*(?:for|with|in)\s+(?:the\s+)?([^,.;{}]+)', tail, re.I)
        label = _measurement_trait_key(suffix[1]) if suffix else ''
        allowed = {trait}
        for origin in ('uploaded', 'loaded', 'source', 'current', 'uploaded source'):
            for noun in ('GWAS', 'dataset', 'data'):
                allowed.add(_measurement_trait_key(f'{origin} {noun}'))
                allowed.add(_measurement_trait_key(f'{origin} {context.source_trait} {noun}'))
                allowed.add(_measurement_trait_key(f'{context.source_trait} {origin} {noun}'))
        if label not in allowed:
            return []
    if re.match(r'\s*[,)]?\s*(?:(?:is|was|has\s+been)\s+)?'
                r'(?:(?:reported|published|retrieved|obtained|taken)\s+(?:by|in|from)|from)\s+'
                r'(?:(?:a|an|the)\s+)?(?:external|published|reported)\b', tail, re.I):
        return []
    resolve = _source_identity_resolver(context, clause)
    identifier = resolve(declared['variant'])
    rows = [row for row in context.source_observations if resolve(str(row.get('variant_id', ''))) == identifier]
    if len(rows) != 1 or any(row.get(key) and _measurement_trait_key(str(row[key])) != trait
        for row in rows for key in ('trait', 'phenotype', 'source_trait', 'source_phenotype')):
        return []
    return rows


def _unique_implicit_source_p(clause, context, *, resolve):
    """Bind a generic source-P sentence only to one explicitly supplied source.

    This narrow grammatical form permits an omitted SNP name, not an omitted
    provenance: external/proxy or phenotype-qualified statistics do not match.
    Equal P values across multiple selected rows never establish an owner.
    """
    from .staged_reasoning import _REPORT_VARIANT, _measurement_trait_key
    if len(context.source_observations) != 1 or not context.source_trait:
        return None
    row = context.source_observations[0]
    if not isinstance(row.get('variant_id'), str):
        return None
    if any(row.get(key) and _measurement_trait_key(str(row[key])) != _measurement_trait_key(context.source_trait)
           for key in ('trait', 'phenotype', 'source_trait', 'source_phenotype')):
        return None
    plain = re.sub(r'[*`]', '', clause).strip()
    metrics = list(_SURFACE_MEASUREMENT.finditer(plain))
    if len(metrics) != 1 or _metric_key(metrics[0]['metric']) != 'p':
        return None
    metric = metrics[0]
    # A negative external-attribution caveat does not turn the preceding
    # uploaded-source statistic into an external-study measurement.
    tail = re.sub(r'\s+and\s+(?:should|must)\s+not\s+be\s+(?:interpreted|treated|used)\s+as\s+'
                  r'evidence\s+from\s+(?:any|an?|the)\s+external\s+(?:publication|study|report)\s*[.]?\s*$',
                  '', plain[metric.end():], flags=re.I)
    prefix = plain[:metric.start()]
    if re.fullmatch(r'in\s+(?:the\s+)?(?:uploaded|loaded|current)\s+(?:source\s+)?'
                    r'(?:dataset|data|GWAS)\s*,\s*its\s*', prefix, re.I):
        # A source-labelled pronoun is unambiguous only for one selected row.
        # Do not borrow its statistic for another trait, proxy or study.
        mentions = {resolve(match.group()) for match in _REPORT_VARIANT.finditer(plain)}
        if (not _EXTERNAL.search(plain) and not re.search(r'\b(?:proxy|proxies)\b', plain, re.I)
                and _source_phenotype_labels_match(plain, context)
                and (not mentions or mentions == {resolve(row['variant_id'])})
                and re.match(r'\s*(?:[.]?$|,\s*meeting\s+(?:the\s+)?configured\s+genome[- ]wide\s+threshold\b)', tail, re.I)):
            return row
        return None
    if not re.fullmatch(r'(?:(?:for|at)\s+' + _REPORT_VARIANT.pattern + r'\s*,\s*)?'
                        r'(?:the\s+)?(?:source|uploaded|loaded|current)\s*', prefix, re.I):
        return None
    if not re.fullmatch(r'(?:\s+for\s+' + _REPORT_VARIANT.pattern + r')?'
                        r'(?:\s+(?:reflects\s+the\s+association\s+in|is\s+from|comes\s+from|was\s+measured\s+in)\s+'
                        r'(?:the\s+)?(?:uploaded|loaded|current)(?:\s+GWAS)?\s+dataset(?:\s+only)?)?\s*[.]?\s*', tail, re.I):
        return None
    mentions = {resolve(match.group()) for match in _REPORT_VARIANT.finditer(plain)}
    return row if not mentions or mentions == {resolve(row['variant_id'])} else None


def _colocalization_entry_binding(markdown, measurement):
    """Bind a displayed H4 to its own generated row, never a neighbouring pair."""
    from .staged_reasoning import _REPORT_VARIANT

    start = markdown.rfind('\n', 0, measurement.start()) + 1
    end = markdown.find('\n', measurement.end())
    row = re.sub(r'[*`]', '', markdown[start:end if end >= 0 else len(markdown)])
    pairs = list(re.finditer(r'\(H4 study pair:\s*(\S+)\s+vs\s+([^\s;()]+)\s*(?:;|\))', row))
    metrics = [item for item in _SURFACE_MEASUREMENT.finditer(row) if _metric_key(item['metric']) == 'h4']
    if len(pairs) != 1 or len(metrics) != 1:
        return None
    binding = {'study_pair': [pairs[0][1], pairs[0][2]], 'row_sha256': _sha(row)}
    # These dimensions come from this generated display row, never a model
    # explanation or a neighbouring gene's H4. They can identify an uncited
    # restatement only when the complete tuple selects one displayed pair.
    dimensions = re.fullmatch(r'\s*-\s*(?P<qtl>(?:e|s|p|tu|sce|scs|scp|sctu)QTL)\s+coloc\s+'
        r'\((?P<biosample>[^()]*)\)\s+with\s+(?P<phenotype>.+?)\s+\(\s*',
        row[:metrics[0].start()], re.IGNORECASE)
    if dimensions:
        binding.update({key: dimensions[key].strip() for key in ('qtl', 'biosample', 'phenotype')})
    lookups = list(re.finditer(r'representative lookup through (source-selected variant|LD proxy)\s+'
        r'(' + _REPORT_VARIANT.pattern + r')', row, re.I))
    if len(lookups) == 1:
        binding.update(lookup_variant=_canonical(lookups[0][2]), lookup_role=lookups[0][1].casefold())
    return binding


def _measurement_registry(context):
    """Read the immutable display facts into a bounded metric/owner registry."""
    from .staged_reasoning import _REPORT_VARIANT, _final_ld_report_scope, _reported_gene_aliases
    from .qtl_measurement_binding import credible_set_lead_binding
    ld_scope = _final_ld_report_scope(context)
    gene_aliases = dict(_reported_gene_aliases(context))
    for fact in context.facts:
        subject = str(fact.model_data.get("subject") or fact.model_data.get("gene") or "")
        first = subject.split(" (", 1)[0]
        if re.fullmatch(r"[A-Z][A-Z0-9-]{1,30}", first):
            gene_aliases.setdefault(first, first)
    registry = []
    for fact in context.facts:
        subject = str(fact.model_data.get("subject") or fact.model_data.get("gene") or "")
        genes = {gene for alias, gene in gene_aliases.items()
                 if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", subject, re.IGNORECASE)}
        variant_owners = {_canonical(match.group()) for match in _REPORT_VARIANT.finditer(subject)}
        representative = re.search(r"representative (?:LD proxy|source-selected variant|analy[sz]ed variant)\s*[\x60*]*"
            r"((?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+)", fact.markdown, re.IGNORECASE)
        if representative:
            variant_owners.add(_canonical(representative.group(1)))
        for match in _SURFACE_MEASUREMENT.finditer(fact.markdown):
            try:
                number = _surface_value(match)
            except (ArithmeticError, ValueError):
                continue
            if not _valid_metric_authority(_metric_key(match["metric"]), number):
                continue
            row_start = fact.markdown.rfind('\n', 0, match.start()) + 1
            row_end = fact.markdown.find('\n', match.end())
            row_text = fact.markdown[row_start:row_end if row_end >= 0 else len(fact.markdown)]
            lead_binding = credible_set_lead_binding(row_text)
            metric_owners = ({_canonical(lead_binding['variant_id'])} if lead_binding is not None
                and _metric_key(match['metric']) in {'p', 'beta', 'se'} else variant_owners)
            entry = {"fact_id": fact.fact_id, "metric": _metric_key(match["metric"]),
                "value": str(number), "genes": sorted(genes), "variants": sorted(metric_owners),
                "source_span": [match.start(), match.end()],
                "unit": "%" if match["percent"] and _metric_key(match["metric"]) not in _PROBABILITY_METRICS else "",
                "source_sha256": _source_sha({"fact_id": fact.fact_id, "subject": subject, "markdown": fact.markdown})}
            if lead_binding is not None:
                entry['qtl_measurement'] = lead_binding
            if entry['metric'] == 'h4':
                binding = _colocalization_entry_binding(fact.markdown, match)
                if binding is not None:
                    entry['colocalization'] = binding
            registry.append(entry)
        # The displayed representative is a foreign key into the validated LD
        # graph. Its own reference edges remain available even when the short
        # molecular fact does not repeat r² beside P and Beta.
        if ld_scope.provided and ld_scope.valid:
            for link in ld_scope.links:
                proxy = _canonical(link["proxy_variant_id"])
                if proxy not in variant_owners:
                    continue
                registry.append({"fact_id": fact.fact_id, "metric": "r2", "value": str(link["r2"]),
                    "genes": sorted(genes), "variants": [proxy],
                    "selected_variant_id": _canonical(link["selected_variant_id"]),
                    "proxy_variant_id": proxy, "population": link["population"],
                    "genome_build": link["genome_build"], "source_sha256": _source_sha(dict(link))})
    return registry, gene_aliases


def _bind_ld_registry_entries(candidates, clause, context):
    """Keep explicit pair/population bindings; a citation never grants all LD."""
    from .staged_reasoning import _REPORT_VARIANT, _final_ld_report_scope
    scope = _final_ld_report_scope(context)
    aliases = {_canonical(key): _canonical(value) for key, value in scope.source_aliases}
    def canonical(identifier):
        key = _canonical(identifier)
        return aliases.get(key, key)
    plain = re.sub(r"[*`]+", "", clause)
    # Only syntax actually joining the identifiers declares a pair. A source
    # mentioned elsewhere in a comparison is not an inferred LD partner.
    pair = re.search(r"(?P<a>" + _REPORT_VARIANT.pattern + r")\s*(?:and|with|vs\.?|versus|[-–])\s*"
        r"(?:(?:the|source|selected|uploaded|LD|proxy|SNP|variant)\s+)*(?P<b>" + _REPORT_VARIANT.pattern + r")", plain, re.IGNORECASE)
    if pair:
        declared = {canonical(pair["a"]), canonical(pair["b"])}
        candidates = [entry for entry in candidates if
            {canonical(entry["selected_variant_id"]), entry["proxy_variant_id"]} == declared]
    partner = re.search(r"\br(?:²|\^?2)\s*=\s*[\d.]+\s+with\s+"
        r"(?:(?:the|source|selected|uploaded|SNP|variant)\s+)*(?P<v>" + _REPORT_VARIANT.pattern + r")", plain, re.IGNORECASE)
    if partner:
        candidates = [entry for entry in candidates if canonical(entry["selected_variant_id"]) == canonical(partner["v"])]
    populations = set(re.findall(r"\b(?:AFR|AMR|EAS|EUR|SAS)\b", plain))
    builds = set(re.findall(r"\bGRCh(?:37|38)\b", plain, re.IGNORECASE))
    if populations:
        candidates = [entry for entry in candidates if populations == {entry["population"]}]
    if builds:
        candidates = [entry for entry in candidates if {item.casefold() for item in builds} == {entry["genome_build"].casefold()}]
    return candidates


def _measurement_citation_scope(clause, measurement):
    """A trailing citation group closes one measurement statement.

    The next statement can name another tissue, study or proxy of the same
    gene. Do not carry the previous statement's variant across its citation.
    Adjacent citations still jointly support a comparison or measurement.
    """
    from .staged_reasoning import _SUMMARY_FACT_CITATION_TOKEN

    anchors = list(_SUMMARY_FACT_CITATION_TOKEN.finditer(clause))
    following = [anchor for anchor in anchors if anchor.start() >= measurement.end()]
    if not following:
        # Separate uncited comparisons must not borrow each other's LD partner.
        start = clause.rfind(';', 0, measurement.start()) + 1
        end = clause.find(';', measurement.end())
        return clause[start:end if end >= 0 else len(clause)], {anchor.group(1) for anchor in anchors}
    first = following[0]
    end = first.end()
    selected = {first.group(1)}
    for anchor in following[1:]:
        if clause[end:anchor.start()].strip():
            break
        selected.add(anchor.group(1))
        end = anchor.end()
    previous = [anchor for anchor in anchors if anchor.end() <= measurement.start()]
    start = previous[-1].end() if previous else 0
    return clause[start:end], selected


def _respectively_measurement_owner(clause, measurement, aliases):
    """Resolve an explicit ordered gene list to its ordered metric list.

    Ownership comes from grammar and supplied gene identities, never from a
    matching numeric value or from whichever gene name happens to be closest.
    Ambiguous lists deliberately leave the ordinary binding unchanged.
    """
    marker = re.search(r'\brespectively\b', clause, re.I)
    if marker is None or marker.start() < measurement.end():
        return None
    metrics = [item for item in _SURFACE_MEASUREMENT.finditer(clause[:marker.start()])
               if _metric_key(item['metric']) == _metric_key(measurement['metric'])]
    if len(metrics) < 2 or not aliases:
        return None
    prefix = re.sub(r'[*`]', '', clause[:metrics[0].start()])
    names = '|'.join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
    name = r'(?<!\w)(?:' + names + r')(?!\w)'
    groups = list(re.finditer(name + r'(?:\s*(?:,\s*(?:and\s+)?|\band\s+)' + name + r')+', prefix, re.I))
    if len(groups) != 1:
        return None
    group = groups[0]
    if re.search(name, prefix[:group.start()] + prefix[group.end():], re.I):
        return None
    lookup = {alias.casefold(): gene for alias, gene in aliases.items()}
    owners = [lookup[item.group().casefold()] for item in re.finditer(name, group.group(), re.I)]
    if len(owners) != len(metrics) or len(set(owners)) != len(owners):
        return None
    index = next((index for index, item in enumerate(metrics) if item.span() == measurement.span()), None)
    return owners[index] if index is not None else None


def _respectively_colocalization_entries(entries, clause, context):
    """Keep a shared phenotype/QTL label bound to each cited signal pair."""
    from .staged_reasoning import _measurement_trait_key
    from .phenotype_scope import validate_phenotype_scope, PhenotypeScopeError

    label = re.search(r'\bcolocali[sz]\w*\s+with\s+(?P<trait>[^.;:]+?)\s+(?:at|with)\s+H4\b', clause, re.I)
    if label is None:
        return entries
    trait = _measurement_trait_key(re.sub(r'[*`]', '', label['trait']))
    qtls = {value.casefold() for value in re.findall(r'\b(?:cis-)?((?:e|s|p|tu)QTL)\b', clause, re.I)}
    target_id = None
    scope = context.biological_context.get('phenotype_scope')
    if isinstance(scope, dict) and label['trait'].casefold().strip() in _source_phenotype_aliases(context):
        try:
            target = validate_phenotype_scope(scope, phenotype=scope.get('phenotype', '')).get('target')
            target_id = target['id'] if target else None
        except (PhenotypeScopeError, TypeError, ValueError):
            pass

    def target_pair(entry):
        if target_id is None:
            return False
        pair = entry.get('colocalization', {}).get('study_pair')
        if not pair:
            return False
        for record in context.result.get('evidence', ()):
            if record.get('kind') != 'colocalization' or record.get('gene_id') not in entry['genes']:
                continue
            payload = record.get('payload') or {}
            raw = (payload.get('upstream') or {}).get('raw') or {}
            if (pair == [raw.get('leftStudyId'), raw.get('rightStudyId')]
                    and target_id in ((payload.get('context') or {}).get('phenotype_ids') or ())):
                return True
        return False

    return [entry for entry in entries if (
        not qtls or entry.get('colocalization', {}).get('qtl', '').casefold() in qtls) and (
        _measurement_trait_key(entry.get('colocalization', {}).get('phenotype', '')) == trait
        or target_pair(entry))]


def _uncited_ld_pair_entries(clause, context, *, sentence_prefix=''):
    """Bind explicit LD endpoints, including one unambiguous same-sentence source."""
    from .staged_reasoning import _REPORT_VARIANT, _final_ld_report_scope

    scope = _final_ld_report_scope(context)
    if not scope.provided or not scope.valid:
        return []
    aliases = {_canonical(key): _canonical(value) for key, value in scope.source_aliases}
    named = {aliases.get(_canonical(match.group()), _canonical(match.group()))
             for match in _REPORT_VARIANT.finditer(clause)}
    known_sources = {aliases.get(_canonical(identifier), _canonical(identifier)) for identifier in scope.source_ids}
    plain = re.sub(r'[*`]+', '', clause)
    explicit_proxies = {_canonical(match['v']) for match in re.finditer(
        r'\bproxy\s+(?:variant\s+)?(?P<v>' + _REPORT_VARIANT.pattern + r')', plain, re.I)}
    known_proxies = {_canonical(identifier) for identifier in scope.proxy_ids}
    if (len(named) == len(known_sources) == 1 and scope.source_count == 1
            and len(context.source_observations) == 1 and named <= known_proxies and explicit_proxies == named
            and not re.search(r'\bwith\s+(?:the\s+)?same\s+', clause, re.I)):
        # In a one-source query an explicitly named verified proxy has one
        # source owner. Multi-source reports must still state their partner.
        named |= known_sources
    if len(named) == 1 and re.search(r'\bwith\s+(?:the\s+)?same\s+(?:selected\s+)?source\s+variant\b', clause, re.IGNORECASE):
        # Resolve only the literal same-source phrase inside this sentence.
        # The current proxy must be explicit and the earlier source unique.
        plain = re.sub(r'[*`]+', '', clause)
        proxy_names = {_canonical(match.group('v')) for match in re.finditer(
            r'\bproxy\s+(?:variant\s+)?(?P<v>' + _REPORT_VARIANT.pattern + r')', plain, re.IGNORECASE)}
        antecedents = {aliases.get(_canonical(match.group('v')), _canonical(match.group('v')))
            for match in re.finditer(r'\b(?:selected|source)\s+(?:(?:selected|source|SNP|variant)\s+)*'
                r'(?P<v>' + _REPORT_VARIANT.pattern + r')', re.sub(r'[*`]+', '', sentence_prefix), re.IGNORECASE)}
        # "Selected A and B" declares two potential source antecedents even
        # though the role label is written only once.
        known_sources = {aliases.get(_canonical(identifier), _canonical(identifier)) for identifier in scope.source_ids}
        earlier_sources = {aliases.get(_canonical(match.group()), _canonical(match.group()))
            for match in _REPORT_VARIANT.finditer(sentence_prefix)} & known_sources
        if len(antecedents) == 1 and len(antecedents | earlier_sources) == 1 and proxy_names == named:
            named |= antecedents
    if len(named) != 2:
        return []
    entries = []
    for link in scope.links:
        source, proxy = _canonical(link['selected_variant_id']), _canonical(link['proxy_variant_id'])
        source = aliases.get(source, source)
        if named != {source, proxy}:
            continue
        entries.append({'fact_id': '', 'metric': 'r2', 'value': str(link['r2']),
            'genes': [], 'variants': [proxy], 'selected_variant_id': source,
            'proxy_variant_id': proxy, 'population': link['population'], 'genome_build': link['genome_build'],
            'source_id': f'ld:{source}:{proxy}:{link["population"]}:{link["genome_build"]}',
            'source_sha256': _source_sha(dict(link))})
    return entries


def _explicit_cited_ld_pair_entries(clause, context):
    """A locally named proxy/source pair has its own validated LD authority.

    A trailing molecular citation need not display the same representative
    proxy. Require the actual parenthetical pair, not two unrelated SNP names.
    """
    from .staged_reasoning import _REPORT_VARIANT
    plain = re.sub(r'[*`]+', '', clause)
    pair = re.search(
        r'\b(?:LD\s+)?proxy\s+(?:variant\s+)?(?P<proxy>' + _REPORT_VARIANT.pattern + r')'
        r'\s*\(\s*r(?:²|\^?2)\s*=\s*(?:\d+(?:\.\d+)?|\.\d+)\s+with\s+'
        r'(?:(?:the|selected|uploaded)\s+)*(?:source|selected)\s+(?:(?:SNP|variant)\s+)?'
        r'(?P<source>' + _REPORT_VARIANT.pattern + r')\s*(?:\)|[,;])', plain, re.I)
    if pair is None:
        return None
    entries = _uncited_ld_pair_entries(clause, context)
    proxy = _canonical(pair['proxy'])
    # Pair population/build checks remain in the ordinary LD binding path.
    return [entry for entry in entries if entry['proxy_variant_id'] == proxy]


def _stated_colocalization_studies(clause, known_studies=()):
    """Identify declared endpoints, including a named endpoint from another row."""
    plain = re.sub(r'[*`]', '', clause)
    studies = set(re.findall(r'\b(?:study(?:\s+pair)?\s*[:=]?\s+|vs\s+)'
        r'([A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*)', plain, re.I))
    studies.update(re.findall(r'\bGCST\d+\b', plain, re.I))
    studies.update(study for study in known_studies
        if re.search(r'(?<!\w)' + re.escape(study) + r'(?!\w)', plain, re.I))
    return {study.casefold() for study in studies}


def _bind_colocalization_registry_entries(candidates, clause):
    pairs = [entry['colocalization']['study_pair'] for entry in candidates if 'colocalization' in entry]
    if not pairs:
        return candidates
    stated = _stated_colocalization_studies(clause, (study for pair in pairs for study in pair))
    if not stated:
        return candidates
    return [entry for entry in candidates if stated.issubset({study.casefold()
        for study in entry.get('colocalization', {}).get('study_pair', ())})]


def _explicit_colocalization_pair_entries(clause, measurement, registry, aliases, allowed_fact_ids):
    """Bind a full gene/tissue/QTL/phenotype pair restatement without guessing.

    The source H4 is not used to select its owner. Incomplete labels, ambiguous
    pairs, conflicting study identifiers and per-variant statements stay under
    the ordinary conservative checks.
    """
    from .staged_reasoning import _measurement_trait_key

    if _metric_key(measurement['metric']) != 'h4':
        return None
    prefix = re.sub(r'[*`]', '', clause[:measurement.start()]).strip().casefold()
    if prefix not in {'', 'the', 'a', 'an'}:
        return None
    after = re.sub(r'[*`]', '', clause[measurement.end():])
    stated = re.match(r'\s+recorded\s+for\s+(?P<gene>[A-Z][A-Z0-9-]*)\s+is\s+(?:a|the)\s+'
        r'study[- ]locus\s+pair\s+result\s+between\s+(?:a|an|the)\s+'
        r'(?P<biosample>[A-Za-z][A-Za-z0-9 -]*?)\s+(?P<qtl>(?:e|s|p|tu|sce|scs|scp|sctu)QTL)\s+'
        r'and\s+(?:a|an|the)\s+external\s+GWAS\s+for\s+["“](?P<phenotype>[^"”]+)["”]',
        after, re.IGNORECASE)
    if not stated:
        return None
    gene = next((gene for alias, gene in aliases.items() if alias.casefold() == stated['gene'].casefold()), None)
    if gene is None:
        return []
    phenotype = _measurement_trait_key(stated['phenotype'].rstrip(','))
    candidates = []
    for entry in registry:
        binding = entry.get('colocalization', {})
        if (entry['fact_id'] in allowed_fact_ids and entry['metric'] == 'h4' and gene in entry['genes']
                and binding.get('study_pair') and binding.get('qtl', '').casefold() == stated['qtl'].casefold()
                and binding.get('biosample', '').casefold() == stated['biosample'].strip().casefold()
                and _measurement_trait_key(binding.get('phenotype', '')) == phenotype):
            candidates.append(entry)
    candidates = _bind_colocalization_registry_entries(candidates, clause)
    return candidates if len(candidates) == 1 else []


def _colocalization_lookup_metadata_matches(clause, measurement, entry):
    """Recognize an explicitly named lookup after a pair-level H4 statement.

    Lookup variants are metadata, never H4 owners. Require the same recorded
    gene fact and pair, and retain normal variant ownership for any other use.
    """
    from .staged_reasoning import _REPORT_VARIANT

    if _metric_key(measurement['metric']) != 'h4':
        return False
    binding = entry.get('colocalization', {})
    if not binding.get('study_pair') or not binding.get('lookup_variant'):
        return False
    variants = list(_REPORT_VARIANT.finditer(clause))
    if not variants or any(match.start() < measurement.end() for match in variants):
        return False
    recorded = binding['lookup_variant']
    origin = re.escape(binding['lookup_role'])
    for match in variants:
        if _canonical(match.group()) != recorded:
            return False
        before = re.sub(r'[*`]', '', clause[measurement.end():match.start()])
        after = re.sub(r'[*`]', '', clause[match.end():])
        labelled = (re.search(r'\b(?:using|through)\s+(?:the\s+)?' + origin + r'\s*$', before, re.I)
                    and re.match(r'\s+for\s+(?:the\s+)?lookup\b', after, re.I))
        labelled = labelled or re.search(r'\blookup\s+(?:through|using)\s+(?:the\s+)?'
            + origin + r'\s*$', before, re.I)
        if not labelled:
            return False
    # A new explicitly named study cannot inherit an old pair's statistic.
    studies = _stated_colocalization_studies(clause, binding['study_pair'])
    return studies.issubset({study.casefold() for study in binding['study_pair']})


def _verified_named_proxy_upper_bound(clause, measurement, reported, context, displayed_text):
    """Check one declared upper bound over proxies actually named in the report."""
    from decimal import Decimal
    from .staged_reasoning import _REPORT_VARIANT, _final_ld_report_scope

    if (not re.search(r'\bup\s+to\s*$', clause[measurement.start('metric'):measurement.start('number')], re.I)
            or not re.match(r'\s+for directly named proxies\b', clause[measurement.end():], re.I)):
        return False
    scope = _final_ld_report_scope(context)
    if not scope.provided or not scope.valid or not Decimal('0') <= reported <= Decimal('1'):
        return False
    named = {_canonical(match.group()) for match in _REPORT_VARIANT.finditer(displayed_text)}
    proxies = named & {_canonical(identifier) for identifier in scope.proxy_ids}
    links = [dict(row) for row in scope.links if _canonical(row['proxy_variant_id']) in proxies]
    populations = set(re.findall(r'\b(?:AFR|AMR|EAS|EUR|SAS)\b', clause))
    builds = {value.casefold() for value in re.findall(r'\bGRCh(?:37|38)\b', clause, re.I)}
    if (not links or len(populations) != 1
            or any(row.get('population') not in populations for row in links)
            or builds and any(str(row.get('genome_build', '')).casefold() not in builds for row in links)):
        return False
    return all(Decimal(str(row['r2'])) <= reported for row in links)


def review_report_hard_facts(content: str, context: HybridFinalContext) -> tuple[str, dict]:
    """Review individual measurement spans without deleting scientific narrative.

    Foreign keys declare the available metric sources. Several adjacent fact
    citations are allowed; scientific comparisons and hypotheses are not gates.
    Unknown/ambiguous measurements remain verbatim with separate audit findings.
    """
    from decimal import Decimal

    from .staged_reasoning import (
        _REPORT_VARIANT,
        _SCIENTIFIC_NUMBER,
        _SUMMARY_FACT_CITATION_TOKEN,
        _measurement_trait_key,
        _source_measurement_values,
    )
    corrected, audit = reconcile_report_hard_facts(content, context)
    audit["unverified_measurements"] = []
    payload = _strict_payload(corrected)
    if payload is None:
        return corrected, audit
    included = {identifier for section in payload.get('evidence_sections', ()) if isinstance(section, dict)
                for identifier in section.get('fact_ids', ()) if isinstance(identifier, str)}
    displayed_text = '\n'.join((str(payload.get('summary_markdown', '')),
        str(payload.get('limitations_uncertainty', {}).get('markdown', '')),
        *(fact.markdown for fact in context.facts if fact.fact_id in included)))
    registry, aliases = _measurement_registry(context)
    audit["measurement_registry_sha256"] = _source_sha(registry)
    from .numeric_provenance import report_numeric_ledger
    from .numeric_assertions import finding_category, selection_threshold_assessment
    numeric_ledger = report_numeric_ledger(context, registry)
    audit["numeric_fact_ledger"] = numeric_ledger
    audit["numeric_assertion_trace"] = []
    def source_fact_ids(rows):
        identifiers = {key for row in rows for key in row.get("numeric_fact_ids", ())}
        for row in rows:
            if row.get("source_id"):
                identifiers.update(fact["fact_id"] for fact in numeric_ledger["facts"]
                    if fact["scope"] == "uploaded_gwas" and fact["metric"] == row["metric"]
                    and fact["owner"].get("variant_id") == _canonical(row["source_id"]))
        return sorted(identifiers)
    known_facts = {fact.fact_id for fact in context.facts}
    fact_identifiers = {fact.fact_id: {item.rstrip("-") for item in re.findall(r"\b[A-Z][A-Z0-9-]*\d[A-Z0-9-]*\b", fact.markdown)}
                        for fact in context.facts}
    edits = []
    for field, key in (("summary_markdown", "summary_markdown"), ("limitations_uncertainty.markdown", "markdown")):
        token = _field_tokens(corrected).get(key)
        if token is None:
            continue
        text = json.loads(token.group())
        boundaries = _raw_boundaries(token.group())
        if len(boundaries) != len(text) + 1:
            continue

        def replace_span(start, end, replacement, *, _token=token, _boundaries=boundaries):
            encoded = json.dumps(replacement, ensure_ascii=True)[1:-1]
            edits.append((_token.start() + _boundaries[start], _token.start() + _boundaries[end], encoded))

        from .report_ld_assertions import report_ld_numeric_bindings
        ld_findings = report_ld_numeric_bindings(text, context)
        for finding in ld_findings:
            audit["numeric_assertion_trace"].append({"field": field, "metric": "r2",
                "span": list(finding["span"]), "status": finding["verification_status"],
                "binding_mode": finding["binding_mode"], "numeric_fact_ids": finding.get("numeric_fact_ids", [])})
            if finding["verification_status"] == "verified":
                continue
            if finding["can_correct"]:
                start, end = finding["number_span"]
                replacement = str(finding["values"][0])
                edge = finding["bindings"][0]
                source_id = f"ld:{edge['source']}:{edge['proxy']}:{edge.get('population', '')}:{edge.get('genome_build', '')}"
                replace_span(start, end, replacement)
                audit["corrections"].append({"field": field, "start": start, "end": end,
                    "metric": "r2", "original_value": finding["original_value"], "corrected_value": replacement,
                    "source_type": finding["binding_mode"] if finding["binding_mode"] == "maximum_recorded_selected_source_ld" else "validated_ld_pair", "source_id": source_id,
                    "numeric_fact_ids": finding.get("numeric_fact_ids", []),
                    "source_sha256": _source_sha(list(finding["bindings"])), "ld_edges": list(finding["bindings"]),
                    "reason": "bound_ld_edge_numeric_correction_v1"})
            else:
                audit["unverified_measurements"].append({"field": field,
                    "start": finding["span"][0], "end": finding["span"][1], "metric": "r2",
                    "original_value": finding["original_value"], "original_span": finding["original_span"],
                    "replacement": "Original statement retained for review", "reason": finding["reason"],
                    "verification_status": finding["verification_status"], "numeric_consistent": finding["numeric_consistent"],
                    "binding_mode": finding["binding_mode"], "candidate_fact_ids": finding["evidence_ids"],
                    "numeric_fact_ids": finding.get("numeric_fact_ids", []), "finding_category": finding["finding_category"],
                    "candidate_bindings": list(finding["bindings"]), "rule": "typed_ld_assertion_v1"})

        for clause_match in _CLAUSE.finditer(text):
            clause = clause_match.group()
            from .ranking_claims import _METRIC, _metric_name
            for threshold in _METRIC.finditer(clause):
                assessment = selection_threshold_assessment(clause, threshold, _metric_name(threshold["metric"]), numeric_ledger)
                if assessment is None:
                    continue
                audit["numeric_assertion_trace"].append({"field": field, "metric": _metric_name(threshold["metric"]),
                    "span": [clause_match.start() + threshold.start(), clause_match.start() + threshold.end()], **assessment})
                if assessment.get("reason"):
                    audit["unverified_measurements"].append({"field": field, "metric": _metric_name(threshold["metric"]),
                        "start": clause_match.start() + threshold.start(), "end": clause_match.start() + threshold.end(),
                        "original_value": threshold["number"], "original_span": threshold.group(),
                        "replacement": "Original statement retained for review", "reason": assessment["reason"],
                        "finding_category": assessment["status"], "numeric_fact_ids": assessment["numeric_fact_ids"]})
            citations = set(_SUMMARY_FACT_CITATION_TOKEN.findall(clause))
            # A source-GWAS statistic can share a sentence with a citation.
            # The citation may support the surrounding biological statement,
            # but it does not change ownership of an explicitly uploaded
            # measurement.  Resolve source rows independently, then bind the
            # citation to the other claims in the clause.  Suppressing source
            # rows whenever a citation was present caused valid user-uploaded
            # P values to be replaced by “P value not verified”.
            sources = _source_owner_rows(clause, context)
            measurements = list(_SURFACE_MEASUREMENT.finditer(clause))
            # Bare source-P values still have an explicit dimension and owner
            # in the same source statement. Proven normalization aliases refer
            # to one observation, not two independent SNPs.
            if len(sources) == 1 and not measurements and re.search(r"\bP\b", clause, re.IGNORECASE):
                try:
                    expected = Decimal(str(sources[0]["p_value"]))
                except (KeyError, ArithmeticError, ValueError):
                    expected = None
                if expected is not None:
                    for number in _SCIENTIFIC_NUMBER.finditer(clause):
                        try:
                            value = Decimal(number.group().replace(",", ""))
                        except (ArithmeticError, ValueError):
                            continue
                        if not (0 < value <= 1) or not _rounds_to_authority(value, expected) or value == expected:
                            continue
                        start, end = clause_match.start() + number.start(), clause_match.start() + number.end()
                        replace_span(start, end, str(expected))
                        audit["corrections"].append({"field": field, "start": start, "end": end,
                            "metric": "p", "original_value": number.group(), "corrected_value": str(expected),
                            "source_type": "source_observation", "source_id": sources[0]["variant_id"],
                            "source_sha256": _source_sha(dict(sources[0])), "reason": "source_precision_normalized"})
            for measurement in measurements:
                if _configured_gwas_threshold(clause, measurement):
                    continue
                name = _metric_key(measurement["metric"])
                if name == "r2" and any(left <= clause_match.start() + measurement.start() < right
                        for left, right in (finding["span"] for finding in ld_findings)):
                    continue
                unfinished_end = _unfinished_scientific_end(clause, measurement)
                if unfinished_end is not None:
                    start = clause_match.start() + measurement.start()
                    end = clause_match.start() + unfinished_end
                    audit["unverified_measurements"].append({"field": field, "start": start, "end": end,
                        "metric": name, "original_value": clause[measurement.start('number'):unfinished_end],
                        "original_span": clause[measurement.start():unfinished_end],
                        "replacement": "Original statement retained for review",
                        "candidate_fact_ids": [], "reason": "measurement_source_not_verified"})
                    continue
                declared_sources = _explicit_selected_source_p_rows(clause, measurement, context)
                measurement_sources = sources if declared_sources is None else declared_sources
                try:
                    reported = _surface_value(measurement)
                except (ArithmeticError, ValueError):
                    reported = None
                if (name == 'r2' and reported is not None and reported.is_finite()
                        and _verified_named_proxy_upper_bound(clause, measurement, reported, context, displayed_text)):
                    continue
                binding_clause, local_citations = _measurement_citation_scope(clause, measurement)
                candidates = [entry for entry in registry if entry["fact_id"] in local_citations and entry["metric"] == name]
                prefix = clause[:measurement.start()]
                owners = [(match.end(), gene) for alias, gene in aliases.items()
                          for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", prefix, re.IGNORECASE)]
                # An explicitly written identifier cannot borrow another
                # gene's metric merely because its value appears in a citation.
                unknown_identifiers = [match for match in re.finditer(r"\b[A-Z][A-Z0-9-]*\d[A-Z0-9-]*\b", prefix)
                    if match.group() not in aliases
                    and _measurement_trait_key(match.group()) != _measurement_trait_key(context.source_trait)
                    and match.group() not in {"H3", "H4", "R2"}
                    and not any(match.group().rstrip("-") in fact_identifiers.get(identifier, ()) for identifier in citations)]
                # Variants following P/Beta ("P=... on proxy ... {{cite_fact}}").
                # belong to the same local claim too; an earlier cited row does not.
                variant_mentions = list(_REPORT_VARIANT.finditer(binding_clause))
                respectively_owner = _respectively_measurement_owner(clause, measurement, aliases)
                if owners or respectively_owner:
                    gene = respectively_owner or max(owners)[1]
                    candidates = [entry for entry in candidates if gene in entry["genes"]]
                if name == 'h4':
                    pair_entries = _explicit_colocalization_pair_entries(
                        clause, measurement, registry, aliases, local_citations or included)
                    if pair_entries is not None:
                        candidates = pair_entries
                    candidates = _bind_colocalization_registry_entries(candidates, binding_clause)
                    if respectively_owner:
                        candidates = _respectively_colocalization_entries(candidates, clause, context)
                if name == 'r2' and not local_citations:
                    boundary = clause.rfind(';', 0, measurement.start()) + 1
                    candidates = _uncited_ld_pair_entries(binding_clause, context, sentence_prefix=clause[:boundary])
                elif name == 'r2':
                    explicit_pair = _explicit_cited_ld_pair_entries(binding_clause, context)
                    if explicit_pair is not None:
                        # An explicit but absent pair cannot borrow the cited
                        # representative's other edge merely by naming its source.
                        keys = {(entry['selected_variant_id'], entry['proxy_variant_id'],
                                 entry['population'], entry['genome_build']) for entry in explicit_pair}
                        candidates = [entry for entry in candidates if
                            (entry.get('selected_variant_id'), entry.get('proxy_variant_id'),
                             entry.get('population'), entry.get('genome_build')) in keys] or explicit_pair
                ld_edges = name == "r2" and any("selected_variant_id" in entry for entry in candidates)
                if ld_edges:
                    candidates = [entry for entry in candidates if "selected_variant_id" in entry]
                    candidates = _bind_ld_registry_entries(candidates, binding_clause, context)
                    scope = {entry["proxy_variant_id"] for entry in candidates}
                    named = {_canonical(match.group()) for match in variant_mentions}
                    named_proxies = named.intersection(scope)
                    if named_proxies:
                        candidates = [entry for entry in candidates if entry["proxy_variant_id"] in named_proxies]
                    elif variant_mentions and not any(_canonical(match.group()) == entry["selected_variant_id"]
                            for match in variant_mentions for entry in candidates):
                        candidates = []
                elif variant_mentions:
                    named = {_canonical(match.group()) for match in variant_mentions}
                    candidates = [entry for entry in candidates if named.issubset(set(entry["variants"]))
                        or owners and _colocalization_lookup_metadata_matches(
                            clause, measurement, entry)]
                # A verified LD edge is owned by two SNPs and its reference
                # population, not by a gene or a cell marker such as CD4.
                # Entity review still independently flags unknown identifiers.
                if not ld_edges and unknown_identifiers and (not owners or unknown_identifiers[-1].end() > max(owners)[0]):
                    candidates = []
                if declared_sources is not None:
                    # A cited molecular statistic cannot substitute for an
                    # explicitly requested uploaded-source measurement.
                    candidates = []
                if measurement_sources:
                    candidates = []
                    for row in measurement_sources:
                        for value in _source_measurement_values(row, name):
                            candidates.append({"fact_id": "", "metric": name, "value": str(value),
                                "source_id": row["variant_id"], "source_sha256": _source_sha(dict(row))})
                # Probabilities support an exact percent-to-fraction conversion.
                # A percentage attached to beta, SE or OR cannot borrow the
                # same bare scalar: its measurement scale is a separate fact.
                unit_mismatch = False
                if name not in _PROBABILITY_METRICS:
                    unit = "%" if measurement["percent"] else ""
                    scoped = [entry for entry in candidates if entry.get("unit", "") == unit]
                    unit_mismatch = bool(candidates) and not scoped
                    candidates = scoped
                matching = [entry for entry in candidates if _rounds_to_authority(reported, Decimal(entry["value"]))]
                values = {entry["value"] for entry in candidates}
                if (len(measurement_sources) > 1 or ld_edges) and len(values) > 1:
                    matching = []
                start = clause_match.start() + measurement.start()
                end = clause_match.start() + measurement.end()
                if matching:
                    audit["numeric_assertion_trace"].append({"field": field, "metric": name, "span": [start, end],
                        "status": "verified", "numeric_fact_ids": source_fact_ids(matching)})
                    expected_values = {entry["value"] for entry in matching}
                    if len(expected_values) == 1 and not measurement["percent"]:
                        expected = next(iter(expected_values))
                        if reported != Decimal(expected):
                            numeric_start = clause_match.start() + measurement.start("number")
                            numeric_end = clause_match.start() + measurement.end("number")
                            replace_span(numeric_start, numeric_end, expected)
                            audit["corrections"].append({"field": field, "start": numeric_start, "end": numeric_end,
                                "metric": name, "original_value": measurement["number"], "corrected_value": expected,
                                "source_type": "metric_registry", "source_id": matching[0].get("source_id") or matching[0]["fact_id"],
                                "source_sha256": matching[0]["source_sha256"], "reason": "input_precision_normalized"})
                    continue
                # A new value may only be corrected when the declared sources
                # agree on exactly one value and the owner is explicit.
                if len(values) == 1 and (measurement_sources or owners or variant_mentions):
                    expected = next(iter(values))
                    numeric_start = clause_match.start() + measurement.start("number")
                    numeric_end = clause_match.start() + measurement.end()
                    replacement = expected + ("%" if candidates[0].get("unit") == "%" else "")
                    replace_span(numeric_start, numeric_end, replacement)
                    audit["corrections"].append({"field": field, "start": numeric_start, "end": numeric_end,
                        "metric": name, "original_value": measurement["number"], "corrected_value": replacement,
                        "source_type": "metric_registry", "source_id": candidates[0].get("source_id") or candidates[0]["fact_id"],
                        "source_sha256": candidates[0]["source_sha256"], "reason": "unique_declared_metric_value"})
                    audit["corrections"][-1]["numeric_fact_ids"] = source_fact_ids(candidates)
                    continue
                audit["unverified_measurements"].append({"field": field, "start": start, "end": end, "metric": name,
                    "original_value": measurement["number"] + ("%" if measurement["percent"] else ""),
                    "original_span": measurement.group(),
                    "replacement": "Original statement retained for review",
                    "candidate_fact_ids": sorted({entry["fact_id"] for entry in candidates if entry["fact_id"]}),
                    "numeric_fact_ids": source_fact_ids(candidates),
                    "reason": "measurement_unit_not_verified" if unit_mismatch else
                        "measurement_not_uniquely_bound" if candidates else "measurement_source_not_verified"})
        for citation in _SUMMARY_FACT_CITATION_TOKEN.finditer(text):
            if citation.group(1) in known_facts:
                continue
            # A citation to an unknown fact is fabricated; remove it with its leading space.
            replacement = ""
            start = citation.start() - (1 if citation.start() and text[citation.start() - 1] == " " else 0)
            replace_span(start, citation.end(), replacement)
            audit["unverified_measurements"].append({"field": field, "start": citation.start(), "end": citation.end(),
                "metric": "citation", "original_value": citation.group(1), "replacement": replacement,
                "reason": "unknown_fact_id"})
        from .gate2_hallucination import invalid_probability_sentence_edits
        for edit in invalid_probability_sentence_edits(text, [item for item in audit['unverified_measurements']
                                                              if item.get('field') == field]):
            raw_start = token.start() + boundaries[edit['start']]
            raw_end = token.start() + boundaries[edit['end']]
            edits = [item for item in edits if not (raw_start <= item[0] and item[1] <= raw_end)]
            replace_span(edit['start'], edit['end'], '')
    for start, end, replacement in sorted(edits, reverse=True):
        corrected = corrected[:start] + replacement + corrected[end:]
    corrected = _tidy_removed_fields(corrected)
    # Explicit unqualified "only candidate with a measured QTL" is a fact
    # about this report's rows, not an inference about biological importance.
    from .novelty import novelty_claim_corrections
    from .report_evidence_coverage import (
        clpp_method_label_corrections,
        exclusive_molqtl_corrections,
        gene_evidence_ownership_corrections,
        ranking_validation_claim_corrections,
        source_variant_measurement_absence_corrections,
    )
    from .report_facts import source_lead_label_corrections
    from .report_rank_facts import saved_rank_reference_corrections
    from .report_retrieval_scope import (
        bounded_literature_state_corrections,
        external_credible_set_state_corrections,
        source_effect_direction_corrections,
    )
    from .report_source_counts import (
        report_scope_count_corrections,
        source_significance_count_corrections,
    )
    from .report_variant_names import source_variant_name_corrections
    presence_edits = []
    for field, key in (("summary_markdown", "summary_markdown"), ("limitations_uncertainty.markdown", "markdown")):
        token = _field_tokens(corrected).get(key)
        if token is None:
            continue
        text = json.loads(token.group())
        boundaries = _raw_boundaries(token.group())
        if len(boundaries) != len(text) + 1:
            continue
        for edit in (*clpp_method_label_corrections(text, context), *exclusive_molqtl_corrections(text, context),
                     *gene_evidence_ownership_corrections(text, context), *source_significance_count_corrections(text, context),
                     *report_scope_count_corrections(text, context),
                     *source_lead_label_corrections(text, context),
                     *novelty_claim_corrections(text, context), *bounded_literature_state_corrections(text, context),
                     *external_credible_set_state_corrections(text, context),
                     *source_effect_direction_corrections(text, context),
                     *source_variant_measurement_absence_corrections(text, context),
                     *ranking_validation_claim_corrections(text, context),
                     *saved_rank_reference_corrections(text, context),
                     *source_variant_name_corrections(text, context)):
            start, end = edit["start"], edit["end"]
            presence_edits.append((token.start() + boundaries[start], token.start() + boundaries[end],
                                  json.dumps(edit["replacement"], ensure_ascii=True)[1:-1]))
            audit["corrections"].append({"field": field, "metric": "evidence_presence",
                "original_value": text[start:end], "corrected_value": edit["replacement"],
                "source_type": "source_observations" if edit.get("metric") in {"source_significance_count", "source_variant_count"} else "display_fact_registry", **edit})
    for start, end, replacement in sorted(presence_edits, reverse=True):
        corrected = corrected[:start] + replacement + corrected[end:]
    audit["corrected_json_sha256"] = _sha(corrected)
    for disposition, findings in (("corrected", audit["corrections"]), ("unverified", audit["unverified_measurements"])):
        for finding in findings:
            if finding.get("metric") not in {"p", "h4", "h3", "r2", "beta", "pip", "clpp"}:
                continue
            finding.setdefault("finding_category", finding_category(finding.get("reason"), disposition=disposition))
            if not finding.get("numeric_fact_ids") and finding.get("source_id"):
                rows = [entry for entry in registry if entry["fact_id"] == finding["source_id"]
                        and entry["metric"] == finding["metric"]]
                if finding.get("source_type") == "source_observation":
                    rows = [{"source_id": finding["source_id"], "metric": finding["metric"]}]
                finding["numeric_fact_ids"] = source_fact_ids(rows)
    return corrected, audit
