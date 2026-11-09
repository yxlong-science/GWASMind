"""Precisely correct a stale retrieval-state sentence against the reviewed scope."""
import re


def source_effect_direction_corrections(text, context):
    """Label explicit unsupported source-direction claims, preserving other prose."""
    from .staged_reasoning import _validate_source_effect_direction, StructuredOutputError
    source = r'(?:source|uploaded|loaded|current)[- ](?:GWAS|dataset|data)'
    comparison = (r'(?P<comparison>(?:(?:is|are|was|were)\s+)?(?:consistent|concordant|aligned|inconsistent|discordant)\s+with\s+(?:the\s+)?'
        r'(?:effect\s+)?direction(?:\s+(?:reported|observed|seen))?\s+(?:in|from|of)\s+(?:the\s+)?' + source + r')')
    same = (r'(?P<same>(?:has|have)\s+(?:the\s+)?(?:same|consistent|concordant|opposite|discordant)\s+'
        r'(?:effect\s+)?direction\s+as\s+(?:that\s+(?:in|from)\s+)?(?:the\s+)?' + source + r')')
    own = (r'(?P<own>' + source + r"(?:['’]s)?\s+(?:effect\s+)?direction\s+(?:is|was|appears)\s+"
        r'(?:(?:the|a)\s+)?(?:positive|negative|neutral|zero|same|opposite|consistent|concordant)\b)')
    pattern = re.compile(comparison + '|' + same + '|' + own, re.I)
    edits = []
    for clause in re.finditer(r'.+?(?:(?<=[.!?])(?=\s)|(?=\n)|$)', text, re.S):
        matches = list(pattern.finditer(clause.group()))
        # A mixed sentence requires per-claim ownership beyond this narrow pass.
        if len(matches) != 1:
            continue
        match = matches[0]
        try:
            _validate_source_effect_direction(clause.group(), context=context, label='source direction')
        except StructuredOutputError:
            verb = re.match(r'(is|are|was|were|has|have)\b', match.group(), re.I)
            replacement = (('have' if verb[1].lower() in {'are', 'were', 'have'} else 'has')
                           + ' no verified comparison with the source-GWAS effect direction' if verb else
                           'with the source-GWAS effect direction not verified' if match['comparison'] else
                           'source-GWAS effect direction is not verified')
            edits.append({'start': clause.start() + match.start(), 'end': clause.start() + match.end(),
                'replacement': replacement, 'metric': 'evidence_presence', 'fact_kind': 'source_effect_direction',
                'source_type': 'source_effect_observations',
                'source_observation_ids': [row.get('variant_id') for row in context.source_observations],
                'reason': 'source_effect_direction_not_supported_by_uploaded_effect_provenance'})
    return edits


def external_credible_set_state_corrections(text, context):
    """Correct an explicit absolute denial against current retrieved records."""
    from .staged_reasoning import _final_analysis_scope, _runtime_evidence_rows
    if not _final_analysis_scope(context).get('external_credible_set_information_available'):
        return []
    pattern = re.compile(r'(?<!\S)No external credible-set information was supplied\.(?=\s|$)', re.I)
    rows = _runtime_evidence_rows(context.result)
    witnesses = [str(row.get('evidence_id')) for row in rows
                 if isinstance(row.get('payload'), dict)
                 and row['payload'].get('module') == 'locus.credible_set' and row.get('evidence_id')]
    edits = []
    for match in pattern.finditer(text):
        if re.split(r'[.!?;\n]', text[:match.start()])[-1].strip():
            continue
        edits.append({'start': match.start(), 'end': match.end(),
            'replacement': 'External credible-set information is available in the retrieved evidence; this does not establish that the uploaded GWAS supplied a credible set.',
            'metric': 'external_credible_set_availability', 'source_type': 'retrieved_evidence_scope',
            'source_evidence_ids': witnesses,
            'reason': 'absolute_external_credible_set_absence_contradicts_retrieved_records'})
    return edits


_EUROPE_PMC_UNFINISHED = re.compile(re.escape(
    "The requested evidence for the Europe PMC literature module is not fully returned and remains unfinished"), re.I)


def bounded_literature_state_corrections(text, context):
    facts = context.safety_review.facts
    scope = facts.review_scope
    module = "evidence.europepmc"
    if module not in scope.get("bounded_literature_modules", ()):
        return []
    if scope.get("requirements", {}).get("require_exhaustive_literature") is True:
        return []
    if scope.get("module_statuses", {}).get(module) not in {"success", "completed", "truncated", "no_data"}:
        return []
    if any(str(check.outcome) in {"warn", "fail"} and module in check.affected.get("modules", ())
           for check in facts.checks):
        return []
    edits = []
    for match in _EUROPE_PMC_UNFINISHED.finditer(text):
        prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
        if re.search(r"\b(?:if|whether|suppose|incorrect|not)\b", prefix, re.I):
            continue
        edits.append({"start": match.start(), "end": match.end(),
            "replacement": "The Europe PMC literature module completed its declared per-gene subset; this view does not cover all publications",
            "metric": "retrieval_state", "source_type": "reviewed_retrieval_scope", "source_id": module,
            "source_sha256": scope.get("sha256"),
            "reason": "declared_bounded_literature_is_not_an_unfinished_query"})
    return edits
