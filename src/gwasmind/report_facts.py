"""Computed report facts and explicit execution provenance, independent of model prose."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from .configuration import parameter

_VARIANT = re.compile(r"\b(?:chr)?(?:\d+|X|Y|MT)[_:]\d+[_:][ACGTN]+[_:][ACGTN]+\b", re.IGNORECASE)
_NUMBERS = {word: number for number, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"))}
_COUNT = r"(?:\d+|" + "|".join(_NUMBERS) + r")"
_RS_ID = r"rs\d+"
_VARIANT_MENTION = re.compile(
    rf"(?:{_RS_ID}\s*\(\s*{_VARIANT.pattern}\s*\)|"
    rf"{_VARIANT.pattern}(?:\s*\(\s*{_RS_ID}\s*\))?|\b{_RS_ID}\b)", re.IGNORECASE)
_COMPARISON_PREDICATE = re.compile(
    r"\s*(?:is|was|are|were|has|have|had|carries|carry|shows|show|yields|yield)\s+"
    r"(?:(?:jointly|joint|tied\s+for|among)\s+)?(?:the\s+)?", re.IGNORECASE)
_COORDINATED_MENTION = re.compile(r"\s*(?:,\s*(?:(?:and|&)\s*)?|(?:and|&)\s*)", re.IGNORECASE)


def _comparative_subjects(sentence: str, comparison: re.Match, facts: Mapping[str, Any]):
    """Bind a comparative to its explicit grammatical SNP subject.

    Parenthetical rs names are accepted as mention syntax. An optional already
    verified alias map can check them; this helper never guesses an rs mapping
    or searches the evidence ledger. Other scientific guards still apply.
    """
    aliases = facts.get("source_variant_aliases")
    aliases = aliases if isinstance(aliases, Mapping) else {}

    def resolve(mention):
        canonical = _VARIANT.search(mention.group())
        rs = re.search(rf"\b{_RS_ID}\b", mention.group(), re.IGNORECASE)
        known = aliases.get(rs.group().lower()) if rs else None
        identity = _variant(canonical.group()) if canonical else _variant(known) if isinstance(known, str) else None
        if identity is not None and isinstance(aliases.get(identity), str):
            identity = _variant(aliases[identity])
        conflict = bool(canonical and isinstance(known, str) and _variant(known) != identity)
        return identity, conflict

    preceding = list(_VARIANT_MENTION.finditer(sentence[:comparison.start()]))
    if preceding and _COMPARISON_PREDICATE.fullmatch(sentence[preceding[-1].end():comparison.start()]):
        # An explicit preceding subject owns the statement even when another
        # SNP is mentioned later in the same sentence.
        mentions = [preceding[-1]]
        for earlier in reversed(preceding[:-1]):
            between = sentence[earlier.end():mentions[0].start()]
            if not _COORDINATED_MENTION.fullmatch(between):
                break
            mentions.insert(0, earlier)
        return [resolve(mention) for mention in mentions]
    following = _VARIANT_MENTION.search(sentence[comparison.end():comparison.end() + 140])
    if following:
        prefix = sentence[comparison.end():comparison.end() + following.start()]
        if not re.search(r";|\b(?:but|whereas|while|however|with)\b", prefix, re.IGNORECASE):
            mentions = [following]
            after = sentence[comparison.end():]
            for later in _VARIANT_MENTION.finditer(after, following.end()):
                if not _COORDINATED_MENTION.fullmatch(after[mentions[-1].end():later.start()]):
                    break
                mentions.append(later)
            return [resolve(mention) for mention in mentions]
    return []


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _variant(value: str) -> str:
    return re.sub(r"^chr", "", value, flags=re.IGNORECASE).upper().replace(":", "_")


def public_source_normalizations(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project verified corrections without exposing full provider receipts to UI."""
    from .source_normalization import validate_source_normalization

    resolution = result.get("variant_resolution", {})
    proofs = resolution.get("normalizations", []) if isinstance(resolution, Mapping) else []
    if not isinstance(proofs, (list, tuple)):
        return []
    output = {}
    for proof in proofs:
        if validate_source_normalization(proof):
            output[proof["original_id"]] = {key: proof[key] for key in (
                "original_id", "canonical_id", "genome_build", "reason_code",
                "source_p_preserved", "effect_sign_changed")}
    return list(output.values())


def computed_report_facts(observations: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> dict[str, Any]:
    from .source_gwas import summarize_source_gwas_observations

    source = [row for row in observations if isinstance(row, Mapping)]
    source_significance = summarize_source_gwas_observations(source, phenotype="")
    source_ids = {row.get("variant_id") for row in source}
    aliases = {row["original_id"]: row["canonical_id"] for row in public_source_normalizations(result)
               if row["original_id"] in source_ids}
    values = []
    for row in source:
        try:
            p = Decimal(str(row.get("p_value")))
        except (InvalidOperation, ValueError):
            continue
        identifier = row.get("variant_id")
        if p.is_finite() and 0 < p <= 1 and isinstance(identifier, str):
            values.append((aliases.get(identifier, identifier), p))
    complete_p = bool(source) and len(values) == len(source)
    minimum = min((p for _, p in values), default=None)
    coverage = result.get("retrieval_metrics", result.get("retrieval_coverage"))
    coverage = coverage if isinstance(coverage, Mapping) else {}
    shortlist = coverage.get("shortlist")
    shortlist = shortlist if isinstance(shortlist, Mapping) else {}
    audit = result.get("candidate_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    shortlist_count = _count(coverage.get("candidate_genes"))
    if shortlist_count is None:
        shortlist_count = _count(audit.get("shortlist_size"))
    pool_count = None
    for pool in (result.get("candidate_pool"), audit.get("genes")):
        if isinstance(pool, (list, tuple)) and all(
            isinstance(row, Mapping) and isinstance(row.get("gene_id"), str) and row["gene_id"]
            for row in pool
        ):
            pool_count = len({row["gene_id"] for row in pool})
            break
    if pool_count is None:
        # Discovery partitions the pool into shortlisted genes and the remaining
        # context genes. context_gene_count alone is NOT the whole candidate pool.
        context_count = _count(shortlist.get("context_gene_count"))
        if context_count is not None and shortlist_count is not None:
            pool_count = context_count + shortlist_count
    ranked = result.get("ranked_genes")
    ranked = [row for row in ranked if isinstance(row, Mapping)] if isinstance(ranked, (list, tuple)) else []
    decisions = result.get("candidate_decisions")
    if isinstance(decisions, (list, tuple)) and decisions:
        ids = [row.get("gene_id") for row in decisions if isinstance(row, Mapping)]
        if (len(ids) == len(decisions) and len(set(ids)) == len(ids)
                and all(isinstance(value, str) and value for value in ids)
                and all(row.get("status") in {"selected", "not_selected", "unresolved"} for row in decisions)):
            pool_count = len(ids)
            shortlist_count = sum(row["status"] == "selected" for row in decisions)
    elif audit.get("mode") == "gene_centric_window" or audit.get("v2g_cards"):
        # Gene-window discovery has not performed model shortlisting yet.
        pool_count = pool_count if pool_count is not None else shortlist_count
        shortlist_count = None
    displayed = ranked[:parameter("ranking.max_ranked_genes")]
    scope = result.get("ld_expansion")
    scope = scope if isinstance(scope, Mapping) else {}
    expanded = scope.get("analysis_variant_count")
    if _count(expanded) is None:
        expanded = scope.get("expanded_variant_count")
    if _count(expanded) is None and isinstance(scope.get("analysis_variants"), (list, tuple)):
        expanded = len(scope["analysis_variants"])
    verified_proxies = _count(scope.get("added_variant_count"))
    if verified_proxies is not None and _count(expanded) is not None:
        if expanded != len(source) + verified_proxies:
            # An inconsistent receipt cannot establish how many proxies were
            # verified; never substitute the enumerated/checked candidate count.
            verified_proxies = None
    return {
        "source_variant_count": len(source),
        "source_variants_with_p_value": source_significance.variants_with_p_value,
        "source_significant_variant_count": source_significance.genome_wide_significant_variants,
        "source_significance_threshold": source_significance.significance_threshold,
        "source_p_values_complete": complete_p,
        "source_minimum_p": str(minimum) if minimum is not None else None,
        "source_lead_variant_ids": [identifier for identifier, p in values if p == minimum] if complete_p else [],
        **({"source_variant_aliases": aliases} if aliases else {}),
        "analysis_variant_count": _count(expanded),
        "verified_proxy_count": verified_proxies,
        "candidate_pool_count": pool_count,
        "shortlist_count": shortlist_count,
        "ranked_count": len(ranked),
        "ranked_gene_ids": [row["gene_id"] for row in ranked if isinstance(row.get("gene_id"), str)],
        "displayed_ranked_count": len(displayed),
        "displayed_ranked_gene_ids": [row["gene_id"] for row in displayed if isinstance(row.get("gene_id"), str)],
    }


def analysis_overview(observations: Sequence[Mapping[str, Any]], result: Mapping[str, Any],
                      *, intent: str, source_trait: str = "") -> dict[str, Any]:
    """Supply recorded analysis scope for the model's opening paragraph.

    This is factual input, not generated prose. An absent stage has no invented
    zero count. Coordinate spans describe selected positions per chromosome,
    rather than claiming that a multi-region selection is one continuous locus.
    """
    source = [row for row in observations if isinstance(row, Mapping)]
    facts = computed_report_facts(source, result)
    overview: dict[str, Any] = {"intent": intent}
    if source:
        selection: dict[str, Any] = {"variant_count": facts["source_variant_count"]}
        if source_trait:
            selection["source_trait"] = source_trait
        spans: dict[str, list[int]] = {}
        builds = set()
        for row in source:
            identifier = _variant(str(row.get("variant_id") or ""))
            if not _VARIANT.fullmatch(identifier):
                spans = {}
                break
            chromosome, position_text, _, _ = identifier.split("_")
            position = int(position_text)
            supplied_chromosome = str(row.get("chromosome") or chromosome)
            supplied_position = row.get("position", position)
            if (_variant(supplied_chromosome) != chromosome or position <= 0
                    or isinstance(supplied_position, bool)
                    or str(supplied_position) != str(position)):
                spans = {}
                break
            spans.setdefault(chromosome, []).append(position)
            build = row.get("genome_build")
            builds.add(build if isinstance(build, str) else None)
        if spans:
            selection["selected_spans"] = [{"chromosome": chromosome,
                "start_bp": min(positions), "end_bp": max(positions),
                "selected_variant_count": len(positions)}
                for chromosome, positions in sorted(spans.items())]
            if len(builds) == 1 and next(iter(builds)) in {"GRCh37", "GRCh38"}:
                selection["genome_build"] = next(iter(builds))
        if facts["source_variants_with_p_value"]:
            selection["source_gwas"] = {"variants_with_p_value": facts["source_variants_with_p_value"],
                "significant_variant_count": facts["source_significant_variant_count"],
                "significance_threshold": facts["source_significance_threshold"],
                "p_values_complete": facts["source_p_values_complete"]}
            if facts["source_p_values_complete"]:
                selection["source_gwas"].update(minimum_p=facts["source_minimum_p"],
                    lead_variant_ids=facts["source_lead_variant_ids"])
        overview["source_selection"] = selection
    scope = result.get("ld_expansion")
    if isinstance(scope, Mapping) and (
        scope.get("ld_lookup_performed") is True
        or (facts["verified_proxy_count"] is not None and facts["verified_proxy_count"] > 0)
    ):
        expansion = {"lookup_performed": True}
        for key in ("verified_proxy_count", "analysis_variant_count"):
            if facts[key] is not None:
                expansion[key] = facts[key]
        if isinstance(scope.get("population"), str) and scope["population"]:
            expansion["reference_population"] = scope["population"]
        overview["ld_expansion"] = expansion
    candidates = {key: facts[key] for key in ("candidate_pool_count", "shortlist_count")
                  if facts[key] is not None}
    if candidates:
        overview["candidate_discovery"] = candidates
    if intent == "prioritize_genes" and isinstance(result.get("ranked_genes"), (list, tuple)):
        overview["gene_prioritization"] = {key: facts[key] for key in (
            "ranked_count", "displayed_ranked_count")}
    return overview


def report_fact_issues(text: str, facts: Mapping[str, Any]) -> tuple[str, ...]:
    """Check explicit comparative/count claims; leave uncertain prose to existing guards."""
    plain = re.sub(r"[*`]", "", text)
    issues = []
    lead_ids = {_variant(value) for value in facts.get("source_lead_variant_ids", [])}
    comparative = re.compile(r"\b(?:strongest|most\s+significant|(?:smallest|lowest)\s+(?:(?:selected[- ]source|source|selected|uploaded|input)\s+)?P(?:[- ]value)?)\b", re.IGNORECASE)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", plain):
        if not re.search(r"\b(?:uploaded|source|input|selected)\b", sentence, re.IGNORECASE):
            continue
        for comparison in comparative.finditer(sentence):
            # "Its strongest support/evidence is an eQTL at source SNP X"
            # compares evidence for a gene, not uploaded association P values.
            # A separate strongest-variant or smallest-P claim is still checked.
            if comparison.group().casefold() == "strongest" and re.match(
                r"\s+(?:support|evidence)\b", sentence[comparison.end():], re.IGNORECASE
            ):
                continue
            nearby = sentence[max(0, comparison.start() - 75):comparison.end() + 140]
            if re.search(r"\b(?:not|cannot|unclear|uncertain|whether|incorrect|falsely)\b", nearby[:comparison.start() - max(0, comparison.start() - 75)], re.IGNORECASE):
                continue
            # A scope explicitly restricted to an external study is not a
            # superlative about the uploaded source data earlier in the sentence.
            clause = re.split(r";|\b(?:but|whereas|while)\b", sentence[:comparison.start()], flags=re.IGNORECASE)[-1]
            scope_terms = list(re.finditer(r"\b(?:source|uploaded|input|selected|external)\b", clause, re.IGNORECASE))
            if scope_terms and scope_terms[-1].group().lower() == "external":
                continue
            if re.match(r"\s+(?:external|published)\b", sentence[comparison.end():], re.IGNORECASE):
                continue
            for subject, alias_conflict in _comparative_subjects(sentence, comparison, facts):
                if alias_conflict:
                    issues.append("computed source-GWAS facts: variant and verified rs alias disagree")
                if subject and lead_ids and subject not in lead_ids:
                    issues.append("computed source-GWAS facts: the strongest selected source variant must be " + ", ".join(sorted(lead_ids)))
    scopes = (
        ("candidate_pool_count", rf"\bcandidate\s+pool\b[^.;\n]{{0,60}}?\b(?:contains|includes|has|comprises|consists\s+of)\s+(?P<n>{_COUNT})\s+(?:candidate\s+)?genes?\b"),
        ("candidate_pool_count", rf"\b(?P<n>{_COUNT})[- ]gene\s+(?:candidate\s+)?pool\b"),
        ("shortlist_count", rf"\bshortlist\b[^.;\n]{{0,80}}?\b(?:contains|includes|has|comprises|consists\s+of)\s+(?P<n>{_COUNT})\s+(?:candidate\s+)?genes?\b"),
        ("shortlist_count", rf"\b(?:there\s+are|a\s+total\s+of)\s+(?P<n>{_COUNT})\s+(?:shortlisted|short-listed)\s+(?:candidate\s+)?genes?\b"),
        ("source_variant_count", rf"\b(?:a\s+total\s+of|all|there\s+are)\s+(?P<n>{_COUNT})\s+(?:(?:user[- ]selected|source[- ]selected|uploaded|input|selected)\s+)(?:source\s+)?(?:SNPs?|variants?)\b"),
        ("ranked_count", rf"\b(?:returned|ranked)\s+(?P<n>{_COUNT})\s+genes?\b"),
    )
    for name, pattern in scopes:
        expected = facts.get(name)
        if type(expected) is not int:
            continue
        for match in re.finditer(pattern, plain, re.IGNORECASE):
            raw = match.group("n").casefold()
            observed = int(raw) if raw.isdigit() else _NUMBERS[raw]
            prefix = plain[max(0, match.start() - 30):match.start()]
            if re.search(r"\b(?:not|only|at\s+least|up\s+to|among|of\s+the|example|another|remaining)\s*$", prefix, re.IGNORECASE):
                continue
            if observed != expected:
                issues.append(f"computed {name}: expected {expected}, not {observed}; shortlist and returned TopN are different scopes")
    return tuple(dict.fromkeys(issues))


def source_lead_label_corrections(text: str, context) -> list[dict[str, Any]]:
    """Correct an explicit uploaded-lead label, independently of that SNP's P.

    Lead status is calculated only from complete uploaded P observations. Tied
    minima are all eligible; LD-added or external-study variants never enter
    the comparison. Never replace a named non-lead SNP with a different SNP.
    """
    from .staged_reasoning import _measurement_trait_key

    facts = computed_report_facts(context.source_observations, context.result)
    leads = {_variant(value) for value in facts['source_lead_variant_ids']}
    if not leads:
        return []
    trait = _measurement_trait_key(context.source_trait)
    if any(row.get(key) and _measurement_trait_key(str(row[key])) != trait
           for row in context.source_observations
           for key in ('trait', 'phenotype', 'source_trait', 'source_phenotype')):
        return []
    aliases = facts.get('source_variant_aliases', {})
    selected = {_variant(aliases.get(row['variant_id'], row['variant_id']))
                for row in context.source_observations if isinstance(row.get('variant_id'), str)}
    pattern = re.compile(r'\b(?:uploaded|loaded|source(?:[- ]selected)?|user[- ]selected|selected|input|current)\s+'
        r'(?P<label>(?:(?:only|unique)\s+)?lead)\s+(?:variant|SNP)\s+(?:is\s+)?[`*]*'
        r'(?P<variant>' + _VARIANT.pattern + r')', re.I)
    edits = []
    for match in pattern.finditer(text):
        sentence = re.split(r'(?<=[.!?])\s+|\n', text[:match.start()])[-1]
        source_labels = re.findall(r'\b(?:loaded|uploaded|source|current)\s+([^,.;{}]{1,100}?)\s+(?:GWAS|dataset)\b',
                                   sentence, re.I)
        source_labels += re.findall(r'\b(?:concerns|in|from|for)\s+(?:(?:a|an|the)\s+)?'
                                    r'([^,.;{}]{1,100}?)\s+(?:source|uploaded|loaded|current)\s+GWAS\b', sentence, re.I)
        if any(_measurement_trait_key(label) not in {'source', 'loaded', 'uploaded', 'current', 'the', trait}
               for label in source_labels):
            continue
        prefix = re.split(r'[;.!?\n]', text[:match.start()])[-1]
        if re.search(r'\b(?:not|no|if|whether|might|may|external|published|reported|proxy)\b', prefix, re.I):
            continue
        identifier = _variant(aliases.get(match['variant'], match['variant']))
        if identifier not in selected:
            continue
        replacement = ('selected' if identifier not in leads else
                       'co-lead' if len(leads) > 1 and match['label'].casefold() != 'lead' else None)
        if replacement is None:
            continue
        start, end = match.span('label')
        edits.append({'start': start, 'end': end, 'replacement': replacement,
            'metric': 'source_lead_variant', 'source_id': match['variant'],
            'source_lead_variant_ids': facts['source_lead_variant_ids'],
            'source_minimum_p': facts['source_minimum_p'],
            'reason': 'lead_status_bound_to_minimum_uploaded_p'})
    return edits


def ranking_provenance(result: Mapping[str, Any], *, fallback: bool = False) -> dict[str, str]:
    existing = result.get("ranking_provenance")
    if isinstance(existing, Mapping) and existing.get("method") in {"model", "rules", "mixed", "not_performed"}:
        return {key: str(existing[key]) for key in ("method", "model_id") if key in existing}
    if not result.get("ranked_genes"):
        return {"method": "not_performed", "model_id": ""}
    gate = result.get("ranking_review", result.get("gate2_review"))
    gate = gate if isinstance(gate, Mapping) else {}
    method = str(gate.get("method", ""))
    identity = result.get("model_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    model = str(identity.get("model_id") or result.get("model_id") or "")
    if "partial" in method or "retained" in method or ((fallback or "fallback" in method) and gate.get("retained_valid_rankings", 0)):
        kind = "mixed"
    elif fallback or "deterministic" in method or model.startswith("evidence-count-") or identity.get("provider") == "deterministic":
        kind = "rules"
    else:
        kind = "model"
    return {"method": kind, "model_id": model}


def retrieval_incomplete_for_report(result: Mapping[str, Any]) -> bool:
    """Keep optional proxy gaps out of the primary retrieval status.

    Only a validated review policy can exclude failed proxy-only modules.
    Source and unattributed retrieval failures remain incomplete.
    """
    coverage = result.get("retrieval_coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    policy = result.get("ld_review_policy")
    policy = policy if isinstance(policy, Mapping) else {}
    optional = (policy.get("proxy_expansion") == "optional"
                and policy.get("selected_source_identities_validated") is True)
    incomplete = coverage.get("complete") is not True
    modules = coverage.get("module_statuses")
    excluded = policy.get("optional_proxy_locus_modules")
    reviewed = result.get("retrieval_review_policy")
    reviewed = reviewed if isinstance(reviewed, Mapping) else {}
    bounded = reviewed.get("bounded_literature_modules")
    bounded = {item for item in bounded if isinstance(item, str)} if isinstance(bounded, (list, tuple)) else set()
    if isinstance(modules, Mapping) and modules and (optional or bounded):
        excluded = ({item for item in excluded if isinstance(item, str)}
                    if optional and isinstance(excluded, (list, tuple)) else set())
        remaining = {key: value for key, value in modules.items() if key not in excluded
                     and not (key in bounded and value == "truncated")}
        incomplete = not remaining or any(
            not isinstance(value, str) or value not in {"success", "no_data", "not_applicable"}
            for value in remaining.values())
    plan = result.get("ld_expansion")
    return bool(incomplete or (not optional and isinstance(plan, Mapping) and plan.get("complete") is False))


def analysis_status(result: Mapping[str, Any], *, report_source: str, report_quality: str,
                    explanation_model: str = "", retrieval_incomplete: bool = False,
                    ranking_requested: bool = True) -> dict[str, str]:
    unreviewed = result.get("external_evidence_processed") is False
    provenance = ranking_provenance(result)
    review = result.get("safety_review_execution")
    review = review if isinstance(review, Mapping) else {}
    review_status = review.get("model_review_status")
    output_review = result.get("gate2_review")
    output_review = output_review if isinstance(output_review, Mapping) else {}
    fact_check = "not_passed" if report_quality == "STOPPED" else "not_assessed"
    if report_quality != "STOPPED" and output_review.get("target") == "final_report":
        if output_review.get("status") == "NOT_RUN":
            fact_check = "not_assessed"
        elif output_review.get("fully_verified") is False:
            fact_check = "partially_verified"
        elif output_review.get("fully_verified") is True and output_review.get("status") in {"PASS", "PASS_WITH_WARNINGS"}:
            fact_check = "passed_for_displayed_content"
    ranking_review = _ranking_review_status(result) if ranking_requested else "not_requested"
    attempts = (result.get("decision") or {}).get("generation_attempts", ())
    sampling_note = ""
    sampled = result.get("sampling_parameters")
    if not isinstance(sampled, Mapping) and attempts and isinstance(attempts[0], Mapping):
        sampled = (attempts[0].get("token_budget") or {}).get("sampling_parameters")
    if isinstance(sampled, Mapping):
        sampling_note = ("Temperature/seed not sent: unsupported in the active reasoning mode."
            if sampled.get("omission_reason") else
            f"Temperature: {sampled.get('temperature')}; seed: {sampled.get('seed')}. These do not guarantee deterministic output.")
    return {
        "sampling_note": sampling_note,
        "execution": "ended",
        "retrieval": "not_reviewed" if unreviewed else "partial" if retrieval_incomplete else "complete_in_scope",
        "ranking": provenance["method"] if ranking_requested else "not_requested",
        "ranking_model": provenance.get("model_id", ""),
        "ranking_review_status": ranking_review,
        "explanation": "model" if report_source in {"model", "model_repaired"} else "mixed" if report_source in {"partial_recovery", "deterministic_repair"} else "evidence_summary",
        "explanation_model": explanation_model if report_source in {"model", "model_repaired", "partial_recovery", "deterministic_repair"} else "",
        "fact_check": fact_check,
        "scope": "uploaded_gwas_only" if unreviewed else "retrieved_evidence",
        **({"safety_model_review": review_status,
            "safety_method": "rules_with_model_explanation" if review_status == "completed" else "rules"}
           if review_status in {"completed", "unavailable", "not_requested"} else {}),
    }


def _ranking_review_status(result: Mapping[str, Any]) -> str:
    """Keep the native ranking review separate from the later report review."""
    if not result.get("ranked_genes"):
        return "not_available"
    decision = result.get("decision")
    decision = decision if isinstance(decision, Mapping) else {}
    for review in (result.get("ranking_review"), result.get("gate2_review"), decision.get("gate2_review")):
        if not isinstance(review, Mapping) or not review or review.get("target") == "final_report":
            continue
        if (review.get("status") in {"needs_review", "partially_verified"}
                or review.get("fully_verified") is False or review.get("validation_passed") is False):
            return "needs_review"
        if (review.get("fully_verified") is True and review.get("validation_passed") is True
                and review.get("status") in {"verified", "PASS"}):
            return "verified"
    return "not_assessed"


def status_markdown(status: Mapping[str, Any], facts: Mapping[str, Any]) -> str:
    from .report_presentation import llm_presentation_text

    retrieval = {"not_reviewed": "Retained; external review incomplete", "partial": "Partial coverage", "complete_in_scope": "Complete within requested scope"}.get(status.get("retrieval"), "Unknown")
    ranking = {"model": "Model ranking", "rules": "Evidence rules (fallback)", "mixed": "Validated model items + evidence rules", "not_performed": "Not performed"}.get(status.get("ranking"), "Unknown")
    ranking = llm_presentation_text(ranking)
    explanation = status.get("explanation_model") or "Verified evidence summary"
    # Only safe configuration model IDs enter this line, not upstream error prose.
    explanation = re.sub(r"[^A-Za-z0-9_.:/ -]", "", str(explanation))[:120]
    labels = [("source_variant_count", "selected SNPs"), ("candidate_pool_count", "candidate genes"),
              ("shortlist_count", "shortlisted"), ("ranked_count", "ranked")]
    if (type(facts.get("displayed_ranked_count")) is int
            and type(facts.get("ranked_count")) is int
            and 0 <= facts["displayed_ranked_count"] < facts["ranked_count"]):
        labels.append(("displayed_ranked_count", "displayed"))
    counts = " · ".join(f"{facts[key]} {'selected SNP' if key == 'source_variant_count' and facts[key] == 1 else label}" for key, label in labels if type(facts.get(key)) is int
        and (status.get("ranking") != "not_requested" or key == "source_variant_count"))
    check = {"not_passed": "Not passed", "not_assessed": "Not assessed",
             "passed_for_displayed_content": "Passed for displayed content",
             "partially_verified": "Checked; some values remain unverified"}.get(
                  status.get("fact_check"), "Not assessed")
    ranking_label = f" · Ranking: {ranking}" if status.get("ranking") != "not_requested" else ""
    ranking_review = {"needs_review": "Checked; some claims remain unverified; original order retained",
                      "verified": "Passed", "not_available": "Not available", "not_assessed": "Not assessed"}.get(
                          status.get("ranking_review_status"), "Not assessed")
    review_label = f"**Ranking review:** {ranking_review} · " if status.get("ranking") != "not_requested" else ""
    model_review_note = " · **Model-assisted review:** Unavailable" if status.get("safety_model_review") == "unavailable" else ""
    model_review_note = llm_presentation_text(model_review_note)
    return (f"**Analysis status:** Request ended · Retrieval: {retrieval}{ranking_label}\n\n"
            f"{review_label}**Report fact check:** {check}\n\n"
            f"**Explanation source:** {explanation}{model_review_note}\n\n"
            + (f"**Analysis scope:** {counts}\n\n" if counts else ""))
