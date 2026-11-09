"""Web ranking availability; retained evidence never substitutes for model ranks."""
from collections.abc import Mapping
from typing import Any

RANKING_UNAVAILABLE_SUMMARY = "Model ranking incomplete · evidence report available"
RANKING_UNAVAILABLE_DETAIL = (
    "Model ranking did not complete. Retrieved evidence remains available; "
    "no rule-based ranking was substituted."
)
RANKING_NOT_CONFIGURED_DETAIL = (
    "Model ranking was not started because no verified ranking model is configured. "
    "Retrieved evidence remains available."
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def model_ranking_unavailable(result: Mapping[str, Any]) -> bool:
    if result.get("ranked_genes"):
        return False
    decision = _mapping(result.get("decision"))
    return bool(
        result.get("ranking_unavailable") is True
        or decision.get("status") == "ranking_unavailable"
        or any(review.get("method") == "model_unavailable" for review in (
            _mapping(result.get("ranking_review")), _mapping(result.get("gate2_review")),
            _mapping(decision.get("gate2_review"))))
    )


def retained_model_review(result: Mapping[str, Any]) -> bool:
    """A bound Gate2 review retains parsed model rows with local uncertainties."""
    if has_rule_based_ranking(result):
        return False
    decision = _mapping(result.get("decision"))
    return any(
        review.get("ranking_available") is True
        and review.get("fully_verified") is False
        and review.get("status") in {"needs_review", "partially_verified"}
        and review.get("method") in {"model_needs_review", "model_fact_corrected", "model"}
        for review in (_mapping(result.get("gate2_review")),
                       _mapping(result.get("ranking_review")),
                       _mapping(decision.get("gate2_review")))
    )


def has_rule_based_ranking(result: Mapping[str, Any]) -> bool:
    decision = _mapping(result.get("decision"))
    if not result.get("ranked_genes") and not decision.get("ranked_genes"):
        return False
    identities = (result, decision, _mapping(result.get("model_identity")),
                  _mapping(decision.get("model_identity")))
    provenance = _mapping(result.get("ranking_provenance"))
    status = _mapping(result.get("analysis_status"))
    model_ids = (identity.get(key) for identity in identities
                 for key in ("model_id", "requested_model_id", "observed_model_id"))
    if (provenance.get("method") in ("rules", "mixed")
            or status.get("ranking") in ("rules", "mixed")
            or any(identity.get("provider") == "deterministic" for identity in identities)
            or any(isinstance(value, str) and value.startswith(("evidence-count-", "deterministic"))
                   for value in model_ids)):
        return True
    reviews = [_mapping(result.get(key)) for key in ("ranking_review", "gate2_review")]
    reviews.append(_mapping(decision.get("gate2_review")))
    return any(
        str(review.get("method", "")).startswith(("deterministic", "partial_model", "partial_validated"))
        or review.get("deterministic_fallback") is True
        for review in reviews
    ) or any(value.get("deterministic_fallback") in (True, 1) for value in (
        result, decision, _mapping(result.get("usage")), _mapping(decision.get("usage"))))


def live_model_only_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Reject legacy rule ranks in a new live result without rewriting its artifact."""
    if result.get("data_mode") != "live" or not has_rule_based_ranking(result):
        return dict(result)
    decision = _mapping(result.get("decision"))
    facts = _mapping(result.get("computed_facts"))
    return {
        **result, "status": "INCOMPLETE", "ranked_genes": [], "ranking_unavailable": True,
        "summary": RANKING_UNAVAILABLE_DETAIL,
        "decision": {**decision, "status": "ranking_unavailable", "ranked_genes": [],
                     "ranking_attempted": decision.get("ranking_attempted") is True,
                     "reason_codes": ["automatic_rule_ranking_rejected"]},
        "ranking_review": {"method": "model_unavailable", "validation_passed": False,
                           "reason": "automatic_rule_ranking_rejected"},
        "ranking_provenance": {"method": "not_performed", "model_id": ""},
        "computed_facts": {**facts, "ranked_count": 0,
            **{key: 0 if key == "displayed_ranked_count" else []
               for key in ("displayed_ranked_count", "ranked_gene_ids", "displayed_ranked_gene_ids") if key in facts}},
        "analysis_status": {**_mapping(result.get("analysis_status")),
                            "ranking": "not_performed", "ranking_model": ""},
        "limitations": [*result.get("limitations", []), RANKING_UNAVAILABLE_DETAIL],
    }
