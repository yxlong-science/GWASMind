"""Validate model decisions without choosing, removing or ranking any candidate."""
from collections import Counter

DECISION_STATUSES = frozenset({"selected", "not_selected", "unresolved"})


def selection_contract_errors(output, candidates, *, required=False):
    decisions = output.candidate_decisions
    if not decisions and not required:
        return ()
    errors = []
    expected = set(candidates)
    counts = Counter(row.gene_id for row in decisions)
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        errors.append("candidate_decisions must account for every input candidate exactly once")
    for row in decisions:
        if row.status not in DECISION_STATUSES:
            errors.append(f"candidate decision for {row.gene_id} has an invalid status")
        if not isinstance(row.reason, str) or not row.reason.strip():
            errors.append(f"candidate decision for {row.gene_id} requires a scientific reason")
    selected = {row.gene_id for row in decisions if row.status == "selected"}
    ranked = [row.gene_id for row in output.ranked_genes]
    if output.model_full_pool_ranking_required:
        if set(ranked) != expected or len(set(ranked)) != len(ranked):
            errors.append("ranked_genes must contain every input candidate exactly once; shortlist labels do not limit ranking")
    elif set(ranked) != selected or len(set(ranked)) != len(ranked):
        errors.append("ranked_genes must contain exactly the model-selected candidates, each once")
    return tuple(dict.fromkeys(errors))


def assert_selection_unchanged(original, reviewed):
    before = tuple((row.gene_id, row.status) for row in original.candidate_decisions)
    after = tuple((row.gene_id, row.status) for row in reviewed.candidate_decisions)
    if before != after:
        raise ValueError("Gate2 fact review must preserve every original model candidate decision")
