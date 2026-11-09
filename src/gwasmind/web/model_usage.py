"""Numeric-only, idempotent per-call progress for the interpretation stage."""
from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy

_TASKS = {"candidate_ranking", "report_generation", "report_repair"}
_STATES = {"running", "checking", "completed", "failed"}
_REVIEW_STATES = {"pending", "passed", "corrected", "needs_review", "not_assessed"}
_COUNTERS = {"output_tokens", "estimated_output_tokens", "reasoning_tokens", "content_tokens",
             "input_tokens", "output_budget_tokens"}


def _count(value):
    return value if type(value) is int and 0 <= value <= 10_000_000 else None


def public_model_usage(value):
    """Only fixed task labels and bounded provider counters reach a browser."""
    if not isinstance(value, Mapping):
        return {}
    identifier = value.get("call_id")
    if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", identifier)
            or value.get("task") not in _TASKS or value.get("status") not in _STATES
            or type(value.get("attempt")) is not int or not 1 <= value["attempt"] <= 24):
        return {}
    result = {key: value[key] for key in ("call_id", "task", "attempt", "status")}
    if isinstance(value.get("review_status"), str) and value["review_status"] in _REVIEW_STATES:
        result["review_status"] = value["review_status"]
    result.update({key: value[key] for key in _COUNTERS if _count(value.get(key)) is not None})
    if "output_tokens" in result:
        result.pop("estimated_output_tokens", None)
    if "reasoning_tokens" in result and "output_tokens" in result and result["reasoning_tokens"] > result["output_tokens"]:
        result.pop("reasoning_tokens")
        result.pop("content_tokens", None)
    if ("content_tokens" in result and "output_tokens" in result
            and (result["content_tokens"] > result["output_tokens"]
                 or "reasoning_tokens" in result
                 and result["content_tokens"] + result["reasoning_tokens"] != result["output_tokens"])):
        result.pop("content_tokens")
    if isinstance(value.get("calls"), (list, tuple)):
        result["calls"] = [row for child in value["calls"][:8]
                           if isinstance(child, Mapping)
                           and (row := public_model_usage({key: val for key, val in child.items() if key != "calls"}))]
    return result


class ModelCallUsage:
    """Repeated generated/validated notifications replace, never add, a call."""
    def __init__(self):
        self.calls = {}
        self.current = None

    def start(self, call_id, task, attempt, output_budget_tokens=None):
        row = public_model_usage({"call_id": call_id, "task": task, "attempt": attempt,
                                  "status": "running", "output_budget_tokens": output_budget_tokens})
        if not row:
            return
        self.calls.setdefault(call_id, row)
        self.current = call_id

    def stream(self, metrics, *, cumulative=False):
        if self.current is None:
            return
        row = self.calls[self.current]
        prior = sum(item.get("output_tokens", item.get("estimated_output_tokens", 0))
                    for identifier, item in self.calls.items() if identifier != self.current) if cumulative else 0
        exact = _count(metrics.get("output_tokens"))
        estimate = _count(metrics.get("estimated_output_tokens"))
        if exact is not None:
            row["output_tokens"] = max(0, exact - prior)
            row.pop("estimated_output_tokens", None)
        elif estimate is not None and "output_tokens" not in row:
            row["estimated_output_tokens"] = max(0, estimate - prior)

    def finish(self, usage, *, status="completed"):
        if self.current is None:
            return
        row = self.calls[self.current]
        row["status"] = status if status in _STATES else "completed"
        if row["status"] == "failed":
            row["review_status"] = "not_assessed"
        usage = usage if isinstance(usage, Mapping) else {}
        count = _count(usage.get("completion_tokens"))
        if count is not None:
            row["output_tokens"] = count
            row.pop("estimated_output_tokens", None)
        prompt = _count(usage.get("prompt_tokens"))
        if prompt is not None:
            row["input_tokens"] = prompt
        details = usage.get("completion_tokens_details")
        reasoning = _count(details.get("reasoning_tokens")) if isinstance(details, Mapping) else None
        if reasoning is None:
            reasoning = _count(usage.get("reasoning_tokens"))
        if count is not None and reasoning is not None and reasoning <= count:
            row.update(reasoning_tokens=reasoning, content_tokens=count - reasoning)

    def ranking_event(self, details):
        attempt = details.get("attempt")
        if type(attempt) is not int or not 1 <= attempt <= 24:
            return
        self.start(f"ranking-{attempt}", "candidate_ranking", attempt, details.get("requested_max_output_tokens"))
        if details.get("status") != "started":
            # Transport completion and local fact review are separate facts.
            # An output-validation notice must never undo a completed request.
            generated = details.get("status") == "generated"
            incomplete = details.get("finish_reason") in {"length", "content_filter", "aborted"}
            status = "completed" if generated and not incomplete else "failed"
            self.finish(details.get("usage"), status=status)
            self.calls[self.current]["review_status"] = (
                "not_assessed" if status == "failed" else
                "passed" if details.get("validation") == "passed" else
                "needs_review" if details.get("validation") == "failed" else "pending")

    def ranking_review(self, details):
        """Apply the final Gate2 outcome without changing generation success."""
        if self.current is None or self.calls[self.current]["task"] != "candidate_ranking":
            return
        review = details.get("gate2_review")
        if not isinstance(review, Mapping):
            return
        row = self.calls[self.current]
        if row["status"] != "completed":
            return
        row["review_status"] = (
            "needs_review" if review.get("fully_verified") is False else
            "corrected" if review.get("method") == "model_fact_corrected" else
            "passed" if review.get("validation_passed") is True else "not_assessed")

    def public(self):
        if self.current is None:
            return {}
        value = deepcopy(self.calls[self.current])
        value["calls"] = list(deepcopy(self.calls).values())[-8:]
        return public_model_usage(value)

    def report_review(self, review):
        if self.current is None or self.calls[self.current]["task"] not in {"report_generation", "report_repair"}:
            return
        row = self.calls[self.current]
        if row["status"] != "completed":
            return
        row["review_status"] = (
            "not_assessed" if review.get("status") == "NOT_RUN" else
            "needs_review" if review.get("fully_verified") is False else
            "corrected" if (review.get("machine_audit") or {}).get("corrections") else
            "passed" if review.get("generated_draft_validated") is True else "not_assessed")
