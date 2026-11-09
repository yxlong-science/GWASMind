"""Preserve observed module completion without inferring it from record counts."""


def module_retrieval_statuses(calls, plan=None):
    planned = {module for step in getattr(plan, "steps", ()) for module in step.modules}
    observed = {}
    mapping = {"completed": "success", "no_data": "no_data", "failed": "failed",
               "truncated": "partial", "blocked": "blocked", "unavailable": "unavailable"}
    for call in calls:
        coverage = call.coverage.to_dict()
        planned.update(coverage.get("requested", ()))
        for key, status in mapping.items():
            for module in coverage.get(key, ()):
                observed.setdefault(module, set()).add(status)
    result = {}
    for module in sorted(planned | observed.keys()):
        states = observed.get(module, set())
        gaps = states.intersection({"partial", "failed", "blocked", "unavailable"})
        if "partial" in states or gaps and states.intersection({"success", "no_data"}):
            result[module] = "partial"
        elif gaps:
            result[module] = next(s for s in ("failed", "blocked", "unavailable") if s in states)
        elif "success" in states:
            result[module] = "success"
        elif "no_data" in states:
            result[module] = "no_data"
        else:
            result[module] = "not_run"
    return result
