"""Typed numeric facts with stable, reproducible source and derivation identities.

This ledger contains observations and policies, never inferred gene rankings.
It is retained beside the exact LLM request rather than duplicated in its text.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from collections.abc import Mapping

from .audit import sha256_json

SCHEMA = "gwasmind_numeric_facts_v1"


def finite_decimal(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() else None


@dataclass(frozen=True)
class NumericFact:
    evidence_id: str
    gene_id: str
    variant_id: str
    trait: str
    metric: str
    value: Decimal
    scope: str = "measurement"
    provenance: tuple[tuple[str, str], ...] = ()

    def to_dict(self):
        value = {"metric": self.metric, "value": str(self.value), "scope": self.scope,
                 "owner": {key: item for key, item in (("gene_id", self.gene_id),
                     ("variant_id", self.variant_id), ("trait", self.trait)) if item},
                 "evidence_id": self.evidence_id, "provenance": dict(self.provenance)}
        return {"fact_id": "nf:" + sha256_json(value), **value}

    @property
    def fact_id(self):
        return self.to_dict()["fact_id"]


def source_p_facts(query):
    trait = query.source_trait or query.trait
    for row in query.variants:
        value = finite_decimal(row.p_value)
        if value is not None and 0 <= value <= 1:
            yield NumericFact("uploaded_source", "", row.variant_id.casefold(), trait, "p", value,
                "uploaded_gwas", (("genome_build", query.genome_build or "not_reported"),
                    ("source_field", "source_observations.p_value"),
                    ("observation_sha256", sha256_json({"variant_id": row.variant_id,
                        "trait": trait, "p_value": row.p_value, "genome_build": query.genome_build}))))


def ld_numeric_facts(query):
    from .ld_assertions import identity_resolver
    plan = query.ld_expansion or {}
    canonical = identity_resolver(plan)
    selected = {canonical(row.variant_id) for row in query.variants}
    pairs, grouped = {}, {}
    for row in plan.get("ld_links", ()):
        if not isinstance(row, Mapping):
            continue
        value = finite_decimal(row.get("r2"))
        source, proxy = canonical(row.get("selected_variant_id")), canonical(row.get("proxy_variant_id"))
        if value is None or not 0 <= value <= 1 or source not in selected or not proxy:
            continue
        population = str(row.get("population") or plan.get("population") or "not_reported")
        build = str(row.get("genome_build") or plan.get("genome_build") or "not_reported")
        fact = NumericFact("", "", proxy, "", "r2", value, "ld_pair", tuple(sorted({
            "selected_variant_id": source, "proxy_variant_id": proxy, "population": population,
            "genome_build": build, "reference_panel": str(plan.get("reference_panel") or "not_reported"),
            "reference_provider": str(plan.get("reference_provider") or "not_reported"),
            "provider_version": str(plan.get("provider_version") or "not_reported"),
            "source_field": "ld_expansion.ld_links.r2", "source_row_sha256": sha256_json(dict(row)),
        }.items())))
        pairs[fact.fact_id] = fact.to_dict()
        grouped.setdefault((proxy, population, build), {})[fact.fact_id] = fact
    values = list(pairs.values())
    for (proxy, population, build), contributors in sorted(grouped.items()):
        # Different values for the same physical pair are conflicting inputs;
        # do not silently turn the larger one into a valid maximum.
        by_pair = {}
        for fact in contributors.values():
            pair = dict(fact.provenance)["selected_variant_id"]
            by_pair.setdefault(pair, set()).add(fact.value)
        if any(len(numbers) > 1 for numbers in by_pair.values()):
            continue
        value = {"metric": "r2", "value": str(max(f.value for f in contributors.values())),
                 "scope": "maximum_recorded_selected_source_ld", "owner": {"variant_id": proxy},
                 "population": population, "genome_build": build,
                 "derivation": {"operation": "maximum", "contributor_fact_ids": sorted(contributors),
                     "scope": "recorded_edges_only", "retrieval_complete": plan.get("complete") is True}}
        values.append({"fact_id": "nf:" + sha256_json(value), **value})
    threshold = finite_decimal(plan.get("threshold"))
    if threshold is not None and 0 <= threshold <= 1:
        values.append(NumericFact("", "", "", "", "r2", threshold, "inclusion_threshold",
            (("source_field", "ld_expansion.threshold"), ("operator", ">="))).to_dict())
    return values


def build_numeric_ledger(query, records, *, measurements=None):
    from .ranking_claims import raw_measurements
    values = [fact.to_dict() for fact in source_p_facts(query)]
    for record in records:
        facts = measurements[id(record)] if measurements is not None and id(record) in measurements else raw_measurements(record)
        values.extend(fact.to_dict() for fact in facts)
        context = record.payload.get("context") or {}
        values.extend(policy_numeric_facts(context))
    values.extend(ld_numeric_facts(query))
    unique = {row["fact_id"]: row for row in values}
    result = {"schema": SCHEMA, "facts": [unique[key] for key in sorted(unique)]}
    return {**result, "ledger_sha256": sha256_json(result)}


def policy_numeric_facts(context):
    for key, metric, operator in (("h4_inclusion_threshold", "h4", ">="),
                                  ("lead_pvalue_max_exclusive", "p", "<")):
        threshold = finite_decimal(context.get(key))
        if threshold is not None and 0 <= threshold <= 1:
            yield NumericFact("", "", "", "", metric, threshold, "inclusion_threshold",
                (("source_field", "context." + key), ("operator", operator))).to_dict()


def validate_numeric_ledger(value):
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA or not isinstance(value.get("facts"), list):
        raise ValueError("Invalid numeric fact ledger")
    body = {"schema": value["schema"], "facts": value["facts"]}
    if value.get("ledger_sha256") != sha256_json(body):
        raise ValueError("Numeric ledger digest does not match")
    seen = set()
    for fact in value["facts"]:
        if not isinstance(fact, Mapping):
            raise ValueError("Numeric fact is not an object")
        content = {key: item for key, item in fact.items() if key != "fact_id"}
        if fact.get("fact_id") != "nf:" + sha256_json(content) or fact["fact_id"] in seen:
            raise ValueError("Numeric fact identity is invalid or duplicated")
        seen.add(fact["fact_id"])
        if finite_decimal(fact.get("value")) is None:
            raise ValueError("Numeric fact value is not finite")
        if not isinstance(fact.get('owner'), Mapping) or not isinstance(fact.get('provenance', {}), Mapping) or not isinstance(fact.get('derivation', {}), Mapping):
            raise ValueError("Numeric fact has invalid owner or source metadata")
        if fact.get('metric') in {'p', 'h4', 'h3', 'pip', 'clpp', 'r2'} and not 0 <= finite_decimal(fact['value']) <= 1:
            raise ValueError("Numeric probability lies outside its domain")
    by_id = {fact["fact_id"]: fact for fact in value["facts"]}
    for fact in value["facts"]:
        contributors = fact.get("derivation", {}).get("contributor_fact_ids", [])
        if not isinstance(contributors, list) or any(not isinstance(key, str) for key in contributors):
            raise ValueError("Numeric derivation has invalid contributors")
        if not set(contributors) <= seen:
            raise ValueError("Numeric derivation references an unknown source fact")
        if fact.get("scope") == "maximum_recorded_selected_source_ld":
            sources = [by_id[key] for key in contributors]
            if not sources or len(set(contributors)) != len(contributors) or fact.get("derivation", {}).get("operation") != "maximum":
                raise ValueError("LD maximum requires distinct source pairs")
            by_pair = {}
            for source in sources:
                metadata = source.get("provenance", {})
                if (source.get("scope") != "ld_pair" or source.get("metric") != "r2"
                        or source.get("owner") != fact.get("owner")
                        or any(metadata.get(key) != fact.get(key) for key in ("population", "genome_build"))):
                    raise ValueError("LD derivation mixes incompatible source pairs")
                by_pair.setdefault(metadata.get("selected_variant_id"), set()).add(finite_decimal(source["value"]))
            if any(len(numbers) != 1 for numbers in by_pair.values()) or finite_decimal(fact["value"]) != max(finite_decimal(row["value"]) for row in sources):
                raise ValueError("LD derivation does not reproduce its source pairs")
    return value


def report_numeric_ledger(context, registry):
    """Bind report values to immutable display facts and their raw evidence IDs.

    A display fact is a program-generated view, not an LLM assertion. Its exact
    snapshot remains the source of legacy reports without a raw numeric ledger.
    """
    import json
    from types import SimpleNamespace
    from .staged_reasoning import _final_ld_report_scope, _runtime_evidence_rows

    scope = _final_ld_report_scope(context)
    plan = dict(context.result.get("ld_expansion") or {})
    plan["ld_links"] = list(scope.links) if scope.valid else []
    plan["resolved_source_aliases"] = dict(scope.source_aliases)
    if not scope.valid or not scope.provided or scope.disabled:
        plan["threshold"] = None
    query = SimpleNamespace(ld_expansion=plan, source_trait=context.source_trait, trait=context.source_trait,
        genome_build=plan.get("genome_build"), variants=tuple(SimpleNamespace(
            variant_id=row["variant_id"], p_value=row.get("p_value"))
            for row in context.source_observations if row.get("variant_id")))
    values = ld_numeric_facts(query) + [fact.to_dict() for fact in source_p_facts(query)]
    for row in _runtime_evidence_rows(context.result):
        values.extend(policy_numeric_facts((row.get("payload") or {}).get("context") or {}))
    facts = {fact.fact_id: fact for fact in context.facts}
    for entry in registry:
        # The shared LD adapter binds pairs directly; a bare display scalar
        # must not become another independent measurement of LD.
        if entry["metric"] == "r2":
            continue
        source = facts[entry["fact_id"]]
        provenance = {"source_field": "report.facts.deterministic_detail", "source_sha256": entry["source_sha256"],
            "report_fact_id": source.fact_id, "evidence_ids": json.dumps(sorted(source.evidence_ids)),
            "variant_owners": json.dumps(entry["variants"]), "gene_owners": json.dumps(entry["genes"]),
            "source_span": json.dumps(entry.get("source_span")), "unit": entry.get("unit", "")}
        for key in ("qtl_measurement", "colocalization"):
            if entry.get(key):
                provenance[key] = json.dumps(entry[key], sort_keys=True)
        fact = NumericFact(source.fact_id, entry["genes"][0] if len(entry["genes"]) == 1 else "",
            entry["variants"][0] if len(entry["variants"]) == 1 else "", "", entry["metric"],
            finite_decimal(entry["value"]), "report_display_measurement", tuple(sorted(provenance.items())))
        entry["numeric_fact_ids"] = [fact.fact_id]
        values.append(fact.to_dict())
    unique = {row["fact_id"]: row for row in values}
    body = {"schema": SCHEMA, "facts": [unique[key] for key in sorted(unique)]}
    return validate_numeric_ledger({**body, "ledger_sha256": sha256_json(body)})
