from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.resources import files

CONTEXT_RULESET_VERSION = "1.0.0"


class ContextResolutionError(ValueError):
    """A phenotype cannot receive a validated nonempty biological search plan."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Biological context could not be resolved ({reason}).")


def evidence_context_values(record: Mapping) -> tuple[str, ...]:
    """Read observed biosample fields, never requested tissues or general prose."""
    keys = {"biosample", "biosampleid", "biosamplename", "tissue", "tissuelabel", "celltype"}
    excluded = {"request", "query", "arguments", "relevanttissues", "relevanttissuesrequested", "relevantbiosampleids"}
    values = []

    def collect(value, *, context=False, depth=0):
        if depth > 7:
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized = str(key).replace("_", "").casefold()
                if normalized in excluded:
                    continue
                observed = normalized in keys or context and normalized in {"id", "name", "label"}
                collect(nested, context=observed, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            for nested in value[:1000]:
                collect(nested, context=context, depth=depth + 1)
        elif context and isinstance(value, str) and value.strip():
            values.append(value.strip())

    collect(record)
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True, slots=True)
class BiologicalContext:
    """Auditable phenotype context supplied to candidate-gene prioritization.

    New plans retain bounded organ, tissue and cell concepts together with their
    frozen Open Targets biosample identity/hierarchy mappings and source provenance.
    Mapping a concept does not establish its causal relevance to the phenotype.
    Historical contexts without a mapping preserve their original labels and
    serialization contract for faithful replay.
    """

    phenotype: str
    tissues: tuple[str, ...] = ()
    cell_types: tuple[str, ...] = ()
    rule_id: str | None = None
    source: str = "unresolved"
    ruleset_version: str = CONTEXT_RULESET_VERSION
    organs: tuple[str, ...] = ()
    biosample_mapping: Mapping | None = None
    phenotype_scope: Mapping | None = None

    @property
    def relevant_terms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.organs, *self.tissues, *self.cell_types)))

    @property
    def is_defined(self) -> bool:
        return bool(self.relevant_terms)

    def to_dict(self) -> dict[str, object]:
        """Return the bounded public/audit representation used by web progress/results."""

        value = {
            "phenotype": self.phenotype,
            "tissues": list(self.tissues),
            "cell_types": list(self.cell_types),
            "relevant_tissues": list(self.relevant_terms),
            "rule_id": self.rule_id,
            "source": self.source,
            "ruleset_version": self.ruleset_version,
            "defined": self.is_defined,
        }
        if self.organs or self.biosample_mapping is not None:
            value["organs"] = list(self.organs)
        if self.biosample_mapping is not None:
            value["biosample_mapping"] = deepcopy(dict(self.biosample_mapping))
        if self.phenotype_scope is not None:
            value["phenotype_scope"] = deepcopy(dict(self.phenotype_scope))
        value["context_sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return value


def biological_context_from_dict(value: Mapping, *, phenotype: str) -> BiologicalContext:
    """Validate the fixed discovery context without inferring missing terms.

    In particular, an explicitly unresolved context must survive worker transport;
    rebuilding it from only ``relevant_tissues`` would change shortlist eligibility.
    """
    if not isinstance(value, Mapping):
        raise ValueError("biological_context must be an object")  # noqa: TRY004 - query validation API
    if str(value.get("phenotype", phenotype)).strip().casefold() != phenotype.strip().casefold():
        raise ValueError("biological_context phenotype must match the query trait")
    terms = {}
    for key in ("organs", "tissues", "cell_types"):
        rows = value.get(key, [])
        if not isinstance(rows, (list, tuple)) or len(rows) > 100 or any(
            not isinstance(item, str) or not item.strip() or len(item) > 120 for item in rows
        ):
            raise ValueError(f"biological_context.{key} must contain bounded non-empty strings")
        terms[key] = tuple(dict.fromkeys(item.strip() for item in rows))
    for key in ("source", "ruleset_version", "rule_id"):
        if value.get(key) is not None and (not isinstance(value[key], str) or len(value[key]) > 200):
            raise ValueError(f"biological_context.{key} must be a bounded string")
    mapping = value.get("biosample_mapping")
    if mapping is not None:
        from .biosample_catalog import map_context_concepts
        expected = map_context_concepts({**terms, "biosample_mapping": mapping})
        if not isinstance(mapping, Mapping) or dict(mapping) != expected:
            raise ValueError("biological_context biosample mapping does not match its frozen concepts")
    scope = value.get("phenotype_scope")
    if scope is not None:
        from .phenotype_scope import validate_phenotype_scope
        scope = validate_phenotype_scope(scope, phenotype=phenotype)
    result = BiologicalContext(phenotype=phenotype, **terms, biosample_mapping=mapping, phenotype_scope=scope,
        rule_id=value.get("rule_id"), source=value.get("source") or "caller supplied",
        ruleset_version=value.get("ruleset_version") or CONTEXT_RULESET_VERSION)
    if "defined" in value and (type(value["defined"]) is not bool or value["defined"] != result.is_defined):
        raise ValueError("biological_context.defined contradicts its terms")
    if "relevant_tissues" in value and value["relevant_tissues"] != list(result.relevant_terms):
        raise ValueError("biological_context.relevant_tissues contradicts its terms")
    if value.get("context_sha256") is not None and value["context_sha256"] != result.to_dict()["context_sha256"]:
        raise ValueError("biological_context context_sha256 mismatch")
    return result


def standardize_biological_context(context: BiologicalContext, *, declared_organs=()) -> BiologicalContext:
    """Map a new plan once; persisted legacy contexts keep their original representation."""
    from .biosample_catalog import biosample_catalog, map_context_concepts
    from .configuration import parameter
    if context.biosample_mapping is not None:
        return context
    catalog = biosample_catalog()
    axes = {"organs": list(declared_organs or context.organs), "tissues": [],
            "cell_types": list(context.cell_types)}
    # Legacy curated terms mix organs and tissues. Reclassify only with catalog evidence.
    for term in context.tissues:
        match = catalog.resolve(term)
        axis = (catalog.axis(match["matches"][0]["id"]) if match["status"] == "resolved" else None)
        axes[axis if axis in axes else "tissues"].append(term)
    limit = parameter("context.max_concepts_per_level")
    for axis, values in axes.items():
        axes[axis] = list(dict.fromkeys(values))
        if len(axes[axis]) > limit:
            raise ValueError(f"Biological context {axis} exceeds the configured concept limit")
    mapping = map_context_concepts(axes)
    return BiologicalContext(phenotype=context.phenotype, organs=tuple(axes["organs"]),
        tissues=tuple(axes["tissues"]), cell_types=tuple(axes["cell_types"]),
        rule_id=context.rule_id, source=context.source, ruleset_version="2.0.0",
        biosample_mapping=mapping, phenotype_scope=deepcopy(context.phenotype_scope))


def assess_biosample_context(context: Mapping, *, tissue: Mapping | None = None,
                            cell_type: Mapping | None = None, biosample: Mapping | None = None) -> dict:
    """Compare provider identities with the frozen context without lexical inference.

    Tissue and cell are independent observed axes. A cell measurement never
    supplies an unreported organ, and contradictory IDs and labels stay unknown.
    """
    from .molecular_context import freeze_molecular_context
    return _assess_frozen_biosamples(freeze_molecular_context(context),
        tissue=tissue, cell_type=cell_type, biosample=biosample)


def compile_biosample_context(context: Mapping):
    """Freeze once and cache identity comparisons within a single retrieval only."""
    from .molecular_context import freeze_molecular_context
    frozen = freeze_molecular_context(context)
    cache = {}

    def assess(*, tissue=None, cell_type=None, biosample=None):
        samples = {"tissue": tissue, "cell_type": cell_type, "biosample": biosample}
        key = tuple((axis, tuple(str((value if isinstance(value, Mapping) else {}).get(field) or "")
            for field in ("biosampleId", "biosampleName", "id", "name", "label")))
            for axis, value in samples.items())
        if key not in cache:
            cache[key] = _assess_frozen_biosamples(frozen, **samples)
        return deepcopy(cache[key])

    return assess


def _assess_frozen_biosamples(frozen, *, tissue=None, cell_type=None, biosample=None):
    from .molecular_context import molecular_context_match
    assessments = {}
    for axis, value in (("tissue", tissue), ("cell_type", cell_type), ("biosample", biosample)):
        if not isinstance(value, Mapping) or not value:
            continue
        raw = {key: value[key] for key in ("biosampleId", "biosampleName") if value.get(key)}
        for source, target in (("id", "biosampleId"), ("name", "biosampleName"), ("label", "biosampleName")):
            if target not in raw and value.get(source):
                raw[target] = value[source]
        assessments[axis] = molecular_context_match({"upstream": {"raw": raw}}, frozen)
    if not assessments:
        assessments["biosample"] = molecular_context_match({}, frozen)
    matches = [row for row in assessments.values() if row["matched"]]
    observed = {item["id"]: item for row in assessments.values() for item in row["observed_biosamples"]}
    bases = [item for row in matches for item in row["match_basis"]]
    resolved = all(row["mapping_status"] == "resolved" for row in assessments.values())
    return {"matched": bool(matches), "context_class": "matched" if matches else
            "other_reported_context" if resolved else "context_unreported",
            "status": "matched" if matches else "not_matched" if resolved else "biosample_unavailable",
            "mapping_status": "resolved" if resolved else "partial" if matches else "unresolved",
            "observed": sorted({item for row in assessments.values() for item in row.get("observed", ())}),
            "observed_biosamples": [observed[key] for key in sorted(observed)], "match_basis": bases,
            "matched_axes": sorted({item["selected_axis"] for item in bases}),
            "exact_tissue_match": any(item["relation"] == "exact_biosample_id"
                                      for item in assessments.get("tissue", {}).get("match_basis", ())),
            "exact_cell_match": any(item["relation"] == "exact_biosample_id"
                                    for item in assessments.get("cell_type", {}).get("match_basis", ())),
            "axis_assessments": assessments, "context_sha256": frozen["context_sha256"]}


@lru_cache(maxsize=1)
def curated_context_catalog() -> dict:
    """Return authored search plans; neither model responses nor phenotype evidence."""
    content = files("gwasmind").joinpath("data/phenotype_context_seeds.json").read_bytes()
    value = json.loads(content)
    value["sha256"] = hashlib.sha256(content).hexdigest()
    return value


def curated_biological_context(phenotype: str, *, reason: str = "model_not_requested") -> BiologicalContext | None:
    """Recover only recognized phenotypes with bounded, catalog-validated identities."""
    from .biosample_catalog import biosample_catalog

    catalog = biosample_catalog()
    seeds = curated_context_catalog()
    names = {" ".join(name.casefold().split()): name for name in seeds["phenotypes"]}
    names.update({" ".join(alias.casefold().split()): name for alias, name in seeds["aliases"].items()})
    name = names.get(" ".join(str(phenotype).casefold().split()))
    if name is None:
        legacy = define_biological_context(phenotype)
        if not legacy.is_defined:
            return None
        # The human search context must not resolve the ambiguous muscle synonym
        # to an invertebrate concept merely because both occur in the catalog.
        legacy = BiologicalContext(phenotype=phenotype,
            organs=legacy.organs,
            tissues=tuple("UBERON_0001134" if term == "skeletal muscle" else term for term in legacy.tissues),
            cell_types=legacy.cell_types, rule_id=legacy.rule_id,
            source=f"curated phenotype context recovery ({reason})", ruleset_version=legacy.ruleset_version)
        result = standardize_biological_context(legacy)
    else:
        profile = seeds["phenotypes"][name]
        axes = {"organs": [], "tissues": [], "cell_types": []}
        for identity in seeds["profiles"][profile]["ids"]:
            resolved = catalog.resolve(identity)
            axis = catalog.axis(identity)
            if resolved["status"] != "resolved" or axis not in axes:
                raise ContextResolutionError("curated_identity_unavailable")
            axes[axis].append(identity)
        result = standardize_biological_context(BiologicalContext(phenotype=phenotype,
            **{axis: tuple(values) for axis, values in axes.items()},
            rule_id=f"{seeds['version']}:{profile}",
            ruleset_version=f"{seeds['version']}:{seeds['sha256'][:16]}",
            source=f"curated phenotype context recovery ({reason})"))
        result = replace(result, ruleset_version=f"{seeds['version']}:{seeds['sha256'][:16]}")
    if not result.is_defined or result.biosample_mapping["status"] != "resolved":
        raise ContextResolutionError("curated_identity_unavailable")
    return result


@dataclass(frozen=True, slots=True)
class _ContextRule:
    rule_id: str
    canonical_phenotype: str
    pattern: re.Pattern[str]
    tissues: tuple[str, ...]
    cell_types: tuple[str, ...]


# These are intentionally narrow phenotype aliases and broad, canonical biosample
# labels.  They are a software rule set, not an inferred biological claim.  The rule ID
# and version are written into the user-facing answer and the resolved query is retained
# in the run manifest, making every applied term auditable.
_CONTEXT_RULES: tuple[_ContextRule, ...] = (
    _ContextRule(
        rule_id="type_1_diabetes",
        canonical_phenotype="type 1 diabetes",
        pattern=re.compile(
            r"(?:\btype\s*(?:1|i)\s+diabet(?:es|ic)\b|\bt1dm?\b|"
            r"(?<!non[- ])\binsulin[- ]dependent diabetes\b)",
            re.IGNORECASE,
        ),
        tissues=("pancreas", "pancreatic islet"),
        cell_types=("pancreatic beta cell", "T cell", "B cell"),
    ),
    _ContextRule(
        rule_id="type_2_diabetes",
        canonical_phenotype="type 2 diabetes",
        pattern=re.compile(
            r"(?:\btype\s*(?:2|ii)\s+diabet(?:es|ic)\b|\bt2dm?\b|"
            r"\bnon[- ]insulin[- ]dependent diabetes\b)",
            re.IGNORECASE,
        ),
        tissues=(
            "pancreas",
            "pancreatic islet",
            "liver",
            "adipose tissue",
            "skeletal muscle",
        ),
        cell_types=("pancreatic beta cell",),
    ),
    _ContextRule(
        rule_id="asthma",
        canonical_phenotype="asthma",
        pattern=re.compile(r"\basthma(?:tic)?\b", re.IGNORECASE),
        tissues=("lung", "bronchus", "airway epithelium"),
        cell_types=("airway epithelial cell", "eosinophil", "mast cell", "T cell"),
    ),
)


_CELL_TYPE_HINT = re.compile(
    r"(?:\bcells?\b|lymphocyte|monocyte|macrophage|eosinophil|neutrophil|"
    r"mast cell|T cell|B cell)",
    re.IGNORECASE,
)


def define_biological_context(
    phenotype: str,
    *,
    explicit_terms: tuple[str, ...] = (),
) -> BiologicalContext:
    """Return an explicit context or one conservative, auditable phenotype rule.

    User-/caller-supplied terms always win.  Unknown phenotypes intentionally return an
    empty context: the ranker then receives no tissue-concordance preference rather than
    a guessed tissue or cell type.
    """

    phenotype_text = str(phenotype or "").strip()
    cleaned_explicit = tuple(
        dict.fromkeys(
            item.strip()
            for item in (str(value) for value in explicit_terms)
            if item.strip()
        )
    )
    if cleaned_explicit:
        from .biosample_catalog import biosample_catalog
        catalog = biosample_catalog()
        cells = []
        for item in cleaned_explicit:
            resolved = catalog.resolve(item)
            axis = (catalog.axis(resolved["matches"][0]["id"])
                    if resolved["status"] == "resolved" else None)
            if axis == "cell_types" or axis is None and _CELL_TYPE_HINT.search(item):
                cells.append(item)
        cells = tuple(cells)
        tissues = tuple(item for item in cleaned_explicit if item not in cells)
        return BiologicalContext(
            phenotype=phenotype_text,
            tissues=tissues,
            cell_types=cells,
            rule_id="explicit_relevant_tissues",
            source="caller supplied",
        )

    matches = tuple(rule for rule in _CONTEXT_RULES if rule.pattern.search(phenotype_text))
    if len(matches) == 1:
        rule = matches[0]
        return BiologicalContext(
            phenotype=phenotype_text,
            tissues=rule.tissues,
            cell_types=rule.cell_types,
            rule_id=rule.rule_id,
            source="conservative curated phenotype rule",
        )

    return BiologicalContext(phenotype=phenotype_text)


def phenotype_from_question(question: str, *, fallback: str) -> str:
    """Resolve only an explicitly named phenotype covered by the conservative rules.

    This helper is intentionally not a general disease NER system.  It exists so the
    interactive chat path can honor an explicit phrase such as ``in T1D`` without
    replacing the loaded phenotype for an unknown or ambiguous question.
    """

    question_text = str(question or "")
    matches = tuple(
        rule.canonical_phenotype
        for rule in _CONTEXT_RULES
        if rule.pattern.search(question_text)
    )
    # Multiple explicit phenotype names make the intended ranking context
    # ambiguous.  Keep the caller's current phenotype instead of silently choosing
    # whichever rule happens to appear first in this module.
    if len(matches) == 1:
        return matches[0]
    return str(fallback or "").strip()


def context_term_matches(term: str, observed: str) -> bool:
    """Case-insensitive conservative text matching for displayed biosample labels."""

    expected = " ".join(str(term or "").casefold().replace("-", " ").split())
    actual = " ".join(str(observed or "").casefold().replace("-", " ").split())
    return bool(expected and actual and (
        re.search(r"(?<!\w)" + re.escape(expected) + r"s?(?!\w)", actual)
        or re.search(r"(?<!\w)" + re.escape(actual) + r"s?(?!\w)", expected)
    ))
