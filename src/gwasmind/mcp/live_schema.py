from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from ..providers.live import OfficialOpenTargetsMCPProvider


SCHEMA_CATEGORIES = (
    "platform-metadata",
    "entity-search",
    "genetic-associations",
    "variant-annotation",
    "functional-genomics",
    "disease-associations",
    "target-safety",
    "target-tractability",
    "molecular-interactions",
    "experimental-models",
    "drug-indications",
    "drug-mechanisms",
    "drug-safety",
    "pharmacogenomics",
)


@dataclass(frozen=True, slots=True)
class LiveSchemaSnapshot:
    api_version: str
    data_version: str
    schema_sha256: str
    schema_text: str
    module_support: Mapping[str, bool]
    categories: tuple[str, ...] = SCHEMA_CATEGORIES

    def to_dict(self, *, include_schema: bool = False) -> dict[str, Any]:
        value = {
            "api_version": self.api_version,
            "data_version": self.data_version,
            "schema_sha256": self.schema_sha256,
            "categories": list(self.categories),
            "module_support": dict(self.module_support),
        }
        if include_schema:
            value["schema_text"] = self.schema_text
        return value


class OfficialSchemaNegotiator:
    """Discover the live Open Targets schema before enabling semantic modules."""

    def __init__(self, provider: OfficialOpenTargetsMCPProvider):
        self.provider = provider

    async def inspect(self) -> LiveSchemaSnapshot:
        scope = getattr(self.provider.transport, "session_scope", None)
        if callable(scope):
            async with scope():
                return await self._inspect_in_session()
        return await self._inspect_in_session()

    async def _inspect_in_session(self) -> LiveSchemaSnapshot:
        schema = await self.provider._call_tool(
            "get_open_targets_graphql_schema",
            {"categories": list(SCHEMA_CATEGORIES)},
        )
        schema_text = _collect_schema_text(schema)
        if not schema_text.strip():
            raise ValueError("official Open Targets endpoint returned an empty GraphQL schema")
        meta_response = await self.provider._call_tool(
            "query_open_targets_graphql",
            {
                "query_string": (
                    "query GWASMindMeta { meta { "
                    "apiVersion { x y z suffix } "
                    "dataVersion { year month iteration } "
                    "} }"
                ),
                "variables": {},
            },
        )
        meta = _extract_meta(meta_response)
        support = {
            "locus.variant": _contains(schema_text, "variant", "Variant"),
            "locus.variant_consequence": _contains(
                schema_text,
                "variant",
                "Variant",
                "transcriptConsequences",
                "TranscriptConsequence",
                "target",
                "approvedSymbol",
                "consequenceScore",
                "distanceFromFootprint",
                "distanceFromTss",
                "siftPrediction",
                "polyphenPrediction",
            ),
            "locus.credible_set": _contains(
                schema_text,
                "credibleSet",
                "CredibleSet",
                "Pagination",
                "locus",
                "Loci",
                "Locus",
                "variant",
                "Variant",
                "posteriorProbability",
            ),
            "locus.fine_mapping": _contains(
                schema_text,
                "credibleSet",
                "CredibleSet",
                "Pagination",
                "locus",
                "Locus",
                "posteriorProbability",
                "logBF",
                "finemappingMethod",
                "confidence",
            ),
            "locus.l2g": _contains(schema_text, "l2GPredictions"),
            "locus.colocalisation": _contains(schema_text, "colocalisation"),
            "locus.enhancer_to_gene": _contains(schema_text, "enhancerToGenes"),
            "target.core": _contains(schema_text, "target", "Target"),
            "target.go": _contains(schema_text, "geneOntology"),
            "target.expression": _contains(schema_text, "baselineExpression"),
            "target.literature": _contains(schema_text, "literatureOcurrences"),
            "target.safety": _contains(schema_text, "safetyLiabilities"),
            "target.tractability": _contains(schema_text, "tractability"),
            "target.interaction": _contains(schema_text, "interactions"),
            "target.mouse_phenotype": _contains(schema_text, "mousePhenotypes"),
            "disease.core": _contains(schema_text, "disease", "Disease"),
            "association.datasource_direct": _contains(
                schema_text, "associatedDiseases", "datasourceScores"
            ),
            "association.raw_evidence": _contains(schema_text, "evidences"),
            "drug.core": _contains(
                schema_text,
                "drug",
                "Drug",
                "drugType",
                "maximumClinicalStage",
            ),
            "drug.mechanism": _contains(
                schema_text,
                "mechanismsOfAction",
                "MechanismOfActionRow",
                "mechanismOfAction",
                "actionType",
            ),
            "drug.indication": _contains(
                schema_text,
                "indications",
                "ClinicalIndicationFromDrug",
                "maxClinicalStage",
                "clinicalReports",
            ),
            "drug.warning": _contains(
                schema_text, "drugWarnings", "DrugWarning", "warningType"
            ),
            "drug.adverse_event": _contains(
                schema_text,
                "adverseEvents",
                "AdverseEvents",
                "criticalValue",
                "logLR",
            ),
            "drug.clinical_report": _contains(schema_text, "clinicalReport"),
            "drug.clinical_target": _contains(
                schema_text, "drugAndClinicalCandidates"
            ),
            "drug.legacy_known_drug": _contains(schema_text, "KnownDrug"),
            "drug.pharmacogenomics": _contains(
                schema_text,
                "pharmacogenomics",
                "Pharmacogenomics",
                "evidenceLevel",
                "variantAnnotation",
            ),
        }
        return LiveSchemaSnapshot(
            api_version=_format_api_version(meta["apiVersion"]),
            data_version=_format_data_version(meta["dataVersion"]),
            schema_sha256=hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
            schema_text=schema_text,
            module_support=support,
        )


def _collect_schema_text(value: Any) -> str:
    values: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return "\n".join(values)


def _extract_meta(value: Mapping[str, Any]) -> Mapping[str, Any]:
    current: Any = value
    if isinstance(current, Mapping) and current.get("status") == "success":
        current = current.get("data")
    if isinstance(current, Mapping) and "data" in current:
        current = current["data"]
    if isinstance(current, Mapping) and "meta" in current:
        current = current["meta"]
    if not isinstance(current, Mapping):
        raise ValueError("official GraphQL metadata response is malformed")
    missing = {"apiVersion", "dataVersion"}.difference(current)
    if missing:
        raise ValueError(
            "official GraphQL metadata omitted: " + ", ".join(sorted(missing))
        )
    return current


def _contains(schema_text: str, *tokens: str) -> bool:
    return all(token in schema_text for token in tokens)


def _format_api_version(value: Any) -> str:
    if not isinstance(value, Mapping):
        return str(value)
    missing = {"x", "y", "z"}.difference(value)
    if missing:
        raise ValueError(
            "official apiVersion omitted: " + ", ".join(sorted(missing))
        )
    version = ".".join(str(int(value[key])) for key in ("x", "y", "z"))
    suffix = str(value.get("suffix") or "").strip()
    return f"{version}-{suffix}" if suffix else version


def _format_data_version(value: Any) -> str:
    if not isinstance(value, Mapping):
        return str(value)
    missing = {"year", "month", "iteration"}.difference(value)
    if missing:
        raise ValueError(
            "official dataVersion omitted: " + ", ".join(sorted(missing))
        )
    version = f"{int(value['year']):02d}.{int(value['month']):02d}"
    iteration = int(value["iteration"] or 0)
    return f"{version}.{iteration}" if iteration else version
