"""Source-bound variant labels, independent of statistical measurement ownership."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .source_normalization import validate_source_normalization

SCHEMA = "gwasmind_variant_identity_registry_v1"
_VARIANT = re.compile(r"^(?:chr)?([1-9]|1[0-9]|2[0-2]|X|Y|MT)[:_]([1-9][0-9]*)[:_]([ACGT]+)[:_]([ACGT]+)$", re.I)
_RS = re.compile(r"^rs[1-9][0-9]*$", re.I)
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _canonical(value: Any) -> str | None:
    match = _VARIANT.fullmatch(str(value or ""))
    if not match or match[3].upper() == match[4].upper():
        return None
    return "_".join(part.upper() for part in match.groups())


def _rsids(values: Any) -> list[str]:
    return sorted({item.lower() for item in values if isinstance(item, str) and _RS.fullmatch(item)},
                  key=lambda item: int(item[2:])) if isinstance(values, (list, tuple)) else []


def ot_identity_records(raw_responses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract explicit OT id/rsIds pairs with their original response provenance.

    Only successful V2G variant/target responses are accepted. A queried variant
    name, study lead label or nearby coordinate alone never supplies an rsID.
    """
    result = []
    for response in raw_responses:
        if not isinstance(response, Mapping) or response.get("operation") not in {
                "variants", "targets", "target_credibleSets_page", "display_variant_identities"}:
            continue
        data = response.get("data")
        if not isinstance(data, Mapping):
            continue
        response_hash = _digest(data)

        def visit(value: Any, path: str) -> None:
            if isinstance(value, Mapping):
                identifier = _canonical(value.get("id"))
                if identifier and "rsIds" in value:
                    result.append({"variant": {key: value[key] for key in (
                                       "id", "rsIds", "chromosome", "position", "referenceAllele", "alternateAllele", "genomeBuild") if key in value}, "genome_build": "GRCh38",
                                   "provider": "opentargets", "response_sha256": response_hash,
                                   "record_path": path})
                for key, child in value.items():
                    visit(child, path + "." + str(key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, path + f"[{index}]")

        visit(data, "data")
    return result


def build_variant_identity_registry(*, variant_records=(), source_normalizations=(),
                                    ld_expansion=None, genome_build="GRCh38") -> dict[str, Any]:
    """Build unambiguous rsID aliases from explicit identity records and proofs."""
    variants: dict[str, dict[str, Any]] = {}
    source_candidates: dict[str, set[str]] = {}
    if genome_build != "GRCh38":
        return _finish(variants, {}, genome_build)

    def add(identifier, rsids, source, *, preferred=True):
        canonical = _canonical(identifier)
        if canonical is None:
            return
        row = variants.setdefault(canonical, {"variant_id": canonical, "rsids": [],
                                             "preferred_rsids": [], "sources": []})
        row["rsids"] = _rsids([*row["rsids"], *rsids])
        if preferred:
            row["preferred_rsids"] = _rsids([*row["preferred_rsids"], *rsids])
        if source not in row["sources"]:
            row["sources"].append(source)

    for record in variant_records:
        if (not isinstance(record, Mapping) or record.get("provider") != "opentargets"
                or record.get("genome_build") != genome_build
                or not _HASH.fullmatch(str(record.get("response_sha256", "")))):
            continue
        value = record.get("variant")
        if not isinstance(value, Mapping) or not (canonical := _canonical(value.get("id"))):
            continue
        if value.get("genomeBuild", genome_build) != genome_build:
            continue
        chromosome, position, ref, alt = canonical.split("_")
        expected = {"chromosome": chromosome, "position": int(position),
                    "referenceAllele": ref, "alternateAllele": alt}
        if any(key in value and str(value[key]).upper() != str(actual).upper()
               for key, actual in expected.items()):
            continue
        add(canonical, _rsids(value.get("rsIds")),
            {"provider": "opentargets", "response_sha256": record["response_sha256"],
             "record_path": str(record.get("record_path") or "variant")})

    plan = ld_expansion if isinstance(ld_expansion, Mapping) else {}
    proofs = [*source_normalizations, *(plan.get("source_identity_normalizations") or ())]
    for proof in proofs:
        if not validate_source_normalization(proof):
            continue
        canonical = proof["canonical_id"]
        source_candidates.setdefault(proof["original_id"], set()).add(canonical)
        add(canonical, _rsids(proof["identity"].get("rsids")),
            {"provider": "opentargets", "proof_sha256": proof["proof_sha256"]})

    # Cross-provider aliases require the previously validated full identity
    # binding, including both allele orientations and response receipts.
    from .providers.ldlink_identity import valid_canonical_binding

    for check in plan.get("proxy_identity_checks", ()):
        if not isinstance(check, Mapping):
            continue
        binding = check.get("canonical_identity_binding")
        if not isinstance(binding, Mapping):
            continue
        canonical = _canonical(binding.get("variant_id"))
        rsid = binding.get("ldlink_rsid")
        if canonical is None or not _RS.fullmatch(str(rsid or "")):
            continue
        if (check.get("variant_id") != canonical or check.get("population") != binding.get("population")
                or check.get("population_request_key") != binding.get("population_request_key")
                or check.get("population_response_sha256") != binding.get("population_response_sha256")):
            continue
        chromosome, position, ref, alt = canonical.split("_")
        identity = {"variant_id": canonical, "genome_build": genome_build,
                    "chromosome": chromosome, "position": int(position), "end": int(position),
                    "reference_allele": ref, "alternate_allele": alt,
                    "rsids": list(binding.get("ot_rsids") or [])}
        meta = {"genome_build": genome_build, "chromosome": chromosome, "position": int(position),
                "end": int(position), "alleles": [ref, alt], "population": binding.get("population"),
                "request_key": binding.get("population_request_key"),
                "response_sha256": binding.get("population_response_sha256")}
        try:
            valid = valid_canonical_binding(binding, identity, meta, rsid)
        except (KeyError, TypeError, ValueError):
            valid = False
        if valid:
            provenance = {"provider": "opentargets_ldlink", "proof_sha256": _digest(binding)}
            add(canonical, identity["rsids"], provenance)
            add(canonical, [rsid], provenance, preferred=False)

    source_aliases = {key: next(iter(values)) for key, values in source_candidates.items() if len(values) == 1}
    return _finish(variants, source_aliases, genome_build)


def _finish(variants, source_aliases, genome_build):
    owners: dict[str, set[str]] = {}
    for identifier, row in variants.items():
        for rsid in row["rsids"]:
            owners.setdefault(rsid, set()).add(identifier)
        row["sources"].sort(key=lambda source: json.dumps(source, sort_keys=True))
    result = {"schema": SCHEMA, "genome_build": genome_build,
              "variants": dict(sorted(variants.items())), "source_aliases": dict(sorted(source_aliases.items())),
              "rsid_aliases": {key: next(iter(values)) for key, values in sorted(owners.items()) if len(values) == 1},
              "ambiguous_rsids": {key: sorted(values) for key, values in sorted(owners.items()) if len(values) > 1}}
    result["identity_sha256"] = _digest(result)
    return result


def validate_variant_identity_registry(registry: Any) -> bool:
    if (not isinstance(registry, Mapping) or registry.get("schema") != SCHEMA
            or registry.get("genome_build") != "GRCh38"):
        return False
    try:
        if set(registry) != {"schema", "genome_build", "variants", "source_aliases", "rsid_aliases",
                             "ambiguous_rsids", "identity_sha256"}:
            return False
        if any(not isinstance(registry[key], Mapping) for key in (
                "variants", "source_aliases", "rsid_aliases", "ambiguous_rsids")):
            return False
        owners: dict[str, set[str]] = {}
        for identifier, row in registry["variants"].items():
            if (_canonical(identifier) != identifier or not isinstance(row, Mapping)
                    or set(row) != {"variant_id", "rsids", "preferred_rsids", "sources"}
                    or row["variant_id"] != identifier or row["rsids"] != _rsids(row["rsids"])
                    or row["preferred_rsids"] != _rsids(row["preferred_rsids"])
                    or not set(row["preferred_rsids"]).issubset(row["rsids"])
                    or not isinstance(row["sources"], list) or not row["sources"]):
                return False
            for source in row["sources"]:
                if (not isinstance(source, Mapping)
                        or source.get("provider") not in {"opentargets", "opentargets_ldlink"}
                        or not _HASH.fullmatch(str(source.get("proof_sha256") or source.get("response_sha256") or ""))):
                    return False
            for rsid in row["rsids"]:
                owners.setdefault(rsid, set()).add(identifier)
        for original, canonical in registry["source_aliases"].items():
            if _canonical(original) != original or canonical not in registry["variants"]:
                return False
            source, target = original.split("_"), canonical.split("_")
            if source[:2] != target[:2] or source[2:] != list(reversed(target[2:])):
                return False
        expected_unique = {rsid: next(iter(values)) for rsid, values in owners.items() if len(values) == 1}
        expected_ambiguous = {rsid: sorted(values) for rsid, values in owners.items() if len(values) > 1}
        if registry["rsid_aliases"] != expected_unique or registry["ambiguous_rsids"] != expected_ambiguous:
            return False
        return _digest({key: value for key, value in registry.items() if key != "identity_sha256"}) == registry.get("identity_sha256")
    except (KeyError, TypeError, ValueError):
        return False


def resolve_variant_identity(identifier: str, registry: Mapping[str, Any] | None) -> str | None:
    """Return no rsID owner when its allele-specific identity is ambiguous."""
    canonical = _canonical(identifier)
    if not validate_variant_identity_registry(registry):
        return canonical
    if canonical:
        return registry["source_aliases"].get(canonical, canonical)
    return registry["rsid_aliases"].get(str(identifier).lower()) if _RS.fullmatch(str(identifier)) else None


def variant_display(identifier: str, registry: Mapping[str, Any] | None) -> dict[str, Any]:
    """Prefer a verified rsID; preserve the source identifier as the fallback."""
    valid = validate_variant_identity_registry(registry)
    canonical = _canonical(identifier)
    if valid:
        canonical = (registry["source_aliases"].get(canonical, canonical) if canonical
                     else registry["rsid_aliases"].get(str(identifier).lower()))
    rsid = None
    if canonical and valid:
        row = registry["variants"].get(canonical, {})
        choices = row.get("preferred_rsids") or row.get("rsids") or []
        rsid = next((item for item in choices if registry["rsid_aliases"].get(item) == canonical), None)
    return {"variant_id": identifier, "canonical_variant_id": canonical,
            "genome_build": "GRCh38" if canonical and valid else None, "rsid": rsid,
            "display_label": rsid or identifier}


def variant_aliases(registry: Mapping[str, Any] | None) -> dict[str, str]:
    """Validate once before a batch of numerical checks; never include ambiguous rsIDs."""
    if not validate_variant_identity_registry(registry):
        return {}
    return {**registry["source_aliases"], **registry["rsid_aliases"]}


def public_variant_labels(registry: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return display-only labels; these cannot establish identity during fact review."""
    if not validate_variant_identity_registry(registry):
        return []
    result = []
    for identifier in sorted(set(registry["variants"]) | set(registry["source_aliases"])):
        canonical = registry["source_aliases"].get(identifier, identifier)
        row = registry["variants"][canonical]
        choices = row["preferred_rsids"] or row["rsids"]
        rsid = next((item for item in choices if registry["rsid_aliases"].get(item) == canonical), None)
        if rsid:
            result.append({"variant_id": identifier, "canonical_variant_id": canonical,
                           "genome_build": "GRCh38", "rsid": rsid})
    return result
