"""Auditable source-ID corrections, separate from effect-allele harmonisation.

Only an exact, unique REF/ALT reversal confirmed by the OT identity resolver on
GRCh38 is eligible. The uploaded observation remains the source of its P value;
this contract authorises neither a new observation nor a beta/sign change.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy

_CANONICAL = re.compile(r"^([1-9]|1[0-9]|2[0-2]|X|Y|MT)_([1-9][0-9]*)_([ACGT])_([ACGT])$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_VERSION = "gwasmind-source-normalization-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _receipt_valid(receipt):
    return (isinstance(receipt, Mapping) and receipt.get("reference_provider") == "opentargets"
            and receipt.get("status") == "success"
            and bool(_HASH.fullmatch(str(receipt.get("request_key", ""))))
            and bool(_HASH.fullmatch(str(receipt.get("sha256", "")))))


def build_source_normalization(original_id, ot_result, genome_build="GRCh38"):
    """Return a reproducible proof for one exact reversal, or None.

    A successful empty lookup, transport failure, multi-allelic ambiguity, build
    mismatch or general identity conflict cannot produce this proof.
    """
    match = _CANONICAL.fullmatch(original_id) if isinstance(original_id, str) else None
    if (genome_build not in {"GRCh38", "grch38", "hg38"} or not match or match[3] == match[4]
            or not isinstance(ot_result, Mapping) or ot_result.get("status") != "conflicting"
            or ot_result.get("reason_code") != "ot_reference_allele_order_conflict"):
        return None
    canonical = f"{match[1]}_{match[2]}_{match[4]}_{match[3]}"
    check = ot_result.get("allele_order_check", {})
    if (not isinstance(check, Mapping) or check.get("original_variant_id") != original_id
            or check.get("reverse_variant_id") != canonical or check.get("reverse_status") != "resolved"
            or check.get("input_rewritten") is not False):
        return None
    candidates, receipts = ot_result.get("candidates"), ot_result.get("receipts")
    if (not isinstance(candidates, (list, tuple)) or len(candidates) != 1
            or not isinstance(receipts, (list, tuple)) or not receipts
            or not all(_receipt_valid(item) for item in receipts)):
        return None
    identity = candidates[0]
    if (not isinstance(identity, Mapping) or identity.get("variant_id") != canonical
            or identity.get("chromosome") != match[1] or type(identity.get("position")) is not int
            or identity["position"] != int(match[2]) or identity.get("genome_build") != "GRCh38"
            or identity.get("reference_allele") != match[4] or identity.get("alternate_allele") != match[3]):
        return None
    own = identity.get("receipts")
    keys = {(row["request_key"], row["sha256"]) for row in receipts}
    if (not isinstance(own, (list, tuple)) or not own or any(not _receipt_valid(row)
            or (row["request_key"], row["sha256"]) not in keys for row in own)):
        return None
    proof = {"schema_version": SCHEMA_VERSION, "original_id": original_id, "canonical_id": canonical,
             "genome_build": "GRCh38", "status": "normalized", "ref_alt_swapped": True,
             "reason_code": "ot_reference_allele_order_conflict",
             "source_p_preserved": True, "effect_sign_changed": False,
             "reference_provider": "opentargets", "provider_version": "1.0.0",
             "identity": deepcopy(dict(identity)), "receipts": deepcopy(list(receipts))}
    proof["proof_sha256"] = _digest(proof)
    return proof


def validate_source_normalization(proof):
    """Validate the complete normalisation contract, including its stored hash."""
    if not isinstance(proof, Mapping) or proof.get("schema_version") != SCHEMA_VERSION:
        return False
    try:
        original, canonical = proof["original_id"], proof["canonical_id"]
        rebuilt = build_source_normalization(original, {
            "status": "conflicting", "reason_code": "ot_reference_allele_order_conflict",
            "candidates": [proof["identity"]], "receipts": proof["receipts"],
            "allele_order_check": {"original_variant_id": original, "reverse_variant_id": canonical,
                                   "reverse_status": "resolved", "input_rewritten": False},
        }, proof["genome_build"])
        return rebuilt is not None and dict(proof) == rebuilt
    except (KeyError, TypeError, ValueError):
        return False
