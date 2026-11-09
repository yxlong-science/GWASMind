"""Bounded GRCh37 LDproxy recovery with explicit GRCh38 identity mapping.

No coordinate offset or strand conversion is inferred. The queried interval
remains a GRCh37 interval, so its coverage of the GRCh38 window is unproven.
"""
from __future__ import annotations

import csv
import io
import re
from copy import deepcopy

from .ldlink import _alleles, _coordinate, _failure, _parse_proxy


def _parse_grch37(text, variant, *, threshold, flank):
    # A provider error is authoritative even when this allele contrast cannot
    # be mapped by the SNV-only fallback. Do not mask it as an identity failure.
    if text.lstrip("\ufeff \t\r\n").startswith(("{", "[")) or re.search(r"(?im)^\s*(?:warning|error)\b", text):
        _parse_proxy(text, variant, threshold=threshold, window_bp=flank)
    # Select the self measurement without applying GRCh38 coordinates to it.
    # The ordinary parser then checks every distance and population contrast.
    expected = set(variant.get("selected_alleles", variant.get("alleles", [])))
    if len(expected) != 2 or any(allele not in {"A", "C", "G", "T"} for allele in expected):
        raise _failure("source_identity_unverified")
    self_rows = []
    for row in csv.DictReader(io.StringIO(text.lstrip("\ufeff")), delimiter="\t"):
        if row.get("RS_Number") != variant["rsid"]:
            continue
        try:
            if int(row.get("Distance", "")) == 0 and float(row.get("R2", "")) == 1:
                chrom, position = _coordinate(row.get("Coord"))
                if set(_alleles(row.get("Alleles"))) == expected:
                    self_rows.append({**variant, "chromosome": chrom, "position": position})
        except (RuntimeError, TypeError, ValueError):
            pass
    if len(self_rows) != 1:
        # Reuse the normal error parser for JSON/text errors, preserving the
        # final response reason rather than the initial GRCh38 not-found code.
        if not text.lstrip("\ufeff").startswith("RS_Number\t"):
            _parse_proxy(text, variant, threshold=threshold, window_bp=flank)
        raise _failure("source_identity_unverified")
    parsed = _parse_proxy(text, self_rows[0], threshold=threshold, window_bp=flank)
    parsed["requested_scope_complete"] = False
    return parsed


def _mapped_identity(bridge, result, row):
    if result.get("status") == "failed":
        raise _failure(result.get("reason_code") or "reference_identity_unresolved")
    candidates = bridge._candidates(result)
    if result.get("status") != "resolved" or candidates is None:
        raise _failure("reference_identity_unverified")
    # A unique exact rsID and SNV allele contrast is required. No coordinate
    # guessing, merged aliases, complementing, or indel normalization is used.
    if len(candidates) != 1:
        raise _failure("reference_identity_ambiguous")
    identity = candidates[0]
    alleles = [identity["reference_allele"], identity["alternate_allele"]]
    if (row["rsid"] not in identity["rsids"]
            or any(allele not in {"A", "C", "G", "T"} for allele in alleles)
            or set(alleles) != set(row["alleles"])
            or not any(receipt.get("status") == "success" and receipt.get("request_key")
                       and receipt.get("sha256") for receipt in result.get("receipts", []))):
        raise _failure("reference_identity_unverified")
    return identity


def _mapped_row(row, identity, source_position):
    return {**row, "query_build": "GRCh37", "analysis_build": "GRCh38",
            "query_chromosome": row["chromosome"], "query_position": row["position"],
            "query_distance": row["distance"], "chromosome": identity["chromosome"],
            "position": identity["position"], "distance": identity["position"] - source_position,
            "build_identity": deepcopy(identity)}


def grch37_proxies(provider, variant, params, *, threshold, flank):
    parsed, receipt = provider._request_ldlink("ldproxy", {**params, "genome_build": "grch37"},
        lambda text: _parse_grch37(text, variant, threshold=threshold, flank=flank))
    source = parsed["source_self_row"]
    eligible = [row for row in parsed["rows"] if row["r2"] >= threshold and abs(row["distance"]) <= flank]
    bridge = provider._identity_bridge
    with bridge._lock:
        results = bridge._lookup(sorted({variant["rsid"], *(row["rsid"] for row in eligible)}))
        try:
            source_identity = _mapped_identity(bridge, results[variant["rsid"]], source)
        except RuntimeError as exc:
            reason = getattr(exc, "reason_code", "source_identity_unverified")
            if reason in {"reference_identity_unverified", "reference_identity_ambiguous"}:
                reason = "source_identity_unverified"
            raise _failure(reason) from None
        expected_id = variant.get("canonical_variant_id") or variant.get("reference_identity", {}).get("variant_id")
        if (source_identity["chromosome"] != variant["chromosome"]
                or source_identity["position"] != variant["position"]
                or source_identity["reference_allele"] != variant.get("reference_allele")
                or (expected_id and source_identity["variant_id"] != expected_id)):
            raise _failure("source_identity_unverified")
        mapped, excluded = [], deepcopy(parsed["excluded_rows"])
        for row in eligible:
            try:
                identity = _mapped_identity(bridge, results[row["rsid"]], row)
                item = _mapped_row(row, identity, variant["position"])
                if identity["chromosome"] != variant["chromosome"] or abs(item["distance"]) > flank:
                    excluded.append({"reported_rsid": row["rsid"], "r2": row["r2"],
                        "reason_code": "outside_grch38_window", "scope_relevant": False,
                        "query_position": row["position"], "analysis_position": identity["position"]})
                    continue
                mapped.append(item)
            except RuntimeError as exc:
                excluded.append({"reported_rsid": row["rsid"], "r2": row["r2"],
                    "reason_code": getattr(exc, "reason_code", "reference_identity_unverified"),
                    "scope_relevant": True})
    return {**parsed, "rows": mapped, "excluded_rows": excluded,
            "query_source_self_row": deepcopy(source),
            "source_self_row": _mapped_row(source, source_identity, variant["position"]),
            "query_build": "GRCh37", "analysis_build": "GRCh38", "fallback_reason": "recovered_via_grch37",
            "window_coverage_reason": "grch37_window_not_proven_complete_in_grch38",
            "requested_scope_complete": False}, receipt
