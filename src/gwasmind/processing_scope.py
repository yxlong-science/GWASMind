"""Public interpretation scope for a receipt-verified, retained large result.

This marker describes the loader's outcome; it does not replace receipt verification.
"""
from collections.abc import Mapping
from typing import Any


RESULT_PROCESSING_SUMMARY = "Results retained · processing limit reached"
RESULT_PROCESSING_LIMIT_MESSAGE = (
    "The complete retrieved result was preserved, but it exceeds the local processing "
    "limit. This response uses only the uploaded GWAS data; external evidence "
    "was not reviewed and gene ranking was not performed. This processing limit does "
    "not establish an absence of evidence or prior associations."
)


def is_source_only_processing_limit(result: Mapping[str, Any]) -> bool:
    marker = result.get("result_processing_limit")
    if not isinstance(marker, Mapping):
        return False
    size, limit = marker.get("result_bytes"), marker.get("limit_bytes")
    return (
        result.get("status") == "INCOMPLETE"
        and result.get("external_evidence_processed") is False
        and result.get("complete_result_retained") is True
        and result.get("evidence") == [] and result.get("ranked_genes") == []
        and marker.get("phase") == "result_processing"
        and marker.get("status") == "size_limit"
        and all(marker.get(key) is True for key in
                ("source_only", "complete_result_retained", "ranking_skipped"))
        and marker.get("external_evidence_processed") is False
        and type(size) is int and type(limit) is int and size > limit > 0
    )
