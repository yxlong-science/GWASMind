"""Check derived evidence volume before building duplicate per-gene card views."""
from ..configuration import parameter
from ..evidence_store import iter_canonical


class EvidenceCapacityError(ValueError):
    """An identified scientific channel exceeds the configured processing scope."""


def check_channel_capacity(bundle):
    channels = {
        "molecular QTL": [row for target in bundle.targets.values()
                          for row in (target.get("credibleSets") or {}).get("rows", [])],
        "expression": [row for target in bundle.targets.values()
                       for row in (target.get("baselineExpression") or {}).get("rows", [])],
        "enhancer-to-gene": [row for variant in bundle.variants.values() if variant
                             for row in (variant.get("enhancerToGenes") or {}).get("rows", [])],
        "colocalisation": bundle.gwas_credible_sets,
    }
    receipts = {}
    for name, rows in channels.items():
        count_limit = parameter("v2g.processing.max_records")
        byte_limit = parameter("v2g.processing.max_channel_bytes")
        if len(rows) > count_limit:
            raise EvidenceCapacityError(f"Locus evidence exceeds processing range: {name} has {len(rows)} records; limit {count_limit}.")
        size = 0
        for chunk in iter_canonical(rows):
            size += len(chunk.encode("utf-8"))
            if size > byte_limit:
                raise EvidenceCapacityError(f"Locus evidence exceeds processing range: {name} exceeds {byte_limit} bytes before classification.")
        receipts[name] = {"records": len(rows), "bytes": size, "record_limit": count_limit, "byte_limit": byte_limit}
    bundle.coverage["processing_capacity"] = receipts
    return receipts
