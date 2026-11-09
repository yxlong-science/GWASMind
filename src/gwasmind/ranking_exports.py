"""Immutable, credential-free exports of the exact frozen ranking request."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .audit import sha256_json

EXPORT_NAMES = frozenset({"ranking-input", "ranking-evidence"})
RECEIPT_NAME = "ranking_input_receipt.json"
_HASH = re.compile(r"^[a-f0-9]{64}$")


def _cell(value):
    if value is None:
        return "Not reported"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in ("\\", "`", "*", "_", "[", "]", "|"):
        text = text.replace(character, "\\" + character)
    return text.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")


def _json_block(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    fence = "`" * max(3, 1 + max((len(x) for x in re.findall(r"`+", text)), default=0))
    return f"{fence}json\n{text}\n{fence}"


def _table(headers, rows):
    return ["| " + " | ".join(_cell(value) for value in headers) + " |",
            "|" + "---|" * len(headers),
            *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows), ""]


def _field_rows(value, prefix=""):
    """Keep provider field names and scalar values; never derive or round values."""
    if isinstance(value, Mapping) and value:
        for key, item in value.items():
            yield from _field_rows(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list) and value:
        for index, item in enumerate(value, 1):
            yield from _field_rows(item, f"{prefix}[{index}]")
    else:
        display = "No fields supplied" if value == {} else "No entries supplied" if value == [] else value
        yield prefix or "Value", display


def _fields(value):
    return _table(("Supplied field", "Value"), _field_rows(value))


def _reference_rows(items, records):
    for item in items:
        for record in _referenced_records(item, records) or [{}]:
            variant = record.get("variant") or {}
            context = record.get("context") or {}
            study = record.get("study") or {}
            assessment = record.get("context_assessment") or {}
            yield (item.get("fact_id"), record.get("evidence_ref") or "; ".join(item.get("evidence_refs") or []),
                   variant.get("variant_id"),
                   context.get("biosample") or study.get("biosample_name") or study.get("tissue"),
                   assessment.get("context_class") or assessment.get("scope"))


def _referenced_records(item, records):
    refs = set(item.get("evidence_refs") or [])
    return [row for row in records if row.get("fact_id") == item.get("fact_id")
            and (not refs or row.get("evidence_ref") in refs)]


def _observation(record, *, level=5):
    label = record.get("evidence_ref", record.get("evidence_id"))
    lines = [f"{'#' * level} {_cell(label)} / {_cell(record.get('fact_id'))}", ""]
    keys = ("kind", "gene_id", "source_ref", "evidence_id", "qtl_type", "finemapping_method",
            "supports", "confidence", "provenance_complete")
    lines.extend(_table(("Observation", "Reported value"), ((key, record[key]) for key in keys if key in record)))
    sections = (("statistics", "Reported measurements"),
                ("credible_set_member", "Credible-set member"),
                ("other_reported_statistics", "Other reported measurements"),
                ("selected_background", "Supplied background items"))
    for key, title in sections:
        if key not in record:
            continue
        lines.extend([title, "", *_fields(record[key])])
    detail_sections = (("variant", "Variant and LD relationship"), ("study", "Study metadata"),
                       ("right_study", "Counterpart study metadata"),
                       ("context", "Reported biological context"),
                       ("context_assessment", "Frozen context assessment"),
                       ("quality_controls", "Reported quality controls"))
    lines.extend(["<details>", "<summary>Study, variant, context and provenance details</summary>", ""])
    for key, title in detail_sections:
        if key in record:
            lines.extend([title, "", *_fields(record[key])])
    known = {*keys, *(key for key, _ in sections), *(key for key, _ in detail_sections), "fact_id", "evidence_ref"}
    extra = {key: value for key, value in record.items() if key not in known}
    if extra:
        lines.extend(["Additional supplied fields", "", *_fields(extra)])
    lines.extend(["</details>", ""])
    return lines


def _render_fact_details(items, records, rendered):
    lines = []
    for item in items:
        fact_id = item.get("fact_id")
        matched = _referenced_records(item, records)
        for record in matched:
            key = (record.get("fact_id"), record.get("evidence_ref"))
            if key not in rendered:
                rendered.add(key)
                lines.extend(_observation(record))
        if not matched:
            lines.extend([f"Observation details for {_cell(fact_id)}: Not reported.", ""])
    return lines


def _render_candidate(card, index, records):
    lines = [f"## Candidate {index}: {_cell(card.get('gene_id'))}", "",
             "### Identity and retrieval coverage", ""]
    lines.extend(_table(("Identity", "Value"), (("Gene ID", card.get("gene_id")),
                                               ("Gene symbol", card.get("gene_symbol")))))
    availability = card.get("availability") or {}
    lines.extend(_table(("Evidence family", "Query status", "Module status", "Retained facts", "Displayed items"),
                        ((family, status.get("query_status"),
                          "; ".join(f"{key}: {value}" for key, value in (status.get("module_statuses") or {}).items()),
                          status.get("retained_fact_count"), status.get("displayed_item_count"))
                         for family, status in availability.items())))
    lines.extend(["An empty retained set is not a negative biological result. Query coverage and retained evidence are distinct.", ""])
    lines.extend(["### Molecular evidence", ""])
    direct = card.get("direct_molecular") or {}
    rendered = set()
    for family, title in (("molecular_qtl", "Molecular QTL"), ("colocalizations", "Colocalization"),
                          ("coding_splicing", "Coding and splicing")):
        lines.extend([f"#### {title}", ""])
        items = direct.get(family) or []
        if items:
            lines.extend(_table(("Fact", "Evidence reference", "Variant", "Reported biosample", "Context class"),
                                _reference_rows(items, records)))
            lines.extend(_render_fact_details(items, records, rendered))
        else:
            lines.extend(["No retained facts supplied for this family; see query status above.", ""])
    lines.extend(["### Regulatory evidence", ""])
    groups = card.get("regulatory_groups") or []
    if not groups:
        lines.extend(["No retained regulatory groups supplied; see query status above.", ""])
    for group_index, group in enumerate(groups, 1):
        lines.extend([f"#### Regulatory group {group_index}", ""])
        keys = ("source_id", "dataset", "method", "model", "model_version", "context_match_class",
                "regulatory_element_type", "display_mode", "relationship_count", "displayed_relationship_count",
                "summarized_relationship_count", "flagged_relationship_count", "independent_experiment_count",
                "score_comparison_within_group_supported", "score_range")
        lines.extend(_fields({key: group.get(key) for key in keys}))
        items = group.get("selected_relationships")
        if items is None:
            items = list((group.get("representatives") or {}).values())
        lines.extend(_table(("Fact", "Evidence reference", "Relationship scope", "Variant reference", "Interval reference"),
                            ((item.get("fact_id"), "; ".join(item.get("evidence_refs") or []),
                              item.get("relationship_scope"), item.get("variant_ref"), item.get("interval_ref"))
                             for item in items)))
        lines.extend(_render_fact_details(items, records, rendered))
        flags = group.get("qualified_or_conflicting_evidence") or []
        if flags:
            lines.extend(["Qualified or conflicting evidence retained", "", *_fields(flags)])
            lines.extend(_render_fact_details(flags, records, rendered))
    lines.extend(["### Positional context", ""])
    positional = card.get("positional_summary") or {}
    lines.extend(_fields(positional))
    for scope in ("source_selected", "ld_proxy"):
        nearest = (positional.get(scope) or {}).get("nearest")
        if nearest:
            lines.extend(_render_fact_details([nearest], records, rendered))
    lines.extend(["### Background evidence", ""])
    background = card.get("background_summary") or {}
    lines.extend(_fields({key: background[key] for key in ("role", "selection_basis") if key in background}))
    for family in ("function", "expression", "pathway", "literature"):
        lines.extend([f"#### {family.capitalize()}", ""])
        summary = background.get(family) or {}
        lines.extend(_fields({key: value for key, value in summary.items() if key != "selected"}))
        items = summary.get("selected") or []
        if not items:
            lines.extend(["No background items supplied for this family.", ""])
        else:
            lines.extend(_table(("Fact", "Evidence reference", "Variant", "Reported biosample", "Context class"),
                                _reference_rows(items, records)))
            lines.extend(_render_fact_details(items, records, rendered))
    lines.extend(["### Card coverage", "", *_fields(card.get("coverage") or {})])
    return lines, rendered


V2G_CARD_FORMAT = "gwasmind_v2g_gene_cards_v1"
_V2G_CATEGORY_TITLES = (("coding", "Variant consequence"), ("qtl", "Molecular QTL"),
                        ("coloc", "GWAS-QTL colocalisation"), ("e2g", "Enhancer-to-gene"),
                        ("expression", "Expression"), ("constraint", "Constraint"),
                        ("association", "Target-disease association"), ("function", "Function"))


def _render_v2g_card(card, index):
    lines = [f"## {index}. {_cell(card.get('symbol') or card.get('gene_id'))} ({_cell(card.get('gene_id'))})", ""]
    if "tier" in card:
        lines.extend([f"Evidence tier {_cell(card.get('tier'))}: {_cell('; '.join(card.get('tier_basis') or []))}", ""])
    lines.extend(["### Position", "", *_fields(card.get("auxiliary") or {})])
    evidence = card.get("evidence") or {}
    for key, title in _V2G_CATEGORY_TITLES:
        category = evidence.get(key)
        if not isinstance(category, Mapping):
            continue
        lines.extend([f"### {title}", ""])
        records = category.get("records")
        summary = {name: value for name, value in category.items() if name != "records"}
        lines.extend(_fields(summary))
        for position, record in enumerate(records or [], 1):
            reference = record.get("evidence_id") if isinstance(record, Mapping) else None
            lines.extend([f"#### Record {position}" + (f" [{reference}]" if reference else ""), "",
                          *_fields(record)])
    return lines


def render_ranking_evidence(bundle):
    """Render saved data without selection, rounding, inference or new retrieval."""
    payload = bundle["model_payload"]
    locus = payload.get("locus", {})
    comparison = payload.get("gene_comparison", {})
    if comparison.get("format") == V2G_CARD_FORMAT:
        from .v2g.compact_input import decode_v2g_payload

        payload = decode_v2g_payload(payload)
        locus = payload.get("locus", {})
        comparison = payload["gene_comparison"]
        lines = ["# Gene evidence cards", "",
                 ("One card per candidate gene in the lead window, exactly as frozen before the ranking request. "
                  "The position is the frozen presentation order, not the model's gene ranking. Bracketed E "
                  "references are the citable evidence identifiers. Values are displayed as supplied; "
                  "Not reported means a supplied null or missing field, not a negative result."), ""]
        lines.extend(_table(("Item", "Value"), (("Query phenotype", locus.get("trait")),
                            ("Source GWAS phenotype", locus.get("source_trait")), ("Task", locus.get("task")),
                            ("Snapshot", bundle["snapshot_id"]), ("Model payload SHA-256", bundle["model_input_sha256"]),
                            ("Messages SHA-256", bundle["messages_sha256"]))))
        lines.extend(["## Selected SNPs and biological context", ""])
        lines.extend(_fields({key: locus[key] for key in
                             ("source_observations", "relevant_tissues", "biological_context") if key in locus}))
        for index, card in enumerate(comparison.get("comparison_cards", []), 1):
            lines.extend(_render_v2g_card(card, index))
        if comparison.get("retrieval_summary"):
            lines.extend(["## Shared retrieval coverage", "", *_fields(comparison["retrieval_summary"])])
        if comparison.get("ld_context"):
            lines.extend(["## Recorded LD relationships", "", *_fields(comparison["ld_context"])])
        lines.extend(["## Evidence projection", "", *_fields(payload.get("evidence_projection", {}))])
        return "\n".join(lines)
    lines = ["# Ranking evidence", "", "This file contains the evidence frozen before the ranking request.", "",
             ("The position is the frozen presentation order, not the model's gene ranking. "
             "Values are displayed as supplied, without rounding or scientific reinterpretation. "
             "Not reported means a supplied null or missing field; it does not mean a negative result. "
             "The complete frozen input and exact request messages are preserved in the accompanying ranking-input JSON."), ""]
    lines.extend(_table(("Item", "Value"), (("Query phenotype", locus.get("trait")),
                        ("Source GWAS phenotype", locus.get("source_trait")), ("Task", locus.get("task")),
                        ("Snapshot", bundle["snapshot_id"]), ("Model payload SHA-256", bundle["model_input_sha256"]),
                        ("Messages SHA-256", bundle["messages_sha256"]))))
    lines.extend(["## Selected SNPs and biological context", ""])
    lines.extend(_fields({key: locus[key] for key in
                         ("source_observations", "relevant_tissues", "biological_context") if key in locus}))
    cards = comparison.get("comparison_cards", [])
    records = comparison.get("citation_records", [])
    rendered = set()
    for index, card in enumerate(cards, 1):
        card_lines, card_rendered = _render_candidate(card, index, records)
        lines.extend(card_lines)
        rendered.update(card_rendered)
    if cards:
        remaining = [record for record in records if (record.get("fact_id"), record.get("evidence_ref")) not in rendered]
        if remaining:
            lines.extend(["## Additional supplied observations", ""])
            for record in remaining:
                lines.extend(_observation(record, level=3))
        lines.extend(["## Sources and overall coverage", "", "### Sources", "", *_fields(comparison.get("sources", {})),
                      "### Overall coverage", "", *_fields(comparison.get("coverage", {})),
                      "### Evidence projection", "", *_fields(payload.get("evidence_projection", {}))])
    else:
        lines.extend(["## Scientific evidence", "",
                      "No gene comparison cards were supplied. The original scientific structure follows.", "",
                      _json_block({key: value for key, value in payload.items() if key != "locus"}), ""])
    return "\n".join(lines)


def _write_bytes_once(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError("A frozen ranking export cannot be replaced with different content")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".ranking-export-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def store_ranking_exports(artifacts, run_id: str, frozen: Mapping):
    """Commit exports before their receipt makes them available to the UI."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", run_id) or run_id in {".", ".."}:
        raise ValueError("Invalid ranking export run identity")
    payload, messages = frozen.get("model_payload"), frozen.get("messages")
    if not isinstance(payload, Mapping) or not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Frozen ranking exports require the exact payload and two messages")
    if [row.get("role") for row in messages if isinstance(row, Mapping)] != ["system", "user"]:
        raise ValueError("Frozen ranking messages have invalid roles")
    if any(set(row) != {"role", "content"} or not isinstance(row["content"], str) for row in messages):
        raise ValueError("Ranking exports accept only message roles and content")
    if json.loads(messages[1]["content"]) != payload:
        raise ValueError("Frozen ranking payload does not match the actual user message")
    payload_hash, messages_hash = sha256_json(payload), sha256_json(messages)
    if frozen.get("model_input_sha256") != payload_hash or frozen.get("messages_sha256") != messages_hash:
        raise ValueError("Frozen ranking input hash verification failed")
    presentation = frozen.get("presentation", {})
    identity = {"model_input_sha256": payload_hash, "messages_sha256": messages_hash, "presentation": presentation}
    snapshot_id = sha256_json(identity)
    base = (artifacts.root / "runs" / run_id).resolve()
    if artifacts.root not in base.parents:
        raise ValueError("Ranking export escaped artifact root")
    bundle = {"format": "gwasmind_ranking_input_v1", "schema_version": 1,
              "snapshot_id": snapshot_id, **identity, "model_payload": payload, "messages": messages}
    candidates = payload.get("locus", {}).get("candidate_genes", [])
    descriptor = {"snapshot_id": snapshot_id, "model_input_sha256": payload_hash,
                  "messages_sha256": messages_hash, "candidate_count": len(candidates)}
    files = {}
    existing_receipt = base / RECEIPT_NAME
    if existing_receipt.exists():
        saved = json.loads(existing_receipt.read_text(encoding='utf-8'))
        if not isinstance(saved, Mapping) or any(saved.get(key) != value for key, value in descriptor.items()):
            raise ValueError("Frozen ranking export cannot be replaced")
    for name, suffix, content in (
        ("ranking-input", ".model-input.json", json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()),
        ("ranking-evidence", ".evidence.md", render_ranking_evidence(bundle).encode()),
    ):
        filename = snapshot_id + suffix
        target = (base / "model_inputs" / filename).resolve()
        if base not in target.parents:
            raise ValueError("Ranking export escaped the run")
        _write_bytes_once(target, content)
        files[name] = {"filename": filename, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    receipt = {"format": "gwasmind_ranking_exports_v1", **descriptor, "files": files}
    _write_bytes_once(base / RECEIPT_NAME, json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
    return descriptor


def verified_export(base: Path, name: str, *, max_receipt_bytes: int):
    """Use only fixed filenames inside a verified run, including after restart."""
    if name not in EXPORT_NAMES:
        raise ValueError("Unknown ranking export")
    receipt_path = base / RECEIPT_NAME
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise FileNotFoundError("Ranking evidence has not been frozen")
    if receipt_path.stat().st_size > max_receipt_bytes:
        raise ValueError("Ranking export receipt exceeds its size limit")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping) or receipt.get("format") != "gwasmind_ranking_exports_v1" or any(
        not isinstance(receipt.get(key), str) or not _HASH.fullmatch(receipt[key])
        for key in ("snapshot_id", "model_input_sha256", "messages_sha256")
    ):
        raise ValueError("Ranking export receipt has invalid identities")
    suffix = {"ranking-input": ".model-input.json", "ranking-evidence": ".evidence.md"}[name]
    expected = receipt["snapshot_id"] + suffix
    files = receipt.get("files")
    entry = files.get(name) if isinstance(files, Mapping) else None
    if not isinstance(entry, Mapping) or entry.get("filename") != expected or not _HASH.fullmatch(str(entry.get("sha256", ""))):
        raise ValueError("Ranking export receipt has an invalid file binding")
    lexical = base / "model_inputs" / expected
    path = lexical.resolve()
    if lexical.is_symlink() or base not in path.parents or not path.is_file():
        raise ValueError("Ranking export is unavailable or outside the run")
    if path.stat().st_size != entry.get("bytes"):
        raise ValueError("Ranking export byte count changed")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != entry["sha256"]:
        raise ValueError("Ranking export content changed after freezing")
    descriptor = {key: receipt.get(key) for key in ("snapshot_id", "model_input_sha256", "messages_sha256", "candidate_count")}
    if type(descriptor["candidate_count"]) is not int or descriptor["candidate_count"] < 0:
        raise ValueError("Ranking export candidate count is invalid")
    return path, descriptor
