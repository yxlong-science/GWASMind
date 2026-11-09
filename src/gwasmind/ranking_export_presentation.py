"""Offline HTML views of an already frozen, credential-free ranking request.

No retrieval, prompt generation, ranking or filesystem access occurs here. The
existing compact codec only decodes shared scientific fields for presentation;
the original JSON and message strings are embedded without reconstruction.
"""
from __future__ import annotations

import base64
import html
import json
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .audit import sha256_json
from .display_numbers import three_significant_figures
from .v2g.compact_input import decode_v2g_payload
from .v2g.report import _lead_qtl_display

_BUNDLE_KEYS = frozenset({"format", "schema_version", "snapshot_id", "messages_sha256",
                          "model_input_sha256", "presentation", "model_payload", "messages"})
_SECRET_FIELDS = frozenset({"api_key", "apikey", "authorization", "access_token", "refresh_token",
                            "client_secret", "password", "private_key"})
_CATEGORIES = {"coding": "Variant consequence", "qtl": "Molecular QTL", "coloc": "Colocalization",
               "e2g": "Enhancer-to-gene", "expression": "Expression", "association": "Trait association",
               "function": "Function", "constraint": "Constraint"}
_HEADERS = {
    "evidence_id": "Evidence", "qtl_type": "QTL type", "biosample": "Biosample", "method": "Method",
    "cs_lead_variant_id": "Credible-set lead", "cs_lead_measurement": "Lead measurements",
    "lead_p_value": "Lead P value", "lead_beta": "Lead beta", "p_value": "Source P value",
    "gwas_trait": "GWAS phenotype", "h4": "PP.H4", "variant_id": "Variant", "score": "Score (3 sig. fig.)",
    "regulatory_interval": "Regulatory interval", "consequence": "Consequence", "overall": "Overall score",
    "disease": "Phenotype", "datasources": "Scores by source", "phenotype_scope": "Phenotype scope",
    "median": "Median expression", "specificity": "Specificity", "unit": "Unit", "datasource": "Source",
    "tissue": "Tissue", "cell_type": "Cell type", "context_matched": "Context measurements",
    "beta": "Beta", "pValueMantissa": "P-value mantissa", "pValueExponent": "P-value exponent",
}
_COLUMNS = {
    "qtl": ("evidence_id", "qtl_type", "biosample", "cs_lead_variant_id", "lead_p_value", "lead_beta"),
    "coloc": ("evidence_id", "qtl_type", "biosample", "gwas_trait", "h4", "method"),
    "e2g": ("evidence_id", "variant_id", "biosample", "regulatory_interval", "score"),
    "coding": ("evidence_id", "variant_id", "consequence"),
    "association": ("evidence_id", "disease", "phenotype_scope", "overall", "datasources"),
    "expression": ("tissue", "cell_type", "datasource", "median", "unit", "specificity"),
    "source": ("variant_id", "p_value"),
}
_CSS = """
:root{color-scheme:light;--ink:#20354a;--muted:#607284;--line:#d9e1e8;--blue:#28597b;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:15px/1.5 Arial,sans-serif}
main{max-width:1280px;margin:28px auto;padding:30px 36px;background:var(--paper);border:1px solid var(--line)}
h1{font-size:28px;margin:0 0 10px}h2{font-size:21px;margin:28px 0 12px}h3{font-size:17px;margin:22px 0 10px}
h4{font-size:15px;margin:18px 0 8px}p{margin:8px 0 14px}.muted,.key{color:var(--muted)}
.lead{max-width:1000px}.overview{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}
.metric{padding:11px 18px;border:1px solid var(--line);border-radius:5px;background:#f7fafc}
.metric strong{font-size:22px;margin-right:8px}.metric span{font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:7px;margin:16px 0 24px}a{color:var(--blue);overflow-wrap:anywhere}
nav a,.download{display:inline-block;border:1px solid #b9cbd8;padding:5px 10px;border-radius:4px;text-decoration:none}
article{padding:4px 0 22px;border-top:2px solid #b9cbd8;scroll-margin:16px}.gene-meta{margin:0;color:var(--muted)}
.table-wrap{overflow:auto;max-width:100%;margin:10px 0 18px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{font-weight:600;background:#eaf0f5}tbody tr:nth-child(even){background:#f8fafc}.kv th{width:27%;font-weight:400}
.kv{margin:4px 0;font-size:12px}.kv td,.kv th{padding:5px 8px}.record-source{min-width:90px}
details{margin:9px 0;border:1px solid var(--line);border-radius:4px;background:#fbfcfe}
summary{cursor:pointer;padding:8px 11px;color:var(--blue)}.details-body{padding:2px 12px 12px}
td details{margin:0}.source-note{font-size:12px;color:var(--muted)}code,pre{font-family:Consolas,monospace}
code{font-size:.92em}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:14px;background:#edf2f6;font-size:12px;line-height:1.5}
.prompt{max-width:1000px}.prompt p{margin:8px 0}.prompt li{margin:4px 0}.key{font-size:11px;display:block;font-weight:400}
.count{font-size:13px;font-weight:400;color:var(--muted)}.empty{font-style:italic;color:var(--muted)}
.message{margin-top:24px;border-top:2px solid #b9cbd8}.message-role{font-size:12px;text-transform:uppercase;letter-spacing:.07em}
footer{margin-top:30px;border-top:1px solid var(--line);padding-top:14px;font-size:12px;color:var(--muted)}
@media(max-width:720px){main{margin:0;padding:20px 15px;border:0}h1{font-size:24px}th,td{padding:6px}}
@media print{body{background:white}main{border:0;margin:0;padding:0;max-width:none}nav,.download{display:none}
article{break-before:auto}thead{display:table-header-group}tr{break-inside:avoid}.table-wrap{overflow:visible}}
"""


def _escape(value: Any) -> str:
    # Character references preserve CRs in an exact message when an HTML parser
    # performs its normal CRLF preprocessing. The JSON download remains bytes.
    return html.escape(str(value), quote=True).replace("\r", "&#13;")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Frozen ranking JSON contains duplicate fields")
        if key.casefold().replace("-", "_") in _SECRET_FIELDS:
            raise ValueError("Frozen ranking export contains a credential field")
        result[key] = value
    return result


def _load(bundle_json: str) -> tuple[dict, dict]:
    if not isinstance(bundle_json, str):
        raise TypeError("Ranking HTML expects the existing frozen JSON text")
    def invalid_constant(_value):
        raise ValueError("Frozen ranking JSON contains a non-finite value")
    bundle = json.loads(bundle_json, object_pairs_hook=_pairs, parse_constant=invalid_constant)
    if (not isinstance(bundle, dict) or set(bundle) != _BUNDLE_KEYS
            or bundle["format"] != "gwasmind_ranking_input_v1"
            or type(bundle["schema_version"]) is not int or bundle["schema_version"] != 1):
        raise ValueError("Unsupported frozen ranking bundle")
    payload, messages = bundle["model_payload"], bundle["messages"]
    if (not isinstance(payload, dict) or not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(row, dict) or set(row) != {"role", "content"}
                   or not isinstance(row["content"], str) for row in messages)
            or [row["role"] for row in messages] != ["system", "user"]):
        raise ValueError("Frozen ranking messages have an invalid structure")
    if json.loads(messages[1]["content"], object_pairs_hook=_pairs, parse_constant=invalid_constant) != payload:
        raise ValueError("Frozen user message does not match its model payload")
    identity = {"model_input_sha256": sha256_json(payload), "messages_sha256": sha256_json(messages),
                "presentation": bundle["presentation"]}
    if any(bundle[key] != identity[key] for key in ("model_input_sha256", "messages_sha256")):
        raise ValueError("Frozen ranking content failed hash verification")
    if bundle["snapshot_id"] != sha256_json(identity):
        raise ValueError("Frozen ranking snapshot identity is inconsistent")
    return bundle, decode_v2g_payload(payload)


def _label(key: Any) -> str:
    return _HEADERS.get(str(key), str(key).replace("_", " ").capitalize())


def _details(label: str, body: str, *, opened: bool = False) -> str:
    return f'<details{" open" if opened else ""}><summary>{_escape(label)}</summary><div class="details-body">{body}</div></details>'


def _scalar(value: Any) -> str:
    if value is None:
        return '<span class="empty">Not reported</span>'
    if isinstance(value, bool):
        return "true" if value else "false"
    return _escape(value)


def _fields(value: Mapping) -> str:
    if not value:
        return '<p class="empty">No fields supplied.</p>'
    return '<table class="kv"><tbody>' + "".join(
        f'<tr><th>{_escape(_label(key))}<code class="key">{_escape(key)}</code></th><td>{_structured(item)}</td></tr>'
        for key, item in value.items()
    ) + "</tbody></table>"


def _structured(value: Any) -> str:
    if isinstance(value, Mapping):
        if set(value) == {"columns", "constant_fields", "rows"}:
            columns, constants, rows = value["columns"], value["constant_fields"], value["rows"]
            if (not isinstance(columns, list) or not all(isinstance(key, str) for key in columns)
                    or len(columns) != len(set(columns)) or not isinstance(constants, Mapping)
                    or set(columns) & set(constants) or not isinstance(rows, list)
                    or any(not isinstance(row, list) or len(row) != len(columns) for row in rows)):
                raise ValueError("Supplied scientific table has invalid column bindings")
            restored = [dict(constants, **dict(zip(columns, row, strict=True))) for row in rows]
            return '<p class="source-note">Decoded supplied table; column and constant values are preserved.</p>' + _records(restored, category="")
        return _fields(value)
    if isinstance(value, list):
        if not value:
            return '<span class="empty">No entries supplied</span>'
        if all(isinstance(item, Mapping) for item in value):
            return _records(value, category="")
        return "<ol>" + "".join(f"<li>{_structured(item)}</li>" for item in value) + "</ol>"
    return _scalar(value)


def _records(rows: list, *, category: str) -> str:
    if not rows:
        return '<p class="empty">No retained rows supplied.</p>'
    if not all(isinstance(row, Mapping) for row in rows):
        return "<ol>" + "".join(f"<li>{_structured(row)}</li>" for row in rows) + "</ol>"
    known = list(dict.fromkeys(key for row in rows for key in row))
    preferred = _COLUMNS.get(category, tuple(known[:5]))
    derived = {"regulatory_interval"} if category == "e2g" else set()
    if category == "qtl" and "cs_lead_measurement" in known:
        derived.update(("lead_p_value", "lead_beta"))
    columns = [key for key in preferred if key in known or key in derived]
    extras = [key for key in known if key not in columns]
    has_source = bool(extras) or category == "e2g" and "score" in known
    headers = "".join(f'<th scope="col">{_escape(_label(key))}</th>' for key in columns)
    if has_source:
        headers += '<th scope="col">Source and other supplied fields</th>'
    body = []
    for index, row in enumerate(rows, 1):
        attrs = f' data-row-index="{index}"'
        if category:
            attrs += f' data-category="{_escape(category)}"'
        if "evidence_id" in row:
            attrs += f' data-evidence-id="{_escape(row["evidence_id"])}"'
        cells = []
        lead_display = _lead_qtl_display(row, {}) if category == "qtl" and derived else {}
        for key in columns:
            if key == "regulatory_interval":
                rendered = (_escape(f"chr{row['chromosome']}:{row['start']}-{row['end']}")
                            if all(row.get(name) is not None for name in ("chromosome", "start", "end"))
                            else '<span class="empty">Interval not fully supplied</span>')
            elif key in {"lead_p_value", "lead_beta"}:
                rendered = _scalar(lead_display.get("p_value" if key == "lead_p_value" else "beta"))
            elif key == "score" and category == "e2g" and type(row.get(key)) in (int, float):
                rendered = f'<span title="Supplied value: {_escape(row[key])}">{three_significant_figures(row[key])}</span>'
            else:
                rendered = _structured(row[key]) if key in row else '<span class="empty">Field not supplied</span>'
            cells.append("<td>" + rendered + "</td>")
        if has_source:
            detail = {key: row[key] for key in extras if key in row}
            if category == "e2g" and "score" in row:
                detail["score"] = row["score"]
            cells.append('<td class="record-source">' + _details("Source", _fields(detail)) + "</td>")
        body.append(f"<tr{attrs}>" + "".join(cells) + "</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + headers + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def _inline(text: str) -> str:
    # Provider or prompt HTML is always text. No images, active links, embedded
    # objects or raw HTML are accepted by this small offline Markdown subset.
    pieces = re.split(r"(`[^`\n]+`)", text)
    for index, piece in enumerate(pieces):
        if index % 2:
            pieces[index] = "<code>" + _escape(piece[1:-1]) + "</code>"
        else:
            escaped = _escape(piece)
            escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", escaped)
            pieces[index] = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", escaped)
    return "".join(pieces)


def _markdown(text: str) -> str:
    result, paragraph, code = [], [], []
    fence = None
    list_kind = None
    def flush():
        if paragraph:
            result.append("<p>" + "<br>".join(_inline(line) for line in paragraph) + "</p>")
            paragraph.clear()
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                result.append("<pre>" + _escape("\n".join(code)) + "</pre>")
                code.clear()
                fence = None
            else:
                code.append(line)
            continue
        if marker:
            flush()
            if list_kind:
                result.append(f"</{list_kind}>")
                list_kind = None
            fence = marker[1]
            continue
        heading = re.match(r"^\s*(#{1,6})\s+(.+)$", line)
        item = re.match(r"^\s*(?:([-*+])\s+|(\d+)[.)]\s+)(.+)$", line)
        if list_kind and not item:
            result.append(f"</{list_kind}>")
            list_kind = None
        if heading:
            flush()
            level = min(6, len(heading[1]) + 2)
            result.append(f"<h{level}>" + _inline(heading[2]) + f"</h{level}>")
        elif item:
            flush()
            kind = "ul" if item[1] else "ol"
            if kind != list_kind:
                if list_kind:
                    result.append(f"</{list_kind}>")
                start = f' start="{int(item[2])}"' if kind == "ol" else ""
                result.append(f"<{kind}{start}>")
                list_kind = kind
            result.append("<li>" + _inline(item[3]) + "</li>")
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
    flush()
    if list_kind:
        result.append(f"</{list_kind}>")
    if fence:
        result.append("<pre>" + _escape("\n".join(code)) + "</pre>")
    return '<div class="prompt">' + "".join(result) + "</div>"


def _snapshot(bundle: Mapping, raw: str) -> str:
    identity = {key: bundle[key] for key in ("snapshot_id", "model_input_sha256", "messages_sha256", "presentation")}
    encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    download = f'<a class="download" download="gwasmind-ranking-input.json" href="data:application/json;base64,{encoded}">Download original JSON</a>'
    exact = '<pre data-original-json="true">' + _escape(raw) + "</pre>"
    return download + _details("Snapshot identity and presentation order", _fields(identity)) + _details("Original frozen JSON", exact)


def _document(title: str, body: str) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; '
            'style-src &#39;unsafe-inline&#39;; base-uri &#39;none&#39;; form-action &#39;none&#39;">'
            f"<title>{_escape(title)} · GWASMind</title><style>{_CSS}</style></head><body><main>"
            f"<h1>{_escape(title)}</h1>{body}"
            '<footer>Created from the frozen ranking request. No new evidence, gene selection or ranking was computed. '
            'Not reported means a supplied null; Field not supplied means the field was absent. '
            'Neither means biological absence. This document requires no network connection.</footer></main></body></html>')


def _card(card: Mapping, index: int) -> str:
    gene_id = card.get("gene_id", "")
    label = card.get("symbol") or card.get("gene_symbol") or gene_id
    result = [f'<article id="gene-{index}" data-gene-id="{_escape(gene_id)}">',
              f"<h2>{_escape(label)} <span class=\"count\">{_escape(gene_id)}</span></h2>"]
    if card.get("name"):
        result.append('<p class="gene-meta">' + _escape(card["name"]) + "</p>")
    metadata = {key: value for key, value in card.items() if key not in {"gene_id", "symbol", "gene_symbol", "name", "evidence"}}
    if metadata:
        result.append(_details("Gene identity, position and supplied card metadata", _fields(metadata)))
    evidence = card.get("evidence")
    if not isinstance(evidence, Mapping):
        result.append(_fields({key: value for key, value in card.items() if key not in {"gene_id", "gene_symbol"}}))
    else:
        for category, section in evidence.items():
            if not isinstance(section, Mapping):
                result.append(f"<h3>{_escape(_CATEGORIES.get(category, _label(category)))}</h3>" + _structured(section))
                continue
            records = section.get("records")
            count = f' <span class="count">{len(records)} retained rows</span>' if isinstance(records, list) else ""
            result.append(f'<section data-evidence-category="{_escape(category)}" data-category="{_escape(category)}" data-gene-id="{_escape(gene_id)}"><h3>{_escape(_CATEGORIES.get(category, _label(category)))}{count}</h3>')
            if "status" in section:
                result.append('<p class="source-note">Retrieval status: ' + _scalar(section["status"]) + "</p>")
            if isinstance(records, list):
                if category == "e2g" and records:
                    result.append('<p class="source-note">Scores use three significant figures here; Source preserves every supplied digit.</p>')
                table = _records(records, category=category)
                result.append(_details(f"Show all {len(records)} retained rows", table)
                              if category in {"coloc", "e2g"} and len(records) > 10 else table)
            if category == "expression" and isinstance(section.get("context_matched"), list):
                result.append("<h4>Context measurements</h4>" + _records(section["context_matched"], category="expression"))
            remaining = {key: value for key, value in section.items()
                         if key not in {"records", "status"} and not (category == "expression" and key == "context_matched")}
            if remaining:
                if category in {"function", "constraint"}:
                    result.append(_fields(remaining))
                else:
                    result.append(_details("Scope, coverage and other supplied fields", _fields(remaining)))
            result.append("</section>")
    result.append("</article>")
    return "".join(result)


def _science(payload: Mapping) -> str:
    comparison = payload.get("gene_comparison", {})
    cards = comparison.get("comparison_cards", ()) if isinstance(comparison, Mapping) else ()
    counts = Counter()
    for card in cards:
        for category, section in card.get("evidence", {}).items():
            if isinstance(section, Mapping):
                counts[category] += len(section.get("records", ()))
    metrics = [("Candidate genes", len(cards)), *( (_CATEGORIES[key], counts[key]) for key in ("qtl", "coloc", "e2g") if key in counts)]
    overview = '<div class="overview">' + "".join(f'<div class="metric"><strong>{value}</strong><span>{_escape(label)}</span></div>' for label, value in metrics) + "</div>"
    nav = "<nav aria-label=\"Candidate genes\">" + "".join(f'<a href="#gene-{index}">{_escape(card.get("symbol") or card.get("gene_symbol") or card.get("gene_id"))}</a>' for index, card in enumerate(cards, 1)) + "</nav>"
    locus = payload.get("locus", {})
    context = locus.get("biological_context", {}) if isinstance(locus, Mapping) else {}
    summary = {key: locus[key] for key in ("trait", "source_trait", "locus_id") if key in locus}
    if isinstance(context, Mapping):
        for key in ("tissues", "cell_types"):
            if isinstance(context.get(key), list):
                summary[key] = "; ".join(str(value) for value in context[key])
    result = [overview, nav, "<h2>Selected locus and biological context</h2>", '<section data-shared-scope="locus">',
              _fields(summary)]
    if isinstance(locus.get("source_observations"), list):
        result.append("<h3>Selected GWAS variants</h3>" + _records(locus["source_observations"], category="source"))
    result.extend((_details("Complete supplied locus, phenotype scope and biosample identities", _structured(locus)), "</section>"))
    result.extend(_card(card, index) for index, card in enumerate(cards, 1))
    shared = {key: value for key, value in comparison.items() if key != "comparison_cards"} if isinstance(comparison, Mapping) else comparison
    result.append('<section data-shared-scope="gene_comparison"><h2>Shared scope and coverage</h2>'
                  + _details("Show supplied policies, retrieval coverage and all LD relationships", _structured(shared)) + "</section>")
    extra = {key: value for key, value in payload.items() if key not in {"locus", "gene_comparison"}}
    if extra:
        result.append('<section data-shared-scope="payload">' + _details("Other supplied scientific fields", _fields(extra)) + "</section>")
    return "".join(result)


def render_ranking_evidence_html(bundle_json: str) -> str:
    """Display every gene and retained observation in the frozen model input."""
    bundle, payload = _load(bundle_json)
    lead = ('<p class="lead">All candidate cards and retained rows from the frozen ranking input. '
            'Shared definitions are decoded into their original fields. Gene order is the frozen presentation order, '
            'not the LLM ranking. Source panels retain study identities, exact values and provenance supplied to the LLM.</p>')
    return _document("Gene evidence card", lead + _science(payload) + "<h2>Original input</h2>" + _snapshot(bundle, bundle_json))


def render_ranking_input_html(bundle_json: str) -> str:
    """Show the actual frozen roles and messages, with a decoded reading view."""
    bundle, payload = _load(bundle_json)
    result = [('<p class="lead">The exact system and user messages frozen before the ranking request. '
               'The reading view formats Markdown and decodes shared scientific tables; the original text and JSON below remain unchanged.</p>')]
    for index, message in enumerate(bundle["messages"], 1):
        role, content = message["role"], message["content"]
        result.append(f'<section class="message" data-message-index="{index}"><p class="message-role">Message {index} · {_escape(role)}</p>')
        result.append("<h2>System instructions</h2>" if role == "system" else "<h2>User message · scientific input</h2>")
        result.append(_markdown(content) if role == "system" else _science(payload))
        result.append(_details("Exact message text", f'<pre data-message-role="{role}">{_escape(content)}</pre>'))
        result.append("</section>")
    result.extend(("<h2>Original input</h2>", _snapshot(bundle, bundle_json)))
    return _document("LLM input", "".join(result))
