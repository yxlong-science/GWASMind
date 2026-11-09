from __future__ import annotations

import html
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from ..configuration import parameter, public_runtime_parameters
from ..prompting import load_prompt

TEMPLATE_ROOT = Path(__file__).with_name("templates").resolve()

_ALLOWED_PAGES = frozenset(
    {
        "index.html",
        "app.html",
        "analysis.html",
        "tutorial.html",
        "contact.html",
    }
)
_EXTENDS = re.compile(r"\{%\s*extends\s+[\"']([^\"']+)[\"']\s*%\}")
_BLOCK = re.compile(
    r"\{%\s*block\s+([A-Za-z_][A-Za-z0-9_]*)\s*%\}(.*?)"
    r"\{%\s*endblock\s*%\}",
    re.DOTALL,
)
_STATIC_URL = re.compile(
    r"\{\{\s*url_for\([\"']static[\"']\s*,\s*"
    r"filename\s*=\s*[\"']([^\"']+)[\"']\s*\)\s*\}\}"
)
_MAIN_URL = re.compile(r"\{\{\s*url_for\([\"']main\.([A-Za-z_][A-Za-z0-9_]*)[\"']\)\s*\}\}")
_ASTHMA_VALUE = re.compile(
    r"\{%\s*if\s+metadata\s+and\s+metadata\.data_source\s*==\s*"
    r"[\"']asthma[\"']\s*%\}(.*?)\{%\s*endif\s*%\}",
    re.DOTALL,
)
_UNRESOLVED = re.compile(r"\{%|\{\{")
_SCRIPT_MARKER = re.compile(r"(?m)^[ \t]*<script\b")

_MAIN_PATHS = {
    "index": "/",
    "app": "/app",
    "analysis": "/analysis",
    "tutorial": "/tutorial",
    "contact": "/contact",
}


def render_page(name: str, *, data_source: str | None = None) -> bytes:
    """Render the GWASMind page templates without a Flask dependency.

    The original DOM and CSS remain the visual source of truth.  Only the small
    subset of Jinja syntax used by those templates is resolved here.
    """

    if name not in _ALLOWED_PAGES:
        raise ValueError("page template is not allowed")
    page = _composed_template(name)
    page = _STATIC_URL.sub(lambda match: f"/static/{match.group(1)}", page)
    page = _MAIN_URL.sub(lambda match: _main_path(match.group(1)), page)
    page = _ASTHMA_VALUE.sub(lambda match: match.group(1) if data_source == "asthma" else "", page)
    threshold = round(-math.log10(parameter("analysis.gwas_significance_p")), parameter("plot.threshold_decimal_places"))
    page = re.sub(
        r'(<input\b[^>]*\bid="threshold-(?:input|range)"[^>]*\bvalue=")[^"]+("[^>]*>)',
        lambda match: match.group(1) + str(threshold) + match.group(2), page,
    )
    page = re.sub(
        r'(<input\b[^>]*\bid="threshold-(?:input|range)"[^>]*\bmin=")[^"]+("[^>]*>)',
        lambda match: match.group(1) + str(min(1, threshold)) + match.group(2), page,
    )
    page = re.sub(
        r'(<input\b[^>]*\bid="threshold-(?:input|range)"[^>]*\bmax=")[^"]+("[^>]*>)',
        lambda match: match.group(1) + str(max(15, threshold)) + match.group(2), page,
    )
    default_concurrency = parameter("execution.auto_default_concurrent_loci")
    concurrency_options = ''.join(
        f'<option value="{number}"' + (' selected' if number == default_concurrency else '') + f'>{number}</option>'
        for number in range(1, parameter("execution.max_concurrent_runs") + 1)
    )
    page = re.sub(r'(<select\b[^>]*\bid="analysis-concurrency"[^>]*>).*?(</select>)',
                  lambda match: match.group(1) + concurrency_options + match.group(2), page, flags=re.S)
    page = page.replace(
        "{{ content|safe }}",
        '<div id="tutorial-markdown" aria-live="polite"></div>',
    )
    ui_prompts = html.escape(
        json.dumps(
            {
                "describe_selected": load_prompt("ui_describe_selected"),
                "analysis_summary": load_prompt("ui_analysis_summary"),
                "analysis_regulatory": load_prompt("ui_analysis_regulatory"),
                "analysis_function": load_prompt("ui_analysis_function"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        quote=True,
    )
    prompt_bootstrap = (
        '    <meta id="gwasmind-runtime-parameters" data-parameters="'
        + html.escape(json.dumps(public_runtime_parameters(), ensure_ascii=False, separators=(",", ":")), quote=True)
        + '">\n'
        '    <script src="/static/js/runtime-config.js?v=1"></script>\n'
        '    <meta id="gwasmind-ui-prompts" data-prompts="' + ui_prompts + '">\n'
        '    <script src="/static/js/prompt-catalog.js?v=1"></script>\n'
    )
    # Head scripts can have different indentation from footer scripts. Load
    # configuration before the first consumer, including the upload processor.
    page = _SCRIPT_MARKER.sub(lambda match: prompt_bootstrap + match.group(), page, count=1)
    if _UNRESOLVED.search(page):
        raise RuntimeError(f"unresolved template expression in {name}")
    return page.encode("utf-8")


def _main_path(endpoint: str) -> str:
    try:
        return _MAIN_PATHS[endpoint]
    except KeyError as exc:
        raise RuntimeError(f"unsupported main endpoint in UI template: {endpoint}") from exc


@lru_cache(maxsize=None)
def _composed_template(name: str) -> str:
    target = (TEMPLATE_ROOT / name).resolve()
    if TEMPLATE_ROOT not in target.parents or not target.is_file():
        raise RuntimeError(f"UI template is missing: {name}")
    child = target.read_text(encoding="utf-8")
    extends = _EXTENDS.search(child)
    if extends is None:
        return child

    base_name = extends.group(1)
    base_target = (TEMPLATE_ROOT / base_name).resolve()
    if TEMPLATE_ROOT not in base_target.parents or not base_target.is_file():
        raise RuntimeError(f"base UI template is missing: {base_name}")
    base = base_target.read_text(encoding="utf-8")
    child_blocks = {match.group(1): match.group(2) for match in _BLOCK.finditer(child)}

    def replace_block(match: re.Match[str]) -> str:
        return child_blocks.get(match.group(1), match.group(2))

    return _BLOCK.sub(replace_block, base)
