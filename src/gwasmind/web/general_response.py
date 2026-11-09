from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..configuration import parameter
from ..prompting import format_prompt, load_prompt

MAX_HISTORY_TURNS = parameter("web.general_history_turns")
MAX_HISTORY_ITEM_CHARS = parameter("web.general_history_item_chars")
MAX_HISTORY_TOTAL_CHARS = parameter("web.general_history_total_chars")
_LOGGER = logging.getLogger(__name__)


class NoRetrievalMode(StrEnum):
    """Supported response styles when the query planner requested no retrieval."""

    CASUAL_GREETING = "casual_greeting"
    SOFTWARE_GUIDANCE = "software_guidance"
    SCIENTIFIC_EXPLANATION = "scientific_explanation"


class NoRetrievalOutputError(ValueError):
    """A model response failed the bounded no-retrieval contract."""


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One user-visible history turn; system or tool roles are never admitted."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError("conversation role must be user or assistant")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("conversation content must be a non-empty string")
        if len(self.content) > MAX_HISTORY_ITEM_CHARS:
            raise ValueError("conversation item exceeds its character limit")


@dataclass(frozen=True, slots=True)
class NoRetrievalContext:
    """Authoritative input to the bounded no-retrieval response writer."""

    question: str
    mode: NoRetrievalMode
    selected_variant_count: int = 0
    authoritative_facts: tuple[str, ...] = ()
    conversation_history: tuple[ConversationTurn, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, NoRetrievalMode):
            raise ValueError("mode must be a NoRetrievalMode")
        question = str(self.question).strip()
        if not question:
            raise ValueError("question is required")
        if len(question) > parameter("web.chat_question_max_chars"):
            raise ValueError(f"question exceeds the {parameter('web.chat_question_max_chars'):,}-character limit")
        if self.selected_variant_count < 0:
            raise ValueError("selected_variant_count cannot be negative")
        if len(self.authoritative_facts) > 20:
            raise ValueError("at most 20 authoritative facts are allowed")
        if len(self.conversation_history) > MAX_HISTORY_TURNS:
            raise ValueError("conversation history exceeds the turn limit")
        if any(not isinstance(item, ConversationTurn) for item in self.conversation_history):
            raise ValueError("conversation history must contain ConversationTurn values")
        if sum(len(item.content) for item in self.conversation_history) > MAX_HISTORY_TOTAL_CHARS:
            raise ValueError("conversation history exceeds its total character budget")
        normalized: list[str] = []
        for fact in self.authoritative_facts:
            value = str(fact).strip()
            if not value:
                raise ValueError("authoritative facts cannot be empty")
            if len(value) > 1_000:
                raise ValueError("an authoritative fact exceeds the 1,000-character limit")
            if _REFERENCE_ARTIFACT.search(value):
                raise ValueError(
                    "authoritative no-retrieval facts cannot contain citations or URLs"
                )
            if self.selected_variant_count and _SELECTED_INTERPRETATION.search(value):
                raise ValueError(
                    "no-retrieval facts cannot interpret the active variant selection"
                )
            normalized.append(value)
        if len({_normalized_text(item) for item in normalized}) != len(normalized):
            raise ValueError("authoritative facts must be unique")

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode.value,
            "question": self.question.strip(),
            "authoritative_facts": list(self.authoritative_facts),
            "conversation_history": [
                {"role": item.role, "content": item.content}
                for item in self.conversation_history
            ],
            "external_retrieval_performed": False,
        }
        # A plot selection is irrelevant to a greeting or a concept definition.
        # Supplying its count made those answers drift into unsolicited locus
        # interpretation/disclaimers. Keep it internally for scope validation;
        # expose it only when explaining the application's selection controls.
        if self.mode is NoRetrievalMode.SOFTWARE_GUIDANCE:
            payload["selected_context"] = {
                "present": self.selected_variant_count > 0,
                "variant_count": self.selected_variant_count,
                "interpretation_permitted": False,
            }
        return payload


@dataclass(frozen=True, slots=True)
class DetailSection:
    heading: str
    points: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NoRetrievalDraft:
    mode: NoRetrievalMode
    message: str | None = None
    summary_markdown: str | None = None
    detail_sections: tuple[DetailSection, ...] = ()
    limitations_uncertainty_markdown: str | None = None


@dataclass(frozen=True, slots=True)
class NoRetrievalResult:
    mode: NoRetrievalMode
    rendered_markdown: str
    source: str
    validation_errors: tuple[str, ...] = ()

    @property
    def used_fallback(self) -> bool:
        return self.source == "deterministic_fallback"


NO_EXTERNAL_REFERENCES_STATEMENT = (
    "No external references were retrieved for this response."
)

_GREETING = re.compile(
    r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)|"
    r"\u4f60\u597d|\u60a8\u597d|\u55e8)[\s!,.?\u3002\uff01\uff1f]*$",
    re.IGNORECASE,
)
_SOFTWARE_GUIDANCE = re.compile(
    r"(?:\bgwasmind\b|\b(?:this|the)\s+(?:app|application|software|interface)\b|"
    r"\b(?:upload|export|settings?|tutorial|hide\s+panel|clear\s+data|"
    r"test\s+connection|lasso)\b|"
    r"\bhow\s+(?:do|can|should)\s+i\s+(?:upload|export|configure|set|use|clear|"
    r"select|hide|show|download)\b|\b(?:upload|export|settings?|tutorial|interface|"
    r"button|panel)\s+(?:feature|screen|page|dialog|option|workflow)\b|"
    r"(?:\u5982\u4f55|\u600e\u4e48|\u600e\u6837).{0,24}(?:"
    r"\u4e0a\u4f20|\u5bfc\u51fa|\u8bbe\u7f6e|\u4f7f\u7528|\u6e05\u9664|"
    r"\u9009\u62e9|\u9690\u85cf|\u663e\u793a|\u70b9\u51fb)|"
    r"(?:\u4e0a\u4f20|\u5bfc\u51fa|\u8bbe\u7f6e|\u6559\u7a0b|\u754c\u9762|"
    r"\u6309\u94ae|\u9762\u677f).{0,16}(?:\u529f\u80fd|\u95ee\u9898|"
    r"\u600e\u4e48|\u5982\u4f55))",
    re.IGNORECASE,
)
_REPORT_HEADING = re.compile(
    r"(?im)(?:^\s*#{1,6}\s+|\*\*(?:summary|details?|evidence|references|"
    r"limitations(?:\s*&\s*uncertainty)?)\*\*)"
)
_REFERENCE_ARTIFACT = re.compile(
    r"(?i)(?:https?://|www\.|\bpmid\s*:?\s*\d+|\bdoi\s*:?\s*10\.\d{4,9}/|"
    r"\[[0-9]+(?:\s*[-,]\s*[0-9]+)*\]|\[[^\]]+\]\([^\)]+\))"
)
_UNSUPPORTED_ATTRIBUTION = re.compile(
    r"(?i)\b(?:(?:according\s+to|evidence\s+from)\s+(?:(?:a|the|this)\s+)?"
    r"(?:study|paper|publication|database|researchers?)\b(?!\s+design)|"
    r"(?:published|reported|retrieved)\s+(?:in|by|from)\s+(?:(?:a|the|this)\s+)?"
    r"(?:study|paper|publication|database|literature|researchers?)\b|"
    r"(?:a|the|this)\s+database\s+"
    r"(?:shows?|reports?|contains?|returned|identified)|"
    r"(?:a\s+)?(?:study|paper|publication)\s+(?:shows?|found|reports?|demonstrates?))\b"
)
_NAMED_EXTERNAL_SOURCE = re.compile(
    r"(?i)\b(?:open\s+targets|gtex|pubmed|europe\s+pmc|ensembl|gnomad|"
    r"uniprot|gene\s+ontology)\b"
)
_SELECTED_INTERPRETATION = re.compile(
    r"(?i)\b(?:selected|current|active|this|your|these)\s+"
    r"(?:\d+\s+)?(?:snps?|variants?|locus|region)\b"
)
_USER_DATA_CONTEXT = re.compile(
    r"\b(?:uploaded|loaded|your|our|current|active)\b[^.;]{0,45}\b"
    r"(?:GWAS|data|dataset|results?|cohort|samples?|patients?|analysis)\b",
    re.I,
)
_EMPIRICAL_ATTRIBUTION = re.compile(
    r"\b(?:we|I|GWASMind)\s+(?:found|measured|observed|retrieved|detected|estimated)\b|"
    r"\b(?:studies|study|cohorts?|papers?|publications?|database|et\s+al\.?)\s+"
    r"(?:showed|found|reported|identified|measured|contains?|returned)\b|"
    r"\b(?:rs\d{3,}|ENSG\d+|(?:chr)?(?:\d{1,2}|X|Y)[_:]\d+[_:][ACGT]+[_:][ACGT]+)\b",
    re.I,
)
_NUMBER = re.compile(r"(?<![\w])\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w])", re.I)
_SHORT_FOLLOWUP = re.compile(
    r"^(?:why|how|what\s+about|and|then|does\s+it|can\s+it|is\s+it|"
    r"that|this|those|it|\u4e3a\u4ec0\u4e48|\u600e\u4e48|\u90a3|"
    r"\u7136\u540e|\u8fd9\u4e2a|\u53ef\u4ee5\u5417)\b|^(?:"
    r"\u4e3a\u4ec0\u4e48|\u600e\u4e48|\u90a3|\u7136\u540e|\u8fd9\u4e2a)",
    re.I,
)

_SOFTWARE_FACT_TOPICS: tuple[
    tuple[str, re.Pattern[str], tuple[str, ...]], ...
] = (
    (
        "upload",
        re.compile(
            r"\b(?:upload|file|csv|tsv|txt|summary statistics)\b|"
            r"\u4e0a\u4f20|\u6587\u4ef6",
            re.I,
        ),
        (
            "Upload GWAS Data accepts CSV, TSV, and TXT files.",
            "The browser processes an uploaded GWAS file before sending sampled plot "
            "data to the local GWASMind service.",
        ),
    ),
    (
        "tutorial",
        re.compile(
            r"\b(?:tutorial|help|guide|documentation)\b|"
            r"\u6559\u7a0b|\u5e2e\u52a9|\u6307\u5357",
            re.I,
        ),
        (
            "Tutorial opens an in-app modal and loads the bundled tutorial content when "
            "it is first opened.",
        ),
    ),
    (
        "settings",
        re.compile(
            r"\b(?:settings?|api|key|model|provider|connection|deepseek|openai)\b|"
            r"\u8bbe\u7f6e|\u5bc6\u94a5|\u6a21\u578b|\u8fde\u63a5",
            re.I,
        ),
        (
            "Settings accepts an API Base URL, API key, and model name.",
            "Save & Test checks the configured credentials and exact model "
            "availability before AI features are enabled.",
            "The API key is kept only in the local GWASMind Python process for the "
            "current session and is not written to browser storage, logs, run manifests, "
            "or project files.",
        ),
    ),
    (
        "selection",
        re.compile(
            r"\b(?:plot|manhattan|select|selection|lasso|shift|zoom|chromosome)\b|"
            r"\u66fc\u54c8\u987f|\u9009\u62e9|\u5957\u7d22|"
            r"\u7f29\u653e|\u67d3\u8272\u4f53",
            re.I,
        ),
        (
            ("Dragging on the Manhattan plot uses box selection by default; "
             "the toolbar also offers lasso selection."),
            "Holding Shift while selecting preserves multiple selected regions.",
            "The plot supports scrolling to zoom and double-clicking a chromosome to focus it.",
        ),
    ),
    (
        "export",
        re.compile(
            r"\b(?:export|png|svg|pdf|download|image)\b|"
            r"\u5bfc\u51fa|\u4e0b\u8f7d|\u56fe\u7247",
            re.I,
        ),
        (
            ("Download saves the current Manhattan plot as PNG, JPG, SVG, or PDF; "
             "PNG/JPG offer a DPI selector."),
        ),
    ),
    (
        "clear",
        re.compile(
            r"\b(?:clear|remove|reset|delete)\s+(?:data|dataset)\b|"
            r"\u6e05\u9664\u6570\u636e|\u5220\u9664\u6570\u636e",
            re.I,
        ),
        (
            "Clear Data asks for confirmation before clearing all loaded data.",
        ),
    ),
    (
        "panel",
        re.compile(
            r"\b(?:hide|show|toggle|collapse)\s+(?:the\s+)?panel\b|"
            r"\u9690\u85cf\u9762\u677f|\u663e\u793a\u9762\u677f",
            re.I,
        ),
        (
            "Hide Panel collapses the right panel and changes the control to Show Panel; "
            "Show Panel restores it.",
        ),
    ),
)


def bound_no_retrieval_history(
    history: Sequence[ConversationTurn | Mapping[str, Any]],
) -> tuple[ConversationTurn, ...]:
    """Keep the newest valid user/assistant turns within all prompt budgets."""

    selected: list[ConversationTurn] = []
    remaining = MAX_HISTORY_TOTAL_CHARS
    for raw in reversed(history):
        if len(selected) >= MAX_HISTORY_TURNS or remaining <= 0:
            break
        if isinstance(raw, ConversationTurn):
            role, content = raw.role, raw.content
        elif isinstance(raw, Mapping):
            role, content = raw.get("role"), raw.get("content")
        else:
            continue
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        text = text[: min(MAX_HISTORY_ITEM_CHARS, remaining)]
        if not text:
            continue
        selected.append(ConversationTurn(role=role, content=text))
        remaining -= len(text)
    return tuple(reversed(selected))


def authoritative_gwasmind_software_facts(
    question: str,
    conversation_history: Sequence[ConversationTurn | Mapping[str, Any]] = (),
) -> tuple[str, ...]:
    """Return only verified runtime UI facts relevant to one software question."""

    normalized = _normalized_text(question)
    if _is_short_followup(normalized):
        bounded = bound_no_retrieval_history(conversation_history)
        prior_users = [item.content for item in bounded if item.role == "user"]
        normalized = _normalized_text(" ".join((*prior_users[-2:], normalized)))
    matched: list[str] = []
    for _topic, pattern, facts in _SOFTWARE_FACT_TOPICS:
        if pattern.search(normalized):
            matched.extend(facts)
    if not matched and _SOFTWARE_GUIDANCE.search(normalized):
        # A broad app-help question receives a compact feature overview.
        matched.extend(facts[0] for _topic, _pattern, facts in _SOFTWARE_FACT_TOPICS)
    return tuple(dict.fromkeys(matched))


def classify_no_retrieval_mode(
    question: str,
    conversation_history: Sequence[ConversationTurn | Mapping[str, Any]] = (),
) -> NoRetrievalMode:
    """Classify the presentation style after retrieval has already been declined."""

    normalized = " ".join(
        unicodedata.normalize("NFKC", str(question)).casefold().split()
    )
    if _GREETING.fullmatch(normalized):
        return NoRetrievalMode.CASUAL_GREETING
    if _SOFTWARE_GUIDANCE.search(normalized):
        return NoRetrievalMode.SOFTWARE_GUIDANCE
    if _is_short_followup(normalized):
        bounded = bound_no_retrieval_history(conversation_history)
        if any(
            item.role == "user" and _SOFTWARE_GUIDANCE.search(_normalized_text(item.content))
            for item in bounded[-4:]
        ):
            return NoRetrievalMode.SOFTWARE_GUIDANCE
    return NoRetrievalMode.SCIENTIFIC_EXPLANATION


def no_retrieval_response_messages(
    context: NoRetrievalContext,
) -> list[dict[str, str]]:
    payload = json.dumps(
        context.to_payload(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": load_prompt("no_retrieval_response_system")},
        {
            "role": "user",
            "content": format_prompt("no_retrieval_response_user", payload=payload),
        },
    ]


def no_retrieval_repair_messages(
    context: NoRetrievalContext,
    *,
    invalid_output: str,
    validation_error: str,
) -> list[dict[str, str]]:
    payload = json.dumps(
        {
            "bounded_context": context.to_payload(),
            "validation_error": str(validation_error)[:1_000],
            "invalid_output": str(invalid_output)[:12_000],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {
            "role": "system",
            "content": load_prompt("no_retrieval_response_repair_system"),
        },
        {
            "role": "user",
            "content": format_prompt(
                "no_retrieval_response_repair_user",
                payload=payload,
            ),
        },
    ]


def parse_no_retrieval_output(
    content: str,
    context: NoRetrievalContext,
) -> NoRetrievalDraft:
    """Strictly parse and ground one no-retrieval model response."""

    value = _strict_json_object(content)
    if context.mode is NoRetrievalMode.CASUAL_GREETING:
        _require_exact_keys(value, {"message"})
        message = _bounded_text(value["message"], label="message", limit=320)
        _validate_generated_text(message, context)
        if "\n" in message:
            raise NoRetrievalOutputError("a casual greeting must be one paragraph")
        return NoRetrievalDraft(mode=context.mode, message=message)

    value = _normalize_equivalent_keys(value, {
        "summary": "summary_markdown", "details": "detail_sections",
        "limitations": "limitations_uncertainty_markdown",
        "limitations_uncertainty": "limitations_uncertainty_markdown",
    })
    _require_exact_keys(
        value,
        {
            "summary_markdown",
            "detail_sections",
            "limitations_uncertainty_markdown",
        },
    )
    summary = _bounded_text(
        value["summary_markdown"],
        label="summary_markdown",
        limit=1_500,
    )
    limitations = _bounded_text(
        value["limitations_uncertainty_markdown"],
        label="limitations_uncertainty_markdown",
        limit=2_000,
    )
    raw_sections = value["detail_sections"]
    if not isinstance(raw_sections, list) or not 1 <= len(raw_sections) <= 4:
        raise NoRetrievalOutputError("detail_sections must contain one to four sections")
    sections: list[DetailSection] = []
    seen_headings: set[str] = set()
    seen_points: set[str] = set()
    for index, raw_section in enumerate(raw_sections, start=1):
        if not isinstance(raw_section, dict):
            raise NoRetrievalOutputError(f"detail section {index} must be an object")
        raw_section = _normalize_equivalent_keys(raw_section, {
            "title": "heading", "bullets": "points", "items": "points",
        })
        _require_exact_keys(raw_section, {"heading", "points"})
        heading = _bounded_text(
            raw_section["heading"],
            label=f"detail section {index} heading",
            limit=100,
        )
        if _REPORT_HEADING.search(heading):
            raise NoRetrievalOutputError("detail headings cannot contain report headings")
        heading_key = _normalized_text(heading)
        if heading_key in seen_headings:
            raise NoRetrievalOutputError("detail section headings must be unique")
        seen_headings.add(heading_key)
        raw_points = raw_section["points"]
        if not isinstance(raw_points, list) or not 1 <= len(raw_points) <= 5:
            raise NoRetrievalOutputError(
                f"detail section {index} points must contain one to five items"
            )
        points: list[str] = []
        for point_index, raw_point in enumerate(raw_points, start=1):
            point = _bounded_text(
                raw_point,
                label=f"detail section {index} point {point_index}",
                limit=1_000,
            )
            point_key = _normalized_text(point)
            if point_key in seen_points:
                raise NoRetrievalOutputError("detail points must be non-duplicative")
            seen_points.add(point_key)
            points.append(point)
        sections.append(DetailSection(heading=heading, points=tuple(points)))

    generated = [summary, limitations]
    generated.extend(section.heading for section in sections)
    generated.extend(point for section in sections for point in section.points)
    for item in generated:
        _validate_generated_text(item, context)
    if context.mode is NoRetrievalMode.SOFTWARE_GUIDANCE and context.authoritative_facts:
        allowed_points = {
            _normalized_text(fact) for fact in context.authoritative_facts
        }
        for section in sections:
            for point in section.points:
                if _normalized_text(point) not in allowed_points:
                    raise NoRetrievalOutputError(
                        "software detail points must copy an authoritative fact exactly"
                    )
    if context.mode is NoRetrievalMode.SOFTWARE_GUIDANCE and not context.authoritative_facts:
        combined = " ".join(generated).casefold()
        if not any(
            phrase in combined
            for phrase in (
                "not supplied",
                "no authoritative",
                "cannot provide",
                "cannot confirm",
                "need more information",
            )
        ):
            raise NoRetrievalOutputError(
                "software guidance without authoritative facts must abstain from feature claims"
            )
    return NoRetrievalDraft(
        mode=context.mode,
        summary_markdown=summary,
        detail_sections=tuple(sections),
        limitations_uncertainty_markdown=limitations,
    )


def render_no_retrieval_response(draft: NoRetrievalDraft) -> str:
    """Render validated JSON into the consistent total-and-parts presentation."""

    if draft.mode is NoRetrievalMode.CASUAL_GREETING:
        return str(draft.message or "Hello! How can I help?").strip()
    parts = ["## **Summary**", "", str(draft.summary_markdown or "").strip(), "", "## **Details**"]
    for index, section in enumerate(draft.detail_sections, start=1):
        parts.extend(["", f"### {index}. {section.heading}", ""])
        parts.extend(f"- {point}" for point in section.points)
    parts.extend(
        [
            "",
            "## **Limitations & Uncertainty**",
            "",
            str(draft.limitations_uncertainty_markdown or "").strip(),
            "",
            "## **References**",
            "",
            NO_EXTERNAL_REFERENCES_STATEMENT,
        ]
    )
    return "\n".join(parts).strip()


def run_no_retrieval_response(
    context: NoRetrievalContext,
    completer: Callable[..., Any] | None,
) -> NoRetrievalResult:
    """Generate, repair once, then deterministically fall back on validation failure."""

    errors: list[str] = []
    invalid_output = ""
    if completer is not None:
        try:
            invalid_output = _complete_text(
                completer,
                no_retrieval_response_messages(context),
                max_tokens=(parameter("model.greeting_max_tokens") if context.mode is NoRetrievalMode.CASUAL_GREETING else parameter("model.general_response_max_tokens")),
            )
            draft = parse_no_retrieval_output(invalid_output, context)
            return NoRetrievalResult(
                mode=context.mode,
                rendered_markdown=render_no_retrieval_response(draft),
                source="model",
            )
        except Exception as exc:  # Model/provider failures also receive one bounded retry.
            errors.append("draft:" + _safe_error(exc))
            _log_response_failure("draft", exc)
        try:
            repaired_output = _complete_text(
                completer,
                no_retrieval_repair_messages(
                    context,
                    invalid_output=invalid_output,
                    validation_error=errors[-1],
                ),
                max_tokens=(parameter("model.greeting_max_tokens") if context.mode is NoRetrievalMode.CASUAL_GREETING else parameter("model.general_response_max_tokens")),
            )
            draft = parse_no_retrieval_output(repaired_output, context)
            return NoRetrievalResult(
                mode=context.mode,
                rendered_markdown=render_no_retrieval_response(draft),
                source="model_repaired",
                validation_errors=tuple(errors),
            )
        except Exception as exc:
            errors.append("repair:" + _safe_error(exc))
            _log_response_failure("repair", exc)

    fallback = _deterministic_fallback(context)
    return NoRetrievalResult(
        mode=context.mode,
        rendered_markdown=render_no_retrieval_response(fallback),
        source="deterministic_fallback",
        validation_errors=tuple(errors),
    )


def _deterministic_fallback(context: NoRetrievalContext) -> NoRetrievalDraft:
    if context.mode is NoRetrievalMode.CASUAL_GREETING:
        return NoRetrievalDraft(
            mode=context.mode,
            message="Hello! How can I help with GWASMind or a general GWAS question?",
        )
    if context.mode is NoRetrievalMode.SOFTWARE_GUIDANCE:
        if context.authoritative_facts:
            summary = "The available GWASMind guidance for this request is summarized below."
            points = context.authoritative_facts
            limitations = (
                "This answer is limited to the authoritative software guidance supplied "
                "to this session and was not checked against external documentation."
            )
        else:
            summary = (
                "I cannot provide reliable feature-specific instructions because no "
                "authoritative GWASMind guidance was supplied for this request."
            )
            points = ("Please specify the feature or action for which you need guidance.",)
            limitations = (
                "Application behavior was not inferred, and no external documentation "
                "was retrieved."
            )
        return NoRetrievalDraft(
            mode=context.mode,
            summary_markdown=summary,
            detail_sections=(DetailSection(heading="Guidance", points=tuple(points)),),
            limitations_uncertainty_markdown=limitations,
        )
    if context.authoritative_facts:
        summary = "The bounded scientific facts available for this question are summarized below."
        points = context.authoritative_facts
    else:
        summary = (
            "I could not produce a validated scientific explanation from the bounded "
            "no-retrieval response path."
        )
        points = (
            "Please rephrase the conceptual question or request an evidence-backed lookup.",
        )
    return NoRetrievalDraft(
        mode=context.mode,
        summary_markdown=summary,
        detail_sections=(DetailSection(heading="Explanation", points=tuple(points)),),
        limitations_uncertainty_markdown=(
            "This response does not interpret the active locus and was not checked against "
            "external databases or literature."
        ),
    )


def _validate_generated_text(text: str, context: NoRetrievalContext) -> None:
    # The runtime contract requires English product prose, even for a Chinese
    # question. Preserve mathematical Unicode and short quoted non-English terms;
    # a predominantly CJK response receives the existing single bounded repair.
    unquoted_prose = re.sub(
        r'`[^`\n]*`|"[^"\n]*"|“[^”]*”|「[^」]*」|『[^』]*』|《[^》]*》|'
        r"(?<!\w)'[^'\n]*'(?!\w)",
        "", text,
    )
    cjk_letters = len(re.findall(r"[\u3400-\u9fff]", unquoted_prose))
    latin_letters = len(re.findall(r"[A-Za-z]", unquoted_prose))
    if (
        cjk_letters >= 6 and cjk_letters > latin_letters
        and re.search(r"[。！？!?,，;；]", unquoted_prose)
    ):
        raise NoRetrievalOutputError("generated product prose must be in English")
    if _REPORT_HEADING.search(text):
        raise NoRetrievalOutputError("model text cannot contain application-rendered headings")
    reference_text = text
    if context.mode is NoRetrievalMode.SCIENTIFIC_EXPLANATION and _is_definition_context(text):
        # [0, 1] and [-1, 1] are mathematical intervals, not bibliography numbers.
        reference_text = re.sub(r"\[\s*-?[01](?:\.0)?\s*,\s*1(?:\.0)?\s*\]", "", text)
    if _REFERENCE_ARTIFACT.search(reference_text):
        raise NoRetrievalOutputError("external references or citation artifacts are forbidden")
    if _UNSUPPORTED_ATTRIBUTION.search(text):
        raise NoRetrievalOutputError("unsupported source attribution is forbidden")
    if (
        context.selected_variant_count or context.mode is NoRetrievalMode.SCIENTIFIC_EXPLANATION
    ) and _contains_selected_interpretation(text):
        raise NoRetrievalOutputError("the active variant selection cannot be interpreted here")
    prior_user_text = tuple(
        item.content for item in context.conversation_history if item.role == "user"
    )
    allowed_corpus = " ".join(
        (context.question, *context.authoritative_facts, *prior_user_text)
    ).casefold()
    for source in _NAMED_EXTERNAL_SOURCE.findall(text):
        if source.casefold() not in allowed_corpus:
            raise NoRetrievalOutputError("an unsupported named external source was introduced")
    if context.mode is NoRetrievalMode.SCIENTIFIC_EXPLANATION:
        if _EMPIRICAL_ATTRIBUTION.search(text):
            raise NoRetrievalOutputError("unretrieved empirical attribution is forbidden")
        # General scientific knowledge and its illustrative quantities are
        # generated under the conceptual prompt, not a retrieved-data ledger.
        # The program enforces scope/source integrity, not a list of permitted
        # scientific constants. Empirical locus findings remain forbidden above.
        return
    allowed_numbers = {float(value) for value in _NUMBER.findall(_numeric_text(allowed_corpus))}
    # The selection size is interface metadata, not an invented measurement.
    # The earlier locus-interpretation guard still rejects biological claims.
    allowed_numbers.add(float(context.selected_variant_count))
    generated_numbers = set(_NUMBER.findall(_numeric_text(text)))
    unsupported_numbers = {
        value for value in generated_numbers if float(value) not in allowed_numbers
    }
    if unsupported_numbers:
        raise NoRetrievalOutputError(
            "unsupported numeric claims were introduced: " + ", ".join(sorted(unsupported_numbers))
        )


def _is_definition_context(text: str) -> bool:
    """Identify a mathematical definition, not any sentence containing a number."""

    return bool(re.search(
        r"\b(?:p[ -]?values?|probabilit(?:y|ies)|correlation(?:\s+coefficients?)?|"
        r"odds\s+ratios?|risk\s+ratios?|hazard\s+ratios?|relative\s+risk)\b", text, re.I,
    ) and re.search(
        r"\b(?:ranges?|intervals?|between|bounded|from|lies?|equals?|"
        r"means?|indicates?|denotes?|null|no\s+(?:effect|association))\b", text, re.I,
    ))


def _numeric_text(text: str) -> str:
    """Compare equivalent decimal and scientific notation without changing prose."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold().replace("−", "-")
    return re.sub(
        r"(\d+(?:\.\d+)?)\s*(?:×|x|\*)\s*10\s*\^?\s*\{?([+-]?\d+)\}?",
        r"\1e\2", normalized,
    )


def _strict_json_object(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise NoRetrievalOutputError("model response is empty")
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        # Presentation wrappers do not change the schema. Prose, multiple JSON
        # objects, invalid syntax and unknown fields still fail normal validation.
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise NoRetrievalOutputError("model response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise NoRetrievalOutputError("model response must be one JSON object")
    return value


def _contains_selected_interpretation(text: str) -> bool:
    """Reject empirical locus claims, not a mere mention in a definition."""

    for sentence in re.split(r"(?<=[.!?])\s+|[;\n]+", str(text)):
        if not (_SELECTED_INTERPRETATION.search(sentence) or _USER_DATA_CONTEXT.search(sentence)):
            continue
        clean = re.sub(r"[*_]", "", sentence).strip()
        scope_disclaimer = re.search(
            r"\b(?:does?\s+not|cannot|can't|without|not|no)\s+(?:an?\s+)?"
            r"(?:interpret(?:ation|ing)?|assess(?:ment|ing)?|analy[sz](?:e|is|ing)|"
            r"evaluat(?:e|ion|ing)|infer(?:ring)?|claim(?:ing)?|review(?:ing)?|"
            r"draw(?:ing)?\s+conclusions?)\b",
            clean, re.IGNORECASE,
        )
        if scope_disclaimer and not re.search(
            r"\b(?:but|however|although|yet|nevertheless)\b", clean, re.IGNORECASE
        ):
            continue
        # Definitions routinely say "the P-value for these variants means ..."
        # or "your data would need ...". Mentioning a selection is not itself a
        # finding. Check for asserted biological conclusions or measurements;
        # named-variant and source-attribution checks remain separate.
        finding = (
            r"^[\s,:]*(?:its\s+)?(?:(?:is|are|was|were|shows?|demonstrates?)\s+"
            r"(?:(?:not|no|strongly|significantly|directly|clearly|a|an)\s+){0,3}"
            r"(?:significant|associated|linked|causal|causative|pathogenic|benign|"
            r"protective|damaging|association|signal)\b|"
            r"(?:causes?|confers?|regulates?|overlaps?)\b|"
            r"(?:has|have|contains?|includes?)\s+\d+\s+significant\b|"
            r"(?:(?:has|have)\s+)?(?:P(?:[ -]?values?)?|beta|H4|odds\s+ratio)\s*"
            r"(?:=|<=|>=|<|>|is|was|of|below|above)\s*\d)"
        )
        subjects = [
            *list(_SELECTED_INTERPRETATION.finditer(clean)),
            *list(_USER_DATA_CONTEXT.finditer(clean)),
        ]
        # The predicate must belong to that subject, not to a later generic
        # explanation of the null hypothesis elsewhere in the sentence.
        if any(re.match(finding, clean[subject.end():], re.IGNORECASE) for subject in subjects):
            return True
    return False


def _log_response_failure(stage: str, error: Exception) -> None:
    """Log fixed diagnostic categories, never model prose, reasoning or credentials."""

    if isinstance(error, NoRetrievalOutputError):
        message = str(error)
        category = next((name for phrase, name in (
            ("numeric", "unsupported_numeric_claim"),
            ("variant selection", "selected_locus_interpretation"),
            ("reference", "unretrieved_reference"),
            ("attribution", "unretrieved_attribution"),
            ("external source", "unsupported_named_source"),
            ("heading", "report_heading"),
            ("JSON", "json_format"),
            ("contract", "json_schema"),
            ("limit", "response_size"),
            ("authoritative fact", "software_grounding"),
            ("English", "response_language"),
        ) if phrase in message), "response_contract")
    else:
        category = "model_service_failure"
    _LOGGER.warning("No-retrieval response %s rejected: %s", stage, category)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str]) -> None:
    actual = set(value)
    if actual == expected:
        return
    raise NoRetrievalOutputError(
        "response fields do not match the contract; expected "
        + ", ".join(sorted(expected))
    )


def _normalize_equivalent_keys(
    value: Mapping[str, Any], aliases: Mapping[str, str],
) -> dict[str, Any]:
    """Accept unambiguous presentation aliases without changing content or schema types."""

    normalized: dict[str, Any] = {}
    for key, content in value.items():
        canonical = aliases.get(key, key)
        if canonical in normalized:
            raise NoRetrievalOutputError("response fields contain ambiguous aliases")
        normalized[canonical] = content
    return normalized


def _bounded_text(value: Any, *, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NoRetrievalOutputError(f"{label} must be a non-empty string")
    text = value.strip()
    if len(text) > limit:
        raise NoRetrievalOutputError(f"{label} exceeds the {limit}-character limit")
    return text


def _complete_text(
    completer: Callable[..., Any],
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> str:
    result = completer(messages, max_tokens=max_tokens)
    if isinstance(result, str):
        content = result
    elif isinstance(result, Mapping):
        content = result.get("content")
    else:
        content = getattr(result, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise NoRetrievalOutputError("model returned an empty response")
    return content.strip()


def _safe_error(error: Exception) -> str:
    if isinstance(error, NoRetrievalOutputError):
        return str(error).strip()[:500] or "Response validation failed."
    # SDK and transport exceptions can contain request headers, credentials or
    # raw provider bodies. Only our own bounded validation text is sent to a
    # repair request or retained in result diagnostics.
    return "The model service failed to provide a valid response."


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _is_short_followup(value: str) -> bool:
    return len(value) <= 160 and bool(_SHORT_FOLLOWUP.search(value))


__all__: Sequence[str] = (
    "ConversationTurn",
    "DetailSection",
    "MAX_HISTORY_ITEM_CHARS",
    "MAX_HISTORY_TOTAL_CHARS",
    "MAX_HISTORY_TURNS",
    "NO_EXTERNAL_REFERENCES_STATEMENT",
    "NoRetrievalContext",
    "NoRetrievalDraft",
    "NoRetrievalMode",
    "NoRetrievalOutputError",
    "NoRetrievalResult",
    "authoritative_gwasmind_software_facts",
    "bound_no_retrieval_history",
    "classify_no_retrieval_mode",
    "no_retrieval_repair_messages",
    "no_retrieval_response_messages",
    "parse_no_retrieval_output",
    "render_no_retrieval_response",
    "run_no_retrieval_response",
)
