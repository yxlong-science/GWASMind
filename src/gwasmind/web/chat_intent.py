from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, replace
from enum import StrEnum

from ..biological_context import phenotype_from_question
from ..configuration import parameter
from ..models import QTL_STUDY_TYPES, SINGLE_CELL_QTL_STUDY_TYPES, EvidenceKind
from ..novelty import is_prior_report_question, strip_report_population_suffix
from ..prompting import format_prompt, load_prompt


class ChatIntent(StrEnum):
    """The bounded question types supported by the local chat bridge."""

    GENERAL = "general"
    VARIANT_DESCRIPTION = "describe_variants"
    GENE_ASSOCIATIONS = "gene_associations"
    GENE_PRIORITIZATION = "prioritize_genes"
    TARGET_ASSESSMENT = "target_assessment"
    DISEASE_ASSOCIATIONS = "disease_associations"
    DRUG_REPURPOSING = "drug_repurposing"

    @property
    def label(self) -> str:
        return {
            self.GENERAL: "general conversation",
            self.VARIANT_DESCRIPTION: "variant description",
            self.GENE_ASSOCIATIONS: "variant-to-gene associations",
            self.GENE_PRIORITIZATION: "gene prioritization",
            self.TARGET_ASSESSMENT: "target assessment",
            self.DISEASE_ASSOCIATIONS: "disease associations",
            self.DRUG_REPURPOSING: "drug repurposing",
        }[self]

    @property
    def retrieval_bundle(self) -> tuple[str, ...]:
        """The predefined evidence domains available to this high-level route."""

        return {
            self.GENERAL: (),
            self.VARIANT_DESCRIPTION: (
                "variant",
                "credible_set",
                "study",
                "fine_mapping",
            ),
            self.GENE_ASSOCIATIONS: (
                "variant",
                "l2g",
                "molecular_qtl",
                "colocalisation",
                "regulatory",
                "expression",
                "function",
                "pathway",
                "literature",
            ),
            self.GENE_PRIORITIZATION: (
                "l2g",
                "molecular_qtl",
                "colocalisation",
                "regulatory",
                "expression",
                "function",
                "pathway",
                "literature",
            ),
            self.TARGET_ASSESSMENT: (
                "target_profile",
                "expression",
                "function",
                "pathway",
                "constraint",
                "tractability",
                "safety",
                "interaction",
                "phenotype",
            ),
            self.DISEASE_ASSOCIATIONS: ("target_disease", "disease_profile"),
            self.DRUG_REPURPOSING: (
                "known_drug",
                "clinical_candidate",
                "indication",
                "mechanism",
                "safety",
                "pharmacogenomics",
            ),
        }[self]


@dataclass(frozen=True, slots=True)
class ChatRoute:
    """Validated chat route plus an optional native QTL subtype filter."""

    intent: ChatIntent
    qtl_types: tuple[str, ...] = ()
    needs_retrieval: bool = True
    phenotype: str | None = None
    context_tissues: tuple[str, ...] = ()
    context_cell_types: tuple[str, ...] = ()
    evidence_kinds: tuple[str, ...] = ()
    planning_source: str = "deterministic_fallback"


class QTLRoutingError(ValueError):
    """A question named a QTL subtype outside the explicit whitelist."""


_DRUG = re.compile(
    r"(?:\b(?:drugs?|medicines?|compounds?|repurpos(?:e|ing)|"
    r"therap(?:y|ies|eutic)|treatments?|clinical\s+candidates?|"
    r"approved\s+(?:drugs?|therapies)|drug\s+indications?)\b)",
    re.IGNORECASE,
)
_DISEASE = re.compile(
    r"(?:\b(?:disease|phenotype|trait)\s+associations?\b|"
    r"\bassociated\s+(?:diseases?|phenotypes?|traits?)\b|"
    r"\bwhich\s+(?:diseases?|phenotypes?|traits?)\b|\bcomorbidit(?:y|ies)\b)",
    re.IGNORECASE,
)
_TARGET = re.compile(
    r"(?:\btarget\s+(?:assessment|profile|safety|tractability|function|"
    r"expression|mechanism)\b|\b(?:assess|evaluate|characteri[sz]e)\s+"
    r"(?:the\s+)?target\b|"
    r"\b(?:druggability|tractability|safety|constraint|expression|function|"
    r"pathway|mechanism)\b)",
    re.IGNORECASE,
)
_GENE_PRIORITIZATION = re.compile(
    r"(?:\b(?:prioriti[sz](?:e|ed|ing)|rank(?:ed|ing)?|infer(?:red|ring)?|"
    r"nominate(?:d|ing)?)\s+(?:the\s+)?(?:candidate\s+|causal\s+|effector\s+|"
    r"target\s+)?genes?\b|"
    r"\bidentify\s+(?:the\s+)?(?:candidate|causal|effector|target)\s+genes?\b|"
    r"\b(?:candidate|causal|effector|target)\s+genes?\b|"
    r"\bgenes?\s+(?:(?:(?:should|can|could|must)\s+be|"
    r"(?:is|are|was|were))\s+)?"
    r"(?:prioriti[sz](?:ed|ation)|rank(?:ed|ing))\b)",
    re.IGNORECASE,
)
_GENE_ASSOCIATION = re.compile(
    r"(?:\b(?:which|what)\s+genes?\b.{0,100}\b"
    r"(?:associate(?:d|s|ion)?|link(?:ed|s|age)?|regulat(?:e|ed|es|ion)|"
    r"affect(?:ed|s)?|implicat(?:e|ed|es|ion))\b|"
    r"\bgenes?\s+(?:(?:is|are|was|were)\s+)?"
    r"(?:associated|linked|regulated|affected|implicated)\b.{0,100}\b"
    r"(?:snps?|variants?|selection|locus|region)\b|"
    r"\b(?:snps?|variants?)\b.{0,100}\b"
    r"(?:gene\s+associations?|associated|linked|regulat(?:e|ed|es)|affect(?:ed|s)?)\b)",
    re.IGNORECASE,
)
_GENE_EVIDENCE_ONLY = re.compile(
    r"(?:\b(?:show|list|report|inspect|retrieve)\b.{0,100}\b"
    r"(?:variant[- ]to[- ]gene|gene\s+links?|molecular\s+qtls?|"
    r"colocali[sz]ation|regulatory|enhancer)\b|"
    r"\b(?:without|no)\s+(?:a\s+)?(?:ranking|prioriti[sz]ation)\b|"
    r"\bonly\b.{0,60}\b(?:qtls?|links?|regulatory|colocali[sz]ation)\b)",
    re.IGNORECASE,
)
_DESCRIBE = re.compile(
    r"(?:\b(?:describe|summari[sz]e|characteri[sz]e|review)\b.{0,80}"
    r"\b(?:snps?|variants?|selection|locus|region)\b|"
    r"\b(?:what|which)\s+(?:traits?|phenotypes?|associations?)\b.{0,80}"
    r"\b(?:snps?|variants?|selection|locus|region)\b|"
    r"\boverview\b.{0,80}\b(?:snps?|variants?|selection|locus|region)\b)",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)|"
    r"\u4f60\u597d|\u60a8\u597d|\u55e8)[\s!,.?\u3002\uff01\uff1f]*$",
    re.IGNORECASE,
)
_GENERAL_KNOWLEDGE = re.compile(
    r"^(?:what\s+(?:is|are)|how\s+(?:does|do|is|are)|define|explain|"
    r"tell\s+me\s+about)\b",
    re.IGNORECASE,
)
_SELECTION_REFERENCE = re.compile(
    r"\b(?:this|these|selected|current|my|our|above|here)\b",
    re.IGNORECASE,
)
_VARIANT_LOOKUP_SUBJECT = re.compile(
    r"\b(?:(?:this|that|these|those|selected|current|my|our)\s+(?:selected\s+)?(?:\d+\s+)?"
    r"(?:snps?|variants?)|(?:this|that|the|selected|current|my|our)\s+(?:[\w'-]+\s+){0,4}(?:locus|loci|region|selection)|rs\d+)\b",
    re.IGNORECASE,
)
_REPORTED_EVIDENCE_REQUEST = re.compile(
    r"(?:\b(?:have|has|are|were|is|was)\b.{0,160}\b"
    r"(?:reports?|reported|published|studies|studied|publications?|associated|linked)\b|"
    r"\b(?:find|search|retrieve|look\s+up|check|show)\b.{0,160}\b"
    r"(?:reports?|reported|studies|publications?|associations?)\b|"
    r"^\s*(?:any|what|which)\b.{0,100}\b(?:reports?|studies|publications?)\b)",
    re.IGNORECASE,
)
_NO_LOOKUP_REQUEST = re.compile(
    r"\b(?:do\s+not|don't|without)\s+(?:any\s+)?"
    r"(?:search(?:ing)?|retriev(?:e|al|ing)|quer(?:y|ying)|look(?:ing)?\s+up)\b|"
    r"(?:\u4e0d\u8981|\u65e0\u9700|\u4e0d\u7528|\u4e0d\u9700\u8981).{0,3}(?:\u68c0\u7d22|\u641c\u7d22|\u67e5\u8be2)|\u4e0d\u67e5(?:\u8be2|\u6587\u732e|\u6570\u636e\u5e93)",
    re.IGNORECASE,
)
_GENOMIC_LOOKUP_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9])rs\d+(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])(?:chr)?(?:[1-9]|1\d|2[0-2]|X|Y|MT)[_:]\d+"
    r"(?:[_:][ACGT]+[_:][ACGT]+|[-–:]\d+)(?![A-Za-z0-9])|"
    r"\bchr(?:[1-9]|1\d|2[0-2]|X|Y|MT):\d+(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CHINESE_LOOKUP_SUBJECT = re.compile(
    r"(?:\u8fd9(?:\u4e2a|\u4e9b)?|\u8be5|\u5f53\u524d|\u6240\u9009|\u9009\u4e2d).{0,24}(?:\u4f4d\u70b9|\u533a\u57df|\u53d8\u5f02|SNP)|"
    r"(?:\u4f4d\u70b9|\u533a\u57df|\u53d8\u5f02|SNP).{0,8}(?:\u9009\u4e2d|\u6240\u9009)", re.IGNORECASE,
)
_LOCUS_STATUS_REQUEST = re.compile(
    r"\b(?:new|novel|novelty|known|unreported|previously\s+reported|already\s+reported)\b|"
    r"(?:\u65b0(?:\u7684)?(?:\u4f4d\u70b9|\u4fe1\u53f7|\u5173\u8054)|\u5df2\u77e5|\u5df2\u6709|\u65e2\u5f80|\u4ee5\u524d|\u4e4b\u524d|\u66fe\u7ecf|\u62a5\u9053\u8fc7|\u62a5\u544a\u8fc7|\u53d1\u8868\u8fc7)|"
    r"(?:\u662f\u5426|\u662f\u4e0d\u662f|\u7b97\u4e0d\u7b97|\u7b97\u662f|\u662f).{0,12}\u65b0(?:\u7684)?(?:\u5417|\u4e48|\u561b|[?？]|$)", re.IGNORECASE,
)
_CHINESE_REPORTED_REQUEST = re.compile(
    r"(?:\u662f\u5426|\u6709\u6ca1\u6709|\u6709\u65e0|\u67e5|\u68c0\u7d22|\u641c\u7d22|\u627e|\u5df2\u6709|\u4ee5\u524d|\u4e4b\u524d|\u66fe\u7ecf).{0,45}(?:\u62a5\u9053|\u62a5\u544a|\u6587\u732e|\u7814\u7a76|\u5173\u8054)|"
    r"(?:\u4f4d\u70b9|\u533a\u57df|\u53d8\u5f02|SNP|rs\d+).{0,50}(?:\u76f8\u5173|\u5173\u8054|\u6709\u5173)(?:\u5417|\u4e48|[?？])", re.I,
)
_LOCUS_CONCEPT_REQUEST = re.compile(
    r"^(?:what\s+(?:is|are)\s+(?:a|an|the)\s+(?:new|novel|known)\s+(?:locus|loci)|"
    r"what\s+does\b.{0,50}\b(?:new|novel|known)\s+locus\b.{0,25}\bmean|"
    r"(?:define|explain)\s+(?:the\s+)?(?:concept|meaning|definition)\b|"
    r"is\s+this\s+(?:the|a)\s+(?:definition|meaning|concept)\b|"
    r"(?:\u8bf7)?(?:\u89e3\u91ca(?:\u4e00\u4e0b)?|\u8bf4\u660e)(?:\u4ec0\u4e48\u662f|\u4f55\u4e3a|\u65b0\u4f4d\u70b9(?:\u7684)?(?:\u6982\u5ff5|\u542b\u4e49|\u5b9a\u4e49))|"
    r"\u4ec0\u4e48(?:\u662f|\u53eb|\u53eb\u505a)|\u5982\u4f55\u5b9a\u4e49|\u600e\u6837\u5b9a\u4e49|(?:\u65b0|\u5df2\u77e5)\u4f4d\u70b9(?:\u7684)?(?:\u5b9a\u4e49|\u542b\u4e49|\u6982\u5ff5|\u662f\u4ec0\u4e48\u610f\u601d|\u662f\u4ec0\u4e48))", re.IGNORECASE,
)


def is_explicit_variant_evidence_lookup(
    question: str, *, selected_variant_count: int = 0, has_selected_region: bool = False,
) -> bool:
    """Recognize explicit locus-report requests, not merely a selected plot."""

    normalized = " ".join(unicodedata.normalize("NFKC", str(question)).split())
    active_selection = (
        isinstance(selected_variant_count, int) and not isinstance(selected_variant_count, bool)
        and selected_variant_count > 0
    ) or has_selected_region is True
    subject = (
        _VARIANT_LOOKUP_SUBJECT.search(normalized) or _GENOMIC_LOOKUP_REFERENCE.search(normalized)
        or _CHINESE_LOOKUP_SUBJECT.search(normalized)
        or active_selection and re.search(r"\b(?:this|these|it|they)\b|(?:\u8fd9\u4e2a|\u8fd9\u4e9b|\u5b83\u4eec|\u8fd9|\u5b83)", normalized, re.I)
    )
    return bool(
        subject
        and (_REPORTED_EVIDENCE_REQUEST.search(normalized) or _LOCUS_STATUS_REQUEST.search(normalized)
             or _CHINESE_REPORTED_REQUEST.search(normalized) or is_prior_report_question(normalized))
        and not _NO_LOOKUP_REQUEST.search(normalized)
        and not _LOCUS_CONCEPT_REQUEST.search(normalized)
    )


def _phenotype_lookup_clauses(question: str) -> tuple[str, ...]:
    """Keep lookup sentences separate from instructions, preserving qualifiers."""
    clauses = []
    start = 0
    brackets: list[str] = []
    closing = {")": "(", "]": "["}
    for index, character in enumerate(question):
        if character in "([":
            brackets.append(character)
        elif character in closing and brackets and brackets[-1] == closing[character]:
            brackets.pop()
        decimal_point = (character == "." and index > 0 and index + 1 < len(question)
                         and question[index - 1].isdigit() and question[index + 1].isdigit())
        if not brackets and (character in "?!;\r\n。？！；" or character == "." and not decimal_point):
            if clause := question[start:index].strip():
                clauses.append(clause)
            start = index + 1
    if clause := question[start:].strip():
        clauses.append(clause)
    return tuple(clauses)


def _explicit_lookup_phenotype(question: str) -> str | None:
    """Keep an explicitly scoped phenotype phrase without guessing a disease name."""

    patterns = (
        (
            r"\breports?\s+(?:related\s+to|about|on|for)\s+(.+?)\s+"
            r"(?:for|among|in|of)\s+(?:these|those|the\s+selected|selected)\s+(?:snps?|variants?)\b"
        ),
        r"\b(?:associated|linked|association)\s+with\s+(.+?)$",
        r"\b(?:GWAS|studies)\s+(?:of|for|on)\s+(.+?)$",
        r"\b(?:new|novel|known|previously\s+reported)\s+(.+?)(?:[- ]associated)?\s+(?:locus|loci|variants?|associations?|signals?)\b",
        r"\b(?:for|in)\s+(.+?)(?:\s+(?:GWAS|studies|literature|before|previously|already))?[?!.]*$",
        r"(?:\u662f\u4e0d\u662f|\u662f\u5426\u662f|\u7b97\u4e0d\u7b97|\u662f\u5426\u4e3a|\u662f|\u4e3a|\u5c5e\u4e8e)(?:\u4e00\u4e2a)?\s*(.+?)(?:\u7684)?(?:\u65b0|\u5df2\u77e5)(?:\u4f4d\u70b9|\u5173\u8054|\u4fe1\u53f7)",
        r"(?:\u65b0(?:\u7684)?|\u5df2\u77e5(?:\u7684)?)\s*(.+?)(?:\u4f4d\u70b9|\u5173\u8054|\u4fe1\u53f7)",
        r"(?:\u5bf9\u4e8e|\u9488\u5bf9|\u5173\u4e8e)\s*(.+?)(?:\u6765\u8bf4|\u800c\u8a00|[，,]|\u662f\u5426|\u7684)",
        r"(?:\u5728)\s*(.+?)(?:\u4e2d|\u76f8\u5173\u7814\u7a76|GWAS)",
        r"(?:\u4e0e|\u548c|\u8ddf)\s*(.+?)(?:\u6709\u5173|\u76f8\u5173|\u5173\u8054)",
        r"(?:\u6709\u6ca1\u6709|\u662f\u5426\u6709|\u6709\u65e0)\s*(.+?)(?:\u76f8\u5173|\u7684)?(?:\u65e2\u5f80)?(?:\u62a5\u9053|\u62a5\u544a|\u6587\u732e)",
    )
    clauses = _phenotype_lookup_clauses(question)
    for pattern in patterns:
        for clause in clauses:
            if (re.match(r"(?:please\s+)?(?:use|using|select|choose|apply|include|prefer)\b", clause, re.I)
                    and re.search(r"\bLD\b|\breference\s+(?:panel|population)\b", clause, re.I)
                    and not (_VARIANT_LOOKUP_SUBJECT.search(clause) or _GENOMIC_LOOKUP_REFERENCE.search(clause)
                             or _CHINESE_LOOKUP_SUBJECT.search(clause))):
                continue
            match = re.search(pattern, clause, flags=re.IGNORECASE)
            if not match:
                continue
            phenotype = " ".join(match.group(1).split()).strip(" \"'，,。？?")
            # A clear LD instruction may follow the phenotype in the same
            # sentence. Clinical qualifiers do not contain this command form.
            instruction = re.search(
                r"(?:,\s*|\s+)(?:(?:and|then)\s+)?(?:please\s+)?"
                r"(?:use|using|include|apply|select|choose|prefer)\s+(?:[\w-]+\s+){0,5}"
                r"(?:LD\b|reference\s+(?:panel|population)\b)", phenotype, re.I,
            )
            if instruction:
                phenotype = phenotype[:instruction.start()].rstrip(" ,")
            phenotype = re.sub(r"\s+(?:before|previously|already)$", "", phenotype, flags=re.I)
            phenotype = strip_report_population_suffix(phenotype)
            if (phenotype and len(phenotype) <= 160 and not _VARIANT_LOOKUP_SUBJECT.search(phenotype)
                    and not _GENOMIC_LOOKUP_REFERENCE.search(phenotype)
                    and not re.fullmatch(r"(?:a|an|the|this|these|this disease|this trait|the current trait|\u65b0\u7684|\u5df2\u77e5\u7684)", phenotype, re.I)):
                return {"\u54ee\u5598": "asthma", "1\u578b\u7cd6\u5c3f\u75c5": "type 1 diabetes", "2\u578b\u7cd6\u5c3f\u75c5": "type 2 diabetes",
                        "\u963f\u5c14\u8328\u6d77\u9ed8\u75c5": "Alzheimer's disease"}.get(phenotype, phenotype)
    return phenotype_from_question(question, fallback="") or None


def reconcile_chat_route(
    question: str, route: ChatRoute, *, selected_variant_count: int = 0,
    has_selected_region: bool = False, source_trait: str | None = None,
) -> ChatRoute:
    """Enforce explicit request scope after the LLM has planned.

    The UI's unqualified Describe action is cross-phenotype. A source trait is
    an observation label, not an implicit instruction to filter that answer.
    Only complete generic requests match this guard: adding a phenotype or a
    follow-up constraint leaves the model's interpretation untouched.
    """

    normalized = " ".join(unicodedata.normalize("NFKC", str(question)).split())
    generic_description = re.fullmatch(
        r"(?:please\s+)?(?:describe|summari[sz]e|characteri[sz]e|review)\s+"
        r"(?:(?:all|the|this|these|those|my|our|user[- ]selected)\s+)*"
        r"(?:selected\s+)?(?:\d+\s+)?(?:snps?|variants?|selection|locus|region)"
        r"(?:\s+please)?[.!?]*",
        normalized, re.IGNORECASE,
    )
    generic_description = generic_description or re.fullmatch(
        r"(?:what|which)\s+(?:traits?|phenotypes?|associations?)\s+"
        r"(?:are\s+)?(?:associated\s+with|reported\s+for|linked\s+to|for)\s+"
        r"(?:(?:all|the|these|those|selected|my|our)\s+)*"
        r"(?:\d+\s+)?(?:snps?|variants?)[.!?]*",
        normalized, re.IGNORECASE,
    )
    if generic_description and route.intent is ChatIntent.VARIANT_DESCRIPTION:
        return replace(route, phenotype=None, context_tissues=(), context_cell_types=())

    if not is_explicit_variant_evidence_lookup(
        question, selected_variant_count=selected_variant_count, has_selected_region=has_selected_region,
    ):
        return route
    status_lookup = bool(_LOCUS_STATUS_REQUEST.search(normalized) or is_prior_report_question(normalized))
    richer_request = bool(_GENE_PRIORITIZATION.search(normalized) or re.search(r"\bgenes?\b|\u57fa\u56e0", normalized, re.I))
    if route.needs_retrieval and route.intent is not ChatIntent.VARIANT_DESCRIPTION and (not status_lookup or richer_request):
        return route
    phenotype = _explicit_lookup_phenotype(question) or route.phenotype
    if not phenotype and status_lookup and (selected_variant_count or has_selected_region):
        candidate = str(source_trait or "").strip()
        if candidate.casefold() not in {"", "loaded phenotype", "selected locus", "current trait", "user-selected gwas locus"}:
            phenotype = candidate
    if route.intent is ChatIntent.VARIANT_DESCRIPTION and route.needs_retrieval and phenotype == route.phenotype:
        return route
    return ChatRoute(
        intent=ChatIntent.VARIANT_DESCRIPTION,
        needs_retrieval=True,
        phenotype=phenotype,
        planning_source="explicit_lookup_consistency_fallback",
    )


# An earlier implementation established the native Open Targets study-type set,
# but delegated selection to an LLM. Here the same semantics are encoded as
# a deliberately small English whitelist. Specific single-cell aliases are
# evaluated before their bulk counterparts so ``single-cell eQTL`` can never
# silently broaden to both ``sceqtl`` and ``eqtl``.
_QTL_ALIASES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "sceqtl",
        re.compile(
            r"(?:(?<![a-z0-9])sc[- ]?eqtls?(?![a-z0-9])|"
            r"\b(?:single[- ]cell|cell[- ]type[- ]specific)\s+"
            r"(?:e|(?:gene\s+)?expression\s+)qtls?\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "scpqtl",
        re.compile(
            r"(?:(?<![a-z0-9])sc[- ]?pqtls?(?![a-z0-9])|"
            r"\b(?:single[- ]cell|cell[- ]type[- ]specific)\s+"
            r"(?:p|protein(?:\s+(?:abundance|level))?\s+)qtls?\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "scsqtl",
        re.compile(
            r"(?:(?<![a-z0-9])sc[- ]?sqtls?(?![a-z0-9])|"
            r"\b(?:single[- ]cell|cell[- ]type[- ]specific)\s+"
            r"(?:s|splicing\s+|splice\s+)qtls?\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "sctuqtl",
        re.compile(
            r"(?:(?<![a-z0-9])sc[- ]?tuqtls?(?![a-z0-9])|"
            r"\b(?:single[- ]cell|cell[- ]type[- ]specific)\s+"
            r"(?:tu|(?:transcript|isoform)\s+usage\s+)qtls?\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "eqtl",
        re.compile(
            r"(?:(?<![a-z0-9])eqtls?(?![a-z0-9])|"
            r"\bexpression\s+(?:quantitative\s+trait\s+(?:locus|loci)|qtls?)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "pqtl",
        re.compile(
            r"(?:(?<![a-z0-9])pqtls?(?![a-z0-9])|"
            r"\bprotein(?:\s+(?:abundance|level))?\s+qtls?\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "sqtl",
        re.compile(
            r"(?:(?<![a-z0-9])sqtls?(?![a-z0-9])|"
            r"\b(?:splicing|splice)\s+qtls?\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "tuqtl",
        re.compile(
            r"(?:(?<![a-z0-9])tuqtls?(?![a-z0-9])|"
            r"\b(?:transcript|isoform)\s+usage\s+qtls?\b)",
            re.IGNORECASE,
        ),
    ),
)
_SINGLE_CELL_QTL = re.compile(
    r"(?:\b(?:single[- ]cell|cell[- ]type[- ]specific)\s+"
    r"(?:molecular\s+)?qtls?\b)",
    re.IGNORECASE,
)
_QTL_ACRONYM = re.compile(r"(?<![a-z0-9])([a-z]{1,12}qtl)s?(?![a-z0-9])", re.I)
_BROAD_QTL_REQUEST = re.compile(
    r"(?:\b(?:broad|all)\s+(?:molecular\s+)?qtls?\b|"
    r"(?<![a-z0-9])molqtls?(?![a-z0-9])|\bmolecular\s+qtls?\b)",
    re.IGNORECASE,
)
_SUPPORTED_QTL_ACRONYMS = frozenset((*QTL_STUDY_TYPES, "qtl", "molqtl"))
_UNSUPPORTED_QTL_WORDING = re.compile(
    r"(?:\b(?:methylation|metabolite|chromatin|histone)\s+"
    r"(?:qtls?|quantitative\s+trait\s+(?:locus|loci))\b)",
    re.IGNORECASE,
)


def route_chat_intent(question: str) -> ChatIntent:
    """Classify a question without a model or any external state.

    The precedence is deliberate: translational and disease questions often
    also contain words such as ``gene`` or ``target``. An unmatched question
    is general conversation; it must never silently become a selected-variant
    description merely because a Manhattan-plot selection exists.
    """

    return route_chat_request(question).intent


def route_chat_request(
    question: str, *, selected_variant_count: int = 0,
    has_selected_region: bool = False, source_trait: str | None = None,
) -> ChatRoute:
    """Return a deterministic intent and exact native QTL subtype selection.

    An empty ``qtl_types`` tuple means the established broad-QTL behaviour.
    It is not a failed parse.  Conversely, an unsupported acronym such as
    ``mQTL`` fails closed instead of being invented, ignored, or broadened.
    """

    normalized = " ".join(unicodedata.normalize("NFKC", str(question)).casefold().split())
    if is_explicit_general_chat(normalized):
        return ChatRoute(
            intent=ChatIntent.GENERAL,
            qtl_types=(),
            needs_retrieval=False,
        )
    qtl_types = _route_qtl_types(normalized)
    gene_evidence_only = bool(
        qtl_types or _BROAD_QTL_REQUEST.search(normalized) or _GENE_EVIDENCE_ONLY.search(normalized)
    )
    if _DRUG.search(normalized):
        intent = ChatIntent.DRUG_REPURPOSING
    elif _DESCRIBE.search(normalized):
        if _GENE_PRIORITIZATION.search(normalized) or (
            _GENE_ASSOCIATION.search(normalized) and not gene_evidence_only
        ):
            intent = ChatIntent.GENE_PRIORITIZATION
        elif _GENE_ASSOCIATION.search(normalized):
            intent = ChatIntent.GENE_ASSOCIATIONS
        else:
            intent = ChatIntent.VARIANT_DESCRIPTION
    elif _DISEASE.search(normalized):
        intent = ChatIntent.DISEASE_ASSOCIATIONS
    elif _TARGET.search(normalized):
        intent = ChatIntent.TARGET_ASSESSMENT
    elif _GENE_PRIORITIZATION.search(normalized):
        intent = ChatIntent.GENE_PRIORITIZATION
    elif _GENE_ASSOCIATION.search(normalized):
        intent = (
            ChatIntent.GENE_ASSOCIATIONS if gene_evidence_only else ChatIntent.GENE_PRIORITIZATION
        )
    else:
        intent = (
            ChatIntent.GENE_ASSOCIATIONS
            if gene_evidence_only
            else ChatIntent.GENERAL
        )
    route = ChatRoute(
        intent=intent,
        qtl_types=qtl_types,
        needs_retrieval=intent is not ChatIntent.GENERAL,
    )
    return reconcile_chat_route(question, route, selected_variant_count=selected_variant_count,
                                has_selected_region=has_selected_region, source_trait=source_trait)


def is_explicit_general_chat(question: str) -> bool:
    """Return true only for clear greetings or non-locus definitions.

    This narrow guard prevents a structurally valid but semantically wrong
    model-routing response from turning a greeting into an Open Targets run.
    Unmatched scientific questions are intentionally not treated as explicit
    general chat, so the model can still route multilingual locus questions.
    """

    normalized = " ".join(unicodedata.normalize("NFKC", str(question)).casefold().split())
    if _GREETING.fullmatch(normalized):
        return True
    if is_explicit_variant_evidence_lookup(normalized):
        return False
    if _LOCUS_CONCEPT_REQUEST.search(normalized):
        return True
    return bool(
        _GENERAL_KNOWLEDGE.search(normalized) and not _SELECTION_REFERENCE.search(normalized)
    )


def model_chat_router_messages(
    question: str,
    *,
    selected_variant_count: int,
    source_trait: str | None = None,
    selected_significant_count: int | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the bounded LLM routing request used by AI Chat.

    The model selects a scientific intent. Deterministic code still validates
    the returned enum and native QTL subtypes before any Open Targets tool can
    run, so the model cannot invent a tool name or broaden a requested subtype.
    """

    payload_value: dict[str, object] = {
        "question": str(question)[:parameter("web.chat_question_max_chars")],
        "selected_variant_count": max(0, int(selected_variant_count)),
    }
    if selected_significant_count is not None:
        payload_value["selected_significant_count"] = max(
            0, int(selected_significant_count)
        )
    if str(source_trait or "").strip():
        payload_value["source_trait"] = str(source_trait).strip()[:500]
    if conversation_history:
        payload_value["conversation_history"] = [
            {"role": item["role"], "content": item["content"][:1500]}
            for item in conversation_history[-4:]
            if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)
        ]
    payload = json.dumps(
        payload_value,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": load_prompt("chat_router_system")},
        {
            "role": "user",
            "content": format_prompt("chat_router_user", payload=payload),
        },
    ]


def parse_model_chat_route(content: str) -> ChatRoute:
    """Parse and validate one JSON-only model routing decision."""

    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model chat router did not return a JSON object") from None
        try:
            value = json.loads(text[start : end + 1])
        except ValueError as exc:
            raise ValueError("model chat router returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("model chat router result must be an object")
    old_fields = frozenset({"intent", "needs_retrieval", "qtl_types"})
    planned_fields = frozenset(
        {
            *old_fields,
            "phenotype",
            "biological_context",
            "evidence_kinds",
        }
    )
    fields = frozenset(value)
    if fields not in {old_fields, planned_fields}:
        raise ValueError("model chat router returned unexpected fields")
    try:
        intent = ChatIntent(value.get("intent"))
    except (TypeError, ValueError) as exc:
        raise ValueError("model chat router returned an unsupported intent") from exc
    needs_retrieval = value.get("needs_retrieval")
    if not isinstance(needs_retrieval, bool):
        raise ValueError("model chat router needs_retrieval must be boolean")
    if needs_retrieval == (intent is ChatIntent.GENERAL):
        raise ValueError("model chat router intent and retrieval decision conflict")
    raw_qtl_types = value.get("qtl_types")
    if not isinstance(raw_qtl_types, list) or any(
        not isinstance(item, str) for item in raw_qtl_types
    ):
        raise ValueError("model chat router qtl_types must be an array of strings")
    qtl_types = tuple(item.casefold() for item in raw_qtl_types)
    if len(qtl_types) != len(set(qtl_types)):
        raise ValueError("model chat router qtl_types contains duplicates")
    unsupported = tuple(item for item in qtl_types if item not in QTL_STUDY_TYPES)
    if unsupported:
        raise ValueError(
            "model chat router returned unsupported QTL subtypes: " + ", ".join(unsupported)
        )
    if intent is ChatIntent.GENERAL and qtl_types:
        raise ValueError("general chat cannot request QTL evidence")
    phenotype: str | None = None
    tissues: tuple[str, ...] = ()
    cell_types: tuple[str, ...] = ()
    evidence_kinds: tuple[str, ...] = ()
    planning_source = "model_router_legacy"
    if fields == planned_fields:
        phenotype_value = value.get("phenotype")
        if phenotype_value is not None and (
            not isinstance(phenotype_value, str)
            or not phenotype_value.strip()
            or len(phenotype_value) > 500
            or any(character in phenotype_value for character in "\r\n\x00")
        ):
            raise ValueError("model chat planner phenotype is invalid")
        phenotype = phenotype_value.strip() if isinstance(phenotype_value, str) else None

        context = value.get("biological_context")
        if not isinstance(context, dict) or set(context) != {
            "defined",
            "tissues",
            "cell_types",
        }:
            raise ValueError("model chat planner biological_context is invalid")
        defined = context.get("defined")
        if not isinstance(defined, bool):
            raise ValueError("model chat planner context defined must be boolean")
        tissues = _bounded_context_terms(context.get("tissues"), label="tissues")
        cell_types = _bounded_context_terms(
            context.get("cell_types"), label="cell_types"
        )
        if defined != bool((*tissues, *cell_types)):
            raise ValueError("model chat planner context definition conflicts with its terms")
        if defined and phenotype is None:
            raise ValueError("model chat planner cannot define context without a phenotype")

        raw_evidence_kinds = value.get("evidence_kinds")
        if not isinstance(raw_evidence_kinds, list) or any(
            not isinstance(item, str) for item in raw_evidence_kinds
        ):
            raise ValueError("model chat planner evidence_kinds must be an array of strings")
        evidence_kinds = tuple(item.casefold() for item in raw_evidence_kinds)
        if len(evidence_kinds) > len(EvidenceKind) or len(evidence_kinds) != len(
            set(evidence_kinds)
        ):
            raise ValueError("model chat planner evidence_kinds is duplicated or too large")
        supported_kinds = {kind.value for kind in EvidenceKind}
        unsupported_kinds = tuple(
            item for item in evidence_kinds if item not in supported_kinds
        )
        if unsupported_kinds:
            raise ValueError(
                "model chat planner returned unsupported evidence kinds: "
                + ", ".join(unsupported_kinds)
            )
        if intent is ChatIntent.GENERAL and evidence_kinds:
            raise ValueError("general chat cannot request evidence kinds")
        planning_source = "model_query_planner"
    return ChatRoute(
        intent=intent,
        qtl_types=tuple(item for item in QTL_STUDY_TYPES if item in qtl_types),
        needs_retrieval=needs_retrieval,
        phenotype=phenotype,
        context_tissues=tissues,
        context_cell_types=cell_types,
        evidence_kinds=evidence_kinds,
        planning_source=planning_source,
    )


def _bounded_context_terms(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"model chat planner {label} must be an array of strings")
    if len(value) > 12:
        raise ValueError(f"model chat planner returned too many {label}")
    terms: list[str] = []
    seen: set[str] = set()
    for item in value:
        term = " ".join(item.split())
        key = term.casefold()
        if (
            not term
            or len(term) > 120
            or any(character in term for character in "\r\n\x00")
            or key in seen
        ):
            raise ValueError(f"model chat planner returned an invalid {label} term")
        seen.add(key)
        terms.append(term)
    return tuple(terms)


def _route_qtl_types(normalized: str) -> tuple[str, ...]:
    unsupported = _UNSUPPORTED_QTL_WORDING.search(normalized)
    if unsupported:
        raise _unsupported_qtl(unsupported.group(0))
    for match in _QTL_ACRONYM.finditer(normalized):
        acronym = match.group(1).casefold()
        if acronym not in _SUPPORTED_QTL_ACRONYMS:
            raise _unsupported_qtl(match.group(0))

    selected: list[str] = []
    occupied: list[tuple[int, int]] = []
    for qtl_type, pattern in _QTL_ALIASES:
        for match in pattern.finditer(normalized):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            if qtl_type not in selected:
                selected.append(qtl_type)
            occupied.append(span)

    if _SINGLE_CELL_QTL.search(normalized):
        selected.extend(item for item in SINGLE_CELL_QTL_STUDY_TYPES if item not in selected)
    return tuple(item for item in QTL_STUDY_TYPES if item in selected)


def _unsupported_qtl(value: str) -> QTLRoutingError:
    return QTLRoutingError(
        f"unsupported or ambiguous QTL subtype {value!r}; supported native types are "
        "eQTL, pQTL, sQTL, tuQTL, sc-eQTL, sc-pQTL, sc-sQTL, and sc-tuQTL"
    )
