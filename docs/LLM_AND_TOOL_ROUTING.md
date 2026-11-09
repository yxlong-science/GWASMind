# LLM and tool-routing contract

## What GWASMind does

GWASMind is a post-GWAS interpretation system. It visualizes uploaded GWAS
summary statistics, lets a researcher select loci or variants, retrieves
relevant evidence with Open Targets as the primary live data source, and helps
interpret that evidence. A narrowly triggered Ensembl region lookup can provide
unranked coverage-gap context; it does not make GWASMind a replacement
biomedical database. GWASMind does not run the original association test.

## Responsibilities

The LLM may perform three bounded roles when the configured model has passed
Save & Test:

1. **Question routing.** Clear requests use predefined deterministic routes.
   Only an otherwise ambiguous selected-locus question asks the model for a
   strict JSON intent. A current plot selection is context, not an implicit
   request to describe the selection.
2. **Evidence synthesis.** For gene-prioritization tasks, it produces a strict,
   validated ranking that can cite only records in the evidence ledger.
3. **Conversational explanation.** For general, target, disease, and drug
   questions it answers from bounded context. Selected-variant and candidate-
   gene reports use a script-governed scientific renderer after any required
   ranking synthesis, so a second model call cannot alter statistics, modules,
   or references.

The LLM does not execute SQL or GraphQL, select a data release, change live or
frozen mode, relax leakage policy, resolve ambiguous entities silently, decide
whether an API failure means biological novelty, or provide hidden
chain-of-thought as evidence.

Deterministic code owns:

- input and identifier validation;
- the allowed intent, evidence-kind, module, and QTL subtype enums;
- dependency-aware semantic-tool planning;
- Open Targets API/MCP execution and normalization, plus the strictly bounded
  and provenance-labeled Ensembl coverage-gap request;
- provenance, release, policy, and resource enforcement;
- evidence sufficiency, conflict detection, output validation, and terminal
  status mapping.

## Request matrix

| User request | Open Targets retrieval | LLM role |
|---|---|---|
| Greeting or general genetics question | No | Direct conversational answer |
| Describe selected variants | Variant annotation, studies, credible sets, fine-mapping | Route and explain retrieved evidence; no gene-ranking synthesis |
| Show a named variant-to-gene evidence layer without ranking | Variant consequences, locus context, L2G shortlist, native QTL, colocalisation, regulatory evidence, and target profiles | Report typed links without a causal ranking |
| Ask which genes are associated with selected variants, or prioritize candidates | The same gene-link and target-profile dependencies, followed by a validated ranking | L2G creates only the hidden shortlist; the model ranks only genes with their own non-L2G QTL, colocalisation, or regulatory link |
| Assess target function or safety | Locus-to-target dependencies, function, expression, pathways, constraint, tractability, safety, interactions, phenotypes | Route and evidence-grounded explanation |
| Ask about associated diseases | Locus-to-target dependencies and direct target-disease evidence | Route and evidence-grounded explanation |
| Ask about drugs or repurposing | Locus-to-target dependencies, mechanisms, indications, warnings, adverse events, pharmacogenomics | Route and evidence-grounded explanation with efficacy/safety limitations |
| Report SNP count, interval, p-value range, or alleles | No external data required | No model is required for the deterministic measurements |

The high-level routes have predefined retrieval bundles. In particular, a
question about traits associated with selected variants always uses the
variant/study/credible-set/fine-mapping bundle. It cannot be changed by a model
into target-disease association or gene-prioritization retrieval.

If no verified LLM is available, explicit supported scientific intents can use
the deterministic planner and evidence summary. Ambiguous or general
conversation asks the user to configure and test a model; it must never fall
through to a selected-variant summary.

## Safety-status policy

`STOPPED` is reserved for invalid or unverifiable variant/build/allele identity,
conflicting critical identity inputs, or no verifiable retrieved evidence or
uploaded observations. Missing genes, direct molecular links or retrieval
coverage produces a Warning and limits the available conclusions. `report_allowed`
and `ranking_allowed` are separate program-computed capabilities: a report may
remain available while candidate-gene ranking is unavailable.

Partial biological-context coverage, a single unavailable complementary
evidence channel, limited evidence independence, conflicting evidence, a
complex genomic region, and missing publication metadata produce
`PASS_WITH_WARNINGS`; they do not suppress an otherwise supported result.
The safety auditor may explain these program-computed facts but cannot change
their status.

Final prose generation is not a scientific hard gate. Registry bookkeeping is
reconstructed locally and all prose is validated again. Scientific errors get
one bounded model repair. If it succeeds, its audit diagnostics do not add a
user warning. If it fails, independently verified statements and evidence can
be retained with an explicit interpretation gap; otherwise deterministic code
renders the verified evidence and references. These incomplete interpretations
produce `PASS_WITH_WARNINGS`. An incorrect statement is corrected or removed,
never published unchanged with a warning.

## Selected-variant description

`describe_variants` is deliberately separate from `prioritize_genes`. It can
return `EVIDENCE_RETRIEVED` with an evidence ledger and no ranked genes. It does
not require a genome-wide-significant selected variant, because describing a
measurement or annotation is not a causal-gene decision.

`gene_associations` is also separate from `prioritize_genes`. It retrieves and
explains the typed links between selected variants and genes, and can enrich
those genes with target profiles, but it does not automatically request drug
evidence or turn regional context into a ranking. `prioritize_genes` is the
explicit ranking task and must still satisfy the ranking and evidence-
sufficiency contracts.

Every retrieval-backed answer uses one outer presentation contract:

1. **Summary** gives the direct conclusion.
2. **Evidence** contains question-specific external records. Candidate-gene
   reports use the same four blocks for every displayed gene: Molecular
   QTL/Colocalization, Regulatory Region-to-Gene (E2G), Expression, and Function.
3. **References** lists only context-supplied publications or authoritative
   data documentation, deduplicated by PMID, with one reference per line.

For candidate-gene ranking, the model produces only the validated scientific
decision. Deterministic code then de-duplicates the evidence, renders the four
modules, and constructs the bibliography from source metadata. L2G rows,
scores, publications, and counts are not visible to the ranker or report.

Open Targets excludes credible sets whose lead variant lies in the documented
MHC interval `chr6:25,726,063-33,400,556`. For a selected MHC locus, an empty
external lookup is reported as a coverage limitation while the uploaded GWAS
trait and measurements remain visible. It is never translated into a claim
that the variants have no associated traits.

For non-benchmark `gene_associations` and `prioritize_genes` requests, GWASMind
first requests exact Open Targets variants and transcript consequences. Only
when **all** exact OT variants are absent may deterministic code derive one
same-chromosome GRCh38 interval and make one bounded Ensembl region-overlap
request. At most ten candidates are retained; exact interval overlaps are used
when present, and nearby genes are considered only when no gene overlaps the
original interval. Every record carries `Ensembl REST` provenance and is marked
`unranked` and `not_l2g_or_qtl`. This is coverage context, not L2G, QTL, or
causal-gene evidence. Benchmark and frozen-safe runs cannot use this path.

## Auto Analysis

Auto Analysis applies the same validated locus workflow to multiple loci. It
does not label an API failure or absent record as `Novel`. Results are reported
as prioritized or as an explicit needs-review/abstention state. If a tested
model is enabled it performs bounded evidence synthesis; otherwise the named
deterministic ranker is used. Follow-up chat reasons over the structured Auto
Analysis context instead of silently launching an unrelated selected-SNP
summary.

## Public progress and private reasoning

Before hosted-model synthesis, deterministic code creates a compact evidence
view from the full permitted ledger. It deduplicates repeated selected members
only when stable scientific identifiers establish the same assertion (for
example, the same gene and credible set or the same colocalisation pair).
Missing stable identifiers fail back to exact `evidence_id` identity rather
than an empty deduplication key. The projection preserves both supporting and
opposing records, gives every evidence-supported candidate at least one record,
balances genes and evidence types, and applies hard record and JSON-character
budgets. The generic compatible-model profile is 240 records / 320,000 evidence
characters; the configured `deepseek-v4` large-context profile is 600 records /
900,000 characters. If candidate coverage cannot fit, synthesis fails closed. Full raw
records remain in local artifacts; the model receives no repeated provider
`upstream.raw` envelope and may cite only top-level model-visible
`evidence_id` values. The decision artifact records the projection counts,
visible IDs, and exact model-input hash.

The UI shows a monotonic public sequence: question/intent analysis, planned
evidence types, the actual Open Targets modules being fetched, candidate-gene
discovery, evidence organization, prioritization, and response formatting.
SSE keepalives contain no model reasoning. Raw hidden reasoning, provider
secrets, untrusted tool payload instructions, paths, and private
chain-of-thought are not UI features and are not scientific evidence.
