# GWASMind Tutorial

A guide to the GWASMind local interface: uploading GWAS summary statistics,
selecting variants on the Manhattan plot, asking questions in AI Chat, screening
many loci with Auto Analysis, and reading and exporting the reviewed reports.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Uploading Your GWAS Data](#2-uploading-your-gwas-data)
3. [Interactive Manhattan Plot](#3-interactive-manhattan-plot)
4. [AI Chat Assistant](#4-ai-chat-assistant)
5. [Auto Known / Novel Screening](#5-auto-known-novel-screening)
6. [API Configuration](#6-api-configuration)

---

## 1. Getting Started

GWASMind is a local web application for visualizing and interpreting
Genome-Wide Association Study (GWAS) results. Key features include:

- **Interactive Manhattan Plot** – Explore a bounded representation of large
  GWAS files with smooth zooming and panning
- **Lasso Selection** – Select regions of interest by drawing directly on the plot
- **Evidence-Aware AI Chat** – Ask questions about selected variants using a
  single GWAS agent with Open Targets as the primary live data source and an
  explicitly labeled, bounded Ensembl coverage-gap lookup
- **Auto Locus Annotation & Gene Prioritization** – Annotate significant loci
  and prioritize candidate genes using fine-mapping and multi-omics evidence,
  with explicit limitations when the available evidence is insufficient

![Homepage showing upload and demo data options](/static/docs/screenshots/homepage.png)

> **Tip:** New to GWASMind? Try our Demo Data (Asthma or Depression GWAS) to explore all features without uploading your own data!

---

## 2. Uploading Your GWAS Data

GWASMind accepts GWAS summary statistics in CSV, TSV, or TXT format.

### Required Columns

| Column | Description |
|--------|-------------|
| `chromosome` | Chromosome number (1-22, X, Y, or chr1-chr22) |
| `position` | Base pair position (1-based) |
| `pvalue` | P-value from association test |
| `ref` | Reference-genome allele at the stated position and genome build |
| `alt` | Alternate allele at that same position |

Coordinate-based variants are interpreted as **GRCh38/hg38**. GWASMind does
not silently liftover GRCh37/hg19 coordinates; use canonical rsIDs or convert
and validate coordinates before analysis.

REF and ALT do not specify the effect allele. Columns named `A1`/`A2` or
`effect_allele` are not silently substituted for REF/ALT. When reference
verification uniquely confirms an exact REF/ALT reversal at the same GRCh38
position, GWASMind uses the corrected ID for retrieval and records the original
and corrected IDs. The uploaded P value, beta and odds ratio remain unchanged.
An ambiguous identity is kept separate from a successful database query with
no matching research evidence.

### Optional Columns

| Column | Description |
|--------|-------------|
| `snp_id` or `rsid` | SNP identifier (rs number) |
| `beta` | Effect size |
| `odds_ratio` | Odds ratio, retained as supplied |
| `se` | Standard error |
| `effect_allele` | Allele to which the supplied beta or odds ratio refers |
| `other_allele` | Other allele of the effect contrast, when provided |

Without an explicit effect allele, GWASMind can report the uploaded effect
value but cannot assign its direction to the reference or alternate allele.

![File upload interface with column requirements](/static/docs/screenshots/upload.png)

> **Note:** Files are parsed locally in your browser in 4 MB chunks. GWASMind
> retains every record with `-log10(p) >= 2.5` and samples lower points in ten
> 0.25-wide bands, with up to 10,000 background records per band. A 300,000-row
> hard ceiling applies only to pathological inputs. The raw file is not posted
> to the Python service.

> **Renderer note:** GWASMind first gives every processed point to Plotly
> `scattergl`. If WebGL is unavailable, a bounded 20,000-point SVG interaction
> proxy retains the genome-wide-significant peaks and a chromosome-spanning
> background sample so the page remains responsive.

---

## 3. Interactive Manhattan Plot

The preserved Plotly Manhattan plot is the main visualization, displaying
-log₁₀(p-value) across all chromosomes.

### Navigation Controls

| Action | Description |
|--------|-------------|
| **Scroll** | Zoom in/out on the plot |
| **Drag** | Lasso selection for regions of interest |
| **Shift + Drag** | Lasso selection for multiple regions |
| **Double-click on chromosome** | Focus on that chromosome |

### Plot Controls

- **Threshold slider** – Adjust the significance threshold line
- **Y-axis Max** – Set maximum -log₁₀(p) value displayed
- **Marker Size** – Increase or decrease point size
- **Reset Axes** – Return to full chromosome view
- **Export** – Save plot as PNG, SVG, or PDF

### Lasso Selection

Draw a selection around SNPs of interest to analyze them:

1. Click and drag on the plot to draw a selection region
2. Selected SNPs will be highlighted
3. A panel will show the selection summary
4. Click "Label Gene" to annotate the lead SNP

![Lasso selection highlighting a genomic region](/static/docs/screenshots/lasso_selection.png)

---

## 4. AI Chat Assistant

The AI Chat panel allows you to ask questions about selected SNPs and receive
selection-aware evidence summaries. One GWAS agent chooses among bounded
semantic tools; identifier validation, retrieval, filtering, and provenance
remain deterministic.

### How to Use

1. Select SNPs using the lasso tool on the Manhattan plot
2. Click the "Describe selected SNPs" quick prompt, or type your own question
3. GWASMind queries the official Open Targets GraphQL API as its primary live
   data source and returns an evidence-linked summary with limitations. For an
   eligible gene question only, if every exact OT variant is absent, it may make
   one bounded Ensembl GRCh38 region-overlap lookup for unranked regional context

Clear questions use predefined retrieval routes; only an otherwise ambiguous
selected-locus question uses the tested model's strict JSON router. The routes
distinguish general conversation, variant description, gene association, gene
prioritization, target assessment, disease association, and drug repurposing.
Gene association asks which genes are linked by retrieved evidence without
automatically requesting drugs or claiming a ranking. Gene prioritization is a
separate task that may produce a validated candidate ranking. General questions
do not query the selected locus. A selected-variant trait question always
retrieves variant, study, credible-set, and fine-mapping context without running
gene ranking. The other scientific intents use their matching dependency-aware
plans.
GWASMind does not guess a disease ontology ID from free text.

After external retrieval, answers always use **Summary**, **Evidence**, and
**References** sections. References are limited to publication metadata returned
with the evidence or authoritative documentation for a stated coverage rule.
An empty Open Targets match is reported as a coverage result and never as proof
that an association does not exist.

For **Describe selected SNPs**, the summary covers the complete selection, then
Evidence shows at most five trait-diverse SNPs and up to five distinct Open
Targets GWAS associations per SNP. Each available selected-member P value and
beta is paired with an inline publication citation. The References list contains
only publications actually cited in those displayed associations; uploaded-GWAS
measurements are not repeated as a separate Source GWAS block.

If a coverage-gap Ensembl lookup is used, its records are shown separately with
`Ensembl REST` provenance and `region_overlap` or `nearby` relation. They are
unranked regional candidates, not Open Targets associations, L2G or QTL links,
or causal-gene evidence.

### Example Questions

- "Which genes are associated with these SNPs?"
- "Describe selected XXX SNPs?"
- "What diseases have been associated with these SNPs?"
- "Explain the biological function of [gene name]"

![AI Chat panel with natural language](/static/docs/screenshots/ai_chat.png)

> **Important:** GWASMind is an AI-assisted interpretation system. Verify
> candidate-gene conclusions against the cited evidence and primary literature;
> a prioritized candidate is not proof of causality.

---

## 5. Auto Known / Novel Screening

Auto Analysis performs one Known / Novel screening task per uploaded locus. It
checks prior associations for the specified phenotype, including the existing
reference-population LD screening when applicable. It does not automatically
start candidate-gene ranking.

### Input Format

Prepare a file with your significant loci in one of these formats:

**Format 1: rsID + Locus ID**

```
rsid,p,locus_id
rs1002706,1e-9,1
rs76527307,2e-8,2
rs12345678,4e-10,3
```

**Format 2: Full coordinates**

```
chr,pos,ref,alt,p,locus_id
1,169519049,T,C,1e-9,1
1,206653061,G,A,2e-8,2
2,102847563,A,G,4e-10,3
```

Each locus should include at least one supplied p-value so GWASMind can verify
the genome-wide signal before retrieving evidence. The bundled Auto Analysis
demo contains 503 real chromosome-1 rows across eight loci, with rsIDs,
coordinates, p-values, locus assignments, and LD r2. Every demo locus has at
least one genome-wide-significant row, so it exercises actual evidence
retrieval and plot highlighting. See `static/demo_data/README.md` for source
hashes and provenance.

> **Legacy layout reference:** The image below illustrates the preserved upload
> panel. The current screening labels and their scope are defined below; example
> counts in the image are illustrative.

![Auto Analysis layout reference](/static/docs/screenshots/auto_analysis_upload.png)

### Analysis Steps

1. Switch to the "Auto Analysis" tab in the right panel
2. Enter your target phenotype (e.g., "Asthma", "Type 2 Diabetes")
3. Upload your loci file or drag-and-drop
4. Click "Start" to begin the analysis
5. Watch each locus's screening progress and open its result or report

The bounded worker pool runs up to six loci at a time by default. There is no `Concurrent loci`
setting. Cancelling a batch preserves completed results. Clicking a
locus still highlights its uploaded variants and opens its discussion panel.
Each locus occupies one collapsed row; expand it for progress and results.
Each row includes its execution time, excluding queue wait. The overall timer
shows the batch's elapsed wall time. Recorded times stop when execution ends and
remain available in History; older records without timing show that it was not
recorded. The shared task, target phenotype and input scope appear directly above
the list. Use the discussion panel's
close button to hide it; selecting a locus reopens the saved conversation.

### Results

For each locus, GWASMind provides:

- **Basic Info** – Chromosome, screening result, SNP count, lead SNP position,
  and significance
- **Known** – A verified prior association or qualifying LD link with
  model-assessed phenotype relevance
- **Novel (in scope)** – A completed negative screen within the recorded database,
  phenotype, reference population and search window
- **Unresolved** – Incomplete or unavailable screening, or only a narrower
  selected-variant lookup without sufficient LD coverage
- **Screening Reason and Report** – Retained associations, references, coverage
  and the reason for the result

`Novel (in scope)` does not establish worldwide novelty. A missing or failed
response is never automatically labeled Novel. Gene prioritization remains
available when explicitly requested in the locus discussion. Existing History
entries retain the ranking results that were originally generated.

The model assesses phenotype meaning, including aliases, subtypes and composite
conditions. Its decisions must reference the original retrieved labels and records;
the factual Gate checks study, publication, variant and LD support. Failed or
uncertain semantic assessment cannot establish Novel.

LD proxy screening is used when the selected SNPs do not already provide a
verified direct prior-report match and the prerequisite checks pass. The existing
search uses r2 >= 0.2 within 250 kb on each side of each selected SNP, with EUR
as the default reference population. A verified direct Known result can finish
without an LD query. The separate gene-prioritization LD expansion is not started
by Auto screening.

> **Legacy layout reference:** The screenshot below illustrates locus cards and
> their detail panel. New screening results use the definitions above.

![Analysis results layout reference](/static/docs/screenshots/analysis_results.png)

### Export Results

Click the "XLSX" button to download a structured Excel report containing the
screening results. The workbook includes classification, reasons, phenotype,
recorded LD scope and report references. New screening exports omit gene-ranking
columns. Saved analyses can also be reopened and exported after a restart.

### Open Targets data, bounded Ensembl context, and benchmark boundary

- Interactive Chat and Auto Analysis query the official Open Targets GraphQL API
  as the primary live data source. The current Open Targets release does not
  need to be downloaded.
- Only non-benchmark gene-association and gene-prioritization queries can use the
  coverage-gap rule. If all exact OT variants are absent, GWASMind may make one
  bounded Ensembl GRCh38 region request and return at most ten explicitly
  unranked overlap genes; nearby genes are used only when none overlap. These
  records are labeled as Ensembl provenance and are not L2G, QTL, rankings, or
  causal-gene evidence.
- An optional `frozen-safe` provider can use a deliberately provisioned,
  versioned local snapshot for reproducibility, time cutoffs, and leakage
  control. Frozen startup requires a verified
  `gwasmind_snapshot_manifest.json`; missing, `unverified`, release-mismatched,
  or changed inventories fail closed. GWASMind does not automatically download
  that snapshot and never silently switches between Live and Frozen.
- Benchmark code, labels, leakage policies and source manifests are kept
  outside this software package. The browser UI cannot start a benchmark run,
  and benchmark execution cannot call the Ensembl coverage-gap service.

---

## 6. API Configuration

To use model-generated ranking and summaries, connect a supported model
provider. Without a configured model, deterministic validation and evidence
handling remain available and the workflow fails closed where a model decision
is required.

### Supported Providers

Settings shows only providers that GWASMind can connect to directly. Providers
that require an unsupported native protocol, OAuth flow, cloud deployment, or
multi-field authentication are omitted. A trusted custom OpenAI-compatible
endpoint can also be used.

DeepSeek is the default provider and `deepseek-v4-pro` is the default model.

### Configuration Steps

1. Click **Settings** in the header.
2. Choose a provider.
3. Choose a model. Click **Refresh** to retrieve the current model IDs from the
   selected provider; refreshing does not save or activate the configuration.
4. Enter the provider API key. Official endpoints are filled in and locked by
   default; use **Advanced** only for a trusted proxy or lab gateway.
5. Keep **DeepSeek Thinking Mode** enabled for the default model, or disable it
   explicitly for a faster non-thinking run. Select **Low**, **High**, or
   **Max** thinking effort; High is the default. Effort changes reasoning
   intensity, while all official DeepSeek analysis calls use the same 384K
   provider output ceiling so hidden reasoning cannot truncate the answer.
6. Click **Save & Test**. GWASMind sends one bounded availability request and
   activates the configuration only after it succeeds. Success closes Settings;
   failure keeps the dialog and entered values open for correction. AI analysis
   remains blocked until the exact provider, endpoint, and model pass the test.

> **Current Settings:** The compact dialog defaults to `deepseek-v4-pro`,
> supports live model refresh, and reports success only after a bounded provider
> request succeeds.

![Current compact API Settings dialog](/static/docs/screenshots/settings.png)

> **Privacy:** The browser sends the key only to the loopback GWASMind service.
> It is held only in local GWASMind process memory for the current session and
> is not stored in browser storage, logs, manifests, exports, or project files.
> The key disappears when the local service stops. The configured provider
> receives the model request and its analysis context. A custom endpoint also
> receives the API key, so use only a service you trust.

---

## Need Help?

If you have questions or encounter issues, please visit our [Contact page](/contact) or reach out to our team.

