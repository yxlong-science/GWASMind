# Runtime configuration and report gates

GWASMind loads three packaged TOML catalogs. They are validated before a web
server starts, and their effective values remain fixed for the life of that
process. Restart after editing a catalog; changing a file does not alter a
running analysis.

| Catalog | Purpose |
| --- | --- |
| `src/gwasmind/config/parameters.toml` | Scientific thresholds, retrieval and model budgets, worker limits, history bounds and plot sampling settings. |
| `src/gwasmind/prompts/llm_prompts.toml` | Model instructions, with purpose, stage, role, placeholders and output contracts stored outside the instruction content. |
| `src/gwasmind/config/gate_rules.toml` | Gate assignment, severity, dependencies, analysis permissions and required output-validator registration. |

Each parameter entry documents its value, original/default value, type, unit,
allowed range or choices, consumer locations and effect. Adjust `value` when
changing a packaged setting. Preserve `default` as the documented project
default. Parameters are not a replacement for curated provider capabilities,
genome assembly facts, schema definitions or identity-validation rules.

Each catalog has exactly one loader, which owns its parsing, validation and
single process snapshot: `gwasmind.configuration` (parameters),
`gwasmind.prompting` (prompts) and `gwasmind.gate_rules` (gate rules).
`gwasmind.config_files` names the three files. No other module reads them, and
Python or browser code must not repeat their values; the browser receives
parameters only through the generated runtime configuration.

The three catalogs are frozen for reproducibility. `docs/CONFIG_FREEZE_2026-09-28.json`
records their file hashes, their effective hashes and the key scientific thresholds.
`gwasmind.config_freeze` compares the loaded snapshots with every record in it.
The comparison runs once per process, when the configuration provenance is first
recorded (server start and run manifests). Drift emits a
`ConfigurationFreezeWarning`; with `GWASMIND_REQUIRE_CONFIG_FREEZE=1` it stops
startup instead. Local overrides that change a frozen threshold are reported
separately and are not drift of the packaged catalogs. A distribution without
the `docs` directory reports the freeze list as unavailable. Print the report
(exit status 1 on drift) with:

```powershell
python -m gwasmind.config_freeze
```

A catalog change without a new manifest is reported as drift. Regenerate the
manifest deliberately together with any catalog change.

## Local overrides and precedence

For a local installation, keep the packaged catalog unchanged and create a
TOML file containing only an `overrides` table. Dotted keys must be quoted:

```toml
[overrides]
"execution.max_concurrent_runs" = 2
"execution.auto_default_concurrent_loci" = 2
"ranking.max_ranked_genes" = 8
```

Set `GWASMIND_PARAMETERS_FILE` to that file's absolute path before launching.
Unknown keys, wrong types, out-of-range values and inconsistent combinations
fail validation before work starts. For example, the Auto default cannot
exceed the worker maximum, the web question limit cannot exceed the final
report's input limit, and the positive P floor cannot exceed the GWAS
significance threshold.

Explicit supported command-line or request options take precedence over their
catalog defaults. The PowerShell launcher also preserves existing environment
resource/model settings when no explicit launcher option overrides them.
Browser settings are generated from the loaded Python catalog; the browser
does not keep another set of default parameter values. The deterministic CLI
ranker retains its own fixed method identifier. An OpenAI-compatible CLI ranker
without explicit model/endpoint options uses the catalog model defaults.

API keys remain in the existing environment or runtime-memory mechanisms.
Do not put credentials in these TOML files. Configured base URLs cannot contain
embedded credentials, query parameters or fragments.

## Gate 1 and Gate 2

Gate policy (`gate_rules.toml`):

* **Stop** only when the question cannot be answered at all: no source variant
  has a verifiable identity, no data is available, or (for gene prioritization)
  the lead window contains no candidate gene. These are operational receipts
  computed in code; no Gate 1 rule can be a hard stop.
* **Warning (W code)** only for a limitation that can change the main
  conclusion, for example no candidate with any locus-to-gene link (W1, the
  ranking is exploratory), a required or core locus channel that produced no
  data (W4), no significant selected variant (W2), a source SNP overlapping a
  complex region (W8), or a requested ranking that produced no output. The
  ranking-unavailable warning names its classified cause (for example no model
  configured, timeout after retries, input capacity exceeded, or invalid
  structure after one repair); provider error text is never shown.
* Every rule has its own code (`W1.1`–`W1.4`, `W4.1`–`W4.4`, `G2.1`, `G2.2`;
  single-rule families keep `W2`, `W3`, `W5`–`W9`). Displayed warnings are
  grouped by family and list the rule codes that fired; the loader rejects
  duplicate codes.
* **Method note** for everything else: optional or background channels,
  channels that stopped early after returning data, tissue mismatch without a
  tissue-mechanism request (W3), provenance bookkeeping without an
  original-publication request (W9), and an unavailable explanatory review.
  `severity = "note"` in `gate_rules.toml` makes a rule note-only.
* **Gate 2** controls hallucination in model output. It corrects uniquely
  grounded values, removes citations that belong to another gene or were never
  shown to the model, drops claims left without citations, removes sentences
  naming identifiers absent from the verified identity registry and evidence.
  Ambiguous or unverifiable numbers remain unchanged and are recorded for
  review; a confirmed impossible value is removed only with its safely identified
  complete assertion. It preserves negation, qualification, source phenotype
  identity, gene order and declared ties. It never adds a
  warning, lowers report quality or stops an analysis; its edits are kept in
  the audit (`hallucination_removals`, corrections and fact issues). The loader
  rejects any Gate 2 rule that is not a non-blocking note.

Gate 1 checks input identity, evidence availability and whether the evidence
permits the requested analysis. Missing retrieval coverage is a cause;
ranking that could not be attempted is a consequence. The UI groups these
together so one missing dataset does not appear as several unrelated failures.

For ordinary gene prioritization, W1 is task-level answerability: a missing
direct variant-to-gene record for one candidate does not invalidate that
candidate or the whole shortlist when qualified contextual, functional,
coding or regulatory evidence remains. A direct-link requirement is enforced
only when the user explicitly asks for a direct V2G mechanism. W5 is emitted
only when independent replication is requested and disjoint cohorts cannot be
verified. W6 is limited to comparable hypotheses: opposite support claims or
opposite Beta directions are compared only after the variant, effect allele,
trait, endpoint, tissue, model, scale and population are aligned. Proxies,
allele flips, missing metadata and cross-context heterogeneity are not treated
as contradictions.

Gate 2 validates every generated ranking/report draft and repair, and the
content retained for the user. Its checks cover schema, evidence identifiers,
ownership, scientific claims and supported report statements. Repair and
partial-report recovery use the same validators as the initial draft.

The public response explains meaningful warnings at the beginning and records
the relevant uncertainty in its limitations. Gate 2 is an objective
fact-verification step, not a new academic warning category: it corrects or
marks unsupported numeric, identity, ownership and citation claims in the
final draft, while the UI shows the correction ledger and any unresolved
fact cards. Gate 2 success does not erase Gate 1 warnings. A supported partial
result or exploration can remain useful with a warning; unsupported claims
cannot become facts merely to avoid a Stop.
The ranking validator receives the frozen shortlist from the acquisition stage
and requires the model to rank every member exactly once; the interface may
display only the first five rows.
When nothing valid can be retained, the response states what was available and
what prevented a defensible answer.

The declarative rules preserve check identifiers. `novelty_coverage` belongs
to required retrieval coverage, `ranking_not_attempted` records an execution
consequence, and `model_output_grounding` belongs to Gate 2. The rule loader
requires critical identity and output-validation contracts: arbitrary code or
rules that disable those contracts are not accepted.

## Scientific scope and display

The LD novelty search uses its own declared population and search flank. Its
default flank is 250 kb on each side, independently of the native gene-context
window of 1 Mb on each side. Missing or truncated LD coverage remains
unresolved; absence of returned data does not become an LD value of zero.

Gene prioritization has an independent LD-expansion module. Its default
`ranking.ld_r2_threshold` is 0.8 and `ranking.ld_window_bp` is 250000 on each
side. `ranking.ld_expansion_enabled` controls this module. The novelty
threshold is configured separately. The other evidence-analysis workflows
continue to use the user's selected SNPs.

Expansion retains the original source observations and their P values. Added
proxy variants carry no copied source P value. Reference-panel identity,
population, source-to-proxy LD links and coverage are recorded separately.
Variant-scoped retrieval uses bounded batches before joint candidate discovery
and a single ranking. A gene supported through a proxy is labeled as supported
through population LD; this does not establish direct molecular evidence for
the selected source SNP. Unresolved proxy identities or unfinished coverage
remain visible in warnings and limitations.

`ld_proxy_scope` and `ld_proxy_coverage` are method-level facts. Verified
population-LD proxies may support an exploratory ranking, but proxy coverage
gaps and proxy-identity failures do not create a W7 warning. W7 is reserved
for a source SNP selected by the user whose identity cannot be resolved.
Available verified evidence may still support a limited ranking; proxy-only
records never inherit the source SNP's P value. Complex-region warnings use
the versioned curated registry in `config/complex_regions.json` (MHC/HLA,
KIR, V(D)J and selected inversion intervals) and are emitted for source-SNP
overlap; proxy-only overlap remains a method note.

GWAS significance counts compare P directly with
`analysis.gwas_significance_p`. `plot.threshold_decimal_places` controls
display coordinates and the significance line, so rounding cannot move a
measurement across the scientific threshold. Imported zero/underflow P values
retain the existing positive-floor normalization. Sampling priority is a
separate display setting; scientifically significant observations remain
priority observations even when their plotted height is below that setting.

## Reproducibility

Run manifests, workflow outcomes and saved Chat-answer audits record
`configuration_provenance`. It includes the three loaded catalog versions,
raw-file hashes, effective-content hashes and non-secret parameter values.
Prompt provenance additionally records the effective prompt content and its
parameter bindings. Prompt descriptions and other metadata are not sent to
the model.

These hashes describe the cached snapshot actually used by the process, even
if a file is edited afterward. Explicit request/model overrides are retained
in their existing request and model records. They do not silently rewrite the
catalog provenance. The Python export below prints the current process's
effective configuration without starting a service:

```powershell
python -c "from gwasmind.configuration import main; main()" --provenance
```
