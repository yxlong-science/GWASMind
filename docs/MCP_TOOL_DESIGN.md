# GWASMind Open Targets semantic MCP design

## Scope and identity

GWASMind is an **unofficial semantic wrapper over Open Targets Platform**. It
must not present its own tools, rankings, or derived values as official Open
Targets MCP tools or scores.

The hosted Open Targets MCP exposes generic schema, GraphQL, batch, and
entity-search operations. The official GraphQL API exposes the same underlying
Open Targets data and is GWASMind's lightweight live default. GWASMind places a
stable, domain-specific tool layer above either official transport and above
the local Parquet release. The same semantic request is therefore usable in two
modes:

- `frozen-safe`: a local, versioned, resource-limited DuckDB provider;
- `live`: the official GraphQL API by default, or an explicitly injected
  hosted-MCP transport, as a schema-discovered provider.

The latest release is **not downloaded by GWASMind**. Live exploration reads it
through the official MCP/API. The 56-dataset catalog is a capability and schema
map, not a download manifest. A local download is used only for an explicitly
frozen release when reproducibility, temporal analysis, or leakage control
requires it.

Frozen entry points require a conventional
`gwasmind_snapshot_manifest.json`. They revalidate its release, manifest hash,
and dataset file-count/byte-size inventory without opening Parquet contents;
`unverified`, missing, mixed-inventory, or release-mismatched roots fail closed.
The optional provisioning-time content digest is bound into the manifest but is
not recomputed on every interactive startup.

"Use all Open Targets data" means that every relevant data module is registered,
discoverable, and reachable by an appropriate semantic tool. It does **not**
mean that every locus query loads every dataset. The single agent selects only
the modules required for the scientific question and the active policy.

## Public semantic tools

| Tool | Purpose | May affect locus-to-gene ranking? |
|---|---|---|
| `ot_get_capabilities` | Report provider, release, schema hash, installed datasets, modules, limits, and policy availability. Runtime-only; not an agent reasoning tool. | No |
| `ot_resolve_entities` | Resolve target, variant, disease, drug, study, biosample, and pathway identifiers without silently choosing an ambiguous top hit. | No |
| `ot_get_locus_context` | Retrieve variant annotation, GWAS/molQTL study metadata, lead variants, credible-set members, PIP, fine-mapping metadata, ancestry, and biosamples. | Establishes locus validity; does not name a causal gene by itself. |
| `ot_get_locus_gene_evidence` | Retrieve distinct variant-to-gene relations: consequences, molecular QTL, colocalisation, enhancer-to-gene, and optionally L2G. | Yes, subject to policy and sufficiency rules. |
| `ot_get_target_profiles` | Retrieve target identity, function, expression, GO, pathways, subcellular location, constraint, homologues, interactions, mouse phenotypes, essentiality, tractability, safety, probes, TEP, hallmarks, and literature. | Contextual/secondary only. |
| `ot_get_target_disease_evidence` | Retrieve direct and ontology-propagated target-disease associations, datasource/datatype scores, and raw evidence. | Contextual outside safe benchmarks; never interpreted as a causal probability. |
| `ot_get_disease_profiles` | Retrieve disease identity, ontology position, therapeutic area, signs/phenotypes, known therapies, and literature. | No; controls phenotype interpretation. |
| `ot_get_drug_evidence` | Retrieve known or clinical candidates, mechanisms, indications, phase, warnings, adverse events, and pharmacogenomics. | No; translational follow-up only. |
| `ot_analyze_gene_set` | Deterministically perform GO/pathway over-representation, tissue coverage, and evidence-coverage analyses with an explicit background and multiple-testing correction. | Secondary analysis only. |

The tools are intentionally broader than one-table functions. Physical Open
Targets tables and GraphQL fields are registered as modules under a semantic
tool. Adding a new upstream field changes a module, not the public API.

Both locus tools accept an optional `qtl_types` list containing only native
Open Targets study types: `eqtl`, `pqtl`, `sqtl`, `tuqtl`, `sceqtl`,
`scpqtl`, `scsqtl`, and `sctuqtl`. An empty list retains broad molecular-QTL
retrieval across that whitelist. Chat phrases are mapped deterministically
from explicit English/Chinese aliases; unsupported or ambiguous subtypes fail
closed. Returned QTL records retain their native `studyType` and are filtered
on that field, rather than classifying every non-GWAS study as interchangeable
QTL evidence.

## Mandatory analysis order

For `prioritize_genes`, the agent may make at most two follow-up rounds and six
agent-visible calls:

1. runtime calls `ot_get_capabilities` and binds the data/policy identity;
2. agent calls `ot_resolve_entities` only for unresolved or ambiguous inputs;
3. agent must call `ot_get_locus_context`;
4. if the locus is valid, agent must call `ot_get_locus_gene_evidence`; its
   modules execute in the fixed dependency order L2G, native QTL, and exact
   QTL-`studyLocusId` colocalisation. E2G retrieval is an independent branch:
   each selected variant is matched only to regulatory intervals that contain
   its exact GRCh38 coordinate, and each interval retains its own predicted
   target gene, biosample, datasource, combined score, component scores, and
   publication provenance;
5. benchmark runs retain their pre-specified candidate universe; exploratory
   runs may leave it empty and deterministically derive at most 20 candidates
   from the typed locus-to-gene records returned in step 4 (top ten L2G first,
   followed by native-QTL genes) before calling `ot_get_target_profiles`;
6. disease, drug, and gene-set tools are optional and task-dependent. Normal
   non-benchmark prioritization requests only bounded `drug.clinical_target`
   follow-up for the preserved Related Drugs panel.

The agent cannot select a release, loosen a leakage policy, issue arbitrary SQL
or GraphQL, alter a benchmark candidate universe, calculate statistical tests,
or turn `no_data` into a novelty claim. An exploratory candidate universe is a
recorded deterministic derivative of step-4 evidence, not an LLM choice.

The current E2G coverage scope is `selected_variants_exact_overlap`. It must not
be described as LD-expanded coverage. `intervalType` is preserved verbatim: an
`intergenic` regulatory interval must not be relabelled as an enhancer. A valid
zero-row E2G connection is `no_data`; a missing/malformed connection, count
mismatch, invalid interval, or non-overlapping row is a technical failure.

## Unified result envelope

Every tool returns the same top-level contract:

```json
{
  "schema_version": "gwasmind-ot-tools/1",
  "tool_name": "ot_get_locus_gene_evidence",
  "call_id": "uuid",
  "status": "success",
  "provider": "frozen",
  "data_release": "25.09",
  "snapshot_sha256": "...",
  "schema_sha256": "...",
  "policy_sha256": "...",
  "input_sha256": "...",
  "coverage": {
    "requested": [],
    "completed": [],
    "no_data": [],
    "unavailable": [],
    "blocked": [],
    "failed": [],
    "truncated": []
  },
  "records": [],
  "pagination": null,
  "warnings": [],
  "error": null,
  "raw_artifact": null
}
```

Status semantics are strict:

- `success`: every requested module completed and at least one returned data;
- `no_data`: every requested module was checked successfully, but no records
  matched the declared filters;
- `partial`: at least one module completed and at least one was unavailable,
  blocked, failed, or truncated;
- `unavailable`: the active release/schema does not expose the requested module;
- `blocked`: policy rejected the request before provider access;
- `invalid_input`: identifiers, build, range, module, or pagination are invalid;
- `error`: provider or contract failure; required-module errors force abstention.

`no_data` always includes checked modules and filters. It never means that a
gene or mechanism is novel.

## Evidence record rules

Normalized records preserve upstream meaning instead of collapsing metrics:

- fine-mapping PIP remains `posterior_probability`;
- COLOC H4 remains `h4` and eCAVIAR output remains `clpp`;
- Open Targets L2G output remains `l2g_score`;
- target-disease association output remains `association_score`;
- fine-mapping `confidence` remains a method/data-quality label.

GWASMind does not create a universal `confidence_score`. These measurements are
not interchangeable and none is automatically a probability of causal truth.

Each record includes subject, relation, object, measurements, study/biosample
context, release, publication provenance, and recursive parent lineage. Raw
upstream values and GWASMind-derived values are kept in separate namespaces.

## Frozen data modules

The current local Open Targets 25.09 snapshot contains these datasets:

| Dataset | Primary semantic use |
|---|---|
| `variant` | entity resolution, alleles, frequency, VEP consequences |
| `study` | GWAS/molQTL metadata, ancestry, sample size, trait, publication |
| `credible_set` | lead and member variants, PIP, fine-mapping, LD/purity |
| `l2g_prediction` | optional exploratory L2G evidence; policy-gated |
| `colocalisation_coloc` | GWAS/QTL colocalisation |
| `enhancer_to_gene` | regulatory variant-to-gene links |
| `target` | target identity, function, GO IDs, pathways, safety, tractability |
| `expression` | baseline RNA/protein tissue and cell-type expression |
| `disease` | disease identity and ontology context |
| `known_drug` | frozen drug-target-disease clinical context |
| `biosample` | tissue/cell ontology normalization |
| `go` | GO identifier labels |
| `reactome` | pathway hierarchy and labels |

Some other local snapshots contain direct and indirect association tables. The
capability registry must detect them from the configured root; their presence
must never be assumed from a release label alone.

## Live compatibility boundary

Live mode discovers the official schema before enabling modules. It records the
actual transport and endpoint, Open Targets API/data versions, and a schema
hash. GraphQL results are never mislabeled as MCP results. Templates and caches
are isolated by that identity.

The official schema is allowed to add fields. Removal or incompatible type
changes to a required field disable only the affected module and fail closed.
In particular, clinical/drug types have changed between releases, so the live
provider feature-detects clinical reports and drug/clinical candidates instead
of assuming a legacy `KnownDrug` type.

Offset pagination and evidence cursor pagination are separate adapters. A tool
may not silently truncate either form, and upstream counts are treated as
unbounded integers.

### Current live adapter coverage

Live startup currently negotiates the official schema and conditionally enables:

- entity candidates without automatic top-1 selection;
- study, credible-set, fine-mapping, QTL, colocalisation, enhancer-to-gene,
  and policy-gated L2G modules;
- target core/function/pathway/GO/expression/literature plus schema-confirmed
  safety, tractability, and molecular interactions;
- disease core and ontology context;
- direct/indirect overall, datasource, and datatype association scores with
  bounded offset pagination;
- the 26.06 `drugAndClinicalCandidates`/clinical-target model;
- schema-confirmed drug core, mechanism, indication, warning, adverse-event,
  and pharmacogenomics modules, with native stage/signal/evidence-level values
  preserved and never relabeled as probabilities.

The following remain unavailable until their schema-specific adapters and
contract tests are added: cursor-paginated raw evidence, mouse phenotypes,
essentiality, and deterministic gene-set enrichment. The capabilities tool
exposes these gaps explicitly.

## Resource controls

Interactive frozen queries default to:

- 2 DuckDB threads;
- 4 GB DuckDB memory limit;
- bounded result pages and model projections;
- chromosome pushdown for variant-to-credible-set matching;
- no background full-table scans during an analysis.

A production snapshot build may create `variant_to_study_locus`,
`study_locus_member`, target-symbol, and biosample-alias indexes in a separate,
explicit maintenance command. The interactive provider consumes those indexes
but never builds them implicitly.
