# Local history

GWASMind keeps restart-safe history in `data/history/history.sqlite3` beside the
software. This directory is local data and is excluded from version control.
SQLite may also create its normal `-wal` and `-shm` companion files while running.
Close GWASMind before backing up or moving the directory, and copy the whole
directory rather than copying a live database file alone.

## What is saved

- Dataset history stores the validated, downsampled plotting rows, their source
  GWAS phenotype, filename, sampling metadata, and upload timestamps. It does not
  make a second copy of the original large GWAS file. Identical normalized uploads
  reuse their history entry. Opened demo datasets are also saved once so restoring
  a demo conversation restores the correct plot and source phenotype.
- Chat history stores the actual visible Markdown messages and their selected
  variants, region, saved dataset identifier, report quality, and run identifiers.
  Reopening history does not silently re-run a model or regenerate an old answer.
- Auto Analysis history stores uploaded locus queries and the actual completed
  reports/results, together with source/target phenotype and status information.
  Each locus also retains its own follow-up conversation. Switching loci or
  reopening a saved analysis restores those exact messages without rerunning it.

Clearing the active plot or starting another conversation does not remove saved
history. There is no automatic retention policy that deletes old user records.
Copying or exporting a report creates a separate user-controlled document and
does not alter its saved history. The current UI has no destructive history-delete
button; to archive local history, close the application and move the entire
`data/history` directory to a backup location. On the next launch a fresh history
database will be created when needed. Existing original GWAS files are unaffected.
Saved plotting rows remain a visualization sample, not a substitute for the
original summary statistics for analyses requiring all variants.

Only report/data fields are accepted by the storage API. Model settings, API keys,
authorization headers, and private model responses are not history fields.
Credential-named fields are stripped recursively; recognizable pasted credential
strings are redacted. Application code must never pass provider configuration or
raw provider responses to a history method.

## HTTP interface

All endpoints inherit the application's loopback Host and same-origin checks.

| Operation | Endpoint |
| --- | --- |
| List newest entries | `GET /api/history/{datasets,chats,analyses}?limit=50&offset=0` |
| Read one saved entry | `GET /api/history/{kind}/{id}` |
| Restore a saved entry and its linked dataset | `POST /api/history/{kind}/{id}/restore` with no body |
| Save a conversation | `POST /api/history/chats` |
| Update a saved conversation | `POST /api/history/chats/{id}` |
| Save an Auto Analysis snapshot | `POST /api/history/analyses` |
| Update an Auto Analysis snapshot | `POST /api/history/analyses/{id}` |

Identifiers are opaque 32-character lowercase hexadecimal values. No endpoint
accepts a filesystem path. Listing returns only small metadata records; plotting
rows and report text are loaded only when an entry is opened. The list page size
is limited to 100. Dataset history is created by the validated `/upload-processed`
workflow, not by an unchecked direct dataset-write endpoint.

Chat save payloads contain `messages` with `role` and `content`. Optional fields
include `title`, `dataset_id`, `source_phenotype`, `selected_variants`,
`selected_region`, `run_id`, `status`, and a bounded public progress snapshot.
The store allows up to 1,000 messages and 32 MB uncompressed per entry, returning
an explicit error instead of silently truncating an old conversation.

Each compressed payload and its index metadata are saved together in one atomic
transaction. Readers detect corrupt payloads and unsupported future schemas.
Old run artifacts that contain only diagnostics are not represented as recovered
conversation text: the original answer must exist to be displayed as history.

Restoration honors the most recent selection even when it is empty, so clearing
the selection does not unexpectedly reactivate an earlier locus. A report-only
history entry clears an unrelated active plotting dataset rather than relabeling
its source phenotype. Auto Analysis follow-ups resolve all original uploaded
observations by saved analysis ID and locus number, independently of the plot.
