# Jubeex Scrutiny Agent

A LlamaAgents application for classifying and extracting structured data from Indian court petitions (JubeeX filings). It uses **LlamaParse** for page markdown, LlamaClassify for petition type, LlamaExtract for a shared **Core Filing Record**, stores results in LlamaCloud Agent Data, and indexes summary + page chunks in Pinecone.

# Running the application

This is a LlamaAgents starter. See the [LlamaAgents (llamactl) getting started guide](https://developers.llamaindex.ai/python/llamaagents/llamactl/getting-started/) for local development and deployment.

1. Clone this repo and install [`uv`](https://docs.astral.sh/uv/).
2. Set environment variables (typically in `.env`):
   - `LLAMA_CLOUD_API_KEY` — required for LlamaCloud classify/extract
   - `LLAMA_DEPLOY_PROJECT_ID` — optional project scope
   - `LLAMA_CLOUD_BASE_URL` — optional non-production endpoint
   - `LLAMA_DEPLOY_DEPLOYMENT_NAME` — set automatically when deployed
   - `PINECONE_API_KEY`, `PINECONE_INDEX`, `PINECONE_CLOUD`, `PINECONE_REGION` — vector DB
   - `PINECONE_EMBED_MODEL` — integrated embed model (default `llama-text-embed-v2`)
   - `PINECONE_TEXT_FIELD` — record text field mapped for embedding (this index uses `normalized_text`)
   - `PINECONE_NAMESPACE` — namespace for filing vectors (default `jubeex-filings`)
   - `VECTOR_BACKEND` — `pinecone` to index after extract, or anything else to skip
3. Run the **production API** (what your other application should call):

```bash
uv run jubeex-api
# or: uv run uvicorn extraction_review.api:app --host 0.0.0.0 --port 8000
```

OpenAPI: `http://localhost:8000/docs`.

Postman: import `postman/JubeeX-Scrutiny-API.postman_collection.json` and optionally `postman/local.postman_environment.json`. Set `baseUrl` (and `apiKey` if `JUBEEX_API_KEY` is set). Start compiled or split, then **Get job status** — `job_id` and `agent_data_id` are saved automatically.

Run the Llama UI / workflow debugger locally:

```bash
uvx llamactl serve
```

Deploy to [LlamaCloud](https://cloud.llamaindex.ai) via the UI, or with `llamactl`:

```bash
# 1. Push main to GitHub (never commit .env)
# 2. Load local secrets, then apply deployment.yaml
set -a && source .env && set +a
uvx llamactl auth login
uvx llamactl deployments apply -f deployment.yaml
```

`deployment.yaml` maps Pinecone and S3 env vars into LlamaCloud deployment secrets. A local `.env` is not uploaded (it is gitignored), so hosted agents only see secrets listed there or in `[tool.llamadeploy.env]`.

## Compiled splitting: optional local OCR, repair and audit

LlamaCloud Split uses `split.categories` and `split.splitting_strategy` in
`configs/config.json`. Each category description uses IDENTITY, START,
CONTINUE, END, DO NOT SPLIT and REPEAT sections; shared instructions define
nesting, physical-page ownership and Index/folio handling. The configured
`min_pages_per_split: 1` disables size-based merging of short documents, as
documented in the [Split API reference](https://developers.llamaindex.ai/reference/resources/split/methods/create/).
Instructions over 5,000 characters, category names over 200 characters and
descriptions over 2,000 characters fail before submitting an inline split job;
instructions are never silently truncated.

`SPLIT_RECONCILIATION_MODE` selects compiled-upload and split-petition behavior:

- `llama_only`: LlamaCloud only. No local OCR, heading/Index/folio rules,
  structure detection, repairs, duplicate filtering or audits run.
- `targeted`: Local text/layout extraction, OCR and repairs run, but label
  changes are limited to missing/ambiguous pages or supported outer boundaries
  and Index/folio evidence. Local page reading runs alongside the remote split.
- `legacy_full`: Apply the full local repair result, for regression comparison.

A valid mode overrides the compatibility flag `SPLIT_PYTHON_RULES_ENABLED`.
Otherwise that flag selects `targeted` for `true`, `1` or `yes` and `llama_only`
for other values or when unset. Values are case-insensitive. The mode is
captured once per split job. `.env.template` explicitly selects `targeted`.
`SPLIT_OCR_CONCURRENCY` only tunes local OCR concurrency; it is not another
enable/disable switch.

In either mode, Python downloads the source PDF, maps document categories to
configured upload slots, copies pages and uploads the resulting PDFs. Pages
without an assigned slot are retained as Unidentified.

In every mode, separate generic `Annexures` and `Application` segments remain
separate files when their pages retain those generic labels. The slicer uses
remote boundaries only within the final assignments: reclassified pages and
numbered local document identities are not overwritten. Newly assigned generic
pages outside matching remote segments are retained in separate contiguous runs.
Generic annexures use `Annexure 1`, `Annexure 2`, etc. (`annexure_1`, `annexure_2`
upload IDs): these are segment ordinals, not inferred printed P/A/R numbers.
Explicit numbered categories returned by LlamaCloud are preserved. Existing
catalog grouping (for example Synopsis + LOD and Vakalatnama + Memo of Appearance)
is retained for frontend compatibility; it does not reclassify page content.

Extraction, separate-file verification and scrutiny checks are unchanged.
Deploy the updated ingestion worker image, set the mode on that worker's
environment (for ECS, its task definition), and restart/roll out the worker.
Editing your laptop's `.env` does not change a deployed ECS worker. Start a
fresh split; existing split artifacts are not rewritten. Look for
`[SplitMode] mode=llama_only`, `mode=targeted` or `mode=legacy_full` and the matching
`[SplitTiming]` entries in worker logs. Enabled mode requires Tesseract and its
language data, which are included in the repository's worker Dockerfiles.

## Features

- **Parse**: LlamaParse produces per-page markdown used for Pinecone page chunks
- **Petition classification**: LlamaClassify labels filings as one of:
  - `SLP_CIVIL`
  - `SLP_CRIMINAL`
  - `TRANSFER_PETITION_CIVIL`
  - `TRANSFER_PETITION_CRIMINAL`
  - `ARBITRATION_PETITION`
  - `WRIT_PETITION_CIVIL`
  - `WRIT_PETITION_CRIMINAL`
  - `other`
- **Core Filing Record extraction**: One schema for all types (`CoreFilingRecord`), including:
  - Court, petition type, special category
  - Cause title (raw + formatted petitioner vs respondent)
  - Petitioners and respondents (identity, address, relations)
  - Matter classification (category / subcategory / PIL)
  - Impugned order details
  - Advocate on Record (AOR) contact and registration
  - Filing summary (documents, annexures, applications, review estimate)
- **Parallel classify + extract**: After parse, extraction starts and classification runs alongside
- **Agent Data storage**: Results land in collection `jubeex-filing-extraction`, deduplicated by file hash
- **Pinecone vector index**: Upserts (1) a filing summary vector and (2) per-page chunks from Parse. Embeddings are **not** generated in-app — Pinecone’s integrated model (`llama-text-embed-v2`) embeds server-side
- **Review UI**: Upload filings, watch workflow progress, edit/approve extracted records
- **Defect check**: `POST /v1/filings/{id}/scrutiny` (or UI `scrutiny-check`) runs after extract. Agent Data is usually `pending_review`; approval is not required. Report is stored on the same Agent Data item.

How the live path is wired (diagrams for non-developers): [docs/how-it-works.md](docs/how-it-works.md). Agent orientation for code changes: [AGENTS.md](AGENTS.md).

## Project layout

```
configs/config.json          # Classify rules + extract-jubeex schema
src/extraction_review/
  process_file.py            # Main process-file workflow
  metadata_workflow.py       # Schema metadata for the UI
  config.py                  # Pydantic models and filing types
  clients.py                 # LlamaCloud client / env wiring
  vector_store.py            # Pinecone upsert/search (integrated embeddings)
  json_util.py               # Schema helpers
ui/                          # Vite + React review app
tests/                       # Workflow tests
pyproject.toml               # Package + llamadeploy workflow/UI config
```

## Configuration

- Runtime settings: `configs/config.json`
  - `parse` — LlamaParse tier/version
  - `classify` — rules, FAST mode, first 5 pages for classification
  - `extract-jubeex` — Core Filing Record JSON schema, agentic tier, source citations + confidence scores
  - `split` — LlamaSplit categories; required when indexing Pinecone (`document_part` labels). A Split failure stops `process-file`.
- Python models and collection name: `src/extraction_review/config.py`
- Workflow registration: `[tool.llamadeploy]` in `pyproject.toml`

## How it works

1. **Upload / ingest**: UI uploads a PDF to LlamaCloud, **or** your backend starts `process-file` with `file_url` (S3/HTTPS). The workflow downloads the PDF, uploads it to LlamaCloud, then classifies and splits.
2. **Parse**: LlamaParse converts the PDF to per-page markdown.
3. **Start extraction**: Starts a LlamaExtract job with the JubeeX schema.
4. **Classify (parallel)**: LlamaClassify picks petition type; on failure defaults to `other`.
5. **Complete extraction**: Waits for extract, validates against `CoreFilingRecord`, stamps classification + parse metadata.
6. **Split** (when Pinecone is on): LlamaSplit labels each page. Empty or failed Split stops the workflow before Agent Data is saved.
7. **Store**: Dedupes by `file_hash`, then creates Agent Data in `jubeex-filing-extraction`.
8. **Vector index**: Upserts a summary vector plus page chunks with `document_part`. Pinecone embeds with the integrated model.
9. **Review**: UI lists items for review/edit.

Vector helpers: `src/extraction_review/vector_store.py` (`upsert_records`, `build_page_records`, `gather_filing_evidence_pool`).

### Workflows

| Workflow | Module | Role |
| --- | --- | --- |
| `process-file` | `src/extraction_review/process_file.py` | Backend entry: `job_type=upload_compiled` classifies/slices a compiled PDF then runs extract; `job_type=upload_separate` runs extract on labeled files |
| `process-split-files` | `src/extraction_review/process_split_files.py` | extract / overlay / Agent Data / Pinecone (UI submit, or nested from `process-file`) |
| `metadata` | `src/extraction_review/metadata_workflow.py` | Expose JSON schema, per-type schemas, and collection name to the UI |
| `scrutiny-check` | `src/extraction_review/scrutiny_workflow.py` | Extracted filings (`pending_review` or `approved`): Pinecone excerpts, OpenRouter per defect, save on same Agent Data item |

### Production API (other applications)

Your backend should call this FastAPI, not LlamaDeploy `/workflows/.../run-nowait`. Set `JUBEEX_API_KEY` in production and send it as `Authorization: Bearer …` or `X-API-Key`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/health` | Liveness (no auth) |
| `GET` | `/v1/catalog` | Filing types and document slots |
| `POST` | `/v1/filings` | Start compiled or separate-file processing |
| `GET` | `/v1/jobs/{job_id}` | Poll until `completed` or `failed` |
| `PATCH` | `/v1/filings/{agent_data_id}` | Set review status (`approved` / `rejected` / `pending_review`) |
| `POST` | `/v1/filings/{agent_data_id}/scrutiny` | Start defect check |
| `GET` | `/v1/jobs/{job_id}` | Poll scrutiny (same jobs resource) |

**1. Start a filing** — body is the payload itself (no `start_event` wrapper):

```json
POST /v1/filings
{
  "job_type": "upload_separate",
  "filing_type": "SLP_CIVIL",
  "organization_id": "3f2a9c1e-8b44-4d21-9a70-1c8d4e6b2f11",
  "workspace_id": "b20c7d91-4e55-48aa-a013-9d6e2f88c104",
  "user_id": "7b12e4aa-0d55-4c91-b3e8-2a6f19c8d447",
  "documents": [
    {
      "name": "01_Petition.pdf",
      "document_id": "11aa22bb-33cc-44dd-85ee-66ff77889900",
      "download_url": "https://storage.example/01_Petition.pdf"
    }
  ]
}
```

`202` response:

```json
{ "job_id": "…", "status": "accepted", "poll_url": "/v1/jobs/…" }
```

Use `job_type: "upload_compiled"` with one compiled PDF in `documents` for the full-petition path (classify, slice, then extract). Send `filing_type` (`SLP_CIVIL`, `SLP_CRIMINAL`, `TRANSFER_PETITION_CIVIL`, or `TRANSFER_PETITION_CRIMINAL`) when you know the type so slicing still runs if classify returns `other`.

**2. Poll** `GET /v1/jobs/{job_id}` until `status` is `completed`. Then read `agent_data_id` (`agd-…`). `organization_id`, `workspace_id`, `user_id`, and `documents` are on `result`. Extracted JSON is on the webhook / S3 artifacts — there is no `GET /v1/filings/{id}`.

**3. Approve (optional)** `PATCH /v1/filings/{agent_data_id}` — no Llama UI:

```json
{ "status": "approved" }
```

**4. Run scrutiny** `POST /v1/filings/{agent_data_id}/scrutiny` with an optional body:

```json
{
  "file_hash": "8adf76ba0ba44d4561ab5a4ad88e3d6e97e56a32010ac8c08f39b5f8c1d01340",
  "file_url": "https://storage.example/filings/Defect_SLP_Civil.pdf"
}
```

Send both, either, or `{}`. `download_url` is accepted as an alias of `file_url`. Then poll `GET /v1/jobs/{job_id}` again. The report is `result.report`.

Job state is in-memory on this process. Poll until complete; do not assume jobs survive a restart.

### Backend integration (LlamaDeploy UI / debugger)

The Llama UI still uses `POST .../workflows/process-file/run-nowait` with a `start_event` wrapper. Applications should use `/v1` above instead. Do not send `"handler_id": "string"`.

## Linting and type checking

Python:

```bash
uv run hatch run lint
uv run hatch run typecheck
uv run hatch run test
# run all at once
uv run hatch run all-fix
```

JavaScript (from `ui/`):

```bash
pnpm run lint
pnpm run format
pnpm run build
# run all at once
pnpm run all-fix
```
