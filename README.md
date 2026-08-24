# Jubeex Scrutiny Agent

A LlamaAgents application for classifying and extracting structured data from Indian court petitions (JubeeX filings). It uses **LlamaParse** + **LlamaSplit** for section-aware RAG, LlamaClassify for petition type, LlamaExtract for a shared **Core Filing Record**, stores results in LlamaCloud Agent Data, and indexes summary + section chunks in Pinecone.

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
3. Run locally:

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

`deployment.yaml` maps Pinecone env vars into LlamaCloud deployment secrets so vector indexing works in the cloud.

## Features

- **Parse + Split**: LlamaParse produces per-page markdown; LlamaSplit labels petition sections (COVER_PAGE, PETITION, IMPUGNED_ORDER, AOR_DECLARATION, APPLICATION, ANNEXURE, AFFIDAVIT)
- **Petition classification**: LlamaClassify labels filings as one of:
  - `SLP_CIVIL`
  - `SLP_CRIMINAL`
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
- **Parallel classify + extract**: After parse/split, extraction starts and classification runs alongside
- **Agent Data storage**: Results land in collection `jubeex-filing-extraction`, deduplicated by file hash
- **Pinecone vector index**: Upserts (1) a filing summary vector and (2) per-section chunks from Parse×Split. Embeddings are **not** generated in-app — Pinecone’s integrated model (`llama-text-embed-v2`) embeds server-side
- **Review UI**: Upload filings, watch workflow progress, edit/approve extracted records

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
  - `split` — petition section categories
  - `classify` — rules, FAST mode, first 5 pages for classification
  - `extract-jubeex` — Core Filing Record JSON schema, agentic tier, source citations + confidence scores
- Python models and collection name: `src/extraction_review/config.py`
- Workflow registration: `[tool.llamadeploy]` in `pyproject.toml`

## How it works

1. **Upload**: User uploads a petition PDF through the UI (`process-file` workflow).
2. **Parse**: LlamaParse converts the PDF to per-page markdown.
3. **Split**: LlamaSplit assigns pages to section categories (cover, petition, impugned order, AOR, applications, annexures, affidavits).
4. **Start extraction**: Starts a LlamaExtract job with the JubeeX schema.
5. **Classify (parallel)**: LlamaClassify picks petition type; on failure defaults to `other`.
6. **Complete extraction**: Waits for extract, validates against `CoreFilingRecord`, stamps classification + parse/split metadata.
7. **Store**: Dedupes by `file_hash`, then creates Agent Data in `jubeex-filing-extraction`.
8. **Vector index** (when `VECTOR_BACKEND=pinecone`): Upserts a summary vector plus section chunks (long sections windowed to ~3500 chars); Pinecone embeds with the integrated model.
9. **Review**: UI lists items for review/edit.

Vector helpers: `src/extraction_review/vector_store.py` (`upsert_records`, `build_section_records`, `search_filings`).

### Workflows

| Workflow | Module | Role |
| --- | --- | --- |
| `process-file` | `src/extraction_review/process_file.py` | parse → split → extract → classify → store → Pinecone |
| `metadata` | `src/extraction_review/metadata_workflow.py` | Expose JSON schema, per-type schemas, and collection name to the UI |
| `metadata` | `src/extraction_review/metadata_workflow.py` | Expose JSON schema, per-type schemas, and collection name to the UI |

Progress is streamed to the UI via `Status` events (and extraction result events).

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
