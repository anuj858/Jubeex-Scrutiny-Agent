# Deployed batch QA

This command processes every PDF in a directory sequentially against the deployed
Scrutiny Agent API. It stages each source PDF in S3, runs split, extraction, and
the assemble-to-scrutiny job, then writes JSON, CSV, HTML, and Excel reports.

## Configuration

```bash
export QA_API_URL="http://your-scrutiny-agent-host"
export QA_API_KEY="..."                    # only when the API requires it
export QA_INPUT_BUCKET="your-staging-bucket"
export AWS_REGION="ap-south-1"
export QA_FILING_TYPE="SLP_CIVIL"          # optional default for every PDF
```

`QA_API_URL` is the direct Scrutiny Agent base URL. Its `/v1/health` endpoint
must return 200. It is not the full-application `/api/v1` URL.

AWS credentials must be supplied through the normal AWS credential chain. Do not
put access keys in the command or repository.

When `QA_FILING_TYPE` and the per-file manifest value are both absent, the runner
uses `POST /v1/filings` so the deployed classifier selects the filing type. When
a filing type is provided, it uses `POST /v1/filings/split-petition` and skips
classification, matching the application's selected-petition-type journey.

Validate file discovery without uploading or starting paid jobs:

```bash
uv run python scripts/run_deployed_batch_qa.py \
  --input-dir "/absolute/path/to/pdfs" \
  --dry-run
```

Run the deployed workflow:

```bash
uv run python scripts/run_deployed_batch_qa.py \
  --input-dir "/absolute/path/to/pdfs" \
  --report-dir "qa_reports"
```

The command uses concurrency one by design. A failed file is recorded and the
next PDF continues. Staged inputs are retained for audit unless
`--delete-staged-inputs` is provided.

The production worker may keep `SKIP_SLOT_PDF_UPLOAD=true`. When a completed
split contains page ranges but no LlamaCloud `file_id` values, the QA runner
creates the corresponding PDF slices locally and stages them under its QA S3
prefix before starting extraction. This fallback is isolated to the QA run and
does not change the frontend or deployed worker behaviour.

## Expected-results file

Expected results are optional. Without them, the report says `not_reviewed`
instead of claiming that a split or defect is correct.

```json
{
  "files": {
    "Defect File_007.pdf": {
      "filing_type": "SLP_CRIMINAL",
      "special_category": null,
      "split": {
        "listing_proforma": [{"start": 8, "end": 10}],
        "petition": [{"start": 22, "end": 31}]
      },
      "extraction": {
        "data.petitioner_name": "Mubarak Ali"
      },
      "scrutiny": {
        "strict": false,
        "checks": {
          "D-58": "defect_found",
          "D-59": "compliant"
        }
      }
    }
  }
}
```

Pass it with `--expected /path/to/expected.json`. Each file may override the
default filing type and special category. Scrutiny comparison is partial by
default. Set `strict` to `true` only when the expected file contains every
applicable defect code and unexpected returned codes should fail the case.

## Email

AWS SES must have a verified sender and permission to send to the recipient.
Email is disabled unless `--email` is explicitly supplied.

```bash
export QA_REPORT_FROM="verified-sender@jubeex.com"
export QA_REPORT_TO="anuj@jubeex.com"

uv run python scripts/run_deployed_batch_qa.py \
  --input-dir "/absolute/path/to/pdfs" \
  --email
```

The email contains the HTML summary and attaches the Excel report. Detailed raw
case JSON remains under `qa_reports/<run-id>/cases/`.
