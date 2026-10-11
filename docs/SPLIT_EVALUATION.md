# Reproducible compiled-PDF split evaluation

This harness scores the output of the existing compiled filing pipeline. It
does not implement a second splitter or modify production behavior.

## Dataset authorization and privacy

Use only a manually reviewed petition set that has explicit authorization for
this evaluation. Keep source PDFs and annotation files in access-controlled
storage. Use a stable pseudonymous `case_id`; do not use cause titles, party
names, filing numbers, or other personal identifiers. Reviewer notes must not
contain petition excerpts or personal data. The evaluator never copies notes,
filenames, extracted text, OCR, or other page content into its report.

At implementation time, no ground-truth annotation set was found in the
repository. Some PDF/replay artifacts exist under temporary workspace folders,
but they are not identified as authorized, manually reviewed ground truth and
were not opened or used. Real-world accuracy therefore remains unmeasured.

## Annotation contract

Use `evaluation/split-ground-truth.schema.json` and start from
`evaluation/split-ground-truth.template.json`. The root schema version is
`jubeex.split-ground-truth.v1`. Every case needs a non-sensitive case ID, a
source PDF identifier or SHA-256, the physical PDF page count, and documents
with stable document IDs, exact canonical categories, and inclusive 1-based
physical page ranges. Optional `input_quality`, reviewer notes, and ambiguity
flags are supported.

Validation also enforces unique case/document IDs, supported canonical
categories, in-range endpoints, no duplicate ranges, no overlaps, and exact
page coverage. The production contract is single-owner: annotate uncertain
pages as `Unidentified` rather than leaving holes or double-assigning pages.
There is no overlap/gap exception in this version. Ambiguity flags are
reported, but selected labels remain scored; reviewers should resolve to the
best supported owner/type before inclusion.

## Run and capture production output

Use the `--run-existing-pipeline` mode to submit each authorized PDF through
the documented production entry point (`POST /v1/filings`,
`job_type: "upload_compiled"`) and capture its persisted `STEP_SPLIT` artifact.
The separate `--sources` JSON maps each case to an authorized, fetchable PDF
URL, known filing type, and a dedicated evaluation organization/workspace:

```json
{
  "reviewed_case_0001": {
    "download_url": "https://authorized-storage.example/signed-case-url",
    "filing_type": "SLP_CIVIL",
    "organization_id": "dedicated-evaluation-organization",
    "workspace_id": "dedicated-evaluation-workspace"
  }
}
```

Run only after obtaining authorization, configuring `JUBEEX_API_KEY`, and
explicitly enabling both remote-run flags. The runner refuses remote operation
without the API key even if the endpoint happens to permit unauthenticated
requests:

```bash
PYTHONPATH=src ./.venv/bin/python -m extraction_review.split_eval \
  --annotations /secure/reviewed/split-ground-truth.json \
  --predictions-dir /secure/production-split-artifacts \
  --output /secure/evaluation-results/split-report.json \
  --run-existing-pipeline --allow-remote-run \
  --api-base-url https://authorized-agent-api.example \
  --sources /secure/reviewed/source-map.json
```

This explicit mode uploads/processes the source through the configured API and
creates filing/split artifacts in the supplied workspace. Do not point it at
production or use it for confidential files without authorization. To score
already captured output without making remote calls, omit the run flags and
place one production `STEP_SPLIT` artifact at `<predictions-dir>/<case_id>.json`.
The artifact must contain `split_audit.page_manifest`, its immutable source
checksum/page count, and config identity/version. Do not hand-edit predictions.

For each run, retain the artifact's `config` identity and the exact exported
JSON. The report records its checksum, the artifact's config identity, and the
local config version/checksum used to validate canonical categories. Prefer
rerunning all cases on one pinned worker/config release; mixed releases remain
visible per case and should not be pooled without a deliberate analysis plan.

Score a batch:

```bash
PYTHONPATH=src ./.venv/bin/python -m extraction_review.split_eval \
  --annotations /secure/reviewed/split-ground-truth.json \
  --predictions-dir /secure/production-split-artifacts \
  --output /secure/evaluation-results/split-report.json
```

For checksum-based annotations, the PDF checksum must match the production
page-manifest checksum. If using `pdf_id`, it must equal the production
`source_document_id`; use SHA-256 if a local curation ID does not match the
pipeline's source ID.

The report contains only case IDs, checksums, ranges, canonical categories,
confidence values already present in the artifact, ownership/review metadata,
and aggregate metrics. It also preserves text-free predicted page assignments
and segments for manual review. Protect this report and raw artifacts according
to the source filing's retention rules.

## Metric definitions

- **Boundary precision/recall/F1:** exact cut positions between adjacent
  physical pages (after pages 1 through N-1); there is no cut before page 1 or
  after page N and no ±1 tolerance. Precision divides true predicted cuts by
  predicted cuts; recall divides true predicted cuts by ground-truth cuts.
  F1 is `0` when either side has cuts but there are no true positives, and is
  `null` only when both the prediction and ground truth have no cuts.
- **Exact document-range accuracy:** one-to-one matched predictions whose
  start and end pages both match, divided by ground-truth document count.
  Unmatched expected documents count incorrect.
- **Page document-ownership accuracy:** in-range source pages whose single
  output-slot owner and matched predicted segment map to the exact annotated
  document ID, divided by all physical pages plus distinct out-of-range page
  assignments. Missing pages, multiple slots, repeated manifest rows, and
  wrong-document assignments count incorrect. This checks identity, not just
  category.
- **Canonical type accuracy:** exact category agreement among matched
  documents. Matching is a global one-to-one maximum-overlap assignment, with
  canonical category and exact range as tie-breakers; zero-overlap pairs are
  not matches. Unmatched predictions/annotations remain visible in segment
  and denominator counts.
- Predicted document segments use contiguous reconciled page labels and
  persisted output-slot ownership, so adjacent repeatable output slots remain
  separate even when their generic labels are identical.
- **Missing/multiply assigned/out-of-range pages:** counted from the
  output-slot ownership list and source physical page numbers in the
  production page manifest, separately from semantic metrics. Out-of-range
  page assignments also increase the page-ownership accuracy denominator.
- **Manual-review rate:** predicted segments (and pages) with at least one
  page marked `needs_review`, divided by predicted segments (and physical
  pages). A separate case-level review status/rate reflects the persisted
  overall review flag; a clear segment rate does not override a case-level
  review requirement.
- **Input quality/document type breakdowns:** included when annotations have
  `input_quality`; type breakdowns use expected canonical categories.

Structural validation is reported separately and is never treated as semantic
correctness. Reviewer ambiguity flags are not silently dropped from scoring.
An empty dataset yields null rates and zero counts, not a fabricated score.
