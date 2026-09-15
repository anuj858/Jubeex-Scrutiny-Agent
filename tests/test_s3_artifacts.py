from extraction_review.s3_artifacts import (
    FALLBACK_ORGANIZATION_ID,
    FALLBACK_WORKSPACE_ID,
    STEP_DEFECTS,
    STEP_EXTRACT,
    STEP_LAYOUT,
    STEP_SPLIT,
    artifact_key,
    recorded_artifacts,
    set_job_context,
    upload_step_json,
)


def test_artifact_key_matches_filing_workspace_layout() -> None:
    set_job_context(
        "job-123",
        "486653bd-9e95-49cb-9cdc-4e73a4db0f25",
        "20cb771e-a750-4c11-b490-aa9c5b696d8d",
    )
    key = artifact_key(STEP_EXTRACT, object_id="abc-def", job_id="job-123")
    assert key == (
        "org/486653bd-9e95-49cb-9cdc-4e73a4db0f25/filing-workspace/"
        "20cb771e-a750-4c11-b490-aa9c5b696d8d/extractedfiles/"
        "abc-def-v001-agent-extract-job-123.json"
    )


def test_split_defect_and_layout_use_their_own_folders() -> None:
    org = "36cc5708-56df-4754-8579-55f8faed93b8"
    workspace = "e56ab02b-fdde-4a51-8a81-3848110deb53"
    object_id = "1f43c6d6-5919-452a-98b5-984a87945b0a"
    job_id = "a736e03a-46c7-41c3-9f0f-69a69b8d852e"
    set_job_context(job_id, org, workspace)
    assert artifact_key(STEP_SPLIT, object_id=object_id, job_id=job_id) == (
        f"org/{org}/filing-workspace/{workspace}/splitfiles/"
        f"{object_id}-v001-agent-split-{job_id}.json"
    )
    assert artifact_key(STEP_DEFECTS, object_id=object_id, job_id=job_id) == (
        f"org/{org}/filing-workspace/{workspace}/defectfiles/"
        f"{object_id}-v001-agent-defects-{job_id}.json"
    )
    assert artifact_key(STEP_LAYOUT, object_id=object_id, job_id=job_id) == (
        f"org/{org}/filing-workspace/{workspace}/layoutfiles/"
        f"{object_id}-v001-agent-layout-{job_id}.json"
    )


def test_upload_step_json_writes_object_and_records_url(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeS3:
        def put_object(self, **kwargs):
            calls["put"] = kwargs

        def generate_presigned_url(self, method, Params, ExpiresIn):
            calls["presign"] = (method, Params, ExpiresIn)
            return "https://s3.example/extract.json"

    monkeypatch.setenv("AWS_S3_BUCKET", "jubeex-893338224943-ap-south-1-an")
    monkeypatch.setattr(
        "extraction_review.s3_artifacts._s3_client",
        lambda: FakeS3(),
    )
    set_job_context("job-1", "org-1", "ws-1")
    record = upload_step_json(STEP_EXTRACT, {"court": "SCI"})
    assert record is not None
    assert record["url"] == "https://s3.example/extract.json"
    assert record["bucket"] == "jubeex-893338224943-ap-south-1-an"
    assert recorded_artifacts("job-1")[STEP_EXTRACT]["url"] == record["url"]
    put = calls["put"]
    assert put["ContentType"] == "application/json"
    assert b'"court"' in put["Body"]
    assert "/extractedfiles/" in put["Key"]
    assert "agent-extract" in put["Key"]


def test_upload_step_json_skips_without_bucket(monkeypatch) -> None:
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    monkeypatch.delenv("JUBEEX_ARTIFACT_BUCKET", raising=False)
    set_job_context("job-1", "org-1", "ws-1")
    assert upload_step_json(STEP_EXTRACT, {"ok": True}) is None
    assert recorded_artifacts("job-1") == {}


def test_upload_step_json_uses_jubeex_bucket_alias(monkeypatch) -> None:
    class FakeS3:
        def put_object(self, **kwargs):
            return None

        def generate_presigned_url(self, method, Params, ExpiresIn):
            return "https://s3.example/extract.json"

    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    monkeypatch.setenv("JUBEEX_ARTIFACT_BUCKET", "jubeex-alias-bucket")
    monkeypatch.setattr(
        "extraction_review.s3_artifacts._s3_client",
        lambda: FakeS3(),
    )
    set_job_context("job-alias", "org-1", "ws-1")
    record = upload_step_json(STEP_EXTRACT, {"court": "SCI"})
    assert record is not None
    assert record["bucket"] == "jubeex-alias-bucket"


def test_upload_step_json_uses_explicit_ids_without_context(monkeypatch) -> None:
    class FakeS3:
        def put_object(self, **kwargs):
            return None

        def generate_presigned_url(self, method, Params, ExpiresIn):
            return f"https://s3.example/{Params['Key']}"

    monkeypatch.setenv("AWS_S3_BUCKET", "jubeex-893338224943-ap-south-1-an")
    monkeypatch.setattr(
        "extraction_review.s3_artifacts._s3_client",
        lambda: FakeS3(),
    )
    record = upload_step_json(
        STEP_EXTRACT,
        {"court": "SCI", "organization_id": "org-9", "workspace_id": "ws-9"},
        organization_id="org-9",
        workspace_id="ws-9",
        job_id="job-9",
    )
    assert record is not None
    assert "org-9" in record["key"]
    assert "/extractedfiles/" in record["key"]
    assert recorded_artifacts("job-9")[STEP_EXTRACT]["url"] == record["url"]


def test_upload_step_json_without_org_workspace_still_uploads(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeS3:
        def put_object(self, **kwargs):
            calls["put"] = kwargs

        def generate_presigned_url(self, method, Params, ExpiresIn):
            return f"https://s3.example/{Params['Key']}"

    monkeypatch.setenv("AWS_S3_BUCKET", "jubeex-893338224943-ap-south-1-an")
    monkeypatch.setattr(
        "extraction_review.s3_artifacts._s3_client",
        lambda: FakeS3(),
    )
    set_job_context(None, None, None)
    record = upload_step_json(
        STEP_LAYOUT,
        {"schema": "layout_index_v1", "pages": {"34": {"words": []}}},
        job_id="job-ui-1",
    )
    assert record is not None
    assert FALLBACK_ORGANIZATION_ID in record["key"]
    assert FALLBACK_WORKSPACE_ID in record["key"]
    assert "/layoutfiles/" in record["key"]
    put = calls["put"]
    assert put["Bucket"] == "jubeex-893338224943-ap-south-1-an"
    assert put["Key"] == record["key"]


def test_download_json_object_reads_s3_key(monkeypatch) -> None:
    from extraction_review.s3_artifacts import download_json_object

    class FakeBody:
        def read(self) -> bytes:
            return b'{"schema": "layout_index_v1", "pages": {"34": {"words": []}}}'

    class FakeS3:
        def get_object(self, **kwargs):
            assert kwargs["Bucket"] == "jubeex-893338224943-ap-south-1-an"
            assert kwargs["Key"] == "org/llamacloud/filing-workspace/default/layoutfiles/x.json"
            return {"Body": FakeBody()}

    monkeypatch.setenv("AWS_S3_BUCKET", "jubeex-893338224943-ap-south-1-an")
    monkeypatch.setattr(
        "extraction_review.s3_artifacts._s3_client",
        lambda: FakeS3(),
    )
    payload = download_json_object(
        "org/llamacloud/filing-workspace/default/layoutfiles/x.json"
    )
    assert payload is not None
    assert payload["schema"] == "layout_index_v1"
    assert "34" in payload["pages"]


def test_legal_extract_record_treats_null_lists_as_empty() -> None:
    from extraction_review.config import LegalExtractRecord

    record = LegalExtractRecord.model_validate(
        {
            "court": "Supreme Court of India",
            "impugned_orders": None,
            "petitioners": None,
            "respondents": None,
            "advocates_on_record": None,
        }
    )
    assert record.impugned_orders == []
    assert record.petitioners == []
    assert record.respondents == []
    assert record.advocates_on_record == []
