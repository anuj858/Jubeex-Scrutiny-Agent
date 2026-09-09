from extraction_review.s3_artifacts import (
    STEP_DEFECTS,
    STEP_EXTRACT,
    STEP_LAYOUT,
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
    key = artifact_key(STEP_EXTRACT, object_id="abc-def")
    assert key == (
        "org/486653bd-9e95-49cb-9cdc-4e73a4db0f25/filing-workspace/"
        "20cb771e-a750-4c11-b490-aa9c5b696d8d/extract/extract.json"
    )


def test_layout_artifact_key_uses_coordinate_folder() -> None:
    set_job_context(
        "job-123",
        "486653bd-9e95-49cb-9cdc-4e73a4db0f25",
        "20cb771e-a750-4c11-b490-aa9c5b696d8d",
    )
    key = artifact_key(STEP_LAYOUT, object_id="abc-def")
    assert key == (
        "org/486653bd-9e95-49cb-9cdc-4e73a4db0f25/filing-workspace/"
        "20cb771e-a750-4c11-b490-aa9c5b696d8d/coordinate/"
        "layout.json"
    )


def test_defects_artifact_key_uses_defect_folder() -> None:
    set_job_context(
        "a736e03a-46c7-41c3-9f0f-69a69b8d852e",
        "36cc5708-56df-4754-8579-55f8faed93b8",
        "e56ab02b-fdde-4a51-8a81-3848110deb53",
    )
    key = artifact_key(STEP_DEFECTS, object_id="1f43c6d6-5919-452a-98b5-984a87945b0a")
    assert key == (
        "org/36cc5708-56df-4754-8579-55f8faed93b8/filing-workspace/"
        "e56ab02b-fdde-4a51-8a81-3848110deb53/defect/"
        "defects.json"
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


def test_upload_step_json_skips_without_bucket(monkeypatch) -> None:
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    monkeypatch.delenv("JUBEEX_ARTIFACT_BUCKET", raising=False)
    set_job_context("job-1", "org-1", "ws-1")
    assert upload_step_json(STEP_EXTRACT, {"ok": True}) is None
    assert recorded_artifacts("job-1") == {}


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
    assert recorded_artifacts("job-9")[STEP_EXTRACT]["url"] == record["url"]
