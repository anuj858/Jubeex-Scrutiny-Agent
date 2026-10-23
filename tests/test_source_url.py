from __future__ import annotations

import pytest

from extraction_review.source_url import renew_workspace_s3_url

ORG = "36cc5708-56df-4754-8579-55f8faed93b8"
WORKSPACE = "8d610cde-6a84-47df-b4b4-fd04cd206cda"
BUCKET = "jubeex-test-bucket"
KEY = f"org/{ORG}/filing-workspace/{WORKSPACE}/document/file.pdf"
SIGNED_QUERY = (
    "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=test"
    "&X-Amz-Date=20261022T075049Z&X-Amz-SignedHeaders=host"
    "&X-Amz-Signature=expired"
)


class FakeS3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def generate_presigned_url(
        self, operation: str, *, Params: dict[str, object], ExpiresIn: int
    ) -> str:
        self.calls.append((operation, Params, ExpiresIn))
        return "https://fresh.example/signed"


def test_renews_matching_workspace_s3_url(monkeypatch) -> None:
    client = FakeS3()
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setattr("extraction_review.source_url._bucket", lambda: BUCKET)
    monkeypatch.setattr("extraction_review.source_url._s3_client", lambda: client)
    old = f"https://s3.ap-south-1.amazonaws.com/{BUCKET}/{KEY}?{SIGNED_QUERY}"

    result = renew_workspace_s3_url(old, organization_id=ORG, workspace_id=WORKSPACE)

    assert result == "https://fresh.example/signed"
    assert client.calls == [("get_object", {"Bucket": BUCKET, "Key": KEY}, 900)]


def test_renews_virtual_hosted_workspace_s3_url(monkeypatch) -> None:
    client = FakeS3()
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setattr("extraction_review.source_url._bucket", lambda: BUCKET)
    monkeypatch.setattr("extraction_review.source_url._s3_client", lambda: client)
    old = f"https://{BUCKET}.s3.ap-south-1.amazonaws.com/{KEY}?{SIGNED_QUERY}"

    assert (
        renew_workspace_s3_url(old, organization_id=ORG, workspace_id=WORKSPACE)
        == "https://fresh.example/signed"
    )
    assert client.calls == [("get_object", {"Bucket": BUCKET, "Key": KEY}, 900)]


def test_renews_matching_splitfile_url(monkeypatch) -> None:
    client = FakeS3()
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setattr("extraction_review.source_url._bucket", lambda: BUCKET)
    monkeypatch.setattr("extraction_review.source_url._s3_client", lambda: client)
    key = f"org/{ORG}/filing-workspace/{WORKSPACE}/splitfiles/part.pdf"
    old = f"https://s3.ap-south-1.amazonaws.com/{BUCKET}/{key}?{SIGNED_QUERY}"

    assert (
        renew_workspace_s3_url(old, organization_id=ORG, workspace_id=WORKSPACE)
        == "https://fresh.example/signed"
    )
    assert client.calls == [("get_object", {"Bucket": BUCKET, "Key": key}, 900)]


def test_does_not_resign_untrusted_or_unsigned_url(monkeypatch) -> None:
    client = FakeS3()
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setattr("extraction_review.source_url._bucket", lambda: BUCKET)
    monkeypatch.setattr("extraction_review.source_url._s3_client", lambda: client)
    urls = [
        "https://example.com/file.pdf",
        f"https://s3.ap-south-1.amazonaws.com/{BUCKET}/{KEY}",
        f"https://s3.ap-south-1.amazonaws.com/other-bucket/{KEY}?X-Amz-Signature=x",
    ]

    for url in urls:
        assert (
            renew_workspace_s3_url(url, organization_id=ORG, workspace_id=WORKSPACE)
            == url
        )
    signed = f"https://s3.ap-south-1.amazonaws.com/{BUCKET}/{KEY}?{SIGNED_QUERY}"
    assert (
        renew_workspace_s3_url(signed, organization_id=None, workspace_id=WORKSPACE)
        == signed
    )
    assert client.calls == []


def test_rejects_cross_workspace_s3_url(monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setattr("extraction_review.source_url._bucket", lambda: BUCKET)
    url = f"https://{BUCKET}.s3.ap-south-1.amazonaws.com/{KEY}?{SIGNED_QUERY}"

    with pytest.raises(ValueError, match="does not belong"):
        renew_workspace_s3_url(url, organization_id=ORG, workspace_id="other-workspace")
