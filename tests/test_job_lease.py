import io

import pytest
from botocore.exceptions import ClientError

from extraction_review.job_lease import JobLease


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.version = 0

    def get_object(self, *, Bucket: str, Key: str):
        del Bucket
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body, etag = self.objects[Key]
        return {"Body": io.BytesIO(body), "ETag": etag}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs):
        del Bucket
        current = self.objects.get(Key)
        if kwargs.get("IfNoneMatch") == "*" and current:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if "IfMatch" in kwargs and (not current or current[1] != kwargs["IfMatch"]):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.version += 1
        self.objects[Key] = (Body, str(self.version))


def test_job_lease_excludes_other_worker_until_release(monkeypatch) -> None:
    store = FakeS3()
    monkeypatch.setattr("extraction_review.job_lease._bucket", lambda: "bucket")
    monkeypatch.setattr("extraction_review.job_lease._s3_client", lambda: store)
    first = JobLease("job-1")
    second = JobLease("job-1")

    assert first.acquire() is True
    assert second.acquire() is False
    first.renew()
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    with pytest.raises(RuntimeError, match="ownership was lost"):
        first.renew()
