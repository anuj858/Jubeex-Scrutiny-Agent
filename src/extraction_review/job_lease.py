"""S3 conditional lease preventing concurrent execution of one SQS job."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .job_progress import _bucket, _s3_client, job_status_key
from .queue import visibility_timeout

_MISSING = {"404", "NoSuchKey", "NotFound"}
_CONFLICT = {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}


def _error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    detail = response.get("Error") if isinstance(response, dict) else None
    return str(detail.get("Code") or "") if isinstance(detail, dict) else ""


class JobLease:
    def __init__(self, job_id: str) -> None:
        self.bucket = _bucket()
        if not self.bucket:
            raise RuntimeError("An artifact S3 bucket is required for SQS job leases")
        self.key = job_status_key(job_id).replace("/status.json", "/lease.json")
        self.owner = str(uuid.uuid4())
        self.client = _s3_client()
        self.duration = max(visibility_timeout() + 120, 180)

    def _read(self) -> tuple[dict[str, Any], str | None]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key)
        except Exception as exc:
            if _error_code(exc) in _MISSING:
                return {}, None
            raise
        stream = response["Body"]
        try:
            payload = json.loads(stream.read())
        finally:
            stream.close()
        if not isinstance(payload, dict):
            raise TypeError("Invalid job lease payload")
        return payload, str(response["ETag"])

    def _write(self, expires_at: float, etag: str | None) -> None:
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self.key,
            "Body": json.dumps(
                {"owner": self.owner, "expires_at": expires_at},
                separators=(",", ":"),
            ).encode(),
            "ContentType": "application/json",
        }
        if etag:
            kwargs["IfMatch"] = etag
        else:
            kwargs["IfNoneMatch"] = "*"
        self.client.put_object(**kwargs)

    def acquire(self) -> bool:
        for _ in range(20):
            # A missing lease is normal for a new job. Create it conditionally
            # before reading: S3 otherwise returns 403 (rather than 404) for
            # a missing key when the task role cannot list the bucket.
            try:
                self._write(time.time() + self.duration, None)
                return True
            except Exception as exc:
                if _error_code(exc) not in _CONFLICT:
                    raise
            payload, etag = self._read()
            if float(payload.get("expires_at") or 0) > time.time():
                return False
            try:
                self._write(time.time() + self.duration, etag)
                return True
            except Exception as exc:
                if _error_code(exc) not in _CONFLICT:
                    raise
                time.sleep(0.025)
        raise RuntimeError("Could not acquire S3 job lease after contention")

    def renew(self) -> None:
        payload, etag = self._read()
        if payload.get("owner") != self.owner or not etag:
            raise RuntimeError("S3 job lease ownership was lost")
        try:
            self._write(time.time() + self.duration, etag)
        except Exception as exc:
            if _error_code(exc) in _CONFLICT:
                raise RuntimeError("S3 job lease ownership was lost") from exc
            raise

    def release(self) -> None:
        payload, etag = self._read()
        if payload.get("owner") == self.owner and etag:
            try:
                self._write(0, etag)
            except Exception as exc:
                if _error_code(exc) not in _CONFLICT:
                    raise
