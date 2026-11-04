"""HTTP and S3 clients used by the deployed batch QA runner."""

from __future__ import annotations

import mimetypes
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
import httpx
from botocore.config import Config


class QaApiError(RuntimeError):
    """A deployed API request or asynchronous job failed."""


@dataclass
class PolledJob:
    payload: dict[str, Any]
    duration_seconds: float
    observations: list[dict[str, Any]] = field(default_factory=list)


class DeployedApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        request_timeout: float = 60,
        retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self.base_url = base_url.rstrip("/")
        self.retries = max(0, retries)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=request_timeout,
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        for attempt in range(self.retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
                if (
                    response.status_code in {429, 500, 502, 503, 504}
                    and attempt < self.retries
                ):
                    time.sleep(min(2**attempt, 8))
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise QaApiError(f"{method} {path} returned non-object JSON")
                return payload
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= self.retries:
                    raise QaApiError(f"{method} {path} failed: {exc}") from exc
                time.sleep(min(2**attempt, 8))
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:1000]
                raise QaApiError(
                    f"{method} {path} returned {exc.response.status_code}: {detail}"
                ) from exc
        raise QaApiError(f"{method} {path} failed")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health")

    def start_split(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/filings/split-petition", json=payload)

    def start_compiled(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/filings", json=payload)

    def start_extract(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/filings/extract", json=payload)

    def start_index(
        self, agent_data_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request("POST", f"/v1/filings/{agent_data_id}/index", json=payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/jobs/{job_id}")

    def fetch_json_url(self, url: str) -> dict[str, Any]:
        try:
            response = httpx.get(
                url, timeout=self._client.timeout, follow_redirects=True
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise QaApiError(f"Could not download JSON artifact: {exc}") from exc
        if not isinstance(payload, dict):
            raise QaApiError("JSON artifact is not an object")
        return payload

    def poll_job(
        self,
        job_id: str,
        *,
        poll_interval: float,
        timeout_seconds: float,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> PolledJob:
        started = time.monotonic()
        observations: list[dict[str, Any]] = []
        previous: tuple[Any, ...] | None = None
        while True:
            payload = self.get_job(job_id)
            snapshot = (
                payload.get("status"),
                payload.get("progress"),
                payload.get("stage"),
                payload.get("stage_message"),
            )
            elapsed = round(time.monotonic() - started, 3)
            if snapshot != previous:
                observation = {
                    "elapsed_seconds": elapsed,
                    "status": snapshot[0],
                    "progress": snapshot[1],
                    "stage": snapshot[2],
                    "message": snapshot[3],
                }
                observations.append(observation)
                if on_progress:
                    on_progress(observation)
                previous = snapshot
            status = str(payload.get("status") or "").lower()
            if status == "completed":
                return PolledJob(payload, elapsed, observations)
            if status == "failed":
                raise QaApiError(str(payload.get("error") or f"Job {job_id} failed"))
            if elapsed >= timeout_seconds:
                raise QaApiError(
                    f"Job {job_id} timed out after {timeout_seconds:.0f} seconds"
                )
            time.sleep(max(0.1, poll_interval))


class S3InputStager:
    def __init__(self, bucket: str, *, region: str, prefix: str) -> None:
        self.bucket = bucket
        self.region = region
        self.prefix = prefix.strip("/")
        # Force a regional endpoint. Some credential/config combinations make
        # boto3 presign the legacy global S3 endpoint even when ``region_name``
        # is supplied. S3 then redirects the GET to the bucket's regional
        # endpoint, changing the signed Host header and invalidating the URL.
        self.client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
            ),
        )

    def upload(self, path: Path, *, case_id: str) -> tuple[str, str]:
        key = f"{self.prefix}/{case_id}/{path.name}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/pdf"
        self.client.upload_file(
            str(path),
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=43200,
        )
        return key, url

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)
