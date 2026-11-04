"""Renew trusted workspace S3 download URLs when queued work starts."""

from __future__ import annotations

import os
from urllib.parse import parse_qs, unquote, urlsplit

from .job_progress import _bucket, _s3_client

_SIGNED_QUERY_FIELDS = (
    "X-Amz-Algorithm",
    "X-Amz-Credential",
    "X-Amz-Date",
    "X-Amz-SignedHeaders",
    "X-Amz-Signature",
)
_SOURCE_FOLDERS = ("document/", "splitfiles/")


def renew_workspace_s3_url(
    url: str, *, organization_id: str | None, workspace_id: str | None
) -> str:
    """Re-sign only this workspace's S3 document URL using the worker role.

    SQS can deliver a job after the backend's 15-minute URL has expired. An
    unrelated or third-party URL keeps its original download behaviour.
    """
    bucket = _bucket()
    region = (os.getenv("AWS_REGION") or "ap-south-1").strip()
    if not bucket or not organization_id or not workspace_id:
        return url
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
        return url
    host = (parsed.hostname or "").lower()
    path_host = f"s3.{region}.amazonaws.com"
    virtual_host = f"{bucket}.s3.{region}.amazonaws.com"
    if host == path_host:
        bucket_prefix = f"/{bucket}/"
        if not parsed.path.startswith(bucket_prefix):
            return url
        key = unquote(parsed.path[len(bucket_prefix) :])
    elif host == virtual_host:
        key = unquote(parsed.path.lstrip("/"))
    else:
        return url
    query = parse_qs(parsed.query)
    if not all(query.get(field) for field in _SIGNED_QUERY_FIELDS):
        return url
    workspace_prefix = f"org/{organization_id}/filing-workspace/{workspace_id}/"
    if not any(
        key.startswith(workspace_prefix + folder) and key != workspace_prefix + folder
        for folder in _SOURCE_FOLDERS
    ):
        raise ValueError("S3 source URL does not belong to the requested workspace")
    return str(
        _s3_client().generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=900
        )
    )
