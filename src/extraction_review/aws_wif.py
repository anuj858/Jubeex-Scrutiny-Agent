"""Refreshable AWS task-role credentials for Google workload identity federation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3
from google.auth import aws


class EcsCredentialSupplier(aws.AwsSecurityCredentialsSupplier):
    """Use botocore's refreshing ECS provider for every Google token exchange."""

    def __init__(self) -> None:
        self._session = boto3.Session()

    def get_aws_region(self, context: Any, request: Any) -> str:
        del context, request
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if not region:
            raise RuntimeError("AWS_REGION is required for ECS workload identity")
        return region

    def get_aws_security_credentials(
        self, context: Any, request: Any
    ) -> aws.AwsSecurityCredentials:
        del context, request
        credentials = self._session.get_credentials()
        if credentials is None:
            raise RuntimeError("ECS task-role credentials are unavailable")
        current = credentials.get_frozen_credentials()
        return aws.AwsSecurityCredentials(
            current.access_key,
            current.secret_key,
            current.token,
        )


def ecs_google_credentials() -> aws.Credentials | None:
    """Build Google WIF credentials only when running under an ECS task role."""
    if not os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
        return None
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is required on ECS")
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    if info.get("type") != "external_account" or not str(
        (info.get("credential_source") or {}).get("environment_id", "")
    ).startswith("aws"):
        raise RuntimeError("ECS requires an AWS external-account WIF configuration")
    # Google auth's default AWS supplier queries EC2 IMDS, not ECS task-role
    # credentials. A custom supplier delegates refresh to botocore instead.
    info.pop("credential_source", None)
    return aws.Credentials.from_info(
        info,
        aws_security_credentials_supplier=EcsCredentialSupplier(),
    )
