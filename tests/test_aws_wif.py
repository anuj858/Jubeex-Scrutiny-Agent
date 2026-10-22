"""ECS WIF must follow refreshed task-role credentials, not startup snapshots."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from extraction_review.aws_wif import EcsCredentialSupplier, ecs_google_credentials


def test_ecs_supplier_uses_refreshed_credentials(monkeypatch) -> None:
    first = SimpleNamespace(access_key="first", secret_key="secret-1", token="one")
    second = SimpleNamespace(access_key="second", secret_key="secret-2", token="two")
    credential_provider = Mock()
    credential_provider.get_frozen_credentials.side_effect = [first, second]
    session = Mock()
    session.get_credentials.return_value = credential_provider
    monkeypatch.setattr("extraction_review.aws_wif.boto3.Session", lambda: session)
    monkeypatch.setenv("AWS_REGION", "ap-south-1")

    supplier = EcsCredentialSupplier()
    assert supplier.get_aws_region(None, None) == "ap-south-1"
    assert supplier.get_aws_security_credentials(None, None).access_key_id == "first"
    assert supplier.get_aws_security_credentials(None, None).access_key_id == "second"


def test_ecs_wif_uses_custom_supplier(tmp_path, monkeypatch) -> None:
    path = tmp_path / "wif.json"
    path.write_text(
        json.dumps(
            {
                "type": "external_account",
                "audience": "//iam.googleapis.com/projects/test/locations/global/pools/p/providers/a",
                "credential_source": {"environment_id": "aws1"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(path))
    monkeypatch.setattr("extraction_review.aws_wif.boto3.Session", Mock())
    from_info = Mock(return_value="credentials")
    monkeypatch.setattr(
        "extraction_review.aws_wif.aws.Credentials.from_info", from_info
    )

    assert ecs_google_credentials() == "credentials"
    info = from_info.call_args.args[0]
    assert "credential_source" not in info
    assert isinstance(
        from_info.call_args.kwargs["aws_security_credentials_supplier"],
        EcsCredentialSupplier,
    )
    assert from_info.call_args.kwargs["scopes"] == [
        "https://www.googleapis.com/auth/cloud-platform"
    ]
