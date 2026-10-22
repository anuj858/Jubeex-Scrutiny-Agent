#!/bin/sh
set -eu

echo "[startup] Preparing Google WIF credentials..."

# Get WIF JSON from AWS Secrets Manager.
/app/.venv/bin/python - <<'PY'
import boto3

client = boto3.client(
    "secretsmanager",
    region_name="ap-south-1",
)

response = client.get_secret_value(
    SecretId="jubeex/google-wif-credentials"
)

with open("/tmp/google-wif-credentials.json", "w") as f:
    f.write(response["SecretString"])

print("[startup] WIF configuration created")
PY

export GOOGLE_APPLICATION_CREDENTIALS=/tmp/google-wif-credentials.json
echo "[startup] Google WIF configuration ready; AWS task credentials will refresh"

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec llamactl serve --host 0.0.0.0 --port 4501
