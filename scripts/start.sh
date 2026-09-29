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

# Get the current Fargate task-role credentials.
eval "$(
/app/.venv/bin/python - <<'PY'
import json
import os
import shlex
import urllib.request

relative = os.environ["AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"]
url = "http://169.254.170.2" + relative

with urllib.request.urlopen(url, timeout=5) as response:
    credentials = json.load(response)

print(
    "export AWS_ACCESS_KEY_ID="
    + shlex.quote(credentials["AccessKeyId"])
)
print(
    "export AWS_SECRET_ACCESS_KEY="
    + shlex.quote(credentials["SecretAccessKey"])
)
print(
    "export AWS_SESSION_TOKEN="
    + shlex.quote(credentials["Token"])
)
PY
)"

echo "[startup] Google ADC/WIF environment prepared"

exec llamactl serve --host 0.0.0.0 --port 4501
