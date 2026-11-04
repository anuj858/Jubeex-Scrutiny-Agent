import logging
import os

import httpx
from llama_cloud import AsyncLlamaCloud

logger = logging.getLogger(__name__)

# deployed agents may infer their name from the deployment name
agent_name = os.getenv("LLAMA_DEPLOY_DEPLOYMENT_NAME")
# required for all llama cloud calls
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
# get this in case running against a different environment than production
base_url = os.getenv("LLAMA_CLOUD_BASE_URL")
project_id = os.getenv("LLAMA_DEPLOY_PROJECT_ID")

DEFAULT_LLAMA_CLOUD_REQUEST_TIMEOUT_S = 900.0


def llama_cloud_request_timeout_seconds() -> float:
    """Allow large PDF uploads more time than the SDK's 60-second default."""
    raw = (os.getenv("LLAMA_CLOUD_REQUEST_TIMEOUT_S") or "").strip()
    if not raw:
        return DEFAULT_LLAMA_CLOUD_REQUEST_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_LLAMA_CLOUD_REQUEST_TIMEOUT_S
    return value if 60.0 <= value <= 3600.0 else DEFAULT_LLAMA_CLOUD_REQUEST_TIMEOUT_S


def get_llama_cloud_client() -> AsyncLlamaCloud:
    """Cloud services connection for file storage and processing."""
    request_timeout = llama_cloud_request_timeout_seconds()
    return AsyncLlamaCloud(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(
            connect=30.0,
            read=request_timeout,
            write=request_timeout,
            pool=60.0,
        ),
        default_headers={"Project-Id": project_id} if project_id else {},
    )
