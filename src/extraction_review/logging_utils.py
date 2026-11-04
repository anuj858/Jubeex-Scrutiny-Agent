"""Logging defaults that avoid leaking signed request URLs."""

import logging
from urllib.parse import urlparse


def safe_url_for_log(url: str) -> str:
    """Return a useful URL identity without credentials or query secrets."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return parsed._replace(netloc=host, params="", query="", fragment="").geturl()


def silence_http_client_request_logs() -> None:
    """Keep dependency request logs from exposing presigned URL queries."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
