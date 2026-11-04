"""Bounded, SSRF-resistant downloads for caller-supplied filing URLs."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from urllib.parse import urljoin, urlsplit

import httpx

from .logging_utils import safe_url_for_log

DEFAULT_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 3


def max_download_bytes() -> int:
    try:
        return max(
            1, int(os.getenv("JUBEEX_MAX_PDF_BYTES", "") or DEFAULT_MAX_DOWNLOAD_BYTES)
        )
    except ValueError:
        return DEFAULT_MAX_DOWNLOAD_BYTES


def _allowed_hosts() -> set[str]:
    return {
        value.strip().casefold()
        for value in (os.getenv("JUBEEX_REMOTE_FILE_HOSTS") or "").split(",")
        if value.strip()
    }


def _address_is_public(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


async def validate_remote_url(url: str) -> str:
    """Allow HTTPS URLs only from configured hosts and reject private DNS targets."""
    parsed = urlsplit((url or "").strip())
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise ValueError("Remote filing URL must use HTTPS")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError(
            "Remote filing URL must not contain credentials or a custom port"
        )
    host = parsed.hostname.casefold().rstrip(".")
    allowed = _allowed_hosts()
    if host not in allowed:
        raise ValueError("Remote filing host is not in JUBEEX_REMOTE_FILE_HOSTS")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not _address_is_public(str(literal)):
            raise ValueError(
                "Remote filing URL resolves to a private or reserved address"
            )
        return parsed.geturl()
    try:
        rows = await asyncio.to_thread(
            socket.getaddrinfo, host, 443, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ValueError("Remote filing host could not be resolved") from exc
    addresses = {str(row[4][0]) for row in rows}
    if not addresses or any(not _address_is_public(address) for address in addresses):
        raise ValueError("Remote filing URL resolves to a private or reserved address")
    return parsed.geturl()


async def _response_bytes(response: object, limit: int) -> bytes:
    headers = getattr(response, "headers", {}) or {}
    raw_length = headers.get("content-length") if hasattr(headers, "get") else None
    if raw_length:
        try:
            if int(raw_length) > limit:
                raise ValueError("Remote PDF exceeds the configured size limit")
        except ValueError as exc:
            if "exceeds" in str(exc):
                raise
    iterator = getattr(response, "aiter_bytes", None)
    if callable(iterator):
        chunks: list[bytes] = []
        total = 0
        async for chunk in iterator():
            total += len(chunk)
            if total > limit:
                raise ValueError("Remote PDF exceeds the configured size limit")
            chunks.append(chunk)
        return b"".join(chunks)
    data = bytes(getattr(response, "content", b"") or b"")
    if len(data) > limit:
        raise ValueError("Remote PDF exceeds the configured size limit")
    return data


async def download_remote_pdf(
    url: str,
    *,
    timeout_seconds: float,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """Download a PDF with validation at every redirect and a hard byte cap."""
    current = await validate_remote_url(url)
    limit = max_download_bytes()
    if client is None:
        timeout = httpx.Timeout(timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as owned:
            return await download_remote_pdf(
                current,
                timeout_seconds=timeout_seconds,
                max_redirects=max_redirects,
                client=owned,
            )
    for redirect_count in range(max_redirects + 1):
        if hasattr(client, "build_request") and hasattr(client, "send"):
            request = client.build_request("GET", current)
            response = await client.send(request, stream=True, follow_redirects=False)
        else:
            response = await client.get(current)
        try:
            if getattr(response, "is_redirect", False):
                if redirect_count >= max_redirects:
                    raise RuntimeError("Remote PDF exceeded the redirect limit")
                location = (getattr(response, "headers", {}) or {}).get("location")
                if not location:
                    raise RuntimeError("Remote PDF redirect omitted Location")
                current = await validate_remote_url(urljoin(current, location))
                continue
            response.raise_for_status()
            content_type = str(
                (getattr(response, "headers", {}) or {}).get("content-type") or ""
            )
            if content_type and not any(
                allowed in content_type.casefold()
                for allowed in ("application/pdf", "application/octet-stream")
            ):
                raise ValueError(
                    f"Remote response from {safe_url_for_log(current)} is not a PDF content type"
                )
            return await _response_bytes(response, limit)
        finally:
            close = getattr(response, "aclose", None)
            if callable(close):
                await close()
    raise RuntimeError("Remote PDF download failed")
