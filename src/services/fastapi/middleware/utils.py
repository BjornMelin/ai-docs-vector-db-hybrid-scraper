"""Utilities shared by FastAPI middleware.

- client_ip(): safe client IP extraction with optional proxy trust.
- safe_escape(): HTML-escape user-provided strings for log safety.
- body_bytes(): robust response-body extraction.

This module removes duplicated helpers across middleware packages.
"""

from __future__ import annotations

import html
from collections.abc import MutableMapping
from typing import Final

from starlette.requests import Request
from starlette.responses import Response


# Header constants
_H_X_FORWARDED_FOR: Final[str] = "x-forwarded-for"
_H_X_REAL_IP: Final[str] = "x-real-ip"
_H_ACCEPT_ENCODING: Final[str] = "accept-encoding"
_V_ACCEPT_ENCODING: Final[str] = "Accept-Encoding"


def client_ip(request: Request, trust_proxy: bool = False) -> str:
    """Return the best-effort client IP.

    Args:
        request: Starlette/FastAPI request.
        trust_proxy: If True, prefer the first hop in X-Forwarded-For.

    Returns:
        IP string; "unknown" if not available.

    Notes:
        When deploying behind proxies/ingress, enable Uvicorn's proxy headers
        handling or place a dedicated proxy-headers middleware in your stack.
        This function is a simple, explicit policy toggle.
    """
    if trust_proxy:
        fwd = request.headers.get(_H_X_FORWARDED_FOR)
        if fwd:
            return fwd.split(",")[0].strip()

        real = request.headers.get(_H_X_REAL_IP)
        if real:
            return real

    if request.client:
        return request.client.host
    return "unknown"


def safe_escape(value: str | None) -> str | None:
    """Escape a possibly unsafe string for logs."""
    return None if value is None else html.escape(str(value))


def body_bytes(response: Response) -> bytes | None:
    """Return response body as bytes if available.

    Args:
        response: Response possibly holding a body in-memory.

    Returns:
        Bytes or None if streaming/not accessible.
    """
    try:
        return response.body  # type: ignore[attr-defined]
    except AttributeError:
        return None


def ensure_vary_accept_encoding(headers: MutableMapping[str, str]) -> None:
    """Ensure 'Vary: Accept-Encoding' is present in headers.

    Args:
        headers: A mutable headers mapping.
    """
    vary = headers.get("vary", "")
    if "accept-encoding" not in vary.lower():
        headers["vary"] = (
            f"{vary}, {_V_ACCEPT_ENCODING}" if vary else _V_ACCEPT_ENCODING
        )
