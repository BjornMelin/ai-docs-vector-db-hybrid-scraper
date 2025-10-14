"""
Correlation ID utilities and integration.

Uses `asgi-correlation-id` to propagate IDs and enrich logs; falls back to UUIDs
if the library is not present.

Primary reference: asgi-correlation-id docs (header names, filter usage).
"""

from __future__ import annotations

import uuid

from starlette.requests import Request


try:
    # As provided by the library.
    from asgi_correlation_id import correlation_id as _cid  # type: ignore
except Exception:  # Library optional at runtime
    _cid = None


def get_correlation_id(_request: Request) -> str:
    """Return the current correlation/request ID.

    If `asgi-correlation-id` is installed and middleware is active, use its
    contextvar. Otherwise generate and return a UUID4 string.

    Returns:
        Correlation ID string.
    """

    if _cid:
        return _cid.get() or str(uuid.uuid4())
    return str(uuid.uuid4())


def set_correlation_id(_request: Request, correlation_id: str) -> None:
    """Set correlation ID in context if the library is present.

    Args:
        _request: Unused; kept for compatibility with previous signatures.
        correlation_id: The value to set.

    Notes:
        This function is a no-op if the library is absent.
    """

    if _cid:
        _cid.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""

    return str(uuid.uuid4())


__all__ = ["generate_correlation_id", "get_correlation_id", "set_correlation_id"]
