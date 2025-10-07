"""
Security headers and rate-limiting glue.

- Injects conservative security headers.
- Provides a function to enable SlowAPI global rate limits with Redis.

References:
- SlowAPI docs (Limiter, SlowAPIMiddleware).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from slowapi import Limiter  # type: ignore
from slowapi.errors import RateLimitExceeded  # type: ignore
from slowapi.middleware import SlowAPIMiddleware  # type: ignore
from slowapi.util import get_remote_address  # type: ignore
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


_DEFAULT_HEADERS: Mapping[str, str] = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-DNS-Prefetch-Control": "off",
    "X-Download-Options": "noopen",
}


class SecurityMiddleware(BaseHTTPMiddleware):
    """Inject security headers without altering existing values."""

    def __init__(
        self,
        app: Callable,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize with optional extra headers."""

        super().__init__(app)
        self._headers = dict(_DEFAULT_HEADERS)
        if extra_headers:
            self._headers.update(extra_headers)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        for k, v in self._headers.items():
            response.headers.setdefault(k, v)
        return response


def enable_global_rate_limit(
    app,
    *,
    default_limit: str = "100/minute",
    key_func=get_remote_address,
    storage_uri: str | None = None,
) -> Limiter:
    """Enable SlowAPI global rate limit and attach handlers.

    Args:
        app: FastAPI app.
        default_limit: E.g. "100/minute", "10/second".
        key_func: Key extraction function (client identifier).
        storage_uri: Optional Redis URI for distributed limits.

    Returns:
        Initialized Limiter instance.
    """

    limiter = Limiter(
        key_func=key_func, default_limits=[default_limit], storage_uri=storage_uri
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limited)
    app.add_middleware(SlowAPIMiddleware)
    return limiter


async def _rate_limited(_: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Default SlowAPI rate-limit handler."""
    # Safely derive headers to satisfy type checks and preserve metadata if available.
    headers: dict[str, str] = {}
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        headers["Retry-After"] = str(retry_after)
    exc_headers = getattr(exc, "headers", None)
    if isinstance(exc_headers, Mapping):
        headers.update({str(k): str(v) for k, v in exc_headers.items()})

    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded"},
        headers=headers,
    )


__all__ = ["SecurityMiddleware", "enable_global_rate_limit"]
