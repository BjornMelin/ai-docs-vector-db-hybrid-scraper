"""Security headers and rate-limiting glue.

- Injects conservative security headers.
- Provides a function to enable SlowAPI global rate limits with Redis.

References:
- SlowAPI docs (Limiter, SlowAPIMiddleware).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from urllib.parse import urlparse, urlunparse

from slowapi import Limiter  # type: ignore
from slowapi.errors import RateLimitExceeded  # type: ignore
from slowapi.middleware import SlowAPIMiddleware  # type: ignore
from slowapi.util import get_remote_address  # type: ignore
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config.loader import get_settings
from src.config.security import SecurityConfig


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
    app: Starlette,
    *,
    config: SecurityConfig | None = None,
    key_func: Callable[[Request], str] = get_remote_address,
    storage_uri: str | None = None,
) -> Limiter | None:
    """Enable SlowAPI global rate limit and attach handlers.

    Args:
        app: Target FastAPI or Starlette application.
        config: Optional :class:`SecurityConfig` override. When ``None`` the
            configuration is loaded from the global settings cache.
        key_func: Key extraction function (client identifier).
        storage_uri: Optional Redis URI override for distributed limits.

    Returns:
        Initialized :class:`Limiter` instance when rate limiting is enabled.
        ``None`` when rate limiting is disabled in the configuration.
    """
    security_config = config or get_settings().security
    if not security_config.enable_rate_limiting:
        app.state.limiter = None
        return None

    limiter = Limiter(
        key_func=key_func,
        default_limits=[_format_default_limit(security_config)],
        storage_uri=storage_uri or _resolve_storage_uri(security_config),
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limited)
    app.add_middleware(SlowAPIMiddleware)
    return limiter


def _format_default_limit(config: SecurityConfig) -> str:
    """Convert :class:`SecurityConfig` limits to SlowAPI notation."""
    window = config.rate_limit_window
    granularity_map: dict[int, str] = {
        1: "second",
        60: "minute",
        3600: "hour",
        86400: "day",
    }
    granularity = granularity_map.get(window)
    if granularity:
        return f"{config.default_rate_limit}/{granularity}"
    return f"{config.default_rate_limit}/{window} second"


def _resolve_storage_uri(config: SecurityConfig) -> str | None:
    """Build the SlowAPI storage URI from security configuration."""
    if not config.redis_url:
        return None

    if not config.redis_password:
        return config.redis_url

    parsed = urlparse(config.redis_url)
    if parsed.password:
        return config.redis_url

    username = parsed.username or ""
    credentials = (
        f"{username}:{config.redis_password}"
        if username
        else f":{config.redis_password}"
    )
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{credentials}@{host}{port}"
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


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
