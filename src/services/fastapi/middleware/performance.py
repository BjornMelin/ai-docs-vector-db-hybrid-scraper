"""Performance middleware and Prometheus integration.

- Adds a lightweight `X-Response-Time` header (milliseconds).
- Exposes a helper to wire Prometheus metrics via the instrumentator.

References:
- prometheus-fastapi-instrumentator usage and /metrics exposure.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Add `X-Response-Time` header."""

    def __init__(self, app: Callable, *, precision_ms: int = 3) -> None:
        """Initialize the middleware.

        Args:
            app: ASGI app.
            precision_ms: Millisecond precision for the header value.
        """
        super().__init__(app)
        self._fmt = f"{{:.{max(0, min(6, precision_ms))}f}}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Measure response time and inject X-Response-Time header."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Response-Time"] = self._fmt.format(elapsed_ms)
        return response


def setup_prometheus(app, *, include_default: bool = True) -> Instrumentator:
    """Instrument app and expose `/metrics`.

    Args:
        app: FastAPI/Starlette app.
        include_default: Whether to include default HTTP metrics.

    Returns:
        The instrumentator for further customization.

    Example:
        instrumentator = setup_prometheus(app)
    """
    inst = Instrumentator()
    if include_default:
        inst.instrument(app)
    inst.expose(app)
    return inst


__all__ = ["PerformanceMiddleware", "setup_prometheus"]
