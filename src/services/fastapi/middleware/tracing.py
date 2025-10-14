"""Request/response logging with correlation IDs.

- Uses asgi-correlation-id context to tag logs and responses.
- Keep name `TracingMiddleware` for compatibility; OTel should be enabled
  via `opentelemetry-instrumentation-fastapi` separately.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .correlation import get_correlation_id
from .utils import client_ip, safe_escape


logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """Lightweight structured logging around requests."""

    def __init__(
        self,
        app: Callable,
        *,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_bytes: int = 1024,
        trust_proxy: bool = False,
    ) -> None:
        """Initialize tracing middleware with structured logging configuration."""
        super().__init__(app)
        self._log_req = log_request_body
        self._log_res = log_response_body
        self._max = max_body_bytes
        self._trust_proxy = trust_proxy

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request/response with correlation ID and inject timing headers."""
        cid = get_correlation_id(request)
        start = time.perf_counter()

        if self._log_req:
            try:
                body = await request.body()
                body = body[: self._max]
                body_str = body.decode("utf-8", errors="replace")
            except Exception:
                body_str = "<unavailable>"
        else:
            body_str = None  # type: ignore[assignment]

        logger.info(
            "request",
            extra={
                "correlation_id": cid,
                "method": request.method,
                "path": request.url.path,
                "query": safe_escape(str(request.query_params))
                if request.query_params
                else None,
                "client_ip": client_ip(request, trust_proxy=self._trust_proxy),
                "content_type": request.headers.get("content-type"),
                "content_length": request.headers.get("content-length"),
                "body": safe_escape(body_str) if body_str else None,
            },
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.exception(
                "request_error",
                extra={
                    "correlation_id": cid,
                    "method": request.method,
                    "path": request.url.path,
                    "duration": elapsed,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise

        # Set response headers
        response.headers.setdefault("X-Request-ID", cid)
        response.headers.setdefault("X-Correlation-ID", cid)
        response.headers.setdefault(
            "X-Request-Duration", f"{(time.perf_counter() - start):.4f}"
        )

        if self._log_res:
            size = None
            try:
                size = len(response.body)  # type: ignore[attr-defined]
            except Exception:
                size = None
            logger.info(
                "response",
                extra={
                    "correlation_id": cid,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "response_size": size,
                    "content_type": response.headers.get("content-type"),
                },
            )
        return response


__all__ = ["TracingMiddleware", "get_correlation_id"]
