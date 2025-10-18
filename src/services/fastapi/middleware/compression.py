"""Compression middleware thin wrappers.

Prefer battle-tested implementations:
- Starlette's ``GZipMiddleware`` for gzip.
- ``brotli-asgi``'s ``BrotliMiddleware`` for Brotli (optional).

References:
- Starlette GZip docs (params & behavior).
- brotli-asgi drop-in usage.
"""

from __future__ import annotations

from typing import Any, cast

from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp


CompressionMiddleware = GZipMiddleware


try:
    from brotli_asgi import (
        BrotliMiddleware as _BrotliMiddleware,  # pylint: disable=import-error  # pyright: ignore[reportAssignmentType]
    )
except ImportError:  # pragma: no cover - optional dependency fallback

    class _BrotliMiddleware(GZipMiddleware):
        """Fallback Brotli middleware using gzip when brotli is unavailable."""

        def __init__(self, app: ASGIApp, **kwargs: Any) -> None:
            """Initialize the Brotli middleware."""
            super().__init__(app, **kwargs)


BrotliCompressionMiddleware = cast(type[GZipMiddleware], _BrotliMiddleware)


__all__ = ["BrotliCompressionMiddleware", "CompressionMiddleware"]
