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
    from brotli_asgi import (  # pylint: disable=import-error  # pyright: ignore[reportMissingImports]
        BrotliMiddleware as _BrotliMiddleware,
    )
except ImportError:  # pragma: no cover - optional dependency fallback
    import logging

    _logger = logging.getLogger(__name__)

    # Brotli-specific kwargs that GZipMiddleware doesn't support
    _BROTLI_ONLY_KWARGS = frozenset({"quality", "mode", "lgwin", "lgblock"})

    class _BrotliMiddleware(GZipMiddleware):
        """Fallback Brotli middleware using gzip when brotli is unavailable."""

        def __init__(self, app: ASGIApp, **kwargs: Any) -> None:
            """Initialize the Brotli middleware.

            Strips Brotli-specific kwargs and warns if they were provided.
            """
            stripped_kwargs = {
                k: v for k, v in kwargs.items() if k not in _BROTLI_ONLY_KWARGS
            }
            ignored_kwargs = set(kwargs.keys()) & _BROTLI_ONLY_KWARGS
            if ignored_kwargs:
                _logger.warning(
                    "Brotli-specific options %s ignored (brotli-asgi not installed)",
                    sorted(ignored_kwargs),
                )
            super().__init__(app, **stripped_kwargs)


BrotliCompressionMiddleware = cast(type[GZipMiddleware], _BrotliMiddleware)


__all__ = ["BrotliCompressionMiddleware", "CompressionMiddleware"]
