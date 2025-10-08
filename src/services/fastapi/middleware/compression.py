"""
Compression middleware thin wrappers.

Prefer battle-tested implementations:
- Starlette's GZipMiddleware for gzip.
- brotli-asgi's BrotliMiddleware for Brotli (optional).

References:
- Starlette GZip docs (params & behavior).
- brotli-asgi drop-in usage.
"""

from __future__ import annotations

from starlette.middleware.gzip import (
    GZipMiddleware as CompressionMiddleware,  # type: ignore
)


try:
    from brotli_asgi import (  # type: ignore
        BrotliMiddleware as BrotliCompressionMiddleware,
    )
except Exception:  # pragma: no cover - optional dep
    BrotliCompressionMiddleware = CompressionMiddleware  # type: ignore[misc]

__all__ = ["CompressionMiddleware", "BrotliCompressionMiddleware"]
