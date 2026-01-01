"""Tests for the FastAPI compression middleware wrappers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from src.services.fastapi.middleware.compression import (
    BrotliCompressionMiddleware,
    CompressionMiddleware,
)


def test_compression_middleware_aliases_starlette_gzip() -> None:
    """CompressionMiddleware is the Starlette GZip middleware."""
    from starlette.middleware.gzip import GZipMiddleware

    assert CompressionMiddleware is GZipMiddleware


def test_brotli_fallback_is_gzip_subclass() -> None:
    """Brotli middleware falls back to gzip-compatible behaviour."""
    from starlette.middleware.gzip import GZipMiddleware

    try:
        from brotli_asgi import (  # pyright: ignore[reportMissingImports]
            BrotliMiddleware as ActualBrotli,
        )
    except ImportError:  # pragma: no cover - optional dependency unavailable
        assert issubclass(BrotliCompressionMiddleware, GZipMiddleware)
    else:
        assert BrotliCompressionMiddleware is ActualBrotli


def test_compression_applies_gzip_encoding() -> None:
    """Compression middleware responds with gzip when accepted by the client."""
    app = FastAPI()
    app.add_middleware(CompressionMiddleware, minimum_size=1)

    @app.get("/data")
    def data() -> PlainTextResponse:
        return PlainTextResponse("payload" * 50)

    with TestClient(app) as client:
        response = client.get("/data", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
