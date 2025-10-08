"""Unit tests for middleware utilities."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request
from starlette.responses import Response

from src.services.fastapi.middleware import utils


@pytest.fixture(name="request_factory")
def _request_factory() -> Callable[..., Request]:
    """Return a factory to create Starlette requests with desired headers."""

    def _factory(**kwargs) -> Request:
        header_items = [
            (key.encode("latin-1"), value.encode("latin-1"))
            for key, value in kwargs.get("headers", {}).items()
        ]
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": header_items,
            "client": kwargs.get("client", ("127.0.0.1", 1234)),
        }
        return Request(scope)

    return _factory


def test_client_ip_without_proxy(request_factory) -> None:
    """Test client IP extraction without proxy."""

    request = request_factory()
    assert utils.client_ip(request) == "127.0.0.1"


def test_client_ip_trusted_proxy(request_factory) -> None:
    """Test client IP extraction with trusted proxy."""

    request = request_factory(
        headers={"x-forwarded-for": "203.0.113.10, 192.0.2.1"},
        client=("127.0.0.1", 4321),
    )
    assert utils.client_ip(request, trust_proxy=True) == "203.0.113.10"


def test_client_ip_without_headers_returns_unknown(request_factory) -> None:
    """Test client IP returns unknown without headers."""

    request = request_factory(client=None)
    assert utils.client_ip(request) == "unknown"


def test_safe_escape_handles_none() -> None:
    """Test safe_escape handles None input."""

    assert utils.safe_escape(None) is None


def test_safe_escape_escapes_html() -> None:
    """Test safe_escape escapes HTML."""

    assert utils.safe_escape("<script>") == "&lt;script&gt;"


def test_body_bytes_returns_body() -> None:
    """Test body_bytes returns response body."""

    response = Response(content=b"payload")
    assert utils.body_bytes(response) == b"payload"


def test_body_bytes_missing_attribute_returns_none() -> None:
    """Test body_bytes returns None for missing attribute."""

    response = SimpleNamespace()
    assert utils.body_bytes(cast(Response, response)) is None


def test_ensure_vary_accept_encoding_adds_header() -> None:
    """Test ensure_vary_accept_encoding adds header."""

    headers: dict[str, str] = {}
    utils.ensure_vary_accept_encoding(headers)
    assert headers["vary"] == "Accept-Encoding"


def test_ensure_vary_accept_encoding_appends() -> None:
    """Test ensure_vary_accept_encoding appends to existing."""

    headers = {"vary": "Cookie"}
    utils.ensure_vary_accept_encoding(headers)
    assert headers["vary"] == "Cookie, Accept-Encoding"
