"""Tests for tracing middleware logging."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.fastapi.middleware.tracing import TracingMiddleware


def test_tracing_middleware_sets_response_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that TracingMiddleware sets response headers."""

    app = FastAPI()
    app.add_middleware(TracingMiddleware)

    @app.get("/ping")
    async def _ping() -> dict[str, str]:
        return {"ok": "true"}

    with caplog.at_level(logging.INFO):
        response = TestClient(app).get("/ping")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert "X-Correlation-ID" in response.headers
    assert "X-Request-Duration" in response.headers

    request_records = [record for record in caplog.records if record.msg == "request"]
    assert request_records, "expected request log entry"
    record = request_records[0]
    assert hasattr(record, "correlation_id")
    path = getattr(record, "path", None)  # noqa: B009
    if path is not None:
        assert path == "/ping"


def test_tracing_middleware_optionally_logs_bodies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that TracingMiddleware optionally logs bodies."""

    app = FastAPI()
    app.add_middleware(
        TracingMiddleware,
        log_request_body=True,
        log_response_body=True,
        max_body_bytes=32,
    )

    @app.post("/echo")
    async def _echo(payload: dict[str, str]) -> dict[str, str]:
        return payload

    with caplog.at_level(logging.INFO):
        response = TestClient(app).post("/echo", json={"hello": "world"})

    assert response.status_code == 200
    request_records = [record for record in caplog.records if record.msg == "request"]
    assert request_records
    body = getattr(request_records[0], "body", None)  # noqa: B009
    assert body is not None
    assert "hello" in body

    response_records = [record for record in caplog.records if record.msg == "response"]
    assert response_records
    status = getattr(response_records[0], "status_code", None)  # noqa: B009
    assert status == 200
