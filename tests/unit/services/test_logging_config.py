"""Unit tests for the structured logging configuration module."""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest

from src.services import logging_config
from src.services.logging_config import (
    LogContext,
    bind_log_context,
    clear_log_context,
    configure_logging,
    with_service_context,
)


@pytest.fixture(autouse=True)
def isolate_logging_state():
    """Provide a clean logging environment for each test."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    original_factory = logging.getLogRecordFactory()

    root_logger.handlers.clear()
    logging_config._FACTORY_WRAPPED.set(False)
    logging_config._LOG_CONTEXT.set(logging_config._EMPTY_CONTEXT)
    logging.setLogRecordFactory(original_factory)

    clear_log_context()

    try:
        yield
    finally:
        for handler in list(root_logger.handlers):
            handler.close()
        root_logger.handlers.clear()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
        logging.setLogRecordFactory(original_factory)
        clear_log_context()
        logging_config._FACTORY_WRAPPED.set(False)
        logging_config._LOG_CONTEXT.set(logging_config._EMPTY_CONTEXT)


def _managed_handlers() -> list[logging.Handler]:
    """Return the list of managed handlers."""
    root = logging.getLogger()
    return [
        handler
        for handler in root.handlers
        if getattr(handler, logging_config._ROOT_SENTINEL, False)
    ]


def test_configure_logging_installs_managed_handlers(tmp_path: Path) -> None:
    """Test that configure_logging installs managed handlers."""
    log_path = tmp_path / "logs" / "service.log"

    configure_logging(level="DEBUG", log_file=log_path, force=True)

    handlers = _managed_handlers()
    assert len(handlers) == 2
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    assert any(isinstance(h, logging.FileHandler) for h in handlers)

    for handler in handlers:
        assert any(isinstance(f, logging_config.ContextFilter) for f in handler.filters)
        assert any(
            isinstance(f, logging_config.RedactionFilter) for f in handler.filters
        )

    file_handler = next(h for h in handlers if isinstance(h, logging.FileHandler))
    assert file_handler.baseFilename == str(log_path)
    assert file_handler.stream.encoding == "utf-8"


def test_configure_logging_is_idempotent() -> None:
    """Test that configure_logging is idempotent."""
    configure_logging(level="INFO", force=True)
    first_handlers = list(_managed_handlers())

    configure_logging(level="INFO")
    second_handlers = list(_managed_handlers())

    assert len(second_handlers) == len(first_handlers)
    assert second_handlers == first_handlers


def test_configure_logging_json_console() -> None:
    """Test that configure_logging sets JSON console formatter."""
    configure_logging(level="INFO", json_console=True, force=True)

    console_handler = next(
        h for h in _managed_handlers() if isinstance(h, logging.StreamHandler)
    )
    assert isinstance(console_handler.formatter, logging_config.JsonFormatter)


def test_log_context_binds_service_and_operation() -> None:
    """Test that LogContext binds service and operation."""
    configure_logging(level="INFO", force=True)
    record_factory = logging.getLogRecordFactory()

    with LogContext(service="Indexer", operation="sync"):
        record = record_factory("test", logging.INFO, __file__, 10, "hello", (), None)

    assert cast(Any, record).service == "Indexer"
    assert cast(Any, record).operation == "sync"


def test_with_service_context_resets_after_exit() -> None:
    """Test that with_service_context resets after exit."""
    configure_logging(level="INFO", force=True)
    record_factory = logging.getLogRecordFactory()

    with with_service_context("Embedder"):
        record_in_context = record_factory(
            "test", logging.INFO, __file__, 10, "msg", (), None
        )
    assert cast(Any, record_in_context).service == "Embedder"

    record_after = record_factory("test", logging.INFO, __file__, 11, "msg", (), None)
    assert getattr(record_after, "service", None) is None


def test_bind_and_clear_log_context() -> None:
    """Test binding and clearing log context."""
    configure_logging(level="INFO", force=True)
    token = bind_log_context(request_id="abc123")
    try:
        record = logging.getLogRecordFactory()(
            "test", logging.INFO, __file__, 12, "msg", (), None
        )
        assert cast(Any, record).context == {"request_id": "abc123"}
    finally:
        reset_token = token
        logging_config.reset_log_context(reset_token)

    clear_log_context()
    record_post_clear = logging.getLogRecordFactory()(
        "test", logging.INFO, __file__, 13, "msg", (), None
    )
    assert getattr(record_post_clear, "context", None) in (None, {})


def test_redaction_filter_masks_sensitive_values() -> None:
    """Test that RedactionFilter masks sensitive values."""
    configure_logging(level="INFO", force=True)
    handler = logging.StreamHandler(StringIO())
    handler.addFilter(logging_config.RedactionFilter())

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="api_key=super-secret token=other-secret",
        args=(),
        exc_info=None,
    )

    assert logging_config.RedactionFilter().filter(record) is True
    assert record.getMessage() == "api_key=*** token=***"


def test_json_formatter_includes_context_metadata() -> None:
    """Test that JsonFormatter includes context metadata."""
    formatter = logging_config.JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=50,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.service = "svc"
    record.operation = "op"
    record.context = {"request_id": "abc"}

    payload = json.loads(formatter.format(record))
    assert payload["service"] == "svc"
    assert payload["operation"] == "op"
    assert payload["request_id"] == "abc"
    assert payload["message"] == "hello"


def test_json_formatter_serializes_exceptions() -> None:
    """Test that JsonFormatter serializes exceptions."""

    def _raise_test_exception():
        raise ValueError("boom")

    formatter = logging_config.JsonFormatter()
    try:
        _raise_test_exception()
    except ValueError as exc:
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=75,
            msg="failure",
            args=(),
            exc_info=(exc.__class__, exc, exc.__traceback__),
        )

        serialized = json.loads(formatter.format(record))
        assert serialized["level"] == "ERROR"
        assert "ValueError" in serialized["exception"]


def test_configure_logging_redaction_applies_to_runtime_logs() -> None:
    """Test that configure_logging applies redaction to runtime logs."""
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.addFilter(logging_config.RedactionFilter())

    logger = logging.getLogger("redaction-test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("password=hunter2")
    assert "password=***" in buffer.getvalue()
