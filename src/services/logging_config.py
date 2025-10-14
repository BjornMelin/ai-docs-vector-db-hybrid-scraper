"""Logging configuration utilities for the AI Docs service layer.

This module provides a structured, context-aware logging configuration that is
safe for asynchronous services and CLI tooling. It exposes helpers for
configuring root logging handlers, binding contextual metadata via
:mod:`contextvars`, and redacting sensitive values before emission.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any


try:
    import colorlog  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency
    colorlog = None

from src.config.loader import Settings, get_settings


_DEFAULT_CONSOLE_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | service=%(service)s | "
    "operation=%(operation)s | %(message)s"
)
_DEFAULT_FILE_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s | "
    "%(filename)s:%(lineno)d %(funcName)s"
)

_EMPTY_CONTEXT: Mapping[str, Any] = MappingProxyType({})

_LOG_CONTEXT: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "ai_docs_log_context",
    default=_EMPTY_CONTEXT,
)

_ROOT_SENTINEL = "ai_docs_managed_handler"
_FACTORY_WRAPPED = contextvars.ContextVar("ai_docs_factory_wrapped", default=False)

_SECRET_PATTERNS = (
    re.compile(r'(?i)(api[_-]?key)[=:]\s*([^\s,;\'"]+)'),
    re.compile(r'(?i)(token)[=:]\s*([^\s,;\'"]+)'),
    re.compile(r'(?i)(password)[=:]\s*([^\s,;\'"]+)'),
    re.compile(r'(?i)(secret)[=:]\s*([^\s,;\'"]+)'),
    re.compile(r'(?i)(authorization)\s*[:=]\s*(?:bearer\s+)?([^\s,;\'"]+)'),
    re.compile(r'(?i)(auth)[=:]\s*([^\s,;\'"]+)'),
    re.compile(r'(?i)(bearer)\s+([^\s,;\'"]+)'),
)


class ContextFilter(logging.Filter):
    """Inject context variables into log records with safe defaults."""

    __slots__ = ("_default_operation", "_default_service")

    def __init__(self, service: str = "-", operation: str = "-") -> None:
        super().__init__(name="ai_docs_context_filter")
        self._default_service = service
        self._default_operation = operation

    def filter(self, record: logging.LogRecord) -> bool:
        context = dict(_LOG_CONTEXT.get())
        service = context.get("service", self._default_service)
        operation = context.get("operation", self._default_operation)

        record.service = service if service else "-"
        record.operation = operation if operation else "-"

        extras = {k: v for k, v in context.items() if k not in {"service", "operation"}}
        record.context = extras
        return True


class RedactionFilter(logging.Filter):
    """Mask sensitive values from formatted log messages before emission."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = message
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub(lambda match: f"{match.group(1)}=***", redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    """Emit structured JSON log lines with contextual metadata."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": getattr(record, "service", None),
            "operation": getattr(record, "operation", None),
        }

        context = getattr(record, "context", {})
        if context:
            payload.update(context)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(
            {key: value for key, value in payload.items() if value is not None}
        )


class LogContext:
    """Bind contextual metadata to log records within a managed scope."""

    def __init__(self, **context: Any) -> None:
        self._token: contextvars.Token[Mapping[str, Any]] | None = None
        self._context = context

    def __enter__(self) -> LogContext:
        self._token = bind_log_context(**self._context)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            reset_log_context(self._token)


def bind_log_context(**context: Any) -> contextvars.Token[Mapping[str, Any]]:
    """Bind contextual data for subsequent log records and return a token."""
    current = dict(_LOG_CONTEXT.get())
    current.update(context)
    return _LOG_CONTEXT.set(current)


def reset_log_context(token: contextvars.Token[Mapping[str, Any]]) -> None:
    """Restore the log context captured by :func:`bind_log_context`."""
    _LOG_CONTEXT.reset(token)


def clear_log_context() -> None:
    """Remove all contextual metadata from subsequent log records."""
    _LOG_CONTEXT.set(_EMPTY_CONTEXT)


def with_service_context(service_name: str) -> LogContext:
    """Return a context manager that sets the service attribute for logs."""
    return LogContext(service=service_name)


# pylint: disable=too-many-arguments,too-many-locals
def configure_logging(
    level: str | None = None,
    *,
    enable_color: bool | None = None,
    log_file: str | os.PathLike[str] | None = None,
    json_console: bool = False,
    force: bool = False,
    settings: Settings | None = None,
) -> None:
    """Configure application-wide logging.

    Args:
        level: Desired root log level. Defaults to the configured log level.
        enable_color: Whether to enable color output for the console handler. If
            ``None``, color is enabled only when :mod:`colorlog` is available and
            the stream is a TTY.
        log_file: Optional file path for structured log output.
        json_console: When ``True`` the console handler emits JSON instead of text.
        force: When ``True`` existing managed handlers are replaced.
    """
    config = settings or get_settings()

    resolved_level = (level or config.log_level.value).upper()
    level_value = getattr(logging, resolved_level, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level_value)

    _ensure_record_factory()

    managed_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, _ROOT_SENTINEL, False)
    ]
    has_managed_handler = bool(managed_handlers)

    if force and has_managed_handler:
        for handler in managed_handlers:
            root_logger.removeHandler(handler)
        has_managed_handler = False

    target_log_file = log_file or getattr(config, "log_file", None)

    if not has_managed_handler:
        console_handler = _build_console_handler(enable_color, json_console)
        _finalize_handler(console_handler)
        root_logger.addHandler(console_handler)

        if target_log_file:
            file_handler = _build_file_handler(target_log_file)
            _finalize_handler(file_handler)
            root_logger.addHandler(file_handler)
    else:
        for handler in managed_handlers:
            handler.setLevel(level_value)

    _tune_library_loggers(level_value)

    logging.getLogger(__name__).debug(
        "Logging configured",
        extra={
            "resolved_level": resolved_level,
            "json_console": json_console,
            "log_file": str(target_log_file or ""),
        },
    )


def _ensure_record_factory() -> None:
    """Ensure the log record factory is wrapped."""
    if _FACTORY_WRAPPED.get():
        return

    original_factory = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = original_factory(*args, **kwargs)
        context = _LOG_CONTEXT.get()
        for key, value in context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        if not hasattr(record, "context"):
            extras = {
                key: value
                for key, value in context.items()
                if key not in {"service", "operation"}
            }
            record.context = extras
        return record

    logging.setLogRecordFactory(factory)
    _FACTORY_WRAPPED.set(True)


def _build_console_handler(
    enable_color: bool | None, json_console: bool
) -> logging.Handler:
    """Build a console handler for logging."""
    stream = logging.StreamHandler(sys.stderr)
    formatter: logging.Formatter

    resolved_enable_color = enable_color
    if resolved_enable_color is None:
        resolved_enable_color = bool(
            colorlog and getattr(stream.stream, "isatty", lambda: False)()
        )

    if json_console:
        formatter = JsonFormatter()
    elif resolved_enable_color and colorlog is not None:
        formatter = colorlog.ColoredFormatter(  # type: ignore[no-untyped-call]
            "%(log_color)s" + _DEFAULT_CONSOLE_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        formatter = logging.Formatter(
            _DEFAULT_CONSOLE_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream.setFormatter(formatter)
    return stream


def _build_file_handler(path: str | os.PathLike[str]) -> logging.Handler:
    """Build a file handler for logging."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(file_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            _DEFAULT_FILE_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _finalize_handler(handler: logging.Handler) -> None:
    """Finalize a logging handler by adding filters and a sentinel."""
    setattr(handler, _ROOT_SENTINEL, True)
    handler.addFilter(ContextFilter())
    handler.addFilter(RedactionFilter())


def _tune_library_loggers(level_value: int) -> None:
    """Tune loggers for external libraries."""
    if level_value <= logging.DEBUG:
        return

    for name in ("httpx", "httpcore", "openai", "qdrant_client"):
        logging.getLogger(name).setLevel(logging.WARNING)
