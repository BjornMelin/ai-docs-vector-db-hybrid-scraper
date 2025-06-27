"""Logging configuration for service layer."""

import logging  # noqa: PLC0415
import sys
from typing import Any

from ..config import get_config


# Custom formatter for structured logging
class ServiceLayerFormatter(logging.Formatter):
    """Custom formatter with structured output for service layer."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with additional context."""
        # Add service context if available
        if hasattr(record, "service"):
            record.msg = f"[{record.service}] {record.msg}"

        # Add operation context if available
        if hasattr(record, "operation"):
            record.msg = f"{record.msg} (op: {record.operation})"

        return super().format(record)


def configure_logging(
    level: str | None = None,
    enable_color: bool = True,
    log_file: str | None = None,
) -> None:
    """Configure logging for the service layer.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_color: Enable colored output for console
        log_file: Optional log file path
    """
    config = get_config()

    if level is None:
        level = config.log_level.value

    if log_file is None and hasattr(config, "log_file"):
        log_file = config.log_file

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if enable_color:
        try:
            import colorlog  # noqa: PLC0415

            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        except ImportError:
            formatter = ServiceLayerFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
    else:
        formatter = ServiceLayerFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - "
            "[%(filename)s:%(lineno)d in %(funcName)s()]",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)

    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={level}, color={enable_color}, file={log_file}"
    )


class LogContext:
    """Context manager for adding context to log messages."""

    def __init__(self, **kwargs: Any):
        """Initialize with context attributes."""
        self.context = kwargs
        self.old_factory = None

    def __enter__(self):
        """Enter context and set log record factory."""
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args):
        """Exit context and restore factory."""
        logging.setLogRecordFactory(self.old_factory)


# Convenience function for adding service context
def with_service_context(service_name: str):
    """Add service context to logs.

    Args:
        service_name: Name of the service

    Returns:
        LogContext instance

    Example:
        with with_service_context("OpenAIProvider"):
            logger.info("Generating embeddings")
    """
    return LogContext(service=service_name)
