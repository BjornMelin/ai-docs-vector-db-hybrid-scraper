"""Tests for service layer logging configuration."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from src.services.logging_config import ServiceLayerFormatter
from src.services.logging_config import configure_logging


class TestServiceLayerFormatter:
    """Test ServiceLayerFormatter functionality."""

    def test_format_basic_message(self):
        """Test formatting basic log message."""
        formatter = ServiceLayerFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "Test message" in result

    def test_format_with_service_context(self):
        """Test formatting with service context."""
        formatter = ServiceLayerFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.service = "qdrant_service"

        result = formatter.format(record)
        assert "[qdrant_service] Test message" in result

    def test_format_with_operation_context(self):
        """Test formatting with operation context."""
        formatter = ServiceLayerFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.operation = "search_vectors"

        result = formatter.format(record)
        assert "Test message (op: search_vectors)" in result

    def test_format_with_both_contexts(self):
        """Test formatting with both service and operation context."""
        formatter = ServiceLayerFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.service = "qdrant_service"
        record.operation = "search_vectors"

        result = formatter.format(record)
        assert "[qdrant_service] Test message (op: search_vectors)" in result


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configure_logging_default_config(self):
        """Test configure_logging with default configuration."""
        with patch("src.services.logging_config.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.log_level.value = "INFO"
            # Use spec to avoid creating unexpected attributes
            mock_config.configure_mock(**{"log_file": None})
            mock_get_config.return_value = mock_config

            # Clear existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            configure_logging()

            # Check that logger level was set
            assert root_logger.level == logging.INFO
            # Check that handlers were added
            assert len(root_logger.handlers) > 0

    def test_configure_logging_custom_level(self):
        """Test configure_logging with custom level."""
        with patch("src.services.logging_config.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.log_level.value = "INFO"  # Default in config
            mock_config.configure_mock(**{"log_file": None})
            mock_get_config.return_value = mock_config

            # Clear existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            configure_logging(level="DEBUG")

            # Should use passed level, not config level
            assert root_logger.level == logging.DEBUG

    def test_configure_logging_with_file(self):
        """Test configure_logging with log file."""
        with patch("src.services.logging_config.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.log_level.value = "INFO"
            mock_get_config.return_value = mock_config

            # Clear existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                log_file_path = temp_file.name

            try:
                configure_logging(log_file=log_file_path)

                # Check that file handler was added
                file_handlers = [
                    h
                    for h in root_logger.handlers
                    if isinstance(h, logging.FileHandler)
                ]
                assert len(file_handlers) > 0

                # Check that log file exists
                assert Path(log_file_path).exists()

            finally:
                # Cleanup
                Path(log_file_path).unlink(missing_ok=True)

    def test_configure_logging_disable_color(self):
        """Test configure_logging with color disabled."""
        with patch("src.services.logging_config.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.log_level.value = "INFO"
            mock_config.configure_mock(**{"log_file": None})
            mock_get_config.return_value = mock_config

            # Clear existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            configure_logging(enable_color=False)

            # Should still configure logging successfully
            assert root_logger.level == logging.INFO
            assert len(root_logger.handlers) > 0

    def test_configure_logging_config_with_log_file(self):
        """Test configure_logging using log file from config."""
        with patch("src.services.logging_config.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.log_level.value = "WARNING"
            mock_config.log_file = "/tmp/test.log"
            mock_get_config.return_value = mock_config

            # Clear existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            with (
                patch("builtins.open"),
                patch("pathlib.Path.exists", return_value=True),
            ):
                configure_logging()

            # Should use config values
            assert root_logger.level == logging.WARNING

    def test_configure_logging_invalid_level(self):
        """Test configure_logging with invalid level."""
        with patch("src.services.logging_config.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.log_level.value = "INFO"
            mock_get_config.return_value = mock_config

            # Clear existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            # Should handle invalid level gracefully
            try:
                configure_logging(level="INVALID")
                # If it doesn't raise, check default behavior
                assert root_logger.level >= logging.INFO
            except (ValueError, AttributeError):
                # It's acceptable to raise an error for invalid level
                pass

    def test_configure_logging_multiple_calls(self):
        """Test that multiple configure_logging calls work correctly."""
        with patch("src.services.logging_config.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.log_level.value = "INFO"
            mock_config.configure_mock(**{"log_file": None})
            mock_get_config.return_value = mock_config

            # Clear existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            configure_logging(level="INFO")
            initial_handler_count = len(root_logger.handlers)

            configure_logging(level="DEBUG")

            # Should handle multiple calls gracefully
            # Either same number of handlers or properly managed
            assert len(root_logger.handlers) >= initial_handler_count
            assert root_logger.level == logging.DEBUG

    def test_service_logger_with_custom_formatter(self):
        """Test ServiceLayerFormatter directly."""
        formatter = ServiceLayerFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create log record with service context
        record = logging.LogRecord(
            name="test.service",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test service message",
            args=(),
            exc_info=None,
        )
        record.service = "test_service"

        formatted = formatter.format(record)
        # Should contain service context
        assert "[test_service]" in formatted
        assert "Test service message" in formatted
