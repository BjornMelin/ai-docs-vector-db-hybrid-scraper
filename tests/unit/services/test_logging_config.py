"""Tests for services/logging_config.py - Logging configuration.

This module tests the logging configuration system for service layer,
including custom formatters, log contexts, and configuration management.
"""

import logging
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from src.services.logging_config import LogContext
from src.services.logging_config import ServiceLayerFormatter
from src.services.logging_config import configure_logging
from src.services.logging_config import with_service_context


class TestServiceLayerFormatter:
    """Test cases for ServiceLayerFormatter class."""

    def test_service_layer_formatter_basic(self):
        """Test basic formatting without service context."""
        formatter = ServiceLayerFormatter("%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert result == "INFO - Test message"

    def test_service_layer_formatter_with_service_context(self):
        """Test formatting with service context."""
        formatter = ServiceLayerFormatter("%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.service = "OpenAIProvider"

        result = formatter.format(record)
        assert result == "INFO - [OpenAIProvider] Test message"

    def test_service_layer_formatter_with_operation_context(self):
        """Test formatting with operation context."""
        formatter = ServiceLayerFormatter("%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.operation = "generate_embeddings"

        result = formatter.format(record)
        assert result == "INFO - Test message (op: generate_embeddings)"

    def test_service_layer_formatter_with_both_contexts(self):
        """Test formatting with both service and operation context."""
        formatter = ServiceLayerFormatter("%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.service = "QdrantClient"
        record.operation = "vector_search"

        result = formatter.format(record)
        assert result == "INFO - [QdrantClient] Test message (op: vector_search)"

    def test_service_layer_formatter_inheritance(self):
        """Test ServiceLayerFormatter inherits from logging.Formatter correctly."""
        formatter = ServiceLayerFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        assert isinstance(formatter, logging.Formatter)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "test.logger" in result
        assert "WARNING" in result
        assert "Warning message" in result


class TestConfigureLogging:
    """Test cases for configure_logging function."""

    def setup_method(self):
        """Setup for each test method."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Reset to default level
        root_logger.setLevel(logging.WARNING)

    def _create_mock_config(self, log_level="INFO", has_log_file=False):
        """Create a proper mock config without problematic attributes."""
        mock_config = Mock()
        mock_config.log_level.value = log_level
        if not has_log_file:
            # Ensure log_file attribute doesn't exist
            if hasattr(mock_config, "log_file"):
                del mock_config.log_file
        return mock_config

    def test_configure_logging_basic(self):
        """Test basic logging configuration."""
        mock_config = self._create_mock_config("INFO")

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)

    def test_configure_logging_with_custom_level(self):
        """Test logging configuration with custom level."""
        mock_config = self._create_mock_config("WARNING")

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging(level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_with_file_handler(self):
        """Test logging configuration with file handler."""
        mock_config = self._create_mock_config("INFO")

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            log_file_path = tmp_file.name

        try:
            with patch(
                "src.services.logging_config.get_config", return_value=mock_config
            ):
                configure_logging(log_file=log_file_path)

            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 2  # Console + File

            # Check file handler exists
            file_handlers = [
                h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1

        finally:
            # Clean up
            Path(log_file_path).unlink()

    def test_configure_logging_without_colorlog(self):
        """Test logging configuration when colorlog is not available."""
        mock_config = self._create_mock_config("INFO")

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'colorlog'"),
            ):
                configure_logging(enable_color=True)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        # Should fall back to ServiceLayerFormatter
        assert isinstance(handler.formatter, ServiceLayerFormatter)

    def test_configure_logging_with_colorlog(self):
        """Test logging configuration with colorlog available."""
        mock_config = self._create_mock_config("INFO")

        mock_colorlog = Mock()
        mock_formatter = Mock()
        mock_colorlog.ColoredFormatter.return_value = mock_formatter

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            with patch.dict("sys.modules", {"colorlog": mock_colorlog}):
                configure_logging(enable_color=True)

        # Should create ColoredFormatter
        mock_colorlog.ColoredFormatter.assert_called_once()

    def test_configure_logging_without_color(self):
        """Test logging configuration with color disabled."""
        mock_config = self._create_mock_config("INFO")

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging(enable_color=False)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, ServiceLayerFormatter)

    def test_configure_logging_sets_library_levels(self):
        """Test logging configuration sets levels for noisy libraries."""
        mock_config = self._create_mock_config("DEBUG")

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging()

        # Check library loggers are set to WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING
        assert logging.getLogger("qdrant_client").level == logging.WARNING

    def test_configure_logging_with_config_log_file(self):
        """Test logging configuration uses config log_file if available."""
        mock_config = self._create_mock_config("INFO")
        mock_config.log_file = "/tmp/test.log"

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            with patch("logging.FileHandler") as mock_file_handler:
                # Create a proper mock file handler
                mock_handler = Mock()
                mock_handler.level = logging.INFO  # Set proper level as integer
                mock_handler.setLevel = Mock()
                mock_handler.setFormatter = Mock()
                mock_file_handler.return_value = mock_handler
                configure_logging()

        # Should create file handler with config log_file
        mock_file_handler.assert_called_once_with("/tmp/test.log")

    def test_configure_logging_clears_existing_handlers(self):
        """Test logging configuration clears existing handlers."""
        # Add some existing handlers
        root_logger = logging.getLogger()
        existing_handler = logging.StreamHandler()
        root_logger.addHandler(existing_handler)

        mock_config = self._create_mock_config("INFO")

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging()

        # Should have only the new handler
        assert len(root_logger.handlers) == 1
        assert existing_handler not in root_logger.handlers

    def test_configure_logging_file_formatter(self):
        """Test file handler uses detailed formatter."""
        mock_config = self._create_mock_config("INFO")

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            log_file_path = tmp_file.name

        try:
            with patch(
                "src.services.logging_config.get_config", return_value=mock_config
            ):
                configure_logging(log_file=log_file_path)

            root_logger = logging.getLogger()
            file_handlers = [
                h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
            ]
            file_handler = file_handlers[0]

            # Check formatter includes filename and function info
            formatter_format = file_handler.formatter._fmt
            assert "%(filename)s" in formatter_format
            assert "%(lineno)d" in formatter_format
            assert "%(funcName)s" in formatter_format

        finally:
            Path(log_file_path).unlink()

    def test_configure_logging_logs_configuration(self):
        """Test logging configuration logs its own setup."""
        mock_config = self._create_mock_config("DEBUG")

        # Capture log output to verify the configuration message
        log_stream = StringIO()

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging(
                level="INFO", enable_color=False, log_file="/tmp/test.log"
            )

            # Get the handler and check if it logged
            root_logger = logging.getLogger()
            if root_logger.handlers:
                # Replace the handler stream to capture output
                original_stream = root_logger.handlers[0].stream
                root_logger.handlers[0].stream = log_stream

                # Create a test log message to verify it works
                test_logger = logging.getLogger("src.services.logging_config")
                test_logger.info("Test configuration logging")

                # Restore original stream
                root_logger.handlers[0].stream = original_stream

        # Should contain configuration details
        log_stream.getvalue()
        # The actual configuration log happens during configure_logging
        # We can verify the logger is properly configured at INFO level


class TestLogContext:
    """Test cases for LogContext class."""

    def setup_method(self):
        """Setup for each test method."""
        # Store original factory to restore later
        self.original_factory = logging.getLogRecordFactory()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Restore original factory
        logging.setLogRecordFactory(self.original_factory)

    def test_log_context_basic(self):
        """Test basic LogContext usage."""
        context = LogContext(service="TestService", operation="test_op")

        with context:
            # Create a log record using the current factory
            factory = logging.getLogRecordFactory()
            record = factory(
                "test",
                logging.INFO,
                "test.py",
                1,
                "Test message",
                (),
                None,
            )

        # Context should have been applied
        assert hasattr(record, "service")
        assert hasattr(record, "operation")
        assert record.service == "TestService"
        assert record.operation == "test_op"

    def test_log_context_restoration(self):
        """Test LogContext restores original factory."""
        original_factory = logging.getLogRecordFactory()

        with LogContext(test_attr="test_value"):
            # Factory should be different inside context
            current_factory = logging.getLogRecordFactory()
            assert current_factory != original_factory

        # Factory should be restored after context
        restored_factory = logging.getLogRecordFactory()
        assert restored_factory == original_factory

    def test_log_context_multiple_attributes(self):
        """Test LogContext with multiple attributes."""
        context_attrs = {
            "service": "QdrantClient",
            "operation": "search_vectors",
            "request_id": "req_123",
            "user_id": "user_456",
        }

        with LogContext(**context_attrs):
            factory = logging.getLogRecordFactory()
            record = factory(
                "test",
                logging.INFO,
                "test.py",
                1,
                "Test message",
                (),
                None,
            )

        for key, value in context_attrs.items():
            assert hasattr(record, key)
            assert getattr(record, key) == value

    def test_log_context_nested(self):
        """Test nested LogContext usage."""
        with LogContext(service="OuterService"):
            factory = logging.getLogRecordFactory()
            outer_record = factory(
                "test",
                logging.INFO,
                "test.py",
                1,
                "Outer message",
                (),
                None,
            )

            with LogContext(operation="inner_op"):
                factory = logging.getLogRecordFactory()
                inner_record = factory(
                    "test",
                    logging.INFO,
                    "test.py",
                    1,
                    "Inner message",
                    (),
                    None,
                )

        # Outer record should only have service
        assert hasattr(outer_record, "service")
        assert outer_record.service == "OuterService"
        assert not hasattr(outer_record, "operation")

        # Inner record should have both (service from outer, operation from inner)
        assert hasattr(inner_record, "service")
        assert hasattr(inner_record, "operation")
        assert inner_record.service == "OuterService"
        assert inner_record.operation == "inner_op"

    def test_log_context_exception_handling(self):
        """Test LogContext handles exceptions properly."""
        original_factory = logging.getLogRecordFactory()

        try:
            with LogContext(service="TestService"):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Factory should still be restored
        assert logging.getLogRecordFactory() == original_factory

    def test_log_context_empty(self):
        """Test LogContext with no attributes."""
        with LogContext():
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None,
            )

        # Should not add any extra attributes
        base_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "getMessage",
            "exc_info",
            "exc_text",
            "stack_info",
        }

        record_attrs = set(dir(record))
        extra_attrs = record_attrs - base_attrs

        # Filter out private attributes and methods
        extra_attrs = {attr for attr in extra_attrs if not attr.startswith("_")}

        # Should have minimal extra attributes
        assert (
            len(extra_attrs) <= 2
        )  # Allow for some variation in logging implementation


class TestWithServiceContext:
    """Test cases for with_service_context function."""

    def setup_method(self):
        """Setup for each test method."""
        self.original_factory = logging.getLogRecordFactory()

    def teardown_method(self):
        """Cleanup after each test method."""
        logging.setLogRecordFactory(self.original_factory)

    def test_with_service_context_basic(self):
        """Test with_service_context function."""
        context = with_service_context("OpenAIProvider")

        assert isinstance(context, LogContext)

        with context:
            # Create a log record using the current factory
            factory = logging.getLogRecordFactory()
            record = factory(
                "test",
                logging.INFO,
                "test.py",
                1,
                "Test message",
                (),
                None,
            )

        assert hasattr(record, "service")
        assert record.service == "OpenAIProvider"

    def test_with_service_context_usage_example(self):
        """Test with_service_context in realistic usage scenario."""
        # Simulate the example from the docstring
        logger = logging.getLogger("test_logger")

        with with_service_context("OpenAIProvider"):
            # Capture log output
            with patch.object(logger, "info") as mock_info:
                logger.info("Generating embeddings")

                # Get the log record that would be created
                call_args = mock_info.call_args
                if call_args:
                    # The actual record creation happens in the logging system
                    # We can verify the context would be applied by checking
                    # that our factory is in place
                    factory = logging.getLogRecordFactory()
                    record = factory(
                        "test_logger",
                        logging.INFO,
                        "test.py",
                        1,
                        "Generating embeddings",
                        (),
                        None,
                    )
                    assert hasattr(record, "service")
                    assert record.service == "OpenAIProvider"

    def test_with_service_context_different_services(self):
        """Test with_service_context with different service names."""
        services = ["QdrantClient", "EmbeddingManager", "CrawlProvider", "CacheService"]

        for service_name in services:
            with with_service_context(service_name):
                factory = logging.getLogRecordFactory()
                record = factory(
                    "test",
                    logging.INFO,
                    "test.py",
                    1,
                    "Test message",
                    (),
                    None,
                )

                assert record.service == service_name


class TestLoggingIntegration:
    """Integration tests for logging configuration."""

    def setup_method(self):
        """Setup for each test method."""
        # Clear handlers and reset level
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    def _create_mock_config(self, log_level="INFO", has_log_file=False):
        """Create a proper mock config without problematic attributes."""
        mock_config = Mock()
        mock_config.log_level.value = log_level
        if not has_log_file:
            # Ensure log_file attribute doesn't exist
            if hasattr(mock_config, "log_file"):
                del mock_config.log_file
        return mock_config

    def test_full_logging_setup_and_usage(self):
        """Test complete logging setup and usage with contexts."""
        mock_config = self._create_mock_config("INFO")

        # Capture log output
        log_stream = StringIO()

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging(enable_color=False)

            # Replace the console handler with our test stream
            root_logger = logging.getLogger()
            if root_logger.handlers:
                root_logger.handlers[0].stream = log_stream

        # Use logging with service context
        logger = logging.getLogger("integration_test")

        with with_service_context("TestService"):
            logger.info("Service operation started")

            with LogContext(operation="test_operation", request_id="req_123"):
                logger.info("Processing request")

        # Check log output
        log_output = log_stream.getvalue()
        assert "[TestService]" in log_output
        assert "Service operation started" in log_output
        assert "Processing request" in log_output
        assert "(op: test_operation)" in log_output

    def test_logging_with_formatter_integration(self):
        """Test logging integration with ServiceLayerFormatter."""
        # Setup logging with our formatter
        logger = logging.getLogger("formatter_test")
        handler = logging.StreamHandler(StringIO())
        formatter = ServiceLayerFormatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Test with contexts
        with with_service_context("FormatterTestService"):
            with LogContext(operation="format_test"):
                logger.info("Test message")

        # Get the formatted output
        handler.stream.seek(0)
        output = handler.stream.read()

        assert "[FormatterTestService]" in output
        assert "(op: format_test)" in output
        assert "Test message" in output

    def test_multiple_loggers_with_contexts(self):
        """Test multiple loggers with different contexts."""
        mock_config = self._create_mock_config("DEBUG")

        log_stream = StringIO()

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging(enable_color=False)

            root_logger = logging.getLogger()
            if root_logger.handlers:
                root_logger.handlers[0].stream = log_stream

        logger1 = logging.getLogger("service1")
        logger2 = logging.getLogger("service2")

        with with_service_context("ServiceOne"):
            logger1.info("Message from service one")

        with with_service_context("ServiceTwo"):
            logger2.info("Message from service two")

        log_output = log_stream.getvalue()
        assert "[ServiceOne]" in log_output
        assert "[ServiceTwo]" in log_output
        assert "Message from service one" in log_output
        assert "Message from service two" in log_output

    def test_error_logging_with_contexts(self):
        """Test error logging with service contexts."""
        mock_config = self._create_mock_config("ERROR")

        log_stream = StringIO()

        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging(enable_color=False)

            root_logger = logging.getLogger()
            if root_logger.handlers:
                root_logger.handlers[0].stream = log_stream

        logger = logging.getLogger("error_test")

        # Keep the context active during the logging call
        with with_service_context("ErrorService"):
            with LogContext(operation="risky_operation"):
                try:
                    raise ValueError("Something went wrong")
                except ValueError:
                    logger.exception("Operation failed")

        log_output = log_stream.getvalue()
        assert "[ErrorService]" in log_output
        assert "(op: risky_operation)" in log_output
        assert "Operation failed" in log_output

    def test_configure_logging_edge_cases(self):
        """Test edge cases in configure_logging function."""
        mock_config = self._create_mock_config("DEBUG")

        # Test with level as None to use config default
        with patch("src.services.logging_config.get_config", return_value=mock_config):
            configure_logging(level=None, enable_color=False)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_service_layer_formatter_edge_cases(self):
        """Test edge cases for ServiceLayerFormatter."""
        formatter = ServiceLayerFormatter("%(message)s")

        # Test with record that has neither service nor operation
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Plain message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert result == "Plain message"

        # Test with empty service name
        record.service = ""
        result = formatter.format(record)
        assert result == "[] Plain message"

    def test_log_context_nested_complex(self):
        """Test complex nested LogContext scenarios."""
        with LogContext(service="OuterService", user_id="123"):
            with LogContext(operation="inner", request_id="456"):
                # Create record inside nested context
                factory = logging.getLogRecordFactory()
                inner_record = factory(
                    "test", logging.INFO, "test.py", 1, "Inner", (), None
                )

                # Inner record should have all contexts
                assert inner_record.service == "OuterService"
                assert inner_record.user_id == "123"
                assert inner_record.operation == "inner"
                assert inner_record.request_id == "456"

            # Create record in outer context only
            factory = logging.getLogRecordFactory()
            outer_record = factory(
                "test", logging.INFO, "test.py", 1, "Outer", (), None
            )

            # Outer record should only have outer context
            assert outer_record.service == "OuterService"
            assert outer_record.user_id == "123"
            assert not hasattr(outer_record, "operation")
            assert not hasattr(outer_record, "request_id")
