"""Shared fixtures for CLI testing.

This module provides comprehensive fixtures for testing CLI components including
mocked dependencies, Rich console capturing, and async testing support.
"""

import asyncio
from io import StringIO
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from rich.console import Console


@pytest.fixture
def cli_runner():
    """Provide a Click CliRunner for testing CLI commands."""
    return CliRunner()


@pytest.fixture
def rich_console():
    """Provide a Rich Console with StringIO for output capture."""
    string_io = StringIO()
    return Console(file=string_io, width=80, force_terminal=True)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    config = MagicMock()
    config.qdrant = MagicMock()
    config.qdrant.host = "localhost"
    config.qdrant.port = 6333
    config.qdrant.api_key = None
    config.qdrant.timeout = 30
    config.qdrant.prefer_grpc = True

    config.openai = MagicMock()
    config.openai.api_key = "test-api-key"
    config.openai.model = "text-embedding-ada-002"

    config.cache = MagicMock()
    config.cache.enabled = True
    config.cache.redis_url = "redis://localhost:6379"

    config.browser = MagicMock()
    config.browser.automation_enabled = True
    config.browser.tier = "lightweight"

    config.performance = MagicMock()
    config.performance.max_workers = 4
    config.performance.batch_size = 100

    return config


@pytest.fixture
def mock_config_loader(mock_config):
    """Mock the ConfigLoader class."""
    mock_loader = MagicMock()
    mock_loader.load_config.return_value = mock_config
    mock_loader.from_file.return_value = mock_config
    return mock_loader


@pytest.fixture
def mock_client_manager():
    """Mock the ClientManager class."""
    mock_manager = MagicMock()
    mock_manager.get_qdrant_client.return_value = MagicMock()
    mock_manager.get_embedding_client.return_value = MagicMock()
    return mock_manager


@pytest.fixture
def mock_vector_db_manager():
    """Mock the VectorDBManager class with async methods."""
    mock_manager = AsyncMock()

    # Mock collection operations
    mock_manager.list_collections.return_value = [
        "collection1",
        "collection2",
        "test_collection",
    ]
    mock_manager.create_collection.return_value = True
    mock_manager.delete_collection.return_value = True
    mock_manager.get_collection_info.return_value = {
        "name": "test_collection",
        "vectors_count": 1000,
        "segments": 2,
        "status": "active",
    }

    # Mock document operations
    mock_manager.add_documents.return_value = {"ids": ["doc1", "doc2"], "success": True}
    mock_manager.search_documents.return_value = [
        {"id": "doc1", "score": 0.95, "payload": {"title": "Test Document 1"}},
        {"id": "doc2", "score": 0.88, "payload": {"title": "Test Document 2"}},
    ]
    mock_manager.delete_documents.return_value = {"deleted": 2, "success": True}

    # Mock cleanup
    mock_manager.cleanup.return_value = None

    return mock_manager


@pytest.fixture
def mock_health_checker():
    """Mock the ServiceHealthChecker class."""
    mock_checker = MagicMock()
    mock_checker.perform_all_health_checks.return_value = {
        "qdrant": {"connected": True, "version": "1.7.0"},
        "redis": {"connected": True, "version": "7.0.0"},
        "openai": {"connected": True, "model": "text-embedding-ada-002"},
    }
    return mock_checker


@pytest.fixture
def mock_cli_context(mock_config):
    """Create a mock Click context with configuration."""
    context = MagicMock()
    context.obj = {"config": mock_config}
    context.ensure_object.return_value = context.obj
    context.invoked_subcommand = None
    return context


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file for testing."""
    config_content = """
# AI Documentation Scraper Configuration
qdrant:
  host: localhost
  port: 6333
  timeout: 30
  prefer_grpc: true

openai:
  model: text-embedding-ada-002

cache:
  enabled: true
  redis_url: redis://localhost:6379

browser:
  automation_enabled: true
  tier: lightweight

performance:
  max_workers: 4
  batch_size: 100
"""
    config_file = tmp_path / "test_config.yml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def sample_collection_data():
    """Provide sample collection data for testing."""
    return [
        {
            "id": "doc1",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "payload": {
                "title": "Test Document 1",
                "url": "https://example.com/doc1",
                "content": "This is test content for document 1",
            },
        },
        {
            "id": "doc2",
            "vector": [0.2, 0.3, 0.4, 0.5],
            "payload": {
                "title": "Test Document 2",
                "url": "https://example.com/doc2",
                "content": "This is test content for document 2",
            },
        },
    ]


@pytest.fixture
def sample_batch_files(tmp_path):
    """Create sample files for batch processing tests."""
    files = []

    # Create test files
    for i in range(3):
        file_path = tmp_path / f"test_file_{i}.txt"
        file_path.write_text(f"This is test content for file {i}\nLine 2 of file {i}")
        files.append(str(file_path))

    return files


@pytest.fixture
def mock_completion_items():
    """Mock completion items for auto-completion testing."""
    from click.shell_completion import CompletionItem

    return [
        CompletionItem("collection1", help="Collection: collection1"),
        CompletionItem("collection2", help="Collection: collection2"),
        CompletionItem("test_collection", help="Collection: test_collection"),
    ]


@pytest.fixture(scope="function")
def event_loop():
    """Create an event loop for async testing."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def rich_output_capturer():
    """Helper fixture to capture and analyze Rich console output."""

    class RichOutputCapturer:
        def __init__(self):
            self.string_io = StringIO()
            self.console = Console(file=self.string_io, width=80, force_terminal=True)

        def get_output(self) -> str:
            """Get the captured output as a string."""
            return self.string_io.getvalue()

        def reset(self):
            """Reset the output buffer."""
            self.string_io = StringIO()
            self.console = Console(file=self.string_io, width=80, force_terminal=True)

        def assert_contains(self, text: str):
            """Assert that the output contains the specified text."""
            output = self.get_output()
            assert text in output, f"'{text}' not found in output: {output}"

        def assert_not_contains(self, text: str):
            """Assert that the output does not contain the specified text."""
            output = self.get_output()
            assert text not in output, f"'{text}' found in output: {output}"

        def get_lines(self) -> list[str]:
            """Get the output as a list of lines."""
            return self.get_output().split("\n")

    return RichOutputCapturer()


# Async testing utilities
class AsyncContextManager:
    """Helper for testing async context managers."""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Provide an async context manager for testing."""
    return AsyncContextManager


# Error simulation fixtures
@pytest.fixture
def connection_error():
    """Simulate connection errors for testing error handling."""
    return ConnectionError("Unable to connect to vector database")


@pytest.fixture
def timeout_error():
    """Simulate timeout errors for testing error handling."""
    return TimeoutError("Request timed out after 30 seconds")


@pytest.fixture
def configuration_error():
    """Simulate configuration errors for testing error handling."""
    return ValueError("Invalid configuration: missing required field")
