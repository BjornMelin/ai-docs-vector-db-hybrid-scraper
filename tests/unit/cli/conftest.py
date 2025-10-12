"""Enhanced fixtures for CLI testing."""

import asyncio
import json
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.shell_completion import CompletionItem
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
    config.cache.dragonfly_url = "redis://localhost:6379"

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
    mock_loader.load_settings.return_value = mock_config
    mock_loader.from_file.return_value = mock_config
    return mock_loader


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
        {"id": "doc1", "score": 0.95, "metadata": {"title": "Test Document 1"}},
        {"id": "doc2", "score": 0.88, "metadata": {"title": "Test Document 2"}},
    ]
    mock_manager.delete_documents.return_value = {"deleted": 2, "success": True}

    # Mock cleanup
    mock_manager.cleanup.return_value = None

    return mock_manager


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
  dragonfly_url: redis://localhost:6379

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
            "metadata": {
                "title": "Test Document 1",
                "url": "https://example.com/doc1",
                "content": "This is test content for document 1",
            },
        },
        {
            "id": "doc2",
            "vector": [0.2, 0.3, 0.4, 0.5],
            "metadata": {
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

    return [
        CompletionItem("collection1", help="Collection: collection1"),
        CompletionItem("collection2", help="Collection: collection2"),
        CompletionItem("test_collection", help="Collection: test_collection"),
    ]


@pytest.fixture
def event_loop():
    """Provide a dedicated event loop for async tests."""

    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture
def rich_output_capturer():
    """Enhanced fixture to capture and analyze Rich console output with  debugging."""

    class RichOutputCapturer:
        def __init__(self):
            self.string_io = StringIO()
            self.console = Console(
                file=self.string_io,
                width=80,
                height=24,
                force_terminal=True,
                no_color=False,
                _environ={},
            )
            self.captured_calls = []

        def get_output(self) -> str:
            """Get the captured output as a string."""
            return self.string_io.getvalue()

        def get_plain_output(self) -> str:
            """Get output without ANSI escape codes."""

            buffer = StringIO()
            plain_console = Console(file=buffer, no_color=True, width=80)
            plain_console.print(self.get_output(), end="")
            return buffer.getvalue()

        def reset(self):
            """Reset the output buffer."""
            self.string_io = StringIO()
            self.console = Console(
                file=self.string_io,
                width=80,
                height=24,
                force_terminal=True,
                no_color=False,
            )
            self.captured_calls.clear()

        def assert_contains(self, text: str, case_sensitive: bool = True):
            """Assert that the output contains the specified text."""
            output = self.get_output()
            if not case_sensitive:
                output = output.lower()
                text = text.lower()
            assert text in output, f"'{text}' not found in output: {output[:500]}..."

        def assert_not_contains(self, text: str, case_sensitive: bool = True):
            """Assert that the output does not contain the specified text."""
            output = self.get_output()
            if not case_sensitive:
                output = output.lower()
                text = text.lower()
            assert text not in output, f"'{text}' found in output: {output[:500]}..."

        def assert_panel_title(self, title: str):
            """Assert that a panel with specific title exists."""
            self.assert_contains(title)

        def assert_table_headers(self, *headers: str):
            """Assert that table headers are present."""
            for header in headers:
                self.assert_contains(header)

        def get_lines(self) -> list[str]:
            """Get the output as a list of lines."""
            return self.get_output().split("\n")

        def count_occurrences(self, text: str) -> int:
            """Count occurrences of text in output."""
            return self.get_output().count(text)

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


# Modern CLI Testing Fixtures


@pytest.fixture
def questionary_mocker():
    """Enhanced fixture for mocking questionary interactions with realistic flows."""

    class QuestionaryMocker:
        def __init__(self):
            self.responses = {}
            self.call_history = []
            self.patches = []

        def setup_responses(self, responses_dict: dict):
            """Setup responses for different questionary methods."""
            self.responses = responses_dict

        def mock_confirm(self, question: str | None = None, default: bool = True):
            """Mock questionary.confirm() calls."""
            self.call_history.append(("confirm", question, default))
            return self.responses.get("confirm", default)

        def mock_select(self, question: str | None = None, choices=None, default=None):
            """Mock questionary.select() calls."""
            self.call_history.append(("select", question, choices, default))
            return self.responses.get(
                "select", default or (choices[0].value if choices else None)
            )

        def mock_text(self, question: str | None = None, default: str = ""):
            """Mock questionary.text() calls."""
            self.call_history.append(("text", question, default))
            return self.responses.get("text", default)

        def mock_password(self, question: str | None = None):
            """Mock questionary.password() calls."""
            self.call_history.append(("password", question))
            return self.responses.get("password", "test-password")

        def start_mocking(self):
            """Start mocking questionary methods."""
            self.patches = [
                patch("questionary.confirm", side_effect=self.mock_confirm),
                patch("questionary.select", side_effect=self.mock_select),
                patch("questionary.text", side_effect=self.mock_text),
                patch("questionary.password", side_effect=self.mock_password),
            ]
            for p in self.patches:
                p.start()

        def stop_mocking(self):
            """Stop mocking questionary methods."""
            for p in self.patches:
                p.stop()
            self.patches.clear()

        def assert_called_with(self, method: str, question_pattern: str | None = None):
            """Assert that a method was called with specific question."""
            calls = [call for call in self.call_history if call[0] == method]
            assert len(calls) > 0, f"No {method} calls found in {self.call_history}"
            if question_pattern:
                matching_calls = [
                    call for call in calls if question_pattern in str(call[1])
                ]
                assert len(matching_calls) > 0, (
                    f"No {method} calls matching '{question_pattern}'"
                )

    mocker = QuestionaryMocker()
    yield mocker
    mocker.stop_mocking()


@pytest.fixture
def interactive_cli_runner():
    """Enhanced CLI runner for testing interactive flows."""

    class InteractiveCLIRunner(CliRunner):
        def __init__(self):
            super().__init__()
            self.input_responses = []
            self.current_response_index = 0

        def set_input_responses(self, responses: list[str]):
            """Set pre-defined responses for interactive inputs."""
            self.input_responses = responses
            self.current_response_index = 0

        def invoke_interactive(self, cli, args=None, input_data=None, **extra):
            """Invoke CLI with interactive input simulation."""
            if input_data is None and self.input_responses:
                input_data = "\\n".join(self.input_responses) + "\\n"

            return self.invoke(cli, args or [], input=input_data, **extra)

        def simulate_keyboard_interrupt(self, cli, args=None, **extra):
            """Simulate keyboard interrupt during CLI execution."""
            with patch("click.abort", side_effect=KeyboardInterrupt):
                return self.invoke(cli, args or [], **extra)

    return InteractiveCLIRunner()


@pytest.fixture
def mock_wizard_components():
    """Mock wizard components for isolated testing."""
    components = {
        "template_manager": MagicMock(),
        "profile_manager": MagicMock(),
        "validator": MagicMock(),
        "config_auditor": MagicMock(),
    }

    # Setup realistic returns
    components["template_manager"].list_templates.return_value = [
        "personal-use",
        "development",
        "production",
        "minimal",
    ]
    components["template_manager"].get_template.return_value = {
        "qdrant": {"host": "localhost", "port": 6333},
        "embeddings": {"provider": "openai"},
    }

    components["profile_manager"].list_profiles.return_value = [
        "personal",
        "development",
        "production",
    ]
    components["profile_manager"].create_profile_config.return_value = Path(
        "/tmp/config.json"  # test temp path
    )

    components["validator"].validate_api_key.return_value = (True, None)
    components["validator"].validate_url.return_value = (True, None)
    components["validator"].validate_and_show_errors.return_value = True

    return components


@pytest.fixture
def temp_profiles_dir(tmp_path):
    """Create temporary profiles directory with sample profiles."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    # Create sample profile files
    profiles = {
        "personal.json": {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"model": "text-embedding-3-small"},
            "browser": {"headless": True},
        },
        "development.json": {
            "qdrant": {"host": "localhost", "port": 6333},
            "debug": True,
            "log_level": "DEBUG",
        },
        "production.json": {
            "qdrant": {"url": "https://qdrant.example.com", "api_key": "prod-key"},
            "monitoring": {"enabled": True},
        },
    }

    for filename, config in profiles.items():
        (profiles_dir / filename).write_text(json.dumps(config, indent=2))

    return profiles_dir


@pytest.fixture
def cli_integration_setup(tmp_path, temp_profiles_dir):
    """Complete setup for CLI integration testing."""
    setup_data = {
        "config_dir": tmp_path / "config",
        "profiles_dir": temp_profiles_dir,
        "temp_config": tmp_path / "test_config.json",
        "env_file": tmp_path / ".env",
    }

    # Create directories
    setup_data["config_dir"].mkdir(exist_ok=True)

    # Create base config
    base_config = {"qdrant": {"host": "localhost", "port": 6333}, "version": "1.0.0"}
    setup_data["temp_config"].write_text(json.dumps(base_config, indent=2))

    return setup_data


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
