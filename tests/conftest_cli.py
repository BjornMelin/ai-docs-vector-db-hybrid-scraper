"""CLI-specific test configuration and shared fixtures.

This module provides CLI-specific pytest configuration and fixtures
for comprehensive CLI testing with Rich console integration.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to Python path for testing
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def pytest_configure(config):
    """Configure pytest for CLI testing."""
    # Add custom markers
    config.addinivalue_line("markers", "cli: mark test as CLI-specific")
    config.addinivalue_line(
        "markers", "interactive: mark test as requiring interactive features"
    )
    config.addinivalue_line(
        "markers", "rich: mark test as requiring Rich console features"
    )
    config.addinivalue_line(
        "markers", "questionary: mark test as requiring questionary interactions"
    )
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for CLI tests."""
    for item in items:
        # Auto-mark CLI tests
        if "cli" in str(item.fspath):
            item.add_marker(pytest.mark.cli)

        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark tests using specific fixtures
        if hasattr(item, "fixturenames"):
            if "rich_output_capturer" in item.fixturenames:
                item.add_marker(pytest.mark.rich)
            if "questionary_mocker" in item.fixturenames:
                item.add_marker(pytest.mark.questionary)
            if "interactive_cli_runner" in item.fixturenames:
                item.add_marker(pytest.mark.interactive)


@pytest.fixture(scope="session", autouse=True)
def cli_test_environment():
    """Setup CLI test environment."""
    # Set environment variables for testing
    test_env = {
        "TESTING": "true",
        "CLI_TESTING": "true",
        "RICH_FORCE_TERMINAL": "true",  # Force Rich to use terminal features in tests
        "NO_COLOR": "0",  # Allow colors in tests
        "TERM": "xterm-256color",  # Set terminal type
    }

    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def cli_test_config():
    """Provide CLI test configuration."""
    return {
        "console_width": 80,
        "console_height": 24,
        "force_terminal": True,
        "no_color": False,
        "test_mode": True,
    }


@pytest.fixture
def mock_rich_dependencies():
    """Mock Rich dependencies for isolated testing."""
    with pytest.MonkeyPatch().context() as m:
        # Mock Rich imports if needed
        yield m


@pytest.fixture
def mock_questionary_dependencies():
    """Mock questionary dependencies for isolated testing."""
    with pytest.MonkeyPatch().context() as m:
        # Mock questionary imports if needed
        yield m


# Performance testing configuration
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time

    return Timer()


# CLI coverage configuration
def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption(
        "--runslow", default=False
    ):
        pytest.skip("need --runslow option to run")


def pytest_addoption(parser):
    """Add CLI-specific pytest options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runinteractive",
        action="store_true",
        default=False,
        help="run interactive tests",
    )
    parser.addoption(
        "--cli-coverage",
        action="store_true",
        default=False,
        help="run with CLI-specific coverage",
    )


# CLI test data fixtures
@pytest.fixture
def sample_cli_responses():
    """Sample CLI responses for testing."""
    return {
        "confirm_responses": [True, False, True],
        "select_responses": ["option1", "option2", "option3"],
        "text_responses": ["test_value", "another_value", ""],
        "password_responses": ["secret123", "password456"],
    }


@pytest.fixture
def sample_config_data():
    """Sample configuration data for CLI testing."""
    return {
        "minimal": {"qdrant": {"host": "localhost", "port": 6333}},
        "personal": {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"model": "text-embedding-3-small"},
            "browser": {"headless": True},
        },
        "development": {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"model": "text-embedding-3-small"},
            "debug": True,
            "log_level": "DEBUG",
        },
        "production": {
            "qdrant": {"url": "https://cloud.qdrant.io", "api_key": "prod-key"},
            "openai": {"model": "text-embedding-3-small"},
            "monitoring": {"enabled": True},
        },
    }


# Error simulation fixtures for CLI
@pytest.fixture
def cli_error_scenarios():
    """CLI error scenarios for testing."""
    return {
        "config_errors": [
            "Invalid JSON in config file",
            "Missing required configuration field",
            "Invalid API key format",
            "Database connection failed",
        ],
        "validation_errors": [
            "Port number out of range",
            "Invalid URL format",
            "File not found",
            "Permission denied",
        ],
        "runtime_errors": [
            "Network timeout",
            "Service unavailable",
            "Disk space full",
            "Memory allocation failed",
        ],
    }


# CLI command testing helpers
@pytest.fixture
def cli_command_tester():
    """Helper for testing CLI commands."""
    from click.testing import CliRunner

    class CLICommandTester:
        def __init__(self):
            self.runner = CliRunner()

        def test_command_help(self, command):
            """Test command help output."""
            result = self.runner.invoke(command, ["--help"])
            assert result.exit_code == 0
            assert len(result.output) > 0
            return result

        def test_command_version(self, command):
            """Test command version output."""
            result = self.runner.invoke(command, ["--version"])
            return result

        def test_command_with_args(self, command, args, expected_exit_code=0):
            """Test command with specific arguments."""
            result = self.runner.invoke(command, args)
            assert result.exit_code == expected_exit_code
            return result

    return CLICommandTester()


# Rich console testing helpers
@pytest.fixture
def rich_testing_utils():
    """Utilities for testing Rich console output."""

    class RichTestingUtils:
        @staticmethod
        def extract_text_content(rich_output):
            """Extract plain text from Rich output."""
            # Remove ANSI escape codes
            import re

            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return ansi_escape.sub("", rich_output)

        @staticmethod
        def count_style_markers(rich_output, marker):
            """Count style markers in Rich output."""
            return rich_output.count(marker)

        @staticmethod
        def find_panels(rich_output):
            """Find panel structures in Rich output."""
            # Simple panel detection
            return [
                line for line in rich_output.split("\n") if "┌" in line or "├" in line
            ]

        @staticmethod
        def find_tables(rich_output):
            """Find table structures in Rich output."""
            # Simple table detection
            return [line for line in rich_output.split("\n") if "│" in line]

    return RichTestingUtils()


# Questionary testing helpers
@pytest.fixture
def questionary_testing_utils():
    """Utilities for testing questionary interactions."""

    class QuestionaryTestingUtils:
        @staticmethod
        def simulate_user_flow(responses_dict):
            """Simulate a complete user interaction flow."""
            from unittest.mock import patch

            patches = []

            if "confirm" in responses_dict:
                confirm_patch = patch("questionary.confirm")
                confirm_mock = confirm_patch.start()
                confirm_mock.return_value.ask.side_effect = responses_dict["confirm"]
                patches.append(confirm_patch)

            if "select" in responses_dict:
                select_patch = patch("questionary.select")
                select_mock = select_patch.start()
                select_mock.return_value.ask.side_effect = responses_dict["select"]
                patches.append(select_patch)

            if "text" in responses_dict:
                text_patch = patch("questionary.text")
                text_mock = text_patch.start()
                text_mock.return_value.ask.side_effect = responses_dict["text"]
                patches.append(text_patch)

            if "password" in responses_dict:
                password_patch = patch("questionary.password")
                password_mock = password_patch.start()
                password_mock.return_value.ask.side_effect = responses_dict["password"]
                patches.append(password_patch)

            return patches

        @staticmethod
        def cleanup_patches(patches):
            """Clean up questionary patches."""
            for patch_obj in patches:
                patch_obj.stop()

    return QuestionaryTestingUtils()


# CLI coverage collection
@pytest.fixture(autouse=True)
def cli_coverage_collector(request):
    """Collect CLI-specific coverage data."""
    if request.config.getoption("--cli-coverage"):
        # Setup CLI coverage collection
        import coverage

        cov = coverage.Coverage(source=["src/cli"])
        cov.start()

        yield

        cov.stop()
        cov.save()
    else:
        yield
