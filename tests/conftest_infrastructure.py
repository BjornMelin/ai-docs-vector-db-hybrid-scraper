"""Enhanced pytest configuration for comprehensive testing infrastructure.

This module extends the base conftest.py with additional fixtures and
configurations specifically for achieving >90% test coverage with modern
testing patterns.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
import respx
from hypothesis import HealthCheck, settings as hypothesis_settings


# Configure hypothesis for CI/CD
hypothesis_settings.register_profile(
    "ci",
    max_examples=100,
    deadline=5000,  # 5 seconds
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)

hypothesis_settings.register_profile(
    "dev",
    max_examples=10,
    deadline=2000,  # 2 seconds
    print_blob=True,
)

hypothesis_settings.register_profile(
    "debug",
    max_examples=1,
    deadline=None,
    print_blob=True,
    verbosity=hypothesis_settings.Verbosity.verbose,
)

# Load appropriate profile based on environment
if os.getenv("CI"):
    hypothesis_settings.load_profile("ci")
else:
    hypothesis_settings.load_profile("dev")

# Import test infrastructure
from tests.fixtures.test_infrastructure import *  # noqa: F403, E402


# ==============================================================================
# Test Session Configuration
# ==============================================================================


def pytest_configure(config):
    """Configure pytest session with enhanced settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "respx: mark test as using respx for HTTP mocking"
    )
    config.addinivalue_line(
        "markers", "hypothesis: mark test as using hypothesis for property testing"
    )
    config.addinivalue_line(
        "markers", "boundary_mock: mark test as using boundary-level mocking"
    )

    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["ENVIRONMENT"] = "test"


def pytest_sessionstart(session):
    """Setup test session."""
    # Ensure test databases/services are ready
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive test session with enhanced infrastructure")


def pytest_sessionfinish(session, exitstatus):
    """Cleanup after test session."""
    logger = logging.getLogger(__name__)
    logger.info(f"Test session finished with status: {exitstatus}")


# ==============================================================================
# Async Test Configuration
# ==============================================================================


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    if sys.platform.startswith("win"):
        # Windows requires special handling
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.get_event_loop_policy()


@pytest_asyncio.fixture(scope="function")
async def async_client():
    """Provide async client with automatic cleanup."""
    clients = []

    def client_factory(**kwargs):
        client = AsyncTestClient(**kwargs)
        clients.append(client)
        return client

    yield client_factory

    # Cleanup all created clients
    for client in clients:
        await client.cleanup()


# ==============================================================================
# HTTP Mocking Configuration
# ==============================================================================


@pytest.fixture(autouse=True)
def auto_respx():
    """Automatically enable respx for all tests."""
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as mock:
        # Configure default mocks for common external services
        _configure_default_mocks(mock)
        yield mock


def _configure_default_mocks(mock: respx.MockRouter):
    """Configure default HTTP mocks for common services."""
    # OpenAI API
    mock.route(
        method="POST",
        host="api.openai.com",
        path__regex=r"/v1/embeddings",
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1] * 1536}],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )
    )

    # Qdrant API
    mock.route(
        host="localhost",
        port=6333,
    ).mock(
        return_value=httpx.Response(
            200,
            json={"result": {"status": "ok"}},
        )
    )

    # Generic health checks
    mock.route(
        method="GET",
        path__regex=r"/health|/healthz|/ping",
    ).mock(return_value=httpx.Response(200, json={"status": "healthy"}))


# ==============================================================================
# Test Isolation
# ==============================================================================


@pytest.fixture(autouse=True)
async def isolate_tests(request):
    """Ensure test isolation."""
    # Save initial state
    initial_env = os.environ.copy()

    yield

    # Restore environment
    os.environ.clear()
    os.environ.update(initial_env)

    # Clear any singleton instances
    _clear_singletons()


def _clear_singletons():
    """Clear singleton instances between tests."""
    # Add any singleton clearing logic here


# ==============================================================================
# Performance Testing Fixtures
# ==============================================================================


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "target_latency_ms": {
            "api_endpoint": 100,
            "database_query": 50,
            "cache_hit": 5,
            "embedding_generation": 200,
        },
        "target_throughput_rps": {
            "search": 100,
            "indexing": 50,
            "health_check": 1000,
        },
        "resource_limits": {
            "memory_mb": 512,
            "cpu_percent": 80,
            "concurrent_connections": 100,
        },
    }


@pytest_asyncio.fixture
async def performance_monitor():
    """Monitor performance during tests."""
    from src.services.monitoring.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor()
    await monitor.start()
    yield monitor
    await monitor.stop()

    # Generate performance report
    report = monitor.get_report()
    if report["violations"]:
        logging.warning(f"Performance violations detected: {report['violations']}")


# ==============================================================================
# Integration Test Fixtures
# ==============================================================================


@pytest_asyncio.fixture
async def integration_services(tmp_path):
    """Setup integration test services."""
    from src.api.app_factory import create_app
    from src.config.settings import Settings

    # Create test settings
    test_settings = Settings(
        environment="test",
        debug=True,
        data_directory=str(tmp_path / "data"),
        cache_directory=str(tmp_path / "cache"),
        log_directory=str(tmp_path / "logs"),
    )

    # Create app with test settings
    app = create_app(settings=test_settings)

    # Start services
    await app.startup()

    yield {
        "app": app,
        "settings": test_settings,
        "tmp_path": tmp_path,
    }

    # Cleanup
    await app.shutdown()


# ==============================================================================
# Test Data Management
# ==============================================================================


@pytest.fixture
def test_data_manager(tmp_path):
    """Manage test data lifecycle."""

    class TestDataManager:
        def __init__(self, base_path: Path):
            self.base_path = base_path
            self.created_files = []

        def create_file(self, name: str, content: str) -> Path:
            """Create a test file."""
            file_path = self.base_path / name
            file_path.write_text(content)
            self.created_files.append(file_path)
            return file_path

        def cleanup(self):
            """Cleanup created files."""
            for file_path in self.created_files:
                if file_path.exists():
                    file_path.unlink()

    manager = TestDataManager(tmp_path)
    yield manager
    manager.cleanup()


# ==============================================================================
# Code Coverage Configuration
# ==============================================================================


@pytest.fixture(scope="session")
def coverage_thresholds():
    """Define coverage thresholds for different components."""
    return {
        "global": 90,  # >90% overall coverage
        "critical_paths": {
            "src/api": 95,
            "src/services/embeddings": 95,
            "src/services/vector_db": 95,
            "src/infrastructure/clients": 90,
        },
        "minimum_per_file": 80,
    }


def pytest_collection_modifyitems(config, items):
    """Modify test collection to ensure proper coverage."""
    # Add markers based on test location
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add async marker for async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# ==============================================================================
# Report Generation
# ==============================================================================


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate enhanced test summary."""
    if config.option.verbose:
        terminalreporter.section("Test Infrastructure Summary")
        terminalreporter.write_line(
            f"Total tests collected: {len(terminalreporter.stats.get('passed', [])) + len(terminalreporter.stats.get('failed', []))}"
        )
        terminalreporter.write_line("HTTP mocks used: respx")
        terminalreporter.write_line("Async framework: pytest-asyncio")
        terminalreporter.write_line("Property testing: hypothesis")

        if hasattr(config, "_coverage_data"):
            terminalreporter.section("Coverage Summary")
            terminalreporter.write_line(
                f"Overall coverage: {config._coverage_data.get('total', 0)}%"
            )


# ==============================================================================
# Custom Test Helpers
# ==============================================================================


class AsyncTestClient:
    """Enhanced async test client."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self._cleanup_tasks = []

    async def cleanup(self):
        """Cleanup client resources."""
        for task in self._cleanup_tasks:
            await task()


# Export all fixtures and utilities
__all__ = [
    "async_client",
    "auto_respx",
    "benchmark_config",
    "coverage_thresholds",
    "integration_services",
    "isolate_tests",
    "performance_monitor",
    "test_data_manager",
]
