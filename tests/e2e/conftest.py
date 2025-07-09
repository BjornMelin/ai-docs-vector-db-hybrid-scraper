"""Pytest configuration for E2E tests using Playwright MCP tools."""

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest


# Test configuration constants
DEFAULT_TIMEOUT = 30000  # 30 seconds
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
MOBILE_VIEWPORT = {"width": 375, "height": 667}
TABLET_VIEWPORT = {"width": 768, "height": 1024}

# Test data directories
TEST_DATA_DIR = Path(__file__).parent / "test_data"
SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"
REPORTS_DIR = Path(__file__).parent / "reports"

# Ensure directories exist
for directory in [TEST_DATA_DIR, SCREENSHOTS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def app_url() -> str:
    """Get the application URL for testing."""
    return os.getenv("TEST_APP_URL", "http://localhost:8000")


@pytest.fixture(params=["chromium", "firefox", "webkit"])
async def browser_type(request) -> str:
    """Parameterized fixture for cross-browser testing."""
    return request.param


@pytest.fixture(params=[DEFAULT_VIEWPORT, MOBILE_VIEWPORT, TABLET_VIEWPORT])
async def viewport_config(request) -> dict[str, int]:
    """Parameterized fixture for responsive design testing."""
    return request.param


@pytest.fixture
async def browser_session(
    browser_type: str, viewport_config: dict[str, int]
) -> AsyncGenerator[dict[str, Any]]:
    """Create a browser session with specified configuration."""
    session_config = {
        "browser_type": browser_type,
        "viewport": viewport_config,
        "timeout": DEFAULT_TIMEOUT,
        "headless": os.getenv("HEADLESS", "true").lower() == "true",
    }

    # Store session info for test cleanup
    return {
        "config": session_config,
        "screenshots": [],
        "console_logs": [],
        "performance_metrics": {},
    }

    # Cleanup: Any cleanup logic would go here


@pytest.fixture
async def test_document() -> Path:
    """Provide a test document for upload scenarios."""
    test_doc = TEST_DATA_DIR / "sample_document.pdf"

    # Create a minimal test document if it doesn't exist
    if not test_doc.exists():
        # For now, create a placeholder file
        # In real implementation, this would be a proper PDF
        test_doc.write_text("Sample test document content for E2E testing")

    return test_doc


@pytest.fixture
async def quality_gate_thresholds() -> dict[str, float]:
    """Define the Minimum Delight quality gate thresholds."""
    return {
        "page_load_time": 2.0,  # seconds
        "search_response_time": 0.5,  # seconds
        "ui_interaction_response": 0.1,  # seconds
        "error_recovery_time": 5.0,  # seconds
        "accessibility_score": 0.9,  # WCAG 2.1 AA compliance ratio
    }


@pytest.fixture
async def performance_metrics() -> dict[str, Any]:
    """Initialize performance metrics collection."""
    return {
        "start_time": None,
        "end_time": None,
        "page_load_times": [],
        "search_response_times": [],
        "ui_interaction_times": [],
        "memory_usage": [],
        "cpu_usage": [],
        "network_requests": [],
    }


@pytest.mark.asyncio
async def pytest_configure(config):
    """Configure pytest for E2E testing."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line(
        "markers", "cross_browser: marks tests for cross-browser validation"
    )
    config.addinivalue_line("markers", "mobile: marks tests for mobile device testing")
    config.addinivalue_line(
        "markers", "accessibility: marks tests for accessibility validation"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests for performance validation"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add appropriate markers."""
    for item in items:
        # Add e2e marker to all tests in e2e directory
        if "tests/e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add slow marker to tests that might take longer
        if any(
            keyword in item.name.lower()
            for keyword in ["load", "stress", "concurrent", "performance"]
        ):
            item.add_marker(pytest.mark.slow)

        # Add cross_browser marker to parameterized browser tests
        if hasattr(item, "callspec") and "browser_type" in item.callspec.params:
            item.add_marker(pytest.mark.cross_browser)

        # Add mobile marker to mobile viewport tests
        if hasattr(item, "callspec") and "viewport_config" in item.callspec.params:
            viewport = item.callspec.params["viewport_config"]
            if viewport["width"] < 768:
                item.add_marker(pytest.mark.mobile)


@pytest.fixture(autouse=True)
async def test_setup_teardown(request):
    """Setup and teardown for each test."""
    test_name = request.node.name

    # Pre-test setup
    print(f"\n🧪 Starting E2E test: {test_name}")

    yield

    # Post-test cleanup
    print(f"✅ Completed E2E test: {test_name}")


# Utility functions for test helpers
def get_screenshot_path(test_name: str, browser_type: str, suffix: str = "") -> Path:
    """Generate a screenshot file path for a test."""
    safe_test_name = "".join(
        c for c in test_name if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    filename = f"{safe_test_name}_{browser_type}{suffix}.png"
    return SCREENSHOTS_DIR / filename


def get_report_path(test_name: str, file_type: str = "json") -> Path:
    """Generate a report file path for a test."""
    safe_test_name = "".join(
        c for c in test_name if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    filename = f"{safe_test_name}_report.{file_type}"
    return REPORTS_DIR / filename
