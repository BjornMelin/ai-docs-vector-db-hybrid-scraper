"""Pytest configuration and shared fixtures."""

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

# Add src to path for all tests - eliminates need for import path manipulation
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for offline envs

    def load_dotenv(*args, **kwargs):
        """Fallback no-op load_dotenv implementation."""
        return False


try:
    from qdrant_client.models import PointStruct  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - basic fallback
    from dataclasses import dataclass

    @dataclass
    class PointStruct:  # type: ignore
        """Simplified stand-in for qdrant_client.models.PointStruct."""

        id: int
        vector: list[float]
        payload: dict


# Load test environment variables at module import
_test_env_path = Path(__file__).parent.parent / ".env.test"
if _test_env_path.exists():
    load_dotenv(_test_env_path, override=True)


# Remove the custom event_loop fixture to use pytest-asyncio's default
# The event loop scope is now configured in pyproject.toml


@pytest.fixture()
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture()
def mock_env_vars() -> Generator[None]:
    """Mock environment variables for testing."""
    # Save current values
    saved_vars = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "QDRANT_URL": os.environ.get("QDRANT_URL"),
    }

    # Set test values if not already set
    os.environ.setdefault("OPENAI_API_KEY", "test_key")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

    yield

    # Restore original values
    for key, value in saved_vars.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture()
def mock_qdrant_client() -> MagicMock:
    """Mock Qdrant client for testing."""
    client = MagicMock()
    client.create_collection = AsyncMock()
    client.delete_collection = AsyncMock()
    client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.count = AsyncMock(return_value=MagicMock(count=0))
    client.close = AsyncMock()
    return client


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client for testing."""
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)],
        ),
    )
    return client


@pytest.fixture()
def mock_crawl4ai() -> MagicMock:
    """Mock Crawl4AI AsyncWebCrawler for testing."""
    crawler = MagicMock()
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=None)
    crawler.arun = AsyncMock(
        return_value=MagicMock(
            success=True,
            cleaned_html="<p>Test content</p>",
            markdown="Test content",
            metadata={"title": "Test Page"},
            links={"internal": ["http://example.com/page1"]},
        ),
    )
    return crawler


@pytest.fixture()
def sample_documentation_site() -> dict:
    """Sample documentation site configuration."""
    return {
        "name": "test-docs",
        "url": "https://test.example.com",
        "max_depth": 2,
        "exclude_patterns": ["/api/", "/internal/"],
    }


@pytest.fixture()
def sample_crawl_result() -> dict:
    """Sample crawl result data."""
    return {
        "url": "https://test.example.com/docs/page1",
        "title": "Test Page",
        "content": "This is test content for the documentation page.",
        "markdown": "# Test Page\n\nThis is test content for the documentation page.",
        "metadata": {
            "description": "Test page description",
            "keywords": ["test", "documentation"],
        },
        "links": ["https://test.example.com/docs/page2"],
        "success": True,
        "error": None,
        "timestamp": "2024-01-01T00:00:00Z",
    }


@pytest.fixture()
def sample_vector_points() -> list[PointStruct]:
    """Sample vector points for testing."""
    return [
        PointStruct(
            id=1,
            vector=[0.1] * 1536,
            payload={
                "url": "https://test.example.com/docs/page1",
                "title": "Test Page 1",
                "content": "Test content 1",
                "chunk_index": 0,
            },
        ),
        PointStruct(
            id=2,
            vector=[0.2] * 1536,
            payload={
                "url": "https://test.example.com/docs/page2",
                "title": "Test Page 2",
                "content": "Test content 2",
                "chunk_index": 0,
            },
        ),
    ]
