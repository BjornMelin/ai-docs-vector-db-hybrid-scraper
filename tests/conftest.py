"""Pytest configuration and shared fixtures.

This module provides the core testing infrastructure with standardized fixtures,
configuration, and utilities that follow 2025 testing best practices.
Implements session-scoped fixtures, parallel execution support, and boundary-only mocking.
"""

import asyncio
import gc
import math
import os
import random
import sys
import tempfile
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import respx
except ImportError:  # pragma: no cover
    respx = None


# Add project root to path for src imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add src directory for backwards compatibility
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for offline envs

    def load_dotenv(*_args, **_kwargs):
        """Fallback no-op load_dotenv implementation."""
        return False


try:
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from qdrant_client.models.models import CollectionInfo, CollectionStatus
except ModuleNotFoundError:  # pragma: no cover - basic fallback
    from dataclasses import dataclass
    from enum import Enum

    class Distance(Enum):
        """Fallback Distance enum."""

        COSINE = "Cosine"
        EUCLID = "Euclid"
        DOT = "Dot"

    @dataclass
    class VectorParams:
        """Fallback VectorParams."""

        size: int
        distance: Distance

    @dataclass
    class PointStruct:  # type: ignore[misc]
        """Simplified stand-in for qdrant_client.models.PointStruct."""

        id: int
        vector: list[float]
        payload: dict

    @dataclass
    class CollectionStatus:
        """Fallback CollectionStatus."""

        status: str = "green"

    @dataclass
    class CollectionInfo:
        """Fallback CollectionInfo."""

        status: CollectionStatus
        vectors_count: int = 0
        points_count: int = 0


# Load test environment variables at module import
_test_env_path = Path(__file__).parent.parent / ".env.test"
if _test_env_path.exists():
    load_dotenv(_test_env_path, override=True)


def is_windows():
    return sys.platform.startswith("win")


def is_macos():
    return sys.platform == "darwin"


def is_linux():
    return sys.platform.startswith("linux")


def is_ci_environment():
    return bool(os.getenv("CI") or os.getenv("GITHUB_ACTIONS"))


def set_platform_environment_defaults():
    return {}


def get_playwright_browser_path():
    return None


@pytest.fixture(scope="session", autouse=True)
def app_config():
    """Shared configuration for entire test session.

    Session-scoped fixture that provides application configuration
    and sets up test environment with proper cleanup.
    """
    original_env = os.environ.copy()

    # Test configuration
    config = {
        "test_mode": True,
        "ci_environment": is_ci_environment(),
        "parallel_workers": get_parallel_worker_count(),
        "timeouts": {"default": 30, "browser": 60, "network": 15, "database": 10},
        "test_dirs": [
            "tests/fixtures/cache",
            "tests/fixtures/data",
            "tests/fixtures/logs",
            "logs",
            "cache",
            "data",
        ],
    }

    try:
        # Set test environment variables
        env_defaults = set_platform_environment_defaults()
        for key, value in env_defaults.items():
            if key not in os.environ:
                os.environ[key] = value

        # CI-specific configuration
        if config["ci_environment"]:
            os.environ.update(
                {
                    "CRAWL4AI_HEADLESS": "true",
                    "CRAWL4AI_SKIP_BROWSER_DOWNLOAD": "false",
                    "PLAYWRIGHT_CHROMIUM_SANDBOX": "false",
                    "PYTEST_TIMEOUT": "300",
                    "PYTEST_XDIST_WORKER_COUNT": str(config["parallel_workers"]),
                }
            )

        # Create test directories
        project_root = Path(__file__).parent.parent
        for test_dir in config["test_dirs"]:
            (project_root / test_dir).mkdir(parents=True, exist_ok=True)

        yield config

    finally:
        # Restore environment
        os.environ.clear()
        os.environ.update(original_env)


def get_parallel_worker_count() -> int:
    """Calculate optimal worker count for parallel test execution."""
    if is_ci_environment():
        # CI environments - be conservative with resources
        cpu_count = os.cpu_count() or 1
        return min(4, max(1, cpu_count // 2))
    # Local development - use fewer workers
    return min(2, os.cpu_count() or 1)


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        yield Path(temp_dir_str)


@pytest.fixture
def mock_env_vars() -> Generator[None]:
    """Mock environment variables for testing."""
    saved_vars = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "QDRANT_URL": os.environ.get("QDRANT_URL"),
    }

    os.environ.setdefault("OPENAI_API_KEY", "test_key")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

    yield

    for key, value in saved_vars.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Mock Qdrant client for testing with comprehensive method mocking."""
    client = MagicMock()

    # Collection operations
    client.create_collection = AsyncMock()
    client.delete_collection = AsyncMock()
    client.recreate_collection = AsyncMock()
    client.update_collection = AsyncMock()
    client.get_collection = AsyncMock(
        return_value=CollectionInfo(
            status=CollectionStatus(status="green"), vectors_count=0, points_count=0
        )
    )
    client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    client.collection_exists = AsyncMock(return_value=False)

    # Point operations
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.search_batch = AsyncMock(return_value=[])
    client.scroll = AsyncMock(return_value=([], None))
    client.retrieve = AsyncMock(return_value=[])
    client.count = AsyncMock(return_value=MagicMock(count=0))
    client.delete = AsyncMock()

    # Alias operations
    client.update_collection_aliases = AsyncMock()
    client.get_collection_aliases = AsyncMock(return_value=MagicMock(aliases=[]))

    # Snapshot operations
    client.create_snapshot = AsyncMock()
    client.list_snapshots = AsyncMock(return_value=[])
    client.delete_snapshot = AsyncMock()

    # Connection
    client.close = AsyncMock()

    return client


@pytest_asyncio.fixture
async def async_qdrant_client() -> AsyncGenerator[MagicMock]:
    """Async Qdrant client fixture with proper cleanup."""
    client = mock_qdrant_client()
    yield client
    await client.close()


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client for testing with comprehensive API coverage."""
    client = MagicMock()

    # Embeddings API
    client.embeddings.create = AsyncMock(
        return_value=MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)],
            model="text-embedding-3-small",
            usage=MagicMock(prompt_tokens=10, total_tokens=10),
        )
    )

    # Chat API (for HyDE)
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="Generated hypothetical document"),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        )
    )

    # Models API
    client.models.list = AsyncMock(
        return_value=MagicMock(
            data=[
                MagicMock(id="text-embedding-3-small"),
                MagicMock(id="text-embedding-3-large"),
            ]
        )
    )

    return client


@pytest_asyncio.fixture
async def async_openai_client() -> AsyncGenerator[MagicMock]:
    """Async OpenAI client fixture."""
    yield mock_openai_client()


@pytest.fixture
def mock_redis_client() -> MagicMock:
    """Mock Redis client for testing with async support."""
    client = MagicMock()

    # Basic operations
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=0)
    client.expire = AsyncMock(return_value=True)
    client.ttl = AsyncMock(return_value=-2)

    # Hash operations
    client.hget = AsyncMock(return_value=None)
    client.hset = AsyncMock(return_value=1)
    client.hgetall = AsyncMock(return_value={})
    client.hdel = AsyncMock(return_value=1)

    # List operations
    client.lpush = AsyncMock(return_value=1)
    client.rpop = AsyncMock(return_value=None)
    client.llen = AsyncMock(return_value=0)

    # Set operations
    client.sadd = AsyncMock(return_value=1)
    client.srem = AsyncMock(return_value=1)
    client.smembers = AsyncMock(return_value=set())

    # Pub/Sub
    client.publish = AsyncMock(return_value=0)

    # Connection
    client.ping = AsyncMock(return_value=True)
    client.close = AsyncMock()
    client.aclose = AsyncMock()

    return client


@pytest_asyncio.fixture
async def async_redis_client() -> AsyncGenerator[MagicMock]:
    """Async Redis client fixture with proper cleanup."""
    client = mock_redis_client()
    yield client
    await client.aclose()


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    """Mock httpx async client for testing."""
    client = MagicMock()

    # Response mock
    response = MagicMock()
    response.status_code = 200
    response.text = "<html><body>Test content</body></html>"
    response.json = MagicMock(return_value={"status": "ok"})
    response.headers = {"content-type": "text/html"}
    response.raise_for_status = MagicMock()

    # Request methods
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.put = AsyncMock(return_value=response)
    client.delete = AsyncMock(return_value=response)
    client.patch = AsyncMock(return_value=response)

    # Connection
    client.aclose = AsyncMock()

    return client


@pytest_asyncio.fixture
async def async_httpx_client() -> AsyncGenerator[MagicMock]:
    """Async httpx client fixture with proper cleanup."""
    client = mock_httpx_client()
    yield client
    await client.aclose()


@pytest.fixture
def mock_aiohttp_session() -> MagicMock:
    """Mock aiohttp client session for testing."""
    session = MagicMock()

    # Response mock
    response = MagicMock()
    response.status = 200
    response.text = AsyncMock(return_value="<html><body>Test content</body></html>")
    response.json = AsyncMock(return_value={"status": "ok"})
    response.headers = {"content-type": "text/html"}
    response.raise_for_status = MagicMock()

    # Context manager for requests
    @asynccontextmanager
    async def mock_request(*args, **kwargs):
        yield response

    # Request methods
    session.get = mock_request
    session.post = mock_request
    session.put = mock_request
    session.delete = mock_request
    session.patch = mock_request

    # Connection
    session.close = AsyncMock()

    return session


@pytest_asyncio.fixture
async def async_aiohttp_session() -> AsyncGenerator[MagicMock]:
    """Async aiohttp session fixture with proper cleanup."""
    session = mock_aiohttp_session()
    yield session
    await session.close()


@pytest.fixture
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
                "metadata": {"source": "test", "timestamp": "2024-01-01T00:00:00Z"},
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
                "metadata": {"source": "test", "timestamp": "2024-01-02T00:00:00Z"},
            },
        ),
    ]


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Mock configuration object for testing."""
    return {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_docs",
            "vector_size": 1536,
            "distance": "Cosine",
        },
        "openai": {
            "api_key": "test-key",
            "model": "text-embedding-3-small",
            "batch_size": 100,
            "max_retries": 3,
        },
        "redis": {"host": "localhost", "port": 6379, "db": 0, "decode_responses": True},
        "crawler": {"max_concurrent": 5, "timeout": 30, "user_agent": "TestBot/1.0"},
        "processing": {"chunk_size": 1000, "chunk_overlap": 200, "min_chunk_size": 100},
    }


@pytest.fixture
def isolated_db_session():
    """Per-test database isolation with automatic rollback.

    Provides database session isolation for tests by ensuring
    all changes are rolled back after each test.
    """
    session = MagicMock()
    transaction = MagicMock()

    # Setup transaction methods
    session.begin = MagicMock(return_value=transaction)
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.close = MagicMock()

    # Start transaction for isolation
    transaction_context = session.begin()

    try:
        yield session
    finally:
        # Always rollback to ensure test isolation
        session.rollback()
        session.close()


try:
    from hypothesis import strategies as st

    @st.composite
    def embedding_strategy(
        draw, min_dim: int = 128, max_dim: int = 1536, normalized: bool = True
    ):
        """Hypothesis strategy for generating realistic embedding vectors."""
        # Use common embedding dimensions
        common_dims = [128, 256, 384, 512, 768, 1024, 1536]
        valid_dims = [d for d in common_dims if min_dim <= d <= max_dim]

        if valid_dims:
            dim = draw(st.sampled_from(valid_dims))
        else:
            dim = draw(st.integers(min_value=min_dim, max_value=max_dim))

        # Generate realistic embedding values
        values = draw(
            st.lists(
                st.floats(
                    min_value=-1.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,  # Use 32-bit floats for consistency
                ),
                min_size=dim,
                max_size=dim,
            )
        )

        if normalized and values:
            # Normalize to unit vector
            norm = sum(x**2 for x in values) ** 0.5
            if norm > 0:
                values = [x / norm for x in values]
            else:
                # Handle zero vector case
                values = [1.0] + [0.0] * (len(values) - 1)

        return values

    @st.composite
    def document_strategy(draw, min_length: int = 10, max_length: int = 500):
        """Hypothesis strategy for generating realistic document text."""
        # Generate more realistic text patterns
        words = draw(
            st.lists(
                st.text(
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll"),
                        min_codepoint=65,
                        max_codepoint=122,
                    ),
                    min_size=2,
                    max_size=12,
                ),
                min_size=2,
                max_size=max_length // 5,  # Approximate word count
            )
        )

        text = " ".join(words)

        # Ensure length constraints
        if len(text) < min_length:
            text = text + " " + "content" * ((min_length - len(text)) // 7 + 1)
        elif len(text) > max_length:
            text = text[:max_length].rsplit(" ", 1)[0]

        return text.strip()

except ImportError:
    # Fallback if Hypothesis not available
    def embedding_strategy(*args, **kwargs):
        """Fallback embedding strategy when Hypothesis not available."""
        dim = kwargs.get("max_dim", 384)
        normalized = kwargs.get("normalized", True)

        values = [random.uniform(-1, 1) for _ in range(dim)]
        if normalized:
            norm = sum(x**2 for x in values) ** 0.5
            if norm > 0:
                values = [x / norm for x in values]
        return values

    def document_strategy(*args, **kwargs):
        """Fallback document strategy when Hypothesis not available."""
        min_length = kwargs.get("min_length", 10)
        max_length = kwargs.get("max_length", 500)

        words = [
            "test",
            "content",
            "document",
            "information",
            "data",
            "analysis",
            "research",
        ]
        word_count = random.randint(max(2, min_length // 5), max_length // 5)
        text = " ".join(random.choices(words, k=word_count))

        # Ensure length constraints
        if len(text) < min_length:
            text = text + " " + "content" * ((min_length - len(text)) // 7 + 1)
        elif len(text) > max_length:
            text = text[:max_length].rsplit(" ", 1)[0]

        return text.strip()


@pytest.fixture
def ai_test_utilities():
    """AI/ML testing utilities for embeddings, similarity, and model validation."""

    class AITestUtilities:
        @staticmethod
        def assert_valid_embedding(embedding: list[float], expected_dim: int = 1536):
            """Assert embedding meets quality criteria."""
            assert len(embedding) == expected_dim, (
                f"Expected {expected_dim}D, got {len(embedding)}D"
            )
            assert all(isinstance(x, int | float) for x in embedding), (
                "All values must be numeric"
            )

            assert not any(math.isnan(x) or math.isinf(x) for x in embedding), (
                "No NaN/Inf values"
            )

            # Check for reasonable value range (normalized embeddings should be < 1)
            assert all(abs(x) <= 2.0 for x in embedding), (
                "Values outside reasonable range"
            )

            # Check vector is not zero
            norm = sum(x**2 for x in embedding) ** 0.5
            assert norm > 0.01, "Vector too close to zero"

        @staticmethod
        def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
            """Calculate cosine similarity between vectors."""
            if len(vec1) != len(vec2):
                msg = f"Dimension mismatch: {len(vec1)} vs {len(vec2)}"
                raise ValueError(msg)

            dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        @staticmethod
        def generate_test_embeddings(
            count: int = 10, dim: int = 1536
        ) -> list[list[float]]:
            """Generate deterministic test embeddings."""
            embeddings = []

            for i in range(count):
                random.seed(42 + i)  # Deterministic
                embedding = [random.uniform(-1, 1) for _ in range(dim)]

                # Normalize
                norm = sum(x**2 for x in embedding) ** 0.5
                if norm > 0:
                    embedding = [x / norm for x in embedding]

                embeddings.append(embedding)

            return embeddings

    return AITestUtilities()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Add any singleton resets here
    return
    # Cleanup after test


@pytest.fixture
def mock_crawl4ai_response():
    """Mock response from Crawl4AI scraper."""
    return {
        "url": "https://test.example.com",
        "html": "<html><body><h1>Test Page</h1><p>Test content</p></body></html>",
        "cleaned_html": "<h1>Test Page</h1><p>Test content</p>",
        "markdown": "# Test Page\n\nTest content",
        "text": "Test Page\nTest content",
        "metadata": {
            "title": "Test Page",
            "description": "Test description",
            "keywords": ["test", "example"],
            "language": "en",
        },
        "links": ["https://test.example.com/link1", "https://test.example.com/link2"],
        "images": [],
        "status_code": 200,
        "error": None,
    }


@pytest.fixture
def respx_mock():
    """Modern respx router for boundary HTTP mocking.

    Provides respx router with common API mocks pre-configured.
    Follows boundary-only mocking principle for external HTTP calls.
    """
    if respx is None:
        pytest.skip("respx not available")

    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        # OpenAI API mocks
        router.post("https://api.openai.com/v1/embeddings").mock(
            return_value=respx.MockResponse(
                status_code=200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
                headers={
                    "x-ratelimit-limit-requests": "10000",
                    "x-ratelimit-remaining-requests": "9999",
                },
            )
        )

        # Generic test endpoints
        router.get("https://test.example.com").mock(
            return_value=respx.MockResponse(
                status_code=200,
                html="<html><body>Test content</body></html>",
                headers={"content-type": "text/html"},
            )
        )

        # API error simulation
        router.get("https://api.error.com/500").mock(
            return_value=respx.MockResponse(status_code=500)
        )

        # Rate limit simulation
        router.get("https://api.ratelimit.com/test").mock(
            return_value=respx.MockResponse(
                status_code=429,
                json={"error": "Rate limit exceeded"},
                headers={"retry-after": "60"},
            )
        )

        yield router


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def mock_mcp_context():
    """Mock MCP (Model Context Protocol) context for testing MCP tools."""
    context = MagicMock()
    context.info = MagicMock()
    context.debug = MagicMock()
    context.warning = MagicMock()
    context.error = MagicMock()
    context.report_progress = MagicMock()
    context.read_resource = AsyncMock(return_value="resource data")
    context.request_id = "test-request-123"
    context.client_id = "test-client"
    return context


@pytest.fixture
def performance_monitor():
    """Advanced performance monitoring for tests.

    Provides comprehensive performance metrics including
    memory usage, execution time, and resource monitoring.
    """

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.start_cpu = None
            self.metrics = {}
            self.checkpoints = []

        def start(self):
            """Start performance monitoring."""
            gc.collect()  # Ensure clean baseline
            self.start_time = time.perf_counter()

            if psutil is not None:
                process = psutil.Process()
                self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
                self.start_cpu = process.cpu_percent()
            else:
                self.start_memory = 0
                self.start_cpu = 0

        def checkpoint(self, name: str):
            """Record a checkpoint during test execution."""
            current_time = time.perf_counter()
            checkpoint = {
                "name": name,
                "elapsed_seconds": current_time - self.start_time
                if self.start_time
                else 0,
                "timestamp": current_time,
            }

            if psutil is not None:
                process = psutil.Process()
                checkpoint.update(
                    {
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "cpu_percent": process.cpu_percent(),
                    }
                )

            self.checkpoints.append(checkpoint)
            return checkpoint

        def stop(self):
            """Stop monitoring and return metrics."""
            if self.start_time is None:
                return {"error": "Monitoring not started"}

            end_time = time.perf_counter()

            self.metrics = {
                "duration_seconds": end_time - self.start_time,
                "checkpoints": self.checkpoints,
            }

            if psutil is not None:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024
                end_cpu = process.cpu_percent()

                self.metrics.update(
                    {
                        "memory_start_mb": self.start_memory,
                        "memory_end_mb": end_memory,
                        "memory_delta_mb": end_memory - self.start_memory,
                        "cpu_start_percent": self.start_cpu,
                        "cpu_end_percent": end_cpu,
                    }
                )

            return self.metrics

        def assert_performance(
            self, max_duration: float = None, max_memory_mb: float = None
        ):
            """Assert performance constraints."""
            if max_duration and self.metrics.get("duration_seconds", 0) > max_duration:
                raise AssertionError(
                    f"Test exceeded max duration: {self.metrics['duration_seconds']:.2f}s > {max_duration}s"
                )

            if max_memory_mb and self.metrics.get("memory_delta_mb", 0) > max_memory_mb:
                raise AssertionError(
                    f"Test exceeded max memory usage: {self.metrics['memory_delta_mb']:.2f}MB > {max_memory_mb}MB"
                )

    return PerformanceMonitor()


@pytest.fixture
def mock_browser_context():
    """Mock browser context for Playwright/Crawl4AI testing."""
    context = MagicMock()

    # Page mock
    page = MagicMock()
    page.goto = AsyncMock()
    page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    page.screenshot = AsyncMock(return_value=b"fake_screenshot_bytes")
    page.evaluate = AsyncMock(return_value={"test": "data"})
    page.wait_for_selector = AsyncMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.close = AsyncMock()

    # Browser mock
    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.new_page = AsyncMock(return_value=page)
    browser.close = AsyncMock()

    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()

    return {"browser": browser, "context": context, "page": page}


@pytest.fixture(scope="session")
def ci_environment_config(app_config):
    """Enhanced CI environment detection and configuration.

    Session-scoped fixture providing CI environment detection
    and  configuration for different CI platforms.
    """
    is_ci = is_ci_environment()

    ci_config = {
        "is_ci": is_ci,
        "is_github_actions": bool(os.getenv("GITHUB_ACTIONS")),
        "is_gitlab_ci": bool(os.getenv("GITLAB_CI")),
        "is_jenkins": bool(os.getenv("JENKINS_URL")),
        "is_azure_devops": bool(os.getenv("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI")),
        "cpu_count": os.cpu_count() or 1,
        "parallel_workers": app_config["parallel_workers"],
        "worker_id": os.getenv("PYTEST_XDIST_WORKER", "master"),
        "platform": {
            "os": sys.platform,
            "is_windows": is_windows(),
            "is_macos": is_macos(),
            "is_linux": is_linux(),
        },
    }

    # Platform-specific CI optimizations
    if is_ci:
        if ci_config["is_github_actions"]:
            # GitHub Actions specific optimizations
            ci_config["max_workers"] = min(4, ci_config["cpu_count"])
            ci_config["timeout_multiplier"] = 1.5
        elif ci_config["is_gitlab_ci"]:
            # GitLab CI specific optimizations
            ci_config["max_workers"] = min(2, ci_config["cpu_count"])
            ci_config["timeout_multiplier"] = 2.0
        else:
            # Generic CI optimizations
            ci_config["max_workers"] = min(3, ci_config["cpu_count"])
            ci_config["timeout_multiplier"] = 1.2

        # Set CI-specific environment variables
        os.environ.setdefault("PYTEST_TIMEOUT", "300")
        os.environ.setdefault("PYTEST_MAXFAIL", "5")

    return ci_config


def pytest_configure(config):
    """Configure pytest with  settings for  testing."""
    # Core test markers
    markers = [
        "browser: mark test as requiring browser automation",
        "slow: mark test as slow running (>5s)",
        "fast: mark test as fast running (<1s)",
        "integration: mark test as integration test",
        "unit: mark test as unit test",
        "e2e: mark test as end-to-end test",
        "performance: mark test as performance test",
        "benchmark: mark test as benchmark test",
        # AI/ML markers
        "ai: mark test as AI/ML specific",
        "embedding: mark test as embedding-related",
        "vector_db: mark test as vector database related",
        "rag: mark test as RAG system related",
        # Infrastructure markers
        "database: mark test as requiring database",
        "network: mark test as requiring network access",
        "redis: mark test as requiring Redis",
        "async_test: mark test as async test",
        # Test framework markers
        "hypothesis: mark test as property-based test using Hypothesis",
        "property: mark test as property-based test",
        "respx: mark test as using respx HTTP mocking",
        # CI/Platform markers
        "ci_only: mark test to run only in CI",
        "local_only: mark test to run only locally",
        "parallel_safe: mark test as safe for parallel execution",
        "isolation_required: mark test as requiring complete isolation",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)

    # Configure parallel execution
    if config.getoption("--numprocesses", default="auto") == "auto":
        import os

        worker_count = min(4, max(1, (os.cpu_count() or 1) // 2))
        config.option.numprocesses = worker_count


def pytest_collection_modifyitems(config, items):
    """Modify test collection for  organization and parallel execution."""
    for item in items:
        # Auto-mark based on file path
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Auto-mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.async_test)

        # Auto-mark based on fixtures used
        if hasattr(item, "fixturenames"):
            fixture_names = item.fixturenames

            if "respx_mock" in fixture_names:
                item.add_marker(pytest.mark.respx)
            if any(name.startswith("mock_") for name in fixture_names):
                item.add_marker(pytest.mark.parallel_safe)
            if "isolated_db_session" in fixture_names:
                item.add_marker(pytest.mark.isolation_required)
            if "performance_monitor" in fixture_names:
                item.add_marker(pytest.mark.performance)


def pytest_runtest_setup(item):
    """Setup for individual test runs with environment checks."""
    # Skip tests based on environment
    if item.get_closest_marker("ci_only") and not is_ci_environment():
        pytest.skip("Test only runs in CI environment")

    if item.get_closest_marker("local_only") and is_ci_environment():
        pytest.skip("Test only runs in local environment")

    # Check for required external services
    if item.get_closest_marker("network"):
        # Could add network connectivity check here
        pass

    if item.get_closest_marker("database"):
        # Could add database connectivity check here
        pass


def pytest_runtest_call(item):
    """Hook for test execution monitoring."""
    # Could add test execution monitoring here


def pytest_sessionstart(session):
    """Session start hook for global setup."""
    # Print test session info
    worker_id = os.getenv("PYTEST_XDIST_WORKER")
    if worker_id:
        print(f"\nStarting test session on worker: {worker_id}")
    else:
        print(
            f"\nStarting test session with {session.config.option.numprocesses} workers"
        )


def pytest_sessionfinish(session, exitstatus):
    """Session finish hook for cleanup and reporting."""
    # Could add session-level cleanup or reporting here


# Import all fixtures from fixture modules
pytest_plugins = [
    "tests.fixtures.mock_factories",
    "tests.fixtures.async_isolation",
    "tests.fixtures.parallel_config",
    "tests.fixtures.async_fixtures",
    "tests.fixtures.external_services",
    "tests.fixtures.test_data",
]
