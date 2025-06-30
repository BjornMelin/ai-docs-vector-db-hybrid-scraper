"""Enhanced pytest configuration and fixtures for production-grade testing.

This module provides the enhanced testing infrastructure with:
- Property-based testing with Hypothesis
- Performance benchmarking fixtures
- Advanced mocking patterns with respx
- Integration test service isolation
- AI/ML testing utilities
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest
import respx
from hypothesis import HealthCheck, settings, strategies as st


# Configure logging for tests
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ===== Enhanced HTTP Mocking with respx =====


@pytest.fixture
async def respx_mock():
    """Enhanced respx mock fixture with comprehensive HTTP mocking capabilities.

    This fixture provides a properly configured respx mock instance that:
    - Handles async operations correctly
    - Provides realistic response mocking
    - Supports trio/asyncio compatibility
    - Includes error injection capabilities
    """
    async with respx.mock(
        base_url="https://api.openai.com",
        assert_all_called=False,  # Allow uncalled mocks for flexibility
        assert_all_mocked=True,  # Ensure all HTTP calls are mocked
    ) as mock:
        # Setup default OpenAI embedding mock
        mock.post("/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1] * 1536}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        yield mock


@pytest.fixture
def mock_qdrant_service():
    """Enhanced Qdrant service mock with comprehensive vector operations."""
    mock = MagicMock()

    # Collection operations
    mock.create_collection = AsyncMock()
    mock.delete_collection = AsyncMock()
    mock.get_collections = AsyncMock(return_value={"collections": []})
    mock.collection_exists = AsyncMock(return_value=True)

    # Point operations
    mock.upsert_points = AsyncMock()
    mock.search_points = AsyncMock(
        return_value={
            "result": [
                {
                    "id": "test_id",
                    "score": 0.95,
                    "payload": {
                        "content": "Test document content",
                        "title": "Test Document",
                        "url": "https://example.com",
                    },
                    "vector": [0.1] * 1536,
                }
            ]
        }
    )
    mock.count_points = AsyncMock(return_value={"count": 100})

    # Health and info
    mock.health_check = AsyncMock(return_value={"status": "ok"})
    mock.get_collection_info = AsyncMock(
        return_value={
            "status": "ready",
            "vectors_count": 100,
            "indexed_vectors_count": 100,
        }
    )

    return mock


@pytest.fixture
def mock_embedding_service():
    """Enhanced embedding service mock with realistic behavior."""
    mock = MagicMock()

    def generate_mock_embedding(text: str) -> list[float]:
        """Generate deterministic mock embedding based on text."""
        # Use text hash for deterministic results
        seed = hash(text) % 1000000
        import random

        random.seed(seed)
        embedding = [random.uniform(-1, 1) for _ in range(1536)]
        # Normalize to unit vector
        norm = sum(x**2 for x in embedding) ** 0.5
        return [x / norm for x in embedding] if norm > 0 else embedding

    mock.generate_embedding = AsyncMock(side_effect=generate_mock_embedding)
    mock.generate_embeddings_batch = AsyncMock(
        side_effect=lambda texts: [generate_mock_embedding(text) for text in texts]
    )
    mock.get_embedding_dimensions = Mock(return_value=1536)

    return mock


# ===== Property-Based Testing with Hypothesis =====


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


# ===== Performance Testing Infrastructure =====


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture for latency and throughput testing."""

    class PerformanceMonitor:
        def __init__(self):
            self.measurements = []

        async def measure_async_operation(self, operation, *args, **kwargs):
            """Measure the performance of an async operation."""
            start_time = time.perf_counter()
            try:
                result = await operation(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
            finally:
                end_time = time.perf_counter()

            measurement = {
                "duration_ms": (end_time - start_time) * 1000,
                "success": success,
                "error": error,
                "timestamp": start_time,
            }
            self.measurements.append(measurement)
            return result, measurement

        def get_statistics(self) -> dict[str, float]:
            """Get performance statistics from measurements."""
            if not self.measurements:
                return {}

            durations = [m["duration_ms"] for m in self.measurements if m["success"]]
            if not durations:
                return {"success_rate": 0.0}

            import statistics

            return {
                "count": len(self.measurements),
                "success_rate": sum(1 for m in self.measurements if m["success"])
                / len(self.measurements),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "mean_duration_ms": statistics.mean(durations),
                "median_duration_ms": statistics.median(durations),
                "p95_duration_ms": statistics.quantiles(durations, n=20)[18]
                if len(durations) >= 20
                else max(durations),
                "p99_duration_ms": statistics.quantiles(durations, n=100)[98]
                if len(durations) >= 100
                else max(durations),
            }

        def assert_performance_requirements(
            self, max_p95_ms: float = 100.0, min_success_rate: float = 0.95
        ):
            """Assert performance requirements are met."""
            stats = self.get_statistics()

            assert stats.get("success_rate", 0) >= min_success_rate, (
                f"Success rate {stats.get('success_rate', 0):.3f} below {min_success_rate}"
            )

            assert stats.get("p95_duration_ms", float("inf")) <= max_p95_ms, (
                f"P95 latency {stats.get('p95_duration_ms', 0):.1f}ms exceeds {max_p95_ms}ms"
            )

    return PerformanceMonitor()


# ===== Integration Test Service Isolation =====


@pytest.fixture
async def isolated_test_environment():
    """Create isolated test environment with mock services."""

    class IsolatedEnvironment:
        def __init__(self):
            self.temp_dir = tempfile.mkdtemp(prefix="ai_docs_test_")
            self.services = {}
            self._cleanup_tasks = []

        async def start_service(self, service_name: str, service_mock):
            """Start a mock service."""
            self.services[service_name] = service_mock

        async def stop_service(self, service_name: str):
            """Stop a mock service."""
            self.services.pop(service_name, None)

        def get_service(self, service_name: str):
            """Get a running service."""
            return self.services.get(service_name)

        async def cleanup(self):
            """Clean up test environment."""
            import contextlib
            import shutil

            with contextlib.suppress(Exception):
                shutil.rmtree(self.temp_dir, ignore_errors=True)

            for task in self._cleanup_tasks:
                try:
                    if not task.done():
                        task.cancel()
                        await asyncio.sleep(0.1)
                except Exception:
                    pass

    env = IsolatedEnvironment()

    try:
        yield env
    finally:
        await env.cleanup()


# ===== AI/ML Testing Utilities =====


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

            import math

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
            import random

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


# ===== Test Data Factories =====


@pytest.fixture
def test_data_factory():
    """Factory for generating test data with realistic patterns."""

    class TestDataFactory:
        @staticmethod
        def create_mock_documents(count: int = 10) -> list[dict[str, Any]]:
            """Create mock documents for testing."""
            return [
                {
                    "id": f"doc_{i}",
                    "title": f"Test Document {i}",
                    "content": f"This is test content for document {i}. " * (i + 2),
                    "url": f"https://example.com/docs/doc_{i}",
                    "metadata": {
                        "category": f"category_{i % 3}",
                        "tags": [f"tag_{i}", f"tag_{i % 2}"],
                        "timestamp": "2025-06-28T00:00:00Z",
                        "word_count": (i + 2) * 10,
                    },
                }
                for i in range(count)
            ]

        @staticmethod
        def create_search_queries(count: int = 5) -> list[str]:
            """Create realistic search queries."""
            queries = [
                "How to implement authentication",
                "Vector database performance optimization",
                "Best practices for API design",
                "Machine learning model deployment",
                "Database connection pooling strategies",
            ]
            return queries[:count]

        @staticmethod
        def create_mock_crawl_results(count: int = 5) -> list[dict[str, Any]]:
            """Create mock crawl results."""
            return [
                {
                    "url": f"https://example.com/page_{i}",
                    "title": f"Page {i}",
                    "content": f"Content for page {i} with relevant information.",
                    "markdown": f"# Page {i}\n\nContent for page {i}.",
                    "success": True,
                    "status_code": 200,
                    "metadata": {
                        "description": f"Description for page {i}",
                        "keywords": [f"keyword_{i}", "test"],
                        "content_type": "text/html",
                    },
                    "links": [f"https://example.com/page_{i + 1}"]
                    if i < count - 1
                    else [],
                }
                for i in range(count)
            ]

    return TestDataFactory()


# ===== Enhanced Pytest Configuration =====


def pytest_configure(config):
    """Enhanced pytest configuration with comprehensive marker support."""

    # Core test markers
    markers = [
        "fast: Fast unit tests (<100ms each)",
        "slow: Slow tests (>5 seconds)",
        "integration: Integration tests (<5s each)",
        "e2e: End-to-end tests (full pipeline)",
        "unit: Unit tests",
        "performance: Performance and benchmark tests",
        "benchmark: Benchmark tests",
        # AI/ML markers
        "ai: AI/ML specific tests",
        "embedding: Embedding-related tests",
        "vector_db: Vector database tests",
        "rag: RAG system tests",
        "property: Property-based tests using Hypothesis",
        "hypothesis: Property-based tests using Hypothesis",
        # Infrastructure markers
        "browser: Browser automation tests",
        "network: Tests requiring network access",
        "database: Tests requiring database connection",
        "asyncio: Async tests",
        # Quality markers
        "security: Security tests",
        "accessibility: Accessibility tests",
        "contract: Contract tests",
        "chaos: Chaos engineering tests",
        "load: Load tests",
        "stress: Stress tests",
        # Platform markers
        "windows: Windows-only tests",
        "macos: macOS-only tests",
        "linux: Linux-only tests",
        "ci_only: CI-only tests",
        "local_only: Local-only tests",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


# ===== Hypothesis Configuration =====

# Configure Hypothesis for better property-based testing
settings.register_profile(
    "test",
    max_examples=20,  # Reduced for faster tests
    deadline=5000,  # 5 second deadline
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.large_base_example,
    ],
)

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=10000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.large_base_example,
    ],
)

# Use appropriate profile
if os.getenv("CI"):
    settings.load_profile("ci")
else:
    settings.load_profile("test")
