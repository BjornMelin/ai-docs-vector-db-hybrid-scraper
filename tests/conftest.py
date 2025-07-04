"""Pytest configuration and shared fixtures.

This module provides the core testing infrastructure with standardized fixtures,
configuration, and utilities that follow 2025 testing best practices.
"""

import math
import os
import random
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


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
    from qdrant_client.models import PointStruct
except ModuleNotFoundError:  # pragma: no cover - basic fallback
    from dataclasses import dataclass

    @dataclass
    class PointStruct:  # type: ignore[misc]
        """Simplified stand-in for qdrant_client.models.PointStruct."""

        id: int
        vector: list[float]
        payload: dict


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
def setup_browser_environment():
    """Set up browser automation environment for CI and local testing."""
    original_env = os.environ.copy()

    try:
        env_defaults = set_platform_environment_defaults()
        for key, value in env_defaults.items():
            if key not in os.environ:
                os.environ[key] = value

        if is_ci_environment():
            os.environ["CRAWL4AI_HEADLESS"] = "true"
            os.environ["CRAWL4AI_SKIP_BROWSER_DOWNLOAD"] = "false"
            os.environ["PLAYWRIGHT_CHROMIUM_SANDBOX"] = "false"

        project_root = Path(__file__).parent.parent
        test_dirs = [
            project_root / "tests" / "fixtures" / "cache",
            project_root / "tests" / "fixtures" / "data",
            project_root / "tests" / "fixtures" / "logs",
            project_root / "logs",
            project_root / "cache",
            project_root / "data",
        ]

        for test_dir in test_dirs:
            test_dir.mkdir(parents=True, exist_ok=True)

        yield

    finally:
        os.environ.clear()
        os.environ.update(original_env)


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


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client for testing."""
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
    )
    return client


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
            },
        )
    ]


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


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "browser: mark test as requiring browser automation"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "ai: AI/ML specific tests")
    config.addinivalue_line("markers", "embedding: Embedding-related tests")
    config.addinivalue_line("markers", "vector_db: Vector database tests")
    config.addinivalue_line("markers", "rag: RAG system tests")
    config.addinivalue_line("markers", "fast: Fast unit tests (<100ms each)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full pipeline)")
    config.addinivalue_line("markers", "asyncio: marks tests as async tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line(
        "markers", "property: Property-based tests using Hypothesis"
    )
    config.addinivalue_line(
        "markers", "hypothesis: Property-based tests using Hypothesis"
    )
