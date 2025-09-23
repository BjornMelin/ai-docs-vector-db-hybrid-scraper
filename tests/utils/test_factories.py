"""Test data factories for consistent test data generation.

This module provides factory classes and functions for creating
consistent, realistic test data across different test scenarios.
"""

import random
import string
from dataclasses import dataclass
from typing import Any


@dataclass
class DocumentFactory:
    """Factory for creating test document data."""

    @staticmethod
    def create_document(
        title: str | None = None,
        content: str | None = None,
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a test document with realistic data."""
        return {
            "title": title or f"Test Document {random.randint(1, 1000)}",  # noqa: S311
            "content": content or DocumentFactory._generate_content(),
            "url": url or f"https://example.com/doc{random.randint(1, 100)}",  # noqa: S311
            "metadata": metadata
            or {
                "author": f"Author {random.randint(1, 50)}",  # noqa: S311
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["test", "sample"],
            },
        }

    @staticmethod
    def _generate_content() -> str:
        """Generate realistic document content."""
        paragraphs = [
            "This is a sample document used for testing purposes.",
            "It contains various types of content including technical information.",
            "The document demonstrates how the system processes different formats.",
            "Testing ensures that all components work together correctly.",
        ]
        return " ".join(paragraphs)


@dataclass
class ChunkFactory:
    """Factory for creating document chunk data."""

    @staticmethod
    def create_chunk(
        content: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        score: float | None = None,
    ) -> dict[str, Any]:
        """Create a test document chunk."""
        return {
            "content": content or "This is a sample document chunk for testing.",
            "title": title or f"Chunk Title {random.randint(1, 100)}",  # noqa: S311
            "metadata": metadata or {"chunk_id": random.randint(1, 1000)},  # noqa: S311
            "score": score or round(random.uniform(0.1, 1.0), 3),  # noqa: S311
        }


@dataclass
class VectorFactory:
    """Factory for creating test vector data."""

    @staticmethod
    def create_vector(dimensions: int = 1536) -> list[float]:
        """Create a test embedding vector."""
        # Generate normalized vector (-1 to 1 range)
        return [round(random.uniform(-1, 1), 4) for _ in range(dimensions)]  # noqa: S311

    @staticmethod
    def create_vectors(count: int, dimensions: int = 1536) -> list[list[float]]:
        """Create multiple test vectors."""
        return [VectorFactory.create_vector(dimensions) for _ in range(count)]


@dataclass
class ResponseFactory:
    """Factory for creating test API response data."""

    @staticmethod
    def create_success_response(data: Any = None) -> dict[str, Any]:
        """Create a successful API response."""
        return {
            "status": "success",
            "data": data or {"message": "Operation completed successfully"},
            "timestamp": "2024-01-01T00:00:00Z",
        }

    @staticmethod
    def create_error_response(
        error_code: str = "INTERNAL_ERROR",
        message: str = "An error occurred",
        status_code: int = 500,
    ) -> dict[str, Any]:
        """Create an error API response."""
        return {
            "status": "error",
            "error": {
                "code": error_code,
                "message": message,
            },
            "status_code": status_code,
            "timestamp": "2024-01-01T00:00:00Z",
        }


@dataclass
class TestDataBuilder:
    """Builder pattern for complex test data construction."""

    def __init__(self):
        self._data: dict[str, Any] = {}

    def with_id(self, doc_id: str) -> "TestDataBuilder":
        """Set document ID."""
        self._data["id"] = doc_id
        return self

    def with_url(self, url: str) -> "TestDataBuilder":
        """Set document URL."""
        self._data["url"] = url
        return self

    def with_title(self, title: str) -> "TestDataBuilder":
        """Set document title."""
        self._data["title"] = title
        return self

    def with_content(self, content: str) -> "TestDataBuilder":
        """Set document content."""
        self._data["content"] = content
        return self

    def with_metadata(self, metadata: dict[str, Any]) -> "TestDataBuilder":
        """Set document metadata."""
        self._data["metadata"] = metadata
        return self

    def with_status(self, status: str) -> "TestDataBuilder":
        """Set document status."""
        self._data["status"] = status
        return self

    def with_vectors(self, vectors: list[list[float]]) -> "TestDataBuilder":
        """Set document vectors."""
        self._data["vectors"] = vectors
        return self

    def with_chunks(self, chunks: list[dict[str, Any]]) -> "TestDataBuilder":
        """Set document chunks."""
        self._data["chunks"] = chunks
        return self

    def build(self) -> dict[str, Any]:
        """Build the final test data object."""
        return self._data.copy()


# Convenience functions for quick test data creation
def quick_success_response(data: Any = None) -> dict[str, Any]:
    """Quickly create a success response for testing."""
    return ResponseFactory.create_success_response(data)


def quick_error_response(
    error_code: str = "TEST_ERROR",
    message: str = "Test error occurred",
) -> dict[str, Any]:
    """Quickly create an error response for testing."""
    return ResponseFactory.create_error_response(error_code, message)


def quick_document(
    title: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    """Quickly create a test document."""
    return DocumentFactory.create_document(title=title, content=content)


def quick_chunk(content: str | None = None) -> dict[str, Any]:
    """Quickly create a test document chunk."""
    return ChunkFactory.create_chunk(content=content)


def quick_vector(dimensions: int = 1536) -> list[float]:
    """Quickly create a test vector."""
    return VectorFactory.create_vector(dimensions)


# Specialized factories for different test scenarios
@dataclass
class SearchTestFactory:
    """Factory for search-related test data."""

    @staticmethod
    def create_search_query(
        text: str = "test query",
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Create a search query object."""
        return {
            "query": text,
            "filters": filters or {},
            "limit": limit,
            "offset": 0,
        }

    @staticmethod
    def create_search_result(
        document: dict[str, Any],
        score: float = 0.8,
        highlights: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a search result object."""
        return {
            "document": document,
            "score": score,
            "highlights": highlights or [],
            "rank": random.randint(1, 100),  # noqa: S311
        }


@dataclass
class AITestFactory:
    """Factory for AI/ML related test data."""

    @staticmethod
    def create_embedding_request(
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> dict[str, Any]:
        """Create an embedding request."""
        return {
            "input": texts,
            "model": model,
            "encoding_format": "float",
        }

    @staticmethod
    def create_completion_request(
        prompt: str,
        model: str = "gpt-4",
        max_tokens: int = 100,
    ) -> dict[str, Any]:
        """Create a completion request."""
        return {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

    @staticmethod
    def create_rag_context(
        query: str,
        documents: list[dict[str, Any]],
        max_context_length: int = 4000,
    ) -> dict[str, Any]:
        """Create RAG context data."""
        return {
            "query": query,
            "documents": documents,
            "max_context_length": max_context_length,
            "strategy": "reciprocal_rank_fusion",
        }


# Random data generators for varied test data
class RandomDataGenerator:
    """Utility class for generating random test data."""

    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate a random string."""
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))  # noqa: S311

    @staticmethod
    def random_email() -> str:
        """Generate a random email address."""
        username = RandomDataGenerator.random_string(8)
        domain = random.choice(["example.com", "test.org", "sample.net"])  # noqa: S311
        return f"{username}@{domain}"

    @staticmethod
    def random_url() -> str:
        """Generate a random URL."""
        domain = random.choice(["example.com", "test.org", "sample.net"])  # noqa: S311
        path = f"/{RandomDataGenerator.random_string(5)}"
        return f"https://{domain}{path}"

    @staticmethod
    def random_tags(count: int = 3) -> list[str]:
        """Generate random tags."""
        return [RandomDataGenerator.random_string(6) for _ in range(count)]
