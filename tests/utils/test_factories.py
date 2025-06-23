import typing

"""Standardized test data factories following 2025 patterns.

This module provides factory classes and functions for generating consistent
test data across all test categories. Uses modern patterns like dataclasses,
type hints, and builder patterns.
"""

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class DocumentFactory:
    """Factory for creating test document data with realistic defaults."""

    @staticmethod
    def create_document(
        url: str = "https://example.com/doc",
        title: str = "Test Document",
        content: str = "This is test content for the document.",
        metadata: typing.Optional[dict[str, Any]] = None,
        doc_id: typing.Optional[str] = None,
        timestamp: typing.Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a test document with specified or default values.

        Args:
            url: Document URL
            title: Document title
            content: Document content
            metadata: Additional metadata
            doc_id: Document ID (generates UUID if not provided)
            timestamp: Document timestamp (uses current time if not provided)

        Returns:
            Dictionary representing a document
        """
        return {
            "id": doc_id or str(uuid4()),
            "url": url,
            "title": title,
            "content": content,
            "metadata": metadata
            or {"language": "en", "content_type": "documentation", "source": "test"},
            "timestamp": timestamp or datetime.utcnow().isoformat() + "Z",
            "processed": True,
            "error": None,
        }

    @staticmethod
    def create_batch(
        count: int = 5,
        url_template: str = "https://example.com/doc{i}",
        title_template: str = "Test Document {i}",
        content_template: str = "This is test content for document {i}.",
    ) -> list[dict[str, Any]]:
        """Create a batch of test documents.

        Args:
            count: Number of documents to create
            url_template: URL template with {i} placeholder
            title_template: Title template with {i} placeholder
            content_template: Content template with {i} placeholder

        Returns:
            List of document dictionaries
        """
        return [
            DocumentFactory.create_document(
                url=url_template.format(i=i),
                title=title_template.format(i=i),
                content=content_template.format(i=i),
            )
            for i in range(count)
        ]


@dataclass
class VectorFactory:
    """Factory for creating test vector data and embeddings."""

    @staticmethod
    def create_vector(
        dimension: int = 1536,
        value_range: tuple[float, float] = (-1.0, 1.0),
        normalize: bool = False,
    ) -> list[float]:
        """Create a test vector with specified characteristics.

        Args:
            dimension: Vector dimension
            value_range: Range for random values
            normalize: Whether to normalize the vector

        Returns:
            List of float values representing the vector
        """
        vector = [
            random.uniform(value_range[0], value_range[1]) for _ in range(dimension)
        ]

        if normalize:
            magnitude = sum(x * x for x in vector) ** 0.5
            if magnitude > 0:
                vector = [x / magnitude for x in vector]

        return vector

    @staticmethod
    def create_point(
        point_id: int | str,
        vector_dim: int = 1536,
        payload: typing.Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a vector point for testing.

        Args:
            point_id: Unique identifier for the point
            vector_dim: Vector dimension
            payload: Additional payload data

        Returns:
            Dictionary representing a vector point
        """
        return {
            "id": point_id,
            "vector": VectorFactory.create_vector(vector_dim),
            "payload": payload
            or {
                "url": f"https://example.com/doc{point_id}",
                "title": f"Document {point_id}",
                "content": f"Content for document {point_id}",
                "chunk_index": 0,
            },
        }

    @staticmethod
    def create_search_result(
        point_id: int | str,
        score: float = 0.95,
        vector_dim: int = 1536,
        payload: typing.Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a search result for testing.

        Args:
            point_id: Point identifier
            score: Search score (0.0 to 1.0)
            vector_dim: Vector dimension
            payload: Search result payload

        Returns:
            Dictionary representing a search result
        """
        return {
            "id": point_id,
            "score": max(0.0, min(1.0, score)),  # Clamp to valid range
            "vector": VectorFactory.create_vector(vector_dim),
            "payload": payload
            or {
                "url": f"https://example.com/doc{point_id}",
                "title": f"Search Result {point_id}",
                "content": f"Relevant content for search result {point_id}",
                "snippet": f"...relevant snippet for {point_id}...",
            },
        }


@dataclass
class ResponseFactory:
    """Factory for creating standardized API response structures."""

    @staticmethod
    def create_success_response(
        data: Any = None,
        message: str = "Operation completed successfully",
        metadata: typing.Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a successful API response.

        Args:
            data: Response data
            message: Success message
            metadata: Additional metadata

        Returns:
            Standardized success response dictionary
        """
        return {
            "success": True,
            "data": data,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": str(uuid4()),
        }

    @staticmethod
    def create_error_response(
        error_code: str = "GENERAL_ERROR",
        message: str = "An error occurred",
        details: typing.Optional[dict[str, Any]] = None,
        status_code: int = 400,
    ) -> dict[str, Any]:
        """Create an error API response.

        Args:
            error_code: Specific error code
            message: Error message
            details: Additional error details
            status_code: HTTP status code

        Returns:
            Standardized error response dictionary
        """
        return {
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
                "status_code": status_code,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": str(uuid4()),
        }

    @staticmethod
    def create_paginated_response(
        items: list[Any],
        page: int = 1,
        per_page: int = 10,
        total: typing.Optional[int] = None,
    ) -> dict[str, Any]:
        """Create a paginated API response.

        Args:
            items: List of items for current page
            page: Current page number
            per_page: Items per page
            total: Total item count (calculated if not provided)

        Returns:
            Standardized paginated response dictionary
        """
        if total is None:
            total = len(items)

        return ResponseFactory.create_success_response(
            data={
                "items": items,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "pages": (total + per_page - 1) // per_page,
                    "has_next": page * per_page < total,
                    "has_prev": page > 1,
                },
            }
        )


@dataclass
class ChunkFactory:
    """Factory for creating document chunk test data."""

    @staticmethod
    def create_chunk(
        content: str = "This is a test chunk of content.",
        title: str = "Test Document",
        url: str = "https://example.com/doc",
        chunk_index: int = 0,
        total_chunks: int = 1,
        start_pos: int = 0,
        end_pos: typing.Optional[int] = None,
        metadata: typing.Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a document chunk for testing.

        Args:
            content: Chunk content
            title: Source document title
            url: Source document URL
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            start_pos: Starting character position
            end_pos: Ending character position
            metadata: Additional metadata

        Returns:
            Dictionary representing a document chunk
        """
        if end_pos is None:
            end_pos = start_pos + len(content)

        return {
            "content": content,
            "title": title if chunk_index == 0 else f"{title} (Part {chunk_index + 1})",
            "url": url,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "char_count": len(content),
            "token_estimate": len(content) // 4,  # Rough token estimate
            "chunk_type": "text",
            "language": "en",
            "has_code": "```" in content or "def " in content,
            "metadata": metadata or {},
        }

    @staticmethod
    def create_code_chunk(
        code_content: str = "def test_function():\n    return True",
        language: str = "python",
        function_name: str = "test_function",
        **kwargs,
    ) -> dict[str, Any]:
        """Create a code chunk for testing.

        Args:
            code_content: Code content
            language: Programming language
            function_name: Function name (if applicable)
            **kwargs: Additional arguments for create_chunk

        Returns:
            Dictionary representing a code chunk
        """
        chunk = ChunkFactory.create_chunk(content=code_content, **kwargs)
        chunk.update(
            {
                "chunk_type": "code",
                "language": language,
                "has_code": True,
                "metadata": {
                    "is_function": True,
                    "function_name": function_name,
                    "language": language,
                    **chunk.get("metadata", {}),
                },
            }
        )
        return chunk


@dataclass
class TestDataBuilder:
    """Builder pattern for complex test data structures."""

    def __init__(self):
        """Initialize empty test data builder."""
        self._data: dict[str, Any] = {}

    def with_id(self, doc_id: str) -> "TestDataBuilder":
        """Add ID to test data."""
        self._data["id"] = doc_id
        return self

    def with_url(self, url: str) -> "TestDataBuilder":
        """Add URL to test data."""
        self._data["url"] = url
        return self

    def with_title(self, title: str) -> "TestDataBuilder":
        """Add title to test data."""
        self._data["title"] = title
        return self

    def with_content(self, content: str) -> "TestDataBuilder":
        """Add content to test data."""
        self._data["content"] = content
        return self

    def with_metadata(self, metadata: dict[str, Any]) -> "TestDataBuilder":
        """Add metadata to test data."""
        self._data["metadata"] = metadata
        return self

    def with_timestamp(self, timestamp: str) -> "TestDataBuilder":
        """Add timestamp to test data."""
        self._data["timestamp"] = timestamp
        return self

    def with_error(self, error: typing.Optional[str] = None) -> "TestDataBuilder":
        """Add error to test data."""
        self._data["error"] = error
        return self

    def with_status(self, status: str) -> "TestDataBuilder":
        """Add status to test data."""
        self._data["status"] = status
        return self

    def with_vector(self, vector: list[float]) -> "TestDataBuilder":
        """Add vector to test data."""
        self._data["vector"] = vector
        return self

    def with_score(self, score: float) -> "TestDataBuilder":
        """Add search score to test data."""
        self._data["score"] = score
        return self

    def build(self) -> dict[str, Any]:
        """Build the final test data structure.

        Returns:
            Copy of the constructed data dictionary
        """
        return self._data.copy()


class ConfigFactory:
    """Factory for creating test configuration objects."""

    @staticmethod
    def create_database_config(
        url: str = "sqlite+aiosqlite:///:memory:",
        pool_size: int = 5,
        echo: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Create database configuration for testing.

        Args:
            url: Database URL
            pool_size: Connection pool size
            echo: Whether to echo SQL queries
            **kwargs: Additional configuration parameters

        Returns:
            Database configuration dictionary
        """
        config = {
            "url": url,
            "pool_size": pool_size,
            "echo": echo,
            "timeout": 30.0,
            "retry_attempts": 3,
            "isolation_level": None,
            **kwargs,
        }
        return config

    @staticmethod
    def create_vector_db_config(
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "test_collection",
        vector_dim: int = 1536,
        **kwargs,
    ) -> dict[str, Any]:
        """Create vector database configuration for testing.

        Args:
            host: Vector DB host
            port: Vector DB port
            collection_name: Collection name
            vector_dim: Vector dimension
            **kwargs: Additional configuration parameters

        Returns:
            Vector database configuration dictionary
        """
        config = {
            "host": host,
            "port": port,
            "collection_name": collection_name,
            "vector_dim": vector_dim,
            "distance_metric": "cosine",
            "timeout": 60.0,
            **kwargs,
        }
        return config

    @staticmethod
    def create_api_config(
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs,
    ) -> dict[str, Any]:
        """Create API configuration for testing.

        Args:
            base_url: API base URL
            timeout: Request timeout
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration parameters

        Returns:
            API configuration dictionary
        """
        config = {
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "retry_delay": 1.0,
            "headers": {
                "Content-Type": "application/json",
                "User-Agent": "test-client/1.0",
            },
            **kwargs,
        }
        return config


class PerformanceDataFactory:
    """Factory for creating performance test data and metrics."""

    @staticmethod
    def create_metrics(
        operation_name: str = "test_operation",
        execution_time: float = 0.1,
        memory_usage_mb: float = 10.0,
        cpu_usage_percent: float = 15.0,
        success: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Create performance metrics for testing.

        Args:
            operation_name: Name of the operation
            execution_time: Execution time in seconds
            memory_usage_mb: Memory usage in MB
            cpu_usage_percent: CPU usage percentage
            success: Whether operation succeeded
            **kwargs: Additional metrics

        Returns:
            Performance metrics dictionary
        """
        return {
            "operation": operation_name,
            "execution_time": execution_time,
            "memory_usage_mb": memory_usage_mb,
            "cpu_usage_percent": cpu_usage_percent,
            "success": success,
            "timestamp": time.time(),
            "thread_count": 1,
            "error_count": 0 if success else 1,
            **kwargs,
        }

    @staticmethod
    def create_load_test_data(
        concurrent_users: int = 10,
        requests_per_second: float = 5.0,
        duration_seconds: float = 60.0,
        success_rate: float = 0.98,
        avg_response_time: float = 0.2,
    ) -> dict[str, Any]:
        """Create load test data for testing.

        Args:
            concurrent_users: Number of concurrent users
            requests_per_second: Request rate
            duration_seconds: Test duration
            success_rate: Success rate (0.0 to 1.0)
            avg_response_time: Average response time in seconds

        Returns:
            Load test data dictionary
        """
        total_requests = int(requests_per_second * duration_seconds)
        successful_requests = int(total_requests * success_rate)

        return {
            "concurrent_users": concurrent_users,
            "requests_per_second": requests_per_second,
            "duration_seconds": duration_seconds,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "p95_response_time": avg_response_time * 1.5,
            "p99_response_time": avg_response_time * 2.0,
            "throughput": successful_requests / duration_seconds,
            "start_time": time.time(),
            "end_time": time.time() + duration_seconds,
        }


# Convenience functions for quick test data generation


def quick_document(title: str = "Quick Test Doc") -> dict[str, Any]:
    """Quickly create a test document with minimal setup."""
    return DocumentFactory.create_document(title=title)


def quick_vector_point(point_id: int | str = 1) -> dict[str, Any]:
    """Quickly create a vector point with minimal setup."""
    return VectorFactory.create_point(point_id)


def quick_success_response(data: Any = None) -> dict[str, Any]:
    """Quickly create a success response with minimal setup."""
    return ResponseFactory.create_success_response(data)


def quick_error_response(message: str = "Test error") -> dict[str, Any]:
    """Quickly create an error response with minimal setup."""
    return ResponseFactory.create_error_response(message=message)


def quick_chunk(content: str = "Test chunk content") -> dict[str, Any]:
    """Quickly create a document chunk with minimal setup."""
    return ChunkFactory.create_chunk(content=content)
