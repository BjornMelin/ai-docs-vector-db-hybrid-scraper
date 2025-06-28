"""AI/ML Testing Utilities for Modern Testing Patterns.

This module provides specialized utilities for testing AI/ML systems including
embeddings, vector databases, RAG systems, and other AI components following
2025 best practices.
"""

import asyncio
import gc
import math
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from hypothesis import strategies as st


class EmbeddingTestUtils:
    """Utilities for testing embedding-related functionality."""

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between -1 and 1

        Raises:
            ValueError: If vectors have different dimensions or are empty
        """
        if len(vec1) != len(vec2):
            raise ValueError(
                f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}"
            )

        if len(vec1) == 0:
            raise ValueError("Cannot compute similarity for empty vectors")

        arr1, arr2 = np.array(vec1), np.array(vec2)

        # Handle zero vectors
        norm1, norm2 = np.linalg.norm(arr1), np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(arr1, arr2) / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Euclidean distance (always non-negative)
        """
        if len(vec1) != len(vec2):
            raise ValueError(
                f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}"
            )

        return float(np.linalg.norm(np.array(vec1) - np.array(vec2)))

    @staticmethod
    def generate_test_embeddings(
        count: int, dim: int = 1536, normalized: bool = True, seed: int | None = None
    ) -> List[List[float]]:
        """Generate test embeddings with consistent properties.

        Args:
            count: Number of embeddings to generate
            dim: Embedding dimension
            normalized: Whether to normalize to unit vectors
            seed: Random seed for reproducibility

        Returns:
            List of embedding vectors
        """
        if seed is not None:
            np.random.seed(seed)

        embeddings = np.random.random((count, dim)).astype(np.float32)

        if normalized:
            # Normalize to unit vectors
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return [emb.tolist() for emb in embeddings]

    @staticmethod
    def generate_similar_embeddings(
        base_embedding: List[float],
        count: int,
        similarity_range: Tuple[float, float] = (0.8, 0.95),
        seed: int | None = None,
    ) -> List[List[float]]:
        """Generate embeddings similar to a base embedding.

        Args:
            base_embedding: Base embedding to generate similar ones from
            count: Number of similar embeddings to generate
            similarity_range: Range of cosine similarity to target
            seed: Random seed for reproducibility

        Returns:
            List of similar embedding vectors
        """
        if seed is not None:
            np.random.seed(seed)

        base_arr = np.array(base_embedding)
        dim = len(base_embedding)
        similar_embeddings = []

        for _ in range(count):
            # Generate random noise
            noise = np.random.normal(0, 0.1, dim)

            # Create similar vector
            similar = base_arr + noise
            similar = similar / np.linalg.norm(similar)  # Normalize

            # Adjust similarity to be within range
            current_sim = EmbeddingTestUtils.cosine_similarity(
                base_embedding, similar.tolist()
            )
            target_sim = np.random.uniform(*similarity_range)

            # Linear interpolation to achieve target similarity
            alpha = math.acos(target_sim) / math.acos(current_sim)
            if not math.isnan(alpha) and alpha > 0:
                adjusted = alpha * similar + (1 - alpha) * base_arr
                adjusted = adjusted / np.linalg.norm(adjusted)
                similar_embeddings.append(adjusted.tolist())
            else:
                similar_embeddings.append(similar.tolist())

        return similar_embeddings

    @staticmethod
    def validate_embedding_properties(embedding: List[float]) -> Dict[str, bool]:
        """Validate embedding meets quality requirements.

        Args:
            embedding: Embedding vector to validate

        Returns:
            Dictionary of validation results
        """
        arr = np.array(embedding)

        validations = {
            "is_finite": np.isfinite(arr).all(),
            "non_zero": not np.allclose(arr, 0),
            "reasonable_magnitude": 0.1 <= np.linalg.norm(arr) <= 10.0,
            "correct_dimension": len(embedding) in [384, 512, 768, 1024, 1536, 3072],
            "no_extreme_values": np.abs(arr).max() <= 100.0,
        }

        validations["is_valid"] = all(validations.values())
        return validations

    @staticmethod
    def batch_validate_embeddings(embeddings: List[List[float]]) -> Dict[str, Any]:
        """Validate a batch of embeddings for consistency.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Batch validation results
        """
        if not embeddings:
            return {"error": "Empty embedding list"}

        dimensions = [len(emb) for emb in embeddings]

        results = {
            "count": len(embeddings),
            "consistent_dimensions": len(set(dimensions)) == 1,
            "dimension": dimensions[0] if dimensions else 0,
            "all_valid": True,
            "invalid_count": 0,
            "similarity_stats": {},
        }

        # Validate individual embeddings
        invalid_count = 0
        for emb in embeddings:
            validation = EmbeddingTestUtils.validate_embedding_properties(emb)
            if not validation["is_valid"]:
                invalid_count += 1

        results["invalid_count"] = invalid_count
        results["all_valid"] = invalid_count == 0

        # Calculate similarity statistics for first 10 embeddings
        if len(embeddings) > 1:
            sample_embeddings = embeddings[: min(10, len(embeddings))]
            similarities = []

            for i in range(len(sample_embeddings)):
                for j in range(i + 1, len(sample_embeddings)):
                    try:
                        sim = EmbeddingTestUtils.cosine_similarity(
                            sample_embeddings[i], sample_embeddings[j]
                        )
                        similarities.append(sim)
                    except ValueError:
                        continue

            if similarities:
                results["similarity_stats"] = {
                    "mean": float(np.mean(similarities)),
                    "std": float(np.std(similarities)),
                    "min": float(np.min(similarities)),
                    "max": float(np.max(similarities)),
                }

        return results


class VectorDatabaseTestUtils:
    """Utilities for testing vector database operations."""

    @staticmethod
    def create_mock_qdrant_client() -> MagicMock:
        """Create a properly configured mock Qdrant client.

        Returns:
            Configured MagicMock for Qdrant client
        """
        client = MagicMock()

        # Configure async methods
        client.create_collection = AsyncMock()
        client.delete_collection = AsyncMock()
        client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
        client.upsert = AsyncMock()
        client.search = AsyncMock(return_value=[])
        client.count = AsyncMock(return_value=MagicMock(count=0))
        client.close = AsyncMock()
        client.get_collection = AsyncMock()

        # Configure search results
        def mock_search(*args, **kwargs):
            limit = kwargs.get("limit", 10)
            # Return mock search results
            results = []
            for i in range(min(limit, 5)):  # Return up to 5 results
                result = MagicMock()
                result.id = f"doc_{i}"
                result.score = 0.9 - (i * 0.1)  # Decreasing scores
                result.payload = {
                    "title": f"Document {i}",
                    "content": f"Content for document {i}",
                    "url": f"https://example.com/doc_{i}",
                }
                results.append(result)
            return results

        client.search.side_effect = mock_search
        return client

    @staticmethod
    def generate_test_points(
        count: int, vector_dim: int = 1536, collection_name: str = "test_collection"
    ) -> List[Dict[str, Any]]:
        """Generate test vector points for database operations.

        Args:
            count: Number of points to generate
            vector_dim: Vector dimension
            collection_name: Collection name for points

        Returns:
            List of point dictionaries
        """
        embeddings = EmbeddingTestUtils.generate_test_embeddings(count, vector_dim)

        points = []
        for i, embedding in enumerate(embeddings):
            point = {
                "id": f"point_{i}",
                "vector": embedding,
                "payload": {
                    "title": f"Test Document {i}",
                    "content": f"This is test content for document {i}",
                    "url": f"https://example.com/doc_{i}",
                    "chunk_index": i % 10,
                    "source": "test_data",
                    "timestamp": "2025-06-28T00:00:00Z",
                },
            }
            points.append(point)

        return points

    @staticmethod
    async def measure_search_performance(
        search_func, query_vectors: List[List[float]], concurrent_requests: int = 10
    ) -> Dict[str, float]:
        """Measure vector search performance metrics.

        Args:
            search_func: Async search function to test
            query_vectors: List of query vectors
            concurrent_requests: Number of concurrent requests

        Returns:
            Performance metrics dictionary
        """
        start_time = time.time()

        # Create concurrent search tasks
        tasks = []
        for i in range(concurrent_requests):
            query_vector = query_vectors[i % len(query_vectors)]
            task = search_func(query_vector=query_vector, limit=10)
            tasks.append(task)

        # Execute all searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_duration = end_time - start_time

        # Analyze results
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        error_count = len(results) - successful_requests

        return {
            "total_duration_seconds": total_duration,
            "requests_per_second": len(tasks) / total_duration,
            "average_latency_ms": (total_duration / len(tasks)) * 1000,
            "successful_requests": successful_requests,
            "error_count": error_count,
            "success_rate": successful_requests / len(tasks),
        }


class RAGTestUtils:
    """Utilities for testing RAG (Retrieval-Augmented Generation) systems."""

    @staticmethod
    def calculate_contextual_precision(
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
        relevance_threshold: float = 0.7,
    ) -> float:
        """Calculate contextual precision for RAG retrieval.

        Args:
            retrieved_contexts: List of retrieved context strings
            ground_truth_contexts: List of known relevant contexts
            relevance_threshold: Similarity threshold for relevance

        Returns:
            Contextual precision score (0.0 to 1.0)
        """
        if not retrieved_contexts:
            return 0.0

        relevant_count = 0

        for retrieved in retrieved_contexts:
            for ground_truth in ground_truth_contexts:
                # Simple string similarity for demonstration
                # In practice, you'd use semantic similarity
                similarity = len(
                    set(retrieved.lower().split()) & set(ground_truth.lower().split())
                ) / max(
                    len(set(retrieved.lower().split())),
                    len(set(ground_truth.lower().split())),
                )

                if similarity >= relevance_threshold:
                    relevant_count += 1
                    break

        return relevant_count / len(retrieved_contexts)

    @staticmethod
    def calculate_contextual_recall(
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
        relevance_threshold: float = 0.7,
    ) -> float:
        """Calculate contextual recall for RAG retrieval.

        Args:
            retrieved_contexts: List of retrieved context strings
            ground_truth_contexts: List of known relevant contexts
            relevance_threshold: Similarity threshold for relevance

        Returns:
            Contextual recall score (0.0 to 1.0)
        """
        if not ground_truth_contexts:
            return 1.0  # Perfect recall if no ground truth

        found_count = 0

        for ground_truth in ground_truth_contexts:
            for retrieved in retrieved_contexts:
                # Simple string similarity for demonstration
                similarity = len(
                    set(retrieved.lower().split()) & set(ground_truth.lower().split())
                ) / max(
                    len(set(retrieved.lower().split())),
                    len(set(ground_truth.lower().split())),
                )

                if similarity >= relevance_threshold:
                    found_count += 1
                    break

        return found_count / len(ground_truth_contexts)

    @staticmethod
    def evaluate_rag_response_quality(
        response: str, query: str, retrieved_contexts: List[str]
    ) -> Dict[str, float]:
        """Evaluate RAG response quality across multiple dimensions.

        Args:
            response: Generated response text
            query: Original query
            retrieved_contexts: Contexts used for generation

        Returns:
            Quality metrics dictionary
        """
        metrics = {}

        # Response completeness (simple heuristic)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        query_coverage = len(query_words & response_words) / len(query_words)
        metrics["query_coverage"] = query_coverage

        # Context utilization
        context_words = set()
        for context in retrieved_contexts:
            context_words.update(context.lower().split())

        if context_words:
            context_utilization = len(response_words & context_words) / len(
                context_words
            )
            metrics["context_utilization"] = min(context_utilization, 1.0)
        else:
            metrics["context_utilization"] = 0.0

        # Response length reasonableness
        response_length = len(response.split())
        metrics["reasonable_length"] = 1.0 if 10 <= response_length <= 500 else 0.5

        # Information density (ratio of unique words)
        if response_words:
            unique_ratio = len(set(response.lower().split())) / len(response.split())
            metrics["information_density"] = unique_ratio
        else:
            metrics["information_density"] = 0.0

        # Overall quality score (weighted average)
        weights = {
            "query_coverage": 0.3,
            "context_utilization": 0.3,
            "reasonable_length": 0.2,
            "information_density": 0.2,
        }

        overall_score = sum(metrics[key] * weights[key] for key in weights)
        metrics["overall_quality"] = overall_score

        return metrics


class PerformanceTestUtils:
    """Utilities for performance testing and monitoring."""

    def __init__(self):
        """Initialize performance monitoring."""
        self.memory_snapshots = []
        self.start_time = None
        self.monitoring_active = False

    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        tracemalloc.start()
        self.start_time = time.time()
        self.monitoring_active = True

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return performance metrics.

        Returns:
            Performance metrics dictionary
        """
        if not self.monitoring_active:
            return {"error": "Monitoring not active"}

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.monitoring_active = False

        return {
            "duration_seconds": end_time - self.start_time,
            "current_memory_mb": current / 1024 / 1024,
            "peak_memory_mb": peak / 1024 / 1024,
            "memory_snapshots": len(self.memory_snapshots),
        }

    def take_memory_snapshot(self, label: str = "") -> None:
        """Take a memory snapshot for later analysis.

        Args:
            label: Optional label for the snapshot
        """
        if not self.monitoring_active:
            return

        current, _ = tracemalloc.get_traced_memory()
        snapshot = {
            "timestamp": time.time() - self.start_time,
            "memory_mb": current / 1024 / 1024,
            "label": label,
        }
        self.memory_snapshots.append(snapshot)

    @staticmethod
    async def measure_async_function_performance(
        func, *args, **kwargs
    ) -> Tuple[Any, Dict[str, float]]:
        """Measure performance of an async function.

        Args:
            func: Async function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (function_result, performance_metrics)
        """
        # Start monitoring
        tracemalloc.start()
        start_time = time.time()
        memory_before, _ = tracemalloc.get_traced_memory()

        try:
            # Execute function
            result = await func(*args, **kwargs)

            # Measure after execution
            end_time = time.time()
            memory_after, peak_memory = tracemalloc.get_traced_memory()

            metrics = {
                "execution_time_seconds": end_time - start_time,
                "memory_used_mb": (memory_after - memory_before) / 1024 / 1024,
                "peak_memory_mb": peak_memory / 1024 / 1024,
            }

            return result, metrics

        finally:
            tracemalloc.stop()

    @staticmethod
    def benchmark_function_calls(
        func, test_cases: List[Tuple], iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark function calls with multiple test cases.

        Args:
            func: Function to benchmark
            test_cases: List of (args, kwargs) tuples
            iterations: Number of iterations per test case

        Returns:
            Benchmark statistics
        """
        all_times = []

        for args, kwargs in test_cases:
            for _ in range(iterations):
                start_time = time.time()
                func(*args, **kwargs)
                end_time = time.time()
                all_times.append(end_time - start_time)

        return {
            "mean_time_seconds": float(np.mean(all_times)),
            "median_time_seconds": float(np.median(all_times)),
            "std_time_seconds": float(np.std(all_times)),
            "min_time_seconds": float(np.min(all_times)),
            "max_time_seconds": float(np.max(all_times)),
            "total_calls": len(all_times),
        }


# Hypothesis strategies for AI/ML testing
class AITestStrategies:
    """Hypothesis strategies for AI/ML property-based testing."""

    @staticmethod
    def embeddings(min_dim: int = 128, max_dim: int = 1536, normalized: bool = True):
        """Generate embedding vectors for property-based testing.

        Args:
            min_dim: Minimum embedding dimension
            max_dim: Maximum embedding dimension
            normalized: Whether to generate normalized vectors

        Returns:
            Hypothesis strategy for embedding vectors
        """

        def generate_embedding(draw):
            dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
            # Generate random floats in reasonable range
            values = draw(
                st.lists(
                    st.floats(
                        min_value=-2.0,
                        max_value=2.0,
                        allow_nan=False,
                        allow_infinity=False,
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

            return values

        return st.deferred(generate_embedding)

    @staticmethod
    def document_chunks(
        min_length: int = 50,
        max_length: int = 2000,
        min_chunks: int = 1,
        max_chunks: int = 10,
    ):
        """Generate document chunks for testing.

        Args:
            min_length: Minimum chunk length
            max_length: Maximum chunk length
            min_chunks: Minimum number of chunks
            max_chunks: Maximum number of chunks

        Returns:
            Hypothesis strategy for document chunks
        """

        chunk_content = st.text(
            min_size=min_length,
            max_size=max_length,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
        )

        return st.lists(chunk_content, min_size=min_chunks, max_size=max_chunks)

    @staticmethod
    def search_queries():
        """Generate realistic search queries for testing.

        Returns:
            Hypothesis strategy for search queries
        """

        # Common query patterns
        simple_queries = st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
        )

        question_queries = st.builds(
            lambda text: f"What is {text}?", st.text(min_size=3, max_size=50)
        )

        how_queries = st.builds(
            lambda text: f"How to {text}?", st.text(min_size=3, max_size=50)
        )

        return st.one_of(simple_queries, question_queries, how_queries)


# Test decorators and markers
def ai_test(test_func):
    """Decorator for AI/ML specific tests."""
    return pytest.mark.ai(test_func)


def embedding_test(test_func):
    """Decorator for embedding-related tests."""
    return pytest.mark.embedding(test_func)


def vector_db_test(test_func):
    """Decorator for vector database tests."""
    return pytest.mark.vector_db(test_func)


def rag_test(test_func):
    """Decorator for RAG system tests."""
    return pytest.mark.rag(test_func)


def performance_critical(test_func):
    """Decorator for performance-critical tests."""
    return pytest.mark.performance_critical(test_func)
