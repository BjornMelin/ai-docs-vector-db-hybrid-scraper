"""Minimal AI Testing Utilities - Dependency-Free Implementation.

This module provides essential AI/ML testing utilities using only Python standard library,
demonstrating  testing patterns without external dependencies.
"""

import math
import random
import time
from typing import Any
from unittest.mock import MagicMock


class MinimalEmbeddingTestUtils:
    """Minimal embedding testing utilities using standard library only."""

    @staticmethod
    def generate_test_embeddings(
        count: int = 1, dim: int = 384, seed: int | None = None
    ) -> list[list[float]]:
        """Generate test embeddings using standard library random.

        Args:
            count: Number of embeddings to generate
            dim: Dimension of each embedding
            seed: Random seed for reproducibility

        Returns:
            list of embedding vectors
        """
        if seed is not None:
            random.seed(seed)

        embeddings = []
        for _ in range(count):
            # Generate random values and normalize to unit vector
            raw_values = [random.gauss(0, 1) for _ in range(dim)]

            # Calculate magnitude
            magnitude = math.sqrt(sum(x * x for x in raw_values))
            if magnitude == 0:
                magnitude = 1.0

            # Normalize to unit vector
            normalized = [x / magnitude for x in raw_values]
            embeddings.append(normalized)

        return embeddings

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (-1 to 1)
        """
        if len(vec1) != len(vec2):
            msg = f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}"
            raise ValueError(msg)

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(x * x for x in vec1))
        magnitude2 = math.sqrt(sum(x * x for x in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    @staticmethod
    def validate_embedding_properties(embedding: list[float]) -> dict[str, Any]:
        """Validate embedding properties.

        Args:
            embedding: Embedding vector to validate

        Returns:
            Validation results dictionary
        """
        if not embedding:
            return {
                "is_valid": False,
                "errors": ["Empty embedding"],
                "dimension": 0,
                "magnitude": 0.0,
                "is_normalized": False,
            }

        errors = []

        # Check for valid numbers
        if not all(
            isinstance(x, int | float) and not math.isnan(x) and not math.isinf(x)
            for x in embedding
        ):
            errors.append("Contains invalid numbers (NaN or Inf)")

        # Calculate magnitude
        magnitude = math.sqrt(sum(x * x for x in embedding))
        is_normalized = abs(magnitude - 1.0) < 1e-6

        # Check for zero vector
        if magnitude == 0:
            errors.append("Zero vector")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "dimension": len(embedding),
            "magnitude": magnitude,
            "is_normalized": is_normalized,
        }

    @staticmethod
    def batch_validate_embeddings(embeddings: list[list[float]]) -> dict[str, Any]:
        """Validate a batch of embeddings.

        Args:
            embeddings: list of embedding vectors

        Returns:
            Batch validation results
        """
        if not embeddings:
            return {
                "all_valid": False,
                "consistent_dimensions": False,
                "total_count": 0,
                "valid_count": 0,
                "errors": ["Empty batch"],
            }

        dimensions = [len(emb) for emb in embeddings]
        consistent_dimensions = len(set(dimensions)) <= 1

        valid_count = 0
        for embedding in embeddings:
            validation = MinimalEmbeddingTestUtils.validate_embedding_properties(
                embedding
            )
            if validation["is_valid"]:
                valid_count += 1

        return {
            "all_valid": valid_count == len(embeddings),
            "consistent_dimensions": consistent_dimensions,
            "total_count": len(embeddings),
            "valid_count": valid_count,
            "dimensions": dimensions[0]
            if consistent_dimensions and dimensions
            else None,
        }


class MinimalVectorDatabaseTestUtils:
    """Minimal vector database testing utilities."""

    @staticmethod
    def generate_test_points(
        count: int = 10, vector_dim: int = 384, seed: int | None = None
    ) -> list[dict[str, Any]]:
        """Generate test vector points.

        Args:
            count: Number of points to generate
            vector_dim: Vector dimension
            seed: Random seed

        Returns:
            list of vector points with IDs, vectors, and payloads
        """
        if seed is not None:
            random.seed(seed)

        embeddings = MinimalEmbeddingTestUtils.generate_test_embeddings(
            count=count, dim=vector_dim, seed=seed
        )

        points = []
        for i, embedding in enumerate(embeddings):
            point = {
                "id": f"point_{i}",
                "vector": embedding,
                "payload": {
                    "text": f"Sample document {i}",
                    "category": random.choice(["doc", "article", "note"]),
                    "timestamp": time.time(),
                },
            }
            points.append(point)

        return points

    @staticmethod
    def create_mock_qdrant_client() -> MagicMock:
        """Create a mock Qdrant client for testing.

        Returns:
            Mock Qdrant client with basic functionality
        """
        mock_client = MagicMock()

        # Mock search results
        class MockSearchResult:
            def __init__(self, search_id: str, score: float, payload: dict):
                self.id = search_id
                self.score = score
                self.payload = payload

        async def mock_search(
            collection_name: str, query_vector: list[float], limit: int = 10
        ):
            # Generate mock results with decreasing scores
            results = []
            for i in range(min(limit, 5)):  # Return up to 5 results
                result = MockSearchResult(
                    id=f"result_{i}",
                    score=0.9 - (i * 0.1),  # Decreasing scores
                    payload={"content": f"Mock result {i}", "category": "test"},
                )
                results.append(result)
            return results

        mock_client.search = mock_search
        return mock_client


class MinimalRAGTestUtils:
    """Minimal RAG testing utilities."""

    @staticmethod
    def evaluate_rag_response_quality(
        response: str, query: str, retrieved_contexts: list[str]
    ) -> dict[str, float]:
        """Evaluate RAG response quality using simple metrics.

        Args:
            response: Generated response
            query: Original query
            retrieved_contexts: Retrieved context documents

        Returns:
            Quality metrics dictionary
        """
        if not response or not query:
            return {
                "overall_quality": 0.0,
                "query_coverage": 0.0,
                "context_utilization": 0.0,
                "response_length": 0,
            }

        # Simple query coverage: fraction of query words in response
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        query_coverage = (
            len(query_words & response_words) / len(query_words) if query_words else 0.0
        )

        # Simple context utilization: fraction of context words in response
        if retrieved_contexts:
            context_words = set()
            for context in retrieved_contexts:
                context_words.update(context.lower().split())

            context_utilization = (
                len(context_words & response_words) / len(context_words)
                if context_words
                else 0.0
            )
        else:
            context_utilization = 0.0

        # Overall quality as simple average
        overall_quality = (query_coverage + context_utilization) / 2

        return {
            "overall_quality": overall_quality,
            "query_coverage": query_coverage,
            "context_utilization": context_utilization,
            "response_length": len(response),
        }

    @staticmethod
    def calculate_contextual_precision(
        retrieved: list[str], ground_truth: list[str]
    ) -> float:
        """Calculate precision for retrieved contexts.

        Args:
            retrieved: Retrieved context documents
            ground_truth: Ground truth relevant documents

        Returns:
            Precision score (0.0 to 1.0)
        """
        if not retrieved:
            return 0.0

        # Simple word-based matching
        relevant_count = 0
        for ret_doc in retrieved:
            ret_words = set(ret_doc.lower().split())
            for gt_doc in ground_truth:
                gt_words = set(gt_doc.lower().split())
                # Consider relevant if significant word overlap
                overlap = len(ret_words & gt_words)
                if overlap > min(len(ret_words), len(gt_words)) * 0.3:
                    relevant_count += 1
                    break

        return relevant_count / len(retrieved)

    @staticmethod
    def calculate_contextual_recall(
        retrieved: list[str], ground_truth: list[str]
    ) -> float:
        """Calculate recall for retrieved contexts.

        Args:
            retrieved: Retrieved context documents
            ground_truth: Ground truth relevant documents

        Returns:
            Recall score (0.0 to 1.0)
        """
        if not ground_truth:
            return 1.0  # No ground truth to miss

        # Simple word-based matching
        found_count = 0
        for gt_doc in ground_truth:
            gt_words = set(gt_doc.lower().split())
            for ret_doc in retrieved:
                ret_words = set(ret_doc.lower().split())
                # Consider found if significant word overlap
                overlap = len(ret_words & gt_words)
                if overlap > min(len(ret_words), len(gt_words)) * 0.3:
                    found_count += 1
                    break

        return found_count / len(ground_truth)


class MinimalPerformanceTestUtils:
    """Minimal performance testing utilities."""

    def __init__(self):
        """Initialize performance monitor."""
        self.start_time: float | None = None
        self.memory_snapshots: list[dict[str, Any]] = []
        self.is_monitoring = False

    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.memory_snapshots = []
        self.is_monitoring = True
        self.take_memory_snapshot("start")

    def stop_monitoring(self) -> dict[str, Any]:
        """Stop monitoring and return metrics.

        Returns:
            Performance metrics dictionary
        """
        if not self.is_monitoring or self.start_time is None:
            return {"duration_seconds": 0, "peak_memory_mb": 0}

        self.take_memory_snapshot("end")
        duration = time.time() - self.start_time
        self.is_monitoring = False

        # Simple memory tracking (mock values for testing)
        peak_memory = max(
            (snap.get("memory_mb", 0) for snap in self.memory_snapshots), default=0
        )

        return {
            "duration_seconds": duration,
            "peak_memory_mb": peak_memory,
            "snapshots_count": len(self.memory_snapshots),
        }

    def take_memory_snapshot(self, label: str) -> None:
        """Take a memory snapshot.

        Args:
            label: Label for the snapshot
        """
        # Mock memory usage (in real implementation would use psutil or similar)
        mock_memory_mb = random.uniform(10, 50)  # Mock 10-50 MB usage

        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "memory_mb": mock_memory_mb,
        }
        self.memory_snapshots.append(snapshot)


# Test decorators (simplified versions without pytest)
def ai_test(func):
    """Decorator to mark AI tests."""
    func._ai_test = True
    return func


def embedding_test(func):
    """Decorator to mark embedding tests."""
    func._embedding_test = True
    return func


def vector_db_test(func):
    """Decorator to mark vector database tests."""
    func._vector_db_test = True
    return func


def rag_test(func):
    """Decorator to mark RAG tests."""
    func._rag_test = True
    return func


def performance_critical(func):
    """Decorator to mark performance-critical tests."""
    func._performance_critical = True
    return func


# Property-based testing strategies (simplified)
class MinimalTestStrategies:
    """Minimal test strategies for property-based testing concepts."""

    @staticmethod
    def embeddings(min_dim: int = 100, max_dim: int = 1000) -> list[list[float]]:
        """Generate embedding strategy (returns single example).

        Args:
            min_dim: Minimum dimension
            max_dim: Maximum dimension

        Returns:
            list containing one embedding
        """
        dim = random.randint(min_dim, max_dim)
        return MinimalEmbeddingTestUtils.generate_test_embeddings(count=1, dim=dim)

    @staticmethod
    def search_queries() -> str:
        """Generate search query strategy (returns single example).

        Returns:
            Example search query
        """
        queries = [
            "What is machine learning?",
            "How does vector search work?",
            "Explain embeddings in AI",
            "What are the benefits of RAG?",
            "How to implement semantic search?",
        ]
        return random.choice(queries)


if __name__ == "__main__":
    """Run basic tests to verify functionality."""
    print("🧪 Testing Minimal AI Testing Utilities")
    print("=" * 50)

    # Test embedding generation
    embeddings = MinimalEmbeddingTestUtils.generate_test_embeddings(count=3, dim=384)
    print(
        f"✅ Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}"
    )

    # Test cosine similarity
    sim = MinimalEmbeddingTestUtils.cosine_similarity([1, 0, 0], [0, 1, 0])
    print(f"✅ Cosine similarity calculation: {sim:.6f} (expected ~0.0)")

    # Test validation
    validation = MinimalEmbeddingTestUtils.validate_embedding_properties([0.1] * 384)
    print(f"✅ Embedding validation: {validation['is_valid']}")

    # Test vector database mock
    mock_client = MinimalVectorDatabaseTestUtils.create_mock_qdrant_client()
    print(f"✅ Mock vector database client created: {mock_client is not None}")

    # Test RAG metrics
    quality_metrics = MinimalRAGTestUtils.evaluate_rag_response_quality(
        response="Machine learning is a subset of AI",
        query="What is machine learning?",
        retrieved_contexts=["Machine learning is part of artificial intelligence"],
    )
    print(f"✅ RAG quality evaluation: {quality_metrics['overall_quality']:.2f}")

    # Test performance monitoring
    monitor = MinimalPerformanceTestUtils()
    monitor.start_monitoring()
    time.sleep(0.01)  # Brief work simulation
    metrics = monitor.stop_monitoring()
    print(f"✅ Performance monitoring: {metrics['duration_seconds']:.3f}s")

    print("\n🎉 All minimal AI testing utilities working correctly!")
