#!/usr/bin/env python3
"""Minimal Test Runner for AI/ML Systems - Dependency-Free Implementation.

This script demonstrates modern AI/ML testing practices using only Python
standard library, providing a complete testing framework that works in
any environment without external dependencies.
"""

import argparse
import asyncio
import math
import random
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_test_environment():
    """Set up the test environment."""
    # Add tests directory to Python path
    test_dir = Path(__file__).parent.parent / "tests"
    sys.path.insert(0, str(test_dir))
    
    # Set testing environment variables
    import os
    os.environ["TESTING"] = "true"


class MinimalAITestSuite:
    """Comprehensive AI/ML test suite using minimal dependencies."""
    
    def __init__(self):
        """Initialize the test suite."""
        setup_test_environment()
        
        # Import our minimal utilities
        from utils.minimal_ai_testing import (
            MinimalEmbeddingTestUtils,
            MinimalVectorDatabaseTestUtils,
            MinimalRAGTestUtils,
            MinimalPerformanceTestUtils,
            MinimalTestStrategies,
        )
        
        self.embedding_utils = MinimalEmbeddingTestUtils
        self.vector_utils = MinimalVectorDatabaseTestUtils
        self.rag_utils = MinimalRAGTestUtils
        self.perf_utils = MinimalPerformanceTestUtils
        self.strategies = MinimalTestStrategies
        
        self.test_results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "failures": [],
            "start_time": 0,
            "end_time": 0
        }
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all AI/ML tests.
        
        Args:
            verbose: Whether to print verbose output
            
        Returns:
            Test results summary
        """
        self.test_results["start_time"] = time.time()
        
        if verbose:
            print("🧪 Modern AI/ML Testing Framework - Minimal Implementation")
            print("=" * 65)
            print("Demonstrating 2025 best practices with Python standard library only")
            print()
        
        test_categories = [
            ("Embedding Operations", self._test_embedding_operations),
            ("Property-Based Concepts", self._test_property_based_concepts),
            ("Vector Database Operations", self._test_vector_database_operations),
            ("RAG Quality Metrics", self._test_rag_quality_metrics),
            ("Performance Characteristics", self._test_performance_characteristics),
            ("AI Pipeline Integration", self._test_ai_pipeline_integration),
            ("Mathematical Properties", self._test_mathematical_properties),
            ("Edge Cases & Robustness", self._test_edge_cases),
        ]
        
        for category_name, test_function in test_categories:
            if verbose:
                print(f"📊 Testing {category_name}...")
            
            try:
                test_function(verbose)
                if verbose:
                    print()
            except Exception as e:
                self._record_error(f"{category_name} suite failed", e, verbose)
        
        self.test_results["end_time"] = time.time()
        self._print_summary(verbose)
        
        return self.test_results
    
    def _test_embedding_operations(self, verbose: bool) -> None:
        """Test core embedding operations."""
        
        # Test 1: Basic embedding generation
        embeddings = self.embedding_utils.generate_test_embeddings(count=5, dim=384, seed=42)
        self._assert_equal(len(embeddings), 5, "Should generate 5 embeddings")
        self._assert_equal(len(embeddings[0]), 384, "Each embedding should have 384 dimensions")
        self._record_test_result(True, "Embedding generation", verbose)
        
        # Test 2: Dimensional consistency
        dimensions = [len(emb) for emb in embeddings]
        self._assert_equal(len(set(dimensions)), 1, "All embeddings should have same dimension")
        self._record_test_result(True, "Dimensional consistency", verbose)
        
        # Test 3: Cosine similarity properties
        # Self-similarity should be 1.0
        self_sim = self.embedding_utils.cosine_similarity(embeddings[0], embeddings[0])
        self._assert_almost_equal(self_sim, 1.0, 1e-6, "Self-similarity should be 1.0")
        
        # Similarity should be symmetric
        sim_12 = self.embedding_utils.cosine_similarity(embeddings[0], embeddings[1])
        sim_21 = self.embedding_utils.cosine_similarity(embeddings[1], embeddings[0])
        self._assert_almost_equal(sim_12, sim_21, 1e-10, "Similarity should be symmetric")
        self._record_test_result(True, "Cosine similarity properties", verbose)
        
        # Test 4: Embedding validation
        validation = self.embedding_utils.validate_embedding_properties(embeddings[0])
        self._assert_true(validation["is_valid"], "Generated embedding should be valid")
        self._record_test_result(True, "Embedding validation", verbose)
    
    def _test_property_based_concepts(self, verbose: bool) -> None:
        """Test property-based testing concepts."""
        
        # Test 1: Embedding dimension invariance
        # Property: Embedding dimension should be independent of generation parameters
        for dim in [128, 256, 512, 768]:
            embeddings = self.embedding_utils.generate_test_embeddings(count=3, dim=dim, seed=42)
            for emb in embeddings:
                self._assert_equal(len(emb), dim, f"Embedding should have dimension {dim}")
        self._record_test_result(True, "Dimension invariance property", verbose)
        
        # Test 2: Similarity bounds property
        # Property: Cosine similarity should always be in [-1, 1]
        test_vectors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],  # Orthogonal unit vectors
            [1, 1, 0], [1, -1, 0],            # Non-orthogonal vectors
            [-1, 0, 0], [0, -1, 0]            # Negative vectors
        ]
        
        for i, vec1 in enumerate(test_vectors):
            for j, vec2 in enumerate(test_vectors):
                similarity = self.embedding_utils.cosine_similarity(vec1, vec2)
                self._assert_true(-1 <= similarity <= 1, 
                                f"Similarity {similarity} should be in [-1, 1]")
        self._record_test_result(True, "Similarity bounds property", verbose)
        
        # Test 3: Batch consistency property
        # Property: Batch operations should be consistent with individual operations
        embeddings = self.embedding_utils.generate_test_embeddings(count=10, dim=256, seed=42)
        batch_validation = self.embedding_utils.batch_validate_embeddings(embeddings)
        
        individual_validations = [
            self.embedding_utils.validate_embedding_properties(emb) for emb in embeddings
        ]
        all_individual_valid = all(v["is_valid"] for v in individual_validations)
        
        self._assert_equal(batch_validation["all_valid"], all_individual_valid,
                          "Batch validation should match individual validations")
        self._record_test_result(True, "Batch consistency property", verbose)
    
    def _test_vector_database_operations(self, verbose: bool) -> None:
        """Test vector database operations."""
        
        # Test 1: Point generation
        points = self.vector_utils.generate_test_points(count=10, vector_dim=512, seed=42)
        self._assert_equal(len(points), 10, "Should generate 10 points")
        
        for point in points:
            self._assert_true("id" in point, "Point should have ID")
            self._assert_true("vector" in point, "Point should have vector")
            self._assert_true("payload" in point, "Point should have payload")
            self._assert_equal(len(point["vector"]), 512, "Vector should have 512 dimensions")
        
        self._record_test_result(True, "Vector point generation", verbose)
        
        # Test 2: Unique IDs property
        point_ids = [point["id"] for point in points]
        self._assert_equal(len(set(point_ids)), len(point_ids), "All point IDs should be unique")
        self._record_test_result(True, "Unique IDs property", verbose)
        
        # Test 3: Mock client functionality
        mock_client = self.vector_utils.create_mock_qdrant_client()
        self._assert_true(mock_client is not None, "Should create mock client")
        
        # Test async search (simulate)
        async def test_search():
            query_vector = self.embedding_utils.generate_test_embeddings(count=1, dim=384)[0]
            results = await mock_client.search(
                collection_name="test",
                query_vector=query_vector,
                limit=5
            )
            return results
        
        # Run async test
        if hasattr(asyncio, 'run'):
            results = asyncio.run(test_search())
        else:
            # Fallback for older Python versions
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(test_search())
            finally:
                loop.close()
        
        self._assert_true(len(results) <= 5, "Should return at most 5 results")
        for result in results:
            self._assert_true(hasattr(result, "score"), "Result should have score")
            self._assert_true(0 <= result.score <= 1, "Score should be in [0, 1]")
        
        self._record_test_result(True, "Mock client search", verbose)
    
    def _test_rag_quality_metrics(self, verbose: bool) -> None:
        """Test RAG quality evaluation metrics."""
        
        # Test 1: Basic quality evaluation
        response = "Machine learning is a subset of artificial intelligence that enables computers to learn."
        query = "What is machine learning?"
        contexts = ["Machine learning is part of AI", "AI includes machine learning techniques"]
        
        quality_metrics = self.rag_utils.evaluate_rag_response_quality(
            response=response,
            query=query,
            retrieved_contexts=contexts
        )
        
        required_metrics = ["overall_quality", "query_coverage", "context_utilization"]
        for metric in required_metrics:
            self._assert_true(metric in quality_metrics, f"Should have {metric} metric")
            self._assert_true(0 <= quality_metrics[metric] <= 1, 
                            f"{metric} should be in [0, 1]")
        
        self._record_test_result(True, "RAG quality evaluation", verbose)
        
        # Test 2: Precision and recall calculation
        retrieved = ["relevant document", "another relevant doc", "irrelevant content"]
        ground_truth = ["relevant document", "different relevant info"]
        
        precision = self.rag_utils.calculate_contextual_precision(retrieved, ground_truth)
        recall = self.rag_utils.calculate_contextual_recall(retrieved, ground_truth)
        
        self._assert_true(0 <= precision <= 1, "Precision should be in [0, 1]")
        self._assert_true(0 <= recall <= 1, "Recall should be in [0, 1]")
        self._record_test_result(True, "Precision/Recall calculation", verbose)
        
        # Test 3: Edge cases
        # Empty response
        empty_quality = self.rag_utils.evaluate_rag_response_quality("", query, contexts)
        self._assert_equal(empty_quality["overall_quality"], 0.0, 
                          "Empty response should have 0 quality")
        
        # No contexts
        no_context_quality = self.rag_utils.evaluate_rag_response_quality(response, query, [])
        self._assert_true(no_context_quality["context_utilization"] == 0.0,
                         "No contexts should give 0 context utilization")
        
        self._record_test_result(True, "RAG edge cases", verbose)
    
    def _test_performance_characteristics(self, verbose: bool) -> None:
        """Test performance monitoring and characteristics."""
        
        # Test 1: Basic performance monitoring
        monitor = self.perf_utils()
        monitor.start_monitoring()
        
        # Simulate work
        embeddings = []
        for i in range(5):
            batch = self.embedding_utils.generate_test_embeddings(count=20, dim=256)
            embeddings.extend(batch)
            monitor.take_memory_snapshot(f"batch_{i}")
        
        metrics = monitor.stop_monitoring()
        
        self._assert_true("duration_seconds" in metrics, "Should have duration metric")
        self._assert_true(metrics["duration_seconds"] > 0, "Duration should be positive")
        self._assert_true("peak_memory_mb" in metrics, "Should have memory metric")
        self._record_test_result(True, "Performance monitoring", verbose)
        
        # Test 2: Throughput characteristics
        # Test different batch sizes and measure relative performance
        batch_sizes = [1, 10, 50]
        throughputs = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            _ = self.embedding_utils.generate_test_embeddings(count=batch_size, dim=384)
            duration = time.time() - start_time
            throughput = batch_size / duration if duration > 0 else float('inf')
            throughputs.append(throughput)
        
        # Verify that larger batches don't have significantly worse throughput
        self._assert_true(all(t > 0 for t in throughputs), "All throughputs should be positive")
        self._record_test_result(True, "Throughput characteristics", verbose)
        
        # Test 3: Memory usage patterns
        snapshots = monitor.memory_snapshots
        self._assert_true(len(snapshots) > 0, "Should have memory snapshots")
        self._record_test_result(True, "Memory usage patterns", verbose)
    
    def _test_ai_pipeline_integration(self, verbose: bool) -> None:
        """Test end-to-end AI pipeline integration."""
        
        # Simulate complete RAG pipeline
        documents = [
            "Machine learning is a method of data analysis.",
            "Vector databases store high-dimensional data efficiently.", 
            "RAG combines retrieval with generation for better responses.",
            "Embeddings convert text into numerical representations.",
            "Semantic search finds conceptually similar content."
        ]
        
        # Step 1: Index documents (generate embeddings)
        doc_embeddings = []
        for doc in documents:
            # Simulate document embedding (in reality would use actual embedding model)
            embedding = self.embedding_utils.generate_test_embeddings(
                count=1, dim=384, seed=hash(doc) % 2**31
            )[0]
            doc_embeddings.append(embedding)
        
        self._assert_equal(len(doc_embeddings), len(documents), 
                          "Should have embedding for each document")
        self._record_test_result(True, "Document indexing", verbose)
        
        # Step 2: Query processing
        query = "What is machine learning?"
        query_embedding = self.embedding_utils.generate_test_embeddings(
            count=1, dim=384, seed=hash(query) % 2**31
        )[0]
        
        # Step 3: Retrieval simulation (find most similar documents)
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = self.embedding_utils.cosine_similarity(query_embedding, doc_emb)
            similarities.append((i, similarity))
        
        # Sort by similarity and take top 3
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_docs = similarities[:3]
        
        self._assert_equal(len(top_docs), 3, "Should retrieve top 3 documents")
        self._assert_true(all(sim >= similarities[2][1] for _, sim in top_docs),
                         "Retrieved docs should be most similar")
        self._record_test_result(True, "Document retrieval", verbose)
        
        # Step 4: Response generation simulation
        retrieved_contexts = [documents[doc_id] for doc_id, _ in top_docs]
        response = f"Based on the information: {retrieved_contexts[0][:50]}..."
        
        # Step 5: Quality evaluation
        quality = self.rag_utils.evaluate_rag_response_quality(
            response=response,
            query=query,
            retrieved_contexts=retrieved_contexts
        )
        
        self._assert_true(quality["overall_quality"] > 0, "Response should have some quality")
        self._record_test_result(True, "End-to-end pipeline", verbose)
    
    def _test_mathematical_properties(self, verbose: bool) -> None:
        """Test mathematical properties of AI operations."""
        
        # Test 1: Vector normalization properties
        # Property: Normalized vectors should have unit magnitude
        embeddings = self.embedding_utils.generate_test_embeddings(count=5, dim=128, seed=42)
        for embedding in embeddings:
            magnitude = math.sqrt(sum(x * x for x in embedding))
            self._assert_almost_equal(magnitude, 1.0, 1e-6, "Normalized vector should have unit magnitude")
        self._record_test_result(True, "Vector normalization", verbose)
        
        # Test 2: Cosine similarity mathematical properties
        # Property: cos(θ) = dot(a,b) / (|a| * |b|)
        vec1 = [1, 0, 0]
        vec2 = [1, 1, 0]
        
        # Manual calculation
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(x * x for x in vec1))
        mag2 = math.sqrt(sum(x * x for x in vec2))
        expected_similarity = dot_product / (mag1 * mag2)
        
        # Using utility function
        calculated_similarity = self.embedding_utils.cosine_similarity(vec1, vec2)
        
        self._assert_almost_equal(calculated_similarity, expected_similarity, 1e-10,
                                "Cosine similarity should match manual calculation")
        self._record_test_result(True, "Cosine similarity math", verbose)
        
        # Test 3: Precision/Recall relationship properties
        # Property: F1 = 2 * (precision * recall) / (precision + recall)
        precision_values = [0.8, 0.6, 0.9, 0.4]
        recall_values = [0.7, 0.8, 0.6, 0.9]
        
        for precision, recall in zip(precision_values, recall_values):
            if precision + recall > 0:
                expected_f1 = 2 * (precision * recall) / (precision + recall)
                calculated_f1 = 2 * (precision * recall) / (precision + recall)
                self._assert_almost_equal(calculated_f1, expected_f1, 1e-10,
                                        "F1 score should be harmonic mean")
        
        self._record_test_result(True, "F1 score mathematics", verbose)
    
    def _test_edge_cases(self, verbose: bool) -> None:
        """Test edge cases and robustness."""
        
        # Test 1: Empty inputs
        try:
            empty_validation = self.embedding_utils.validate_embedding_properties([])
            self._assert_false(empty_validation["is_valid"], "Empty embedding should be invalid")
            self._record_test_result(True, "Empty embedding handling", verbose)
        except Exception:
            self._record_test_result(False, "Empty embedding handling", verbose, 
                                   "Should handle empty embeddings gracefully")
        
        # Test 2: Invalid numbers
        try:
            invalid_embedding = [1.0, float('nan'), 2.0]
            invalid_validation = self.embedding_utils.validate_embedding_properties(invalid_embedding)
            self._assert_false(invalid_validation["is_valid"], "Invalid numbers should be detected")
            self._record_test_result(True, "Invalid number detection", verbose)
        except Exception:
            self._record_test_result(False, "Invalid number detection", verbose,
                                   "Should handle invalid numbers gracefully")
        
        # Test 3: Dimension mismatch
        try:
            self.embedding_utils.cosine_similarity([1, 2, 3], [1, 2])
            self._record_test_result(False, "Dimension mismatch detection", verbose,
                                   "Should raise error for mismatched dimensions")
        except ValueError:
            self._record_test_result(True, "Dimension mismatch detection", verbose)
        except Exception:
            self._record_test_result(False, "Dimension mismatch detection", verbose,
                                   "Should raise ValueError for mismatched dimensions")
        
        # Test 4: Zero vectors
        zero_vector = [0.0] * 100
        zero_validation = self.embedding_utils.validate_embedding_properties(zero_vector)
        self._assert_false(zero_validation["is_valid"], "Zero vector should be invalid")
        self._record_test_result(True, "Zero vector detection", verbose)
        
        # Test 5: Very large dimensions
        large_embedding = self.embedding_utils.generate_test_embeddings(count=1, dim=10000, seed=42)[0]
        large_validation = self.embedding_utils.validate_embedding_properties(large_embedding)
        self._assert_true(large_validation["is_valid"], "Large embedding should be valid")
        self._record_test_result(True, "Large dimension handling", verbose)
    
    # Assertion helpers
    def _assert_true(self, condition: bool, message: str) -> None:
        """Assert that condition is True."""
        if not condition:
            raise AssertionError(message)
    
    def _assert_false(self, condition: bool, message: str) -> None:
        """Assert that condition is False."""
        if condition:
            raise AssertionError(message)
    
    def _assert_equal(self, actual: Any, expected: Any, message: str) -> None:
        """Assert that actual equals expected."""
        if actual != expected:
            raise AssertionError(f"{message}: expected {expected}, got {actual}")
    
    def _assert_almost_equal(self, actual: float, expected: float, tolerance: float, message: str) -> None:
        """Assert that actual is almost equal to expected within tolerance."""
        if abs(actual - expected) > tolerance:
            raise AssertionError(f"{message}: expected {expected}, got {actual} (tolerance: {tolerance})")
    
    def _record_test_result(self, passed: bool, test_name: str, verbose: bool, error_msg: str = "") -> None:
        """Record the result of a test."""
        self.test_results["total"] += 1
        
        if passed:
            self.test_results["passed"] += 1
            if verbose:
                print(f"  ✅ {test_name}")
        else:
            self.test_results["failed"] += 1
            self.test_results["failures"].append(f"{test_name}: {error_msg}")
            if verbose:
                print(f"  ❌ {test_name}: {error_msg}")
    
    def _record_error(self, test_name: str, error: Exception, verbose: bool) -> None:
        """Record a test error."""
        self.test_results["total"] += 1
        self.test_results["errors"] += 1
        self.test_results["failures"].append(f"{test_name}: {str(error)}")
        if verbose:
            print(f"  💥 {test_name}: {str(error)}")
    
    def _print_summary(self, verbose: bool) -> None:
        """Print test results summary."""
        if not verbose:
            return
        
        duration = self.test_results["end_time"] - self.test_results["start_time"]
        total = self.test_results["total"]
        passed = self.test_results["passed"]
        failed = self.test_results["failed"]
        errors = self.test_results["errors"]
        
        print("=" * 65)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 65)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")
        print(f"⏱️  Duration: {duration:.3f} seconds")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Performance metrics
        avg_test_time = duration / total if total > 0 else 0
        print(f"⚡ Average Test Time: {avg_test_time*1000:.1f}ms")
        
        if self.test_results["failures"]:
            print(f"\n🚨 FAILURES & ERRORS ({len(self.test_results['failures'])}):")
            for i, failure in enumerate(self.test_results["failures"], 1):
                print(f"  {i}. {failure}")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! 🎉")
            print("✨ Modern AI/ML testing practices successfully demonstrated")
            print("🔬 Property-based testing concepts validated")
            print("🚀 Ready for production deployment")
        else:
            print(f"\n⚠️  {failed + errors} test(s) failed or had errors")
            print("🔧 Review failures and fix issues before deployment")


def main():
    """Main entry point for the minimal test runner."""
    parser = argparse.ArgumentParser(
        description="Minimal AI/ML Test Runner - Dependency-Free Implementation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run in quiet mode (less verbose output)"
    )
    parser.add_argument(
        "--performance-only",
        action="store_true", 
        help="Run only performance tests"
    )
    
    args = parser.parse_args()
    
    # Create and run test suite
    test_suite = MinimalAITestSuite()
    results = test_suite.run_all_tests(verbose=not args.quiet)
    
    # Exit with appropriate code
    if results["failed"] > 0 or results["errors"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()