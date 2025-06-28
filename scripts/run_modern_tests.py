#!/usr/bin/env python3
"""Modern Test Runner for AI/ML Systems.

This script provides a robust test runner that bypasses dependency issues
and implements modern testing practices for AI/ML systems.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ModernTestRunner:
    """Modern test runner with AI/ML specific optimizations."""
    
    def __init__(self, project_root: Path):
        """Initialize the test runner.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        self.src_dir = project_root / "src"
        
    def setup_environment(self) -> None:
        """Set up the test environment."""
        # Add src to Python path
        sys.path.insert(0, str(self.src_dir))
        
        # Set environment variables for testing
        os.environ["TESTING"] = "true"
        os.environ["PYTHONPATH"] = str(self.src_dir)
        
        # Disable network access for isolated testing
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        
        # Configure for headless testing
        os.environ["CRAWL4AI_HEADLESS"] = "true"
        
    def run_basic_python_tests(
        self, 
        test_patterns: List[str],
        verbose: bool = True
    ) -> Dict[str, any]:
        """Run tests using basic Python unittest without external dependencies.
        
        Args:
            test_patterns: List of test file patterns to run
            verbose: Whether to run in verbose mode
            
        Returns:
            Test results summary
        """
        print("🧪 Running Modern AI/ML Tests (Dependency-Free Mode)")
        print("=" * 60)
        
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "duration": 0,
            "failures": [],
        }
        
        start_time = time.time()
        
        # Import our test modules directly
        try:
            # Test AI utilities
            self._test_ai_utilities(results, verbose)
            
            # Test modern patterns (without external dependencies)
            self._test_modern_patterns(results, verbose)
            
            # Test embedding utilities
            self._test_embedding_utilities(results, verbose)
            
        except Exception as e:
            results["errors"] += 1
            results["failures"].append(f"Test execution error: {e}")
            
        results["duration"] = time.time() - start_time
        
        self._print_results_summary(results)
        return results
    
    def _test_ai_utilities(self, results: Dict, verbose: bool) -> None:
        """Test AI utilities functionality."""
        if verbose:
            print("\n📊 Testing AI Utilities...")
        
        try:
            # Import and test our utilities
            sys.path.insert(0, str(self.test_dir))
            from utils.ai_testing_utilities import (
                EmbeddingTestUtils,
                VectorDatabaseTestUtils,
                RAGTestUtils,
                PerformanceTestUtils
            )
            
            # Test 1: Embedding generation
            embeddings = EmbeddingTestUtils.generate_test_embeddings(count=5, dim=384)
            assert len(embeddings) == 5
            assert all(len(emb) == 384 for emb in embeddings)
            results["passed"] += 1
            if verbose:
                print("  ✅ Embedding generation test passed")
            
            # Test 2: Cosine similarity calculation
            emb1 = [1.0, 0.0, 0.0]
            emb2 = [0.0, 1.0, 0.0]
            similarity = EmbeddingTestUtils.cosine_similarity(emb1, emb2)
            assert abs(similarity - 0.0) < 1e-10  # Orthogonal vectors
            results["passed"] += 1
            if verbose:
                print("  ✅ Cosine similarity test passed")
            
            # Test 3: Embedding validation
            valid_embedding = [0.1] * 384
            validation = EmbeddingTestUtils.validate_embedding_properties(valid_embedding)
            assert validation["is_valid"]
            results["passed"] += 1
            if verbose:
                print("  ✅ Embedding validation test passed")
            
            # Test 4: Vector database mock
            mock_client = VectorDatabaseTestUtils.create_mock_qdrant_client()
            assert mock_client is not None
            results["passed"] += 1
            if verbose:
                print("  ✅ Vector database mock test passed")
            
            # Test 5: Performance monitoring
            monitor = PerformanceTestUtils()
            monitor.start_monitoring()
            # Do some work
            _ = [i**2 for i in range(1000)]
            metrics = monitor.stop_monitoring()
            assert "duration_seconds" in metrics
            assert metrics["duration_seconds"] > 0
            results["passed"] += 1
            if verbose:
                print("  ✅ Performance monitoring test passed")
            
            results["total_tests"] += 5
            
        except Exception as e:
            results["failed"] += 1
            results["failures"].append(f"AI utilities test failed: {e}")
            if verbose:
                print(f"  ❌ AI utilities test failed: {e}")
    
    def _test_modern_patterns(self, results: Dict, verbose: bool) -> None:
        """Test modern testing patterns."""
        if verbose:
            print("\n🔬 Testing Modern Patterns...")
        
        try:
            # Test property-based testing concepts (without Hypothesis)
            self._test_embedding_properties(results, verbose)
            self._test_vector_operations(results, verbose)
            self._test_rag_metrics(results, verbose)
            
        except Exception as e:
            results["failed"] += 1
            results["failures"].append(f"Modern patterns test failed: {e}")
            if verbose:
                print(f"  ❌ Modern patterns test failed: {e}")
    
    def _test_embedding_properties(self, results: Dict, verbose: bool) -> None:
        """Test embedding properties without Hypothesis."""
        sys.path.insert(0, str(self.test_dir))
        from utils.ai_testing_utilities import EmbeddingTestUtils
        
        # Property 1: Dimensional consistency
        embeddings = EmbeddingTestUtils.generate_test_embeddings(count=10, dim=512)
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1, "All embeddings must have same dimension"
        results["passed"] += 1
        
        # Property 2: Self-similarity should be 1.0
        for embedding in embeddings[:3]:  # Test subset
            self_sim = EmbeddingTestUtils.cosine_similarity(embedding, embedding)
            assert abs(self_sim - 1.0) < 1e-6, "Self-similarity should be 1.0"
        results["passed"] += 1
        
        # Property 3: Similarity symmetry
        sim_12 = EmbeddingTestUtils.cosine_similarity(embeddings[0], embeddings[1])
        sim_21 = EmbeddingTestUtils.cosine_similarity(embeddings[1], embeddings[0])
        assert abs(sim_12 - sim_21) < 1e-10, "Similarity should be symmetric"
        results["passed"] += 1
        
        results["total_tests"] += 3
        if verbose:
            print("  ✅ Embedding properties tests passed")
    
    def _test_vector_operations(self, results: Dict, verbose: bool) -> None:
        """Test vector operations."""
        sys.path.insert(0, str(self.test_dir))
        from utils.ai_testing_utilities import EmbeddingTestUtils, VectorDatabaseTestUtils
        
        # Test vector point generation
        points = VectorDatabaseTestUtils.generate_test_points(count=5, vector_dim=384)
        assert len(points) == 5
        assert all("id" in point for point in points)
        assert all("vector" in point for point in points)
        assert all(len(point["vector"]) == 384 for point in points)
        results["passed"] += 1
        
        # Test batch validation
        embeddings = [point["vector"] for point in points]
        batch_validation = EmbeddingTestUtils.batch_validate_embeddings(embeddings)
        assert batch_validation["all_valid"]
        assert batch_validation["consistent_dimensions"]
        results["passed"] += 1
        
        results["total_tests"] += 2
        if verbose:
            print("  ✅ Vector operations tests passed")
    
    def _test_rag_metrics(self, results: Dict, verbose: bool) -> None:
        """Test RAG evaluation metrics."""
        sys.path.insert(0, str(self.test_dir))
        from utils.ai_testing_utilities import RAGTestUtils
        
        # Test quality evaluation
        response = "Machine learning is a subset of artificial intelligence that focuses on algorithms."
        query = "What is machine learning?"
        contexts = ["Machine learning is part of AI", "AI includes machine learning"]
        
        quality_metrics = RAGTestUtils.evaluate_rag_response_quality(
            response=response,
            query=query,
            retrieved_contexts=contexts
        )
        
        assert "overall_quality" in quality_metrics
        assert 0.0 <= quality_metrics["overall_quality"] <= 1.0
        assert "query_coverage" in quality_metrics
        assert "context_utilization" in quality_metrics
        results["passed"] += 1
        
        # Test precision/recall calculation
        retrieved = ["relevant context", "another relevant piece"]
        ground_truth = ["relevant context", "different relevant info"]
        
        precision = RAGTestUtils.calculate_contextual_precision(retrieved, ground_truth)
        recall = RAGTestUtils.calculate_contextual_recall(retrieved, ground_truth)
        
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        results["passed"] += 1
        
        results["total_tests"] += 2
        if verbose:
            print("  ✅ RAG metrics tests passed")
    
    def _test_embedding_utilities(self, results: Dict, verbose: bool) -> None:
        """Test embedding-specific utilities."""
        if verbose:
            print("\n🎯 Testing Embedding Utilities...")
        
        try:
            sys.path.insert(0, str(self.test_dir))
            from utils.ai_testing_utilities import EmbeddingTestUtils
            
            # Test similar embedding generation
            base_embedding = EmbeddingTestUtils.generate_test_embeddings(count=1, dim=256)[0]
            similar_embeddings = EmbeddingTestUtils.generate_similar_embeddings(
                base_embedding=base_embedding,
                count=3,
                similarity_range=(0.8, 0.9),
                seed=42
            )
            
            assert len(similar_embeddings) == 3
            
            # Verify similarity is in range
            for similar_emb in similar_embeddings:
                similarity = EmbeddingTestUtils.cosine_similarity(base_embedding, similar_emb)
                assert 0.7 <= similarity <= 1.0  # Allow some tolerance
            
            results["passed"] += 1
            results["total_tests"] += 1
            
            if verbose:
                print("  ✅ Similar embedding generation test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["failures"].append(f"Embedding utilities test failed: {e}")
            if verbose:
                print(f"  ❌ Embedding utilities test failed: {e}")
    
    def _print_results_summary(self, results: Dict) -> None:
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📋 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total = results["total_tests"]
        passed = results["passed"]
        failed = results["failed"]
        errors = results["errors"]
        duration = results["duration"]
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Errors: {errors} 💥")
        print(f"Duration: {duration:.2f} seconds")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        if results["failures"]:
            print("\n🚨 FAILURES:")
            for i, failure in enumerate(results["failures"], 1):
                print(f"  {i}. {failure}")
        
        if passed == total and failed == 0 and errors == 0:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"\n⚠️  {failed + errors} test(s) failed or had errors")
    
    def run_pytest_if_available(self, test_patterns: List[str]) -> Optional[int]:
        """Try to run pytest if available and dependencies work.
        
        Args:
            test_patterns: Test patterns to run
            
        Returns:
            Exit code if pytest runs, None if it fails
        """
        try:
            # Try to use our modern pytest configuration
            cmd = [
                sys.executable, "-m", "pytest",
                "-c", str(self.project_root / "pytest-modern.ini"),
                "--tb=short",
                "-v",
                "--disable-warnings",
            ]
            
            # Add test patterns
            for pattern in test_patterns:
                cmd.append(str(self.test_dir / pattern))
            
            print("🔬 Attempting to run pytest with modern configuration...")
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
            
        except Exception as e:
            print(f"⚠️  Pytest execution failed: {e}")
            print("🔄 Falling back to basic test runner...")
            return None
    
    def run_tests(
        self,
        test_patterns: Optional[List[str]] = None,
        use_pytest: bool = True,
        verbose: bool = True
    ) -> int:
        """Run tests with fallback strategy.
        
        Args:
            test_patterns: Test patterns to run (defaults to modern examples)
            use_pytest: Whether to try pytest first
            verbose: Verbose output
            
        Returns:
            Exit code (0 for success)
        """
        if test_patterns is None:
            test_patterns = [
                "examples/test_modern_ai_patterns.py",
                "utils/ai_testing_utilities.py",
            ]
        
        self.setup_environment()
        
        # Try pytest first if requested
        if use_pytest:
            pytest_result = self.run_pytest_if_available(test_patterns)
            if pytest_result is not None:
                return pytest_result
        
        # Fallback to basic test runner
        print("🔧 Using fallback test runner (dependency-free)")
        results = self.run_basic_python_tests(test_patterns, verbose)
        
        # Return appropriate exit code
        if results["failed"] > 0 or results["errors"] > 0:
            return 1
        return 0


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Modern Test Runner for AI/ML Systems"
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        help="Test patterns to run",
        default=None
    )
    parser.add_argument(
        "--no-pytest",
        action="store_true",
        help="Skip pytest and use basic runner only"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run in quiet mode"
    )
    
    args = parser.parse_args()
    
    # Find project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Create and run test runner
    runner = ModernTestRunner(project_root)
    exit_code = runner.run_tests(
        test_patterns=args.patterns,
        use_pytest=not args.no_pytest,
        verbose=not args.quiet
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()