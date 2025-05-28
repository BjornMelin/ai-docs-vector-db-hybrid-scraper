#!/usr/bin/env python3
"""
HNSW Configuration Optimization Benchmark Script (Issue #57).

Extends the existing payload indexing benchmark to include comprehensive
HNSW parameter testing and optimization capabilities.
"""

import asyncio
import contextlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Import the existing benchmark as a base
from benchmark_payload_indexing import PayloadIndexingBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HNSWOptimizationBenchmark(PayloadIndexingBenchmark):
    """Extended benchmark for HNSW parameter optimization."""

    def __init__(self):
        """Initialize HNSW benchmark with extended configuration."""
        super().__init__()

        # HNSW test configurations
        self.hnsw_configs = {
            "default": {
                "m": 16,
                "ef_construct": 128,
                "description": "Qdrant default configuration",
            },
            "documentation_optimized": {
                "m": 16,
                "ef_construct": 200,
                "description": "Optimized for documentation search workloads",
            },
            "high_accuracy": {
                "m": 20,
                "ef_construct": 300,
                "description": "High accuracy configuration for API reference",
            },
            "balanced": {
                "m": 16,
                "ef_construct": 200,
                "description": "Balanced configuration for tutorials",
            },
            "fast": {
                "m": 12,
                "ef_construct": 150,
                "description": "Fast configuration for blog posts",
            },
        }

        # Runtime EF testing values
        self.ef_test_values = [50, 75, 100, 125, 150, 175, 200]

        # Benchmark results for HNSW
        self.hnsw_results = {
            "configuration_tests": {},
            "ef_runtime_tests": {},
            "adaptive_ef_tests": {},
            "comparison_summary": {},
        }

    async def benchmark_hnsw_configurations(self) -> dict[str, Any]:
        """Benchmark different HNSW build configurations.

        Returns:
            Dictionary with results for each configuration
        """
        logger.info("Benchmarking HNSW build configurations")

        config_results = {}

        for config_name, hnsw_config in self.hnsw_configs.items():
            logger.info(f"Testing HNSW configuration: {config_name}")

            # Create test collection with specific HNSW config
            collection_name = f"hnsw_test_{config_name}"

            try:
                # Delete existing collection if present
                with contextlib.suppress(Exception):
                    await self.qdrant_service.delete_collection(collection_name)

                # Time collection creation with HNSW config
                creation_start = time.time()

                await self._create_collection_with_hnsw_config(
                    collection_name, hnsw_config
                )

                creation_time = time.time() - creation_start

                # Generate and insert test data
                sample_data = await self._generate_sample_data(
                    500
                )  # Smaller for config testing

                insertion_start = time.time()
                await self.qdrant_service.upsert_points(
                    collection_name=collection_name, points=sample_data, batch_size=50
                )
                insertion_time = time.time() - insertion_start

                # Test search performance with different EF values
                search_results = await self._test_search_performance(
                    collection_name,
                    sample_data[:10],  # Use first 10 as queries
                )

                config_results[config_name] = {
                    "config": hnsw_config,
                    "creation_time_seconds": creation_time,
                    "insertion_time_seconds": insertion_time,
                    "points_count": len(sample_data),
                    "search_performance": search_results,
                    "index_size_estimate": len(sample_data)
                    * 1536
                    * 4
                    * (hnsw_config["m"] / 16),  # Rough estimate
                }

                logger.info(
                    f"Completed {config_name}: {creation_time:.2f}s creation, {insertion_time:.2f}s insertion"
                )

            except Exception as e:
                logger.error(f"Failed to test configuration {config_name}: {e}")
                config_results[config_name] = {
                    "config": hnsw_config,
                    "error": str(e),
                    "creation_time_seconds": -1,
                    "insertion_time_seconds": -1,
                }

            finally:
                # Cleanup test collection
                try:
                    await self.qdrant_service.delete_collection(collection_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to cleanup collection {collection_name}: {e}"
                    )

        return config_results

    async def _create_collection_with_hnsw_config(
        self, collection_name: str, hnsw_config: dict[str, Any]
    ) -> None:
        """Create collection with specific HNSW configuration.

        Args:
            collection_name: Name for the collection
            hnsw_config: HNSW configuration parameters
        """
        from qdrant_client import models

        # Configure HNSW parameters
        hnsw_config_obj = models.HnswConfigDiff(
            m=hnsw_config.get("m", 16),
            ef_construct=hnsw_config.get("ef_construct", 128),
            full_scan_threshold=hnsw_config.get("full_scan_threshold", 10000),
            max_indexing_threads=hnsw_config.get("max_indexing_threads", 0),  # 0 = auto
        )

        # Create vectors config with HNSW settings
        vectors_config = {
            "dense": models.VectorParams(
                size=1536,  # OpenAI embedding size
                distance=models.Distance.COSINE,
                hnsw_config=hnsw_config_obj,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.95,
                        always_ram=True,
                    )
                ),
            )
        }

        # Create collection with HNSW config
        await self.qdrant_service._client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )

    async def _test_search_performance(
        self, collection_name: str, query_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Test search performance with different EF values.

        Args:
            collection_name: Collection to test
            query_data: Sample data to use as queries

        Returns:
            Performance results for different EF values
        """
        ef_results = {}

        for ef_value in self.ef_test_values:
            logger.debug(f"Testing EF={ef_value}")

            times = []
            recall_scores = []

            # Run multiple search iterations for reliable timing
            for query_item in query_data[:5]:  # Use first 5 queries
                query_vector = query_item["vector"]

                # Measure search time
                start_time = time.time()

                try:
                    # Perform search with specific EF
                    results = await self.qdrant_service._client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        limit=10,
                        search_params=models.SearchParams(
                            hnsw_ef=ef_value, exact=False
                        ),
                    )

                    search_time = time.time() - start_time
                    times.append(search_time)

                    # Calculate recall (simplified - compare with exact search)
                    recall = await self._calculate_recall(
                        collection_name, query_vector, results, ef_value
                    )
                    recall_scores.append(recall)

                except Exception as e:
                    logger.warning(f"Search failed for EF={ef_value}: {e}")
                    continue

            if times:
                ef_results[f"ef_{ef_value}"] = {
                    "ef_value": ef_value,
                    "avg_time_ms": np.mean(times) * 1000,
                    "min_time_ms": np.min(times) * 1000,
                    "max_time_ms": np.max(times) * 1000,
                    "p95_time_ms": np.percentile(times, 95) * 1000,
                    "avg_recall": np.mean(recall_scores) if recall_scores else 0.0,
                    "min_recall": np.min(recall_scores) if recall_scores else 0.0,
                    "iterations": len(times),
                }

        return ef_results

    async def _calculate_recall(
        self,
        collection_name: str,
        query_vector: list[float],
        hnsw_results: list,
        ef_value: int,
    ) -> float:
        """Calculate recall by comparing with exact search.

        Args:
            collection_name: Collection name
            query_vector: Query vector
            hnsw_results: Results from HNSW search
            ef_value: EF value used

        Returns:
            Recall score (0.0 to 1.0)
        """
        try:
            # Get exact search results
            exact_results = await self.qdrant_service._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=len(hnsw_results),
                search_params=models.SearchParams(exact=True),
            )

            # Extract IDs
            hnsw_ids = {str(result.id) for result in hnsw_results}
            exact_ids = {str(result.id) for result in exact_results}

            # Calculate recall
            intersection = len(hnsw_ids.intersection(exact_ids))
            recall = intersection / len(exact_ids) if exact_ids else 0.0

            return recall

        except Exception as e:
            logger.debug(f"Failed to calculate recall for EF={ef_value}: {e}")
            return 0.0

    async def benchmark_adaptive_ef(self) -> dict[str, Any]:
        """Benchmark adaptive EF selection based on time budget.

        Returns:
            Results from adaptive EF testing
        """
        logger.info("Benchmarking adaptive EF selection")

        # Create optimized collection for testing
        _test_collection = "adaptive_ef_test"

        try:
            await self.setup_test_collection(with_indexes=True)

            # Test different time budgets
            time_budgets = [20, 50, 100, 200, 500]  # milliseconds

            adaptive_results = {}

            for time_budget in time_budgets:
                logger.info(f"Testing time budget: {time_budget}ms")

                results = await self._test_adaptive_ef_for_budget(
                    self.test_collection, time_budget
                )

                adaptive_results[f"budget_{time_budget}ms"] = results

            return adaptive_results

        except Exception as e:
            logger.error(f"Adaptive EF testing failed: {e}")
            return {"error": str(e)}

    async def _test_adaptive_ef_for_budget(
        self, collection_name: str, time_budget_ms: int
    ) -> dict[str, Any]:
        """Test adaptive EF selection for a specific time budget.

        Args:
            collection_name: Collection to test
            time_budget_ms: Time budget in milliseconds

        Returns:
            Results for this time budget
        """
        # Generate test query
        query_text = "sample documentation content for adaptive testing"
        embeddings = await self.embedding_manager.generate_embeddings([query_text])
        query_vector = embeddings[0]

        # Implement adaptive EF algorithm
        ef = 50  # Starting EF
        max_ef = 200
        search_times = []
        ef_values_used = []

        while ef <= max_ef:
            start_time = time.time()

            try:
                await self.qdrant_service._client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=10,
                    search_params=models.SearchParams(hnsw_ef=ef, exact=False),
                )

                search_time_ms = (time.time() - start_time) * 1000
                search_times.append(search_time_ms)
                ef_values_used.append(ef)

                # If we have time budget remaining, increase quality
                if search_time_ms < time_budget_ms * 0.7:
                    ef = min(ef + 25, max_ef)
                else:
                    break

            except Exception as e:
                logger.warning(f"Adaptive EF test failed at EF={ef}: {e}")
                break

        # Calculate final performance
        final_ef = ef_values_used[-1] if ef_values_used else 50
        final_time = search_times[-1] if search_times else 0.0

        return {
            "time_budget_ms": time_budget_ms,
            "final_ef": final_ef,
            "final_search_time_ms": final_time,
            "ef_progression": ef_values_used,
            "time_progression": search_times,
            "budget_utilized_percent": (final_time / time_budget_ms) * 100
            if time_budget_ms > 0
            else 0,
            "ef_efficiency": final_ef / final_time if final_time > 0 else 0,
        }

    async def run_comprehensive_hnsw_benchmark(self) -> dict[str, Any]:
        """Run complete HNSW optimization benchmark suite.

        Returns:
            Comprehensive benchmark results
        """
        logger.info("Starting comprehensive HNSW optimization benchmark")

        start_time = datetime.now()

        try:
            # 1. Test different HNSW build configurations
            self.hnsw_results[
                "configuration_tests"
            ] = await self.benchmark_hnsw_configurations()

            # 2. Test adaptive EF selection
            self.hnsw_results["adaptive_ef_tests"] = await self.benchmark_adaptive_ef()

            # 3. Create comparison summary
            self.hnsw_results["comparison_summary"] = self._create_comparison_summary()

            # 4. Add metadata
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.hnsw_results["metadata"] = {
                "benchmark_start": start_time.isoformat(),
                "benchmark_end": end_time.isoformat(),
                "total_duration_seconds": duration,
                "qdrant_url": self.config.qdrant.url,
                "hnsw_configs_tested": len(self.hnsw_configs),
                "ef_values_tested": len(self.ef_test_values),
            }

            return self.hnsw_results

        except Exception as e:
            logger.error(f"Comprehensive HNSW benchmark failed: {e}")
            raise

    def _create_comparison_summary(self) -> dict[str, Any]:
        """Create summary comparing all HNSW configurations.

        Returns:
            Comparison summary with recommendations
        """
        config_tests = self.hnsw_results.get("configuration_tests", {})

        if not config_tests:
            return {"error": "No configuration test results available"}

        # Analyze results
        best_creation_time = None
        best_search_performance = None
        best_overall = None

        performance_scores = {}

        for config_name, results in config_tests.items():
            if "error" in results:
                continue

            creation_time = results.get("creation_time_seconds", float("inf"))
            search_perf = results.get("search_performance", {})

            # Calculate average search time across EF values
            avg_search_times = []
            avg_recalls = []

            for ef_result in search_perf.values():
                if isinstance(ef_result, dict):
                    avg_search_times.append(ef_result.get("avg_time_ms", 0))
                    avg_recalls.append(ef_result.get("avg_recall", 0))

            if avg_search_times and avg_recalls:
                avg_search_time = np.mean(avg_search_times)
                avg_recall = np.mean(avg_recalls)

                # Calculate performance score (higher is better)
                # Balance between speed (inverse of time) and accuracy (recall)
                speed_score = 1000 / avg_search_time if avg_search_time > 0 else 0
                accuracy_score = avg_recall * 100

                # Weighted combination (60% accuracy, 40% speed)
                performance_score = (accuracy_score * 0.6) + (speed_score * 0.4)

                performance_scores[config_name] = {
                    "performance_score": performance_score,
                    "avg_search_time_ms": avg_search_time,
                    "avg_recall": avg_recall,
                    "creation_time_seconds": creation_time,
                    "config": results.get("config", {}),
                }

                # Track bests
                if best_creation_time is None or creation_time < best_creation_time[1]:
                    best_creation_time = (config_name, creation_time)

                if (
                    best_search_performance is None
                    or avg_search_time < best_search_performance[1]
                ):
                    best_search_performance = (config_name, avg_search_time)

                if best_overall is None or performance_score > best_overall[1]:
                    best_overall = (config_name, performance_score)

        # Generate recommendations
        recommendations = {
            "fastest_creation": best_creation_time[0] if best_creation_time else None,
            "fastest_search": best_search_performance[0]
            if best_search_performance
            else None,
            "best_overall": best_overall[0] if best_overall else None,
            "recommended_config": best_overall[0] if best_overall else "balanced",
        }

        if best_overall:
            best_config_details = performance_scores[best_overall[0]]
            recommendations["recommended_settings"] = {
                "m": best_config_details["config"].get("m", 16),
                "ef_construct": best_config_details["config"].get("ef_construct", 200),
                "expected_performance": {
                    "avg_search_time_ms": best_config_details["avg_search_time_ms"],
                    "avg_recall": best_config_details["avg_recall"],
                    "performance_score": best_config_details["performance_score"],
                },
            }

        return {
            "performance_scores": performance_scores,
            "recommendations": recommendations,
            "summary_stats": {
                "configs_tested": len(performance_scores),
                "best_performance_score": best_overall[1] if best_overall else 0,
                "avg_performance_score": np.mean(
                    [s["performance_score"] for s in performance_scores.values()]
                )
                if performance_scores
                else 0,
            },
        }

    def print_hnsw_results(self, results: dict[str, Any]):
        """Print formatted HNSW benchmark results."""
        self._print_header()
        self._print_metadata(results)
        self._print_config_tests(results)
        self._print_recommendations(results)
        self._print_adaptive_results(results)
        self._print_footer()

    def _print_header(self):
        """Print benchmark header."""
        print("\n" + "=" * 80)
        print("HNSW CONFIGURATION OPTIMIZATION BENCHMARK RESULTS")
        print("=" * 80)

    def _print_metadata(self, results: dict[str, Any]):
        """Print benchmark metadata."""
        metadata = results.get("metadata", {})
        print(
            f"\nBenchmark Duration: {metadata.get('total_duration_seconds', 0):.2f} seconds"
        )
        print(f"Qdrant URL: {metadata.get('qdrant_url', 'unknown')}")
        print(f"Configurations Tested: {metadata.get('hnsw_configs_tested', 0)}")

    def _print_config_tests(self, results: dict[str, Any]):
        """Print configuration test results."""
        config_tests = results.get("configuration_tests", {})
        if not config_tests:
            return

        print("\n🔧 HNSW CONFIGURATION PERFORMANCE")
        print("-" * 50)

        for config_name, config_results in config_tests.items():
            self._print_config_result(config_name, config_results)

    def _print_config_result(self, config_name: str, config_results: dict[str, Any]):
        """Print individual configuration result."""
        if "error" in config_results:
            print(f"{config_name}: ERROR - {config_results['error']}")
            return

        creation_time = config_results.get("creation_time_seconds", 0)
        insertion_time = config_results.get("insertion_time_seconds", 0)
        points_count = config_results.get("points_count", 0)

        print(f"\n{config_name.replace('_', ' ').title()}:")
        print(f"  Creation: {creation_time:.2f}s")
        print(f"  Insertion: {insertion_time:.2f}s")
        print(f"  Points: {points_count}")

        # Show best EF performance
        search_perf = config_results.get("search_performance", {})
        if search_perf:
            best_ef = min(
                search_perf.values(),
                key=lambda x: x.get("avg_time_ms", float("inf")),
                default={},
            )
            if best_ef:
                print(
                    f"  Best EF: {best_ef.get('ef_value', 'unknown')} "
                    f"({best_ef.get('avg_time_ms', 0):.1f}ms, "
                    f"{best_ef.get('avg_recall', 0):.1%} recall)"
                )

    def _print_recommendations(self, results: dict[str, Any]):
        """Print recommendations summary."""
        summary = results.get("comparison_summary", {})
        if not summary or "recommendations" not in summary:
            return

        recommendations = summary["recommendations"]
        print("\n🎯 RECOMMENDATIONS")
        print("-" * 40)
        print(f"Best Overall: {recommendations.get('recommended_config', 'unknown')}")
        print(f"Fastest Creation: {recommendations.get('fastest_creation', 'unknown')}")
        print(f"Fastest Search: {recommendations.get('fastest_search', 'unknown')}")

        if "recommended_settings" in recommendations:
            settings = recommendations["recommended_settings"]
            print("\nRecommended HNSW Settings:")
            print(f"  m = {settings.get('m', 16)}")
            print(f"  ef_construct = {settings.get('ef_construct', 200)}")

            expected = settings.get("expected_performance", {})
            print(
                f"  Expected: {expected.get('avg_search_time_ms', 0):.1f}ms, "
                f"{expected.get('avg_recall', 0):.1%} recall"
            )

    def _print_adaptive_results(self, results: dict[str, Any]):
        """Print adaptive EF results."""
        adaptive_results = results.get("adaptive_ef_tests", {})
        if not adaptive_results or "error" in adaptive_results:
            return

        print("\n⚡ ADAPTIVE EF PERFORMANCE")
        print("-" * 40)

        for _budget_key, budget_results in adaptive_results.items():
            if isinstance(budget_results, dict) and "time_budget_ms" in budget_results:
                budget = budget_results["time_budget_ms"]
                final_ef = budget_results.get("final_ef", 0)
                final_time = budget_results.get("final_search_time_ms", 0)
                utilization = budget_results.get("budget_utilized_percent", 0)

                print(
                    f"Budget {budget}ms: EF={final_ef}, "
                    f"Time={final_time:.1f}ms ({utilization:.0f}% used)"
                )

    def _print_footer(self):
        """Print benchmark footer."""
        print("\n" + "=" * 80)


async def main():
    """Main HNSW benchmark function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark HNSW configuration optimizations"
    )
    parser.add_argument(
        "--config-only", action="store_true", help="Only test build configurations"
    )
    parser.add_argument(
        "--adaptive-only", action="store_true", help="Only test adaptive EF selection"
    )
    parser.add_argument("--output", type=str, help="Output file for JSON results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize benchmark
    benchmark = HNSWOptimizationBenchmark()

    try:
        await benchmark.initialize()

        # Run selected benchmarks
        if args.config_only:
            logger.info("Running HNSW configuration tests only")
            results = {
                "configuration_tests": await benchmark.benchmark_hnsw_configurations()
            }
        elif args.adaptive_only:
            logger.info("Running adaptive EF tests only")
            results = {"adaptive_ef_tests": await benchmark.benchmark_adaptive_ef()}
        else:
            logger.info("Running comprehensive HNSW benchmark")
            results = await benchmark.run_comprehensive_hnsw_benchmark()

        # Print results
        benchmark.print_hnsw_results(results)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")

    except KeyboardInterrupt:
        logger.info("Benchmark cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    # Fix missing imports
    from qdrant_client import models

    asyncio.run(main())
