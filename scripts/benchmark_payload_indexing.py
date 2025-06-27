#!/usr/bin/env python3
"""
Performance benchmark script for payload indexing (Issue #56).

Demonstrates the 10-100x performance improvement achieved through
payload indexing on metadata fields.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

from ..config import Config
from src.services.embeddings.manager import EmbeddingManager
from src.services.vector_db.service import QdrantService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PayloadIndexingBenchmark:
    """Benchmarks payload indexing performance improvements."""

    def __init__(self):
        """Initialize benchmark with service dependencies."""
        self.config = Config()
        self.qdrant_service = QdrantService(self.config)
        self.embedding_manager = EmbeddingManager(self.config)

        # Benchmark configuration
        self.test_collection = "benchmark_payload_indexing"
        self.benchmark_results = {
            "setup_info": {},
            "index_creation_times": {},
            "search_performance": {},
            "filter_performance": {},
            "complex_queries": {},
            "summary": {},
        }

    async def initialize(self):
        """Initialize services."""
        try:
            await self.qdrant_service.initialize()
            await self.embedding_manager.initialize()
            logger.info("Successfully initialized services")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    async def cleanup(self):
        """Cleanup services and test collection."""
        try:
            # Clean up test collection
            try:
                await self.qdrant_service.delete_collection(self.test_collection)
                logger.info(f"Cleaned up test collection: {self.test_collection}")
            except Exception as e:
                logger.warning(f"Could not delete test collection: {e}")

            await self.qdrant_service.cleanup()
            await self.embedding_manager.cleanup()
            logger.info("Successfully cleaned up services")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    async def setup_test_collection(self, with_indexes: bool = True) -> dict[str, Any]:
        """Setup test collection with sample data."""
        logger.info(f"Setting up test collection (indexes: {with_indexes})")

        setup_start = time.time()

        # Create collection
        await self.qdrant_service.create_collection(
            collection_name=self.test_collection,
            vector_size=1536,  # OpenAI embedding size
            distance="Cosine",
            sparse_vector_name="sparse",
            enable_quantization=True,
        )

        # Generate sample data points
        sample_data = await self._generate_sample_data(1000)  # 1K points for testing

        # Insert sample data
        await self.qdrant_service.upsert_points(
            collection_name=self.test_collection, points=sample_data, batch_size=100
        )

        setup_time = time.time() - setup_start

        # Create indexes if requested
        index_time = 0
        if with_indexes:
            index_start = time.time()
            await self.qdrant_service.create_payload_indexes(self.test_collection)
            index_time = time.time() - index_start

        return {
            "collection_name": self.test_collection,
            "points_count": len(sample_data),
            "setup_time_seconds": setup_time,
            "index_creation_time_seconds": index_time,
            "has_indexes": with_indexes,
        }

    async def _generate_sample_data(self, count: int) -> list[dict[str, Any]]:
        """Generate sample data points with realistic metadata."""
        logger.info(f"Generating {count} sample data points")

        # Sample configurations
        doc_types = ["api", "guide", "tutorial", "reference"]
        languages = ["python", "typescript", "javascript", "rust"]
        frameworks = ["fastapi", "nextjs", "react", "django", "flask"]
        versions = ["1.0", "2.0", "3.0", "latest"]
        crawl_sources = ["crawl4ai", "browser_use", "playwright"]
        sites = [
            "FastAPI Documentation",
            "Next.js Docs",
            "React Documentation",
            "Django Docs",
        ]

        # Generate embeddings for sample texts
        sample_texts = [
            f"Documentation content for item {i}. This is a sample text that provides information about various topics in software development."
            for i in range(count)
        ]

        # Generate embeddings in batches
        embeddings = []
        batch_size = 50
        for i in range(0, len(sample_texts), batch_size):
            batch = sample_texts[i : i + batch_size]
            batch_embeddings = await self.embedding_manager.generate_embeddings(batch)
            embeddings.extend(batch_embeddings)

        # Create data points
        points = []
        for i in range(count):
            point = {
                "id": f"benchmark_point_{i}",
                "vector": embeddings[i],
                "payload": {
                    "content": sample_texts[i],
                    "title": f"Sample Document {i}",
                    # Core indexed fields
                    "doc_type": doc_types[i % len(doc_types)],
                    "language": languages[i % len(languages)],
                    "framework": frameworks[i % len(frameworks)],
                    "version": versions[i % len(versions)],
                    "crawl_source": crawl_sources[i % len(crawl_sources)],
                    # System fields
                    "site_name": sites[i % len(sites)],
                    "embedding_model": "text-embedding-3-small",
                    "embedding_provider": "openai",
                    "search_strategy": "hybrid",
                    "scraper_version": "3.0-Advanced",
                    # Metrics
                    "word_count": 50 + (i % 200),  # 50-250 words
                    "char_count": 300 + (i % 1200),  # 300-1500 chars
                    "quality_score": 60 + (i % 40),  # 60-100 quality score
                    "chunk_index": i % 10,
                    "total_chunks": 10,
                    "depth": (i % 5) + 1,  # 1-5 depth
                    "links_count": i % 50,  # 0-49 links
                    # Timestamps
                    "created_at": 1640995200 + (i * 3600),  # Spread over time
                    "last_updated": 1640995200 + (i * 3600),
                    "crawl_timestamp": 1640995200 + (i * 3600),
                },
            }
            points.append(point)

        return points

    async def benchmark_index_creation(self) -> dict[str, float]:
        """Benchmark index creation performance."""
        logger.info("Benchmarking index creation performance")

        # Setup collection without indexes first
        await self.setup_test_collection(with_indexes=False)

        # Time index creation
        start_time = time.time()
        await self.qdrant_service.create_payload_indexes(self.test_collection)
        creation_time = time.time() - start_time

        # Get index statistics
        stats = await self.qdrant_service.get_payload_index_stats(self.test_collection)

        results = {
            "creation_time_seconds": creation_time,
            "indexes_created": stats["indexed_fields_count"],
            "points_indexed": stats["total_points"],
            "creation_rate_points_per_second": stats["total_points"] / creation_time
            if creation_time > 0
            else 0,
        }

        logger.info(
            f"Index creation completed in {creation_time:.2f}s for {stats['total_points']} points"
        )
        return results

    async def benchmark_filtered_search(self) -> dict[str, Any]:
        """Benchmark filtered search performance with and without indexes."""
        logger.info("Benchmarking filtered search performance")

        # Generate test query
        query_text = "sample documentation content"
        embeddings = await self.embedding_manager.generate_embeddings([query_text])
        query_vector = embeddings[0]

        # Test scenarios
        test_scenarios = [
            {
                "name": "language_filter",
                "filters": {"language": "python"},
                "description": "Filter by programming language",
            },
            {
                "name": "framework_filter",
                "filters": {"framework": "fastapi"},
                "description": "Filter by framework",
            },
            {
                "name": "doc_type_filter",
                "filters": {"doc_type": "api"},
                "description": "Filter by document type",
            },
            {
                "name": "word_count_range",
                "filters": {"min_word_count": 100, "max_word_count": 200},
                "description": "Filter by word count range",
            },
            {
                "name": "quality_filter",
                "filters": {"min_quality_score": 80},
                "description": "Filter by quality score",
            },
            {
                "name": "complex_multi_filter",
                "filters": {
                    "language": "python",
                    "framework": "fastapi",
                    "doc_type": "api",
                    "min_word_count": 100,
                    "min_quality_score": 75,
                },
                "description": "Complex multi-field filter",
            },
        ]

        results = {}

        for scenario in test_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")

            # Run multiple iterations for reliable timing
            iterations = 5
            total_time = 0

            for _i in range(iterations):
                start_time = time.time()

                search_results = await self.qdrant_service.filtered_search(
                    collection_name=self.test_collection,
                    query_vector=query_vector,
                    filters=scenario["filters"],
                    limit=10,
                    search_accuracy="balanced",
                )

                iteration_time = time.time() - start_time
                total_time += iteration_time

            avg_time = total_time / iterations

            results[scenario["name"]] = {
                "description": scenario["description"],
                "filters": scenario["filters"],
                "avg_time_ms": avg_time * 1000,
                "results_found": len(search_results),
                "iterations": iterations,
            }

        return results

    async def benchmark_comparison(self) -> dict[str, Any]:
        """Compare performance with and without indexes."""
        logger.info("Running comprehensive comparison benchmark")

        comparison_results = {
            "with_indexes": {},
            "without_indexes": {},
            "improvements": {},
        }

        # Test with indexes
        logger.info("Testing WITH payload indexes")
        await self.setup_test_collection(with_indexes=True)
        comparison_results["with_indexes"] = await self.benchmark_filtered_search()

        # Clean up and test without indexes
        await self.qdrant_service.delete_collection(self.test_collection)
        logger.info("Testing WITHOUT payload indexes")
        await self.setup_test_collection(with_indexes=False)
        comparison_results["without_indexes"] = await self.benchmark_filtered_search()

        # Calculate improvements
        improvements = {}
        for scenario_name in comparison_results["with_indexes"]:
            with_time = comparison_results["with_indexes"][scenario_name]["avg_time_ms"]
            without_time = comparison_results["without_indexes"][scenario_name][
                "avg_time_ms"
            ]

            if without_time > 0:
                improvement_factor = without_time / with_time
                improvement_percent = ((without_time - with_time) / without_time) * 100
            else:
                improvement_factor = 1.0
                improvement_percent = 0.0

            improvements[scenario_name] = {
                "improvement_factor": improvement_factor,
                "improvement_percent": improvement_percent,
                "time_with_indexes_ms": with_time,
                "time_without_indexes_ms": without_time,
            }

        comparison_results["improvements"] = improvements
        return comparison_results

    async def run_full_benchmark(self) -> dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting comprehensive payload indexing benchmark")

        start_time = datetime.now(tz=timezone.utc)

        try:
            # Benchmark 1: Index creation performance
            self.benchmark_results[
                "index_creation_times"
            ] = await self.benchmark_index_creation()

            # Benchmark 2: Filtered search performance
            await self.qdrant_service.delete_collection(self.test_collection)
            await self.setup_test_collection(with_indexes=True)
            self.benchmark_results[
                "search_performance"
            ] = await self.benchmark_filtered_search()

            # Benchmark 3: Comprehensive comparison
            self.benchmark_results[
                "filter_performance"
            ] = await self.benchmark_comparison()

            # Add summary
            end_time = datetime.now(tz=timezone.utc)
            duration = (end_time - start_time).total_seconds()

            self.benchmark_results["setup_info"] = {
                "benchmark_start": start_time.isoformat(),
                "benchmark_end": end_time.isoformat(),
                "total_duration_seconds": duration,
                "test_collection": self.test_collection,
                "qdrant_url": self.config.qdrant.url,
            }

            return self.benchmark_results

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def print_benchmark_results(self, results: dict[str, Any]):
        """Print formatted benchmark results."""
        print("\n" + "=" * 80)
        print("PAYLOAD INDEXING PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)

        # Setup info
        setup = results["setup_info"]
        print(f"\nBenchmark Duration: {setup['total_duration_seconds']:.2f} seconds")
        print(f"Test Collection: {setup['test_collection']}")
        print(f"Qdrant URL: {setup['qdrant_url']}")

        # Index creation performance
        if "index_creation_times" in results:
            index_stats = results["index_creation_times"]
            print("\n📊 INDEX CREATION PERFORMANCE")
            print("-" * 40)
            print(f"Creation Time: {index_stats['creation_time_seconds']:.2f} seconds")
            print(f"Indexes Created: {index_stats['indexes_created']}")
            print(f"Points Indexed: {index_stats['points_indexed']}")
            print(
                f"Rate: {index_stats['creation_rate_points_per_second']:.0f} points/second"
            )

        # Search performance
        if "search_performance" in results:
            print("\n🔍 SEARCH PERFORMANCE (WITH INDEXES)")
            print("-" * 40)
            for scenario, data in results["search_performance"].items():
                print(
                    f"{scenario.replace('_', ' ').title()}: {data['avg_time_ms']:.2f}ms ({data['results_found']} results)"
                )

        # Performance comparison
        if (
            "filter_performance" in results
            and "improvements" in results["filter_performance"]
        ):
            print("\n🚀 PERFORMANCE IMPROVEMENTS")
            print("-" * 40)
            improvements = results["filter_performance"]["improvements"]

            for scenario, data in improvements.items():
                factor = data["improvement_factor"]
                percent = data["improvement_percent"]
                with_idx = data["time_with_indexes_ms"]
                without_idx = data["time_without_indexes_ms"]

                print(f"{scenario.replace('_', ' ').title()}:")
                print(f"  With indexes:    {with_idx:.2f}ms")
                print(f"  Without indexes: {without_idx:.2f}ms")
                print(
                    f"  Improvement:     {factor:.1f}x faster ({percent:.1f}% reduction)"
                )
                print()

        # Summary
        if improvements:
            avg_improvement = sum(
                data["improvement_factor"] for data in improvements.values()
            ) / len(improvements)
            max_improvement = max(
                data["improvement_factor"] for data in improvements.values()
            )

            print("🎯 SUMMARY")
            print("-" * 40)
            print(f"Average improvement: {avg_improvement:.1f}x faster")
            print(f"Maximum improvement: {max_improvement:.1f}x faster")
            print(
                f"Target achieved: {'✅ YES' if avg_improvement >= 10 else '❌ NO'} (target: 10-100x)"
            )

        print("=" * 80)


async def main():
    """Main benchmark function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark payload indexing performance improvements"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Number of test points to generate (default: 1000)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize benchmark
    benchmark = PayloadIndexingBenchmark()

    try:
        await benchmark.initialize()

        # Run benchmark
        results = await benchmark.run_full_benchmark()

        # Print results
        benchmark.print_benchmark_results(results)

    except KeyboardInterrupt:
        logger.info("Benchmark cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
