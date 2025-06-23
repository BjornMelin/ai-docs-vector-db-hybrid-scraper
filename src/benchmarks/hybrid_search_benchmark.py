import typing

"""Main benchmark orchestrator for Advanced Hybrid Search system.

This module provides the primary benchmarking interface for comprehensive
performance testing of the advanced hybrid search implementation.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ..config import Config
from ..models.vector_search import AdvancedHybridSearchRequest
from ..models.vector_search import FusionConfig
from ..models.vector_search import SearchParams
from ..services.vector_db.hybrid_search import AdvancedHybridSearchService
from .benchmark_reporter import BenchmarkReporter
from .component_benchmarks import ComponentBenchmarks
from .load_test_runner import LoadTestConfig
from .load_test_runner import LoadTestRunner
from .metrics_collector import MetricsCollector
from .performance_profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""

    name: str = Field(..., description="Benchmark name")
    description: str = Field(..., description="Benchmark description")
    target_latency_p95_ms: float = Field(
        300.0, description="Target p95 latency in milliseconds"
    )
    target_throughput_qps: float = Field(
        500.0, description="Target throughput in queries per second"
    )
    target_memory_mb: float = Field(2048.0, description="Target memory usage in MB")
    target_cache_hit_rate: float = Field(0.8, description="Target cache hit rate")
    target_accuracy: float = Field(0.85, description="Target ML accuracy")

    # Test data configuration
    test_queries_per_type: int = Field(
        100, description="Number of test queries per type"
    )
    concurrent_users: list[int] = Field(
        [10, 50, 200], description="Concurrent user levels to test"
    )
    test_duration_seconds: int = Field(
        300, description="Duration for sustained load tests"
    )

    # Component testing
    enable_component_benchmarks: bool = Field(
        True, description="Enable component-level benchmarks"
    )
    enable_load_testing: bool = Field(True, description="Enable load testing")
    enable_profiling: bool = Field(True, description="Enable detailed profiling")
    enable_optimization_analysis: bool = Field(
        True, description="Enable optimization analysis"
    )


class BenchmarkResults(BaseModel):
    """Comprehensive benchmark results."""

    benchmark_name: str = Field(..., description="Name of the benchmark")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Benchmark execution time"
    )
    duration_seconds: float = Field(..., description="Total benchmark duration")

    # Performance metrics
    latency_metrics: dict[str, float] = Field(
        default_factory=dict, description="Latency percentiles"
    )
    throughput_metrics: dict[str, float] = Field(
        default_factory=dict, description="Throughput measurements"
    )
    resource_metrics: dict[str, float] = Field(
        default_factory=dict, description="Resource utilization"
    )
    accuracy_metrics: dict[str, float] = Field(
        default_factory=dict, description="ML accuracy metrics"
    )

    # Component-specific results
    component_results: dict[str, Any] = Field(
        default_factory=dict, description="Component benchmark results"
    )
    load_test_results: dict[str, Any] = Field(
        default_factory=dict, description="Load test results"
    )
    profiling_results: dict[str, Any] = Field(
        default_factory=dict, description="Profiling results"
    )

    # Optimization recommendations
    optimization_recommendations: list[str] = Field(
        default_factory=list, description="Optimization suggestions"
    )
    performance_bottlenecks: list[str] = Field(
        default_factory=list, description="Identified bottlenecks"
    )

    # Pass/fail status
    meets_targets: bool = Field(..., description="Whether benchmark meets all targets")
    failed_targets: list[str] = Field(
        default_factory=list, description="List of failed targets"
    )


class AdvancedHybridSearchBenchmark:
    """Main orchestrator for comprehensive Advanced Hybrid Search benchmarks."""

    def __init__(
        self,
        config: Config,
        search_service: AdvancedHybridSearchService,
        benchmark_config: BenchmarkConfig,
    ):
        """Initialize the benchmark orchestrator.

        Args:
            config: Unified configuration
            search_service: Advanced hybrid search service to benchmark
            benchmark_config: Benchmark configuration
        """
        self.config = config
        self.search_service = search_service
        self.benchmark_config = benchmark_config

        # Initialize benchmark components
        self.component_benchmarks = ComponentBenchmarks(config)
        self.load_test_runner = LoadTestRunner(config)
        self.performance_profiler = PerformanceProfiler()
        self.metrics_collector = MetricsCollector(config)
        self.benchmark_reporter = BenchmarkReporter()

        # Test data
        self.test_queries = self._generate_test_queries()

    def _generate_test_queries(self) -> list[AdvancedHybridSearchRequest]:
        """Generate realistic test queries for benchmarking."""
        test_data = [
            # Code queries
            ("How to implement async functions in Python?", "code"),
            ("JavaScript promises vs async/await performance", "code"),
            ("def factorial(n): return 1 if n <= 1 else n * factorial(n-1)", "code"),
            ("React useEffect cleanup function best practices", "code"),
            ("Python list comprehension memory usage optimization", "code"),
            # Documentation queries
            ("FastAPI dependency injection tutorial", "documentation"),
            ("Django ORM relationship configuration guide", "documentation"),
            ("TypeScript interface vs type alias differences", "documentation"),
            ("Docker multi-stage build optimization techniques", "documentation"),
            ("Kubernetes horizontal pod autoscaling setup", "documentation"),
            # API reference queries
            ("GET /api/users endpoint parameters", "api_reference"),
            ("REST API authentication header format", "api_reference"),
            ("GraphQL mutation syntax examples", "api_reference"),
            ("OpenAPI schema validation rules", "api_reference"),
            ("JWT token payload structure", "api_reference"),
            # Troubleshooting queries
            ("TypeError: 'NoneType' object is not iterable", "troubleshooting"),
            ("CORS error in React application", "troubleshooting"),
            ("Database connection timeout handling", "troubleshooting"),
            ("Memory leak in Node.js application", "troubleshooting"),
            ("502 Bad Gateway nginx configuration", "troubleshooting"),
            # Conceptual queries
            ("What is microservices architecture?", "conceptual"),
            ("Machine learning vs deep learning differences", "conceptual"),
            ("Event-driven architecture benefits and drawbacks", "conceptual"),
            ("SOLID principles in software design", "conceptual"),
            ("CAP theorem implications for distributed systems", "conceptual"),
            # Complex multi-hop queries
            (
                "How to implement OAuth2 flow in React with Node.js backend and PostgreSQL database?",
                "complex",
            ),
            (
                "Best practices for scaling microservices with Kubernetes, monitoring, and CI/CD pipeline?",
                "complex",
            ),
            (
                "Optimize database performance with indexing, caching, and query optimization techniques?",
                "complex",
            ),
        ]

        queries = []
        for query_text, _query_category in test_data:
            request = AdvancedHybridSearchRequest(
                query=query_text,
                collection_name="benchmark_collection",
                limit=10,
                search_params=SearchParams(),
                fusion_config=FusionConfig(),
                enable_query_classification=True,
                enable_model_selection=True,
                enable_adaptive_fusion=True,
                enable_splade=True,
                user_id=f"benchmark_user_{hash(query_text) % 1000}",
                session_id=f"benchmark_session_{int(time.time())}",
            )
            queries.append(request)

        # Repeat queries to reach target count
        target_count = self.benchmark_config.test_queries_per_type * 6  # 6 categories
        while len(queries) < target_count:
            queries.extend(queries[: min(len(queries), target_count - len(queries))])

        return queries[:target_count]

    async def run_comprehensive_benchmark(self) -> BenchmarkResults:
        """Run complete benchmark suite and return results.

        Returns:
            Comprehensive benchmark results with performance metrics and recommendations
        """
        start_time = time.time()
        logger.info(f"Starting comprehensive benchmark: {self.benchmark_config.name}")

        results = BenchmarkResults(
            benchmark_name=self.benchmark_config.name,
            duration_seconds=0.0,
            meets_targets=False,
        )

        try:
            # Initialize search service
            await self.search_service.initialize()

            # 1. Component-level benchmarks
            if self.benchmark_config.enable_component_benchmarks:
                logger.info("Running component benchmarks...")
                component_results = await self._run_component_benchmarks()
                results.component_results = component_results

            # 2. Load testing
            if self.benchmark_config.enable_load_testing:
                logger.info("Running load tests...")
                load_results = await self._run_load_tests()
                results.load_test_results = load_results
                results.latency_metrics.update(load_results.get("latency_metrics", {}))
                results.throughput_metrics.update(
                    load_results.get("throughput_metrics", {})
                )

            # 3. Performance profiling
            if self.benchmark_config.enable_profiling:
                logger.info("Running performance profiling...")
                profiling_results = await self._run_profiling()
                results.profiling_results = profiling_results
                results.resource_metrics.update(
                    profiling_results.get("resource_metrics", {})
                )

            # 4. Accuracy assessment
            logger.info("Assessing ML accuracy...")
            accuracy_results = await self._assess_accuracy()
            results.accuracy_metrics = accuracy_results

            # 5. Optimization analysis
            if self.benchmark_config.enable_optimization_analysis:
                logger.info("Analyzing optimization opportunities...")
                optimization_analysis = self._analyze_optimization_opportunities(
                    results
                )
                results.optimization_recommendations = optimization_analysis[
                    "recommendations"
                ]
                results.performance_bottlenecks = optimization_analysis["bottlenecks"]

            # 6. Target compliance check
            compliance_check = self._check_target_compliance(results)
            results.meets_targets = compliance_check["meets_targets"]
            results.failed_targets = compliance_check["failed_targets"]

            # Calculate total duration
            results.duration_seconds = time.time() - start_time

            logger.info(
                f"Benchmark completed in {results.duration_seconds:.2f} seconds"
            )
            logger.info(f"Meets targets: {results.meets_targets}")

            return results

        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            results.duration_seconds = time.time() - start_time
            results.failed_targets.append(f"Benchmark execution failed: {e!s}")
            return results

    async def _run_component_benchmarks(self) -> dict[str, Any]:
        """Run component-level benchmarks."""
        return await self.component_benchmarks.run_all_component_benchmarks(
            self.search_service,
            self.test_queries[:50],  # Use subset for component tests
        )

    async def _run_load_tests(self) -> dict[str, Any]:
        """Run load testing scenarios."""
        load_results = {}

        for concurrent_users in self.benchmark_config.concurrent_users:
            logger.info(f"Running load test with {concurrent_users} concurrent users")

            load_config = LoadTestConfig(
                concurrent_users=concurrent_users,
                total_requests=concurrent_users * 20,  # 20 requests per user
                duration_seconds=min(60, self.benchmark_config.test_duration_seconds),
                ramp_up_seconds=10,
            )

            result = await self.load_test_runner.run_load_test(
                self.search_service, self.test_queries, load_config
            )

            load_results[f"{concurrent_users}_users"] = result

        return load_results

    async def _run_profiling(self) -> dict[str, Any]:
        """Run detailed performance profiling."""
        return await self.performance_profiler.profile_search_service(
            self.search_service,
            self.test_queries[:20],  # Use subset for profiling
        )

    async def _assess_accuracy(self) -> dict[str, float]:
        """Assess ML component accuracy."""
        accuracy_metrics = {}

        # Query classification accuracy
        classification_accuracy = await self._measure_classification_accuracy()
        accuracy_metrics["query_classification_accuracy"] = classification_accuracy

        # Model selection effectiveness
        model_selection_accuracy = await self._measure_model_selection_accuracy()
        accuracy_metrics["model_selection_accuracy"] = model_selection_accuracy

        # Adaptive fusion effectiveness
        fusion_effectiveness = await self._measure_fusion_effectiveness()
        accuracy_metrics["adaptive_fusion_effectiveness"] = fusion_effectiveness

        return accuracy_metrics

    async def _measure_classification_accuracy(self) -> float:
        """Measure query classification accuracy with ground truth."""
        correct_predictions = 0
        total_predictions = 0

        # Ground truth data (simplified for benchmark)
        ground_truth = {
            "How to implement async functions in Python?": "code",
            "FastAPI dependency injection tutorial": "documentation",
            "GET /api/users endpoint parameters": "api_reference",
            "TypeError: 'NoneType' object is not iterable": "troubleshooting",
            "What is microservices architecture?": "conceptual",
        }

        for query_text, expected_type in ground_truth.items():
            try:
                classification = (
                    await self.search_service.query_classifier.classify_query(
                        query_text
                    )
                )
                predicted_type = classification.query_type.value

                if predicted_type == expected_type:
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                logger.warning(f"Classification failed for query '{query_text}': {e}")
                total_predictions += 1

        return correct_predictions / max(total_predictions, 1)

    async def _measure_model_selection_accuracy(self) -> float:
        """Measure model selection accuracy and appropriateness."""
        # Simplified accuracy measurement
        appropriate_selections = 0
        total_selections = 0

        test_cases = [
            ("Python code example", "code"),
            ("API documentation", "documentation"),
            ("Complex algorithm explanation", "conceptual"),
        ]

        for query_text, query_domain in test_cases:
            try:
                classification = (
                    await self.search_service.query_classifier.classify_query(
                        query_text
                    )
                )
                selection = (
                    await self.search_service.model_selector.select_optimal_model(
                        classification
                    )
                )

                # Check if selection is appropriate for domain
                model_info = self.search_service.model_selector.model_registry[
                    selection.primary_model
                ]
                specializations = model_info.get("specializations", [])

                if query_domain in specializations or "general" in specializations:
                    appropriate_selections += 1
                total_selections += 1

            except Exception as e:
                logger.warning(f"Model selection failed for query '{query_text}': {e}")
                total_selections += 1

        return appropriate_selections / max(total_selections, 1)

    async def _measure_fusion_effectiveness(self) -> float:
        """Measure adaptive fusion effectiveness."""
        # Simplified effectiveness measurement
        effective_fusions = 0
        total_fusions = 0

        for query in self.test_queries[:10]:
            try:
                classification = (
                    await self.search_service.query_classifier.classify_query(
                        query.query
                    )
                )
                weights = await self.search_service.adaptive_fusion_tuner.compute_adaptive_weights(
                    classification, f"benchmark_{total_fusions}"
                )

                # Check if weights are reasonable (not extreme)
                if 0.2 <= weights.dense_weight <= 0.8 and weights.confidence > 0.5:
                    effective_fusions += 1
                total_fusions += 1

            except Exception as e:
                logger.warning(f"Fusion tuning failed for query '{query.query}': {e}")
                total_fusions += 1

        return effective_fusions / max(total_fusions, 1)

    def _analyze_optimization_opportunities(
        self, results: BenchmarkResults
    ) -> dict[str, list[str]]:
        """Analyze results and provide optimization recommendations."""
        recommendations = []
        bottlenecks = []

        # Latency analysis
        p95_latency = results.latency_metrics.get("p95_ms", 0)
        if p95_latency > self.benchmark_config.target_latency_p95_ms:
            bottlenecks.append(
                f"High p95 latency: {p95_latency:.1f}ms > {self.benchmark_config.target_latency_p95_ms}ms"
            )
            recommendations.append("Consider caching frequently accessed embeddings")
            recommendations.append(
                "Optimize model selection logic for faster decisions"
            )

        # Throughput analysis
        max_qps = (
            max(results.throughput_metrics.values())
            if results.throughput_metrics
            else 0
        )
        if max_qps < self.benchmark_config.target_throughput_qps:
            bottlenecks.append(
                f"Low throughput: {max_qps:.1f} QPS < {self.benchmark_config.target_throughput_qps} QPS"
            )
            recommendations.append(
                "Implement connection pooling for database operations"
            )
            recommendations.append(
                "Add async processing for non-critical ML components"
            )

        # Memory analysis
        peak_memory = results.resource_metrics.get("peak_memory_mb", 0)
        if peak_memory > self.benchmark_config.target_memory_mb:
            bottlenecks.append(
                f"High memory usage: {peak_memory:.1f}MB > {self.benchmark_config.target_memory_mb}MB"
            )
            recommendations.append("Implement LRU cache for SPLADE embeddings")
            recommendations.append("Optimize model registry memory footprint")

        # Accuracy analysis
        avg_accuracy = sum(results.accuracy_metrics.values()) / max(
            len(results.accuracy_metrics), 1
        )
        if avg_accuracy < self.benchmark_config.target_accuracy:
            bottlenecks.append(
                f"Low ML accuracy: {avg_accuracy:.3f} < {self.benchmark_config.target_accuracy}"
            )
            recommendations.append(
                "Retrain query classification with more diverse data"
            )
            recommendations.append("Fine-tune adaptive fusion parameters")

        return {"recommendations": recommendations, "bottlenecks": bottlenecks}

    def _check_target_compliance(self, results: BenchmarkResults) -> dict[str, Any]:
        """Check if benchmark results meet performance targets."""
        failed_targets = []

        # Check latency target
        p95_latency = results.latency_metrics.get("p95_ms", float("inf"))
        if p95_latency > self.benchmark_config.target_latency_p95_ms:
            failed_targets.append(
                f"Latency target: {p95_latency:.1f}ms > {self.benchmark_config.target_latency_p95_ms}ms"
            )

        # Check throughput target
        max_qps = (
            max(results.throughput_metrics.values())
            if results.throughput_metrics
            else 0
        )
        if max_qps < self.benchmark_config.target_throughput_qps:
            failed_targets.append(
                f"Throughput target: {max_qps:.1f} < {self.benchmark_config.target_throughput_qps} QPS"
            )

        # Check memory target
        peak_memory = results.resource_metrics.get("peak_memory_mb", float("inf"))
        if peak_memory > self.benchmark_config.target_memory_mb:
            failed_targets.append(
                f"Memory target: {peak_memory:.1f}MB > {self.benchmark_config.target_memory_mb}MB"
            )

        # Check accuracy target
        avg_accuracy = sum(results.accuracy_metrics.values()) / max(
            len(results.accuracy_metrics), 1
        )
        if avg_accuracy < self.benchmark_config.target_accuracy:
            failed_targets.append(
                f"Accuracy target: {avg_accuracy:.3f} < {self.benchmark_config.target_accuracy}"
            )

        return {
            "meets_targets": len(failed_targets) == 0,
            "failed_targets": failed_targets,
        }

    async def save_results(self, results: BenchmarkResults, output_dir: Path) -> None:
        """Save benchmark results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_file = output_dir / f"benchmark_results_{int(time.time())}.json"
        with open(json_file, "w") as f:
            json.dump(results.model_dump(), f, indent=2, default=str)

        # Generate HTML report
        html_report = await self.benchmark_reporter.generate_html_report(results)
        html_file = output_dir / f"benchmark_report_{int(time.time())}.html"
        with open(html_file, "w") as f:
            f.write(html_report)

        logger.info(f"Results saved to {output_dir}")
        logger.info(f"JSON: {json_file}")
        logger.info(f"HTML: {html_file}")
