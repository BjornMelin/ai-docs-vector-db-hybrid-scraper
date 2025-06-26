"""Tests for the performance optimization showcase system.

This test suite validates the sophisticated performance optimization components
including adaptive caching, async optimization, and comprehensive benchmarking.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.services.cache.adaptive_cache import (
    AccessPattern,
    AdaptiveCacheOptimizer,
    CacheStrategy,
)
from src.services.performance.async_optimization import (
    AdaptiveConcurrencyLimiter,
    AsyncPerformanceOptimizer,
    IntelligentTaskScheduler,
    TaskPriority,
)
from src.services.performance.benchmark_suite import (
    BenchmarkMetrics,
    LoadTestConfig,
    PerformanceBenchmarkSuite,
)
from src.services.performance.optimization_showcase import (
    OptimizationResults,
    PerformanceOptimizationShowcase,
)


class TestAdaptiveConcurrencyLimiter:
    """Test adaptive concurrency limiting functionality."""

    @pytest.fixture
    def limiter(self):
        """Create a concurrency limiter for testing."""
        return AdaptiveConcurrencyLimiter(initial_limit=5, min_limit=1, max_limit=20)

    @pytest.mark.asyncio
    async def test_acquire_release_basic(self, limiter):
        """Test basic acquire and release functionality."""
        # Test acquire
        await limiter.acquire()
        assert limiter.metrics.active_tasks == 1

        # Test release
        await limiter.release(execution_time=0.1, success=True)
        assert limiter.metrics.active_tasks == 0
        assert limiter.metrics.completed_tasks == 1

    @pytest.mark.asyncio
    async def test_adaptive_scaling_up(self, limiter):
        """Test concurrency limit scaling up under good conditions."""
        # Simulate good performance conditions
        limiter.metrics.resource_utilization = 50.0  # Low resource usage
        limiter.consecutive_good_measurements = 15  # Above stability threshold

        # Mock system metrics to return good values
        with patch.object(
            limiter,
            "_get_system_metrics",
            return_value={
                "cpu_percent": 50.0,
                "memory_used_mb": 1000.0,
                "memory_percent": 50.0,
            },
        ):
            original_limit = limiter.current_limit

            # Simulate task execution that should trigger scaling
            await limiter.acquire()
            await limiter.release(execution_time=0.05, success=True)

            # Should increase limit under good conditions
            assert limiter.current_limit >= original_limit

    @pytest.mark.asyncio
    async def test_adaptive_scaling_down(self, limiter):
        """Test concurrency limit scaling down under poor conditions."""
        # Set high resource utilization
        limiter.metrics.resource_utilization = 95.0

        with patch.object(
            limiter,
            "_get_system_metrics",
            return_value={
                "cpu_percent": 95.0,
                "memory_used_mb": 8000.0,
                "memory_percent": 95.0,
            },
        ):
            original_limit = limiter.current_limit

            # Simulate task execution with high resource usage
            await limiter.acquire()
            await limiter.release(execution_time=1.0, success=True)

            # Should decrease limit under poor conditions
            assert limiter.current_limit <= original_limit

    def test_metrics_tracking(self, limiter):
        """Test metrics tracking functionality."""
        metrics = limiter.get_metrics()

        assert "current_limit" in metrics
        assert "active_tasks" in metrics
        assert "completed_tasks" in metrics
        assert "failed_tasks" in metrics
        assert metrics["current_limit"] == 5  # Initial limit


class TestIntelligentTaskScheduler:
    """Test intelligent task scheduling functionality."""

    @pytest.fixture
    def scheduler(self):
        """Create a task scheduler for testing."""
        limiter = AdaptiveConcurrencyLimiter(initial_limit=5)
        return IntelligentTaskScheduler(
            concurrency_limiter=limiter, enable_batching=True, batch_size=3
        )

    @pytest.mark.asyncio
    async def test_task_scheduling_basic(self, scheduler):
        """Test basic task scheduling functionality."""

        async def test_task():
            await asyncio.sleep(0.01)
            return "test_result"

        # Schedule and execute task
        result = await scheduler.schedule_task(test_task(), priority=TaskPriority.HIGH)

        assert result == "test_result"

        # Check metrics
        metrics = scheduler.get_metrics()
        assert metrics["scheduler_metrics"]["tasks_completed"][TaskPriority.HIGH] == 1

    @pytest.mark.asyncio
    async def test_priority_ordering(self, scheduler):
        """Test that high priority tasks are executed first."""
        results = []

        async def priority_task(priority_name: str):
            await asyncio.sleep(0.01)
            results.append(priority_name)
            return priority_name

        # Start scheduler
        await scheduler.start()

        # Schedule tasks in reverse priority order
        tasks = [
            scheduler.schedule_task(priority_task("low"), TaskPriority.LOW),
            scheduler.schedule_task(priority_task("high"), TaskPriority.HIGH),
            scheduler.schedule_task(priority_task("normal"), TaskPriority.NORMAL),
        ]

        await asyncio.gather(*tasks)
        await scheduler.stop()

        # High priority should execute first
        assert results[0] == "high"

    @pytest.mark.asyncio
    async def test_batch_processing(self, scheduler):
        """Test batch processing functionality."""

        async def batch_task(task_id: int):
            await asyncio.sleep(0.001)
            return f"task_{task_id}"

        await scheduler.start()

        # Schedule multiple tasks for batching
        tasks = [
            scheduler.schedule_task(batch_task(i), TaskPriority.NORMAL)
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)
        await scheduler.stop()

        assert len(results) == 5
        assert all(f"task_{i}" in results for i in range(5))

        # Check that batching occurred
        metrics = scheduler.get_metrics()
        assert (
            metrics["scheduler_metrics"]["batches_processed"][TaskPriority.NORMAL] > 0
        )


class TestAdaptiveCacheOptimizer:
    """Test adaptive cache optimization functionality."""

    @pytest.fixture
    def cache_manager_mock(self):
        """Create a mock cache manager."""
        cache_manager = MagicMock()
        cache_manager.get = AsyncMock(return_value=None)
        cache_manager.set = AsyncMock(return_value=True)
        cache_manager.delete = AsyncMock(return_value=True)
        return cache_manager

    @pytest.fixture
    def optimizer(self, cache_manager_mock):
        """Create a cache optimizer for testing."""
        return AdaptiveCacheOptimizer(
            cache_manager=cache_manager_mock,
            strategy=CacheStrategy(prefetch_threshold=0.7, min_access_count=3),
            enable_ml_predictions=True,
        )

    @pytest.mark.asyncio
    async def test_access_pattern_tracking(self, optimizer):
        """Test access pattern tracking functionality."""
        from src.config.enums import CacheType

        # Track multiple accesses to the same key
        key = "test_key"
        for i in range(5):
            await optimizer.track_access(
                key=key,
                cache_type=CacheType.CRAWL,
                hit=(i > 0),  # First access is miss, rest are hits
                response_time_ms=10.0 + i,
            )
            await asyncio.sleep(0.01)  # Small delay between accesses

        # Check that pattern was recorded
        assert key in optimizer.access_patterns
        pattern = optimizer.access_patterns[key]
        assert len(pattern.access_times) == 5
        assert pattern.access_frequency > 0

    @pytest.mark.asyncio
    async def test_optimization_hint_generation(self, optimizer):
        """Test optimization hint generation."""
        from src.config.enums import CacheType

        # Create access pattern that should generate prefetch hint
        key = "hot_key"
        current_time = time.time()

        # Simulate frequent recent accesses
        pattern = AccessPattern(key=key)
        for i in range(10):
            pattern.access_times.append(current_time - (10 - i) * 0.1)
        pattern.access_frequency = 5.0  # High frequency
        pattern.avg_interval = 0.1  # Short interval
        pattern.last_access = current_time - 0.05  # Recent access

        optimizer.access_patterns[key] = pattern

        # Generate hints
        hints = await optimizer.get_optimization_hints()

        # Should generate prefetch hint for hot key
        prefetch_hints = [h for h in hints if h.operation == "prefetch"]
        assert len(prefetch_hints) > 0
        assert any(h.key == key for h in prefetch_hints)

    @pytest.mark.asyncio
    async def test_ml_prediction(self, optimizer):
        """Test ML-based access prediction."""
        # Create pattern for prediction
        pattern = AccessPattern(key="test_key")
        pattern.access_frequency = 1.0
        pattern.last_access = time.time() - 0.5
        pattern.avg_interval = 1.0
        pattern.trend_slope = 0.1  # Positive trend
        pattern.prediction_confidence = 0.8

        # Test prediction
        current_time = time.time()
        probability = await optimizer._predict_next_access(pattern, current_time)

        assert 0.0 <= probability <= 1.0
        assert probability > 0.5  # Should predict likely access

    @pytest.mark.asyncio
    async def test_optimization_performance(self, optimizer):
        """Test cache optimization execution."""
        # Add some patterns
        patterns = {f"key_{i}": AccessPattern(key=f"key_{i}") for i in range(5)}
        optimizer.access_patterns.update(patterns)

        # Run optimization
        results = await optimizer.optimize_cache_performance()

        assert "hints_generated" in results
        assert "hints_applied" in results
        assert "duration_ms" in results
        assert results["duration_ms"] > 0


class TestPerformanceBenchmarkSuite:
    """Test performance benchmarking functionality."""

    @pytest.fixture
    def benchmark_suite(self):
        """Create a benchmark suite for testing."""
        return PerformanceBenchmarkSuite()

    @pytest.mark.asyncio
    async def test_single_measurement(self, benchmark_suite):
        """Test single performance measurement."""

        async def test_function():
            await asyncio.sleep(0.01)
            return "test_result"

        result = await benchmark_suite._execute_single_measurement(test_function)

        assert result.success is True
        assert result.duration_ms >= 10  # At least 10ms due to sleep
        assert result.memory_mb > 0
        assert result.cpu_percent >= 0

    @pytest.mark.asyncio
    async def test_warmup_phase(self, benchmark_suite):
        """Test warmup phase execution."""

        async def test_function():
            await asyncio.sleep(0.001)
            return "warmup_result"

        results = await benchmark_suite._run_warmup_phase(test_function)

        assert "total_iterations" in results
        assert "successful_iterations" in results
        assert "avg_latency_ms" in results
        assert "success_rate" in results
        assert results["total_iterations"] == 10
        assert results["success_rate"] > 0.9  # Should have high success rate

    @pytest.mark.asyncio
    async def test_baseline_measurement(self, benchmark_suite):
        """Test baseline performance measurement."""

        async def test_function():
            await asyncio.sleep(0.002)
            return "baseline_result"

        results = await benchmark_suite._run_baseline_phase(test_function)

        assert "total_requests" in results
        assert "successful_requests" in results
        assert "avg_latency_ms" in results
        assert "throughput_rps" in results
        assert results["total_requests"] == 50
        assert results["throughput_rps"] > 0

    @pytest.mark.asyncio
    async def test_metrics_calculation(self, benchmark_suite):
        """Test final metrics calculation."""
        # Add some measurement history
        from src.services.performance.benchmark_suite import BenchmarkResult

        measurements = [
            BenchmarkResult(
                duration_ms=10.0 + i,
                success=True,
                memory_mb=100.0 + i,
                cpu_percent=50.0 + i,
            )
            for i in range(10)
        ]

        benchmark_suite.measurement_history.extend(measurements)

        metrics = benchmark_suite._calculate_final_metrics()

        assert metrics.total_requests == 10
        assert metrics.successful_requests == 10
        assert metrics.failed_requests == 0
        assert metrics.avg_latency_ms > 10
        assert metrics.p95_latency_ms > metrics.avg_latency_ms
        assert 0 <= metrics.stability_score <= 1
        assert 0 <= metrics.efficiency_score <= 1
        assert metrics.reliability_score == 1.0  # All successful


class TestPerformanceOptimizationShowcase:
    """Test the complete performance optimization showcase."""

    @pytest.fixture
    def showcase(self):
        """Create a showcase instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield PerformanceOptimizationShowcase(output_dir=Path(temp_dir))

    @pytest.mark.asyncio
    async def test_showcase_initialization(self, showcase):
        """Test showcase system initialization."""
        # Mock the heavy components
        with (
            patch("src.services.cache.manager.CacheManager") as mock_cache,
            patch(
                "src.services.cache.adaptive_cache.EnhancedCacheManager"
            ) as mock_enhanced_cache,
            patch(
                "src.services.performance.async_optimization.initialize_async_optimizer"
            ) as mock_async_init,
            patch(
                "src.services.performance.benchmark_suite.PerformanceBenchmarkSuite"
            ) as mock_benchmark,
        ):
            # Setup mocks
            mock_async_init.return_value = MagicMock()

            await showcase._initialize_showcase_systems()

            # Verify components were initialized
            assert showcase.baseline_cache_manager is not None
            assert showcase.optimized_cache_manager is not None
            assert showcase.async_optimizer is not None
            assert showcase.benchmark_suite is not None

    def test_technical_complexity_scoring(self, showcase):
        """Test technical complexity score calculation."""
        score = showcase._calculate_technical_complexity_score()

        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high complexity

    def test_business_impact_scoring(self, showcase):
        """Test business impact score calculation."""
        business_impact = {
            "cost_savings": {
                "infrastructure_cost_reduction_percent": 30.0,
                "operational_efficiency_gain_percent": 20.0,
            },
            "scalability": {
                "capacity_increase_percent": 50.0,
            },
        }

        score = showcase._calculate_business_impact_score(business_impact)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should show good business impact

    def test_innovation_scoring(self, showcase):
        """Test innovation score calculation."""
        score = showcase._calculate_innovation_score()

        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be highly innovative

    @pytest.mark.asyncio
    async def test_result_saving(self, showcase):
        """Test result saving functionality."""
        # Create mock results
        results = OptimizationResults(
            showcase_name="test_showcase",
            start_time=time.time(),
            end_time=time.time(),
            duration_seconds=60.0,
            baseline_performance={},
            optimized_performance={},
            performance_improvements={},
            optimizations_applied=["test_optimization"],
            cache_optimization_results={},
            async_optimization_results={},
            statistical_significance={},
            confidence_intervals={},
            cost_savings_estimate={},
            scalability_improvements={},
            bottleneck_analysis={},
            optimization_recommendations=[],
            technical_complexity_score=0.9,
            business_impact_score=0.8,
            innovation_score=0.85,
        )

        await showcase._save_showcase_results(results)

        # Check that files were created
        results_file = showcase.output_dir / "performance_optimization_results.json"
        summary_file = showcase.output_dir / "performance_summary.json"

        assert results_file.exists()
        assert summary_file.exists()

        # Validate JSON content
        with open(results_file) as f:
            saved_results = json.load(f)
            assert saved_results["showcase_name"] == "test_showcase"
            assert saved_results["technical_complexity_score"] == 0.9

        with open(summary_file) as f:
            summary = json.load(f)
            assert "showcase_name" in summary
            assert "technical_complexity_score" in summary

    @pytest.mark.asyncio
    async def test_documentation_generation(self, showcase):
        """Test portfolio documentation generation."""
        # Create mock results
        results = OptimizationResults(
            showcase_name="test_showcase",
            start_time=time.time(),
            end_time=time.time(),
            duration_seconds=60.0,
            baseline_performance={},
            optimized_performance={},
            performance_improvements={
                "cache_performance": {
                    "avg_latency_ms": {
                        "baseline_value": 100.0,
                        "optimized_value": 70.0,
                        "improvement_percent": 30.0,
                    }
                }
            },
            optimizations_applied=["ML-driven caching", "Async optimization"],
            cache_optimization_results={},
            async_optimization_results={},
            statistical_significance={},
            confidence_intervals={},
            cost_savings_estimate={"infrastructure_cost_reduction_percent": 25.0},
            scalability_improvements={"capacity_increase_percent": 40.0},
            bottleneck_analysis={},
            optimization_recommendations=["Expand caching", "Monitor performance"],
            technical_complexity_score=0.9,
            business_impact_score=0.8,
            innovation_score=0.85,
        )

        await showcase._generate_portfolio_documentation(results)

        # Check that documentation was created
        doc_file = showcase.output_dir / "PERFORMANCE_OPTIMIZATION_SHOWCASE.md"
        assert doc_file.exists()

        # Validate content
        with open(doc_file) as f:
            content = f.read()
            assert "Performance Optimization Showcase" in content
            assert "ML-driven cache optimization" in content
            assert "30.0% improvement" in content
            assert "Technical Complexity Score: 0.90" in content


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for the complete performance system."""

    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self):
        """Test end-to-end performance optimization flow."""
        # This is a simplified integration test
        from src.services.performance.async_optimization import (
            initialize_async_optimizer,
        )

        # Initialize optimizer
        optimizer = await initialize_async_optimizer(
            initial_concurrency=3,
            enable_adaptive_limiting=True,
            enable_intelligent_scheduling=True,
        )

        # Test workload
        async def test_workload():
            await asyncio.sleep(0.01)
            return "workload_result"

        # Execute multiple tasks
        tasks = [
            optimizer.execute_optimized(test_workload(), TaskPriority.NORMAL)
            for _ in range(10)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time

        # Validate results
        assert len(results) == 10
        assert all(r == "workload_result" for r in results)
        assert execution_time < 1.0  # Should complete reasonably quickly

        # Check performance metrics
        performance_report = optimizer.get_performance_report()
        assert "concurrency_metrics" in performance_report
        assert performance_report["optimization_enabled"]["adaptive_limiting"] is True

    @pytest.mark.asyncio
    async def test_cache_optimization_integration(self):
        """Test cache optimization integration."""
        from src.config.enums import CacheType
        from src.services.cache.adaptive_cache import EnhancedCacheManager

        # Create enhanced cache manager
        cache_manager = EnhancedCacheManager(
            enable_local_cache=True,
            enable_distributed_cache=False,  # Skip Redis for testing
            enable_adaptive_optimization=True,
        )

        try:
            # Simulate cache access patterns
            for i in range(20):
                key = f"integration_key_{i % 5}"  # Create hot keys

                # Try to get from cache
                result = await cache_manager.get(key, CacheType.CRAWL)

                if result is None:
                    # Cache miss - store data
                    data = {"integration_test": True, "iteration": i}
                    await cache_manager.set(key, data, CacheType.CRAWL)

                await asyncio.sleep(0.001)

            # Get cache statistics
            stats = await cache_manager.get_stats()

            # Validate cache functionality
            assert "local" in stats
            assert stats["local"]["size"] > 0

            # Test optimization analytics if available
            if hasattr(cache_manager, "get_optimization_analytics"):
                analytics = await cache_manager.get_optimization_analytics()
                assert "global_stats" in analytics
                assert analytics["global_stats"]["total_accesses"] > 0

        finally:
            await cache_manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
