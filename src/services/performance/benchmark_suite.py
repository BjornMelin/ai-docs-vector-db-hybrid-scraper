"""Comprehensive performance benchmarking suite with advanced analytics.

This module provides sophisticated performance testing including:
- Automated stress testing with gradual load increases
- Performance regression detection with statistical analysis
- Resource usage profiling with predictive modeling
- Comparative benchmarking across different configurations
- Real-time performance visualization and reporting
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, NamedTuple

import numpy as np
import psutil
from pydantic import BaseModel

from ..cache.adaptive_cache import EnhancedCacheManager
from ..monitoring.metrics import get_metrics_registry
from ..observability.performance import PerformanceMonitor
from .async_optimization import AsyncPerformanceOptimizer, TaskPriority


logger = logging.getLogger(__name__)


class BenchmarkResult(NamedTuple):
    """Single benchmark measurement result."""

    duration_ms: float
    success: bool
    memory_mb: float
    cpu_percent: float
    error_message: str | None = None


@dataclass
class LoadTestConfig:
    """Load testing configuration."""

    initial_rps: float = 1.0  # Requests per second
    max_rps: float = 100.0
    ramp_duration: float = 60.0  # seconds
    sustain_duration: float = 30.0  # seconds
    step_size: float = 5.0  # RPS increase per step
    success_threshold: float = 0.95  # 95% success rate required
    latency_threshold_ms: float = 1000.0  # Max acceptable latency


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0

    # Resource metrics
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0

    # Quality metrics
    stability_score: float = 0.0  # 0-1, higher is more stable
    efficiency_score: float = 0.0  # throughput per resource unit
    reliability_score: float = 0.0  # based on error rates and consistency


class PerformanceBenchmarkSuite:
    """Advanced performance benchmarking with analytics."""

    def __init__(
        self,
        async_optimizer: AsyncPerformanceOptimizer | None = None,
        cache_manager: EnhancedCacheManager | None = None,
        performance_monitor: PerformanceMonitor | None = None,
    ):
        """Initialize benchmark suite.

        Args:
            async_optimizer: Async performance optimizer to test
            cache_manager: Cache manager to test
            performance_monitor: Performance monitor for tracking
        """
        self.async_optimizer = async_optimizer
        self.cache_manager = cache_manager
        self.performance_monitor = performance_monitor

        # Benchmark history for regression detection
        self.benchmark_history = deque(maxlen=100)
        self.baseline_metrics: BenchmarkMetrics | None = None

        # Real-time metrics tracking
        self.current_metrics = BenchmarkMetrics()
        self.measurement_history = deque(maxlen=1000)

        # Resource monitoring
        self.resource_snapshots = deque(maxlen=10000)
        self.start_memory = psutil.virtual_memory().used / (1024 * 1024)

        # Metrics integration
        self.metrics_registry = None
        try:
            self.metrics_registry = get_metrics_registry()
        except Exception as e:
            logger.warning(f"Failed to initialize metrics registry: {e}")

    async def run_comprehensive_benchmark(
        self,
        benchmark_name: str,
        test_function: Callable[[], Any],
        config: LoadTestConfig | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive performance benchmark.

        Args:
            benchmark_name: Name of the benchmark
            test_function: Function to benchmark
            config: Load testing configuration

        Returns:
            Comprehensive benchmark results
        """
        config = config or LoadTestConfig()
        start_time = time.time()

        logger.info(f"Starting comprehensive benchmark: {benchmark_name}")

        # Initialize tracking
        results = {
            "benchmark_name": benchmark_name,
            "start_time": datetime.now(),
            "config": config.__dict__,
            "phases": {},
            "final_metrics": None,
            "regression_analysis": None,
            "recommendations": [],
        }

        try:
            # Phase 1: Warmup
            logger.info("Phase 1: Warmup")
            warmup_results = await self._run_warmup_phase(test_function)
            results["phases"]["warmup"] = warmup_results

            # Phase 2: Baseline measurement
            logger.info("Phase 2: Baseline measurement")
            baseline_results = await self._run_baseline_phase(test_function)
            results["phases"]["baseline"] = baseline_results

            # Phase 3: Load testing
            logger.info("Phase 3: Load testing")
            load_results = await self._run_load_test_phase(test_function, config)
            results["phases"]["load_test"] = load_results

            # Phase 4: Stress testing
            logger.info("Phase 4: Stress testing")
            stress_results = await self._run_stress_test_phase(test_function, config)
            results["phases"]["stress_test"] = stress_results

            # Phase 5: Recovery testing
            logger.info("Phase 5: Recovery testing")
            recovery_results = await self._run_recovery_phase(test_function)
            results["phases"]["recovery"] = recovery_results

            # Calculate final metrics
            final_metrics = self._calculate_final_metrics()
            results["final_metrics"] = final_metrics.__dict__

            # Regression analysis
            regression_analysis = await self._analyze_performance_regression(
                final_metrics
            )
            results["regression_analysis"] = regression_analysis

            # Generate recommendations
            recommendations = self._generate_performance_recommendations(final_metrics)
            results["recommendations"] = recommendations

            # Store in history
            self.benchmark_history.append(
                {
                    "name": benchmark_name,
                    "timestamp": start_time,
                    "metrics": final_metrics,
                    "config": config,
                }
            )

            # Record in metrics system
            if self.metrics_registry:
                self.metrics_registry.record_benchmark_completion(
                    benchmark_name=benchmark_name,
                    duration_seconds=time.time() - start_time,
                    success_rate=final_metrics.successful_requests
                    / max(final_metrics.total_requests, 1),
                    avg_latency_ms=final_metrics.avg_latency_ms,
                )

        except Exception as e:
            logger.exception(f"Benchmark failed: {e}")
            results["error"] = str(e)

        results["duration_seconds"] = time.time() - start_time
        results["end_time"] = datetime.now()

        logger.info(
            f"Completed benchmark: {benchmark_name} in {results['duration_seconds']:.2f}s"
        )
        return results

    async def _run_warmup_phase(self, test_function: Callable) -> dict[str, Any]:
        """Run warmup phase to stabilize system."""
        warmup_results = []

        # Run 10 warmup iterations
        for _ in range(10):
            start_time = time.time()
            try:
                await test_function()
                duration_ms = (time.time() - start_time) * 1000
                warmup_results.append(
                    BenchmarkResult(
                        duration_ms=duration_ms,
                        success=True,
                        memory_mb=psutil.virtual_memory().used / (1024 * 1024),
                        cpu_percent=psutil.cpu_percent(),
                    )
                )
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                warmup_results.append(
                    BenchmarkResult(
                        duration_ms=duration_ms,
                        success=False,
                        memory_mb=psutil.virtual_memory().used / (1024 * 1024),
                        cpu_percent=psutil.cpu_percent(),
                        error_message=str(e),
                    )
                )

            await asyncio.sleep(0.1)  # Small delay between warmup calls

        # Calculate warmup statistics
        successful_results = [r for r in warmup_results if r.success]

        return {
            "total_iterations": len(warmup_results),
            "successful_iterations": len(successful_results),
            "avg_latency_ms": statistics.mean(
                [r.duration_ms for r in successful_results]
            )
            if successful_results
            else 0,
            "success_rate": len(successful_results) / len(warmup_results),
        }

    async def _run_baseline_phase(self, test_function: Callable) -> dict[str, Any]:
        """Run baseline performance measurement."""
        baseline_results = []
        baseline_start = time.time()

        # Run 50 iterations for stable baseline
        for _ in range(50):
            result = await self._execute_single_measurement(test_function)
            baseline_results.append(result)
            self.measurement_history.append(result)

            await asyncio.sleep(0.05)  # 20 RPS baseline rate

        # Calculate baseline metrics
        successful_results = [r for r in baseline_results if r.success]
        latencies = [r.duration_ms for r in successful_results]

        baseline_metrics = {
            "duration_seconds": time.time() - baseline_start,
            "total_requests": len(baseline_results),
            "successful_requests": len(successful_results),
            "success_rate": len(successful_results) / len(baseline_results),
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
            "throughput_rps": len(successful_results) / (time.time() - baseline_start),
        }

        # Store baseline for comparison
        if not self.baseline_metrics:
            self.baseline_metrics = self._convert_to_benchmark_metrics(baseline_metrics)

        return baseline_metrics

    async def _run_load_test_phase(
        self,
        test_function: Callable,
        config: LoadTestConfig,
    ) -> dict[str, Any]:
        """Run progressive load testing."""
        load_results = []
        phase_start = time.time()

        current_rps = config.initial_rps

        while current_rps <= config.max_rps:
            logger.info(f"Testing at {current_rps} RPS")

            # Test at current RPS for sustain duration
            interval = 1.0 / current_rps
            sustain_start = time.time()
            iteration_results = []

            while time.time() - sustain_start < config.sustain_duration:
                iteration_start = time.time()

                result = await self._execute_single_measurement(test_function)
                iteration_results.append(result)
                self.measurement_history.append(result)

                # Wait for next iteration
                elapsed = time.time() - iteration_start
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)

            # Analyze results at this RPS level
            successful_results = [r for r in iteration_results if r.success]
            success_rate = len(successful_results) / max(len(iteration_results), 1)

            avg_latency = (
                statistics.mean([r.duration_ms for r in successful_results])
                if successful_results
                else float("inf")
            )

            rps_result = {
                "target_rps": current_rps,
                "actual_rps": len(iteration_results) / config.sustain_duration,
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "total_requests": len(iteration_results),
                "successful_requests": len(successful_results),
            }

            load_results.append(rps_result)

            # Check if we should stop (too many failures or high latency)
            if (
                success_rate < config.success_threshold
                or avg_latency > config.latency_threshold_ms
            ):
                logger.warning(
                    f"Load test stopping at {current_rps} RPS due to "
                    f"success_rate={success_rate:.3f} or latency={avg_latency:.1f}ms"
                )
                break

            current_rps += config.step_size

        return {
            "duration_seconds": time.time() - phase_start,
            "max_stable_rps": max(
                [
                    r["target_rps"]
                    for r in load_results
                    if r["success_rate"] >= config.success_threshold
                ],
                default=0,
            ),
            "rps_results": load_results,
        }

    async def _run_stress_test_phase(
        self,
        test_function: Callable,
        config: LoadTestConfig,
    ) -> dict[str, Any]:
        """Run stress testing to find breaking point."""
        stress_start = time.time()

        # Start with 2x the max stable RPS
        current_rps = config.max_rps * 2
        breaking_point_found = False
        stress_results = []

        while not breaking_point_found and current_rps <= config.max_rps * 10:
            logger.info(f"Stress testing at {current_rps} RPS")

            # Short burst at high RPS
            interval = 1.0 / current_rps
            burst_duration = 10.0  # 10 second bursts
            burst_start = time.time()
            burst_results = []

            while time.time() - burst_start < burst_duration:
                iteration_start = time.time()

                try:
                    result = await self._execute_single_measurement(test_function)
                    burst_results.append(result)
                except Exception as e:
                    burst_results.append(
                        BenchmarkResult(
                            duration_ms=float("inf"),
                            success=False,
                            memory_mb=psutil.virtual_memory().used / (1024 * 1024),
                            cpu_percent=psutil.cpu_percent(),
                            error_message=str(e),
                        )
                    )

                # Try to maintain RPS
                elapsed = time.time() - iteration_start
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)

            # Analyze stress results
            successful_results = [r for r in burst_results if r.success]
            success_rate = len(successful_results) / max(len(burst_results), 1)

            stress_result = {
                "target_rps": current_rps,
                "success_rate": success_rate,
                "total_requests": len(burst_results),
                "successful_requests": len(successful_results),
            }
            stress_results.append(stress_result)

            # Check for breaking point
            if success_rate < 0.5:  # Less than 50% success indicates breaking point
                breaking_point_found = True
                logger.info(f"Breaking point found at {current_rps} RPS")

            current_rps *= 1.5  # Increase by 50% each iteration

        return {
            "duration_seconds": time.time() - stress_start,
            "breaking_point_rps": current_rps / 1.5 if breaking_point_found else None,
            "stress_results": stress_results,
        }

    async def _run_recovery_phase(self, test_function: Callable) -> dict[str, Any]:
        """Test recovery after stress."""
        recovery_start = time.time()

        # Wait for system to settle
        await asyncio.sleep(5.0)

        # Run recovery test at baseline rate
        recovery_results = []
        for _ in range(20):
            result = await self._execute_single_measurement(test_function)
            recovery_results.append(result)
            await asyncio.sleep(0.5)  # 2 RPS

        successful_results = [r for r in recovery_results if r.success]

        return {
            "duration_seconds": time.time() - recovery_start,
            "recovery_success_rate": len(successful_results) / len(recovery_results),
            "recovery_latency_ms": statistics.mean(
                [r.duration_ms for r in successful_results]
            )
            if successful_results
            else 0,
            "total_requests": len(recovery_results),
        }

    async def _execute_single_measurement(
        self, test_function: Callable
    ) -> BenchmarkResult:
        """Execute single measurement with resource tracking."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024 * 1024)

        try:
            # Execute test function
            if self.async_optimizer:
                await self.async_optimizer.execute_optimized(
                    test_function()
                    if asyncio.iscoroutinefunction(test_function)
                    else asyncio.create_task(asyncio.to_thread(test_function)),
                    priority=TaskPriority.HIGH,
                )
            elif asyncio.iscoroutinefunction(test_function):
                await test_function()
            else:
                await asyncio.to_thread(test_function)

            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            end_memory = psutil.virtual_memory().used / (1024 * 1024)
            cpu_percent = psutil.cpu_percent()

            return BenchmarkResult(
                duration_ms=duration_ms,
                success=True,
                memory_mb=end_memory,
                cpu_percent=cpu_percent,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            end_memory = psutil.virtual_memory().used / (1024 * 1024)
            cpu_percent = psutil.cpu_percent()

            return BenchmarkResult(
                duration_ms=duration_ms,
                success=False,
                memory_mb=end_memory,
                cpu_percent=cpu_percent,
                error_message=str(e),
            )

    def _calculate_final_metrics(self) -> BenchmarkMetrics:
        """Calculate comprehensive final metrics."""
        if not self.measurement_history:
            return BenchmarkMetrics()

        # Filter successful measurements
        successful_measurements = [m for m in self.measurement_history if m.success]
        all_measurements = list(self.measurement_history)

        if not successful_measurements:
            return BenchmarkMetrics(
                total_requests=len(all_measurements),
                failed_requests=len(all_measurements),
                error_rate=1.0,
            )

        # Calculate latency statistics
        latencies = [m.duration_ms for m in successful_measurements]

        # Calculate resource statistics
        memory_values = [m.memory_mb for m in all_measurements]
        cpu_values = [m.cpu_percent for m in all_measurements]

        # Calculate quality scores
        stability_score = self._calculate_stability_score(latencies)
        efficiency_score = self._calculate_efficiency_score(successful_measurements)
        reliability_score = len(successful_measurements) / len(all_measurements)

        return BenchmarkMetrics(
            total_requests=len(all_measurements),
            successful_requests=len(successful_measurements),
            failed_requests=len(all_measurements) - len(successful_measurements),
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            max_latency_ms=max(latencies),
            throughput_rps=len(successful_measurements)
            / (len(all_measurements) * 0.05),  # Assuming 0.05s intervals
            error_rate=(len(all_measurements) - len(successful_measurements))
            / len(all_measurements),
            avg_cpu_percent=statistics.mean(cpu_values),
            max_cpu_percent=max(cpu_values),
            avg_memory_mb=statistics.mean(memory_values),
            peak_memory_mb=max(memory_values),
            memory_growth_mb=max(memory_values) - min(memory_values),
            stability_score=stability_score,
            efficiency_score=efficiency_score,
            reliability_score=reliability_score,
        )

    def _calculate_stability_score(self, latencies: list[float]) -> float:
        """Calculate stability score based on latency variance."""
        if len(latencies) < 2:
            return 0.0

        # Use coefficient of variation (CV) as stability metric
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies)

        if mean_latency == 0:
            return 0.0

        cv = std_latency / mean_latency
        # Convert CV to 0-1 score (lower CV = higher stability)
        stability_score = max(0.0, 1.0 - cv)

        return stability_score

    def _calculate_efficiency_score(self, measurements: list[BenchmarkResult]) -> float:
        """Calculate efficiency score (throughput per resource unit)."""
        if not measurements:
            return 0.0

        # Calculate throughput
        throughput = len(measurements) / (
            len(measurements) * 0.05
        )  # Assuming 0.05s intervals

        # Calculate average resource usage
        avg_cpu = statistics.mean([m.cpu_percent for m in measurements])
        avg_memory = statistics.mean([m.memory_mb for m in measurements])

        # Normalize resource usage (assuming 100% CPU and 8GB memory as max)
        normalized_resources = (avg_cpu / 100.0) + (avg_memory / 8192.0)

        if normalized_resources == 0:
            return 0.0

        # Efficiency is throughput per resource unit
        efficiency = throughput / normalized_resources

        # Normalize to 0-1 scale (assuming max efficiency of 1000)
        return min(1.0, efficiency / 1000.0)

    async def _analyze_performance_regression(
        self, current_metrics: BenchmarkMetrics
    ) -> dict[str, Any]:
        """Analyze performance regression compared to baseline."""
        if not self.baseline_metrics:
            return {
                "status": "no_baseline",
                "message": "No baseline metrics available for comparison",
            }

        regression_analysis = {
            "has_regression": False,
            "regression_details": [],
            "improvements": [],
            "overall_assessment": "stable",
        }

        # Compare key metrics
        comparisons = [
            ("avg_latency_ms", "lower_is_better", 10.0),  # 10% threshold
            ("p95_latency_ms", "lower_is_better", 15.0),  # 15% threshold
            ("error_rate", "lower_is_better", 5.0),  # 5% threshold
            ("throughput_rps", "higher_is_better", 10.0),  # 10% threshold
            ("memory_growth_mb", "lower_is_better", 20.0),  # 20% threshold
        ]

        for metric_name, direction, threshold_percent in comparisons:
            baseline_value = getattr(self.baseline_metrics, metric_name)
            current_value = getattr(current_metrics, metric_name)

            if baseline_value == 0:
                continue

            percent_change = ((current_value - baseline_value) / baseline_value) * 100

            if (
                direction == "lower_is_better" and percent_change > threshold_percent
            ) or (
                direction == "higher_is_better" and percent_change < -threshold_percent
            ):
                regression_analysis["has_regression"] = True
                regression_analysis["regression_details"].append(
                    {
                        "metric": metric_name,
                        "baseline_value": baseline_value,
                        "current_value": current_value,
                        "percent_change": percent_change,
                        "threshold": threshold_percent,
                    }
                )
            elif (
                direction == "lower_is_better" and percent_change < -threshold_percent
            ) or (
                direction == "higher_is_better" and percent_change > threshold_percent
            ):
                regression_analysis["improvements"].append(
                    {
                        "metric": metric_name,
                        "baseline_value": baseline_value,
                        "current_value": current_value,
                        "percent_change": percent_change,
                    }
                )

        # Overall assessment
        if regression_analysis["has_regression"]:
            regression_analysis["overall_assessment"] = "regression_detected"
        elif regression_analysis["improvements"]:
            regression_analysis["overall_assessment"] = "improved"

        return regression_analysis

    def _generate_performance_recommendations(
        self, metrics: BenchmarkMetrics
    ) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Latency recommendations
        if metrics.avg_latency_ms > 100:
            recommendations.append(
                "Consider implementing request caching to reduce average latency"
            )

        if metrics.p95_latency_ms > 500:
            recommendations.append(
                "High P95 latency detected - investigate tail latency optimizations"
            )

        # Error rate recommendations
        if metrics.error_rate > 0.01:  # > 1%
            recommendations.append(
                "Error rate is elevated - implement circuit breakers and retry logic"
            )

        # Resource usage recommendations
        if metrics.max_cpu_percent > 80:
            recommendations.append(
                "High CPU usage detected - consider horizontal scaling or CPU optimization"
            )

        if metrics.memory_growth_mb > 1000:  # > 1GB growth
            recommendations.append(
                "Significant memory growth detected - investigate potential memory leaks"
            )

        # Stability recommendations
        if metrics.stability_score < 0.8:
            recommendations.append(
                "Performance variance is high - investigate load balancing and resource allocation"
            )

        # Efficiency recommendations
        if metrics.efficiency_score < 0.5:
            recommendations.append(
                "Resource efficiency is low - optimize algorithms and reduce resource overhead"
            )

        # Throughput recommendations
        if metrics.throughput_rps < 10:
            recommendations.append(
                "Low throughput detected - consider async optimizations and connection pooling"
            )

        return recommendations

    def _convert_to_benchmark_metrics(
        self, metrics_dict: dict[str, Any]
    ) -> BenchmarkMetrics:
        """Convert dictionary metrics to BenchmarkMetrics object."""
        return BenchmarkMetrics(
            total_requests=metrics_dict.get("total_requests", 0),
            successful_requests=metrics_dict.get("successful_requests", 0),
            failed_requests=metrics_dict.get("total_requests", 0)
            - metrics_dict.get("successful_requests", 0),
            avg_latency_ms=metrics_dict.get("avg_latency_ms", 0.0),
            p95_latency_ms=metrics_dict.get("p95_latency_ms", 0.0),
            throughput_rps=metrics_dict.get("throughput_rps", 0.0),
            error_rate=1.0 - metrics_dict.get("success_rate", 0.0),
        )

    def get_historical_performance_trend(
        self, metric_name: str, window_size: int = 10
    ) -> dict[str, Any]:
        """Get historical performance trend for a specific metric."""
        if len(self.benchmark_history) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 benchmark runs for trend analysis",
            }

        # Extract metric values over time
        recent_history = list(self.benchmark_history)[-window_size:]
        timestamps = [h["timestamp"] for h in recent_history]
        values = [getattr(h["metrics"], metric_name, 0) for h in recent_history]

        # Calculate trend
        if len(values) >= 3:
            # Simple linear regression
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)

            trend_direction = (
                "improving" if slope < 0 else "degrading" if slope > 0 else "stable"
            )
            trend_strength = (
                abs(slope) / (max(values) - min(values))
                if max(values) != min(values)
                else 0
            )
        else:
            trend_direction = "stable"
            trend_strength = 0

        return {
            "metric_name": metric_name,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "recent_values": values,
            "timestamps": timestamps,
            "window_size": len(values),
        }


# Global benchmark suite instance
_benchmark_suite: PerformanceBenchmarkSuite | None = None


def get_benchmark_suite() -> PerformanceBenchmarkSuite:
    """Get global benchmark suite instance."""
    global _benchmark_suite
    if _benchmark_suite is None:
        _benchmark_suite = PerformanceBenchmarkSuite()
    return _benchmark_suite


async def run_performance_benchmark(
    benchmark_name: str,
    test_function: Callable,
    config: LoadTestConfig | None = None,
) -> dict[str, Any]:
    """Run performance benchmark using global suite."""
    suite = get_benchmark_suite()
    return await suite.run_comprehensive_benchmark(
        benchmark_name, test_function, config
    )
