
"""Performance measurement and monitoring utilities for testing.

This module provides tools for measuring execution time, memory usage, and other
performance metrics during testing, enabling performance regression detection
and benchmarking.
"""

import asyncio
import functools
import gc

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import statistics
import time
import tracemalloc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any


@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""

    execution_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    cpu_percent: float = 0.0
    operation_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    additional_metrics: dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """Track and analyze performance metrics across test runs."""

    def __init__(self):
        """Initialize the performance tracker."""
        self.measurements = []
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self._baseline_memory = self._get_memory_usage()
        else:
            self.process = None
            self._baseline_memory = 0.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.process is not None:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        if self.process is not None:
            return self.process.cpu_percent(interval=0.1)
        return 0.0

    @contextmanager
    def measure(self, operation_name: str = "operation"):
        """Context manager for measuring performance metrics.

        Args:
            operation_name: Name of the operation being measured
        """
        # Start tracking
        tracemalloc.start()
        gc.collect()  # Clean garbage before measurement

        start_time = time.perf_counter()
        self._get_memory_usage()

        try:
            yield
        finally:
            # End tracking
            end_time = time.perf_counter()
            self._get_memory_usage()

            # Get tracemalloc peak
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate metrics
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_peak_mb=peak / 1024 / 1024,
                memory_current_mb=current / 1024 / 1024,
                cpu_percent=self._get_cpu_percent(),
                operation_name=operation_name,
            )

            self.measurements.append(metrics)

    def get_statistics(
        self, operation_name: str | None = None
    ) -> dict[str, Any]:
        """Get performance statistics for all or specific operations.

        Args:
            operation_name: Filter by specific operation name

        Returns:
            Dictionary containing performance statistics
        """
        measurements = self.measurements
        if operation_name:
            measurements = [
                m for m in measurements if m.operation_name == operation_name
            ]

        if not measurements:
            return {"error": "No measurements available"}

        execution_times = [m.execution_time for m in measurements]
        memory_peaks = [m.memory_peak_mb for m in measurements]
        cpu_usage = [m.cpu_percent for m in measurements]

        return {
            "count": len(measurements),
            "operation_name": operation_name or "all",
            "execution_time": {
                "min": min(execution_times),
                "max": max(execution_times),
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "std_dev": statistics.stdev(execution_times)
                if len(execution_times) > 1
                else 0.0,
                "p95": self._percentile(execution_times, 95),
                "p99": self._percentile(execution_times, 99),
            },
            "memory_peak_mb": {
                "min": min(memory_peaks),
                "max": max(memory_peaks),
                "mean": statistics.mean(memory_peaks),
                "median": statistics.median(memory_peaks),
            },
            "cpu_percent": {
                "min": min(cpu_usage),
                "max": max(cpu_usage),
                "mean": statistics.mean(cpu_usage),
            },
        }

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def clear_measurements(self):
        """Clear all stored measurements."""
        self.measurements.clear()

    def export_measurements(self) -> list[dict[str, Any]]:
        """Export measurements as list of dictionaries."""
        return [
            {
                "operation_name": m.operation_name,
                "execution_time": m.execution_time,
                "memory_peak_mb": m.memory_peak_mb,
                "memory_current_mb": m.memory_current_mb,
                "cpu_percent": m.cpu_percent,
                "timestamp": m.timestamp,
                "additional_metrics": m.additional_metrics,
            }
            for m in self.measurements
        ]


def measure_execution_time(
    func: Callable | None = None, *, name: str | None = None
):
    """Decorator to measure function execution time.

    Args:
        func: Function to decorate
        name: Custom name for the measurement

    Returns:
        Decorated function that measures execution time
    """

    def decorator(f):
        operation_name = name or f.__name__

        if asyncio.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await f(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time

                    # Store measurement in function attribute
                    if not hasattr(f, "_performance_measurements"):
                        f._performance_measurements = []
                    f._performance_measurements.append(
                        {
                            "operation": operation_name,
                            "execution_time": execution_time,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

            return async_wrapper
        else:

            @functools.wraps(f)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time

                    # Store measurement in function attribute
                    if not hasattr(f, "_performance_measurements"):
                        f._performance_measurements = []
                    f._performance_measurements.append(
                        {
                            "operation": operation_name,
                            "execution_time": execution_time,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

            return sync_wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def memory_profiler(func: Callable | None = None, *, detailed: bool = False):
    """Decorator to profile memory usage of a function.

    Args:
        func: Function to decorate
        detailed: Whether to include detailed tracemalloc statistics

    Returns:
        Decorated function that profiles memory usage
    """

    def decorator(f):
        if asyncio.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                tracemalloc.start()
                gc.collect()

                try:
                    result = await f(*args, **kwargs)
                    return result
                finally:
                    current, peak = tracemalloc.get_traced_memory()

                    memory_info = {
                        "operation": f.__name__,
                        "memory_current_mb": current / 1024 / 1024,
                        "memory_peak_mb": peak / 1024 / 1024,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    if detailed:
                        snapshot = tracemalloc.take_snapshot()
                        top_stats = snapshot.statistics("lineno")[:10]
                        memory_info["top_allocations"] = [
                            f"{stat.traceback.format()}: {stat.size_diff / 1024:.1f} KB"
                            for stat in top_stats
                        ]

                    tracemalloc.stop()

                    # Store measurement in function attribute
                    if not hasattr(f, "_memory_measurements"):
                        f._memory_measurements = []
                    f._memory_measurements.append(memory_info)

            return async_wrapper
        else:

            @functools.wraps(f)
            def sync_wrapper(*args, **kwargs):
                tracemalloc.start()
                gc.collect()

                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    current, peak = tracemalloc.get_traced_memory()

                    memory_info = {
                        "operation": f.__name__,
                        "memory_current_mb": current / 1024 / 1024,
                        "memory_peak_mb": peak / 1024 / 1024,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    if detailed:
                        snapshot = tracemalloc.take_snapshot()
                        top_stats = snapshot.statistics("lineno")[:10]
                        memory_info["top_allocations"] = [
                            f"{stat.traceback.format()}: {stat.size_diff / 1024:.1f} KB"
                            for stat in top_stats
                        ]

                    tracemalloc.stop()

                    # Store measurement in function attribute
                    if not hasattr(f, "_memory_measurements"):
                        f._memory_measurements = []
                    f._memory_measurements.append(memory_info)

            return sync_wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    operations_per_second: float
    memory_usage_mb: float


class BenchmarkSuite:
    """Suite for running performance benchmarks."""

    def __init__(self, warmup_iterations: int = 3):
        """Initialize benchmark suite.

        Args:
            warmup_iterations: Number of warmup iterations before measurement
        """
        self.warmup_iterations = warmup_iterations
        self.results = []

    def benchmark_function(
        self,
        func: Callable,
        name: str | None = None,
        iterations: int = 100,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a function's performance.

        Args:
            func: Function to benchmark
            name: Name for the benchmark
            iterations: Number of iterations to run
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Benchmark results
        """
        benchmark_name = name or func.__name__

        # Warmup runs
        for _ in range(self.warmup_iterations):
            if asyncio.iscoroutinefunction(func):
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)

        # Measured runs
        execution_times = []
        tracemalloc.start()

        for _ in range(iterations):
            start_time = time.perf_counter()

            if asyncio.iscoroutinefunction(func):
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)

            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)

        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate statistics
        total_time = sum(execution_times)
        avg_time = total_time / iterations
        min_time = min(execution_times)
        max_time = max(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        ops_per_second = 1.0 / avg_time if avg_time > 0 else 0.0

        result = BenchmarkResult(
            name=benchmark_name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            operations_per_second=ops_per_second,
            memory_usage_mb=peak_memory / 1024 / 1024,
        )

        self.results.append(result)
        return result

    def compare_functions(
        self,
        functions: list[Callable],
        names: list[str] | None = None,
        iterations: int = 100,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Compare performance of multiple functions.

        Args:
            functions: List of functions to compare
            names: Optional names for the functions
            iterations: Number of iterations per function
            *args: Arguments to pass to functions
            **kwargs: Keyword arguments to pass to functions

        Returns:
            Comparison results
        """
        if names is None:
            names = [f.__name__ for f in functions]

        results = []
        for func, name in zip(functions, names, strict=False):
            result = self.benchmark_function(func, name, iterations, *args, **kwargs)
            results.append(result)

        # Find fastest and slowest
        fastest = min(results, key=lambda r: r.avg_time)
        slowest = max(results, key=lambda r: r.avg_time)

        return {
            "results": results,
            "fastest": fastest.name,
            "slowest": slowest.name,
            "speed_difference": slowest.avg_time / fastest.avg_time,
            "comparison": [
                {
                    "name": r.name,
                    "avg_time_ms": r.avg_time * 1000,
                    "ops_per_second": r.operations_per_second,
                    "relative_speed": fastest.avg_time / r.avg_time,
                    "memory_mb": r.memory_usage_mb,
                }
                for r in results
            ],
        }

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all benchmark results."""
        if not self.results:
            return {"message": "No benchmarks run yet"}

        return {
            "total_benchmarks": len(self.results),
            "benchmarks": [
                {
                    "name": r.name,
                    "iterations": r.iterations,
                    "avg_time_ms": r.avg_time * 1000,
                    "min_time_ms": r.min_time * 1000,
                    "max_time_ms": r.max_time * 1000,
                    "std_dev_ms": r.std_dev * 1000,
                    "ops_per_second": r.operations_per_second,
                    "memory_usage_mb": r.memory_usage_mb,
                }
                for r in self.results
            ],
            "fastest_benchmark": min(self.results, key=lambda r: r.avg_time).name,
            "slowest_benchmark": max(self.results, key=lambda r: r.avg_time).name,
        }

    def clear_results(self):
        """Clear all benchmark results."""
        self.results.clear()


# Utility functions for quick performance checks
def time_function(func: Callable, *args, **kwargs) -> float:
    """Time a single function execution.

    Args:
        func: Function to time
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function

    Returns:
        Execution time in seconds
    """
    start_time = time.perf_counter()

    if asyncio.iscoroutinefunction(func):
        asyncio.run(func(*args, **kwargs))
    else:
        func(*args, **kwargs)

    return time.perf_counter() - start_time


def profile_memory_usage(func: Callable, *args, **kwargs) -> dict[str, float]:
    """Profile memory usage of a function.

    Args:
        func: Function to profile
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function

    Returns:
        Dictionary with memory usage statistics
    """
    tracemalloc.start()
    gc.collect()

    if asyncio.iscoroutinefunction(func):
        asyncio.run(func(*args, **kwargs))
    else:
        func(*args, **kwargs)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "current_memory_mb": current / 1024 / 1024,
        "peak_memory_mb": peak / 1024 / 1024,
    }
