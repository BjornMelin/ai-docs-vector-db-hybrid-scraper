"""Advanced Performance Optimization Suite.

This module provides sophisticated performance optimization capabilities including:
- Multi-tier adaptive caching with ML-driven optimization
- Intelligent async patterns with adaptive concurrency management
- Comprehensive performance benchmarking and analysis
- Automated optimization with statistical validation
- Real-time performance monitoring and alerting

Portfolio Showcase Components:
- AdaptiveCacheOptimizer: ML-driven cache optimization
- AsyncPerformanceOptimizer: Intelligent concurrency management
- PerformanceBenchmarkSuite: Comprehensive benchmarking
- PerformanceOptimizationShowcase: Portfolio demonstration

Usage:
    from src.services.performance import run_performance_optimization_showcase

    results = await run_performance_optimization_showcase()
    print(f"Performance improved by {results.average_improvement:.1f}%")
"""

from .async_optimization import (
    AdaptiveConcurrencyLimiter,
    AsyncPerformanceOptimizer,
    IntelligentTaskScheduler,
    TaskPriority,
    execute_batch_optimized,
    execute_optimized,
    get_async_optimizer,
    initialize_async_optimizer,
)
from .benchmark_suite import (
    BenchmarkMetrics,
    BenchmarkResult,
    LoadTestConfig,
    PerformanceBenchmarkSuite,
    get_benchmark_suite,
    run_performance_benchmark,
)
from .optimization_showcase import (
    OptimizationResults,
    PerformanceOptimizationShowcase,
    run_performance_optimization_showcase,
)


__all__ = [
    "AdaptiveConcurrencyLimiter",
    # Async Optimization
    "AsyncPerformanceOptimizer",
    "BenchmarkMetrics",
    "BenchmarkResult",
    "IntelligentTaskScheduler",
    "LoadTestConfig",
    "OptimizationResults",
    # Benchmarking
    "PerformanceBenchmarkSuite",
    # Showcase
    "PerformanceOptimizationShowcase",
    "TaskPriority",
    "execute_batch_optimized",
    "execute_optimized",
    "get_async_optimizer",
    "get_benchmark_suite",
    "initialize_async_optimizer",
    "run_performance_benchmark",
    "run_performance_optimization_showcase",
]
