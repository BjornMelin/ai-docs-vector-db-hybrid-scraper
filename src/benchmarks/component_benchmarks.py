"""Component-level benchmarks for individual ML components.

This module provides isolated performance testing for each component
of the advanced hybrid search system.
"""

import asyncio
import logging
import statistics
import time
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..config import UnifiedConfig
from ..models.vector_search import AdvancedHybridSearchRequest, QueryClassification
from ..services.vector_db.advanced_hybrid_search import AdvancedHybridSearchService

logger = logging.getLogger(__name__)


class ComponentBenchmarkResult(BaseModel):
    """Results for a single component benchmark."""
    
    component_name: str = Field(..., description="Name of the benchmarked component")
    total_executions: int = Field(..., description="Total number of executions")
    successful_executions: int = Field(..., description="Number of successful executions")
    failed_executions: int = Field(..., description="Number of failed executions")
    
    # Timing metrics (in milliseconds)
    avg_latency_ms: float = Field(..., description="Average latency")
    p50_latency_ms: float = Field(..., description="50th percentile latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    min_latency_ms: float = Field(..., description="Minimum latency")
    max_latency_ms: float = Field(..., description="Maximum latency")
    
    # Quality metrics
    accuracy: float = Field(0.0, description="Component accuracy (if applicable)")
    error_rate: float = Field(..., description="Error rate")
    
    # Additional metrics
    throughput_per_second: float = Field(..., description="Operations per second")
    memory_usage_mb: float = Field(0.0, description="Peak memory usage in MB")


class ComponentBenchmarks:
    """Individual component benchmark runner."""
    
    def __init__(self, config: UnifiedConfig):
        """Initialize component benchmarks.
        
        Args:
            config: Unified configuration
        """
        self.config = config
        
    async def run_all_component_benchmarks(
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: List[AdvancedHybridSearchRequest]
    ) -> Dict[str, ComponentBenchmarkResult]:
        """Run benchmarks for all components.
        
        Args:
            search_service: Advanced hybrid search service
            test_queries: Test queries to use for benchmarking
            
        Returns:
            Dictionary mapping component names to benchmark results
        """
        results = {}
        
        # Query Classifier benchmarks
        logger.info("Benchmarking Query Classifier...")
        results["query_classifier"] = await self.benchmark_query_classifier(
            search_service.query_classifier, test_queries
        )
        
        # Model Selector benchmarks
        logger.info("Benchmarking Model Selector...")
        results["model_selector"] = await self.benchmark_model_selector(
            search_service.model_selector, search_service.query_classifier, test_queries
        )
        
        # Adaptive Fusion Tuner benchmarks
        logger.info("Benchmarking Adaptive Fusion Tuner...")
        results["adaptive_fusion_tuner"] = await self.benchmark_adaptive_fusion_tuner(
            search_service.adaptive_fusion_tuner, search_service.query_classifier, test_queries
        )
        
        # SPLADE Provider benchmarks
        logger.info("Benchmarking SPLADE Provider...")
        results["splade_provider"] = await self.benchmark_splade_provider(
            search_service.splade_provider, test_queries
        )
        
        # End-to-end search benchmarks
        logger.info("Benchmarking End-to-End Search...")
        results["end_to_end_search"] = await self.benchmark_end_to_end_search(
            search_service, test_queries
        )
        
        return results
    
    async def benchmark_query_classifier(
        self, query_classifier, test_queries: List[AdvancedHybridSearchRequest]
    ) -> ComponentBenchmarkResult:
        """Benchmark query classifier performance."""
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        
        for query in test_queries:
            try:
                start = time.perf_counter()
                await query_classifier.classify_query(query.query)
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1
                
            except Exception as e:
                logger.debug(f"Query classification failed: {e}")
                failures += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return ComponentBenchmarkResult(
            component_name="query_classifier",
            total_executions=len(test_queries),
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            error_rate=failures / len(test_queries) if test_queries else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0
        )
    
    async def benchmark_model_selector(
        self, model_selector, query_classifier, test_queries: List[AdvancedHybridSearchRequest]
    ) -> ComponentBenchmarkResult:
        """Benchmark model selector performance."""
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        
        for query in test_queries[:20]:  # Use subset to avoid quota limits
            try:
                # First classify the query
                classification = await query_classifier.classify_query(query.query)
                
                start = time.perf_counter()
                await model_selector.select_optimal_model(classification)
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1
                
            except Exception as e:
                logger.debug(f"Model selection failed: {e}")
                failures += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return ComponentBenchmarkResult(
            component_name="model_selector",
            total_executions=min(20, len(test_queries)),
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            error_rate=failures / min(20, len(test_queries)) if test_queries else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0
        )
    
    async def benchmark_adaptive_fusion_tuner(
        self, fusion_tuner, query_classifier, test_queries: List[AdvancedHybridSearchRequest]
    ) -> ComponentBenchmarkResult:
        """Benchmark adaptive fusion tuner performance."""
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        
        for i, query in enumerate(test_queries[:15]):  # Use smaller subset
            try:
                # First classify the query
                classification = await query_classifier.classify_query(query.query)
                
                start = time.perf_counter()
                await fusion_tuner.compute_adaptive_weights(
                    classification, f"benchmark_{i}"
                )
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1
                
            except Exception as e:
                logger.debug(f"Adaptive fusion failed: {e}")
                failures += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return ComponentBenchmarkResult(
            component_name="adaptive_fusion_tuner",
            total_executions=min(15, len(test_queries)),
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            error_rate=failures / min(15, len(test_queries)) if test_queries else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0
        )
    
    async def benchmark_splade_provider(
        self, splade_provider, test_queries: List[AdvancedHybridSearchRequest]
    ) -> ComponentBenchmarkResult:
        """Benchmark SPLADE provider performance."""
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        
        for query in test_queries[:25]:  # Use subset for SPLADE
            try:
                start = time.perf_counter()
                await splade_provider.generate_sparse_vector(query.query)
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1
                
            except Exception as e:
                logger.debug(f"SPLADE generation failed: {e}")
                failures += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return ComponentBenchmarkResult(
            component_name="splade_provider",
            total_executions=min(25, len(test_queries)),
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            error_rate=failures / min(25, len(test_queries)) if test_queries else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0
        )
    
    async def benchmark_end_to_end_search(
        self, search_service: AdvancedHybridSearchService, test_queries: List[AdvancedHybridSearchRequest]
    ) -> ComponentBenchmarkResult:
        """Benchmark end-to-end search performance."""
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        
        for query in test_queries[:10]:  # Use small subset for full end-to-end
            try:
                start = time.perf_counter()
                await search_service.advanced_hybrid_search(query)
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1
                
            except Exception as e:
                logger.debug(f"End-to-end search failed: {e}")
                failures += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return ComponentBenchmarkResult(
            component_name="end_to_end_search",
            total_executions=min(10, len(test_queries)),
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            error_rate=failures / min(10, len(test_queries)) if test_queries else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]
    
    async def benchmark_cache_performance(
        self, search_service: AdvancedHybridSearchService, test_queries: List[AdvancedHybridSearchRequest]
    ) -> Dict[str, float]:
        """Benchmark cache performance specifically."""
        cache_metrics = {}
        
        # Clear caches first
        if hasattr(search_service.splade_provider, 'clear_cache'):
            search_service.splade_provider.clear_cache()
        
        # First run (cold cache)
        cold_latencies = []
        for query in test_queries[:5]:
            try:
                start = time.perf_counter()
                await search_service.splade_provider.generate_sparse_vector(query.query)
                end = time.perf_counter()
                cold_latencies.append((end - start) * 1000)
            except Exception:
                pass
        
        # Second run (warm cache) - same queries
        warm_latencies = []
        for query in test_queries[:5]:
            try:
                start = time.perf_counter()
                await search_service.splade_provider.generate_sparse_vector(query.query)
                end = time.perf_counter()
                warm_latencies.append((end - start) * 1000)
            except Exception:
                pass
        
        if cold_latencies and warm_latencies:
            cache_metrics["cold_avg_latency_ms"] = statistics.mean(cold_latencies)
            cache_metrics["warm_avg_latency_ms"] = statistics.mean(warm_latencies)
            cache_metrics["cache_speedup_factor"] = (
                statistics.mean(cold_latencies) / statistics.mean(warm_latencies)
            )
        
        # Get cache statistics
        if hasattr(search_service.splade_provider, 'get_cache_stats'):
            cache_stats = search_service.splade_provider.get_cache_stats()
            cache_metrics.update(cache_stats)
        
        return cache_metrics
    
    async def benchmark_concurrent_component_access(
        self, search_service: AdvancedHybridSearchService, test_queries: List[AdvancedHybridSearchRequest]
    ) -> Dict[str, ComponentBenchmarkResult]:
        """Benchmark components under concurrent access."""
        concurrent_results = {}
        
        # Test query classifier under concurrency
        async def concurrent_classify(query_text: str):
            return await search_service.query_classifier.classify_query(query_text)
        
        tasks = [concurrent_classify(q.query) for q in test_queries[:10]]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = len(results) - successes
        total_time = end_time - start_time
        
        concurrent_results["query_classifier_concurrent"] = ComponentBenchmarkResult(
            component_name="query_classifier_concurrent",
            total_executions=len(tasks),
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=(total_time * 1000) / len(tasks),
            p50_latency_ms=(total_time * 1000) / len(tasks),
            p95_latency_ms=(total_time * 1000) / len(tasks),
            p99_latency_ms=(total_time * 1000) / len(tasks),
            min_latency_ms=0,
            max_latency_ms=(total_time * 1000),
            error_rate=failures / len(tasks),
            throughput_per_second=successes / total_time if total_time > 0 else 0
        )
        
        return concurrent_results