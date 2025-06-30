"""Parallel processing integration layer for unified ML optimization.

This module provides a unified interface that integrates all parallel processing
components: embeddings, text analysis, caching, and performance monitoring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from src.services.cache.intelligent import (
    CacheConfig,
    EmbeddingCache,
    IntelligentCache,
    SearchResultCache,
)
from src.services.embeddings.parallel import (
    ParallelConfig,
    ParallelEmbeddingProcessor,
    parallel_content_classification,
    parallel_metadata_extraction,
)
from src.services.processing.algorithms import OptimizedTextAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class SystemPerformanceMetrics:
    """System-wide performance metrics."""

    total_requests: int = 0
    avg_response_time_ms: float = 0.0
    throughput_requests_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    parallel_efficiency: float = 0.0
    memory_usage_mb: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for the optimization system."""

    enable_parallel_processing: bool = True
    enable_intelligent_caching: bool = True
    enable_optimized_algorithms: bool = True
    parallel_config: ParallelConfig | None = None
    cache_config: CacheConfig | None = None
    performance_monitoring: bool = True
    auto_optimization: bool = True


class ParallelProcessingSystem:
    """Unified parallel processing system for ML operations."""

    def __init__(
        self,
        embedding_manager: Any,
        config: OptimizationConfig | None = None,
    ):
        """Initialize the parallel processing system.

        Args:
            embedding_manager: EmbeddingManager instance
            config: Optimization configuration
        """
        self.embedding_manager = embedding_manager
        self.config = config or OptimizationConfig()

        # Initialize components
        self._initialize_components()

        # Performance tracking
        self.start_time = time.time()
        self.metrics = SystemPerformanceMetrics()
        self._request_times: list[float] = []
        self._error_count = 0

        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()

        logger.info("ParallelProcessingSystem initialized with optimizations enabled")

    def _initialize_components(self) -> None:
        """Initialize all optimization components."""
        # Parallel processing configuration
        if self.config.parallel_config is None:
            self.config.parallel_config = ParallelConfig(
                max_concurrent_tasks=50,
                batch_size_per_worker=10,
                adaptive_batching=True,
                performance_monitoring=True,
            )

        # Cache configuration
        if self.config.cache_config is None:
            self.config.cache_config = CacheConfig(
                max_memory_mb=256,
                enable_compression=True,
                enable_cache_warming=True,
                enable_persistence=True,
            )

        # Initialize optimized text analyzer
        if self.config.enable_optimized_algorithms:
            self.text_analyzer = OptimizedTextAnalyzer()

        # Initialize caching systems
        if self.config.enable_intelligent_caching:
            self.embedding_cache = EmbeddingCache(self.config.cache_config)
            self.search_cache = SearchResultCache(self.config.cache_config)
            self.general_cache = IntelligentCache[str, Any](self.config.cache_config)

        # Initialize parallel embedding processor
        if self.config.enable_parallel_processing:
            self.parallel_embeddings = ParallelEmbeddingProcessor(
                self.embedding_manager, self.config.parallel_config
            )

    async def process_documents_parallel(
        self,
        documents: list[dict[str, Any]],
        enable_classification: bool = True,
        enable_metadata_extraction: bool = True,
        enable_embedding_generation: bool = True,
    ) -> dict[str, Any]:
        """Process documents in parallel with full optimization.

        Args:
            documents: List of document dictionaries with 'content' and 'url'
            enable_classification: Whether to perform content classification
            enable_metadata_extraction: Whether to extract metadata
            enable_embedding_generation: Whether to generate embeddings

        Returns:
            Processing results with performance metrics
        """
        request_start = time.time()

        try:
            results = {
                "documents": [],
                "processing_stats": {},
                "performance_metrics": {},
                "optimization_enabled": {
                    "parallel_processing": self.config.enable_parallel_processing,
                    "intelligent_caching": self.config.enable_intelligent_caching,
                    "optimized_algorithms": self.config.enable_optimized_algorithms,
                },
            }

            # Extract content for processing
            texts = [doc.get("content", "") for doc in documents]
            urls = [doc.get("url", "") for doc in documents]

            # Parallel processing pipeline
            processing_tasks = []

            # 1. Text analysis (optimized algorithms)
            if self.config.enable_optimized_algorithms:
                analysis_task = asyncio.create_task(self._parallel_text_analysis(texts))
                processing_tasks.append(("text_analysis", analysis_task))

            # 2. Embedding generation (parallel processing)
            if enable_embedding_generation and self.config.enable_parallel_processing:
                embedding_task = asyncio.create_task(
                    self._parallel_embedding_generation(texts)
                )
                processing_tasks.append(("embeddings", embedding_task))

            # 3. Content classification (parallel processing)
            if enable_classification and hasattr(self, "content_classifier"):
                classification_task = asyncio.create_task(
                    parallel_content_classification(
                        texts, self.content_classifier, self.config.parallel_config
                    )
                )
                processing_tasks.append(("classification", classification_task))

            # 4. Metadata extraction (parallel processing)
            if enable_metadata_extraction and hasattr(self, "metadata_extractor"):
                metadata_items = list(zip(texts, urls, strict=False))
                metadata_task = asyncio.create_task(
                    parallel_metadata_extraction(
                        metadata_items,
                        self.metadata_extractor,
                        self.config.parallel_config,
                    )
                )
                processing_tasks.append(("metadata", metadata_task))

            # Execute all tasks in parallel
            task_results = {}
            if processing_tasks:
                async with asyncio.TaskGroup() as tg:
                    running_tasks = {}
                    for task_name, task in processing_tasks:
                        running_task = tg.create_task(task)
                        running_tasks[task_name] = running_task

                # Collect results
                for task_name, task in running_tasks.items():
                    task_results[task_name] = task.result()

            # Combine results
            for i, document in enumerate(documents):
                processed_doc = document.copy()

                # Add text analysis results
                if "text_analysis" in task_results:
                    processed_doc["text_analysis"] = task_results["text_analysis"][i]

                # Add embedding results
                if "embeddings" in task_results:
                    embed_result = task_results["embeddings"]
                    if embed_result and "embeddings" in embed_result:
                        processed_doc["embedding"] = embed_result["embeddings"][i]
                        processed_doc["embedding_metrics"] = embed_result.get("metrics")

                # Add classification results
                if "classification" in task_results:
                    processed_doc["classification"] = task_results["classification"][i]

                # Add metadata results
                if "metadata" in task_results:
                    processed_doc["metadata"] = task_results["metadata"][i]

                results["documents"].append(processed_doc)

            # Calculate performance metrics
            processing_time_ms = (time.time() - request_start) * 1000

            results["processing_stats"] = {
                "total_documents": len(documents),
                "processing_time_ms": processing_time_ms,
                "avg_time_per_document_ms": processing_time_ms / max(len(documents), 1),
                "throughput_docs_per_second": len(documents)
                / max(processing_time_ms / 1000, 0.001),
            }

            # Add detailed performance metrics
            if self.config.performance_monitoring:
                results[
                    "performance_metrics"
                ] = await self._calculate_performance_metrics(
                    task_results, processing_time_ms
                )

            # Update system metrics
            await self._update_system_metrics(processing_time_ms, success=True)

            return results

        except Exception as e:
            logger.exception(
                f"Parallel document processing failed: {e}"
            )  # TODO: Convert f-string to logging format
            await self._update_system_metrics(
                (time.time() - request_start) * 1000, success=False
            )
            raise

    async def _parallel_text_analysis(self, texts: list[str]) -> list[Any]:
        """Perform parallel text analysis using optimized algorithms.

        Args:
            texts: List of texts to analyze

        Returns:
            List of text analysis results
        """
        # Use optimized algorithms with caching
        return self.text_analyzer.batch_analyze_texts(texts)

    async def _parallel_embedding_generation(self, texts: list[str]) -> dict[str, Any]:
        """Generate embeddings in parallel with caching.

        Args:
            texts: List of texts to embed

        Returns:
            Embedding generation results
        """
        # Check cache for existing embeddings
        if self.config.enable_intelligent_caching:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cached = await self.embedding_cache.get_embedding(
                    text=text, provider="default", model="default", dimensions=384
                )

                if cached is not None:
                    cached_embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = (
                    await self.parallel_embeddings.generate_embeddings_parallel(
                        uncached_texts
                    )
                )

                # Cache new embeddings
                for j, text in enumerate(uncached_texts):
                    if "embeddings" in new_embeddings and j < len(
                        new_embeddings["embeddings"]
                    ):
                        await self.embedding_cache.set_embedding(
                            text=text,
                            provider="default",
                            model="default",
                            dimensions=384,
                            embedding=new_embeddings["embeddings"][j],
                        )

                # Combine cached and new embeddings
                all_embeddings = [None] * len(texts)

                # Add cached embeddings
                for i, embedding in cached_embeddings:
                    all_embeddings[i] = embedding

                # Add new embeddings
                for j, i in enumerate(uncached_indices):
                    if "embeddings" in new_embeddings and j < len(
                        new_embeddings["embeddings"]
                    ):
                        all_embeddings[i] = new_embeddings["embeddings"][j]

                # Update result
                new_embeddings["embeddings"] = all_embeddings
                new_embeddings["cache_hits"] = len(cached_embeddings)
                new_embeddings["cache_misses"] = len(uncached_texts)

                return new_embeddings
            # All embeddings were cached
            all_embeddings = [None] * len(texts)
            for i, embedding in cached_embeddings:
                all_embeddings[i] = embedding

            return {
                "embeddings": all_embeddings,
                "cache_hits": len(cached_embeddings),
                "cache_misses": 0,
                "parallel_enabled": False,  # No generation needed
            }
        # Generate embeddings without caching
        return await self.parallel_embeddings.generate_embeddings_parallel(texts)

    async def _calculate_performance_metrics(
        self,
        task_results: dict[str, Any],
        total_processing_time_ms: float,
    ) -> dict[str, Any]:
        """Calculate detailed performance metrics.

        Args:
            task_results: Results from all processing tasks
            total_processing_time_ms: Total processing time

        Returns:
            Detailed performance metrics
        """
        metrics = {
            "total_processing_time_ms": total_processing_time_ms,
            "optimization_gains": {},
            "cache_performance": {},
            "parallel_efficiency": {},
        }

        # Text analysis optimization gains
        if "text_analysis" in task_results:
            analysis_results = task_results["text_analysis"]
            if analysis_results:
                avg_processing_time = sum(
                    r.processing_time_ms for r in analysis_results
                ) / len(analysis_results)

                metrics["optimization_gains"]["text_analysis"] = {
                    "avg_processing_time_ms": avg_processing_time,
                    "algorithm_complexity": "O(n)",
                    "cache_enabled": True,
                }

        # Embedding generation performance
        if "embeddings" in task_results:
            embed_results = task_results["embeddings"]
            if isinstance(embed_results, dict):
                metrics["parallel_efficiency"]["embeddings"] = {
                    "parallel_enabled": embed_results.get("parallel_enabled", False),
                    "speedup_achieved": embed_results.get("speedup_achieved", "1.0x"),
                    "efficiency": embed_results.get("efficiency", "0%"),
                }

                if "cache_hits" in embed_results:
                    total_requests = (
                        embed_results["cache_hits"] + embed_results["cache_misses"]
                    )
                    cache_hit_rate = embed_results["cache_hits"] / max(
                        total_requests, 1
                    )

                    metrics["cache_performance"]["embeddings"] = {
                        "cache_hit_rate": cache_hit_rate,
                        "cache_hits": embed_results["cache_hits"],
                        "cache_misses": embed_results["cache_misses"],
                    }

        # System cache performance
        if self.config.enable_intelligent_caching:
            embedding_cache_stats = self.embedding_cache.get_stats()
            metrics["cache_performance"]["system"] = {
                "hit_rate": embedding_cache_stats.hit_rate,
                "memory_usage_mb": embedding_cache_stats.memory_usage_mb,
                "item_count": embedding_cache_stats.item_count,
            }

        return metrics

    async def _update_system_metrics(
        self,
        processing_time_ms: float,
        success: bool,
    ) -> None:
        """Update system-wide performance metrics.

        Args:
            processing_time_ms: Request processing time
            success: Whether the request succeeded
        """
        self.metrics.total_requests += 1

        if not success:
            self._error_count += 1

        # Update request times (keep last 1000)
        self._request_times.append(processing_time_ms)
        if len(self._request_times) > 1000:
            self._request_times = self._request_times[-1000:]

        # Calculate moving averages
        if self._request_times:
            self.metrics.avg_response_time_ms = sum(self._request_times) / len(
                self._request_times
            )
            self.metrics.throughput_requests_per_second = 1000 / max(
                self.metrics.avg_response_time_ms, 1
            )

        # Calculate error rate
        self.metrics.error_rate = self._error_count / max(
            self.metrics.total_requests, 1
        )

        # Update uptime
        self.metrics.uptime_seconds = time.time() - self.start_time

        # Update cache hit rate
        if self.config.enable_intelligent_caching:
            cache_stats = self.embedding_cache.get_stats()
            self.metrics.cache_hit_rate = cache_stats.hit_rate
            self.metrics.memory_usage_mb = cache_stats.memory_usage_mb

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status and performance metrics.

        Returns:
            System status information
        """
        status = {
            "system_health": {
                "status": "healthy" if self.metrics.error_rate < 0.05 else "degraded",
                "uptime_seconds": self.metrics.uptime_seconds,
                "total_requests": self.metrics.total_requests,
                "error_rate": self.metrics.error_rate,
            },
            "performance_metrics": {
                "avg_response_time_ms": self.metrics.avg_response_time_ms,
                "throughput_rps": self.metrics.throughput_requests_per_second,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "memory_usage_mb": self.metrics.memory_usage_mb,
            },
            "optimization_status": {
                "parallel_processing": self.config.enable_parallel_processing,
                "intelligent_caching": self.config.enable_intelligent_caching,
                "optimized_algorithms": self.config.enable_optimized_algorithms,
                "auto_optimization": self.config.auto_optimization,
            },
        }

        # Add component-specific status
        if self.config.enable_parallel_processing:
            embedding_stats = self.parallel_embeddings.get_performance_stats()
            status["parallel_processing"] = embedding_stats

        if self.config.enable_optimized_algorithms:
            # Get text analyzer cache info
            cache_info = self.text_analyzer.analyze_text_optimized.cache_info()
            status["text_analysis"] = {
                "cache_hits": cache_info.hits,
                "cache_misses": cache_info.misses,
                "hit_rate": cache_info.hits
                / max(cache_info.hits + cache_info.misses, 1),
                "algorithm_complexity": "O(n)",
            }

        if self.config.enable_intelligent_caching:
            cache_memory = self.embedding_cache.get_memory_usage()
            status["caching_system"] = cache_memory

        return status

    async def optimize_performance(self) -> dict[str, Any]:
        """Automatically optimize system performance based on metrics.

        Returns:
            Optimization results
        """
        if not self.config.auto_optimization:
            return {"status": "auto_optimization_disabled"}

        optimizations_applied = []

        # Clear caches if hit rate is low
        if self.metrics.cache_hit_rate < 0.3 and self.config.enable_intelligent_caching:
            await self.embedding_cache.clear()
            await self.search_cache.clear()
            optimizations_applied.append("cache_cleared_low_hit_rate")

        # Clear text analyzer cache periodically
        if self.config.enable_optimized_algorithms:
            cache_stats = self.text_analyzer.clear_cache()
            if cache_stats["analyze_text_cache"]["hit_rate"] > 0:
                optimizations_applied.append("text_analyzer_cache_optimized")

        # Adjust parallel processing config based on performance
        if (
            self.config.enable_parallel_processing
            and self.metrics.avg_response_time_ms > 5000
        ):  # > 5 seconds
            # Increase concurrent tasks for better parallelization
            current_config = self.config.parallel_config
            if current_config.max_concurrent_tasks < 100:
                current_config.max_concurrent_tasks = min(
                    current_config.max_concurrent_tasks + 10, 100
                )
                optimizations_applied.append("increased_parallelization")

        return {
            "status": "completed",
            "optimizations_applied": optimizations_applied,
            "timestamp": time.time(),
        }

    async def cleanup(self) -> None:
        """Cleanup system resources."""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()

            # Clear caches
            if self.config.enable_intelligent_caching:
                await self.embedding_cache.clear()
                await self.search_cache.clear()
                await self.general_cache.clear()

            # Clear algorithm caches
            if self.config.enable_optimized_algorithms:
                self.text_analyzer.clear_cache()

            logger.info("ParallelProcessingSystem cleanup completed")

        except Exception as e:
            logger.exception(
                f"Error during cleanup: {e}"
            )  # TODO: Convert f-string to logging format


# Factory function for easy initialization
def create_optimized_system(
    embedding_manager: Any,
    enable_all_optimizations: bool = True,
    custom_config: OptimizationConfig | None = None,
) -> ParallelProcessingSystem:
    """Create an optimized parallel processing system.

    Args:
        embedding_manager: EmbeddingManager instance
        enable_all_optimizations: Whether to enable all optimizations
        custom_config: Custom optimization configuration

    Returns:
        Configured ParallelProcessingSystem
    """
    if custom_config is None:
        config = OptimizationConfig(
            enable_parallel_processing=enable_all_optimizations,
            enable_intelligent_caching=enable_all_optimizations,
            enable_optimized_algorithms=enable_all_optimizations,
            performance_monitoring=True,
            auto_optimization=enable_all_optimizations,
        )
    else:
        config = custom_config

    return ParallelProcessingSystem(embedding_manager, config)
