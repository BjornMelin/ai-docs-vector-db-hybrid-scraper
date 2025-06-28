"""Intelligent batch processing for optimal throughput and performance.

This module provides advanced batch processing capabilities with dynamic batching,
intelligent timing, and optimized resource utilization for maximum performance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Configuration for batch processing optimization."""

    max_batch_size: int = 50
    max_wait_time: float = 0.1  # 100ms
    min_batch_size: int = 1
    adaptive_sizing: bool = True
    performance_target_ms: float = 100.0  # Target processing time per batch


class BatchProcessor(Generic[T, R]):
    """Intelligent batch processing for optimal throughput."""

    def __init__(
        self,
        process_func: Callable[[List[T]], List[R]],
        config: BatchConfig | None = None,
    ):
        """Initialize batch processor with processing function and configuration.

        Args:
            process_func: Function to process batches of items
            config: Batch processing configuration

        """
        self.process_func = process_func
        self.config = config or BatchConfig()
        self.pending_items: List[T] = []
        self.pending_futures: List[asyncio.Future] = []
        self.last_batch_time = time.time()
        self.processing_lock = asyncio.Lock()

        # Performance tracking for adaptive sizing
        self.batch_performance_history: List[tuple[int, float]] = []
        self.optimal_batch_size = self.config.max_batch_size

    async def process_item(self, item: T) -> R:
        """Add item to batch and return result when processed.

        Args:
            item: Item to process

        Returns:
            Processing result for the item

        """
        future = asyncio.Future()

        async with self.processing_lock:
            self.pending_items.append(item)
            self.pending_futures.append(future)

            # Check if we should process the batch
            should_process = self._should_process_batch()

            if should_process:
                await self._process_batch()

        # Schedule delayed batch processing if not already processed
        if not should_process:
            asyncio.create_task(self._delayed_batch_processing())

        return await future

    def _should_process_batch(self) -> bool:
        """Determine if the current batch should be processed.

        Returns:
            True if batch should be processed now

        """
        current_size = len(self.pending_items)
        time_since_last = time.time() - self.last_batch_time

        # Use adaptive batch size if enabled
        target_size = (
            self.optimal_batch_size if self.config.adaptive_sizing 
            else self.config.max_batch_size
        )

        return (
            current_size >= target_size or
            (current_size >= self.config.min_batch_size and
             time_since_last > self.config.max_wait_time)
        )

    async def _process_batch(self) -> None:
        """Process current batch of items with performance tracking."""
        if not self.pending_items:
            return

        items = self.pending_items.copy()
        futures = self.pending_futures.copy()

        # Clear pending lists
        self.pending_items.clear()
        self.pending_futures.clear()
        self.last_batch_time = time.time()

        batch_size = len(items)
        start_time = time.time()

        try:
            # Process batch
            if asyncio.iscoroutinefunction(self.process_func):
                results = await self.process_func(items)
            else:
                results = await asyncio.get_event_loop().run_in_executor(
                    None, self.process_func, items
                )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            # Track performance for adaptive sizing
            self._update_performance_metrics(batch_size, processing_time)

            # Return results to waiting coroutines
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

            logger.debug(
                f"Processed batch of {batch_size} items in {processing_time:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Propagate error to all waiting coroutines
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    def _update_performance_metrics(self, batch_size: int, processing_time: float) -> None:
        """Update performance metrics and adjust optimal batch size.

        Args:
            batch_size: Size of the processed batch
            processing_time: Time taken to process the batch in milliseconds

        """
        # Add to performance history
        self.batch_performance_history.append((batch_size, processing_time))

        # Keep only recent history (last 50 batches)
        if len(self.batch_performance_history) > 50:
            self.batch_performance_history = self.batch_performance_history[-50:]

        # Update optimal batch size if adaptive sizing is enabled
        if self.config.adaptive_sizing:
            self._calculate_optimal_batch_size()

    def _calculate_optimal_batch_size(self) -> None:
        """Calculate optimal batch size based on performance history."""
        if len(self.batch_performance_history) < 5:
            return  # Need more data

        # Find the batch size that gives the best performance per item
        size_performance = {}

        for batch_size, processing_time in self.batch_performance_history:
            if batch_size not in size_performance:
                size_performance[batch_size] = []
            
            # Calculate processing time per item
            time_per_item = processing_time / batch_size
            size_performance[batch_size].append(time_per_item)

        # Find the batch size with the best average time per item
        best_size = self.config.max_batch_size
        best_avg_time = float('inf')

        for size, times in size_performance.items():
            if len(times) >= 2:  # Need at least 2 samples
                avg_time = sum(times) / len(times)
                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_size = size

        # Update optimal batch size within bounds
        self.optimal_batch_size = max(
            self.config.min_batch_size,
            min(best_size, self.config.max_batch_size)
        )

        logger.debug(f"Updated optimal batch size to {self.optimal_batch_size}")

    async def _delayed_batch_processing(self) -> None:
        """Process batch after delay if minimum wait time exceeded."""
        await asyncio.sleep(self.config.max_wait_time)

        async with self.processing_lock:
            if (self.pending_items and 
                time.time() - self.last_batch_time >= self.config.max_wait_time):
                await self._process_batch()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics.

        Returns:
            Dict containing performance metrics

        """
        if not self.batch_performance_history:
            return {"status": "no_data"}

        # Calculate statistics
        total_batches = len(self.batch_performance_history)
        total_items = sum(size for size, _ in self.batch_performance_history)
        total_time = sum(time for _, time in self.batch_performance_history)

        avg_batch_size = total_items / total_batches if total_batches > 0 else 0
        avg_processing_time = total_time / total_batches if total_batches > 0 else 0
        avg_time_per_item = total_time / total_items if total_items > 0 else 0

        return {
            "total_batches": total_batches,
            "total_items_processed": total_items,
            "avg_batch_size": avg_batch_size,
            "avg_processing_time_ms": avg_processing_time,
            "avg_time_per_item_ms": avg_time_per_item,
            "optimal_batch_size": self.optimal_batch_size,
            "current_config": {
                "max_batch_size": self.config.max_batch_size,
                "max_wait_time_ms": self.config.max_wait_time * 1000,
                "adaptive_sizing": self.config.adaptive_sizing,
            },
        }

    async def flush_pending(self) -> None:
        """Force processing of any pending items."""
        async with self.processing_lock:
            if self.pending_items:
                await self._process_batch()


# Example usage for embeddings service
class OptimizedEmbeddingService:
    """Example optimized embedding service using batch processing."""

    def __init__(self):
        """Initialize the optimized embedding service."""
        self.batch_processor = BatchProcessor(
            self._batch_generate_embeddings,
            BatchConfig(
                max_batch_size=50,
                max_wait_time=0.1,
                adaptive_sizing=True,
                performance_target_ms=100.0,
            ),
        )

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate single embedding with batching optimization.

        Args:
            text: Text to generate embedding for

        Returns:
            Generated embedding vector

        """
        return await self.batch_processor.process_item(text)

    async def _batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Actual batch embedding generation implementation.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors

        """
        # This would be replaced with actual embedding service calls
        # For example, using OpenAI batch API or FastEmbed batch processing
        
        # Simulate processing time
        await asyncio.sleep(0.01 * len(texts))
        
        # Return mock embeddings
        return [[0.1] * 384 for _ in texts]

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get embedding service performance statistics.

        Returns:
            Performance statistics for the embedding service

        """
        return self.batch_processor.get_performance_stats()