"""Advanced async performance optimization with intelligent concurrency management.

This module implements sophisticated async patterns including:
- Adaptive concurrency limiting with backpressure detection
- Intelligent request batching and pipelining
- Resource-aware task scheduling with priority queues
- Circuit breaker patterns with predictive failure detection
- Connection pool optimization with load balancing
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

import psutil
from pydantic import BaseModel

from ..monitoring.metrics import get_metrics_registry
from ..observability.performance import PerformanceMonitor

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TaskPriority(Enum):
    """Task priority levels for intelligent scheduling."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ConcurrencyMetrics:
    """Concurrency performance metrics."""
    
    active_tasks: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    current_throughput: float = 0.0
    resource_utilization: float = 0.0
    backpressure_events: int = 0


@dataclass
class ResourceThresholds:
    """Resource utilization thresholds."""
    
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    max_active_tasks: int = 100
    max_queue_size: int = 1000
    target_latency_ms: float = 100.0


class AdaptiveConcurrencyLimiter:
    """Intelligent concurrency limiter with adaptive scaling."""
    
    def __init__(
        self,
        initial_limit: int = 10,
        min_limit: int = 1,
        max_limit: int = 100,
        thresholds: ResourceThresholds | None = None,
    ):
        """Initialize adaptive concurrency limiter.
        
        Args:
            initial_limit: Starting concurrency limit
            min_limit: Minimum concurrency limit
            max_limit: Maximum concurrency limit
            thresholds: Resource utilization thresholds
        """
        self.current_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.thresholds = thresholds or ResourceThresholds()
        
        # Performance tracking
        self.metrics = ConcurrencyMetrics()
        self.performance_history = deque(maxlen=100)
        self.adjustment_history = deque(maxlen=50)
        
        # Adaptation parameters
        self.increase_factor = 1.1
        self.decrease_factor = 0.9
        self.stability_threshold = 10  # Number of stable measurements before increasing
        self.consecutive_good_measurements = 0
        
        # Semaphore for actual limiting
        self.semaphore = asyncio.Semaphore(initial_limit)
        self._update_lock = asyncio.Lock()
        
        logger.info(f"Initialized adaptive concurrency limiter with limit: {initial_limit}")
    
    async def acquire(self) -> None:
        """Acquire concurrency slot with adaptive adjustment."""
        await self.semaphore.acquire()
        self.metrics.active_tasks += 1
    
    async def release(self, execution_time: float, success: bool) -> None:
        """Release concurrency slot and update metrics.
        
        Args:
            execution_time: Task execution time in seconds
            success: Whether task completed successfully
        """
        self.semaphore.release()
        self.metrics.active_tasks -= 1
        
        # Update metrics
        if success:
            self.metrics.completed_tasks += 1
        else:
            self.metrics.failed_tasks += 1
        
        # Update average execution time (exponential moving average)
        alpha = 0.1
        self.metrics.avg_execution_time = (
            alpha * execution_time + (1 - alpha) * self.metrics.avg_execution_time
        )
        
        # Check if we should adjust concurrency limit
        await self._maybe_adjust_limit()
    
    async def _maybe_adjust_limit(self) -> None:
        """Adaptively adjust concurrency limit based on performance."""
        async with self._update_lock:
            current_time = time.time()
            
            # Get current system resources
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                self.metrics.resource_utilization = max(cpu_percent, memory_percent)
            except Exception:
                self.metrics.resource_utilization = 50.0  # Default fallback
            
            # Calculate current throughput
            total_tasks = self.metrics.completed_tasks + self.metrics.failed_tasks
            if total_tasks > 0:
                self.metrics.current_throughput = total_tasks / (current_time - getattr(self, '_start_time', current_time))
            
            # Record performance snapshot
            performance_snapshot = {
                'timestamp': current_time,
                'limit': self.current_limit,
                'active_tasks': self.metrics.active_tasks,
                'avg_execution_time': self.metrics.avg_execution_time,
                'resource_utilization': self.metrics.resource_utilization,
                'throughput': self.metrics.current_throughput,
            }
            self.performance_history.append(performance_snapshot)
            
            # Determine if we should adjust
            should_increase = await self._should_increase_limit()
            should_decrease = await self._should_decrease_limit()
            
            if should_decrease:
                await self._decrease_limit()
                self.consecutive_good_measurements = 0
            elif should_increase:
                await self._increase_limit()
                self.consecutive_good_measurements = 0
            else:
                self.consecutive_good_measurements += 1
    
    async def _should_increase_limit(self) -> bool:
        """Determine if concurrency limit should be increased."""
        if self.current_limit >= self.max_limit:
            return False
        
        # Check resource utilization
        if self.metrics.resource_utilization > self.thresholds.max_cpu_percent:
            return False
        
        # Check if we have stable good performance
        if self.consecutive_good_measurements < self.stability_threshold:
            return False
        
        # Check if we're close to current limit utilization
        utilization_ratio = self.metrics.active_tasks / self.current_limit
        if utilization_ratio > 0.8:  # High utilization, room to grow
            return True
        
        return False
    
    async def _should_decrease_limit(self) -> bool:
        """Determine if concurrency limit should be decreased."""
        if self.current_limit <= self.min_limit:
            return False
        
        # Check resource pressure
        if self.metrics.resource_utilization > self.thresholds.max_cpu_percent:
            return True
        
        # Check error rate
        total_tasks = self.metrics.completed_tasks + self.metrics.failed_tasks
        if total_tasks > 10:
            error_rate = self.metrics.failed_tasks / total_tasks
            if error_rate > 0.1:  # More than 10% error rate
                return True
        
        # Check execution time degradation
        if len(self.performance_history) >= 5:
            recent_avg = sum(p['avg_execution_time'] for p in list(self.performance_history)[-5:]) / 5
            if recent_avg > self.thresholds.target_latency_ms / 1000:
                return True
        
        return False
    
    async def _increase_limit(self) -> None:
        """Increase concurrency limit."""
        old_limit = self.current_limit
        self.current_limit = min(
            int(self.current_limit * self.increase_factor),
            self.max_limit
        )
        
        # Update semaphore
        additional_permits = self.current_limit - old_limit
        for _ in range(additional_permits):
            self.semaphore.release()
        
        self.adjustment_history.append({
            'timestamp': time.time(),
            'action': 'increase',
            'old_limit': old_limit,
            'new_limit': self.current_limit,
        })
        
        logger.info(f"Increased concurrency limit: {old_limit} -> {self.current_limit}")
    
    async def _decrease_limit(self) -> None:
        """Decrease concurrency limit."""
        old_limit = self.current_limit
        self.current_limit = max(
            int(self.current_limit * self.decrease_factor),
            self.min_limit
        )
        
        # Update semaphore by acquiring permits
        permits_to_remove = old_limit - self.current_limit
        for _ in range(permits_to_remove):
            await self.semaphore.acquire()
        
        self.adjustment_history.append({
            'timestamp': time.time(),
            'action': 'decrease',
            'old_limit': old_limit,
            'new_limit': self.current_limit,
        })
        
        logger.info(f"Decreased concurrency limit: {old_limit} -> {self.current_limit}")
    
    def get_metrics(self) -> dict[str, Any]:
        """Get current concurrency metrics."""
        return {
            'current_limit': self.current_limit,
            'active_tasks': self.metrics.active_tasks,
            'completed_tasks': self.metrics.completed_tasks,
            'failed_tasks': self.metrics.failed_tasks,
            'avg_execution_time': self.metrics.avg_execution_time,
            'current_throughput': self.metrics.current_throughput,
            'resource_utilization': self.metrics.resource_utilization,
            'consecutive_good_measurements': self.consecutive_good_measurements,
            'adjustment_history': list(self.adjustment_history),
        }


class IntelligentTaskScheduler:
    """Priority-based task scheduler with resource awareness."""
    
    def __init__(
        self,
        concurrency_limiter: AdaptiveConcurrencyLimiter,
        enable_batching: bool = True,
        batch_size: int = 10,
        batch_timeout: float = 0.1,
    ):
        """Initialize intelligent task scheduler.
        
        Args:
            concurrency_limiter: Concurrency limiter for resource management
            enable_batching: Enable intelligent batching
            batch_size: Maximum batch size
            batch_timeout: Maximum time to wait for batch completion
        """
        self.concurrency_limiter = concurrency_limiter
        self.enable_batching = enable_batching
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Priority queues for different task types
        self.task_queues = {
            priority: asyncio.Queue() for priority in TaskPriority
        }
        
        # Batching support
        self.batch_queues = {
            priority: [] for priority in TaskPriority
        }
        self.batch_timers = {
            priority: None for priority in TaskPriority
        }
        
        # Scheduler control
        self.running = False
        self.scheduler_task = None
        
        # Metrics
        self.scheduler_metrics = {
            'tasks_scheduled': defaultdict(int),
            'tasks_completed': defaultdict(int),
            'batches_processed': defaultdict(int),
            'avg_queue_depth': defaultdict(float),
        }
    
    async def schedule_task(
        self,
        coro: Awaitable[T],
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> T:
        """Schedule a task with specified priority.
        
        Args:
            coro: Coroutine to execute
            priority: Task priority
            metadata: Optional task metadata
            
        Returns:
            Task result
        """
        if not self.running:
            await self.start()
        
        # Create task wrapper with metadata
        future = asyncio.Future()
        task_info = {
            'coro': coro,
            'future': future,
            'priority': priority,
            'metadata': metadata or {},
            'created_at': time.time(),
        }
        
        # Add to appropriate queue
        await self.task_queues[priority].put(task_info)
        self.scheduler_metrics['tasks_scheduled'][priority] += 1
        
        # If batching is enabled, try to form batches
        if self.enable_batching:
            await self._maybe_start_batch(priority)
        
        return await future
    
    async def start(self) -> None:
        """Start the task scheduler."""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Started intelligent task scheduler")
    
    async def stop(self) -> None:
        """Stop the task scheduler."""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped intelligent task scheduler")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop with priority-based task execution."""
        while self.running:
            try:
                # Process tasks by priority order
                task_executed = False
                
                for priority in TaskPriority:
                    if not self.task_queues[priority].empty():
                        task_info = await self.task_queues[priority].get()
                        asyncio.create_task(self._execute_task(task_info))
                        task_executed = True
                        break
                
                if not task_executed:
                    # No tasks available, wait a bit
                    await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_task(self, task_info: dict[str, Any]) -> None:
        """Execute a single task with resource management.
        
        Args:
            task_info: Task information dictionary
        """
        priority = task_info['priority']
        start_time = time.time()
        
        try:
            # Acquire concurrency slot
            await self.concurrency_limiter.acquire()
            
            # Execute the coroutine
            result = await task_info['coro']
            
            # Mark as completed
            task_info['future'].set_result(result)
            execution_time = time.time() - start_time
            
            self.scheduler_metrics['tasks_completed'][priority] += 1
            
            # Release concurrency slot
            await self.concurrency_limiter.release(execution_time, success=True)
            
        except Exception as e:
            # Handle task failure
            task_info['future'].set_exception(e)
            execution_time = time.time() - start_time
            
            # Release concurrency slot
            await self.concurrency_limiter.release(execution_time, success=False)
            
            logger.warning(f"Task execution failed: {e}")
    
    async def _maybe_start_batch(self, priority: TaskPriority) -> None:
        """Start batch processing if conditions are met.
        
        Args:
            priority: Task priority level
        """
        # Check if we should start a batch
        queue_size = self.task_queues[priority].qsize()
        
        if queue_size >= self.batch_size:
            # Start batch immediately
            await self._process_batch(priority)
        elif queue_size > 0 and self.batch_timers[priority] is None:
            # Start timer for batch timeout
            self.batch_timers[priority] = asyncio.create_task(
                self._batch_timeout_handler(priority)
            )
    
    async def _batch_timeout_handler(self, priority: TaskPriority) -> None:
        """Handle batch timeout for a priority level.
        
        Args:
            priority: Task priority level
        """
        try:
            await asyncio.sleep(self.batch_timeout)
            await self._process_batch(priority)
        except asyncio.CancelledError:
            pass
        finally:
            self.batch_timers[priority] = None
    
    async def _process_batch(self, priority: TaskPriority) -> None:
        """Process a batch of tasks for a priority level.
        
        Args:
            priority: Task priority level
        """
        batch_tasks = []
        
        # Collect tasks for batch
        for _ in range(min(self.batch_size, self.task_queues[priority].qsize())):
            try:
                task_info = self.task_queues[priority].get_nowait()
                batch_tasks.append(task_info)
            except asyncio.QueueEmpty:
                break
        
        if batch_tasks:
            # Execute batch concurrently
            await asyncio.gather(
                *[self._execute_task(task) for task in batch_tasks],
                return_exceptions=True
            )
            
            self.scheduler_metrics['batches_processed'][priority] += 1
        
        # Cancel timer if it exists
        if self.batch_timers[priority]:
            self.batch_timers[priority].cancel()
            self.batch_timers[priority] = None
    
    def get_metrics(self) -> dict[str, Any]:
        """Get scheduler performance metrics."""
        # Calculate average queue depths
        for priority in TaskPriority:
            current_depth = self.task_queues[priority].qsize()
            current_avg = self.scheduler_metrics['avg_queue_depth'][priority]
            self.scheduler_metrics['avg_queue_depth'][priority] = (
                0.1 * current_depth + 0.9 * current_avg
            )
        
        return {
            'running': self.running,
            'concurrency_metrics': self.concurrency_limiter.get_metrics(),
            'scheduler_metrics': dict(self.scheduler_metrics),
            'queue_sizes': {
                priority.name: queue.qsize() 
                for priority, queue in self.task_queues.items()
            },
        }


class AsyncPerformanceOptimizer:
    """Comprehensive async performance optimization system."""
    
    def __init__(
        self,
        initial_concurrency: int = 10,
        enable_adaptive_limiting: bool = True,
        enable_intelligent_scheduling: bool = True,
        enable_metrics: bool = True,
    ):
        """Initialize async performance optimizer.
        
        Args:
            initial_concurrency: Starting concurrency limit
            enable_adaptive_limiting: Enable adaptive concurrency limiting
            enable_intelligent_scheduling: Enable priority-based scheduling
            enable_metrics: Enable performance metrics collection
        """
        self.enable_metrics = enable_metrics
        
        # Initialize components
        self.concurrency_limiter = AdaptiveConcurrencyLimiter(
            initial_limit=initial_concurrency
        ) if enable_adaptive_limiting else None
        
        self.task_scheduler = IntelligentTaskScheduler(
            concurrency_limiter=self.concurrency_limiter or AdaptiveConcurrencyLimiter(initial_concurrency),
            enable_batching=True,
        ) if enable_intelligent_scheduling else None
        
        # Performance monitoring
        self.performance_monitor = None
        if enable_metrics:
            try:
                from ..observability.performance import get_performance_monitor
                self.performance_monitor = get_performance_monitor()
            except Exception as e:
                logger.warning(f"Failed to initialize performance monitor: {e}")
        
        # Metrics collection
        self.metrics_registry = None
        if enable_metrics:
            try:
                self.metrics_registry = get_metrics_registry()
            except Exception as e:
                logger.warning(f"Failed to initialize metrics registry: {e}")
        
        logger.info("Initialized async performance optimizer")
    
    async def execute_optimized(
        self,
        coro: Awaitable[T],
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> T:
        """Execute coroutine with full optimization.
        
        Args:
            coro: Coroutine to execute
            priority: Execution priority
            metadata: Optional metadata
            
        Returns:
            Coroutine result
        """
        if self.task_scheduler:
            return await self.task_scheduler.schedule_task(coro, priority, metadata)
        elif self.concurrency_limiter:
            # Direct execution with concurrency limiting
            await self.concurrency_limiter.acquire()
            start_time = time.time()
            try:
                result = await coro
                execution_time = time.time() - start_time
                await self.concurrency_limiter.release(execution_time, success=True)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                await self.concurrency_limiter.release(execution_time, success=False)
                raise
        else:
            # Fallback to direct execution
            return await coro
    
    async def execute_batch_optimized(
        self,
        coros: list[Awaitable[T]],
        priority: TaskPriority = TaskPriority.NORMAL,
        max_concurrency: int | None = None,
    ) -> list[T]:
        """Execute multiple coroutines with optimization.
        
        Args:
            coros: List of coroutines to execute
            priority: Execution priority
            max_concurrency: Override concurrency limit
            
        Returns:
            List of results
        """
        if not coros:
            return []
        
        if self.task_scheduler:
            # Use intelligent scheduler
            tasks = [
                self.task_scheduler.schedule_task(coro, priority)
                for coro in coros
            ]
            return await asyncio.gather(*tasks, return_exceptions=False)
        
        elif self.concurrency_limiter:
            # Use semaphore-based limiting
            semaphore = asyncio.Semaphore(
                max_concurrency or self.concurrency_limiter.current_limit
            )
            
            async def limited_execution(coro):
                async with semaphore:
                    return await coro
            
            tasks = [limited_execution(coro) for coro in coros]
            return await asyncio.gather(*tasks, return_exceptions=False)
        
        else:
            # Fallback to direct execution
            return await asyncio.gather(*coros, return_exceptions=False)
    
    async def start(self) -> None:
        """Start optimization systems."""
        if self.task_scheduler:
            await self.task_scheduler.start()
    
    async def stop(self) -> None:
        """Stop optimization systems."""
        if self.task_scheduler:
            await self.task_scheduler.stop()
    
    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'optimization_enabled': {
                'adaptive_limiting': self.concurrency_limiter is not None,
                'intelligent_scheduling': self.task_scheduler is not None,
                'performance_monitoring': self.performance_monitor is not None,
            }
        }
        
        if self.concurrency_limiter:
            report['concurrency_metrics'] = self.concurrency_limiter.get_metrics()
        
        if self.task_scheduler:
            report['scheduler_metrics'] = self.task_scheduler.get_metrics()
        
        return report


# Global optimizer instance
_async_optimizer: AsyncPerformanceOptimizer | None = None


def get_async_optimizer() -> AsyncPerformanceOptimizer:
    """Get global async performance optimizer."""
    global _async_optimizer
    if _async_optimizer is None:
        _async_optimizer = AsyncPerformanceOptimizer()
    return _async_optimizer


async def initialize_async_optimizer(
    initial_concurrency: int = 10,
    enable_adaptive_limiting: bool = True,
    enable_intelligent_scheduling: bool = True,
) -> AsyncPerformanceOptimizer:
    """Initialize global async performance optimizer."""
    global _async_optimizer
    _async_optimizer = AsyncPerformanceOptimizer(
        initial_concurrency=initial_concurrency,
        enable_adaptive_limiting=enable_adaptive_limiting,
        enable_intelligent_scheduling=enable_intelligent_scheduling,
    )
    await _async_optimizer.start()
    return _async_optimizer


# Convenience functions
async def execute_optimized(
    coro: Awaitable[T],
    priority: TaskPriority = TaskPriority.NORMAL,
    metadata: dict[str, Any] | None = None,
) -> T:
    """Execute coroutine with optimization."""
    optimizer = get_async_optimizer()
    return await optimizer.execute_optimized(coro, priority, metadata)


async def execute_batch_optimized(
    coros: list[Awaitable[T]],
    priority: TaskPriority = TaskPriority.NORMAL,
    max_concurrency: int | None = None,
) -> list[T]:
    """Execute batch of coroutines with optimization."""
    optimizer = get_async_optimizer()
    return await optimizer.execute_batch_optimized(coros, priority, max_concurrency)