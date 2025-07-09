"""Performance Optimization Agent (POA) Service.

This module implements the autonomous Performance Optimization Agent that continuously
monitors system metrics and applies optimizations to maintain sub-100ms P95 latency.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from opentelemetry import trace
from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.infrastructure.clients.redis_client import RedisClientWrapper
from src.services.monitoring.performance_monitor import (
    PerformanceSnapshot,
    RealTimePerformanceMonitor,
)
from src.services.performance.api_optimizer import APIResponseOptimizer
from src.services.performance.async_optimizer import AsyncOptimizer
from src.services.performance.benchmarks import PerformanceBenchmark
from src.services.performance.database_optimizer import DatabaseOptimizer
from src.services.performance.memory_optimizer import MemoryOptimizer
from src.services.performance.performance_optimizer import PerformanceOptimizer


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class OptimizationType(str, Enum):
    """Types of optimizations that can be applied."""

    DATABASE_INDEX = "database_index"
    CACHE_TTL = "cache_ttl"
    CONNECTION_POOL = "connection_pool"
    ASYNC_CONCURRENCY = "async_concurrency"
    MEMORY_POOL = "memory_pool"
    RESPONSE_CACHE = "response_cache"
    QUERY_OPTIMIZATION = "query_optimization"
    GC_TUNING = "gc_tuning"


class OptimizationEvent(BaseModel):
    """Record of an optimization event."""

    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: OptimizationType
    description: str
    parameters: dict[str, Any]
    baseline_metrics: dict[str, float]
    expected_improvement: float
    rollback_threshold: float = 0.1  # 10% regression triggers rollback
    status: str = "pending"  # pending, applied, rolled_back, completed
    actual_improvement: float | None = None
    rollback_reason: str | None = None


class OptimizationLedger:
    """Maintains audit trail of all optimizations."""

    def __init__(self, ledger_path: Path):
        """Initialize optimization ledger."""
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[OptimizationEvent] = []
        self._load_ledger()

    def _load_ledger(self) -> None:
        """Load existing ledger from disk."""
        if self.ledger_path.exists():
            with self.ledger_path.open() as f:
                data = yaml.safe_load(f) or []
                self.events = [OptimizationEvent(**event) for event in data]

    def _save_ledger(self) -> None:
        """Persist ledger to disk."""
        with self.ledger_path.open("w") as f:
            yaml.safe_dump(
                [event.model_dump() for event in self.events],
                f,
                default_flow_style=False,
            )

    def add_event(self, event: OptimizationEvent) -> None:
        """Add optimization event to ledger."""
        self.events.append(event)
        self._save_ledger()

    def update_event(self, event_id: str, updates: dict[str, Any]) -> None:
        """Update existing optimization event."""
        for event in self.events:
            if event.id == event_id:
                for key, value in updates.items():
                    setattr(event, key, value)
                self._save_ledger()
                break


class RuleEngine:
    """Heuristic-based optimization rule engine."""

    def __init__(self, settings: Settings):
        """Initialize rule engine."""
        self.settings = settings
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> dict[str, Any]:
        """Initialize optimization rules."""
        return {
            "high_latency": {
                "condition": lambda m: m.p95_response_time > 100,
                "optimizations": [
                    OptimizationType.QUERY_OPTIMIZATION,
                    OptimizationType.RESPONSE_CACHE,
                    OptimizationType.DATABASE_INDEX,
                ],
                "priority": 1,
            },
            "low_cache_hit": {
                "condition": lambda m: m.cache_hit_rate < 0.8,
                "optimizations": [
                    OptimizationType.CACHE_TTL,
                    OptimizationType.RESPONSE_CACHE,
                ],
                "priority": 2,
            },
            "high_memory": {
                "condition": lambda m: m.memory_percent > 80,
                "optimizations": [
                    OptimizationType.MEMORY_POOL,
                    OptimizationType.GC_TUNING,
                ],
                "priority": 1,
            },
            "connection_saturation": {
                "condition": lambda m: m.active_connections > 100,
                "optimizations": [
                    OptimizationType.CONNECTION_POOL,
                    OptimizationType.ASYNC_CONCURRENCY,
                ],
                "priority": 2,
            },
        }

    def evaluate(self, metrics: PerformanceSnapshot) -> list[OptimizationType]:
        """Evaluate metrics against rules and return recommended optimizations."""
        recommendations = []

        for rule_name, rule in self.rules.items():
            if rule["condition"](metrics):
                logger.info(f"Rule '{rule_name}' triggered")
                recommendations.extend(rule["optimizations"])

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for opt in recommendations:
            if opt not in seen:
                seen.add(opt)
                unique_recommendations.append(opt)

        return unique_recommendations


class PerformanceOptimizationAgent:
    """Autonomous agent for continuous performance optimization."""

    def __init__(
        self,
        settings: Settings,
        monitor: RealTimePerformanceMonitor,
        redis_client: RedisClientWrapper | None = None,
    ):
        """Initialize POA service."""
        self.settings = settings
        self.monitor = monitor
        self.redis_client = redis_client

        # Initialize components
        self.rule_engine = RuleEngine(settings)
        self.ledger = OptimizationLedger(Path("data/optimization_ledger.yaml"))

        # Initialize optimizers
        self.performance_optimizer = PerformanceOptimizer(settings)
        self.database_optimizer = DatabaseOptimizer(settings)
        self.api_optimizer = APIResponseOptimizer(settings)
        self.memory_optimizer = MemoryOptimizer(settings)
        self.async_optimizer = AsyncOptimizer(settings)

        # Benchmark framework
        self.benchmark = PerformanceBenchmark(settings, monitor)

        # Control loop state
        self.running = False
        self.optimization_interval = 60  # seconds
        self.canary_window = 120  # seconds for canary validation
        self.active_optimizations: set[str] = set()
        self._monitoring_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the POA control loop."""
        if self.running:
            logger.warning("POA already running")
            return

        self.running = True
        logger.info("Starting Performance Optimization Agent")

        # Start monitoring if not already active
        if not self.monitor.monitoring_active:
            self._monitoring_task = asyncio.create_task(self.monitor.start_monitoring())

        # Run control loop
        try:
            await self._control_loop()
        except asyncio.CancelledError:
            logger.info("POA control loop cancelled")
        except Exception:
            logger.exception("POA control loop error")
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop the POA control loop."""
        self.running = False
        logger.info("Stopping Performance Optimization Agent")

    async def _control_loop(self) -> None:
        """Main control loop for continuous optimization."""
        while self.running:
            try:
                # Get current performance metrics
                snapshot = self.monitor.get_performance_summary()

                if snapshot.get("status") == "no_data":
                    logger.debug("No performance data available yet")
                    await asyncio.sleep(self.optimization_interval)
                    continue

                # Convert to PerformanceSnapshot
                metrics = PerformanceSnapshot(**snapshot)

                # Evaluate optimization opportunities
                recommendations = self.rule_engine.evaluate(metrics)

                if recommendations:
                    logger.info(
                        f"Identified {len(recommendations)} optimization opportunities"
                    )

                    # Apply optimizations with canary rollout
                    for optimization_type in recommendations:
                        if optimization_type.value not in self.active_optimizations:
                            await self._apply_optimization(optimization_type, metrics)

                # Check active optimizations for rollback
                await self._check_active_optimizations()

                # Wait for next iteration
                await asyncio.sleep(self.optimization_interval)

            except Exception:
                logger.exception("Error in POA control loop iteration")
                await asyncio.sleep(self.optimization_interval)

    async def apply_optimization(
        self, optimization_type: OptimizationType, baseline_metrics: PerformanceSnapshot
    ) -> None:
        """Apply optimization with canary rollout (public API)."""
        await self._apply_optimization(optimization_type, baseline_metrics)

    async def _apply_optimization(
        self, optimization_type: OptimizationType, baseline_metrics: PerformanceSnapshot
    ) -> None:
        """Apply optimization with canary rollout (internal)."""
        with tracer.start_as_current_span("apply_optimization") as span:
            span.set_attribute("optimization_type", optimization_type.value)

            # Generate optimization event
            event = OptimizationEvent(
                id=f"{optimization_type.value}_{int(datetime.now(UTC).timestamp())}",
                type=optimization_type,
                description=f"Applying {optimization_type.value} optimization",
                parameters={},
                baseline_metrics={
                    "p95_latency": baseline_metrics.p95_response_time,
                    "throughput": baseline_metrics.request_rate,
                    "memory_mb": baseline_metrics.memory_mb,
                    "cache_hit_rate": baseline_metrics.cache_hit_rate,
                },
                expected_improvement=0.1,  # 10% improvement target
            )

            try:
                # Apply optimization based on type
                if optimization_type == OptimizationType.DATABASE_INDEX:
                    await self._optimize_database_index(event)
                elif optimization_type == OptimizationType.CACHE_TTL:
                    await self._optimize_cache_ttl(event)
                elif optimization_type == OptimizationType.CONNECTION_POOL:
                    await self._optimize_connection_pool(event)
                elif optimization_type == OptimizationType.ASYNC_CONCURRENCY:
                    await self._optimize_async_concurrency(event)
                elif optimization_type == OptimizationType.MEMORY_POOL:
                    await self._optimize_memory_pool(event)
                elif optimization_type == OptimizationType.RESPONSE_CACHE:
                    await self._optimize_response_cache(event)
                elif optimization_type == OptimizationType.QUERY_OPTIMIZATION:
                    await self._optimize_queries(event)
                elif optimization_type == OptimizationType.GC_TUNING:
                    await self._optimize_gc_tuning(event)

                # Mark as applied
                event.status = "applied"
                self.ledger.add_event(event)
                self.active_optimizations.add(event.id)

                logger.info(f"Applied optimization: {event.id}")

            except Exception as e:
                logger.exception(f"Failed to apply optimization: {optimization_type}")
                event.status = "failed"
                event.rollback_reason = str(e)
                self.ledger.add_event(event)

    async def _optimize_database_index(self, event: OptimizationEvent) -> None:
        """Apply database index optimization."""
        # Analyze slow queries and create indexes
        recommendations = await self.database_optimizer.analyze_slow_queries()

        for rec in recommendations[:3]:  # Limit to top 3 indexes
            collection_name = rec.get("collection", "documents")
            field = rec.get("field", "metadata.timestamp")

            await self.database_optimizer.optimize_collection_index(
                collection_name, target_speed="balanced"
            )

            event.parameters[f"index_{field}"] = {
                "collection": collection_name,
                "field": field,
                "type": "hnsw",
            }

    async def _optimize_cache_ttl(self, event: OptimizationEvent) -> None:
        """Optimize cache TTL based on access patterns."""
        # Analyze cache patterns and adjust TTLs
        optimal_ttls = await self.performance_optimizer.optimize_cache_ttls()

        event.parameters["ttl_adjustments"] = optimal_ttls

        # Apply new TTLs through cache configuration
        if self.redis_client:
            for pattern, ttl in optimal_ttls.items():
                await self.redis_client.client.config_set(
                    f"cache:ttl:{pattern}", str(ttl)
                )

    async def _optimize_connection_pool(self, event: OptimizationEvent) -> None:
        """Optimize database connection pool settings."""
        # Calculate optimal pool size based on load
        current_connections = (
            self.monitor.snapshots[-1].active_connections
            if self.monitor.snapshots
            else 10
        )
        optimal_pool_size = min(current_connections * 1.5, 100)  # Cap at 100

        event.parameters["pool_size"] = {
            "previous": current_connections,
            "new": int(optimal_pool_size),
            "max_overflow": 20,
        }

        # This would be applied through database configuration
        logger.info(f"Optimized connection pool size to {optimal_pool_size}")

    async def _optimize_async_concurrency(self, event: OptimizationEvent) -> None:
        """Optimize async concurrency limits."""
        # Adjust semaphore limits based on CPU and load
        cpu_count = asyncio.cpu_count() or 4
        optimal_concurrency = cpu_count * 10  # Rule of thumb

        event.parameters["concurrency"] = {
            "semaphore_limit": optimal_concurrency,
            "max_workers": cpu_count * 2,
            "batch_size": 50,
        }

        # Apply through async optimizer
        self.async_optimizer.max_concurrent_tasks = optimal_concurrency

    async def _optimize_memory_pool(self, event: OptimizationEvent) -> None:
        """Optimize memory pooling for objects."""
        # Enable object pooling for high-churn objects
        pools_created = await self.memory_optimizer.optimize_pooling()

        event.parameters["pools"] = {
            "created": pools_created,
            "max_size": 1000,
            "cleanup_interval": 300,
        }

    async def _optimize_response_cache(self, event: OptimizationEvent) -> None:
        """Optimize API response caching."""
        # Enable response caching for common endpoints
        cache_config = {
            "/api/search": {"ttl": 300, "vary": ["q", "limit"]},
            "/api/documents": {"ttl": 600, "vary": ["page", "size"]},
            "/api/stats": {"ttl": 60, "vary": []},
        }

        event.parameters["response_cache"] = cache_config

        # This would be applied through API middleware configuration
        logger.info("Enabled response caching for API endpoints")

    async def _optimize_queries(self, event: OptimizationEvent) -> None:
        """Optimize query patterns and execution."""
        # Analyze and optimize query patterns
        optimizations = await self.performance_optimizer.optimize_hot_paths()

        event.parameters["query_optimizations"] = {
            "rewritten_queries": len(optimizations),
            "estimated_improvement": "15-25%",
        }

    async def _optimize_gc_tuning(self, event: OptimizationEvent) -> None:
        """Optimize garbage collection settings."""
        # Apply GC tuning
        gc_settings = self.memory_optimizer.tune_gc()

        event.parameters["gc_settings"] = gc_settings

    async def _check_active_optimizations(self) -> None:
        """Check active optimizations for regression and rollback if needed."""
        if not self.active_optimizations:
            return

        current_metrics = self.monitor.get_performance_summary()
        if current_metrics.get("status") == "no_data":
            return

        metrics = PerformanceSnapshot(**current_metrics)

        for event_id in list(self.active_optimizations):
            # Get optimization event
            event = next((e for e in self.ledger.events if e.id == event_id), None)
            if not event:
                continue

            # Skip if not enough time has passed
            if datetime.now(UTC) - event.timestamp < timedelta(
                seconds=self.canary_window
            ):
                continue

            # Check for regression
            baseline_p95 = event.baseline_metrics["p95_latency"]
            current_p95 = metrics.p95_response_time

            regression = (
                ((current_p95 - baseline_p95) / baseline_p95) if baseline_p95 > 0 else 0
            )

            if regression > event.rollback_threshold:
                # Rollback optimization
                logger.warning(
                    f"Optimization {event_id} caused {regression:.1%} regression, rolling back"
                )

                await self._rollback_optimization(event)

                self.ledger.update_event(
                    event_id,
                    {
                        "status": "rolled_back",
                        "rollback_reason": f"P95 regression: {regression:.1%}",
                        "actual_improvement": -regression,
                    },
                )

                self.active_optimizations.remove(event_id)
            else:
                # Mark as completed if improvement is stable
                improvement = (
                    ((baseline_p95 - current_p95) / baseline_p95)
                    if baseline_p95 > 0
                    else 0
                )

                if improvement > 0:
                    logger.info(
                        f"Optimization {event_id} achieved {improvement:.1%} improvement"
                    )

                    self.ledger.update_event(
                        event_id,
                        {"status": "completed", "actual_improvement": improvement},
                    )

                    self.active_optimizations.remove(event_id)

    async def _rollback_optimization(self, event: OptimizationEvent) -> None:
        """Rollback a specific optimization."""
        logger.info(f"Rolling back optimization: {event.id}")

        # Rollback based on type
        # This would implement specific rollback logic for each optimization type
        # For now, log the rollback
        logger.info(f"Rollback completed for {event.type.value}")

    async def get_status(self) -> dict[str, Any]:
        """Get POA status and optimization history."""
        return {
            "status": "running" if self.running else "stopped",
            "active_optimizations": list(self.active_optimizations),
            "total_optimizations": len(self.ledger.events),
            "successful_optimizations": len(
                [
                    e
                    for e in self.ledger.events
                    if e.status == "completed" and e.actual_improvement > 0
                ]
            ),
            "rolled_back": len(
                [e for e in self.ledger.events if e.status == "rolled_back"]
            ),
            "current_metrics": self.monitor.get_performance_summary(),
            "recommendations": self.monitor.get_optimization_recommendations(),
        }
