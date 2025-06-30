"""Infrastructure automation for self-healing and auto-scaling.

This module provides:
- Self-healing database connections and services
- Automatic resource scaling
- Circuit breakers with adaptive thresholds
- Error recovery and retry mechanisms
"""

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"


@dataclass
class SystemMetrics:
    """System resource metrics."""

    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: dict[str, int]
    connection_count: int
    error_rate: float
    response_time_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""

    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "closed"  # closed, open, half_open
    success_count: int = 0
    total_requests: int = 0

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failure_count / self.total_requests


class AdaptiveCircuitBreaker:
    """Circuit breaker with ML-based threshold adaptation."""

    def __init__(
        self,
        name: str,
        initial_failure_threshold: float = 0.5,
        initial_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = initial_failure_threshold
        self.timeout = initial_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitBreakerState()
        self.historical_data: list[dict[str, Any]] = []

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self._should_reject():
            msg = f"Circuit breaker {self.name} is open"
            raise CircuitBreakerOpenError(msg)

        try:
            start_time = time.time()
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # Record success
            self._record_success(time.time() - start_time)
            return result

        except Exception as e:
            # Record failure
            self._record_failure(time.time() - start_time)
            raise

    def _should_reject(self) -> bool:
        """Determine if request should be rejected."""
        if self.state.state == "open":
            if time.time() - self.state.last_failure_time > self.timeout:
                self.state.state = "half_open"
                self.state.success_count = 0
                return False
            return True

        if self.state.state == "half_open":
            return self.state.success_count >= self.half_open_max_calls

        return False

    def _record_success(self, response_time: float):
        """Record successful execution."""
        self.state.success_count += 1
        self.state.total_requests += 1

        if (
            self.state.state == "half_open"
            and self.state.success_count >= self.half_open_max_calls
        ):
            self.state.state = "closed"
            self.state.failure_count = 0

        # Adapt thresholds based on success patterns
        self._adapt_thresholds(success=True, response_time=response_time)

    def _record_failure(self, response_time: float):
        """Record failed execution."""
        self.state.failure_count += 1
        self.state.total_requests += 1
        self.state.last_failure_time = time.time()

        if self.state.failure_rate >= self.failure_threshold:
            self.state.state = "open"

        # Adapt thresholds based on failure patterns
        self._adapt_thresholds(success=False, response_time=response_time)

    def _adapt_thresholds(self, success: bool, response_time: float):
        """Adapt circuit breaker thresholds based on historical data."""
        # Record data point
        data_point = {
            "timestamp": time.time(),
            "success": success,
            "response_time": response_time,
            "failure_rate": self.state.failure_rate,
        }
        self.historical_data.append(data_point)

        # Keep only recent data (last 1000 points)
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]

        # Adapt thresholds every 100 requests
        if len(self.historical_data) % 100 == 0:
            self._optimize_thresholds()

    def _optimize_thresholds(self):
        """Optimize thresholds based on historical performance."""
        if len(self.historical_data) < 50:
            return

        recent_data = self.historical_data[-50:]

        # Calculate metrics
        avg_response_time = sum(d["response_time"] for d in recent_data) / len(
            recent_data
        )
        failure_rate = sum(1 for d in recent_data if not d["success"]) / len(
            recent_data
        )

        # Adapt failure threshold
        if failure_rate < 0.1 and avg_response_time < 1.0:
            # System is healthy, can be more lenient
            self.failure_threshold = min(self.failure_threshold * 1.1, 0.8)
        elif failure_rate > 0.3 or avg_response_time > 5.0:
            # System is struggling, be more strict
            self.failure_threshold = max(self.failure_threshold * 0.9, 0.1)

        # Adapt timeout based on recovery patterns
        if (
            self.state.state == "closed"
            and len([d for d in recent_data if d["success"]]) > 40
        ):
            # Fast recovery, can reduce timeout
            self.timeout = max(self.timeout * 0.9, 10.0)
        elif failure_rate > 0.5:
            # Slow recovery, increase timeout
            self.timeout = min(self.timeout * 1.2, 300.0)


class SelfHealingDatabaseManager:
    """Database manager with automatic error recovery and connection healing."""

    def __init__(self, database_url: str, **engine_kwargs):
        self.database_url = database_url
        self.engine_kwargs = engine_kwargs
        self.engine: AsyncEngine | None = None

        self.circuit_breaker = AdaptiveCircuitBreaker("database")
        self.connection_count = 0
        self.error_count = 0
        self.last_health_check = 0

        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 60.0

    async def initialize(self):
        """Initialize database engine with self-healing capabilities."""
        try:
            self.engine = await self._create_engine_with_retry()
            logger.info("Database connection initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize database")
            # Start background recovery
            asyncio.create_task(self._background_recovery())

    async def _create_engine_with_retry(self) -> AsyncEngine:
        """Create database engine with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                engine = create_async_engine(
                    self.database_url,
                    pool_pre_ping=True,  # Essential for connection health
                    pool_recycle=3600,  # Recycle connections hourly
                    **self.engine_kwargs,
                )

                # Test connection
                async with engine.begin() as conn:
                    await conn.execute("SELECT 1")

                return engine

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise

                delay = min(self.base_delay * (2**attempt), self.max_delay)
                logger.warning(
                    f"Database connection attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)

        msg = "Failed to establish database connection after retries"
        raise RuntimeError(msg)

    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic error recovery."""
        if not self.engine:
            await self._attempt_recovery()

        session = None
        try:
            # Use circuit breaker protection
            session = await self.circuit_breaker.call(self._create_session)
            self.connection_count += 1

            yield session

        except Exception as e:
            self.error_count += 1
            logger.exception("Database session error")

            # Attempt automatic recovery
            if self._should_attempt_recovery():
                await self._attempt_recovery()

            raise

        finally:
            if session:
                await session.close()

    async def _create_session(self) -> AsyncSession:
        """Create a new database session."""
        if not self.engine:
            msg = "Database engine not available"
            raise DatabaseNotAvailableError(msg)

        return AsyncSession(self.engine)

    def _should_attempt_recovery(self) -> bool:
        """Determine if automatic recovery should be attempted."""
        # Avoid recovery loops
        time_since_last_check = time.time() - self.last_health_check
        return time_since_last_check > 30  # At most once per 30 seconds

    async def _attempt_recovery(self):
        """Attempt to recover database connection."""
        self.last_health_check = time.time()
        logger.info("Attempting database recovery")

        try:
            # Close existing engine if unhealthy
            if self.engine:
                await self.engine.dispose()

            # Recreate engine
            self.engine = await self._create_engine_with_retry()
            logger.info("Database recovery successful")

        except Exception as e:
            logger.exception("Database recovery failed")
            # Schedule retry
            asyncio.create_task(self._delayed_recovery())

    async def _delayed_recovery(self):
        """Attempt recovery after a delay."""
        await asyncio.sleep(60)  # Wait 1 minute before retry
        await self._attempt_recovery()

    async def _background_recovery(self):
        """Background task for continuous recovery attempts."""
        while True:
            if not self.engine:
                await self._attempt_recovery()

            await asyncio.sleep(120)  # Check every 2 minutes

    async def health_check(self) -> HealthStatus:
        """Check database health status."""
        if not self.engine:
            return HealthStatus.UNHEALTHY

        try:
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")

            failure_rate = self.circuit_breaker.state.failure_rate
            if failure_rate < 0.1:
                return HealthStatus.HEALTHY
            if failure_rate < 0.5:
                return HealthStatus.DEGRADED
            return HealthStatus.UNHEALTHY

        except Exception:
            return HealthStatus.UNHEALTHY


class AutoScalingManager:
    """Automatically scales system resources based on demand."""

    def __init__(self):
        self.metrics_history: list[SystemMetrics] = []
        self.scaling_enabled = True
        self.last_scale_action = 0
        self.min_scale_interval = 300  # 5 minutes between scaling actions

        # Scaling thresholds
        self.cpu_scale_up_threshold = 70.0
        self.cpu_scale_down_threshold = 30.0
        self.memory_scale_up_threshold = 80.0
        self.memory_scale_down_threshold = 40.0

    async def start_monitoring(self, check_interval: float = 60):
        """Start automatic resource monitoring and scaling."""
        logger.info("Starting auto-scaling monitoring")

        while self.scaling_enabled:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)

                # Keep only recent metrics (last 100 points)
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]

                # Check if scaling is needed
                await self._evaluate_scaling(metrics)

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.exception("Auto-scaling monitoring error")
                await asyncio.sleep(30)  # Shorter interval on error

    async def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk metrics
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100

        # Network metrics
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
        }

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_io=network_io,
            connection_count=self._get_connection_count(),
            error_rate=self._calculate_error_rate(),
            response_time_ms=self._get_avg_response_time(),
        )

    def _get_connection_count(self) -> int:
        """Get current connection count."""
        # Placeholder - would integrate with actual connection tracking
        return 0

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        # Placeholder - would integrate with actual error tracking
        return 0.0

    def _get_avg_response_time(self) -> float:
        """Get average response time."""
        # Placeholder - would integrate with actual performance tracking
        return 0.0

    async def _evaluate_scaling(self, metrics: SystemMetrics):
        """Evaluate if scaling action is needed."""
        # Prevent frequent scaling
        if time.time() - self.last_scale_action < self.min_scale_interval:
            return

        # Check for scale-up conditions
        if (
            metrics.cpu_percent > self.cpu_scale_up_threshold
            or metrics.memory_percent > self.memory_scale_up_threshold
        ):
            await self._scale_up(metrics)

        # Check for scale-down conditions
        elif (
            metrics.cpu_percent < self.cpu_scale_down_threshold
            and metrics.memory_percent < self.memory_scale_down_threshold
        ):
            await self._scale_down(metrics)

    async def _scale_up(self, metrics: SystemMetrics):
        """Scale up system resources."""
        logger.info(
            f"Scaling up - CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%"
        )

        # In a real implementation, this would:
        # - Increase worker processes
        # - Expand connection pools
        # - Request additional compute resources
        # - Update load balancer configuration

        self.last_scale_action = time.time()

    async def _scale_down(self, metrics: SystemMetrics):
        """Scale down system resources."""
        logger.info(
            f"Scaling down - CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%"
        )

        # In a real implementation, this would:
        # - Reduce worker processes
        # - Shrink connection pools
        # - Release unused compute resources
        # - Update load balancer configuration

        self.last_scale_action = time.time()

    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.scaling_enabled = False
        logger.info("Auto-scaling monitoring stopped")


class SelfHealingManager:
    """Coordinating manager for all self-healing infrastructure components."""

    def __init__(self):
        self.database_manager: SelfHealingDatabaseManager | None = None
        self.scaling_manager: AutoScalingManager | None = None
        self.circuit_breakers: dict[str, AdaptiveCircuitBreaker] = {}

    async def initialize(self, database_url: str):
        """Initialize all self-healing components."""
        # Initialize database manager
        self.database_manager = SelfHealingDatabaseManager(database_url)
        await self.database_manager.initialize()

        # Initialize auto-scaling
        self.scaling_manager = AutoScalingManager()
        asyncio.create_task(self.scaling_manager.start_monitoring())

        logger.info("Self-healing infrastructure initialized")

    def get_circuit_breaker(self, name: str) -> AdaptiveCircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = AdaptiveCircuitBreaker(name)
        return self.circuit_breakers[name]

    async def health_check(self) -> dict[str, HealthStatus]:
        """Get health status of all managed components."""
        health_status = {}

        if self.database_manager:
            health_status["database"] = await self.database_manager.health_check()

        # Check circuit breaker states
        for name, breaker in self.circuit_breakers.items():
            if breaker.state.state == "open":
                health_status[f"circuit_breaker_{name}"] = HealthStatus.UNHEALTHY
            elif breaker.state.failure_rate > 0.3:
                health_status[f"circuit_breaker_{name}"] = HealthStatus.DEGRADED
            else:
                health_status[f"circuit_breaker_{name}"] = HealthStatus.HEALTHY

        return health_status


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""


class DatabaseNotAvailableError(Exception):
    """Raised when database is not available."""


# Global instance
_self_healing_manager: SelfHealingManager | None = None


async def get_self_healing_manager() -> SelfHealingManager:
    """Get the global self-healing manager instance."""
    global _self_healing_manager

    if _self_healing_manager is None:
        _self_healing_manager = SelfHealingManager()

    return _self_healing_manager
