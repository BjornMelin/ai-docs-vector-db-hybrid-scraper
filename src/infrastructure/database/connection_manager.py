"""Async database connection manager with dynamic pool optimization.

This module provides a comprehensive database connection management system
with dynamic pool sizing, health checks, and performance monitoring.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine

from ...config.models import SQLAlchemyConfig
from ...infrastructure.shared import CircuitBreaker
from .load_monitor import LoadMonitor
from .load_monitor import LoadMonitorConfig
from .query_monitor import QueryMonitor
from .query_monitor import QueryMonitorConfig

logger = logging.getLogger(__name__)


class AsyncConnectionManager:
    """Manages async database connections with dynamic pool optimization.

    This class provides:
    - Dynamic connection pool sizing based on load metrics
    - Health checks with automatic reconnection
    - Query performance monitoring
    - Circuit breaker integration for fault tolerance
    - Comprehensive metrics and monitoring
    """

    def __init__(
        self,
        config: SQLAlchemyConfig,
        load_monitor: LoadMonitor | None = None,
        query_monitor: QueryMonitor | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        """Initialize the connection manager.

        Args:
            config: SQLAlchemy configuration
            load_monitor: Optional load monitor (creates default if None)
            query_monitor: Optional query monitor (creates default if None)
            circuit_breaker: Optional circuit breaker (creates default if None)
        """
        self.config = config
        self.load_monitor = load_monitor or LoadMonitor(LoadMonitorConfig())
        self.query_monitor = query_monitor or QueryMonitor(
            QueryMonitorConfig(
                enabled=config.enable_query_monitoring,
                slow_query_threshold_ms=config.slow_query_threshold_ms,
            )
        )

        # Circuit breaker for fault tolerance
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=5, recovery_timeout=60.0, half_open_requests=1
        )

        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._is_initialized = False
        self._health_check_task: asyncio.Task[Any] | None = None
        self._metrics_task: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()

        # Performance tracking
        self._total_connections_created = 0
        self._total_connection_errors = 0
        self._last_pool_size_adjustment = 0.0
        self._current_pool_size = config.pool_size

    async def initialize(self) -> None:
        """Initialize the connection manager."""
        if self._is_initialized:
            return

        async with self._lock:
            if self._is_initialized:
                return

            try:
                # Create the async engine with initial configuration
                await self._create_engine()

                # Start monitoring tasks
                await self.load_monitor.start()

                self._health_check_task = asyncio.create_task(self._health_check_loop())
                self._metrics_task = asyncio.create_task(self._metrics_loop())

                self._is_initialized = True
                logger.info(
                    f"AsyncConnectionManager initialized with pool_size={self._current_pool_size}"
                )

            except Exception as e:
                logger.error(f"Failed to initialize AsyncConnectionManager: {e}")
                await self._cleanup()
                raise

    async def shutdown(self) -> None:
        """Shutdown the connection manager."""
        async with self._lock:
            if not self._is_initialized:
                return

            await self._cleanup()
            self._is_initialized = False
            logger.info("AsyncConnectionManager shutdown complete")

    async def cleanup(self) -> None:
        """Cleanup method for compatibility with ClientManager pattern."""
        await self.shutdown()

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession]:
        """Get a database session with monitoring and error handling.

        Yields:
            AsyncSession: Database session

        Raises:
            RuntimeError: If not initialized
            Exception: Database connection errors
        """
        if not self._is_initialized:
            raise RuntimeError("AsyncConnectionManager not initialized")

        # Record request start for load monitoring
        await self.load_monitor.record_request_start()

        start_time = time.time()
        session = None

        try:
            # Create session using circuit breaker protection
            if not self._session_factory:
                raise RuntimeError("Session factory not initialized")

            # Use circuit breaker to protect session creation and connection test
            async def create_and_test_session():
                session = self._session_factory()
                # Test connection with a simple query
                await session.execute(text("SELECT 1"))
                return session

            session = await self.circuit_breaker.call(create_and_test_session)

            yield session

        except Exception as e:
            logger.error(f"Database session error: {e}")

            # Record failure for load monitoring
            await self.load_monitor.record_connection_error()
            self._total_connection_errors += 1

            if session:
                try:
                    await session.rollback()
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback session: {rollback_error}")

            raise

        finally:
            # Clean up session
            if session:
                try:
                    await session.close()
                except Exception as close_error:
                    logger.error(f"Failed to close session: {close_error}")

            # Record request end
            response_time_ms = (time.time() - start_time) * 1000
            await self.load_monitor.record_request_end(response_time_ms)

    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute a query with monitoring.

        Args:
            query: SQL query string
            parameters: Query parameters

        Returns:
            Query result
        """
        query_id = await self.query_monitor.start_query(query)

        try:
            async with self.get_session() as session:
                # Wrap string queries in text() for SQLAlchemy
                sql_query = text(query) if isinstance(query, str) else query
                result = await session.execute(sql_query, parameters or {})
                await session.commit()

                # Record successful query
                execution_time = await self.query_monitor.end_query(
                    query_id, query, success=True
                )
                logger.debug(
                    f"Query executed in {execution_time:.2f}ms: {query[:100]}..."
                )

                return result

        except Exception as e:
            # Record failed query
            await self.query_monitor.end_query(query_id, query, success=False)
            logger.error(f"Query execution failed: {e}")
            raise

    async def get_connection_stats(self) -> dict[str, Any]:
        """Get connection pool statistics.

        Returns:
            Dictionary with connection statistics
        """
        if not self._engine:
            return {}

        pool = self._engine.pool
        load_metrics = await self.load_monitor.get_current_load()
        query_stats = await self.query_monitor.get_summary_stats()

        # Build base stats
        stats = {
            "pool_size": self._current_pool_size,
            "total_connections_created": self._total_connections_created,
            "total_connection_errors": self._total_connection_errors,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_failures": self.circuit_breaker._failure_count,
            "load_metrics": {
                "concurrent_requests": load_metrics.concurrent_requests,
                "memory_usage_percent": load_metrics.memory_usage_percent,
                "avg_response_time_ms": load_metrics.avg_response_time_ms,
                "connection_errors": load_metrics.connection_errors,
            },
            "query_stats": query_stats,
        }

        # Add pool-specific stats if available (not available for SQLite's StaticPool)
        if hasattr(pool, "checkedin"):
            stats.update(
                {
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalidated": pool.invalidated(),
                }
            )
        else:
            # For SQLite or other pools without these methods
            stats.update(
                {
                    "checked_in": 0,
                    "checked_out": 0,
                    "overflow": 0,
                    "invalidated": 0,
                }
            )

        return stats

    async def _create_engine(self) -> None:
        """Create the SQLAlchemy async engine."""
        # Check if we're using SQLite which doesn't support connection pooling
        is_sqlite = "sqlite" in self.config.database_url.lower()

        if is_sqlite:
            logger.info("Creating SQLite engine (no pooling)")
            self._engine = create_async_engine(
                self.config.database_url,
                echo=self.config.echo_queries,
                future=True,
            )
            pool_size = self.config.pool_size  # Use default pool size for tracking
        else:
            pool_size = self._calculate_pool_size()
            max_overflow = self._calculate_max_overflow()

            logger.info(
                f"Creating engine with pool_size={pool_size}, max_overflow={max_overflow}"
            )

            self._engine = create_async_engine(
                self.config.database_url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo_queries,
                future=True,
            )

        # Create session factory
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

        self._current_pool_size = pool_size
        self._total_connections_created += 1

    def _calculate_pool_size(self) -> int:
        """Calculate optimal pool size based on current load.

        Returns:
            Optimal pool size
        """
        if not self.config.adaptive_pool_sizing:
            return self.config.pool_size

        # Get current load factor (0.0 to 1.0)
        load_factor = 0.0
        try:
            # This might fail if load monitor isn't started yet
            current_load = asyncio.create_task(self.load_monitor.get_current_load())
            if current_load.done():
                load_metrics = current_load.result()
                load_factor = self.load_monitor.calculate_load_factor(load_metrics)
        except Exception:
            # Use default if load monitoring not available
            pass

        # Calculate pool size based on load factor
        base_size = self.config.min_pool_size
        max_size = min(self.config.max_pool_size, self.config.pool_size * 2)

        # Scale between min and max based on load
        pool_size = int(base_size + (max_size - base_size) * load_factor)

        # Apply growth factor for gradual scaling
        if pool_size > self._current_pool_size:
            growth = int(
                (pool_size - self._current_pool_size) * self.config.pool_growth_factor
            )
            pool_size = min(self._current_pool_size + max(1, growth), max_size)

        return max(self.config.min_pool_size, min(pool_size, max_size))

    def _calculate_max_overflow(self) -> int:
        """Calculate max overflow based on pool size.

        Returns:
            Max overflow connections
        """
        # Scale overflow with pool size, but cap it
        base_overflow = self.config.max_overflow
        scale_factor = self._current_pool_size / self.config.pool_size
        scaled_overflow = int(base_overflow * scale_factor)

        return min(scaled_overflow, base_overflow * 2)

    async def _health_check_loop(self) -> None:
        """Periodic health checks for the database connection."""
        while self._is_initialized:
            try:
                if self._engine:
                    # Test connection health using circuit breaker
                    async def health_check():
                        async with self._engine.begin() as conn:
                            await conn.execute(text("SELECT 1"))

                    await self.circuit_breaker.call(health_check)

                await asyncio.sleep(30.0)  # Health check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                await self.load_monitor.record_connection_error()
                await asyncio.sleep(10.0)  # Shorter retry interval on failure

    async def _metrics_loop(self) -> None:
        """Periodic metrics collection and pool optimization."""
        while self._is_initialized:
            try:
                # Check if we need to adjust pool size
                if self.config.adaptive_pool_sizing:
                    await self._maybe_adjust_pool_size()

                # Clean up old query stats
                await self.query_monitor.cleanup_old_stats()

                await asyncio.sleep(60.0)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(30.0)

    async def _maybe_adjust_pool_size(self) -> None:
        """Adjust pool size if needed based on current load."""
        now = time.time()

        # Don't adjust too frequently
        if now - self._last_pool_size_adjustment < 300:  # 5 minutes
            return

        optimal_size = self._calculate_pool_size()

        # Only recreate engine if size change is significant
        if abs(optimal_size - self._current_pool_size) >= 2:
            logger.info(
                f"Adjusting pool size from {self._current_pool_size} to {optimal_size}"
            )

            try:
                # Create new engine with updated pool size
                old_engine = self._engine
                await self._create_engine()

                # Dispose of old engine
                if old_engine:
                    await old_engine.dispose()

                self._last_pool_size_adjustment = now

            except Exception as e:
                logger.error(f"Failed to adjust pool size: {e}")

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cancel monitoring tasks
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            if self._metrics_task:
                self._metrics_task.cancel()
                try:
                    await self._metrics_task
                except asyncio.CancelledError:
                    pass

            # Stop load monitor
            await self.load_monitor.stop()

            # Dispose of engine
            if self._engine:
                await self._engine.dispose()
                self._engine = None

            self._session_factory = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
