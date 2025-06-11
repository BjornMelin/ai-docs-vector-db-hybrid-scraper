"""Async database connection manager with dynamic pool optimization.

This module provides a comprehensive database connection management system
with dynamic pool sizing, health checks, and performance monitoring.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextlib import suppress
from typing import Any
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine

from ...config.models import SQLAlchemyConfig
from ...infrastructure.shared import CircuitBreaker
from .adaptive_config import AdaptiveConfigManager
from .adaptive_config import AdaptationStrategy
from .connection_affinity import ConnectionAffinityManager
from .connection_affinity import QueryType
from .enhanced_circuit_breaker import FailureType
from .enhanced_circuit_breaker import MultiLevelCircuitBreaker
from .load_monitor import LoadMonitor
from .load_monitor import LoadMonitorConfig
from .predictive_monitor import PredictiveLoadMonitor
from .query_monitor import QueryMonitor
from .query_monitor import QueryMonitorConfig

logger = logging.getLogger(__name__)


class AsyncConnectionManager:
    """Manages async database connections with advanced optimization features.

    This enhanced class provides:
    - Dynamic connection pool sizing based on predictive load metrics
    - Health checks with automatic reconnection
    - Query performance monitoring and pattern optimization
    - Multi-level circuit breaker with failure categorization
    - Connection affinity for query pattern optimization
    - Adaptive configuration based on system conditions
    - Comprehensive metrics and monitoring
    """

    def __init__(
        self,
        config: SQLAlchemyConfig,
        load_monitor: LoadMonitor | None = None,
        query_monitor: QueryMonitor | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        enable_predictive_monitoring: bool = True,
        enable_connection_affinity: bool = True,
        enable_adaptive_config: bool = True,
        adaptation_strategy: AdaptationStrategy = AdaptationStrategy.MODERATE,
    ):
        """Initialize the enhanced connection manager.

        Args:
            config: SQLAlchemy configuration
            load_monitor: Optional load monitor (creates default if None)
            query_monitor: Optional query monitor (creates default if None)
            circuit_breaker: Optional circuit breaker (creates default if None)
            enable_predictive_monitoring: Enable ML-based predictive load monitoring
            enable_connection_affinity: Enable connection affinity for query optimization
            enable_adaptive_config: Enable adaptive configuration management
            adaptation_strategy: Strategy for adaptive configuration changes
        """
        self.config = config
        
        # Enhanced monitoring components
        if enable_predictive_monitoring:
            self.load_monitor = PredictiveLoadMonitor(LoadMonitorConfig())
        else:
            self.load_monitor = load_monitor or LoadMonitor(LoadMonitorConfig())
            
        self.query_monitor = query_monitor or QueryMonitor(
            QueryMonitorConfig(
                enabled=config.enable_query_monitoring,
                slow_query_threshold_ms=config.slow_query_threshold_ms,
            )
        )

        # Enhanced circuit breaker with multi-level failure categorization
        if circuit_breaker:
            self.circuit_breaker = circuit_breaker
        else:
            from .enhanced_circuit_breaker import CircuitBreakerConfig
            cb_config = CircuitBreakerConfig(
                connection_threshold=3,
                timeout_threshold=5,
                query_threshold=10,
                transaction_threshold=5,
                recovery_timeout=60.0
            )
            self.circuit_breaker = MultiLevelCircuitBreaker(cb_config)

        # Connection affinity manager for query optimization
        self.connection_affinity: Optional[ConnectionAffinityManager] = None
        if enable_connection_affinity:
            self.connection_affinity = ConnectionAffinityManager(
                max_patterns=1000,
                max_connections=config.max_pool_size
            )

        # Adaptive configuration manager
        self.adaptive_config: Optional[AdaptiveConfigManager] = None
        if enable_adaptive_config:
            self.adaptive_config = AdaptiveConfigManager(
                strategy=adaptation_strategy
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

                # Start adaptive configuration monitoring if enabled
                if self.adaptive_config:
                    await self.adaptive_config.start_monitoring()

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
        self, 
        query: str, 
        parameters: dict[str, Any] | None = None,
        query_type: QueryType = QueryType.READ,
        timeout: Optional[float] = None
    ) -> Any:
        """Execute a query with advanced monitoring and optimization.

        Args:
            query: SQL query string
            parameters: Query parameters
            query_type: Type of query for connection affinity optimization
            timeout: Optional query timeout override

        Returns:
            Query result
        """
        query_id = await self.query_monitor.start_query(query)
        start_time = time.time()

        try:
            # Get optimal connection using affinity manager if available
            optimal_connection_id = None
            if self.connection_affinity:
                optimal_connection_id = await self.connection_affinity.get_optimal_connection(
                    query, query_type
                )
                if optimal_connection_id:
                    logger.debug(f"Using affinity-optimized connection {optimal_connection_id}")

            # Determine appropriate failure type for circuit breaker
            failure_type = self._map_query_type_to_failure_type(query_type)
            
            # Get timeout from adaptive config if available
            execution_timeout = timeout
            if not execution_timeout and self.adaptive_config:
                config_state = await self.adaptive_config.get_current_configuration()
                execution_timeout = config_state['current_settings']['timeout_ms'] / 1000.0

            # Execute query with circuit breaker protection
            async def _execute_query():
                async with self.get_session() as session:
                    # Wrap string queries in text() for SQLAlchemy
                    sql_query = text(query) if isinstance(query, str) else query
                    
                    if execution_timeout:
                        result = await asyncio.wait_for(
                            session.execute(sql_query, parameters or {}),
                            timeout=execution_timeout
                        )
                    else:
                        result = await session.execute(sql_query, parameters or {})
                    
                    await session.commit()
                    return result

            # Execute with circuit breaker protection
            if isinstance(self.circuit_breaker, MultiLevelCircuitBreaker):
                result = await self.circuit_breaker.execute(
                    _execute_query,
                    failure_type=failure_type,
                    timeout=execution_timeout
                )
            else:
                # Fallback for legacy circuit breaker
                result = await self.circuit_breaker.call(_execute_query)

            # Record successful query performance
            execution_time_ms = (time.time() - start_time) * 1000
            execution_time = await self.query_monitor.end_query(
                query_id, query, success=True
            )

            # Track performance in connection affinity manager
            if self.connection_affinity and optimal_connection_id:
                await self.connection_affinity.track_query_performance(
                    optimal_connection_id, query, execution_time_ms, query_type, success=True
                )

            logger.debug(
                f"Query executed in {execution_time:.2f}ms: {query[:100]}..."
            )

            return result

        except Exception as e:
            # Record failed query
            execution_time_ms = (time.time() - start_time) * 1000
            await self.query_monitor.end_query(query_id, query, success=False)
            
            # Track failure in connection affinity manager
            if self.connection_affinity and optimal_connection_id:
                await self.connection_affinity.track_query_performance(
                    optimal_connection_id, query, execution_time_ms, query_type, success=False
                )

            logger.error(f"Query execution failed: {e}")
            raise

    def _map_query_type_to_failure_type(self, query_type: QueryType) -> FailureType:
        """Map query type to appropriate circuit breaker failure type."""
        mapping = {
            QueryType.READ: FailureType.QUERY,
            QueryType.WRITE: FailureType.QUERY,
            QueryType.ANALYTICS: FailureType.QUERY,
            QueryType.TRANSACTION: FailureType.TRANSACTION,
            QueryType.MAINTENANCE: FailureType.QUERY,
        }
        return mapping.get(query_type, FailureType.QUERY)

    async def register_connection(
        self, 
        connection_id: str, 
        specialization: Optional[str] = None
    ) -> None:
        """Register a connection with the affinity manager.
        
        Args:
            connection_id: Unique connection identifier
            specialization: Optional specialization type
        """
        if self.connection_affinity:
            from .connection_affinity import ConnectionSpecialization
            
            # Map string specialization to enum
            spec_mapping = {
                "read": ConnectionSpecialization.READ_OPTIMIZED,
                "write": ConnectionSpecialization.WRITE_OPTIMIZED,
                "analytics": ConnectionSpecialization.ANALYTICS_OPTIMIZED,
                "transaction": ConnectionSpecialization.TRANSACTION_OPTIMIZED,
            }
            
            spec = spec_mapping.get(specialization, ConnectionSpecialization.GENERAL)
            await self.connection_affinity.register_connection(connection_id, spec)

    async def unregister_connection(self, connection_id: str) -> None:
        """Unregister a connection from the affinity manager.
        
        Args:
            connection_id: Connection identifier to remove
        """
        if self.connection_affinity:
            await self.connection_affinity.unregister_connection(connection_id)

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

        # Add enhanced circuit breaker stats
        if isinstance(self.circuit_breaker, MultiLevelCircuitBreaker):
            health_status = self.circuit_breaker.get_health_status()
            stats["enhanced_circuit_breaker"] = health_status

        # Add connection affinity stats
        if self.connection_affinity:
            try:
                affinity_report = await self.connection_affinity.get_performance_report()
                stats["connection_affinity"] = affinity_report
            except Exception as e:
                logger.debug(f"Failed to get connection affinity stats: {e}")
                stats["connection_affinity"] = {"error": str(e)}

        # Add adaptive configuration stats
        if self.adaptive_config:
            try:
                adaptive_config_info = await self.adaptive_config.get_current_configuration()
                stats["adaptive_config"] = adaptive_config_info
            except Exception as e:
                logger.debug(f"Failed to get adaptive config stats: {e}")
                stats["adaptive_config"] = {"error": str(e)}

        # Add predictive monitoring stats
        if isinstance(self.load_monitor, PredictiveLoadMonitor):
            try:
                prediction_metrics = await self.load_monitor.get_prediction_metrics()
                stats["predictive_monitoring"] = prediction_metrics
            except Exception as e:
                logger.debug(f"Failed to get prediction metrics: {e}")
                stats["predictive_monitoring"] = {"error": str(e)}

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
                with suppress(asyncio.CancelledError):
                    await self._health_check_task

            if self._metrics_task:
                self._metrics_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._metrics_task

            # Stop load monitor
            await self.load_monitor.stop()

            # Stop adaptive configuration monitoring if enabled
            if self.adaptive_config:
                await self.adaptive_config.stop_monitoring()

            # Dispose of engine
            if self._engine:
                await self._engine.dispose()
                self._engine = None

            self._session_factory = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
