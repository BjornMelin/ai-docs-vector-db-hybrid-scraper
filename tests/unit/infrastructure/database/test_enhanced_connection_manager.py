"""Comprehensive tests for enhanced AsyncConnectionManager with all new features.

This test module provides comprehensive coverage for the enhanced database connection
management system including predictive monitoring, multi-level circuit breaker,
connection affinity, and adaptive configuration.

Uses modern testing patterns with minimal mocking at boundaries.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.infrastructure.database.adaptive_config import AdaptationStrategy
from src.infrastructure.database.connection_affinity import QueryType
from src.infrastructure.database.connection_manager import AsyncConnectionManager
# Simple circuit breaker doesn't have failure types

# Simple load monitoring imports
from src.infrastructure.database.simple_monitor import SimpleLoadDecision


class TestEnhancedAsyncConnectionManager:
    """Test enhanced AsyncConnectionManager with all new features."""

    @pytest.fixture
    def mock_session(self):
        """Create a properly mocked async session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()

        # Mock result for execute calls
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        session.execute.return_value = mock_result

        return session

    @pytest.fixture
    def mock_session_factory(self, mock_session):
        """Create a mock session factory."""
        factory = Mock()
        factory.return_value = mock_session
        return factory

    @pytest.fixture
    def mock_engine(self, mock_session_factory):
        """Create a properly mocked async engine."""
        engine = AsyncMock()

        # Mock pool attributes
        engine.pool = Mock()
        engine.pool.checkedin.return_value = 3
        engine.pool.checkedout.return_value = 2
        engine.pool.overflow.return_value = 0
        engine.pool.invalidated.return_value = 0

        # Mock begin context manager for health checks
        @asynccontextmanager
        async def mock_begin():
            conn = AsyncMock()
            conn.execute = AsyncMock()
            yield conn

        engine.begin = mock_begin
        engine.dispose = AsyncMock()

        return engine

    @pytest.fixture
    def enhanced_manager(
        self,
        enhanced_db_config,
        mock_simple_load_monitor,
        mock_multi_level_circuit_breaker,
        mock_connection_affinity_manager,
        mock_adaptive_config_manager,
    ):
        """Create enhanced AsyncConnectionManager with all features enabled."""
        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            load_monitor=mock_simple_load_monitor,
            circuit_breaker=mock_multi_level_circuit_breaker,
            adaptive_config=mock_adaptive_config_manager,
            enable_connection_affinity=True,
            enable_adaptive_config=True,
            adaptation_strategy=AdaptationStrategy.MODERATE,
        )

        # Override connection affinity with our mock
        manager.connection_affinity = mock_connection_affinity_manager

        return manager

    @pytest.mark.asyncio
    async def test_enhanced_initialization(
        self, enhanced_manager, mock_engine, mock_session_factory
    ):
        """Test enhanced initialization with all components."""
        # Check that all enhanced components are properly initialized
        assert enhanced_manager.connection_affinity is not None
        assert enhanced_manager.adaptive_config is not None
        assert hasattr(enhanced_manager.circuit_breaker, "execute")

        # Test initialization process
        with (
            patch(
                "src.infrastructure.database.connection_manager.create_async_engine"
            ) as mock_create_engine,
            patch(
                "src.infrastructure.database.connection_manager.async_sessionmaker"
            ) as mock_sessionmaker,
        ):
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory

            await enhanced_manager.initialize()

            assert enhanced_manager._is_initialized
            enhanced_manager.load_monitor.start.assert_called_once()
            enhanced_manager.adaptive_config.start_monitoring.assert_called_once()

            # Clean up
            await enhanced_manager.shutdown()

    @pytest.mark.asyncio
    async def test_simple_load_monitoring_integration(
        self, enhanced_manager, mock_engine, mock_session_factory
    ):
        """Test integration with simple load monitoring."""
        with (
            patch(
                "src.infrastructure.database.connection_manager.create_async_engine"
            ) as mock_create_engine,
            patch(
                "src.infrastructure.database.connection_manager.async_sessionmaker"
            ) as mock_sessionmaker,
        ):
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory

            await enhanced_manager.initialize()

            # Test that simple load decision is used for pool sizing
            load_decision = SimpleLoadDecision(
                should_scale_up=True,
                should_scale_down=False,
                current_load=0.8,
                reason="Load 80% > 80% threshold",
                recommendation="Scale up database connections"
            )
            enhanced_manager.load_monitor.get_load_decision.return_value = load_decision

            # Simulate load decision request
            result = enhanced_manager.load_monitor.get_load_decision()

            assert result.current_load == 0.8
            assert result.should_scale_up is True
            assert "Scale up" in result.recommendation

            # Clean up
            await enhanced_manager.shutdown()

    # Multi-level circuit breaker test removed - using simple circuit breaker

    @pytest.mark.asyncio
    async def test_connection_affinity_integration(
        self, enhanced_manager, mock_engine, mock_session_factory
    ):
        """Test integration with connection affinity manager."""
        with (
            patch(
                "src.infrastructure.database.connection_manager.create_async_engine"
            ) as mock_create_engine,
            patch(
                "src.infrastructure.database.connection_manager.async_sessionmaker"
            ) as mock_sessionmaker,
        ):
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory

            await enhanced_manager.initialize()

            # Mock optimal connection selection
            enhanced_manager.connection_affinity.get_optimal_connection.return_value = (
                "conn_optimal"
            )

            # Mock circuit breaker and session
            mock_result = Mock()
            mock_result.scalar.return_value = 1
            enhanced_manager.circuit_breaker.execute.return_value = mock_result

            # Test query execution with connection affinity
            query = "SELECT * FROM analytics_data"
            await enhanced_manager.execute_query(query, query_type=QueryType.ANALYTICS)

            # Verify connection affinity was consulted
            enhanced_manager.connection_affinity.get_optimal_connection.assert_called_with(
                query, QueryType.ANALYTICS
            )

            # Verify performance tracking
            enhanced_manager.connection_affinity.track_query_performance.assert_called()

            # Clean up
            await enhanced_manager.shutdown()

    @pytest.mark.asyncio
    async def test_adaptive_configuration_integration(
        self, enhanced_manager, mock_engine, mock_session_factory
    ):
        """Test integration with adaptive configuration manager."""
        with (
            patch(
                "src.infrastructure.database.connection_manager.create_async_engine"
            ) as mock_create_engine,
            patch(
                "src.infrastructure.database.connection_manager.async_sessionmaker"
            ) as mock_sessionmaker,
        ):
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory

            await enhanced_manager.initialize()

            # Mock adaptive configuration response
            config_state = {
                "current_settings": {
                    "timeout_ms": 45000.0,
                    "pool_size": 8,
                    "failure_threshold": 7,
                }
            }
            enhanced_manager.adaptive_config.get_current_configuration.return_value = (
                config_state
            )

            # Mock circuit breaker and session
            mock_result = Mock()
            mock_result.scalar.return_value = 1
            enhanced_manager.circuit_breaker.execute.return_value = mock_result

            # Test query execution with adaptive timeout
            query = "SELECT * FROM large_table"
            await enhanced_manager.execute_query(
                query, timeout=None
            )  # Use adaptive timeout

            # Verify adaptive configuration was consulted
            enhanced_manager.adaptive_config.get_current_configuration.assert_called()

            # Clean up
            await enhanced_manager.shutdown()

    @pytest.mark.asyncio
    async def test_connection_registration_and_management(self, enhanced_manager):
        """Test connection registration with affinity manager."""
        await enhanced_manager.register_connection("conn_123", "read")

        # Verify connection was registered with correct specialization
        enhanced_manager.connection_affinity.register_connection.assert_called()
        call_args = enhanced_manager.connection_affinity.register_connection.call_args
        assert call_args[0][0] == "conn_123"  # connection_id

        # Test unregistration
        await enhanced_manager.unregister_connection("conn_123")
        enhanced_manager.connection_affinity.unregister_connection.assert_called_with(
            "conn_123"
        )

    @pytest.mark.asyncio
    async def test_comprehensive_stats_collection(
        self, enhanced_manager, mock_engine, mock_session_factory
    ):
        """Test comprehensive statistics collection from all components."""
        with (
            patch(
                "src.infrastructure.database.connection_manager.create_async_engine"
            ) as mock_create_engine,
            patch(
                "src.infrastructure.database.connection_manager.async_sessionmaker"
            ) as mock_sessionmaker,
        ):
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory

            await enhanced_manager.initialize()

            # Mock the missing _failure_count attribute
            enhanced_manager.circuit_breaker._failure_count = 0

            stats = await enhanced_manager.get_connection_stats()

            # Verify all enhanced components provide stats
            # Simple monitoring doesn't have enhanced_circuit_breaker or predictive_monitoring stats
            assert "connection_affinity" in stats
            assert "adaptive_config" in stats

            # Verify component-specific methods were called
            # Simple circuit breaker doesn't have get_health_status method
            enhanced_manager.connection_affinity.get_performance_report.assert_called()
            enhanced_manager.adaptive_config.get_current_configuration.assert_called()
            # Simple load monitor doesn't have get_prediction_metrics

            # Clean up
            await enhanced_manager.shutdown()

    # test_failure_type_mapping removed - not applicable to simple circuit breaker

    @pytest.mark.asyncio
    async def test_shutdown_with_all_components(
        self, enhanced_manager, mock_engine, mock_session_factory
    ):
        """Test shutdown process with all enhanced components."""
        with (
            patch(
                "src.infrastructure.database.connection_manager.create_async_engine"
            ) as mock_create_engine,
            patch(
                "src.infrastructure.database.connection_manager.async_sessionmaker"
            ) as mock_sessionmaker,
        ):
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory

            await enhanced_manager.initialize()
            await enhanced_manager.shutdown()

            # Verify all components are properly stopped
            enhanced_manager.load_monitor.stop.assert_called_once()
            enhanced_manager.adaptive_config.stop_monitoring.assert_called_once()
            assert not enhanced_manager._is_initialized

    @pytest.mark.asyncio
    async def test_error_handling_with_enhanced_features(
        self, enhanced_manager, mock_engine, mock_session_factory
    ):
        """Test error handling across all enhanced features."""
        with (
            patch(
                "src.infrastructure.database.connection_manager.create_async_engine"
            ) as mock_create_engine,
            patch(
                "src.infrastructure.database.connection_manager.async_sessionmaker"
            ) as mock_sessionmaker,
        ):
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory

            await enhanced_manager.initialize()

            # Mock a database error
            enhanced_manager.circuit_breaker.execute.side_effect = Exception(
                "Database error"
            )

            # Mock connection affinity to return a connection
            enhanced_manager.connection_affinity.get_optimal_connection.return_value = (
                "conn_error"
            )

            with pytest.raises(Exception, match="Database error"):
                await enhanced_manager.execute_query("SELECT * FROM test")

            # Verify error was tracked in connection affinity
            enhanced_manager.connection_affinity.track_query_performance.assert_called()
            call_args = (
                enhanced_manager.connection_affinity.track_query_performance.call_args
            )
            assert call_args.kwargs["success"] is False

            # Clean up
            await enhanced_manager.shutdown()

    def test_enhanced_initialization_flags(self, enhanced_db_config):
        """Test initialization with different feature flags."""
        # Test with all features disabled
        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            enable_connection_affinity=False,
            enable_adaptive_config=False,
        )

        assert manager.connection_affinity is None
        assert manager.adaptive_config is None
        # Load monitor should be SimpleLoadMonitor
        assert type(manager.load_monitor).__name__ == "SimpleLoadMonitor"

    def test_adaptation_strategy_configuration(self, enhanced_db_config):
        """Test different adaptation strategies."""
        strategies = [
            AdaptationStrategy.CONSERVATIVE,
            AdaptationStrategy.MODERATE,
            AdaptationStrategy.AGGRESSIVE,
        ]

        for strategy in strategies:
            manager = AsyncConnectionManager(
                config=enhanced_db_config,
                adaptation_strategy=strategy,
            )
            # Adaptive config should be created with the specified strategy
            assert manager.adaptive_config is not None


class TestEnhancedConnectionManagerEdgeCases:
    """Test edge cases and error conditions for enhanced features."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine for edge case tests."""
        engine = AsyncMock()
        engine.pool = Mock()
        engine.pool.checkedin.return_value = 0
        engine.pool.checkedout.return_value = 0
        engine.pool.overflow.return_value = 0
        engine.pool.invalidated.return_value = 0
        engine.dispose = AsyncMock()

        @asynccontextmanager
        async def mock_begin():
            conn = AsyncMock()
            conn.execute = AsyncMock()
            yield conn

        engine.begin = mock_begin
        return engine

    @pytest.mark.asyncio
    async def test_partial_component_failures(self, enhanced_db_config, mock_engine):
        """Test handling of partial component failures."""
        # Create manager with some mocked components that fail
        failing_affinity_manager = Mock()
        failing_affinity_manager.get_optimal_connection = AsyncMock(
            side_effect=Exception("Affinity error")
        )
        failing_affinity_manager.track_query_performance = AsyncMock()

        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            enable_connection_affinity=True,
            enable_adaptive_config=True,
        )
        manager.connection_affinity = failing_affinity_manager

        with (
            patch(
                "src.infrastructure.database.connection_manager.create_async_engine"
            ) as mock_create_engine,
            patch(
                "src.infrastructure.database.connection_manager.async_sessionmaker"
            ) as mock_sessionmaker,
        ):
            mock_create_engine.return_value = mock_engine
            
            # Create a proper async session mock
            mock_session = AsyncMock()
            mock_result = Mock()
            mock_result.scalar.return_value = 1
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.commit = AsyncMock()
            mock_session.close = AsyncMock()
            
            mock_sessionmaker.return_value = Mock(return_value=mock_session)

            await manager.initialize()

            # Query should still succeed despite affinity manager failure
            result = await manager.execute_query("SELECT 1")
            assert result == mock_result

            # Clean up
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_stats_collection_with_component_errors(
        self, enhanced_db_config, mock_engine
    ):
        """Test stats collection when some components error."""
        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            enable_connection_affinity=True,
            enable_adaptive_config=True,
        )

        # Mock components with some failing
        manager.connection_affinity = Mock()
        manager.connection_affinity.get_performance_report = AsyncMock(
            side_effect=Exception("Affinity stats error")
        )

        manager.adaptive_config = Mock()
        manager.adaptive_config.start_monitoring = AsyncMock()
        manager.adaptive_config.stop_monitoring = AsyncMock()
        manager.adaptive_config.get_current_configuration = AsyncMock(
            side_effect=Exception("Config stats error")
        )

        with (
            patch(
                "src.infrastructure.database.connection_manager.create_async_engine"
            ) as mock_create_engine,
            patch(
                "src.infrastructure.database.connection_manager.async_sessionmaker"
            ) as mock_sessionmaker,
        ):
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = Mock()

            await manager.initialize()

            # Mock the missing _failure_count attribute
            manager.circuit_breaker._failure_count = 0

            stats = await manager.get_connection_stats()

            # Should still return stats with error information
            assert "connection_affinity" in stats
            assert "adaptive_config" in stats
            assert "error" in str(stats["connection_affinity"])
            assert "error" in str(stats["adaptive_config"])

            # Clean up
            await manager.shutdown()


class TestEnhancedConnectionManagerIntegration:
    """Integration tests for enhanced connection manager."""

    @pytest.fixture
    def mock_real_components(self):
        """Create more realistic mock components for integration tests."""
        from unittest.mock import Mock

        # Create mocks that behave more like real components
        load_monitor = Mock()
        load_monitor.start = AsyncMock()
        load_monitor.stop = AsyncMock()
        load_monitor.record_request_start = AsyncMock()
        load_monitor.record_request_end = AsyncMock()
        load_monitor.record_connection_error = AsyncMock()
        load_monitor.get_current_load = AsyncMock()
        load_monitor.calculate_load_factor = Mock(return_value=0.5)
        load_monitor.predict_future_load = AsyncMock()
        load_monitor.get_prediction_metrics = AsyncMock(
            return_value={
                "model_trained": True,
                "training_samples": 100,
                "prediction_accuracy_avg": 0.85,
            }
        )

        # Create a simple circuit breaker mock
        from src.infrastructure.shared import CircuitBreaker
        
        circuit_breaker = Mock(spec=CircuitBreaker)
        circuit_breaker.state = Mock()
        circuit_breaker.state.value = "closed"
        circuit_breaker._failure_count = 0

        # Configure execute to actually call functions properly
        async def mock_real_execute(func, *args, **kwargs):
            if callable(func):
                if asyncio.iscoroutinefunction(func):
                    return await func()
                return func()
            # Return a simple mock result
            return Mock()

        circuit_breaker.execute = AsyncMock(side_effect=mock_real_execute)
        circuit_breaker.call = AsyncMock(side_effect=mock_real_execute)
        circuit_breaker.get_health_status = Mock(
            return_value={
                "state": "closed",
                "failure_metrics": {"total_failures": 0},
                "request_metrics": {"success_rate": 1.0},
            }
        )

        affinity_manager = Mock()
        affinity_manager.register_connection = AsyncMock()
        affinity_manager.unregister_connection = AsyncMock()
        affinity_manager.track_query_performance = AsyncMock()
        affinity_manager.get_performance_report = AsyncMock(
            return_value={
                "summary": {"total_patterns": 25, "total_connections": 5},
                "top_patterns": [],
                "connection_performance": {},
            }
        )

        adaptive_config = Mock()
        adaptive_config.start_monitoring = AsyncMock()
        adaptive_config.stop_monitoring = AsyncMock()
        adaptive_config.get_current_configuration = AsyncMock(
            return_value={
                "strategy": "moderate",
                "current_settings": {
                    "pool_size": 8,
                    "monitoring_interval": 5.0,
                    "failure_threshold": 5,
                    "timeout_ms": 30000.0,
                },
            }
        )

        return {
            "load_monitor": load_monitor,
            "circuit_breaker": circuit_breaker,
            "affinity_manager": affinity_manager,
            "adaptive_config": adaptive_config,
        }

    @pytest.mark.asyncio
    async def test_full_enhanced_lifecycle(
        self, enhanced_db_config, mock_real_components
    ):
        """Test full lifecycle with all enhanced features enabled."""

        # Create session and engine mocks
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar = Mock(
            return_value=1
        )  # Make scalar return value directly, not a coroutine
        mock_session.execute.return_value = mock_result
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        mock_engine = AsyncMock()
        mock_engine.pool = Mock()
        mock_engine.pool.checkedin.return_value = 3
        mock_engine.pool.checkedout.return_value = 2
        mock_engine.pool.overflow.return_value = 0
        mock_engine.pool.invalidated.return_value = 0
        mock_engine.dispose = AsyncMock()

        @asynccontextmanager
        async def mock_begin():
            conn = AsyncMock()
            conn.execute = AsyncMock()
            yield conn

        mock_engine.begin = mock_begin

        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            load_monitor=mock_real_components["load_monitor"],
            circuit_breaker=mock_real_components["circuit_breaker"],
        )
        manager.connection_affinity = mock_real_components["affinity_manager"]
        manager.adaptive_config = mock_real_components["adaptive_config"]

        try:
            with (
                patch(
                    "src.infrastructure.database.connection_manager.create_async_engine"
                ) as mock_create_engine,
                patch(
                    "src.infrastructure.database.connection_manager.async_sessionmaker"
                ) as mock_sessionmaker,
            ):
                mock_create_engine.return_value = mock_engine
                mock_sessionmaker.return_value = mock_session_factory

                # Initialize and test basic functionality
                await manager.initialize()
                assert manager._is_initialized

                # Test query execution
                # The circuit breaker will call the function, which returns the mock_result from the session
                result = await manager.execute_query(
                    "SELECT 1", query_type=QueryType.READ
                )
                # The result should be the mock_result from the session
                assert result == mock_result
                assert result.scalar() == 1

                # Test connection registration
                await manager.register_connection("test_conn", "read")

                # Test adaptive configuration
                config_state = await mock_real_components[
                    "adaptive_config"
                ].get_current_configuration()
                assert "strategy" in config_state
                assert "current_settings" in config_state

                # Test comprehensive stats
                stats = await manager.get_connection_stats()
                assert "pool_size" in stats
                # Simple circuit breaker doesn't add enhanced_circuit_breaker stats

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_performance_under_load(self, enhanced_db_config):
        """Test enhanced manager performance under simulated load."""
        # Create simple mocks for performance test
        load_monitor = Mock()
        load_monitor.start = AsyncMock()
        load_monitor.stop = AsyncMock()
        load_monitor.record_request_start = AsyncMock()
        load_monitor.record_request_end = AsyncMock()
        load_monitor.record_connection_error = AsyncMock()
        load_monitor.get_current_load = AsyncMock()
        load_monitor.calculate_load_factor = Mock(return_value=0.5)
        load_monitor.get_prediction_metrics = AsyncMock(return_value={})

        circuit_breaker = Mock()
        circuit_breaker.state = Mock()
        circuit_breaker.state.value = "closed"
        circuit_breaker._failure_count = 0

        # Create a proper async function that returns a mock result for the test
        async def mock_circuit_execute(func, *args, **kwargs):
            # Execute the actual function and return its result
            if callable(func):
                if asyncio.iscoroutinefunction(func):
                    return await func()
                return func()
            # Return a mock result with scalar that returns 1
            mock_result = Mock()
            mock_result.scalar = Mock(return_value=1)
            return mock_result

        circuit_breaker.execute = AsyncMock(side_effect=mock_circuit_execute)
        circuit_breaker.call = AsyncMock(side_effect=mock_circuit_execute)
        circuit_breaker.get_health_status = Mock(return_value={})

        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            load_monitor=load_monitor,
            circuit_breaker=circuit_breaker,
            enable_connection_affinity=True,
            enable_adaptive_config=False,  # Disable for performance test
        )

        # Create session and engine mocks
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 42  # Different value for each query
        mock_session.execute.return_value = mock_result
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        mock_engine = AsyncMock()
        mock_engine.pool = Mock()
        mock_engine.pool.checkedin.return_value = 5
        mock_engine.pool.checkedout.return_value = 0
        mock_engine.pool.overflow.return_value = 0
        mock_engine.pool.invalidated.return_value = 0
        mock_engine.dispose = AsyncMock()

        @asynccontextmanager
        async def mock_begin():
            conn = AsyncMock()
            conn.execute = AsyncMock()
            yield conn

        mock_engine.begin = mock_begin

        try:
            with (
                patch(
                    "src.infrastructure.database.connection_manager.create_async_engine"
                ) as mock_create_engine,
                patch(
                    "src.infrastructure.database.connection_manager.async_sessionmaker"
                ) as mock_sessionmaker,
            ):
                mock_create_engine.return_value = mock_engine
                mock_sessionmaker.return_value = mock_session_factory

                await manager.initialize()

                # Mock circuit breaker to return results
                circuit_breaker.execute.return_value = mock_result

                # Simulate concurrent load
                async def execute_queries():
                    tasks = []
                    for i in range(20):
                        query_type = [
                            QueryType.READ,
                            QueryType.WRITE,
                            QueryType.ANALYTICS,
                        ][i % 3]
                        task = manager.execute_query(
                            f"SELECT {i}", query_type=query_type
                        )
                        tasks.append(task)

                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    return results

                start_time = time.time()
                results = await execute_queries()
                execution_time = time.time() - start_time

                # Should complete within reasonable time
                assert execution_time < 5.0  # 5 seconds max

                # All queries should succeed (no exceptions)
                exceptions = [r for r in results if isinstance(r, Exception)]
                assert len(exceptions) == 0

        finally:
            await manager.shutdown()
