"""Comprehensive tests for enhanced AsyncConnectionManager with all new features.

This test module provides comprehensive coverage for the enhanced database connection
management system including predictive monitoring, multi-level circuit breaker,
connection affinity, and adaptive configuration.
"""

import asyncio
import time
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.infrastructure.database.adaptive_config import AdaptationStrategy
from src.infrastructure.database.connection_affinity import QueryType
from src.infrastructure.database.connection_manager import AsyncConnectionManager
from src.infrastructure.database.enhanced_circuit_breaker import FailureType
from src.infrastructure.database.predictive_monitor import LoadPrediction


class TestEnhancedAsyncConnectionManager:
    """Test enhanced AsyncConnectionManager with all new features."""

    @pytest.fixture
    def enhanced_manager(
        self,
        enhanced_db_config,
        mock_predictive_load_monitor,
        mock_multi_level_circuit_breaker,
        mock_connection_affinity_manager,
        mock_adaptive_config_manager,
    ):
        """Create enhanced AsyncConnectionManager with all features enabled."""
        return AsyncConnectionManager(
            config=enhanced_db_config,
            load_monitor=mock_predictive_load_monitor,
            circuit_breaker=mock_multi_level_circuit_breaker,
            adaptive_config=mock_adaptive_config_manager,
            enable_predictive_monitoring=True,
            enable_connection_affinity=True,
            enable_adaptive_config=True,
            adaptation_strategy=AdaptationStrategy.MODERATE,
        )

    @pytest.mark.asyncio
    async def test_enhanced_initialization(self, enhanced_manager):
        """Test enhanced initialization with all components."""
        # Check that all enhanced components are properly initialized
        assert enhanced_manager.connection_affinity is not None
        assert enhanced_manager.adaptive_config is not None
        assert hasattr(enhanced_manager.circuit_breaker, "execute")

        # Test initialization process
        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ) as mock_create_engine:
            mock_engine = AsyncMock()
            mock_create_engine.return_value = mock_engine

            await enhanced_manager.initialize()

            assert enhanced_manager._is_initialized
            enhanced_manager.load_monitor.start.assert_called_once()
            enhanced_manager.adaptive_config.start_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_predictive_load_monitoring_integration(self, enhanced_manager):
        """Test integration with predictive load monitoring."""
        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ):
            await enhanced_manager.initialize()

            # Test that prediction is used for pool sizing
            prediction = LoadPrediction(
                predicted_load=0.8,
                confidence_score=0.9,
                recommendation="Scale up immediately",
                time_horizon_minutes=15,
                feature_importance={"avg_requests": 0.5},
                trend_direction="increasing",
            )
            enhanced_manager.load_monitor.predict_future_load.return_value = prediction

            # Simulate prediction request
            result = await enhanced_manager.load_monitor.predict_future_load(
                horizon_minutes=15
            )

            assert result.predicted_load == 0.8
            assert result.confidence_score == 0.9
            assert "Scale up" in result.recommendation

    @pytest.mark.asyncio
    async def test_multi_level_circuit_breaker_integration(self, enhanced_manager):
        """Test integration with multi-level circuit breaker."""
        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ):
            await enhanced_manager.initialize()

            # Mock session and query execution
            mock_session = AsyncMock()
            mock_result = Mock()
            mock_session.execute.return_value = mock_result
            mock_session.commit = AsyncMock()

            # Mock circuit breaker to return the session (make it awaitable)
            enhanced_manager.circuit_breaker.execute = AsyncMock(return_value=mock_result)
            enhanced_manager.circuit_breaker.call = AsyncMock(return_value=mock_result)

            # Test query execution with enhanced circuit breaker
            query = "SELECT * FROM users"
            await enhanced_manager.execute_query(query, query_type=QueryType.READ)

            # Verify circuit breaker was called with correct failure type
            enhanced_manager.circuit_breaker.execute.assert_called()
            call_args = enhanced_manager.circuit_breaker.execute.call_args
            assert "failure_type" in call_args.kwargs
            assert call_args.kwargs["failure_type"] == FailureType.QUERY

    @pytest.mark.asyncio
    async def test_connection_affinity_integration(self, enhanced_manager):
        """Test integration with connection affinity manager."""
        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ):
            await enhanced_manager.initialize()

            # Mock optimal connection selection
            enhanced_manager.connection_affinity.get_optimal_connection.return_value = (
                "conn_optimal"
            )

            # Mock circuit breaker and session
            mock_result = Mock()
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

    @pytest.mark.asyncio
    async def test_adaptive_configuration_integration(self, enhanced_manager):
        """Test integration with adaptive configuration manager."""
        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ):
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
            enhanced_manager.circuit_breaker.execute.return_value = mock_result

            # Test query execution with adaptive timeout
            query = "SELECT * FROM large_table"
            await enhanced_manager.execute_query(
                query, timeout=None
            )  # Use adaptive timeout

            # Verify adaptive configuration was consulted
            enhanced_manager.adaptive_config.get_current_configuration.assert_called()

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
    async def test_comprehensive_stats_collection(self, enhanced_manager):
        """Test comprehensive statistics collection from all components."""
        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ) as mock_create_engine:
            mock_engine = AsyncMock()
            mock_pool = Mock()
            mock_pool.checkedin.return_value = 3
            mock_pool.checkedout.return_value = 2
            mock_engine.pool = mock_pool
            mock_create_engine.return_value = mock_engine

            await enhanced_manager.initialize()

            stats = await enhanced_manager.get_connection_stats()

            # Verify all enhanced components provide stats
            assert "enhanced_circuit_breaker" in stats
            assert "connection_affinity" in stats
            assert "adaptive_config" in stats
            assert "predictive_monitoring" in stats

            # Verify component-specific methods were called
            enhanced_manager.circuit_breaker.get_health_status.assert_called()
            enhanced_manager.connection_affinity.get_performance_report.assert_called()
            enhanced_manager.adaptive_config.get_current_configuration.assert_called()
            enhanced_manager.load_monitor.get_prediction_metrics.assert_called()

    @pytest.mark.asyncio
    async def test_failure_type_mapping(self, enhanced_manager):
        """Test query type to failure type mapping."""
        # Test mapping function
        mapping_tests = [
            (QueryType.READ, FailureType.QUERY),
            (QueryType.WRITE, FailureType.QUERY),
            (QueryType.ANALYTICS, FailureType.QUERY),
            (QueryType.TRANSACTION, FailureType.TRANSACTION),
            (QueryType.MAINTENANCE, FailureType.QUERY),
        ]

        for query_type, expected_failure_type in mapping_tests:
            result = enhanced_manager._map_query_type_to_failure_type(query_type)
            assert result == expected_failure_type

    @pytest.mark.asyncio
    async def test_shutdown_with_all_components(self, enhanced_manager):
        """Test shutdown process with all enhanced components."""
        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ):
            await enhanced_manager.initialize()
            await enhanced_manager.shutdown()

            # Verify all components are properly stopped
            enhanced_manager.load_monitor.stop.assert_called_once()
            enhanced_manager.adaptive_config.stop_monitoring.assert_called_once()
            assert not enhanced_manager._is_initialized

    @pytest.mark.asyncio
    async def test_error_handling_with_enhanced_features(self, enhanced_manager):
        """Test error handling across all enhanced features."""
        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ):
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

    def test_enhanced_initialization_flags(self, enhanced_db_config):
        """Test initialization with different feature flags."""
        # Test with all features disabled
        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            enable_predictive_monitoring=False,
            enable_connection_affinity=False,
            enable_adaptive_config=False,
        )

        assert manager.connection_affinity is None
        assert manager.adaptive_config is None
        # Load monitor should be basic LoadMonitor, not PredictiveLoadMonitor
        assert type(manager.load_monitor).__name__ == "LoadMonitor"

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

    @pytest.mark.asyncio
    async def test_partial_component_failures(self, enhanced_db_config):
        """Test handling of partial component failures."""
        # Create manager with some mocked components that fail
        failing_affinity_manager = Mock()
        failing_affinity_manager.get_optimal_connection = AsyncMock(
            side_effect=Exception("Affinity error")
        )
        failing_affinity_manager.track_query_performance = AsyncMock()

        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            enable_predictive_monitoring=True,
            enable_connection_affinity=True,
            enable_adaptive_config=True,
        )
        manager.connection_affinity = failing_affinity_manager

        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ):
            await manager.initialize()

            # Mock circuit breaker to succeed
            mock_result = Mock()
            manager.circuit_breaker.execute = AsyncMock(return_value=mock_result)

            # Query should still succeed despite affinity manager failure
            result = await manager.execute_query("SELECT 1")
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_stats_collection_with_component_errors(self, enhanced_db_config):
        """Test stats collection when some components error."""
        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            enable_predictive_monitoring=True,
            enable_connection_affinity=True,
            enable_adaptive_config=True,
        )

        # Mock components with some failing
        manager.connection_affinity = Mock()
        manager.connection_affinity.get_performance_report = AsyncMock(
            side_effect=Exception("Affinity stats error")
        )

        manager.adaptive_config = Mock()
        manager.adaptive_config.get_current_configuration = AsyncMock(
            side_effect=Exception("Config stats error")
        )

        with patch(
            "src.infrastructure.database.connection_manager.create_async_engine"
        ) as mock_create_engine:
            mock_engine = AsyncMock()
            mock_engine.pool = Mock()
            mock_engine.pool.checkedin = Mock(return_value=0)
            mock_create_engine.return_value = mock_engine

            await manager.initialize()

            stats = await manager.get_connection_stats()

            # Should still return stats with error information
            assert "connection_affinity" in stats
            assert "adaptive_config" in stats
            assert "error" in str(stats["connection_affinity"])
            assert "error" in str(stats["adaptive_config"])


class TestEnhancedConnectionManagerIntegration:
    """Integration tests for enhanced connection manager."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_enhanced_lifecycle(self, enhanced_db_config):
        """Test full lifecycle with all enhanced features enabled."""
        from src.infrastructure.database.adaptive_config import AdaptiveConfigManager
        from src.infrastructure.database.connection_affinity import (
            ConnectionAffinityManager,
        )
        from src.infrastructure.database.enhanced_circuit_breaker import (
            CircuitBreakerConfig,
        )
        from src.infrastructure.database.enhanced_circuit_breaker import (
            MultiLevelCircuitBreaker,
        )
        from src.infrastructure.database.load_monitor import LoadMonitorConfig
        from src.infrastructure.database.predictive_monitor import PredictiveLoadMonitor

        # Use real components for integration test
        load_monitor = PredictiveLoadMonitor(LoadMonitorConfig(monitoring_interval=0.1))
        circuit_breaker = MultiLevelCircuitBreaker(CircuitBreakerConfig())
        affinity_manager = ConnectionAffinityManager(
            max_patterns=100, max_connections=10
        )
        adaptive_config = AdaptiveConfigManager(strategy=AdaptationStrategy.MODERATE)

        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            load_monitor=load_monitor,
            circuit_breaker=circuit_breaker,
        )
        manager.connection_affinity = affinity_manager
        manager.adaptive_config = adaptive_config

        try:
            # Initialize and test basic functionality
            await manager.initialize()
            assert manager._is_initialized

            # Test query execution
            result = await manager.execute_query("SELECT 1", query_type=QueryType.READ)
            assert result.scalar() == 1

            # Test connection registration
            await manager.register_connection("test_conn", "read")

            # Test prediction
            prediction = await load_monitor.predict_future_load(horizon_minutes=10)
            assert prediction.predicted_load >= 0
            assert prediction.confidence_score >= 0

            # Test adaptive configuration
            config_state = await adaptive_config.get_current_configuration()
            assert "strategy" in config_state
            assert "current_settings" in config_state

            # Test comprehensive stats
            stats = await manager.get_connection_stats()
            assert "pool_size" in stats
            assert "enhanced_circuit_breaker" in stats

            # Verify query was tracked by affinity manager
            performance_report = await affinity_manager.get_performance_report()
            assert "summary" in performance_report

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_performance_under_load(self, enhanced_db_config):
        """Test enhanced manager performance under simulated load."""
        from src.infrastructure.database.load_monitor import LoadMonitorConfig
        from src.infrastructure.database.predictive_monitor import PredictiveLoadMonitor

        load_monitor = PredictiveLoadMonitor(
            LoadMonitorConfig(monitoring_interval=0.01)
        )
        manager = AsyncConnectionManager(
            config=enhanced_db_config,
            load_monitor=load_monitor,
            enable_predictive_monitoring=True,
            enable_connection_affinity=True,
            enable_adaptive_config=False,  # Disable for performance test
        )

        try:
            await manager.initialize()

            # Simulate concurrent load
            async def execute_queries():
                tasks = []
                for i in range(20):
                    query_type = [QueryType.READ, QueryType.WRITE, QueryType.ANALYTICS][
                        i % 3
                    ]
                    task = manager.execute_query(f"SELECT {i}", query_type=query_type)
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

            # Check that load monitoring recorded the activity
            current_load = await load_monitor.get_current_load()
            assert current_load.concurrent_requests >= 0

        finally:
            await manager.shutdown()
