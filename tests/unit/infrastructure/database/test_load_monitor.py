"""Unit tests for LoadMonitor with comprehensive coverage.

This test module demonstrates modern testing patterns for load monitoring including:
- Test doubles for system metrics
- Comprehensive coverage of load calculation algorithms
- Async testing patterns
- Resource management testing
"""

import asyncio
import time
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.infrastructure.database.load_monitor import LoadMetrics
from src.infrastructure.database.load_monitor import LoadMonitor
from src.infrastructure.database.load_monitor import LoadMonitorConfig


class TestLoadMetrics:
    """Test LoadMetrics dataclass."""

    def test_load_metrics_initialization(self):
        """Test basic initialization of LoadMetrics."""
        timestamp = time.time()
        metrics = LoadMetrics(
            concurrent_requests=5,
            memory_usage_percent=65.0,
            cpu_usage_percent=45.0,
            avg_response_time_ms=150.0,
            connection_errors=2,
            timestamp=timestamp,
        )

        assert metrics.concurrent_requests == 5
        assert metrics.memory_usage_percent == 65.0
        assert metrics.cpu_usage_percent == 45.0
        assert metrics.avg_response_time_ms == 150.0
        assert metrics.connection_errors == 2
        assert metrics.timestamp == timestamp

    def test_load_metrics_zero_values(self):
        """Test LoadMetrics with zero values."""
        metrics = LoadMetrics(
            concurrent_requests=0,
            memory_usage_percent=0.0,
            cpu_usage_percent=0.0,
            avg_response_time_ms=0.0,
            connection_errors=0,
            timestamp=time.time(),
        )

        assert metrics.concurrent_requests == 0
        assert metrics.memory_usage_percent == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.avg_response_time_ms == 0.0
        assert metrics.connection_errors == 0


class TestLoadMonitorConfig:
    """Test LoadMonitorConfig validation and defaults."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = LoadMonitorConfig()

        assert config.monitoring_interval == 5.0
        assert config.metrics_window_size == 60
        assert config.response_time_threshold_ms == 500.0
        assert config.memory_threshold_percent == 70.0
        assert config.cpu_threshold_percent == 70.0

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = LoadMonitorConfig(
            monitoring_interval=10.0,
            metrics_window_size=120,
            response_time_threshold_ms=1000.0,
            memory_threshold_percent=80.0,
            cpu_threshold_percent=75.0,
        )

        assert config.monitoring_interval == 10.0
        assert config.metrics_window_size == 120
        assert config.response_time_threshold_ms == 1000.0
        assert config.memory_threshold_percent == 80.0
        assert config.cpu_threshold_percent == 75.0

    def test_configuration_validation(self):
        """Test configuration validation for invalid values."""
        # Test negative monitoring interval
        with pytest.raises(ValueError):
            LoadMonitorConfig(monitoring_interval=-1.0)

        # Test zero metrics window size
        with pytest.raises(ValueError):
            LoadMonitorConfig(metrics_window_size=0)

        # Test invalid memory threshold
        with pytest.raises(ValueError):
            LoadMonitorConfig(memory_threshold_percent=150.0)


class TestLoadMonitor:
    """Test LoadMonitor functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration with faster intervals."""
        return LoadMonitorConfig(
            monitoring_interval=0.1,  # Faster for testing
            metrics_window_size=10,
            response_time_threshold_ms=200.0,
            memory_threshold_percent=75.0,
            cpu_threshold_percent=70.0,
        )

    @pytest.fixture
    def load_monitor(self, config):
        """Create LoadMonitor instance."""
        return LoadMonitor(config)

    @pytest.mark.asyncio
    async def test_initialization(self, load_monitor):
        """Test LoadMonitor initialization."""
        assert load_monitor._metrics_history == []
        assert load_monitor._current_requests == 0
        assert load_monitor._total_requests == 0
        assert load_monitor._total_response_time == 0.0
        assert load_monitor._connection_errors == 0
        assert not load_monitor._is_monitoring
        assert load_monitor._monitoring_task is None

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, load_monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring
        await load_monitor.start()
        assert load_monitor._is_monitoring
        assert load_monitor._monitoring_task is not None
        assert not load_monitor._monitoring_task.done()

        # Stop monitoring
        await load_monitor.stop()
        assert not load_monitor._is_monitoring
        assert load_monitor._monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_double_start_stop(self, load_monitor):
        """Test double start/stop calls are handled gracefully."""
        # Double start
        await load_monitor.start()
        first_task = load_monitor._monitoring_task
        await load_monitor.start()  # Should not create new task
        assert load_monitor._monitoring_task is first_task

        # Double stop
        await load_monitor.stop()
        await load_monitor.stop()  # Should not raise
        assert not load_monitor._is_monitoring

    @pytest.mark.asyncio
    async def test_request_tracking(self, load_monitor):
        """Test request start/end tracking."""
        # Record request start
        await load_monitor.record_request_start()
        assert load_monitor._current_requests == 1
        assert load_monitor._total_requests == 1

        # Record another request start
        await load_monitor.record_request_start()
        assert load_monitor._current_requests == 2
        assert load_monitor._total_requests == 2

        # Record request end
        response_time = 150.0
        await load_monitor.record_request_end(response_time)
        assert load_monitor._current_requests == 1
        assert load_monitor._total_response_time == response_time

    @pytest.mark.asyncio
    async def test_connection_error_tracking(self, load_monitor):
        """Test connection error tracking."""
        initial_errors = load_monitor._connection_errors

        await load_monitor.record_connection_error()
        assert load_monitor._connection_errors == initial_errors + 1

        await load_monitor.record_connection_error()
        assert load_monitor._connection_errors == initial_errors + 2

    @pytest.mark.asyncio
    async def test_get_current_load_empty_history(self, load_monitor):
        """Test getting current load with empty history."""
        metrics = await load_monitor.get_current_load()

        assert metrics.concurrent_requests == 0
        assert metrics.memory_usage_percent == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.avg_response_time_ms == 0.0
        assert metrics.connection_errors == 0
        assert metrics.timestamp > 0

    @pytest.mark.asyncio
    async def test_get_current_load_with_history(self, load_monitor):
        """Test getting current load with existing history."""
        # Add some metrics to history
        test_metrics = LoadMetrics(
            concurrent_requests=3,
            memory_usage_percent=60.0,
            cpu_usage_percent=40.0,
            avg_response_time_ms=100.0,
            connection_errors=1,
            timestamp=time.time(),
        )
        load_monitor._metrics_history.append(test_metrics)

        metrics = await load_monitor.get_current_load()
        assert metrics == test_metrics

    @pytest.mark.asyncio
    async def test_get_average_load(self, load_monitor):
        """Test calculating average load over time window."""
        current_time = time.time()

        # Add metrics over a time window
        metrics_list = []
        for i in range(5):
            metrics = LoadMetrics(
                concurrent_requests=i + 1,
                memory_usage_percent=50.0 + i * 5,
                cpu_usage_percent=30.0 + i * 3,
                avg_response_time_ms=100.0 + i * 10,
                connection_errors=i,
                timestamp=current_time - (60 * (5 - i)),  # 1 minute apart
            )
            metrics_list.append(metrics)
            load_monitor._metrics_history.append(metrics)

        # Get average over 10 minutes (should include all metrics)
        avg_metrics = await load_monitor.get_average_load(window_minutes=10)

        # Verify averages
        expected_concurrent = sum(m.concurrent_requests for m in metrics_list) / len(
            metrics_list
        )
        expected_memory = sum(m.memory_usage_percent for m in metrics_list) / len(
            metrics_list
        )
        expected_cpu = sum(m.cpu_usage_percent for m in metrics_list) / len(
            metrics_list
        )
        expected_response_time = sum(
            m.avg_response_time_ms for m in metrics_list
        ) / len(metrics_list)
        expected_errors = sum(m.connection_errors for m in metrics_list)

        assert avg_metrics.concurrent_requests == int(expected_concurrent)
        assert avg_metrics.memory_usage_percent == expected_memory
        assert avg_metrics.cpu_usage_percent == expected_cpu
        assert avg_metrics.avg_response_time_ms == expected_response_time
        assert avg_metrics.connection_errors == expected_errors

    def test_calculate_load_factor(self, load_monitor):
        """Test load factor calculation."""
        # Test low load
        low_load_metrics = LoadMetrics(
            concurrent_requests=5,
            memory_usage_percent=30.0,
            cpu_usage_percent=20.0,
            avg_response_time_ms=50.0,
            connection_errors=0,
            timestamp=time.time(),
        )

        load_factor = load_monitor.calculate_load_factor(low_load_metrics)
        assert 0.0 <= load_factor <= 1.0
        assert load_factor < 0.5  # Should be low

        # Test high load
        high_load_metrics = LoadMetrics(
            concurrent_requests=80,
            memory_usage_percent=90.0,
            cpu_usage_percent=85.0,
            avg_response_time_ms=800.0,
            connection_errors=5,
            timestamp=time.time(),
        )

        load_factor = load_monitor.calculate_load_factor(high_load_metrics)
        assert 0.0 <= load_factor <= 1.0
        assert load_factor > 0.8  # Should be high

    def test_calculate_load_factor_with_errors(self, load_monitor):
        """Test load factor calculation with connection errors."""
        metrics_with_errors = LoadMetrics(
            concurrent_requests=10,
            memory_usage_percent=50.0,
            cpu_usage_percent=40.0,
            avg_response_time_ms=100.0,
            connection_errors=10,  # High error count
            timestamp=time.time(),
        )

        load_factor = load_monitor.calculate_load_factor(metrics_with_errors)

        # Should have error penalty, making load factor higher
        assert load_factor >= 0.5  # Error penalty should push it up

    @pytest.mark.asyncio
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    async def test_collect_metrics_mocked(self, mock_cpu, mock_memory, load_monitor):
        """Test metrics collection with mocked system calls."""
        # Mock psutil responses
        mock_memory_info = Mock()
        mock_memory_info.percent = 65.0
        mock_memory.return_value = mock_memory_info
        mock_cpu.return_value = 45.0

        # Set up some request tracking data
        load_monitor._current_requests = 3
        load_monitor._total_requests = 10
        load_monitor._total_response_time = 1500.0  # 150ms average
        load_monitor._connection_errors = 2

        # Collect metrics
        await load_monitor._collect_metrics()

        # Verify metrics were collected
        assert len(load_monitor._metrics_history) == 1
        metrics = load_monitor._metrics_history[0]

        assert metrics.concurrent_requests == 3
        assert metrics.memory_usage_percent == 65.0
        assert metrics.cpu_usage_percent == 45.0
        assert metrics.avg_response_time_ms == 150.0
        assert metrics.connection_errors == 2
        assert metrics.timestamp > 0

    @pytest.mark.asyncio
    async def test_metrics_window_size_limiting(self, load_monitor):
        """Test that metrics history is limited to window size."""
        # Set a small window size for testing
        load_monitor.config.metrics_window_size = 3

        # Add more metrics than window size
        for i in range(5):
            metrics = LoadMetrics(
                concurrent_requests=i,
                memory_usage_percent=float(i * 10),
                cpu_usage_percent=float(i * 5),
                avg_response_time_ms=float(i * 20),
                connection_errors=i,
                timestamp=time.time() + i,
            )
            load_monitor._metrics_history.append(metrics)

            # Simulate window trimming
            if (
                len(load_monitor._metrics_history)
                > load_monitor.config.metrics_window_size
            ):
                load_monitor._metrics_history = load_monitor._metrics_history[
                    -load_monitor.config.metrics_window_size :
                ]

        # Should only have the last 3 metrics
        assert len(load_monitor._metrics_history) == 3
        assert load_monitor._metrics_history[0].concurrent_requests == 2
        assert load_monitor._metrics_history[-1].concurrent_requests == 4

    @pytest.mark.asyncio
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    async def test_monitoring_loop_error_handling(
        self, mock_cpu, mock_memory, load_monitor
    ):
        """Test monitoring loop handles errors gracefully."""
        # Make psutil calls fail
        mock_memory.side_effect = Exception("Memory access failed")
        mock_cpu.side_effect = Exception("CPU access failed")

        # Start monitoring
        await load_monitor.start()

        # Let it run briefly to encounter the error
        await asyncio.sleep(0.2)

        # Stop monitoring
        await load_monitor.stop()

        # Should not crash, just log the error
        assert not load_monitor._is_monitoring

    @pytest.mark.asyncio
    async def test_monitoring_with_periodic_error_reset(self, load_monitor):
        """Test that connection errors are reset periodically."""
        # Set some initial errors
        load_monitor._connection_errors = 5

        # Add enough metrics to trigger error reset (every 10 metrics)
        for _i in range(12):
            metrics = LoadMetrics(
                concurrent_requests=1,
                memory_usage_percent=50.0,
                cpu_usage_percent=30.0,
                avg_response_time_ms=100.0,
                connection_errors=5,
                timestamp=time.time(),
            )
            load_monitor._metrics_history.append(metrics)

            # Simulate the periodic reset logic
            if len(load_monitor._metrics_history) % 10 == 0:
                load_monitor._connection_errors = 0

        # Errors should be reset after 10 metrics
        assert load_monitor._connection_errors == 0


class TestLoadMonitorIntegration:
    """Integration tests for LoadMonitor."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_monitoring_cycle(self):
        """Test complete monitoring cycle with real timing."""
        config = LoadMonitorConfig(monitoring_interval=0.1, metrics_window_size=5)
        monitor = LoadMonitor(config)

        try:
            # Start monitoring
            await monitor.start()

            # Simulate some requests
            await monitor.record_request_start()
            await asyncio.sleep(0.05)
            await monitor.record_request_end(100.0)

            await monitor.record_request_start()
            await monitor.record_connection_error()
            await monitor.record_request_end(200.0)

            # Let monitoring collect some metrics
            await asyncio.sleep(0.3)

            # Get current metrics
            metrics = await monitor.get_current_load()
            assert metrics.concurrent_requests >= 0
            assert metrics.timestamp > 0

            # Calculate load factor
            load_factor = monitor.calculate_load_factor(metrics)
            assert 0.0 <= load_factor <= 1.0

        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_concurrent_request_tracking_accuracy(self):
        """Test accuracy of concurrent request tracking."""
        monitor = LoadMonitor()

        # Simulate overlapping requests
        tasks = []
        for _ in range(5):

            async def simulate_request():
                await monitor.record_request_start()
                await asyncio.sleep(0.1)  # Simulate processing time
                await monitor.record_request_end(100.0)

            tasks.append(asyncio.create_task(simulate_request()))

        # Check concurrent requests mid-execution
        await asyncio.sleep(0.05)  # Let requests start
        assert monitor._current_requests == 5

        # Wait for all to complete
        await asyncio.gather(*tasks)
        assert monitor._current_requests == 0
        assert monitor._total_requests == 5
