"""Tests for performance monitoring module."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

import src.services.observability.performance as perf_module
from src.services.observability.performance import (
    PerformanceMetrics,
    PerformanceMonitor,
    PerformanceThresholds,
    get_operation_statistics,
    get_performance_monitor,
    get_system_performance_summary,
    initialize_performance_monitor,
    monitor_ai_model_inference,
    monitor_async_operation,
    monitor_database_query,
    monitor_external_api_call,
    monitor_operation,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            duration_ms=150.5,
            cpu_usage_percent=45.2,
            memory_usage_mb=512.0,
            success=True,
        )

        assert metrics.operation_name == "test_operation"
        assert metrics.duration_ms == 150.5
        assert metrics.cpu_usage_percent == 45.2
        assert metrics.memory_usage_mb == 512.0
        assert metrics.success is True

    def test_performance_metrics_defaults(self):
        """Test default values in performance metrics."""
        metrics = PerformanceMetrics(operation_name="test_operation", duration_ms=100.0)

        assert metrics.cpu_usage_percent is None
        assert metrics.memory_usage_mb is None
        assert metrics.disk_io_mb is None
        assert metrics.network_io_mb is None
        assert metrics.success is True
        assert metrics.error_message is None
        assert isinstance(metrics.metadata, dict)
        assert len(metrics.metadata) == 0

    def test_performance_metrics_with_metadata(self):
        """Test performance metrics with metadata."""
        metadata = {"cache_hit": True, "retry_count": 2}

        metrics = PerformanceMetrics(
            operation_name="cached_operation", duration_ms=25.0, metadata=metadata
        )

        assert metrics.metadata == metadata
        assert metrics.metadata["cache_hit"] is True
        assert metrics.metadata["retry_count"] == 2


class TestPerformanceThresholds:
    """Test PerformanceThresholds dataclass."""

    def test_performance_thresholds_defaults(self):
        """Test default threshold values."""
        thresholds = PerformanceThresholds()

        assert thresholds.max_duration_ms == 30000
        assert thresholds.max_cpu_percent == 80.0
        assert thresholds.max_memory_mb == 1024.0
        assert thresholds.max_error_rate == 0.05
        assert thresholds.min_throughput_ops_per_sec == 1.0

    def test_performance_thresholds_custom(self):
        """Test custom threshold values."""
        thresholds = PerformanceThresholds(
            max_duration_ms=5000,
            max_cpu_percent=60.0,
            max_memory_mb=512.0,
            max_error_rate=0.01,
            min_throughput_ops_per_sec=10.0,
        )

        assert thresholds.max_duration_ms == 5000
        assert thresholds.max_cpu_percent == 60.0
        assert thresholds.max_memory_mb == 512.0
        assert thresholds.max_error_rate == 0.01
        assert thresholds.min_throughput_ops_per_sec == 10.0


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()

        assert isinstance(monitor.thresholds, PerformanceThresholds)
        assert monitor.tracer is not None
        assert isinstance(monitor._operation_history, dict)
        assert isinstance(monitor._error_counts, dict)
        assert isinstance(monitor._success_counts, dict)

    def test_monitor_initialization_with_custom_thresholds(self):
        """Test monitor initialization with custom thresholds."""
        thresholds = PerformanceThresholds(max_duration_ms=10000)
        monitor = PerformanceMonitor(thresholds)

        assert monitor.thresholds is thresholds
        assert monitor.thresholds.max_duration_ms == 10000

    @patch("src.services.observability.performance.psutil")
    def test_get_system_metrics(self, mock_psutil):
        """Test getting system metrics."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 45.5

        mock_memory = Mock()
        mock_memory.used = 1024 * 1024 * 512  # 512 MB
        mock_memory.percent = 45.0
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_disk_io = Mock()
        mock_disk_io.read_bytes = 1024 * 1024 * 100  # 100 MB
        mock_disk_io.write_bytes = 1024 * 1024 * 50  # 50 MB
        mock_psutil.disk_io_counters.return_value = mock_disk_io

        mock_net_io = Mock()
        mock_net_io.bytes_sent = 1024 * 1024 * 10  # 10 MB
        mock_net_io.bytes_recv = 1024 * 1024 * 20  # 20 MB
        mock_psutil.net_io_counters.return_value = mock_net_io

        monitor = PerformanceMonitor()
        metrics = monitor._get_system_metrics()

        assert metrics["cpu_percent"] == 45.5
        assert metrics["memory_used_mb"] == 512.0
        assert metrics["memory_percent"] == 45.0
        assert metrics["disk_read_mb"] == 100.0
        assert metrics["disk_write_mb"] == 50.0
        assert metrics["network_sent_mb"] == 10.0
        assert metrics["network_recv_mb"] == 20.0

    @patch("src.services.observability.performance.psutil")
    def test_get_system_metrics_with_exception(self, mock_psutil):
        """Test system metrics with psutil exception."""
        mock_psutil.cpu_percent.side_effect = Exception("psutil error")

        monitor = PerformanceMonitor()
        metrics = monitor._get_system_metrics()

        assert metrics == {}

    def test_monitor_operation_basic(self):
        """Test basic operation monitoring."""
        monitor = PerformanceMonitor()

        with monitor.monitor_operation("test_operation") as performance_data:
            time.sleep(0.01)  # Small delay to test timing
            performance_data["custom_metrics"]["test_value"] = 42

    def test_monitor_operation_with_categories(self):
        """Test operation monitoring with categories."""
        monitor = PerformanceMonitor()

        categories = ["database", "api", "ai_inference", "cache"]

        for category in categories:
            with monitor.monitor_operation(
                f"test_{category}_operation", category=category, track_resources=False
            ):
                time.sleep(0.001)

    def test_monitor_operation_with_exception(self):
        """Test operation monitoring with exceptions."""
        monitor = PerformanceMonitor()

        error_msg = "Test error"
        with (
            monitor.monitor_operation("failing_operation"),
            pytest.raises(ValueError, match="Test error"),
        ):
            raise ValueError(error_msg)

    def test_monitor_operation_without_resource_tracking(self):
        """Test operation monitoring without resource tracking."""
        monitor = PerformanceMonitor()

        with monitor.monitor_operation(
            "lightweight_operation", track_resources=False
        ) as performance_data:
            performance_data["custom_metrics"]["result"] = "success"

    def test_monitor_operation_without_threshold_alerts(self):
        """Test operation monitoring without threshold alerts."""
        monitor = PerformanceMonitor()

        with monitor.monitor_operation("no_alert_operation", alert_on_threshold=False):
            time.sleep(0.001)

    @pytest.mark.asyncio
    async def test_monitor_async_operation(self):
        """Test async operation monitoring."""
        monitor = PerformanceMonitor()

        async def async_test():
            async with monitor.monitor_async_operation(
                "async_test_operation"
            ) as performance_data:
                await asyncio.sleep(0.01)
                performance_data["custom_metrics"]["async_result"] = "success"

        await async_test()

    @pytest.mark.asyncio
    async def test_monitor_async_operation_with_exception(self):
        """Test async operation monitoring with exceptions."""
        monitor = PerformanceMonitor()

        async def async_failing_test():
            async with monitor.monitor_async_operation("async_failing_operation"):
                msg = "Async test error"
                raise ValueError(msg)

        with pytest.raises(ValueError):
            await async_failing_test()


class TestPerformanceTracking:
    """Test performance tracking and analysis."""

    def test_record_performance_metrics(self):
        """Test recording performance metrics."""
        monitor = PerformanceMonitor()

        metrics = PerformanceMetrics(
            operation_name="test_operation",
            duration_ms=150.0,
            cpu_usage_percent=45.0,
            memory_usage_mb=256.0,
            success=True,
        )

        monitor._record_performance_metrics(metrics, "test_category")

        # Verify metrics are recorded
        assert len(monitor._operation_history["test_operation"]) == 1
        assert monitor._success_counts["test_operation"] == 1
        assert monitor._error_counts["test_operation"] == 0

    def test_record_error_metrics(self):
        """Test recording error metrics."""
        monitor = PerformanceMonitor()

        error_metrics = PerformanceMetrics(
            operation_name="failing_operation",
            duration_ms=100.0,
            success=False,
            error_message="Test error",
        )

        monitor._record_performance_metrics(error_metrics, "error_category")

        assert len(monitor._operation_history["failing_operation"]) == 1
        assert monitor._success_counts["failing_operation"] == 0
        assert monitor._error_counts["failing_operation"] == 1

    def test_check_thresholds_duration(self):
        """Test threshold checking for duration."""
        thresholds = PerformanceThresholds(max_duration_ms=100.0)
        monitor = PerformanceMonitor(thresholds)

        # Mock span for threshold checking
        span = Mock()

        # Metrics exceeding duration threshold
        metrics = PerformanceMetrics(
            operation_name="slow_operation", duration_ms=200.0, success=True
        )

        monitor._check_thresholds(metrics, span)

        # Verify alert was recorded
        span.add_event.assert_called()
        span.set_attribute.assert_called_with("performance.alerts_count", 1)

    def test_check_thresholds_cpu(self):
        """Test threshold checking for CPU usage."""
        thresholds = PerformanceThresholds(max_cpu_percent=50.0)
        monitor = PerformanceMonitor(thresholds)

        span = Mock()

        metrics = PerformanceMetrics(
            operation_name="cpu_intensive_operation",
            duration_ms=100.0,
            cpu_usage_percent=75.0,
            success=True,
        )

        monitor._check_thresholds(metrics, span)

        span.add_event.assert_called()

    def test_check_thresholds_memory(self):
        """Test threshold checking for memory usage."""
        thresholds = PerformanceThresholds(max_memory_mb=512.0)
        monitor = PerformanceMonitor(thresholds)

        span = Mock()

        metrics = PerformanceMetrics(
            operation_name="memory_intensive_operation",
            duration_ms=100.0,
            memory_usage_mb=1024.0,
            success=True,
        )

        monitor._check_thresholds(metrics, span)

        span.add_event.assert_called()

    def test_check_thresholds_error_rate(self):
        """Test threshold checking for error rate."""
        thresholds = PerformanceThresholds(max_error_rate=0.1)
        monitor = PerformanceMonitor(thresholds)

        # Record multiple operations to establish error rate
        for i in range(20):
            success = i < 15  # 5 errors out of 20 = 25% error rate
            metrics = PerformanceMetrics(
                operation_name="error_prone_operation",
                duration_ms=100.0,
                success=success,
                error_message=None if success else f"Error {i}",
            )
            monitor._record_performance_metrics(metrics, "test")

        span = Mock()

        # Check thresholds with latest metrics
        monitor._check_thresholds(metrics, span)

        # Error rate (25%) exceeds threshold (10%)
        span.add_event.assert_called()

    def test_check_thresholds_no_violations(self):
        """Test threshold checking with no violations."""
        monitor = PerformanceMonitor()

        span = Mock()

        metrics = PerformanceMetrics(
            operation_name="good_operation",
            duration_ms=100.0,
            cpu_usage_percent=30.0,
            memory_usage_mb=256.0,
            success=True,
        )

        monitor._check_thresholds(metrics, span)

        # No alerts should be recorded
        span.add_event.assert_not_called()
        span.set_attribute.assert_not_called()


class TestOperationStatistics:
    """Test operation statistics and reporting."""

    def test_get_operation_statistics(self):
        """Test getting operation statistics."""
        monitor = PerformanceMonitor()

        # Record multiple operations
        for i in range(10):
            metrics = PerformanceMetrics(
                operation_name="test_operation",
                duration_ms=100.0 + i * 10,  # Varying durations
                cpu_usage_percent=30.0 + i * 2,
                memory_usage_mb=200.0 + i * 20,
                success=i < 8,  # 2 failures
            )
            monitor._record_performance_metrics(metrics, "test")

        stats = monitor.get_operation_statistics("test_operation")

        assert stats["operation_name"] == "test_operation"
        assert stats["_total_operations"] == 10
        assert stats["success_count"] == 8
        assert stats["error_count"] == 2
        assert stats["error_rate"] == 0.2

        # Check duration statistics
        duration_stats = stats["duration_stats"]
        assert duration_stats["min_ms"] == 100.0
        assert duration_stats["max_ms"] == 190.0
        assert duration_stats["avg_ms"] == 145.0

        # Check CPU and memory statistics
        assert "cpu_stats" in stats
        assert "memory_stats" in stats

    def test_get_operation_statistics_no_data(self):
        """Test getting statistics for non-existent operation."""
        monitor = PerformanceMonitor()

        stats = monitor.get_operation_statistics("non_existent_operation")

        assert "error" in stats
        assert stats["error"] == "No performance data available"

    def test_get_system_performance_summary(self):
        """Test getting system performance summary."""
        monitor = PerformanceMonitor()

        # Record some operations
        for i in range(5):
            metrics = PerformanceMetrics(
                operation_name=f"operation_{i}",
                duration_ms=100.0,
                success=i < 4,  # 1 error
            )
            monitor._record_performance_metrics(metrics, "test")

        summary = monitor.get_system_performance_summary()

        assert "timestamp" in summary
        assert "current_metrics" in summary
        assert "operation_count" in summary
        assert "_total_operations" in summary
        assert "_total_errors" in summary
        assert "overall_error_rate" in summary

        assert summary["operation_count"] == 5  # 5 different operations
        assert summary["_total_operations"] == 5
        assert summary["_total_errors"] == 1
        assert summary["overall_error_rate"] == 0.2


class TestSpecializedMonitoring:
    """Test specialized monitoring methods."""

    def test_monitor_database_query(self):
        """Test database query monitoring."""
        monitor = PerformanceMonitor()

        with monitor.monitor_database_query("select") as performance_data:
            time.sleep(0.001)
            performance_data["custom_metrics"]["rows_affected"] = 150

    def test_monitor_database_query_types(self):
        """Test different database query types."""
        monitor = PerformanceMonitor()

        query_types = ["select", "insert", "update", "delete"]

        for query_type in query_types:
            with monitor.monitor_database_query(query_type):
                time.sleep(0.001)

    def test_monitor_external_api_call(self):
        """Test external API call monitoring."""
        monitor = PerformanceMonitor()

        with monitor.monitor_external_api_call(
            "github", "get_user"
        ) as performance_data:
            time.sleep(0.01)
            performance_data["custom_metrics"]["status_code"] = 200

    def test_monitor_ai_model_inference(self):
        """Test AI model inference monitoring."""
        monitor = PerformanceMonitor()

        with monitor.monitor_ai_model_inference("gpt-4", "openai") as performance_data:
            time.sleep(0.1)
            performance_data["custom_metrics"]["tokens_generated"] = 150

    def test_monitor_cache_operation(self):
        """Test cache operation monitoring."""
        monitor = PerformanceMonitor()

        cache_operations = ["get", "set", "delete", "invalidate"]

        for operation in cache_operations:
            with monitor.monitor_cache_operation("redis", operation):
                time.sleep(0.001)


class TestGlobalMonitorInstance:
    """Test global monitor instance management."""

    def test_initialize_performance_monitor(self):
        """Test initializing global performance monitor."""
        thresholds = PerformanceThresholds(max_duration_ms=5000)
        monitor = initialize_performance_monitor(thresholds)

        assert isinstance(monitor, PerformanceMonitor)
        assert monitor.thresholds is thresholds

    def test_get_performance_monitor(self):
        """Test getting global performance monitor."""
        # Initialize first
        initialize_performance_monitor()

        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        assert monitor1 is monitor2

    def test_get_performance_monitor_not_initialized(self):
        """Test getting monitor when not initialized."""
        # Reset global instance

        perf_module._performance_monitor = None

        with pytest.raises(RuntimeError, match="Performance monitor not initialized"):
            get_performance_monitor()

    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience functions."""
        initialize_performance_monitor()

        # Test operation monitoring
        with monitor_operation("test_operation"):
            await asyncio.sleep(0.001)

        # Test async operation monitoring
        async def async_test():
            async with monitor_async_operation("async_operation"):
                await asyncio.sleep(0.001)

        await async_test()

        # Test specialized monitoring
        with monitor_database_query("select"):
            await asyncio.sleep(0.001)

        with monitor_external_api_call("api_service", "endpoint"):
            await asyncio.sleep(0.001)

        with monitor_ai_model_inference("gpt-4", "openai"):
            await asyncio.sleep(0.001)

        # Test statistics
        stats = get_operation_statistics("test_operation")
        assert isinstance(stats, dict)

        summary = get_system_performance_summary()
        assert isinstance(summary, dict)


@pytest.fixture
def mock_psutil():
    """Fixture providing mocked psutil operations."""
    with patch("src.services.observability.performance.psutil") as mock:
        mock.cpu_percent.return_value = 25.0

        memory = Mock()
        memory.used = 1024 * 1024 * 256  # 256 MB
        memory.percent = 25.0
        mock.virtual_memory.return_value = memory

        disk_io = Mock()
        disk_io.read_bytes = 1024 * 1024 * 10  # 10 MB
        disk_io.write_bytes = 1024 * 1024 * 5  # 5 MB
        mock.disk_io_counters.return_value = disk_io

        net_io = Mock()
        net_io.bytes_sent = 1024 * 1024 * 2  # 2 MB
        net_io.bytes_recv = 1024 * 1024 * 3  # 3 MB
        mock.net_io_counters.return_value = net_io

        yield mock


@pytest.fixture
def mock_metrics_bridge():
    """Fixture providing mocked metrics bridge."""
    with patch("src.services.observability.performance.get_metrics_bridge") as mock:
        bridge = Mock()
        bridge.record_batch_metrics = Mock()
        mock.return_value = bridge
        yield bridge


class TestPerformanceIntegration:
    """Test integration scenarios with mocked dependencies."""

    def test_monitor_with_mocked_psutil(self, mock_psutil):
        """Test performance monitor with mocked psutil."""
        monitor = PerformanceMonitor()

        with monitor.monitor_operation("test_operation", track_resources=True):
            time.sleep(0.01)

        # Verify psutil was called
        mock_psutil.cpu_percent.assert_called()
        mock_psutil.virtual_memory.assert_called()

    def test_monitor_with_mocked_metrics_bridge(self, mock_metrics_bridge):
        """Test performance monitor with mocked metrics bridge."""
        monitor = PerformanceMonitor()

        with monitor.monitor_operation("test_operation"):
            time.sleep(0.01)

        # Verify metrics bridge was called
        mock_metrics_bridge.record_batch_metrics.assert_called()

    def test_error_handling_with_mocked_failures(self, mock_psutil):
        """Test error handling with mocked failures."""
        # Make psutil fail
        mock_psutil.cpu_percent.side_effect = Exception("psutil failure")

        monitor = PerformanceMonitor()

        # Should still work despite psutil failure
        with monitor.monitor_operation("test_operation", track_resources=True):
            time.sleep(0.001)

    def test_threshold_violations_logging(self):
        """Test that threshold violations are properly logged."""
        thresholds = PerformanceThresholds(max_duration_ms=1.0)  # Very low threshold
        monitor = PerformanceMonitor(thresholds)

        with patch("src.services.observability.performance.logger") as mock_logger:
            with monitor.monitor_operation("slow_operation"):
                time.sleep(0.01)  # This should exceed 1ms threshold

            # Verify warning was logged
            mock_logger.warning.assert_called()


class TestPerformanceEdgeCases:
    """Test edge cases and error scenarios."""

    def test_monitor_operation_very_fast(self):
        """Test monitoring very fast operations."""
        monitor = PerformanceMonitor()

        with monitor.monitor_operation("fast_operation"):
            pass  # No delay - very fast operation

    def test_monitor_operation_nested(self):
        """Test nested operation monitoring."""
        monitor = PerformanceMonitor()

        with (
            monitor.monitor_operation("outer_operation"),
            monitor.monitor_operation("inner_operation"),
        ):
            time.sleep(0.001)

    @pytest.mark.asyncio
    async def test_concurrent_operation_monitoring(self):
        """Test concurrent operation monitoring."""
        monitor = PerformanceMonitor()

        async def concurrent_operation(op_id):
            async with monitor.monitor_async_operation(f"concurrent_op_{op_id}"):
                await asyncio.sleep(0.01)

        async def run_concurrent():
            tasks = [concurrent_operation(i) for i in range(5)]
            await asyncio.gather(*tasks)

        await run_concurrent()

    def test_operation_history_size_limit(self):
        """Test operation history respects size limit."""
        monitor = PerformanceMonitor()

        # Record more operations than the deque maxlen (100)
        for i in range(150):
            metrics = PerformanceMetrics(
                operation_name="limited_history_operation",
                duration_ms=100.0 + i,
                success=True,
            )
            monitor._record_performance_metrics(metrics, "test")

        # Should only keep last 100 operations
        history = monitor._operation_history["limited_history_operation"]
        assert len(history) == 100
        assert history[-1].duration_ms == 249.0  # Last recorded duration

    def test_statistics_with_empty_lists(self):
        """Test statistics calculation with empty data."""
        monitor = PerformanceMonitor()

        # Record operation without CPU/memory data
        metrics = PerformanceMetrics(
            operation_name="no_resource_operation", duration_ms=100.0, success=True
        )
        monitor._record_performance_metrics(metrics, "test")

        stats = monitor.get_operation_statistics("no_resource_operation")

        # Should handle missing CPU/memory data gracefully
        assert "duration_stats" in stats
        assert "cpu_stats" not in stats
        assert "memory_stats" not in stats
