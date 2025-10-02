"""Metrics collection integration tests.

This module tests metrics collection and aggregation across service boundaries,
validating counter, gauge, and histogram metrics with proper tagging and timing.
"""

from tests.fixtures.test_data_observability import ObservabilityTestDataFactory
from tests.fixtures.test_utils_observability import (
    MetricsTestHelper,
    ObservabilityTestAssertions,
)


class TestCounterMetrics:
    """Test counter metric collection and validation."""

    def test_record_simple_counter(self, mock_metrics_collector):
        """Test recording a simple counter metric."""
        collector = mock_metrics_collector

        collector.record_counter("api_requests_total", 1, {"endpoint": "/api/search"})

        metrics = collector.get_metrics("api_requests_total")
        assert len(metrics) == 1

        metric = metrics[0]
        assert metric.name == "api_requests_total"
        assert metric.value == 1
        assert metric.metric_type == "counter"
        assert metric.tags["endpoint"] == "/api/search"

    def test_increment_counter_multiple_times(self, mock_metrics_collector):
        """Test incrementing a counter multiple times."""
        collector = mock_metrics_collector

        collector.record_counter("api_requests_total", 1, {"endpoint": "/api/search"})
        collector.record_counter("api_requests_total", 2, {"endpoint": "/api/search"})
        collector.record_counter("api_requests_total", 3, {"endpoint": "/api/search"})

        metrics = collector.get_metrics("api_requests_total")
        assert len(metrics) == 3

        total_value = sum(m.value for m in metrics)
        assert total_value == 6

    def test_counter_with_different_tags(self, mock_metrics_collector):
        """Test counters with different tag combinations."""
        collector = mock_metrics_collector

        collector.record_counter(
            "api_requests_total", 1, {"endpoint": "/api/search", "method": "GET"}
        )
        collector.record_counter(
            "api_requests_total", 1, {"endpoint": "/api/search", "method": "POST"}
        )
        collector.record_counter(
            "api_requests_total", 1, {"endpoint": "/api/embed", "method": "POST"}
        )

        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) == 3

        # Check tag combinations
        tags_list = [m.tags for m in all_metrics]
        assert {"endpoint": "/api/search", "method": "GET"} in tags_list
        assert {"endpoint": "/api/search", "method": "POST"} in tags_list
        assert {"endpoint": "/api/embed", "method": "POST"} in tags_list


class TestGaugeMetrics:
    """Test gauge metric collection and validation."""

    def test_record_gauge_metric(self, mock_metrics_collector):
        """Test recording gauge metrics."""
        collector = mock_metrics_collector

        collector.record_gauge("memory_usage_mb", 512.5, {"service": "api_gateway"})
        collector.record_gauge("cpu_usage_percent", 75.2, {"service": "api_gateway"})

        memory_metrics = collector.get_metrics("memory_usage_mb")
        cpu_metrics = collector.get_metrics("cpu_usage_percent")

        assert len(memory_metrics) == 1
        assert len(cpu_metrics) == 1

        assert memory_metrics[0].value == 512.5
        assert memory_metrics[0].metric_type == "gauge"
        assert cpu_metrics[0].value == 75.2

    def test_gauge_updates_over_time(self, mock_metrics_collector):
        """Test gauge values changing over time."""
        collector = mock_metrics_collector

        # Initial value
        collector.record_gauge("queue_size", 0, {"service": "worker"})

        # Values increase
        collector.record_gauge("queue_size", 5, {"service": "worker"})
        collector.record_gauge("queue_size", 12, {"service": "worker"})

        # Values decrease
        collector.record_gauge("queue_size", 8, {"service": "worker"})
        collector.record_gauge("queue_size", 2, {"service": "worker"})

        metrics = collector.get_metrics("queue_size")
        assert len(metrics) == 5

        values = [m.value for m in metrics]
        assert values == [0, 5, 12, 8, 2]

        # Check timestamps are increasing
        timestamps = [m.timestamp for m in metrics]
        assert timestamps == sorted(timestamps)


class TestHistogramMetrics:
    """Test histogram metric collection and aggregation."""

    def test_record_histogram_metric(self, mock_metrics_collector):
        """Test recording histogram metrics."""
        collector = mock_metrics_collector

        collector.record_histogram(
            "request_duration_ms", 125.5, {"endpoint": "/api/search"}
        )
        collector.record_histogram(
            "request_duration_ms", 89.2, {"endpoint": "/api/search"}
        )
        collector.record_histogram(
            "request_duration_ms", 234.1, {"endpoint": "/api/search"}
        )

        metrics = collector.get_metrics("request_duration_ms")
        assert len(metrics) == 3

        for metric in metrics:
            assert metric.metric_type == "histogram"
            assert metric.tags["endpoint"] == "/api/search"

        values = [m.value for m in metrics]
        assert 89.2 in values
        assert 125.5 in values
        assert 234.1 in values


class TestMetricsAggregation:
    """Test metrics aggregation and analysis."""

    def test_aggregate_counter_by_tags(self, mock_metrics_collector):
        """Test aggregating counter metrics by tags."""
        collector = mock_metrics_collector

        # Record metrics with different tag combinations
        collector.record_counter(
            "api_requests_total", 10, {"service": "api", "endpoint": "/search"}
        )
        collector.record_counter(
            "api_requests_total", 5, {"service": "api", "endpoint": "/embed"}
        )
        collector.record_counter(
            "api_requests_total", 8, {"service": "worker", "endpoint": "/search"}
        )

        # Aggregate by service
        service_agg = MetricsTestHelper.aggregate_by_tag(
            collector.get_all_metrics(), "service"
        )

        assert service_agg["api"] == 15  # 10 + 5
        assert service_agg["worker"] == 8

        # Aggregate by endpoint
        endpoint_agg = MetricsTestHelper.aggregate_by_tag(
            collector.get_all_metrics(), "endpoint"
        )

        assert endpoint_agg["/search"] == 18  # 10 + 8
        assert endpoint_agg["/embed"] == 5

    def test_calculate_metric_rates(self, mock_metrics_collector):
        """Test calculating rates from counter metrics."""
        collector = mock_metrics_collector

        # Simulate metrics over time
        import time

        base_time = time.time()

        # Create metrics with controlled timestamps
        metric1 = ObservabilityTestDataFactory.create_sample_metric(
            "requests_total", 100, "counter", "api", {"period": "1m"}
        )
        metric1.timestamp = base_time

        metric2 = ObservabilityTestDataFactory.create_sample_metric(
            "requests_total", 150, "counter", "api", {"period": "1m"}
        )
        metric2.timestamp = base_time + 60  # 1 minute later

        # Manually add to collector
        collector.metrics.extend([metric1, metric2])

        # Calculate rate (requests per second)
        rate = MetricsTestHelper.calculate_rate([metric1, metric2])
        assert abs(rate - (50.0 / 60.0)) < 0.001  # (150 - 100) / 60 seconds

    def test_metrics_summary_statistics(self, mock_metrics_collector):
        """Test calculating summary statistics for metrics."""
        collector = mock_metrics_collector

        # Record response time metrics
        response_times = [120, 95, 180, 85, 145, 110, 200]
        for rt in response_times:
            collector.record_histogram("response_time_ms", rt, {"service": "api"})

        metrics = collector.get_metrics("response_time_ms")
        stats = MetricsTestHelper.calculate_summary_stats(metrics)

        assert stats["count"] == 7
        assert stats["min"] == 85
        assert stats["max"] == 200
        assert stats["sum"] == sum(response_times)
        assert abs(stats["avg"] - (sum(response_times) / len(response_times))) < 0.01


class TestServiceMetricsIntegration:
    """Integration tests for complete service metrics scenarios."""

    def test_end_to_end_service_metrics(self, mock_metrics_collector):
        """Test comprehensive metrics collection for a service."""
        collector = mock_metrics_collector

        # Simulate a service handling requests
        service_metrics = ObservabilityTestDataFactory.create_service_metrics_scenario()

        # Add metrics to collector
        for metric in service_metrics:
            if metric.metric_type == "counter":
                collector.record_counter(metric.name, metric.value, metric.tags)
            elif metric.metric_type == "gauge":
                collector.record_gauge(metric.name, metric.value, metric.tags)
            elif metric.metric_type == "histogram":
                collector.record_histogram(metric.name, metric.value, metric.tags)

        # Validate comprehensive metrics collection
        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) >= 3  # Should have at least the basic metric types

        # Check for expected metric types
        metric_types = {m.metric_type for m in all_metrics}
        assert "counter" in metric_types
        assert "gauge" in metric_types
        assert "histogram" in metric_types

        # Validate service-specific metrics
        ObservabilityTestAssertions.assert_service_metrics_complete(
            all_metrics, "api_gateway"
        )

    def test_multi_service_metrics_aggregation(self, mock_metrics_collector):
        """Test aggregating metrics across multiple services."""
        collector = mock_metrics_collector

        # Create metrics for multiple services
        services = ["api_gateway", "embedding_service", "vector_db_service"]

        for service in services:
            service_metrics = (
                ObservabilityTestDataFactory.create_service_metrics_scenario(service)
            )
            for metric in service_metrics:
                if metric.metric_type == "counter":
                    collector.record_counter(metric.name, metric.value, metric.tags)
                elif metric.metric_type == "gauge":
                    collector.record_gauge(metric.name, metric.value, metric.tags)

        # Aggregate by service
        all_metrics = collector.get_all_metrics()
        service_totals = MetricsTestHelper.aggregate_by_tag(all_metrics, "service")

        # All services should have metrics
        for service in services:
            assert service in service_totals
            assert service_totals[service] > 0

    def test_metrics_with_error_conditions(self, mock_metrics_collector):
        """Test metrics collection during error conditions."""
        collector = mock_metrics_collector

        # Normal operation metrics
        collector.record_counter("requests_total", 100, {"status": "success"})
        collector.record_gauge("error_rate", 0.02, {"service": "api"})

        # Error condition metrics
        collector.record_counter(
            "requests_total", 5, {"status": "error", "error_type": "timeout"}
        )
        collector.record_counter(
            "requests_total", 3, {"status": "error", "error_type": "connection"}
        )
        collector.record_gauge("error_rate", 0.08, {"service": "api"})

        # Validate error tracking
        error_metrics = [
            m for m in collector.get_all_metrics() if m.tags.get("status") == "error"
        ]

        assert len(error_metrics) == 2
        total_errors = sum(m.value for m in error_metrics)
        assert total_errors == 8

        # Check error rate increased
        error_rate_metrics = collector.get_metrics("error_rate")
        assert len(error_rate_metrics) == 2

        rates = sorted([m.value for m in error_rate_metrics])
        assert rates[0] == 0.02  # Initial rate
        assert rates[1] == 0.08  # Increased during errors
