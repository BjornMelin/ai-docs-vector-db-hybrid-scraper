"""Tests for OpenTelemetry metrics bridge module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.services.observability.metrics_bridge import (
    OpenTelemetryMetricsBridge,
    initialize_metrics_bridge,
    get_metrics_bridge,
    record_ai_operation,
    record_vector_search,
    record_cache_operation,
    update_service_health,
)


class TestOpenTelemetryMetricsBridge:
    """Test OpenTelemetryMetricsBridge class."""

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_bridge_initialization(self, mock_metrics):
        """Test metrics bridge initialization."""
        # Mock meter and instruments
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        # Mock instrument creation
        counter = Mock()
        histogram = Mock()
        gauge = Mock()
        up_down_counter = Mock()
        
        meter.create_counter.return_value = counter
        meter.create_histogram.return_value = histogram
        meter.create_gauge.return_value = gauge
        meter.create_up_down_counter.return_value = up_down_counter
        
        bridge = OpenTelemetryMetricsBridge()
        
        assert bridge.meter is meter
        assert bridge.prometheus_registry is None
        assert len(bridge._instruments) > 0
        
        # Verify instrument creation calls
        assert meter.create_counter.call_count >= 3
        assert meter.create_histogram.call_count >= 4
        assert meter.create_gauge.call_count >= 4
        assert meter.create_up_down_counter.call_count >= 1

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_bridge_initialization_with_prometheus(self, mock_metrics):
        """Test bridge initialization with Prometheus registry."""
        prometheus_registry = Mock()
        
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        meter.create_counter.return_value = Mock()
        meter.create_histogram.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        bridge = OpenTelemetryMetricsBridge(prometheus_registry)
        
        assert bridge.prometheus_registry is prometheus_registry

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_ai_operation(self, mock_metrics):
        """Test recording AI operation metrics."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        # Mock instruments
        duration_hist = Mock()
        requests_counter = Mock()
        tokens_counter = Mock()
        cost_counter = Mock()
        
        # Setup side effects to return mocks for all 23 instrument creations
        # 8 histograms, 5 counters, 4 gauges, 1 up_down_counter, plus dynamic ones
        histograms = [duration_hist] + [Mock() for _ in range(10)]  # Extra for dynamic creation
        counters = [requests_counter, tokens_counter, cost_counter] + [Mock() for _ in range(10)]
        gauges = [Mock() for _ in range(10)]
        up_down_counters = [Mock() for _ in range(5)]
        
        meter.create_histogram.side_effect = histograms
        meter.create_counter.side_effect = counters
        meter.create_gauge.side_effect = gauges
        meter.create_up_down_counter.side_effect = up_down_counters
        
        bridge = OpenTelemetryMetricsBridge()
        
        bridge.record_ai_operation(
            operation_type="embedding_generation",
            provider="openai",
            model="text-embedding-ada-002",
            duration_ms=150.5,
            tokens_used=100,
            cost_usd=0.002,
            success=True
        )
        
        # Verify instrument calls
        duration_hist.record.assert_called_once()
        requests_counter.add.assert_called_once()
        tokens_counter.add.assert_called_once()
        cost_counter.add.assert_called_once()

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_ai_operation_without_optional_fields(self, mock_metrics):
        """Test recording AI operation without optional fields."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        duration_hist = Mock()
        requests_counter = Mock()
        
        meter.create_histogram.side_effect = [duration_hist] + [Mock()] * 15
        meter.create_counter.side_effect = [requests_counter] + [Mock()] * 15
        meter.create_gauge.side_effect = [Mock()] * 10
        meter.create_up_down_counter.side_effect = [Mock()] * 5
        
        bridge = OpenTelemetryMetricsBridge()
        
        bridge.record_ai_operation(
            operation_type="llm_call",
            provider="anthropic",
            model="claude-3-sonnet",
            duration_ms=800.0,
            success=False
        )
        
        # Should still record duration and requests
        duration_hist.record.assert_called_once()
        requests_counter.add.assert_called_once()

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_vector_search(self, mock_metrics):
        """Test recording vector search metrics."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        # Mock instruments - create enough mocks for all instrument types
        search_duration = Mock()
        search_results = Mock()
        search_quality = Mock()
        
        histograms = [Mock(), search_duration, search_results, search_quality] + [Mock()] * 10
        meter.create_histogram.side_effect = histograms
        meter.create_counter.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        bridge = OpenTelemetryMetricsBridge()
        
        bridge.record_vector_search(
            collection="documents",
            query_type="semantic",
            duration_ms=45.2,
            results_count=10,
            top_score=0.95,
            success=True
        )
        
        # Verify instrument calls
        search_duration.record.assert_called_once()
        search_results.record.assert_called_once()
        search_quality.record.assert_called_once()

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_vector_search_with_prometheus_bridge(self, mock_metrics):
        """Test vector search recording with Prometheus bridge."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        # Setup instruments
        histograms = [Mock()] * 20
        meter.create_histogram.side_effect = histograms
        meter.create_counter.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        # Mock Prometheus registry
        prometheus_registry = Mock()
        prometheus_registry._metrics = {
            "search_requests": Mock(),
            "search_duration": Mock()
        }
        
        # Mock Prometheus metric objects
        search_requests_metric = Mock()
        search_duration_metric = Mock()
        
        prometheus_registry._metrics["search_requests"].labels.return_value = search_requests_metric
        prometheus_registry._metrics["search_duration"].labels.return_value = search_duration_metric
        
        bridge = OpenTelemetryMetricsBridge(prometheus_registry)
        
        bridge.record_vector_search(
            collection="test_collection",
            query_type="hybrid",
            duration_ms=75.5,
            results_count=5,
            success=True
        )
        
        # Verify Prometheus metrics were updated
        search_requests_metric.inc.assert_called_once()
        search_duration_metric.observe.assert_called_once()

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_cache_operation(self, mock_metrics):
        """Test recording cache operation metrics."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        cache_ops_counter = Mock()
        cache_latency_hist = Mock()
        cache_hit_ratio_gauge = Mock()
        
        # Order matches the metrics bridge setup:
        # histograms: ai_operation_duration, vector_search_duration, vector_search_results, vector_search_quality, cache_latency
        histograms = [Mock(), Mock(), Mock(), Mock(), cache_latency_hist] + [Mock()] * 10
        # counters: ai_operation_requests, ai_tokens_used, ai_cost, cache_operations
        counters = [Mock(), Mock(), Mock(), cache_ops_counter] + [Mock()] * 10
        # 4 gauges including cache_hit_ratio
        gauges = [cache_hit_ratio_gauge] + [Mock()] * 10
        
        meter.create_histogram.side_effect = histograms
        meter.create_counter.side_effect = counters
        meter.create_gauge.side_effect = gauges
        meter.create_up_down_counter.side_effect = [Mock()] * 5
        
        bridge = OpenTelemetryMetricsBridge()
        
        bridge.record_cache_operation(
            cache_type="redis",
            operation="get",
            duration_ms=2.5,
            hit=True,
            cache_name="embedding_cache"
        )
        
        # Verify cache metrics
        cache_ops_counter.add.assert_called_once()
        cache_latency_hist.record.assert_called_once()

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_request_metrics(self, mock_metrics):
        """Test recording HTTP request metrics."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        request_duration = Mock()
        request_size = Mock()
        response_size = Mock()
        
        # Order: ai_operation_duration, vector_search_duration, vector_search_results, vector_search_quality, 
        # cache_latency, request_duration, request_size, response_size
        histograms = [Mock(), Mock(), Mock(), Mock(), Mock(), request_duration, request_size, response_size] + [Mock()] * 10
        
        meter.create_histogram.side_effect = histograms
        meter.create_counter.side_effect = [Mock()] * 15
        meter.create_gauge.side_effect = [Mock()] * 10
        meter.create_up_down_counter.side_effect = [Mock()] * 5
        
        bridge = OpenTelemetryMetricsBridge()
        
        bridge.record_request_metrics(
            method="POST",
            endpoint="/api/search",
            status_code=200,
            duration_ms=125.5,
            request_size_bytes=1024,
            response_size_bytes=2048
        )
        
        # Verify request metrics
        request_duration.record.assert_called_once()
        request_size.record.assert_called_once()
        response_size.record.assert_called_once()

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_error(self, mock_metrics):
        """Test recording error metrics."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        error_counter = Mock()
        counters = [Mock(), Mock(), Mock(), Mock(), error_counter] + [Mock()] * 10
        
        meter.create_histogram.return_value = Mock()
        meter.create_counter.side_effect = counters
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        bridge = OpenTelemetryMetricsBridge()
        
        bridge.record_error(
            error_type="validation_error",
            component="search_api",
            severity="high",
            user_impact="medium"
        )
        
        # Verify error counter
        error_counter.add.assert_called_once()

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_update_concurrent_operations(self, mock_metrics):
        """Test updating concurrent operations counter."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        concurrent_ops = Mock()
        
        meter.create_histogram.return_value = Mock()
        meter.create_counter.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = concurrent_ops
        
        bridge = OpenTelemetryMetricsBridge()
        
        # Test incrementing
        bridge.update_concurrent_operations("search", 1)
        concurrent_ops.add.assert_called_with(1, {"operation_type": "search"})
        
        # Test decrementing
        bridge.update_concurrent_operations("search", -1)
        concurrent_ops.add.assert_called_with(-1, {"operation_type": "search"})

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_update_queue_depth(self, mock_metrics):
        """Test updating queue depth gauge."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        queue_depth = Mock()
        # Order: cache_hit_ratio, error_rate, queue_depth, service_health, dependency_health
        gauges = [Mock(), Mock(), queue_depth] + [Mock()] * 10
        
        meter.create_histogram.side_effect = [Mock()] * 15
        meter.create_counter.side_effect = [Mock()] * 15
        meter.create_gauge.side_effect = gauges
        meter.create_up_down_counter.side_effect = [Mock()] * 5
        
        bridge = OpenTelemetryMetricsBridge()
        
        bridge.update_queue_depth("processing", 25)
        
        queue_depth.set.assert_called_once_with(25, {"queue_type": "processing"})

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_update_service_health(self, mock_metrics):
        """Test updating service health status."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        service_health = Mock()
        # Order: cache_hit_ratio, error_rate, queue_depth, service_health, dependency_health
        gauges = [Mock(), Mock(), Mock(), service_health] + [Mock()] * 10
        
        meter.create_histogram.side_effect = [Mock()] * 15
        meter.create_counter.side_effect = [Mock()] * 15
        meter.create_gauge.side_effect = gauges
        meter.create_up_down_counter.side_effect = [Mock()] * 5
        
        # Mock Prometheus registry
        prometheus_registry = Mock()
        prometheus_registry.update_service_health = Mock()
        
        bridge = OpenTelemetryMetricsBridge(prometheus_registry)
        
        # Test healthy service
        bridge.update_service_health("search_api", True)
        
        service_health.set.assert_called_with(1, {"service": "search_api"})
        prometheus_registry.update_service_health.assert_called_with("search_api", True)
        
        # Test unhealthy service
        bridge.update_service_health("database", False)
        
        service_health.set.assert_called_with(0, {"service": "database"})
        prometheus_registry.update_service_health.assert_called_with("database", False)

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_update_dependency_health(self, mock_metrics):
        """Test updating dependency health status."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        dependency_health = Mock()
        # Order: cache_hit_ratio, error_rate, queue_depth, service_health, dependency_health
        gauges = [Mock(), Mock(), Mock(), Mock(), dependency_health] + [Mock()] * 10
        
        meter.create_histogram.side_effect = [Mock()] * 15
        meter.create_counter.side_effect = [Mock()] * 15
        meter.create_gauge.side_effect = gauges
        meter.create_up_down_counter.side_effect = [Mock()] * 5
        
        bridge = OpenTelemetryMetricsBridge()
        
        bridge.update_dependency_health("qdrant", True)
        
        dependency_health.set.assert_called_once_with(1, {"dependency": "qdrant"})

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_batch_metrics(self, mock_metrics):
        """Test recording batch metrics."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        # Mock instruments
        counter = Mock()
        gauge = Mock()
        histogram = Mock()
        
        meter.create_histogram.return_value = histogram
        meter.create_counter.return_value = counter
        meter.create_gauge.return_value = gauge
        meter.create_up_down_counter.return_value = Mock()
        
        bridge = OpenTelemetryMetricsBridge()
        
        # Add mock instruments to bridge
        bridge._instruments["test_counter"] = counter
        bridge._instruments["test_gauge"] = gauge
        bridge._instruments["test_histogram"] = histogram
        
        metrics_batch = {
            "test_counter": {"value": 5, "labels": {"type": "test"}},
            "test_gauge": {"value": 0.85, "labels": {"service": "api"}},
            "test_histogram": {"value": 150.5, "labels": {"operation": "search"}}
        }
        
        # Mock the isinstance checks in the method since we're using Mock objects
        with patch('src.services.observability.metrics_bridge.isinstance') as mock_isinstance:
            # Make isinstance return appropriate values for our test objects
            def isinstance_side_effect(obj, type_class):
                if obj is counter:
                    return 'Counter' in str(type_class)
                elif obj is histogram:
                    return 'Histogram' in str(type_class)
                elif obj is gauge:
                    return False  # Force hasattr check for gauges
                return False
            
            mock_isinstance.side_effect = isinstance_side_effect
            
            bridge.record_batch_metrics(metrics_batch)
        
        # Verify batch recording
        counter.add.assert_called_once_with(5, {"type": "test"})
        gauge.set.assert_called_once_with(0.85, {"service": "api"})
        histogram.record.assert_called_once_with(150.5, {"operation": "search"})

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_create_custom_instruments(self, mock_metrics):
        """Test creating custom metric instruments."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        # Set up side effects for initialization only
        meter.create_counter.side_effect = [Mock()] * 15
        meter.create_gauge.side_effect = [Mock()] * 15
        meter.create_histogram.side_effect = [Mock()] * 15
        meter.create_up_down_counter.side_effect = [Mock()] * 5
        
        bridge = OpenTelemetryMetricsBridge()
        
        # Now set up specific returns for custom instrument creation
        custom_counter = Mock()
        custom_gauge = Mock()
        custom_histogram = Mock()
        
        meter.create_counter.return_value = custom_counter
        meter.create_gauge.return_value = custom_gauge
        meter.create_histogram.return_value = custom_histogram
        
        # Test custom counter creation
        result_counter = bridge.create_custom_counter(
            "custom_operations_total",
            "Total custom operations",
            "operations"
        )
        
        assert result_counter is not None
        assert "custom_operations_total" in bridge._instruments
        assert bridge._instruments["custom_operations_total"] is result_counter
        
        # Test custom gauge creation
        result_gauge = bridge.create_custom_gauge(
            "custom_status",
            "Custom status indicator"
        )
        
        assert result_gauge is not None
        assert "custom_status" in bridge._instruments
        assert bridge._instruments["custom_status"] is result_gauge
        
        # Test custom histogram creation
        result_histogram = bridge.create_custom_histogram(
            "custom_latency",
            "Custom operation latency",
            "ms",
            boundaries=[10, 50, 100, 500, 1000]
        )
        
        assert result_histogram is not None
        assert "custom_latency" in bridge._instruments
        assert bridge._instruments["custom_latency"] is result_histogram

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_get_instrument(self, mock_metrics):
        """Test getting instruments by name."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        counter = Mock()
        meter.create_histogram.return_value = Mock()
        # ai_operation_requests is the first counter created
        meter.create_counter.side_effect = [counter] + [Mock()] * 15
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        bridge = OpenTelemetryMetricsBridge()
        
        # Get existing instrument (use the key name, not the metric name)
        instrument = bridge.get_instrument("ai_operation_requests")
        assert instrument is counter
        
        # Get non-existent instrument
        missing = bridge.get_instrument("non_existent_metric")
        assert missing is None


class TestGlobalMetricsBridge:
    """Test global metrics bridge management."""

    def test_initialize_metrics_bridge(self):
        """Test initializing global metrics bridge."""
        prometheus_registry = Mock()
        
        with patch('src.services.observability.metrics_bridge.metrics'):
            bridge = initialize_metrics_bridge(prometheus_registry)
            
            assert isinstance(bridge, OpenTelemetryMetricsBridge)
            assert bridge.prometheus_registry is prometheus_registry

    def test_get_metrics_bridge(self):
        """Test getting global metrics bridge."""
        with patch('src.services.observability.metrics_bridge.metrics'):
            # Initialize first
            initialize_metrics_bridge()
            
            bridge1 = get_metrics_bridge()
            bridge2 = get_metrics_bridge()
            
            assert bridge1 is bridge2

    def test_get_metrics_bridge_not_initialized(self):
        """Test getting bridge when not initialized."""
        # Reset global instance
        import src.services.observability.metrics_bridge as bridge_module
        bridge_module._metrics_bridge = None
        
        with pytest.raises(RuntimeError, match="Metrics bridge not initialized"):
            get_metrics_bridge()

    def test_convenience_functions(self):
        """Test convenience functions."""
        with patch('src.services.observability.metrics_bridge.metrics'):
            initialize_metrics_bridge()
            
            with patch('src.services.observability.metrics_bridge.get_metrics_bridge') as mock_get:
                bridge = Mock()
                mock_get.return_value = bridge
                
                # Test AI operation recording
                record_ai_operation("embedding", "openai", "ada-002", 100.0)
                bridge.record_ai_operation.assert_called_once()
                
                # Test vector search recording
                record_vector_search("docs", "semantic", 50.0, 10)
                bridge.record_vector_search.assert_called_once()
                
                # Test cache operation recording
                record_cache_operation("redis", "get", 2.0, True)
                bridge.record_cache_operation.assert_called_once()
                
                # Test service health update
                update_service_health("api", True)
                bridge.update_service_health.assert_called_once()


class TestPrometheusIntegration:
    """Test Prometheus integration scenarios."""

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_bridge_with_prometheus_cache_methods(self, mock_metrics):
        """Test bridge with Prometheus cache methods."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        meter.create_histogram.return_value = Mock()
        meter.create_counter.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        # Mock Prometheus registry with cache methods
        prometheus_registry = Mock()
        prometheus_registry.record_cache_hit = Mock()
        prometheus_registry.record_cache_miss = Mock()
        
        bridge = OpenTelemetryMetricsBridge(prometheus_registry)
        
        # Test cache hit
        bridge.record_cache_operation("redis", "get", 1.5, True, "test_cache")
        prometheus_registry.record_cache_hit.assert_called_with("redis", "test_cache")
        
        # Test cache miss
        bridge.record_cache_operation("redis", "get", 5.0, False, "test_cache")
        prometheus_registry.record_cache_miss.assert_called_with("redis")

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_bridge_with_prometheus_embedding_cost(self, mock_metrics):
        """Test bridge with Prometheus embedding cost tracking."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        meter.create_histogram.return_value = Mock()
        meter.create_counter.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        # Mock Prometheus registry with embedding cost method
        prometheus_registry = Mock()
        prometheus_registry.record_embedding_cost = Mock()
        
        bridge = OpenTelemetryMetricsBridge(prometheus_registry)
        
        bridge.record_ai_operation(
            "embedding_generation", "openai", "ada-002", 100.0, cost_usd=0.005
        )
        
        prometheus_registry.record_embedding_cost.assert_called_with("openai", "ada-002", 0.005)

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_prometheus_bridge_error_handling(self, mock_metrics):
        """Test error handling in Prometheus bridge operations."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        histograms = [Mock()] * 20
        meter.create_histogram.side_effect = histograms
        meter.create_counter.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        # Mock Prometheus registry that raises errors
        prometheus_registry = Mock()
        prometheus_registry._metrics = {"search_requests": Mock(), "search_duration": Mock()}
        prometheus_registry._metrics["search_requests"].labels.side_effect = Exception("Prometheus error")
        
        bridge = OpenTelemetryMetricsBridge(prometheus_registry)
        
        # Should handle Prometheus errors gracefully
        bridge.record_vector_search("test", "semantic", 100.0, 5, success=True)
        
        # OpenTelemetry metrics should still be recorded
        histograms[1].record.assert_called()  # duration histogram


class TestMetricsBridgeEdgeCases:
    """Test edge cases and error scenarios."""

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_batch_metrics_with_invalid_instrument(self, mock_metrics):
        """Test batch metrics recording with invalid instrument types."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        meter.create_histogram.return_value = Mock()
        meter.create_counter.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        bridge = OpenTelemetryMetricsBridge()
        
        # Add invalid instrument type
        bridge._instruments["invalid_metric"] = "not_an_instrument"
        
        metrics_batch = {
            "invalid_metric": {"value": 10, "labels": {}},
            "non_existent_metric": {"value": 20, "labels": {}}
        }
        
        # Should handle invalid instruments gracefully
        bridge.record_batch_metrics(metrics_batch)

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_record_batch_metrics_with_exception(self, mock_metrics):
        """Test batch metrics recording with exceptions."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        failing_counter = Mock()
        failing_counter.add.side_effect = Exception("Instrument error")
        
        meter.create_histogram.return_value = Mock()
        meter.create_counter.side_effect = [failing_counter] + [Mock()] * 20
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        bridge = OpenTelemetryMetricsBridge()
        
        metrics_batch = {
            "ai_operation_requests": {"value": 1, "labels": {"provider": "test"}}
        }
        
        # Should handle instrument exceptions gracefully
        with patch('src.services.observability.metrics_bridge.logger') as mock_logger:
            with patch('src.services.observability.metrics_bridge.isinstance') as mock_isinstance:
                # Make isinstance return True for our failing counter
                mock_isinstance.side_effect = lambda obj, type_class: obj is failing_counter and 'Counter' in str(type_class)
                
                bridge.record_batch_metrics(metrics_batch)
                mock_logger.warning.assert_called()

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_create_custom_histogram_with_boundaries(self, mock_metrics):
        """Test creating custom histogram with specific boundaries."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        # Set up side effects for initialization
        meter.create_histogram.side_effect = [Mock()] * 15
        meter.create_counter.side_effect = [Mock()] * 15
        meter.create_gauge.side_effect = [Mock()] * 15  
        meter.create_up_down_counter.side_effect = [Mock()] * 5
        
        bridge = OpenTelemetryMetricsBridge()
        
        # Set up specific return for custom histogram creation
        custom_histogram = Mock()
        meter.create_histogram.return_value = custom_histogram
        
        boundaries = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
        
        result = bridge.create_custom_histogram(
            "custom_response_time",
            "Custom response time distribution",
            "ms",
            boundaries=boundaries
        )
        
        assert result is not None
        assert "custom_response_time" in bridge._instruments
        assert bridge._instruments["custom_response_time"] is result
        
        # Verify histogram was created with boundaries
        meter.create_histogram.assert_called_with(
            "custom_response_time",
            "Custom response time distribution",
            "ms",
            boundaries=boundaries
        )

    @patch('src.services.observability.metrics_bridge.metrics')  
    def test_instrument_name_conflicts(self, mock_metrics):
        """Test handling of instrument name conflicts."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        
        existing_counter = Mock()
        new_counter = Mock()
        
        meter.create_histogram.return_value = Mock()
        meter.create_counter.side_effect = [Mock()] * 20
        meter.create_gauge.side_effect = [Mock()] * 15
        meter.create_histogram.side_effect = [Mock()] * 15
        meter.create_up_down_counter.side_effect = [Mock()] * 5
        
        bridge = OpenTelemetryMetricsBridge()
        
        # Set up return value for the custom counter call
        meter.create_counter.return_value = new_counter
        
        # Try to create instrument with duplicate name (overwrite existing)
        result = bridge.create_custom_counter(
            "ai_operation_requests",  # This key already exists in _instruments
            "Duplicate counter",
            "requests"
        )
        
        # Should replace existing instrument and store the new one
        assert result is not None
        assert bridge._instruments["ai_operation_requests"] is result
        # Verify the meter.create_counter was called for the new instrument
        assert meter.create_counter.called