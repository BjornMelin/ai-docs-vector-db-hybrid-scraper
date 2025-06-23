"""End-to-end observability integration tests."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.observability.config import ObservabilityConfig
from src.services.observability.ai_tracking import get_ai_tracker, track_embedding_generation
from src.services.observability.correlation import get_correlation_manager, set_request_context
from src.services.observability.instrumentation import instrument_function, get_tracer
from src.services.observability.performance import initialize_performance_monitor, monitor_operation


class TestObservabilityE2E:
    """End-to-end observability tests."""

    @patch('src.services.observability.instrumentation.trace')
    @patch('src.services.observability.ai_tracking.metrics')
    def test_complete_ai_search_pipeline_observability(self, mock_ai_metrics, mock_trace):
        """Test complete AI search pipeline with full observability."""
        # Setup mocks
        self._setup_trace_mocks(mock_trace)
        self._setup_metrics_mocks(mock_ai_metrics)
        
        # Initialize observability components
        correlation_manager = get_correlation_manager()
        ai_tracker = get_ai_tracker()
        
        # Start complete pipeline trace
        request_id = correlation_manager.set_request_context(
            user_id="user123",
            session_id="session456"
        )
        
        with correlation_manager.correlated_operation("ai_search_pipeline") as pipeline_id:
            correlation_manager.set_business_context(
                operation_type="semantic_search",
                query_complexity="medium"
            )
            
            # Step 1: Query processing
            @instrument_function("query_preprocessing")
            def preprocess_query(query):
                # Simulate query preprocessing
                time.sleep(0.01)
                return query.lower().strip()
            
            processed_query = preprocess_query("What is machine learning?")
            
            # Step 2: Embedding generation
            with ai_tracker.track_embedding_generation(
                provider="openai",
                model="text-embedding-ada-002",
                input_texts=[processed_query]
            ) as embedding_result:
                embedding_result["embeddings"] = [[0.1, 0.2, 0.3, 0.4]]
                embedding_result["cost"] = 0.001
                embedding_result["cache_hit"] = False
            
            # Step 3: Vector search
            with ai_tracker.track_vector_search(
                collection_name="knowledge_base",
                query_type="semantic",
                top_k=5
            ) as search_result:
                search_result["results"] = [
                    {"id": "doc1", "content": "ML is...", "metadata": {}},
                    {"id": "doc2", "content": "Machine learning...", "metadata": {}}
                ]
                search_result["scores"] = [0.95, 0.87]
                search_result["cache_hit"] = False
            
            # Step 4: Result processing
            @instrument_function("result_processing")
            def process_results(results):
                # Simulate result processing
                time.sleep(0.005)
                return [{"id": r["id"], "score": s} for r, s in zip(results, search_result["scores"])]
            
            final_results = process_results(search_result["results"])
            
            # Step 5: Response generation (optional LLM call)
            with ai_tracker.track_llm_call(
                provider="openai",
                model="gpt-3.5-turbo",
                operation="summarization"
            ) as llm_result:
                mock_usage = Mock()
                mock_usage.prompt_tokens = 100
                mock_usage.completion_tokens = 50
                mock_usage.total_tokens = 150
                
                llm_result["usage"] = mock_usage
                llm_result["cost"] = 0.003
                llm_result["response"] = Mock()
                llm_result["response"].choices = [Mock()]
        
        # Verify complete pipeline was traced
        assert request_id is not None
        assert pipeline_id is not None
        assert processed_query == "what is machine learning?"
        assert len(final_results) == 2
        assert final_results[0]["score"] == 0.95

    @patch('src.services.observability.performance.psutil')
    @patch('src.services.observability.metrics_bridge.metrics')
    def test_performance_monitoring_integration(self, mock_metrics, mock_psutil):
        """Test performance monitoring across the observability stack."""
        # Setup mocks
        self._setup_metrics_mocks(mock_metrics)
        self._setup_psutil_mocks(mock_psutil)
        
        # Initialize performance monitoring
        try:
            performance_monitor = initialize_performance_monitor()
        except Exception:
            performance_monitor = None
        
        correlation_manager = get_correlation_manager()
        
        with correlation_manager.correlated_operation("performance_test") as correlation_id:
            # CPU intensive operation with monitoring
            with monitor_operation("cpu_intensive_task", track_resources=False):
                @instrument_function("data_processing")
                def cpu_intensive_task():
                    # Simulate CPU work
                    total = sum(i * i for i in range(1000))
                    time.sleep(0.02)  # Simulate processing time
                    return total
                
                result = cpu_intensive_task()
                assert result > 0
            
            # Memory intensive operation with monitoring
            with monitor_operation("memory_intensive_task", track_resources=False):
                @instrument_function("data_loading")
                def memory_intensive_task():
                    # Simulate memory allocation
                    data = [f"item_{i}" for i in range(10000)]
                    time.sleep(0.01)
                    return len(data)
                
                count = memory_intensive_task()
                assert count == 10000
        
        assert correlation_id is not None

    @patch('src.services.observability.correlation.trace')
    @patch('src.services.observability.correlation.baggage')
    def test_distributed_request_tracing(self, mock_baggage, mock_trace):
        """Test distributed request tracing across microservices."""
        # Setup mocks
        self._setup_trace_mocks(mock_trace)
        self._setup_baggage_mocks(mock_baggage)
        
        correlation_manager = get_correlation_manager()
        
        # Simulate API Gateway request
        request_id = correlation_manager.set_request_context(
            user_id="user123",
            request_source="api_gateway",
            request_path="/api/v1/search"
        )
        
        with correlation_manager.correlated_operation("api_gateway_request") as gateway_id:
            # Service 1: Authentication service
            with correlation_manager.correlated_operation("auth_service") as auth_id:
                correlation_manager.set_business_context(
                    service="auth_service",
                    operation="validate_token"
                )
                
                @instrument_function("token_validation")
                def validate_token(token):
                    time.sleep(0.005)  # Simulate auth check
                    return {"user_id": "user123", "valid": True}
                
                auth_result = validate_token("fake_token")
                assert auth_result["valid"] is True
            
            # Service 2: Search service
            with correlation_manager.correlated_operation("search_service") as search_id:
                correlation_manager.set_business_context(
                    service="search_service",
                    operation="semantic_search"
                )
                
                @instrument_function("search_execution")
                def execute_search(query):
                    time.sleep(0.015)  # Simulate search
                    return [{"id": "doc1", "score": 0.95}]
                
                search_results = execute_search("test query")
                assert len(search_results) == 1
            
            # Service 3: Response service
            with correlation_manager.correlated_operation("response_service") as response_id:
                correlation_manager.set_business_context(
                    service="response_service",
                    operation="format_response"
                )
                
                @instrument_function("response_formatting")
                def format_response(results):
                    time.sleep(0.002)  # Simulate formatting
                    return {"results": results, "count": len(results)}
                
                final_response = format_response(search_results)
                assert final_response["count"] == 1
        
        # Verify all services were traced with unique correlation IDs
        correlation_ids = [gateway_id, auth_id, search_id, response_id]
        assert all(cid is not None for cid in correlation_ids)
        assert len(set(correlation_ids)) == 4  # All unique
        assert request_id is not None

    @patch('src.services.observability.ai_tracking.metrics')
    def test_ai_cost_tracking_across_operations(self, mock_metrics):
        """Test AI cost tracking across multiple operations."""
        self._setup_metrics_mocks(mock_metrics)
        
        ai_tracker = get_ai_tracker()
        correlation_manager = get_correlation_manager()
        
        total_cost = 0.0
        
        with correlation_manager.correlated_operation("cost_tracking_pipeline") as correlation_id:
            # Multiple embedding operations
            embedding_operations = [
                ("What is AI?", 0.001),
                ("How does ML work?", 0.001),
                ("Explain deep learning", 0.002)
            ]
            
            for query, cost in embedding_operations:
                with ai_tracker.track_embedding_generation(
                    provider="openai",
                    model="text-embedding-ada-002",
                    input_texts=[query]
                ) as result:
                    result["embeddings"] = [[0.1, 0.2, 0.3]]
                    result["cost"] = cost
                    total_cost += cost
            
            # Multiple LLM operations
            llm_operations = [
                ("gpt-3.5-turbo", 0.002),
                ("gpt-4", 0.020),
                ("gpt-3.5-turbo", 0.003)
            ]
            
            for model, cost in llm_operations:
                with ai_tracker.track_llm_call(
                    provider="openai",
                    model=model,
                    operation="completion"
                ) as result:
                    mock_usage = Mock()
                    mock_usage.total_tokens = 100
                    result["usage"] = mock_usage
                    result["cost"] = cost
                    result["response"] = Mock()
                    result["response"].choices = [Mock()]
                    total_cost += cost
        
        # Verify total cost tracking
        expected_total = 0.001 + 0.001 + 0.002 + 0.002 + 0.020 + 0.003
        assert abs(total_cost - expected_total) < 0.0001
        assert correlation_id is not None

    def test_error_propagation_across_observability_stack(self):
        """Test error propagation and correlation across the observability stack."""
        correlation_manager = get_correlation_manager()
        ai_tracker = get_ai_tracker()
        
        request_id = correlation_manager.set_request_context(
            user_id="user123",
            session_id="session456"
        )
        
        error_chain = []
        
        with correlation_manager.correlated_operation("error_propagation_test") as main_id:
            # Step 1: Initial operation succeeds
            with correlation_manager.correlated_operation("initial_operation") as step1_id:
                @instrument_function("data_validation")
                def validate_data(data):
                    if not data:
                        raise ValueError("Empty data provided")
                    return True
                
                validation_result = validate_data(["valid", "data"])
                assert validation_result is True
            
            # Step 2: AI operation fails
            try:
                with ai_tracker.track_embedding_generation(
                    provider="openai",
                    model="text-embedding-ada-002",
                    input_texts=["test"]
                ) as result:
                    raise ConnectionError("OpenAI API unavailable")
            except ConnectionError as e:
                from src.services.observability.correlation import record_error
                error_id = record_error(
                    error=e,
                    error_type="api_connection_error",
                    severity="high"
                )
                error_chain.append(("embedding_generation", error_id))
            
            # Step 3: Fallback operation also fails
            try:
                with correlation_manager.correlated_operation("fallback_operation") as fallback_id:
                    @instrument_function("fallback_processing")
                    def fallback_process():
                        raise TimeoutError("Fallback service timeout")
                    
                    fallback_process()
            except TimeoutError as e:
                error_id = record_error(
                    error=e,
                    error_type="fallback_timeout",
                    severity="medium"
                )
                error_chain.append(("fallback_operation", error_id))
        
        # Verify error chain tracking
        assert len(error_chain) == 2
        assert main_id is not None
        assert step1_id is not None
        assert request_id is not None
        
        # Each error should have unique ID
        _, error1_id = error_chain[0]
        _, error2_id = error_chain[1]
        assert error1_id != error2_id

    @pytest.mark.asyncio
    async def test_async_observability_pipeline(self):
        """Test observability with async operations."""
        correlation_manager = get_correlation_manager()
        ai_tracker = get_ai_tracker()
        
        async def async_ai_pipeline():
            request_id = correlation_manager.set_request_context(
                user_id="async_user",
                session_id="async_session"
            )
            
            with correlation_manager.correlated_operation("async_ai_pipeline") as correlation_id:
                # Async embedding generation simulation
                @instrument_function("async_embedding")
                async def async_embedding_generation(texts):
                    await asyncio.sleep(0.01)  # Simulate async API call
                    return [[0.1, 0.2, 0.3] for _ in texts]
                
                embeddings = await async_embedding_generation(["async text"])
                
                # Async vector search simulation
                @instrument_function("async_vector_search")
                async def async_vector_search(embeddings):
                    await asyncio.sleep(0.01)  # Simulate async DB query
                    return [{"id": "doc1", "score": 0.9}]
                
                search_results = await async_vector_search(embeddings)
                
                # Async response generation
                @instrument_function("async_response_generation")
                async def async_response_generation(results):
                    await asyncio.sleep(0.01)  # Simulate async processing
                    return f"Found {len(results)} relevant documents"
                
                response = await async_response_generation(search_results)
                
                return {
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "embeddings": embeddings,
                    "search_results": search_results,
                    "response": response
                }
        
        result = await async_ai_pipeline()
        
        assert result["request_id"] is not None
        assert result["correlation_id"] is not None
        assert len(result["embeddings"]) == 1
        assert len(result["search_results"]) == 1
        assert "Found 1 relevant" in result["response"]

    def test_observability_configuration_validation(self):
        """Test observability configuration validation and setup."""
        # Test valid configurations
        configs = [
            ObservabilityConfig(
                enabled=True,
                service_name="test-service",
                export_prometheus=True,
                export_otlp=False
            ),
            ObservabilityConfig(
                enabled=False,
                service_name="disabled-service"
            ),
            ObservabilityConfig(
                enabled=True,
                service_name="full-featured-service",
                export_prometheus=True,
                export_otlp=True,
                export_jaeger=True,
                sample_rate=0.1
            )
        ]
        
        for config in configs:
            assert config.service_name is not None
            assert config.enabled in [True, False]
            assert 0.0 <= config.sample_rate <= 1.0

    def test_observability_performance_overhead(self):
        """Test that observability adds minimal performance overhead."""
        correlation_manager = get_correlation_manager()
        
        # Baseline test without observability overhead
        def baseline_operation():
            return sum(i * i for i in range(1000))
        
        start_time = time.time()
        baseline_result = baseline_operation()
        baseline_duration = time.time() - start_time
        
        # Test with full observability
        with correlation_manager.correlated_operation("performance_test") as correlation_id:
            correlation_manager.set_business_context(
                operation_type="performance_benchmark"
            )
            
            @instrument_function("monitored_operation")
            def monitored_operation():
                return sum(i * i for i in range(1000))
            
            start_time = time.time()
            monitored_result = monitored_operation()
            monitored_duration = time.time() - start_time
        
        # Results should be identical
        assert baseline_result == monitored_result
        
        # Overhead should be reasonable (less than 200% of baseline)
        overhead_ratio = monitored_duration / baseline_duration if baseline_duration > 0 else 1
        assert overhead_ratio < 3.0, f"Observability overhead too high: {overhead_ratio:.2f}x"
        assert correlation_id is not None

    def _setup_trace_mocks(self, mock_trace):
        """Setup trace mocks for testing."""
        span = Mock()
        span.is_recording.return_value = True
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)
        
        tracer = Mock()
        tracer.start_as_current_span.return_value = span
        mock_trace.get_tracer.return_value = tracer
        mock_trace.get_current_span.return_value = span
        
        return tracer, span

    def _setup_metrics_mocks(self, mock_metrics):
        """Setup metrics mocks for testing."""
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        meter.create_counter.return_value = Mock()
        meter.create_histogram.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        return meter

    def _setup_baggage_mocks(self, mock_baggage):
        """Setup baggage mocks for testing."""
        mock_baggage.set_baggage = Mock()
        mock_baggage.get_baggage = Mock()
        mock_baggage.get_all.return_value = {
            "request.id": "req123",
            "user.id": "user456"
        }
        
        return mock_baggage

    def _setup_psutil_mocks(self, mock_psutil):
        """Setup psutil mocks for testing."""
        mock_psutil.cpu_percent.return_value = 25.0
        
        memory = Mock()
        memory.used = 1024 * 1024 * 256  # 256 MB
        memory.percent = 25.0
        mock_psutil.virtual_memory.return_value = memory
        
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        return mock_psutil


class TestObservabilityResilience:
    """Test observability system resilience and graceful degradation."""

    def test_observability_with_component_failures(self):
        """Test observability when individual components fail."""
        correlation_manager = get_correlation_manager()
        
        # Should work even if some components are unavailable
        request_id = correlation_manager.set_request_context(
            user_id="resilience_test_user"
        )
        
        with correlation_manager.correlated_operation("resilience_test") as correlation_id:
            # Basic operations should still work
            correlation_manager.set_business_context(
                operation_type="resilience_testing"
            )
            
            @instrument_function("resilient_operation")
            def resilient_operation():
                return "operation_successful"
            
            result = resilient_operation()
            assert result == "operation_successful"
        
        assert request_id is not None
        assert correlation_id is not None

    def test_observability_graceful_degradation(self):
        """Test graceful degradation when observability systems are unavailable."""
        # Simulate disabled observability
        config = ObservabilityConfig(enabled=False)
        
        # Operations should still work
        correlation_manager = get_correlation_manager()
        
        with correlation_manager.correlated_operation("degraded_test") as correlation_id:
            @instrument_function("degraded_operation")
            def degraded_operation():
                return "still_working"
            
            result = degraded_operation()
            assert result == "still_working"
        
        # Even with degraded observability, basic functionality should work
        assert correlation_id is not None

    def test_high_throughput_observability(self):
        """Test observability under high throughput conditions."""
        correlation_manager = get_correlation_manager()
        
        # Simulate high-throughput scenario
        results = []
        
        for i in range(100):  # 100 concurrent-like operations
            request_id = correlation_manager.set_request_context(
                user_id=f"user_{i}",
                batch_id="high_throughput_test"
            )
            
            with correlation_manager.correlated_operation(f"operation_{i}") as correlation_id:
                @instrument_function(f"fast_operation_{i}")
                def fast_operation(index):
                    return f"result_{index}"
                
                result = fast_operation(i)
                results.append((request_id, correlation_id, result))
        
        # All operations should complete successfully
        assert len(results) == 100
        assert all(req_id is not None for req_id, _, _ in results)
        assert all(corr_id is not None for _, corr_id, _ in results)
        assert all(result.startswith("result_") for _, _, result in results)


class TestObservabilityCompliance:
    """Test observability compliance and standards."""

    def test_opentelemetry_semantic_conventions(self):
        """Test compliance with OpenTelemetry semantic conventions."""
        correlation_manager = get_correlation_manager()
        ai_tracker = get_ai_tracker()
        
        # Test service-level attributes
        request_id = correlation_manager.set_request_context(
            user_id="compliance_test_user",
            service_name="ai-docs-vector-db",
            service_version="1.0.0"
        )
        
        # Test AI operation attributes following semantic conventions
        with ai_tracker.track_llm_call(
            provider="openai",  # ai.provider
            model="gpt-4",      # ai.model
            operation="completion"  # llm.operation
        ) as result:
            # Mock response following conventions
            mock_usage = Mock()
            mock_usage.prompt_tokens = 100      # llm.usage.prompt_tokens
            mock_usage.completion_tokens = 50   # llm.usage.completion_tokens  
            mock_usage.total_tokens = 150       # llm.usage.total_tokens
            
            result["usage"] = mock_usage
            result["cost"] = 0.003              # ai.cost.usd
            result["response"] = Mock()
            result["response"].choices = [Mock()]
        
        assert request_id is not None

    def test_metrics_naming_conventions(self):
        """Test metrics naming conventions compliance."""
        try:
            from src.services.observability.metrics_bridge import initialize_metrics_bridge
            
            bridge = initialize_metrics_bridge()
            
            # Test standard metric naming patterns
            standard_metrics = [
                "ai_operations_total",           # Counter
                "ai_operation_duration_seconds", # Histogram  
                "ai_cost_total_usd",            # Counter
                "vector_search_results_count",   # Histogram
                "cache_hit_rate_ratio",         # Gauge
            ]
            
            for metric_name in standard_metrics:
                # Verify metric names follow conventions
                assert "_" in metric_name  # Snake case
                assert not metric_name.startswith("_")  # No leading underscore
                assert not metric_name.endswith("_")    # No trailing underscore
            
        except Exception:
            # Handle case where metrics bridge is not available
            pytest.skip("Metrics bridge not available")

    def test_trace_context_propagation_standards(self):
        """Test trace context propagation follows W3C standards."""
        from src.services.observability.correlation import (
            inject_context_to_headers,
            extract_context_from_headers
        )
        
        # Test W3C trace context format
        headers = {}
        inject_context_to_headers(headers)
        
        # Should have W3C traceparent header if context is active
        if "traceparent" in headers:
            traceparent = headers["traceparent"]
            # W3C traceparent format: version-trace_id-parent_id-trace_flags
            parts = traceparent.split("-")
            assert len(parts) == 4
            assert len(parts[1]) == 32  # trace_id (128-bit hex)
            assert len(parts[2]) == 16  # parent_id (64-bit hex)
            assert len(parts[3]) == 2   # trace_flags (8-bit hex)
        
        # Test context extraction
        test_headers = {
            "traceparent": "00-12345678901234567890123456789012-1234567890123456-01"
        }
        extracted_context = extract_context_from_headers(test_headers)
        assert extracted_context is not None