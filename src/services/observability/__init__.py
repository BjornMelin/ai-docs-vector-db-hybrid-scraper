"""Comprehensive OpenTelemetry observability for AI/ML documentation system.

This module provides enterprise-grade observability with advanced instrumentation,
AI/ML operation tracking, trace correlation, performance monitoring, and metrics
bridging that integrates seamlessly with existing monitoring infrastructure.

Modern 2025 Features:
- Advanced auto-instrumentation for AI/ML operations
- Distributed tracing with context propagation
- Cost attribution and performance monitoring
- Error correlation and alerting
- Dual metrics export (OpenTelemetry + Prometheus)
"""

# Core configuration and initialization
from .config import (
    ObservabilityConfig,
    get_observability_config,
    get_resource_attributes,
)
from .init import (
    initialize_observability,
    shutdown_observability,
    is_observability_enabled,
)

# Advanced instrumentation
from .instrumentation import (
    get_tracer,
    instrument_function,
    instrument_vector_search,
    instrument_embedding_generation,
    instrument_llm_call,
    trace_operation,
    trace_async_operation,
    add_span_attribute,
    add_span_event,
    set_user_context,
    set_business_context,
    get_current_trace_id,
    get_current_span_id,
)

# AI/ML operation tracking
from .ai_tracking import (
    AIOperationMetrics,
    AIOperationTracker,
    get_ai_tracker,
    track_embedding_generation,
    track_llm_call,
    track_vector_search,
    track_rag_pipeline,
)

# Trace correlation and context propagation
from .correlation import (
    TraceCorrelationManager,
    ErrorCorrelationTracker,
    get_correlation_manager,
    get_error_tracker,
    set_request_context,
    correlated_operation,
    record_error,
    get_current_trace_context,
)

# Metrics bridge
from .metrics_bridge import (
    OpenTelemetryMetricsBridge,
    initialize_metrics_bridge,
    get_metrics_bridge,
    record_ai_operation as record_ai_metrics,
    record_vector_search as record_vector_metrics,
    record_cache_operation as record_cache_metrics,
    update_service_health,
)

# Performance monitoring
from .performance import (
    PerformanceMonitor,
    PerformanceMetrics,
    PerformanceThresholds,
    initialize_performance_monitor,
    get_performance_monitor,
    monitor_operation,
    monitor_async_operation,
    monitor_database_query,
    monitor_external_api_call,
    monitor_ai_model_inference,
    get_operation_statistics,
    get_system_performance_summary,
)

# Legacy imports for backward compatibility
from .middleware import FastAPIObservabilityMiddleware
from .tracking import record_ai_operation, track_cost

__all__ = [
    # Core configuration
    "ObservabilityConfig",
    "get_observability_config",
    "get_resource_attributes",
    "initialize_observability",
    "shutdown_observability",
    "is_observability_enabled",
    # Advanced instrumentation
    "get_tracer",
    "instrument_function",
    "instrument_vector_search",
    "instrument_embedding_generation",
    "instrument_llm_call",
    "trace_operation",
    "trace_async_operation",
    "add_span_attribute",
    "add_span_event",
    "set_user_context",
    "set_business_context",
    "get_current_trace_id",
    "get_current_span_id",
    # AI/ML tracking
    "AIOperationMetrics",
    "AIOperationTracker",
    "get_ai_tracker",
    "track_embedding_generation",
    "track_llm_call",
    "track_vector_search",
    "track_rag_pipeline",
    # Correlation and context
    "TraceCorrelationManager",
    "ErrorCorrelationTracker",
    "get_correlation_manager",
    "get_error_tracker",
    "set_request_context",
    "correlated_operation",
    "record_error",
    "get_current_trace_context",
    # Metrics bridge
    "OpenTelemetryMetricsBridge",
    "initialize_metrics_bridge",
    "get_metrics_bridge",
    "record_ai_metrics",
    "record_vector_metrics",
    "record_cache_metrics",
    "update_service_health",
    # Performance monitoring
    "PerformanceMonitor",
    "PerformanceMetrics",
    "PerformanceThresholds",
    "initialize_performance_monitor",
    "get_performance_monitor",
    "monitor_operation",
    "monitor_async_operation",
    "monitor_database_query",
    "monitor_external_api_call",
    "monitor_ai_model_inference",
    "get_operation_statistics",
    "get_system_performance_summary",
    # Legacy compatibility
    "FastAPIObservabilityMiddleware",
    "record_ai_operation",
    "track_cost",
]
