import typing


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
# AI/ML operation tracking
from .ai_tracking import (
    AIOperationMetrics,
    AIOperationTracker,
    get_ai_tracker,
    track_embedding_generation,
    track_llm_call,
    track_rag_pipeline,
    track_vector_search,
)
from .config import (
    ObservabilityConfig,
    get_observability_config,
    get_resource_attributes,
)

# NOTE: Configuration-specific instrumentation modules were removed:
# - config_instrumentation.py (over-engineered config tracking)
# - config_performance.py (custom config performance monitoring)
# Core observability features remain available through other modules
# Trace correlation and context propagation
from .correlation import (
    ErrorCorrelationTracker,
    TraceCorrelationManager,
    correlated_operation,
    get_correlation_manager,
    get_current_trace_context,
    get_error_tracker,
    record_error,
    set_request_context,
)
from .init import (
    initialize_observability,
    is_observability_enabled,
    shutdown_observability,
)

# Advanced instrumentation
from .instrumentation import (
    add_span_attribute,
    add_span_event,
    get_current_span_id,
    get_current_trace_id,
    get_tracer,
    instrument_embedding_generation,
    instrument_function,
    instrument_llm_call,
    instrument_vector_search,
    set_business_context,
    set_user_context,
    trace_async_operation,
    trace_operation,
)

# Metrics bridge
from .metrics_bridge import (
    OpenTelemetryMetricsBridge,
    get_metrics_bridge,
    initialize_metrics_bridge,
    record_ai_operation as record_ai_metrics,
    record_cache_operation as record_cache_metrics,
    record_vector_search as record_vector_metrics,
    update_service_health,
)

# Legacy imports for backward compatibility
from .middleware import FastAPIObservabilityMiddleware

# Performance monitoring
from .performance import (
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
from .tracking import record_ai_operation, track_cost


__all__ = [
    # AI/ML tracking
    "AIOperationMetrics",
    "AIOperationTracker",
    "ErrorCorrelationTracker",
    # Legacy compatibility
    "FastAPIObservabilityMiddleware",
    # Core configuration
    "ObservabilityConfig",
    # Metrics bridge
    "OpenTelemetryMetricsBridge",
    "PerformanceMetrics",
    # Performance monitoring
    "PerformanceMonitor",
    "PerformanceThresholds",
    # Correlation and context
    "TraceCorrelationManager",
    "add_span_attribute",
    "add_span_event",
    "correlated_operation",
    "get_ai_tracker",
    "get_correlation_manager",
    "get_current_span_id",
    "get_current_trace_context",
    "get_current_trace_id",
    "get_error_tracker",
    "get_metrics_bridge",
    "get_observability_config",
    "get_operation_statistics",
    "get_performance_monitor",
    "get_resource_attributes",
    "get_system_performance_summary",
    # Advanced instrumentation
    "get_tracer",
    "initialize_metrics_bridge",
    "initialize_observability",
    "initialize_performance_monitor",
    "instrument_embedding_generation",
    "instrument_function",
    "instrument_llm_call",
    "instrument_vector_search",
    "is_observability_enabled",
    "monitor_ai_model_inference",
    "monitor_async_operation",
    "monitor_database_query",
    "monitor_external_api_call",
    "monitor_operation",
    "record_ai_metrics",
    "record_ai_operation",
    "record_cache_metrics",
    "record_error",
    "record_vector_metrics",
    "set_business_context",
    "set_request_context",
    "set_user_context",
    "shutdown_observability",
    "trace_async_operation",
    "trace_operation",
    "track_cost",
    "track_embedding_generation",
    "track_llm_call",
    "track_rag_pipeline",
    "track_vector_search",
    "update_service_health",
]
