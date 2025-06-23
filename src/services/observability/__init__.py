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
from .ai_tracking import AIOperationMetrics
from .ai_tracking import AIOperationTracker
from .ai_tracking import get_ai_tracker
from .ai_tracking import track_embedding_generation
from .ai_tracking import track_llm_call
from .ai_tracking import track_rag_pipeline
from .ai_tracking import track_vector_search
from .config import ObservabilityConfig
from .config import get_observability_config
from .config import get_resource_attributes
from .correlation import ErrorCorrelationTracker

# Trace correlation and context propagation
from .correlation import TraceCorrelationManager
from .correlation import correlated_operation
from .correlation import get_correlation_manager
from .correlation import get_current_trace_context
from .correlation import get_error_tracker
from .correlation import record_error
from .correlation import set_request_context
from .init import initialize_observability
from .init import is_observability_enabled
from .init import shutdown_observability
from .instrumentation import add_span_attribute
from .instrumentation import add_span_event
from .instrumentation import get_current_span_id
from .instrumentation import get_current_trace_id

# Advanced instrumentation
from .instrumentation import get_tracer
from .instrumentation import instrument_embedding_generation
from .instrumentation import instrument_function
from .instrumentation import instrument_llm_call
from .instrumentation import instrument_vector_search
from .instrumentation import set_business_context
from .instrumentation import set_user_context
from .instrumentation import trace_async_operation
from .instrumentation import trace_operation

# Metrics bridge
from .metrics_bridge import OpenTelemetryMetricsBridge
from .metrics_bridge import get_metrics_bridge
from .metrics_bridge import initialize_metrics_bridge
from .metrics_bridge import record_ai_operation as record_ai_metrics
from .metrics_bridge import record_cache_operation as record_cache_metrics
from .metrics_bridge import record_vector_search as record_vector_metrics
from .metrics_bridge import update_service_health

# Legacy imports for backward compatibility
from .middleware import FastAPIObservabilityMiddleware
from .performance import PerformanceMetrics

# Performance monitoring
from .performance import PerformanceMonitor
from .performance import PerformanceThresholds
from .performance import get_operation_statistics
from .performance import get_performance_monitor
from .performance import get_system_performance_summary
from .performance import initialize_performance_monitor
from .performance import monitor_ai_model_inference
from .performance import monitor_async_operation
from .performance import monitor_database_query
from .performance import monitor_external_api_call
from .performance import monitor_operation
from .tracking import record_ai_operation
from .tracking import track_cost

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
