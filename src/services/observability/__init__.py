"""Observability surface exported for consumers."""

from .config import (
    ObservabilityConfig,
    clear_observability_cache,
    get_observability_config,
    get_resource_attributes,
)
from .dependencies import (
    AITracerDep,
    ObservabilityConfigDep,
    ObservabilityServiceDep,
    get_observability_service,
    record_ai_operation_metrics,
    track_ai_cost_metrics,
)
from .init import (
    initialize_observability,
    is_observability_enabled,
    shutdown_observability,
)
from .tracing import (
    ConfigOperationType,
    current_trace_context,
    get_tracer,
    instrument_config_operation,
    log_extra_with_trace,
    set_span_attributes,
    span,
    trace_function,
)
from .tracking import (
    AIOperationTracker,
    PerformanceTracker,
    TraceCorrelationManager,
    get_ai_tracker,
    get_correlation_manager,
    record_ai_operation,
    track_cost,
    track_embedding_generation,
    track_llm_call,
    track_rag_pipeline,
    track_vector_search,
)


__all__ = [
    "AIOperationTracker",
    "AITracerDep",
    "ConfigOperationType",
    "ObservabilityConfig",
    "ObservabilityConfigDep",
    "ObservabilityServiceDep",
    "PerformanceTracker",
    "TraceCorrelationManager",
    "clear_observability_cache",
    "current_trace_context",
    "get_ai_tracker",
    "get_correlation_manager",
    "get_observability_config",
    "get_observability_service",
    "get_resource_attributes",
    "get_tracer",
    "initialize_observability",
    "instrument_config_operation",
    "is_observability_enabled",
    "log_extra_with_trace",
    "record_ai_operation",
    "record_ai_operation_metrics",
    "set_span_attributes",
    "shutdown_observability",
    "span",
    "trace_function",
    "track_ai_cost_metrics",
    "track_cost",
    "track_embedding_generation",
    "track_llm_call",
    "track_rag_pipeline",
    "track_vector_search",
]
