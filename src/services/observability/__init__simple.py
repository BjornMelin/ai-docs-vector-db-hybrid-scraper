"""Simplified observability module for AI documentation scraper.

This module provides minimal but effective observability focused on AI/ML operations,
keeping only the most valuable tracking for a portfolio project.
"""

# Keep the impressive AI tracking
from .ai_tracking import (
    AIOperationMetrics,
    get_ai_tracker,
    track_embedding_generation,
    track_llm_call,
    track_rag_pipeline,
    track_vector_search,
)

# Use our wrapper for unified tracking
from .ai_tracking_wrapper import (
    simple_ai_decorator,
    track_ai_operation,
)

# Keep basic initialization
from .init import (
    initialize_observability,
    is_observability_enabled,
    shutdown_observability,
)

# Use simplified tracing
from .simple_tracing import (
    AICostTracker,
    cost_tracker,
    setup_tracing,
    trace_operation,
    track_ai_cost,
    track_vector_search as track_vector_search_simple,
)


__all__ = [
    # Simplified tracing
    "AICostTracker",
    # AI tracking (impressive for portfolio)
    "AIOperationMetrics",
    "cost_tracker",
    "get_ai_tracker",
    # Core functions
    "initialize_observability",
    "is_observability_enabled",
    "setup_tracing",
    "shutdown_observability",
    "simple_ai_decorator",
    "trace_operation",
    "track_ai_cost",
    # Unified tracking
    "track_ai_operation",
    "track_embedding_generation",
    "track_llm_call",
    "track_rag_pipeline",
    "track_vector_search",
    "track_vector_search_simple",
]
