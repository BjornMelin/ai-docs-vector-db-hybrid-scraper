"""Function-based service layer with FastAPI dependency injection.

This module provides modern, function-based alternatives to the legacy Manager classes.
Uses FastAPI's dependency injection system with async patterns and circuit breaker resilience.

Architecture Principles:
- Function composition over inheritance
- Dependency injection via FastAPI Depends()
- Resource management with yield dependencies
- Circuit breaker patterns for resilience
- Stateless, testable functions
- Async-first patterns with proper error handling
"""

from .dependencies import (
    get_cache_client,
    get_config,
    get_embedding_client,
    get_rate_limiter,
    get_vector_db_client,
)
from .embeddings import (
    generate_embeddings,
    rerank_results,
    analyze_text_characteristics,
    estimate_embedding_cost,
)
from .cache import (
    cache_get,
    cache_set,
    cache_delete,
    cache_clear,
    get_cache_stats,
)
from .crawling import (
    crawl_url,
    crawl_site,
    get_crawl_metrics,
)
from .circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    create_circuit_breaker,
    circuit_breaker_middleware,
)

__all__ = [
    # Dependencies
    "get_cache_client",
    "get_config",
    "get_embedding_client",
    "get_rate_limiter",
    "get_vector_db_client",
    # Embedding functions
    "generate_embeddings",
    "rerank_results",
    "analyze_text_characteristics",
    "estimate_embedding_cost",
    # Cache functions
    "cache_get",
    "cache_set",
    "cache_delete",
    "cache_clear",
    "get_cache_stats",
    # Crawling functions
    "crawl_url",
    "crawl_site",
    "get_crawl_metrics",
    # Circuit breaker
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "create_circuit_breaker",
    "circuit_breaker_middleware",
]
