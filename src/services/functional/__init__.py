import typing

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

from .cache import cache_clear
from .cache import cache_delete
from .cache import cache_get
from .cache import cache_set
from .cache import get_cache_stats
from .circuit_breaker import CircuitBreakerConfig
from .circuit_breaker import CircuitBreakerState
from .circuit_breaker import circuit_breaker_middleware
from .circuit_breaker import create_circuit_breaker
from .crawling import crawl_site
from .crawling import crawl_url
from .crawling import get_crawl_metrics
from .dependencies import get_cache_client
from .dependencies import get_config
from .dependencies import get_embedding_client
from .dependencies import get_rate_limiter
from .dependencies import get_vector_db_client
from .embeddings import analyze_text_characteristics
from .embeddings import estimate_embedding_cost
from .embeddings import generate_embeddings
from .embeddings import rerank_results

__all__ = [
    # Circuit breaker
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "analyze_text_characteristics",
    "cache_clear",
    "cache_delete",
    # Cache functions
    "cache_get",
    "cache_set",
    "circuit_breaker_middleware",
    "crawl_site",
    # Crawling functions
    "crawl_url",
    "create_circuit_breaker",
    "estimate_embedding_cost",
    # Embedding functions
    "generate_embeddings",
    # Dependencies
    "get_cache_client",
    "get_cache_stats",
    "get_config",
    "get_crawl_metrics",
    "get_embedding_client",
    "get_rate_limiter",
    "get_vector_db_client",
    "rerank_results",
]
