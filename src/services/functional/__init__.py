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

from .auto_detection import (
    auto_configure_services,
    check_service_availability,
    detect_environment,
    discover_services,
    get_connection_info,
    get_service_metrics,
    test_service_endpoints,
)
from .cache import cache_clear, cache_delete, cache_get, cache_set, get_cache_stats
from .circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    circuit_breaker_middleware,
    create_circuit_breaker,
)
from .crawling import crawl_site, crawl_url, get_crawl_metrics
from .dependencies import (
    get_cache_client,
    get_config,
    get_embedding_client,
    get_vector_db_client,
)
from .deployment import (
    check_canary_readiness,
    create_ab_test,
    get_ab_test_variant,
    get_deployment_status,
    get_feature_flag,
    perform_blue_green_switch,
    rollback_deployment,
    set_feature_flag,
)
from .embeddings import (
    analyze_text_characteristics,
    estimate_embedding_cost,
    generate_embeddings,
    rerank_results,
)
from .enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    EnhancedCircuitBreakerConfig,
    create_enhanced_circuit_breaker,
    enhanced_circuit_breaker,
    enhanced_circuit_breaker_middleware,
    get_all_circuit_breaker_metrics,
    get_circuit_breaker_registry,
    register_circuit_breaker,
)
from .monitoring import (
    check_service_health,
    get_metrics_summary,
    get_performance_report,
    get_system_status,
    increment_counter,
    log_api_call,
    record_timer,
    set_gauge,
    timed,
)
from .rate_limiting import (
    acquire_rate_limit,
    get_rate_limit_status,
    get_rate_limiter,
    handle_api_response,
    rate_limited,
)


__all__ = [
    # Circuit breaker
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    # Enhanced circuit breaker
    "EnhancedCircuitBreaker",
    "EnhancedCircuitBreakerConfig",
    # Rate limiting functions
    "acquire_rate_limit",
    # Embedding functions
    "analyze_text_characteristics",
    # Auto-detection functions
    "auto_configure_services",
    # Cache functions
    "cache_clear",
    "cache_delete",
    "cache_get",
    "cache_set",
    # Deployment functions
    "check_canary_readiness",
    "check_service_availability",
    # Monitoring functions
    "check_service_health",
    "circuit_breaker_middleware",
    # Crawling functions
    "crawl_site",
    "crawl_url",
    "create_ab_test",
    "create_circuit_breaker",
    "create_enhanced_circuit_breaker",
    "detect_environment",
    "discover_services",
    "enhanced_circuit_breaker",
    "enhanced_circuit_breaker_middleware",
    "estimate_embedding_cost",
    "generate_embeddings",
    "get_ab_test_variant",
    "get_all_circuit_breaker_metrics",
    # Dependencies
    "get_cache_client",
    "get_cache_stats",
    "get_circuit_breaker_registry",
    "get_config",
    "get_connection_info",
    "get_crawl_metrics",
    "get_deployment_status",
    "get_embedding_client",
    "get_feature_flag",
    "get_metrics_summary",
    "get_performance_report",
    "get_rate_limit_status",
    "get_rate_limiter",
    "get_service_metrics",
    "get_system_status",
    "get_vector_db_client",
    "handle_api_response",
    "increment_counter",
    "log_api_call",
    "perform_blue_green_switch",
    "rate_limited",
    "record_timer",
    "register_circuit_breaker",
    "rerank_results",
    "rollback_deployment",
    "set_feature_flag",
    "set_gauge",
    "test_service_endpoints",
    "timed",
]
