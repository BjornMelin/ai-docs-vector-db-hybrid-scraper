"""Progressive API interface for AI Documentation System.

This module provides elegant, progressive API interfaces that offer immediate value
while progressively revealing sophisticated features.

Design Philosophy:
- One-line setup for immediate gratification
- Progressive disclosure of advanced features
- Builder patterns for sophisticated configuration
- FastAPI-style dependency injection
- Enterprise-level type safety and error handling

Usage Patterns:
    Basic (30-second success):
        >>> system = AIDocSystem.quick_start()
        >>> results = await system.search("your query")

    Intermediate (builder pattern):
        >>> system = (
        ...     AIDocSystem.builder()
        ...     .with_embedding_provider("openai")
        ...     .with_cache(enabled=True)
        ...     .build()
        ... )

    Advanced (full configuration):
        >>> config = AdvancedConfig.builder()...
        >>> system = AIDocSystem(config=config)
"""

from .builders import (
    AdvancedConfigBuilder,
    AIDocSystemBuilder,
    EmbeddingConfigBuilder,
    SearchConfigBuilder,
)
from .factory import (
    FeatureDiscovery,
    create_embedding_service,
    create_search_service,
    create_system,
)
from .protocols import (
    CacheProtocol,
    DocumentProcessorProtocol,
    EmbeddingProtocol,
    SearchProtocol,
)
from .simple import AIDocSystem, SimpleSearchResult
from .types import (
    EmbeddingOptions,
    ProcessingOptions,
    ProgressiveResponse,
    SearchOptions,
)


__all__ = [
    # Core API
    "AIDocSystem",
    # Builders
    "AIDocSystemBuilder",
    "AdvancedConfigBuilder",
    "CacheProtocol",
    "DocumentProcessorProtocol",
    "EmbeddingConfigBuilder",
    "EmbeddingOptions",
    "EmbeddingProtocol",
    "FeatureDiscovery",
    "ProcessingOptions",
    "ProgressiveResponse",
    "SearchConfigBuilder",
    # Types
    "SearchOptions",
    # Protocols
    "SearchProtocol",
    "SimpleSearchResult",
    "create_embedding_service",
    "create_search_service",
    # Factory & Discovery
    "create_system",
]
