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
        >>> system = (AIDocSystem.builder()
        ...     .with_embedding_provider("openai")
        ...     .with_cache(enabled=True)
        ...     .build())

    Advanced (full configuration):
        >>> config = AdvancedConfig.builder()...
        >>> system = AIDocSystem(config=config)
"""

from .simple import AIDocSystem, SimpleSearchResult
from .builders import (
    AIDocSystemBuilder,
    AdvancedConfigBuilder,
    EmbeddingConfigBuilder,
    SearchConfigBuilder,
)
from .protocols import (
    SearchProtocol,
    EmbeddingProtocol,
    DocumentProcessorProtocol,
    CacheProtocol,
)
from .types import (
    SearchOptions,
    EmbeddingOptions,
    ProcessingOptions,
    ProgressiveResponse,
)
from .factory import (
    create_system,
    create_embedding_service,
    create_search_service,
    FeatureDiscovery,
)

__all__ = [
    # Core API
    "AIDocSystem",
    "SimpleSearchResult",
    
    # Builders
    "AIDocSystemBuilder", 
    "AdvancedConfigBuilder",
    "EmbeddingConfigBuilder",
    "SearchConfigBuilder",
    
    # Protocols
    "SearchProtocol",
    "EmbeddingProtocol", 
    "DocumentProcessorProtocol",
    "CacheProtocol",
    
    # Types
    "SearchOptions",
    "EmbeddingOptions",
    "ProcessingOptions", 
    "ProgressiveResponse",
    
    # Factory & Discovery
    "create_system",
    "create_embedding_service",
    "create_search_service",
    "FeatureDiscovery",
]