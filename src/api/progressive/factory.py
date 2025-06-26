"""Factory and feature discovery for the progressive API system.

This module provides factory functions and feature discovery mechanisms
that help users understand and access the sophisticated capabilities
of the AI Documentation System.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .simple import AIDocSystem
from .types import (
    QualityTier,
    SystemConfiguration,
    FeatureCapability,
    SystemCapabilities,
)
from .protocols import (
    SearchProtocol,
    EmbeddingProtocol,
    DocumentProcessorProtocol,
    CacheProtocol,
    MonitoringProtocol,
)


logger = logging.getLogger(__name__)


class FeatureLevel(str, Enum):
    """Feature sophistication levels."""
    BASIC = "basic"           # One-line setup, immediate value
    PROGRESSIVE = "progressive"  # Builder patterns, configurable
    EXPERT = "expert"         # Full control, custom implementations


@dataclass
class ProviderInfo:
    """Information about a service provider."""
    name: str
    description: str
    capabilities: List[str]
    requirements: List[str]
    cost_model: str
    performance_tier: QualityTier
    example_usage: str


class FeatureDiscovery:
    """Feature discovery and capability introspection.
    
    This class helps users understand what features are available
    and how to access them as they become more sophisticated.
    
    Examples:
        Basic discovery:
            >>> discovery = FeatureDiscovery()
            >>> features = discovery.get_basic_features()
        
        Provider discovery:
            >>> providers = discovery.discover_embedding_providers()
            >>> openai_info = discovery.get_provider_info("openai")
        
        Learning path:
            >>> path = discovery.get_learning_path("beginner")
    """
    
    def __init__(self):
        """Initialize feature discovery."""
        self._capabilities_cache: Optional[SystemCapabilities] = None
    
    def get_system_capabilities(self) -> SystemCapabilities:
        """Get complete system capabilities overview.
        
        Returns:
            SystemCapabilities with all features and learning paths
        """
        if self._capabilities_cache is None:
            self._capabilities_cache = self._build_capabilities()
        return self._capabilities_cache
    
    def get_basic_features(self) -> List[FeatureCapability]:
        """Get basic features for immediate use.
        
        Returns:
            List of basic feature capabilities
        """
        capabilities = self.get_system_capabilities()
        return capabilities.basic_features
    
    def get_progressive_features(self) -> List[FeatureCapability]:
        """Get progressive features for building sophistication.
        
        Returns:
            List of progressive feature capabilities
        """
        capabilities = self.get_system_capabilities()
        return capabilities.progressive_features
    
    def get_expert_features(self) -> List[FeatureCapability]:
        """Get expert features for maximum control.
        
        Returns:
            List of expert feature capabilities
        """
        capabilities = self.get_system_capabilities()
        return capabilities.expert_features
    
    def discover_embedding_providers(self) -> List[ProviderInfo]:
        """Discover available embedding providers.
        
        Returns:
            List of embedding provider information
        """
        return [
            ProviderInfo(
                name="fastembed",
                description="Local embedding models with fast inference",
                capabilities=["multilingual", "code_understanding", "local_inference"],
                requirements=["disk_space"],
                cost_model="free",
                performance_tier=QualityTier.FAST,
                example_usage="EmbeddingConfigBuilder().with_provider('fastembed')",
            ),
            ProviderInfo(
                name="openai",
                description="OpenAI's state-of-the-art embedding models",
                capabilities=["high_quality", "large_context", "multilingual"],
                requirements=["api_key", "internet_connection"],
                cost_model="pay_per_token",
                performance_tier=QualityTier.BEST,
                example_usage="EmbeddingConfigBuilder().with_provider('openai')",
            ),
            ProviderInfo(
                name="huggingface",
                description="Hugging Face transformer models",
                capabilities=["diverse_models", "customizable", "research_models"],
                requirements=["internet_connection", "model_download"],
                cost_model="free",
                performance_tier=QualityTier.BALANCED,
                example_usage="EmbeddingConfigBuilder().with_provider('huggingface')",
            ),
        ]
    
    def discover_search_strategies(self) -> List[Dict[str, Any]]:
        """Discover available search strategies.
        
        Returns:
            List of search strategy information
        """
        return [
            {
                "name": "vector",
                "description": "Pure vector similarity search",
                "best_for": ["semantic similarity", "concept matching"],
                "performance": "fast",
                "accuracy": "good",
                "example": "SearchConfigBuilder().with_strategy('vector')",
            },
            {
                "name": "hybrid",
                "description": "Combines vector and keyword search",
                "best_for": ["general purpose", "balanced results"],
                "performance": "medium",
                "accuracy": "excellent",
                "example": "SearchConfigBuilder().with_strategy('hybrid')",
            },
            {
                "name": "semantic",
                "description": "Vector search with neural reranking",
                "best_for": ["high precision", "complex queries"],
                "performance": "slower",
                "accuracy": "best",
                "example": "SearchConfigBuilder().with_strategy('semantic')",
            },
            {
                "name": "adaptive",
                "description": "Automatically selects best strategy",
                "best_for": ["ease of use", "varied query types"],
                "performance": "variable",
                "accuracy": "optimized",
                "example": "SearchConfigBuilder().with_strategy('adaptive')",
            },
        ]
    
    def get_provider_info(self, provider_name: str) -> Optional[ProviderInfo]:
        """Get detailed information about a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider information or None if not found
        """
        providers = self.discover_embedding_providers()
        for provider in providers:
            if provider.name == provider_name:
                return provider
        return None
    
    def get_learning_path(self, experience_level: str = "beginner") -> List[str]:
        """Get recommended learning path based on experience level.
        
        Args:
            experience_level: "beginner", "intermediate", or "advanced"
            
        Returns:
            List of recommended learning steps
        """
        paths = {
            "beginner": [
                "Start with AIDocSystem.quick_start() for immediate results",
                "Try basic search: await system.search('your query')",
                "Add documents: await system.add_document(content='...')",
                "Explore search results: result.get_analysis()",
                "Check system stats: await system.get_stats()",
            ],
            "intermediate": [
                "Use the builder pattern: AIDocSystem.builder().build()",
                "Configure search options: SearchOptions(rerank=True)",
                "Set up caching: builder.with_cache(enabled=True, ttl=3600)",
                "Try different embedding providers",
                "Enable monitoring: builder.with_monitoring(enabled=True)",
            ],
            "advanced": [
                "Create custom configurations with AdvancedConfigBuilder",
                "Implement custom protocols (SearchProtocol, EmbeddingProtocol)",
                "Use experimental features and debug mode",
                "Set up distributed caching and monitoring",
                "Implement custom processing pipelines",
            ],
        }
        return paths.get(experience_level, paths["beginner"])
    
    def _build_capabilities(self) -> SystemCapabilities:
        """Build the complete system capabilities description."""
        basic_features = [
            FeatureCapability(
                name="Quick Start",
                description="One-line setup for immediate use",
                level="basic",
                example="system = AIDocSystem.quick_start()",
                documentation_url="/docs/quick-start",
            ),
            FeatureCapability(
                name="Simple Search",
                description="Basic document search functionality", 
                level="basic",
                example="results = await system.search('machine learning')",
                documentation_url="/docs/search",
            ),
            FeatureCapability(
                name="Document Addition",
                description="Add documents to the system",
                level="basic",
                example="doc_id = await system.add_document(content='...')",
                documentation_url="/docs/documents",
            ),
            FeatureCapability(
                name="URL Crawling",
                description="Automatically crawl and index web content",
                level="basic",
                example="ids = await system.add_documents_from_urls(['url1', 'url2'])",
                documentation_url="/docs/crawling",
            ),
        ]
        
        progressive_features = [
            FeatureCapability(
                name="Builder Pattern",
                description="Progressive configuration through builders",
                level="progressive",
                example="system = AIDocSystem.builder().with_cache(ttl=3600).build()",
                documentation_url="/docs/builders",
            ),
            FeatureCapability(
                name="Advanced Search",
                description="Configurable search with multiple strategies",
                level="progressive", 
                example="options = SearchOptions(strategy='hybrid', rerank=True)",
                documentation_url="/docs/advanced-search",
            ),
            FeatureCapability(
                name="Provider Selection",
                description="Choose between different embedding providers",
                level="progressive",
                example="builder.with_embedding_provider('openai')",
                documentation_url="/docs/providers",
            ),
            FeatureCapability(
                name="Performance Monitoring",
                description="Track system performance and costs",
                level="progressive",
                example="builder.with_monitoring(track_costs=True)",
                documentation_url="/docs/monitoring",
            ),
        ]
        
        expert_features = [
            FeatureCapability(
                name="Custom Protocols",
                description="Implement custom search and embedding protocols",
                level="expert",
                example="class CustomSearch(SearchProtocol): ...",
                requirements=["protocol_knowledge"],
                documentation_url="/docs/protocols",
            ),
            FeatureCapability(
                name="Advanced Configuration",
                description="Full system configuration control",
                level="expert",
                example="config = AdvancedConfigBuilder().with_experimental_features(...)",
                requirements=["system_knowledge"],
                documentation_url="/docs/advanced-config",
            ),
            FeatureCapability(
                name="Custom Processing",
                description="Implement custom document processing pipelines",
                level="expert",
                example="class CustomProcessor(DocumentProcessorProtocol): ...",
                requirements=["nlp_knowledge"],
                documentation_url="/docs/processing",
            ),
        ]
        
        return SystemCapabilities(
            basic_features=basic_features,
            progressive_features=progressive_features,
            expert_features=expert_features,
            next_steps=[
                "Try the quick start guide",
                "Explore different embedding providers",
                "Set up monitoring for production use",
                "Learn about custom protocols",
            ],
            learning_path=[
                "Basic usage patterns",
                "Progressive configuration",
                "Advanced features",
                "Expert customization",
            ],
            examples={
                "quick_start": "AIDocSystem.quick_start()",
                "builder": "AIDocSystem.builder().with_cache(enabled=True).build()",
                "search": "await system.search('query', options=SearchOptions(rerank=True))",
                "monitoring": "stats = await system.get_stats()",
            },
        )


def create_system(
    *,
    provider: str = "fastembed",
    quality: str = "balanced", 
    workspace: Optional[Union[str, Path]] = None,
    **kwargs,
) -> AIDocSystem:
    """Factory function for creating AI Documentation Systems.
    
    This provides a clean factory interface for system creation
    with sensible defaults and progressive options.
    
    Args:
        provider: Embedding provider ("fastembed", "openai")
        quality: Quality tier ("fast", "balanced", "best")
        workspace: Workspace directory
        **kwargs: Additional configuration options
        
    Returns:
        Configured AIDocSystem instance
        
    Examples:
        Basic:
            >>> system = create_system()
        
        With options:
            >>> system = create_system(
            ...     provider="openai",
            ...     quality="best", 
            ...     enable_cache=True,
            ...     enable_monitoring=True
            ... )
    """
    return AIDocSystem(
        embedding_provider=provider,
        quality_tier=quality,
        workspace_dir=workspace,
        **kwargs,
    )


async def create_embedding_service(
    provider: str = "fastembed",
    **kwargs,
) -> EmbeddingProtocol:
    """Factory function for creating embedding services.
    
    Args:
        provider: Embedding provider name
        **kwargs: Provider-specific configuration
        
    Returns:
        Configured embedding service
    """
    from src.services.embeddings.manager import EmbeddingManager
    from src.config import get_settings
    from src.infrastructure.client_manager import ClientManager
    
    config = get_settings()
    client_manager = ClientManager()
    
    manager = EmbeddingManager(
        config=config,
        client_manager=client_manager,
        **kwargs,
    )
    
    await manager.initialize()
    return manager


async def create_search_service(
    embedding_service: Optional[EmbeddingProtocol] = None,
    **kwargs,
) -> SearchProtocol:
    """Factory function for creating search services.
    
    Args:
        embedding_service: Optional embedding service instance
        **kwargs: Search service configuration
        
    Returns:
        Configured search service
    """
    from src.services.vector_db.search import VectorSearchService
    from src.config import get_settings
    
    config = get_settings()
    
    if embedding_service is None:
        embedding_service = await create_embedding_service()
    
    service = VectorSearchService(
        config=config,
        embedding_manager=embedding_service,
        **kwargs,
    )
    
    await service.initialize()
    return service


def discover_features() -> FeatureDiscovery:
    """Create a feature discovery instance.
    
    Returns:
        FeatureDiscovery instance for capability exploration
        
    Examples:
        >>> discovery = discover_features()
        >>> basic_features = discovery.get_basic_features()
        >>> providers = discovery.discover_embedding_providers()
    """
    return FeatureDiscovery()


def get_quick_examples() -> Dict[str, str]:
    """Get quick examples for common use cases.
    
    Returns:
        Dictionary of example code snippets
    """
    return {
        "quick_start": """
# 30-second setup
system = AIDocSystem.quick_start()
async with system:
    results = await system.search("machine learning")
    for result in results:
        print(f"{result.title}: {result.content[:100]}...")
        """,
        
        "builder_pattern": """
# Progressive configuration
system = (AIDocSystem.builder()
    .with_embedding_provider("openai")
    .with_cache(enabled=True, ttl=3600)
    .with_monitoring(enabled=True, track_costs=True)
    .build())
        """,
        
        "advanced_search": """
# Sophisticated search options
options = SearchOptions(
    strategy=SearchStrategy.SEMANTIC,
    rerank=True,
    include_embeddings=True,
    similarity_threshold=0.7,
    diversity_factor=0.3
)
results = await system.search("query", options=options)
        """,
        
        "document_processing": """
# Batch document processing
urls = ["https://docs.example.com", "https://api.example.com"]
doc_ids = await system.add_documents_from_urls(
    urls, 
    batch_size=5,
    progress_callback=lambda p, msg: print(f"{p:.1%}: {msg}")
)
        """,
        
        "feature_discovery": """
# Discover available features
discovery = discover_features()
providers = discovery.discover_embedding_providers()
learning_path = discovery.get_learning_path("intermediate")
        """,
    }


def validate_configuration(config: SystemConfiguration) -> List[str]:
    """Validate system configuration and return any issues.
    
    Args:
        config: System configuration to validate
        
    Returns:
        List of validation warnings/errors
    """
    issues = []
    
    # Check embedding provider
    valid_providers = ["fastembed", "openai", "huggingface"]
    if config.embedding_provider not in valid_providers:
        issues.append(f"Unknown embedding provider: {config.embedding_provider}")
    
    # Check workspace directory
    if config.workspace_dir:
        workspace_path = Path(config.workspace_dir)
        if not workspace_path.exists():
            issues.append(f"Workspace directory does not exist: {config.workspace_dir}")
        elif not workspace_path.is_dir():
            issues.append(f"Workspace path is not a directory: {config.workspace_dir}")
    
    # Check cache configuration
    if config.cache_options.enabled:
        if config.cache_options.ttl_seconds < 60:
            issues.append("Cache TTL should be at least 60 seconds")
        if config.cache_options.max_size < 10:
            issues.append("Cache max size should be at least 10")
    
    # Check monitoring configuration
    if config.monitoring_options.track_costs and not config.monitoring_options.enabled:
        issues.append("Cost tracking requires monitoring to be enabled")
    
    if config.monitoring_options.budget_limit and config.monitoring_options.budget_limit <= 0:
        issues.append("Budget limit must be positive")
    
    return issues


__all__ = [
    "FeatureDiscovery",
    "ProviderInfo", 
    "FeatureLevel",
    "create_system",
    "create_embedding_service",
    "create_search_service",
    "discover_features",
    "get_quick_examples",
    "validate_configuration",
]