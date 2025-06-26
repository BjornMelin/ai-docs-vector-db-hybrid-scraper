"""Builder patterns for progressive API configuration.

This module provides sophisticated builder patterns that allow users
to progressively discover and configure advanced features while
maintaining simple defaults.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .protocols import (
    CacheProtocol,
    DocumentProcessorProtocol,
    EmbeddingProtocol,
    MonitoringProtocol,
    SearchProtocol,
)
from .types import (
    CacheOptions,
    ContentType,
    EmbeddingOptions,
    MonitoringOptions,
    ProcessingMode,
    ProcessingOptions,
    QualityTier,
    SearchOptions,
    SearchStrategy,
    SystemConfiguration,
)


class EmbeddingConfigBuilder:
    """Builder for embedding configuration with progressive disclosure.

    This builder allows users to start simple and progressively
    add more sophisticated embedding configuration.

    Examples:
        Simple:
            >>> config = EmbeddingConfigBuilder().build()

        Progressive:
            >>> config = (
            ...     EmbeddingConfigBuilder()
            ...     .with_provider("openai")
            ...     .with_quality_tier("best")
            ...     .build()
            ... )

        Advanced:
            >>> config = (
            ...     EmbeddingConfigBuilder()
            ...     .with_provider("openai")
            ...     .with_model("text-embedding-3-large")
            ...     .with_batch_size(64)
            ...     .with_custom_preprocessing({"clean_html": True})
            ...     .build()
            ... )
    """

    def __init__(self):
        """Initialize builder with sensible defaults."""
        self._config = EmbeddingOptions()

    def with_provider(self, provider: str) -> "EmbeddingConfigBuilder":
        """Set embedding provider.

        Args:
            provider: Provider name ("openai", "fastembed", "huggingface")

        Returns:
            Builder instance for chaining
        """
        self._config.provider = provider
        return self

    def with_model(self, model_name: str) -> "EmbeddingConfigBuilder":
        """Set specific model name.

        Args:
            model_name: Model identifier

        Returns:
            Builder instance for chaining
        """
        self._config.model_name = model_name
        return self

    def with_quality_tier(
        self, tier: Union[QualityTier, str]
    ) -> "EmbeddingConfigBuilder":
        """Set quality tier for automatic model selection.

        Args:
            tier: Quality tier ("fast", "balanced", "best")

        Returns:
            Builder instance for chaining
        """
        if isinstance(tier, str):
            tier = QualityTier(tier)
        self._config.quality_tier = tier
        return self

    def with_batch_size(self, batch_size: int) -> "EmbeddingConfigBuilder":
        """Set batch size for embedding generation.

        Args:
            batch_size: Number of texts to process in one batch

        Returns:
            Builder instance for chaining
        """
        self._config.batch_size = batch_size
        return self

    def with_normalization(self, normalize: bool = True) -> "EmbeddingConfigBuilder":
        """Enable or disable vector normalization.

        Args:
            normalize: Whether to normalize embeddings

        Returns:
            Builder instance for chaining
        """
        self._config.normalize = normalize
        return self

    def with_chunking_strategy(self, strategy: str) -> "EmbeddingConfigBuilder":
        """Set chunking strategy for long documents.

        Args:
            strategy: Chunking strategy name

        Returns:
            Builder instance for chaining
        """
        self._config.chunk_strategy = strategy
        return self

    def with_overlap_size(self, overlap: int) -> "EmbeddingConfigBuilder":
        """Set overlap size for document chunking.

        Args:
            overlap: Number of characters to overlap between chunks

        Returns:
            Builder instance for chaining
        """
        self._config.overlap_size = overlap
        return self

    def with_custom_preprocessing(
        self, preprocessing: Dict[str, Any]
    ) -> "EmbeddingConfigBuilder":
        """Set custom preprocessing options.

        Args:
            preprocessing: Custom preprocessing configuration

        Returns:
            Builder instance for chaining
        """
        self._config.custom_preprocessing = preprocessing
        return self

    def build(self) -> EmbeddingOptions:
        """Build the embedding configuration.

        Returns:
            Configured EmbeddingOptions instance
        """
        return self._config


class SearchConfigBuilder:
    """Builder for search configuration with progressive disclosure."""

    def __init__(self):
        """Initialize builder with sensible defaults."""
        self._config = SearchOptions()

    def with_strategy(
        self, strategy: Union[SearchStrategy, str]
    ) -> "SearchConfigBuilder":
        """Set search strategy.

        Args:
            strategy: Search strategy ("vector", "hybrid", "semantic", "adaptive")

        Returns:
            Builder instance for chaining
        """
        if isinstance(strategy, str):
            strategy = SearchStrategy(strategy)
        self._config.strategy = strategy
        return self

    def with_reranking(self, enabled: bool = True) -> "SearchConfigBuilder":
        """Enable or disable result reranking.

        Args:
            enabled: Whether to enable reranking

        Returns:
            Builder instance for chaining
        """
        self._config.rerank = enabled
        return self

    def with_quality_tier(self, tier: Union[QualityTier, str]) -> "SearchConfigBuilder":
        """Set quality tier for search.

        Args:
            tier: Quality tier ("fast", "balanced", "best")

        Returns:
            Builder instance for chaining
        """
        if isinstance(tier, str):
            tier = QualityTier(tier)
        self._config.quality_tier = tier
        return self

    def with_content_types(
        self, content_types: List[Union[ContentType, str]]
    ) -> "SearchConfigBuilder":
        """Filter by content types.

        Args:
            content_types: List of content types to include

        Returns:
            Builder instance for chaining
        """
        types = []
        for ct in content_types:
            if isinstance(ct, str):
                ct = ContentType(ct)
            types.append(ct)
        self._config.content_types = types
        return self

    def with_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> "SearchConfigBuilder":
        """Filter by date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Builder instance for chaining
        """
        self._config.date_range = (start_date, end_date)
        return self

    def with_metadata_filters(self, filters: Dict[str, Any]) -> "SearchConfigBuilder":
        """Add metadata filters.

        Args:
            filters: Metadata filters to apply

        Returns:
            Builder instance for chaining
        """
        self._config.metadata_filters = filters
        return self

    def with_embeddings(self, include: bool = True) -> "SearchConfigBuilder":
        """Include embeddings in results.

        Args:
            include: Whether to include embeddings

        Returns:
            Builder instance for chaining
        """
        self._config.include_embeddings = include
        return self

    def with_analysis(self, include: bool = True) -> "SearchConfigBuilder":
        """Include content analysis in results.

        Args:
            include: Whether to include analysis

        Returns:
            Builder instance for chaining
        """
        self._config.include_analysis = include
        return self

    def with_suggestions(self, include: bool = True) -> "SearchConfigBuilder":
        """Include search suggestions in results.

        Args:
            include: Whether to include suggestions

        Returns:
            Builder instance for chaining
        """
        self._config.include_suggestions = include
        return self

    def with_similarity_threshold(self, threshold: float) -> "SearchConfigBuilder":
        """Set minimum similarity threshold.

        Args:
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            Builder instance for chaining
        """
        self._config.similarity_threshold = threshold
        return self

    def with_diversity_factor(self, factor: float) -> "SearchConfigBuilder":
        """Set diversity factor for results.

        Args:
            factor: Diversity factor (0.0 to 1.0)

        Returns:
            Builder instance for chaining
        """
        self._config.diversity_factor = factor
        return self

    def with_custom_weights(self, weights: Dict[str, float]) -> "SearchConfigBuilder":
        """Set custom scoring weights.

        Args:
            weights: Custom weights for scoring components

        Returns:
            Builder instance for chaining
        """
        self._config.custom_weights = weights
        return self

    def with_fusion_algorithm(self, algorithm: str) -> "SearchConfigBuilder":
        """Set fusion algorithm for hybrid search.

        Args:
            algorithm: Fusion algorithm name

        Returns:
            Builder instance for chaining
        """
        self._config.fusion_algorithm = algorithm
        return self

    def with_rerank_model(self, model: str) -> "SearchConfigBuilder":
        """Set reranking model.

        Args:
            model: Reranking model identifier

        Returns:
            Builder instance for chaining
        """
        self._config.rerank_model = model
        return self

    def build(self) -> SearchOptions:
        """Build the search configuration.

        Returns:
            Configured SearchOptions instance
        """
        return self._config


class ProcessingConfigBuilder:
    """Builder for document processing configuration."""

    def __init__(self):
        """Initialize builder with sensible defaults."""
        self._config = ProcessingOptions()

    def with_mode(self, mode: Union[ProcessingMode, str]) -> "ProcessingConfigBuilder":
        """Set processing mode.

        Args:
            mode: Processing mode ("fast", "standard", "enhanced", "custom")

        Returns:
            Builder instance for chaining
        """
        if isinstance(mode, str):
            mode = ProcessingMode(mode)
        self._config.mode = mode
        return self

    def with_metadata_extraction(
        self, enabled: bool = True
    ) -> "ProcessingConfigBuilder":
        """Enable or disable metadata extraction.

        Args:
            enabled: Whether to extract metadata

        Returns:
            Builder instance for chaining
        """
        self._config.extract_metadata = enabled
        return self

    def with_language_detection(
        self, enabled: bool = True
    ) -> "ProcessingConfigBuilder":
        """Enable or disable language detection.

        Args:
            enabled: Whether to detect language

        Returns:
            Builder instance for chaining
        """
        self._config.detect_language = enabled
        return self

    def with_content_cleaning(self, enabled: bool = True) -> "ProcessingConfigBuilder":
        """Enable or disable content cleaning.

        Args:
            enabled: Whether to clean content

        Returns:
            Builder instance for chaining
        """
        self._config.clean_content = enabled
        return self

    def with_chunk_size(self, size: int) -> "ProcessingConfigBuilder":
        """Set chunk size for document splitting.

        Args:
            size: Maximum chunk size in characters

        Returns:
            Builder instance for chaining
        """
        self._config.chunk_size = size
        return self

    def with_chunk_overlap(self, overlap: int) -> "ProcessingConfigBuilder":
        """Set overlap between chunks.

        Args:
            overlap: Overlap size in characters

        Returns:
            Builder instance for chaining
        """
        self._config.chunk_overlap = overlap
        return self

    def with_entity_extraction(self, enabled: bool = True) -> "ProcessingConfigBuilder":
        """Enable or disable entity extraction.

        Args:
            enabled: Whether to extract entities

        Returns:
            Builder instance for chaining
        """
        self._config.extract_entities = enabled
        return self

    def with_summarization(self, enabled: bool = True) -> "ProcessingConfigBuilder":
        """Enable or disable content summarization.

        Args:
            enabled: Whether to generate summaries

        Returns:
            Builder instance for chaining
        """
        self._config.generate_summary = enabled
        return self

    def with_content_classification(
        self, enabled: bool = True
    ) -> "ProcessingConfigBuilder":
        """Enable or disable content classification.

        Args:
            enabled: Whether to classify content

        Returns:
            Builder instance for chaining
        """
        self._config.classify_content = enabled
        return self

    def with_quality_assessment(
        self, enabled: bool = True
    ) -> "ProcessingConfigBuilder":
        """Enable or disable quality assessment.

        Args:
            enabled: Whether to assess content quality

        Returns:
            Builder instance for chaining
        """
        self._config.quality_assessment = enabled
        return self

    def build(self) -> ProcessingOptions:
        """Build the processing configuration.

        Returns:
            Configured ProcessingOptions instance
        """
        return self._config


class CacheConfigBuilder:
    """Builder for cache configuration."""

    def __init__(self):
        """Initialize builder with sensible defaults."""
        self._config = CacheOptions()

    def with_caching(self, enabled: bool = True) -> "CacheConfigBuilder":
        """Enable or disable caching.

        Args:
            enabled: Whether to enable caching

        Returns:
            Builder instance for chaining
        """
        self._config.enabled = enabled
        return self

    def with_ttl(self, ttl_seconds: int) -> "CacheConfigBuilder":
        """Set cache TTL (time to live).

        Args:
            ttl_seconds: TTL in seconds

        Returns:
            Builder instance for chaining
        """
        self._config.ttl_seconds = ttl_seconds
        return self

    def with_max_size(self, max_size: int) -> "CacheConfigBuilder":
        """Set maximum cache size.

        Args:
            max_size: Maximum number of cached items

        Returns:
            Builder instance for chaining
        """
        self._config.max_size = max_size
        return self

    def with_embedding_cache(self, enabled: bool = True) -> "CacheConfigBuilder":
        """Enable or disable embedding caching.

        Args:
            enabled: Whether to cache embeddings

        Returns:
            Builder instance for chaining
        """
        self._config.cache_embeddings = enabled
        return self

    def with_search_cache(self, enabled: bool = True) -> "CacheConfigBuilder":
        """Enable or disable search result caching.

        Args:
            enabled: Whether to cache search results

        Returns:
            Builder instance for chaining
        """
        self._config.cache_search_results = enabled
        return self

    def with_analysis_cache(self, enabled: bool = False) -> "CacheConfigBuilder":
        """Enable or disable analysis caching.

        Args:
            enabled: Whether to cache analysis results

        Returns:
            Builder instance for chaining
        """
        self._config.cache_analysis = enabled
        return self

    def with_eviction_policy(self, policy: str) -> "CacheConfigBuilder":
        """Set cache eviction policy.

        Args:
            policy: Eviction policy ("lru", "lfu", "fifo")

        Returns:
            Builder instance for chaining
        """
        self._config.eviction_policy = policy
        return self

    def with_compression(self, enabled: bool = False) -> "CacheConfigBuilder":
        """Enable or disable cache compression.

        Args:
            enabled: Whether to compress cached data

        Returns:
            Builder instance for chaining
        """
        self._config.compression = enabled
        return self

    def with_distributed_cache(self, enabled: bool = False) -> "CacheConfigBuilder":
        """Enable or disable distributed caching.

        Args:
            enabled: Whether to use distributed cache

        Returns:
            Builder instance for chaining
        """
        self._config.distributed = enabled
        return self

    def build(self) -> CacheOptions:
        """Build the cache configuration.

        Returns:
            Configured CacheOptions instance
        """
        return self._config


class MonitoringConfigBuilder:
    """Builder for monitoring configuration."""

    def __init__(self):
        """Initialize builder with sensible defaults."""
        self._config = MonitoringOptions()

    def with_monitoring(self, enabled: bool = True) -> "MonitoringConfigBuilder":
        """Enable or disable monitoring.

        Args:
            enabled: Whether to enable monitoring

        Returns:
            Builder instance for chaining
        """
        self._config.enabled = enabled
        return self

    def with_metrics_collection(
        self, enabled: bool = True
    ) -> "MonitoringConfigBuilder":
        """Enable or disable metrics collection.

        Args:
            enabled: Whether to collect metrics

        Returns:
            Builder instance for chaining
        """
        self._config.collect_metrics = enabled
        return self

    def with_request_tracing(self, enabled: bool = False) -> "MonitoringConfigBuilder":
        """Enable or disable request tracing.

        Args:
            enabled: Whether to trace requests

        Returns:
            Builder instance for chaining
        """
        self._config.trace_requests = enabled
        return self

    def with_performance_tracking(
        self,
        latency: bool = True,
        throughput: bool = True,
        errors: bool = True,
    ) -> "MonitoringConfigBuilder":
        """Configure performance tracking.

        Args:
            latency: Track latency metrics
            throughput: Track throughput metrics
            errors: Track error metrics

        Returns:
            Builder instance for chaining
        """
        self._config.track_latency = latency
        self._config.track_throughput = throughput
        self._config.track_errors = errors
        return self

    def with_cost_tracking(
        self,
        enabled: bool = True,
        alerts: bool = False,
        budget_limit: float | None = None,
    ) -> "MonitoringConfigBuilder":
        """Configure cost tracking.

        Args:
            enabled: Enable cost tracking
            alerts: Enable budget alerts
            budget_limit: Budget limit in USD

        Returns:
            Builder instance for chaining
        """
        self._config.track_costs = enabled
        self._config.budget_alerts = alerts
        self._config.budget_limit = budget_limit
        return self

    def with_detailed_tracing(self, enabled: bool = False) -> "MonitoringConfigBuilder":
        """Enable or disable detailed tracing.

        Args:
            enabled: Whether to enable detailed tracing

        Returns:
            Builder instance for chaining
        """
        self._config.detailed_tracing = enabled
        return self

    def with_metrics_export(
        self,
        enabled: bool = False,
        endpoint: str | None = None,
    ) -> "MonitoringConfigBuilder":
        """Configure metrics export.

        Args:
            enabled: Enable metrics export
            endpoint: Metrics endpoint URL

        Returns:
            Builder instance for chaining
        """
        self._config.export_metrics = enabled
        self._config.metrics_endpoint = endpoint
        return self

    def build(self) -> MonitoringOptions:
        """Build the monitoring configuration.

        Returns:
            Configured MonitoringOptions instance
        """
        return self._config


class AdvancedConfigBuilder:
    """Builder for complete system configuration with expert features."""

    def __init__(self):
        """Initialize builder with sensible defaults."""
        self._config = SystemConfiguration()
        self._embedding_builder = EmbeddingConfigBuilder()
        self._search_builder = SearchConfigBuilder()
        self._processing_builder = ProcessingConfigBuilder()
        self._cache_builder = CacheConfigBuilder()
        self._monitoring_builder = MonitoringConfigBuilder()

    def with_embedding_provider(self, provider: str) -> "AdvancedConfigBuilder":
        """Set embedding provider.

        Args:
            provider: Provider name

        Returns:
            Builder instance for chaining
        """
        self._config.embedding_provider = provider
        return self

    def with_quality_tier(
        self, tier: Union[QualityTier, str]
    ) -> "AdvancedConfigBuilder":
        """Set overall quality tier.

        Args:
            tier: Quality tier

        Returns:
            Builder instance for chaining
        """
        if isinstance(tier, str):
            tier = QualityTier(tier)
        self._config.quality_tier = tier
        return self

    def with_workspace(
        self, workspace_dir: Union[str, Path]
    ) -> "AdvancedConfigBuilder":
        """Set workspace directory.

        Args:
            workspace_dir: Workspace directory path

        Returns:
            Builder instance for chaining
        """
        self._config.workspace_dir = str(workspace_dir)
        return self

    def with_embedding_config(
        self,
        config_builder: Callable[[EmbeddingConfigBuilder], EmbeddingConfigBuilder],
    ) -> "AdvancedConfigBuilder":
        """Configure embeddings using builder function.

        Args:
            config_builder: Function that configures the embedding builder

        Returns:
            Builder instance for chaining

        Example:
            >>> builder.with_embedding_config(
            ...     lambda b: b.with_provider("openai").with_quality_tier("best")
            ... )
        """
        self._embedding_builder = config_builder(self._embedding_builder)
        return self

    def with_search_config(
        self,
        config_builder: Callable[[SearchConfigBuilder], SearchConfigBuilder],
    ) -> "AdvancedConfigBuilder":
        """Configure search using builder function.

        Args:
            config_builder: Function that configures the search builder

        Returns:
            Builder instance for chaining
        """
        self._search_builder = config_builder(self._search_builder)
        return self

    def with_processing_config(
        self,
        config_builder: Callable[[ProcessingConfigBuilder], ProcessingConfigBuilder],
    ) -> "AdvancedConfigBuilder":
        """Configure processing using builder function.

        Args:
            config_builder: Function that configures the processing builder

        Returns:
            Builder instance for chaining
        """
        self._processing_builder = config_builder(self._processing_builder)
        return self

    def with_cache_config(
        self,
        config_builder: Callable[[CacheConfigBuilder], CacheConfigBuilder],
    ) -> "AdvancedConfigBuilder":
        """Configure caching using builder function.

        Args:
            config_builder: Function that configures the cache builder

        Returns:
            Builder instance for chaining
        """
        self._cache_builder = config_builder(self._cache_builder)
        return self

    def with_monitoring_config(
        self,
        config_builder: Callable[[MonitoringConfigBuilder], MonitoringConfigBuilder],
    ) -> "AdvancedConfigBuilder":
        """Configure monitoring using builder function.

        Args:
            config_builder: Function that configures the monitoring builder

        Returns:
            Builder instance for chaining
        """
        self._monitoring_builder = config_builder(self._monitoring_builder)
        return self

    def with_custom_providers(
        self, providers: Dict[str, Any]
    ) -> "AdvancedConfigBuilder":
        """Set custom provider implementations.

        Args:
            providers: Dictionary of custom providers

        Returns:
            Builder instance for chaining
        """
        self._config.custom_providers = providers
        return self

    def with_experimental_features(
        self, features: Dict[str, bool]
    ) -> "AdvancedConfigBuilder":
        """Enable experimental features.

        Args:
            features: Dictionary of experimental feature flags

        Returns:
            Builder instance for chaining
        """
        self._config.experimental_features = features
        return self

    def with_debug_mode(self, enabled: bool = True) -> "AdvancedConfigBuilder":
        """Enable or disable debug mode.

        Args:
            enabled: Whether to enable debug mode

        Returns:
            Builder instance for chaining
        """
        self._config.debug_mode = enabled
        return self

    def build(self) -> SystemConfiguration:
        """Build the complete system configuration.

        Returns:
            Configured SystemConfiguration instance
        """
        # Build all sub-configurations
        self._config.embedding_options = self._embedding_builder.build()
        self._config.search_options = self._search_builder.build()
        self._config.processing_options = self._processing_builder.build()
        self._config.cache_options = self._cache_builder.build()
        self._config.monitoring_options = self._monitoring_builder.build()

        return self._config


class AIDocSystemBuilder:
    """Main builder for AIDocSystem with progressive disclosure.

    This builder provides the entry point for progressive configuration
    of the AI Documentation System.

    Examples:
        Basic:
            >>> system = AIDocSystemBuilder().build()

        Progressive:
            >>> system = (
            ...     AIDocSystemBuilder()
            ...     .with_embedding_provider("openai")
            ...     .with_cache(enabled=True, ttl=3600)
            ...     .build()
            ... )

        Advanced:
            >>> system = (
            ...     AIDocSystemBuilder()
            ...     .with_advanced_config(
            ...         lambda c: c.with_monitoring_config(
            ...             lambda m: m.with_cost_tracking(
            ...                 enabled=True, budget_limit=100.0
            ...             )
            ...         )
            ...     )
            ...     .build()
            ... )
    """

    def __init__(self):
        """Initialize builder with sensible defaults."""
        self._config_builder = AdvancedConfigBuilder()

    def with_embedding_provider(self, provider: str) -> "AIDocSystemBuilder":
        """Set embedding provider.

        Args:
            provider: Provider name ("openai", "fastembed")

        Returns:
            Builder instance for chaining
        """
        self._config_builder.with_embedding_provider(provider)
        return self

    def with_quality_tier(self, tier: Union[QualityTier, str]) -> "AIDocSystemBuilder":
        """Set quality tier.

        Args:
            tier: Quality tier ("fast", "balanced", "best")

        Returns:
            Builder instance for chaining
        """
        self._config_builder.with_quality_tier(tier)
        return self

    def with_workspace(self, workspace_dir: Union[str, Path]) -> "AIDocSystemBuilder":
        """Set workspace directory.

        Args:
            workspace_dir: Workspace directory path

        Returns:
            Builder instance for chaining
        """
        self._config_builder.with_workspace(workspace_dir)
        return self

    def with_cache(
        self,
        enabled: bool = True,
        ttl: int = 3600,
        max_size: int = 1000,
    ) -> "AIDocSystemBuilder":
        """Configure caching (progressive feature).

        Args:
            enabled: Enable caching
            ttl: Cache TTL in seconds
            max_size: Maximum cache size

        Returns:
            Builder instance for chaining
        """
        self._config_builder.with_cache_config(
            lambda c: c.with_caching(enabled).with_ttl(ttl).with_max_size(max_size)
        )
        return self

    def with_monitoring(
        self,
        enabled: bool = True,
        track_costs: bool = False,
        budget_limit: float | None = None,
    ) -> "AIDocSystemBuilder":
        """Configure monitoring (progressive feature).

        Args:
            enabled: Enable monitoring
            track_costs: Track API costs
            budget_limit: Budget limit in USD

        Returns:
            Builder instance for chaining
        """
        self._config_builder.with_monitoring_config(
            lambda m: m.with_monitoring(enabled).with_cost_tracking(
                track_costs, alerts=budget_limit is not None, budget_limit=budget_limit
            )
        )
        return self

    def with_advanced_config(
        self,
        config_builder: Callable[[AdvancedConfigBuilder], AdvancedConfigBuilder],
    ) -> "AIDocSystemBuilder":
        """Configure advanced options (expert feature).

        Args:
            config_builder: Function that configures advanced options

        Returns:
            Builder instance for chaining

        Example:
            >>> builder.with_advanced_config(
            ...     lambda c: c.with_experimental_features({"new_feature": True})
            ... )
        """
        self._config_builder = config_builder(self._config_builder)
        return self

    def build(self) -> "AIDocSystem":
        """Build the configured AIDocSystem.

        Returns:
            Configured AIDocSystem instance
        """
        from .simple import AIDocSystem

        config = self._config_builder.build()

        return AIDocSystem(
            embedding_provider=config.embedding_provider,
            quality_tier=config.quality_tier.value,
            enable_cache=config.cache_options.enabled,
            enable_monitoring=config.monitoring_options.enabled,
            workspace_dir=config.workspace_dir,
        )
