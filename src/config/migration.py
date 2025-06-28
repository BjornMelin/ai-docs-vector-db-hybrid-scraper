"""Configuration migration utilities.

Provides utilities to migrate from the old 18-file configuration system
to the new modern Pydantic Settings system while maintaining backward compatibility.
"""

import logging
from typing import Any, Dict, Optional

from .modern import (
    ApplicationMode,
    Config as ModernConfig,
    CrawlProvider,
    EmbeddingProvider,
    Environment,
)


logger = logging.getLogger(__name__)


class ConfigMigrator:
    """Handles migration from legacy configuration to modern configuration."""

    def __init__(self):
        self._mapping_cache: Dict[str, Any] | None = None

    def migrate_from_legacy(self, legacy_config: Any) -> ModernConfig:
        """Migrate from legacy configuration to modern configuration.

        Args:
            legacy_config: Legacy configuration object from the old system.

        Returns:
            Modern configuration instance with migrated settings.
        """
        try:
            # Extract settings from legacy config
            legacy_data = self._extract_legacy_data(legacy_config)

            # Apply migrations and transformations
            modern_data = self._transform_legacy_data(legacy_data)

            # Create modern config instance
            return ModernConfig(**modern_data)

        except Exception as e:
            logger.warning(f"Migration failed, using defaults: {e}")
            return ModernConfig()

    def _extract_legacy_data(self, legacy_config: Any) -> Dict[str, Any]:
        """Extract data from legacy configuration object.

        Args:
            legacy_config: Legacy configuration object.

        Returns:
            Dictionary of extracted configuration data.
        """
        data = {}

        # Handle different legacy config types
        if hasattr(legacy_config, "model_dump"):
            # Pydantic model
            data = legacy_config.model_dump()
        elif hasattr(legacy_config, "dict"):
            # Pydantic v1 model
            data = legacy_config.dict()
        elif hasattr(legacy_config, "__dict__"):
            # Regular class instance
            data = legacy_config.__dict__.copy()
        elif isinstance(legacy_config, dict):
            # Dictionary
            data = legacy_config.copy()

        return data

    def _transform_legacy_data(self, legacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform legacy data to modern configuration format.

        Args:
            legacy_data: Dictionary of legacy configuration data.

        Returns:
            Dictionary of transformed data for modern configuration.
        """
        modern_data = {}

        # Map legacy fields to modern equivalents
        field_mapping = self._get_field_mapping()

        for legacy_field, modern_field in field_mapping.items():
            if legacy_field in legacy_data:
                value = legacy_data[legacy_field]
                if modern_field and value is not None:
                    self._set_nested_value(modern_data, modern_field, value)

        # Handle special transformations
        self._apply_special_transformations(legacy_data, modern_data)

        # Set reasonable defaults
        self._apply_migration_defaults(modern_data)

        return modern_data

    def _get_field_mapping(self) -> Dict[str, str]:
        """Get mapping from legacy field names to modern field names.

        Returns:
            Dictionary mapping legacy fields to modern fields.
        """
        if self._mapping_cache is None:
            self._mapping_cache = {
                # Direct mappings
                "debug": "debug",
                "log_level": "log_level",
                "environment": "environment",
                # Provider mappings
                "embedding_provider": "embedding_provider",
                "crawl_provider": "crawl_provider",
                # Service URLs
                "qdrant_url": "qdrant_url",
                "redis_url": "redis_url",
                # API Keys
                "openai_api_key": "openai_api_key",
                "firecrawl_api_key": "firecrawl_api_key",
                "qdrant_api_key": "qdrant_api_key",
                # Nested configuration mappings
                "qdrant.url": "qdrant.url",
                "qdrant.api_key": "qdrant.api_key",
                "qdrant.collection_name": "qdrant.default_collection",
                "qdrant.timeout": "qdrant.timeout",
                "qdrant.grpc_port": "qdrant.grpc_port",
                "qdrant.prefer_grpc": "qdrant.use_grpc",
                "openai.api_key": "openai.api_key",
                "openai.model": "openai.embedding_model",
                "openai.dimensions": "openai.dimensions",
                "firecrawl.api_key": "firecrawl.api_key",
                "firecrawl.api_url": "firecrawl.api_base",
                "firecrawl.timeout": "firecrawl.timeout",
                "cache.enable_caching": "cache.enable_caching",
                "cache.enable_local_cache": "cache.enable_local_cache",
                "cache.enable_dragonfly_cache": "cache.enable_redis_cache",
                "cache.dragonfly_url": "cache.redis_url",
                "cache.local_max_size": "cache.local_max_size",
                "cache.local_max_memory_mb": "cache.local_max_memory_mb",
                "cache.ttl_seconds": "cache.ttl_queries",
                "cache.cache_ttl_seconds.embeddings": "cache.ttl_embeddings",
                "cache.cache_ttl_seconds.crawl": "cache.ttl_crawl",
                "performance.max_concurrent_crawls": "performance.max_concurrent_crawls",
                "performance.max_concurrent_embeddings": "performance.max_concurrent_embeddings",
                "performance.request_timeout": "performance.request_timeout",
                "performance.max_memory_usage_mb": "performance.max_memory_usage_mb",
                "chunking.strategy": "chunking.strategy",
                "chunking.max_chunk_size": "chunking.max_chunk_size",
                "chunking.min_chunk_size": "chunking.min_chunk_size",
                "chunking.overlap": "chunking.overlap",
                "security.max_query_length": "security.max_query_length",
                "security.max_url_length": "security.max_url_length",
                "security.rate_limit_requests_per_minute": "security.rate_limit_requests_per_minute",
                "security.allowed_domains": "security.allowed_domains",
                "security.require_api_keys": "security.require_api_keys",
                "security.enable_rate_limiting": "security.enable_rate_limiting",
            }

        return self._mapping_cache

    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in a dictionary using dot notation.

        Args:
            data: Dictionary to set value in.
            path: Dot-separated path to the value.
            value: Value to set.
        """
        keys = path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _apply_special_transformations(
        self, legacy_data: Dict[str, Any], modern_data: Dict[str, Any]
    ) -> None:
        """Apply special transformations that require custom logic.

        Args:
            legacy_data: Legacy configuration data.
            modern_data: Modern configuration data to modify.
        """
        # Determine application mode based on legacy settings
        if "deployment_tier" in legacy_data:
            tier = legacy_data.get("deployment_tier", "").lower()
            if tier in ["enterprise", "production", "premium"]:
                modern_data["mode"] = ApplicationMode.ENTERPRISE.value
            else:
                modern_data["mode"] = ApplicationMode.SIMPLE.value

        # Map legacy environment values
        if "environment" in legacy_data:
            env = legacy_data["environment"]
            if isinstance(env, str):
                try:
                    modern_data["environment"] = Environment(env.lower()).value
                except ValueError:
                    modern_data["environment"] = Environment.DEVELOPMENT.value

        # Map legacy provider values
        if "embedding_provider" in legacy_data:
            provider = legacy_data["embedding_provider"]
            if isinstance(provider, str):
                try:
                    modern_data["embedding_provider"] = EmbeddingProvider(
                        provider.lower()
                    ).value
                except ValueError:
                    modern_data["embedding_provider"] = (
                        EmbeddingProvider.FASTEMBED.value
                    )

        if "crawl_provider" in legacy_data:
            provider = legacy_data["crawl_provider"]
            if isinstance(provider, str):
                try:
                    modern_data["crawl_provider"] = CrawlProvider(
                        provider.lower()
                    ).value
                except ValueError:
                    modern_data["crawl_provider"] = CrawlProvider.CRAWL4AI.value

        # Handle legacy HyDE configuration
        if "hyde" in legacy_data or any(k.startswith("hyde_") for k in legacy_data):
            hyde_config = {}
            for key, value in legacy_data.items():
                if key.startswith("hyde_"):
                    hyde_key = key[5:]  # Remove "hyde_" prefix
                    hyde_config[hyde_key] = value
                elif key == "hyde" and isinstance(value, dict):
                    hyde_config.update(value)

            if hyde_config:
                modern_data["hyde"] = hyde_config

        # Handle legacy reranking configuration
        if "reranking" in legacy_data or any(
            k.startswith("rerank_") for k in legacy_data
        ):
            rerank_config = {}
            for key, value in legacy_data.items():
                if key.startswith("rerank_"):
                    rerank_key = key[7:]  # Remove "rerank_" prefix
                    rerank_config[rerank_key] = value
                elif key == "reranking" and isinstance(value, dict):
                    rerank_config.update(value)

            if rerank_config:
                modern_data["reranking"] = rerank_config

    def _apply_migration_defaults(self, modern_data: Dict[str, Any]) -> None:
        """Apply default values for migration.

        Args:
            modern_data: Modern configuration data to modify.
        """
        # Set default mode if not specified
        if "mode" not in modern_data:
            modern_data["mode"] = ApplicationMode.SIMPLE.value

        # Set default environment if not specified
        if "environment" not in modern_data:
            modern_data["environment"] = Environment.DEVELOPMENT.value

        # Set default providers if not specified
        if "embedding_provider" not in modern_data:
            modern_data["embedding_provider"] = EmbeddingProvider.FASTEMBED.value

        if "crawl_provider" not in modern_data:
            modern_data["crawl_provider"] = CrawlProvider.CRAWL4AI.value


def migrate_legacy_config(legacy_config: Any) -> ModernConfig:
    """Migrate from legacy configuration to modern configuration.

    Args:
        legacy_config: Legacy configuration object.

    Returns:
        Modern configuration instance.
    """
    migrator = ConfigMigrator()
    return migrator.migrate_from_legacy(legacy_config)


def create_migration_compatibility_wrapper(
    modern_config: ModernConfig,
) -> Dict[str, Any]:
    """Create a compatibility wrapper for legacy code.

    Args:
        modern_config: Modern configuration instance.

    Returns:
        Dictionary that provides backward compatibility for legacy code.
    """
    # Extract all configuration data
    config_data = modern_config.model_dump()

    # Create compatibility wrapper with legacy field names
    wrapper = {}

    # Direct field mappings
    wrapper.update(
        {
            "debug": config_data.get("debug"),
            "log_level": config_data.get("log_level"),
            "environment": config_data.get("environment"),
            "embedding_provider": config_data.get("embedding_provider"),
            "crawl_provider": config_data.get("crawl_provider"),
            "qdrant_url": config_data.get("qdrant_url"),
            "redis_url": config_data.get("redis_url"),
            "openai_api_key": config_data.get("openai_api_key"),
            "firecrawl_api_key": config_data.get("firecrawl_api_key"),
            "qdrant_api_key": config_data.get("qdrant_api_key"),
        }
    )

    # Nested configurations
    for section in [
        "qdrant",
        "openai",
        "firecrawl",
        "cache",
        "performance",
        "security",
        "chunking",
        "hyde",
        "reranking",
    ]:
        if section in config_data:
            wrapper[section] = config_data[section]

    return wrapper


# Global migrator instance
_migrator = ConfigMigrator()


__all__ = [
    "ConfigMigrator",
    "create_migration_compatibility_wrapper",
    "migrate_legacy_config",
]
