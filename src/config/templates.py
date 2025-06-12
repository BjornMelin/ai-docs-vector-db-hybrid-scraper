"""Configuration templates for different use cases and environments.

This module provides predefined configuration templates that can be used to quickly
set up the system for different deployment scenarios and use cases.

SECURITY WARNING: All templates contain placeholder values for database passwords
and URLs. These MUST be replaced with secure values before production use.
Never use default credentials or expose real API keys in templates.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from .utils import ConfigPathManager
from .utils import ConfigVersioning
from .utils import generate_timestamp


class TemplateMetadata(BaseModel):
    """Metadata for configuration templates."""

    name: str = Field(description="Template name")
    description: str = Field(description="Template description")
    use_case: str = Field(description="Primary use case")
    environment: str = Field(description="Target environment")
    version: str = Field(default="1.0.0", description="Template version")
    created_at: str = Field(
        default_factory=generate_timestamp, description="Creation timestamp"
    )
    tags: list[str] = Field(default_factory=list, description="Template tags")


class ConfigurationTemplate(BaseModel):
    """Complete configuration template with metadata and configuration data."""

    metadata: TemplateMetadata
    configuration: dict[str, Any]
    overrides: dict[str, dict[str, Any]] | None = Field(
        default=None, description="Environment-specific overrides"
    )


class ConfigurationTemplates:
    """Manager for configuration templates with predefined templates for common use cases."""

    def __init__(self, templates_dir: Path | None = None):
        """Initialize template manager.

        Args:
            templates_dir: Directory for storing custom templates
        """
        self.path_manager = ConfigPathManager()
        if templates_dir:
            self.path_manager.templates_dir = templates_dir
        self.path_manager.ensure_directories()

    @staticmethod
    def development_template() -> ConfigurationTemplate:
        """Development configuration template with debugging and fast iteration settings."""
        return ConfigurationTemplate(
            metadata=TemplateMetadata(
                name="development",
                description="Development environment with debugging enabled and fast iteration",
                use_case="Local development and testing",
                environment="development",
                tags=["development", "debug", "local"],
            ),
            configuration={
                "environment": "development",
                "debug": True,
                "log_level": "DEBUG",
                # Embedding provider - use local FastEmbed for development
                "embedding_provider": "fastembed",
                "crawl_provider": "crawl4ai",
                # Database settings optimized for development
                "database": {
                    "database_url": "sqlite+aiosqlite:///data/dev.db",
                    "echo_queries": True,
                    "pool_size": 5,
                    "max_overflow": 5,
                    "pool_timeout": 10.0,
                    "enable_query_monitoring": True,
                    "slow_query_threshold_ms": 50.0,
                },
                # Cache settings for fast development iteration
                "cache": {
                    "enable_caching": True,
                    "enable_local_cache": True,
                    "enable_dragonfly_cache": False,  # Disable Redis for simplicity
                    "local_max_size": 500,
                    "local_max_memory_mb": 50.0,
                    "cache_ttl_seconds": {
                        "embeddings": 300,  # 5 minutes for fast testing
                        "crawl": 300,
                        "search": 300,
                        "hyde": 300,
                    },
                },
                # Qdrant settings for development
                "qdrant": {
                    "url": "http://localhost:6333",
                    "timeout": 10.0,
                    "batch_size": 50,
                    "max_retries": 2,
                },
                # Performance settings for development
                "performance": {
                    "max_concurrent_requests": 5,
                    "request_timeout": 10.0,
                    "max_retries": 2,
                    "max_memory_usage_mb": 500.0,
                },
                # Security settings for development
                "security": {
                    "require_api_keys": False,  # Relaxed for development
                    "enable_rate_limiting": False,
                    "allowed_domains": [],
                    "blocked_domains": [],
                },
                # Monitoring settings
                "monitoring": {
                    "enabled": True,
                    "include_system_metrics": True,
                    "system_metrics_interval": 60.0,
                },
            },
        )

    @staticmethod
    def production_template() -> ConfigurationTemplate:
        """Production configuration template with security and performance optimizations."""
        return ConfigurationTemplate(
            metadata=TemplateMetadata(
                name="production",
                description="Production environment with security hardening and performance optimization",
                use_case="Production deployment with high security and performance",
                environment="production",
                tags=["production", "security", "performance", "scalable"],
            ),
            configuration={
                "environment": "production",
                "debug": False,
                "log_level": "INFO",
                # Embedding provider - use OpenAI for production quality
                "embedding_provider": "openai",
                "crawl_provider": "crawl4ai",
                # Database settings optimized for production
                "database": {
                    "database_url": "postgresql+asyncpg://user:CHANGEME_DB_PASSWORD@localhost:5432/aidocs_prod",
                    "echo_queries": False,
                    "pool_size": 20,
                    "max_overflow": 10,
                    "pool_timeout": 30.0,
                    "pool_recycle": 3600,
                    "pool_pre_ping": True,
                    "adaptive_pool_sizing": True,
                    "min_pool_size": 10,
                    "max_pool_size": 50,
                    "enable_query_monitoring": True,
                    "slow_query_threshold_ms": 100.0,
                },
                # Cache settings for production
                "cache": {
                    "enable_caching": True,
                    "enable_local_cache": True,
                    "enable_dragonfly_cache": True,
                    "dragonfly_url": "redis://dragonfly:6379",
                    "local_max_size": 2000,
                    "local_max_memory_mb": 200.0,
                    "cache_ttl_seconds": {
                        "embeddings": 86400,  # 24 hours
                        "crawl": 3600,  # 1 hour
                        "search": 7200,  # 2 hours
                        "hyde": 3600,  # 1 hour
                    },
                },
                # Qdrant settings for production
                "qdrant": {
                    "url": "http://qdrant:6333",
                    "timeout": 30.0,
                    "batch_size": 100,
                    "max_retries": 3,
                    "max_connections": 50,
                    "connection_pool_size": 20,
                    "quantization_enabled": True,
                    "enable_hnsw_optimization": True,
                },
                # Performance settings for production
                "performance": {
                    "max_concurrent_requests": 100,
                    "request_timeout": 30.0,
                    "max_retries": 3,
                    "max_memory_usage_mb": 2000.0,
                    "canary_deployment_enabled": True,
                    "enable_dragonfly_compression": True,
                },
                # Security settings for production
                "security": {
                    "require_api_keys": True,
                    "enable_rate_limiting": True,
                    "rate_limit_requests": 100,
                    "allowed_domains": [],
                    "blocked_domains": [],
                },
                # Monitoring settings for production
                "monitoring": {
                    "enabled": True,
                    "include_system_metrics": True,
                    "system_metrics_interval": 30.0,
                    "enable_performance_monitoring": True,
                    "enable_cost_tracking": True,
                    "cpu_threshold": 80.0,
                    "memory_threshold": 85.0,
                    "disk_threshold": 90.0,
                },
            },
        )

    @staticmethod
    def high_performance_template() -> ConfigurationTemplate:
        """High-performance configuration template for maximum throughput."""
        return ConfigurationTemplate(
            metadata=TemplateMetadata(
                name="high_performance",
                description="High-performance configuration optimized for maximum throughput",
                use_case="High-traffic applications requiring maximum performance",
                environment="production",
                tags=["performance", "throughput", "optimization", "scalable"],
            ),
            configuration={
                "environment": "production",
                "debug": False,
                "log_level": "WARNING",  # Minimal logging for performance
                # High-performance embedding settings
                "embedding_provider": "fastembed",  # Local for speed
                "crawl_provider": "crawl4ai",
                # Database optimized for high performance
                "database": {
                    "database_url": "postgresql+asyncpg://user:CHANGEME_DB_PASSWORD@localhost:5432/aidocs_perf",
                    "echo_queries": False,
                    "pool_size": 50,
                    "max_overflow": 20,
                    "pool_timeout": 5.0,  # Aggressive timeout
                    "pool_recycle": 1800,  # Faster recycling
                    "adaptive_pool_sizing": True,
                    "min_pool_size": 20,
                    "max_pool_size": 100,
                    "pool_growth_factor": 2.0,
                },
                # Aggressive caching for performance
                "cache": {
                    "enable_caching": True,
                    "enable_local_cache": True,
                    "enable_dragonfly_cache": True,
                    "dragonfly_url": "redis://dragonfly:6379",
                    "local_max_size": 5000,
                    "local_max_memory_mb": 500.0,
                    "cache_ttl_seconds": {
                        "embeddings": 172800,  # 48 hours
                        "crawl": 7200,  # 2 hours
                        "search": 14400,  # 4 hours
                        "hyde": 7200,  # 2 hours
                    },
                },
                # Qdrant optimized for performance
                "qdrant": {
                    "url": "http://qdrant:6333",
                    "timeout": 10.0,  # Aggressive timeout
                    "batch_size": 200,  # Larger batches
                    "max_retries": 2,  # Fewer retries for speed
                    "max_connections": 100,
                    "connection_pool_size": 50,
                    "quantization_enabled": True,
                },
                # Maximum performance settings
                "performance": {
                    "max_concurrent_requests": 200,
                    "request_timeout": 15.0,
                    "max_retries": 2,
                    "max_memory_usage_mb": 4000.0,
                    "dragonfly_pipeline_size": 200,
                    "dragonfly_scan_count": 2000,
                    "enable_dragonfly_compression": True,
                },
                # Monitoring focused on performance metrics
                "monitoring": {
                    "enabled": True,
                    "include_system_metrics": True,
                    "system_metrics_interval": 15.0,  # Frequent monitoring
                    "enable_performance_monitoring": True,
                },
            },
        )

    @staticmethod
    def memory_optimized_template() -> ConfigurationTemplate:
        """Memory-optimized configuration template for resource-constrained environments."""
        return ConfigurationTemplate(
            metadata=TemplateMetadata(
                name="memory_optimized",
                description="Memory-optimized configuration for resource-constrained environments",
                use_case="Deployment in memory-limited environments like containers or edge devices",
                environment="production",
                tags=["memory", "optimization", "resource-constrained", "efficient"],
            ),
            configuration={
                "environment": "production",
                "debug": False,
                "log_level": "WARNING",
                # Memory-efficient embedding settings
                "embedding_provider": "fastembed",
                "crawl_provider": "crawl4ai",
                # Database with conservative memory usage
                "database": {
                    "database_url": "sqlite+aiosqlite:///data/aidocs_optimized.db",
                    "echo_queries": False,
                    "pool_size": 3,  # Minimal pool
                    "max_overflow": 2,  # Minimal overflow
                    "pool_timeout": 30.0,
                    "adaptive_pool_sizing": False,  # Disable dynamic scaling
                    "enable_query_monitoring": False,  # Reduce overhead
                },
                # Conservative caching to save memory
                "cache": {
                    "enable_caching": True,
                    "enable_local_cache": True,
                    "enable_dragonfly_cache": False,  # Skip Redis to save memory
                    "local_max_size": 200,
                    "local_max_memory_mb": 25.0,
                    "cache_ttl_seconds": {
                        "embeddings": 3600,  # Shorter TTL to save memory
                        "crawl": 1800,
                        "search": 1800,
                        "hyde": 1800,
                    },
                },
                # Qdrant with memory optimization
                "qdrant": {
                    "url": "http://localhost:6333",
                    "timeout": 30.0,
                    "batch_size": 50,  # Smaller batches
                    "max_retries": 3,
                    "max_connections": 10,  # Fewer connections
                    "connection_pool_size": 5,
                    "quantization_enabled": True,  # Save memory with quantization
                },
                # Conservative performance settings
                "performance": {
                    "max_concurrent_requests": 10,
                    "request_timeout": 30.0,
                    "max_retries": 3,
                    "max_memory_usage_mb": 256.0,  # Low memory limit
                    "gc_threshold": 0.7,  # Earlier garbage collection
                },
                # Minimal monitoring to save resources
                "monitoring": {
                    "enabled": True,
                    "include_system_metrics": False,  # Skip system metrics
                    "enable_performance_monitoring": False,
                    "enable_cost_tracking": False,
                },
            },
        )

    @staticmethod
    def distributed_template() -> ConfigurationTemplate:
        """Distributed configuration template for multi-node deployments."""
        return ConfigurationTemplate(
            metadata=TemplateMetadata(
                name="distributed",
                description="Distributed configuration for multi-node cluster deployments",
                use_case="Large-scale distributed deployments with multiple nodes",
                environment="production",
                tags=["distributed", "cluster", "scalable", "multi-node"],
            ),
            configuration={
                "environment": "production",
                "debug": False,
                "log_level": "INFO",
                # Provider settings for distributed
                "embedding_provider": "openai",  # Centralized service
                "crawl_provider": "crawl4ai",
                # Database for distributed deployment
                "database": {
                    "database_url": "postgresql+asyncpg://user:CHANGEME_DB_PASSWORD@db-cluster:5432/aidocs_distributed",
                    "echo_queries": False,
                    "pool_size": 30,
                    "max_overflow": 15,
                    "pool_timeout": 30.0,
                    "pool_recycle": 3600,
                    "pool_pre_ping": True,
                    "adaptive_pool_sizing": True,
                    "min_pool_size": 15,
                    "max_pool_size": 75,
                },
                # Distributed caching with Redis cluster
                "cache": {
                    "enable_caching": True,
                    "enable_local_cache": True,
                    "enable_dragonfly_cache": True,
                    "dragonfly_url": "redis://redis-cluster:6379",
                    "local_max_size": 1000,
                    "local_max_memory_mb": 100.0,
                    "redis_pool_size": 20,
                },
                # Qdrant cluster configuration
                "qdrant": {
                    "url": "http://qdrant-cluster:6333",
                    "timeout": 30.0,
                    "batch_size": 100,
                    "max_retries": 3,
                    "max_connections": 75,
                    "connection_pool_size": 30,
                    "quantization_enabled": True,
                },
                # Performance settings for distributed load
                "performance": {
                    "max_concurrent_requests": 150,
                    "request_timeout": 30.0,
                    "max_retries": 3,
                    "max_memory_usage_mb": 1500.0,
                    "canary_deployment_enabled": True,
                    "canary_health_check_interval": 15,
                    "enable_dragonfly_compression": True,
                },
                # Security for distributed deployment
                "security": {
                    "require_api_keys": True,
                    "enable_rate_limiting": True,
                    "rate_limit_requests": 1000,  # Higher for distributed load
                    "allowed_domains": [],
                    "blocked_domains": [],
                },
                # Enhanced monitoring for distributed systems
                "monitoring": {
                    "enabled": True,
                    "include_system_metrics": True,
                    "system_metrics_interval": 30.0,
                    "enable_performance_monitoring": True,
                    "enable_cost_tracking": True,
                    "cpu_threshold": 75.0,
                    "memory_threshold": 80.0,
                    "disk_threshold": 85.0,
                },
                # Task queue for distributed processing
                "task_queue": {
                    "redis_url": "redis://redis-cluster:6379",
                    "max_jobs": 20,
                    "job_timeout": 7200,  # Longer timeout for distributed
                    "worker_pool_size": 8,
                },
            },
        )

    def get_template(self, template_name: str) -> ConfigurationTemplate | None:
        """Get a predefined template by name.

        Args:
            template_name: Name of the template to retrieve

        Returns:
            ConfigurationTemplate or None if not found
        """
        templates = {
            "development": self.development_template,
            "production": self.production_template,
            "high_performance": self.high_performance_template,
            "memory_optimized": self.memory_optimized_template,
            "distributed": self.distributed_template,
        }

        if template_name in templates:
            return templates[template_name]()

        # Try loading from custom template file
        return self.load_template(template_name)

    def list_available_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            List of template names
        """
        predefined = [
            "development",
            "production",
            "high_performance",
            "memory_optimized",
            "distributed",
        ]
        custom = [f.stem for f in self.path_manager.templates_dir.glob("*.json")]
        return sorted(set(predefined + custom))

    def save_template(
        self, template: ConfigurationTemplate, template_name: str
    ) -> None:
        """Save a custom template to disk.

        Args:
            template: Template to save
            template_name: Name for the template file
        """
        template_path = self.path_manager.get_template_path(template_name)

        # Convert to JSON-serializable format
        template_data = template.model_dump()

        import json

        with open(template_path, "w") as f:
            json.dump(template_data, f, indent=2)

    def load_template(self, template_name: str) -> ConfigurationTemplate | None:
        """Load a custom template from disk.

        Args:
            template_name: Name of the template to load

        Returns:
            ConfigurationTemplate or None if not found
        """
        template_path = self.path_manager.get_template_path(template_name)

        if not template_path.exists():
            return None

        import json

        try:
            with open(template_path) as f:
                data = json.load(f)
            return ConfigurationTemplate(**data)
        except Exception:
            return None

    def apply_template_to_config(
        self,
        template_name: str,
        base_config: dict[str, Any] | None = None,
        environment_overrides: str | None = None,
    ) -> dict[str, Any] | None:
        """Apply a template to create a configuration.

        Args:
            template_name: Name of template to apply
            base_config: Base configuration to merge with (optional)
            environment_overrides: Environment-specific overrides to apply

        Returns:
            Generated configuration dictionary or None if template not found
        """
        template = self.get_template(template_name)
        if not template:
            return None

        from .utils import ConfigMerger

        # Start with template configuration
        result_config = template.configuration.copy()

        # Apply environment overrides if specified
        if (
            environment_overrides
            and template.overrides
            and environment_overrides in template.overrides
        ):
            result_config = ConfigMerger.deep_merge(
                result_config, template.overrides[environment_overrides]
            )

        # Merge with base config if provided
        if base_config:
            result_config = ConfigMerger.deep_merge(base_config, result_config)

        # Add metadata
        result_config.update(
            {
                "template_source": template_name,
                "created_at": generate_timestamp(),
                "config_hash": ConfigVersioning.generate_config_hash(result_config),
            }
        )

        return result_config
