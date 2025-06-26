"""Progressive configuration builders for elegant complexity demonstration.

This module provides persona-based configuration builders that showcase the sophistication
of our configuration system while maintaining simplicity for common use cases.

Features:
- Persona-based builders (Development, Production, Research, Enterprise)
- Progressive disclosure of advanced features
- Intelligent auto-detection and smart defaults
- Helpful validation with detailed error messages
- Configuration discovery and guided setup

Portfolio showcase elements:
- Advanced Pydantic v2 validation patterns
- Sophisticated auto-detection and service discovery
- Enterprise-grade configuration management
- Clean API design hiding complex validation logic
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError, validator

from .auto_detect import AutoDetectedServices, AutoDetectionConfig
from .core import (
    CacheConfig,
    ChunkingConfig,
    CircuitBreakerConfig,
    Config,
    Crawl4AIConfig,
    DeploymentConfig,
    EmbeddingConfig,
    FastEmbedConfig,
    FirecrawlConfig,
    HyDEConfig,
    MonitoringConfig,
    ObservabilityConfig,
    OpenAIConfig,
    PerformanceConfig,
    QdrantConfig,
    RAGConfig,
    SecurityConfig,
    SQLAlchemyConfig,
    TaskQueueConfig,
)
from .enums import (
    ChunkingStrategy,
    CrawlProvider,
    EmbeddingModel,
    EmbeddingProvider,
    Environment,
    LogLevel,
    SearchStrategy,
)


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Enhanced configuration validation error with helpful guidance."""

    def __init__(
        self, message: str, field_path: str = "", suggestions: List[str] = None
    ):
        self.field_path = field_path
        self.suggestions = suggestions or []
        super().__init__(message)

    def __str__(self) -> str:
        msg = super().__str__()
        if self.field_path:
            msg = f"{self.field_path}: {msg}"
        if self.suggestions:
            suggestions_text = "\n".join(f"  • {s}" for s in self.suggestions)
            msg += f"\n\nSuggestions:\n{suggestions_text}"
        return msg


class ConfigurationDiscovery(BaseModel):
    """Discovered configuration capabilities and recommendations."""

    persona: str = Field(description="Detected or selected persona")
    auto_detected_services: List[str] = Field(
        default_factory=list, description="Auto-detected services"
    )
    recommended_providers: Dict[str, str] = Field(
        default_factory=dict, description="Recommended providers"
    )
    available_features: List[str] = Field(
        default_factory=list, description="Available features"
    )
    configuration_complexity: str = Field(
        default="simple", description="Configuration complexity level"
    )
    estimated_setup_time: str = Field(
        default="< 5 minutes", description="Estimated setup time"
    )
    next_steps: List[str] = Field(
        default_factory=list, description="Recommended next steps"
    )


class BaseConfigBuilder(ABC):
    """Abstract base for configuration builders."""

    def __init__(self, auto_detect: bool = True):
        self.auto_detect = auto_detect
        self.auto_detected_services: AutoDetectedServices | None = None
        self._config_data: Dict[str, Any] = {}
        self._validation_errors: List[str] = []

    @abstractmethod
    def get_persona_name(self) -> str:
        """Get the persona name for this builder."""
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this persona."""
        pass

    @abstractmethod
    def get_recommended_features(self) -> List[str]:
        """Get recommended features for this persona."""
        pass

    async def discover_configuration(self) -> ConfigurationDiscovery:
        """Discover configuration capabilities and provide recommendations."""
        if self.auto_detect:
            # Perform auto-detection
            await self._perform_auto_detection()

        auto_detected_services = []
        if self.auto_detected_services:
            auto_detected_services = [
                service.service_type for service in self.auto_detected_services.services
            ]

        recommended_providers = self._get_recommended_providers()
        available_features = self.get_recommended_features()

        return ConfigurationDiscovery(
            persona=self.get_persona_name(),
            auto_detected_services=auto_detected_services,
            recommended_providers=recommended_providers,
            available_features=available_features,
            configuration_complexity=self._get_complexity_level(),
            estimated_setup_time=self._get_estimated_setup_time(),
            next_steps=self._get_next_steps(),
        )

    async def _perform_auto_detection(self) -> None:
        """Perform service auto-detection."""
        try:
            from ..services.auto_detection import EnvironmentDetector, ServiceDiscovery

            auto_config = AutoDetectionConfig()

            # Detect environment
            env_detector = EnvironmentDetector(auto_config)
            detected_env = await env_detector.detect()

            # Discover services
            service_discovery = ServiceDiscovery(auto_config)
            discovery_result = await service_discovery.discover_all_services()

            # Create auto-detected services container
            self.auto_detected_services = AutoDetectedServices(
                environment=detected_env,
                services=discovery_result.services,
                errors=discovery_result.errors,
            )
            self.auto_detected_services.mark_completed()

            logger.info(
                f"Auto-detected {len(discovery_result.services)} services "
                f"in {self.auto_detected_services.total_detection_time_ms:.1f}ms"
            )
        except Exception as e:
            logger.warning(f"Auto-detection failed: {e}")

    def _get_recommended_providers(self) -> Dict[str, str]:
        """Get recommended providers based on auto-detection and persona."""
        providers = {}

        # Base recommendations
        default_config = self.get_default_config()
        if "embedding_provider" in default_config:
            providers["embedding"] = default_config["embedding_provider"]
        if "crawl_provider" in default_config:
            providers["crawling"] = default_config["crawl_provider"]

        # Auto-detection overrides
        if self.auto_detected_services:
            if self.auto_detected_services.redis_service:
                providers["cache"] = "redis"
            if self.auto_detected_services.qdrant_service:
                providers["vector_db"] = "qdrant"
            if self.auto_detected_services.postgresql_service:
                providers["database"] = "postgresql"

        return providers

    def _get_complexity_level(self) -> str:
        """Determine configuration complexity level."""
        feature_count = len(self.get_recommended_features())
        if feature_count <= 3:
            return "simple"
        elif feature_count <= 7:
            return "moderate"
        else:
            return "advanced"

    def _get_estimated_setup_time(self) -> str:
        """Estimate setup time based on complexity."""
        complexity = self._get_complexity_level()
        if complexity == "simple":
            return "< 5 minutes"
        elif complexity == "moderate":
            return "10-15 minutes"
        else:
            return "20-30 minutes"

    def _get_next_steps(self) -> List[str]:
        """Get recommended next steps."""
        steps = []

        if self.auto_detected_services:
            if self.auto_detected_services.errors:
                steps.append(
                    "Review auto-detection errors and resolve connectivity issues"
                )

        steps.extend(
            [
                "Configure API keys for selected providers",
                "Review and adjust performance settings",
                "Set up monitoring and observability",
            ]
        )

        return steps

    def build(self, **overrides) -> Config:
        """Build configuration with optional overrides."""
        # Start with persona defaults
        config_data = self.get_default_config().copy()

        # Apply auto-detection if available
        if self.auto_detected_services:
            config_data = self._apply_auto_detection(config_data)

        # Apply user overrides
        config_data.update(overrides)

        # Validate and create config
        try:
            config = Config(**config_data)

            # Apply auto-detected services if available
            if self.auto_detected_services:
                config = config.apply_auto_detected_services(
                    self.auto_detected_services
                )

            return config
        except ValidationError as e:
            raise ConfigValidationError(
                "Configuration validation failed",
                suggestions=self._get_validation_suggestions(e),
            ) from e

    def _apply_auto_detection(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply auto-detected services to configuration."""
        if not self.auto_detected_services:
            return config_data

        # Apply Redis detection
        if self.auto_detected_services.redis_service:
            redis_service = self.auto_detected_services.redis_service
            if "cache" not in config_data:
                config_data["cache"] = {}
            config_data["cache"]["enable_dragonfly_cache"] = True
            config_data["cache"]["dragonfly_url"] = redis_service.connection_string

        # Apply Qdrant detection
        if self.auto_detected_services.qdrant_service:
            qdrant_service = self.auto_detected_services.qdrant_service
            if "qdrant" not in config_data:
                config_data["qdrant"] = {}
            config_data["qdrant"]["url"] = qdrant_service.connection_string

        return config_data

    def _get_validation_suggestions(self, error: ValidationError) -> List[str]:
        """Get helpful suggestions for validation errors."""
        suggestions = []

        for err in error.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            msg = err["msg"]

            if "api_key" in field.lower():
                suggestions.append(
                    "Set API keys via environment variables or configuration file"
                )
            elif "url" in field.lower():
                suggestions.append("Verify service URLs and connectivity")
            elif "required" in msg.lower():
                suggestions.append(
                    f"The field '{field}' is required for this configuration"
                )
            else:
                suggestions.append(f"Check '{field}': {msg}")

        return suggestions


class DevelopmentConfigBuilder(BaseConfigBuilder):
    """Configuration builder optimized for development workflows."""

    def get_persona_name(self) -> str:
        return "development"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "embedding_provider": "fastembed",
            "crawl_provider": "crawl4ai",
            "cache": {
                "enable_caching": True,
                "enable_local_cache": True,
                "enable_dragonfly_cache": False,
                "local_max_size": 500,
                "local_max_memory_mb": 50.0,
            },
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "dev_documents",
                "batch_size": 50,
            },
            "fastembed": {
                "model": "BAAI/bge-small-en-v1.5",
                "batch_size": 16,
            },
            "crawl4ai": {
                "headless": False,
                "max_concurrent_crawls": 5,
                "page_timeout": 60.0,
            },
            "chunking": {
                "strategy": "enhanced",
                "chunk_size": 1600,
                "chunk_overlap": 200,
            },
            "performance": {
                "max_concurrent_requests": 5,
                "request_timeout": 60.0,
                "max_memory_usage_mb": 500.0,
            },
            "security": {
                "require_api_keys": False,
                "enable_rate_limiting": False,
            },
            "monitoring": {
                "enabled": True,
                "enable_metrics": False,
            },
            "observability": {
                "enabled": False,
            },
        }

    def get_recommended_features(self) -> List[str]:
        return [
            "Local embeddings (FastEmbed)",
            "Browser automation (Crawl4AI)",
            "Local caching",
            "Debug logging",
            "Relaxed security",
        ]


class ProductionConfigBuilder(BaseConfigBuilder):
    """Configuration builder optimized for production deployments."""

    def get_persona_name(self) -> str:
        return "production"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "environment": "production",
            "debug": False,
            "log_level": "INFO",
            "embedding_provider": "openai",
            "crawl_provider": "crawl4ai",
            "cache": {
                "enable_caching": True,
                "enable_local_cache": True,
                "enable_dragonfly_cache": True,
                "local_max_size": 1000,
                "local_max_memory_mb": 100.0,
            },
            "qdrant": {
                "url": "http://qdrant:6333",
                "collection_name": "documents",
                "batch_size": 100,
                "prefer_grpc": True,
            },
            "openai": {
                "model": "text-embedding-3-small",
                "dimensions": 1536,
                "batch_size": 100,
                "max_requests_per_minute": 3000,
            },
            "crawl4ai": {
                "headless": True,
                "max_concurrent_crawls": 10,
                "page_timeout": 30.0,
            },
            "chunking": {
                "strategy": "enhanced",
                "chunk_size": 1600,
                "chunk_overlap": 200,
            },
            "performance": {
                "max_concurrent_requests": 20,
                "request_timeout": 30.0,
                "max_retries": 3,
                "max_memory_usage_mb": 2000.0,
            },
            "security": {
                "require_api_keys": True,
                "enable_rate_limiting": True,
                "rate_limit_requests": 100,
            },
            "monitoring": {
                "enabled": True,
                "enable_metrics": True,
                "include_system_metrics": True,
            },
            "observability": {
                "enabled": True,
                "track_ai_operations": True,
                "track_costs": True,
            },
            "circuit_breaker": {
                "use_enhanced_circuit_breaker": True,
                "enable_detailed_metrics": True,
                "enable_fallback_mechanisms": True,
            },
            "deployment": {
                "tier": "enterprise",
                "enable_feature_flags": True,
                "enable_deployment_services": True,
            },
        }

    def get_recommended_features(self) -> List[str]:
        return [
            "OpenAI embeddings",
            "Redis caching",
            "Circuit breakers",
            "Monitoring & observability",
            "Security & rate limiting",
            "Feature flags",
            "Performance optimization",
        ]


class ResearchConfigBuilder(BaseConfigBuilder):
    """Configuration builder optimized for research and experimentation."""

    def get_persona_name(self) -> str:
        return "research"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "embedding_provider": "openai",
            "crawl_provider": "firecrawl",
            "cache": {
                "enable_caching": True,
                "enable_local_cache": True,
                "cache_ttl_seconds": {
                    "search_results": 7200,  # Longer for research
                    "embeddings": 172800,  # 2 days
                    "collections": 14400,  # 4 hours
                },
            },
            "embedding": {
                "search_strategy": "hybrid",
                "enable_quantization": False,  # Full precision for research
            },
            "chunking": {
                "strategy": "enhanced",
                "chunk_size": 2000,  # Larger chunks for research
                "chunk_overlap": 400,
                "enable_ast_chunking": True,
                "preserve_code_blocks": True,
            },
            "hyde": {
                "enable_hyde": True,
                "num_generations": 5,
                "generation_temperature": 0.7,
            },
            "rag": {
                "enable_rag": True,
                "max_context_length": 8000,
                "max_results_for_context": 10,
                "include_confidence_score": True,
                "enable_answer_metrics": True,
            },
            "performance": {
                "max_concurrent_requests": 3,  # Conservative for research
                "request_timeout": 120.0,  # Longer timeouts
            },
            "monitoring": {
                "enabled": True,
                "enable_metrics": True,
            },
            "observability": {
                "enabled": True,
                "track_ai_operations": True,
                "track_costs": True,
                "console_exporter": True,  # For debugging
            },
        }

    def get_recommended_features(self) -> List[str]:
        return [
            "OpenAI embeddings",
            "Firecrawl web scraping",
            "HyDE query expansion",
            "RAG generation",
            "Hybrid search",
            "Extended caching",
            "Cost tracking",
            "Debug logging",
        ]


class EnterpriseConfigBuilder(BaseConfigBuilder):
    """Configuration builder for enterprise deployments with all features."""

    def get_persona_name(self) -> str:
        return "enterprise"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "environment": "production",
            "debug": False,
            "log_level": "INFO",
            "embedding_provider": "openai",
            "crawl_provider": "crawl4ai",
            "cache": {
                "enable_caching": True,
                "enable_local_cache": True,
                "enable_dragonfly_cache": True,
                "local_max_size": 2000,
                "local_max_memory_mb": 500.0,
            },
            "qdrant": {
                "url": "http://qdrant:6333",
                "batch_size": 200,
                "prefer_grpc": True,
            },
            "openai": {
                "model": "text-embedding-3-large",
                "dimensions": 3072,
                "batch_size": 200,
                "max_requests_per_minute": 5000,
            },
            "embedding": {
                "search_strategy": "hybrid",
                "enable_quantization": True,
            },
            "chunking": {
                "strategy": "enhanced",
                "chunk_size": 1600,
                "chunk_overlap": 320,
                "enable_ast_chunking": True,
                "preserve_code_blocks": True,
            },
            "hyde": {
                "enable_hyde": True,
                "num_generations": 3,
            },
            "rag": {
                "enable_rag": True,
                "max_context_length": 6000,
                "include_sources": True,
                "enable_answer_metrics": True,
                "enable_caching": True,
            },
            "performance": {
                "max_concurrent_requests": 50,
                "request_timeout": 30.0,
                "max_retries": 5,
                "max_memory_usage_mb": 4000.0,
            },
            "security": {
                "require_api_keys": True,
                "enable_rate_limiting": True,
                "rate_limit_requests": 1000,
                "x_frame_options": "SAMEORIGIN",
                "content_security_policy": "default-src 'self'; script-src 'self' 'unsafe-inline'",
            },
            "monitoring": {
                "enabled": True,
                "enable_metrics": True,
                "include_system_metrics": True,
                "system_metrics_interval": 15.0,
            },
            "observability": {
                "enabled": True,
                "track_ai_operations": True,
                "track_costs": True,
                "instrument_fastapi": True,
                "instrument_httpx": True,
                "instrument_redis": True,
                "instrument_sqlalchemy": True,
            },
            "circuit_breaker": {
                "use_enhanced_circuit_breaker": True,
                "enable_detailed_metrics": True,
                "enable_fallback_mechanisms": True,
                "enable_adaptive_timeout": True,
                "enable_bulkhead_isolation": True,
            },
            "deployment": {
                "tier": "enterprise",
                "enable_feature_flags": True,
                "enable_deployment_services": True,
                "enable_ab_testing": True,
                "enable_blue_green": True,
                "enable_canary": True,
            },
            "task_queue": {
                "max_jobs": 50,
                "job_timeout": 600,
            },
            "drift_detection": {
                "enabled": True,
                "snapshot_interval_minutes": 10,
                "alert_on_severity": ["medium", "high", "critical"],
                "enable_auto_remediation": True,
            },
        }

    def get_recommended_features(self) -> List[str]:
        return [
            "OpenAI large embeddings",
            "Hybrid search strategy",
            "HyDE & RAG generation",
            "Enterprise security",
            "Full observability stack",
            "Circuit breakers & resilience",
            "Feature flags & deployment",
            "Configuration drift detection",
            "Task queue management",
            "Auto-remediation",
        ]


class ConfigBuilderFactory:
    """Factory for creating configuration builders based on persona or use case."""

    _builders = {
        "development": DevelopmentConfigBuilder,
        "production": ProductionConfigBuilder,
        "research": ResearchConfigBuilder,
        "enterprise": EnterpriseConfigBuilder,
    }

    @classmethod
    def create_builder(
        self, persona: str, auto_detect: bool = True
    ) -> BaseConfigBuilder:
        """Create a configuration builder for the specified persona.

        Args:
            persona: The configuration persona (development, production, research, enterprise)
            auto_detect: Whether to enable service auto-detection

        Returns:
            Configured builder instance

        Raises:
            ValueError: If persona is not supported
        """
        if persona not in self._builders:
            available = ", ".join(self._builders.keys())
            raise ValueError(f"Unknown persona '{persona}'. Available: {available}")

        builder_class = self._builders[persona]
        return builder_class(auto_detect=auto_detect)

    @classmethod
    def get_available_personas(cls) -> List[str]:
        """Get list of available configuration personas."""
        return list(cls._builders.keys())

    @classmethod
    async def discover_optimal_persona(cls) -> str:
        """Discover the optimal persona based on environment auto-detection."""
        try:
            from ..services.auto_detection import EnvironmentDetector

            auto_config = AutoDetectionConfig()
            env_detector = EnvironmentDetector(auto_config)
            detected_env = await env_detector.detect()

            # Map environment to persona
            if detected_env.environment_type == Environment.PRODUCTION:
                if detected_env.cloud_provider:
                    return "enterprise"
                else:
                    return "production"
            elif detected_env.environment_type == Environment.DEVELOPMENT:
                return "development"
            else:
                return "development"  # Default fallback

        except Exception as e:
            logger.warning(f"Failed to discover optimal persona: {e}")
            return "development"  # Safe default


# Progressive Configuration Guide
class ProgressiveConfigurationGuide:
    """Guide users through progressive configuration discovery and setup."""

    def __init__(self, persona: str | None = None, auto_detect: bool = True):
        self.persona = persona
        self.auto_detect = auto_detect
        self.builder: BaseConfigBuilder | None = None

    async def start_guided_setup(self) -> ConfigurationDiscovery:
        """Start guided configuration setup with discovery."""
        # Discover optimal persona if not specified
        if not self.persona:
            self.persona = await ConfigBuilderFactory.discover_optimal_persona()
            logger.info(f"Auto-detected optimal persona: {self.persona}")

        # Create builder
        self.builder = ConfigBuilderFactory.create_builder(
            self.persona, auto_detect=self.auto_detect
        )

        # Perform discovery
        discovery = await self.builder.discover_configuration()

        logger.info(
            f"Configuration discovery complete for {discovery.persona} persona "
            f"({discovery.configuration_complexity} complexity, "
            f"{discovery.estimated_setup_time} setup time)"
        )

        return discovery

    def get_progressive_features(self) -> Dict[str, List[str]]:
        """Get features organized by complexity level for progressive disclosure."""
        if not self.builder:
            raise RuntimeError("Setup not started. Call start_guided_setup() first.")

        all_features = self.builder.get_recommended_features()

        # Categorize features by complexity
        essential = []
        intermediate = []
        advanced = []

        for feature in all_features:
            if any(
                keyword in feature.lower() for keyword in ["local", "debug", "basic"]
            ):
                essential.append(feature)
            elif any(
                keyword in feature.lower()
                for keyword in ["cache", "monitoring", "security"]
            ):
                intermediate.append(feature)
            else:
                advanced.append(feature)

        return {
            "essential": essential,
            "intermediate": intermediate,
            "advanced": advanced,
        }

    async def build_configuration(
        self, feature_level: str = "essential", **overrides
    ) -> Config:
        """Build configuration with specified feature level.

        Args:
            feature_level: Feature complexity level (essential, intermediate, advanced)
            **overrides: Configuration overrides

        Returns:
            Built configuration instance
        """
        if not self.builder:
            raise RuntimeError("Setup not started. Call start_guided_setup() first.")

        # Apply feature level adjustments
        config_overrides = self._get_feature_level_overrides(feature_level)
        config_overrides.update(overrides)

        return self.builder.build(**config_overrides)

    def _get_feature_level_overrides(self, feature_level: str) -> Dict[str, Any]:
        """Get configuration overrides for the specified feature level."""
        overrides = {}

        if feature_level == "essential":
            # Minimal configuration
            overrides.update(
                {
                    "monitoring": {"enabled": True, "enable_metrics": False},
                    "observability": {"enabled": False},
                    "security": {"enable_rate_limiting": False},
                }
            )
        elif feature_level == "intermediate":
            # Moderate configuration
            overrides.update(
                {
                    "monitoring": {"enabled": True, "enable_metrics": True},
                    "observability": {"enabled": True, "track_ai_operations": False},
                    "security": {"enable_rate_limiting": True},
                }
            )
        elif feature_level == "advanced":
            # Full configuration - use builder defaults
            pass
        else:
            raise ValueError(f"Unknown feature level: {feature_level}")

        return overrides

    def get_configuration_examples(self) -> Dict[str, str]:
        """Get configuration examples for different scenarios."""
        if not self.builder:
            return {}

        persona = self.builder.get_persona_name()

        examples = {
            "quick_start": f"""
# Quick start for {persona} persona
from src.config.builders import ConfigBuilderFactory

builder = ConfigBuilderFactory.create_builder("{persona}")
discovery = await builder.discover_configuration()
config = builder.build()
""",
            "with_overrides": f"""
# {persona.title()} with custom settings
builder = ConfigBuilderFactory.create_builder("{persona}")
config = builder.build(
    openai={{"api_key": "your-api-key"}},
    qdrant={{"url": "http://custom-qdrant:6333"}},
    performance={{"max_concurrent_requests": 10}}
)
""",
            "progressive_setup": f"""
# Progressive setup with guided discovery
from src.config.builders import ProgressiveConfigurationGuide

guide = ProgressiveConfigurationGuide(persona="{persona}")
discovery = await guide.start_guided_setup()

# Start with essential features
config = await guide.build_configuration("essential")

# Upgrade to intermediate when ready
config = await guide.build_configuration("intermediate")
""",
        }

        return examples


# Convenience functions for common use cases
async def quick_config(persona: str = "development", **overrides) -> Config:
    """Create configuration quickly with smart defaults.

    Args:
        persona: Configuration persona (development, production, research, enterprise)
        **overrides: Configuration overrides

    Returns:
        Built configuration instance
    """
    builder = ConfigBuilderFactory.create_builder(persona, auto_detect=True)
    return builder.build(**overrides)


async def guided_config_setup(
    persona: str | None = None,
) -> ProgressiveConfigurationGuide:
    """Set up configuration with guided discovery.

    Args:
        persona: Optional persona override (auto-detected if not provided)

    Returns:
        Progressive configuration guide instance
    """
    guide = ProgressiveConfigurationGuide(persona=persona, auto_detect=True)
    await guide.start_guided_setup()
    return guide


def validate_configuration_with_help(config_data: Dict[str, Any]) -> List[str]:
    """Validate configuration with helpful error messages.

    Args:
        config_data: Configuration data to validate

    Returns:
        List of validation errors with suggestions
    """
    try:
        Config(**config_data)
        return []
    except ValidationError as e:
        errors = []
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            msg = err["msg"]

            # Add helpful context
            if "api_key" in field.lower():
                msg += " (Set via environment variable or config file)"
            elif "url" in field.lower():
                msg += " (Check service connectivity and format)"
            elif "required" in msg.lower():
                msg += " (This field is required for the selected configuration)"

            errors.append(f"{field}: {msg}")

        return errors
