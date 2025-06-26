"""Configuration discovery and intelligent validation helpers.

This module provides advanced configuration discovery capabilities that showcase
sophisticated auto-detection, intelligent defaults, and helpful validation patterns.

Portfolio showcase elements:
- Smart environment and service discovery
- Context-aware configuration recommendations
- Intelligent validation with helpful error messages
- Configuration optimization suggestions
"""

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from .auto_detect import AutoDetectedServices, AutoDetectionConfig
from .core import Config
from .enums import CrawlProvider, EmbeddingProvider, Environment


logger = logging.getLogger(__name__)


class SystemEnvironment(BaseModel):
    """Detected system environment information."""

    platform: str = Field(description="Operating system platform")
    python_version: str = Field(description="Python version")
    architecture: str = Field(description="System architecture")
    memory_gb: float = Field(description="Available memory in GB")
    cpu_count: int = Field(description="Number of CPU cores")
    is_docker: bool = Field(description="Running in Docker container")
    is_ci: bool = Field(description="Running in CI environment")
    has_gpu: bool = Field(description="GPU available")


class ConfigurationRecommendation(BaseModel):
    """Smart configuration recommendation based on environment analysis."""

    recommended_persona: str = Field(description="Recommended configuration persona")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Recommendation confidence"
    )
    reasoning: List[str] = Field(description="Reasoning for recommendation")
    suggested_providers: Dict[str, str] = Field(
        description="Suggested service providers"
    )
    performance_recommendations: List[str] = Field(
        description="Performance optimizations"
    )
    security_recommendations: List[str] = Field(description="Security recommendations")
    estimated_costs: Dict[str, str] = Field(description="Estimated operational costs")


class ConfigurationValidationReport(BaseModel):
    """Comprehensive configuration validation report."""

    is_valid: bool = Field(description="Overall validation status")
    error_count: int = Field(description="Number of validation errors")
    warning_count: int = Field(description="Number of warnings")
    suggestions_count: int = Field(description="Number of improvement suggestions")

    errors: List[Dict[str, Any]] = Field(description="Validation errors with context")
    warnings: List[Dict[str, Any]] = Field(description="Configuration warnings")
    suggestions: List[Dict[str, Any]] = Field(description="Optimization suggestions")

    security_score: int = Field(
        ge=0, le=100, description="Security configuration score"
    )
    performance_score: int = Field(
        ge=0, le=100, description="Performance configuration score"
    )
    maintainability_score: int = Field(
        ge=0, le=100, description="Maintainability score"
    )


class ConfigurationOptimizer:
    """Intelligent configuration optimizer with context-aware recommendations."""

    def __init__(self):
        self.system_env: SystemEnvironment | None = None
        self.auto_detected: AutoDetectedServices | None = None

    async def analyze_environment(self) -> SystemEnvironment:
        """Analyze system environment for configuration optimization."""
        try:
            # Detect system characteristics
            memory_bytes = (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                if hasattr(os, "sysconf")
                else 8 * 1024**3
            )
            memory_gb = memory_bytes / (1024**3)

            # Check for Docker
            is_docker = os.path.exists("/.dockerenv") or (
                os.path.exists("/proc/1/cgroup")
                and "docker" in open("/proc/1/cgroup").read()
            )

            # Check for CI environment
            ci_indicators = [
                "CI",
                "CONTINUOUS_INTEGRATION",
                "GITHUB_ACTIONS",
                "GITLAB_CI",
                "JENKINS_URL",
            ]
            is_ci = any(os.getenv(indicator) for indicator in ci_indicators)

            # Check for GPU (simplified)
            has_gpu = any(
                keyword in str(platform.processor()).lower()
                for keyword in ["nvidia", "amd", "gpu"]
            )

            self.system_env = SystemEnvironment(
                platform=platform.system(),
                python_version=sys.version.split()[0],
                architecture=platform.machine(),
                memory_gb=round(memory_gb, 1),
                cpu_count=os.cpu_count() or 1,
                is_docker=is_docker,
                is_ci=is_ci,
                has_gpu=has_gpu,
            )

            logger.info(
                f"System analysis complete: {self.system_env.platform} "
                f"with {self.system_env.memory_gb}GB RAM, {self.system_env.cpu_count} cores"
            )

            return self.system_env

        except Exception as e:
            logger.warning(f"System analysis failed: {e}")
            # Return safe defaults
            return SystemEnvironment(
                platform="Unknown",
                python_version=sys.version.split()[0],
                architecture="Unknown",
                memory_gb=8.0,
                cpu_count=4,
                is_docker=False,
                is_ci=False,
                has_gpu=False,
            )

    async def generate_recommendations(
        self, auto_detected: AutoDetectedServices | None = None
    ) -> ConfigurationRecommendation:
        """Generate intelligent configuration recommendations."""

        if not self.system_env:
            await self.analyze_environment()

        self.auto_detected = auto_detected

        # Determine optimal persona
        persona, confidence, reasoning = self._recommend_persona()

        # Suggest providers based on environment
        providers = self._recommend_providers()

        # Performance recommendations
        performance_recs = self._generate_performance_recommendations()

        # Security recommendations
        security_recs = self._generate_security_recommendations()

        # Cost estimates
        costs = self._estimate_costs(providers)

        return ConfigurationRecommendation(
            recommended_persona=persona,
            confidence_score=confidence,
            reasoning=reasoning,
            suggested_providers=providers,
            performance_recommendations=performance_recs,
            security_recommendations=security_recs,
            estimated_costs=costs,
        )

    def _recommend_persona(self) -> Tuple[str, float, List[str]]:
        """Recommend optimal persona based on environment analysis."""
        reasoning = []
        confidence = 0.8  # Base confidence

        # Check for production indicators
        if self.auto_detected and self.auto_detected.environment.cloud_provider:
            reasoning.append("Cloud deployment detected")
            confidence += 0.1
            return "enterprise", min(confidence, 1.0), reasoning

        if self.system_env.is_ci:
            reasoning.append("CI environment detected")
            return "production", confidence, reasoning

        if self.system_env.memory_gb >= 16 and self.system_env.cpu_count >= 8:
            reasoning.append("High-performance system detected")
            if self.auto_detected and len(self.auto_detected.services) >= 2:
                reasoning.append("Multiple services available")
                return "enterprise", confidence, reasoning
            return "production", confidence, reasoning

        if self.system_env.memory_gb >= 8:
            reasoning.append("Adequate resources for development")
            return "development", confidence, reasoning

        reasoning.append("Limited resources detected")
        return "development", confidence - 0.2, reasoning

    def _recommend_providers(self) -> Dict[str, str]:
        """Recommend optimal service providers."""
        providers = {}

        # Embedding provider recommendation
        if self.system_env.memory_gb >= 8 and not self.system_env.is_ci:
            if os.getenv("OPENAI_API_KEY"):
                providers["embedding"] = "openai"
            else:
                providers["embedding"] = "fastembed"  # Local fallback
        else:
            providers["embedding"] = "fastembed"  # Resource-efficient

        # Crawling provider recommendation
        if self.system_env.memory_gb >= 4:
            providers["crawling"] = "crawl4ai"
        else:
            providers["crawling"] = "firecrawl"  # External service for low resources

        # Cache provider recommendation
        if self.auto_detected and self.auto_detected.redis_service:
            providers["cache"] = "redis"
        else:
            providers["cache"] = "local"

        # Vector DB recommendation
        if self.auto_detected and self.auto_detected.qdrant_service:
            providers["vector_db"] = "qdrant"
        else:
            providers["vector_db"] = "local_qdrant"

        return providers

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if self.system_env.memory_gb >= 16:
            recommendations.append("Increase batch sizes for better throughput")
            recommendations.append("Enable parallel processing")
        elif self.system_env.memory_gb <= 4:
            recommendations.append("Reduce batch sizes to prevent OOM")
            recommendations.append("Enable memory optimization features")

        if self.system_env.cpu_count >= 8:
            recommendations.append(
                f"Set max_concurrent_requests to {self.system_env.cpu_count * 2}"
            )

        if self.system_env.has_gpu:
            recommendations.append("Consider GPU-accelerated embedding models")

        if self.system_env.is_docker:
            recommendations.append("Configure container resource limits")
            recommendations.append("Use multi-stage builds for smaller images")

        return recommendations

    def _generate_security_recommendations(self) -> List[str]:
        """Generate security configuration recommendations."""
        recommendations = []

        if not self.system_env.is_ci:
            recommendations.append("Enable API key authentication")
            recommendations.append("Configure rate limiting")

        if self.auto_detected and self.auto_detected.environment.cloud_provider:
            recommendations.append("Enable security headers")
            recommendations.append("Configure TLS/SSL termination")
            recommendations.append("Set up monitoring and alerting")

        if self.system_env.platform == "Linux":
            recommendations.append("Consider running services as non-root user")

        recommendations.append("Regular security updates and vulnerability scanning")

        return recommendations

    def _estimate_costs(self, providers: Dict[str, str]) -> Dict[str, str]:
        """Estimate operational costs based on provider selection."""
        costs = {}

        if providers.get("embedding") == "openai":
            costs["embeddings"] = "$0.01-$0.10 per 1M tokens"
        else:
            costs["embeddings"] = "Free (local processing)"

        if providers.get("crawling") == "firecrawl":
            costs["crawling"] = "$0.001-$0.01 per page"
        else:
            costs["crawling"] = "Free (local processing)"

        if providers.get("cache") == "redis" and self.auto_detected:
            costs["cache"] = "Infrastructure cost (Redis server)"
        else:
            costs["cache"] = "Free (local memory)"

        return costs


class IntelligentValidator:
    """Intelligent configuration validator with helpful error messages."""

    def __init__(self):
        self.optimizer = ConfigurationOptimizer()

    async def validate_with_intelligence(
        self, config_data: Dict[str, Any], persona: str | None = None
    ) -> ConfigurationValidationReport:
        """Perform intelligent validation with context-aware suggestions."""

        errors = []
        warnings = []
        suggestions = []

        # Basic Pydantic validation
        try:
            config = Config(**config_data)
            is_valid = True
        except ValidationError as e:
            is_valid = False
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                errors.append(
                    {
                        "field": field_path,
                        "message": error["msg"],
                        "value": error.get("input"),
                        "suggestions": self._get_field_suggestions(field_path, error),
                    }
                )

        # Context-aware validation
        await self._validate_context_aware(config_data, warnings, suggestions, persona)

        # Calculate scores
        security_score = self._calculate_security_score(config_data)
        performance_score = self._calculate_performance_score(config_data)
        maintainability_score = self._calculate_maintainability_score(config_data)

        return ConfigurationValidationReport(
            is_valid=is_valid,
            error_count=len(errors),
            warning_count=len(warnings),
            suggestions_count=len(suggestions),
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            security_score=security_score,
            performance_score=performance_score,
            maintainability_score=maintainability_score,
        )

    def _get_field_suggestions(
        self, field_path: str, error: Dict[str, Any]
    ) -> List[str]:
        """Get helpful suggestions for specific field validation errors."""
        suggestions = []

        if "api_key" in field_path.lower():
            if "openai" in field_path.lower():
                suggestions.extend(
                    [
                        "Get API key from https://platform.openai.com/api-keys",
                        "Set via environment variable: AI_DOCS_OPENAI__API_KEY=sk-...",
                        "API key should start with 'sk-'",
                    ]
                )
            elif "firecrawl" in field_path.lower():
                suggestions.extend(
                    [
                        "Get API key from https://firecrawl.dev",
                        "Set via environment variable: AI_DOCS_FIRECRAWL__API_KEY=fc-...",
                        "API key should start with 'fc-'",
                    ]
                )

        elif "url" in field_path.lower():
            if "qdrant" in field_path.lower():
                suggestions.extend(
                    [
                        "Default Qdrant URL: http://localhost:6333",
                        "For Docker: http://qdrant:6333",
                        "Check Qdrant service is running",
                    ]
                )
            elif "redis" in field_path.lower():
                suggestions.extend(
                    [
                        "Default Redis URL: redis://localhost:6379",
                        "For Docker: redis://redis:6379",
                        "Check Redis/Dragonfly service is running",
                    ]
                )

        elif "embedding_provider" in field_path:
            suggestions.extend(
                [
                    "Valid providers: 'openai', 'fastembed'",
                    "Use 'fastembed' for local, cost-free embeddings",
                    "Use 'openai' for higher quality (requires API key)",
                ]
            )

        elif "crawl_provider" in field_path:
            suggestions.extend(
                [
                    "Valid providers: 'crawl4ai', 'firecrawl'",
                    "Use 'crawl4ai' for local browser automation",
                    "Use 'firecrawl' for cloud-based crawling (requires API key)",
                ]
            )

        return suggestions

    async def _validate_context_aware(
        self,
        config_data: Dict[str, Any],
        warnings: List[Dict[str, Any]],
        suggestions: List[Dict[str, Any]],
        persona: str | None = None,
    ) -> None:
        """Perform context-aware validation and generate suggestions."""

        # Analyze system environment
        await self.optimizer.analyze_environment()
        system_env = self.optimizer.system_env

        # Check resource compatibility
        if system_env.memory_gb < 4:
            warnings.append(
                {
                    "category": "performance",
                    "message": "Low memory detected, consider reducing batch sizes",
                    "field": "performance.max_memory_usage_mb",
                    "recommendation": "Set max_memory_usage_mb to 2000 or less",
                }
            )

        # Check provider configuration
        embedding_provider = config_data.get("embedding_provider")
        if embedding_provider == "openai":
            openai_config = config_data.get("openai", {})
            if not openai_config.get("api_key"):
                warnings.append(
                    {
                        "category": "configuration",
                        "message": "OpenAI provider selected but no API key configured",
                        "field": "openai.api_key",
                        "recommendation": "Set OPENAI_API_KEY environment variable",
                    }
                )

        # Environment-specific suggestions
        if persona == "production":
            if not config_data.get("monitoring", {}).get("enabled"):
                suggestions.append(
                    {
                        "category": "monitoring",
                        "message": "Enable monitoring for production deployments",
                        "field": "monitoring.enabled",
                        "recommendation": "Set monitoring.enabled = true",
                    }
                )

            if not config_data.get("security", {}).get("enable_rate_limiting"):
                suggestions.append(
                    {
                        "category": "security",
                        "message": "Enable rate limiting for production security",
                        "field": "security.enable_rate_limiting",
                        "recommendation": "Set security.enable_rate_limiting = true",
                    }
                )

    def _calculate_security_score(self, config_data: Dict[str, Any]) -> int:
        """Calculate security configuration score (0-100)."""
        score = 0
        max_score = 100

        security_config = config_data.get("security", {})

        # API key enforcement (+20)
        if security_config.get("require_api_keys"):
            score += 20

        # Rate limiting (+20)
        if security_config.get("enable_rate_limiting"):
            score += 20

        # Security headers (+20)
        if security_config.get("x_frame_options"):
            score += 10
        if security_config.get("content_security_policy"):
            score += 10

        # HTTPS enforcement (+15)
        if "https" in str(config_data.get("qdrant", {}).get("url", "")):
            score += 15

        # Monitoring enabled (+25)
        if config_data.get("monitoring", {}).get("enabled"):
            score += 25

        return min(score, max_score)

    def _calculate_performance_score(self, config_data: Dict[str, Any]) -> int:
        """Calculate performance configuration score (0-100)."""
        score = 0
        max_score = 100

        performance_config = config_data.get("performance", {})
        cache_config = config_data.get("cache", {})

        # Caching enabled (+25)
        if cache_config.get("enable_caching"):
            score += 25

        # Distributed caching (+15)
        if cache_config.get("enable_dragonfly_cache"):
            score += 15

        # Reasonable concurrency (+20)
        max_requests = performance_config.get("max_concurrent_requests", 10)
        if 5 <= max_requests <= 50:
            score += 20

        # Circuit breakers (+20)
        if config_data.get("circuit_breaker", {}).get("use_enhanced_circuit_breaker"):
            score += 20

        # Batch processing (+20)
        embedding_batch = config_data.get("openai", {}).get(
            "batch_size"
        ) or config_data.get("fastembed", {}).get("batch_size")
        if embedding_batch and embedding_batch >= 50:
            score += 20

        return min(score, max_score)

    def _calculate_maintainability_score(self, config_data: Dict[str, Any]) -> int:
        """Calculate maintainability configuration score (0-100)."""
        score = 0
        max_score = 100

        # Monitoring and observability (+30)
        if config_data.get("monitoring", {}).get("enabled"):
            score += 15
        if config_data.get("observability", {}).get("enabled"):
            score += 15

        # Health checks (+20)
        if config_data.get("monitoring", {}).get("enable_health_checks"):
            score += 20

        # Configuration drift detection (+25)
        if config_data.get("drift_detection", {}).get("enabled"):
            score += 25

        # Reasonable defaults (+25)
        if config_data.get("environment") in ["development", "production"]:
            score += 10
        if config_data.get("log_level") in ["DEBUG", "INFO", "WARNING"]:
            score += 15

        return min(score, max_score)


# Convenience functions
async def discover_optimal_configuration(
    persona: str | None = None,
) -> ConfigurationRecommendation:
    """Discover optimal configuration based on environment analysis."""
    optimizer = ConfigurationOptimizer()
    await optimizer.analyze_environment()
    return await optimizer.generate_recommendations()


async def validate_configuration_intelligently(
    config_data: Dict[str, Any], persona: str | None = None
) -> ConfigurationValidationReport:
    """Validate configuration with intelligent suggestions."""
    validator = IntelligentValidator()
    return await validator.validate_with_intelligence(config_data, persona)


async def get_system_recommendations() -> Tuple[
    SystemEnvironment, ConfigurationRecommendation
]:
    """Get system analysis and configuration recommendations."""
    optimizer = ConfigurationOptimizer()
    system_env = await optimizer.analyze_environment()
    recommendations = await optimizer.generate_recommendations()
    return system_env, recommendations
