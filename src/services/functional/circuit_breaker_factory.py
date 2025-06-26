"""Circuit breaker factory for service integration.

Provides a unified factory interface for creating circuit breakers based on configuration.
Supports both legacy and enhanced circuit breaker implementations.
"""

import logging
from typing import Any

from src.config.core import Config

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    EnhancedCircuitBreakerConfig,
)


logger = logging.getLogger(__name__)


class CircuitBreakerFactory:
    """Factory for creating circuit breakers based on configuration."""

    def __init__(self, config: Config):
        """Initialize factory with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self._breaker_registry: dict[str, EnhancedCircuitBreaker | CircuitBreaker] = {}

    def create_service_circuit_breaker(
        self, service_name: str, config_overrides: dict[str, Any] | None = None
    ) -> EnhancedCircuitBreaker | CircuitBreaker:
        """Create a circuit breaker for a specific service.

        Args:
            service_name: Name of the service (e.g., "openai", "firecrawl")
            config_overrides: Additional configuration overrides

        Returns:
            Configured circuit breaker instance
        """
        # Check if already created for this service
        if service_name in self._breaker_registry:
            return self._breaker_registry[service_name]

        # Use enhanced circuit breaker if enabled
        if self.config.circuit_breaker.use_enhanced_circuit_breaker:
            breaker = self._create_enhanced_circuit_breaker(
                service_name, config_overrides
            )
        else:
            breaker = self._create_legacy_circuit_breaker(
                service_name, config_overrides
            )

        # Register for monitoring
        self._breaker_registry[service_name] = breaker
        logger.info(f"Created circuit breaker for service '{service_name}'")

        return breaker

    def _create_enhanced_circuit_breaker(
        self, service_name: str, config_overrides: dict[str, Any] | None = None
    ) -> EnhancedCircuitBreaker:
        """Create an enhanced circuit breaker."""
        # Get service-specific configuration
        service_config = self.config.circuit_breaker.service_overrides.get(
            service_name, {}
        )

        # Apply additional overrides
        if config_overrides:
            service_config.update(config_overrides)

        # Create enhanced circuit breaker configuration
        enhanced_config = EnhancedCircuitBreakerConfig(
            service_name=service_name,
            failure_threshold=service_config.get(
                "failure_threshold", self.config.circuit_breaker.failure_threshold
            ),
            recovery_timeout=service_config.get(
                "recovery_timeout", self.config.circuit_breaker.recovery_timeout
            ),
            enable_metrics=service_config.get(
                "enable_metrics", self.config.circuit_breaker.enable_metrics_collection
            ),
            enable_fallback=service_config.get(
                "enable_fallback",
                self.config.circuit_breaker.enable_fallback_mechanisms,
            ),
            collect_detailed_metrics=self.config.circuit_breaker.enable_detailed_metrics,
        )

        return EnhancedCircuitBreaker(enhanced_config)

    def _create_legacy_circuit_breaker(
        self, service_name: str, config_overrides: dict[str, Any] | None = None
    ) -> CircuitBreaker:
        """Create a legacy circuit breaker."""
        # Get service-specific configuration
        service_config = self.config.circuit_breaker.service_overrides.get(
            service_name, {}
        )

        # Apply additional overrides
        if config_overrides:
            service_config.update(config_overrides)

        # Create legacy circuit breaker configuration
        legacy_config = CircuitBreakerConfig(
            failure_threshold=service_config.get(
                "failure_threshold", self.config.circuit_breaker.failure_threshold
            ),
            recovery_timeout=service_config.get(
                "recovery_timeout", self.config.circuit_breaker.recovery_timeout
            ),
            enable_metrics=service_config.get(
                "enable_metrics", self.config.circuit_breaker.enable_metrics_collection
            ),
            enable_fallback=service_config.get(
                "enable_fallback",
                self.config.circuit_breaker.enable_fallback_mechanisms,
            ),
        )

        return CircuitBreaker(legacy_config)

    def get_service_circuit_breaker(
        self, service_name: str
    ) -> EnhancedCircuitBreaker | CircuitBreaker | None:
        """Get an existing circuit breaker for a service.

        Args:
            service_name: Name of the service

        Returns:
            Circuit breaker instance or None if not found
        """
        return self._breaker_registry.get(service_name)

    def get_all_circuit_breakers(
        self,
    ) -> dict[str, EnhancedCircuitBreaker | CircuitBreaker]:
        """Get all registered circuit breakers.

        Returns:
            Dictionary of service names to circuit breaker instances
        """
        return self._breaker_registry.copy()

    def get_circuit_breaker_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all circuit breakers.

        Returns:
            Dictionary of service names to metrics
        """
        metrics = {}
        for service_name, breaker in self._breaker_registry.items():
            try:
                metrics[service_name] = breaker.get_metrics()
            except Exception as e:
                logger.warning(
                    f"Failed to get metrics for service '{service_name}': {e}"
                )
                metrics[service_name] = {"error": str(e)}

        return metrics

    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        for service_name, breaker in self._breaker_registry.items():
            try:
                breaker.reset()
                logger.info(f"Reset circuit breaker for service '{service_name}'")
            except Exception as e:
                logger.warning(
                    f"Failed to reset circuit breaker for service '{service_name}': {e}"
                )

    def create_circuit_breaker_for_external_service(
        self, service_name: str, endpoint_url: str, **kwargs: Any
    ) -> EnhancedCircuitBreaker | CircuitBreaker:
        """Create a circuit breaker for an external service with URL-based configuration.

        Args:
            service_name: Name of the service
            endpoint_url: Service endpoint URL
            **kwargs: Additional configuration parameters

        Returns:
            Configured circuit breaker instance
        """
        # Determine appropriate default configuration based on URL patterns
        config_defaults = self._get_defaults_for_url(endpoint_url)
        config_defaults.update(kwargs)

        return self.create_service_circuit_breaker(service_name, config_defaults)

    def _get_defaults_for_url(self, url: str) -> dict[str, Any]:
        """Get default configuration based on URL patterns.

        Args:
            url: Service endpoint URL

        Returns:
            Default configuration dictionary
        """
        url_lower = url.lower()

        # AI service endpoints typically need more lenient settings
        if any(
            ai_service in url_lower for ai_service in ["openai", "anthropic", "google"]
        ):
            return {
                "failure_threshold": 3,
                "recovery_timeout": 30.0,
                "enable_fallback": True,
            }

        # Vector database endpoints
        if any(
            db_service in url_lower for db_service in ["qdrant", "pinecone", "weaviate"]
        ):
            return {
                "failure_threshold": 3,
                "recovery_timeout": 15.0,
                "enable_fallback": False,
            }

        # Crawling services
        if any(
            crawl_service in url_lower for crawl_service in ["firecrawl", "scrapfly"]
        ):
            return {
                "failure_threshold": 5,
                "recovery_timeout": 60.0,
                "enable_fallback": True,
            }

        # Redis/cache services
        if any(cache_service in url_lower for cache_service in ["redis", "memcached"]):
            return {
                "failure_threshold": 2,
                "recovery_timeout": 10.0,
                "enable_fallback": False,
            }

        # Default configuration for unknown services
        return {
            "failure_threshold": 5,
            "recovery_timeout": 60.0,
            "enable_fallback": True,
        }


# Global factory instance
_circuit_breaker_factory: CircuitBreakerFactory | None = None


def get_circuit_breaker_factory(config: Config | None = None) -> CircuitBreakerFactory:
    """Get the global circuit breaker factory instance.

    Args:
        config: Optional configuration to initialize the factory

    Returns:
        Circuit breaker factory instance
    """
    global _circuit_breaker_factory  # noqa: PLW0603

    if _circuit_breaker_factory is None:
        if config is None:
            from src.config.core import get_config

            config = get_config()
        _circuit_breaker_factory = CircuitBreakerFactory(config)

    return _circuit_breaker_factory


def reset_circuit_breaker_factory() -> None:
    """Reset the global circuit breaker factory."""
    global _circuit_breaker_factory  # noqa: PLW0603
    _circuit_breaker_factory = None


# Convenience functions for common service types
def create_openai_circuit_breaker(
    config: Config | None = None,
) -> EnhancedCircuitBreaker | CircuitBreaker:
    """Create a circuit breaker specifically for OpenAI API calls."""
    factory = get_circuit_breaker_factory(config)
    return factory.create_service_circuit_breaker("openai")


def create_firecrawl_circuit_breaker(
    config: Config | None = None,
) -> EnhancedCircuitBreaker | CircuitBreaker:
    """Create a circuit breaker specifically for Firecrawl API calls."""
    factory = get_circuit_breaker_factory(config)
    return factory.create_service_circuit_breaker("firecrawl")


def create_qdrant_circuit_breaker(
    config: Config | None = None,
) -> EnhancedCircuitBreaker | CircuitBreaker:
    """Create a circuit breaker specifically for Qdrant database calls."""
    factory = get_circuit_breaker_factory(config)
    return factory.create_service_circuit_breaker("qdrant")


def create_redis_circuit_breaker(
    config: Config | None = None,
) -> EnhancedCircuitBreaker | CircuitBreaker:
    """Create a circuit breaker specifically for Redis cache calls."""
    factory = get_circuit_breaker_factory(config)
    return factory.create_service_circuit_breaker("redis")


def create_crawl4ai_circuit_breaker(
    config: Config | None = None,
) -> EnhancedCircuitBreaker | CircuitBreaker:
    """Create a circuit breaker specifically for Crawl4AI calls."""
    factory = get_circuit_breaker_factory(config)
    return factory.create_service_circuit_breaker("crawl4ai")
