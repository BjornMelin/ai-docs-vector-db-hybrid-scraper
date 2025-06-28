"""Centralized health check utilities for external service connections.

This module provides a unified interface for checking connectivity to all
external services used by the AI Documentation Vector DB system.
"""

import asyncio

import httpx
import redis
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import Config


class CustomError(Exception):
    """Custom exception for this module."""


def _raise_openai_client_unavailable() -> None:
    """Raise CustomError for OpenAI client not available."""
    msg = "OpenAI client not available"
    raise CustomError(msg)


class ServiceHealthChecker:
    """Centralized service health checking utility."""

    @staticmethod
    def check_qdrant_connection(config: Config) -> dict[str, object]:
        """Check Qdrant connection status.

        Args:
            config: Unified configuration object containing Qdrant settings

        Returns:
            dict[str, object]: Connection status with:
                - service: Service name ("qdrant")
                - connected: Whether connection was successful
                - error: Error message if connection failed
                - details: Additional information (collections count, URL)

        """
        result = {"service": "qdrant", "connected": False, "error": None, "details": {}}

        try:
            client = QdrantClient(
                url=config.qdrant.url, api_key=config.qdrant.api_key, timeout=5.0
            )
            collections = client.get_collections()
            result["connected"] = True
            result["details"]["collections_count"] = len(collections.collections)
            result["details"]["url"] = config.qdrant.url

        except UnexpectedResponse as e:
            if e.status_code == 401:
                result["error"] = "Authentication failed - check API key"
            else:
                result["error"] = f"HTTP {e.status_code}: {e.reason_phrase}"
        except Exception as e:
            result["error"] = str(e)

        return result

    @staticmethod
    def check_dragonfly_connection(config: Config) -> dict[str, object]:
        """Check DragonflyDB connection status.

        Args:
            config: Unified configuration object

        Returns:
            Dictionary with connection status and details

        """
        result = {
            "service": "dragonfly",
            "connected": False,
            "error": None,
            "details": {},
        }

        if not config.cache.enable_dragonfly_cache:
            result["error"] = "DragonflyDB cache not enabled in configuration"
            return result

        try:
            r = redis.from_url(config.cache.dragonfly_url, socket_connect_timeout=5)
            r.ping()
            result["connected"] = True
            result["details"]["url"] = config.cache.dragonfly_url

        except redis.ConnectionError:
            result["error"] = "Connection refused - is DragonflyDB running?"
        except Exception as e:
            result["error"] = str(e)

        return result

    @staticmethod
    def check_openai_connection(
        config: Config, client_manager=None
    ) -> dict[str, object]:
        """Check OpenAI API connection status.

        Args:
            config: Unified configuration object containing OpenAI settings
            client_manager: Optional ClientManager instance for service layer access

        Returns:
            dict[str, object]: Connection status with:
                - service: Service name ("openai")
                - connected: Whether connection was successful
                - error: Error message if connection failed or not configured
                - details: Model info and available models count

        """
        result = {"service": "openai", "connected": False, "error": None, "details": {}}

        if config.embedding_provider != "openai" or not config.openai.api_key:
            result["error"] = (
                "OpenAI not configured as embedding provider or API key missing"
            )
            return result

        try:
            if client_manager:
                # Use modern service layer pattern through ClientManager

                async def _check_async():
                    client = await client_manager.get_openai_client()
                    if not client:
                        _raise_openai_client_unavailable()
                    models = await client.models.list()
                    return list(models)

                # Run async call in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    models = loop.run_until_complete(_check_async())
                finally:
                    loop.close()
            else:
                # Fallback to direct client instantiation (legacy pattern)

                client = OpenAI(api_key=config.openai.api_key, timeout=5.0)
                models = list(client.models.list())

            result["connected"] = True
            result["details"]["model"] = config.openai.model
            result["details"]["dimensions"] = config.openai.dimensions
            result["details"]["available_models_count"] = len(models)

        except Exception as e:
            result["error"] = str(e)

        return result

    @staticmethod
    def check_firecrawl_connection(config: Config) -> dict[str, object]:
        """Check Firecrawl API connection status.

        Args:
            config: Unified configuration object containing Firecrawl settings

        Returns:
            dict[str, object]: Connection status with:
                - service: Service name ("firecrawl")
                - connected: Whether connection was successful
                - error: Error message if connection failed or not configured
                - details: API URL and credits remaining (if available)

        """
        result = {
            "service": "firecrawl",
            "connected": False,
            "error": None,
            "details": {},
        }

        if config.crawl_provider != "firecrawl" or not config.firecrawl.api_key:
            result["error"] = (
                "Firecrawl not configured as crawl provider or API key missing"
            )
            return result

        try:
            headers = {"Authorization": f"Bearer {config.firecrawl.api_key}"}
            response = httpx.get(
                f"{config.firecrawl.api_url}/health", headers=headers, timeout=5.0
            )

            if response.status_code == 200:
                result["connected"] = True
                result["details"]["api_url"] = config.firecrawl.api_url
                result["details"]["status_code"] = response.status_code
            else:
                result["error"] = f"API returned status {response.status_code}"

        except Exception as e:
            result["error"] = str(e)

        return result

    @classmethod
    def perform_all_health_checks(
        cls, config: Config, client_manager=None
    ) -> dict[str, dict[str, object]]:
        """Perform health checks for all configured services.

        Args:
            config: Unified configuration object
            client_manager: Optional ClientManager instance for service layer access

        Returns:
            dict[str, dict[str, object]]: Service names mapped to health check results.
                Each result contains service, connected, error, and details fields.

        """
        results = {}

        # Always check Qdrant as it's required
        results["qdrant"] = cls.check_qdrant_connection(config)

        # Check cache services if enabled
        if config.cache.enable_dragonfly_cache:
            results["dragonfly"] = cls.check_dragonfly_connection(config)

        # Check embedding provider
        if config.embedding_provider == "openai":
            results["openai"] = cls.check_openai_connection(config, client_manager)

        # Check crawl provider
        if config.crawl_provider == "firecrawl":
            results["firecrawl"] = cls.check_firecrawl_connection(config)

        return results

    @classmethod
    def get_connection_summary(
        cls, config: Config, client_manager=None
    ) -> dict[str, object]:
        """Get a summary of all service connection statuses.

        Args:
            config: Unified configuration object
            client_manager: Optional ClientManager instance for service layer access

        Returns:
            Summary dictionary with overall status and individual service results

        """
        results = cls.perform_all_health_checks(config, client_manager)

        connected_services = [
            service for service, result in results.items() if result["connected"]
        ]

        failed_services = [
            service for service, result in results.items() if not result["connected"]
        ]

        return {
            "overall_status": "healthy" if not failed_services else "unhealthy",
            "total_services": len(results),
            "connected_count": len(connected_services),
            "failed_count": len(failed_services),
            "connected_services": connected_services,
            "failed_services": failed_services,
            "detailed_results": results,
        }