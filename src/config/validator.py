"""Configuration validation utilities with detailed environment variable checks.

This module provides comprehensive validation for configuration values,
especially focusing on environment variable validation and type checking.
"""

import os
import re
from typing import Any
from urllib.parse import urlparse

from .models import UnifiedConfig


class ConfigValidator:
    """Advanced configuration validation utilities."""

    @staticmethod
    def validate_env_var_format(
        var_name: str, expected_pattern: str | None = None
    ) -> bool:
        """Validate environment variable name format.

        Args:
            var_name: Environment variable name
            expected_pattern: Optional regex pattern to match

        Returns:
            True if valid format
        """
        # Check basic format (uppercase with underscores)
        if not re.match(r"^[A-Z][A-Z0-9_]*$", var_name):
            return False

        # Check against expected pattern if provided
        return not (expected_pattern and not re.match(expected_pattern, var_name))

    @staticmethod
    def validate_url(url: str, schemes: list[str] | None = None) -> tuple[bool, str]:
        """Validate URL format and optionally check connectivity.

        Args:
            url: URL to validate
            schemes: Allowed URL schemes (default: ['http', 'https'])

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not schemes:
            schemes = ["http", "https"]

        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in schemes:
                return (
                    False,
                    f"Invalid URL scheme: {parsed.scheme}. Expected one of: {schemes}",
                )

            # Check netloc
            if not parsed.netloc:
                return False, "URL missing network location (host)"

            return True, ""

        except Exception as e:
            return False, f"URL parsing error: {e}"

    @staticmethod
    def validate_api_key(key: str, provider: str) -> tuple[bool, str]:
        """Validate API key format for specific providers.

        Args:
            key: API key to validate
            provider: Provider name (openai, firecrawl, etc.)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not key:
            return False, "API key is empty"

        # Provider-specific validation
        if provider == "openai":
            if not key.startswith("sk-"):
                return False, "OpenAI API key must start with 'sk-'"
            if len(key) < 40:
                return False, "OpenAI API key appears too short"

        elif provider == "firecrawl":
            # Firecrawl uses various formats
            if len(key) < 20:
                return False, "Firecrawl API key appears too short"

        elif provider == "qdrant":
            # Qdrant API key is optional but if provided should be non-empty
            if key and len(key) < 10:
                return False, "Qdrant API key appears too short"

        return True, ""

    @staticmethod
    def validate_env_var_value(  # noqa: PLR0911
        var_name: str, value: str, expected_type: type
    ) -> tuple[bool, Any, str]:
        """Validate and convert environment variable value to expected type.

        Args:
            var_name: Variable name (for error messages)
            value: String value from environment
            expected_type: Expected Python type

        Returns:
            Tuple of (is_valid, converted_value, error_message)
        """
        try:
            # Handle different types
            if expected_type is bool:
                if value.lower() in ["true", "1", "yes", "on"]:
                    return True, True, ""
                elif value.lower() in ["false", "0", "no", "off"]:
                    return True, False, ""
                else:
                    return False, None, f"{var_name}: Invalid boolean value '{value}'"

            elif expected_type is int:
                return True, int(value), ""

            elif expected_type is float:
                return True, float(value), ""

            elif expected_type in [list, dict]:
                # Try to parse as JSON
                import json

                parsed = json.loads(value)
                if not isinstance(parsed, expected_type):
                    return (
                        False,
                        None,
                        f"{var_name}: Expected {expected_type.__name__}, got {type(parsed).__name__}",
                    )
                return True, parsed, ""

            else:
                # String or other types
                return True, value, ""

        except ValueError as e:
            return (
                False,
                None,
                f"{var_name}: Failed to convert '{value}' to {expected_type.__name__}: {e}",
            )
        except Exception as e:
            if "json" in str(e) or "JSONDecodeError" in str(type(e).__name__):
                return False, None, f"{var_name}: Failed to parse JSON '{value}': {e}"
            return (
                False,
                None,
                f"{var_name}: Unexpected error converting '{value}': {e}",
            )

    @staticmethod
    def check_env_vars(prefix: str = "AI_DOCS__") -> dict[str, dict[str, Any]]:
        """Check all environment variables with the given prefix.

        Returns:
            Dictionary mapping variable names to their status and issues
        """
        results = {}

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            result = {"value": value, "valid_format": True, "issues": []}

            # Check format
            if not ConfigValidator.validate_env_var_format(key):
                result["valid_format"] = False
                result["issues"].append("Invalid variable name format")

            # Check for common issues
            if not value:
                result["issues"].append("Empty value")
            elif value.startswith(" ") or value.endswith(" "):
                result["issues"].append("Value has leading/trailing whitespace")
            elif "your-" in value or "xxx" in value:
                result["issues"].append("Value appears to be a placeholder")

            results[key] = result

        return results

    @staticmethod
    def validate_config_connections(config: UnifiedConfig) -> dict[str, dict[str, Any]]:
        """Validate connections to external services.

        Returns:
            Dictionary mapping service names to their connection status
        """
        results = {}

        # Check Qdrant
        results["qdrant"] = {"connected": False, "error": None}
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.exceptions import UnexpectedResponse

            client = QdrantClient(
                url=config.qdrant.url, api_key=config.qdrant.api_key, timeout=5.0
            )
            client.get_collections()
            results["qdrant"]["connected"] = True
        except UnexpectedResponse as e:
            if e.status_code == 401:
                results["qdrant"]["error"] = "Authentication failed - check API key"
            else:
                results["qdrant"]["error"] = f"HTTP {e.status_code}: {e.reason_phrase}"
        except Exception as e:
            results["qdrant"]["error"] = str(e)

        # Check DragonflyDB/Redis if enabled
        if config.cache.enable_dragonfly_cache:
            results["redis"] = {"connected": False, "error": None}
            try:
                import redis

                r = redis.from_url(config.cache.dragonfly_url, socket_connect_timeout=5)
                r.ping()
                results["redis"]["connected"] = True
            except redis.ConnectionError:
                results["redis"]["error"] = "Connection refused - is Redis running?"
            except Exception as e:
                results["redis"]["error"] = str(e)

        # Check OpenAI if configured
        if config.embedding_provider == "openai" and config.openai.api_key:
            results["openai"] = {"connected": False, "error": None}
            try:
                from openai import OpenAI

                client = OpenAI(api_key=config.openai.api_key, timeout=5.0)
                # Just list models to test connection
                list(client.models.list())
                results["openai"]["connected"] = True
            except Exception as e:
                results["openai"]["error"] = str(e)

        return results

    @staticmethod
    def generate_validation_report(config: UnifiedConfig) -> str:
        """Generate a comprehensive validation report.

        Returns:
            Formatted validation report as string
        """
        report = []
        report.append("=" * 60)
        report.append("AI Documentation Vector DB Configuration Validation Report")
        report.append("=" * 60)
        report.append("")

        # Basic validation
        issues = config.validate_completeness()
        if issues:
            report.append("⚠️  Configuration Issues:")
            for issue in issues:
                report.append(f"   - {issue}")
        else:
            report.append("✅ Basic configuration validation passed")

        report.append("")

        # Environment variables
        report.append("Environment Variables:")
        env_vars = ConfigValidator.check_env_vars()
        if env_vars:
            for var, info in env_vars.items():
                status = "✅" if not info["issues"] else "⚠️"
                report.append(f"  {status} {var}")
                if info["issues"]:
                    for issue in info["issues"]:
                        report.append(f"     - {issue}")
        else:
            report.append(
                "  [info] No environment variables found with prefix AI_DOCS__"
            )

        report.append("")

        # Service connections
        report.append("Service Connections:")
        connections = ConfigValidator.validate_config_connections(config)
        for service, info in connections.items():
            status = "✅" if info["connected"] else "❌"
            report.append(f"  {status} {service.capitalize()}")
            if info["error"]:
                report.append(f"     Error: {info['error']}")

        report.append("")

        # Configuration details
        report.append("Active Configuration:")
        report.append(f"  Environment: {config.environment}")
        report.append(f"  Debug Mode: {config.debug}")
        report.append(f"  Log Level: {config.log_level}")
        report.append(f"  Embedding Provider: {config.embedding_provider}")
        report.append(f"  Crawl Provider: {config.crawl_provider}")
        report.append(f"  Caching Enabled: {config.cache.enable_caching}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
