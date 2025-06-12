"""Configuration validators for the unified config system.

This module provides validation functions and classes for configuration models,
avoiding circular import dependencies by keeping validators close to the models.
Consolidates validators.py and validator.py into a single module.
"""

import os
import re
from typing import Any
from urllib.parse import urlparse


def validate_api_key_common(
    value: str | None,
    prefix: str,
    service_name: str,
    min_length: int = 10,
    max_length: int = 200,
    allowed_chars: str = r"[A-Za-z0-9-]+",
) -> str | None:
    """Validate API key format and structure.

    Args:
        value: API key value to validate
        prefix: Expected prefix (e.g., "sk-", "fc-")
        service_name: Service name for error messages
        min_length: Minimum key length
        max_length: Maximum key length
        allowed_chars: Regex pattern for allowed characters

    Returns:
        Validated API key or None

    Raises:
        ValueError: If API key format is invalid
    """
    if value is None:
        return value
    value = value.strip()
    if not value:
        return None
    try:
        value.encode("ascii")
    except UnicodeEncodeError as err:
        raise ValueError(
            f"{service_name} API key contains non-ASCII characters"
        ) from err
    if not value.startswith(prefix):
        raise ValueError(f"{service_name} API key must start with '{prefix}'")

    # Special handling for OpenAI test keys - check characters first, then length
    if service_name == "OpenAI" and value.startswith("sk-test"):
        # Test keys still need to follow character rules
        if not re.match(r"^sk-test[A-Za-z0-9-]+$", value):
            raise ValueError(f"{service_name} test API key contains invalid characters")
        # Relaxed length requirements for test keys
        if len(value) < 8:  # Minimum reasonable test key length
            raise ValueError(f"{service_name} test API key appears to be too short")
        return value

    if len(value) < min_length:
        raise ValueError(f"{service_name} API key appears to be too short")
    if len(value) > max_length:
        raise ValueError(f"{service_name} API key appears to be too long")

    if not re.match(f"^{re.escape(prefix)}{allowed_chars}$", value):
        raise ValueError(f"{service_name} API key contains invalid characters")
    return value


def validate_url_format(value: str) -> str:
    """Validate URL format.

    Args:
        value: URL to validate

    Returns:
        Validated URL with trailing slash removed

    Raises:
        ValueError: If URL format is invalid
    """
    if not value.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return value.rstrip("/")


def validate_chunk_sizes(
    chunk_size: int, chunk_overlap: int, min_chunk_size: int, max_chunk_size: int
) -> None:
    """Validate chunk size relationships.

    Args:
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum allowed chunk size
        max_chunk_size: Maximum allowed chunk size

    Raises:
        ValueError: If chunk size relationships are invalid
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    if min_chunk_size >= max_chunk_size:
        raise ValueError("min_chunk_size must be less than max_chunk_size")
    if chunk_size > max_chunk_size:
        raise ValueError("chunk_size cannot exceed max_chunk_size")


def validate_rate_limit_config(
    value: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Validate rate limit configuration structure.

    Args:
        value: Rate limit configuration dictionary

    Returns:
        Validated rate limit configuration

    Raises:
        ValueError: If rate limit structure is invalid
    """
    for provider, limits in value.items():
        if not isinstance(limits, dict):
            raise ValueError(
                f"Rate limits for provider '{provider}' must be a dictionary"
            )
        required_keys = {"max_calls", "time_window"}
        if not required_keys.issubset(limits.keys()):
            raise ValueError(
                f"Rate limits for provider '{provider}' must contain keys: {required_keys}, got: {set(limits.keys())}"
            )
        if limits["max_calls"] <= 0:
            raise ValueError(f"max_calls for provider '{provider}' must be positive")
        if limits["time_window"] <= 0:
            raise ValueError(f"time_window for provider '{provider}' must be positive")
    return value


def validate_scoring_weights(
    quality_weight: float, speed_weight: float, cost_weight: float
) -> None:
    """Validate that scoring weights sum to approximately 1.0.

    Args:
        quality_weight: Quality scoring weight
        speed_weight: Speed scoring weight
        cost_weight: Cost scoring weight

    Raises:
        ValueError: If weights don't sum to 1.0
    """
    total = quality_weight + speed_weight + cost_weight
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


def validate_model_benchmark_consistency(key: str, model_name: str) -> str:
    """Validate model benchmark key consistency.

    Args:
        key: Dictionary key
        model_name: Model name from benchmark

    Returns:
        Validated key

    Raises:
        ValueError: If key doesn't match model name
    """
    if key != model_name:
        raise ValueError(
            f"Dictionary key '{key}' does not match ModelBenchmark.model_name '{model_name}'. Keys must be consistent for proper model identification."
        )
    return key


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
            # Allow test keys for testing
            if not key.startswith("sk-test") and len(key) < 40:
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
    def validate_env_var_value(
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
    def validate_config_connections(config) -> dict[str, dict[str, Any]]:
        """Validate connections to external services.

        Returns:
            Dictionary mapping service names to their connection status
        """
        from ..utils.health_checks import ServiceHealthChecker

        # Use centralized health checker and convert format for backwards compatibility
        health_results = ServiceHealthChecker.perform_all_health_checks(config)

        # Convert to the expected format for validator
        results = {}
        for service_name, health_result in health_results.items():
            results[service_name] = {
                "connected": health_result["connected"],
                "error": health_result["error"],
            }

        return results

    @staticmethod
    def generate_validation_report(config) -> str:
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
