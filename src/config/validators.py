"""
Configuration Validators

This module provides comprehensive validation for configuration schemas
and deployment environments, supporting the GitOps configuration management workflow.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a configuration validation error"""

    path: str
    message: str
    severity: str = "error"  # error, warning, info


class ConfigurationValidator:
    """
    Comprehensive configuration validator for GitOps deployments

    Validates configuration schemas, environment-specific requirements,
    security compliance, and deployment readiness.
    """

    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def validate_config_schema(
        self, config: Dict[str, Any], environment: str = "development"
    ) -> bool:
        """
        Validate configuration schema and environment-specific requirements

        Args:
            config: Configuration dictionary to validate
            environment: Target environment (development, staging, production)

        Returns:
            bool: True if validation passes, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()

        # Basic schema validation
        self._validate_required_sections(config)
        self._validate_environment_config(config, environment)
        self._validate_cache_config(config.get("cache", {}))
        self._validate_qdrant_config(config.get("qdrant", {}))
        self._validate_openai_config(config.get("openai", {}))
        self._validate_performance_config(config.get("performance", {}))
        self._validate_security_config(config.get("security", {}), environment)
        self._validate_crawl4ai_config(config.get("crawl4ai", {}))
        self._validate_chunking_config(config.get("chunking", {}))

        # Environment-specific validations
        if environment == "production":
            self._validate_production_requirements(config)
        elif environment == "staging":
            self._validate_staging_requirements(config)

        return len(self.errors) == 0

    def _validate_required_sections(self, config: Dict[str, Any]) -> None:
        """Validate that all required configuration sections are present"""
        required_sections = [
            "environment",
            "cache",
            "qdrant",
            "performance",
            "security",
        ]

        for section in required_sections:
            if section not in config:
                self.errors.append(
                    ValidationError(
                        path=section,
                        message=f"Required configuration section '{section}' is missing",
                    )
                )

    def _validate_environment_config(
        self, config: Dict[str, Any], environment: str
    ) -> None:
        """Validate environment-specific configuration"""
        config_env = config.get("environment")

        if config_env != environment:
            self.errors.append(
                ValidationError(
                    path="environment",
                    message=f"Environment mismatch: expected '{environment}', got '{config_env}'",
                )
            )

        # Validate debug settings
        debug = config.get("debug")
        if debug is not None and not isinstance(debug, bool):
            self.errors.append(
                ValidationError(path="debug", message="Debug setting must be a boolean")
            )

        # Validate log level
        log_level = config.get("log_level")
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level and log_level not in valid_log_levels:
            self.errors.append(
                ValidationError(
                    path="log_level",
                    message=f"Invalid log level '{log_level}'. Must be one of {valid_log_levels}",
                )
            )

    def _validate_cache_config(self, cache_config: Dict[str, Any]) -> None:
        """Validate cache configuration"""
        if not cache_config:
            self.errors.append(
                ValidationError(path="cache", message="Cache configuration is required")
            )
            return

        # Validate boolean settings
        boolean_settings = [
            "enable_caching",
            "enable_local_cache",
            "enable_dragonfly_cache",
        ]
        for setting in boolean_settings:
            value = cache_config.get(setting)
            if value is not None and not isinstance(value, bool):
                self.errors.append(
                    ValidationError(
                        path=f"cache.{setting}",
                        message=f"Cache setting '{setting}' must be a boolean",
                    )
                )

        # Validate TTL values
        ttl_settings = ["ttl_embeddings", "ttl_crawl", "ttl_queries"]
        for setting in ttl_settings:
            value = cache_config.get(setting)
            if value is not None:
                if not isinstance(value, int) or value <= 0:
                    self.errors.append(
                        ValidationError(
                            path=f"cache.{setting}",
                            message=f"Cache TTL '{setting}' must be a positive integer",
                        )
                    )

        # Validate memory settings
        max_memory = cache_config.get("local_max_memory_mb")
        if max_memory is not None:
            if not isinstance(max_memory, int | float) or max_memory <= 0:
                self.errors.append(
                    ValidationError(
                        path="cache.local_max_memory_mb",
                        message="Cache max memory must be a positive number",
                    )
                )

        # Validate pool size
        pool_size = cache_config.get("redis_pool_size")
        if pool_size is not None:
            if not isinstance(pool_size, int) or pool_size <= 0:
                self.errors.append(
                    ValidationError(
                        path="cache.redis_pool_size",
                        message="Redis pool size must be a positive integer",
                    )
                )

    def _validate_qdrant_config(self, qdrant_config: Dict[str, Any]) -> None:
        """Validate Qdrant configuration"""
        if not qdrant_config:
            self.errors.append(
                ValidationError(
                    path="qdrant", message="Qdrant configuration is required"
                )
            )
            return

        # Validate URL
        url = qdrant_config.get("url")
        if not url or not isinstance(url, str):
            self.errors.append(
                ValidationError(
                    path="qdrant.url",
                    message="Qdrant URL is required and must be a string",
                )
            )
        elif not (url.startswith("http://") or url.startswith("https://")):
            self.warnings.append(
                ValidationError(
                    path="qdrant.url",
                    message="Qdrant URL should use http:// or https:// protocol",
                    severity="warning",
                )
            )

        # Validate numeric settings
        numeric_settings = {
            "batch_size": (1, 10000),
            "max_retries": (0, 100),
            "hnsw_ef_construct": (1, 1000),
            "hnsw_m": (1, 100),
        }

        for setting, (min_val, max_val) in numeric_settings.items():
            value = qdrant_config.get(setting)
            if value is not None:
                if not isinstance(value, int) or value < min_val or value > max_val:
                    self.errors.append(
                        ValidationError(
                            path=f"qdrant.{setting}",
                            message=f"Qdrant {setting} must be an integer between {min_val} and {max_val}",
                        )
                    )

    def _validate_openai_config(self, openai_config: Dict[str, Any]) -> None:
        """Validate OpenAI configuration"""
        if not openai_config:
            self.warnings.append(
                ValidationError(
                    path="openai",
                    message="OpenAI configuration is missing - required if using OpenAI embeddings",
                    severity="warning",
                )
            )
            return

        # Validate API key format (but don't require actual key in config)
        api_key = openai_config.get("api_key", "")
        if api_key and not api_key.startswith("sk-"):
            if "template_placeholder" not in api_key:
                self.warnings.append(
                    ValidationError(
                        path="openai.api_key",
                        message="OpenAI API key should start with 'sk-' or be externalized",
                        severity="warning",
                    )
                )

        # Validate model
        model = openai_config.get("model")
        valid_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]
        if model and model not in valid_models:
            self.warnings.append(
                ValidationError(
                    path="openai.model",
                    message=f"OpenAI model '{model}' may not be supported. Recommended: {valid_models}",
                    severity="warning",
                )
            )

        # Validate dimensions
        dimensions = openai_config.get("dimensions")
        if dimensions is not None:
            if not isinstance(dimensions, int) or dimensions <= 0:
                self.errors.append(
                    ValidationError(
                        path="openai.dimensions",
                        message="OpenAI dimensions must be a positive integer",
                    )
                )

        # Validate batch size
        batch_size = openai_config.get("batch_size")
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > 2000:
                self.errors.append(
                    ValidationError(
                        path="openai.batch_size",
                        message="OpenAI batch size must be between 1 and 2000",
                    )
                )

        # Validate budget settings
        budget_limit = openai_config.get("budget_limit")
        if budget_limit is not None:
            if not isinstance(budget_limit, int | float) or budget_limit < 0:
                self.errors.append(
                    ValidationError(
                        path="openai.budget_limit",
                        message="OpenAI budget limit must be a non-negative number",
                    )
                )

    def _validate_performance_config(self, performance_config: Dict[str, Any]) -> None:
        """Validate performance configuration"""
        if not performance_config:
            self.warnings.append(
                ValidationError(
                    path="performance",
                    message="Performance configuration is missing",
                    severity="warning",
                )
            )
            return

        # Validate concurrent requests
        max_concurrent = performance_config.get("max_concurrent_requests")
        if max_concurrent is not None:
            if not isinstance(max_concurrent, int) or max_concurrent <= 0:
                self.errors.append(
                    ValidationError(
                        path="performance.max_concurrent_requests",
                        message="Max concurrent requests must be a positive integer",
                    )
                )

        # Validate timeout settings
        timeout_settings = ["request_timeout", "retry_base_delay", "retry_max_delay"]
        for setting in timeout_settings:
            value = performance_config.get(setting)
            if value is not None:
                if not isinstance(value, int | float) or value <= 0:
                    self.errors.append(
                        ValidationError(
                            path=f"performance.{setting}",
                            message=f"Performance {setting} must be a positive number",
                        )
                    )

        # Validate retry settings
        max_retries = performance_config.get("max_retries")
        if max_retries is not None:
            if not isinstance(max_retries, int) or max_retries < 0:
                self.errors.append(
                    ValidationError(
                        path="performance.max_retries",
                        message="Max retries must be a non-negative integer",
                    )
                )

        # Validate memory settings
        max_memory = performance_config.get("max_memory_usage_mb")
        if max_memory is not None:
            if not isinstance(max_memory, int | float) or max_memory <= 0:
                self.errors.append(
                    ValidationError(
                        path="performance.max_memory_usage_mb",
                        message="Max memory usage must be a positive number",
                    )
                )

    def _validate_security_config(
        self, security_config: Dict[str, Any], environment: str
    ) -> None:
        """Validate security configuration"""
        if not security_config:
            self.errors.append(
                ValidationError(
                    path="security", message="Security configuration is required"
                )
            )
            return

        # Validate boolean settings
        boolean_settings = ["require_api_keys", "enable_rate_limiting"]
        for setting in boolean_settings:
            value = security_config.get(setting)
            if value is not None and not isinstance(value, bool):
                self.errors.append(
                    ValidationError(
                        path=f"security.{setting}",
                        message=f"Security setting '{setting}' must be a boolean",
                    )
                )

        # Validate rate limiting
        rate_limit = security_config.get("rate_limit_requests")
        if rate_limit is not None:
            if not isinstance(rate_limit, int) or rate_limit <= 0:
                self.errors.append(
                    ValidationError(
                        path="security.rate_limit_requests",
                        message="Rate limit requests must be a positive integer",
                    )
                )

        # Environment-specific security validations
        if environment == "production":
            if not security_config.get("require_api_keys", False):
                self.errors.append(
                    ValidationError(
                        path="security.require_api_keys",
                        message="API key requirement must be enabled in production",
                    )
                )

            if not security_config.get("enable_rate_limiting", False):
                self.warnings.append(
                    ValidationError(
                        path="security.enable_rate_limiting",
                        message="Rate limiting should be enabled in production",
                        severity="warning",
                    )
                )

    def _validate_crawl4ai_config(self, crawl4ai_config: Dict[str, Any]) -> None:
        """Validate Crawl4AI configuration"""
        if not crawl4ai_config:
            return  # Optional configuration

        # Validate boolean settings
        boolean_settings = ["headless", "remove_scripts", "remove_styles"]
        for setting in boolean_settings:
            value = crawl4ai_config.get(setting)
            if value is not None and not isinstance(value, bool):
                self.errors.append(
                    ValidationError(
                        path=f"crawl4ai.{setting}",
                        message=f"Crawl4AI setting '{setting}' must be a boolean",
                    )
                )

        # Validate numeric settings
        max_crawls = crawl4ai_config.get("max_concurrent_crawls")
        if max_crawls is not None:
            if not isinstance(max_crawls, int) or max_crawls <= 0:
                self.errors.append(
                    ValidationError(
                        path="crawl4ai.max_concurrent_crawls",
                        message="Max concurrent crawls must be a positive integer",
                    )
                )

        page_timeout = crawl4ai_config.get("page_timeout")
        if page_timeout is not None:
            if not isinstance(page_timeout, int | float) or page_timeout <= 0:
                self.errors.append(
                    ValidationError(
                        path="crawl4ai.page_timeout",
                        message="Page timeout must be a positive number",
                    )
                )

    def _validate_chunking_config(self, chunking_config: Dict[str, Any]) -> None:
        """Validate chunking configuration"""
        if not chunking_config:
            return  # Optional configuration

        # Validate strategy
        strategy = chunking_config.get("strategy")
        valid_strategies = ["simple", "enhanced", "recursive"]
        if strategy and strategy not in valid_strategies:
            self.warnings.append(
                ValidationError(
                    path="chunking.strategy",
                    message=f"Chunking strategy '{strategy}' may not be supported. Valid: {valid_strategies}",
                    severity="warning",
                )
            )

        # Validate chunk size
        chunk_size = chunking_config.get("chunk_size")
        if chunk_size is not None:
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                self.errors.append(
                    ValidationError(
                        path="chunking.chunk_size",
                        message="Chunk size must be a positive integer",
                    )
                )

        # Validate chunk overlap
        chunk_overlap = chunking_config.get("chunk_overlap")
        if chunk_overlap is not None:
            if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                self.errors.append(
                    ValidationError(
                        path="chunking.chunk_overlap",
                        message="Chunk overlap must be a non-negative integer",
                    )
                )

            # Check overlap vs size ratio
            if chunk_size and chunk_overlap >= chunk_size:
                self.warnings.append(
                    ValidationError(
                        path="chunking.chunk_overlap",
                        message="Chunk overlap should be less than chunk size",
                        severity="warning",
                    )
                )

    def _validate_production_requirements(self, config: Dict[str, Any]) -> None:
        """Validate production-specific requirements"""
        # Debug mode should be disabled
        if config.get("debug", False):
            self.errors.append(
                ValidationError(
                    path="debug", message="Debug mode must be disabled in production"
                )
            )

        # Log level should not be DEBUG
        if config.get("log_level") == "DEBUG":
            self.warnings.append(
                ValidationError(
                    path="log_level",
                    message="DEBUG log level not recommended for production",
                    severity="warning",
                )
            )

        # Cache should be enabled
        cache_config = config.get("cache", {})
        if not cache_config.get("enable_caching", False):
            self.warnings.append(
                ValidationError(
                    path="cache.enable_caching",
                    message="Caching should be enabled in production for better performance",
                    severity="warning",
                )
            )

        # Validate production performance settings
        performance_config = config.get("performance", {})
        max_concurrent = performance_config.get("max_concurrent_requests", 0)
        if max_concurrent < 10:
            self.warnings.append(
                ValidationError(
                    path="performance.max_concurrent_requests",
                    message="Consider increasing concurrent requests limit for production",
                    severity="warning",
                )
            )

    def _validate_staging_requirements(self, config: Dict[str, Any]) -> None:
        """Validate staging-specific requirements"""
        # Staging should have similar settings to production but allow some flexibility

        # Log level can be more verbose
        log_level = config.get("log_level")
        if log_level not in ["INFO", "DEBUG", "WARNING"]:
            self.warnings.append(
                ValidationError(
                    path="log_level",
                    message="Consider using INFO, DEBUG, or WARNING log level for staging",
                    severity="warning",
                )
            )

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results"""
        return {
            "passed": len(self.errors) == 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [{"path": e.path, "message": e.message} for e in self.errors],
            "warnings": [{"path": w.path, "message": w.message} for w in self.warnings],
        }


# Convenience functions for backward compatibility
def validate_config_schema(
    config: Dict[str, Any], environment: str = "development"
) -> bool:
    """
    Validate configuration schema

    Args:
        config: Configuration dictionary to validate
        environment: Target environment

    Returns:
        bool: True if validation passes, False otherwise
    """
    validator = ConfigurationValidator()
    return validator.validate_config_schema(config, environment)


def validate_all_configs(config_dir: Union[str, Path]) -> bool:
    """
    Validate all configuration files in a directory

    Args:
        config_dir: Path to configuration directory

    Returns:
        bool: True if all configurations are valid, False otherwise
    """
    config_dir = Path(config_dir)

    if not config_dir.exists():
        logger.error(f"Configuration directory not found: {config_dir}")
        return False

    templates_dir = config_dir / "templates"
    if not templates_dir.exists():
        logger.warning(f"Templates directory not found: {templates_dir}")
        return True

    validator = ConfigurationValidator()
    all_valid = True

    for config_file in templates_dir.glob("*.json"):
        environment = config_file.stem

        try:
            with open(config_file) as f:
                config = json.load(f)

            logger.info(f"Validating configuration for {environment}")
            is_valid = validator.validate_config_schema(config, environment)

            if not is_valid:
                logger.error(f"Validation failed for {environment}")
                summary = validator.get_validation_summary()
                for error in summary["errors"]:
                    logger.error(f"  {error['path']}: {error['message']}")
                all_valid = False
            else:
                logger.info(f"Configuration valid for {environment}")
                summary = validator.get_validation_summary()
                for warning in summary["warnings"]:
                    logger.warning(f"  {warning['path']}: {warning['message']}")

        except Exception as e:
            logger.exception(f"Failed to validate {config_file}: {e}")
            all_valid = False

    return all_valid


def validate_deployment_readiness(
    config_dir: Union[str, Path], environment: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate deployment readiness for a specific environment

    Args:
        config_dir: Path to configuration directory
        environment: Target environment

    Returns:
        Tuple[bool, Dict]: (is_ready, validation_details)
    """
    config_dir = Path(config_dir)
    config_file = config_dir / "templates" / f"{environment}.json"

    if not config_file.exists():
        return False, {
            "error": f"Configuration file not found: {config_file}",
            "passed": False,
            "error_count": 1,
            "warning_count": 0,
            "errors": [
                {"path": str(config_file), "message": "Configuration file not found"}
            ],
            "warnings": [],
        }

    try:
        with open(config_file) as f:
            config = json.load(f)

        validator = ConfigurationValidator()
        is_valid = validator.validate_config_schema(config, environment)
        summary = validator.get_validation_summary()

        return is_valid, summary

    except Exception as e:
        return False, {
            "error": f"Failed to load configuration: {e}",
            "passed": False,
            "error_count": 1,
            "warning_count": 0,
            "errors": [{"path": str(config_file), "message": str(e)}],
            "warnings": [],
        }
