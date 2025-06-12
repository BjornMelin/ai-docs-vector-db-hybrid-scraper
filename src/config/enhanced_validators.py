"""Enhanced validation system for configuration management.

This module provides advanced validation capabilities beyond basic Pydantic validation,
including suggestion generation, business rule validation, and environment-specific checks.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import ValidationError

from .utils import ValidationHelper


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a single validation issue with context and suggestions."""

    field_path: str
    message: str
    severity: ValidationSeverity
    category: str
    suggestion: str | None = None
    fix_command: str | None = None

    def __str__(self) -> str:
        """String representation of validation issue."""
        severity_symbol = {
            ValidationSeverity.ERROR: "❌",
            ValidationSeverity.WARNING: "⚠️",
            ValidationSeverity.INFO: "i",
        }

        result = f"{severity_symbol[self.severity]} {self.field_path}: {self.message}"
        if self.suggestion:
            result += f"\n   💡 Suggestion: {self.suggestion}"
        if self.fix_command:
            result += f"\n   🔧 Fix: {self.fix_command}"
        return result


@dataclass
class ValidationReport:
    """Comprehensive validation report with issues and summary."""

    issues: list[ValidationIssue]
    is_valid: bool
    config_hash: str
    environment: str | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [
            issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        ]

    @property
    def info(self) -> list[ValidationIssue]:
        """Get only info-level issues."""
        return [
            issue for issue in self.issues if issue.severity == ValidationSeverity.INFO
        ]

    def summary(self) -> str:
        """Generate a summary of validation results."""
        total = len(self.issues)
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        info_count = len(self.info)

        if self.is_valid:
            if total == 0:
                return "✅ Configuration is valid with no issues"
            else:
                return f"✅ Configuration is valid with {warning_count} warnings and {info_count} info items"
        else:
            return f"❌ Configuration has {error_count} errors, {warning_count} warnings, and {info_count} info items"


class ConfigurationValidator:
    """Advanced configuration validator with suggestion engine."""

    def __init__(self, environment: str | None = None):
        """Initialize validator for specific environment.

        Args:
            environment: Target environment for validation
        """
        self.environment = environment or "development"
        self.helper = ValidationHelper()

    def validate_configuration(self, config: Any) -> ValidationReport:
        """Perform comprehensive validation of configuration.

        Args:
            config: Configuration object to validate

        Returns:
            ValidationReport with detailed issues and suggestions
        """
        from .utils import ConfigVersioning

        issues = []

        # Convert config to dict for hashing
        if hasattr(config, "model_dump"):
            config_dict = config.model_dump()
            config_hash = ConfigVersioning.generate_config_hash(config_dict)
        else:
            config_dict = config if isinstance(config, dict) else {}
            config_hash = ConfigVersioning.generate_config_hash(config_dict)

        # Perform Pydantic validation first
        try:
            if hasattr(config, "model_validate"):
                # Re-validate to catch any issues
                config.__class__.model_validate(config_dict)
        except ValidationError as e:
            issues.extend(self._process_pydantic_errors(e))

        # Business rule validation
        issues.extend(self._validate_business_rules(config))

        # Environment-specific validation
        if self.environment:
            issues.extend(
                self._validate_environment_constraints(config, self.environment)
            )

        # Provider compatibility validation
        issues.extend(self._validate_provider_compatibility(config))

        # Performance and resource validation
        issues.extend(self._validate_performance_settings(config))

        # Security validation
        issues.extend(self._validate_security_settings(config))

        # Determine overall validity (only errors make config invalid)
        is_valid = not any(
            issue.severity == ValidationSeverity.ERROR for issue in issues
        )

        return ValidationReport(
            issues=issues,
            is_valid=is_valid,
            config_hash=config_hash,
            environment=self.environment,
        )

    def _process_pydantic_errors(self, error: ValidationError) -> list[ValidationIssue]:
        """Process Pydantic validation errors into structured issues.

        Args:
            error: Pydantic ValidationError

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        for pydantic_error in error.errors():
            field_path = ".".join(str(loc) for loc in pydantic_error["loc"])
            message = pydantic_error["msg"]

            # Generate suggestion
            suggestion = self.helper.suggest_fix_for_error(message, field_path)

            # Categorize error
            category = self.helper.categorize_validation_error(message)

            # Generate fix command if applicable
            fix_command = self._generate_fix_command(field_path, message)

            issues.append(
                ValidationIssue(
                    field_path=field_path,
                    message=message,
                    severity=ValidationSeverity.ERROR,
                    category=category,
                    suggestion=suggestion,
                    fix_command=fix_command,
                )
            )

        return issues

    def _validate_business_rules(self, config: Any) -> list[ValidationIssue]:
        """Validate business logic rules.

        This method performs comprehensive validation of business rules including:
        - Provider API key requirements
        - Cache configuration consistency
        - Database connection pool settings
        - URL format validation
        - Performance configuration limits

        Args:
            config: Configuration to validate (either object or dictionary)

        Returns:
            List of validation issues found

        Raises:
            None: All exceptions are caught and converted to validation issues
        """
        issues = []
        config_dict, config_obj = self._normalize_config_input(config)

        if not config_dict:
            return self._create_empty_config_issue()

        # Validate core business rules in separate methods
        issues.extend(self._validate_provider_api_keys(config_dict, config_obj))
        issues.extend(self._validate_cache_consistency(config_dict, config_obj))
        issues.extend(self._validate_database_settings(config_dict, config_obj))
        issues.extend(self._validate_url_formats(config_dict))
        issues.extend(self._validate_port_ranges(config_dict))
        issues.extend(self._validate_performance_limits(config_dict, config_obj))

        return issues

    def _normalize_config_input(self, config: Any) -> tuple[dict[str, Any], Any]:
        """Normalize configuration input to dict and object formats.

        Args:
            config: Configuration in various formats

        Returns:
            Tuple of (config_dict, config_obj)
        """
        if hasattr(config, "model_dump"):
            return config.model_dump(), config
        elif isinstance(config, dict):
            return config, None
        else:
            return {}, None

    def _create_empty_config_issue(self) -> list[ValidationIssue]:
        """Create validation issue for empty configuration.

        Returns:
            List containing single validation issue for empty config
        """
        return [
            ValidationIssue(
                field_path="environment",
                message="Configuration is empty - missing required environment setting",
                severity=ValidationSeverity.ERROR,
                category="missing_required",
                suggestion="Add basic configuration settings like environment, debug, log_level",
            )
        ]

    def _validate_provider_api_keys(
        self, config_dict: dict[str, Any], config_obj: Any
    ) -> list[ValidationIssue]:
        """Validate provider API key requirements.

        Args:
            config_dict: Configuration as dictionary
            config_obj: Configuration as object (may be None)

        Returns:
            List of validation issues
        """
        issues = []

        # Get embedding provider
        embedding_provider = self._get_embedding_provider(config_dict, config_obj)
        if embedding_provider == "openai":
            openai_config = self._get_openai_config(config_dict, config_obj)
            api_key = self._extract_api_key(openai_config)

            if not api_key:
                issues.append(
                    ValidationIssue(
                        field_path="openai.api_key",
                        message="OpenAI API key required when using OpenAI embedding provider",
                        severity=ValidationSeverity.ERROR,
                        category="missing_required",
                        suggestion="Set OPENAI_API_KEY environment variable or add to config file",
                        fix_command="export AI_DOCS__OPENAI__API_KEY=sk-your_actual_openai_api_key_here",
                    )
                )

        return issues

    def _validate_cache_consistency(
        self, config_dict: dict[str, Any], config_obj: Any
    ) -> list[ValidationIssue]:
        """Validate cache configuration consistency.

        Args:
            config_dict: Configuration as dictionary
            config_obj: Configuration as object (may be None)

        Returns:
            List of validation issues
        """
        issues = []
        cache_config = self._get_cache_config(config_dict, config_obj)

        if cache_config:
            enable_dragonfly, dragonfly_url = self._get_dragonfly_settings(cache_config)

            if enable_dragonfly and not dragonfly_url:
                issues.append(
                    ValidationIssue(
                        field_path="cache.dragonfly_url",
                        message="DragonflyDB URL required when DragonflyDB cache is enabled",
                        severity=ValidationSeverity.ERROR,
                        category="missing_required",
                        suggestion="Provide Redis-compatible DragonflyDB connection URL",
                    )
                )

        return issues

    def _validate_database_settings(
        self, config_dict: dict[str, Any], config_obj: Any
    ) -> list[ValidationIssue]:
        """Validate database connection pool settings.

        Args:
            config_dict: Configuration as dictionary
            config_obj: Configuration as object (may be None)

        Returns:
            List of validation issues
        """
        issues = []
        db_config = self._get_database_config(config_dict, config_obj)

        if db_config:
            pool_size, max_overflow = self._get_pool_settings(db_config)

            if pool_size and max_overflow:
                total_connections = pool_size + max_overflow
                if total_connections > 100:
                    issues.append(
                        ValidationIssue(
                            field_path="database.pool_size",
                            message=f"Total database connections ({total_connections}) may be excessive",
                            severity=ValidationSeverity.WARNING,
                            category="performance",
                            suggestion="Consider reducing pool_size or max_overflow for better resource usage",
                        )
                    )

        return issues

    def _validate_url_formats(
        self, config_dict: dict[str, Any]
    ) -> list[ValidationIssue]:
        """Validate URL format for common configuration fields.

        Args:
            config_dict: Configuration as dictionary

        Returns:
            List of validation issues
        """
        issues = []
        url_fields = [
            ("qdrant.url", ["qdrant", "url"]),
            ("cache.dragonfly_url", ["cache", "dragonfly_url"]),
        ]

        for field_path, keys in url_fields:
            value = self._get_nested_value(config_dict, keys)
            if (
                value
                and isinstance(value, str)
                and not self.helper.validate_url_format(value)
            ):
                issues.append(
                    ValidationIssue(
                        field_path=field_path,
                        message="Invalid URL format",
                        severity=ValidationSeverity.ERROR,
                        category="format_error",
                        suggestion="Ensure URL includes protocol (http:// or https://)",
                    )
                )

        return issues

    def _validate_port_ranges(
        self, config_dict: dict[str, Any]
    ) -> list[ValidationIssue]:
        """Validate port number ranges for common configuration fields.

        Args:
            config_dict: Configuration as dictionary

        Returns:
            List of validation issues
        """
        issues = []
        port_fields = [
            ("qdrant.port", ["qdrant", "port"]),
            ("database.port", ["database", "port"]),
        ]

        for field_path, keys in port_fields:
            value = self._get_nested_value(config_dict, keys)
            if (
                value is not None
                and isinstance(value, int)
                and not (1 <= value <= 65535)
            ):
                issues.append(
                    ValidationIssue(
                        field_path=field_path,
                        message="Port must be between 1 and 65535",
                        severity=ValidationSeverity.ERROR,
                        category="value_range",
                        suggestion="Use a valid port number between 1 and 65535",
                    )
                )

        return issues

    def _validate_performance_limits(
        self, config_dict: dict[str, Any], config_obj: Any
    ) -> list[ValidationIssue]:
        """Validate performance configuration limits.

        Args:
            config_dict: Configuration as dictionary
            config_obj: Configuration as object (may be None)

        Returns:
            List of validation issues
        """
        issues = []
        perf_config = self._get_performance_config(config_dict, config_obj)

        if perf_config:
            max_requests, timeout = self._get_performance_settings(perf_config)

            if max_requests and max_requests > 100:
                issues.append(
                    ValidationIssue(
                        field_path="performance.max_concurrent_requests",
                        message="High concurrency setting may impact performance",
                        severity=ValidationSeverity.WARNING,
                        category="performance",
                        suggestion="Consider reducing max_concurrent_requests for better stability",
                    )
                )

            if timeout is not None and timeout < 1.0:
                issues.append(
                    ValidationIssue(
                        field_path="performance.request_timeout",
                        message="Very low timeout may cause request failures",
                        severity=ValidationSeverity.WARNING,
                        category="performance",
                        suggestion="Consider increasing request_timeout to at least 1 second",
                    )
                )

        return issues

    # Helper methods for configuration extraction
    def _get_embedding_provider(
        self, config_dict: dict[str, Any], config_obj: Any
    ) -> str | None:
        """Extract embedding provider from configuration."""
        if config_obj and hasattr(config_obj, "embedding_provider"):
            return config_obj.embedding_provider.value
        return config_dict.get("embedding_provider")

    def _get_openai_config(self, config_dict: dict[str, Any], config_obj: Any) -> Any:
        """Extract OpenAI configuration from configuration."""
        if config_obj and hasattr(config_obj, "openai"):
            return config_obj.openai
        return config_dict.get("openai")

    def _extract_api_key(self, config: Any) -> str | None:
        """Extract API key from configuration object or dictionary."""
        if not config:
            return None
        if hasattr(config, "api_key"):
            return config.api_key
        elif isinstance(config, dict):
            return config.get("api_key")
        return None

    def _get_cache_config(self, config_dict: dict[str, Any], config_obj: Any) -> Any:
        """Extract cache configuration from configuration."""
        if config_obj and hasattr(config_obj, "cache"):
            return config_obj.cache
        return config_dict.get("cache")

    def _get_dragonfly_settings(self, cache_config: Any) -> tuple[bool, str | None]:
        """Extract DragonflyDB settings from cache configuration."""
        if hasattr(cache_config, "enable_dragonfly_cache"):
            return cache_config.enable_dragonfly_cache, cache_config.dragonfly_url
        elif isinstance(cache_config, dict):
            return (
                cache_config.get("enable_dragonfly_cache", False),
                cache_config.get("dragonfly_url"),
            )
        return False, None

    def _get_database_config(self, config_dict: dict[str, Any], config_obj: Any) -> Any:
        """Extract database configuration from configuration."""
        if config_obj and hasattr(config_obj, "database"):
            return config_obj.database
        return config_dict.get("database")

    def _get_pool_settings(self, db_config: Any) -> tuple[int | None, int | None]:
        """Extract database pool settings from database configuration."""
        if hasattr(db_config, "pool_size"):
            return db_config.pool_size, getattr(db_config, "max_overflow", 0)
        elif isinstance(db_config, dict):
            return db_config.get("pool_size", 0), db_config.get("max_overflow", 0)
        return None, None

    def _get_performance_config(
        self, config_dict: dict[str, Any], config_obj: Any
    ) -> Any:
        """Extract performance configuration from configuration."""
        if config_obj and hasattr(config_obj, "performance"):
            return config_obj.performance
        return config_dict.get("performance")

    def _get_performance_settings(
        self, perf_config: Any
    ) -> tuple[int | None, float | None]:
        """Extract performance settings from performance configuration."""
        if hasattr(perf_config, "max_concurrent_requests"):
            return (
                perf_config.max_concurrent_requests,
                getattr(perf_config, "request_timeout", None),
            )
        elif isinstance(perf_config, dict):
            return (
                perf_config.get("max_concurrent_requests"),
                perf_config.get("request_timeout"),
            )
        return None, None

    def _get_nested_value(self, config_dict: dict[str, Any], keys: list[str]) -> Any:
        """Get nested value from configuration dictionary.

        Args:
            config_dict: Configuration dictionary
            keys: List of keys for nested access

        Returns:
            Value if found, None otherwise
        """
        value = config_dict
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _validate_environment_constraints(
        self, config: Any, environment: str
    ) -> list[ValidationIssue]:
        """Validate environment-specific constraints.

        Args:
            config: Configuration to validate
            environment: Target environment

        Returns:
            List of validation issues
        """
        issues = []

        if environment == "production":
            # Production-specific validations
            if hasattr(config, "debug") and config.debug:
                issues.append(
                    ValidationIssue(
                        field_path="debug",
                        message="Debug mode should be disabled in production",
                        severity=ValidationSeverity.ERROR,
                        category="environment_constraint",
                        suggestion="Set debug=false for production deployment",
                        fix_command="export AI_DOCS__DEBUG=false",
                    )
                )

            if (
                hasattr(config, "log_level")
                and str(config.log_level).upper() == "DEBUG"
            ):
                issues.append(
                    ValidationIssue(
                        field_path="log_level",
                        message="DEBUG log level not recommended for production",
                        severity=ValidationSeverity.WARNING,
                        category="environment_constraint",
                        suggestion="Use INFO or WARNING log level in production",
                    )
                )

            if hasattr(config, "security") and not config.security.require_api_keys:
                issues.append(
                    ValidationIssue(
                        field_path="security.require_api_keys",
                        message="API key authentication should be required in production",
                        severity=ValidationSeverity.ERROR,
                        category="environment_constraint",
                        suggestion="Enable API key requirement for production security",
                    )
                )

        elif environment == "development":
            # Development-specific suggestions
            if (
                hasattr(config, "performance")
                and config.performance.max_concurrent_requests > 50
            ):
                issues.append(
                    ValidationIssue(
                        field_path="performance.max_concurrent_requests",
                        message="High concurrency setting in development may consume excessive resources",
                        severity=ValidationSeverity.INFO,
                        category="development_optimization",
                        suggestion="Consider reducing for development to save system resources",
                    )
                )

        return issues

    def _validate_provider_compatibility(self, config: Any) -> list[ValidationIssue]:
        """Validate provider compatibility and configuration consistency.

        Args:
            config: Configuration to validate

        Returns:
            List of validation issues
        """
        issues = []

        # Check if selected providers have required configuration
        if (
            hasattr(config, "crawl_provider")
            and hasattr(config, "firecrawl")
            and config.crawl_provider.value == "firecrawl"
            and not config.firecrawl.api_key
        ):
            issues.append(
                ValidationIssue(
                    field_path="firecrawl.api_key",
                    message="Firecrawl API key required when using Firecrawl provider",
                    severity=ValidationSeverity.ERROR,
                    category="missing_required",
                    suggestion="Set Firecrawl API key or switch to Crawl4AI provider",
                )
            )

        return issues

    def _validate_performance_settings(self, config: Any) -> list[ValidationIssue]:
        """Validate performance-related settings.

        Args:
            config: Configuration to validate

        Returns:
            List of validation issues
        """
        issues = []

        if hasattr(config, "performance"):
            perf = config.performance

            # Check memory settings
            if hasattr(perf, "max_memory_usage_mb") and perf.max_memory_usage_mb < 512:
                issues.append(
                    ValidationIssue(
                        field_path="performance.max_memory_usage_mb",
                        message="Low memory limit may cause performance issues",
                        severity=ValidationSeverity.WARNING,
                        category="performance",
                        suggestion="Consider increasing memory limit to at least 1GB for optimal performance",
                    )
                )

        return issues

    def _validate_security_settings(self, config: Any) -> list[ValidationIssue]:
        """Validate security-related settings.

        Args:
            config: Configuration to validate

        Returns:
            List of validation issues
        """
        issues = []

        if hasattr(config, "security"):
            security = config.security

            # Check API key requirement
            if not security.require_api_keys:
                issues.append(
                    ValidationIssue(
                        field_path="security.require_api_keys",
                        message="API key authentication disabled - consider security implications",
                        severity=ValidationSeverity.WARNING,
                        category="security",
                        suggestion="Enable API key requirement for better security",
                    )
                )

            # Check rate limiting
            if not security.enable_rate_limiting:
                issues.append(
                    ValidationIssue(
                        field_path="security.enable_rate_limiting",
                        message="Rate limiting disabled - may allow abuse",
                        severity=ValidationSeverity.WARNING,
                        category="security",
                        suggestion="Enable rate limiting to prevent API abuse",
                    )
                )

        return issues

    def _generate_fix_command(self, field_path: str, message: str) -> str | None:
        """Generate command to fix validation issue.

        Args:
            field_path: Path to the field with issue
            message: Validation error message

        Returns:
            Fix command or None if not applicable
        """
        if "api_key" in field_path.lower() and "required" in message.lower():
            env_var = f"AI_DOCS__{field_path.replace('.', '__').upper()}"
            return f"export {env_var}=your_actual_api_key_here"

        if field_path == "debug" and "production" in message.lower():
            return "export AI_DOCS__DEBUG=false"

        if "log_level" in field_path and "production" in message.lower():
            return "export AI_DOCS__LOG_LEVEL=INFO"

        return None

    def _generate_suggestions(
        self, issue: ValidationIssue, config_data: dict
    ) -> list[str]:
        """Generate suggestions for a validation issue.

        Args:
            issue: The validation issue
            config_data: Configuration data for context

        Returns:
            List of suggestion strings
        """
        suggestions = []

        if "api_key" in issue.field_path.lower():
            service = issue.field_path.split(".")[0]
            suggestions.append(f"Set {service.upper()}_API_KEY environment variable")
            suggestions.append(f"Add {service} API key to your configuration file")

        elif "url" in issue.field_path.lower():
            if "localhost" in str(config_data.get(issue.field_path.split(".")[0], {})):
                suggestions.append("Add http:// or https:// protocol to the URL")
                suggestions.append("Example: http://localhost:6333")
            else:
                suggestions.append("Ensure URL includes protocol (http:// or https://)")

        elif "debug" in issue.field_path.lower():
            suggestions.append("Set debug=false for production environment")
            suggestions.append("Use environment variables: AI_DOCS__DEBUG=false")

        elif "port" in issue.field_path.lower():
            if "qdrant" in issue.field_path.lower():
                suggestions.append("Use default Qdrant port: 6333")
            suggestions.append("Choose a port between 1 and 65535")

        elif "performance" in issue.field_path.lower():
            if (
                "concurrent" in issue.field_path.lower()
                or "high" in issue.message.lower()
            ):
                suggestions.append(
                    "Reduce max_concurrent_requests for better stability"
                )
                suggestions.append("Consider values between 10-50 for production")
            elif "timeout" in issue.field_path.lower():
                suggestions.append("Increase timeout to at least 1 second")

        # Generic suggestions for unknown issues
        if not suggestions:
            suggestions.append("Check the documentation for this configuration option")
            suggestions.append(
                "Verify the value meets the expected format and constraints"
            )

        return suggestions
