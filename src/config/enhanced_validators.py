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
            ValidationSeverity.INFO: "ℹ️"
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
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]

    @property
    def info(self) -> list[ValidationIssue]:
        """Get only info-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.INFO]

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
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
            config_hash = ConfigVersioning.generate_config_hash(config_dict)
        else:
            config_dict = config if isinstance(config, dict) else {}
            config_hash = ConfigVersioning.generate_config_hash(config_dict)

        # Perform Pydantic validation first
        try:
            if hasattr(config, 'model_validate'):
                # Re-validate to catch any issues
                config.__class__.model_validate(config_dict)
        except ValidationError as e:
            issues.extend(self._process_pydantic_errors(e))

        # Business rule validation
        issues.extend(self._validate_business_rules(config))

        # Environment-specific validation
        if self.environment:
            issues.extend(self._validate_environment_constraints(config, self.environment))

        # Provider compatibility validation
        issues.extend(self._validate_provider_compatibility(config))

        # Performance and resource validation
        issues.extend(self._validate_performance_settings(config))

        # Security validation
        issues.extend(self._validate_security_settings(config))

        # Determine overall validity (only errors make config invalid)
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationReport(
            issues=issues,
            is_valid=is_valid,
            config_hash=config_hash,
            environment=self.environment
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
            field_path = '.'.join(str(loc) for loc in pydantic_error['loc'])
            message = pydantic_error['msg']

            # Generate suggestion
            suggestion = self.helper.suggest_fix_for_error(message, field_path)

            # Categorize error
            category = self.helper.categorize_validation_error(message)

            # Generate fix command if applicable
            fix_command = self._generate_fix_command(field_path, message)

            issues.append(ValidationIssue(
                field_path=field_path,
                message=message,
                severity=ValidationSeverity.ERROR,
                category=category,
                suggestion=suggestion,
                fix_command=fix_command
            ))

        return issues

    def _validate_business_rules(self, config: Any) -> list[ValidationIssue]:
        """Validate business logic rules.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues
        """
        issues = []

        # Handle both config objects and dictionaries
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
            config_obj = config
        elif isinstance(config, dict):
            config_dict = config
            config_obj = None
        else:
            return issues

        # Empty configuration validation
        if not config_dict or len(config_dict) == 0:
            issues.append(ValidationIssue(
                field_path="environment",
                message="Configuration is empty - missing required environment setting",
                severity=ValidationSeverity.ERROR,
                category="missing_required",
                suggestion="Add basic configuration settings like environment, debug, log_level"
            ))
            return issues

        # Provider API key validation
        embedding_provider = None
        if config_obj and hasattr(config_obj, 'embedding_provider'):
            embedding_provider = config_obj.embedding_provider.value
        elif "embedding_provider" in config_dict:
            embedding_provider = config_dict["embedding_provider"]
            
        openai_config = None
        if config_obj and hasattr(config_obj, 'openai'):
            openai_config = config_obj.openai
        elif "openai" in config_dict:
            openai_config = config_dict["openai"]
            
        if embedding_provider == "openai":
            api_key = None
            if openai_config:
                if hasattr(openai_config, 'api_key'):
                    api_key = openai_config.api_key
                elif isinstance(openai_config, dict):
                    api_key = openai_config.get("api_key")
            
            if not api_key:
                issues.append(ValidationIssue(
                    field_path="openai.api_key",
                    message="OpenAI API key required when using OpenAI embedding provider",
                    severity=ValidationSeverity.ERROR,
                    category="missing_required",
                    suggestion="Set OPENAI_API_KEY environment variable or add to config file",
                    fix_command="export AI_DOCS__OPENAI__API_KEY=your-api-key"
                ))

        # Cache consistency validation
        cache_config = None
        if config_obj and hasattr(config_obj, 'cache'):
            cache_config = config_obj.cache
        elif "cache" in config_dict:
            cache_config = config_dict["cache"]
            
        if cache_config:
            enable_dragonfly = False
            dragonfly_url = None
            
            if hasattr(cache_config, 'enable_dragonfly_cache'):
                enable_dragonfly = cache_config.enable_dragonfly_cache
                dragonfly_url = cache_config.dragonfly_url
            elif isinstance(cache_config, dict):
                enable_dragonfly = cache_config.get("enable_dragonfly_cache", False)
                dragonfly_url = cache_config.get("dragonfly_url")
                
            if enable_dragonfly and not dragonfly_url:
                issues.append(ValidationIssue(
                    field_path="cache.dragonfly_url",
                    message="DragonflyDB URL required when DragonflyDB cache is enabled",
                    severity=ValidationSeverity.ERROR,
                    category="missing_required",
                    suggestion="Provide Redis-compatible DragonflyDB connection URL"
                ))

        # Database pool size validation
        db_config = None
        if config_obj and hasattr(config_obj, 'database'):
            db_config = config_obj.database
        elif "database" in config_dict:
            db_config = config_dict["database"]
            
        if db_config:
            pool_size = None
            max_overflow = None
            
            if hasattr(db_config, 'pool_size'):
                pool_size = db_config.pool_size
                max_overflow = getattr(db_config, 'max_overflow', 0)
            elif isinstance(db_config, dict):
                pool_size = db_config.get("pool_size", 0)
                max_overflow = db_config.get("max_overflow", 0)
                
            if pool_size and max_overflow:
                total_connections = pool_size + max_overflow
                if total_connections > 100:
                    issues.append(ValidationIssue(
                        field_path="database.pool_size",
                        message=f"Total database connections ({total_connections}) may be excessive",
                        severity=ValidationSeverity.WARNING,
                        category="performance",
                        suggestion="Consider reducing pool_size or max_overflow for better resource usage"
                    ))

        # URL validation for common fields
        url_fields = [
            ("qdrant.url", ["qdrant", "url"]),
            ("cache.dragonfly_url", ["cache", "dragonfly_url"])
        ]
        
        for field_path, keys in url_fields:
            value = config_dict
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
                    
            if value and isinstance(value, str):
                if not self.helper.validate_url_format(value):
                    issues.append(ValidationIssue(
                        field_path=field_path,
                        message="Invalid URL format",
                        severity=ValidationSeverity.ERROR,
                        category="format_error",
                        suggestion=f"Ensure URL includes protocol (http:// or https://)"
                    ))

        # Port validation for common fields
        port_fields = [
            ("qdrant.port", ["qdrant", "port"]),
            ("database.port", ["database", "port"])
        ]
        
        for field_path, keys in port_fields:
            value = config_dict
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
                    
            if value is not None and isinstance(value, int):
                if not (1 <= value <= 65535):
                    issues.append(ValidationIssue(
                        field_path=field_path,
                        message="Port must be between 1 and 65535",
                        severity=ValidationSeverity.ERROR,
                        category="value_range",
                        suggestion="Use a valid port number between 1 and 65535"
                    ))

        # Performance settings validation
        perf_config = None
        if config_obj and hasattr(config_obj, 'performance'):
            perf_config = config_obj.performance
        elif "performance" in config_dict:
            perf_config = config_dict["performance"]
            
        if perf_config:
            max_requests = None
            timeout = None
            
            if hasattr(perf_config, 'max_concurrent_requests'):
                max_requests = perf_config.max_concurrent_requests
                timeout = getattr(perf_config, 'request_timeout', None)
            elif isinstance(perf_config, dict):
                max_requests = perf_config.get("max_concurrent_requests")
                timeout = perf_config.get("request_timeout")
                
            if max_requests and max_requests > 100:
                issues.append(ValidationIssue(
                    field_path="performance.max_concurrent_requests",
                    message="High concurrency setting may impact performance",
                    severity=ValidationSeverity.WARNING,
                    category="performance",
                    suggestion="Consider reducing max_concurrent_requests for better stability"
                ))
                
            if timeout is not None and timeout < 1.0:
                issues.append(ValidationIssue(
                    field_path="performance.request_timeout",
                    message="Very low timeout may cause request failures",
                    severity=ValidationSeverity.WARNING,
                    category="performance",
                    suggestion="Consider increasing request_timeout to at least 1 second"
                ))

        return issues

    def _validate_environment_constraints(self, config: Any, environment: str) -> list[ValidationIssue]:
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
            if hasattr(config, 'debug') and config.debug:
                issues.append(ValidationIssue(
                    field_path="debug",
                    message="Debug mode should be disabled in production",
                    severity=ValidationSeverity.ERROR,
                    category="environment_constraint",
                    suggestion="Set debug=false for production deployment",
                    fix_command="export AI_DOCS__DEBUG=false"
                ))

            if hasattr(config, 'log_level') and str(config.log_level).upper() == "DEBUG":
                issues.append(ValidationIssue(
                    field_path="log_level",
                    message="DEBUG log level not recommended for production",
                    severity=ValidationSeverity.WARNING,
                    category="environment_constraint",
                    suggestion="Use INFO or WARNING log level in production"
                ))

            if hasattr(config, 'security') and not config.security.require_api_keys:
                issues.append(ValidationIssue(
                    field_path="security.require_api_keys",
                    message="API key authentication should be required in production",
                    severity=ValidationSeverity.ERROR,
                    category="environment_constraint",
                    suggestion="Enable API key requirement for production security"
                ))

        elif environment == "development":
            # Development-specific suggestions
            if hasattr(config, 'performance') and config.performance.max_concurrent_requests > 50:
                issues.append(ValidationIssue(
                    field_path="performance.max_concurrent_requests",
                    message="High concurrency setting in development may consume excessive resources",
                    severity=ValidationSeverity.INFO,
                    category="development_optimization",
                    suggestion="Consider reducing for development to save system resources"
                ))

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
        if hasattr(config, 'crawl_provider') and hasattr(config, 'firecrawl'):
            if config.crawl_provider.value == "firecrawl" and not config.firecrawl.api_key:
                issues.append(ValidationIssue(
                    field_path="firecrawl.api_key",
                    message="Firecrawl API key required when using Firecrawl provider",
                    severity=ValidationSeverity.ERROR,
                    category="missing_required",
                    suggestion="Set Firecrawl API key or switch to Crawl4AI provider"
                ))

        return issues

    def _validate_performance_settings(self, config: Any) -> list[ValidationIssue]:
        """Validate performance-related settings.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues
        """
        issues = []

        if hasattr(config, 'performance'):
            perf = config.performance

            # Check memory settings
            if hasattr(perf, 'max_memory_usage_mb') and perf.max_memory_usage_mb < 512:
                issues.append(ValidationIssue(
                    field_path="performance.max_memory_usage_mb",
                    message="Low memory limit may cause performance issues",
                    severity=ValidationSeverity.WARNING,
                    category="performance",
                    suggestion="Consider increasing memory limit to at least 1GB for optimal performance"
                ))

        return issues

    def _validate_security_settings(self, config: Any) -> list[ValidationIssue]:
        """Validate security-related settings.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues
        """
        issues = []

        if hasattr(config, 'security'):
            security = config.security

            # Check API key requirement
            if not security.require_api_keys:
                issues.append(ValidationIssue(
                    field_path="security.require_api_keys",
                    message="API key authentication disabled - consider security implications",
                    severity=ValidationSeverity.WARNING,
                    category="security",
                    suggestion="Enable API key requirement for better security"
                ))

            # Check rate limiting
            if not security.enable_rate_limiting:
                issues.append(ValidationIssue(
                    field_path="security.enable_rate_limiting",
                    message="Rate limiting disabled - may allow abuse",
                    severity=ValidationSeverity.WARNING,
                    category="security",
                    suggestion="Enable rate limiting to prevent API abuse"
                ))

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
            return f"export {env_var}=your-api-key"

        if field_path == "debug" and "production" in message.lower():
            return "export AI_DOCS__DEBUG=false"

        if "log_level" in field_path and "production" in message.lower():
            return "export AI_DOCS__LOG_LEVEL=INFO"

        return None

    def _generate_suggestions(self, issue: ValidationIssue, config_data: dict) -> list[str]:
        """Generate suggestions for a validation issue.
        
        Args:
            issue: The validation issue
            config_data: Configuration data for context
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        if "api_key" in issue.field_path.lower():
            service = issue.field_path.split('.')[0]
            suggestions.append(f"Set {service.upper()}_API_KEY environment variable")
            suggestions.append(f"Add {service} API key to your configuration file")
            
        elif "url" in issue.field_path.lower():
            if "localhost" in str(config_data.get(issue.field_path.split('.')[0], {})):
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
            if "concurrent" in issue.field_path.lower() or "high" in issue.message.lower():
                suggestions.append("Reduce max_concurrent_requests for better stability")
                suggestions.append("Consider values between 10-50 for production")
            elif "timeout" in issue.field_path.lower():
                suggestions.append("Increase timeout to at least 1 second")
                
        # Generic suggestions for unknown issues
        if not suggestions:
            suggestions.append("Check the documentation for this configuration option")
            suggestions.append("Verify the value meets the expected format and constraints")
            
        return suggestions
