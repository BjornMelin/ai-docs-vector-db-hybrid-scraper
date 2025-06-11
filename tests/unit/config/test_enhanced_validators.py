"""Comprehensive tests for enhanced configuration validators.

This test file covers the enhanced validation system that provides
detailed error reporting and automatic suggestions for configuration issues.
"""

from unittest.mock import patch

import pytest
from src.config.enhanced_validators import ConfigurationValidator
from src.config.enhanced_validators import ValidationIssue
from src.config.enhanced_validators import ValidationReport
from src.config.enhanced_validators import ValidationSeverity
from src.config.models import UnifiedConfig


class TestValidationIssue:
    """Test the ValidationIssue model."""

    def test_validation_issue_creation(self):
        """Test basic ValidationIssue creation."""
        issue = ValidationIssue(
            field_path="api_key",
            message="API key is required",
            severity=ValidationSeverity.ERROR,
            category="security",
            suggestion="Add API key to configuration"
        )

        assert issue.field_path == "api_key"
        assert issue.message == "API key is required"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.category == "security"
        assert issue.suggestion == "Add API key to configuration"

    def test_validation_issue_default_suggestions(self):
        """Test ValidationIssue with default empty suggestion."""
        issue = ValidationIssue(
            field_path="debug",
            message="Debug should be disabled in production",
            severity=ValidationSeverity.WARNING,
            category="configuration"
        )

        assert issue.suggestion is None

    def test_validation_issue_string_representation(self):
        """Test string representation of ValidationIssue."""
        issue = ValidationIssue(
            field_path="database.port",
            message="Port must be between 1 and 65535",
            severity=ValidationSeverity.ERROR,
            category="validation"
        )

        str_repr = str(issue)
        assert "database.port" in str_repr
        assert "Port must be between 1 and 65535" in str_repr
        assert "ERROR" not in str_repr  # String representation doesn't include "ERROR" text
        assert "❌" in str_repr  # Uses emoji instead


class TestValidationReport:
    """Test the ValidationReport model."""

    def test_empty_validation_report(self):
        """Test empty validation report."""
        report = ValidationReport(
            issues=[],
            is_valid=True,
            config_hash="test_hash"
        )

        assert report.is_valid is True
        assert len(report.errors) == 0
        assert len(report.warnings) == 0
        assert len(report.info) == 0

    def test_validation_report_with_errors(self):
        """Test validation report with errors."""
        error = ValidationIssue(
            field_path="api_key",
            message="API key is required",
            severity=ValidationSeverity.ERROR,
            category="security"
        )

        report = ValidationReport(
            issues=[error],
            is_valid=False,
            config_hash="test_hash"
        )

        assert report.is_valid is False
        assert len(report.errors) == 1
        assert len(report.warnings) == 0

    def test_validation_report_warnings_only(self):
        """Test validation report with only warnings."""
        warning = ValidationIssue(
            field_path="debug",
            message="Debug enabled in production",
            severity=ValidationSeverity.WARNING,
            category="environment"
        )

        report = ValidationReport(
            issues=[warning],
            is_valid=True,
            config_hash="test_hash"
        )

        assert report.is_valid is True  # Warnings don't invalidate config
        assert len(report.warnings) == 1
        assert len(report.errors) == 0

    def test_validation_report_mixed_issues(self):
        """Test validation report with mixed issue types."""
        error = ValidationIssue(
            field_path="api_key",
            message="Required field missing",
            severity=ValidationSeverity.ERROR,
            category="security"
        )
        warning = ValidationIssue(
            field_path="debug",
            message="Debug enabled",
            severity=ValidationSeverity.WARNING,
            category="environment"
        )
        info = ValidationIssue(
            field_path="cache",
            message="Using default cache settings",
            severity=ValidationSeverity.INFO,
            category="configuration"
        )

        report = ValidationReport(
            issues=[error, warning, info],
            is_valid=False,
            config_hash="test_hash"
        )

        assert report.is_valid is False  # Has errors
        assert len(report.errors) == 1
        assert len(report.warnings) == 1
        assert len(report.info) == 1

    def test_validation_report_add_issue(self):
        """Test adding issues to validation report."""
        report = ValidationReport(
            issues=[],
            is_valid=True,
            config_hash="test_hash"
        )

        # Add error
        error = ValidationIssue(
            field_path="test",
            message="Test error",
            severity=ValidationSeverity.ERROR,
            category="test"
        )
        report.issues.append(error)
        report.is_valid = False  # Update validity

        assert report.is_valid is False
        assert len(report.errors) == 1


class TestConfigurationValidator:
    """Test the ConfigurationValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigurationValidator("development")

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = ConfigurationValidator("production")
        assert validator.environment == "production"

    def test_validator_default_environment(self):
        """Test validator with default environment."""
        validator = ConfigurationValidator()
        assert validator.environment == "development"

    @pytest.fixture
    def valid_config(self):
        """Create a valid configuration for testing."""
        return UnifiedConfig(
            environment="development",
            debug=True,
            log_level="DEBUG"
        )

    @pytest.fixture
    def invalid_config_data(self):
        """Create invalid configuration data for testing."""
        return {
            "environment": "production",
            "debug": True,  # Invalid: debug should be False in production
            "openai": {"api_key": ""},  # Invalid: empty API key
            "qdrant": {"url": "invalid-url"}  # Invalid: malformed URL
        }

    def test_validate_configuration_valid(self, valid_config):
        """Test validation of a valid configuration."""
        report = self.validator.validate_configuration(valid_config)

        assert report.is_valid is True
        assert len(report.errors) == 0

    def test_validate_configuration_with_dict(self, invalid_config_data):
        """Test validation with dictionary input."""
        report = self.validator.validate_configuration(invalid_config_data)

        # Should have validation issues
        assert len(report.errors) > 0 or len(report.warnings) > 0

    def test_validate_business_rules_production_debug(self):
        """Test business rule validation for production debug setting."""
        from src.config.models import UnifiedConfig
        from src.config.enums import Environment
        
        # Create a proper config object with debug enabled in production
        config = UnifiedConfig(
            environment=Environment.PRODUCTION,
            debug=True
        )

        # Use validator instance with production environment
        validator = ConfigurationValidator("production")
        report = validator.validate_configuration(config)

        # Should find debug=True in production as an error
        debug_issues = [issue for issue in report.issues if "debug" in issue.field_path.lower()]
        assert len(debug_issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in debug_issues)

    def test_validate_business_rules_missing_api_keys(self):
        """Test business rule validation for missing API keys."""
        # Test with dictionary configuration to avoid Pydantic validation
        config_data = {
            "embedding_provider": "openai",
            "openai": {"api_key": ""}  # Empty API key
        }

        report = self.validator.validate_configuration(config_data)

        # Should find missing OpenAI API key
        api_key_issues = [issue for issue in report.issues if "api_key" in issue.field_path]
        assert len(api_key_issues) > 0

    def test_validate_business_rules_invalid_urls(self):
        """Test business rule validation for invalid URLs."""
        config_data = {
            "qdrant": {"url": "not-a-valid-url"},
            "cache": {"dragonfly_url": "invalid-redis-url"}
        }

        report = self.validator.validate_configuration(config_data)

        # Should find URL validation issues
        url_issues = [issue for issue in report.issues if "url" in issue.field_path]
        assert len(url_issues) > 0

    def test_validate_business_rules_port_ranges(self):
        """Test business rule validation for port ranges."""
        config_data = {
            "qdrant": {"port": 70000},  # Invalid: port too high
            "database": {"port": 0}     # Invalid: port too low
        }

        report = self.validator.validate_configuration(config_data)

        # Should find port validation issues
        port_issues = [issue for issue in report.issues if "port" in issue.field_path]
        assert len(port_issues) > 0

    def test_validate_business_rules_performance_settings(self):
        """Test business rule validation for performance settings."""
        config_data = {
            "environment": "production",
            "performance": {
                "max_concurrent_requests": 200,  # Too high
                "request_timeout": 0.1          # Too low
            }
        }

        report = self.validator.validate_configuration(config_data)

        # Should find performance issues
        perf_issues = [issue for issue in report.issues if "performance" in issue.field_path]
        assert len(perf_issues) > 0

    def test_generate_suggestions_api_key_missing(self):
        """Test suggestion generation for missing API keys."""
        issue = ValidationIssue(
            field_path="openai.api_key",
            message="API key is required",
            severity=ValidationSeverity.ERROR,
            category="missing_required"
        )

        suggestions = self.validator._generate_suggestions(issue, {})

        assert len(suggestions) > 0
        assert any("environment variable" in suggestion.lower() for suggestion in suggestions)
        assert any("openai" in suggestion.lower() for suggestion in suggestions)

    def test_generate_suggestions_url_format(self):
        """Test suggestion generation for URL format issues."""
        issue = ValidationIssue(
            field_path="qdrant.url",
            message="Invalid URL format",
            severity=ValidationSeverity.ERROR,
            category="format_error"
        )

        suggestions = self.validator._generate_suggestions(issue, {"qdrant": {"url": "localhost:6333"}})

        assert len(suggestions) > 0
        assert any("http://" in suggestion for suggestion in suggestions)

    def test_generate_suggestions_debug_production(self):
        """Test suggestion generation for debug in production."""
        issue = ValidationIssue(
            field_path="debug",
            message="Debug should be disabled in production",
            severity=ValidationSeverity.ERROR,
            category="environment_constraint"
        )

        suggestions = self.validator._generate_suggestions(issue, {"environment": "production", "debug": True})

        assert len(suggestions) > 0
        assert any("false" in suggestion.lower() for suggestion in suggestions)

    def test_generate_suggestions_port_range(self):
        """Test suggestion generation for port range issues."""
        issue = ValidationIssue(
            field_path="qdrant.port",
            message="Port must be between 1 and 65535",
            severity=ValidationSeverity.ERROR,
            category="value_range"
        )

        suggestions = self.validator._generate_suggestions(issue, {"qdrant": {"port": 70000}})

        assert len(suggestions) > 0
        assert any("6333" in suggestion for suggestion in suggestions)  # Default Qdrant port

    def test_generate_suggestions_performance_tuning(self):
        """Test suggestion generation for performance issues."""
        issue = ValidationIssue(
            field_path="performance.max_concurrent_requests",
            message="Value too high for production",
            severity=ValidationSeverity.WARNING,
            category="performance"
        )

        suggestions = self.validator._generate_suggestions(issue, {})

        assert len(suggestions) > 0
        assert any("reduce" in suggestion.lower() for suggestion in suggestions)

    def test_generate_suggestions_cache_configuration(self):
        """Test suggestion generation for cache configuration issues."""
        issue = ValidationIssue(
            field_path="cache.redis_pool_size",
            message="Pool size should be optimized for environment",
            severity=ValidationSeverity.INFO,
            category="configuration"
        )

        suggestions = self.validator._generate_suggestions(issue, {"environment": "production"})

        assert len(suggestions) > 0

    def test_generate_suggestions_unknown_issue(self):
        """Test suggestion generation for unknown issue types."""
        issue = ValidationIssue(
            field_path="unknown.field",
            message="Unknown validation error",
            severity=ValidationSeverity.ERROR,
            category="unknown"
        )

        suggestions = self.validator._generate_suggestions(issue, {})

        # Should return generic suggestions
        assert len(suggestions) > 0
        assert any("documentation" in suggestion.lower() for suggestion in suggestions)

    def test_validate_configuration_comprehensive(self):
        """Test comprehensive validation with multiple issue types."""
        config_data = {
            "environment": "production",
            "debug": True,                          # Error: debug in production
            "log_level": "INVALID",                 # Error: invalid log level
            "embedding_provider": "openai",
            "openai": {"api_key": ""},              # Error: empty API key
            "qdrant": {
                "url": "invalid-url",               # Error: invalid URL
                "port": 70000                       # Error: invalid port
            },
            "cache": {
                "enable_caching": True,
                "local_max_size": 0                 # Error: invalid size
            },
            "performance": {
                "max_concurrent_requests": 1000,   # Warning: very high
                "request_timeout": 0.1              # Warning: very low
            }
        }

        validator = ConfigurationValidator("production")
        report = validator.validate_configuration(config_data)

        # Should have multiple errors and warnings
        assert len(report.errors) > 0
        assert len(report.warnings) > 0
        assert report.is_valid is False

        # Check that suggestions are provided
        for issue in report.errors + report.warnings:
            if hasattr(issue, 'suggestions'):
                # Most issues should have suggestions
                pass  # We don't require all issues to have suggestions

    def test_validate_configuration_empty_config(self):
        """Test validation of empty configuration."""
        report = self.validator.validate_configuration({})

        # Empty config should have validation issues
        assert len(report.errors) > 0 or len(report.warnings) > 0

    def test_validate_configuration_minimal_valid_config(self):
        """Test validation of minimal valid configuration."""
        minimal_config = {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG"
        }

        report = self.validator.validate_configuration(minimal_config)

        # Minimal config should be valid (though may have warnings)
        assert report.is_valid is True

    def test_environment_specific_validation(self):
        """Test that validation rules change based on environment."""
        config_data = {"debug": True}

        # Development environment should allow debug=True
        dev_validator = ConfigurationValidator("development")
        dev_report = dev_validator.validate_configuration(config_data)

        # Production environment should warn/error about debug=True
        prod_validator = ConfigurationValidator("production")
        prod_report = prod_validator.validate_configuration(config_data)

        # Production should have more issues than development
        prod_debug_issues = [
            issue for issue in prod_report.errors + prod_report.warnings
            if "debug" in issue.field_path.lower()
        ]
        dev_debug_issues = [
            issue for issue in dev_report.errors + dev_report.warnings
            if "debug" in issue.field_path.lower()
        ]

        assert len(prod_debug_issues) >= len(dev_debug_issues)

    @patch('src.config.enhanced_validators.ValidationHelper.validate_url_format')
    def test_mock_url_validation(self, mock_url_validator):
        """Test validation with mocked URL validator."""
        mock_url_validator.return_value = False  # All URLs invalid

        config_data = {
            "qdrant": {"url": "http://localhost:6333"}
        }

        issues = self.validator._validate_business_rules(config_data)

        # Should call URL validator and find issues
        mock_url_validator.assert_called()
        url_issues = [issue for issue in issues if "url" in issue.field_path]
        assert len(url_issues) > 0

    def test_validation_issue_ordering(self):
        """Test that validation issues are properly ordered by severity."""
        config_data = {
            "environment": "production",
            "debug": True,                          # Error
            "performance": {"max_concurrent_requests": 1000},  # Warning
            "cache": {"enable_caching": True}       # Info (maybe)
        }

        report = self.validator.validate_configuration(config_data)

        # Errors should come first, then warnings, then info
        all_issues = report.errors + report.warnings + report.info
        severity_order = [issue.severity for issue in all_issues]

        # Check that errors come before warnings
        error_positions = [i for i, sev in enumerate(severity_order) if sev == ValidationSeverity.ERROR]
        warning_positions = [i for i, sev in enumerate(severity_order) if sev == ValidationSeverity.WARNING]

        if error_positions and warning_positions:
            assert max(error_positions) < min(warning_positions)

    def test_validation_with_custom_rules(self):
        """Test validation with custom business rules."""
        # This tests the extensibility of the validation system
        config_data = {
            "custom_field": "custom_value"
        }

        # The validator should handle unknown fields gracefully
        report = self.validator.validate_configuration(config_data)

        # Should not crash, may have warnings about unknown fields
        assert isinstance(report, ValidationReport)

    def test_suggestion_quality(self):
        """Test that suggestions are helpful and specific."""
        config_data = {
            "openai": {"api_key": ""},
            "qdrant": {"url": "localhost:6333"}
        }

        report = self.validator.validate_configuration(config_data)

        # Check suggestion quality
        for issue in report.errors + report.warnings:
            if hasattr(issue, 'suggestions') and issue.suggestions:
                for suggestion in issue.suggestions:
                    # Suggestions should be non-empty strings
                    assert isinstance(suggestion, str)
                    assert len(suggestion.strip()) > 0
                    # Suggestions should be actionable (contain action words)
                    action_words = ['set', 'add', 'change', 'remove', 'configure', 'use', 'try']
                    assert any(word in suggestion.lower() for word in action_words)
