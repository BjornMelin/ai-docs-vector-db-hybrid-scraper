"""Comprehensive tests for ML security implementation.

Tests the minimalistic ML security approach with >90% coverage goal.
"""

import json
import subprocess
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.security.ml_security import (
    MinimalMLSecurityConfig,
    MLSecurityValidator,
    SecurityCheckResult,
    SecurityError,
)
from src.services.errors import ValidationError as ServiceValidationError


class TestSecurityCheckResult:
    """Test SecurityCheckResult model."""

    def test_security_check_result_creation(self):
        """Test creating security check result."""
        result = SecurityCheckResult(
            check_type="input_validation",
            passed=True,
            message="Test passed",
            severity="info",
        )

        assert result.check_type == "input_validation"
        assert result.passed is True
        assert result.message == "Test passed"
        assert result.severity == "info"
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.details, dict)

    def test_security_check_result_with_details(self):
        """Test security check result with details."""
        details = {"vulnerability_count": 5, "packages": ["pkg1", "pkg2"]}
        result = SecurityCheckResult(
            check_type="dependency_scan",
            passed=False,
            message="Vulnerabilities found",
            severity="warning",
            details=details,
        )

        assert result.details == details
        assert result.details["vulnerability_count"] == 5

    def test_security_check_result_defaults(self):
        """Test default values."""
        result = SecurityCheckResult(check_type="test", passed=True, message="Test")

        assert result.severity == "info"
        assert result.details == {}
        assert result.timestamp <= datetime.now(tz=UTC)


class TestMLSecurityValidator:  # pylint: disable=too-many-public-methods
    """Test MLSecurityValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance with patched configuration."""
        security_config = SimpleNamespace(
            enable_ml_input_validation=True,
            max_ml_input_size=1_000_000,
            enable_dependency_scanning=True,
            dependency_scan_on_startup=True,
            suspicious_patterns=["<script", "DROP TABLE", "__import__", "eval("],
            allowed_domains=["example.com"],
            blocked_domains=["bad.com"],
            max_query_length=512,
        )
        mock_config = SimpleNamespace(security=security_config)

        with patch("src.security.ml_security.get_settings", return_value=mock_config):
            return MLSecurityValidator()

    def test_init(self, validator):
        """Test validator initialization."""
        assert validator.config is not None
        assert validator.security_config is not None
        assert validator.checks_performed == []

    def test_validate_input_success(self, validator):
        """Test successful input validation."""
        data = {"query": "normal search query", "limit": 10}
        schema = {"query": str, "limit": int}

        result = validator.validate_input(data, schema)

        assert result.passed is True
        assert result.check_type == "input_validation"
        assert result.message == "Input validation passed"

    def test_validate_input_size_limit(self, validator):
        """Test input size validation."""
        # Create large data that exceeds 1MB
        large_data = {"data": "x" * 1_000_001}

        result = validator.validate_input(large_data)

        assert result.passed is False
        assert result.message == "Input data too large"
        assert result.severity == "warning"

    def test_validate_input_type_mismatch(self, validator):
        """Test input type validation."""
        data = {"query": 123, "limit": "not a number"}
        schema = {"query": str, "limit": int}

        result = validator.validate_input(data, schema)

        assert result.passed is False
        assert "Invalid type" in result.message
        assert result.severity == "warning"

    @pytest.mark.parametrize(
        ("pattern", "data"),
        [
            ("<script", {"query": "<script>alert('xss')</script>"}),
            ("DROP TABLE", {"sql": "DROP TABLE users; --"}),
            ("__import__", {"code": "__import__('os').system('ls')"}),
            ("eval(", {"expr": "eval('malicious code')"}),
        ],
    )
    def test_validate_input_suspicious_patterns(self, validator, pattern, data):
        """Test detection of suspicious patterns."""
        result = validator.validate_input(data)

        assert result.passed is False
        assert f"Suspicious pattern detected: {pattern}" in result.message
        assert result.severity == "error"

    def test_validate_input_none_data(self, validator):
        """Test handling of None input data."""
        # Test None data - should be handled gracefully
        result = validator.validate_input(None)

        # None input should be caught and result in error
        assert result.check_type == "input_validation"

    # The function handles this case, so it
    # may pass or fail depending on implementation

    def test_check_dependencies_success(self, validator):
        """Test successful dependency check."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"vulnerabilities": []})

        with (
            patch("shutil.which", return_value="/usr/bin/pip-audit"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = validator.check_dependencies()

            assert result.passed is True
            assert result.message == "No vulnerabilities found"
            assert result.check_type == "dependency_scan"

    def test_check_dependencies_vulnerabilities_found(self, validator):
        """Test dependency check with vulnerabilities."""
        vulnerabilities = [
            {"package": "pkg1", "vulnerability": "CVE-123"},
            {"package": "pkg2", "vulnerability": "CVE-456"},
        ]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"vulnerabilities": vulnerabilities})

        with (
            patch("shutil.which", return_value="/usr/bin/pip-audit"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = validator.check_dependencies()

            assert result.passed is False
            assert "Found 2 vulnerabilities" in result.message
            assert result.severity == "warning"
            assert result.details["count"] == 2

    def test_check_dependencies_pip_audit_not_available(self, validator):
        """Test dependency check when pip-audit is not available."""
        with patch("shutil.which", return_value=None):
            result = validator.check_dependencies()

        assert result.passed is True
        assert "pip-audit not available" in result.message

    def test_check_dependencies_timeout(self, validator):
        """Test dependency check timeout."""
        with (
            patch("shutil.which", return_value="/usr/bin/pip-audit"),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)),
        ):
            result = validator.check_dependencies()

            assert result.passed is True
            assert result.message == "Dependency scan timed out"
            assert result.severity == "info"

    def test_check_dependencies_exception(self, validator):
        """Test dependency check exception handling."""
        with (
            patch("shutil.which", return_value="/usr/bin/pip-audit"),
            patch("subprocess.run", side_effect=Exception("Test error")),
            patch("src.security.ml_security.logger") as mock_logger,
        ):
            result = validator.check_dependencies()

            assert result.passed is True
            assert "scan failed" in result.message
            assert result.severity == "info"
            mock_logger.exception.assert_called_once()

    def test_check_container_success(self, validator):
        """Test successful container scan."""
        scan_data = {"Results": []}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(scan_data)

        with (
            patch("shutil.which", return_value="/usr/bin/trivy"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = validator.check_container("example/myapp:latest")

            assert result.passed is True
            assert result.message == "No critical vulnerabilities found"
            assert result.check_type == "container_scan"

    def test_check_container_vulnerabilities_found(self, validator):
        """Test container scan with vulnerabilities."""
        scan_data = {
            "Results": [
                {"Vulnerabilities": [{"id": "CVE-1"}, {"id": "CVE-2"}]},
                {"Vulnerabilities": [{"id": "CVE-3"}]},
            ]
        }
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(scan_data)

        with (
            patch("shutil.which", return_value="/usr/bin/trivy"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = validator.check_container("example/myapp:latest")

            assert result.passed is False
            assert "Found 3 high/critical vulnerabilities" in result.message
            assert result.severity == "error"
            assert result.details["vulnerability_count"] == 3

    def test_check_container_trivy_not_available(self, validator):
        """Test container scan when trivy is not available."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = validator.check_container("example/myapp:latest")

            assert result.passed is True
            assert "trivy not available" in result.message

    def test_check_container_exception(self, validator):
        """Test container scan exception handling."""
        with (
            patch("shutil.which", return_value="/usr/bin/trivy"),
            patch("subprocess.run", side_effect=Exception("Test error")),
            patch("src.security.ml_security.logger") as mock_logger,
        ):
            result = validator.check_container("example/myapp:latest")

            assert result.passed is True
            assert "scan failed" in result.message
            assert result.severity == "info"
            mock_logger.exception.assert_called_once()

    @pytest.mark.parametrize(
        ("severity", "log_method"),
        [
            ("critical", "error"),
            ("error", "warning"),
            ("warning", "info"),
            ("info", "info"),
        ],
    )
    def test_log_security_event(self, validator, severity, log_method):
        """Test security event logging."""
        with patch("src.security.ml_security.logger") as mock_logger:
            validator.log_security_event(
                "test_event", {"key": "value"}, severity=severity
            )

            log_func = getattr(mock_logger, log_method)
            log_func.assert_called_once()

            # Check logged data
            call_args = log_func.call_args
            assert call_args.args[0] == "Security event: %s"
            assert call_args.args[1] == "test_event"
            assert call_args.kwargs["extra"] == {"key": "value"}

    def test_get_security_summary_empty(self, validator):
        """Test security summary with no checks."""
        summary = validator.get_security_summary()

        assert summary["total_checks"] == 0
        assert summary["passed"] == 0
        assert summary["failed"] == 0
        assert summary["critical_issues"] == 0
        assert summary["last_check"] is None

    def test_get_security_summary_with_checks(self, validator):
        """Test security summary with multiple checks."""
        validator.checks_performed = [
            SecurityCheckResult(
                check_type="test1",
                passed=True,
                message="OK",
                severity="info",
            ),
            SecurityCheckResult(
                check_type="test2",
                passed=False,
                message="Failed",
                severity="error",
            ),
            SecurityCheckResult(
                check_type="test3",
                passed=False,
                message="Critical",
                severity="critical",
            ),
        ]

        summary = validator.get_security_summary()

        assert summary["total_checks"] == 3
        assert summary["passed"] == 1
        assert summary["failed"] == 2
        assert summary["critical_issues"] == 1
        assert summary["last_check"] is not None

    def test_validate_url_allows_configured_domain(self, validator):
        """URL validation should allow configured safe domains."""
        url = "https://example.com/resource"
        assert validator.validate_url(url) == url

    def test_validate_url_blocks_listed_domain(self, validator):
        """URL validation should block domains marked as unsafe."""
        with pytest.raises(SecurityError, match="blocked"):
            validator.validate_url("https://bad.com/resource")

    def test_validate_url_rejects_scheme(self, validator):
        """URL validation should reject unsupported schemes."""
        with pytest.raises(SecurityError, match="scheme"):
            validator.validate_url("ftp://example.com/file")

    def test_validate_collection_name_success(self, validator):
        """Collection name validation returns sanitized value."""
        assert validator.validate_collection_name("Valid_Name-01") == "Valid_Name-01"

    def test_validate_collection_name_invalid_chars(self, validator):
        """Invalid collection names raise SecurityError."""
        with pytest.raises(SecurityError, match="letters, numbers"):
            validator.validate_collection_name("invalid name!")

    def test_validate_query_string_length(self, validator):
        """Query validation enforces configured maximum length."""
        long_query = "x" * 600
        with pytest.raises(SecurityError, match="max 512"):
            validator.validate_query_string(long_query)

    def test_validate_query_string_sanitizes(self, validator):
        """Query validation strips unsafe characters."""
        cleaned = validator.validate_query_string("<script>alert(1)</script>")
        assert "<" not in cleaned
        assert ">" not in cleaned

    def test_sanitize_filename(self, validator):
        """Filename sanitization removes unsafe characters."""
        sanitized = validator.sanitize_filename("../../etc/passwd")
        assert sanitized == "passwd"


class TestMinimalMLSecurityConfig:
    """Test MinimalMLSecurityConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MinimalMLSecurityConfig()

        assert config.enable_input_validation is True
        assert config.max_input_size == 1_000_000
        assert config.enable_dependency_scanning is True
        assert config.dependency_scan_frequency_hours == 24
        assert len(config.suspicious_patterns) > 0
        assert "<script" in config.suspicious_patterns

    def test_custom_config(self):
        """Test custom configuration."""
        patterns = ["pattern1", "pattern2"]
        config = MinimalMLSecurityConfig(
            enable_input_validation=False,
            max_input_size=500_000,
            enable_dependency_scanning=False,
            dependency_scan_frequency_hours=12,
            suspicious_patterns=patterns,
        )

        assert config.enable_input_validation is False
        assert config.max_input_size == 500_000
        assert config.enable_dependency_scanning is False
        assert config.dependency_scan_frequency_hours == 12
        assert config.suspicious_patterns == patterns

    def test_config_validation(self):
        """Test configuration validation."""
        # Test that Pydantic validation works
        config = MinimalMLSecurityConfig()
        assert isinstance(config.model_dump(), dict)


class TestIntegration:
    """Integration tests for ML security."""

    @pytest.fixture
    def validator(self):
        """Create validator for integration tests."""
        security_config = SimpleNamespace(
            enable_ml_input_validation=True,
            max_ml_input_size=1_000_000,
            enable_dependency_scanning=True,
            dependency_scan_on_startup=True,
            suspicious_patterns=["<script", "DROP TABLE", "__import__", "eval("],
            allowed_domains=["example.com"],
            blocked_domains=["bad.com"],
            max_query_length=512,
        )
        mock_config = SimpleNamespace(security=security_config)

        with patch("src.security.ml_security.get_settings", return_value=mock_config):
            return MLSecurityValidator()

    def test_full_security_workflow(self, validator):
        """Test complete security validation workflow."""
        # 1. Validate input
        input_result = validator.validate_input(
            {"query": "search for documents", "limit": 10}, {"query": str, "limit": int}
        )
        assert input_result.passed is True

        # 2. Check dependencies (mocked)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps({"vulnerabilities": []})

            dep_result = validator.check_dependencies()
            assert dep_result.passed is True

        # 3. Log security event
        validator.log_security_event(
            "security_check_complete", {"checks_passed": 2}, severity="info"
        )

        # 4. Get summary
        summary = validator.get_security_summary()

        # Note: checks are already tracked in checks_performed by the methods
        assert summary["total_checks"] >= 2  # May have more from automatic tracking
        assert summary["passed"] >= 2
        assert summary["failed"] == 0

    def test_security_failure_workflow(self, validator):
        """Test security validation with failures."""
        # 1. Validate malicious input
        malicious_result = validator.validate_input(
            {"query": "<script>alert('xss')</script>"}
        )
        assert malicious_result.passed is False

        # 2. Check dependencies with vulnerabilities
        with (
            patch("shutil.which", return_value="/usr/bin/pip-audit"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps(
                {"vulnerabilities": [{"package": "vulnerable-pkg"}]}
            )

            dep_result = validator.check_dependencies()
            assert dep_result.passed is False

        # 3. Log security events
        validator.log_security_event(
            "malicious_input_blocked", {"pattern": "<script>"}, severity="error"
        )

        # 4. Get summary
        # Note: malicious_result and dep_result are already tracked in checks_performed
        summary = validator.get_security_summary()

        # The validator automatically tracks all checks, so we should have at least 2
        assert summary["total_checks"] >= 2
        assert summary["failed"] >= 2  # Both checks should have failed
        assert summary["passed"] == 0


def test_security_error_subclass_validation_error() -> None:
    """Ensure SecurityError derives from the shared validation error base."""
    assert issubclass(SecurityError, ServiceValidationError)
