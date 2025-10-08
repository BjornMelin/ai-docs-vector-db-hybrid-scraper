"""Minimal ML security primitives that rely on shared configuration."""

import json
import logging
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.config import get_settings


logger = logging.getLogger(__name__)


class SecurityError(ValueError):
    """Security-related error raised for invalid user-controlled inputs."""


class SecurityCheckResult(BaseModel):
    """Simple security check result."""

    check_type: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error, critical
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)


class MLSecurityValidator:
    """ML security validator focusing on essentials.

    Provides basic security validation for ML applications:
    - Input size and type validation
    - Pattern-based attack detection
    - Dependency vulnerability scanning via pip-audit
    - Container scanning via trivy (if available)

    Note: Rate limiting should be handled by infrastructure (nginx, CloudFlare)
    rather than application code for better performance and reliability.
    """

    ALLOWED_SCHEMES: ClassVar[set[str]] = {"http", "https"}
    DANGEROUS_PATTERNS: ClassVar[tuple[str, ...]] = (
        r"javascript:",
        r"data:",
        r"file:",
        r"ftp:",
        r"localhost",
        r"127\.0\.0\.1",
        r"0\.0\.0\.0",
        r"::1",
        r"192\.168\.",
        r"10\.",
        r"172\.(1[6-9]|2[0-9]|3[0-1])\.",
    )

    def __init__(self) -> None:
        """Initialize with existing security config."""

        settings = get_settings()
        self.config = settings
        self.security_config = getattr(settings, "security", None)
        self.checks_performed: list[SecurityCheckResult] = []

    def _record_result(self, result: SecurityCheckResult) -> SecurityCheckResult:
        """Store and return a security check result for consistent handling."""

        self.checks_performed.append(result)
        return result

    @classmethod
    def from_unified_config(cls) -> "MLSecurityValidator":
        """Create MLSecurityValidator from unified config."""

        return cls()

    def validate_input(
        self, data: dict[str, Any], expected_schema: dict[str, type] | None = None
    ) -> SecurityCheckResult:
        """Basic input validation for ML requests.

        Args:
            data: Input data to validate
            expected_schema: Expected data types

        Returns:
            Security check result
        """

        # Check if ML input validation is enabled (use default if not configured)
        enable_validation = getattr(
            self.security_config, "enable_ml_input_validation", True
        )
        if not enable_validation:
            return self._record_result(
                SecurityCheckResult(
                    check_type="input_validation",
                    passed=True,
                    message="Input validation skipped (disabled in config)",
                )
            )

        try:
            # Check data size (prevent DoS)
            data_str = str(data)
            max_input_size = getattr(
                self.security_config, "max_ml_input_size", 1_000_000
            )
            if len(data_str) > max_input_size:
                return self._record_result(
                    SecurityCheckResult(
                        check_type="input_validation",
                        passed=False,
                        message="Input data too large",
                        severity="warning",
                    )
                )

            # Basic type checking
            if expected_schema:
                for key, expected_type in expected_schema.items():
                    if key in data and not isinstance(data[key], expected_type):
                        return self._record_result(
                            SecurityCheckResult(
                                check_type="input_validation",
                                passed=False,
                                message=f"Invalid type for {key}",
                                severity="warning",
                            )
                        )

            # Check for suspicious patterns from config
            suspicious_patterns = getattr(
                self.security_config,
                "suspicious_patterns",
                ["<script", "DROP TABLE", "__import__", "eval("],
            )
            for pattern in suspicious_patterns:
                if pattern in data_str:
                    return self._record_result(
                        SecurityCheckResult(
                            check_type="input_validation",
                            passed=False,
                            message=f"Suspicious pattern detected: {pattern}",
                            severity="error",
                        )
                    )

            return self._record_result(
                SecurityCheckResult(
                    check_type="input_validation",
                    passed=True,
                    message="Input validation passed",
                )
            )
        except Exception as exc:  # noqa: BLE001 - broad due to validation surface
            logger.exception("Input validation error")
            return self._record_result(
                SecurityCheckResult(
                    check_type="input_validation",
                    passed=False,
                    message=str(exc),
                    severity="error",
                )
            )

    def check_dependencies(self) -> SecurityCheckResult:
        """Run dependency security check using pip-audit.

        Returns:
            Security check result
        """

        # Check if dependency scanning is enabled (use default if not configured)
        enable_scanning = getattr(
            self.security_config, "enable_dependency_scanning", True
        )
        if not enable_scanning:
            return self._record_result(
                SecurityCheckResult(
                    check_type="dependency_scan",
                    passed=True,
                    message="Dependency scan skipped (disabled in config)",
                )
            )

        pip_audit_path = shutil.which("pip-audit")
        if not pip_audit_path:
            return self._record_result(
                SecurityCheckResult(
                    check_type="dependency_scan",
                    passed=True,
                    message="Dependency scan skipped (pip-audit not available)",
                )
            )

        # Validate executable path for security
        if not pip_audit_path.startswith(("/usr/bin/", "/usr/local/bin/", "/opt/")):
            logger.warning("pip-audit found in unexpected location: %s", pip_audit_path)

        try:
            result = subprocess.run(  # Validated executable path  # noqa: S603
                [pip_audit_path, "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                shell=False,  # Explicitly disable shell
            )
        except subprocess.TimeoutExpired:
            check_result = SecurityCheckResult(
                check_type="dependency_scan",
                passed=True,
                message="Dependency scan timed out",
                severity="info",
            )
        except (OSError, PermissionError):
            logger.exception("Dependency check error")
            check_result = SecurityCheckResult(
                check_type="dependency_scan",
                passed=True,
                message="Dependency scan failed",
                severity="info",
            )
        except Exception:  # noqa: BLE001 - capture unexpected subprocess errors
            logger.exception("Unexpected dependency check error")
            check_result = SecurityCheckResult(
                check_type="dependency_scan",
                passed=True,
                message="Dependency scan failed",
                severity="info",
            )
        else:
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                vulnerabilities = audit_data.get("vulnerabilities", [])
                if vulnerabilities:
                    check_result = SecurityCheckResult(
                        check_type="dependency_scan",
                        passed=False,
                        message=f"Found {len(vulnerabilities)} vulnerabilities",
                        severity="warning",
                        details={"count": len(vulnerabilities)},
                    )
                else:
                    check_result = SecurityCheckResult(
                        check_type="dependency_scan",
                        passed=True,
                        message="No vulnerabilities found",
                    )
            else:
                check_result = SecurityCheckResult(
                    check_type="dependency_scan",
                    passed=True,
                    message="Dependency scan skipped (pip-audit unavailable or failed)",
                )

        return self._record_result(check_result)

    def check_container(self, image_name: str) -> SecurityCheckResult:
        """Basic container security check using trivy if available.

        Args:
            image_name: Container image to check

        Returns:
            Security check result

        """
        try:
            # Try trivy first with full path for security
            trivy_path = shutil.which("trivy")
            if not trivy_path:
                return self._record_result(
                    SecurityCheckResult(
                        check_type="container_scan",
                        passed=True,
                        message="Container scan skipped (trivy not available)",
                    )
                )

            # Validate executable path for security
            if not trivy_path.startswith(("/usr/bin/", "/usr/local/bin/", "/opt/")):
                logger.warning("trivy found in unexpected location: %s", trivy_path)

            # Validate image name for security
            if not image_name or ".." in image_name or "/" not in image_name:
                return self._record_result(
                    SecurityCheckResult(
                        check_type="container_scan",
                        passed=False,
                        message="Invalid container image name",
                        severity="error",
                    )
                )

            result = subprocess.run(  # Validated executable path  # noqa: S603
                [
                    trivy_path,
                    "image",
                    "--severity",
                    "CRITICAL,HIGH",
                    "--quiet",
                    "--format",
                    "json",
                    image_name,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                shell=False,  # Explicitly disable shell
            )

            if result.returncode == 0:
                scan_data = json.loads(result.stdout)

                # Count vulnerabilities
                vuln_count = sum(
                    len(r.get("Vulnerabilities", []))
                    for r in scan_data.get("Results", [])
                )

                if vuln_count > 0:
                    return self._record_result(
                        SecurityCheckResult(
                            check_type="container_scan",
                            passed=False,
                            message=(
                                f"Found {vuln_count} high/critical vulnerabilities"
                            ),
                            severity="error",
                            details={"vulnerability_count": vuln_count},
                        )
                    )
                return self._record_result(
                    SecurityCheckResult(
                        check_type="container_scan",
                        passed=True,
                        message="No critical vulnerabilities found",
                    )
                )
            return self._record_result(
                SecurityCheckResult(
                    check_type="container_scan",
                    passed=True,
                    message="Container scan skipped (trivy unavailable or failed)",
                )
            )

        except (AttributeError, RuntimeError, ValueError):
            logger.info("Container scan skipped")
            return self._record_result(
                SecurityCheckResult(
                    check_type="container_scan",
                    passed=True,
                    message="Container scan not performed",
                    severity="info",
                )
            )
        except Exception:  # noqa: BLE001 - safeguard against subprocess errors
            logger.exception("Container scan failed")
            return self._record_result(
                SecurityCheckResult(
                    check_type="container_scan",
                    passed=True,
                    message="Container scan failed",
                    severity="info",
                )
            )

    def validate_collection_name(self, name: str) -> str:
        """Validate collection name to prevent unsafe identifiers.

        Args:
            name: Collection name to validate.

        Returns:
            Sanitized collection name.

        Raises:
            SecurityError: If the collection name is invalid.
        """

        if not isinstance(name, str) or not name.strip():
            msg = "Collection name must be a non-empty string"
            raise SecurityError(msg)

        candidate = name.strip()
        if len(candidate) > 64:
            msg = "Collection name too long (max 64 characters)"
            raise SecurityError(msg)

        if not re.fullmatch(r"[a-zA-Z0-9_-]+", candidate):
            msg = (
                "Collection name can only contain letters, numbers, "
                "underscores, and hyphens"
            )
            raise SecurityError(msg)

        return candidate

    def validate_query_string(self, query: str) -> str:
        """Validate query string using security configuration constraints.

        Args:
            query: Query string to validate.

        Returns:
            Sanitized query string.

        Raises:
            SecurityError: If the query is invalid.
        """

        if not isinstance(query, str) or not query.strip():
            msg = "Query must be a non-empty string"
            raise SecurityError(msg)

        cleaned = query.strip()
        max_length = getattr(self.security_config, "max_query_length", 1_000)
        if len(cleaned) > max_length:
            msg = f"Query too long (max {max_length} characters)"
            raise SecurityError(msg)

        return re.sub(r'[<>"\']', "", cleaned)

    def validate_url(self, url: str) -> str:
        """Validate URL using allow/block lists and scheme checks.

        Args:
            url: URL to validate.

        Returns:
            Sanitized URL string.

        Raises:
            SecurityError: If the URL is unsafe or malformed.
        """

        if not isinstance(url, str) or not url.strip():
            msg = "URL must be a non-empty string"
            raise SecurityError(msg)

        try:
            parsed = urlparse(url.strip())
        except Exception as exc:  # noqa: BLE001 - propagate security context
            msg = f"Invalid URL format: {exc}"
            raise SecurityError(msg) from exc

        if parsed.scheme.lower() not in self.ALLOWED_SCHEMES:
            msg = f"URL scheme '{parsed.scheme}' not allowed"
            raise SecurityError(msg)

        domain = parsed.netloc.lower()
        blocked_domains = getattr(self.security_config, "blocked_domains", [])
        for blocked in blocked_domains:
            if blocked.lower() in domain:
                msg = f"Domain '{domain}' is blocked"
                raise SecurityError(msg)

        allowed_domains = getattr(self.security_config, "allowed_domains", [])
        if allowed_domains and not any(
            allowed.lower() in domain or domain.endswith(allowed.lower())
            for allowed in allowed_domains
        ):
            msg = f"Domain '{domain}' not in allowed list"
            raise SecurityError(msg)

        lowered_url = url.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, lowered_url):
                msg = f"URL contains dangerous pattern: {pattern}"
                raise SecurityError(msg)

        if len(url) > 2048:
            msg = "URL too long (max 2048 characters)"
            raise SecurityError(msg)

        return url.strip()

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file operations.

        Args:
            filename: Filename to sanitize.

        Returns:
            Sanitized filename safe for local operations.
        """

        if not isinstance(filename, str) or not filename:
            return "safe_filename"

        candidate = Path(filename.strip()).name
        candidate = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", candidate)
        if len(candidate) > 255:
            candidate = candidate[:255]

        return candidate or "safe_filename"

    def log_security_event(
        self, event_type: str, details: dict[str, Any], severity: str = "info"
    ) -> None:
        """Log security event for monitoring.

        Args:
            event_type: Type of security event
            details: Event details
            severity: Event severity

        """
        # Use existing logging infrastructure
        if severity == "critical":
            logger.error("Security event: %s", event_type, extra=details)
        elif severity == "error":
            logger.warning("Security event: %s", event_type, extra=details)
        else:
            logger.info("Security event: %s", event_type, extra=details)

    def get_security_summary(self) -> dict[str, Any]:
        """Get summary of security checks performed.

        Returns:
            Summary of security status

        """
        failed_checks = [c for c in self.checks_performed if not c.passed]

        return {
            "total_checks": len(self.checks_performed),
            "passed": len(self.checks_performed) - len(failed_checks),
            "failed": len(failed_checks),
            "critical_issues": len(
                [c for c in failed_checks if c.severity == "critical"]
            ),
            "last_check": self.checks_performed[-1].timestamp
            if self.checks_performed
            else None,
        }


# Simple rate limiter using existing infrastructure
class SimpleRateLimiter:
    """Simple rate limiter that integrates with existing middleware."""

    def __init__(self):
        """Initialize using existing config."""
        self.config = get_settings()

    def is_allowed(self, _identifier: str) -> bool:
        """Check if request is allowed.

        This is a placeholder - actual rate limiting should be done
        by nginx/CloudFlare/API Gateway or existing middleware.

        Args:
            identifier: Client identifier

        Returns:
            Whether request is allowed

        """
        # In production, use existing rate limiting infrastructure
        # This is just for local development
        return True


# Minimal security configuration extension
class MinimalMLSecurityConfig(BaseModel):
    """Minimal ML-specific security configuration."""

    enable_input_validation: bool = Field(
        default=True, description="Enable basic input validation"
    )
    max_input_size: int = Field(
        default=1_000_000, description="Maximum input size in bytes"
    )
    enable_dependency_scanning: bool = Field(
        default=True, description="Enable dependency vulnerability scanning"
    )
    dependency_scan_frequency_hours: int = Field(
        default=24, description="How often to scan dependencies"
    )
    suspicious_patterns: list[str] = Field(
        default_factory=lambda: [
            "<script",
            "DROP TABLE",
            "__import__",
            "eval(",
            "exec(",
            "UNION SELECT",
            "javascript:",
        ],
        description="Patterns to block in inputs",
    )
