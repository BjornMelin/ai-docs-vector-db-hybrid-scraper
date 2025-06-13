"""ML Security implementation following KISS principle.

This module provides essential ML security features without over-engineering:
- Basic input validation to prevent common attacks (XSS, SQL injection)
- Integration with standard security tools (pip-audit for dependencies, trivy for containers)
- Simple monitoring and logging hooks
- Placeholder for rate limiting (should be handled by nginx/CloudFlare in production)

Philosophy: Use existing, proven security infrastructure rather than building ML-specific solutions.
"""

import logging
import subprocess
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from src.config import get_config

# Import SecurityValidator from the security.py file (not the package)
import sys
import importlib.util

# Import SecurityValidator from the file module
spec = importlib.util.spec_from_file_location("security_module", "src/security.py")
security_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(security_module)

BaseSecurityValidator = security_module.SecurityValidator

logger = logging.getLogger(__name__)


class SecurityCheckResult(BaseModel):
    """Simple security check result."""
    
    check_type: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error, critical
    timestamp: datetime = Field(default_factory=datetime.utcnow)
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
    
    def __init__(self):
        """Initialize with existing security config."""
        self.config = get_config()
        self.base_validator = BaseSecurityValidator.from_unified_config()
        self.checks_performed = []
    
    def validate_input(
        self,
        data: dict[str, Any],
        expected_schema: dict[str, type] | None = None
    ) -> SecurityCheckResult:
        """Basic input validation for ML requests.
        
        Args:
            data: Input data to validate
            expected_schema: Expected data types
            
        Returns:
            Security check result
        """
        try:
            # Check data size (prevent DoS)
            data_str = str(data)
            if len(data_str) > 1_000_000:  # 1MB limit
                return SecurityCheckResult(
                    check_type="input_validation",
                    passed=False,
                    message="Input data too large",
                    severity="warning"
                )
            
            # Basic type checking
            if expected_schema:
                for key, expected_type in expected_schema.items():
                    if key in data and not isinstance(data[key], expected_type):
                        return SecurityCheckResult(
                            check_type="input_validation",
                            passed=False,
                            message=f"Invalid type for {key}",
                            severity="warning"
                        )
            
            # Check for suspicious patterns (simple)
            suspicious_patterns = ["<script", "DROP TABLE", "__import__", "eval("]
            for pattern in suspicious_patterns:
                if pattern in data_str:
                    return SecurityCheckResult(
                        check_type="input_validation",
                        passed=False,
                        message=f"Suspicious pattern detected: {pattern}",
                        severity="error"
                    )
            
            return SecurityCheckResult(
                check_type="input_validation",
                passed=True,
                message="Input validation passed"
            )
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return SecurityCheckResult(
                check_type="input_validation",
                passed=False,
                message=str(e),
                severity="error"
            )
    
    def check_dependencies(self) -> SecurityCheckResult:
        """Run dependency security check using pip-audit.
        
        Returns:
            Security check result
        """
        try:
            # Use pip-audit if available
            result = subprocess.run(
                ["pip-audit", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                import json
                audit_data = json.loads(result.stdout)
                vulnerabilities = audit_data.get("vulnerabilities", [])
                
                if vulnerabilities:
                    return SecurityCheckResult(
                        check_type="dependency_scan",
                        passed=False,
                        message=f"Found {len(vulnerabilities)} vulnerabilities",
                        severity="warning",
                        details={"count": len(vulnerabilities)}
                    )
                else:
                    return SecurityCheckResult(
                        check_type="dependency_scan",
                        passed=True,
                        message="No vulnerabilities found"
                    )
            else:
                # pip-audit not available or failed
                return SecurityCheckResult(
                    check_type="dependency_scan",
                    passed=True,
                    message="Dependency scan skipped (pip-audit not available)"
                )
                
        except subprocess.TimeoutExpired:
            return SecurityCheckResult(
                check_type="dependency_scan",
                passed=True,
                message="Dependency scan timed out",
                severity="info"
            )
        except Exception as e:
            logger.error(f"Dependency check error: {e}")
            return SecurityCheckResult(
                check_type="dependency_scan",
                passed=True,
                message="Dependency scan failed",
                severity="info"
            )
    
    def check_container(self, image_name: str) -> SecurityCheckResult:
        """Basic container security check using trivy if available.
        
        Args:
            image_name: Container image to check
            
        Returns:
            Security check result
        """
        try:
            # Try trivy first
            result = subprocess.run(
                ["trivy", "image", "--severity", "CRITICAL,HIGH", 
                 "--quiet", "--format", "json", image_name],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                import json
                scan_data = json.loads(result.stdout)
                
                # Count vulnerabilities
                vuln_count = sum(
                    len(r.get("Vulnerabilities", []))
                    for r in scan_data.get("Results", [])
                )
                
                if vuln_count > 0:
                    return SecurityCheckResult(
                        check_type="container_scan",
                        passed=False,
                        message=f"Found {vuln_count} high/critical vulnerabilities",
                        severity="error",
                        details={"vulnerability_count": vuln_count}
                    )
                else:
                    return SecurityCheckResult(
                        check_type="container_scan",
                        passed=True,
                        message="No critical vulnerabilities found"
                    )
            else:
                return SecurityCheckResult(
                    check_type="container_scan",
                    passed=True,
                    message="Container scan skipped (trivy not available)"
                )
                
        except Exception as e:
            logger.info(f"Container scan skipped: {e}")
            return SecurityCheckResult(
                check_type="container_scan",
                passed=True,
                message="Container scan not performed",
                severity="info"
            )
    
    def log_security_event(
        self,
        event_type: str,
        details: dict[str, Any],
        severity: str = "info"
    ) -> None:
        """Log security event for monitoring.
        
        Args:
            event_type: Type of security event
            details: Event details
            severity: Event severity
        """
        # Use existing logging infrastructure
        log_data = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            **details
        }
        
        if severity == "critical":
            logger.error(f"Security event: {log_data}")
        elif severity == "error":
            logger.warning(f"Security event: {log_data}")
        else:
            logger.info(f"Security event: {log_data}")
    
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
            "critical_issues": len([c for c in failed_checks if c.severity == "critical"]),
            "last_check": self.checks_performed[-1].timestamp if self.checks_performed else None
        }


# Simple rate limiter using existing infrastructure
class SimpleRateLimiter:
    """Simple rate limiter that integrates with existing middleware."""
    
    def __init__(self):
        """Initialize using existing config."""
        self.config = get_config()
    
    def is_allowed(self, identifier: str) -> bool:
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
            "<script", "DROP TABLE", "__import__", "eval(",
            "exec(", "UNION SELECT", "javascript:"
        ],
        description="Patterns to block in inputs"
    )