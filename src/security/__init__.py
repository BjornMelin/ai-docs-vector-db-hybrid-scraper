"""Security module for ML applications.

This module provides essential ML security features following KISS principles:
- Basic input validation to prevent common attacks
- Integration with standard security tools (pip-audit, trivy)
- Simple monitoring and logging hooks
- Leverages existing infrastructure (nginx, CloudFlare for rate limiting)
"""

# Avoid circular imports - import only what's needed
from .ml_security import (
    MLSecurityValidator,
    SecurityCheckResult,
    SimpleRateLimiter,
    MinimalMLSecurityConfig,
)

__all__ = [
    "MLSecurityValidator",
    "SecurityCheckResult",
    "SimpleRateLimiter",
    "MinimalMLSecurityConfig",
]