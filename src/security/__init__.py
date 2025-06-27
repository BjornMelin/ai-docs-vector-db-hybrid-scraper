"""Security module for ML applications.

This module provides essential ML security features following KISS principles:
- Basic input validation to prevent common attacks
- Integration with standard security tools (pip-audit, trivy)
- Simple monitoring and logging hooks
- Leverages existing infrastructure (nginx, CloudFlare for rate limiting)
"""

# Avoid circular imports - import only what's needed
from .ml_security import (
    MinimalMLSecurityConfig,
    MLSecurityValidator,
    SecurityCheckResult,
    SimpleRateLimiter,
)


# Add aliases for test compatibility
SecurityError = Exception  # Simple alias for backward compatibility
SecurityValidator = MLSecurityValidator  # Alias for test compatibility

__all__ = [
    "MLSecurityValidator",
    "MinimalMLSecurityConfig",
    "SecurityCheckResult",
    "SecurityError",
    "SecurityValidator",
    "SimpleRateLimiter",
]
