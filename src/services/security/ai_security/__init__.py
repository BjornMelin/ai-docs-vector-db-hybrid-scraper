"""AI Security module for comprehensive AI threat protection."""

from .ai_security_service import (
    AISecurityConfig,
    AISecurityMetrics,
    AISecurityService,
    AIThreatType,
    SecurityLevel,
    ValidationResult,
)


# Compatibility alias
AISecurityValidator = AISecurityService

__all__ = [
    "AISecurityConfig",
    "AISecurityMetrics",
    "AISecurityService",
    "AISecurityValidator",  # Alias for compatibility
    "AIThreatType",
    "SecurityLevel",
    "ValidationResult",
]
