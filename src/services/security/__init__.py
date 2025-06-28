#!/usr/bin/env python3
"""Comprehensive security framework for AI documentation system.

This package provides production-ready security components for the AI documentation
system, including:

- Distributed rate limiting with Redis backend and local fallback
- AI-specific security validation and prompt injection protection
- Comprehensive input validation and sanitization
- Security headers and CORS configuration
- Real-time security monitoring and threat detection
- Security event logging and audit trails
- Production deployment security best practices

Key Components:
- DistributedRateLimiter: Advanced rate limiting with Redis and fallback
- AISecurityValidator: AI-specific threat detection and prevention
- SecurityMiddleware: Comprehensive request/response security
- SecurityMonitor: Real-time monitoring and alerting
- SecurityManager: Centralized security management and integration

Usage:
    from src.services.security import setup_application_security
    
    # Setup comprehensive security for FastAPI app
    security_manager = setup_application_security(app, security_config)
"""

from src.services.security.rate_limiter import DistributedRateLimiter
from src.services.security.ai_security import (
    AISecurityValidator,
    SecurityThreat,
    ThreatLevel
)
from src.services.security.middleware import SecurityMiddleware
from src.services.security.monitoring import (
    SecurityMonitor,
    SecurityEvent,
    SecurityEventType,
    SecuritySeverity
)
from src.services.security.integration import (
    SecurityManager,
    security_manager,
    get_security_manager,
    setup_application_security
)

__all__ = [
    # Rate limiting
    "DistributedRateLimiter",
    
    # AI security
    "AISecurityValidator",
    "SecurityThreat", 
    "ThreatLevel",
    
    # Middleware
    "SecurityMiddleware",
    
    # Monitoring
    "SecurityMonitor",
    "SecurityEvent",
    "SecurityEventType", 
    "SecuritySeverity",
    
    # Integration
    "SecurityManager",
    "security_manager",
    "get_security_manager",
    "setup_application_security"
]