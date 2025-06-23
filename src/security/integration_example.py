
"""Example integration of simplified ML security with FastAPI.

This shows how to integrate the minimal security features into existing endpoints.
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi.security import APIKeyHeader

from src.config import get_config
from src.security.ml_security import MLSecurityValidator
from src.security.ml_security import SimpleRateLimiter

# Example of how to integrate with existing FastAPI app
security_router = APIRouter(prefix="/api/v1", tags=["ML Security Example"])

# Use existing API key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Verify API key - integrates with existing auth."""
    config = get_config()
    if not config.security.require_api_keys:
        return "no-auth-required"

    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")

    # In production, verify against database/cache
    # This is just an example
    if api_key != "valid-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


# Initialize security validator once
ml_security = MLSecurityValidator()
rate_limiter = SimpleRateLimiter()


@security_router.post("/embed")
async def secure_embed_endpoint(
    request: Request, data: dict, api_key: str = Depends(verify_api_key)
):
    """Example endpoint with minimal ML security.

    This shows how to add basic security to existing endpoints:
    1. API key authentication (existing)
    2. Input validation (new, simple)
    3. Rate limiting (delegated to nginx/cloudflare)
    """
    # Step 1: Basic input validation
    validation_result = ml_security.validate_input(
        data, expected_schema={"text": str, "model": str, "collection": str}
    )

    if not validation_result.passed:
        # Log security event
        ml_security.log_security_event(
            "input_validation_failed",
            {"reason": validation_result.message},
            severity=validation_result.severity,
        )
        raise HTTPException(status_code=400, detail=validation_result.message)

    # Step 2: Rate limiting (placeholder - use nginx in production)
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Step 3: Process request (existing logic)
    # ... existing embedding logic ...

    return {"status": "success", "message": "Request processed"}


@security_router.get("/security/status")
async def security_status(api_key: str = Depends(verify_api_key)):
    """Get security status - useful for monitoring.

    This endpoint can be called by monitoring systems to check security health.
    """
    # Run basic checks
    dependency_check = ml_security.check_dependencies()

    # Get summary
    summary = ml_security.get_security_summary()
    summary["dependency_status"] = dependency_check.passed

    return summary


# Example middleware for input validation (optional)
async def ml_security_middleware(request: Request, call_next):
    """Simple middleware for ML security.

    In production, this would be more sophisticated but still simple.
    """
    # Only check POST requests with JSON body
    if (
        request.method == "POST"
        and request.headers.get("content-type") == "application/json"
    ):
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1_000_000:
            raise HTTPException(status_code=413, detail="Request too large")

    response = await call_next(request)
    return response


# Startup task for dependency scanning
async def startup_security_check():
    """Run security checks on startup."""
    config = get_config()

    if config.security.dependency_scan_on_startup:
        result = ml_security.check_dependencies()
        if not result.passed:
            # Log warning but don't block startup
            ml_security.log_security_event(
                "startup_dependency_scan",
                {"vulnerabilities": result.details.get("count", 0)},
                severity="warning",
            )


# Integration with existing monitoring
def get_security_metrics() -> dict:
    """Get security metrics for Prometheus/Grafana.

    This integrates with existing monitoring infrastructure.
    """
    summary = ml_security.get_security_summary()

    # Convert to Prometheus-friendly metrics
    return {
        "ml_security_checks_total": summary["total_checks"],
        "ml_security_checks_failed": summary["failed"],
        "ml_security_critical_issues": summary["critical_issues"],
    }
