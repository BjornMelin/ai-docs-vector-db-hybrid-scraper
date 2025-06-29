"""Configuration Observability API.

FastAPI endpoints for configuration automation system integration.
Provides RESTful API access to drift detection, validation, optimization,
and real-time monitoring capabilities.

Endpoints:
- GET /api/v1/config/status - System status
- GET /api/v1/config/health - Health check
- POST /api/v1/config/drift/check - Trigger drift check
- POST /api/v1/config/validate - Validate configurations
- POST /api/v1/config/optimize - Generate optimizations
- GET /api/v1/config/report - Detailed report
- POST /api/v1/config/remediate - Auto-remediate issues
- WebSocket /api/v1/config/ws - Real-time updates
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .automation import (
    ConfigDrift,
    ConfigObservabilityAutomation,
    ConfigValidationResult,
    OptimizationRecommendation,
    get_automation_system,
)


logger = logging.getLogger(__name__)


# Request/Response Models
class DriftCheckRequest(BaseModel):
    """Request model for drift checking."""

    environment: str | None = Field(None, description="Specific environment to check")
    auto_fix: bool = Field(False, description="Apply automatic fixes")


class DriftCheckResponse(BaseModel):
    """Response model for drift checking."""

    drifts_detected: int
    critical_drifts: int
    auto_fixes_applied: int
    drifts: list[dict]
    remediation_results: dict[str, bool] | None = None


class ValidationRequest(BaseModel):
    """Request model for configuration validation."""

    environment: str | None = Field(
        None, description="Specific environment to validate"
    )
    fix_issues: bool = Field(False, description="Attempt to fix validation issues")


class ValidationResponse(BaseModel):
    """Response model for configuration validation."""

    total_checks: int
    errors: int
    warnings: int
    critical_issues: int
    results: list[dict]


class OptimizationRequest(BaseModel):
    """Request model for optimization generation."""

    environment: str | None = Field(
        None, description="Specific environment to optimize"
    )
    apply_recommendations: bool = Field(
        False, description="Apply optimization recommendations"
    )


class OptimizationResponse(BaseModel):
    """Response model for optimization."""

    recommendations_count: int
    high_confidence_count: int
    expected_improvements: list[str]
    recommendations: list[dict]


class RemediationRequest(BaseModel):
    """Request model for remediation."""

    drift_parameters: list[str] = Field(description="Specific parameters to remediate")
    force: bool = Field(False, description="Force remediation even for risky changes")


class RemediationResponse(BaseModel):
    """Response model for remediation."""

    remediated_count: int
    failed_count: int
    results: dict[str, bool]


class SystemStatusResponse(BaseModel):
    """Response model for system status."""

    automation_enabled: bool
    auto_remediation_enabled: bool
    file_monitoring_active: bool
    environments_monitored: int
    last_drift_check: str
    last_optimization_check: str
    drift_analysis: dict
    validation_status: dict
    optimization: dict
    environments: dict


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    automation_system: str
    uptime_seconds: float
    environments: list[str]
    last_check: str


# WebSocket Manager
class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.automation_system: ConfigObservabilityAutomation | None = None
        self._update_task: asyncio.Task | None = None

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

        # Start update task if not already running
        if not self._update_task or self._update_task.done():
            self._update_task = asyncio.create_task(self._periodic_updates())

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        # Remove disconnected clients
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def _periodic_updates(self):
        """Send periodic updates to connected clients."""
        while self.active_connections:
            try:
                if not self.automation_system:
                    self.automation_system = get_automation_system()

                # Get current status
                status = self.automation_system.get_system_status()

                # Broadcast update
                await self.broadcast(
                    {
                        "type": "status_update",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "data": status,
                    }
                )

                # Wait before next update
                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.exception(f"Error in periodic WebSocket updates: {e}")
                await asyncio.sleep(60)  # Wait longer on error


# Global WebSocket manager
websocket_manager = WebSocketManager()


# Create API router
router = APIRouter(prefix="/api/v1/config", tags=["Configuration Automation"])


def get_automation_system_dependency() -> ConfigObservabilityAutomation:
    """Dependency to get automation system instance."""
    try:
        return get_automation_system()
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Automation system unavailable: {e}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    automation_system: ConfigObservabilityAutomation = Depends(
        get_automation_system_dependency
    ),
):
    """Health check endpoint."""
    try:
        status = automation_system.get_system_status()

        return HealthResponse(
            status="healthy",
            automation_system="active",
            uptime_seconds=0.0,  # Would calculate actual uptime
            environments=status["environments"]["detected"],
            last_check=datetime.now(UTC).isoformat(),
        )
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    detailed: bool = False,
    automation_system: ConfigObservabilityAutomation = Depends(
        get_automation_system_dependency
    ),
):
    """Get comprehensive system status."""
    try:
        if detailed:
            status = automation_system.get_detailed_report()
        else:
            status = automation_system.get_system_status()

        return SystemStatusResponse(
            automation_enabled=status["system_status"]["automation_enabled"],
            auto_remediation_enabled=status["system_status"][
                "auto_remediation_enabled"
            ],
            file_monitoring_active=status["system_status"]["file_monitoring_active"],
            environments_monitored=status["system_status"]["environments_monitored"],
            last_drift_check=status["system_status"]["last_drift_check"],
            last_optimization_check=status["system_status"]["last_optimization_check"],
            drift_analysis=status["drift_analysis"],
            validation_status=status["validation_status"],
            optimization=status["optimization"],
            environments=status["environments"],
        )
    except Exception as e:
        logger.exception(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.post("/drift/check", response_model=DriftCheckResponse)
async def check_configuration_drift(
    request: DriftCheckRequest,
    automation_system: ConfigObservabilityAutomation = Depends(
        get_automation_system_dependency
    ),
):
    """Check for configuration drift."""
    try:
        logger.info(
            f"Starting drift check for environment: {request.environment or 'all'}"
        )

        # Detect drift
        drifts = await automation_system.detect_configuration_drift()

        # Filter by environment if specified
        if request.environment:
            drifts = [d for d in drifts if d.environment == request.environment]

        # Count critical drifts
        critical_drifts = len(
            [d for d in drifts if d.severity.value in ["critical", "fatal"]]
        )

        # Convert drifts to dict format
        drift_dicts = []
        for drift in drifts:
            drift_dicts.append(
                {
                    "parameter": drift.parameter,
                    "severity": drift.severity.value,
                    "environment": drift.environment,
                    "expected_value": drift.expected_value,
                    "current_value": drift.current_value,
                    "impact_score": drift.impact_score,
                    "auto_fix_available": drift.auto_fix_available,
                    "timestamp": drift.timestamp.isoformat(),
                }
            )

        # Auto-remediate if requested
        remediation_results = None
        auto_fixes_applied = 0

        if request.auto_fix and drifts:
            remediation_results = await automation_system.auto_remediate_issues(drifts)
            auto_fixes_applied = sum(
                1 for success in remediation_results.values() if success
            )

        response = DriftCheckResponse(
            drifts_detected=len(drifts),
            critical_drifts=critical_drifts,
            auto_fixes_applied=auto_fixes_applied,
            drifts=drift_dicts,
            remediation_results=remediation_results,
        )

        # Broadcast update via WebSocket
        await websocket_manager.broadcast(
            {
                "type": "drift_check_completed",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {
                    "drifts_detected": len(drifts),
                    "critical_drifts": critical_drifts,
                    "environment": request.environment,
                },
            }
        )

        return response

    except Exception as e:
        logger.exception(f"Error checking configuration drift: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to check configuration drift"
        )


@router.post("/validate", response_model=ValidationResponse)
async def validate_configurations(
    request: ValidationRequest,
    automation_system: ConfigObservabilityAutomation = Depends(
        get_automation_system_dependency
    ),
):
    """Validate configuration health."""
    try:
        logger.info(
            f"Starting configuration validation for environment: {request.environment or 'all'}"
        )

        # Run validation
        validation_results = await automation_system.validate_configuration_health()

        # Filter by environment if specified
        if request.environment:
            validation_results = [
                v for v in validation_results if v.environment == request.environment
            ]

        # Count by status
        errors = len([v for v in validation_results if v.status.value == "error"])
        warnings = len([v for v in validation_results if v.status.value == "warning"])
        critical_issues = len(
            [v for v in validation_results if v.status.value == "critical"]
        )

        # Convert to dict format
        result_dicts = []
        for result in validation_results:
            result_dicts.append(
                {
                    "parameter": result.parameter,
                    "status": result.status.value,
                    "message": result.message,
                    "environment": result.environment,
                    "suggestions": result.suggestions,
                    "timestamp": result.timestamp.isoformat(),
                }
            )

        response = ValidationResponse(
            total_checks=len(validation_results),
            errors=errors,
            warnings=warnings,
            critical_issues=critical_issues,
            results=result_dicts,
        )

        # Broadcast update via WebSocket
        await websocket_manager.broadcast(
            {
                "type": "validation_completed",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {
                    "total_checks": len(validation_results),
                    "errors": errors,
                    "warnings": warnings,
                    "critical_issues": critical_issues,
                    "environment": request.environment,
                },
            }
        )

        return response

    except Exception as e:
        logger.exception(f"Error validating configurations: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate configurations")


@router.post("/optimize", response_model=OptimizationResponse)
async def generate_optimizations(
    request: OptimizationRequest,
    automation_system: ConfigObservabilityAutomation = Depends(
        get_automation_system_dependency
    ),
):
    """Generate configuration optimization recommendations."""
    try:
        logger.info(
            f"Generating optimizations for environment: {request.environment or 'all'}"
        )

        # Generate recommendations
        recommendations = (
            await automation_system.generate_optimization_recommendations()
        )

        # Filter by environment if specified
        if request.environment:
            recommendations = [
                r for r in recommendations if r.environment == request.environment
            ]

        # Count high confidence recommendations
        high_confidence_count = len(
            [r for r in recommendations if r.confidence_score > 0.8]
        )

        # Extract expected improvements
        expected_improvements = list({r.expected_improvement for r in recommendations})

        # Convert to dict format
        rec_dicts = []
        for rec in recommendations:
            rec_dicts.append(
                {
                    "parameter": rec.parameter,
                    "current_value": rec.current_value,
                    "recommended_value": rec.recommended_value,
                    "expected_improvement": rec.expected_improvement,
                    "confidence_score": rec.confidence_score,
                    "performance_impact": rec.performance_impact,
                    "environment": rec.environment,
                    "reasoning": rec.reasoning,
                }
            )

        response = OptimizationResponse(
            recommendations_count=len(recommendations),
            high_confidence_count=high_confidence_count,
            expected_improvements=expected_improvements,
            recommendations=rec_dicts,
        )

        # Broadcast update via WebSocket
        await websocket_manager.broadcast(
            {
                "type": "optimization_completed",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {
                    "recommendations_count": len(recommendations),
                    "high_confidence_count": high_confidence_count,
                    "environment": request.environment,
                },
            }
        )

        return response

    except Exception as e:
        logger.exception(f"Error generating optimizations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate optimizations")


@router.post("/remediate", response_model=RemediationResponse)
async def remediate_issues(
    request: RemediationRequest,
    automation_system: ConfigObservabilityAutomation = Depends(
        get_automation_system_dependency
    ),
):
    """Remediate specific configuration issues."""
    try:
        logger.info(f"Remediating issues for parameters: {request.drift_parameters}")

        # Get current drifts
        all_drifts = await automation_system.detect_configuration_drift()

        # Filter to requested parameters
        target_drifts = [
            d for d in all_drifts if d.parameter in request.drift_parameters
        ]

        if not target_drifts:
            return RemediationResponse(
                remediated_count=0,
                failed_count=0,
                results={},
            )

        # Apply remediation
        results = await automation_system.auto_remediate_issues(target_drifts)

        remediated_count = sum(1 for success in results.values() if success)
        failed_count = len(results) - remediated_count

        response = RemediationResponse(
            remediated_count=remediated_count,
            failed_count=failed_count,
            results=results,
        )

        # Broadcast update via WebSocket
        await websocket_manager.broadcast(
            {
                "type": "remediation_completed",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {
                    "remediated_count": remediated_count,
                    "failed_count": failed_count,
                    "parameters": request.drift_parameters,
                },
            }
        )

        return response

    except Exception as e:
        logger.exception(f"Error remediating issues: {e}")
        raise HTTPException(status_code=500, detail="Failed to remediate issues")


@router.get("/report")
async def get_detailed_report(
    format: str = "json",
    automation_system: ConfigObservabilityAutomation = Depends(
        get_automation_system_dependency
    ),
):
    """Get detailed automation system report."""
    try:
        report = automation_system.get_detailed_report()

        # Add report metadata
        report["report_metadata"] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "format": format,
            "automation_version": "1.0.0",
        }

        if format.lower() == "yaml":
            import yaml

            return yaml.dump(report, default_flow_style=False, indent=2)
        return report

    except Exception as e:
        logger.exception(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)

    try:
        while True:
            # Wait for client messages (ping/pong, etc.)
            data = await websocket.receive_text()

            # Handle client commands
            try:
                message = json.loads(data)
                command = message.get("command")

                if command == "get_status":
                    automation_system = get_automation_system()
                    status = automation_system.get_system_status()

                    await websocket.send_json(
                        {
                            "type": "status_response",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "data": status,
                        }
                    )

                elif command == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

            except json.JSONDecodeError:
                # Ignore non-JSON messages
                pass

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


# Include router in main FastAPI app
def setup_config_automation_api(app):
    """Setup configuration automation API routes."""
    app.include_router(router)

    # Add startup event to initialize automation system
    @app.on_event("startup")
    async def startup_automation():
        """Start automation system on app startup."""
        try:
            from .automation import start_automation_system

            await start_automation_system(
                enable_auto_remediation=False,  # Default to disabled in production
                enable_performance_optimization=True,
            )
            logger.info("Configuration automation system started")
        except Exception as e:
            logger.exception(f"Failed to start automation system: {e}")

    @app.on_event("shutdown")
    async def shutdown_automation():
        """Stop automation system on app shutdown."""
        try:
            from .automation import stop_automation_system

            await stop_automation_system()
            logger.info("Configuration automation system stopped")
        except Exception as e:
            logger.exception(f"Failed to stop automation system: {e}")
