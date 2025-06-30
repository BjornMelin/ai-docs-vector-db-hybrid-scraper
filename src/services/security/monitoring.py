#!/usr/bin/env python3
"""Security monitoring and logging for production deployment.

This module provides comprehensive security event monitoring, logging,
and alerting capabilities for the AI documentation system:

- Real-time security event detection and logging
- Structured logging for security incidents
- Integration with external monitoring systems
- Threat intelligence and correlation
- Security metrics and reporting
- Automated alerting for critical events
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import Request

from src.config.security import SecurityConfig


logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events that can be monitored."""

    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BLOCKED_REQUEST = "blocked_request"
    AI_THREAT_DETECTED = "ai_threat_detected"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    IP_BLOCKED = "ip_blocked"
    REQUEST_SUCCESS = "request_success"
    MIDDLEWARE_ERROR = "middleware_error"
    SECURITY_CONFIGURATION_CHANGE = "security_config_change"


class SecuritySeverity(Enum):
    """Security event severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Structured security event data."""

    event_type: SecurityEventType
    severity: SecuritySeverity
    timestamp: datetime
    client_ip: str
    user_agent: str
    endpoint: str
    method: str
    event_data: dict[str, Any]
    request_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    threat_indicators: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data


class SecurityMonitor:
    """Comprehensive security monitoring and logging system.

    This class provides real-time security monitoring capabilities including:
    - Structured logging of security events
    - Real-time threat detection and correlation
    - Integration with external monitoring systems
    - Automated alerting for critical security events
    - Security metrics collection and reporting
    - Compliance logging and audit trails
    """

    def __init__(
        self,
        security_config: SecurityConfig | None = None,
        log_dir: Path | None = None,
    ):
        """Initialize security monitor.

        Args:
            security_config: Security configuration
            log_dir: Directory for security logs
        """
        self.config = security_config or SecurityConfig()
        self.log_dir = log_dir or Path("logs/security")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize security-specific loggers
        self._setup_security_logging()

        # Event correlation and detection
        self.recent_events: list[SecurityEvent] = []
        self.ip_event_counts: dict[str, int] = {}
        self.threat_patterns: dict[str, int] = {}

        # Metrics
        self.metrics = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "blocked_requests": 0,
            "rate_limit_violations": 0,
            "ai_threats_detected": 0,
        }

        logger.info("Security monitor initialized")

    def _setup_security_logging(self) -> None:
        """Setup specialized security logging configuration."""
        # Security events logger
        self.security_logger = logging.getLogger("security_events")
        self.security_logger.setLevel(logging.INFO)

        # Security events file handler
        security_handler = logging.FileHandler(self.log_dir / "security_events.log")
        security_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        security_handler.setFormatter(security_formatter)
        self.security_logger.addHandler(security_handler)

        # Threat detection logger
        self.threat_logger = logging.getLogger("threat_detection")
        self.threat_logger.setLevel(logging.WARNING)

        threat_handler = logging.FileHandler(self.log_dir / "threats.log")
        threat_formatter = logging.Formatter(
            "%(asctime)s - THREAT - %(levelname)s - %(message)s"
        )
        threat_handler.setFormatter(threat_formatter)
        self.threat_logger.addHandler(threat_handler)

        # Audit logger for compliance
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)

        audit_handler = logging.FileHandler(self.log_dir / "audit.log")
        audit_formatter = logging.Formatter("%(asctime)s - AUDIT - %(message)s")
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)

    def log_security_event(
        self,
        event_type: str,
        request: Request | None = None,
        event_data: dict[str, Any] | None = None,
        severity: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Log a security event with comprehensive context.

        Args:
            event_type: type of security event
            request: HTTP request object (if available)
            event_data: Additional event data
            severity: Event severity level
            user_id: User ID if authenticated
        """
        try:
            # Determine event type enum
            try:
                event_type_enum = SecurityEventType(event_type)
            except ValueError:
                event_type_enum = SecurityEventType.SUSPICIOUS_ACTIVITY
                logger.warning(
                    f"Unknown event type: {event_type}, using SUSPICIOUS_ACTIVITY"
                )

            # Determine severity
            if severity:
                try:
                    severity_enum = SecuritySeverity(severity.lower())
                except ValueError:
                    severity_enum = SecuritySeverity.MEDIUM
            else:
                severity_enum = self._determine_severity(event_type_enum)

            # Extract request information
            if request:
                client_ip = self._get_client_ip(request)
                user_agent = request.headers.get("user-agent", "unknown")
                endpoint = request.url.path
                method = request.method
                request_id = request.headers.get("x-request-id")
            else:
                client_ip = "unknown"
                user_agent = "system"
                endpoint = "system"
                method = "SYSTEM"
                request_id = None

            # Create security event
            security_event = SecurityEvent(
                event_type=event_type_enum,
                severity=severity_enum,
                timestamp=datetime.now(UTC),
                client_ip=client_ip,
                user_agent=user_agent,
                endpoint=endpoint,
                method=method,
                event_data=event_data or {},
                request_id=request_id,
                user_id=user_id,
                threat_indicators=self._extract_threat_indicators(event_data or {}),
            )

            # Log the event
            self._write_security_event(security_event)

            # Update metrics
            self._update_metrics(security_event)

            # Perform real-time analysis
            self._analyze_event_patterns(security_event)

            # Check for automated response triggers
            self._check_automated_responses(security_event)

        except Exception as e:
            logger.exception("Failed to log security event")

    def _determine_severity(self, event_type: SecurityEventType) -> SecuritySeverity:
        """Determine severity based on event type.

        Args:
            event_type: type of security event

        Returns:
            Appropriate severity level
        """
        severity_mapping = {
            SecurityEventType.RATE_LIMIT_EXCEEDED: SecuritySeverity.MEDIUM,
            SecurityEventType.SUSPICIOUS_ACTIVITY: SecuritySeverity.HIGH,
            SecurityEventType.BLOCKED_REQUEST: SecuritySeverity.HIGH,
            SecurityEventType.AI_THREAT_DETECTED: SecuritySeverity.CRITICAL,
            SecurityEventType.AUTHENTICATION_FAILURE: SecuritySeverity.MEDIUM,
            SecurityEventType.AUTHORIZATION_FAILURE: SecuritySeverity.HIGH,
            SecurityEventType.INPUT_VALIDATION_FAILURE: SecuritySeverity.MEDIUM,
            SecurityEventType.IP_BLOCKED: SecuritySeverity.HIGH,
            SecurityEventType.REQUEST_SUCCESS: SecuritySeverity.INFO,
            SecurityEventType.MIDDLEWARE_ERROR: SecuritySeverity.MEDIUM,
            SecurityEventType.SECURITY_CONFIGURATION_CHANGE: SecuritySeverity.HIGH,
        }

        return severity_mapping.get(event_type, SecuritySeverity.MEDIUM)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers.

        Args:
            request: HTTP request

        Returns:
            Client IP address
        """
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fallback to direct client IP
        return getattr(request.client, "host", "unknown")

    def _extract_threat_indicators(self, event_data: dict[str, Any]) -> list[str]:
        """Extract threat indicators from event data.

        Args:
            event_data: Event data to analyze

        Returns:
            list of threat indicators
        """
        indicators = []

        # Check for common threat patterns
        if "sql_injection" in str(event_data).lower():
            indicators.append("sql_injection_pattern")

        if "xss" in str(event_data).lower():
            indicators.append("xss_pattern")

        if "prompt_injection" in str(event_data).lower():
            indicators.append("prompt_injection_pattern")

        # Check for automated tools
        user_agent = event_data.get("user_agent", "").lower()
        suspicious_agents = ["sqlmap", "nikto", "nmap", "gobuster"]
        if any(agent in user_agent for agent in suspicious_agents):
            indicators.append("automated_tool")

        # Check for excessive requests
        if "rate_limit" in str(event_data).lower():
            indicators.append("rate_limit_abuse")

        return indicators

    def _write_security_event(self, event: SecurityEvent) -> None:
        """Write security event to appropriate logs.

        Args:
            event: Security event to log
        """
        # Convert to JSON for structured logging
        event_json = json.dumps(event.to_dict(), default=str)

        # Log to appropriate logger based on severity
        if event.severity in (SecuritySeverity.CRITICAL, SecuritySeverity.HIGH):
            self.threat_logger.error(f"HIGH_PRIORITY_EVENT: {event_json}")

        # Always log to security events
        self.security_logger.info(f"SECURITY_EVENT: {event_json}")

        # Log to audit trail for compliance
        if event.event_type in (
            SecurityEventType.AUTHENTICATION_FAILURE,
            SecurityEventType.AUTHORIZATION_FAILURE,
            SecurityEventType.IP_BLOCKED,
            SecurityEventType.SECURITY_CONFIGURATION_CHANGE,
        ):
            self.audit_logger.info(f"AUDIT_EVENT: {event_json}")

    def _update_metrics(self, event: SecurityEvent) -> None:
        """Update security metrics with new event.

        Args:
            event: Security event to count
        """
        self.metrics["total_events"] += 1

        # Count by event type
        event_type = event.event_type.value
        self.metrics["events_by_type"][event_type] = (
            self.metrics["events_by_type"].get(event_type, 0) + 1
        )

        # Count by severity
        severity = event.severity.value
        self.metrics["events_by_severity"][severity] = (
            self.metrics["events_by_severity"].get(severity, 0) + 1
        )

        # Specific metrics
        if event.event_type == SecurityEventType.BLOCKED_REQUEST:
            self.metrics["blocked_requests"] += 1
        elif event.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED:
            self.metrics["rate_limit_violations"] += 1
        elif event.event_type == SecurityEventType.AI_THREAT_DETECTED:
            self.metrics["ai_threats_detected"] += 1

    def _analyze_event_patterns(self, event: SecurityEvent) -> None:
        """Analyze event patterns for threat correlation.

        Args:
            event: Security event to analyze
        """
        # Add to recent events (keep last 1000)
        self.recent_events.append(event)
        if len(self.recent_events) > 1000:
            self.recent_events = self.recent_events[-1000:]

        # Track IP-based patterns
        ip = event.client_ip
        self.ip_event_counts[ip] = self.ip_event_counts.get(ip, 0) + 1

        # Track threat patterns
        for indicator in event.threat_indicators or []:
            self.threat_patterns[indicator] = self.threat_patterns.get(indicator, 0) + 1

        # Detect suspicious patterns
        self._detect_attack_patterns(event)

    def _detect_attack_patterns(self, event: SecurityEvent) -> None:
        """Detect potential attack patterns from event correlation.

        Args:
            event: Latest security event
        """
        current_time = time.time()

        # Check for IP-based attacks (multiple events from same IP)
        ip_events = [
            e
            for e in self.recent_events[-100:]  # Last 100 events
            if e.client_ip == event.client_ip
            and (current_time - e.timestamp.timestamp()) < 300  # Last 5 minutes
        ]

        if len(ip_events) > 10:  # More than 10 events in 5 minutes
            self.log_security_event(
                "coordinated_attack_detected",
                event_data={
                    "attack_ip": event.client_ip,
                    "event_count": len(ip_events),
                    "time_window": "5_minutes",
                    "attack_types": list({e.event_type.value for e in ip_events}),
                },
                severity="critical",
            )

        # Check for distributed attacks (same patterns from multiple IPs)
        recent_high_severity = [
            e
            for e in self.recent_events[-50:]
            if e.severity in (SecuritySeverity.HIGH, SecuritySeverity.CRITICAL)
            and (current_time - e.timestamp.timestamp()) < 600  # Last 10 minutes
        ]

        if len(recent_high_severity) > 5:
            unique_ips = {e.client_ip for e in recent_high_severity}
            if len(unique_ips) > 3:  # Attacks from multiple IPs
                self.log_security_event(
                    "distributed_attack_detected",
                    event_data={
                        "unique_ips": len(unique_ips),
                        "total_events": len(recent_high_severity),
                        "time_window": "10_minutes",
                    },
                    severity="critical",
                )

    def _check_automated_responses(self, event: SecurityEvent) -> None:
        """Check if automated security responses should be triggered.

        Args:
            event: Security event to evaluate
        """
        # Auto-block IPs with critical threats
        if event.severity == SecuritySeverity.CRITICAL:
            if event.event_type in (
                SecurityEventType.AI_THREAT_DETECTED,
                SecurityEventType.SUSPICIOUS_ACTIVITY,
            ):
                logger.critical(
                    f"Critical security event detected - consider IP blocking: {event.client_ip}",
                    extra={
                        "event_type": event.event_type.value,
                        "client_ip": event.client_ip,
                        "endpoint": event.endpoint,
                        "event_data": event.event_data,
                    },
                )

        # Alert on repeated threats from same IP
        recent_ip_events = [
            e
            for e in self.recent_events[-20:]
            if e.client_ip == event.client_ip
            and e.severity in (SecuritySeverity.HIGH, SecuritySeverity.CRITICAL)
        ]

        if len(recent_ip_events) >= 3:
            logger.critical(
                f"Repeated security violations from IP {event.client_ip} - automatic blocking recommended",
                extra={
                    "violation_count": len(recent_ip_events),
                    "client_ip": event.client_ip,
                    "recent_events": [e.event_type.value for e in recent_ip_events],
                },
            )

    def log_rate_limit_violation(self, request: Request, limit: int) -> None:
        """Log rate limit violation with specific context.

        Args:
            request: HTTP request that exceeded rate limit
            limit: Rate limit that was exceeded
        """
        self.log_security_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED.value,
            request,
            {
                "rate_limit": limit,
                "endpoint": request.url.path,
                "method": request.method,
                "user_agent": request.headers.get("user-agent", "unknown"),
            },
            severity=SecuritySeverity.MEDIUM.value,
        )

    def log_suspicious_activity(
        self, activity_type: str, details: dict[str, Any]
    ) -> None:
        """Log suspicious activity with details.

        Args:
            activity_type: type of suspicious activity
            details: Activity details
        """
        self.log_security_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY.value,
            event_data={"activity_type": activity_type, "details": details},
            severity=SecuritySeverity.HIGH.value,
        )

    def log_request_success(self, request: Request, processing_time: float) -> None:
        """Log successful request for baseline metrics.

        Args:
            request: Successful HTTP request
            processing_time: Request processing time in seconds
        """
        # Only log sample of successful requests to avoid log flooding
        if hash(request.url.path) % 100 == 0:  # Log 1% of successful requests
            self.log_security_event(
                SecurityEventType.REQUEST_SUCCESS.value,
                request,
                {
                    "processing_time": processing_time,
                    "endpoint": request.url.path,
                    "method": request.method,
                },
                severity=SecuritySeverity.INFO.value,
            )

    def get_security_metrics(self) -> dict[str, Any]:
        """Get current security metrics.

        Returns:
            Dictionary with security metrics
        """
        current_time = time.time()

        # Calculate recent event rates
        recent_events_1h = [
            e
            for e in self.recent_events
            if (current_time - e.timestamp.timestamp()) < 3600
        ]

        recent_events_24h = [
            e
            for e in self.recent_events
            if (current_time - e.timestamp.timestamp()) < 86400
        ]

        return {
            "total_events": self.metrics["total_events"],
            "events_by_type": self.metrics["events_by_type"],
            "events_by_severity": self.metrics["events_by_severity"],
            "blocked_requests": self.metrics["blocked_requests"],
            "rate_limit_violations": self.metrics["rate_limit_violations"],
            "ai_threats_detected": self.metrics["ai_threats_detected"],
            "recent_events_1h": len(recent_events_1h),
            "recent_events_24h": len(recent_events_24h),
            "top_threat_patterns": dict(
                sorted(self.threat_patterns.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ),
            "top_source_ips": dict(
                sorted(self.ip_event_counts.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ),
        }

    def get_threat_report(self, hours: int = 24) -> dict[str, Any]:
        """Generate comprehensive threat report.

        Args:
            hours: Number of hours to include in report

        Returns:
            Threat analysis report
        """
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)

        # Filter events by time window
        relevant_events = [
            e for e in self.recent_events if e.timestamp.timestamp() > cutoff_time
        ]

        # Analyze threats
        high_severity_events = [
            e
            for e in relevant_events
            if e.severity in (SecuritySeverity.HIGH, SecuritySeverity.CRITICAL)
        ]

        threat_types = {}
        source_ips = {}
        target_endpoints = {}

        for event in high_severity_events:
            # Count threat types
            event_type = event.event_type.value
            threat_types[event_type] = threat_types.get(event_type, 0) + 1

            # Count source IPs
            source_ips[event.client_ip] = source_ips.get(event.client_ip, 0) + 1

            # Count target endpoints
            target_endpoints[event.endpoint] = (
                target_endpoints.get(event.endpoint, 0) + 1
            )

        return {
            "report_period_hours": hours,
            "total_events": len(relevant_events),
            "high_severity_events": len(high_severity_events),
            "threat_types": threat_types,
            "top_source_ips": dict(
                sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "targeted_endpoints": dict(
                sorted(target_endpoints.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "threat_indicators": dict(
                sorted(self.threat_patterns.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ),
            "generated_at": datetime.now(UTC).isoformat(),
        }

    def export_security_logs(self, hours: int = 24, format: str = "json") -> str:
        """Export security logs for external analysis.

        Args:
            hours: Number of hours to export
            format: Export format (json, csv)

        Returns:
            Exported log data as string
        """
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)

        # Filter events by time window
        relevant_events = [
            e for e in self.recent_events if e.timestamp.timestamp() > cutoff_time
        ]

        if format.lower() == "json":
            return json.dumps(
                [e.to_dict() for e in relevant_events], indent=2, default=str
            )
        if format.lower() == "csv":
            # Simple CSV export
            lines = ["timestamp,event_type,severity,client_ip,endpoint,method"]
            for event in relevant_events:
                lines.append(
                    f"{event.timestamp.isoformat()},{event.event_type.value},"
                    f"{event.severity.value},{event.client_ip},{event.endpoint},{event.method}"
                )
            return "\n".join(lines)
        msg = f"Unsupported export format: {format}"
        raise ValueError(msg)

    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old security data to manage memory usage.

        Args:
            days: Number of days to retain data

        Returns:
            Number of events cleaned up
        """
        current_time = time.time()
        cutoff_time = current_time - (days * 86400)

        # Keep only recent events
        old_count = len(self.recent_events)
        self.recent_events = [
            e for e in self.recent_events if e.timestamp.timestamp() > cutoff_time
        ]
        new_count = len(self.recent_events)

        # Clean up IP counts (keep only IPs with recent activity)
        active_ips = {e.client_ip for e in self.recent_events}
        old_ip_counts = self.ip_event_counts.copy()
        self.ip_event_counts = {
            ip: count for ip, count in old_ip_counts.items() if ip in active_ips
        }

        cleaned_count = old_count - new_count
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old security events")

        return cleaned_count
