"""Security audit logger implementation."""

import logging
from datetime import UTC, datetime
from typing import Any


logger = logging.getLogger(__name__)


class SecurityAuditLogger:
    """Logger for security audit events."""

    def __init__(self):
        """Initialize security audit logger."""
        self.events = []

    def log_security_event(
        self,
        event_type: str,
        user_id: str,
        resource: str,
        action: str,
        resource_id: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log a security event.

        Args:
            event_type: Type of security event
            user_id: User ID who triggered the event
            resource: Resource being accessed
            action: Action being performed
            resource_id: ID of the resource
            context: Additional context information
        """
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "resource_id": resource_id,
            "context": context or {},
        }

        self.events.append(event)
        logger.info(
            f"Security event: {event_type} - {user_id} - {resource}/{resource_id}"
        )

    def get_recent_events(self, limit: int = 100) -> list:
        """Get recent security events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent security events
        """
        return self.events[-limit:]
