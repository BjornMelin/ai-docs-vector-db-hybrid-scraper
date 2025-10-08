"""Shared infrastructure components to avoid circular imports.

This module contains classes that are used by multiple infrastructure components
to prevent circular import dependencies.
"""

import logging
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class ClientState(Enum):
    """Client connection state enumeration.

    Values:
        UNINITIALIZED: Client not yet initialized or connected
        HEALTHY: Client is connected and operating normally
        DEGRADED: Client is experiencing issues but partially functional
        FAILED: Client has failed and is not operational
    """

    UNINITIALIZED = "uninitialized"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class ClientHealth:
    """Client health status tracking.

    Attributes:
        state: Current connection state of the client
        last_check: Unix timestamp of last health check
        last_error: Description of the last error encountered, if any
        consecutive_failures: Number of consecutive failures for this client

    """

    state: ClientState
    last_check: float
    last_error: str | None = None
    consecutive_failures: int = 0
