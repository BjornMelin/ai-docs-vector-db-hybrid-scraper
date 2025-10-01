"""FastMCP 2.0+ modular server composition."""

from .analytics_service import AnalyticsService
from .document_service import DocumentService
from .orchestrator_service import OrchestratorService
from .search_service import SearchService
from .system_service import SystemService


__all__ = [
    "AnalyticsService",
    "DocumentService",
    "OrchestratorService",
    "SearchService",
    "SystemService",
]
