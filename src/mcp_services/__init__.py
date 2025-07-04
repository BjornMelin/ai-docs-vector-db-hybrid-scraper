"""FastMCP 2.0+ modular server composition.

This module implements the H1 research findings for server composition,
splitting the monolithic server into domain-specific services following
FastMCP 2.0+ best practices.
"""

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
