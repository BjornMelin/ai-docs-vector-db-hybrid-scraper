"""FastMCP 2.0+ modular server composition.

This module implements the H1 research findings for server composition,
splitting the monolithic server into domain-specific services following
FastMCP 2.0+ best practices.
"""

from .search_service import SearchService
from .document_service import DocumentService  
from .analytics_service import AnalyticsService
from .system_service import SystemService
from .orchestrator_service import OrchestratorService

__all__ = [
    "SearchService",
    "DocumentService", 
    "AnalyticsService",
    "SystemService",
    "OrchestratorService",
]
