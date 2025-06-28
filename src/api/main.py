"""Main FastAPI application for the AI Docs Vector DB Hybrid Scraper.

This module provides the main FastAPI application instance using the dual-mode architecture
that supports both simple mode (25K lines) and enterprise mode (70K lines).
"""

from src.architecture.modes import get_current_mode

from .app_factory import create_app


# Detect current mode and create appropriate app
current_mode = get_current_mode()
app = create_app(current_mode)

__all__ = ["app"]
