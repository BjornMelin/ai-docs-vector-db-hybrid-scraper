"""Main FastAPI application for the AI Docs Vector DB Hybrid Scraper.

This module provides the main FastAPI application instance using the dual-mode
architecture that supports both simple mode (25K lines) and enterprise mode
(70K lines).

The application automatically detects the desired mode from the APPLICATION_MODE
environment variable and creates the appropriate FastAPI instance with all
necessary configurations, middleware, and services.

Environment Variables:
    APPLICATION_MODE: Set to "simple" or "enterprise" (default: "simple")

Usage:
    Run with uvicorn:
        uvicorn src.api.main:app --reload

    Or programmatically:
        from src.api.main import app
        # app is ready to use

Example:
    # Set mode via environment
    $ export APPLICATION_MODE=enterprise
    $ uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Note:
    The app instance is created at module import time, making it
    suitable for use with ASGI servers like uvicorn or gunicorn.
"""

from src.architecture.modes import get_current_mode

from .app_factory import create_app


# Detect current mode and create appropriate app
current_mode = get_current_mode()
app = create_app(current_mode)

__all__ = ["app"]
