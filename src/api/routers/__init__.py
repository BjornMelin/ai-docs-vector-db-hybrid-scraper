"""API routers package.

This package contains FastAPI routers for different API endpoints.
"""

from .config import router as config_router

__all__ = ["config_router"]