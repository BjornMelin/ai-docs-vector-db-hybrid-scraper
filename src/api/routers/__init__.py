"""API routers package.

This package contains FastAPI routers for different API endpoints.
"""

from .config import router as config_router
from .v1 import cache as v1_cache, documents as v1_documents, search as v1_search


__all__ = ["config_router", "v1_cache", "v1_documents", "v1_search"]
