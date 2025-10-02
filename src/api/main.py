"""Main FastAPI application factory entrypoint."""

from src.api.app_profiles import detect_profile

from .app_factory import create_app


app = create_app(detect_profile())

__all__ = ["app"]
