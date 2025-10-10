"""Main FastAPI application factory entrypoint."""

from .app_factory import create_app


app = create_app()

__all__ = ["app"]
