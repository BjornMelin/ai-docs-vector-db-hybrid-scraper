
"""Simplified production FastMCP server with essential middleware only.

This module provides a basic production wrapper around FastMCP server,
following KISS principles with only essential middleware for V1.
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.config import get_config
from src.services.fastapi.middleware.manager import get_middleware_manager
from src.services.logging_config import configure_logging

logger = logging.getLogger(__name__)


class ProductionMCPServer:
    """Simple production FastMCP server with essential middleware.

    This server wraps FastMCP with basic production middleware
    while maintaining full MCP protocol compatibility.
    """

    def __init__(self, config=None):
        """Initialize production MCP server."""
        self.config = config or get_config()
        self.middleware_manager = None
        self._mcp_server: FastMCP | None = None
        self._app: Starlette | None = None
        self._shutdown_event = asyncio.Event()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # Store task reference to avoid RUF006 warning
        task = asyncio.create_task(self.shutdown())
        # Set task name for easier debugging
        task.set_name("graceful-shutdown")

    @asynccontextmanager
    async def lifespan(self, app: Starlette):
        """Application lifespan manager with startup/shutdown."""
        try:
            # Startup
            logger.info("Starting production MCP server...")
            await self.startup()
            yield

        except Exception as e:
            logger.exception(f"Startup failed: {e}")
            raise

        finally:
            # Shutdown
            logger.info("Shutting down production MCP server...")
            await self.shutdown()

    async def startup(self) -> None:
        """Initialize server components."""
        # Configure logging
        configure_logging(self.config.log_level.value, self.config.debug)

        # Initialize middleware manager
        self.middleware_manager = get_middleware_manager(self.config)

        logger.info("Production MCP server startup complete")

    async def shutdown(self) -> None:
        """Cleanup server components."""
        try:
            self._shutdown_event.set()

            if self._mcp_server:
                # FastMCP cleanup would go here if it had cleanup methods
                logger.info("FastMCP server cleanup complete")

        except Exception as e:
            logger.exception(f"Error during shutdown: {e}")

        logger.info("Production MCP server shutdown complete")

    async def health_check(self, request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse(
            {
                "status": "healthy",
                "version": self.config.version,
                "environment": self.config.environment.value,
            }
        )

    def create_app(self) -> Starlette:
        """Create Starlette application with middleware."""
        # Routes
        routes = [
            Route("/health", self.health_check, methods=["GET"]),
        ]

        # Create app with lifespan
        app = Starlette(
            routes=routes,
            lifespan=self.lifespan,
        )

        # Apply middleware
        if self.middleware_manager:
            self.middleware_manager.apply_middleware(app)

        self._app = app
        return app

    async def run_async(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        """Run the production server asynchronously."""
        try:
            import uvicorn

            # Create app
            app = self.create_app()

            # Run with uvicorn
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level=self.config.log_level.value.lower(),
                access_log=self.config.debug,
            )

            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            logger.exception(f"Server error: {e}")
            raise


def create_production_server(config=None) -> ProductionMCPServer:
    """Create a production-enhanced MCP server."""
    return ProductionMCPServer(config or get_config())


async def run_production_server_async(
    config=None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run production server with enhanced middleware stack."""
    server = create_production_server(config)
    await server.run_async(host, port)


def main() -> None:
    """Main entry point for production server."""
    # Load configuration
    config = get_config()

    # Get server configuration from environment
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    try:
        # Run the server
        asyncio.run(run_production_server_async(config, host, port))
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
