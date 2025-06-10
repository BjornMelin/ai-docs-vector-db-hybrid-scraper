"""Production-enhanced FastMCP server with comprehensive middleware and monitoring.

This module provides a production-ready wrapper around the existing FastMCP server,
adding enterprise-grade middleware, monitoring, and operational capabilities.
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from src.config.fastapi import FastAPIProductionConfig
from src.config.fastapi import get_fastapi_config
from src.services.fastapi.background import cleanup_task_manager
from src.services.fastapi.background import get_task_manager
from src.services.fastapi.background import initialize_task_manager
from src.services.fastapi.dependencies import cleanup_dependencies
from src.services.fastapi.dependencies import get_health_checker
from src.services.fastapi.dependencies import initialize_dependencies
from src.services.fastapi.middleware.manager import create_middleware_manager
from src.services.logging_config import configure_logging
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


class ProductionMCPServer:
    """Production-enhanced FastMCP server with comprehensive middleware stack.

    This server wraps the existing FastMCP server with production-grade
    middleware, monitoring, and operational capabilities while maintaining
    full compatibility with the MCP protocol.
    """

    def __init__(self, config: FastAPIProductionConfig | None = None):
        """Initialize production MCP server.

        Args:
            config: FastAPI production configuration
        """
        self.config = config or get_fastapi_config()
        self.middleware_manager = None
        self._mcp_server: FastMCP | None = None
        self._app: Starlette | None = None
        self._shutdown_event = asyncio.Event()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(self._handle_shutdown(s))
                )

    async def _handle_shutdown(self, sig):
        """Handle shutdown signal.

        Args:
            sig: Signal number
        """
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        self._shutdown_event.set()

    def create_health_routes(self) -> list:
        """Create health check and monitoring routes.

        Returns:
            List of Starlette routes
        """

        async def health_check(request):
            """Health check endpoint."""
            try:
                health_checker = get_health_checker()
                health_status = await health_checker.check_health()

                # Add middleware health information
                if self.middleware_manager:
                    middleware_health = self.middleware_manager.get_health_status()
                    health_status["middleware"] = middleware_health

                # Add task manager statistics
                task_manager = get_task_manager()
                if task_manager:
                    health_status["background_tasks"] = task_manager.get_statistics()

                status_code = 200
                if health_status["status"] == "unhealthy":
                    status_code = 503
                elif health_status["status"] == "degraded":
                    status_code = 200  # Still serving requests

                return JSONResponse(health_status, status_code=status_code)

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse(
                    {"status": "unhealthy", "error": str(e)}, status_code=503
                )

        async def metrics(request):
            """Metrics endpoint for monitoring."""
            try:
                metrics_data = {}

                # Performance metrics
                if self.middleware_manager:
                    performance_metrics = (
                        self.middleware_manager.get_performance_metrics()
                    )
                    if performance_metrics:
                        metrics_data["performance"] = performance_metrics

                    # Circuit breaker metrics
                    circuit_stats = self.middleware_manager.get_circuit_breaker_stats()
                    if circuit_stats:
                        metrics_data["circuit_breakers"] = circuit_stats

                    # Middleware info
                    metrics_data["middleware"] = (
                        self.middleware_manager.get_middleware_info()
                    )

                # Task manager metrics
                task_manager = get_task_manager()
                if task_manager:
                    metrics_data["background_tasks"] = task_manager.get_statistics()

                return JSONResponse(metrics_data)

            except Exception as e:
                logger.error(f"Metrics endpoint failed: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        async def reset_metrics(request):
            """Reset metrics endpoint (for testing/maintenance)."""
            try:
                # Reset middleware metrics
                if self.middleware_manager:
                    self.middleware_manager.reset_middleware_state()

                return JSONResponse({"message": "Metrics reset successfully"})

            except Exception as e:
                logger.error(f"Metrics reset failed: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        return [
            Route("/health", health_check, methods=["GET"]),
            Route("/metrics", metrics, methods=["GET"]),
            Route("/admin/reset-metrics", reset_metrics, methods=["POST"]),
        ]

    @asynccontextmanager
    async def lifespan(self, app: Starlette):
        """Production server lifespan management."""
        try:
            logger.info("Starting production MCP server initialization...")

            # Initialize dependencies
            await initialize_dependencies()
            logger.info("Dependencies initialized")

            # Initialize background task manager
            await initialize_task_manager(
                max_workers=self.config.workers, max_queue_size=1000
            )
            logger.info("Background task manager initialized")

            # Create and configure middleware manager
            self.middleware_manager = create_middleware_manager(self.config)
            logger.info("Middleware manager created")

            # Initialize the existing MCP server with its lifespan
            from src.unified_mcp_server import lifespan as mcp_lifespan
            from src.unified_mcp_server import mcp

            self._mcp_server = mcp

            # Start the MCP server lifespan
            async with mcp_lifespan():
                logger.info("Production MCP server initialization complete")
                yield

        except Exception as e:
            logger.error(f"Production server initialization failed: {e}")
            raise
        finally:
            # Cleanup
            logger.info("Starting production server cleanup...")

            try:
                # Stop background task manager
                await cleanup_task_manager()
                logger.info("Background task manager cleaned up")

                # Cleanup dependencies
                await cleanup_dependencies()
                logger.info("Dependencies cleaned up")

            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

            logger.info("Production server cleanup complete")

    def create_app(self) -> Starlette:
        """Create the production Starlette application.

        Returns:
            Configured Starlette application
        """
        # Get middleware stack
        middleware_stack = []
        if self.middleware_manager:
            middleware_stack = self.middleware_manager.get_middleware_stack()

        # Create health and monitoring routes
        routes = self.create_health_routes()

        # Create Starlette app with middleware
        app = Starlette(
            middleware=middleware_stack, routes=routes, lifespan=self.lifespan
        )

        self._app = app
        return app

    async def run_production_server(
        self, host: str = "127.0.0.1", port: int = 8000, **kwargs
    ):
        """Run the production server with enhanced capabilities.

        Args:
            host: Server host
            port: Server port
            **kwargs: Additional server configuration
        """
        import uvicorn

        # Configure logging for production
        configure_logging()

        logger.info(f"Starting production MCP server on {host}:{port}")
        logger.info(f"Environment: {self.config.environment.value}")
        logger.info(f"Debug mode: {self.config.debug}")

        # Create the application
        app = self.create_app()

        # Production uvicorn configuration
        uvicorn_config = {
            "host": host,
            "port": port,
            "log_level": "info",
            "access_log": True,
            "server_header": False,  # Security: don't expose server info
            "date_header": False,  # Security: don't expose date info
            **kwargs,
        }

        # Adjust configuration based on environment
        if self.config.is_production():
            uvicorn_config.update(
                {
                    "workers": self.config.workers,
                    "log_level": "warning",
                    "access_log": False,  # Use middleware logging instead
                }
            )
        elif self.config.debug:
            uvicorn_config.update(
                {
                    "reload": True,
                    "log_level": "debug",
                }
            )

        # Run the server
        server = uvicorn.Server(uvicorn.Config(app, **uvicorn_config))

        try:
            await server.serve()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise

    def run_mcp_protocol(self, transport: str = "streamable-http", **kwargs):
        """Run the original MCP protocol server with production enhancements.

        This method runs the FastMCP server directly while still benefiting
        from the production middleware stack.

        Args:
            transport: MCP transport type
            **kwargs: Additional MCP server configuration
        """
        from src.unified_mcp_server import mcp

        # Apply middleware to the FastMCP server's underlying app
        if hasattr(mcp, "_app") and self.middleware_manager:
            self.middleware_manager.configure_app(mcp._app)

        # Run the original MCP server
        mcp.run(transport=transport, **kwargs)


def create_production_server(
    config: FastAPIProductionConfig | None = None,
) -> ProductionMCPServer:
    """Create a production-enhanced MCP server.

    Args:
        config: FastAPI production configuration

    Returns:
        Production MCP server instance
    """
    return ProductionMCPServer(config)


async def run_production_server_async(
    config: FastAPIProductionConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    **kwargs,
):
    """Run production server asynchronously.

    Args:
        config: FastAPI production configuration
        host: Server host
        port: Server port
        **kwargs: Additional server configuration
    """
    server = create_production_server(config)
    await server.run_production_server(host, port, **kwargs)


def main():
    """Main entry point for production server."""
    # Load configuration
    config = get_fastapi_config()

    # Get server configuration from environment
    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    transport = os.getenv("FASTMCP_TRANSPORT", "production")

    # Create production server
    server = create_production_server(config)

    if transport == "production":
        # Run full production server with Starlette/Uvicorn
        logger.info("Running production server with full middleware stack")
        asyncio.run(server.run_production_server(host, port))
    else:
        # Run MCP protocol server with production enhancements
        logger.info(f"Running MCP protocol server with {transport} transport")
        server.run_mcp_protocol(transport, host=host, port=port)


if __name__ == "__main__":
    main()


# Export server classes and functions
__all__ = [
    "ProductionMCPServer",
    "create_production_server",
    "main",
    "run_production_server_async",
]
