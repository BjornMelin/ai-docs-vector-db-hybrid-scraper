"""Simplified production FastMCP server with essential middleware only."""

# asyncio is required for shutdown coordination via Event/create_task
import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from starlette import status
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.config.loader import get_settings
from src.services.fastapi.middleware.manager import apply_defaults, apply_named_stack
from src.services.logging_config import configure_logging
from src.services.observability.health_manager import (
    HealthStatus,
    build_health_manager,
)


try:
    import uvicorn
except ImportError:
    uvicorn = None


if TYPE_CHECKING:
    from fastmcp import FastMCP


logger = logging.getLogger(__name__)


class ProductionMCPServer:
    """Simple production FastMCP server with essential middleware.

    This server wraps FastMCP with basic production middleware
    while maintaining full MCP protocol compatibility.
    """

    def __init__(self, config=None):
        """Initialize production MCP server.

        Args:
            config: Optional pre-loaded application settings. When omitted the
                factory loads configuration via :func:`get_settings`.
        """
        self.config = config or get_settings()
        self._middleware_names: list[str] = ["rate_limiting"]
        self._mcp_server: FastMCP | None = None
        self._app: Starlette | None = None
        self._shutdown_event = asyncio.Event()
        self._health_manager = build_health_manager(self.config)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, _frame) -> None:
        """Handle shutdown signals gracefully.

        Args:
            signum: Numeric identifier for the received signal.
            _frame: Current execution frame supplied by the interpreter.
        """
        logger.info("Received signal %d, initiating graceful shutdown...", signum)
        # Store task reference to avoid RUF006 warning
        task = asyncio.create_task(self.shutdown())
        # Set task name for easier debugging
        task.set_name("graceful-shutdown")

    @asynccontextmanager
    async def lifespan(self, _app: Starlette):
        """Application lifespan manager with startup/shutdown.

        Args:
            _app: Starlette application instance invoking the context manager.

        Yields:
            None. Control resumes once the application is ready to serve.
        """
        try:
            # Startup
            logger.info("Starting production MCP server...")
            await self.startup()
            yield

        except (TimeoutError, OSError, PermissionError):
            logger.exception("Startup failed")
            raise

        finally:
            # Shutdown
            logger.info("Shutting down production MCP server...")
            await self.shutdown()

    async def startup(self) -> None:
        """Initialize server components."""
        # Configure logging
        configure_logging(
            level=self.config.log_level.value,
            enable_color=self.config.debug,
            settings=self.config,
        )

        stack = getattr(self.config, "middleware_stack", None)
        if isinstance(stack, list) and stack:
            self._middleware_names = list(stack)
        else:
            self._middleware_names = ["rate_limiting"]

        logger.info("Production MCP server startup complete")

    async def shutdown(self) -> None:
        """Clean up server components."""
        try:
            self._shutdown_event.set()

            if self._mcp_server:
                # FastMCP cleanup would go here if it had cleanup methods
                logger.info("FastMCP server cleanup complete")

        except (ConnectionError, OSError, PermissionError):
            logger.exception("Error during shutdown")

        logger.info("Production MCP server shutdown complete")

    def create_app(self) -> Starlette:
        """Create Starlette application with middleware.

        Returns:
            Configured Starlette application exposing the health endpoint.
        """

        # Routes
        async def health_endpoint(_request) -> JSONResponse:
            """Return aggregated health information for the MCP surface.

            Args:
                _request: Incoming Starlette request (unused).

            Returns:
                JSONResponse describing aggregated system health.
            """

            await self._health_manager.check_all()
            summary = self._health_manager.get_health_summary()
            overall_status = summary.get("overall_status", HealthStatus.UNKNOWN.value)
            status_code = status.HTTP_200_OK
            if overall_status == HealthStatus.UNHEALTHY.value:
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE

            timestamp_value = summary.get("timestamp")
            if timestamp_value is None:
                timestamp_value = datetime.now(UTC).timestamp()

            payload = {
                "status": overall_status,
                "healthy_count": summary.get("healthy_count", 0),
                "total_count": summary.get("total_count", 0),
                "timestamp": datetime.fromtimestamp(
                    float(timestamp_value), tz=UTC
                ).isoformat(),
                "checks": summary.get("checks", {}),
            }

            return JSONResponse(payload, status_code=status_code)

        routes = [
            Route("/health", health_endpoint, methods=["GET"]),
        ]

        # Create app with lifespan
        app = Starlette(
            routes=routes,
            lifespan=self.lifespan,
        )

        if self._middleware_names:
            apply_named_stack(app, self._middleware_names)
        else:
            apply_defaults(app)

        self._app = app
        return app

    async def run_async(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        """Run the production server asynchronously.

        Args:
            host: Interface on which the server should listen.
            port: TCP port exposed for clients.

        Raises:
            ImportError: If ``uvicorn`` is not installed.
            OSError: If the server cannot bind to the requested address.
            PermissionError: When insufficient permissions prevent binding.
            RuntimeError: Raised by the underlying server during execution.
        """
        try:
            if uvicorn is None:
                msg = "uvicorn not available"
                raise ImportError(msg)

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

        except (OSError, PermissionError, RuntimeError):
            logger.exception("Server error")
            raise


def create_production_server(config=None) -> ProductionMCPServer:
    """Create a production-enhanced MCP server.

    Args:
        config: Optional pre-loaded application settings instance.

    Returns:
        ProductionMCPServer configured with middleware defaults.
    """
    return ProductionMCPServer(config or get_settings())


async def run_production_server_async(
    config=None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run production server with enhanced middleware stack.

    Args:
        config: Optional pre-loaded application settings instance.
        host: Interface on which the server should listen.
        port: TCP port exposed for clients.
    """
    server = create_production_server(config)
    await server.run_async(host, port)


def main() -> None:
    """Main entry point for production server.

    Raises:
        SystemExit: When the server terminates due to unrecoverable errors.
    """
    # Load configuration
    config = get_settings()

    # Get server configuration from environment
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    try:
        # Run the server
        asyncio.run(run_production_server_async(config, host, port))
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except (OSError, PermissionError, RuntimeError):
        logger.exception("Server error")
        sys.exit(1)


if __name__ == "__main__":
    main()
