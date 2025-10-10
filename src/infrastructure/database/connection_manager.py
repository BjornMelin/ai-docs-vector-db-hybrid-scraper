"""Database connection utilities built around async SQLAlchemy."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import metrics, trace
from opentelemetry.metrics import Histogram
from opentelemetry.trace import Status, StatusCode

from src.config import Settings
from src.services.circuit_breaker import CircuitBreakerManager


logger = logging.getLogger(__name__)


_QUERY_DURATION_HISTOGRAM: Histogram = metrics.get_meter(__name__).create_histogram(
    "db.query.duration",
    description="Duration of database operations in milliseconds",
    unit="ms",
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    SessionFactory = Callable[[], AsyncSession]
else:  # pragma: no cover - runtime typing fallbacks
    AsyncEngine = Any  # type: ignore[assignment]
    AsyncSession = Any  # type: ignore[assignment]
    SessionFactory = Callable[[], Any]

try:  # pragma: no cover - optional dependency import
    from sqlalchemy.ext.asyncio import (  # type: ignore[import]
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
except ImportError:  # pragma: no cover - runtime fallback when SQLAlchemy unavailable

    def _missing_sqlalchemy(*_args: Any, **_kwargs: Any) -> Any:
        msg = "SQLAlchemy async dependencies are required for DatabaseManager"
        raise RuntimeError(msg)

    async_sessionmaker = _missing_sqlalchemy  # type: ignore[assignment]
    create_async_engine = _missing_sqlalchemy  # type: ignore[assignment]


class DatabaseManager:
    """Coordinate async SQLAlchemy sessions with OpenTelemetry instrumentation."""

    def __init__(
        self,
        config: Settings,
        circuit_breaker_manager: CircuitBreakerManager | None = None,
        breaker_service_name: str = "infrastructure.database",
    ):  # pylint: disable=too-many-arguments
        """Initialize the database manager.

        Args:
            config: Unified application settings instance.
            circuit_breaker_manager: Optional circuit breaker manager.
            breaker_service_name: Service identifier for breaker acquisition.
        """

        self.config = config
        self._engine: AsyncEngine | None = None
        self._session_factory: SessionFactory | None = None
        self._circuit_breaker_manager = circuit_breaker_manager
        self._breaker_service_name = breaker_service_name
        self._connection_count = 0
        self._tracer = trace.get_tracer(__name__)
        self._histogram = _QUERY_DURATION_HISTOGRAM

    async def initialize(self) -> None:
        """Initialize database infrastructure."""

        if self._engine is not None:
            return

        try:
            self._engine = create_async_engine(
                self.config.database.database_url,
                echo=self.config.database.echo_queries,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow,
                pool_timeout=self.config.database.pool_timeout,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo_pool="debug" if self.config.database.echo_queries else False,
            )

            session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
            )
            self._session_factory = cast(SessionFactory, session_factory)

            logger.info(
                "Database manager initialized (pool_size: %s)",
                self.config.database.pool_size,
            )

        except (OSError, AttributeError, ConnectionError, ImportError):
            logger.exception("Failed to initialize database manager")
            raise

    async def cleanup(self) -> None:
        """Clean up database resources."""

        try:
            if self._engine:
                await self._engine.dispose()
                self._engine = None
                self._session_factory = None

            logger.info("Database manager cleaned up")

        except (ConnectionError, OSError, PermissionError):
            logger.exception("Error during database cleanup")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[Any, None]:
        """Yield a monitored database session."""

        session_factory = self._session_factory
        if session_factory is None:
            msg = "Database manager not initialized"
            raise RuntimeError(msg)

        async with AsyncExitStack() as stack:
            if self._circuit_breaker_manager is not None:
                breaker = await self._circuit_breaker_manager.get_breaker(
                    self._breaker_service_name
                )
                await stack.enter_async_context(breaker)

            session = await stack.enter_async_context(session_factory())
            with self._tracer.start_as_current_span("database.session") as span:
                operation_start = perf_counter()
                span.set_attribute("db.system", "sqlalchemy")
                span.set_attribute(
                    "db.session.pool_size",
                    int(getattr(self.config.database, "pool_size", 0)),
                )
                try:
                    self._connection_count += 1
                    yield session
                    await session.commit()
                    duration_ms = (perf_counter() - operation_start) * 1000
                    self._histogram.record(
                        duration_ms,
                        {"outcome": "success", "service": self._breaker_service_name},
                    )
                    span.set_attribute("db.session.outcome", "success")
                except Exception as exc:
                    await session.rollback()
                    duration_ms = (perf_counter() - operation_start) * 1000
                    self._histogram.record(
                        duration_ms,
                        {"outcome": "failure", "service": self._breaker_service_name},
                    )
                    span.set_attribute("db.session.outcome", "failure")
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get database performance metrics."""

        pool_snapshot: dict[str, int] = {}
        if self._engine is not None:
            pool = getattr(self._engine, "pool", None)
            if pool is not None:
                size_fn = getattr(pool, "size", None)
                checked_out_fn = getattr(pool, "checked_out", None)
                if callable(size_fn):
                    size_value = cast(int, size_fn())
                    pool_snapshot["size"] = int(size_value)
                if callable(checked_out_fn):
                    checked_out_value = cast(int, checked_out_fn())
                    pool_snapshot["checked_out"] = int(checked_out_value)

        return {
            "connection_count": self._connection_count,
            "circuit_breaker_status": await self._get_circuit_breaker_state(),
            "pool": pool_snapshot,
            "telemetry": {
                "histogram": "db.query.duration",
                "span": "database.session",
            },
        }

    @property
    def is_initialized(self) -> bool:
        """Check if the database manager is ready."""

        return self._engine is not None

    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine."""

        if not self._engine:
            msg = "Database manager not initialized"
            raise RuntimeError(msg)
        return self._engine

    async def _get_circuit_breaker_state(self) -> str:
        """Return current circuit breaker state or fallback when unavailable."""

        if self._circuit_breaker_manager is None:
            return "unconfigured"

        status = await self._circuit_breaker_manager.get_breaker_status(
            self._breaker_service_name
        )
        return str(status.get("state", "unknown"))
