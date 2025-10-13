"""Tests for the asynchronous database manager."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pytest_mock import MockerFixture

from src.infrastructure.database.connection_manager import DatabaseManager


class _DummySession:
    """Async context manager that mimics an SQLAlchemy session."""

    async def __aenter__(self) -> _DummySession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _FailingSession(_DummySession):
    async def commit(self) -> None:  # noqa: D401 - intentionally raise for failure path
        raise RuntimeError("commit failed")


@pytest.fixture()
def tracer_fixture(mocker: MockerFixture) -> tuple[Any, Any, Any]:
    """Patch OpenTelemetry tracer and histogram for deterministic assertions."""

    span = mocker.MagicMock()
    span_cm = mocker.MagicMock()
    span_cm.__enter__.return_value = span
    span_cm.__exit__.return_value = None

    tracer = mocker.MagicMock()
    tracer.start_as_current_span.return_value = span_cm

    histogram = mocker.MagicMock()

    mocker.patch(
        "src.infrastructure.database.connection_manager.trace.get_tracer",
        return_value=tracer,
    )
    mocker.patch(
        "src.infrastructure.database.connection_manager._QUERY_DURATION_HISTOGRAM",
        histogram,
    )

    return tracer, span, histogram


@pytest.fixture()
def configure_engine(mocker: MockerFixture) -> Callable[[Callable[[], Any]], Any]:
    """Return a helper to provision async engine and session factory patches."""

    def _configure(session_factory: Callable[[], Any]) -> Any:
        engine = mocker.Mock()
        engine.dispose = mocker.AsyncMock()
        engine.pool = mocker.Mock(
            size=mocker.Mock(return_value=5),
            checked_out=mocker.Mock(return_value=1),
        )
        mocker.patch(
            "src.infrastructure.database.connection_manager.create_async_engine",
            return_value=engine,
        )
        mocker.patch(
            "src.infrastructure.database.connection_manager.async_sessionmaker",
            return_value=session_factory,
        )
        return engine

    return _configure


@pytest.mark.asyncio()
async def test_initialize_creates_engine_and_session_factory(
    mocker: MockerFixture,
    config_factory,
    configure_engine: Callable[[Callable[[], Any]], Any],
) -> None:
    """initialize should configure the async engine and session factory."""

    settings = config_factory(
        database={
            "database_url": "sqlite+aiosqlite:///:memory:",
            "echo_queries": False,
            "pool_size": 5,
            "max_overflow": 2,
            "pool_timeout": 30,
        }
    )

    session_factory = mocker.Mock(return_value=_DummySession())
    configure_engine(session_factory)

    manager = DatabaseManager(settings)
    await manager.initialize()

    assert manager.is_initialized is True


@pytest.mark.asyncio()
async def test_session_records_success(
    mocker: MockerFixture,
    config_factory,
    configure_engine: Callable[[Callable[[], Any]], Any],
    tracer_fixture: tuple[Any, Any, Any],
) -> None:
    """session should commit work and publish telemetry on success."""

    tracer, span, histogram = tracer_fixture
    settings = config_factory(
        database={
            "database_url": "sqlite+aiosqlite:///:memory:",
            "echo_queries": False,
            "pool_size": 5,
            "max_overflow": 2,
            "pool_timeout": 30,
        }
    )

    session_factory = mocker.Mock(return_value=_DummySession())
    configure_engine(session_factory)

    mocker.patch(
        "src.infrastructure.database.connection_manager.perf_counter",
        side_effect=[1.0, 1.05],
    )

    manager = DatabaseManager(settings)
    await manager.initialize()

    async with manager.session() as session:
        assert isinstance(session, _DummySession)

    tracer.start_as_current_span.assert_called_once_with("database.session")
    span.set_attribute.assert_any_call("db.session.outcome", "success")
    histogram.record.assert_called_once()
    recorded_duration, attributes = histogram.record.call_args.args
    assert recorded_duration == pytest.approx(50.0, rel=1e-3)
    assert attributes["outcome"] == "success"
    assert attributes["service"] == "infrastructure.database"


@pytest.mark.asyncio()
async def test_session_records_failure_on_exception(
    mocker: MockerFixture,
    config_factory,
    configure_engine: Callable[[Callable[[], Any]], Any],
    tracer_fixture: tuple[Any, Any, Any],
) -> None:
    """session should rollback and record failure telemetry when errors occur."""

    tracer, span, histogram = tracer_fixture
    settings = config_factory(
        database={
            "database_url": "sqlite+aiosqlite:///:memory:",
            "echo_queries": False,
            "pool_size": 5,
            "max_overflow": 2,
            "pool_timeout": 30,
        }
    )

    session_factory = mocker.Mock(return_value=_FailingSession())
    configure_engine(session_factory)

    mocker.patch(
        "src.infrastructure.database.connection_manager.perf_counter",
        side_effect=[1.0, 1.02],
    )

    manager = DatabaseManager(settings)
    await manager.initialize()

    with pytest.raises(RuntimeError, match="commit failed"):
        async with manager.session():
            pass

    tracer.start_as_current_span.assert_called_once_with("database.session")
    span.record_exception.assert_called_once()
    span.set_status.assert_called_once()
    histogram.record.assert_called_once()
    recorded_duration, attributes = histogram.record.call_args.args
    assert recorded_duration == pytest.approx(20.0, rel=1e-3)
    assert attributes["outcome"] == "failure"
    assert attributes["service"] == "infrastructure.database"


@pytest.mark.asyncio()
async def test_get_performance_metrics_returns_snapshot(
    mocker: MockerFixture,
    config_factory,
    configure_engine: Callable[[Callable[[], Any]], Any],
) -> None:
    """get_performance_metrics should surface pool stats and telemetry metadata."""

    settings = config_factory(
        database={
            "database_url": "sqlite+aiosqlite:///:memory:",
            "echo_queries": False,
            "pool_size": 5,
            "max_overflow": 2,
            "pool_timeout": 30,
        }
    )

    session_factory = mocker.Mock(return_value=_DummySession())
    engine = configure_engine(session_factory)

    breaker_manager = mocker.MagicMock()
    breaker_manager.get_breaker_status = mocker.AsyncMock(
        return_value={"state": "closed"}
    )

    manager = DatabaseManager(
        settings,
        circuit_breaker_manager=breaker_manager,
    )
    await manager.initialize()

    metrics = await manager.get_performance_metrics()

    assert metrics["connection_count"] == 0
    assert metrics["pool"]["size"] == 5
    assert metrics["pool"]["checked_out"] == 1
    assert metrics["circuit_breaker_status"] == "closed"
    assert metrics["telemetry"] == {
        "histogram": "db.query.duration",
        "span": "database.session",
    }
    breaker_manager.get_breaker_status.assert_awaited_once_with(
        "infrastructure.database"
    )

    # Ensure the engine patch remains reachable for cleanup.
    assert engine.dispose.called is False
