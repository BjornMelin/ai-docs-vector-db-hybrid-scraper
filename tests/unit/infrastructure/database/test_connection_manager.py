"""Tests for the asynchronous database manager."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pytest_mock import MockerFixture

from src.infrastructure.database.connection_manager import DatabaseManager
from src.infrastructure.database.monitoring import QueryMonitor


class _DummySession:
    """Async context manager that mimics an SQLAlchemy session."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


@pytest.fixture()
def stub_query_monitor(mocker: MockerFixture) -> QueryMonitor:
    """Provide a stubbed query monitor with async hooks."""

    monitor = mocker.create_autospec(QueryMonitor, instance=True)
    monitor.initialize = mocker.AsyncMock()
    monitor.cleanup = mocker.AsyncMock()
    monitor.start_query.return_value = "query-1"
    monitor.record_success = mocker.Mock()
    monitor.record_failure = mocker.Mock()
    monitor.get_performance_summary = mocker.AsyncMock(
        return_value={"total_queries": 0}
    )
    return cast(QueryMonitor, monitor)


@pytest.mark.asyncio()
async def test_initialize_creates_engine_and_session_factory(
    mocker: MockerFixture,
    config_factory,
    stub_query_monitor: QueryMonitor,
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

    engine = mocker.Mock()
    engine.dispose = mocker.AsyncMock()
    engine.pool = mocker.Mock(
        size=mocker.Mock(return_value=5), checked_out=mocker.Mock(return_value=1)
    )
    dummy_factory = mocker.Mock(return_value=_DummySession())

    mocker.patch(
        "src.infrastructure.database.connection_manager.create_async_engine",
        return_value=engine,
    )
    mocker.patch(
        "src.infrastructure.database.connection_manager.async_sessionmaker",
        return_value=dummy_factory,
    )

    manager = DatabaseManager(settings, query_monitor=stub_query_monitor)
    await manager.initialize()

    monitor_mock = cast(Any, stub_query_monitor)
    monitor_mock.initialize.assert_awaited_once()
    assert manager.is_initialized is True


@pytest.mark.asyncio()
async def test_session_records_success(
    mocker: MockerFixture,
    config_factory,
    stub_query_monitor: QueryMonitor,
) -> None:
    """session should commit work and record success metrics."""

    settings = config_factory(
        database={
            "database_url": "sqlite+aiosqlite:///:memory:",
            "echo_queries": False,
            "pool_size": 5,
            "max_overflow": 2,
            "pool_timeout": 30,
        }
    )

    engine = mocker.Mock()
    engine.dispose = mocker.AsyncMock()
    engine.pool = mocker.Mock(
        size=mocker.Mock(return_value=5), checked_out=mocker.Mock(return_value=1)
    )
    dummy_factory = mocker.Mock(return_value=_DummySession())

    mocker.patch(
        "src.infrastructure.database.connection_manager.create_async_engine",
        return_value=engine,
    )
    mocker.patch(
        "src.infrastructure.database.connection_manager.async_sessionmaker",
        return_value=dummy_factory,
    )

    manager = DatabaseManager(settings, query_monitor=stub_query_monitor)
    await manager.initialize()

    async with manager.session() as session:
        assert isinstance(session, _DummySession)

    monitor_mock = cast(Any, stub_query_monitor)
    monitor_mock.record_success.assert_called_once_with("query-1")


@pytest.mark.asyncio()
async def test_session_records_failure_on_exception(
    mocker: MockerFixture,
    config_factory,
    stub_query_monitor: QueryMonitor,
) -> None:
    """session should rollback and record failure when errors occur."""

    settings = config_factory(
        database={
            "database_url": "sqlite+aiosqlite:///:memory:",
            "echo_queries": False,
            "pool_size": 5,
            "max_overflow": 2,
            "pool_timeout": 30,
        }
    )

    engine = mocker.Mock()
    engine.dispose = mocker.AsyncMock()
    engine.pool = mocker.Mock(
        size=mocker.Mock(return_value=5), checked_out=mocker.Mock(return_value=1)
    )

    class _FailingSession(_DummySession):
        async def commit(self) -> None:  # noqa: D401 - intentionally raise
            raise RuntimeError("commit failed")

    dummy_factory = mocker.Mock(return_value=_FailingSession())

    mocker.patch(
        "src.infrastructure.database.connection_manager.create_async_engine",
        return_value=engine,
    )
    mocker.patch(
        "src.infrastructure.database.connection_manager.async_sessionmaker",
        return_value=dummy_factory,
    )

    manager = DatabaseManager(settings, query_monitor=stub_query_monitor)
    await manager.initialize()

    with pytest.raises(RuntimeError, match="commit failed"):
        async with manager.session():
            pass

    monitor_mock = cast(Any, stub_query_monitor)
    monitor_mock.record_failure.assert_called_once()


@pytest.mark.asyncio()
async def test_get_performance_metrics_returns_snapshot(
    mocker: MockerFixture,
    config_factory,
    stub_query_monitor: QueryMonitor,
) -> None:
    """get_performance_metrics should surface query summaries and pool stats."""

    settings = config_factory(
        database={
            "database_url": "sqlite+aiosqlite:///:memory:",
            "echo_queries": False,
            "pool_size": 5,
            "max_overflow": 2,
            "pool_timeout": 30,
        }
    )

    engine = mocker.Mock()
    engine.dispose = mocker.AsyncMock()
    engine.pool = mocker.Mock(
        size=mocker.Mock(return_value=5), checked_out=mocker.Mock(return_value=1)
    )
    dummy_factory = mocker.Mock(return_value=_DummySession())

    mocker.patch(
        "src.infrastructure.database.connection_manager.create_async_engine",
        return_value=engine,
    )
    mocker.patch(
        "src.infrastructure.database.connection_manager.async_sessionmaker",
        return_value=dummy_factory,
    )

    monitor_mock = cast(Any, stub_query_monitor)
    monitor_mock.get_performance_summary.return_value = {"total_queries": 2}

    breaker_manager = mocker.MagicMock()
    breaker_manager.get_breaker_status = mocker.AsyncMock(
        return_value={"state": "closed"}
    )

    manager = DatabaseManager(
        settings,
        query_monitor=stub_query_monitor,
        circuit_breaker_manager=breaker_manager,
    )
    await manager.initialize()

    metrics = await manager.get_performance_metrics()

    assert metrics["connection_count"] == 0
    assert metrics["pool"]["size"] == 5
    assert metrics["pool"]["checked_out"] == 1
    assert metrics["query_metrics"] == {"total_queries": 2}
    assert metrics["circuit_breaker_status"] == "closed"
    breaker_manager.get_breaker_status.assert_awaited_once_with(
        "infrastructure.database"
    )
