"""Unit tests for service helpers on :mod:`src.infrastructure.client_manager`."""

from __future__ import annotations

import asyncio

import pytest

from src.infrastructure.client_manager import ClientManager, HealthCheckRegistration
from src.infrastructure.shared import ClientHealth, ClientState


@pytest.fixture(autouse=True)
def _reset_client_manager() -> None:
    """Reset the ClientManager singleton around each test."""

    ClientManager.reset_singleton()


@pytest.mark.asyncio
async def test_database_session_exposes_resolved_services(
    mocker: pytest.MockFixture,
) -> None:
    """database_session should provide redis, cache, and vector services."""

    manager = ClientManager()
    redis_client = object()
    cache_manager = object()
    vector_service = object()

    mocker.patch.object(manager, "get_redis_client", return_value=redis_client)
    mocker.patch.object(manager, "get_cache_manager", return_value=cache_manager)
    mocker.patch.object(
        manager, "get_vector_store_service", return_value=vector_service
    )

    async with manager.database_session() as session:
        assert session.redis is redis_client
        assert session.cache_manager is cache_manager
        assert session.vector_service is vector_service


@pytest.mark.asyncio
async def test_upsert_vector_records_wraps_payload(mocker: pytest.MockFixture) -> None:
    """Vector payloads are converted to VectorRecord instances."""

    manager = ClientManager()
    upsert_mock = mocker.AsyncMock()
    vector_service = mocker.Mock(upsert_vectors=upsert_mock)
    mocker.patch.object(
        manager, "get_vector_store_service", return_value=vector_service
    )

    points = [
        {"id": "alpha", "vector": [0.1, 0.2, 0.3], "payload": {"source": "doc"}},
    ]

    result = await manager.upsert_vector_records("demo", points)

    assert result is True
    upsert_mock.assert_awaited_once()
    args, _ = upsert_mock.call_args
    assert args[0] == "demo"
    records = list(args[1])
    assert records[0].id == "alpha"
    assert records[0].vector == [0.1, 0.2, 0.3]
    assert records[0].payload == {"source": "doc"}


@pytest.mark.asyncio
async def test_search_vector_records_serialises_matches(
    mocker: pytest.MockFixture,
) -> None:
    """Vector search results should be coerced into simple dictionaries."""

    manager = ClientManager()
    match = mocker.Mock(id="beta", score=0.77, metadata={"lang": "en"})
    vector_service = mocker.Mock(
        search_vector=mocker.AsyncMock(return_value=[match]),
    )
    mocker.patch.object(
        manager, "get_vector_store_service", return_value=vector_service
    )

    results = await manager.search_vector_records("demo", [0.4, 0.5, 0.6])

    assert results == [
        {"id": "beta", "score": 0.77, "metadata": {"lang": "en"}},
    ]


@pytest.mark.asyncio
async def test_get_health_status_runs_checks(mocker: pytest.MockFixture) -> None:
    """Health checks should execute and surface standard metadata."""

    manager = ClientManager()
    manager._monitoring_initialized = True  # pylint: disable=protected-access
    manager._health_checks = {}  # pylint: disable=protected-access
    manager._health_checks["vector"] = HealthCheckRegistration(  # pylint: disable=protected-access
        health=ClientHealth(state=ClientState.HEALTHY, last_check=0.0),
        check_function=mocker.AsyncMock(return_value=True),
        check_interval=30,
    )
    mocker.patch.object(manager, "_ensure_monitoring_ready")

    status = await manager.get_health_status()

    assert "vector" in status
    assert status["vector"]["is_healthy"] is True


@pytest.mark.asyncio
async def test_track_performance_records_metrics(mocker: pytest.MockFixture) -> None:
    """track_performance should emit latency and counters for success and errors."""

    manager = ClientManager()
    histogram = mocker.patch.object(manager, "record_histogram")
    counter = mocker.patch.object(manager, "increment_counter")

    async def _noop() -> str:
        await asyncio.sleep(0)
        return "ok"

    result = await manager.track_performance("demo", _noop)

    assert result == "ok"
    histogram.assert_called()
    counter.assert_called()
