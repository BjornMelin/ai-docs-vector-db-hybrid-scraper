"""Unit tests for service helpers on :mod:`src.infrastructure.client_manager`."""

# ruff: noqa: E402
# pylint: disable=wrong-import-position,unused-import

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest


def _ensure_stubbed_module(name: str) -> types.ModuleType:
    """Create a minimal stub module when optional dependencies are missing."""

    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


try:  # pragma: no cover - optional dependency
    from dependency_injector.wiring import Provide, inject  # type: ignore
except ImportError:  # pragma: no cover - test environment fallback
    wiring_module = _ensure_stubbed_module("dependency_injector.wiring")

    def inject(func):  # type: ignore
        return func

    class _Provide:
        """Minimal callable sentinel matching dependency-injector Provide."""

        def __call__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def __getitem__(self, _item: Any) -> None:
            return None

    Provide = _Provide()

    wiring_module.inject = inject  # type: ignore[attr-defined]
    wiring_module.Provide = Provide  # type: ignore[attr-defined]

try:  # pragma: no cover - optional dependency
    import langchain_mcp_adapters.client  # type: ignore  # noqa: F401
    import langchain_mcp_adapters.sessions  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - test environment fallback
    _ensure_stubbed_module("langchain_mcp_adapters")

    client_module = _ensure_stubbed_module("langchain_mcp_adapters.client")

    class _MultiServerMCPClient:  # pragma: no cover - minimal stub
        def __init__(self, *_args, **_kwargs) -> None:
            self.connected = False

        async def connect(self) -> None:
            self.connected = True

        async def disconnect(self) -> None:
            self.connected = False

    client_module.MultiServerMCPClient = _MultiServerMCPClient  # type: ignore[attr-defined]

    sessions_module = _ensure_stubbed_module("langchain_mcp_adapters.sessions")

    class _Connection:  # pragma: no cover - minimal stub
        def __init__(self, *_args, **_kwargs) -> None:
            self.connected = False

        async def __aenter__(self) -> _Connection:
            self.connected = True
            return self

        async def __aexit__(self, *_exc_info) -> None:
            self.connected = False

    sessions_module.Connection = _Connection  # type: ignore[attr-defined]


if "aiohttp" not in sys.modules:  # pragma: no cover - test stub
    aiohttp_module = types.ModuleType("aiohttp")

    class _ClientSession:  # pragma: no cover - minimal stub
        async def __aenter__(self) -> _ClientSession:
            return self

        async def __aexit__(self, *_exc_info) -> None:
            return None

        async def get(self, *_args, **_kwargs):
            return self

    aiohttp_module.ClientSession = _ClientSession  # type: ignore[attr-defined]

    class _ClientResponse:  # pragma: no cover - minimal stub
        status = 200
        headers: dict[str, str] = {}

    aiohttp_module.ClientResponse = _ClientResponse  # type: ignore[attr-defined]

    class _ClientTimeout:  # pragma: no cover - minimal stub
        def __init__(self, total: float) -> None:
            self.total = total

    aiohttp_module.ClientTimeout = _ClientTimeout  # type: ignore[attr-defined]
    sys.modules["aiohttp"] = aiohttp_module


if "redis.asyncio" not in sys.modules:  # pragma: no cover - test stub
    redis_module = types.ModuleType("redis.asyncio")

    class _Redis:  # pragma: no cover - minimal stub
        async def ping(self) -> bool:
            return True

        async def info(self) -> dict[str, str]:
            return {}

        async def aclose(self) -> None:
            return None

    redis_module.Redis = _Redis  # type: ignore[attr-defined]
    redis_module.ConnectionError = ConnectionError  # type: ignore[attr-defined]
    redis_module.RedisError = RuntimeError  # type: ignore[attr-defined]

    def from_url(*_args, **_kwargs) -> _Redis:  # type: ignore
        return _Redis()

    redis_module.from_url = from_url  # type: ignore[attr-defined]
    redis_package = types.ModuleType("redis")
    redis_package.asyncio = redis_module  # type: ignore[attr-defined]
    sys.modules["redis"] = redis_package
    sys.modules["redis.asyncio"] = redis_module


if "openai" not in sys.modules:  # pragma: no cover - test stub
    openai_module = types.ModuleType("openai")

    class _Models:  # pragma: no cover - minimal stub
        async def list(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(data=[])

    class _AsyncOpenAI:  # pragma: no cover - minimal stub
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.models = _Models()

    openai_module.AsyncOpenAI = _AsyncOpenAI  # type: ignore[attr-defined]
    sys.modules["openai"] = openai_module

if "qdrant_client" not in sys.modules:  # pragma: no cover - test stub
    qdrant_module = types.ModuleType("qdrant_client")

    class _AsyncQdrantClient:  # pragma: no cover - minimal stub
        async def get_collections(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(collections=[])

    qdrant_module.AsyncQdrantClient = _AsyncQdrantClient  # type: ignore[attr-defined]
    sys.modules["qdrant_client"] = qdrant_module
    qdrant_models_module = types.ModuleType("qdrant_client.models")
    qdrant_models_module.VectorParams = types.SimpleNamespace  # type: ignore[attr-defined]
    qdrant_models_module.CreateAlias = types.SimpleNamespace  # type: ignore[attr-defined]
    qdrant_models_module.CreateAliasOperation = types.SimpleNamespace  # type: ignore[attr-defined]
    qdrant_models_module.DeleteAlias = types.SimpleNamespace  # type: ignore[attr-defined]
    qdrant_models_module.DeleteAliasOperation = types.SimpleNamespace  # type: ignore[attr-defined]
    sys.modules["qdrant_client.models"] = qdrant_models_module

if "langchain_community" not in sys.modules:  # pragma: no cover - test stub
    community_module = types.ModuleType("langchain_community")
    embeddings_module = types.ModuleType("langchain_community.embeddings")
    fastembed_module = types.ModuleType("langchain_community.embeddings.fastembed")

    class _FastEmbedEmbeddings:  # pragma: no cover - minimal stub
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def aembed_documents(
            self, *_args: Any, **_kwargs: Any
        ) -> list[list[float]]:
            return []

    fastembed_module.FastEmbedEmbeddings = _FastEmbedEmbeddings  # type: ignore[attr-defined]
    embeddings_module.fastembed = fastembed_module  # type: ignore[attr-defined]
    community_module.embeddings = embeddings_module  # type: ignore[attr-defined]
    sys.modules["langchain_community"] = community_module
    sys.modules["langchain_community.embeddings"] = embeddings_module
    sys.modules["langchain_community.embeddings.fastembed"] = fastembed_module

if "firecrawl" not in sys.modules:  # pragma: no cover - test stub
    firecrawl_module = types.ModuleType("firecrawl")

    class _AsyncFirecrawlApp:  # pragma: no cover - minimal stub
        async def __aenter__(self) -> _AsyncFirecrawlApp:
            return self

        async def __aexit__(self, *_exc_info) -> None:
            return None

    firecrawl_module.AsyncFirecrawlApp = _AsyncFirecrawlApp  # type: ignore[attr-defined]
    sys.modules["firecrawl"] = firecrawl_module

if "firecrawl.app" not in sys.modules:  # pragma: no cover - test stub
    firecrawl_app_module = types.ModuleType("firecrawl.app")
    sys.modules["firecrawl.app"] = firecrawl_app_module

if "prometheus_client" not in sys.modules:  # pragma: no cover - test stub
    prometheus_module = types.ModuleType("prometheus_client")

    class _Metric:  # pragma: no cover - minimal stub
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def labels(self, *_args: Any, **_kwargs: Any) -> _Metric:
            return self

        def inc(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def observe(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    def start_http_server(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        return None

    prometheus_module.Counter = _Metric  # type: ignore[attr-defined]
    prometheus_module.Gauge = _Metric  # type: ignore[attr-defined]
    prometheus_module.Histogram = _Metric  # type: ignore[attr-defined]
    prometheus_module.CONTENT_TYPE_LATEST = "text/plain"  # type: ignore[attr-defined]

    def generate_latest(*_args: Any, **_kwargs: Any) -> bytes:  # pragma: no cover
        return b""

    prometheus_module.generate_latest = generate_latest  # type: ignore[attr-defined]
    prometheus_module.start_http_server = start_http_server  # type: ignore[attr-defined]
    sys.modules["prometheus_client"] = prometheus_module
    registry_module = types.ModuleType("prometheus_client.registry")

    class _Registry:  # pragma: no cover - minimal stub
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def register(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    registry_module.REGISTRY = _Registry()  # type: ignore[attr-defined]
    registry_module.CollectorRegistry = _Registry  # type: ignore[attr-defined]
    sys.modules["prometheus_client.registry"] = registry_module

if "prometheus_fastapi_instrumentator" not in sys.modules:  # pragma: no cover
    instrumentator_module = types.ModuleType("prometheus_fastapi_instrumentator")

    class _Instrumentator:  # pragma: no cover - minimal stub
        def instrument(self, *_args: Any, **_kwargs: Any) -> _Instrumentator:
            return self

        def expose(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    instrumentator_module.Instrumentator = _Instrumentator  # type: ignore[attr-defined]
    instrumentator_module.metrics = types.SimpleNamespace(  # pragma: no cover
        default=lambda *_args, **_kwargs: None,
    )
    sys.modules["prometheus_fastapi_instrumentator"] = instrumentator_module

if "numpy" not in sys.modules:  # pragma: no cover - test stub
    numpy_module = types.ModuleType("numpy")

    def array(values, *_args: Any, **_kwargs: Any):  # pragma: no cover
        return list(values)

    numpy_module.array = array  # type: ignore[attr-defined]
    numpy_module.float32 = float  # type: ignore[attr-defined]
    sys.modules["numpy"] = numpy_module

if "tenacity" not in sys.modules:  # pragma: no cover - test stub
    tenacity_module = types.ModuleType("tenacity")

    def retry(*_args: Any, **_kwargs: Any):  # pragma: no cover
        def decorator(func):
            return func

        return decorator

    tenacity_module.retry = retry  # type: ignore[attr-defined]
    tenacity_module.retry_if_exception_type = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    tenacity_module.retry_if_exception = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    tenacity_module.stop_after_attempt = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    tenacity_module.wait_fixed = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    tenacity_module.wait_exponential = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    tenacity_module.AsyncRetrying = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    sys.modules["tenacity"] = tenacity_module


from src.infrastructure.client_manager import ClientManager
from src.services.health.manager import (
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
)


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
    stub_result = HealthCheckResult(
        name="vector",
        status=HealthStatus.HEALTHY,
        message="ok",
        duration_ms=1.0,
        metadata={"latency_ms": 12.3},
    )
    health_manager = mocker.MagicMock(spec=HealthCheckManager)
    health_manager.check_all = mocker.AsyncMock(return_value={"vector": stub_result})
    manager._health_manager = health_manager  # pylint: disable=protected-access
    mocker.patch.object(manager, "_ensure_monitoring_ready")

    status = await manager.get_health_status()

    assert status == {
        "vector": {
            "status": HealthStatus.HEALTHY.value,
            "message": "ok",
            "timestamp": stub_result.timestamp,
            "duration_ms": stub_result.duration_ms,
            "metadata": {"latency_ms": 12.3},
            "is_healthy": True,
        }
    }
    health_manager.check_all.assert_awaited_once()


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
