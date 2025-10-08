"""Shared helpers for stubbing optional service dependencies in tests."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


__all__ = [
    "register_module",
    "stub_content_intelligence_dependencies",
    "load_content_intelligence_module",
]

ROOT_PATH = Path(__file__).resolve().parents[4]
SRC_PATH = ROOT_PATH / "src"
PACKAGE_NAME = "src.services.content_intelligence"


def _ensure_package(name: str, path: Path) -> ModuleType:
    """Ensure a namespace package exists in ``sys.modules``.

    Args:
        name: Fully qualified package name.
        path: Filesystem path backing the package.

    Returns:
        ModuleType: Existing or newly created package module.
    """

    package = sys.modules.get(name)
    if package is None:
        package = ModuleType(name)
        package.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = package
    return package


def register_module(name: str, module: ModuleType) -> None:
    """Register a stub module in ``sys.modules`` if it is missing.

    Args:
        name: Fully qualified module name to register.
        module: Module instance that should be associated with the name.
    """

    if name not in sys.modules:
        sys.modules[name] = module


def stub_content_intelligence_dependencies() -> None:
    """Provide lightweight stubs for optional runtime dependencies.

    Returns:
        None: The function mutates ``sys.modules`` in place.
    """

    redis_module = ModuleType("redis")
    redis_async_module = ModuleType("redis.asyncio")
    redis_module.asyncio = redis_async_module  # type: ignore[attr-defined]
    register_module("redis", redis_module)
    register_module("redis.asyncio", redis_async_module)

    register_module("psutil", ModuleType("psutil"))
    register_module("aiohttp", ModuleType("aiohttp"))

    fastapi_module = ModuleType("fastapi")
    fastapi_module.FastAPI = object  # type: ignore[attr-defined]
    fastapi_module.Request = object  # type: ignore[attr-defined]
    fastapi_module.Response = object  # type: ignore[attr-defined]
    fastapi_responses_module = ModuleType("fastapi.responses")
    fastapi_responses_module.JSONResponse = object  # type: ignore[attr-defined]
    fastapi_module.responses = fastapi_responses_module  # type: ignore[attr-defined]
    register_module("fastapi", fastapi_module)
    register_module("fastapi.responses", fastapi_responses_module)

    prometheus_module = ModuleType("prometheus_client")
    prometheus_module.Counter = object  # type: ignore[attr-defined]
    prometheus_module.Gauge = object  # type: ignore[attr-defined]
    prometheus_module.Histogram = object  # type: ignore[attr-defined]
    prometheus_module.start_http_server = lambda *args, **kwargs: None
    prometheus_module.CONTENT_TYPE_LATEST = "text/plain"
    prometheus_module.generate_latest = lambda *args, **kwargs: b""
    prometheus_registry_module = ModuleType("prometheus_client.registry")
    prometheus_registry_module.REGISTRY = object()
    prometheus_registry_module.CollectorRegistry = object  # type: ignore[attr-defined]
    prometheus_module.registry = prometheus_registry_module  # type: ignore[attr-defined]
    register_module("prometheus_client", prometheus_module)
    register_module("prometheus_client.registry", prometheus_registry_module)

    instrumentator_module = ModuleType("prometheus_fastapi_instrumentator")
    instrumentator_module.Instrumentator = object  # type: ignore[attr-defined]
    instrumentator_module.metrics = ModuleType(
        "prometheus_fastapi_instrumentator.metrics"
    )
    register_module("prometheus_fastapi_instrumentator", instrumentator_module)

    qdrant_module = ModuleType("qdrant_client")
    qdrant_module.AsyncQdrantClient = object  # type: ignore[attr-defined]
    qdrant_http_module = ModuleType("qdrant_client.http")
    qdrant_exceptions_module = ModuleType("qdrant_client.http.exceptions")
    qdrant_http_module.exceptions = qdrant_exceptions_module  # type: ignore[attr-defined]
    qdrant_exceptions_module.UnexpectedResponse = Exception  # type: ignore[attr-defined]
    register_module("qdrant_client", qdrant_module)
    register_module("qdrant_client.http", qdrant_http_module)
    register_module("qdrant_client.http.exceptions", qdrant_exceptions_module)


def _ensure_src_on_path() -> None:
    """Make sure the repository source directory is importable.

    Returns:
        None: The function appends the ``src`` directory to ``sys.path`` when missing.
    """

    src_str = str(SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def load_content_intelligence_module(module: str) -> ModuleType:
    """Load a content intelligence module with dependencies stubbed.

    Args:
        module: Module name relative to ``src.services.content_intelligence``.

    Returns:
        ModuleType: The loaded Python module object.

    Raises:
        ImportError: If the module specification cannot be resolved.
    """

    stub_content_intelligence_dependencies()
    _ensure_src_on_path()

    parent_package = PACKAGE_NAME
    full_name = f"{parent_package}.{module}"

    src_package = _ensure_package("src", SRC_PATH)
    services_package = _ensure_package("src.services", SRC_PATH / "services")
    src_package.services = services_package
    ci_path = SRC_PATH / "services" / "content_intelligence"
    ci_package = _ensure_package(parent_package, ci_path)
    services_package.content_intelligence = ci_package

    if full_name in sys.modules:
        return sys.modules[full_name]

    module_path = ci_path / f"{module}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load module specification for {full_name}"
        raise ImportError(msg)

    loaded_module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = loaded_module
    spec.loader.exec_module(loaded_module)

    setattr(ci_package, module, loaded_module)

    return loaded_module
