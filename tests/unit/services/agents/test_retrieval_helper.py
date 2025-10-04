"""Tests for the retrieval helper."""

from __future__ import annotations

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest


MODULE_PATH = Path(__file__).resolve().parents[4] / "src/services/agents/retrieval.py"

for module_name in (
    "src.services",
    "src.services.agents",
    "src.services.agents.retrieval",
    "src.infrastructure.client_manager",
):
    sys.modules.pop(module_name, None)

infra_stub: Any = types.ModuleType("src.infrastructure.client_manager")


class ClientManager:  # noqa: D401 - lightweight stub for import
    pass


infra_stub.ClientManager = ClientManager
sys.modules["src.infrastructure.client_manager"] = infra_stub

_spec = spec_from_file_location("_retrieval_helper_under_test", MODULE_PATH)
assert _spec and _spec.loader
_module = module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)  # type: ignore[arg-type]

RetrievalHelper = _module.RetrievalHelper
RetrievalQuery = _module.RetrievalQuery


@pytest.mark.asyncio
async def test_fetch_uses_client_manager() -> None:
    """Retrieval helper should normalise vector store results."""

    class DummyMatch:
        def __init__(self, identifier: str, score: float) -> None:
            self.id = identifier
            self.score = score
            self.payload = {"title": f"Doc {identifier}"}

    class DummyVectorStoreService:
        async def search_documents(self, collection, query, *, limit, filters=None):  # noqa: D401, ANN001
            assert collection == "docs"
            assert query == "what is langgraph"
            assert limit == 3
            assert filters == {"topic": "rag"}
            return [DummyMatch("1", 0.9), DummyMatch("2", 0.7)]

    class DummyClientManager(ClientManager):
        def __init__(self) -> None:
            self._service = DummyVectorStoreService()

        async def get_vector_store_service(self):  # noqa: D401
            return self._service

    helper = RetrievalHelper(DummyClientManager())
    query = RetrievalQuery(
        collection="docs",
        text="what is langgraph",
        top_k=3,
        filters={"topic": "rag"},
    )

    results = await helper.fetch(query)
    assert len(results) == 2
    assert results[0].id == "1"
    assert results[0].payload == {"title": "Doc 1"}
    assert results[0].raw is not None
