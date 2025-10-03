"""Tests for MCP search tool wrappers that delegate to VectorStoreService."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.mcp_tools.tools.search import register_tools
from src.services.vector_db.types import VectorMatch


@pytest.fixture()
def mock_vector_service() -> Mock:
    service = Mock()
    service.is_initialized.return_value = True
    service.initialize = AsyncMock()
    service.search_documents = AsyncMock()
    service.list_documents = AsyncMock()
    service.get_document = AsyncMock()
    service.recommend = AsyncMock()
    return service


@pytest.fixture()
def mock_client_manager(mock_vector_service: Mock) -> Mock:
    manager = Mock()
    manager.get_vector_store_service = AsyncMock(return_value=mock_vector_service)
    return manager


@pytest.fixture()
def register(mock_client_manager: Mock) -> dict[str, Callable]:
    mock_mcp = MagicMock()
    registered: dict[str, Callable] = {}

    def capture(func: Callable) -> Callable:
        registered[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture
    register_tools(mock_mcp, mock_client_manager)
    return registered


@pytest.fixture()
def mock_context() -> Mock:
    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()
    return ctx


def _match(
    identifier: str,
    score: float = 0.5,
    payload: dict[str, str] | None = None,
    *,
    vector: tuple[float, ...] | None = None,
) -> VectorMatch:
    return VectorMatch(
        id=identifier,
        score=score,
        payload=payload or {"content": identifier},
        vector=vector,
    )


@pytest.mark.asyncio
async def test_registers_expected_tools(register: dict[str, Callable]) -> None:
    expected = {
        "search_documents",
        "hybrid_search",
        "scroll_collection",
        "search_with_context",
        "recommend_similar",
        "hyde_search",
        "reranked_search",
        "multi_stage_search",
        "filtered_search",
    }
    assert expected <= set(register)


@pytest.mark.asyncio
async def test_search_documents_applies_threshold(
    register: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.search_documents.return_value = [
        _match("keep", 0.9),
        _match("drop", 0.3),
    ]
    results = await register["search_documents"](
        query="test",
        collection="docs",
        limit=5,
        score_threshold=0.5,
        filter_conditions={"tag": "howto"},
        ctx=mock_context,
    )
    mock_vector_service.search_documents.assert_awaited_once_with(
        "docs",
        "test",
        limit=5,
        filters={"tag": "howto"},
    )
    assert [result["id"] for result in results] == ["keep"]


@pytest.mark.asyncio
async def test_hybrid_search_forwards_filters(
    register: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.search_documents.return_value = [_match("hybrid", 0.8)]
    results = await register["hybrid_search"](
        query="hybrid",
        limit=1,
        filter_conditions={"stage": {"gte": 2}},
        ctx=mock_context,
    )
    mock_vector_service.search_documents.assert_awaited_once_with(
        "documentation",
        "hybrid",
        limit=1,
        filters={"stage": {"gte": 2}},
    )
    assert results[0]["id"] == "hybrid"


@pytest.mark.asyncio
async def test_scroll_collection_formats_payload(
    register: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.list_documents.return_value = (
        [
            {"id": "doc-1", "content": "first"},
            {"id": "doc-2", "content": "second"},
        ],
        "cursor",
    )
    payload = await register["scroll_collection"](
        collection="docs",
        limit=2,
        ctx=mock_context,
    )
    assert payload["count"] == 2
    assert payload["documents"][0]["id"] == "doc-1"
    assert payload["next_page_offset"] == "cursor"


@pytest.mark.asyncio
async def test_recommend_similar_excludes_seed_and_applies_threshold(
    register: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.get_document.return_value = {"id": "seed"}
    mock_vector_service.recommend.return_value = [
        _match("seed", 1.0),
        _match("candidate-1", 0.9),
        _match("candidate-2", 0.4),
    ]
    results = await register["recommend_similar"](
        point_id="seed",
        limit=5,
        score_threshold=0.5,
        ctx=mock_context,
    )
    mock_vector_service.recommend.assert_awaited_once_with(
        "documentation",
        positive_ids=["seed"],
        limit=6,
        filters=None,
    )
    assert [result["id"] for result in results] == ["candidate-1"]
    assert results[0]["method"] == "recommend_similar"


@pytest.mark.asyncio
async def test_recommend_similar_raises_when_document_missing(
    register: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.get_document.return_value = None
    with pytest.raises(ValueError, match="Document missing not found"):
        await register["recommend_similar"](
            point_id="missing",
            ctx=mock_context,
        )


@pytest.mark.asyncio
async def test_reranked_search_adds_rrf_metadata(
    register: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.search_documents.return_value = [
        _match("first", 0.9),
        _match("second", 0.8),
    ]
    results = await register["reranked_search"](
        query="test",
        rerank_limit=5,
        ctx=mock_context,
    )
    assert all(result["method"] == "rrf_reranked" for result in results)
    assert "rerank_score" in results[0]


@pytest.mark.asyncio
async def test_multi_stage_search_deduplicates_by_score(
    register: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.search_documents.side_effect = [
        [_match("dup", 0.4), _match("unique", 0.3)],
        [_match("dup", 0.9)],
    ]
    results = await register["multi_stage_search"](
        query="test",
        stages=2,
        limit=2,
        ctx=mock_context,
    )
    assert [result["id"] for result in results] == ["dup", "unique"]
    assert results[0]["method"].startswith("multi_stage_")


@pytest.mark.asyncio
async def test_filtered_search_converts_boolean_filters(
    register: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.search_documents.return_value = [_match("doc", 0.7)]
    await register["filtered_search"](
        query="doc",
        must_conditions=[{"key": "tag", "value": "api"}],
        should_conditions=[{"key": "lang", "values": ["py", "rs"]}],
        must_not_conditions=[{"key": "status", "value": "draft"}],
        ctx=mock_context,
    )
    mock_vector_service.search_documents.assert_awaited_once()
    call_kwargs = mock_vector_service.search_documents.await_args.kwargs
    assert "filters" in call_kwargs
    filters = call_kwargs["filters"]
    assert filters["must"][0]["value"] == "api"
    assert set(filters["should"][0]["values"]) == {"py", "rs"}
    assert filters["must_not"][0]["value"] == "draft"
