"""Tests for unified retrieval tools."""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.contracts.retrieval import SearchRecord
from src.mcp_tools.tools import retrieval
from src.models.search import SearchRequest


class _Ctx:
    """Async logging stub."""

    async def info(self, *_args, **_kwargs):  # noqa: D401
        """Discard log calls."""

        return None


@pytest_asyncio.fixture()
async def registered(fake_mcp, fake_client_manager):
    """Register tools and yield registry."""

    retrieval.register_tools(fake_mcp, fake_client_manager)
    return fake_mcp.tools


@pytest.mark.asyncio
async def test_search_documents_basic(registered):
    """Search returns SearchRecord objects."""

    fn = registered["search_documents"]
    res = await fn(
        request=SearchRequest(
            query="hello",
            collection="documentation",
            limit=3,
            offset=0,
            include_metadata=True,
        ),
        ctx=_Ctx(),
    )
    assert len(res) == 3
    assert all(isinstance(item, SearchRecord) for item in res)
    assert res[0].id == "1"
    assert res[0].metadata is not None


@pytest.mark.asyncio
async def test_filtered_search_forwards_filters(registered):
    """Filtered search forwards filters."""

    fn = registered["filtered_search"]
    res = await fn(
        request=SearchRequest(
            query="q",
            collection="documentation",
            limit=2,
            offset=0,
            filters={"site_name": {"value": "docs"}},
            include_metadata=True,
        ),
        ctx=_Ctx(),
    )
    assert res and res[0].metadata and res[0].metadata["q"] == "q"


@pytest.mark.asyncio
async def test_multi_stage_merges_and_dedupes(registered):
    """Multi-stage returns unique items by id."""

    fn = registered["multi_stage_search"]
    req = retrieval.MultiStageSearchPayload(
        collection="documentation",
        query="x",
        limit=5,
        stages=[{"limit": 5}, {"limit": 5, "filters": {"k": 1}}],
        include_metadata=False,
    )
    res = await fn(payload=req, ctx=_Ctx())
    ids = [r.id for r in res]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_search_with_context_expands_limit(registered):
    """Context search increases retrieved candidates."""

    fn = registered["search_with_context"]
    res = await fn(
        query="context",
        collection="documentation",
        limit=3,
        context_size=2,
        include_metadata=False,
        ctx=_Ctx(),
    )
    assert len(res) == 5  # 3 base + 2 context hits
    assert all(record.metadata is None for record in res)


@pytest.mark.asyncio
async def test_recommend_similar_excludes_seed(registered):
    """Recommendation excludes the seed id."""

    fn = registered["recommend_similar"]
    res = await fn(
        point_id="d0",
        collection="documentation",
        limit=5,
        score_threshold=0.0,
        filters=None,
        ctx=None,
    )
    assert all(r.id != "d0" for r in res)
    assert all(r.metadata is not None for r in res)


@pytest.mark.asyncio
async def test_reranked_search_returns_limit(registered):
    """Reranked search trims to requested limit."""

    fn = registered["reranked_search"]
    req = SearchRequest(
        collection="documentation",
        query="y",
        limit=7,
        offset=0,
        include_metadata=False,
    )
    res = await fn(request=req, ctx=_Ctx())
    assert len(res) == 7


@pytest.mark.asyncio
async def test_scroll_collection_paginates(registered):
    """Scroll returns documents and next offset."""

    fn = registered["scroll_collection"]
    page1 = await fn(collection="documentation", limit=2, offset=None, ctx=None)
    assert len(page1["documents"]) == 2
    assert page1["next_offset"] == "2"
