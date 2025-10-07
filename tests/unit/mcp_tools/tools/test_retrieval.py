"""Tests for unified retrieval tools."""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.mcp_tools.tools import retrieval


@pytest_asyncio.fixture()
async def registered(fake_mcp, fake_client_manager):
    """Register tools and yield registry."""

    retrieval.register_tools(fake_mcp, fake_client_manager)
    return fake_mcp.tools


@pytest.mark.asyncio
async def test_search_documents_basic(registered):
    """Search returns SearchResult objects."""

    fn = registered["search_documents"]
    res = await fn(
        request=type(
            "Req",
            (),
            {
                "collection": "documentation",
                "query": "hello",
                "limit": 3,
                "filters": None,
                "include_metadata": True,
                "strategy": type("S", (), {"value": "hybrid"})(),
            },
        )(),
        ctx=type("C", (), {"info": lambda *_: None})(),
    )
    assert len(res) == 3
    assert res[0].id == "1"
    assert res[0].metadata is not None


@pytest.mark.asyncio
async def test_filtered_search_forwards_filters(registered):
    """Filtered search forwards filters."""

    fn = registered["filtered_search"]
    res = await fn(
        request=type(
            "Req",
            (),
            {
                "collection": "documentation",
                "query": "q",
                "limit": 2,
                "filters": {"site_name": {"value": "docs"}},
                "include_metadata": True,
            },
        )(),
        ctx=type("C", (), {"info": lambda *_: None})(),
    )
    assert res and res[0].metadata and res[0].metadata["q"] == "q"


@pytest.mark.asyncio
async def test_multi_stage_merges_and_dedupes(registered):
    """Multi-stage returns unique items by id."""

    fn = registered["multi_stage_search"]
    req = type(
        "Req",
        (),
        {
            "collection": "documentation",
            "query": "x",
            "limit": 5,
            "stages": [{"limit": 5}, {"limit": 5, "filters": {"k": 1}}],
            "include_metadata": False,
        },
    )()
    res = await fn(request=req, ctx=type("C", (), {"info": lambda *_: None})())
    ids = [r.id for r in res]
    assert len(ids) == len(set(ids))


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


@pytest.mark.asyncio
async def test_reranked_search_returns_limit(registered):
    """Reranked search trims to requested limit."""

    fn = registered["reranked_search"]
    req = type(
        "Req",
        (),
        {
            "collection": "documentation",
            "query": "y",
            "limit": 7,
            "filters": None,
            "include_metadata": False,
            "strategy": type("S", (), {"value": "hybrid"})(),
        },
    )()
    res = await fn(request=req, ctx=type("C", (), {"info": lambda *_: None})())
    assert len(res) == 7


@pytest.mark.asyncio
async def test_scroll_collection_paginates(registered):
    """Scroll returns documents and next offset."""

    fn = registered["scroll_collection"]
    page1 = await fn(collection="documentation", limit=2, offset=None, ctx=None)
    assert len(page1["documents"]) == 2
    assert page1["next_offset"] == "2"
