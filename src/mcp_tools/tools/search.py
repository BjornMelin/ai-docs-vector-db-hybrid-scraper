# pylint: disable=too-many-statements

"""Search tools backed by the unified VectorStoreService.

All tools in this module delegate to :class:`VectorStoreService` instead of the
legacy ``QdrantService`` shim. This keeps the MCP surface aligned with the
shared adapter layer while preserving the existing tool signatures.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager
from src.services.vector_db.adapter_base import VectorMatch
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "documentation"
_DEFAULT_SCORE_THRESHOLD = 0.7
_MAX_MULTI_STAGE = 5
_MIN_MULTI_STAGE = 2


async def _get_vector_service(client_manager: ClientManager) -> VectorStoreService:
    """Return an initialised vector store service instance."""

    service = await client_manager.get_vector_store_service()
    if not service.is_initialized():
        await service.initialize()
    return service


def _format_match(match: VectorMatch, *, method: str | None = None) -> dict[str, Any]:
    """Convert a VectorMatch into an MCP-friendly response mapping."""

    payload = dict(match.payload or {})
    result: dict[str, Any] = {
        "id": match.id,
        "score": float(match.score),
        "payload": payload,
    }
    if method:
        result["method"] = method
    return result


def _normalize_filters(
    filter_conditions: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return filters in the shape expected by the adapter layer."""

    if not filter_conditions:
        return None
    normalized: dict[str, Any] = {}
    for key, value in filter_conditions.items():
        if isinstance(value, dict):
            normalized[key] = value
        elif isinstance(value, (list, tuple, set)):
            normalized[key] = list(value)
        else:
            normalized[key] = value
    return normalized


def _build_boolean_filters(
    must: list[dict[str, Any]] | None,
    should: list[dict[str, Any]] | None,
    must_not: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Convert boolean filter clauses into adapter-friendly mappings."""

    if not any((must, should, must_not)):
        return None

    def _convert(conditions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not conditions:
            return []
        converted: list[dict[str, Any]] = []
        for condition in conditions:
            if "key" not in condition:
                raise ValueError("Filter condition must include a 'key' entry")
            entry: dict[str, Any] = {"key": condition["key"]}
            if "range" in condition:
                entry["range"] = condition["range"]
            elif "values" in condition:
                entry["values"] = condition["values"]
            elif "in" in condition:
                entry["in"] = condition["in"]
            elif "value" in condition:
                entry["value"] = condition["value"]
            else:
                raise ValueError(
                    "Unsupported filter condition format; supply 'value', "
                    "'values', 'in', or 'range'"
                )
            converted.append(entry)
        return converted

    payload: dict[str, Any] = {}
    if converted_must := _convert(must):
        payload["must"] = converted_must
    if converted_should := _convert(should):
        payload["should"] = converted_should
    if converted_must_not := _convert(must_not):
        payload["must_not"] = converted_must_not
    return payload or None


def _deduplicate_by_score(matches: Iterable[VectorMatch]) -> list[VectorMatch]:
    """Remove duplicates by identifier, keeping the highest-score entry."""

    ranked: OrderedDict[str, VectorMatch] = OrderedDict()
    for match in matches:
        existing = ranked.get(match.id)
        if existing is None or match.score > existing.score:
            ranked[match.id] = match
    return list(ranked.values())


def _extract_vector(match: VectorMatch) -> Sequence[float]:
    """Return the dense vector from a VectorMatch, if available."""

    if match.vector is None:
        raise ValueError("Vector data unavailable for the requested document")
    return match.vector


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Register search-related MCP tools using the vector service."""

    async def _search(
        *,
        query: str,
        collection: str,
        limit: int,
        filters: dict[str, Any] | None,
        ctx: Context | None,
    ) -> list[dict[str, Any]]:
        service = await _get_vector_service(client_manager)
        try:
            matches = await service.search_documents(
                collection,
                query,
                limit=limit,
                filters=filters,
            )
            return [_format_match(match) for match in matches]
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Vector search failed")
            if ctx:
                await ctx.error(f"Search error: {exc}")
            return []

    async def _hybrid(
        *,
        query: str,
        collection: str,
        limit: int,
        filters: dict[str, Any] | None,
        ctx: Context | None,
    ) -> list[dict[str, Any]]:
        service = await _get_vector_service(client_manager)
        try:
            matches = await service.hybrid_search(
                collection,
                query,
                limit=limit,
                filters=filters,
            )
            return [_format_match(match) for match in matches]
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Hybrid search failed")
            if ctx:
                await ctx.error(f"Hybrid search error: {exc}")
            return []

    @mcp.tool()
    async def search_documents(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        query: str,
        collection: str = _DEFAULT_COLLECTION,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Perform dense vector search using the shared vector service."""

        filters = _normalize_filters(filter_conditions)
        results = await _search(
            query=query,
            collection=collection,
            limit=limit,
            filters=filters,
            ctx=ctx,
        )
        if score_threshold is None:
            return results
        return [
            result for result in results if result.get("score", 0.0) >= score_threshold
        ]

    @mcp.tool()
    async def hybrid_search(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        query: str,
        collection: str = _DEFAULT_COLLECTION,
        limit: int = 10,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Perform sparse+dense hybrid retrieval."""

        filters = _normalize_filters(filter_conditions)
        return await _hybrid(
            query=query,
            collection=collection,
            limit=limit,
            filters=filters,
            ctx=ctx,
        )

    @mcp.tool()
    async def scroll_collection(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        collection: str = _DEFAULT_COLLECTION,
        limit: int = 100,
        filter_conditions: dict[str, Any] | None = None,
        offset: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Paginate through an entire collection."""

        filters = _normalize_filters(filter_conditions)
        service = await _get_vector_service(client_manager)
        try:
            matches, next_offset = await service.scroll(
                collection,
                limit=limit,
                offset=offset,
                filters=filters,
                with_payload=True,
            )
            payload = {
                "points": [
                    {
                        "id": match.id,
                        "payload": dict(match.payload or {}),
                    }
                    for match in matches
                ],
                "count": len(matches),
                "next_page_offset": next_offset,
            }
            if ctx:
                await ctx.info(
                    f"Retrieved {payload['count']} points from '{collection}'"
                )
            return payload
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Scroll operation failed")
            if ctx:
                await ctx.error(f"Scroll error: {exc}")
            return {"points": [], "count": 0, "next_page_offset": None}

    @mcp.tool()
    async def search_with_context(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        query: str,
        collection: str = _DEFAULT_COLLECTION,
        limit: int = 10,
        context_size: int = 3,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Return primary search results along with additional context hits."""

        filters = _normalize_filters(filter_conditions)
        effective_limit = max(limit + max(context_size, 0), limit)
        results = await _hybrid(
            query=query,
            collection=collection,
            limit=effective_limit,
            filters=filters,
            ctx=ctx,
        )
        if ctx:
            await ctx.info(
                f"Context search produced {len(results)} candidates for '{collection}'"
            )
        return results[:effective_limit]

    @mcp.tool()
    async def recommend_similar(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        point_id: str,
        collection: str = _DEFAULT_COLLECTION,
        limit: int = 10,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Recommend documents similar to the supplied identifier."""

        filters = _normalize_filters(filter_conditions)
        service = await _get_vector_service(client_manager)
        try:
            documents = await service.retrieve_documents(
                collection,
                [point_id],
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load source document for recommendation")
            if ctx:
                await ctx.error(f"Recommendation error: {exc}")
            raise

        if not documents:
            msg = f"Document {point_id} not found"
            logger.error(msg)
            if ctx:
                await ctx.error(msg)
            raise ValueError(msg)

        document = documents[0]

        try:
            base_vector = _extract_vector(document)
            matches = await service.recommend_similar(
                collection,
                vector=base_vector,
                limit=limit + 1,  # allow excluding the seed document
                filters=filters,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Recommendation query failed")
            if ctx:
                await ctx.error(f"Recommendation error: {exc}")
            raise

        filtered_matches = [
            match
            for match in matches
            if match.id != point_id and match.score >= score_threshold
        ]
        if ctx:
            await ctx.info(
                f"Recommendation produced {len(filtered_matches)} candidates"
            )
        return [
            _format_match(match, method="recommend_similar")
            for match in filtered_matches[:limit]
        ]

    @mcp.tool()
    async def hyde_search(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        query: str,
        collection: str = _DEFAULT_COLLECTION,
        limit: int = 10,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Compatibility wrapper around hybrid search for HyDE requests."""

        results = await hybrid_search(
            query=query,
            collection=collection,
            limit=limit,
            filter_conditions=filter_conditions,
            ctx=ctx,
        )
        return [result | {"method": "hyde"} for result in results]

    @mcp.tool()
    async def reranked_search(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        query: str,
        collection: str = _DEFAULT_COLLECTION,
        limit: int = 10,
        rerank_limit: int = 50,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a search and apply Reciprocal Rank Fusion scoring."""

        filters = _normalize_filters(filter_conditions)
        rerank_limit = max(rerank_limit, limit)
        matches = await _hybrid(
            query=query,
            collection=collection,
            limit=rerank_limit,
            filters=filters,
            ctx=ctx,
        )
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, match in enumerate(matches, start=1):
            rrf_score = 1.0 / (60 + idx)
            scored.append((rrf_score, match))
        scored.sort(key=lambda item: item[0], reverse=True)
        result = [
            match | {"method": "rrf_reranked", "rerank_score": score}
            for score, match in scored[:limit]
        ]
        if ctx:
            await ctx.info(
                f"Reranked search collapsed {rerank_limit} → {len(result)} results"
            )
        return result

    @mcp.tool()
    async def multi_stage_search(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        query: str,
        collection: str = _DEFAULT_COLLECTION,
        limit: int = 10,
        stages: int = 3,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a progressive multi-stage search pipeline."""

        filters = _normalize_filters(filter_conditions)
        service = await _get_vector_service(client_manager)
        num_stages = max(min(stages, _MAX_MULTI_STAGE), _MIN_MULTI_STAGE)
        stage_limits = [
            min(limit * (10 ** (num_stages - idx)), 1000) for idx in range(num_stages)
        ]

        collected: list[VectorMatch] = []
        for stage_limit in stage_limits:
            try:
                stage_matches = await service.hybrid_search(
                    collection,
                    query,
                    limit=stage_limit,
                    filters=filters,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Multi-stage search stage failed")
                if ctx:
                    await ctx.warning(f"Stage with limit {stage_limit} failed: {exc}")
                continue
            collected.extend(stage_matches)

        deduped = _deduplicate_by_score(collected)
        if ctx:
            await ctx.info(
                f"Multi-stage search aggregated {len(deduped)} unique candidates"
            )
        return [
            _format_match(match, method=f"multi_stage_{num_stages}")
            for match in deduped[:limit]
        ]

    @mcp.tool()
    async def filtered_search(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        query: str,
        collection: str = _DEFAULT_COLLECTION,
        limit: int = 10,
        must_conditions: list[dict[str, Any]] | None = None,
        should_conditions: list[dict[str, Any]] | None = None,
        must_not_conditions: list[dict[str, Any]] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a search constrained by boolean payload filters."""

        filters = _build_boolean_filters(
            must_conditions, should_conditions, must_not_conditions
        )
        if filters is None:
            filters = {}
        matches = await _search(
            query=query,
            collection=collection,
            limit=limit,
            filters=filters,
            ctx=ctx,
        )
        return [result | {"method": "boolean_filter"} for result in matches]
