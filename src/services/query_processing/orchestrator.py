"""Search orchestrator built on top of vector search and optional RAG.

This module provides the main search orchestrator that coordinates
query expansion, vector search, ranking, and optional RAG (Retrieval
Augmented Generation) to deliver search results.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from src.contracts.retrieval import SearchRecord, SearchResponse
from src.models.search import SearchRequest
from src.services.base import BaseService
from src.services.rag.langgraph_pipeline import LangGraphRAGPipeline
from src.services.rag.models import RAGConfig as ServiceRAGConfig
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

DEFAULT_MAX_EXPANDED_TERMS = 10
_DEFAULT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "install": ("setup", "configure"),
    "configuration": ("settings", "setup"),
    "error": ("issue", "problem"),
    "update": ("upgrade", "patch"),
    "delete": ("remove", "uninstall"),
    "create": ("make", "build"),
    "performance": ("speed", "throughput"),
    "security": ("auth", "authorization"),
}


@dataclass(slots=True)
class _RAGInvocationContext:
    """Arguments required to invoke the RAG generation pipeline."""

    query: str
    records: list[SearchRecord]
    request: SearchRequest
    collection: str


class SearchOrchestrator(BaseService):
    """Coordinate query expansion, vector search, ranking, and optional RAG.

    This orchestrator serves as the central hub for search operations,
    coordinating multiple services to provide search
    functionality including query expansion, vector search, personalized
    ranking, and optional RAG answer generation.
    """

    def __init__(
        self,
        *,
        vector_store_service: VectorStoreService,
        rag_config: ServiceRAGConfig | None = None,
    ) -> None:
        """Initialize the search orchestrator.

        Args:
            vector_store_service: Service for vector database operations.
            rag_config: Configuration for RAG (optional).
        """
        super().__init__(None)
        self._vector_store_service: VectorStoreService = vector_store_service
        self._rag_config = rag_config
        self._rag_pipeline: LangGraphRAGPipeline | None = None

    async def initialize(self) -> None:
        """Initialize all dependent services.

        This method initializes the vector store service, expansion service,
        and ranking service, ensuring they are ready for search operations.
        """
        if self._initialized:
            return
        await self._vector_store_service.initialize()
        self._initialized = True

    async def cleanup(self) -> None:  # pragma: no cover - symmetrical teardown
        """Clean up resources and mark as uninitialized.

        This method cleans up the vector store service and marks the
        orchestrator as uninitialized.
        """
        self._initialized = False
        await self._vector_store_service.cleanup()

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a search operation.

        This method orchestrates the entire search pipeline including query
        expansion, vector search, personalized ranking, and optional RAG
        answer generation.

        Args:
            request: Search request containing query and parameters.

        Returns:
            Search response containing results and metadata.
        """
        self._validate_initialized()
        start_time = time.perf_counter()

        processed_query = request.query.strip()
        expanded_query = None
        features_used: list[str] = []

        if processed_query:
            processed_query, expanded_query = await self._maybe_expand_query(
                request,
                processed_query,
                features_used,
            )

        collection = await self._resolve_collection(request)
        records = await self._vector_store_service.search_documents(
            collection=collection,
            query=processed_query,
            limit=request.limit,
            filters=request.filters,
            group_by=request.group_by,
            group_size=request.group_size,
            overfetch_multiplier=request.overfetch_multiplier,
            normalize_scores=request.normalize_scores,
        )

        grouping_applied = any((record.grouping_applied is True) for record in records)

        records = await self._maybe_apply_personalization(
            request,
            records,
            features_used,
        )

        (
            rag_answer,
            rag_confidence,
            rag_sources,
        ) = await self._maybe_generate_rag_outputs(
            _RAGInvocationContext(
                query=processed_query,
                records=records,
                request=request,
                collection=collection,
            ),
            features_used,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if request.normalize_scores and any(
            record.normalized_score is not None for record in records
        ):
            features_used.append("score_normalization")
        return SearchResponse(
            records=records,
            total_results=len(records),
            query=processed_query,
            expanded_query=expanded_query,
            processing_time_ms=elapsed_ms,
            features_used=features_used,
            grouping_applied=grouping_applied,
            generated_answer=rag_answer,
            answer_confidence=rag_confidence,
            answer_sources=rag_sources,
        )

    async def _maybe_expand_query(
        self,
        request: SearchRequest,
        query: str,
        features_used: list[str],
    ) -> tuple[str, str | None]:
        """Expand a query when the feature is enabled."""
        if not request.enable_expansion:
            return query, None

        expanded_query = _expand_query_terms(
            query,
            max_terms=min(DEFAULT_MAX_EXPANDED_TERMS, request.limit),
        )
        if expanded_query is None:
            return query, None
        features_used.append("query_expansion")
        return expanded_query, expanded_query

    async def _maybe_apply_personalization(
        self,
        request: SearchRequest,
        records: list[SearchRecord],
        features_used: list[str],
    ) -> list[SearchRecord]:
        """Apply personalized ranking when enabled and results are available."""
        if not (request.enable_personalization and records):
            return records
        preferences = request.user_preferences or {}
        if not preferences:
            return records

        base_scores = [float(record.score) for record in records]
        metadata = [dict(record.metadata or {}) for record in records]
        adjusted_scores = _apply_preferences(base_scores, metadata, preferences)
        normalized_scores = _normalize_scores(adjusted_scores)

        ranked = list(zip(records, normalized_scores, strict=False))
        ranked.sort(key=lambda item: item[1], reverse=True)

        features_used.append("personalized_ranking")

        personalized: list[SearchRecord] = []
        for record, score in ranked:
            payload = dict(record.metadata or {})
            payload.setdefault("preferences", preferences)
            personalized.append(
                SearchRecord(
                    id=record.id,
                    content=record.content,
                    title=record.title,
                    score=score,
                    metadata=payload,
                    url=record.url,
                    content_type=record.content_type,
                    raw_score=record.raw_score,
                    normalized_score=score,
                    group_id=record.group_id,
                    group_rank=record.group_rank,
                    grouping_applied=record.grouping_applied,
                )
            )
        return personalized

    async def _maybe_generate_rag_outputs(
        self,
        context: _RAGInvocationContext,
        features_used: list[str],
    ) -> tuple[str | None, float | None, list[dict[str, Any]] | None]:
        """Generate RAG outputs when enabled and capable."""
        if not (
            context.request.enable_rag
            and self._rag_config is not None
            and context.records
        ):
            return None, None, None

        rag_result = await self._generate_rag_answer(
            context.query,
            context.records,
            context.request,
            context.collection,
        )
        if not rag_result:
            return None, None, None

        features_used.append("rag_answer_generation")
        return (
            rag_result["answer"],
            rag_result.get("confidence"),
            rag_result.get("sources"),
        )

    async def _resolve_collection(self, request: SearchRequest) -> str:
        """Determine which collection should be queried for this request.

        Args:
            request: Search request containing collection information.

        Returns:
            Collection name to use for the search.
        """
        if request.collection:
            return request.collection

        config = getattr(self._vector_store_service, "config", None)
        qdrant_config = getattr(config, "qdrant", None)
        default_collection = getattr(qdrant_config, "collection_name", None)
        if default_collection:
            return default_collection

        collections = await self._vector_store_service.list_collections()
        if not collections:
            msg = "Unable to determine a default collection for search"
            raise RuntimeError(msg)
        return collections[0]

    async def _generate_rag_answer(
        self,
        query: str,
        records: list[SearchRecord],
        request: SearchRequest,
        collection: str,
    ) -> dict[str, Any] | None:
        """Generate a RAG answer using the LangGraph pipeline."""
        if self._rag_config is None:
            return None

        rag_config = ServiceRAGConfig(
            model=self._rag_config.model,
            temperature=self._rag_config.temperature,
            max_tokens=request.rag_max_tokens or self._rag_config.max_tokens,
            retriever_top_k=request.rag_top_k or self._rag_config.retriever_top_k,
            include_sources=self._rag_config.include_sources,
            confidence_from_scores=self._rag_config.confidence_from_scores,
            compression_enabled=self._rag_config.compression_enabled,
            compression_similarity_threshold=self._rag_config.compression_similarity_threshold,
            compression_mmr_lambda=self._rag_config.compression_mmr_lambda,
            compression_token_ratio=self._rag_config.compression_token_ratio,
            compression_absolute_max_tokens=self._rag_config.compression_absolute_max_tokens,
            compression_min_sentences=self._rag_config.compression_min_sentences,
            compression_max_sentences=self._rag_config.compression_max_sentences,
        )

        if self._rag_pipeline is None:
            self._rag_pipeline = LangGraphRAGPipeline(self._vector_store_service)

        try:
            return await self._rag_pipeline.run(
                query=query,
                request=request,
                rag_config=rag_config,
                collection=collection,
                prefetched_records=records,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("LangGraph RAG pipeline failed")
            return None


def _split_terms(query: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", query.lower())


def _expand_query_terms(query: str, *, max_terms: int) -> str | None:
    seen: set[str] = set()
    expansions: list[str] = []

    for term in _split_terms(query):
        if term in seen:
            continue
        seen.add(term)
        expansions.append(term)

        for variant in (f"{term}s", f"{term}ing"):
            if variant not in seen:
                seen.add(variant)
                expansions.append(variant)

        synonyms = cast(tuple[str, ...], _DEFAULT_SYNONYMS.get(term) or ())
        for synonym in synonyms:
            if synonym not in seen:
                seen.add(synonym)
                expansions.append(synonym)

        if len(expansions) >= max_terms:
            break

    if len(expansions) <= 1:
        return None
    additional = " OR ".join(expansions[1:])
    return f"{query} ({additional})" if additional else None


def _apply_preferences(
    base_scores: list[float],
    metadata: list[dict[str, Any]],
    preferences: Mapping[str, float],
) -> list[float]:
    mapping = {key.lower(): float(value) for key, value in preferences.items()}
    adjusted: list[float] = []
    for score, meta in zip(base_scores, metadata, strict=False):
        boost = 0.0
        categories = meta.get("categories") if isinstance(meta, dict) else None
        if isinstance(categories, list):
            for category in categories:
                if isinstance(category, str):
                    boost += mapping.get(category.lower(), 0.0)
        adjusted.append(score + 0.1 * boost)
    return adjusted


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if minimum == maximum:
        return [1.0 for _ in scores]
    span = maximum - minimum
    return [(score - minimum) / span for score in scores]


__all__ = ["SearchOrchestrator"]
