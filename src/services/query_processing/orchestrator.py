"""Search orchestrator built on top of vector search and optional RAG.

This module provides the main search orchestrator that coordinates
query expansion, vector search, ranking, and optional RAG (Retrieval
Augmented Generation) to deliver search results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from src.contracts.retrieval import SearchRecord
from src.services.base import BaseService
from src.services.rag import (
    RAGConfig as ServiceRAGConfig,
    RAGGenerator,
    RAGRequest,
    RAGResult,
    VectorServiceRetriever,
)
from src.services.vector_db.adapter_base import VectorMatch
from src.services.vector_db.service import VectorStoreService

from .expansion import (
    QueryExpansionRequest,
    QueryExpansionResult,
    QueryExpansionService,
)
from .models import SearchRequest, SearchResponse
from .ranking import (
    PersonalizedRankingRequest,
    PersonalizedRankingResponse,
    PersonalizedRankingService,
)


logger = logging.getLogger(__name__)

DEFAULT_MAX_EXPANDED_TERMS = 10


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
        vector_store_service: VectorStoreService | None = None,
        *,
        expansion_service: QueryExpansionService | None = None,
        ranking_service: PersonalizedRankingService | None = None,
        rag_config: ServiceRAGConfig | None = None,
    ) -> None:
        """Initialize the search orchestrator.

        Args:
            vector_store_service: Service for vector database operations.
            expansion_service: Service for query expansion.
            ranking_service: Service for personalized result ranking.
            rag_config: Configuration for RAG (optional).
        """
        super().__init__(None)
        self._vector_store_service: VectorStoreService = (
            vector_store_service or VectorStoreService()
        )
        self._expansion_service: QueryExpansionService = (
            expansion_service or QueryExpansionService()
        )
        self._ranking_service: PersonalizedRankingService = (
            ranking_service or PersonalizedRankingService()
        )
        self._rag_config = rag_config

    async def initialize(self) -> None:
        """Initialize all dependent services.

        This method initializes the vector store service, expansion service,
        and ranking service, ensuring they are ready for search operations.
        """
        if self._initialized:
            return
        await self._vector_store_service.initialize()
        await self._expansion_service.initialize()
        await self._ranking_service.initialize()
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
        matches = await self._vector_store_service.search_documents(
            collection=collection,
            query=processed_query,
            limit=request.limit,
            filters=request.filters,
            group_by=request.group_by,
            group_size=request.group_size,
            overfetch_multiplier=request.overfetch_multiplier,
            normalize_scores=request.normalize_scores,
        )
        records = self._build_search_records(
            matches,
            collection=collection,
            normalize_applied=request.normalize_scores,
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

        max_expanded_terms = min(DEFAULT_MAX_EXPANDED_TERMS, request.limit)
        expansion_request = QueryExpansionRequest(
            original_query=query,
            max_expanded_terms=max_expanded_terms,
        )
        expansion: QueryExpansionResult = await self._expansion_service.expand_query(
            expansion_request
        )
        features_used.append("query_expansion")
        return expansion.expanded_query, expansion.expanded_query

    async def _maybe_apply_personalization(
        self,
        request: SearchRequest,
        records: list[SearchRecord],
        features_used: list[str],
    ) -> list[SearchRecord]:
        """Apply personalized ranking when enabled and results are available."""

        if not (request.enable_personalization and records):
            return records

        ranking_request = PersonalizedRankingRequest(
            user_id=request.user_id,
            results=[record.model_dump() for record in records],
            preferences=request.user_preferences,
        )
        ranking_response: PersonalizedRankingResponse = (
            await self._ranking_service.rank_results(ranking_request)
        )
        features_used.append("personalized_ranking")

        ranked_records: list[SearchRecord] = []
        for item in ranking_response.ranked_results:
            payload = dict(item.metadata)
            ranked_records.append(
                SearchRecord(
                    id=item.result_id,
                    content=item.content,
                    title=item.title,
                    score=item.final_score,
                    metadata=payload,
                    raw_score=item.metadata.get("raw_score"),
                    normalized_score=item.metadata.get("normalized_score"),
                    collection=payload.get("collection"),
                    group_id=payload.get("_grouping", {}).get("group_id"),
                    group_rank=payload.get("_grouping", {}).get("rank"),
                    grouping_applied=payload.get("_grouping", {}).get("applied"),
                )
            )
        return ranked_records

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

    @staticmethod
    def _build_search_records(
        matches: list[VectorMatch],
        *,
        collection: str,
        normalize_applied: bool,
    ) -> list[SearchRecord]:
        """Build search records from vector matches.

        Args:
            matches: List of vector matches from the search.
            collection: Collection name for the search.
            normalize_applied: Whether score normalization was applied.

        Returns:
            List of search records with extracted metadata.
        """
        records: list[SearchRecord] = []
        for match in matches:
            payload = dict(match.payload or {})
            source_collection = (
                payload.get("collection")
                or payload.get("_collection")
                or match.collection
                or collection
            )
            group_info = payload.get("_grouping") or {}
            normalized_score = match.normalized_score if normalize_applied else None
            score_value = (
                normalized_score
                if normalized_score is not None
                else float(match.raw_score or match.score)
            )
            record = SearchRecord(
                id=match.id,
                content=payload.get("content") or payload.get("text") or "",
                title=payload.get("title") or payload.get("name"),
                url=payload.get("url"),
                metadata=payload,
                score=score_value,
                raw_score=float(match.raw_score or score_value),
                normalized_score=normalized_score,
                collection=source_collection,
                group_id=group_info.get("group_id"),
                group_rank=group_info.get("rank"),
                grouping_applied=group_info.get("applied"),
            )
            records.append(record)
        return records

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
        """Generate a RAG answer using the search results.

        Args:
            query: The search query.
            records: Search records to use as context.
            request: Original search request with RAG parameters.
            collection: Collection identifier used for retrieval.

        Returns:
            Dictionary containing answer, confidence, and sources, or None if RAG fails.
        """
        if self._rag_config is None:
            return None
        rag_config = ServiceRAGConfig(
            model=self._rag_config.model,
            temperature=self._rag_config.temperature,
            max_tokens=request.rag_max_tokens or self._rag_config.max_tokens,
            retriever_top_k=request.rag_top_k or self._rag_config.retriever_top_k,
            include_sources=self._rag_config.include_sources,
            confidence_from_scores=self._rag_config.confidence_from_scores,
        )
        retriever = VectorServiceRetriever(
            vector_service=self._vector_store_service,
            collection=collection,
            k=rag_config.retriever_top_k,
            rag_config=rag_config,
        )
        rag_generator = RAGGenerator(rag_config, retriever)
        await rag_generator.initialize()
        rag_request = RAGRequest(
            query=query,
            top_k=rag_config.retriever_top_k,
            filters=request.filters,
            max_tokens=rag_config.max_tokens,
            temperature=rag_config.temperature,
            include_sources=rag_config.include_sources,
        )
        try:
            rag_result: RAGResult = await rag_generator.generate_answer(rag_request)
        except Exception:  # pragma: no cover - defensive
            logger.exception("RAG generation failed")
            return None
        finally:
            await rag_generator.cleanup()
        return {
            "answer": rag_result.answer,
            "confidence": rag_result.confidence_score,
            "sources": [source.model_dump() for source in rag_result.sources],
        }


__all__ = ["SearchOrchestrator", "SearchRequest", "SearchResponse"]
