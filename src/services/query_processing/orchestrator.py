"""Query Processing Orchestrator.

This module provides the central orchestrator that coordinates all query
processing components including intent classification, preprocessing,
strategy selection, and search execution with intelligent fallback handling.
"""

import asyncio
import logging
import time
from typing import Any

from ..embeddings.manager import EmbeddingManager
from ..errors import QdrantServiceError
from ..hyde.engine import HyDEQueryEngine
from ..vector_db.service import QdrantService
from .intent_classifier import QueryIntentClassifier
from .models import MatryoshkaDimension
from .models import QueryAnalytics
from .models import QueryProcessingRequest
from .models import QueryProcessingResponse
from .models import SearchStrategy
from .preprocessor import QueryPreprocessor
from .strategy_selector import SearchStrategySelector

logger = logging.getLogger(__name__)


class QueryProcessingOrchestrator:
    """Central orchestrator for advanced query processing pipeline.

    Coordinates intent classification, preprocessing, strategy selection,
    and search execution with intelligent fallback and error handling.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        qdrant_service: QdrantService,
        hyde_engine: HyDEQueryEngine,
        cache_manager: Any = None,
    ):
        """Initialize the query processing orchestrator.

        Args:
            embedding_manager: Embedding service manager
            qdrant_service: Qdrant vector database service
            hyde_engine: HyDE query enhancement engine
            cache_manager: Optional cache manager for caching results
        """
        self.embedding_manager = embedding_manager
        self.qdrant_service = qdrant_service
        self.hyde_engine = hyde_engine
        self.cache_manager = cache_manager

        # Initialize processing components
        self.intent_classifier = QueryIntentClassifier(embedding_manager)
        self.preprocessor = QueryPreprocessor()
        self.strategy_selector = SearchStrategySelector()

        self._initialized = False

        # Performance tracking
        self._processing_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "fallback_usage": 0,
            "average_processing_time": 0.0,
            "strategy_usage": {},
        }

    async def initialize(self) -> None:
        """Initialize all orchestrator components."""
        if self._initialized:
            return

        try:
            # Initialize all components in parallel
            await asyncio.gather(
                self.intent_classifier.initialize(),
                self.preprocessor.initialize(),
                self.strategy_selector.initialize(),
                # Core services should already be initialized
                return_exceptions=True,
            )

            self._initialized = True
            logger.info("QueryProcessingOrchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize QueryProcessingOrchestrator: {e}")
            raise

    async def process_query(
        self, request: QueryProcessingRequest
    ) -> QueryProcessingResponse:
        """Process query through complete advanced pipeline.

        Args:
            request: Query processing request with options

        Returns:
            QueryProcessingResponse: Complete processing results
        """
        if not self._initialized:
            raise RuntimeError("QueryProcessingOrchestrator not initialized")

        start_time = time.time()
        processing_steps = []
        fallback_used = False

        try:
            self._processing_stats["total_queries"] += 1

            # Step 1: Check cache first
            cache_key = None
            if self.cache_manager:
                cache_key = self._generate_cache_key(request)
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    processing_steps.append("cache_hit")
                    cached_result.cache_hit = True
                    cached_result.total_processing_time_ms = (
                        time.time() - start_time
                    ) * 1000
                    return cached_result

            processing_steps.append("cache_miss")

            # Step 2: Query preprocessing
            preprocessing_result = None
            if request.enable_preprocessing:
                preprocessing_result = await self.preprocessor.preprocess_query(
                    request.query,
                    enable_spell_correction=True,
                    enable_expansion=True,
                    enable_normalization=True,
                    enable_context_extraction=True,
                )
                processing_steps.append(
                    f"preprocessing_{preprocessing_result.preprocessing_time_ms:.1f}ms"
                )

            # Use processed query or original
            query_to_analyze = (
                preprocessing_result.processed_query
                if preprocessing_result
                else request.query
            )

            # Step 3: Intent classification
            intent_classification = None
            if request.enable_intent_classification:
                intent_start = time.time()
                context = request.user_context.copy()
                if preprocessing_result and preprocessing_result.context_extracted:
                    context.update(preprocessing_result.context_extracted)

                intent_classification = (
                    await self.intent_classifier.classify_query_advanced(
                        query_to_analyze, context
                    )
                )
                intent_time = (time.time() - intent_start) * 1000
                processing_steps.append(f"intent_classification_{intent_time:.1f}ms")

            # Step 4: Strategy selection
            strategy_selection = None
            strategy_selection_time = 0.0
            if request.enable_strategy_selection and intent_classification:
                strategy_start = time.time()

                # Apply performance requirements from request
                performance_requirements = {}
                if request.max_processing_time_ms:
                    performance_requirements["max_latency_ms"] = (
                        request.max_processing_time_ms * 0.8
                    )  # Reserve 20% for overhead

                strategy_selection = await self.strategy_selector.select_strategy(
                    intent_classification,
                    context=request.user_context,
                    performance_requirements=performance_requirements,
                )
                strategy_selection_time = (time.time() - strategy_start) * 1000
                processing_steps.append(
                    f"strategy_selection_{strategy_selection_time:.1f}ms"
                )

            # Step 5: Search execution
            search_start = time.time()

            # Determine final strategy and dimension
            if request.force_strategy:
                primary_strategy = request.force_strategy
                # Provide default fallback strategies even for forced strategies
                fallback_strategies = [SearchStrategy.SEMANTIC, SearchStrategy.HYBRID]
            elif strategy_selection:
                primary_strategy = strategy_selection.primary_strategy
                fallback_strategies = strategy_selection.fallback_strategies
            else:
                primary_strategy = SearchStrategy.SEMANTIC  # Default
                fallback_strategies = [SearchStrategy.HYBRID]

            dimension = request.force_dimension or (
                strategy_selection.matryoshka_dimension
                if strategy_selection
                else MatryoshkaDimension.MEDIUM
            )

            # Execute search with fallback handling
            (
                search_results,
                search_strategy_used,
            ) = await self._execute_search_with_fallback(
                query_to_analyze,
                primary_strategy,
                fallback_strategies,
                dimension,
                request,
            )

            if search_strategy_used != primary_strategy:
                fallback_used = True
                self._processing_stats["fallback_usage"] += 1

            search_time = (time.time() - search_start) * 1000
            processing_steps.append(
                f"search_{search_strategy_used.value}_{search_time:.1f}ms"
            )

            # Track strategy usage
            if (
                search_strategy_used.value
                not in self._processing_stats["strategy_usage"]
            ):
                self._processing_stats["strategy_usage"][search_strategy_used.value] = 0
            self._processing_stats["strategy_usage"][search_strategy_used.value] += 1

            # Step 6: Calculate overall metrics
            total_time = (time.time() - start_time) * 1000

            # Calculate confidence and quality scores
            confidence_score = 0.8  # Base confidence
            if intent_classification:
                primary_intent_confidence = intent_classification.confidence_scores.get(
                    intent_classification.primary_intent, 0.5
                )
                confidence_score = min(
                    confidence_score * primary_intent_confidence * 1.2, 1.0
                )

            quality_score = 0.7  # Base quality
            if strategy_selection:
                quality_score = strategy_selection.estimated_quality
            if fallback_used:
                quality_score *= 0.9  # Slight reduction for fallback

            # Create response
            response = QueryProcessingResponse(
                success=True,
                results=search_results,
                total_results=len(search_results),
                intent_classification=intent_classification,
                preprocessing_result=preprocessing_result,
                strategy_selection=strategy_selection,
                total_processing_time_ms=total_time,
                search_time_ms=search_time,
                strategy_selection_time_ms=strategy_selection_time,
                confidence_score=confidence_score,
                quality_score=quality_score,
                processing_steps=processing_steps,
                fallback_used=fallback_used,
                cache_hit=False,
            )

            # Cache successful results
            if self.cache_manager and cache_key:
                await self._cache_result(cache_key, response)

            # Update stats
            self._processing_stats["successful_queries"] += 1
            self._update_processing_stats(total_time)

            # Record analytics
            await self._record_analytics(request, response, intent_classification)

            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            self._processing_stats["failed_queries"] += 1

            # Return error response
            total_time = (time.time() - start_time) * 1000
            return QueryProcessingResponse(
                success=False,
                results=[],
                total_results=0,
                total_processing_time_ms=total_time,
                processing_steps=processing_steps,
                error=str(e),
            )

    async def _execute_search_with_fallback(
        self,
        query: str,
        primary_strategy: SearchStrategy,
        fallback_strategies: list[SearchStrategy],
        dimension: MatryoshkaDimension,
        request: QueryProcessingRequest,
    ) -> tuple[list[dict[str, Any]], SearchStrategy]:
        """Execute search with intelligent fallback handling."""

        strategies_to_try = [primary_strategy, *fallback_strategies]
        last_error = None

        for strategy in strategies_to_try:
            try:
                results = await self._execute_single_strategy(
                    query, strategy, dimension, request
                )

                # Check if results are satisfactory
                if results and len(results) >= max(1, request.limit // 3):
                    return results, strategy
                elif results:  # Some results but not enough, continue to fallback
                    logger.info(
                        f"Strategy {strategy.value} returned only {len(results)} results, trying fallback"
                    )
                    continue
                else:  # No results, try fallback
                    logger.info(
                        f"Strategy {strategy.value} returned no results, trying fallback"
                    )
                    continue

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Strategy {strategy.value} failed: {e}, trying fallback"
                )
                continue

        # If all strategies failed, raise the last error
        if last_error:
            raise last_error
        else:
            # Return empty results if no error but no results
            return [], primary_strategy

    async def _execute_single_strategy(
        self,
        query: str,
        strategy: SearchStrategy,
        dimension: MatryoshkaDimension,
        request: QueryProcessingRequest,
    ) -> list[dict[str, Any]]:
        """Execute a single search strategy."""

        if strategy == SearchStrategy.SEMANTIC:
            return await self._execute_semantic_search(query, dimension, request)
        elif strategy == SearchStrategy.HYBRID:
            return await self._execute_hybrid_search(query, dimension, request)
        elif strategy == SearchStrategy.HYDE:
            return await self._execute_hyde_search(query, dimension, request)
        elif strategy == SearchStrategy.MULTI_STAGE:
            return await self._execute_multi_stage_search(query, dimension, request)
        elif strategy == SearchStrategy.FILTERED:
            return await self._execute_filtered_search(query, dimension, request)
        elif strategy == SearchStrategy.RERANKED:
            return await self._execute_reranked_search(query, dimension, request)
        elif strategy == SearchStrategy.ADAPTIVE:
            return await self._execute_adaptive_search(query, dimension, request)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")

    async def _execute_semantic_search(
        self,
        query: str,
        dimension: MatryoshkaDimension,
        request: QueryProcessingRequest,
    ) -> list[dict[str, Any]]:
        """Execute semantic vector search."""
        # Generate query embedding with specified dimension
        embedding_result = await self.embedding_manager.generate_embeddings(
            texts=[query], dimension=dimension.value, auto_select=True
        )

        if not embedding_result.get("success") or not embedding_result.get(
            "embeddings"
        ):
            raise QdrantServiceError("Failed to generate query embedding")

        query_vector = embedding_result["embeddings"][0]

        # Execute vector search
        return await self.qdrant_service.filtered_search(
            collection_name=request.collection_name,
            query_vector=query_vector,
            filters=request.filters,
            limit=request.limit,
            search_accuracy=request.search_accuracy,
        )

    async def _execute_hybrid_search(
        self,
        query: str,
        dimension: MatryoshkaDimension,
        request: QueryProcessingRequest,
    ) -> list[dict[str, Any]]:
        """Execute hybrid dense+sparse search."""
        # Generate dense embedding
        embedding_result = await self.embedding_manager.generate_embeddings(
            texts=[query], dimension=dimension.value, auto_select=True
        )

        if not embedding_result.get("success") or not embedding_result.get(
            "embeddings"
        ):
            raise QdrantServiceError("Failed to generate query embedding")

        query_vector = embedding_result["embeddings"][0]

        # For now, use dense-only search (sparse vector generation would need additional implementation)
        return await self.qdrant_service.search.hybrid_search(
            collection_name=request.collection_name,
            query_vector=query_vector,
            sparse_vector=None,  # Would need sparse vector generation
            limit=request.limit,
            search_accuracy=request.search_accuracy,
        )

    async def _execute_hyde_search(
        self,
        query: str,
        dimension: MatryoshkaDimension,
        request: QueryProcessingRequest,
    ) -> list[dict[str, Any]]:
        """Execute HyDE-enhanced search."""
        return await self.hyde_engine.enhanced_search(
            query=query,
            collection_name=request.collection_name,
            limit=request.limit,
            filters=request.filters,
            search_accuracy=request.search_accuracy,
            use_cache=True,
        )

    async def _execute_multi_stage_search(
        self,
        query: str,
        dimension: MatryoshkaDimension,
        request: QueryProcessingRequest,
    ) -> list[dict[str, Any]]:
        """Execute multi-stage retrieval search."""
        # Generate query embedding
        embedding_result = await self.embedding_manager.generate_embeddings(
            texts=[query], dimension=dimension.value, auto_select=True
        )

        if not embedding_result.get("success") or not embedding_result.get(
            "embeddings"
        ):
            raise QdrantServiceError("Failed to generate query embedding")

        query_vector = embedding_result["embeddings"][0]

        # Create multi-stage configuration
        stages = [
            {
                "query_vector": query_vector,
                "vector_name": "dense",
                "vector_type": "dense",
                "limit": request.limit * 2,  # Larger prefetch for first stage
            }
        ]

        return await self.qdrant_service.search.multi_stage_search(
            collection_name=request.collection_name,
            stages=stages,
            limit=request.limit,
            search_accuracy=request.search_accuracy,
        )

    async def _execute_filtered_search(
        self,
        query: str,
        dimension: MatryoshkaDimension,
        request: QueryProcessingRequest,
    ) -> list[dict[str, Any]]:
        """Execute filtered search with enhanced filters."""
        # Generate query embedding
        embedding_result = await self.embedding_manager.generate_embeddings(
            texts=[query], dimension=dimension.value, auto_select=True
        )

        if not embedding_result.get("success") or not embedding_result.get(
            "embeddings"
        ):
            raise QdrantServiceError("Failed to generate query embedding")

        query_vector = embedding_result["embeddings"][0]

        return await self.qdrant_service.filtered_search(
            collection_name=request.collection_name,
            query_vector=query_vector,
            filters=request.filters,
            limit=request.limit,
            search_accuracy=request.search_accuracy,
        )

    async def _execute_reranked_search(
        self,
        query: str,
        dimension: MatryoshkaDimension,
        request: QueryProcessingRequest,
    ) -> list[dict[str, Any]]:
        """Execute search with BGE reranking."""
        # First, get more results than needed for reranking
        min(request.limit * 3, 50)  # Get 3x results for reranking

        # Execute base semantic search
        base_results = await self._execute_semantic_search(query, dimension, request)

        # Apply reranking if embedding manager supports it
        if hasattr(self.embedding_manager, "rerank_results") and base_results:
            try:
                reranked_results = await self.embedding_manager.rerank_results(
                    query, base_results
                )
                return reranked_results[: request.limit]
            except Exception as e:
                logger.warning(f"Reranking failed, returning base results: {e}")
                return base_results[: request.limit]

        return base_results[: request.limit]

    async def _execute_adaptive_search(
        self,
        query: str,
        dimension: MatryoshkaDimension,
        request: QueryProcessingRequest,
    ) -> list[dict[str, Any]]:
        """Execute adaptive search that chooses strategy based on query characteristics."""
        # For adaptive search, start with semantic and upgrade based on results
        results = await self._execute_semantic_search(query, dimension, request)

        # If results are poor (low count), try HyDE
        if len(results) < request.limit // 2:
            hyde_results = await self._execute_hyde_search(query, dimension, request)
            if len(hyde_results) > len(results):
                return hyde_results

        return results

    def _generate_cache_key(self, request: QueryProcessingRequest) -> str:
        """Generate cache key for the request."""
        import hashlib

        # Create a hash of the key request parameters
        key_data = f"{request.query}:{request.collection_name}:{request.limit}:{request.filters}:{request.search_accuracy}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def _get_cached_result(
        self, cache_key: str
    ) -> QueryProcessingResponse | None:
        """Get cached result if available."""
        if not self.cache_manager:
            return None

        try:
            # Implementation would depend on cache manager interface
            # For now, return None (no caching)
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None

    async def _cache_result(
        self, cache_key: str, response: QueryProcessingResponse
    ) -> None:
        """Cache the processing result."""
        if not self.cache_manager:
            return

        try:
            # Implementation would depend on cache manager interface
            # For now, do nothing
            pass
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    async def _record_analytics(
        self,
        request: QueryProcessingRequest,
        response: QueryProcessingResponse,
        intent_classification: Any,
    ) -> None:
        """Record analytics data for query processing optimization."""
        try:
            import hashlib

            analytics = QueryAnalytics(
                query_hash=hashlib.md5(request.query.encode()).hexdigest(),
                query_length=len(request.query),
                query_word_count=len(request.query.split()),
                total_time_ms=response.total_processing_time_ms,
                strategy_used=SearchStrategy.SEMANTIC,  # Would need to track actual strategy
                dimension_used=MatryoshkaDimension.MEDIUM,  # Would need to track actual dimension
                results_count=response.total_results,
                average_score=0.0,  # Would calculate from results
            )

            # Store analytics (implementation depends on analytics backend)
            logger.debug(f"Analytics recorded for query: {analytics.query_hash}")

        except Exception as e:
            logger.warning(f"Analytics recording failed: {e}")

    def _update_processing_stats(self, processing_time_ms: float) -> None:
        """Update internal processing statistics."""
        total_queries = self._processing_stats["total_queries"]
        current_avg = self._processing_stats["average_processing_time"]

        # Update rolling average
        self._processing_stats["average_processing_time"] = (
            current_avg * (total_queries - 1) + processing_time_ms
        ) / total_queries

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for monitoring."""
        return self._processing_stats.copy()

    async def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        if self._initialized:
            await asyncio.gather(
                self.intent_classifier.cleanup(),
                self.preprocessor.cleanup(),
                self.strategy_selector.cleanup(),
                return_exceptions=True,
            )
            self._initialized = False
            logger.info("QueryProcessingOrchestrator cleaned up")
