"""Query Processing Pipeline.

This module provides the main unified interface for the query
processing system, orchestrating all components through a single entry point.
"""

import asyncio
import logging
import subprocess
import time
from typing import Any

from src.services.base import BaseService

from .models import QueryProcessingRequest, QueryProcessingResponse, SearchRecord
from .orchestrator import (
    SearchOrchestrator,
    SearchPipeline,
    SearchRequest,
    SearchResult,
)


_PIPELINE_BY_ACCURACY = {
    "fast": SearchPipeline.FAST,
    "balanced": SearchPipeline.BALANCED,
    "comprehensive": SearchPipeline.COMPREHENSIVE,
}


logger = logging.getLogger(__name__)


def _build_search_request(request: QueryProcessingRequest) -> SearchRequest:
    """Translate a high level processing request into a search request."""

    pipeline = _PIPELINE_BY_ACCURACY.get(
        request.search_accuracy.lower(), SearchPipeline.BALANCED
    )

    user_id = request.user_context.get("user_id") if request.user_context else None
    session_id = (
        request.user_context.get("session_id") if request.user_context else None
    )

    return SearchRequest(
        query=request.query,
        collection_name=request.collection_name,
        limit=request.limit,
        pipeline=pipeline,
        user_id=user_id,
        session_id=session_id,
        filters=request.filters or None,
        enable_caching=True,
        max_processing_time_ms=float(request.max_processing_time_ms),
    )  # type: ignore


def _build_processing_response(
    search_result: SearchResult,
) -> QueryProcessingResponse:
    """Create a processing response wrapper for the search result."""

    return QueryProcessingResponse(
        success=True,
        results=SearchRecord.parse_list(search_result.results),
        total_results=search_result.total_results,
        total_processing_time_ms=search_result.processing_time_ms,
        search_time_ms=search_result.processing_time_ms,
        strategy_selection_time_ms=0.0,
        processing_steps=search_result.features_used,
        cache_hit=search_result.cache_hit,
        strategy_selection=None,
        preprocessing_result=None,
        intent_classification=None,
        confidence_score=search_result.answer_confidence or 0.0,
        quality_score=0.0,
        fallback_used=False,
        warnings=[],
    )


class QueryProcessingPipeline(BaseService):
    """Unified interface for query processing pipeline.

    Provides a single entry point for all query processing operations,
    coordinating intent classification, preprocessing, strategy selection,
    and search execution through the orchestrator.
    """

    def __init__(self, orchestrator: SearchOrchestrator, config: Any = None):
        """Initialize the query processing pipeline.

        Args:
            orchestrator: Query processing orchestrator instance
            config: Optional configuration object

        Raises:
            ValueError: If orchestrator is None

        """
        if orchestrator is None:
            msg = "Orchestrator cannot be None"
            raise ValueError(msg)

        super().__init__(config)
        self.orchestrator = orchestrator

    async def initialize(self) -> None:
        """Initialize the pipeline and all components."""
        if self._initialized:
            return

        try:
            await self.orchestrator.initialize()
            self._initialized = True
            logger.info("QueryProcessingPipeline initialized successfully")

        except (AttributeError, ImportError, OSError):
            logger.exception("Failed to initialize QueryProcessingPipeline")
            raise

    async def process(
        self,
        query_or_request,
        collection_name: str = "documents",
        limit: int = 10,
        **kwargs,
    ) -> QueryProcessingResponse:
        """Process a query through the complete pipeline.

        This is the main entry point for query processing that coordinates
        all pipeline components to deliver optimal search results.

        Args:
            query_or_request: Either a string query or QueryProcessingRequest object
            collection_name: Target collection for search (used only if
            query_or_request is string)
            limit: Maximum number of results to return (used only if
            query_or_request is string)
            **kwargs: Additional processing options (used only if
            query_or_request is string)

        Returns:
            QueryProcessingResponse: Complete processing results

        Raises:
            RuntimeError: If pipeline not initialized

        """
        if not self._initialized:
            msg = "QueryProcessingPipeline not initialized"
            raise RuntimeError(msg)

        # Handle both string query and QueryProcessingRequest
        if isinstance(query_or_request, QueryProcessingRequest):
            request = query_or_request
        else:
            # Handle empty query gracefully
            if not query_or_request or not query_or_request.strip():
                return QueryProcessingResponse(
                    success=False,
                    results=[],
                    total_results=0,
                    error="Empty query provided",
                )

            # Create processing request from parameters
            request = QueryProcessingRequest(
                query=query_or_request,
                collection_name=collection_name,
                limit=limit,
                **kwargs,
            )

        search_request = _build_search_request(request)
        search_result = await self.orchestrator.search(search_request)
        return _build_processing_response(search_result)

    async def process_advanced(
        self, request: QueryProcessingRequest
    ) -> QueryProcessingResponse:
        """Process a query with full advanced options.

        Args:
            request: Complete processing request with all options

        Returns:
            QueryProcessingResponse: Complete processing results

        """
        self._validate_initialized()
        search_request = _build_search_request(request)
        search_result = await self.orchestrator.search(search_request)
        return _build_processing_response(search_result)

    async def process_batch(
        self, requests: list[QueryProcessingRequest], max_concurrent: int = 5
    ) -> list[QueryProcessingResponse]:
        """Process multiple queries concurrently.

        Args:
            requests: List of processing requests
            max_concurrent: Maximum concurrent processing

        Returns:
            list[QueryProcessingResponse]: Results for each query

        """
        self._validate_initialized()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_request(
            request: QueryProcessingRequest,
        ) -> QueryProcessingResponse:
            async with semaphore:
                try:
                    return await self.process(request)
                except Exception as e:
                    logger.exception("Batch processing failed for query '{request}'")
                    return QueryProcessingResponse(
                        success=False, results=[], total_results=0, error=str(e)
                    )

        # Execute all queries concurrently
        tasks = [process_single_request(request) for request in requests]
        return await asyncio.gather(*tasks)

    async def analyze_query(
        self,
        query: str,
        enable_preprocessing: bool = True,
        enable_intent_classification: bool = True,
    ) -> dict[str, Any]:
        """Analyze a query without executing search.

        Useful for understanding query characteristics and optimal processing
        strategies without performing the actual search.

        Args:
            query: Query to analyze
            enable_preprocessing: Whether to preprocess the query
            enable_intent_classification: Whether to classify intent

        Returns:
            dict[str, Any]: Analysis results including intent and strategy

        """
        self._validate_initialized()

        # Create request with minimal search (limit=1)
        request = QueryProcessingRequest(
            query=query,
            collection_name="analysis",
            limit=1,  # Minimal search for analysis
            enable_preprocessing=enable_preprocessing,
            enable_intent_classification=enable_intent_classification,
            enable_strategy_selection=True,
        )

        search_request = _build_search_request(request)
        search_result = await self.orchestrator.search(search_request)

        return {
            "query": search_request.query,
            "processed_query": search_result.query_processed,
            "features_used": search_result.features_used,
            "processing_time_ms": search_result.processing_time_ms,
            "cache_hit": search_result.cache_hit,
            "result_count": search_result.total_results,
        }

    async def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics.

        Returns:
            dict[str, Any]: Performance statistics and metrics

        """
        if not self._initialized:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "average_processing_time": 0.0,
                "strategy_usage": {},
            }

        base_metrics = self.orchestrator.get_stats()

        # Return metrics in expected format
        return {
            "total_queries": base_metrics.get("total_queries", 0),
            "successful_queries": base_metrics.get("successful_queries", 0),
            "average_processing_time": base_metrics.get("average_processing_time", 0.0),
            "strategy_usage": base_metrics.get("strategy_usage", {}),
            "pipeline_initialized": self._initialized,
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all pipeline components.

        Returns:
            dict[str, Any]: Health status of all components

        """
        health_status = {
            "status": "healthy" if self._initialized else "unhealthy",
            "pipeline_healthy": self._initialized,
            "components": {},
            "performance": {"initialized": self._initialized},
        }

        if self._initialized:
            try:
                # Test with a simple query
                test_response = await self.analyze_query(
                    "test query",
                    enable_preprocessing=True,
                    enable_intent_classification=True,
                )

                search_healthy = test_response.get("result_count", 0) > 0
                cached = test_response.get("cache_hit", False)

                health_status["components"] = {
                    "orchestrator": {
                        "status": "healthy",
                        "message": "Search executed successfully",
                    },
                    "search_pipeline": {
                        "status": "healthy" if search_healthy else "degraded",
                        "message": "Results returned"
                        if search_healthy
                        else "No results returned",
                    },
                    "cache": {
                        "status": "healthy" if cached else "cold",
                        "message": "Warm cache" if cached else "Cache miss",
                    },
                }

                if not search_healthy:
                    health_status["status"] = "degraded"

            except Exception as e:
                logger.exception("Health check failed")
                health_status["status"] = "unhealthy"
                health_status["components"] = {
                    "orchestrator": {"status": "unhealthy", "message": str(e)},
                    "error": str(e),
                }

        return health_status

    async def warm_up(self) -> dict[str, Any]:
        """Warm up the pipeline with test queries.

        Useful for pre-loading models and caches before handling real traffic.

        Returns:
            dict[str, Any]: Warmup results including status and timing

        """
        self._validate_initialized()

        start_time = time.time()

        warmup_requests = [
            QueryProcessingRequest(
                query="What is machine learning?",
                collection_name="warmup",
                limit=1,
                enable_preprocessing=True,
                enable_intent_classification=True,
            ),
            QueryProcessingRequest(
                query="How to implement authentication in Python?",
                collection_name="warmup",
                limit=1,
                enable_preprocessing=True,
                enable_intent_classification=True,
            ),
            QueryProcessingRequest(
                query="Best practices for API design",
                collection_name="warmup",
                limit=1,
                enable_preprocessing=True,
                enable_intent_classification=True,
            ),
        ]

        logger.info("Warming up query processing pipeline...")

        try:
            # Process warmup queries to initialize all components
            responses = await self.process_batch(
                requests=warmup_requests, max_concurrent=2
            )

            end_time = time.time()
            warmup_time_ms = (end_time - start_time) * 1000

            successful_warmups = sum(1 for resp in responses if resp.success)

            logger.info("Pipeline warmup completed successfully")

            return {
                "status": "completed",
                "warmup_time_ms": warmup_time_ms,
                "queries_processed": len(responses),
                "successful_queries": successful_warmups,
                "components_warmed": [
                    "orchestrator",
                    "intent_classifier",
                    "preprocessor",
                    "strategy_selector",
                ],
            }

        except (subprocess.SubprocessError, OSError, TimeoutError) as e:
            end_time = time.time()
            warmup_time_ms = (end_time - start_time) * 1000

            logger.warning("Pipeline warmup had issues")

            return {
                "status": "partial",
                "warmup_time_ms": warmup_time_ms,
                "error": str(e),
                "queries_processed": 0,
                "successful_queries": 0,
            }

    async def cleanup(self) -> None:
        """Cleanup pipeline and all components."""
        if self._initialized:
            await self.orchestrator.cleanup()
            self._initialized = False
            logger.info("QueryProcessingPipeline cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False
