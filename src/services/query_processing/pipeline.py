
"""Advanced Query Processing Pipeline.

This module provides the main unified interface for the advanced query
processing system, orchestrating all components through a single entry point.
"""

import logging
from typing import Any

from ..base import BaseService
from .models import QueryProcessingRequest
from .models import QueryProcessingResponse
from .orchestrator import SearchOrchestrator as AdvancedSearchOrchestrator

logger = logging.getLogger(__name__)


class QueryProcessingPipeline(BaseService):
    """Unified interface for advanced query processing pipeline.

    Provides a single entry point for all query processing operations,
    coordinating intent classification, preprocessing, strategy selection,
    and search execution through the orchestrator.
    """

    def __init__(self, orchestrator: AdvancedSearchOrchestrator, config: Any = None):
        """Initialize the query processing pipeline.

        Args:
            orchestrator: Query processing orchestrator instance
            config: Optional configuration object

        Raises:
            ValueError: If orchestrator is None
        """
        if orchestrator is None:
            raise ValueError("Orchestrator cannot be None")

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

        except Exception as e:
            logger.exception(f"Failed to initialize QueryProcessingPipeline: {e}")
            raise

    async def process(
        self,
        query_or_request,
        collection_name: str = "documents",
        limit: int = 10,
        **kwargs,
    ) -> QueryProcessingResponse:
        """Process a query through the complete advanced pipeline.

        This is the main entry point for query processing that coordinates
        all pipeline components to deliver optimal search results.

        Args:
            query_or_request: Either a string query or QueryProcessingRequest object
            collection_name: Target collection for search (used only if query_or_request is string)
            limit: Maximum number of results to return (used only if query_or_request is string)
            **kwargs: Additional processing options (used only if query_or_request is string)

        Returns:
            QueryProcessingResponse: Complete processing results

        Raises:
            RuntimeError: If pipeline not initialized
        """
        if not self._initialized:
            raise RuntimeError("QueryProcessingPipeline not initialized")

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

        # Process through orchestrator
        return await self.orchestrator.process_query(request)

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
        return await self.orchestrator.process_query(request)

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

        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_request(
            request: QueryProcessingRequest,
        ) -> QueryProcessingResponse:
            async with semaphore:
                try:
                    return await self.process(request)
                except Exception as e:
                    logger.exception(
                        f"Batch processing failed for query '{request}': {e}"
                    )
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

        # Process through orchestrator
        response = await self.orchestrator.process_query(request)

        # Return analysis components
        analysis = {
            "query": query,
            "preprocessing": response.preprocessing_result,
            "intent_classification": response.intent_classification,
            "complexity": response.intent_classification.complexity_level
            if response.intent_classification
            else None,
            "strategy": response.strategy_selection,
            "processing_time_ms": response.total_processing_time_ms,
        }

        return analysis

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

        base_metrics = self.orchestrator.get_performance_stats()

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

                # Check component health based on successful analysis
                intent_healthy = test_response.get("intent_classification") is not None
                preprocessing_healthy = test_response.get("preprocessing") is not None
                strategy_healthy = test_response.get("strategy") is not None

                health_status["components"] = {
                    "orchestrator": {
                        "status": "healthy",
                        "message": "Working correctly",
                    },
                    "intent_classifier": {
                        "status": "healthy" if intent_healthy else "degraded",
                        "message": "Classification working"
                        if intent_healthy
                        else "Classification issues",
                    },
                    "preprocessor": {
                        "status": "healthy" if preprocessing_healthy else "degraded",
                        "message": "Preprocessing working"
                        if preprocessing_healthy
                        else "Preprocessing issues",
                    },
                    "strategy_selector": {
                        "status": "healthy" if strategy_healthy else "degraded",
                        "message": "Selection working"
                        if strategy_healthy
                        else "Selection issues",
                    },
                }

                # Overall status based on components
                if not all(
                    comp["status"] == "healthy"
                    for comp in health_status["components"].values()
                    if isinstance(comp, dict)
                ):
                    health_status["status"] = "degraded"

            except Exception as e:
                logger.exception(f"Health check failed: {e}")
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

        import time

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

        except Exception as e:
            end_time = time.time()
            warmup_time_ms = (end_time - start_time) * 1000

            logger.warning(f"Pipeline warmup had issues: {e}")

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
