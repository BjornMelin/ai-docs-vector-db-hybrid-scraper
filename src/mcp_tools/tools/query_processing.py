"""Advanced Query Processing tools for MCP server."""

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from ...infrastructure.client_manager import ClientManager
from ...security import SecurityValidator
from ...services.query_processing.models import (
    MatryoshkaDimension,
    QueryProcessingRequest,
    SearchStrategy,
)
from ...services.query_processing.pipeline import QueryProcessingPipeline
from ..models.requests import AdvancedQueryProcessingRequest, QueryAnalysisRequest
from ..models.responses import (
    AdvancedQueryProcessingResponse,
    QueryAnalysisResponse,
    QueryIntentResult,
    QueryPreprocessingResult,
    SearchResult,
    SearchStrategyResult,
)

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register advanced query processing tools with the MCP server."""

    async def _get_query_processing_pipeline(ctx: "Context | None" = None) -> QueryProcessingPipeline:
        """Get initialized query processing pipeline."""
        try:
            # Get required services
            embedding_manager = await client_manager.get_embedding_manager()
            qdrant_service = await client_manager.get_qdrant_service()
            hyde_engine = await client_manager.get_hyde_engine()
            
            # Get cache manager if available
            cache_manager = None
            try:
                cache_manager = await client_manager.get_cache_manager()
            except Exception:
                if ctx:
                    await ctx.debug("Cache manager not available, proceeding without caching")
            
            # Create orchestrator
            from ...services.query_processing.orchestrator import QueryProcessingOrchestrator
            
            orchestrator = QueryProcessingOrchestrator(
                embedding_manager=embedding_manager,
                qdrant_service=qdrant_service,
                hyde_engine=hyde_engine,
                cache_manager=cache_manager
            )
            
            # Create pipeline
            pipeline = QueryProcessingPipeline(orchestrator=orchestrator)
            await pipeline.initialize()
            
            return pipeline
            
        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to initialize query processing pipeline: {e}")
            logger.error(f"Pipeline initialization failed: {e}")
            raise

    def _convert_to_mcp_response(
        response, include_analytics: bool = False
    ) -> AdvancedQueryProcessingResponse:
        """Convert internal response to MCP response format."""
        # Convert intent classification
        intent_result = None
        if response.intent_classification:
            intent_result = QueryIntentResult(
                primary_intent=response.intent_classification.primary_intent.value,
                secondary_intents=[
                    intent.value for intent in response.intent_classification.secondary_intents
                ],
                confidence_scores={
                    intent.value: score
                    for intent, score in response.intent_classification.confidence_scores.items()
                },
                complexity_level=response.intent_classification.complexity_level.value,
                domain_category=response.intent_classification.domain_category,
                classification_reasoning=response.intent_classification.classification_reasoning,
                requires_context=response.intent_classification.requires_context,
                suggested_followups=response.intent_classification.suggested_followups,
            )

        # Convert preprocessing result
        preprocessing_result = None
        if response.preprocessing_result:
            preprocessing_result = QueryPreprocessingResult(
                original_query=response.preprocessing_result.original_query,
                processed_query=response.preprocessing_result.processed_query,
                corrections_applied=response.preprocessing_result.corrections_applied,
                expansions_added=response.preprocessing_result.expansions_added,
                normalization_applied=response.preprocessing_result.normalization_applied,
                context_extracted=response.preprocessing_result.context_extracted,
                preprocessing_time_ms=response.preprocessing_result.preprocessing_time_ms,
            )

        # Convert strategy selection
        strategy_result = None
        if response.strategy_selection:
            strategy_result = SearchStrategyResult(
                primary_strategy=response.strategy_selection.primary_strategy.value,
                fallback_strategies=[
                    strategy.value for strategy in response.strategy_selection.fallback_strategies
                ],
                matryoshka_dimension=response.strategy_selection.matryoshka_dimension.value,
                confidence=response.strategy_selection.confidence,
                reasoning=response.strategy_selection.reasoning,
                estimated_quality=response.strategy_selection.estimated_quality,
                estimated_latency_ms=response.strategy_selection.estimated_latency_ms,
            )

        # Convert search results
        search_results = []
        for result in response.results:
            search_result = SearchResult(
                id=str(result.get("id", uuid4())),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                url=result.get("url"),
                title=result.get("title"),
                metadata=result.get("metadata") if include_analytics else None,
            )
            search_results.append(search_result)

        return AdvancedQueryProcessingResponse(
            success=response.success,
            results=search_results,
            total_results=response.total_results,
            intent_classification=intent_result,
            preprocessing_result=preprocessing_result,
            strategy_selection=strategy_result,
            total_processing_time_ms=response.total_processing_time_ms,
            search_time_ms=response.search_time_ms,
            strategy_selection_time_ms=response.strategy_selection_time_ms,
            confidence_score=response.confidence_score,
            quality_score=response.quality_score,
            processing_steps=response.processing_steps,
            fallback_used=response.fallback_used,
            cache_hit=response.cache_hit,
            error=response.error,
        )

    @mcp.tool()
    async def advanced_query_processing(
        request: AdvancedQueryProcessingRequest, ctx: Context
    ) -> AdvancedQueryProcessingResponse:
        """
        Process queries using advanced intent classification and intelligent strategy selection.

        Implements complete advanced query processing pipeline with:
        - 14 query intent categories (conceptual, procedural, factual, troubleshooting + 10 advanced)
        - Intelligent preprocessing with spell correction and expansion
        - Dynamic strategy selection based on intent and complexity
        - Matryoshka embeddings with dimension optimization
        - Comprehensive fallback handling and performance tracking
        """
        # Generate request ID for tracking
        request_id = str(uuid4())
        await ctx.info(
            f"Starting advanced query processing {request_id} for query: {request.query[:50]}..."
        )

        try:
            # Validate inputs
            security_validator = SecurityValidator.from_unified_config()
            validated_collection = security_validator.validate_collection_name(
                request.collection
            )
            validated_query = security_validator.validate_query_string(request.query)

            await ctx.debug(
                f"Processing with intent classification: {request.enable_intent_classification}, "
                f"preprocessing: {request.enable_preprocessing}, "
                f"strategy selection: {request.enable_strategy_selection}"
            )

            # Get query processing pipeline
            pipeline = await _get_query_processing_pipeline(ctx)

            # Convert force options to internal types
            force_strategy = None
            if request.force_strategy:
                try:
                    force_strategy = SearchStrategy(request.force_strategy.lower())
                except ValueError:
                    await ctx.warning(
                        f"Invalid force_strategy '{request.force_strategy}', ignoring"
                    )

            force_dimension = None
            if request.force_dimension:
                try:
                    # Map dimension values to MatryoshkaDimension
                    dimension_map = {512: MatryoshkaDimension.SMALL, 768: MatryoshkaDimension.MEDIUM, 1536: MatryoshkaDimension.LARGE}
                    force_dimension = dimension_map.get(request.force_dimension)
                    if not force_dimension:
                        await ctx.warning(
                            f"Invalid force_dimension '{request.force_dimension}', ignoring"
                        )
                except Exception:
                    await ctx.warning(
                        f"Invalid force_dimension '{request.force_dimension}', ignoring"
                    )

            # Create internal processing request
            processing_request = QueryProcessingRequest(
                query=validated_query,
                collection_name=validated_collection,
                limit=request.limit,
                enable_preprocessing=request.enable_preprocessing,
                enable_intent_classification=request.enable_intent_classification,
                enable_strategy_selection=request.enable_strategy_selection,
                force_strategy=force_strategy,
                force_dimension=force_dimension,
                user_context=request.user_context,
                filters=request.filters,
                max_processing_time_ms=request.max_processing_time_ms,
                search_accuracy=request.search_accuracy,
            )

            await ctx.debug(f"Created internal processing request for {request_id}")

            # Process through pipeline
            response = await pipeline.process_advanced(processing_request)

            await ctx.debug(
                f"Pipeline processing completed for {request_id}: "
                f"{response.total_results} results in {response.total_processing_time_ms:.2f}ms"
            )

            # Convert to MCP response format
            mcp_response = _convert_to_mcp_response(response, request.include_analytics)

            await ctx.info(
                f"Advanced query processing {request_id} completed successfully: "
                f"{mcp_response.total_results} results, confidence: {mcp_response.confidence_score:.2f}"
            )

            return mcp_response

        except Exception as e:
            await ctx.error(f"Advanced query processing {request_id} failed: {e}")
            logger.error(f"Advanced query processing failed: {e}", exc_info=True)
            
            # Return error response
            return AdvancedQueryProcessingResponse(
                success=False,
                results=[],
                total_results=0,
                error=str(e)
            )

    @mcp.tool()
    async def analyze_query(
        request: QueryAnalysisRequest, ctx: Context
    ) -> QueryAnalysisResponse:
        """
        Analyze query characteristics without executing search.

        Provides detailed analysis of query intent, complexity, and optimal processing
        strategies without performing the actual search. Useful for understanding
        query characteristics and strategy selection logic.
        """
        # Generate request ID for tracking
        request_id = str(uuid4())
        await ctx.info(
            f"Starting query analysis {request_id} for query: {request.query[:50]}..."
        )

        try:
            # Validate inputs
            security_validator = SecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(request.query)

            # Get query processing pipeline
            pipeline = await _get_query_processing_pipeline(ctx)

            await ctx.debug(f"Analyzing query with preprocessing: {request.enable_preprocessing}, intent classification: {request.enable_intent_classification}")

            # Analyze query without search
            analysis = await pipeline.analyze_query(
                query=validated_query,
                enable_preprocessing=request.enable_preprocessing,
                enable_intent_classification=request.enable_intent_classification,
            )

            # Convert preprocessing result
            preprocessing_result = None
            if analysis.get("preprocessing"):
                preprocessing_data = analysis["preprocessing"]
                preprocessing_result = QueryPreprocessingResult(
                    original_query=preprocessing_data.original_query,
                    processed_query=preprocessing_data.processed_query,
                    corrections_applied=preprocessing_data.corrections_applied,
                    expansions_added=preprocessing_data.expansions_added,
                    normalization_applied=preprocessing_data.normalization_applied,
                    context_extracted=preprocessing_data.context_extracted,
                    preprocessing_time_ms=preprocessing_data.preprocessing_time_ms,
                )

            # Convert intent classification
            intent_result = None
            if analysis.get("intent_classification"):
                intent_data = analysis["intent_classification"]
                intent_result = QueryIntentResult(
                    primary_intent=intent_data.primary_intent.value,
                    secondary_intents=[
                        intent.value for intent in intent_data.secondary_intents
                    ],
                    confidence_scores={
                        intent.value: score
                        for intent, score in intent_data.confidence_scores.items()
                    },
                    complexity_level=intent_data.complexity_level.value,
                    domain_category=intent_data.domain_category,
                    classification_reasoning=intent_data.classification_reasoning,
                    requires_context=intent_data.requires_context,
                    suggested_followups=intent_data.suggested_followups,
                )

            # Convert strategy selection
            strategy_result = None
            if analysis.get("strategy_selection"):
                strategy_data = analysis["strategy_selection"]
                strategy_result = SearchStrategyResult(
                    primary_strategy=strategy_data.primary_strategy.value,
                    fallback_strategies=[
                        strategy.value for strategy in strategy_data.fallback_strategies
                    ],
                    matryoshka_dimension=strategy_data.matryoshka_dimension.value,
                    confidence=strategy_data.confidence,
                    reasoning=strategy_data.reasoning,
                    estimated_quality=strategy_data.estimated_quality,
                    estimated_latency_ms=strategy_data.estimated_latency_ms,
                )

            response = QueryAnalysisResponse(
                query=validated_query,
                preprocessing_result=preprocessing_result,
                intent_classification=intent_result,
                strategy_selection=strategy_result,
                processing_time_ms=analysis.get("processing_time_ms", 0.0),
            )

            await ctx.info(
                f"Query analysis {request_id} completed: "
                f"intent={intent_result.primary_intent if intent_result else 'N/A'}, "
                f"complexity={intent_result.complexity_level if intent_result else 'N/A'}"
            )

            return response

        except Exception as e:
            await ctx.error(f"Query analysis {request_id} failed: {e}")
            logger.error(f"Query analysis failed: {e}", exc_info=True)
            raise

    @mcp.tool()
    async def get_processing_pipeline_health(ctx: Context) -> dict:
        """
        Get health status of the advanced query processing pipeline.

        Returns detailed health information for all pipeline components including
        orchestrator, intent classifier, preprocessor, and strategy selector.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting pipeline health check {request_id}")

        try:
            # Get query processing pipeline
            pipeline = await _get_query_processing_pipeline(ctx)

            # Perform health check
            health_status = await pipeline.health_check()

            await ctx.info(
                f"Pipeline health check {request_id} completed: "
                f"status={'healthy' if health_status.get('pipeline_healthy') else 'unhealthy'}"
            )

            return health_status

        except Exception as e:
            await ctx.error(f"Pipeline health check {request_id} failed: {e}")
            logger.error(f"Pipeline health check failed: {e}")
            return {
                "pipeline_healthy": False,
                "error": str(e),
                "components": {"health_check": "failed"},
            }

    @mcp.tool()
    async def get_processing_pipeline_metrics(ctx: Context) -> dict:
        """
        Get performance metrics from the advanced query processing pipeline.

        Returns comprehensive performance statistics including processing times,
        strategy usage, success rates, and fallback utilization.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting pipeline metrics collection {request_id}")

        try:
            # Get query processing pipeline
            pipeline = await _get_query_processing_pipeline(ctx)

            # Get performance metrics
            metrics = pipeline.get_performance_metrics()

            await ctx.info(f"Pipeline metrics collection {request_id} completed")

            return metrics

        except Exception as e:
            await ctx.error(f"Pipeline metrics collection {request_id} failed: {e}")
            logger.error(f"Pipeline metrics collection failed: {e}")
            return {"error": str(e), "metrics_available": False}

    @mcp.tool()
    async def warm_up_processing_pipeline(ctx: Context) -> dict:
        """
        Warm up the advanced query processing pipeline.

        Pre-loads models and caches by processing test queries to ensure
        optimal performance for subsequent real queries.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting pipeline warm-up {request_id}")

        try:
            # Get query processing pipeline
            pipeline = await _get_query_processing_pipeline(ctx)

            # Warm up pipeline
            await pipeline.warm_up()

            await ctx.info(f"Pipeline warm-up {request_id} completed successfully")

            return {"status": "success", "message": "Pipeline warmed up successfully"}

        except Exception as e:
            await ctx.warning(f"Pipeline warm-up {request_id} had issues: {e}")
            logger.warning(f"Pipeline warm-up failed: {e}")
            return {
                "status": "partial_success", 
                "message": f"Pipeline warm-up completed with issues: {e}"
            }