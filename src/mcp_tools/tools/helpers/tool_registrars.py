"""Tool registration functions for query processing MCP tools."""

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


from src.mcp_tools.models.requests import (
    AdvancedQueryProcessingRequest,
    QueryAnalysisRequest,
)
from src.mcp_tools.models.responses import (
    AdvancedQueryProcessingResponse,
    QueryAnalysisResponse,
)
from src.services.query_processing.models import QueryProcessingRequest

from .pipeline_factory import QueryProcessingPipelineFactory
from .response_converter import ResponseConverter
from .validation_helper import QueryValidationHelper


logger = logging.getLogger(__name__)


def register_advanced_query_processing_tool(
    mcp,
    factory: QueryProcessingPipelineFactory,
    converter: ResponseConverter,
    validator: QueryValidationHelper,
):
    """Register the advanced query processing tool."""

    @mcp.tool()
    async def advanced_query_processing(
        request: AdvancedQueryProcessingRequest, ctx: Context
    ) -> AdvancedQueryProcessingResponse:
        """Process queries using advanced intent classification and intelligent strategy selection.

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
            validated_collection, validated_query = validator.validate_query_request(
                request
            )

            await ctx.debug(
                f"Processing with intent classification: {request.enable_intent_classification}, "
                f"preprocessing: {request.enable_preprocessing}, "
                f"strategy selection: {request.enable_strategy_selection}"
            )

            # Get query processing pipeline
            pipeline = await factory.create_pipeline(ctx)

            # Validate force options
            force_strategy, force_dimension = await validator.validate_force_options(
                request, ctx
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
                filters=request.filters or {},
                max_processing_time_ms=request.max_processing_time_ms or 5000,
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
            mcp_response = converter.convert_to_mcp_response(
                response, request.include_analytics
            )

            await ctx.info(
                f"Advanced query processing {request_id} completed successfully: "
                f"{mcp_response.total_results} results, confidence: {mcp_response.confidence_score:.2f}"
            )

        except Exception as e:
            await ctx.error(f"Advanced query processing {request_id} failed: {e}")
            logger.exception(
                "Advanced query processing failed: "
            )  # TODO: Convert f-string to logging format

            # Return error response
            return AdvancedQueryProcessingResponse(
                success=False, results=[], total_results=0, error=str(e)
            )

        else:
            return mcp_response


def register_query_analysis_tool(
    mcp,
    factory: QueryProcessingPipelineFactory,
    converter: ResponseConverter,
    validator: QueryValidationHelper,
):
    """Register the query analysis tool."""

    @mcp.tool()
    async def analyze_query(
        request: QueryAnalysisRequest, ctx: Context
    ) -> QueryAnalysisResponse:
        """Analyze query characteristics without executing search.

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
            validated_query = validator.security_validator.validate_query_string(
                request.query
            )

            # Get query processing pipeline
            pipeline = await factory.create_pipeline(ctx)

            await ctx.debug(
                f"Analyzing query with preprocessing: {request.enable_preprocessing}, intent classification: {request.enable_intent_classification}"
            )

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
                preprocessing_result = converter.convert_preprocessing_result(
                    preprocessing_data
                )

            # Convert intent classification
            intent_result = None
            if analysis.get("intent_classification"):
                intent_data = analysis["intent_classification"]
                intent_result = converter.convert_intent_classification(intent_data)

            # Convert strategy selection
            strategy_result = None
            if analysis.get("strategy_selection"):
                strategy_data = analysis["strategy_selection"]
                strategy_result = converter.convert_strategy_selection(strategy_data)

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

        except Exception as e:
            await ctx.error(f"Query analysis {request_id} failed: {e}")
            logger.exception(
                "Query analysis failed: "
            )  # TODO: Convert f-string to logging format
            raise
        else:
            return response
