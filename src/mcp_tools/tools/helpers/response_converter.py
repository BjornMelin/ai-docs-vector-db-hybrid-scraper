import typing

"""Response converters for query processing MCP tools."""

from uuid import uuid4

from ...models.responses import AdvancedQueryProcessingResponse
from ...models.responses import QueryIntentResult
from ...models.responses import QueryPreprocessingResult
from ...models.responses import SearchResult
from ...models.responses import SearchStrategyResult


class ResponseConverter:
    """Converts internal response formats to MCP response formats."""

    @staticmethod
    def convert_intent_classification(intent_data) -> QueryIntentResult | None:
        """Convert intent classification data to MCP format."""
        if not intent_data:
            return None

        return QueryIntentResult(
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

    @staticmethod
    def convert_preprocessing_result(
        preprocessing_data,
    ) -> QueryPreprocessingResult | None:
        """Convert preprocessing result to MCP format."""
        if not preprocessing_data:
            return None

        return QueryPreprocessingResult(
            original_query=preprocessing_data.original_query,
            processed_query=preprocessing_data.processed_query,
            corrections_applied=preprocessing_data.corrections_applied,
            expansions_added=preprocessing_data.expansions_added,
            normalization_applied=preprocessing_data.normalization_applied,
            context_extracted=preprocessing_data.context_extracted,
            preprocessing_time_ms=preprocessing_data.preprocessing_time_ms,
        )

    @staticmethod
    def convert_strategy_selection(strategy_data) -> SearchStrategyResult | None:
        """Convert strategy selection data to MCP format."""
        if not strategy_data:
            return None

        return SearchStrategyResult(
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

    @staticmethod
    def convert_search_results(
        results, include_analytics: bool = False
    ) -> list[SearchResult]:
        """Convert search results to MCP format."""
        search_results = []
        for result in results:
            search_result = SearchResult(
                id=str(result.get("id", uuid4())),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                url=result.get("url"),
                title=result.get("title"),
                metadata=result.get("metadata") if include_analytics else None,
            )
            search_results.append(search_result)
        return search_results

    def convert_to_mcp_response(
        self, response, include_analytics: bool = False
    ) -> AdvancedQueryProcessingResponse:
        """Convert internal response to MCP response format."""
        # Convert components using helper methods
        intent_result = self.convert_intent_classification(
            response.intent_classification
        )
        preprocessing_result = self.convert_preprocessing_result(
            response.preprocessing_result
        )
        strategy_result = self.convert_strategy_selection(response.strategy_selection)
        search_results = self.convert_search_results(
            response.results, include_analytics
        )

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
