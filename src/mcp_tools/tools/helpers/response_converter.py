"""Response converters for query processing MCP tools."""

from enum import Enum

from src.mcp_tools.models.responses import (
    QueryIntentResult,
    QueryPreprocessingResult,
    QueryProcessingResponse,
    SearchResult,
    SearchStrategyResult,
)
from src.services.query_processing.models import SearchRecord


def _enum_or_value(value: object) -> object:
    """Return enum value if provided, otherwise passthrough."""

    if isinstance(value, Enum):
        return value.value
    return value


class ResponseConverter:
    """Converts internal response formats to MCP response formats."""

    @staticmethod
    def convert_intent_classification(intent_data) -> QueryIntentResult | None:
        """Convert intent classification data to MCP format."""
        if not intent_data:
            return None

        payload = None
        model_dump = getattr(intent_data, "model_dump", None)
        if callable(model_dump):
            try:
                candidate = model_dump(mode="json")
                if isinstance(candidate, dict):
                    payload = candidate
            except Exception:  # pragma: no cover - fall back to manual mapping
                payload = None

        if payload is None:
            payload = {
                "primary_intent": _enum_or_value(
                    getattr(intent_data, "primary_intent", None)
                ),
                "secondary_intents": [
                    _enum_or_value(value)
                    for value in getattr(intent_data, "secondary_intents", [])
                ],
                "confidence_scores": {
                    _enum_or_value(key): score
                    for key, score in getattr(
                        intent_data, "confidence_scores", {}
                    ).items()
                },
                "complexity_level": _enum_or_value(
                    getattr(intent_data, "complexity_level", None)
                ),
                "domain_category": getattr(intent_data, "domain_category", None),
                "classification_reasoning": getattr(
                    intent_data, "classification_reasoning", ""
                ),
                "requires_context": getattr(intent_data, "requires_context", False),
                "suggested_followups": list(
                    getattr(intent_data, "suggested_followups", [])
                ),
            }

        return QueryIntentResult.model_validate(payload)

    @staticmethod
    def convert_preprocessing_result(
        preprocessing_data,
    ) -> QueryPreprocessingResult | None:
        """Convert preprocessing result to MCP format."""
        if not preprocessing_data:
            return None

        payload = None
        model_dump = getattr(preprocessing_data, "model_dump", None)
        if callable(model_dump):
            try:
                candidate = model_dump(mode="json")
                if isinstance(candidate, dict):
                    payload = candidate
            except Exception:  # pragma: no cover - fall back to manual mapping
                payload = None

        if payload is None:
            payload = {
                "original_query": getattr(preprocessing_data, "original_query", ""),
                "processed_query": getattr(
                    preprocessing_data, "processed_query", ""
                ),
                "corrections_applied": list(
                    getattr(preprocessing_data, "corrections_applied", [])
                ),
                "expansions_added": list(
                    getattr(preprocessing_data, "expansions_added", [])
                ),
                "normalization_applied": getattr(
                    preprocessing_data, "normalization_applied", False
                ),
                "context_extracted": dict(
                    getattr(preprocessing_data, "context_extracted", {})
                ),
                "preprocessing_time_ms": getattr(
                    preprocessing_data, "preprocessing_time_ms", 0.0
                ),
            }

        return QueryPreprocessingResult.model_validate(payload)

    @staticmethod
    def convert_strategy_selection(strategy_data) -> SearchStrategyResult | None:
        """Convert strategy selection data to MCP format."""
        if not strategy_data:
            return None

        payload = None
        model_dump = getattr(strategy_data, "model_dump", None)
        if callable(model_dump):
            try:
                candidate = model_dump(mode="json")
                if isinstance(candidate, dict):
                    payload = candidate
            except Exception:  # pragma: no cover - fall back to manual mapping
                payload = None

        if payload is None:
            payload = {
                "primary_strategy": _enum_or_value(
                    getattr(strategy_data, "primary_strategy", None)
                ),
                "fallback_strategies": [
                    _enum_or_value(value)
                    for value in getattr(strategy_data, "fallback_strategies", [])
                ],
                "matryoshka_dimension": _enum_or_value(
                    getattr(strategy_data, "matryoshka_dimension", None)
                ),
                "confidence": getattr(strategy_data, "confidence", 0.0),
                "reasoning": getattr(strategy_data, "reasoning", ""),
                "estimated_quality": getattr(
                    strategy_data, "estimated_quality", 0.0
                ),
                "estimated_latency_ms": getattr(
                    strategy_data, "estimated_latency_ms", 0.0
                ),
            }

        return SearchStrategyResult.model_validate(payload)

    @staticmethod
    def convert_search_results(
        results, include_analytics: bool = False
    ) -> list[SearchResult]:
        """Convert search results to MCP format."""

        search_results: list[SearchResult] = []
        for payload in results:
            record = SearchRecord.from_payload(payload)
            metadata = record.metadata if include_analytics else None

            search_results.append(
                SearchResult(
                    id=record.id,
                    content=record.content,
                    score=record.score,
                    url=record.url,
                    title=record.title,
                    metadata=metadata,
                    content_type=record.content_type,
                    content_confidence=record.content_confidence,
                    quality_overall=record.quality_overall,
                    quality_completeness=record.quality_completeness,
                    quality_relevance=record.quality_relevance,
                    quality_confidence=record.quality_confidence,
                    content_intelligence_analyzed=record.content_intelligence_analyzed,
                )
            )

        return search_results

    def convert_to_mcp_response(
        self, response, include_analytics: bool = False
    ) -> QueryProcessingResponse:
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

        warnings_data = getattr(response, "warnings", None)
        if isinstance(warnings_data, list):
            warnings = warnings_data
        elif warnings_data is None:
            warnings = []
        elif isinstance(warnings_data, (tuple, set)):
            warnings = list(warnings_data)
        else:
            warnings = [str(warnings_data)]

        return QueryProcessingResponse(
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
            warnings=warnings,
        )
