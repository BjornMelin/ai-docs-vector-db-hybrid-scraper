"""Preference-based ranking.

This module provides functionality for ranking search results based on
user preferences, applying category-based boosts and score normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class RankedResult(BaseModel):
    """Represents a ranked search result with scoring information.

    Attributes:
        result_id: Unique identifier for the result.
        title: Title of the result.
        content: Content/body of the result.
        original_score: Original relevance score before ranking.
        final_score: Final score after preference adjustments.
        metadata: Additional metadata associated with the result.
    """

    result_id: str
    title: str
    content: str
    original_score: float
    final_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonalizedRankingRequest(BaseModel):
    """Request model for personalized result ranking.

    Attributes:
        user_id: User identifier for personalization.
        results: List of result dictionaries to rank.
        preferences: Category preference weights (optional).
    """

    user_id: str | None = None
    results: list[dict[str, Any]]
    preferences: dict[str, float] | None = None


class PersonalizedRankingResponse(BaseModel):
    """Response model containing ranked search results.

    Attributes:
        ranked_results: List of results sorted by final score.
    """

    ranked_results: list[RankedResult]


def _apply_preferences(
    base_scores: list[float],
    metadata: list[dict[str, Any]],
    preferences: dict[str, float] | None,
) -> list[float]:
    """Apply category-based preference boosts to scores.

    Args:
        base_scores: Original relevance scores.
        metadata: Result metadata containing categories.
        preferences: Category preference weights.

    Returns:
        Adjusted scores with preference boosts applied.
    """
    if not preferences:
        return base_scores
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


def _normalize(scores: list[float]) -> list[float]:
    """Normalize scores to 0-1 range.

    Args:
        scores: List of scores to normalize.

    Returns:
        Normalized scores in range [0, 1].
    """
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if minimum == maximum:
        return [1.0 for _ in scores]
    span = maximum - minimum
    return [(score - minimum) / span for score in scores]


@dataclass(slots=True)
class PersonalizedRankingService:
    """Service for ranking search results based on user preferences.

    This service applies category-based preference boosts and normalizes
    scores to provide personalized result ordering.
    """

    async def initialize(self) -> None:  # pragma: no cover
        """Initialize the ranking service.

        This method is a no-op but maintained for interface consistency.
        """
        return

    async def rank_results(
        self, request: PersonalizedRankingRequest
    ) -> PersonalizedRankingResponse:
        """Rank search results based on user preferences.

        Args:
            request: Request containing results and preference data.

        Returns:
            Response with results ranked by adjusted scores.
        """
        base_scores = [float(item.get("score", 0.0)) for item in request.results]
        metadata = [dict(item.get("metadata") or {}) for item in request.results]
        adjusted = _apply_preferences(base_scores, metadata, request.preferences)
        normalized = _normalize(adjusted)
        ranked = [
            RankedResult(
                result_id=str(item.get("id", index)),
                title=str(item.get("title", "")),
                content=str(item.get("content", "")),
                original_score=base_score,
                final_score=score,
                metadata=metadata[index],
            )
            for index, (item, base_score, score) in enumerate(
                zip(request.results, base_scores, normalized, strict=False)
            )
        ]
        ranked.sort(key=lambda item: item.final_score, reverse=True)
        return PersonalizedRankingResponse(ranked_results=ranked)


__all__ = [
    "PersonalizedRankingService",
    "PersonalizedRankingRequest",
    "PersonalizedRankingResponse",
    "RankedResult",
]
