"""Keyword-based intent classifier.

This module provides intent classification for search queries using
keyword matching to determine the user's intent (procedural, troubleshooting,
conceptual, comparative, or general).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class QueryIntentClassification(BaseModel):
    """Represents the classification result for a query intent.

    Attributes:
        primary_intent: The primary intent classification.
        confidence: Confidence score for the classification (0.0-1.0).
    """

    primary_intent: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)


@dataclass(slots=True)
class QueryIntentClassifier:
    """Service for classifying query intents based on keyword matching.

    This classifier uses a keyword-based approach to determine the
    intent behind user queries, helping to route them to appropriate processing
    pipelines or result formatting.
    """

    keyword_map: Mapping[str, str] = field(
        default_factory=lambda: {
            "install": "procedural",
            "configure": "procedural",
            "setup": "procedural",
            "error": "troubleshooting",
            "exception": "troubleshooting",
            "how": "procedural",
            "why": "conceptual",
            "compare": "comparative",
        }
    )

    async def initialize(self) -> None:  # pragma: no cover
        """Initialize the intent classifier.

        This method is a no-op but maintained for interface consistency.
        """
        return

    async def classify(self, query: str) -> QueryIntentClassification:
        """Classify the intent of a query based on keyword matching.

        Args:
            query: The query string to classify.

        Returns:
            Classification result with intent and confidence score.
        """
        lowered = query.lower()
        for keyword, label in self.keyword_map.items():
            if keyword in lowered:
                return QueryIntentClassification(primary_intent=label, confidence=0.7)
        return QueryIntentClassification(primary_intent="general", confidence=0.3)


__all__ = ["QueryIntentClassifier", "QueryIntentClassification"]
