"""Query expansion utilities built on synonym heuristics.

This module provides query expansion functionality to increase search recall
by adding synonyms, variants, and related terms to user queries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


_DEFAULT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "install": ("setup", "configure"),
    "configuration": ("settings", "setup"),
    "error": ("issue", "problem"),
    "update": ("upgrade", "patch"),
    "delete": ("remove", "uninstall"),
    "create": ("make", "build"),
    "performance": ("speed", "throughput"),
    "security": ("auth", "authorization"),
}


class ExpandedTerm(BaseModel):
    """Represents an expanded term with metadata.

    Attributes:
        term: The expanded term text.
        original_term: The original term from the query.
        relation_type: Type of relationship (original, variant, synonym).
        confidence: Confidence score for the expansion (0.0-1.0).
        source: Source of the expansion (input, auto_variant, synonym_map).
    """

    term: str = Field(..., description="Expanded term")
    original_term: str = Field(..., description="Original term")
    relation_type: str = Field("synonym", description="Relationship type")
    confidence: float = Field(0.6, ge=0.0, le=1.0, description="Heuristic confidence")
    source: str = Field("synonym_map", description="Expansion source")


class QueryExpansionRequest(BaseModel):
    """Request model for query expansion.

    Attributes:
        original_query: The original search query to expand.
        max_expanded_terms: Maximum number of expanded terms to generate (1-50).
    """

    original_query: str = Field(..., description="Original search query")
    max_expanded_terms: int = Field(10, ge=1, le=50, description="Maximum expansions")


class QueryExpansionResult(BaseModel):
    """Result model containing expanded query information.

    Attributes:
        original_query: The original query text.
        expanded_terms: List of expanded terms with metadata.
        expanded_query: The expanded query string.
        confidence_score: Overall confidence score for the expansion (0.0-1.0).
    """

    original_query: str
    expanded_terms: list[ExpandedTerm]
    expanded_query: str
    confidence_score: float = Field(0.6, ge=0.0, le=1.0)


def _split_terms(query: str) -> Iterable[str]:
    """Split a query into individual terms.

    Args:
        query: The query string to split.

    Returns:
        Iterable of individual terms from the query.
    """
    return re.findall(r"[A-Za-z0-9_]+", query.lower())


@dataclass(slots=True)
class QueryExpansionService:
    """Service for expanding search queries with synonyms and variants.

    This service improves search recall by expanding queries with related terms
    including plurals, gerunds, and predefined synonyms.
    """

    synonym_map: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(_DEFAULT_SYNONYMS)
    )

    async def initialize(self) -> None:  # pragma: no cover
        """Initialize the query expansion service.

        This method is a no-op but maintained for interface consistency.
        """
        return

    async def expand_query(
        self, request: QueryExpansionRequest
    ) -> QueryExpansionResult:
        """Expand a query with synonyms and variants.

        Args:
            request: Request containing the query to expand and parameters.

        Returns:
            Result containing the expanded query and metadata about expansions.
        """
        seen: set[str] = set()
        expansions: list[ExpandedTerm] = []

        for term in _split_terms(request.original_query):
            if term in seen:
                continue
            seen.add(term)
            expansions.append(
                ExpandedTerm(
                    term=term,
                    original_term=request.original_query,
                    relation_type="original",
                    source="input",
                    confidence=0.8,
                )
            )
            for variant in (f"{term}s", f"{term}ing"):
                if variant not in seen:
                    seen.add(variant)
                    expansions.append(
                        ExpandedTerm(
                            term=variant,
                            original_term=request.original_query,
                            relation_type="variant",
                            source="auto_variant",
                            confidence=0.5,
                        )
                    )
            for synonym in self.synonym_map.get(term, ()):  # type: ignore[arg-type]
                if synonym not in seen:
                    seen.add(synonym)
                    expansions.append(
                        ExpandedTerm(
                            term=synonym,
                            original_term=request.original_query,
                            relation_type="synonym",
                            source="synonym_map",
                            confidence=0.7,
                        )
                    )
            if len(expansions) >= request.max_expanded_terms:
                break

        combined_query = request.original_query
        if expansions:
            additional = " OR ".join(term.term for term in expansions[1:])
            if additional:
                combined_query = f"{request.original_query} ({additional})"

        return QueryExpansionResult(
            original_query=request.original_query,
            expanded_terms=expansions,
            expanded_query=combined_query,
            confidence_score=0.6,
        )


__all__ = [
    "QueryExpansionService",
    "QueryExpansionRequest",
    "QueryExpansionResult",
    "ExpandedTerm",
]
