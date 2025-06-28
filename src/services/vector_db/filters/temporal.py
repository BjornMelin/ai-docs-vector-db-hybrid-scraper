"""Temporal filtering for date-based filtering and content freshness analysis.

This module provides sophisticated temporal filtering capabilities including
absolute and relative date filtering, content freshness scoring, and
time-based content relevance analysis.
"""

import logging
import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator
from qdrant_client import models

from .base import BaseFilter, FilterError, FilterResult


logger = logging.getLogger(__name__)


class TemporalCriteria(BaseModel):
    """Criteria for temporal filtering operations."""

    # Absolute date filters
    created_after: datetime | None = Field(
        None, description="Filter documents created after this datetime"
    )
    created_before: datetime | None = Field(
        None, description="Filter documents created before this datetime"
    )
    updated_after: datetime | None = Field(
        None, description="Filter documents updated after this datetime"
    )
    updated_before: datetime | None = Field(
        None, description="Filter documents updated before this datetime"
    )
    crawled_after: datetime | None = Field(
        None, description="Filter documents crawled after this datetime"
    )
    crawled_before: datetime | None = Field(
        None, description="Filter documents crawled before this datetime"
    )

    # Relative date filters
    created_within_days: int | None = Field(
        None, ge=1, description="Filter documents created within N days"
    )
    updated_within_days: int | None = Field(
        None, ge=1, description="Filter documents updated within N days"
    )
    crawled_within_days: int | None = Field(
        None, ge=1, description="Filter documents crawled within N days"
    )

    # Content freshness
    max_age_days: int | None = Field(
        None, ge=1, description="Maximum age of content in days"
    )
    freshness_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum freshness score (0-1)"
    )

    # Time-based relevance
    time_decay_factor: float = Field(
        0.1, ge=0.0, le=1.0, description="Time decay factor for relevance scoring"
    )
    boost_recent_content: bool = Field(
        False, description="Boost scores for more recent content"
    )

    @field_validator("created_before", "updated_before", "crawled_before")
    @classmethod
    def validate_before_dates(cls, v, _info):
        """Ensure 'before' dates are not in the future."""
        if v and v > datetime.now(tz=UTC):
            msg = f"Date cannot be in the future: {v}"
            raise ValueError(msg)
        return v

    @field_validator("created_after", "updated_after", "crawled_after")
    @classmethod
    def validate_after_dates(cls, v, _info):
        """Validate 'after' dates are reasonable."""
        if v and v > datetime.now(tz=UTC):
            msg = f"Date cannot be in the future: {v}"
            raise ValueError(msg)
        return v


class FreshnessScore(BaseModel):
    """Content freshness scoring configuration."""

    score: float = Field(..., ge=0.0, le=1.0, description="Freshness score")
    age_days: int = Field(..., ge=0, description="Content age in days")
    decay_function: str = Field(
        "exponential", description="Decay function (exponential, linear, step)"
    )
    half_life_days: int = Field(
        30, ge=1, description="Half-life for exponential decay in days"
    )


class TemporalFilter(BaseFilter):
    """Advanced temporal filtering with freshness analysis and relative dates."""

    def __init__(
        self,
        name: str = "temporal_filter",
        description: str = "Filter documents based on temporal criteria and content freshness",
        enabled: bool = True,
        priority: int = 90,
    ):
        """Initialize temporal filter.

        Args:
            name: Filter name
            description: Filter description
            enabled: Whether filter is enabled
            priority: Filter priority (higher = earlier execution)

        """
        super().__init__(name, description, enabled, priority)

        # Temporal field mappings for Qdrant
        self.temporal_fields = {
            "created": "created_at",
            "updated": "last_updated",
            "crawled": "crawl_timestamp",
        }

        # Freshness calculation parameters
        self.default_half_life_days = 30
        self.max_freshness_age_days = 365

    async def apply(
        self, filter_criteria: dict[str, Any], context: dict[str, Any] | None = None
    ) -> FilterResult:
        """Apply temporal filtering with freshness analysis.

        Args:
            filter_criteria: Temporal filter criteria
            context: Optional context with current time and collection info

        Returns:
            FilterResult with Qdrant temporal filter conditions

        Raises:
            FilterError: If temporal filter application fails

        """
        try:
            # Validate and parse criteria
            criteria = TemporalCriteria.model_validate(filter_criteria)

            # Get current time from context or use UTC now
            current_time = datetime.now(tz=UTC)
            if context and "current_time" in context:
                current_time = context["current_time"]

            # Build Qdrant filter conditions
            conditions = []
            metadata = {"applied_filters": [], "freshness_info": {}}

            # Process absolute date filters
            absolute_conditions = self._build_absolute_date_filters(criteria)
            conditions.extend(absolute_conditions)
            if absolute_conditions:
                metadata["applied_filters"].append("absolute_dates")

            # Process relative date filters
            relative_conditions = self._build_relative_date_filters(
                criteria, current_time
            )
            conditions.extend(relative_conditions)
            if relative_conditions:
                metadata["applied_filters"].append("relative_dates")

            # Process freshness filters
            freshness_conditions = self._build_freshness_filters(criteria, current_time)
            conditions.extend(freshness_conditions)
            if freshness_conditions:
                metadata["applied_filters"].append("freshness")

            # Calculate performance impact
            performance_impact = self._estimate_performance_impact(len(conditions))

            # Build final filter
            final_filter = None
            if conditions:
                final_filter = models.Filter(must=conditions)

                # Add freshness scoring metadata if enabled
                if criteria.boost_recent_content:
                    metadata["freshness_info"] = {
                        "time_decay_factor": criteria.time_decay_factor,
                        "boost_enabled": True,
                        "current_time": current_time.isoformat(),
                    }

            self._logger.info(
                f"Applied temporal filter with {len(conditions)} conditions: "
                f"{metadata['applied_filters']}"
            )

            return FilterResult(
                filter_conditions=final_filter,
                metadata=metadata,
                confidence_score=0.95,
                performance_impact=performance_impact,
            )

        except Exception as e:
            error_msg = f"Failed to apply temporal filter: {e}"
            self._logger.error(error_msg, exc_info=True)
            raise FilterError(
                error_msg,
                filter_name=self.name,
                filter_criteria=filter_criteria,
                underlying_error=e,
            ) from e

    def _build_absolute_date_filters(
        self, criteria: TemporalCriteria
    ) -> list[models.FieldCondition]:
        """Build filters for absolute date ranges."""
        conditions = []

        # Created date filters
        if criteria.created_after:
            conditions.append(
                models.FieldCondition(
                    key=self.temporal_fields["created"],
                    range=models.Range(gte=criteria.created_after.timestamp()),
                )
            )

        if criteria.created_before:
            conditions.append(
                models.FieldCondition(
                    key=self.temporal_fields["created"],
                    range=models.Range(lte=criteria.created_before.timestamp()),
                )
            )

        # Updated date filters
        if criteria.updated_after:
            conditions.append(
                models.FieldCondition(
                    key=self.temporal_fields["updated"],
                    range=models.Range(gte=criteria.updated_after.timestamp()),
                )
            )

        if criteria.updated_before:
            conditions.append(
                models.FieldCondition(
                    key=self.temporal_fields["updated"],
                    range=models.Range(lte=criteria.updated_before.timestamp()),
                )
            )

        # Crawled date filters
        if criteria.crawled_after:
            conditions.append(
                models.FieldCondition(
                    key=self.temporal_fields["crawled"],
                    range=models.Range(gte=criteria.crawled_after.timestamp()),
                )
            )

        if criteria.crawled_before:
            conditions.append(
                models.FieldCondition(
                    key=self.temporal_fields["crawled"],
                    range=models.Range(lte=criteria.crawled_before.timestamp()),
                )
            )

        return conditions

    def _build_relative_date_filters(
        self, criteria: TemporalCriteria, current_time: datetime
    ) -> list[models.FieldCondition]:
        """Build filters for relative date ranges."""
        conditions = []

        # Created within N days
        if criteria.created_within_days:
            cutoff_time = current_time - timedelta(days=criteria.created_within_days)
            conditions.append(
                models.FieldCondition(
                    key=self.temporal_fields["created"],
                    range=models.Range(gte=cutoff_time.timestamp()),
                )
            )

        # Updated within N days
        if criteria.updated_within_days:
            cutoff_time = current_time - timedelta(days=criteria.updated_within_days)
            conditions.append(
                models.FieldCondition(
                    key=self.temporal_fields["updated"],
                    range=models.Range(gte=cutoff_time.timestamp()),
                )
            )

        # Crawled within N days
        if criteria.crawled_within_days:
            cutoff_time = current_time - timedelta(days=criteria.crawled_within_days)
            conditions.append(
                models.FieldCondition(
                    key=self.temporal_fields["crawled"],
                    range=models.Range(gte=cutoff_time.timestamp()),
                )
            )

        # Max age filter
        if criteria.max_age_days:
            cutoff_time = current_time - timedelta(days=criteria.max_age_days)
            # Apply to the most recent of created/updated/crawled
            conditions.append(
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key=self.temporal_fields["created"],
                            range=models.Range(gte=cutoff_time.timestamp()),
                        ),
                        models.FieldCondition(
                            key=self.temporal_fields["updated"],
                            range=models.Range(gte=cutoff_time.timestamp()),
                        ),
                        models.FieldCondition(
                            key=self.temporal_fields["crawled"],
                            range=models.Range(gte=cutoff_time.timestamp()),
                        ),
                    ]
                )
            )

        return conditions

    def _build_freshness_filters(
        self, criteria: TemporalCriteria, current_time: datetime
    ) -> list[models.FieldCondition]:
        """Build filters based on content freshness scores."""
        conditions = []

        if criteria.freshness_threshold:
            # Calculate age threshold based on freshness score
            # Using exponential decay: freshness = e^(-age/half_life)
            # Solving for age: age = -half_life * ln(freshness_threshold)

            half_life_days = self.default_half_life_days
            max_age_for_threshold = -half_life_days * math.log(
                criteria.freshness_threshold
            )

            # Limit to reasonable maximum
            max_age_for_threshold = min(
                max_age_for_threshold, self.max_freshness_age_days
            )

            cutoff_time = current_time - timedelta(days=max_age_for_threshold)

            # Apply freshness filter to most recent timestamp
            conditions.append(
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key=self.temporal_fields["updated"],
                            range=models.Range(gte=cutoff_time.timestamp()),
                        ),
                        models.FieldCondition(
                            key=self.temporal_fields["crawled"],
                            range=models.Range(gte=cutoff_time.timestamp()),
                        ),
                    ]
                )
            )

        return conditions

    def _estimate_performance_impact(self, condition_count: int) -> str:
        """Estimate performance impact based on filter complexity."""
        if condition_count == 0:
            return "none"
        if condition_count <= 2:
            return "low"
        if condition_count <= 4:
            return "medium"
        return "high"

    async def validate_criteria(self, filter_criteria: dict[str, Any]) -> bool:
        """Validate temporal filter criteria."""
        try:
            TemporalCriteria.model_validate(filter_criteria)
            return True
        except Exception as e:
            self._logger.warning(f"Invalid temporal criteria: {e}")
            return False

    def get_supported_operators(self) -> list[str]:
        """Get supported temporal operators."""
        return [
            "created_after",
            "created_before",
            "created_within_days",
            "updated_after",
            "updated_before",
            "updated_within_days",
            "crawled_after",
            "crawled_before",
            "crawled_within_days",
            "max_age_days",
            "freshness_threshold",
        ]

    def calculate_freshness_score(
        self,
        content_date: datetime,
        current_time: datetime | None = None,
        decay_function: str = "exponential",
        half_life_days: int | None = None,
    ) -> FreshnessScore:
        """Calculate content freshness score.

        Args:
            content_date: Date when content was created/updated
            current_time: Current time for calculation
            decay_function: Type of decay (exponential, linear, step)
            half_life_days: Half-life for exponential decay

        Returns:
            FreshnessScore with calculated score and metadata

        """
        if current_time is None:
            current_time = datetime.now(tz=UTC)

        if half_life_days is None:
            half_life_days = self.default_half_life_days

        # Calculate age in days
        age_delta = current_time - content_date
        age_days = max(0, age_delta.days)

        # Calculate freshness score based on decay function
        if decay_function == "exponential":
            score = math.exp(-age_days / half_life_days)
        elif decay_function == "linear":
            max_age = half_life_days * 3  # Linear decay over 3x half-life
            score = max(0.0, 1.0 - (age_days / max_age))
        elif decay_function == "step":
            score = (
                1.0
                if age_days <= half_life_days
                else 0.5
                if age_days <= half_life_days * 2
                else 0.1
            )
        else:
            # Default to exponential

            score = math.exp(-age_days / half_life_days)

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        return FreshnessScore(
            score=score,
            age_days=age_days,
            decay_function=decay_function,
            half_life_days=half_life_days,
        )

    def parse_relative_date(self, relative_date_str: str) -> datetime | None:
        """Parse relative date strings like 'last week', 'past month', etc.

        Args:
            relative_date_str: Relative date string

        Returns:
            Parsed datetime or None if parsing fails

        """
        relative_date_str = relative_date_str.lower().strip()
        current_time = datetime.now(tz=UTC)

        # Pattern matching for common relative dates
        patterns = [
            (
                r"last (\d+) days?",
                lambda m: current_time - timedelta(days=int(m.group(1))),
            ),
            (
                r"past (\d+) days?",
                lambda m: current_time - timedelta(days=int(m.group(1))),
            ),
            (r"last week", lambda _m: current_time - timedelta(weeks=1)),
            (r"past week", lambda _m: current_time - timedelta(weeks=1)),
            (r"last month", lambda _m: current_time - timedelta(days=30)),
            (r"past month", lambda _m: current_time - timedelta(days=30)),
            (r"last year", lambda _m: current_time - timedelta(days=365)),
            (r"past year", lambda _m: current_time - timedelta(days=365)),
            (r"yesterday", lambda _m: current_time - timedelta(days=1)),
            (
                r"today",
                lambda _m: current_time.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ),
            ),
        ]

        for pattern, calculator in patterns:
            match = re.match(pattern, relative_date_str)
            if match:
                try:
                    return calculator(match)
                except Exception as e:
                    self._logger.warning(
                        f"Failed to calculate relative date for '{relative_date_str}': {e}"
                    )
                    return None

        self._logger.warning(
            f"Unrecognized relative date format: '{relative_date_str}'"
        )
        return None
