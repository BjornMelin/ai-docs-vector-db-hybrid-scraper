"""Metadata filtering with advanced boolean logic and nested expressions.

This module provides sophisticated metadata filtering capabilities including
complex boolean logic (AND, OR, NOT), nested expressions, flexible field matching,
and integration with custom metadata schemas.
"""

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from qdrant_client import models

from .base import BaseFilter
from .base import FilterError
from .base import FilterResult

logger = logging.getLogger(__name__)


class FieldOperator(str, Enum):
    """Supported field operators for metadata filtering."""

    # Equality operators
    EQ = "eq"  # Equal
    NE = "ne"  # Not equal

    # Comparison operators
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal

    # Collection operators
    IN = "in"  # Value in list
    NIN = "nin"  # Value not in list

    # String operators
    CONTAINS = "contains"  # String contains substring
    STARTS_WITH = "starts_with"  # String starts with
    ENDS_WITH = "ends_with"  # String ends with
    REGEX = "regex"  # Regular expression match

    # Existence operators
    EXISTS = "exists"  # Field exists
    NOT_EXISTS = "not_exists"  # Field does not exist

    # Text search operators
    TEXT_MATCH = "text_match"  # Full text search

    # Geographic operators (for future use)
    GEO_WITHIN = "geo_within"  # Within geographic bounds


class BooleanOperator(str, Enum):
    """Boolean logic operators."""

    AND = "and"
    OR = "or"
    NOT = "not"


class FieldConditionModel(BaseModel):
    """Single field condition for metadata filtering."""

    field: str = Field(..., description="Field name to filter on")
    operator: FieldOperator = Field(..., description="Comparison operator")
    value: Any = Field(None, description="Value to compare against")
    values: list[Any] | None = Field(
        None, description="List of values for IN/NIN operators"
    )
    case_sensitive: bool = Field(
        True, description="Case sensitivity for string operations"
    )

    @field_validator("values")
    @classmethod
    def validate_values_for_operator(cls, v, info):
        """Validate values list is provided for appropriate operators."""
        if info.data.get("operator") in [FieldOperator.IN, FieldOperator.NIN] and not v:
            raise ValueError(
                f"values list required for {info.data['operator']} operator"
            )
        return v

    @model_validator(mode="after")
    def validate_value_or_values(self):
        """Ensure either value or values is provided based on operator."""
        if self.operator in [FieldOperator.IN, FieldOperator.NIN]:
            if not self.values:
                raise ValueError(f"values list required for {self.operator} operator")
        elif (
            self.operator not in [FieldOperator.EXISTS, FieldOperator.NOT_EXISTS]
            and self.value is None
        ):
            raise ValueError(f"value required for {self.operator} operator")

        return self


class BooleanExpressionModel(BaseModel):
    """Boolean expression for combining multiple conditions."""

    operator: BooleanOperator = Field(..., description="Boolean operator")
    conditions: list["BooleanExpressionModel" | FieldConditionModel] = Field(
        default_factory=list, description="List of sub-conditions"
    )

    @field_validator("conditions")
    @classmethod
    def validate_conditions_for_operator(cls, v, info):
        """Validate conditions based on boolean operator."""
        operator = info.data.get("operator")

        if operator == BooleanOperator.NOT:
            if len(v) != 1:
                raise ValueError("NOT operator requires exactly one condition")
        elif operator in [BooleanOperator.AND, BooleanOperator.OR] and len(v) < 2:
            raise ValueError(f"{operator} operator requires at least two conditions")

        return v


# Enable forward references for recursive model
BooleanExpressionModel.model_rebuild()


class MetadataFilterCriteria(BaseModel):
    """Criteria for metadata filtering operations."""

    # Direct field conditions (simple interface)
    field_conditions: list[FieldConditionModel] | None = Field(
        None, description="List of field conditions"
    )

    # Boolean expression (advanced interface)
    expression: BooleanExpressionModel | None = Field(
        None, description="Complex boolean expression"
    )

    # Shorthand conditions for common cases
    exact_matches: dict[str, Any] | None = Field(
        None, description="Fields that must match exactly"
    )
    exclude_matches: dict[str, Any] | None = Field(
        None, description="Fields that must not match"
    )
    range_filters: dict[str, dict[str, Any]] | None = Field(
        None, description="Range filters with gte/lte/gt/lt"
    )
    text_searches: dict[str, str] | None = Field(
        None, description="Text search filters"
    )

    # Advanced options
    default_boolean_operator: BooleanOperator = Field(
        BooleanOperator.AND, description="Default operator for combining conditions"
    )
    ignore_case: bool = Field(
        False, description="Global case insensitivity for string operations"
    )

    @model_validator(mode="after")
    def validate_at_least_one_condition(self):
        """Ensure at least one filtering condition is provided."""
        has_conditions = any(
            [
                self.field_conditions,
                self.expression,
                self.exact_matches,
                self.exclude_matches,
                self.range_filters,
                self.text_searches,
            ]
        )

        if not has_conditions:
            raise ValueError("At least one filtering condition must be provided")

        return self


class MetadataFilter(BaseFilter):
    """Advanced metadata filtering with boolean logic and nested expressions."""

    def __init__(
        self,
        name: str = "metadata_filter",
        description: str = "Filter documents using complex metadata conditions with boolean logic",
        enabled: bool = True,
        priority: int = 70,
    ):
        """Initialize metadata filter.

        Args:
            name: Filter name
            description: Filter description
            enabled: Whether filter is enabled
            priority: Filter priority (higher = earlier execution)
        """
        super().__init__(name, description, enabled, priority)

        # Field type mappings for Qdrant optimization
        self.field_types = {
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
            "list": list,
            "dict": dict,
        }

    async def apply(
        self, filter_criteria: dict[str, Any], context: dict[str, Any] | None = None
    ) -> FilterResult:
        """Apply metadata filtering with boolean logic.

        Args:
            filter_criteria: Metadata filter criteria
            context: Optional context with field schemas and settings

        Returns:
            FilterResult with Qdrant metadata filter conditions

        Raises:
            FilterError: If metadata filter application fails
        """
        try:
            # Validate and parse criteria
            criteria = MetadataFilterCriteria.model_validate(filter_criteria)

            # Build Qdrant filter conditions
            conditions = []
            metadata = {"applied_filters": [], "boolean_logic": {}}

            # Process shorthand conditions first
            shorthand_conditions = self._build_shorthand_conditions(criteria)
            conditions.extend(shorthand_conditions)
            if shorthand_conditions:
                metadata["applied_filters"].append("shorthand")

            # Process direct field conditions
            if criteria.field_conditions:
                field_conditions = self._build_field_conditions(
                    criteria.field_conditions, criteria
                )
                conditions.extend(field_conditions)
                metadata["applied_filters"].append("field_conditions")

            # Process complex boolean expression
            if criteria.expression:
                expression_condition = self._build_boolean_expression(
                    criteria.expression, criteria
                )
                if expression_condition:
                    conditions.append(expression_condition)
                    metadata["applied_filters"].append("boolean_expression")
                    metadata["boolean_logic"]["has_complex_expression"] = True

            # Calculate performance impact
            performance_impact = self._estimate_performance_impact(
                len(conditions), criteria.expression is not None
            )

            # Build final filter
            final_filter = None
            if conditions:
                # Combine conditions using default boolean operator
                if len(conditions) == 1:
                    final_filter = models.Filter(must=[conditions[0]])
                elif criteria.default_boolean_operator == BooleanOperator.AND:
                    final_filter = models.Filter(must=conditions)
                elif criteria.default_boolean_operator == BooleanOperator.OR:
                    final_filter = models.Filter(should=conditions)
                else:
                    # Default to AND for safety
                    final_filter = models.Filter(must=conditions)

                # Add metadata about the filter structure
                metadata["boolean_logic"].update(
                    {
                        "total_conditions": len(conditions),
                        "default_operator": criteria.default_boolean_operator.value,
                        "case_insensitive": criteria.ignore_case,
                    }
                )

            self._logger.info(
                f"Applied metadata filter with {len(conditions)} conditions: "
                f"{metadata['applied_filters']}"
            )

            return FilterResult(
                filter_conditions=final_filter,
                metadata=metadata,
                confidence_score=0.95,
                performance_impact=performance_impact,
            )

        except Exception as e:
            error_msg = f"Failed to apply metadata filter: {e}"
            self._logger.error(error_msg, exc_info=True)
            raise FilterError(
                error_msg,
                filter_name=self.name,
                filter_criteria=filter_criteria,
                underlying_error=e,
            ) from e

    def _build_shorthand_conditions(
        self, criteria: MetadataFilterCriteria
    ) -> list[models.FieldCondition]:
        """Build conditions from shorthand syntax."""
        conditions = []

        # Exact matches
        if criteria.exact_matches:
            for field, value in criteria.exact_matches.items():
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchValue(value=value)
                    )
                )

        # Exclude matches
        if criteria.exclude_matches:
            for field, value in criteria.exclude_matches.items():
                if isinstance(value, list):
                    # Convert all values to strings if they're not string/int (but exclude bool even though it's int)
                    exclude_values = [
                        (
                            str(v)
                            if isinstance(v, bool) or not isinstance(v, str | int)
                            else v
                        )
                        for v in value
                    ]
                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchExcept(**{"except": exclude_values}),
                        )
                    )
                else:
                    # Convert value to string if it's not string/int (but exclude bool even though it's int)
                    exclude_value = (
                        str(value)
                        if isinstance(value, bool) or not isinstance(value, str | int)
                        else value
                    )
                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchExcept(**{"except": [exclude_value]}),
                        )
                    )

        # Range filters
        if criteria.range_filters:
            for field, range_params in criteria.range_filters.items():
                conditions.append(
                    models.FieldCondition(key=field, range=models.Range(**range_params))
                )

        # Text searches
        if criteria.text_searches:
            for field, text in criteria.text_searches.items():
                conditions.append(
                    models.FieldCondition(key=field, match=models.MatchText(text=text))
                )

        return conditions

    def _build_field_conditions(
        self,
        field_conditions: list[FieldConditionModel],
        criteria: MetadataFilterCriteria,
    ) -> list[models.FieldCondition]:
        """Build Qdrant conditions from field condition models."""
        conditions = []

        for condition in field_conditions:
            qdrant_condition = self._build_single_field_condition(condition, criteria)
            if qdrant_condition:
                conditions.append(qdrant_condition)

        return conditions

    def _build_single_field_condition(
        self, condition: FieldConditionModel, criteria: MetadataFilterCriteria
    ) -> models.FieldCondition | None:
        """Build a single Qdrant field condition."""
        field = condition.field
        operator = condition.operator
        value = condition.value
        values = condition.values

        # Apply global case sensitivity setting (unused for now)
        # case_sensitive = condition.case_sensitive and not criteria.ignore_case

        try:
            # Existence operators
            if operator == FieldOperator.EXISTS:
                # For exists, we use a range that matches any value
                return models.FieldCondition(
                    key=field, range=models.Range(gte=float("-inf"), lte=float("inf"))
                )
            elif operator == FieldOperator.NOT_EXISTS:
                # For not exists, we use must_not with the exists condition
                # This needs to be handled at a higher level
                return None

            # Equality operators
            elif operator == FieldOperator.EQ:
                return models.FieldCondition(
                    key=field, match=models.MatchValue(value=value)
                )
            elif operator == FieldOperator.NE:
                # Convert value to string if it's not string/int (but exclude bool even though it's int)
                exclude_value = (
                    str(value)
                    if isinstance(value, bool) or not isinstance(value, str | int)
                    else value
                )
                return models.FieldCondition(
                    key=field, match=models.MatchExcept(**{"except": [exclude_value]})
                )

            # Comparison operators
            elif operator == FieldOperator.GT:
                return models.FieldCondition(key=field, range=models.Range(gt=value))
            elif operator == FieldOperator.GTE:
                return models.FieldCondition(key=field, range=models.Range(gte=value))
            elif operator == FieldOperator.LT:
                return models.FieldCondition(key=field, range=models.Range(lt=value))
            elif operator == FieldOperator.LTE:
                return models.FieldCondition(key=field, range=models.Range(lte=value))

            # Collection operators
            elif operator == FieldOperator.IN:
                return models.FieldCondition(
                    key=field, match=models.MatchAny(any=values)
                )
            elif operator == FieldOperator.NIN:
                # Convert all values to strings if they're not string/int (but exclude bool even though it's int)
                exclude_values = [
                    str(v) if isinstance(v, bool) or not isinstance(v, str | int) else v
                    for v in values
                ]
                return models.FieldCondition(
                    key=field, match=models.MatchExcept(**{"except": exclude_values})
                )

            # String operators (implemented as text search)
            elif operator in [
                FieldOperator.CONTAINS,
                FieldOperator.STARTS_WITH,
                FieldOperator.ENDS_WITH,
                FieldOperator.TEXT_MATCH,
            ]:
                # Convert to appropriate text search
                if operator == FieldOperator.STARTS_WITH:
                    search_text = f"{value}*"
                elif operator == FieldOperator.ENDS_WITH:
                    search_text = f"*{value}"
                else:
                    search_text = str(value)

                return models.FieldCondition(
                    key=field, match=models.MatchText(text=search_text)
                )

            # Regular expression (approximated with text search)
            elif operator == FieldOperator.REGEX:
                # Note: Qdrant doesn't support regex directly, so we use text search
                # This is a limitation that would need to be handled differently
                self._logger.warning(
                    f"Regex operator on field '{field}' approximated with text search"
                )
                return models.FieldCondition(
                    key=field, match=models.MatchText(text=str(value))
                )

            else:
                self._logger.warning(f"Unsupported operator: {operator}")
                return None

        except Exception as e:
            self._logger.error(f"Failed to build condition for field '{field}': {e}")
            return None

    def _build_boolean_expression(
        self, expression: BooleanExpressionModel, criteria: MetadataFilterCriteria
    ) -> models.Filter | models.FieldCondition | None:
        """Build Qdrant filter from boolean expression."""
        operator = expression.operator
        conditions = []

        # Process each sub-condition
        for condition in expression.conditions:
            if isinstance(condition, FieldConditionModel):
                # Single field condition
                qdrant_condition = self._build_single_field_condition(
                    condition, criteria
                )
                if qdrant_condition:
                    conditions.append(qdrant_condition)
            elif isinstance(condition, BooleanExpressionModel):
                # Nested boolean expression
                nested_filter = self._build_boolean_expression(condition, criteria)
                if nested_filter:
                    conditions.append(nested_filter)

        if not conditions:
            return None

        # Apply boolean operator
        if operator == BooleanOperator.AND:
            return models.Filter(must=conditions)
        elif operator == BooleanOperator.OR:
            return models.Filter(should=conditions)
        elif operator == BooleanOperator.NOT:
            # NOT with single condition
            if len(conditions) == 1:
                return models.Filter(must_not=[conditions[0]])
            else:
                # NOT with multiple conditions (treat as NOT (condition1 OR condition2 ...))
                return models.Filter(must_not=[models.Filter(should=conditions)])

        return None

    def _estimate_performance_impact(
        self, condition_count: int, has_complex_expression: bool
    ) -> str:
        """Estimate performance impact based on filter complexity."""
        base_impact = "none"

        if condition_count == 0:
            return "none"
        elif condition_count <= 2:
            base_impact = "low"
        elif condition_count <= 5:
            base_impact = "medium"
        else:
            base_impact = "high"

        # Increase impact for complex boolean expressions
        if has_complex_expression:
            if base_impact == "low":
                base_impact = "medium"
            elif base_impact == "medium":
                base_impact = "high"

        return base_impact

    async def validate_criteria(self, filter_criteria: dict[str, Any]) -> bool:
        """Validate metadata filter criteria."""
        try:
            MetadataFilterCriteria.model_validate(filter_criteria)
            return True
        except Exception as e:
            self._logger.warning(f"Invalid metadata criteria: {e}")
            return False

    def get_supported_operators(self) -> list[str]:
        """Get supported metadata operators."""
        return [op.value for op in FieldOperator] + [op.value for op in BooleanOperator]

    def build_expression_from_dict(
        self, expression_dict: dict[str, Any]
    ) -> BooleanExpressionModel:
        """Build a boolean expression from a dictionary representation.

        Args:
            expression_dict: Dictionary representation of boolean expression

        Returns:
            BooleanExpressionModel instance

        Example:
            {
                "and": [
                    {"field": "category", "operator": "eq", "value": "tutorial"},
                    {
                        "or": [
                            {"field": "difficulty", "operator": "eq", "value": "beginner"},
                            {"field": "tags", "operator": "contains", "value": "easy"}
                        ]
                    }
                ]
            }
        """
        # Find the boolean operator (should be the only key)
        if len(expression_dict) != 1:
            raise ValueError(
                "Expression dict must have exactly one boolean operator key"
            )

        operator_key, conditions_list = next(iter(expression_dict.items()))

        try:
            operator = BooleanOperator(operator_key)
        except ValueError:
            raise ValueError(f"Invalid boolean operator: {operator_key}") from None

        # Parse conditions
        parsed_conditions = []
        for condition in conditions_list:
            if isinstance(condition, dict):
                # Check if it's a field condition or nested boolean expression
                if "field" in condition and "operator" in condition:
                    # Field condition
                    parsed_conditions.append(
                        FieldConditionModel.model_validate(condition)
                    )
                else:
                    # Nested boolean expression
                    parsed_conditions.append(self.build_expression_from_dict(condition))
            else:
                raise ValueError(f"Invalid condition format: {condition}")

        return BooleanExpressionModel(operator=operator, conditions=parsed_conditions)

    def optimize_expression(
        self, expression: BooleanExpressionModel
    ) -> BooleanExpressionModel:
        """Optimize boolean expression for better performance.

        Args:
            expression: Boolean expression to optimize

        Returns:
            Optimized boolean expression
        """
        # Simple optimizations:
        # 1. Flatten nested expressions with same operator
        # 2. Remove redundant conditions
        # 3. Reorder conditions for better performance

        optimized_conditions = []

        for condition in expression.conditions:
            if isinstance(condition, BooleanExpressionModel):
                # Recursively optimize nested expressions
                optimized_nested = self.optimize_expression(condition)

                # Flatten if same operator
                if (
                    optimized_nested.operator == expression.operator
                    and expression.operator in [BooleanOperator.AND, BooleanOperator.OR]
                ):
                    optimized_conditions.extend(optimized_nested.conditions)
                else:
                    optimized_conditions.append(optimized_nested)
            else:
                optimized_conditions.append(condition)

        # Remove duplicates (simple field conditions only)
        unique_conditions = []
        seen_field_conditions = set()

        for condition in optimized_conditions:
            if isinstance(condition, FieldConditionModel):
                condition_key = (
                    condition.field,
                    condition.operator,
                    str(condition.value),
                )
                if condition_key not in seen_field_conditions:
                    seen_field_conditions.add(condition_key)
                    unique_conditions.append(condition)
            else:
                unique_conditions.append(condition)

        return BooleanExpressionModel(
            operator=expression.operator, conditions=unique_conditions
        )

    def explain_filter(self, criteria: MetadataFilterCriteria) -> str:
        """Generate human-readable explanation of the filter criteria.

        Args:
            criteria: Metadata filter criteria

        Returns:
            Human-readable explanation string
        """
        explanations = []

        # Explain shorthand conditions
        if criteria.exact_matches:
            explanations.append(f"Exact matches: {criteria.exact_matches}")

        if criteria.exclude_matches:
            explanations.append(f"Exclude matches: {criteria.exclude_matches}")

        if criteria.range_filters:
            explanations.append(f"Range filters: {criteria.range_filters}")

        if criteria.text_searches:
            explanations.append(f"Text searches: {criteria.text_searches}")

        # Explain field conditions
        if criteria.field_conditions:
            field_explanations = []
            for condition in criteria.field_conditions:
                field_explanations.append(
                    f"{condition.field} {condition.operator.value} {condition.value or condition.values}"
                )
            explanations.append(f"Field conditions: [{', '.join(field_explanations)}]")

        # Explain boolean expression
        if criteria.expression:
            explanations.append(
                f"Boolean expression: {self._explain_expression(criteria.expression)}"
            )

        return f"Metadata filter: {' AND '.join(explanations)}"

    def _explain_expression(self, expression: BooleanExpressionModel) -> str:
        """Generate explanation for boolean expression."""
        condition_explanations = []

        for condition in expression.conditions:
            if isinstance(condition, FieldConditionModel):
                condition_explanations.append(
                    f"{condition.field} {condition.operator.value} {condition.value or condition.values}"
                )
            elif isinstance(condition, BooleanExpressionModel):
                condition_explanations.append(
                    f"({self._explain_expression(condition)})"
                )

        operator_word = expression.operator.value.upper()
        if expression.operator == BooleanOperator.NOT:
            return f"NOT ({condition_explanations[0]})"
        else:
            return f" {operator_word} ".join(condition_explanations)
