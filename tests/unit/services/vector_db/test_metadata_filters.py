"""Tests for the metadata filter implementation."""

from typing import Any

import pytest
from qdrant_client import models

from src.services.vector_db.filters.base import FilterError, FilterResult
from src.services.vector_db.filters.metadata import (
    BooleanExpressionModel,
    BooleanOperator,
    FieldConditionModel,
    FieldOperator,
    MetadataFilter,
    MetadataFilterCriteria,
)


def build_field_condition(**kwargs: Any) -> FieldConditionModel:
    """Helper to create FieldConditionModel instances for tests."""

    return FieldConditionModel.model_validate(kwargs)


def build_boolean_expression(**kwargs: Any) -> BooleanExpressionModel:
    """Helper to create BooleanExpressionModel instances for tests."""

    return BooleanExpressionModel.model_validate(kwargs)


def build_metadata_criteria(**kwargs: Any) -> MetadataFilterCriteria:
    """Helper to create MetadataFilterCriteria instances for tests."""

    return MetadataFilterCriteria.model_validate(kwargs)


class TestFieldOperator:
    """Test FieldOperator enum."""

    def test_equality_operators(self):
        """Test equality operators."""
        assert FieldOperator.EQ.value == "eq"
        assert FieldOperator.NE.value == "ne"

    def test_comparison_operators(self):
        """Test comparison operators."""
        assert FieldOperator.GT.value == "gt"
        assert FieldOperator.GTE.value == "gte"
        assert FieldOperator.LT.value == "lt"
        assert FieldOperator.LTE.value == "lte"

    def test_collection_operators(self):
        """Test collection operators."""
        assert FieldOperator.IN.value == "in"
        assert FieldOperator.NIN.value == "nin"

    def test_string_operators(self):
        """Test string operators."""
        assert FieldOperator.CONTAINS.value == "contains"
        assert FieldOperator.STARTS_WITH.value == "starts_with"
        assert FieldOperator.ENDS_WITH.value == "ends_with"
        assert FieldOperator.REGEX.value == "regex"
        assert FieldOperator.TEXT_MATCH.value == "text_match"

    def test_existence_operators(self):
        """Test existence operators."""
        assert FieldOperator.EXISTS.value == "exists"
        assert FieldOperator.NOT_EXISTS.value == "not_exists"


class TestBooleanOperator:
    """Test BooleanOperator enum."""

    def test_boolean_operator_values(self):
        """Test boolean operator values."""
        assert BooleanOperator.AND.value == "and"
        assert BooleanOperator.OR.value == "or"
        assert BooleanOperator.NOT.value == "not"


class TestFieldConditionModel:
    """Test FieldConditionModel."""

    def test_basic_condition(self):
        """Test basic field condition."""
        condition = build_field_condition(
            field="author", operator=FieldOperator.EQ, value="John Doe"
        )

        assert condition.field == "author"
        assert condition.operator == FieldOperator.EQ
        assert condition.value == "John Doe"
        assert condition.case_sensitive is True

    def test_collection_operators_with_values(self):
        """Test operators that require values list."""
        condition = build_field_condition(
            field="tags", operator=FieldOperator.IN, values=["python", "django", "web"]
        )

        assert condition.field == "tags"
        assert condition.operator == FieldOperator.IN
        assert condition.values == ["python", "django", "web"]

    def test_validation_in_operator_requires_values(self):
        """Test IN operator validation requires values."""
        with pytest.raises(ValueError, match="values list required"):
            build_field_condition(
                field="tags",
                operator=FieldOperator.IN,
                value="python",  # Should use values, not value
            )

    def test_validation_eq_operator_requires_value(self):
        """Test EQ operator validation requires value."""
        with pytest.raises(ValueError, match="value required"):
            build_field_condition(
                field="author",
                operator=FieldOperator.EQ,
                # Missing value
            )

    def test_exists_operator_no_value_required(self):
        """Test EXISTS operator doesn't require value."""
        condition = build_field_condition(
            field="category", operator=FieldOperator.EXISTS
        )

        assert condition.field == "category"
        assert condition.operator == FieldOperator.EXISTS

    def test_case_sensitivity_setting(self):
        """Test case sensitivity setting."""
        condition = build_field_condition(
            field="title",
            operator=FieldOperator.CONTAINS,
            value="Machine Learning",
            case_sensitive=False,
        )

        assert condition.case_sensitive is False


class TestBooleanExpressionModel:
    """Test BooleanExpressionModel."""

    def test_and_expression(self):
        """Test AND expression."""
        conditions = [
            build_field_condition(
                field="author", operator=FieldOperator.EQ, value="Smith"
            ),
            build_field_condition(field="year", operator=FieldOperator.GT, value=2020),
        ]

        expression = build_boolean_expression(
            operator=BooleanOperator.AND, conditions=conditions
        )

        assert expression.operator == BooleanOperator.AND
        assert len(expression.conditions) == 2

    def test_or_expression(self):
        """Test OR expression."""
        conditions = [
            build_field_condition(
                field="category", operator=FieldOperator.EQ, value="AI"
            ),
            build_field_condition(
                field="category", operator=FieldOperator.EQ, value="ML"
            ),
        ]

        expression = build_boolean_expression(
            operator=BooleanOperator.OR, conditions=conditions
        )

        assert expression.operator == BooleanOperator.OR
        assert len(expression.conditions) == 2

    def test_not_expression(self):
        """Test NOT expression."""
        condition = build_field_condition(
            field="draft", operator=FieldOperator.EQ, value=True
        )

        expression = build_boolean_expression(
            operator=BooleanOperator.NOT, conditions=[condition]
        )

        assert expression.operator == BooleanOperator.NOT
        assert len(expression.conditions) == 1

    def test_validation_not_requires_one_condition(self):
        """Test NOT operator validation."""
        with pytest.raises(
            ValueError, match="NOT operator requires exactly one condition"
        ):
            build_boolean_expression(
                operator=BooleanOperator.NOT,
                conditions=[
                    build_field_condition(
                        field="a", operator=FieldOperator.EQ, value="1"
                    ),
                    build_field_condition(
                        field="b", operator=FieldOperator.EQ, value="2"
                    ),
                ],
            )

    def test_validation_and_requires_multiple_conditions(self):
        """Test AND operator validation."""
        with pytest.raises(
            ValueError, match="operator requires at least two conditions"
        ):
            build_boolean_expression(
                operator=BooleanOperator.AND,
                conditions=[
                    build_field_condition(
                        field="a", operator=FieldOperator.EQ, value="1"
                    )
                ],
            )

    def test_nested_expressions(self):
        """Test nested boolean expressions."""
        inner_expression = build_boolean_expression(
            operator=BooleanOperator.OR,
            conditions=[
                build_field_condition(
                    field="category", operator=FieldOperator.EQ, value="AI"
                ),
                build_field_condition(
                    field="category", operator=FieldOperator.EQ, value="ML"
                ),
            ],
        )

        outer_expression = build_boolean_expression(
            operator=BooleanOperator.AND,
            conditions=[
                build_field_condition(
                    field="author", operator=FieldOperator.EQ, value="Smith"
                ),
                inner_expression,
            ],
        )

        assert len(outer_expression.conditions) == 2
        assert isinstance(outer_expression.conditions[1], BooleanExpressionModel)


class TestMetadataFilterCriteria:
    """Test MetadataFilterCriteria."""

    def test_field_conditions_only(self):
        """Test criteria with field conditions only."""
        conditions = [
            build_field_condition(
                field="author", operator=FieldOperator.EQ, value="Smith"
            )
        ]

        criteria = build_metadata_criteria(field_conditions=conditions)

        assert criteria.field_conditions is not None
        assert len(criteria.field_conditions) == 1
        assert criteria.expression is None
        assert criteria.default_boolean_operator == BooleanOperator.AND

    def test_expression_only(self):
        """Test criteria with expression only."""
        expression = build_boolean_expression(
            operator=BooleanOperator.AND,
            conditions=[
                build_field_condition(
                    field="author", operator=FieldOperator.EQ, value="Smith"
                ),
                build_field_condition(
                    field="year", operator=FieldOperator.GT, value=2020
                ),
            ],
        )

        criteria = build_metadata_criteria(expression=expression)

        assert criteria.field_conditions is None
        assert criteria.expression is not None

    def test_shorthand_conditions(self):
        """Test shorthand condition formats."""
        criteria = build_metadata_criteria(
            exact_matches={"author": "Smith", "year": 2024},
            exclude_matches={"status": "draft"},
            range_filters={"score": {"gte": 8.0, "lte": 10.0}},
            text_searches={"content": "machine learning"},
        )

        assert criteria.exact_matches is not None
        assert criteria.exclude_matches is not None
        assert criteria.range_filters is not None
        assert criteria.text_searches is not None
        assert criteria.exact_matches["author"] == "Smith"
        assert criteria.exclude_matches["status"] == "draft"
        assert criteria.range_filters["score"]["gte"] == 8.0
        assert criteria.text_searches["content"] == "machine learning"

    def test_validation_requires_at_least_one_condition(self):
        """Test validation requires at least one condition."""
        with pytest.raises(
            ValueError, match="At least one filtering condition must be provided"
        ):
            build_metadata_criteria()

    def test_default_boolean_operator(self):
        """Test default boolean operator setting."""
        criteria = build_metadata_criteria(
            exact_matches={"test": "value"}, default_boolean_operator=BooleanOperator.OR
        )

        assert criteria.default_boolean_operator == BooleanOperator.OR

    def test_ignore_case_setting(self):
        """Test ignore case setting."""
        criteria = build_metadata_criteria(
            exact_matches={"test": "value"}, ignore_case=True
        )

        assert criteria.ignore_case is True


@pytest.fixture
def metadata_filter() -> MetadataFilter:
    """Create metadata filter instance."""

    return MetadataFilter()


class TestMetadataFilterBasics:
    """Test MetadataFilter construction and helper utilities."""

    def test_initialization(self, metadata_filter):
        """Test filter initialization."""

        assert metadata_filter.name == "metadata_filter"
        assert metadata_filter.enabled is True
        assert metadata_filter.priority == 70

    def test_custom_initialization(self):
        """Test custom initialization."""

        filter_obj = MetadataFilter(
            name="custom_metadata",
            description="Custom metadata filter",
            enabled=False,
            priority=150,
        )

        assert filter_obj.name == "custom_metadata"
        assert filter_obj.description == "Custom metadata filter"
        assert filter_obj.enabled is False
        assert filter_obj.priority == 150

    def test_build_match_any_preserves_numeric_variants(self, metadata_filter):
        """Ensure MatchAny retains numeric payloads when floats are provided."""

        match = metadata_filter._build_match_any([0.5, 2])

        assert isinstance(match, models.MatchAny)
        assert match.any == [0.5, 2]

    def test_build_match_any_coerces_mixed_str_and_int(self, metadata_filter):
        """Ensure mixed string/integer collections use validated construction."""

        match = metadata_filter._build_match_any(["alpha", 2])

        assert isinstance(match, models.MatchAny)
        assert match.any == ["alpha", "2"]

    def test_build_match_except_preserves_numeric_variants(self, metadata_filter):
        """Ensure MatchExcept retains numeric payloads when floats are provided."""

        match_except = metadata_filter._build_match_except([1.5, 3])

        assert isinstance(match_except, models.MatchExcept)
        assert match_except.except_ == [1.5, 3]

    def test_get_supported_operators(self, metadata_filter):
        """Test getting supported operators."""

        operators = metadata_filter.get_supported_operators()

        # Should include all field operators and boolean operators
        assert "eq" in operators
        assert "ne" in operators
        assert "gt" in operators
        assert "gte" in operators
        assert "lt" in operators
        assert "lte" in operators
        assert "in" in operators
        assert "nin" in operators
        assert "contains" in operators
        assert "starts_with" in operators
        assert "ends_with" in operators
        assert "regex" in operators
        assert "exists" in operators
        assert "not_exists" in operators
        assert "text_match" in operators
        assert "and" in operators
        assert "or" in operators
        assert "not" in operators


class TestMetadataFilterApply:
    """Test applying metadata filters and related behaviours."""

    @pytest.mark.asyncio
    async def test_apply_exact_matches(self, metadata_filter):
        """Test applying filter with exact matches."""

        filter_criteria = {
            "exact_matches": {"author": "John Smith", "status": "published"}
        }

        result = await metadata_filter.apply(filter_criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is not None
        assert isinstance(result.filter_conditions, models.Filter)
        assert "shorthand" in result.metadata["applied_filters"]
        assert result.confidence_score == 0.95

    @pytest.mark.asyncio
    async def test_apply_exclude_matches(self, metadata_filter):
        """Test applying filter with exclude matches."""

        filter_criteria = {"exclude_matches": {"status": "draft", "deleted": True}}

        result = await metadata_filter.apply(filter_criteria)

        assert result.filter_conditions is not None
        assert "shorthand" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_range_filters(self, metadata_filter):
        """Test applying filter with range conditions."""

        filter_criteria = {
            "range_filters": {"year": {"gte": 2020, "lte": 2024}, "score": {"gt": 8.0}}
        }

        result = await metadata_filter.apply(filter_criteria)

        assert result.filter_conditions is not None
        assert "shorthand" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_text_searches(self, metadata_filter):
        """Test applying filter with text searches."""

        filter_criteria = {
            "text_searches": {"title": "machine learning", "content": "neural networks"}
        }

        result = await metadata_filter.apply(filter_criteria)

        assert result.filter_conditions is not None
        assert "shorthand" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_field_conditions(self, metadata_filter):
        """Test applying filter with field conditions."""

        filter_criteria = {
            "field_conditions": [
                {"field": "author", "operator": "eq", "value": "Smith"},
                {"field": "tags", "operator": "in", "values": ["python", "django"]},
            ]
        }

        result = await metadata_filter.apply(filter_criteria)

        assert result.filter_conditions is not None
        assert "field_conditions" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_boolean_expression(self, metadata_filter):
        """Test applying filter with boolean expression."""

        filter_criteria = {
            "expression": {
                "operator": "and",
                "conditions": [
                    {"field": "author", "operator": "eq", "value": "Smith"},
                    {"field": "year", "operator": "gte", "value": 2020},
                ],
            }
        }

        result = await metadata_filter.apply(filter_criteria)

        assert result.filter_conditions is not None
        assert "boolean_expression" in result.metadata["applied_filters"]
        assert result.metadata["boolean_logic"]["has_complex_expression"] is True

    @pytest.mark.asyncio
    async def test_apply_with_or_operator(self, metadata_filter):
        """Test applying filter with OR operator."""

        filter_criteria = {
            "exact_matches": {"category": "AI"},
            "default_boolean_operator": "or",
        }

        result = await metadata_filter.apply(filter_criteria)

        assert result.filter_conditions is not None
        assert result.metadata["boolean_logic"]["default_operator"] == "or"

    @pytest.mark.asyncio
    async def test_apply_case_insensitive(self, metadata_filter):
        """Test applying filter with case insensitive setting."""

        filter_criteria = {
            "field_conditions": [
                {
                    "field": "title",
                    "operator": "contains",
                    "value": "Machine Learning",
                    "case_sensitive": False,
                }
            ],
            "ignore_case": True,
        }

        result = await metadata_filter.apply(filter_criteria)

        assert result.filter_conditions is not None
        assert result.metadata["boolean_logic"]["case_insensitive"] is True

    @pytest.mark.asyncio
    async def test_performance_impact_estimation(self, metadata_filter):
        """Test performance impact estimation."""

        # Low impact (simple condition)
        filter_criteria = {"exact_matches": {"author": "Smith"}}

        result = await metadata_filter.apply(filter_criteria)
        assert result.performance_impact == "low"

        # Medium impact (multiple conditions)
        filter_criteria = {
            "exact_matches": {"author": "Smith", "year": 2024},
            "range_filters": {"score": {"gte": 8.0}},
        }

        result = await metadata_filter.apply(filter_criteria)
        assert result.performance_impact == "medium"

    @pytest.mark.asyncio
    async def test_error_handling(self, metadata_filter):
        """Test error handling during filter application."""

        filter_criteria = "invalid_criteria_format"

        with pytest.raises(FilterError) as exc_info:
            await metadata_filter.apply(filter_criteria)

        error = exc_info.value
        assert error.filter_name == "metadata_filter"
        assert "Failed to validate metadata filter criteria" in str(error)

    @pytest.mark.asyncio
    async def test_single_condition_filter(self, metadata_filter):
        """Test filter with single condition."""

        filter_criteria = {"exact_matches": {"author": "Smith"}}

        result = await metadata_filter.apply(filter_criteria)

        assert result.filter_conditions is not None
        assert result.filter_conditions.must is not None
        assert len(result.filter_conditions.must) == 1

    @pytest.mark.asyncio
    async def test_complex_nested_expression(self, metadata_filter):
        """Test complex nested boolean expression."""

        filter_criteria = {
            "expression": {
                "operator": "or",
                "conditions": [
                    {
                        "operator": "and",
                        "conditions": [
                            {"field": "author", "operator": "eq", "value": "Smith"},
                            {"field": "year", "operator": "gte", "value": 2020},
                        ],
                    },
                    {
                        "operator": "and",
                        "conditions": [
                            {"field": "category", "operator": "eq", "value": "AI"},
                            {"field": "featured", "operator": "eq", "value": True},
                        ],
                    },
                ],
            }
        }

        result = await metadata_filter.apply(filter_criteria)

        assert result.filter_conditions is not None
        assert result.metadata["boolean_logic"]["has_complex_expression"] is True
        assert result.performance_impact in ["medium", "high"]


class TestMetadataFilterValidation:
    """Test validation utilities for metadata filters."""

    @pytest.mark.asyncio
    async def test_validate_criteria_valid(self, metadata_filter):
        """Test validating valid criteria."""

        filter_criteria = {"exact_matches": {"author": "Smith"}}

        is_valid = await metadata_filter.validate_criteria(filter_criteria)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_criteria_invalid(self, metadata_filter):
        """Test validating invalid criteria."""

        filter_criteria = {}

        is_valid = await metadata_filter.validate_criteria(filter_criteria)
        assert is_valid is False

        filter_criteria = {
            "field_conditions": [
                {"field": "author", "operator": "invalid_operator", "value": "Smith"}
            ]
        }

        is_valid = await metadata_filter.validate_criteria(filter_criteria)
        assert is_valid is False


class TestMetadataFilterExpressions:
    """Test expression building and explanation helpers."""

    def test_build_expression_from_dict(self, metadata_filter):
        """Test building expression from dictionary."""

        expression_dict = {
            "and": [
                {"field": "author", "operator": "eq", "value": "Smith"},
                {
                    "or": [
                        {"field": "category", "operator": "eq", "value": "AI"},
                        {"field": "category", "operator": "eq", "value": "ML"},
                    ]
                },
            ]
        }

        expression = metadata_filter.build_expression_from_dict(expression_dict)

        assert expression.operator == BooleanOperator.AND
        assert len(expression.conditions) == 2
        assert isinstance(expression.conditions[0], FieldConditionModel)
        assert isinstance(expression.conditions[1], BooleanExpressionModel)

    def test_build_expression_from_dict_invalid(self, metadata_filter):
        """Test building expression from invalid dictionary."""

        with pytest.raises(
            ValueError, match="must have exactly one boolean operator key"
        ):
            metadata_filter.build_expression_from_dict({"and": [], "or": []})

        with pytest.raises(ValueError, match="Invalid boolean operator"):
            metadata_filter.build_expression_from_dict({"invalid": []})

    def test_optimize_expression(self, metadata_filter):
        """Test expression optimization."""

        inner_and = build_boolean_expression(
            operator=BooleanOperator.AND,
            conditions=[
                build_field_condition(field="b", operator=FieldOperator.EQ, value="2"),
                build_field_condition(field="c", operator=FieldOperator.EQ, value="3"),
            ],
        )

        expression = build_boolean_expression(
            operator=BooleanOperator.AND,
            conditions=[
                build_field_condition(field="a", operator=FieldOperator.EQ, value="1"),
                inner_and,
            ],
        )

        optimized = metadata_filter.optimize_expression(expression)

        assert len(optimized.conditions) == 3
        assert all(isinstance(c, FieldConditionModel) for c in optimized.conditions)

    def test_explain_filter(self, metadata_filter):
        """Test filter explanation."""

        criteria = build_metadata_criteria(
            exact_matches={"author": "Smith"},
            range_filters={"year": {"gte": 2020}},
            field_conditions=[
                build_field_condition(
                    field="status", operator=FieldOperator.EQ, value="published"
                )
            ],
        )

        explanation = metadata_filter.explain_filter(criteria)

        assert "Exact matches" in explanation
        assert "Range filters" in explanation
        assert "Field conditions" in explanation
        assert "Smith" in explanation
        assert "2020" in explanation
        assert "published" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
