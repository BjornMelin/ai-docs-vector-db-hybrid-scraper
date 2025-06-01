"""Test SmartSelectionConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import SmartSelectionConfig


class TestSmartSelectionConfig:
    """Test SmartSelectionConfig model validation and behavior."""

    def test_default_values(self):
        """Test SmartSelectionConfig with default values."""
        config = SmartSelectionConfig()

        # Token estimation
        assert config.chars_per_token == 4.0

        # Scoring weights (must sum to 1.0)
        assert config.quality_weight == 0.4
        assert config.speed_weight == 0.3
        assert config.cost_weight == 0.3

        # Quality thresholds (0-100 scale)
        assert config.quality_fast_threshold == 60.0
        assert config.quality_balanced_threshold == 75.0
        assert config.quality_best_threshold == 85.0

        # Speed thresholds (tokens/second)
        assert config.speed_fast_threshold == 500.0
        assert config.speed_balanced_threshold == 200.0
        assert config.speed_slow_threshold == 100.0

        # Cost thresholds (per million tokens)
        assert config.cost_cheap_threshold == 50.0
        assert config.cost_moderate_threshold == 100.0
        assert config.cost_expensive_threshold == 200.0

        # Budget management
        assert config.budget_warning_threshold == 0.8
        assert config.budget_critical_threshold == 0.9

        # Text analysis
        assert config.short_text_threshold == 100
        assert config.long_text_threshold == 2000

        # Code detection keywords
        assert isinstance(config.code_keywords, set)
        assert "def" in config.code_keywords
        assert "class" in config.code_keywords
        assert "import" in config.code_keywords

    def test_weights_sum_to_one_valid(self):
        """Test valid weight combinations that sum to 1.0."""
        # Exact sum to 1.0
        config1 = SmartSelectionConfig(
            quality_weight=0.5, speed_weight=0.3, cost_weight=0.2
        )
        assert config1.quality_weight == 0.5
        assert config1.speed_weight == 0.3
        assert config1.cost_weight == 0.2

        # Within floating point tolerance
        config2 = SmartSelectionConfig(
            quality_weight=0.333, speed_weight=0.333, cost_weight=0.334
        )
        assert (
            abs(
                (config2.quality_weight + config2.speed_weight + config2.cost_weight)
                - 1.0
            )
            <= 0.01
        )

    def test_weights_sum_validation_error(self):
        """Test that weights must sum to 1.0."""
        # Sum > 1.0
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(quality_weight=0.5, speed_weight=0.4, cost_weight=0.3)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Scoring weights must sum to 1.0" in str(errors[0]["msg"])

        # Sum < 1.0
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(quality_weight=0.2, speed_weight=0.2, cost_weight=0.2)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Scoring weights must sum to 1.0" in str(errors[0]["msg"])

    def test_weight_range_constraints(self):
        """Test weight range constraints (0 <= weight <= 1)."""
        # Valid edge cases
        config1 = SmartSelectionConfig(
            quality_weight=1.0, speed_weight=0.0, cost_weight=0.0
        )
        assert config1.quality_weight == 1.0

        # Invalid: negative weight
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(quality_weight=-0.1, speed_weight=0.5, cost_weight=0.6)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("quality_weight",)
        assert "greater than or equal to 0" in str(errors[0]["msg"])

        # Invalid: weight > 1
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(quality_weight=1.1, speed_weight=0.0, cost_weight=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("quality_weight",)
        assert "less than or equal to 1" in str(errors[0]["msg"])

    def test_chars_per_token_positive(self):
        """Test chars_per_token must be positive."""
        config = SmartSelectionConfig(chars_per_token=3.5)
        assert config.chars_per_token == 3.5

        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(chars_per_token=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("chars_per_token",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_quality_thresholds(self):
        """Test quality threshold constraints (0-100 scale)."""
        # Valid thresholds
        config = SmartSelectionConfig(
            quality_fast_threshold=50.0,
            quality_balanced_threshold=70.0,
            quality_best_threshold=90.0,
        )
        assert config.quality_fast_threshold == 50.0

        # Invalid: below 0
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(quality_fast_threshold=-10.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("quality_fast_threshold",)

        # Invalid: above 100
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(quality_best_threshold=101.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("quality_best_threshold",)

    def test_speed_thresholds_positive(self):
        """Test speed thresholds must be positive."""
        config = SmartSelectionConfig(
            speed_fast_threshold=1000.0,
            speed_balanced_threshold=300.0,
            speed_slow_threshold=50.0,
        )
        assert config.speed_fast_threshold == 1000.0

        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(speed_fast_threshold=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("speed_fast_threshold",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_cost_thresholds_non_negative(self):
        """Test cost thresholds must be non-negative."""
        # Valid: 0 cost (free)
        config = SmartSelectionConfig(cost_cheap_threshold=0.0)
        assert config.cost_cheap_threshold == 0.0

        # Invalid: negative cost
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(cost_cheap_threshold=-10.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("cost_cheap_threshold",)
        assert "greater than or equal to 0" in str(errors[0]["msg"])

    def test_budget_thresholds(self):
        """Test budget threshold constraints (0 < threshold <= 1)."""
        config = SmartSelectionConfig(
            budget_warning_threshold=0.7, budget_critical_threshold=0.95
        )
        assert config.budget_warning_threshold == 0.7
        assert config.budget_critical_threshold == 0.95

        # Invalid: <= 0
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(budget_warning_threshold=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("budget_warning_threshold",)
        assert "greater than 0" in str(errors[0]["msg"])

        # Invalid: > 1
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(budget_critical_threshold=1.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("budget_critical_threshold",)
        assert "less than or equal to 1" in str(errors[0]["msg"])

    def test_text_thresholds_positive(self):
        """Test text thresholds must be positive."""
        config = SmartSelectionConfig(short_text_threshold=50, long_text_threshold=3000)
        assert config.short_text_threshold == 50
        assert config.long_text_threshold == 3000

        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(short_text_threshold=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("short_text_threshold",)

    def test_code_keywords_set(self):
        """Test code keywords set field."""
        # Custom keywords
        custom_keywords = {"func", "method", "module", "package"}
        config = SmartSelectionConfig(code_keywords=custom_keywords)
        assert config.code_keywords == custom_keywords

        # Empty set is valid
        config2 = SmartSelectionConfig(code_keywords=set())
        assert config2.code_keywords == set()

        # Test default keywords
        default_config = SmartSelectionConfig()
        assert len(default_config.code_keywords) > 10
        assert all(isinstance(kw, str) for kw in default_config.code_keywords)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(
                quality_weight=0.4,
                speed_weight=0.3,
                cost_weight=0.3,
                unknown_field="value",
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = SmartSelectionConfig(
            quality_weight=0.5,
            speed_weight=0.2,
            cost_weight=0.3,
            code_keywords={"def", "class", "import"},
        )

        # Test model_dump
        data = config.model_dump()
        assert data["quality_weight"] == 0.5
        assert data["speed_weight"] == 0.2
        assert data["cost_weight"] == 0.3
        assert data["code_keywords"] == {"def", "class", "import"}

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"quality_weight":0.5' in json_str
        assert '"speed_weight":0.2' in json_str
