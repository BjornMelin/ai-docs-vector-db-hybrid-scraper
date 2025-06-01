"""Test HyDEConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import HyDEConfig


class TestHyDEConfig:
    """Test HyDEConfig model validation and behavior."""

    def test_default_values(self):
        """Test HyDEConfig with default values."""
        config = HyDEConfig()

        # Feature flags
        assert config.enable_hyde is True
        assert config.enable_fallback is True
        assert config.enable_reranking is True
        assert config.enable_caching is True

        # Generation settings
        assert config.num_generations == 5
        assert config.generation_temperature == 0.7
        assert config.max_generation_tokens == 200
        assert config.generation_model == "gpt-3.5-turbo"
        assert config.generation_timeout_seconds == 10

        # Search settings
        assert config.hyde_prefetch_limit == 50
        assert config.query_prefetch_limit == 30
        assert config.hyde_weight_in_fusion == 0.6
        assert config.fusion_algorithm == "rrf"

        # Caching settings
        assert config.cache_ttl_seconds == 3600
        assert config.cache_hypothetical_docs is True
        assert config.cache_prefix == "hyde"

        # Performance settings
        assert config.parallel_generation is True
        assert config.max_concurrent_generations == 5

        # Prompt engineering
        assert config.use_domain_specific_prompts is True
        assert config.prompt_variation is True

        # Quality control
        assert config.min_generation_length == 20
        assert config.filter_duplicates is True
        assert config.diversity_threshold == 0.3

        # Monitoring and debugging
        assert config.log_generations is False
        assert config.track_metrics is True

        # A/B testing
        assert config.ab_testing_enabled is False
        assert config.control_group_percentage == 0.5

    def test_custom_values(self):
        """Test HyDEConfig with custom values."""
        config = HyDEConfig(
            enable_hyde=False,
            num_generations=3,
            generation_temperature=0.5,
            max_generation_tokens=300,
            generation_model="gpt-4",
            hyde_prefetch_limit=100,
            query_prefetch_limit=50,
            hyde_weight_in_fusion=0.8,
            fusion_algorithm="dbsf",
            cache_ttl_seconds=7200,
            parallel_generation=False,
            min_generation_length=30,
            diversity_threshold=0.5,
            log_generations=True,
            ab_testing_enabled=True,
            control_group_percentage=0.3,
        )

        assert config.enable_hyde is False
        assert config.num_generations == 3
        assert config.generation_temperature == 0.5
        assert config.max_generation_tokens == 300
        assert config.generation_model == "gpt-4"
        assert config.hyde_prefetch_limit == 100
        assert config.query_prefetch_limit == 50
        assert config.hyde_weight_in_fusion == 0.8
        assert config.fusion_algorithm == "dbsf"
        assert config.cache_ttl_seconds == 7200
        assert config.parallel_generation is False
        assert config.min_generation_length == 30
        assert config.diversity_threshold == 0.5
        assert config.log_generations is True
        assert config.ab_testing_enabled is True
        assert config.control_group_percentage == 0.3

    def test_num_generations_constraints(self):
        """Test num_generations constraints (1 <= value <= 10)."""
        # Valid values
        config1 = HyDEConfig(num_generations=1)
        assert config1.num_generations == 1

        config2 = HyDEConfig(num_generations=10)
        assert config2.num_generations == 10

        # Invalid: below minimum
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(num_generations=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("num_generations",)
        assert "greater than or equal to 1" in str(errors[0]["msg"])

        # Invalid: above maximum
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(num_generations=11)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("num_generations",)
        assert "less than or equal to 10" in str(errors[0]["msg"])

    def test_generation_temperature_constraints(self):
        """Test generation_temperature constraints (0.0 <= value <= 1.0)."""
        # Valid values
        config1 = HyDEConfig(generation_temperature=0.0)
        assert config1.generation_temperature == 0.0

        config2 = HyDEConfig(generation_temperature=1.0)
        assert config2.generation_temperature == 1.0

        # Invalid: below minimum
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_temperature=-0.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("generation_temperature",)

        # Invalid: above maximum
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_temperature=1.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("generation_temperature",)

    def test_max_generation_tokens_constraints(self):
        """Test max_generation_tokens constraints (50 <= value <= 500)."""
        # Valid values
        config1 = HyDEConfig(max_generation_tokens=50)
        assert config1.max_generation_tokens == 50

        config2 = HyDEConfig(max_generation_tokens=500)
        assert config2.max_generation_tokens == 500

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_generation_tokens=49)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_generation_tokens",)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_generation_tokens=501)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_generation_tokens",)

    def test_generation_timeout_constraints(self):
        """Test generation_timeout_seconds constraints (1 <= value <= 60)."""
        # Valid values
        config1 = HyDEConfig(generation_timeout_seconds=1)
        assert config1.generation_timeout_seconds == 1

        config2 = HyDEConfig(generation_timeout_seconds=60)
        assert config2.generation_timeout_seconds == 60

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_timeout_seconds=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("generation_timeout_seconds",)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_timeout_seconds=61)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("generation_timeout_seconds",)

    def test_prefetch_limit_constraints(self):
        """Test prefetch limit constraints."""
        # Valid hyde_prefetch_limit (10 <= value <= 200)
        config1 = HyDEConfig(hyde_prefetch_limit=10)
        assert config1.hyde_prefetch_limit == 10

        config2 = HyDEConfig(hyde_prefetch_limit=200)
        assert config2.hyde_prefetch_limit == 200

        # Invalid hyde_prefetch_limit
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(hyde_prefetch_limit=9)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("hyde_prefetch_limit",)

        # Valid query_prefetch_limit (10 <= value <= 100)
        config3 = HyDEConfig(query_prefetch_limit=10)
        assert config3.query_prefetch_limit == 10

        config4 = HyDEConfig(query_prefetch_limit=100)
        assert config4.query_prefetch_limit == 100

        # Invalid query_prefetch_limit
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(query_prefetch_limit=101)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("query_prefetch_limit",)

    def test_hyde_weight_in_fusion_constraints(self):
        """Test hyde_weight_in_fusion constraints (0.0 <= value <= 1.0)."""
        # Valid values
        config1 = HyDEConfig(hyde_weight_in_fusion=0.0)
        assert config1.hyde_weight_in_fusion == 0.0

        config2 = HyDEConfig(hyde_weight_in_fusion=1.0)
        assert config2.hyde_weight_in_fusion == 1.0

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(hyde_weight_in_fusion=-0.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("hyde_weight_in_fusion",)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(hyde_weight_in_fusion=1.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("hyde_weight_in_fusion",)

    def test_cache_ttl_constraints(self):
        """Test cache_ttl_seconds constraints (300 <= value <= 86400)."""
        # Valid values
        config1 = HyDEConfig(cache_ttl_seconds=300)  # 5 minutes
        assert config1.cache_ttl_seconds == 300

        config2 = HyDEConfig(cache_ttl_seconds=86400)  # 24 hours
        assert config2.cache_ttl_seconds == 86400

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(cache_ttl_seconds=299)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("cache_ttl_seconds",)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(cache_ttl_seconds=86401)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("cache_ttl_seconds",)

    def test_max_concurrent_generations_constraints(self):
        """Test max_concurrent_generations constraints (1 <= value <= 10)."""
        # Valid values
        config1 = HyDEConfig(max_concurrent_generations=1)
        assert config1.max_concurrent_generations == 1

        config2 = HyDEConfig(max_concurrent_generations=10)
        assert config2.max_concurrent_generations == 10

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_concurrent_generations=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_concurrent_generations",)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_concurrent_generations=11)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_concurrent_generations",)

    def test_min_generation_length_constraints(self):
        """Test min_generation_length constraints (10 <= value <= 100)."""
        # Valid values
        config1 = HyDEConfig(min_generation_length=10)
        assert config1.min_generation_length == 10

        config2 = HyDEConfig(min_generation_length=100)
        assert config2.min_generation_length == 100

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(min_generation_length=9)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("min_generation_length",)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(min_generation_length=101)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("min_generation_length",)

    def test_diversity_threshold_constraints(self):
        """Test diversity_threshold constraints (0.0 <= value <= 1.0)."""
        # Valid values
        config1 = HyDEConfig(diversity_threshold=0.0)
        assert config1.diversity_threshold == 0.0

        config2 = HyDEConfig(diversity_threshold=1.0)
        assert config2.diversity_threshold == 1.0

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(diversity_threshold=-0.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("diversity_threshold",)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(diversity_threshold=1.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("diversity_threshold",)

    def test_control_group_percentage_constraints(self):
        """Test control_group_percentage constraints (0.0 <= value <= 1.0)."""
        # Valid values
        config1 = HyDEConfig(control_group_percentage=0.0)
        assert config1.control_group_percentage == 0.0

        config2 = HyDEConfig(control_group_percentage=1.0)
        assert config2.control_group_percentage == 1.0

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(control_group_percentage=-0.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("control_group_percentage",)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(control_group_percentage=1.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("control_group_percentage",)

    def test_boolean_fields(self):
        """Test all boolean fields."""
        config = HyDEConfig(
            enable_hyde=False,
            enable_fallback=False,
            enable_reranking=False,
            enable_caching=False,
            cache_hypothetical_docs=False,
            parallel_generation=False,
            use_domain_specific_prompts=False,
            prompt_variation=False,
            filter_duplicates=False,
            log_generations=True,
            track_metrics=False,
            ab_testing_enabled=True,
        )

        assert config.enable_hyde is False
        assert config.enable_fallback is False
        assert config.enable_reranking is False
        assert config.enable_caching is False
        assert config.cache_hypothetical_docs is False
        assert config.parallel_generation is False
        assert config.use_domain_specific_prompts is False
        assert config.prompt_variation is False
        assert config.filter_duplicates is False
        assert config.log_generations is True
        assert config.track_metrics is False
        assert config.ab_testing_enabled is True

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(enable_hyde=True, unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = HyDEConfig(
            enable_hyde=False,
            num_generations=7,
            generation_model="gpt-4-turbo",
            fusion_algorithm="dbsf",
            cache_prefix="hyde_test",
        )

        # Test model_dump
        data = config.model_dump()
        assert data["enable_hyde"] is False
        assert data["num_generations"] == 7
        assert data["generation_model"] == "gpt-4-turbo"
        assert data["fusion_algorithm"] == "dbsf"
        assert data["cache_prefix"] == "hyde_test"

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"enable_hyde":false' in json_str
        assert '"num_generations":7' in json_str
        assert '"generation_model":"gpt-4-turbo"' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = HyDEConfig(num_generations=5, generation_temperature=0.7)

        updated = original.model_copy(
            update={
                "num_generations": 8,
                "generation_temperature": 0.9,
                "enable_hyde": False,
            }
        )

        assert original.num_generations == 5
        assert original.generation_temperature == 0.7
        assert original.enable_hyde is True  # Default
        assert updated.num_generations == 8
        assert updated.generation_temperature == 0.9
        assert updated.enable_hyde is False

    def test_typical_configurations(self):
        """Test typical HyDE configuration scenarios."""
        # Conservative configuration
        conservative = HyDEConfig(
            num_generations=3,
            generation_temperature=0.3,
            max_generation_tokens=100,
            hyde_weight_in_fusion=0.4,
            enable_fallback=True,
        )
        assert conservative.num_generations == 3
        assert conservative.generation_temperature == 0.3

        # Aggressive configuration
        aggressive = HyDEConfig(
            num_generations=10,
            generation_temperature=0.9,
            max_generation_tokens=400,
            hyde_weight_in_fusion=0.8,
            parallel_generation=True,
            max_concurrent_generations=10,
        )
        assert aggressive.num_generations == 10
        assert aggressive.max_concurrent_generations == 10

        # Testing configuration
        testing = HyDEConfig(
            log_generations=True,
            track_metrics=True,
            ab_testing_enabled=True,
            control_group_percentage=0.5,
        )
        assert testing.log_generations is True
        assert testing.ab_testing_enabled is True
