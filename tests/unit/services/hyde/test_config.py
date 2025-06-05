"""Tests for HyDE configuration models."""

import pytest
from pydantic import ValidationError
from src.services.hyde.config import HyDEConfig
from src.services.hyde.config import HyDEMetricsConfig
from src.services.hyde.config import HyDEPromptConfig


class TestHyDEConfig:
    """Tests for HyDEConfig model."""

    def test_default_config(self):
        """Test default HyDE configuration."""
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

        # Monitoring
        assert config.log_generations is False
        assert config.track_metrics is True

    def test_custom_config(self):
        """Test custom HyDE configuration."""
        config = HyDEConfig(
            enable_hyde=False,
            num_generations=3,
            generation_temperature=0.5,
            max_generation_tokens=150,
            generation_model="gpt-4",
            generation_timeout_seconds=15,
            hyde_prefetch_limit=30,
            query_prefetch_limit=20,
            hyde_weight_in_fusion=0.8,
            fusion_algorithm="dbsf",
            cache_ttl_seconds=1800,
            cache_hypothetical_docs=False,
            cache_prefix="custom_hyde",
            parallel_generation=False,
            max_concurrent_generations=3,
            use_domain_specific_prompts=False,
            prompt_variation=False,
            min_generation_length=30,
            filter_duplicates=False,
            diversity_threshold=0.5,
            log_generations=True,
            track_metrics=False,
        )

        assert config.enable_hyde is False
        assert config.num_generations == 3
        assert config.generation_temperature == 0.5
        assert config.max_generation_tokens == 150
        assert config.generation_model == "gpt-4"
        assert config.generation_timeout_seconds == 15
        assert config.hyde_prefetch_limit == 30
        assert config.query_prefetch_limit == 20
        assert config.hyde_weight_in_fusion == 0.8
        assert config.fusion_algorithm == "dbsf"
        assert config.cache_ttl_seconds == 1800
        assert config.cache_hypothetical_docs is False
        assert config.cache_prefix == "custom_hyde"
        assert config.parallel_generation is False
        assert config.max_concurrent_generations == 3
        assert config.use_domain_specific_prompts is False
        assert config.prompt_variation is False
        assert config.min_generation_length == 30
        assert config.filter_duplicates is False
        assert config.diversity_threshold == 0.5
        assert config.log_generations is True
        assert config.track_metrics is False

    def test_validation_num_generations(self):
        """Test validation for num_generations field."""
        # Valid values
        HyDEConfig(num_generations=1)
        HyDEConfig(num_generations=5)
        HyDEConfig(num_generations=10)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(num_generations=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(num_generations=11)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_validation_generation_temperature(self):
        """Test validation for generation_temperature field."""
        # Valid values
        HyDEConfig(generation_temperature=0.0)
        HyDEConfig(generation_temperature=0.7)
        HyDEConfig(generation_temperature=1.0)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_temperature=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_temperature=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_validation_max_generation_tokens(self):
        """Test validation for max_generation_tokens field."""
        # Valid values
        HyDEConfig(max_generation_tokens=50)
        HyDEConfig(max_generation_tokens=200)
        HyDEConfig(max_generation_tokens=500)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_generation_tokens=49)
        assert "greater than or equal to 50" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_generation_tokens=501)
        assert "less than or equal to 500" in str(exc_info.value)

    def test_validation_generation_timeout_seconds(self):
        """Test validation for generation_timeout_seconds field."""
        # Valid values
        HyDEConfig(generation_timeout_seconds=1)
        HyDEConfig(generation_timeout_seconds=10)
        HyDEConfig(generation_timeout_seconds=60)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_timeout_seconds=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_timeout_seconds=61)
        assert "less than or equal to 60" in str(exc_info.value)

    def test_validation_hyde_prefetch_limit(self):
        """Test validation for hyde_prefetch_limit field."""
        # Valid values
        HyDEConfig(hyde_prefetch_limit=10)
        HyDEConfig(hyde_prefetch_limit=50)
        HyDEConfig(hyde_prefetch_limit=200)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(hyde_prefetch_limit=9)
        assert "greater than or equal to 10" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(hyde_prefetch_limit=201)
        assert "less than or equal to 200" in str(exc_info.value)

    def test_validation_query_prefetch_limit(self):
        """Test validation for query_prefetch_limit field."""
        # Valid values
        HyDEConfig(query_prefetch_limit=10)
        HyDEConfig(query_prefetch_limit=30)
        HyDEConfig(query_prefetch_limit=100)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(query_prefetch_limit=9)
        assert "greater than or equal to 10" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(query_prefetch_limit=101)
        assert "less than or equal to 100" in str(exc_info.value)

    def test_validation_hyde_weight_in_fusion(self):
        """Test validation for hyde_weight_in_fusion field."""
        # Valid values
        HyDEConfig(hyde_weight_in_fusion=0.0)
        HyDEConfig(hyde_weight_in_fusion=0.6)
        HyDEConfig(hyde_weight_in_fusion=1.0)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(hyde_weight_in_fusion=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(hyde_weight_in_fusion=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_validation_cache_ttl_seconds(self):
        """Test validation for cache_ttl_seconds field."""
        # Valid values
        HyDEConfig(cache_ttl_seconds=300)
        HyDEConfig(cache_ttl_seconds=3600)
        HyDEConfig(cache_ttl_seconds=86400)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(cache_ttl_seconds=299)
        assert "greater than or equal to 300" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(cache_ttl_seconds=86401)
        assert "less than or equal to 86400" in str(exc_info.value)

    def test_validation_max_concurrent_generations(self):
        """Test validation for max_concurrent_generations field."""
        # Valid values
        HyDEConfig(max_concurrent_generations=1)
        HyDEConfig(max_concurrent_generations=5)
        HyDEConfig(max_concurrent_generations=10)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_concurrent_generations=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_concurrent_generations=11)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_validation_min_generation_length(self):
        """Test validation for min_generation_length field."""
        # Valid values
        HyDEConfig(min_generation_length=10)
        HyDEConfig(min_generation_length=20)
        HyDEConfig(min_generation_length=100)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(min_generation_length=9)
        assert "greater than or equal to 10" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(min_generation_length=101)
        assert "less than or equal to 100" in str(exc_info.value)

    def test_validation_diversity_threshold(self):
        """Test validation for diversity_threshold field."""
        # Valid values
        HyDEConfig(diversity_threshold=0.0)
        HyDEConfig(diversity_threshold=0.3)
        HyDEConfig(diversity_threshold=1.0)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(diversity_threshold=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(diversity_threshold=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_json_schema_extra(self):
        """Test that JSON schema extra example is accessible."""
        config = HyDEConfig()
        example = config.model_config.get("json_schema_extra", {}).get("example", {})

        assert example["enable_hyde"] is True
        assert example["num_generations"] == 5
        assert example["generation_temperature"] == 0.7
        assert example["max_generation_tokens"] == 200
        assert example["generation_model"] == "gpt-3.5-turbo"
        assert example["hyde_prefetch_limit"] == 50
        assert example["query_prefetch_limit"] == 30
        assert example["cache_ttl_seconds"] == 3600
        assert example["parallel_generation"] is True

    def test_serialization(self):
        """Test config serialization and deserialization."""
        config = HyDEConfig(
            enable_hyde=False,
            num_generations=3,
            generation_temperature=0.8,
            generation_model="gpt-4",
        )

        # Test to dict
        config_dict = config.model_dump()
        assert config_dict["enable_hyde"] is False
        assert config_dict["num_generations"] == 3
        assert config_dict["generation_temperature"] == 0.8
        assert config_dict["generation_model"] == "gpt-4"

        # Test from dict
        restored_config = HyDEConfig.model_validate(config_dict)
        assert restored_config.enable_hyde is False
        assert restored_config.num_generations == 3
        assert restored_config.generation_temperature == 0.8
        assert restored_config.generation_model == "gpt-4"


class TestHyDEPromptConfig:
    """Tests for HyDEPromptConfig model."""

    def test_default_prompt_config(self):
        """Test default prompt configuration."""
        config = HyDEPromptConfig()

        # Check default prompts are present
        assert "technical documentation expert" in config.technical_prompt
        assert "{query}" in config.technical_prompt
        assert "code documentation expert" in config.code_prompt
        assert "tutorial writer" in config.tutorial_prompt
        assert "Answer the following question" in config.general_prompt

        # Check keyword lists
        assert "api" in config.technical_keywords
        assert "function" in config.technical_keywords
        assert "how to" in config.code_keywords
        assert "python" in config.code_keywords
        assert "tutorial" in config.tutorial_keywords
        assert "guide" in config.tutorial_keywords

        # Check variation templates
        assert "prefixes" in config.variation_templates
        assert "instruction_styles" in config.variation_templates
        assert "context_additions" in config.variation_templates

    def test_custom_prompt_config(self):
        """Test custom prompt configuration."""
        custom_technical_prompt = "Custom technical prompt: {query}"
        custom_code_prompt = "Custom code prompt: {query}"
        custom_tutorial_prompt = "Custom tutorial prompt: {query}"
        custom_general_prompt = "Custom general prompt: {query}"

        custom_technical_keywords = ["custom_api", "custom_function"]
        custom_code_keywords = ["custom_code", "custom_implement"]
        custom_tutorial_keywords = ["custom_tutorial", "custom_guide"]

        custom_variations = {
            "prefixes": ["Custom prefix {domain}"],
            "instruction_styles": ["Custom instruction"],
            "context_additions": ["Custom context"],
        }

        config = HyDEPromptConfig(
            technical_prompt=custom_technical_prompt,
            code_prompt=custom_code_prompt,
            tutorial_prompt=custom_tutorial_prompt,
            general_prompt=custom_general_prompt,
            technical_keywords=custom_technical_keywords,
            code_keywords=custom_code_keywords,
            tutorial_keywords=custom_tutorial_keywords,
            variation_templates=custom_variations,
        )

        assert config.technical_prompt == custom_technical_prompt
        assert config.code_prompt == custom_code_prompt
        assert config.tutorial_prompt == custom_tutorial_prompt
        assert config.general_prompt == custom_general_prompt
        assert config.technical_keywords == custom_technical_keywords
        assert config.code_keywords == custom_code_keywords
        assert config.tutorial_keywords == custom_tutorial_keywords
        assert config.variation_templates == custom_variations

    def test_prompt_template_formatting(self):
        """Test that prompt templates can be formatted with query."""
        config = HyDEPromptConfig()
        test_query = "How to use API authentication?"

        # Test each prompt template can be formatted
        technical_formatted = config.technical_prompt.format(query=test_query)
        assert test_query in technical_formatted
        assert technical_formatted != config.technical_prompt

        code_formatted = config.code_prompt.format(query=test_query)
        assert test_query in code_formatted
        assert code_formatted != config.code_prompt

        tutorial_formatted = config.tutorial_prompt.format(query=test_query)
        assert test_query in tutorial_formatted
        assert tutorial_formatted != config.tutorial_prompt

        general_formatted = config.general_prompt.format(query=test_query)
        assert test_query in general_formatted
        assert general_formatted != config.general_prompt

    def test_keyword_lists_not_empty(self):
        """Test that keyword lists are not empty."""
        config = HyDEPromptConfig()

        assert len(config.technical_keywords) > 0
        assert len(config.code_keywords) > 0
        assert len(config.tutorial_keywords) > 0

        # All keywords should be strings
        assert all(isinstance(keyword, str) for keyword in config.technical_keywords)
        assert all(isinstance(keyword, str) for keyword in config.code_keywords)
        assert all(isinstance(keyword, str) for keyword in config.tutorial_keywords)

    def test_variation_templates_structure(self):
        """Test variation templates have correct structure."""
        config = HyDEPromptConfig()

        variations = config.variation_templates
        assert isinstance(variations, dict)

        # Check required keys
        assert "prefixes" in variations
        assert "instruction_styles" in variations
        assert "context_additions" in variations

        # Check all values are lists of strings
        assert isinstance(variations["prefixes"], list)
        assert isinstance(variations["instruction_styles"], list)
        assert isinstance(variations["context_additions"], list)

        assert all(isinstance(prefix, str) for prefix in variations["prefixes"])
        assert all(isinstance(style, str) for style in variations["instruction_styles"])
        assert all(
            isinstance(addition, str) for addition in variations["context_additions"]
        )

        # Check lists are not empty
        assert len(variations["prefixes"]) > 0
        assert len(variations["instruction_styles"]) > 0
        assert len(variations["context_additions"]) > 0

    def test_domain_placeholder_in_variations(self):
        """Test that domain placeholder exists in variation prefixes."""
        config = HyDEPromptConfig()

        prefixes = config.variation_templates["prefixes"]
        domain_prefixes = [prefix for prefix in prefixes if "{domain}" in prefix]

        # At least some prefixes should have domain placeholder
        assert len(domain_prefixes) > 0

    def test_serialization(self):
        """Test prompt config serialization and deserialization."""
        config = HyDEPromptConfig(
            technical_keywords=["custom_api", "custom_function"],
            code_keywords=["custom_code"],
        )

        # Test to dict
        config_dict = config.model_dump()
        assert config_dict["technical_keywords"] == ["custom_api", "custom_function"]
        assert config_dict["code_keywords"] == ["custom_code"]

        # Test from dict
        restored_config = HyDEPromptConfig.model_validate(config_dict)
        assert restored_config.technical_keywords == ["custom_api", "custom_function"]
        assert restored_config.code_keywords == ["custom_code"]


class TestHyDEMetricsConfig:
    """Tests for HyDEMetricsConfig model."""

    def test_default_metrics_config(self):
        """Test default metrics configuration."""
        config = HyDEMetricsConfig()

        # Performance metrics
        assert config.track_generation_time is True
        assert config.track_cache_hits is True
        assert config.track_search_quality is True
        assert config.track_cost_savings is True

        # Quality metrics
        assert config.measure_diversity is True
        assert config.measure_relevance is True
        assert config.measure_coverage is True

        # A/B testing
        assert config.ab_testing_enabled is False
        assert config.control_group_percentage == 0.5

        # Reporting
        assert config.metrics_export_interval == 300
        assert config.detailed_logging is False

    def test_custom_metrics_config(self):
        """Test custom metrics configuration."""
        config = HyDEMetricsConfig(
            track_generation_time=False,
            track_cache_hits=False,
            track_search_quality=False,
            track_cost_savings=False,
            measure_diversity=False,
            measure_relevance=False,
            measure_coverage=False,
            ab_testing_enabled=True,
            control_group_percentage=0.3,
            metrics_export_interval=600,
            detailed_logging=True,
        )

        assert config.track_generation_time is False
        assert config.track_cache_hits is False
        assert config.track_search_quality is False
        assert config.track_cost_savings is False
        assert config.measure_diversity is False
        assert config.measure_relevance is False
        assert config.measure_coverage is False
        assert config.ab_testing_enabled is True
        assert config.control_group_percentage == 0.3
        assert config.metrics_export_interval == 600
        assert config.detailed_logging is True

    def test_validation_control_group_percentage(self):
        """Test validation for control_group_percentage field."""
        # Valid values
        HyDEMetricsConfig(control_group_percentage=0.0)
        HyDEMetricsConfig(control_group_percentage=0.5)
        HyDEMetricsConfig(control_group_percentage=1.0)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEMetricsConfig(control_group_percentage=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEMetricsConfig(control_group_percentage=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_validation_metrics_export_interval(self):
        """Test validation for metrics_export_interval field."""
        # Valid values
        HyDEMetricsConfig(metrics_export_interval=60)
        HyDEMetricsConfig(metrics_export_interval=300)
        HyDEMetricsConfig(metrics_export_interval=3600)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDEMetricsConfig(metrics_export_interval=59)
        assert "greater than or equal to 60" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEMetricsConfig(metrics_export_interval=3601)
        assert "less than or equal to 3600" in str(exc_info.value)

    def test_json_schema_extra(self):
        """Test that JSON schema extra example is accessible."""
        config = HyDEMetricsConfig()
        example = config.model_config.get("json_schema_extra", {}).get("example", {})

        assert example["track_generation_time"] is True
        assert example["track_cache_hits"] is True
        assert example["track_search_quality"] is True
        assert example["ab_testing_enabled"] is False
        assert example["control_group_percentage"] == 0.5

    def test_ab_testing_configuration(self):
        """Test A/B testing configuration combinations."""
        # A/B testing disabled
        config_disabled = HyDEMetricsConfig(ab_testing_enabled=False)
        assert config_disabled.ab_testing_enabled is False
        # control_group_percentage should still be valid even when disabled
        assert 0.0 <= config_disabled.control_group_percentage <= 1.0

        # A/B testing enabled with different control group sizes
        config_small_control = HyDEMetricsConfig(
            ab_testing_enabled=True, control_group_percentage=0.1
        )
        assert config_small_control.ab_testing_enabled is True
        assert config_small_control.control_group_percentage == 0.1

        config_large_control = HyDEMetricsConfig(
            ab_testing_enabled=True, control_group_percentage=0.9
        )
        assert config_large_control.ab_testing_enabled is True
        assert config_large_control.control_group_percentage == 0.9

    def test_metrics_tracking_combinations(self):
        """Test different combinations of metrics tracking."""
        # All metrics enabled
        config_all = HyDEMetricsConfig(
            track_generation_time=True,
            track_cache_hits=True,
            track_search_quality=True,
            track_cost_savings=True,
            measure_diversity=True,
            measure_relevance=True,
            measure_coverage=True,
        )

        assert config_all.track_generation_time is True
        assert config_all.track_cache_hits is True
        assert config_all.track_search_quality is True
        assert config_all.track_cost_savings is True
        assert config_all.measure_diversity is True
        assert config_all.measure_relevance is True
        assert config_all.measure_coverage is True

        # Minimal metrics
        config_minimal = HyDEMetricsConfig(
            track_generation_time=False,
            track_cache_hits=False,
            track_search_quality=False,
            track_cost_savings=False,
            measure_diversity=False,
            measure_relevance=False,
            measure_coverage=False,
        )

        assert config_minimal.track_generation_time is False
        assert config_minimal.track_cache_hits is False
        assert config_minimal.track_search_quality is False
        assert config_minimal.track_cost_savings is False
        assert config_minimal.measure_diversity is False
        assert config_minimal.measure_relevance is False
        assert config_minimal.measure_coverage is False

    def test_serialization(self):
        """Test metrics config serialization and deserialization."""
        config = HyDEMetricsConfig(
            ab_testing_enabled=True,
            control_group_percentage=0.3,
            metrics_export_interval=600,
            detailed_logging=True,
        )

        # Test to dict
        config_dict = config.model_dump()
        assert config_dict["ab_testing_enabled"] is True
        assert config_dict["control_group_percentage"] == 0.3
        assert config_dict["metrics_export_interval"] == 600
        assert config_dict["detailed_logging"] is True

        # Test from dict
        restored_config = HyDEMetricsConfig.model_validate(config_dict)
        assert restored_config.ab_testing_enabled is True
        assert restored_config.control_group_percentage == 0.3
        assert restored_config.metrics_export_interval == 600
        assert restored_config.detailed_logging is True


class TestHyDEConfigIntegration:
    """Integration tests for HyDE configuration classes."""

    def test_config_combination(self):
        """Test using all three configuration classes together."""
        hyde_config = HyDEConfig(
            enable_hyde=True,
            num_generations=3,
            generation_model="gpt-4",
            parallel_generation=True,
        )

        prompt_config = HyDEPromptConfig(
            technical_keywords=["api", "function", "method"],
            code_keywords=["python", "javascript", "implementation"],
        )

        metrics_config = HyDEMetricsConfig(
            track_generation_time=True,
            ab_testing_enabled=True,
            control_group_percentage=0.2,
        )

        # Verify configs are independent and work together
        assert hyde_config.enable_hyde is True
        assert hyde_config.num_generations == 3
        assert hyde_config.generation_model == "gpt-4"

        assert "api" in prompt_config.technical_keywords
        assert "python" in prompt_config.code_keywords

        assert metrics_config.track_generation_time is True
        assert metrics_config.ab_testing_enabled is True
        assert metrics_config.control_group_percentage == 0.2

    def test_config_serialization_together(self):
        """Test serialization of all config classes together."""
        configs = {
            "hyde": HyDEConfig(enable_hyde=True, num_generations=4),
            "prompt": HyDEPromptConfig(technical_keywords=["test"]),
            "metrics": HyDEMetricsConfig(ab_testing_enabled=True),
        }

        # Serialize all configs
        serialized = {name: config.model_dump() for name, config in configs.items()}

        # Verify serialized structure
        assert "hyde" in serialized
        assert "prompt" in serialized
        assert "metrics" in serialized

        assert serialized["hyde"]["enable_hyde"] is True
        assert serialized["hyde"]["num_generations"] == 4
        assert serialized["prompt"]["technical_keywords"] == ["test"]
        assert serialized["metrics"]["ab_testing_enabled"] is True

        # Deserialize back
        restored_configs = {
            "hyde": HyDEConfig.model_validate(serialized["hyde"]),
            "prompt": HyDEPromptConfig.model_validate(serialized["prompt"]),
            "metrics": HyDEMetricsConfig.model_validate(serialized["metrics"]),
        }

        assert restored_configs["hyde"].enable_hyde is True
        assert restored_configs["hyde"].num_generations == 4
        assert restored_configs["prompt"].technical_keywords == ["test"]
        assert restored_configs["metrics"].ab_testing_enabled is True
