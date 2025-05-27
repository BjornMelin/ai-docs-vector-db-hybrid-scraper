#!/usr/bin/env python3
"""
Unit tests for HyDE configuration models.

Tests configuration validation, defaults, and edge cases for all HyDE config classes.
"""

import pytest
from pydantic import ValidationError
from src.services.hyde.config import HyDEConfig
from src.services.hyde.config import HyDEMetricsConfig
from src.services.hyde.config import HyDEPromptConfig


class TestHyDEConfig:
    """Test cases for HyDEConfig validation and defaults."""

    def test_default_config(self):
        """Test that default configuration is valid."""
        config = HyDEConfig()

        assert config.enable_hyde is True
        assert config.num_generations == 5
        assert config.generation_temperature == 0.7
        assert config.max_generation_tokens == 200
        assert config.parallel_generation is True
        assert config.generation_timeout_seconds == 10
        assert config.enable_fallback is True
        assert config.cache_hypothetical_docs is True
        assert config.enable_reranking is True
        assert config.enable_caching is True

    def test_valid_custom_config(self):
        """Test valid custom configuration values."""
        config = HyDEConfig(
            enable_hyde=False,
            num_generations=3,
            generation_temperature=0.5,
            max_generation_tokens=150,
            parallel_generation=False,
            generation_timeout_seconds=30,
            cache_hypothetical_docs=False,
            enable_fallback=False,
            enable_reranking=False,
            enable_caching=False,
        )

        assert config.enable_hyde is False
        assert config.num_generations == 3
        assert config.generation_temperature == 0.5
        assert config.max_generation_tokens == 150
        assert config.parallel_generation is False
        assert config.generation_timeout_seconds == 30
        assert config.cache_hypothetical_docs is False
        assert config.enable_fallback is False
        assert config.enable_reranking is False
        assert config.enable_caching is False

    def test_invalid_num_generations(self):
        """Test validation error for invalid num_generations."""
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(num_generations=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(num_generations=11)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_invalid_temperature(self):
        """Test validation error for invalid temperature."""
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_temperature=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_temperature=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_invalid_max_tokens(self):
        """Test validation error for invalid max_generation_tokens."""
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_generation_tokens=49)
        assert "greater than or equal to 50" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(max_generation_tokens=501)
        assert "less than or equal to 500" in str(exc_info.value)

    def test_invalid_timeout(self):
        """Test validation error for invalid timeout."""
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_timeout_seconds=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(generation_timeout_seconds=61)
        assert "less than or equal to 60" in str(exc_info.value)

    def test_invalid_cache_ttl(self):
        """Test validation error for invalid cache TTL."""
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(cache_ttl_seconds=299)
        assert "greater than or equal to 300" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(cache_ttl_seconds=86401)
        assert "less than or equal to 86400" in str(exc_info.value)

    def test_invalid_hyde_weight(self):
        """Test validation error for invalid HyDE weight in fusion."""
        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(hyde_weight_in_fusion=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEConfig(hyde_weight_in_fusion=1.1)
        assert "less than or equal to 1" in str(exc_info.value)


class TestHyDEPromptConfig:
    """Test cases for HyDEPromptConfig validation and functionality."""

    def test_default_prompt_config(self):
        """Test that default prompt configuration is valid."""
        config = HyDEPromptConfig()

        assert "technical documentation" in config.technical_prompt
        assert "code documentation" in config.code_prompt
        assert "tutorial" in config.tutorial_prompt
        assert "Answer" in config.general_prompt
        assert len(config.technical_keywords) > 0
        assert len(config.code_keywords) > 0
        assert len(config.tutorial_keywords) > 0
        assert "api" in config.technical_keywords
        assert "code" in config.code_keywords
        assert "tutorial" in config.tutorial_keywords

    def test_keyword_lists(self):
        """Test keyword classification lists."""
        config = HyDEPromptConfig()

        # Technical keywords
        assert "api" in config.technical_keywords
        assert "function" in config.technical_keywords
        assert "error" in config.technical_keywords

        # Code keywords
        assert "how to" in config.code_keywords
        assert "example" in config.code_keywords
        assert "python" in config.code_keywords

        # Tutorial keywords
        assert "tutorial" in config.tutorial_keywords
        assert "guide" in config.tutorial_keywords
        assert "learn" in config.tutorial_keywords

    def test_prompt_templates(self):
        """Test prompt template formatting."""
        config = HyDEPromptConfig()

        # Test that prompts have placeholder for query
        assert "{query}" in config.technical_prompt
        assert "{query}" in config.code_prompt
        assert "{query}" in config.tutorial_prompt
        assert "{query}" in config.general_prompt

    def test_variation_templates(self):
        """Test prompt variation templates."""
        config = HyDEPromptConfig()

        variations = config.variation_templates
        assert "prefixes" in variations
        assert "instruction_styles" in variations
        assert "context_additions" in variations

        assert len(variations["prefixes"]) > 0
        assert len(variations["instruction_styles"]) > 0
        assert len(variations["context_additions"]) > 0

        # Test that variations have domain placeholder
        assert any("{domain}" in prefix for prefix in variations["prefixes"])


class TestHyDEMetricsConfig:
    """Test cases for HyDEMetricsConfig validation and functionality."""

    def test_default_metrics_config(self):
        """Test that default metrics configuration is valid."""
        config = HyDEMetricsConfig()

        assert config.track_generation_time is True
        assert config.track_cache_hits is True
        assert config.track_search_quality is True
        assert config.track_cost_savings is True
        assert config.ab_testing_enabled is False
        assert config.control_group_percentage == 0.5
        assert config.measure_diversity is True
        assert config.measure_relevance is True
        assert config.measure_coverage is True

    def test_valid_custom_metrics_config(self):
        """Test valid custom metrics configuration."""
        config = HyDEMetricsConfig(
            track_generation_time=False,
            track_cache_hits=False,
            track_search_quality=False,
            track_cost_savings=False,
            ab_testing_enabled=True,
            control_group_percentage=0.3,
            measure_diversity=False,
            detailed_logging=True,
        )

        assert config.track_generation_time is False
        assert config.track_cache_hits is False
        assert config.track_search_quality is False
        assert config.track_cost_savings is False
        assert config.ab_testing_enabled is True
        assert config.control_group_percentage == 0.3
        assert config.measure_diversity is False
        assert config.detailed_logging is True

    def test_invalid_control_group_percentage(self):
        """Test validation error for invalid control_group_percentage."""
        with pytest.raises(ValidationError) as exc_info:
            HyDEMetricsConfig(control_group_percentage=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEMetricsConfig(control_group_percentage=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_invalid_metrics_export_interval(self):
        """Test validation error for invalid metrics export interval."""
        with pytest.raises(ValidationError) as exc_info:
            HyDEMetricsConfig(metrics_export_interval=59)
        assert "greater than or equal to 60" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            HyDEMetricsConfig(metrics_export_interval=3601)
        assert "less than or equal to 3600" in str(exc_info.value)

    def test_metrics_tracking_flags(self):
        """Test all metrics tracking flags."""
        config = HyDEMetricsConfig()

        # Performance metrics
        assert isinstance(config.track_generation_time, bool)
        assert isinstance(config.track_cache_hits, bool)
        assert isinstance(config.track_search_quality, bool)
        assert isinstance(config.track_cost_savings, bool)

        # Quality metrics
        assert isinstance(config.measure_diversity, bool)
        assert isinstance(config.measure_relevance, bool)
        assert isinstance(config.measure_coverage, bool)

        # A/B testing
        assert isinstance(config.ab_testing_enabled, bool)

        # Reporting
        assert isinstance(config.detailed_logging, bool)
