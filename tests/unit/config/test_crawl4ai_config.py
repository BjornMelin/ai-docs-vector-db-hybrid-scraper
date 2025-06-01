"""Test Crawl4AIConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import Crawl4AIConfig


class TestCrawl4AIConfig:
    """Test Crawl4AIConfig model validation and behavior."""

    def test_default_values(self):
        """Test Crawl4AIConfig with default values."""
        config = Crawl4AIConfig()

        # Browser settings
        assert config.browser_type == "chromium"
        assert config.headless is True
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080

        # Performance settings
        assert config.max_concurrent_crawls == 10
        assert config.page_timeout == 30.0
        assert config.wait_for_selector is None

        # Content extraction
        assert config.remove_scripts is True
        assert config.remove_styles is True
        assert config.extract_links is True

    def test_custom_values(self):
        """Test Crawl4AIConfig with custom values."""
        config = Crawl4AIConfig(
            browser_type="firefox",
            headless=False,
            viewport_width=1366,
            viewport_height=768,
            max_concurrent_crawls=20,
            page_timeout=60.0,
            wait_for_selector=".content-loaded",
            remove_scripts=False,
            remove_styles=False,
            extract_links=False,
        )

        assert config.browser_type == "firefox"
        assert config.headless is False
        assert config.viewport_width == 1366
        assert config.viewport_height == 768
        assert config.max_concurrent_crawls == 20
        assert config.page_timeout == 60.0
        assert config.wait_for_selector == ".content-loaded"
        assert config.remove_scripts is False
        assert config.remove_styles is False
        assert config.extract_links is False

    def test_browser_type_values(self):
        """Test browser type values."""
        # Valid browser types
        valid_browsers = ["chromium", "firefox", "webkit"]
        for browser in valid_browsers:
            config = Crawl4AIConfig(browser_type=browser)
            assert config.browser_type == browser

        # Note: browser_type is a string field without enum validation
        # so any string is accepted
        config = Crawl4AIConfig(browser_type="edge")
        assert config.browser_type == "edge"

    def test_viewport_constraints(self):
        """Test viewport dimension constraints."""
        # Valid viewport dimensions
        config1 = Crawl4AIConfig(viewport_width=800, viewport_height=600)
        assert config1.viewport_width == 800
        assert config1.viewport_height == 600

        # Large viewport
        config2 = Crawl4AIConfig(viewport_width=3840, viewport_height=2160)
        assert config2.viewport_width == 3840
        assert config2.viewport_height == 2160

        # Invalid: zero width
        with pytest.raises(ValidationError) as exc_info:
            Crawl4AIConfig(viewport_width=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("viewport_width",)
        assert "greater than 0" in str(errors[0]["msg"])

        # Invalid: negative height
        with pytest.raises(ValidationError) as exc_info:
            Crawl4AIConfig(viewport_height=-100)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("viewport_height",)

    def test_max_concurrent_crawls_constraints(self):
        """Test max concurrent crawls constraints (0 < value <= 50)."""
        # Valid values
        config1 = Crawl4AIConfig(max_concurrent_crawls=1)
        assert config1.max_concurrent_crawls == 1

        config2 = Crawl4AIConfig(max_concurrent_crawls=50)
        assert config2.max_concurrent_crawls == 50

        # Invalid: zero
        with pytest.raises(ValidationError) as exc_info:
            Crawl4AIConfig(max_concurrent_crawls=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_concurrent_crawls",)
        assert "greater than 0" in str(errors[0]["msg"])

        # Invalid: exceeds maximum
        with pytest.raises(ValidationError) as exc_info:
            Crawl4AIConfig(max_concurrent_crawls=51)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_concurrent_crawls",)
        assert "less than or equal to 50" in str(errors[0]["msg"])

    def test_page_timeout_constraints(self):
        """Test page timeout must be positive."""
        # Valid timeout
        config = Crawl4AIConfig(page_timeout=120.0)
        assert config.page_timeout == 120.0

        # Invalid: zero timeout
        with pytest.raises(ValidationError) as exc_info:
            Crawl4AIConfig(page_timeout=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("page_timeout",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_wait_for_selector_optional(self):
        """Test wait_for_selector is optional."""
        # None by default
        config1 = Crawl4AIConfig()
        assert config1.wait_for_selector is None

        # Can be set to a CSS selector
        config2 = Crawl4AIConfig(wait_for_selector="#main-content")
        assert config2.wait_for_selector == "#main-content"

        # Can be complex selector
        config3 = Crawl4AIConfig(wait_for_selector="div.content[data-loaded='true']")
        assert config3.wait_for_selector == "div.content[data-loaded='true']"

    def test_boolean_fields(self):
        """Test boolean field validation."""
        # All true
        config1 = Crawl4AIConfig(
            headless=True, remove_scripts=True, remove_styles=True, extract_links=True
        )
        assert all(
            [
                config1.headless,
                config1.remove_scripts,
                config1.remove_styles,
                config1.extract_links,
            ]
        )

        # All false
        config2 = Crawl4AIConfig(
            headless=False,
            remove_scripts=False,
            remove_styles=False,
            extract_links=False,
        )
        assert not any(
            [
                config2.headless,
                config2.remove_scripts,
                config2.remove_styles,
                config2.extract_links,
            ]
        )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            Crawl4AIConfig(browser_type="chromium", unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = Crawl4AIConfig(
            browser_type="webkit",
            headless=False,
            viewport_width=1440,
            viewport_height=900,
            max_concurrent_crawls=15,
            page_timeout=45.0,
            wait_for_selector=".ready",
            remove_scripts=False,
            extract_links=False,
        )

        # Test model_dump
        data = config.model_dump()
        assert data["browser_type"] == "webkit"
        assert data["headless"] is False
        assert data["viewport_width"] == 1440
        assert data["viewport_height"] == 900
        assert data["max_concurrent_crawls"] == 15
        assert data["page_timeout"] == 45.0
        assert data["wait_for_selector"] == ".ready"
        assert data["remove_scripts"] is False
        assert data["extract_links"] is False

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"browser_type":"webkit"' in json_str
        assert '"headless":false' in json_str
        assert '"viewport_width":1440' in json_str
        assert '"page_timeout":45.0' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = Crawl4AIConfig(browser_type="chromium", max_concurrent_crawls=10)

        updated = original.model_copy(
            update={
                "browser_type": "firefox",
                "max_concurrent_crawls": 25,
                "headless": False,
            }
        )

        assert original.browser_type == "chromium"
        assert original.max_concurrent_crawls == 10
        assert original.headless is True  # Default
        assert updated.browser_type == "firefox"
        assert updated.max_concurrent_crawls == 25
        assert updated.headless is False

    def test_type_validation(self):
        """Test type validation for fields."""
        # Test string field with wrong type
        with pytest.raises(ValidationError) as exc_info:
            Crawl4AIConfig(browser_type=123)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("browser_type",)

        # Test boolean field with wrong type
        with pytest.raises(ValidationError) as exc_info:
            Crawl4AIConfig(headless={"value": True})  # Dict can't coerce to bool

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("headless",)

        # Test float field with wrong type (string that can't convert)
        with pytest.raises(ValidationError) as exc_info:
            Crawl4AIConfig(page_timeout="thirty seconds")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("page_timeout",)

    def test_performance_configuration_scenario(self):
        """Test a performance-optimized configuration scenario."""
        config = Crawl4AIConfig(
            browser_type="chromium",
            headless=True,  # Faster without UI
            viewport_width=1280,  # Smaller viewport for speed
            viewport_height=720,
            max_concurrent_crawls=50,  # Maximum allowed
            page_timeout=15.0,  # Shorter timeout
            wait_for_selector=None,  # Don't wait
            remove_scripts=True,  # Reduce content size
            remove_styles=True,  # Reduce content size
            extract_links=False,  # Skip if not needed
        )

        assert config.max_concurrent_crawls == 50
        assert config.page_timeout == 15.0
        assert config.remove_scripts is True
        assert config.remove_styles is True
