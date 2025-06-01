"""Test DocumentationSite Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import DocumentationSite


class TestDocumentationSite:
    """Test DocumentationSite model validation and behavior."""

    def test_minimal_required_fields(self):
        """Test DocumentationSite with only required fields."""
        site = DocumentationSite(name="Python Docs", url="https://docs.python.org")

        assert site.name == "Python Docs"
        assert (
            str(site.url) == "https://docs.python.org/"
        )  # HttpUrl adds trailing slash
        assert site.max_pages == 50  # Default
        assert site.priority == "medium"  # Default
        assert site.description is None
        assert site.max_depth == 2  # Default
        assert site.crawl_pattern is None
        assert site.exclude_patterns == []
        assert site.url_patterns == [
            "*docs*",
            "*guide*",
            "*tutorial*",
            "*api*",
            "*reference*",
            "*concepts*",
        ]

    def test_all_fields(self):
        """Test DocumentationSite with all fields specified."""
        site = DocumentationSite(
            name="FastAPI Documentation",
            url="https://fastapi.tiangolo.com",
            max_pages=100,
            priority="high",
            description="FastAPI framework documentation",
            max_depth=3,
            crawl_pattern="https://fastapi.tiangolo.com/[^/]+/$",
            exclude_patterns=["*/release-notes/*", "*/contributing/*"],
            url_patterns=["*tutorial*", "*advanced*", "*deployment*"],
        )

        assert site.name == "FastAPI Documentation"
        assert str(site.url) == "https://fastapi.tiangolo.com/"
        assert site.max_pages == 100
        assert site.priority == "high"
        assert site.description == "FastAPI framework documentation"
        assert site.max_depth == 3
        assert site.crawl_pattern == "https://fastapi.tiangolo.com/[^/]+/$"
        assert site.exclude_patterns == ["*/release-notes/*", "*/contributing/*"]
        assert site.url_patterns == ["*tutorial*", "*advanced*", "*deployment*"]

    def test_url_validation(self):
        """Test URL validation with HttpUrl."""
        # Valid URLs
        valid_urls = [
            "https://docs.python.org",
            "http://localhost:8000",
            "https://api.example.com:443/docs",
            "https://docs.example.com/v2/",
            "https://user:pass@docs.example.com",
        ]

        for url in valid_urls:
            site = DocumentationSite(name="Test", url=url)
            assert site.url is not None

        # Invalid URLs
        invalid_urls = [
            "not-a-url",
            "ftp://files.example.com",  # Wrong protocol
            "docs.example.com",  # Missing protocol
            "https://",  # Incomplete
            "",  # Empty
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError) as exc_info:
                DocumentationSite(name="Test", url=url)
            errors = exc_info.value.errors()
            assert any("url" in str(e["loc"]) for e in errors)

    def test_max_pages_constraint(self):
        """Test max_pages must be positive."""
        # Valid values
        site1 = DocumentationSite(name="Test", url="https://example.com", max_pages=1)
        assert site1.max_pages == 1

        site2 = DocumentationSite(
            name="Test", url="https://example.com", max_pages=10000
        )
        assert site2.max_pages == 10000

        # Invalid: zero
        with pytest.raises(ValidationError) as exc_info:
            DocumentationSite(name="Test", url="https://example.com", max_pages=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_pages",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_max_depth_constraint(self):
        """Test max_depth must be positive."""
        # Valid values
        site1 = DocumentationSite(name="Test", url="https://example.com", max_depth=1)
        assert site1.max_depth == 1

        site2 = DocumentationSite(name="Test", url="https://example.com", max_depth=10)
        assert site2.max_depth == 10

        # Invalid: zero
        with pytest.raises(ValidationError) as exc_info:
            DocumentationSite(name="Test", url="https://example.com", max_depth=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_depth",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_priority_values(self):
        """Test priority field values."""
        # Common priority values
        priorities = ["low", "medium", "high", "critical", "urgent"]

        for priority in priorities:
            site = DocumentationSite(
                name="Test", url="https://example.com", priority=priority
            )
            assert site.priority == priority

        # Numeric priorities
        site1 = DocumentationSite(name="Test", url="https://example.com", priority="1")
        assert site1.priority == "1"

        # Custom priority
        site2 = DocumentationSite(
            name="Test", url="https://example.com", priority="important-docs"
        )
        assert site2.priority == "important-docs"

    def test_pattern_lists(self):
        """Test exclude_patterns and url_patterns lists."""
        # Empty exclude patterns
        site1 = DocumentationSite(
            name="Test", url="https://example.com", exclude_patterns=[]
        )
        assert site1.exclude_patterns == []

        # Multiple exclude patterns
        site2 = DocumentationSite(
            name="Test",
            url="https://example.com",
            exclude_patterns=["*/admin/*", "*/private/*", "*.pdf", "*/old-docs/*"],
        )
        assert len(site2.exclude_patterns) == 4
        assert "*.pdf" in site2.exclude_patterns

        # Custom URL patterns
        site3 = DocumentationSite(
            name="Test",
            url="https://example.com",
            url_patterns=["*manual*", "*howto*", "*faq*"],
        )
        assert len(site3.url_patterns) == 3
        assert "*manual*" in site3.url_patterns

    def test_crawl_pattern_regex(self):
        """Test crawl_pattern field with regex patterns."""
        # Simple pattern
        site1 = DocumentationSite(
            name="Test",
            url="https://example.com",
            crawl_pattern="https://example.com/docs/.*",
        )
        assert site1.crawl_pattern == "https://example.com/docs/.*"

        # Complex pattern
        site2 = DocumentationSite(
            name="Test",
            url="https://example.com",
            crawl_pattern=r"^https://example\.com/(docs|api|guide)/[a-z0-9-]+/?$",
        )
        assert (
            site2.crawl_pattern
            == r"^https://example\.com/(docs|api|guide)/[a-z0-9-]+/?$"
        )

        # None is valid
        site3 = DocumentationSite(
            name="Test", url="https://example.com", crawl_pattern=None
        )
        assert site3.crawl_pattern is None

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentationSite(
                name="Test", url="https://example.com", unknown_field="value"
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        site = DocumentationSite(
            name="Django Docs",
            url="https://docs.djangoproject.com",
            max_pages=200,
            priority="high",
            description="Django web framework documentation",
            exclude_patterns=["*/dev/*", "*/1.x/*"],
        )

        # Test model_dump
        data = site.model_dump()
        assert data["name"] == "Django Docs"
        assert str(data["url"]) == "https://docs.djangoproject.com/"
        assert data["max_pages"] == 200
        assert data["priority"] == "high"
        assert data["description"] == "Django web framework documentation"
        assert data["exclude_patterns"] == ["*/dev/*", "*/1.x/*"]

        # Test model_dump_json
        json_str = site.model_dump_json()
        assert '"name":"Django Docs"' in json_str
        assert '"max_pages":200' in json_str
        assert '"priority":"high"' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = DocumentationSite(
            name="Original Docs", url="https://original.com", max_pages=50
        )

        updated = original.model_copy(
            update={"name": "Updated Docs", "max_pages": 100, "priority": "high"}
        )

        assert original.name == "Original Docs"
        assert original.max_pages == 50
        assert original.priority == "medium"  # Default
        assert updated.name == "Updated Docs"
        assert updated.max_pages == 100
        assert updated.priority == "high"
        assert str(updated.url) == str(original.url)  # URL unchanged

    def test_typical_documentation_sites(self):
        """Test typical documentation site configurations."""
        # API documentation
        api_docs = DocumentationSite(
            name="REST API Docs",
            url="https://api.service.com/docs",
            max_pages=150,
            priority="high",
            description="REST API reference documentation",
            max_depth=3,
            url_patterns=["*endpoints*", "*schemas*", "*authentication*"],
            exclude_patterns=["*/deprecated/*", "*/internal/*"],
        )
        assert api_docs.priority == "high"
        assert api_docs.max_depth == 3

        # Tutorial site
        tutorial_site = DocumentationSite(
            name="Framework Tutorials",
            url="https://learn.framework.com",
            max_pages=75,
            priority="medium",
            max_depth=2,
            url_patterns=["*getting-started*", "*tutorial*", "*example*"],
            exclude_patterns=["*/advanced/*"],  # Skip advanced topics
        )
        assert tutorial_site.max_pages == 75
        assert "*/advanced/*" in tutorial_site.exclude_patterns

        # Large documentation portal
        large_portal = DocumentationSite(
            name="Enterprise Docs Portal",
            url="https://docs.enterprise.com",
            max_pages=500,
            priority="critical",
            max_depth=5,
            crawl_pattern="https://docs.enterprise.com/v3/.*",
            url_patterns=["*"],  # Crawl everything matching the pattern
            exclude_patterns=["*/archive/*", "*/legacy/*", "*.pdf", "*.zip"],
        )
        assert large_portal.max_pages == 500
        assert large_portal.max_depth == 5
        assert "*.pdf" in large_portal.exclude_patterns

    def test_url_normalization(self):
        """Test that HttpUrl normalizes URLs."""
        # URL without trailing slash gets one added
        site1 = DocumentationSite(name="Test", url="https://example.com")
        assert str(site1.url) == "https://example.com/"

        # URL with trailing slash keeps it
        site2 = DocumentationSite(name="Test", url="https://example.com/")
        assert str(site2.url) == "https://example.com/"

        # URL with path
        site3 = DocumentationSite(name="Test", url="https://example.com/docs")
        assert str(site3.url) == "https://example.com/docs"
