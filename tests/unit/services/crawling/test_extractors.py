"""Tests for crawling extractors module."""

from src.services.crawling.extractors import DocumentationExtractor
from src.services.crawling.extractors import JavaScriptExecutor


class TestJavaScriptExecutor:
    """Test cases for JavaScriptExecutor class."""

    def test_init_creates_common_patterns(self):
        """Test JavaScript executor initializes with common patterns."""
        executor = JavaScriptExecutor()

        assert "spa_navigation" in executor.common_patterns
        assert "infinite_scroll" in executor.common_patterns
        assert "click_show_more" in executor.common_patterns

        # Verify patterns are non-empty strings
        for pattern_code in executor.common_patterns.values():
            assert isinstance(pattern_code, str)
            assert len(pattern_code.strip()) > 0

    def test_get_js_for_site_python_docs(self):
        """Test JavaScript retrieval for Python documentation."""
        executor = JavaScriptExecutor()

        js_code = executor.get_js_for_site(
            "https://docs.python.org/3/library/asyncio.html"
        )

        assert js_code is not None
        assert js_code == executor.common_patterns["spa_navigation"]

    def test_get_js_for_site_react_docs(self):
        """Test JavaScript retrieval for React documentation."""
        executor = JavaScriptExecutor()

        # Test both reactjs.org and react.dev
        js_code_old = executor.get_js_for_site(
            "https://reactjs.org/docs/hooks-intro.html"
        )
        js_code_new = executor.get_js_for_site("https://react.dev/learn")

        assert js_code_old is not None
        assert js_code_new is not None
        assert js_code_old == executor.common_patterns["spa_navigation"]
        assert js_code_new == executor.common_patterns["spa_navigation"]

    def test_get_js_for_site_mdn_docs(self):
        """Test JavaScript retrieval for MDN documentation."""
        executor = JavaScriptExecutor()

        js_code = executor.get_js_for_site(
            "https://developer.mozilla.org/en-US/docs/Web/API"
        )

        assert js_code is not None
        assert js_code == executor.common_patterns["click_show_more"]

    def test_get_js_for_site_stackoverflow(self):
        """Test JavaScript retrieval for Stack Overflow."""
        executor = JavaScriptExecutor()

        js_code = executor.get_js_for_site(
            "https://stackoverflow.com/questions/123456/example"
        )

        assert js_code is not None
        assert js_code == executor.common_patterns["infinite_scroll"]

    def test_get_js_for_site_unknown_domain(self):
        """Test JavaScript retrieval for unknown domain returns None."""
        executor = JavaScriptExecutor()

        js_code = executor.get_js_for_site("https://unknown-domain.com/page")

        assert js_code is None

    def test_get_js_for_site_with_subdomain(self):
        """Test JavaScript retrieval works with subdomains."""
        executor = JavaScriptExecutor()

        # Should not match subdomain of known domain
        js_code = executor.get_js_for_site("https://subdomain.docs.python.org/page")

        assert js_code is None


class TestDocumentationExtractor:
    """Test cases for DocumentationExtractor class."""

    def test_init_creates_selectors(self):
        """Test documentation extractor initializes with selectors."""
        extractor = DocumentationExtractor()

        assert "content" in extractor.selectors
        assert "code" in extractor.selectors
        assert "nav" in extractor.selectors
        assert "metadata" in extractor.selectors

        # Verify content selectors are lists
        assert isinstance(extractor.selectors["content"], list)
        assert len(extractor.selectors["content"]) > 0

        # Verify code selectors are lists
        assert isinstance(extractor.selectors["code"], list)
        assert len(extractor.selectors["code"]) > 0

    def test_create_extraction_schema_general(self):
        """Test creating extraction schema for general documentation."""
        extractor = DocumentationExtractor()

        schema = extractor.create_extraction_schema("general")

        # Should contain base schema elements
        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema

        # Should match selector definitions
        assert schema["title"] == extractor.selectors["metadata"]["title"]
        assert schema["content"] == extractor.selectors["content"]
        assert schema["code_blocks"] == extractor.selectors["code"]

    def test_create_extraction_schema_api_reference(self):
        """Test creating extraction schema for API reference documentation."""
        extractor = DocumentationExtractor()

        schema = extractor.create_extraction_schema("api_reference")

        # Should contain base schema elements
        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema

        # Should contain API-specific elements
        assert "endpoints" in schema
        assert "parameters" in schema
        assert "responses" in schema
        assert "examples" in schema

        # Verify specific selectors
        assert schema["endpoints"] == "section.endpoint"
        assert schema["parameters"] == ".parameter"

    def test_create_extraction_schema_tutorial(self):
        """Test creating extraction schema for tutorial documentation."""
        extractor = DocumentationExtractor()

        schema = extractor.create_extraction_schema("tutorial")

        # Should contain base schema elements
        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema

        # Should contain tutorial-specific elements
        assert "steps" in schema
        assert "code_examples" in schema
        assert "prerequisites" in schema
        assert "objectives" in schema

        # Verify specific selectors
        assert schema["steps"] == ".step, .tutorial-step"
        assert schema["code_examples"] == "pre code"

    def test_create_extraction_schema_guide(self):
        """Test creating extraction schema for guide documentation."""
        extractor = DocumentationExtractor()

        schema = extractor.create_extraction_schema("guide")

        # Should contain base schema elements
        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema

        # Should contain guide-specific elements
        assert "sections" in schema
        assert "content" in schema  # Should be overridden with guide-specific content
        assert "callouts" in schema
        assert "related" in schema

        # Verify specific selectors
        assert schema["sections"] == "h2, h3"
        assert schema["callouts"] == ".note, .warning, .tip"

    def test_create_extraction_schema_unknown_type(self):
        """Test creating extraction schema for unknown documentation type."""
        extractor = DocumentationExtractor()

        schema = extractor.create_extraction_schema("unknown_type")

        # Should only contain base schema elements
        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema

        # Should not contain any type-specific elements
        assert "endpoints" not in schema
        assert "steps" not in schema
        assert "sections" not in schema

    def test_metadata_selectors_structure(self):
        """Test metadata selectors have correct structure."""
        extractor = DocumentationExtractor()

        metadata = extractor.selectors["metadata"]

        # Verify required metadata fields exist
        required_fields = ["title", "description", "author", "version", "last_updated"]
        for field in required_fields:
            assert field in metadata
            assert isinstance(metadata[field], list)
            assert len(metadata[field]) > 0

    def test_content_selectors_comprehensive(self):
        """Test content selectors cover common documentation patterns."""
        extractor = DocumentationExtractor()

        content_selectors = extractor.selectors["content"]

        # Verify common documentation selectors are included
        expected_selectors = ["main", "article", ".content", ".documentation"]
        for selector in expected_selectors:
            assert selector in content_selectors

    def test_code_selectors_comprehensive(self):
        """Test code selectors cover common code block patterns."""
        extractor = DocumentationExtractor()

        code_selectors = extractor.selectors["code"]

        # Verify common code block selectors are included
        expected_selectors = ["pre code", ".highlight", ".code-block"]
        for selector in expected_selectors:
            assert selector in code_selectors


class TestExtractorsIntegration:
    """Integration tests for extractors working together."""

    def test_extractors_can_be_instantiated_together(self):
        """Test that both extractors can be created and used together."""
        js_executor = JavaScriptExecutor()
        doc_extractor = DocumentationExtractor()

        # Should be able to use both
        js_code = js_executor.get_js_for_site("https://docs.python.org")
        schema = doc_extractor.create_extraction_schema("api_reference")

        assert js_code is not None
        assert schema is not None
        assert len(schema) > 0

    def test_extractors_are_independent(self):
        """Test that extractors don't interfere with each other."""
        js_executor1 = JavaScriptExecutor()
        js_executor2 = JavaScriptExecutor()
        doc_extractor1 = DocumentationExtractor()
        doc_extractor2 = DocumentationExtractor()

        # Should be independent instances
        assert js_executor1 is not js_executor2
        assert doc_extractor1 is not doc_extractor2

        # Should have same behavior
        assert js_executor1.get_js_for_site(
            "https://docs.python.org"
        ) == js_executor2.get_js_for_site("https://docs.python.org")
        assert doc_extractor1.create_extraction_schema(
            "tutorial"
        ) == doc_extractor2.create_extraction_schema("tutorial")
