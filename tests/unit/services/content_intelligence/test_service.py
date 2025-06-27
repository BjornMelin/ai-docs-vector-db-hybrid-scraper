"""Simple tests for content intelligence service."""

from src.services.content_intelligence.service import ContentIntelligenceService


class TestContentIntelligenceService:
    """Simple tests for content intelligence service."""

    def test_service_class_exists(self):
        """Test that ContentIntelligenceService class can be imported."""
        assert ContentIntelligenceService is not None
        assert ContentIntelligenceService.__name__ == "ContentIntelligenceService"

    def test_service_is_callable_class(self):
        """Test that service class is properly defined."""
        assert callable(ContentIntelligenceService)
        assert hasattr(ContentIntelligenceService, "__init__")

    def test_service_module_structure(self):
        """Test that service module imports correctly."""
        # Just verify we can import the class without errors
        from src.services.content_intelligence.service import (  # noqa: PLC0415
            ContentIntelligenceService as CIService,
        )

        assert CIService is ContentIntelligenceService
