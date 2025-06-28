"""Tests for crawling base module."""

import inspect
from abc import ABC
from typing import Any

import pytest

from src.services.crawling.base import CrawlProvider


class TestCrawlProvider:
    """Test the CrawlProvider abstract base class."""

    def test_crawl_provider_is_abstract(self):
        """Test that CrawlProvider is an abstract base class."""
        assert issubclass(CrawlProvider, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CrawlProvider()

    def test_abstract_methods_exist(self):
        """Test that required abstract methods are defined."""
        abstract_methods = CrawlProvider.__abstractmethods__
        expected_methods = {"scrape_url", "crawl_site", "initialize", "cleanup"}
        assert abstract_methods == expected_methods

    def test_scrape_url_signature(self):
        """Test scrape_url method signature."""
        method = CrawlProvider.scrape_url
        assert hasattr(method, "__annotations__")
        annotations = method.__annotations__

        # Check parameter types
        assert "url" in annotations
        assert annotations["url"] is str
        assert "formats" in annotations
        assert annotations["formats"] == list[str] | None
        assert annotations["return"] == dict[str, Any]

    def test_crawl_site_signature(self):
        """Test crawl_site method signature."""
        method = CrawlProvider.crawl_site
        assert hasattr(method, "__annotations__")
        annotations = method.__annotations__

        # Check parameter types
        assert "url" in annotations
        assert annotations["url"] is str
        assert "max_pages" in annotations
        assert annotations["max_pages"] is int
        assert "formats" in annotations
        assert annotations["formats"] == list[str] | None
        assert annotations["return"] == dict[str, Any]

    def test_initialize_signature(self):
        """Test initialize method signature."""
        method = CrawlProvider.initialize
        assert hasattr(method, "__annotations__")
        annotations = method.__annotations__

        assert annotations["return"] is None

    def test_cleanup_signature(self):
        """Test cleanup method signature."""
        method = CrawlProvider.cleanup
        assert hasattr(method, "__annotations__")
        annotations = method.__annotations__

        assert annotations["return"] is None

    def test_docstrings_exist(self):
        """Test that methods have proper docstrings."""
        assert CrawlProvider.scrape_url.__doc__ is not None
        assert "Scrape a single URL" in CrawlProvider.scrape_url.__doc__

        assert CrawlProvider.crawl_site.__doc__ is not None
        assert "Crawl an entire site" in CrawlProvider.crawl_site.__doc__

        assert CrawlProvider.initialize.__doc__ is not None
        assert "Initialize the provider" in CrawlProvider.initialize.__doc__

        assert CrawlProvider.cleanup.__doc__ is not None
        assert "Cleanup provider resources" in CrawlProvider.cleanup.__doc__

    def test_concrete_implementation_valid(self):
        """Test that a concrete implementation can be created."""

        class ConcreteCrawlProvider(CrawlProvider):
            """Concrete implementation for testing."""

            async def scrape_url(
                self, url: str, _formats: list[str] | None = None
            ) -> dict[str, Any]:
                return {"success": True, "url": url, "content": "test"}

            async def crawl_site(
                self,
                _url: str,
                _max_pages: int = 50,
                _formats: list[str] | None = None,
            ) -> dict[str, Any]:
                return {"success": True, "pages": [], "total": 0}

            async def initialize(self) -> None:
                pass

            async def cleanup(self) -> None:
                pass

        # Should be able to instantiate concrete implementation
        provider = ConcreteCrawlProvider()
        assert isinstance(provider, CrawlProvider)

    @pytest.mark.asyncio
    async def test_concrete_implementation_methods(self):
        """Test that concrete implementation methods work as expected."""

        class TestCrawlProvider(CrawlProvider):
            """Test implementation with mock behavior."""

            def __init__(self):
                self.initialized = False
                self.cleanup_called = False

            async def scrape_url(
                self, url: str, formats: list[str] | None = None
            ) -> dict[str, Any]:
                return {
                    "success": True,
                    "url": url,
                    "content": f"Content from {url}",
                    "formats": formats or ["markdown"],
                }

            async def crawl_site(
                self,
                url: str,
                max_pages: int = 50,
                _formats: list[str] | None = None,
            ) -> dict[str, Any]:
                pages = [
                    {"url": f"{url}/page{i}", "content": f"Page {i}"}
                    for i in range(min(max_pages, 3))
                ]
                return {
                    "success": True,
                    "pages": pages,
                    "total": len(pages),
                    "max_pages": max_pages,
                }

            async def initialize(self) -> None:
                self.initialized = True

            async def cleanup(self) -> None:
                self.cleanup_called = True
                self.initialized = False

        provider = TestCrawlProvider()

        # Test initialization
        await provider.initialize()
        assert provider.initialized is True

        # Test scrape_url
        result = await provider.scrape_url("https://example.com")
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert "Content from https://example.com" in result["content"]

        # Test scrape_url with formats
        result = await provider.scrape_url("https://example.com", ["html", "markdown"])
        assert result["formats"] == ["html", "markdown"]

        # Test crawl_site
        result = await provider.crawl_site("https://example.com", max_pages=2)
        assert result["success"] is True
        assert result["total"] == 2
        assert len(result["pages"]) == 2
        assert result["max_pages"] == 2

        # Test cleanup
        await provider.cleanup()
        assert provider.cleanup_called is True
        assert provider.initialized is False

    def test_inheritance_hierarchy(self):
        """Test the class inheritance hierarchy."""
        # CrawlProvider should inherit from ABC
        assert ABC in CrawlProvider.__mro__

        # Should have proper method resolution order
        mro_classes = [cls.__name__ for cls in CrawlProvider.__mro__]
        assert "CrawlProvider" in mro_classes
        assert "ABC" in mro_classes

    def test_abstract_method_enforcement(self):
        """Test that missing abstract methods prevent instantiation."""

        # Missing scrape_url
        class IncompleteProvider1(CrawlProvider):
            async def crawl_site(
                self, _url: str, _max_pages: int = 50, _formats: list[str] | None = None
            ) -> dict[str, Any]:
                return {}

            async def initialize(self) -> None:
                pass

            async def cleanup(self) -> None:
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider1()

        # Missing crawl_site
        class IncompleteProvider2(CrawlProvider):
            async def scrape_url(
                self, _url: str, _formats: list[str] | None = None
            ) -> dict[str, Any]:
                return {}

            async def initialize(self) -> None:
                pass

            async def cleanup(self) -> None:
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider2()

        # Missing initialize
        class IncompleteProvider3(CrawlProvider):
            async def scrape_url(
                self, _url: str, _formats: list[str] | None = None
            ) -> dict[str, Any]:
                return {}

            async def crawl_site(
                self, _url: str, _max_pages: int = 50, _formats: list[str] | None = None
            ) -> dict[str, Any]:
                return {}

            async def cleanup(self) -> None:
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider3()

        # Missing cleanup
        class IncompleteProvider4(CrawlProvider):
            async def scrape_url(
                self, _url: str, _formats: list[str] | None = None
            ) -> dict[str, Any]:
                return {}

            async def crawl_site(
                self, _url: str, _max_pages: int = 50, _formats: list[str] | None = None
            ) -> dict[str, Any]:
                return {}

            async def initialize(self) -> None:
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider4()

    def test_method_parameter_defaults(self):
        """Test default parameter values are correctly defined."""
        # Check crawl_site default parameters

        sig = inspect.signature(CrawlProvider.crawl_site)

        assert sig.parameters["max_pages"].default == 50
        assert sig.parameters["formats"].default is None

        # Check scrape_url default parameters
        sig = inspect.signature(CrawlProvider.scrape_url)
        assert sig.parameters["formats"].default is None

    @pytest.mark.asyncio
    async def test_provider_lifecycle_pattern(self):
        """Test typical provider lifecycle pattern."""

        class LifecycleTestProvider(CrawlProvider):
            """Provider to test lifecycle pattern."""

            def __init__(self):
                self.state = "created"
                self.operations = []

            async def initialize(self) -> None:
                self.state = "initialized"
                self.operations.append("initialize")

            async def scrape_url(
                self, url: str, _formats: list[str] | None = None
            ) -> dict[str, Any]:
                if self.state != "initialized":
                    raise RuntimeError("Provider not initialized")
                self.operations.append(f"scrape_{url}")
                return {"success": True, "url": url}

            async def crawl_site(
                self, url: str, _max_pages: int = 50, _formats: list[str] | None = None
            ) -> dict[str, Any]:
                if self.state != "initialized":
                    raise RuntimeError("Provider not initialized")
                self.operations.append(f"crawl_{url}")
                return {"success": True, "pages": []}

            async def cleanup(self) -> None:
                self.state = "cleaned_up"
                self.operations.append("cleanup")

        provider = LifecycleTestProvider()

        # Initial state
        assert provider.state == "created"
        assert provider.operations == []

        # Should fail before initialization
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await provider.scrape_url("https://example.com")

        # Initialize
        await provider.initialize()
        assert provider.state == "initialized"
        assert "initialize" in provider.operations

        # Should work after initialization
        await provider.scrape_url("https://example.com")
        await provider.crawl_site("https://example.com")

        # Cleanup
        await provider.cleanup()
        assert provider.state == "cleaned_up"
        assert "cleanup" in provider.operations

        # Should fail after cleanup
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await provider.scrape_url("https://example.com")
