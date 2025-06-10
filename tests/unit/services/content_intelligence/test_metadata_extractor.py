"""Tests for content intelligence metadata extractor."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from datetime import datetime, timezone

from src.services.content_intelligence.metadata_extractor import MetadataExtractor
from src.services.content_intelligence.models import ContentMetadata


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    manager = AsyncMock()
    manager.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
    return manager


@pytest.fixture
def metadata_extractor(mock_embedding_manager):
    """Create MetadataExtractor instance with mocked dependencies."""
    return MetadataExtractor(embedding_manager=mock_embedding_manager)


class TestMetadataExtractor:
    """Test MetadataExtractor functionality."""

    async def test_initialize(self, metadata_extractor):
        """Test metadata extractor initialization."""
        await metadata_extractor.initialize()
        assert metadata_extractor._initialized is True

    async def test_extract_basic_metadata(self, metadata_extractor):
        """Test extraction of basic content metadata."""
        content = """
        This is a comprehensive article about machine learning fundamentals.
        
        Machine learning is a powerful subset of artificial intelligence that enables 
        computers to learn and make decisions from data without being explicitly programmed.
        
        The field encompasses various algorithms and techniques including supervised learning,
        unsupervised learning, and reinforcement learning.
        
        Applications of machine learning are found in computer vision, natural language processing,
        recommendation systems, and autonomous vehicles.
        """
        url = "https://example.com/ml-guide"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url)
        
        assert isinstance(result, ContentMetadata)
        assert result.url == url
        assert result.word_count > 50
        assert result.char_count > 300
        assert result.paragraph_count >= 4
        assert len(result.topics) > 0  # Should extract topics like "machine learning"
        assert len(result.tags) > 0  # Should extract relevant tags

    async def test_extract_from_html(self, metadata_extractor):
        """Test metadata extraction from HTML content."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Complete Guide to Python Programming</title>
            <meta name="description" content="Learn Python programming from basics to advanced concepts">
            <meta name="author" content="Jane Smith">
            <meta name="keywords" content="python, programming, tutorial, coding">
            <meta property="og:title" content="Complete Guide to Python Programming">
            <meta property="og:description" content="Comprehensive Python tutorial">
            <meta property="og:image" content="https://example.com/python-guide.jpg">
            <meta name="twitter:card" content="summary_large_image">
        </head>
        <body>
            <article>
                <h1>Complete Guide to Python Programming</h1>
                <p class="author">By Jane Smith</p>
                <p class="published">Published on March 15, 2024</p>
                
                <p>Python is a versatile programming language used for web development, 
                data science, automation, and more.</p>
                
                <h2>Getting Started</h2>
                <p>To begin with Python, you'll need to install it on your system...</p>
                
                <img src="https://example.com/python-logo.png" alt="Python Logo">
                <a href="https://python.org">Official Python Website</a>
                <a href="https://docs.python.org">Python Documentation</a>
            </article>
        </body>
        </html>
        """
        
        content = "Python is a versatile programming language used for web development..."
        url = "https://example.com/python-guide"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url, raw_html=html)
        
        assert result.title == "Complete Guide to Python Programming"
        assert result.description == "Learn Python programming from basics to advanced concepts"
        assert result.author == "Jane Smith"
        assert result.language == "en"
        assert len(result.links) >= 2  # Should extract links
        assert len(result.images) >= 1  # Should extract images
        assert "python" in [tag.lower() for tag in result.tags]

    async def test_extract_structured_data(self, metadata_extractor):
        """Test extraction of structured data (JSON-LD, microdata)."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Article with Structured Data</title>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": "Understanding Machine Learning",
                "author": {
                    "@type": "Person",
                    "name": "Dr. John Doe"
                },
                "datePublished": "2024-03-15",
                "dateModified": "2024-03-16",
                "description": "A comprehensive guide to machine learning concepts"
            }
            </script>
        </head>
        <body>
            <article itemscope itemtype="https://schema.org/Article">
                <h1 itemprop="headline">Understanding Machine Learning</h1>
                <p>Published by <span itemprop="author">Dr. John Doe</span></p>
                <time itemprop="datePublished" datetime="2024-03-15">March 15, 2024</time>
                <p itemprop="description">A comprehensive guide to machine learning concepts</p>
            </article>
        </body>
        </html>
        """
        
        content = "Machine learning content..."
        url = "https://example.com/ml-article"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url, raw_html=html)
        
        assert result.title == "Understanding Machine Learning"
        assert result.author == "Dr. John Doe"
        assert "schema.org" in str(result.schema_types)
        assert result.description == "A comprehensive guide to machine learning concepts"

    async def test_extract_timestamps(self, metadata_extractor):
        """Test extraction of publication and modification timestamps."""
        content = """
        # Latest AI Research Findings
        
        Published: March 15, 2024
        Last Updated: March 16, 2024
        
        Recent developments in artificial intelligence research have shown...
        
        This article was first published on March 15, 2024, and last modified on March 16, 2024.
        """
        url = "https://example.com/ai-research"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url)
        
        assert result.published_date is not None
        assert result.last_modified is not None
        # Both dates should be in March 2024
        assert result.published_date.year == 2024
        assert result.published_date.month == 3

    async def test_extract_technical_content(self, metadata_extractor):
        """Test metadata extraction from technical/code content."""
        content = """
        # Python Data Analysis Tutorial
        
        This tutorial covers data analysis using pandas and numpy.
        
        ## Installation
        
        ```bash
        pip install pandas numpy matplotlib
        ```
        
        ## Basic Usage
        
        ```python
        import pandas as pd
        import numpy as np
        
        # Read data
        df = pd.read_csv('data.csv')
        
        # Basic statistics
        print(df.describe())
        
        # Data visualization
        df.plot(kind='hist')
        ```
        
        Topics covered:
        - Data loading and cleaning
        - Statistical analysis
        - Data visualization
        - Performance optimization
        """
        url = "https://tutorial.example.com/python-data-analysis"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url)
        
        # Should identify programming-related tags and topics
        tags_lower = [tag.lower() for tag in result.tags]
        topics_lower = [topic.lower() for topic in result.topics]
        
        assert any("python" in tag for tag in tags_lower)
        assert any("data" in tag for tag in tags_lower)
        assert any("analysis" in topic for topic in topics_lower)
        assert result.programming_language == "python"

    async def test_extract_with_existing_metadata(self, metadata_extractor):
        """Test metadata extraction with existing extraction metadata."""
        content = "Sample content for testing metadata extraction."
        url = "https://example.com/test"
        
        extraction_metadata = {
            "extraction_method": "crawl4ai",
            "timestamp": "2024-03-15T10:30:00Z",
            "success": True,
            "load_time_ms": 200,
            "status_code": 200,
        }
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(
            content, url, extraction_metadata=extraction_metadata
        )
        
        assert result.extraction_method == "crawl4ai"
        assert result.load_time_ms == 200
        assert result.crawl_timestamp is not None

    async def test_extract_breadcrumbs_and_hierarchy(self, metadata_extractor):
        """Test extraction of page hierarchy and breadcrumbs."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Python Basics - Programming Tutorial - Example.com</title>
        </head>
        <body>
            <nav aria-label="breadcrumb">
                <ol class="breadcrumb">
                    <li><a href="/">Home</a></li>
                    <li><a href="/programming">Programming</a></li>
                    <li><a href="/programming/python">Python</a></li>
                    <li class="active">Basics</li>
                </ol>
            </nav>
            
            <main>
                <h1>Python Basics</h1>
                <p>Learn the fundamentals of Python programming...</p>
            </main>
        </body>
        </html>
        """
        
        content = "Learn the fundamentals of Python programming..."
        url = "https://example.com/programming/python/basics"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url, raw_html=html)
        
        # Should extract hierarchy information
        assert len(result.breadcrumbs) > 0
        assert any("python" in crumb.lower() for crumb in result.breadcrumbs)

    async def test_extract_content_hash(self, metadata_extractor):
        """Test generation of content hash for duplicate detection."""
        content1 = "This is unique content for testing."
        content2 = "This is unique content for testing."  # Same content
        content3 = "This is different content for testing."
        
        url = "https://example.com/test"
        
        await metadata_extractor.initialize()
        
        result1 = await metadata_extractor.extract_metadata(content1, url)
        result2 = await metadata_extractor.extract_metadata(content2, url)
        result3 = await metadata_extractor.extract_metadata(content3, url)
        
        # Same content should have same hash
        assert result1.content_hash == result2.content_hash
        # Different content should have different hash
        assert result1.content_hash != result3.content_hash

    async def test_extract_minimal_content(self, metadata_extractor):
        """Test metadata extraction from minimal content."""
        content = "Short."
        url = "https://example.com/minimal"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url)
        
        assert result.url == url
        assert result.word_count == 1
        assert result.char_count == 6  # "Short."
        assert result.paragraph_count >= 0

    async def test_extract_multilingual_content(self, metadata_extractor):
        """Test metadata extraction from non-English content."""
        content = """
        # Guide de Python pour Débutants
        
        Python est un langage de programmation polyvalent et facile à apprendre.
        Il est largement utilisé dans le développement web, la science des données,
        et l'automatisation.
        
        ## Installation
        
        Pour installer Python, visitez le site officiel...
        """
        
        html = """
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <title>Guide Python</title>
            <meta name="description" content="Guide de programmation Python en français">
        </head>
        <body>
            <h1>Guide de Python pour Débutants</h1>
        </body>
        </html>
        """
        
        url = "https://example.fr/python-guide"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url, raw_html=html)
        
        assert result.language == "fr"
        assert "python" in [tag.lower() for tag in result.tags]

    async def test_extract_social_media_metadata(self, metadata_extractor):
        """Test extraction of social media metadata (Open Graph, Twitter Cards)."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Social Media Article</title>
            <meta property="og:title" content="Amazing Python Tutorial">
            <meta property="og:description" content="Learn Python programming easily">
            <meta property="og:image" content="https://example.com/python-tutorial.jpg">
            <meta property="og:url" content="https://example.com/python-tutorial">
            <meta property="og:type" content="article">
            <meta name="twitter:card" content="summary_large_image">
            <meta name="twitter:title" content="Amazing Python Tutorial">
            <meta name="twitter:description" content="Learn Python programming easily">
            <meta name="twitter:image" content="https://example.com/python-tutorial.jpg">
        </head>
        <body>
            <h1>Python Tutorial</h1>
        </body>
        </html>
        """
        
        content = "Python tutorial content..."
        url = "https://example.com/python-tutorial"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url, raw_html=html)
        
        # Should extract social media metadata
        assert result.title == "Amazing Python Tutorial"
        assert result.description == "Learn Python programming easily"
        assert len(result.images) > 0

    async def test_extract_not_initialized_error(self, metadata_extractor):
        """Test that extractor raises error when not initialized."""
        with pytest.raises(RuntimeError, match="MetadataExtractor not initialized"):
            await metadata_extractor.extract_metadata("test content", "https://example.com")

    async def test_embedding_error_handling(self, metadata_extractor, mock_embedding_manager):
        """Test handling of embedding generation errors during topic extraction."""
        mock_embedding_manager.generate_embeddings.side_effect = Exception("Embedding error")
        
        content = "Content for testing error handling."
        url = "https://example.com/test"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url)
        
        # Should still return valid metadata even with embedding errors
        assert isinstance(result, ContentMetadata)
        assert result.url == url
        # Topics might be empty or limited due to embedding error
        assert isinstance(result.topics, list)

    async def test_invalid_html_handling(self, metadata_extractor):
        """Test handling of malformed HTML."""
        content = "Content with malformed HTML."
        url = "https://example.com/test"
        malformed_html = "<html><head><title>Test</title><body><h1>Unclosed heading<p>Paragraph</body>"
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(content, url, raw_html=malformed_html)
        
        # Should handle malformed HTML gracefully
        assert isinstance(result, ContentMetadata)
        assert result.url == url

    async def test_cleanup(self, metadata_extractor):
        """Test metadata extractor cleanup."""
        await metadata_extractor.initialize()
        await metadata_extractor.cleanup()
        assert metadata_extractor._initialized is False

    async def test_extract_with_empty_html(self, metadata_extractor):
        """Test metadata extraction with empty or None HTML."""
        content = "Content without HTML."
        url = "https://example.com/test"
        
        await metadata_extractor.initialize()
        
        # Test with None HTML
        result1 = await metadata_extractor.extract_metadata(content, url, raw_html=None)
        
        # Test with empty HTML
        result2 = await metadata_extractor.extract_metadata(content, url, raw_html="")
        
        assert isinstance(result1, ContentMetadata)
        assert isinstance(result2, ContentMetadata)
        assert result1.url == url
        assert result2.url == url

    async def test_extract_performance_metadata(self, metadata_extractor):
        """Test extraction of performance-related metadata."""
        content = "Performance test content."
        url = "https://example.com/performance"
        
        extraction_metadata = {
            "load_time_ms": 350,
            "total_size_bytes": 15000,
            "image_count": 5,
            "script_count": 3,
        }
        
        await metadata_extractor.initialize()
        result = await metadata_extractor.extract_metadata(
            content, url, extraction_metadata=extraction_metadata
        )
        
        assert result.load_time_ms == 350