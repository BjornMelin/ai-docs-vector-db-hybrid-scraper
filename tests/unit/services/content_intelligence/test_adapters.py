"""Tests for content intelligence site adapters."""

import pytest
from unittest.mock import AsyncMock

from src.services.content_intelligence.adapters import SiteAdapter
from src.services.content_intelligence.models import AdaptationRecommendation, AdaptationStrategy


@pytest.fixture
def site_adapter():
    """Create SiteAdapter instance."""
    return SiteAdapter()


class TestSiteAdapter:
    """Test SiteAdapter functionality."""

    async def test_initialize(self, site_adapter):
        """Test site adapter initialization."""
        await site_adapter.initialize()
        assert site_adapter._initialized is True

    async def test_analyze_github_patterns(self, site_adapter):
        """Test pattern analysis for GitHub URLs."""
        url = "https://github.com/user/repo/blob/main/README.md"
        content = """
        # My Project
        
        This is a README file for my project.
        
        ## Installation
        
        ```bash
        npm install
        ```
        
        ## Usage
        
        ```javascript
        const myLib = require('my-lib');
        ```
        """
        
        await site_adapter.initialize()
        patterns = await site_adapter.analyze_site_patterns(url, content)
        
        assert "known_site:github.com" in patterns
        # Should detect documentation patterns due to README content

    async def test_analyze_stackoverflow_patterns(self, site_adapter):
        """Test pattern analysis for Stack Overflow URLs."""
        url = "https://stackoverflow.com/questions/12345/how-to-fix-error"
        content = """
        **Question:** How do I fix this Python error?
        
        I'm getting this error when running my code:
        ```
        TypeError: 'NoneType' object is not subscriptable
        ```
        
        **Answer:** This error occurs when...
        
        **Comments:** Thanks, that worked!
        """
        
        await site_adapter.initialize()
        patterns = await site_adapter.analyze_site_patterns(url, content)
        
        assert "known_site:stackoverflow.com" in patterns

    async def test_analyze_spa_patterns(self, site_adapter):
        """Test detection of Single Page Application patterns."""
        url = "https://app.example.com/dashboard"
        html = """
        <html>
        <head>
            <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
        </head>
        <body>
            <div id="root" data-reactroot=""></div>
            <script>
                // React app initialization
            </script>
        </body>
        </html>
        """
        
        await site_adapter.initialize()
        patterns = await site_adapter.analyze_site_patterns(url, html=html)
        
        assert "react_app" in patterns
        assert "spa_content" in patterns or any("react" in p for p in patterns)

    async def test_analyze_lazy_loading_patterns(self, site_adapter):
        """Test detection of lazy loading patterns."""
        url = "https://blog.example.com/article"
        html = """
        <html>
        <body>
            <img data-src="image1.jpg" class="lazy" alt="Image 1">
            <img data-src="image2.jpg" loading="lazy" alt="Image 2">
            <div class="lazy-load-container">
                <p>Content that loads on scroll...</p>
            </div>
        </body>
        </html>
        """
        
        await site_adapter.initialize()
        patterns = await site_adapter.analyze_site_patterns(url, html=html)
        
        assert "lazy_loading" in patterns

    async def test_analyze_infinite_scroll_patterns(self, site_adapter):
        """Test detection of infinite scroll patterns."""
        url = "https://social.example.com/feed"
        content = "Loading more posts... infinite scroll pagination"
        html = """
        <div class="infinite-scroll-container">
            <div class="post">Post 1</div>
            <div class="post">Post 2</div>
            <div class="load-more-trigger">Loading more...</div>
        </div>
        """
        
        await site_adapter.initialize()
        patterns = await site_adapter.analyze_site_patterns(url, content, html)
        
        assert "infinite_scroll" in patterns

    async def test_analyze_paywall_patterns(self, site_adapter):
        """Test detection of paywall patterns."""
        url = "https://premium.example.com/article"
        content = """
        This is a premium article. To continue reading, please subscribe to our service.
        
        Paywall: Subscribe now for unlimited access to premium content.
        """
        
        await site_adapter.initialize()
        patterns = await site_adapter.analyze_site_patterns(url, content)
        
        assert "paywall_detected" in patterns

    async def test_analyze_structured_data_patterns(self, site_adapter):
        """Test detection of structured data patterns."""
        url = "https://example.com/article"
        html = """
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": "Test Article"
            }
            </script>
        </head>
        <body>
            <article itemscope itemtype="https://schema.org/Article">
                <h1 itemprop="headline">Test Article</h1>
            </article>
        </body>
        </html>
        """
        
        await site_adapter.initialize()
        patterns = await site_adapter.analyze_site_patterns(url, html=html)
        
        assert "structured_data" in patterns
        assert "microdata" in patterns

    async def test_analyze_url_patterns(self, site_adapter):
        """Test detection of URL-based patterns."""
        await site_adapter.initialize()
        
        # Documentation URL
        doc_patterns = await site_adapter.analyze_site_patterns("https://example.com/docs/api")
        assert "documentation_site" in doc_patterns
        
        # Blog URL
        blog_patterns = await site_adapter.analyze_site_patterns("https://example.com/blog/post-title")
        assert "blog_content" in blog_patterns
        
        # API URL
        api_patterns = await site_adapter.analyze_site_patterns("https://api.example.com/v1/users")
        assert "api_endpoint" in api_patterns
        
        # Wiki URL
        wiki_patterns = await site_adapter.analyze_site_patterns("https://example.com/wiki/page")
        assert "wiki_content" in wiki_patterns

    async def test_get_github_recommendations(self, site_adapter):
        """Test site-specific recommendations for GitHub."""
        url = "https://github.com/user/repo"
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(url)
        
        assert len(recommendations) > 0
        
        # Should have GitHub-specific recommendations
        github_rec = next((r for r in recommendations if r.site_domain == "github.com"), None)
        assert github_rec is not None
        assert github_rec.strategy == AdaptationStrategy.EXTRACT_MAIN_CONTENT
        assert ".markdown-body" in github_rec.selector_patterns

    async def test_get_stackoverflow_recommendations(self, site_adapter):
        """Test site-specific recommendations for Stack Overflow."""
        url = "https://stackoverflow.com/questions/12345"
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(url)
        
        assert len(recommendations) > 0
        
        # Should have Stack Overflow-specific recommendations
        so_rec = next((r for r in recommendations if r.site_domain == "stackoverflow.com"), None)
        assert so_rec is not None
        assert AdaptationStrategy.FOLLOW_SCHEMA in [so_rec.strategy] + [s for s in (so_rec.fallback_strategies or [])]

    async def test_get_medium_recommendations(self, site_adapter):
        """Test site-specific recommendations for Medium."""
        url = "https://medium.com/@author/article-title"
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(url)
        
        assert len(recommendations) > 0
        
        # Should recommend waiting for dynamic content
        medium_rec = next((r for r in recommendations if r.site_domain == "medium.com"), None)
        assert medium_rec is not None
        assert medium_rec.strategy in [AdaptationStrategy.WAIT_FOR_LOAD, AdaptationStrategy.HANDLE_DYNAMIC]

    async def test_get_pattern_based_recommendations(self, site_adapter):
        """Test recommendations based on detected patterns."""
        url = "https://unknown-site.com/page"
        detected_patterns = ["spa_content", "lazy_loading", "structured_data"]
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(
            url, detected_patterns=detected_patterns
        )
        
        assert len(recommendations) > 0
        
        # Should have recommendations for each detected pattern
        strategies = [r.strategy for r in recommendations]
        assert AdaptationStrategy.WAIT_FOR_LOAD in strategies  # For SPA
        assert AdaptationStrategy.SCROLL_TO_LOAD in strategies  # For lazy loading
        assert AdaptationStrategy.FOLLOW_SCHEMA in strategies  # For structured data

    async def test_get_quality_issue_recommendations(self, site_adapter):
        """Test recommendations based on quality issues."""
        url = "https://example.com/page"
        quality_issues = [
            "Incomplete content extraction",
            "Poor content structure",
            "Loading timeout issues"
        ]
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(
            url, quality_issues=quality_issues
        )
        
        assert len(recommendations) > 0
        
        # Should have recommendations addressing the quality issues
        strategies = [r.strategy for r in recommendations]
        assert AdaptationStrategy.EXTRACT_MAIN_CONTENT in strategies  # For incomplete content
        assert AdaptationStrategy.FOLLOW_SCHEMA in strategies  # For poor structure
        assert AdaptationStrategy.WAIT_FOR_LOAD in strategies  # For loading issues

    async def test_recommendation_prioritization(self, site_adapter):
        """Test that recommendations are properly prioritized."""
        url = "https://github.com/user/repo"
        detected_patterns = ["known_site:github.com", "spa_content"]
        quality_issues = ["Minor formatting issues"]
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(
            url, 
            detected_patterns=detected_patterns,
            quality_issues=quality_issues
        )
        
        assert len(recommendations) > 0
        
        # First recommendation should be high priority
        assert recommendations[0].priority in ["critical", "high"]
        
        # Recommendations should be sorted by priority and confidence
        for i in range(1, len(recommendations)):
            prev_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}[recommendations[i-1].priority]
            curr_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}[recommendations[i].priority]
            
            # Either same or lower priority
            assert curr_priority <= prev_priority

    async def test_recommendation_confidence_scores(self, site_adapter):
        """Test that recommendations have appropriate confidence scores."""
        url = "https://docs.python.org/3/tutorial/"
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(url)
        
        assert len(recommendations) > 0
        
        # All recommendations should have confidence scores
        for rec in recommendations:
            assert 0.0 <= rec.confidence <= 1.0
            assert rec.reasoning is not None
            assert len(rec.reasoning) > 0

    async def test_recommendation_implementation_notes(self, site_adapter):
        """Test that recommendations include implementation notes."""
        url = "https://github.com/user/repo/blob/main/README.md"
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(url)
        
        assert len(recommendations) > 0
        
        # Recommendations should have implementation notes
        github_rec = next((r for r in recommendations if r.site_domain == "github.com"), None)
        assert github_rec is not None
        assert github_rec.implementation_notes is not None
        assert len(github_rec.implementation_notes) > 0

    async def test_extract_domain(self, site_adapter):
        """Test domain extraction from URLs."""
        assert site_adapter._extract_domain("https://www.github.com/user/repo") == "github.com"
        assert site_adapter._extract_domain("https://docs.python.org/3/") == "docs.python.org"
        assert site_adapter._extract_domain("http://stackoverflow.com/questions/123") == "stackoverflow.com"
        assert site_adapter._extract_domain("invalid-url") == ""

    async def test_detect_content_patterns(self, site_adapter):
        """Test content pattern detection methods."""
        # SPA content
        spa_content = "This app uses React and dynamic loading..."
        patterns = site_adapter._detect_content_patterns(spa_content)
        assert "spa_content" in patterns
        
        # Dynamic content
        dynamic_content = "Loading... please wait for AJAX to complete"
        patterns = site_adapter._detect_content_patterns(dynamic_content)
        assert "dynamic_content" in patterns
        
        # Paywall content
        paywall_content = "Subscribe now for premium access to exclusive content"
        patterns = site_adapter._detect_content_patterns(paywall_content)
        assert "paywall_detected" in patterns
        
        # Incomplete content
        short_content = "Brief"
        patterns = site_adapter._detect_content_patterns(short_content)
        assert "incomplete_content" in patterns

    async def test_detect_html_patterns(self, site_adapter):
        """Test HTML pattern detection methods."""
        # React app
        react_html = '<div data-reactroot=""></div>'
        patterns = site_adapter._detect_html_patterns(react_html)
        assert "react_app" in patterns
        
        # Angular app
        angular_html = '<div ng-app="myApp"></div>'
        patterns = site_adapter._detect_html_patterns(angular_html)
        assert "angular_app" in patterns
        
        # Lazy loading
        lazy_html = '<img data-src="image.jpg" loading="lazy">'
        patterns = site_adapter._detect_html_patterns(lazy_html)
        assert "lazy_loading" in patterns
        
        # Structured data
        structured_html = '<script type="application/ld+json">{}</script>'
        patterns = site_adapter._detect_html_patterns(structured_html)
        assert "structured_data" in patterns

    async def test_detect_url_patterns(self, site_adapter):
        """Test URL pattern detection methods."""
        # API endpoint
        api_patterns = site_adapter._detect_url_patterns("https://api.example.com/v1/users")
        assert "api_endpoint" in api_patterns
        
        # Documentation
        doc_patterns = site_adapter._detect_url_patterns("https://example.com/docs/guide")
        assert "documentation_site" in doc_patterns
        
        # Blog
        blog_patterns = site_adapter._detect_url_patterns("https://example.com/blog/post-title")
        assert "blog_content" in blog_patterns
        
        # Dynamic parameters
        param_patterns = site_adapter._detect_url_patterns("https://example.com/page?id=123&page=2")
        assert "dynamic_params" in param_patterns

    async def test_fallback_strategies(self, site_adapter):
        """Test that recommendations include fallback strategies."""
        url = "https://complex-spa.example.com/app"
        detected_patterns = ["spa_content", "dynamic_content", "lazy_loading"]
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(
            url, detected_patterns=detected_patterns
        )
        
        # Some recommendations should have fallback strategies
        fallback_found = any(rec.fallback_strategies for rec in recommendations)
        assert fallback_found

    async def test_estimated_improvement(self, site_adapter):
        """Test that recommendations include estimated improvement scores."""
        url = "https://github.com/user/repo"
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(url)
        
        assert len(recommendations) > 0
        
        # All recommendations should have estimated improvement
        for rec in recommendations:
            assert 0.0 <= rec.estimated_improvement <= 1.0

    async def test_multiple_pattern_handling(self, site_adapter):
        """Test handling of multiple simultaneous patterns."""
        url = "https://modern-app.example.com/page"
        detected_patterns = [
            "spa_content",
            "lazy_loading", 
            "structured_data",
            "infinite_scroll",
            "dynamic_content"
        ]
        
        await site_adapter.initialize()
        recommendations = await site_adapter.get_site_recommendations(
            url, detected_patterns=detected_patterns
        )
        
        # Should generate multiple recommendations without conflicts
        assert len(recommendations) >= len(detected_patterns) - 1  # Some patterns might share strategies
        
        # Should cover all major patterns
        strategies = [r.strategy for r in recommendations]
        assert AdaptationStrategy.WAIT_FOR_LOAD in strategies
        assert AdaptationStrategy.SCROLL_TO_LOAD in strategies
        assert AdaptationStrategy.FOLLOW_SCHEMA in strategies

    async def test_cleanup(self, site_adapter):
        """Test site adapter cleanup."""
        await site_adapter.initialize()
        await site_adapter.cleanup()
        assert site_adapter._initialized is False