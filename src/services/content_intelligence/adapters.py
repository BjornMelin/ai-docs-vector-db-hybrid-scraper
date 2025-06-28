"""Site-specific adaptation patterns and optimization strategies.

This module provides automated site adaptation analysis and recommendations
for improving extraction quality based on site patterns and characteristics.
"""

import logging
from typing import Any
from urllib.parse import urlparse

from .models import AdaptationRecommendation, AdaptationStrategy


logger = logging.getLogger(__name__)


class SiteAdapter:
    """Site-specific optimization and adaptation analyzer."""

    def __init__(self):
        """Initialize site adapter with pattern databases."""
        self._initialized = False

        # Site-specific adaptation patterns
        self._site_patterns = {
            "github.com": {
                "selectors": {
                    "main_content": [".markdown-body", ".blob-wrapper", ".readme"],
                    "code_blocks": [".highlight", ".blob-code"],
                    "navigation": [".pagehead", ".reponav"],
                },
                "strategies": [AdaptationStrategy.EXTRACT_MAIN_CONTENT],
                "wait_conditions": [".js-navigation-container"],
                "confidence": 0.9,
                "notes": "GitHub has consistent structure with reliable selectors",
            },
            "stackoverflow.com": {
                "selectors": {
                    "main_content": [".question", ".answer"],
                    "code_blocks": [".s-code-block", ".highlight"],
                    "metadata": [".vote-count-post", ".post-signature"],
                },
                "strategies": [
                    AdaptationStrategy.FOLLOW_SCHEMA,
                    AdaptationStrategy.EXTRACT_MAIN_CONTENT,
                ],
                "confidence": 0.85,
                "notes": "Stack Overflow uses consistent Q&A schema",
            },
            "medium.com": {
                "selectors": {
                    "main_content": ["article[data-post-id]", ".postArticle-content"],
                    "metadata": [".postMetaInline", ".u-marginTop20"],
                },
                "strategies": [
                    AdaptationStrategy.WAIT_FOR_LOAD,
                    AdaptationStrategy.HANDLE_DYNAMIC,
                ],
                "wait_conditions": ["article[data-post-id]", ".postArticle-content"],
                "confidence": 0.75,
                "notes": "Medium uses dynamic loading, requires waiting",
            },
            "reddit.com": {
                "selectors": {
                    "main_content": ["[data-testid='post-content']", ".usertext-body"],
                    "comments": ["[data-testid='comment']", ".Comment"],
                },
                "strategies": [
                    AdaptationStrategy.HANDLE_DYNAMIC,
                    AdaptationStrategy.SCROLL_TO_LOAD,
                ],
                "wait_conditions": ["[data-testid='post-content']"],
                "confidence": 0.7,
                "notes": "Reddit has dynamic content loading",
            },
            "docs.python.org": {
                "selectors": {
                    "main_content": [".body", ".document", ".section"],
                    "navigation": [".sphinxsidebar", ".related"],
                    "code_blocks": [".highlight", ".code"],
                },
                "strategies": [
                    AdaptationStrategy.EXTRACT_MAIN_CONTENT,
                    AdaptationStrategy.FOLLOW_SCHEMA,
                ],
                "confidence": 0.9,
                "notes": "Sphinx documentation has consistent structure",
            },
            "wikipedia.org": {
                "selectors": {
                    "main_content": ["#mw-content-text", ".mw-parser-output"],
                    "metadata": [".infobox", ".navbox"],
                    "references": [".references", ".reflist"],
                },
                "strategies": [
                    AdaptationStrategy.EXTRACT_MAIN_CONTENT,
                    AdaptationStrategy.BYPASS_NAVIGATION,
                ],
                "confidence": 0.88,
                "notes": "Wikipedia has consistent MediaWiki structure",
            },
        }

        # Common pattern detection rules
        self._pattern_rules = {
            "spa_indicators": [
                "react",
                "angular",
                "vue",
                "spa",
                "single-page",
                "data-reactroot",
                "ng-app",
                "__nuxt",
                "_next",
            ],
            "infinite_scroll_indicators": [
                "infinite-scroll",
                "lazy-load",
                "load-more",
                "pagination",
                "scroll-to-load",
            ],
            "dynamic_content_indicators": [
                "ajax",
                "xhr",
                "fetch",
                "dynamic",
                "async-load",
                "data-src",
                "lazy",
                "skeleton",
            ],
            "paywall_indicators": [
                "paywall",
                "subscription",
                "premium",
                "paid-content",
                "member-only",
                "subscriber",
            ],
        }

    async def initialize(self) -> None:
        """Initialize the site adapter."""
        self._initialized = True
        logger.info("SiteAdapter initialized with site pattern database")

    async def analyze_site_patterns(
        self,
        url: str,
        content: str | None = None,
        html: str | None = None,
    ) -> list[str]:
        """Analyze site for specific patterns that affect extraction.

        Args:
            url: Site URL to analyze
            content: Optional content for pattern detection
            html: Optional HTML for pattern detection

        Returns:
            list[str]: List of detected patterns

        """
        patterns = []
        domain = self._extract_domain(url)

        # Check for known site patterns
        if domain in self._site_patterns:
            patterns.append(f"known_site:{domain}")

        # Analyze content patterns if available
        if content:
            patterns.extend(self._detect_content_patterns(content))

        # Analyze HTML patterns if available
        if html:
            patterns.extend(self._detect_html_patterns(html))

        # Analyze URL patterns
        patterns.extend(self._detect_url_patterns(url))

        return list(set(patterns))  # Remove duplicates

    async def get_site_recommendations(
        self,
        url: str,
        detected_patterns: list[str] | None = None,
        quality_issues: list[str] | None = None,
    ) -> list[AdaptationRecommendation]:
        """Get site-specific adaptation recommendations.

        Args:
            url: Target URL
            detected_patterns: Optional detected patterns
            quality_issues: Optional quality issues to address

        Returns:
            list[AdaptationRecommendation]: Site-specific recommendations

        """
        recommendations = []
        domain = self._extract_domain(url)

        # Get site-specific recommendations
        if domain in self._site_patterns:
            site_config = self._site_patterns[domain]
            recommendations.extend(
                self._generate_site_specific_recommendations(domain, site_config)
            )

        # Get pattern-based recommendations
        if detected_patterns:
            recommendations.extend(
                self._generate_pattern_recommendations(detected_patterns, url)
            )

        # Get quality-issue based recommendations
        if quality_issues:
            recommendations.extend(
                self._generate_issue_recommendations(quality_issues, url)
            )

        # Sort by priority and confidence
        recommendations.sort(
            key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.priority],
                x.confidence,
            ),
            reverse=True,
        )

        return recommendations

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL.

        Args:
            url: URL to parse

        Returns:
            str: Domain name

        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove 'www.' prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""

    def _detect_content_patterns(self, content: str) -> list[str]:
        """Detect patterns in text content.

        Args:
            content: Text content to analyze

        Returns:
            list[str]: Detected content patterns

        """
        patterns = []
        content_lower = content.lower()

        # Check for SPA indicators
        if any(
            indicator in content_lower
            for indicator in self._pattern_rules["spa_indicators"]
        ):
            patterns.append("spa_content")

        # Check for dynamic content indicators
        if any(
            indicator in content_lower
            for indicator in self._pattern_rules["dynamic_content_indicators"]
        ):
            patterns.append("dynamic_content")

        # Check for paywall indicators
        if any(
            indicator in content_lower
            for indicator in self._pattern_rules["paywall_indicators"]
        ):
            patterns.append("paywall_detected")

        # Check content completeness indicators
        if len(content) < 100:
            patterns.append("incomplete_content")
        elif "loading" in content_lower or "please wait" in content_lower:
            patterns.append("loading_content")

        return patterns

    def _detect_html_patterns(self, html: str) -> list[str]:
        """Detect patterns in HTML markup.

        Args:
            html: HTML content to analyze

        Returns:
            list[str]: Detected HTML patterns

        """
        patterns = []
        html_lower = html.lower()

        # Check for framework indicators
        if "data-reactroot" in html_lower or "react" in html_lower:
            patterns.append("react_app")
        elif "ng-app" in html_lower or "angular" in html_lower:
            patterns.append("angular_app")
        elif "__nuxt" in html_lower:
            patterns.append("nuxt_app")
        elif "_next" in html_lower:
            patterns.append("nextjs_app")

        # Check for lazy loading
        if "data-src" in html_lower or "lazy" in html_lower:
            patterns.append("lazy_loading")

        # Check for infinite scroll
        if any(
            indicator in html_lower
            for indicator in self._pattern_rules["infinite_scroll_indicators"]
        ):
            patterns.append("infinite_scroll")

        # Check for schema markup
        if "application/ld+json" in html_lower:
            patterns.append("structured_data")
        if "itemtype" in html_lower:
            patterns.append("microdata")

        return patterns

    def _detect_url_patterns(self, url: str) -> list[str]:
        """Detect patterns in URL structure.

        Args:
            url: URL to analyze

        Returns:
            list[str]: Detected URL patterns

        """
        patterns = []
        url_lower = url.lower()

        # Check for common URL patterns (use if statements instead of elif to detect multiple patterns)
        if "/api/" in url_lower or url_lower.startswith(
            ("https://api.", "http://api.")
        ):
            patterns.append("api_endpoint")
        if "/docs/" in url_lower or "/documentation/" in url_lower:
            patterns.append("documentation_site")
        if "/blog/" in url_lower:
            patterns.append("blog_content")
        if "/wiki/" in url_lower:
            patterns.append("wiki_content")
        if "/forum/" in url_lower or "/discussion/" in url_lower:
            patterns.append("forum_content")

        # Check for dynamic parameters
        if "?" in url and ("id=" in url_lower or "page=" in url_lower):
            patterns.append("dynamic_params")

        return patterns

    def _generate_site_specific_recommendations(
        self,
        domain: str,
        site_config: dict[str, Any],
    ) -> list[AdaptationRecommendation]:
        """Generate recommendations for known sites.

        Args:
            domain: Site domain
            site_config: Site configuration data

        Returns:
            list[AdaptationRecommendation]: Site-specific recommendations

        """
        recommendations = []

        for strategy in site_config.get("strategies", []):
            recommendation = AdaptationRecommendation(
                strategy=strategy,
                priority="high",
                confidence=site_config.get("confidence", 0.8),
                reasoning=f"Site-specific optimization for {domain}",
                implementation_notes=site_config.get("notes", ""),
                estimated_improvement=0.3,
                site_domain=domain,
                selector_patterns=site_config.get("selectors", {}).get(
                    "main_content", []
                ),
                wait_conditions=site_config.get("wait_conditions", []),
            )
            recommendations.append(recommendation)

        return recommendations

    def _generate_pattern_recommendations(
        self,
        patterns: list[str],
        _url: str,
    ) -> list[AdaptationRecommendation]:
        """Generate recommendations based on detected patterns.

        Args:
            patterns: Detected patterns
            url: Target URL

        Returns:
            list[AdaptationRecommendation]: Pattern-based recommendations

        """
        recommendations = []

        # SPA/React app recommendations
        if any(
            p in patterns
            for p in ["spa_content", "react_app", "angular_app", "nextjs_app"]
        ):
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.WAIT_FOR_LOAD,
                    priority="high",
                    confidence=0.85,
                    reasoning="Single Page Application detected - content loads dynamically",
                    implementation_notes="Wait for content to be rendered by JavaScript",
                    estimated_improvement=0.4,
                    wait_conditions=[
                        "[data-testid]",
                        ".content-loaded",
                        ".main-content",
                    ],
                    fallback_strategies=[AdaptationStrategy.HANDLE_DYNAMIC],
                )
            )

        # Infinite scroll recommendations
        if "infinite_scroll" in patterns:
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.SCROLL_TO_LOAD,
                    priority="medium",
                    confidence=0.8,
                    reasoning="Infinite scroll pattern detected",
                    implementation_notes="Scroll down to load additional content",
                    estimated_improvement=0.3,
                    fallback_strategies=[AdaptationStrategy.HANDLE_DYNAMIC],
                )
            )

        # Lazy loading recommendations
        if "lazy_loading" in patterns:
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.SCROLL_TO_LOAD,
                    priority="medium",
                    confidence=0.75,
                    reasoning="Lazy loading detected - images/content load on scroll",
                    implementation_notes="Scroll to trigger lazy loading of content",
                    estimated_improvement=0.25,
                )
            )

        # Dynamic content recommendations
        if "dynamic_content" in patterns:
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.HANDLE_DYNAMIC,
                    priority="medium",
                    confidence=0.7,
                    reasoning="Dynamic content loading patterns detected",
                    implementation_notes="Use JavaScript execution to handle dynamic content",
                    estimated_improvement=0.25,
                    fallback_strategies=[AdaptationStrategy.WAIT_FOR_LOAD],
                )
            )

        # Structured data recommendations
        if any(p in patterns for p in ["structured_data", "microdata"]):
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.FOLLOW_SCHEMA,
                    priority="medium",
                    confidence=0.8,
                    reasoning="Structured data markup detected",
                    implementation_notes="Extract data using schema.org or microdata patterns",
                    estimated_improvement=0.2,
                )
            )

        # Paywall handling
        if "paywall_detected" in patterns:
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.BYPASS_NAVIGATION,
                    priority="low",
                    confidence=0.5,
                    reasoning="Paywall detected - content may be limited",
                    implementation_notes="Extract available preview content",
                    estimated_improvement=0.1,
                )
            )

        return recommendations

    def _generate_issue_recommendations(
        self,
        quality_issues: list[str],
        _url: str,
    ) -> list[AdaptationRecommendation]:
        """Generate recommendations based on quality issues.

        Args:
            quality_issues: List of quality issues
            url: Target URL

        Returns:
            list[AdaptationRecommendation]: Issue-specific recommendations

        """
        recommendations = []

        # Incomplete content issues
        if any(
            "incomplete" in issue.lower() or "short" in issue.lower()
            for issue in quality_issues
        ):
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.EXTRACT_MAIN_CONTENT,
                    priority="high",
                    confidence=0.8,
                    reasoning="Incomplete content detected - better extraction needed",
                    implementation_notes="Focus on main content areas, avoid navigation/ads",
                    estimated_improvement=0.4,
                    selector_patterns=[".main", ".content", "article", ".post"],
                )
            )

        # Structure issues
        if any(
            "structure" in issue.lower() or "formatting" in issue.lower()
            for issue in quality_issues
        ):
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.FOLLOW_SCHEMA,
                    priority="medium",
                    confidence=0.7,
                    reasoning="Poor content structure - schema-based extraction may help",
                    implementation_notes="Look for semantic HTML elements and structured data",
                    estimated_improvement=0.3,
                )
            )

        # Loading issues
        if any(
            "loading" in issue.lower() or "dynamic" in issue.lower()
            for issue in quality_issues
        ):
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.WAIT_FOR_LOAD,
                    priority="high",
                    confidence=0.85,
                    reasoning="Content loading issues detected",
                    implementation_notes="Wait for dynamic content to fully load",
                    estimated_improvement=0.35,
                    wait_conditions=[".loaded", ".ready", "[data-loaded='true']"],
                )
            )

        return recommendations

    async def cleanup(self) -> None:
        """Cleanup adapter resources."""
        self._initialized = False
        logger.info("SiteAdapter cleaned up")