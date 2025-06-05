"""Shared extraction utilities for web crawling providers.

This module provides reusable components for JavaScript execution and
documentation extraction that can be used across different crawling providers.
"""

from urllib.parse import urlparse


class JavaScriptExecutor:
    """Handle complex JavaScript execution for dynamic content.

    Provides site-specific JavaScript patterns and execution logic
    that can be shared across different crawling providers.
    """

    def __init__(self):
        """Initialize JavaScript executor with common patterns."""
        self.common_patterns = {
            "spa_navigation": """
                // Wait for SPA navigation
                await new Promise(resolve => {
                    const observer = new MutationObserver(() => {
                        if (document.querySelector('.content-loaded')) {
                            observer.disconnect();
                            resolve();
                        }
                    });
                    observer.observe(document.body, {childList: true, subtree: true});
                    setTimeout(resolve, 5000); // Timeout fallback
                });
            """,
            "infinite_scroll": """
                // Load all content via infinite scroll
                let lastHeight = 0;
                while (true) {
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(r => setTimeout(r, 1000));
                    let newHeight = document.body.scrollHeight;
                    if (newHeight === lastHeight) break;
                    lastHeight = newHeight;
                }
            """,
            "click_show_more": """
                // Click all "show more" buttons
                const buttons = document.querySelectorAll('[class*="show-more"], [class*="load-more"]');
                for (const button of buttons) {
                    button.click();
                    await new Promise(r => setTimeout(r, 500));
                }
            """,
        }

    def get_js_for_site(self, url: str) -> str | None:
        """Get custom JavaScript for specific documentation sites.

        Args:
            url: The URL to get site-specific JavaScript for

        Returns:
            JavaScript code as string or None if no specific pattern available
        """
        domain = urlparse(url).netloc

        # Site-specific JavaScript mapping
        site_js = {
            "docs.python.org": self.common_patterns["spa_navigation"],
            "reactjs.org": self.common_patterns["spa_navigation"],
            "react.dev": self.common_patterns["spa_navigation"],
            "developer.mozilla.org": self.common_patterns["click_show_more"],
            "stackoverflow.com": self.common_patterns["infinite_scroll"],
        }

        return site_js.get(domain)


class DocumentationExtractor:
    """Optimized extraction for technical documentation.

    Provides selectors and schema creation for extracting structured
    data from technical documentation sites.
    """

    def __init__(self):
        """Initialize documentation extractor with selector mappings."""
        self.selectors = {
            # Common documentation selectors
            "content": [
                "main",
                "article",
                ".content",
                ".documentation",
                "#main-content",
                ".markdown-body",
                ".doc-content",
            ],
            # Code blocks
            "code": [
                "pre code",
                ".highlight",
                ".code-block",
                ".language-*",
            ],
            # Navigation (to extract structure)
            "nav": [
                ".sidebar",
                ".toc",
                "nav",
                ".navigation",
            ],
            # Metadata
            "metadata": {
                "title": ["h1", ".title", "title"],
                "description": ["meta[name='description']", ".description"],
                "author": [".author", "meta[name='author']"],
                "version": [".version", ".release"],
                "last_updated": ["time", ".last-updated", ".modified"],
            },
        }

    def create_extraction_schema(self, doc_type: str = "general") -> dict:
        """Create extraction schema based on documentation type.

        Args:
            doc_type: Type of documentation ("api_reference", "tutorial", "guide", "general")

        Returns:
            Dictionary containing CSS selectors for structured extraction
        """
        schemas = {
            "api_reference": {
                "endpoints": "section.endpoint",
                "parameters": ".parameter",
                "responses": ".response",
                "examples": ".example",
            },
            "tutorial": {
                "steps": ".step, .tutorial-step",
                "code_examples": "pre code",
                "prerequisites": ".prerequisites",
                "objectives": ".objectives",
            },
            "guide": {
                "sections": "h2, h3",
                "content": "p, ul, ol",
                "callouts": ".note, .warning, .tip",
                "related": ".related-links",
            },
        }

        base_schema = {
            "title": self.selectors["metadata"]["title"],
            "content": self.selectors["content"],
            "code_blocks": self.selectors["code"],
        }

        return {**base_schema, **schemas.get(doc_type, {})}
