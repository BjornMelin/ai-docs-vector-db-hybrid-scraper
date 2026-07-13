"""AI Docs Vector DB Hybrid Scraper.

A hybrid AI documentation scraping system combining Crawl4AI (bulk) + Firecrawl MCP
(on-demand) with Qdrant vector database for Claude Desktop/Code integration.
"""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("ai-docs-vector-db-hybrid-scraper")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0+unknown"
__author__ = "BjornMelin"

__all__ = [
    "__author__",
    "__version__",
]
