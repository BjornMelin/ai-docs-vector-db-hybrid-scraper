"""Crawl4AI service entry points."""

from .c4a_provider import crawl_best_first, crawl_deep_bfs, crawl_page


__all__ = ["crawl_page", "crawl_deep_bfs", "crawl_best_first"]
