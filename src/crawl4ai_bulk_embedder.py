#!/usr/bin/env python3
"""
Advanced Documentation Scraper using Crawl4AI
Ultra-fast bulk documentation scraping with OpenAI embeddings and Qdrant storage
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Core libraries
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CONCURRENT_CRAWLS = 10
CHUNK_SIZE = 2000  # Characters per chunk for embedding

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl4ai_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedDocumentationScraper:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        self.setup_collection()
        self.processed_urls = set()
        self.stats = {
            'total_processed': 0,
            'successful_embeddings': 0,
            'failed_crawls': 0,
            'total_chunks': 0
        }

    def setup_collection(self):
        """Create or ensure Qdrant collection exists"""
        try:
            collections = self.qdrant_client.get_collections().collections
            if not any(c.name == COLLECTION_NAME for c in collections):
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {COLLECTION_NAME}")
            else:
                logger.info(f"Collection {COLLECTION_NAME} already exists")
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise

    async def crawl_documentation_site(self, base_url: str, max_pages: int = 100) -> List[Dict]:
        """Crawl an entire documentation site using Crawl4AI's deep crawling"""
        browser_config = BrowserConfig(
            headless=True,
            user_agent="Documentation-Scraper/1.0 (AI Training)",
            accept_downloads=False,
            java_script_enabled=True
        )

        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=50,
            exclude_external_links=True,
            remove_overlay_elements=True,
            process_iframes=False,
            wait_for_images=False,
            page_timeout=30000,
            css_selector="main, article, .content, .documentation, #content",
            excluded_tags=['nav', 'footer', 'header', 'aside', 'script', 'style']
        )

        crawled_pages = []
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            try:
                logger.info(f"Starting crawl of {base_url}")
                result = await crawler.arun(url=base_url, config=run_config)
                
                if result.success:
                    crawled_pages.append({
                        'url': base_url,
                        'title': result.metadata.get('title', ''),
                        'markdown': result.markdown,
                        'word_count': len(result.markdown.split()),
                        'links': result.links.get('internal', [])
                    })
                    
                    # Extract internal documentation links for deep crawling
                    doc_links = self._filter_documentation_links(result.links.get('internal', []), base_url)
                    doc_links = doc_links[:max_pages-1]  # Reserve one slot for base URL
                    
                    if doc_links:
                        logger.info(f"Found {len(doc_links)} documentation links to crawl")
                        
                        # Create semaphore to limit concurrent requests
                        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CRAWLS)
                        
                        # Crawl all documentation pages concurrently
                        tasks = [
                            self._crawl_single_page(crawler, url, run_config, semaphore)
                            for url in doc_links
                        ]
                        
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Process results 
                        for result in results:
                            if isinstance(result, dict) and result:
                                crawled_pages.append(result)
                            elif isinstance(result, Exception):
                                logger.error(f"Crawl error: {result}")
                                self.stats['failed_crawls'] += 1

                logger.info(f"Successfully crawled {len(crawled_pages)} pages from {base_url}")
                
            except Exception as e:
                logger.error(f"Error crawling {base_url}: {e}")
                self.stats['failed_crawls'] += 1

        return crawled_pages
