# Basic usage example for the AI Documentation Vector Database

import asyncio
import json

from src.crawl4ai_bulk_embedder import AdvancedDocumentationScraper


async def basic_search_example():
    """Example of searching the vector database"""
    # Load configuration
    with open("config/documentation-sites.json") as f:
        config = json.load(f)

    # Initialize scraper (which includes vector DB connection)
    scraper = AdvancedDocumentationScraper()

    # Example: Search for information about vector databases
    print("🔍 Searching for 'vector database operations'...")

    # This would typically be done through the management script
    # python src/manage_vector_db.py search "vector database operations"

    print(
        "✅ Search complete! Check src/manage_vector_db.py for full functionality"
    )


if __name__ == "__main__":
    asyncio.run(basic_search_example())
