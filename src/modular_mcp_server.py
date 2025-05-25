#!/usr/bin/env python3
"""
Modular MCP Server for AI Documentation Vector DB

This is the new modularized implementation that maintains backward compatibility
while providing better code organization and maintainability.
"""

import asyncio
import logging

# Import the modularized MCP server
from mcp.server import mcp

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the modular MCP server."""
    logger.info("Starting Modular MCP Server for AI Documentation Vector DB")
    logger.info("Features: Hybrid search, sparse vectors, reranking, caching, projects")
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()