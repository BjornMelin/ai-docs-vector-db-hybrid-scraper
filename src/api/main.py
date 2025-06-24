"""Main FastAPI application for the AI Docs Vector DB Hybrid Scraper.

This module provides the main FastAPI application instance and core API endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_config

# Get configuration
config = get_config()

# Create FastAPI application
app = FastAPI(
    title="AI Docs Vector DB Hybrid Scraper",
    description="Hybrid AI documentation scraping system with vector database integration",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Docs Vector DB Hybrid Scraper API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2025-06-23T23:55:00Z"
    }


@app.get("/info")
async def info():
    """Information about the API."""
    return {
        "name": "AI Docs Vector DB Hybrid Scraper",
        "version": "0.1.0",
        "description": "Hybrid AI documentation scraping system with vector database integration",
        "python_version": "3.13+",
        "framework": "FastAPI"
    }


# Additional router imports can be added here as the API grows
# from .routers import search, documents, collections
# app.include_router(search.router, prefix="/api/v1")
# app.include_router(documents.router, prefix="/api/v1")  
# app.include_router(collections.router, prefix="/api/v1")

__all__ = ["app"]