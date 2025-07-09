"""Fixtures for test data generation and management.

This module provides fixtures for generating consistent test data
including documents, embeddings, and configuration objects.
"""

from datetime import datetime, timedelta

import pytest


@pytest.fixture
def sample_documents():
    """Generate sample documents for testing."""
    base_time = datetime(2024, 1, 1)

    return [
        {
            "id": "doc1",
            "url": "https://docs.example.com/guide/intro",
            "title": "Introduction to Example Framework",
            "content": "This guide provides a comprehensive introduction to the Example Framework...",
            "metadata": {
                "author": "Documentation Team",
                "version": "1.0",
                "category": "tutorial",
                "tags": ["intro", "beginner", "guide"],
                "last_updated": (base_time + timedelta(days=0)).isoformat(),
            },
        },
        {
            "id": "doc2",
            "url": "https://docs.example.com/api/reference",
            "title": "API Reference",
            "content": "Complete API reference for all available endpoints and methods...",
            "metadata": {
                "author": "API Team",
                "version": "2.0",
                "category": "reference",
                "tags": ["api", "reference", "advanced"],
                "last_updated": (base_time + timedelta(days=7)).isoformat(),
            },
        },
        {
            "id": "doc3",
            "url": "https://docs.example.com/examples/advanced",
            "title": "Advanced Examples",
            "content": "Collection of advanced usage examples and patterns...",
            "metadata": {
                "author": "Community",
                "version": "1.5",
                "category": "examples",
                "tags": ["examples", "advanced", "patterns"],
                "last_updated": (base_time + timedelta(days=14)).isoformat(),
            },
        },
    ]


@pytest.fixture
def sample_chunks():
    """Generate sample text chunks with overlap."""
    text = """
    Machine learning is a subset of artificial intelligence that focuses on
    building applications that learn from data and improve their accuracy
    over time without being programmed to do so. In data science, machine
    learning is used to analyze massive quantities of data and automate
    analytical model building. Using algorithms that iteratively learn from
    data, machine learning allows computers to find hidden insights without
    being explicitly programmed where to look.
    """

    chunks = []
    chunk_size = 100
    overlap = 20

    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(
            {
                "text": chunk,
                "start_index": i,
                "end_index": min(i + chunk_size, len(words)),
                "chunk_index": len(chunks),
            }
        )

    return chunks


@pytest.fixture
def config_templates():
    """Configuration templates for different environments."""
    return {
        "development": {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "services": {
                "qdrant": {
                    "host": "localhost",
                    "port": 6333,
                    "https": False,
                    "api_key": None,
                },
                "redis": {"host": "localhost", "port": 6379, "db": 0, "password": None},
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "text-embedding-3-small",
                    "max_retries": 3,
                },
            },
            "features": {"rate_limiting": False, "caching": True, "monitoring": False},
        },
        "production": {
            "environment": "production",
            "debug": False,
            "log_level": "INFO",
            "services": {
                "qdrant": {
                    "host": "${QDRANT_HOST}",
                    "port": 6333,
                    "https": True,
                    "api_key": "${QDRANT_API_KEY}",
                },
                "redis": {
                    "host": "${REDIS_HOST}",
                    "port": 6379,
                    "db": 0,
                    "password": "${REDIS_PASSWORD}",
                    "ssl": True,
                },
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "text-embedding-3-large",
                    "max_retries": 5,
                },
            },
            "features": {"rate_limiting": True, "caching": True, "monitoring": True},
        },
    }


@pytest.fixture
def embedding_test_cases():
    """Test cases for embedding operations."""
    return {
        "similar_texts": [
            "Machine learning is a type of artificial intelligence",
            "AI and machine learning are closely related fields",
            "Artificial intelligence includes machine learning",
        ],
        "dissimilar_texts": [
            "The weather today is sunny and warm",
            "Pizza is a popular Italian dish",
            "Swimming is a great form of exercise",
        ],
        "edge_cases": [
            "",  # Empty string
            "a",  # Single character
            "🚀🌟✨",  # Only emojis
            " " * 1000,  # Only spaces
            "test" * 500,  # Very long repetitive text
        ],
    }


@pytest.fixture
def query_test_suite():
    """Comprehensive query test suite."""
    return {
        "simple_queries": [
            "What is machine learning?",
            "How to install the framework?",
            "API authentication",
        ],
        "complex_queries": [
            "Explain the difference between supervised and unsupervised learning with examples",
            "What are the performance implications of using batch processing vs stream processing?",
            "How to implement OAuth2 authentication with refresh tokens?",
        ],
        "edge_queries": [
            "?",
            "!!!",
            "a" * 1000,
            "SELECT * FROM users WHERE 1=1",  # SQL injection attempt
            "<script>alert('test')</script>",  # XSS attempt
        ],
    }


@pytest.fixture
def performance_baselines():
    """Performance baseline metrics for comparison."""
    return {
        "embedding_generation": {"p50_ms": 50, "p95_ms": 100, "p99_ms": 200},
        "vector_search": {"p50_ms": 10, "p95_ms": 25, "p99_ms": 50},
        "document_processing": {"docs_per_second": 10, "mb_per_second": 5},
        "api_latency": {"search_p95_ms": 150, "index_p95_ms": 300},
    }
