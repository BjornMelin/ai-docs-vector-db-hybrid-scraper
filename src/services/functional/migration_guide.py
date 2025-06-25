"""Migration guide from service classes to functional services.

This module demonstrates how to convert existing service class usage
to the new functional approach with dependency injection.

BEFORE (Complex service classes):
```python
from src.services.embeddings.manager import EmbeddingManager
from src.services.monitoring.health import HealthChecker
from src.services.utilities.rate_limiter import RateLimitManager

# Complex initialization
config = Config()
embedding_manager = EmbeddingManager(
    config=config, client_manager=client_manager
)
await embedding_manager.initialize()

# Complex usage
result = await embedding_manager.generate_embeddings(texts)
await embedding_manager.cleanup()
```

AFTER (Simple functions with DI):
```python
from src.services.functional import generate_embeddings

# Simple usage with automatic resource management
result = await generate_embeddings(texts)
```
"""

from typing import Any, List

from fastapi import Depends, FastAPI

from src.services.functional import (
    # Rate limiting
    acquire_rate_limit,
    analyze_text_characteristics,
    # Auto-detection
    auto_configure_services,
    # Cache
    cache_get,
    cache_set,
    check_service_availability,
    # Monitoring
    check_service_health,
    # Crawling
    crawl_url,
    detect_environment,
    # Embeddings
    generate_embeddings,
    get_ab_test_variant,
    # Deployment
    get_feature_flag,
    get_system_status,
    increment_counter,
    rate_limited,
    timed,
)


# Example FastAPI application using functional services
app = FastAPI(title="Functional Services Example")


@app.get("/health")
@timed("health_check_endpoint")
async def health_check():
    """Example health check endpoint using functional monitoring."""
    # Before: complex HealthChecker initialization and usage
    # After: simple function call with automatic DI
    return await get_system_status()


@app.get("/environment")
async def get_environment():
    """Example environment detection using functional auto-detection."""
    # Before: EnvironmentDetector().detect()
    # After: simple function with DI
    environment = await detect_environment()
    services = await auto_configure_services()

    return {
        "environment": environment,
        "services": services,
    }


@app.post("/embeddings")
@rate_limited("openai", "embeddings")  # Automatic rate limiting
@timed("embedding_generation")  # Automatic timing
async def generate_text_embeddings(texts: List[str]):
    """Example embedding generation using functional embeddings service."""
    # Before: Complex EmbeddingManager lifecycle management
    # After: Simple function call with automatic resource management

    # Increment request counter
    await increment_counter("embedding_requests", tags={"endpoint": "batch"})

    # Generate embeddings (automatic provider selection, cost tracking, etc.)
    result = await generate_embeddings(
        texts=texts,
        auto_select=True,  # Smart provider selection
        generate_sparse=False,  # Dense embeddings only
    )

    return result


@app.get("/cache/{key}")
async def get_cached_value(key: str):
    """Example cache usage with functional cache service."""
    # Before: CacheManager initialization and lifecycle
    # After: Simple function calls

    value = await cache_get(key)
    if value is None:
        # Simulate generating new value
        new_value = f"Generated value for {key}"
        await cache_set(key, new_value, ttl=300)
        return {"key": key, "value": new_value, "cached": False}

    return {"key": key, "value": value, "cached": True}


@app.get("/crawl")
async def crawl_website(url: str):
    """Example crawling using functional crawling service."""
    # Before: Complex CrawlManager with provider selection
    # After: Simple function with automatic provider routing

    result = await crawl_url(url)
    return result


@app.get("/feature-flag/{flag_name}")
async def check_feature_flag(flag_name: str, user_id: str = "anonymous"):
    """Example feature flag checking using functional deployment service."""
    # Before: FeatureFlagManager initialization
    # After: Simple function call

    enabled = await get_feature_flag(flag_name, user_id)
    variant = await get_ab_test_variant(f"{flag_name}_test", user_id)

    return {
        "flag": flag_name,
        "enabled": enabled,
        "variant": variant,
        "user_id": user_id,
    }


@app.get("/service-status/{service_name}")
async def get_service_status(service_name: str):
    """Example service status check using functional monitoring."""
    # Before: Multiple service manager initializations
    # After: Simple function calls

    health = await check_service_health(service_name)
    availability = await check_service_availability(
        "localhost", 6333 if service_name == "qdrant" else 6379
    )

    return {
        "service": service_name,
        "health": health,
        "availability": availability,
    }


# Example of migrating complex service composition
class LegacyServiceOrchestrator:
    """BEFORE: Complex service orchestration with manual lifecycle management."""

    def __init__(self, config):
        self.config = config
        self.embedding_manager = None
        self.cache_manager = None
        self.rate_limiter = None
        self.health_checker = None

    async def initialize(self):
        # Complex initialization logic
        self.embedding_manager = EmbeddingManager(self.config)
        await self.embedding_manager.initialize()
        # ... more initialization

    async def process_documents(self, documents: List[str]) -> List[dict]:
        # Complex orchestration with manual error handling
        results = []
        for doc in documents:
            await self.rate_limiter.acquire("openai")
            cached = await self.cache_manager.get(f"embed:{doc}")
            if not cached:
                embedding = await self.embedding_manager.generate_embeddings([doc])
                await self.cache_manager.set(f"embed:{doc}", embedding)
                results.append(embedding)
            else:
                results.append(cached)
        return results

    async def cleanup(self):
        # Manual cleanup
        if self.embedding_manager:
            await self.embedding_manager.cleanup()
        # ... more cleanup


# AFTER: Simple functional composition
@app.post("/documents/process")
@rate_limited("openai", "embeddings")  # Automatic rate limiting
@timed("document_processing")  # Automatic timing
async def process_documents_functional(documents: List[str]):
    """AFTER: Simple functional composition with automatic resource management."""
    results = []

    for doc in documents:
        # Check cache first
        cache_key = f"embed:{hash(doc)}"
        cached = await cache_get(cache_key)

        if cached:
            results.append(cached)
            await increment_counter("cache_hits", tags={"operation": "embeddings"})
        else:
            # Generate embedding (automatic provider selection, error handling, etc.)
            embedding = await generate_embeddings([doc])

            # Cache result
            await cache_set(cache_key, embedding, ttl=3600)
            results.append(embedding)
            await increment_counter("cache_misses", tags={"operation": "embeddings"})

    return {"processed": len(documents), "results": results}


# Example of using functional services in business logic
async def intelligent_document_analysis(document_text: str) -> dict[str, Any]:
    """Example business logic using functional services."""

    # 1. Analyze text characteristics (automatic provider selection)
    analysis = await analyze_text_characteristics([document_text])

    # 2. Check feature flag for enhanced processing
    enhanced_processing = await get_feature_flag("enhanced_analysis", "system")

    # 3. Generate embeddings with smart provider selection
    embeddings = await generate_embeddings(
        texts=[document_text],
        auto_select=True,
        speed_priority=not enhanced_processing,
    )

    # 4. Log metrics
    await increment_counter(
        "document_analysis",
        tags={
            "enhanced": str(enhanced_processing),
            "text_type": analysis.text_type,
        },
    )

    return {
        "analysis": analysis,
        "embeddings": embeddings,
        "enhanced_processing": enhanced_processing,
    }


if __name__ == "__main__":
    import uvicorn

    # Example of running the functional services demo
    uvicorn.run(app, host="0.0.0.0", port=8000)
