# 5-Tier Browser Automation API Documentation

This document provides comprehensive API documentation for the advanced 5-tier browser automation system with intelligent routing, caching, rate limiting, and monitoring.

## Overview

The browser automation system implements a sophisticated 5-tier hierarchy with automatic performance optimization:

1. **Tier 0: Lightweight HTTP** - httpx + BeautifulSoup (5-10x faster for static content)
2. **Tier 1: Crawl4AI Basic** - Standard browser automation for dynamic content
3. **Tier 2: Crawl4AI Enhanced** - Interactive content with custom JavaScript
4. **Tier 3: Browser-use AI** - Complex interactions with AI-powered automation
5. **Tier 4: Playwright + Firecrawl** - Maximum control + API fallback

## Core Architecture Components

### UnifiedBrowserManager

The central entry point providing a single interface to all 5 tiers with intelligent routing, caching, and monitoring.

#### `async scrape(request: UnifiedScrapingRequest | None = None, url: str | None = None, **kwargs) -> UnifiedScrapingResponse`

Primary scraping method with automatic tier selection and performance optimization.

**Parameters:**

- `request` (UnifiedScrapingRequest, optional): Structured request object (recommended)
- `url` (str, optional): URL to scrape (for simple usage)
- `**kwargs`: Additional parameters (tier, interaction_required, etc.)

**UnifiedScrapingRequest:**

```python
from src.services.browser.unified_manager import UnifiedScrapingRequest

request = UnifiedScrapingRequest(
    url="https://docs.example.com",
    tier="auto",  # "auto", "lightweight", "crawl4ai", "crawl4ai_enhanced", "browser_use", "playwright", "firecrawl"
    interaction_required=False,
    custom_actions=[  # For tier 3+ interactions
        {"type": "click", "selector": "#button"},
        {"type": "wait", "duration": 1000}
    ],
    timeout=30000,
    wait_for_selector=".content",
    extract_metadata=True
)
```

**UnifiedScrapingResponse:**

```python
{
    "success": bool,
    "content": str,
    "url": str,
    "title": str,
    "metadata": dict,
    
    # Execution details
    "tier_used": str,
    "execution_time_ms": float,
    "fallback_attempted": bool,
    
    # Quality metrics
    "content_length": int,
    "quality_score": float,  # 0.0-1.0
    
    # Error information
    "error": str,
    "failed_tiers": list
}
```

**Usage Examples:**

```python
from src.services.browser.unified_manager import UnifiedBrowserManager, UnifiedScrapingRequest
from src.config import UnifiedConfig

# Initialize manager
config = UnifiedConfig()
manager = UnifiedBrowserManager(config)
await manager.initialize()

# Simple scraping (automatic tier selection)
response = await manager.scrape(url="https://docs.example.com")

# Structured request with tier preference
request = UnifiedScrapingRequest(
    url="https://complex-spa.com",
    tier="browser_use",
    interaction_required=True,
    custom_actions=[
        {"type": "wait_for_selector", "selector": ".dynamic-content"},
        {"type": "click", "selector": "#load-more"},
        {"type": "extract", "target": "documentation"}
    ]
)
response = await manager.scrape(request)

# Force specific tier
response = await manager.scrape(
    url="https://static-site.com",
    tier="lightweight"
)
```

#### `async analyze_url(url: str) -> dict`

Analyze URL to determine optimal tier and provide performance insights.

```python
analysis = await manager.analyze_url("https://docs.example.com")
# Returns:
{
    "url": "https://docs.example.com",
    "domain": "docs.example.com",
    "recommended_tier": "crawl4ai",
    "expected_performance": {
        "estimated_time_ms": 1500.0,
        "success_rate": 0.95
    }
}
```

#### `get_system_status() -> dict`

Get comprehensive system health and performance information.

```python
status = manager.get_system_status()
# Returns:
{
    "status": "healthy",  # healthy, degraded, unhealthy
    "initialized": true,
    "total_requests": 1250,
    "overall_success_rate": 0.92,
    "tier_count": 5,
    "router_available": true,
    "cache_enabled": true,
    "cache_stats": {
        "hit_rate": 0.73,
        "total_size": "45.2MB",
        "entries": 342
    },
    "monitoring_enabled": true,
    "monitoring_health": {
        "overall_status": "healthy",
        "tier_health": {"total": 5, "healthy": 5},
        "recent_alerts": 0
    },
    "tier_metrics": {
        "lightweight": {"total_requests": 450, "success_rate": 0.98},
        "crawl4ai": {"total_requests": 380, "success_rate": 0.94},
        "browser_use": {"total_requests": 125, "success_rate": 0.87},
        "playwright": {"total_requests": 95, "success_rate": 0.91},
        "firecrawl": {"total_requests": 200, "success_rate": 0.96}
    }
}
```

### EnhancedAutomationRouter

Advanced routing engine with tier-specific configuration, performance metrics, and intelligent fallback.

#### `async scrape(url: str, interaction_required: bool = False, custom_actions: list = None, force_tool: str = None, timeout: int = 30000) -> dict`

Enhanced scraping with intelligent tier selection based on URL patterns and performance history.

**Features:**
- **URL Pattern Matching**: Domain-specific tier recommendations
- **Performance-Based Selection**: Historical success rate optimization
- **Circuit Breaker Pattern**: Automatic tier health management
- **Intelligent Fallback**: Multi-tier fallback strategies
- **Rate Limiting**: Tier-specific request throttling

**Configuration:**

```python
from src.services.browser.tier_config import EnhancedRoutingConfig, TierConfiguration

config = EnhancedRoutingConfig(
    tier_configs={
        "lightweight": TierConfiguration(
            tier_name="lightweight",
            tier_level=0,
            priority_score=10,
            requests_per_minute=300,
            max_concurrent_requests=20,
            timeout_ms=5000,
            fallback_tiers=["crawl4ai"]
        ),
        "crawl4ai": TierConfiguration(
            tier_name="crawl4ai",
            tier_level=1,
            priority_score=8,
            requests_per_minute=120,
            max_concurrent_requests=10,
            timeout_ms=15000,
            fallback_tiers=["crawl4ai_enhanced", "browser_use"]
        )
    },
    url_patterns={
        r".*docs\..*": "crawl4ai",
        r".*api\..*": "lightweight",
        r".*github\.com.*": "crawl4ai_enhanced"
    },
    domain_preferences={
        "docs.python.org": "crawl4ai",
        "stackoverflow.com": "lightweight",
        "complex-spa.com": "browser_use"
    }
)
```

#### `async get_performance_report() -> dict`

Comprehensive performance analytics and optimization insights.

```python
report = await router.get_performance_report()
# Returns:
{
    "total_requests": 1500,
    "success_rate": 0.924,
    "avg_response_time_ms": 2340.5,
    "tier_performance": {
        "lightweight": {
            "requests": 600,
            "success_rate": 0.982,
            "avg_time_ms": 450.2,
            "cache_hit_rate": 0.85
        },
        "crawl4ai": {
            "requests": 450,
            "success_rate": 0.945,
            "avg_time_ms": 1850.3,
            "cache_hit_rate": 0.72
        }
    },
    "fallback_analysis": {
        "frequency": 0.12,
        "most_common": "lightweight -> crawl4ai",
        "success_after_fallback": 0.89
    },
    "optimization_recommendations": [
        "Consider increasing lightweight tier capacity",
        "Review browser_use timeout settings"
    ]
}
```

## Advanced Features

### Browser Result Caching

Dynamic TTL-based caching system with content-type optimization.

```python
from src.services.cache.browser_cache import BrowserCache

cache = BrowserCache(
    default_ttl=3600,
    dynamic_content_ttl=300,    # Short TTL for dynamic content
    static_content_ttl=86400,   # Long TTL for static content
)

# Automatic cache key generation based on URL and tier
cache_key = cache._generate_cache_key("https://docs.example.com", "crawl4ai")

# Content-based TTL determination
ttl = cache._determine_ttl("https://static-site.com/guide.pdf", 50000)  # Longer TTL for PDFs
ttl = cache._determine_ttl("https://api.example.com/search", 1200)      # Shorter TTL for API endpoints
```

**Cache Performance:**
- 70-85% hit rate for documentation sites
- Dynamic TTL based on content type
- Automatic cache invalidation
- Memory-efficient LRU eviction

### Tier-Specific Rate Limiting

Advanced rate limiting with sliding window algorithm and concurrency control.

```python
from src.services.browser.tier_rate_limiter import TierRateLimiter, RateLimitContext

# Rate limiter with tier-specific limits
rate_limiter = TierRateLimiter(tier_configs)

# Acquire rate limit permission
async with RateLimitContext(rate_limiter, "browser_use") as allowed:
    if allowed:
        result = await perform_scraping()
    else:
        # Handle rate limit
        wait_time = rate_limiter.get_wait_time("browser_use")
        await asyncio.sleep(wait_time)

# Check rate limit status
status = rate_limiter.get_status("crawl4ai")
# Returns:
{
    "enabled": true,
    "concurrent_requests": 3,
    "max_concurrent": 10,
    "recent_requests": 15,
    "rate_limit_hits": 2,
    "requests_per_minute": 120,
    "remaining_capacity": 105,
    "wait_time_seconds": 0.0
}
```

### Real-Time Monitoring and Alerting

Comprehensive monitoring system with health status tracking and alert generation.

```python
from src.services.browser.monitoring import BrowserAutomationMonitor, MonitoringConfig, AlertSeverity

# Configure monitoring
monitor_config = MonitoringConfig(
    error_rate_threshold=0.1,
    response_time_threshold_ms=10000,
    cache_miss_threshold=0.8,
    alert_cooldown_seconds=300,
    max_alerts_per_hour=10
)

monitor = BrowserAutomationMonitor(monitor_config)
await monitor.start_monitoring()

# Record request metrics
await monitor.record_request_metrics(
    tier="crawl4ai",
    success=True,
    response_time_ms=1500.0,
    cache_hit=False
)

# Get system health
health = monitor.get_system_health()
# Returns:
{
    "overall_status": "healthy",
    "tier_health": {
        "total": 5,
        "healthy": 4,
        "degraded": 1,
        "unhealthy": 0
    },
    "recent_alerts": 1,
    "monitoring_active": true,
    "tier_details": {
        "lightweight": {"status": "healthy", "success_rate": 0.98},
        "crawl4ai": {"status": "degraded", "success_rate": 0.85}
    }
}

# Get active alerts
alerts = monitor.get_active_alerts(severity=AlertSeverity.HIGH)
for alert in alerts:
    print(f"Alert: {alert.message} (Tier: {alert.tier})")

# Add custom alert handler
def custom_alert_handler(alert):
    if alert.severity == AlertSeverity.CRITICAL:
        send_notification(alert.message)

monitor.add_alert_handler(custom_alert_handler)
```

## Tier-Specific Documentation

### Tier 0: Lightweight HTTP

**Best For:** Static content, documentation sites, API endpoints

```python
# Automatic selection for static content
response = await manager.scrape(
    url="https://docs.python.org/3/tutorial/",
    tier="lightweight"
)
```

**Performance:**
- 5-10x faster than browser-based tiers
- 95%+ success rate for static content
- Minimal resource usage
- Excellent for bulk processing

### Tier 1: Crawl4AI Basic

**Best For:** Standard dynamic content, most documentation sites

```python
response = await manager.scrape(
    url="https://react.dev/learn",
    tier="crawl4ai"
)
```

**Features:**
- JavaScript execution
- Dynamic content loading
- Intelligent content extraction
- Balanced performance/capability

### Tier 2: Crawl4AI Enhanced

**Best For:** Interactive content, SPAs with custom JavaScript

```python
response = await manager.scrape(
    url="https://interactive-docs.com",
    tier="crawl4ai_enhanced",
    custom_actions=[
        {"type": "execute_js", "script": "expandAllSections()"},
        {"type": "wait", "duration": 2000}
    ]
)
```

### Tier 3: Browser-use AI

**Best For:** Complex interactions requiring AI reasoning

```python
request = UnifiedScrapingRequest(
    url="https://complex-dashboard.com",
    tier="browser_use",
    interaction_required=True,
    custom_actions=[
        {
            "type": "ai_task",
            "instruction": "Navigate to the documentation section and extract all API endpoints with their parameters"
        }
    ]
)
response = await manager.scrape(request)
```

### Tier 4: Playwright + Firecrawl

**Best For:** Maximum control, authentication, complex workflows

```python
request = UnifiedScrapingRequest(
    url="https://authenticated-site.com",
    tier="playwright",
    interaction_required=True,
    custom_actions=[
        {"type": "fill", "selector": "#username", "value": "demo"},
        {"type": "fill", "selector": "#password", "value": "demo"},
        {"type": "click", "selector": "#login"},
        {"type": "wait_for_selector", "selector": ".dashboard"},
        {"type": "extract_content", "selector": ".documentation"}
    ]
)
response = await manager.scrape(request)
```

## Error Handling and Resilience

### Circuit Breaker Pattern

Automatic tier health management with fallback routing:

```python
# Circuit breaker configuration per tier
circuit_config = {
    "failure_threshold": 5,      # Failures before opening circuit
    "timeout_seconds": 60,       # Circuit open duration
    "half_open_max_calls": 3     # Test calls in half-open state
}
```

### Intelligent Fallback Strategies

```python
# Automatic fallback hierarchy
fallback_chains = {
    "lightweight": ["crawl4ai", "crawl4ai_enhanced"],
    "crawl4ai": ["crawl4ai_enhanced", "browser_use"],
    "browser_use": ["playwright", "firecrawl"],
    "playwright": ["firecrawl"]
}
```

### Comprehensive Error Context

```python
error_response = {
    "success": False,
    "error": "Tier execution failed: Connection timeout",
    "tier_used": "none",
    "failed_tiers": ["lightweight", "crawl4ai"],
    "execution_time_ms": 15000,
    "fallback_attempted": True,
    "error_context": {
        "last_tier_attempted": "crawl4ai",
        "failure_reason": "timeout",
        "retry_count": 2,
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

## Performance Optimization

### Connection Pooling and Resource Management

```python
# Optimized resource management
async def efficient_bulk_scraping(urls: list[str]):
    manager = UnifiedBrowserManager(config)
    await manager.initialize()
    
    try:
        # Batch process with concurrency control
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def scrape_single(url):
            async with semaphore:
                return await manager.scrape(url)
        
        tasks = [scrape_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    finally:
        await manager.cleanup()  # Always cleanup resources
```

### Cache Optimization

```python
# Cache performance monitoring
cache_stats = manager._browser_cache.get_stats()
# Returns:
{
    "hit_rate": 0.78,
    "total_entries": 1250,
    "total_size_mb": 45.2,
    "avg_ttl_seconds": 3600,
    "eviction_count": 23
}

# Cache warming for high-traffic URLs
priority_urls = ["https://docs.main.com", "https://api.main.com"]
for url in priority_urls:
    await manager.scrape(url)  # Populate cache
```

## Configuration Reference

### Environment Variables

```bash
# Core Configuration
UNIFIED_BROWSER_CACHE_ENABLED=true
UNIFIED_BROWSER_MONITORING_ENABLED=true
UNIFIED_BROWSER_RATE_LIMITING_ENABLED=true

# Cache Configuration
BROWSER_CACHE_TTL=3600
BROWSER_DYNAMIC_TTL=300
BROWSER_STATIC_TTL=86400

# Rate Limiting
LIGHTWEIGHT_REQUESTS_PER_MINUTE=300
CRAWL4AI_REQUESTS_PER_MINUTE=120
BROWSER_USE_REQUESTS_PER_MINUTE=30

# Monitoring
MONITORING_ERROR_THRESHOLD=0.1
MONITORING_RESPONSE_TIME_THRESHOLD=10000
MONITORING_ALERT_COOLDOWN=300

# API Keys
OPENAI_API_KEY=sk-...         # For AI-powered tiers
ANTHROPIC_API_KEY=sk-ant-...  # Alternative AI provider
FIRECRAWL_API_KEY=fc-...      # For Firecrawl fallback
```

### Complete Configuration Example

```python
from src.config import UnifiedConfig
from src.services.browser.unified_manager import UnifiedBrowserManager

# Load configuration
config = UnifiedConfig()

# Initialize with custom settings
config.cache.enable_browser_cache = True
config.cache.browser_cache_ttl = 3600
config.performance.enable_monitoring = True
config.performance.enable_rate_limiting = True

# Create and initialize manager
manager = UnifiedBrowserManager(config)
await manager.initialize()

# Production-ready scraping
try:
    response = await manager.scrape(
        url="https://production-docs.com",
        tier="auto",  # Let system choose optimal tier
        extract_metadata=True
    )
    
    if response.success:
        print(f"Success! Used {response.tier_used} tier")
        print(f"Content length: {response.content_length}")
        print(f"Quality score: {response.quality_score:.2f}")
        print(f"Execution time: {response.execution_time_ms:.1f}ms")
    else:
        print(f"Failed: {response.error}")
        print(f"Failed tiers: {response.failed_tiers}")
        
finally:
    await manager.cleanup()
```

## Best Practices

### 1. Tier Selection Strategy

```python
# Let the system choose for optimal performance
response = await manager.scrape(url=url, tier="auto")  # Recommended

# Use specific tiers only when needed
response = await manager.scrape(url=url, tier="browser_use")  # Complex interactions only
```

### 2. Error Handling Patterns

```python
async def robust_scraping(url: str) -> dict:
    """Production-ready scraping with comprehensive error handling."""
    try:
        response = await manager.scrape(url)
        
        if not response.success:
            # Log detailed error context
            logger.error(f"Scraping failed for {url}: {response.error}")
            logger.error(f"Failed tiers: {response.failed_tiers}")
            
            # Implement custom fallback logic if needed
            if "authentication" in response.error.lower():
                return await handle_auth_required(url)
            
        return response
        
    except Exception as e:
        logger.exception(f"Unexpected error scraping {url}")
        return UnifiedScrapingResponse(
            success=False,
            error=str(e),
            url=url,
            tier_used="none",
            execution_time_ms=0,
            content_length=0
        )
```

### 3. Performance Monitoring

```python
# Regular performance monitoring
async def monitor_system_health():
    status = manager.get_system_status()
    
    if status["overall_success_rate"] < 0.9:
        logger.warning("System performance degraded")
        
    if status["cache_stats"]["hit_rate"] < 0.7:
        logger.info("Consider cache optimization")
        
    # Check tier-specific metrics
    for tier, metrics in status["tier_metrics"].items():
        if metrics["success_rate"] < 0.8:
            logger.warning(f"Tier {tier} performance issues")
```

### 4. Resource Management

```python
# Always use context managers or try/finally
async def safe_scraping_session():
    manager = UnifiedBrowserManager(config)
    
    try:
        await manager.initialize()
        
        # Perform scraping operations
        results = []
        for url in urls:
            result = await manager.scrape(url)
            results.append(result)
            
        return results
        
    finally:
        await manager.cleanup()  # Essential for resource cleanup
```

This comprehensive 5-tier browser automation system provides production-ready web scraping with intelligent optimization, comprehensive monitoring, and robust error handling.