# Browser Automation API Documentation

This document provides comprehensive API documentation for the browser automation system, focusing on complex methods and their usage patterns.

## Overview

The browser automation system implements a three-tier fallback hierarchy:

1. **Crawl4AI** - High-performance bulk processing (primary)
2. **browser-use** - AI-powered intelligent automation with multi-LLM support (fallback)
3. **Playwright** - Direct browser control (final fallback)

## Core Components

### AutomationRouter

The central orchestrator that manages adapter selection and fallback behavior.

#### `async scrape(url: str, actions: list = None, instructions: list = None, **kwargs) -> dict`

Primary scraping method with intelligent routing and fallback.

**Parameters:**

- `url` (str): Target URL to scrape
- `actions` (list, optional): Structured actions for Playwright adapter
- `instructions` (list, optional): Natural language instructions for browser-use
- `**kwargs`: Additional adapter-specific configuration

**Returns:**

- `dict`: Standardized result format:

  ```python
  {
      "success": bool,
      "url": str,
      "content": str,
      "html": str,
      "title": str,
      "metadata": {
          "extraction_method": str,
          "processing_time_ms": float,
          "adapter_used": str,
          "fallback_count": int
      },
      "error": str  # Only if success=False
  }
  ```

**Usage Examples:**

```python
# Basic scraping with automatic adapter selection
result = await router.scrape("https://docs.example.com")

# AI-powered scraping with natural language instructions
result = await router.scrape(
    "https://complex-spa.com",
    instructions=[
        "Wait for the page to load completely",
        "Extract all documentation content",
        "Take a screenshot for verification"
    ]
)

# Structured browser actions for precise control
result = await router.scrape(
    "https://login-required.com",
    actions=[
        {"type": "fill", "selector": "input[name='username']", "value": "demo"},
        {"type": "fill", "selector": "input[name='password']", "value": "demo"},
        {"type": "click", "selector": "button[type='submit']"},
        {"type": "wait_for_selector", "selector": ".dashboard", "timeout": 5000}
    ]
)
```

**Error Handling:**

- Automatic fallback between adapters on failure
- Comprehensive error logging with adapter context
- Graceful degradation with partial results when possible

#### `async force_adapter(adapter_name: str, *args, **kwargs) -> dict`

Force usage of a specific adapter, bypassing routing logic.

**Parameters:**

- `adapter_name` (str): One of 'crawl4ai', 'browser_use', 'playwright'
- `*args, **kwargs`: Arguments passed directly to adapter

**Returns:**

- `dict`: Same format as `scrape()` method

**Usage:**

```python
# Force browser-use for complex interactions
result = await router.force_adapter(
    "browser_use",
    "https://complex-site.com",
    ["Navigate to settings", "Enable notifications"]
)
```

#### `get_performance_metrics() -> dict`

Retrieve detailed performance metrics for monitoring and optimization.

**Returns:**

```python
{
    "total_requests": int,
    "adapter_usage": {
        "crawl4ai": {"count": int, "success_rate": float},
        "browser_use": {"count": int, "success_rate": float},
        "playwright": {"count": int, "success_rate": float}
    },
    "avg_response_times": {
        "crawl4ai": float,  # milliseconds
        "browser_use": float,
        "playwright": float
    },
    "fallback_frequency": float,  # percentage
    "error_breakdown": dict
}
```

### BrowserUseAdapter

AI-powered browser automation using natural language tasks with multi-LLM support.

#### `async scrape(url: str, task: str, timeout: int = 30000) -> dict`

Execute AI-driven automation based on natural language task descriptions.

**Key Features:**

- **Multi-LLM Support**: OpenAI, Anthropic, Gemini, and local models
- **Cost Optimization**: Default GPT-4o-mini for routine tasks, GPT-4o for complex interactions
- **Self-Correcting Behavior**: AI learns from mistakes and adapts execution
- **Natural Language Tasks**: Describe what you want, not how to do it
- **Python-Native**: No TypeScript dependencies, fully async

**Task Examples:**

1. **Simple Content Extraction:**

   ```python
   task = "Extract all documentation content including code examples"
   result = await adapter.scrape("https://docs.example.com", task)
   ```

2. **Complex Interactive Tasks:**

   ```python
   task = """
   Navigate to the API section, expand any collapsed code examples,
   extract all endpoint documentation with their parameters,
   and collect any authentication requirements.
   """
   result = await adapter.scrape("https://api-docs.example.com", task)
   ```

3. **Multi-Step Documentation Scraping:**

   ```python
   task = """
   1. Wait for the page to fully load
   2. Handle any cookie banners by dismissing them
   3. Navigate to the getting started guide
   4. Extract all step-by-step instructions
   5. Find and expand any collapsed sections
   6. Collect all code snippets and examples
   """
   result = await adapter.scrape("https://docs.example.com/guide", task)
   ```

**LLM Provider Configuration:**

```python
# OpenAI (default)
config = {
    "llm_provider": "openai",
    "model": "gpt-4o-mini",  # Cost-optimized
    "headless": True,
    "max_steps": 20,
    "max_retries": 3
}

# Anthropic
config = {
    "llm_provider": "anthropic", 
    "model": "claude-3-haiku-20240307",
    "headless": True
}

# Gemini
config = {
    "llm_provider": "gemini",
    "model": "gemini-pro",
    "headless": True
}
```

#### `async scrape_with_instructions(url: str, instructions: list, timeout: int = 30000) -> dict`

Compatibility method that converts instruction lists to natural language tasks.

#### `async test_ai_capabilities(test_url: str = "https://example.com") -> dict`

Comprehensive AI capability testing for validation and benchmarking.

**Returns:**

```python
{
    "success": bool,
    "test_url": str,
    "task_description": str,
    "execution_time_ms": float,
    "content_length": int,
    "ai_insights": dict,
    "metadata": dict,
    "error": str  # Only if success=False
}
```

### PlaywrightAdapter

Direct browser control with structured actions for precise automation.

#### Action Schema System

The adapter uses a comprehensive Pydantic-based action schema for type safety and validation:

```python
# Click actions with enhanced targeting
click_action = {
    "type": "click",
    "selector": "button.primary",
    "button": "left",  # left, right, middle
    "click_count": 1,
    "modifiers": ["Shift"],  # Alt, Control, Meta, Shift
    "position": {"x": 10, "y": 5},  # Relative to element
    "force": False,  # Bypass actionability checks
    "no_wait_after": False,  # Skip post-action waiting
    "timeout": 30000,
    "trial": False  # Test without performing action
}

# Form filling with validation
fill_action = {
    "type": "fill",
    "selector": "input[name='email']",
    "value": "user@example.com",
    "force": False,
    "no_wait_after": False,
    "timeout": 30000
}

# Advanced waiting strategies
wait_action = {
    "type": "wait_for_selector",
    "selector": ".dynamic-content",
    "state": "visible",  # attached, detached, visible, hidden
    "timeout": 10000
}

# JavaScript evaluation with result capture
evaluate_action = {
    "type": "evaluate",
    "expression": "document.querySelector('.data').textContent",
    "return_value": True  # Capture return value
}
```

#### Complex Automation Patterns

**Multi-Step Form Handling:**

```python
form_actions = [
    {"type": "fill", "selector": "input[name='username']", "value": "demo"},
    {"type": "fill", "selector": "input[name='password']", "value": "secure123"},
    {"type": "select", "selector": "select[name='role']", "value": "admin"},
    {"type": "check", "selector": "input[name='remember']"},
    {"type": "click", "selector": "button[type='submit']"},
    {"type": "wait_for_selector", "selector": ".success-message", "timeout": 5000}
]
```

**Dynamic Content Handling:**

```python
dynamic_actions = [
    {"type": "wait_for_load_state", "state": "networkidle"},
    {"type": "evaluate", "expression": "window.APP_READY", "return_value": True},
    {"type": "wait_for_selector", "selector": "[data-testid='content-loaded']"},
    {"type": "scroll", "selector": "body", "position": {"top": 0, "left": 0}},
    {"type": "screenshot", "full_page": True}
]
```

**SPA Navigation Pattern:**

```python
spa_actions = [
    {"type": "click", "selector": "nav a[href='/api-docs']"},
    {"type": "wait_for_url", "url": "**/api-docs", "timeout": 10000},
    {"type": "wait_for_selector", "selector": ".api-endpoint", "timeout": 15000},
    {"type": "evaluate", "expression": "document.querySelectorAll('.api-endpoint').length", "return_value": True}
]
```

### Crawl4AIAdapter

High-performance bulk processing adapter optimized for documentation sites.

#### `async scrape(url: str, **kwargs) -> dict`

Optimized scraping with intelligent content extraction and caching.

**Performance Features:**

- **Intelligent Caching**: Content-based cache keys with 80%+ hit rates
- **Batch Processing**: Optimized for bulk documentation processing
- **Smart Content Detection**: Automatic article/documentation content identification
- **Link Discovery**: Intelligent sitemap and link following

**Configuration Options:**

```python
config = {
    "max_pages": 100,
    "concurrent_limit": 10,
    "cache_duration": 3600,  # seconds
    "content_filters": {
        "min_content_length": 500,
        "exclude_patterns": ["*/admin/*", "*/login/*"],
        "include_patterns": ["*/docs/*", "*/api/*"]
    },
    "extraction_strategy": "article_aware"  # article_aware, full_page, custom
}

result = await adapter.scrape("https://docs.example.com", **config)
```

## Error Handling & Resilience

### Circuit Breaker Pattern

All adapters implement circuit breaker patterns for external service failures:

```python
# Automatic fallback configuration
circuit_config = {
    "failure_threshold": 5,     # Failures before opening circuit
    "timeout": 30,              # Circuit open duration (seconds)
    "expected_exception": (ConnectionError, TimeoutError)
}
```

### Retry Strategies

**Exponential Backoff:**

```python
retry_config = {
    "max_retries": 3,
    "base_delay": 1.0,          # Initial delay (seconds)
    "max_delay": 10.0,          # Maximum delay cap
    "backoff_factor": 2.0,      # Exponential multiplier
    "jitter": True              # Add randomization
}
```

### Error Context and Logging

All methods provide rich error context for debugging:

```python
error_result = {
    "success": False,
    "error": "Detailed error message",
    "error_context": {
        "adapter": "browser_use",
        "operation": "page_navigation",
        "url": "https://example.com",
        "instruction": "click on submit button",
        "timestamp": "2024-01-15T10:30:00Z",
        "retry_count": 2,
        "total_time_ms": 15000
    },
    "debugging_info": {
        "last_screenshot": "base64_encoded_image",
        "page_html": "current_page_html",
        "console_logs": ["log1", "log2"],
        "network_activity": [...]
    }
}
```

## Performance Optimization

### Connection Pooling

All adapters use connection pooling for optimal performance:

```python
pool_config = {
    "max_connections": 100,
    "max_keepalive_connections": 20,
    "keepalive_expiry": 30.0,
    "timeout": {"connect": 10.0, "read": 30.0, "write": 10.0}
}
```

### Memory Management

**Resource Cleanup Patterns:**

```python
async def safe_scrape_pattern(url: str):
    adapter = None
    try:
        adapter = BrowserUseAdapter(config)
        await adapter.initialize()
        result = await adapter.scrape(url, instructions)
        return result
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if adapter:
            await adapter.cleanup()  # Always cleanup resources
```

### Monitoring and Metrics

**Health Check Implementation:**

```python
health_status = await adapter.health_check()
# Returns:
{
    "healthy": bool,
    "status": "operational|degraded|timeout|error",
    "message": str,
    "response_time_ms": float,
    "test_url": str,
    "available": bool,
    "capabilities": dict
}
```

**Performance Monitoring:**

```python
metrics = router.get_performance_metrics()
# Monitor:
# - Success rates per adapter
# - Average response times
# - Fallback frequency
# - Error patterns
# - Resource utilization
```

## Best Practices

### 1. Adapter Selection Strategy

```python
# Use routing for automatic optimization
result = await router.scrape(url)  # Preferred

# Force specific adapters only when needed
result = await router.force_adapter("browser_use", url, task)
```

### 2. Task Design for Browser-Use

```python
# Good: Clear and specific natural language tasks
task = """
Navigate to the documentation page, expand any collapsed sections,
and extract all API endpoint information including parameters,
examples, and response formats.
"""

# Good: Simple extraction task
task = "Extract all code examples and their explanations from this tutorial page"

# Avoid: Vague or overly complex
task = "Do something with the page"  # Too vague
task = "Analyze everything and give me all information"  # Too broad
```

### 3. Error Handling Patterns

```python
async def robust_scraping(url: str):
    try:
        result = await router.scrape(url)
        if not result["success"]:
            # Log detailed error context
            logger.error(f"Scraping failed: {result['error']}")
            # Implement fallback or retry logic
            return await fallback_scraping(url)
        return result
    except Exception as e:
        logger.exception(f"Unexpected error scraping {url}")
        return {"success": False, "error": str(e)}
```

### 4. Performance Optimization

```python
# Batch related operations
urls = ["url1", "url2", "url3"]
tasks = [router.scrape(url) for url in urls]
results = await asyncio.gather(*tasks, return_exceptions=True)

# Use appropriate timeouts
result = await router.scrape(url, timeout=60000)  # For complex pages

# Monitor and adjust based on metrics
metrics = router.get_performance_metrics()
if metrics["fallback_frequency"] > 0.3:  # 30% fallback rate
    # Consider adjusting routing rules or adapter configuration
    pass
```

## Configuration Reference

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...  # For AI-powered features

# Optional
PLAYWRIGHT_BROWSERS_PATH=/opt/playwright  # Custom browser path
BROWSER_USE_LLM_PROVIDER=openai          # LLM provider (openai, anthropic, gemini)
BROWSER_USE_MODEL=gpt-4o-mini             # Model for cost optimization
CRAWL4AI_CACHE_DIR=/tmp/crawl_cache       # Cache directory
```

### Adapter-Specific Configuration

```python
config = {
    "browser_use": {
        "llm_provider": "openai",
        "model": "gpt-4o-mini",
        "headless": True,
        "timeout": 30000,
        "max_retries": 3,
        "max_steps": 20,
        "debug_screenshots": False
    },
    "playwright": {
        "headless": True,
        "browser_type": "chromium",
        "viewport": {"width": 1280, "height": 720},
        "timeout": 30000,
        "user_agent": "custom-user-agent"
    },
    "crawl4ai": {
        "max_pages": 100,
        "concurrent_limit": 10,
        "cache_duration": 3600,
        "content_strategy": "article_aware"
    }
}
```

This comprehensive API documentation provides the foundation for effective browser automation integration and advanced usage patterns.
