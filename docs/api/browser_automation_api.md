# Browser Automation API Documentation

This document provides comprehensive API documentation for the browser automation system, focusing on complex methods and their usage patterns.

## Overview

The browser automation system implements a three-tier fallback hierarchy:

1. **Crawl4AI** - High-performance bulk processing (primary)
2. **Stagehand** - AI-powered intelligent automation (fallback)
3. **Playwright** - Direct browser control (final fallback)

## Core Components

### AutomationRouter

The central orchestrator that manages adapter selection and fallback behavior.

#### `async scrape(url: str, actions: list = None, instructions: list = None, **kwargs) -> dict`

Primary scraping method with intelligent routing and fallback.

**Parameters:**

- `url` (str): Target URL to scrape
- `actions` (list, optional): Structured actions for Playwright adapter
- `instructions` (list, optional): Natural language instructions for Stagehand
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

- `adapter_name` (str): One of 'crawl4ai', 'stagehand', 'playwright'
- `*args, **kwargs`: Arguments passed directly to adapter

**Returns:**

- `dict`: Same format as `scrape()` method

**Usage:**

```python
# Force Stagehand for complex interactions
result = await router.force_adapter(
    "stagehand",
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
        "stagehand": {"count": int, "success_rate": float},
        "playwright": {"count": int, "success_rate": float}
    },
    "avg_response_times": {
        "crawl4ai": float,  # milliseconds
        "stagehand": float,
        "playwright": float
    },
    "fallback_frequency": float,  # percentage
    "error_breakdown": dict
}
```

### StagehandAdapter

AI-powered browser automation using natural language instructions.

#### `async scrape(url: str, instructions: list, timeout: int = 30000) -> dict`

Execute AI-driven automation based on natural language instructions.

**Key Features:**

- **Intelligent Instruction Parsing**: Automatically categorizes instructions (click, type, extract, etc.)
- **Adaptive Execution**: AI adapts to UI changes and dynamic content
- **Content Extraction**: Combines structured extractions with final content sweep
- **Screenshot Capture**: Automatic visual documentation of key steps

**Instruction Categories:**

1. **Navigation & Interaction:**

   ```python
   instructions = [
       "click on the login button",
       "type 'username' in the email field",
       "scroll to the bottom of the page",
       "wait for 3 seconds"
   ]
   ```

2. **Content Extraction:**

   ```python
   instructions = [
       "extract the main article content",
       "find all code examples on the page",
       "get the page title and metadata"
   ]
   ```

3. **Documentation & Verification:**

   ```python
   instructions = [
       "take a screenshot of the current state",
       "verify the page loaded correctly",
       "capture any error messages"
   ]
   ```

**Advanced Usage Pattern:**

```python
# Multi-stage documentation scraping
complex_instructions = [
    "Navigate to the API documentation section",
    "Extract all endpoint descriptions",
    "Click on each code example to expand it",
    "Screenshot each expanded example",
    "Extract the complete code snippets",
    "Navigate to the authentication section",
    "Extract authentication requirements"
]

result = await adapter.scrape(
    "https://api-docs.example.com",
    complex_instructions,
    timeout=60000  # Extended timeout for complex operations
)

# Access structured results
extractions = result["extraction_results"]
screenshots = result["screenshots"]
ai_insights = result["ai_insights"]
```

#### `async test_ai_capabilities(test_url: str = "https://example.com") -> dict`

Comprehensive AI capability testing for validation and benchmarking.

**Returns:**

```python
{
    "success": bool,
    "test_url": str,
    "instructions_count": int,
    "execution_time_ms": float,
    "extractions_count": int,
    "screenshots_count": int,
    "content_length": int,
    "ai_insights": dict,
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
        "adapter": "stagehand",
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
        adapter = StagehandAdapter(config)
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
result = await router.force_adapter("stagehand", url, instructions)
```

### 2. Instruction Design for Stagehand

```python
# Good: Specific and actionable
instructions = [
    "Click the 'Get Started' button in the hero section",
    "Wait for the tutorial page to load",
    "Extract the step-by-step instructions"
]

# Avoid: Vague or overly complex
instructions = [
    "Do something with the page",  # Too vague
    "Navigate through all sections and extract everything"  # Too complex
]
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
STAGEHAND_MODEL=ollama/llama2             # Custom AI model
CRAWL4AI_CACHE_DIR=/tmp/crawl_cache       # Cache directory
```

### Adapter-Specific Configuration

```python
config = {
    "stagehand": {
        "model": "ollama/llama2",
        "headless": True,
        "viewport": {"width": 1920, "height": 1080},
        "enable_caching": True,
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
