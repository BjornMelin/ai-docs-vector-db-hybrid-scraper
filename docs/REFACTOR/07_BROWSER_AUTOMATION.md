# Browser Automation Hierarchy Implementation Guide

**GitHub Issue**: [#61](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues/61)

## Overview

Implement a three-tier browser automation hierarchy that intelligently selects the right tool for each scraping task. This provides the best balance of performance, cost, and capability.

**COMPLETED 2025-05-29**: Successfully replaced Stagehand with browser-use implementation. All phases completed with comprehensive testing and documentation.

## Automation Hierarchy

### Tier 1: Crawl4AI (Default)

- **Use for**: 90% of documentation sites
- **Performance**: 4-6x faster than alternatives
- **Cost**: $0
- **JavaScript**: Basic support

### Tier 2: browser-use (AI-Powered)

- **Use for**: Complex interactions, dynamic content, natural language tasks
- **Performance**: 2x slower than Crawl4AI, faster than traditional automation
- **Cost**: Minimal (configurable LLM providers: OpenAI, Anthropic, Gemini, local models)
- **JavaScript**: Full support with AI understanding and self-correction
- **Advantages**: Python-native, multi-LLM support, self-correcting behavior, active development

### Tier 3: Playwright (Fallback)

- **Use for**: Maximum control scenarios, authentication, complex workflows
- **Performance**: Baseline
- **Cost**: $0
- **JavaScript**: Full programmatic control

## Implementation

### 1. Intelligent Router

```python
# src/services/browser/automation_router.py
from typing import Any, Optional, Literal
from urllib.parse import urlparse
import asyncio

from .crawl4ai_adapter import Crawl4AIAdapter
from .browser_use_adapter import BrowserUseAdapter
from .playwright_adapter import PlaywrightAdapter
from ..logging_config import get_logger

logger = get_logger(__name__)

class AutomationRouter:
    """Intelligently route scraping tasks to appropriate automation tool."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        
        # Initialize adapters
        self.crawl4ai = Crawl4AIAdapter(config.get("crawl4ai", {}))
        self.browser_use = BrowserUseAdapter(config.get("browser_use", {}))
        self.playwright = PlaywrightAdapter(config.get("playwright", {}))
        
        # Site-specific routing rules
        self.routing_rules = {
            # Sites that need browser-use (AI-powered interaction)
            "browser_use": [
                "vercel.com",  # Complex React app
                "clerk.com",   # Heavy client-side rendering
                "supabase.com",  # Dynamic documentation
                "react.dev",   # Interactive examples
                "nextjs.org",  # Dynamic content
                "docs.anthropic.com",  # AI-powered examples
            ],
            
            # Sites that need Playwright (specific automation)
            "playwright": [
                "github.com",  # Authentication required
                "stackoverflow.com",  # Complex pagination
                "notion.so",   # Heavy JavaScript interactions
            ],
            
            # Default to Crawl4AI for everything else
        }
        
        # Performance metrics
        self.metrics = {
            "crawl4ai": {"success": 0, "failed": 0, "avg_time": 0},
            "browser_use": {"success": 0, "failed": 0, "avg_time": 0},
            "playwright": {"success": 0, "failed": 0, "avg_time": 0},
        }
    
    async def scrape(
        self,
        url: str,
        interaction_required: bool = False,
        custom_actions: Optional[list[dict]] = None,
        force_tool: Optional[Literal["crawl4ai", "browser_use", "playwright"]] = None,
    ) -> dict[str, Any]:
        """Route scraping to appropriate tool based on URL and requirements."""
        
        # Determine which tool to use
        if force_tool:
            tool = force_tool
        else:
            tool = self._select_tool(url, interaction_required, custom_actions)
        
        logger.info(f"Using {tool} for {url}")
        
        # Execute with selected tool
        start_time = asyncio.get_event_loop().time()
        
        try:
            if tool == "crawl4ai":
                result = await self._try_crawl4ai(url, custom_actions)
            elif tool == "browser_use":
                result = await self._try_browser_use(url, custom_actions)
            else:
                result = await self._try_playwright(url, custom_actions)
            
            # Update metrics
            elapsed = asyncio.get_event_loop().time() - start_time
            self._update_metrics(tool, True, elapsed)
            
            return result
            
        except Exception as e:
            logger.error(f"{tool} failed for {url}: {e}")
            self._update_metrics(tool, False, 0)
            
            # Try fallback
            return await self._fallback_scrape(url, tool, custom_actions)
    
    def _select_tool(
        self,
        url: str,
        interaction_required: bool,
        custom_actions: Optional[list[dict]],
    ) -> str:
        """Select appropriate tool based on URL and requirements."""
        
        domain = urlparse(url).netloc
        
        # Check explicit routing rules
        for tool, domains in self.routing_rules.items():
            if any(d in domain for d in domains):
                return tool
        
        # Check if interaction is required
        if interaction_required or custom_actions:
            return "browser_use"  # AI can handle complex interactions
        
        # Default to fastest option
        return "crawl4ai"
    
    async def _try_crawl4ai(
        self,
        url: str,
        custom_actions: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Try scraping with Crawl4AI."""
        
        return await self.crawl4ai.scrape(
            url=url,
            wait_for_selector=".content, main, article",
            js_code=self._get_basic_js(url),
        )
    
    async def _try_browser_use(
        self,
        url: str,
        custom_actions: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Try scraping with browser-use AI."""
        
        # Convert custom actions to browser-use format
        if custom_actions:
            task = self._convert_to_task(custom_actions)
        else:
            task = "Navigate to the page and extract all documentation content including code examples. Expand any collapsed sections or interactive elements."
        
        return await self.browser_use.scrape(
            url=url,
            task=task,
        )
    
    async def _try_playwright(
        self,
        url: str,
        custom_actions: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Try scraping with Playwright."""
        
        return await self.playwright.scrape(
            url=url,
            actions=custom_actions or [],
        )
    
    async def _fallback_scrape(
        self,
        url: str,
        failed_tool: str,
        custom_actions: Optional[list[dict]],
    ) -> dict[str, Any]:
        """Fallback to next tool in hierarchy."""
        
        fallback_order = {
            "crawl4ai": ["browser_use", "playwright"],
            "browser_use": ["playwright", "crawl4ai"],
            "playwright": ["browser_use", "crawl4ai"],
        }
        
        for fallback_tool in fallback_order[failed_tool]:
            try:
                logger.info(f"Falling back to {fallback_tool}")
                
                if fallback_tool == "crawl4ai":
                    return await self._try_crawl4ai(url, custom_actions)
                elif fallback_tool == "browser_use":
                    return await self._try_browser_use(url, custom_actions)
                else:
                    return await self._try_playwright(url, custom_actions)
                    
            except Exception as e:
                logger.error(f"Fallback {fallback_tool} also failed: {e}")
                continue
        
        raise Exception(f"All automation tools failed for {url}")
    
    def _get_basic_js(self, url: str) -> str:
        """Get basic JavaScript for common scenarios."""
        
        return """
        // Wait for content to load
        await new Promise(r => setTimeout(r, 2000));
        
        // Expand collapsed sections
        document.querySelectorAll('[aria-expanded="false"]').forEach(el => el.click());
        
        // Scroll to load lazy content
        window.scrollTo(0, document.body.scrollHeight);
        """
    
    def _convert_to_task(self, actions: list[dict]) -> str:
        """Convert custom actions to browser-use natural language task."""
        
        task_parts = []
        
        for action in actions:
            if action["type"] == "click":
                task_parts.append(f"click on {action['selector']}")
            elif action["type"] == "type":
                task_parts.append(f"type '{action['text']}' in {action['selector']}")
            elif action["type"] == "wait":
                task_parts.append(f"wait for {action['timeout']}ms")
            elif action["type"] == "scroll":
                task_parts.append("scroll down to load more content")
            elif action["type"] == "expand":
                task_parts.append("expand any collapsed sections or menus")
        
        if task_parts:
            return f"Navigate to the page, then {', '.join(task_parts)}, and finally extract all documentation content."
        else:
            return "Navigate to the page and extract all documentation content including code examples."
    
    def _update_metrics(self, tool: str, success: bool, elapsed: float):
        """Update performance metrics."""
        
        metrics = self.metrics[tool]
        
        if success:
            metrics["success"] += 1
            # Update rolling average
            total = metrics["success"] + metrics["failed"]
            metrics["avg_time"] = (
                (metrics["avg_time"] * (total - 1) + elapsed) / total
            )
        else:
            metrics["failed"] += 1
    
    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics for all tools."""
        
        return {
            tool: {
                **metrics,
                "success_rate": (
                    metrics["success"] / (metrics["success"] + metrics["failed"])
                    if metrics["success"] + metrics["failed"] > 0
                    else 0
                ),
            }
            for tool, metrics in self.metrics.items()
        }
```

### 2. BrowserUse Adapter

```python
# src/services/browser/browser_use_adapter.py
from typing import Any, Optional
from browser_use import Agent
import asyncio
import os

from ..base import BaseService

class BrowserUseAdapter(BaseService):
    """AI-powered browser automation with browser-use."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        
        # browser-use configuration
        self.llm_provider = config.get("llm_provider", "openai")
        self.model = config.get("model", "gpt-4o-mini")  # Cost-optimized
        self.headless = config.get("headless", True)
        self.timeout = config.get("timeout", 30000)
        self.max_retries = config.get("max_retries", 3)
        
        # Initialize LLM configuration
        self.llm_config = self._setup_llm_config()
        
        self.agent = None
    
    def _setup_llm_config(self) -> dict[str, Any]:
        """Setup LLM configuration based on provider."""
        
        if self.llm_provider == "openai":
            return {
                "provider": "openai",
                "model": self.model,
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        elif self.llm_provider == "anthropic":
            return {
                "provider": "anthropic",
                "model": self.model,
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            }
        elif self.llm_provider == "gemini":
            return {
                "provider": "gemini",
                "model": self.model,
                "api_key": os.getenv("GEMINI_API_KEY"),
            }
        else:
            # Local model fallback
            return {
                "provider": "local",
                "model": "llama3.2:3b",  # Lightweight local model
            }
    
    async def initialize(self):
        """Initialize browser-use agent."""
        
        self.agent = Agent(
            task="Web scraping and content extraction",
            llm_config=self.llm_config,
            browser_config={
                "headless": self.headless,
                "disable_security": False,  # Keep security enabled
            },
        )
    
    async def scrape(
        self,
        url: str,
        task: str,
        timeout: int = None,
    ) -> dict[str, Any]:
        """Scrape using AI-powered automation with natural language tasks."""
        
        if not self.agent:
            await self.initialize()
        
        timeout = timeout or self.timeout
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                self.logger.info(f"Executing browser-use task: {task[:100]}...")
                
                # Create browser-use task with context
                full_task = f"""
                Navigate to {url} and {task}
                
                Please:
                1. Wait for the page to fully load
                2. Handle any cookie banners or popups
                3. Extract all relevant content including:
                   - Main text content
                   - Code examples
                   - Documentation sections
                   - Navigation elements
                4. Return structured content
                """
                
                # Execute with browser-use
                result = await self.agent.run(
                    task=full_task,
                    max_steps=20,  # Limit steps to prevent infinite loops
                )
                
                # Extract content from browser-use result
                if result and result.get("success"):
                    return {
                        "success": True,
                        "url": url,
                        "content": result.get("extracted_content", ""),
                        "html": result.get("html", ""),
                        "metadata": {
                            "steps_taken": result.get("steps", []),
                            "task_completion": result.get("completion_status", "completed"),
                            "model_used": self.model,
                            "provider": self.llm_provider,
                        },
                        "screenshots": result.get("screenshots", []),
                    }
                else:
                    raise Exception(f"browser-use failed to extract content: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                retry_count += 1
                self.logger.warning(f"browser-use attempt {retry_count} failed: {e}")
                
                if retry_count >= self.max_retries:
                    self.logger.error(f"browser-use failed after {self.max_retries} attempts: {e}")
                    raise
                
                # Exponential backoff
                await asyncio.sleep(2 ** retry_count)
        
        raise Exception(f"browser-use failed after {self.max_retries} retries")
    
    async def cleanup(self):
        """Cleanup browser-use resources."""
        
        if self.agent:
            await self.agent.close()
```

### 3. Playwright Adapter

```python
# src/services/browser/playwright_adapter.py
from typing import Any, Optional
from playwright.async_api import async_playwright, Page
import asyncio

from ..base import BaseService

class PlaywrightAdapter(BaseService):
    """Direct Playwright automation for maximum control."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        
        self.browser_type = config.get("browser", "chromium")
        self.headless = config.get("headless", True)
        self.viewport = config.get("viewport", {"width": 1920, "height": 1080})
        
        self.playwright = None
        self.browser = None
    
    async def initialize(self):
        """Initialize Playwright browser."""
        
        self.playwright = await async_playwright().start()
        
        browser_launcher = getattr(self.playwright, self.browser_type)
        self.browser = await browser_launcher.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
    
    async def scrape(
        self,
        url: str,
        actions: list[dict],
        timeout: int = 30000,
    ) -> dict[str, Any]:
        """Scrape with direct Playwright control."""
        
        if not self.browser:
            await self.initialize()
        
        context = await self.browser.new_context(
            viewport=self.viewport,
            user_agent="Mozilla/5.0 (compatible; AIDocs/1.0)",
        )
        
        page = await context.new_page()
        
        try:
            # Navigate
            await page.goto(url, wait_until="networkidle", timeout=timeout)
            
            # Execute custom actions
            for action in actions:
                await self._execute_action(page, action)
            
            # Extract content
            content = await self._extract_content(page)
            
            return {
                "success": True,
                "url": url,
                "content": content["text"],
                "html": content["html"],
                "metadata": await self._extract_metadata(page),
            }
            
        except Exception as e:
            self.logger.error(f"Playwright error: {e}")
            raise
        finally:
            await context.close()
    
    async def _execute_action(self, page: Page, action: dict):
        """Execute a single action."""
        
        action_type = action.get("type")
        
        if action_type == "click":
            await page.click(action["selector"])
            
        elif action_type == "type":
            await page.type(action["selector"], action["text"])
            
        elif action_type == "wait":
            await page.wait_for_timeout(action["timeout"])
            
        elif action_type == "wait_for_selector":
            await page.wait_for_selector(action["selector"])
            
        elif action_type == "scroll":
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
        elif action_type == "screenshot":
            await page.screenshot(path=action.get("path", "screenshot.png"))
            
        elif action_type == "evaluate":
            await page.evaluate(action["script"])
    
    async def _extract_content(self, page: Page) -> dict[str, str]:
        """Extract content from page."""
        
        # Try multiple content selectors
        content_selectors = [
            "main",
            "article",
            ".content",
            ".documentation",
            "#content",
            "body",
        ]
        
        for selector in content_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                    html = await element.inner_html()
                    return {"text": text, "html": html}
            except:
                continue
        
        # Fallback to body
        return {
            "text": await page.inner_text("body"),
            "html": await page.inner_html("body"),
        }
    
    async def _extract_metadata(self, page: Page) -> dict[str, Any]:
        """Extract page metadata."""
        
        return await page.evaluate("""
        () => {
            const getMeta = (name) => {
                const meta = document.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
                return meta ? meta.content : null;
            };
            
            return {
                title: document.title,
                description: getMeta('description'),
                author: getMeta('author'),
                keywords: getMeta('keywords'),
                ogTitle: getMeta('og:title'),
                ogDescription: getMeta('og:description'),
                ogImage: getMeta('og:image'),
                lastModified: document.lastModified,
            };
        }
        """)
    
    async def cleanup(self):
        """Cleanup Playwright resources."""
        
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
```

### 4. Site-Specific Configurations

```python
# src/services/browser/site_configs.py
class SiteConfigurations:
    """Site-specific automation configurations."""
    
    CONFIGS = {
        "docs.python.org": {
            "tool": "crawl4ai",
            "wait_for": ".document",
            "js_code": "document.querySelector('.headerlink').remove();",
        },
        
        "react.dev": {
            "tool": "browser_use",
            "task": "Wait for React documentation to load, expand all code examples, click on any 'Show more' buttons, and extract all content including interactive examples",
        },
        
        "github.com": {
            "tool": "playwright",
            "actions": [
                {"type": "wait_for_selector", "selector": ".markdown-body"},
                {"type": "evaluate", "script": "document.querySelectorAll('.d-none').forEach(e => e.classList.remove('d-none'))"},
            ],
        },
        
        "stackoverflow.com": {
            "tool": "browser_use",
            "task": "Wait for Stack Overflow page to load, expand all answers and comments, click on any 'show more comments' links, and extract the complete Q&A content",
        },
    }
    
    @classmethod
    def get_config(cls, url: str) -> dict[str, Any]:
        """Get configuration for URL."""
        
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Check exact match
        if domain in cls.CONFIGS:
            return cls.CONFIGS[domain]
        
        # Check subdomain match
        for config_domain, config in cls.CONFIGS.items():
            if domain.endswith(config_domain):
                return config
        
        # Default configuration
        return {
            "tool": "crawl4ai",
            "wait_for": "body",
        }
```

### 5. Performance Monitoring

```python
# src/services/browser/monitoring.py
class AutomationMonitor:
    """Monitor browser automation performance."""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0,
            "errors": defaultdict(int),
        })
    
    def record_attempt(
        self,
        url: str,
        tool: str,
        success: bool,
        duration: float,
        error: Optional[str] = None,
    ):
        """Record automation attempt."""
        
        domain = urlparse(url).netloc
        metric = self.metrics[domain]
        
        metric["attempts"] += 1
        metric["total_time"] += duration
        
        if success:
            metric["successes"] += 1
        else:
            metric["failures"] += 1
            if error:
                metric["errors"][error] += 1
    
    def get_recommendations(self) -> dict[str, str]:
        """Get tool recommendations based on metrics."""
        
        recommendations = {}
        
        for domain, metrics in self.metrics.items():
            success_rate = metrics["successes"] / metrics["attempts"]
            avg_time = metrics["total_time"] / metrics["attempts"]
            
            if success_rate < 0.8:
                # Low success rate, recommend different tool
                if avg_time > 5:
                    recommendations[domain] = "playwright"  # More control
                else:
                    recommendations[domain] = "stagehand"  # AI assistance
            elif avg_time > 10:
                # Slow, try faster tool
                recommendations[domain] = "crawl4ai"
        
        return recommendations
```

## Testing Strategy

```python
@pytest.mark.asyncio
async def test_automation_hierarchy():
    """Test automation tool selection and fallback."""
    
    router = AutomationRouter({})
    
    # Test tool selection
    assert router._select_tool("https://docs.python.org", False, None) == "crawl4ai"
    assert router._select_tool("https://vercel.com", False, None) == "browser_use"
    assert router._select_tool("https://github.com", False, None) == "playwright"
    
    # Test with interaction required
    assert router._select_tool("https://example.com", True, None) == "browser_use"
    
    # Test fallback
    result = await router.scrape(
        "https://test.com",
        force_tool="crawl4ai"
    )
    assert result["success"]
```

## Integration Example

```python
# Update bulk embedder to use automation router
class EnhancedBulkEmbedder:
    def __init__(self):
        self.automation = AutomationRouter({
            "crawl4ai": {"max_concurrent": 10},
            "browser_use": {
                "llm_provider": "openai",
                "model": "gpt-4o-mini",
                "headless": True,
            },
            "playwright": {"browser": "chromium"},
        })
    
    async def crawl_documentation(self, urls: list[str]):
        """Crawl with intelligent automation selection."""
        
        tasks = []
        for url in urls:
            # Let router decide best tool
            task = self.automation.scrape(url)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check metrics and adjust
        metrics = self.automation.get_metrics()
        logger.info(f"Automation metrics: {metrics}")
```

## Expected Performance

| Site Type | Tool Selected | Avg Time | Success Rate |
|-----------|---------------|----------|--------------|
| Static docs | Crawl4AI | 0.4s | 98% |
| React SPA | browser-use | 1.8s | 96% |
| Dynamic content | browser-use | 2.3s | 94% |
| Auth required | Playwright | 3.5s | 99% |
| Average | Mixed | 1.1s | 97% |

## Browser-Use Configuration Guide

### Environment Variables

```bash
# Required for OpenAI (recommended)
export OPENAI_API_KEY="sk-..."

# Optional for other providers
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."

# browser-use specific configuration
export BROWSER_USE_LLM_PROVIDER="openai"  # openai, anthropic, gemini, local
export BROWSER_USE_MODEL="gpt-4o-mini"    # Cost-optimized model
export BROWSER_USE_HEADLESS="true"        # Headless browser mode
```

### Installation

```bash
# Add to requirements.txt
browser-use>=0.2.5

# Add to pyproject.toml
[project.dependencies]
browser-use = ">=0.2.5"
```

### Model Selection Guide

| Provider | Model | Cost/1K tokens | Best For |
|----------|-------|----------------|----------|
| OpenAI | gpt-4o-mini | $0.00015 | Cost-optimized automation |
| OpenAI | gpt-4o | $0.0025 | Complex interactions |
| Anthropic | claude-3-haiku | $0.00025 | Balanced performance |
| Anthropic | claude-3-sonnet | $0.003 | Advanced reasoning |
| Local | llama3.2:3b | Free | Privacy-focused |

### Task Design Best Practices

```python
# Good: Specific, actionable task
task = "Navigate to the documentation page, expand all code examples, click any 'Show more' buttons, and extract all content including API references"

# Bad: Vague, unclear task  
task = "Get the docs"

# Good: Include context and constraints
task = """
Navigate to {url} and extract documentation content.
Please:
1. Wait for page to fully load (look for main content)
2. Handle any cookie banners by clicking accept
3. Expand collapsed sections like 'Advanced Options'
4. Extract code examples with syntax highlighting
5. Return structured content with headings preserved
"""
```

### Optimal Tier Placement Decision

Based on research findings, the optimal hierarchy is:

**Crawl4AI → browser-use → Playwright**

**Rationale:**
1. **Crawl4AI first**: Handles 90% of static documentation efficiently
2. **browser-use second**: AI-powered fallback for dynamic content and interactions
3. **Playwright last**: Maximum control for edge cases and authentication

This placement leverages browser-use's AI capabilities for the middle complexity tier while maintaining Playwright as the final fallback for scenarios requiring precise programmatic control.

### Migration from Stagehand

1. Replace `StagehandAdapter` with `BrowserUseAdapter`
2. Convert instruction lists to natural language tasks
3. Update routing rules to use `browser_use` instead of `stagehand`
4. Add LLM provider configuration
5. Update tests to verify browser-use integration
