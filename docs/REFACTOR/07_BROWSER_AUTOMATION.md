# Browser Automation Hierarchy Implementation Guide

## Overview

Implement a three-tier browser automation hierarchy that intelligently selects the right tool for each scraping task. This provides the best balance of performance, cost, and capability.

## Automation Hierarchy

### Tier 1: Crawl4AI (Default)

- **Use for**: 90% of documentation sites
- **Performance**: 4-6x faster than alternatives
- **Cost**: $0
- **JavaScript**: Basic support

### Tier 2: Stagehand (AI-Powered)

- **Use for**: Complex interactions, dynamic content
- **Performance**: 2x slower than Crawl4AI
- **Cost**: Minimal (local LLM)
- **JavaScript**: Full support with AI understanding

### Tier 3: Playwright (Fallback)

- **Use for**: Maximum control scenarios
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
from .stagehand_adapter import StagehandAdapter
from .playwright_adapter import PlaywrightAdapter
from ..logging_config import get_logger

logger = get_logger(__name__)

class AutomationRouter:
    """Intelligently route scraping tasks to appropriate automation tool."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        
        # Initialize adapters
        self.crawl4ai = Crawl4AIAdapter(config.get("crawl4ai", {}))
        self.stagehand = StagehandAdapter(config.get("stagehand", {}))
        self.playwright = PlaywrightAdapter(config.get("playwright", {}))
        
        # Site-specific routing rules
        self.routing_rules = {
            # Sites that need Stagehand (AI-powered interaction)
            "stagehand": [
                "vercel.com",  # Complex React app
                "clerk.com",   # Heavy client-side rendering
                "supabase.com",  # Dynamic documentation
            ],
            
            # Sites that need Playwright (specific automation)
            "playwright": [
                "github.com",  # Authentication required
                "stackoverflow.com",  # Complex pagination
            ],
            
            # Default to Crawl4AI for everything else
        }
        
        # Performance metrics
        self.metrics = {
            "crawl4ai": {"success": 0, "failed": 0, "avg_time": 0},
            "stagehand": {"success": 0, "failed": 0, "avg_time": 0},
            "playwright": {"success": 0, "failed": 0, "avg_time": 0},
        }
    
    async def scrape(
        self,
        url: str,
        interaction_required: bool = False,
        custom_actions: Optional[list[dict]] = None,
        force_tool: Optional[Literal["crawl4ai", "stagehand", "playwright"]] = None,
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
            elif tool == "stagehand":
                result = await self._try_stagehand(url, custom_actions)
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
            return "stagehand"  # AI can handle complex interactions
        
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
    
    async def _try_stagehand(
        self,
        url: str,
        custom_actions: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Try scraping with Stagehand AI."""
        
        # Convert custom actions to Stagehand format
        if custom_actions:
            instructions = self._convert_to_instructions(custom_actions)
        else:
            instructions = ["Extract all documentation content", "Expand collapsed sections"]
        
        return await self.stagehand.scrape(
            url=url,
            instructions=instructions,
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
            "crawl4ai": ["stagehand", "playwright"],
            "stagehand": ["playwright", "crawl4ai"],
            "playwright": ["stagehand", "crawl4ai"],
        }
        
        for fallback_tool in fallback_order[failed_tool]:
            try:
                logger.info(f"Falling back to {fallback_tool}")
                
                if fallback_tool == "crawl4ai":
                    return await self._try_crawl4ai(url, custom_actions)
                elif fallback_tool == "stagehand":
                    return await self._try_stagehand(url, custom_actions)
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
    
    def _convert_to_instructions(self, actions: list[dict]) -> list[str]:
        """Convert custom actions to Stagehand instructions."""
        
        instructions = []
        
        for action in actions:
            if action["type"] == "click":
                instructions.append(f"Click on {action['selector']}")
            elif action["type"] == "type":
                instructions.append(f"Type '{action['text']}' in {action['selector']}")
            elif action["type"] == "wait":
                instructions.append(f"Wait for {action['timeout']}ms")
        
        return instructions
    
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

### 2. Stagehand Adapter

```python
# src/services/browser/stagehand_adapter.py
from typing import Any, Optional
from stagehand import Stagehand
import asyncio

from ..base import BaseService

class StagehandAdapter(BaseService):
    """AI-powered browser automation with Stagehand."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        
        # Stagehand configuration
        self.stagehand_config = {
            "env": config.get("env", "LOCAL"),  # Use local LLM
            "headless": config.get("headless", True),
            "model": config.get("model", "ollama/llama2"),
            "enable_caching": config.get("enable_caching", True),
            "debug_screenshots": config.get("debug", False),
        }
        
        self.stagehand = None
    
    async def initialize(self):
        """Initialize Stagehand instance."""
        
        self.stagehand = Stagehand(**self.stagehand_config)
        await self.stagehand.start()
    
    async def scrape(
        self,
        url: str,
        instructions: list[str],
        timeout: int = 30000,
    ) -> dict[str, Any]:
        """Scrape using AI-powered automation."""
        
        if not self.stagehand:
            await self.initialize()
        
        try:
            # Navigate to page
            page = await self.stagehand.new_page()
            await page.goto(url, wait_until="networkidle")
            
            # Execute AI-driven actions
            for instruction in instructions:
                self.logger.info(f"Executing: {instruction}")
                
                if "click" in instruction.lower():
                    await self.stagehand.click(page, instruction)
                elif "type" in instruction.lower():
                    await self.stagehand.type(page, instruction)
                elif "extract" in instruction.lower():
                    content = await self.stagehand.extract(page, instruction)
                else:
                    # General action
                    await self.stagehand.act(page, instruction)
            
            # Extract final content
            result = await self.stagehand.extract(
                page,
                "Extract all documentation content including code examples"
            )
            
            # Get page content
            html = await page.content()
            
            return {
                "success": True,
                "url": url,
                "content": result.get("content", ""),
                "html": html,
                "metadata": result.get("metadata", {}),
                "screenshots": result.get("screenshots", []),
            }
            
        except Exception as e:
            self.logger.error(f"Stagehand error: {e}")
            raise
        finally:
            if page:
                await page.close()
    
    async def cleanup(self):
        """Cleanup Stagehand resources."""
        
        if self.stagehand:
            await self.stagehand.stop()
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
            "tool": "stagehand",
            "instructions": [
                "Wait for React documentation to load",
                "Expand all code examples",
                "Click on 'Show more' buttons if present",
            ],
        },
        
        "github.com": {
            "tool": "playwright",
            "actions": [
                {"type": "wait_for_selector", "selector": ".markdown-body"},
                {"type": "evaluate", "script": "document.querySelectorAll('.d-none').forEach(e => e.classList.remove('d-none'))"},
            ],
        },
        
        "stackoverflow.com": {
            "tool": "playwright",
            "actions": [
                {"type": "wait_for_selector", "selector": ".answercell"},
                {"type": "click", "selector": ".js-show-link.comments-link"},
                {"type": "wait", "timeout": 1000},
            ],
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
    assert router._select_tool("https://vercel.com", False, None) == "stagehand"
    assert router._select_tool("https://github.com", False, None) == "playwright"
    
    # Test with interaction required
    assert router._select_tool("https://example.com", True, None) == "stagehand"
    
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
            "stagehand": {"headless": True},
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
| React SPA | Stagehand | 2.1s | 95% |
| Auth required | Playwright | 3.5s | 99% |
| Average | Mixed | 1.2s | 97% |
