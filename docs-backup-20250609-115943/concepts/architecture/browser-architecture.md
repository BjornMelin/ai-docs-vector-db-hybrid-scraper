# Browser Automation Architecture

> **Status**: Current  
> **Last Updated**: 2025-06-09  
> **Purpose**: Browser Architecture concept explanation  
> **Audience**: Developers wanting to understand design

## Overview

The browser automation system provides intelligent, multi-tier web scraping capabilities using a combination of lightweight HTTP scraping, browser automation tools, and AI-powered interaction systems.

## Current Architecture Status

### ✅ Implemented Components

The system includes a complete browser automation stack:

- **AutomationRouter** - Intelligent tool selection and routing
- **BrowserUseAdapter** - AI-powered automation with multi-LLM support (OpenAI, Anthropic, Gemini)
- **Crawl4AIAdapter** - High-performance browser automation
- **PlaywrightAdapter** - Programmatic browser control
- **LightweightScraper** - HTTP-only scraping for static content

### ❌ Critical Integration Gap

The browser automation system exists as a **parallel system** but is **not integrated** with the main crawling flow:

```
CrawlManager (Main System)     AutomationRouter (Unused System)
├─ LightweightScraper         ├─ Crawl4AIAdapter
├─ Crawl4AIProvider           ├─ BrowserUseAdapter  
└─ FirecrawlProvider          └─ PlaywrightAdapter

❌ NO INTEGRATION BETWEEN SYSTEMS
```

## Target Architecture: Unified 5-Tier System

### Tier Hierarchy

```
Tier 0: Lightweight HTTP (httpx + BeautifulSoup)
├─ Use case: Static HTML, documentation, raw files
├─ Performance: 5-10x faster than browser automation
└─ Cost: $0

Tier 1: Crawl4AI Basic (Browser automation)
├─ Use case: Standard dynamic content, basic JavaScript
├─ Performance: 4-6x faster than complex automation
└─ Cost: $0

Tier 2: Crawl4AI Enhanced (Browser + Custom JavaScript)
├─ Use case: Interactive content, form submissions
├─ Performance: Optimized for specific interactions
└─ Cost: $0

Tier 3: Browser-use AI (Multi-LLM automation)
├─ Use case: Complex interactions requiring reasoning
├─ Performance: Natural language task processing
└─ Cost: LLM API costs ($0.001-0.01 per request)

Tier 4: Playwright + Firecrawl (Maximum control)
├─ Use case: Complex auth, multi-step workflows
├─ Performance: Full programmatic control
└─ Cost: Firecrawl API costs ($0.002-0.01 per request)
```

### Intelligent Routing Logic

The system automatically selects the optimal tier based on:

1. **URL Pattern Analysis**
   - Static file extensions (.md, .txt, .json) → Tier 0
   - GitHub raw content → Tier 0
   - Documentation sites → Tier 1
   - Interactive applications → Tier 2-3

2. **Site-Specific Rules**
   ```json
   {
     "browser_use_sites": [
       "vercel.com", "clerk.com", "supabase.com", 
       "react.dev", "docs.anthropic.com"
     ],
     "playwright_sites": [
       "github.com", "stackoverflow.com", "discord.com"
     ]
   }
   ```

3. **Content Complexity Analysis**
   - JavaScript requirement detection
   - SPA (Single Page Application) identification
   - Interactive element detection
   - Authentication requirements

4. **Performance-Based Learning**
   - Success rate tracking per tier and domain
   - Response time optimization
   - Cost-aware routing decisions

### Fallback Strategy

```
Network errors → Retry same tier
Content loading issues → Escalate to higher tier  
JavaScript execution errors → Escalate to browser automation
Anti-bot detection → Escalate to browser-use AI
Complete failure → Try Firecrawl API as last resort
```

## Configuration

### Browser Automation Settings

```yaml
browser_automation:
  tier_selection: "auto"  # auto, manual, performance_based
  fallback_enabled: true
  session_pooling: true
  
  providers:
    lightweight:
      enabled: true
      timeout: 5
      content_threshold: 100
      
    crawl4ai:
      enabled: true
      enhanced_mode: true
      browser_type: "chromium"
      headless: true
      
    browser_use:
      enabled: true
      llm_provider: "openai"  # openai, anthropic, gemini
      model: "gpt-4o-mini"
      timeout: 30000
      
    playwright:
      enabled: true
      browser_type: "chromium"
      headless: true
      
    firecrawl:
      enabled: true
      api_key: "${FIRECRAWL_API_KEY}"
```

### Site-Specific Routing

Configuration file: `config/browser-routing-rules.json`

```json
{
  "routing_rules": {
    "browser_use": [
      "vercel.com", "clerk.com", "supabase.com",
      "netlify.com", "railway.app", "react.dev"
    ],
    "playwright": [
      "github.com", "stackoverflow.com", "discord.com",
      "slack.com", "notion.so"
    ]
  }
}
```

## API Interface

### Unified Browser Manager (Target)

```python
class UnifiedBrowserManager:
    async def scrape(
        self,
        url: str,
        tier: Optional[int] = None,  # Force specific tier
        interaction_required: bool = False,
        custom_actions: Optional[List[dict]] = None,
        formats: List[str] = ["markdown"],
        timeout: int = 30000,
    ) -> Dict[str, Any]:
        """
        Intelligent multi-tier scraping with automatic tool selection.
        
        Returns:
            {
                "success": bool,
                "content": {"markdown": str, "html": str, "text": str},
                "metadata": {"title": str, "description": str, "tier": int},
                "performance": {"elapsed_ms": float, "tier": int, "provider": str},
                "url": str
            }
        """
```

### Current Workaround

Until integration is complete, access browser automation directly:

```python
from src.infrastructure.client_manager import ClientManager
from src.config import UnifiedConfig

config = UnifiedConfig()
client_manager = ClientManager(config)
router = await client_manager.get_browser_automation_router()

# Use browser-use for complex sites
result = await router.scrape(
    url="https://docs.anthropic.com/claude/docs",
    interaction_required=True,
    force_tool="browser_use"
)
```

## Implementation Phases

### Phase 1: Immediate Fixes (Priority: HIGH)
- [ ] Fix `ClientManager` class import error
- [ ] Implement missing `get_task_queue_manager()` method
- [ ] Add browser automation tools to MCP server
- [ ] Create integration tests

### Phase 2: Architectural Integration (Priority: HIGH)
- [ ] Design `UnifiedBrowserManager` interface
- [ ] Integrate AutomationRouter into CrawlManager
- [ ] Implement content complexity analysis
- [ ] Add unified metrics tracking

### Phase 3: Performance Optimization (Priority: MEDIUM)
- [ ] Implement session pooling across all tiers
- [ ] Add memory-adaptive dispatching
- [ ] Create cost-aware routing
- [ ] Optimize provider initialization

### Phase 4: Advanced Features (Priority: LOW)
- [ ] Add learning from failure patterns
- [ ] Implement proxy rotation
- [ ] Create advanced site configurations
- [ ] Add monitoring dashboard

## Performance Characteristics

| Tier | Speed | Cost | Capability | Use Case |
|------|-------|------|------------|----------|
| 0 | 5-10x faster | $0 | Static content | Docs, raw files |
| 1 | 4-6x faster | $0 | Basic JS | Standard sites |
| 2 | 2-4x faster | $0 | Interactive | Forms, SPAs |
| 3 | Baseline | $0.001-0.01 | AI reasoning | Complex interactions |
| 4 | Variable | $0.002-0.01 | Max control | Auth, workflows |

## Monitoring and Metrics

### Key Performance Indicators

- **Success Rate** by tier and domain
- **Response Time** percentiles (p50, p95, p99)
- **Cost Per Request** across all tiers
- **Escalation Rate** between tiers
- **Cache Hit Rate** for repeated requests

### Health Checks

- Provider availability and response times
- LLM API key validation and quota
- Browser pool resource utilization
- Memory and CPU usage patterns

## Security Considerations

- **API Key Management**: Environment-based configuration only
- **Rate Limiting**: Per-provider and global limits
- **Proxy Support**: Anti-bot detection mitigation
- **Resource Limits**: Memory and CPU constraints
- **Timeout Handling**: Prevent hung requests

## Related Documentation

- [Browser Automation User Guide](../tutorials/browser-automation.md) - Complete 5-tier implementation guide
- [Crawl4AI User Guide](../tutorials/crawl4ai-setup.md) - Complete configuration and troubleshooting
- [api/browser_automation_api.md](../api/browser_automation_api.md) - API reference