# UnifiedBrowserManager Refactoring Plan

## Issues Identified

### 1. Over-Engineering
- Quality score calculation is arbitrary and not data-driven
- Tier confidence calculations are overly complex
- Alternative tier suggestions aren't used anywhere

### 2. Backward Compatibility Code to Remove
- `formats` parameter in CrawlManager (marked as legacy)
- `provider` field duplication in CrawlManager response
- Legacy format conversion in get_metrics()

### 3. Simplification Opportunities
- Consolidate error handling
- Remove unused methods
- Simplify metrics tracking (AutomationRouter already tracks metrics)
- Remove redundant tier analysis methods

## Refactoring Actions

1. **Remove from UnifiedBrowserManager**:
   - `_calculate_tier_confidence()` - Not used
   - `_get_tier_reliability()` - Not used
   - `_get_alternative_tiers()` - Not used
   - Complex quality score calculation

2. **Remove from CrawlManager**:
   - Legacy `formats` parameter
   - Duplicate `provider` field
   - Legacy metric format conversion

3. **Simplify**:
   - Quality score to simple content length check
   - Metrics tracking (rely on AutomationRouter)
   - Error handling consolidation