# AsyncIO TaskGroup Migration Guide

## Overview

This document describes the migration from `asyncio.gather()` to `asyncio.TaskGroup` for better structured concurrency in Python 3.11+.

## Why Migrate?

1. **Better Exception Handling**: TaskGroup provides structured exception handling with `ExceptionGroup`, making it easier to handle and debug concurrent failures.
2. **Structured Concurrency**: Tasks are automatically cancelled if any task in the group fails (unless explicitly handling exceptions).
3. **Modern Python**: TaskGroup is the recommended approach for concurrent tasks in Python 3.11+.
4. **Cleaner Resource Management**: Automatic cleanup with context managers.

## Migration Pattern

### Old Pattern (asyncio.gather)

```python
# Basic usage
results = await asyncio.gather(*tasks)

# With exception handling
results = await asyncio.gather(*tasks, return_exceptions=True)

# With timeout
results = await asyncio.wait_for(
    asyncio.gather(*tasks, return_exceptions=True),
    timeout=10.0
)
```

### New Pattern (TaskGroup via utility)

```python
from src.utils.async_utils import gather_with_taskgroup

# Basic usage - raises ExceptionGroup on any failure
results = await gather_with_taskgroup(*tasks)

# With exception handling - returns exceptions as results
results = await gather_with_taskgroup(*tasks, return_exceptions=True)

# With timeout
results = await asyncio.wait_for(
    gather_with_taskgroup(*tasks, return_exceptions=True),
    timeout=10.0
)
```

## Utility Functions

### gather_with_taskgroup

A drop-in replacement for `asyncio.gather()` that uses TaskGroup internally:

```python
async def gather_with_taskgroup(
    *coros: Coroutine[Any, Any, T],
    return_exceptions: bool = False,
) -> list[T | BaseException]
```

**Key features:**
- Maintains result order (same as asyncio.gather)
- Supports `return_exceptions` parameter
- Handles ExceptionGroup internally when `return_exceptions=True`
- Compatible with existing error handling patterns

### gather_limited

For concurrent execution with a concurrency limit:

```python
async def gather_limited(
    *coros: Coroutine[Any, Any, T],
    limit: int,
    return_exceptions: bool = False,
) -> list[T | BaseException]
```

**Use cases:**
- Rate limiting
- Resource management
- Preventing overwhelming external services

## Migration Examples

### Example 1: Simple Parallel Execution

**Before:**
```python
tasks = [fetch_data(url) for url in urls]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**After:**
```python
from src.utils.async_utils import gather_with_taskgroup

tasks = [fetch_data(url) for url in urls]
results = await gather_with_taskgroup(*tasks, return_exceptions=True)
```

### Example 2: With Semaphore

**Before:**
```python
semaphore = asyncio.Semaphore(5)

async def limited_task(item):
    async with semaphore:
        return await process_item(item)

tasks = [limited_task(item) for item in items]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**After:**
```python
from src.utils.async_utils import gather_limited

tasks = [process_item(item) for item in items]
results = await gather_limited(*tasks, limit=5, return_exceptions=True)
```

### Example 3: Error Handling

**Before:**
```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for i, result in enumerate(results):
    if isinstance(result, Exception):
        logger.error(f"Task {i} failed: {result}")
    else:
        process_result(result)
```

**After (no change needed):**
```python
results = await gather_with_taskgroup(*tasks, return_exceptions=True)
for i, result in enumerate(results):
    if isinstance(result, Exception):
        logger.error(f"Task {i} failed: {result}")
    else:
        process_result(result)
```

## Exception Handling

### With return_exceptions=False

When `return_exceptions=False` (default), any exception will cause an `ExceptionGroup` to be raised:

```python
try:
    results = await gather_with_taskgroup(*tasks)
except* ValueError as eg:
    # Handle ValueError exceptions
    for exc in eg.exceptions:
        logger.error(f"ValueError: {exc}")
except* Exception as eg:
    # Handle other exceptions
    for exc in eg.exceptions:
        logger.error(f"Unexpected error: {exc}")
```

### With return_exceptions=True

When `return_exceptions=True`, exceptions are returned in the results list:

```python
results = await gather_with_taskgroup(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        # Handle exception
        logger.error(f"Task failed: {result}")
    else:
        # Process successful result
        process_result(result)
```

## Files Migrated

The following files have been migrated to use TaskGroup:

1. **Core Services:**
   - `src/services/query_processing/federated.py`
   - `src/services/managers/crawling_manager.py`
   - `src/services/functional/embeddings.py`
   - `src/services/crawling/crawl4ai_provider.py`

2. **HyDE System:**
   - `src/services/hyde/generator.py`
   - `src/services/hyde/engine.py`
   - `src/services/hyde/cache.py`

3. **Monitoring & Health:**
   - `src/services/monitoring/health.py`
   - `src/services/monitoring/initialization.py`
   - `src/services/auto_detection/health_checks.py`

4. **Other Services:**
   - `src/services/enterprise/search.py`
   - `src/services/cache/modern.py`
   - `src/services/agents/tool_orchestration.py`
   - And more...

## Testing

The migration includes comprehensive tests in `tests/unit/test_async_utils.py` that verify:

1. Basic functionality matches asyncio.gather
2. Exception handling with return_exceptions
3. ExceptionGroup behavior
4. Concurrency limiting
5. Empty coroutine lists
6. Order preservation

## Best Practices

1. **Use the utility functions**: They provide a smooth migration path and consistent behavior.

2. **Handle ExceptionGroup**: When not using `return_exceptions=True`, be prepared to handle ExceptionGroup.

3. **Preserve order**: The utilities maintain result order, which is important for many use cases.

4. **Consider concurrency limits**: Use `gather_limited` when you need to control resource usage.

5. **Test thoroughly**: Ensure your error handling works correctly with the new exception model.

## Performance Considerations

- TaskGroup has minimal overhead compared to asyncio.gather
- The utility functions add a thin wrapper for compatibility
- Concurrency limiting with `gather_limited` can improve overall throughput by preventing resource exhaustion

## Future Considerations

As the codebase evolves, consider:

1. Using TaskGroup directly for new code that doesn't need gather compatibility
2. Adopting more structured concurrency patterns
3. Leveraging ExceptionGroup's advanced filtering capabilities
4. Moving to native TaskGroup usage once all code is migrated