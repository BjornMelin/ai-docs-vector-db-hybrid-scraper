# QdrantClient Usage Examples

This document demonstrates how to use the new focused `QdrantClient` module for managing Qdrant connections.

## Basic Usage

```python
from src.config import UnifiedConfig
from src.services.core.client import QdrantClient

# Load configuration
config = UnifiedConfig.load()

# Create client manager
client_manager = QdrantClient(config)

# Initialize connection
await client_manager.initialize()

# Get the underlying client for operations
qdrant_client = await client_manager.get_client()

# Use the client...
collections = await qdrant_client.get_collections()

# Cleanup when done
await client_manager.cleanup()
```

## Context Manager Usage

```python
async with QdrantClient(config) as client_manager:
    # Client is automatically initialized
    qdrant_client = await client_manager.get_client()
    
    # Perform operations...
    collections = await qdrant_client.get_collections()
    
    # Cleanup is automatic when exiting context
```

## Health Monitoring

```python
# Check client health
health_status = await client_manager.health_check()
print(f"Status: {health_status['status']}")
print(f"Response time: {health_status['response_time_ms']}ms")
print(f"Collections: {health_status['collections_count']}")
```

## Configuration Validation

```python
# Validate configuration
validation_result = await client_manager.validate_configuration()

if not validation_result['valid']:
    print("Configuration issues:")
    for issue in validation_result['config_issues']:
        print(f"  - {issue}")

if validation_result['recommendations']:
    print("Recommendations:")
    for rec in validation_result['recommendations']:
        print(f"  - {rec}")
```

## Connection Recovery

```python
try:
    # Perform some operation
    await qdrant_client.get_collections()
except Exception as e:
    print(f"Connection failed: {e}")
    
    # Reconnect and retry
    await client_manager.reconnect()
    qdrant_client = await client_manager.get_client()
    collections = await qdrant_client.get_collections()
```

## Testing Operations

```python
# Test basic connectivity
test_result = await client_manager.test_operation("basic_connectivity")

if test_result['success']:
    print(f"Test passed in {test_result['execution_time_ms']}ms")
    print(f"Found {test_result['result']['collections_found']} collections")
else:
    print(f"Test failed: {test_result['error']}")
```

## Integration with Other Services

The `QdrantClient` can be easily integrated with higher-level services:

```python
from src.services.core.qdrant_service import QdrantService

class MyService:
    def __init__(self, config: UnifiedConfig):
        self.client_manager = QdrantClient(config)
        self.qdrant_service = QdrantService(config)
    
    async def initialize(self):
        await self.client_manager.initialize()
        await self.qdrant_service.initialize()
    
    async def health_check(self):
        # Use client manager for health checks
        return await self.client_manager.health_check()
    
    async def create_collection(self, name: str, vector_size: int):
        # Use higher-level service for operations
        return await self.qdrant_service.create_collection(name, vector_size)
```

## Key Benefits

1. **Focused Responsibility**: Only handles client management, not business logic
2. **Better Error Handling**: Specific error types and detailed context
3. **Health Monitoring**: Built-in health checks and diagnostics
4. **Configuration Validation**: Proactive configuration issue detection
5. **Connection Recovery**: Easy reconnection for handling network issues
6. **Clean Separation**: Can be used independently or with higher-level services