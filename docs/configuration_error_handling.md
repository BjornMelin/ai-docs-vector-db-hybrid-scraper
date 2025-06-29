# Configuration Error Handling Documentation

## Overview

This document describes the comprehensive error handling system implemented for the configuration management in the AI Documentation Vector DB Hybrid Scraper project.

## Key Components

### 1. Error Handling Module (`src/config/error_handling.py`)

Provides a comprehensive error handling framework with:

- **Custom Error Classes**:
  - `ConfigError`: Base exception for configuration errors
  - `ConfigLoadError`: Error loading configuration from source
  - `ConfigValidationError`: Error validating configuration values
  - `ConfigReloadError`: Error during configuration reload operation
  - `ConfigFileWatchError`: Error in file watching operations

- **Error Context Management**:
  - `ErrorContext`: Context manager for capturing detailed error context
  - `async_error_context`: Async version for error context management
  - Automatic logging with full context information

- **Retry Logic**:
  - `RetryableConfigOperation`: Decorator for retryable operations
  - Uses `tenacity` library for exponential backoff
  - Configurable retry attempts and timing

- **Safe Configuration Loading**:
  - `SafeConfigLoader`: Loads configurations with comprehensive error handling
  - Supports JSON, YAML, and TOML formats
  - Fallback configuration support

- **Graceful Degradation**:
  - `GracefulDegradationHandler`: Manages graceful degradation
  - Tracks failure rates and activates degradation mode
  - Skips non-critical operations during degradation

### 2. Enhanced Configuration Manager (`src/config/enhanced_config.py`)

Extends the simplified configuration system with:

- **Enhanced File Watching**:
  - `EnhancedConfigFileWatcher`: Robust file watching with error recovery
  - Consecutive failure tracking
  - Automatic disabling after repeated failures

- **Configuration Management**:
  - `EnhancedConfigManager`: Main configuration manager with error handling
  - Configuration backup and restore functionality
  - Change listener error isolation
  - Async reload support

## Key Features

### 1. Error Recovery

```python
# Example: Configuration reload with error recovery
manager = EnhancedConfigManager(
    config_class=Config,
    config_file=Path("config.json"),
    fallback_config=default_config
)

# If reload fails, keeps existing configuration
success = manager.reload_config()
if not success:
    print("Reload failed, using existing config")
```

### 2. Validation Error Details

```python
try:
    config = loader.create_config(invalid_data)
except ConfigValidationError as e:
    print(f"Validation failed: {e}")
    for error in e.validation_errors:
        print(f"  - {error['loc']}: {error['msg']}")
```

### 3. Retry Logic

```python
@retry_config_operation
async def load_remote_config(url: str):
    # Automatically retries on transient errors
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### 4. Graceful Degradation

```python
degradation = get_degradation_handler()

# Record failures
for error in errors:
    degradation.record_failure("config_reload", error)

# Check if should skip non-critical operations
if degradation.should_skip_operation("file_watch"):
    print("Skipping file watching due to degradation")
```

### 5. Configuration Backups

```python
# Automatic backup on successful reload
manager.reload_config()  # Creates backup of old config

# Manual restore from backup
manager.restore_from_backup(-1)  # Restore most recent backup
```

## Error Handling Patterns

### 1. Load with Fallback

```python
fallback_config = Config(
    openai=OpenAIConfig(api_key="sk-fallback"),
    qdrant=QdrantConfig(url="http://localhost:6333")
)

manager = EnhancedConfigManager(
    config_file=Path("config.json"),
    fallback_config=fallback_config
)
```

### 2. Change Listener Error Isolation

```python
def safe_listener(old_config, new_config):
    try:
        # Process configuration change
        update_services(new_config)
    except Exception as e:
        logger.error(f"Listener failed: {e}")
        # Error doesn't affect other listeners

manager.add_change_listener(safe_listener)
```

### 3. Async Configuration Loading

```python
manager, config = await create_and_load_config_async(
    config_class=Config,
    config_file=Path("config.json"),
    enable_file_watching=True
)
```

## Security Considerations

- **Secret Masking**: Sensitive values are masked in logs and error messages
- **Hash-based Change Detection**: Includes partial hash of secrets for change detection while maintaining security
- **SecretStr Support**: Uses Pydantic's SecretStr for sensitive fields

## Testing

Comprehensive test coverage includes:

- Unit tests for all error handling components
- Integration tests for real-world error scenarios
- Performance tests for retry logic
- Security tests for secret masking

## Usage Examples

See `examples/config_error_handling_demo.py` for comprehensive usage examples including:

- Basic error handling
- Validation error handling
- Configuration reload with recovery
- Change listeners with error isolation
- Graceful degradation
- Async operations
- Backup and restore

## Best Practices

1. **Always provide a fallback configuration** for critical services
2. **Use change listeners** for reacting to configuration updates
3. **Enable graceful degradation** for production environments
4. **Monitor the degradation handler** to detect systemic issues
5. **Test error scenarios** including corrupted files, permission errors, and validation failures
6. **Use async operations** for non-blocking configuration updates
7. **Implement proper logging** to track configuration changes and errors