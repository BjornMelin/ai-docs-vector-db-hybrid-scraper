# Cross-Platform Compatibility Guide

This document outlines the cross-platform compatibility improvements implemented for the AI Docs Vector DB Hybrid Scraper project to ensure consistent behavior across Windows, macOS, and Linux environments.

## Overview

The project now includes comprehensive cross-platform support with optimized CI/CD pipelines, platform-specific configurations, and robust error handling for different operating systems.

## Key Improvements

### 1. Cross-Platform Utilities (`src/utils/cross_platform.py`)

A new comprehensive utility module provides:

- **Platform Detection**: Reliable detection of Windows, macOS, and Linux
- **CI Environment Detection**: Automatic detection of CI/CD environments
- **Path Handling**: Platform-appropriate directory structures for cache, config, and data
- **Browser Configuration**: Platform-specific browser paths and settings
- **Environment Setup**: Automatic platform-specific environment variable configuration

Key functions:
```python
- is_windows() / is_macos() / is_linux()
- is_ci_environment()
- get_platform_cache_dir() / get_platform_config_dir() / get_platform_data_dir()
- get_playwright_browser_path()
- set_platform_environment_defaults()
```

### 2. Enhanced CI/CD Configuration (`.github/workflows/ci.yml`)

#### Platform Matrix Optimization
- **Reduced Matrix Size**: Focus on Python 3.13 for Windows/macOS to reduce CI time
- **Platform-Specific Test Selection**: Different test markers for each platform
- **Optimized Test Execution**: Skip integration tests on Windows/macOS for faster CI

#### Windows-Specific Improvements
- **Dependency Installation**: Fallback installation strategies for native dependencies
- **Browser Setup**: Platform-appropriate Playwright browser installation
- **Environment Variables**: UTF-8 encoding and compiler flags for native extensions
- **Test Timeouts**: Extended timeouts for Windows environments (300s vs 120s)

#### macOS-Specific Improvements  
- **Browser Installation**: macOS-specific browser paths and homebrew integration
- **System Dependencies**: Avoid system-level dependencies that require sudo
- **Test Configuration**: Optimized test execution with appropriate timeouts

#### Linux Optimizations
- **Full Test Suite**: Complete integration and unit test execution
- **System Dependencies**: Install browser dependencies with system packages
- **Performance Baseline**: Use Linux as performance benchmark

### 3. Improved Browser Setup (`scripts/test_browser_setup.py`)

Enhanced browser automation setup with:

- **Platform-Aware Installation**: Different strategies for each OS
- **CI Environment Handling**: Optimized browser installation for CI
- **Fallback Mechanisms**: Multiple installation strategies with error recovery
- **Environment Configuration**: Automatic platform-specific environment setup

### 4. Enhanced Test Configuration (`tests/conftest.py`)

#### Platform-Aware Test Fixtures
- **Platform Information**: `platform_info` fixture for test platform detection
- **Conditional Skipping**: `skip_if_windows`, `skip_if_macos`, `skip_if_linux` fixtures
- **CI Requirements**: `require_ci`, `skip_if_ci` fixtures for environment-specific tests
- **Browser Configuration**: Platform-specific browser test configurations

#### Environment Setup
- **Automatic Configuration**: Platform-specific environment variable defaults
- **Test Isolation**: Proper environment restoration between tests
- **Directory Creation**: Cross-platform test directory setup with proper permissions

### 5. Platform-Specific Test Configuration (`pytest-platform.ini`)

Comprehensive test configuration including:

- **Test Markers**: Platform-specific test markers (windows, macos, linux, etc.)
- **Timeout Settings**: Platform-appropriate test timeouts
- **Test Selection**: Platform-specific test directory filters
- **Performance Thresholds**: Platform-adjusted performance expectations

## Platform-Specific Configurations

### Windows
- **UTF-8 Encoding**: `PYTHONUTF8=1` for proper Unicode handling
- **Browser Arguments**: `--disable-gpu`, `--disable-dev-shm-usage` for stability
- **Extended Timeouts**: 300s for tests, 60s for browser operations
- **Build Isolation**: `--no-build-isolation` for problematic native dependencies
- **File Permissions**: Windows-compatible permission handling

### macOS
- **Browser Paths**: `~/Library/Caches/ms-playwright`
- **System Tools**: Proper PATH setup for Homebrew and system tools
- **Browser Installation**: Browser-only installation (no system dependencies)
- **Timeout Configuration**: 240s for tests, 45s for browser operations

### Linux
- **System Dependencies**: Full browser installation with system packages
- **Performance Baseline**: Standard timeout and performance expectations
- **Complete Test Suite**: Full integration and unit test execution
- **POSIX Compliance**: Standard Unix file permissions and paths

## Test Execution Strategies

### Local Development
```bash
# Run platform-appropriate tests
uv run pytest tests/unit -m "not slow"

# Run with platform-specific markers
uv run pytest tests/unit -m "linux and not integration"  # Linux only
uv run pytest tests/unit -m "not windows"                # Skip Windows-specific
```

### CI Environment
```bash
# Linux (full suite)
uv run pytest tests/unit tests/integration --cov=src

# Windows/macOS (optimized)
uv run pytest tests/unit -m "not slow and not integration" --timeout=300
```

## Browser Automation

### Cross-Platform Browser Setup
The browser setup process now automatically:

1. **Detects Platform**: Identifies Windows, macOS, or Linux
2. **Configures Environment**: Sets platform-specific environment variables
3. **Installs Browsers**: Uses appropriate installation strategy for each platform
4. **Handles Fallbacks**: Multiple installation attempts with error recovery
5. **Validates Setup**: Comprehensive testing of browser automation capabilities

### Platform-Specific Browser Configuration
- **Windows**: Chromium with GPU disabled, extended timeouts
- **macOS**: Chromium with system browser fallback
- **Linux**: Chromium with system dependencies

## Error Handling and Debugging

### Common Issues and Solutions

#### Windows
- **Native Dependency Issues**: Use `--no-build-isolation` flag
- **UTF-8 Encoding**: Ensure `PYTHONUTF8=1` environment variable
- **Browser Installation**: May require multiple installation attempts
- **File Path Length**: Use short paths for test fixtures

#### macOS
- **Permission Issues**: Browser-only installation avoids sudo requirements
- **System Dependencies**: Use Homebrew for additional tools
- **Sandboxing**: May require additional browser arguments

#### Linux
- **System Dependencies**: Ensure proper package manager access
- **Browser Installation**: Use `--with-deps` for complete setup
- **Display Issues**: Set `DISPLAY` environment variable if needed

### Debug Commands
```bash
# Test platform detection
uv run python -c "from src.utils.cross_platform import *; print(get_process_info())"

# Test browser setup
uv run python scripts/test_browser_setup.py

# Platform-specific pytest run
uv run pytest --collect-only -m "windows or macos or linux"
```

## Future Improvements

### Planned Enhancements
1. **Performance Optimization**: Platform-specific performance tuning
2. **Container Support**: Docker-based cross-platform testing
3. **ARM Support**: Apple Silicon and ARM Linux optimization
4. **Dependency Caching**: Better dependency caching strategies

### Monitoring
- **CI Performance**: Track platform-specific CI execution times
- **Test Reliability**: Monitor platform-specific test failure rates
- **Resource Usage**: Platform-specific resource consumption tracking

## Configuration Examples

### Environment Variables
```bash
# Windows
export PYTHONUTF8=1
export PLAYWRIGHT_BROWSERS_PATH="$USERPROFILE/AppData/Local/ms-playwright"

# macOS  
export PLAYWRIGHT_BROWSERS_PATH="$HOME/Library/Caches/ms-playwright"

# Linux
export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
```

### Test Execution
```bash
# Platform-specific test runs
pytest tests/ -m "not slow" --timeout=300        # Windows
pytest tests/ -m "not slow" --timeout=240        # macOS
pytest tests/ --timeout=180                      # Linux
```

This cross-platform compatibility implementation ensures reliable operation across all supported platforms while optimizing for platform-specific performance characteristics and limitations.