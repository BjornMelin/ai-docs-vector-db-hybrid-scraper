# Browser Automation in CI/CD Environments

This document provides guidance for setting up browser automation testing in CI/CD environments, specifically addressing the challenges with Playwright and Crawl4AI dependencies.

## Overview

Our project uses both Playwright and Crawl4AI for browser automation. These tools require specific browser installations and system dependencies that need careful handling in CI environments.

## CI Environment Configuration

### GitHub Actions Setup

The CI workflow includes optimized browser installation steps:

```yaml
- name: Install browser dependencies
  shell: bash
  run: |
    # Install Playwright browsers and system dependencies for testing
    if [ "${{ runner.os }}" == "Linux" ]; then
      # Linux: Install Playwright browsers with system dependencies
      uv run python -m playwright install --with-deps chromium
      # Fallback setup with proper error handling
      export CRAWL4AI_SKIP_DB_INIT=1
      # Additional browser setup validation
    elif [ "${{ runner.os }}" == "macOS" ]; then
      # macOS: Install Playwright browsers
      uv run python -m playwright install chromium
    elif [ "${{ runner.os }}" == "Windows" ]; then
      # Windows: Install Playwright browsers
      uv run python -m playwright install chromium
    fi

- name: Setup test environment
  shell: bash
  run: |
    # Set environment variables for headless testing
    echo "PLAYWRIGHT_BROWSERS_PATH=0" >> $GITHUB_ENV
    echo "CRAWL4AI_HEADLESS=true" >> $GITHUB_ENV
```

### Key Environment Variables

- `CRAWL4AI_HEADLESS=true`: Forces headless mode for all browser operations
- `PLAYWRIGHT_BROWSERS_PATH=0`: Uses default browser installation path
- `PLAYWRIGHT_CHROMIUM_SANDBOX=false`: Disables sandbox for CI environments
- `CI=true`: Automatically set by GitHub Actions, used for CI-specific configurations

## Browser Configuration

### Playwright Configuration

For CI environments, use these browser arguments:

```python
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"]
    )
```

### Crawl4AI Configuration

```python
from crawl4ai.async_configs import BrowserConfig

# Basic configuration
browser_config = BrowserConfig(
    headless=True,
    browser_type="chromium"
)

# CI-specific configuration
if os.getenv("CI"):
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        extra_args=["--no-sandbox", "--disable-dev-shm-usage"]
    )
```

## Troubleshooting

### Common Issues

1. **Browser installation fails with sudo requirement**
   ```bash
   # Solution: Use browser-only install
   python -m playwright install chromium
   ```

2. **System dependencies missing**
   ```bash
   # For Linux CI environments
   python -m playwright install --with-deps chromium
   ```

3. **Async/Sync API conflicts**
   ```python
   # Use async API consistently
   from playwright.async_api import async_playwright  # ✓
   # not
   from playwright.sync_api import sync_playwright    # ✗ in async context
   ```

### Testing Browser Setup

Use the provided test script to validate your setup:

```bash
uv run python scripts/test_browser_setup.py
```

This script tests:
- Playwright installation and browser availability
- Crawl4AI browser configuration
- Basic browser automation functionality
- CI environment compatibility

## Platform-Specific Notes

### Ubuntu/Linux
- Requires system dependencies for full browser functionality
- Uses `--with-deps` flag for complete installation
- May need additional font packages for rendering

### macOS
- Generally works with browser-only installation
- System dependencies usually available
- No special sandbox configuration needed

### Windows
- Browser-only installation typically sufficient
- May require Visual C++ Redistributables
- Windows Defender exclusions might be needed

## Best Practices

1. **Always use headless mode in CI**
2. **Set appropriate timeouts for browser operations**
3. **Use proper cleanup in test fixtures**
4. **Monitor CI performance and adjust resource allocation**
5. **Cache browser installations when possible**

## Test Configuration

The test suite automatically configures browser environments:

```python
@pytest.fixture(scope="session", autouse=True)
def setup_browser_environment():
    """Set up browser automation environment for CI and local testing."""
    if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
        os.environ["CRAWL4AI_HEADLESS"] = "true"
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
        os.environ["PLAYWRIGHT_CHROMIUM_SANDBOX"] = "false"
```

## Performance Considerations

- Browser installation adds 2-5 minutes to CI time
- Consider using Docker images with pre-installed browsers
- Cache browser installations between CI runs when possible
- Use matrix builds to parallelize cross-platform testing

## Security Notes

- `--no-sandbox` flag reduces security but is necessary for containerized CI
- Only use these configurations in trusted CI environments
- Regular security updates for browser dependencies are essential