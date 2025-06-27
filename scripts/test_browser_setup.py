#!/usr/bin/env python3
"""
Test script to verify browser automation setup for CI and local development.

This script tests:
1. Playwright browser installation
2. Crawl4AI browser configuration
3. Basic browser automation functionality
4. CI environment compatibility
"""

import asyncio  # noqa: PLC0415
import os  # noqa: PLC0415
import subprocess
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import cross-platform utilities
from utils.cross_platform import (
    is_windows,
    is_macos,
    is_linux,
    is_ci_environment,
    get_playwright_browser_path,
    set_platform_environment_defaults,
)


def print_status(message: str, status: str = "INFO") -> None:
    """Print status message with formatting."""
    status_map = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
    print(f"{status_map.get(status, 'ℹ️')} {message}")


def check_environment() -> dict:
    """Check current environment setup."""
    print_status("Checking environment setup...")

    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "is_windows": is_windows(),
        "is_macos": is_macos(),
        "is_linux": is_linux(),
        "ci_environment": is_ci_environment(),
        "crawl4ai_headless": os.getenv("CRAWL4AI_HEADLESS", "false"),
        "playwright_path": get_playwright_browser_path(),
    }

    for key, value in env_info.items():
        print(f"  {key}: {value}")

    return env_info


def test_playwright_installation() -> bool:
    """Test Playwright browser installation."""
    print_status("Testing Playwright installation...")

    try:
        import playwright

        try:
            print_status(f"Playwright version: {playwright.__version__}", "SUCCESS")
        except AttributeError:
            print_status("Playwright installed (version unknown)", "SUCCESS")

        # Check if browsers are installed (use simple import test to avoid sync/async issues)
        try:
            from playwright.async_api import async_playwright
            print_status("Playwright browser modules available", "SUCCESS")
            return True
        except Exception as e:
            print_status(f"Playwright browser modules not available: {e}", "ERROR")
            return False

    except ImportError as e:
        print_status(f"Playwright not available: {e}", "ERROR")
        return False


def test_crawl4ai_browser_config() -> bool:
    """Test Crawl4AI browser configuration."""
    print_status("Testing Crawl4AI browser configuration...")

    try:
        from crawl4ai.async_configs import BrowserConfig

        # Test basic configuration
        config = BrowserConfig(
            headless=True,
            browser_type="chromium",
        )

        print_status("Crawl4AI BrowserConfig created successfully", "SUCCESS")

        # Test with CI-specific options
        if os.getenv("CI"):
            ci_config = BrowserConfig(
                headless=True,
                browser_type="chromium",
                extra_args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            print_status("CI-specific BrowserConfig created successfully", "SUCCESS")

        return True

    except ImportError as e:
        print_status(f"Crawl4AI not available: {e}", "ERROR")
        return False
    except Exception as e:
        print_status(f"Crawl4AI configuration error: {e}", "ERROR")
        return False


async def test_basic_crawling() -> bool:
    """Test basic crawling functionality."""
    print_status("Testing basic crawling functionality...")

    try:
        from crawl4ai import AsyncWebCrawler
        from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

        browser_config = BrowserConfig(
            headless=True,
            browser_type="chromium",
        )

        # Add platform and CI-specific args
        extra_args = []
        if is_ci_environment():
            extra_args.extend(["--no-sandbox", "--disable-dev-shm-usage"])
        if is_windows():
            extra_args.extend(["--disable-gpu", "--disable-dev-shm-usage"])
        
        if extra_args:
            browser_config.extra_args = extra_args

        run_config = CrawlerRunConfig(
            word_count_threshold=10,
            page_timeout=30000,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url="https://httpbin.org/html", config=run_config
            )

            if result.success:
                print_status("Basic crawling test passed", "SUCCESS")
                print(f"  Extracted {len(result.markdown)} characters of content")
                return True
            else:
                print_status(f"Crawling failed: {result.error}", "ERROR")
                return False

    except Exception as e:
        print_status(f"Crawling test error: {e}", "ERROR")
        return False


def install_browsers_if_needed() -> bool:
    """Install Playwright browsers if needed."""
    print_status("Checking browser installation...")

    try:
        # Check if browsers are already installed by trying to find browser binary
        try:
            check_cmd = [sys.executable, "-c", 
                "from playwright.async_api import async_playwright; import asyncio; "
                "async def check(): "
                "  p = await async_playwright().start(); "
                "  b = await p.chromium.launch(); "
                "  await b.close(); "
                "  await p.stop(); "
                "asyncio.run(check())"]
            
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print_status("Browsers already installed and working", "SUCCESS")
                return True
            else:
                print_status("Browsers need to be installed...", "INFO")
        except:
            pass

        # Determine installation strategy based on platform and environment
        if is_ci_environment():
            # CI environment: Platform-specific installation
            print_status("CI environment detected, attempting browser install...")
            if is_linux():
                cmd = [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print_status("System deps install failed, trying browser only...", "WARNING")
                    cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            else:
                # Windows/macOS: Browser-only installation in CI
                cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        else:
            # Local environment: Platform-specific strategy
            if is_windows():
                print_status("Windows detected, installing browsers only...")
                cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
            elif is_macos():
                print_status("macOS detected, installing browsers only...")
                cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
            else:
                print_status("Linux detected, trying browser-only first...")
                cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # For Linux local dev, try with system deps if browser-only fails
            if result.returncode != 0 and is_linux():
                print_status("Browser-only install failed, trying with system deps...", "WARNING")
                cmd = [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print_status("Browser installation completed", "SUCCESS")
            return True
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            print_status(f"Browser installation failed: {error_msg}", "ERROR")
            return False

    except subprocess.TimeoutExpired:
        print_status("Browser installation timed out", "ERROR")
        return False
    except Exception as e:
        print_status(f"Browser installation error: {e}", "ERROR")
        return False


async def main():
    """Main test function."""
    print_status("Starting browser automation setup test...")
    print()

    # Set platform-specific environment defaults
    env_defaults = set_platform_environment_defaults()
    for key, value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            print_status(f"Set environment variable: {key}={value}")

    # Check environment
    env_info = check_environment()
    print()

    # Install browsers if needed
    browsers_ok = install_browsers_if_needed()
    print()

    # Test Playwright
    playwright_ok = test_playwright_installation()
    print()

    # Test Crawl4AI config
    crawl4ai_ok = test_crawl4ai_browser_config()
    print()

    # Test actual crawling
    crawling_ok = await test_basic_crawling()
    print()

    # Final results
    all_tests_passed = all([browsers_ok, playwright_ok, crawl4ai_ok, crawling_ok])

    if all_tests_passed:
        print_status("All browser automation tests passed!", "SUCCESS")
        print_status("Browser setup is ready for testing", "SUCCESS")
        return 0
    else:
        print_status("Some browser automation tests failed", "ERROR")
        failed_tests = []
        if not browsers_ok:
            failed_tests.append("browser installation")
        if not playwright_ok:
            failed_tests.append("playwright setup")
        if not crawl4ai_ok:
            failed_tests.append("crawl4ai config")
        if not crawling_ok:
            failed_tests.append("basic crawling")

        print_status(f"Failed tests: {', '.join(failed_tests)}", "ERROR")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
